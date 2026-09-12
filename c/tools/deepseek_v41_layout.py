#!/usr/bin/env python3
"""The DeepSeek V4.1-Flash tensor layout: every name, its shape and its quantization.

This is the map the other tools are written from -- the fixture generator writes
its tensors in these names with these encodings, and the expert pin holds the C
dequantizer to the same encoding. It is *read off* two places, never invented:

  * the released `inference/convert.py` (`--inference DIR`, never vendored), which
    states the renaming rules and, as assertions, the shape rules of the two
    quantized families:
        assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size
        assert (out_block_size, in_block_size) in ((32, 32), (128, 128))
    and whose `cast_e2m1fn_to_e4m3fn` decodes the expert nibbles (`low = x & 0x0F`,
    `high = (x >> 4) & 0x0F`, stacked along the last axis, so the *even* column is
    the low nibble) and folds the fp4 range into the scale;
  * the engine's own loader (`c/deepseek_v41.c`: the per-layer plan), which is what
    the engine will actually ask the checkpoint for.

Three encodings exist in this checkpoint and only three:

  * `fp8`      e4m3 payload + a **32x32** UE8M0 scale tile  (`scale.shape ==
               [rows // 32, columns // 32]`).  Every dense weight in the released
               checkpoint is of this form -- including `wo_a`, which in V4 shipped
               either 32x32 or 128x128 and therefore had to be dequantized at load.
  * `fp4`      2 values per byte packed along K, **one UE8M0 exponent per row per
               32 columns** (`scale.shape == [rows, columns // 32]`), the value
               table being `[0, .5, 1, 1.5, 2, 3, 4, 6]` with a sign bit. This is
               the *routed experts* only (and the V4 engine's expert store already
               reads exactly this: `block_rows == 1`, `block_columns == 32`).
               The fp4 *range* is not applied at load in the vendor's own stack --
               `convert.py` folds it into the fp8 scale -- so a loader that keeps
               the payload as fp4 must apply it itself. See the delta doc.
  * `raw`      bf16/f32/i64 as stored, quantized at runtime if at all (the
               compressor's and the indexer's weights are bf16 in the checkpoint
               and quantize at runtime, in two different regimes).

Usage:

    # hold the map to the released checkpoint's own headers (no weights)
    python tools/deepseek_v41_layout.py --check --config config.json \
        --headers-json headers.json

    # additionally diff it against what the engine's loader declares
    python tools/deepseek_v41_layout.py --check --config config.json \
        --headers-json headers.json --engine deepseek_v41.c

    # dump the map (the fixture generator's input)
    python tools/deepseek_v41_layout.py --config config.json --emit layout.json

Fail-closed: a tensor nobody classified, or one whose dtype/shape/scale differs
from the rule that claims it, is a failure -- an unclassified tensor is how a
wrong-model run starts.
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------ encodings

FP4_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

FP8 = "fp8-e4m3"
FP4 = "fp4-e2m1"
RAW = "raw"
F32 = "f32"


@dataclass(frozen=True)
class Quant:
    """How one tensor family is stored, and how its scale tensor is shaped."""

    kind: str
    dtype: str
    note: str
    scale_dtype: str | None = None
    scale_block_rows: int | None = None
    scale_block_columns: int | None = None
    runtime: str | None = None      # quantization deferred to runtime, if any

    def scale_shape(self, rows: int, columns: int) -> list[int] | None:
        if self.scale_dtype is None:
            return None
        return [rows // self.scale_block_rows, columns // self.scale_block_columns]


FP8_32X32 = Quant(FP8, "F8_E4M3",
                  "e4m3 payload, one UE8M0 exponent per 32x32 block",
                  "F8_E8M0", 32, 32)
# the engram table is the one fp8 tensor with a per-row scale: one exponent per
# 32 columns of a row, no row blocking (`[rows, 256]` weight, `[rows, 8]` scale)
FP8_ROW32 = Quant(FP8, "F8_E4M3",
                  "e4m3 payload, one UE8M0 exponent per row per 32 columns",
                  "F8_E8M0", 1, 32)
# experts: per-row scale over 32-wide blocks along K, fp4 packed 2 per byte
FP4_ROW32 = Quant(FP4, "I8",
                  "fp4 packed 2/byte along K (even column = low nibble), one UE8M0 "
                  "exponent per row per 32 columns; fp4 range folded into the scale "
                  "by the vendor's converter",
                  "F8_E8M0", 1, 32)
BF16_RAW = Quant(RAW, "BF16", "bf16 as stored (not quantized)")
F32_RAW = Quant(RAW, "F32", "float32 as stored (not quantized)")
I64_RAW = Quant(RAW, "I64", "int64 as stored")
# weights that the checkpoint stores in bf16 and the engine quantizes at runtime
BF16_RUNTIME_FP4_E4M3 = Quant(RAW, "BF16",
                              "bf16 as stored; quantized at runtime as fp4 in groups "
                              "of 16 with e4m3 scales",
                              runtime="fp4 g16 / e4m3 scales")
BF16_RUNTIME_FP4_E8M0 = Quant(RAW, "BF16",
                              "bf16 as stored; quantized at runtime as fp4 in groups "
                              "of 32 with UE8M0 scales",
                              runtime="fp4 g32 / ue8m0 scales")


@dataclass
class Entry:
    name: str
    quant: Quant
    shape: list[int]
    scale_name: str | None = None
    scale_shape: list[int] | None = None
    scale_dtype: str | None = None

    @property
    def bytes(self) -> int:
        width = 2 if self.quant.dtype == "BF16" else 1
        if self.quant.kind == FP4:
            width = 1        # two nibbles per byte
        elif self.quant.dtype in ("F32", "I32"):
            width = 4
        elif self.quant.dtype == "I64":
            width = 8
        total = width * math.prod(self.shape)
        if self.scale_shape:
            total += math.prod(self.scale_shape)
        return total

    def to_json(self) -> dict:
        out = {"dtype": self.quant.dtype, "shape": self.shape,
               "quant": self.quant.kind, "note": self.quant.note}
        if self.scale_name:
            out["scale"] = {"name": self.scale_name, "dtype": self.scale_dtype,
                            "shape": self.scale_shape}
        if self.quant.runtime:
            out["runtime_quant"] = self.quant.runtime
        return out


# --------------------------------------------------------------------- config

def text_config(raw: dict) -> dict:
    """The HuggingFace config of this family nests the text tower under `text_config`."""
    return dict(raw.get("text_config") or raw)


@dataclass
class Geometry:
    hidden: int
    layers: int
    heads: int
    head_dim: int
    q_rank: int
    o_rank: int
    o_groups: int
    moe: int
    experts: int
    vocab: int
    hc: int
    index_heads: int
    index_dim: int
    engram_layers: list[int]
    engram_rows: list[int]
    engram_head_dim: int
    engram_heads: int
    engram_max_ngram: int
    kv_sources: list[int]
    index_sources: list[int]
    ratios: list[int]
    nextn: int
    dspark_experts: int
    markov_rank: int
    vision: dict | None

    @classmethod
    def from_config(cls, raw: dict) -> "Geometry":
        text = text_config(raw)
        vision = raw.get("vision_config")
        return cls(
            hidden=text["hidden_size"],
            layers=text["num_hidden_layers"],
            heads=text["num_attention_heads"],
            head_dim=text["head_dim"],
            q_rank=text["q_lora_rank"],
            o_rank=text["o_lora_rank"],
            o_groups=text["o_groups"],
            moe=text["moe_intermediate_size"],
            experts=text["n_routed_experts"],
            vocab=text["vocab_size"],
            hc=text["hc_mult"],
            index_heads=text["index_n_heads"],
            index_dim=text["index_head_dim"],
            engram_layers=list(text["engram_layer_ids"]),
            engram_rows=list(text["engram_num_embeddings"]),
            engram_head_dim=text["engram_head_dim"],
            engram_heads=text["engram_n_heads"],
            engram_max_ngram=text["engram_max_ngram_size"],
            kv_sources=list(text["kv_source_layer_ids"]),
            index_sources=list(text["index_source_layer_ids"]),
            ratios=list(text["compress_ratios"]),
            nextn=text["num_nextn_predict_layers"],
            dspark_experts=text["dspark_n_routed_experts"],
            markov_rank=text["dspark_markov_rank"],
            vision=vision,
        )

    @property
    def o_group_width(self) -> int:
        return (self.heads // self.o_groups) * self.head_dim

    @property
    def o_width(self) -> int:
        return self.o_groups * self.o_rank

    @property
    def hc_rows(self) -> int:
        return (2 + self.hc) * self.hc


# ----------------------------------------------------------------------- rules
#
# One rule per tensor family. `where` selects the layers that carry it: all of the
# backbone, an explicit id set, or a predicate over the compress ratios.

def build_layout(geometry: Geometry, names: str = "checkpoint") -> list[Entry]:
    """Every tensor of the model, in checkpoint names, for this geometry.

    `names` picks the vocabulary: `checkpoint` is the released HuggingFace naming
    (`ffn`, `attn`), which is what `convert.py` passes through essentially
    unchanged -- its `mlp`->`ffn` and `self_attn`->`attn` rules are no-ops on this
    checkpoint and its `*_scale_inv`/`e_score_correction_bias` rules never fire.
    """
    g = geometry
    entries: list[Entry] = []

    def add(name: str, quant: Quant, shape: list[int], scale: bool = False,
            scale_shape: list[int] | None = None) -> None:
        entry = Entry(name, quant, list(shape))
        if scale:
            entry.scale_name = name[:-len(".weight")] + ".scale" if name.endswith(".weight") \
                else name + ".scale"
            entry.scale_shape = scale_shape if scale_shape is not None \
                else quant.scale_shape(shape[0], shape[1] if len(shape) > 1 else 1)
            entry.scale_dtype = quant.scale_dtype
        entries.append(entry)

    # ---- per-layer backbone -------------------------------------------------
    for layer in range(g.layers):
        prefix = f"layers.{layer}."
        add(prefix + "attn_norm.weight", BF16_RAW, [g.hidden])
        add(prefix + "ffn_norm.weight", BF16_RAW, [g.hidden])
        add(prefix + "attn.attn_sink", F32_RAW, [g.heads])
        add(prefix + "attn.q_norm.weight", BF16_RAW, [g.q_rank])
        add(prefix + "attn.kv_norm.weight", BF16_RAW, [g.head_dim])
        add(prefix + "attn.wq_a.weight", FP8_32X32, [g.q_rank, g.hidden], scale=True)
        add(prefix + "attn.wq_b.weight", FP8_32X32, [g.heads * g.head_dim, g.q_rank], scale=True)
        add(prefix + "attn.wkv.weight", FP8_32X32, [g.head_dim, g.hidden], scale=True)
        add(prefix + "attn.wo_a.weight", FP8_32X32, [g.o_width, g.o_group_width], scale=True)
        add(prefix + "attn.wo_b.weight", FP8_32X32, [g.hidden, g.o_width], scale=True)

        # the compressor and the index keys live on the KV source layers only
        if layer in g.kv_sources:
            add(prefix + "attn.compressor.norm.weight", BF16_RUNTIME_FP4_E4M3, [g.head_dim])
            add(prefix + "attn.compressor.wkv.weight", BF16_RUNTIME_FP4_E4M3,
                [g.head_dim, g.hidden])
            if g.ratios[layer] > 1:
                add(prefix + "attn.compressor.wgate.weight", BF16_RAW, [g.head_dim, g.hidden])
            add(prefix + "attn.indexer.wk.weight", BF16_RUNTIME_FP4_E8M0,
                [g.index_dim, g.head_dim])
            add(prefix + "attn.indexer.k_norm.weight", BF16_RAW, [g.index_dim])
        if layer in g.index_sources:
            add(prefix + "attn.indexer.wq_b.weight", FP8_32X32,
                [g.index_heads * g.index_dim, g.q_rank], scale=True)
            add(prefix + "attn.indexer.weights_proj.weight", BF16_RAW,
                [g.index_heads, g.hidden])

        # the engram tables: one row block per engram layer, mmapped, never resident
        if layer in g.engram_layers:
            position = g.engram_layers.index(layer)
            rows = g.engram_rows[position]
            add(prefix + "engram.q_weight", BF16_RAW, [g.hc, g.hidden])
            add(prefix + "engram.k_weight", BF16_RAW, [g.hc, g.hidden])
            # the vendor's own construction: Linear(n_hash_cols * head_dim, dim * (hc_mult + 1))
            # with n_hash_cols = (max_ngram_size - 1) * n_heads. On the released checkpoint both
            # spellings coincide (3*8*256 = 6144 = hidden + o_lora_rank, 5120*5 = 5*hidden), so a
            # tiny geometry is what tells the two apart -- see the delta doc.
            add(prefix + "engram.wkv.weight", FP8_32X32,
                [(g.hc + 1) * g.hidden, (g.engram_max_ngram - 1) * g.engram_heads * g.engram_head_dim],
                scale=True)
            add(prefix + "engram.embed.weight", FP8_ROW32, [rows, g.engram_head_dim], scale=True)

        # MoE
        add(prefix + "ffn.gate.weight", BF16_RAW, [g.experts, g.hidden])
        add(prefix + "ffn.gate.bias", F32_RAW, [g.experts])
        if g.vision is not None:
            # the vision-masked routing bias exists exactly where there is a vision tower to
            # mask; a text-only checkpoint has none, and the engine only applies it when an
            # image span is present (the vendor's model allocates it with the vision tower too)
            add(prefix + "ffn.gate.bias_vl", F32_RAW, [g.experts])
        add(prefix + "ffn.shared_experts.w1.weight", FP8_32X32, [g.moe, g.hidden], scale=True)
        add(prefix + "ffn.shared_experts.w2.weight", FP8_32X32, [g.hidden, g.moe], scale=True)
        add(prefix + "ffn.shared_experts.w3.weight", FP8_32X32, [g.moe, g.hidden], scale=True)
        for expert in range(g.experts):
            e = f"{prefix}ffn.experts.{expert}."
            add(e + "w1.weight", FP4_ROW32, [g.moe, g.hidden // 2], scale=True,
                scale_shape=[g.moe, g.hidden // 32])
            add(e + "w2.weight", FP4_ROW32, [g.hidden, g.moe // 2], scale=True,
                scale_shape=[g.hidden, g.moe // 32])
            add(e + "w3.weight", FP4_ROW32, [g.moe, g.hidden // 2], scale=True,
                scale_shape=[g.moe, g.hidden // 32])

        for stem in ("attn", "ffn"):
            add(prefix + f"hc_{stem}_base", F32_RAW, [g.hc_rows])
            add(prefix + f"hc_{stem}_fn", F32_RAW, [g.hc_rows, g.hc * g.hidden])
            add(prefix + f"hc_{stem}_scale", F32_RAW, [3])

    # ---- DSpark (MTP) stages ------------------------------------------------
    # heterogeneous on purpose: stage 0 carries the target-state conditioning and
    # the last stage carries the head -- see docs/deepseek-v41-delta.md
    for stage in range(g.nextn):
        prefix = f"mtp.{stage}."
        add(prefix + "attn_norm.weight", BF16_RAW, [g.hidden])
        add(prefix + "ffn_norm.weight", BF16_RAW, [g.hidden])
        add(prefix + "attn.attn_sink", F32_RAW, [g.heads])
        add(prefix + "attn.q_norm.weight", BF16_RAW, [g.q_rank])
        add(prefix + "attn.kv_norm.weight", BF16_RAW, [g.head_dim])
        add(prefix + "attn.wq_a.weight", FP8_32X32, [g.q_rank, g.hidden], scale=True)
        add(prefix + "attn.wq_b.weight", FP8_32X32, [g.heads * g.head_dim, g.q_rank], scale=True)
        add(prefix + "attn.wkv.weight", FP8_32X32, [g.head_dim, g.hidden], scale=True)
        add(prefix + "attn.wo_a.weight", FP8_32X32, [g.o_width, g.o_group_width], scale=True)
        add(prefix + "attn.wo_b.weight", FP8_32X32, [g.hidden, g.o_width], scale=True)
        add(prefix + "ffn.gate.weight", BF16_RAW, [g.dspark_experts, g.hidden])
        add(prefix + "ffn.gate.bias", F32_RAW, [g.dspark_experts])
        if g.vision is not None:
            add(prefix + "ffn.gate.bias_vl", F32_RAW, [g.dspark_experts])
        add(prefix + "ffn.shared_experts.w1.weight", FP8_32X32, [g.moe, g.hidden], scale=True)
        add(prefix + "ffn.shared_experts.w2.weight", FP8_32X32, [g.hidden, g.moe], scale=True)
        add(prefix + "ffn.shared_experts.w3.weight", FP8_32X32, [g.moe, g.hidden], scale=True)
        for expert in range(g.dspark_experts):
            e = f"{prefix}ffn.experts.{expert}."
            add(e + "w1.weight", FP4_ROW32, [g.moe, g.hidden // 2], scale=True,
                scale_shape=[g.moe, g.hidden // 32])
            add(e + "w2.weight", FP4_ROW32, [g.hidden, g.moe // 2], scale=True,
                scale_shape=[g.hidden, g.moe // 32])
            add(e + "w3.weight", FP4_ROW32, [g.moe, g.hidden // 2], scale=True,
                scale_shape=[g.moe, g.hidden // 32])
        for stem in ("attn", "ffn"):
            add(prefix + f"hc_{stem}_base", F32_RAW, [g.hc_rows])
            add(prefix + f"hc_{stem}_fn", F32_RAW, [g.hc_rows, g.hc * g.hidden])
            add(prefix + f"hc_{stem}_scale", F32_RAW, [3])
        if stage == 0:
            add(prefix + "main_norm.weight", BF16_RAW, [g.hidden])
            add(prefix + "main_proj.weight", FP8_32X32, [g.hidden, 3 * g.hidden], scale=True)
        if stage == g.nextn - 1:
            add(prefix + "norm.weight", BF16_RAW, [g.hidden])
            add(prefix + "markov_head.embed.weight", BF16_RAW, [g.vocab, g.markov_rank])
            add(prefix + "markov_head.head.weight", BF16_RAW, [g.vocab, g.markov_rank])
            add(prefix + "confidence_head.proj.weight", BF16_RAW, [1, g.hidden + g.markov_rank])

    # ---- globals and the vision tower ---------------------------------------
    add("embed.weight", BF16_RAW, [g.vocab, g.hidden])
    add("head.weight", BF16_RAW, [g.vocab, g.hidden])
    add("norm.weight", BF16_RAW, [g.hidden])
    if g.vision:
        v = g.vision
        vhidden, patch = v["hidden_size"], v["patch_size"]
        for name in ("image_start", "image_end", "image_newline"):
            add(name, BF16_RAW, [g.hidden])
        add("aligner.w1.weight", BF16_RAW, [g.hidden, vhidden * v["downsample_ratio"] ** 2])
        add("aligner.w1.bias", BF16_RAW, [g.hidden])
        add("aligner.w2.weight", BF16_RAW, [g.hidden, g.hidden])
        add("aligner.w2.bias", BF16_RAW, [g.hidden])
        add("vision.patch_embed.proj.weight", BF16_RAW, [vhidden, patch * patch * 3])
        add("vision.patch_embed.proj.bias", BF16_RAW, [vhidden])
        add("vision.norm.weight", BF16_RAW, [vhidden])
        for block in range(v["num_hidden_layers"]):
            b = f"vision.blocks.{block}."
            add(b + "norm1.weight", BF16_RAW, [vhidden])
            add(b + "norm2.weight", BF16_RAW, [vhidden])
            add(b + "attn.wqkv.weight", BF16_RAW, [3 * vhidden, vhidden])
            add(b + "attn.wqkv.bias", BF16_RAW, [3 * vhidden])
            add(b + "attn.wo.weight", BF16_RAW, [vhidden, vhidden])
            add(b + "attn.wo.bias", BF16_RAW, [vhidden])
            add(b + "mlp.w1.weight", BF16_RAW, [2 * v["intermediate_size"], vhidden])
            add(b + "mlp.w2.weight", BF16_RAW, [vhidden, v["intermediate_size"]])
    return entries


# ---------------------------------------------------------- convert.py pass-through
#
# The released converter renames a handful of keys and slices some tensors across
# model-parallel ranks. None of that changes the *stored* layout, but a port that
# claims "native load, no converter" has to be right about which rules apply at
# all -- so the rules are reproduced here and applied to the real headers, and the
# tool reports which ones fire (they are no-ops on this checkpoint).

CONVERT_MAPPING = {"embed": ("embed", 0), "wq_b": ("wq_b", 0), "wo_a": ("wo_a", 0),
                   "wo_b": ("wo_b", 1), "head": ("head", 0),
                   "attn_sink": ("attn_sink", 0), "weights_proj": ("weights_proj", 0)}
NO_WEIGHT_SUFFIX = ("hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_")


def convert_renames(name: str) -> list[str]:
    """Every name `convert.py` would turn `name` into (before the mp slicing)."""
    rules: list[str] = []
    out = name[len("model."):] if name.startswith("model.") else name
    if out != name:
        rules.append("strip model.")
    renamed = out.replace("self_attn", "attn")
    if renamed != out:
        rules.append("self_attn -> attn")
    out = renamed
    if not out.startswith("vision."):
        renamed = out.replace("mlp", "ffn")
        if renamed != out:
            rules.append("mlp -> ffn")
        out = renamed
    for old, new in (("weight_scale_inv", "scale"), ("e_score_correction_bias", "bias")):
        if old in out:
            rules.append(f"{old} -> {new}")
            out = out.replace(old, new)
    key = out.split(".")[-1] if any(x in out for x in NO_WEIGHT_SUFFIX) else out.split(".")[-2]
    if key in CONVERT_MAPPING:
        new_key = CONVERT_MAPPING[key][0]
        rules.append(f"key {key} -> {new_key}" if new_key != key
                     else f"key {key} (identity)")
    return rules


# ----------------------------------------------------------------- verification

def classify(name: str) -> str:
    key = re.sub(r"^layers\.\d+\.", "layers.N.", name)
    key = re.sub(r"^mtp\.\d+\.", "mtp.N.", key)
    key = re.sub(r"\.experts\.\d+\.", ".experts.E.", key)
    key = re.sub(r"^vision\.blocks\.\d+\.", "vision.blocks.N.", key)
    return key


def check_checkpoint(geometry: Geometry, tensors: dict[str, dict], quiet: bool) -> tuple[int, list[str]]:
    expected: dict[str, Entry] = {}         # weight name -> entry
    scales: dict[str, tuple[Entry, str, list[int]]] = {}   # scale name -> (owner, dtype, shape)
    quant_by_family: dict[str, Quant] = {}
    for entry in build_layout(geometry):
        if entry.name in expected:
            raise SystemExit(f"layout tool bug: duplicate entry {entry.name}")
        expected[entry.name] = entry
        quant_by_family[classify(entry.name)] = entry.quant
        if entry.scale_name:
            scales[entry.scale_name] = (entry, entry.scale_dtype, entry.scale_shape)

    failures: list[str] = []
    families: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for name in sorted(tensors):
        spec = tensors[name]
        entry, dtype, shape = None, None, None
        if name in expected:
            entry = expected[name]
            dtype, shape = entry.quant.dtype, entry.shape
        elif name in scales:
            entry, dtype, shape = scales[name]
            dtype, shape = scales[name][1], scales[name][2]
        if entry is None:
            failures.append(f"unclassified tensor: {name}")
            continue
        if spec["dtype"] != dtype:
            failures.append(f"invalid {name}: {spec['dtype']}, expected {dtype}")
        if spec["shape"] != shape:
            failures.append(f"invalid {name}: {spec['shape']}, expected {shape}")
        size = spec["data_offsets"][1] - spec["data_offsets"][0]
        families[classify(entry.name)][0] += 1
        families[classify(entry.name)][1] += size
    for name in list(expected) + list(scales):
        if name not in tensors:
            failures.append(f"missing tensor: {name}")

    renames = collections.Counter()
    for name in tensors:
        for rule in convert_renames(name):
            renames[rule] += 1

    if not quiet:
        print(f"V4.1 layout: {len(expected)} weights + {len(scales)} scales declared, "
              f"{len(tensors)} tensors in the checkpoint")
        print("\nfamily                                              count       GiB   quantization")
        for key, (count, size) in sorted(families.items(), key=lambda item: -item[1][1])[:14]:
            quant = quant_by_family.get(key)
            note = f"{quant.kind:5s} {quant.dtype}" if quant else "?"
            print(f"  {key:50s} {count:6d} {size / 2**30:9.2f}   {note}")
        print("\nconvert.py rules that fire on this checkpoint:")
        if renames:
            for rule, count in renames.most_common():
                print(f"  {count:6d}  {rule}")
        else:
            print("  none -- the checkpoint already uses the converter's own naming")
    return len(failures), failures


LAYER_LOCAL = ("layers.", "mtp.")
GLOBALS = ("embed.weight", "head.weight", "norm.weight", "aligner.", "image_", "vision.")


def engine_patterns(source: Path) -> tuple[set[str], str, set[str]]:
    """The tensor families the engine's source asks the checkpoint for, as patterns.

    Two things have to be read, because the loader has two entry points: the
    per-layer plan's `add_*` calls (layer-local names, and `add_fp8` declares the
    scale itself -- which is where the base engine's block geometry lives), and
    every quoted name or name *builder* elsewhere in the source (the expert store
    builds `layers.%d.ffn.experts.%d.%s.weight` at run time, so the builder is
    what carries the family). `%d`/`%s` fold into a pattern, so one builder covers
    w1/w2/w3.
    """
    text = source.read_text(encoding="utf-8", errors="replace")
    names: set[str] = set()
    for match in re.finditer(r"ADD\(\s*(?:add_[12]d|add_fp8)\((.*?)\)\s*\)\s*;", text, re.S):
        names.update(re.findall(r'"([^"]+)"', match.group(1)))

    literal = re.compile(r'"([A-Za-z_][A-Za-z0-9_.%]*)"')
    for candidate in literal.findall(text):
        if not (candidate.endswith((".weight", ".scale", ".bias"))
                or candidate.endswith("attn_sink")
                or candidate.startswith(("hc_attn", "hc_ffn", "hc_head"))
                or ".experts." in candidate or ".engram." in candidate):
            continue
        names.add(candidate)

    patterns: set[str] = set()
    generic: set[str] = set()
    for name in names:
        # the plan's names are layer-local (`attn.wkv`, `attn_norm.weight`); anything
        # already carrying a scope keeps it, and the genuinely global tensors do too
        if not name.startswith(LAYER_LOCAL) and not name.startswith(GLOBALS):
            name = "layers.0." + name
        name = re.sub(r"%[0-9]*d", "0", name).replace("%s", "*")
        if not name.endswith((".weight", ".scale", ".bias", "attn_sink", "tid2eid",
                              "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                              "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
                              "hc_head_base", "hc_head_fn", "hc_head_scale")):
            # `add_fp8` names the weight by its stem and appends `.weight` itself
            name += ".weight"
        key = classify(name)
        if key.endswith(".scale"):
            continue            # scales travel with their weight
        if re.match(r"^(layers|mtp)\.N\.\*", key):
            generic.add(key)    # `layers.%d.%s.weight`: the plan's own resolver
            continue
        patterns.add(key)

    scale_rule = ""
    geometry = re.search(r"add_fp8\(Coli\w+ \*plan,(.*?)\n\}", text, re.S)
    if geometry:
        found = re.search(r"\(rows \+ (\d+)\) / (\d+)", geometry.group(1))
        if found:
            scale_rule = found.group(0)
    return patterns, scale_rule, generic


def compare_with_engine(patterns: set[str], mapping: dict[str, Entry],
                        scale_rule: str, quiet: bool, generic: set[str] | None = None
                        ) -> tuple[list[str], list[str]]:
    """Diff the loader's expectations against what the checkpoint actually holds.

    A pattern the layer plan declares applies to the backbone *and* to the DSpark
    stages (they run the same layer code with an `mtp.` scope), so each pattern is
    tried under both scopes.
    """
    held = {classify(entry.name) for entry in mapping.values() if not entry.name.endswith(".scale")}
    residue, unasked = [], []
    matched: set[str] = set()
    for pattern in sorted(patterns):
        variants = {pattern, pattern.replace("layers.N.", "mtp.N.")}
        hits = [key for key in held if any(fnmatch.fnmatch(key, v) for v in variants)]
        if not hits:
            residue.append(pattern)
        matched.update(hits)
    for key in sorted(held - matched):
        unasked.append(key)

    if not quiet:
        print(f"\nloader: {len(patterns)} tensor families declared, scale rule `{scale_rule}`"
              f"{', ' + str(len(generic or ())) + ' generic resolvers' if generic else ''}")
        print(f"\nthe loader asks for {len(residue)} families the checkpoint does not hold (V4 residue):")
        for name in residue:
            print(f"  - {name}")
        print(f"\nthe checkpoint holds {len(unasked)} families the loader never asks for:")
        for name in unasked:
            print(f"  - {name}")
        if scale_rule:
            block = scale_rule.rsplit("/", 1)[1].strip(" ;)")
            print(f"\nscale geometry: `{scale_rule}` is a block of {block}, while every fp8 "
                  f"scale in the checkpoint is one exponent per 32x32 block\n"
                  f"  (`layers.0.attn.wq_a`: [1280, 5120] weight, [40, 160] scale)")
    return residue, unasked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true",
                        help="hold the map to the checkpoint's own headers")
    parser.add_argument("--headers-json", type=Path,
                        help="safetensors headers, flat or {shard: {name: spec}}")
    parser.add_argument("--engine", type=Path, help="c/deepseek_v41.c, for the loader cross-check")
    parser.add_argument("--emit", type=Path, help="write the map as JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.check and not args.headers_json:
        parser.error("--check needs --headers-json")

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    geometry = Geometry.from_config(raw)
    mapping = {entry.name: entry for entry in build_layout(geometry)}
    failures: list[str] = []

    if args.emit:
        payload = {name: entry.to_json() for name, entry in mapping.items()}
        args.emit.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"wrote {len(payload)} entries to {args.emit}")

    if args.headers_json:
        loaded = json.loads(args.headers_json.read_text(encoding="utf-8"))
        if all(isinstance(v, dict) and "dtype" not in v for v in loaded.values()):
            loaded = {name: spec for shard in loaded.values() for name, spec in shard.items()}
        count, failures = check_checkpoint(geometry, loaded, args.quiet)
        if count:
            print(f"\nV4.1 layout FAILED: {count} problems")
            for failure in failures[:40]:
                print(f"  - {failure}")
            if count > 40:
                print(f"  ... and {count - 40} more")

    if args.engine:
        patterns, scale_rule, generic = engine_patterns(args.engine)
        compare_with_engine(patterns, mapping, scale_rule, args.quiet, generic)

    if failures:
        return 1
    if args.headers_json or args.engine:
        print("\nV4.1 layout OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
