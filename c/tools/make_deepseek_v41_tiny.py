#!/usr/bin/env python3
"""Generate the DeepSeek V4.1-Flash tiny fixture, and the tokens it has to produce.

One source of truth: the vendor's own architecture. The model is built by the released
`inference/model.py` (imported unmodified, with the six leaf ops supplied by
`tools/deepseek_v41_reference.py`), filled with a fixed seed, quantized into the *checkpoint's*
encodings, written back into the model, and only then is the reference generation run. So the
bytes in the fixture are the bytes the reference read, and the tokens in `ref.json` are what
those bytes produce -- not a re-implementation's opinion of them.

That is why this generator needs no `transformers` (the V4 tiny generator does): the
architecture, the router, the hyper-connections, the engram and the quantized weight objects
are all in the vendor's tree, and `ModelArgs` already carries the tiny default config. The
vendor sources are taken **by path** (`--inference DIR`), never vendored here.

Scope of this first fixture -- deliberately narrow, so that what it exercises is what is
being ported:

    * vision off (no `vision_config`), no DSpark (`num_nextn_predict_layers: 0`)
    * `compress_ratios` all 0 and no source layers: no compressor, no indexer
    * engram ON, one layer, with a table whose primes are drawn by their own rule
    * the dense path in the checkpoint's real geometry: **fp8 e4m3 with 32x32 UE8M0 scales**
    * experts as the checkpoint stores them: packed fp4, one exponent per row per 32 columns

Everything the fixture contains is validated against `tools/deepseek_v41_layout.py` -- the same
map the expert pin and the engine's loader are read against -- in both directions: every
parameter of the vendor model must have a place in the map, and every entry of the map (for this
geometry) must be produced. A family that drifts in the map fails here rather than in the oracle.

Usage:

    python c/tools/make_deepseek_v41_tiny.py --inference /path/to/vendor/inference --force
    python c/tools/make_deepseek_v41_tiny.py --manifest      # names/shapes only, no torch

The output directory (`c/deepseek_v41_tiny/` by default) is what `tests/test_deepseek_v41_tiny.py`
loads; it is the family's oracle, so it ships in the repo.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
from collections import OrderedDict
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import deepseek_v41_layout as layout  # noqa: E402 - after the path insert, on purpose

SEED = 20260912
SCHEMA_VERSION = 1
GENERATOR_VERSION = 1

# ---- the tiny geometry -------------------------------------------------------
# Every number is small enough to ship and still division-clean: hidden/heads/ranks divide by
# 32 (the fp8 block), moe and hidden halve into the fp4 payload, hc_mult keeps the
# hyper-connection split non-degenerate, and the engram table is a few thousand rows.
#
# NOTE: `dim` is what the vendor calls `hidden_size`.
VOCAB = 64
HIDDEN = 256
LAYERS = 3
HEADS = 4
HEAD_DIM = 64
ROPE_HEAD_DIM = 32
Q_RANK = 64
O_RANK = 32
O_GROUPS = 2
MOE = 128
EXPERTS = 4
TOP_K = 2
HC = 2
WINDOW = 32
ENGRAM_LAYERS = (1,)
ENGRAM_NGRAM = 3
ENGRAM_HEADS = 2
ENGRAM_HEAD_DIM = 64
ENGRAM_VOCAB = 512          # the primes are drawn just above this, by the vendor's own rule
MAX_SEQ = 64
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - regeneration-only diagnostic
        raise SystemExit(
            "generating the tiny fixture needs torch (the oracle generators of the other "
            f"families need it too): {exc}"
        )
    return torch


# ----------------------------------------------------------------- quantization
# Each helper returns (payload, scale) for one weight family, and the *dequantized* values so
# the caller can put exactly what the fixture stores back into the model.

def ceil_scale_exponent(maximum: float, limit: float) -> int:
    if maximum <= 0.0:
        return -127
    return max(-127, min(127, math.ceil(math.log2(maximum / limit))))


def quantize_fp8_blocks(torch, value, block_rows: int, block_columns: int, limit: float = 448.0):
    """e4m3 payload + UE8M0 exponents over a block_rows x block_columns tile."""
    value = value.detach().to(torch.float32).contiguous()
    rows, columns = value.shape
    if rows % block_rows or columns % block_columns:
        raise ValueError(f"fp8 block {block_rows}x{block_columns} does not tile {value.shape}")
    codes = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows // block_rows, columns // block_columns),
                         dtype=torch.float8_e8m0fnu)
    dequantized = torch.empty_like(value)
    for row in range(0, rows, block_rows):
        for column in range(0, columns, block_columns):
            block = value[row:row + block_rows, column:column + block_columns]
            exponent = ceil_scale_exponent(float(block.abs().max()), limit)
            scale = math.ldexp(1.0, exponent)
            tile = (block / scale).clamp(-limit, limit).to(torch.float8_e4m3fn)
            codes[row:row + block_rows, column:column + block_columns] = tile
            scales[row // block_rows, column // block_columns] = scale
            dequantized[row:row + block_rows, column:column + block_columns] = tile.float() * scale
    return codes, scales, dequantized


def quantize_fp4_rows(torch, value):
    """Native packed e2m1 (2 per byte along K, low nibble on the even column) + UE8M0 per row
    per 32 columns -- the checkpoint's expert layout."""
    value = value.detach().to(torch.float32).contiguous()
    rows, columns = value.shape
    if columns % 32 or columns % 2:
        raise ValueError(f"fp4 wants 32-wide blocks on an even width: {value.shape}")
    grid = torch.tensor(E2M1, dtype=torch.float32)
    blocks = value.reshape(rows, columns // 32, 32)
    exponents = [ceil_scale_exponent(float(block.abs().max()), 6.0) for block in blocks.reshape(-1, 32)]
    scales = torch.tensor(exponents, dtype=torch.float32).reshape(rows, columns // 32)
    scaled = (value.reshape(rows, columns // 32, 32)
              / scales.unsqueeze(-1)).clamp(-6.0, 6.0)
    # nearest value on the e2m1 grid, sign carried by the top bit of the 4-bit code
    flat = scaled.reshape(-1, 1)
    distance = (flat.abs() - grid[:8].reshape(1, -1)).abs()
    codes = distance.argmin(dim=-1).to(torch.uint8) | torch.where(flat[:, 0] < 0, 8, 0).to(torch.uint8)
    codes = codes.reshape(rows, columns)
    dequantized = (grid[codes.long()].reshape(rows, columns // 32, 32)
                   * scales.unsqueeze(-1)).reshape(rows, columns)
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    packed = (low | (high << 4)).contiguous().view(torch.uint8)
    return packed, scales.to(torch.float8_e8m0fnu), dequantized


# --------------------------------------------------------------------- writing

def safetensors_dtype(torch, tensor) -> str:
    mapping = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3",
               torch.float8_e8m0fnu: "F8_E8M0", torch.int64: "I64", torch.int8: "I8",
               torch.uint8: "U8"}
    try:
        return mapping[tensor.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype: {tensor.dtype}") from exc


def tensor_bytes(torch, tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def write_safetensors(torch, path: Path, tensors: OrderedDict) -> None:
    """A small standard safetensors file, in insertion order.

    Order matters downstream: the engine's expert store wants the three packed weights of an
    expert to form one contiguous range, and its three scale tensors a second one.
    """
    header: OrderedDict[str, object] = OrderedDict()
    payloads: list[bytes] = []
    offset = 0
    header["__metadata__"] = {"format": "pt",
                              "generator": "c/tools/make_deepseek_v41_tiny.py"}
    for name, tensor in tensors.items():
        payload = tensor_bytes(torch, tensor)
        header[name] = {"dtype": safetensors_dtype(torch, tensor), "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(payload)]}
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for payload in payloads:
            stream.write(payload)


def write_tokenizer() -> dict:
    """The synthetic tokenizer the engram token map is rebuilt from.

    The real checkpoint's `tokenizer.json` is 6 MB and the map is derived from it, so a tiny
    fixture ships its own: one special token per id, no merges. The engine rebuilds the
    compressed map from this file with the same normalizer chain, which is the point.
    """
    added = [{"id": token, "content": f"<t{token:03d}>", "single_word": False, "lstrip": False,
              "rstrip": False, "normalized": False, "special": True} for token in range(VOCAB)]
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": None, "pre_tokenizer": None, "post_processor": None, "decoder": None,
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": False,
                  "byte_fallback": False, "ignore_merges": True,
                  "vocab": {"x": VOCAB - 1}, "merges": []},
    }


# ------------------------------------------------------------- the vendor's model

class _BackendTokenizer:
    """The `backend_tokenizer` face of a HuggingFace tokenizer, over a raw `tokenizers` one.

    The vendor's `build_compressed_token_map` reads `len(tokenizer)`,
    `tokenizer.backend_tokenizer.decode(...)` and `.id_to_token(...)`. `transformers` supplies
    those; a raw `tokenizers.Tokenizer` has all but the middle one, and building the fixture
    should not need `transformers`.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def id_to_token(self, index: int):
        return self._tokenizer.id_to_token(index)


class FixtureTokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.backend_tokenizer = _BackendTokenizer(tokenizer)

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)


def engram_geometry(engram_module) -> tuple[tuple, int]:
    """The tiny engram's primes and table rows, drawn by the vendor's own rule.

    `EngramLayout.from_args` walks up from `engram_vocab_size - 1` handing out the primes below
    it -- `n_heads` per n-gram order per layer, never reusing one -- and the table is exactly as
    many rows as those primes sum to.
    """
    seen: set[int] = set()
    primes = []
    for _ in ENGRAM_LAYERS:
        per_ngram = []
        for _ in range(ENGRAM_NGRAM - 1):
            sizes = []
            for _ in range(ENGRAM_HEADS):
                prime = engram_module.find_next_prime(ENGRAM_VOCAB - 1, seen)
                seen.add(prime)
                sizes.append(prime)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    rows = sum(sum(order) for per_ngram in primes for order in per_ngram)
    return tuple(primes), rows


def build_model(inference: Path, tokenizer, primes, rows):
    """The vendor's own Transformer at the tiny geometry, with the narrow scope above."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "deepseek_v41_reference", TOOLS / "deepseek_v41_reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    model = reference.load_inference(inference)
    from engram import build_compressed_token_map

    _, compressed = build_compressed_token_map(tokenizer)
    torch = require_torch()
    args = model.ModelArgs(
        max_seq_len=MAX_SEQ, temperature=0,          # greedy: sample() then returns an argmax
        vocab_size=VOCAB, dim=HIDDEN, moe_inter_dim=MOE, n_layers=LAYERS, n_mtp_layers=0,
        n_heads=HEADS, n_routed_experts=EXPERTS, n_shared_experts=1, n_activated_experts=TOP_K,
        q_lora_rank=Q_RANK, head_dim=HEAD_DIM, rope_head_dim=ROPE_HEAD_DIM, o_groups=O_GROUPS,
        o_lora_rank=O_RANK, window_size=WINDOW, compress_ratios=(0,) * LAYERS,
        kv_source_layers=(), index_source_layers=(), index_n_heads=2, index_head_dim=32,
        index_topk=8, candidate_source_layer=-1, candidate_topk_blocks=0, candidate_block_size=0,
        hc_mult=HC, hc_sinkhorn_iters=3, engram_layer_ids=ENGRAM_LAYERS,
        engram_num_embeddings=(rows,), engram_max_ngram_size=ENGRAM_NGRAM,
        engram_vocab_size=ENGRAM_VOCAB, engram_n_heads=ENGRAM_HEADS,
        engram_head_dim=ENGRAM_HEAD_DIM, engram_compressed_vocab_size=compressed,
        vision_n_layers=0, dspark_block_size=0, dspark_n_routed_experts=0,
        dspark_n_activated_experts=0, dspark_target_layer_ids=(),
    )
    torch.set_default_dtype(torch.bfloat16)          # the driver's duty, not the model's
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    return torch, model.Transformer(args, tokenizer), args


def geometry_from_args() -> layout.Geometry:
    """The layout map's view of the tiny geometry (the fixture's own `config.json`)."""
    return layout.Geometry.from_config(make_checkpoint_config())


def make_checkpoint_config() -> dict:
    """The tiny config in the checkpoint's own shape (HF `text_config`, no `vision_config`)."""
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                                "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
                                "expert_dtype": "fp4"},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "moe_intermediate_size": MOE,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": 1,
            "head_dim": HEAD_DIM,
            "qk_rope_head_dim": ROPE_HEAD_DIM,
            "q_lora_rank": Q_RANK,
            "o_lora_rank": O_RANK,
            "o_groups": O_GROUPS,
            "hidden_act": "silu",
            "swiglu_limit": 0.0,
            "rms_norm_eps": 1e-20,
            "attention_bias": False,
            "use_cache": True,
            "tie_word_embeddings": False,
            "max_position_embeddings": MAX_SEQ * 16,
            "rope_theta": 10000.0,
            "rope_scaling": {"rope_type": "yarn", "factor": 40.0, "beta_fast": 32.0,
                             "beta_slow": 1.0, "original_max_position_embeddings": 256},
            "n_routed_experts": EXPERTS,
            "n_shared_experts": 1,
            "num_experts_per_tok": TOP_K,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
            "sliding_window": WINDOW,
            "compress_ratios": [0] * LAYERS,
            "compress_rope_theta": 40000.0,
            "kv_source_layer_ids": [],
            "index_source_layer_ids": [],
            "index_n_heads": 2,
            "index_head_dim": 32,
            "index_topk": 8,
            "candidate_source_layer_id": -1,
            "candidate_topk_blocks": 0,
            "candidate_block_size": 0,
            "hc_mult": HC,
            "hc_sinkhorn_iters": 3,
            "hc_eps": 1e-06,
            "engram_layer_ids": list(ENGRAM_LAYERS),
            "engram_num_embeddings": [],          # filled in by main(), from the primes
            "engram_max_ngram_size": ENGRAM_NGRAM,
            "engram_vocab_size": ENGRAM_VOCAB,
            "engram_n_heads": ENGRAM_HEADS,
            "engram_head_dim": ENGRAM_HEAD_DIM,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 0,    # filled in by main()
            "num_nextn_predict_layers": 0,
            "dspark_block_size": 0,
            "dspark_markov_rank": 0,
            "dspark_n_routed_experts": 0,
            "dspark_num_experts_per_tok": 0,
            "dspark_target_layer_ids": [],
        },
    }


def quantize_parameters(torch, net, expected: dict) -> OrderedDict:
    """Fill every parameter with a seeded init, quantize it into the fixture, and put the
    quantized values back into the model, so the reference reads the fixture's own bytes."""
    torch.manual_seed(SEED)
    tensors: OrderedDict[str, object] = OrderedDict()
    parameters = dict(net.named_parameters())
    produced = set()
    for name in sorted(parameters):
        if name.endswith(".scale"):
            continue
        entry = expected.get(name)
        if entry is None:
            raise SystemExit(f"the vendor model has a parameter the layout map does not: {name}")
        parameter = parameters[name]
        if entry.quant.kind == layout.FP4:
            logical = torch.empty(entry.shape[0], entry.shape[1] * 2,
                                  dtype=torch.float32).normal_(0.0, 0.02)
            packed, scales, dequantized = quantize_fp4_rows(torch, logical)
            tensors[name] = packed.view(torch.int8)
            tensors[entry.scale_name] = scales
            parameter.data = packed.view(torch.float4_e2m1fn_x2).reshape(parameter.shape)
            parameters[entry.scale_name].data = scales
        elif entry.quant.kind == layout.FP8:
            logical = torch.empty(*entry.shape, dtype=torch.float32).normal_(0.0, 0.02)
            rows = 1 if entry.quant.scale_block_rows == 1 else entry.quant.scale_block_rows
            codes, scales, dequantized = quantize_fp8_blocks(
                torch, logical, rows, entry.quant.scale_block_columns)
            tensors[name] = codes
            tensors[entry.scale_name] = scales
            scale_parameter = parameters.get(entry.scale_name)
            if scale_parameter is not None:            # the fixture's bytes, in the model
                parameter.data = codes
                scale_parameter.data = scales
            else:                                      # wo_a: fp8 in the checkpoint, bf16 at runtime
                parameter.data = dequantized.to(parameter.dtype).reshape(parameter.shape)
        else:
            dtype = torch.bfloat16 if entry.quant.dtype == "BF16" else torch.float32
            values = torch.empty(*entry.shape, dtype=torch.float32).normal_(0.0, 0.02)
            values = values.to(dtype).float()           # the round trip the engine will read
            tensors[name] = values.to(dtype)
            # the model keeps its own dtype (its head computes in f32), but its numbers are the
            # fixture's: whatever the engine reads is what the reference ran on
            parameter.data = values.to(dtype).to(parameter.dtype).reshape(parameter.shape)
        produced.add(name)
    return tensors


def validate_against_layout(torch, tensors: OrderedDict, geometry: layout.Geometry) -> list[str]:
    """The fixture and the layout map must agree in both directions."""
    problems: list[str] = []
    mapped = {}
    for entry in layout.build_layout(geometry):
        mapped[entry.name] = entry
        if entry.scale_name:
            mapped[entry.scale_name] = entry
    for name, tensor in tensors.items():
        entry = mapped.get(name)
        if entry is None:
            problems.append(f"the fixture holds a tensor the layout map does not: {name}")
            continue
        want = entry.shape if name == entry.name else entry.scale_shape
        dtype = entry.quant.dtype if name == entry.name else entry.scale_dtype
        if list(tensor.shape) != want:
            problems.append(f"{name}: fixture {list(tensor.shape)} vs map {want}")
        if safetensors_dtype(torch, tensor) != dtype:
            problems.append(f"{name}: fixture {safetensors_dtype(torch, tensor)} vs map {dtype}")
    for name in mapped:
        if name not in tensors:
            problems.append(f"the layout map has a tensor the fixture does not: {name}")
    return problems


# ------------------------------------------------------------- the reference run

def walk(torch, net, ids: list[int]) -> list[int]:
    """The argmax the model emits after every prefix, one token at a time.

    `Transformer.forward` returns the head's output for the *last* position only (that is what
    generation needs), so the per-position expectations come from feeding the sequence the way
    the engine runs it: one token per call, `start_pos` advancing. `outputs[i]` is what the model
    picks after reading `ids[:i + 1]`.
    """
    outputs = []
    for position, token in enumerate(ids):
        output_ids, _, _ = net(torch.tensor([[token]], dtype=torch.long), start_pos=position)
        outputs.append(int(output_ids.reshape(-1)[0]))
    return outputs


def reference_cases(torch, net, prompts: dict, max_new: dict) -> dict:
    cases = {}
    for label, prompt in prompts.items():
        prompt_outputs = walk(torch, net, prompt)
        sequence = list(prompt) + [prompt_outputs[-1]]
        for position in range(len(prompt), len(prompt) + max_new[label] - 1):
            output_ids, _, _ = net(torch.tensor([[sequence[position]]], dtype=torch.long),
                                   start_pos=position)
            sequence.append(int(output_ids.reshape(-1)[0]))
        generated = sequence[len(prompt):]
        teacher = walk(torch, net, sequence)
        if teacher[:len(prompt)] != prompt_outputs:
            raise SystemExit(f"the model is not reproducible across walks ({label}): the same "
                             "prefix gives different argmaxes, so the kv cache is carrying state")
        if len(generated) != max_new[label] or 1 in generated:
            raise SystemExit(f"the reference generation truncated or hit EOS: "
                             f"{label} prompt={prompt} generated={generated}")
        cases[label] = {"prompt_ids": prompt, "greedy_full_ids": sequence,
                        "greedy_new_ids": generated, "teacher_forcing_ids": teacher,
                        "max_new_tokens": max_new[label]}
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default = Path(__file__).resolve().parents[1] / "deepseek_v41_tiny"
    parser.add_argument("--inference", type=Path, required=True,
                        help="the released inference/ directory (not vendored here)")
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch = require_torch()
    sys.path.insert(0, str(args.inference.resolve()))
    from tokenizers import Tokenizer
    import engram as engram_module

    tokenizer = FixtureTokenizer(Tokenizer.from_str(json.dumps(write_tokenizer())))
    primes, rows = engram_geometry(engram_module)
    torch, net, model_args = build_model(args.inference.resolve(), tokenizer, primes, rows)
    net.eval()

    config = make_checkpoint_config()
    config["text_config"]["engram_num_embeddings"] = [rows]
    geometry = layout.Geometry.from_config(config)
    config["text_config"]["engram_compressed_vocab_size"] = int(model_args.engram_compressed_vocab_size)

    expected = {entry.name: entry for entry in layout.build_layout(geometry)}
    tensors = quantize_parameters(torch, net, expected)
    problems = validate_against_layout(torch, tensors, geometry)

    prompts = {"short": [5, 7, 9, 11, 13, 17, 19, 23],
               "window": [3 + (index * 7) % (VOCAB - 4) for index in range(WINDOW + 8)],
               "engram": [11, 12, 11, 12, 13, 14, 11, 12, 13, 15, 16, 17]}
    max_new = {"short": 8, "window": 4, "engram": 6}
    cases = reference_cases(torch, net, prompts, max_new)

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists (use --force): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "tokenizer.json").write_text(
        json.dumps(write_tokenizer(), separators=(",", ":")) + "\n", encoding="utf-8")
    write_safetensors(torch, output / "model.safetensors", tensors)
    reference = {
        "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION, "seed": SEED,
        "source": "the vendor's own inference/model.py, run through the shim of "
                  "tools/deepseek_v41_reference.py",
        "torch_version": torch.__version__,
        "quantization_format": {
            "dense": "e4m3 payload + UE8M0 32x32 exponents (the checkpoint's own geometry)",
            "engram_table": "e4m3 payload + UE8M0 per row per 32 columns",
            "routed_experts": "packed e2m1 + UE8M0 per row per 32 columns",
            "other": "f32 / bf16 as the checkpoint stores it, round-tripped before the reference",
        },
        "geometry": {
            "vocab_size": VOCAB, "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS, "head_dim": HEAD_DIM, "q_lora_rank": Q_RANK,
            "o_lora_rank": O_RANK, "o_groups": O_GROUPS, "moe_intermediate_size": MOE,
            "n_routed_experts": EXPERTS, "num_experts_per_tok": TOP_K, "hc_mult": HC,
            "sliding_window": WINDOW, "compress_ratios": [0] * LAYERS,
            "engram_layer_ids": list(ENGRAM_LAYERS), "engram_table_rows": rows,
            "engram_primes": [list(order) for per_ngram in primes for order in per_ngram],
            "engram_max_ngram_size": ENGRAM_NGRAM, "engram_compressed_vocab_size":
                int(model_args.engram_compressed_vocab_size),
            "num_nextn_predict_layers": 0, "vision": "off",
        },
        "tensor_count": len(tensors),
        "cases": cases,
    }
    (output / "ref.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

    print(f"[fixture] {len(tensors)} tensors, engram table {rows} rows, "
          f"compressed vocab {int(model_args.engram_compressed_vocab_size)}")
    for name, tensor in tensors.items():
        print(f"  {name:52s} {str(list(tensor.shape)):16s} {safetensors_dtype(torch, tensor)}")
    for label, case in cases.items():
        print(f"[reference] {label:8s} new={case['greedy_new_ids']}")
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"wrote {output} ({total} bytes)")
    if problems:
        print("\nthe fixture and the layout map disagree:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("the fixture matches the layout map in both directions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
