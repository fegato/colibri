#!/usr/bin/env python3
"""Freeze the verified engram layout into C artifacts, plus golden hash vectors.

Stage two of the engram pipeline. Stage one
(`tools/make_deepseek_v41_engram.py`) reconstructs the layout *from* the checkpoint
and verifies it against numbers the checkpoint declares (class count, table rows);
this tool takes that verified layout and emits:

  * `deepseek_v41_engram_tables.h` -- the primes, the flat bucket offsets and the
    frozen multipliers, i.e. everything the hash needs that is not a weight
  * `tests/deepseek_v41_engram_vectors.h` -- golden hash ids for a few sequences,
    computed here with the same rules as the reference so the C side can be held to
    them without any checkpoint

The hash recipe (reference `NgramHashState.forward`): for position p, take the
`max_ngram_size - 1` previous compressed ids (a dead token or the start of the
sequence blocks that n-gram and every longer one), multiply each by its layer
multiplier, XOR them into a rolling value one lookback at a time, then take the
rolling value modulo one prime per head and add the layer's flat bucket offset.

The token map itself is deliberately NOT emitted here: its storage (a ~274 KB packed
table vs. reproducing the normalizer chain in C) is an open decision, so the hash
takes the compressed ids as input and stays independent of it.

    python tools/make_deepseek_v41_engram_fixture.py \
        --layout layout.json --tables deepseek_v41_engram_tables.h \
        --vectors tests/deepseek_v41_engram_vectors.h
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEAD = -1


def hash_ids(classes: list[int], history: list[int], max_ngram: int, heads: int,
             primes: list[list[int]], multipliers: list[int],
             offsets: list[int], pad: int) -> list[int]:
    """The hash ids for one position (or a span), given the classes before it.

    `history` is the most recent classes, oldest first, at most max_ngram - 1 long.
    """
    out: list[int] = []
    window = history + classes
    base = len(history)
    for position in range(base, len(window)):
        rolling = 0
        blocked = False
        tokens = []
        for shift in range(max_ngram):
            index = position - shift
            source = window[index] if index >= 0 else window[0]
            if position < shift or source == DEAD:
                blocked = True
            tokens.append(pad if blocked else source)
        # unigram first, then XOR one lookback at a time: after step i the rolling
        # value is the hash of the (i+1)-gram
        rolling = tokens[0] * multipliers[0]
        for shift in range(1, max_ngram):
            rolling ^= tokens[shift] * multipliers[shift]
            for head in range(heads):
                prime = primes[shift - 1][head]
                out.append(rolling % prime + offsets[(shift - 1) * heads + head])
    return out


def fnv1a64(values) -> int:
    """The map's fingerprint, over the little-endian bytes of each class in id order.
    The C side recomputes it from the generated payload, so the whole 129,280-entry
    map is verified by one number plus a count."""
    digest = 0xcbf29ce484222325
    for value in values:
        for byte in (value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF,
                     (value >> 24) & 0xFF):
            digest ^= byte
            digest = (digest * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return digest


def encode_token_map(lookup: list[int]) -> tuple[bytes, bytes, int]:
    """Classes for a token id are handed out on first appearance and reused after,
    so the map is fully described by *which* ids are first appearances plus, for
    every other id, which earlier id it repeats. That is a bitmap (16 KB) and one
    17-bit representative per repeat (64 KB) -- 80 KB, within 4% of what zlib
    achieves on any other encoding of the same map (274 KB packed, 517 KB raw)."""
    count = len(lookup)
    bitmap = bytearray((count + 7) // 8)
    reps: list[int] = []
    first: dict[int, int] = {}
    for token_id, class_id in enumerate(lookup):
        if class_id not in first:
            first[class_id] = token_id
            bitmap[token_id >> 3] |= 0x80 >> (token_id & 7)
        else:
            reps.append(first[class_id])
    packed = bytearray()
    accumulator = 0
    used = 0
    for rep in reps:
        accumulator = (accumulator << 17) | rep
        used += 17
        while used >= 8:
            used -= 8
            packed.append((accumulator >> used) & 0xFF)
    if used:
        packed.append((accumulator << (8 - used)) & 0xFF)
    return bytes(bitmap), bytes(packed), len(reps)


def render_token_map(lookup: list[int]) -> str:
    bitmap, packed, reps = encode_token_map(lookup)
    digest = fnv1a64(lookup)

    def payload(name: str, blob: bytes) -> list[str]:
        lines = [f"static const unsigned char {name}[{len(blob)}] = {{"]
        for start in range(0, len(blob), 16):
            chunk = blob[start:start + 16]
            lines.append("    " + " ".join(f"0x{byte:02x}," for byte in chunk))
        lines.append("};")
        return lines

    lines = [
        "/* deepseek_v41_engram_tokens.h -- generated by",
        " * tools/make_deepseek_v41_engram_fixture.py from the token map the reference",
        " * implementation builds (its own class-count assertion passes on it).",
        " * Do not edit by hand; regenerate with:",
        " *",
        " *   make_deepseek_v41_engram.py       (reconstruct + verify against the checkpoint)",
        " *   make_deepseek_v41_engram_fixture.py (freeze the artifacts)",
        " *",
        " * Layout: a bitmap marking the token ids that introduce a new class, in id",
        " * order, then 17 bits per *repeat* naming the earlier id it repeats (bits packed",
        " * MSB-first, like the bytes below). Rebuilding walks the ids once: a set bit",
        " * takes the next class number, a clear bit copies the class already computed",
        " * for its representative. 80,310 bytes describe all 129,280 ids exactly --",
        " * against 274,720 packed and 517,120 raw, and within 4% of zlib's floor on",
        " * either. */",
        "#ifndef COLIBRI_DEEPSEEK_V41_ENGRAM_TOKENS_H",
        "#define COLIBRI_DEEPSEEK_V41_ENGRAM_TOKENS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define COLI_V41_ENGRAM_TOKEN_COUNT {len(lookup)}",
        f"#define COLI_V41_ENGRAM_CLASS_COUNT {max(lookup) + 1}",
        f"#define COLI_V41_ENGRAM_REPEAT_COUNT {reps}",
        f"#define COLI_V41_ENGRAM_MAP_FNV1A64 UINT64_C(0x{digest:016x})",
        "",
    ]
    lines += payload("coli_v41_engram_first_bitmap", bitmap)
    lines.append("")
    lines += payload("coli_v41_engram_repeat_reps", packed)
    lines.append("")
    lines.append("#endif /* COLIBRI_DEEPSEEK_V41_ENGRAM_TOKENS_H */")
    return "\n".join(lines) + "\n"


def render_map_checks(lookup: list[int]) -> str:
    """Spot values and the fingerprint, for the C test."""
    samples = [0, 1, 2, 3, 100, 127, 1000, 5000, 99091, 128000, 129279]
    lines = [
        "/* Generated by tools/make_deepseek_v41_engram_fixture.py: what the C side must",
        " * reproduce when it rebuilds the token map from the packed header. The",
        " * fingerprint covers every one of the entries, the samples make a failure",
        " * readable. */",
        "#ifndef COLIBRI_DEEPSEEK_V41_ENGRAM_MAP_CHECKS_H",
        "#define COLIBRI_DEEPSEEK_V41_ENGRAM_MAP_CHECKS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define COLI_V41_ENGRAM_MAP_TOKEN_COUNT {len(lookup)}",
        f"#define COLI_V41_ENGRAM_MAP_CLASS_COUNT {max(lookup) + 1}",
        "",
        "typedef struct {",
        "    int token_id;",
        "    uint32_t class_id;",
        "} ColiV41EngramMapSample;",
        "",
    ]
    for token_id in samples:
        lines.append(f"/* token {token_id:6d} -> class {lookup[token_id]} */")
    lines.append("static const ColiV41EngramMapSample coli_v41_engram_map_samples[] = {")
    for token_id in samples:
        lines.append(f"    {{ {token_id}, {lookup[token_id]} }},")
    lines.append("};")
    lines.append("static const int coli_v41_engram_map_sample_count = "
                 f"{len(samples)};")
    lines.append("static const uint64_t coli_v41_engram_map_expected_fnv1a64 = "
                 f"UINT64_C(0x{fnv1a64(lookup):016x});")
    lines.append("")
    lines.append("#endif /* COLIBRI_DEEPSEEK_V41_ENGRAM_MAP_CHECKS_H */")
    return "\n".join(lines) + "\n"


def render_tables(layout: dict) -> str:
    layers = layout["layer_ids"]
    heads = layout["n_heads"]
    max_ngram = layout["max_ngram_size"]
    lines = [
        "/* deepseek_v41_engram_tables.h -- generated by",
        " * tools/make_deepseek_v41_engram_fixture.py from a layout verified against the",
        " * checkpoint (class count and table rows). Do not edit by hand.",
        " *",
        " * The primes are the bucket moduli: one per (layer, n-gram size, head), drawn in",
        " * order and never reused, so the per-head ranges inside a layer's table never",
        " * overlap. The offsets are the flat bucket starts. The multipliers come from the",
        " * reference's numpy PCG64 seeding and are FROZEN here: numpy does not promise",
        " * stream stability across releases, and a different multiplier rehashes the",
        " * whole 94 GiB table.",
        " */",
        "#ifndef COLIBRI_DEEPSEEK_V41_ENGRAM_TABLES_H",
        "#define COLIBRI_DEEPSEEK_V41_ENGRAM_TABLES_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define COLI_V41_ENGRAM_LAYERS {len(layers)}",
        f"#define COLI_V41_ENGRAM_MAX_NGRAM {max_ngram}",
        f"#define COLI_V41_ENGRAM_HEADS {heads}",
        f"#define COLI_V41_ENGRAM_HEAD_DIM {layout['head_dim']}",
        f"#define COLI_V41_ENGRAM_HASH_COLS {layout['n_hash_cols']}",
        f"#define COLI_V41_ENGRAM_COMPRESSED_VOCAB {layout['compressed_vocab_size']}",
        "",
        "static const int coli_v41_engram_layer_ids[COLI_V41_ENGRAM_LAYERS] = {"
        + ", ".join(str(layer) for layer in layers) + "};",
        "",
    ]
    for position, layer in enumerate(layers):
        lines.append(f"/* layer {layer}: {layout['table_rows'][position]} rows, "
                     f"{layout['primes'][position]} */".replace("'", ""))
    lines.append("static const int coli_v41_engram_primes[COLI_V41_ENGRAM_LAYERS]"
                 f"[{max_ngram - 1}][COLI_V41_ENGRAM_HEADS] = {{")
    for layer in layout["primes"]:
        rows = ",\n        ".join("{ " + ", ".join(str(prime) for prime in per_ngram) + " }"
                                 for per_ngram in layer)
        lines.append("    {\n        " + rows + "\n    },")
    lines.append("};")
    lines.append("")
    lines.append("static const int coli_v41_engram_offsets[COLI_V41_ENGRAM_LAYERS]"
                 "[COLI_V41_ENGRAM_HASH_COLS] = {")
    for offsets in layout["offsets"]:
        lines.append("    { " + ", ".join(str(value) for value in offsets) + " },")
    lines.append("};")
    lines.append("")
    lines.append("static const int64_t coli_v41_engram_multipliers[COLI_V41_ENGRAM_LAYERS]"
                 f"[{max_ngram}] = {{")
    for multipliers in layout["multipliers"]:
        lines.append("    { " + ", ".join(str(value) for value in multipliers) + " },")
    lines.append("};")
    lines.append("")
    lines.append("static const int coli_v41_engram_table_rows[COLI_V41_ENGRAM_LAYERS] = {"
                 + ", ".join(str(rows) for rows in layout["table_rows"]) + "};")
    lines.append("")
    lines.append("#endif /* COLIBRI_DEEPSEEK_V41_ENGRAM_TABLES_H */")
    return "\n".join(lines) + "\n"


def build_vectors(layout: dict, tokens: dict, pad_class: int) -> dict:
    lookup = tokens["lookup"]
    classes = [lookup[token] for token in range(len(lookup))]
    heads = layout["n_heads"]
    max_ngram = layout["max_ngram_size"]
    vectors = []
    sequences = {
        "ascii": list(range(0, 16)),
        "mixed": [128000, 5, 5, 99091, 7, 0, 100, 100, 100, 129279],
        "with_dead": [3, 4, DEAD, 6, 7, 8],
    }
    for name, ids in sequences.items():
        span = [classes[token] if token != DEAD else DEAD for token in ids]
        for layer_position, layer in enumerate(layout["layer_ids"]):
            primes = layout["primes"][layer_position]
            offsets = layout["offsets"][layer_position]
            multipliers = layout["multipliers"][layer_position]
            prefill = hash_ids(span, [], max_ngram, heads, primes, multipliers,
                               offsets, pad_class)
            # a decode step: the last class arrives with the previous ones as history
            decode = hash_ids([span[-1]], span[:-1], max_ngram, heads, primes,
                              multipliers, offsets, pad_class)
            vectors.append({
                "name": f"{name}_layer{layer}",
                "span": span,
                "layer": layer_position,
                "prefill": prefill,
                "decode": decode,
            })
    return {"max_ngram": max_ngram, "heads": heads,
            "hash_cols": layout["n_hash_cols"], "vectors": vectors}


def render_vectors(bundle: dict) -> str:
    lines = [
        "/* deepseek_v41_engram_vectors.h -- generated by",
        " * tools/make_deepseek_v41_engram_fixture.py from the verified layout.",
        " *",
        " * Golden n-gram hash ids, so the C side can be held to the reference without a",
        " * checkpoint: `span` is the compressed-id sequence (DEAD = -1 blocks the n-gram),",
        " * `prefill` the ids for the whole span and `decode` the ids for its last class",
        " * arriving alone with the rest as history. Both must match. */",
        "#ifndef COLIBRI_DEEPSEEK_V41_ENGRAM_VECTORS_H",
        "#define COLIBRI_DEEPSEEK_V41_ENGRAM_VECTORS_H",
        "",
        "#include <stdint.h>",
        "",
        "typedef struct {",
        "    const char *name;",
        "    const int *span;",
        "    int span_count;",
        "    int layer;",
        "    const int64_t *prefill;",
        "    const int64_t *decode;",
        "} ColiV41EngramVector;",
        "",
    ]
    names = []
    for vector in bundle["vectors"]:
        tag = vector["name"].replace("_", "__")
        names.append(tag)
        lines.append(f"static const int coli_v41_vec_{tag}_span[] = {{"
                     + ", ".join(str(value) for value in vector["span"]) + "};")
        lines.append(f"static const int64_t coli_v41_vec_{tag}_prefill[] = {{"
                     + ", ".join(str(value) for value in vector["prefill"]) + "};")
        lines.append(f"static const int64_t coli_v41_vec_{tag}_decode[] = {{"
                     + ", ".join(str(value) for value in vector["decode"]) + "};")
        lines.append("")
    lines.append("static const ColiV41EngramVector coli_v41_engram_vectors[] = {")
    for vector, tag in zip(bundle["vectors"], names):
        lines.append(f'    {{ "{vector["name"]}", coli_v41_vec_{tag}_span, '
                     f'{vector["span_count"] if "span_count" in vector else len(vector["span"])}, '
                     f'{vector["layer"]}, coli_v41_vec_{tag}_prefill, '
                     f'coli_v41_vec_{tag}_decode }},')
    lines.append("};")
    lines.append("")
    lines.append("static const int coli_v41_engram_vector_count = "
                 f"{len(bundle['vectors'])};")
    lines.append("")
    lines.append("#endif /* COLIBRI_DEEPSEEK_V41_ENGRAM_VECTORS_H */")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="the checkpoint config: supplies engram_pad_token_id")
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--tokens-header", type=Path, required=True)
    parser.add_argument("--map-checks", type=Path, required=True)
    args = parser.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    tokens = json.loads(args.tokens.read_text(encoding="utf-8"))
    layout["compressed_vocab_size"] = tokens["compressed_vocab_size"]
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    text = raw.get("text_config") or raw
    pad_token = text["engram_pad_token_id"]
    pad_class = tokens["lookup"][pad_token]
    print(f"pad token {pad_token} -> compressed class {pad_class}")

    tables = render_tables(layout)
    args.tables.write_text(tables, encoding="utf-8")
    print(f"wrote {args.tables} ({len(tables)} bytes)")

    banner = render_token_map(tokens["lookup"])
    args.tokens_header.write_text(banner, encoding="utf-8")
    print(f"wrote {args.tokens_header} ({len(banner)} bytes)")

    checks = render_map_checks(tokens["lookup"])
    args.map_checks.write_text(checks, encoding="utf-8")
    print(f"wrote {args.map_checks} ({len(checks)} bytes)")

    bundle = build_vectors(layout, tokens, pad_class)
    args.vectors.write_text(render_vectors(bundle), encoding="utf-8")
    total = sum(len(v["prefill"]) + len(v["decode"]) for v in bundle["vectors"])
    print(f"wrote {args.vectors} ({args.vectors.stat().st_size} bytes, "
          f"{len(bundle['vectors'])} vectors, {total} expected ids)")

    # self-check: the prefill path must agree with the decode path where they overlap
    for vector in bundle["vectors"]:
        cols = bundle["hash_cols"]
        tail = vector["prefill"][-cols:]
        if tail != vector["decode"]:
            print(f"MISMATCH: {vector['name']} prefill tail != decode")
            return 1
    print("self-check OK: the last prefill position matches the decode step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
