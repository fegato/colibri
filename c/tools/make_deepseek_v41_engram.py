#!/usr/bin/env python3
"""Reference check for the V4.1 engram compressed token map.

The engram n-grams are hashed over a *compressed* id space: tokens whose text
normalizes alike collapse onto one class. The classes are numbered in token-id
order, so both the normalizer chain and the iteration range are load-bearing --
a different order silently rehashes every table entry.

This script reproduces the reference algorithm (the checkpoint's own engram
implementation) using the real Rust tokenizers library, and checks the one thing
that can be verified without any weights: the number of classes must equal the
config's `engram_compressed_vocab_size`. Usage:

    python tools/make_deepseek_v41_engram.py --tokenizer tokenizer.json \
        --config config.json --output engram_tokens.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def reference_token_map(tokenizer_path: Path):
    from tokenizers import Regex, Tokenizer, normalizers

    # A private-use char, so a token that is exactly one space survives Strip()
    # instead of collapsing to the empty string and merging with unrelated tokens.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # The reference runs on `tokenizer.backend_tokenizer` (the HF fast-tokenizer
    # wrapper); the raw `tokenizers.Tokenizer` *is* that Rust object, so it is used
    # directly here.
    backend = tokenizer
    # The reference's len(tokenizer) counts the BPE vocab plus the added tokens.
    count = backend.get_vocab_size(with_added_tokens=True)
    key_to_class: dict[str, int] = {}
    lookup: list[int] = [0] * count
    for token_id in range(count):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, key it by raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_class.get(key)
        if new_id is None:
            new_id = len(key_to_class)
            key_to_class[key] = new_id
        lookup[token_id] = new_id
    return lookup, key_to_class


def is_prime(value: int) -> bool:
    """Deterministic Miller-Rabin: exact for every value these tables can reach."""
    if value < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % small == 0:
            return value == small
    d, s = value - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for base in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(base, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above `start` not handed out yet: bucket ranges never
    overlap, which is what keeps the per-(n-gram, head) ranges disjoint."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def hash_layout(text: dict):
    """Reproduce EngramLayout: the primes, the flat bucket offsets and the per-layer
    multipliers. The multipliers come from numpy's PCG64 seeded with 10007*layer, so
    they are *frozen* here rather than recomputed at build time (numpy does not
    promise stream stability across versions, and a different multiplier rehashes
    every row)."""
    import numpy as np

    layer_ids = tuple(text["engram_layer_ids"])
    max_ngram = text["engram_max_ngram_size"]
    heads = text["engram_n_heads"]
    vocab = text["engram_vocab_size"]
    compressed = text["engram_compressed_vocab_size"]

    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram - 1):
            sizes, current = [], vocab - 1
            for _ in range(heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))

    offsets, rows = [], []
    for layer in primes:
        flat = [prime for per_ngram in layer for prime in per_ngram]
        running, acc = [0], 0
        for prime in flat[:-1]:
            acc += prime
            running.append(acc)
        offsets.append(running)
        rows.append(sum(flat))

    multiplier_bound = max(1, (2**63 - 1) // compressed // 2)
    multipliers = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound,
                                    size=(max_ngram,), dtype=np.int64)
        multipliers.append([int(value) * 2 + 1 for value in values])
    return {
        "layer_ids": list(layer_ids),
        "max_ngram_size": max_ngram,
        "n_heads": heads,
        "head_dim": text["engram_head_dim"],
        "n_hash_cols": (max_ngram - 1) * heads,
        "multiplier_bound": multiplier_bound,
        "primes": [[list(per_ngram) for per_ngram in layer] for layer in primes],
        "offsets": offsets,
        "table_rows": rows,
        "multipliers": multipliers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", type=Path)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    text = raw.get("text_config") or raw
    expected = text["engram_compressed_vocab_size"]

    lookup, classes = reference_token_map(args.tokenizer)
    print(f"token ids covered      : {len(lookup)}")
    print(f"distinct classes       : {len(classes)}")
    print(f"config says            : {expected}")
    members: dict[int, int] = {}
    for class_id in lookup:
        members[class_id] = members.get(class_id, 0) + 1
    merged = sum(1 for count in members.values() if count > 1)
    print(f"classes with >1 member : {merged} (of {len(classes)})")
    print(f"largest class          : {max(members.values())} members")
    if len(classes) != expected:
        print("\nMISMATCH: the compressed token map does not reproduce "
              "engram_compressed_vocab_size -- do not hash anything with it")
        return 1
    print("reference parity OK: class count matches engram_compressed_vocab_size")

    # The bucket layout is checkable without any weights: the 24 prime moduli of a
    # layer are its bucket ranges, so their sum has to be the row count the
    # checkpoint declares for that layer's table.
    layout = hash_layout(text)
    declared = list(text["engram_num_embeddings"])
    print()
    for position, layer_id in enumerate(layout["layer_ids"]):
        computed, stated = layout["table_rows"][position], declared[position]
        print(f"engram layer {layer_id:2d}: {computed} rows computed, {stated} declared"
              f"  {'OK' if computed == stated else 'MISMATCH'}")
        if computed != stated:
            print("MISMATCH: the prime layout does not reproduce the table size")
            return 1
    for position, layer_id in enumerate(layout["layer_ids"]):
        print(f"  multipliers layer {layer_id}: "
              + ", ".join(str(value) for value in layout["multipliers"][position]))
    print("reference parity OK: prime layout sums to the declared table rows")

    if args.output:
        args.output.write_text(json.dumps({
            "compressed_vocab_size": len(classes),
            "token_count": len(lookup),
            "lookup": lookup,
        }), encoding="utf-8")
        print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    if args.layout:
        args.layout.write_text(json.dumps(layout, indent=1), encoding="utf-8")
        print(f"wrote {args.layout} ({args.layout.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
