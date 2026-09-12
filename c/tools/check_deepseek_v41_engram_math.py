#!/usr/bin/env python3
"""Hold the engram fetch and gate maths to the reference implementation.

The C side is covered by tests/test_deepseek_v41.c, but those expectations are
hand-derived: they prove the code does what this repository *says* the reference
does. This tool runs the reference's own classes instead, extracted verbatim from
`v41_ref_model.py` at run time, on the same inputs the C probe prints:

  * `ParallelEngramEmbedding.forward`   -- the row gather and fp8/E8M0 dequant
  * `Engram.forward`                    -- the gate maths, with the projection
                                           injected (the reference's `wkv` is an
                                           fp8 GEMM from the external `kernel`
                                           module; the projection itself is the
                                           engine's shared fp8 matvec, covered by
                                           the V4 tests)

    python tools/check_deepseek_v41_engram_math.py --probe ./engram_math_probe
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REFERENCE_MODEL = Path('C:/Users/david/v41_ref_model.py')


def extract_reference_classes() -> str:
    """Take `Linear`, `ParallelEngramEmbedding` and `Engram` verbatim out of the
    reference: `Engram.__init__` builds a `Linear`, so it comes along."""
    text = REFERENCE_MODEL.read_text(encoding='utf-8')
    start = text.index('class Linear')
    end = text.index('@lru_cache(2)')
    block = text[start:end]
    for required in ('class Linear', 'class ParallelEngramEmbedding', 'class Engram',
                     'def forward'):
        if required not in block:
            raise SystemExit(f'the reference moved ({required} missing): '
                             'update this extraction')
    return block


def build_reference():
    import torch
    from torch import nn
    import torch.nn.functional as F

    namespace = {
        'torch': torch, 'nn': nn, 'F': F,
        'fp8_block_size': 32,
        'fp4_block_size': 32,
        'scale_fmt': 'ue8m0',
        'scale_dtype': torch.float8_e8m0fnu,
        'default_dtype': torch.float8_e4m3fn,
        'world_size': 1, 'rank': 0,
    }
    exec(compile(extract_reference_classes(), 'reference_engram_classes', 'exec'),
         namespace)
    return namespace


def parse_probe(output: str) -> dict:
    fields: dict[str, list] = {}
    for line in output.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        if name in ('table', 'scales'):
            fields[name] = [int(value, 16) for value in parts[1:]]
        elif name in ('hc_mult', 'dim', 'table_rows', 'head_dim'):
            fields[name] = int(parts[1])
        elif name in ('order',):
            fields[name] = [int(value) for value in parts[1:]]
        else:
            fields[name] = [float(value) for value in parts[1:]]
    for required in ('table', 'scales', 'order', 'fetch', 'gate', 'stream', 'key',
                     'value', 'weight', 'head_dim', 'hc_mult', 'dim', 'table_rows'):
        if required not in fields:
            raise SystemExit(f'probe output is missing {required}')
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=2 ** -8)
    args = parser.parse_args()

    import torch

    finished = subprocess.run([str(args.probe)], capture_output=True, text=True)
    if finished.returncode != 0:
        print(finished.stdout, finished.stderr)
        raise SystemExit('the probe failed')
    probe = parse_probe(finished.stdout)
    namespace = build_reference()
    ParallelEngramEmbedding = namespace['ParallelEngramEmbedding']
    Engram = namespace['Engram']

    head_dim = probe['head_dim']
    rows = probe['table_rows']
    failures = 0

    # ---- row gather and dequant -------------------------------------------
    weight = torch.tensor(probe['table'], dtype=torch.uint8).view(
        rows, head_dim).view(torch.float8_e4m3fn)
    scale = torch.tensor(probe['scales'], dtype=torch.uint8).view(
        rows, head_dim // 32).view(torch.float8_e8m0fnu)
    embedding = ParallelEngramEmbedding(rows, head_dim)
    with torch.no_grad():
        embedding.weight.copy_(weight)
        embedding.scale.copy_(scale)
    reference_rows = embedding.forward(
        torch.tensor([probe['order']], dtype=torch.int64)).to(torch.float32)
    our_rows = torch.tensor(probe['fetch'], dtype=torch.float32).view(
        len(probe['order']), head_dim)
    # the reference rounds its dequantized rows to bf16 and returns them; ours are
    # fp32 and go through the same rounding, so the two agree exactly after it
    worst = float((reference_rows[0] - our_rows).abs().max().item())
    rounded_equal = torch.equal(reference_rows[0].to(torch.bfloat16),
                                our_rows.to(torch.bfloat16))
    print(f"row gather/dequant: max |delta| {worst:.3g} in fp32, "
          f"identical after the reference's bf16 rounding: {rounded_equal}")
    if not rounded_equal:
        failures += 1
        print(f"  MISMATCH at {int((reference_rows[0] - our_rows).abs().argmax())}")

    # ---- the gate ---------------------------------------------------------
    class Layout:
        head_dim = 0
        num_embeddings = ()
        max_ngram_size = 0
        n_heads = 0
        layer_ids = ()

    layout = Layout()
    layout.head_dim = head_dim
    layout.num_embeddings = (rows,)
    layout.max_ngram_size = 4
    layout.n_heads = 8
    layout.layer_ids = (0,)

    class Args:
        dim = 0
        hc_mult = 0
        norm_eps = 0.0

    args_object = Args()
    args_object.dim = probe['dim']
    args_object.hc_mult = probe['hc_mult']
    args_object.norm_eps = 1e-6

    eng = Engram(args_object, 0, layout)
    stream = torch.tensor(probe['stream'], dtype=torch.float32).view(
        1, 1, probe['hc_mult'], probe['dim'])
    key = torch.tensor(probe['key'], dtype=torch.float32).view(
        probe['hc_mult'], probe['dim'])
    value = torch.tensor(probe['value'], dtype=torch.float32)
    with torch.no_grad():
        eng.q_weight.copy_(torch.tensor(probe['weight'], dtype=torch.float32).view(
            probe['hc_mult'], probe['dim']))
        eng.k_weight.copy_(torch.ones(probe['hc_mult'], probe['dim']))

        # Inject the projection: nn.Module enforces the child type, and from here the
        # reference's own forward body runs unchanged -- that is the part verified.
        class Projection(torch.nn.Module):
            def forward(self, _input):
                # what the reference's own wkv returns: [key | value] along the last
                # axis, key being hc_mult x dim and value dim
                return torch.cat([key.flatten(), value.flatten()]).view(1, 1, -1)

        eng.wkv = Projection()
        reference = eng.forward(stream, torch.zeros(1, 1, 1, dtype=torch.int64))
    ours = torch.tensor(probe['gate'], dtype=torch.float32).view(
        probe['hc_mult'], probe['dim'])
    worst = float((reference[0, 0] - ours).abs().max().item())
    print(f"gate maths:        max |delta| {worst:.3g} "
          f"(k_weight = 1, so weight is q_weight; the reduction order differs by "
          f"last bits, tolerance {args.tolerance:g})")
    if worst > args.tolerance:
        failures += 1
        print(f"  reference {reference[0, 0].flatten().tolist()}")
        print(f"  ours      {ours.flatten().tolist()}")

    if failures:
        print(f"\n{failures} mismatch(es) against the reference implementation")
        return 1
    print("\nreference parity OK: the row gather and the gate maths match the "
          "reference's own classes")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
