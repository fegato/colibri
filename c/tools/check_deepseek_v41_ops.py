#!/usr/bin/env python3
"""Hold the torch reference to the engine's own primitives.

`tools/deepseek_v41_reference.py` reimplements six ops from the vendor's host wrappers.
Those wrappers document the semantics, but a reimplementation is still a reading -- so this
checker runs the *engine's* implementations (already validated by the V4 oracle, which is
what the family's tests pin) on fixed inputs, and compares:

    fp8_act_qdq       the quantized values AND the E8M0 scale bytes (the scale rule is
                      deterministic, so the bytes must match exactly, not approximately)
    fp8_matvec_128    the shared matvec at V4's 128-wide geometry, fed the same quantized
                      activation the engine builds internally
    hc_split_sinkhorn pre / post / comb after the sinkhorn iterations

Usage
-----
    python3 c/tools/check_deepseek_v41_ops.py --probe ./build/v41_ops_probe

When the probe runs on another platform (a Linux binary under a Windows python), capture its
output and read it instead -- the comparison is the same:

    ./build/v41_ops_probe > build/v41_ops_probe.jsonl      # where it was built
    python3 c/tools/check_deepseek_v41_ops.py --probe-output build/v41_ops_probe.jsonl

Build the probe from c/ (the objects the engine's own test links):

    gcc -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -O2 -I. tests/v41_ops_probe.c \
        COLI_V41_UNIT_NATIVE_QUANT.o COLI_V41_UNIT_MATH.o COLI_V41_UNIT_BLOCK_HYBRID.o \
        -o build/v41_ops_probe -lm -fopenmp -pthread
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import deepseek_v41_reference as ref  # noqa: E402 - after the path insert, on purpose

import torch  # noqa: E402


def _tensor(values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


def _dequantize_blocks(quantized, scales, block):
    """Rebuild what the engine's qdq returns: fp8 values times their block scale."""
    width = quantized.numel()
    scale = torch.pow(2.0, _tensor(scales, dtype=torch.float32) - 127.0)
    return quantized.float() * scale.repeat_interleave(block)[:width]


def _max_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        return float("inf")
    return float((a.float() - b.float()).abs().max())


def check_fp8_act_qdq(record: dict) -> list[str]:
    block = record["block"]
    x = _tensor(record["input"])
    quantized, scale_bytes = ref.act_quant(x, block, scale_fmt="ue8m0",
                                           scale_dtype=torch.float8_e8m0fnu)
    problems = []
    theirs = _tensor(record["scales"], dtype=torch.uint8)
    if list(scale_bytes.flatten().tolist()) != list(theirs.tolist()):
        problems.append(f"scale bytes differ: ours {scale_bytes.flatten().tolist()} "
                        f"vs engine {theirs.tolist()}")
    ours = _dequantize_blocks(quantized.flatten(), scale_bytes.flatten(), block)
    delta = _max_delta(ours, _tensor(record["output"]))
    if delta > 1e-6:
        problems.append(f"dequantized values differ by {delta:.3e}")
    print(f"  fp8_act_qdq block={block:<4d} scale bytes "
          f"{'identical' if not problems or 'scale' not in problems[0] else 'DIFFERENT'}, "
          f"max |delta| {delta:.3e}")
    return problems


def check_fp8_matvec(record: dict) -> list[str]:
    rows, columns = record["rows"], record["columns"]
    weight = _tensor(record["weight"], dtype=torch.uint8)
    scales = _tensor(record["scales"])
    x = _tensor(record["input"])
    # the engine quantizes the activation itself, at its own block width, then dequantizes
    # both sides for the product: do the same so the comparison is of the same pipeline
    block = 128
    quantized, scale_bytes = ref.act_quant(x, block, scale_fmt="ue8m0",
                                           scale_dtype=torch.float8_e8m0fnu)
    activation = _dequantize_blocks(quantized, scale_bytes.flatten(), block)
    # the probe prints the weight BYTES: they are e4m3 codes, and reading them as values
    # would scale the product by ~0x30
    weights = (weight.view(torch.float8_e4m3fn).float().reshape(rows, columns)
               * scales.reshape((rows + block - 1) // block, columns // block)
               .repeat_interleave(block, 0).repeat_interleave(block, 1)[:rows, :columns])
    ours = activation @ weights.t()
    delta = _max_delta(ours, _tensor(record["output"]))
    print(f"  fp8_matvec_128      {rows}x{columns} max |delta| {delta:.3e} "
          f"(relative {delta / max(1e-9, float(ours.abs().max())):.2e})")
    return [] if delta <= 1e-3 else [f"matvec differs by {delta:.3e}"]


def check_hc(record: dict) -> list[str]:
    hc, count = record["hc"], record["count"]
    mix_hc = (2 + hc) * hc
    mixes = _tensor(record["mixes"]).reshape(count, mix_hc)
    pre, post, comb = ref.hc_split_sinkhorn(mixes, _tensor(record["scale"]),
                                            _tensor(record["base"]), hc, 3, 1e-6)
    deltas = {
        "pre": _max_delta(pre.reshape(-1), _tensor(record["pre"])),
        "post": _max_delta(post.reshape(-1), _tensor(record["post"])),
        "comb": _max_delta(comb.reshape(-1), _tensor(record["comb"])),
    }
    print("  hc_split_sinkhorn   " + ", ".join(f"{k} {v:.3e}" for k, v in deltas.items()))
    return [f"{k} differs by {v:.3e}" for k, v in deltas.items() if v > 1e-4]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path,
                        help="the built v41_ops_probe binary, run here")
    parser.add_argument("--probe-output", type=Path,
                        help="or its captured JSONL, when the probe runs another platform")
    args = parser.parse_args()
    if not args.probe and not args.probe_output:
        parser.error("pass --probe or --probe-output")
    if args.probe_output:
        text = args.probe_output.read_text(encoding="utf-8")
    else:
        if not args.probe.is_file():
            raise SystemExit(f"no probe at {args.probe}: build it (the docstring has the "
                             "command) or capture its output and use --probe-output")
        try:
            run = subprocess.run([str(args.probe)], capture_output=True, text=True, timeout=600)
        except OSError as error:
            raise SystemExit(f"cannot execute {args.probe} ({error}): build the probe for this "
                             "platform, or run it where it was built and pass "
                             "--probe-output with the captured JSON lines")
        if run.returncode != 0:
            raise SystemExit(f"the probe failed: {run.stderr[-400:]}")
        text = run.stdout

    problems = []
    for line in text.splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        op = record.get("op")
        if "error" in record:
            problems.append(f"{op}: the engine refused the input ({record['error']})")
            continue
        print(f"{op}:")
        if op == "fp8_act_qdq":
            problems += check_fp8_act_qdq(record)
        elif op == "fp8_matvec_128":
            problems += check_fp8_matvec(record)
        elif op == "hc_split_sinkhorn":
            problems += check_hc(record)

    print()
    if problems:
        print("the torch reference does NOT match the engine:")
        for problem in problems:
            print("  -", problem)
        return 1
    print("parity OK: the reference matches the engine's primitives on every case above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
