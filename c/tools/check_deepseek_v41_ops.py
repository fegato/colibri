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
    fp4_tables        the e2m1 value grid and the E8M0 exponent decode, against the vendor's
                      own FP4_TABLE and against 2**(byte - 127)
    fp4_expert_matvec the engine's fp4 matvec on a synthetic expert payload, held to the
                      vendor's own fp4 -> fp8 cast (`convert.py`, taken verbatim by AST): the
                      two must describe the *same tensor*, so the values must agree exactly,
                      not approximately

Usage
-----
    python3 c/tools/check_deepseek_v41_ops.py --probe ./build/v41_ops_probe \
        --inference /path/to/vendor/inference

`--inference` is needed for the fp4 checks only: they run the vendor's cast, which is not
vendored here (the rest of the file needs nothing outside the repo). Without it those two
records are reported as skipped, with the reason, rather than passed silently.

When the probe runs on another platform (a Linux binary under a Windows python), capture its
output and read it instead -- the comparison is the same:

    ./build/v41_ops_probe > build/v41_ops_probe.jsonl      # where it was built
    python3 c/tools/check_deepseek_v41_ops.py --probe-output build/v41_ops_probe.jsonl

Build the probe from c/ (the objects the engine's own test links):

    make -f Makefile.deepseek-v41 COLI_V41_UNIT_NATIVE_QUANT.o COLI_V41_UNIT_MATH.o \
        COLI_V41_UNIT_BLOCK_HYBRID.o
    gcc -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -O2 -I. tests/v41_ops_probe.c \
        COLI_V41_UNIT_NATIVE_QUANT.o COLI_V41_UNIT_MATH.o COLI_V41_UNIT_BLOCK_HYBRID.o \
        -o build/v41_ops_probe -lm -fopenmp -pthread

Two things that only show up off Windows: the engine's own flags force-include `pthread.h`
(`-include pthread.h`, which is what declares `pthread_once` here), and the link needs `-flto`
or the `-O3` objects keep references to units the probe does not link against -- the linker
drops them only with LTO. Build the units with `-D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -O3
-march=native -fopenmp -pthread -include pthread.h -flto` to match.
"""
from __future__ import annotations

import argparse
import ast
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


def _e8m0_values(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 exponents as floats, whether torch converts the dtype or not."""
    try:
        values = scale.float()
        if bool(torch.isfinite(values).all()) and float(values.min()) > 0.0:
            return values
    except (RuntimeError, TypeError):
        pass
    return torch.pow(2.0, scale.view(torch.uint8).float() - 127.0)


def vendor_cast(inference: Path) -> tuple:
    """The vendor's own `FP4_TABLE` and `cast_e2m1fn_to_e4m3fn`, taken verbatim.

    `convert.py` imports safetensors and tqdm, which a CPU-only check should not need, so
    the two definitions are lifted out of its AST and executed as they are written. Nothing
    is retyped: if the vendor changes the cast, this runs the changed cast.
    """
    path = inference / "convert.py"
    if not path.is_file():
        raise SystemExit(f"no convert.py under {inference}: --inference must point at the "
                         "vendor's inference directory")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "FP4_TABLE":
            wanted.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "cast_e2m1fn_to_e4m3fn":
            wanted.append(node)
    if len(wanted) != 2:
        raise SystemExit(f"convert.py no longer defines FP4_TABLE and "
                         f"cast_e2m1fn_to_e4m3fn at the top level (found {len(wanted)})")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["FP4_TABLE"], namespace["cast_e2m1fn_to_e4m3fn"]


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


def check_fp4_tables(record: dict, vendor: tuple | None) -> list[str]:
    """The engine's fp4 value grid and exponent decode, against the vendor's own table."""
    theirs = _tensor(record["e2m1"])
    exponents = _tensor(record["e8m0_bytes"], dtype=torch.uint8)
    decoded = _tensor(record["e8m0"])
    problems = []
    if vendor is not None:
        table, _ = vendor
        ours = table.to(theirs.dtype).reshape(-1)
        if ours.numel() != theirs.numel() or not bool(torch.equal(ours, theirs)):
            problems.append(f"e2m1 grid differs: engine {theirs.tolist()} "
                            f"vs vendor {ours.tolist()}")
    else:
        ours = _tensor(ref.E2M1_GRID, dtype=theirs.dtype)
        if not bool(torch.equal(ours, theirs)):
            problems.append(f"e2m1 grid differs: engine {theirs.tolist()}")
    want = torch.pow(2.0, exponents.float() - 127.0)
    delta = _max_delta(decoded, want)
    print(f"  fp4_tables          e2m1 grid {'identical' if not problems else 'DIFFERENT'} "
          f"to the vendor's, e8m0 decode max |delta| {delta:.3e}")
    if delta > 0.0:
        problems.append(f"e8m0 decode differs by {delta:.3e} (it is a pure exponent shift)")
    return problems


def check_fp4_expert_matvec(record: dict, vendor: tuple | None) -> list[str]:
    """The engine's fp4 expert read, held to the vendor's own fp4 -> fp8 cast.

    The cast folds the fp4 range into an fp8 payload (and returns the folded scale), so
    dequantizing *its* output with *its* scale must reproduce the tensor the engine reads
    natively: fp4 code x the checkpoint's per-row, per-32-column exponent. Both sides are
    exact binary values, so this is an equality, not a tolerance.
    """
    if vendor is None:
        print("  fp4_expert_matvec   SKIPPED: pass --inference DIR to run it against the "
              "vendor's cast")
        return []
    rows, columns = record["rows"], record["columns"]
    activation = float(record["activation"])
    payload = (_tensor(record["payload"], dtype=torch.uint8).view(torch.int8)
               .reshape(rows, columns // 2))
    scale_bytes = _tensor(record["scales"], dtype=torch.uint8)
    scales = torch.pow(2.0, scale_bytes.float() - 127.0).reshape(rows, columns // 32)
    _, cast = vendor
    fp8_weight, fp8_scale = cast(payload, scales)
    # the cast returns *one* folded exponent per fp8 block (32 rows x 32 columns): the row
    # variation has moved into the payload, so the scale is broadcast over the block's rows
    blocks = columns // 32
    folded = _e8m0_values(fp8_scale).reshape(rows // 32, blocks).repeat_interleave(32, dim=0)
    weights = fp8_weight.float() * folded.repeat_interleave(32, dim=1)[:, :columns]
    # the engine's matvec each output is activation * value * scale, and the activation is
    # a single 0.5 that the engine's own quantizer reproduces exactly at any block width
    expected = activation * weights[:, 0]
    delta = _max_delta(expected, _tensor(record["output"]))
    distinct = sorted(set(scale_bytes.tolist()))
    print(f"  fp4_expert_matvec   {rows}x{columns} max |delta| {delta:.3e} "
          f"(scales exercised: {[2 ** (b - 127) for b in distinct]})")
    if delta != 0.0:
        return [f"the engine's fp4 read and the vendor's cast disagree by {delta:.3e}"]
    return []


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
    parser.add_argument("--inference", type=Path,
                        help="the vendor's inference directory, for the fp4 checks (convert.py)")
    args = parser.parse_args()
    if not args.probe and not args.probe_output:
        parser.error("pass --probe or --probe-output")
    vendor = vendor_cast(args.inference) if args.inference else None
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
        elif op == "fp4_tables":
            problems += check_fp4_tables(record, vendor)
        elif op == "fp4_expert_matvec":
            problems += check_fp4_expert_matvec(record, vendor)
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
