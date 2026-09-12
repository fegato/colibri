#!/usr/bin/env python3
"""Compare the per-unit warning profile of the V4.1 fork against its V4 base.

The fork is only trustworthy if a tier-less build of `deepseek_v41.c` warns
exactly as much as a tier-less build of `deepseek_v4.c` (the shared base): a
warning that appears on one side only is either a fork defect or a real V4.1
delta and has to be explained.  Run it with a compiler on PATH:

    python tools/v41_warning_parity.py --cc gcc --arch x86-64-v3

Exit status is nonzero when the two profiles differ.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITS = re.findall(r"COLI_V4_UNIT_[A-Z0-9_]+",
                   (ROOT / "Makefile.deepseek-v4.units").read_text(encoding="utf-8"))
WARNING = re.compile(r"^(deepseek_v\d+\.c):(\d+):\d+: warning: (.*)$")


def flags(cc: str, arch: str, unit: str, flavour: str) -> list[str]:
    prefix = "COLI_V4" if flavour == "v4" else "COLI_V41"
    return [
        cc, "-D_GNU_SOURCE", "-D_FILE_OFFSET_BITS=64", "-O2", f"-march={arch}",
        "-fopenmp", "-include", "pthread.h", "-Wall", "-Wextra",
        "-Wno-unused-parameter", "-Wno-misleading-indentation",
        "-Wno-unused-function",
        f"-D{prefix}_MAX_PIN_SLOTS_PER_LAYER=16", f"-D{prefix}_PIN_RAMP_REQUESTS=24",
        f"-D{prefix}_EXPERIMENTAL_DUAL_EXPERT_LOADER", f"-D{unit}", "-c",
    ]


def profile(cc: str, arch: str, flavour: str, out_dir: Path) -> dict[str, list[str]]:
    """Tier-less warning profile per unit: the base always, the fork from the base."""
    source = "deepseek_v4.c" if flavour == "v4" else "deepseek_v41.c"
    units = UNITS if flavour == "v4" else [u.replace("COLI_V4_", "COLI_V41_") for u in UNITS]
    result: dict[str, list[str]] = {}
    for unit in units:
        command = flags(cc, arch, unit, flavour) + [
            source, "-o", str(out_dir / f"{flavour}_{unit}.o")]
        finished = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        lines = []
        for line in finished.stderr.splitlines():
            match = WARNING.match(line)
            if match:
                lines.append(match.group(3).strip())
        if finished.returncode != 0:
            lines.append("ERROR: compile failed")
        result[unit] = sorted(lines)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--arch", default="x86-64-v3")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        out_dir = Path(scratch)
        base = profile(args.cc, args.arch, "v4", out_dir)
        fork = profile(args.cc, args.arch, "v41", out_dir)

    different = 0
    for v4_unit, warnings in base.items():
        v41_unit = v4_unit.replace("COLI_V4_", "COLI_V41_")
        if fork[v41_unit] == warnings:
            continue
        different += 1
        print(f"{v41_unit}:")
        for line in sorted(set(fork[v41_unit]) - set(warnings)):
            print(f"    only in V4.1: {line}")
        for line in sorted(set(warnings) - set(fork[v41_unit])):
            print(f"    only in V4:   {line}")

    total_v41 = sum(len(w) for w in fork.values())
    total_v4 = sum(len(w) for w in base.values())
    print(f"{len(UNITS)} units; warnings: V4 base {total_v4}, V4.1 fork {total_v41}, "
          f"units with a different profile: {different}")
    return 1 if different else 0


if __name__ == "__main__":
    raise SystemExit(main())
