#!/usr/bin/env python3
"""Bootstrap the DeepSeek V4.1 engine fork from the V4 engine.

The V4.1-Flash architecture is the same *layout family* as V4 (native-load
verdict, docs/deepseek-v41-delta.md): fp4 experts + fp8-e4m3/UE8M0 dense, MLA
attention with sinks, DSA indexer, compressor, hyper-connections, MTP/DSpark
heads.  So the port starts as a mechanical fork of the amortised V4 engine
source instead of a from-scratch engine -- this script is that fork, kept in
the tree so the transformation stays reviewable and reproducible (re-run it
after an upstream V4 engine change to see the same deltas).

Fidelity rules:

* Every rename below is listed explicitly; nothing is renamed "by luck".
* Identifiers that the fork *shares* with files it does not own
  (native_quant.h, hybrid_split.h) are protected: renaming them would break
  the shared headers.  See PROTECTED.
* Line endings are preserved byte-for-byte.

Usage:  python tools/fork_deepseek_v41.py [--check]

--check regenerates in memory and reports whether the tree matches, without
writing (CI can use it to prove the fork is in sync with its V4 base).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- files: (source, target) -------------------------------------------------
FILES = [
    ("deepseek_v4.c", "deepseek_v41.c"),
    ("deepseek_v4.h", "deepseek_v41.h"),
    ("deepseek_v4_internal.h", "deepseek_v41_internal.h"),
    ("deepseek_v4_dspark.inc", "deepseek_v41_dspark.inc"),
    ("deepseek_v4_bank_pair.h", "deepseek_v41_bank_pair.h"),
    ("deepseek_v4_hybrid.h", "deepseek_v41_hybrid.h"),
    ("expert_store_registry.h", "deepseek_v41_expert_store_registry.h"),
    ("expert_store_registry.c", "deepseek_v41_expert_store_registry.c"),
    ("Makefile.deepseek-v4", "Makefile.deepseek-v41"),
    ("Makefile.deepseek-v4.units", "Makefile.deepseek-v41.units"),
]

# Files the script bootstraps once and the tree then owns: the V4.1 architecture
# deltas (config shape, shared KV/index, engram, DSpark, gates) are hand edits
# marked `V41 DELTA` in the sources and tracked in docs/deepseek-v41-delta.md.
# --check skips them instead of reporting them as drift.
HAND_MAINTAINED = {
    "deepseek_v41.c",
    "deepseek_v41.h",
    "deepseek_v41_internal.h",
}

# Identifiers owned by *shared* headers: the fork keeps calling them by name.
# (The two adapter-register entry points are declared in edge_adapters.h /
# segment_adapters.h as `coli_deepseek_v4_*`; the fork renames them to
# `coli_deepseek_v41_*` and those two headers gain the matching v41 declaration,
# exactly like every other family has its own entry point there.)
PROTECTED = [
    "COLI_V4_GPU_TIER",           # native_quant.h / expert_store.h GPU-tier gate
    "coli_v4_hybrid_ema",         # hybrid_split.h, shared with kimi_k3.c
    "coli_v4_hybrid_fill_count",  # hybrid_split.h
    # native_quant.h is shared infrastructure: it declares these quant entry
    # points and the engine implements them.  The fork has to keep answering to
    # the names the shared header declares, or the unit that includes it sees a
    # call to a function nobody declared (and the twin unit defines another one).
    "coli_v4_qdq_scratch",
    "coli_v4_gpu_fp8_matvec",
    "coli_v4_gpu_matvec_grouped",
    "coli_v4_gpu_fp8_matmul_batch",
]

# Kept out of the fork on purpose, with the reason (see docs/deepseek-v41-delta.md):
#   COLI_V4_GPU_TIER  – the V4 CUDA tier (coli_cuda_dsv4*.dll) implements V4
#                       kernels; V4.1 must not resolve them.  Not defined for the
#                       V4.1 build, so every `#ifdef COLI_V4_GPU_TIER` block
#                       compiles out and the CPU path is the only one.
#   coli_cuda_dsv4_*  – same: names stay with the V4 DLL.

# --- renames, in order: (pattern, replacement, description) ------------------
RENAMES = [
    # 1. headers/sources the fork owns, most specific names first
    (r"deepseek_v4_internal\.h", "deepseek_v41_internal.h", "internal header"),
    (r"deepseek_v4_dspark\.inc", "deepseek_v41_dspark.inc", "dspark include"),
    (r"deepseek_v4_bank_pair\.h", "deepseek_v41_bank_pair.h", "bank-pair header"),
    (r"deepseek_v4_hybrid\.h", "deepseek_v41_hybrid.h", "hybrid shim"),
    (r"expert_store_registry\.h", "deepseek_v41_expert_store_registry.h",
     "expert-store registry header"),
    (r"expert_store_registry\.c", "deepseek_v41_expert_store_registry.c",
     "expert-store registry source"),
    (r"deepseek_v4\.h", "deepseek_v41.h", "umbrella header"),
    (r"deepseek_v4\.c", "deepseek_v41.c", "engine source"),
    (r"deepseek_v4\.exe", "deepseek_v41.exe", "engine binary"),
    (r"Makefile\.deepseek-v4", "Makefile.deepseek-v41", "engine makefile"),
    # 2. include guards and preprocessor namespace
    # No \b before these two: the flags reach the compiler as -DCOLI_V4_... and
    # the guards sit behind the COLIBRI_/DEEPSEEK_ prefix, so a word-boundary
    # anchor silently skipped 60+ occurrences (and the header guards entirely).
    (r"DEEPSEEK_V4", "DEEPSEEK_V41", "header guard namespace"),
    (r"COLI_V4_", "COLI_V41_", "preprocessor namespace"),
    # 3. types and API
    (r"\bColiDeepSeekV4", "ColiDeepSeekV41", "config/plan types"),
    (r"\bDeepSeekV4", "DeepSeekV41", "adapter engine types"),
    (r"\bColiV4", "ColiV41", "engine/session types"),
    (r"\bcoli_v4_", "coli_v41_", "engine API symbols"),
    # 4. translation-unit-local helpers/types.  The lookbehind excludes a
    #    preceding alphanumeric so the shared CUDA backend keeps its own names
    #    (dsv4_cuda_*, DSV4_*): only a v4 that starts its own identifier moves.
    (r"(?<![A-Za-z0-9])v4_", "v41_", "file-local helpers"),
    (r"(?<![A-Za-z0-9])V4_", "V41_", "file-local macros"),
    (r"(?<![A-Za-z0-9])v4(?=[a-z])", "v41", "file-local globals"),
    (r"(?<![A-Za-z0-9])V4(?=[A-Z])", "V41", "file-local types"),
    # 5. two strings that must NOT stay equal to their V4 counterparts: the
    #    prefix-checkpoint magic (a V4 snapshot must never load into V4.1, and the
    #    layout it stores is V4's) and the hwinfo engine tag.  The magic stays 8
    #    characters: the writer stores exactly 8 bytes and the reader compares 8,
    #    so a 9-char magic is silently truncated.
    (r"COLIV4CK", "COLIV41C", "prefix-checkpoint magic"),
    (r"v4-cpu", "v41-cpu", "hwinfo engine tag"),
    # 6. prose / target names / model ids left over.  The (?!1) lookahead keeps
    #    these two idempotent: without it, the already-renamed `deepseek_v41`
    #    matched again and produced `deepseek_v411`.
    (r"deepseek-v4(?!1)", "deepseek-v41", "hyphenated names"),
    (r"deepseek_v4(?!1)", "deepseek_v41", "underscored names"),
]


def fork_text(text: str) -> tuple[str, list[tuple[str, int]]]:
    counts: list[tuple[str, int]] = []
    for index, name in enumerate(PROTECTED):
        sentinel = f"\x00P{index}\x00"
        occurrences = text.count(name)
        if occurrences:
            text = text.replace(name, sentinel)
    for pattern, replacement, description in RENAMES:
        text, moved = re.subn(pattern, replacement, text)
        counts.append((description + f"  ({pattern} -> {replacement})", moved))
    for index, name in enumerate(PROTECTED):
        text = text.replace(f"\x00P{index}\x00", name)
    return text, counts


CUDA_TIER_NOTE = """\
# V4.1 does NOT get the V4 CUDA tier.  coli_cuda_dsv4*.dll (backend_cuda_dsv4.cu)
# implements V4 kernels -- one KV per layer, no attention sink, no engram, V4's
# indexer/compressor wiring.  Pointing the V4.1 engine at it would upload V4.1
# weights into V4 maths, i.e. silently compute a different model, which is the
# one failure this port refuses to have.  A V4.1 tier is its own piece of work
# (attn sinks, shared KV/index, engram, DSpark layout) and arrives with its own
# backend + loader; until then V41_WIN_EXTRA_OBJS stays empty on every host and
# COLI_V4_GPU_TIER is deliberately undefined, so every `#ifdef COLI_V4_GPU_TIER`
# block in the sources (native_quant.h, expert_store.h) compiles out and the CPU
# path is the only path.  See docs/deepseek-v41-delta.md.
"""


def v41_makefile_deltas(text: str) -> str:
    """Structural edits the mechanical fork cannot make on its own."""
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")

    # (a) headline: say what this file is and where the CUDA tier went.
    old_head = text[: text.index("\n", text.index("The parent Makefile gates COLI_V41_SUPPORTED")) + 1]
    text = text.replace(old_head, """\
# DeepSeek V4.1-Flash amalgamation build -- x86-64/aarch64 Linux, Windows/MSYS2,
# and arm64 macOS.  Forked from Makefile.deepseek-v4 by tools/fork_deepseek_v41.py
# and hand-extended: the V4.1 deltas are marked V41 DELTA below and tracked in
# docs/deepseek-v41-delta.md.  The parent Makefile gates COLI_V41_SUPPORTED.
""")

    # (b) Windows flags: no GPU tier (see CUDA_TIER_NOTE).
    text = text.replace("""\t-DCOLI_V41_PIN_RAMP_REQUESTS=24 \\
\t-DCOLI_V4_GPU_TIER \\
\t-DCOLI_V41_EXPERIMENTAL_DUAL_EXPERT_LOADER
LDFLAGS = -lm -fopenmp -static -pthread
V41_BINARY := deepseek_v41.exe
V41_WIN_EXTRA_OBJS = backend_loader_dsv4.o
""", """\t-DCOLI_V41_PIN_RAMP_REQUESTS=24 \\
\t-DCOLI_V41_EXPERIMENTAL_DUAL_EXPERT_LOADER
LDFLAGS = -lm -fopenmp -static -pthread
V41_BINARY := deepseek_v41.exe
V41_WIN_EXTRA_OBJS =
""" + CUDA_TIER_NOTE)

    # (c) drop the CUDA=1 / DEEPGEMM block wholesale.
    start = text.index("# ---- Linux/macOS CUDA tier (opt-in): CUDA=1")
    end = text.index("include Makefile.deepseek-v41.units")
    text = text[:start] + CUDA_TIER_NOTE + "\n" + text[end:]

    # (d) drop the backend rules that block built.
    text = text.replace("""# The CUDA tier loader object. Compiled only on Windows, where it is appended
# to V41_OBJS and resolves the MSVC-built coli_cuda_dsv4.dll at runtime. On
# Linux it is empty so the engine links without it.
backend_loader_dsv4.o: backend_loader_dsv4.c backend_cuda_dsv4.h
\t$(CC) $(CFLAGS) -c backend_loader_dsv4.c -o $@

# Linux/macOS CUDA=1: the tier compiled straight into the engine.
backend_cuda_dsv4.o: backend_cuda_dsv4.cu backend_cuda_dsv4.h $(V41_DEEPGEMM_DEP)
\t"$(NVCC)" $(V41_NVCCFLAGS) -c backend_cuda_dsv4.cu -o $@

ifneq ($(DEEPGEMM_STAMP),)
$(DEEPGEMM_STAMP):
\tDEEPGEMM_HOME="$(DEEPGEMM_HOME)" tools/fetch_deepgemm.sh "$(DEEPGEMM_PIN)"
endif

""", CUDA_TIER_NOTE)

    # (e) the registry object must not collide with the V4 build's copy in the
    #     same directory, and the V4 registry test is V4's, not ours.
    text = text.replace("""REGISTRY_OBJ = expert_store_registry.o""",
                        """REGISTRY_OBJ = deepseek_v41_expert_store_registry.o
""")
    text = text.replace("""# Registry unit test (no engine link -- stubs the auto open fn).
deepseek-v41-test-registry: test_expert_store_registry
\t./test_expert_store_registry
test_expert_store_registry: deepseek_v41_expert_store_registry.c deepseek_v41_expert_store_registry.h expert_store.h
\t$(CC) -O2 -Wall -Wextra deepseek_v41_expert_store_registry.c deepseek_v41_expert_store_registry.c -o $@

""", """# V41 DELTA: the V4 registry unit test is not forked.  The registry object above
# still builds and links; its own unit test lands with the CUDA expert-store tier,
# which is the only thing that exercises it (the CPU path never opens a backend).

""")
    text = text.replace("""deepseek-v41-test-registry print-v4-objs""",
                        """deepseek-v41-test-objs""")
    text = text.replace("""print-v4-objs:
\t@echo $(V41_OBJS)""", """print-v41-objs:
\t@echo $(V41_OBJS)""")

    # (f) the serve-framing test is V4's runtime contract, built from V4's source;
    #     V4.1's own framing test comes with its serve path.
    text = text.replace("""V41_SERVE_TEST := tests/test_v41_serve_framing$(if $(IS_WIN),.exe,)
V41_SERVE_TEST_OBJS := $(filter-out COLI_V41_UNIT_GENERATE_STATS.o,$(V41_OBJS))

""", """# V41 DELTA: no serve-framing test yet.  tests/test_v4_serve_framing.c is the V4
# runtime's contract and is built from V4's source; V4.1 gets its own when its
# serve path exists (docs/deepseek-v41-delta.md, port plan step 5).

""")
    text = text.replace("""$(V41_SERVE_TEST): tests/test_v41_serve_framing.c deepseek_v41.c deepseek_v41.h \\
\t\tdeepseek_v41_internal.h serve_codec.h $(V41_SERVE_TEST_OBJS)
\t$(CC) $(CFLAGS) $< $(V41_SERVE_TEST_OBJS) -o $@ $(LDFLAGS)

""", "")
    text = text.replace("""\t\t$(V41_BATCH_TEST_OBJ) \\
\t\t$(V41_SERVE_TEST) test_expert_store_registry""",
                        """\t\t$(V41_BATCH_TEST_OBJ)""")

    return text.replace("\n", "\r\n") if crlf else text


def units_deltas(text: str) -> str:
    return text.replace(
        "# Generated by _amalgamate_v4.py — object units for amalgamated sources.",
        "# Object units for the amalgamated V4.1 source. Forked from\n"
        "# Makefile.deepseek-v4.units by tools/fork_deepseek_v41.py; the macros are\n"
        "# the -D names the engine's unit blocks use (COLI_V41_UNIT_*).")


DELTAS = {
    "Makefile.deepseek-v41": v41_makefile_deltas,
    "Makefile.deepseek-v41.units": units_deltas,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the fork is in sync without writing")
    args = parser.parse_args()

    total: dict[str, int] = {}
    stale: list[str] = []
    for source_name, target_name in FILES:
        source = ROOT / source_name
        target = ROOT / target_name
        if not source.exists():
            print(f"missing source: {source_name}", file=sys.stderr)
            return 2
        raw = source.read_bytes()
        text = raw.decode("utf-8")
        forked, counts = fork_text(text)
        apply_delta = DELTAS.get(target_name)
        if apply_delta is not None:
            forked = apply_delta(forked)
        for description, moved in counts:
            total[description] = total.get(description, 0) + moved
        data = forked.encode("utf-8")
        if args.check:
            if target_name in HAND_MAINTAINED:
                continue
            if not target.exists() or target.read_bytes() != data:
                stale.append(target_name)
            continue
        target.write_bytes(data)
        print(f"{target_name:38s} <- {source_name}  ({len(raw)} bytes)")

    if args.check:
        if stale:
            print("out of sync with the V4 base: " + ", ".join(stale))
            return 1
        print("fork is in sync with its V4 base "
              f"({len(HAND_MAINTAINED)} hand-maintained source(s) not compared: "
              + ", ".join(sorted(HAND_MAINTAINED)) + ")")
        return 0

    print()
    for description, moved in sorted(total.items(), key=lambda item: -item[1]):
        print(f"{moved:7d}  {description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
