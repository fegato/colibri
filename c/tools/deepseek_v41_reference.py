#!/usr/bin/env python3
"""The torch reference forward for DeepSeek V4.1-Flash, and the ops it needs.

Why this file exists
--------------------
The oracle for a family here is a tiny checkpoint plus the tokens the architecture
produces for a fixed prompt (see `tools/make_deepseek_v4_tiny.py` and its test). For V4.1
the *official* implementation shipped with the checkpoint cannot play that role on this
repo's CI:  `inference/kernel.py` is **tilelang**, GPU-only, imported at module level, so
the official forward needs a CUDA stack. Requiring that would make V4.1 the only family
whose oracle cannot run where `make check` runs.

So the reference is the official forward with only its six leaf ops reimplemented here, in
torch, and the architecture left alone: `inference/model.py` is imported unmodified, with a
module named `kernel` injected in front of it. Fidelity lives in their file; only the ops
are ours. The ops are written from the *host wrappers* in `inference/kernel.py` -- which
are pure torch and carry the semantics, the shapes and the assertions -- never from a
transcription of the tilelang bodies.

The vendor sources are taken **by path** (`--inference`), never vendored here: they are
DeepSeek's code under their own licence, and a copy would rot.

Usage
-----
    python3 c/tools/deepseek_v41_reference.py --inference /path/to/inference --self-check

`--self-check` builds the architecture at the small default config its own `ModelArgs`
documents, runs one short forward, and prints what came out. It is the feasibility probe
for the tiny oracle: if it returns logits, the oracle is mechanical.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - the CI has no torch, and must not need it
    raise SystemExit(
        "this tool needs torch (the oracle generators of the other families need it too): "
        f"{exc}"
    )

FP8_MAX = 448.0
FP4_MAX = 6.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class Unsupported(Exception):
    """A mechanism this reference does not implement -- raised, never approximated."""


# --- the six ops -------------------------------------------------------------- *
# Each mirrors the host wrapper in inference/kernel.py, including its assertions: the
# wrappers are the specification, the tilelang bodies are one implementation of it.


def _e8m0_scale(amax: torch.Tensor, limit: float) -> torch.Tensor:
    """2 ** ceil(log2(amax / limit)): the power-of-2 (MXFP) scale, as E8M0 stores it."""
    ratio = torch.clamp(amax / limit, min=1e-38)
    return torch.pow(2.0, torch.ceil(torch.log2(ratio)))


def _e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    exponent = torch.round(torch.log2(scale)) + 127.0
    return exponent.clamp(0, 255).to(torch.uint8)


def _scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 -> 2**(exponent - 127). A byte tensor is an exponent, never a value."""
    if scale.dtype in (torch.uint8, torch.int8):
        return torch.pow(2.0, scale.to(torch.int32) - 127)
    if "e8m0" in str(scale.dtype):
        return torch.pow(2.0, scale.view(torch.uint8).to(torch.int32) - 127)
    return scale.float()


def _round_to_e2m1(value: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 CODE INDEX (0..15), which is what the packed storage holds."""
    grid = torch.tensor(E2M1_GRID, dtype=value.dtype, device=value.device)
    magnitude = value.abs().unsqueeze(-1)
    index = (magnitude - grid).abs().argmin(dim=-1).to(torch.uint8)
    return index + torch.where(value < 0, torch.tensor(8, dtype=torch.uint8, device=value.device),
                               torch.tensor(0, dtype=torch.uint8, device=value.device))


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """[..., K // 2] packed nibbles -> [..., K]. Low nibble is the even K element."""
    if "float4" in str(packed.dtype):
        # torch's packed fp4 dtype has no cast implemented: reinterpret the storage
        packed = packed.view(torch.uint8)
    low = (packed.to(torch.int16) & 0x0F).to(torch.long)
    high = ((packed.to(torch.int16) >> 4) & 0x0F).to(torch.long)

    def decode(nibble):
        magnitude = torch.tensor(E2M1_GRID, dtype=torch.float32,
                                 device=nibble.device)[nibble & 0x07]
        return torch.where(nibble >= 8, -magnitude, magnitude)

    return torch.stack([decode(low), decode(high)], dim=-1).reshape(*packed.shape[:-1], -1)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 quantization; scale_fmt set means power-of-2 (MXFP) scales."""
    assert x.size(-1) % block_size == 0
    z = x.contiguous().float()
    *lead, n = z.shape
    blocks = z.reshape(*lead, n // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = _e8m0_scale(amax, FP8_MAX) if scale_fmt is not None else \
        torch.clamp(amax, min=1e-4) / FP8_MAX
    # the codes, NOT the dequantized values: multiplying back here and casting again would
    # quantize a second time at scale 1, which only looks right when the scale is 1
    codes = torch.clamp(blocks / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((codes.float() * scale).reshape(*lead, n).to(x.dtype))
        return x
    scale_out = (_e8m0_bytes(scale.squeeze(-1)) if scale_dtype == torch.float8_e8m0fnu
                 else scale.squeeze(-1).float())
    return codes.reshape(*lead, n), scale_out


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """FP4 (e2m1) with per-block scales: E8M0 for the indexer, E4M3 for compressed KV."""
    assert x.size(-1) % block_size == 0
    z = x.contiguous().float()
    *lead, n = z.shape
    blocks = z.reshape(*lead, n // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        scale = torch.clamp(amax, min=1e-4) / FP4_MAX
        scale = scale.to(torch.float8_e4m3fn).float()
        scale_out = scale.squeeze(-1).to(torch.float8_e4m3fn)
    else:
        scale = _e8m0_scale(amax, FP4_MAX)
        scale_out = _e8m0_bytes(scale.squeeze(-1))
    codes = _round_to_e2m1(torch.clamp(blocks / scale, -FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((codes * scale).reshape(*lead, n).to(x.dtype))
        return x
    # the kernel returns the packed [..., N // 2] tensor: pack the codes, not their values
    return _pack_codes(codes).reshape(*lead[:-1], n // 2), scale_out


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack e2m1 code indices (0..15, sign in bit 3) two per byte, low nibble first."""
    nibbles = codes.to(torch.uint8)
    nibbles = nibbles.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return nibbles[..., 0] | (nibbles[..., 1] << 4)


def _dequant_weight(values, scales, rows, columns, block):
    """Weight scales cover a block x block tile: repeat along rows *and* columns."""
    dense = values.reshape(rows, columns).float()
    scale = _scale_to_float(scales).reshape(-1, columns // block)
    expanded = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return dense * expanded[:rows, :columns]


def _dequant_activation(values, scales, rows, columns, block):
    """Activation scales are one row per token: repeat along K only."""
    dense = values.reshape(rows, columns).float()
    return dense * _scale_to_float(scales).reshape(rows, columns // block) \
        .repeat_interleave(block, dim=1)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[M,N] = A[M,K] @ B[N,K]^T, per-block FP8 scaling on both sides."""
    assert block_size in (32, 128)
    k = a.size(-1)
    m = a.numel() // k
    n = b.size(0)
    assert k % block_size == 0
    assert a_s.numel() == m * (k // block_size)
    assert b_s.numel() == ((n + block_size - 1) // block_size) * (k // block_size)
    dense = _dequant_activation(a, a_s, m, k, block_size)
    weights = _dequant_weight(b, b_s, n, k, block_size)
    return (dense @ weights.t()).reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T; B packed [N, K//2], scales per 32 on K."""
    assert act_block_size in (32, 128)
    k = a.size(-1)
    m = a.numel() // k
    n = b.size(0)
    assert k % act_block_size == 0
    assert a_s.numel() == m * (k // act_block_size)
    assert b_s.numel() == n * (k // 32)
    dense = _dequant_activation(a, a_s, m, k, act_block_size)
    weights = (_unpack_e2m1(b.reshape(n, k // 2))
               * _scale_to_float(b_s).reshape(n, k // 32).repeat_interleave(32, dim=1))
    return (dense @ weights.t()).reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gather top-k KV positions, flash-style softmax, plus a learnable attention sink.

    A row with no valid index (-1) yields zeros, not NaN: the kernel starts its running
    maximum from a finite bound for exactly that reason.
    """
    b, s, h, d = q.size()
    topk = topk_idxs.size(-1)
    valid = topk_idxs >= 0
    gathered = kv.gather(1, topk_idxs.clamp(min=0).reshape(b, -1, 1).expand(-1, -1, d))
    gathered = gathered.reshape(b, s, topk, d).float()
    scores = torch.einsum("bshd,bskd->bshk", q.float(), gathered) * softmax_scale
    scores = torch.nan_to_num(scores.masked_fill(~valid.unsqueeze(2), float("-inf")),
                              nan=float("-inf"))
    maximum = torch.clamp(scores.amax(dim=-1, keepdim=True), min=-1e30)
    weights = torch.exp(scores - maximum)
    denominator = weights.sum(dim=-1) + torch.exp(attn_sink.float().view(1, 1, h)
                                                  - maximum.squeeze(-1))
    return torch.einsum("bshk,bskd->bshd", weights / denominator.unsqueeze(-1),
                        gathered).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """pre/post/comb from one projection of the flattened stream, then sinkhorn on comb.

    mixes is [..., (2 + hc) * hc]. comb gets a softmax(-1) + eps, then alternating column
    and row normalisation, always `+ eps`.
    """
    lead = mixes.shape[:-1]
    mix_hc = (2 + hc_mult) * hc_mult
    flat = mixes.reshape(-1, mix_hc).float()
    n = flat.size(0)
    pre = torch.sigmoid(flat[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(flat[:, hc_mult:2 * hc_mult] * hc_scale[1]
                               + hc_base[hc_mult:2 * hc_mult])
    comb = (flat[:, 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:])
    comb = comb.reshape(n, hc_mult, hc_mult).softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return (pre.reshape(*lead, hc_mult).to(mixes.dtype),
            post.reshape(*lead, hc_mult).to(mixes.dtype),
            comb.reshape(*lead, hc_mult, hc_mult).to(mixes.dtype))


OPS = {
    "act_quant": act_quant,
    "fp4_act_quant": fp4_act_quant,
    "fp4_gemm": fp4_gemm,
    "fp8_gemm": fp8_gemm,
    "hc_split_sinkhorn": hc_split_sinkhorn,
    "sparse_attn": sparse_attn,
}

# Every other public name in the vendor's kernel.py, as a stub that raises when reached:
# a forward that needs something we did not implement must say so by name.
KERNEL_STUBS = ("fast_log2_ceil", "fast_pow2", "fast_round_scale",
                "act_quant_kernel", "fp4_quant_kernel", "fp8_gemm_kernel",
                "fp4_gemm_kernel", "sparse_attn_kernel", "hc_split_sinkhorn_kernel")


def _raiser(prefix: str, name: str):
    def _stub(*_args, **_kwargs):
        raise Unsupported(f"{prefix}.{name} is not implemented in the torch reference")
    return _stub


def _plain_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)

    def _missing(attribute):
        # AttributeError, not a custom type: the import system and hasattr() only
        # swallow that one, and anything else breaks `from module import name`.
        raise AttributeError(attribute)

    module.__getattr__ = _missing
    return module


def install_shims() -> None:
    """Put our ops in front of the vendor's `kernel`, and stub what the tiny does not use."""
    kernel = _plain_module("kernel")
    for name, function in OPS.items():
        setattr(kernel, name, function)
    for name in KERNEL_STUBS:
        setattr(kernel, name, _raiser("kernel", name))
    sys.modules["kernel"] = kernel

    # vision is off in the tiny fixture; these stay importable and loud if ever called
    images = _plain_module("image_processor")
    for index, name in enumerate(("IMAGE", "IMAGE_START", "IMAGE_END", "IMAGE_NEW_LINE")):
        setattr(images, name, 900000 + index)
    sys.modules["image_processor"] = images
    vision = _plain_module("vision")
    for name in ("ViT", "Aligner"):
        setattr(vision, name, _raiser("vision", name))
    sys.modules["vision"] = vision


def load_inference(directory: Path):
    """Import the vendor's `model` module with our ops in front of its kernels."""
    directory = directory.resolve()
    if not (directory / "model.py").is_file():
        raise SystemExit(
            f"no model.py in {directory}: pass --inference pointing at the released "
            "inference/ directory (the checkpoint repo ships it; it is not vendored here)"
        )
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    install_shims()
    import model  # noqa: PLC0415 - deliberately imported after the shims are installed

    return model


def build_vendor_model(model, *, seed: int = 0):
    """The architecture at the small default config its own ModelArgs documents.

    The driver's contract: `generate.py` sets the default dtype, not the model.
    """
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(seed)
    return model.Transformer(model.ModelArgs())


def self_check(directory: Path, tokens: list[int]) -> int:
    model = load_inference(directory)
    net = build_vendor_model(model)
    parameters = sum(parameter.numel() for parameter in net.parameters())
    print(f"built the architecture: {parameters / 1e6:.1f}M parameters, "
          f"default dtype {torch.get_default_dtype()}")
    with torch.no_grad():
        output_ids, logits, main_hidden = net(torch.tensor([tokens]))
    print(f"forward: sampled token {int(output_ids[0])}, logits {tuple(logits.shape)} "
          f"({logits.dtype}), main_hidden {type(main_hidden).__name__}")
    print("the reference is usable: the architecture runs unmodified on these six ops")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inference", type=Path, required=True,
                        help="the released inference/ directory (not vendored here)")
    parser.add_argument("--self-check", action="store_true",
                        help="build the small default model and run one forward")
    parser.add_argument("--tokens", default="1,5,9,3",
                        help="comma-separated prompt for --self-check")
    args = parser.parse_args()
    if args.self_check:
        return self_check(args.inference, [int(t) for t in args.tokens.split(",")])
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
