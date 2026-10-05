"""The kernel loader, the block-scale contract, the matched delta, the input contract and the plain route.

  _ops()                    loads the compiled library (delta_matching._C, or the file named by DM_SO) on first use
  blocks(D)                 rows per quantization scale: (Q / dO rows, K / V keys) = (64, 32) at head_dim 256, (128, 64) at 128
  _matched_delta_inkernel   the matched delta from pass 0 of the backward kernel (the backward's own fp8 P / dP pair)
  _check_inputs             the input contract shared by both routes
  dm_attn_func_plain        the plain route: the API route's numerics (bitwise) through the quant.py / doq.py producers
                            and the bwd_fp8 op, which makes the Q / dO twins itself; kept as a cross-check of the API route
Workspace: the backward materializes its dS' code tiles in e4m3 over [batch * heads, N, N] bytes (1 GiB at batch 1, 16 heads,
N = 8192), transient per call; the head_dim-256 matched delta adds 1/8 of that in fp32 partial sums (split.py bounds both).
Determinism: O, dQ, dK and dV are bitwise deterministic (no atomics or cross-CTA reductions in any pass).
"""
import os

import torch

from .quant import fused_quant
from .doq import fused_doq2

# Quantization scale-block rows: Q/dO scales per 64 rows, K/V scales per 32 rows
# (blocks(256); the kernels' per-tile scale lookups are compiled against
# these numbers, so they are not tunable from here).
BM, BN = 64, 32


def blocks(D: int):
    """The block-scale contract per head dim: (Q / dO rows, K / V keys) per scale = (64, 32) at head_dim 256 (BM, BN
    above), (128, 64) at head_dim 128.  The kernels' scale lookups are compiled against these numbers per head dim."""
    return (BM, BN) if D >= 256 else (128, 64)


_HD128_CHECKED = False


def _check_hd(D, ops):
    """## M19: head_dim 128 needs a Delta-Matching-128 binary (build_hd128() == 1); refuse loudly otherwise."""
    global _HD128_CHECKED
    if D == 256:
        return
    if D != 128:
        raise ValueError(f"delta_matching: head_dim must be 128 or 256 (got {D})")
    if not _HD128_CHECKED:
        if int(ops.build_hd128()) != 1:
            raise RuntimeError("head_dim 128 needs a Delta-Matching-128 binary (FA3FP11_HD128=1 build) at DM_SO")
        _HD128_CHECKED = True


def _quant_native(q, k, v):
    """## M16 native GQA: fused_quant's per-tensor loop with K/V at their own
    head count (fused_quant asserts one shape for q/k/v).  Same Triton
    kernels, same per-(bh, block) math: codes/scales/twin are identical to
    fused_quant's on an expanded copy, per KV head."""
    from .quant import _kmean_kernel, _quant_kernel
    B, H, N, D = q.shape
    Hk = k.shape[1]
    dev = q.device
    sm = D ** -0.5
    BM, BN = blocks(D)   # ## M19: per head dim
    qc = q.reshape(B * H, N, D).contiguous()
    kc = k.reshape(B * Hk, N, D).contiguous()
    vc = v.reshape(B * Hk, N, D).contiguous()
    km = torch.empty(B * Hk, D, device=dev, dtype=torch.float32)
    _kmean_kernel[(B * Hk, D // 64)](kc, km, N, D=D, BD=64)
    outs = []
    for (x, mode, blk, wt8, bh, h) in ((qc, 0, BM, False, B * H, H), (kc, 1, BN, True, B * Hk, Hk),
                                       (vc, 2, BN, False, B * Hk, Hk)):
        x8 = torch.empty(bh, N, D, device=dev, dtype=torch.float8_e4m3fn)
        x8s = torch.empty(bh, N // blk, device=dev, dtype=torch.float32)
        x8t = torch.empty(bh, D, N, device=dev, dtype=torch.float8_e4m3fn) if wt8 else x8
        _quant_kernel[(N // blk, bh)](x, km, x8, x8s, x8, x8, x8t, N, h,
                                      SM=sm, BLK=blk, D=D, MODE=mode, WANT_T16=False,
                                      T16_IS_FP8=False, WANT_F16=False, T16_IS_BF16=False,
                                      F16_IS_BF16=False, WANT_T8=wt8, num_warps=8)
        outs.append((x8, x8s, x8t if wt8 else None))
    (Q8, Q8s, _), (K8, K8s, kTt), (V8, V8s, _) = outs
    return Q8, Q8s, K8, K8s, V8, V8s, kTt

_LOADED = False


def _ops():
    """Load the compiled extension on first use: `delta_matching._C` (built by setup.py / the JIT build) or the shared
    library named by DM_SO.  Nothing else is ever loaded in its place."""
    global _LOADED
    if not _LOADED:
        so = os.environ.get("DM_SO")
        if so:
            if not os.path.isfile(so):
                raise RuntimeError(f"DM_SO={so!r} does not exist")
        else:
            import glob
            here = os.path.dirname(os.path.abspath(__file__))
            cands = sorted(glob.glob(os.path.join(here, "_C*.so")))
            if not cands:
                raise RuntimeError("delta_matching._C is not built: `pip install .` (or `python -m delta_matching.jit`) in the "
                                   "delta-matching checkout, or point DM_SO at a built library")
            if len(cands) > 1:
                raise RuntimeError(f"several built libraries in {here}: {[os.path.basename(c) for c in cands]} (a pip build "
                                   "and a JIT build side by side?); remove the stale one, or name one with DM_SO")
            so = cands[0]
        torch.ops.load_library(so)
        _ops.so_path = so
        _LOADED = True
    return torch.ops.delta_matching


def loaded_library():
    """The path of the loaded shared library (None before the first _ops() call)."""
    return getattr(_ops, "so_path", None)


def _check_inputs(q, k, v, causal, softmax_scale, mode):
    """The input contract of both routes, on (B, H, N, D) views: raises ValueError / TypeError / NotImplementedError
    naming the violated constraint (explicit checks, so they also hold under `python -O`)."""
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
        raise TypeError("delta_matching: q, k, v must be torch tensors")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"delta_matching: q, k, v must be 4-D (got {q.dim()}-D, {k.dim()}-D, {v.dim()}-D)")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError(f"delta_matching: q, k, v must be CUDA tensors (got {q.device}, {k.device}, {v.device})")
    if not (q.device == k.device == v.device):
        raise ValueError(f"delta_matching: q, k, v must be on one device (got {q.device}, {k.device}, {v.device})")
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        raise TypeError(f"delta_matching: q, k, v must be torch.bfloat16 (got {q.dtype}, {k.dtype}, {v.dtype})")
    if mode not in ("matched", "stale"):
        raise ValueError(f"delta_matching: mode must be 'matched' or 'stale' (got {mode!r})")
    if mode == "matched" and not causal:
        raise ValueError("delta_matching: mode='matched' needs causal=True (the matched delta is built for causal attention); "
                         "use mode='stale' for non-causal attention")
    B, H, N, D = q.shape
    if k.shape != v.shape:
        raise ValueError(f"delta_matching: k and v must have the same shape (got {tuple(k.shape)} and {tuple(v.shape)})")
    Bk, Hk, Nk, Dk = k.shape
    if Bk != B or Dk != D:
        raise ValueError(f"delta_matching: k and v must match q's batch size and head_dim (q {tuple(q.shape)}, k {tuple(k.shape)})")
    if Nk != N:
        raise ValueError(f"delta_matching: q, k, v must have the same seqlen (self-attention only; got {N} and {Nk})")
    if Hk == 0 or H % Hk:
        raise ValueError(f"delta_matching: nheads ({H}) must be a positive multiple of nheads_k ({Hk})")
    if D not in (128, 256):
        raise ValueError(f"delta_matching: head_dim must be 128 or 256 (got {D})")
    mult = 256 if D == 128 else 128
    if N == 0 or N % mult:
        raise ValueError(f"delta_matching: seqlen must be a positive multiple of {mult} at head_dim {D} (got {N})")
    if B == 0:
        raise ValueError("delta_matching: empty batch")
    if B * H > 65535:
        raise ValueError(f"delta_matching: batch * nheads must be at most 65535 (got {B * H})")
    if softmax_scale is not None and abs(softmax_scale - D ** -0.5) > 1e-12:
        raise NotImplementedError(f"delta_matching: softmax_scale is fixed at head_dim ** -0.5 = {D ** -0.5!r} (it is baked "
                                  f"into the Q codes); got {softmax_scale!r}")
    major, minor = torch.cuda.get_device_capability(q.device)
    if (major, minor) != (9, 0):
        raise ValueError(f"delta_matching: needs a Hopper GPU (sm90); {q.device} is sm{major}{minor}")


def _matched_delta_inkernel(Q8, Q8s, K8, K8s, V8, V8s, dO8, dO8s, kTt, Lse_bh,
                            H, N, D, causal=True, Hk=None):
    """The matched delta: the delta_fp8 op runs pass 0 of the backward kernel, which recomputes the backward's own fp8
    P and dP tiles bit for bit, and reduces rowsum(P * dP) per row in a fixed order (fp64), in the units the backward
    consumes.  Operands as the backward's (codes, scales, the K twin kTt, the natural-log LSE); returns (B*H, N) fp32."""
    ops = _ops()
    B = Q8.shape[0] // H
    Hk = H if Hk is None else Hk   # ## M16 native GQA: K/V per KV head
    _check_hd(D, ops)

    def bshd(x, h=H):
        return x.view(B, h, N, D).permute(0, 2, 1, 3)

    return ops.delta_fp8(bshd(dO8), bshd(Q8), bshd(K8, Hk), bshd(V8, Hk),
                         Lse_bh.view(B, H, N).contiguous(),
                         kTt.view(B, Hk, D, N),
                         Q8s.view(B, H, -1), K8s.view(B, Hk, -1),
                         V8s.view(B, Hk, -1), dO8s.view(B, H, -1),
                         causal).view(B * H, N)


class _DMAttnPlain(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, delta_mode):
        with torch.cuda.device(q.device):   # the Triton producers launch on the current device
            return _DMAttnPlain._forward(ctx, q, k, v, causal, delta_mode)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dO):
        with torch.cuda.device(dO.device):
            return _DMAttnPlain._backward(ctx, dO)

    @staticmethod
    def _forward(ctx, q, k, v, causal, delta_mode):
        B, H, N, D = q.shape
        Hk = k.shape[1]
        ops = _ops()
        _check_hd(D, ops)   # ## M19
        sm = D ** -0.5
        BMd, BNd = blocks(D)
        # Codes [BH, N, D] e4m3, scales [BH, N/blk] fp32, kTt [BH, D, N] e4m3
        # (the K twin; the bwd_fp8 op makes the Q / dO twins itself).
        # smooth-K centering and the sm bake happen inside.
        if Hk == H:
            Q8, Q8s, K8, K8s, V8, V8s, kTt = fused_quant(
                q, k, v, BMd, BNd, want_t16=False, want_t8s=(False, True, False))
        else:
            # ## M16 native GQA (READ side): K/V are quantized PRE-expansion
            # (B*Hk heads) and the ops index them by KV head; the per-block
            # codes/scales are identical to fused_quant's on an expanded copy.
            Q8, Q8s, K8, K8s, V8, V8s, kTt = _quant_native(q, k, v)

        def bshd(x, h=H):
            return x.view(B, h, N, D).permute(0, 2, 1, 3)

        O, Lse = ops.fwd_fp8(bshd(Q8), bshd(K8, Hk), bshd(V8, Hk),
                             Q8s.view(B, H, -1), K8s.view(B, Hk, -1),
                             V8s.view(B, Hk, -1), sm, causal)
        ctx.save_for_backward(Q8, Q8s, K8, K8s, V8, V8s, kTt, O, Lse)
        ctx.dims = (B, H, N, D)
        ctx.hk = Hk
        ctx.causal = causal
        ctx.delta_mode = delta_mode
        return O  # (B, N, H, D) bf16 contiguous

    @staticmethod
    def _backward(ctx, dO):
        B, H, N, D = ctx.dims
        Hk = ctx.hk
        ops = _ops()
        sm = D ** -0.5
        saved = ctx.saved_tensors
        Q8, Q8s, K8, K8s, V8, V8s, kTt, O, Lse = saved[:9]
        # dO arrives in O's (B, N, H, D) layout; doq2 wants (B, H, N, D) and
        # makes its own contiguous copy (counted in whole-composition timing).
        dO8, dO8s, Dlt, _ = fused_doq2(dO.permute(0, 2, 1, 3), blocks(D)[0], O, H,
                                       want_dof=False)
        if ctx.delta_mode == "matched":
            # the in-kernel pass-0 + fixed-order-reduce delta
            Dlt = _matched_delta_inkernel(Q8, Q8s, K8, K8s, V8, V8s, dO8,
                                          dO8s, kTt, Lse.view(B * H, N),
                                          H, N, D, causal=ctx.causal, Hk=Hk)

        def bshd(x, h=H):
            return x.view(B, h, N, D).permute(0, 2, 1, 3)

        dq, dk, dv = ops.bwd_fp8(
            bshd(dO8), bshd(Q8), bshd(K8, Hk), bshd(V8, Hk),
            Lse, Dlt.view(B, H, N), kTt.view(B, Hk, D, N),
            Q8s.view(B, H, -1), K8s.view(B, Hk, -1), V8s.view(B, Hk, -1),
            dO8s.view(B, H, -1), sm, ctx.causal)
        # ops return (B, N, H, D) contiguous; hand back (B, H, N, D) views.
        dq = dq.permute(0, 2, 1, 3)
        dk = dk.permute(0, 2, 1, 3)
        dv = dv.permute(0, 2, 1, 3)
        if Hk < H:
            # ## M16 native GQA: dk/dv come back per QUERY head; fold to KV
            # heads exactly as the fused-producer orchestrator does (the aten
            # repeat_interleave backward realization).
            rep = H // Hk
            dk = dk.reshape(B, Hk, rep, N, D).sum(2)
            dv = dv.reshape(B, Hk, rep, N, D).sum(2)
        return (dq, dk, dv, None, None)


def dm_attn_func_plain(q, k, v, causal=True, softmax_scale=None,
                      delta_mode="stale"):
    """The plain route: the numerics of the API route (attn.dm_attn_func_bhnd, bitwise) through the quant.py / doq.py
    producers and the bwd_fp8 op, which makes the Q / dO twins itself (twin_pi).  q (B, H, N, D), k / v (B, Hk, N, D) bf16
    on CUDA (any strides); returns O (B, N, H, D) bf16 contiguous.  delta_mode: "stale" (delta = rowsum(dO * O)) or
    "matched" (the in-kernel matched delta; needs causal=True)."""
    _check_inputs(q, k, v, causal, softmax_scale, delta_mode)
    return _DMAttnPlain.apply(q, k, v, causal, delta_mode)
