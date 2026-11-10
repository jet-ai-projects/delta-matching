"""sm100 (GB200) hd128 Delta-Matching attention, sm100-mx numerics: dm_attn_func(q, k, v, causal, mode).

forward : quant_qkv2 (fused producers) -> dm_fwd (C1: fp8 QK, MXFP8 PV)
backward: dO quant -> delta (matched: pass-0 kernel dm_delta, sum P dP / sum P; stale: rowsum(dO * O))
          -> dm_bwd_main (B1: 2-CTA, MXFP8 dV / dK / dQ, deterministic)
q (B, N, H, 128) bf16, k / v (B, N, Hk, 128) bf16 -> o (B, N, H, 128) bf16.  N % 128 == 0, softmax scale 128^-0.5.
"""
import os

import torch

from .quant import quant_qkv2, quant_do_delta
from .fwd import dm_fwd
from .bwd import dm_bwd_main, dm_bwd_prestage
from .pass0 import dm_delta
from .fastpath import env as _env

D = 128


def _det():
    """DM_BWD_DET=0 or DM_DETERMINISTIC=0: non-deterministic backward (FA4-style unordered dQ / GQA dK,dV accumulation). Default 1."""
    return _env("DM_BWD_DET", _env("DM_DETERMINISTIC", "1")) != "0"


class _DMAttnSm100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, mode, det):
        sm = D ** -0.5
        q8, qs, k8, ks, v8, vs, kvw, _ = quant_qkv2(q, k, v, sm)
        o, lse = dm_fwd(q8, qs, k8, kvw, v8, None, causal)
        ctx.save_for_backward(q8, qs, k8, ks, v8, vs, o, lse)
        ctx.causal, ctx.mode, ctx.det = causal, mode, det
        return o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do):
        q8, qs, k8, ks, v8, vs, o, lse = ctx.saved_tensors
        do = do.contiguous()
        dm_bwd_prestage(q8, k8, lse, ctx.causal, deterministic=ctx.det)    # scratch alloc / zero / lse_log2 on a side stream (overlaps pass 0)
        do8, dos, delta = quant_do_delta(do, o if ctx.mode == "stale" else None)   # stale delta fused
        if ctx.mode == "matched":
            delta = dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, ctx.causal)
        dq, dk, dv = dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, ctx.causal, D ** -0.5, deterministic=ctx.det)
        return dq, dk, dv, None, None, None


def _check_nd(det):
    # hook for gating the non-deterministic backward (currently allowed; the default and the validated mode is deterministic)
    return det


def dm_attn_func(q, k, v, causal=True, mode="matched", deterministic=None):
    """deterministic: None -> env DM_DETERMINISTIC / DM_BWD_DET (default deterministic); False -> non-deterministic
    backward (unordered dQ / GQA dK,dV accumulation). Either way delta and dS use bitwise the same P/dP."""
    assert q.shape[-1] == D and q.dtype == torch.bfloat16 and q.shape[1] % 128 == 0
    assert mode in ("matched", "stale")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    det = _check_nd(_det() if deterministic is None else bool(deterministic))
    return _DMAttnSm100.apply(q, k, v, causal, mode, det)


@torch.no_grad()
def dm_attn_debug(q, k, v, do, causal=True, mode="matched", identity=None, deterministic=None):
    """Validator hook: the exact delta the backward consumes and the LSE it recomputes P from, for these inputs
    (same producers, forward kernel and delta path as dm_attn_func's autograd backward).  (B,H,N) fp32 each, LSE natural.
    identity (opt-in: identity=True or env DM_DEBUG_IDENTITY=1; matched mode): also run the backward main kernel in its
    debug build (DM_BWD_OPTS += rsdbg, same arithmetic) and return, accumulated IN-KERNEL from the backward's own tiles
    (f64 atomics, before dS quantization):
      "rowsum_dS":   (B,H,N) fp32  sum_j dS_ij,  dS_ij = P_ij * fma(dPacc_ij, a, -delta_i)  exactly as the backward
      "sum_abs_PdP": (B,H,N) fp32  sum_j |P_ij * (dPacc_ij * a)|
    The production path (dm_attn_func) is unchanged."""
    sm = D ** -0.5
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    q8, qs, k8, ks, v8, vs, kvw, _ = quant_qkv2(q, k, v, sm)
    o, lse = dm_fwd(q8, qs, k8, kvw, v8, None, causal)
    do8, dos, delta = quant_do_delta(do.contiguous(), o if mode == "stale" else None)
    if mode == "matched":
        delta = dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal)
    out = {"delta": delta.float().contiguous(), "lse": lse.float().contiguous()}
    if identity is None:
        identity = _env("DM_DEBUG_IDENTITY", "0") == "1"
    if identity and mode == "matched":
        B, N, H, _ = q8.shape
        rs = torch.zeros(B, H, N, 2, device=q8.device, dtype=torch.float64)
        old = _env("DM_BWD_OPTS", "")
        os.environ["DM_BWD_OPTS"] = ",".join([o for o in old.split(",") if o] + ["rsdbg"])
        try:
            dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, sm, trace=rs,
                        deterministic=_check_nd(_det() if deterministic is None else bool(deterministic)))
        finally:
            os.environ["DM_BWD_OPTS"] = old
        out["rowsum_dS"] = rs[..., 0].float().contiguous()
        out["sum_abs_PdP"] = rs[..., 1].float().contiguous()
    return out
