"""
Delta-Matching attention at head_dim 128 on sm100: the autograd glue around the
producers, the forward, pass 0 and the backward.
"""
import os

import torch

from delta_matching.sm100.bwd import dm_bwd_main, dm_bwd_prestage
from delta_matching.sm100.fwd import dm_fwd
from delta_matching.sm100.launch_cache_sm100 import env
from delta_matching.sm100.pass0 import dm_delta
from delta_matching.sm100.quant import quant_do_delta, quant_qkv2

__all__ = ["AttentionHd128", "attention_hd128", "attention_debug_hd128"]

D = 128
SOFTMAX_SCALE = D ** -0.5


def deterministic_default():
    """
    DM_DETERMINISTIC=0 selects the unordered dQ / dK / dV accumulation; the
    default is deterministic.
    """
    return env("DM_DETERMINISTIC", "1") != "0"


class AttentionHd128(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, mode, deterministic):
        q8, qs, k8, ks, v8, vs, kv_words, _ = quant_qkv2(q, k, v, SOFTMAX_SCALE)
        o, lse = dm_fwd(q8, qs, k8, kv_words, v8, None, causal)
        ctx.save_for_backward(q8, qs, k8, ks, v8, vs, o, lse)
        ctx.causal, ctx.mode, ctx.deterministic = causal, mode, deterministic
        return o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do):
        q8, qs, k8, ks, v8, vs, o, lse = ctx.saved_tensors
        do = do.contiguous()
        # scratch allocation, zeroing and lse * log2(e) run on a side stream, overlapping pass 0
        dm_bwd_prestage(q8, k8, lse, ctx.causal, deterministic=ctx.deterministic)
        do8, dos, delta = quant_do_delta(do, o if ctx.mode == "stale" else None)
        if ctx.mode == "matched":
            delta = dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, ctx.causal)
        dq, dk, dv = dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, ctx.causal, SOFTMAX_SCALE, deterministic=ctx.deterministic)
        return dq, dk, dv, None, None, None


def attention_hd128(q, k, v, causal=True, mode="matched", deterministic=None):
    """
    dm_attn_func at head_dim 128.
    """
    assert q.shape[-1] == D and q.dtype == torch.bfloat16 and q.shape[1] % 128 == 0
    assert mode in ("matched", "stale")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    deterministic = deterministic_default() if deterministic is None else bool(deterministic)
    return AttentionHd128.apply(q, k, v, causal, mode, deterministic)


@torch.no_grad()
def attention_debug_hd128(q, k, v, do, causal=True, mode="matched", identity=False, deterministic=None):
    """
    dm_attn_debug at head_dim 128.
    """
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    q8, qs, k8, ks, v8, vs, kv_words, _ = quant_qkv2(q, k, v, SOFTMAX_SCALE)
    o, lse = dm_fwd(q8, qs, k8, kv_words, v8, None, causal)
    do8, dos, delta = quant_do_delta(do.contiguous(), o if mode == "stale" else None)
    if mode == "matched":
        delta = dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal)
    out = {"delta": delta.float().contiguous(), "lse": lse.float().contiguous()}
    if identity and mode == "matched":
        B, N, H, _ = q8.shape
        sums = torch.zeros(B, H, N, 2, device=q8.device, dtype=torch.float64)
        # the identity build of the backward (same arithmetic) accumulates rowsum(dS) and sum |P dP| into sums
        opts = env("DM_BWD_OPTS", "")
        os.environ["DM_BWD_OPTS"] = ",".join([o for o in opts.split(",") if o] + ["rsdbg"])
        try:
            deterministic = deterministic_default() if deterministic is None else bool(deterministic)
            dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, SOFTMAX_SCALE, trace=sums, deterministic=deterministic)
        finally:
            os.environ["DM_BWD_OPTS"] = opts
        out["rowsum_dS"] = sums[..., 0].float().contiguous()
        out["sum_abs_PdP"] = sums[..., 1].float().contiguous()
    return out
