"""The API route: dm_attn_func_bhnd(q, k, v, causal, softmax_scale, delta_mode) on (B, H, N, D) inputs, returning
(B, N, H, D).  Forward: producers.quant_qkv_m17 (codes, scales, the Q and K twins), then the fwd_fp8 op.  Backward:
producers.doq_m17 (dO codes, scales, twin and, in the stale mode, delta = rowsum(dO * O)); in the matched mode the delta
from pass 0 of the kernel (delta_fp8); then the bwd_fp8_t op on the global-memory twins, split per head group at long
context (split.py).  Saved for the backward: the fp8 codes, their scales, the twins, O and the LSE.
"""
from __future__ import annotations

import torch

from .core import _ops, _matched_delta_inkernel, blocks, _check_hd, _check_inputs
from .split import _bwd_heads_per_call, _bwd_by_head_groups
from .producers import quant_qkv_m17, doq_m17

_GT_CHECKED = False


def _ops_gt():
    global _GT_CHECKED
    ops = _ops()
    if not _GT_CHECKED:
        if int(ops.build_gt()) != 1:
            raise RuntimeError("the delta_matching library lacks the global-memory twins (build_gt() != 1): rebuild the package")
        _GT_CHECKED = True
    return ops


def _refuse_bf16dp(ops):
    """dm_attn_func implements the standard numerics only: a DM_BF16_DP=1 library (build_bf16dp() == 1) serves the bf16-dP
    baseline through delta-matching-train's dm_port route (the bwd_fp8_bf16dp op) and is refused here."""
    if int(ops.build_bf16dp()) == 1:
        raise RuntimeError("this delta_matching library is the bf16-dP variant (built with DM_BF16_DP=1): dm_attn_func implements "
                           "the standard numerics only; the bf16-dP mode is reached through delta-matching-train's dm_port "
                           "route (the bwd_fp8_bf16dp op)")


class _DMAttn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, delta_mode):
        with torch.cuda.device(q.device):   # the Triton producers launch on the current device
            return _DMAttn._forward(ctx, q, k, v, causal, delta_mode)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dO):
        with torch.cuda.device(dO.device):
            return _DMAttn._backward(ctx, dO)

    @staticmethod
    def _forward(ctx, q, k, v, causal, delta_mode):
        B, H, N, D = q.shape
        Hk = k.shape[1]
        ops = _ops_gt()
        _refuse_bf16dp(ops)   # the API route implements the standard numerics only
        _check_hd(D, ops)   # ## M19 Delta-Matching-128: hd128 needs the dual binary
        sm = D ** -0.5
        # the producers: quant.py's per-block quantizer math (bitwise), the split k-mean, the Q and K twins.
        # ## M19: the block-scale contract per head dim ((64, 32) at d256 = the M17 call; (128, 64) at hd128).
        Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_m17(q, k, v, want_qt=True, blocks=blocks(D))

        def bshd(x, h=H):
            return x.view(B, h, N, D).permute(0, 2, 1, 3)

        O, Lse = ops.fwd_fp8(bshd(Q8), bshd(K8, Hk), bshd(V8, Hk),
                             Q8s.view(B, H, -1), K8s.view(B, Hk, -1),
                             V8s.view(B, Hk, -1), sm, causal)
        ctx.save_for_backward(Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt, O, Lse)
        ctx.dims = (B, H, N, D)
        ctx.hk = Hk
        ctx.causal = causal
        ctx.delta_mode = delta_mode
        return O

    @staticmethod
    def _backward(ctx, dO):
        B, H, N, D = ctx.dims
        Hk = ctx.hk
        ops = _ops_gt()
        sm = D ** -0.5
        Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt, O, Lse = ctx.saved_tensors
        if dO.stride(-1) != 1:          # e.g. the expanded gradient of o.sum(): the producer reads rows with a unit lane stride
            dO = dO.contiguous()
        # dO read in its native (B, N, H, D) layout (no permute copy); the stale delta only when needed (the matched mode
        # never reads O); the dO twin alongside.
        want_delta = ctx.delta_mode != "matched"
        dO8, dO8s, Dlt, dOt = doq_m17(dO, O if want_delta else None, H, blocks(D)[0], want_delta=want_delta, want_twin=True)
        # The backward op materializes its dS' tiles over [b*h, s_q, s_k] (64 GiB at 64k for 16 heads), so the call is split
        # per KV group / per query head when that workspace exceeds DM_BWD_WS_LIMIT_GB (split.py); a split call takes the
        # matched delta per group too (bitwise the all-heads delta), so the limit bounds pass 0's workspace as well.
        n_rep = H // Hk
        hp = _bwd_heads_per_call(B, H, Hk, n_rep, N)
        per_group_delta = ctx.delta_mode == "matched" and hp < H
        if ctx.delta_mode == "matched" and not per_group_delta:
            Dlt = _matched_delta_inkernel(Q8, Q8s, K8, K8s, V8, V8s, dO8, dO8s, kTt,
                                          Lse.view(B * H, N), H, N, D, causal=ctx.causal, Hk=Hk)

        def bshd(x, h=H):
            return x.view(B, h, N, D).permute(0, 2, 1, 3)

        bwd_args = (bshd(dO8), bshd(Q8), bshd(K8, Hk), bshd(V8, Hk),
                    Lse, None if per_group_delta else Dlt.view(B, H, N), kTt.view(B, Hk, D, N),
                    Qt.view(B, H, D, N), dOt.view(B, H, D, N),
                    Q8s.view(B, H, -1), K8s.view(B, Hk, -1), V8s.view(B, Hk, -1),
                    dO8s.view(B, H, -1))
        if hp >= H:
            dq, dk, dv = ops.bwd_fp8_t(*bwd_args, sm, ctx.causal)
        else:
            dq, dk, dv = _bwd_by_head_groups(ops, bwd_args, sm, B, H, Hk, n_rep, N, D, hp, matched_delta=per_group_delta,
                                             causal=ctx.causal)
        dq = dq.permute(0, 2, 1, 3)
        dk = dk.permute(0, 2, 1, 3)
        dv = dv.permute(0, 2, 1, 3)
        if Hk < H:
            rep = H // Hk
            dk = dk.reshape(B, Hk, rep, N, D).sum(2)
            dv = dv.reshape(B, Hk, rep, N, D).sum(2)
        return (dq, dk, dv, None, None)


def dm_attn_func_bhnd(q, k, v, causal=True, softmax_scale=None, delta_mode="stale"):
    """Attention on (B, H, N, D) inputs: q (B, H, N, D), k / v (B, Hk, N, D) bf16 on CUDA (any strides; H % Hk == 0) ->
    O (B, N, H, D) bf16 contiguous (the kernel's native output layout).  delta_mode: "stale" (delta = rowsum(dO * O)) or
    "matched" (the Delta-Matching backward; needs causal=True).  Input errors raise ValueError / TypeError /
    NotImplementedError (core._check_inputs)."""
    _check_inputs(q, k, v, causal, softmax_scale, delta_mode)
    return _DMAttn.apply(q, k, v, causal, delta_mode)
