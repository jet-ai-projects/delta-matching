"""CuTe DSL producers / consumers around the head_dim-256 attention core, each BITWISE the op it replaces.

  gqa_fold(dk, dv, n_rep)                 the GQA dK / dV fold of the orchestrator, both tensors in ONE launch:
                                          (B, N, H, D) bf16 per query head -> (B, Hkv, N, D) bf16 dense, the exact aten
                                          `x.permute(0, 2, 1, 3).reshape(B, Hkv, n_rep, N, D).sum(2)`: fp32 accumulate
                                          (((0 + a0) + (0 + a1)) + (0 + a2)) + (0 + a3) (reduce_kernel's four
                                          accumulators, combined in order), bf16 RTNE.
  fold_rope_bwd(dk, dv, dq, cos, sin, n_rep)
                                          the fold above and the q-RoPE backward (m13_rope_bwd.rope_bwd_q) in ONE launch.

Design (Transformer Engine's cast/transpose kernels): memory-bound streaming, every global access a 16 B vector, one warp
per 512 B row segment so every load / store instruction of a warp is one contiguous 512 B burst; the RoPE partner lanes
come from a warp shuffle instead of a second load.  Explicitly rounded mul.rn / add.rn (never FMA-contracted), as the
Triton reference's libdevice mul_rn / add_rn.
"""
from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch

F32 = cutlass.Float32
BF16 = cutlass.BFloat16
_V = 8            # bf16 per 16 B vector
_CACHE = {}


def _ct(x):
    return from_dlpack(x.detach(), assumed_align=16)


def _mul_rn(a, b):
    return cute.arch.inline_ptx("mul.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


def _add_rn(a, b):
    return cute.arch.inline_ptx("add.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


def _add0_rn(a):
    """a + 0.0 (RN): normalizes -0 to +0, as torch's add of a zeros tensor."""
    return cute.arch.inline_ptx("add.rn.f32 {$w0}, {$r0}, 0f00000000;", write_only_types=[F32], read_only_args=[a])


def _fold_row(X, O, b, n, hk, c, R: cutlass.Constexpr):
    """O[b, hk, n, c, :] = bf16(sum_r (0 + X[b, n, hk, r, c, :])) in aten's order."""
    a0 = X[b, n, hk, 0, c, None].load().to(F32)
    z = cute.full_like(a0, 0.0)
    acc = z + a0
    for r in range(1, R):
        acc = acc + (z + X[b, n, hk, r, c, None].load().to(F32))
    O[b, hk, n, c, None].store(acc.to(BF16))


@cute.kernel
def _fold_kernel(XK: cute.Tensor, XV: cute.Tensor, OK: cute.Tensor, OV: cute.Tensor,
                 HKV: cutlass.Constexpr, R: cutlass.Constexpr, NC: cutlass.Constexpr, TOK: cutlass.Constexpr):
    t, _, _ = cute.arch.thread_idx()
    bx, b, _ = cute.arch.block_idx()
    c = t % NC
    hk = (t // NC) % HKV
    if t < NC * HKV:
        for i in cutlass.range_constexpr(TOK):
            _fold_row(XK, OK, b, bx * TOK + i, hk, c, R)
    else:
        for i in cutlass.range_constexpr(TOK):
            _fold_row(XV, OV, b, bx * TOK + i, hk, c, R)


@cute.jit
def _fold(XK, XV, OK, OV, B: cutlass.Constexpr, N: cutlass.Constexpr, HKV: cutlass.Constexpr, R: cutlass.Constexpr,
          NC: cutlass.Constexpr, TOK: cutlass.Constexpr, stream):
    _fold_kernel(XK, XV, OK, OV, HKV, R, NC, TOK).launch(grid=[N // TOK, B, 1], block=[2 * HKV * NC, 1, 1],
                                                         stream=stream)


def _fold_ok(dk, dv, n_rep):
    if dk.dtype != torch.bfloat16 or dv.dtype != torch.bfloat16 or n_rep < 1 or n_rep > 4:
        return False
    if not (dk.is_contiguous() and dv.is_contiguous()) or dk.shape != dv.shape or dk.dim() != 4:
        return False
    B, N, H, D = dk.shape
    return H % n_rep == 0 and D % _V == 0 and dk.data_ptr() % 16 == 0 and dv.data_ptr() % 16 == 0


def _fold_ref(dk, dv, n_rep):
    B, N, H, D = dk.shape
    Hkv = H // n_rep
    return (dk.permute(0, 2, 1, 3).reshape(B, Hkv, n_rep, N, D).sum(2),
            dv.permute(0, 2, 1, 3).reshape(B, Hkv, n_rep, N, D).sum(2))


def _pick_tok(N):
    for t in (4, 2, 1):
        if N % t == 0:
            return t
    return 1


def gqa_fold(dk: torch.Tensor, dv: torch.Tensor, n_rep: int):
    """dk, dv: bf16 (B, N, H, D) contiguous (the kernel's per-query-head gradients) -> (B, Hkv, N, D) bf16 dense each,
    bitwise the orchestrator's permute + reshape + sum(2).  Falls back to that torch expression off the fast path."""
    if not _fold_ok(dk, dv, n_rep):
        return _fold_ref(dk, dv, n_rep)
    B, N, H, D = dk.shape
    Hkv = H // n_rep
    NC = D // _V
    if 2 * Hkv * NC > 1024:
        return _fold_ref(dk, dv, n_rep)
    TOK = _pick_tok(N)
    ok = torch.empty(B, Hkv, N, D, device=dk.device, dtype=torch.bfloat16)
    ov = torch.empty_like(ok)
    args = [_ct(dk.view(B, N, Hkv, n_rep, NC, _V)), _ct(dv.view(B, N, Hkv, n_rep, NC, _V)),
            _ct(ok.view(B, Hkv, N, NC, _V)), _ct(ov.view(B, Hkv, N, NC, _V))]
    key = ("fold", dk.device.index, B, N, H, D, n_rep)   # contiguous inputs only (_fold_ok): strides fixed by the shape
    st = cutlass_torch.current_stream()
    if key not in _CACHE:
        _CACHE[key] = cute.compile(_fold, *args, B, N, Hkv, n_rep, NC, TOK, st)
    _CACHE[key](*args, st)
    return ok, ov


# ------------------------------------------------------------------ fold + q-RoPE backward in one launch
def _rope_row(DQ, COS, SIN, b, n, h, lane, HALFC, RDC):
    """One (b, n, h) row of rope_bwd_q (straight-line; the caller stores): lane = 16 B chunk of D.  Rotated chunks
    (lane < RDC) get the partner half's dq by a butterfly shuffle (lane ^ HALFC).  Per element:
      lane < HALFC:  (0 + g[j+HALF] sin[j+HALF]) + g[j] cos[j]
      lane >= HALFC: (-(g[j-HALF] sin[j-HALF]) + 0) + g[j] cos[j]
    written as add_rn(add_rn(sgn * t, 0), gc) with sgn = +-1 (an exact sign flip; IEEE add commutes bitwise)."""
    graw = DQ[b, n, h, lane, None].load()
    g = graw.to(F32)
    lr = lane % RDC
    cc = COS[b, n, lr, None].load().to(F32)
    ss = SIN[b, n, lr ^ HALFC, None].load().to(F32)
    sgn = F32(1.0) - F32(2.0) * F32((lr // HALFC) % 2)
    out = cute.make_rmem_tensor((_V,), F32)
    for j in range(_V):
        gp = cute.arch.shuffle_sync_bfly(g[j], offset=HALFC)
        t = _mul_rn(_mul_rn(gp, ss[j]), sgn)
        out[j] = _add_rn(_add0_rn(t), _mul_rn(g[j], cc[j]))
    return graw, out.load().to(BF16)


@cute.kernel
def _fold_rope_kernel(XK: cute.Tensor, XV: cute.Tensor, OK: cute.Tensor, OV: cute.Tensor, DQ: cute.Tensor,
                      DQO: cute.Tensor, COS: cute.Tensor, SIN: cute.Tensor,
                      HKV: cutlass.Constexpr, R: cutlass.Constexpr, H: cutlass.Constexpr, TOK: cutlass.Constexpr,
                      HALFC: cutlass.Constexpr, RDC: cutlass.Constexpr):
    """NC = 32 (D = 256): one warp per 512 B row segment.  Warps 0..HKV-1 fold dK, HKV..2HKV-1 fold dV; every warp then
    runs H / (2 HKV) dq rows."""
    t, _, _ = cute.arch.thread_idx()
    bx, b, _ = cute.arch.block_idx()
    lane = t % 32
    w = t // 32
    hk = w % HKV
    HPW = H // (2 * HKV)
    for i in cutlass.range_constexpr(TOK):
        n = bx * TOK + i
        if w < HKV:
            _fold_row(XK, OK, b, n, hk, lane, R)
        else:
            _fold_row(XV, OV, b, n, hk, lane, R)
        for k in cutlass.range_constexpr(HPW):
            h = w * HPW + k
            graw, rot = _rope_row(DQ, COS, SIN, b, n, h, lane, HALFC, RDC)
            if lane < RDC:
                DQO[b, h, n, lane, None].store(rot)
            else:
                DQO[b, h, n, lane, None].store(graw)


@cute.jit
def _fold_rope(XK, XV, OK, OV, DQ, DQO, COS, SIN, B: cutlass.Constexpr, N: cutlass.Constexpr, HKV: cutlass.Constexpr,
               R: cutlass.Constexpr, H: cutlass.Constexpr, TOK: cutlass.Constexpr, HALFC: cutlass.Constexpr,
               RDC: cutlass.Constexpr, stream):
    _fold_rope_kernel(XK, XV, OK, OV, DQ, DQO, COS, SIN, HKV, R, H, TOK, HALFC, RDC).launch(
        grid=[N // TOK, B, 1], block=[2 * HKV * 32, 1, 1], stream=stream)


def fold_rope_bwd(dk, dv, dq, cos, sin, n_rep: int, tok: int | None = None):
    """dk, dv, dq: bf16 (B, N, H, D) contiguous; cos / sin: bf16 (B, N, RD) contiguous ->
    (dq_pre (B, H, N, D) dense = rope_bwd_q(dq, cos, sin, H), dk (B, Hkv, N, D), dv (B, Hkv, N, D)), one launch.
    Returns None off the fast path (the caller runs the separate ops)."""
    if not _fold_ok(dk, dv, n_rep) or dq.dtype != torch.bfloat16 or not dq.is_contiguous():
        return None
    B, N, H, D = dk.shape
    Hkv = H // n_rep
    RD = cos.shape[-1]
    if (D != 256 or dq.shape != dk.shape or cos.shape != (B, N, RD) or sin.shape != cos.shape or not cos.is_contiguous()
            or not sin.is_contiguous() or cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16 or RD % 16 != 0
            or RD > D or ((RD // 16) & (RD // 16 - 1)) != 0 or H % (2 * Hkv) != 0 or dq.data_ptr() % 16 != 0
            or cos.data_ptr() % 16 != 0 or sin.data_ptr() % 16 != 0):
        return None
    NC = D // _V
    TOK = tok or _pick_tok(N)
    ok = torch.empty(B, Hkv, N, D, device=dk.device, dtype=torch.bfloat16)
    ov = torch.empty_like(ok)
    dqo = torch.empty(B, H, N, D, device=dk.device, dtype=torch.bfloat16)
    args = [_ct(dk.view(B, N, Hkv, n_rep, NC, _V)), _ct(dv.view(B, N, Hkv, n_rep, NC, _V)),
            _ct(ok.view(B, Hkv, N, NC, _V)), _ct(ov.view(B, Hkv, N, NC, _V)),
            _ct(dq.view(B, N, H, NC, _V)), _ct(dqo.view(B, H, N, NC, _V)),
            _ct(cos.view(B, N, RD // _V, _V)), _ct(sin.view(B, N, RD // _V, _V))]
    key = ("fold_rope", dk.device.index, B, N, H, D, n_rep, RD, TOK)   # contiguous inputs only: strides fixed by the shape
    st = cutlass_torch.current_stream()
    if key not in _CACHE:
        _CACHE[key] = cute.compile(_fold_rope, *args, B, N, Hkv, n_rep, H, TOK, RD // 16, RD // _V, st)
    _CACHE[key](*args, st)
    return dqo, ok, ov
