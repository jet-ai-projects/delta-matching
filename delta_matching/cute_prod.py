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
    return x.detach()


_TVM = __import__("os").environ.get("DM_CUTE_TVM_FFI", "1") == "1"


def _run(key, fn, targs, consts, st):
    """Compile once per key (static layouts from the torch views), then call.  TVM-FFI ABI: the compiled function takes the
    torch tensors directly (~10 us host per call instead of ~100 us of from_dlpack + the default call path)."""
    f = _CACHE.get(key)
    if f is None:
        cargs = [from_dlpack(x, assumed_align=16, enable_tvm_ffi=_TVM) for x in targs]
        f = cute.compile(fn, *cargs, *consts, st, options="--enable-tvm-ffi") if _TVM else cute.compile(fn, *cargs, *consts, st)
        _CACHE[key] = (f, _TVM)
    else:
        f = f[0]
    if _TVM:
        f(*targs, st)
    else:
        f(*[from_dlpack(x, assumed_align=16) for x in targs], st)


def _mul_rn(a, b):
    return cute.arch.inline_ptx("mul.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


def _add_rn(a, b):
    return cute.arch.inline_ptx("add.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


def _f32hex(c):
    import struct
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", c))[0]


def _mul_rn_c(a, c):
    """a * c (RN) with c a Python float, rounded to fp32, as a PTX immediate."""
    return cute.arch.inline_ptx("mul.rn.f32 {$w0}, {$r0}, " + _f32hex(c) + ";", write_only_types=[F32],
                                read_only_args=[a])


def _add0_rn(a):
    """a + 0.0 (RN): normalizes -0 to +0, as torch's add of a zeros tensor."""
    return cute.arch.inline_ptx("add.rn.f32 {$w0}, {$r0}, 0f00000000;", write_only_types=[F32], read_only_args=[a])


# The Triton references' libdevice mul_rn / add_rn compile with nvvm-reflect-ftz=1: mul.rn.ftz.f32 / add.rn.ftz.f32
# (checked in their PTX); their plain `*` (the sm bake), div_rn and the e4m3 cast keep denormals.
def _mul_rnz(a, b):
    return cute.arch.inline_ptx("mul.rn.ftz.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


def _add_rnz(a, b):
    return cute.arch.inline_ptx("add.rn.ftz.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


def _add0_rnz(a):
    return cute.arch.inline_ptx("add.rn.ftz.f32 {$w0}, {$r0}, 0f00000000;", write_only_types=[F32], read_only_args=[a])


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
    key = ("fold", _TVM, dk.device.index, B, N, H, D, n_rep)   # contiguous inputs only (_fold_ok): strides fixed by the shape
    _run(key, _fold, args, (B, N, Hkv, n_rep, NC, TOK), cutlass_torch.current_stream())
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
        t = _mul_rn(_mul_rnz(gp, ss[j]), sgn)          # sgn: an exact sign flip (torch.neg)
        out[j] = _add_rnz(_add0_rnz(t), _mul_rnz(g[j], cc[j]))
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
    key = ("fold_rope", _TVM, dk.device.index, B, N, H, D, n_rep, RD, TOK)   # contiguous inputs only: strides fixed by the shape
    _run(key, _fold_rope, args, (B, N, Hkv, n_rep, H, TOK, RD // 16, RD // _V), cutlass_torch.current_stream())
    return dqo, ok, ov


# ------------------------------------------------------------------ Q / dO cast-transpose: codes + scales + pi twin, one read
# TE's quantize_transpose (block scaling) design: ONE read of the bf16 tile, the block amax reduced in registers (warp
# shuffle + a CTA smem step), then BOTH outputs written from registers: the row-major codes and the transposed (pi-ordered)
# twin.  The thread tile is 16 rows (one pi window) x 4 columns, so each twin segment (16 positions of one column) is
# assembled in registers and leaves as ONE 16 B store; no shared-memory transpose is needed.
_E4M3_MAX = 448.0
_INV_E4M3_MAX = float(__import__("numpy").float32(1.0) / __import__("numpy").float32(448.0))
_SIGMA = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]   # pi window: physical position p holds row SIGMA[p]


def _rcp_rn(a):
    return cute.arch.inline_ptx("rcp.rn.f32 {$w0}, {$r0};", write_only_types=[F32], read_only_args=[a])


def _fdiv(x, ns, y):
    """RN(x / s) from ns = -s and y = RN(1 / s): q0 = x y; q1 = q0 + (x - s q0) y; q2 = q1 + (x - s q1) y; copysign(q2, x)."""
    return cute.arch.inline_ptx(
        "{\n.reg .f32 q, e;\nmul.rn.f32 q, {$r0}, {$r2};\nfma.rn.f32 e, {$r1}, q, {$r0};\nfma.rn.f32 q, e, {$r2}, q;\n"
        "fma.rn.f32 e, {$r1}, q, {$r0};\nfma.rn.f32 q, e, {$r2}, q;\ncopysign.f32 {$w0}, {$r0}, q;\n}",
        write_only_types=[F32], read_only_args=[x, ns, y])


def _div_rn(a, b):
    return cute.arch.inline_ptx("div.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[F32], read_only_args=[a, b])


@cute.kernel
def _ct_kernel(X: cute.Tensor, COS: cute.Tensor, SIN: cute.Tensor, Q8: cute.Tensor, QS: cute.Tensor, QT: cute.Tensor,
               H: cutlass.Constexpr, BLK: cutlass.Constexpr, D: cutlass.Constexpr, RD: cutlass.Constexpr,
               ROPE: cutlass.Constexpr, SM: cutlass.Constexpr, WANT_T: cutlass.Constexpr,
               FASTDIV: cutlass.Constexpr, SHFL: cutlass.Constexpr):
    """One CTA per (64-row block, b*H + h), D threads; thread = 16 rows (one pi window) x 4 columns, lane = window * 8 +
    column group, so each warp store of the twin writes 8 x 64 B contiguous runs (4 windows of one column).
    ROPE (the Q producer, tier3a_ropeq): partial RoPE (mul_rn / add_rn, bf16 RTNE), x SM, s = div.full(max(amax, 1e-30),
    448).  Otherwise (the dO producer, doq_m17): x as is, s = max(amax, 1e-30) * fp32(1 / 448).
    Codes: div_rn(x, s), clamp +-448 (max / min), RTNE e4m3."""
    t, _, _ = cute.arch.thread_idx()
    pm, pbh, _ = cute.arch.block_idx()
    b = pbh // H
    h = pbh - b * H
    lane = t % 32
    w = t // 32
    NW = D // 32                            # warps: 8 column groups (32 columns) each
    k = lane % 8
    wr = lane // 8                          # pi window (16 rows) of the 64-row block
    cg = w * 8 + k                          # 4-column group, 0 .. D/4 - 1
    if cutlass.const_expr(ROPE and SHFL):
        # RoPE partners in the same warp: lane bit 2 drives cg bit log2(HALF / 4), so the partner is lane ^ 4 (a shuffle,
        # no second load); the other cg bits come from the warp index, so "rotated" is warp-uniform.
        cg = (k & 3) | ((k >> 2) << _log2(RD // 8))
        for i, pos in enumerate(_free_bits(D // 4, _log2(RD // 8))):
            cg = cg | (((w >> i) & 1) << pos)
    n0 = pm * BLK + wr * 16
    # the tile stays in registers as bf16 (32 regs): the RoPE output IS bf16, the sm bake (x SM, exact at D = 256) and the
    # fp32 upcast are recomputed where used (same op, same input, same bits), so occupancy is not register-bound
    vals = cute.make_rmem_tensor(cute.make_layout((16, 4), stride=(4, 1)), BF16)
    for r in cutlass.range_constexpr(16):
        vals[r, None].store(X[b, h, n0 + r, cg, None].load())
    if cutlass.const_expr(ROPE):
        HALFG = RD // 8                      # partner group offset (HALF / 4)
        RDG = RD // 4
        if cutlass.const_expr(SHFL):
            if cg < RDG:                     # warp-uniform: every lane of the warp is rotated
                sgn = F32(2.0) * F32((cg // HALFG) % 2) - F32(1.0)
                for hh in cutlass.range_constexpr(2):
                    cc = cute.make_rmem_tensor(cute.make_layout((8, 4), stride=(4, 1)), BF16)
                    ss = cute.make_rmem_tensor(cute.make_layout((8, 4), stride=(4, 1)), BF16)
                    for rr in cutlass.range_constexpr(8):
                        cc[rr, None].store(COS[b, n0 + hh * 8 + rr, cg, None].load())
                        ss[rr, None].store(SIN[b, n0 + hh * 8 + rr, cg, None].load())
                    for rr in cutlass.range_constexpr(8):
                        r = hh * 8 + rr
                        xo = vals[r, None].load().to(F32)
                        c4 = cc[rr, None].load().to(F32)
                        s4 = ss[rr, None].load().to(F32)
                        ov = cute.make_rmem_tensor((4,), F32)
                        for j in cutlass.range_constexpr(4):
                            xp = cute.arch.shuffle_sync_bfly(xo[j], offset=4)
                            ov[j] = _add_rnz(_mul_rnz(xo[j], c4[j]), _mul_rnz(_mul_rn(xp, sgn), s4[j]))
                        vals[r, None].store(ov.load().to(BF16))
        elif cg < RDG:                       # rotated lanes: own + partner (L1 hits), cos / sin of the own lanes
            sgn = F32(2.0) * F32((cg // HALFG) % 2) - F32(1.0)   # -1: lanes < HALF pair with -x2; +1: >= HALF with x1
            for r in cutlass.range_constexpr(16):
                xo = vals[r, None].load().to(F32)
                xp = X[b, h, n0 + r, cg ^ HALFG, None].load().to(F32)
                cc = COS[b, n0 + r, cg, None].load().to(F32)
                ss = SIN[b, n0 + r, cg, None].load().to(F32)
                ov = cute.make_rmem_tensor((4,), F32)
                for j in cutlass.range_constexpr(4):
                    ov[j] = _add_rnz(_mul_rnz(xo[j], cc[j]), _mul_rnz(_mul_rn(xp[j], sgn), ss[j]))
                vals[r, None].store(ov.load().to(BF16))
    # block amax: max |RN(x SM)| = RN(max |x| SM) (RN is monotone), so the bake is applied once to the max
    a = F32(0.0)
    for r in cutlass.range_constexpr(16):
        xr = vals[r, None].load().to(F32)
        for j in cutlass.range_constexpr(4):
            a = cute.arch.fmax(a, cute.math.absf(xr[j]))
    if cutlass.const_expr(ROPE):
        a = _mul_rn_c(a, SM)
    smem = cute.make_tensor(cute.arch.alloc_smem(F32, NW), cute.make_layout(NW))
    wm = cute.arch.warp_reduction_max(a)
    if lane == 0:
        smem[w] = wm
    cute.arch.barrier()
    m = smem[0]
    for i in cutlass.range_constexpr(1, NW):
        m = cute.arch.fmax(m, smem[i])
    s = F32(0.0)
    if cutlass.const_expr(ROPE):
        s = cute.math.div(cute.arch.fmax(m, F32(1e-30)), F32(_E4M3_MAX), full=True)
    else:
        s = _mul_rn_c(cute.arch.fmax(m, F32(1e-30)), _INV_E4M3_MAX)
    if t == 0:
        QS[pbh, pm] = s
    codes = cute.make_rmem_tensor(cute.make_layout((4, 16), stride=(16, 1)), cutlass.Float8E4M3FN)  # [col j, pi pos p]
    # fast division only for finite blocks with amax >= 2^-80; an Inf / NaN block (s = Inf: x / Inf, Inf / Inf) and tiny
    # blocks take div.rn, so every code (incl. the clamp of NaN quotients to -448) is the reference's
    fast = (m >= F32(2.0 ** -80)) & (m <= F32(3.4028234663852886e38))
    y = _rcp_rn(s)
    ns = -s
    q8 = cute.recast_tensor(Q8, cutlass.Float8E4M3FN)
    for p in cutlass.range_constexpr(16):    # row by row (pi order): 4 codes -> the row store now, the twin bytes kept
        r = _SIGMA[p]
        qc = cute.make_rmem_tensor((4,), F32)
        for j in cutlass.range_constexpr(4):
            x = vals[r, j].to(F32)
            if cutlass.const_expr(ROPE):
                x = _mul_rn_c(x, SM)
            if cutlass.const_expr(FASTDIV):
                # x / s correctly rounded from y = RN(1 / s): q = x y, two fma corrections (the CUDA div.rn fast path;
                # Markstein: y within 1/2 ulp and q within 1 ulp -> RN(x / s)), sign copied from x (zeros, underflow).
                # Exact while every remainder that can move a code is normal: blocks with amax >= 2^-80 (a code needs
                # |x / s| >= 2^-11, so |x| >= 2^-99 there); smaller blocks take div.rn.
                qc[j] = _fdiv(x, ns, y)
            else:
                qc[j] = _div_rn(x, s)
        if cutlass.const_expr(FASTDIV):
            if not fast:
                for j in cutlass.range_constexpr(4):
                    x = vals[r, j].to(F32)
                    if cutlass.const_expr(ROPE):
                        x = _mul_rn_c(x, SM)
                    qc[j] = _div_rn(x, s)
        for j in cutlass.range_constexpr(4):
            qc[j] = cute.arch.fmin(cute.arch.fmax(qc[j], F32(-_E4M3_MAX)), F32(_E4M3_MAX))
        codes[None, p].store(qc.load().to(cutlass.Float8E4M3FN))
        q8[pbh, n0 + r, cg, None].store(codes[None, p].load())
    if cutlass.const_expr(WANT_T):
        qt = cute.recast_tensor(QT, cutlass.Float8E4M3FN)
        for j in cutlass.range_constexpr(4):
            qt[pbh, cg * 4 + j, n0 // 16, None].store(codes[j, None].load())


@cute.jit
def _ct_host(X, COS, SIN, Q8, QS, QT, NB: cutlass.Constexpr, BH: cutlass.Constexpr, H: cutlass.Constexpr,
        BLK: cutlass.Constexpr, D: cutlass.Constexpr, RD: cutlass.Constexpr, ROPE: cutlass.Constexpr,
        SM: cutlass.Constexpr, WANT_T: cutlass.Constexpr, FASTDIV: cutlass.Constexpr, SHFL: cutlass.Constexpr,
        MINB: cutlass.Constexpr, stream):
    _ct_kernel(X, COS, SIN, Q8, QS, QT, H, BLK, D, RD, ROPE, SM, WANT_T, FASTDIV, SHFL).launch(
        grid=[NB, BH, 1], block=[D, 1, 1], stream=stream, min_blocks_per_mp=MINB)


_FASTDIV = __import__("os").environ.get("DM_CUTE_FASTDIV", "1") == "1"
_SHFL = __import__("os").environ.get("DM_CUTE_CT_SHFL", "1") == "1"


def _log2(n):
    assert n > 0 and n & (n - 1) == 0, n
    return n.bit_length() - 1


def _free_bits(ncg, pb):
    """cg bit positions filled from the warp index: all of [0, log2(ncg)) except 0, 1 (lane bits) and pb (lane bit 2)."""
    return [i for i in range(_log2(ncg)) if i not in (0, 1, pb)]


_DUMMY = {}


def _dummy(dev):
    t = _DUMMY.get(dev)
    if t is None:
        t = _DUMMY[dev] = torch.zeros(1, 1, 1, 4, device=dev, dtype=torch.bfloat16)
    return t


def _ct_launch(x_bhnd, cos, sin, blk, rope, want_twin, fastdiv=None, minb=None):
    # measured on H200 (B2 H16 N8192, same-process alternation): Q 0.104 ms with fast division + min 2 CTAs / SM;
    # dO 0.077 ms with div.rn and no occupancy hint (the fast division costs it registers)
    fastdiv = (_FASTDIV and rope) if fastdiv is None else fastdiv
    minb = (2 if rope else 0) if minb is None else minb
    D = x_bhnd.shape[-1]
    RDc = cos.shape[-1] if rope else 0
    shfl = bool(rope and _SHFL and RDc >= 32 and (D // 4) >= 16 and _log2(RDc // 8) < _log2(D // 4))
    B, H, N, D = x_bhnd.shape
    BH = B * H
    dev = x_bhnd.device
    Q8 = torch.empty(BH, N, D, device=dev, dtype=torch.float8_e4m3fn)
    Q8s = torch.empty(BH, N // blk, device=dev, dtype=torch.float32)
    Qt = torch.empty(BH, D, N, device=dev, dtype=torch.float8_e4m3fn) if want_twin else None
    RD = cos.shape[-1] if rope else 4
    xc = x_bhnd.view(B, H, N, D // 4, 4)
    if rope:
        cc, sc = cos.view(B, N, RD // 4, 4), sin.view(B, N, RD // 4, 4)
    else:
        cc = sc = _dummy(dev)
    qt = (Qt if want_twin else Q8).view(torch.uint8)
    args = [_ct(xc), _ct(cc), _ct(sc), _ct(Q8.view(torch.uint8).view(BH, N, D // 4, 4)), _ct(Q8s),
            _ct(qt.view(BH, D, N // 16, 16))]
    sm = D ** -0.5
    key = ("ct", _TVM, x_bhnd.device.index, tuple(xc.shape), tuple(xc.stride()), tuple(cc.shape), tuple(cc.stride()), tuple(sc.stride()), blk,
           rope, want_twin, fastdiv, shfl, minb)
    _run(key, _ct_host, args, (N // blk, BH, H, blk, D, RD, rope, sm, want_twin, fastdiv, shfl, minb),
         cutlass_torch.current_stream())
    return Q8, Q8s, Qt


def _ct_ok(x, blk):
    if x.dim() != 4 or x.dtype != torch.bfloat16 or x.stride(-1) != 1 or x.data_ptr() % 16 != 0:
        return False
    B, H, N, D = x.shape
    return D == 256 and blk == 64 and N % blk == 0 and \
        all(s % 4 == 0 for s in x.stride()[:3])


def ropeq_twin(q, cos, sin, blk: int = 64, want_twin: bool = True, fastdiv=None):
    """q bf16 (B, H, N, D) pre-RoPE (any strides, lane stride 1); cos / sin bf16 (B, N, RD) ->
    (Q8 [BH, N, D], Q8s [BH, N // blk], Qt [BH, D, N] pi twin | None), bitwise tier3a_ropeq.ropeq_q + twins.twin_pi,
    in one launch.  None off the fast path."""
    if not _ct_ok(q, blk):
        return None
    B, H, N, D = q.shape
    RD = cos.shape[-1]
    if (cos.shape != (B, N, RD) or sin.shape != cos.shape or cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16
            or cos.stride(-1) != 1 or sin.stride(-1) != 1 or RD % 8 != 0 or RD > D or RD > 128
            or ((RD // 8) & (RD // 8 - 1)) != 0 or cos.data_ptr() % 16 != 0 or sin.data_ptr() % 16 != 0
            or any(s % 4 for s in cos.stride()[:2]) or any(s % 4 for s in sin.stride()[:2])):
        return None
    return _ct_launch(q, cos, sin, blk, True, want_twin, fastdiv)


def doq_twin(dO_bnhd, H: int, blk: int = 64, want_twin: bool = True, fastdiv=None):
    """dO bf16 (B, N, H, D) (lane stride 1) -> (dO8 [BH, N, D], dO8s [BH, N // blk], dOt [BH, D, N] pi twin | None),
    bitwise producers.doq_m17(want_delta=False) + twins.twin_pi, in one launch.  None off the fast path."""
    x = dO_bnhd.permute(0, 2, 1, 3)
    if dO_bnhd.shape[2] != H or not _ct_ok(x, blk):
        return None
    return _ct_launch(x, None, None, blk, False, want_twin, fastdiv)
