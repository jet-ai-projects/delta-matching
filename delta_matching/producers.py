"""The producers of the API route (attn.py): bitwise the quant.py / doq.py codes and scales, with less data movement.

  quant_qkv_m17  Q, K, V codes and scales plus the Q twin and the K twin (the K twin from the same pass as the K codes)
  quant_k_m17    K codes, scales and twin only (for callers that produce the V codes themselves)
  doq_m17        dO codes and scales read in the op's native (B, N, H, D) layout, the dO twin, and the stale delta
                 rowsum(dO * O) when asked (the matched mode never reads O)
The K mean runs as parallel per-64-row partial sums plus a serial fold in quant._kmean_kernel's order (bitwise; checked
against it on the first use of every shape, which falls back to it on a mismatch).  Fused twin stores (DM_FUSED_TWIN=1)
are off by default: a two-store Triton kernel miscompiles under some ptxas versions, so their first use per shape is
self-checked (DM_PRODUCER_SELFCHECK=1).  `selftest()` checks every output against the quant.py / doq.py path.
"""
from __future__ import annotations

import os
import warnings

import torch
import triton
import triton.language as tl

from . import quant as FQ
from .constants import _TL_TINY, _TL_INV_E4M3_MAX, _TL_E4M3_MAX
from .twins import twin_pi

E4M3_MAX = tl.constexpr(448.0)
SELFCHECK = os.environ.get("DM_PRODUCER_SELFCHECK", "1") == "1"
# Measured (H100, B2 H16 Hkv4 N8192): the fused two-store variants are CORRECT only in
# TWIN_MODE 2 with [64, 64] chunks under the bundled ptxas 12.8 (mode 1 = tl.trans is miscompiled: a ptxas
# two-store hazard: 25 M bad bytes, run-to-run nondeterministic; clean with the CUDA 13 ptxas) but that
# mode's scattered byte stores are SLOW (quant_q 0.62 ms, doq 0.62 ms vs 0.06 + 0.044 / 0.10 + 0.042 for
# the rowwise kernel + the single-store twin_pi pass).  Default = the two-kernel path; the fused kernels
# stay available (DM_FUSED_TWIN=1) behind the first-use self-check.
FUSED_TWIN = os.environ.get("DM_FUSED_TWIN", "0") == "1"
TWIN_MODE = int(os.environ.get("DM_TWIN_MODE", "2"))   # 1 = tl.trans, 2 = transposed addressing
_checked: set = set()
_kmean_ok: dict = {}


# ----------------------------------------------------------------------------- k-mean, split
@triton.jit
def _kmean_partial_kernel(K, KP, N, D: tl.constexpr, BD: tl.constexpr):
    """KP[bh, j, d] = tl.sum(K[bh, 64j:64j+64, d].float(), axis=0): the frozen kernel's per-chunk
    term, same [64, BD] fp32 block, same reduction axis (launched with the frozen kernel's num_warps)."""
    pid_bh = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_d = tl.program_id(2)
    rd = pid_d * BD + tl.arange(0, BD)
    base = pid_bh.to(tl.int64) * N * D
    rn = pid_j * 64 + tl.arange(0, 64)
    x = tl.load(K + base + rn[:, None] * D + rd[None, :]).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.store(KP + (pid_bh.to(tl.int64) * (N // 64) + pid_j) * D + rd, s)


@triton.jit
def _kmean_fold_kernel(KP, KM, N, D: tl.constexpr, BD: tl.constexpr):
    """KM[bh, d] = (((0 + s_0) + s_1) + ...) / N, the frozen kernel's serial order."""
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    rd = pid_d * BD + tl.arange(0, BD)
    acc = tl.zeros([BD], dtype=tl.float32)
    base = pid_bh.to(tl.int64) * (N // 64) * D
    for j in range(0, N // 64):
        acc += tl.load(KP + base + j * D + rd)
    tl.store(KM + pid_bh.to(tl.int64) * D + rd, acc / N)


def kmean_frozen(kc: torch.Tensor) -> torch.Tensor:
    BH, N, D = kc.shape
    km = torch.empty(BH, D, device=kc.device, dtype=torch.float32)
    FQ._kmean_kernel[(BH, D // 64)](kc, km, N, D=D, BD=64)
    return km


def kmean_m17(kc: torch.Tensor) -> torch.Tensor:
    """kc: [BH, N, D] bf16 contiguous -> [BH, D] fp32, bitwise the frozen _kmean_kernel."""
    BH, N, D = kc.shape
    assert N % 64 == 0 and D % 64 == 0
    key = (BH, N, D)
    if _kmean_ok.get(key) is False:
        return kmean_frozen(kc)
    kp = torch.empty(BH, N // 64, D, device=kc.device, dtype=torch.float32)
    km = torch.empty(BH, D, device=kc.device, dtype=torch.float32)
    _kmean_partial_kernel[(BH, N // 64, D // 64)](kc, kp, N, D=D, BD=64)
    _kmean_fold_kernel[(BH, D // 64)](kp, km, N, D=D, BD=64, num_warps=1)
    if key not in _kmean_ok:
        ref = kmean_frozen(kc)
        ok = torch.equal(ref, km)
        _kmean_ok[key] = ok
        if not ok:
            warnings.warn(f"delta_matching: the split k-mean differs from quant._kmean_kernel at shape {key}; "
                          "using quant._kmean_kernel for this shape", RuntimeWarning)
            return ref
    return km


# ----------------------------------------------------------------------------- Q: codes + scales + twin
@triton.jit
def _quant_q_twin_kernel(X, X8, X8S, X8T, N,
                         SM: tl.constexpr, BLK: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                         WANT_T8: tl.constexpr, TWIN_MODE: tl.constexpr = 2):
    """fused_quant._quant_kernel MODE 0's arithmetic (x*SM; s = max(amax, 1e-30)/448; div_rn by the
    broadcast s; clamp; RTNE e4m3) in the doq2 two-pass chunk structure ([BLK, BD] column chunks:
    amax is an exact max so chunking is order-free; every other op is elementwise), plus the
    pi-ordered twin store per chunk (the [64, 64]-chunk two-store shape self-checks clean under the
    bundled ptxas in TWIN_MODE 2, unlike the whole-row [64, 256] tile)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    rm = pid_m * BLK + tl.arange(0, BLK)
    base = pid_bh.to(tl.int64) * N * D + rm.to(tl.int64)[:, None] * D
    amax = tl.zeros([1], dtype=tl.float32)
    for c in tl.static_range(D // BD):
        rd = c * BD + tl.arange(0, BD)
        x = tl.load(X + base + rd[None, :]).to(tl.float32)
        x = x * SM
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    s = tl.maximum(amax, 1e-30) / E4M3_MAX
    tl.store(X8S + pid_bh.to(tl.int64) * (N // BLK) + pid_m + tl.arange(0, 1), s)
    sv = tl.reshape(s, (1, 1))
    pos = (rm & -16) | ((rm & 6) << 1) | ((rm & 8) >> 2) | (rm & 1)
    for c in tl.static_range(D // BD):
        rd = c * BD + tl.arange(0, BD)
        x = tl.load(X + base + rd[None, :]).to(tl.float32)
        x = x * SM
        q = tl.math.div_rn(x, tl.broadcast_to(sv, (BLK, BD)))
        q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)
        q8 = q.to(tl.float8e4nv)
        tl.store(X8 + base + rd[None, :], q8)
        if WANT_T8:
            if TWIN_MODE == 1:
                t8_off = pid_bh.to(tl.int64) * N * D + rd.to(tl.int64)[:, None] * N + pos[None, :]
                tl.store(X8T + t8_off, tl.trans(q8))
            else:
                t8_off = pid_bh.to(tl.int64) * N * D + rd.to(tl.int64)[None, :] * N + pos[:, None]
                tl.store(X8T + t8_off, q8)


def _selfcheck_twin(tag, x8, x8t):
    if not SELFCHECK or tag in _checked:
        return
    _checked.add(tag)
    ref = twin_pi(x8)
    if not torch.equal(ref.view(torch.uint8), x8t.view(torch.uint8)):
        raise RuntimeError(f"[producers] SELF-CHECK FAILED for {tag}: the fused twin store is wrong on this "
                           "toolchain (a ptxas two-store hazard); set DM_FUSED_TWIN=0")
    print(f"[producers] fused-twin self-check OK: {tag}", flush=True)


def quant_q_m17(q: torch.Tensor, blk: int = 64, want_twin: bool = True):
    """q: [B, H, N, D] bf16 (any dense strides) -> (Q8 [BH, N, D], Q8s [BH, N//blk], Qt [BH, D, N] | None)."""
    B, H, N, D = q.shape
    BH = B * H
    qc = q.reshape(BH, N, D).contiguous()
    sm = D ** -0.5
    dev = q.device
    Q8 = torch.empty(BH, N, D, device=dev, dtype=torch.float8_e4m3fn)
    Q8s = torch.empty(BH, N // blk, device=dev, dtype=torch.float32)
    fused = want_twin and FUSED_TWIN
    if fused:
        Qt = torch.empty(BH, D, N, device=dev, dtype=torch.float8_e4m3fn)
        _quant_q_twin_kernel[(N // blk, BH)](qc, Q8, Q8s, Qt, N, SM=sm, BLK=blk, D=D, BD=64, WANT_T8=True, TWIN_MODE=TWIN_MODE, num_warps=8)
        _selfcheck_twin(("quant_q", BH, N, D, blk, TWIN_MODE), Q8, Qt)
        return Q8, Q8s, Qt
    # the frozen _quant_kernel MODE 0, the exact launch of _quant_native's Q slot (bitwise by identity)
    FQ._quant_kernel[(N // blk, BH)](qc, qc, Q8, Q8s, Q8, Q8, Q8, N, H,
                                     SM=sm, BLK=blk, D=D, MODE=0, WANT_T16=False,
                                     T16_IS_FP8=False, WANT_F16=False, T16_IS_BF16=False,
                                     F16_IS_BF16=False, WANT_T8=False, num_warps=8)
    if want_twin:
        return Q8, Q8s, twin_pi(Q8)
    return Q8, Q8s, None


def fp11_blocks(D: int):
    """The block-scale contract per head dim (core.blocks): (Q/dO rows per scale, K/V keys per scale) = (64, 32) at
    head_dim 256, (128, 64) at head_dim 128."""
    return (64, 32) if D >= 256 else (128, 64)


def quant_kv_m17(k: torch.Tensor, v: torch.Tensor, blk_k: int = 32):
    """k, v: [B, Hk, N, D] bf16 -> (K8, K8s, kTt, V8, V8s) in the quant.py realization (the
    _quant_kernel launches, MODE 1 with the WANT_T8 twin / MODE 2), with the split k-mean.
    ## M19: blk_k = keys per K/V scale block (32 = the M17 d256 default; 64 at hd128)."""
    B, Hk, N, D = k.shape
    BH = B * Hk
    dev = k.device
    kc = k.reshape(BH, N, D).contiguous()
    vc = v.reshape(BH, N, D).contiguous()
    km = kmean_m17(kc)
    sm = D ** -0.5
    K8 = torch.empty(BH, N, D, device=dev, dtype=torch.float8_e4m3fn)
    K8s = torch.empty(BH, N // blk_k, device=dev, dtype=torch.float32)
    K8t = torch.empty(BH, D, N, device=dev, dtype=torch.float8_e4m3fn)
    FQ._quant_kernel[(N // blk_k, BH)](kc, km, K8, K8s, K8, K8, K8t, N, Hk,
                                       SM=sm, BLK=blk_k, D=D, MODE=1, WANT_T16=False,
                                       T16_IS_FP8=False, WANT_F16=False, T16_IS_BF16=False,
                                       F16_IS_BF16=False, WANT_T8=True, num_warps=8)
    V8 = torch.empty(BH, N, D, device=dev, dtype=torch.float8_e4m3fn)
    V8s = torch.empty(BH, N // blk_k, device=dev, dtype=torch.float32)
    FQ._quant_kernel[(N // blk_k, BH)](vc, km, V8, V8s, V8, V8, V8, N, Hk,
                                       SM=sm, BLK=blk_k, D=D, MODE=2, WANT_T16=False,
                                       T16_IS_FP8=False, WANT_F16=False, T16_IS_BF16=False,
                                       F16_IS_BF16=False, WANT_T8=False, num_warps=8)
    return K8, K8s, K8t, V8, V8s


def quant_k_m17(k: torch.Tensor, blk_k: int = 32):
    """k: [B, Hk, N, D] bf16 -> (K8, K8s, kTt) in the quant.py realization with the split k-mean
    (for callers that produce the V codes themselves).  ## M19: blk_k as quant_kv_m17."""
    B, Hk, N, D = k.shape
    BH = B * Hk
    dev = k.device
    kc = k.reshape(BH, N, D).contiguous()
    km = kmean_m17(kc)
    sm = D ** -0.5
    K8 = torch.empty(BH, N, D, device=dev, dtype=torch.float8_e4m3fn)
    K8s = torch.empty(BH, N // blk_k, device=dev, dtype=torch.float32)
    K8t = torch.empty(BH, D, N, device=dev, dtype=torch.float8_e4m3fn)
    FQ._quant_kernel[(N // blk_k, BH)](kc, km, K8, K8s, K8, K8, K8t, N, Hk,
                                       SM=sm, BLK=blk_k, D=D, MODE=1, WANT_T16=False,
                                       T16_IS_FP8=False, WANT_F16=False, T16_IS_BF16=False,
                                       F16_IS_BF16=False, WANT_T8=True, num_warps=8)
    return K8, K8s, K8t


def quant_qkv_m17(q, k, v, want_qt: bool = True, blocks=None):
    """## M19: blocks = (blk_q, blk_k) per head dim (default = fp11_blocks(D): (64, 32) at d256 = the M17 call)."""
    blk_q, blk_k = blocks if blocks is not None else fp11_blocks(q.shape[-1])
    Q8, Q8s, Qt = quant_q_m17(q, blk_q, want_qt)
    K8, K8s, kTt, V8, V8s = quant_kv_m17(k, v, blk_k)
    return Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt


# ----------------------------------------------------------------------------- dO: codes + scales (+ delta) + twin
@triton.jit
def _doq_m17_kernel(
    DO,          # bf16 (B, N, H, D) with strides (sdb, sdn, sdh, 1)
    O,           # bf16 [B, N, H, D] contiguous (WANT_DELTA only)
    DO8,         # [BH, N, D] e4m3 out
    DO8S,        # [BH, N // BM] fp32 out
    DLT,         # [BH, N] fp32 out (WANT_DELTA)
    DOT,         # [BH, D, N] e4m3 out (WANT_T8), pi-ordered N
    N, H, sdb, sdn, sdh,
    BM: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    WANT_DELTA: tl.constexpr,
    WANT_T8: tl.constexpr,
    TWIN_MODE: tl.constexpr = 2,
    DELTA_FROM_CODES: tl.constexpr = 0,   # delta = rowsum(dequant(dO8) * O), the operand dP actually sees
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    mask_m = rm < N
    b = pid_bh // H
    h = pid_bh % H
    dbase = b.to(tl.int64) * sdb + h.to(tl.int64) * sdh + rm.to(tl.int64)[:, None] * sdn
    obase = ((b.to(tl.int64) * N + rm.to(tl.int64)[:, None]) * H + h) * D
    cbase = pid_bh.to(tl.int64) * N * D + rm.to(tl.int64)[:, None] * D
    # ---- pass A: block absmax over the whole [BM, D] quantization block (doq2 verbatim) ----
    amax = tl.zeros([1], dtype=tl.float32)
    for c in tl.static_range(D // BD):
        rd = c * BD + tl.arange(0, BD)
        x = tl.load(DO + dbase + rd[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    s = (tl.maximum(amax, tl.full([1], _TL_TINY, tl.float32))
         * tl.full([1], _TL_INV_E4M3_MAX, tl.float32))
    tl.store(DO8S + pid_bh.to(tl.int64) * (N // BM) + pid_m + tl.arange(0, 1), s)
    sv = tl.reshape(s, (1, 1))
    # ---- pass B: quantize + code stores (+ stale delta) (+ twin) (doq2 verbatim) ----
    dlt = tl.zeros([BM], dtype=tl.float32)
    pos = (rm & -16) | ((rm & 6) << 1) | ((rm & 8) >> 2) | (rm & 1)
    for c in tl.static_range(D // BD):
        rd = c * BD + tl.arange(0, BD)
        x = tl.load(DO + dbase + rd[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        q = tl.math.div_rn(x, tl.broadcast_to(sv, (BM, BD)))
        q = tl.minimum(tl.maximum(q, tl.full([1, 1], -_TL_E4M3_MAX, tl.float32)),
                       tl.full([1, 1], _TL_E4M3_MAX, tl.float32))
        q8 = q.to(tl.float8e4nv)
        tl.store(DO8 + cbase + rd[None, :], q8, mask=mask_m[:, None])
        if WANT_T8:
            if TWIN_MODE == 1:
                t_off = pid_bh.to(tl.int64) * N * D + rd.to(tl.int64)[:, None] * N + pos[None, :]
                tl.store(DOT + t_off, tl.trans(q8))
            else:
                t_off = pid_bh.to(tl.int64) * N * D + rd.to(tl.int64)[None, :] * N + pos[:, None]
                tl.store(DOT + t_off, q8, mask=mask_m[:, None])
        if WANT_DELTA:
            ov = tl.load(O + obase + rd[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
            if DELTA_FROM_CODES:
                # the consistent-operand stale delta: the dequantized codes (what the dP gemm multiplies), not the raw dO
                dlt += tl.sum(q8.to(tl.float32) * tl.broadcast_to(sv, (BM, BD)) * ov, axis=1)
            else:
                dlt += tl.sum(x * ov, axis=1)
    if WANT_DELTA:
        tl.store(DLT + pid_bh.to(tl.int64) * N + rm, dlt, mask=mask_m)


def doq_m17(dO_bnhd: torch.Tensor, o_bnhd, H: int, blk: int = 64, want_delta: bool = False,
            want_twin: bool = True, delta_from_codes: bool = False):
    """dO_bnhd: bf16 (B, N, H, D), any dense strides with lane stride 1 (no copy); o_bnhd: bf16
    [B, N, H, D] contiguous (only read when want_delta).  Returns (dO8 [BH, N, D], dO8s [BH, N//blk],
    Dlt [BH, N] | None, dOt [BH, D, N] | None); dO8/dO8s/Dlt bitwise fused_doq2's."""
    B, N, Hd, D = dO_bnhd.shape
    assert Hd == H and dO_bnhd.dtype == torch.bfloat16 and dO_bnhd.stride(-1) == 1 and N % blk == 0
    if want_delta:
        assert o_bnhd is not None and o_bnhd.shape == (B, N, H, D) and o_bnhd.is_contiguous()
    BH = B * H
    BD = 64 if D % 64 == 0 else D
    num_warps = 16 if blk * BD >= 4096 else 8            # doq2.py verbatim
    dev = dO_bnhd.device
    dO8 = torch.empty((BH, N, D), device=dev, dtype=torch.float8_e4m3fn)
    dO8s = torch.empty((BH, N // blk), device=dev, dtype=torch.float32)
    dlt = torch.empty((BH, N), device=dev, dtype=torch.float32) if want_delta else dO8s
    fused = want_twin and FUSED_TWIN
    dOt = torch.empty((BH, D, N), device=dev, dtype=torch.float8_e4m3fn) if fused else dO8
    _doq_m17_kernel[(N // blk, BH)](
        dO_bnhd, o_bnhd if want_delta else dO_bnhd, dO8, dO8s, dlt, dOt, N, H,
        dO_bnhd.stride(0), dO_bnhd.stride(1), dO_bnhd.stride(2),
        BM=blk, D=D, BD=BD, WANT_DELTA=want_delta, WANT_T8=fused, TWIN_MODE=TWIN_MODE,
        DELTA_FROM_CODES=1 if (want_delta and delta_from_codes) else 0, num_warps=num_warps, num_stages=2)
    if fused:
        _selfcheck_twin(("doq", BH, N, D, blk, want_delta, TWIN_MODE), dO8, dOt)
    elif want_twin:
        dOt = twin_pi(dO8)
    else:
        dOt = None
    return dO8, dO8s, (dlt if want_delta else None), dOt


# ----------------------------------------------------------------------------- gates + bench hooks
def eq8(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def selftest(shapes=((2, 16, 4, 8192), (2, 8, 2, 4096), (1, 4, 1, 2048), (2, 6, 2, 8192)), verbose=True) -> bool:
    from .doq import fused_doq2
    from .core import _quant_native, BM
    from .twins import twin_pi_reference
    ok = True
    torch.manual_seed(0)
    for (B, H, Hk, N) in shapes:
        D = 256
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, Hk, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, Hk, N, D, device="cuda", dtype=torch.bfloat16)
        q[0, 0, :64] = 0   # a zero block (scale floor path)
        dO = (torch.randn(B, N, H, D, device="cuda") * 1e-3).to(torch.bfloat16)
        dO[0, :64, 0] = 0
        O = torch.randn(B, N, H, D, device="cuda").to(torch.bfloat16)
        rQ8, rQ8s, rK8, rK8s, rV8, rV8s, rkTt = _quant_native(q, k, v)
        Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_m17(q, k, v)
        e = dict(Q8=eq8(rQ8, Q8), Q8s=torch.equal(rQ8s, Q8s), K8=eq8(rK8, K8), K8s=torch.equal(rK8s, K8s),
                 V8=eq8(rV8, V8), V8s=torch.equal(rV8s, V8s), kTt=eq8(rkTt, kTt), Qt=eq8(twin_pi_reference(rQ8), Qt),
                 kmean=torch.equal(kmean_frozen(k.reshape(B * Hk, N, D).contiguous()), kmean_m17(k.reshape(B * Hk, N, D).contiguous())))
        r8, r8s, rdl, _ = fused_doq2(dO.permute(0, 2, 1, 3), BM, O, H, want_dof=False)
        g8, g8s, gdl, gt = doq_m17(dO, O, H, BM, want_delta=True)
        m8, m8s, _, mt = doq_m17(dO, None, H, BM, want_delta=False)
        e.update(dO8=eq8(r8, g8), dO8s=torch.equal(r8s, g8s), Dlt=torch.equal(rdl, gdl), dOt=eq8(twin_pi_reference(r8), gt),
                 dO8_md=eq8(r8, m8), dO8s_md=torch.equal(r8s, m8s), dOt_md=eq8(twin_pi_reference(r8), mt))
        ok &= all(e.values())
        if verbose:
            print(f"B{B} H{H} Hk{Hk} N{N}: " + " ".join(f"{kk}={'T' if vv else 'F'}" for kk, vv in e.items()))
    return ok


if __name__ == "__main__":
    print("SELFTEST", "PASS" if selftest() else "FAIL")
