"""Triton quantizers: Q, K, V to e4m3 codes with per-block fp32 scales.

Per (batch * head, block of rows): Q is multiplied by the softmax scale head_dim ** -0.5 before quantization (the kernels
never apply it again), K is centered by its per-head mean over the sequence (softmax is invariant to that shift; the mean
is accumulated in fp32, 64 rows at a time, in a fixed order), V is quantized as is.  Scale = max(absmax, 1e-30) / 448,
code = round-to-nearest-even e4m3 of x / scale (IEEE division, clamped to +-448).  Rows per scale block: core.blocks(D).
`fused_quant` returns codes, scales and, on request, the transposed K twin [BH, D, N]; producers.py and core.py launch
`_kmean_kernel` and `_quant_kernel` directly.
"""
import torch
import triton
import triton.language as tl

E4M3_MAX = triton.language.constexpr(448.0)


@triton.jit
def _kmean_kernel(K, KM, N, D: tl.constexpr, BD: tl.constexpr):
    """KM[bh, d] = mean over N of K[bh, :, d].  One program per (bh, d-chunk)."""
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    rd = pid_d * BD + tl.arange(0, BD)
    acc = tl.zeros([BD], dtype=tl.float32)
    base = pid_bh.to(tl.int64) * N * D
    for n in range(0, N, 64):
        rn = n + tl.arange(0, 64)
        x = tl.load(K + base + rn[:, None] * D + rd[None, :]).to(tl.float32)
        acc += tl.sum(x, axis=0)
    tl.store(KM + pid_bh.to(tl.int64) * D + rd, acc / N)


@triton.jit
def _quant_kernel(X, KM, X8, X8S, XT16, XF16, X8T, N, H,
                  SM: tl.constexpr, BLK: tl.constexpr, D: tl.constexpr,
                  MODE: tl.constexpr, WANT_T16: tl.constexpr,
                  T16_IS_FP8: tl.constexpr = False,
                  WANT_F16: tl.constexpr = False,
                  T16_IS_BF16: tl.constexpr = False,
                  F16_IS_BF16: tl.constexpr = False,
                  WANT_T8: tl.constexpr = False):
    """MODE 0: Q (scale by SM) | 1: K (subtract KM) | 2: V (plain).
    XT16 is fp16 codes at [B, N, H, D] when WANT_T16.
    XF16 is fp16 CODE values at [BH, N, D] (contiguous) when WANT_F16: the bwd's
    dKV/dQ GEMM operands (exact fp16 of the e4m3 codes), consumed via permuted view."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    rm = pid_m * BLK + tl.arange(0, BLK)
    rd = tl.arange(0, D)
    base = pid_bh.to(tl.int64) * N * D + rm.to(tl.int64)[:, None] * D + rd[None, :]

    x = tl.load(X + base).to(tl.float32)
    if MODE == 0:
        x = x * SM
    if MODE == 1:
        km = tl.load(KM + pid_bh.to(tl.int64) * D + rd)
        x = x - km[None, :]

    amax = tl.max(tl.abs(x))
    s = tl.maximum(amax, 1e-30) / E4M3_MAX
    tl.store(X8S + pid_bh.to(tl.int64) * (N // BLK) + pid_m, s)
    # div_rn: IEEE round-to-nearest division. A plain `/` may lower to a reciprocal
    # multiply and fuse with the SM multiply above, flipping ~4e-4 of Q codes vs the
    # torch reference (measured); _fused_doq learned the same lesson.
    q = tl.math.div_rn(x, tl.full([1, 1], 1.0, tl.float32) * s)
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)
    q8 = q.to(tl.float8e4nv)
    tl.store(X8 + base, q8)
    if WANT_T8:
        ## the transposed e4m3 twin [BH, D, N] (plain transpose), stored in the same pass
        t8_off = pid_bh.to(tl.int64) * N * D + rd.to(tl.int64)[:, None] * N + rm[None, :]
        tl.store(X8T + t8_off, tl.trans(q8))
    if WANT_F16:
        if F16_IS_BF16:
            tl.store(XF16 + base, q8.to(tl.bfloat16))
        else:
            tl.store(XF16 + base, q8.to(tl.float16))
    if WANT_T16:
        b = pid_bh // H
        h = pid_bh % H
        t_off = ((b.to(tl.int64) * N + rm.to(tl.int64)[:, None]) * H + h) * D + rd[None, :]
        if T16_IS_FP8:
            tl.store(XT16 + t_off, q8)
        elif T16_IS_BF16:
            tl.store(XT16 + t_off, q8.to(tl.bfloat16))
        else:
            tl.store(XT16 + t_off, q8.to(tl.float16))


def fused_quant(q, k, v, BM, BN, want_t16=True, t16_dtype=torch.float16,
                t16_dtypes=None, want_fp16_codes=None, code16_dtype=torch.float16,
                want_t16s=None, want_t8s=None):
    """q/k/v: [B, H, N, D] bf16.  Returns (Q8, Q8s, K8, K8s, V8, V8s[, qT, kT, vT]
    [, fp16 codes][, transposed e4m3 twins]).
    t16_dtype: the dtype of the optional (B, N, H, D) code copies (fp16, bf16 or e4m3).
    want_t16s: per-tensor override of want_t16 (d256 all8 drops the dead V codes).
    want_t8s: per-tensor [BH, D, N] e4m3 transposed twins fused into the same pass
    (appended last in the return tuple)."""
    B, H, N, D = q.shape
    BH = B * H
    dev = q.device
    sm = D ** -0.5
    qc = q.reshape(BH, N, D).contiguous()
    kc = k.reshape(BH, N, D).contiguous()
    vc = v.reshape(BH, N, D).contiguous()
    km = torch.empty(BH, D, device=dev, dtype=torch.float32)
    _kmean_kernel[(BH, D // 64)](kc, km, N, D=D, BD=64)

    if t16_dtypes is None:
        t16_dtypes = (t16_dtype, t16_dtype, t16_dtype)
    if want_fp16_codes is None:
        want_fp16_codes = (False, False, False)
    if want_t16s is None:
        want_t16s = (want_t16, want_t16, want_t16)
    if want_t8s is None:
        want_t8s = (False, False, False)
    outs = []
    t16s = []
    f16s = []
    t8s = []
    for (x, mode, blk), td, wf, wt16, wt8 in zip(
            ((qc, 0, BM), (kc, 1, BN), (vc, 2, BN)),
            t16_dtypes, want_fp16_codes, want_t16s, want_t8s):
        x8 = torch.empty(BH, N, D, device=dev, dtype=torch.float8_e4m3fn)
        x8s = torch.empty(BH, N // blk, device=dev, dtype=torch.float32)
        xt = torch.empty(B, N, H, D, device=dev, dtype=td) if wt16 else x8
        xf = torch.empty(BH, N, D, device=dev, dtype=code16_dtype) if wf else x8
        x8t = torch.empty(BH, D, N, device=dev, dtype=torch.float8_e4m3fn) if wt8 else x8
        _quant_kernel[(N // blk, BH)](x, km, x8, x8s, xt, xf, x8t, N, H,
                                      SM=sm, BLK=blk, D=D,
                                      MODE=mode, WANT_T16=wt16,
                                      T16_IS_FP8=(td == torch.float8_e4m3fn),
                                      WANT_F16=wf,
                                      T16_IS_BF16=(td == torch.bfloat16),
                                      F16_IS_BF16=(code16_dtype == torch.bfloat16),
                                      WANT_T8=wt8,
                                      num_warps=8)
        outs.extend([x8, x8s])
        if wt16:
            t16s.append(xt)
        if wf:
            f16s.append(xf)
        if wt8:
            t8s.append(x8t)
    return (*outs, *t16s, *f16s, *t8s)
