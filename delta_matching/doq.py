"""The dO quantizer of the plain route (core.dm_attn_func_plain): dO to e4m3 codes with per-block fp32 scales (the
quant.py arithmetic) and, in the same pass, the stale delta = rowsum(dO * O), reading O in the forward's (B, N, H, D)
output layout.  The API route uses producers.doq_m17 instead (bitwise the same codes, scales and delta).
"""
import torch
import triton
import triton.language as tl

from .constants import _TL_TINY, _TL_INV_E4M3_MAX, _TL_E4M3_MAX


@triton.jit
def _doq2_kernel(
    DO,          # [BH, N, D] any float dtype, contiguous
    O,           # [B, N, H, D] fp16, contiguous (the fwd kernel's native output)
    DO8,         # [BH, N, D] float8_e4m3fn  out
    DO8S,        # [BH, N // BM] fp32        out
    DLT,         # [BH, N] fp32              out
    DOF,         # [BH, N, D] fp16 CODES     out
    N, H,
    BM: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    DOF_BF16: tl.constexpr = False,
    HAS_DOF: tl.constexpr = True,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    mask_m = rm < N
    base = pid_bh.to(tl.int64) * N * D + rm.to(tl.int64)[:, None] * D
    b = pid_bh // H
    h = pid_bh % H
    obase = ((b.to(tl.int64) * N + rm.to(tl.int64)[:, None]) * H + h) * D

    # ---- pass A: block absmax over the whole [BM, D] quantization block ----
    amax = tl.zeros([1], dtype=tl.float32)
    for c in tl.static_range(D // BD):
        rd = c * BD + tl.arange(0, BD)
        x = tl.load(DO + base + rd[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))

    s = (tl.maximum(amax, tl.full([1], _TL_TINY, tl.float32))
         * tl.full([1], _TL_INV_E4M3_MAX, tl.float32))
    tl.store(DO8S + pid_bh.to(tl.int64) * (N // BM) + pid_m + tl.arange(0, 1), s)
    sv = tl.reshape(s, (1, 1))

    # ---- pass B: quantize + code stores + stale delta, same registers ----
    dlt = tl.zeros([BM], dtype=tl.float32)
    for c in tl.static_range(D // BD):
        rd = c * BD + tl.arange(0, BD)
        off = base + rd[None, :]
        x = tl.load(DO + off, mask=mask_m[:, None], other=0.0).to(tl.float32)
        q = tl.math.div_rn(x, tl.broadcast_to(sv, (BM, BD)))
        q = tl.minimum(tl.maximum(q, tl.full([1, 1], -_TL_E4M3_MAX, tl.float32)),
                       tl.full([1, 1], _TL_E4M3_MAX, tl.float32))
        q8 = q.to(tl.float8e4nv)
        tl.store(DO8 + off, q8, mask=mask_m[:, None])
        ## NB an in-loop fused transposed-twin store (tl.trans) MISCOMPILES here
        ## in the pinned Triton (one d%4 lane garbage regardless of num_stages or
        ## cvt order) -- a twin is made in a separate pass.
        if HAS_DOF:
            if DOF_BF16:
                tl.store(DOF + off, q8.to(tl.bfloat16), mask=mask_m[:, None])
            else:
                tl.store(DOF + off, q8.to(tl.float16), mask=mask_m[:, None])
        ov = tl.load(O + obase + rd[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        dlt += tl.sum(x * ov, axis=1)
    tl.store(DLT + pid_bh.to(tl.int64) * N + rm, dlt, mask=mask_m)


def fused_doq2(do, blk, o_bnhd, H, dof_dtype=torch.float16, want_dof=True):
    """do: [B, H, N, D] or [BH, N, D]; o_bnhd: fp16/bf16 [B, N, H, D] (the fwd's raw
    output, or an exact forward's output in the hybrid composition: read exactly via
    .to(fp32)). Returns (dO8 e4m3 [BH,N,D], dO8s fp32 [BH,N//blk], Dlt fp32 [BH,N],
             doF fp16/bf16 CODES [BH,N,D])."""
    if do.dim() == 4:
        B, Hd, N, D = do.shape
        assert Hd == H
        BH = B * H
    else:
        BH, N, D = do.shape
        B = BH // H
    do = do.contiguous().view(BH, N, D)
    assert o_bnhd.shape == (B, N, H, D)
    assert o_bnhd.dtype in (torch.float16, torch.bfloat16)
    assert o_bnhd.is_contiguous()
    assert N % blk == 0

    BD = 64 if D % 64 == 0 else D
    num_warps = 16 if blk * BD >= 4096 else 8

    dO8 = torch.empty((BH, N, D), device=do.device, dtype=torch.float8_e4m3fn)
    dO8s = torch.empty((BH, N // blk), device=do.device, dtype=torch.float32)
    dlt = torch.empty((BH, N), device=do.device, dtype=torch.float32)
    dof = (torch.empty((BH, N, D), device=do.device, dtype=dof_dtype)
           if want_dof else None)

    _doq2_kernel[(N // blk, BH)](
        do, o_bnhd, dO8, dO8s, dlt, dof if want_dof else dO8, N, H,
        BM=blk, D=D, BD=BD, DOF_BF16=(dof_dtype == torch.bfloat16),
        HAS_DOF=want_dof,
        num_warps=num_warps, num_stages=2,
    )
    return dO8, dO8s, dlt, dof
