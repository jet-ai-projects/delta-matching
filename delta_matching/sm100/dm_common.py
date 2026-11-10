"""Arithmetic shared bit-for-bit by the matched-delta pass 0 and the main backward.

Contract (sm100-mx backward): x = fma(S, c, -L2) with c = (ks * qs) * log2(e), L2 = lse * log2(e);  P = exp2(x).
Both kernels must call dm_bwd_p_pair so that P is bitwise identical (the tcgen05 fp8 MMA gives bitwise-equal S in
both orientations); any change of the exp implementation is made here, once.
"""
import math

import cutlass.cute as cute
from cutlass import Float32

LOG2E = math.log2(math.e)


def dm_bwd_c(ks: Float32, qs: Float32) -> Float32:
    """c = (ks * qs) * log2(e)  (this multiplication order)."""
    return (ks * qs) * Float32(LOG2E)


def dm_bwd_dp_scale(vs: Float32, dos: Float32) -> Float32:
    """a = vs * dos  (this multiplication order); dP = dP_acc * a."""
    return vs * dos


def dm_bwd_p_pair(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """(P0, P1) = exp2(fma((s0, s1), c, -l2)) with ex2.approx.ftz.f32."""
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return cute.math.exp2(x0, fastmath=True), cute.math.exp2(x1, fastmath=True)


# ---- alternative exp variants (not used by default; switch both kernels together by changing dm_bwd_p_pair) ----
from cutlass.cutlass_dsl import T, dsl_user_op  # noqa: E402
from cutlass._mlir.dialects import llvm  # noqa: E402


@dsl_user_op
def _ex2_f16x2(x0: Float32, x1: Float32, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x0).ir_value(loc=loc, ip=ip), Float32(x1).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 h;\n\t.reg .f16 lo, hi;\n\t"
        "cvt.rn.f16x2.f32 h, $3, $2;\n\t"
        "ex2.approx.f16x2 h, h;\n\t"
        "mov.b32 {lo, hi}, h;\n\t"
        "cvt.f32.f16 $0, lo;\n\t"
        "cvt.f32.f16 $1, hi;\n\t}",
        "=f,=f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)))


def dm_bwd_p_pair_f16x2(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """Proposal: the same fp32 FFMA, exp2 via MUFU ex2.approx.f16x2 (2 per op; rel err ~2^-11, P < 2^-24 -> 0)."""
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return _ex2_f16x2(x0, x1)


def dm_bwd_p_pair_poly(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """Proposal: the same fp32 FFMA, exp2 via FA4's degree-3 FMA polynomial (no MUFU)."""
    from flash_attn.cute import utils as fa_utils
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return fa_utils.ex2_emulation_2(x0, x1)


P_PAIR_VARIANTS = {"mufu": dm_bwd_p_pair, "f16x2": dm_bwd_p_pair_f16x2, "poly": dm_bwd_p_pair_poly}


def dm_bwd_p_pair_mix(s0: Float32, s1: Float32, c: Float32, l2: Float32, odd):
    """Proposal: element (q, k) uses FA4's FMA polynomial iff (q + k) is odd, MUFU ex2 otherwise.
    (s0, s1) are adjacent columns (even, odd) of a row whose index parity is `odd` (Boolean). Both kernels see
    exactly one MUFU + one polynomial per pair, via selects (no branches), and every element uses the same method
    in the Q-stationary pass 0 (rows = queries) and the KV-stationary backward (rows = keys)."""
    import cutlass
    from flash_attn.cute import utils as fa_utils
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    xm = Float32(cutlass.select_(odd, x1, x0))
    xp = Float32(cutlass.select_(odd, x0, x1))
    pm = cute.math.exp2(xm, fastmath=True)
    pp = fa_utils.ex2_emulation(xp)
    return Float32(cutlass.select_(odd, pp, pm)), Float32(cutlass.select_(odd, pm, pp))


def dm_bwd_p_pair_poly5(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """Same fp32 FFMA, exp2 via an accurate degree-5 FMA polynomial (max rel err 1.7e-7 = MUFU ex2.approx accuracy).
    Mixing it with MUFU in any static pattern keeps P within ~1e-7 between passes, so the delta_norm identity
    rowsum(dS) = 0 still holds to fp32 round-off (not bitwise)."""
    from . import dm_utils
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return dm_utils.ex2_poly5_2(x0, x1)


def dm_bwd_p_pair_poly6(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """Same fp32 FFMA as dm_bwd_p_pair, exp2 via dm_utils.ex2_poly6_2 (degree-6 FMA polynomial, at least as accurate as
    ex2.approx.ftz on every element metric). Pass 0 and the backward MUST pick it for exactly the same elements
    (b6 rule: 32x32 block (qb, kb) uses it iff (qb + kb) % m == 0), so P stays bitwise identical in both kernels."""
    from . import dm_utils
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return dm_utils.ex2_poly6_2(x0, x1)


def b6_m() -> int:
    """DM_B6=<m>: 32x32 blocks with (q//32 + k//32) % m == 0 use the degree-6 polynomial exp in pass 0 AND the backward
    (and forward P); 0 = all MUFU (default)."""
    import os
    return int(os.environ.get("DM_B6", "0"))


def p5_uses_poly(variant: str, v: int) -> bool:
    """Static pair schedule for 'p5_<f>_<r>' (deg-5) / 'p3_<f>_<r>' (FA4 deg-3): pair v (of 16 per 32-col chunk) uses the
    polynomial iff v % f < r (contiguous); 'p3b_<f>_<r>': iff (v * r) % f < r (spread, Bresenham)."""
    kind, f, r = variant.split("_")
    f, r = int(f), int(r)
    if kind == "p3b":
        return (v * r) % f < r
    return v % f < r


def is_static_mix(variant: str) -> bool:
    return variant.split("_")[0] in ("p5", "p3", "p3b")


P_PAIR_VARIANTS["mix"] = None  # needs the row parity: dispatched explicitly by the kernels
