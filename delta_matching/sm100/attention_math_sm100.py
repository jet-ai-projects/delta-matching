"""
Attention arithmetic shared bit for bit by the sm100 matched-delta pass 0 and
the backward, both head dims.

x = fma(S, c, -l2) with c = (ks * qs) * log2(e), l2 = lse * log2(e); P =
exp2(x); dP = dP_acc * (vs * dos). Pass 0 and the backward call dm_bwd_p_pair
(hd256: dm_mx256_p_pair / dm_mx256_p_pair2) so that P is bitwise identical in
both; any change of the exp implementation is made here, once.
"""
import math

import cutlass  # noqa: F401  (the CuTe DSL preprocessor rewrites helper code into references to the cutlass module)
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

__all__ = ["LOG2E", "dm_bwd_c", "dm_bwd_dp_scale", "dm_bwd_p_pair", "ex2_f16x2", "dm_bwd_p_pair_f16x2", "dm_bwd_p_pair_poly", "P_PAIR_VARIANTS", "dm_bwd_p_pair_mix", "dm_bwd_p_pair_poly5", "dm_bwd_p_pair_poly6", "b6_m", "p5_uses_poly", "is_static_mix", "POLY5_EX2", "ex2_poly5_2", "POLY6_EX2", "ex2_poly6_2", "dm_mx256_c", "dm_mx256_l2", "dm_mx256_dp_scale", "dm_mx256_p_pair", "dm_mx256_p_pair2"]


LOG2E = math.log2(math.e)


def dm_bwd_c(ks: Float32, qs: Float32) -> Float32:
    """
    c = (ks * qs) * log2(e)  (this multiplication order).
    """
    return (ks * qs) * Float32(LOG2E)


def dm_bwd_dp_scale(vs: Float32, dos: Float32) -> Float32:
    """
    a = vs * dos  (this multiplication order); dP = dP_acc * a.
    """
    return vs * dos


def dm_bwd_p_pair(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """
    (P0, P1) = exp2(fma((s0, s1), c, -l2)) with ex2.approx.ftz.f32.
    """
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return cute.math.exp2(x0, fastmath=True), cute.math.exp2(x1, fastmath=True)


@dsl_user_op
def ex2_f16x2(x0: Float32, x1: Float32, *, loc=None, ip=None):
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
    """
    Alternative (not used by default): the same fp32 FFMA, exp2 via MUFU ex2.approx.f16x2 (2 per op;
    rel err ~2^-11, P < 2^-24 -> 0).
    """
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return ex2_f16x2(x0, x1)


def dm_bwd_p_pair_poly(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """
    Alternative (not used by default): the same fp32 FFMA, exp2 via FA4's degree-3 FMA
    polynomial (no MUFU).
    """
    from flash_attn.cute import utils as fa_utils
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return fa_utils.ex2_emulation_2(x0, x1)


P_PAIR_VARIANTS = {"mufu": dm_bwd_p_pair, "f16x2": dm_bwd_p_pair_f16x2, "poly": dm_bwd_p_pair_poly}


def dm_bwd_p_pair_mix(s0: Float32, s1: Float32, c: Float32, l2: Float32, odd):
    """
    Alternative (not used by default): element (q, k) uses FA4's FMA polynomial iff (q + k) is odd,
    MUFU ex2 otherwise. (s0, s1) are adjacent columns (even, odd) of a row whose index parity is
    `odd` (Boolean). Both kernels see exactly one MUFU + one polynomial per pair, via selects (no
    branches), and every element uses the same method in the Q-stationary pass 0 (rows = queries)
    and the KV-stationary backward (rows = keys).
    """
    from flash_attn.cute import utils as fa_utils
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    xm = Float32(cutlass.select_(odd, x1, x0))
    xp = Float32(cutlass.select_(odd, x0, x1))
    pm = cute.math.exp2(xm, fastmath=True)
    pp = fa_utils.ex2_emulation(xp)
    return Float32(cutlass.select_(odd, pp, pm)), Float32(cutlass.select_(odd, pm, pp))


def dm_bwd_p_pair_poly5(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """
    Same fp32 FFMA, exp2 via an accurate degree-5 FMA polynomial (max rel err 1.7e-7 = MUFU
    ex2.approx accuracy). Mixing it with MUFU in any static pattern keeps P within ~1e-7 between
    passes, so the delta_norm identity rowsum(dS) = 0 still holds to fp32 round-off (not bitwise).
    """
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return ex2_poly5_2(x0, x1)


def dm_bwd_p_pair_poly6(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """
    Same fp32 FFMA as dm_bwd_p_pair, exp2 via ex2_poly6_2 (degree-6 FMA polynomial, at least as
    accurate as ex2.approx.ftz on every element metric). Pass 0 and the backward MUST pick it for
    exactly the same elements (b6 rule: 32x32 block (qb, kb) uses it iff (qb + kb) % m == 0), so P
    stays bitwise identical in both kernels.
    """
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l2, -l2))
    return ex2_poly6_2(x0, x1)


def b6_m() -> int:
    """
    DM_B6=<m>: 32x32 blocks with (q//32 + k//32) % m == 0 use the degree-6 polynomial exp in pass 0
    AND the backward (and forward P); 0 = all MUFU (default).
    """
    import os
    return int(os.environ.get("DM_B6", "0"))


def p5_uses_poly(variant: str, v: int) -> bool:
    """
    Static pair schedule for 'p5_<f>_<r>' (deg-5) / 'p3_<f>_<r>' (FA4 deg-3): pair v (of 16 per
    32-col chunk) uses the polynomial iff v % f < r (contiguous); 'p3b_<f>_<r>': iff (v * r) % f <
    r (spread, Bresenham).
    """
    kind, f, r = variant.split("_")
    f, r = int(f), int(r)
    if kind == "p3b":
        return (v * r) % f < r
    return v % f < r


def is_static_mix(variant: str) -> bool:
    return variant.split("_")[0] in ("p5", "p3", "p3b")


# 2^f on [0, 1), fp32 Horner, degree 5: max rel error 1.7e-7 (= MUFU ex2.approx accuracy); fitted IRLS minimax
POLY5_EX2 = (0.9999999403953552, 0.6931530833244324, 0.24015361070632935, 0.05582635849714279,
             0.008989292196929455, 0.001877596601843834)


@dsl_user_op
def ex2_poly5_2(x: Float32, y: Float32, *, loc=None, ip=None):
    """
    Accurate (1.7e-7 rel) 2^x for a pair on the FMA pipe (FA4's ex2_emulation_2 with a degree-5
    polynomial). Inputs are clamped at -126 (results ~2^-126 instead of 0 below that; negligible vs
    any row sum).
    """
    from flash_attn.cute import utils as fu
    fp32_round_int = float(2**23 + 2**22)
    # clamp at -126: c0 < 1 means 2^frac can be < 1, and floor(x) = -127 would wrap the exponent field
    xy = (cute.arch.fmax(x, -126.0), cute.arch.fmax(y, -126.0))
    r = cute.arch.add_packed_f32x2(xy, (fp32_round_int, fp32_round_int), rnd="rm")
    rb = cute.arch.sub_packed_f32x2(r, (fp32_round_int, fp32_round_int))
    fr = cute.arch.sub_packed_f32x2(xy, rb)
    pe = fu.evaluate_polynomial_2(*fr, POLY5_EX2, loc=loc, ip=ip)
    return fu.combine_int_frac_ex2(r[0], pe[0], loc=loc, ip=ip), fu.combine_int_frac_ex2(r[1], pe[1], loc=loc, ip=ip)


# degree-6 exp2 on [0,1) in the form 1 + f*q(f):
# exhaustive over all fp32 exp arguments in (-64,-0.5] U [0.5,8): max rel 6.6e-8, mean 2.3e-8, 91.8% correctly rounded,
# e4m3 RN-code mismatch 4.1e-7 vs MUFU ex2.approx.ftz 1.46e-7 / 4.4e-8 / 57.5% / 4.9e-7
POLY6_EX2 = (1.0, float.fromhex("0x1.62e42cp-1"), float.fromhex("0x1.ebfd58p-3"), float.fromhex("0x1.c68912p-5"),
             float.fromhex("0x1.3d0b92p-7"), float.fromhex("0x1.46d648p-10"), float.fromhex("0x1.c54182p-13"))


def ex2_poly6_2(x: Float32, y: Float32):
    """
    2^x for a pair on the FMA pipe, at least as accurate as ex2.approx.ftz.f32 on every
    element metric; results below 2^-126 (x < -126, incl. -inf) are flushed to 0
    exactly like ex2.approx.ftz.
    """
    from flash_attn.cute import utils as fu
    fp32_round_int = float(2**23 + 2**22)
    xy = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    r = cute.arch.add_packed_f32x2(xy, (fp32_round_int, fp32_round_int), rnd="rm")
    rb = cute.arch.sub_packed_f32x2(r, (fp32_round_int, fp32_round_int))
    fr = cute.arch.sub_packed_f32x2(xy, rb)
    pe = fu.evaluate_polynomial_2(*fr, POLY6_EX2)
    ox = fu.combine_int_frac_ex2(r[0], pe[0])
    oy = fu.combine_int_frac_ex2(r[1], pe[1])
    return (Float32(cutlass.select_(x < Float32(-126.0), Float32(0.0), ox)),
            Float32(cutlass.select_(y < Float32(-126.0), Float32(0.0), oy)))


def dm_mx256_c(ks: Float32, qs: Float32) -> Float32:
    """
    c = (ks * qs) * log2(e)  (= dm_bwd_c, this association).
    """
    return dm_bwd_c(ks, qs)


def dm_mx256_l2(lse: Float32) -> Float32:
    """
    l2 = lse * log2(e) (FMUL RN; = hd128 pass 0 / bwd).
    """
    return lse * Float32(LOG2E)


def dm_mx256_dp_scale(vs: Float32, dos: Float32) -> Float32:
    """
    a = vs * dos  (= dm_bwd_dp_scale); dP = dP_acc * a.
    """
    return dm_bwd_dp_scale(vs, dos)


def dm_mx256_p_pair(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """
    (P0, P1) = ex2.approx.ftz.f32(fma.rn((s0, s1), c, -l2))  (= dm_bwd_p_pair).
    """
    return dm_bwd_p_pair(s0, s1, c, l2)


def dm_mx256_p_pair2(s0: Float32, s1: Float32, c: Float32, l20: Float32, l21: Float32):
    """
    Keys-on-rows form (backward dK/dV kernels: the two elements of a pair are two QUERIES of one
    key, so each has its own l2): (P0, P1) = ex2.approx.ftz.f32(fma.rn((s0, s1), c, (-l20, -l21))).
    Per element this is the same FFMA RN + MUFU.EX2 as dm_mx256_p_pair (fma_packed_f32x2 is
    per-element FFMA RN), hence bitwise the same P.
    """
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l20, -l21))
    return cute.math.exp2(x0, fastmath=True), cute.math.exp2(x1, fastmath=True)
