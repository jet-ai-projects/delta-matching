"""hd256 arithmetic shared bit-for-bit by the hd256 matched-delta pass 0 and the hd256 backward.

=== SPEC: MXFP8 = sm100-mx at D=256 -> dm_mx256_* ===
The hd128 sm100-mx P / dP line VERBATIM (each helper calls the hd128 dm_common function, so hd128 and hd256 share one
implementation).  Inputs: Q / K / V / dO e4m3 codes with ONE pow2 (UE8M0) fp32 scale per 32 tokens over all of D
(quant.quant_rows rule), Q pre-multiplied by sm = 256^-0.5, K centered.  Per element (row i = query, col j = key):
  S_acc  = raw fp32 accumulator of tcgen05.mma kind::f8f6f4 (Q8 K8^T) over D=256 as 8 K=32 steps in increasing d
           (zero-init); causal: S_acc = -inf where j > i (on the raw accumulator)
  c      = (ks[j//32] * qs[i//32]) * log2(e)          dm_mx256_c(ks, qs)         FMUL, FMUL (this association)
  l2     = lse[i] * log2(e)                           dm_mx256_l2(lse)           FMUL RN (lse natural log, fp32)
  x      = fma(S_acc, c, -l2)                         fma.rn.f32x2 (per element = FFMA RN)
  P      = ex2.approx.ftz.f32(x)                      dm_mx256_p_pair(s0, s1, c, l2)   (no shift, no cap: hd128 line)
  a      = vs[j//32] * dos[i//32]                     dm_mx256_dp_scale(vs, dos) FMUL
  dP     = dP_acc * a        (dP_acc = raw f8f6f4 accumulator of dO8 V8^T, same instruction shape as S)
The scales are on the MMAs' non-contraction dims (rows / cols of S and dP) and are exact powers of two, so the
post-MMA multiply equals the kind::mxf8f6f4 block-scaled product with uniform SFs; like hd128's pass 0, no SF TMEM is used.
The backward must compute P with dm_mx256_p_pair from bitwise the same S_acc, c, l2 (and dP_acc, a).

=== PTX helpers (inline asm) of the hd256 backward kernels (DmHd256Mx, DmHd256Mx2, DmHd256Kv2) ===
Moved verbatim from the deleted pre-MXFP8 backward module; hd128's dm_utils is untouched.
"""
import math

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from delta_matching.sm100 import dm_common

LOG2E = math.log2(math.e)


# ---------------------------------------------------------------------------------- MXFP8 (sm100-mx-d256) = THE spec
def dm_mx256_c(ks: Float32, qs: Float32) -> Float32:
    """c = (ks * qs) * log2(e)  (= dm_common.dm_bwd_c, this association)."""
    return dm_common.dm_bwd_c(ks, qs)


def dm_mx256_l2(lse: Float32) -> Float32:
    """l2 = lse * log2(e) (FMUL RN; = hd128 pass 0 / bwd)."""
    return lse * Float32(dm_common.LOG2E)


def dm_mx256_dp_scale(vs: Float32, dos: Float32) -> Float32:
    """a = vs * dos  (= dm_common.dm_bwd_dp_scale); dP = dP_acc * a."""
    return dm_common.dm_bwd_dp_scale(vs, dos)


def dm_mx256_p_pair(s0: Float32, s1: Float32, c: Float32, l2: Float32):
    """(P0, P1) = ex2.approx.ftz.f32(fma.rn((s0, s1), c, -l2))  (= dm_common.dm_bwd_p_pair)."""
    return dm_common.dm_bwd_p_pair(s0, s1, c, l2)


def dm_mx256_p_pair2(s0: Float32, s1: Float32, c: Float32, l20: Float32, l21: Float32):
    """Keys-on-rows form (backward dK/dV kernels: the two elements of a pair are two QUERIES of one key, so each has its
    own l2): (P0, P1) = ex2.approx.ftz.f32(fma.rn((s0, s1), c, (-l20, -l21))).  Per element this is the same FFMA RN +
    MUFU.EX2 as dm_mx256_p_pair (fma_packed_f32x2 is per-element FFMA RN), hence bitwise the same P."""
    x0, x1 = cute.arch.fma_packed_f32x2((s0, s1), (c, c), (-l20, -l21))
    return cute.math.exp2(x0, fastmath=True), cute.math.exp2(x1, fastmath=True)


# ---------------------------------------------------------------- PTX helpers (local; hd128's dm_utils untouched)
def _asm(ret, args, asm, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(ret, args, asm, cons, has_side_effects=side, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tmem_ld32(addr: Int32, *, loc=None, ip=None):
    """tcgen05.ld.32x32b.x32: thread t of warp w gets TMEM lane (addr.lane + t), columns addr.col .. +31."""
    regs = ",".join(f"${i}" for i in range(32))
    res = _asm(llvm.StructType.get_literal([T.f32()] * 32), [Int32(addr).ir_value(loc=loc, ip=ip)],
               "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];", ",".join(["=f"] * 32) + ",r", True,
               loc, ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def mul_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """FMUL RN, never contracted with a following add."""
    return Float32(_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "mul.rn.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def add_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "add.rn.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def abs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)], "abs.f32 $0, $1;", "=f,f", False, loc, ip))


@dsl_user_op
def max_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """fmaxf (IEEE maxNum: NaN-ignoring)."""
    return Float32(_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "max.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def e4m3x4(x0: Float32, x1: Float32, x2: Float32, x3: Float32, *, loc=None, ip=None) -> Int32:
    """cvt.rn.satfinite.e4m3x2.f32 x2: bytes [x0, x1, x2, x3] little-endian."""
    return Int32(_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3)],
                      "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
                      "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}",
                      "=r,f,f,f,f", False, loc, ip))


@dsl_user_op
def bf16x2(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Int32:
    """cvt.rn.bf16x2.f32: low half x0, high half x1."""
    return Int32(_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1)],
                      "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", False, loc, ip))


@dsl_user_op
def st_shared_f32(addr: Int32, v: Float32, *, loc=None, ip=None):
    _asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
         "st.shared.f32 [$0], $1;", "r,f", True, loc, ip)


@dsl_user_op
def ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", True,
                        loc, ip))


@dsl_user_op
def ld_shared_v4(addr: Int32, *, loc=None, ip=None):
    res = _asm(llvm.StructType.get_literal([T.i32()] * 4), [Int32(addr).ir_value(loc=loc, ip=ip)],
               "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", True, loc, ip)
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def st_global_v4(addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
         "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", True, loc, ip)


@dsl_user_op
def st_global_v4f(addr: Int64, a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
         "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", True, loc, ip)


@dsl_user_op
def st_global_f32(addr: Int64, v: Float32, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
         "st.global.f32 [$0], $1;", "l,f", True, loc, ip)


def _gaddr(t: cute.Tensor) -> Int64:
    return Int64(t.iterator.toint())
