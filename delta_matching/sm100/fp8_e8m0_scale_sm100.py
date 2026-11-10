"""
E8M0 (UE8M0) scale arithmetic shared by the sm100 kernels: power-of-two scales,
their biased exponent bytes and four-byte scale words.
"""
import cutlass  # noqa: F401  (the CuTe DSL preprocessor rewrites helper code into references to the cutlass module)
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from delta_matching.sm100.cute_ptx_sm100 import f32_unary

__all__ = ["exp2i_f32", "ue8m0_byte", "byte_shl23_f32", "pow2_exponent_byte", "pack_bytes4", "pow2_sb", "inv_pow2_from_sb", "add_sb_clamp", "byte_pow2_f32"]


@dsl_user_op
def exp2i_f32(e: Float32, *, loc=None, ip=None) -> Float32:
    """
    2^e for an integer-valued fp32 e (exact); 0 if e < -126; e is clamped to <= 127.
    """
    return f32_unary(
        "{\n\t.reg .s32 i;\n\t.reg .pred p;\n\t"
        "cvt.rni.s32.f32 i, $1;\n\t"
        "min.s32 i, i, 127;\n\t"
        "setp.lt.s32 p, i, -126;\n\t"
        "add.s32 i, i, 127;\n\t"
        "shl.b32 i, i, 23;\n\t"
        "selp.b32 i, 0, i, p;\n\t"
        "mov.b32 $0, i;\n\t}", e, loc, ip)


@dsl_user_op
def ue8m0_byte(e: Float32, *, loc=None, ip=None) -> Int32:
    """
    UE8M0 byte for 2^e (e integer-valued fp32): clamp(e + 127, 0, 254).
    """
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(e).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .s32 i;\n\t"
        "cvt.rni.s32.f32 i, $1;\n\t"
        "add.s32 i, i, 127;\n\t"
        "max.s32 i, i, 0;\n\t"
        "min.s32 i, i, 254;\n\t"
        "mov.b32 $0, i;\n\t}",
        "=r,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def byte_shl23_f32(b: Int32, *, loc=None, ip=None) -> Float32:
    """
    bitcast(b << 23): 2^(b - 127) for a UE8M0 byte b in [1, 254], +0 for b == 0 (= exp2i_f32 of
    the same e).
    """
    return Float32(llvm.inline_asm(T.f32(), [Int32(b).ir_value(loc=loc, ip=ip)], "shl.b32 $0, $1, 23;", "=f,r",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def pow2_exponent_byte(s: Float32, *, loc=None, ip=None) -> Int32:
    """
    Biased exponent field of a power-of-2 fp32 (= its UE8M0 byte).
    """
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(s).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 i;\n\tmov.b32 i, $1;\n\tbfe.u32 $0, i, 23, 8;\n\t}",
        "=r,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def pack_bytes4(b0: Int32, b1: Int32, b2: Int32, b3: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(b).ir_value(loc=loc, ip=ip) for b in (b0, b1, b2, b3)],
        "{\n\t.reg .b32 t0, t1;\n\t"
        "prmt.b32 t0, $1, $2, 0x0040;\n\t"     # bytes: [b0.0, b1.0, -, -]
        "prmt.b32 t1, $3, $4, 0x0040;\n\t"
        "prmt.b32 $0, t0, t1, 0x5410;\n\t}",
        "=r,r,r,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def pow2_sb(amax: Float32, *, loc=None, ip=None) -> Int32:
    """
    Biased exponent (UE8M0 byte) of the smallest power of 2 s with amax <= 448 s (amax >= 0 fp32):
    amax = m 2^E -> E-8 if m <= 1.75 else E-7; amax == 0 -> 127 (s = 1). Clamped to [1, 254].
    """
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(amax).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 a, e, m;\n\t.reg .pred pm, pz;\n\t"
        "mov.b32 a, $1;\n\t"
        "bfe.u32 e, a, 23, 8;\n\t"
        "and.b32 m, a, 0x7FFFFF;\n\t"
        "setp.gt.u32 pm, m, 0x600000;\n\t"
        "sub.s32 e, e, 8;\n\t"
        "@pm add.s32 e, e, 1;\n\t"
        "max.s32 e, e, 1;\n\t"
        "min.s32 e, e, 254;\n\t"
        "setp.eq.b32 pz, a, 0;\n\t"
        "selp.b32 $0, 127, e, pz;\n\t}",
        "=r,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def inv_pow2_from_sb(sb: Int32, *, loc=None, ip=None) -> Float32:
    """
    2^-(sb-127) as fp32 (exact).
    """
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(sb).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t;\n\tsub.s32 t, 254, $1;\n\tshl.b32 $0, t, 23;\n\t}",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def add_sb_clamp(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """
    UE8M0 byte of the product of two powers of 2 given their biased exponents: clamp(a + b -
    127, 0, 254).
    """
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .s32 t;\n\tadd.s32 t, $1, $2;\n\tsub.s32 t, t, 127;\n\tmax.s32 t, t, 0;\n\tmin.s32 $0, t, 254;\n\t}",
        "=r,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def byte_pow2_f32(word: Int32, j: int, *, loc=None, ip=None) -> Float32:
    """
    2^(byte_j(word) - 127) as fp32 (byte j of a packed UE8M0 word).
    """
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(word).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t;\n\t" f"bfe.u32 t, $1, {8 * j}, 8;\n\t" "shl.b32 $0, t, 23;\n\t}",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))
