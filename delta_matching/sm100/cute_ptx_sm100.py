"""
Inline-PTX operations (CuTe DSL dsl_user_op wrappers) used by two or
more sm100 kernel files. The mbarrier wait / arrive operations here
are the raw instructions; kernels use the fenced versions of them
that tcgen05_fence_sm100 exports.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

__all__ = ["f32_unary", "inline_asm", "floor_f32", "ceil_f32", "bitcast_i32_f32", "redux_max_abs_f32", "tmem_st_x1", "tmem_st_x4", "mbar_arrive_expect_tx", "mbar_wait_parity_nh", "mbar_wait_parity", "ld_shared_b32", "st_shared_b32", "clock64", "red_add_u64", "poll_acquire_ge_or_trap", "fence_proxy_async_global", "clock32", "globaltimer", "smid", "mbar_arrive_rank", "mbar_arrive_local", "mbar_wait_parity_cluster", "f32_to_f64_exact", "f32_to_f64_fastpath", "f32_is_subnormal", "mbar_init", "mbar_arrive", "tc_commit", "st_shared_v4_b32", "bitcast_f32_i32", "nanosleep_1us", "tmem_ld_16x64b_x32", "tmem_ld_16x64b_x16", "tmem_ld32", "mul_f32", "add_f32", "abs_f32", "max_f32", "e4m3x4", "bf16x2", "st_shared_f32", "ld_shared_f32", "ld_shared_v4", "st_global_v4", "st_global_v4f", "st_global_f32", "global_address"]


def f32_unary(asm, x, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)], asm, "=f,f",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def inline_asm(ret, args, asm, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(ret, args, asm, cons, has_side_effects=side, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def floor_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return f32_unary("cvt.rmi.f32.f32 $0, $1;", x, loc, ip)


@dsl_user_op
def ceil_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return f32_unary("cvt.rpi.f32.f32 $0, $1;", x, loc, ip)


@dsl_user_op
def bitcast_i32_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(), [Int32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def redux_max_abs_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """
    Warp-wide max |x| (redux.sync.max.abs.f32, sm_100a).
    """
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        "redux.sync.max.abs.f32 $0, $1, 0xffffffff;",
        "=f,f", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def tmem_st_x1(addr: Int32, w: Int32, *, loc=None, ip=None):
    """
    tcgen05.st.32x32b.x1: this thread's lane, one 32-bit column at addr (lane base in bits 16+).
    """
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(w).ir_value(loc=loc, ip=ip)],
                    "tcgen05.st.sync.aligned.32x32b.x1.b32 [$0], {$1};", "r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tmem_st_x4(addr: Int32, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(x).ir_value(loc=loc, ip=ip) for x in (addr, w0, w1, w2, w3)],
                    "tcgen05.st.sync.aligned.32x32b.x4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def mbar_arrive_expect_tx(mbar: Int32, nbytes: int, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
                    f"mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], {nbytes};", "r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def mbar_wait_parity_nh(mbar: Int32, phase: Int32, *, loc=None, ip=None):
    """
    mbar_wait_parity without the try_wait suspend-time hint (hardware default time limit).
    """
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .pred P1;\n\tDM_WAITN:\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\n\t"
                    "@P1 bra DM_DONEN;\n\tbra DM_WAITN;\n\tDM_DONEN:\n\t}",
                    "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def mbar_wait_parity(mbar: Int32, phase: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .pred P1;\n\tDM_WAIT:\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1, 10000000;\n\t"
                    "@P1 bra DM_DONE;\n\tbra DM_WAIT;\n\tDM_DONE:\n\t}",
                    "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_b32(addr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.volatile.shared.b32 $0, [$1];",
                                 "=r,r", has_side_effects=True, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def st_shared_b32(addr: Int32, val: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
                    "st.volatile.shared.b32 [$0], $1;", "r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def clock64(*, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l",
                                         has_side_effects=True, is_align_stack=False,
                                         asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def red_add_u64(addr: cutlass.Int64, val: cutlass.Int64, *, loc=None, ip=None):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int64(val).ir_value(loc=loc, ip=ip)],
                    "red.global.add.u64 [$0], $1;", "l,l", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def poll_acquire_ge_or_trap(addr: cutlass.Int64, target: Int32, limit: Int32, *, loc=None, ip=None):
    """
    Spin with ld.acquire.gpu on the int32 at addr until >= target; trap after `limit` polls (a
    missing release becomes a launch failure, never a silent hang).
    """
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(target).ir_value(loc=loc, ip=ip),
                           Int32(limit).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .pred p;\n\t.reg .b32 v, it;\n\tmov.b32 it, 0;\n"
                    "DM_TCPG:\n\tld.acquire.gpu.global.b32 v, [$0];\n\tsetp.ge.s32 p, v, $1;\n\t@p bra DM_TCPGD;\n\t"
                    "add.s32 it, it, 1;\n\tsetp.gt.s32 p, it, $2;\n\t@p trap;\n\tnanosleep.u32 256;\n\tbra DM_TCPG;\n"
                    "DM_TCPGD:\n\t}",
                    "l,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "fence.proxy.async.global;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def clock32(*, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %clock;", "=r",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
                                         has_side_effects=True, is_align_stack=False,
                                         asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def mbar_arrive_rank(mbar: Int32, rank: Int32, *, loc=None, ip=None):
    """
    mbarrier arrive (release, cluster scope) on CTA `rank`'s mbarrier at the offset of local `mbar`.
    """
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(rank).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .b32 rm;\n\t"
                    "mapa.shared::cluster.u32 rm, $0, $1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [rm];\n\t}",
                    "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def mbar_arrive_local(mbar: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
                    "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];",
                    "r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def mbar_wait_parity_cluster(mbar: Int32, phase: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
                    "{\n\t.reg .pred P1;\n\tDM_WAITC:\n\t"
                    "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [$0], $1, 10000000;\n\t"
                    "@P1 bra DM_DONEC;\n\tbra DM_WAITC;\n\tDM_DONEC:\n\t}",
                    "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def f32_to_f64_exact(x: Float32, *, loc=None, ip=None) -> Float64:
    """
    Exact fp32 -> fp64 conversion without F2F (which issues on the XU/MUFU pipe): integer bit
    assembly for normals, inf and NaN; an exact FP64-pipe path (2^52 magic subtraction, power-of-two
    scaling) for zero and subnormals. Bitwise identical to cvt.f64.f32 for every fp32 input (NaN
    payload: quiet bit and upper mantissa preserved).
    """
    asm = (
        "{\n"
        ".reg .b32 a, e, m, s, hi, lo, hii, mg, shi, slo;\n"
        ".reg .b64 dn, ds;\n"
        ".reg .f64 t;\n"
        ".reg .pred pz, pi;\n"
        "mov.b32 a, $1;\n"
        "and.b32 s, a, 0x80000000;\n"
        "and.b32 e, a, 0x7f800000;\n"
        "and.b32 m, a, 0x007fffff;\n"
        "shr.u32 hi, a, 3;\n"
        "and.b32 hi, hi, 0x0fffffff;\n"
        "add.u32 hi, hi, 0x38000000;\n"
        "or.b32 hi, hi, s;\n"
        "shl.b32 lo, a, 29;\n"
        "shr.u32 hii, m, 3;\n"
        "or.b32 hii, hii, 0x7ff00000;\n"
        "or.b32 hii, hii, s;\n"
        "setp.ne.u32 pz, m, 0;\n"
        "selp.b32 mg, 0x00080000, 0, pz;\n"
        "or.b32 hii, hii, mg;\n"
        "setp.eq.u32 pi, e, 0x7f800000;\n"
        "selp.b32 hi, hii, hi, pi;\n"
        "mov.b64 dn, {lo, hi};\n"
        "mov.b32 mg, 0x43300000;\n"
        "mov.b64 ds, {m, mg};\n"
        "mov.b64 t, ds;\n"
        "sub.rn.f64 t, t, 0d4330000000000000;\n"
        "mul.rn.f64 t, t, 0d36A0000000000000;\n"
        "mov.b64 {slo, shi}, t;\n"
        "or.b32 shi, shi, s;\n"
        "mov.b64 ds, {slo, shi};\n"
        "setp.eq.u32 pz, e, 0;\n"
        "selp.b64 dn, ds, dn, pz;\n"
        "mov.b64 $0, dn;\n"
        "}\n"
    )
    return Float64(llvm.inline_asm(T.f64(), [Float32(x).ir_value(loc=loc, ip=ip)], asm, "=d,f",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def f32_to_f64_fastpath(x: Float32, *, loc=None, ip=None) -> Float64:
    """
    fp32 -> fp64 by integer bit assembly only (ALU): exact for zero, normals, inf and NaN (quiet bit
    set like cvt.f64.f32). NOT valid for subnormals (exponent field 0, mantissa != 0): callers must
    route those to Float64(x).
    """
    asm = (
        "{\n"
        ".reg .b32 a, e, m, s, hi, lo, hii, q;\n"
        ".reg .pred pz, pi, pq;\n"
        "mov.b32 a, $1;\n"
        "and.b32 s, a, 0x80000000;\n"
        "and.b32 e, a, 0x7f800000;\n"
        "and.b32 m, a, 0x007fffff;\n"
        "shr.u32 hi, a, 3;\n"
        "and.b32 hi, hi, 0x0fffffff;\n"
        "add.u32 hi, hi, 0x38000000;\n"
        "or.b32 hi, hi, s;\n"
        "shl.b32 lo, a, 29;\n"
        "shr.u32 hii, m, 3;\n"
        "or.b32 hii, hii, 0x7ff00000;\n"
        "or.b32 hii, hii, s;\n"
        "setp.ne.u32 pq, m, 0;\n"
        "selp.b32 q, 0x00080000, 0, pq;\n"
        "or.b32 hii, hii, q;\n"
        "setp.eq.u32 pi, e, 0x7f800000;\n"
        "selp.b32 hi, hii, hi, pi;\n"
        "setp.eq.u32 pz, e, 0;\n"
        "selp.b32 hi, s, hi, pz;\n"
        "selp.b32 lo, 0, lo, pz;\n"
        "mov.b64 $0, {lo, hi};\n"
        "}\n"
    )
    return Float64(llvm.inline_asm(T.f64(), [Float32(x).ir_value(loc=loc, ip=ip)], asm, "=d,f",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def f32_is_subnormal(x: Float32, *, loc=None, ip=None) -> Int32:
    """
    1 if x is a nonzero fp32 subnormal (exponent field 0, mantissa != 0), else 0 (ALU only).
    """
    return Int32(llvm.inline_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)],
                                 "{\n.reg .b32 a, e, m;\n.reg .pred p, q;\nmov.b32 a, $1;\nand.b32 e, a, 0x7f800000;\n"
                                 "and.b32 m, a, 0x007fffff;\nsetp.eq.u32 p, e, 0;\nsetp.ne.and.u32 q, m, 0, p;\n"
                                 "selp.u32 $0, 1, 0, q;\n}\n", "=r,f",
                                 has_side_effects=False, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def mbar_init(mbar: Int32, count: int, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)], f"mbarrier.init.shared::cta.b64 [$0], {count};", "r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def mbar_arrive(mbar: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)], "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];",
                    "r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tc_commit(mbar: Int32, *, loc=None, ip=None):
    """
    tcgen05.commit (cta_group::1): mbarrier arrives once all prior async tcgen05 ops of this
    thread (mma / cp / shift) have completed; tcgen05.ld / st are NOT tracked (they use
    tcgen05.wait::ld / ::st).
    """
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def st_shared_v4_b32(addr: Int32, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(x).ir_value(loc=loc, ip=ip) for x in (addr, a, b, c, d)],
                    "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def bitcast_f32_i32(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f",
                                 has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def nanosleep_1us(*, loc=None, ip=None):
    """
    DEBUG only: nanosleep ~1 us (used to widen race windows in positive controls).
    """
    llvm.inline_asm(None, [], "nanosleep.u32 1000;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


# tcgen05.ld.16x64b raw loads (pass 0 two-lane layout).
# Measured mapping: thread t -> TMEM lane base + t//4 + 8*(t&1), column parity (t>>1)&1,
# repetition r -> column 2r + parity.
@dsl_user_op
def tmem_ld_16x64b_x32(addr: Int32, *, loc=None, ip=None):
    regs = ",".join(f"${i}" for i in range(32))
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 32), [Int32(addr).ir_value(loc=loc, ip=ip)],
        "tcgen05.ld.sync.aligned.16x64b.x32.b32 {" + regs + "}, [$32];",
        ",".join(["=f"] * 32) + ",r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def tmem_ld_16x64b_x16(addr: Int32, *, loc=None, ip=None):
    regs = ",".join(f"${i}" for i in range(16))
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 16), [Int32(addr).ir_value(loc=loc, ip=ip)],
        "tcgen05.ld.sync.aligned.16x64b.x16.b32 {" + regs + "}, [$16];",
        ",".join(["=f"] * 16) + ",r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(16)]


@dsl_user_op
def tmem_ld32(addr: Int32, *, loc=None, ip=None):
    """
    tcgen05.ld.32x32b.x32: thread t of warp w gets TMEM lane (addr.lane + t), columns
    addr.col .. +31.
    """
    regs = ",".join(f"${i}" for i in range(32))
    res = inline_asm(llvm.StructType.get_literal([T.f32()] * 32), [Int32(addr).ir_value(loc=loc, ip=ip)],
               "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];", ",".join(["=f"] * 32) + ",r", True,
               loc, ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def mul_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """
    FMUL RN, never contracted with a following add.
    """
    return Float32(inline_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "mul.rn.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def add_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(inline_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "add.rn.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def abs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)], "abs.f32 $0, $1;", "=f,f", False, loc, ip))


@dsl_user_op
def max_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """
    fmaxf (IEEE maxNum: NaN-ignoring).
    """
    return Float32(inline_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
                        "max.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def e4m3x4(x0: Float32, x1: Float32, x2: Float32, x3: Float32, *, loc=None, ip=None) -> Int32:
    """
    cvt.rn.satfinite.e4m3x2.f32 x2: bytes [x0, x1, x2, x3] little-endian.
    """
    return Int32(inline_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3)],
                      "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
                      "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}",
                      "=r,f,f,f,f", False, loc, ip))


@dsl_user_op
def bf16x2(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Int32:
    """
    cvt.rn.bf16x2.f32: low half x0, high half x1.
    """
    return Int32(inline_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1)],
                      "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", False, loc, ip))


@dsl_user_op
def st_shared_f32(addr: Int32, v: Float32, *, loc=None, ip=None):
    inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
         "st.shared.f32 [$0], $1;", "r,f", True, loc, ip)


@dsl_user_op
def ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(inline_asm(T.f32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", True,
                        loc, ip))


@dsl_user_op
def ld_shared_v4(addr: Int32, *, loc=None, ip=None):
    res = inline_asm(llvm.StructType.get_literal([T.i32()] * 4), [Int32(addr).ir_value(loc=loc, ip=ip)],
               "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", True, loc, ip)
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def st_global_v4(addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    inline_asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
         "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", True, loc, ip)


@dsl_user_op
def st_global_v4f(addr: Int64, a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None):
    inline_asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
         "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", True, loc, ip)


@dsl_user_op
def st_global_f32(addr: Int64, v: Float32, *, loc=None, ip=None):
    inline_asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
         "st.global.f32 [$0], $1;", "l,f", True, loc, ip)


def global_address(t: cute.Tensor) -> Int64:
    return Int64(t.iterator.toint())
