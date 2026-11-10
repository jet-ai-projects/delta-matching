"""hd256 backward under the MXFP8 spec (reference preset "sm100-mx-d256";
Part B / B.3).  hd128 does not import this file.

One kernel class, three modes (1-CTA, tcgen05, no atomics, fixed loop orders -> bitwise deterministic; every output element
has exactly one writer CTA):

  mode "dv": CTA = (128-key tile, kv head, batch).  Keys on TMEM lanes.  For each q head of the group (ascending) and each
             128-query tile (ascending): S^T = K8 Q8^T (kind::f8f6f4, M=128 keys, N=128 queries, 8 K=32 steps ascending d,
             zero-init = the pass-0 instruction transposed: bitwise transposes),
             P = ex2(fma(S^T, c, -l2)), P^T codes pow2 per (key, 32 queries), SFA_dV = p_sb (+) dos_sb;
             dV += P^T-codes . dO8^T  (kind::mxf8f6f4 M=128 N=256 K=128, SFs in TMEM; fp32 accumulation in TMEM across
             all q heads and tiles; one bf16 rounding).
  mode "dk": as dv plus dP^T = V8 dO8^T; dS = P * fma(dP^T, a, -delta); one pow2 scale per (32 keys = this warp's lanes) x
             (128-query tile) (warp redux of the fp32 dS); SFA_dK = ds_sb (+) qs_sb; dK += dS^T-codes . Q8^T.
  mode "dq": CTA = (128-query tile, q head, batch).  Queries on lanes.  For each 128-key tile: S = Q8 K8^T, dP = dO8 V8^T
             (the pass-0 instruction and orientation), same P / dS line, one pow2 scale per (128-query tile) x (32 keys)
             (max over the 4 compute warps: order-free, so the codes / SFs are bitwise the dk mode's), SFA_dQ = ds_sb (+)
             ks_sb; dQ = sm * (dS-codes . K8^T)   (sm = 2^-4 exact).
Every site is the sm100-mx-d256 rule (= the validated hd128 sm100-mx rule at D=256):
  inputs: one pow2 scale per 32 tokens over all D (quant.quant_qkv2 / quant.quant_do_delta), one code set;
  P / dP / dS line: dm_common_hd256.dm_mx256_* (= hd128 dm_common.dm_bwd_*: c = (ks*qs)*log2e, l2 = lse*log2e,
  P = ex2.approx.ftz(fma(S_acc, c, -l2)), a = vs*dos, dS = P * fma(dP_acc, a, -delta));  delta = num/den from pass 0;
  quantization: pow2_sb / inv_pow2_from_sb / add_sb_clamp / cvt.rn.satfinite.e4m3 exactly as hd128 dm_bwd_sm100.

TMEM (512 cols): S 0-127, dP 128-255, out 256-511.  SFA (1 word per lane = 4 k-step bytes) at col 128, SFB (constant
0x7F = 1.0) at cols 132-135, written by the compute warps after their dP read; the MMA warp issues dP(t+1) only after
out(t) completed (od barrier), so the SF words are never overwritten while in use.
Warps: 0-3 compute (row = tidx), 4 MMA, 5 TMA.

dbg bit 1 (dq mode): per (row, 32-key group) [sum dS (fp32, pre-quant), sum |P*dP_acc*a|, sum P] -> mIdent.
dbg bit 2 (all modes): per (query, key) [S_acc, dP_acc, P, dS] -> mDump (L, N, N, 4); codes -> mCodes (L, N, N) indexed
  [query, key]; unadorned block exponent -> mSB (L, N, N/32) indexed [row token, col-32-block].
dbg bit 4 (dq mode only, "delta" sub-mode): no dS / dQ; per row num = sum P*dP (fp64, ascending keys), den = sum P (fp64)
  -> mIdent[l, row, 0, 0:2] (a test-only matched-delta pass over bitwise this kernel's P / dP).
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Float64, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from delta_matching.sm100 import dm_common_hd256 as dmh
from delta_matching.sm100 import dm_utils
from delta_matching.sm100.dm_common_hd256 import (tmem_ld32, e4m3x4, bf16x2, st_global_v4, st_global_v4f,
                                                 st_global_f32, ld_shared_f32, st_shared_f32, abs_f32, max_f32,
                                                 add_f32, mul_f32, _asm, _gaddr)
from delta_matching.sm100.dm_common_hd256 import ld_shared_v4 as dm_utils_ld_shared_v4

F8 = cutlass.Float8E4M3FN
SM = 0.0625          # 256 ** -0.5, exact


@dsl_user_op
def st_shared_v4(addr: Int32, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    _asm(None, [Int32(x).ir_value(loc=loc, ip=ip) for x in (addr, a, b, c, d)],
         "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", True, loc, ip)


@dsl_user_op
def ld_shared_v4f(addr: Int32, *, loc=None, ip=None):
    res = _asm(llvm.StructType.get_literal([T.f32()] * 4), [Int32(addr).ir_value(loc=loc, ip=ip)],
               "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", True, loc, ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def spin(cycles: int, *, loc=None, ip=None):
    """Busy-wait ~cycles SM clocks (positive-control delay knobs only; never in the default build)."""
    _asm(None, [], "{\n\t.reg .u64 t0, t1;\n\t.reg .pred p;\n\tmov.u64 t0, %clock64;\n\tLSPIN:\n\t"
         "mov.u64 t1, %clock64;\n\tsub.u64 t1, t1, t0;\n\tsetp.lt.u64 p, t1, " + str(int(cycles)) + ";\n\t"
         "@p bra LSPIN;\n\t}", "", True, loc, ip)


@dsl_user_op
def tmem_st32(addr: Int32, v, *, loc=None, ip=None):
    """tcgen05.st.32x32b.x32: this thread's lane, 32 fp32 columns at addr."""
    regs = ",".join(f"${i + 1}" for i in range(32))
    _asm(None, [Int32(addr).ir_value(loc=loc, ip=ip)] + [Float32(x).ir_value(loc=loc, ip=ip) for x in v],
         "tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {" + regs + "};", ",".join(["r"] + ["f"] * 32), True, loc, ip)


@dsl_user_op
def sb_of(x: Float32, *, loc=None, ip=None) -> Int32:
    """Biased exponent byte of an fp32 power of two (its UE8M0 code)."""
    return Int32(_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)],
                      "{\n\t.reg .b32 t;\n\tmov.b32 t, $1;\n\tbfe.u32 $0, t, 23, 8;\n\t}", "=r,f", False, loc, ip))


@dsl_user_op
def pack4(b0: Int32, b1: Int32, b2: Int32, b3: Int32, *, loc=None, ip=None) -> Int32:
    """bytes [b0, b1, b2, b3] little-endian (each < 256)."""
    return Int32(_asm(T.i32(), [Int32(x).ir_value(loc=loc, ip=ip) for x in (b0, b1, b2, b3)],
                      "{\n\t.reg .b32 t;\n\tshl.b32 t, $4, 8;\n\tor.b32 t, t, $3;\n\tshl.b32 t, t, 8;\n\tor.b32 t, t, $2;\n\t"
                      "shl.b32 t, t, 8;\n\tor.b32 $0, t, $1;\n\t}", "=r,r,r,r,r", False, loc, ip))


@dsl_user_op
def tc_fence_before(*, loc=None, ip=None):
    _asm(None, [], "tcgen05.fence::before_thread_sync;", "", True, loc, ip)


@dsl_user_op
def tc_fence_after(*, loc=None, ip=None):
    _asm(None, [], "tcgen05.fence::after_thread_sync;", "", True, loc, ip)


@dsl_user_op
def st_global_u8(addr: Int64, v: Int32, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
         "st.global.u8 [$0], $1;", "l,r", True, loc, ip)


@dsl_user_op
def st_global_f64x2(addr: Int64, a: Float64, b: Float64, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Float64(a).ir_value(loc=loc, ip=ip),
                Float64(b).ir_value(loc=loc, ip=ip)],
         "st.global.v2.f64 [$0], {$1, $2};", "l,d,d", True, loc, ip)


@dsl_user_op
def f2i(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f", False, loc, ip))


@dsl_user_op
def hmix(v: Int32, idx: Int32, *, loc=None, ip=None) -> Int32:
    """dbg 8 coupling hash of one element (value bits v at element index idx); summed (mod 2^32, order-free) per block."""
    return Int32(_asm(T.i32(), [Int32(v).ir_value(loc=loc, ip=ip), Int32(idx).ir_value(loc=loc, ip=ip)],
                      "{\n\t.reg .b32 x, y;\n\tmul.lo.u32 x, $1, 0x9E3779B1;\n\tmul.lo.u32 y, $2, 0x85EBCA6B;\n\t"
                      "add.u32 x, x, y;\n\tshr.u32 y, x, 15;\n\txor.b32 x, x, y;\n\tmul.lo.u32 x, x, 0x2C1B3C6D;\n\t"
                      "shr.u32 y, x, 12;\n\txor.b32 x, x, y;\n\tmul.lo.u32 x, x, 0x297A2D39;\n\tshr.u32 y, x, 15;\n\t"
                      "xor.b32 $0, x, y;\n\t}", "=r,r,r", False, loc, ip))


@dsl_user_op
def redux_add_u32(x: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [Int32(x).ir_value(loc=loc, ip=ip)], "redux.sync.add.u32 $0, $1, 0xffffffff;", "=r,r",
                      True, loc, ip))


@dsl_user_op
def zext64(x: Int32, *, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [Int32(x).ir_value(loc=loc, ip=ip)], "cvt.u64.u32 $0, $1;", "=l,r", False, loc, ip))


@dsl_user_op
def red_add_u64(addr: Int64, v: Int64, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int64(v).ir_value(loc=loc, ip=ip)],
         "red.global.add.u64 [$0], $1;", "l,l", True, loc, ip)


@dsl_user_op
def red_add_f64(addr: Int64, v: Float64, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Float64(v).ir_value(loc=loc, ip=ip)],
         "red.global.add.f64 [$0], $1;", "l,d", True, loc, ip)


@dsl_user_op
def bfly_sum_f64(x: Float64, *, loc=None, ip=None) -> Float64:
    """warp butterfly fp64 sum (fixed order per lane; dbg only)."""
    body = "{\n\t.reg .b32 lo, hi, olo, ohi;\n\t.reg .f64 a, o;\n\tmov.f64 a, $1;\n\t"
    for m in (16, 8, 4, 2, 1):
        body += ("mov.b64 {lo, hi}, a;\n\t"
                 f"shfl.sync.bfly.b32 olo, lo, {m}, 31, 0xffffffff;\n\tshfl.sync.bfly.b32 ohi, hi, {m}, 31, 0xffffffff;\n\t"
                 "mov.b64 o, {olo, ohi};\n\tadd.rn.f64 a, a, o;\n\t")
    body += "mov.f64 $0, a;\n\t}"
    return Float64(_asm(T.f64(), [Float64(x).ir_value(loc=loc, ip=ip)], body, "=d,d", True, loc, ip))


@dsl_user_op
def sel_eq_i32(x: Int32, y: Int32, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """a if x == y else b"""
    return Int32(_asm(T.i32(), [Int32(v).ir_value(loc=loc, ip=ip) for v in (x, y, a, b)],
                      "{\n\t.reg .pred p;\n\tsetp.eq.s32 p, $1, $2;\n\tselp.b32 $0, $3, $4, p;\n\t}", "=r,r,r,r,r",
                      False, loc, ip))


@dsl_user_op
def sel_eq_f64(x: Int32, y: Int32, a: Float64, b: Float64, *, loc=None, ip=None) -> Float64:
    return Float64(_asm(T.f64(), [Int32(x).ir_value(loc=loc, ip=ip), Int32(y).ir_value(loc=loc, ip=ip),
                                  Float64(a).ir_value(loc=loc, ip=ip), Float64(b).ir_value(loc=loc, ip=ip)],
                        "{\n\t.reg .pred p;\n\tsetp.eq.s32 p, $1, $2;\n\tselp.f64 $0, $3, $4, p;\n\t}", "=d,r,r,d,d",
                        False, loc, ip))


def lk_cols_u32(hl, lane, addr0):
    """dbg 8, keys-on-rows: per query column i of a 32-col chunk, the wrapping u32 sum over this warp's 32 key lanes,
    zero-extended and added (u64, order-free) at addr0 + 32 i (lane i adds column i)."""
    mine = Int32(0)
    for i in range(32):
        s_ = redux_add_u32(hl[i])
        mine = sel_eq_i32(lane, Int32(i), s_, mine)
    red_add_u64(addr0 + Int64(lane) * 32, zext64(mine))


def lk_cols_f64(vals32, lane, addr0):
    mine = Float64(0.0)
    for i in range(32):
        x_ = bfly_sum_f64(Float64(vals32[i]))
        mine = sel_eq_f64(lane, Int32(i), x_, mine)
    red_add_f64(addr0 + Int64(lane) * 32, mine)


def hidx(q, k):
    """element index for the dbg 8 hash (wraps mod 2^32; identical in every kernel)."""
    return q * 65599 + k


class DmHd256Mx:
    TM, TN, D = 128, 128, 256
    COL_DP, COL_SFA, COL_SFB, COL_OUT = 128, 128, 132, 256

    KNOBS = ("acc", "cd", "od", "tw", "fin", "ld", "am", "lsk")
    DLY = 1 << 15

    def __init__(self, mode: str, causal: bool, dbg: int = 0, knob: str = "", ordf: bool = False):
        """knob (positive controls of every TMEM / SMEM hand-off, ordering table; "" = shipped build):
        "<h>" = UNSAFE order for hand-off h + its delay, "<h>_safe" = shipped order + the same delay (must be bitwise
        the default build)."""
        assert mode in ("dv", "dk", "dq")
        # ordf (sites BW1-BW3, BW9): the tcgen05 thread-sync fence pair on every
        # cross-thread TMEM hand-off that lacks it (PTX 9.7.18.6.4.4: [ld; wait::ld; fence::before; arrive] ->
        # [wait; fence::after; tcgen05 op], and commit -> [wait; fence::after; tcgen05 op]). Host default True (DM_HD256_ORDF);
        # False = the code without the fence pair, debug only.
        self.ordf = bool(ordf)
        self.kn = knob.replace("_safe", "")
        self.unsafe = bool(knob) and not knob.endswith("_safe")
        assert self.kn in ("",) + self.KNOBS, knob
        self.mode = mode
        self.causal = bool(causal)
        self.dbg = int(dbg)
        self.delta_only = bool(self.dbg & 4)
        assert not self.delta_only or mode == "dq"
        self.has_dp = mode != "dv"
        self.rows_are_keys = mode != "dq"
        self.threads = 192

    @cute.jit
    def __call__(self, mXa: cute.Tensor, mXb: cute.Tensor, mYa: cute.Tensor, mYb: cute.Tensor, mT: cute.Tensor,
                 mQs: cute.Tensor, mKs: cute.Tensor, mVs: cute.Tensor, mdOs: cute.Tensor,
                 mLse: cute.Tensor, mDelta: cute.Tensor, mOut: cute.Tensor,
                 mIdent: cute.Tensor, mDump: cute.Tensor, mCodes: cute.Tensor, mSB: cute.Tensor, mHash: cute.Tensor,
                 n_t: Int32, H: Int32, Hk: Int32, stream):
        D = self.D
        s_tiler = (self.TM, self.TN, D)
        mma_s = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                                                   Float32, tcgen05.CtaGroup.ONE, s_tiler[:2])
        a_l = sm100_utils.make_smem_layout_a(mma_s, s_tiler, F8, 1)
        b_l = sm100_utils.make_smem_layout_b(mma_s, s_tiler, F8, 1)
        o_tiler = (self.TM, D, self.TN)
        # out MMAs: B = dO / Q / K read MN-major (d contiguous) straight from the natural (B, N, heads, 256) layout (F1)
        mma_o = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
                                                   Float32, tcgen05.CtaGroup.ONE, o_tiler[:2])
        c_l = sm100_utils.make_smem_layout_a(mma_o, o_tiler, F8, 1)
        t_l = sm100_utils.make_smem_layout_b(mma_o, o_tiler, F8, 1)
        a1 = cute.select(a_l, mode=[0, 1, 2])
        b1 = cute.select(b_l, mode=[0, 1, 2])
        t1 = cute.select(t_l, mode=[0, 1, 2])
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma_s.thr_id,))
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_Xa, tXa = cute.nvgpu.make_tiled_tma_atom_A(op, mXa, a1, s_tiler, mma_s, vmnk.shape)
        tma_Xb, tXb = cute.nvgpu.make_tiled_tma_atom_A(op, mXb, a1, s_tiler, mma_s, vmnk.shape)
        tma_Ya, tYa = cute.nvgpu.make_tiled_tma_atom_B(op, mYa, b1, s_tiler, mma_s, vmnk.shape)
        tma_Yb, tYb = cute.nvgpu.make_tiled_tma_atom_B(op, mYb, b1, s_tiler, mma_s, vmnk.shape)
        tma_T, tT = cute.nvgpu.make_tiled_tma_atom_B(op, mT, t1, o_tiler, mma_o, vmnk.shape)
        self.bytes_a = cute.size_in_bytes(F8, a1)
        self.bytes_b = cute.size_in_bytes(F8, b1)
        self.bytes_t = cute.size_in_bytes(F8, t1)
        nx = 2 if const_expr(self.has_dp) else 1

        @cute.struct
        class SS:
            bars: cute.struct.MemRange[cutlass.Int64, 24]
            tmem_hold: Int32
            amax: cute.struct.Align[cute.struct.MemRange[Float32, 32], 16]
            sLD: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 2 * 128], 16]
            sXa: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sXb: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l) if nx == 2 else 1024], 1024]
            sYa: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l)], 1024]
            sYb: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l) if nx == 2 else 1024], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(t_l)], 1024]
            sC: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(c_l)], 1024]
        self.SS = SS
        nh = Hk if const_expr(self.rows_are_keys) else H
        self.kernel(mma_s, mma_o, tma_Xa, tXa, tma_Xb, tXb, tma_Ya, tYa, tma_Yb, tYb, tma_T, tT, a_l, b_l, c_l, t_l,
                    mQs, mKs, mVs, mdOs, mLse, mDelta, mOut, mIdent, mDump, mCodes, mSB, mHash, n_t, H, Hk).launch(
            grid=[n_t, nh, cute.size(mXa, mode=[3])], block=[self.threads, 1, 1], smem=SS.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, mma_s: cute.TiledMma, mma_o: cute.TiledMma, tma_Xa: cute.CopyAtom, mXa: cute.Tensor,
               tma_Xb: cute.CopyAtom, mXb: cute.Tensor, tma_Ya: cute.CopyAtom, mYa: cute.Tensor, tma_Yb: cute.CopyAtom,
               mYb: cute.Tensor, tma_T: cute.CopyAtom, mT: cute.Tensor, a_l: cute.ComposedLayout,
               b_l: cute.ComposedLayout, c_l: cute.ComposedLayout, t_l: cute.ComposedLayout,
               mQs: cute.Tensor, mKs: cute.Tensor, mVs: cute.Tensor, mdOs: cute.Tensor, mLse: cute.Tensor,
               mDelta: cute.Tensor, mOut: cute.Tensor, mIdent: cute.Tensor, mDump: cute.Tensor, mCodes: cute.Tensor,
               mSB: cute.Tensor, mHash: cute.Tensor, n_t: Int32, H: Int32, Hk: Int32):
        D = self.D
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        rt, hx, b = cute.arch.block_idx()
        rep = H // Hk
        N = n_t * 128
        # row head (X operands) and column heads (Y operands / twin): keys-on-rows: X = kv head hx, loop q heads of the group
        nhl = Int32(1)
        if const_expr(self.rows_are_keys):
            nhl = rep
        c_lo = Int32(0)
        c_hi = n_t
        if const_expr(self.causal):
            if const_expr(self.rows_are_keys):
                c_lo = rt
            else:
                c_hi = rt + 1
        ncs = c_hi - c_lo
        total = nhl * ncs

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.SS)
        sXa = st.sXa.get_tensor(a_l.outer, swizzle=a_l.inner)
        sXb = st.sXb.get_tensor(a_l.outer, swizzle=a_l.inner)
        sYa = st.sYa.get_tensor(b_l.outer, swizzle=b_l.inner)
        sYb = st.sYb.get_tensor(b_l.outer, swizzle=b_l.inner)
        sT = st.sT.get_tensor(t_l.outer, swizzle=t_l.inner)
        sC = st.sC.get_tensor(c_l.outer, swizzle=c_l.inner)
        sC_addr = Int32(st.sC.data_ptr().toint())
        sLD_addr = Int32(st.sLD.data_ptr().toint())
        amax_addr = Int32(st.amax.data_ptr().toint())
        bars = st.bars.data_ptr()
        od_bar = bars + 20

        if warp == 5:
            cpasync.prefetch_descriptor(tma_Xa)
            cpasync.prefetch_descriptor(tma_Ya)
            cpasync.prefetch_descriptor(tma_T)
            if const_expr(self.has_dp):
                cpasync.prefetch_descriptor(tma_Xb)
                cpasync.prefetch_descriptor(tma_Yb)
        nx = 2 if const_expr(self.has_dp) else 1
        thr1 = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        thr128 = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        x_p, x_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                   tx_count=nx * self.bytes_a, barrier_storage=bars).make_participants()
        y_p, y_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                   tx_count=nx * self.bytes_b, barrier_storage=bars + 2).make_participants()
        t_p, t_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                   tx_count=self.bytes_t, barrier_storage=bars + 4).make_participants()
        acc_p, acc_c = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=thr1, consumer_group=thr128,
                                                         barrier_storage=bars + 6).make_participants()
        cd_p, cd_c = pipeline.PipelineAsyncUmma.create(num_stages=1, producer_group=thr128, consumer_group=thr1,
                                                       barrier_storage=bars + 8).make_participants()
        fin_p, fin_c = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=thr1, consumer_group=thr128,
                                                         barrier_storage=bars + 10).make_participants()
        if warp == 4:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(od_bar, 1)
            cute.arch.mbarrier_init_fence()

        tmem = utils.TmemAllocator(st.tmem_hold.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads))
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tmem_base = Int32(tmem_ptr.toint())

        thr_s = mma_s.get_slice(0)
        acc_l = mma_s.make_fragment_C(mma_s.partition_shape_C((self.TM, self.TN))).layout
        tCrXa = mma_s.make_fragment_A(sXa)
        tCrXb = mma_s.make_fragment_A(sXb)
        tCrYa = mma_s.make_fragment_B(sYa)
        tCrYb = mma_s.make_fragment_B(sYb)
        tCrC = mma_o.make_fragment_A(sC)
        tCrT = mma_o.make_fragment_B(sT)

        # ------------------------------------------------------------------ TMA warp
        if warp == 5:
            hy0 = hx            # dq: column operand head = kv head of q head hx
            if const_expr(not self.rows_are_keys):
                hy0 = hx // rep
            gXa = cute.local_tile(mXa[None, None, hx, b], (self.TM, D), (None, 0))
            tXas, tXag = cpasync.tma_partition(tma_Xa, 0, cute.make_layout(1), cute.group_modes(sXa, 0, 3),
                                               cute.group_modes(thr_s.partition_A(gXa), 0, 3))
            xe = x_p.acquire_and_advance()
            cute.copy(tma_Xa, tXag[(None, rt)], tXas[(None, 0)], tma_bar_ptr=xe.barrier)
            if const_expr(self.has_dp):
                gXb = cute.local_tile(mXb[None, None, hx, b], (self.TM, D), (None, 0))
                tXbs, tXbg = cpasync.tma_partition(tma_Xb, 0, cute.make_layout(1), cute.group_modes(sXb, 0, 3),
                                                   cute.group_modes(thr_s.partition_A(gXb), 0, 3))
                cute.copy(tma_Xb, tXbg[(None, rt)], tXbs[(None, 0)], tma_bar_ptr=xe.barrier)
            thr_o = mma_o.get_slice(0)
            for hi in cutlass.range(nhl, unroll=1):
                hy = hy0
                if const_expr(self.rows_are_keys):
                    hy = hx * rep + hi
                gYa = cute.local_tile(mYa[None, None, hy, b], (self.TN, D), (None, 0))
                tYas, tYag = cpasync.tma_partition(tma_Ya, 0, cute.make_layout(1), cute.group_modes(sYa, 0, 3),
                                                   cute.group_modes(thr_s.partition_B(gYa), 0, 3))
                gYb = cute.local_tile(mYb[None, None, hy, b], (self.TN, D), (None, 0))
                tYbs, tYbg = cpasync.tma_partition(tma_Yb, 0, cute.make_layout(1), cute.group_modes(sYb, 0, 3),
                                                   cute.group_modes(thr_s.partition_B(gYb), 0, 3))
                nyh = H
                if const_expr(not self.rows_are_keys):
                    nyh = Hk
                gT = cute.local_tile(mT[None, None, hy, b], (D, self.TN), (0, None))
                tTs, tTg = cpasync.tma_partition(tma_T, 0, cute.make_layout(1), cute.group_modes(sT, 0, 3),
                                                 cute.group_modes(thr_o.partition_B(gT), 0, 3))
                for ci in cutlass.range(ncs, unroll=1):
                    ct = c_lo + ci
                    ye = y_p.acquire_and_advance()
                    cute.copy(tma_Ya, tYag[(None, ct)], tYas[(None, 0)], tma_bar_ptr=ye.barrier)
                    if const_expr(self.has_dp):
                        cute.copy(tma_Yb, tYbg[(None, ct)], tYbs[(None, 0)], tma_bar_ptr=ye.barrier)
                    if const_expr(not self.delta_only):
                        te = t_p.acquire_and_advance()
                        cute.copy(tma_T, tTg[(None, ct)], tTs[(None, 0)], tma_bar_ptr=te.barrier)
            y_p.tail()
            if const_expr(not self.delta_only):
                t_p.tail()

        # ------------------------------------------------------------------ MMA warp
        if warp == 4:
            x_c.wait_and_advance()
            mma_op = cute.make_mma_atom(mma_o.op).op
            accS = cute.make_tensor(tmem_ptr, acc_l)
            accP = cute.make_tensor(tmem_ptr + self.COL_DP, acc_l)
            # prologue: S(0), dP(0)
            yf = y_c.wait_and_advance()
            ae = acc_p.acquire_and_advance()
            if const_expr(self.ordf):
                tc_fence_after()
            mma_s.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(cute.size(tCrXa, mode=[2])):
                cute.gemm(mma_s, accS, tCrXa[(None, None, kb, 0)], tCrYa[(None, None, kb, 0)], accS)
                mma_s.set(tcgen05.Field.ACCUMULATE, True)
            if const_expr(self.has_dp):
                mma_s.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(cute.size(tCrXb, mode=[2])):
                    cute.gemm(mma_s, accP, tCrXb[(None, None, kb, 0)], tCrYb[(None, None, kb, 0)], accP)
                    mma_s.set(tcgen05.Field.ACCUMULATE, True)
            yf.release()
            ae.commit()
            for t in cutlass.range(total, unroll=1):
                if t + 1 < total:
                    # S(t+1): its TMEM cols were released by the compute warps (acc) after their S/dP(t) reads
                    yn = y_c.wait_and_advance()
                    an = acc_p.acquire_and_advance()
                    if const_expr(self.ordf):
                        tc_fence_after()     # BW2: compute's S/dP(t) loads (wait::ld + fence::before + acc release) -> S(t+1)
                    mma_s.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(cute.size(tCrXa, mode=[2])):
                        cute.gemm(mma_s, accS, tCrXa[(None, None, kb, 0)], tCrYa[(None, None, kb, 0)], accS)
                        mma_s.set(tcgen05.Field.ACCUMULATE, True)
                    if const_expr(self.has_dp and (self.delta_only or (self.kn == "od" and self.unsafe))):
                        # (od UNSAFE: dP(t+1) issued before out(t) -> clobbers out(t)'s SF words)
                        mma_s.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(cute.size(tCrXb, mode=[2])):
                            cute.gemm(mma_s, accP, tCrXb[(None, None, kb, 0)], tCrYb[(None, None, kb, 0)], accP)
                            mma_s.set(tcgen05.Field.ACCUMULATE, True)
                    if const_expr(not self.has_dp):
                        # dv: S(t+1) is all the compute warps need; out(t)'s codes / SFs are guarded by cd
                        yn.release()
                        an.commit()
                    if const_expr(not self.delta_only):
                        if const_expr(self.kn in ("cd", "fin", "od")):
                            spin(self.DLY)
                        tf = t_c.wait_and_advance()
                        cf = cd_c.wait_and_advance()
                        tc_fence_after()
                        if const_expr(self.kn == "tw"):
                            if const_expr(self.unsafe):
                                tf.release()
                            spin(self.DLY)
                        dm_utils.gemm_ptx_bs(
                            mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                            sC[(None, None, None, 0)], sT[(None, None, None, 0)],
                            tmem_base + self.COL_SFA, tmem_base + self.COL_SFB,
                            tuple((0, k) for k in range(4)), tuple((0, 0) for k in range(4)),
                            zero_init=cutlass.Boolean(t == 0))
                        if const_expr(not (self.kn == "tw" and self.unsafe)):
                            tf.release()
                        cf.release()
                    if const_expr(self.has_dp and not self.delta_only and not (self.kn == "od" and self.unsafe)):
                        # dP(t+1) overwrites the SF words of out(t): wait until out(t) has completed
                        with cute.arch.elect_one():
                            tcgen05.commit(od_bar)
                        cute.arch.mbarrier_wait(od_bar, t & 1)
                        tc_fence_after()
                        mma_s.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(cute.size(tCrXb, mode=[2])):
                            cute.gemm(mma_s, accP, tCrXb[(None, None, kb, 0)], tCrYb[(None, None, kb, 0)], accP)
                            mma_s.set(tcgen05.Field.ACCUMULATE, True)
                    if const_expr(self.has_dp):
                        yn.release()
                        an.commit()
                else:
                    if const_expr(not self.delta_only):
                        if const_expr(self.kn in ("cd", "fin", "od")):
                            spin(self.DLY)
                        tf = t_c.wait_and_advance()
                        cf = cd_c.wait_and_advance()
                        tc_fence_after()
                        if const_expr(self.kn == "tw"):
                            if const_expr(self.unsafe):
                                tf.release()
                            spin(self.DLY)
                        dm_utils.gemm_ptx_bs(
                            mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                            sC[(None, None, None, 0)], sT[(None, None, None, 0)],
                            tmem_base + self.COL_SFA, tmem_base + self.COL_SFB,
                            tuple((0, k) for k in range(4)), tuple((0, 0) for k in range(4)),
                            zero_init=cutlass.Boolean(t == 0))
                        if const_expr(not (self.kn == "tw" and self.unsafe)):
                            tf.release()
                        cf.release()
            fe = fin_p.acquire_and_advance()
            fe.commit()
            acc_p.tail()
            fin_p.tail()

        # ------------------------------------------------------------------ compute warps
        if warp < 4:
            r = tidx
            lane_off = Int32((warp * 32) << 16)
            rtok = rt * 128 + r
            tb = tmem_base + lane_off
            hk_row = hx
            if const_expr(not self.rows_are_keys):
                hk_row = hx // rep
            # row-constant quantities
            row_s = Float32(0.0)     # keys-on-rows: ks of the key; dq: qs of the query
            row_a = Float32(0.0)     # keys-on-rows: vs; dq: dos
            l2r = Float32(0.0)
            dlr = Float32(0.0)
            if const_expr(self.rows_are_keys):
                row_s = Float32(mKs[b, hx, rtok // 32])
                row_a = Float32(mVs[b, hx, rtok // 32])
            else:
                row_s = Float32(mQs[b, hx, rtok // 32])
                row_a = Float32(mdOs[b, hx, rtok // 32])
                l2r = dmh.dm_mx256_l2(Float32(mLse[b, hx, rtok]))
                dlr = Float32(mDelta[b, hx, rtok])
            one = Int32(0x7F7F7F7F)
            num = Float64(0.0)
            den = Float64(0.0)
            idn = Float64(0.0)
            idd = Float64(0.0)
            idp = Float64(0.0)
            lk_p = Int64(0)
            lk_s = Int64(0)
            lk_c = Int64(0)
            lk_r = Float64(0.0)
            # codes smem: row r, 16-byte chunk v of the 128-byte K row at SW128 K-major position
            crow = sC_addr + (r // 8) * 1024 + (r % 8) * 128
            rsw = r % 8
            for t in cutlass.range(total, unroll=1):
                hi = t // ncs
                ct = c_lo + t - hi * ncs
                h = hx                   # q head of this iteration
                if const_expr(self.rows_are_keys):
                    h = hx * rep + hi
                l = b * H + h
                par = t & 1
                ctok0 = ct * 128
                cs = [Float32(0.0)] * 4
                ca = [Float32(0.0)] * 4
                osb = [Int32(0)] * 4     # the other operand's sb folded into SFA
                for j in cutlass.range_constexpr(4):
                    if const_expr(self.rows_are_keys):
                        qsj = Float32(mQs[b, h, ct * 4 + j])
                        dosj = Float32(mdOs[b, h, ct * 4 + j])
                        cs[j] = dmh.dm_mx256_c(row_s, qsj)
                        ca[j] = dmh.dm_mx256_dp_scale(row_a, dosj)
                        if const_expr(self.mode == "dv"):
                            osb[j] = sb_of(dosj)
                        else:
                            osb[j] = sb_of(qsj)
                    else:
                        ksj = Float32(mKs[b, hk_row, ct * 4 + j])
                        vsj = Float32(mVs[b, hk_row, ct * 4 + j])
                        cs[j] = dmh.dm_mx256_c(ksj, row_s)
                        ca[j] = dmh.dm_mx256_dp_scale(vsj, row_a)
                        osb[j] = sb_of(ksj)
                ldb = sLD_addr + par * 1024
                if const_expr(self.kn == "ld" and self.unsafe):
                    ldb = sLD_addr
                if const_expr(self.rows_are_keys):
                    # stage l2 / delta of the 128 queries of this tile (parity buffers; one barrier per tile)
                    st_shared_f32(ldb + r * 4, dmh.dm_mx256_l2(Float32(mLse[b, h, ctok0 + r])))
                    st_shared_f32(ldb + 512 + r * 4, Float32(mDelta[b, h, ctok0 + r]))
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    if const_expr(self.kn == "ld"):
                        if warp == 3:
                            spin(self.DLY)
                sbs = [Int32(0)] * 4
                ce = None
                if const_expr(self.mode == "dv" and not (self.kn == "cd" and self.unsafe)):
                    ce = cd_p.acquire_and_advance()
                    if const_expr(self.ordf):
                        tc_fence_after()     # BW3: out(t-1) SF reads (tcgen05.commit -> cd empty) -> SF tcgen05.st
                af = acc_c.wait_and_advance()
                if const_expr(self.ordf):
                    tc_fence_after()         # BW1: S/dP(t) commit -> acc full -> tcgen05.ld
                if const_expr(self.kn == "acc"):
                    if const_expr(self.unsafe):
                        af.release()
                    spin(self.DLY)
                stash = tb + self.COL_DP           # fp32 dS of chunk j -> cols 128 + 32 j (its dP already read)
                amk = Float32(0.0)                 # dk: running |dS| max of this row
                amq = [Float32(0.0)] * 4           # dq: per-chunk |dS| max of this row
                for j in cutlass.range_constexpr(4):
                    sv = tmem_ld32(tb + 32 * j)
                    dv_ = [Float32(0.0)] * 32
                    if const_expr(self.has_dp):
                        dv_ = tmem_ld32(tb + self.COL_DP + 32 * j)
                    cute.arch.fence_view_async_tmem_load()
                    if const_expr(j == 3 and not (self.dbg & 2) and not (self.kn == "acc" and self.unsafe)):
                        if const_expr(self.ordf):
                            tc_fence_before()    # BW2
                        af.release()
                    vals = cute.make_rmem_tensor(cute.make_layout(32), Float32)   # P (dv) or dS (dk / dq) of chunk j
                    id1 = Float32(0.0)
                    id2 = Float32(0.0)
                    id3 = Float32(0.0)
                    hpl = [Int32(0)] * 32
                    for i in cutlass.range_constexpr(0, 32, 4):
                        l2v = [l2r] * 4
                        dlv = [dlr] * 4
                        if const_expr(self.rows_are_keys):
                            l2v = ld_shared_v4f(ldb + (32 * j + i) * 4)
                            dlv = ld_shared_v4f(ldb + 512 + (32 * j + i) * 4)
                        pv = [None] * 4
                        for q2 in cutlass.range_constexpr(0, 4, 2):
                            s0 = sv[i + q2]
                            s1 = sv[i + q2 + 1]
                            if const_expr(self.causal):
                                c0 = ctok0 + 32 * j + i + q2
                                if const_expr(self.rows_are_keys):
                                    s0 = -Float32.inf if rtok > c0 else s0
                                    s1 = -Float32.inf if rtok > c0 + 1 else s1
                                else:
                                    s0 = -Float32.inf if c0 > rtok else s0
                                    s1 = -Float32.inf if c0 + 1 > rtok else s1
                            if const_expr(self.rows_are_keys):
                                p0, p1 = dmh.dm_mx256_p_pair2(s0, s1, cs[j], l2v[q2], l2v[q2 + 1])
                            else:
                                p0, p1 = dmh.dm_mx256_p_pair(s0, s1, cs[j], l2r)
                            pv[q2] = p0
                            pv[q2 + 1] = p1
                        if const_expr((self.dbg & 8) and self.mode != "dk"):
                            for q in cutlass.range_constexpr(4):
                                ce_ = ctok0 + 32 * j + i + q
                                if const_expr(self.rows_are_keys):
                                    hpl[i + q] = hmix(f2i(pv[q]), hidx(ce_, rtok))
                                else:
                                    hpl[i + q] = hmix(f2i(pv[q]), hidx(rtok, ce_))
                        for q2 in cutlass.range_constexpr(0, 4, 2):
                            if const_expr(self.mode == "dv"):
                                vals[i + q2] = pv[q2]
                                vals[i + q2 + 1] = pv[q2 + 1]
                            elif const_expr(self.delta_only):
                                for u in cutlass.range_constexpr(2):
                                    pd = Float64(pv[q2 + u])
                                    num = num + pd * (Float64(dv_[i + q2 + u]) * Float64(ca[j]))
                                    den = den + pd
                            else:
                                x0, x1 = cute.arch.fma_packed_f32x2((dv_[i + q2], dv_[i + q2 + 1]), (ca[j], ca[j]),
                                                                    (-dlv[q2], -dlv[q2 + 1]))
                                x0, x1 = cute.arch.mul_packed_f32x2((pv[q2], pv[q2 + 1]), (x0, x1))
                                vals[i + q2] = x0
                                vals[i + q2 + 1] = x1
                        if const_expr((self.dbg & 1) and self.mode == "dq" and not self.delta_only):
                            for q in cutlass.range_constexpr(4):
                                id1 = add_f32(id1, vals[i + q])
                                id2 = add_f32(id2, abs_f32(mul_f32(pv[q], mul_f32(dv_[i + q], ca[j]))))
                                id3 = add_f32(id3, pv[q])
                    if const_expr((self.dbg & 8) and self.mode != "dk"):
                        if const_expr(self.rows_are_keys):
                            lk_cols_u32(hpl, lane, _gaddr(mHash) + (Int64(l) * N + ctok0 + 32 * j) * 32)
                        else:
                            hs_ = Int32(0)
                            for i in cutlass.range_constexpr(32):
                                hs_ = hs_ + hpl[i]
                            lk_p = lk_p + zext64(hs_)
                    if const_expr((self.dbg & 2) and self.has_dp and not self.delta_only):
                        qi0 = rtok
                        ki0 = ctok0 + 32 * j
                        stp = Int64(16)
                        if const_expr(self.rows_are_keys):
                            qi0 = ctok0 + 32 * j
                            ki0 = rtok
                            stp = Int64(N) * 16
                        pa = _gaddr(mDump) + ((Int64(l) * N + qi0) * N + ki0) * 16 + 4
                        for i in cutlass.range_constexpr(32):     # raw dP_acc (its TMEM cols get the dS stash)
                            st_global_f32(pa, dv_[i])
                            pa = pa + stp
                    if const_expr(self.mode == "dv"):
                        # P^T codes of this (key, 32-query) block right away
                        am = Float32(0.0)
                        for i in cutlass.range_constexpr(32):
                            am = cute.arch.fmax(am, vals[i])
                        sbs[j] = dm_utils.pow2_sb(am)
                        inv = dm_utils.inv_pow2_from_sb(sbs[j])
                        for hv in cutlass.range_constexpr(2):
                            w = [None] * 4
                            for u in cutlass.range_constexpr(4):
                                i0 = 16 * hv + 4 * u
                                y0, y1 = cute.arch.mul_packed_f32x2((vals[i0], vals[i0 + 1]), (inv, inv))
                                y2, y3 = cute.arch.mul_packed_f32x2((vals[i0 + 2], vals[i0 + 3]), (inv, inv))
                                w[u] = e4m3x4(y0, y1, y2, y3)
                            st_shared_v4(crow + (((2 * j + hv) ^ rsw) * 16), w[0], w[1], w[2], w[3])
                    elif const_expr(not self.delta_only):
                        am = Float32(0.0)
                        for i in cutlass.range_constexpr(32):
                            am = cute.arch.fmax(am, abs_f32(vals[i]))
                        if const_expr(self.mode == "dk"):
                            amk = cute.arch.fmax(amk, am)
                        else:
                            amq[j] = am
                        tmem_st32(stash + 32 * j, [vals[i] for i in range(32)])
                    if const_expr((self.dbg & 1) and self.mode == "dq" and not self.delta_only):
                        idn = idn + Float64(id1)
                        idd = idd + Float64(id2)
                        idp = idp + Float64(id3)
                if const_expr(self.has_dp and not self.delta_only):
                    cute.arch.fence_view_async_tmem_store()
                if const_expr(self.dbg & 2):
                    # raw dump pass (dbg only): re-read S, recompute P with the same helper, dS from the stash
                    for j in cutlass.range_constexpr(4):
                        sv = tmem_ld32(tb + 32 * j)
                        dsj = [Float32(0.0)] * 32
                        if const_expr(self.has_dp and not self.delta_only):
                            dsj = tmem_ld32(stash + 32 * j)
                        cute.arch.fence_view_async_tmem_load()
                        dstp = Int64(16)
                        dpa = _gaddr(mDump) + ((Int64(l) * N + rtok) * N + ctok0 + 32 * j) * 16
                        if const_expr(self.rows_are_keys):
                            dstp = Int64(N) * 16
                            dpa = _gaddr(mDump) + ((Int64(l) * N + ctok0 + 32 * j) * N + rtok) * 16
                        for i in cutlass.range_constexpr(0, 32, 2):
                            c0 = ctok0 + 32 * j + i
                            s0 = sv[i]
                            s1 = sv[i + 1]
                            if const_expr(self.causal):
                                if const_expr(self.rows_are_keys):
                                    s0 = -Float32.inf if rtok > c0 else s0
                                    s1 = -Float32.inf if rtok > c0 + 1 else s1
                                else:
                                    s0 = -Float32.inf if c0 > rtok else s0
                                    s1 = -Float32.inf if c0 + 1 > rtok else s1
                            if const_expr(self.mode == "dk"):
                                p0, p1 = Float32(0.0), Float32(0.0)     # dk dump: S_acc, dP_acc, dS (P covered by dv / dq)
                            elif const_expr(self.rows_are_keys):
                                p0, p1 = dmh.dm_mx256_p_pair2(s0, s1, cs[j], ld_shared_f32(ldb + (32 * j + i) * 4),
                                                              ld_shared_f32(ldb + (32 * j + i + 1) * 4))
                            else:
                                p0, p1 = dmh.dm_mx256_p_pair(s0, s1, cs[j], l2r)
                            pp = [p0, p1]
                            for u in cutlass.range_constexpr(2):
                                st_global_f32(dpa, sv[i + u])
                                st_global_f32(dpa + 8, pp[u])
                                st_global_f32(dpa + 12, dsj[i + u])
                                dpa = dpa + dstp
                    if const_expr(self.ordf):
                        tc_fence_before()
                    af.release()
                if const_expr(self.delta_only):
                    pass
                else:
                    # ---- block scales (unadorned sb per 32-col chunk) ----
                    if const_expr(self.mode == "dk"):
                        sb = dm_utils.pow2_sb(dm_utils.redux_max_abs_f32(amk))
                        sbs = [sb] * 4
                    elif const_expr(self.mode == "dq"):
                        slot = amax_addr + par * 64
                        for j in cutlass.range_constexpr(4):
                            am = dm_utils.redux_max_abs_f32(amq[j])
                            if const_expr(self.kn == "am" and j == 0):
                                if warp == 0:
                                    spin(self.DLY)
                            if lane == 0:
                                st_shared_f32(slot + (j * 4 + warp) * 4, am)
                        if const_expr(not (self.kn == "am" and self.unsafe)):    # am UNSAFE: no barrier
                            cute.arch.barrier(barrier_id=2, number_of_threads=128)
                        for j in cutlass.range_constexpr(4):
                            m4 = ld_shared_v4f(slot + j * 16)
                            am = max_f32(max_f32(m4[0], m4[1]), max_f32(m4[2], m4[3]))
                            sbs[j] = dm_utils.pow2_sb(am)
                    # ---- codes -> smem (after out(t-1) consumed the previous codes / SFs) ----
                    if const_expr(self.mode != "dv" and not (self.kn == "cd" and self.unsafe)):
                        ce = cd_p.acquire_and_advance()
                        if const_expr(self.ordf):
                            tc_fence_after()     # BW3
                    for j in cutlass.range_constexpr(4):
                        if const_expr(self.mode != "dv"):
                            inv = dm_utils.inv_pow2_from_sb(sbs[j])
                            dsj = tmem_ld32(stash + 32 * j)
                            cute.arch.fence_view_async_tmem_load()
                            hsl = [Int32(0)] * 32
                            hcl = [Int32(0)] * 32
                            for hv in cutlass.range_constexpr(2):
                                w = [None] * 4
                                for u in cutlass.range_constexpr(4):
                                    i0 = 16 * hv + 4 * u
                                    y0, y1 = cute.arch.mul_packed_f32x2((dsj[i0], dsj[i0 + 1]), (inv, inv))
                                    y2, y3 = cute.arch.mul_packed_f32x2((dsj[i0 + 2], dsj[i0 + 3]), (inv, inv))
                                    w[u] = e4m3x4(y0, y1, y2, y3)
                                if const_expr(self.kn == "lsk" and self.unsafe and self.mode == "dk"):
                                    # planted control: flip the low bit of ONE dS code (b 0, kv head 0, key 0, first tile)
                                    if const_expr(hv == 0 and j == 0):
                                        w[0] = w[0] ^ sel_eq_i32(t + rtok + hx + b, Int32(0), Int32(1), Int32(0))
                                st_shared_v4(crow + (((2 * j + hv) ^ rsw) * 16), w[0], w[1], w[2], w[3])
                                if const_expr(self.dbg & 8):
                                    for u in cutlass.range_constexpr(4):
                                        for q in cutlass.range_constexpr(4):
                                            ce_ = ctok0 + 32 * j + 16 * hv + 4 * u + q
                                            ix = hidx(rtok, ce_)
                                            if const_expr(self.rows_are_keys):
                                                ix = hidx(ce_, rtok)
                                            hsl[16 * hv + 4 * u + q] = hmix(f2i(dsj[16 * hv + 4 * u + q]), ix)
                                            hcl[16 * hv + 4 * u + q] = hmix(((w[u] >> (8 * q)) & 255) | (sbs[j] << 8), ix)
                            if const_expr(self.dbg & 8):
                                if const_expr(self.rows_are_keys):
                                    a0_ = _gaddr(mHash) + (Int64(l) * N + ctok0 + 32 * j) * 32
                                    lk_cols_u32(hsl, lane, a0_ + 8)
                                    lk_cols_u32(hcl, lane, a0_ + 16)
                                    lk_cols_f64(dsj, lane, a0_ + 24)
                                else:
                                    hs_ = Int32(0)
                                    hc_ = Int32(0)
                                    rs_ = Float64(0.0)
                                    for i in cutlass.range_constexpr(32):
                                        hs_ = hs_ + hsl[i]
                                        hc_ = hc_ + hcl[i]
                                        rs_ = rs_ + Float64(dsj[i])
                                    lk_s = lk_s + zext64(hs_)
                                    lk_c = lk_c + zext64(hc_)
                                    lk_r = lk_r + rs_
                        if const_expr(self.dbg & 2):
                            mSB[l, rtok, ct * 4 + j] = sbs[j]
                    if const_expr(self.dbg & 2):
                        # codes dump from the own smem row (after all chunks are written)
                        stc = Int64(1)
                        ca0 = _gaddr(mCodes) + (Int64(l) * N + rtok) * N + ctok0
                        if const_expr(self.rows_are_keys):
                            stc = Int64(N)
                            ca0 = _gaddr(mCodes) + (Int64(l) * N + ctok0) * N + rtok
                        for v in cutlass.range(8, unroll=1):
                            x4 = dm_utils_ld_shared_v4(crow + ((v ^ rsw) * 16))
                            cp = ca0 + Int64(16 * v) * stc
                            for u in cutlass.range_constexpr(4):
                                for q in cutlass.range_constexpr(4):
                                    st_global_u8(cp, x4[u] >> (8 * q))
                                    cp = cp + stc
                    sfa = pack4(dm_utils.add_sb_clamp(sbs[0], osb[0]), dm_utils.add_sb_clamp(sbs[1], osb[1]),
                                dm_utils.add_sb_clamp(sbs[2], osb[2]), dm_utils.add_sb_clamp(sbs[3], osb[3]))
                    # SFA (M=128): row r at lane r, column COL_SFA + r // 32;
                    # SFB (N=256): column COL_SFB + n // 32, lanes 32 q + n % 32 -> 8 columns of 1.0 (0x7F)
                    dm_utils.tmem_st_x1(tb + self.COL_SFA + warp, sfa)
                    dm_utils.tmem_st_x4(tb + self.COL_SFB, one, one, one, one)
                    dm_utils.tmem_st_x4(tb + self.COL_SFB + 4, one, one, one, one)
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_view_async_shared()
                    tc_fence_before()
                    if const_expr(self.kn == "cd" and self.unsafe):
                        ce = cd_p.acquire_and_advance()     # UNSAFE: codes / SFs were written before the slot was free
                    ce.commit()
            if const_expr(self.delta_only):
                fw = fin_c.wait_and_advance()
                fw.release()
                st_global_f64x2(_gaddr(mIdent) + (Int64(b * H + hx) * N + rtok) * 16, num, den)
            else:
                if const_expr((self.dbg & 8) and self.mode == "dq"):
                    ha = _gaddr(mHash) + (Int64(b * H + hx) * N + rtok) * 32
                    red_add_u64(ha, lk_p)
                    red_add_u64(ha + 8, lk_s)
                    red_add_u64(ha + 16, lk_c)
                    red_add_f64(ha + 24, lk_r)
                if const_expr((self.dbg & 1) and self.mode == "dq"):
                    ia = _gaddr(mIdent) + (Int64(b * H + hx) * N + rtok) * 32
                    st_global_f64x2(ia, idn, idd)
                    st_global_f64x2(ia + 16, idp, Float64(0.0))
                # ---- epilogue: out row (256 fp32 in TMEM) -> bf16 ----
                fw = None
                if const_expr(not (self.kn == "fin" and self.unsafe)):
                    fw = fin_c.wait_and_advance()
                tc_fence_after()
                hout = hx
                nho = H
                if const_expr(self.rows_are_keys):
                    nho = Hk
                ob = _gaddr(mOut) + (((Int64(b) * N + rtok) * nho + hout) * D) * 2
                for cblk in cutlass.range_constexpr(8):
                    ov = tmem_ld32(tb + self.COL_OUT + 32 * cblk)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(0, 32, 8):
                        w = [None] * 4
                        for q in cutlass.range_constexpr(4):
                            x0 = ov[i + 2 * q]
                            x1 = ov[i + 2 * q + 1]
                            if const_expr(self.mode == "dq"):
                                x0 = mul_f32(x0, Float32(SM))
                                x1 = mul_f32(x1, Float32(SM))
                            w[q] = bf16x2(x0, x1)
                        st_global_v4(ob + (32 * cblk + i) * 2, w[0], w[1], w[2], w[3])
                if const_expr(self.kn == "fin" and self.unsafe):
                    fw = fin_c.wait_and_advance()           # UNSAFE: out was read before the final MMA completed
                if const_expr(self.ordf):
                    tc_fence_before()
                fw.release()

        tmem.relinquish_alloc_permit()
        if const_expr(self.ordf):
            tc_fence_before()                # BW9: every thread's TMEM accesses before the dealloc barrier
        cute.arch.barrier(barrier_id=3, number_of_threads=self.threads)
        if const_expr(self.ordf):
            tc_fence_after()
        tmem.free(tmem_ptr)
