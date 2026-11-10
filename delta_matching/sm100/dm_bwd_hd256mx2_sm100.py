"""hd256 MXFP8 backward, schedule O1.  Same numerics as DmHd256Mx (dm_bwd_hd256mx_sm100.py), bitwise: every S / dP
is the same M=128 N=128 K=256 kind::f8f6f4 chain (8 ascending K=32 steps, zero init), P / dS via the same dm_mx256_* helpers
in the same fp32 op order, the same quantization blocks (dv: P^T per (key, 32 q); dk / dq: one pow2 per (128 q x 32 keys),
an order-free max), the same out MMA (kind::mxf8f6f4 M=128 N=256, 4 ascending K=32 steps, zero-init on t == 0), the same
bf16 epilogue.  Only the schedule / ownership changes:

  TMEM: cols 0-127 ONE score region, used S(t) then dP(t) (dk / dq) or S(t) only (dv); cols 128+16s .. +11 SF set s
        (SFA 4 cols, SFB 8 cols of constant 0x7F written once), s = t & 1; cols 256-511 out accumulator.
        dqopt (impl 3, the dq default): SF set s uses all 16 columns: SFA block +0..3 (half 0, k-steps 0, 1), +4..7 (half 1,
        k-steps 2, 3), SFB +8..15.
  Warps: 0-7 compute (warp w: TMEM lane quadrant w % 4, row r = 32 (w % 4) + lane, score columns 64 (w // 4) .. +63 =
         quantization chunks j = 2 (w // 4) + {0, 1}); 8 MMA; 9 TMA.  Each thread holds its 64 P / dS in registers (no stash).
  MMA order per tile: S(t) -> out(t-1) -> dP(t) (dk / dq), S(t) -> out(t-1) (dv).  No od barrier: SFs never alias scores.
  Hand-offs (each with a positive-control knob, see KNOBS):
    sc  score region: MMA S/dP -> 256 compute threads tcgen05.ld -> release (sc pipeline, 1 stage, used twice per tile)
    cd  codes SMEM stage + SFA set (t & 1): compute -> out(t) MMA (cd pipeline, 2 stages, released by UMMA commit)
    ld  l2 / delta staging (keys-on-rows): stage t+1 written before the per-tile barrier of t (parity buffers)
    xc  per-tile exchange slots (dk |dS| max per row half, dq per-chunk warp max, dv sbs of the upper half) -> barrier
    fin out accumulator -> epilogue
dbg bit 1 (dq): identity sums [sum dS, sum |P dP a|, sum P] (fp32 per 32-key chunk, fp64 over chunks; the two column halves
  combined at the end) -> mIdent.  dbg bit 2: dumps exactly as DmHd256Mx (S_acc, dP_acc, P, dS per (q, k); codes; sb).
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Float64, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

from delta_matching.sm100 import dm_common_hd256 as dmh
from delta_matching.sm100 import dm_utils
from delta_matching.sm100.dm_common_hd256 import (tmem_ld32, e4m3x4, bf16x2, st_global_v4, st_global_f32,
                                                 ld_shared_f32, st_shared_f32, abs_f32, max_f32, add_f32, mul_f32,
                                                 _gaddr)
from delta_matching.sm100.dm_common_hd256 import ld_shared_v4 as ld_shared_v4i
from delta_matching.sm100.dm_bwd_hd256mx_sm100 import (st_shared_v4, ld_shared_v4f, spin, sb_of, pack4, tc_fence_before,
                                                       tc_fence_after, st_global_u8, st_global_f64x2, f2i, hmix, hidx,
                                                       zext64, red_add_u64, red_add_f64, lk_cols_u32, lk_cols_f64,
                                                       sel_eq_i32)

F8 = cutlass.Float8E4M3FN
SM = 0.0625


class DmHd256Mx2:
    TM, TN, D = 128, 128, 256
    COL_SF, SF_STRIDE, COL_OUT = 128, 16, 256
    KNOBS = ("sc", "cd", "ld", "xc", "fin")
    # dq / dqopt positive controls, kept
    # out of KNOBS (t_bwdmx knobs_main loops KNOBS over all modes).  Unsafe = the edge's order broken + delay; "_safe" = shipped
    # order + the same delay.  xh0/xh1 per-half max exchange (bar 5 / 6); xi identity exchange (bar 2, dbg 1); s1 / s3 S / dP MMA ->
    # tcgen05.ld (RAW); s4 ld dP(t) -> S(t+1) (WAR); sf0/sf1 SFA st -> out (per half / SFA block); cw0/cw1 codes st.shared -> out
    # (per half); sfb0/sfb1 one-time SFB write -> first out of SF set 0 / 1.
    DQCTL = ("xh0", "xh1", "xi", "s1", "s3", "s4", "sf0", "sf1", "cw0", "cw1", "sfb0", "sfb1")
    # yr (Ya reuse) control: yw = out(t-1) completion -> TMA overwrite of its Ya stage (WAR); unsafe = the stage released right
    # after S(t-1)'s commit (the pre-yr place) instead of after out(t-1); both arms delay out(t-1) in the MMA warp.
    YRCTL = ("yw",)
    DLY = 1 << 15

    def __init__(self, mode: str, causal: bool, dbg: int = 0, knob: str = "", ordf: bool = False, dqopt: bool = False,
                 q3: bool = False, lpt: str = "", xs: bool = False, sp: bool = False, x2: bool = False, esf: bool = False,
                 yr: bool = False):
        """ordf: tcgen05 thread-sync fence pairs on every cross-thread TMEM hand-off (ordinv sites BW1-BW3, BW9 mapped to this
        schedule: sc full -> ld (after), ld -> sc release (before), sc acquire -> S/dP MMA (after), cd acquire -> SF st (after),
        epilogue / teardown); host default True (DM_HD256_ORDF), as DmHd256Mx."""
        self.ordf = bool(ordf)
        # dqopt: Q1 next tile's ks / vs prefetched one
        # tile ahead; Q2 SFA split per column half (k-steps 0,1 <- SFA block +0..3 written by half 0, k-steps 2,3 <- block +4..7
        # written by half 1; SFB moved to +8..15), per-half 128-thread exchange barriers, each half reads only its 2 chunk slots.
        # Same scale bytes, same MMA order: bitwise.
        self.dqopt = bool(dqopt) and mode == "dq"
        self.SFB_OFF = 8 if self.dqopt else 4
        # variants, dq + dqopt only, bitwise by construction:
        # q3: MMA order per tile S(t) -> dP(t) -> out(t-1) (was S(t) -> out(t-1) -> dP(t)); the TMA warp issues in that consumer
        #     order Ya(t), Yb(t), T(t-1) and drains T(last) after the loop.
        # lpt (causal): "g" = global longest-first CTA order (L = x + n_t (y + H z) -> rt = n_t-1 - L // (H B)), "1" = per head
        #     reversed rt.  Each CTA keeps its ascending key traversal.
        self.q3 = bool(q3) and self.dqopt
        self.lpt = lpt if (self.dqopt and causal) else ""
        # xs (X3): per-half max exchange read as ONE lane-distributed LDS.32 (lane l: slot 8 hf + (l & 7))
        #     + xor-1 / xor-2 shuffles (the same max(max(m0, m1), max(m2, m3)) tree) + 2 broadcasts, instead of 2 x LDS.128.
        # sp (Q4 split out): out(t-1) k-steps 0-1 -> dP(t) (committed) -> out(t-1) k-steps 2-3 (non-q3 order only).
        self.xs = bool(xs) and self.dqopt
        self.sp = bool(sp) and self.dqopt and not self.q3
        # dq + dqopt + non-q3, bitwise by construction:
        # x2: S(t+1) chunk 0 (score cols col0 .. +31) read between the per-half exchange loads and their use (blocking sc full
        #     wait on the NEXT phase without advancing, tcgen05.ld + wait::ld), carried in registers; the loop top then reads only
        #     chunk 1 and releases as before (tile 0: chunk 0 read before the loop).  Same S values, same release point.
        # esf: the SFA word (sfw) is stored to TMEM right after the cd acquire, before the code quantization / st.shared
        #     (was after them); same completion (wait::st), fences and commit.
        self.x2 = bool(x2) and self.dqopt and not self.q3 and not self.xs
        self.esf = bool(esf) and self.dqopt and not self.q3
        assert not (self.x2 and knob.replace("_safe", "") in ("s1", "sc")), knob    # s1 / sc not relocated for x2
        # yr (Ya reuse): the out MMA's B operand (K, MN-major, N = d, K = keys) is read from the
        #     Ya stage that S used (K-major K tile): T's smem layout make_smem_layout_b(mma_o) is ((1, 16384), 128) x k-step 4096 =
        #     byte d % 128 + 16384 (d // 128) + 128 key, Ya's is ((128, 1), (32, 16384)) = the same map, same swizzle S<3,4,3>
        #     and both TMA-load the same k8 bytes -> the out MMA reads the same values.  Ya gets 2
        #     stages (released by out(t-1)'s commit instead of S(t)'s); the T TMA load / pipeline are dropped.
        self.yr = bool(yr) and self.dqopt and not self.q3 and not self.sp
        assert lpt in ("", "g", "1")          # dv / dk: q3 / lpt do not apply (as dqopt), the build equals impl 2
        assert mode in ("dv", "dk", "dq")
        assert not (dbg & 4)
        self.kn = knob.replace("_safe", "")
        self.unsafe = bool(knob) and not knob.endswith("_safe")
        assert self.kn in ("",) + self.KNOBS + self.DQCTL + self.YRCTL, knob
        assert self.kn not in self.YRCTL or self.yr, knob
        assert self.kn not in self.DQCTL or self.dqopt, knob
        self.mode = mode
        self.causal = bool(causal)
        self.dbg = int(dbg)
        self.has_dp = mode != "dv"
        self.rows_are_keys = mode != "dq"
        self.threads = 320

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
        mma_o = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
                                                   Float32, tcgen05.CtaGroup.ONE, o_tiler[:2])
        c_l = sm100_utils.make_smem_layout_a(mma_o, o_tiler, F8, 2)       # 2 code stages
        t_l = sm100_utils.make_smem_layout_b(mma_o, o_tiler, F8, 1)
        a1 = cute.select(a_l, mode=[0, 1, 2])
        b1 = cute.select(b_l, mode=[0, 1, 2])
        t1 = cute.select(t_l, mode=[0, 1, 2])
        c1 = cute.select(c_l, mode=[0, 1, 2])
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
        self.bytes_c = cute.size_in_bytes(F8, c1)
        nx = 2 if const_expr(self.has_dp) else 1

        @cute.struct
        class SS:
            bars: cute.struct.MemRange[cutlass.Int64, 24]
            tmem_hold: Int32
            xch: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 2 * 128], 16]
            sLD: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 2 * 128], 16]
            sId: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 128 * 8], 16]
            sXa: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sXb: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l) if nx == 2 else 1024], 1024]
            sYa: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l) * (2 if self.yr else 1)], 1024]
            sYb: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l) if nx == 2 else 1024], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[F8, 1024 if self.yr else cute.cosize(t_l)], 1024]
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
        if const_expr(self.lpt == "1"):
            rt = n_t - 1 - rt
        elif const_expr(self.lpt == "g"):
            nbz = cute.arch.grid_dim()[2]
            lin = rt + n_t * (hx + H * b)
            rt = n_t - 1 - lin // (H * nbz)
            rr_ = lin - (lin // (H * nbz)) * (H * nbz)
            hx = rr_ % H
            b = rr_ // H
        rep = H // Hk
        N = n_t * 128
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
        sYb = st.sYb.get_tensor(b_l.outer, swizzle=b_l.inner)
        if const_expr(self.yr):
            ya2_l = sm100_utils.make_smem_layout_b(mma_s, (self.TM, self.TN, self.D), F8, 2)
            ty2_l = sm100_utils.make_smem_layout_b(mma_o, (self.TM, self.D, self.TN), F8, 2)
            assert cute.cosize(ya2_l) == 2 * cute.cosize(b_l) and cute.cosize(ty2_l) == 2 * cute.cosize(t_l)
            sYa = st.sYa.get_tensor(ya2_l.outer, swizzle=ya2_l.inner)
            sT = st.sYa.get_tensor(ty2_l.outer, swizzle=ty2_l.inner)      # out-B view of the same 2 Ya stages
        else:
            sYa = st.sYa.get_tensor(b_l.outer, swizzle=b_l.inner)
            sT = st.sT.get_tensor(t_l.outer, swizzle=t_l.inner)
        sC = st.sC.get_tensor(c_l.outer, swizzle=c_l.inner)
        sC_addr = Int32(st.sC.data_ptr().toint())
        sLD_addr = Int32(st.sLD.data_ptr().toint())
        xch_addr = Int32(st.xch.data_ptr().toint())
        sId_addr = Int32(st.sId.data_ptr().toint())
        bars = st.bars.data_ptr()

        if warp == 9:
            cpasync.prefetch_descriptor(tma_Xa)
            cpasync.prefetch_descriptor(tma_Ya)
            cpasync.prefetch_descriptor(tma_T)
            if const_expr(self.has_dp):
                cpasync.prefetch_descriptor(tma_Xb)
                cpasync.prefetch_descriptor(tma_Yb)
        nx = 2 if const_expr(self.has_dp) else 1
        thr1 = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        thr256 = pipeline.CooperativeGroup(pipeline.Agent.Thread, 256)
        x_p, x_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                   tx_count=nx * self.bytes_a, barrier_storage=bars).make_participants()
        ya_p, ya_c = pipeline.PipelineTmaUmma.create(num_stages=2 if self.yr else 1, producer_group=thr1, consumer_group=thr1,
                                                     tx_count=self.bytes_b,
                                                     barrier_storage=bars + (16 if self.yr else 2)).make_participants()
        yb_p, yb_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                     tx_count=self.bytes_b, barrier_storage=bars + 4).make_participants()
        t_p, t_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=thr1, consumer_group=thr1,
                                                   tx_count=self.bytes_t, barrier_storage=bars + 6).make_participants()
        sc_p, sc_c = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=thr1, consumer_group=thr256,
                                                       barrier_storage=bars + 8).make_participants()
        cd_p, cd_c = pipeline.PipelineAsyncUmma.create(num_stages=2, producer_group=thr256, consumer_group=thr1,
                                                       barrier_storage=bars + 10).make_participants()
        fin_p, fin_c = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=thr1, consumer_group=thr256,
                                                         barrier_storage=bars + 14).make_participants()

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
        if warp == 9:
            hy0 = hx
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
                gT = cute.local_tile(mT[None, None, hy, b], (D, self.TN), (0, None))
                tTs, tTg = cpasync.tma_partition(tma_T, 0, cute.make_layout(1), cute.group_modes(sT, 0, 3),
                                                 cute.group_modes(thr_o.partition_B(gT), 0, 3))
                for ci in cutlass.range(ncs, unroll=1):
                    ct = c_lo + ci
                    ye = ya_p.acquire_and_advance()
                    if const_expr(self.yr):
                        cute.copy(tma_Ya, tYag[(None, ct)], tYas[(None, ye.index)], tma_bar_ptr=ye.barrier)
                    else:
                        cute.copy(tma_Ya, tYag[(None, ct)], tYas[(None, 0)], tma_bar_ptr=ye.barrier)
                    if const_expr(self.has_dp):
                        yb = yb_p.acquire_and_advance()
                        cute.copy(tma_Yb, tYbg[(None, ct)], tYbs[(None, 0)], tma_bar_ptr=yb.barrier)
                    if const_expr(self.q3):
                        if ci > 0:       # q3: T(t-1) after Ya(t), Yb(t) (consumer order)
                            te = t_p.acquire_and_advance()
                            cute.copy(tma_T, tTg[(None, ct - 1)], tTs[(None, 0)], tma_bar_ptr=te.barrier)
                    elif const_expr(not self.yr):
                        te = t_p.acquire_and_advance()
                        cute.copy(tma_T, tTg[(None, ct)], tTs[(None, 0)], tma_bar_ptr=te.barrier)
                if const_expr(self.q3):  # q3: drain T(last) before the producer tails
                    te = t_p.acquire_and_advance()
                    cute.copy(tma_T, tTg[(None, c_hi - 1)], tTs[(None, 0)], tma_bar_ptr=te.barrier)
            ya_p.tail()
            if const_expr(self.has_dp):
                yb_p.tail()
            if const_expr(not self.yr):
                t_p.tail()

        # ------------------------------------------------------------------ MMA warp
        if warp == 8:
            x_c.wait_and_advance()
            mma_op = cute.make_mma_atom(mma_o.op).op
            accS = cute.make_tensor(tmem_ptr, acc_l)
            ya_o = None
            if const_expr(self.yr):
                ya_o = ya_c.clone()                            # yr: second cursor over the Ya stages, released by out(t-1)
            for t in cutlass.range(total + 1, unroll=1):
                if t < total:
                    yf = ya_c.wait_and_advance()
                    se = sc_p.acquire_and_advance()            # score region free (compute read dP(t-1) / S(t-1))
                    if const_expr(self.ordf):
                        tc_fence_after()                       # BW2
                    if const_expr(self.kn == "s1"):
                        spin(self.DLY)                         # control s1: S(t) producer late
                    mma_s.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(cute.size(tCrXa, mode=[2])):
                        if const_expr(self.yr):
                            cute.gemm(mma_s, accS, tCrXa[(None, None, kb, 0)], tCrYa[(None, None, kb, yf.index)], accS)
                        else:
                            cute.gemm(mma_s, accS, tCrXa[(None, None, kb, 0)], tCrYa[(None, None, kb, 0)], accS)
                        mma_s.set(tcgen05.Field.ACCUMULATE, True)
                    se.commit()
                    if const_expr(not self.yr or (self.kn == "yw" and self.unsafe)):
                        yf.release()                           # yr: released after out(t) instead (control yw unsafe: here)
                if const_expr(self.sp):
                    # Q4 split: out(t-1) k 0-1 -> dP(t) (commit) -> out(t-1) k 2-3; same k order, zero init on k 0 of tile 0
                    if t > 0:
                        if t < total:
                            # out(t-1): codes stage / SF set (t-1) & 1
                            if const_expr(self.kn in ("cd", "fin")):
                                spin(self.DLY)
                            tp = t - 1
                            s = tp & 1
                            if const_expr(self.kn == "cd" and self.unsafe):
                                s = tp * 0
                            tf = t_c.wait_and_advance()
                            cf = cd_c.wait_and_advance()
                            tc_fence_after()
                            dm_utils.gemm_ptx_bs(
                                mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                                sC[(None, None, None, s)], sT[(None, None, None, 0)],
                                tmem_base + self.COL_SF + self.SF_STRIDE * s, tmem_base + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF,
                                ((0, 0), (0, 1), (4, 2), (4, 3)) if self.dqopt else tuple((0, k) for k in range(4)),
                                tuple((0, 0) for k in range(4)),
                                zero_init=cutlass.Boolean(tp == 0), k_range=(0, 1))
                            yg = yb_c.wait_and_advance()
                            de = sc_p.acquire_and_advance()        # compute read S(t)
                            if const_expr(self.ordf):
                                tc_fence_after()                   # BW2
                            if const_expr(self.kn == "s3"):
                                spin(self.DLY)                     # control s3: dP(t) producer late
                            mma_s.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in cutlass.range_constexpr(cute.size(tCrXb, mode=[2])):
                                cute.gemm(mma_s, accS, tCrXb[(None, None, kb, 0)], tCrYb[(None, None, kb, 0)], accS)
                                mma_s.set(tcgen05.Field.ACCUMULATE, True)
                            de.commit()
                            yg.release()
                            dm_utils.gemm_ptx_bs(
                                mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                                sC[(None, None, None, s)], sT[(None, None, None, 0)],
                                tmem_base + self.COL_SF + self.SF_STRIDE * s, tmem_base + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF,
                                ((0, 0), (0, 1), (4, 2), (4, 3)) if self.dqopt else tuple((0, k) for k in range(4)),
                                tuple((0, 0) for k in range(4)),
                                zero_init=False, k_range=(2, 3))
                            tf.release()
                            cf.release()
                        else:
                            # out(t-1): codes stage / SF set (t-1) & 1
                            if const_expr(self.kn in ("cd", "fin")):
                                spin(self.DLY)
                            tp = t - 1
                            s = tp & 1
                            if const_expr(self.kn == "cd" and self.unsafe):
                                s = tp * 0
                            tf = t_c.wait_and_advance()
                            cf = cd_c.wait_and_advance()
                            tc_fence_after()
                            dm_utils.gemm_ptx_bs(
                                mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                                sC[(None, None, None, s)], sT[(None, None, None, 0)],
                                tmem_base + self.COL_SF + self.SF_STRIDE * s, tmem_base + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF,
                                ((0, 0), (0, 1), (4, 2), (4, 3)) if self.dqopt else tuple((0, k) for k in range(4)),
                                tuple((0, 0) for k in range(4)),
                                zero_init=cutlass.Boolean(tp == 0))
                            tf.release()
                            cf.release()
                    else:
                        yg = yb_c.wait_and_advance()
                        de = sc_p.acquire_and_advance()        # compute read S(t)
                        if const_expr(self.ordf):
                            tc_fence_after()                   # BW2
                        if const_expr(self.kn == "s3"):
                            spin(self.DLY)                     # control s3: dP(t) producer late
                        mma_s.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(cute.size(tCrXb, mode=[2])):
                            cute.gemm(mma_s, accS, tCrXb[(None, None, kb, 0)], tCrYb[(None, None, kb, 0)], accS)
                            mma_s.set(tcgen05.Field.ACCUMULATE, True)
                        de.commit()
                        yg.release()
                else:
                    if const_expr(self.yr):
                        if t > 0:
                            # out(t-1): codes stage / SF set (t-1) & 1; B = K(t-1) from its Ya stage (ya_o cursor)
                            if const_expr(self.kn in ("cd", "fin", "yw")):
                                spin(self.DLY)
                            tp = t - 1
                            s = tp & 1
                            if const_expr(self.kn == "cd" and self.unsafe):
                                s = tp * 0
                            yo = ya_o.current_handle()
                            cf = cd_c.wait_and_advance()
                            tc_fence_after()
                            dm_utils.gemm_ptx_bs(
                                mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                                sC[(None, None, None, s)], sT[(None, None, None, yo.index)],
                                tmem_base + self.COL_SF + self.SF_STRIDE * s, tmem_base + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF,
                                ((0, 0), (0, 1), (4, 2), (4, 3)),
                                tuple((0, 0) for k in range(4)),
                                zero_init=cutlass.Boolean(tp == 0))
                            if const_expr(not (self.kn == "yw" and self.unsafe)):
                                yo.release()                       # Ya stage free once out(t-1) (and S(t-1) before it) completed
                            ya_o.advance()
                            cf.release()
                    if const_expr(not self.q3 and not self.yr):
                        if t > 0:
                            # out(t-1): codes stage / SF set (t-1) & 1
                            if const_expr(self.kn in ("cd", "fin")):
                                spin(self.DLY)
                            tp = t - 1
                            s = tp & 1
                            if const_expr(self.kn == "cd" and self.unsafe):
                                s = tp * 0
                            tf = t_c.wait_and_advance()
                            cf = cd_c.wait_and_advance()
                            tc_fence_after()
                            dm_utils.gemm_ptx_bs(
                                mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                                sC[(None, None, None, s)], sT[(None, None, None, 0)],
                                tmem_base + self.COL_SF + self.SF_STRIDE * s, tmem_base + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF,
                                ((0, 0), (0, 1), (4, 2), (4, 3)) if self.dqopt else tuple((0, k) for k in range(4)),
                                tuple((0, 0) for k in range(4)),
                                zero_init=cutlass.Boolean(tp == 0))
                            tf.release()
                            cf.release()
                    if const_expr(self.has_dp):
                        if t < total:
                            yg = yb_c.wait_and_advance()
                            de = sc_p.acquire_and_advance()        # compute read S(t)
                            if const_expr(self.ordf):
                                tc_fence_after()                   # BW2
                            if const_expr(self.kn == "s3"):
                                spin(self.DLY)                     # control s3: dP(t) producer late
                            mma_s.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in cutlass.range_constexpr(cute.size(tCrXb, mode=[2])):
                                cute.gemm(mma_s, accS, tCrXb[(None, None, kb, 0)], tCrYb[(None, None, kb, 0)], accS)
                                mma_s.set(tcgen05.Field.ACCUMULATE, True)
                            de.commit()
                            yg.release()
                if const_expr(self.q3):     # q3: out(t-1) after dP(t)
                    if t > 0:
                        # out(t-1): codes stage / SF set (t-1) & 1
                        if const_expr(self.kn in ("cd", "fin")):
                            spin(self.DLY)
                        tp = t - 1
                        s = tp & 1
                        if const_expr(self.kn == "cd" and self.unsafe):
                            s = tp * 0
                        tf = t_c.wait_and_advance()
                        cf = cd_c.wait_and_advance()
                        tc_fence_after()
                        dm_utils.gemm_ptx_bs(
                            mma_op, tmem_base + self.COL_OUT, tCrC[(None, None, None, 0)], tCrT[(None, None, None, 0)],
                            sC[(None, None, None, s)], sT[(None, None, None, 0)],
                            tmem_base + self.COL_SF + self.SF_STRIDE * s, tmem_base + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF,
                            ((0, 0), (0, 1), (4, 2), (4, 3)) if self.dqopt else tuple((0, k) for k in range(4)),
                            tuple((0, 0) for k in range(4)),
                            zero_init=cutlass.Boolean(tp == 0))
                        tf.release()
                        cf.release()
            fe = fin_p.acquire_and_advance()
            fe.commit()
            sc_p.tail()
            fin_p.tail()

        # ------------------------------------------------------------------ compute warps
        if warp < 8:
            qd = warp % 4
            hf = warp // 4
            r = qd * 32 + lane
            rtok = rt * 128 + r
            ctid = tidx                 # 0..255
            tb = tmem_base + Int32((qd * 32) << 16)
            col0 = hf * 64
            hk_row = hx
            if const_expr(not self.rows_are_keys):
                hk_row = hx // rep
            row_s = Float32(0.0)
            row_a = Float32(0.0)
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
            idn = Float64(0.0)
            idd = Float64(0.0)
            idp = Float64(0.0)
            lk_p = Int64(0)
            lk_s = Int64(0)
            lk_c = Int64(0)
            lk_r = Float64(0.0)
            crow = sC_addr + (r // 8) * 1024 + (r % 8) * 128
            rsw = r % 8
            # permanent SFB (1.0) of both SF sets: one writer per lane quadrant (warps 0-3), ordered before the first
            # out MMA of each set by the cd pipeline (fence + arrive below)
            if hf == 0:
                one = Int32(0x7F7F7F7F)
                for s in cutlass.range_constexpr(2):
                    if const_expr(self.kn == "sfb%d" % s):
                        # control sfb<s>: POISON 0x80 (2.0) first, completed; the real 0x7F follows here (safe) or only after
                        # set s's first publication (unsafe, below)
                        pz = Int32(0x80808080 - (1 << 32))
                        dm_utils.tmem_st_x4(tb + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF, pz, pz, pz, pz)
                        dm_utils.tmem_st_x4(tb + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF + 4, pz, pz, pz, pz)
                        cute.arch.fence_view_async_tmem_store()
                    if const_expr(not (self.kn == "sfb%d" % s and self.unsafe)):
                        dm_utils.tmem_st_x4(tb + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF, one, one, one, one)
                        dm_utils.tmem_st_x4(tb + self.COL_SF + self.SF_STRIDE * s + self.SFB_OFF + 4, one, one, one, one)
            # keys-on-rows: stage l2 / delta of tile 0 (threads 0-127 l2, 128-255 delta)
            if const_expr(self.rows_are_keys):
                h0 = hx * rep
                if ctid < 128:
                    st_shared_f32(sLD_addr + ctid * 4, dmh.dm_mx256_l2(Float32(mLse[b, h0, c_lo * 128 + ctid])))
                else:
                    st_shared_f32(sLD_addr + ctid * 4, Float32(mDelta[b, h0, c_lo * 128 + ctid - 128]))
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
            nk0 = Float32(0.0)
            nk1 = Float32(0.0)
            nv0 = Float32(0.0)
            nv1 = Float32(0.0)
            if const_expr(self.dqopt):      # Q1: ks / vs of this thread's two chunks of tile 0 (dq: c_lo = 0)
                nk0 = Float32(mKs[b, hk_row, 2 * hf])
                nk1 = Float32(mKs[b, hk_row, 2 * hf + 1])
                nv0 = Float32(mVs[b, hk_row, 2 * hf])
                nv1 = Float32(mVs[b, hk_row, 2 * hf + 1])
            # q3 interior controls: tile t-1's delayed code / SFA stores carried into tile t
            pend_c = [(Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)) for _ in range(4)]
            sfw_c = Int32(0)
            sx0 = [Float32(0.0)] * 32         # x2: S chunk 0 of the next tile (carried)
            if const_expr(self.x2):
                sc_c.wait()                   # S(0) full (not advanced: the loop top's wait_and_advance consumes it)
                if const_expr(self.ordf):
                    tc_fence_after()
                sx0 = tmem_ld32(tb + col0)
                cute.arch.fence_view_async_tmem_load()
            for t in cutlass.range(total, unroll=1):
                hi = t // ncs
                ct = c_lo + t - hi * ncs
                h = hx
                if const_expr(self.rows_are_keys):
                    h = hx * rep + hi
                l = b * H + h
                par = t & 1
                stg = t & 1
                if const_expr(self.kn == "cd" and self.unsafe):
                    stg = t * 0
                ctok0 = ct * 128
                # per-chunk scales of this thread's two chunks j = 2 hf + jj, and the other operand's sb of all 4 chunks
                cs = [Float32(0.0)] * 2
                ca = [Float32(0.0)] * 2
                osb = [Int32(0)] * 4
                osbo = [Int32(0)] * 2         # dqopt: the other operand's sb of this thread's two chunks
                if const_expr(self.dqopt):
                    kc = [nk0, nk1]
                    vc = [nv0, nv1]
                    ctq = cutlass.min(ct + 1, c_hi - 1)          # Q1: issue tile t+1's loads now (clamped; last tile reloads itself)
                    nk0 = Float32(mKs[b, hk_row, ctq * 4 + 2 * hf])
                    nk1 = Float32(mKs[b, hk_row, ctq * 4 + 2 * hf + 1])
                    nv0 = Float32(mVs[b, hk_row, ctq * 4 + 2 * hf])
                    nv1 = Float32(mVs[b, hk_row, ctq * 4 + 2 * hf + 1])
                    for jj in cutlass.range_constexpr(2):
                        cs[jj] = dmh.dm_mx256_c(kc[jj], row_s)
                        ca[jj] = dmh.dm_mx256_dp_scale(vc[jj], row_a)
                        osbo[jj] = sb_of(kc[jj])
                for jj in cutlass.range_constexpr(0 if self.dqopt else 2):
                    cj = ct * 4 + 2 * hf + jj
                    if const_expr(self.rows_are_keys):
                        qsj = Float32(mQs[b, h, cj])
                        dosj = Float32(mdOs[b, h, cj])
                        cs[jj] = dmh.dm_mx256_c(row_s, qsj)
                        ca[jj] = dmh.dm_mx256_dp_scale(row_a, dosj)
                    else:
                        ksj = Float32(mKs[b, hk_row, cj])
                        vsj = Float32(mVs[b, hk_row, cj])
                        cs[jj] = dmh.dm_mx256_c(ksj, row_s)
                        ca[jj] = dmh.dm_mx256_dp_scale(vsj, row_a)
                for j in cutlass.range_constexpr(0 if self.dqopt else 4):
                    if const_expr(self.mode == "dv"):
                        osb[j] = sb_of(Float32(mdOs[b, h, ct * 4 + j]))
                    elif const_expr(self.mode == "dk"):
                        osb[j] = sb_of(Float32(mQs[b, h, ct * 4 + j]))
                    else:
                        osb[j] = sb_of(Float32(mKs[b, hk_row, ct * 4 + j]))
                ldb = sLD_addr + par * 1024
                ldn = sLD_addr + (1 - par) * 1024
                if const_expr(self.kn == "ld" and self.unsafe):
                    ldb = sLD_addr
                    ldn = sLD_addr
                xb = xch_addr + par * 1024
                # ---- S(t) -> registers, release the score region ----
                if const_expr(self.kn == "s1" and self.unsafe):
                    # control s1 (unsafe): read the score region BEFORE S(t)'s full wait
                    sv = [tmem_ld32(tb + col0), tmem_ld32(tb + col0 + 32)]
                    cute.arch.fence_view_async_tmem_load()
                    sf = sc_c.wait_and_advance()
                    if const_expr(self.ordf):
                        tc_fence_after()
                elif const_expr(self.x2):
                    sf = sc_c.wait_and_advance()
                    if const_expr(self.ordf):
                        tc_fence_after()                       # BW1
                    sv = [sx0, tmem_ld32(tb + col0 + 32)]      # chunk 0 read early (previous tile's exchange / prologue)
                    cute.arch.fence_view_async_tmem_load()
                else:
                    sf = sc_c.wait_and_advance()
                    if const_expr(self.ordf):
                        tc_fence_after()                       # BW1
                    if const_expr(self.kn == "sc"):
                        if const_expr(self.unsafe):
                            sf.release()
                        spin(self.DLY)
                    sv = [tmem_ld32(tb + col0), tmem_ld32(tb + col0 + 32)]
                    cute.arch.fence_view_async_tmem_load()
                if const_expr(not (self.kn == "sc" and self.unsafe)):
                    if const_expr(self.ordf):
                        tc_fence_before()                      # BW2
                    sf.release()
                if const_expr(self.q3 and self.kn in ("sf0", "sf1", "cw0", "cw1")):
                    # q3: out(t-1) runs only after dP(t), i.e. after this release: the window for tile t-1's late stores
                    if t > 0:
                        if hf == int(self.kn[2]):
                            spin(self.DLY)
                            if const_expr(self.unsafe):
                                if const_expr(self.kn[:2] == "cw"):
                                    for pa_ in pend_c:
                                        st_shared_v4(*pa_)
                                else:
                                    dm_utils.tmem_st_x1(tb + self.COL_SF + self.SF_STRIDE * ((t - 1) & 1) + 4 * hf + qd, sfw_c)
                                    cute.arch.fence_view_async_tmem_store()
                if const_expr(self.q3 and self.kn in ("sfb0", "sfb1")):
                    if t == int(self.kn[3]) + 1:
                        if hf == 0:
                            spin(self.DLY)
                            if const_expr(self.unsafe):
                                one_ = Int32(0x7F7F7F7F)
                                sfo = self.COL_SF + self.SF_STRIDE * int(self.kn[3]) + self.SFB_OFF
                                dm_utils.tmem_st_x4(tb + sfo, one_, one_, one_, one_)
                                dm_utils.tmem_st_x4(tb + sfo + 4, one_, one_, one_, one_)
                                cute.arch.fence_view_async_tmem_store()
                if const_expr(self.kn == "ld"):
                    if warp == 3:
                        spin(self.DLY)
                vals = [[Float32(0.0)] * 32, [Float32(0.0)] * 32]       # P (then dS) of chunk jj
                for jj in cutlass.range_constexpr(2):
                    cb = col0 + 32 * jj                                  # first score column of this chunk
                    for i in cutlass.range_constexpr(0, 32, 4):
                        l2v = [l2r] * 4
                        if const_expr(self.rows_are_keys):
                            l2v = ld_shared_v4f(ldb + (cb + i) * 4)
                        for q2 in cutlass.range_constexpr(0, 4, 2):
                            s0 = sv[jj][i + q2]
                            s1 = sv[jj][i + q2 + 1]
                            if const_expr(self.causal):
                                c0 = ctok0 + cb + i + q2
                                if const_expr(self.rows_are_keys):
                                    s0 = -Float32.inf if rtok > c0 else s0
                                    s1 = -Float32.inf if rtok > c0 + 1 else s1
                                else:
                                    s0 = -Float32.inf if c0 > rtok else s0
                                    s1 = -Float32.inf if c0 + 1 > rtok else s1
                            if const_expr(self.rows_are_keys):
                                p0, p1 = dmh.dm_mx256_p_pair2(s0, s1, cs[jj], l2v[q2], l2v[q2 + 1])
                            else:
                                p0, p1 = dmh.dm_mx256_p_pair(s0, s1, cs[jj], l2r)
                            vals[jj][i + q2] = p0
                            vals[jj][i + q2 + 1] = p1
                if const_expr((self.dbg & 8) and self.mode != "dk"):
                    for jj in cutlass.range_constexpr(2):
                        cb = col0 + 32 * jj
                        hpl = [Int32(0)] * 32
                        for i in cutlass.range_constexpr(32):
                            if const_expr(self.rows_are_keys):
                                hpl[i] = hmix(f2i(vals[jj][i]), hidx(ctok0 + cb + i, rtok))
                            else:
                                hpl[i] = hmix(f2i(vals[jj][i]), hidx(rtok, ctok0 + cb + i))
                        if const_expr(self.rows_are_keys):
                            lk_cols_u32(hpl, lane, _gaddr(mHash) + (Int64(l) * N + ctok0 + cb) * 32)
                        else:
                            hs_ = Int32(0)
                            for i in cutlass.range_constexpr(32):
                                hs_ = hs_ + hpl[i]
                            lk_p = lk_p + zext64(hs_)
                if const_expr(self.dbg & 2):
                    for jj in cutlass.range_constexpr(2):
                        cb = col0 + 32 * jj
                        dstp = Int64(16)
                        dpa = _gaddr(mDump) + ((Int64(l) * N + rtok) * N + ctok0 + cb) * 16
                        if const_expr(self.rows_are_keys):
                            dstp = Int64(N) * 16
                            dpa = _gaddr(mDump) + ((Int64(l) * N + ctok0 + cb) * N + rtok) * 16
                        for i in cutlass.range_constexpr(32):
                            st_global_f32(dpa, sv[jj][i])
                            if const_expr(self.mode != "dk"):
                                st_global_f32(dpa + 8, vals[jj][i])
                            dpa = dpa + dstp
                sbs = [Int32(0)] * 2         # this thread's two chunks
                ce = None
                if const_expr(self.has_dp):
                    # ---- dP(t) -> registers, release; dS = P * fma(dP, a, -delta) ----
                    if const_expr(self.kn == "s3" and self.unsafe):
                        # control s3 (unsafe): read dP BEFORE dP(t)'s full wait
                        dvv = [tmem_ld32(tb + col0), tmem_ld32(tb + col0 + 32)]
                        cute.arch.fence_view_async_tmem_load()
                        df = sc_c.wait_and_advance()
                        if const_expr(self.ordf):
                            tc_fence_after()
                        if const_expr(self.ordf):
                            tc_fence_before()
                        df.release()
                    elif const_expr(self.kn == "s4" and self.unsafe):
                        # control s4 (unsafe): release the score region BEFORE reading dP(t), then delay the read
                        df = sc_c.wait_and_advance()
                        if const_expr(self.ordf):
                            tc_fence_after()
                        df.release()
                        spin(self.DLY)
                        dvv = [tmem_ld32(tb + col0), tmem_ld32(tb + col0 + 32)]
                        cute.arch.fence_view_async_tmem_load()
                    else:
                        df = sc_c.wait_and_advance()
                        if const_expr(self.ordf):
                            tc_fence_after()                   # BW1
                        if const_expr(self.kn == "s4"):
                            spin(self.DLY)
                        dvv = [tmem_ld32(tb + col0), tmem_ld32(tb + col0 + 32)]
                        cute.arch.fence_view_async_tmem_load()
                        if const_expr(self.ordf):
                            tc_fence_before()                  # BW2
                        df.release()
                    am2 = [Float32(0.0)] * 2
                    idc = [[Float32(0.0)] * 3, [Float32(0.0)] * 3]
                    for jj in cutlass.range_constexpr(2):
                        cb = col0 + 32 * jj
                        id1 = Float32(0.0)
                        id2 = Float32(0.0)
                        id3 = Float32(0.0)
                        for i in cutlass.range_constexpr(0, 32, 4):
                            dlv = [dlr] * 4
                            if const_expr(self.rows_are_keys):
                                dlv = ld_shared_v4f(ldb + 512 + (cb + i) * 4)
                            pv = [vals[jj][i + q] for q in range(4)]
                            for q2 in cutlass.range_constexpr(0, 4, 2):
                                x0, x1 = cute.arch.fma_packed_f32x2((dvv[jj][i + q2], dvv[jj][i + q2 + 1]),
                                                                    (ca[jj], ca[jj]), (-dlv[q2], -dlv[q2 + 1]))
                                x0, x1 = cute.arch.mul_packed_f32x2((pv[q2], pv[q2 + 1]), (x0, x1))
                                vals[jj][i + q2] = x0
                                vals[jj][i + q2 + 1] = x1
                            if const_expr((self.dbg & 1) and self.mode == "dq"):
                                for q in cutlass.range_constexpr(4):
                                    id1 = add_f32(id1, vals[jj][i + q])
                                    id2 = add_f32(id2, abs_f32(mul_f32(pv[q], mul_f32(dvv[jj][i + q], ca[jj]))))
                                    id3 = add_f32(id3, pv[q])
                        idc[jj] = [id1, id2, id3]
                        am = Float32(0.0)
                        for i in cutlass.range_constexpr(32):
                            am = cute.arch.fmax(am, abs_f32(vals[jj][i]))
                        am2[jj] = am
                    if const_expr(self.dbg & 2):
                        for jj in cutlass.range_constexpr(2):
                            cb = col0 + 32 * jj
                            dstp = Int64(16)
                            dpa = _gaddr(mDump) + ((Int64(l) * N + rtok) * N + ctok0 + cb) * 16
                            if const_expr(self.rows_are_keys):
                                dstp = Int64(N) * 16
                                dpa = _gaddr(mDump) + ((Int64(l) * N + ctok0 + cb) * N + rtok) * 16
                            for i in cutlass.range_constexpr(32):
                                st_global_f32(dpa + 4, dvv[jj][i])
                                st_global_f32(dpa + 12, vals[jj][i])
                                dpa = dpa + dstp
                    # ---- stage l2 / delta of tile t+1 (before this tile's barrier) ----
                    if const_expr(self.rows_are_keys):
                        if t + 1 < total:
                            hn = hx * rep + (t + 1) // ncs
                            ctn = c_lo + (t + 1) - ((t + 1) // ncs) * ncs
                            if ctid < 128:
                                st_shared_f32(ldn + ctid * 4, dmh.dm_mx256_l2(Float32(mLse[b, hn, ctn * 128 + ctid])))
                            else:
                                st_shared_f32(ldn + ctid * 4, Float32(mDelta[b, hn, ctn * 128 + ctid - 128]))
                    # ---- block max exchange ----
                    if const_expr(self.mode == "dk"):
                        # row max over this half -> slot [hf][r]; partner half's max after the barrier
                        st_shared_f32(xb + (hf * 128 + r) * 4, cute.arch.fmax(am2[0], am2[1]))
                    else:
                        if const_expr(self.kn in ("xh0", "xh1")):
                            if warp == 4 * int(self.kn[2]):
                                spin(self.DLY)                 # control xh<h>: warp 0 / 4 late BEFORE its slot stores
                        for jj in cutlass.range_constexpr(2):
                            a_ = dm_utils.redux_max_abs_f32(am2[jj])
                            if lane == 0:
                                st_shared_f32(xb + ((2 * hf + jj) * 4 + qd) * 4, a_)
                    if const_expr((self.dbg & 1) and self.mode == "dq"):
                        if hf == 1:
                            if const_expr(self.kn == "xi"):
                                spin(self.DLY)                 # control xi: upper half late before its identity stores
                            for jj in cutlass.range_constexpr(2):
                                for u in cutlass.range_constexpr(3):
                                    st_shared_f32(sId_addr + ((par * 128 + r) * 8 + 4 * jj + u) * 4, idc[jj][u])
                    if const_expr(self.kn == "xc"):
                        if warp == 0:
                            spin(self.DLY)
                    if const_expr(not (self.kn == "xc" and self.unsafe)):
                        if const_expr(self.dqopt):
                            # Q2: chunks 2 hf, 2 hf + 1 are reduced over the 4 lane quadrants of THIS half only
                            if hf == 0:
                                if const_expr(not (self.kn == "xh0" and self.unsafe)):
                                    cute.arch.barrier(barrier_id=5, number_of_threads=128)
                            else:
                                if const_expr(not (self.kn == "xh1" and self.unsafe)):
                                    cute.arch.barrier(barrier_id=6, number_of_threads=128)
                            if const_expr((self.dbg & 1) and not (self.kn == "xi" and self.unsafe)):
                                cute.arch.barrier(barrier_id=2, number_of_threads=256)    # identity exchange (dbg only)
                        else:
                            cute.arch.barrier(barrier_id=2, number_of_threads=256)
                    if const_expr((self.dbg & 1) and self.mode == "dq"):
                        if hf == 0:
                            # original order: chunks 0, 1 (own) then 2, 3 (upper half), fp64 over ascending tiles
                            for jj in cutlass.range_constexpr(2):
                                idn = idn + Float64(idc[jj][0])
                                idd = idd + Float64(idc[jj][1])
                                idp = idp + Float64(idc[jj][2])
                            for jj in cutlass.range_constexpr(2):
                                idn = idn + Float64(ld_shared_f32(sId_addr + ((par * 128 + r) * 8 + 4 * jj) * 4))
                                idd = idd + Float64(ld_shared_f32(sId_addr + ((par * 128 + r) * 8 + 4 * jj + 1) * 4))
                                idp = idp + Float64(ld_shared_f32(sId_addr + ((par * 128 + r) * 8 + 4 * jj + 2) * 4))
                    if const_expr(self.mode == "dk"):
                        mo = ld_shared_f32(xb + ((1 - hf) * 128 + r) * 4)
                        sb = dm_utils.pow2_sb(dm_utils.redux_max_abs_f32(cute.arch.fmax(cute.arch.fmax(am2[0], am2[1]), mo)))
                        sbs = [sb, sb]
                        sball = [sb] * 4
                    elif const_expr(self.dqopt and self.xs):
                        sball = [Int32(0)] * 4
                        xv = ld_shared_f32(xb + (8 * hf + (lane & 7)) * 4)       # lanes 4c..4c+3: chunk 2 hf + c, quadrants 0..3
                        xv = max_f32(xv, cute.arch.shuffle_sync_bfly(xv, offset=1))   # lane 4c: max(m0, m1); 4c+2: max(m2, m3)
                        xv = max_f32(xv, cute.arch.shuffle_sync_bfly(xv, offset=2))   # lane 4c: max(max(m0, m1), max(m2, m3))
                        sbs[0] = dm_utils.pow2_sb(cute.arch.shuffle_sync(xv, offset=0))
                        sbs[1] = dm_utils.pow2_sb(cute.arch.shuffle_sync(xv, offset=4))
                    elif const_expr(self.dqopt and self.x2):
                        sball = [Int32(0)] * 4
                        m4s = [ld_shared_v4f(xb + (2 * hf + jj) * 16) for jj in range(2)]
                        if t + 1 < total:
                            # x2: S(t+1) chunk 0 while the exchange loads are in flight (S(t+1) issued at dP(t)'s release)
                            sc_c.wait()
                            if const_expr(self.ordf):
                                tc_fence_after()               # BW1
                            sx0 = tmem_ld32(tb + col0)
                            cute.arch.fence_view_async_tmem_load()
                        for jj in cutlass.range_constexpr(2):
                            m4 = m4s[jj]
                            sbs[jj] = dm_utils.pow2_sb(max_f32(max_f32(m4[0], m4[1]), max_f32(m4[2], m4[3])))
                    elif const_expr(self.dqopt):
                        sball = [Int32(0)] * 4
                        for jj in cutlass.range_constexpr(2):
                            m4 = ld_shared_v4f(xb + (2 * hf + jj) * 16)
                            sbs[jj] = dm_utils.pow2_sb(max_f32(max_f32(m4[0], m4[1]), max_f32(m4[2], m4[3])))
                    else:
                        sball = [Int32(0)] * 4
                        for j in cutlass.range_constexpr(4):
                            m4 = ld_shared_v4f(xb + j * 16)
                            sball[j] = dm_utils.pow2_sb(max_f32(max_f32(m4[0], m4[1]), max_f32(m4[2], m4[3])))
                        sbs = [sel_eq_i32(hf, Int32(0), sball[0], sball[2]), sel_eq_i32(hf, Int32(0), sball[1], sball[3])]
                    if const_expr(self.kn == "cd" and self.unsafe):
                        pass
                    else:
                        ce = cd_p.acquire_and_advance()
                        if const_expr(self.ordf):
                            tc_fence_after()                   # BW3
                else:
                    # dv: P^T codes per (key, 32 queries), own chunks
                    for jj in cutlass.range_constexpr(2):
                        am = Float32(0.0)
                        for i in cutlass.range_constexpr(32):
                            am = cute.arch.fmax(am, vals[jj][i])
                        sbs[jj] = dm_utils.pow2_sb(am)
                    if const_expr(self.rows_are_keys):
                        if t + 1 < total:
                            hn = hx * rep + (t + 1) // ncs
                            ctn = c_lo + (t + 1) - ((t + 1) // ncs) * ncs
                            if ctid < 128:
                                st_shared_f32(ldn + ctid * 4, dmh.dm_mx256_l2(Float32(mLse[b, hn, ctn * 128 + ctid])))
                            else:
                                st_shared_f32(ldn + ctid * 4, Float32(mDelta[b, hn, ctn * 128 + ctid - 128]))
                    if hf == 1:
                        st_shared_f32(xb + r * 4, Float32(0.0) + Float32(sbs[0] + 256 * sbs[1]))
                    if const_expr(self.kn == "xc"):
                        if warp == 0:
                            spin(self.DLY)
                    if const_expr(not (self.kn == "xc" and self.unsafe)):
                        cute.arch.barrier(barrier_id=2, number_of_threads=256)
                    hv_ = Int32(ld_shared_f32(xb + r * 4))      # upper half's packed sbs (only the hf == 0 owner uses it)
                    sball = [sbs[0], sbs[1], hv_ % 256, hv_ // 256]
                    if const_expr(not (self.kn == "cd" and self.unsafe)):
                        ce = cd_p.acquire_and_advance()
                        if const_expr(self.ordf):
                            tc_fence_after()                   # BW3
                if const_expr(self.esf and self.kn not in ("cw0", "cw1", "sf0", "sf1")):
                    # esf: this thread's SFA word first (after the cd acquire + BW3 fence above), completed with the codes below
                    v0e = dm_utils.add_sb_clamp(sbs[0], osbo[0])
                    v1e = dm_utils.add_sb_clamp(sbs[1], osbo[1])
                    ze = Int32(0)
                    sfe = sel_eq_i32(hf, Int32(0), pack4(v0e, v1e, ze, ze), pack4(ze, ze, v0e, v1e))
                    dm_utils.tmem_st_x1(tb + self.COL_SF + self.SF_STRIDE * stg + 4 * hf + qd, sfe)
                # ---- codes -> smem stage stg (own two chunks) ----
                cbase = crow + stg * self.bytes_c
                pend = []                    # control cw<h>: code stores deferred (emitted below), else stored here
                for jj in cutlass.range_constexpr(2):
                    j2 = 2 * hf + jj
                    hsl = [Int32(0)] * 32
                    hcl = [Int32(0)] * 32
                    inv = dm_utils.inv_pow2_from_sb(sbs[jj])
                    for hv in cutlass.range_constexpr(2):
                        w = [None] * 4
                        for u in cutlass.range_constexpr(4):
                            i0 = 16 * hv + 4 * u
                            y0, y1 = cute.arch.mul_packed_f32x2((vals[jj][i0], vals[jj][i0 + 1]), (inv, inv))
                            y2, y3 = cute.arch.mul_packed_f32x2((vals[jj][i0 + 2], vals[jj][i0 + 3]), (inv, inv))
                            w[u] = e4m3x4(y0, y1, y2, y3)
                        if const_expr(self.kn in ("cw0", "cw1")):
                            pend.append((cbase + (((2 * j2 + hv) ^ rsw) * 16), w[0], w[1], w[2], w[3]))
                        else:
                            st_shared_v4(cbase + (((2 * j2 + hv) ^ rsw) * 16), w[0], w[1], w[2], w[3])
                        if const_expr((self.dbg & 8) and self.has_dp):
                            for u in cutlass.range_constexpr(4):
                                for q in cutlass.range_constexpr(4):
                                    e_ = 16 * hv + 4 * u + q
                                    ce_ = ctok0 + 32 * j2 + e_
                                    ix = hidx(rtok, ce_)
                                    if const_expr(self.rows_are_keys):
                                        ix = hidx(ce_, rtok)
                                    hsl[e_] = hmix(f2i(vals[jj][e_]), ix)
                                    hcl[e_] = hmix(((w[u] >> (8 * q)) & 255) | (sbs[jj] << 8), ix)
                    if const_expr((self.dbg & 8) and self.has_dp):
                        if const_expr(self.rows_are_keys):
                            a0_ = _gaddr(mHash) + (Int64(l) * N + ctok0 + 32 * j2) * 32
                            lk_cols_u32(hsl, lane, a0_ + 8)
                            lk_cols_u32(hcl, lane, a0_ + 16)
                            lk_cols_f64(vals[jj], lane, a0_ + 24)
                        else:
                            hs_ = Int32(0)
                            hc_ = Int32(0)
                            rs_ = Float64(0.0)
                            for i in cutlass.range_constexpr(32):
                                hs_ = hs_ + hsl[i]
                                hc_ = hc_ + hcl[i]
                                rs_ = rs_ + Float64(vals[jj][i])
                            lk_s = lk_s + zext64(hs_)
                            lk_c = lk_c + zext64(hc_)
                            lk_r = lk_r + rs_
                    if const_expr(self.dbg & 2):
                        mSB[l, rtok, ct * 4 + j2] = sbs[jj]
                if const_expr(self.dbg & 2):
                    stc = Int64(1)
                    ca0 = _gaddr(mCodes) + (Int64(l) * N + rtok) * N + ctok0
                    if const_expr(self.rows_are_keys):
                        stc = Int64(N)
                        ca0 = _gaddr(mCodes) + (Int64(l) * N + ctok0) * N + rtok
                    for vv in cutlass.range(4, unroll=1):
                        v = 4 * hf + vv
                        x4 = ld_shared_v4i(cbase + ((v ^ rsw) * 16))
                        cp = ca0 + Int64(16 * v) * stc
                        for u in cutlass.range_constexpr(4):
                            for q in cutlass.range_constexpr(4):
                                st_global_u8(cp, x4[u] >> (8 * q))
                                cp = cp + stc
                sfw = Int32(0)
                if const_expr(self.dqopt):
                    v0_ = dm_utils.add_sb_clamp(sbs[0], osbo[0])
                    v1_ = dm_utils.add_sb_clamp(sbs[1], osbo[1])
                    z_ = Int32(0)
                    sfw = sel_eq_i32(hf, Int32(0), pack4(v0_, v1_, z_, z_), pack4(z_, z_, v0_, v1_))
                if const_expr(self.kn in ("cw0", "cw1", "sf0", "sf1")):
                    # controls (dqopt only): half h = knob digit publishes early (unsafe) / late but in order (safe)
                    hk_ = int(self.kn[2])
                    late_cw = self.kn[:2] == "cw"
                    if const_expr(not self.unsafe):
                        if hf == hk_:
                            if const_expr(self.q3):
                                if t + 1 == total:      # q3 interior tiles: delayed after the next S release (above)
                                    spin(self.DLY)
                            else:
                                spin(self.DLY)
                    if const_expr(late_cw):
                        if const_expr(self.unsafe):
                            if hf != hk_:
                                for pa_ in pend:
                                    st_shared_v4(*pa_)
                        else:
                            for pa_ in pend:
                                st_shared_v4(*pa_)
                    if const_expr(self.unsafe and not late_cw):
                        if hf != hk_:
                            dm_utils.tmem_st_x1(tb + self.COL_SF + self.SF_STRIDE * stg + 4 * hf + qd, sfw)
                    else:
                        dm_utils.tmem_st_x1(tb + self.COL_SF + self.SF_STRIDE * stg + 4 * hf + qd, sfw)
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_view_async_shared()
                    tc_fence_before()
                    ce.commit()
                    if const_expr(self.unsafe):
                        if hf == hk_:
                            lastq = t + 1 == total
                            if const_expr(not self.q3):
                                lastq = t >= 0
                            if lastq:                   # q3: only the last tile repairs here; interior ones after S(t+1) release
                                spin(self.DLY)
                                if const_expr(late_cw):
                                    for pa_ in pend:
                                        st_shared_v4(*pa_)
                                else:
                                    dm_utils.tmem_st_x1(tb + self.COL_SF + self.SF_STRIDE * stg + 4 * hf + qd, sfw)
                                    cute.arch.fence_view_async_tmem_store()
                    if const_expr(self.q3):
                        if const_expr(late_cw):
                            pend_c = [tuple(x) for x in pend]
                        else:
                            sfw_c = sfw
                else:
                    if const_expr(self.dqopt):
                        if const_expr(not self.esf):          # esf: stored before the quantization (above)
                            dm_utils.tmem_st_x1(tb + self.COL_SF + self.SF_STRIDE * stg + 4 * hf + qd, sfw)
                    elif hf == 0:
                        sfa = pack4(dm_utils.add_sb_clamp(sball[0], osb[0]), dm_utils.add_sb_clamp(sball[1], osb[1]),
                                    dm_utils.add_sb_clamp(sball[2], osb[2]), dm_utils.add_sb_clamp(sball[3], osb[3]))
                        dm_utils.tmem_st_x1(tb + self.COL_SF + self.SF_STRIDE * stg + qd, sfa)
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_view_async_shared()
                    tc_fence_before()
                    if const_expr(self.kn == "cd" and self.unsafe):
                        ce = cd_p.acquire_and_advance()
                    ce.commit()
                if const_expr(self.kn in ("sfb0", "sfb1")):
                    # control sfb<s>: after SF set s's FIRST publication (t == s): delay; unsafe writes the real SFB only now
                    if t == int(self.kn[3]):
                        if const_expr(self.q3):
                            if t + 1 == total:      # q3 interior tiles: repaired after S(s+1)'s release (above)
                                if hf == 0:
                                    spin(self.DLY)
                                    if const_expr(self.unsafe):
                                        one_ = Int32(0x7F7F7F7F)
                                        sfo = self.COL_SF + self.SF_STRIDE * int(self.kn[3]) + self.SFB_OFF
                                        dm_utils.tmem_st_x4(tb + sfo, one_, one_, one_, one_)
                                        dm_utils.tmem_st_x4(tb + sfo + 4, one_, one_, one_, one_)
                                        cute.arch.fence_view_async_tmem_store()
                        else:
                            if hf == 0:
                                spin(self.DLY)
                                if const_expr(self.unsafe):
                                    one_ = Int32(0x7F7F7F7F)
                                    sfo = self.COL_SF + self.SF_STRIDE * int(self.kn[3]) + self.SFB_OFF
                                    dm_utils.tmem_st_x4(tb + sfo, one_, one_, one_, one_)
                                    dm_utils.tmem_st_x4(tb + sfo + 4, one_, one_, one_, one_)
                                    cute.arch.fence_view_async_tmem_store()
            if const_expr((self.dbg & 1) and self.mode == "dq"):
                if hf == 0:
                    ia = _gaddr(mIdent) + (Int64(b * H + hx) * N + rtok) * 32
                    st_global_f64x2(ia, idn, idd)
                    st_global_f64x2(ia + 16, idp, Float64(0.0))
            if const_expr((self.dbg & 8) and self.mode == "dq"):
                ha = _gaddr(mHash) + (Int64(b * H + hx) * N + rtok) * 32
                red_add_u64(ha, lk_p)
                red_add_u64(ha + 8, lk_s)
                red_add_u64(ha + 16, lk_c)
                red_add_f64(ha + 24, lk_r)
            # ---- epilogue: out row (256 fp32 in TMEM) -> bf16; this half's 128 columns ----
            fw = None
            if const_expr(not (self.kn == "fin" and self.unsafe)):
                fw = fin_c.wait_and_advance()
            tc_fence_after()
            nho = H
            if const_expr(self.rows_are_keys):
                nho = Hk
            ob = _gaddr(mOut) + (((Int64(b) * N + rtok) * nho + hx) * D) * 2
            for cq in cutlass.range_constexpr(4):
                cblk = 4 * hf + cq
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
                fw = fin_c.wait_and_advance()
            if const_expr(self.ordf):
                tc_fence_before()
            fw.release()

        tmem.relinquish_alloc_permit()
        if const_expr(self.ordf):
            tc_fence_before()                                  # BW9
        cute.arch.barrier(barrier_id=3, number_of_threads=self.threads)
        if const_expr(self.ordf):
            tc_fence_after()
        tmem.free(tmem_ptr)
