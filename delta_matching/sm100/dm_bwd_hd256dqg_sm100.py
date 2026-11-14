"""hd256 dQ GEMM kernel "Dqg".  Spec sm100-mx-d256 verbatim.

dQ = dS . K from the dS codes that KV2 (DmHd256Kv2 opt dqw) stored to the workspace W / Wsb -- NO S / dP / exp recompute here.
Bitwise the dq kernel (DmHd256Mx2 impl 13) by construction:
  * the same codes (KV2's dS codes == the dq kernel's, lockstep dbg-8 slot 2) and the same raw block exponents ds_sb;
  * the same MMA: kind::mxf8f6f4 cta_group::1 M=128 N=256 K=32 (gemm_ptx_bs), 4 k-steps ascending per 128-key tile, key tiles
    ascending (causal: 0..q tile), zero-init on the first tile; SFA = add_sb_clamp(ds_sb, sb_of(ks)) per 32-key block (one word,
    byte k = k-step k), SFB = 0x7F; A = W tile via TMA (MN-major A: queries contiguous), B = K tile via the T view (MN-major B, as yr);
  * the same epilogue: x * 2^-4, bf16x2 RN, one writer per element.

Grid: one CTA per (q tile, q head, batch), causal global LPT (longest first).  192 threads:
  warps 0-3: SF writers (SFA word per tile, SFB once) + epilogue (row = 32 w + lane); warp 4: MMA (one elected thread); warp 5: TMA.
smem: S stages x [A 16 KB | B 32 KB].  TMEM (512 cols): acc 0-255, SFA stage s at 256 + 4 s (+ lane quadrant), SFB 8 cols at 256 + 4 S.
Hand-offs (each with a positive-control knob):
  ab  TMA A/B -> MMA                       PipelineTmaUmma (tx), released by tcgen05.commit after the stage's 4 MMAs
  sf  SF warps tcgen05.st (SFA, + SFB at t == 0) -> MMA     mbarrier sff[s] (4 warp arrivals, release.cta), MMA acquire + fence::after
  sw  MMA(t - S) -> SF warps' st of tile t (SFA stage WAR)  the SF warps wait ab-empty[s] (the commit after MMA(t - S)) + fence::after
  fin last MMA -> epilogue tcgen05.ld                       tcgen05.commit -> mbarrier fin
  W   KV2 st.global -> this kernel's TMA / ld.global        kernel boundary (stream order); host control "wsk"
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import cpasync, tcgen05

from delta_matching.sm100.cute_ptx_sm100 import tmem_ld32, bf16x2, st_global_v4, mul_f32, global_address, inline_asm
from delta_matching.sm100.cute_ptx_sm100 import mbar_init, tc_commit, tmem_st_x1, tmem_st_x4
from delta_matching.sm100.fp8_e8m0_scale_sm100 import add_sb_clamp
from delta_matching.sm100.mma_sm100 import gemm_ptx_bs
from delta_matching.sm100.tcgen05_fence_sm100 import mbar_wait_parity
from delta_matching.sm100.dm_bwd_hd256mx_sm100 import spin, sb_of, pack4, tc_fence_before, tc_fence_after

F8 = cutlass.Float8E4M3FN
SM = 0.0625


@dsl_user_op
def ld_global_u32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(inline_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.u32 $0, [$1];", "=r,l", True, loc, ip))


@dsl_user_op
def byte_of(w: Int32, k: int, *, loc=None, ip=None) -> Int32:
    return Int32(inline_asm(T.i32(), [Int32(w).ir_value(loc=loc, ip=ip)], f"bfe.u32 $0, $1, {8 * k}, 8;", "=r,r", False, loc, ip))


@dsl_user_op
def bulk_g2s(saddr: Int32, gaddr: Int64, nbytes: int, mbar: Int32, *, loc=None, ip=None):
    """1-D bulk copy gmem -> smem completing nbytes of tx on mbar (wsw: the W tile is already the swizzled smem image)."""
    inline_asm(None, [Int32(saddr).ir_value(loc=loc, ip=ip), Int64(gaddr).ir_value(loc=loc, ip=ip), Int32(mbar).ir_value(loc=loc, ip=ip)],
         f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {int(nbytes)}, [$2];",
         "r,l,r", True, loc, ip)


@dsl_user_op
def warp_arrive(mbar: Int32, *, loc=None, ip=None):
    """bar.warp.sync, then lane 0 mbarrier.arrive.release.cta (every lane did its own wait::st + fence::before first)."""
    inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
         "{\n\t.reg .pred p;\n\t.reg .b32 l;\n\tbar.warp.sync 0xffffffff;\n\tmov.u32 l, %laneid;\n\t"
         "setp.eq.u32 p, l, 0;\n\t@p mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];\n\t}", "r", True, loc, ip)


class DmHd256Dqg:
    TM, TN, D = 128, 128, 256
    COL_ACC, COL_SF = 0, 256
    MMA_WARP, LOAD_WARP = 4, 5
    # positive controls ("<k>" UNSAFE order + delay; "<k>_safe" shipped order + the same delay, must be bitwise the default):
    #   ab : A/B stage released right after the full wait (before the MMAs) + delay before the MMAs
    #   sf : SFA published (arrive) before it is written + delay before the st
    #   sw : SF warps skip the SFA-stage WAR wait at the first reuse (t == S); the MMA warp delays MMA(0)
    #   fin: epilogue reads acc before the fin wait; the last MMA tile is delayed
    KNOBS = ("ab", "sf", "sw", "fin")
    DLY = 1 << 15

    def __init__(self, causal: bool, stages: int = 3, knob: str = "", wsw: bool = False, band: bool = False):
        """wsw: W tiles hold the swizzled MN-major SW128 smem image (KV2 opt dqt) -> 1-D bulk copy; else plain rows (dqw) -> TMA."""
        self.wsw = bool(wsw)
        # band (KV2 opt dqc chunks; causal only): this launch forms dQ of q tiles [q0, q0 + n_t) only (n_t argument =
        # the band's tile count), from a band-local W (T = l * tiles(q0, q1) + (ct (ct + 1) - q0 (q0 + 1)) / 2 + kt, kt = 0..ct: every
        # key tile of these q tiles is in the band's W -> complete dQ rows, no carry); dQ rows at their global position (mdQ full N).
        self.band = bool(band)
        assert not self.band or causal
        self.causal = bool(causal)
        self.S = int(stages)
        assert 2 <= self.S <= 4
        self.kn = knob.replace("_safe", "")
        self.unsafe = bool(knob) and not knob.endswith("_safe")
        assert self.kn in ("",) + self.KNOBS, knob
        self.threads = 192
        self.COL_SFB = self.COL_SF + 4 * self.S

    @cute.jit
    def __call__(self, mA: cute.Tensor, mKt: cute.Tensor, mWsb: cute.Tensor, mKs: cute.Tensor, mdQ: cute.Tensor,
                 n_t: Int32, H: Int32, Hk: Int32, q0: Int32, stream):
        D = self.D
        # static 128 x 128 tile modes (TMA needs a static first mode): W viewed as (q, key, T), q contiguous
        mW = mA
        mA = cute.make_tensor(mA.iterator, cute.make_layout((self.TM, self.TN, cute.size(mA, mode=[2])),
                                                            stride=(1, self.TM, self.TM * self.TN)))
        o_tiler = (self.TM, D, self.TN)               # M = 128 queries, N = 256 d, K = 128 keys
        mma_o = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.MN,
                                                   Float32, tcgen05.CtaGroup.ONE, o_tiler[:2])
        a_l = sm100_utils.make_smem_layout_a(mma_o, o_tiler, F8, self.S)
        t_l = sm100_utils.make_smem_layout_b(mma_o, o_tiler, F8, self.S)
        a1 = cute.select(a_l, mode=[0, 1, 2])
        t1 = cute.select(t_l, mode=[0, 1, 2])
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma_o.thr_id,))
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_A, tA = cute.nvgpu.make_tiled_tma_atom_A(op, mA, a1, o_tiler, mma_o, vmnk.shape)
        tma_T, tT = cute.nvgpu.make_tiled_tma_atom_B(op, mKt, t1, o_tiler, mma_o, vmnk.shape)
        self.bytes_a = cute.size_in_bytes(F8, a1)
        self.bytes_t = cute.size_in_bytes(F8, t1)

        @cute.struct
        class SS:
            bars: cute.struct.MemRange[cutlass.Int64, 16]
            kb: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 8], 8]   # sff[0..S-1], fin
            tmem_hold: Int32
            sA: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(t_l)], 1024]
        self.SS = SS
        nbz = cute.size(mdQ, mode=[0])
        self.kernel(mma_o, tma_A, tA, tma_T, tT, a_l, t_l, mW, mWsb, mKs, mdQ, n_t, H, Hk, q0).launch(
            grid=[n_t * H * nbz, 1, 1], block=[self.threads, 1, 1], smem=SS.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, mma_o: cute.TiledMma, tma_A: cute.CopyAtom, mA: cute.Tensor, tma_T: cute.CopyAtom, mKt: cute.Tensor,
               a_l: cute.ComposedLayout, t_l: cute.ComposedLayout, mW: cute.Tensor, mWsb: cute.Tensor, mKs: cute.Tensor, mdQ: cute.Tensor,
               n_t: Int32, H: Int32, Hk: Int32, q0: Int32):
        D = self.D
        S = self.S
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        lin, _, _ = cute.arch.block_idx()
        nbz = cute.arch.grid_dim()[0] // (n_t * H)
        # causal global LPT: longest q tiles first over all (head, batch)
        ct = n_t - 1 - lin // (H * nbz)
        rr_ = lin - (lin // (H * nbz)) * (H * nbz)
        h = rr_ % H
        b = rr_ // H
        hk = h // (H // Hk)
        l = b * H + h
        total = n_t
        tbase = (Int64(l) * n_t + ct) * n_t
        nrow = n_t * 128                 # dQ rows of the sequence (band: the full mdQ length)
        if const_expr(self.band):
            ct = ct + q0
            q1 = q0 + n_t
            total = ct + 1
            tbase = Int64(l) * ((q1 * (q1 + 1) - q0 * (q0 + 1)) // 2) + (ct * (ct + 1) - q0 * (q0 + 1)) // 2
            nrow = cute.size(mdQ, mode=[1])
        elif const_expr(self.causal):
            total = ct + 1
            tbase = Int64(l) * (n_t * (n_t + 1) // 2) + ct * (ct + 1) // 2

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.SS)
        sA = st.sA.get_tensor(a_l.outer, swizzle=a_l.inner)
        sT = st.sT.get_tensor(t_l.outer, swizzle=t_l.inner)
        bars = st.bars.data_ptr()
        kbar = Int32(st.kb.data_ptr().toint())
        SFF = kbar
        FIN = kbar + 8 * S

        if warp == self.LOAD_WARP:
            cpasync.prefetch_descriptor(tma_A)
            cpasync.prefetch_descriptor(tma_T)
        thr1 = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_p, ab_c = pipeline.PipelineTmaUmma.create(num_stages=S, producer_group=thr1, consumer_group=thr1,
                                                     tx_count=self.bytes_a + self.bytes_t,
                                                     barrier_storage=bars).make_participants()
        if warp == self.MMA_WARP:
            with cute.arch.elect_one():
                for s_ in cutlass.range_constexpr(S):
                    mbar_init(SFF + 8 * s_, 4)
                mbar_init(FIN, 1)
        cute.arch.mbarrier_init_fence()

        tmem = utils.TmemAllocator(st.tmem_hold.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads))
        tmem.allocate(512)
        tc_fence_before()            # canonical: alloc -> fence::before -> bar.sync -> fence::after -> ld.shared
        tmem.wait_for_alloc()
        tc_fence_after()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tmem_base = Int32(tmem_ptr.toint())
        abe = bars + S               # PipelineTmaUmma: full[0..S-1], empty[S..2S-1]
        abe_addr = Int32(abe.toint())

        # ------------------------------------------------------------------ TMA warp
        if warp == self.LOAD_WARP:
            thr_o = mma_o.get_slice(0)
            sA_addr = Int32(st.sA.data_ptr().toint())
            a_stage = cute.cosize(a_l) // S
            wbase = global_address(mW)
            tAs, tAg = cpasync.tma_partition(tma_A, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
                                             cute.group_modes(thr_o.partition_A(mA), 0, 3))
            gT = cute.local_tile(mKt[None, None, hk, b], (D, self.TN), (0, None))
            tTs, tTg = cpasync.tma_partition(tma_T, 0, cute.make_layout(1), cute.group_modes(sT, 0, 3),
                                             cute.group_modes(thr_o.partition_B(gT), 0, 3))
            for kt in cutlass.range(total, unroll=1):
                e = ab_p.acquire_and_advance()
                if const_expr(self.wsw):
                    if lane == 0:
                        bulk_g2s(sA_addr + e.index * a_stage, wbase + (tbase + kt) * 16384, 16384, Int32(e.barrier.toint()))
                else:
                    cute.copy(tma_A, tAg[(None, Int32(tbase + kt))], tAs[(None, e.index)], tma_bar_ptr=e.barrier)
                cute.copy(tma_T, tTg[(None, kt)], tTs[(None, e.index)], tma_bar_ptr=e.barrier)
            ab_p.tail()

        # ------------------------------------------------------------------ MMA warp
        if warp == self.MMA_WARP:
            mma_op = cute.make_mma_atom(mma_o.op).op
            tCrA = mma_o.make_fragment_A(sA)
            tCrT = mma_o.make_fragment_B(sT)
            for kt in cutlass.range(total, unroll=1):
                s = kt % S
                f = ab_c.wait_and_advance()
                mbar_wait_parity(SFF + 8 * s, (kt // S) & 1)
                tc_fence_after()
                if const_expr(self.kn == "ab"):
                    if const_expr(self.unsafe):
                        f.release()
                    spin(self.DLY)
                if const_expr(self.kn == "sw"):
                    if kt == 0:
                        spin(self.DLY)     # MMA(0) still pending while the SF warps reach kt == S
                if const_expr(self.kn == "fin"):
                    if kt == total - 1:
                        spin(self.DLY)
                gemm_ptx_bs(
                    mma_op, tmem_base + self.COL_ACC, tCrA[(None, None, None, f.index)], tCrT[(None, None, None, f.index)],
                    sA[(None, None, None, f.index)], sT[(None, None, None, f.index)],
                    tmem_base + self.COL_SF + 4 * s, tmem_base + self.COL_SFB,
                    tuple((0, k) for k in range(4)), tuple((0, 0) for k in range(4)),
                    zero_init=cutlass.Boolean(kt == 0))
                if const_expr(not (self.kn == "ab" and self.unsafe)):
                    f.release()      # tcgen05.commit -> ab-empty[s]: covers this stage's 4 MMAs
            with cute.arch.elect_one():
                tc_commit(FIN)

        # ------------------------------------------------------------------ SF writers + epilogue (warps 0-3)
        if warp < 4:
            tb = tmem_base + Int32((warp * 32) << 16)
            # SFB = 1.0 (0x7F), once; published by the first sff arrive below (same threads, program order + fences)
            onew = Int32(0x7F7F7F7F)
            tmem_st_x4(tb + self.COL_SFB, onew, onew, onew, onew)       # N = 256: 8 SFB columns (as Mx2)
            tmem_st_x4(tb + self.COL_SFB + 4, onew, onew, onew, onew)
            wsb_base = global_address(mWsb)
            wnext = ld_global_u32(wsb_base + tbase * 4)
            for kt in cutlass.range(total, unroll=1):
                s = kt % S
                w = wnext
                kv = [Float32(0.0)] * 4
                for k in cutlass.range_constexpr(4):
                    kv[k] = Float32(mKs[b, hk, kt * 4 + k])
                if kt + 1 < total:
                    wnext = ld_global_u32(wsb_base + (tbase + kt + 1) * 4)
                sfa = pack4(add_sb_clamp(byte_of(w, 0), sb_of(kv[0])),
                            add_sb_clamp(byte_of(w, 1), sb_of(kv[1])),
                            add_sb_clamp(byte_of(w, 2), sb_of(kv[2])),
                            add_sb_clamp(byte_of(w, 3), sb_of(kv[3])))
                if kt >= S:
                    if const_expr(self.kn == "sw" and self.unsafe):
                        if kt > S:     # UNSAFE: the first reuse (kt == S) skips the WAR wait (bounded lead: no phase aliasing)
                            mbar_wait_parity(abe_addr + 8 * s, ((kt // S) - 1) & 1)
                            tc_fence_after()
                    else:
                        mbar_wait_parity(abe_addr + 8 * s, ((kt // S) - 1) & 1)   # MMA(kt - S) complete
                        tc_fence_after()
                if const_expr(self.kn == "sf"):
                    if const_expr(self.unsafe):
                        warp_arrive(SFF + 8 * s)       # UNSAFE: published before the SFA word is written
                    spin(self.DLY)
                tmem_st_x1(tb + self.COL_SF + 4 * s + warp, sfa)       # SFA of rows 32 w.. at column + w (as Mx2)
                cute.arch.fence_view_async_tmem_store()
                tc_fence_before()
                if const_expr(not (self.kn == "sf" and self.unsafe)):
                    warp_arrive(SFF + 8 * s)
            # ---- epilogue: row 32 w + lane, 256 fp32 -> x 2^-4 -> bf16 ----
            if const_expr(not (self.kn == "fin" and self.unsafe)):
                mbar_wait_parity(FIN, Int32(0))
            tc_fence_after()
            q = ct * 128 + warp * 32 + lane
            ob = global_address(mdQ) + ((Int64(b) * nrow + q) * H + h) * (D * 2)
            for cb in cutlass.range_constexpr(8):
                ov = tmem_ld32(tb + self.COL_ACC + 32 * cb)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(0, 32, 8):
                    wv = [None] * 4
                    for qq in cutlass.range_constexpr(4):
                        wv[qq] = bf16x2(mul_f32(ov[i + 2 * qq], Float32(SM)), mul_f32(ov[i + 2 * qq + 1], Float32(SM)))
                    st_global_v4(ob + (32 * cb + i) * 2, wv[0], wv[1], wv[2], wv[3])
            if const_expr(self.kn == "fin" and self.unsafe):
                mbar_wait_parity(FIN, Int32(0))   # UNSAFE: acc read before the final commit
            tc_fence_before()
        # teardown: every TMEM reader / MMA issuer fences before the barrier, the deallocator after
        if warp == self.MMA_WARP:
            tc_fence_before()
        tmem.relinquish_alloc_permit()
        cute.arch.barrier(barrier_id=2, number_of_threads=self.threads)
        tc_fence_after()
        tmem.free(tmem_ptr)
