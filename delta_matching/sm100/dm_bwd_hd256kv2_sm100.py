"""hd256 fused paired-CTA dK/dV backward kernel "KV2".  Spec sm100-mx-d256 verbatim; replaces the dv + dk launches of DmHd256Mx
(dm_bwd_hd256mx_sm100.py, NOT modified), dQ stays on DmHd256Mx's dq mode.  hd128 does not import this file.

Cluster (2,1,1) = one 128-key tile; CTA c owns keys 64c..64c+63.  All MMAs cta_group::2, M=128-collective, issued by one thread of
CTA 0.  For every q head of the kv group (ascending) and every 128-query tile (ascending; causal: from the key tile on):
  S^T  = K8 Q8^T   kind::f8f6f4 N=128 K=256 (8 x K=32 ascending d, zero-init)  -> TMEM [256,320) per CTA
  dP^T = V8 dO8^T  same                                                         -> TMEM [320,384)
  P = ex2(fma(S, c, -l2)) (dm_mx256_p_pair2), dS = P * fma(dP, a, -delta)  (exactly DmHd256Mx dv / dk modes)
  P^T codes pow2 per (key, 32 q); dS codes pow2 per (32 keys x 128 q); SFA_dV = p_sb (+) dos_sb, SFA_dK = ds_sb (+) qs_sb, SFB = 1.0
  dV += P^T . dO, dK += dS^T . Q   kind::mxf8f6f4 N=256 K=128 (4 x K=32 ascending q), B = dO / Q d-half c MN-major (natural layout)
  -> TMEM dK [0,128), dV [128,256) (fp32 across all heads / tiles, one bf16 rounding).
ONE S/dP recompute and ONE exp pass feed both dK and dV.  Every output element has one writer -> deterministic.

M=128-collective TMEM layout: CTA c, lane l, col j = D[64c + l%64, j + (N/2) * (l//64)].
Compute warp w (0-7): quadrant qd = w%4, column half hf = w//4; key row r = 32 (qd%2) + lane; it reads S/dP cols 32 hf..+31 of
its lanes = queries 64 (qd//2) + 32 hf + [0,32) = 32-query block jb = 2 (qd//2) + hf: exactly one P^T scale block.
SF TMEM: SFB [384,392) (0x7F everywhere, written once), SFA_dV stage s at 392 + 4 s (+ r//32), SFA_dK at 394 + 4 s.
SFA row r is read at lanes r and r+64 (col + r//32): its 4 bytes come from 4 warps (keys 0-31: warps 0,4,2,6; 32-63: 1,5,3,7)
-> smem byte exchange, then warps 0-3 write the word in their own quadrant.

Hand-offs (every one a PTX completion mechanism):
  kv  : TMA K,V -> MMA                       PipelineTmaUmma (2-CTA, tx on the leader)
  q   : TMA Q,dO (K-major N-halves) -> S/dP  PipelineTmaUmma, released by tcgen05.commit after dP(t)
  d   : TMA Qd,dOd (MN-major d-halves) -> dV/dK  PipelineTmaUmma, released by commit after dK(t)
  s   : S/dP TMEM: MMA commit (multicast) -> compute; compute wait::ld + fence::before + remote arrive -> next S/dP
  cd  : codes smem + SFA TMEM (2 stages): compute st.shared/tcgen05.st + fence.proxy.async + wait::st + fence::before + arrive
        -> MMA (fence::after); stage reuse after the commit following dK(t) (covers dV(t): same issuing thread)
  ex  : P-sb bytes / dS warp-max smem exchange + l2/delta staging: named barriers, parity double buffers
  fin : last dK commit (multicast) -> epilogue tcgen05.ld
  sfb : SFB written by warps 0-3 of both CTAs before the cluster barrier that precedes every MMA.

dbg bit 2: per (query, key) [S_acc, dP_acc, P, dS] -> mDump (L, N, N, 4); P^T codes -> mCodesP, dS codes -> mCodesS (L, N, N)
  [query, key]; unadorned sb -> mSBP / mSBS (L, N, N/32) [key, q-block].
dbg bit 8: lockstep checksums, definition of DmHd256Mx (keys-on-rows form): per query row [P-bits hash, dS-bits hash,
  dS code|sb hash, fp64 dS rowsum] -> mHash (L, N, 4).
dbg bit 16: per query row, fp64 sums over every key this
  launch visits of the SAME fp32 dS values this kernel quantizes (dsv) and of |P * (dP_acc * ca)| (the DmHd256Mx dbg-1 definition)
  -> mHash[l, q, 0] / mHash[l, q, 1] as f64 bits (fp64 red.add: order-free up to fp64 round-off).  Compiled out unless set.
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import cpasync, tcgen05

from delta_matching.sm100.cute_ptx_sm100 import tmem_ld32, e4m3x4, bf16x2, st_global_v4, st_global_f32, st_global_v4f, ld_shared_f32, st_shared_f32, abs_f32, max_f32, mul_f32, global_address, inline_asm
from delta_matching.sm100.attention_math_sm100 import dm_mx256_c, dm_mx256_dp_scale, dm_mx256_l2, dm_mx256_p_pair2
from delta_matching.sm100.cute_ptx_sm100 import ld_shared_b32, mbar_init, redux_max_abs_f32, st_shared_b32, tmem_st_x1, tmem_st_x4
from delta_matching.sm100.fp8_e8m0_scale_sm100 import add_sb_clamp, inv_pow2_from_sb, pow2_sb
from delta_matching.sm100.mma_sm100 import gemm_ptx_bs
from delta_matching.sm100.tcgen05_fence_sm100 import mbar_arrive_local, mbar_wait_parity, mbar_wait_parity_cluster
from delta_matching.sm100.dm_bwd_hd256mx_sm100 import (st_shared_v4, ld_shared_v4f, spin, sb_of, pack4, tc_fence_before,
                                                       tc_fence_after, st_global_u8, f2i, hmix, hidx, lk_cols_u32,
                                                       lk_cols_f64, sel_eq_i32)
from delta_matching.sm100.dm_bwd_hd256mx_sm100 import tmem_st32 as dmh_st32

F8 = cutlass.Float8E4M3FN


@dsl_user_op
def commit_mc(mbar: Int32, *, loc=None, ip=None):
    """tcgen05.commit (cta_group::2) arriving on the mbarrier at this smem offset in BOTH CTAs (mask 0b11); hardware completion."""
    inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
         "{\n\t.reg .b16 m;\n\tmov.b16 m, 3;\n\t"
         "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], m;\n\t}",
         "r", True, loc, ip)


@dsl_user_op
def warp_arrive_c0(mbar: Int32, cta_scope: bool = False, *, loc=None, ip=None):
    """Per-warp aggregated cross-CTA arrive: every lane has already done its own tcgen05 wait/fence::before (and proxy fence);
    bar.warp.sync orders the lanes' memory operations before lane 0's mbarrier.arrive.release.cluster on the leader (rank 0)
    CTA's barrier at this smem offset."""
    inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
         "{\n\t.reg .pred p;\n\t.reg .b32 l, rm;\n\tbar.warp.sync 0xffffffff;\n\tmov.u32 l, %laneid;\n\t"
         "setp.eq.u32 p, l, 0;\n\tmapa.shared::cluster.u32 rm, $0, 0;\n\t"
         "@p mbarrier.arrive" + ("" if cta_scope else ".release.cluster") + ".shared::cluster.b64 _, [rm];\n\t}", "r", True, loc, ip)


@dsl_user_op
def warp_arrive_local(mbar: Int32, *, loc=None, ip=None):
    """bar.warp.sync, then lane 0 mbarrier.arrive (release.cta) on a barrier of THIS CTA (same-CTA hand-off: .cta scope is morally
    strong; used for CTA 0 -> leader barrier, and CTA 1 -> its local relay barrier)."""
    inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)],
         "{\n\t.reg .pred p;\n\t.reg .b32 l;\n\tbar.warp.sync 0xffffffff;\n\tmov.u32 l, %laneid;\n\t"
         "setp.eq.u32 p, l, 0;\n\t@p mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];\n\t}", "r", True, loc, ip)


@dsl_user_op
def bulk_s2g_wait(gaddr: Int64, saddr: Int32, nbytes: int, *, loc=None, ip=None):
    """opt dqt: cp.async.bulk smem -> gmem (bulk_group), commit, then wait until the smem SOURCE has been read
    (cp.async.bulk.wait_group.read 0): after this the stage may be overwritten."""
    inline_asm(None, [Int64(gaddr).ir_value(loc=loc, ip=ip), Int32(saddr).ir_value(loc=loc, ip=ip)],
         f"cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], {int(nbytes)};\n\t"
         "cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 0;", "l,r", True, loc, ip)


@dsl_user_op
def bulk_wait_all(*, loc=None, ip=None):
    inline_asm(None, [], "cp.async.bulk.wait_group 0;", "", True, loc, ip)


@dsl_user_op
def ld_global_v4f(addr: Int64, *, loc=None, ip=None):
    """opt dqc: 4 raw fp32 words of the carry (bits preserved: no arithmetic between this load and the tcgen05.st)."""
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T as _T
    res = inline_asm(llvm.StructType.get_literal([_T.f32()] * 4), [Int64(addr).ir_value(loc=loc, ip=ip)],
               "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l", True, loc, ip)
    return [Float32(llvm.extractvalue(_T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_global_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    from cutlass.cutlass_dsl import T as _T
    return Int32(inline_asm(_T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.u32 $0, [$1];", "=r,l", True, loc, ip))


def _coop(n):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, n)


class DmHd256Kv2:
    TK, TQ, D = 64, 128, 256            # keys per CTA, queries per iteration, head dim
    COL_DK, COL_DV, COL_S, COL_DP = 0, 128, 256, 320
    COL_SFB, COL_SFA_P, COL_SFA_S = 384, 392, 394     # + 4 * stage
    NCOMP = 256                          # compute threads per CTA (warps 0-7)
    MMA_WARP, LOAD_WARP = 8, 9
    # lsk: planted control (one dS code flipped: b 0, kv head 0, key 0, first tile).  Positive controls of every new hand-off
    # ("<h>" = UNSAFE order + delay, "<h>_safe" = shipped order + the same delay, must be bitwise the default build):
    #   q: Q/dO smem released before the S/dP MMAs; d: Qd/dOd released before dV/dK; s: S/dP TMEM released before the
    #   compute loads; cd: codes/SFA published before they are written; ex: exchange barrier dropped; ld: l2/delta single
    #   buffer; fin: epilogue reads dK/dV before the final commit.
    # pds (identity-hook liveness, DEBUG ONLY): + PDS_EPS on the pre-quant fp32 dS of ONE element of the launch (row
    # l = 0, query PDS_Q, key PDS_K) BEFORE the dbg-16 rowsum and the quantization; "pds_safe" plants nothing.  Compiled out unless
    # knob == "pds" (const_expr): the default / dbg builds are untouched.  Kept out of KNOBS (not a hand-off control).
    KNOBS = ("lsk", "q", "d", "s", "cd", "ex", "ld", "fin", "meta")
    IDKN = ("pds",)
    PDS_Q, PDS_K, PDS_EPS = 5000, 1234, 2.0 ** -20
    DLY = 1 << 15

    # qb2: Q/dO/Qd/dOd 2 TMA stages (C1a); xpt: p_sb exchange [block][row] (C1e); meta: warp 10 stages l2/delta/qs/dos per
    # iteration into 2 smem stages (local PipelineAsync, C1b); xsv: |dS| warp maxima stored [group][k] -> one LDS.128 gather (C1f)
    # rly: cross-CTA hand-offs routed so that only ONE relay warp per pair (CTA 1, warp 11) issues the expensive
    # arrive.release.cluster; CTA 0 warps arrive at .cta scope on the leader's own barrier, CTA 1 warps on a local relay barrier
    OPTS = ("qb2", "xpt", "meta", "xsv", "ctasc", "relcl", "acqcl", "rly", "dqw", "dqt", "dqc")
    # dqc (requires dqt): EXACT-CARRY chunk of the (q head, q tile) reduction of every key tile.  The launch covers
    # q heads [0, rep) of its (copied) head tensors x q tiles [Q0, Q1) (Q1 = the n_t argument; causal: key tiles 0..Q1-1 = the grid).
    # Chunk descriptor (int32, in the mSBP slot): [Q0, n_t of the full sequence, cin, cout]; cin = 1: this launch continues the
    # reduction of EVERY key tile it visits, else of the key tiles kt < Q0 only (causal; non-causal needs Q0 = 0); cout = 1: the
    # reduction is not finished -> the raw fp32 dK / dV TMEM state is stored instead of the bf16 output.  Carry buffers (fp32, raw
    # bits, layout (B, N_full, Hk, 256)): dK in the mDump slot, dV in the mCodesP slot.  The carry is restored into the dK / dV TMEM
    # columns by the compute threads that own them (tcgen05.st + wait::st) BEFORE their t = 0 codes publication (cd hand-off: the
    # first dV / dK MMA waits for it), and the first MMA then accumulates on it (zero_init only on the globally first contribution)
    # -> every MMA sees the identical fp32 D as the unchunked launch: bitwise by induction.  W packing per launch: band-local
    # T = l * tiles(Q0, Q1) + (ct (ct + 1) - Q0 (Q0 + 1)) / 2 + kt.
    # Positive control cr: carry restored AFTER the t = 0 publication (unsafe) / before it (safe), both after a delay.
    # dqt: as dqw, but the dS codes reach W by a 1-D bulk copy of each CTA's staged 8 KiB codes image (sCS stage,
    # SW128 swizzled bytes) issued by a store warp per CTA (CTA 0 warp 11, CTA 1 warp 8) -> W tile T = [CTA 0 image | CTA 1 image]
    # (the MN-major SW128 A image of the dQ GEMM).  New hand-offs: wr (codes written -> bulk read: the existing codes-ready
    # barrier, CTA 0 F_cd / CTA 1 relay L_cd) and ww (bulk read done -> compute rewrites the stage: new local mbarrier SD[stg]).
    DQTCTL = ("wr", "ww", "cr", "dd")      # dd: DEBUG dump of the dqc chunk state (thread 0 of each CTA -> carry dK row bx)
    # dqw: also store the dS codes + raw dS block exponents to a dQ workspace for the dQ GEMM kernel DmHd256Dqg
    # (dm_bwd_hd256dqg_sm100.py). The workspace tensors are passed in the mCodesS / mSBS slots (dbg 2 must be off):
    #   mCodesS = W  uint8 (T, 128, 128): tile T = (l, q tile ct, key tile kt) [key in tile][query in tile] (plain row-major,
    #             causal packing T = l * n_t (n_t + 1) / 2 + ct (ct + 1) / 2 + kt, else l * n_t^2 + ct * n_t + kt);
    #   mSBS   = Wsb uint8 (T, 1, 4): raw pow2_sb of the 4 32-key blocks of the tile (shared by all 128 queries).
    # Plain st.global only; the consumer is a later kernel in stream order (no in-kernel hand-off).
    RELAY_WARP = 11   # relcl/acqcl: DEBUG: only release / only acquire at cluster   # ctasc: DEBUG ONLY (cost attribution): cross-CTA arrive/wait at .cta scope
    META_WARP = 10
    META_STAGE = 1088          # bytes: l2[128], delta[128], qs[4], dos[4] (+ pad)
    DEFAULT_OPTS = "meta qb2 xpt rly"      # KV2-C1 (bench_c1b3: 1.16-1.27x over C0 at all 8 shapes)

    def _arr(self, leader_bar, relay_bar, crank):
        """compute-warp arrive for a cross-CTA hand-off (see opt rly)."""
        if const_expr("rly" in self.opts):
            warp_arrive_local(leader_bar + (relay_bar - leader_bar) * crank)
        else:
            warp_arrive_c0(leader_bar, ("ctasc" in self.opts or "acqcl" in self.opts))

    def _wcl(self, mbar, ph):
        """leader-side wait on a cross-CTA arrive barrier: acquire.cluster (default) / .cta (debug opt ctasc)."""
        if const_expr("ctasc" in self.opts or "relcl" in self.opts):
            mbar_wait_parity(mbar, ph)
        else:
            mbar_wait_parity_cluster(mbar, ph)

    def __init__(self, causal: bool, dbg: int = 0, knob: str = "", opts=None):
        self.opts = tuple(sorted((self.DEFAULT_OPTS if opts is None else opts).replace(",", " ").split()))
        assert all(o in self.OPTS for o in self.opts), self.opts
        assert not (("dqw" in self.opts or "dqt" in self.opts) and (int(dbg) & 2)), "dqw/dqt reuse the dbg-2 codes slots"
        self.nqs = 2 if "qb2" in self.opts else 1
        self.causal = bool(causal)
        self.dbg = int(dbg)
        self.kn = knob.replace("_safe", "")
        self.unsafe = bool(knob) and not knob.endswith("_safe")
        assert self.kn in ("",) + self.KNOBS + self.DQTCTL + self.IDKN, knob
        assert not ("dqt" in self.opts and "dqw" in self.opts)
        assert "dqt" not in self.opts or "rly" in self.opts
        self.wsp = "dqw" in self.opts or "dqt" in self.opts
        self.dqc = "dqc" in self.opts
        assert not (self.dbg & 16 and self.dbg & 8), "dbg 16 (identity) and 8 (lockstep) share mHash"
        assert not self.dqc or ("dqt" in self.opts and (self.dbg & ~16) == 0)
        self.threads = 384

    @cute.jit
    def _epi_bf16(self, tb, mdK, mdV, b, key, hx, Hk, N, qd, hf):
        D = self.D
        d0 = 128 * (qd // 2) + 64 * hf
        for which in cutlass.range_constexpr(2):
            col = self.COL_DK if which == 0 else self.COL_DV
            mOut = mdK if which == 0 else mdV
            ob = global_address(mOut) + (((Int64(b) * N + key) * Hk + hx) * D + d0) * 2
            for cb in cutlass.range_constexpr(2):
                ov = tmem_ld32(tb + col + 64 * hf + 32 * cb)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(0, 32, 8):
                    w = [None] * 4
                    for q in cutlass.range_constexpr(4):
                        w[q] = bf16x2(ov[i + 2 * q], ov[i + 2 * q + 1])
                    st_global_v4(ob + (32 * cb + i) * 2, w[0], w[1], w[2], w[3])

    @cute.jit
    def _carry_store(self, tb, mCK, mCV, b, key, hx, Hk, nfull, qd, hf):
        """opt dqc, cout: this thread's 64 dK + 64 dV fp32 TMEM words (key row, d0..d0+63) -> carry, raw bits."""
        D = self.D
        d0 = 128 * (qd // 2) + 64 * hf
        for which in cutlass.range_constexpr(2):
            col = self.COL_DK if which == 0 else self.COL_DV
            cb_ = global_address(mCK if which == 0 else mCV) + (((Int64(b) * nfull * 128 + key) * Hk + hx) * D + d0) * 4
            for cb in cutlass.range_constexpr(2):
                ov = tmem_ld32(tb + col + 64 * hf + 32 * cb)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(0, 32, 4):
                    st_global_v4f(cb_ + (32 * cb + i) * 4, ov[i], ov[i + 1], ov[i + 2], ov[i + 3])

    @cute.jit
    def _carry_restore(self, nfirst, tb, mCK, mCV, b, key, hx, Hk, nfull, qd, hf):
        """opt dqc: if this launch continues the key tile's reduction, the raw fp32 carry -> the dK / dV TMEM words this thread owns
        (the same (lane, column) words its epilogue reads), then tcgen05.wait::st; published to the MMA issuer by this thread's t = 0
        codes publication (fence::before + arrive, cd hand-off)."""
        D = self.D
        if nfirst != 0:
            d0 = 128 * (qd // 2) + 64 * hf
            for which in cutlass.range_constexpr(2):
                col = self.COL_DK if which == 0 else self.COL_DV
                cb_ = global_address(mCK if which == 0 else mCV) + (((Int64(b) * nfull * 128 + key) * Hk + hx) * D + d0) * 4
                for cb in cutlass.range_constexpr(2):
                    v = []
                    for i in cutlass.range_constexpr(0, 32, 4):
                        v = v + ld_global_v4f(cb_ + (32 * cb + i) * 4)
                    dmh_st32(tb + col + 64 * hf + 32 * cb, v)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def __call__(self, mK: cute.Tensor, mV: cute.Tensor, mQ: cute.Tensor, mdO: cute.Tensor, mQd: cute.Tensor,
                 mdOd: cute.Tensor, mQs: cute.Tensor, mKs: cute.Tensor, mVs: cute.Tensor, mdOs: cute.Tensor,
                 mLse: cute.Tensor, mDelta: cute.Tensor, mdK: cute.Tensor, mdV: cute.Tensor,
                 mDump: cute.Tensor, mCodesP: cute.Tensor, mCodesS: cute.Tensor, mSBP: cute.Tensor, mSBS: cute.Tensor,
                 mHash: cute.Tensor, n_t: Int32, H: Int32, Hk: Int32, stream):
        D = self.D
        cg2 = tcgen05.CtaGroup.TWO
        s_tiler = (2 * self.TK, self.TQ, D)           # S^T / dP^T: M=128 keys (pair), N=128 q, K=256 d
        mma_s = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
                                                   Float32, cg2, s_tiler[:2])
        o_tiler = (2 * self.TK, D, self.TQ)           # dV / dK: M=128 keys, N=256 d, K=128 q
        mma_o = sm100_utils.make_trivial_tiled_mma(F8, F8, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
                                                   Float32, cg2, o_tiler[:2])
        a_l = sm100_utils.make_smem_layout_a(mma_s, s_tiler, F8, 1)
        b_l = sm100_utils.make_smem_layout_b(mma_s, s_tiler, F8, self.nqs)
        c_l = sm100_utils.make_smem_layout_a(mma_o, o_tiler, F8, 2)       # codes, 2 stages
        t_l = sm100_utils.make_smem_layout_b(mma_o, o_tiler, F8, self.nqs)
        a1 = cute.select(a_l, mode=[0, 1, 2])
        b1 = cute.select(b_l, mode=[0, 1, 2])
        t1 = cute.select(t_l, mode=[0, 1, 2])
        vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (mma_s.thr_id,))
        op = cpasync.CopyBulkTensorTileG2SOp(cg2)
        tma_K, tK = cute.nvgpu.make_tiled_tma_atom_A(op, mK, a1, s_tiler, mma_s, vmnk.shape)
        tma_V, tV = cute.nvgpu.make_tiled_tma_atom_A(op, mV, a1, s_tiler, mma_s, vmnk.shape)
        tma_Q, tQ = cute.nvgpu.make_tiled_tma_atom_B(op, mQ, b1, s_tiler, mma_s, vmnk.shape)
        tma_dO, tdO = cute.nvgpu.make_tiled_tma_atom_B(op, mdO, b1, s_tiler, mma_s, vmnk.shape)
        tma_Qd, tQd = cute.nvgpu.make_tiled_tma_atom_B(op, mQd, t1, o_tiler, mma_o, vmnk.shape)
        tma_dOd, tdOd = cute.nvgpu.make_tiled_tma_atom_B(op, mdOd, t1, o_tiler, mma_o, vmnk.shape)
        self.bytes_a = cute.size_in_bytes(F8, a1)
        self.bytes_b = cute.size_in_bytes(F8, b1)
        self.bytes_t = cute.size_in_bytes(F8, t1)

        @cute.struct
        class SS:
            bars: cute.struct.MemRange[cutlass.Int64, 32]
            # cross-CTA hand-offs with ordinary / tcgen05.st / ld data: raw mbarriers, arrive.release.cluster on
            # the leader + try_wait.acquire.cluster there; [F_s, E_s, F_cd0, F_cd1, E_cd0, E_cd1, F_f, E_f]
            kb: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 14 if "dqt" in self.opts else 12], 8]   # + relay [L_s, L_cd0, L_cd1, L_f] (+ dqt SD0, SD1)
            tmem_hold: Int32
            tmem_dealloc: cutlass.Int64
            sLD: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 2 * 128], 16]     # [par][l2 | delta][q]
            sXP: cute.struct.Align[cute.struct.MemRange[Int32, 2 * 64 * 4], 16]        # [par][row][q-block] p_sb
            sXS: cute.struct.Align[cute.struct.MemRange[Float32, 2 * 8], 16]           # [par][warp] |dS| warp max
            sMeta: cute.struct.Align[cute.struct.MemRange[Float32, 2 * self.META_STAGE // 4], 16]
            sK: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(a_l)], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l)], 1024]
            sdO: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(b_l)], 1024]
            sQd: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(t_l)], 1024]
            sdOd: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(t_l)], 1024]
            sCP: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(c_l)], 1024]
            sCS: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(c_l)], 1024]
        self.SS = SS
        self.kernel(mma_s, mma_o, tma_K, tK, tma_V, tV, tma_Q, tQ, tma_dO, tdO, tma_Qd, tQd, tma_dOd, tdOd,
                    a_l, b_l, c_l, t_l, mQs, mKs, mVs, mdOs, mLse, mDelta, mdK, mdV, mDump, mCodesP, mCodesS,
                    mSBP, mSBS, mHash, n_t, H, Hk).launch(
            grid=[2 * n_t, Hk, cute.size(mK, mode=[3])], block=[self.threads, 1, 1], cluster=[2, 1, 1],
            smem=SS.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, mma_s: cute.TiledMma, mma_o: cute.TiledMma, tma_K: cute.CopyAtom, mK: cute.Tensor,
               tma_V: cute.CopyAtom, mV: cute.Tensor, tma_Q: cute.CopyAtom, mQ: cute.Tensor, tma_dO: cute.CopyAtom,
               mdO: cute.Tensor, tma_Qd: cute.CopyAtom, mQd: cute.Tensor, tma_dOd: cute.CopyAtom, mdOd: cute.Tensor,
               a_l: cute.ComposedLayout, b_l: cute.ComposedLayout, c_l: cute.ComposedLayout, t_l: cute.ComposedLayout,
               mQs: cute.Tensor, mKs: cute.Tensor, mVs: cute.Tensor, mdOs: cute.Tensor, mLse: cute.Tensor,
               mDelta: cute.Tensor, mdK: cute.Tensor, mdV: cute.Tensor, mDump: cute.Tensor, mCodesP: cute.Tensor,
               mCodesS: cute.Tensor, mSBP: cute.Tensor, mSBS: cute.Tensor, mHash: cute.Tensor,
               n_t: Int32, H: Int32, Hk: Int32):
        D = self.D
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bx, hx, b = cute.arch.block_idx()
        crank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        kt = bx // 2                       # 128-key tile of the pair
        rep = H // Hk
        N = n_t * 128
        c_lo = Int32(0)
        if const_expr(self.causal):
            c_lo = kt
        if const_expr(self.dqc):   # (names defined only in dqc builds: the default build's IR / loop-carried values are untouched)
            # nfirst: 1 = this launch continues the key tile's reduction (carry restore, no zero-init)
            dsc = global_address(mSBP)
            cq0 = ld_global_s32(dsc)
            nfull = ld_global_s32(dsc + 4)
            cin = ld_global_s32(dsc + 8)
            cout = ld_global_s32(dsc + 12)
            nfirst = cin
            if const_expr(self.causal):
                c_lo = cutlass.max(kt, cq0)
                nfirst = cutlass.max(cin, Int32(cutlass.select_(kt < cq0, Int32(1), Int32(0))))
            tilesb = Int64((n_t * (n_t + 1) - cq0 * (cq0 + 1)) // 2)
        ncs = n_t - c_lo
        total = rep * ncs
        if const_expr(self.dqc and self.kn == "dd"):
            if cute.arch.thread_idx()[0] == 0:
                dda = global_address(mDump) + Int64(bx + hx * 2 * n_t) * 32
                st_global_v4(dda, cq0, nfull, cin, cout)
                st_global_v4(dda + 16, c_lo, nfirst, total, Int32(tilesb))

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.SS)
        sK = st.sK.get_tensor(a_l.outer, swizzle=a_l.inner)
        sV = st.sV.get_tensor(a_l.outer, swizzle=a_l.inner)
        sQ = st.sQ.get_tensor(b_l.outer, swizzle=b_l.inner)
        sdO = st.sdO.get_tensor(b_l.outer, swizzle=b_l.inner)
        sQd = st.sQd.get_tensor(t_l.outer, swizzle=t_l.inner)
        sdOd = st.sdOd.get_tensor(t_l.outer, swizzle=t_l.inner)
        sCP = st.sCP.get_tensor(c_l.outer, swizzle=c_l.inner)
        sCS = st.sCS.get_tensor(c_l.outer, swizzle=c_l.inner)
        sCP_addr = Int32(st.sCP.data_ptr().toint())
        sCS_addr = Int32(st.sCS.data_ptr().toint())
        cstage_bytes = Int32(cute.cosize(c_l) // 2)
        sLD_addr = Int32(st.sLD.data_ptr().toint())
        sXP_addr = Int32(st.sXP.data_ptr().toint())
        sXS_addr = Int32(st.sXS.data_ptr().toint())
        sMeta_addr = Int32(st.sMeta.data_ptr().toint())
        bars = st.bars.data_ptr()

        if warp == self.LOAD_WARP:
            for a_ in (tma_K, tma_V, tma_Q, tma_dO, tma_Qd, tma_dOd):
                cpasync.prefetch_descriptor(a_)
        vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (mma_s.thr_id,))
        one = _coop(1)
        comp2 = _coop(2 * self.NCOMP)       # compute threads of BOTH CTAs arrive on the leader's barrier
        comp1 = _coop(self.NCOMP)
        kv_p, kv_c = pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=one, consumer_group=one,
                                                     tx_count=2 * 2 * self.bytes_a, barrier_storage=bars,
                                                     cta_layout_vmnk=vmnk, defer_sync=True).make_participants()
        q_p, q_c = pipeline.PipelineTmaUmma.create(num_stages=self.nqs, producer_group=one, consumer_group=one,
                                                   tx_count=2 * 2 * self.bytes_b, barrier_storage=bars + 2,
                                                   cta_layout_vmnk=vmnk, defer_sync=True).make_participants()
        d_p, d_c = pipeline.PipelineTmaUmma.create(num_stages=self.nqs, producer_group=one, consumer_group=one,
                                                   tx_count=2 * 2 * self.bytes_t, barrier_storage=bars + 6,
                                                   cta_layout_vmnk=vmnk, defer_sync=True).make_participants()
        kbar = Int32(st.kb.data_ptr().toint())
        F_S, E_S, F_F, E_F = kbar, kbar + 8, kbar + 48, kbar + 56
        # F_s / E_cd / F_f: one tcgen05.commit (multicast, hardware completion); E_s / F_cd / E_f: 8 compute warps x 2 CTAs
        # arrive (lane 0 after bar.warp.sync, release.cluster) on the LEADER's barrier; only the leader waits on those (acquire.cluster)
        if warp == self.MMA_WARP:
            with cute.arch.elect_one():
                ncw = (self.NCOMP // 32 + 1) if const_expr("rly" in self.opts) else 2 * self.NCOMP // 32   # arrivals
                for i_ in cutlass.range_constexpr(4):
                    mbar_init(kbar + 64 + 8 * i_, self.NCOMP // 32)                           # relay (CTA 1)
                mbar_init(F_S, 1)
                mbar_init(E_S, ncw)
                for i_ in cutlass.range_constexpr(2):
                    mbar_init(kbar + 16 + 8 * i_, ncw)
                    mbar_init(kbar + 32 + 8 * i_, 1)
                mbar_init(F_F, 1)
                mbar_init(E_F, ncw)
                if const_expr("dqt" in self.opts):
                    for i_ in cutlass.range_constexpr(2):
                        mbar_init(kbar + 96 + 8 * i_, 1)                                       # SD (local)
        m_p, m_c = pipeline.PipelineAsync.create(num_stages=2, producer_group=_coop(32), consumer_group=comp1,
                                                 barrier_storage=bars + 18, defer_sync=True).make_participants()
        cute.arch.mbarrier_init_fence()

        tmem = utils.TmemAllocator(st.tmem_hold.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads),
                                   allocator_warp_id=self.LOAD_WARP, is_two_cta=True,
                                   two_cta_tmem_dealloc_mbar_ptr=st.tmem_dealloc.ptr)
        # racecheck site: tcgen05.alloc.cta_group::2 lowers to a peer handshake (UTCATOMSWS.2CTA,
        # remote STAS of the TMEM address + remote SYNCS.ARRIVE on a reserved smem mbarrier). Both CTAs pass a cluster barrier (all their
        # threads, mbarrier inits fenced) before the allocator warps start it, as in the hd128 2-CTA bwd where it runs after the cluster sync.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.allocate(512)
        tc_fence_before()            # canonical: tcgen05 alloc -> fence::before_thread_sync -> bar.sync -> fence::after -> ld.shared
        tmem.wait_for_alloc()
        tc_fence_after()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tmem_base = Int32(tmem_ptr.toint())

        # SFB = 1.0 (0x7F) in all 8 cols x 128 lanes, both CTAs, before the cluster barrier that precedes every MMA
        if warp < 4:
            onew = Int32(0x7F7F7F7F)
            tsb = tmem_base + Int32((warp * 32) << 16) + self.COL_SFB
            tmem_st_x4(tsb, onew, onew, onew, onew)
            tmem_st_x4(tsb + 4, onew, onew, onew, onew)
            cute.arch.fence_view_async_tmem_store()
            tc_fence_before()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tc_fence_after()

        # ------------------------------------------------------------------ TMA warp (both CTAs)
        if warp == self.LOAD_WARP:
            thr_s = mma_s.get_slice(crank % 2)
            thr_o = mma_o.get_slice(crank % 2)
            gK = cute.local_tile(mK[None, None, hx, b], (2 * self.TK, D), (None, 0))
            gV = cute.local_tile(mV[None, None, hx, b], (2 * self.TK, D), (None, 0))
            tKs, tKg = cpasync.tma_partition(tma_K, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3),
                                             cute.group_modes(thr_s.partition_A(gK), 0, 3))
            tVs, tVg = cpasync.tma_partition(tma_V, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3),
                                             cute.group_modes(thr_s.partition_A(gV), 0, 3))
            ke = kv_p.acquire_and_advance()
            cute.copy(tma_K, tKg[(None, kt)], tKs[(None, 0)], tma_bar_ptr=ke.barrier)
            cute.copy(tma_V, tVg[(None, kt)], tVs[(None, 0)], tma_bar_ptr=ke.barrier)
            for hi in cutlass.range(rep, unroll=1):
                h = hx * rep + hi
                gQ = cute.local_tile(mQ[None, None, h, b], (self.TQ, D), (None, 0))
                gdO = cute.local_tile(mdO[None, None, h, b], (self.TQ, D), (None, 0))
                gQd = cute.local_tile(mQd[None, None, h, b], (D, self.TQ), (0, None))
                gdOd = cute.local_tile(mdOd[None, None, h, b], (D, self.TQ), (0, None))
                tQs, tQg = cpasync.tma_partition(tma_Q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3),
                                                 cute.group_modes(thr_s.partition_B(gQ), 0, 3))
                tdOs, tdOg = cpasync.tma_partition(tma_dO, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3),
                                                   cute.group_modes(thr_s.partition_B(gdO), 0, 3))
                tQds, tQdg = cpasync.tma_partition(tma_Qd, 0, cute.make_layout(1), cute.group_modes(sQd, 0, 3),
                                                   cute.group_modes(thr_o.partition_B(gQd), 0, 3))
                tdOds, tdOdg = cpasync.tma_partition(tma_dOd, 0, cute.make_layout(1), cute.group_modes(sdOd, 0, 3),
                                                     cute.group_modes(thr_o.partition_B(gdOd), 0, 3))
                for ci in cutlass.range(ncs, unroll=1):
                    ct = c_lo + ci
                    qe = q_p.acquire_and_advance()
                    cute.copy(tma_Q, tQg[(None, ct)], tQs[(None, qe.index)], tma_bar_ptr=qe.barrier)
                    cute.copy(tma_dO, tdOg[(None, ct)], tdOs[(None, qe.index)], tma_bar_ptr=qe.barrier)
                    de = d_p.acquire_and_advance()
                    cute.copy(tma_Qd, tQdg[(None, ct)], tQds[(None, de.index)], tma_bar_ptr=de.barrier)
                    cute.copy(tma_dOd, tdOdg[(None, ct)], tdOds[(None, de.index)], tma_bar_ptr=de.barrier)
            kv_p.tail()
            q_p.tail()
            d_p.tail()

        # ------------------------------------------------------------------ metadata warp (both CTAs, opt meta)
        if const_expr("meta" in self.opts):
            if warp == self.META_WARP:
                for t in cutlass.range(total, unroll=1):
                    hi = t // ncs
                    ct = c_lo + t - hi * ncs
                    h = hx * rep + hi
                    me = m_p.acquire_and_advance()
                    mb = sMeta_addr + me.index * self.META_STAGE
                    for i in cutlass.range_constexpr(4):
                        qq = lane + 32 * i
                        st_shared_f32(mb + qq * 4, dm_mx256_l2(Float32(mLse[b, h, ct * 128 + qq])))
                        st_shared_f32(mb + 512 + qq * 4, Float32(mDelta[b, h, ct * 128 + qq]))
                    if lane < 4:
                        st_shared_f32(mb + 1024 + lane * 4, Float32(mQs[b, h, ct * 4 + lane]))
                        st_shared_f32(mb + 1040 + lane * 4, Float32(mdOs[b, h, ct * 4 + lane]))
                    me.commit()
                m_p.tail()

        # ------------------------------------------------------------------ relay warp (CTA 1, opt rly)
        if const_expr("rly" in self.opts):
            if warp == self.RELAY_WARP:
                if crank % 2 == 1:
                    sc_ = ("ctasc" in self.opts or "acqcl" in self.opts)
                    for t in cutlass.range(total, unroll=1):
                        mbar_wait_parity(kbar + 64, t & 1)            # CTA-1 readers of S/dP(t) done (acquire.cta)
                        warp_arrive_c0(E_S, sc_)                               # -> leader, release.cluster
                        stg = t & 1
                        mbar_wait_parity(kbar + 72 + 8 * stg, (t >> 1) & 1)   # CTA-1 codes / SFA of t written
                        warp_arrive_c0(kbar + 16 + 8 * stg, sc_)
                    mbar_wait_parity(kbar + 88, Int32(0))            # CTA-1 epilogue readers done
                    warp_arrive_c0(E_F, sc_)

        # ------------------------------------------------------------------ dqt store warp (CTA 0 warp 11, CTA 1 warp 8)
        if const_expr("dqt" in self.opts):
            if (crank % 2 == 0 and warp == self.RELAY_WARP) or (crank % 2 == 1 and warp == self.MMA_WARP):
                rdy = kbar + 16 + 56 * (crank % 2)       # CTA 0: F_cd[stg] (all codes of the pair written); CTA 1: relay L_cd[stg]
                wbase = global_address(mCodesS) + 8192 * (crank % 2)
                for t in cutlass.range(total, unroll=1):
                    hi = t // ncs
                    ct = c_lo + t - hi * ncs
                    l = b * H + hx * rep + hi
                    stg = t & 1
                    wtile = Int64(0)
                    if const_expr(self.dqc and self.causal):
                        wtile = Int64(l) * tilesb + (ct * (ct + 1) - cq0 * (cq0 + 1)) // 2 + kt
                    elif const_expr(self.causal):
                        wtile = Int64(l) * (n_t * (n_t + 1) // 2) + ct * (ct + 1) // 2 + kt
                    else:
                        wtile = (Int64(l) * n_t + ct) * n_t + kt
                    if const_expr(self.kn == "wr" and self.unsafe):
                        if t > 0:          # UNSAFE wr: t == 0 is stored without waiting for its codes (compute delayed at t == 0)
                            mbar_wait_parity_cluster(rdy + 8 * stg, (t >> 1) & 1)
                    else:
                        mbar_wait_parity_cluster(rdy + 8 * stg, (t >> 1) & 1)   # acquire.cluster (CTA 0: remote relay arrive)
                    if const_expr(self.kn == "ww"):
                        if t == 0:
                            spin(self.DLY)
                    if lane == 0:
                        bulk_s2g_wait(wbase + wtile * 16384, sCS_addr + stg * cstage_bytes, 8192)
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive_local(kbar + 96 + 8 * stg)
                if lane == 0:
                    bulk_wait_all()

        # ------------------------------------------------------------------ MMA warp (leader CTA issues)
        if warp == self.MMA_WARP:
            if crank % 2 == 0:
                mma_op = cute.make_mma_atom(mma_o.op).op
                acc_l = mma_s.make_fragment_C(mma_s.partition_shape_C((2 * self.TK, self.TQ))).layout
                accS = cute.make_tensor(tmem_ptr + self.COL_S, acc_l)
                accP = cute.make_tensor(tmem_ptr + self.COL_DP, acc_l)
                tKr = mma_s.make_fragment_A(sK)
                tVr = mma_s.make_fragment_A(sV)
                tQr = mma_s.make_fragment_B(sQ)
                tdOr = mma_s.make_fragment_B(sdO)
                tCPr = mma_o.make_fragment_A(sCP)
                tCSr = mma_o.make_fragment_A(sCS)
                tQdr = mma_o.make_fragment_B(sQd)
                tdOdr = mma_o.make_fragment_B(sdOd)
                kf = kv_c.wait_and_advance()
                # prologue: S(0), dP(0)
                qf = q_c.wait_and_advance()
                self._wcl(E_S, Int32(1))       # fresh slot
                tc_fence_after()
                if const_expr(self.kn == "q"):
                    if const_expr(self.unsafe):
                        qf.release()
                    spin(self.DLY)
                mma_s.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(cute.size(tKr, mode=[2])):
                    cute.gemm(mma_s, accS, tKr[(None, None, kb, 0)], tQr[(None, None, kb, qf.index)], accS)
                    mma_s.set(tcgen05.Field.ACCUMULATE, True)
                mma_s.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(cute.size(tVr, mode=[2])):
                    cute.gemm(mma_s, accP, tVr[(None, None, kb, 0)], tdOr[(None, None, kb, qf.index)], accP)
                    mma_s.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    commit_mc(F_S)
                if const_expr(not (self.kn == "q" and self.unsafe)):
                    qf.release()
                for t in cutlass.range(total, unroll=1):
                    if t + 1 < total:
                        # S/dP(t+1): the slot was released by every compute thread of both CTAs after its S/dP(t) loads
                        qf = q_c.wait_and_advance()
                        self._wcl(E_S, t & 1)    # all 512 readers of S/dP(t) done (release.cluster)
                        tc_fence_after()
                        if const_expr(self.kn == "q"):
                            if const_expr(self.unsafe):
                                qf.release()
                            spin(self.DLY)
                        mma_s.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(cute.size(tKr, mode=[2])):
                            cute.gemm(mma_s, accS, tKr[(None, None, kb, 0)], tQr[(None, None, kb, qf.index)], accS)
                            mma_s.set(tcgen05.Field.ACCUMULATE, True)
                        mma_s.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(cute.size(tVr, mode=[2])):
                            cute.gemm(mma_s, accP, tVr[(None, None, kb, 0)], tdOr[(None, None, kb, qf.index)], accP)
                            mma_s.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            commit_mc(F_S)
                        if const_expr(not (self.kn == "q" and self.unsafe)):
                            qf.release()
                    df = d_c.wait_and_advance()
                    stg = t & 1
                    self._wcl(kbar + 16 + 8 * stg, (t >> 1) & 1)   # codes / SFA of t published
                    tc_fence_after()
                    if const_expr(self.kn == "d"):
                        if const_expr(self.unsafe):
                            df.release()
                        spin(self.DLY)
                    if const_expr(self.kn == "fin"):
                        if t == total - 1:
                            spin(self.DLY)
                    gemm_ptx_bs(
                        mma_op, tmem_base + self.COL_DV, tCPr[(None, None, None, stg)], tdOdr[(None, None, None, df.index)],
                        sCP[(None, None, None, stg)], sdOd[(None, None, None, df.index)],
                        tmem_base + self.COL_SFA_P + 4 * stg, tmem_base + self.COL_SFB,
                        tuple((0, k) for k in range(4)), tuple((0, k) for k in range(4)),
                        zero_init=cutlass.Boolean((t + nfirst == 0) if const_expr(self.dqc) else (t == 0)), cta_group=2)
                    gemm_ptx_bs(
                        mma_op, tmem_base + self.COL_DK, tCSr[(None, None, None, stg)], tQdr[(None, None, None, df.index)],
                        sCS[(None, None, None, stg)], sQd[(None, None, None, df.index)],
                        tmem_base + self.COL_SFA_S + 4 * stg, tmem_base + self.COL_SFB,
                        tuple((0, k) for k in range(4)), tuple((0, k) for k in range(4)),
                        zero_init=cutlass.Boolean((t + nfirst == 0) if const_expr(self.dqc) else (t == 0)), cta_group=2)
                    with cute.arch.elect_one():
                        commit_mc(kbar + 32 + 8 * stg)   # after dK(t): covers dV(t) and dK(t) (same issuing thread)
                    if const_expr(not (self.kn == "d" and self.unsafe)):
                        df.release()
                with cute.arch.elect_one():
                    commit_mc(F_F)
                kf.release()
                self._wcl(E_S, (total - 1) & 1)   # last S/dP slot release arrived
                self._wcl(E_F, Int32(0))          # epilogue readers done

        # ------------------------------------------------------------------ compute warps 0-7 (both CTAs)
        if warp < 8:
            qd = warp % 4
            hf = warp // 4
            r = 32 * (qd % 2) + lane                      # local key row
            jb = 2 * (qd // 2) + hf                       # this thread's 32-query block of the tile
            key = kt * 128 + 64 * crank + r
            tb = tmem_base + Int32((qd * 32) << 16)
            ks_r = Float32(mKs[b, hx, key // 32])
            vs_r = Float32(mVs[b, hx, key // 32])
            crow_off = (r // 8) * 1024 + (r % 8) * 128
            rsw = r % 8
            if const_expr(self.dqc):
                if const_expr(self.kn == "cr" and not self.unsafe):
                    spin(self.DLY)               # safe: the same delay, before the restore (and so before the t = 0 publication)
                if const_expr(not (self.kn == "cr" and self.unsafe)):
                    self._carry_restore(nfirst, tb, mDump, mCodesP, b, key, hx, Hk, nfull, qd, hf)
            for t in cutlass.range(total, unroll=1):
                hi = t // ncs
                ct = c_lo + t - hi * ncs
                h = hx * rep + hi
                l = b * H + h
                par = t & 1
                ctok0 = ct * 128
                q0 = ctok0 + 32 * jb                      # first query of this thread's block
                ldb = sLD_addr + par * 1024
                wtile = Int64(0)
                if const_expr(self.wsp):
                    if const_expr(self.dqc and self.causal):
                        wtile = Int64(l) * tilesb + (ct * (ct + 1) - cq0 * (cq0 + 1)) // 2 + kt
                    elif const_expr(self.causal):
                        wtile = Int64(l) * (n_t * (n_t + 1) // 2) + ct * (ct + 1) // 2 + kt
                    else:
                        wtile = (Int64(l) * n_t + ct) * n_t + kt
                mf = None
                qsj = Float32(0.0)
                dosj = Float32(0.0)
                if const_expr("meta" in self.opts):
                    mf = m_c.wait_and_advance()
                    ldb = sMeta_addr + mf.index * self.META_STAGE
                    if const_expr(self.kn == "meta"):
                        if const_expr(self.unsafe):
                            mf.release()                  # UNSAFE: stage released before it is read
                        spin(self.DLY)
                    qsj = ld_shared_f32(ldb + 1024 + jb * 4)
                    dosj = ld_shared_f32(ldb + 1040 + jb * 4)
                else:
                    qsj = Float32(mQs[b, h, ct * 4 + jb])
                    dosj = Float32(mdOs[b, h, ct * 4 + jb])
                    if const_expr(self.kn == "ld" and self.unsafe):
                        ldb = sLD_addr
                    if tidx < 128:
                        st_shared_f32(ldb + tidx * 4, dm_mx256_l2(Float32(mLse[b, h, ctok0 + tidx])))
                    else:
                        st_shared_f32(ldb + 512 + (tidx - 128) * 4, Float32(mDelta[b, h, ctok0 + tidx - 128]))
                    cute.arch.barrier(barrier_id=2, number_of_threads=self.NCOMP)
                    if const_expr(self.kn == "ld"):
                        if warp == 3:
                            spin(self.DLY)
                cs = dm_mx256_c(ks_r, qsj)
                ca = dm_mx256_dp_scale(vs_r, dosj)
                stg = t & 1
                mbar_wait_parity(kbar + 32 + 8 * stg, ((t >> 1) & 1) ^ 1)   # stage free (commit after dK(t-2))
                if const_expr("dqt" in self.opts):
                    if t >= 2:
                        if const_expr(self.kn == "ww"):
                            if const_expr(self.unsafe):
                                if t > 2:      # UNSAFE: the first reuse skips the bulk-read wait (store warp delayed at t == 0)
                                    mbar_wait_parity(kbar + 96 + 8 * stg, ((t >> 1) - 1) & 1)
                            else:
                                mbar_wait_parity(kbar + 96 + 8 * stg, ((t >> 1) - 1) & 1)
                        else:
                            mbar_wait_parity(kbar + 96 + 8 * stg, ((t >> 1) - 1) & 1)   # bulk read of t-2 done
                if const_expr(self.kn == "wr"):
                    if t == 0:
                        spin(self.DLY)                    # codes of t = 0 written late (wr: the store warp must wait for them)
                if const_expr(self.kn == "cd" and self.unsafe):
                    self._arr(kbar + 16 + 8 * stg, kbar + 72 + 8 * stg, crank)   # UNSAFE: published before the codes are written
                    spin(self.DLY)
                mbar_wait_parity(F_S, t & 1)       # S/dP(t) complete (commit)
                tc_fence_after()
                if const_expr(self.kn == "s"):
                    if const_expr(self.unsafe):
                        self._arr(E_S, kbar + 64, crank)   # UNSAFE: slot released before the loads
                    spin(self.DLY)
                sv = tmem_ld32(tb + self.COL_S + 32 * hf)
                dpv = tmem_ld32(tb + self.COL_DP + 32 * hf)
                cute.arch.fence_view_async_tmem_load()
                tc_fence_before()
                if const_expr(not (self.kn == "s" and self.unsafe)):
                    self._arr(E_S, kbar + 64, crank)    # release.cluster on the leader's barrier
                pv = cute.make_rmem_tensor(cute.make_layout(32), Float32)
                dsv = cute.make_rmem_tensor(cute.make_layout(32), Float32)
                for i in cutlass.range_constexpr(0, 32, 4):
                    l2v = ld_shared_v4f(ldb + (32 * jb + i) * 4)
                    dlv = ld_shared_v4f(ldb + 512 + (32 * jb + i) * 4)
                    for q2 in cutlass.range_constexpr(0, 4, 2):
                        s0 = sv[i + q2]
                        s1 = sv[i + q2 + 1]
                        if const_expr(self.causal):
                            c0 = q0 + i + q2
                            s0 = -Float32.inf if key > c0 else s0
                            s1 = -Float32.inf if key > c0 + 1 else s1
                        p0, p1 = dm_mx256_p_pair2(s0, s1, cs, l2v[q2], l2v[q2 + 1])
                        pv[i + q2] = p0
                        pv[i + q2 + 1] = p1
                        x0, x1 = cute.arch.fma_packed_f32x2((dpv[i + q2], dpv[i + q2 + 1]), (ca, ca),
                                                            (-dlv[q2], -dlv[q2 + 1]))
                        x0, x1 = cute.arch.mul_packed_f32x2((p0, p1), (x0, x1))
                        dsv[i + q2] = x0
                        dsv[i + q2 + 1] = x1
                if const_expr(self.kn == "pds" and self.unsafe):
                    hit = (sel_eq_i32(l, Int32(0), Int32(1), Int32(0)) * sel_eq_i32(key, Int32(self.PDS_K), Int32(1), Int32(0))
                           * sel_eq_i32(q0, Int32(self.PDS_Q - self.PDS_Q % 32), Int32(1), Int32(0)))
                    dsv[self.PDS_Q % 32] = dsv[self.PDS_Q % 32] + Float32(hit) * Float32(self.PDS_EPS)
                if const_expr(self.dbg & 2):
                    pa = global_address(mDump) + ((Int64(l) * N + q0) * N + key) * 16
                    for i in cutlass.range_constexpr(32):
                        st_global_f32(pa, sv[i])
                        st_global_f32(pa + 4, dpv[i])
                        st_global_f32(pa + 8, pv[i])
                        st_global_f32(pa + 12, dsv[i])
                        pa = pa + Int64(N) * 16
                # ---- P^T codes: pow2 per (key, 32 queries), thread-local ----
                am = Float32(0.0)
                for i in cutlass.range_constexpr(32):
                    am = cute.arch.fmax(am, pv[i])
                psb = pow2_sb(am)
                inv = inv_pow2_from_sb(psb)
                crow = sCP_addr + stg * cstage_bytes + crow_off
                hpl = [Int32(0)] * 32
                for hv in cutlass.range_constexpr(2):
                    w = [None] * 4
                    for u in cutlass.range_constexpr(4):
                        i0 = 16 * hv + 4 * u
                        y0, y1 = cute.arch.mul_packed_f32x2((pv[i0], pv[i0 + 1]), (inv, inv))
                        y2, y3 = cute.arch.mul_packed_f32x2((pv[i0 + 2], pv[i0 + 3]), (inv, inv))
                        w[u] = e4m3x4(y0, y1, y2, y3)
                    st_shared_v4(crow + (((2 * jb + hv) ^ rsw) * 16), w[0], w[1], w[2], w[3])
                    if const_expr(self.dbg & 2):
                        cp_ = global_address(mCodesP) + (Int64(l) * N + q0 + 16 * hv) * N + key
                        for u in cutlass.range_constexpr(4):
                            for q in cutlass.range_constexpr(4):
                                st_global_u8(cp_, w[u] >> (8 * q))
                                cp_ = cp_ + Int64(N)
                if const_expr(self.dbg & 2):
                    mSBP[l, key, ct * 4 + jb] = psb
                if const_expr(self.dbg & 8):
                    for i in cutlass.range_constexpr(32):
                        hpl[i] = hmix(f2i(pv[i]), hidx(q0 + i, key))
                    lk_cols_u32(hpl, lane, global_address(mHash) + (Int64(l) * N + q0) * 32)
                # ---- exchange: p_sb byte per (row, q-block); |dS| warp max per warp ----
                xp = sXP_addr + par * 1024
                if const_expr(self.kn == "ex"):
                    if warp == 6:
                        spin(self.DLY)
                if const_expr("xpt" in self.opts):
                    st_shared_b32(xp + (jb * 64 + r) * 4, psb)
                else:
                    st_shared_b32(xp + (r * 4 + jb) * 4, psb)
                amd = Float32(0.0)
                for i in cutlass.range_constexpr(32):
                    amd = cute.arch.fmax(amd, abs_f32(dsv[i]))
                wm = redux_max_abs_f32(amd)
                if lane == 0:
                    if const_expr("xsv" in self.opts):
                        st_shared_f32(sXS_addr + (par * 8 + (qd % 2) * 4 + 2 * hf + qd // 2) * 4, wm)
                    else:
                        st_shared_f32(sXS_addr + (par * 8 + warp) * 4, wm)
                cute.arch.fence_view_async_shared()
                if const_expr(not (self.kn == "ex" and self.unsafe)):
                    cute.arch.barrier(barrier_id=3, number_of_threads=self.NCOMP)
                g = qd % 2                                 # key group: warps g, g+2, g+4, g+6
                xs = sXS_addr + par * 32
                m4 = Float32(0.0)
                if const_expr("xsv" in self.opts):
                    x4 = ld_shared_v4f(xs + g * 16)
                    m4 = max_f32(max_f32(x4[0], x4[1]), max_f32(x4[2], x4[3]))
                else:
                    m4 = max_f32(max_f32(ld_shared_f32(xs + g * 4), ld_shared_f32(xs + (g + 2) * 4)),
                                 max_f32(ld_shared_f32(xs + (g + 4) * 4), ld_shared_f32(xs + (g + 6) * 4)))
                dsb = pow2_sb(m4)
                invs = inv_pow2_from_sb(dsb)
                srow = sCS_addr + stg * cstage_bytes + crow_off
                hsl = [Int32(0)] * 32
                hcl = [Int32(0)] * 32
                for hv in cutlass.range_constexpr(2):
                    w = [None] * 4
                    for u in cutlass.range_constexpr(4):
                        i0 = 16 * hv + 4 * u
                        y0, y1 = cute.arch.mul_packed_f32x2((dsv[i0], dsv[i0 + 1]), (invs, invs))
                        y2, y3 = cute.arch.mul_packed_f32x2((dsv[i0 + 2], dsv[i0 + 3]), (invs, invs))
                        w[u] = e4m3x4(y0, y1, y2, y3)
                    if const_expr(self.kn == "lsk" and self.unsafe):
                        if const_expr(hv == 0):
                            w[0] = w[0] ^ sel_eq_i32(t + key + hx + b, Int32(0), Int32(1), Int32(0))
                    st_shared_v4(srow + (((2 * jb + hv) ^ rsw) * 16), w[0], w[1], w[2], w[3])
                    if const_expr("dqw" in self.opts):
                        st_global_v4(global_address(mCodesS) + wtile * 16384 + (64 * crank + r) * 128 + 32 * jb + 16 * hv,
                                     w[0], w[1], w[2], w[3])
                    if const_expr(self.dbg & 8):
                        for u in cutlass.range_constexpr(4):
                            for q in cutlass.range_constexpr(4):
                                i_ = 16 * hv + 4 * u + q
                                ix = hidx(q0 + i_, key)
                                hsl[i_] = hmix(f2i(dsv[i_]), ix)
                                hcl[i_] = hmix(((w[u] >> (8 * q)) & 255) | (dsb << 8), ix)
                    if const_expr(self.dbg & 2):
                        cp_ = global_address(mCodesS) + (Int64(l) * N + q0 + 16 * hv) * N + key
                        for u in cutlass.range_constexpr(4):
                            for q in cutlass.range_constexpr(4):
                                st_global_u8(cp_, w[u] >> (8 * q))
                                cp_ = cp_ + Int64(N)
                if const_expr(self.dbg & 2):
                    mSBS[l, key, ct * 4 + jb] = dsb
                if const_expr(self.wsp):
                    if warp < 2:
                        if lane == 0:
                            st_global_u8(global_address(mSBS) + wtile * 4 + 2 * crank + qd % 2, dsb)
                if const_expr(self.dbg & 8):
                    a0_ = global_address(mHash) + (Int64(l) * N + q0) * 32
                    lk_cols_u32(hsl, lane, a0_ + 8)
                    lk_cols_u32(hcl, lane, a0_ + 16)
                    lk_cols_f64(dsv, lane, a0_ + 24)
                if const_expr(self.dbg & 16):
                    a1_ = global_address(mHash) + (Int64(l) * N + q0) * 32
                    pdl = [abs_f32(mul_f32(pv[i], mul_f32(dpv[i], ca))) for i in range(32)]
                    lk_cols_f64(dsv, lane, a1_)
                    lk_cols_f64(pdl, lane, a1_ + 8)
                # ---- SFA words (warps 0-3 write their own quadrant; rows r at lanes r and r + 64) ----
                if warp < 4:
                    bp = [Int32(0)] * 4
                    bs = [Int32(0)] * 4
                    for j in cutlass.range_constexpr(4):
                        psj = Int32(0)
                        if const_expr("xpt" in self.opts):
                            psj = ld_shared_b32(xp + (j * 64 + r) * 4)
                        else:
                            psj = ld_shared_b32(xp + (r * 4 + j) * 4)
                        dsc_ = Float32(0.0)
                        qsc_ = Float32(0.0)
                        if const_expr("meta" in self.opts):
                            dsc_ = ld_shared_f32(ldb + 1040 + j * 4)
                            qsc_ = ld_shared_f32(ldb + 1024 + j * 4)
                        else:
                            dsc_ = Float32(mdOs[b, h, ct * 4 + j])
                            qsc_ = Float32(mQs[b, h, ct * 4 + j])
                        bp[j] = add_sb_clamp(psj, sb_of(dsc_))
                        bs[j] = add_sb_clamp(dsb, sb_of(qsc_))
                    tmem_st_x1(tb + self.COL_SFA_P + 4 * stg + (qd % 2), pack4(bp[0], bp[1], bp[2], bp[3]))
                    tmem_st_x1(tb + self.COL_SFA_S + 4 * stg + (qd % 2), pack4(bs[0], bs[1], bs[2], bs[3]))
                    cute.arch.fence_view_async_tmem_store()
                if const_expr("meta" in self.opts and not (self.kn == "meta" and self.unsafe)):
                    mf.release()
                cute.arch.fence_view_async_shared()
                tc_fence_before()
                if const_expr(self.kn == "cd"):
                    if const_expr(not self.unsafe):
                        spin(self.DLY)
                if const_expr(not (self.kn == "cd" and self.unsafe)):
                    self._arr(kbar + 16 + 8 * stg, kbar + 72 + 8 * stg, crank)    # release.cluster on the leader's barrier
                if const_expr(self.dqc and self.kn == "cr" and self.unsafe):
                    if t == 0:       # UNSAFE cr: carry restored after the t = 0 publication + a delay (the first dV / dK MMA runs first)
                        spin(self.DLY)
                        self._carry_restore(nfirst, tb, mDump, mCodesP, b, key, hx, Hk, nfull, qd, hf)
            # ---- epilogue: dK / dV rows (64 cols of this warp's half) -> bf16 ----
            if const_expr(not (self.kn == "fin" and self.unsafe)):
                mbar_wait_parity(F_F, Int32(0))
            tc_fence_after()
            if const_expr(self.dqc):
                if cout != 0:
                    self._carry_store(tb, mDump, mCodesP, b, key, hx, Hk, nfull, qd, hf)
                else:
                    self._epi_bf16(tb, mdK, mdV, b, key, hx, Hk, N, qd, hf)
            else:
                self._epi_bf16(tb, mdK, mdV, b, key, hx, Hk, N, qd, hf)
            tc_fence_before()
            if const_expr(self.kn == "fin" and self.unsafe):
                mbar_wait_parity(F_F, Int32(0))  # UNSAFE: dK / dV were read before the final commit
            self._arr(E_F, kbar + 88, crank)
            # code-stage tail: the last commit on each stage has landed in this CTA before teardown (no in-flight arrive)
            for s_ in cutlass.range_constexpr(2):
                nu = (total - s_ + 1) // 2
                if nu > 0:
                    mbar_wait_parity(kbar + 32 + 8 * s_, (nu - 1) & 1)

        # teardown (ordinv K9): every TMEM reader / MMA issuer fences before the CTA + cluster barriers, the deallocator after
        tc_fence_before()
        cute.arch.barrier(barrier_id=4, number_of_threads=self.threads)
        tmem.relinquish_alloc_permit()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tc_fence_after()
        tmem.free(tmem_ptr)
