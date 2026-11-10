from typing import Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from quack import copy_utils, layout_utils

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from delta_matching.sm100.attention_math_sm100 import dm_mx256_c, dm_mx256_dp_scale, dm_mx256_l2, dm_mx256_p_pair
from delta_matching.sm100.cute_ptx_sm100 import tmem_ld_16x64b_x16, tmem_ld_16x64b_x32
from delta_matching.sm100.tcgen05_fence_sm100 import FenceSkipSite, begin_kernel_fences, end_kernel_fences, mark_fenced
import flash_attn.cute.pipeline as pipeline_custom
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
)
from flash_attn.cute.utils import AuxData
from flash_attn.cute.config import (
    SM100_MAX_KV_STAGE,
    FwdConfig,
    sm100_smem_capacity,
    sm100_uneven_kv_smem,
)


# NOTE [SM100 forward coordinate spaces]
# R = m_block_size, C = cta_group_size, S = q_stage; a work tile spans R*C*S effective rows.
#
#   physical CTA
#        |  scheduler (NOTE [SM100 forward work coordinates], tile_scheduler.py)
#   (work_m, head, batch, split)
#        |  work_q_block(work_m, stage, cta_rank)
#   q_block                             R-row block index
#        |  * R + row_in_cta
#   q_row                               batch-local row
#
# Row strides for (row_in_cta, cta_rank, stage, work_m) are (1, R, R*C, R*C*S).


def work_q_block(work_m, stage, cta_rank, *, q_stage: int, cta_group_size: int):
    """R-row block index for one CTA and Q stage of a work tile.

    See NOTE [SM100 forward coordinate spaces].
    """
    return (work_m * q_stage + stage) * cta_group_size + cta_rank


from flash_attn.cute.flash_fwd_sm100 import DescaleTensors  # same class as interface.py builds


class DMPass0Hd256Sm100:
    """hd256 MXFP8 (sm100-mx at D=256, preset sm100-mx-d256) matched-delta pass 0: 1-CTA, q_stage 2, kv_stage 2,
    two lanes per query row. Per KV tile S = Q8 K8^T and dP = dO8 V8^T (both kind::f8f6f4, V loaded K-major like K;
    S slot i at TMEM column 128*i, dP slot i at 256 + 128*i). Q / K / V / dO carry one pow2 (UE8M0) scale per 32 tokens
    over all D, so the scales sit on the MMAs' non-contraction dims and multiply the fp32 accumulator after the plain
    MMA. P / c / dP-scale from attention_math_sm100.dm_mx256_*; num += P*(dP*a), den += P with per-lane TwoSum
    cross-tile sums and one fp64 combine; delta = num / den written to mLSE (B, H, N).
    aux_data.tensors = [qs, ks, vs, dos, dO8 (uint8 view, (B,N,H,D)), lse (B,H,N) natural]."""


    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        has_tile_count_semaphore: bool = False,
        seqlen_k_per_split: Optional[int] = None,
        *,
        config: FwdConfig,
    ):
        """config is a validated SM100/SM110 FwdConfig; the kernel copies its fields."""
        assert head_dim == 256 and head_dim_v == 256, "DMPass0Hd256Sm100 is the head_dim 256 pass 0"
        assert not (is_local or score_mod is not None or mask_mod is not None or paged_kv_non_tma or is_varlen_q
                    or has_tile_count_semaphore or seqlen_k_per_split is not None) and has_aux_tensors
        assert config.tile_m == 128 and config.tile_n == 128 and config.q_stage == 2 and config.num_splits == 1
        assert not (config.use_2cta_instrs or config.pack_gqa or config.use_tma_o or config.use_s_ping_pong
                    or config.use_clc_scheduler or config.overlap_sO_sQ or config.lpt_global)
        self.head_dim_padded = head_dim
        self.head_dim_v_padded = head_dim_v
        self.m_block_size = config.tile_m
        self.n_block_size = config.tile_n
        self.q_stage = config.q_stage
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch.is_family_of(Arch.sm_100f) or self.arch.is_family_of(Arch.sm_110f), \
            "Only SM 10.x and 11.x are supported"

        self.cta_group_size = 1
        self.cta_tiler = (self.q_stage * self.m_block_size, self.n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (self.m_block_size, self.n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (self.m_block_size, self.head_dim_v_padded, self.n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = config.is_static_persistent
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.kv_stage = config.kv_stage
        assert 2 <= self.kv_stage <= SM100_MAX_KV_STAGE, f"kv_stage={self.kv_stage}"
        self.s_stage = 2
        assert not sm100_uneven_kv_smem(self.head_dim_padded, self.head_dim_v_padded, self.kv_stage, False)

        if self.is_causal:
            self.TileScheduler = SingleTileLPTScheduler
        elif self.is_persistent:
            self.TileScheduler = StaticPersistentTileScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        # Two lanes per query row (tcgen05.ld.16x64b), 4 compute warps / SMSP: warps 0-3 / 4-7 = stage 0 / 1
        # rows 0-15 of each quadrant, warps 8-11 / 12-15 (the "correction" ids) = stage 0 / 1 rows 16-31.
        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11, 12, 13, 14, 15)
        self.mma_warp_id = 16
        self.load_warp_ids = (18,)
        self.empty_warp_ids = (19, 17)
        self.threads_per_cta = 640

        self.tmem_s_offset = [0, self.n_block_size]  # 0, 128
        # dP slot i at 256 + 128 * i (placed by n_block, not by head_dim_v): S0 0, S1 128, dP0 256, dP1 384
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.n_block_size
            for i in range(self.q_stage)
        ]  # 256, 384

        self.num_regs_softmax = config.registers.softmax
        self.num_regs_correction = config.registers.correction
        self.num_regs_other = config.registers.other

        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d)
        mK: cute.Tensor,  # (b_k, s_k, h_k, d)
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        tile_count_semaphore: Optional[cute.Tensor] = None,
        virtual_batch_idx_ptr: Optional[cute.Tensor] = None,
        num_nheads_in_l2_ptr: Optional[cute.Tensor] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mCuTotalSplitsMBlocks: Optional[cute.Tensor] = None,
        mBlocksToBatchIdx: Optional[cute.Tensor] = None,
        max_seqlen_q: Int32 | int | None = None,
        tensormap_workspace: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Lay out the tensors, build the TMA atoms, MMAs and scheduler, and launch the kernel."""
        assert mCuSeqlensQ is None and mCuSeqlensK is None and mSeqUsedQ is None and mSeqUsedK is None
        assert mPageTable is None and learnable_sink is None and blocksparse_tensors is None and max_seqlen_q is not None
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        # P0: dO codes (uint8 aux tensor) viewed as fp8, same layout handling as Q; lse input
        mdO_u8 = aux_data.tensors[4]
        mdO = cute.make_tensor(
            cute.recast_ptr(mdO_u8.iterator, dtype=self.q_dtype), cute.select(mdO_u8.layout, mode=Q_layout_transpose)
        )
        mdO = assume_tensor_aligned(mdO)
        # (s_k, d, h_k, b_k)
        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        O_layout_transpose = [1, 3, 2, 0]
        LSE_layout_transpose = [2, 1, 0]
        num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
        # P0: V stays (s, d, h, b) and is loaded K-major exactly like K

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")

        cta_group = tcgen05.CtaGroup.ONE
        q_major_mode = cute.nvgpu.OperandMajorMode.K
        k_major_mode = cute.nvgpu.OperandMajorMode.K
        v_major_mode = cute.nvgpu.OperandMajorMode.MN
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = cute.nvgpu.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        sV_layout = sK_layout  # P0: V is a K-major B operand of dP = dO V^T

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        self.tma_copy_bytes["Q"] *= 2  # P0: Q and dO tiles land on the same pipeline_q stage
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        tma_atom_dO, mdO = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mdO,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )

        # TMA load for K
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        # TMA load for V (P0: K-major, same as K)
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )

        eff_seqlen_q = max_seqlen_q
        TileScheduler = self.TileScheduler
        num_work_m = cute.ceil_div(eff_seqlen_q, self.cta_tiler[0] * self.cta_group_size)
        tile_sched_args = TileSchedulerArguments(
            num_work_m,
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            num_splits,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mQ.shape[1],  # P0: V is not transposed; head dim
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=False,
            lpt_global=False,
            num_splits_dynamic_ptr=num_splits_dynamic_ptr,
            virtual_batch_idx_ptr=virtual_batch_idx_ptr,
            num_nheads_in_l2_ptr=num_nheads_in_l2_ptr,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            cu_total_splits_m_blocks_ptr=mCuTotalSplitsMBlocks,
            blocks_to_batch_idx_ptr=mBlocksToBatchIdx,
            tile_count_semaphore=tile_count_semaphore.iterator if tile_count_semaphore is not None else None,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=SchedulingMode.STATIC
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # pass 0 never writes O -> no sO buffer (a 128 KiB bf16 sO would not fit)
        sQ_size = cute.cosize(sQ_layout)
        # K and V alias the same physical buffer and may have different extents.
        sKV_size = cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_O_epi: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            mbar_load_epi: cute.struct.MemRange[cutlass.Int64, 0]
            mbar_s0_s1_sequence: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: cutlass.Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            sched_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 0]
            sched_response: cute.struct.MemRange[Int32, 0]
            # Large TMA buffers with 1024-byte alignment
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, 0], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sKV_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        smem_capacity = sm100_smem_capacity(self.arch.major * 10 + self.arch.minor)
        assert SharedStorage.size_in_bytes() <= smem_capacity, (
            f"SMEM {SharedStorage.size_in_bytes()} > {smem_capacity}: kv_stage={self.kv_stage}"
        )

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, None)
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, False, aux_data.tensors, None)

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            descale_tensors,
            sQ_layout,
            sK_layout,
            sV_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            num_splits_dynamic_ptr,
            tile_count_semaphore,
            virtual_batch_idx_ptr,
            num_nheads_in_l2_ptr,
            aux_data,
            fastdiv_mods,
            tma_atom_dO,
            mdO,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b)
        mK: cute.Tensor,  # (s_k, d, h_k, b_k)
        mV: cute.Tensor,  # (s_k, d, h_k, b_k), loaded K-major like K
        mO: cute.Tensor,  # unused (pass 0 writes no O); a kernel parameter of the shipped binary
        mLSE: cute.Tensor,  # (s_q, h, b): delta out
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        descale_tensors: Optional[DescaleTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        tile_count_semaphore: Optional[cute.Tensor] = None,
        virtual_batch_idx_ptr: Optional[cute.Tensor] = None,
        num_nheads_in_l2_ptr: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        tma_atom_dO: Optional[cute.CopyAtom] = None,
        mdO: Optional[cute.Tensor] = None,
    ):
        """Warp-specialized pass 0: load warp (Q, dO, K, V by TMA), MMA warp (S and dP per KV tile), and 16 compute
        warps (two lanes per query row) that form P, num and den and write delta."""
        begin_kernel_fences(1)   # tcgen05 thread-sync fences (FENCE_MODE)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO):
                cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        mma_tile_coord_v = 0

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # TMEM handoffs span warp-role branches, so all waits and arrivals must be unaligned.
        tmem_alloc_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len(
                (self.mma_warp_id,
                 *self.softmax0_warp_ids,
                 *self.softmax1_warp_ids,
                 *self.correction_warp_ids)
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # The MMA warp signals S ready (producer); all 16 compute warps release the S slot (consumer).
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=ThreadCooperativeGroup(
                cute.arch.WARP_SIZE * len(self.softmax0_warp_ids) * self.cta_group_size * 2
            ),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(  # P0: dP slot pipeline (MMA -> compute warps)
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=ThreadCooperativeGroup(
                cute.arch.WARP_SIZE * len(self.softmax0_warp_ids) * self.cta_group_size * 2
            ),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_s0_s1_sequence.data_ptr(),
            num_stages=2,
            producer_group=softmax_threads,
            consumer_group=softmax_threads,
            defer_sync=True,
        )
        pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2
        )


        # mark the tcgen05 hand-off pipelines / barriers (mark_fenced)
        mark_fenced("ua", pipeline_s_p_o, pipeline_o_acc)
        mark_fenced("au", pipeline_p_lastsplit)
        mark_fenced("bar", sm_stats_barrier, tmem_alloc_barrier)
        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))

        block_info = BlockInfo(
            self.cta_tiler[0] * self.cta_group_size,
            self.cta_tiler[1],
            self.is_causal,
            False,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
            num_splits=num_splits,
            pack_split_idx=num_splits_dynamic_ptr is not None,
            num_n_blocks_per_split=None,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
            mCuTotalMBlocks=None,
            mCuBlockIdxOffsets=None,
        )
        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                sdO=sdO,
                tma_atom_dO=tma_atom_dO,
                mdO=mdO,
            )

        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                sQ,
                sK,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_o_acc,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                sdO=sdO,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem.free(tmem_ptr)

        # The constant 'or' term is kept on purpose: without it the compiler lays the role blocks out in another order.
        if warp_idx <= self.softmax1_warp_ids[-1] or const_expr(False):
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if warp_idx < self.softmax1_warp_ids[0]:
                self.tl_loop(0, 0, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_o_acc, aux_data, block_info,
                             num_splits, SeqlenInfoCls, tile_scheduler)
            else:
                self.tl_loop(1, 0, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_o_acc, aux_data, block_info,
                             num_splits, SeqlenInfoCls, tile_scheduler)
            tmem_alloc_barrier.arrive_unaligned()

        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_correction)   # 640-thread launch: 96 regs
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if warp_idx < self.correction_warp_ids[4]:
                self.tl_loop(0, 1, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_o_acc, aux_data, block_info,
                             num_splits, SeqlenInfoCls, tile_scheduler)
            else:
                self.tl_loop(1, 1, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_o_acc, aux_data, block_info,
                             num_splits, SeqlenInfoCls, tile_scheduler)
            tmem_alloc_barrier.arrive_unaligned()

        end_kernel_fences()
        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        sdO: Optional[cute.Tensor] = None,
        tma_atom_dO: Optional[cute.CopyAtom] = None,
        mdO: Optional[cute.Tensor] = None,
    ):
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            head_idx_kv = head_idx // self.qhead_per_kvhead
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))  # P0: K-like
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_qk.partition_B(gV)  # P0
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (work_m, 0))  # (128 * 2, 128)
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )  # (128, 128, 2)
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
            gdO = cute.local_tile(mdO_cur, tiler_gQ, (work_m, 0))
            gdO = layout_utils.select(cute.flat_divide(gdO, (self.mma_tiler_qk[0],)), mode=[0, 2, 1])
            tSgdO = thr_mma_qk.partition_A(gdO)
            load_dO_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO, 0, cute.make_layout(1), tSgdO, sdO
            )
            load_Q = partial(self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase,
                             load_dO_fn=load_dO_fn)

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
            )

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                work_m,
                split_idx=split_idx,
                num_splits=num_splits,
            )
            if self.process_work_tile(block_info, seqlen, n_block_min, n_block_max):
                load_K(block=n_block_max - 1, producer_state=kv_producer_state)  # K0
                load_Q(block=0, stage=0)
                kv_producer_state.advance()
                load_Q(block=1, stage=1)
                q_producer_phase ^= 1
                load_V(block=n_block_max - 1, producer_state=kv_producer_state)  # V0
                kv_producer_state.advance()
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    n_block = n_block_max - 2 - i
                    load_K(block=n_block, producer_state=kv_producer_state)  # Ki
                    kv_producer_state.advance()
                    load_V(block=n_block, producer_state=kv_producer_state)  # Vi
                    kv_producer_state.advance()

            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail for the TMA-Q producer warp.
        pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler=None,
        sdO: Optional[cute.Tensor] = None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)

        qk_mma_op = tiled_mma_qk.op
        qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        # P0: V is K-major (sK view); no PV operand
        q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]

        sm100_utils.declare_ptx_smem_desc(q_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="fa_fwd_q_smem_desc")
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        do_smem_start = [sm100_desc.make_smem_desc_start_addr(sdO[None, None, None, stage].iterator) for stage in range(self.q_stage)]
        sm100_utils.declare_ptx_smem_desc(do_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="dm_p0_do_smem_desc")

        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="fa_fwd_q_smem_desc",
                idesc_var_name="fa_fwd_qk_mma_idesc",
                kind=qk_mma_kind,
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]
        # P0: dP_stage = dO_stage V^T into the dP slots (same instruction shape as S = Q K^T)
        gemm_dPi = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_o_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="dm_p0_do_smem_desc",
                idesc_var_name="fa_fwd_qk_mma_idesc",
                kind=qk_mma_kind,
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        dm_slot_empty_phase = Int32(1)  # P0: S/dP slots start empty

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                work_m,
                split_idx=split_idx,
                num_splits=num_splits,
            )
            block_iter_count = n_block_max - n_block_min
            process_tile = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)

            if process_tile:
                # P0 schedule per KV tile: S_s = Q_s K^T (both stages), then dP_s = dO_s V^T, commit stage s.
                for stage in cutlass.range_constexpr(self.q_stage):
                    pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                mma_q_consumer_phase ^= 1
                for i in cutlass.range(block_iter_count, unroll=1):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, dm_slot_empty_phase)
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                        )
                        pipeline_s_p_o.producer_commit_w_index(stage)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    sV_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        pipeline_o_acc.producer_acquire_w_index_phase(stage, dm_slot_empty_phase)
                        gemm_dPi[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator)
                        )
                        pipeline_o_acc.producer_commit_w_index(stage)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    dm_slot_empty_phase ^= 1
                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

    def _q_block(self, work_m, stage, cta_rank=0):
        return work_q_block(
            work_m, stage, cta_rank, q_stage=self.q_stage, cta_group_size=self.cta_group_size
        )

    @cute.jit
    def _kv_head_idx(self, head_idx: Int32) -> Int32:
        """Map query-head tile index -> KV-head index."""
        return head_idx // self.qhead_per_kvhead

    @cute.jit
    def tl_loop(self, stage: int, rh: int, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_dp, aux_data, block_info,
                num_splits, SeqlenInfoCls, tile_scheduler):
        """Two lanes per query row, 4 compute warps / SMSP (stage, row half rh).
        tcgen05.ld.16x64b: thread t of the warp holds TMEM lane quad*32 + 16*rh + t//4 + 8*(t&1) and the columns of
        parity p = (t>>1)&1, so the row's two lanes hold the x / y halves of every packed pair. Every scalar den /
        num-acc / TwoSum chain lives on one lane (bitwise P, dP, num, den, delta); the halves meet only in the final
        fp64 combine (shuffle xor 2)."""
        aux = aux_data.tensors
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        quad = warp_idx % 4
        lane = cute.arch.lane_idx()
        par = (lane >> 1) & 1
        row_in = quad * 32 + 16 * rh + (lane >> 2) + 8 * (lane & 1)
        lane_base = Int32((quad * 32 + 16 * rh) << 16)
        tSAcc = tStS[(None, None), 0, 0, stage]
        s_addr = Int32(tSAcc.iterator.toint()) + lane_base
        d_addr = s_addr + Int32(self.tmem_o_offset[0] - self.tmem_s_offset[0])
        s_full_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            has_work = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)
            m_block = self._q_block(work_m, stage)
            row = m_block * self.m_block_size + row_in
            rowc = cutlass.min(row, seqlen.seqlen_q - 1)
            # hd256 MXFP8 (sm100-mx): one pow2 Q / dO scale per 32 rows, one K / V scale per 32 keys (4 per n_block)
            qsv = Float32(aux[0][batch_idx, head_idx, rowc // 32])
            dosv = Float32(aux[3][batch_idx, head_idx, rowc // 32])
            l2 = dm_mx256_l2(Float32(aux[5][batch_idx, head_idx, rowc]))
            # this lane's component of the TwoSum pairs, packed as (num, den)
            ts_s = (Float32(0.0), Float32(0.0))
            ts_e = (Float32(0.0), Float32(0.0))
            # the raw K / V scales of a tile are loaded one iteration ahead
            nb0 = cutlass.max(n_block_max - 1, 0) * (self.n_block_size // 32)
            pks = tuple(Float32(aux[1][batch_idx, kv_head_idx, nb0 + j]) for j in range(4))
            pvs = tuple(Float32(aux[2][batch_idx, kv_head_idx, nb0 + j]) for j in range(4))
            if has_work:
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    cs = [None] * 4
                    as_ = [None] * 4
                    for j in cutlass.range_constexpr(4):
                        cs[j] = dm_mx256_c(pks[j], qsv)
                        as_[j] = dm_mx256_dp_scale(pvs[j], dosv)
                    nbn = cutlass.max(n_block - 1, n_block_min) * (self.n_block_size // 32)
                    pks = tuple(Float32(aux[1][batch_idx, kv_head_idx, nbn + j]) for j in range(4))
                    pvs = tuple(Float32(aux[2][batch_idx, kv_head_idx, nbn + j]) for j in range(4))
                    # den chains of this lane's component: d01 = (c0, c1), d23 = (c2, c3) (den[v % 4], v = pair)
                    d01 = (Float32(0.0), Float32(0.0))
                    d23 = (Float32(0.0), Float32(0.0))
                    num = (Float32(0.0), Float32(0.0))   # (num of this component, unused)
                    pipeline_s_p_o.consumer_wait_w_index_phase(stage, s_full_phase)
                    diag = False
                    if const_expr(self.is_causal):
                        diag = (n_block + 1) * self.n_block_size > m_block * self.m_block_size
                    kbase = n_block * self.n_block_size
                    r = cute.make_rmem_tensor(cute.make_layout(64), self.qk_acc_dtype)
                    # both S halves to registers first; the S slot is released before any exp, so the MMA warp's
                    # next S does not wait on this tile's dP
                    for h in cutlass.range_constexpr(2):
                        vals = tmem_ld_16x64b_x32(s_addr + Int32(64 * h))
                        for i in cutlass.range_constexpr(32):
                            r[32 * h + i] = vals[i]
                    cute.arch.fence_view_async_tmem_load()
                    with FenceSkipSite("r0"):
                        pipeline_s_p_o.consumer_release_w_index(stage)
                    for h in cutlass.range_constexpr(2):
                        if const_expr(self.is_causal):
                            if diag:
                                for i in cutlass.range_constexpr(32 * h, 32 * h + 32):
                                    r[i] = -Float32.inf if kbase + 2 * i + par > row else r[i]
                        for j in cutlass.range_constexpr(2 * h, 2 * h + 2):
                            for v in cutlass.range_constexpr(0, 16, 2):
                                i0 = 16 * j + v
                                p = dm_mx256_p_pair(r[i0], r[i0 + 1], cs[j], l2)
                                r[i0], r[i0 + 1] = p
                                if const_expr(v % 4 == 0):
                                    d01 = cute.arch.add_packed_f32x2(d01, p)
                                else:
                                    d23 = cute.arch.add_packed_f32x2(d23, p)
                        if const_expr(h == 0):
                            pipeline_dp.consumer_wait_w_index_phase(stage, s_full_phase)
                        for j in cutlass.range_constexpr(2 * h, 2 * h + 2):
                            dpv = tmem_ld_16x64b_x16(d_addr + Int32(32 * j))
                            cute.arch.fence_view_async_tmem_load()
                            if const_expr(j == 3):
                                with FenceSkipSite("r1"):
                                    pipeline_dp.consumer_release_w_index(stage)
                            a01 = (Float32(0.0), Float32(0.0))
                            a23 = (Float32(0.0), Float32(0.0))
                            for v in cutlass.range_constexpr(0, 16, 2):
                                i0 = 16 * j + v
                                if const_expr(v % 4 == 0):
                                    a01 = cute.arch.fma_packed_f32x2((r[i0], r[i0 + 1]), (dpv[v], dpv[v + 1]), a01)
                                else:
                                    a23 = cute.arch.fma_packed_f32x2((r[i0], r[i0 + 1]), (dpv[v], dpv[v + 1]), a23)
                            # acc01 = acc0 + acc1, acc23 = acc2 + acc3, num = fma(acc01 + acc23, a_j, num) per component
                            t = cute.arch.add_packed_f32x2((a01[0], a23[0]), (a01[1], a23[1]))
                            u = cute.arch.add_packed_f32x2((t[0], Float32(0.0)), (t[1], Float32(0.0)))
                            num = cute.arch.fma_packed_f32x2((u[0], Float32(0.0)), (as_[j], Float32(0.0)), num)
                    # dent = (den + den1) + (den2 + den3) per component
                    t = cute.arch.add_packed_f32x2((d01[0], d23[0]), (d01[1], d23[1]))
                    dent = cute.arch.add_packed_f32x2((t[0], Float32(0.0)), (t[1], Float32(0.0)))
                    # packed TwoSum of (num, den) of this component
                    b = (num[0], dent[0])
                    sn = cute.arch.add_packed_f32x2(ts_s, b)
                    bb = cute.arch.add_packed_f32x2(sn, (-ts_s[0], -ts_s[1]))
                    an = cute.arch.add_packed_f32x2(ts_s, (-(sn[0] - bb[0]), -(sn[1] - bb[1])))
                    br = cute.arch.add_packed_f32x2(b, (-bb[0], -bb[1]))
                    ts_e = cute.arch.add_packed_f32x2(ts_e, cute.arch.add_packed_f32x2(an, br))
                    ts_s = sn
                    s_full_phase ^= 1
            # partner lane (other column parity of the same row) = lane ^ 2
            o_sn = cute.arch.shuffle_sync_bfly(ts_s[0], offset=2)
            o_sd = cute.arch.shuffle_sync_bfly(ts_s[1], offset=2)
            o_en = cute.arch.shuffle_sync_bfly(ts_e[0], offset=2)
            o_ed = cute.arch.shuffle_sync_bfly(ts_e[1], offset=2)
            if par == 0:
                tn64 = (Float64(ts_s[0]) + Float64(o_sn)) + (Float64(ts_e[0]) + Float64(o_en))
                td64 = (Float64(ts_s[1]) + Float64(o_sd)) + (Float64(ts_e[1]) + Float64(o_ed))
                delta = Float32(tn64 / td64) if td64 != Float64(0.0) else Float32(0.0)
                if row < seqlen.seqlen_q:
                    mLSE[row, head_idx, batch_idx] = delta
            work_tile = tile_scheduler.advance_to_next_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
        load_dO_fn: Optional[Callable] = None,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))
        load_dO_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))

    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
    ):
        assert K_or_V in ("K", "V")
        stage = producer_state.index
        extra_tx_count = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        pipeline_kv.producer_acquire(producer_state, extra_tx_count=extra_tx_count)
        tXsX_cur = tXsX[None, stage]
        tXgX_cur = tXgX[None, block]
        cute.copy(
            tma_atom, tXgX_cur, tXsX_cur,
            tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
        )

    @cute.jit
    def process_work_tile(
        self,
        block_info: BlockInfo,
        seqlen_info: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
    ):
        return block_info.has_kv_work(n_block_min, n_block_max)
