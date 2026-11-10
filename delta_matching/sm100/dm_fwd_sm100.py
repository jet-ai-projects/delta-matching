# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# - split-kv
# Unsupported features that will be added later:
# - page size != 128
# - more hdim (192, 256)
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import math
from typing import Tuple, Callable, Optional, Literal, NamedTuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import ClcDynamicPersistentTileScheduler
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from quack import copy_utils, layout_utils

from flash_attn.cute.paged_kv import PagedKVManager, resolve_paged_token
from flash_attn.cute.tensormap_utils import acquire_tensormap, bound_rows, cta_tensormap_slot
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from delta_matching.sm100 import dm_utils
import flash_attn.cute.pipeline as pipeline_custom
import cutlass.pipeline as cutlass_pipeline
from flash_attn.cute.mask import AttentionMask, mask_r2p_lambda, r2p_bitmask_below
from flash_attn.cute.softmax import (
    SoftmaxSm100,
    apply_score_mod_inner,
    apply_learnable_sink,
    load_learnable_sink,
)
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    get_total_block_count,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
    handle_block_sparse_empty_tile_correction_sm100,
)
from flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from cutlass.cute import FastDivmodDivisorV2
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    SchedulerState,
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    DynamicPersistentVarlenScheduler,
)
from flash_attn.cute.fa_logging import fa_log, fa_printf
from flash_attn.cute.utils import smid
from flash_attn.cute.utils import AuxData
from flash_attn.cute.config import (
    SM100_LAUNCH_REGISTERS,
    SM100_MAX_KV_STAGE,
    SM100_THREADS_PER_CTA,
    FwdConfig,
    sm100_smem_capacity,
    sm100_uneven_kv_smem,
)

# Note [Low Precision Scaling]
# P is in (0, 1] and is cast to the input dtype before P @ V, so scaling it by 2^max_offset
# spends the dtype's unused upper code points on the probability tail. A positive
# rescale_threshold lets the row max lag by that many log2 units, so P can reach
# 2^(max_offset + rescale_threshold); above the dtype max the top probabilities saturate
# while the FP32 denominator still counts them in full, shrinking the output (#2716).

# log2 of the largest finite value representable in each supported input dtype.
_LOG2_DTYPE_MAX = {
    cutlass.Float8E4M3FN: math.log2(448.0),
    cutlass.Float8E5M2: math.log2(57344.0),
    cutlass.Float16: math.log2(65504.0),
    cutlass.BFloat16: math.log2(3.3895313892515355e38),
}


# NOTE [SM100 forward coordinate spaces]
# R = m_block_size, C = cta_group_size, S = q_stage; a work tile spans R*C*S effective rows.
#
#   physical CTA / CLC response
#        |  scheduler (NOTE [SM100 forward work coordinates], tile_scheduler.py)
#   (work_m, head, batch, split)        same on every CTA of a cluster
#        |  work_q_block(work_m, stage, cta_rank)
#   q_block                             R-row block index
#        |  * R + row_in_cta
#   q_row                               batch-local effective row, packed under pack_gqa
#        |  layouts, seqlen_info offsets
#   memory
#
# Row strides for (row_in_cta, cta_rank, stage, work_m) are (1, R, R*C, R*C*S).
# R=128, C=S=2, work_m=0:
#              CTA 0       CTA 1
#   stage 0    0..127      128..255
#   stage 1  256..383      384..511
#
# Users: LSE and non-TMA Q/O index by the CTA's q_block. TMA Q/O tile work_m and partition by
# stage and CTA. Mask and score_mod pass cta_rank=0 because their thr_mma partitions already
# include the CTA's rows; the block-sparse stage end is (work_q_block(work_m, stage, 0) + C) * R.
# q_row drives the Q/O/LSE tail predicates and the packed-GQA sink head.


def work_q_block(work_m, stage, cta_rank, *, q_stage: int, cta_group_size: int):
    """R-row block index for one CTA and Q stage of a work tile.

    See NOTE [SM100 forward coordinate spaces].
    """
    return (work_m * q_stage + stage) * cta_group_size + cta_rank


from flash_attn.cute.flash_fwd_sm100 import DescaleTensors  # same class as interface.py builds


class DMForwardSm100:
    """FA4 sm100 forward forked for the Delta-Matching sm100-mx numerics.
    descale_tensors carries (qs, ks, vs): power-of-2 fp32 scales per 32 tokens, (B,H,N/32), (B,Hk,N/32), (B,Hk,N/32).
    P is MXFP8: e4m3 codes per (row, 32-key block) in [128,256) at the block max, UE8M0 SFA = 2^(kb-7-m),
    SFB = V block scale; PV runs as tcgen05.mma kind::mxf8f6f4.block_scale with P and both SFs in TMEM."""


    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
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
        self.use_tma_KV = not paged_kv_non_tma
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = config.tile_m
        self.n_block_size = config.tile_n
        self.q_stage = config.q_stage
        assert self.q_stage in [1, 2]
        self.use_2cta_instrs = config.use_2cta_instrs
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = self.n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size
        assert seqlen_k_per_split is None or seqlen_k_per_split % self.n_block_size == 0
        self.num_n_blocks_per_split = (
            seqlen_k_per_split // self.n_block_size if seqlen_k_per_split is not None else None
        )
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch.is_family_of(Arch.sm_100f) or self.arch.is_family_of(Arch.sm_110f), \
            "Only SM 10.x and 11.x are supported"

        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        # cta_tiler M covers one CTA; a scheduler work tile spans cta_group_size CTAs (work_q_block).
        self.cta_tiler = (self.q_stage * self.m_block_size, self.n_block_size, self.head_dim_padded)
        # With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
        # Each CTA owns m_block_size rows; the 2CTA MMA instruction spans both.
        self.mma_tiler_qk = (
            self.cta_group_size * self.m_block_size, self.n_block_size, self.head_dim_padded
        )
        self.mma_tiler_pv = (
            self.cta_group_size * self.m_block_size, self.head_dim_v_padded, self.n_block_size
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        self.is_static_persistent = config.is_static_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = config.num_splits > 1
        self.pack_gqa = config.pack_gqa
        # Half-width fp32 sO preserves two KV stages; unchanged column bounds require exact hd256.
        self.chunked_split_epi = self.is_split_kv and head_dim_v == 256
        self.epi_head_dim_v = 128 if self.chunked_split_epi else self.head_dim_v_padded
        assert not (config.use_tma_o and self.chunked_split_epi), "hd256 SplitKV uses a chunked epilogue"
        self.use_tma_O = config.use_tma_o
        self.use_correction_warps_for_epi = not self.use_tma_O
        self.q_subtile_factor = q_subtile_factor
        self.kv_subtile_factor = kv_subtile_factor
        assert (
            not self.is_split_kv or self.head_dim_v_padded < 192 or self.chunked_split_epi
        ), "SplitKV is not supported for hdim_v >= 192 except 256"
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.score_vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        self.mask_vec_size: cutlass.Constexpr = getattr(mask_mod, "__vec_size__", 1)
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        # SM103 ld.red is profitable except for D32 when scores are unmodified. is_family_of
        # intentionally also matches later sm_10x.
        self.use_ldred_rowmax = (
            self.arch.is_family_of(Arch.sm_103f)
            and self.score_mod is None
            and self.mask_mod is None
            and self.head_dim_padded != 32
        )
        # FwdConfig carries the exp2 emulation choice; see config.select_ex2_emu.
        self.ex2_emu_freq = config.ex2_emu_freq
        self.ex2_emu_start_frg = config.ex2_emu_start_frg
        self.s0_s1_barrier = False

        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )

        # validate_fwd_config rejects CLC without TMA KV or with a varlen-Q tile-count semaphore.
        assert not config.use_clc_scheduler or (
            self.use_tma_KV and not (has_tile_count_semaphore and is_varlen_q)
        ), "CLC requires TMA KV and no varlen-Q tile-count semaphore"
        self.use_clc_scheduler = config.use_clc_scheduler
        self.dynamic_persistent = (
            has_tile_count_semaphore and is_varlen_q
        ) or self.use_clc_scheduler
        self.is_persistent = self.dynamic_persistent or self.is_static_persistent
        # FwdConfig owns the SMEM layout (config.select_sm100_smem_layout); validate_fwd_config
        # checks it fits, and __call__ asserts the resulting SharedStorage does.
        self.overlap_sO_sQ = config.overlap_sO_sQ
        self.kv_stage = config.kv_stage
        assert 2 <= self.kv_stage <= SM100_MAX_KV_STAGE, f"kv_stage={self.kv_stage}"
        self.s_stage = 2
        assert self.s_stage >= self.q_stage
        # For hdim 192,128 1CTA, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = sm100_uneven_kv_smem(
            self.head_dim_padded, self.head_dim_v_padded, self.kv_stage, self.use_2cta_instrs
        )
        self.uneven_kv_smem_offset = (
            self.n_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0
        self.sched_stages = 1
        if self.use_clc_scheduler:
            assert self.cluster_shape_mn[1] == 1, f"CLC requires cluster N == 1: {self.cluster_shape_mn}"
            assert self.cluster_shape_mn[0] in (1, 2), f"bad CLC cluster M: {self.cluster_shape_mn}"
            assert self.cluster_shape_mn[0] == self.cta_group_size, (
                f"CLC cluster M != cta_group_size: {self.cluster_shape_mn}, {self.cta_group_size}"
            )

        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler
            else SchedulingMode.DYNAMIC if self.dynamic_persistent
            else SchedulingMode.STATIC
        )

        self.use_varlen_scheduler = False
        if is_varlen_q:
            if self.dynamic_persistent and not self.use_clc_scheduler:
                self.use_varlen_scheduler = True
                self.TileScheduler = DynamicPersistentVarlenScheduler
            elif self.is_static_persistent and not self.use_clc_scheduler:
                self.TileScheduler = StaticPersistentTileScheduler
            else:
                self.use_varlen_scheduler = True
                self.TileScheduler = SingleTileVarlenScheduler
        elif self.is_causal or self.is_local or self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        elif self.is_static_persistent:
            self.TileScheduler = StaticPersistentTileScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        fa_log(1, f"TileScheduler={self.TileScheduler.__name__}, scheduling_mode={self.scheduling_mode.name}, USE_2CTA={self.use_2cta_instrs}")

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )
        # config.select_sm100_registers and its validation assume this launch shape (slice mode re-launches below).
        assert self.threads_per_cta == SM100_THREADS_PER_CTA

        self.use_tma_Q = not (self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0)
        self.use_s_ping_pong = config.use_s_ping_pong
        if self.use_s_ping_pong:
            assert self.q_stage == 1, "S ping-pong requires q_stage == 1 (one Q tile per CTA)"
            assert self.head_dim_padded in (64, 128, 256) and self.n_block_size == 128
        # S/P/O pipeline depth: two ping-pong TMEM slots for S/P, else one per Q stage
        self.s_p_o_stage = 2 if self.use_s_ping_pong else self.q_stage

        if self.q_stage == 1:
            if not self.use_tma_KV or not self.use_tma_Q:
                self.empty_warp_ids = self.empty_warp_ids + self.load_warp_ids
                self.load_warp_ids = self.softmax1_warp_ids
            else:
                self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        elif not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()

        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids

        if self.dynamic_persistent:
            assert len(self.empty_warp_ids) > 0, (
                "dynamic persistent scheduling requires an empty warp to serve as the scheduler warp"
            )
        self.scheduler_warp_id = self.empty_warp_ids[0] if self.dynamic_persistent else None
        # NOTE [global LPT] (config.py); validate_fwd_config admits it only on this scheduler path.
        self.lpt_global = config.lpt_global
        assert not self.lpt_global or (
            self.use_clc_scheduler
            and self.cta_group_size == 1
            and not self.is_split_kv
            and self.TileScheduler is SingleTileLPTScheduler
        ), "lpt_global needs 1CTA unsplit dense CLC"
        # Claim after the current tile's loads to avoid reserving work during long causal tiles.
        self.clc_late_claim = (
            self.use_clc_scheduler
            and self.cta_group_size == 1
            and self.TileScheduler is SingleTileLPTScheduler
            and not self.is_split_kv
            and len(self.load_warp_ids) == 1
        )

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # DM sm100-mx: SFA (4 cols) + SFB (4 cols) right after the fp8 P columns of each S stage
        self.dm_sf_col = self.tmem_s_to_p_offset + self.n_block_size // 4
        # variant knobs: DM_FWD_OPTS="k=v,k=v" (hashed into the compile-cache fingerprint by fwd.py)
        import os as _os
        self.dm_opts = dict(kv.split("=") for kv in _os.environ.get("DM_FWD_OPTS", "").split(",") if "=" in kv)
        # det2 defaults (bitwise, fwdrw R6): ss_pv=1 (P codes in SMEM, SS PV), kvs=6 (smem budget), lean=1; an explicit
        # DM_FWD_OPTS value overrides (e.g. "ss_pv=0,lean=0,kvs=99" = the previous forward)
        # (only where ss_pv is supported: q_stage == 2, 1-CTA, no S ping-pong; short N picks q_stage = 1 -> old path)
        # Re-enabled after the fix (sScale read/write handshake mbar_dm_cread +
        # tcgen05 thread-sync fences); an earlier revert was for the sScale WAR race. Ordering: handshake + fences.
        # (an explicit pipe=1 debug variant keeps the old path: pipe=1 bypasses the SS-PV P/SF protocol)
        if (self.q_stage == 2 and not self.use_s_ping_pong and self.cta_group_size == 1
                and self.dm_opts.get("pipe", "0") != "1"):
            # ss_pv default OFF. Every race-free sScale hand-off measured +34..45% fwd (crd, dbs, late),
            # and racy ss_pv was only 0.1-0.6% faster than the old path with lean=1,kvs=6 (bitwise, no
            # new hand-off). dbs (+2 KB SMEM) needs one K/V ring stage less: kvs=5.
            for _k, _v in (("ss_pv", "0"), ("kvs", "5" if (self.dm_opts.get("ss_pv", "0") == "1" and self.dm_opts.get("dbs", "0") == "1") else "6"), ("lean", "1")):
                self.dm_opts.setdefault(_k, _v)
        self.dm_rescale_T = float(self.dm_opts.get("T", 32.0))
        if "split" in self.dm_opts:
            # FA4 split_P_arrive hangs with the DM forward
            raise ValueError("DM_FWD_OPTS split= hangs the DM forward (unsupported); removed")
        # fwdrw: s01=1 -> FA4's S0/S1 exp-sequencing barrier (softmax WG1 exps wait for WG0's, alternating) to keep the
        # two softmax warpgroups out of phase so one's non-exp work overlaps the other's MUFU exps (bitwise: order only)
        if self.dm_opts.get("s01", "0") == "1":
            self.s0_s1_barrier = True
        # fwdrw ss_pv (layout L2/L3): P codes in SMEM (SS block-scaled PV), ONE shared S scratch at TMEM cols 256..383,
        # O0/O1 at 0/128, SF_s at 448 + 8 s. QK of the next tile only waits for the previous S consumer's TMEM loads
        # (mbarrier sfree); P_s/SF_s reuse waits for PV_s done (mbarrier pvdone). Numerics unchanged.
        if "kvs" in self.dm_opts:   # fwdrw: cap the K/V smem ring depth (budget for sP); timing-checked on the baseline
            self.kv_stage = min(self.kv_stage, int(self.dm_opts["kvs"]))
        # ss_pv applies only where its layout is supported (q_stage == 2, 1-CTA, no S ping-pong); elsewhere (e.g. short N
        # with q_stage == 1) the knob falls back to the old path, exactly like the former default block did
        self.dm_sspv = (self.dm_opts.get("ss_pv", "0") == "1" and self.q_stage == 2 and not self.use_s_ping_pong
                        and self.cta_group_size == 1)
        # dbs (opt-in with ss_pv=1): double-buffered sScale per stage (2 slots by event parity) handed off with per-warp
        # full/empty mbarriers instead of the sm_stats named barrier + single-slot cread handshake (which serialized the
        # softmax behind the correction: +35% fwd). dbs=0 restores the crd protocol.
        self.dm_dbs = self.dm_sspv and self.dm_opts.get("dbs", "0") == "1"
        # late (ss_pv default protocol): sScale write + stats arrival moved after the softmax's pvdone_s(k-1) wait, so the slot reuse is
        # ordered by the existing chain read(k-1) -> O-rescaled(k-1) -> PV(k-1) -> pvdone; no extra handshake
        self.dm_late = self.dm_sspv and not self.dm_dbs and self.dm_opts.get("late", "1") == "1"
        self.dm_crd = self.dm_sspv and not self.dm_dbs and not self.dm_late and self.dm_opts.get("crd", "1") == "1"
        self.dm_trc = int(self.dm_opts.get("trc", "0")) if self.dm_sspv else 0   # DEBUG ONLY: LSE <- correction clock counters (1 stats sync, 2 pvdone wait, 3 loop total)
        if self.dm_sspv:
            assert self.n_block_size == 128 and self.head_dim_v_padded == 128
            # the pipe=1 softmax branch stores P to TMEM and bypasses the SS-PV P/SF/pvdone protocol
            assert self.dm_opts.get("pipe", "0") != "1", "ss_pv=1 is incompatible with pipe=1"
            self.split_P_arrive = 0
            self.tmem_o_offset = [0, 128]
            self.tmem_s_offset = [256, 256]
            self.dm_sf_abs = [448, 456]
        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        # fwdmufu slice=1: MUFU-first softmax. 4 slice warpgroups (warps 0-15); slice WG q owns
        # key block q (32 keys = one MX block) of the CURRENT stage; all slices process stage 0 then stage 1 of each
        # n_block, so every exp warp feeds one stage while the other stage's PV -> QK turnaround runs. Row-level work
        # (m, acc_scale, SFA bytes, weights, l, O rescale, epilogue) moves to the correction WG (warps 16-19): slices
        # publish x_b = max(S_b) c_b and the fp32 block sum through SMEM. P codes of block q go to S_s + 32 q (the
        # block's own S columns), SF to S_s + 16 (block 0's columns, written after every slice loaded S). Same TMEM map,
        # same MMA order; every op on a value is today's op (bitwise O / LSE / P codes).
        # DEFAULT ON (fwdmufu promotion, bitwise vs the previous forward); DM_FWD_OPTS slice=0 = previous softmax layout
        self.dm_slice = (self.dm_opts.get("slice", "1") == "1" and self.q_stage == 2 and not self.use_s_ping_pong
                         and self.cta_group_size == 1 and not self.dm_sspv and self.n_block_size == 128
                         and self.head_dim_v_padded == 128 and score_mod is None and mask_mod is None
                         and not self.is_local and not self.is_split_kv and not self.pack_gqa
                         and self.dm_opts.get("pipe", "0") != "1" and self.dm_opts.get("exp", "f32") == "f32"
                         and self.dm_opts.get("kvpack", "1") == "1" and config.ex2_emu_freq == 0)
        if self.dm_slice:
            self.softmax0_warp_ids = tuple(range(16))
            self.softmax1_warp_ids = ()
            self.correction_warp_ids = (16, 17, 18, 19)
            self.mma_warp_id = 20
            if self.use_correction_warps_for_epi:
                self.epilogue_warp_ids = self.correction_warp_ids
                self.empty_warp_ids = (21, 23)
            else:
                self.epilogue_warp_ids = (21,)
                self.empty_warp_ids = (23,)
            self.load_warp_ids = (22,)
            self.scheduler_warp_id = self.empty_warp_ids[0] if self.dynamic_persistent else None
            self.threads_per_cta = 24 * cute.arch.WARP_SIZE
            self.split_P_arrive = 0
            self.dm_sl_sfcol = 16
            # setmaxnreg split (per thread). Launch = 65536 / 768 -> 80 per thread; the CTA register pool only holds what
            # setmaxnreg.dec released, so 4 (rs - 80) <= (80 - rc) + (80 - ro) or the slice WGs' alloc spins forever.
            self.dm_sl_regs = (int(self.dm_opts.get("rs", "80")), int(self.dm_opts.get("rc", "112")),
                               int(self.dm_opts.get("ro", "48")))
            _rs, _rc, _ro = self.dm_sl_regs
            assert 4 * max(_rs - 80, 0) + max(_rc - 80, 0) <= 4 * max(80 - _rs, 0) + max(80 - _rc, 0) + max(80 - _ro, 0), \
                f"setmaxnreg pool: {self.dm_sl_regs}"

        # ordinv (DEFAULT ON; ordw=0 / ordld=0 = DEBUG ONLY, the
        # pre-ordinv issue-order build): ordw makes the PV_s -> QK_s TMEM reuse PTX-ordered.
        # PV_s reads its SFA/SFB scale factors (block-scaled tcgen05.mma) from S_s's TMEM columns, and the next QK_s
        # (a different accumulator/shape/kind) overwrites S_s. PTX ISA 9.7.18.6.2 pipelines MMA A/metadata reads ->
        # later MMA D writes, but not scale-factor reads, so issue order alone is not a guarantee. ordw=1: tcgen05.commit
        # after every PV_s -> mbarrier ord[s]; the MMA warp waits it (+ fence::after_thread_sync) before the next QK_s.
        # Same MMAs, same operands: bitwise.
        self.dm_ordld = self.dm_opts.get("ordld", "1") == "1"   # ordinv: tcgen05.wait::ld before the slice kb arrival
        self.dm_ordw = (self.dm_opts.get("ordw", "1") == "1" and not self.use_s_ping_pong and not self.dm_sspv
                        and self.cta_group_size == 1)
        # ordinv planted controls (DEBUG ONLY, compiled out by default): ordpc=1 -> the MMA warp overwrites SFA of S_s (lanes 0-31)
        # with tcgen05.st right before QK_s(k+1) (where the QK overwrite lands; caught unless ordw orders it after PV_s(k));
        # ordpc=4 -> slice warps 0-3 re-load S_s cols 16..23 and check them against their first load, poisoning l on a
        # mismatch; with ordld the re-load completes (wait::ld) before the kb arrival, without it the load is moved to the
        # latest point PTX permits (after the arrival + 2 us), so a racing correction SF store is seen.
        self.dm_ordpc = int(self.dm_opts.get("ordpc", "0"))
        self.num_regs_softmax = config.registers.softmax
        self.num_regs_correction = config.registers.correction
        self.num_regs_other = config.registers.other
        if self.dm_slice:
            self.num_regs_softmax, self.num_regs_correction, self.num_regs_other = self.dm_sl_regs

        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
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
        tensormap_workspace: Optional[cute.Tensor] = None,  # (num_ctas, 16) Int64, one TMA descriptor per CTA
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")

        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        q_major_mode = cute.nvgpu.OperandMajorMode.K
        k_major_mode = cute.nvgpu.OperandMajorMode.K
        v_major_mode = cute.nvgpu.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
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

        # epi_tile is per-CTA (not full 2CTA) since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.epi_head_dim_v)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        if const_expr(self.dm_sspv):
            self.dm_tiled_mma_pv_ss = sm100_utils_basic.make_trivial_tiled_mma(
                self.v_dtype, self.v_dtype, p_major_mode, v_major_mode, self.pv_acc_dtype, cta_group, self.mma_tiler_pv[:2])
            self.dm_sP_layout = sm100_utils_basic.make_smem_layout_a(
                self.dm_tiled_mma_pv_ss, self.mma_tiler_pv, self.q_dtype, self.q_stage)
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        if const_expr(self.use_tma_Q):
            tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mQ,
                cute.select(sQ_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            gmem_tiled_copy_Q = None
        else:
            tma_atom_Q = None
            async_copy_elems = 128 // self.q_dtype.width
            num_load_threads = cute.arch.WARP_SIZE * len(self.load_warp_ids)
            threads_per_row = math.gcd(self.head_dim_padded // async_copy_elems, num_load_threads)
            gmem_tiled_copy_Q = copy_utils.tiled_copy_2d(
                self.q_dtype, threads_per_row, num_load_threads, async_copy_elems, is_async=True
            )

        tma_atom_K = None
        tma_atom_V = None
        # Bound V loads by the logical sequence length: the load warp rewrites its CTA's copy
        # of the V TMA descriptor per work tile (tensormap.replace) so rows at or past the
        # sequence end are out of bounds for TMA and arrive zero-filled. Masking P alone cannot
        # prevent unused NaN cache entries from entering the PV MMA (0 * NaN). A descriptor
        # trick with a per-launch extent is not an option: TMA prefetches past the tensor
        # footprint it was given (#3068).
        self.ragged_tma_V = (
            self.use_tma_KV
            and mPageTable is None
            and (mCuSeqlensK is not None or mSeqUsedK is not None)
            and not (isinstance(mV.stride[1], int) and mV.stride[1] == 0)
        )
        if const_expr(self.ragged_tma_V):
            assert tensormap_workspace is not None, "ragged V TMA loads need a tensormap workspace"
        mV_gmem = mV if const_expr(self.ragged_tma_V) else None
        if const_expr(self.use_tma_KV):
            # TMA load for K
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                cta_layout_vmnk.shape,
            )

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        if const_expr(max_seqlen_q is None):
            eff_seqlen_q = cute.size(mQ.shape[0])
        else:
            eff_seqlen_q = max_seqlen_q if const_expr(not self.pack_gqa) else max_seqlen_q * self.qhead_per_kvhead
        TileScheduler = self.TileScheduler
        # Every scheduling policy counts logical work tiles and gives both CTAs of a cluster
        # the same one (use_cluster_idx); see work_q_block.
        num_work_m = cute.ceil_div(eff_seqlen_q, self.cta_tiler[0] * self.cta_group_size)
        tile_sched_args = TileSchedulerArguments(
            num_work_m,
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=self.cta_group_size > 1,
            lpt_global=self.lpt_global,
            num_splits_dynamic_ptr=num_splits_dynamic_ptr,
            virtual_batch_idx_ptr=virtual_batch_idx_ptr,
            num_nheads_in_l2_ptr=num_nheads_in_l2_ptr,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            cu_total_splits_m_blocks_ptr=mCuTotalSplitsMBlocks,
            blocks_to_batch_idx_ptr=mBlocksToBatchIdx,
            tile_count_semaphore=tile_count_semaphore.iterator if tile_count_semaphore is not None else None,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_size = (
            cute.cosize(sQ_layout) if const_expr(not self.overlap_sO_sQ) else
            cutlass.max(cute.cosize(sQ_layout), cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width)
        )
        # K and V alias the same physical buffer and may have different extents.
        sKV_size = cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))

        sched_response_size = self.sched_stages * 4 if self.dynamic_persistent else 0
        sched_mbar_size = self.sched_stages * 2 if self.dynamic_persistent else 0
        load_epi_mbar_size = 2 if const_expr(self.overlap_sO_sQ) else 0

        dm_nb1, dm_nb2 = (1, 3) if self.dm_sspv else (0, 0)   # 4 x 8 B keeps the following smem 16-B aligned
        dm_nb3 = (32 if self.dm_dbs else 2) if self.dm_sspv else 0   # crd: 2 (one per stage); dbs: [full|empty][stage][slot][warp]
        dm_sc = 2 if self.dm_dbs else 1   # dbs: two sScale slots
        dm_sP_size = cute.cosize(self.dm_sP_layout) if self.dm_sspv else 0
        dm_slb = 6 if self.dm_slice else 0      # slice: kbfull[2], bsfull[2], bsempty[2]
        dm_ordn = 2 if self.dm_ordw else 0      # ordinv ordw: PV_s completion barrier per stage
        dm_slf = 2 * 2 * 4 * self.m_block_size if self.dm_slice else 0   # slice: sX[2][4][128], sB[2][4][128] fp32

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_epi: cute.struct.MemRange[Int64, load_epi_mbar_size]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            mbar_dm_sfree: cute.struct.MemRange[Int64, dm_nb1]
            mbar_dm_pvdone: cute.struct.MemRange[Int64, dm_nb2]
            mbar_dm_cread: cute.struct.MemRange[Int64, dm_nb3]
            mbar_dm_slice: cute.struct.MemRange[Int64, dm_slb]
            mbar_dm_ord: cute.struct.MemRange[Int64, dm_ordn]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_p_o_stage * self.m_block_size * 2 * dm_sc]
            sDmSlice: cute.struct.MemRange[Float32, dm_slf]
            # Scheduler buffers placed here to utilize padding before sO's 1024-byte
            # alignment. This avoids adding bytes at the end when we're at the smem limit.
            # PipelineClcFetchAsync / PipelineAsync both expect
            # 2 * sched_stages mbarriers (full + empty). Response is 4 Int32 per stage
            # (CLC HW response, or work_info written by dynamic persistent producer).
            sched_mbar_ptr: cute.struct.MemRange[Int64, sched_mbar_size]
            sched_response: cute.struct.MemRange[Int32, sched_response_size]
            # Large TMA buffers with 1024-byte alignment
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sKV_size],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, dm_sP_size],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorageSPF:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.s_p_o_stage * 2]
            # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_epi: cute.struct.MemRange[Int64, load_epi_mbar_size]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            mbar_dm_sfree: cute.struct.MemRange[Int64, dm_nb1]
            mbar_dm_pvdone: cute.struct.MemRange[Int64, dm_nb2]
            mbar_dm_cread: cute.struct.MemRange[Int64, dm_nb3]
            mbar_dm_slice: cute.struct.MemRange[Int64, dm_slb]
            mbar_dm_ord: cute.struct.MemRange[Int64, dm_ordn]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_p_o_stage * self.m_block_size * 2 * dm_sc]
            sDmSlice: cute.struct.MemRange[Float32, dm_slf]
            # Scheduler buffers placed here to utilize padding before sO's 1024-byte
            # alignment. This avoids adding bytes at the end when we're at the smem limit.
            # PipelineClcFetchAsync / PipelineAsync both expect
            # 2 * sched_stages mbarriers (full + empty). Response is 4 Int32 per stage
            # (CLC HW response, or work_info written by dynamic persistent producer).
            sched_mbar_ptr: cute.struct.MemRange[Int64, sched_mbar_size]
            sched_response: cute.struct.MemRange[Int32, sched_response_size]
            # Large TMA buffers with 1024-byte alignment
            # DM spf: sP placed BEFORE sO / sQ / sK (debug: K/V-ring overrun test)
            sP: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, dm_sP_size],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sKV_size],
                self.buffer_align_bytes,
            ]

        if self.dm_opts.get("spf", "0") == "1":
            SharedStorage = SharedStorageSPF
        self.shared_storage = SharedStorage
        smem_capacity = sm100_smem_capacity(self.arch.major * 10 + self.arch.minor)
        assert SharedStorage.size_in_bytes() <= smem_capacity, (
            f"SMEM {SharedStorage.size_in_bytes()} > {smem_capacity}: "
            f"overlap_sO_sQ={self.overlap_sO_sQ}, kv_stage={self.kv_stage}"
        )

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors, mPageTable)

        head_divmod = None
        if cutlass.const_expr(self.pack_gqa):
            head_divmod = FastDivmodDivisorV2(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        # slice=1 implements the dense iteration and no learnable sink: refuse rather than mis-run
        assert not (self.dm_slice and (blocksparse_tensors is not None or learnable_sink is not None)), \
            "DM_FWD_OPTS slice=1 does not support block sparsity or a learnable sink"
        if cutlass.const_expr(self.use_block_sparsity and self.use_s_ping_pong):
            raise NotImplementedError("Block sparsity + S ping-pong not supported on SM100")
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError("Block sparsity + paged KV not supported on SM100")
        if cutlass.const_expr(self.use_block_sparsity and self.is_varlen_q):
            assert const_expr(blocksparse_tensors.cu_total_m_blocks is not None), (
                "blocksparse_tensors.cu_total_m_blocks must be provided for varlen blocksparsity"
            )
        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mV_gmem,
            tensormap_workspace if const_expr(self.ragged_tma_V) else None,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            descale_tensors,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_O,
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
            head_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    def _generate_attention_mask_cls(self, window_size_left, window_size_right):
        return partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mV_gmem: Optional[cute.Tensor],  # the gmem V tensor behind tma_atom_V (ragged V only)
        tensormap_workspace: Optional[cute.Tensor],  # (num_ctas, 16) Int64 (ragged V only)
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
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
        head_divmod=None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """
        dm_utils.tcf_begin(int(self.dm_opts.get("tcf", "1")))   # tcgen05 thread-sync fences (dm_utils.TCF)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

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
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        load_warps = ThreadCooperativeGroup(len(self.load_warp_ids))
        load_threads = ThreadCooperativeGroup(len(self.load_warp_ids) * cute.arch.WARP_SIZE)
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        # softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        # correction_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_warps = ThreadCooperativeGroup(len(self.epilogue_warp_ids))
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        # For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids) * self.cta_group_size
        )
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_load_Q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["Q"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_Q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        if const_expr(self.use_tma_KV):
            pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_kv = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.s_p_o_stage,
            producer_group=mma_warp,
            # slice: one elected arrival per slice / correction warp (20) instead of 640 thread arrivals
            consumer_group=(ThreadCooperativeGroup(len(self.softmax0_warp_ids + self.correction_warp_ids))
                            if const_expr(self.dm_slice and self.dm_opts.get("warr", "0") == "1")
                            else softmax_correction_threads_cluster),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.s_p_o_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        # Ping-pong commits it after every PV (slot pv_count & 1); otherwise once per tile per stage.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.s_p_o_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s0_s1_sequence = None
        if const_expr(self.s0_s1_barrier and self.q_stage > 1):
            # This is not a typical producer-consumer pipeline. We will directly use
            # pipeline_s0_s1_sequence.sync_object_full and will not use
            # pipeline_s0_s1_sequence.sync_object_empty.
            pipeline_s0_s1_sequence = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_s0_s1_sequence.data_ptr(),
                num_stages=2,
                producer_group=softmax_threads,
                consumer_group=softmax_threads,
                defer_sync=True,
            )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.s_p_o_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2
        )
        pipeline_o_epi = None
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_O_epi.data_ptr(),
                num_stages=self.q_stage,
                producer_group=correction_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )

        pipeline_load_epi = None
        if const_expr(self.overlap_sO_sQ and self.is_persistent):
            # when overlapping sO and sQ with a persistent kernel, we need this
            # additional pipeline to ensure sO from the previous work tile is
            # free for use by sQ in the current one.
            epi_warps_for_release = (
                ThreadCooperativeGroup(len(self.correction_warp_ids))
                if self.use_correction_warps_for_epi
                else epilogue_warps
            )
            pipeline_load_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_load_epi.data_ptr(),
                num_stages=1,
                producer_group=epi_warps_for_release,
                consumer_group=load_warps,
                defer_sync=True,
            )


        # mark the tcgen05 hand-off pipelines / barriers (dm_utils.tcf_mark)
        dm_utils.tcf_mark("ua", pipeline_s_p_o, pipeline_o_acc)
        dm_utils.tcf_mark("au", pipeline_p_lastsplit)
        dm_utils.tcf_mark("bar", sm_stats_barrier, tmem_alloc_barrier)
        # Cluster arrive after barrier init
        if const_expr(self.dm_sspv):
            self.dm_sfree = Int32(storage.mbar_dm_sfree.data_ptr().toint())
            self.dm_pvdone = Int32(storage.mbar_dm_pvdone.data_ptr().toint())
            self.dm_cread = Int32(storage.mbar_dm_cread.data_ptr().toint())
            if cute.arch.thread_idx()[0] == 0:
                dm_utils.mbar_init(self.dm_sfree, 4 * cute.arch.WARP_SIZE)
                dm_utils.mbar_init(self.dm_pvdone, 1)
                dm_utils.mbar_init(self.dm_pvdone + 8, 1)
                if const_expr(self.dm_dbs):
                    _ = [dm_utils.mbar_init(self.dm_cread + 8 * _b, cute.arch.WARP_SIZE) for _b in range(32)]   # per warp: 32 arrivals
                else:
                    dm_utils.mbar_init(self.dm_cread, 4 * cute.arch.WARP_SIZE)
                    dm_utils.mbar_init(self.dm_cread + 8, 4 * cute.arch.WARP_SIZE)
                cute.arch.mbarrier_init_fence()
        if const_expr(self.dm_slice):
            self.dm_slbar = Int32(storage.mbar_dm_slice.data_ptr().toint())
            self.dm_slbuf = Int32(storage.sDmSlice.data_ptr().toint())
            if cute.arch.thread_idx()[0] == 0:
                for _s in range(2):
                    dm_utils.mbar_init(self.dm_slbar + 8 * _s, 16)        # kbfull[s]: one arrival per slice warp
                    dm_utils.mbar_init(self.dm_slbar + 16 + 8 * _s, 16)   # bsfull[s]
                    dm_utils.mbar_init(self.dm_slbar + 32 + 8 * _s, 4)    # bsempty[s]: one per correction warp
                cute.arch.mbarrier_init_fence()
        if const_expr(self.dm_ordw):
            self.dm_ord = Int32(storage.mbar_dm_ord.data_ptr().toint())
            if cute.arch.thread_idx()[0] == 0:
                dm_utils.mbar_init(self.dm_ord, 1)
                dm_utils.mbar_init(self.dm_ord + 8, 1)
                cute.arch.mbarrier_init_fence()
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.s_p_o_stage * self.m_block_size * 2 * (2 if self.dm_dbs else 1)))
        if const_expr(self.dm_sspv):
            # rebuild the SS-PV MMA atom / sP layout inside the kernel region (static metadata)
            self.dm_tiled_mma_pv_ss = sm100_utils_basic.make_trivial_tiled_mma(
                self.v_dtype, self.v_dtype, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
                self.pv_acc_dtype, tcgen05.CtaGroup.ONE, self.mma_tiler_pv[:2])
            dm_sP_layout_k = sm100_utils_basic.make_smem_layout_a(
                self.dm_tiled_mma_pv_ss, self.mma_tiler_pv, self.q_dtype, self.q_stage)
            self.dm_sP = storage.sP.get_tensor(dm_sP_layout_k.outer, swizzle=dm_sP_layout_k.inner)

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (self.tmem_p_offset[1] - self.tmem_p_offset[0]) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(tOrP.layout, cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)))
        )

        block_info = BlockInfo(
            # A 2CTA pair shares one cluster-wide K/V range.
            self.cta_tiler[0] * self.cta_group_size,
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            num_splits=num_splits,
            pack_split_idx=num_splits_dynamic_ptr is not None,
            num_n_blocks_per_split=self.num_n_blocks_per_split,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks if blocksparse_tensors is not None else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
        )
        AttentionMaskCls = self._generate_attention_mask_cls(
            window_size_left, window_size_right
        )
        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        sched_ctx = None
        if const_expr(self.use_clc_scheduler or self.dynamic_persistent):
            sched_response_ptr = storage.sched_response.data_ptr()
            sched_mbar_ptr = storage.sched_mbar_ptr.data_ptr()
            sched_producer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread
            )
            num_sched_consumer_warps_per_cta = self.threads_per_cta // cute.arch.WARP_SIZE
            # NB on CTA0 warp15 == scheduler on CTA1 == empty but still both consume
            num_sched_consumer_warps = num_sched_consumer_warps_per_cta * self.cta_group_size
            sched_consumer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread,
                cute.arch.WARP_SIZE * num_sched_consumer_warps,
            )
            if const_expr(self.use_clc_scheduler):
                _block_idx = cute.arch.block_idx()
                sched_ctx = SchedulerState.create_clc(
                    hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                        self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                        _block_idx,
                        cute.arch.grid_dim(),
                        sched_response_ptr,
                    ),
                    pipeline=cutlass_pipeline.PipelineClcFetchAsync.create(
                        barrier_storage=sched_mbar_ptr,
                        num_stages=self.sched_stages,
                        producer_group=sched_producer_group,
                        consumer_group=sched_consumer_group,
                        tx_count=16,
                        cta_layout_vmnk=cta_layout_vmnk,
                    ),
                    consumer_state=cutlass_pipeline.make_pipeline_state(
                        cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
                    ),
                    producer_state=cutlass_pipeline.make_pipeline_state(
                        cutlass_pipeline.PipelineUserType.Producer, self.sched_stages
                    ),
                )
            else:
                assert tile_count_semaphore is not None
                sched_ctx = SchedulerState.create_dynamic_persistent(
                    work_info=storage.sched_response.get_tensor((4,)),
                    pipeline=cutlass_pipeline.PipelineAsync.create(
                        barrier_storage=sched_mbar_ptr,
                        num_stages=self.sched_stages,
                        producer_group=sched_producer_group,
                        consumer_group=sched_consumer_group,
                    ),
                    consumer_state=cutlass_pipeline.make_pipeline_state(
                        cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
                    ),
                    producer_state=cutlass_pipeline.make_pipeline_state(
                        cutlass_pipeline.PipelineUserType.Producer, self.sched_stages
                    ),
                )
        if const_expr(self.use_clc_scheduler or self.dynamic_persistent):
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, ctx=sched_ctx)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY / SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.dynamic_persistent):
            if warp_idx == self.scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                # CLC: only the leader produces; late claims belong to the load warp.
                if is_leader_cta and not self.clc_late_claim:
                    self.scheduler_warp(tile_scheduler)
                else:
                    self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.scheduler_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                mV_gmem,
                tensormap_workspace,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                gmem_tiled_copy_Q,
                pipeline_q,
                pipeline_kv,
                pipeline_load_epi,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    pipeline_o_epi,
                    pipeline_load_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    mma_tile_coord_v,
                    blocksparse_tensors=blocksparse_tensors,
                    tile_scheduler=tile_scheduler,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.dm_slice):
            if warp_idx < 16:
                if const_expr(self.num_regs_softmax > 80):
                    cute.arch.setmaxregister_increase(self.num_regs_softmax)
                elif const_expr(self.num_regs_softmax < 80):
                    cute.arch.setmaxregister_decrease(self.num_regs_softmax)
                tmem_alloc_barrier.arrive_and_wait_unaligned()
                tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
                self.slice_softmax_loop(
                    thr_mma_qk, tStS, descale_tensors, pipeline_s_p_o, block_info, num_splits, SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                )
                tmem_alloc_barrier.arrive_unaligned()
        elif (
            (const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]) or
            (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1])
        ):
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                descale_tensors=descale_tensors,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                learnable_sink=learnable_sink,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                blocksparse_tensors=blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if const_expr(self.q_stage == 1) or warp_idx < self.softmax1_warp_ids[0] else 1)
                softmax_loop(stage=stage, tStS=tStS)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    softmax_loop(stage=0, tStS=tStS)
                if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                    softmax_loop(stage=1, tStS=tStS)

            tmem_alloc_barrier.arrive_unaligned()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            # setmaxnreg.dec cannot exceed the launch register count.
            if const_expr(self.dm_slice):
                if const_expr(self.num_regs_correction > 65536 // self.threads_per_cta // 8 * 8):
                    cute.arch.setmaxregister_increase(self.num_regs_correction)
                elif const_expr(self.num_regs_correction < 65536 // self.threads_per_cta // 8 * 8):
                    cute.arch.setmaxregister_decrease(self.num_regs_correction)
            elif const_expr(self.num_regs_correction > SM100_LAUNCH_REGISTERS):
                cute.arch.setmaxregister_increase(self.num_regs_correction)
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if const_expr(self.dm_slice):
                self.slice_correction_loop(
                    thr_mma_qk, thr_mma_pv, tStS, tOtO, mO, mLSE, sO, pipeline_s_p_o, pipeline_o_acc, pipeline_o_epi,
                    pipeline_load_epi, descale_tensors, gmem_tiled_copy_O, block_info, num_splits, SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                )
                tmem_alloc_barrier.arrive_unaligned()
            else:
                self.correction_loop(
                    thr_mma_qk,
                    thr_mma_pv,
                    tStS,
                    tOtO,
                    sScale,
                    mO,
                    mLSE,
                    sO,
                    pipeline_s_p_o,
                    pipeline_o_acc,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    pipeline_o_epi,
                    pipeline_load_epi,
                    learnable_sink,
                    descale_tensors,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    softmax_scale_log2,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    blocksparse_tensors,
                    tile_scheduler=tile_scheduler,
                )
                tmem_alloc_barrier.arrive_unaligned()

        dm_utils.tcf_end()
        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mV_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_load_epi: Optional[pipeline.PipelineAsync],
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler: TileSchedulerProtocol,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        load_epi_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        issue_kv_for_this_warp = (
            const_expr(not self.use_tma_KV or len(self.load_warp_ids) == 1) or
            warp_idx == self.load_warp_ids[0]
        )
        issue_q_for_this_warp = (
            const_expr(not self.use_tma_Q or len(self.load_warp_ids) == 1) or
            warp_idx == self.load_warp_ids[0]
        )
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        tensormap_manager = None
        tensormap_V_gmem = None
        v_desc_ptr = None
        if const_expr(self.ragged_tma_V):
            tensormap_manager, tensormap_V_gmem, v_desc_ptr = cta_tensormap_slot(
                tensormap_workspace
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_V, tensormap_V_gmem, self.load_warp_ids[0]
            )
            tensormap_manager.fence_tensormap_initialization()
        # The V extent depends on the batch only: rewrite the descriptor when it changes and
        # acquire it right before the first V load so the K/Q issue overlaps the fence.
        v_desc_batch_idx = Int32(-1)
        v_desc_acquire = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            if const_expr(self.ragged_tma_V) and batch_idx != v_desc_batch_idx:
                # Shrink the rows extent of V to this sequence's end (packed: absolute row;
                # dense: per-batch row). The TMA unit then zero-fills every row at or past
                # the end and never touches memory beyond it. Extents must stay >= 1.
                v_desc_batch_idx = batch_idx
                v_desc_acquire = Int32(1)
                v_rows_end = (
                    seqlen.offset_k + seqlen.seqlen_k
                    if const_expr(seqlen.has_cu_seqlens_k)
                    else seqlen.seqlen_k
                )
                tensormap_manager.update_tensormap(
                    (bound_rows(mV_gmem, v_rows_end),), (tma_atom_V,), (tensormap_V_gmem,),
                    self.load_warp_ids[0], (None,),
                )

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur = mK[None, None, head_idx_kv, batch_idx]
                else:
                    mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mV_cur = mV[None, None, head_idx_kv, batch_idx]
                else:
                    mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV) if const_expr(gV is not None) else None
            if const_expr(self.use_tma_Q):
                tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
                gQ = cute.local_tile(mQ_cur, tiler_gQ, (work_m, 0))  # (128 * 2, 128)
                gQ = layout_utils.select(
                    cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
                )  # (128, 128, 2)
                tSgQ = thr_mma_qk.partition_A(gQ)
                load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
                )
                load_Q = partial(self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase)
            else:
                assert gmem_tiled_copy_Q is not None
                load_Q = partial(
                    self.load_Q_non_tma,
                    mQ_cur,
                    sQ,
                    gmem_tiled_copy_Q,
                    pipeline_q,
                    tidx,
                    seqlen.seqlen_q,
                    work_m,
                    phase=q_producer_phase,
                )

            if const_expr(self.use_tma_KV):
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
                paged_kv_manager = None
            else:
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmodDivisorV2(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
                tma_desc_ptr=v_desc_ptr,
            )

            if const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    work_m,
                    split_idx=split_idx,
                    num_splits=num_splits,
                )
                if const_expr(self.is_split_kv and block_info.pack_split_idx):
                    split_idx = split_idx & 0xFFFF
                if self.process_work_tile(block_info, seqlen, n_block_min, n_block_max):
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                    page_idx = (
                        resolve_paged_token(
                            mPageTable[batch_idx, None], n_block_first * self.n_block_size, self.n_block_size
                        )[0]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block_first)
                    if issue_kv_for_this_warp:
                        load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # K0
                    # load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx, extra_tx_count=self.tma_copy_bytes["Q"])  # K0
                    if issue_q_for_this_warp:
                        load_Q(block=0, stage=0)
                    if issue_kv_for_this_warp:
                        kv_producer_state.advance()
                    if const_expr(self.q_stage == 2) and issue_q_for_this_warp:
                        load_Q(block=1, stage=1)
                    if const_expr(self.ragged_tma_V) and v_desc_acquire == 1:
                        acquire_tensormap(tensormap_manager, tensormap_V_gmem)
                        v_desc_acquire = Int32(0)
                    q_producer_phase ^= 1
                    if const_expr(not self.use_s_ping_pong):
                        if issue_kv_for_this_warp:
                            load_V(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # V0
                            kv_producer_state.advance()
                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        page_idx = (
                            resolve_paged_token(
                                mPageTable[batch_idx, None], n_block * self.n_block_size, self.n_block_size
                            )[0]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                    # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("n_block = {}, page_idx = {}", n_block, page_idx)
                        if issue_kv_for_this_warp:
                            load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Ki
                            kv_producer_state.advance()
                            if const_expr(self.use_s_ping_pong):
                                # V lags K by one block to match the MMA order QK(i+1) -> PV(i)
                                page_idx_v = (
                                    resolve_paged_token(
                                        mPageTable[batch_idx, None], (n_block + 1) * self.n_block_size, self.n_block_size
                                    )[0]
                                    if const_expr(mPageTable is not None and self.use_tma_KV)
                                    else None
                                )
                                load_V(block=n_block + 1, producer_state=kv_producer_state, page_idx=page_idx_v)  # V(i-1)
                            else:
                                load_V(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Vi
                            kv_producer_state.advance()
                    if const_expr(self.use_s_ping_pong):
                        if issue_kv_for_this_warp:
                            page_idx_v = (
                                resolve_paged_token(
                                    mPageTable[batch_idx, None], n_block_min * self.n_block_size, self.n_block_size
                                )[0]
                                if const_expr(mPageTable is not None and self.use_tma_KV)
                                else None
                            )
                            load_V(block=n_block_min, producer_state=kv_producer_state, page_idx=page_idx_v)  # V_last
                            kv_producer_state.advance()

            else:
                # Match the dense path (get_n_block_min_max): the scheduler packs the
                # per-batch dynamic num_splits into the top 16 bits of split_idx.
                if const_expr(self.ragged_tma_V) and v_desc_acquire == 1:
                    acquire_tensormap(tensormap_manager, tensormap_V_gmem)
                    v_desc_acquire = Int32(0)
                num_splits_dyn = num_splits
                if const_expr(self.is_split_kv and block_info.pack_split_idx):
                    num_splits_dyn = split_idx >> 16
                    split_idx = split_idx & 0xFFFF
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    work_m,
                    seqlen,
                    split_idx,
                    num_splits_dyn,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                    self.kv_subtile_factor,
                )

            if const_expr(self.clc_late_claim):
                tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            if const_expr(pipeline_load_epi is not None):
                pipeline_load_epi.consumer_wait(load_epi_consumer_state)
                with cute.arch.elect_one():
                    pipeline_load_epi.consumer_release(load_epi_consumer_state)
                load_epi_consumer_state.advance()

        if const_expr(self.clc_late_claim):
            tile_scheduler.producer_tail()
        if issue_kv_for_this_warp:
            pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail for the TMA-Q producer warp.
        if issue_q_for_this_warp:
            pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler=None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0],)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        qk_mma_idesc, pv_mma_idesc = sm100_desc.mma_op_to_idesc(qk_mma_op), sm100_desc.mma_op_to_idesc(pv_mma_op)
        qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        v_smem_base = sm100_desc.smem_desc_base_from_tensor(sV, sm100_desc.Major.MN)
        q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]

        sm100_utils.declare_ptx_smem_desc(q_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="fa_fwd_q_smem_desc")
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.q_stage == 1):
            sQ_stage_stride = 0
        gemm_Si = [
            partial(
                # sm100_utils.gemm_ptx_precomputed,
                # self.tmem_s_offset[stage],
                # smem_desc_start_a=q_smem_start[stage],
                # idesc=qk_mma_idesc,
                # smem_desc_base_a=q_smem_base,
                # smem_desc_base_b=k_smem_base,
                # tCrA_layout=tSrQ[None, None, None, 0].layout,
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                # idesc=qk_mma_idesc,
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="fa_fwd_q_smem_desc",
                idesc_var_name="fa_fwd_qk_mma_idesc",
                kind=qk_mma_kind,
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.s_p_o_stage)
        ]
        # gemm_Si = [
        #     partial(
        #         sm100_utils.gemm,
        #         tiled_mma_qk,
        #         tStS[None, None, None, stage],
        #         tCrA=tSrQ[None, None, None, stage],
        #         zero_init=True,
        #     )
        #     for stage in range(self.q_stage)
        # ]
        gemm_Pi = [
            partial(
                dm_utils.gemm_ptx_bs_ts,
                pv_mma_op,
                self.tmem_o_offset[stage % self.q_stage],
                tOrP[None, None, None, stage],
                sfa_tmem_addr=(self.dm_sf_abs[stage] if const_expr(self.dm_sspv) else self.tmem_s_offset[stage] + (self.dm_sl_sfcol if self.dm_slice else self.dm_sf_col)),
                sfb_tmem_addr=(self.dm_sf_abs[stage] if const_expr(self.dm_sspv) else self.tmem_s_offset[stage] + (self.dm_sl_sfcol if self.dm_slice else self.dm_sf_col)) + 4,
                split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
                # slice: P codes of k-step (key block) k at S_s + 32 k, SF at S_s + 16
                tA_addr=(Int32(self.tmem_s_offset[stage]) if const_expr(self.dm_slice) else None),
                offs_a=((0, 32, 64, 96) if const_expr(self.dm_slice) else None),
                # smem_desc_start_a=tOrP[None, None, None, stage].iterator.toint(),
                # smem_desc_start_a=self.tmem_p_offset[stage],
                # idesc=pv_mma_idesc,
                # smem_desc_base_a=None,
                # smem_desc_base_b=v_smem_base,
                # tCrA_layout=tOrP[None, None, None, 0].layout,
                # tCrB_layout=tOrV[None, None, None, 0].layout
                cta_group=self.cta_group_size,
            )
            for stage in range(self.s_p_o_stage)
        ]
        # gemm_Pi = [
        #     partial(
        #         sm100_utils.gemm, tOtO[None, None, None, stage], tCrA=tOrP[None, None, None, stage]
        #     )
        #     for stage in range(self.q_stage)
        # ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)
        if const_expr(self.use_s_ping_pong):
            mma_pv_count = Int32(0)
        dm_pvn = Int32(0)   # ordinv ordw: PV rounds issued (per stage) by this CTA so far, across work tiles
        if const_expr(self.dm_sspv):
            # fwdrw ss_pv: SS block-scaled PV from sP; dummy pvdone completion #0 per stage (no PV precedes the 1st P store)
            dm_tm_ss = self.dm_tiled_mma_pv_ss.get_slice(0)
            dm_tCrP = dm_tm_ss.make_fragment_A(self.dm_sP)
            dm_sfmap = tuple((0, k) for k in range(4))
            dm_q = Int32(0)
            with cute.arch.elect_one():
                dm_utils.tc_commit(self.dm_pvdone)
                dm_utils.tc_commit(self.dm_pvdone + 8)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):
                # See the load warp: unpack dynamic num_splits packed by the scheduler.
                num_splits_dyn = num_splits
                if const_expr(self.is_split_kv and block_info.pack_split_idx):
                    num_splits_dyn = split_idx >> 16
                    split_idx = split_idx & 0xFFFF
                block_iter_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    work_m,
                    split_idx,
                    num_splits_dyn,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                    seqlen_info=seqlen,
                    kv_subtile_factor=self.kv_subtile_factor,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    work_m,
                    split_idx=split_idx,
                    num_splits=num_splits,
                )
                block_iter_count = n_block_max - n_block_min
                process_tile = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)

            if process_tile and is_leader_cta:
                if const_expr(self.use_s_ping_pong):
                    # Software pipeline: QK(0); [QK(i+1), PV(i)] for i in 0..n-2; PV(n-1).
                    # Block i uses S/P slot i & 1; the slot's barrier phase flips on each reuse.
                    pipeline_q.consumer_wait_w_index_phase(0, mma_q_consumer_phase)
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    sK_cur = self.kv_smem_stage(sK, mma_kv_consumer_state)
                    self.mma_ping_pong_qk(gemm_Si, pipeline_s_p_o, sK_cur, mma_pv_count)
                    mma_q_consumer_phase ^= 1
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                    O_should_accumulate = False
                    for i in cutlass.range(block_iter_count - 1, unroll=1):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        sK_cur = self.kv_smem_stage(sK, mma_kv_consumer_state)
                        self.mma_ping_pong_qk(gemm_Si, pipeline_s_p_o, sK_cur, mma_pv_count + 1)
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        tOrVi = tOrV[None, None, None, mma_kv_consumer_state.index]
                        sV_cur = self.kv_smem_stage(sV, mma_kv_consumer_state)
                        self.mma_ping_pong_pv(
                            gemm_Pi, pipeline_s_p_o, pipeline_p_lastsplit, pipeline_o_acc,
                            tOrVi, sV_cur, O_should_accumulate, mma_pv_count,
                        )
                        mma_pv_count += 1
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                        O_should_accumulate = True
                    for stage in cutlass.range(self.q_stage):
                        pipeline_q.consumer_release_w_index(stage)
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    tOrVi = tOrV[None, None, None, mma_kv_consumer_state.index]
                    sV_cur = self.kv_smem_stage(sV, mma_kv_consumer_state)
                    self.mma_ping_pong_pv(
                        gemm_Pi, pipeline_s_p_o, pipeline_p_lastsplit, pipeline_o_acc,
                        tOrVi, sV_cur, O_should_accumulate, mma_pv_count,
                    )
                    mma_pv_count += 1
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                elif const_expr(self.dm_sspv):
                    # QK0(0), [sfree] QK1(0); per block i: [sfree] QK0(i+1), [P0] PV0(i), [sfree] QK1(i+1), [P1] PV1(i)
                    for stage in cutlass.range_constexpr(self.q_stage):
                        pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                        if const_expr(stage == 0):
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        if dm_q > 0:
                            dm_utils.mbar_wait_parity(self.dm_sfree, (dm_q - 1) & 1)
                        if const_expr(self.dm_opts.get("tcf", "1") == "1"):
                            dm_utils.tc_fence_after()
                        cute.arch.fence_view_async_tmem_load()
                        gemm_Si[stage](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        dm_q += 1
                    mma_q_consumer_phase ^= 1
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    block_loop_count = block_iter_count - 1
                    O_should_accumulate = False
                    for i in cutlass.range(block_loop_count, unroll=1):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        tOrVi = tOrV[None, None, None, Vi_index]
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        mma_kv_consumer_state.advance()
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        for stage in cutlass.range_constexpr(self.q_stage):
                            dm_utils.mbar_wait_parity(self.dm_sfree, (dm_q - 1) & 1)
                            if const_expr(self.dm_opts.get("tcf", "1") == "1"):
                                dm_utils.tc_fence_after()
                            cute.arch.fence_view_async_tmem_load()
                            gemm_Si[stage](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
                            pipeline_s_p_o.producer_commit_w_index(stage)
                            dm_q += 1
                            pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                            if const_expr(self.dm_opts.get("tcf", "1") == "1"):
                                dm_utils.tc_fence_after()   # SF/O tcgen05.st of softmax/correction before this PV
                            dm_utils.gemm_ptx_bs(
                                pv_mma_op, self.tmem_o_offset[stage], dm_tCrP[None, None, None, stage], tOrVi,
                                self.dm_sP[None, None, None, stage], sV_cur,
                                self.dm_sf_abs[stage], self.dm_sf_abs[stage] + 4, dm_sfmap, dm_sfmap,
                                zero_init=not O_should_accumulate, cta_group=self.cta_group_size)
                            with cute.arch.elect_one():
                                dm_utils.tc_commit(self.dm_pvdone + 8 * stage)
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                        P_full_O_rescaled_phase ^= 1
                        O_should_accumulate = True
                    for stage in cutlass.range(self.q_stage):
                        pipeline_q.consumer_release_w_index(stage)
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    for stage in cutlass.range_constexpr(self.q_stage):
                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                        if const_expr(self.dm_opts.get("tcf", "1") == "1"):
                            dm_utils.tc_fence_after()
                        dm_utils.gemm_ptx_bs(
                            pv_mma_op, self.tmem_o_offset[stage], dm_tCrP[None, None, None, stage], tOrVi,
                            self.dm_sP[None, None, None, stage], sV_cur,
                            self.dm_sf_abs[stage], self.dm_sf_abs[stage] + 4, dm_sfmap, dm_sfmap,
                            zero_init=not O_should_accumulate, cta_group=self.cta_group_size)
                        with cute.arch.elect_one():
                            dm_utils.tc_commit(self.dm_pvdone + 8 * stage)
                        pipeline_o_acc.producer_commit_w_index(stage)
                    P_full_O_rescaled_phase ^= 1
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                else:
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                        # 1. wait for Q0 / Q1
                        pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                        # 2. wait for K0
                        if const_expr(stage == 0):
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        tSrKi = tSrK[None, None, None, Ki_index]
                        # We don't need to acquire empty S0 / S1.
                        # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                        # are empty. For subsequent iterations, the wait happened at the end
                        # of the while loop.
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrKi, zero_init=True)
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        if const_expr(self.dm_ordw):
                            # ordinv: the previous work tile's last PV_s (reads SF in S_s) must be complete before QK_s
                            if dm_pvn > 0:
                                dm_utils.mbar_wait_parity(self.dm_ord + 8 * stage, (dm_pvn - 1) & 1)
                                dm_utils.tc_fence_after()
                        if const_expr(self.dm_ordpc == 1):
                            if dm_pvn > 0:   # DEBUG planted control at the work-tile boundary (previous tile's last PV_s)
                                dm_pc_a = ((cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32) << 16
                                dm_pc_a = dm_pc_a + Int32(tStS[(None, None), 0, 0, 0].iterator.toint()) + Int32(
                                    self.tmem_s_offset[stage] + (self.dm_sl_sfcol if self.dm_slice else self.dm_sf_col))
                                dm_pc_w = Int32(-2139062144)   # 0x80808080
                                dm_utils.tmem_st_x4(dm_pc_a, dm_pc_w, dm_pc_w, dm_pc_w, dm_pc_w)
                                cute.arch.fence_view_async_tmem_store()
                        # gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                        )
                        # gemm_Si[stage](tCrB=tSrKi)
                        # 4. release S0 / S1
                        pipeline_s_p_o.producer_commit_w_index(stage)
                    mma_q_consumer_phase ^= 1
                    # 5. release K0
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    # End of GEMM (Q1 * K0 -> S1)
                    # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                    # so we need to release them after the seqlen_kv loop

                    # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                    block_loop_count = block_iter_count - 1
                    O_should_accumulate = False
                    for i in cutlass.range(block_loop_count, unroll=1):
                        # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                        # 1. wait for V0
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        tOrVi = tOrV[None, None, None, Vi_index]
                        for stage in cutlass.range_constexpr(self.q_stage):
                            # 2. acquire corrected O0/O1_partial and P0 / P1
                            # For the first iteration in this work tile, waiting for O0/O1_partial
                            # means that the correction warps has finished reading tO during
                            # the last iteration of the previous work tile.
                            pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                            if const_expr(self.dm_slice):
                                dm_utils.tc_fence_after()   # slice: P / SF / O tcgen05.st of other warps before this PV
                            # 3. gemm
                            # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                            # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                            sV_cur = sV[None, None, None, Vi_index]
                            if const_expr(self.uneven_kv_smem):
                                sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                            gemm_Pi[stage](
                                tCrB=tOrVi,
                                sB=sV_cur,
                                # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                                zero_init=not O_should_accumulate,
                                mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                                mbar_phase=P_full_O_rescaled_phase,
                            )
                            if const_expr(self.dm_ordw):
                                with cute.arch.elect_one():
                                    dm_utils.tc_commit(self.dm_ord + 8 * stage)   # ordinv: PV_s(i) completion #dm_pvn
                            # Don't need to signal O_full to the correction warps since the
                            # correction warps wait for the softmax warps anyway. By the time the softmax
                            # warps finished, S_i for the next iteration must have been done, so O_i-1
                            # must have been done as well.
                            # pipeline_o_acc.producer_commit_w_index(stage)
                            # 4. release V(i-1)
                            if const_expr(stage == self.q_stage - 1):
                                pipeline_kv.consumer_release(mma_kv_release_state)
                                mma_kv_release_state.advance()
                            # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                            # GEMM_QK0i (Q0 * Ki -> S0)
                            # 1. wait for Ki
                            if const_expr(stage == 0):
                                mma_kv_consumer_state.advance()
                                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                            Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                            # 2. gemm
                            # Don't need to wait for the softmax warp to have finished reading the previous
                            # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                            # has been read and Pi has been written.
                            # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrK[None, None, None, Ki_index], zero_init=True)
                            sK_cur = sK[None, None, None, Ki_index]
                            if const_expr(self.uneven_kv_smem):
                                sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                            if const_expr(self.dm_ordw):
                                # ordinv C0: PV_s(i) read SFA/SFB from S_s; wait its completion before QK_s(i+1) overwrites S_s.
                                # Phase-safe: PV_s(i-1) is complete (P_full(i) needs S_s(i) loaded, whose commit tracks it).
                                dm_utils.mbar_wait_parity(self.dm_ord + 8 * stage, dm_pvn & 1)
                                dm_utils.tc_fence_after()
                            if const_expr(self.dm_ordpc == 1):
                                # DEBUG planted control: poison SFA of S_s (lanes of this warp's quadrant) before QK_s(i+1)
                                dm_pc_a = ((cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32) << 16
                                dm_pc_a = dm_pc_a + Int32(tStS[(None, None), 0, 0, 0].iterator.toint()) + Int32(
                                    self.tmem_s_offset[stage] + (self.dm_sl_sfcol if self.dm_slice else self.dm_sf_col))
                                dm_pc_w = Int32(-2139062144)   # 0x80808080
                                dm_utils.tmem_st_x4(dm_pc_a, dm_pc_w, dm_pc_w, dm_pc_w, dm_pc_w)
                                cute.arch.fence_view_async_tmem_store()
                            # gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                            gemm_Si[stage](
                                smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                            )
                            # gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index])
                            # 3. release S0 / S1
                            pipeline_s_p_o.producer_commit_w_index(stage)
                            # End of GEMM_QK0i (Q0 * Ki -> S0)
                        # 4. release Ki
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                        P_full_O_rescaled_phase ^= 1
                        O_should_accumulate = True
                        dm_pvn += 1
                    # End of seqlen_kv loop

                    # release Q0 & Q1
                    for stage in cutlass.range(self.q_stage):
                        pipeline_q.consumer_release_w_index(stage)

                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected Oi_partial and Pi
                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                        if const_expr(self.dm_slice):
                            dm_utils.tc_fence_after()
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                            zero_init=not O_should_accumulate,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                        if const_expr(self.dm_ordw):
                            with cute.arch.elect_one():
                                dm_utils.tc_commit(self.dm_ord + 8 * stage)
                        # 4. release accumulated O0_partial
                        # We do need O_full here since for the last tile, by the time the softmax warp
                        # has signaled to the correction warps, the softmax warp has just finished
                        # computing the row sum of the current tile. It does not guarantee that the 1st
                        # tile of the next work tile has been computed yet.
                        pipeline_o_acc.producer_commit_w_index(stage)
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                    P_full_O_rescaled_phase ^= 1
                    dm_pvn += 1
                    # 5. release Vi_end
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # pipeline_s_p_o.producer_acquire_w_index_phase(self.q_stage - 1, P_full_O_rescaled_phase)
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    @cute.jit
    def kv_smem_stage(self, sX: cute.Tensor, kv_state: pipeline.PipelineState):
        """K or V smem for the stage kv_state points at."""
        sX_cur = sX[None, None, None, kv_state.index]
        if const_expr(self.uneven_kv_smem):
            sX_cur = self.offset_kv_smem(sX_cur, kv_state.index, kv_state.phase)
        return sX_cur

    @cute.jit
    def mma_ping_pong_qk(
        self,
        gemm_Si: list,
        pipeline_s_p_o: pipeline.PipelineAsync,
        sK_cur: cute.Tensor,
        block_count: Int32,
    ):
        """Issue QK for the block with global count block_count into S slot block_count & 1."""
        if (block_count & 1) == 0:
            gemm_Si[0](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
            pipeline_s_p_o.producer_commit_w_index(0)
        else:
            gemm_Si[1](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
            pipeline_s_p_o.producer_commit_w_index(1)

    @cute.jit
    def mma_ping_pong_pv(
        self,
        gemm_Pi: list,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        tOrVi: cute.Tensor,
        sV_cur: cute.Tensor,
        O_should_accumulate: Boolean,
        pv_count: Int32,
    ):
        """Issue PV for the block with global count pv_count from P slot pv_count & 1."""
        pv_phase = (pv_count >> 1) & 1
        if (pv_count & 1) == 0:
            pipeline_s_p_o.producer_acquire_w_index_phase(0, pv_phase)
            gemm_Pi[0](
                tCrB=tOrVi,
                sB=sV_cur,
                zero_init=not O_should_accumulate,
                mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(0) if self.split_P_arrive > 0 else None,
                mbar_phase=pv_phase,
            )
        else:
            pipeline_s_p_o.producer_acquire_w_index_phase(1, pv_phase)
            gemm_Pi[1](
                tCrB=tOrVi,
                sB=sV_cur,
                zero_init=not O_should_accumulate,
                mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(1) if self.split_P_arrive > 0 else None,
                mbar_phase=pv_phase,
            )
        # QK(i+1) is issued before PV(i), so S(i+1) being ready no longer implies O(i) is done:
        # the correction warps wait on O_full before every O read.
        pipeline_o_acc.producer_commit_w_index(pv_count & 1)

    def _q_block(self, work_m, stage, cta_rank=0):
        return work_q_block(
            work_m, stage, cta_rank, q_stage=self.q_stage, cta_group_size=self.cta_group_size
        )

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def _kv_head_idx(self, head_idx: Int32) -> Int32:
        """Map query-head tile index -> KV-head index (FA3 descale semantics)."""
        if cutlass.const_expr(self.pack_gqa):
            return head_idx
        return head_idx // self.qhead_per_kvhead

    @cute.jit
    def _load_effective_descales(
        self,
        descale_tensors,
        batch_idx: Int32,
        kv_head_idx: Int32,
    ) -> Tuple[Float32, Float32]:
        """DM sm100-mx: all scales are applied in the softmax step / MX scale factors."""
        return Float32(1.0), Float32(1.0)

    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        descale_tensors: Optional[DescaleTensors],
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        aux_tensors = aux_data.tensors

        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        if const_expr(self.dm_sspv):  # one shared S scratch at cols 256..383 for both stages
            tSAcc = cute.make_tensor(tStS[(None, None), 0, 0, 0].iterator + self.tmem_s_offset[0], tSAcc.layout)
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_op = (
            tcgen05.copy.LdRed32x32bOp(tcgen05.copy.Repetition(32))
            if const_expr(self.use_ldred_rowmax)
            else tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
        )
        tmem_load_atom = cute.make_copy_atom(tmem_load_op, self.qk_acc_dtype)
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(
            tidx
        )
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(8 if const_expr(self.q_dtype.width == 8) else 16)
            ),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)
        # DM sm100-mx: scale-factor words (4 x SFA replicated, 4 x SFB) in the free S columns after fp8 P
        tStSF = cute.make_tensor(
            tSAcc.iterator + (self.dm_sf_abs[0] - self.tmem_s_offset[0] + 8 * stage if const_expr(self.dm_sspv) else self.dm_sf_col),
            cute.composition(tSAcc.layout, cute.make_layout((self.m_block_size, 8)))
        )
        tmem_store_sf_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32)
        thr_tmem_store_sf = tcgen05.make_tmem_copy(tmem_store_sf_atom, tStSF).get_slice(tidx)
        tStSF_r2t = thr_tmem_store_sf.partition_D(tStSF)

        s_p_o_consumer_count = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        # DM ss_pv crd: per-WG count of sScale writes (persists across work tiles; pairs with mbar_dm_cread phases)
        dm_wcnt = cute.make_rmem_tensor((1,), Int32)
        dm_wcnt[0] = Int32(0)
        self._dm_wcnt = dm_wcnt   # set once before the tile loop (softmax_step reads it)
        # DM dbs: per-WG count of sScale EVENTS (initial notification, n-1 scale writes, final stats; persists across tiles)
        dm_ecnt = cute.make_rmem_tensor((1,), Int32)
        dm_ecnt[0] = Int32(0)
        self._dm_ecnt = dm_ecnt
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            # Keep the dynamic num_splits for the block-sparse helpers below.
            num_splits_dyn = num_splits
            if const_expr(self.is_split_kv and block_info.pack_split_idx):
                num_splits_dyn = split_idx >> 16
                split_idx = split_idx & 0xFFFF

            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                # cta_rank=0: see NOTE [SM100 forward coordinate spaces].
                m_block=self._q_block(work_m, stage),
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_data=aux_data,
                vec_size=self.mask_vec_size,
            )

            # Recompute fastdiv_mods if necessary
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisorV2(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisorV2(seqlen.seqlen_k),
                )

            mask_mod = self.mask_mod if const_expr(self.mask_mod is not None) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            # DM sm100-mx: row max kept in log2 units (integer valued); scales applied per 32-key block in softmax_step.
            dm_row = self._q_block(work_m, stage, thr_mma_qk.thr_idx) * self.m_block_size + tidx
            # rows past seqlen_q (second q stage of the last tile) are never stored: clamp the scale index (no OOB read)
            dm_qblk = cutlass.min(dm_row // 32, (seqlen.seqlen_q + 31) // 32 - 1)
            dm_qsc = Float32(descale_tensors.q_descale[batch_idx, head_idx, dm_qblk]) * Float32(math.log2(math.e))
            softmax = SoftmaxSm100.create(
                Float32(1.0),
                rescale_threshold=0.0,
                softmax_scale=None,
                max_offset=0,
            )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    work_m,
                    split_idx,
                    num_splits_dyn,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                    seqlen_info=seqlen,
                    kv_subtile_factor=self.kv_subtile_factor,
                )
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                thr_tmem_store_sf=thr_tmem_store_sf,
                tStSF_r2t=tStSF_r2t,
                dm_qsc=dm_qsc,
                dm_scales=descale_tensors,
                kv_head_idx=kv_head_idx,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=self._q_block(work_m, stage),
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
            )

            if const_expr(not self.use_s_ping_pong):
                if const_expr(self.use_block_sparsity) or has_work:
                    pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
                    sm_stats_producer_phase ^= 1

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                # When aux_tensors exist, Q indices beyond seqlen_q must be wrapped to avoid
                # OOB aux_tensor access. Only edge tiles (where m_tile_end > seqlen_q) need this.
                if const_expr(aux_tensors is not None):
                    m_tile_end = (self._q_block(work_m, stage) + self.cta_group_size) * self.m_block_size
                    check_m_boundary = m_tile_end > seqlen.seqlen_q
                else:
                    check_m_boundary = False
                (
                    s_p_o_consumer_count,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    work_m,
                    seqlen,
                    split_idx,
                    num_splits_dyn,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    s_p_o_consumer_count,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    self.q_stage,
                    Int32(stage),
                    check_m_boundary,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                    self.kv_subtile_factor,
                )
                if not empty_tile:
                    if const_expr(self.dm_dbs):
                        # dbs final-stats event (row_sum, row_max) into slot e & 1 (see softmax_step)
                        dm_e = dm_ecnt[0]
                        dm_slot = dm_e & 1
                        if dm_e >= 2:
                            self._dm_wait(self._dm_sb(1, stage, dm_slot, warp_idx), ((dm_e >> 1) - 1) & 1)
                        dm_so = dm_slot * (2 * self.s_p_o_stage * self.m_block_size)
                        sScale[tidx + stage * self.m_block_size + dm_so] = softmax.row_sum[0]
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            sScale[tidx + stage * self.m_block_size + self.s_p_o_stage * self.m_block_size + dm_so] = softmax.row_max[0]
                        dm_utils.mbar_arrive(self._dm_sb(0, stage, dm_slot, warp_idx))
                        dm_ecnt[0] = dm_e + 1
                    else:
                        if const_expr(self.dm_crd):
                            # ss_pv crd: the correction must have read the previous sScale value of this stage before it is overwritten
                            dm_wc = dm_wcnt[0]
                            if dm_wc > 0:
                                dm_utils.mbar_wait_parity(self.dm_cread + 8 * (stage), (dm_wc - 1) & 1)
                            dm_wcnt[0] = dm_wc + 1
                        if const_expr(self.dm_late):
                            # late final stats: PV_s(K-1) complete => the correction read scale_s(K-1) (and, across tiles, its previous
                            # final-stats read preceded its O release that PV_s(0) of this tile waited for); waits global completion #j (j = steps
                            # before this tile + K); exact: completed count in {j, j+1} (PV of the next tile needs a later P store)
                            dm_utils.mbar_wait_parity(self.dm_pvdone + 8 * stage, s_p_o_consumer_count)
                        sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            sScale[
                                tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                            ] = softmax.row_max[0]
                        # if tidx == 0:
                        #     cute.printf("softmax row sum stage %d: %f, row_max = %f\n", stage, softmax.row_sum[0], softmax.row_max[0])
                        # See block_sparse_utils.py NOTE [SM100 block-sparse empty tiles: mbarrier contract].
                        # pipeline_sm_stats.producer_commit_w_index(stage)
                        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
                    # if tidx == 0: cute.printf("softmax row sum stage %d: %f\n", stage, softmax.row_sum[0])
            else:
                if has_work:
                    s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                        s_p_o_consumer_count,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                    n_block_max -= 1
                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                            seqlen, work_m, n_block_min
                        )
                        for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    s_p_o_consumer_count,
                                    sm_stats_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    # The remaining iterations have no masking (but may still need mask_mod)
                    n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                        seqlen, work_m, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                        n_block = n_block_max - n_tile - 1
                        if const_expr(self.mask_mod is not None):
                            s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                                s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase, n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        else:
                            s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                                s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase, n_block,
                            )
                    # Separate iterations with local masking on the left
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            s_p_o_consumer_count, sm_stats_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    s_p_o_consumer_count,
                                    sm_stats_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                            # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                    # Dense path always writes scale / signals
                    if const_expr(self.use_s_ping_pong):
                        sm_stats_index = (s_p_o_consumer_count - 1) & 1
                    else:
                        sm_stats_index = stage
                    if const_expr(self.dm_dbs):
                        # dbs final-stats event (row_sum, row_max) into slot e & 1 (see softmax_step)
                        dm_e = dm_ecnt[0]
                        dm_slot = dm_e & 1
                        if dm_e >= 2:
                            self._dm_wait(self._dm_sb(1, sm_stats_index, dm_slot, warp_idx), ((dm_e >> 1) - 1) & 1)
                        dm_so = dm_slot * (2 * self.s_p_o_stage * self.m_block_size)
                        sScale[tidx + sm_stats_index * self.m_block_size + dm_so] = softmax.row_sum[0]
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            sScale[tidx + sm_stats_index * self.m_block_size + self.s_p_o_stage * self.m_block_size + dm_so] = softmax.row_max[0]
                        dm_utils.mbar_arrive(self._dm_sb(0, sm_stats_index, dm_slot, warp_idx))
                        dm_ecnt[0] = dm_e + 1
                    else:
                        if const_expr(self.dm_crd):
                            # ss_pv crd: the correction must have read the previous sScale value of this stage before it is overwritten
                            dm_wc = dm_wcnt[0]
                            if dm_wc > 0:
                                dm_utils.mbar_wait_parity(self.dm_cread + 8 * (sm_stats_index), (dm_wc - 1) & 1)
                            dm_wcnt[0] = dm_wc + 1
                        if const_expr(self.dm_late):
                            # late final stats: PV_s(K-1) complete => the correction read scale_s(K-1) (and, across tiles, its previous
                            # final-stats read preceded its O release that PV_s(0) of this tile waited for); waits global completion #j (j = steps
                            # before this tile + K); exact: completed count in {j, j+1} (PV of the next tile needs a later P store)
                            dm_utils.mbar_wait_parity(self.dm_pvdone + 8 * stage, s_p_o_consumer_count)
                        sScale[tidx + sm_stats_index * self.m_block_size] = softmax.row_sum[0]
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            sScale[
                                tidx + sm_stats_index * self.m_block_size + self.s_p_o_stage * self.m_block_size
                            ] = softmax.row_max[0]
                        # pipeline_sm_stats.producer_commit_w_index(stage)
                        sm_stats_barrier.arrive_w_index(index=sm_stats_index * 4 + warp_idx)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + cute.math.log2(softmax.row_sum[0], fastmath=True)) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (work_m * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (work_m * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        if const_expr(self.use_s_ping_pong):
            pipeline_sm_stats.producer_acquire_w_index_phase(
                s_p_o_consumer_count & 1, 1 ^ ((s_p_o_consumer_count >> 1) & 1)
            )
        else:
            pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        # This is equivalent to pipeline_s0_s1.producer_tail
        if const_expr(self.s0_s1_barrier):
            if stage == 0:
                pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)

    # ------------------------------------------------------------------------------------------------------------------
    # fwdmufu slice=1: slice softmax (warps 0-15) + row-level correction (warps 16-19). See __init__ and
    # Every value is computed by today's op sequence (bitwise O / LSE / P codes).
    # ------------------------------------------------------------------------------------------------------------------
    @cute.jit
    def slice_softmax_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,
        descale_tensors: Optional[DescaleTensors],
        pipeline_s_p_o: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (4 * cute.arch.WARP_SIZE)   # row; warp % 4 = TMEM lane quadrant
        q = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4     # key block owned by this slice warpgroup
        tSAcc = tStS[(None, None), 0, 0, 0]
        tSblk = cute.make_tensor(tSAcc.iterator, cute.composition(tSAcc.layout, cute.make_layout((self.m_block_size, 32))))
        thr_ld = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype), tSblk
        ).get_slice(tidx)
        tSt_ld = thr_ld.partition_S(tSblk)
        rS = cute.make_rmem_tensor(thr_ld.partition_D(cute.make_identity_tensor((self.m_block_size, 32))).shape, Float32)
        tSP = cute.make_tensor(tSAcc.iterator, cute.composition(tSAcc.layout, cute.make_layout((self.m_block_size, 8))))
        thr_st = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32), tSP
        ).get_slice(tidx)
        tSt_st = thr_st.partition_D(tSP)
        rP = cute.make_rmem_tensor(thr_st.partition_S(cute.make_identity_tensor((self.m_block_size, 8))).shape, Float32)
        rP8 = cute.make_tensor(cute.recast_ptr(rP.iterator, dtype=self.q_dtype), rS.layout)
        dm_lean = const_expr(self.dm_opts.get("lean", "0") == "1")
        st = cute.make_rmem_tensor((3,), Int32)   # [0] s_p_o full phase, [1] steps done per stage, [2] prefetched K scale word
        st[0] = Int32(0)
        st[1] = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            has_work = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)
            qsc = [None] * 2
            for s in cutlass.range_constexpr(2):
                dm_row = self._q_block(work_m, s, thr_mma_qk.thr_idx) * self.m_block_size + tidx
                dm_qblk = cutlass.min(dm_row // 32, (seqlen.seqlen_q + 31) // 32 - 1)
                qsc[s] = Float32(descale_tensors.q_descale[batch_idx, head_idx, dm_qblk]) * Float32(math.log2(math.e))
            step = partial(self._slice_step, sbase=Int32(tSAcc.iterator.toint()), q=q, tidx=tidx, st=st, qsc=qsc, rS=rS, rP=rP, rP8=rP8, tSt_ld=tSt_ld,
                           tSt_st=tSt_st, thr_ld=thr_ld, thr_st=thr_st, dm_lean=dm_lean, pipeline_s_p_o=pipeline_s_p_o,
                           descale_tensors=descale_tensors, batch_idx=batch_idx, kv_head_idx=kv_head_idx,
                           work_m=work_m, seqlen=seqlen)
            if has_work:
                st[2] = Int32(descale_tensors.k_descale[batch_idx, kv_head_idx, 2 * cutlass.max(n_block_max - 1, 0)])
                step(n_block_max - 1, mask_seqlen=True, mask_causal=self.is_causal)
                n_block_max -= 1
                if const_expr(self.is_causal):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, work_m, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                        step(n_block_max - 1 - n_tile, mask_seqlen=False, mask_causal=True)
                    n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    step(n_block_max - 1 - n_tile, mask_seqlen=False, mask_causal=False)
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def _slice_step(
        self, n_block: Int32, mask_seqlen: cutlass.Constexpr, mask_causal: cutlass.Constexpr, sbase: Int32, q: Int32, tidx: Int32,
        st: cute.Tensor, qsc, rS: cute.Tensor, rP: cute.Tensor, rP8: cute.Tensor, tSt_ld: cute.Tensor,
        tSt_st: cute.Tensor, thr_ld, thr_st, dm_lean: cutlass.Constexpr, pipeline_s_p_o, descale_tensors,
        batch_idx: Int32, kv_head_idx: Int32, work_m: Int32, seqlen,
    ):
        """One n_block for both Q stages (stage 0 then stage 1) of this slice warp's key block q."""
        if const_expr(self.dm_opts.get("pf", "1") == "1"):
            dm_kw = st[2]
            # next n_block's K scale word (loaded now, consumed next step; clamped index on the last step)
            st[2] = Int32(descale_tensors.k_descale[batch_idx, kv_head_idx, 2 * cutlass.max(n_block - 1, 0)])
        else:
            dm_kw = Int32(descale_tensors.k_descale[batch_idx, kv_head_idx, 2 * n_block])
        dm_ks = dm_utils.byte_pow2_f32_rt(dm_kw, q * 8)
        ph = st[0]
        nstep = st[1]
        for s in cutlass.range_constexpr(2):
            c = qsc[s] * dm_ks
            pipeline_s_p_o.consumer_wait_w_index_phase(s, ph)
            col = Int32(self.tmem_s_offset[s]) + q * 32
            cute.copy(thr_ld, cute.make_tensor(tSt_ld.iterator + col, tSt_ld.layout), rS)
            if const_expr(self.dm_ordpc in (4, 5)):
                dm_sv = [rS[16 + j] for j in range(8)]   # DEBUG planted control: raw S of cols 16..23 (first load)
            if const_expr(mask_seqlen or mask_causal):
                # = AttentionMask.apply_mask_sm100 (non-pack-GQA, 1-CTA) restricted to columns [32 q, 32 q + 32)
                n_blk = cutlass.max(n_block, 0)   # apply_mask_sm100 treats negative n_blocks as block 0 (empty KV range)
                seqlenk_col_limit = seqlen.seqlen_k - n_blk * self.n_block_size
                if const_expr(mask_causal):
                    causal_row_offset = seqlen.seqlen_k - n_blk * self.n_block_size - seqlen.seqlen_q
                    row_idx = tidx + self._q_block(work_m, s) * self.m_block_size
                    col_limit = row_idx + causal_row_offset + 1
                    if const_expr(mask_seqlen):
                        col_limit = cutlass.min(col_limit, seqlenk_col_limit)
                else:
                    col_limit = seqlenk_col_limit
                lim = col_limit - q * 32
                mask_r2p_lambda(rS, lambda ch: r2p_bitmask_below(lim, ch), rank1=True)
            # block max: today's FMNMX order (lean / max3), then x_b = max * c_b
            if const_expr(dm_lean):
                lm = [utils.fmax(rS[3 * t], rS[3 * t + 1], rS[3 * t + 2]) for t in range(4)]
                for i in cutlass.range_constexpr(12, 32, 2):
                    t = ((i - 12) // 2) % 4
                    lm[t] = utils.fmax(lm[t], rS[i], rS[i + 1])
                v = utils.fmax(utils.fmax(lm[0], lm[1], lm[2]), lm[3])
            else:
                lm = [utils.fmax(rS[0], rS[1]), utils.fmax(rS[2], rS[3]), utils.fmax(rS[4], rS[5]), utils.fmax(rS[6], rS[7])]
                for i in cutlass.range_constexpr(8, 32, 8):
                    for t in cutlass.range_constexpr(4):
                        lm[t] = utils.fmax(lm[t], rS[i + 2 * t], rS[i + 2 * t + 1])
                v = utils.fmax(utils.fmax(lm[0], lm[1]), lm[2], lm[3])
            xb = v * c
            dm_utils.st_shared_b32(self.dm_slbuf + 4 * ((s * 4 + q) * self.m_block_size + tidx), dm_utils.bitcast_f32_i32(xb))
            if const_expr(self.dm_ordpc in (4, 5)):
                dm_pc_a = sbase + ((cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32 << 16) + col + 16
            if const_expr((self.dm_ordpc == 4 and self.dm_ordld) or self.dm_ordpc == 5):
                # ordpc=5: the re-load at the SAME place as ordpc=4 with ordld, i.e. before the
                # arrival; only the wait::ld is controlled by ordld (with ordld=0 its completion is forced after the arrival)
                dm_pv = dm_utils.tmem_ld_32x32b_x8(dm_pc_a)   # re-load of this block's cols 16..23 (q = 0: the SF columns)
            if const_expr(self.dm_ordld):
                # ordinv: the correction WG's SF tcgen05.st overwrites S_s cols 16..23 that slice WG 0 loaded; PTX
                # 9.7.18.6.4.4/.5: a cross-thread ld -> st hand-off needs tcgen05.wait::ld before the fence + arrive
                # (register use of the loaded values is not a memory ordering)
                cute.arch.fence_view_async_tmem_load()
            dm_utils.tc_fence_before()     # this warp's S loads ordered before the kb-published arrival (SF overwrites S)
            dm_utils.tcf_relay()   # fwdfence v3: every lane fences + warp sync before the elected leader's release
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                dm_utils.mbar_arrive(self.dm_slbar + 8 * s)
            if const_expr(self.dm_ordpc in (4, 5)):
                dm_pois = Float32(0.0)
                if const_expr(self.dm_ordpc == 5 and not self.dm_ordld):
                    cute.arch.fence_view_async_tmem_load()
                if const_expr(self.dm_ordpc == 4 and not self.dm_ordld):
                    dm_utils.nanosleep_1us()
                    dm_utils.nanosleep_1us()
                    dm_pv = dm_utils.tmem_ld_32x32b_x8(dm_pc_a)   # the re-load at the latest point PTX permits without wait::ld
                    cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range_constexpr(8):
                    if dm_utils.bitcast_f32_i32(dm_pv[j]) != dm_utils.bitcast_f32_i32(dm_sv[j]):
                        dm_pois = Float32(float("nan"))
            kb = cute.arch.fmax(dm_utils.floor_f32(xb), Float32(-1e30))
            bias = Float32(7.0) - kb
            for i in cutlass.range_constexpr(0, 32, 2):
                rS[i], rS[i + 1] = cute.arch.fma_packed_f32x2((rS[i], rS[i + 1]), (c, c), (bias, bias))
            for i in cutlass.range_constexpr(0, 32, 2):
                rS[i] = cute.math.exp2(rS[i], fastmath=True)
                rS[i + 1] = cute.math.exp2(rS[i + 1], fastmath=True)
            rP8.store(rS.load().to(self.q_dtype))
            cute.copy(thr_st, rP, cute.make_tensor(tSt_st.iterator + col, tSt_st.layout))
            cute.arch.fence_view_async_tmem_store()
            dm_utils.tc_fence_before()     # P tcgen05.st ordered before the P-full arrival (PV reads it)
            if const_expr(self.dm_opts.get("warr", "0") == "1"):
                dm_utils.tcf_relay()   # fwdfence v3: every lane fences + warp sync before the elected leader's release
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_s_p_o.consumer_release_w_index(s)
            else:
                pipeline_s_p_o.consumer_release_w_index(s)
            ps = (rS[0], rS[1])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (rS[i], rS[i + 1]))
            bsum = ps[0] + ps[1]
            if const_expr(self.dm_ordpc in (4, 5)):
                bsum = bsum + dm_pois   # DEBUG planted control: a re-load that saw the SF store poisons l (-> O, LSE NaN)
            if nstep > 0:
                if const_expr(self.dm_opts.get("slnobe", "0") != "1"):   # slnobe=1: DEBUG planted WAR bug (no bsempty wait)
                    dm_utils.mbar_wait_parity(self.dm_slbar + 32 + 8 * s, (nstep - 1) & 1)
            dm_utils.st_shared_b32(self.dm_slbuf + 4 * (1024 + (s * 4 + q) * self.m_block_size + tidx),
                                   dm_utils.bitcast_f32_i32(bsum))
            dm_utils.tcf_relay()   # fwdfence v3: every lane fences + warp sync before the elected leader's release
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                dm_utils.mbar_arrive(self.dm_slbar + 16 + 8 * s)
        st[0] = ph ^ 1
        st[1] = nstep + 1

    @cute.jit
    def slice_correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_o_epi: Optional[pipeline.PipelineAsync],
        pipeline_load_epi: Optional[pipeline.PipelineAsync],
        descale_tensors: Optional[DescaleTensors],
        gmem_tiled_copy_O: cute.TiledCopy,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        mma_tile_coord_v = thr_mma_qk.thr_idx
        dm_lean = const_expr(self.dm_opts.get("lean", "0") == "1")
        tSAcc = tStS[(None, None), 0, 0, 0]
        tSF = cute.make_tensor(tSAcc.iterator, cute.composition(tSAcc.layout, cute.make_layout((self.m_block_size, 8))))
        thr_sf = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32), tSF
        ).get_slice(tidx)
        tSt_sf = thr_sf.partition_D(tSF)
        rSF = cute.make_rmem_tensor(thr_sf.partition_S(cute.make_identity_tensor((self.m_block_size, 8))).shape, Float32)
        rst = cute.make_rmem_tensor((5,), Float32)    # m0, m1, l0, l1, acc (scratch)
        cnt = cute.make_rmem_tensor((2,), Int32)      # [0] steps done per stage (kb / bs barrier parity), [1] V scale word
        cnt[0] = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)
        load_epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.epi_head_dim_v)
            gO = cute.local_tile(mO_cur, tiler_gO, (work_m, 0))
            gO = layout_utils.select(cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1])
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]
            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None) else None, True)] * self.q_stage
            total_block_count = n_block_max - n_block_min
            has_work = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)
            if has_work:
                cnt[1] = Int32(descale_tensors.k_descale[batch_idx, kv_head_idx, 2 * cutlass.max(n_block_max - 1, 0) + 1])
                # one step even when the range is empty (seqlen_k == 0): slices and MMA always run the first step
                for k in cutlass.range(cutlass.max(total_block_count, 1), unroll=1):
                    n_block = n_block_max - 1 - k
                    sfb_word = dm_utils.bitcast_i32_f32(cnt[1])   # V scale word of n_block (prefetched one step ahead)
                    cnt[1] = Int32(descale_tensors.k_descale[batch_idx, kv_head_idx, 2 * cutlass.max(n_block - 1, 0) + 1])
                    nstep = cnt[0]
                    for s in cutlass.range_constexpr(2):
                        # ---- row max / rescale / SF bytes (today's softmax_step prefix, verbatim ops) ----
                        if const_expr(self.dm_opts.get("slnokb", "0") != "1"):   # slnokb=1: DEBUG planted RAW bug (no kbfull wait)
                            dm_utils.mbar_wait_parity(self.dm_slbar + 8 * s, nstep & 1)
                        dm_utils.tc_fence_after()
                        x = [dm_utils.bitcast_i32_f32(dm_utils.ld_shared_b32(
                            self.dm_slbuf + 4 * ((s * 4 + j) * self.m_block_size + tidx))) for j in range(4)]
                        if const_expr(dm_lean):
                            dm_xmax = utils.fmax(utils.fmax(x[0], x[1], x[2]), x[3])
                        else:
                            dm_xmax = cute.arch.fmax(cute.arch.fmax(x[0], x[1]), cute.arch.fmax(x[2], x[3]))
                        dm_mcand = dm_utils.ceil_f32(dm_xmax)
                        m_old = rst[s]
                        row_max = m_old
                        acc_scale = Float32(1.0)
                        if k == 0:
                            row_max = dm_mcand
                            acc_scale = Float32(0.0)
                        else:
                            if dm_mcand > m_old + Float32(self.dm_rescale_T):
                                row_max = dm_mcand
                                acc_scale = dm_utils.exp2i_f32(m_old - dm_mcand)
                        rst[s] = row_max
                        m_safe = row_max if row_max != -Float32.inf else Float32(0.0)
                        dm_w = [None] * 4
                        dm_sfa = [None] * 4
                        for j in cutlass.range_constexpr(4):
                            kb = cute.arch.fmax(dm_utils.floor_f32(x[j]), Float32(-1e30))
                            e = kb - Float32(7.0) - m_safe
                            if const_expr(dm_lean):
                                dm_sfa[j] = dm_utils.ue8m0_byte(e)
                                dm_w[j] = dm_utils.byte_shl23_f32(dm_sfa[j])
                            else:
                                dm_w[j] = dm_utils.exp2i_f32(e)
                                dm_sfa[j] = dm_utils.ue8m0_byte(e)
                        sfa_word = dm_utils.bitcast_i32_f32(dm_utils.pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3]))
                        for j in cutlass.range_constexpr(4):
                            rSF[j] = sfa_word
                            rSF[4 + j] = sfb_word
                        cute.copy(thr_sf, rSF, cute.make_tensor(
                            tSt_sf.iterator + (self.tmem_s_offset[s] + self.dm_sl_sfcol), tSt_sf.layout))
                        if k > 0:
                            # S_s(k) complete (seen through the slices' kb arrivals) => PV_s(k-1) complete
                            should_rescale = cute.arch.vote_ballot_sync(acc_scale < 1.0) != 0
                            if const_expr(self.dm_opts.get("slnr", "0") == "1"):
                                should_rescale = False   # DEBUG ONLY (positive control): planted bug, O never rescaled
                            if should_rescale:
                                self.correction_rescale(thr_mma_pv, tOtO[None, None, None, s], tidx, acc_scale)
                        cute.arch.fence_view_async_tmem_store()
                        cute.arch.fence_view_async_tmem_load()   # (k == 0) previous tile's epilogue O loads done
                        dm_utils.tc_fence_before()
                        if const_expr(self.dm_opts.get("warr", "0") == "1"):
                            dm_utils.tcf_relay()   # fwdfence v3: every lane fences + warp sync before the elected leader's release
                            cute.arch.sync_warp()
                            with cute.arch.elect_one():
                                pipeline_s_p_o.consumer_release_w_index(s)    # SF written + O rescaled (+ O free at k == 0)
                        else:
                            pipeline_s_p_o.consumer_release_w_index(s)
                        # ---- l = l * acc + sum_b s_b w_b (today's order) ----
                        dm_utils.mbar_wait_parity(self.dm_slbar + 16 + 8 * s, nstep & 1)
                        if const_expr(int(self.dm_opts.get("slcdly", "0")) > 0):   # DEBUG: widen the sB WAR window
                            for _i in cutlass.range_constexpr(int(self.dm_opts.get("slcdly", "0"))):
                                dm_utils.nanosleep_1us()
                        bs = [dm_utils.bitcast_i32_f32(dm_utils.ld_shared_b32(
                            self.dm_slbuf + 4 * (1024 + (s * 4 + j) * self.m_block_size + tidx))) for j in range(4)]
                        dm_utils.tcf_relay()   # fwdfence v3: every lane fences + warp sync before the elected leader's release
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            dm_utils.mbar_arrive(self.dm_slbar + 32 + 8 * s)
                        dm_l = rst[2 + s] * acc_scale
                        if k == 0:
                            dm_l = Float32(0.0)
                        for j in cutlass.range_constexpr(4):
                            dm_l = dm_l + bs[j] * dm_w[j]
                        rst[2 + s] = dm_l
                    cnt[0] = nstep + 1
                # ---- epilogue (= correction_loop, dense non-split path) ----
                for stage in cutlass.range_constexpr(self.q_stage):
                    row_sum = rst[2 + stage]
                    row_max = rst[stage] if const_expr(mLSE is not None) else None
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    scale = scale * Float32(1.0)
                    pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                    dm_utils.tc_fence_after()
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        work_m,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO[None, None, stage],
                        gmem_tiled_copy_O,
                        zero_fill=seqlen.seqlen_k == 0 if const_expr(self.use_tma_KV) else False,
                    )
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_commit_w_index(stage)
                o_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    q_block = self._q_block(work_m, stage, mma_tile_coord_v)
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * Float32(1.0) + (cute.math.log2(row_sum, fastmath=True) - Float32(0.0))) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (q_block,))
                    if tidx < seqlen.seqlen_q - q_block * self.m_block_size:
                        gLSE[tidx] = lse
            if const_expr(pipeline_load_epi is not None and self.use_correction_warps_for_epi):
                pipeline_load_epi.producer_acquire(load_epi_producer_state)
                with cute.arch.elect_one():
                    pipeline_load_epi.producer_commit(load_epi_producer_state)
                load_epi_producer_state.advance()
            work_tile = tile_scheduler.advance_to_next_work()
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)


    def _dm_wait(self, mbar, phase):
        if const_expr(self.dm_opts.get("wnh", "1") == "1"):
            dm_utils.mbar_wait_parity_nh(mbar, phase)
        else:
            dm_utils.mbar_wait_parity(mbar, phase)

    def _dm_sb(self, kind, stage, slot, warp):
        """dbs mbarrier address: kind 0 = full (softmax -> correction), 1 = empty (correction -> softmax)."""
        return self.dm_cread + 8 * (((kind * 2 + stage) * 2 + slot) * 4 + warp)

    @cute.jit
    def softmax_step(
        self,
        s_p_o_consumer_count: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        thr_tmem_store_sf: cute.CopyAtom,
        tStSF_r2t: cute.Tensor,
        dm_qsc: Float32,
        dm_scales,
        kv_head_idx: Int32,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages

        A None mask_fn means the tcgen05.ld.red hardware max is valid.
        s_p_o_consumer_count is a global block count with S ping-pong, else the S consumer phase bit.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        # tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # DM: issue this tile's K/V block-scale loads before waiting for S (hides the L2 latency behind the QK MMA)
        nb4 = n_block * 4
        if const_expr(self.dm_opts.get("kvpack", "1") == "1"):
            # packed UE8M0 bytes: k_descale[b, hk, 2*n_block] = K scale bytes, [.., 2*n_block+1] = V scale bytes (= SFB word)
            dm_kw = Int32(dm_scales.k_descale[batch_idx, kv_head_idx, 2 * n_block])
            dm_vw = Int32(dm_scales.k_descale[batch_idx, kv_head_idx, 2 * n_block + 1])
            dm_ks = [dm_utils.byte_pow2_f32(dm_kw, j) for j in range(4)]
        else:
            dm_ks = [Float32(dm_scales.k_descale[batch_idx, kv_head_idx, nb4 + j]) for j in range(4)]
            dm_vs = [Float32(dm_scales.v_descale[batch_idx, kv_head_idx, nb4 + j]) for j in range(4)]
        # Wait for Si
        if const_expr(self.use_s_ping_pong):
            s_p_o_index = s_p_o_consumer_count & 1
            s_p_o_phase = (s_p_o_consumer_count >> 1) & 1
            pipeline_s_p_o.consumer_wait_w_index_phase(s_p_o_index, s_p_o_phase)
            tmem_s_p_offset = s_p_o_index * self.n_block_size
            tStS_t2r = cute.make_tensor(tStS_t2r.iterator + tmem_s_p_offset, tStS_t2r.layout)
            tStP_r2t = cute.make_tensor(tStP_r2t.iterator + tmem_s_p_offset, tStP_r2t.layout)
        else:
            s_p_o_index = stage
            pipeline_s_p_o.consumer_wait_w_index_phase(stage, s_p_o_consumer_count)
            if const_expr(self.dm_sspv and self.dm_opts.get("tcf", "1") == "1"):
                dm_utils.tc_fence_after()   # QK_s commit (S full) -> this WG's S tcgen05.ld
        tSrS_t2r = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        hw_row_max = Float32(-Float32.inf)
        if const_expr(self.use_ldred_rowmax):
            # ld.red returns each x32 tile's max in an extra register.
            tSrS_red = cute.make_rmem_tensor(((1, 1), *tSrS_t2r.shape[1:]), self.qk_acc_dtype)
            cute.copy(thr_tmem_load, tStS_t2r, (tSrS_t2r, tSrS_red))
            for i in cutlass.range_constexpr(cute.size(tSrS_red.shape)):
                hw_row_max = cute.arch.fmax(hw_row_max, tSrS_red[i])
        else:
            cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if const_expr(not self.dm_sspv and self.dm_opts.get("tcf", "1") != "0"):
            # formal tcgen05.wait::ld of S before the P release lets the next QK overwrite S (P aliases S)
            cute.arch.fence_view_async_tmem_load()
        if const_expr(self.dm_sspv):
            # S scratch fully read by this warpgroup -> the MMA warp may issue the next QK into it
            cute.arch.fence_view_async_tmem_load()
            if const_expr(self.dm_opts.get("tcf", "1") == "1"):
                dm_utils.tc_fence_before()   # S loads ordered before the arrive that lets the next QK overwrite S
            dm_utils.mbar_arrive(self.dm_sfree)
        # tSrS_t2r = copy_utils.load_t2r(thr_tmem_load, tScS_shape, tStS_t2r)
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_data,
                fastdiv_mods,
                head_divmod,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        # ---------------- DM sm100-mx softmax step ----------------
        # x = S * c_b (log2 units), c_b = qs_row * log2e * ks_b; per 32-key block b: kb = floor(max_b x),
        # p~ = exp2(x - kb + 7) in [0, 256) -> e4m3 code; SFA_b = 2^(kb - 7 - m), SFB_b = vs_b; m = integer running max.
        dm_c = [dm_qsc * dm_ks[j] for j in range(4)]
        dm_bmax = [None] * 4
        dm_lean = const_expr(self.dm_opts.get("lean", "0") == "1")
        for j in cutlass.range_constexpr(4):
            if const_expr(dm_lean):
                # C5: 16 FMNMX3 per 32-key block (4 independent chains), exact max -> bitwise
                b0 = 32 * j
                lm = [utils.fmax(tSrS_t2r[b0 + 3 * t], tSrS_t2r[b0 + 3 * t + 1], tSrS_t2r[b0 + 3 * t + 2])
                      for t in range(4)]
                for i in cutlass.range_constexpr(12, 32, 2):
                    t = ((i - 12) // 2) % 4
                    lm[t] = utils.fmax(lm[t], tSrS_t2r[b0 + i], tSrS_t2r[b0 + i + 1])
                v = utils.fmax(utils.fmax(lm[0], lm[1], lm[2]), lm[3])
            elif const_expr(self.dm_opts.get("max3", "1") == "1"):
                b0 = 32 * j
                lm = [utils.fmax(tSrS_t2r[b0], tSrS_t2r[b0 + 1]), utils.fmax(tSrS_t2r[b0 + 2], tSrS_t2r[b0 + 3]),
                      utils.fmax(tSrS_t2r[b0 + 4], tSrS_t2r[b0 + 5]), utils.fmax(tSrS_t2r[b0 + 6], tSrS_t2r[b0 + 7])]
                for i in cutlass.range_constexpr(8, 32, 8):
                    for t in cutlass.range_constexpr(4):
                        lm[t] = utils.fmax(lm[t], tSrS_t2r[b0 + i + 2 * t], tSrS_t2r[b0 + i + 2 * t + 1])
                v = utils.fmax(utils.fmax(lm[0], lm[1]), lm[2], lm[3])
            else:
                v = tSrS_t2r[32 * j]
                for i in cutlass.range_constexpr(1, 32):
                    v = cute.arch.fmax(v, tSrS_t2r[32 * j + i])
            dm_bmax[j] = v * dm_c[j]
        if const_expr(dm_lean):
            dm_xmax = utils.fmax(utils.fmax(dm_bmax[0], dm_bmax[1], dm_bmax[2]), dm_bmax[3])
        else:
            dm_xmax = cute.arch.fmax(cute.arch.fmax(dm_bmax[0], dm_bmax[1]), cute.arch.fmax(dm_bmax[2], dm_bmax[3]))
        dm_mcand = dm_utils.ceil_f32(dm_xmax)
        acc_scale = Float32(1.0)
        row_max = dm_mcand
        if const_expr(is_first):
            row_max = dm_mcand
            acc_scale = Float32(0.0)
        else:
            m_old = softmax.row_max[0]
            row_max = m_old
            if dm_mcand > m_old + Float32(self.dm_rescale_T):
                row_max = dm_mcand
                acc_scale = dm_utils.exp2i_f32(m_old - dm_mcand)
        softmax.row_max[0] = row_max
        m_safe = row_max if row_max != -Float32.inf else Float32(0.0)
        # fully masked blocks: kb clamps to -1e30 -> weight 0, SF byte 0, finite bias (x stays -inf -> p 0)
        dm_bias = [None] * 4
        dm_w = [None] * 4
        dm_sfa = [None] * 4
        for j in cutlass.range_constexpr(4):
            kb = cute.arch.fmax(dm_utils.floor_f32(dm_bmax[j]), Float32(-1e30))
            e = kb - Float32(7.0) - m_safe
            dm_bias[j] = Float32(7.0) - kb
            if const_expr(dm_lean):
                # one cvt/clamp builds the UE8M0 byte; the weight 2^e is the same byte shifted into the fp32 exponent
                dm_sfa[j] = dm_utils.ue8m0_byte(e)
                dm_w[j] = dm_utils.byte_shl23_f32(dm_sfa[j])
            else:
                dm_w[j] = dm_utils.exp2i_f32(e)
                dm_sfa[j] = dm_utils.ue8m0_byte(e)
        sfa_word = dm_utils.bitcast_i32_f32(dm_utils.pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3]))
        if const_expr(self.dm_opts.get("kvpack", "1") == "1"):
            sfb_word = dm_utils.bitcast_i32_f32(dm_vw)
        else:
            sfb_word = dm_utils.bitcast_i32_f32(dm_utils.pack_bytes4(*[dm_utils.pow2_exponent_byte(dm_vs[j]) for j in range(4)]))
        tSrSF = cute.make_rmem_tensor(thr_tmem_store_sf.partition_S(cute.make_identity_tensor((self.m_block_size, 8))).shape, Float32)
        for j in cutlass.range_constexpr(4):
            tSrSF[j] = sfa_word
            tSrSF[4 + j] = sfb_word
        if const_expr(not self.dm_sspv):
            cute.copy(thr_tmem_store_sf, tSrSF, tStSF_r2t)

        if const_expr(self.use_s_ping_pong):
            sm_stats_index = (s_p_o_consumer_count - 1) & 1
        else:
            sm_stats_index = s_p_o_index
        if const_expr(self.dm_dbs):
            # dbs event e: slot e & 1; reuse of a slot waits until the correction released event e-2 (empty #(e/2 - 1))
            dm_ecnt_l = self._dm_ecnt
            dm_e = dm_ecnt_l[0]
            dm_slot = dm_e & 1
            if dm_e >= 2:
                self._dm_wait(self._dm_sb(1, sm_stats_index, dm_slot, warp_idx), ((dm_e >> 1) - 1) & 1)
            if const_expr(not is_first):
                sScale[thr_tmem_load.thr_idx + sm_stats_index * self.m_block_size
                       + dm_slot * (2 * self.s_p_o_stage * self.m_block_size)] = acc_scale
            dm_utils.mbar_arrive(self._dm_sb(0, sm_stats_index, dm_slot, warp_idx))
            dm_ecnt_l[0] = dm_e + 1
        elif const_expr(not self.dm_late):
            if const_expr(not is_first):
                thread_idx = thr_tmem_load.thr_idx
                if const_expr(self.dm_crd):
                    # ss_pv crd: the correction must have read the previous sScale value of this stage before it is overwritten
                    dm_wcnt_l = self._dm_wcnt
                    dm_wc = dm_wcnt_l[0]
                    if dm_wc > 0:
                        dm_utils.mbar_wait_parity(self.dm_cread + 8 * (sm_stats_index), (dm_wc - 1) & 1)
                    dm_wcnt_l[0] = dm_wc + 1
                sScale[thread_idx + sm_stats_index * self.m_block_size] = acc_scale
            if const_expr(not self.use_s_ping_pong or not is_first):
                sm_stats_barrier.arrive_w_index(index=sm_stats_index * 4 + warp_idx)

        # x = S * c_b + (7 - kb)
        for j in cutlass.range_constexpr(4):
            for i in cutlass.range_constexpr(0, 32, 2):
                tSrS_t2r[32 * j + i], tSrS_t2r[32 * j + i + 1] = cute.arch.fma_packed_f32x2(
                    (tSrS_t2r[32 * j + i], tSrS_t2r[32 * j + i + 1]),
                    (dm_c[j], dm_c[j]),
                    (dm_bias[j], dm_bias[j]),
                )
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        if const_expr(self.dm_opts.get("pipe", "0") == "1" and self.dm_opts.get("exp", "f32") != "h2"
                      and self.ex2_emu_freq == 0 and self.dm_opts.get("nosum", "0") != "1"):
            # fwdrw pipe=1: per-32-key-block software pipeline (bitwise = default): block j's FFMA2 already applied above;
            # exps+convert+TMEM store of block j are interleaved (in program order) with the UNWEIGHTED row-sum chain of block
            # j-1, so the FADD2 sum work overlaps the MUFU exps instead of trailing after the P release. The weighted
            # accumulation l = ((l*acc + s0 w0) + s1 w1) + ... keeps today's exact operation order.
            frgS = cute.logical_divide(tSrS_t2r, cute.make_layout(32))
            frgP = cute.logical_divide(tSrP_r2t, cute.make_layout(32))
            n_st = const_expr(cute.size(tStP_r2t.shape[2]))
            dm_bs = [None] * 4
            for j in cutlass.range_constexpr(4):
                for k in cutlass.range_constexpr(0, 32, 2):
                    frgS[k, j] = cute.math.exp2(frgS[k, j], fastmath=True)
                    frgS[k + 1, j] = cute.math.exp2(frgS[k + 1, j], fastmath=True)
                frgP[None, j].store(frgS[None, j].load().to(tSrP_r2t.element_type))
                if const_expr(j >= 1):
                    ps = (tSrS_t2r[32 * (j - 1)], tSrS_t2r[32 * (j - 1) + 1])
                    for i in cutlass.range_constexpr(2, 32, 2):
                        ps = cute.arch.add_packed_f32x2(ps, (tSrS_t2r[32 * (j - 1) + i], tSrS_t2r[32 * (j - 1) + i + 1]))
                    dm_bs[j - 1] = ps[0] + ps[1]
                if const_expr(n_st == 4):
                    cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, j], tStP_r2t[None, None, j])
                    if const_expr(self.split_P_arrive > 0):
                        if const_expr(j + 1 == 4 * self.split_P_arrive // self.n_block_size):
                            cute.arch.fence_view_async_tmem_store()
                            pipeline_s_p_o.consumer_release_w_index(s_p_o_index)
            if const_expr(n_st != 4):
                for i in cutlass.range_constexpr(n_st):
                    cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
                    if const_expr(self.split_P_arrive > 0):
                        if const_expr(i + 1 == n_st * self.split_P_arrive // self.n_block_size):
                            cute.arch.fence_view_async_tmem_store()
                            pipeline_s_p_o.consumer_release_w_index(s_p_o_index)
            cute.arch.fence_view_async_tmem_store()
            if const_expr(self.split_P_arrive > 0):
                cute.arch.sync_warp()
                dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                with cute.arch.elect_one():
                    pipeline_p_lastsplit.producer_commit_w_index(s_p_o_index)
            else:
                pipeline_s_p_o.consumer_release_w_index(s_p_o_index)
            ps = (tSrS_t2r[96], tSrS_t2r[97])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (tSrS_t2r[96 + i], tSrS_t2r[96 + i + 1]))
            dm_bs[3] = ps[0] + ps[1]
            if const_expr(self.use_s_ping_pong):
                pipeline_sm_stats.producer_acquire_w_index_phase(
                    s_p_o_consumer_count & 1, 1 ^ ((s_p_o_consumer_count >> 1) & 1)
                )
            else:
                pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
            dm_l = Float32(0.0) if const_expr(is_first) else softmax.row_sum[0] * acc_scale
            for j in cutlass.range_constexpr(4):
                dm_l = dm_l + dm_bs[j] * dm_w[j]
            softmax.row_sum[0] = dm_l
        else:
            dm_bsum = [None] * 4
            if const_expr(self.dm_opts.get("exp", "f32") == "h2"):
                # f16x2 MUFU exp (2 exps / instruction) + direct f16x2 -> e4m3x2 convert; block sums in f16x2 (tree) -> f32
                for j in cutlass.range_constexpr(4):
                    hs = []
                    for w in cutlass.range_constexpr(8):
                        i0 = 32 * j + 4 * w
                        word, e0, e1 = dm_utils.exp2_h2_e4m3x4(tSrS_t2r[i0], tSrS_t2r[i0 + 1], tSrS_t2r[i0 + 2], tSrS_t2r[i0 + 3])
                        tSrP_r2t_f32[8 * j + w] = dm_utils.bitcast_i32_f32(word)
                        hs.append(e0)
                        hs.append(e1)
                    dm_bsum[j] = dm_utils.h2_sum_f32(dm_utils.hadd2_tree(hs))
            else:
                softmax.apply_exp2_convert(
                    tSrS_t2r,
                    tSrP_r2t,
                    ex2_emu_freq=self.ex2_emu_freq,
                    ex2_emu_start_frg=self.ex2_emu_start_frg,
                )
            if const_expr(self.s0_s1_barrier):
                pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
            if const_expr(self.dm_sspv):
                # SF_s and P_s(previous tile) are free once PV_s(previous) completed (completion #k incl. a dummy #0);
                # waited only after the exps so the PV latency hides under them
                if const_expr(self.dm_opts.get("nosw", "0") != "1"):
                    dm_utils.mbar_wait_parity(self.dm_pvdone + 8 * stage, s_p_o_consumer_count)
                if const_expr(self.dm_opts.get("tcf", "1") == "1"):
                    dm_utils.tc_fence_after()   # SF tcgen05.st must not be ordered before PV_s(k-1)'s completion
                if const_expr(self.dm_late):
                    # late: PV_s(k-1) complete => the correction already read scale_s(k-1) (it read it before its
                    # O-rescaled(k-1) arrival, which PV_s(k-1) waited for) => the single sScale_s slot is free
                    if const_expr(not is_first):
                        sScale[thr_tmem_load.thr_idx + sm_stats_index * self.m_block_size] = acc_scale
                    if const_expr(not self.use_s_ping_pong or not is_first):
                        sm_stats_barrier.arrive_w_index(index=sm_stats_index * 4 + warp_idx)
                cute.copy(thr_tmem_store_sf, tSrSF, tStSF_r2t)
                # P codes -> sP[stage] (K-major SW128: row r at r*128 B, 16-B chunk c at (c ^ (r & 7)) * 16)
                tidx_row = Int32(thr_tmem_load.thr_idx)
                dm_prow = Int32(self.dm_sP.iterator.toint()) + Int32(16384) * stage + tidx_row * 128
                for c in cutlass.range_constexpr(8):
                    dm_utils.st_shared_v4_b32(
                        dm_prow + ((Int32(c) ^ (tidx_row & 7)) << 4),
                        dm_utils.bitcast_f32_i32(tSrP_r2t_f32[4 * c]), dm_utils.bitcast_f32_i32(tSrP_r2t_f32[4 * c + 1]),
                        dm_utils.bitcast_f32_i32(tSrP_r2t_f32[4 * c + 2]), dm_utils.bitcast_f32_i32(tSrP_r2t_f32[4 * c + 3]))
                cute.arch.fence_view_async_shared()
            for i in cutlass.range_constexpr(0 if const_expr(self.dm_sspv) else cute.size(tStP_r2t.shape[2])):
                cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
                if const_expr(self.split_P_arrive > 0):
                    split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
                    if const_expr(i + 1 == split_P_arrive_idx):
                        cute.arch.fence_view_async_tmem_store()
                        pipeline_s_p_o.consumer_release_w_index(s_p_o_index)
            cute.arch.fence_view_async_tmem_store()
            if const_expr(self.dm_opts.get("tcf", "1") != "0"):   # fwdfence: default path too (all lanes)
                dm_utils.tc_fence_before()   # SF tcgen05.st ordered before the P-full release
            if const_expr(self.split_P_arrive > 0):
                cute.arch.sync_warp()
                dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                with cute.arch.elect_one():
                    pipeline_p_lastsplit.producer_commit_w_index(s_p_o_index)
            else:
                pipeline_s_p_o.consumer_release_w_index(s_p_o_index)
            if const_expr(self.use_s_ping_pong):
                pipeline_sm_stats.producer_acquire_w_index_phase(
                    s_p_o_consumer_count & 1, 1 ^ ((s_p_o_consumer_count >> 1) & 1)
                )
            else:
                pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
            # l = l * acc_scale + sum_b w_b * sum_{j in b} p~_j   (fp32 pre-rounding values)
            dm_l = Float32(0.0) if const_expr(is_first) else softmax.row_sum[0] * acc_scale
            for j in cutlass.range_constexpr(0 if const_expr(self.dm_opts.get("nosum", "0") == "1") else 4):  # nosum: UPPER-BOUND experiment only (l wrong)
                if const_expr(self.dm_opts.get("exp", "f32") == "h2"):
                    dm_l = dm_l + dm_bsum[j] * dm_w[j]
                else:
                    ps = (tSrS_t2r[32 * j], tSrS_t2r[32 * j + 1])
                    for i in cutlass.range_constexpr(2, 32, 2):
                        ps = cute.arch.add_packed_f32x2(ps, (tSrS_t2r[32 * j + i], tSrS_t2r[32 * j + i + 1]))
                    dm_l = dm_l + (ps[0] + ps[1]) * dm_w[j]
            if const_expr(self.dm_opts.get("nosum", "0") == "1"):
                dm_l = dm_l + dm_w[0]
            softmax.row_sum[0] = dm_l
        # acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
        if const_expr(self.use_s_ping_pong):
            return s_p_o_consumer_count + 1, sm_stats_producer_phase, s0_s1_sequence_phase ^ 1
        return s_p_o_consumer_count ^ 1, sm_stats_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        pipeline_load_epi: Optional[pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        tStScales = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.qk_acc_dtype
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)

        tStScales_t2r = [thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range(self.q_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)
        dm_ck = Int32(0)  # fwdrw ss_pv: global softmax-step index k (pvdone completion #k precedes step k)
        # DM dbs: per-stage count of consumed sScale events (pairs with the softmax WG's dm_ecnt; persists across tiles)
        dm_trt = cute.make_rmem_tensor((8,), Float32)
        dm_rcnt = cute.make_rmem_tensor((2,), Int32)
        dm_rcnt[0] = Int32(0)
        dm_rcnt[1] = Int32(0)
        if const_expr(self.use_s_ping_pong):
            o_acc_consumer_count = Int32(0)
        load_epi_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            # DM sm100-mx: row_max is in log2 units, row_sum carries the SFA weights, V scales live in SFB.
            v_descale = Float32(1.0)
            softmax_scale_log2_eff = Float32(1.0)
            max_offset = Float32(0.0)
            max_offset_scale = Float32(1.0)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            # Keep the dynamic num_splits for the block-sparse helper below.
            num_splits_dyn = num_splits
            if const_expr(self.is_split_kv and block_info.pack_split_idx):
                num_splits_dyn = split_idx >> 16
                split_idx = split_idx & 0xFFFF

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = None
            if const_expr(self.use_tma_O or not self.pack_gqa):
                tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.epi_head_dim_v)
                gO = cute.local_tile(mO_cur, tiler_gO, (work_m, 0))  # (128 * 2, 128)
                gO = layout_utils.select(
                    cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                )  # (128, 128, 2)
                gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]

            # Default LSE to -inf for invalid split_idx tiles
            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None, True)] * self.q_stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    work_m,
                    split_idx,
                    num_splits_dyn,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor,
                    seqlen_info=seqlen,
                    kv_subtile_factor=self.kv_subtile_factor,
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = self.process_work_tile(block_info, seqlen, n_block_min, n_block_max)

            if has_work:
                # Ignore first signal from softmax as no correction is required
                # pipeline_sm_stats.consumer_wait_w_index_phase(0, sm_stats_consumer_phase)
                if const_expr(self.dm_dbs):
                    # dbs: consume the initial (data-less) event of each stage: wait full, release empty
                    dm_r = dm_rcnt[0]
                    self._dm_wait(self._dm_sb(0, 0, dm_r & 1, warp_idx), (dm_r >> 1) & 1)
                    dm_utils.mbar_arrive(self._dm_sb(1, 0, dm_r & 1, warp_idx))
                    dm_rcnt[0] = dm_r + 1
                    pipeline_sm_stats.consumer_release_w_index(0)
                    dm_r = dm_rcnt[1]
                    self._dm_wait(self._dm_sb(0, 1, dm_r & 1, warp_idx), (dm_r >> 1) & 1)
                    dm_utils.mbar_arrive(self._dm_sb(1, 1, dm_r & 1, warp_idx))
                    dm_rcnt[1] = dm_r + 1
                else:
                    if const_expr(not self.use_s_ping_pong):
                        sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                        pipeline_sm_stats.consumer_release_w_index(0)
                    if const_expr(self.q_stage == 2):
                        # pipeline_sm_stats.consumer_wait_w_index_phase(1, sm_stats_consumer_phase)
                        sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                tSrScale_t2r = cute.make_rmem_tensor(tSrScale_t2r_shape, Float32)
                if const_expr(self.dm_sspv):
                    dm_ck += 1  # the work tile's first step needs no correction
                if const_expr(self.dm_trc > 0):
                    dm_trt[0] = Float32(0.0)
                    dm_trt[1] = Float32(0.0)
                    dm_trt[3] = Float32(0.0)
                    dm_trt[4] = Float32(0.0)
                    dm_trt[5] = Float32(0.0)
                    dm_trt[6] = Float32(0.0)
                    dm_trt[7] = Float32(0.0)
                    dm_trt0 = dm_utils.clock64()
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        if const_expr(self.use_s_ping_pong):
                            sm_stats_index = o_acc_consumer_count & 1
                        else:
                            sm_stats_index = stage
                        if const_expr(self.dm_trc > 0):
                            dm_trb = dm_utils.clock64()
                        # wait for S0 / S1
                        # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                        if const_expr(self.dm_dbs):
                            dm_r = dm_rcnt[stage]
                            self._dm_wait(self._dm_sb(0, stage, dm_r & 1, warp_idx), (dm_r >> 1) & 1)
                            scale = sScale[tidx + stage * self.m_block_size
                                           + (dm_r & 1) * (2 * self.s_p_o_stage * self.m_block_size)]
                            dm_utils.mbar_arrive(self._dm_sb(1, stage, dm_r & 1, warp_idx))   # slot free for event r+2
                            dm_rcnt[stage] = dm_r + 1
                        else:
                            if const_expr(self.dm_trc > 0):
                                dm_tra = dm_utils.clock64()
                            sm_stats_barrier.arrive_and_wait_w_index(index=sm_stats_index * 4 + warp_idx)
                            if const_expr(self.dm_trc > 0):
                                dm_trc_s = dm_utils.clock64()
                                dm_trt[0] = dm_trt[0] + Float32(dm_trc_s - dm_tra)
                            # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                            # cute.arch.fence_view_async_tmem_load()
                            # scale = tSrScale_t2r[0]
                            scale = sScale[tidx + sm_stats_index * self.m_block_size]
                            if const_expr(self.dm_sspv and self.dm_opts.get("nsr", "0") == "1"):
                                scale = Float32(1.0)   # DEBUG ONLY (wrong when a rescale is due): no sScale read
                            if const_expr(self.dm_crd):
                                dm_utils.mbar_arrive(self.dm_cread + 8 * sm_stats_index)   # read done -> softmax may overwrite
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        if const_expr(self.dm_trc > 0 and not self.dm_dbs):
                            dm_trt[6] = dm_trt[6] + Float32(dm_utils.clock64() - dm_trc_s)
                        # should_rescale = True
                        # if tidx == 0: cute.printf("Correction scale i = %d, for stage %d: %f, should_rescale = %d\n", i, stage, scale, should_rescale)
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                        if const_expr(self.dm_sspv and self.dm_opts.get("cwu", "1") == "1"):
                            # ss_pv: wait for PV_s(k-1) UNCONDITIONALLY (also when this warp skips the rescale). It
                            # orders the rescale after PV_s(k-1) and gates this warp's "O rescaled" arrival on
                            # s_p_o.empty[s] until phase k-1 completed (PV_s(k-1) issued => phase k-1 done). Without
                            # it a non-rescaling correction warp can arrive twice in one phase and complete it before
                            # a slow softmax warp released its P/SF rows.
                            if const_expr(self.dm_trc > 0):
                                dm_tra = dm_utils.clock64()
                            dm_utils.mbar_wait_parity(self.dm_pvdone + 8 * stage, dm_ck & 1)
                            if const_expr(self.dm_trc > 0):
                                dm_trt[1] = dm_trt[1] + Float32(dm_utils.clock64() - dm_tra)
                            if const_expr(self.dm_opts.get("tcf", "1") == "1"):
                                dm_utils.tc_fence_after()   # O rescale (tcgen05.ld/st) after PV_s(k-1) completes
                        if should_rescale:
                            if const_expr(self.dm_sspv and self.dm_opts.get("cwu", "1") != "1"
                                          and self.dm_opts.get("nocw", "0") != "1"):
                                # (debug cwu=0: the former conditional wait)
                                dm_utils.mbar_wait_parity(self.dm_pvdone + 8 * stage, dm_ck & 1)
                                if const_expr(self.dm_opts.get("tcf", "1") == "1"):
                                    dm_utils.tc_fence_after()
                            if const_expr(self.use_s_ping_pong):
                                # Ping-pong issues QK(i+1) before PV(i), so S_i+1 does not imply O_i is done
                                pipeline_o_acc.consumer_wait_w_index_phase(
                                    o_acc_consumer_count & 1, (o_acc_consumer_count >> 1) & 1
                                )
                            if const_expr(self.dm_trc > 0):
                                dm_tra = dm_utils.clock64()
                            self.correction_rescale(thr_mma_pv, tOtO[None, None, None, stage], tidx, scale)
                            if const_expr(self.dm_trc > 0):
                                dm_trt[3] = dm_trt[3] + Float32(dm_utils.clock64() - dm_tra)
                                dm_trt[4] = dm_trt[4] + Float32(1.0)
                        # Notify mma warp that O has been rescaled
                        if const_expr(self.use_s_ping_pong):
                            pipeline_s_p_o.consumer_release_w_index((o_acc_consumer_count + 1) & 1)
                            pipeline_sm_stats.consumer_release_w_index(sm_stats_index)
                            o_acc_consumer_count += 1
                        else:
                            if const_expr(self.dm_trc > 0):
                                dm_trr = dm_utils.clock64()
                            if const_expr(self.dm_opts.get("tcf", "1") != "0"):
                                cute.arch.fence_view_async_tmem_load()   # fwdfence: O rescale loads complete (formal)
                                dm_utils.tc_fence_before()   # O rescale tcgen05.st ordered before the release
                            pipeline_s_p_o.consumer_release_w_index(stage)
                            pipeline_sm_stats.consumer_release_w_index(self.q_stage - 1 - stage)
                            if const_expr(self.dm_trc > 0):
                                dm_tre = dm_utils.clock64()
                                dm_trt[7] = dm_trt[7] + Float32(dm_tre - dm_trr)
                                dm_trt[5] = dm_trt[5] + Float32(dm_tre - dm_trb)
                    sm_stats_consumer_phase ^= 1
                    if const_expr(self.dm_sspv):
                        dm_ck += 1
                    # o_corr_consumer_phase ^= 1
                if const_expr(self.q_stage == 2):
                    pipeline_sm_stats.consumer_release_w_index(1)
                if const_expr(self.dm_trc > 0):
                    dm_trt[2] = Float32(dm_utils.clock64() - dm_trt0)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        learnable_sink_val = [Float32(learnable_sink[head_idx])] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            packed_row = (
                                self._q_block(work_m, stage, mma_tile_coord_v) * self.m_block_size + tidx
                            )
                            learnable_sink_val[stage] = load_learnable_sink(
                                learnable_sink, head_idx, packed_row, self.qhead_per_kvhead, self.pack_gqa
                            )
                for stage in cutlass.range_constexpr(self.q_stage):
                    if const_expr(self.use_s_ping_pong):
                        sm_stats_index = o_acc_consumer_count & 1
                    else:
                        sm_stats_index = stage
                    # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                    if const_expr(self.dm_dbs):
                        dm_r = dm_rcnt[stage]
                        self._dm_wait(self._dm_sb(0, stage, dm_r & 1, warp_idx), (dm_r >> 1) & 1)
                        dm_so = (dm_r & 1) * (2 * self.s_p_o_stage * self.m_block_size)
                        row_sum = sScale[tidx + stage * self.m_block_size + dm_so]
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            row_max = sScale[tidx + stage * self.m_block_size + self.s_p_o_stage * self.m_block_size + dm_so]
                        else:
                            row_max = None
                        dm_utils.mbar_arrive(self._dm_sb(1, stage, dm_r & 1, warp_idx))
                        dm_rcnt[stage] = dm_r + 1
                    else:
                        sm_stats_barrier.arrive_and_wait_w_index(index=sm_stats_index * 4 + warp_idx)
                        # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                        # cute.arch.fence_view_async_tmem_load()
                        # scale = tSrScale_t2r[0]
                        row_sum = sScale[tidx + sm_stats_index * self.m_block_size]
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            row_max = sScale[tidx + sm_stats_index * self.m_block_size + self.s_p_o_stage * self.m_block_size]
                        else:
                            row_max = None
                        if const_expr(self.dm_crd):
                            dm_utils.mbar_arrive(self.dm_cread + 8 * sm_stats_index)
                    pipeline_sm_stats.consumer_release_w_index(sm_stats_index)
                    if const_expr(learnable_sink is not None):
                        # Only the first split owns the sink column; empty rows occur with splitKV.
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            row_max, row_sum = apply_learnable_sink(
                                row_max,
                                row_sum,
                                learnable_sink_val[stage],
                                softmax_scale_log2_eff,
                                max_offset=max_offset,
                                empty_row_sum=max_offset_scale,
                            )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    scale = scale * v_descale
                    # Wait for the last O to be ready from the MMA warp
                    if const_expr(self.use_s_ping_pong):
                        pipeline_o_acc.consumer_wait_w_index_phase(
                            o_acc_consumer_count & 1, (o_acc_consumer_count >> 1) & 1
                        )
                    else:
                        pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                        if const_expr(self.dm_sspv and self.dm_opts.get("tcf", "1") == "1"):
                            dm_utils.tc_fence_after()   # last PV_s commit -> epilogue O tcgen05.ld
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                    gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        work_m,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO_stage,
                        gmem_tiled_copy_O,
                        # TMA KV loads a dummy tile when seqlen_k == 0 (see correction_epilogue)
                        zero_fill=seqlen.seqlen_k == 0 if const_expr(self.use_tma_KV) else False,
                    )
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    if const_expr(self.use_s_ping_pong):
                        pipeline_s_p_o.consumer_release_w_index((o_acc_consumer_count + 1) & 1)
                        o_acc_consumer_count += 1
                    else:
                        if const_expr(self.dm_opts.get("tcf", "1") != "0"):   # fwdfence: default path too
                            # epilogue O tcgen05.ld complete + ordered before the release that lets the next
                            # work tile's PV_s (zero-init) overwrite O_s
                            cute.arch.fence_view_async_tmem_load()
                            dm_utils.tc_fence_before()
                        pipeline_s_p_o.consumer_release_w_index(stage)
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_commit_w_index(stage)
                    # if tidx == 0: cute.printf("Correction final scale for stage %d: %f\n", stage, scale)

                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                if const_expr(self.use_block_sparsity):
                    (
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        work_m,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtO,
                        sO,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        pipeline_o_epi,
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2_eff,
                        max_offset,
                        max_offset_scale,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx, split_idx])
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    q_block = self._q_block(work_m, stage, mma_tile_coord_v)
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    # if tidx == 0 and stage <= 1:
                    #     cute.printf("row_sum = {}, row_max = {}, acc_O_mn_row_is_zero_or_nan = {}\n", row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2_eff + (cute.math.log2(row_sum, fastmath=True) - max_offset)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if const_expr(self.dm_trc > 0):
                        lse = dm_trt[self.dm_trc - 1]
                    seqlen_q = (
                        seqlen.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen.seqlen_q * self.qhead_per_kvhead
                    )
                    if const_expr(not self.pack_gqa or self.m_block_size % self.qhead_per_kvhead == 0):
                        gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (q_block,))
                        if tidx < seqlen_q - q_block * self.m_block_size:
                            # This actually just works with PackGQA too
                            gLSE[tidx] = lse
                    else:
                        idx = q_block * self.m_block_size + tidx
                        if idx < seqlen_q:
                            m_idx = idx // self.qhead_per_kvhead
                            h_idx = idx - m_idx * self.qhead_per_kvhead
                            lse_ptr_i64 = utils.elem_pointer(mLSE_cur, ((h_idx, m_idx),)).toint()
                            lse_gmem_ptr = cute.make_ptr(
                                mLSE_cur.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
                            )
                            cute.make_tensor(lse_gmem_ptr, (1,))[0] = lse

            if const_expr(pipeline_load_epi is not None and self.use_correction_warps_for_epi):
                pipeline_load_epi.producer_acquire(load_epi_producer_state)
                with cute.arch.elect_one():
                    pipeline_load_epi.producer_commit(load_epi_producer_state)
                load_epi_producer_state.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.pv_acc_dtype
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_rmem_tensor((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        work_m: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
        zero_fill: bool | Boolean = False,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 32 if const_expr(self.chunked_split_epi) else 8 * 32 // self.o_dtype.width
        # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        num_frg_per_chunk = self.epi_head_dim_v // corr_tile_size
        for chunk in cutlass.range_constexpr(self.head_dim_v_padded // self.epi_head_dim_v):
            if const_expr(chunk > 0):
                # All correction warps must finish reading sO before it is reused.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                    number_of_threads=self.num_epilogue_threads,
                )
                tOtO_t2r = cute.domain_offset((0, 0, 0, num_frg_per_chunk), tOtO_t2r)
                mO_cur = cute.domain_offset((0, self.epi_head_dim_v), mO_cur)
                if const_expr(gO is not None):
                    gO = cute.domain_offset((0, self.epi_head_dim_v), gO)
            for i in cutlass.range(num_frg_per_chunk, unroll_full=True):
                tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
                tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
                tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
                if const_expr(zero_fill is True):
                    # Empty tile: O accumulator was never written, so write zeros directly
                    # rather than scaling whatever the TMEM columns happen to hold.
                    tOrO_frg.fill(0.0)
                elif zero_fill:
                    # Run time (seqlen_k == 0 with TMA KV): the fully masked dummy block still loaded
                    # a KV tile, and P = 0 times a NaN V row is NaN; scaling by 0 would keep it.
                    tOrO_frg.fill(0.0)
                else:
                    cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
                    for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                        tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                            (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                        )
                copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
            cute.arch.fence_view_async_shared()

            if const_expr(self.use_correction_warps_for_epi):
                assert(not self.use_tma_O)
                assert(gmem_tiled_copy_O is not None)
                cute.arch.barrier(barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                                  number_of_threads=self.num_epilogue_threads)
                mma_tile_coord_v = thr_mma.thr_idx
                q_block = self._q_block(work_m, stage, mma_tile_coord_v)
                self._store_O_to_gmem(
                    sO, gO, mO_cur, gmem_tiled_copy_O, tidx, seqlen_q, q_block
                )

    @cute.jit
    def _store_O_to_gmem(
        self,
        sO_stage: cute.Tensor,
        gO: Optional[cute.Tensor],
        mO_cur: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        seqlen_q: Int32,
        q_block: Int32,
    ):
        """Copy a single stage of O from smem to gmem via registers."""
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO_stage)
        cO = cute.make_identity_tensor(self.epi_tile)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = copy_utils.predicate_k(tOcO, limit=mO_cur.shape[1])
        pack_gqa = PackGQA(
            self.m_block_size,
            self.epi_head_dim_v,
            self.check_hdim_v_oob,
            self.qhead_per_kvhead,
        )

        # load acc O from smem to rmem for wider vectorization
        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        cute.autovec_copy(tOsO, tOrO)
        # copy acc O from rmem to gmem
        if const_expr(not self.pack_gqa):
            assert gO is not None
            tOgO = gmem_thr_copy_O.partition_D(gO)
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0] < seqlen_q - q_block * self.m_block_size - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            pack_gqa.store_O(
                mO_cur, tOrO, gmem_tiled_copy_O, tidx, q_block, seqlen_q
            )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_o_epi: pipeline.PipelineAsync,
        pipeline_load_epi: Optional[pipeline.PipelineAsync],
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        mma_tile_coord_v: Int32 = 0,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        epi_consumer_phase = Int32(0)
        load_epi_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits,
            )
            if const_expr(self.is_split_kv and block_info.pack_split_idx):
                split_idx = split_idx & 0xFFFF

            if const_expr(self.use_block_sparsity) or self.process_work_tile(
                block_info, seqlen, n_block_min, n_block_max
            ):
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                gO = None
                if const_expr(self.use_tma_O or not self.pack_gqa):
                    tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                    gO = cute.local_tile(mO_cur, tiler_gO, (work_m, 0))  # (128 * 2, 128)
                    gO = layout_utils.select(
                        cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                    )  # (128, 128, 2)
                    gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]

                if const_expr(self.use_tma_O):
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    for stage in cutlass.range(self.q_stage, unroll_full=True):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                        # 2. copy O0 / O1 to gmem
                        store_O(src_idx=stage, dst_idx=stage)
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Ensure O0 / O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(self.q_stage - 1 - stage, read=True)
                        pipeline_o_epi.consumer_release_w_index(stage)
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                        # 2. copy O0 / O1 to gmem
                        q_block = self._q_block(work_m, stage, mma_tile_coord_v)
                        gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
                        self._store_O_to_gmem(
                            sO[None, None, stage], gO_stage, mO_cur, gmem_tiled_copy_O,
                            tidx, seqlen.seqlen_q, q_block,
                        )
                        pipeline_o_epi.consumer_release_w_index(stage)

                epi_consumer_phase ^= 1

            if const_expr(pipeline_load_epi is not None):
                pipeline_load_epi.producer_acquire(load_epi_producer_state)
                with cute.arch.elect_one():
                    pipeline_load_epi.producer_commit(load_epi_producer_state)
                load_epi_producer_state.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            if cute.arch.thread_idx()[0] == self.scheduler_warp_id * cute.arch.WARP_SIZE:
                prefix_str = "[CLC] query " if const_expr(self.use_clc_scheduler) else "[DYNAMIC] info "
                fa_printf(
                    3,
                    prefix_str + "sm={} cta={} (m_blk={},h={},b={},s={}) valid={}\n",
                    smid(),
                    cute.arch.block_idx()[0],
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                    work_tile.tile_idx[3],
                    work_tile.is_valid_tile,
                )
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))

    def load_Q_non_tma(
        self,
        mQ: cute.Tensor,
        sQ: cute.Tensor,
        gmem_tiled_copy_Q: cute.TiledCopy,
        pipeline_q: pipeline.PipelineAsync,
        tidx: Int32,
        seqlen_q: Int32,
        work_m: Int32,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        assert self.cta_group_size == 1, "cta_group_size must be 1 for non-tma Q load"
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_padded,
            self.check_hdim_oob,
            self.qhead_per_kvhead,
        )
        sQ_stage = sQ[None, None, None, stage]
        sQ_pi = cute.make_tensor(
            sQ_stage.iterator,
            cute.make_layout(
                (sQ_stage.shape[0][0], (sQ_stage.shape[0][1], sQ_stage.shape[2])),
                stride=(sQ_stage.stride[0][0], (sQ_stage.stride[0][1], sQ_stage.stride[2])),
            ),
        )
        pack_gqa.load_Q(mQ, sQ_pi, gmem_tiled_copy_Q, tidx, self._q_block(work_m, block), seqlen_q)
        cute.arch.cp_async_commit_group()
        pipeline_q.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        extra_tx_count: Optional[Int32] = None,
        tma_desc_ptr: Optional[cute.Pointer] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        extra_tx_count_kv = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        extra_tx_count = (
            extra_tx_count_kv + (extra_tx_count if extra_tx_count is not None else 0) if const_expr(self.use_tma_KV)
            else None
        )
        extra_kwargs = {"extra_tx_count": extra_tx_count} if const_expr(self.use_tma_KV) else {}
        pipeline_kv.producer_acquire(producer_state, **extra_kwargs)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                pipeline_kv.sync_object_empty.wait(1, phase)

        if const_expr(self.use_tma_KV):
            assert tXgX is not None and tXsX is not None and tma_atom is not None
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            # Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
            tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
            desc_kwargs = {"tma_desc_ptr": tma_desc_ptr} if const_expr(tma_desc_ptr is not None) else {}
            cute.copy(
                tma_atom, tXgX_cur, tXsX_cur,
                tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state), **desc_kwargs,
            )
        else:
            assert paged_kv_manager is not None
            assert extra_tx_count is None
            sX_cur = sX[None, None, None, stage]
            if const_expr(self.uneven_kv_smem):
                sX_cur = self.offset_kv_smem(sX_cur, stage, phase ^ 1)
            paged_kv_manager.load_KV(block, sX_cur, K_or_V)
            cute.arch.cp_async_commit_group()
            pipeline_kv.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            # Hint that the offset is 128-bit aligned so that
            # ptr + offset preserves the alignment needed by cp.async.
            offset = cute.assume(offset, divby=128 // self.k_dtype.width)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
        tScS = thr_mma_qk.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            assert head_divmod is not None
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical, head_offset = divmod(q_physical, head_divmod)
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_data.tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = divmod(q_idx_logical, seqlen_q_divmod)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1,
        )

    @cute.jit
    def process_work_tile(
        self,
        block_info: BlockInfo,
        seqlen_info: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
    ):
        is_varlen_q = seqlen_info.has_cu_seqlens_q or seqlen_info.has_seqused_q
        process_work_tile_k = block_info.has_kv_work(n_block_min, n_block_max)
        if const_expr(is_varlen_q and not self.use_varlen_scheduler):
            process_work_tile_q = seqlen_info.seqlen_q > 0
        else:
            process_work_tile_q = True
        return process_work_tile_k and process_work_tile_q
