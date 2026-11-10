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
from cutlass import Float32, Float64, Int32, Int64, Boolean, const_expr
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
from delta_matching.sm100 import dm_utils, dm_common, dm_common_hd256 as dmh
import flash_attn.cute.pipeline as pipeline_custom
import cutlass.pipeline as cutlass_pipeline
from flash_attn.cute.mask import AttentionMask
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


def _r3m(variant: str) -> int:
    """'r3_<m>' exp variants -> m (rotated 32x32-block MUFU/deg-3 mix); 0 otherwise."""
    base = variant.split("+")[0]
    return int(base.split("_")[1]) if base.startswith("r3_") else 0


def work_q_block(work_m, stage, cta_rank, *, q_stage: int, cta_group_size: int):
    """R-row block index for one CTA and Q stage of a work tile.

    See NOTE [SM100 forward coordinate spaces].
    """
    return (work_m * q_stage + stage) * cta_group_size + cta_rank


from flash_attn.cute.flash_fwd_sm100 import DescaleTensors  # same class as interface.py builds


class DMPass0Hd256Sm100:
    dm_exp_variant = "mufu"  # dm_common contract (proposal variants: "f16x2", "poly")
    """hd256 MXFP8 (sm100-mx at D=256, preset sm100-mx-d256) matched-delta pass 0 (P1 = 1-CTA, q_stage 2,
    kv_stage 2, +tl only). Copied from dm_pass0_sm100.DMPass0Sm100 (hd128); changes vs hd128: dP TMEM slots placed
    by n_block (256 + 128*i), no sO buffer, head_dim 256 (S / dP = 8 K=32 f8f6f4 steps, increasing d). Numerics are the
    hd128 sm100-mx ones verbatim: Q / K / V / dO pow2 (UE8M0) scale per 32 tokens over all D, so the scales sit on the
    MMAs' non-contraction dims and multiply the fp32 accumulator after the plain f8f6f4 MMA (exactly as hd128 pass 0; no
    SF TMEM); P / c / dP-scale from dm_common_hd256.dm_mx256_* (= dm_common.dm_bwd_*); delta = num / den with per-lane
    TwoSum cross-tile sums and one fp64 combine (+dnd: the only hd256 delta form).
    --- hd128 docstring follows ---
    Matched-delta pass 0, forked from DMForwardSm100: per KV tile S = Q8 K8^T and dP = dO8 V8^T
    (both kind::f8f6f4, V loaded K-major like K, dP in the forward's O TMEM slots); softmax warps form
    P = exp2(fma(S, c, -lse*log2e)), num += P*(dP*a), den += P; delta = num / den written to mLSE (B, H, N).
    aux_data.tensors = [qs, ks, vs, dos, dO8 (uint8 view, (B,N,H,D)), lse (B,H,N) natural].
    --- original forward docstring follows ---
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
        # P0 '+sq': sequence the two softmax WGs' exp phases (WG0 exps while WG1 does its dP phase and vice versa)
        self.s0_s1_barrier = "sq" in self.dm_exp_variant.split("+")[1:] or "sqh" in self.dm_exp_variant.split("+")[1:]
        # +sqh (h2 path): the two softmax WGs take turns on the exp halves (s0/s1 sequence barrier); +sq alone (h2) =
        # per-stage specialized code only
        self.dm_sqh = "sqh" in self.dm_exp_variant.split("+")[1:]
        # +c3: the (otherwise idle) correction warpgroup computes column chunk 3 of BOTH q stages; the softmax
        # warpgroups keep chunks 0-2 -> 3 compute warps per SMSP. Partials meet in smem (sScale) per work tile.
        self.dm_c3 = "c3" in self.dm_exp_variant.split("+")[1:]
        # +ma: multi-accumulator den/num with a factored per chunk, causal kernels only (the non-causal build spills
        # 35-51 regs with it at any softmax budget 208-224 and gets ~5% slower); +maa: always
        _fl0 = self.dm_exp_variant.split("+")[1:]
        self.dm_ma = ("ma" in _fl0 and self.is_causal) or "maa" in _fl0
        self.dm_b6 = next((int(t[3:]) for t in _fl0 if t.startswith("b6_")), 0)   # deg-6 poly exp on (qb+kb)%m==0 blocks
        self.dm_h2 = "h2" in _fl0   # two 64-column halves: S half -> exp/den -> (dP wait) dP half -> num
        self.dm_pf = "pf" in _fl0   # prefetch the next n_block's raw K/V scales one iteration ahead
        self.dm_tr = "tr" in _fl0   # per-segment clock trace of the softmax loop into aux[6] (int64[64])
        # +f64: per-tile fp32 partial sums (short chains), accumulated across tiles in fp64; delta = num / den in fp64,
        # rounded once to fp32. P and dP are untouched (bitwise the backward's), only the row-sum round-off shrinks.
        self.dm_f64 = "f64" in _fl0 or "f64x" in _fl0 or "f64y" in _fl0 or any(f.startswith("f64k") for f in _fl0)
        # +f64k<K>: flush the fp32 partials into the fp64 sums only every K tiles (and on the last tile)
        self.dm_f64k = next((int(f[4:]) for f in _fl0 if f.startswith("f64k")), 1)
        # +f64x: exact fp32->fp64 conversion on the ALU/FP64 pipes (dm_utils.f32_to_f64_exact) instead of F2F, which
        # issues on the XU pipe shared with MUFU ex2 (pass 0 is MUFU-bound): bitwise the same values as Float64(x)
        self.dm_f64x = "f64x" in _fl0
        self.dm_f64y = "f64y" in _fl0   # ALU bit path + rare F2F branch for subnormals (bitwise = Float64(x))
        # +kh: per-tile fp32 partials (as +f64) accumulated across tiles with Kahan compensation in fp32 (no F2F /
        # FP64 on the per-tile path); the compensated sums are combined in fp64 once at the end: delta = (s - c) / (s' - c')
        self.dm_kh = "kh" in _fl0
        # +ts: per-lane error-free TwoSum (double-float) cross-tile accumulation of the fp32 per-tile partials (packed f32x2):
        # (sum, err) pairs carry ~2x fp32 precision, then all four terms are combined in fp64 once at the end
        self.dm_ts = "ts" in _fl0
        self.dm_nd = "nd" in _fl0   # no denominator: sum_j P := 1 (P is normalized by the forward LSE)
        # +ni: the correction WG and non-scheduler empty warps have no per-tile work in pass 0: they leave the
        # dynamic-scheduler consumer group and skip the schedule walk (no CLC-response spin stealing issue slots)
        self.dm_ni = "ni" in _fl0
        # +es: tl_loop loads both S halves and releases the S slot before the exps (profiled: the S-slot
        # release waited on dP(n), stalling the MMA warp's S(n+1)); bitwise the same P / dP / num / den / delta
        self.dm_es = "es" in _fl0
        self.dm_es2 = "es2" in _fl0   # S half 1 loaded + S released after the h0 exps, before the dP wait
        assert not (self.dm_es and self.dm_es2)
        # +ea (E2, needs +es): all 64 exps + den chains (same order) before the dP wait, then the 4 num chunks j = 0..3
        self.dm_ea = "ea" in _fl0
        assert not self.dm_ea or self.dm_es
        # +tl: two lanes per query row (tcgen05.ld.16x64b), 4 compute warps / SMSP: warps 0-3 / 4-7 = stage 0 / 1
        # rows 0-15 of each quadrant, warps 8-11 / 12-15 (the "correction" ids) = stage 0 / 1 rows 16-31; MMA / epilogue /
        # load / empty move to 16-19 (640 threads). Bitwise the h2+ts path (see tl_loop).
        self.dm_tl = "tl" in _fl0 and self.q_stage == 2 and self.use_tma_KV   # else: the default h2 path
        if self.dm_tl:
            assert self.dm_h2 and self.dm_ts and not self.dm_c3 and self.dm_b6 == 0 and _r3m(self.dm_exp_variant) == 0
            assert self.dm_exp_variant.split("+")[0] in ("p3_16_0", "mufu"), "tl: all-MUFU P only"
        # hd256 MXFP8: delta = num / den (+dnd) only
        self.dm_delta_numden = "dnd" in _fl0
        assert self.dm_delta_numden, "hd256 MXFP8 pass 0: delta form num/den (+dnd) only"
        assert head_dim == 256 and head_dim_v == 256, "DMPass0Hd256Sm100 is the head_dim 256 pass 0"
        assert self.dm_tl and not self.use_2cta_instrs and self.q_stage == 2 and self.n_block_size == 128 \
            and self.m_block_size == 128, "hd256 P1: 1-CTA, q_stage 2, 128x128 tiles, +tl"
        if self.dm_c3:
            _fl = self.dm_exp_variant.split("+")[1:]
            assert not any(f in _fl for f in ("fu", "fa", "sq")), "c3 supports the plain static-mix path only"
            assert _r3m(self.dm_exp_variant) == 0 and not self.dm_exp_variant.startswith(("b3_", "mix"))

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
        if self.dm_tl:
            self.correction_warp_ids = (8, 9, 10, 11, 12, 13, 14, 15)
            self.mma_warp_id = 16
            self.epilogue_warp_ids = (17,)
            self.load_warp_ids = (18,)
            self.empty_warp_ids = (19,)
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
        # config.select_sm100_registers and its validation assume this launch shape.
        assert self.threads_per_cta == (640 if self.dm_tl else SM100_THREADS_PER_CTA)

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
        # hd256 P1: dP slot i at 256 + 128 * i (placed by n_block, not by head_dim_v): S0 0, S1 128, dP0 256, dP1 384
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.n_block_size
            for i in range(self.q_stage)
        ]  # 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.n_block_size
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # DM sm100-mx: SFA (4 cols) + SFB (4 cols) right after the fp8 P columns of each S stage
        self.dm_sf_col = self.tmem_s_to_p_offset + self.n_block_size // 4
        self.dm_rescale_T = 32.0
        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        self.num_regs_softmax = config.registers.softmax
        self.num_regs_correction = config.registers.correction
        self.num_regs_other = config.registers.other

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
        # P0: dO codes (uint8 aux tensor) viewed as fp8, same layout handling as Q; lse input
        mdO_u8 = aux_data.tensors[4]
        mdO = cute.make_tensor(
            cute.recast_ptr(mdO_u8.iterator, dtype=self.q_dtype), cute.select(mdO_u8.layout, mode=Q_layout_transpose)
        )
        mdO = assume_tensor_aligned(mdO)
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
        # P0: V stays (s, d, h, b) and is loaded K-major exactly like K

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
        sV_layout = sK_layout  # P0: V is a K-major B operand of dP = dO V^T
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
        self.tma_copy_bytes["Q"] *= 2  # P0: Q and dO tiles land on the same pipeline_q stage
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
            tma_atom_dO, mdO = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mdO,
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
            # TMA load for V (P0: K-major, same as K)
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
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
            mQ.shape[1],  # P0: V is not transposed; head dim
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

        # hd256 P1: pass 0 never writes O -> no sO buffer (a 128 KiB bf16 sO would not fit)
        sO_size = 0
        sQ_size = cute.cosize(sQ_layout)
        # K and V alias the same physical buffer and may have different extents.
        sKV_size = cutlass.max(cute.cosize(sK_layout), cute.cosize(sV_layout))

        sched_response_size = self.sched_stages * 4 if self.dynamic_persistent else 0
        sched_mbar_size = self.sched_stages * 2 if self.dynamic_persistent else 0
        load_epi_mbar_size = 2 if const_expr(self.overlap_sO_sQ) else 0

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
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_p_o_stage * self.m_block_size * 2]
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
            tma_atom_dO,
            mdO,
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
        tma_atom_dO: Optional[cute.CopyAtom] = None,
        mdO: Optional[cute.Tensor] = None,
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
        dm_utils.tcf_begin(0 if "ntcf" in self.dm_exp_variant.split("+")[1:] else 2 if "tcfb" in self.dm_exp_variant.split("+")[1:] else 1)   # tcgen05 thread-sync fences (dm_utils.TCF)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O, tma_atom_dO):
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
            consumer_group=ThreadCooperativeGroup(
                cute.arch.WARP_SIZE * len(self.softmax0_warp_ids) * self.cta_group_size * (2 if (self.dm_c3 or self.dm_tl) else 1)
            ),  # P0: softmax threads (+ c3 correction WG) release the S/dP slot
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
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(  # P0: dP slot pipeline (MMA -> softmax)
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.s_p_o_stage,
            producer_group=mma_warp,
            consumer_group=ThreadCooperativeGroup(
                cute.arch.WARP_SIZE * len(self.softmax0_warp_ids) * self.cta_group_size * (2 if (self.dm_c3 or self.dm_tl) else 1)
            ),
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
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        # hd256 P1: no sO storage; an (unused) view over sQ keeps the epilogue signatures
        sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.s_p_o_stage * self.m_block_size * 2))

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
            if const_expr(self.dm_ni and not self.dm_c3 and not self.dm_tl):
                num_sched_consumer_warps_per_cta -= len(self.correction_warp_ids) + len(self._dm_idle_empty_warps())
            elif const_expr(self.dm_ni and self.dm_tl):
                num_sched_consumer_warps_per_cta -= len(self._dm_idle_empty_warps())
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
                    if const_expr(not (self.dm_ni and not self.dm_c3)):
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
                sdO=sdO,
                tma_atom_dO=tma_atom_dO,
                mdO=mdO,
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
                sdO=sdO,
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
        if (
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
                pipeline_dp=pipeline_o_acc,
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

            if const_expr(self.dm_tl):
                if warp_idx < self.softmax1_warp_ids[0]:
                    self.tl_loop(0, 0, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_o_acc, aux_data, block_info,
                                 num_splits, SeqlenInfoCls, tile_scheduler)
                else:
                    self.tl_loop(1, 0, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_o_acc, aux_data, block_info,
                                 num_splits, SeqlenInfoCls, tile_scheduler)
            elif const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if const_expr(self.q_stage == 1) or warp_idx < self.softmax1_warp_ids[0] else 1)
                softmax_loop(stage=stage, tStS=tStS)
            elif const_expr(self.q_stage == 1 or len(self.softmax1_warp_ids) == 0):
                # single q stage (short N, e.g. N=128): no second softmax warpgroup -> only stage 0
                softmax_loop(stage=0, tStS=tStS)
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
            if const_expr(self.dm_tl):
                cute.arch.setmaxregister_increase(self.num_regs_correction)   # 640-thread launch: 96 regs
            elif const_expr(self.num_regs_correction > SM100_LAUNCH_REGISTERS):
                cute.arch.setmaxregister_increase(self.num_regs_correction)
            else:
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait_unaligned()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if const_expr(self.dm_tl):
                if warp_idx < self.correction_warp_ids[4]:
                    self.tl_loop(0, 1, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_o_acc, aux_data, block_info,
                                 num_splits, SeqlenInfoCls, tile_scheduler)
                else:
                    self.tl_loop(1, 1, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_o_acc, aux_data, block_info,
                                 num_splits, SeqlenInfoCls, tile_scheduler)
            elif const_expr(self.dm_c3):
                self.c3_loop(thr_mma_qk, tStS, sScale, aux_data, pipeline_s_p_o, pipeline_o_acc, block_info, num_splits,
                             SeqlenInfoCls, tile_scheduler)
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
        sdO: Optional[cute.Tensor] = None,
        tma_atom_dO: Optional[cute.CopyAtom] = None,
        mdO: Optional[cute.Tensor] = None,
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
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))  # P0: K-like
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
            tOgV = thr_mma_qk.partition_B(gV) if const_expr(gV is not None) else None  # P0
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
                mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
                gdO = cute.local_tile(mdO_cur, tiler_gQ, (work_m, 0))
                gdO = layout_utils.select(cute.flat_divide(gdO, (self.mma_tiler_qk[0],)), mode=[0, 2, 1])
                tSgdO = thr_mma_qk.partition_A(gdO)
                load_dO_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), tSgdO, sdO
                )
                load_Q = partial(self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase,
                                 load_dO_fn=load_dO_fn)
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
        sdO: Optional[cute.Tensor] = None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = None  # P0
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0],)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        qk_mma_idesc, pv_mma_idesc = sm100_desc.mma_op_to_idesc(qk_mma_op), sm100_desc.mma_op_to_idesc(pv_mma_op)
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
        # P0: dP_stage = dO_stage V^T into the forward's O slots (same instruction shape as S = Q K^T)
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
        gemm_Pi = None  # P0: no PV
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
        dm_slot_empty_phase = Int32(1)  # P0: S/dP slots start empty
        if const_expr(self.use_s_ping_pong):
            mma_pv_count = Int32(0)

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
        descale_tensors,
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
        pipeline_dp=None,
    ):
        """P0: per query row delta = sum_j P dP / sum_j P over all visible keys; written to mLSE (s, h, b)."""
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        aux = aux_data.tensors
        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128) S slot of this stage
        tdPAcc = cute.make_tensor(
            tSAcc.iterator + (self.tmem_o_offset[0] - self.tmem_s_offset[0]), tSAcc.layout  # same offset for both stages
        )
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)
        tdPt_t2r = thr_tmem_load.partition_S(tdPAcc)
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        n_chunks = cute.size(tStS_t2r.shape[2])
        nc = n_chunks - 1 if const_expr(self.dm_c3) else n_chunks   # c3: chunk 3 belongs to the correction WG
        s_full_phase = Int32(0)
        s0s1_phase = Int32(1 if stage == 0 else 0)

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
            row = m_block * self.m_block_size + tidx
            row_odd = (row & 1) == 1
            # r3_<m>: 32x32-block MUFU/deg-3 mix, element (q, k) uses the deg-3 poly iff ((q>>5)+(k>>5)) % m == 0.
            # Chunks are loaded rotated by rot = (-(q>>5)) mod m (warp-uniform), so the poly chunks sit in static
            # register slots (slot 0 for m=4; slots 0, 2 for m=2) -> no branches or selects.
            rot = ((_r3m(self.dm_exp_variant) - ((row >> 5) % _r3m(self.dm_exp_variant))) % _r3m(self.dm_exp_variant)
                   if const_expr(_r3m(self.dm_exp_variant) > 0) else Int32(0))
            # rows past seqlen_q (second q stage of the last tile) are never stored: clamp reads (no OOB)
            rowc = cutlass.min(row, seqlen.seqlen_q - 1)
            qsv = Float32(aux[0][batch_idx, head_idx, rowc // 32])
            dosv = Float32(aux[3][batch_idx, head_idx, rowc // 32])
            l2 = Float32(aux[5][batch_idx, head_idx, rowc]) * Float32(dm_common.LOG2E)
            if const_expr(self.dm_tr):
                tr_s = [Int32(0) for _ in range(6)]
                tr_prev = dm_utils.clock32()
            num = (Float32(0.0), Float32(0.0))
            den = (Float32(0.0), Float32(0.0))
            den1 = (Float32(0.0), Float32(0.0))
            den2 = (Float32(0.0), Float32(0.0))
            den3 = (Float32(0.0), Float32(0.0))
            if const_expr(self.dm_pf):
                nb0 = cutlass.max(n_block_max - 1, 0) * (self.n_block_size // 32)
                pks = tuple(Float32(aux[1][batch_idx, kv_head_idx, nb0 + j]) for j in range(4))
                pvs = tuple(Float32(aux[2][batch_idx, kv_head_idx, nb0 + j]) for j in range(4))
            num64 = Float64(0.0)
            den64 = Float64(0.0)
            ts_sn = (Float32(0.0), Float32(0.0))
            ts_en = (Float32(0.0), Float32(0.0))
            ts_sd = (Float32(0.0), Float32(0.0))
            ts_ed = (Float32(0.0), Float32(0.0))
            ks_n = Float32(0.0)
            kc_n = Float32(0.0)
            ks_d = Float32(0.0)
            kc_d = Float32(0.0)
            if has_work:
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    if const_expr((self.dm_f64 and self.dm_f64k == 1) or self.dm_kh or self.dm_ts):
                        # tile-local fp32 partials (reset every tile)
                        num = (Float32(0.0), Float32(0.0))
                        den = (Float32(0.0), Float32(0.0))
                        den1 = (Float32(0.0), Float32(0.0))
                        den2 = (Float32(0.0), Float32(0.0))
                        den3 = (Float32(0.0), Float32(0.0))
                    # per-chunk scales first (gmem/L2 latency overlaps the S wait)
                    cs = [None] * n_chunks
                    as_ = [None] * n_chunks
                    if const_expr(self.dm_pf):
                        # +pf: this tile's raw scales were loaded one iteration ago; issue the next tile's now
                        for j in cutlass.range_constexpr(nc):
                            cs[j] = dm_common.dm_bwd_c(pks[j], qsv)
                            as_[j] = dm_common.dm_bwd_dp_scale(pvs[j], dosv)
                        nbn = cutlass.max(n_block - 1, n_block_min) * (self.n_block_size // 32)
                        pks = tuple(Float32(aux[1][batch_idx, kv_head_idx, nbn + j]) for j in range(4))
                        pvs = tuple(Float32(aux[2][batch_idx, kv_head_idx, nbn + j]) for j in range(4))
                    else:
                        for j in cutlass.range_constexpr(nc):
                            kb = n_block * (self.n_block_size // 32) + ((j + rot) % n_chunks if const_expr(_r3m(self.dm_exp_variant) > 0) else j)
                            cs[j] = dm_common.dm_bwd_c(Float32(aux[1][batch_idx, kv_head_idx, kb]), qsv)
                            as_[j] = dm_common.dm_bwd_dp_scale(Float32(aux[2][batch_idx, kv_head_idx, kb]), dosv)
                    if const_expr(self.dm_tr):
                        tr_now = dm_utils.clock32()
                        tr_s[0] = tr_s[0] + (tr_now - tr_prev)
                        tr_prev = tr_now
                    pipeline_s_p_o.consumer_wait_w_index_phase(stage, s_full_phase)
                    if const_expr(self.dm_tr):
                        tr_now = dm_utils.clock32()
                        tr_s[1] = tr_s[1] + (tr_now - tr_prev)
                        tr_prev = tr_now
                    tSrS = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
                    if const_expr(self.dm_h2):
                        diag = False
                        if const_expr(self.is_causal):
                            diag = (n_block + 1) * self.n_block_size > m_block * self.m_block_size
                        kbase = n_block * self.n_block_size
                        for h in cutlass.range_constexpr(2):
                            if const_expr(self.dm_sqh and h == 0):
                                pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0s1_phase)
                            for j in cutlass.range_constexpr(2 * h, 2 * h + 2):
                                cute.copy(thr_tmem_load, tStS_t2r[None, None, j], tSrS[None, None, j])
                            if const_expr(h == 1):
                                cute.arch.fence_view_async_tmem_load()
                                with dm_utils.tcf_site("r0"):
                                    pipeline_s_p_o.consumer_release_w_index(stage)
                            if const_expr(self.is_causal):
                                if diag:
                                    for i in cutlass.range_constexpr(64 * h, 64 * h + 64):
                                        tSrS[i] = -Float32.inf if kbase + tScS_t2r[i][1] > row else tSrS[i]
                            for j in cutlass.range_constexpr(2 * h, 2 * h + 2):
                                if const_expr(self.dm_b6 > 0):
                                    # b6: this warp's 32 query rows x chunk j's 32 keys = one 32x32 block; deg-6 poly
                                    # iff (qb + kb) % m == 0 (warp-uniform), the same rule the backward applies
                                    if ((row >> 5) + n_block * (self.n_block_size // 32) + j) % self.dm_b6 == 0:
                                        for v in cutlass.range_constexpr(16):
                                            i0 = 32 * j + 2 * v
                                            tSrS[i0], tSrS[i0 + 1] = dm_common.dm_bwd_p_pair_poly6(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    else:
                                        for v in cutlass.range_constexpr(16):
                                            i0 = 32 * j + 2 * v
                                            tSrS[i0], tSrS[i0 + 1] = dm_common.dm_bwd_p_pair(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                for v in cutlass.range_constexpr(16):
                                    i0 = 32 * j + 2 * v
                                    if const_expr(self.dm_b6 > 0):
                                        p = (tSrS[i0], tSrS[i0 + 1])
                                    elif const_expr(dm_common.p5_uses_poly(self.dm_exp_variant.split("+")[0], v)):
                                        p = dm_common.dm_bwd_p_pair_poly(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    else:
                                        p = dm_common.dm_bwd_p_pair(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    tSrS[i0], tSrS[i0 + 1] = p
                                    if const_expr("pd" in self.dm_exp_variant.split("+")[1:]):
                                        # DEBUG: P dump, h2 (production) path
                                        if row < seqlen.seqlen_q:
                                            for e_ in cutlass.range_constexpr(2):
                                                aux[len(aux) - 2][batch_idx, head_idx, row,
                                                                  n_block * self.n_block_size + tScS_t2r[i0 + e_][1]] = tSrS[i0 + e_]
                                    if const_expr(v % 4 == 0):
                                        den = cute.arch.add_packed_f32x2(den, p)
                                    elif const_expr(v % 4 == 1):
                                        den1 = cute.arch.add_packed_f32x2(den1, p)
                                    elif const_expr(v % 4 == 2):
                                        den2 = cute.arch.add_packed_f32x2(den2, p)
                                    else:
                                        den3 = cute.arch.add_packed_f32x2(den3, p)
                            if const_expr(self.dm_sqh and h == 1):
                                pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
                                s0s1_phase ^= 1
                            if const_expr(h == 0):
                                pipeline_dp.consumer_wait_w_index_phase(stage, s_full_phase)
                            for j in cutlass.range_constexpr(2 * h, 2 * h + 2):
                                tdPr = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS)[None, None, j].shape, self.qk_acc_dtype)
                                cute.copy(thr_tmem_load, tdPt_t2r[None, None, j], tdPr)
                                if const_expr(j == 3):
                                    cute.arch.fence_view_async_tmem_load()
                                    with dm_utils.tcf_site("r1"):
                                        pipeline_dp.consumer_release_w_index(stage)
                                if const_expr("pd" in self.dm_exp_variant.split("+")[1:]):
                                    # DEBUG: raw dP accumulator dump, h2 (production) path
                                    if row < seqlen.seqlen_q:
                                        for i_ in cutlass.range_constexpr(32):
                                            aux[len(aux) - 1][batch_idx, head_idx, row,
                                                              n_block * self.n_block_size + tScS_t2r[32 * j + i_][1]] = tdPr[i_]
                                acc = [(Float32(0.0), Float32(0.0)) for _ in range(4)]
                                for v in cutlass.range_constexpr(16):
                                    acc[v % 4] = cute.arch.fma_packed_f32x2(
                                        (tSrS[32 * j + 2 * v], tSrS[32 * j + 2 * v + 1]), (tdPr[2 * v], tdPr[2 * v + 1]), acc[v % 4])
                                acc01 = cute.arch.add_packed_f32x2(acc[0], acc[1])
                                acc23 = cute.arch.add_packed_f32x2(acc[2], acc[3])
                                num = cute.arch.fma_packed_f32x2(cute.arch.add_packed_f32x2(acc01, acc23), (as_[j], as_[j]), num)
                    else:
                        pass
                        if const_expr(_r3m(self.dm_exp_variant) > 0):
                            for j in cutlass.range_constexpr(n_chunks):
                                cute.copy(thr_tmem_load, tStS_t2r[None, None, (j + rot) % n_chunks], tSrS[None, None, j])
                        elif const_expr(self.dm_c3):
                            for j in cutlass.range_constexpr(nc):
                                cute.copy(thr_tmem_load, tStS_t2r[None, None, j], tSrS[None, None, j])
                        else:
                            cute.copy(thr_tmem_load, tStS_t2r, tSrS)
                        cute.arch.fence_view_async_tmem_load()
                        with dm_utils.tcf_site("r2"):
                            pipeline_s_p_o.consumer_release_w_index(stage)  # next S can be issued now
                        if const_expr(self.is_causal):
                            # keys past this row: S = -inf -> P = 0 (same as the main backward's mask)
                            if (n_block + 1) * self.n_block_size > m_block * self.m_block_size:
                                kbase = n_block * self.n_block_size
                                if const_expr(_r3m(self.dm_exp_variant) > 0):
                                    # 32x32b TMEM load: register 32*j + w holds column 32*chunk + w
                                    for i in cutlass.range_constexpr(cute.size(tSrS)):
                                        kcol = kbase + ((i // 32 + rot) % n_chunks) * 32 + (i % 32)
                                        tSrS[i] = -Float32.inf if kcol > row else tSrS[i]
                                else:
                                    for i in cutlass.range_constexpr(32 * nc):
                                        tSrS[i] = -Float32.inf if kbase + tScS_t2r[i][1] > row else tSrS[i]
                        if const_expr("fu" in self.dm_exp_variant.split("+")[1:]):
                            # fused: dP waited up front, then per 32-column chunk exp + den + num in one basic block,
                            # so ptxas can interleave MUFU (exp) with FMA-pipe work (num, poly)
                            pipeline_dp.consumer_wait_w_index_phase(stage, s_full_phase)
                            for j in cutlass.range_constexpr(n_chunks):
                                tdPr = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS)[None, None, j].shape, self.qk_acc_dtype)
                                cute.copy(thr_tmem_load, tdPt_t2r[None, None, j], tdPr)
                                if const_expr(j == n_chunks - 1):
                                    cute.arch.fence_view_async_tmem_load()
                                    with dm_utils.tcf_site("r3"):
                                        pipeline_dp.consumer_release_w_index(stage)
                                acc0 = (Float32(0.0), Float32(0.0))
                                acc1 = (Float32(0.0), Float32(0.0))
                                if const_expr(self.dm_exp_variant.split("+")[0].startswith("b3_")):
                                    # 32x32-block MUFU/deg-3 mix, bitwise-mirrorable in the KV-stationary backward:
                                    # element (q, k) uses FA4's deg-3 polynomial iff ((q >> 5) + (k >> 5)) % m == 0.
                                    # q >> 5 is warp-uniform here (32 consecutive rows per warp), so the branch is uniform.
                                    b3_m = int(self.dm_exp_variant.split("+")[0].split("_")[1])
                                    kb32 = n_block * (self.n_block_size // 32) + j
                                    if ((row >> 5) + kb32) % b3_m == 0:
                                        for v in cutlass.range_constexpr(16):
                                            i0 = 32 * j + 2 * v
                                            tSrS[i0], tSrS[i0 + 1] = dm_common.dm_bwd_p_pair_poly(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    else:
                                        for v in cutlass.range_constexpr(16):
                                            i0 = 32 * j + 2 * v
                                            tSrS[i0], tSrS[i0 + 1] = dm_common.dm_bwd_p_pair(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                for v in cutlass.range_constexpr(16):
                                    i0 = 32 * j + 2 * v
                                    if const_expr(self.dm_exp_variant.split("+")[0].startswith("b3_")):
                                        p = (tSrS[i0], tSrS[i0 + 1])
                                    elif const_expr(self.dm_exp_variant.split("+")[0] == "mix"):
                                        p = dm_common.dm_bwd_p_pair_mix(tSrS[i0], tSrS[i0 + 1], cs[j], l2, row_odd)
                                    elif const_expr(dm_common.is_static_mix(self.dm_exp_variant.split("+")[0])):
                                        if const_expr(dm_common.p5_uses_poly(self.dm_exp_variant.split("+")[0], v)):
                                            if const_expr(self.dm_exp_variant.startswith("p5_")):
                                                p = dm_common.dm_bwd_p_pair_poly5(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                            else:
                                                p = dm_common.dm_bwd_p_pair_poly(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                        else:
                                            p = dm_common.dm_bwd_p_pair(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    else:
                                        p = dm_common.P_PAIR_VARIANTS[self.dm_exp_variant.split("+")[0]](tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    if const_expr(v % 2 == 0):
                                        den = cute.arch.add_packed_f32x2(den, p)
                                        acc0 = cute.arch.fma_packed_f32x2(p, (tdPr[2 * v], tdPr[2 * v + 1]), acc0)
                                    else:
                                        den1 = cute.arch.add_packed_f32x2(den1, p)
                                        acc1 = cute.arch.fma_packed_f32x2(p, (tdPr[2 * v], tdPr[2 * v + 1]), acc1)
                                acc0 = cute.arch.add_packed_f32x2(acc0, acc1)
                                num = cute.arch.fma_packed_f32x2(acc0, (as_[j], as_[j]), num)
                        else:
                            if const_expr(self.s0_s1_barrier):
                                pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0s1_phase)
                            # P in place
                            for j in cutlass.range_constexpr(nc):
                                if const_expr(self.dm_exp_variant.split("+")[0].startswith("b3_")):
                                    b3_m = int(self.dm_exp_variant.split("+")[0].split("_")[1])
                                    kb32 = n_block * (self.n_block_size // 32) + j
                                    if ((row >> 5) + kb32) % b3_m == 0:
                                        for v in cutlass.range_constexpr(16):
                                            i0 = 32 * j + 2 * v
                                            tSrS[i0], tSrS[i0 + 1] = dm_common.dm_bwd_p_pair_poly(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    else:
                                        for v in cutlass.range_constexpr(16):
                                            i0 = 32 * j + 2 * v
                                            tSrS[i0], tSrS[i0 + 1] = dm_common.dm_bwd_p_pair(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    for v in cutlass.range_constexpr(16):
                                        den = cute.arch.add_packed_f32x2(den, (tSrS[32 * j + 2 * v], tSrS[32 * j + 2 * v + 1]))
                                for v in cutlass.range_constexpr(16 if const_expr(not self.dm_exp_variant.split("+")[0].startswith("b3_")) else 0):
                                    i0 = 32 * j + 2 * v
                                    if const_expr(_r3m(self.dm_exp_variant) > 0):
                                        if const_expr(j % _r3m(self.dm_exp_variant) == 0):
                                            p = dm_common.dm_bwd_p_pair_poly(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                        else:
                                            p = dm_common.dm_bwd_p_pair(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    elif const_expr(self.dm_exp_variant.split("+")[0] == "mix"):
                                        p = dm_common.dm_bwd_p_pair_mix(tSrS[i0], tSrS[i0 + 1], cs[j], l2, row_odd)
                                    elif const_expr(dm_common.is_static_mix(self.dm_exp_variant.split("+")[0])):
                                        if const_expr(dm_common.p5_uses_poly(self.dm_exp_variant.split("+")[0], v)):
                                            if const_expr(self.dm_exp_variant.startswith("p5_")):
                                                p = dm_common.dm_bwd_p_pair_poly5(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                            else:
                                                p = dm_common.dm_bwd_p_pair_poly(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                        else:
                                            p = dm_common.dm_bwd_p_pair(tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    else:
                                        p = dm_common.P_PAIR_VARIANTS[self.dm_exp_variant.split("+")[0]](tSrS[i0], tSrS[i0 + 1], cs[j], l2)
                                    tSrS[i0], tSrS[i0 + 1] = p
                                    if const_expr("pd" in self.dm_exp_variant.split("+")[1:]):
                                        # DEBUG: dump this kernel's P for bitwise comparison
                                        if row < seqlen.seqlen_q:
                                            for e_ in cutlass.range_constexpr(2):
                                                aux[len(aux) - 2][batch_idx, head_idx, row,
                                                                  n_block * self.n_block_size + tScS_t2r[i0 + e_][1]] = tSrS[i0 + e_]
                                    if const_expr(self.dm_ma):
                                        # 4 independent accumulators (breaks the serial FADD2 chain)
                                        if const_expr(v % 4 == 0):
                                            den = cute.arch.add_packed_f32x2(den, p)
                                        elif const_expr(v % 4 == 1):
                                            den1 = cute.arch.add_packed_f32x2(den1, p)
                                        elif const_expr(v % 4 == 2):
                                            den2 = cute.arch.add_packed_f32x2(den2, p)
                                        else:
                                            den3 = cute.arch.add_packed_f32x2(den3, p)
                                    else:
                                        den = cute.arch.add_packed_f32x2(den, p)
                            if const_expr(self.s0_s1_barrier):
                                pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
                                s0s1_phase ^= 1
                            if const_expr(self.dm_tr):
                                tr_now = dm_utils.clock32()
                                tr_s[2] = tr_s[2] + (tr_now - tr_prev)
                                tr_prev = tr_now
                            # dP streamed in 32-column chunks; the slot is released after the last load
                            pipeline_dp.consumer_wait_w_index_phase(stage, s_full_phase)
                            if const_expr(self.dm_tr):
                                tr_now = dm_utils.clock32()
                                tr_s[3] = tr_s[3] + (tr_now - tr_prev)
                                tr_prev = tr_now
                            for j in cutlass.range_constexpr(nc):
                                tdPr = cute.make_rmem_tensor(thr_tmem_load.partition_D(tScS)[None, None, j].shape, self.qk_acc_dtype)
                                cute.copy(thr_tmem_load, tdPt_t2r[None, None, (j + rot) % n_chunks if const_expr(_r3m(self.dm_exp_variant) > 0) else j], tdPr)
                                if const_expr(j == nc - 1):
                                    cute.arch.fence_view_async_tmem_load()
                                    with dm_utils.tcf_site("r4"):
                                        pipeline_dp.consumer_release_w_index(stage)
                                if const_expr(self.dm_ma):
                                    # 4 independent sum P*dP_acc chains per 32-column chunk, then num += a * acc
                                    acc = [(Float32(0.0), Float32(0.0)) for _ in range(4)]
                                    for v in cutlass.range_constexpr(16):
                                        acc[v % 4] = cute.arch.fma_packed_f32x2(
                                            (tSrS[32 * j + 2 * v], tSrS[32 * j + 2 * v + 1]), (tdPr[2 * v], tdPr[2 * v + 1]), acc[v % 4])
                                    acc01 = cute.arch.add_packed_f32x2(acc[0], acc[1])
                                    acc23 = cute.arch.add_packed_f32x2(acc[2], acc[3])
                                    num = cute.arch.fma_packed_f32x2(cute.arch.add_packed_f32x2(acc01, acc23), (as_[j], as_[j]), num)
                                elif const_expr("fa" in self.dm_exp_variant.split("+")[1:]):
                                    # a factored out of the 32-column chunk: acc = sum P * dP_acc, num += a * acc
                                    acc = (Float32(0.0), Float32(0.0))
                                    for v in cutlass.range_constexpr(16):
                                        acc = cute.arch.fma_packed_f32x2(
                                            (tSrS[32 * j + 2 * v], tSrS[32 * j + 2 * v + 1]), (tdPr[2 * v], tdPr[2 * v + 1]), acc)
                                    num = cute.arch.fma_packed_f32x2(acc, (as_[j], as_[j]), num)
                                else:
                                    for v in cutlass.range_constexpr(16):
                                        dpv = cute.arch.mul_packed_f32x2((tdPr[2 * v], tdPr[2 * v + 1]), (as_[j], as_[j]))
                                        if const_expr("pd" in self.dm_exp_variant.split("+")[1:]):
                                            # DEBUG: raw dP accumulator dump
                                            if row < seqlen.seqlen_q:
                                                for e_ in cutlass.range_constexpr(2):
                                                    aux[len(aux) - 1][batch_idx, head_idx, row,
                                                                      n_block * self.n_block_size + tScS_t2r[32 * j + 2 * v + e_][1]] = tdPr[2 * v + e_]
                                        num = cute.arch.fma_packed_f32x2((tSrS[32 * j + 2 * v], tSrS[32 * j + 2 * v + 1]), dpv, num)
                    if const_expr(self.dm_tr):
                        tr_now = dm_utils.clock32()
                        tr_s[4] = tr_s[4] + (tr_now - tr_prev)
                        tr_prev = tr_now
                    if const_expr(self.dm_tr):
                        tr_s[5] = tr_s[5] + Int32(1)
                    if const_expr(self.dm_ts):
                        dent = cute.arch.add_packed_f32x2(cute.arch.add_packed_f32x2(den, den1), cute.arch.add_packed_f32x2(den2, den3))
                        # TwoSum(a=sum, b=partial): s = a + b; bb = s - a; err = (a - (s - bb)) + (b - bb)
                        sn = cute.arch.add_packed_f32x2(ts_sn, num)
                        bbn = cute.arch.add_packed_f32x2(sn, (-ts_sn[0], -ts_sn[1]))
                        an = cute.arch.add_packed_f32x2(ts_sn, (-(sn[0] - bbn[0]), -(sn[1] - bbn[1])))
                        brn = cute.arch.add_packed_f32x2(num, (-bbn[0], -bbn[1]))
                        ts_en = cute.arch.add_packed_f32x2(ts_en, cute.arch.add_packed_f32x2(an, brn))
                        ts_sn = sn
                        sd = cute.arch.add_packed_f32x2(ts_sd, dent)
                        bbd = cute.arch.add_packed_f32x2(sd, (-ts_sd[0], -ts_sd[1]))
                        ad = cute.arch.add_packed_f32x2(ts_sd, (-(sd[0] - bbd[0]), -(sd[1] - bbd[1])))
                        brd = cute.arch.add_packed_f32x2(dent, (-bbd[0], -bbd[1]))
                        ts_ed = cute.arch.add_packed_f32x2(ts_ed, cute.arch.add_packed_f32x2(ad, brd))
                        ts_sd = sd
                    elif const_expr(self.dm_kh):
                        dent = cute.arch.add_packed_f32x2(cute.arch.add_packed_f32x2(den, den1), cute.arch.add_packed_f32x2(den2, den3))
                        pn = num[0] + num[1]
                        pd = dent[0] + dent[1]
                        yn = pn - kc_n
                        tn = ks_n + yn
                        kc_n = (tn - ks_n) - yn
                        ks_n = tn
                        yd = pd - kc_d
                        td = ks_d + yd
                        kc_d = (td - ks_d) - yd
                        ks_d = td
                    elif const_expr(self.dm_f64 and self.dm_f64k == 1):
                        dent = cute.arch.add_packed_f32x2(cute.arch.add_packed_f32x2(den, den1), cute.arch.add_packed_f32x2(den2, den3))
                        if const_expr(self.dm_f64y):
                            # ALU bit assembly; nonzero subnormal partials (practically never) take F2F in a divergent branch
                            c0 = dm_utils.f32_to_f64_fastpath(num[0])
                            c1 = dm_utils.f32_to_f64_fastpath(num[1])
                            c2 = dm_utils.f32_to_f64_fastpath(dent[0])
                            c3 = dm_utils.f32_to_f64_fastpath(dent[1])
                            if (dm_utils.f32_is_subnormal(num[0]) | dm_utils.f32_is_subnormal(num[1])
                                    | dm_utils.f32_is_subnormal(dent[0]) | dm_utils.f32_is_subnormal(dent[1])) != Int32(0):
                                c0 = Float64(num[0])
                                c1 = Float64(num[1])
                                c2 = Float64(dent[0])
                                c3 = Float64(dent[1])
                            num64 = num64 + (c0 + c1)
                            den64 = den64 + (c2 + c3)
                        elif const_expr(self.dm_f64x):
                            num64 = num64 + (dm_utils.f32_to_f64_exact(num[0]) + dm_utils.f32_to_f64_exact(num[1]))
                            den64 = den64 + (dm_utils.f32_to_f64_exact(dent[0]) + dm_utils.f32_to_f64_exact(dent[1]))
                        else:
                            num64 = num64 + (Float64(num[0]) + Float64(num[1]))
                            den64 = den64 + (Float64(dent[0]) + Float64(dent[1]))
                    elif const_expr(self.dm_f64):
                        if (n_tile + 1) % self.dm_f64k == 0 or n_tile == n_block_max - n_block_min - 1:
                            dent = cute.arch.add_packed_f32x2(cute.arch.add_packed_f32x2(den, den1), cute.arch.add_packed_f32x2(den2, den3))
                            num64 = num64 + Float64(num[0] + num[1])
                            den64 = den64 + Float64(dent[0] + dent[1])
                            num = (Float32(0.0), Float32(0.0))
                            den = (Float32(0.0), Float32(0.0))
                            den1 = (Float32(0.0), Float32(0.0))
                            den2 = (Float32(0.0), Float32(0.0))
                            den3 = (Float32(0.0), Float32(0.0))
                    s_full_phase ^= 1
            den = cute.arch.add_packed_f32x2(cute.arch.add_packed_f32x2(den, den1), cute.arch.add_packed_f32x2(den2, den3))
            if const_expr(self.dm_tr):
                if tidx == 0:
                    for k_ in cutlass.range_constexpr(6):
                        dm_utils.red_add_u64(aux[6].iterator.toint() + 8 * (8 * stage + k_), Int64(tr_s[k_]))
            num_t = num[0] + num[1]
            den_t = den[0] + den[1]
            if const_expr(self.dm_c3):
                # chunk-3 partials of this stage from the correction WG (named barrier 12 + stage, 256 threads)
                cute.arch.barrier(barrier_id=12 + stage, number_of_threads=256)
                num_t = num_t + sScale[stage * 256 + 2 * tidx]
                den_t = den_t + sScale[stage * 256 + 2 * tidx + 1]
            delta = num_t / den_t if den_t != Float32(0.0) else Float32(0.0)
            if const_expr(self.dm_f64):
                delta = Float32(num64 / den64) if den64 != Float64(0.0) else Float32(0.0)
            if const_expr(self.dm_ts):
                tn64 = (Float64(ts_sn[0]) + Float64(ts_sn[1])) + (Float64(ts_en[0]) + Float64(ts_en[1]))
                td64 = (Float64(ts_sd[0]) + Float64(ts_sd[1])) + (Float64(ts_ed[0]) + Float64(ts_ed[1]))
                delta = Float32(tn64 / td64) if td64 != Float64(0.0) else Float32(0.0)
            if const_expr(self.dm_kh):
                kn64 = Float64(ks_n) - Float64(kc_n)
                kd64 = Float64(ks_d) - Float64(kc_d)
                delta = Float32(kn64 / kd64) if kd64 != Float64(0.0) else Float32(0.0)
            if const_expr(self.dm_nd):
                # +nd: sum_j P = 1 to the forward's exp accuracy (P is normalized by the forward LSE) -> delta = sum P dP
                delta = num_t
            if row < seqlen.seqlen_q:
                mLSE[row, head_idx, batch_idx] = delta
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def tl_loop(self, stage: int, rh: int, thr_mma_qk, tStS, mLSE, pipeline_s_p_o, pipeline_dp, aux_data, block_info,
                num_splits, SeqlenInfoCls, tile_scheduler):
        """P0 +tl: two lanes per query row, 4 compute warps / SMSP (stage, row half rh).
        tcgen05.ld.16x64b: thread t of the warp holds TMEM lane quad*32 + 16*rh + t//4 + 8*(t&1) and the columns of
        parity p = (t>>1)&1, so the row's two lanes hold the x / y halves of every packed pair of the h2 path. Every scalar
        den / num-acc / TwoSum chain lives on one lane in the h2 path's exact order (bitwise P, dP, num, den, delta); the
        halves meet only in the final fp64 combine (shuffle xor 2). Requires +h2+ts, all-MUFU P, no c3/b6/r3."""
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
            l2 = dmh.dm_mx256_l2(Float32(aux[5][batch_idx, head_idx, rowc]))
            # this lane's component of the h2 path's TwoSum pairs, packed as (num, den)
            ts_s = (Float32(0.0), Float32(0.0))
            ts_e = (Float32(0.0), Float32(0.0))
            if const_expr(self.dm_pf):
                nb0 = cutlass.max(n_block_max - 1, 0) * (self.n_block_size // 32)
                pks = tuple(Float32(aux[1][batch_idx, kv_head_idx, nb0 + j]) for j in range(4))
                pvs = tuple(Float32(aux[2][batch_idx, kv_head_idx, nb0 + j]) for j in range(4))
            if has_work:
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    cs = [None] * 4
                    as_ = [None] * 4
                    if const_expr(self.dm_pf):
                        for j in cutlass.range_constexpr(4):
                            cs[j] = dmh.dm_mx256_c(pks[j], qsv)
                            as_[j] = dmh.dm_mx256_dp_scale(pvs[j], dosv)
                        nbn = cutlass.max(n_block - 1, n_block_min) * (self.n_block_size // 32)
                        pks = tuple(Float32(aux[1][batch_idx, kv_head_idx, nbn + j]) for j in range(4))
                        pvs = tuple(Float32(aux[2][batch_idx, kv_head_idx, nbn + j]) for j in range(4))
                    else:
                        for j in cutlass.range_constexpr(4):
                            kb = n_block * (self.n_block_size // 32) + j
                            cs[j] = dmh.dm_mx256_c(Float32(aux[1][batch_idx, kv_head_idx, kb]), qsv)
                            as_[j] = dmh.dm_mx256_dp_scale(Float32(aux[2][batch_idx, kv_head_idx, kb]), dosv)
                    # den chains of this lane's component: d01 = (c0, c1), d23 = (c2, c3) (h2: den[v % 4], v = pair)
                    d01 = (Float32(0.0), Float32(0.0))
                    d23 = (Float32(0.0), Float32(0.0))
                    num = (Float32(0.0), Float32(0.0))   # (num of this component, unused)
                    pipeline_s_p_o.consumer_wait_w_index_phase(stage, s_full_phase)
                    diag = False
                    if const_expr(self.is_causal):
                        diag = (n_block + 1) * self.n_block_size > m_block * self.m_block_size
                    kbase = n_block * self.n_block_size
                    r = cute.make_rmem_tensor(cute.make_layout(64), self.qk_acc_dtype)
                    if const_expr(self.dm_es):
                        # +es: both S halves to registers first, S slot released before any exp (values/order unchanged)
                        for h in cutlass.range_constexpr(2):
                            vals = dm_utils.tmem_ld_16x64b_x32(s_addr + Int32(64 * h))
                            for i in cutlass.range_constexpr(32):
                                r[32 * h + i] = vals[i]
                        cute.arch.fence_view_async_tmem_load()
                        with dm_utils.tcf_site("r0"):
                            pipeline_s_p_o.consumer_release_w_index(stage)
                    for h in cutlass.range_constexpr(2):
                        if const_expr(not self.dm_es and not (self.dm_es2 and h == 1)):
                            vals = dm_utils.tmem_ld_16x64b_x32(s_addr + Int32(64 * h))
                            for i in cutlass.range_constexpr(32):
                                r[32 * h + i] = vals[i]
                            if const_expr(h == 1):
                                cute.arch.fence_view_async_tmem_load()
                                with dm_utils.tcf_site("r0"):
                                    pipeline_s_p_o.consumer_release_w_index(stage)
                            else:
                                cute.arch.fence_view_async_tmem_load()
                        if const_expr(self.is_causal):
                            if diag:
                                for i in cutlass.range_constexpr(32 * h, 32 * h + 32):
                                    r[i] = -Float32.inf if kbase + 2 * i + par > row else r[i]
                        for j in cutlass.range_constexpr(2 * h, 2 * h + 2):
                            for v in cutlass.range_constexpr(0, 16, 2):
                                i0 = 16 * j + v
                                p = dmh.dm_mx256_p_pair(r[i0], r[i0 + 1], cs[j], l2)
                                r[i0], r[i0 + 1] = p
                                if const_expr(v % 4 == 0):
                                    d01 = cute.arch.add_packed_f32x2(d01, p)
                                else:
                                    d23 = cute.arch.add_packed_f32x2(d23, p)
                        if const_expr(h == 0 and self.dm_es2):
                            # +es2: S half 1 to registers and S slot released after the h0 exps, before the dP wait
                            vals = dm_utils.tmem_ld_16x64b_x32(s_addr + Int32(64))
                            for i in cutlass.range_constexpr(32):
                                r[32 + i] = vals[i]
                            cute.arch.fence_view_async_tmem_load()
                            with dm_utils.tcf_site("r0"):
                                pipeline_s_p_o.consumer_release_w_index(stage)
                        if const_expr(h == 0 and not self.dm_ea):
                            pipeline_dp.consumer_wait_w_index_phase(stage, s_full_phase)
                        if const_expr(self.dm_ea and h == 1):
                            # +ea: every exp (and the den chains, same order) is done; now the dP wait and all 4 num chunks
                            pipeline_dp.consumer_wait_w_index_phase(stage, s_full_phase)
                        jlo, jhi = (2 * h, 2 * h + 2) if not self.dm_ea else ((0, 4) if h == 1 else (0, 0))
                        for j in cutlass.range_constexpr(jlo, jhi):
                            dpv = dm_utils.tmem_ld_16x64b_x16(d_addr + Int32(32 * j))
                            cute.arch.fence_view_async_tmem_load()
                            if const_expr(j == 3):
                                with dm_utils.tcf_site("r1"):
                                    pipeline_dp.consumer_release_w_index(stage)
                            a01 = (Float32(0.0), Float32(0.0))
                            a23 = (Float32(0.0), Float32(0.0))
                            for v in cutlass.range_constexpr(0, 16, 2):
                                i0 = 16 * j + v
                                if const_expr(v % 4 == 0):
                                    a01 = cute.arch.fma_packed_f32x2((r[i0], r[i0 + 1]), (dpv[v], dpv[v + 1]), a01)
                                else:
                                    a23 = cute.arch.fma_packed_f32x2((r[i0], r[i0 + 1]), (dpv[v], dpv[v + 1]), a23)
                            # h2: acc01 = acc0 + acc1, acc23 = acc2 + acc3, num = fma(acc01 + acc23, a_j, num) per component
                            t = cute.arch.add_packed_f32x2((a01[0], a23[0]), (a01[1], a23[1]))
                            u = cute.arch.add_packed_f32x2((t[0], Float32(0.0)), (t[1], Float32(0.0)))
                            num = cute.arch.fma_packed_f32x2((u[0], Float32(0.0)), (as_[j], Float32(0.0)), num)
                    # dent = (den + den1) + (den2 + den3) per component
                    t = cute.arch.add_packed_f32x2((d01[0], d23[0]), (d01[1], d23[1]))
                    dent = cute.arch.add_packed_f32x2((t[0], Float32(0.0)), (t[1], Float32(0.0)))
                    # packed TwoSum of (num, den) of this component (h2+ts per-component arithmetic)
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

    @cute.jit
    def c3_loop(self, thr_mma_qk, tStS, sScale, aux_data, pipeline_s_p_o, pipeline_dp, block_info, num_splits,
                SeqlenInfoCls, tile_scheduler):
        """P0 +c3: correction warpgroup computes the partial sums of column chunk 3 for both q stages."""
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        aux = aux_data.tensors
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype)
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        J = 3
        tS = [None] * self.q_stage
        tdP = [None] * self.q_stage
        thr_tmem_load = None
        for st in cutlass.range_constexpr(self.q_stage):
            tSAcc = tStS[(None, None), 0, 0, st]
            tdPAcc = cute.make_tensor(tSAcc.iterator + (self.tmem_o_offset[0] - self.tmem_s_offset[0]), tSAcc.layout)
            thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
            tS[st] = thr_tmem_load.partition_S(tSAcc)
            tdP[st] = thr_tmem_load.partition_S(tdPAcc)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
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
            rows = [None] * self.q_stage
            qsv = [None] * self.q_stage
            dosv = [None] * self.q_stage
            l2 = [None] * self.q_stage
            num = [None] * self.q_stage
            den = [None] * self.q_stage
            denb = [None] * self.q_stage
            for st in cutlass.range_constexpr(self.q_stage):
                rows[st] = self._q_block(work_m, st) * self.m_block_size + tidx
                rowc = cutlass.min(rows[st], seqlen.seqlen_q - 1)
                qsv[st] = Float32(aux[0][batch_idx, head_idx, rowc // 32])
                dosv[st] = Float32(aux[3][batch_idx, head_idx, rowc // 32])
                l2[st] = Float32(aux[5][batch_idx, head_idx, rowc]) * Float32(dm_common.LOG2E)
                num[st] = (Float32(0.0), Float32(0.0))
                den[st] = (Float32(0.0), Float32(0.0))
                denb[st] = (Float32(0.0), Float32(0.0))
            if has_work:
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    kb = n_block * (self.n_block_size // 32) + J
                    ksv = Float32(aux[1][batch_idx, kv_head_idx, kb])
                    vsv = Float32(aux[2][batch_idx, kv_head_idx, kb])
                    tSrS = [None] * self.q_stage
                    for st in cutlass.range_constexpr(self.q_stage):
                        pipeline_s_p_o.consumer_wait_w_index_phase(st, s_full_phase)
                        tSrS[st] = cute.make_rmem_tensor(tScS_t2r[None, None, J].shape, self.qk_acc_dtype)
                        cute.copy(thr_tmem_load, tS[st][None, None, J], tSrS[st])
                        cute.arch.fence_view_async_tmem_load()
                        with dm_utils.tcf_site("r5"):
                            pipeline_s_p_o.consumer_release_w_index(st)
                    variant = self.dm_exp_variant.split("+")[0]
                    for st in cutlass.range_constexpr(self.q_stage):
                        c = dm_common.dm_bwd_c(ksv, qsv[st])
                        a = dm_common.dm_bwd_dp_scale(vsv, dosv[st])
                        if const_expr(self.is_causal):
                            if (n_block + 1) * self.n_block_size > self._q_block(work_m, st) * self.m_block_size:
                                kbase = n_block * self.n_block_size + 32 * J
                                for i in cutlass.range_constexpr(32):
                                    tSrS[st][i] = -Float32.inf if kbase + i > rows[st] else tSrS[st][i]
                        for v in cutlass.range_constexpr(16):
                            if const_expr(dm_common.is_static_mix(variant)):
                                if const_expr(dm_common.p5_uses_poly(variant, v)):
                                    if const_expr(variant.startswith("p5_")):
                                        p = dm_common.dm_bwd_p_pair_poly5(tSrS[st][2 * v], tSrS[st][2 * v + 1], c, l2[st])
                                    else:
                                        p = dm_common.dm_bwd_p_pair_poly(tSrS[st][2 * v], tSrS[st][2 * v + 1], c, l2[st])
                                else:
                                    p = dm_common.dm_bwd_p_pair(tSrS[st][2 * v], tSrS[st][2 * v + 1], c, l2[st])
                            else:
                                p = dm_common.P_PAIR_VARIANTS[variant](tSrS[st][2 * v], tSrS[st][2 * v + 1], c, l2[st])
                            tSrS[st][2 * v], tSrS[st][2 * v + 1] = p
                            if const_expr(self.dm_ma and v % 2 == 1):
                                denb[st] = cute.arch.add_packed_f32x2(denb[st], p)
                            else:
                                den[st] = cute.arch.add_packed_f32x2(den[st], p)
                        pipeline_dp.consumer_wait_w_index_phase(st, s_full_phase)
                        tdPr = cute.make_rmem_tensor(tScS_t2r[None, None, J].shape, self.qk_acc_dtype)
                        cute.copy(thr_tmem_load, tdP[st][None, None, J], tdPr)
                        cute.arch.fence_view_async_tmem_load()
                        with dm_utils.tcf_site("r6"):
                            pipeline_dp.consumer_release_w_index(st)
                        if const_expr(self.dm_ma):
                            acc = [(Float32(0.0), Float32(0.0)) for _ in range(4)]
                            for v in cutlass.range_constexpr(16):
                                acc[v % 4] = cute.arch.fma_packed_f32x2(
                                    (tSrS[st][2 * v], tSrS[st][2 * v + 1]), (tdPr[2 * v], tdPr[2 * v + 1]), acc[v % 4])
                            accs = cute.arch.add_packed_f32x2(cute.arch.add_packed_f32x2(acc[0], acc[1]),
                                                              cute.arch.add_packed_f32x2(acc[2], acc[3]))
                            num[st] = cute.arch.fma_packed_f32x2(accs, (a, a), num[st])
                        else:
                            for v in cutlass.range_constexpr(16):
                                dpv = cute.arch.mul_packed_f32x2((tdPr[2 * v], tdPr[2 * v + 1]), (a, a))
                                num[st] = cute.arch.fma_packed_f32x2((tSrS[st][2 * v], tSrS[st][2 * v + 1]), dpv, num[st])
                    s_full_phase ^= 1
            for st in cutlass.range_constexpr(self.q_stage):
                sScale[st * 256 + 2 * tidx] = num[st][0] + num[st][1]
                if const_expr(not self.dm_nd):
                    dsum = cute.arch.add_packed_f32x2(den[st], denb[st])
                    sScale[st * 256 + 2 * tidx + 1] = dsum[0] + dsum[1]
                cute.arch.barrier_arrive(barrier_id=12 + st, number_of_threads=256)
            work_tile = tile_scheduler.advance_to_next_work()

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
        nb4 = n_block * 4
        dm_c = [dm_qsc * Float32(dm_scales.k_descale[batch_idx, kv_head_idx, nb4 + j]) for j in range(4)]
        dm_bmax = [None] * 4
        for j in cutlass.range_constexpr(4):
            v = tSrS_t2r[32 * j]
            for i in cutlass.range_constexpr(1, 32):
                v = cute.arch.fmax(v, tSrS_t2r[32 * j + i])
            dm_bmax[j] = v * dm_c[j]
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
            dm_w[j] = dm_utils.exp2i_f32(e)
            dm_sfa[j] = dm_utils.ue8m0_byte(e)
        sfa_word = dm_utils.bitcast_i32_f32(dm_utils.pack_bytes4(dm_sfa[0], dm_sfa[1], dm_sfa[2], dm_sfa[3]))
        sfb_word = dm_utils.bitcast_i32_f32(dm_utils.pack_bytes4(
            *[dm_utils.pow2_exponent_byte(Float32(dm_scales.v_descale[batch_idx, kv_head_idx, nb4 + j])) for j in range(4)]))
        tSrSF = cute.make_rmem_tensor(thr_tmem_store_sf.partition_S(cute.make_identity_tensor((self.m_block_size, 8))).shape, Float32)
        for j in cutlass.range_constexpr(4):
            tSrSF[j] = sfa_word
            tSrSF[4 + j] = sfb_word
        cute.copy(thr_tmem_store_sf, tSrSF, tStSF_r2t)

        if const_expr(self.use_s_ping_pong):
            sm_stats_index = (s_p_o_consumer_count - 1) & 1
        else:
            sm_stats_index = s_p_o_index
        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
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
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
                if const_expr(i + 1 == split_P_arrive_idx):
                    cute.arch.fence_view_async_tmem_store()
                    with dm_utils.tcf_site("r7"):
                        pipeline_s_p_o.consumer_release_w_index(s_p_o_index)
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(s_p_o_index)
        else:
            with dm_utils.tcf_site("r8"):
                pipeline_s_p_o.consumer_release_w_index(s_p_o_index)
        if const_expr(self.use_s_ping_pong):
            pipeline_sm_stats.producer_acquire_w_index_phase(
                s_p_o_consumer_count & 1, 1 ^ ((s_p_o_consumer_count >> 1) & 1)
            )
        else:
            pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        # l = l * acc_scale + sum_b w_b * sum_{j in b} p~_j   (fp32 pre-rounding values)
        dm_l = Float32(0.0) if const_expr(is_first) else softmax.row_sum[0] * acc_scale
        for j in cutlass.range_constexpr(4):
            ps = (tSrS_t2r[32 * j], tSrS_t2r[32 * j + 1])
            for i in cutlass.range_constexpr(2, 32, 2):
                ps = cute.arch.add_packed_f32x2(ps, (tSrS_t2r[32 * j + i], tSrS_t2r[32 * j + i + 1]))
            dm_l = dm_l + (ps[0] + ps[1]) * dm_w[j]
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
        """P0: no O to correct; only walk the tile schedule (CLC / dynamic schedulers count every warp)."""
        if const_expr(not (self.dm_ni and self.dynamic_persistent)):
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                work_tile = tile_scheduler.advance_to_next_work()

    def _dm_idle_empty_warps(self):
        """Empty warps that are not the scheduler warp (they only walk the schedule)."""
        return tuple(w for w in self.empty_warp_ids if w != self.scheduler_warp_id)

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
        load_dO_fn: Optional[Callable] = None,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))
        load_dO_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))

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
