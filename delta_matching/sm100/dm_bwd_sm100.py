# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
import enum
import math
import os
from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute import FastDivmodDivisorV2
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.utils import LayoutEnum, TensorMapManager, TensorMapUpdateMode
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.pipeline import PipelineAsync

from quack import layout_utils
from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import copy_utils
from flash_attn.cute import pipeline
from flash_attn.cute.bwd_accum_tile import AccumTileOrder, accum_tile_layout
from flash_attn.cute.config import Sm100BwdRegisters
from flash_attn.cute.blackwell_helpers import gemm_w_idx, gemm_ptx_w_idx  # noqa
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.pack_gqa import pack_gqa_layout
from flash_attn.cute.seqlen_info import SeqlenInfoQK, sequence_row_offset
from flash_attn.cute.block_info import BlockInfo
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTBwdScheduler,  # noqa
    SingleTileVarlenScheduler,
)

from flash_attn.cute.tile_scheduler import WorkTileInfo as _DMWorkTileInfo

_DM_HSWZ_CLS = {}


def dm_hswz_scheduler(qh: int):
    """DM hswz (GQA, deterministic): the LPT bwd scheduler with the query head permuted so that consecutive
    scheduler heads h map to DIFFERENT KV heads at the same group position: head(h) = (h % Hk) * qh + h // Hk.
    The deterministic dK/dV lock order is the group position (head % qh), so the 4 (qh) CTAs of one KV tile no longer
    run concurrently and serialize on the dK/dV semaphores; accumulation order (bits) is unchanged."""
    if qh in _DM_HSWZ_CLS:
        return _DM_HSWZ_CLS[qh]

    class DMHswzLPTBwdScheduler(SingleTileLPTBwdScheduler):
        @staticmethod
        @cute.jit
        def create(params, *, loc=None, ip=None):
            return DMHswzLPTBwdScheduler(params, cute.arch.block_idx()[0], loc=loc, ip=ip)

        @cute.jit
        def get_current_work(self, *, loc=None, ip=None):
            w = SingleTileLPTBwdScheduler.get_current_work(self, loc=loc, ip=ip)
            blk, h, b, sp = w.tile_idx
            hk = self.params.num_head_divmod.divisor // qh
            hn = (h % hk) * qh + h // hk
            return _DMWorkTileInfo((blk, Int32(hn), b, sp), w.is_valid_tile)

    _DM_HSWZ_CLS[qh] = DMHswzLPTBwdScheduler
    return DMHswzLPTBwdScheduler


_DM_POSMAJ_CLS = {}


def dm_posmajor_scheduler(qh: int):
    """DM posmaj (GQA, deterministic): LPT bwd scheduler over (KV head, batch, block) tiles, run qh times in
    group-position-major order: all tiles of group position 0, then 1, ... (query head = kv_head * qh + pos). The
    deterministic dK/dV lock order is the group position, so a tile's predecessor ran ~total/qh tiles earlier instead
    of concurrently (no dK/dV semaphore serialization); dQ lock order per query head is unchanged (bitwise).
    Built with num_head = Hk; grid x qh."""
    if qh in _DM_POSMAJ_CLS:
        return _DM_POSMAJ_CLS[qh]

    class DMPosMajLPTBwdScheduler(SingleTileLPTBwdScheduler):
        @staticmethod
        @cute.jit
        def create(params, *, loc=None, ip=None):
            return DMPosMajLPTBwdScheduler(params, cute.arch.block_idx()[0], loc=loc, ip=ip)

        @staticmethod
        def get_grid_shape(params, *, loc=None, ip=None):
            return (params.total_blocks * qh, Int32(1), Int32(1))

        @cute.jit
        def get_current_work(self, *, loc=None, ip=None):
            params = self.params
            pos = self._tile_idx // params.total_blocks
            local = self._tile_idx - pos * params.total_blocks
            cluster_idx = local // params.cluster_shape_mn[0]
            bidhb, l2_mod = divmod(cluster_idx, params.l2_major_divmod)
            block, bidhb_residual = 0, 0
            if bidhb < params.num_hb_quotient:
                block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
            else:
                block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
            bidhb_actual = bidhb * params.l2_minor + bidhb_residual
            batch_idx, kv_head = divmod(bidhb_actual, params.num_head_divmod)
            if const_expr(params.spt):
                block = params.num_block - 1 - block
            is_valid = self._tile_idx < params.total_blocks * qh
            return _DMWorkTileInfo((Int32(block), Int32(kv_head * qh + pos), Int32(batch_idx), Int32(0)), is_valid)

        def advance_to_next_work(self, *, loc=None, ip=None):
            # single tile per CTA: the end marker must be past the qh x larger range
            self._tile_idx = self.params.total_blocks * qh
            return self.get_current_work()

    _DM_POSMAJ_CLS[qh] = DMPosMajLPTBwdScheduler
    return DMPosMajLPTBwdScheduler


_DM_PM1_CLS = {}


def dm_posmajor_once_scheduler(qh: int):
    """DM pm1 (det GQA): the posmaj scheduler with the SAME tile mapping, but the work tile is computed once at create()
    and the scheduler state is only (block, head, batch, split, valid). advance_to_next_work just clears valid, so no
    role keeps the LPT FastDivmod parameters live across its main loop (they cost the 48-register MMA/load warps
    ~60 spill instructions in det). Same CTA -> tile mapping and lock orders: bitwise."""
    if qh in _DM_PM1_CLS:
        return _DM_PM1_CLS[qh]
    base = dm_posmajor_scheduler(qh)

    class DMPosMajOnceScheduler:
        Params = base.Params

        def __init__(self, c0, c1, c2, c3, v, *, loc=None, ip=None):
            self._c = [c0, c1, c2, c3]
            self._v = v
            self._loc = loc
            self._ip = ip

        @staticmethod
        def to_underlying_arguments(args, *, scheduling_mode=None, loc=None, ip=None):
            return base.to_underlying_arguments(args, loc=loc, ip=ip)

        @staticmethod
        def get_grid_shape(params, *, loc=None, ip=None):
            return base.get_grid_shape(params, loc=loc, ip=ip)

        @staticmethod
        @cute.jit
        def create(params, *, loc=None, ip=None):
            w = base.create(params, loc=loc, ip=ip).get_current_work()
            b_, h_, z_, s_ = w.tile_idx
            return DMPosMajOnceScheduler(Int32(b_), Int32(h_), Int32(z_), Int32(s_), Int32(w.is_valid_tile), loc=loc, ip=ip)

        @cute.jit
        def get_current_work(self, *, loc=None, ip=None):
            return _DMWorkTileInfo((self._c[0], self._c[1], self._c[2], self._c[3]), self._v != Int32(0))

        def initial_work_tile_info(self, *, loc=None, ip=None):
            return self.get_current_work(loc=loc, ip=ip)

        def prefetch_next_work(self, *, loc=None, ip=None):
            pass

        def advance_to_next_work(self, *, loc=None, ip=None):
            self._v = Int32(0)
            return self.get_current_work()

        def __extract_mlir_values__(self):
            values = []
            for obj in self._c + [self._v]:
                values += cutlass.extract_mlir_values(obj)
            return values

        def __new_from_mlir_values__(self, values):
            objs = []
            for obj in self._c + [self._v]:
                n = len(cutlass.extract_mlir_values(obj))
                objs.append(cutlass.new_from_mlir_values(obj, values[:n]))
                values = values[n:]
            return self.__class__(*objs, loc=self._loc)

    _DM_PM1_CLS[qh] = DMPosMajOnceScheduler
    return DMPosMajOnceScheduler


def _dm_drop_dz(fn):
    """MMA helpers without DM d3nh's opaque-zero operand: accept and ignore dz."""
    def f(*a, dz=None, **kw):
        return fn(*a, **kw)
    return f


_DM_PERS_CLS = {}


def dm_persistent_scheduler(qh: int, hswz: bool, max_ctas: int, snake: bool = False):
    """DM pers: grid-stride persistent version of the LPT bwd scheduler (optionally with the hswz head permutation).
    CTA b handles linear work tiles b, b + G, b + 2G, ... (G = min(total, max_ctas) physical CTAs, a cluster multiple), in
    increasing order, so every deterministic predecessor (smaller linear index) is earlier in some CTA's queue: no
    deadlock. Saves the per-tile CTA teardown/launch/prologue; the work and its order are unchanged (bitwise)."""
    key = (qh, hswz, max_ctas, snake)
    if key in _DM_PERS_CLS:
        return _DM_PERS_CLS[key]

    class DMPersLPTBwdScheduler(SingleTileLPTBwdScheduler):
        @staticmethod
        @cute.jit
        def create(params, *, loc=None, ip=None):
            return DMPersLPTBwdScheduler(params, cute.arch.block_idx()[0], loc=loc, ip=ip)

        @staticmethod
        def get_grid_shape(params, *, loc=None, ip=None):
            return (cutlass.min(params.total_blocks, Int32(max_ctas)), Int32(1), Int32(1))

        @cute.jit
        def get_current_work(self, *, loc=None, ip=None):
            w = SingleTileLPTBwdScheduler.get_current_work(self, loc=loc, ip=ip)
            blk, h, b, sp = w.tile_idx
            hn = h
            if const_expr(hswz and qh > 1):
                hk = self.params.num_head_divmod.divisor // qh
                hn = (h % hk) * qh + h // hk
            return _DMWorkTileInfo((blk, Int32(hn), b, sp), w.is_valid_tile)

        def advance_to_next_work(self, *, loc=None, ip=None):
            if const_expr(snake):
                # boustrophedon over clusters: round r covers cluster tiles [r C, (r+1) C) (contiguous, so every
                # deterministic predecessor is in an earlier or the same round: no deadlock); odd rounds reversed,
                # so the LPT-sorted (longest-first) tiles pair long with short per CTA
                cs = Int32(self.params.cluster_shape_mn[0])
                C = cute.arch.grid_dim()[0] // cs
                c = cute.arch.block_idx()[0] // cs
                r = self._tile_idx // cs // C + 1
                pos = c + (r & 1) * (C - 1 - 2 * c)
                self._tile_idx = (r * C + pos) * cs + cute.arch.block_idx()[0] % cs
            else:
                self._tile_idx = self._tile_idx + cute.arch.grid_dim()[0]
            return self.get_current_work()

    _DM_PERS_CLS[key] = DMPersLPTBwdScheduler
    return DMPersLPTBwdScheduler


_DM_DYN_CLS = {}


def dm_dynamic_scheduler(qh: int, hswz: bool, max_ctas: int, n_cons: int):
    """DM pdyn: dynamic persistent version of the LPT bwd scheduler. The leader CTA's empty warp claims cluster tiles in
    increasing linear order with a global atomic and broadcasts them to both CTAs through a 2-slot smem ring (full /
    empty mbarriers; every role warp of both CTAs consumes every slot). Claims are globally ordered, so the smallest
    unfinished tile is always runnable (its deterministic predecessors are finished): no deadlock, any grid size."""
    key = (qh, hswz, max_ctas, n_cons)
    if key in _DM_DYN_CLS:
        return _DM_DYN_CLS[key]
    base_cls = dm_persistent_scheduler(qh, hswz, max_ctas)

    class DMDynLPTBwdScheduler(base_cls):
        def __init__(self, params, tile_idx, k, sbase, *, loc=None, ip=None):
            super().__init__(params, tile_idx, loc=loc, ip=ip)
            self._k = k
            self._sbase = sbase

        @staticmethod
        @cute.jit
        def create(params, sbase, *, loc=None, ip=None):
            t = DMDynLPTBwdScheduler._fetch(params, sbase, Int32(0))
            return DMDynLPTBwdScheduler(params, t, Int32(0), sbase, loc=loc, ip=ip)

        @staticmethod
        def _fetch(params, sbase, k):
            s = k & 1
            dm_utils.mbar_wait_parity_cluster(sbase + 8 * 3 + 8 * s, (k >> 1) & 1)
            t = dm_utils.ld_shared_b32(sbase + 8 * 7 + 8 * s)
            with cute.arch.elect_one():
                dm_utils.mbar_arrive_rank(sbase + 8 * 5 + 8 * s, Int32(0))
            cs = params.cluster_shape_mn[0]
            return t * cs + cute.arch.block_idx()[0] % cs

        def advance_to_next_work(self, *, loc=None, ip=None):
            self._k = self._k + 1
            self._tile_idx = DMDynLPTBwdScheduler._fetch(self.params, self._sbase, self._k)
            return self.get_current_work()

        def __extract_mlir_values__(self):
            values, self._values_pos = [], []
            for obj in [self.params, self._tile_idx, self._k, self._sbase]:
                obj_values = cutlass.extract_mlir_values(obj)
                values += obj_values
                self._values_pos.append(len(obj_values))
            return values

        def __new_from_mlir_values__(self, values):
            obj_list = []
            for obj, n_items in zip([self.params, self._tile_idx, self._k, self._sbase], self._values_pos):
                obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
                values = values[n_items:]
            return self.__class__(*(tuple(obj_list)), loc=self._loc)

    _DM_DYN_CLS[key] = DMDynLPTBwdScheduler
    return DMDynLPTBwdScheduler


from flash_attn.cute import barrier
from flash_attn.cute.named_barrier import NamedBarrierBwdSm100
from flash_attn.cute.softmax import apply_score_mod_inner, apply_score_mod_bwd_inner
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.utils import AuxData
from delta_matching.sm100 import dm_utils


def utils_abs(x):
    return cute.arch.fmax(x, -x)


from flash_attn.cute.block_sparse_utils import (
    get_total_q_block_count_bwd,
    get_block_sparse_iteration_info_bwd,
    get_m_block_from_iter_bwd,
    produce_block_sparse_q_loads_bwd_sm100_2cta_hdim192,
    produce_block_sparse_q_loads_bwd_sm100_default,
)


# NOTE [hdim64 dedicated P/dS TMEM slots]
# The 1-CTA hdim64 backward uses TMEM columns [0, 384), leaving room for two 64-column
# slots. With split_P_dS, P and dS live there instead of aliasing S and dP:
#   * S is released as soon as the compute warps have loaded it, so the MMA warp issues
#     QK_{t+1} while the softmax of tile t is still running; dV += P.T @ dO for tile t
#     then trails QK_{t+1} and is gated by its own pipeline_P.
#   * The compute-wide barriers that ordered the P-over-S and dS-over-dP overwrites are
#     gone. They were cross-warp hazards: a warp's P (dS) columns overlap S (dP) lanes
#     that another warp still has to load.
#   * The remaining compute-wide barriers before pipeline arrivals become sync_warp: every
#     mbarrier signalled from the compute warps expects one arrival per warp, so the
#     mbarrier itself is the cross-warp join. This is 1-CTA only: at 2-CTA the barrier
#     before the dS commit also orders every warp's sdS_xchg write ahead of the
#     single-thread DSMEM copy to the peer CTA.
# The S handshake keeps its own phase/state (producer_phase_S, consumer_state_S) so its
# parity never depends on where the other pipelines flip. Perf evidence in PR #2804; the
# aliased layout is what every other shape runs.


# NOTE [SM100 backward coordinate spaces]
# C = cta_group_size, N = tile_n; a work tile spans C * N KV rows.
#
#   physical CTA (blockIdx.x = C * cluster_slot + cta_rank)
#        |  scheduler (NOTE [SM100 backward work coordinates], tile_scheduler.py)
#   (work_n, head_idx, batch_idx)        same on every CTA of a cluster
#        |  work_kv_block(work_n, cta_rank)
#   kv_block                             N-row KV block of this CTA
#        |  * N + row_in_cta
#   kv_row                               batch-local KV row
#        |  layouts, seqlen_info offsets
#   memory
#
# Users of work_n (the C * N-row cluster tile): the BlockInfo m_block range, the mask and
# score_mod coordinates, the direct dK/dV store, the dQ semaphore lock value and the K/V/Kt
# TMA tiles; their thr_mma partitions (sliced by mma_tile_coord_v) already select the CTA's
# rows, so they take rank 0. The empty-tile dK/dV zero-fill (a local / varlen Q range or
# block sparsity leaving no Q block) is the exception: it is not an MMA partition, and both
# CTAs zero the whole C * N-row work tile, the same rows twice. Users of kv_block: the TMA
# dK/dV epilogues, the dK/dV semaphores, the direct-store row predicate and the block-sparse
# KV column. The rank also selects the CTA's dQaccum reduce stages (rows [r * M/2,
# (r+1) * M/2) of a Q tile) and its dQ semaphore slot; it never enters a head coordinate.
#
# Head spaces (qh = qhead_per_kvhead, pf = pack_factor = qh with pack_gqa, else 1):
#   head_idx      scheduler head: a query head, or a KV head with pack_gqa
#   kv_head       bwd_kv_head(head_idx): K, V, dK, dV and the dK/dV semaphores
#   scratch head  = head_idx: LSE_log2, dPsum, dQaccum and the dQ semaphores, pf rows per token
#   packed row r  (token r // pf, query head head_idx * pf + r % pf), NOTE [bwd pack_gqa]:
#                 with pf = 1 that is query head head_idx itself, with pack_gqa head_idx is
#                 the KV head and its pf query heads interleave per token
#   kv_group_pos  bwd_kv_group_pos(head_idx) = head_idx % qh: deterministic dK/dV order
#
# Worked example, C = 2, N = 64, qh = 4, batch 0:
#   blockIdx.x = 5 is cluster slot 2, cta_rank 1. Say the scheduler hands that cluster
#   work_n = 2 (both CTAs see it): the work tile is KV rows [256, 384), and
#   work_kv_block(2, 1) = 5 gives this CTA rows [320, 384), kv_row = 5 * 64 + row_in_cta.
#   pack_gqa off, head_idx = 5: kv_head 5 // 4 = 1, kv_group_pos 5 % 4 = 1, scratch head 5,
#   pf = 1, so packed row r is token r of query head 5.
#   pack_gqa on, head_idx = 1: that is kv_head 1 (one tile per KV head), scratch head 1 with
#   pf = 4 rows per token; packed row 6 is token 6 // 4 = 1 of query head 1 * 4 + 6 % 4 = 6.


def work_kv_block(work_n, cta_rank, *, cta_group_size: int):
    """tile_n-row KV block of one CTA of a work tile; the CTA rank is applied here only."""
    return work_n * cta_group_size + cta_rank


def bwd_kv_head(head_idx, *, qhead_per_kvhead: int, pack_gqa: bool):
    """KV head of a scheduler head: tiles are per query head, or per KV head with pack_gqa."""
    return head_idx if pack_gqa else head_idx // qhead_per_kvhead


def bwd_kv_group_pos(head_idx, *, qhead_per_kvhead: int):
    """Position of a query-head tile in its KV group: the deterministic dK/dV accumulation
    order of the GQA postprocess path."""
    return head_idx % qhead_per_kvhead


class CtaSchedule(enum.Enum):
    """Backward only: how a CTA (pair) issues the five GEMMs; see NOTE [2CTA schedules]."""

    ONE_CTA = enum.auto()
    TWO_CTA_PIPELINED = enum.auto()
    TWO_CTA_SERIAL = enum.auto()


class DMBackwardSm100:
    """FA4 sm100 backward forked for the Delta-Matching sm100-mx numerics: fp8 Q/K/V/dO codes with pow2
    per-32-row scales (aux_data.tensors = [qs, ks, vs, dos]), P^T / dS^T e4m3 with MX block scales, MXFP8 dV / dK / dQ."""

    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
        cluster_size: int = 1,
        use_2cta_instrs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        q_subtile_factor: cutlass.Constexpr[int] = 1,
        kv_subtile_factor: cutlass.Constexpr[int] = 1,
        pack_gqa: bool = False,
        *,
        spt: bool,
        registers: Sm100BwdRegisters,
        Q_stage: int,
        dO_stage: int,
    ):
        # Pad head_dim to a multiple of 32 (k_block_size). Must stay a multiple of 32 so
        # that the dQ accumulator reduce (dQ_reduce_ncol up to 32) tiles evenly and matches
        # both the preprocess kernel and the interface's head_dim_rounded (also mult of 32).
        hdim_multiple_of = 32
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v

        self.tile_m = tile_m
        self.tile_n = tile_n

        is_hdim256 = self.tile_hdim == 256 and self.tile_hdimv == 256
        assert (
            self.tile_hdim <= 128
            or (self.tile_hdim == 192 and self.tile_hdimv == 128)
            or is_hdim256
        )
        assert self.tile_hdimv <= 128 or is_hdim256

        self.use_2cta_instrs = bool(use_2cta_instrs and cluster_size == 2)
        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        # A cluster is the 2CTA pair and nothing else: the work tile is C * tile_n rows
        # (NOTE [SM100 backward coordinate spaces]).
        assert cluster_size == self.cta_group_size, (
            f"cluster_size={cluster_size} must equal the CTA group size {self.cta_group_size}"
        )

        assert self.tile_hdim not in (192, 256) or self.use_2cta_instrs, (
            "Must use 2CTA for hdim 192 / 256"
        )

        # NOTE [2CTA schedules]
        # Two 2CTA schedules exist and a 2CTA configuration runs exactly one of them:
        # - pipelined (hdim <= 128): own sQt / sdOt / sdS_xchg buffers, the MMA warp software-
        #   pipelines S(t+1) with dK/dP/dQ/dV, and dQ overlaps the tail of S in TMEM, so the
        #   dS pipeline also signals that S has been read.
        # - serial (hdim 192): Qt and dOt alias Q and dO (one stage), sdS_xchg aliases
        #   sdQaccum (guarded by dQaccum_empty_mbar; M=128 layout only), the MMAs run
        #   serially, and dP overlaps the tail of S, so S is released right after it is read.
        # hdim 192 and 256 need the serial schedule's SMEM aliasing; hdim <= 128 runs the
        # pipelined one.
        if not self.use_2cta_instrs:
            self.cta_schedule = CtaSchedule.ONE_CTA
        elif self.tile_hdim in (192, 256):
            self.cta_schedule = CtaSchedule.TWO_CTA_SERIAL
        else:  # hdim <= 128; in practice 128 (the interface enables 2CTA for hdim >= 128 only)
            self.cta_schedule = CtaSchedule.TWO_CTA_PIPELINED
        self.cluster_tile_n = self.cta_group_size * tile_n
        # NOTE [M=64 accumulator layout]
        # With 2CTA and a 64-row KV tile per CTA the pair MMA has M = 128, and each TMEM lane
        # holds half a row: lanes 0-63 carry columns [0, N/2) and lanes 64-127 columns [N/2, N)
        # of the same 64 rows. Every accumulator then takes N // 2 columns, and P / dS are
        # staged in SMEM as the A operands of the dV / dK MMAs instead of living in TMEM.
        self.acc_m64 = self.use_2cta_instrs and tile_n == 64
        # With P / dS^T in SMEM, the dS half the peer needs for its dQ MMA is a contiguous
        # block of sdS_dK in exactly the peer's dQ A-operand layout (checked in
        # _check_dS_exchange_layouts), so the exchange is copied from there: no sdS_xchg
        # staging buffer, and no sdS_xchg / sdQaccum alias in the serial schedule.
        self.has_sdS_xchg = self.use_2cta_instrs and not self.acc_m64
        self.xchg_aliases_dQaccum = (
            self.has_sdS_xchg and self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL
        )
        # Storage order of the fp32 accumulate tiles the postprocess reads (bwd_accum_tile).
        self.dQaccum_order, self.dKVaccum_order = self.accum_tile_orders(
            self.use_2cta_instrs, tile_n
        )
        # hdim 256: dK + dV + S + dP + dQ only fit TMEM with the M=64 layout, and only the serial
        # schedule's SMEM aliasing fits the 256-wide Q / dO / K / V tiles.
        assert not is_hdim256 or (
            self.acc_m64 and self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL
        ), "hdim 256 backward needs 2CTA with a 64-row KV tile (serial schedule)"

        # CTA tiler
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (self.cta_group_size * tile_n, tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (self.cta_group_size * tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (self.cta_group_size * tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q
        self.mma_tiler_dsq = (self.cta_group_size * tile_n, self.tile_hdim, tile_m)
        # dQ = dS @ K
        # 2-CTA: reduction dim is cluster-wide (tile_n * cta_group_size).
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, self.cluster_tile_n)

        self.acc_dtype = Float32

        assert cluster_size in (1, 2), "Only cluster_size=1 or 2 is supported"
        self.cluster_shape_mn = (cluster_size, 1)
        # The MMA warp's producer_phase_acc (shared by the S/P, dP and dQ empty waits in the
        # aliased layout) is only consistent across one work tile per CTA.
        assert not is_persistent, "persistent tile scheduling is not supported"
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.qhead_per_kvhead = qhead_per_kvhead
        # pack_gqa: one tile per KV head over the packed rows of its query heads, see
        # NOTE [bwd pack_gqa] (pack_gqa.py)
        self.pack_gqa = pack_gqa
        self.pack_factor = qhead_per_kvhead if pack_gqa else 1
        assert not pack_gqa or (qhead_per_kvhead > 1 and tile_m % qhead_per_kvhead == 0), (
            "pack_gqa needs 1 < qhead_per_kvhead | tile_m"
        )
        self.deterministic = deterministic
        # Deterministic dQ: tiles with fewer m_blocks (short varlen sequences, local windows)
        # release each one immediately instead of one m_block behind, see dQacc_reduce.
        self.delay_release_min_m_blocks = 16
        self.spt = spt

        # Score mod and mask mod support
        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.q_subtile_factor = q_subtile_factor
        self.kv_subtile_factor = kv_subtile_factor
        assert self.kv_subtile_factor == 1 or self.kv_subtile_factor % self.cta_group_size == 0
        # For score_mod, use vec_size=1 (like forward) to handle per-element indices
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32

        # Speed optimizations, does not affect correctness
        self.shuffle_LSE = "shlse" in tuple(os.environ.get("DM_BWD_OPTS", "").split(","))
        self.shuffle_dPsum = "shdps" in tuple(os.environ.get("DM_BWD_OPTS", "").split(","))
        # Generally slower to use store dS in smem for dK, and doesn't work for 2cta
        self.use_smem_dS_for_mma_dK = False

        # See NOTE [hdim64 dedicated P/dS TMEM slots]
        self.split_P_dS = (
            cluster_size == 1
            and self.tile_m == 128
            and self.tile_n == 128
            and self.tile_hdim == 64
            and self.tile_hdimv == 64
            and not self.use_smem_dS_for_mma_dK
        )

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.relay_warp_id = 14
        self.empty_warp_id = 15
        # DM cwg4: 4 compute warpgroups (16 warps, 32 query columns per thread) -> 24 warps / 768 threads
        self.dm_cwg4 = "cwg4" in tuple(x for x in os.environ.get("DM_BWD_OPTS", "").split(",") if x)
        if self.dm_cwg4:
            self.compute_warp_ids = tuple(range(4, 20))
            self.mma_warp_id = 20
            self.load_warp_id = 21
            self.relay_warp_id = 22
            self.empty_warp_id = 23
            regs = [int(x) for x in next((o[5:] for o in os.environ.get("DM_BWD_OPTS", "").split(",") if o.startswith("regs=")), "96/88/64").split("/")]
            registers = type(registers)(reduce=regs[0], compute=regs[1], load=regs[2], mma=regs[2], empty=regs[2])
        self.dm_epi_barrier = 8 if self.dm_cwg4 else int(NamedBarrierBwdSm100.EpilogueWG1)
        # DM dsqw: a dedicated dS-quantization warpgroup (warps 16-19, TMEM lane quadrant = warp % 4) instead of
        # the reduce warps (dsq). Register split dsqr=R:C:L:H (reduce, compute, load/mma/relay, dS helper), per SMSP
        # R + 2C + max(L, empty) + H <= 512.
        _o = tuple(x for x in os.environ.get("DM_BWD_OPTS", "").split(",") if x)
        self.dm_dsqw = "dsqw" in _o
        # DM tlev=<bid>: per-tile event timeline (clock64) of CTA <bid> at trace[40000 + 32 * j + e] (timing only)
        self.dm_tlev = int(next((x[5:] for x in _o if x.startswith("tlev=")), "-1"))
        self.dm_noreg = "dsqnoreg" in _o   # debug: no setmaxnreg at all
        self.dm_dqstream = "dqstream" in _o   # stream the dQ drain in 32-column t2r chunks
        self.dsq_warp_ids = (16, 17, 18, 19) if self.dm_dsqw else ()
        if self.dm_dsqw:
            r_ = [int(x) for x in next((o[5:] for o in _o if o.startswith("dsqr=")), "80:120:72:88").split(":")]
            # setmaxnreg only redistributes the CTA's launch allocation: per SMSP 5 warps x 96 (640 threads) = 480
            assert r_[0] + 2 * r_[1] + max(r_[2], 24) + r_[3] <= 480, f"dsqr {r_}: per-SMSP register sum > 480"
            assert all(x % 8 == 0 for x in r_), "setmaxnreg needs multiples of 8"
            registers = type(registers)(reduce=r_[0], compute=r_[1], load=r_[2], mma=r_[2], empty=24)
            self.num_regs_dsq = r_[3]
        self.dsq_sync_barrier = cutlass.pipeline.NamedBarrier(barrier_id=12, num_threads=4 * cute.arch.WARP_SIZE)

        # 16 warps -> 512 threads (dsqw: 20 warps -> 640)
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
                *self.dsq_warp_ids,
            )
        )
        self.dm_launch_regs = (65536 // self.threads_per_cta) // 8 * 8
        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        # self.epilogue_sync_barrier = pipeline.NamedBarrier(
        #     barrier_id=2,
        #     num_threads=self.num_compute_warps * self.threads_per_warp,
        # )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        # TMEM setup
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        # TMEM columns per accumulator. S^T and dP^T are (tile_n, tile_m) per CTA, so their
        # width is the MMA N = tile_m; with the M=64 layout every width halves.
        lane_rows = 2 if self.acc_m64 else 1
        self.tmem_cols_S = self.tile_m // lane_rows
        self.tmem_cols_dP = self.tile_m // lane_rows
        self.tmem_cols_dV = self.tile_hdimv // lane_rows
        self.tmem_cols_dK = self.tile_hdim // lane_rows
        # dQ = dS @ K is (tile_m, hdim); in 2CTA mode each CTA holds half of it.
        self.tmem_cols_dQ = self.tile_hdim // self.cta_group_size
        # P / dS are packed 16-bit over the S / dP columns (not in TMEM at all with acc_m64).
        # DM: P / dS are e4m3 (4 per 32-bit column)
        self.tmem_cols_P = 0 if self.acc_m64 else self.tmem_cols_S // 4
        self.tmem_cols_dS = 0 if self.acc_m64 else self.tmem_cols_dP // 4

        if self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL:
            assert self.tile_m == 128
            assert self.tile_n in (64, 128)
            self.tmem_dV_offset = 0
            self.tmem_dK_offset = self.tmem_dV_offset + self.tmem_cols_dV
            self.tmem_S_offset = self.tmem_dK_offset + self.tmem_cols_dK
            self.tmem_P_offset = self.tmem_S_offset  # overlap with S
            # dP overlaps the tail of S (S is released once read), dQ the tail of dP.
            self.tmem_dP_offset = self.tmem_alloc_cols - self.tmem_cols_dP
            self.tmem_dS_offset = self.tmem_dP_offset  # overlaps with dP
            self.tmem_dQ_offset = self.tmem_alloc_cols - self.tmem_cols_dQ
        else:
            self.tmem_S_offset = 0
            self.tmem_dV_offset = self.tmem_S_offset + self.tmem_cols_S
            self.tmem_dP_offset = self.tmem_dV_offset + self.tmem_cols_dV
            # 2CTA: dQ overlaps the tail of S (the dS pipeline signals S has been read before
            # dQ is issued); 1CTA: dQ overlaps dP.
            self.tmem_dQ_offset = (
                self.tmem_S_offset + self.tmem_cols_S - self.tmem_cols_dQ
                if self.use_2cta_instrs
                else self.tmem_dP_offset
            )
            self.tmem_dK_offset = self.tmem_dP_offset + self.tmem_cols_dP
            if self.split_P_dS:
                self.tmem_P_offset = self.tmem_dK_offset + self.tmem_cols_dK
                self.tmem_dS_offset = self.tmem_P_offset + self.tmem_cols_P
            else:
                self.tmem_P_offset = self.tmem_S_offset  # overlap with S
                self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP
        # DM sm100-mx scale-factor columns (2-CTA pipelined: S region = P[0,32) SF[32,64) dQ[64,128)).
        # Layouts: M=256 SFA row l -> lane l, col base + l/32 (per CTA, own rows);
        # SFB full-N replicated in every quadrant of both CTAs. One group = 4 cols, one group per k-step.
        # perf ablations (numerically WRONG, timing attribution only): comma list of
        # pamax (fixed P^T scale), dsamax (no dS amax / exchange), dqx (no dQ SF exchange wait)
        self.dm_abl = tuple(x for x in os.environ.get("DM_ABL", "").split(",") if x)
        # variant knobs for same-process A/B (bwd.py keys the compile cache on DM_BWD_OPTS)
        self.dm_opts = tuple(x for x in os.environ.get("DM_BWD_OPTS", "").split(",") if x)
        # DM d3nh (default in det; 'd3nh' forces it in nd, 'nod3nh' disables): an opaque per-iteration zero in the
        # MMA warp's smem descriptor starts keeps ptxas from hoisting every per-k-step 64-bit descriptor into the
        # uniform registers (UR exhaustion -> 69 STL/LDL in the 48-register MMA warp in det). Same MMAs: bitwise.
        self.dm_d3nh = "d3nh" in self.dm_opts or (self.deterministic and "nod3nh" not in self.dm_opts)
        # allocf (DEFAULT; 'noallocf' = DEBUG previous build): the TMEM-alloc hand-off gets the canonical tcgen05 fence pair
        # (PTX 9.7.18.6.4.4: tcgen05.alloc -> fence::before_thread_sync -> bar.sync -> fence::after_thread_sync -> ld.shared tmem_hold);
        # the cutlass TmemAllocator.wait_for_alloc barrier emitted neither (fwd / pass 0 already have both). Fences only: bitwise.
        self.dm_allocf = "noallocf" not in self.dm_opts
        # ordinv (DEFAULT ON; 'noordw' = DEBUG ONLY, the pre-ordinv
        # issue-order build). 2-CTA pipelined schedule: dV(t) reads its SFA/SFB scale
        # factors from TMEM cols in the S region (P 0..31, SF 32..51) and the next S^T MMA overwrites the S region; dK(t)
        # reads SFA/SFB_dk at dS_offset + 32.. (dP region) and the next dP^T MMA overwrites it. PTX ISA 9.7.18.6.2 pipelines
        # MMA A/metadata reads -> later MMA D writes, not scale-factor reads, so issue order alone is not a guarantee.
        # 'ordw': tcgen05.commit after dV / dK -> mbarrier ord[0] / ord[1]; the MMA warp waits it (+ fence::after) before
        # the next S^T / dP^T MMA. Same MMAs: bitwise. (The 1-CTA / serial schedules are not used at hd128: FA4 picks 2-CTA
        # for every hdim >= 128 problem; they are not covered.)
        _ow = self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED and "noordw" not in self.dm_opts
        assert not (_ow and "rdo" in self.dm_opts), "rdo (debug order) is not ordinv-ordered: use it with noordw"
        self.dm_ordw_v = _ow
        self.dm_ordw_k = _ow
        # ordinv planted controls (DEBUG ONLY): ordpc1 / ordpc3 -> the MMA warp overwrites SFA_dv (S region col 32) / SFA_dk
        # (dP region col dS+32) of its lanes with tcgen05.st (+ wait::st) right before S^T(t+1) / dP^T(t+1), i.e. where the
        # next MMA's overwrite lands; caught unless ordw orders it after dV(t) / dK(t) completed.
        self.dm_ordpc = 1 if "ordpc1" in self.dm_opts else (3 if "ordpc3" in self.dm_opts else (2 if "ordpc2" in self.dm_opts else 0))
        # ordpc2: SFA_dv poison right after dV(t) issue (after dV(t)'s completion wait with ordw): B1's window is closed in
        # practice by the dQ-empty wait before S^T(t+1) (ordpc1 never fires), so ordpc2 shows the harness sees a dV SF overwrite
        self.dm_abl = self.dm_abl + tuple(x[4:] for x in self.dm_opts if x.startswith("abl:"))
        # DM dsq: the dS tile amax / pow2 scale / e4m3 quantization / dS code+SF stores /
        # dQ SF word + peer dS exchange move from the 8 compute warps to the 4 reduce (dQ drain) warps. Reduce warp q owns
        # TMEM lanes 32q..32q+31 = 32 keys = exactly one dS scale block (32 keys x 128 queries): no cross-WG amax exchange.
        # The compute warps store the exact fp32 dS in place over dP and arrive. Same values bitwise.
        self.dm_dsq = "dsq" in self.dm_opts or "dsqw" in self.dm_opts
        _p5 = next((x[3:] for x in self.dm_opts if x.startswith("p5=")), None)   # p5=freq:res
        self.dm_p5 = tuple(int(t) for t in _p5.split(":")) if _p5 else None
        # fws (default on): first-writer-stores dQ accumulation (no dq_accum zero-fill); nofws disables
        self.dm_fws = "nofws" not in self.dm_opts
        # rw8k tconv: tail converter (tailconv.py) launched with PDL after this kernel: launch_dependents after the TMEM
        # allocation, fence.proxy.async.global before every det semaphore release (the converter reads the accumulators
        # with generic loads after an ld.acquire of the semaphore). No arithmetic change.
        self.dm_tconv = "tconv" in self.dm_opts
        self.dm_tcfence = self.dm_tconv and "tcnofence" not in self.dm_opts   # bisect knobs (timing only)
        self.dm_tcpush = self.dm_tconv and "tcnopush" not in self.dm_opts
        # pers=<CTAs>: persistent grid of that many CTAs (= SM count, a cluster multiple), 0 = single-tile CTAs
        self.dm_pers = int(next((x[5:] for x in self.dm_opts if x.startswith("pers=")), "0"))
        # pdyn: dynamic (atomic-counter) persistent claim order instead of static grid-stride (needs pers=N)
        self.dm_pdyn = self.dm_pers > 0 and "pdyn" in self.dm_opts and self.qhead_per_kvhead == 1
        # DM gloop (GQA): one CTA (pair) per (KV head, KV tile) loops over the qh query heads of the group (single
        # m-block stream of qh * L blocks, head-major); dK/dV stay in TMEM across the group and are stored once
        # (no fp32 dK/dV accumulation, no dK/dV semaphores, no dKV postprocess). Order of the dK/dV fp32 sums:
        # query head g = 0..qh-1 outer, m_block ascending inner (one TMEM accumulator).
        self.dm_gloop = self.qhead_per_kvhead > 1 and not self.pack_gqa and "gloop" in self.dm_opts
        self.dm_G = self.qhead_per_kvhead if self.dm_gloop else 1
        _p3 = next((x[3:] for x in self.dm_opts if x.startswith("p3=")), None)   # p3=freq:res (deg-3)
        self.dm_p3 = tuple(int(t) for t in _p3.split(":")) if _p3 else None
        self.dm_b6 = int(next((x[3:] for x in self.dm_opts if x.startswith("b6=")), "0"))   # deg-6 poly exp blocks
        self.dm_sfa_dv_col = self.tmem_P_offset + self.tmem_cols_P            # 4 groups x 4 cols
        self.dm_sfb_dv_col = self.dm_sfa_dv_col + 16                         # 4 cols (const 1.0)
        self.dm_sfa_dq_col = self.dm_sfb_dv_col + 4                          # 2 cols (const 1.0)
        self.dm_sfb_dq_col = self.dm_sfa_dq_col + 2                          # 2 groups x 2 cols
        self.dm_sfa_dk_col = self.tmem_dS_offset + self.tmem_cols_dS         # 4 groups x 4 cols
        self.dm_sfb_dk_col = self.dm_sfa_dk_col + 16                         # 4 cols (const 1.0)
        assert self.dm_sfb_dq_col + 4 <= self.tmem_S_offset + self.tmem_cols_S - self.tmem_cols_dQ or not self.use_2cta_instrs
        tmem_end = max(
            self.tmem_S_offset + self.tmem_cols_S,
            self.tmem_dP_offset + self.tmem_cols_dP,
            self.tmem_dV_offset + self.tmem_cols_dV,
            self.tmem_dK_offset + self.tmem_cols_dK,
            self.tmem_dQ_offset + self.tmem_cols_dQ,
            self.tmem_P_offset + self.tmem_cols_P,
            self.tmem_dS_offset + self.tmem_cols_dS,
        )
        assert tmem_end <= self.tmem_alloc_cols, (
            f"TMEM map needs {tmem_end} > {self.tmem_alloc_cols} columns"
        )

        # BwdConfig.registers carries the split; config.validate_bwd_registers checks it.
        self.num_regs_reduce = registers.reduce
        self.num_regs_compute = registers.compute
        self.num_regs_load = registers.load
        self.num_regs_mma = registers.mma
        self.num_regs_empty = registers.empty
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.buffer_align_bytes = 1024

    @staticmethod
    def accum_tile_orders(
        use_2cta_instrs: bool, tile_n: int
    ) -> tuple[AccumTileOrder, AccumTileOrder]:
        """(dQ, dK / dV) accumulate tile orders of a configuration; the interface compiles the
        postprocess for the same orders. dQ is reduced out of its TMEM fragment, per CTA pair
        with 2CTA. GQA dK / dV come out of the M=128 epilogue in fragment order and out of the
        M=64 one (2CTA, 64-row KV tile, NOTE [M=64 accumulator layout]) row-major."""
        acc_m64 = use_2cta_instrs and tile_n == 64
        dQaccum_order = (
            AccumTileOrder.TMEM_FRAG_2CTA if use_2cta_instrs else AccumTileOrder.TMEM_FRAG_1CTA
        )
        dKVaccum_order = AccumTileOrder.ROW_MAJOR if acc_m64 else AccumTileOrder.TMEM_FRAG_1CTA
        return dQaccum_order, dKVaccum_order

    def _check_accum_tile_orders(self):
        """The reduce epilogues stream fixed-width column stages of a TMEM fragment with 1-D
        copies (dQacc_reduce, epilogue_dK_or_dV_reduce_ncol); check on sample coordinates that
        accum_tile_layout(order) describes the storage order those copies produce. The M=64
        dK / dV epilogue derives its addresses from the layout instead (epilogue_dK_or_dV)."""
        rows, hdim = self.tile_m, self.tile_hdim
        assert len(self.reduce_warp_ids) == 4, "the 1-D copies below assume 128 threads"
        # dQ: CTA rank r streams stage s (ncol columns of its fragment) as the flat block
        # (s + stages_per_cta * r) * 128 * ncol, thread t holding lane t, 4 columns per copy.
        # Lane t is row t with one CTA; with two it is row t % 64 of CTA r's 64 rows, column
        # half t // 64.
        dQ_layout = accum_tile_layout(self.dQaccum_order, rows, hdim)
        ncol = self.dQ_reduce_ncol
        stages_per_cta = self.dQaccum_reduce_stage // self.cta_group_size
        for r in range(self.cta_group_size):
            for s in range(stages_per_cta):
                for t in (0, 1, 63, 64, 127):
                    for j in (0, 1, 3, 4, ncol - 1):
                        flat = (
                            (s + stages_per_cta * r) * 128 * ncol + 4 * t + j % 4 + 512 * (j // 4)
                        )
                        if self.cta_group_size == 1:
                            row, col = t, s * ncol + j
                        else:
                            row, col = t % 64 + 64 * r, s * ncol + j + hdim // 2 * (t // 64)
                        assert cute.crd2idx((row, col), dQ_layout) == flat, (
                            "dQaccum_order does not match the dQ reduce"
                        )
        if self.dKVaccum_order == AccumTileOrder.ROW_MAJOR:
            # M=64 dK / dV: the tile goes out as two row halves through a half-tile staging
            # buffer, so each half must be one block laid out like a half-height tile.
            rows_per_half = self.tile_n // 2
            for hd in (self.tile_hdim, self.tile_hdimv):
                tile_layout = accum_tile_layout(self.dKVaccum_order, self.tile_n, hd)
                half_layout = accum_tile_layout(self.dKVaccum_order, rows_per_half, hd)
                for half in range(2):
                    for r in (0, 1, rows_per_half - 1):
                        for c in (0, 1, 4, hd - 1):
                            assert cute.crd2idx((half * rows_per_half + r, c), tile_layout) == (
                                half * rows_per_half * hd + cute.crd2idx((r, c), half_layout)
                            ), "dKVaccum_order must store each row half of the tile as one block"
        else:
            assert self.dKVaccum_order == AccumTileOrder.TMEM_FRAG_1CTA
            # M=128 dK / dV: warpgroup w owns columns [w * hdim / 2, (w + 1) * hdim / 2) of
            # the tile_n rows as the flat block w * tile_n * hdim / 2, streamed as
            # dK_reduce_ncol-column stages, thread t holding row t, 4 columns per copy.
            assert self.tile_n == 128
            ncol = self.dK_reduce_ncol
            for hd in (self.tile_hdim, self.tile_hdimv):
                layout = accum_tile_layout(self.dKVaccum_order, self.tile_n, hd)
                for w in range(2):
                    for s in range(hd // 2 // ncol):
                        for t in (0, 1, 127):
                            for j in (0, 1, 3, 4, ncol - 1):
                                flat = (
                                    w * self.tile_n * hd // 2
                                    + s * self.tile_n * ncol
                                    + 4 * t
                                    + j % 4
                                    + 512 * (j // 4)
                                )
                                col = w * hd // 2 + s * ncol + j
                                assert cute.crd2idx((t, col), layout) == flat, (
                                    "dKVaccum_order does not match the dK / dV reduce"
                                )

    def _setup_attributes(self):
        self.single_stage = 1
        self.P_stage = 1 if self.split_P_dS else 0  # dedicated P pipeline, else P aliases S
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        self.sdKVaccum_stage = 2
        # number of tma reduce adds per dQacc mma
        # todo: try 32/1 or 48/2 for 2cta d=192 dv=128
        if self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL:
            self.dQ_reduce_ncol_t2r = 32
            if self.tile_hdim == 192:
                # 24 tiles the 96 columns per CTA
                self.dQ_reduce_ncol = 24 if not self.is_causal else 32
                self.sdQaccum_stage = 2 if not self.is_causal else 1
            else:
                # hdim 256: 1 KB under the SMEM limit (the separate sKt tile for the cluster-wide
                # dQ reduction costs 32 KB), so 16 KB of staging as two 16-column stages, which
                # measured faster than one 32-column stage.
                assert self.tile_hdim == 256
                self.dQ_reduce_ncol = 16
                self.sdQaccum_stage = 2
        else:
            if self.use_2cta_instrs:
                self.dQ_reduce_ncol = 16 if self.deterministic else 8
                self.sdQaccum_stage = 2 if self.deterministic else 4
                self.dQ_reduce_ncol_t2r = 32
                # DM dq32 / dq32s1: 32-column staging chunks (half the per-tile r2s/barrier/bulk-add rounds),
                # 2 stages (+16 KB smem) or 1 stage (same smem as today); same adds per element, same order
                _dqo = tuple(x for x in os.environ.get("DM_BWD_OPTS", "").split(",") if x)
                if "dq32" in _dqo:
                    self.dQ_reduce_ncol, self.sdQaccum_stage = 32, 2
                # DM dqnd nd default 32x2 is OFF in the composed build: on top of dsqw it
                # measured 0.998x (leave-one-out, 8 shapes) - the helper WG already removed what it saved. Opt in with 'dq32'.
                elif "dq32s1" in _dqo:
                    self.dQ_reduce_ncol, self.sdQaccum_stage = 32, 1
                # DM dqc=<ncol>,dqs=<stages>: explicit staging geometry; same adds per element
                _dqc = next((int(x[4:]) for x in _dqo if x.startswith("dqc=")), 0)
                _dqs = next((int(x[4:]) for x in _dqo if x.startswith("dqs=")), 0)
                if _dqc and not self.deterministic:   # nd only (det's delayed release allows <= 2 stages)
                    self.dQ_reduce_ncol = _dqc
                if _dqs and not self.deterministic:
                    self.sdQaccum_stage = _dqs
            else:
                self.dQ_reduce_ncol = 32
                self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
                self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol
        assert (self.tile_hdim // self.cta_group_size) % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.dQaccum_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
        self.cluster_reduce_dQ = False and cute.size(self.cluster_shape_mn) > 1
        # number of tma reduce adds for dKacc and dVacc epilogue (must divide hdim_per_wg)
        self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)
        self._check_accum_tile_orders()
        # CTA group for MMA operations
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

    def _get_tiled_mma(self):
        # S.T = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_kq[:2],
        )
        # dP.T = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            self.do_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P.T @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            self.do_dtype,
            cute.nvgpu.OperandMajorMode.K,  # P_major_mode
            cute.nvgpu.OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.SMEM if self.acc_m64 else tcgen05.OperandSource.TMEM,
        )
        # dK += dS.T @ Q
        if const_expr(self.use_smem_dS_for_mma_dK or self.acc_m64):
            mma_dK_a_src = tcgen05.OperandSource.SMEM
        else:
            mma_dK_a_src = tcgen05.OperandSource.TMEM
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            self.do_dtype,
            cute.nvgpu.OperandMajorMode.K,  # dS_major_mode
            cute.nvgpu.OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsq[:2],
            a_source=mma_dK_a_src,
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            self.k_dtype,
            cute.nvgpu.OperandMajorMode.MN,  # dS_major_mode
            cute.nvgpu.OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S.T = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP.T = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dV += P.T @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        if self.acc_m64:
            self._check_dS_exchange_layouts()
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))

        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdK_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        self.sdV_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdimv // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        # headdim_64 gets 1 stage
        self.num_epi_stages = max(1, (self.tile_hdim // 2) // self.sdK_epi_tile[1])
        self.num_epi_stages_v = max(1, (self.tile_hdimv // 2) // self.sdV_epi_tile[1])
        self.sdK_flat_epi_tile = self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages
        self.sdV_flat_epi_tile = self.tile_n * (self.tile_hdimv // 2) // self.num_epi_stages_v
        if const_expr(not self.dKV_postprocess):
            # One (tile_n, epi) stage per compute warpgroup; with the M=64 layout the whole CTA
            # stages the (tile_n, hdim) tile, i.e. hdim / epi stages (the same bytes for hdim 128).
            self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdK_epi_tile,
                self.tile_hdim // self.sdK_epi_tile[1] if self.acc_m64 else 2,
            )
            self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dv_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdV_epi_tile,
                self.tile_hdimv // self.sdV_epi_tile[1] if self.acc_m64 else 2,
            )
        elif self.acc_m64:
            # fp32 row-major staging of half the (tile_n, hdim) tile (NOTE [M=64 accumulator
            # layout] GQA path): all of the K / V storage.
            self.sdK_layout = cute.make_layout((self.tile_n // 2) * self.tile_hdim)
            self.sdV_layout = cute.make_layout((self.tile_n // 2) * self.tile_hdimv)
        else:
            self.sdK_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))
            # self.dK_reduce_ncol same for dV
            self.sdV_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        # Block-sparse tensors (Q direction - for iterating m_blocks per n_block):
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        tensormap_workspace: Optional[
            cute.Tensor
        ] = None,  # (num_ctas, 3, 16) Int64: K, V, Kt descriptors per CTA
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        self.is_varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        # dK / dV stored directly (MHA, or pack_gqa keeping them in TMEM) with a varlen K take
        # the predicated non-TMA store path.
        self.use_tma_store = not (
            (self.qhead_per_kvhead == 1 or self.pack_gqa) and mCuSeqlensK is not None
        )
        # DM dkvdir: MHA dK/dV stored straight from registers to global (no sdK/sdV smem staging, which aliases sK/sV)
        self.dm_dkvdir = "dkvdir" in self.dm_opts and self.qhead_per_kvhead == 1 and not self.pack_gqa
        if const_expr(self.dm_dkvdir):
            self.use_tma_store = False
        # self.use_tma_store = not self.qhead_per_kvhead == 1
        self.dKV_postprocess = self.qhead_per_kvhead > 1 and not self.pack_gqa and not self.dm_gloop

        if const_expr(self.dKV_postprocess):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision for GQA"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision for GQA"

        mdQaccum, mdK, mdV, mLSE, mdPsum = [
            assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV, mLSE, mdPsum)
        ]

        # (b, s, n, h) --> (s, h, n, b) or (t, n, h) -> (t, h, n)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]

        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]

        # (b, n, s) --> (s, n, b) or (n, t) --> (t, n)
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, mode=LSE_dPsum_dQaccum_transpose)
            for t in (mLSE, mdPsum, mdQaccum)
        ]

        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = [2, 1, 0] if const_expr(mCuSeqlensK is None) else [1, 0]
        mdK, mdV = [layout_utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)]
        if const_expr(self.pack_gqa):
            # (s, h, n, b) -> ((qh, s), h, n_kv, b), NOTE [bwd pack_gqa]
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mdO = pack_gqa_layout(mdO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            # LSE / dPsum / dQaccum are allocated per KV head in packed row order already
        # (s, h, n, b) --> (h, s, n, b) or (t, h, n) -> (h, t, b)
        dO_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        mdO = layout_utils.select(mdO, mode=dO_transpose)

        # Transposes for 2-CTA K/Q paths (Q follows Q seqlens, K follows K seqlens)
        transpose_sh_q = dO_transpose
        transpose_sh_k = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]

        # (b, n, block, stage) -> (block, stage, n, b)
        semaphore_transpose = [2, 3, 1, 0]
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=semaphore_transpose)

        if const_expr(self.dKV_postprocess and self.deterministic):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=semaphore_transpose)
                for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(not self.dKV_postprocess):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != cute.nvgpu.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != cute.nvgpu.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(self.use_tma_store and not self.dKV_postprocess):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(self.sdK_layout, mode=[0, 1]),
                self.sdK_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdV_layout, mode=[0, 1]),
                self.sdV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        if const_expr(not self.dKV_postprocess):
            thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))  # 128 threads
            val_layout_r2s_dKV = cute.make_ordered_layout(
                (1, 128 // self.dk_dtype.width), order=(1, 0)
            )  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
                copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
            )
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(
                Float32, 128, num_copy_elems=128 // Float32.width
            )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # Bound the K and V TMA loads by the logical seqlen_k (cu_seqlens_k and/or seqused_k) so
        # rows in [seqlen_k, tile_n) arrive zero-filled instead of as whatever the buffer holds.
        # The masked P / dS are exactly zero there, but 0 * NaN = NaN in dP = V @ dO.T and
        # dQ = dS @ K. The load warp rewrites its CTA's copies of the K, V (and Kt) descriptors
        # per work tile with the sequence's row extent (tensormap.replace); TMA then zero-fills
        # rows at or past the end and never touches memory beyond it. A descriptor with a
        # per-launch extent cannot do this: TMA prefetches past the footprint it is given (#3068).
        self.ragged_kv_tma = mCuSeqlensK is not None or mSeqUsedK is not None
        if const_expr(self.ragged_kv_tma):
            assert tensormap_workspace is not None, (
                "ragged K/V TMA loads need a tensormap workspace"
            )
        mK_tma, mV_tma = mK, mV
        mK_gmem, mV_gmem = (mK, mV) if const_expr(self.ragged_kv_tma) else (None, None)
        mKt_gmem = None
        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK_tma,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV_tma,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        # dV = P.T @ dO
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )
        # ------------------------------------------------------------
        # 2-CTA
        # ------------------------------------------------------------
        tma_atom_dOt = tma_tensor_dOt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op,
                layout_utils.select(mdO, mode=transpose_sh_q),
                cute.select(self.sdOt_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo,
                self.tiled_mma_dP,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Qt = tma_tensor_Qt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op,
                layout_utils.select(mQ, mode=transpose_sh_q),
                cute.select(self.sQt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq,
                self.tiled_mma_dK,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Kt = tma_tensor_Kt = None
        if const_expr(self.use_2cta_instrs):
            Kt_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_dQ.thr_id
            )
            mKt_tma = layout_utils.select(mK, mode=transpose_sh_k)
            if const_expr(self.ragged_kv_tma):
                mKt_gmem = mKt_tma  # (hdim, seqlen_k, ...): the rows are mode 1
            tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                Kt_tma_op,
                mKt_tma,
                cute.select(self.sKt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsk,
                self.tiled_mma_dQ,
                self.cluster_layout_vmnk.shape,
            )

        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)
        self.tma_copy_bytes["sdS_xchg"] = self.tma_copy_bytes["dS"] // 2  # Half of dS for exchange

        # TileScheduler = SingleTileScheduler
        if const_expr(self.is_varlen_k):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic and "dschedst" in self.dm_opts):
            # DM dschedst (det, opt-in): the nd SingleTileScheduler (grid order, no LPT / head swizzle). Bitwise: the dQ and
            # dK/dV summation orders are fixed by the semaphore lock values, not by the schedule.
            TileScheduler = SingleTileScheduler
        elif const_expr(self.dm_gloop and self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        elif const_expr(self.dm_gloop):
            TileScheduler = SingleTileScheduler
        elif const_expr(self.deterministic and "noposmaj" not in self.dm_opts and self.qhead_per_kvhead > 1
                        and not self.pack_gqa and self.cta_group_size > 1):
            TileScheduler = (dm_posmajor_once_scheduler(self.qhead_per_kvhead) if const_expr("pm1" in self.dm_opts)
                             else dm_posmajor_scheduler(self.qhead_per_kvhead))
        elif const_expr(self.deterministic and self.dm_pers > 0 and not self.pack_gqa):
            if const_expr(self.dm_pdyn):
                TileScheduler = dm_dynamic_scheduler(self.qhead_per_kvhead, "nohswz" not in self.dm_opts, self.dm_pers,
                                                     2 * (len(self.reduce_warp_ids) + len(self.compute_warp_ids) + 3))
            else:
                TileScheduler = dm_persistent_scheduler(self.qhead_per_kvhead, "nohswz" not in self.dm_opts,
                                                        self.dm_pers, "psnake" in self.dm_opts)
        elif const_expr(self.deterministic and "nohswz" not in self.dm_opts and self.qhead_per_kvhead > 1
                        and not self.pack_gqa):
            TileScheduler = dm_hswz_scheduler(self.qhead_per_kvhead)
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            # work tiles of C * tile_n KV rows, see NOTE [SM100 backward work coordinates]
            cute.ceil_div(cute.size(mK.shape[0]), self.cluster_tile_n),  # num_blocks
            cute.size(mQ.shape[2]) // (self.qhead_per_kvhead if const_expr(self.dm_gloop or (
                self.deterministic and "noposmaj" not in self.dm_opts and "dschedst" not in self.dm_opts
                and self.qhead_per_kvhead > 1 and not self.pack_gqa
                and self.cta_group_size > 1)) else 1),  # num_heads = num_query_heads (posmaj: KV heads)
            cute.size(mK.shape[3])
            if const_expr(mCuSeqlensK is None)
            else cute.size(mCuSeqlensK.shape[0] - 1),  # num_batches
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0])  # pass total_k for total_q
            if const_expr(mCuSeqlensK is not None)
            else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],  # (tile_n, tile_m)
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            use_cluster_idx=self.cta_group_size > 1,
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            # the backward tiles KV rows: the Q-row packing factor is for block_info / mask only
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,  # persistent mode not tested
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            lpt=self.spt,
            head_swizzle=self.deterministic and "dschedst" not in self.dm_opts,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdK_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
        )

        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdK_layout)
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"
        # 2-CTA: sdV reuses sV, sdK reuses sK
        sV_bytes = cute.size_in_bytes(self.v_dtype, self.sV_layout)
        sK_bytes = cute.size_in_bytes(self.k_dtype, self.sK_layout)
        # DM: fp8 K / V are smaller than the bf16 sdK / sdV epilogue tiles that alias them (2-CTA)
        dm_sK_alloc_bytes = max(sK_bytes, sdK_bytes) if const_expr(self.use_2cta_instrs) else sK_bytes
        dm_sV_alloc_bytes = max(sV_bytes, sdV_bytes) if const_expr(self.use_2cta_instrs) else sV_bytes

        if const_expr(self.use_2cta_instrs):
            # The serial schedule aliases Qt / dOt into Q / dO and sdS_xchg into sdQaccum; the
            # M=64 layout has no exchange staging buffer at all (see xchg_aliases_dQaccum).
            own_xchg = const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED)
            sQt_size = cute.cosize(self.sQt_layout) if own_xchg else 0
            sdOt_size = cute.cosize(self.sdOt_layout) if own_xchg else 0
            sdS_xchg_size = (
                cute.cosize(self.sdS_xchg_layout) if own_xchg and self.has_sdS_xchg else 0
            )
            # M=64 layout: P^T and this CTA's dS^T are the SMEM A operands of the dV / dK MMAs.
            sP_size = cute.cosize(self.tP_layout) if const_expr(self.acc_m64) else 0
            sdS_dK_size = cute.cosize(self.sdSt_layout) if const_expr(self.acc_m64) else 0
            if const_expr(self.xchg_aliases_dQaccum):
                assert cute.size_in_bytes(
                    self.ds_dtype, self.sdS_xchg_layout
                ) <= cute.size_in_bytes(self.dqaccum_dtype, self.sdQaccum_layout), (
                    "sdS_xchg does not fit in the sdQaccum storage it aliases"
                )

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar: cutlass.Int64

                # 2-CTA
                Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_cluster_empty_mbar_ptr: cutlass.Int64
                dS_cluster_full_mbar_ptr: cutlass.Int64
                dS_cluster_leader_mbar_ptr: cutlass.Int64
                dm_xchg_mbar: cute.struct.MemRange[cutlass.Int64, 11 + (2 if (self.dm_ordw_v or self.dm_ordw_k) else 0)]
                dQaccum_empty_mbar_ptr: cutlass.Int64

                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, dm_sK_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, dm_sV_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                    self.buffer_align_bytes,
                ]
                sQt: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, sQt_size],
                    self.buffer_align_bytes,
                ]
                sdOt: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sdOt_size],
                    self.buffer_align_bytes,
                ]
                sdS_xchg: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_xchg_size],
                    self.buffer_align_bytes,
                ]
                sP: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sP_size],
                    self.buffer_align_bytes,
                ]
                sdS_dK: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_dK_size],
                    self.buffer_align_bytes,
                ]
                sKt: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sKt_layout)],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    self.buffer_align_bytes,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]
                sDm: cute.struct.Align[cute.struct.MemRange[Float32, 64], 128]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes if sdS_xchg_size == 0 else 128,
                ]

        else:

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.P_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar: Int64

                sQ: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, dm_sK_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, dm_sV_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    128,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]
                sDm: cute.struct.Align[cute.struct.MemRange[Float32, 64], 128]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            # Without score_mod: bake scale into log2
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            # With score_mod: score_mod applied to S * softmax_scale, then use LOG2_E only
            softmax_scale_log2 = LOG2_E

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if const_expr(aux_data.tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // self.pack_factor
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisorV2(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisorV2(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if const_expr(self.use_block_sparsity and self.use_2cta_instrs):
            # Both CTAs of a cluster must map to the same sparse KV column or they deadlock.
            assert self.kv_subtile_factor % self.cta_group_size == 0, (
                "2-CTA block-sparse backward requires kv_subtile_factor % cta_group_size == 0"
            )

        # 2-CTA: 231424 and 1-CTA: 232448
        # print("SMEM: ", self.shared_storage.size_in_bytes())
        if const_expr(self.use_block_sparsity):
            assert all(x is None for x in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)), (
                "Variable sequence length is not supported yet for blocksparse in bwd"
            )

        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qt,
            tma_tensor_K,
            tma_tensor_Kt,
            tma_tensor_V,
            mK_gmem,
            mV_gmem,
            mKt_gmem,
            tensormap_workspace if const_expr(self.ragged_kv_tma) else None,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            tma_tensor_dOt,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_Qt,
            tma_atom_K,
            tma_atom_Kt,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dOt,
            tma_atom_dV,
            tma_atom_dK,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdS_xchg_layout,
            self.sdQaccum_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            tile_sched_params,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    def _generate_attention_mask_cls(self, window_size_left, window_size_right):
        return partial(
            AttentionMask,
            self.tile_m,
            self.cluster_tile_n,
            swap_AB=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.pack_factor,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mK_gmem: Optional[cute.Tensor],  # gmem tensors behind tma_atom_K/V/Kt (ragged K/V only)
        mV_gmem: Optional[cute.Tensor],
        mKt_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],  # (num_ctas, 3, 16) Int64 (ragged K/V only)
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: Optional[cute.Tensor],
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdS_xchg_layout: cute.Layout,
        sdQaccum_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tile_sched_params: ParamsBase,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        dm_utils.tcf_begin(0 if "notcf" in self.dm_opts else 2 if "tcfb" in self.dm_opts else 1)   # tcgen05 thread-sync fences (dm_utils.TCF)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        if const_expr("tl2" in self.dm_opts and aux_data.tensors is not None and len(aux_data.tensors) > 4):
            # DM tl2: kernel-entry globaltimer per CTA at [64 + 8 * bid + 4]
            if cute.arch.thread_idx()[0] == 0:
                dm_utils.red_add_u64(aux_data.tensors[4].iterator.toint() + 8 * (64 + 8 * bidx + 4), dm_utils.globaltimer())
        # The CTA rank of a 2CTA pair; NOTE [SM100 backward coordinate spaces].
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                if const_expr(tma_atom_Qt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Qt)
                cpasync.prefetch_descriptor(tma_atom_K)
                if const_expr(tma_atom_Kt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Kt)
                cpasync.prefetch_descriptor(tma_atom_V)
                if const_expr(tma_atom_dOt is not None):
                    cpasync.prefetch_descriptor(tma_atom_dOt)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        dQ_cluster_full_mbar_ptr = storage.dQ_cluster_full_mbar_ptr.data_ptr()
        dQ_cluster_empty_mbar_ptr = storage.dQ_cluster_empty_mbar_ptr.data_ptr()

        if const_expr(self.use_2cta_instrs):
            dS_cluster_full_mbar_ptr = storage.dS_cluster_full_mbar_ptr
            dS_cluster_empty_mbar_ptr = storage.dS_cluster_empty_mbar_ptr
            dS_cluster_leader_mbar_ptr = storage.dS_cluster_leader_mbar_ptr
        else:
            dS_cluster_full_mbar_ptr = None
            dS_cluster_empty_mbar_ptr = None
            dS_cluster_leader_mbar_ptr = None
        # The dQaccum_empty handshake exists only where sdS_xchg aliases sdQaccum; the roles
        # specialize on this handle (and on sdS_xchg) rather than on the configuration.
        if const_expr(self.xchg_aliases_dQaccum):
            dQaccum_empty_mbar_ptr = storage.dQaccum_empty_mbar_ptr
        else:
            dQaccum_empty_mbar_ptr = None

        # Barrier initialization
        if const_expr(self.use_2cta_instrs):
            if const_expr(dQaccum_empty_mbar_ptr is not None):
                if warp_idx == 2:
                    cute.arch.mbarrier_init(
                        dQaccum_empty_mbar_ptr,
                        len(self.reduce_warp_ids),
                    )
            if warp_idx == 4:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_empty_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)
                # DM: dQ scale-factor word exchange with the peer CTA (one per tile parity)
                cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr(), 1)
                cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 1, 1)
                # DM pers: [2] = this CTA's dK/dV epilogue has drained sK/sV (sdK/sdV alias them): next tile may load K/V
                cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 2, 1)
                # DM pdyn: [3, 4] tile-slot full (1 arrival + 4 tx on the peer), [5, 6] empty (every role warp of both
                # CTAs, arrived on the leader's copy), [7, 8] the claimed cluster tile index (low word)
                cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 3, 1)
                cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 4, 1)
                cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 5, 2 * (len(self.reduce_warp_ids) + len(self.compute_warp_ids) + 3))
                cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 6, 2 * (len(self.reduce_warp_ids) + len(self.compute_warp_ids) + 3))
                # DM dsq: [9] fp32 dS of tile t stored (one arrival per compute warp), [10] S(t+1) read by every compute
                # warp (dQ(t) may overwrite the S tail; the reduce warps then commit pipeline_dS(t))
                cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 9, len(self.compute_warp_ids))
                cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 10, len(self.compute_warp_ids))
                if const_expr(self.dm_ordw_v or self.dm_ordw_k):
                    # ordinv ordw: [11] dV(t) complete, [12] dK(t) complete (tcgen05.commit by the leader's MMA warp)
                    cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 11, 1)
                    cute.arch.mbarrier_init(storage.dm_xchg_mbar.data_ptr() + 12, 1)

        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                for i in range(self.dQaccum_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids, *self.dsq_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.cta_group_size
        )
        # With split_P_dS, this pipeline guards only the S TMEM buffer (P has pipeline_P).
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            # DM dsq: the dS codes that overwrite dP are written by the reduce warps, which release dP -> dS
            consumer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.reduce_warp_ids) * self.cta_group_size)
            if const_expr(self.dm_dsq) else pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.reduce_warp_ids) * self.cta_group_size)
            if const_expr(self.dm_dsq) else pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        if const_expr(self.split_P_dS):
            pipeline_P = cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=pipeline_PdS_producer_group,
                consumer_group=pipeline_PdS_consumer_group,
                barrier_storage=storage.P_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
            )
        else:
            pipeline_P = None

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]) * self.num_mcast_ctas_b
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        if const_expr(self.use_2cta_instrs):
            if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL):
                pipeline_Qt = pipeline_Q
            else:
                pipeline_Qt = pipeline.PipelineTmaUmma.create(
                    barrier_storage=storage.Qt_mbar_ptr.data_ptr(),
                    num_stages=self.Q_stage,
                    producer_group=pipeline_producer_group,
                    consumer_group=pipeline_consumer_group,
                    tx_count=self.tma_copy_bytes["Q"],
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
            pipeline_Kt = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.Kt_mbar_ptr.data_ptr(),
                num_stages=self.single_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_Qt = pipeline_Kt = pipeline_Q

        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        # mark the tcgen05 hand-off pipelines / barriers (dm_utils.tcf_mark)
        dm_utils.tcf_mark("ua", pipeline_S_P, pipeline_dP, pipeline_dKV, pipeline_dQ)
        dm_utils.tcf_mark("au", pipeline_dS, pipeline_P)
        dm_utils.tcf_mark("bar", self.compute_sync_barrier, tmem_alloc_barrier)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
            sQt = storage.sQt.get_tensor(
                sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype
            )
        else:
            sQt = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype), sQt_layout.outer
            )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner, dtype=self.k_dtype)
        if const_expr(self.use_2cta_instrs):
            sKt = storage.sKt.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
        else:
            sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=self.v_dtype)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)
        if const_expr(self.has_sdS_xchg):
            if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
                sdS_xchg = storage.sdS_xchg.get_tensor(sdS_xchg_layout)
            else:
                sdS_xchg = storage.sdQaccum.get_tensor(sdS_xchg_layout, dtype=self.ds_dtype)
        else:
            sdS_xchg = None
        if const_expr(self.acc_m64):
            # P^T and this CTA's dS^T: the SMEM A operands of the dV / dK MMAs.
            sP = storage.sP.get_tensor(
                tP_layout.outer, swizzle=tP_layout.inner, dtype=self.do_dtype
            )
            sdS_dK = storage.sdS_dK.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        else:
            sP = None
            sdS_dK = None

        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
            sdOt = storage.sdOt.get_tensor(
                sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.do_dtype
            )
        else:
            sdOt = cute.make_tensor(
                cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
                sdOt_layout.outer,
            )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        sDm = storage.sDm.get_tensor(cute.make_layout(64))
        if const_expr(self.use_2cta_instrs):
            if const_expr(not self.dKV_postprocess):
                sdV = storage.sV.get_tensor(
                    sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
                )
                sdK = storage.sK.get_tensor(
                    sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
                )
            else:
                sdV = storage.sV.get_tensor(sdV_layout, dtype=self.dv_dtype)
                sdK = storage.sK.get_tensor(sdK_layout, dtype=self.dk_dtype)
        elif const_expr(not self.dKV_postprocess):
            sdV = storage.sdO.get_tensor(
                sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
            )
            sdK = storage.sQ.get_tensor(
                sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
            )
        else:
            sdV = storage.sdO.get_tensor(sdV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdK_layout, dtype=self.dk_dtype)

        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # S
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype), tP_layout.outer
        )
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer
        )
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        block_info = BlockInfo(
            self.tile_m,
            self.cluster_tile_n,  # m_block ranges are per work tile
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.pack_factor,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            # With ragged K/V TMA tensors mode 0 is a sentinel extent, but cu_seqlens_k or
            # seqused_k is then always present and the static length is unused.
            seqlen_k_static=mK.shape[0] if const_expr(not self.ragged_kv_tma) else Int32(0),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.cluster_tile_n,
        )
        if const_expr(self.dm_pdyn):
            dm_sbase = storage.dm_xchg_mbar.data_ptr().toint()
            TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params, dm_sbase)
        else:
            TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        AttentionMaskCls = self._generate_attention_mask_cls(window_size_left, window_size_right)
        self.dm_kvfree_mbar = storage.dm_xchg_mbar.data_ptr() + 2
        if const_expr(self.dm_ordw_v or self.dm_ordw_k):
            self.dm_ord_ptr = storage.dm_xchg_mbar.data_ptr() + 11
        self.dm_prog_addr = None
        if const_expr("prog" in self.dm_opts):
            self.dm_prog_addr = aux_data.tensors[4].iterator.toint() + 2048
        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            self._dm_setmaxreg_dec(self.num_regs_empty)
            if const_expr(self.dm_pdyn):
                self._dm_claim_loop(tile_sched_params, dm_sbase, aux_data.tensors[-1].iterator.toint())
            if const_expr(self.dm_tcpush and self.deterministic):
                self._tcq_empty(aux_data.tensors[4], mdQ_semaphore, TileSchedulerCls, SeqlenInfoCls, block_info)

        #  RELAY
        # (14)
        if warp_idx == self.relay_warp_id:
            self._dm_setmaxreg_dec(
                self.num_regs_mma if self.use_2cta_instrs else self.num_regs_empty
            )
            if const_expr(self.use_2cta_instrs):
                self.relay(
                    dS_cluster_full_mbar_ptr,
                    dS_cluster_empty_mbar_ptr,
                    dS_cluster_leader_mbar_ptr,
                    cluster_layout_vmnk,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                )

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            self._dm_setmaxreg_dec(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                thr_mma_dQ,
                mQ,
                mK,
                mKt,
                mV,
                mK_gmem,
                mV_gmem,
                mKt_gmem,
                tensormap_workspace,
                mdO,
                mQt,
                mdOt,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sKt,
                sV,
                sdO,
                sQt,
                sdOt,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Kt,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_Qt,
                tma_atom_dOt,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                should_load_Q=True,
                should_load_dO=True,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            self._dm_setmaxreg_dec(self.num_regs_mma)

            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            if const_expr(self.dm_allocf):
                dm_utils.tc_fence_before()   # allocf: tcgen05.alloc (writes tmem_hold) ordered before the alloc barrier
            tmem.wait_for_alloc()
            if const_expr(self.dm_allocf):
                dm_utils.tc_fence_after()    # allocf: ... and the tmem_hold read (retrieve_ptr) after it
            tmem_ptr = tmem.retrieve_ptr(Float32)
            if const_expr(self.dm_tconv):
                # rw8k tconv: this CTA holds its TMEM (its last obstructable resource); the PDL dependent (the tail
                # converter, which uses no TMEM) may launch once every CTA got here or exited
                cute.arch.griddepcontrol_launch_dependents()

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sKt,
                sV,
                sdO,
                sdOt,
                tP,
                sdSt,
                sdS,
                tdS,
                sP,
                sdS_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                dS_cluster_full_mbar_ptr,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_leader_mbar_ptr,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                pipeline_P,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                is_leader_cta,
                blocksparse_tensors,
                dm_trace=aux_data.tensors[4] if const_expr(aux_data.tensors is not None and len(aux_data.tensors) > 4 and "lwc" not in self.dm_opts and ("trace" in self.dm_opts or "prog" in self.dm_opts)) else None,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            if const_expr(self.dm_allocf):
                # allocf: the 2-CTA dealloc handshake inside TmemAllocator.free (remote mbarrier arrive -> wait -> tcgen05.dealloc) gets
                # the fence pair too (blanket tcf mode for this call: fence::before the arrive, fence::after the wait)
                _tcf_saved = dm_utils.TCF[0]
                # the build-trap guard (tcf_end) must keep counting only the marked-pipeline wrapper fences: the blanket fences
                # of this one call are not evidence that the pipeline wrappers are bound (S6 positive control)
                _tcf_f0 = dm_utils._tcf_save_fences()
                dm_utils.TCF[0] = 2
            tmem.free(tmem_ptr)
            if const_expr(self.dm_allocf):
                dm_utils.TCF[0] = _tcf_saved
                dm_utils._tcf_restore_fences(_tcf_f0)
            if const_expr("tl4" in self.dm_opts and aux_data.tensors is not None and len(aux_data.tensors) > 4):
                if cute.arch.lane_idx() == 0:
                    dm_utils.red_add_u64(aux_data.tensors[4].iterator.toint() + 8 * (64 + 8 * bidx + 7), dm_utils.globaltimer())

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            self._dm_setmaxreg_inc(self.num_regs_compute)  # 8 warps
            tmem.wait_for_alloc()
            if const_expr(self.dm_allocf):
                dm_utils.tc_fence_after()    # allocf: canonical alloc -> fence::before -> bar -> fence::after -> ld tmem_hold
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdV,
                mdK,
                sdS,
                sdS_xchg,
                sP,
                sdS_dK,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_P,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_full_mbar_ptr,
                dQaccum_empty_mbar_ptr,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                mdK_semaphore,
                mdV_semaphore,
                aux_data,
                fastdiv_mods,
                blocksparse_tensors,
                sDm=sDm,
                dm_trace=aux_data.tensors[4] if const_expr(aux_data.tensors is not None and len(aux_data.tensors) > 4 and "lwc" not in self.dm_opts and ("trace" in self.dm_opts or "prog" in self.dm_opts)) else None,
                dm_xchg_mbar=storage.dm_xchg_mbar.data_ptr() if const_expr(self.use_2cta_instrs) else None,
            )
            tmem_alloc_barrier.arrive()

        # DM dsqw: dS quantization warpgroup (16, 17, 18, 19)
        if const_expr(self.dm_dsqw):
            if warp_idx >= self.dsq_warp_ids[0] and warp_idx <= self.dsq_warp_ids[-1]:
                if const_expr(self.num_regs_dsq < self.dm_launch_regs):
                    self._dm_setmaxreg_dec(self.num_regs_dsq)
                else:
                    self._dm_setmaxreg_inc(self.num_regs_dsq)
                tmem.wait_for_alloc()
                if const_expr(self.dm_allocf):
                    dm_utils.tc_fence_after()    # allocf (see the MMA warp)
                tmem_ptr = tmem.retrieve_ptr(Float32)
                self.dsq_loop(
                    thr_mma_S, tdPtdP, sdS, sdS_xchg, sdS_dK, pipeline_dP, pipeline_dS, dS_cluster_full_mbar_ptr,
                    sDm, storage.dm_xchg_mbar.data_ptr(), aux_data, block_info, SeqlenInfoCls, TileSchedulerCls,
                    dm_tr=aux_data.tensors[4] if const_expr(aux_data.tensors is not None and len(aux_data.tensors) > 4 and "lwc" not in self.dm_opts and "trace" in self.dm_opts and self.dm_tlev >= 0) else None,
                )
                tmem_alloc_barrier.arrive()

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            if const_expr(self.num_regs_reduce < self.dm_launch_regs):
                self._dm_setmaxreg_dec(self.num_regs_reduce)
            else:
                self._dm_setmaxreg_inc(self.num_regs_reduce)
            tmem.wait_for_alloc()
            if const_expr(self.dm_allocf):
                dm_utils.tc_fence_after()    # allocf: canonical alloc -> fence::before -> bar -> fence::after -> ld tmem_hold
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                dQaccum_empty_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
                blocksparse_tensors,
                dm_dq=aux_data.tensors[4] if const_expr("lwc" in self.dm_opts) else None,
                thr_mma_S=thr_mma_S if const_expr(self.dm_dsq) else None,
                tdPtdP=tdPtdP if const_expr(self.dm_dsq) else None,
                sdS=sdS if const_expr(self.dm_dsq) else None,
                sdS_xchg=sdS_xchg if const_expr(self.dm_dsq) else None,
                sdS_dK=sdS_dK if const_expr(self.dm_dsq) else None,
                pipeline_dP=pipeline_dP if const_expr(self.dm_dsq) else None,
                pipeline_dS=pipeline_dS if const_expr(self.dm_dsq) else None,
                dS_cluster_full_mbar_ptr=dS_cluster_full_mbar_ptr if const_expr(self.dm_dsq) else None,
                sDm=sDm if const_expr(self.dm_dsq) else None,
                dm_xchg_mbar=storage.dm_xchg_mbar.data_ptr() if const_expr(self.dm_dsq) else None,
                aux_data=aux_data if const_expr(self.dm_dsq) else None,
                dm_trw=(aux_data.tensors[4].iterator.toint() + 8 * 44
                        if const_expr(aux_data.tensors is not None and len(aux_data.tensors) > 4 and "lwc" not in self.dm_opts
                                      and "trace" in self.dm_opts and "semw" in self.dm_opts) else None),
                dm_tlv=(aux_data.tensors[4] if const_expr(aux_data.tensors is not None and len(aux_data.tensors) > 4 and "lwc" not in self.dm_opts
                                                         and "trace" in self.dm_opts and self.dm_tlev >= 0) else None),
                dm_tcq=aux_data.tensors[4] if const_expr(self.dm_tconv and self.deterministic) else None,
            )
            if const_expr("tl4" in self.dm_opts and aux_data.tensors is not None and len(aux_data.tensors) > 4):
                if warp_idx == self.reduce_warp_ids[0] and cute.arch.lane_idx() == 0:
                    dm_utils.red_add_u64(aux_data.tensors[4].iterator.toint() + 8 * (64 + 8 * bidx + 6), dm_utils.globaltimer())
            tmem_alloc_barrier.arrive()

        dm_utils.tcf_end()
        return

    @cute.jit
    def relay(
        self,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dm_pt = Int32(0)
        while work_tile.is_valid_tile:
            self._dm_prog(Int32(0), dm_pt, Int32(0), Int32(1))
            _, kv_block, head_idx, _, batch_idx, _, m_block_min, m_block_max, process_tile = (
                self._work_tile(work_tile, cta_rank_in_cluster, SeqlenInfoCls, block_info)
            )
            num_iters = (m_block_max - m_block_min) * self.dm_G
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                num_iters = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    kv_block // self.kv_subtile_factor,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = num_iters > Int32(0)

            if process_tile:
                for _ in cutlass.range(num_iters, unroll=1):
                    # Wait for dS_xchg from peer CTA
                    cute.arch.mbarrier_wait(dS_cluster_full_mbar_ptr, phase=dS_cluster_phase)

                    # Arrive on MMA leader warp
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dS_cluster_leader_mbar_ptr, Int32(0))

                    dS_cluster_phase ^= 1

            tile_scheduler.prefetch_next_work()
            self._dm_prog(Int32(0), dm_pt, Int32(0), Int32(15))
            dm_pt = dm_pt + 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        thr_mma_dQ: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mK_gmem: Optional[cute.Tensor],
        mV_gmem: Optional[cute.Tensor],
        mKt_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mdOt: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sQt: cute.Tensor,
        sdOt: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_dOt: Optional[cute.CopyAtom],  # 2-CTA only
        pipeline_Q: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_Q_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_O_Ot = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        tensormap_manager = None
        tensormap_gmem = ()
        desc_ptrs = (None, None, None)
        atoms = ()
        if const_expr(self.ragged_kv_tma):
            tensormap_manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
            bidx, bidy, bidz = cute.arch.block_idx()
            gdx, gdy, _ = cute.arch.grid_dim()
            cta_linear_idx = bidx + bidy * gdx + bidz * gdx * gdy
            if cta_linear_idx >= cute.size(tensormap_workspace, mode=[0]):
                # An undersized workspace would silently corrupt memory: fail loudly instead.
                cute.arch.inline_ptx("trap;", read_only_args=[])
            n_desc = 3 if const_expr(self.use_2cta_instrs) else 2
            atoms = (tma_atom_K, tma_atom_V, tma_atom_Kt)[:n_desc]
            tensormap_gmem = tuple(
                tensormap_manager.get_tensormap_ptr(
                    tensormap_workspace[(cta_linear_idx, i, None)].iterator
                )
                for i in range(n_desc)
            )
            desc_ptrs = tuple(
                tensormap_manager.get_tensormap_ptr(p, cute.AddressSpace.generic)
                for p in tensormap_gmem
            ) + (None,) * (3 - n_desc)
            for atom, gmem_ptr in zip(atoms, tensormap_gmem):
                tensormap_manager.init_tensormap_from_atom(atom, gmem_ptr, self.load_warp_id)
            tensormap_manager.fence_tensormap_initialization()
        # The K/V extent depends on the batch only: rewrite the descriptors when it changes.
        kv_desc_batch_idx = Int32(-1)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dm_pt = Int32(0)
        while work_tile.is_valid_tile:
            self._dm_prog(Int32(1), dm_pt, Int32(0), Int32(1))
            if const_expr(self.dm_pers > 0):
                if dm_pt > 0:
                    # DM pers: previous tile's dK/dV epilogue (sdK/sdV alias sK/sV) has drained, and its MMAs are done
                    # with sK/sV/sKt (the Kt release is committed after the last dQ MMA)
                    cute.arch.mbarrier_wait(self.dm_kvfree_mbar, phase=(dm_pt - 1) & 1)
                    self._dm_prog(Int32(1), dm_pt, Int32(0), Int32(3))
                    pipeline_Kt.sync_object_empty.wait(producer_state_Kt.index, producer_state_Kt.phase)
                    self._dm_prog(Int32(1), dm_pt, Int32(0), Int32(4))
            (
                work_n,
                kv_block,
                head_idx,
                kv_head,
                batch_idx,
                seqlen,
                m_block_min,
                m_block_max,
                process_tile,
            ) = self._work_tile(work_tile, cta_rank_in_cluster, SeqlenInfoCls, block_info)
            if const_expr(self.ragged_kv_tma) and batch_idx != kv_desc_batch_idx:
                # Shrink the rows extent of K, V (mode 0) and Kt (mode 1) to this sequence's
                # end: packed layouts count absolute rows, dense layouts per-batch rows.
                # Extents must stay >= 1.
                kv_desc_batch_idx = batch_idx
                kv_rows_end = (
                    seqlen.offset_k + seqlen.seqlen_k
                    if const_expr(seqlen.has_cu_seqlens_k)
                    else seqlen.seqlen_k
                )
                kv_rows_end = cutlass.max(kv_rows_end, 1)
                bounded = []
                for t, rows_mode in ((mK_gmem, 0), (mV_gmem, 0), (mKt_gmem, 1)):
                    if const_expr(t is not None):
                        shape = list(t.shape)
                        shape[rows_mode] = kv_rows_end
                        bounded.append(
                            cute.make_tensor(
                                t.iterator, cute.make_layout(tuple(shape), stride=t.stride)
                            )
                        )
                tensormap_manager.update_tensormap(
                    tuple(bounded), atoms, tensormap_gmem, self.load_warp_id, (None,) * len(atoms)
                )
                for p in tensormap_gmem:
                    tensormap_manager.fence_tensormap_update(p)
                    cute.arch.inline_ptx("prefetch.tensormap [{$r0}];", read_only_args=[p.toint()])
            if const_expr(self.dm_gloop):
                # gloop: one stream over the qh query heads of the KV head; K / V / Kt once (first head)
                assert not self.use_block_sparsity and self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED
                dm_head0 = head_idx
                if process_tile:
                    for dm_g in cutlass.range(self.dm_G, unroll=1):
                        head_idx = dm_head0 + dm_g
                        n_block_sparse = kv_block // self.kv_subtile_factor

                        # GMEM tensors (varlen-aware)
                        # pack_gqa: the Q row mode is ((qh, seqlen)), so a varlen offset applies to its
                        # second mode; LSE / dPsum are per KV head in packed rows, pack_factor per token.
                        q_off = seqlen.offset_q if const_expr(not self.pack_gqa) else (None, seqlen.offset_q)
                        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                        mK_cur, mV_cur = [
                            seqlen.offset_batch_K(t, batch_idx, dim=3)[None, None, kv_head] for t in (mK, mV)
                        ]
                        if const_expr(not seqlen.has_cu_seqlens_q):
                            mdO_cur = mdO[None, None, head_idx, batch_idx]
                            mLSE_cur = mLSE[None, head_idx, batch_idx]
                            mdPsum_cur = mdPsum[None, head_idx, batch_idx]
                        else:
                            mdO_cur = cute.domain_offset((0, q_off), mdO[None, None, head_idx])
                            row_offset = sequence_row_offset(
                                seqlen.offset_q, batch_idx, self.pack_factor, self.tile_m
                            )
                            mLSE_cur = cute.domain_offset((row_offset,), mLSE[None, head_idx])
                            mdPsum_cur = cute.domain_offset((row_offset,), mdPsum[None, head_idx])

                        if const_expr(self.use_2cta_instrs):
                            if const_expr(not seqlen.has_cu_seqlens_q):
                                mQt_cur = mQt[None, None, head_idx, batch_idx]
                                mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                            else:
                                mQt_cur = cute.domain_offset((0, q_off, 0), mQt)[None, None, head_idx]
                                mdOt_cur = cute.domain_offset((q_off, 0, 0), mdOt)[None, None, head_idx]
                            if const_expr(not seqlen.has_cu_seqlens_k):
                                mKt_cur = mKt[None, None, kv_head, batch_idx]
                            else:
                                mKt_cur = cute.domain_offset((0, seqlen.offset_k, 0), mKt)[None, None, kv_head]

                        # (1) S.T = K @ Q.T
                        gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (work_n, 0))
                        tSgK = thr_mma_S.partition_A(gK)

                        gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
                        tSgQ = thr_mma_S.partition_B(gQ)
                        gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                        gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
                        gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
                        tdPgdO = thr_mma_dV.partition_B(gdO)

                        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                        load_K, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_K,
                            block_in_cluster_coord_vmnk[2],
                            a_cta_layout,
                            tSgK,
                            sK,
                            single_stage=True,
                            **({"tma_desc_ptr": desc_ptrs[0]} if const_expr(self.ragged_kv_tma) else {}),
                        )

                        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                        load_Q, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_Q,
                            cta_coord=block_in_cluster_coord_vmnk[1],
                            cta_layout=b_cta_layout,
                            src_tensor=tSgQ,
                            dst_tensor=sQ,
                            mcast_mask=q_do_mcast_mask,
                        )
                        load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

                        # (2) dP = V @ dO.T
                        gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (work_n, 0))
                        tdPgV = thr_mma_dP.partition_A(gV)

                        load_V, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_V,
                            0,
                            cute.make_layout(1),
                            tdPgV,
                            sV,
                            single_stage=True,
                            **({"tma_desc_ptr": desc_ptrs[1]} if const_expr(self.ragged_kv_tma) else {}),
                        )

                        load_dOt = None
                        if const_expr(tma_atom_dOt is not None):
                            gdOt = cute.local_tile(
                                mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0)
                            )
                            tdPgdO = thr_mma_dP.partition_B(gdOt)
                            load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_dOt,
                                cta_coord=block_in_cluster_coord_vmnk[1],
                                cta_layout=b_cta_layout,
                                src_tensor=tdPgdO,
                                dst_tensor=sdOt,
                                mcast_mask=q_do_mcast_mask,
                            )
                            load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)

                        # (3) dV += P.T @ dO
                        gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
                        tdVgdO = thr_mma_dV.partition_B(gdO)
                        load_dO, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_dO,
                            cta_coord=block_in_cluster_coord_vmnk[1],
                            cta_layout=b_cta_layout,
                            src_tensor=tdVgdO,
                            dst_tensor=sdO,
                            mcast_mask=q_do_mcast_mask,
                        )
                        load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

                        # (4) dK += dS.T @ Q (2-CTA: needs separate Qt load)
                        load_Qt = None
                        if const_expr(tma_atom_Qt is not None):
                            gQt = cute.local_tile(
                                mQt_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None)
                            )
                            tdKgQt = thr_mma_dK.partition_B(gQt)
                            load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_Qt,
                                cta_coord=block_in_cluster_coord_vmnk[1],
                                cta_layout=b_cta_layout,
                                src_tensor=tdKgQt,
                                dst_tensor=sQt,
                                mcast_mask=q_do_mcast_mask,
                            )
                            load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)

                        # (5) dQ = dS @ K
                        load_Kt = None
                        if const_expr(self.use_2cta_instrs):
                            gKt = cute.local_tile(
                                mKt_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, work_n)
                            )
                            tdQgK = thr_mma_dQ.partition_B(gKt)

                            load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_Kt,
                                block_in_cluster_coord_vmnk[1],
                                b_cta_layout,
                                tdQgK,
                                sKt,
                                single_stage=True,
                                **({"tma_desc_ptr": desc_ptrs[2]} if const_expr(self.ragged_kv_tma) else {}),
                            )

                        copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
                        copy_stats = partial(cute.copy, copy_atom_stats)
                        # copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SMulticastOp(), Float32)
                        # sLSE = cute.logical_divide(sLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
                        # gLSE = cute.logical_divide(gLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
                        # sdPsum = cute.logical_divide(sdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
                        # gdPsum = cute.logical_divide(gdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
                        # copy_stats = partial(cute.copy, copy_atom_stats, mcast_mask=q_do_mcast_mask)

                        first_m_block = m_block_min
                        #### Prologue ####
                        if const_expr(should_load_Q):
                            # K & Q (for S); gloop: K only with the first query head
                            if dm_g == 0:
                                pipeline_Q.producer_acquire(
                                    producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                                )
                                load_K(
                                    tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE)
                                )
                            else:
                                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                            load_Q(first_m_block, producer_state=producer_state_Q_LSE)
                            pipeline_Q.producer_commit(producer_state_Q_LSE)

                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gLSE[None, first_m_block],
                                    sLSE[None, producer_state_Q_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(
                                        producer_state_Q_LSE
                                    ),
                                )
                            producer_state_Q_LSE.advance()

                        if const_expr(should_load_dO):
                            if dm_g == 0:
                                pipeline_dO.producer_acquire(
                                    producer_state_dO_dPsum,
                                    extra_tx_count=self.tma_copy_bytes["V"] + self.tma_copy_bytes["dO"]
                                    if const_expr(load_dOt is not None)
                                    else self.tma_copy_bytes["V"],
                                )
                                load_V(
                                    tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    )
                                )
                            else:
                                pipeline_dO.producer_acquire(
                                    producer_state_dO_dPsum,
                                    extra_tx_count=self.tma_copy_bytes["dO"]
                                    if const_expr(load_dOt is not None)
                                    else 0,
                                )
                            load_dO(first_m_block, producer_state=producer_state_dO_dPsum)
                            if const_expr(load_dOt is not None):
                                load_dOt(first_m_block, producer_state=producer_state_dO_dPsum)
                            pipeline_dO.producer_commit(producer_state_dO_dPsum)

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gdPsum[None, first_m_block],
                                    sdPsum[None, producer_state_dO_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    ),
                                )
                            producer_state_dO_dPsum.advance()

                        if const_expr(self.use_2cta_instrs):
                            if dm_g == 0:
                                pipeline_Kt.producer_acquire(producer_state_Kt)
                                load_Kt(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                                pipeline_Kt.producer_commit(producer_state_Kt)
                                producer_state_Kt.advance()
                        #### Main Loop ####
                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            if const_expr(should_load_Q):
                                if const_expr(load_Qt is not None):
                                    pipeline_Qt.producer_acquire(producer_state_Qt)
                                    load_Qt(m_block - 1, producer_state=producer_state_Qt)
                                    pipeline_Qt.producer_commit(producer_state_Qt)
                                    producer_state_Qt.advance()

                                # Q (for S)
                                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                                load_Q(m_block, producer_state=producer_state_Q_LSE)
                                pipeline_Q.producer_commit(producer_state_Q_LSE)

                                # LSE
                                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gLSE[None, m_block],
                                        sLSE[None, producer_state_Q_LSE.index],
                                        mbar_ptr=pipeline_LSE.producer_get_barrier(
                                            producer_state_Q_LSE
                                        ),
                                    )
                                producer_state_Q_LSE.advance()

                            if const_expr(should_load_dO):
                                pipeline_dO.producer_acquire(
                                    producer_state_dO_dPsum,
                                    extra_tx_count=self.tma_copy_bytes["dO"]
                                    if const_expr(load_dOt is not None)
                                    else 0,
                                )
                                load_dO(m_block, producer_state=producer_state_dO_dPsum)
                                if const_expr(load_dOt is not None):
                                    load_dOt(m_block, producer_state=producer_state_dO_dPsum)
                                pipeline_dO.producer_commit(producer_state_dO_dPsum)

                                # dPsum
                                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gdPsum[None, m_block],
                                        sdPsum[None, producer_state_dO_dPsum.index],
                                        mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                            producer_state_dO_dPsum
                                        ),
                                    )
                                producer_state_dO_dPsum.advance()

                        if const_expr(should_load_Q):
                            if const_expr(load_Qt is not None):
                                pipeline_Qt.producer_acquire(producer_state_Qt)
                                load_Qt(m_block_max - 1, producer_state=producer_state_Qt)
                                pipeline_Qt.producer_commit(producer_state_Qt)
                                producer_state_Qt.advance()

                    if const_expr(should_load_Q):
                        if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                            pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                        if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                            pipeline_LSE.producer_tail(producer_state_Q_LSE)
                        if const_expr(tma_atom_Qt is not None):
                            if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                pipeline_Qt.producer_tail(producer_state_Qt)
                    if const_expr(should_load_dO):
                        if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                            pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                        if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                            pipeline_dPsum.producer_tail(producer_state_dO_dPsum)
            else:
                n_block_sparse = kv_block // self.kv_subtile_factor

                # GMEM tensors (varlen-aware)
                # pack_gqa: the Q row mode is ((qh, seqlen)), so a varlen offset applies to its
                # second mode; LSE / dPsum are per KV head in packed rows, pack_factor per token.
                q_off = seqlen.offset_q if const_expr(not self.pack_gqa) else (None, seqlen.offset_q)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                mK_cur, mV_cur = [
                    seqlen.offset_batch_K(t, batch_idx, dim=3)[None, None, kv_head] for t in (mK, mV)
                ]
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mdO_cur = mdO[None, None, head_idx, batch_idx]
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                    mdPsum_cur = mdPsum[None, head_idx, batch_idx]
                else:
                    mdO_cur = cute.domain_offset((0, q_off), mdO[None, None, head_idx])
                    row_offset = sequence_row_offset(
                        seqlen.offset_q, batch_idx, self.pack_factor, self.tile_m
                    )
                    mLSE_cur = cute.domain_offset((row_offset,), mLSE[None, head_idx])
                    mdPsum_cur = cute.domain_offset((row_offset,), mdPsum[None, head_idx])

                if const_expr(self.use_2cta_instrs):
                    if const_expr(not seqlen.has_cu_seqlens_q):
                        mQt_cur = mQt[None, None, head_idx, batch_idx]
                        mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                    else:
                        mQt_cur = cute.domain_offset((0, q_off, 0), mQt)[None, None, head_idx]
                        mdOt_cur = cute.domain_offset((q_off, 0, 0), mdOt)[None, None, head_idx]
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        mKt_cur = mKt[None, None, kv_head, batch_idx]
                    else:
                        mKt_cur = cute.domain_offset((0, seqlen.offset_k, 0), mKt)[None, None, kv_head]

                # (1) S.T = K @ Q.T
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (work_n, 0))
                tSgK = thr_mma_S.partition_A(gK)

                gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
                tSgQ = thr_mma_S.partition_B(gQ)
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
                gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
                tdPgdO = thr_mma_dV.partition_B(gdO)

                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    tSgK,
                    sK,
                    single_stage=True,
                    **({"tma_desc_ptr": desc_ptrs[0]} if const_expr(self.ragged_kv_tma) else {}),
                )

                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tSgQ,
                    dst_tensor=sQ,
                    mcast_mask=q_do_mcast_mask,
                )
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

                # (2) dP = V @ dO.T
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (work_n, 0))
                tdPgV = thr_mma_dP.partition_A(gV)

                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    tdPgV,
                    sV,
                    single_stage=True,
                    **({"tma_desc_ptr": desc_ptrs[1]} if const_expr(self.ragged_kv_tma) else {}),
                )

                load_dOt = None
                if const_expr(tma_atom_dOt is not None):
                    gdOt = cute.local_tile(
                        mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0)
                    )
                    tdPgdO = thr_mma_dP.partition_B(gdOt)
                    load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_dOt,
                        cta_coord=block_in_cluster_coord_vmnk[1],
                        cta_layout=b_cta_layout,
                        src_tensor=tdPgdO,
                        dst_tensor=sdOt,
                        mcast_mask=q_do_mcast_mask,
                    )
                    load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)

                # (3) dV += P.T @ dO
                gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
                tdVgdO = thr_mma_dV.partition_B(gdO)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdVgdO,
                    dst_tensor=sdO,
                    mcast_mask=q_do_mcast_mask,
                )
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

                # (4) dK += dS.T @ Q (2-CTA: needs separate Qt load)
                load_Qt = None
                if const_expr(tma_atom_Qt is not None):
                    gQt = cute.local_tile(
                        mQt_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None)
                    )
                    tdKgQt = thr_mma_dK.partition_B(gQt)
                    load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Qt,
                        cta_coord=block_in_cluster_coord_vmnk[1],
                        cta_layout=b_cta_layout,
                        src_tensor=tdKgQt,
                        dst_tensor=sQt,
                        mcast_mask=q_do_mcast_mask,
                    )
                    load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)

                # (5) dQ = dS @ K
                load_Kt = None
                if const_expr(self.use_2cta_instrs):
                    gKt = cute.local_tile(
                        mKt_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, work_n)
                    )
                    tdQgK = thr_mma_dQ.partition_B(gKt)

                    load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Kt,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        tdQgK,
                        sKt,
                        single_stage=True,
                        **({"tma_desc_ptr": desc_ptrs[2]} if const_expr(self.ragged_kv_tma) else {}),
                    )

                copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
                copy_stats = partial(cute.copy, copy_atom_stats)
                # copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SMulticastOp(), Float32)
                # sLSE = cute.logical_divide(sLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
                # gLSE = cute.logical_divide(gLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
                # sdPsum = cute.logical_divide(sdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
                # gdPsum = cute.logical_divide(gdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
                # copy_stats = partial(cute.copy, copy_atom_stats, mcast_mask=q_do_mcast_mask)

                if const_expr(not self.use_block_sparsity):
                    if process_tile:
                        first_m_block = m_block_min
                        if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL):
                            #### Prologue ####
                            assert should_load_Q and should_load_dO
                            assert load_dOt is not None and load_Qt is not None
                            # K & Q (for S)
                            pipeline_Q.producer_acquire(
                                producer_state_Q_Qt,
                                extra_tx_count=self.tma_copy_bytes["K"],
                            )
                            load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_Qt))
                            load_Q(first_m_block, producer_state=producer_state_Q_Qt)
                            pipeline_Q.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()
                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_LSE)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gLSE[None, first_m_block],
                                    sLSE[None, producer_state_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                                )
                            producer_state_LSE.advance()

                            # dOt + V, for dP.T = V @ dO.T
                            pipeline_dO.producer_acquire(
                                producer_state_O_Ot,
                                extra_tx_count=self.tma_copy_bytes["V"],
                            )
                            load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_O_Ot))
                            load_dOt(first_m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()
                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dPsum)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gdPsum[None, first_m_block],
                                    sdPsum[None, producer_state_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dPsum),
                                )
                            producer_state_dPsum.advance()

                            # Qt, for dK = dS.T @ Q
                            pipeline_Qt.producer_acquire(
                                producer_state_Q_Qt,
                                extra_tx_count=self.tma_copy_bytes["K"],
                            )
                            load_Qt(first_m_block, producer_state=producer_state_Q_Qt)
                            load_Kt(tma_bar_ptr=pipeline_Qt.producer_get_barrier(producer_state_Q_Qt))
                            pipeline_Qt.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()

                            # dO, for dV = P.T @ dO
                            pipeline_dO.producer_acquire(producer_state_O_Ot)
                            load_dO(first_m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()

                            #### Mainloop ####
                            # 2CTA: [lse | Q | dOt | dPsum | Qt | dO]
                            for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                                # LSE
                                pipeline_LSE.producer_acquire(producer_state_LSE)
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gLSE[None, m_block],
                                        sLSE[None, producer_state_LSE.index],
                                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                                    )
                                producer_state_LSE.advance()

                                # Q
                                pipeline_Q.producer_acquire(producer_state_Q_Qt)
                                load_Q(m_block, producer_state=producer_state_Q_Qt)
                                pipeline_Q.producer_commit(producer_state_Q_Qt)
                                producer_state_Q_Qt.advance()

                                # dPsum
                                pipeline_dPsum.producer_acquire(producer_state_dPsum)
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gdPsum[None, m_block],
                                        sdPsum[None, producer_state_dPsum.index],
                                        mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                            producer_state_dPsum
                                        ),
                                    )
                                producer_state_dPsum.advance()

                                # dOt, for dP.T = V @ dO.T
                                pipeline_dO.producer_acquire(producer_state_O_Ot)
                                load_dOt(m_block, producer_state=producer_state_O_Ot)
                                pipeline_dO.producer_commit(producer_state_O_Ot)
                                producer_state_O_Ot.advance()

                                # Qt, for dK = dS.T @ Q
                                pipeline_Qt.producer_acquire(producer_state_Q_Qt)
                                load_Qt(m_block, producer_state=producer_state_Q_Qt)
                                pipeline_Qt.producer_commit(producer_state_Q_Qt)
                                producer_state_Q_Qt.advance()

                                # dO, for dV = P.T @ dO
                                pipeline_dO.producer_acquire(producer_state_O_Ot)
                                load_dO(m_block, producer_state=producer_state_O_Ot)
                                pipeline_dO.producer_commit(producer_state_O_Ot)
                                producer_state_O_Ot.advance()

                            if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                pipeline_Q.producer_tail(producer_state_Q_Qt)
                            if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                pipeline_LSE.producer_tail(producer_state_LSE)
                            if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                pipeline_dO.producer_tail(producer_state_O_Ot)
                            if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                pipeline_dPsum.producer_tail(producer_state_dPsum)
                        else:
                            #### Prologue ####
                            if const_expr(should_load_Q):
                                # K & Q (for S)
                                pipeline_Q.producer_acquire(
                                    producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                                )
                                load_K(
                                    tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE)
                                )
                                load_Q(first_m_block, producer_state=producer_state_Q_LSE)
                                pipeline_Q.producer_commit(producer_state_Q_LSE)

                                # LSE
                                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gLSE[None, first_m_block],
                                        sLSE[None, producer_state_Q_LSE.index],
                                        mbar_ptr=pipeline_LSE.producer_get_barrier(
                                            producer_state_Q_LSE
                                        ),
                                    )
                                producer_state_Q_LSE.advance()

                            if const_expr(should_load_dO):
                                pipeline_dO.producer_acquire(
                                    producer_state_dO_dPsum,
                                    extra_tx_count=self.tma_copy_bytes["V"] + self.tma_copy_bytes["dO"]
                                    if const_expr(load_dOt is not None)
                                    else self.tma_copy_bytes["V"],
                                )
                                load_V(
                                    tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    )
                                )
                                load_dO(first_m_block, producer_state=producer_state_dO_dPsum)
                                if const_expr(load_dOt is not None):
                                    load_dOt(first_m_block, producer_state=producer_state_dO_dPsum)
                                pipeline_dO.producer_commit(producer_state_dO_dPsum)

                                # dPsum
                                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                                with cute.arch.elect_one():
                                    copy_stats(
                                        gdPsum[None, first_m_block],
                                        sdPsum[None, producer_state_dO_dPsum.index],
                                        mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                            producer_state_dO_dPsum
                                        ),
                                    )
                                producer_state_dO_dPsum.advance()

                            if const_expr(self.use_2cta_instrs):
                                pipeline_Kt.producer_acquire(producer_state_Kt)
                                load_Kt(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                                pipeline_Kt.producer_commit(producer_state_Kt)
                                producer_state_Kt.advance()
                            #### Main Loop ####
                            for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                                if const_expr(should_load_Q):
                                    if const_expr(load_Qt is not None):
                                        pipeline_Qt.producer_acquire(producer_state_Qt)
                                        load_Qt(m_block - 1, producer_state=producer_state_Qt)
                                        pipeline_Qt.producer_commit(producer_state_Qt)
                                        producer_state_Qt.advance()

                                    # Q (for S)
                                    pipeline_Q.producer_acquire(producer_state_Q_LSE)
                                    load_Q(m_block, producer_state=producer_state_Q_LSE)
                                    pipeline_Q.producer_commit(producer_state_Q_LSE)

                                    # LSE
                                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                                    with cute.arch.elect_one():
                                        copy_stats(
                                            gLSE[None, m_block],
                                            sLSE[None, producer_state_Q_LSE.index],
                                            mbar_ptr=pipeline_LSE.producer_get_barrier(
                                                producer_state_Q_LSE
                                            ),
                                        )
                                    producer_state_Q_LSE.advance()

                                if const_expr(should_load_dO):
                                    pipeline_dO.producer_acquire(
                                        producer_state_dO_dPsum,
                                        extra_tx_count=self.tma_copy_bytes["dO"]
                                        if const_expr(load_dOt is not None)
                                        else 0,
                                    )
                                    load_dO(m_block, producer_state=producer_state_dO_dPsum)
                                    if const_expr(load_dOt is not None):
                                        load_dOt(m_block, producer_state=producer_state_dO_dPsum)
                                    pipeline_dO.producer_commit(producer_state_dO_dPsum)

                                    # dPsum
                                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                                    with cute.arch.elect_one():
                                        copy_stats(
                                            gdPsum[None, m_block],
                                            sdPsum[None, producer_state_dO_dPsum.index],
                                            mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                                producer_state_dO_dPsum
                                            ),
                                        )
                                    producer_state_dO_dPsum.advance()

                            #### Tail ####
                            if const_expr(should_load_Q):
                                if const_expr(load_Qt is not None):
                                    pipeline_Qt.producer_acquire(producer_state_Qt)
                                    load_Qt(m_block_max - 1, producer_state=producer_state_Qt)
                                    pipeline_Qt.producer_commit(producer_state_Qt)
                                    producer_state_Qt.advance()

                                if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                    pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                                if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                    pipeline_LSE.producer_tail(producer_state_Q_LSE)
                                if const_expr(load_Qt is not None):
                                    if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                        pipeline_Qt.producer_tail(producer_state_Qt)
                            if const_expr(should_load_dO):
                                if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                    pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                                if const_expr(self.dm_pers == 0):  # DM pers: tails advance the shared states; per-tile is wrong
                                    pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

                else:
                    assert blocksparse_tensors is not None
                    if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL):
                        assert should_load_Q and should_load_dO
                        assert load_dOt is not None and load_Qt is not None
                        assert load_Kt is not None and pipeline_Qt is not None
                        (
                            producer_state_Q_Qt,
                            producer_state_O_Ot,
                            producer_state_LSE,
                            producer_state_dPsum,
                        ) = produce_block_sparse_q_loads_bwd_sm100_2cta_hdim192(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block_sparse,
                            producer_state_Q_Qt,
                            producer_state_O_Ot,
                            producer_state_LSE,
                            producer_state_dPsum,
                            pipeline_Q,
                            pipeline_LSE,
                            pipeline_dO,
                            pipeline_dPsum,
                            pipeline_Qt,
                            load_K,
                            load_V,
                            load_Q,
                            load_dO,
                            load_Qt,
                            load_Kt,
                            load_dOt,
                            copy_stats,
                            gLSE,
                            sLSE,
                            gdPsum,
                            sdPsum,
                            self.tma_copy_bytes["K"],
                            self.tma_copy_bytes["V"],
                            q_subtile_factor=self.q_subtile_factor,
                            m_block_max=m_block_max,
                        )
                    else:
                        (
                            producer_state_Q_LSE,
                            producer_state_dO_dPsum,
                            producer_state_Qt,
                            producer_state_Kt,
                        ) = produce_block_sparse_q_loads_bwd_sm100_default(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block_sparse,
                            producer_state_Q_LSE,
                            producer_state_dO_dPsum,
                            pipeline_Q,
                            pipeline_LSE,
                            pipeline_dO,
                            pipeline_dPsum,
                            load_K,
                            load_V,
                            load_Q,
                            load_dO,
                            copy_stats,
                            gLSE,
                            sLSE,
                            gdPsum,
                            sdPsum,
                            self.tma_copy_bytes["K"],
                            self.tma_copy_bytes["V"],
                            should_load_Q,
                            should_load_dO,
                            q_subtile_factor=self.q_subtile_factor,
                            m_block_max=m_block_max,
                            use_2cta_instrs=self.use_2cta_instrs,
                            producer_state_Qt=producer_state_Qt,
                            producer_state_Kt=producer_state_Kt,
                            pipeline_Qt=pipeline_Qt,
                            pipeline_Kt=pipeline_Kt,
                            load_Qt=load_Qt,
                            load_Kt=load_Kt,
                            load_dOt=load_dOt,
                            tma_copy_bytes_dO=self.tma_copy_bytes["dO"],
                        )
            tile_scheduler.prefetch_next_work()
            self._dm_prog(Int32(1), dm_pt, Int32(0), Int32(15))
            dm_pt = dm_pt + 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        tP: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        tdS: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS_dK: Optional[cute.Tensor],
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        pipeline_Q: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        pipeline_P: Optional[PipelineAsync],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        is_leader_cta: cutlass.Boolean,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        dm_trace=None,
    ):
        # For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem / tmem tensors
        # S = K @ Q.T
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dOt.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        # For 2-CTA, dS (dK mma) MUST come from TMEM (cannot use SMEM)
        if const_expr(self.acc_m64):
            tdKrdS = tiled_mma_dK.make_fragment_A(sdS_dK)  # this CTA's dS^T rows, from SMEM
        elif const_expr(self.use_smem_dS_for_mma_dK and not self.use_2cta_instrs):
            tdKrdS = tiled_mma_dK.make_fragment_A(sdSt)  # From SMEM
        else:
            tdKrdS = tiled_mma_dK.make_fragment_A(tdS)  # From TMEM

        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        # dQ = dS @ K
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdVrP = tiled_mma_dV.make_fragment_A(sP if const_expr(self.acc_m64) else tP)

        # mma_qk_fn = partial(gemm_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, zero_init=True)
        _ss_gemm = dm_utils.gemm_ptx_ss_w_idx if const_expr(self.dm_d3nh) else _dm_drop_dz(gemm_ptx_w_idx)
        mma_qk_fn = partial(
            _ss_gemm,
            tiled_mma_S,
            tStS,
            tSrK,
            tSrQ,
            sA=sK,
            sB=sQ,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        # mma_dov_fn = partial(gemm_w_idx, tiled_mma_dP, tdPtdP, tdPrV, tdPrdOt, zero_init=True)
        mma_dov_fn = partial(
            _ss_gemm,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        # mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO)
        assert not self.acc_m64, "DM: TMEM P / dS layout only"
        mma_pdo_fn = partial(
            dm_utils.gemm_ptx_bs_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            None,
            sdO,
            tA_addr=self.tmem_P_offset,
            sfa_base=self.dm_sfa_dv_col,
            sfb_base=self.dm_sfb_dv_col,
            sfa_map=tuple((4 * k, 0) for k in range(self.tile_m // 32)),
            sfb_map=tuple((0, 0) for k in range(self.tile_m // 32)),
            cta_group=self.cta_group_size,
        )
        num_unroll_groups = 2 if const_expr(self.use_2cta_instrs) else 1
        dm_nk_dq = self.cluster_tile_n // 32
        mma_dsk_fn = partial(
            dm_utils.gemm_ptx_bs_w_idx,
            tiled_mma_dQ,
            tdQtdQ,
            tdQrdS,
            tdQrK,
            sdS,
            sKt,
            zero_init=True,
            sfa_base=self.dm_sfa_dq_col,
            sfb_base=self.dm_sfb_dq_col,
            sfa_map=tuple((0, 0) for k in range(dm_nk_dq)),
            sfb_map=tuple((2 * (k // 4), k % 4) for k in range(dm_nk_dq)),
            cta_group=self.cta_group_size,
        )
        # mma_dsk_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, sA=sdS, sB=sKt, zero_init=True
        # )
        if const_expr(self.acc_m64):
            mma_dsq_fn = partial(
                _dm_drop_dz(gemm_ptx_w_idx),
                tiled_mma_dK,
                tdKtdK,
                tdKrdS,
                tdKrQ,
                sA=sdS_dK,
                sB=sQt,
                cta_group=self.cta_group_size,
            )
        elif const_expr(self.use_smem_dS_for_mma_dK and not self.use_2cta_instrs):
            mma_dsq_fn = partial(_dm_drop_dz(gemm_w_idx), tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ)
        else:
            # Need to explicitly pass in tA_addr for correctness
            mma_dsq_fn = partial(
                dm_utils.gemm_ptx_bs_w_idx,
                tiled_mma_dK,
                tdKtdK,
                tdKrdS,
                tdKrQ,
                None,
                sQt,
                tA_addr=self.tmem_dS_offset,
                sfa_base=self.dm_sfa_dk_col,
                sfb_base=self.dm_sfb_dk_col,
                sfa_map=tuple((4 * k, 0) for k in range(self.tile_m // 32)),
                sfb_map=tuple((0, 0) for k in range(self.tile_m // 32)),
                cta_group=self.cta_group_size,
            )

        pipeline_Q_consumer = pipeline_Q.make_consumer()

        consumer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        producer_phase_dQ = Int32(1)  # 2-CTA: separate phase for dQ pipeline
        producer_phase_S = Int32(1)  # split_P_dS: separate phase for the S handshake
        consumer_state_P = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group
        # DM pers: the shared producer_phase_acc is only valid within one tile; each tile's S/P, dP and dQ empty
        # barriers complete M = #m_blocks times, so the next tile starts at base ^ (M & 1)
        dm_ph_base = Int32(1)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_phase = Int32(0)
        dm_dvn = Int32(0)   # ordinv ordw: dV MMAs committed to ord[0] so far
        dm_dkn = Int32(0)   # ordinv ordw: dK MMAs committed to ord[1] so far

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dm_pt = Int32(0)
        while work_tile.is_valid_tile:
            self._dm_prog(Int32(2), dm_pt, Int32(0), Int32(1))
            _, kv_block, head_idx, _, batch_idx, _, m_block_min, m_block_max, process_tile = (
                self._work_tile(work_tile, cta_rank_in_cluster, SeqlenInfoCls, block_info)
            )

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    kv_block // self.kv_subtile_factor,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                block_iter_count = (m_block_max - m_block_min) * self.dm_G

            if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL):
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    accumulate_dV = False

                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dP.T = V    @ dO.T
                    # 3. dK   = dS.T @ Q
                    # 4. dV   = P.T  @ dO
                    # 5. dQ   = dS   @ K

                    main_loop_iters = block_iter_count

                    # empty waits
                    # pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    # pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # 1) S.T = K @ Q.T
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dQ.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dQ tmem overlaps with S
                        mma_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        producer_phase_acc ^= 1

                        # 2) dP.T = V @ dO.T
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        pipeline_S_P.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dP tmem overlaps with S
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                        # 3) dK = dS.T @ Q
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                        mma_dsq_fn(B_idx=consumer_state_Q.index, zero_init=not accumulate_dK)
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()
                        accumulate_dK = True

                        # 4) dV = P.T @ dO
                        # Note: if dS is written to tmem, P must be written to tmem
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=not accumulate_dV)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                        accumulate_dV = True

                        # 5) dQ = dS @ K
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1
            elif const_expr(self.use_2cta_instrs):
                if is_leader_cta and process_tile:
                    if const_expr(self.dm_pers > 0):
                        # DM pers: acc parity flips M + 1 times per tile but the S/P and dP empty barriers complete
                        # M times, so restart each tile from the running base (see dm_ph_base)
                        producer_phase_acc = dm_ph_base
                    accumulate_dK = False
                    # -----------------------------------------------------------
                    ###### Prologue
                    # -----------------------------------------------------------
                    # 1. S  = Q0 @ K.T
                    # 2. dP = V @ dOt.T
                    # 3. dV = P @ dO

                    # 1) S = K @ Q
                    pipeline_Q.consumer_wait(consumer_state_Q)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(3))
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    if const_expr(self.dm_pers > 0):
                        # DM pers: S(0) of this tile writes the TMEM columns that hold the previous tile's last dQ;
                        # wait until the reduce warps drained it (the main loop's S waits the same way)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(4))
                    if const_expr(self.dm_ordw_v):
                        self._dm_ord_wait(0, dm_dvn)   # previous work tile's last dV (SF in the S region)
                    mma_qk_fn(B_idx=consumer_state_Q.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                    # 2) dP = V @ dOt.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(5))
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(6))
                    if const_expr(self.dm_ordw_k):
                        self._dm_ord_wait(1, dm_dkn)   # previous work tile's last dK (SF in the dP region)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    # 3) dV = P.T @ dO
                    producer_phase_acc ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(7))
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                    if const_expr(self.dm_ordw_v):
                        self._dm_ord_commit(0, cta_group, cta_rank_in_cluster)
                        dm_dvn += 1
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    pipeline_Kt.consumer_wait(consumer_state_Kt)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(8))
                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dK   = dS.T @ Q
                    # 3. dP.T = V    @ dO.T
                    # 4. dQ   = dS   @ K
                    # 5. dV   = P.T  @ dO

                    main_loop_iters = (
                        block_iter_count - 1
                        if const_expr(self.use_block_sparsity)
                        else (m_block_max - m_block_min) * self.dm_G - 1
                    )

                    if const_expr(dm_trace is not None):
                        dm_acc0, dm_acc1, dm_acc2, dm_acc3 = Int64(0), Int64(0), Int64(0), Int64(0)
                        dm_acc4, dm_acc5, dm_acc6, dm_acc7 = Int64(0), Int64(0), Int64(0), Int64(0)
                        dm_tl0 = dm_utils.clock64()
                    for dm_mi in cutlass.range(main_loop_iters, unroll=1):
                        dmz = dm_utils.opaque_zero(dm_mi) if const_expr(self.dm_d3nh) else None   # DM d3nh
                        self._dm_prog(Int32(2), dm_pt, dm_mi, Int32(2))
                        # (1) S.T = K @ Q.T (next)
                        if const_expr(dm_trace is not None):
                            dm_t0 = dm_utils.clock64()
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        if const_expr(dm_trace is not None):
                            dm_acc0 = dm_acc0 + (dm_utils.clock64() - dm_t0)
                        self._tlev(dm_trace, dm_mi, 0, True)
                        if const_expr(dm_trace is not None):
                            dm_t0 = dm_utils.clock64()
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                        if const_expr(dm_trace is not None):
                            dm_acc1 = dm_acc1 + (dm_utils.clock64() - dm_t0)
                        self._tlev(dm_trace, dm_mi, 1, True)
                        if const_expr(self.dm_ordw_v):
                            # ordinv C0: dV(t) read SFA/SFB_dv (S region cols 32..51); S^T(t+1) overwrites the S region
                            self._dm_ord_wait(0, dm_dvn)
                        if const_expr(self.dm_ordpc == 1):
                            self._dm_ord_plant(tStS, self.dm_sfa_dv_col)
                        mma_qk_fn(B_idx=consumer_state_Q.index, dz=dmz)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        # pipeline_dS.consumer_wait(consumer_state_dS)
                        # (2) dK += dS.T @ Q (cur)
                        if const_expr(dm_trace is not None):
                            dm_t0 = dm_utils.clock64()
                        pipeline_Qt.consumer_wait(consumer_state_Qt)
                        if const_expr(dm_trace is not None):
                            dm_acc2 = dm_acc2 + (dm_utils.clock64() - dm_t0)
                        self._tlev(dm_trace, dm_mi, 2, True)
                        if const_expr(dm_trace is not None):
                            dm_t0 = dm_utils.clock64()
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                        if const_expr(dm_trace is not None):
                            dm_acc3 = dm_acc3 + (dm_utils.clock64() - dm_t0)
                        self._tlev(dm_trace, dm_mi, 3, True)
                        mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK, dz=dmz)
                        if const_expr(self.dm_ordw_k):
                            self._dm_ord_commit(1, cta_group, cta_rank_in_cluster)
                            dm_dkn += 1
                        accumulate_dK = True
                        pipeline_Qt.consumer_release(consumer_state_Qt)
                        consumer_state_Qt.advance()

                        # (3) dP.T = V @ dO.T (next)
                        if const_expr(dm_trace is not None):
                            dm_t0 = dm_utils.clock64()
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        if const_expr(dm_trace is not None):
                            dm_acc4 = dm_acc4 + (dm_utils.clock64() - dm_t0)
                        self._tlev(dm_trace, dm_mi, 4, True)
                        if const_expr(self.dm_ordw_k):
                            # ordinv C0: dK(t) read SFA/SFB_dk (dP region cols dS+32..); dP^T(t+1) overwrites the dP region
                            self._dm_ord_wait(1, dm_dkn)
                        if const_expr(self.dm_ordpc == 3):
                            self._dm_ord_plant(tStS, self.dm_sfa_dk_col)
                        mma_dov_fn(B_idx=consumer_state_dO.index, dz=dmz)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                        # DM rdo: readiness-based order - if dQ(t)'s operands are not ready yet but P(t+1) is, issue dV(t+1)
                        # first (independent TMEM columns and accumulators: arithmetic unchanged, bitwise)
                        dm_dv_first = False
                        if const_expr("rdo" in self.dm_opts):
                            dm_dq_rdy = pipeline_dS.sync_object_full.test_wait(consumer_state_dS.index, consumer_state_dS.phase)
                            dm_p_rdy = pipeline_S_P.sync_object_empty.test_wait(0, producer_phase_acc ^ 1)
                            dm_dv_first = Int32(dm_p_rdy) - Int32(dm_dq_rdy)   # 1 iff P(t+1) ready and dQ(t) not
                            if dm_dv_first > 0:
                                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc ^ 1)  # S -> P (already complete)
                                dm_utils.tc_fence_after()
                                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False, dz=dmz)
                        # (5) dQ = dS @ K (cur)
                        if const_expr(dm_trace is not None):
                            dm_t0 = dm_utils.clock64()
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        if const_expr(dm_trace is not None):
                            dm_acc5 = dm_acc5 + (dm_utils.clock64() - dm_t0)
                        self._tlev(dm_trace, dm_mi, 5, True)
                        if const_expr(dm_trace is not None):
                            dm_t0 = dm_utils.clock64()
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        if const_expr(dm_trace is not None):
                            dm_acc6 = dm_acc6 + (dm_utils.clock64() - dm_t0)
                        self._tlev(dm_trace, dm_mi, 6, True)
                        mma_dsk_fn(dz=dmz)
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1
                        producer_phase_dQ ^= 1

                        # (4) dV += P.T @ dO (next)
                        producer_phase_acc ^= 1
                        if const_expr(dm_trace is not None):
                            dm_t0 = dm_utils.clock64()
                        if const_expr("rdo" not in self.dm_opts):
                            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)  # S -> P
                        else:
                            if dm_dv_first <= 0:
                                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)  # S -> P
                        if const_expr(dm_trace is not None):
                            dm_acc7 = dm_acc7 + (dm_utils.clock64() - dm_t0)
                        self._tlev(dm_trace, dm_mi, 7, True)
                        if const_expr("rdo" not in self.dm_opts):
                            mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False, dz=dmz)
                            if const_expr(self.dm_ordw_v):
                                self._dm_ord_commit(0, cta_group, cta_rank_in_cluster)
                                dm_dvn += 1
                            if const_expr(self.dm_ordpc == 2):
                                # DEBUG planted control (hd256 pc1/pc2 analogue): SFA_dv poison right after dV(t) issue; with
                                # ordw it is issued after dV(t)'s completion wait (the ordw protocol), without it races dV(t)
                                if const_expr(self.dm_ordw_v):
                                    self._dm_ord_wait(0, dm_dvn)
                                self._dm_ord_plant(tStS, self.dm_sfa_dv_col)
                        else:
                            if dm_dv_first <= 0:
                                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False, dz=dmz)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                    if const_expr(dm_trace is not None):
                        dm_tot = dm_utils.clock64() - dm_tl0
                        if cute.arch.lane_idx() == 0:
                            dm_base = dm_trace.iterator.toint()
                            for k_, a_ in enumerate((dm_acc0, dm_acc1, dm_acc2, dm_acc3, dm_acc4, dm_acc5, dm_acc6, dm_acc7, dm_tot, Int64(main_loop_iters))):
                                dm_utils.red_add_u64(dm_base + 8 * k_, a_)
                    if const_expr(self.dm_pers == 0):
                        pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)  # unmatched by the compute warps (harmless with one tile per CTA)

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(9))
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(14))

                    # -----------------------------------------------------------
                    # Tail: Remaining dK and dQ
                    # -----------------------------------------------------------
                    # pipeline_dS.consumer_wait(consumer_state_dS)
                    # dK += dS.T @ Q
                    pipeline_Qt.consumer_wait(consumer_state_Qt)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(10))
                    mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                    if const_expr(self.dm_ordw_k):
                        self._dm_ord_commit(1, cta_group, cta_rank_in_cluster)
                        dm_dkn += 1
                    pipeline_Qt.consumer_release(consumer_state_Qt)
                    consumer_state_Qt.advance()
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

                    # dQ = dS @ K
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(11))
                    cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(12))
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(13))
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dS.consumer_release(consumer_state_dS)
                    pipeline_Kt.consumer_release(consumer_state_Kt)
                    consumer_state_dS.advance()
                    consumer_state_Kt.advance()
                    dS_cluster_phase ^= 1
                    producer_phase_dQ ^= 1

                    producer_phase_acc ^= 1
                    if const_expr(self.dm_pers > 0):
                        dm_ph_base = dm_ph_base ^ (block_iter_count & 1)
            else:
                if is_leader_cta and process_tile:
                    if const_expr(self.dm_pers > 0):
                        producer_phase_acc = dm_ph_base
                    accumulate_dK = False
                    accumulate_dV = False
                    # -----------------------------------------------------------
                    ###### Prologue
                    # -----------------------------------------------------------
                    # 1. S.T  = K @ Q.T
                    # 2. dP.T = V @ dO.T
                    # 3. dV   = P.T @ dO   (split_P_dS: issued in the main loop, after QK_1)

                    # 1) S.T = K @ Q.T
                    self._dm_prog(Int32(2), dm_pt, producer_phase_acc, Int32(3))
                    handle_Q = pipeline_Q_consumer.wait_and_advance()
                    self._dm_prog(Int32(2), dm_pt, Int32(0), Int32(4))
                    if const_expr(self.split_P_dS):
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_S)
                    else:
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    self._dm_prog(Int32(2), dm_pt, Int32(0), Int32(5))
                    mma_qk_fn(B_idx=handle_Q.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                    if const_expr(self.split_P_dS):
                        producer_phase_S ^= 1

                    # 2) dP.T = V @ dO.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    self._dm_prog(Int32(2), dm_pt, Int32(0), Int32(6))
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    self._dm_prog(Int32(2), dm_pt, Int32(0), Int32(7))
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                    self._dm_prog(Int32(2), dm_pt, Int32(0), Int32(8))
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    producer_phase_acc ^= 1
                    # 3) dV = P.T @ dO
                    if const_expr(not self.split_P_dS):
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                        self._dm_prog(Int32(2), dm_pt, Int32(0), Int32(9))
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # Per trip t (the CTA owns a KV tile, so S and dP come out transposed):
                    # 1.  S.T  = K    @ Q.T    (tile t+1)
                    # 1b. dV  += P.T  @ dO     (tile t; split_P_dS only -- otherwise at 5)
                    # 2.  dK  += dS.T @ Q
                    # 3.  dQ   = dS   @ K
                    # 4.  dP.T = V    @ dO.T   (tile t+1)
                    # 5.  dV  += P.T  @ dO     (tile t+1; aliased layout only)

                    # For block sparsity, we use block_iter_count; for dense, use m_block range
                    # MMA doesn't need actual m_block indices, just the iteration count
                    main_loop_iters = (
                        block_iter_count - 1
                        if const_expr(self.use_block_sparsity)
                        else (m_block_max - m_block_min) * self.dm_G - 1
                    )

                    handle_Q_next = handle_Q
                    for dm_mi in cutlass.range(main_loop_iters, unroll=1):
                        self._dm_prog(Int32(2), dm_pt, dm_mi, Int32(2))
                        # (1) S.T = K @ Q.T
                        handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                        if const_expr(self.split_P_dS):
                            # S empty means "S_t loaded", not "P_t stored", so this QK
                            # overlaps the softmax of tile t.
                            pipeline_S_P.sync_object_empty.wait(0, producer_phase_S)
                        mma_qk_fn(B_idx=handle_Q_next.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        if const_expr(self.split_P_dS):
                            producer_phase_S ^= 1
                            # (1b) dV += P_t.T @ dO_t, issued after QK_{t+1} so QK never
                            # waits behind the P store.
                            pipeline_P.consumer_wait(consumer_state_P)
                            mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=not accumulate_dV)
                            accumulate_dV = True
                            pipeline_P.consumer_release(consumer_state_P)
                            consumer_state_P.advance()
                            pipeline_dO.consumer_release(consumer_state_dO)
                            consumer_state_dO.advance()

                        # (2) dK += dS.T @ Q
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
                        handle_Q.release()

                        # (3) dQ = dS @ K
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()

                        # (4) dP.T = V @ dO.T
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                        # (5) dV += P.T @ dO   (split_P_dS: done at (1b) of the next trip)
                        producer_phase_acc ^= 1
                        if const_expr(not self.split_P_dS):
                            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                            mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                            pipeline_dO.consumer_release(consumer_state_dO)
                            consumer_state_dO.advance()

                        handle_Q = handle_Q_next

                    if const_expr(self.split_P_dS):
                        # the last tile's dV += P.T @ dO; there is no further QK to overlap
                        pipeline_P.consumer_wait(consumer_state_P)
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=not accumulate_dV)
                        pipeline_P.consumer_release(consumer_state_P)
                        consumer_state_P.advance()
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                    elif const_expr(self.dm_pers == 0):
                        # (an extra S_P full arrival nobody waits for: with persistent tiles it would shift the
                        # next tile's S_P phase, so DM pers drops it)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )

                    # signal to the epilogue that dV is ready
                    # pipeline_dKV.producer_acquire(producer_state_dKV)
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    # pipeline_dKV.producer_commit(producer_state_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    # producer_state_dKV.advance()
                    # pipeline_dKV.producer_acquire(producer_state_dKV)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # -----------------------------------------------------------
                    # Tail: Remaining dK and dQ
                    # -----------------------------------------------------------
                    # 1) dK += dS.T @ Q
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

                    # 2) dQ = dS @ K
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    handle_Q.release()
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    producer_phase_acc ^= 1
                    if const_expr(self.dm_pers > 0):
                        dm_ph_base = dm_ph_base ^ ((m_block_max - m_block_min) & 1)
            self._dm_prog(Int32(2), dm_pt, Int32(0), Int32(15))
            dm_pt = dm_pt + 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # Currently it hangs if we have this S_P.producer_tail, will need to understand why
        # pipeline_S_P.producer_tail(producer_state_S_P)
        # pipeline_dP.producer_tail(producer_state_dP)
        # pipeline_dKV.producer_tail(producer_state_dKV)
        # pipeline_dQ.producer_tail(producer_state_dQ)

    @cute.jit
    def _dm_ord_commit(self, k: int, cta_group, cta_rank):
        """ordinv ordw: tcgen05.commit (this MMA warp's cta_group) of every prior MMA -> ord[k] of this (leader) CTA."""
        with cute.arch.elect_one():
            cute.nvgpu.tcgen05.commit(self.dm_ord_ptr + k, 1 << cta_rank, cta_group)

    @cute.jit
    def _dm_ord_wait(self, k: int, n: Int32):
        """ordinv ordw: wait for completion #(n-1) of ord[k] (n = commits issued so far; at most one outstanding)."""
        if n > 0:
            cute.arch.mbarrier_wait(self.dm_ord_ptr + k, phase=(n - 1) & 1)
            dm_utils.tc_fence_after()

    @cute.jit
    def _dm_ord_plant(self, tStS: cute.Tensor, col: int):
        """ordinv DEBUG planted control: tcgen05.st 0x80808080 (E8M0 2.0) into 4 SF columns of this warp's lane quadrant."""
        a = ((cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32) << 16
        a = a + Int32(tStS.iterator.toint()) - Int32(self.tmem_S_offset) + Int32(col)
        w = Int32(-2139062144)
        dm_utils.tmem_st_x4(a, w, w, w, w)
        cute.arch.fence_view_async_tmem_store()

    def _check_dS_exchange_layouts(self):
        """The dS exchange of the M=64 layout copies the peer's Q-column half of this CTA's
        dS^T from sdS_dK (K-major (kv, q), q = tile_m // 2 per K block) into the peer's sdS
        (M-major (q, kv)) at this CTA's KV-row block (_dS_exchange_source). Check that
        contract on the layouts: same swizzle, both halves one contiguous block at the same
        offset, same element order."""
        src = self.sdSt_layout.outer  # ((kv, q16), 1, (q16_chunks, q_half))
        dst = cute.logical_divide(
            self.sdS_layout.outer, (None, None, cute.size(self.sdS_layout.outer.shape[2]) // 2)
        )  # ((q, kv16), 1, (kv16_chunks, kv_block))
        assert str(self.sdSt_layout.inner) == str(self.sdS_layout.inner)
        assert cute.crd2idx((0, 0, (0, 1)), src) == cute.crd2idx((0, 0, (0, 1)), dst)
        for kv in (0, 1, 17, self.tile_n - 1):
            for q in (0, 1, 15, 16, self.tile_m // 2 - 1):
                assert cute.crd2idx(((kv, q % 16), 0, (q // 16, 1)), src) == cute.crd2idx(
                    ((q, kv % 16), 0, (kv // 16, 1)), dst
                ), "dS exchange: sdS_dK half is not in the peer's dQ A-operand layout"

    def _dS_exchange_source(
        self,
        sdS_xchg: Optional[cute.Tensor],
        sdS_dK: Optional[cute.Tensor],
        peer_cta_rank_in_cluster: Int32,
    ) -> cute.Tensor:
        """The SMEM tile this CTA sends its peer in the dS exchange: the peer's Q-column half of
        this CTA's dS^T, in the layout of the peer's dQ A-operand half (sdS, M-major (q, kv))
        at this CTA's KV-row block.

        With the M=128 layout the compute warps stage that half in sdS_xchg. With P / dS^T in
        SMEM (M=64 layout) it already sits in sdS_dK, the dK A operand, as the peer's K block
        (the layout contract is checked at setup, _check_dS_exchange_layouts).
        """
        if const_expr(sdS_xchg is not None):
            return sdS_xchg
        assert sdS_dK is not None
        return sdS_dK[None, None, (None, peer_cta_rank_in_cluster)]

    @cute.jit
    def _dm_claim_loop(self, params, sbase: Int32, counter: cutlass.Int64):
        """DM pdyn producer (empty warp, lane 0). Leader CTA: claim cluster tiles with a global atomic, publish each to
        both CTAs' slot k % 2; the claim that is past the end is published too (terminates every role). Peer CTA:
        arm its own full barrier per slot (4 tx bytes come from the leader's st.async) until the end marker."""
        n_cl = params.total_blocks // params.cluster_shape_mn[0]
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if cute.arch.lane_idx() == 0:
            k = Int32(0)
            go = Int32(1)
            while go > 0:
                s = k & 1
                if rank == 0:
                    if k >= 1:
                        # claim tile k only when this CTA has finished tile k - 1 (its dK/dV epilogue drained): a
                        # claimed-but-not-started tile would block the deterministic semaphore chains through it
                        dm_utils.mbar_wait_parity(sbase + 8 * 2, (k - 1) & 1)
                    if k >= 2:
                        dm_utils.mbar_wait_parity_cluster(sbase + 8 * 5 + 8 * s, ((k >> 1) - 1) & 1)
                    t = dm_utils.atom_add_global_u32(counter, Int32(1))
                    dm_utils.st_shared_b32(sbase + 8 * 7 + 8 * s, t)
                    dm_utils.mbar_arrive_local(sbase + 8 * 3 + 8 * s)
                    dm_utils.st_async_peer_b32(sbase + 8 * 7 + 8 * s, sbase + 8 * 3 + 8 * s, Int32(1), t)
                    go = Int32(t < n_cl)
                else:
                    dm_utils.mbar_arrive_expect_tx(sbase + 8 * 3 + 8 * s, 4)
                    dm_utils.mbar_wait_parity(sbase + 8 * 3 + 8 * s, (k >> 1) & 1)
                    t = dm_utils.ld_shared_b32(sbase + 8 * 7 + 8 * s)
                    go = Int32(t < n_cl)
                k = k + 1

    @cute.jit
    def _dm_prog(self, slot: Int32, tile: Int32, it: Int32, step: Int32):
        # DM prog (debug): latest (tile, iter, step) per role, zero-copy to host-mapped memory (hang localization)
        if const_expr("prog" in self.dm_opts):
            if cute.arch.lane_idx() == 0:
                dm_utils.st_sys_u32(self.dm_prog_addr + 4 * (cute.arch.block_idx()[0] * 32 + slot),
                                    (tile << 24) + (it << 8) + step)
                if slot == 2:
                    dm_utils.st_sys_u32(self.dm_prog_addr + 4 * (cute.arch.block_idx()[0] * 32 + 8 + step),
                                        (tile << 24) + (it << 8) + step)

    @cute.jit
    def kv_head_of(self, head_idx):
        """KV head of a scheduler head (NOTE [SM100 backward coordinate spaces])."""
        return bwd_kv_head(head_idx, qhead_per_kvhead=self.qhead_per_kvhead, pack_gqa=self.pack_gqa)

    @cute.jit
    def _work_tile(self, work_tile, cta_rank, SeqlenInfoCls, block_info):
        """Decode a scheduler work tile (NOTE [SM100 backward coordinate spaces]): the logical
        KV tile and this CTA's block of it, the scheduler and KV heads, the sequence, the
        m_block range and whether the tile has work at all (block sparsity refines that from
        its own block count)."""
        work_n, head_idx, batch_idx, _ = work_tile.tile_idx
        kv_block = work_kv_block(work_n, cta_rank, cta_group_size=self.cta_group_size)
        if const_expr(self.dm_gloop):
            # gloop: the scheduler head is the KV head; head_idx = first query head of the group (iteration
            # it of the qh * L stream uses query head head_idx + it // L)
            kv_head = head_idx
            head_idx = head_idx * self.qhead_per_kvhead
        else:
            kv_head = self.kv_head_of(head_idx)
        seqlen = SeqlenInfoCls(batch_idx)
        m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, work_n)
        process_tile = (
            const_expr(not self.is_local and not self.is_varlen_q) or m_block_min < m_block_max
        )
        return (
            work_n,
            kv_block,
            head_idx,
            kv_head,
            batch_idx,
            seqlen,
            m_block_min,
            m_block_max,
            process_tile,
        )

    @cute.jit
    def _cluster_S_coords(self, thr_mma_S, work_n, m_block):
        """Identity coordinates of this CTA's part of the (C * tile_n, tile_m) S tile of a
        work tile: (kv_row, packed_q_row) per accumulator element."""
        cS = cute.make_identity_tensor((self.cluster_tile_n, self.tile_m))
        cS = cute.domain_offset((work_n * self.cluster_tile_n, m_block * self.tile_m), cS)
        return thr_mma_S.partition_C(cS)

    @cute.jit
    def _dm_write_dq_sf(self, t: Int32, dm_sdm: Int32, dm_mbar: Int32, cta_rank: Int32):
        """dQ (M=128 2-CTA, K=256) scale factors for tile t: SFA const 1.0 (2 cols); SFB group g (2 cols) holds the 4
        key-block bytes of CTA g's keys (k-steps 4g..4g+3). Uniform over n, so every lane gets the same words."""
        slot = t & 1
        if const_expr("dqx" not in self.dm_abl):
            dm_utils.mbar_wait_parity(dm_mbar + 8 * slot, (t >> 1) & 1)
        own = dm_utils.ld_shared_b32(dm_sdm + 4 * (24 + slot))
        peer = dm_utils.ld_shared_b32(dm_sdm + 4 * (32 + slot))
        w0 = own
        w1 = peer
        if cta_rank != 0:
            w0 = peer
            w1 = own
        lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
        one = Int32(0x7F7F7F7F)
        dm_utils.tmem_st_x4((lane << 16) + self.dm_sfa_dq_col, one, one, w0, w0)
        dm_utils.tmem_st_x4((lane << 16) + self.dm_sfa_dq_col + 4, w1, w1, one, one)

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_copy_t2r,
        thr_mma_S,
        batch_idx,
        head_idx,
        m_block,
        work_n,
        softmax_scale,
        seqlen_info,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        wg_split=None,
    ):
        """Apply forward score modification for SM100 backward pass."""
        # In bwd, S is computed as K @ Q.T so dimensions are (tile_n, tile_m); the partition
        # of the work tile's coordinates gives this CTA its own rows.
        tScS = self._cluster_S_coords(thr_mma_S, work_n, m_block)
        tScS_idx = thr_copy_t2r.partition_D(tScS)
        if const_expr(wg_split is not None):
            # M=64 layout: the thread's chunk is the warpgroup's half of the partition
            tScS_idx = self.split_wg(tScS_idx, wg_split[0], wg_split[1])

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_idx,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.pack_factor,
            transpose_indices=True,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor,
        score_tensor,
        index_tensor,
        batch_idx,
        head_idx,
        softmax_scale,
        seqlen_info,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ):
        """Apply backward score modification (joint graph) for SM100."""
        apply_score_mod_bwd_inner(
            grad_tensor,
            score_tensor,
            index_tensor,
            self.score_mod_bwd,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.pack_factor,
            transpose_indices=True,
        )

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        sdS_xchg: Optional[cute.Tensor],
        sP: Optional[cute.Tensor],
        sdS_dK: Optional[cute.Tensor],
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_P: Optional[PipelineAsync],
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        sDm: Optional[cute.Tensor] = None,
        dm_xchg_mbar=None,
        dm_trace=None,
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.cluster_tile_n if self.acc_m64 else self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.cluster_tile_n if self.acc_m64 else self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # if const_expr(self.SdP_swapAB):
        if const_expr(True):
            sLSE_2D = layout_utils.transpose_view(sLSE_2D)
            sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        assert not self.dm_dsq or (not self.acc_m64 and self.use_2cta_instrs and not self.dm_cwg4 and not self.split_P_dS
                                   and self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED and len(self.compute_warp_ids) == 8
                                   and not self.use_block_sparsity and "ilp2" not in self.dm_opts
                                   and "early_dq" not in self.dm_opts and not self.xchg_aliases_dQaccum), "dsq: unsupported config"
        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        # tidx = cute.arch.thread_idx()[0] - (cute.arch.WARP_SIZE * self.compute_warp_ids[0])
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        if const_expr(not self.acc_m64):
            tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
            # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
            tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
            # Rebase the iterator explicitly (P may alias S), otherwise the tmem address is wrong
            tStP = cute.make_tensor(
                tStS.iterator + (self.tmem_P_offset - self.tmem_S_offset), tStP.layout
            )
            tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
            tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
            tdPtdS = cute.composition(
                tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1)
            )
            tdPtdS = cute.make_tensor(
                tdPtdP.iterator + (self.tmem_dS_offset - self.tmem_dP_offset), tdPtdS.layout
            )
            tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
            tdPcdS = cute.composition(
                tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1)
            )

            # 2-CTA assumes: repetiton should always be 32 & 16
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
            )
            # DM: fp8 P / dS -> 8 x 32-bit columns per 32-query stage
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32
            )

            # tmem -> rmem
            thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
            tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
            tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
            tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
            t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)
            # ((32, 1), 2, 1, 1, STAGE)
            tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
            tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
            # rmem -> tmem
            thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
            tScP_r2t = thr_copy_r2t.partition_S(tScP)
            tStP_r2t = thr_copy_r2t.partition_D(tStP)
            tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
            tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
            if const_expr(self.dm_dsq):
                # DM dsq: fp32 dS stored back over dP with the dP load's partitioning (32 x 32b per stage)
                dm_thr_copy_r2t_f32 = copy_utils.make_tmem_copy(
                    cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), Float32), num_wg
                ).get_slice(tidx)
                dm_tdPtdP_r2t_f32 = dm_thr_copy_r2t_f32.partition_D(tdPtdP)
            # rmem -> smem
            # This part is a bit iffy, we might be making a lot of assumptions here
            copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
                LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r
            )
            thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)

            # We assume the swizzle (i.e. layout.inner) stays the same
            sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
                self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
            )
            sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))  # ((8,16), (64,2))
            if const_expr(self.ds_dtype.width == 8):
                # DM: the dQ A operand (MN-major, 64 queries per CTA) is SW64 with offset m + 64*key; each per-CTA
                # query half is one 128-key x 64-query block of 8 KiB. The bf16 epilogue
                # layout FA4 reuses here (SW128 row-major) does not match it for 8-bit types.
                half_elems = (self.tile_m // 2) * self.tile_n
                sdS_layout = cute.make_layout(
                    ((8, self.tile_n // 8), (self.tile_m // 2, 2)),
                    stride=((self.tile_m // 2, 8 * (self.tile_m // 2)), (1, half_elems)),
                )
            # Need to group into 1 mode to be compatible w thr_copy_r2s
            sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
            sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
            tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

            if const_expr(sdS_xchg is not None):
                sdS_xchg_layout_epi = sdS_layout
                if const_expr(self.dm_cwg4):
                    # each thread holds a single query half: fold both halves onto the one exchange block
                    sdS_xchg_layout_epi = cute.make_layout(
                        ((8, self.tile_n // 8), (self.tile_m // 2, 2)),
                        stride=((self.tile_m // 2, 8 * (self.tile_m // 2)), (1, 0)),
                    )
                    sdS_xchg_layout_epi = cute.make_layout((sdS_xchg_layout_epi.shape,), stride=(sdS_xchg_layout_epi.stride,))
                sdS_xchg_epi = cute.make_tensor(
                    cute.recast_ptr(
                        sdS_xchg.iterator,
                        cute.make_swizzle(2, 4, 3) if const_expr(self.ds_dtype.width == 8) else sdS_epi_layout.inner,
                    ),
                    sdS_xchg_layout_epi,
                )
                tRS_sdS_xchg = thr_copy_r2s.partition_D(sdS_xchg_epi)
        else:
            # NOTE [M=64 accumulator layout]: the layout-aware tiled copy gives each of the 128
            # lanes its 64 columns as two 32-column chunks (lanes 0-63 hold columns [0, 64),
            # lanes 64-127 columns [64, 128) of the same rows); warpgroup g takes chunk g, so
            # every thread owns one (row, 32-column) piece of S^T / dP^T and the per-thread
            # register vector is a single stage.
            wg_idx = tidx // 128
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx % 128)
            tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
            tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
            tStS_t2r = self.split_wg(thr_copy_t2r.partition_S(tStS), wg_idx, num_wg)
            tdPtdP_t2r = self.split_wg(thr_copy_t2r.partition_S(tdPtdP), wg_idx, num_wg)
            tScS_t2r = self.split_wg(thr_copy_t2r.partition_D(tScS), wg_idx, num_wg)
            t0ScS_t2r = self.split_wg(tiled_copy_t2r.get_slice(0).partition_D(tScS), 0, num_wg)
            tSsLSE = self.split_wg(
                thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D)), wg_idx, num_wg
            )
            tSsdPsum = self.split_wg(
                thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D)), wg_idx, num_wg
            )
            # P^T and dS^T are staged in SMEM (the A operands of the dV / dK MMAs) with the same
            # store op the dS exchange uses; the epilogue layout and the MMA A layout share the
            # 128B swizzle, so the MMA reads what the compute warps wrote.
            thr_copy_r2t = None
            tScP_r2t = tStP_r2t = tdPcdS_r2t = tdPtdS_r2t = None
            copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
                LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r
            )
            thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx % 128)
            sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
                self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
            )
            sdS_layout_2d = cute.slice_(sdS_epi_layout.outer, (None, None, 0))
            # Rank 3 like the TMEM fragment, ((rows, cols), 1, 1), so the layout-aware r2s
            # tiled copy can partition it.
            sdS_layout = cute.make_layout(
                (sdS_layout_2d.shape, 1, 1), stride=(sdS_layout_2d.stride, 0, 0)
            )
            sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
            tRS_sdS = self.split_wg(thr_copy_r2s.partition_D(sdS_epi), wg_idx, num_wg)
            sP_epi = cute.make_tensor(sP.iterator, sdS_layout)
            tRS_sP = self.split_wg(thr_copy_r2s.partition_D(sP_epi), wg_idx, num_wg)
            sdS_dK_epi = cute.make_tensor(sdS_dK.iterator, sdS_layout)
            tRS_sdS_dK = self.split_wg(thr_copy_r2s.partition_D(sdS_dK_epi), wg_idx, num_wg)
            # Q-column half this thread's chunk belongs to (from its partition coordinates); the
            # CTA keeps the half it owns for dQ (the peer's half reaches it from sdS_dK, see the
            # exchange copy below).
            q_half = tScS_t2r[0][1] // (self.tile_m // 2)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dS_cluster_empty_phase = Int32(1)
        # 2-CTA: CTA 0 exchanges stage 1 (bottom half), CTA 1 exchanges stage 0 (top half)
        exchange_stage = cta_rank_in_cluster ^ 1 if const_expr(self.use_2cta_instrs) else Int32(0)
        if const_expr(self.use_2cta_instrs):
            peer_cta_rank_in_cluster = cta_rank_in_cluster ^ 1
            sdS_xchg_src = self._dS_exchange_source(sdS_xchg, sdS_dK, peer_cta_rank_in_cluster)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # consumer_phase_S_P_dP = Int32(0)
        consumer_state_S = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_state_P = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        producer_state_dS = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        assert self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED, "DM bwd: 2-CTA pipelined schedule only"
        dm_t = Int32(0)  # tiles whose dQ scale word has been sent
        dm_sdm = sDm.iterator.toint()
        dm_mbar = dm_xchg_mbar.toint()
        dm_tid = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        dm_tr = dm_trace
        if const_expr(dm_tr is not None):
            dm_s0, dm_s1, dm_s2, dm_s3, dm_s4 = Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
            dm_s5, dm_s6, dm_s7, dm_s8, dm_s9 = Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
            dm_s10, dm_s11, dm_s12, dm_s13 = Int32(0), Int32(0), Int32(0), Int32(0)
            dm_tprev = dm_utils.clock32()
            dm_tl0 = dm_utils.globaltimer()
        dm_pt = Int32(0)
        while work_tile.is_valid_tile:
            self._dm_prog(Int32(3) + cute.arch.warp_idx() // 4 % 2, dm_pt, Int32(0), Int32(1))
            (
                work_n,
                kv_block,
                head_idx,
                kv_head,
                batch_idx,
                seqlen,
                m_block_min,
                m_block_max,
                process_tile,
            ) = self._work_tile(work_tile, cta_rank_in_cluster, SeqlenInfoCls, block_info)

            recompute_fastdiv_mods_q = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if const_expr(fastdiv_mods is not None and fastdiv_mods[0] is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisorV2(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisorV2(seqlen.seqlen_k),
                )
            mask = AttentionMaskCls(seqlen)
            # TODO: condition mask_seqlen
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=work_n,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                mask_mod=self.mask_mod,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )

            # prefetch_LSE = not self.is_causal
            prefetch_LSE = "pflse" in self.dm_opts

            curr_q_cnt = Int32(0)
            curr_q_idx = None
            curr_full_cnt = Int32(0)
            curr_full_idx = None
            loop_count = m_block_max - m_block_min
            dm_L = loop_count                       # gloop: m blocks per query head
            if const_expr(self.dm_gloop):
                loop_count = loop_count * self.dm_G
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                (
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    loop_count,
                ) = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    kv_block // self.kv_subtile_factor,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = loop_count > Int32(0)

            # DM: hoisted per-key-row scales and per-stage query-block offsets; prefetch the first query-block scales
            dm_idx0 = thr_copy_t2r.partition_D(self._cluster_S_coords(thr_mma_S, work_n, 0))
            dm_kv_blk = dm_idx0[0][0] // 32
            # keys past seqlen_k (the empty half of a partial 2-CTA cluster) are masked; clamp their scale index
            # so the dQ SFB byte never comes from an out-of-bounds (possibly 0xFF = NaN) scale read
            dm_kv_blk = cutlass.min(dm_kv_blk, (seqlen.seqlen_k + 31) // 32 - 1)
            dm_ks = Float32(aux_data.tensors[1][batch_idx, kv_head, dm_kv_blk])
            dm_vs = Float32(aux_data.tensors[2][batch_idx, kv_head, dm_kv_blk])
            dm_ksb = dm_utils.pow2_exponent_byte(dm_ks)
            dm_nst = cute.size(tScS_t2r, mode=[1])
            dm_qoff = [dm_idx0[None, st, 0, 0][0][1] // 32 for st in range(dm_nst)]
            if const_expr(dm_nst == 1):
                dm_qoff = dm_qoff + dm_qoff   # cwg4: one stage per thread (second slot unused)
            dm_m0 = m_block_min * (self.tile_m // 32)
            dm_qs0 = Float32(aux_data.tensors[0][batch_idx, head_idx, dm_m0 + dm_qoff[0]])
            dm_qs1 = Float32(aux_data.tensors[0][batch_idx, head_idx, dm_m0 + dm_qoff[1]])
            dm_dos0 = Float32(aux_data.tensors[3][batch_idx, head_idx, dm_m0 + dm_qoff[0]])
            dm_dos1 = Float32(aux_data.tensors[3][batch_idx, head_idx, dm_m0 + dm_qoff[1]])
            # Mainloop
            # Block sparsity: iterate over sparse m_block count and derive actual m_block
            # from Q_IDX/FULL_Q_IDX tensors. Dense: iterate m_block_min..m_block_max directly.
            if const_expr("spf" in self.dm_opts):
                dm_Spf = cute.make_rmem_tensor(tScS_t2r.shape, Float32)   # S(t+1) prefetched during dS(t)
            dm_head0 = head_idx
            # DM msplit (default; nomsplit disables; causal, not gloop): split the m-block loop into masked head (diagonal tiles), unmasked middle
            # and masked tail (partial last m block) phases -> the mask code exists only in the masked copies (the
            # CTA-uniform mskipu test was if-converted into ~234 predicated-off instructions per warp-tile).
            dm_split = const_expr("nomsplit" not in self.dm_opts and self.is_causal and not self.is_local and not self.dm_gloop
                                  and self.mask_mod is None and mask.qhead_per_kvhead_packgqa == 1
                                  and not self.use_block_sparsity)
            if const_expr(dm_split):
                dm_hd = ((work_n + 1) * mask.tile_n + mask.tile_m - 1) // mask.tile_m - m_block_min
                dm_hd = cutlass.max(cutlass.min(dm_hd, loop_count), 0)
                dm_tl = Int32(0)
                if m_block_max * mask.tile_m > seqlen.seqlen_q and loop_count - dm_hd > 0:
                    dm_tl = Int32(1)
                dm_lo = (Int32(0), dm_hd, loop_count - dm_tl)
                dm_hi = (dm_hd, loop_count - dm_tl, loop_count)
                dm_nph = 3
            else:
                dm_lo = (Int32(0),)
                dm_hi = (loop_count,)
                dm_nph = 1
            for dm_ph in cutlass.range_constexpr(dm_nph):
              for iter_idx in cutlass.range(dm_lo[dm_ph], dm_hi[dm_ph], unroll=1):
                  self._dm_prog(Int32(3) + cute.arch.warp_idx() // 4 % 2, dm_pt, iter_idx, Int32(2))
                  m_block = m_block_min + iter_idx
                  if const_expr(self.dm_gloop):
                      dm_g = iter_idx // dm_L
                      m_block = m_block_min + iter_idx - dm_g * dm_L
                      head_idx = dm_head0 + dm_g
                  m_block_oob = False
                  is_full_block = False
                  if const_expr(self.use_block_sparsity):
                      m_block, is_full_block = get_m_block_from_iter_bwd(
                          iter_idx,
                          curr_q_cnt,
                          curr_q_idx,
                          curr_full_cnt,
                          curr_full_idx,
                          q_subtile_factor=self.q_subtile_factor,
                          m_block_max=m_block_max,
                      )
                      m_block_oob = m_block >= m_block_max
                  if const_expr(dm_tr is not None):
                      dm_now = dm_utils.clock32()
                      dm_s0 = dm_s0 + (dm_now - dm_tprev)
                      dm_tprev = dm_now
                  # Prefetch 1 stage of LSE
                  pipeline_LSE.consumer_wait(consumer_state_LSE)
                  if const_expr(dm_tr is not None):
                      dm_now = dm_utils.clock32()
                      dm_s1 = dm_s1 + (dm_now - dm_tprev)
                      dm_tprev = dm_now
                  tSrLSE_s2r = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
                  if const_expr(prefetch_LSE and not self.shuffle_LSE):
                      cute.autovec_copy(tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r)
                  # DM lseall: LSE for every stage loaded right after the LSE wait (before the S wait)
                  dm_lse_all = [None] * cute.size(tScS_t2r, mode=[1])
                  if const_expr("lseall" in self.dm_opts):
                      for st_ in cutlass.range_constexpr(cute.size(tScS_t2r, mode=[1])):
                          dm_lse_all[st_] = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
                          cute.autovec_copy(tSsLSE[None, st_, 0, 0, consumer_state_LSE.index], dm_lse_all[st_])

                  if const_expr("spf" in self.dm_opts and not self.split_P_dS):
                      # spf: S(t) was waited for and TMEM-loaded during dS(t-1) (except the first m_block)
                      tSrS_t2r = dm_Spf
                      if iter_idx == 0:
                          pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                          cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)
                      cute.arch.fence_view_async_tmem_load()
                  else:
                      if const_expr(self.split_P_dS):
                          pipeline_S_P.consumer_wait(consumer_state_S)
                      else:
                          pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                      if const_expr(dm_tr is not None):
                          dm_now = dm_utils.clock32()
                          dm_s2 = dm_s2 + (dm_now - dm_tprev)
                          dm_tprev = dm_now
                          self._tlev(dm_tr, iter_idx, 8, dm_tid == 0)
                      # pipeline_S_P.sync_object_full.wait(0, consumer_phase_S_P_dP)
                      #### TMEM->RMEM (Load S from TMEM)
                      tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                      cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)
                      if const_expr(self.dm_dsq and "esr" in self.dm_opts):
                          # DM esr: S(t) is in registers once this warp's TMEM load completed -> the dS helper may let
                          # dQ(t-1) overwrite the S tail (and write the dQ(t-1) SFs) now instead of after the exps
                          if iter_idx > 0:
                              cute.arch.fence_view_async_tmem_load()
                              dm_utils.tc_fence_before()
                              cute.arch.sync_warp()
                              if cute.arch.lane_idx() == 0:
                                  dm_utils.mbar_arrive(dm_mbar + 8 * 10)
                  if const_expr(dm_tr is not None):
                      dm_now = dm_utils.clock32()
                      dm_s3 = dm_s3 + (dm_now - dm_tprev)
                      dm_tprev = dm_now
                      self._tlev(dm_tr, iter_idx, 9, dm_tid == 0)

                  if const_expr(self.split_P_dS):
                      # S is in registers: release it before the softmax so the MMA warp can
                      # issue the next QK into the slot
                      cute.arch.fence_view_async_tmem_load()
                      cute.arch.sync_warp()
                      dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                      with cute.arch.elect_one():
                          pipeline_S_P.consumer_release(consumer_state_S)
                      consumer_state_S.advance()
                  elif const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_SERIAL):
                      # Signal S tmem load completion using pipeline_S_P: dP overlaps S
                      cute.arch.fence_view_async_tmem_load()
                      dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                      with cute.arch.elect_one():
                          pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                  elif const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
                      # Signal S tmem load completion using pipeline_dS: dQ overlaps S
                      # DM: the "S read" signal releases dQ(t-1), whose accumulator and scale factors live in the S
                      # columns, so every warp must have loaded S(t) first. early_dq: barrier right after the S load,
                      # then install dQ(t-1) scales and signal (FA4's early handoff); else at the stage-0 P barrier.
                      if const_expr("early_dq" in self.dm_opts):
                          cute.arch.fence_view_async_tmem_load()
                          self.compute_sync_barrier.arrive_and_wait()
                          if iter_idx > 0:
                              if const_expr(self.dm_dsq):
                                  if const_expr("esr" not in self.dm_opts):   # DM esr: arrived right after the S load instead
                                      # DM dsq: S(t) read by this warp (wait::ld done before the pair barrier) -> the reduce warps may let
                                      # dQ(t-1) overwrite the S tail; they install the dQ(t-1) SFs and commit pipeline_dS(t-1)
                                      dm_utils.tc_fence_before()
                                      cute.arch.sync_warp()
                                      if cute.arch.lane_idx() == 0:
                                          dm_utils.mbar_arrive(dm_mbar + 8 * 10)
                              else:
                                  self._dm_write_dq_sf(dm_t - 1, dm_sdm, dm_mbar, cta_rank_in_cluster)
                                  cute.arch.fence_view_async_tmem_store()
                                  dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                                  with cute.arch.elect_one():
                                      pipeline_dS.producer_commit(producer_state_dS)
                                  producer_state_dS.advance()

                  if const_expr(self.score_mod_bwd is not None):
                      tSrS_pre = cute.make_fragment_like(tSrS_t2r)
                      cute.autovec_copy(tSrS_t2r, tSrS_pre)

                  if const_expr(self.score_mod is not None):
                      # Apply score_mod FIRST -> matches forward
                      self.apply_score_mod(
                          tSrS_t2r,
                          thr_copy_t2r,
                          thr_mma_S,
                          batch_idx,
                          head_idx,
                          m_block,
                          work_n,
                          softmax_scale,
                          seqlen,
                          aux_data,
                          fastdiv_mods,
                          wg_split=(wg_idx, num_wg) if const_expr(self.acc_m64) else None,
                      )

                  #### APPLY MASK (after score_mod, matching forward pass order)
                  check_m_boundary = (m_block + 1) * self.tile_m > seqlen.seqlen_q
                  if const_expr(dm_split):
                      if const_expr(dm_ph != 1):
                          mask_fn(
                              tSrS_t2r,
                              m_block=m_block,
                              is_full_block=is_full_block,
                              check_m_boundary=check_m_boundary,
                          )
                  elif const_expr("nomskipu" not in self.dm_opts and self.is_causal and not self.is_local
                                and self.mask_mod is None and mask.qhead_per_kvhead_packgqa == 1):
                      # DM mskipu (default; nomskipu disables): CTA-uniform test - every query of the tile is >= every key of the cluster
                      if m_block * mask.tile_m < (work_n + 1) * mask.tile_n or check_m_boundary:
                          mask_fn(
                              tSrS_t2r,
                              m_block=m_block,
                              is_full_block=is_full_block,
                              check_m_boundary=check_m_boundary,
                          )
                  elif const_expr("mskip" in self.dm_opts and self.is_causal and not self.is_local
                                and self.mask_mod is None and mask.qhead_per_kvhead_packgqa == 1):
                      # DM mskip (opt-in; the dynamic branch makes causal ~8% SLOWER than always masking): per-thread causal offset (as in apply_mask_sm100_transposed); <= 0 means no
                      # element of this thread's rows is above the diagonal -> skip the R2P/SEL mask.
                      # ROW = query, COL = key coordinate (index 1/0 when swap_AB, as in the mask)
                      dm_ROW = 0 if const_expr(not mask.swap_AB) else 1
                      dm_COL = 1 - dm_ROW
                      dm_col_lim = mask.seqlen_k - work_n * mask.tile_n - tScS_t2r[0][dm_COL]
                      dm_row_lim = mask.seqlen_q - m_block * mask.tile_m - tScS_t2r[0][dm_ROW]
                      if dm_row_lim - dm_col_lim > 0 or dm_col_lim <= 0 or check_m_boundary:
                          mask_fn(
                              tSrS_t2r,
                              m_block=m_block,
                              is_full_block=is_full_block,
                              check_m_boundary=check_m_boundary,
                          )
                  else:
                      mask_fn(
                          tSrS_t2r,
                          m_block=m_block,
                          is_full_block=is_full_block,
                          check_m_boundary=check_m_boundary,
                      )
                  num_stages = cute.size(tScS_t2r, mode=[1])
                  # ---------------------------------------------
                  #### P = exp(S - LSE)
                  # ---------------------------------------------
                  lane_idx = cute.arch.lane_idx()
                  if const_expr(not self.acc_m64):
                      tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)  # 64
                      tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                  # DM: per-stage scales (query-block scales prefetched one m_block ahead; key-row scales hoisted)
                  dm_qsv = (dm_qs0, dm_qs1)
                  dm_dosv = (dm_dos0, dm_dos1)
                  dm_mn = cutlass.min(m_block + 1, m_block_max - 1)
                  dm_hn = head_idx
                  if const_expr(self.dm_gloop):
                      dm_itn = cutlass.min(iter_idx + 1, loop_count - 1)
                      dm_gn = dm_itn // dm_L
                      dm_mn = m_block_min + dm_itn - dm_gn * dm_L
                      dm_hn = dm_head0 + dm_gn
                  dm_qs0 = Float32(aux_data.tensors[0][batch_idx, dm_hn, dm_mn * (self.tile_m // 32) + dm_qoff[0]])
                  dm_qs1 = Float32(aux_data.tensors[0][batch_idx, dm_hn, dm_mn * (self.tile_m // 32) + dm_qoff[1]])
                  dm_dos0 = Float32(aux_data.tensors[3][batch_idx, dm_hn, dm_mn * (self.tile_m // 32) + dm_qoff[0]])
                  dm_dos1 = Float32(aux_data.tensors[3][batch_idx, dm_hn, dm_mn * (self.tile_m // 32) + dm_qoff[1]])
                  dm_c = [None] * num_stages
                  dm_a = [None] * num_stages
                  dm_qsb = [None] * num_stages
                  dm_sfa_dv = [None] * num_stages
                  for stage in cutlass.range_constexpr(num_stages):
                      dm_c[stage] = (dm_ks * dm_qsv[stage]) * Float32(math.log2(math.e))
                      dm_a[stage] = dm_vs * dm_dosv[stage]
                      dm_qsb[stage] = dm_utils.pow2_exponent_byte(dm_qsv[stage])
                      dm_sfa_dv[stage] = dm_utils.pow2_exponent_byte(dm_dosv[stage])
                  for stage in cutlass.range_constexpr(num_stages):
                      tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                      tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                      if const_expr("lseall" in self.dm_opts):
                          tSrLSE = dm_lse_all[stage]
                      elif const_expr(not self.shuffle_LSE and "lse2" in self.dm_opts and stage > 0):
                          # lse2: a fresh register buffer per stage (no WAR on stage 0's LSE regs -> loads can hoist)
                          tSrLSE = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
                          cute.autovec_copy(tSsLSE_cur, tSrLSE)
                      elif const_expr(not self.shuffle_LSE):
                          if const_expr(stage > 0 or not prefetch_LSE):
                              cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                          tSrLSE = tSrLSE_s2r
                      else:
                          tSrLSE = tSsLSE_cur[lane_idx]
                      for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                          if const_expr(not self.shuffle_LSE):
                              lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                          else:
                              lse_pair = (
                                  utils.shuffle_sync(tSrLSE, offset=2 * v),
                                  utils.shuffle_sync(tSrLSE, offset=2 * v + 1),
                              )
                          tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                              ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                              (dm_c[stage], dm_c[stage]),
                              (-lse_pair[0], -lse_pair[1]),
                          )
                          if const_expr("noexp" in self.dm_abl or self.dm_b6 > 0):
                              pass   # b6: exps below, per 32x32 block
                          elif const_expr("poly3" in self.dm_opts):
                              # FA4 degree-3 FMA polynomial for every element: bitwise = dm_common.dm_bwd_p_pair_poly in
                              # pass 0 (own-P delta keeps rowsum(dS)=0 at round-off; P error 8.8e-5 << 2e-3 tolerance)
                              tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = utils.ex2_emulation_2(tSrS_cur[2 * v], tSrS_cur[2 * v + 1])
                          elif const_expr(self.dm_p3 is not None and v % self.dm_p3[0] < self.dm_p3[1]):
                              # FA4 degree-3 FMA polynomial (8.8e-5) for a static subset of pairs (MUFU offload)
                              tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = utils.ex2_emulation_2(tSrS_cur[2 * v], tSrS_cur[2 * v + 1])
                          elif const_expr(self.dm_p5 is not None and v % self.dm_p5[0] < self.dm_p5[1]):
                              # accurate (1.7e-7) degree-5 FMA polynomial for a static subset of pairs (MUFU offload)
                              tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = dm_utils.ex2_poly5_2(tSrS_cur[2 * v], tSrS_cur[2 * v + 1])
                          else:
                              tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                              tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                      if const_expr(self.dm_b6 > 0 and "noexp" not in self.dm_abl):
                          # b6: this warp's 32 keys x this stage's 32 queries = one 32x32 block; deg-6 poly iff
                          # (qb + kb) % m == 0 (warp-uniform) - the same rule and op sequence as pass 0 -> bitwise P
                          if (m_block * (self.tile_m // 32) + dm_qoff[stage] + dm_kv_blk) % self.dm_b6 == 0:
                              for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                                  tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = dm_utils.ex2_poly6_2(tSrS_cur[2 * v], tSrS_cur[2 * v + 1])
                          else:
                              for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                                  tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                                  tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                      if const_expr(dm_tr is not None):
                          dm_now = dm_utils.clock32()
                          dm_s10 = dm_s10 + (dm_now - dm_tprev)
                          dm_tprev = dm_now
                      if const_expr(self.acc_m64):
                          # P^T goes to SMEM; S stays untouched in TMEM until the dS pipeline
                          # releases it, so no cross-warp barrier is needed before the store.
                          tSrP_cvt = cute.make_fragment_like(tSrS_cur, self.q_dtype)
                          utils.cvt_f16(tSrS_cur, tSrP_cvt)
                          cute.autovec_copy(tSrP_cvt, tRS_sP[None, 0, 0, 0])
                      else:
                          # DM: P^T codes with a pow2 scale per (key row, 32-query block); SFA_dV = s * dos
                          p_amax = Float32(1.0)
                          if const_expr("pamax" not in self.dm_abl and "amax3" not in self.dm_opts and "amaxt" not in self.dm_opts):
                              p_amax = Float32(0.0)
                              for v in cutlass.range_constexpr(cute.size(tSrS_cur)):
                                  p_amax = cute.arch.fmax(p_amax, tSrS_cur[v])
                          elif const_expr("pamax" not in self.dm_abl):
                              # 4 independent 3-input max chains (P >= 0)
                              pm = [utils.fmax(tSrS_cur[2 * t], tSrS_cur[2 * t + 1]) for t in range(4)]
                              for v in cutlass.range_constexpr(8, cute.size(tSrS_cur), 8):
                                  for t in cutlass.range_constexpr(4):
                                      pm[t] = utils.fmax(pm[t], tSrS_cur[v + 2 * t], tSrS_cur[v + 2 * t + 1])
                              p_amax = utils.fmax(utils.fmax(pm[0], pm[1]), pm[2], pm[3])
                          p_sb = dm_utils.pow2_sb(p_amax)
                          p_inv = dm_utils.inv_pow2_from_sb(p_sb)
                          dm_sfa_dv[stage] = dm_utils.add_sb_clamp(p_sb, dm_sfa_dv[stage])
                          tSrPq = cute.make_fragment_like(tSrS_cur, Float32)
                          for v in cutlass.range_constexpr(cute.size(tSrS_cur) // 2):
                              tSrPq[2 * v], tSrPq[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                  (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]), (p_inv, p_inv))
                          tSrP_r2t[None, stage, 0, 0].store(tSrPq.load().to(self.q_dtype))
                          if const_expr(dm_tr is not None):
                              dm_now = dm_utils.clock32()
                              dm_s11 = dm_s11 + (dm_now - dm_tprev)
                              dm_tprev = dm_now
                          if const_expr(stage == 0 and "noilp" in self.dm_opts):
                              if const_expr(self.split_P_dS):
                                  # wait until the previous pdo UMMA has consumed the P slot
                                  pipeline_P.producer_acquire(producer_state_P)
                              elif const_expr("early_dq" not in self.dm_opts):
                                  cute.arch.fence_view_async_tmem_load()
                                  # P overwrites S lanes another warp may still be loading, so
                                  # every warp must have loaded S before any stores P.
                                  if const_expr("fullbar" not in self.dm_opts and not self.dm_cwg4):
                                      # TMEM lanes are warp-quadrant private: only warps w and w+4 share lanes, and the
                                      # dQ SF columns are written per own lanes; the dQ MMA itself waits for every warp's
                                      # dS-pipe arrival -> a 64-thread pair barrier is enough
                                      cute.arch.barrier(
                                          barrier_id=8 + cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4,
                                          number_of_threads=64)
                                  else:
                                      self.compute_sync_barrier.arrive_and_wait()
                                  if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
                                      if iter_idx > 0:
                                          if const_expr(self.dm_dsq):
                                              if const_expr("esr" not in self.dm_opts):   # DM esr: arrived right after the S load instead
                                                  # DM dsq: S(t) read by this warp (wait::ld done before the pair barrier) -> the reduce warps may let
                                                  # dQ(t-1) overwrite the S tail; they install the dQ(t-1) SFs and commit pipeline_dS(t-1)
                                                  dm_utils.tc_fence_before()
                                                  cute.arch.sync_warp()
                                                  if cute.arch.lane_idx() == 0:
                                                      dm_utils.mbar_arrive(dm_mbar + 8 * 10)
                                          else:
                                              self._dm_write_dq_sf(dm_t - 1, dm_sdm, dm_mbar, cta_rank_in_cluster)
                                              cute.arch.fence_view_async_tmem_store()
                                              dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                                              with cute.arch.elect_one():
                                                  pipeline_dS.producer_commit(producer_state_dS)
                                              producer_state_dS.advance()
                          if const_expr("noilp" in self.dm_opts):
                              cute.copy(
                                  thr_copy_r2t,
                                  tSrP_r2t_f32[None, stage, None, None],
                                  tStP_r2t[None, stage, None, None],
                              )
                  if const_expr("noilp" not in self.dm_opts and not self.acc_m64):
                      # ilp: exps / P quantization of every stage first (no barrier between stages), then the S-read
                      # barrier, dQ SFs and the P TMEM stores
                      if const_expr(self.split_P_dS):
                          # wait until the previous pdo UMMA has consumed the P slot
                          pipeline_P.producer_acquire(producer_state_P)
                      elif const_expr("early_dq" not in self.dm_opts):
                          cute.arch.fence_view_async_tmem_load()
                          # P overwrites S lanes another warp may still be loading, so
                          # every warp must have loaded S before any stores P.
                          if const_expr("fullbar" not in self.dm_opts and not self.dm_cwg4):
                              # TMEM lanes are warp-quadrant private: only warps w and w+4 share lanes, and the
                              # dQ SF columns are written per own lanes; the dQ MMA itself waits for every warp's
                              # dS-pipe arrival -> a 64-thread pair barrier is enough
                              cute.arch.barrier(
                                  barrier_id=8 + cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4,
                                  number_of_threads=64)
                          else:
                              self.compute_sync_barrier.arrive_and_wait()
                          if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED):
                              if iter_idx > 0:
                                  if const_expr(self.dm_dsq):
                                      if const_expr("esr" not in self.dm_opts):   # DM esr: arrived right after the S load instead
                                          # DM dsq: S(t) read by this warp (wait::ld done before the pair barrier) -> the reduce warps may let
                                          # dQ(t-1) overwrite the S tail; they install the dQ(t-1) SFs and commit pipeline_dS(t-1)
                                          dm_utils.tc_fence_before()
                                          cute.arch.sync_warp()
                                          if cute.arch.lane_idx() == 0:
                                              dm_utils.mbar_arrive(dm_mbar + 8 * 10)
                                  else:
                                      self._dm_write_dq_sf(dm_t - 1, dm_sdm, dm_mbar, cta_rank_in_cluster)
                                      cute.arch.fence_view_async_tmem_store()
                                      dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                                      with cute.arch.elect_one():
                                          pipeline_dS.producer_commit(producer_state_dS)
                                      producer_state_dS.advance()
                      for stage in cutlass.range_constexpr(num_stages):
                          cute.copy(
                              thr_copy_r2t,
                              tSrP_r2t_f32[None, stage, None, None],
                              tStP_r2t[None, stage, None, None],
                          )

                  if const_expr(dm_tr is not None):
                      dm_now = dm_utils.clock32()
                      dm_s4 = dm_s4 + (dm_now - dm_tprev)
                      dm_tprev = dm_now
                      self._tlev(dm_tr, iter_idx, 10, dm_tid == 0)
                  # DM: dV SFA (per key row, per 32-query k-step) and constant SFB
                  dm_lane = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 32
                  for stage in cutlass.range_constexpr(num_stages):
                      dm_k = dm_qoff[stage]
                      dm_utils.tmem_st_x1((dm_lane << 16) + self.dm_sfa_dv_col + dm_k * 4 + dm_lane // 32,
                                          dm_utils.bytes4_same(dm_sfa_dv[stage]))
                  dm_one = Int32(0x7F7F7F7F)
                  dm_utils.tmem_st_x4((dm_lane << 16) + self.dm_sfb_dv_col, dm_one, dm_one, dm_one, dm_one)
                  if const_expr(not self.acc_m64):
                      cute.arch.fence_view_async_tmem_store()
                  cute.arch.fence_view_async_shared()
                  # The P (or S_P) and LSE mbarriers expect one arrival per compute warp
                  if const_expr(self.split_P_dS or "bar2" not in self.dm_opts):
                      # nobar2: per-warp arrivals suffice (consumer groups count compute warps; each warp fenced its
                      # own TMEM stores above), so no all-compute-warps barrier is needed before the S_P release
                      cute.arch.sync_warp()
                  else:
                      self.compute_sync_barrier.arrive_and_wait()
                  if const_expr(self.split_P_dS):
                      dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                      with cute.arch.elect_one():
                          pipeline_P.producer_commit(producer_state_P)
                      producer_state_P.advance()
                  elif const_expr(self.cta_schedule != CtaSchedule.TWO_CTA_SERIAL):
                      # Signal tmem store P completion with pipeline_S_P
                      dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                      with cute.arch.elect_one():
                          pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                          # pipeline_S_P.sync_object_empty.arrive(0, pipeline_S_P.consumer_mask)
                  # Normally we'd need syncwarp here since only 1 thread will signal in
                  # consumer_release, but we already have the barrier or sync_warp before this
                  pipeline_LSE.consumer_release(consumer_state_LSE)
                  consumer_state_LSE.advance()
                  # ---------------------------------------------
                  # dS.T = P.T * (dP.T - D)
                  # ---------------------------------------------
                  pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                  # DM dpsearly: delta (dPsum) smem loads issued before the dP wait (LDS latency hidden by the wait)
                  dm_dps_all = [None] * num_stages
                  if const_expr("dpsearly" in self.dm_opts):
                      dm_dps_all = [None] * num_stages
                      for stage in cutlass.range_constexpr(num_stages):
                          tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                          dm_dps_all[stage] = cute.make_fragment_like(tSsdPsum_cur, Float32)
                          cute.autovec_copy(tSsdPsum_cur, dm_dps_all[stage])
                  pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                  if const_expr(dm_tr is not None):
                      dm_now = dm_utils.clock32()
                      dm_s5 = dm_s5 + (dm_now - dm_tprev)
                      dm_tprev = dm_now
                      self._tlev(dm_tr, iter_idx, 11, dm_tid == 0)
                  # pipeline_dP.sync_object_full.wait(0, consumer_phase_S_P_dP)
                  ### Now delayed to after loop
                  # consumer_state_S_P_dP.advance()
                  # consumer_phase_S_P_dP ^= 1

                  ##### DM: dS.T = P.T * (dP.T * (vs*dos) - delta), all stages first (tile-wide dS scale)
                  # delta (dPsum) smem loads for all stages issued before the TMEM dP loads (latency overlap)
                  if const_expr("dpsearly" not in self.dm_opts):
                      dm_dps_all = [None] * num_stages
                      for stage in cutlass.range_constexpr(num_stages):
                          tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                          dm_dps_all[stage] = cute.make_fragment_like(tSsdPsum_cur, Float32)
                          cute.autovec_copy(tSsdPsum_cur, dm_dps_all[stage])
                  tdPrdP_all = [None] * num_stages
                  ds_amax = Float32(0.0)
                  for stage in cutlass.range_constexpr(num_stages):
                      tdPrdP_t2r = cute.make_rmem_tensor(tScS_t2r[None, 0, None, None].shape, Float32)
                      cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
                      tdPrdP_all[stage] = tdPrdP_t2r
                  cute.arch.fence_view_async_tmem_load()
                  dm_dsm = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]   # amaxt: 4 independent chains
                  for stage in cutlass.range_constexpr(num_stages):
                      tdPrdP_cur = tdPrdP_all[stage][None, 0, 0]
                      tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                      tSrdPsum = dm_dps_all[stage]
                      for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                          if const_expr("rsdbg" in self.dm_opts):
                              # DEBUG: exact per-query rowsum(dS) and sum |P (dP a)| of THIS kernel's P,
                              # dS replicated with the same scalar ops (FFMA2 == FFMA rounding), accumulated in f64
                              for e_ in cutlass.range_constexpr(2):
                                  i_ = 2 * v + e_
                                  dm_qg = m_block * self.tile_m + dm_idx0[None, stage, 0, 0][i_][1]
                                  dm_ds = tSrS_cur[i_] * cute.arch.fma_packed_f32x2(
                                      (tdPrdP_cur[i_], tdPrdP_cur[i_]), (dm_a[stage], dm_a[stage]),
                                      (-tSrdPsum[i_], -tSrdPsum[i_]))[0]
                                  dm_pd = tSrS_cur[i_] * (tdPrdP_cur[i_] * dm_a[stage])
                                  dm_rs = aux_data.tensors[4]
                                  dm_off = cute.crd2idx((batch_idx, head_idx, dm_qg, 0), dm_rs.layout)
                                  dm_ad = dm_rs.iterator.toint() + 8 * dm_off
                                  dm_utils.red_add_f64_from_f32(dm_ad, dm_ds)
                                  dm_utils.red_add_f64_from_f32(dm_ad + 8, dm_utils.absf(dm_pd))
                                  if const_expr("pdump" in self.dm_opts):
                                      dm_pb = aux_data.tensors[5]
                                      dm_kg = dm_idx0[None, stage, 0, 0][i_][0]
                                      if dm_kg < seqlen.seqlen_k:   # keys past N (empty half of a partial cluster)
                                          dm_pb[batch_idx, head_idx, dm_qg, dm_kg] = tSrS_cur[i_]
                                          aux_data.tensors[6][batch_idx, head_idx, dm_qg, dm_kg] = tdPrdP_cur[i_]
                          tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                              (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                              (dm_a[stage], dm_a[stage]),
                              (-tSrdPsum[2 * v], -tSrdPsum[2 * v + 1]),
                          )
                          tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                              (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                              (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                          )
                          if const_expr(self.dm_dsq and "apub2" in self.dm_opts):
                              # DM apub2: 4 independent 3-input max chains (short dependency chain), merged below
                              dm_dsm[v % 4] = utils.fmax(
                                  dm_dsm[v % 4], dm_utils.absf(tdPrdP_cur[2 * v]), dm_utils.absf(tdPrdP_cur[2 * v + 1]))
                          elif const_expr(self.dm_dsq and "apub" in self.dm_opts):
                              # DM apub: per-thread |dS| max of this warp's half tile (exact; published below)
                              ds_amax = cute.arch.fmax(
                                  ds_amax, cute.arch.fmax(utils_abs(tdPrdP_cur[2 * v]), utils_abs(tdPrdP_cur[2 * v + 1]))
                              )
                          elif const_expr(self.dm_dsq):
                              pass   # DM dsq: the reduce warps take the tile amax
                          elif const_expr("dsamax" not in self.dm_abl and "amaxt" in self.dm_opts):
                              dm_dsm[v % 4] = utils.fmax(
                                  dm_dsm[v % 4], dm_utils.absf(tdPrdP_cur[2 * v]), dm_utils.absf(tdPrdP_cur[2 * v + 1]))
                          elif const_expr("dsamax" not in self.dm_abl and "amax3" not in self.dm_opts):
                              ds_amax = cute.arch.fmax(
                                  ds_amax, cute.arch.fmax(utils_abs(tdPrdP_cur[2 * v]), utils_abs(tdPrdP_cur[2 * v + 1]))
                              )
                          elif const_expr("dsamax" not in self.dm_abl):
                              ds_amax = utils.fmax(
                                  ds_amax, dm_utils.absf(tdPrdP_cur[2 * v]), dm_utils.absf(tdPrdP_cur[2 * v + 1])
                              )
                  if const_expr("spf" in self.dm_opts and not self.split_P_dS):
                      # P(t) (in tSrS) is dead now: wait for S(t+1) (issued after P(t) was stored) and start its TMEM load
                      if iter_idx + 1 < loop_count:
                          pipeline_S_P.sync_object_full.wait(0, consumer_state_S_P_dP.phase ^ 1)
                          cute.copy(thr_copy_t2r, tStS_t2r, dm_Spf)
                  if const_expr(not self.dm_dsq):
                      if const_expr("dsamax" not in self.dm_abl and "amaxt" in self.dm_opts):
                          ds_amax = utils.fmax(utils.fmax(dm_dsm[0], dm_dsm[1]), dm_dsm[2], dm_dsm[3])
                      # scale per (32 keys = this warp's lanes) x (128-query tile = both warpgroups)
                      ds_amax = dm_utils.redux_max_abs_f32(ds_amax)
                      if const_expr(dm_tr is not None):
                          dm_now = dm_utils.clock32()
                          dm_s6 = dm_s6 + (dm_now - dm_tprev)
                          dm_tprev = dm_now
                      dm_cw = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.compute_warp_ids[0])
                      if cute.arch.lane_idx() == 0:
                          sDm[dm_cw] = ds_amax
                      # every warp has loaded dP (dS overwrites dP lanes of the other warpgroup) and published its max
                      if const_expr("fullbar" not in self.dm_opts and not self.dm_cwg4):
                          # dP/dS lanes and the dS scale block (32 keys x 128 queries) belong to warps w and w+4 only
                          cute.arch.barrier(barrier_id=8 + cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4,
                                            number_of_threads=64)
                      else:
                          self.compute_sync_barrier.arrive_and_wait()
                      if const_expr(dm_tr is not None):
                          dm_now = dm_utils.clock32()
                          dm_s7 = dm_s7 + (dm_now - dm_tprev)
                          dm_tprev = dm_now
                      for dm_j in cutlass.range_constexpr(1, len(self.compute_warp_ids) // 4):
                          ds_amax = cute.arch.fmax(ds_amax, sDm[(dm_cw + 4 * dm_j) % len(self.compute_warp_ids)])
                      ds_sb = dm_utils.pow2_sb(ds_amax)
                      ds_inv = dm_utils.inv_pow2_from_sb(ds_sb)
                      dm_sfa_dk = [dm_utils.add_sb_clamp(ds_sb, dm_qsb[st]) for st in range(num_stages)]
                      dm_sfb_dq = dm_utils.add_sb_clamp(ds_sb, dm_ksb)
                      if dm_cw < 4:
                          if cute.arch.lane_idx() == 0:
                              dm_utils.st_shared_b32(dm_sdm + 4 * (16 + dm_cw), dm_sfb_dq)
                      if const_expr("ilp2" in self.dm_opts and not self.acc_m64 and not self.dm_cwg4
                                    and self.use_2cta_instrs):
                          # ilp2: quantize every stage, store dS to TMEM (no sdS dependency), then wait for the sdS slot
                          # and write the smem copies
                          dm_cvt = [None] * num_stages
                          for stage in cutlass.range_constexpr(num_stages):
                              tdPrdP_cur = tdPrdP_all[stage][None, 0, 0]
                              tdPrdSq = cute.make_fragment_like(tdPrdP_cur, Float32)
                              for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                                  tdPrdSq[2 * v], tdPrdSq[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                      (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (ds_inv, ds_inv))
                              dm_cvt[stage] = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                              dm_cvt[stage].store(tdPrdSq.load().to(self.ds_dtype))
                              cute.copy(thr_copy_r2t, cute.recast_tensor(dm_cvt[stage], Float32), tdPtdS_r2t[None, stage, 0, 0])
                          pipeline_dS.producer_acquire(producer_state_dS)
                          tdPrdS_xchg = cute.make_fragment_like(dm_cvt[0], self.ds_dtype)
                          for stage in cutlass.range_constexpr(num_stages):
                              if exchange_stage == stage:
                                  cute.autovec_copy(dm_cvt[stage], tdPrdS_xchg)
                              else:
                                  cute.autovec_copy(dm_cvt[stage], tRS_sdS[None, stage])
                      for stage in cutlass.range_constexpr(num_stages if const_expr(not ("ilp2" in self.dm_opts
                              and not self.acc_m64 and not self.dm_cwg4 and self.use_2cta_instrs)) else 0):
                          tdPrdP_cur = tdPrdP_all[stage][None, 0, 0]
                          tdPrdSq = cute.make_fragment_like(tdPrdP_cur, Float32)
                          for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                              tdPrdSq[2 * v], tdPrdSq[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                  (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), (ds_inv, ds_inv))
                          tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                          tdPrdS_cvt.store(tdPrdSq.load().to(self.ds_dtype))
                          if const_expr(stage == 0):
                              if const_expr(dm_tr is not None):
                                  dm_now = dm_utils.clock32()
                                  dm_s8 = dm_s8 + (dm_now - dm_tprev)
                                  dm_tprev = dm_now
                              pipeline_dS.producer_acquire(producer_state_dS)
                              if const_expr(dm_tr is not None):
                                  dm_now = dm_utils.clock32()
                                  dm_s12 = dm_s12 + (dm_now - dm_tprev)
                                  dm_tprev = dm_now
                              if const_expr(self.use_2cta_instrs and not self.acc_m64):
                                  tdPrdS_xchg = cute.make_fragment_like(tdPrdS_cvt, self.ds_dtype)

                          if const_expr(self.acc_m64):
                              # dS^T for the dK MMA: this CTA's rows, both Q-column halves; the half
                              # the peer needs for its dQ MMA is exchanged straight from here.
                              cute.autovec_copy(tdPrdS_cvt, tRS_sdS_dK[None, 0, 0, 0])
                              # dS for this CTA's own dQ MMA: the Q-column half it owns.
                              if q_half == cta_rank_in_cluster:
                                  cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, 0, 0, 0])
                          else:
                              # RMEM->TMEM: always write to TMEM for MMA
                              if const_expr(not self.use_smem_dS_for_mma_dK or self.use_2cta_instrs):
                                  tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                                  cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])

                              # RMEM->SMEM: For 2-CTA, keep exchange stage in registers, write
                              # non-exchange to sdS
                              if const_expr(self.use_2cta_instrs):
                                  # cwg4: one 32-query stage per thread; its query half (k-step // 2) decides
                                  if exchange_stage == (dm_qoff[stage] // 2 if const_expr(self.dm_cwg4) else stage):
                                      cute.autovec_copy(tdPrdS_cvt, tdPrdS_xchg)
                                  else:
                                      cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
                              else:
                                  cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

                      if const_expr(self.acc_m64):
                          # dS^T is in SMEM: make this warp's stores visible to the async proxy before
                          # the per-warp arrival the MMA warp waits on.
                          cute.arch.fence_view_async_shared()
                      elif const_expr(not self.use_smem_dS_for_mma_dK):
                          # DM: dK SFA (per key row, per 32-query k-step) and constant SFB
                          for stage in cutlass.range_constexpr(num_stages):
                              dm_k = dm_qoff[stage]
                              dm_utils.tmem_st_x1((dm_lane << 16) + self.dm_sfa_dk_col + dm_k * 4 + dm_lane // 32,
                                                  dm_utils.bytes4_same(dm_sfa_dk[stage]))
                          dm_utils.tmem_st_x4((dm_lane << 16) + self.dm_sfb_dk_col, dm_one, dm_one, dm_one, dm_one)
                          cute.arch.fence_view_async_tmem_store()

                      if const_expr(self.use_2cta_instrs):
                          # use pipeline_dP to signal the dS store (TMEM, or SMEM with acc_m64)
                          dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                          with cute.arch.elect_one():
                              pipeline_dP.consumer_release(consumer_state_S_P_dP)
                      consumer_state_S_P_dP.advance()

                      # After the loop: copy exchange registers to sdS_xchg buffer
                      if const_expr(sdS_xchg is not None):
                          # sdS_xchg aliases sdQaccum: wait until the dQ reduce has read it
                          if const_expr(dQaccum_empty_mbar_ptr is not None):
                              cute.arch.mbarrier_wait(
                                  dQaccum_empty_mbar_ptr, phase=producer_state_dS.phase
                              )
                          if const_expr(self.dm_cwg4):
                              if dm_qoff[0] // 2 == exchange_stage:
                                  cute.autovec_copy(tdPrdS_xchg, tRS_sdS_xchg[None, 0])
                          else:
                              cute.autovec_copy(tdPrdS_xchg, tRS_sdS_xchg[None, 0])

                      cute.arch.fence_view_async_shared()
                      # The dS and dPsum mbarriers expect one arrival per compute warp
                      if const_expr(self.split_P_dS or "bar4" in self.dm_abl):
                          cute.arch.sync_warp()
                      else:
                          if const_expr(dm_tr is not None):
                              dm_now = dm_utils.clock32()
                              dm_s8 = dm_s8 + (dm_now - dm_tprev)
                              dm_tprev = dm_now
                          self.compute_sync_barrier.arrive_and_wait()
                      if const_expr(dm_tr is not None):
                          dm_now = dm_utils.clock32()
                          dm_s13 = dm_s13 + (dm_now - dm_tprev)
                          dm_tprev = dm_now
                      # DM: this CTA's 4 key-block dQ scale bytes -> one word, to self and to the peer CTA
                      if dm_tid == 0:
                          dm_slot = dm_t & 1
                          dm_word = dm_utils.pack_bytes4(
                              dm_utils.ld_shared_b32(dm_sdm + 4 * 16), dm_utils.ld_shared_b32(dm_sdm + 4 * 17),
                              dm_utils.ld_shared_b32(dm_sdm + 4 * 18), dm_utils.ld_shared_b32(dm_sdm + 4 * 19))
                          dm_utils.st_shared_b32(dm_sdm + 4 * (24 + dm_slot), dm_word)
                          dm_utils.st_async_peer_b32(dm_sdm + 4 * (32 + dm_slot), dm_mbar + 8 * dm_slot,
                                                     peer_cta_rank_in_cluster, dm_word)
                          dm_utils.mbar_arrive_expect_tx(dm_mbar + 8 * dm_slot, 4)
                      dm_t = dm_t + 1
                      # Normally we'd need syncwarp here since only 1 thread will signal in
                      # consumer_release, but we already have the barrier or sync_warp before this
                      pipeline_dPsum.consumer_release(consumer_state_dPsum)
                      if const_expr(dm_tr is not None):
                          dm_now = dm_utils.clock32()
                          dm_s9 = dm_s9 + (dm_now - dm_tprev)
                          dm_tprev = dm_now
                      consumer_state_dPsum.advance()
                      # pipelined 2CTA schedule: pipeline_dS also signals S tmem load completion so is deferred
                      if const_expr(self.cta_schedule != CtaSchedule.TWO_CTA_PIPELINED):
                          dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                          with cute.arch.elect_one():
                              pipeline_dS.producer_commit(producer_state_dS)
                          producer_state_dS.advance()

                      # 2-CTA: DSMEM copy of the exchanged dS half to the peer's sdS buffer
                      if const_expr(self.use_2cta_instrs):
                          stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
                          stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
                          if tidx == 0:
                              smem_src_ptr = sdS_xchg_src.iterator
                              # Destination is peer's sdS at our CTA's offset (exchange_stage position)
                              smem_dst_ptr = sdS.iterator + cta_rank_in_cluster * stage_copy_elems
                              cute.arch.mbarrier_arrive_and_expect_tx(
                                  dS_cluster_full_mbar_ptr,
                                  stage_copy_bytes,
                                  peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                              )
                              copy_utils.cpasync_bulk_s2cluster(
                                  smem_src_ptr,
                                  smem_dst_ptr,
                                  dS_cluster_full_mbar_ptr,
                                  stage_copy_bytes,
                                  peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                              )
                  else:
                      # DM dsq: exact fp32 dS (same packed FMA / multiply as above) stored in place over dP; the reduce warps do the
                      # tile amax, scale, e4m3 codes (TMEM + SMEM), dK/dQ SFs, the peer exchange, the dP release and the dS commit
                      if const_expr("apub2" in self.dm_opts):
                          ds_amax = utils.fmax(utils.fmax(dm_dsm[0], dm_dsm[1]), dm_dsm[2], dm_dsm[3])
                      if const_expr("apub" in self.dm_opts):
                          # DM apub: warp max over its 32 keys x 64 queries -> sDm[40 + compute warp]; the helper warp of this
                          # TMEM lane quadrant takes the max of the two halves instead of re-loading dS (same scale, exact max)
                          ds_amax = dm_utils.redux_max_abs_f32(ds_amax)
                          dm_cwp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.compute_warp_ids[0]
                          if cute.arch.lane_idx() == 0:
                              dm_utils.st_shared_b32(dm_sdm + 4 * (40 + dm_cwp), dm_utils.bitcast_f32_i32(ds_amax))
                      for stage in cutlass.range_constexpr(num_stages):
                          cute.copy(dm_thr_copy_r2t_f32, tdPrdP_all[stage], dm_tdPtdP_r2t_f32[None, stage, None, None])
                      cute.arch.fence_view_async_tmem_store()
                      dm_utils.tc_fence_before()
                      cute.arch.sync_warp()
                      if cute.arch.lane_idx() == 0:
                          dm_utils.mbar_arrive(dm_mbar + 8 * 9)
                      self._tlev(dm_tr, iter_idx, 12, dm_tid == 0)
                      self._tlev(dm_tr, iter_idx, 13, dm_tid == 128)
                      consumer_state_S_P_dP.advance()
                      pipeline_dPsum.consumer_release(consumer_state_dPsum)
                      consumer_state_dPsum.advance()

            # Final signal for dS smem store completion (DM dsq: the reduce warps commit the last tile)
            if const_expr(self.cta_schedule == CtaSchedule.TWO_CTA_PIPELINED and not self.dm_dsq):
                if process_tile:
                    self._dm_write_dq_sf(dm_t - 1, dm_sdm, dm_mbar, cta_rank_in_cluster)
                    cute.arch.fence_view_async_tmem_store()
                    dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

            if const_expr(dm_tr is not None and "tl" in self.dm_opts):
                # DM tl: per-CTA timeline (start/end globaltimer ns of the compute mainloop, SM id) at [64 + 4 * bid]
                if dm_tid == 0:
                    dm_bid = cute.arch.block_idx()[0]
                    dm_tb = dm_tr.iterator.toint() + 8 * (64 + (8 if const_expr("tl2" in self.dm_opts) else 4) * dm_bid)
                    dm_utils.red_add_u64(dm_tb, dm_tl0)
                    dm_utils.red_add_u64(dm_tb + 8, dm_utils.globaltimer())
                    dm_utils.red_add_u64(dm_tb + 16, Int64(dm_utils.smid()))
                    dm_utils.red_add_u64(dm_tb + 24, Int64(m_block_max - m_block_min))
            if const_expr(dm_tr is not None):
                for wg_ in range(2 if const_expr("noflush" not in self.dm_opts) else 0):
                    if dm_tid == 128 * wg_:
                        dm_vals = (dm_s0, dm_s1, dm_s2, dm_s3, dm_s4, dm_s5, dm_s6, dm_s7, dm_s8, dm_s9)
                        if wg_ == 0:
                            dm_utils.red_add_u64(dm_tr.iterator.toint() + 8 * 40, Int64(dm_s10))
                            dm_utils.red_add_u64(dm_tr.iterator.toint() + 8 * 41, Int64(dm_s11))
                            dm_utils.red_add_u64(dm_tr.iterator.toint() + 8 * 42, Int64(dm_s12))
                            dm_utils.red_add_u64(dm_tr.iterator.toint() + 8 * 43, Int64(dm_s13))
                        if const_expr("flushconst" in self.dm_opts):
                            dm_vals = tuple(Int64(1) for _ in range(10))
                        for k_, a_ in enumerate(dm_vals):
                            dm_utils.red_add_u64(dm_tr.iterator.toint() + 8 * (16 + 10 * wg_ + k_), Int64(a_))
                dm_s0, dm_s1, dm_s2, dm_s3, dm_s4 = Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
                dm_s5, dm_s6, dm_s7, dm_s8, dm_s9 = Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
                dm_s10, dm_s11, dm_s12, dm_s13 = Int32(0), Int32(0), Int32(0), Int32(0)
            if const_expr(self.dm_pers > 0 and self.dm_dkvdir):
                # DM pers + dkvdir: the epilogue no longer touches sK/sV (direct gmem stores), so the load warp may
                # bring in the next tile's K/V as soon as every compute warp is past the main loop
                self.compute_sync_barrier.arrive_and_wait()
                if dm_tid == 0:
                    cute.arch.mbarrier_arrive(dm_xchg_mbar + 2)
            # Epilogue
            # Run epilogue if we processed any m_blocks for this n_block
            if process_tile:
                if const_expr(not self.use_tma_store):
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        warp_idx,
                        batch_idx,
                        head_idx,
                        work_n,
                        kv_block,
                        seqlen,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                    )
                else:
                    thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                    #### STORE dV
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        kv_head,
                        kv_block,
                        seqlen,
                        thr_mma_dV,
                        tdVtdV,
                        mdV_tma_tensor,
                        sdV,
                        tma_atom_dV,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        None,  # Don't scale
                        self.dm_epi_barrier,  # barrier_id
                        mdV_semaphore,
                        "V",
                        dm_tsw=(dm_tr.iterator.toint() + 8 * 46 if const_expr(dm_tr is not None and "semw" in self.dm_opts) else None),
                        dm_tlp=(dm_tr.iterator.toint() + 8 * (64 + 8 * cute.arch.block_idx()[0] + 6)
                                if const_expr(dm_tr is not None and "tl3" in self.dm_opts) else None),
                    )
                    #### STORE dK
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        kv_head,
                        kv_block,
                        seqlen,
                        thr_mma_dK,
                        tdKtdK,
                        mdK_tma_tensor,
                        sdK,
                        tma_atom_dK,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        None,  # DM: Q codes carry the softmax scale
                        self.dm_epi_barrier,  # barrier_id
                        mdK_semaphore,
                        "K",
                        dm_tsw=(dm_tr.iterator.toint() + 8 * 46 if const_expr(dm_tr is not None and "semw" in self.dm_opts) else None),
                    )
            if const_expr(dm_tr is not None and "tl2" in self.dm_opts):
                if dm_tid == 0:
                    dm_utils.red_add_u64(dm_tr.iterator.toint() + 8 * (64 + 8 * cute.arch.block_idx()[0] + 5),
                                         dm_utils.globaltimer())
            if const_expr(self.dm_pers > 0 and not self.dm_dkvdir):
                # DM pers: every compute thread's bulk stores have finished READING sdK/sdV (= sK/sV), then one
                # arrival per tile lets the load warp bring in the next tile's K/V
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.compute_sync_barrier.arrive_and_wait()
                if dm_tid == 0:
                    cute.arch.mbarrier_arrive(dm_xchg_mbar + 2)
            # Zero dK/dV for empty tiles (local attention or block sparsity)
            # When total_m_block_cnt == 0 for block sparsity, no Q tiles contribute to this KV tile
            if const_expr(not self.dKV_postprocess):
                should_zero_dKV = False
                if const_expr(self.is_local or self.is_varlen_q):
                    should_zero_dKV = m_block_min >= m_block_max
                if const_expr(self.use_block_sparsity):
                    # For block sparsity, zero when no m_blocks contribute to this n_block
                    if not process_tile:
                        should_zero_dKV = True

                if should_zero_dKV:
                    # Both CTAs zero the whole work tile (the same rows twice).
                    gmem_tiled_copy_zero_dK = copy_utils.tiled_copy_2d(
                        self.dk_dtype,
                        math.gcd(64, self.tile_hdim),
                        128,  # num_threads
                    )
                    gmem_tiled_copy_zero_dV = copy_utils.tiled_copy_2d(
                        self.dv_dtype,
                        math.gcd(64, self.tile_hdimv),
                        128,  # num_threads
                    )
                    gmem_thr_copy_zero_dK = gmem_tiled_copy_zero_dK.get_slice(dp_idx)
                    gmem_thr_copy_zero_dV = gmem_tiled_copy_zero_dV.get_slice(dp_idx)
                    mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, kv_head]
                    mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, kv_head]
                    gdK = cute.local_tile(
                        mdK_cur, (self.cluster_tile_n, self.tile_hdim), (work_n, 0)
                    )
                    gdV = cute.local_tile(
                        mdV_cur, (self.cluster_tile_n, self.tile_hdimv), (work_n, 0)
                    )
                    tdKgdK = gmem_thr_copy_zero_dK.partition_D(gdK)
                    tdVgdV = gmem_thr_copy_zero_dV.partition_D(gdV)
                    cdK = cute.make_identity_tensor((self.cluster_tile_n, self.tile_hdim))
                    cdV = cute.make_identity_tensor((self.cluster_tile_n, self.tile_hdimv))
                    tdKcdK = gmem_thr_copy_zero_dK.partition_D(cdK)
                    tdVcdV = gmem_thr_copy_zero_dV.partition_D(cdV)
                    assert cute.size(tdKgdK[None, 0, 0]) == cute.size(tdVgdV[None, 0, 0])
                    zero = cute.make_fragment_like(tdKgdK[None, 0, 0])
                    zero.fill(0.0)
                    if tidx < 128:
                        for i in cutlass.range_constexpr(tdKgdK.shape[1]):
                            row_idx = tdKcdK[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - self.cluster_tile_n * work_n:
                                for j in cutlass.range_constexpr(tdKgdK.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dK, zero, tdKgdK[None, i, j])
                    else:
                        for i in cutlass.range_constexpr(tdVgdV.shape[1]):
                            row_idx = tdVcdV[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - self.cluster_tile_n * work_n:
                                for j in cutlass.range_constexpr(tdVgdV.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dV, zero, tdVgdV[None, i, j])

            self._dm_prog(Int32(3) + cute.arch.warp_idx() // 4 % 2, dm_pt, Int32(0), Int32(15))
            dm_pt = dm_pt + 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _dq_semaphore_lock_value(
        self,
        iter_idx: Int32,
        curr_q_cnt: Int32,
        curr_dq_write_order: Optional[cute.Tensor],
        curr_dq_write_order_full: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        seqlen,
        m_block: Int32,
        work_n: Int32,
    ) -> Int32:
        """Semaphore value the dQ reduce of (m_block, work_n) waits for: the work tiles of a Q
        tile add in work_n order (reversed under spt), one slot per CTA rank."""
        lock_value = work_n
        if const_expr(self.spt):
            n_block_max_for_m_block = block_info.get_n_block_max_for_m_block(seqlen, m_block)
            lock_value = n_block_max_for_m_block - 1 - work_n
        if const_expr(self.use_block_sparsity):
            assert blocksparse_tensors is not None
            if const_expr(blocksparse_tensors.dq_write_order is not None):
                sparse_iter = iter_idx // self.q_subtile_factor
                if sparse_iter < curr_q_cnt:
                    assert curr_dq_write_order is not None
                    lock_value = curr_dq_write_order[sparse_iter]
                else:
                    assert curr_dq_write_order_full is not None
                    lock_value = curr_dq_write_order_full[sparse_iter - curr_q_cnt]
                if const_expr(self.kv_subtile_factor > self.cta_group_size):
                    groups_per_sparse_block = self.kv_subtile_factor // self.cta_group_size
                    local_group = work_n % groups_per_sparse_block
                    if const_expr(self.spt):
                        # [NOTE] KV_subtile determ + spt
                        # dq_write_order stores one rank per sparse block; each physical tile
                        # derives its slot as
                        # rank * groups_per_sparse_block + (work_n % groups_per_sparse_block).
                        # W/ kv_subtile the tail sparse column can have a physical tile that is
                        # never scheduled; e.g. kv_tile = 128, seqlen = 1023, KV_Block = 384 -> 3
                        # sparse blocks. The last block is covered in [768, 896), [896, 1024) and
                        # then [1024, 1152) which no CTA ever runs. Since the highest tile gets
                        # the lowest lock value under spt, we would hang!
                        # In this case we locally reverse, [N+2, N+1, N*] where N* is not scheduled
                        # -> [N+1, N, N+2*]. Ahh but won't the vacant N+2 slot stall the next sparse
                        # block's CTAs? It will, so the writer holding N+1 bumps the increment by 2
                        # instead of 1 (i.e. 1 + #unscheduled) :)
                        total_groups = cute.ceil_div(seqlen.seqlen_k, self.cluster_tile_n)
                        groups_in_own_block = cutlass.min(
                            groups_per_sparse_block,
                            total_groups
                            - (work_n // groups_per_sparse_block) * groups_per_sparse_block,
                        )
                        local_group = groups_in_own_block - 1 - local_group
                    lock_value = lock_value * groups_per_sparse_block + local_group
        return lock_value

    @cute.jit
    def _dsq_setup(self, tidx: Int32, thr_mma_S, tdPtdP, sdS, sdS_xchg):
        """DM dsq: copy partitions of the two compute threads (tidx, tidx + 128) whose dS elements reduce thread tidx
        quantizes (same TMEM lanes = keys, the two 64-query halves), built exactly like compute_loop's."""
        num_wg = 2
        tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPtdS = cute.make_tensor(tdPtdP.iterator + (self.tmem_dS_offset - self.tmem_dP_offset), tdPtdS.layout)
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        st_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), Float32)
        half_elems = (self.tile_m // 2) * self.tile_n
        sdS_layout = cute.make_layout(
            ((8, self.tile_n // 8), (self.tile_m // 2, 2)),
            stride=((self.tile_m // 2, 8 * (self.tile_m // 2)), (1, half_elems)),
        )
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
        sdS_xchg_epi = cute.make_tensor(
            cute.recast_ptr(sdS_xchg.iterator, cute.make_swizzle(2, 4, 3)), sdS_layout
        )
        res = []
        for h in cutlass.range_constexpr(2):
            vt = tidx + 128 * h
            thr_t2r = copy_utils.make_tmem_copy(ld_atom, num_wg).get_slice(vt)
            thr_r2t = copy_utils.make_tmem_copy(st_atom, num_wg).get_slice(vt)
            copy_atom_r2s = sm100_utils_basic.get_smem_store_op(LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_t2r)
            thr_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_t2r).get_slice(vt)
            res.append((
                thr_t2r,
                thr_t2r.partition_S(tdPtdP),
                thr_t2r.partition_D(tScS),
                thr_r2t,
                thr_r2t.partition_D(tdPtdS),
                thr_r2s.partition_D(sdS_epi),
                thr_r2s.partition_D(sdS_xchg_epi),
            ))
        return res

    @cute.jit
    def _dsq_quant_tile(
        self, dsq, tidx: Int32, q: Int32, thr_mma_S, work_n: Int32, m_block: Int32, head_idx: Int32, kv_head: Int32,
        batch_idx: Int32, seqlen, aux_data, sdS, sdS_xchg_src, pipeline_dP, pipeline_dS, dsq_dP_state, dsq_dS_state,
        dS_cluster_full_mbar_ptr, dm_sdm: Int32, dm_mbar: Int32, cta_rank: Int32, dsq_j: Int32, dsq_sr: Int32, dsq_last,
        dm_tr=None,
    ):
        """DM dsq: quantize dS of one m_block (the reduce warps' share of compute_loop's dS phase, same values)."""
        lane_base = q * 32
        # fp32 dS of this tile stored by every compute warp
        dm_utils.mbar_wait_parity(dm_mbar + 8 * 9, dsq_j & 1)
        dm_utils.tc_fence_after()
        if const_expr(dm_tr is not None):
            self._tlev(dm_tr, dsq_j, 16, tidx == 0)
        # scales: query blocks of the 4 (half, stage) chunks, key block of this warp (hoisted per tile like compute)
        dm_idx = [dsq[h][0].partition_D(self._cluster_S_coords(thr_mma_S, work_n, 0)) for h in range(2)]
        dm_kv_blk = dm_idx[0][0][0] // 32
        dm_kv_blk = cutlass.min(dm_kv_blk, (seqlen.seqlen_k + 31) // 32 - 1)
        dm_ksb = dm_utils.pow2_exponent_byte(Float32(aux_data.tensors[1][batch_idx, kv_head, dm_kv_blk]))
        nst = cute.size(dsq[0][2], mode=[1])
        # pass 1: tile amax over this warp's 32 keys x all 128 queries (max is exact: same scale byte as the 2-warp exchange)
        ds_amax = Float32(0.0)
        if const_expr("apub" in self.dm_opts):
            # DM apub: the two compute warps of this lane quadrant published their half-tile maxima (ordered by mbar 9)
            ds_amax = cute.arch.fmax(dm_utils.bitcast_i32_f32(dm_utils.ld_shared_b32(dm_sdm + 4 * (40 + q))),
                                     dm_utils.bitcast_i32_f32(dm_utils.ld_shared_b32(dm_sdm + 4 * (44 + q))))
        for h in cutlass.range_constexpr(0 if const_expr("apub" in self.dm_opts) else 2):
            for st in cutlass.range_constexpr(nst):
                frag = cute.make_rmem_tensor(dsq[h][2][None, 0, None, None].shape, Float32)
                cute.copy(dsq[h][0], dsq[h][1][None, st, None, None], frag)
                cute.arch.fence_view_async_tmem_load()
                fv = frag[None, 0, 0]
                for v in cutlass.range_constexpr(cute.size(fv) // 2):
                    ds_amax = cute.arch.fmax(ds_amax, cute.arch.fmax(utils_abs(fv[2 * v]), utils_abs(fv[2 * v + 1])))
        if const_expr("apub" not in self.dm_opts):
            ds_amax = dm_utils.redux_max_abs_f32(ds_amax)
        ds_sb = dm_utils.pow2_sb(ds_amax)
        ds_inv = dm_utils.inv_pow2_from_sb(ds_sb)
        dm_sfb_dq = dm_utils.add_sb_clamp(ds_sb, dm_ksb)
        if cute.arch.lane_idx() == 0:
            dm_utils.st_shared_b32(dm_sdm + 4 * (16 + q), dm_sfb_dq)
        # pass 2: re-load, scale, e4m3 (codes kept in registers until every fp32 read of the dP region is done)
        codes = [[None] * nst, [None] * nst]
        for h in cutlass.range_constexpr(2):
            for st in cutlass.range_constexpr(nst):
                frag = cute.make_rmem_tensor(dsq[h][2][None, 0, None, None].shape, Float32)
                cute.copy(dsq[h][0], dsq[h][1][None, st, None, None], frag)
                cute.arch.fence_view_async_tmem_load()
                fv = frag[None, 0, 0]
                fq = cute.make_fragment_like(fv, Float32)
                for v in cutlass.range_constexpr(cute.size(fv) // 2):
                    fq[2 * v], fq[2 * v + 1] = cute.arch.mul_packed_f32x2((fv[2 * v], fv[2 * v + 1]), (ds_inv, ds_inv))
                c = cute.make_fragment_like(fv, self.ds_dtype)
                c.store(fq.load().to(self.ds_dtype))
                codes[h][st] = c
        if const_expr(dm_tr is not None):
            self._tlev(dm_tr, dsq_j, 17, tidx == 0)
        # dK operand: dS codes (TS A operand) + dK SFA / constant SFB, then release dP -> dS to the MMA warp
        for h in cutlass.range_constexpr(2):
            for st in cutlass.range_constexpr(nst):
                cute.copy(dsq[h][3], cute.recast_tensor(codes[h][st], Float32), dsq[h][4][None, st, 0, 0])
        for h in cutlass.range_constexpr(2):
            for st in cutlass.range_constexpr(nst):
                qoff = dm_idx[h][None, st, 0, 0][0][1] // 32
                qsb = dm_utils.pow2_exponent_byte(
                    Float32(aux_data.tensors[0][batch_idx, head_idx, m_block * (self.tile_m // 32) + qoff]))
                dm_utils.tmem_st_x1((lane_base << 16) + self.dm_sfa_dk_col + qoff * 4 + q,
                                    dm_utils.bytes4_same(dm_utils.add_sb_clamp(ds_sb, qsb)))
        dm_one = Int32(0x7F7F7F7F)
        dm_utils.tmem_st_x4((lane_base << 16) + self.dm_sfb_dk_col, dm_one, dm_one, dm_one, dm_one)
        cute.arch.fence_view_async_tmem_store()
        dm_utils.tc_fence_before()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dP.consumer_release(dsq_dP_state)
        if const_expr(dm_tr is not None):
            self._tlev(dm_tr, dsq_j, 18, tidx == 0)
        # dQ operand: dS codes in SMEM (own query half) + the peer's half in the exchange buffer
        pipeline_dS.producer_acquire(dsq_dS_state)
        if const_expr(dm_tr is not None):
            self._tlev(dm_tr, dsq_j, 19, tidx == 0)
        exchange_stage = cta_rank ^ 1
        for h in cutlass.range_constexpr(2):
            for st in cutlass.range_constexpr(nst):
                if exchange_stage == st:
                    cute.autovec_copy(codes[h][st], dsq[h][6][None, 0])
                else:
                    cute.autovec_copy(codes[h][st], dsq[h][5][None, st])
        cute.arch.fence_view_async_shared()
        if const_expr(self.dm_dsqw):
            self.dsq_sync_barrier.arrive_and_wait()
        else:
            self.reduce_sync_barrier.arrive_and_wait()
        peer = cta_rank ^ 1
        if tidx == 0:
            slot = dsq_j & 1
            word = dm_utils.pack_bytes4(
                dm_utils.ld_shared_b32(dm_sdm + 4 * 16), dm_utils.ld_shared_b32(dm_sdm + 4 * 17),
                dm_utils.ld_shared_b32(dm_sdm + 4 * 18), dm_utils.ld_shared_b32(dm_sdm + 4 * 19))
            dm_utils.st_shared_b32(dm_sdm + 4 * (24 + slot), word)
            dm_utils.st_async_peer_b32(dm_sdm + 4 * (32 + slot), dm_mbar + 8 * slot, peer, word)
            dm_utils.mbar_arrive_expect_tx(dm_mbar + 8 * slot, 4)
            stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
            stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
            cute.arch.mbarrier_arrive_and_expect_tx(
                dS_cluster_full_mbar_ptr, stage_copy_bytes, peer_cta_rank_in_cluster=peer)
            copy_utils.cpasync_bulk_s2cluster(
                sdS_xchg_src.iterator, sdS.iterator + cta_rank * stage_copy_elems, dS_cluster_full_mbar_ptr,
                stage_copy_bytes, peer_cta_rank_in_cluster=peer)
        if const_expr(dm_tr is not None):
            self._tlev(dm_tr, dsq_j, 20, tidx == 0)
        # pipeline_dS(t): dQ(t) overwrites the S tail -> every compute warp must have loaded S(t+1) first
        if not dsq_last:
            dm_utils.mbar_wait_parity(dm_mbar + 8 * 10, dsq_sr & 1)
            dm_utils.tc_fence_after()
        if const_expr(dm_tr is not None):
            self._tlev(dm_tr, dsq_j, 21, tidx == 0)
        self._dm_write_dq_sf(dsq_j, dm_sdm, dm_mbar, cta_rank)
        cute.arch.fence_view_async_tmem_store()
        dm_utils.tc_fence_before()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dS.producer_commit(dsq_dS_state)
        if const_expr(dm_tr is not None):
            self._tlev(dm_tr, dsq_j, 22, tidx == 0)

    @cute.jit
    def _tlev(self, dm_tr, j: Int32, e: int, pred):
        """DM tlev: record clock64 of event e of tile j (j < 64) for CTA self.dm_tlev (one thread: pred and lane 0)."""
        if const_expr(dm_tr is not None and self.dm_tlev >= 0):
            if pred and cute.arch.lane_idx() == 0 and cute.arch.block_idx()[0] == self.dm_tlev and cute.arch.block_idx()[1] == 0 and cute.arch.block_idx()[2] == 0 and j < 64:
                dm_utils.red_add_u64(dm_tr.iterator.toint() + 8 * (40000 + 32 * j + e), dm_utils.clock64())

    def _dm_setmaxreg_dec(self, n: int):
        if not self.dm_noreg:   # trace-time python
            cute.arch.setmaxregister_decrease(n)

    def _dm_setmaxreg_inc(self, n: int):
        if not self.dm_noreg:
            cute.arch.setmaxregister_increase(n)

    @cute.jit
    def dsq_loop(
        self, thr_mma_S, tdPtdP, sdS, sdS_xchg, sdS_dK, pipeline_dP, pipeline_dS, dS_cluster_full_mbar_ptr, sDm,
        dm_xchg_mbar, aux_data, block_info, SeqlenInfoCls, TileSchedulerCls, dm_tr=None,
    ):
        """DM dsqw: the dS-quantization warpgroup walks the same (work tile, head, m_block) stream as the compute warps."""
        tidx = cute.arch.thread_idx()[0] % (4 * cute.arch.WARP_SIZE)
        q = cute.arch.make_warp_uniform(cute.arch.warp_idx() % 4)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        dsq = self._dsq_setup(tidx, thr_mma_S, tdPtdP, sdS, sdS_xchg)
        dsq_xchg_src = self._dS_exchange_source(sdS_xchg, sdS_dK, cta_rank_in_cluster ^ 1)
        dsq_dP_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        dsq_dS_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        dsq_sdm = sDm.iterator.toint()
        dsq_mbar = dm_xchg_mbar.toint()
        dsq_j = Int32(0)
        dsq_sr = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            (
                work_n,
                kv_block,
                head_idx,
                kv_head,
                batch_idx,
                seqlen,
                m_block_min,
                m_block_max,
                process_tile,
            ) = self._work_tile(work_tile, cta_rank_in_cluster, SeqlenInfoCls, block_info)
            dm_head0 = head_idx
            loop_count = m_block_max - m_block_min
            for dm_g in cutlass.range(self.dm_G, unroll=1):
                head_idx = dm_head0 + dm_g
                for iter_idx in cutlass.range(loop_count, unroll=1):
                    m_block = m_block_min + iter_idx
                    dsq_last = iter_idx == loop_count - 1
                    if const_expr(self.dm_gloop):
                        dsq_last = dsq_last and dm_g == self.dm_G - 1
                    self._dsq_quant_tile(
                        dsq, tidx, q, thr_mma_S, work_n, m_block, head_idx, kv_head, batch_idx, seqlen, aux_data, sdS,
                        dsq_xchg_src, pipeline_dP, pipeline_dS, dsq_dP_state, dsq_dS_state, dS_cluster_full_mbar_ptr,
                        dsq_sdm, dsq_mbar, cta_rank_in_cluster, dsq_j, dsq_sr, dsq_last, dm_tr=dm_tr)
                    dsq_dP_state.advance()
                    dsq_dS_state.advance()
                    if not dsq_last:
                        dsq_sr = dsq_sr + 1
                    dsq_j = dsq_j + 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _tcq_empty(self, dm_tcq, mdQ_semaphore, TileSchedulerCls, SeqlenInfoCls, block_info):
        """rw8k tconv readiness queue, fed by the otherwise idle EMPTY warp (a scheduling hint for the PDL tail converter;
        correctness rests on the converter's own ld.acquire of the det counters). For every dQ tile m this CTA is the LAST
        writer of in the det order (lock value == writers - 1; under spt: work_n == 0), the leader CTA's empty warp waits
        (ld.acquire.gpu, nanosleep backoff) until both 2-CTA slots of the tile's det counter are final, then appends it:
        pos = atom.add(tail, 1); entry[pos] = tile + 1. Tiles are appended in the order they become final. The compute /
        reduce / MMA warps run no extra code. Layout (int32, zeroed per call on the prestage side stream): [0] tail,
        [1, 1 + T) entries."""
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if cute.arch.lane_idx() == 0 and cta_rank_in_cluster == 0:
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                (work_n, kv_block, head_idx, kv_head, batch_idx, seqlen, m_block_min, m_block_max,
                 process_tile) = self._work_tile(work_tile, cta_rank_in_cluster, SeqlenInfoCls, block_info)
                mt = cute.size(mdQ_semaphore, mode=[0])     # (Mt, rank, H, B) view (semaphore_transpose)
                nh = cute.size(mdQ_semaphore, mode=[2])
                ns = cute.size(mdQ_semaphore, mode=[1])
                base = dm_tcq.iterator.toint()
                if process_tile:
                    # DM gloop: one CTA serves the dm_G query heads of its KV group (same order as the reduce warps)
                    for dm_g in cutlass.range(self.dm_G, unroll=1):
                        hq = head_idx + dm_g
                        for m in cutlass.range(m_block_min, m_block_max, unroll=1):
                            nmax = block_info.get_n_block_max_for_m_block(seqlen, m)
                            lock = self._dq_semaphore_lock_value(m - m_block_min, Int32(0), None, None, None, block_info,
                                                                 seqlen, m, work_n)
                            if lock == nmax - 1:
                                srow = mdQ_semaphore.iterator.toint() + Int64(4) * Int64(((batch_idx * nh + hq) * mt + m) * ns)
                                for r in cutlass.range_constexpr(2):
                                    dm_utils.poll_acquire_ge_or_trap(srow + Int64(4 * r), nmax, Int32(1 << 26))
                                pos = dm_utils.atom_add_relaxed_gpu_s32(base, Int32(1))
                                dm_utils.st_relaxed_gpu_s32(base + Int64(4) * Int64(1 + pos), (batch_idx * nh + hq) * mt + m + 1)
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mdQ_semaphore: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        dm_dq: Optional[cute.Tensor] = None,
        thr_mma_S=None,
        tdPtdP=None,
        sdS=None,
        sdS_xchg=None,
        sdS_dK=None,
        pipeline_dP=None,
        pipeline_dS=None,
        dS_cluster_full_mbar_ptr=None,
        sDm=None,
        dm_xchg_mbar=None,
        aux_data=None,
        dm_trw=None,
        dm_tlv=None,
        dm_tcq=None,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        # For 2-CTA: reduce_stage = dQaccum_reduce_stage_t2r / cta_group_size
        expected_reduce_stages_t2r = self.dQaccum_reduce_stage_t2r // self.cta_group_size
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == expected_reduce_stages_t2r, (
            "dQaccum t2r reduce stage mismatch"
        )
        expected_reduce_stages = self.dQaccum_reduce_stage // self.cta_group_size
        # 2-CTA: CTA 0 -> (M/2, D) (stage 0, 1) & CTA 1 -> (M/2, D) (stage 2, 3)
        stage_offset = (
            expected_reduce_stages * cta_rank_in_cluster if const_expr(self.use_2cta_instrs) else 0
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        # DM abl:nofull (timing only, breaks the det ordering guarantee): never wait for bulk-add COMPLETION, only smem reads
        read_flag = const_expr(not self.deterministic or "nofull" in self.dm_abl)

        if const_expr(self.dm_dsq and not self.dm_dsqw):
            dsq = self._dsq_setup(tidx, thr_mma_S, tdPtdP, sdS, sdS_xchg)
            dsq_xchg_src = self._dS_exchange_source(sdS_xchg, sdS_dK, cta_rank_in_cluster ^ 1)
            dsq_dP_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
            dsq_dS_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
            dsq_sdm = sDm.iterator.toint()
            dsq_mbar = dm_xchg_mbar.toint()
            dsq_j = Int32(0)    # tiles quantized (fp32-dS barrier parity, dQ SF word slot)
            dsq_sr = Int32(0)   # S-read barrier completions consumed
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sdQaccum_stage
        )
        dm_pt = Int32(0)
        while work_tile.is_valid_tile:
            self._dm_prog(Int32(5), dm_pt, Int32(0), Int32(1))
            (
                work_n,
                kv_block,
                head_idx,
                dsq_kv_head,
                batch_idx,
                seqlen,
                m_block_min,
                m_block_max,
                process_tile,
            ) = self._work_tile(work_tile, cta_rank_in_cluster, SeqlenInfoCls, block_info)
            # DM gloop: the reduce warps drain dQ of the qh query heads of the group in stream order
            dm_head0 = head_idx
            for dm_g in cutlass.range(self.dm_G, unroll=1):
                head_idx = dm_head0 + dm_g
                n_block_sparse = kv_block // self.kv_subtile_factor
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
                else:
                    # dQaccum rows of this sequence (pack_gqa: per KV head, pack_factor per token)
                    row_offset = sequence_row_offset(
                        seqlen.offset_q, batch_idx, self.pack_factor, self.tile_m
                    )
                    mdQaccum_cur = cute.domain_offset(
                        (row_offset * self.tile_hdim,), mdQaccum[None, head_idx]
                    )
                gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
                # (M * K / STAGE, STAGE, _)
                gdQaccum = cute.flat_divide(
                    gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,)
                )

                if const_expr(self.deterministic):
                    assert mdQ_semaphore is not None
                    mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

                dq_sem_release_inc = Int32(1)
                if const_expr(
                    self.deterministic
                    and self.use_block_sparsity
                    and self.spt
                    and self.kv_subtile_factor > self.cta_group_size
                ):
                    # A truncated tail block's last writer releases the missing increments,
                    # see: [NOTE] KV_subtile determ + spt
                    groups_per_sparse_block = self.kv_subtile_factor // self.cta_group_size
                    total_groups = cute.ceil_div(seqlen.seqlen_k, self.cluster_tile_n)
                    tail_sparse_block_idx = (total_groups - 1) // groups_per_sparse_block
                    groups_in_tail = total_groups - tail_sparse_block_idx * groups_per_sparse_block
                    is_tail_bridge_group = (
                        work_n // groups_per_sparse_block == tail_sparse_block_idx
                        and work_n % groups_per_sparse_block == 0
                    )
                    if is_tail_bridge_group:
                        dq_sem_release_inc = Int32(1) + groups_per_sparse_block - groups_in_tail

                curr_q_cnt = Int32(0)
                curr_q_idx = None
                curr_full_cnt = Int32(0)
                curr_full_idx = None
                curr_dq_write_order = None
                curr_dq_write_order_full = None
                loop_count = m_block_max - m_block_min
                if const_expr(self.use_block_sparsity):
                    assert blocksparse_tensors is not None
                    (
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        loop_count,
                    ) = get_block_sparse_iteration_info_bwd(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block_sparse,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                    )
                    process_tile = loop_count > Int32(0)
                if const_expr(self.deterministic and self.use_block_sparsity):
                    assert blocksparse_tensors is not None
                    if const_expr(blocksparse_tensors.dq_write_order is not None):
                        assert blocksparse_tensors.dq_write_order is not None
                        curr_dq_write_order = blocksparse_tensors.dq_write_order[
                            batch_idx, head_idx, n_block_sparse, None
                        ]
                        if const_expr(blocksparse_tensors.dq_write_order_full is not None):
                            assert blocksparse_tensors.dq_write_order_full is not None
                            curr_dq_write_order_full = blocksparse_tensors.dq_write_order_full[
                                batch_idx, head_idx, n_block_sparse, None
                            ]

                # Overlapped release: m_block - 1 is released while m_block's adds are issued, unless
                # the tile order is block sparse or sdQaccum aliases the dS exchange buffer (M=128
                # serial), which is handed back to the peer at every m_block.
                delay_semaphore_release = not self.use_block_sparsity and not self.xchg_aliases_dQaccum
                if const_expr(self.deterministic and delay_semaphore_release):
                    assert self.sdQaccum_stage <= 2, (
                        "Delayed dQ release requires at most two staging buffers"
                    )
                # release_now: each m_block is released right after its own adds complete instead.
                # Short tiles (few m_blocks: short varlen sequences, local windows) do so too: the
                # sibling n_blocks of a chain run in lockstep there and the delayed hop stalls them,
                # up to a 2-4x deterministic slowdown of local attention at long sequences.
                release_now = cutlass.Boolean(self.deterministic and not delay_semaphore_release)
                if const_expr(self.deterministic and delay_semaphore_release):
                    release_now = loop_count < self.delay_release_min_m_blocks

                # dQacc_reduce mainloop
                # Block sparsity: iterate over sparse m_block count and derive actual m_block
                # from Q_IDX/FULL_Q_IDX tensors. Dense: iterate m_block_min..m_block_max directly.
                for iter_idx in cutlass.range(loop_count, unroll=1):
                    m_block = m_block_min + iter_idx
                    m_block_oob_upper = False
                    if const_expr(self.use_block_sparsity):
                        m_block, _ = get_m_block_from_iter_bwd(
                            iter_idx,
                            curr_q_cnt,
                            curr_q_idx,
                            curr_full_cnt,
                            curr_full_idx,
                            q_subtile_factor=self.q_subtile_factor,
                            m_block_max=m_block_max,
                        )
                        m_block_oob_upper = m_block >= m_block_max
                    # semaphore acquire, before waiting for the MMA's dQ tile. On main the acquire
                    # followed consumer_release, so the MMA warp could start the next dQ MMA during
                    # the poll; now the poll overlaps the current dQ MMA instead, which measured
                    # faster. This m_block's adds are only issued below, after the acquire.
                    if const_expr(self.dm_dsq and not self.dm_dsqw):
                        # DM dsq: quantize this m_block's dS before waiting for its dQ (dQ(t) needs these codes)
                        dsq_last = iter_idx == loop_count - 1
                        if const_expr(self.dm_gloop):
                            dsq_last = dsq_last and dm_g == self.dm_G - 1
                        self._dsq_quant_tile(
                            dsq, tidx, warp_idx, thr_mma_S, work_n, m_block, head_idx, dsq_kv_head, batch_idx, seqlen,
                            aux_data, sdS, dsq_xchg_src, pipeline_dP, pipeline_dS, dsq_dP_state, dsq_dS_state,
                            dS_cluster_full_mbar_ptr, dsq_sdm, dsq_mbar, cta_rank_in_cluster, dsq_j, dsq_sr, dsq_last)
                        dsq_dP_state.advance()
                        dsq_dS_state.advance()
                        if not dsq_last:
                            dsq_sr = dsq_sr + 1
                        dsq_j = dsq_j + 1
                    dm_first = cutlass.Boolean(False)
                    dm_last = cutlass.Boolean(False)
                    lock_value = Int32(0)
                    if const_expr(self.deterministic):
                        if not m_block_oob_upper:
                            lock_value = self._dq_semaphore_lock_value(
                                iter_idx,
                                curr_q_cnt,
                                curr_dq_write_order,
                                curr_dq_write_order_full,
                                blocksparse_tensors,
                                block_info,
                                seqlen,
                                m_block,
                                work_n,
                            )
                            if const_expr("dqearly" not in self.dm_opts and "dqnone" not in self.dm_abl and "nodqw" not in self.dm_abl):
                                if const_expr(dm_trw is not None):
                                    dm_w0 = dm_utils.globaltimer()
                                if const_expr(dm_tlv is not None):
                                    self._tlev(dm_tlv, iter_idx, 24, tidx == 0)
                                barrier.wait_eq(
                                    mdQ_semaphore_cur[(m_block, None)].iterator,
                                    tidx,
                                    cta_rank_in_cluster,
                                    lock_value,
                                )
                                if const_expr(dm_tlv is not None):
                                    self._tlev(dm_tlv, iter_idx, 25, tidx == 0)
                                if const_expr(dm_trw is not None):
                                    # DM semw (instrumentation): dQ semaphore wait ns / count -> trace header [44], [45]
                                    if tidx == 0:
                                        dm_utils.red_add_u64(dm_trw, dm_utils.globaltimer() - dm_w0)
                                        dm_utils.red_add_u64(dm_trw + 8, Int64(1))
                            if const_expr(self.dm_fws):
                                # DM fws: the first writer in the deterministic order stores instead of adding, so
                                # dq_accum needs no zero-fill (dense causal / non-causal: lock 0 always exists per m_block)
                                dm_first = lock_value == 0
                            if const_expr(dm_dq is not None):
                                # DM lwc: the last writer in the deterministic order converts (accum + own partial) * sm
                                # to bf16 dQ itself (no postprocess kernel)
                                dm_last = lock_value == block_info.get_n_block_max_for_m_block(seqlen, m_block) - 1
                    pipeline_dQ.consumer_wait(dQ_consumer_state)
                    if const_expr(dm_tlv is not None):
                        self._tlev(dm_tlv, iter_idx, 27, tidx == 0)
                    # TMEM -> RMEM
                    tdQrdQ_t2r = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
                    if const_expr(not self.dm_dqstream):
                        cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.sync_warp()
                        dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                        with cute.arch.elect_one():
                            pipeline_dQ.consumer_release(dQ_consumer_state)
                        dQ_consumer_state.advance()
                        if const_expr(dm_tlv is not None):
                            self._tlev(dm_tlv, iter_idx, 26, tidx == 0)
                    if const_expr(self.deterministic and "dqearly" in self.dm_opts and "nodqw" not in self.dm_abl):
                        # DM dqearly: the dQ tile is already in registers and its TMEM (= the S columns) handed back to
                        # the MMA warp; only now poll the deterministic order (same adds, same order: bitwise)
                        if not m_block_oob_upper:
                            barrier.wait_eq(
                                mdQ_semaphore_cur[(m_block, None)].iterator,
                                tidx,
                                cta_rank_in_cluster,
                                lock_value,
                            )

                    if m_block_max > 0:
                        m_block = cutlass.min(m_block, m_block_max - 1)
                    gdQaccum_cur = gdQaccum[None, None, m_block]

                    tdQrdQ_shape = (
                        self.dQ_reduce_ncol,
                        self.tile_hdim // self.cta_group_size // self.dQ_reduce_ncol,
                    )
                    tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)

                    if const_expr("dqred" in self.dm_opts):
                        # DM dqred (REJECTED: +11% slower, not bitwise on non-causal N4096): no SMEM staging / bulk adds: each thread adds its registers straight to dq_accum with
                        # red.global.add.v4.f32 (first writer: st.global.v4), then fences before the reduce barrier; the
                        # (delayed) semaphore release below follows that barrier. Same per-element adds, same CTA order.
                        if not m_block_oob_upper:
                            for dm_st in cutlass.range_constexpr(cute.size(tdQrdQ, mode=[1])):
                                for dm_k in cutlass.range_constexpr(self.dQ_reduce_ncol // 4):
                                    dm_ptr = gdQaccum_cur[None, dm_st + stage_offset].iterator.toint() + Int64(4 * (4 * tidx + 512 * dm_k))
                                    dm_a = tdQrdQ[4 * dm_k + 0, dm_st]
                                    dm_b = tdQrdQ[4 * dm_k + 1, dm_st]
                                    dm_c = tdQrdQ[4 * dm_k + 2, dm_st]
                                    dm_d = tdQrdQ[4 * dm_k + 3, dm_st]
                                    if dm_first:
                                        dm_utils.st_v4_f32(dm_ptr, dm_a, dm_b, dm_c, dm_d)
                                    else:
                                        dm_utils.red_add_v4_f32(dm_ptr, dm_a, dm_b, dm_c, dm_d)
                        dm_utils.fence_acq_rel_gpu()
                        self.reduce_sync_barrier.arrive_and_wait()
                        if const_expr(self.deterministic and delay_semaphore_release):
                            if not release_now and m_block > m_block_min:
                                if const_expr(self.dm_tcfence):
                                    dm_utils.fence_proxy_async_global()
                                barrier.arrive_inc_async(
                                    mdQ_semaphore_cur[(m_block - 1, None)].iterator,
                                    tidx,
                                    cta_rank_in_cluster,
                                    1,
                                )
                    for stage in cutlass.range_constexpr(cute.size(tdQrdQ, mode=[1]) if const_expr("dqnone" not in self.dm_abl and "dqred" not in self.dm_opts) else 0):
                        if const_expr(self.dm_dqstream and (stage * self.dQ_reduce_ncol) % self.dQ_reduce_ncol_t2r == 0):
                            # DM dqstream: TMEM -> RMEM one 32-column t2r chunk at a time (only 32 dQ floats live); the
                            # dQ TMEM slot is released after the last chunk is in registers. Same values, same adds.
                            dm_ts = stage * self.dQ_reduce_ncol // self.dQ_reduce_ncol_t2r
                            dm_rk = cute.rank(tdQtdQ_t2r)
                            cute.copy(thr_copy_t2r, tdQtdQ_t2r[(None, dm_ts) + (None,) * (dm_rk - 2)],
                                      tdQrdQ_t2r[(None, dm_ts) + (None,) * (dm_rk - 2)])
                            cute.arch.fence_view_async_tmem_load()
                            if const_expr(dm_ts == cute.size(tdQrdQ_t2r_shape, mode=[1]) - 1):
                                cute.arch.sync_warp()
                                dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
                                with cute.arch.elect_one():
                                    pipeline_dQ.consumer_release(dQ_consumer_state)
                                dQ_consumer_state.advance()
                        smem_idx = dQ_tma_store_producer_state.index
                        tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                        tdQrdQ_r2s = cute.make_tensor(tdQrdQ[None, stage].iterator, tdQsdQ_r2s.shape)
                        cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_view_async_shared()
                        self.reduce_sync_barrier.arrive_and_wait()
                        # Copy from shared memory to global memory
                        if const_expr(dm_dq is not None):
                            if dm_last and not m_block_oob_upper:
                                dm_row = m_block * self.tile_m + tidx % 64 + 64 * cta_rank_in_cluster
                                dm_col0 = stage * self.dQ_reduce_ncol + (self.tile_hdim // 2) * (tidx // 64)
                                if dm_row < seqlen.seqlen_q:
                                    for dm_k in cutlass.range_constexpr(self.dQ_reduce_ncol // 4):
                                        for dm_i in cutlass.range_constexpr(4):
                                            dm_j = 4 * dm_k + dm_i
                                            dm_v = tdQrdQ[dm_j, stage]
                                            if not dm_first:
                                                dm_v = gdQaccum_cur[4 * tidx + dm_i + 512 * dm_k, stage + stage_offset] + dm_v
                                            dm_dq[batch_idx, dm_row, head_idx, dm_col0 + dm_j] = dm_dq.element_type(
                                                dm_v * Float32(self.tile_hdim ** -0.5))
                        if is_tma_warp and not m_block_oob_upper and not dm_last and const_expr("dqtma" not in self.dm_abl):
                            with cute.arch.elect_one():
                                if dm_first:
                                    dm_utils.cpasync_bulk_store_s2g(
                                        sdQaccum[None, smem_idx].iterator,
                                        gdQaccum_cur[None, stage + stage_offset].iterator,
                                        self.tma_copy_bytes["dQ"] // 1,
                                    )
                                else:
                                    copy_utils.cpasync_reduce_bulk_add_f32(
                                        sdQaccum[None, smem_idx].iterator,
                                        gdQaccum_cur[None, stage + stage_offset].iterator,
                                        self.tma_copy_bytes["dQ"] // 1,
                                    )
                            cute.arch.cp_async_bulk_commit_group()
                            if const_expr("nodqrd" not in self.dm_opts and self.deterministic and self.sdQaccum_stage > 1):   # dqrd default (bitwise)
                                # DM dqrd: staging-buffer reuse only needs the older bulk adds to have READ smem;
                                # completion (visibility in global) is needed only before the semaphore release of
                                # the previous m_block (below, stage 0) and at tile end. Same adds, same order: bitwise.
                                cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=True)
                                if const_expr(stage == 0 and delay_semaphore_release and "nofull" not in self.dm_abl):
                                    cute.arch.cp_async_bulk_wait_group(1, read=False)
                            else:
                                cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=read_flag)
                        elif is_tma_warp:
                            # Drain pending TMA stores so SMEM buffers are safe to reuse
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                        self.reduce_sync_barrier.arrive_and_wait()
                        dQ_tma_store_producer_state.advance()

                        if const_expr(self.deterministic and stage == 0 and delay_semaphore_release):
                            if not release_now and m_block > m_block_min:
                                if const_expr(self.dm_tcfence):
                                    dm_utils.fence_proxy_async_global()
                                barrier.arrive_inc_async(
                                    mdQ_semaphore_cur[(m_block - 1, None)].iterator,
                                    tidx,
                                    cta_rank_in_cluster,
                                    1,
                                )

                    # With more than one staging stage the tile's newest bulk adds are still in
                    # flight here (each stage only waits for the older groups). Drain them once
                    # when something below needs them complete: the deterministic release of this
                    # m_block (the next CTA in order must find the adds performed, which the
                    # deterministic read_flag = False waits for) and the exchange-buffer alias
                    # handshake (the peer's dS exchange reuses sdQaccum once its reads are done).
                    if const_expr(
                        self.sdQaccum_stage > 1
                        and (self.deterministic or dQaccum_empty_mbar_ptr is not None)
                    ):
                        if const_expr(dQaccum_empty_mbar_ptr is not None) or release_now:
                            if is_tma_warp and not m_block_oob_upper:
                                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                            self.reduce_sync_barrier.arrive_and_wait()
                    if const_expr(dQaccum_empty_mbar_ptr is not None):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(dQaccum_empty_mbar_ptr)

                    # semaphore release: red.async.release.gpu, every release of an m_block's adds
                    # in this role follows their wait_group + barrier, see barrier.arrive_inc_async.
                    if const_expr(self.deterministic):
                        if release_now and not m_block_oob_upper:
                            if const_expr(self.dm_tcfence):
                                dm_utils.fence_proxy_async_global()
                            barrier.arrive_inc_async(
                                mdQ_semaphore_cur[m_block, None].iterator,
                                tidx,
                                cta_rank_in_cluster,
                                dq_sem_release_inc,
                            )

                if process_tile:
                    if is_tma_warp:
                        cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                    self.reduce_sync_barrier.arrive_and_wait()
                    # final semaphore release
                    if const_expr(self.deterministic and delay_semaphore_release):
                        if not release_now:
                            if const_expr(self.dm_tcfence):
                                dm_utils.fence_proxy_async_global()
                            barrier.arrive_inc_async(
                                mdQ_semaphore_cur[(m_block_max - 1, None)].iterator,
                                tidx,
                                cta_rank_in_cluster,
                                1,
                            )

                if const_expr(
                    self.deterministic
                    and not self.spt
                    and not self.use_block_sparsity
                    and block_info.window_size_left is not None
                ):
                    m_block_global_max = cute.ceil_div(seqlen.seqlen_q, self.tile_m)
                    for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                        if const_expr(self.dm_tcfence):
                            dm_utils.fence_proxy_async_global()
                        barrier.arrive_inc(
                            mdQ_semaphore_cur[(m_block, None)].iterator, tidx, cta_rank_in_cluster, 1
                        )

            self._dm_prog(Int32(5), dm_pt, Int32(0), Int32(15))
            dm_pt = dm_pt + 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.deterministic):
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        work_n: Int32,
        kv_block: Int32,
        seqlen,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
    ):
        wg_idx = (
            cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        ) // 128
        num_wg = cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128

        # On this path the scheduler head is the KV head.
        assert self.qhead_per_kvhead == 1 or self.pack_gqa or self.dm_gloop, "direct dK/dV store: MHA, pack_gqa or gloop"
        mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
        mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]

        # The gmem store below has no head-dim OOB protection from TMA, so when head_dim is
        # not a multiple of the tile we predicate each 128-bit store vector (8 bf16 columns).
        # The store's predicate granularity is one tmem-load repetition, so shrink the
        # repetition to 8 columns for that case; otherwise the widest chunk (<= 16) that still
        # gives every warpgroup a whole number of chunks of the lane's columns (hdim, or hdim / 2
        # with the M=64 layout).
        lane_cols_div = 2 if const_expr(self.acc_m64) else 1
        rep_dV = (
            8
            if const_expr(self.check_hdim_v_oob)
            else math.gcd(16, self.tile_hdimv // lane_cols_div // num_wg)
        )
        rep_dK = (
            8
            if const_expr(self.check_hdim_oob)
            else math.gcd(16, self.tile_hdim // lane_cols_div // num_wg)
        )
        tmem_load_atom_dV = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(rep_dV)), Float32
        )
        tmem_load_atom_dK = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(rep_dK)), Float32
        )
        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom_dV, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_rmem_tensor(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dv_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tiled_gmem_store_dV = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dV.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dV.tiler_mn,
        )

        tdVrdV_r2s = cute.make_rmem_tensor(tdVrdV_t2r.shape, self.dv_dtype)
        for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
            dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
            tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(mdV_cur, (self.mma_tiler_pdo[0], self.tile_hdimv), (None, 0))
        gdV_tile = gdV[None, None, work_n]  # partition_C selects this CTA's rows

        tdVgdV = thr_mma_dV.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        tdVpdV = None
        if const_expr(self.check_hdim_v_oob):
            tdVpdV = self.predicate_hdim(tdVcdV_t2r, self.head_dim_v)
        # The thread's row from its partition coordinates (per-CTA row = cluster row mod
        # tile_n); with the M=64 accumulator layout it is not the thread index.
        row_dV = tdVcdV_t2r[0][0] % self.tile_n
        if row_dV < seqlen.seqlen_k - self.tile_n * kv_block:
            cute.copy(tiled_gmem_store_dV, tdVrdV_r2s, tdVgdV_r2g, pred=tdVpdV)

        cute.arch.sync_warp()
        dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom_dK, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_rmem_tensor(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tiled_gmem_store_dK = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dK.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dK.tiler_mn,
        )

        tdKrdK_r2s = cute.make_rmem_tensor(tdKrdK_t2r.shape, self.dk_dtype)

        for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
            dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load()  # DM: Q codes carry the softmax scale
            tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(mdK_cur, (self.mma_tiler_dsq[0], self.tile_hdim), (None, 0))
        gdK_tile = gdK[None, None, work_n]

        tdKgdK = thr_mma_dK.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
        tdKgdK_r2g = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)

        tdKpdK = None
        if const_expr(self.check_hdim_oob):
            tdKpdK = self.predicate_hdim(tdKcdK_t2r, self.head_dim)
        row_dK = tdKcdK_t2r[0][0] % self.tile_n
        if row_dK < seqlen.seqlen_k - self.tile_n * kv_block:
            cute.copy(tiled_gmem_store_dK, tdKrdK_r2s, tdKgdK_r2g, pred=tdKpdK)

        cute.arch.sync_warp()
        dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()   # DM: needed once a CTA runs >1 tile (pers); no-op for a single tile
        return consumer_state_dKV

    @cute.jit
    def predicate_hdim(self, tcoord: cute.Tensor, limit: cutlass.Constexpr[int]) -> cute.Tensor:
        """Per-store-vector head-dim predicate for the non-TMA dK/dV epilogue store.

        ``tcoord`` is the (tile_n, hdim) identity tensor partitioned like the register data,
        shape ((rep, 1), chunks, 1, 1) with one contiguous ``rep``-wide column chunk per copy
        vector; the predicate drops the vector mode and marks each chunk whose first column
        is inside ``limit``. Valid because ``limit`` is a multiple of ``rep``.
        """
        tpred = cute.make_rmem_tensor(
            cute.make_layout((1, cute.size(tcoord, mode=[1]), 1, 1)), cutlass.Boolean
        )
        for i in cutlass.range_constexpr(cute.size(tcoord, mode=[1])):
            tpred[0, i, 0, 0] = tcoord[0, i, 0, 0][1] < limit
        return tpred

    @cute.jit
    def epilogue_dK_or_dV_tma_m64(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        kv_head: Int32,
        kv_block: Int32,
        seqlen,
        thr_mma: cute.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        mdKV_semaphore: Optional[cute.Tensor],
        K_or_V: cutlass.Constexpr[str],
    ) -> cutlass.pipeline.PipelineState:
        """dK / dV epilogue for NOTE [M=64 accumulator layout].

        Every thread owns (row, column-chunk) pieces of the (tile_n, hdim) accumulator and a
        warpgroup's columns are not contiguous, so the whole CTA stages the tile in SMEM through
        the layout-aware TMEM -> RMEM -> SMEM copies and one warp issues the TMA stores, one per
        (tile_n, epi) stage. The staging buffer's column and stage modes regroup into the
        (tile_n, hdim) tile view, keeping its strides and swizzle.
        """
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0
        deterministic_KV = self.deterministic and self.dKV_postprocess
        if const_expr(deterministic_KV):
            assert mdKV_semaphore is not None
            # One flag per KV tile (slot 0): the whole CTA accumulates as one unit here.
            mdKV_semaphore_cur = mdKV_semaphore[kv_block, None, kv_head, batch_idx]

        if const_expr(not self.dKV_postprocess):
            num_epi_stages = tile_hdim // epi_tile[1]
            assert cute.size(sdKV, mode=[2]) == num_epi_stages, "staging buffer must hold the tile"
            assert not seqlen.has_cu_seqlens_k, "varlen uses non tma store path"
            mdKV_cur = mdKV[None, None, kv_head, batch_idx]  # (seqlen, hdim)
            gdKV_p = cute.local_tile(mdKV_cur, (self.tile_n, tile_hdim), (kv_block, 0))
            gdKV_epi = cute.local_tile(gdKV_p, epi_tile, (0, None))  # (tile_n, 64, hdim / 64)
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdKV, 0, 2),
                cute.group_modes(gdKV_epi, 0, 2),
            )  # (TMA, EPI_STAGE) and (TMA, EPI_STAGE)
        else:
            # GQA: fp32 accumulate into the flat (tile_n * hdim) tile of the dK/dV accumulate
            # buffer in the order the postprocess reads it (self.dKVaccum_order). The fp32
            # staging buffer holds half the tile, so the tile goes out as its two row halves,
            # which the order has to store as one block each, laid out like a half-height tile
            rows_per_half = self.tile_n // 2
            half_elems = rows_per_half * tile_hdim
            half_bytes = half_elems * Float32.width // 8
            assert cute.size_in_bytes(dtype, sdKV) >= half_bytes, "fp32 staging too small"
            # (_check_accum_tile_orders holds the order to that.)
            tile_layout = accum_tile_layout(self.dKVaccum_order, self.tile_n, tile_hdim)
            half_layout = accum_tile_layout(self.dKVaccum_order, rows_per_half, tile_hdim)
            if const_expr(not seqlen.has_cu_seqlens_k):
                mdKV_cur = mdKV[None, kv_head, batch_idx]  # (seqlen * hdim)
            else:
                mdKV_cur = cute.domain_offset(
                    (seqlen.padded_offset_k * tile_hdim,), mdKV[None, kv_head]
                )
            gdKV_flat = cute.local_tile(mdKV_cur, (self.tile_n * tile_hdim,), (kv_block,))
            gdKV_tile = cute.make_tensor(gdKV_flat.iterator, tile_layout)  # (tile_n, hdim)
            sdKV_half = cute.make_tensor(
                cute.recast_ptr(sdKV.iterator, dtype=Float32), half_layout
            )  # (rows_per_half, hdim)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # TMEM -> RMEM (NOTE [M=64 accumulator layout]): a lane holds hdim / 2 columns of the
        # gradient and the two warpgroups split them, so a thread loads the widest chunk
        # (<= 32 columns) that leaves every warpgroup a whole number of chunks: one 32-column
        # chunk for hdim 128, one 16-column chunk for hdim 64. Narrower chunks come out of the
        # TMEM partition as a nested chunk mode that does not split evenly, so hdim must be a
        # multiple of 64 here (the interface keeps other head dims on the 128-row tile).
        assert (tile_hdim // 2 // num_wg) % 16 == 0, (
            "the M=64 dK/dV epilogue needs a head dim that is a multiple of 64"
        )
        rep = math.gcd(32, tile_hdim // 2 // num_wg)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(rep)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx % 128)
        tdKVtdKV_t2r = self.split_wg(thr_copy_t2r.partition_S(tdKVtdKV), wg_idx, num_wg)
        cdKV = cute.make_identity_tensor((self.cluster_tile_n, tile_hdim))
        tdKVcdKV = thr_mma.partition_C(cdKV)
        tdKVcdKV_t2r = self.split_wg(thr_copy_t2r.partition_D(tdKVcdKV), wg_idx, num_wg)
        tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)
        cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
        cute.arch.fence_view_async_tmem_load()

        if const_expr(scale is not None):
            for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = cute.arch.mul_packed_f32x2(
                    (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                )
        tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)
        tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))

        if const_expr(not self.dKV_postprocess):
            # RMEM -> SMEM through the (tile_n, hdim) view of the staging buffer: stage s holds
            # columns [s * epi, (s + 1) * epi), so regrouping the column and stage modes of
            # sdKV gives the tile view with the buffer's own strides and swizzle. Rank 3 like
            # the TMEM fragment, so the layout-aware r2s tiled copy can partition it.
            copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
                LayoutEnum.ROW_MAJOR, dtype, Float32, thr_copy_t2r
            )
            thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx % 128)
            sdKV_full = cute.make_tensor(
                sdKV.iterator,
                cute.make_layout(
                    ((sdKV.shape[0], (sdKV.shape[1], sdKV.shape[2])), 1, 1),
                    stride=((sdKV.stride[0], (sdKV.stride[1], sdKV.stride[2])), 0, 0),
                ),
            )
            tRS_sdKV = self.split_wg(thr_copy_r2s.partition_D(sdKV_full), wg_idx, num_wg)
            for chunk in cutlass.range_constexpr(cute.size(tRS_sdKV, mode=[1])):
                cute.autovec_copy(tdKVrdKV[None, chunk, 0, 0], tRS_sdKV[None, chunk, 0, 0])
            cute.arch.fence_view_async_shared()
            self.compute_sync_barrier.arrive_and_wait()

            # SMEM -> GMEM: one warp issues every epilogue stage, then waits for the reads
            if wg_idx == 0 and leader_warp:
                for epi_stage in cutlass.range_constexpr(num_epi_stages):
                    cute.copy(tma_atom_dKV, tdKVsdKV[None, epi_stage], tdKVgdKV[None, epi_stage])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            # The staging buffer is reused by the next tile / the other gradient
            self.compute_sync_barrier.arrive_and_wait()
        else:
            # Each thread owns column chunks of one row: (row, col0) per chunk from the
            # partition coordinates.
            row = tdKVcdKV_t2r[0][0] % self.tile_n
            num_chunks = cute.size(tdKVcdKV_t2r, mode=[1])
            chunk_cols = cute.size(tdKVrdKV, mode=[0])
            # One thread of the whole CTA drives the semaphore (tidx is warpgroup-local).
            tidx_cta = cute.arch.thread_idx()[0] % num_compute_threads
            if const_expr(deterministic_KV):
                # Q heads of a KV head accumulate in order: wait for the previous head's adds.
                if const_expr("nodkvw" not in self.dm_abl):
                    barrier.wait_eq(
                        mdKV_semaphore_cur.iterator,
                        tidx_cta,
                        0,
                        bwd_kv_group_pos(head_idx, qhead_per_kvhead=self.qhead_per_kvhead),
                    )
                self.compute_sync_barrier.arrive_and_wait()
            for half in cutlass.range_constexpr(2):
                if row // rows_per_half == half:
                    for chunk in cutlass.range_constexpr(num_chunks):
                        col0 = tdKVcdKV_t2r[(0, chunk, 0, 0)][1]
                        # this thread's chunk of the staging half: the chunk_cols columns of
                        # row (row - half * rows_per_half) starting at col0
                        row_view = cute.domain_offset((row - half * rows_per_half, col0), sdKV_half)
                        dst = cute.local_tile(row_view[0, None], (chunk_cols,), (0,))
                        cute.autovec_copy(tdKVrdKV[None, chunk, 0, 0], dst)
                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()
                if wg_idx == 0 and leader_warp:
                    gdKV_half = cute.local_tile(gdKV_tile, (rows_per_half, tile_hdim), (half, 0))
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV_half.iterator, gdKV_half.iterator, half_bytes
                        )
                    cute.arch.cp_async_bulk_commit_group()
                    # Deterministic: the adds must have landed before the next head is released.
                    cute.arch.cp_async_bulk_wait_group(0, read=not deterministic_KV)
                self.compute_sync_barrier.arrive_and_wait()
            if const_expr(deterministic_KV):
                if const_expr(self.dm_tcfence):
                    dm_utils.fence_proxy_async_global()
                barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx_cta, 0, 1)

        cute.arch.sync_warp()
        dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        kv_head: Int32,
        kv_block: Int32,
        seqlen,
        thr_mma: cute.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdKV_semaphore: Optional[cute.Tensor],
        K_or_V: cutlass.Constexpr[str],
        dm_tlp=None,
        dm_tsw=None,
    ) -> cutlass.pipeline.PipelineState:
        assert K_or_V in ("K", "V")
        if const_expr(self.acc_m64):
            return self.epilogue_dK_or_dV_tma_m64(
                tidx,
                batch_idx,
                head_idx,
                kv_head,
                kv_block,
                seqlen,
                thr_mma,
                tdKVtdKV,
                mdKV,
                sdKV,
                tma_atom_dKV,
                pipeline_dKV,
                consumer_state_dKV,
                scale,
                mdKV_semaphore,
                K_or_V,
            )
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        flat_epi_tile = (
            self.sdK_flat_epi_tile if const_expr(K_or_V == "K") else self.sdV_flat_epi_tile
        )
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        if const_expr(not self.dKV_postprocess):
            sdKV = sdKV[None, None, wg_idx]  # (tile_n, 64) for bf16
        else:
            sdKV = sdKV[None, wg_idx]  # (tile_n * 32) for fp32

        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

        if const_expr(not self.dKV_postprocess):
            assert not seqlen.has_cu_seqlens_k, "varlen uses non tma store path"
            mdKV_cur = mdKV[None, None, kv_head, batch_idx]  # (seqlen, hdim)
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n, tile_hdim), (kv_block, 0)
            )  # (tile_n, hdim) - per CTA
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tile_n, hdim / 2)
            gdKV_epi = cute.local_tile(
                gdKV, epi_tile, (0, None)
            )  # (tile_n, 64, epi_stage = (hdim / 2) / 64)
        else:
            # n_block_group = kv_block // self.cta_group_size
            if const_expr(not seqlen.has_cu_seqlens_k):
                mdKV_cur = mdKV[None, kv_head, batch_idx]  # (seqlen * hdim)
            else:
                mdKV_cur = cute.domain_offset(
                    (seqlen.padded_offset_k * tile_hdim,), mdKV[None, kv_head]
                )
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n * tile_hdim,), (kv_block,)
            )  # (tile_n * hdim)
            gdKV = cute.logical_divide(gdKV_p, (self.tile_n * tile_hdim // num_wg,))[
                ((None, wg_idx),)
            ]  # (tile_n * hdim / 2)
            gdKV_epi = cute.flat_divide(
                gdKV, (flat_epi_tile,)
            )  # (tile_n * hdim / 2 / epi_stage, epi_stage)

        deterministic_KV = self.deterministic and self.dKV_postprocess
        if const_expr(deterministic_KV):
            assert mdKV_semaphore is not None
            mdKV_semaphore_cur = mdKV_semaphore[kv_block, None, kv_head, batch_idx]

        if const_expr(not self.dKV_postprocess):
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdKV, 0, 2),
                cute.group_modes(gdKV_epi, 0, 2),
            )  # (TMA) and (TMA, EPI_STAGE)
            assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
            assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
            num_epi_stages = cute.size(tdKVgdKV.shape[1])
            if const_expr(K_or_V == "K"):
                assert num_epi_stages == self.num_epi_stages, "Epi stage calculation is wrong (K)"
            else:
                assert num_epi_stages == self.num_epi_stages_v, "Epi stage calculation is wrong (V)"
        else:
            num_epi_stages = (
                self.num_epi_stages if const_expr(K_or_V == "K") else self.num_epi_stages_v
            )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dK_reduce_ncol)), Float32
        )

        read_flag = const_expr(not deterministic_KV or "nofull" in self.dm_abl)

        pipeline_dKV.consumer_wait(consumer_state_dKV)
        if const_expr(dm_tlp is not None):
            if tidx == 0:
                dm_utils.red_add_u64(dm_tlp, dm_utils.globaltimer())

        # semaphore acquire
        dm_kv_first = cutlass.Boolean(False)
        if const_expr(deterministic_KV):
            dm_kv_pos = bwd_kv_group_pos(head_idx, qhead_per_kvhead=self.qhead_per_kvhead)
            if const_expr("nodkvw" not in self.dm_abl):
                if const_expr(dm_tsw is not None):
                    dm_w0 = dm_utils.globaltimer()
                barrier.wait_eq(
                    mdKV_semaphore_cur.iterator,
                    tidx,
                    wg_idx,
                    dm_kv_pos,
                )
                if const_expr(dm_tsw is not None):
                    # DM semw (instrumentation): dK/dV semaphore wait ns / count (per WG) -> trace header [46], [47]
                    if tidx == 0:
                        dm_utils.red_add_u64(dm_tsw, dm_utils.globaltimer() - dm_w0)
                        dm_utils.red_add_u64(dm_tsw + 8, Int64(1))
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            if const_expr(dm_tlp is not None):
                if tidx == 0:
                    dm_utils.red_add_u64(dm_tlp + 8, dm_utils.globaltimer())
            if const_expr(self.dm_fws):
                dm_kv_first = dm_kv_pos == 0   # DM fws: first q head of the group stores (no dK/dV accum zero-fill)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((self.cluster_tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, (
                "RMEM<->TMEM fragment size mismatch"
            )

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            if const_expr(scale is not None):
                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = cute.arch.mul_packed_f32x2(
                        (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                    )
            tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)  # (32 columns)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                if const_expr(not self.dKV_postprocess):
                    cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                else:
                    with cute.arch.elect_one():
                        if dm_kv_first:
                            dm_utils.cpasync_bulk_store_s2g(
                                sdKV.iterator,
                                gdKV_epi[None, epi_stage].iterator,
                                self.tma_copy_bytes["dKacc"],
                            )
                        else:
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdKV.iterator,
                                gdKV_epi[None, epi_stage].iterator,
                                self.tma_copy_bytes["dKacc"],
                            )
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    # DM epird: intermediate stages only need the smem READ done before reuse; global completion
                    # (needed before the semaphore release) is awaited once after the last stage (same bits)
                    cute.arch.cp_async_bulk_wait_group(
                        0, read=(read_flag or const_expr("epird" in self.dm_opts)))
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )

        # semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(deterministic_KV):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            if const_expr(self.dm_tcfence):
                dm_utils.fence_proxy_async_global()
            barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

        cute.arch.sync_warp()
        dm_utils.tcf_relay()   # fwdfence: all lanes fence + warp sync before the leader's arrive
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV
