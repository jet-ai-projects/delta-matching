/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/
// ## FP11-M17 PASS 0 v4 (T2a, bitwise delta): pass 0 v3 (256 keys per CTA, two
// n-blocks per warpgroup) with the wgmma issue SOFTWARE-PIPELINED ACROSS STAGES.
//
// WHY: ncu on v3 (locked clocks): tensor pipe 57% active,
// top stalls wait (wgmma) 21% / dispatch 13% / long_scoreboard 13%.  In v3 each
// warpgroup issues its four groups (S0 dP0 S1 dP1) at the top of a stage and the
// tensor pipe goes idle between the retirement of dP1 and the NEXT stage's issue
// (rowsum(1) + the stage wait + the issue latency), and the two warpgroups share
// every stage so their gaps coincide.  v4 issues the next stage's (S0, dP0) right
// after rowsum(0) of the current stage (its fragments are dead by then; no new
// accumulator registers), so two groups are always in flight across the stage
// boundary; the last stage falls back to the v3 tail.
//
// WHAT IS IDENTICAL: every wgmma group (same atoms, same operand bytes, same
// zero-init k chain), the exp2 line, the per-half fixed-order fmaf chains and
// their [BH, nbn, 2, N] slots, the reduce.  Only WHEN independent groups are
// issued changes.  Checked: delta / dq / dk bitwise equal to the earlier pass 0 and to the M16
// build, determinism.
//
#pragma once

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "cuda_check.h"
#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "static_switch.h"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "utils.h"
#include "softmax.h"

#include "flash_bwd_fp8_params.h"

// ## FP11-M17 P0MS (T2b, bitwise): m-split of the pass-0 CTAs for the causal tail.  With 256-key
// CTAs the longest CTA (128 m-blocks x 4 n-blocks) is ~2/3 of the whole pass-0 time at H6 (384 CTAs =
// 2.9 waves), so the LPT tail costs ~15% there.  P0MS = the maximum m-blocks per CTA: each (bh,
// group) m range is cut into chunks of that length, full chunks first, remainders by length
// descending (a host-built unit table, cached per shape); every (m-block, n-block) tile is computed
// exactly as before into the same [BH, nbn, 2, N] slot (disjoint across chunks), so delta is bitwise.
// 0 = off (the v3 LPT scheduler).
#ifndef FA3FP11_M16_P0V3_NBWG
#define FA3FP11_M16_P0V3_NBWG 2
#endif

namespace flash {

using namespace cute;

// ## Delta-Matching-128 (M19): kHeadDim_ template parameter (default 256 = the M17 instantiation, textually
// unchanged).  At hd128 an n-block (64 keys) is ONE scale group (the [128, 64] contract): both 32-key halves of the
// [BH, nbn, 2, N] slot layout carry the same k8s / v8s (the fixed-order reduce is unchanged), and the Q/dO scale is
// per 128 rows (m-block >> 1).  Q/dO stages are 8 KB, K/V n-blocks 8 KB: 256 keys per CTA = 32 KB K/V resident.
template <bool Is_causal, int kAtomN_, int kNbPerWG_, int kHeadDim_ = 256>
struct DeltaV4 {
    using Element = cutlass::float_e4m3_t;
    using ElementAccum = float;
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 64;       // per n-block
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kQuantKeys = kHeadDim >= 256 ? 32 : 64;
    static constexpr int kScalesPerNb = kBlockN / kQuantKeys;     // 2 at d256, 1 at hd128
    static constexpr int kQShift = kHeadDim >= 256 ? 0 : 1;      // m-blocks per Q/dO scale block = 1 << kQShift
    // ## Delta-Matching-128 lever P0-D (M19, hd128 only): the Q/dO stage tiles are half the d256 size at head_dim 128, so a
    // deeper pass-0 pipeline can fit; the d256 pin stays 3.  Depth changes no arithmetic (bitwise expected, measured).
    static constexpr int kStages = kHeadDim_ >= 256 ? 3 : FA3FP11_HD128_P0ST;
    static constexpr int kAtomN = kAtomN_;
    static_assert(kAtomN == 32 || kAtomN == 64, "kAtomN must be 32 (certified realization) or 64 (probe)");
    static constexpr int NumMmaWarpGroups = 2;
    static constexpr int kNbPerWG = kNbPerWG_;                       // n-blocks per consumer warpgroup
    // ## Delta-Matching-128 lever P0-B (M19): kNbPerWG 4 = 512 keys per CTA, each warpgroup's four n-blocks processed as two
    // software-pipelined PAIRS per stage (the v4b cross-stage issue applied across pairs); kNbPerWG 2 is the v4 path verbatim.
    static_assert(kNbPerWG == 1 || kNbPerWG == 2 || kNbPerWG == 4, "kNbPerWG must be 1 (v2 geometry), 2 (256 keys per CTA) or 4 (512, P0-B)");
    static constexpr int kPairs = kNbPerWG / 2;
    static constexpr int kNbPerCTA = NumMmaWarpGroups * kNbPerWG;    // resident n-blocks
    static constexpr int kGroupN = kNbPerCTA * kBlockN;              // keys per CTA
    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int MaxThreadsPerBlock = NumMmaThreads + NumProducerThreads;
    static constexpr int MinBlocksPerMultiprocessor = 1;   // cutlass::device_kernel launch bounds
    static constexpr uint32_t LoadRegisterRequirement = 40;
    static constexpr uint32_t MmaRegisterRequirement = 232;

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using SeqlenInfo_t = flash::SeqlenInfoQK<false, kBlockM>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, false>;

    // The S / dP tiled MMA: ONE warpgroup, atom m64 x kAtomN x k32, C tile 64x64
    // (MMA_N = 64 / kAtomN).  kAtomN = 32 reproduces the mainloop's
    // TileShapeAtomSdP = (64, 32, 256) atom exactly.
    using TileShapeAtomS = Shape<Int<kBlockM>, Int<kAtomN>, Int<kHeadDim>>;
    using TiledMmaS = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomS>(),
        Layout<Shape<_1, _1, _1>>{}));

    // Smem layouts: the mainloop's atoms (SmemLayoutAtomQdO / SmemLayoutAtomK
    // with the same selector arguments), so the TMA boxes and swizzles are
    // the ones the certified kernels stage through.
    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDim / 2>>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages>{})));
    using SmemLayoutdO = SmemLayoutQ;
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim / 2>>());
    using SmemLayoutK1 = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));   // one n-block
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kNbPerCTA>{})));                      // kNbPerCTA n-blocks
    using SmemLayoutV = SmemLayoutK;
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<kBlockM>>>;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;

    using TMA_QdO = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}),
        TileShape_MNK{},
        ClusterShape{}));
    using TMA_K = decltype(make_tma_copy_B_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK1{},
        TileShape_MNK{},
        ClusterShape{}));

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using Scheduler = flash::SingleTileBwdLPTScheduler<false /*Varlen*/, kGroupN, false /*SPT*/>;

    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK1 = static_cast<uint32_t>(size(SmemLayoutK1{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);
    static constexpr uint32_t TmaTransactionBytesStage = 2 * TmaTransactionBytesQ + TmaTransactionBytesLSE;   // Q + dO + LSE

    struct TensorStorage : cute::aligned_struct<256> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 256> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, 256> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 256> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, 256> smem_do;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
    };
    struct SharedStorage {
        TensorStorage tensors;
        struct PipelineStorage : cute::aligned_struct<16> {
            alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_KV;
            alignas(16) typename MainloopPipeline::SharedStorage pipeline_qdo;
            alignas(16) typename Scheduler::SharedStorage smem_scheduler;
        } pipelines;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    static_assert(SharedStorageSize <= 232448, "DeltaV4: shared storage exceeds the sm90 227 KB limit");
    static_assert(kPairs == 1 || kHeadDim < 256, "DeltaV4: the two-pair CTA (P0-B) is the hd128 lever");

    struct Params {
        ShapeQKV shape_Q;
        ShapeQKV shape_K;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_QdO tma_load_Q, tma_load_dO;
        TMA_K tma_load_K, tma_load_V;
        float const* ptr_q8s; float const* ptr_k8s; float const* ptr_v8s; float const* ptr_do8s;
        int nblk_q, nblk_k;
        float const* ptr_LSE_log2;
        ShapeLSE shape_LSE;
        StrideLSE stride_LSE_log2;
        float* ptr_dltpart;
        int nbm, nbn, ngroups;
        int num_batch;
        typename Scheduler::Params scheduler;
        int const* unit_tab;   // ## FP11-M17 P0MS: [nunits][4] = (bh, group, m_lo, m_hi); nullptr when off
        int nunits;
        int num_heads;
    };

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();
        if (warp_idx == 0 && lane_predicate) { prefetch_tma_descriptors(params); }
        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        int const warp_group_idx = cutlass::canonical_warp_group_idx();

        typename MainloopPipeline::Params pipeline_params;
        pipeline_params.transaction_bytes = TmaTransactionBytesStage;
        pipeline_params.role = warp_group_idx == 0
            ? MainloopPipeline::ThreadCategory::Producer
            : MainloopPipeline::ThreadCategory::Consumer;
        pipeline_params.is_leader = warp_group_thread_idx == 0;
        pipeline_params.num_consumers = NumMmaThreads;
        if (warp_idx == 0 && lane_predicate) {
            shared_storage.pipelines.barrier_KV.init(1 /*numThreads*/);
        }
        // pipeline ctor fences the barrier init
        MainloopPipeline pipeline_qdo(shared_storage.pipelines.pipeline_qdo, pipeline_params, ClusterShape{});
        __syncthreads();

        int group, bidh, bidb, m_block_min, m_block_max;
        if constexpr (FA3FP11_M17_P0MS > 0) {
            // ## FP11-M17 P0MS: one work unit = (bh, group, [m_lo, m_hi)) from the host-built table.
            int const u = blockIdx.x;
            if (u >= params.nunits) { return; }
            int const* e = params.unit_tab + 4 * u;
            int const bh = e[0];
            group = e[1]; m_block_min = e[2]; m_block_max = e[3];
            bidb = bh / params.num_heads; bidh = bh - bidb * params.num_heads;
            if (m_block_max <= m_block_min) { return; }
        } else {
            Scheduler scheduler(reinterpret_cast<typename Scheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));
            auto work_tile_info = scheduler.template get_initial_work<false>(params.scheduler);
            if (!work_tile_info.is_valid(params.scheduler)) { return; }
            auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
            group = get<0>(block_coord_);
            bidh = get<1>(block_coord_);
            bidb = get<2>(block_coord_);
            int const nb0_ = kNbPerCTA * group;
            SeqlenInfo_t seqlen_info_{bidb, get<0>(params.shape_Q), size<0>(params.shape_K), nullptr, nullptr, nullptr, nullptr};
            // The CTA's m range = the union of its n-blocks' ranges = the lowest n-block's (causal: the smallest min).
            auto mm = BlockMN_t::get_m_block_min_max(seqlen_info_, nb0_, bidb, -1, -1, 0);
            m_block_min = get<0>(mm); m_block_max = get<1>(mm);
            if constexpr (Is_causal) {
                if (m_block_max <= m_block_min) { return; }
            }
        }
        int const nb0 = kNbPerCTA * group;
        SeqlenInfo_t seqlen_info{bidb, get<0>(params.shape_Q), size<0>(params.shape_K), nullptr, nullptr, nullptr, nullptr};
        int const bidh_kv = params.qhead_per_khead_divmod.divide(bidh);

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.smem_v.data()), SmemLayoutV{});
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.smem_lse.data()), SmemLayoutLSE{});

        if (warp_group_idx == 0) {  // ---------------- producer warpgroup ----------------
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            if (warp_idx_in_warpgroup != 0) { return; }
            Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, bidb);
            Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_Q)(_, _, bidh, bidb);
            Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, bidb);
            Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_K)(_, _, bidh_kv, bidb);
            Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, bidh, bidb);
            Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));    // (M, K, nbm)
            Tensor gdO = local_tile(mdO, select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
            Tensor gLSE = local_tile(mLSE, select<0>(TileShape_MNK{}), make_coord(_));           // (M, nbm)

            auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
            auto block_tma_dO = params.tma_load_dO.get_slice(_0{});
            Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
            Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
            Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
            Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
            auto kv_part = [&](auto const& gX, auto const& sX_stage, auto const& tma) {
                Tensor sX_x = make_tensor(sX_stage.data(), make_layout(sX_stage.layout(), Layout<_1>{}));
                Tensor gX_x = make_tensor(gX.data(), make_layout(gX.layout(), Layout<_1>{}));
                return tma_partition(tma, _0{}, Layout<_1>{}, group_modes<0, 2>(sX_x), group_modes<0, 2>(gX_x));
            };
            auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};

            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
            if (lane_predicate) {
                // K/V: kNbPerCTA + kNbPerCTA one-shot tiles on the KV barrier (n-blocks past nbn load nb0: clamped, never stored).
                auto& barKV = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV);
                shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(2 * kNbPerCTA * TmaTransactionBytesK1);
                #pragma unroll
                for (int j = 0; j < kNbPerCTA; ++j) {
                    int const nb = nb0 + j < params.nbn ? nb0 + j : nb0;
                    Tensor gKj = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(nb, _0{}));      // (N, K)
                    Tensor gVj = local_tile(mV, select<1, 2>(TileShape_MNK{}), make_coord(nb, _0{}));
                    auto [tKgK, tKsK] = kv_part(gKj, sK(_, _, j), params.tma_load_K);
                    auto [tVgV, tVsV] = kv_part(gVj, sV(_, _, j), params.tma_load_V);
                    copy(params.tma_load_K.with(barKV, 0), tKgK, tKsK);
                    copy(params.tma_load_V.with(barKV, 0), tVgV, tVsV);
                }
                #pragma unroll 1
                for (int m_block = m_block_min; m_block < m_block_max; ++m_block) {
                    pipeline_qdo.producer_acquire(smem_pipe_write);
                    auto& bar = *pipeline_qdo.producer_get_barrier(smem_pipe_write);
                    copy(params.tma_load_Q.with(bar, 0, TMA::CacheHintSm90::EVICT_LAST),
                         tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
                    copy(params.tma_load_dO.with(bar, 0, TMA::CacheHintSm90::EVICT_LAST),
                         tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write.index()));
                    copy(bulk_copy.with(bar), gLSE(_, m_block), sLSE(_, smem_pipe_write.index()));
                    ++smem_pipe_write;
                }
                pipeline_qdo.producer_tail(smem_pipe_write);
            }
        } else {  // ---------------- consumer warpgroups ----------------
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
            int const thread_idx = threadIdx.x - NumProducerThreads;   // 0..255
            int const wg = thread_idx / cutlass::NumThreadsPerWarpGroup;  // this warpgroup's n-block slots: kNbPerWG*wg + j
            int const wtid = thread_idx % cutlass::NumThreadsPerWarpGroup;
            int const nb_base = nb0 + kNbPerWG * wg;
            int const seqlen_q = seqlen_info.seqlen_q;
            int const seqlen_k = seqlen_info.seqlen_k;

            TiledMmaS tiled_mma;
            auto wg_mma = tiled_mma.get_slice(_0{});
            auto thread_mma = tiled_mma.get_thread_slice(wtid);
            Tensor tSrQ = wg_mma.partition_fragment_A(sQ);                 // (MMA, MMA_M, MMA_K, stages)
            Tensor tSrK = wg_mma.partition_fragment_B(sK);                 // (MMA, MMA_N, MMA_K, kNbPerCTA)
            Tensor tdPrdO = wg_mma.partition_fragment_A(sdO);
            Tensor tdPrV = wg_mma.partition_fragment_B(sV);
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = thread_mma.partition_C(cS);
            Tensor tScS_rc = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));
            int const row0 = get<0>(tScS_rc(_0{}, _0{}));
            int const row1 = get<0>(tScS_rc(_1{}, _0{}));

            cutlass::FastDivmod chunk_divmod(1);
            chunk_divmod.divisor = 0;
            flash::Mask<kBlockM, kBlockN, false /*PackGQA*/, TiledMmaS, false /*SwapAB*/> mask(
                wtid, seqlen_q, seqlen_k, -1, -1, 0 /*sink_token_length*/,
                chunk_divmod, params.qhead_per_khead_divmod);

            // Scales (mma() bookkeeping, per KV head; per n-block j: two 32-key groups).
            int const fq8_bh = bidb * get<2>(params.shape_Q) + bidh;
            int const fq8_bh_kv = bidb * get<2>(params.shape_K) + bidh_kv;
            float fq8_k8s_g0[kNbPerWG], fq8_k8s_g1[kNbPerWG], fq8_v8s_g0[kNbPerWG], fq8_v8s_g1[kNbPerWG];
            float* fq8_part_h0[kNbPerWG];
            float* fq8_part_h1[kNbPerWG];
            bool nb_valid[kNbPerWG];
            #pragma unroll
            for (int j = 0; j < kNbPerWG; ++j) {
                int const n_block = nb_base + j;
                nb_valid[j] = n_block < params.nbn;
                int const nbc = nb_valid[j] ? n_block : nb0;   // clamped like the producer (values unused when invalid)
                fq8_k8s_g0[j] = params.ptr_k8s[fq8_bh_kv * params.nblk_k + kScalesPerNb * nbc];
                fq8_k8s_g1[j] = params.ptr_k8s[fq8_bh_kv * params.nblk_k + kScalesPerNb * nbc + (kScalesPerNb - 1)];
                fq8_v8s_g0[j] = params.ptr_v8s[fq8_bh_kv * params.nblk_k + kScalesPerNb * nbc];
                fq8_v8s_g1[j] = params.ptr_v8s[fq8_bh_kv * params.nblk_k + kScalesPerNb * nbc + (kScalesPerNb - 1)];
                // This n-block's two half columns of the [BH, nbn, 2, N] workspace.
                fq8_part_h0[j] = params.ptr_dltpart
                    + ((int64_t(fq8_bh) * params.nbn + nbc) * 2 + 0) * int64_t(seqlen_q);
                fq8_part_h1[j] = fq8_part_h0[j] + int64_t(seqlen_q);
            }
            float const* fq8_q8s_base = params.ptr_q8s + fq8_bh * params.nblk_q;
            float const* fq8_do8s_base = params.ptr_do8s + fq8_bh * params.nblk_q;
            static constexpr float kFq8MoE = 8.0f;
            static constexpr float kFq8MoInv = 1.0f / 256.0f;

            cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(0));
            if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.pipelines.barrier_KV.wait(0); }

            // The v2 per-tile arithmetic, verbatim, as two steps applied per n-block j:
            // (1) S~ -> P~ (mask, descale, LSE, F1 clamp, exp2), (2) the fixed-order half row sums.
            auto exp_step = [&](auto& tSrS, int m_block, int n_block, float lse0, float lse1, float dsc_g0, float dsc_g1) {
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
                static_assert(decltype(size<0>(scores))::value == 2 && decltype(size<1>(scores))::value == 16,
                              "DeltaV4: expected 2 rows x 16 columns per thread for a 64x64 tile");
                mask.template apply<true /*Seqlenk_mask*/, Is_causal, false /*Is_local*/>(tSrS, m_block, n_block);
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    float const lse_scaled = mi == 0 ? lse0 : lse1;
                    #pragma unroll
                    for (int ni = 0; ni < 16; ++ni) {
                        float const dsc = (ni >> 3) == 0 ? dsc_g0 : dsc_g1;
                        // The mma() exp2 line VERBATIM (F1 clamp included).
                        scores(mi, ni) = exp2f(fminf(scores(mi, ni) * dsc - lse_scaled + kFq8MoE, kFq8MoE + 4.0f));
                    }
                }
            };
            flash::SumOp<float> fq8_sum_op;
            auto rowsum_step = [&](auto& tSrS, auto& tdPrdP, int m_block, int j, float dp_g0, float dp_g1) {
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
                Tensor dPv = make_tensor(tdPrdP.data(), scores.layout());
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    // half 0: rowcol columns 0..7 (physical keys 0..31 of the n-block), half 1: 8..15.
                    float a0 = 0.f, a1 = 0.f;
                    #pragma unroll
                    for (int ni = 0; ni < 8; ++ni) {
                        a0 = fmaf(scores(mi, ni), dPv(mi, ni) * dp_g0, a0);
                    }
                    #pragma unroll
                    for (int ni = 8; ni < 16; ++ni) {
                        a1 = fmaf(scores(mi, ni), dPv(mi, ni) * dp_g1, a1);
                    }
                    a0 = flash::Allreduce<4>::run(a0, fq8_sum_op);
                    a1 = flash::Allreduce<4>::run(a1, fq8_sum_op);
                    if ((wtid & 3) == 0 && nb_valid[j]) {
                        int const fq8_row = mi == 0 ? row0 : row1;
                        fq8_part_h0[j][m_block * kBlockM + fq8_row] = a0 * kFq8MoInv;
                        fq8_part_h1[j][m_block * kBlockM + fq8_row] = a1 * kFq8MoInv;
                    }
                }
            };

            float fq8_q8sc_next = fq8_q8s_base[m_block_min >> kQShift];
            float fq8_do8sc_next = fq8_do8s_base[m_block_min >> kQShift];
            PipelineState smem_pipe_read;
            static_assert(kNbPerWG == 2 || kNbPerWG == 4, "DeltaV4: the cross-stage software pipeline is written for two n-blocks per warpgroup (or two pairs, P0-B)");
            int const slot0 = kNbPerWG * wg;   // this WG's first slot; pair pp uses slots (slot0 + 2 pp, slot0 + 2 pp + 1)
            Tensor tSrS0 = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            Tensor tdPrdP0 = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            Tensor tSrS1 = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            Tensor tdPrdP1 = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            // Prologue: the first stage's (S0, dP0) of the first pair go in flight before the loop.
            {
                auto tok = pipeline_qdo.consumer_try_wait(smem_pipe_read);
                pipeline_qdo.consumer_wait(smem_pipe_read, tok);
                int const stage = smem_pipe_read.index();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tSrQ(_, _, _, stage), tSrK(_, _, _, slot0), tSrS0);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tdPrdO(_, _, _, stage), tdPrV(_, _, _, slot0), tdPrdP0);
                warpgroup_wait<0>();
                warpgroup_fence_operand(tSrS0);
                warpgroup_fence_operand(tdPrdP0);
            }
            // The unit body = one PAIR of n-blocks (slots sl, sl + 1) of one stage; HasNext / LastPair are compile-time bools
            // so every warpgroup_wait<N> sits on straight-line code (a runtime-selected wait count made ptxas inject waits and
            // serialize the wgmma pipeline: C7514/C7517 on the first v4 build).  The NEXT unit is either the next pair of the
            // same stage (!LastPair: its (S0, dP0) are issued with no pipeline wait) or the first pair of the next stage
            // (LastPair: acquire it first; release the current stage once this pair's dP1 retired).  kNbPerWG 2 = one pair
            // per stage = the v4 step verbatim.
            auto step = [&](int m_block, auto pp_c, auto has_next_c, auto last_pair_c) {
                constexpr int pp = decltype(pp_c)::value;
                constexpr bool HasNext = decltype(has_next_c)::value;
                constexpr bool LastPair = decltype(last_pair_c)::value;
                int const sl0 = slot0 + 2 * pp, sl1 = sl0 + 1;               // this pair's K/V smem slots
                int const nsl0 = LastPair ? slot0 : sl0 + 2;                  // the next unit's first slot
                int const j0 = 2 * pp, j1 = j0 + 1;                           // this pair's n-block indices within the WG
                // stage s is already acquired (prologue / previous unit); (S0, dP0) of this pair are in flight.
                int const stage = smem_pipe_read.index();
                float const lse0 = __fmul_rn(sLSE(row0, stage), float(M_LOG2E));   // ## FP11-M17 LSEK
                float const lse1 = __fmul_rn(sLSE(row1, stage), float(M_LOG2E));
                float const fq8_q8sc = fq8_q8sc_next;
                float const fq8_do8sc = fq8_do8sc_next;
                if constexpr (HasNext && LastPair) {
                    fq8_q8sc_next = fq8_q8s_base[(m_block + 1) >> kQShift];
                    fq8_do8sc_next = fq8_do8s_base[(m_block + 1) >> kQShift];
                }
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tSrQ(_, _, _, stage), tSrK(_, _, _, sl1), tSrS1);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tdPrdO(_, _, _, stage), tdPrV(_, _, _, sl1), tdPrdP1);
                // in flight: S1 dP1 (S0 / dP0 retired at the previous unit's tail: no wgmma group is ever carried across the
                // loop back-edge, which ptxas cannot track and answered with C7514 / C7517 on v4's first cut)
                warpgroup_wait<2>();   // no-op guard: S0 already retired
                warpgroup_fence_operand(tSrS0);
                exp_step(tSrS0, m_block, nb_base + j0, lse0, lse1,
                         fq8_q8sc * fq8_k8s_g0[j0] * float(M_LOG2E), fq8_q8sc * fq8_k8s_g1[j0] * float(M_LOG2E));
                warpgroup_fence_operand(tdPrdP0);   // dP0 retired at the previous tail
                rowsum_step(tSrS0, tdPrdP0, m_block, j0, fq8_do8sc * fq8_v8s_g0[j0], fq8_do8sc * fq8_v8s_g1[j0]);
                // tSrS0 / tdPrdP0 are dead: issue the NEXT unit's (S0, dP0) into them now.
                PipelineState smem_pipe_read_next = smem_pipe_read;
                if constexpr (LastPair) { ++smem_pipe_read_next; }
                if constexpr (HasNext) {
                    if constexpr (LastPair) {
                        auto tok = pipeline_qdo.consumer_try_wait(smem_pipe_read_next);
                        pipeline_qdo.consumer_wait(smem_pipe_read_next, tok);
                    }
                    int const stage_n = smem_pipe_read_next.index();
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tSrQ(_, _, _, stage_n), tSrK(_, _, _, nsl0), tSrS0);
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tdPrdO(_, _, _, stage_n), tdPrV(_, _, _, nsl0), tdPrdP0);
                    // in flight: S1 dP1 S0n dP0n
                    warpgroup_wait<3>();   // S1 retired
                } else {
                    warpgroup_wait<1>();   // S1 retired
                }
                warpgroup_fence_operand(tSrS1);
                exp_step(tSrS1, m_block, nb_base + j1, lse0, lse1,
                         fq8_q8sc * fq8_k8s_g0[j1] * float(M_LOG2E), fq8_q8sc * fq8_k8s_g1[j1] * float(M_LOG2E));
                if constexpr (HasNext) { warpgroup_wait<2>(); } else { warpgroup_wait<0>(); }   // dP1 retired: every read of this pair is done
                warpgroup_fence_operand(tdPrdP1);
                if constexpr (LastPair) {
                    pipeline_qdo.consumer_release(smem_pipe_read);   // every read of stage s is done
                    smem_pipe_read = smem_pipe_read_next;
                }
                rowsum_step(tSrS1, tdPrdP1, m_block, j1, fq8_do8sc * fq8_v8s_g0[j1], fq8_do8sc * fq8_v8s_g1[j1]);
                // Retire the next unit's (S0, dP0) before the back-edge (see the note at the top of
                // the step); their tensor time overlapped exp/rowsum of the pair's second n-block above.
                warpgroup_wait<0>();
                warpgroup_fence_operand(tSrS0);
                warpgroup_fence_operand(tdPrdP0);
            };
            CUTLASS_PRAGMA_NO_UNROLL
            for (int m_block = m_block_min; m_block < m_block_max - 1; ++m_block) {
                if constexpr (kPairs == 2) { step(m_block, cute::Int<0>{}, cute::true_type{}, cute::false_type{}); }
                step(m_block, cute::Int<kPairs - 1>{}, cute::true_type{}, cute::true_type{});
            }
            if constexpr (kPairs == 2) { step(m_block_max - 1, cute::Int<0>{}, cute::true_type{}, cute::false_type{}); }
            step(m_block_max - 1, cute::Int<kPairs - 1>{}, cute::false_type{}, cute::true_type{});
        }
    }
};

}  // namespace flash

// Host launch: pass 0 v3 + the unchanged pass 0.5 reduce (fp11_delta.h's kernel).
template <bool Is_causal, int kHeadDim_>
void run_flash_delta_v4_fp11_impl(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    using Kernel = flash::DeltaV4<Is_causal, FA3FP11_M16_P0V2_ATOMN, (kHeadDim_ < 256 ? FA3FP11_HD128_P0NBWG : FA3FP11_M16_P0V3_NBWG), kHeadDim_>;
    using Element = typename Kernel::Element;
    using namespace cute;
    int const seqlen_q = params.seqlen_q;
    int const seqlen_k = params.seqlen_k;
    int const nbm = cute::ceil_div(seqlen_q, Kernel::kBlockM);
    int const nbn = cute::ceil_div(seqlen_k, Kernel::kBlockN);
    int const ngroups = cute::ceil_div(nbn, Kernel::kNbPerCTA);

    typename Kernel::ShapeQKV shape_Q = make_shape(seqlen_q, params.d, params.h, params.b);
    typename Kernel::StrideQKV stride_Q = make_stride(params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride);
    typename Kernel::StrideQKV stride_dO = make_stride(params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride);
    typename Kernel::ShapeQKV shape_K = make_shape(seqlen_k, params.d, params.h_k, params.b);
    typename Kernel::StrideQKV stride_K = make_stride(params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride);
    typename Kernel::StrideQKV stride_V = make_stride(params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride);
    Tensor mQ = make_tensor(make_gmem_ptr(static_cast<Element const*>(params.q_ptr)), shape_Q, stride_Q);
    Tensor mdO = make_tensor(make_gmem_ptr(static_cast<Element const*>(params.do_ptr)), shape_Q, stride_dO);
    Tensor mK = make_tensor(make_gmem_ptr(static_cast<Element const*>(params.k_ptr)), shape_K, stride_K);
    Tensor mV = make_tensor(make_gmem_ptr(static_cast<Element const*>(params.v_ptr)), shape_K, stride_V);
    typename Kernel::TMA_QdO tma_load_Q = make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{}, mQ, take<0, 2>(typename Kernel::SmemLayoutQ{}), typename Kernel::TileShape_MNK{}, typename Kernel::ClusterShape{});
    typename Kernel::TMA_QdO tma_load_dO = make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{}, mdO, take<0, 2>(typename Kernel::SmemLayoutdO{}), typename Kernel::TileShape_MNK{}, typename Kernel::ClusterShape{});
    typename Kernel::TMA_K tma_load_K = make_tma_copy_B_sm90(
        cute::SM90_TMA_LOAD{}, mK, typename Kernel::SmemLayoutK1{}, typename Kernel::TileShape_MNK{}, typename Kernel::ClusterShape{});
    typename Kernel::TMA_K tma_load_V = make_tma_copy_B_sm90(
        cute::SM90_TMA_LOAD{}, mV, typename Kernel::SmemLayoutK1{}, typename Kernel::TileShape_MNK{}, typename Kernel::ClusterShape{});

    typename flash::TileSchedulerArguments scheduler_args {
        ngroups, params.h, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        params.seqlen_k,
        params.seqlen_q, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_k, params.seqused_k
    };
    // ## FP11-M17 P0MS: the unit table (bh, group, m_lo, m_hi), full chunks first then remainders by
    // length descending; built once per shape and cached on the device (one entry per distinct
    // (b, h, seqlen_q, seqlen_k, causal, device)).
    int const* unit_tab = nullptr;
    int nunits = 0;
    if constexpr (FA3FP11_M17_P0MS > 0) {
        static std::map<std::tuple<int, int, int, int, bool, int>, std::pair<int*, int>> cache;   // device buffer (kept for the process lifetime), nunits
        static std::mutex mu;
        int dev = -1; cudaGetDevice(&dev);
        auto key = std::make_tuple(params.b, params.h, seqlen_q, seqlen_k, Is_causal, dev);
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it == cache.end()) {
            std::vector<std::array<int, 4>> full, part;
            for (int bh = 0; bh < params.b * params.h; ++bh) {
                for (int g = 0; g < ngroups; ++g) {
                    int const nb0 = Kernel::kNbPerCTA * g;
                    int m_min = 0;
                    if (Is_causal) { m_min = std::max(0, (nb0 * Kernel::kBlockN + seqlen_q - seqlen_k) / Kernel::kBlockM); }
                    for (int lo = m_min; lo < nbm; lo += FA3FP11_M17_P0MS) {
                        int const hi = std::min(nbm, lo + FA3FP11_M17_P0MS);
                        std::array<int, 4> e{bh, g, lo, hi};
                        if (hi - lo == FA3FP11_M17_P0MS) { full.push_back(e); } else { part.push_back(e); }
                    }
                }
            }
            std::stable_sort(part.begin(), part.end(), [](auto const& a, auto const& b) { return (a[3] - a[2]) > (b[3] - b[2]); });
            std::vector<int> flat;
            for (auto const& e : full) { flat.insert(flat.end(), e.begin(), e.end()); }
            for (auto const& e : part) { flat.insert(flat.end(), e.begin(), e.end()); }
            int* dbuf = nullptr;
            CHECK_CUDA(cudaMalloc(&dbuf, flat.size() * sizeof(int)));
            // stream-ordered and synchronous for this call (the host vector dies at the end of the scope)
            CHECK_CUDA(cudaMemcpyAsync(dbuf, flat.data(), flat.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            it = cache.emplace(key, std::make_pair(dbuf, int(full.size() + part.size()))).first;
        }
        unit_tab = it->second.first;
        nunits = it->second.second;
    }
    typename Kernel::Params kp {
        shape_Q, shape_K,
        cutlass::FastDivmod(params.h / params.h_k),
        tma_load_Q, tma_load_dO, tma_load_K, tma_load_V,
        params.q8s_ptr, params.k8s_ptr, params.v8s_ptr, params.do8s_ptr,
        params.nblk_q, params.nblk_k,
        static_cast<float const*>(params.softmax_lse_log2_ptr),
        make_shape(seqlen_q, params.h, params.b),
        make_stride(_1{}, int64_t(seqlen_q), int64_t(params.h) * seqlen_q),
        params.dlt_part_ptr,
        nbm, nbn, ngroups, params.b,
        Kernel::Scheduler::to_underlying_arguments(scheduler_args),
        unit_tab, nunits, params.h
    };
    dim3 grid_dims = Kernel::Scheduler::get_grid_shape(kp.scheduler, params.num_sm);
    if constexpr (FA3FP11_M17_P0MS > 0) { grid_dims = dim3(nunits, 1, 1); }
    dim3 block_dims(Kernel::MaxThreadsPerBlock, 1, 1);
    int smem_size = Kernel::SharedStorageSize;
    auto kernel = cutlass::device_kernel<Kernel>;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    if (std::getenv("DM_DEBUG")) {
        fprintf(stderr, "[delta_matching pass 0] smem_size=%d grid=(%d,%d,%d) block=%d atomN=%d nbwg=%d\n",
                smem_size, grid_dims.x, grid_dims.y, grid_dims.z, block_dims.x, FA3FP11_M16_P0V2_ATOMN, FA3FP11_M16_P0V3_NBWG);
    }
    kernel<<<grid_dims, block_dims, smem_size, stream>>>(kp);
    { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { fprintf(stderr, "[delta_matching pass 0] launch: %s\n", cudaGetErrorString(e_)); exit(1); } }

    // PASS 0.5: the M5 fixed-order reduce, unchanged.
    {
        dim3 grid(nbm, params.b * params.h, 1);
        fp11d::fp11_delta_reduce_kernel<<<grid, 64, 0, stream>>>(
            params.dlt_part_ptr, params.dlt_ptr, nbn, params.seqlen_q, Is_causal);
        cudaError_t e_ = cudaGetLastError();
        if (e_ != cudaSuccess) { fprintf(stderr, "[delta_matching pass 0.5] launch: %s\n", cudaGetErrorString(e_)); exit(1); }
    }
}

template <bool Is_causal>
void run_flash_delta_v4_fp11_d256(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    run_flash_delta_v4_fp11_impl<Is_causal, 256>(params, stream);
}
