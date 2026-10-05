/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/
// ## M21 PASS 0 "P0-D" (FA3FP11_HD128_P0V6=1, hd128 only; the certified build sets it): the row-CTA pass 0 of an earlier, unshipped variant (P0-F, "v5")
// with the INSTRUCTION DIET the M21 diagnosis asked for (the development notes: pass 0 is issue-bound at 62% with 394
// instructions per warp per 64 x 64 tile, 64 of them the causal mask on tiles that never need it, ~50 of them wgmma
// descriptor traffic through vector registers).  Per element the arithmetic is v5's VERBATIM (exp_step's exp2 line with the F1
// clamp, rowsum_step's fixed-order fmaf chains per 32-key half + Allreduce<4>, x 2^-8, the fp64 fold): delta is expected
// BITWISE.  What changes: (1) the causal mask is applied only to the n-blocks that can hold a masked element (keys past the
// row's diagonal: n_block * 64 + 63 > m_block * 64 + (seqlen_k - seqlen_q), or the seqlen_k tail), a warpgroup-uniform branch;
// (2) the warpgroup index is shfl-broadcast so that ptxas keeps the K / V slot descriptors in uniform registers (v5 derived it
// from threadIdx, which routed every wgmma descriptor through vector registers + R2UR).  v5 is not part of this repository.
// (v5 header follows)
// ## Delta-Matching-128 v2 (M20) PASS 0 "P0-F" (FA3FP11_HD128_P0V5=1, hd128 only, default 0): the in-kernel
// matched delta in the ROW-CTA structure.  A CTA owns 128 query rows (= one Q / dO scale block; warpgroup w owns the
// 64-row m-block 2p + w) with Q / dO / LSE resident, and streams K / V in 128-key stages (two 64-key n-blocks, one K/V
// scale group each) through a kStages-deep TMA pipeline.  Per n-block the pass-0 v4b arithmetic is VERBATIM (the same
// m64n64k32 atoms on the same Q / K and dO / V operand roles, exp_step with the F1 cliff clamp, rowsum_step's fixed-order
// fmaf chains per 32-key half + Allreduce<4>, x 2^-8) and the M5 pass-0.5 fixed-order fp64 fold (n-block ascending, half 0
// then half 1, one fp32 rounding at the store) runs IN REGISTERS: delta is expected BITWISE the v4b + reduce delta.  What
// disappears: the [BH, nbn, 2, N] partial workspace and its reduce launch, the per-n-block Q / dO re-streaming (each
// K / V byte is read once per 128-row CTA instead of each Q / dO byte once per 256-key CTA), and the m-split unit table.
// Causal: the CTA's stage range is [0, p + 1) at seqlen_q == seqlen_k (longest CTAs first); the mask is per 64 x 64 tile
// as in v4b.  Deterministic by construction (one writer per row, one fixed order).
#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_check.h"
#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "flash.h"
#include "seqlen.h"
#include "mask.h"
#include "utils.h"
#include "softmax.h"

#include "flash_bwd_fp8_params.h"

#ifndef FA3FP11_HD128_P0V5_ST
#define FA3FP11_HD128_P0V5_ST 3   // K/V pipeline depth (32 KB per stage)
#endif
// ## M20 P0-F ping-pong (FA3's forward warp-scheduler protocol on the two consumer warpgroups): a warpgroup issues its
// S / dP gemm pair only when it holds the turn (named barrier 1 + wg) and hands the turn to the other warpgroup right after
// issuing, so one warpgroup's exp2 / rowsum chain runs under the other's gemms instead of both contending for the tensor
// pipe at the same moments.  Schedule only (the per-tile arithmetic and the fold order are untouched: bitwise).
// MEASURED (2 processes): bitwise, but 0.91x / 0.90x / 0.88x of v4b at 12q / 16q / 32q (the strict
// alternation costs more barrier latency than the contention it removes): default OFF.
#ifndef FA3FP11_P0V5_PP
#define FA3FP11_P0V5_PP 0
#endif
// ## M21 P0V6_UNROLL3 (bitwise-class, default off): the K / V stage loop unrolled by the ring depth (3) so that the K / V slot
// indices of the wgmma descriptors are compile-time constants; the runtime pipeline states still drive the barriers.
#ifndef FA3FP11_P0V6_UNROLL3
#define FA3FP11_P0V6_UNROLL3 0
#endif
// ## M23 P0V7_LAZY (a NEW NUMERICS ARM, default off = the certified v6 delta byte for byte): the exp2 line and
// the per-element products are unchanged, the row sum's realization is cheaper: (1) the half's dP descale (dp_g) is
// applied once to the half's 8-key fp32 partial instead of to every element (one rounding per half instead of one per
// element); (2) the 4-lane Allreduce and the fp64 fold move from every tile to the loop exit: each thread carries its two
// rows' partials in fp64 across all n-blocks (the fp64 add of each fp32 half partial, as the per-tile fold did) and the
// quad's four lanes are folded in a fixed order (lane 0 + 1 + 2 + 3) once at the end.  Deterministic; fewer roundings per
// element than v6; NOT bitwise the v6 delta (not part of this release).
#ifndef FA3FP11_P0V7_LAZY
#define FA3FP11_P0V7_LAZY 0
#endif

namespace flash {

using namespace cute;

template <bool Is_causal>
struct DeltaV6 {
    using Element = cutlass::float_e4m3_t;
    using ElementAccum = float;
    static constexpr int kBlockM = 64;       // per warpgroup
    static constexpr int kRowsCTA = 128;     // two warpgroups = one Q/dO scale block
    static constexpr int kBlockN = 64;       // per n-block (one K/V scale group)
    static constexpr int kNbPerStage = 2;    // 128 keys per stage
    static constexpr int kHeadDim = 128;
    static constexpr int kStages = FA3FP11_HD128_P0V5_ST;
    static constexpr int kAtomN = 64;
    static constexpr int NumMmaWarpGroups = 2;
    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int MaxThreadsPerBlock = NumMmaThreads + NumProducerThreads;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr uint32_t LoadRegisterRequirement = 40;
    static constexpr uint32_t MmaRegisterRequirement = 232;

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using SeqlenInfo_t = flash::SeqlenInfoQK<false, kBlockM>;

    // The S / dP tiled MMA: v4b's (one warpgroup, atom m64 x 64 x k32, C tile 64x64).
    using TileShapeAtomS = Shape<Int<kBlockM>, Int<kAtomN>, Int<kHeadDim>>;
    using TiledMmaS = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomS>(),
        Layout<Shape<_1, _1, _1>>{}));

    // Smem layouts: v4b's atoms (the same TMA boxes and swizzles the certified pass 0 stages through).
    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDim / 2>>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<NumMmaWarpGroups>{})));    // one m-block per warpgroup
    using SmemLayoutdO = SmemLayoutQ;
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim / 2>>());
    using SmemLayoutK1 = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));   // one n-block
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kNbPerStage * kStages>{})));   // slot = kNbPerStage * stage + j
    using SmemLayoutV = SmemLayoutK;
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<NumMmaWarpGroups>>, cute::Stride<_1, Int<kBlockM>>>;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;

    using TMA_QdO = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}), TileShape_MNK{}, ClusterShape{}));
    using TMA_K = decltype(make_tma_copy_B_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK1{}, TileShape_MNK{}, ClusterShape{}));

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;

    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK1 = static_cast<uint32_t>(size(SmemLayoutK1{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);
    static constexpr uint32_t TmaTransactionBytesQdO = NumMmaWarpGroups * (2 * TmaTransactionBytesQ + TmaTransactionBytesLSE);   // Q + dO + LSE, two m-blocks
    static constexpr uint32_t TmaTransactionBytesStage = 2 * kNbPerStage * TmaTransactionBytesK1;   // K + V, two n-blocks

    struct TensorStorage : cute::aligned_struct<256> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 256> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, 256> smem_do;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 256> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, 256> smem_v;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
    };
    struct SharedStorage {
        TensorStorage tensors;
        struct PipelineStorage : cute::aligned_struct<16> {
            alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_QdO;
            alignas(16) typename MainloopPipeline::SharedStorage pipeline_kv;
        } pipelines;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    static_assert(SharedStorageSize <= 232448, "DeltaV6: shared storage exceeds the sm90 227 KB limit");

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
        float* ptr_dlt;          // [B, H, seqlen_q] fp32
        int nbm, nbn, npairs;
        int num_batch, num_heads;
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
            shared_storage.pipelines.barrier_QdO.init(1 /*numThreads*/);
        }
        MainloopPipeline pipeline_kv(shared_storage.pipelines.pipeline_kv, pipeline_params, ClusterShape{});
        __syncthreads();

        // CTA = (m-pair p, bh); longest first for the causal tail (as pass 2).
        int const p = params.npairs - 1 - int(blockIdx.x);
        int const bh = int(blockIdx.y);
        int const bidb = bh / params.num_heads;
        int const bidh = bh - bidb * params.num_heads;
        int const bidh_kv = params.qhead_per_khead_divmod.divide(bidh);
        int const seqlen_q = get<0>(params.shape_Q);
        int const seqlen_k = get<0>(params.shape_K);
        // Keys this CTA's rows attend to: causal -> up to the last row's diagonal (seqlen_q == seqlen_k is the certified
        // configuration; the general FA3 offset seqlen_k - seqlen_q is kept), else all.
        int n_block_max = params.nbn;
        constexpr int kbn = kBlockN, knb = kNbPerStage, krows = kRowsCTA;   // local copies: no ODR-use of the static members in device code
        if constexpr (Is_causal) {
            int const key_end = (p + 1) * krows + seqlen_k - seqlen_q;
            n_block_max = key_end <= 0 ? 0 : (key_end < seqlen_k ? (key_end + kbn - 1) / kbn : params.nbn);
        }
        int const nstages = (n_block_max + knb - 1) / knb;
        if (nstages <= 0) {
            // Every row of this CTA sees no key: delta = 0 (fully masked rows have zero P; the M5 reduce wrote 0.0 too).
            if (warp_group_idx > 0 && warp_group_thread_idx < kBlockM) {
                int const row = p * kRowsCTA + (warp_group_idx - 1) * kBlockM + warp_group_thread_idx;
                if (row < seqlen_q) { params.ptr_dlt[int64_t(bh) * seqlen_q + row] = 0.0f; }
            }
            return;
        }

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
                // Q / dO / LSE: the CTA's two m-blocks, one-shot on barrier_QdO.
                auto& barQ = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_QdO);
                shared_storage.pipelines.barrier_QdO.arrive_and_expect_tx(TmaTransactionBytesQdO);
                #pragma unroll
                for (int w = 0; w < NumMmaWarpGroups; ++w) {
                    int const m_block = kRowsCTA / kBlockM * p + w;
                    copy(params.tma_load_Q.with(barQ, 0), tQgQ(_, m_block), tQsQ(_, w));
                    copy(params.tma_load_dO.with(barQ, 0), tdOgdO(_, m_block), tdOsdO(_, w));
                    copy(bulk_copy.with(barQ), gLSE(_, m_block), sLSE(_, w));
                }
                #pragma unroll 1
                for (int s = 0; s < nstages; ++s) {
                    pipeline_kv.producer_acquire(smem_pipe_write);
                    auto& bar = *pipeline_kv.producer_get_barrier(smem_pipe_write);
                    #pragma unroll
                    for (int j = 0; j < kNbPerStage; ++j) {
                        int const nb = kNbPerStage * s + j;
                        int const nbc = nb < params.nbn ? nb : params.nbn - 1;   // clamped (never consumed when out of range)
                        int const slot = kNbPerStage * smem_pipe_write.index() + j;
                        Tensor gKj = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(nbc, _0{}));
                        Tensor gVj = local_tile(mV, select<1, 2>(TileShape_MNK{}), make_coord(nbc, _0{}));
                        auto [tKgK, tKsK] = kv_part(gKj, sK(_, _, slot), params.tma_load_K);
                        auto [tVgV, tVsV] = kv_part(gVj, sV(_, _, slot), params.tma_load_V);
                        copy(params.tma_load_K.with(bar, 0, TMA::CacheHintSm90::EVICT_LAST), tKgK, tKsK);
                        copy(params.tma_load_V.with(bar, 0, TMA::CacheHintSm90::EVICT_LAST), tVgV, tVsV);
                    }
                    ++smem_pipe_write;
                }
                pipeline_kv.producer_tail(smem_pipe_write);
            }
        } else {  // ---------------- consumer warpgroups ----------------
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
            int const thread_idx = threadIdx.x - NumProducerThreads;   // 0..255
            int const wg = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);   // ## M21: warp-uniform for ptxas (uniform descriptors)
            int const wtid = thread_idx % cutlass::NumThreadsPerWarpGroup;
            int const m_block = kRowsCTA / kBlockM * p + wg;

            TiledMmaS tiled_mma;
            auto wg_mma = tiled_mma.get_slice(_0{});
            auto thread_mma = tiled_mma.get_thread_slice(wtid);
            Tensor tSrQ = wg_mma.partition_fragment_A(sQ);                 // (MMA, MMA_M, MMA_K, wg)
            Tensor tSrK = wg_mma.partition_fragment_B(sK);                 // (MMA, MMA_N, MMA_K, slots)
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

            int const fq8_bh_kv = bidb * get<2>(params.shape_K) + bidh_kv;
            float const* fq8_k8s_base = params.ptr_k8s + fq8_bh_kv * params.nblk_k;
            float const* fq8_v8s_base = params.ptr_v8s + fq8_bh_kv * params.nblk_k;
            float const fq8_q8sc = params.ptr_q8s[bh * params.nblk_q + p];      // one Q/dO scale block per CTA (128 rows)
            float const fq8_do8sc = params.ptr_do8s[bh * params.nblk_q + p];
            static constexpr float kFq8MoE = 8.0f;
            static constexpr float kFq8MoInv = 1.0f / 256.0f;

            cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_QdO.try_wait(0));
            if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.pipelines.barrier_QdO.wait(0); }
            float const lse0 = __fmul_rn(sLSE(row0, wg), float(M_LOG2E));   // the host prep's exact multiply
            float const lse1 = __fmul_rn(sLSE(row1, wg), float(M_LOG2E));
            // The v4b per-tile arithmetic, VERBATIM: (1) S~ -> P~ (mask, descale, LSE, F1 clamp, exp2), (2) the fixed-order
            // half row sums; (3) the M5 reduce's fp64 fold (n-block ascending, half 0 then half 1) in registers.
            // ## M21 diet: the mask is a no-op on every n-block whose keys all lie at or below the row block's diagonal and
            // inside seqlen_k; skipping it there is bitwise (no element would have been set to -inf).  Warpgroup-uniform.
            int const fq8_diag_off = seqlen_k - seqlen_q;
            auto exp_step = [&](auto& tSrS, int n_block, float dsc_g0, float dsc_g1) {
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
                static_assert(decltype(size<0>(scores))::value == 2 && decltype(size<1>(scores))::value == 16,
                              "DeltaV6: expected 2 rows x 16 columns per thread for a 64x64 tile");
                bool const need_mask = (Is_causal && (n_block + 1) * kBlockN > m_block * kBlockM + fq8_diag_off)
                                       || ((n_block + 1) * kBlockN > seqlen_k);
                if (need_mask) {
                    mask.template apply<true /*Seqlenk_mask*/, Is_causal, false /*Is_local*/>(tSrS, m_block, n_block);
                }
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    float const lse_scaled = mi == 0 ? lse0 : lse1;
                    #pragma unroll
                    for (int ni = 0; ni < 16; ++ni) {
                        float const dsc = (ni >> 3) == 0 ? dsc_g0 : dsc_g1;
                        scores(mi, ni) = exp2f(fminf(scores(mi, ni) * dsc - lse_scaled + kFq8MoE, kFq8MoE + 4.0f));
                    }
                }
            };
            flash::SumOp<float> fq8_sum_op;
            double acc0 = 0.0, acc1 = 0.0;
            auto rowsum_step = [&](auto& tSrS, auto& tdPrdP, float dp_g0, float dp_g1) {
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
                Tensor dPv = make_tensor(tdPrdP.data(), scores.layout());
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    float a0 = 0.f, a1 = 0.f;
                    #pragma unroll
                    for (int ni = 0; ni < 8; ++ni) { a0 = fmaf(scores(mi, ni), dPv(mi, ni) * dp_g0, a0); }
                    #pragma unroll
                    for (int ni = 8; ni < 16; ++ni) { a1 = fmaf(scores(mi, ni), dPv(mi, ni) * dp_g1, a1); }
                    a0 = flash::Allreduce<4>::run(a0, fq8_sum_op);
                    a1 = flash::Allreduce<4>::run(a1, fq8_sum_op);
                    // the pass-0.5 fold: part(j, half 0) then part(j, half 1), each a fp32 value added into fp64
                    float const p0 = a0 * kFq8MoInv;
                    float const p1 = a1 * kFq8MoInv;
                    if (mi == 0) { acc0 += double(p0); acc0 += double(p1); } else { acc1 += double(p0); acc1 += double(p1); }
                }
            };

            // Warp-scheduler ping-pong: WG0 takes the first turn (WG1 pre-arrives on WG0's barrier).
            auto sched_sync = [&]() {
            };
            auto sched_arrive = [&]() {
            };
            PipelineState smem_pipe_read;
            Tensor tSrS0 = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            Tensor tdPrdP0 = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            Tensor tSrS1 = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            Tensor tdPrdP1 = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            // Per-n-block scales: this stage's two groups (prefetched one stage ahead).
            float k8s_n0 = fq8_k8s_base[0], k8s_n1 = fq8_k8s_base[1 < params.nbn ? 1 : 0];
            float v8s_n0 = fq8_v8s_base[0], v8s_n1 = fq8_v8s_base[1 < params.nbn ? 1 : 0];
            // Prologue: the first stage's (S0, dP0) in flight before the loop (v4b).
            {
                auto tok = pipeline_kv.consumer_try_wait(smem_pipe_read);
                pipeline_kv.consumer_wait(smem_pipe_read, tok);
                int const slot0 = kNbPerStage * smem_pipe_read.index();
                sched_sync();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tSrQ(_, _, _, wg), tSrK(_, _, _, slot0), tSrS0);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tdPrdO(_, _, _, wg), tdPrV(_, _, _, slot0), tdPrdP0);
                sched_arrive();
                warpgroup_wait<0>();
                warpgroup_fence_operand(tSrS0);
                warpgroup_fence_operand(tdPrdP0);
            }
            // The v4b unit body on one stage (two n-blocks), HasNext compile-time so every wait count is straight-line.
            auto step = [&](int s, auto has_next_c, auto stage_c) {
                constexpr bool HasNext = decltype(has_next_c)::value;
                constexpr int kSt = decltype(stage_c)::value;   // compile-time ring stage under UNROLL3 (-1 = runtime)
                int const slot0 = kSt >= 0 ? kNbPerStage * kSt : kNbPerStage * smem_pipe_read.index();
                int const slot1 = slot0 + 1;
                int const nb0 = kNbPerStage * s, nb1 = nb0 + 1;
                float const dsc0 = fq8_q8sc * k8s_n0 * float(M_LOG2E);
                float const dsc1 = fq8_q8sc * k8s_n1 * float(M_LOG2E);
                float const dp0 = fq8_do8sc * v8s_n0;
                float const dp1 = fq8_do8sc * v8s_n1;
                if constexpr (HasNext) {
                    int const nn0 = nb0 + kNbPerStage, nn1 = nn0 + 1;
                    k8s_n0 = fq8_k8s_base[nn0 < params.nbn ? nn0 : 0]; k8s_n1 = fq8_k8s_base[nn1 < params.nbn ? nn1 : 0];
                    v8s_n0 = fq8_v8s_base[nn0 < params.nbn ? nn0 : 0]; v8s_n1 = fq8_v8s_base[nn1 < params.nbn ? nn1 : 0];
                }
                sched_sync();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tSrQ(_, _, _, wg), tSrK(_, _, _, slot1), tSrS1);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tdPrdO(_, _, _, wg), tdPrV(_, _, _, slot1), tdPrdP1);
                sched_arrive();
                warpgroup_wait<2>();   // no-op guard: S0 already retired
                warpgroup_fence_operand(tSrS0);
                exp_step(tSrS0, nb0, dsc0, dsc0);
                warpgroup_fence_operand(tdPrdP0);
                rowsum_step(tSrS0, tdPrdP0, dp0, dp0);
                PipelineState smem_pipe_read_next = smem_pipe_read;
                ++smem_pipe_read_next;
                if constexpr (HasNext) {
                    auto tok = pipeline_kv.consumer_try_wait(smem_pipe_read_next);
                    pipeline_kv.consumer_wait(smem_pipe_read_next, tok);
                    int const nslot0 = kSt >= 0 ? kNbPerStage * ((kSt + 1) % kStages) : kNbPerStage * smem_pipe_read_next.index();
                    sched_sync();
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tSrQ(_, _, _, wg), tSrK(_, _, _, nslot0), tSrS0);
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma, tdPrdO(_, _, _, wg), tdPrV(_, _, _, nslot0), tdPrdP0);
                    sched_arrive();
                    warpgroup_wait<3>();   // S1 retired
                } else {
                    warpgroup_wait<1>();
                }
                warpgroup_fence_operand(tSrS1);
                exp_step(tSrS1, nb1, dsc1, dsc1);
                if constexpr (HasNext) { warpgroup_wait<2>(); } else { warpgroup_wait<0>(); }   // dP1 retired: every read of this stage is done
                warpgroup_fence_operand(tdPrdP1);
                pipeline_kv.consumer_release(smem_pipe_read);
                smem_pipe_read = smem_pipe_read_next;
                rowsum_step(tSrS1, tdPrdP1, dp1, dp1);
                warpgroup_wait<0>();   // the next stage's (S0, dP0) retired before the back-edge (ptxas C7514 discipline)
                warpgroup_fence_operand(tSrS0);
                warpgroup_fence_operand(tdPrdP0);
            };
            CUTLASS_PRAGMA_NO_UNROLL
            for (int s = 0; s < nstages - 1; ++s) { step(s, cute::true_type{}, cute::Int<-1>{}); }
            step(nstages - 1, cute::false_type{}, cute::Int<-1>{});
            // One writer per row: the quad leader stores the fp64 fold rounded once to fp32.
            if ((wtid & 3) == 0) {
                int const r0 = m_block * kBlockM + row0, r1 = m_block * kBlockM + row1;
                if (r0 < seqlen_q) { params.ptr_dlt[int64_t(bh) * seqlen_q + r0] = float(acc0); }
                if (r1 < seqlen_q) { params.ptr_dlt[int64_t(bh) * seqlen_q + r1] = float(acc1); }
            }
        }
    }
};

}  // namespace flash

template <bool Is_causal>
void run_flash_delta_v6_fp11_hd128(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    using Kernel = flash::DeltaV6<Is_causal>;
    using Element = typename Kernel::Element;
    using namespace cute;
    int const seqlen_q = params.seqlen_q;
    int const seqlen_k = params.seqlen_k;
    if (seqlen_q % Kernel::kRowsCTA != 0 || seqlen_k % (Kernel::kNbPerStage * Kernel::kBlockN) != 0) {
        fprintf(stderr, "[delta_matching pass 0] seqlen_q and seqlen_k must be 128-multiples (got %d, %d)\n", seqlen_q, seqlen_k); exit(1);
    }
    int const nbm = seqlen_q / Kernel::kBlockM;
    int const nbn = seqlen_k / Kernel::kBlockN;
    int const npairs = seqlen_q / Kernel::kRowsCTA;
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
    typename Kernel::Params kp {
        shape_Q, shape_K,
        cutlass::FastDivmod(params.h / params.h_k),
        tma_load_Q, tma_load_dO, tma_load_K, tma_load_V,
        params.q8s_ptr, params.k8s_ptr, params.v8s_ptr, params.do8s_ptr,
        params.nblk_q, params.nblk_k,
        static_cast<float const*>(params.softmax_lse_log2_ptr),
        make_shape(seqlen_q, params.h, params.b),
        make_stride(_1{}, int64_t(seqlen_q), int64_t(params.h) * seqlen_q),
        params.dlt_ptr,
        nbm, nbn, npairs, params.b, params.h
    };
    dim3 grid_dims(npairs, params.b * params.h, 1);
    dim3 block_dims(Kernel::MaxThreadsPerBlock, 1, 1);
    int smem_size = Kernel::SharedStorageSize;
    auto kernel = cutlass::device_kernel<Kernel>;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    if (std::getenv("DM_DEBUG")) {
        fprintf(stderr, "[delta_matching pass 0] smem_size=%d grid=(%d,%d,%d) block=%d stages=%d\n",
                smem_size, grid_dims.x, grid_dims.y, grid_dims.z, block_dims.x, Kernel::kStages);
    }
    kernel<<<grid_dims, block_dims, smem_size, stream>>>(kp);
    { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { fprintf(stderr, "[delta_matching pass 0] launch: %s\n", cudaGetErrorString(e_)); exit(1); } }
    // No pass 0.5: the fold ran in registers.
}
