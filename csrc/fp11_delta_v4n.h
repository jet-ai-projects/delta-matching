// ## KDA-P0 N128 (head_dim 256 pass 0, FA3FP11_D256_P0N128=1): DeltaV4's CTA (256 keys resident, Q / dO / LSE
// streamed in a 3-deep TMA pipeline, two consumer warpgroups of two 64-key n-blocks each), with each warpgroup's two
// n-blocks computed by ONE m64 n128 k32 wgmma chain per operand pair (S = Q K^T and dP = dO V^T over 128 keys) instead
// of two m64 n64 chains.
//
// WHY: the SS wgmma reads its A operand (the Q / dO stage) from shared memory once per chain; at n64 the A bytes equal
// the B bytes, and the four n64 chains per warpgroup per stage read ~141 B/clk at the tensor peak (> the 128 B/clk
// shared-memory port).  One n128 chain per operand reads A once per 128 keys: ~25% fewer shared-memory bytes per MAC.
//
// WHAT IS IDENTICAL: every output element of the n128 chain is the same k-ordered (zero-init, 8 x k32) fp32
// accumulation of the same fp8 operand bytes as the n64 chain (the tensor core's per-element datapath does not depend
// on the instruction's N); the accumulator registers of key columns [0, 64) / [64, 128) are exactly the two n64
// fragments (CLayout_64xN: value = col-pair + 2 row-half + 4 n-group), so exp_step / rowsum_step run VERBATIM on the
// two 32-register halves, with the same mask, descale, exp2 line, fixed-order fmaf chains, Allreduce<4> and
// [BH, nbn, 2, N] slots; the reduce is unchanged.  K / V smem: per warpgroup one (128 keys, 256) K-major SW128 tile
// (k-half outermost), each n-block TMA'd as two (64 keys, 128) boxes into it (same bytes, same transaction count).
#pragma once

#ifndef FA3FP11_D256_P0N128
#define FA3FP11_D256_P0N128 1
#endif

namespace flash {

using namespace cute;

template <bool Is_causal>
struct DeltaV4N {
    using Element = cutlass::float_e4m3_t;
    using ElementAccum = float;
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 64;       // per n-block
    static constexpr int kHeadDim = 256;
    static constexpr int kScalesPerNb = 2;   // two 32-key K/V scale groups per n-block at d256
    static constexpr int kQShift = 0;
    static constexpr int kStages = 3;
    static constexpr int NumMmaWarpGroups = 2;
    static constexpr int kNbPerWG = 2;
    static constexpr int kNbPerCTA = NumMmaWarpGroups * kNbPerWG;
    static constexpr int kGroupN = kNbPerCTA * kBlockN;              // keys per CTA
    static constexpr int kWGN = kNbPerWG * kBlockN;                  // keys per warpgroup (one wgmma N)
    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int MaxThreadsPerBlock = NumMmaThreads + NumProducerThreads;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr uint32_t LoadRegisterRequirement = 40;
    static constexpr uint32_t MmaRegisterRequirement = 232;

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShapeKTma = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim / 2>>;   // B box: (64 keys, 128 dims)
    using ClusterShape = Shape<_1, _1, _1>;
    using SeqlenInfo_t = flash::SeqlenInfoQK<false, kBlockM>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, false>;

    // n64 tiled MMA: only for the per-n-block fragment layout (mask / views), never issued.
    using TiledMmaS = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>>(),
        Layout<Shape<_1, _1, _1>>{}));
    // the issued MMA: m64 n128 k32
    using TiledMmaS2 = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kWGN>, Int<kHeadDim>>>(),
        Layout<Shape<_1, _1, _1>>{}));

    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDim / 2>>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages>{})));
    using SmemLayoutdO = SmemLayoutQ;
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim / 2>>());
    using SmemLayoutKTma = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim / 2>{})));   // one TMA box
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(Int<kWGN>{}, Int<kHeadDim>{}, Int<NumMmaWarpGroups>{})));      // per warpgroup (128, 256)
    using SmemLayoutV = SmemLayoutK;
    static constexpr int kBoxBytes = kBlockN * (kHeadDim / 2);               // 8 KB
    static constexpr int kHalfBytes = kWGN * (kHeadDim / 2);                 // 16 KB: k-half stride in a warpgroup tile
    static constexpr int kWGBytes = kWGN * kHeadDim;                         // 32 KB
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<kBlockM>>>;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
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
        SmemLayoutKTma{}, TileShapeKTma{}, ClusterShape{}));

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using Scheduler = flash::SingleTileBwdLPTScheduler<false /*Varlen*/, kGroupN, false /*SPT*/>;

    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK1 = 2 * kBoxBytes;     // one n-block (two boxes)
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);
    static constexpr uint32_t TmaTransactionBytesStage = 2 * TmaTransactionBytesQ + TmaTransactionBytesLSE;

    struct TensorStorage : cute::aligned_struct<1024> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 1024> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, 1024> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 1024> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, 1024> smem_do;
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
    static_assert(SharedStorageSize <= 232448, "DeltaV4N: shared storage exceeds the sm90 227 KB limit");
    static_assert(cute::cosize_v<SmemLayoutK> == NumMmaWarpGroups * kWGBytes, "DeltaV4N: K tile size");

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
        int const* unit_tab;
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
        MainloopPipeline pipeline_qdo(shared_storage.pipelines.pipeline_qdo, pipeline_params, ClusterShape{});
        __syncthreads();

        int group, bidh, bidb, m_block_min, m_block_max;
        if constexpr (FA3FP11_M17_P0MS > 0) {
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
            Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
            Tensor gdO = local_tile(mdO, select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
            Tensor gLSE = local_tile(mLSE, select<0>(TileShape_MNK{}), make_coord(_));

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
                auto& barKV = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV);
                shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(2 * kNbPerCTA * TmaTransactionBytesK1);
                #pragma unroll
                for (int j = 0; j < kNbPerCTA; ++j) {
                    int const nb = nb0 + j < params.nbn ? nb0 + j : nb0;
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        // n-block j = (warpgroup j / 2, key half j % 2): rows [64 (j % 2), +64) of that warpgroup's tile
                        int const off = (j / kNbPerWG) * kWGBytes + h * kHalfBytes + (j % kNbPerWG) * kBoxBytes;
                        Tensor sKb = make_tensor(make_smem_ptr(shared_storage.tensors.smem_k.data() + off), SmemLayoutKTma{});
                        Tensor sVb = make_tensor(make_smem_ptr(shared_storage.tensors.smem_v.data() + off), SmemLayoutKTma{});
                        Tensor gKb = local_tile(mK, select<1, 2>(TileShapeKTma{}), make_coord(nb, h));
                        Tensor gVb = local_tile(mV, select<1, 2>(TileShapeKTma{}), make_coord(nb, h));
                        auto [tKgK, tKsK] = kv_part(gKb, sKb, params.tma_load_K);
                        auto [tVgV, tVsV] = kv_part(gVb, sVb, params.tma_load_V);
                        copy(params.tma_load_K.with(barKV, 0), tKgK, tKsK);
                        copy(params.tma_load_V.with(barKV, 0), tVgV, tVsV);
                    }
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
            int const thread_idx = threadIdx.x - NumProducerThreads;
            int const wg = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
            int const wtid = thread_idx % cutlass::NumThreadsPerWarpGroup;
            int const nb_base = nb0 + kNbPerWG * wg;
            int const seqlen_q = seqlen_info.seqlen_q;
            int const seqlen_k = seqlen_info.seqlen_k;

            TiledMmaS tiled_mma;      // n64: fragment layout + mask only
            TiledMmaS2 tiled_mma2;    // n128: issued
            auto wg_mma2 = tiled_mma2.get_slice(_0{});
            auto thread_mma = tiled_mma.get_thread_slice(wtid);
            Tensor tSrQ = wg_mma2.partition_fragment_A(sQ);                // (MMA, MMA_M, MMA_K, stages)
            Tensor tSrK = wg_mma2.partition_fragment_B(sK);                // (MMA, MMA_N, MMA_K, warpgroups)
            Tensor tdPrdO = wg_mma2.partition_fragment_A(sdO);
            Tensor tdPrV = wg_mma2.partition_fragment_B(sV);
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
                int const nbc = nb_valid[j] ? n_block : nb0;
                fq8_k8s_g0[j] = params.ptr_k8s[fq8_bh_kv * params.nblk_k + kScalesPerNb * nbc];
                fq8_k8s_g1[j] = params.ptr_k8s[fq8_bh_kv * params.nblk_k + kScalesPerNb * nbc + (kScalesPerNb - 1)];
                fq8_v8s_g0[j] = params.ptr_v8s[fq8_bh_kv * params.nblk_k + kScalesPerNb * nbc];
                fq8_v8s_g1[j] = params.ptr_v8s[fq8_bh_kv * params.nblk_k + kScalesPerNb * nbc + (kScalesPerNb - 1)];
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

            // DeltaV4's exp_step / rowsum_step, verbatim.
            // KDA-P0B MSPLIT: the tile's mask predicate (DeltaV4's, verbatim).  It is non-increasing in m_block (the causal
            // term falls as m_block grows, the seqlen_k tail term does not depend on m_block), so the m loop below is split
            // at the first m-block where it is false for BOTH of the warpgroup's n-blocks: the leading (diagonal) part runs
            // the masked body (per-n-block predicate unchanged), the rest a body with no mask code at all (bitwise: there
            // the predicate is false, so the masked body would have skipped the mask and the floor too).
            auto need_mask_fn = [&](int m_block, int n_block) {
                return (Is_causal && (n_block + 1) * kBlockN > m_block * kBlockM + (seqlen_k - seqlen_q))
                       || ((n_block + 1) * kBlockN > seqlen_k);
            };
            auto exp_step = [&](auto& tSrS, int m_block, int n_block, float lse0, float lse1, float dsc_g0, float dsc_g1, auto masked_c) {
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
                static_assert(decltype(size<0>(scores))::value == 2 && decltype(size<1>(scores))::value == 16,
                              "DeltaV4N: expected 2 rows x 16 columns per thread for a 64x64 tile");
                if constexpr (decltype(masked_c)::value) {
                    if (need_mask_fn(m_block, n_block)) {
                        mask.template apply<true /*Seqlenk_mask*/, Is_causal, false /*Is_local*/>(tSrS, m_block, n_block);
                        dsc_g0 = fmaxf(dsc_g0, 1.17549435e-38f); dsc_g1 = fmaxf(dsc_g1, 1.17549435e-38f);
                    }
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
            auto rowsum_step = [&](auto& tSrS, auto& tdPrdP, int m_block, int j, float dp_g0, float dp_g1) {
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol(tSrS.layout()));
                Tensor dPv = make_tensor(tdPrdP.data(), scores.layout());
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
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

            PipelineState smem_pipe_read;
            Tensor tSrS = partition_fragment_C(tiled_mma2, Shape<Int<kBlockM>, Int<kWGN>>{});     // 64 x 128
            Tensor tdPrdP = partition_fragment_C(tiled_mma2, Shape<Int<kBlockM>, Int<kWGN>>{});
            auto lay64 = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{})).layout();   // one n-block's fragment
            static_assert(decltype(size(lay64))::value * 2 == decltype(size(tSrS))::value, "DeltaV4N: n128 fragment = two n64 fragments");
            Tensor tSrS_j0 = make_tensor(tSrS.data(), lay64);
            Tensor tSrS_j1 = make_tensor(tSrS.data() + decltype(size(lay64))::value, lay64);
            Tensor tdPrdP_j0 = make_tensor(tdPrdP.data(), lay64);
            Tensor tdPrdP_j1 = make_tensor(tdPrdP.data() + decltype(size(lay64))::value, lay64);

            auto body = [&](int m_block, auto masked_c) {
                auto tok = pipeline_qdo.consumer_try_wait(smem_pipe_read);
                pipeline_qdo.consumer_wait(smem_pipe_read, tok);
                int const stage = smem_pipe_read.index();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma2, tSrQ(_, _, _, stage), tSrK(_, _, _, wg), tSrS);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma2, tdPrdO(_, _, _, stage), tdPrV(_, _, _, wg), tdPrdP);
                float const lse0 = __fmul_rn(sLSE(row0, stage), float(M_LOG2E));
                float const lse1 = __fmul_rn(sLSE(row1, stage), float(M_LOG2E));
                float const fq8_q8sc = fq8_q8s_base[m_block >> kQShift];
                float const fq8_do8sc = fq8_do8s_base[m_block >> kQShift];
                warpgroup_wait<1>();   // S retired
                warpgroup_fence_operand(tSrS);
                exp_step(tSrS_j0, m_block, nb_base + 0, lse0, lse1,
                         fq8_q8sc * fq8_k8s_g0[0] * float(M_LOG2E), fq8_q8sc * fq8_k8s_g1[0] * float(M_LOG2E), masked_c);
                exp_step(tSrS_j1, m_block, nb_base + 1, lse0, lse1,
                         fq8_q8sc * fq8_k8s_g0[1] * float(M_LOG2E), fq8_q8sc * fq8_k8s_g1[1] * float(M_LOG2E), masked_c);
                warpgroup_wait<0>();   // dP retired: every read of the stage is done
                warpgroup_fence_operand(tdPrdP);
                pipeline_qdo.consumer_release(smem_pipe_read);
                ++smem_pipe_read;
                rowsum_step(tSrS_j0, tdPrdP_j0, m_block, 0, fq8_do8sc * fq8_v8s_g0[0], fq8_do8sc * fq8_v8s_g1[0]);
                rowsum_step(tSrS_j1, tdPrdP_j1, m_block, 1, fq8_do8sc * fq8_v8s_g0[1], fq8_do8sc * fq8_v8s_g1[1]);
                warpgroup_fence_operand(tSrS);
                warpgroup_fence_operand(tdPrdP);
            };
            int m_block = m_block_min;
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < m_block_max && (need_mask_fn(m_block, nb_base) || need_mask_fn(m_block, nb_base + 1)); ++m_block) {
                body(m_block, cute::true_type{});
            }
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < m_block_max; ++m_block) {
                body(m_block, cute::false_type{});
            }
        }
    }
};

}  // namespace flash
