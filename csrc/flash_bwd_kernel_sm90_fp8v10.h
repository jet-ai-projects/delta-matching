
/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once
// ## FQ8-V9 kernel: adds the transposed-operand pipelines (PipelineAsync)
// and the transpose warps (2-3 of the load WG).  Forked from FA3's
// flash_bwd_kernel_sm90.h.

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "utils.h"


namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnBwdSm90Fp8V10 {

public:

    // Type Aliases
    static constexpr bool Is_causal = CollectiveMainloop_::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop_::Is_local;
    static_assert(CollectiveMainloop_::Varlen == CollectiveEpilogue_::Varlen);
    static constexpr bool Varlen = CollectiveMainloop_::Varlen;
    // ## FP11-M5: pass-0 (in-kernel matched delta) mode, taken from the
    // collective.  In pass-0 mode: no epilogue storage/calls (no dK/dV), no
    // transpose warps, no store_dq loop; the consumer runs mma_pass0.
    static constexpr bool DeltaPass0 = CollectiveMainloop_::DeltaPass0;

    // Mainloop derived types
    using CollectiveMainloop = CollectiveMainloop_;
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMmaSdP = typename CollectiveMainloop::TiledMmaSdP;
    using TiledMmadKV = typename CollectiveMainloop::TiledMmadKV;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using ClusterShape = typename CollectiveMainloop::ClusterShape;
    using MainloopPipelineQt = typename CollectiveMainloop::MainloopPipelineQt;
    using PipelineStateQt = typename CollectiveMainloop::PipelineStateQt;
    using MainloopPipelineDOt = typename CollectiveMainloop::MainloopPipelineDOt;
    using PipelineStateDOt = typename CollectiveMainloop::PipelineStateDOt;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;
    static constexpr bool dKV_swapAB = CollectiveMainloop::dKV_swapAB;

    // Epilogue derived types
    using CollectiveEpilogue = CollectiveEpilogue_;
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaSdP{})) / cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaSdP{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    /// Register requirement for Load and Math WGs
    // ## FQ8-V9: the transpose warps live in the load WG; 24 regs starves the
    // LDSM fragments + quad-shuffle temps + two-round addressing into local
    // spills.  40/232 fits the file: 128*(40 + 2*232) = 64512 <= 65536.
    // ## FQ8-V9 register split, MEASURED repeatedly: 24 spills the transposer
    // to 31.7ms; 32 starves it into a crawl even at ILP=1; 40/232 is the point.
    static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 2 ? 40 : 32;
    static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 232 : 160;
    // If you want to print from the producer warp, you'd need to increase the number of registers
    // Otherwise you'll get CUDA error.
    // static constexpr uint32_t LoadRegisterRequirement = 40;
    // static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 232 : 152;

    // Kernel level shared memory storage
    // ## FP11-M5: pass-1 keeps the M4 union layout verbatim; pass-0 has no
    // epilogue, so its tensors member is the (much smaller) mainloop storage
    // alone: sizeof(SharedStorage) drops to ~100 KB and pass 0 runs 2 CTAs/SM.
    struct TensorStorageFull : cute::aligned_struct<128> {
        union {
            typename CollectiveMainloop::TensorStorage mainloop;
            typename CollectiveEpilogue::TensorStorage epilogue;
        };
    };
    struct TensorStorageDelta : cute::aligned_struct<128> {
        typename CollectiveMainloop::TensorStorage mainloop;
    };
    struct SharedStorage {
        std::conditional_t<DeltaPass0, TensorStorageDelta, TensorStorageFull> tensors;

        struct PipelineStorage : cute::aligned_struct<16> {
            alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_KV;
            alignas(16) typename CollectiveMainloop::MainloopPipeline::SharedStorage pipeline_q;
            alignas(16) typename CollectiveMainloop::MainloopPipeline_dO::SharedStorage pipeline_do;
            alignas(16) typename MainloopPipelineQt::SharedStorage pipeline_qt;
            alignas(16) typename MainloopPipelineDOt::SharedStorage pipeline_dot;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
        } pipelines;

    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };

    // Kernel entry point API
    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
    };

    //
    // Methods
    //

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");

        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler)
        };
    }

    // Computes the kernel launch grid shape based on runtime parameters
    static dim3
    get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
    }

    static dim3
    get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int NumCopyThreads = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
        using PipelineParams = typename MainloopPipeline::Params;
        using PipelineState = typename MainloopPipeline::PipelineState;
        using MainloopPipeline_dO = typename CollectiveMainloop::MainloopPipeline_dO;
        using PipelineParams_dO = typename MainloopPipeline_dO::Params;
        using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
        static constexpr bool Q_dO_same_stages = std::is_same_v<MainloopPipeline, MainloopPipeline_dO>;

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();

        // Issue Tma Descriptor Prefetch from a single thread
        if (warp_idx == 0 && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }

        // Obtain warp index
        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

        PipelineParams pipeline_params;
        pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ + CollectiveMainloop::TmaTransactionBytesLSE;
        int warp_group_idx = cutlass::canonical_warp_group_idx();
        pipeline_params.role = warp_group_idx == 0
            ? MainloopPipeline::ThreadCategory::Producer
            : MainloopPipeline::ThreadCategory::Consumer;
        pipeline_params.is_leader = warp_group_thread_idx == 0;
        pipeline_params.num_consumers = NumMmaThreads;

        if (warp_idx == 0 && lane_predicate) {
            shared_storage.pipelines.barrier_KV.init(1 /*numThreads*/);
        }
        // We're counting on pipeline_q to call cutlass::arch::fence_barrier_init();
        MainloopPipeline pipeline_q(shared_storage.pipelines.pipeline_q, pipeline_params, ClusterShape{});
        auto role_dO = warp_group_idx == 0
            ? MainloopPipeline_dO::ThreadCategory::Producer
            : MainloopPipeline_dO::ThreadCategory::Consumer;
        PipelineParams_dO pipeline_params_dO {(CollectiveMainloop::kAbl & 128) ? CollectiveMainloop::TmaTransactionBytesLSE : pipeline_params.transaction_bytes, role_dO, pipeline_params.is_leader, pipeline_params.num_consumers};   // M16 ABL 128: dPsum only
        MainloopPipeline_dO pipeline_do(shared_storage.pipelines.pipeline_do, cute::conditional_return<Q_dO_same_stages && !(CollectiveMainloop::kAbl & 128)>(pipeline_params, pipeline_params_dO), ClusterShape{});   // M16 ABL 128 forces the dPsum-only byte count
        // ## FP11-M17 GT: the twin pipelines are TMA pipelines (one transaction
        // barrier per stage, fed by the producer warp; consumers = the MMA WGs).
        typename MainloopPipelineQt::Params pipeline_params_qt;
        pipeline_params_qt.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQt;
        pipeline_params_qt.role = warp_group_idx == 0
            ? MainloopPipelineQt::ThreadCategory::Producer
            : MainloopPipelineQt::ThreadCategory::Consumer;
        pipeline_params_qt.is_leader = warp_group_thread_idx == 0;
        pipeline_params_qt.num_consumers = NumMmaThreads;
        MainloopPipelineQt pipeline_qt(shared_storage.pipelines.pipeline_qt, pipeline_params_qt, ClusterShape{});
        typename MainloopPipelineDOt::Params pipeline_params_dot;
        pipeline_params_dot.transaction_bytes = CollectiveMainloop::TmaTransactionBytesdOt;
        pipeline_params_dot.role = warp_group_idx == 0
            ? MainloopPipelineDOt::ThreadCategory::Producer
            : MainloopPipelineDOt::ThreadCategory::Consumer;
        pipeline_params_dot.is_leader = warp_group_thread_idx == 0;
        pipeline_params_dot.num_consumers = NumMmaThreads;
        MainloopPipelineDOt pipeline_dot(shared_storage.pipelines.pipeline_dot, pipeline_params_dot, ClusterShape{});

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
        if constexpr (size(ClusterShape{}) > 1) {
            cute::cluster_arrive_relaxed();
            cute::cluster_wait();
        } else {
            __syncthreads();
        }

        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

        if (warp_group_idx == 0) {  // Producer
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            if (warp_idx_in_warpgroup == 0) {  // Load K, V, and do TMA on Q and dO
                PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
                PipelineState_dO smem_pipe_write_do = cutlass::make_producer_start_state<MainloopPipeline_dO>();
                // ## FP11-M17 GT: the producer warp also feeds the Q^T / dO^T twin pipelines.
                PipelineStateQt smem_pipe_write_qt = cutlass::make_producer_start_state<MainloopPipelineQt>();
                PipelineStateDOt smem_pipe_write_dot = cutlass::make_producer_start_state<MainloopPipelineDOt>();
                for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler);
                     work_tile_info.is_valid(params.scheduler);
                     work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info)) {
                    auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
                    auto [n_block, bidh, bidb, _ /*split_idx*/] = block_coord_;
                    cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};
                    auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                        scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                    };
                    mainloop.load_gt(params.mainloop, pipeline_q, pipeline_do, pipeline_qt, pipeline_dot,
                                     smem_pipe_write, smem_pipe_write_do, smem_pipe_write_qt, smem_pipe_write_dot,
                                     shared_storage, scheduler_prefetch, block_coord);
                }
                mainloop.load_tail_gt(pipeline_q, pipeline_do, pipeline_qt, pipeline_dot,
                                      smem_pipe_write, smem_pipe_write_do, smem_pipe_write_qt, smem_pipe_write_dot);
            } else if (warp_idx_in_warpgroup == 1) {
                // ## FP11-M5: pass 0 has no dQ flush warp work (store_dq's
                // post-return body names pass-1-only smem members; keep it
                // uninstantiated in pass-0 mode).
                if constexpr (!DeltaPass0) {
                    for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                         work_tile_info.is_valid(params.scheduler);
                         work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
                        auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
                        auto [n_block, bidh, bidb, _ /*split_idx*/] = block_coord_;
                        cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};
                        mainloop.store_dq(params.mainloop, shared_storage, block_coord);
                    }
                }
            } else if (warp_idx_in_warpgroup == 2 || warp_idx_in_warpgroup == 3) {
                // ## FQ8-V9: the transpose warps.
                // ## FP11-M5: pass 0 consumes no transposed twins; the warps
                // idle (running them would deadlock the qt/dot pipelines,
                // which have no consumer in pass 0).
                // ## FP11-M17 GT: the twins arrive by TMA; these warps idle.
                if constexpr (!DeltaPass0 && !FA3FP11_M17_GT) {
                    PipelineState smem_pipe_read_q_t;
                    PipelineState_dO smem_pipe_read_do_t;
                    PipelineStateQt smem_pipe_write_qt = cutlass::make_producer_start_state<MainloopPipelineQt>();
                    PipelineStateDOt smem_pipe_write_dot = cutlass::make_producer_start_state<MainloopPipelineDOt>();
                    int const thread_idx_t = (threadIdx.x % cutlass::NumThreadsPerWarpGroup) - 2 * cutlass::NumThreadsPerWarp;
                    for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                         work_tile_info.is_valid(params.scheduler);
                         work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
                        auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
                        auto [n_block, bidh, bidb, _ /*split_idx*/] = block_coord_;
                        cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};
                        mainloop.transpose_loop(params.mainloop, pipeline_q, pipeline_do, pipeline_qt, pipeline_dot,
                                                smem_pipe_read_q_t, smem_pipe_read_do_t,
                                                smem_pipe_write_qt, smem_pipe_write_dot,
                                                thread_idx_t, block_coord, shared_storage);
                    }
                }
            }
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
            // Initialize matmul objects.
            TiledMmadKV tiled_mma_dKV;

            PipelineState smem_pipe_read;
            PipelineState_dO smem_pipe_read_do;
            PipelineStateQt smem_pipe_read_qt;
            PipelineStateDOt smem_pipe_read_dot;

            mainloop.mma_init();
            scheduler.init_consumer();

            int work_idx = 0;
            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {
                auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
                auto [n_block, bidh, bidb, _ /*split_idx*/] = block_coord_;
                cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};

                if constexpr (DeltaPass0) {
                    // ## FP11-M5 pass 0: delta partials only; no dK/dV
                    // fragments, no epilogue (an empty causal CTA writes
                    // nothing, and the fixed-order reduce never reads its
                    // tiles).
                    mainloop.mma_pass0(
                        params.mainloop, pipeline_q, pipeline_do,
                        smem_pipe_read, smem_pipe_read_do,
                        threadIdx.x - NumCopyThreads, work_idx, block_coord, shared_storage);
                } else {
                    // dK and dV output accumulator.
                    Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
                    Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
                    bool tile_valid = mainloop.mma(
                        params.mainloop, pipeline_q, pipeline_do, pipeline_qt, pipeline_dot,
                        smem_pipe_read, smem_pipe_read_do, smem_pipe_read_qt, smem_pipe_read_dot,
                        tdKrdK, tdVrdV, threadIdx.x - NumCopyThreads, work_idx, block_coord, shared_storage);
                    if (tile_valid) {
                        epilogue.store(params.epilogue, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV,
                                       threadIdx.x - NumCopyThreads, block_coord);
                    } else {
                        epilogue.store_zero(params.epilogue, threadIdx.x - NumCopyThreads, block_coord);
                    }
                }

            }
            if constexpr (!DeltaPass0) { epilogue.store_tail(); }
        }

    }

};

} // namespace flash
