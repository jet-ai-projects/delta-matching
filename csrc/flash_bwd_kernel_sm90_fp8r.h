/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/
/******************************************************************************
 * ## Delta-Matching-128 v2 (M20): the pass-1 kernel wrapper for the P1-R collective (mainloop_bwd_sm90_fp8r.hpp).
 * Forked from flash_bwd_kernel_sm90_fp8v10.h with the transposer warps, the dQ flush warp and the pass-0 mode removed:
 * one producer warp (TMA), two MMA warpgroups, FA3's 24 / 240 register split taken from the collective.
 ******************************************************************************/
#pragma once

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
class FlashAttnBwdSm90Fp8R {

public:
    static constexpr bool Is_causal = CollectiveMainloop_::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop_::Is_local;
    static_assert(CollectiveMainloop_::Varlen == CollectiveEpilogue_::Varlen);
    static constexpr bool Varlen = CollectiveMainloop_::Varlen;

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
    static_assert(NumMmaWarpGroups == 2);

    static constexpr uint32_t LoadRegisterRequirement = CollectiveMainloop::LoadRegisterRequirement;
    static constexpr uint32_t MmaRegisterRequirement = CollectiveMainloop::MmaRegisterRequirement;

    struct TensorStorageFull : cute::aligned_struct<128> {
        union {
            typename CollectiveMainloop::TensorStorage mainloop;
            typename CollectiveEpilogue::TensorStorage epilogue;
        };
    };
    struct SharedStorage {
        TensorStorageFull tensors;
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

    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };
    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
    };

    static Params to_underlying_arguments(Arguments const& args) {
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) { sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id); }
        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {CollectiveMainloop::to_underlying_arguments(args.mainloop),
                CollectiveEpilogue::to_underlying_arguments(args.epilogue),
                hw_info, TileScheduler::to_underlying_arguments(args.scheduler)};
    }
    static dim3 get_grid_shape(Params const& params) { return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count); }
    static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {
        static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int NumCopyThreads = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
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
        if (warp_idx == 0 && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }
        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        PipelineParams pipeline_params;
        pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ + CollectiveMainloop::TmaTransactionBytesLSE;
        int warp_group_idx = cutlass::canonical_warp_group_idx();
        pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
        pipeline_params.is_leader = warp_group_thread_idx == 0;
        pipeline_params.num_consumers = NumMmaThreads;
        if (warp_idx == 0 && lane_predicate) { shared_storage.pipelines.barrier_KV.init(1 /*numThreads*/); }
        MainloopPipeline pipeline_q(shared_storage.pipelines.pipeline_q, pipeline_params, ClusterShape{});
        auto role_dO = warp_group_idx == 0 ? MainloopPipeline_dO::ThreadCategory::Producer : MainloopPipeline_dO::ThreadCategory::Consumer;
        // The dO pipeline carries raw dO + dPsum. Its transaction count tracked Q's while both were e4m3;
        // under bf16-dP raw dO is bf16, so it must be sized from the collective's own dO constant.
        PipelineParams_dO pipeline_params_dO {CollectiveMainloop::TmaTransactionBytesdO + CollectiveMainloop::TmaTransactionBytesLSE,
                                              role_dO, pipeline_params.is_leader, pipeline_params.num_consumers};
        MainloopPipeline_dO pipeline_do(shared_storage.pipelines.pipeline_do, cute::conditional_return<Q_dO_same_stages>(pipeline_params, pipeline_params_dO), ClusterShape{});
        typename MainloopPipelineQt::Params pipeline_params_qt;
        pipeline_params_qt.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQt;
        pipeline_params_qt.role = warp_group_idx == 0 ? MainloopPipelineQt::ThreadCategory::Producer : MainloopPipelineQt::ThreadCategory::Consumer;
        pipeline_params_qt.is_leader = warp_group_thread_idx == 0;
        pipeline_params_qt.num_consumers = NumMmaThreads;
        MainloopPipelineQt pipeline_qt(shared_storage.pipelines.pipeline_qt, pipeline_params_qt, ClusterShape{});
        typename MainloopPipelineDOt::Params pipeline_params_dot;
        pipeline_params_dot.transaction_bytes = CollectiveMainloop::TmaTransactionBytesdOt;
        pipeline_params_dot.role = warp_group_idx == 0 ? MainloopPipelineDOt::ThreadCategory::Producer : MainloopPipelineDOt::ThreadCategory::Consumer;
        pipeline_params_dot.is_leader = warp_group_thread_idx == 0;
        pipeline_params_dot.num_consumers = NumMmaThreads;
        MainloopPipelineDOt pipeline_dot(shared_storage.pipelines.pipeline_dot, pipeline_params_dot, ClusterShape{});

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;
        __syncthreads();   // the pipeline init must be visible to every warp
        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

        if (warp_group_idx == 0) {  // Producer
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            if (warp_idx_in_warpgroup == 0) {
                PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
                PipelineState_dO smem_pipe_write_do = cutlass::make_producer_start_state<MainloopPipeline_dO>();
                PipelineStateQt smem_pipe_write_qt = cutlass::make_producer_start_state<MainloopPipelineQt>();
                PipelineStateDOt smem_pipe_write_dot = cutlass::make_producer_start_state<MainloopPipelineDOt>();
                for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler);
                     work_tile_info.is_valid(params.scheduler);
                     work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info)) {
                    auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
                    auto [n_block, bidh, bidb, _ /*split_idx*/] = block_coord_;
                    cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};
                    auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() { scheduler.prefetch_next_work(params.scheduler, work_tile_info); };
                    mainloop.load_gt(params.mainloop, pipeline_q, pipeline_do, pipeline_qt, pipeline_dot,
                                     smem_pipe_write, smem_pipe_write_do, smem_pipe_write_qt, smem_pipe_write_dot,
                                     shared_storage, scheduler_prefetch, block_coord);
                }
                mainloop.load_tail_gt(pipeline_q, pipeline_do, pipeline_qt, pipeline_dot,
                                      smem_pipe_write, smem_pipe_write_do, smem_pipe_write_qt, smem_pipe_write_dot);
            }
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
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
                // dK and dV accumulators: (kBlockN keys, kHeadDim) per WG slice (!dKV_swapAB).
                Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select<1, 2>(TileShape_MNK{}));
                Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select<1, 2>(TileShape_MNK{}));
                bool tile_valid = mainloop.mma(
                    params.mainloop, pipeline_q, pipeline_do, pipeline_qt, pipeline_dot,
                    smem_pipe_read, smem_pipe_read_do, smem_pipe_read_qt, smem_pipe_read_dot,
                    tdKrdK, tdVrdV, threadIdx.x - NumCopyThreads, work_idx, block_coord, shared_storage);
                if (tile_valid) {
                    epilogue.store(params.epilogue, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV, threadIdx.x - NumCopyThreads, block_coord);
                } else {
                    epilogue.store_zero(params.epilogue, threadIdx.x - NumCopyThreads, block_coord);
                }
            }
            epilogue.store_tail();
        }
    }
};

}  // namespace flash
