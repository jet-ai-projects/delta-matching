/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/
#include "cuda_check.h"
// Launch glue of the fp8 split backward on FA3's schedule: pass 1 (dK / dV and the dS' code tiles; head_dim 256 on
// the V10 collective, head_dim 128 on the P1-R collective), then pass 2 (dQ as a block-scaled e4m3 GEMM). No
// preprocess launch: the host API supplies the LSE (times log2 e), the delta (stale or matched), the e4m3 codes, their
// transposed twins and the per-block scales. Adapted from FlashAttention-3 (hopper/flash_bwd_launch_template.h).
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"
#include "static_switch.h"
#include "flash.h"
#include "flash_bwd_postprocess_kernel.h"
#include "tile_scheduler.hpp"
#include "mainloop_bwd_sm90_fp8v10.hpp"
#include "epilogue_bwd.hpp"
#include "flash_bwd_kernel_sm90_fp8v10.h"

#include "flash_bwd_fp8_params.h"
#include "fq8_dq_pass2.h"
#include "fq8_dq_pass2_hd128.h"   // ## Delta-Matching-128 (M19): pass 2 at head_dim 128 (64-row x 128-key tiles)
#ifndef FA3FP11_HD128_P1X
#define FA3FP11_HD128_P1X 0
#endif
// ## Delta-Matching-128 v2 (M20): pass 1 in FA3's hd128 geometry (swapAB S/dP, RS dV/dK from the register fragments);
// hd128 only (the release build sets FA3FP11_HD128_P1R); pass 2 and the d256 path are untouched.
#include "mainloop_bwd_sm90_fp8r.hpp"
#include "flash_bwd_kernel_sm90_fp8r.h"
#ifndef FA3FP11_HD128_P1P
#define FA3FP11_HD128_P1P 0
#endif

using namespace cute;

// ## FP11 (M2): v11-specific launch symbol.  The M1 package kept the parent's
// run_flash_bwd_fp8v10_d256 name, which is a WEAK template symbol overlapping
// the other FA3-derived libraries; RTLD_LOCAL keeps them separate today, but
// now that the v11 backward diverges numerically from v10 the overlap must be
// impossible by construction (a hypothetical RTLD_GLOBAL load order could
// otherwise cross-bind one family's backward into the other).
// ## Delta-Matching-128 (M19): the launch body is templated on the head dim.  d256 keeps its name and
// configuration verbatim (kBlockM 64, kBlockN 64); hd128 = kBlockM 64, kBlockN 128 (FA3's own hd128 bwd tile; each
// MMA warpgroup owns a 64-key scale group), pass 2 = fq8_dq_pass2_hd128.h.
// ## Delta-Matching-128 v2 (M20) P1-R launch: the RS pass 1 at hd128 (kBlockM 64, kBlockN 128) + the unchanged pass 2.
template <bool Is_causal>
void run_flash_bwd_fp11r_hd128(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    using Element = cutlass::float_e4m3_t;
    using ElementOut = cutlass::bfloat16_t;
    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 128;
    static constexpr int kHeadDim = 128;
    int const seqlen_q = params.seqlen_q;
    int const seqlen_k = params.seqlen_k;
    int const seqlen_q_rounded = params.seqlen_q_rounded;
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopBwdSm90Fp8R<FA3FP11_HD128_QST, FA3FP11_HD128_DOST, FA3FP11_HD128_DOTST, Is_causal>;
    using CollectiveEpilogue = flash::CollectiveEpilogueBwd<
        TileShape_MNK, ElementOut, ArchTag, CollectiveMainloop::NumMmaThreads,
        false /*Varlen*/, false /*dKV_swapAB*/, 1 /*AtomLayoutKdKV = NumMmaWarpGroups / AtomLayoutNdKV*/>;
    using Scheduler = std::conditional_t<
        Is_causal,
        flash::SingleTileBwdLPTScheduler<false /*Varlen*/, kBlockN, false /*SPT*/>,
        flash::SingleTileScheduler<false /*Varlen*/, false /*Split*/, false /*PackGQA*/, kBlockN>
    >;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnBwdSm90Fp8R<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, params.b},
        {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},
        static_cast<Element const*>(params.k_ptr),
        {seqlen_k, params.d, params.h_k, params.b},
        {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},
        // ## bf16-dP: the dP operands are the RAW bf16 tensors at SAGEDP=1 and the e4m3 codes otherwise.
#if FA3FP11_P1R_SAGEDP
        static_cast<typename CollectiveMainloop::ElementDp const*>(params.vraw_ptr),
        {seqlen_k, params.dv, params.h_k, params.b},
        {params.vraw_row_stride, _1{}, params.vraw_head_stride, params.vraw_batch_stride},
        static_cast<typename CollectiveMainloop::ElementDp const*>(params.doraw_ptr),
        {seqlen_q, params.dv, params.h, params.b},
        {params.doraw_row_stride, _1{}, params.doraw_head_stride, params.doraw_batch_stride},
#else
        static_cast<Element const*>(params.v_ptr),
        {seqlen_k, params.dv, params.h_k, params.b},
        {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},
        static_cast<Element const*>(params.do_ptr),
        {seqlen_q, params.dv, params.h, params.b},
        {params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride},
#endif
        static_cast<Element const*>(params.qt_ptr),
        {params.d, seqlen_q, params.h, params.b},
        {int64_t(seqlen_q), _1{}, int64_t(params.d) * seqlen_q, int64_t(params.d) * seqlen_q * params.h},
        static_cast<Element const*>(params.kt_ptr),
        {params.d, seqlen_k, params.h_k, params.b},
        {int64_t(seqlen_k), _1{}, int64_t(params.d) * seqlen_k, int64_t(params.d) * seqlen_k * params.h_k},
        static_cast<Element const*>(params.dot_ptr),
        {params.dv, seqlen_q, params.h, params.b},
        {int64_t(seqlen_q), _1{}, int64_t(params.dv) * seqlen_q, int64_t(params.dv) * seqlen_q * params.h},
        params.q8s_ptr, params.k8s_ptr, params.v8s_ptr, params.do8s_ptr,
        params.nblk_q, params.nblk_k,
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, params.b},
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * seqlen_q_rounded * params.h},
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {seqlen_q_rounded, params.h, params.b},
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},
        1.0f,
        -1, -1, 0,
        0.0f,
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k,
        static_cast<Element*>(params.ds8_ptr), params.psi_ptr,
        cute::ceil_div(params.seqlen_q, kBlockM), cute::ceil_div(params.seqlen_k, kBlockN),
        params.dlt_part_ptr, params.dbg_rows_ptr, params.dbg_rowsabs_ptr,
        params.dbg_cks_ptr
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.dk_ptr),
        {seqlen_k, params.d, params.h, params.b},
        {params.dk_row_stride, _1{}, params.dk_head_stride, params.dk_batch_stride},
        static_cast<ElementOut*>(params.dv_ptr),
        {seqlen_k, params.dv, params.h, params.b},
        {params.dv_row_stride, _1{}, params.dv_head_stride, params.dv_batch_stride},
        params.b, params.h, params.dk_semaphore, params.dv_semaphore, params.cu_seqlens_k, params.seqused_k,
    };
    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_n, params.h, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        params.seqlen_k,
        params.seqlen_q, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_k, params.seqused_k
    };
    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args});
    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<AttnKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    if (std::getenv("DM_DEBUG")) {
        fprintf(stderr, "[delta_matching hd128] smem_size=%d grid=(%d,%d,%d) block=(%d,%d,%d)\n",
                smem_size, grid_dims.x, grid_dims.y, grid_dims.z, block_dims.x, block_dims.y, block_dims.z);
    }
    CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/));
    if (std::getenv("DM_DEBUG")) {
        cudaError_t e_ = cudaGetLastError();
        fprintf(stderr, "[delta_matching hd128] post-launch cudaGetLastError: %s\n", cudaGetErrorString(e_));
    }
    run_fq8_dq_pass2_fp11_d128<Is_causal>(params, stream);   // pass 2 unchanged (the same dS' tiles + psi contract)
}


template <bool Is_causal, int kHeadDim_>
void run_flash_bwd_fp11_impl(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    if constexpr (kHeadDim_ < 256) { run_flash_bwd_fp11r_hd128<Is_causal>(params, stream); return; }   // ## M20 P1-R
    using Element = cutlass::float_e4m3_t;
    using ElementOut = cutlass::bfloat16_t;
    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;
    // ## Delta-Matching-128 lever P1-C' (FA3FP11_HD128_BM128=1, hd128 only): 128-row pipeline stages processed as two 64-row
    // halves (mainloop kBlockMh); the certified d256 configuration stays kBlockM 64.
    static constexpr int kBlockM = (kHeadDim_ < 256 && FA3FP11_HD128_BM128) ? 128 : 64;
    static constexpr int kBlockN = kHeadDim_ >= 256 ? 64 : 128;
    static constexpr int kHeadDim = kHeadDim_;
    // ## FQ8: e4m3 halves every element buffer vs bf16: dO double-buffers (the
    // bf16 d256 config could only afford Stages_dO=1).
    static constexpr int Stages_d256 = 2;
    static constexpr int Stages_dO_d256 = 1;
    // ## Delta-Matching-128 lever P1-F (hd128 only; the d256 pins above are untouched): pass 1 at head_dim 128 runs ONE CTA
    // per SM already (169 KB of smem at P1-A), so the remaining ~58 KB is dead space.  Spending it on pipeline DEPTH
    // (raw Q stages FA3FP11_HD128_QST, which also deepens the Q twin, and raw dO stages FA3FP11_HD128_DOST) buys latency
    // hiding for the per-tile chain that M16 proved is pass 1's cost, and is a SCHEDULE change only: each tile's
    // arithmetic, order and slots are unchanged, so the outputs must be bitwise (measured, not argued).
    static constexpr int Stages = kHeadDim_ >= 256 ? Stages_d256 : FA3FP11_HD128_QST;
    static constexpr int Stages_dO = kHeadDim_ >= 256 ? Stages_dO_d256 : FA3FP11_HD128_DOST;
    static_assert(Stages >= Stages_dO, "pass 1: the dO pipeline cannot be deeper than the Q pipeline");
    static constexpr int Stages_dS = 1;

    int const seqlen_q = params.seqlen_q;
    int const seqlen_k = params.seqlen_k;
    int const seqlen_q_rounded = params.seqlen_q_rounded;

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopBwdSm90Fp8V10<
        Stages, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK, Element, ElementAccum, ArchTag,
        Is_causal, false /*Is_local*/, false /*Has_softcap*/, false /*Varlen*/, false /*Deterministic*/,
        false /*SdP_swapAB*/, true /*dKV_swapAB*/, true /*dQ_swapAB*/,
        2 /*NumMmaWarpGroups*/, 1, 1, 1, false /*V_in_regs*/>;
    using CollectiveEpilogue = flash::CollectiveEpilogueBwd<
        TileShape_MNK, ElementOut, ArchTag, CollectiveMainloop::NumMmaThreads,
        false /*Varlen*/, true /*dKV_swapAB*/, 2 /*AtomLayoutKdKV-ish*/>;
    using Scheduler = std::conditional_t<
        Is_causal,
        flash::SingleTileBwdLPTScheduler<false /*Varlen*/, kBlockN, false /*SPT*/>,
        flash::SingleTileScheduler<false /*Varlen*/, false /*Split*/, false /*PackGQA*/, kBlockN>
    >;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnBwdSm90Fp8V10<
        CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, params.b},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},
        static_cast<Element const*>(params.k_ptr),
        {seqlen_k, params.d, params.h_k, params.b},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},
        static_cast<Element const*>(params.v_ptr),
        {seqlen_k, params.dv, params.h_k, params.b},  // shape_V
        {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},
        static_cast<Element const*>(params.do_ptr),
        {seqlen_q, params.dv, params.h, params.b},  // shape_dO
        {params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride},
        // ## FQ8 twins: [d, seq, head, batch], seq-contiguous ([BH, D, N] tensors).
        static_cast<Element const*>(params.qt_ptr),
        {params.d, seqlen_q, params.h, params.b},   // shape_Qt
        {int64_t(seqlen_q), _1{}, int64_t(params.d) * seqlen_q, int64_t(params.d) * seqlen_q * params.h},
        static_cast<Element const*>(params.kt_ptr),
        {params.d, seqlen_k, params.h_k, params.b},  // shape_Kt
        {int64_t(seqlen_k), _1{}, int64_t(params.d) * seqlen_k, int64_t(params.d) * seqlen_k * params.h_k},
        static_cast<Element const*>(params.dot_ptr),
        {params.dv, seqlen_q, params.h, params.b},  // shape_dOt
        {int64_t(seqlen_q), _1{}, int64_t(params.dv) * seqlen_q, int64_t(params.dv) * seqlen_q * params.h},
        params.q8s_ptr, params.k8s_ptr, params.v8s_ptr, params.do8s_ptr,
        params.nblk_q, params.nblk_k,
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, params.b},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * seqlen_q_rounded * params.h},
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {seqlen_q_rounded, params.h, params.b},  // shape_LSE
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},  // stride_dPsum
        1.0f,  // ## FQ8: softmax_scale=1 in the mainloop (sm is baked into the Q
               // codes; dQ's explicit sm factor rides in the postprocess).
        -1, -1, 0,  // window_size_left/right, attention_chunk
        0.0f,       // softcap
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k,
        static_cast<Element*>(params.ds8_ptr), params.psi_ptr,
        cute::ceil_div(params.seqlen_q, CollectiveMainloop::kBlockMh), cute::ceil_div(params.seqlen_k, kBlockN),   // nbm counts the 64-row dS' / psi tiles
        // ## FP11-M5: pass-0 partial ptr (null in pass 1) + DBG-only slots.
        params.dlt_part_ptr, params.dbg_rows_ptr, params.dbg_rowsabs_ptr,
        params.dbg_cks_ptr
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.dk_ptr),
        {seqlen_k, params.d, params.h, params.b},  // shape_dK
        {params.dk_row_stride, _1{}, params.dk_head_stride, params.dk_batch_stride},
        static_cast<ElementOut*>(params.dv_ptr),
        {seqlen_k, params.dv, params.h, params.b},
        {params.dv_row_stride, _1{}, params.dv_head_stride, params.dv_batch_stride},
        params.b,
        params.h,
        params.dk_semaphore,
        params.dv_semaphore,
        params.cu_seqlens_k,
        params.seqused_k,
    };

    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_n, params.h, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        params.seqlen_k,
        params.seqlen_q, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_k, params.seqused_k
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<AttnKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    if (std::getenv("DM_DEBUG")) {
        fprintf(stderr, "[delta_matching] smem_size=%d grid=(%d,%d,%d) block=(%d,%d,%d)\n",
                smem_size, grid_dims.x, grid_dims.y, grid_dims.z, block_dims.x, block_dims.y, block_dims.z);
    }
    CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/));
    if (std::getenv("DM_DEBUG")) {
        cudaError_t e_ = cudaGetLastError();
        fprintf(stderr, "[delta_matching] post-launch cudaGetLastError: %s\n", cudaGetErrorString(e_));
    }

    // ## V10 (split bwd): pass 2 = the block-scaled dQ GEMM over the dS8
    // tiles pass 1 just stored (stream-ordered behind the main kernel).
    if constexpr (kHeadDim >= 256) {
    run_fq8_dq_pass2_fp11<Is_causal>(params, stream);
    } else {
    run_fq8_dq_pass2_fp11_d128<Is_causal>(params, stream);   // ## Delta-Matching-128
    }
}

template <bool Is_causal>
void run_flash_bwd_fp11_d256(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    run_flash_bwd_fp11_impl<Is_causal, 256>(params, stream);
}

// ## Delta-Matching-128
template <bool Is_causal>
void run_flash_bwd_fp11_d128(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    run_flash_bwd_fp11_impl<Is_causal, 128>(params, stream);
}
