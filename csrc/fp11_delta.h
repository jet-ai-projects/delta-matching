/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/
// ## FP11-M5 (in-kernel matched delta, the design notes): PASS 0 launch + the
// PASS 0.5 fixed-order reduce.
//
// PASS 0 = the pass-1 collective (mainloop_bwd_sm90_fp8v10.hpp) instantiated
// with DeltaPass0=true: the SAME grid (SingleTileBwdLPTScheduler over
// nbn*H*B for causal), the same 384-thread CTA with the same 40/232 register
// split, the same producer warp issuing the same TMA copies (Q, dO, K, V,
// the Kt twin, LSE, dPsum) with unchanged transaction-byte counts, and the
// same S-then-dP zero-init wgmma chains per tile: the realized fp32 P~ and
// raw dP~ fragments are bit-identical to pass 1's (checksum-verified in the
// FA3FP11_DBG build).  The consumer (mma_pass0) reduces the tile into the
// per-CTA per-row per-warpgroup-half partial delta workspace
// [BH, nbn, 2, N] fp32:
//
//   partial[bh][nb][wg][i] = 2^-8 * sum_{ni in wg's 32-key half of nb}
//                     P~(i,ni) * (dP~(i,ni) * do8sc(i-block)*v8s(ni-half))
//
// (the 2 warpgroup column halves are NOT combined in-CTA: the combine would
// need one 256-thread named barrier per tile, measured as a material part
// of the pass-0 tail; the fixed-order reduce below folds the halves
// instead -- a flagged deviation from D3's [BH, nbn, N] workspace shape,
// ~128 MB instead of ~64 MB at B2H8 N8192, still transient).
//
// The transpose warps idle and the epilogue is compiled out
// (DeltaNoEpilogue); occupancy stays register-bound at 1 CTA/SM (the 40/232
// setmaxnreg split fills the register file), so pass-0 speed comes from the
// consumer loop shape, not occupancy (iteration record in the mma_pass0
// loop comment).
//
// Dead-slot conventions the caller must honor (documented, load() runs its
// pass-1 instruction stream UNCHANGED): kt_ptr must be a real [BH, D, N]
// twin (its 16-KB one-shot load lands in smem_kt and is never consumed);
// dsoftmax_sum must be any valid (b, h, s) fp32 tensor (its 256-B bulk copy
// per tile keeps the dO pipeline transaction bytes identical; pass 0 never
// reads the values -- it is COMPUTING delta); ds8_ptr/psi_ptr need valid
// device pointers only for host-side TMA-descriptor construction and
// prefetch (never dereferenced: pass 0 issues no dS8 store and no psi
// publish).
//
// PASS 0.5 = fp11_delta_reduce_kernel: delta_i = sum over (n-tile j,
// warpgroup half w) of partial_i^(j,w), THE documented fixed order: one
// thread per row, j ascending with w = 0 then 1 inside each j (j < mb+1
// for causal, exactly the tile set pass 0 wrote; all nbn otherwise), fp64
// accumulator (summation ORDER and precision of delta are free by the derivation notes
// A.2 item 1 -- reassociation moves delta by round-off class only; fp64
// makes the cross-tile stage's own noise negligible against the fp32
// in-tile realization), rounded once to fp32 at the store.  Deterministic:
// one writer per row, one fixed order.
#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_check.h"
#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"
#include "static_switch.h"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "mainloop_bwd_sm90_fp8v10.hpp"
#include "flash_bwd_kernel_sm90_fp8v10.h"

#include "flash_bwd_fp8_params.h"

using namespace cute;

namespace flash {

// ## FP11-M5: the no-op epilogue for pass 0 (no dK/dV outputs).  Provides
// exactly the pieces FlashAttnBwdSm90Fp8V10 references unconditionally
// (Varlen, Arguments/Params, to_underlying_arguments, prefetch, default
// construction); store/store_zero/store_tail are only named in the
// constexpr-discarded pass-1 branch.
struct DeltaNoEpilogue {
    static constexpr bool Varlen = false;
    struct TensorStorage {};
    struct Arguments {};
    struct Params {};
    static Params to_underlying_arguments(Arguments const&) { return {}; }
    CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const&) {}
};

}  // namespace flash

namespace fp11d {

// One thread per dQ row; ascending-(j, w) fp64 chain (header comment).
__global__ void __launch_bounds__(64, 1)
fp11_delta_reduce_kernel(float const* __restrict__ part, float* __restrict__ dlt,
                         int nbn, int seqlen_q, bool is_causal) {
    int const mb = blockIdx.x;
    int const bh = blockIdx.y;
    int const i = mb * 64 + threadIdx.x;
    if (i >= seqlen_q) { return; }
    int const jn = is_causal ? (mb + 1 < nbn ? mb + 1 : nbn) : nbn;
    float const* col = part + int64_t(bh) * nbn * 2 * seqlen_q + i;
    double acc = 0.0;
    // Unrolled for load-issue ILP only: the fp64 accumulation ORDER is the
    // documented (j ascending, half 0 then 1) chain, unchanged.
    #pragma unroll 4
    for (int j = 0; j < jn; ++j) {
        acc += double(col[int64_t(2 * j) * seqlen_q]);        // warpgroup half 0
        acc += double(col[int64_t(2 * j + 1) * seqlen_q]);    // warpgroup half 1
    }
    dlt[int64_t(bh) * seqlen_q + i] = float(acc);
}

}  // namespace fp11d

#include "fp11_delta_v3.h"   // ## FP11-M16 lever G: 256 keys per CTA, two n-blocks per warpgroup (takes precedence over P0V2)
#include "fp11_delta_v4.h"   // ## FP11-M17 T2a: v3 with the wgmma issue software-pipelined across stages (precedence over P0V3)

// ## FP11-M5: v11-specific symbol (same no-weak-overlap rationale as
// run_flash_bwd_fp11_d256).  Launches pass 0 then the reduce, stream-ordered.
template <bool Is_causal>
void run_flash_delta_fp11_d256(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    run_flash_delta_v4_fp11_d256<Is_causal>(params, stream);
    return;
    using Element = cutlass::float_e4m3_t;
    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 64;
    static constexpr int kHeadDim = 256;
    // FLAGGED DEVIATION from D3's "same pipeline", speed-only: pass 0 runs
    // Q and dO at 3 stages (pass 1: 2 and 1).  Pass 0's per-tile work is
    // ~4x smaller than pass 1's, so the shallow staging exposed both the dO
    // release->refill latency and the Q/dO tile stream's L2 latency (pass 0
    // re-reads every Q/dO tile once per key-block CTA, ~4.3 GB of mostly
    // L2-served traffic at B2H8 N8192: near the kernel's binding resource).
    // Staging depth changes NO realized arithmetic (same bytes staged, same
    // chains); the M5 gates re-verify the checksums + all output hashes on
    // this exact configuration.
    static constexpr int Stages = 3;
    static constexpr int Stages_dO = 3;
    static constexpr int Stages_dS = 1;

    int const seqlen_q = params.seqlen_q;
    int const seqlen_k = params.seqlen_k;
    int const seqlen_q_rounded = params.seqlen_q_rounded;

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopBwdSm90Fp8V10<
        Stages, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK, Element, ElementAccum, ArchTag,
        Is_causal, false /*Is_local*/, false /*Has_softcap*/, false /*Varlen*/, false /*Deterministic*/,
        false /*SdP_swapAB*/, true /*dKV_swapAB*/, true /*dQ_swapAB*/,
        2 /*NumMmaWarpGroups*/, 1, 1, 1, false /*Mma_dP_is_RS*/, true /*DeltaPass0*/>;
    using CollectiveEpilogue = flash::DeltaNoEpilogue;
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
        {params.d, seqlen_q, params.h, params.b},   // shape_Qt (dead slot)
        {int64_t(seqlen_q), _1{}, int64_t(params.d) * seqlen_q, int64_t(params.d) * seqlen_q * params.h},
        static_cast<Element const*>(params.kt_ptr),
        {params.d, seqlen_k, params.h_k, params.b},  // shape_Kt
        {int64_t(seqlen_k), _1{}, int64_t(params.d) * seqlen_k, int64_t(params.d) * seqlen_k * params.h_k},
        static_cast<Element const*>(params.dot_ptr),
        {params.dv, seqlen_q, params.h, params.b},  // shape_dOt (dead slot)
        {int64_t(seqlen_q), _1{}, int64_t(params.dv) * seqlen_q, int64_t(params.dv) * seqlen_q * params.h},
        params.q8s_ptr, params.k8s_ptr, params.v8s_ptr, params.do8s_ptr,
        params.nblk_q, params.nblk_k,
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, params.b},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * seqlen_q_rounded * params.h},
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {seqlen_q_rounded, params.h, params.b},  // shape_LSE
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),   // dead values (loaded, never read)
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},  // stride_dPsum
        1.0f,       // softmax_scale (sm is baked into the Q codes; unused here)
        -1, -1, 0,  // window_size_left/right, attention_chunk
        0.0f,       // softcap
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k,
        static_cast<Element*>(params.ds8_ptr),      // descriptor-only dummy
        params.psi_ptr,                              // never written in pass 0
        cute::ceil_div(params.seqlen_q, kBlockM), cute::ceil_div(params.seqlen_k, kBlockN),
        params.dlt_part_ptr,
        params.dbg_rows_ptr, params.dbg_rowsabs_ptr, params.dbg_cks_ptr
    };
    typename CollectiveEpilogue::Arguments epilogue_args {};

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
        fprintf(stderr, "[delta_matching pass 0] smem_size=%d grid=(%d,%d,%d) block=(%d,%d,%d)\n",
                smem_size, grid_dims.x, grid_dims.y, grid_dims.z, block_dims.x, block_dims.y, block_dims.z);
    }
    CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/));
    if (std::getenv("DM_DEBUG")) {
        cudaError_t e_ = cudaGetLastError();
        fprintf(stderr, "[delta_matching pass 0] post-launch cudaGetLastError: %s\n", cudaGetErrorString(e_));
    }

    // ## PASS 0.5: the fixed-order reduce (stream-ordered behind pass 0).
    {
        int const nbm = cute::ceil_div(params.seqlen_q, kBlockM);
        int const nbn = cute::ceil_div(params.seqlen_k, kBlockN);
        dim3 grid(nbm, params.b * params.h, 1);
        fp11d::fp11_delta_reduce_kernel<<<grid, 64, 0, stream>>>(
            params.dlt_part_ptr, params.dlt_ptr, nbn, params.seqlen_q, Is_causal);
        cudaError_t e_ = cudaGetLastError();
        if (e_ != cudaSuccess) { fprintf(stderr, "[delta_matching pass 0.5] launch: %s\n", cudaGetErrorString(e_)); exit(1); }
    }
}
