// ## P2DG (head_dim 256, build flag FA3FP11_D256_P2DG=1, default 0): a DeepGEMM-class pass 2 -- a SEPARATE NUMERICS
// ARM for dQ only (o / dk / dv untouched: pass 2 reads the dS' workspace and psi, nothing else consumes its output).
//
//   dQ[m, :] = sm/256 * sum_nb psi[mb, nb] * (dS'8[m, nb tile] @ Kt8[nb tile, :])
//
// Numerics: every tile's GEMM runs in a FRESH tensor-core accumulator (scale_d = 0 on the first k32 step, the second
// k32 step accumulates in the tensor core exactly as the bitwise kernel's in-tile chain), and is promoted with one fp32
// FFMA per element into a register accumulator: acc = fmaf(D_tile, psi_tile, acc), tile order.  No psi-ratio rescale,
// no window, no skip rule (a psi == 0 sentinel tile has exact-zero codes, so its FFMA adds +0), no smem_dq: the
// accumulator lives in registers for the whole row.  Deterministic (fixed tile order, one CTA per output row block).
//
// Structure: one CTA = one 64-row block x 256 columns; two math warpgroups own 128 columns each (64 accumulator + 2 x 64
// double-buffered tile registers per thread, setmaxnreg 232) and pipeline their own tiles: issue wgmma(t) into D[t & 1],
// then retire wgmma(t-1) (wait<1>) and FFMA-promote it while wgmma(t) runs.  A producer warpgroup (setmaxnreg 40) feeds
// kStages stages of (dS' tile, K^T tile); the K^T tile is TMA-multicast over a (1, kCL, 1) cluster of the GQA heads that
// share the KV head (same row block, same tile sequence), so its L2 reads drop by kCL.  Each warpgroup runs kU = 8
// tiles per loop iteration through its two tile buffers (7 of 8 FFMA promotes overlap its own next wgmma) with no
// wgmma group outstanding across the back-edge (ptxas otherwise serializes the wgmmas, C7514 / C7517).
#pragma once

#include <cstdio>
#include <cstdlib>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

#include "utils.h"

namespace fq8p2dg {

using namespace cute;
using Element = cutlass::float_e4m3_t;

constexpr int kBM = 64;       // dQ rows per CTA (= pass-1 kBlockM)
constexpr int kBD = 256;      // head dim
constexpr int kBK = 64;       // keys per tile (= pass-1 kBlockN)
constexpr int kNH = 2;        // column halves = math warpgroups
constexpr int kBDH = kBD / kNH;
constexpr int kStages = 8;
constexpr int kThreads = 128 * (kNH + 1);
constexpr int kRegProducer = 40;
constexpr int kRegMath = 232;
static_assert(kNH * 128 * kRegMath + 128 * kRegProducer <= 65536, "P2DG: setmaxnreg split exceeds the register file");
constexpr int kStPitchW = kBD / 2 + 4;   // bf16 staging tile row pitch in u32 (== 4 mod 32: conflict-free fragment stores)

using TileShapeH = Shape<Int<kBM>, Int<kBDH>, Int<kBK>>;
using TiledMmaH = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, float, TileShapeH, GMMA::Major::K, GMMA::Major::K>(),
    Layout<Shape<_1, _1, _1>>{}));

using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GMMA::Major::K, Element, Int<kBM>, Int<kBK>>());
using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<kBM>{}, Int<kBK>{}, Int<kStages>{})));
using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GMMA::Major::K, Element, Int<kBD>, Int<kBK>>());
using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<kBD>{}, Int<kBK>{}, Int<kStages>{})));

struct SharedStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutA>, 1024> smem_a;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutB>, 1024> smem_b;
    cute::array_aligned<uint32_t, kBM * kStPitchW, 128> smem_st;   // bf16 epilogue staging tile
    alignas(16) cutlass::arch::ClusterTransactionBarrier full_bar[kStages];
    alignas(16) cutlass::arch::ClusterBarrier empty_bar[kStages];
};

using ShapeA = cute::Shape<Int<kBM>, Int<kBK>, int32_t, int32_t, int32_t>;   // (m, k, nb, mb, bh)
using StrideA = cute::Stride<Int<kBK>, _1, int64_t, int64_t, int64_t>;
using ShapeB = cute::Shape<Int<kBD>, int32_t, int32_t>;                       // (d, s, bh_kv)
using StrideB = cute::Stride<int64_t, _1, int64_t>;
using TmaA = decltype(make_tma_copy(
    cute::SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeA{}, StrideA{}),
    SmemLayoutA{}(_, _, _0{})));
template <int kCL>
using TmaBT = decltype(make_tma_copy(
    std::conditional_t<(kCL > 1), cute::SM90_TMA_LOAD_MULTICAST, cute::SM90_TMA_LOAD>{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeB{}, StrideB{}),
    SmemLayoutB{}(_, _, _0{}), Int<kCL>{}));

constexpr uint32_t kTxBytes = kBM * kBK + kBD * kBK;   // one dS' tile + the full K^T tile (multicast parts included)

template <int kCL>
struct P2DGParams {
    TmaA tma_a;
    TmaBT<kCL> tma_b;
    ShapeA shape_a;
    ShapeB shape_b;
    float const* ptr_psi;
    void* dq_ptr;
    int64_t dq_row_stride, dq_head_stride, dq_batch_stride;
    float final_scale;   // softmax_scale / 256 (the 2^8 MO factor)
    int nbm, nbn, h, h_k;
};

template <bool Is_causal, int kCL>
__global__ void __launch_bounds__(kThreads, 1)
fq8_dq_pass2_dg_kernel(CUTE_GRID_CONSTANT P2DGParams<kCL> const params) {
    extern __shared__ char smem_raw[];
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_raw);
    int const mb = params.nbm - 1 - int(blockIdx.x);   // longest-first (causal)
    int const bh = blockIdx.y;
    int const nb_end = Is_causal ? (mb + 1) : params.nbn;
    int const wg = __shfl_sync(0xffffffff, int(threadIdx.x) / 128, 0);
    int const tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < kStages; ++s) { shared.full_bar[s].init(1); shared.empty_bar[s].init(kNH * kCL); }
    }
    cutlass::arch::fence_barrier_init();
    if constexpr (kCL > 1) { cute::cluster_arrive_relaxed(); cute::cluster_wait(); } else { __syncthreads(); }

    Tensor sA = make_tensor(make_smem_ptr(shared.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared.smem_b.data()), SmemLayoutB{});

    if (wg == kNH) {   // producer warpgroup: one elected thread of its first warp
        cutlass::arch::warpgroup_reg_dealloc<kRegProducer>();
        if (tid / 32 == 0 && cute::elect_one_sync()) {
            Tensor mA = params.tma_a.get_tma_tensor(params.shape_a);
            Tensor mB = params.tma_b.get_tma_tensor(params.shape_b);
            int const bh_kv = (bh / params.h) * params.h_k + (bh % params.h) / (params.h / params.h_k);
            Tensor gB = flat_divide(mB(_, _, bh_kv), Shape<Int<kBD>, Int<kBK>>{})(_, _, _0{}, _);   // (256, 64, nbn)
            auto blkA = params.tma_a.get_slice(_0{});
            auto blkB = params.tma_b.get_slice(kCL > 1 ? int(cute::block_rank_in_cluster()) : 0);
            Tensor tAgA = group_modes<0, 3>(blkA.partition_S(mA(_, _, _, mb, bh)));   // (TMA, nbn)
            Tensor tAsA = group_modes<0, 3>(blkA.partition_D(sA));                    // (TMA, stages)
            Tensor tBgB = group_modes<0, 3>(blkB.partition_S(gB));
            Tensor tBsB = group_modes<0, 3>(blkB.partition_D(sB));
            uint16_t const mcast_mask = kCL > 1 ? uint16_t((1u << kCL) - 1) : uint16_t(0);
            #pragma unroll 1
            for (int nb = 0; nb < nb_end; ++nb) {
                int const s = nb % kStages;
                if (nb >= kStages) { shared.empty_bar[s].wait(((nb / kStages) - 1) & 1); }
                shared.full_bar[s].arrive_and_expect_tx(kTxBytes);
                auto& bar = reinterpret_cast<uint64_t&>(shared.full_bar[s]);
                copy(params.tma_a.with(bar, 0), tAgA(_, nb), tAsA(_, s));
                copy(params.tma_b.with(bar, mcast_mask), tBgB(_, nb), tBsB(_, s));
            }
        }
        __syncwarp();
    } else {   // math warpgroup wg: columns [wg * 128, wg * 128 + 128)
        cutlass::arch::warpgroup_reg_alloc<kRegMath>();
        TiledMmaH tiled_mma;
        auto wg_mma = tiled_mma.get_thread_slice(tid);
        Tensor sBh = flat_divide(sB, Shape<Int<kBDH>, Int<kBK>>{})(_, _, wg, _0{}, _);   // (128, 64, stages)
        Tensor tCrA = wg_mma.partition_fragment_A(sA);
        Tensor tCrB = wg_mma.partition_fragment_B(sBh);
        Tensor acc = partition_fragment_C(tiled_mma, select<0, 1>(TileShapeH{}));
        Tensor D0 = partition_fragment_C(tiled_mma, select<0, 1>(TileShapeH{}));
        Tensor D1 = partition_fragment_C(tiled_mma, select<0, 1>(TileShapeH{}));
        clear(acc);
        constexpr int kA = decltype(size(acc))::value;   // 64
        float const* psi_row = params.ptr_psi + (int64_t(bh) * params.nbm + mb) * params.nbn;

        // issue tile t's wgmma into D (fresh accumulator: scale_d = 0 on its first k32 step)
        auto issue = [&](auto& D, int t) {
            int const s = t % kStages;
            shared.full_bar[s].wait((t / kStages) & 1);
            warpgroup_fence_operand(D);
            warpgroup_arrive();
            tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
            #pragma unroll
            for (int kb = 0; kb < size<2>(tCrA); ++kb) {
                cute::gemm(tiled_mma, tCrA(_, _, kb, s), tCrB(_, _, kb, s), D);
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
            }
            warpgroup_commit_batch();
        };
        // tile t's wgmma has retired: release its stage in every cluster CTA, FFMA-promote it
        auto retire = [&](auto& D, int t, float psi) {
            warpgroup_fence_operand(D);
            // M10 invariant: every thread of this warpgroup passed full_bar.wait(t) before the stage is re-armed.
            cutlass::arch::NamedBarrier::sync(128, 1 + wg);
            int const s = t % kStages;
            if constexpr (kCL == 1) {
                if (tid == 0) { shared.empty_bar[s].arrive(); }
            } else {
                if (tid < kCL) { shared.empty_bar[s].arrive(uint32_t(tid), 1u); }
            }
            #pragma unroll
            for (int i = 0; i < kA; ++i) { acc(i) = fmaf(D(i), psi, acc(i)); }
        };

        // kU tiles per iteration through the two tile buffers, no wgmma group outstanding across the loop back-edge
        // (ptxas can then prove the D0 / D1 hand-offs and does not serialize): issue t -> D0, issue t+1 -> D1, then for
        // each tile: wait<1> (the older buffer retired) -> release + FFMA it while the newer one runs -> re-issue the
        // freed buffer with tile t+2..; the last tile of the iteration retires under wait<0>.  Only 1 of kU FFMA
        // promotes is not overlapped with this warpgroup's own wgmma.  psi is read one iteration ahead.
        constexpr int kU = 8;
        float pn[kU];
        #pragma unroll
        for (int j = 0; j < kU; ++j) { pn[j] = j < nb_end ? psi_row[j] : 0.0f; }
        int t = 0;
        #pragma unroll 1
        for (; t + kU <= nb_end; t += kU) {
            float pc[kU];
            #pragma unroll
            for (int j = 0; j < kU; ++j) { pc[j] = pn[j]; pn[j] = t + kU + j < nb_end ? psi_row[t + kU + j] : 0.0f; }
            issue(D0, t);
            issue(D1, t + 1);
            #pragma unroll
            for (int j = 0; j < kU; ++j) {
                if (j + 1 < kU) { warpgroup_wait<1>(); } else { warpgroup_wait<0>(); }
                if ((j & 1) == 0) {
                    retire(D0, t + j, pc[j]);
                    if (j + 2 < kU) { issue(D0, t + j + 2); }
                } else {
                    retire(D1, t + j, pc[j]);
                    if (j + 2 < kU) { issue(D1, t + j + 2); }
                }
            }
        }
        // remainder (< kU tiles): one tile at a time, alternating buffers is not needed (no overlap left to win much)
        #pragma unroll 1
        for (int j = 0; t < nb_end; ++t, ++j) {
            issue(D0, t);
            warpgroup_wait<0>();
            retire(D0, t, pn[j]);
        }

        // Epilogue: dq = acc * sm/256 -> bf16, staged through smem (both warpgroups), coalesced 16-B row stores.
        float const fin = params.final_scale;
        Tensor cC = cute::make_identity_tensor(make_shape(Int<kBM>{}, Int<kBDH>{}));
        Tensor tCcC = wg_mma.partition_C(cC);
        uint32_t* const st = shared.smem_st.data();
        #pragma unroll
        for (int i = 0; i < kA; i += 2) {
            int const m = get<0>(tCcC(i));
            int const d = get<1>(tCcC(i)) + wg * kBDH;
            cutlass::bfloat16_t lo(acc(i) * fin), hi(acc(i + 1) * fin);
            st[m * kStPitchW + d / 2] = (uint32_t(reinterpret_cast<uint16_t const&>(hi)) << 16)
                                      | uint32_t(reinterpret_cast<uint16_t const&>(lo));
        }
        cutlass::arch::NamedBarrier::sync(128 * kNH, 1 + kNH);
        int const b_idx = bh / params.h;
        int const h_idx = bh % params.h;
        auto* dq_base = reinterpret_cast<cutlass::bfloat16_t*>(params.dq_ptr)
            + int64_t(b_idx) * params.dq_batch_stride + int64_t(h_idx) * params.dq_head_stride
            + int64_t(mb) * kBM * params.dq_row_stride;
        int const ctid = int(threadIdx.x);   // 0 .. 255 over both math warpgroups
        #pragma unroll
        for (int it = 0; it < kBM / (kNH * 4); ++it) {
            int const row = it * (kNH * 4) + ctid / 32;
            int const c = ctid % 32;   // 16-B chunk of the 512-B row
            uint4 const v = *reinterpret_cast<uint4 const*>(st + row * kStPitchW + c * 4);
            *reinterpret_cast<uint4*>(dq_base + int64_t(row) * params.dq_row_stride + c * 8) = v;
        }
    }
    // no CTA leaves while a cluster peer can still multicast into its smem or arrive on its empty barriers
    if constexpr (kCL > 1) { cute::cluster_arrive(); cute::cluster_wait(); }
}

template <bool Is_causal, int kCL>
void run_cl(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    int const nbm = cute::ceil_div(params.seqlen_q, kBM);
    int const nbn = cute::ceil_div(params.seqlen_k, kBK);
    int const bh = params.b * params.h;
    ShapeA shape_a = make_shape(Int<kBM>{}, Int<kBK>{}, nbn, nbm, bh);
    StrideA stride_a = make_stride(Int<kBK>{}, _1{}, int64_t(kBM) * kBK, int64_t(kBM) * kBK * nbn,
                                   int64_t(kBM) * kBK * nbn * nbm);
    Tensor mA = make_tensor(make_gmem_ptr(static_cast<Element const*>(params.ds8_ptr)), shape_a, stride_a);
    TmaA tma_a = make_tma_copy(cute::SM90_TMA_LOAD{}, mA, SmemLayoutA{}(_, _, _0{}));
    ShapeB shape_b = make_shape(Int<kBD>{}, params.seqlen_k, params.b * params.h_k);
    StrideB stride_b = make_stride(params.kt_d_stride, _1{}, params.kt_head_stride);   // the API's K-twin strides (strided ring twins too)
    Tensor mB = make_tensor(make_gmem_ptr(static_cast<Element const*>(params.kt_ptr)), shape_b, stride_b);
    TmaBT<kCL> tma_b = make_tma_copy(std::conditional_t<(kCL > 1), cute::SM90_TMA_LOAD_MULTICAST, cute::SM90_TMA_LOAD>{},
                                     mB, SmemLayoutB{}(_, _, _0{}), Int<kCL>{});
    P2DGParams<kCL> p{tma_a, tma_b, shape_a, shape_b, params.psi_ptr, params.dq_ptr, params.dq_row_stride,
                      params.dq_head_stride, params.dq_batch_stride, params.scale_softmax / 256.0f, nbm, nbn,
                      params.h, params.h_k};
    auto* kernel = &fq8_dq_pass2_dg_kernel<Is_causal, kCL>;
    constexpr int kSmem = int(sizeof(SharedStorage));
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    dim3 grid(nbm, bh, 1);
    if (std::getenv("DM_DEBUG")) {
        fprintf(stderr, "[delta_matching pass 2 DG] smem=%d grid=(%d,%d) block=%d cluster=%d stages=%d\n",
                kSmem, nbm, bh, kThreads, kCL, kStages);
    }
    if constexpr (kCL == 1) {
        kernel<<<grid, kThreads, kSmem, stream>>>(p);
    } else {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid; cfg.blockDim = dim3(kThreads, 1, 1); cfg.dynamicSmemBytes = kSmem; cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1; attr[0].val.clusterDim.y = kCL; attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr; cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, kernel, p);
    }
    { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { fprintf(stderr, "[delta_matching pass 2 DG] launch: %s\n", cudaGetErrorString(e_)); exit(1); } }
}

}  // namespace fq8p2dg

// Cluster size over bh (heads of one GQA group share the K^T twin): DM_P2DG_CLUSTER = 1 / 2 / 4 (default 2: measured
// blk:1:16384:8192:f pass 2 1.244 ms at 2 vs 1.293 at 1 and 1.317 at 4).
template <bool Is_causal>
void run_fq8_dq_pass2_dg(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    static int const cl_req = [] { char const* e = std::getenv("DM_P2DG_CLUSTER"); return e ? std::atoi(e) : 2; }();
    int const nrep = params.h / params.h_k;
    if (cl_req >= 4 && nrep % 4 == 0) { fq8p2dg::run_cl<Is_causal, 4>(params, stream); }
    else if (cl_req >= 2 && nrep % 2 == 0) { fq8p2dg::run_cl<Is_causal, 2>(params, stream); }
    else { fq8p2dg::run_cl<Is_causal, 1>(params, stream); }
}
