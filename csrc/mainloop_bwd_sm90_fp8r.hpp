/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/
/******************************************************************************
 * ## Delta-Matching-128 v2 (M20): PASS 1 "P1-R" = the delta_matching fp8 split backward's dK/dV pass at head_dim 128
 * in FA3's OWN hd128 backward geometry (kBlockM 64, kBlockN 128, SdP_swapAB, Mma_dKV_is_RS, AtomLayoutNdKV 2), selected
 * by FA3FP11_HD128_P1R=1 (set in the release build; the P1-A collective in mainloop_bwd_sm90_fp8v10.hpp stays the
 * d256 path).
 *
 * What is the same as P1-A (the certified hd128 port), per element and per tile, so that dq / dk / dv / psi / the dS'
 * codes are EXPECTED BITWISE (measured, not argued; the wgmma operand-swap symmetry was probed first):
 *   - the block-scale contract ([128, 64]: q8s / do8s per 128 rows = 2 m-blocks, k8s / v8s per 64 keys = one WG),
 *   - the exp2 line with the F1 cliff clamp, the dS line with the k8s fold, the psi rule (tile amax / 448 across both
 *     WGs, the F2 zero-tile sentinel, the F2b emission floor), the e4m3 conversions, the dS' workspace + psi contract pass 2
 *     consumes ([BH, nbm, nbn, 64, 128] tiles, the same bytes), the M4 dV window (rescale by the clamped f ratio, promote
 *     every 8 m-tiles into the smem overlay in true units) and the plan-A dK per-tile temp + fold, the exit k8s unfold.
 * What changes (schedule / geometry / instruction count only):
 *   - S^T = K Q^T and dP^T = V dO^T per WG (A = the WG's 64 keys of the natural K / V tiles, B = the natural Q / dO stage):
 *     the same m64n64k32 atoms over the same k order, so every element is the bitwise transpose of P1-A's S / dP (probe).
 *   - dV = P^T dO and dK_tmp = dS'^T Q as RS wgmma (m64n128k32 x 2, the same atoms as P1-A's dV^T / dK^T gemms with M and
 *     N swapped): the A operands are the e4m3 codes STRAIGHT FROM THE REGISTER FRAGMENTS (permute_Cregs_fp8 register
 *     renaming, then the FA3 Aregs view), against the dOt / Qt twins (B, K-major in rows).  The twins' pi row order
 *     (SIGMA = [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] per 16) is EXACTLY the k order the fp8 A fragment carries (register
 *     h, k slot 4t'+j <-> accumulator column 8(j>>1) + 2t' + (j&1)), so the M17 producers and the op schema are unchanged
 *     and the contraction runs over the same k sequence as P1-A.
 *   - No P~^T / dS'^T exchange tiles, no byte scatters, no cross-WG PdS barriers, no smem A-operand reads for dV / dK.
 *   - The ONE transpose left (dS' natural for pass 2) is done per WG with 16 shfl + 16 prmt + 2 stmatrix.trans per thread
 *     into a private 64 x 64 SW64 half tile with its own TMA box store: no cross-WG synchronization on that path.
 *   - LSE / dPsum are read from the stage's smem rows per column (swapAB: 16 per thread), never held in registers.
 *   - The psi cross-WG max keeps one named barrier per tile (parity double-buffered slots: no other CTA-wide barrier
 *     orders the slot reads any more).
 *   - Producer warpgroup: FA3's 24 / 240 register split (no transposer warps).
 * DBG (FA3FP11_DBG): the same rows / rowsabs / cks instrumentation contract as P1-A (per (bh, n-tile, WG, row) dS row
 * sums in true units; the WG's 64-key sum is reduced 2 keys per thread -> 8 lanes -> 4 warps in a fixed order).
 ******************************************************************************/
#pragma once

#include <climits>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/barrier.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "softmax.h"
#include "utils.h"

static_assert(FA3FP11_M17_LSEK == 1, "Delta-Matching-128 v2 (P1-R) is written for the in-kernel lse*log2e (FA3FP11_M17_LSEK=1) build line");
// ## M20 micro-levers (bitwise-class schedule changes; the release build sets RFSKIP, DSDEFER stays off):
//   FA3FP11_P1R_DSDEFER: each WG's dS' half tile is double-buffered and its TMA box store is issued ONE TILE LATER, right
//     after the next tile's psi barrier (which every thread reaches only after its stmatrix writes + proxy fence), so the two
//     per-tile WG-local named barriers disappear; the elected thread drains store(t-2) before that barrier.
//   FA3FP11_P1R_RFSKIP: the dV window rescale (64 FMUL per thread) is skipped when the clamped ratio is exactly 1.0f (every
//     second m-block at hd128: one dO scale per 128 rows).  x * 1.0f is exact for normal x; the promote's FFMA flushes a
//     denormal window value either way.
#ifndef FA3FP11_P1R_DSDEFER
#define FA3FP11_P1R_DSDEFER 0
#endif
// ## M21 P1R_MASKDIAG (bitwise-class; the release build sets it): the causal mask is applied only to the m-blocks whose rows can
// see a masked key of this CTA's 128 keys (the two diagonal m-blocks; the seqlen_k tail masks every tile), instead of every tile.
// ## M21 TIMING PROBE ONLY (numerics wrong: psi from this warpgroup's half alone, no cross-WG barrier): the upper bound of
// decoupling the two warpgroups.  Never in a shipped build.
#ifndef FA3FP11_P1R_NOPSIBAR
#define FA3FP11_P1R_NOPSIBAR 0
#endif
// ## M23 candidate B, HALFPSI (a NEW NUMERICS ARM, default off): psi per (tile, warpgroup half) = the amax of
// this warpgroup's 64-key half tile / 448 (WG-local max over its four warps, a WG-local named barrier), published at
// ptr_psi[(tile) * 2 + wg]; no cross-warpgroup exchange or CTA-wide barrier per tile.  The dK fold and the F2b floor were
// per warpgroup already.  Pass 2 (fq8_dq_pass2_hd128.h, HALFPSI) consumes the two scales with two accumulators.
#ifndef FA3FP11_HALFPSI
#define FA3FP11_HALFPSI 0
#endif
// ## M21 P1R_UNROLL2 (bitwise-class, default off): the m-block loop unrolled by the Q / Qt stage period (2) so that every
// smem stage index (Q, LSE, Qt; dO / dPsum / dOt are single-stage) is a compile-time constant: the wgmma descriptors and the
// per-column statistic addresses become immediates instead of per-tile integer / uniform arithmetic.  Same arithmetic,
// same order, same pipeline waits (the runtime pipeline states still drive the barriers).
#ifndef FA3FP11_P1R_UNROLL2
#define FA3FP11_P1R_UNROLL2 0
#endif
// ## bf16-dP (NEW ARM, default off): SageAttention's SageBwd exempts exactly one of the seven attention
// matmuls from quantization, (5) dP = dO V^T, which it keeps in raw 16-bit while quantizing the other six
// (arXiv 2603.02170: "SageBwd retains dP = dOV^T in FP16 precision, while quantizing the remaining four matrix
// multiplications ... This design choice avoids error amplification through dS, dQ, and dK").  At SAGEDP=1 pass 1
// reads the RAW bf16 V and dO for that one gemm and its accumulator is already in true dP units, so the code-scale
// product drops out of the dS' line.  Everything else, including the delta (which stays the STALE rowsum(dO*O) of
// the quantized forward, as in SageBwd), is bit-identical to the certified line.  This is a NEW ARM, never a
// bitwise change.  At 0 the whole thing compiles out.
#ifndef FA3FP11_P1R_SAGEDP
#define FA3FP11_P1R_SAGEDP 0
#endif

namespace flash {

using namespace cute;

template <int Stages, int Stages_dO, int Stages_dOt, bool Is_causal_>
struct CollectiveMainloopBwdSm90Fp8R {
    using Element = cutlass::float_e4m3_t;
    // ## bf16-dP: SageBwd's single exemption. dP = dO V^T is computed in bf16 from the RAW
    // (pre-quantization) dO and V instead of their e4m3 codes; S, dV, dQ, dK, the delta and the dS' path are
    // untouched, so this is exactly matmul (5) of the seven attention
    // matmuls.  V and raw dO feed NOTHING else in this collective (dV reads the dO twin), so the fp8 tiles are
    // replaced rather than duplicated: +16 KB for V, +8 KB per dO stage.  At FA3FP11_P1R_SAGEDP=0 every line
    // below collapses to the certified head_dim-128 code.
#if FA3FP11_P1R_SAGEDP
    using ElementDp = cutlass::bfloat16_t;
#else
    using ElementDp = Element;
#endif
    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;
    using ClusterShape = Shape<_1, _1, _1>;

    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 128;
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockMh = kBlockM;    // interface parity with the v10 collective (the launch header reads it)
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    static constexpr int kQuantKeys = 64;       // one K/V scale group per warpgroup column half
    static constexpr int kQuantRowsQ = 128;     // one Q/dO scale block per two m-blocks
    static constexpr int kQShift = 1;
    static constexpr int kStages = Stages;
    static constexpr int kStages_dO = Stages_dO;
    static constexpr int kStages_qt = Stages;
    static constexpr int kStages_dOt = Stages_dOt;
    static_assert(kStages >= kStages_dO, "pass 1: the dO pipeline cannot be deeper than the Q pipeline");
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = false;
    static constexpr bool Has_softcap = false;
    static constexpr bool Varlen = false;
    static constexpr bool DeltaPass0 = false;
    static constexpr int kAbl = 0;
    static constexpr bool Q_dO_same_stages = kStages == kStages_dO;
    // When the stage counts match, Q and dO SHARE one pipeline object and therefore one transaction count.
    // That is only sound while raw dO has Q's element size, which bf16-dP breaks (bf16 dO vs e4m3 Q).
    static_assert(!(FA3FP11_P1R_SAGEDP && Q_dO_same_stages),
                  "bf16-dP needs FA3FP11_HD128_DOST != FA3FP11_HD128_QST: a shared Q/dO pipeline would size the bf16 dO TMA with Q's byte count");
    static constexpr bool SdP_swapAB = true;
    static constexpr bool dKV_swapAB = false;
    static constexpr int NumMmaWarpGroups = 2;
    static constexpr int AtomLayoutNdKV = 2;
    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;
    // FA3's bwd split for two MMA warpgroups: the producer warpgroup only issues TMA.
    static constexpr uint32_t LoadRegisterRequirement = 24;
    static constexpr uint32_t MmaRegisterRequirement = 240;

    using SeqlenInfo_t = flash::SeqlenInfoQK<Varlen, kBlockM>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local>;

    // S^T / dP^T: (M = kBlockN keys, N = kBlockM rows, K = d); the two warpgroups tile M (64 keys each).
    using TileShapeAtomSdP = Shape<Int<kBlockN>, Int<kBlockM>, Int<kHeadDim>>;
    using AtomLayoutSdP = Layout<Shape<Int<NumMmaWarpGroups>, _1, _1>>;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutSdP{}));
    // dP^T = V dO^T: same tile shape and atom layout, ElementDp operands (== TiledMmaSdP at SAGEDP=0).
    using TiledMmadP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<ElementDp, ElementDp, ElementAccum, TileShapeAtomSdP, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutSdP{}));
    // dV / dK: (M = kBlockN keys, N = d, K = kBlockM rows), RS (A from registers), the two warpgroups tile M.
    using TileShapeAtomdKV = Shape<Int<kBlockN>, Int<kHeadDim>, Int<kBlockM>>;
    using AtomLayoutdKV = Layout<Shape<Int<AtomLayoutNdKV>, _1, _1>>;
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutdKV{}));
    static_assert(CUTE_STATIC_V(size(TiledMmaSdP{})) == NumMmaThreads && CUTE_STATIC_V(size(TiledMmadKV{})) == NumMmaThreads);

    // Smem layouts (K-major everywhere; fp8 wgmma reads no MN-major operand).
    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages>{})));
    using SmemLayoutAtomdOdp = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementDp,
                                        Int<kBlockM>, Int<kHeadDim>>());
    using SmemLayoutdO = decltype(tile_to_shape(SmemLayoutAtomdOdp{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages_dO>{})));
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutAtomVdp = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementDp,
                                       Int<kBlockN>, Int<kHeadDim>>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomVdp{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutAtomQdOt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                        Int<kHeadDim>, Int<kBlockM>>());
    using SmemLayoutQt = decltype(tile_to_shape(SmemLayoutAtomQdOt{}, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_qt>{})));
    using SmemLayoutdOt = decltype(tile_to_shape(SmemLayoutAtomQdOt{}, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_dOt>{})));
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<cute::round_up(kBlockM, 64)>>>;
    // swapAB: the (keys, rows, stage) view broadcasting the per-row statistic along keys (FA3's SmemLayoutLSEMma).
    using SmemLayoutLSEMma = cute::Layout<cute::Shape<Int<kBlockN>, Int<kBlockM>, Int<kStages>>, cute::Stride<_0, _1, Int<cute::round_up(kBlockM, 64)>>>;
    // The dS' natural HALF tile of one warpgroup: kBlockM rows x 64 keys, 64-byte rows (SW64), stored by its own TMA box.
    static constexpr int kDsHalfN = kBlockN / NumMmaWarpGroups;
    using SmemLayoutAtomDS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                      Int<kBlockM>, Int<kDsHalfN>>());
    using SmemLayoutDS = decltype(tile_to_shape(SmemLayoutAtomDS{}, make_shape(Int<kBlockM>{}, Int<kDsHalfN>{})));
    static constexpr int kDsHalfElems = kBlockM * kDsHalfN;
    static_assert(cute::cosize_v<SmemLayoutDS> == kDsHalfElems, "P1-R: the dS' half tile must be dense");
    // dV persistent accumulator overlay: kBlockN x kHeadDim fp32, slot(i, thread) = i * NumMmaThreads + thread (conflict-free).
    static constexpr int kDvOverlayFloats = kBlockN * kHeadDim;

    using GmemTiledCopyQdO = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;
    using ShapedQaccum = cute::Shape<int32_t, int32_t, int32_t>;
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    using TMA_QdO = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}), TileShape_MNK{}, ClusterShape{}));
    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK{}, TileShape_MNK{}, ClusterShape{}));
    using TMA_V = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<ElementDp const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutV{}, TileShape_MNK{}, ClusterShape{}));
    using TMA_dOdp = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<ElementDp const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutdO{}), TileShape_MNK{}, ClusterShape{}));
    // The twins are [d, seq, head, batch] (seq contiguous, pi-ordered inside every 16-position window).
    using TileShape_TwinM = Shape<Int<kHeadDim>, _1, Int<kBlockM>>;
    using TMA_Qt = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQt{}), TileShape_TwinM{}, ClusterShape{}));
    using TMA_dOt = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutdOt{}), TileShape_TwinM{}, ClusterShape{}));
    // The dS' tile store: [BH, nbm, nbn, 64, 128] gmem (tile-contiguous), one (64 x 64) box per warpgroup.
    using ShapeDS8 = cute::Shape<Int<kBlockM>, Int<kBlockN>, int32_t, int32_t, int32_t>;
    using StrideDS8 = cute::Stride<Int<kBlockN>, _1, int64_t, int64_t, int64_t>;
    using TMA_dS8 = decltype(make_tma_copy(
        cute::SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeDS8{}, StrideDS8{}),
        SmemLayoutDS{}));

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
    using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
    using MainloopPipelineQt = typename cutlass::PipelineTmaAsync<kStages_qt>;
    using PipelineStateQt = typename MainloopPipelineQt::PipelineState;
    using MainloopPipelineDOt = typename cutlass::PipelineTmaAsync<kStages_dOt>;
    using PipelineStateDOt = typename MainloopPipelineDOt::PipelineState;

    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(SmemLayoutV{}) * cutlass::sizeof_bits_v<ElementDp> / 8);
    // The dO pipeline carried Q's transaction count while both were e4m3; under SAGEDP raw dO is bf16 and needs its own.
    static constexpr uint32_t TmaTransactionBytesdO = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutdO{})) * cutlass::sizeof_bits_v<ElementDp> / 8);
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);
    static constexpr uint32_t TmaTransactionBytesQt = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQt{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesdOt = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutdOt{})) * cutlass::sizeof_bits_v<Element> / 8);

    static constexpr size_t SmemAlignmentQKVdO = 128;

    // The epilogue unions its bf16 dK / dV staging (2 x 32 KB) over the FIRST 64 KB of this struct; the persistent dV
    // overlay (read back at the mainloop exit, before the epilogue) and the dS' half tiles sit past the 80-KB mark.
    struct TensorStorage : cute::aligned_struct<1024> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
        cute::array_aligned<ElementDp, cute::cosize_v<SmemLayoutV>, SmemAlignmentQKVdO> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<ElementDp, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQt>, SmemAlignmentQKVdO> smem_qt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdOt>, SmemAlignmentQKVdO> smem_dot;
        cute::array_aligned<ElementAccum, kDvOverlayFloats, 128> smem_dvacc;
        cute::array_aligned<Element, NumMmaWarpGroups * (1 + FA3FP11_P1R_DSDEFER) * kDsHalfElems, 1024> smem_ds;   // SW64 half tiles per WG (x2 under DSDEFER); 1024-B aligned: the swizzle anchors to the address
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_dpsum;
        cute::array_aligned<ElementAccum, 16, 16> smem_psired;   // 8 warps x 2 tile parities
#ifdef FA3FP11_DBG
        cute::array_aligned<ElementAccum, NumMmaWarpGroups * 4 * 2 * kBlockM, 16> smem_dbg_rows;   // [wg][warp][sum|abs][row]
#endif
    };
    static_assert(offsetof(TensorStorage, smem_dvacc) >= 64 * 1024, "P1-R: the dV overlay must sit past the epilogue's 64-KB staging window");

    // Host side kernel arguments: byte-compatible with the v10 collective's Arguments (the launch header brace-initializes it).
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* const ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        ElementDp const* const ptr_V;
        ShapeQKV const shape_V;
        StrideQKV const stride_V;
        ElementDp const* const ptr_dO;
        ShapeQKV const shape_dO;
        StrideQKV const stride_dO;
        Element const* const ptr_Qt;
        ShapeQKV const shape_Qt;
        StrideQKV const stride_Qt;
        Element const* const ptr_Kt;
        ShapeQKV const shape_Kt;
        StrideQKV const stride_Kt;
        Element const* const ptr_dOt;
        ShapeQKV const shape_dOt;
        StrideQKV const stride_dOt;
        float const* const ptr_q8s;
        float const* const ptr_k8s;
        float const* const ptr_v8s;
        float const* const ptr_do8s;
        int const nblk_q;
        int const nblk_k;
        ElementAccum* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        int const window_size_left, window_size_right, attention_chunk;
        float const softcap_val;
        int const num_batch;
        int* const dq_semaphore;
        int const* const cu_seqlens_q;
        int const* const cu_seqlens_k;
        int const* const seqused_q;
        int const* const seqused_k;
        Element* const ptr_dS8;
        float* const ptr_psi;
        int const nbm;
        int const nbn;
        float* const ptr_dltpart;
        float* const ptr_dbg_rows;
        float* const ptr_dbg_rowsabs;
        unsigned int* const ptr_dbg_cks;
    };

    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ShapeQKV const shape_V;
        ShapeQKV const shape_dO;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_QdO tma_load_Q;
        TMA_dOdp tma_load_dO;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        TMA_Qt tma_load_Qt;
        TMA_dOt tma_load_dOt;
        float const* const ptr_q8s;
        float const* const ptr_k8s;
        float const* const ptr_v8s;
        float const* const ptr_do8s;
        int const nblk_q;
        int const nblk_k;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        int const num_batch;
        int const* const cu_seqlens_q;
        int const* const cu_seqlens_k;
        int const* const seqused_q;
        int const* const seqused_k;
        TMA_dS8 tma_store_dS8;
        float* const ptr_psi;
        int const nbm;
        int const nbn;
        float* const ptr_dbg_rows;
        float* const ptr_dbg_rowsabs;
        unsigned int* const ptr_dbg_cks;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_QdO tma_load_Q = make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mQ, SmemLayoutQ{}(_, _, _0{}), TileShape_MNK{}, ClusterShape{});
        Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_dO, args.stride_dO);
        TMA_dOdp tma_load_dO = make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mdO, SmemLayoutdO{}(_, _, _0{}), TileShape_MNK{}, ClusterShape{});
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(GmemTiledCopyKV{}, mK, SmemLayoutK{}, TileShape_MNK{}, ClusterShape{});
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_V, args.stride_V);
        TMA_V tma_load_V = make_tma_copy_B_sm90(GmemTiledCopyKV{}, mV, SmemLayoutV{}, TileShape_MNK{}, ClusterShape{});
        Tensor mQt = make_tensor(make_gmem_ptr(args.ptr_Qt), args.shape_Qt, args.stride_Qt);
        TMA_Qt tma_load_Qt = make_tma_copy_A_sm90(cute::SM90_TMA_LOAD{}, mQt, take<0, 2>(SmemLayoutQt{}), TileShape_TwinM{}, ClusterShape{});
        Tensor mdOt = make_tensor(make_gmem_ptr(args.ptr_dOt), args.shape_dOt, args.stride_dOt);
        TMA_dOt tma_load_dOt = make_tma_copy_A_sm90(cute::SM90_TMA_LOAD{}, mdOt, take<0, 2>(SmemLayoutdOt{}), TileShape_TwinM{}, ClusterShape{});
        Tensor mDS8 = make_tensor(make_gmem_ptr(args.ptr_dS8),
            make_shape(Int<kBlockM>{}, Int<kBlockN>{}, args.nbn, args.nbm, args.num_batch * get<2>(args.shape_Q)),
            make_stride(Int<kBlockN>{}, _1{}, int64_t(kBlockM) * kBlockN,
                        int64_t(kBlockM) * kBlockN * args.nbn,
                        int64_t(kBlockM) * kBlockN * args.nbn * args.nbm));
        TMA_dS8 tma_store_dS8 = make_tma_copy(cute::SM90_TMA_STORE{}, mDS8, SmemLayoutDS{});
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        return {args.shape_Q, args.shape_K, args.shape_V, args.shape_dO,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_dO, tma_load_K, tma_load_V, tma_load_Qt, tma_load_dOt,
                args.ptr_q8s, args.ptr_k8s, args.ptr_v8s, args.ptr_do8s, args.nblk_q, args.nblk_k,
                args.ptr_LSE_log2, args.shape_LSE, args.stride_LSE_log2, args.ptr_dPsum, args.stride_dPsum,
                args.softmax_scale, args.window_size_left, args.window_size_right, attention_chunk_divmod,
                args.num_batch, args.cu_seqlens_q, args.cu_seqlens_k, args.seqused_q, args.seqused_k,
                tma_store_dS8, args.ptr_psi, args.nbm, args.nbn,
                args.ptr_dbg_rows, args.ptr_dbg_rowsabs, args.ptr_dbg_cks};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_Qt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dOt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_store_dS8.get_tma_descriptor());
    }

    // Producer (the M17 GT issue order per m-block: dO(m), dOt(m), Q(m+1), Qt(m+1); K / V one-shot on barrier_KV).
    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load_gt(Params const& params,
            MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
            MainloopPipelineQt pipeline_qt, MainloopPipelineDOt pipeline_dot,
            PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do,
            PipelineStateQt& smem_pipe_write_qt, PipelineStateDOt& smem_pipe_write_dot,
            SharedStorage &shared_storage, SchedulerPrefetch const& scheduler_prefetch,
            cute::tuple<int32_t, int32_t, int32_t> block_coord) {
        auto [n_block, bidh, bidb] = block_coord;
        SeqlenInfo_t seqlen_info{bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
                                 params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k};
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/);
        if constexpr (Is_causal) {
            if (m_block_max <= m_block_min) { scheduler_prefetch(); return; }
        }
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
        Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qt.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dot.data()), SmemLayoutdOt{});

        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, bidb);
        Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_dO)(_, _, bidh, bidb);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, bidb);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_V)(_, _, bidh_kv, bidb);
        auto shape_Qt = make_shape(get<1>(params.shape_Q), get<0>(params.shape_Q), get<2>(params.shape_Q), get<3>(params.shape_Q));
        auto shape_dOt = make_shape(get<1>(params.shape_dO), get<0>(params.shape_dO), get<2>(params.shape_dO), get<3>(params.shape_dO));
        Tensor mQt = params.tma_load_Qt.get_tma_tensor(shape_Qt)(_, _, bidh, bidb);
        Tensor mdOt = params.tma_load_dOt.get_tma_tensor(shape_dOt)(_, _, bidh, bidb);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, bidh, bidb);
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, bidh, bidb);

        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gdO = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));
        Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));
        Tensor gQt = local_tile(mQt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));
        Tensor gdOt = local_tile(mdOt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));
        Tensor gLSE = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mLSE), select<0>(TileShape_MNK{}), make_coord(_));
        Tensor gdPsum = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));

        Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
        Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
        Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
        Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
        auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
        auto block_tma_dO = params.tma_load_dO.get_slice(_0{});
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
        Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
        Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
        auto block_tma_Qt = params.tma_load_Qt.get_slice(_0{});
        auto block_tma_dOt = params.tma_load_dOt.get_slice(_0{});
        Tensor tQtgQt = group_modes<0, 3>(block_tma_Qt.partition_S(gQt));
        Tensor tQtsQt = group_modes<0, 3>(block_tma_Qt.partition_D(sQt));
        Tensor tdOtgdOt = group_modes<0, 3>(block_tma_dOt.partition_S(gdOt));
        Tensor tdOtsdOt = group_modes<0, 3>(block_tma_dOt.partition_D(sdOt));
        auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{}, group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));
        auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{}, group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));
        auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
        uint16_t const mcast_mask_qdo = 0;

        int m_block = m_block_min;
        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            pipeline_q.producer_acquire(smem_pipe_write);
            copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
            copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)), gLSE(_, m_block), sLSE(_, smem_pipe_write.index()));
            pipeline_qt.producer_acquire(smem_pipe_write_qt);
            copy(params.tma_load_Qt.with(*pipeline_qt.producer_get_barrier(smem_pipe_write_qt), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tQtgQt(_, m_block), tQtsQt(_, smem_pipe_write_qt.index()));
        }
        if (lane_predicate) {
            shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
            copy(params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0), tKgK, tKsK);
            copy(params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0), tVgV, tVsV);
            #pragma unroll 1
            for (; m_block < m_block_max - 1; ++m_block) {
                PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
                pipeline_do.producer_acquire(smem_pipe_write_do_cur);
                copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
                copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)), gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
                if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
                pipeline_dot.producer_acquire(smem_pipe_write_dot);
                copy(params.tma_load_dOt.with(*pipeline_dot.producer_get_barrier(smem_pipe_write_dot), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_dot.index()));
                ++smem_pipe_write_dot;
                ++smem_pipe_write;
                pipeline_q.producer_acquire(smem_pipe_write);
                copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tQgQ(_, m_block + 1), tQsQ(_, smem_pipe_write.index()));
                copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)), gLSE(_, m_block + 1), sLSE(_, smem_pipe_write.index()));
                ++smem_pipe_write_qt;
                pipeline_qt.producer_acquire(smem_pipe_write_qt);
                copy(params.tma_load_Qt.with(*pipeline_qt.producer_get_barrier(smem_pipe_write_qt), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tQtgQt(_, m_block + 1), tQtsQt(_, smem_pipe_write_qt.index()));
            }
        }
        scheduler_prefetch();
        if (lane_predicate) {
            PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
            pipeline_do.producer_acquire(smem_pipe_write_do_cur);
            copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
            copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)), gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
            pipeline_dot.producer_acquire(smem_pipe_write_dot);
            copy(params.tma_load_dOt.with(*pipeline_dot.producer_get_barrier(smem_pipe_write_dot), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_dot.index()));
            ++smem_pipe_write_dot;
            ++smem_pipe_write;
            ++smem_pipe_write_qt;
        }
        if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
    }

    CUTLASS_DEVICE void
    load_tail_gt(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
                 MainloopPipelineQt pipeline_qt, MainloopPipelineDOt pipeline_dot,
                 PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do,
                 PipelineStateQt& smem_pipe_write_qt, PipelineStateDOt& smem_pipe_write_dot) {
        if (cute::elect_one_sync()) {
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
            pipeline_qt.producer_tail(smem_pipe_write_qt);
            pipeline_dot.producer_tail(smem_pipe_write_dot);
        }
    }

    CUTLASS_DEVICE void mma_init() {}

    template <typename SharedStorage, typename FrgTensordKV>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
        MainloopPipelineQt pipeline_qt, MainloopPipelineDOt pipeline_dot,
        PipelineState& smem_pipe_read, PipelineState_dO& smem_pipe_read_do,
        PipelineStateQt& smem_pipe_read_qt, PipelineStateDOt& smem_pipe_read_dot,
        FrgTensordKV& tdKrdK, FrgTensordKV& tdVrdV,
        int thread_idx, int &work_idx,
        cute::tuple<int32_t, int32_t, int32_t> block_coord, SharedStorage& shared_storage) {
        static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");
        int const n_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
                                 params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k};
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/);
        if constexpr (Is_causal) {
            if (m_block_max <= m_block_min) { return false; }
        }
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qt.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dot.data()), SmemLayoutdOt{});
        Tensor sLSEMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});

        int const warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        int const wg_thread = thread_idx % cutlass::NumThreadsPerWarpGroup;
        int const lane = thread_idx % cutlass::NumThreadsPerWarp;
        int const warp_in_wg = wg_thread / cutlass::NumThreadsPerWarp;
        uint32_t const wg_barrier_id = static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) + warp_group_idx;   // ids 5, 6: WG-local
        // This warpgroup's dS' natural half tile (kBlockM rows x 64 keys, SW64) and its gmem box.
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data() + warp_group_idx * (1 + FA3FP11_P1R_DSDEFER) * kDsHalfElems), SmemLayoutDS{});
        Tensor sdS1 = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data() + (warp_group_idx * (1 + FA3FP11_P1R_DSDEFER) + FA3FP11_P1R_DSDEFER) * kDsHalfElems), SmemLayoutDS{});   // the second buffer (== sdS without DSDEFER)

        Layout warp_group_thread_layout = make_layout(make_shape(Int<NumMmaWarpGroups>{}), make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        TiledMmaSdP tiled_mma_SdP;
        TiledMmadP tiled_mma_dP;
        TiledMmadKV tiled_mma_dKV;
        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dP = tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));

        Tensor tSrK = wg_mma_SdP.partition_fragment_A(sK);          // this WG's 64 keys (MMA, MMA_M, MMA_K)
        Tensor tSrQ = wg_mma_SdP.partition_fragment_B(sQ);          // (MMA, MMA_N, MMA_K, PIPE)
        Tensor tdPrV = wg_mma_dP.partition_fragment_A(sV);
        Tensor tdPrdO = wg_mma_dP.partition_fragment_B(sdO);
        Tensor tdVrdOt = wg_mma_dKV.partition_fragment_B(sdOt);     // (MMA, MMA_N, MMA_K, PIPE): (d, rows)
        Tensor tdKrQt = wg_mma_dKV.partition_fragment_B(sQt);
        // Per-column (query row) statistics of the swapAB fragment: (2, V, MMA_N, PIPE) -> (16, PIPE); read from smem at use.
        Tensor tLSEsLSE = group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _, _));
        Tensor tLSEsdPsum = group_modes<0, 3>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_, _0{}, _), _0{}, _, _));
        static_assert(decltype(size<0>(tLSEsLSE))::value == 16, "P1-R: 16 query rows per thread in the swapAB fragment");

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        flash::Mask<kBlockM, kBlockN, false /*PackGQA*/, TiledMmaSdP, SdP_swapAB> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod);

        int m_block = m_block_min;
        clear(tdKrdK);
        clear(tdVrdV);
        static_assert(decltype(size(tdVrdV))::value == 64 && decltype(size(tdKrdK))::value == 64, "P1-R: 64-value dKV fragments per thread");
        float* const fq8_smem_dv = shared_storage.tensors.mainloop.smem_dvacc.data() + thread_idx;   // slot(i) = i * NumMmaThreads + thread
        #pragma unroll
        for (int i = 0; i < size(tdVrdV); ++i) { fq8_smem_dv[i * NumMmaThreads] = 0.0f; }

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.pipelines.barrier_KV.wait(work_idx % 2); }

        // Scales: this WG's 64-key group (k8s / v8s), the Q / dO scale per 128 rows (m-block >> 1).
        int const fq8_bh = bidb * get<2>(params.shape_Q) + bidh;
        int const fq8_bh_kv = bidb * get<2>(params.shape_K) + params.qhead_per_khead_divmod.divide(bidh);
        float const fq8_k8s_wg = params.ptr_k8s[fq8_bh_kv * params.nblk_k + 2 * n_block + warp_group_idx];
        [[maybe_unused]] float const fq8_v8s_wg = params.ptr_v8s[fq8_bh_kv * params.nblk_k + 2 * n_block + warp_group_idx];   // dP descale; unread under bf16-dP
        float const* fq8_q8s_base = params.ptr_q8s + fq8_bh * params.nblk_q;
        float const* fq8_do8s_base = params.ptr_do8s + fq8_bh * params.nblk_q;
        auto fq8_qblk = [](int m) { return m >> kQShift; };
        float fq8_q8sc_next = fq8_q8s_base[fq8_qblk(m_block_min)];
        float fq8_do8sc_next = fq8_do8s_base[fq8_qblk(m_block_min)];
        float fq8_f_prev = 1.0f;     // dV window basis (2^-8 * do8sc(i))
        float fq8_g_max = 0.0f;      // row-global max of the dK basis (the F2b emission floor anchor)
        float* fq8_psired = shared_storage.tensors.mainloop.smem_psired.data();
        static constexpr float kFq8MoE = 8.0f;
        static constexpr float kFq8MoInv = 1.0f / 256.0f;
        constexpr int kFq8PromoteTiles = 8;   // the 512-row dV promote window (pinned, as P1-A)
        auto fq8_clamp_ratio = [](float r) { return fminf(fmaxf(r, 1e-8f), 1e8f); };

        // The dS' natural transpose bookkeeping (per lane, per CTA): lane holds key pair ip = lane / 4 of its warp's 16
        // keys; the two source lanes hold keys 2 ip (srcA) and 2 ip + 1 (srcB) in slot h = ip / 4.
        int const fq8_ip = lane >> 2;
        int const fq8_srcA = 8 * (fq8_ip & 3) + (lane & 3);
        int const fq8_srcB = fq8_srcA + 4;
        uint32_t const fq8_sel = fq8_ip < 4 ? 0x5140u : 0x7362u;
        // stmatrix.x4.trans row addresses: instruction s covers rows 32 s + lane of this warp's 16-key chunk.
        uint32_t const fq8_st_addr0 = cute::cast_smem_ptr_to_uint(&sdS(lane, 16 * warp_in_wg));
        uint32_t const fq8_st_addr1 = cute::cast_smem_ptr_to_uint(&sdS(32 + lane, 16 * warp_in_wg));
        uint32_t const fq8_st_addr0b = cute::cast_smem_ptr_to_uint(&sdS1(lane, 16 * warp_in_wg));
        uint32_t const fq8_st_addr1b = cute::cast_smem_ptr_to_uint(&sdS1(32 + lane, 16 * warp_in_wg));
        Tensor mdS8 = params.tma_store_dS8.get_tma_tensor(
            make_shape(Int<kBlockM>{}, Int<kBlockN>{}, params.nbn, params.nbm, params.num_batch * get<2>(params.shape_Q)));
        auto blk_tma_dS8 = params.tma_store_dS8.get_slice(_0{});
        Tensor tdS8s = blk_tma_dS8.partition_S(sdS);
        Tensor tdS8s1 = blk_tma_dS8.partition_S(sdS1);
        // The WG's box store of tile m (buffer m & 1 under DSDEFER; the single buffer otherwise), issued by the elected thread.
        auto fq8_issue_ds_store = [&](int m) {
            Tensor gdS8 = mdS8(_, _, n_block, m, fq8_bh);
            Tensor gdS8h = local_tile(gdS8, Shape<Int<kBlockM>, Int<kDsHalfN>>{}, make_coord(_0{}, warp_group_idx));
            Tensor tdS8g = blk_tma_dS8.partition_D(gdS8h);
            if (FA3FP11_P1R_DSDEFER && (m & 1)) { cute::copy(params.tma_store_dS8, tdS8s1, tdS8g); }
            else { cute::copy(params.tma_store_dS8, tdS8s, tdS8g); }
            cute::tma_store_arrive();
        };

        auto bwd_step = [&](int m_block, auto mask_fn, auto stage_c) {
            constexpr int kSt = decltype(stage_c)::value;   // compile-time Q / Qt stage under UNROLL2 (-1 = runtime)
            Tensor fq8_tmpK = make_fragment_like(tdKrdK);
            float fq8_g_fold = 0.0f;
            // (kBlockN, kBlockM) = the swapAB fragment: rows = keys, columns = query rows.
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<1, 0>(TileShape_MNK{}));
            consumer_wait(pipeline_q, smem_pipe_read);
            if constexpr (kSt >= 0) {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tSrK, tSrQ(_, _, _, Int<kSt>{}), tSrS);
            } else {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_SdP, tSrK, tSrQ(_, _, _, smem_pipe_read.index()), tSrS);
            }
            float const fq8_q8sc = fq8_q8sc_next;
            float const fq8_do8sc = fq8_do8sc_next;
            if (m_block + 1 < m_block_max) {
                fq8_q8sc_next = fq8_q8s_base[fq8_qblk(m_block + 1)];
                fq8_do8sc_next = fq8_do8s_base[fq8_qblk(m_block + 1)];
            }
            float const fq8_s_descale_log2 = fq8_q8sc * fq8_k8s_wg * float(M_LOG2E);
            // dP accumulator -> TRUE dP units.  With e4m3 operands that is the product of their code scales;
            // under bf16-dP the operands are the raw bf16 tensors, so the accumulator is already true dP and the
            // only surviving factor is pass 1's k8s fold (dS' = 2^8 * P * (dP_true - delta) * k8s_wg).
#if FA3FP11_P1R_SAGEDP
            float const fq8_dp_descale = 1.0f;
#else
            float const fq8_dp_descale = fq8_do8sc * fq8_v8s_wg;
#endif
            float const fq8_dp_descale_f = fq8_dp_descale * fq8_k8s_wg;
            Tensor tdPrdP = partition_fragment_C(tiled_mma_dP, select<1, 0>(TileShape_MNK{}));
            // dS rides on tdPrdP under tSrS's rowcol layout below, so the two accumulators must agree.
            static_assert(cute::is_same_v<decltype(partition_fragment_C(TiledMmadP{}, select<1, 0>(TileShape_MNK{}))),
                                          decltype(partition_fragment_C(TiledMmaSdP{}, select<1, 0>(TileShape_MNK{})))>,
                          "bf16-dP: the bf16 dP accumulator fragment must match the fp8 S one");
            PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
            consumer_wait(pipeline_do, smem_pipe_read_do_cur);
            if constexpr (kSt >= 0) {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dP, tdPrV, tdPrdO(_, _, _, Int<0>{}), tdPrdP);
            } else {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dP, tdPrV, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrdP);
            }
            warpgroup_wait<1>();   // S retired
            auto const stage_q = [&] { if constexpr (kSt >= 0) { return Int<kSt>{}; } else { return smem_pipe_read.index(); } }();
            // rowcol view: (16 query rows, 2 keys) per thread.
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
            mask_fn(tSrS, m_block);
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
                float const lse_scaled = __fmul_rn(tLSEsLSE(mi, stage_q), float(M_LOG2E));   // LSEK: the host prep's exact multiply
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
                    // P-hat = 2^8 * exp(S - lse) with the F1 cliff clamp (P1-A's line, verbatim).
                    scores(mi, ni) = exp2f(fminf(scores(mi, ni) * fq8_s_descale_log2 - lse_scaled + kFq8MoE, kFq8MoE + 4.0f));
                }
            }
            // The stage's LSE rows were read by the loop above and the S gemm has retired: raw Q's readers are done.
            pipeline_q.consumer_release(smem_pipe_read);
            warpgroup_wait<0>();   // dP retired
            auto const stage_do = [&] { if constexpr (kSt >= 0) { return Int<0>{}; } else { return smem_pipe_read_do_cur.index(); } }();
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
#ifdef FA3FP11_DBG
            if (params.ptr_dbg_cks != nullptr) {
                unsigned int fq8_xp = 0u, fq8_xd = 0u;
                #pragma unroll
                for (int i = 0; i < size(tSrS); ++i) { fq8_xp ^= __float_as_uint(tSrS(i)); }
                #pragma unroll
                for (int i = 0; i < size(tdPrdP); ++i) { fq8_xd ^= __float_as_uint(tdPrdP(i)); }
                #pragma unroll
                for (int off = 16; off > 0; off /= 2) {
                    fq8_xp ^= __shfl_xor_sync(0xffffffff, fq8_xp, off);
                    fq8_xd ^= __shfl_xor_sync(0xffffffff, fq8_xd, off);
                }
                if (lane == 0) {
                    int64_t const fq8_nt = int64_t(params.num_batch) * get<2>(params.shape_Q) * params.nbm * params.nbn;
                    int64_t const fq8_ti = (int64_t(fq8_bh) * params.nbm + m_block) * params.nbn + n_block;
                    atomicXor(params.ptr_dbg_cks + fq8_ti, fq8_xp);
                    atomicXor(params.ptr_dbg_cks + fq8_nt + fq8_ti, fq8_xd);
                }
            }
            if (params.ptr_dbg_rows != nullptr) {
                // Per-(tile, WG, query row) sums of the realized dS in TRUE units: 2 keys per thread, then the 8 lanes of
                // a warp that share the column (lane bits 2..4), then the 4 warps in a fixed order through smem.
                float* dbg_smem = shared_storage.tensors.mainloop.smem_dbg_rows.data() + warp_group_idx * (4 * 2 * kBlockM);
                Tensor tScS = thread_mma_SdP.partition_C(cute::make_identity_tensor(select<1, 0>(TileShape_MNK{})));
                Tensor tScS_rc = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tScS.layout()));
                #pragma unroll
                for (int mi = 0; mi < size<0>(dS); ++mi) {
                    float const dsum = tLSEsdPsum(mi, stage_do);
                    float ra = 0.f, rb = 0.f;
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) {
                        float const t = scores(mi, ni) * (dS(mi, ni) * fq8_dp_descale - dsum);
                        ra += t; rb += fabsf(t);
                    }
                    #pragma unroll
                    for (int off = 4; off < 32; off *= 2) {
                        ra += __shfl_xor_sync(0xffffffff, ra, off);
                        rb += __shfl_xor_sync(0xffffffff, rb, off);
                    }
                    if ((lane >> 2) == 0) {
                        int const row = get<1>(tScS_rc(mi, _0{}));   // the query row of this column
                        dbg_smem[(warp_in_wg * 2 + 0) * kBlockM + row] = ra;
                        dbg_smem[(warp_in_wg * 2 + 1) * kBlockM + row] = rb;
                    }
                }
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
                if (wg_thread < kBlockM) {
                    float sa = 0.f, sb = 0.f;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        sa += dbg_smem[(w * 2 + 0) * kBlockM + wg_thread];
                        sb += dbg_smem[(w * 2 + 1) * kBlockM + wg_thread];
                    }
                    int64_t const gi = ((int64_t(fq8_bh) * params.nbn + n_block) * 2 + warp_group_idx) * int64_t(seqlen_q) + m_block * kBlockM + wg_thread;
                    params.ptr_dbg_rows[gi] = sa;
                    params.ptr_dbg_rowsabs[gi] = sb;
                }
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
            }
#endif
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                float const dP_sum_cur_f = tLSEsdPsum(mi, stage_do) * fq8_k8s_wg;
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) {
                    // dS' = 2^8 * P * (dP_true - delta) * k8s_wg (the k8s fold; P1-A's line, verbatim).
                    dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) * fq8_dp_descale_f - dP_sum_cur_f);
                }
            }
            // The stage's dPsum rows were read above and the dP gemm has retired: raw dO's readers are done.
            pipeline_do.consumer_release(smem_pipe_read_do_cur);
            // psi = amax(|dS_tile|) / 448 across BOTH warpgroups (the tile is column-split per WG); parity-buffered slots.
            float fq8_amax = 0.f;
            #pragma unroll
            for (int i = 0; i < size(tdPrdP); ++i) { fq8_amax = fmaxf(fq8_amax, fabsf(tdPrdP(i))); }
            #pragma unroll
            for (int off = 16; off > 0; off /= 2) { fq8_amax = fmaxf(fq8_amax, __shfl_xor_sync(0xffffffff, fq8_amax, off)); }
            float* const fq8_psi_slots = fq8_psired + (m_block & 1) * 8;
            if (lane == 0) { fq8_psi_slots[thread_idx / cutlass::NumThreadsPerWarp] = fq8_amax; }
            cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG3) /*id 4*/);
            #pragma unroll
            for (int w = 0; w < NumMmaThreads / cutlass::NumThreadsPerWarp; ++w) { fq8_amax = fmaxf(fq8_amax, fq8_psi_slots[w]); }
            bool const fq8_tile_zero = fq8_amax * (1.0f / 448.0f) < 1e-30f;
            float const fq8_psi_raw = fmaxf(fq8_amax * (1.0f / 448.0f), 1e-30f);
            float const fq8_psi = fmaxf(fq8_psi_raw, fq8_g_max * (256.0f * 1e-8f) / fq8_q8sc);
            float const fq8_psi_inv = fq8_tile_zero ? 0.0f : 1.0f / fq8_psi;
            if (thread_idx == 0) {
                params.ptr_psi[(int64_t(fq8_bh) * params.nbm + m_block) * params.nbn + n_block] = fq8_tile_zero ? 0.0f : fq8_psi;
            }
            // P~ codes as the RS A operand of dV (register renaming + the FA3 Aregs view; no smem).
            flash::permute_Cregs_fp8(tSrS);
            Tensor tdVrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
            Tensor rP = make_tensor_like<Element>(tdVrP_acc);
            flash::convert_type_out(tdVrP_acc, rP);
            // dS' codes = e4m3(dS / psi) as the RS A operand of dK.
            #pragma unroll
            for (int i = 0; i < size(tdPrdP); ++i) { tdPrdP(i) *= fq8_psi_inv; }
            flash::permute_Cregs_fp8(tdPrdP);
            Tensor tdKrdS_acc = make_tensor(tdPrdP.data(), flash::convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
            Tensor rdS = make_tensor_like<Element>(tdKrdS_acc);
            flash::convert_type_out(tdKrdS_acc, rdS);
            static_assert(decltype(size(rdS))::value == 32, "P1-R: 32 dS' codes per thread per tile");
            // dS' natural (rows x keys) for pass 2: the ONE transpose.  A-register word 2i = (h0: g 2i, g 2i+1), word
            // 2i+1 = (h1: g 2i, g 2i+1); W[g] = (h0 pair(g) | h1 pair(g) << 16) is the natural word; the key pair this lane
            // stores comes from lanes srcA / srcB (slot h = ip / 4); stmatrix.trans writes 16 keys x 8 rows per matrix.
            {
                Tensor rdS32 = recast<uint32_t>(rdS);
                uint32_t fq8_out[8];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t const w0 = __byte_perm(rdS32(2 * i), rdS32(2 * i + 1), 0x5410);   // g = 2i
                    uint32_t const w1 = __byte_perm(rdS32(2 * i), rdS32(2 * i + 1), 0x7632);   // g = 2i + 1
                    uint32_t const a0 = __shfl_sync(0xffffffff, w0, fq8_srcA);
                    uint32_t const b0 = __shfl_sync(0xffffffff, w0, fq8_srcB);
                    uint32_t const a1 = __shfl_sync(0xffffffff, w1, fq8_srcA);
                    uint32_t const b1 = __shfl_sync(0xffffffff, w1, fq8_srcB);
                    fq8_out[2 * i] = __byte_perm(a0, b0, fq8_sel);
                    fq8_out[2 * i + 1] = __byte_perm(a1, b1, fq8_sel);
                }
                // The previous tile's TMA box store of this WG must have drained before the tile is overwritten.
                if (wg_thread == 0) { cute::tma_store_wait<0>(); }
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
                asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                             :: "r"(fq8_st_addr0), "r"(fq8_out[0]), "r"(fq8_out[1]), "r"(fq8_out[2]), "r"(fq8_out[3]));
                asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                             :: "r"(fq8_st_addr1), "r"(fq8_out[4]), "r"(fq8_out[5]), "r"(fq8_out[6]), "r"(fq8_out[7]));
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, wg_barrier_id);
                if (wg_thread == 0) { fq8_issue_ds_store(m_block); }
            }
            // dV window rescale (v10 / P1-A verbatim), the dK fold basis and the F2b anchor.
            float const fq8_f_cur = kFq8MoInv * fq8_do8sc;
            float const fq8_rf = fq8_clamp_ratio(fq8_f_prev / fq8_f_cur);
            if (fq8_rf != 1.0f) {   // WG-uniform: the ratio is exactly 1.0f on the second m-block of every 128-row dO scale block
            #pragma unroll
            for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) *= fq8_rf; }
            }
            fq8_f_prev = fq8_f_cur;
            fq8_g_fold = 0.0f;
            if (!fq8_tile_zero) {
                float const fq8_g_cur = fq8_psi * kFq8MoInv * fq8_q8sc;
                fq8_g_fold = fq8_g_cur;
                fq8_g_max = fmaxf(fq8_g_max, fq8_g_cur);
            }
            // dV += P^T dO (RS), dK_tmp = dS'^T Q (RS, zero-init); FIFO retire.
            consumer_wait(pipeline_dot, smem_pipe_read_dot);
            if constexpr (kSt >= 0) {
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, rP, tdVrdOt(_, _, _, Int<0>{}), tdVrdV);
            } else {
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, rP, tdVrdOt(_, _, _, smem_pipe_read_dot.index()), tdVrdV);
            }
            consumer_wait(pipeline_qt, smem_pipe_read_qt);
            if constexpr (kSt >= 0) {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dKV, rdS, tdKrQt(_, _, _, Int<kSt>{}), fq8_tmpK);
            } else {
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dKV, rdS, tdKrQt(_, _, _, smem_pipe_read_qt.index()), fq8_tmpK);
            }
            warpgroup_wait<1>();
            pipeline_dot.consumer_release(smem_pipe_read_dot);
            ++smem_pipe_read_dot;
            warpgroup_wait<0>();
            pipeline_qt.consumer_release(smem_pipe_read_qt);
            ++smem_pipe_read_qt;
            ++smem_pipe_read;
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
            // plan-A dK fold: one fp32 FMA per element per tile in the tile's own g basis.
            #pragma unroll
            for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) = fmaf(fq8_tmpK(i), fq8_g_fold, tdKrdK(i)); }
            // dV window boundary (geometry-fixed): promote in TRUE units into the overlay, re-zero the window.
            if ((m_block & (kFq8PromoteTiles - 1)) == kFq8PromoteTiles - 1 || m_block == m_block_max - 1) {
                #pragma unroll
                for (int i = 0; i < size(tdVrdV); ++i) {
                    float* const slot = fq8_smem_dv + i * NumMmaThreads;
                    *slot = fmaf(tdVrdV(i), fq8_f_prev, *slot);
                    tdVrdV(i) = 0.0f;
                }
            }
        };

        auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); };
        auto no_mask_fn = [&](auto& tSrS, int m_block) {};
        // Tiles whose rows all lie at or past this CTA's last key (m_block * 64 + (seqlen_k - seqlen_q) >= (n_block + 1) * 128)
        // have no maskable element; the seqlen_k tail (never at the training shapes) keeps the mask everywhere.
        constexpr int kbm_ = kBlockM, kbn_ = kBlockN;
        int const fq8_mask_m_end = ((n_block + 1) * kbn_ > seqlen_k) ? INT32_MAX
                                   : ((n_block + 1) * kbn_ - (seqlen_k - seqlen_q) + kbm_ - 1) / kbm_;
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max && m_block < fq8_mask_m_end; ++m_block) { bwd_step(m_block, mask_fn, Int<-1>{}); }
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max; ++m_block) { bwd_step(m_block, no_mask_fn, Int<-1>{}); }

        // Drain the last dS' box store (non-issuing threads return immediately).
        cute::tma_store_wait<0>();
        // Exit: the k8s unfold of dK (this WG's keys are one scale group) and the dV read-back from the overlay.
        float const fq8_ik = params.softmax_scale / fq8_k8s_wg;
        #pragma unroll
        for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= fq8_ik; }
        #pragma unroll
        for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) = fq8_smem_dv[i * NumMmaThreads]; }
        if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
        ++work_idx;
        return true;
    }
};

}  // namespace flash
