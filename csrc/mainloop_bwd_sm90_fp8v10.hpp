/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

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
#include "copy_sm90_bulk_reduce.hpp"

// ## FP11-M16 lever macros (the build passes them explicitly; 0 = the M10 code path).
// ## FP11-M16 ER: release the RAW Q / dO stages right after the S / dP gemms
// retire (gated by the transposer's twin commits), not mid/late tile.
// ## FP11-M16 Q3: raw Q at 3 stages (launch header) with the in-smem Q twin at
// ONE stage (the transposer refills it per tile), net-zero smem.
// ## FP11-M16 TWINPERM: the in-smem Q^T / dO^T twins keep the LDSM.T output's
// native row order (actual d = 2r + s stored at physical row r + 8s inside
// each 16-row group) instead of the fq8_dcorr un-permute, so each STSM.x4
// matrix writes 8 CONSECUTIVE 64-B rows (bank-conflict-free under SW64; the
// dcorr placement put them 128 B apart = 2-way conflicts, 8 wavefronts per
// STSM measured).  The dV / dK accumulator rows are then in that permuted
// order; they are un-permuted ONCE per CTA at the exit (dV: read the smem
// overlay slot of the permuted row; dK: one staging pass through the same
// overlay).  Data movement only: every (d, n) output element is the same
// wgmma dot product on the same operands in the same k order.
// ## FP11-M16 TILP: transposer LDSM.T/STSM blocks in flight per iteration (M10: 2).
#ifndef FA3FP11_M16_TILP
#define FA3FP11_M16_TILP 2
#endif
// ## FP11-M16 ABLATION bitmask (TIMING-ONLY builds, numerically garbage, never
// shipped): 1 = no dV/dK gemms, 2 = no psi cross-WG reduction (warp-local amax,
// no named barrier), 4 = no exchange stores / PdS barrier / dS8 TMA store,
// 8 = no S/dP gemms, 16 = no transposer LDSM/STSM, 32 = no exp2/dS arithmetic.
#ifndef FA3FP11_ABL
#define FA3FP11_ABL 0
#endif
// ## FP11-M17 GT (gmem twins, bitwise): the Q^T / dO^T wgmma-ready twins arrive
// from GLOBAL memory by TMA (producer-side twins written by the quantizers with
// the m axis in the transposer's stmatrix interleave order pi, so the smem
// bytes are IDENTICAL to what the in-smem transposer produced) instead of being
// built in smem by the load warpgroup's warps 2-3 (LDSM.T + byte_perm + STSM,
// the one pass-1 chain link whose deletion helped in the M16 ablation: 0.72 ms
// at B2 H16 N8192).  The qt / dot pipelines become PipelineTmaAsync fed by the
// producer warp; the transposer warps idle; the dead Kt one-shot load is
// dropped (pass 1 never reads Kt since V10) and its 16 KB funds a second dOt
// stage (FA3FP11_M17_DOTST).  Raw Q / dO are released right after their S / dP
// gemms retire (the transposer no longer reads them).  0 = the M16 code path.
// ## FP11-M17 LSEK (lever F, bitwise): the kernels take the RAW natural-log LSE
// and form lse*log2(e) in-register with a non-contractible fp32 multiply
// (__fmul_rn: the same single IEEE multiply the host prep `lse * float(M_LOG2E)`
// performed), so the host prep kernel and its allocation disappear from every
// backward op.  0 = the host-prepared lse_log2 buffer (M16).

namespace flash {

using namespace cute;

// ## FP11-M5: DeltaPass0_ = the in-kernel matched-delta prepass mode (the design notes
// D3).  false (default) = the certified pass-1 mainloop, TOKEN-IDENTICAL to
// M4 (every M5 change to the pass-1 path is an appended-defaulted template
// parameter, appended Arguments/Params fields, or code behind
// `if constexpr (DeltaPass0)` / `#ifdef FA3FP11_DBG`).  true = "pass 0": the
// SAME collective (same smem staging layouts, same TMA copies, same
// TiledMmaSdP atom split and zero-init 8-atom chains, same descale ordering,
// same exp2 + F1 clamp, same grid/scheduler/producer) with the consumer
// swapped to mma_pass0(): S~ and dP~ are recomputed bit-identically to
// pass 1's and reduced into per-CTA per-row partial deltas
//   partial[bh][n_block][i] = 2^-8 * sum_{ni in this CTA's 64 keys}
//                               P~(i,ni) * (dP~(i,ni) * do8sc*v8s(ni-half))
// (P~ = the fp32 exp2 recompute = 2^8*P; the k8s fold and psi never enter
// delta), written per warpgroup half to the [BH, nbn, 2, N] fp32
// workspace that the pass-0.5
// fixed-order reduce (fp11_delta.h) sums into delta.  All dV/dK/dS'-store/
// psi/transpose machinery is compiled out; the lean TensorStorageDelta drops
// those buffers so pass 0 runs 2 CTAs/SM.
template <int Stages, int Stages_dO, int Stages_dS, class ClusterShape_, class TileShape_MNK_, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool Deterministic,
        bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
        int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
        bool Mma_dP_is_RS=false, bool DeltaPass0_=false>
struct CollectiveMainloopBwdSm90Fp8V10 {
    // ## FQ8 (fa3fp8): the 7/7-fp8 d256 bwd on FA3's schedule.  Element is e4m3;
    // every transposed operand that bf16 read as an MN-major stride-view becomes a
    // real K-major buffer: Q^T/K^T/dO^T arrive as GMEM TWINS (fused_quant want_t8s),
    // P-hat^T/dS^T are written transposed in-smem from the SdP accumulator.  See
    // the development notes for the operand-major resolution table.
    static_assert(cute::is_same_v<Element_, cutlass::float_e4m3_t>, "delta_matching: Element must be e4m3");
    // ## Delta-Matching-128 (M19): kBlockN = two scale groups (2 x 32 keys at d256; 2 x 64 keys at hd128,
    // the Triton reference's [128, 64] blocks); each MMA warpgroup owns one group = its column half, as at d256.
    static_assert(get<1>(TileShape_MNK_{}) == 2 * (get<2>(TileShape_MNK_{}) >= 256 ? 32 : 64),
                  "delta_matching: kBlockN must be two scale groups (64 at d256, 128 at hd128)");
    static_assert(!SdP_swapAB_ && dKV_swapAB_ && dQ_swapAB_, "delta_matching operand-major plan");
    static_assert(NumMmaWarpGroups == 2 && AtomLayoutMSdP == 1 && AtomLayoutNdKV == 1 && AtomLayoutMdQ == 1,
                  "delta_matching: built for the 2-warpgroup shape");
    static_assert(Stages_dS == 1, "delta_matching: single-stage PdS exchanges");

    static constexpr int kStages = Stages;
    static constexpr int kStages_dO = Stages_dO;
    static constexpr int kStages_dS = Stages_dS;
    // ## FP11-M16 DO2 (staging only, bitwise): pass 1's RAW dO pipeline runs
    // at Stages_dO = 2 (the launch header) while the in-smem transposed twin
    // dOt keeps ONE stage (the transposer refills it every tile).  ncu at
    // B2 H16 N8192 showed pass 1 bound by its per-tile
    // chain, and with a single raw-dO stage the refill TMA is issued mid-tile
    // (after the transposer + dP gemm released it) and must land ~half a
    // tile later: one L2 TMA latency, exposed on every tile.  The second raw
    // stage takes the 16 KB of the DEAD smem_kt member (pass 1 never reads
    // the Kt twin since V10 removed the in-pass dQ gemm; pass 2 loads its own).
    // ## FP11-M17 GT: the dOt twin (TMA-fed) runs at FA3FP11_M17_DOTST stages (its
    // own pipeline; the raw dO stays at Stages_dO); the Qt twin keeps Stages.
    static constexpr int kStages_dOt = (FA3FP11_M17_GT && !DeltaPass0_) ? (get<2>(TileShape_MNK_{}) >= 256 ? FA3FP11_M17_DOTST : FA3FP11_HD128_DOTST)
        : ((FA3FP11_M16_DO2 && !DeltaPass0_) ? 1 : Stages_dO);
    static constexpr int kStages_qt = (FA3FP11_M16_Q3 && !DeltaPass0_) ? 1 : Stages;
    using SmemLayoutQtRawAbl = decltype(tile_to_shape(
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element_, Int<get<0>(TileShape_MNK_{})>, Int<get<2>(TileShape_MNK_{}) / (NumMmaWarpGroups / AtomLayoutNdKV)>>()){},
        make_shape(Int<get<0>(TileShape_MNK_{})>{}, Int<get<2>(TileShape_MNK_{})>{}, Int<kStages_qt>{})));
    using SmemLayoutdOtRawAbl = decltype(tile_to_shape(
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element_, Int<get<0>(TileShape_MNK_{})>, Int<get<2>(TileShape_MNK_{}) / (NumMmaWarpGroups / AtomLayoutNdKV)>>()){},
        make_shape(Int<get<0>(TileShape_MNK_{})>{}, Int<get<2>(TileShape_MNK_{})>{}, Int<kStages_dOt>{})));
    static_assert(kStages >= kStages_dO);
    static_assert(Stages_dS == 1 || Stages_dS == kStages);
    static_assert(!Mma_dP_is_RS || SdP_swapAB_);  // If Mma_dP_is_RS, we need SdP_SwapAB
    using ClusterShape = ClusterShape_;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool DeltaPass0 = DeltaPass0_;
    static constexpr int kAbl = FA3FP11_ABL;   // M16 timing ablations (0 in every shipped build)

    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;

    static constexpr bool Q_dO_same_stages = kStages == kStages_dO;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    // ## Delta-Matching-128: the [128, 64] block-scale contract at hd128 ([64, 32] at d256).  Q/dO scales per
    // kQuantRowsQ rows (kQShift = log2(kQuantRowsQ / kBlockM) m-blocks per scale), K/V per kQuantKeys keys.
    static constexpr int kQuantKeys = kHeadDim >= 256 ? 32 : 64;
    static constexpr int kQuantRowsQ = kHeadDim >= 256 ? 64 : 128;
    static_assert(kQuantRowsQ % kBlockM == 0 && (kQuantRowsQ / kBlockM == 1 || kQuantRowsQ / kBlockM == 2),
                  "delta_matching: one or two m-blocks per Q scale block");
    static constexpr int kQShift = kQuantRowsQ / kBlockM == 2 ? 1 : 0;
    // The dV smem overlay covers the CTA's kBlockN x kHeadDim fp32 tile (= kBlockM x kHeadDim at d256).
    static constexpr int kFq8DvOverlayFloats = kBlockN * kHeadDim;
    // ## Delta-Matching-128 lever P1-C' (M19): kBlockM = the PIPELINE stage (Q / dO / Qt / dOt / LSE rows per TMA
    // transaction), kBlockMh = the per-tile arithmetic (S / dP fragments, psi, dS' codes and their [64, kBlockN] store tile,
    // the exchange, the mask, the dK fold): at hd128 with FA3FP11_HD128_BM128 the stage is 128 rows processed as kHalves = 2
    // sequential 64-row halves with the P1-A arithmetic verbatim (pass 2 and the psi field are untouched); everywhere else
    // kBlockMh == kBlockM and kHalves == 1 (the certified path, textually unchanged: the two-half step is a separate lambda).
    static constexpr int kBlockMh = kBlockM > 64 ? 64 : kBlockM;
    static constexpr int kHalves = kBlockM / kBlockMh;
    static_assert(kHalves == 1 || (kHalves == 2 && kHeadDim < 256), "delta_matching: two-half stages are the hd128 P1-C' lever only");
    using TileShape_MhNK = Shape<Int<kBlockMh>, Int<kBlockN>, Int<kHeadDim>>;

    using SeqlenInfo_t = flash::SeqlenInfoQK<Varlen, kBlockM>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local>;

    static_assert(ArchTag::kMinComputeCapability >= 90);
    static_assert(get<0>(ClusterShape{}) == 1 && get<2>(ClusterShape{}) == 1);

    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp * 2;

    static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0);
    static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0);
    static_assert(NumMmaWarpGroups % AtomLayoutMdQ == 0);
    static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 && AtomLayoutNdKV == NumMmaWarpGroups && SdP_swapAB && !dKV_swapAB;
    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == NumMmaWarpGroups && AtomLayoutMdQ == NumMmaWarpGroups && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS

    static constexpr GMMA::Major PdS_Major = GMMA::Major::K;
    // static constexpr GMMA::Major PdS_Major = GMMA::Major::MN;
    static constexpr GMMA::Major PdSt_Major = PdS_Major == GMMA::Major::K ? GMMA::Major::MN : GMMA::Major::K;

    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockMh>, Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN>, Int<kBlockMh / AtomLayoutMSdP>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<NumMmaWarpGroups / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    // ## FQ8: the RS dP variant is dead (Mma_dP_is_RS requires SdP_swapAB, which
    // the fp8 plan forbids) and its selector would instantiate an MN-major fp8
    // atom (a static_assert): alias it away.
    using TiledMmadPRS = TiledMmaSdP;

    using TileShapeAtomdKV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<NumMmaWarpGroups / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    // ## FQ8: fp8 wgmma is K-major-only.  A = dO^T / Q^T gmem twins
    // (m-contiguous), B = P-hat^T / dS^T in-smem transposed tiles (m-contiguous):
    // both Major::K on real buffers (bf16 used MN-major stride-views here, which
    // e4m3 wgmma forbids).  NB std::conditional_t instantiates BOTH branches, so
    // the dead RS selector (MN-major) cannot even appear as a decltype.
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutdKV{}));

    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim>, Int<kBlockM / AtomLayoutMdQ>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<NumMmaWarpGroups / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
    >;
    // ## FQ8: A = K^T gmem twin (n-contiguous), B = dS natural exchange
    // (n-contiguous): both Major::K (same conditional_t note as dKV).
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::K>(),
        AtomLayoutdQ{}));

    // We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
    // Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
    // Since this is GMMA::Major::K, the M dimension (kBlockM) doesn't matter for the layout, only the K dimension
    // changes the layout.
    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                       Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>>()); // for dKV_Mma
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutdO =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages_dO>{})));
    // ## M16 ABL 64 (timing-only): raw-layout views over the twin buffers (declared after kStages_qt / kStages_dOt below)

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutMdQ)>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomPdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<PdS_Major, Element,
                                       Int<kBlockMh / AtomLayoutMSdP>,
                                       Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>>());
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockMh>{}, Int<kBlockN>{}, Int<kStages_dS>{}),
        std::conditional_t<PdS_Major == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    // Need stride to be multiple of 32, otherwise we get error (misaligned address) when doing TMA if e.g. kBlockM=80
    // We set stride to be multiple of 64 so that if ShuffleLSE, even if threads read from sLSE but out of bounds,
    // it's still a valid smem address.
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<cute::round_up(kBlockM, 64)>>>;
    using SmemLayoutLSEMma = std::conditional_t<
        SdP_swapAB,
        cute::Layout<cute::Shape<Int<kBlockN>, Int<kBlockM>, Int<kStages>>, cute::Stride<_0, _1, Int<cute::round_up(kBlockM, 64)>>>,
        cute::Layout<cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>, cute::Stride<_1, _0, Int<cute::round_up(kBlockM, 64)>>>
    >;
    // ## P1-C': the (kBlockMh, kBlockN) MMA view of one half of the LSE / dPsum stage (row offset kBlockMh * h applied to the pointer).
    using SmemLayoutLSEMmaH = cute::Layout<cute::Shape<Int<kBlockMh>, Int<kBlockN>, Int<kStages>>, cute::Stride<_1, _0, Int<cute::round_up(kBlockM, 64)>>>;

    // Note this is the transpose in terms of the view, not in terms of memory.
    // ## FQ8: REAL K-major buffers replace the bf16 stride-composition views (fp8
    // wgmma cannot read MN-major).  Qt/dOt/Kt are TMA'd gmem twins [d, seq]; PdSt
    // holds the in-smem transposed P-hat / dS tiles.
    using SmemLayoutAtomQdOt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                        Int<kHeadDim>, Int<kBlockM>>());
    using SmemLayoutQt =
        decltype(tile_to_shape(SmemLayoutAtomQdOt{},
                 make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_qt>{})));
    using SmemLayoutdOt =
        decltype(tile_to_shape(SmemLayoutAtomQdOt{},
                 make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_dOt>{})));
    using SmemLayoutAtomKt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                      Int<kHeadDim>, Int<kBlockN>>());
    using SmemLayoutKt = decltype(tile_to_shape(SmemLayoutAtomKt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));
    using SmemLayoutAtomPdSt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                        Int<kBlockN>, Int<kBlockMh>>());
    using SmemLayoutPdSt = decltype(tile_to_shape(
        SmemLayoutAtomPdSt{},
        make_shape(Int<kBlockN>{}, Int<kBlockMh>{}, Int<kStages_dS>{})));

    // ## FQ8-V9: in-smem Q/dO transpose (the fwd Transpose_V pattern with
    // X = kHeadDim = 256, Y = kBlockM = 64; the multiple-64 branch).  Source:
    // the raw K-major (kBlockM, kHeadDim) staging buffer REINTERPRETED as
    // logical (kHeadDim, kBlockM) MN-major (same bytes); dest: the existing
    // wgmma-ready transposed twins.  LDSM.T pairs + byte-perm unzip + STSM.
    using SmemLayoutAtomQTrans = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::MN, Element,
        Int<kHeadDim>, Int<kBlockM>>());
    using SmemLayoutQTrans = decltype(tile_to_shape(
        SmemLayoutAtomQTrans{},
        make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages>{}),
        cute::Step<_2, _1, _3>{}));
    using SmemLayoutdOTrans = decltype(tile_to_shape(
        SmemLayoutAtomQTrans{},
        make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_dO>{}),
        cute::Step<_2, _1, _3>{}));
    static_assert(cute::cosize_v<SmemLayoutQTrans> == cute::cosize_v<SmemLayoutQ>);
    static_assert(kHeadDim % 64 == 0 && kBlockM % 32 == 0);
    using LDSM_thread_shape  = Shape<_32, _4, _1, _1>;
    using LDSM_thread_stride = Stride<_4, _1, _0, _0>;
    using LDSM_value_shape = Shape<_2, _2, _1, _4>;
    using LDSM_value_stride = Stride<_1, _2, _16, _4>;
    using LDSM_divide_shape = Shape<_64, _8>;
    using S2RTiledCopyQt = decltype(make_tiled_copy(
        Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>{}, Layout<LDSM_thread_shape, LDSM_thread_stride>{},
        Layout<LDSM_value_shape, LDSM_value_stride>{}));
    using STSM_thread_shape  = Shape<_8, _4, _4, _1>;
    using STSM_thread_stride = Stride<_4, _1, _32, _0>;
    using STSM_value_shape = Shape<_1, _4, _2, _2>;
    using STSM_value_stride = Stride<_0, _1, _4, _8>;
    using STSM_divide_shape = Shape<_8, _16>;
    using R2STiledCopyQt = decltype(make_tiled_copy(
        Copy_Atom<cute::SM90_U32x4_STSM_N, Element>{}, Layout<STSM_thread_shape, STSM_thread_stride>{},
        Layout<STSM_value_shape, STSM_value_stride>{}));

    // Thread layout, 256 or 384 threads per row
    // We split into NumMmaWarpGroups so that we can do Bulk reduce add for each WG separately.
    using R2SLayoutAtomdQaccum = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumMmaWarpGroups>>>;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
    using SmemLayoutdQaccum = Layout<Shape<Int<kFq8DvOverlayFloats / NumMmaWarpGroups>, Int<NumMmaWarpGroups>>>;

    static constexpr int kNumPdSStore = kBlockMh * kBlockN / NumMmaThreads;
    // If !SdP_swapAB, the accum registers hold P / dS, otherwise they hold Pt / dSt.
    // If PdS_major is MN, then we need to "transpose" the write.
    using SmemCopyAtomPdS = Copy_Atom<
        std::conditional_t<(!SdP_swapAB) ^ (PdS_Major == GMMA::Major::MN),
            std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U32x4_STSM_N, cute::SM90_U32x2_STSM_N>,
            std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>
        >,
        Element
    >;

    using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;  // (seqlen, head, batch)
    using ShapedQaccum = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_q * d, head, batch)
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    using TMA_QdO = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}),
        TileShape_MNK{},
        ClusterShape{})); // mcast along N mode for this M load, if any

    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK{},
        TileShape_MNK{},
        ClusterShape{})); // no mcast for KV

    using TMA_V = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutV{},
        TileShape_MNK{},
        ClusterShape{})); // no mcast for KV

    // ## FQ8: gmem-twin TMA types.  Twins are [d, seq] per (head, batch): the
    // 4-int ShapeQKV tuple is reused with (d, seq, head, batch) semantics; the _1
    // stride sits at mode 1 in both orderings so StrideQKV is reused too.
    // ## FQ8: built via make_tma_copy_A_sm90 (the proven multi-mode bookkeeping;
    // a generic rank-2-tiler make_tma_copy mis-programs the descriptor once the
    // head mode has extent > 1 -- measured as head-triggered corruption).
    using TileShape_TwinN = Shape<Int<kHeadDim>, _1, Int<kBlockN>>;
    using TMA_Kt = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutKt{},
        TileShape_TwinN{},
        ClusterShape{}));

    // ## FP11-M17 GT: gmem-twin TMA types for Q^T / dO^T ([d, seq] per (head,
    // batch), seq positions pi-ordered by the producer; TileShape_TwinM =
    // (kHeadDim, kBlockM) boxes into the existing SmemLayoutQt / SmemLayoutdOt
    // stages, built like TMA_Kt).
    using TileShape_TwinM = Shape<Int<kHeadDim>, _1, Int<kBlockM>>;
    using TMA_Qt = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQt{}),
        TileShape_TwinM{},
        ClusterShape{}));
    using TMA_dOt = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutdOt{}),
        TileShape_TwinM{},
        ClusterShape{}));

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
    using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
    // ## FP11-M17 GT: the twin pipelines are TMA pipelines fed by the producer
    // warp (one full barrier with a transaction count per stage); consumers =
    // the MMA WGs, released after the dV / dK gemms retire exactly as before.
    using MainloopPipelineQt = typename cutlass::PipelineTmaAsync<kStages_qt>;
    using PipelineStateQt = typename MainloopPipelineQt::PipelineState;
    using MainloopPipelineDOt = typename cutlass::PipelineTmaAsync<kStages_dOt>;
    using PipelineStateDOt = typename MainloopPipelineDOt::PipelineState;
    static constexpr int NumTransposeThreads = 2 * cutlass::NumThreadsPerWarp;
    // ## FP11-M17 GT: per-stage twin transaction bytes (16 KB each at d256, kBlockM 64).
    static constexpr uint32_t TmaTransactionBytesQt = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQt{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesdOt = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutdOt{})) * cutlass::sizeof_bits_v<Element> / 8);

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    // ## FQ8-V9: twins are produced IN-SMEM by the transpose warps; TMA carries
    // only the natural tiles (the dO pipeline byte count mirrors Q's: symmetry
    // Q+LSE == dO+dPsum).  barrier_KV still covers the K twin (one-shot).
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8)
        * ((FA3FP11_ABL & 64) ? 2u : 1u);   // ## M16 ABL 64: + the twin's bytes on the same barrier
    // ## FP11-M5: pass 0 never consumes the Kt twin, so it skips the one-shot
    // Kt load (the copy is constexpr-guarded in load(); the barrier byte
    // count drops in lockstep).  Pass 1's value is unchanged.
    // ## FP11-M16 DO2: pass 1 no longer loads the dead Kt twin either.
    // ## FP11-M17 GT: the dead Kt load is dropped too (its 16 KB funds the dOt stage).
    static constexpr bool kLoadKt = !DeltaPass0 && !FA3FP11_M16_DO2 && !FA3FP11_M17_GT;
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8)
        + (!kLoadKt ? 0u
           : static_cast<uint32_t>(size(SmemLayoutKt{}) * cutlass::sizeof_bits_v<Element> / 8));
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(SmemLayoutV{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> / 8);

    // These are tuned for speed. They don't affect correctness.
    // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
    // this helps quite a bit to not have to do causal masking for most of the iterations.
    // For hdim 192, separating masking iterations results in register spills.
    static constexpr bool SeparateMaskingIterations = kHeadDim <= 64;
    // Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share them and then
    // shuffle to get the value whenever we need? This can reduce register pressure when SdP_swapAB, where each
    // thread needs to keep statistics for (kBlockM / 4) rows. If !SdP_swapAB, each thread only needs to keep
    // statistic for 2 rows.
    static constexpr bool ShuffleLSE = SdP_swapAB && kHeadDim <= 64;
    static constexpr bool ShuffledPsum = SdP_swapAB && kHeadDim <= 64;
    // ## V10 (split bwd): TMA tile-store of the natural dS' smem tile into
    // [BH, nbm, nbn, 64, 64] gmem (tile-contiguous for the pass-2 GEMM).
    using ShapeDS8 = cute::Shape<Int<kBlockMh>, Int<kBlockN>, int32_t, int32_t, int32_t>;
    using StrideDS8 = cute::Stride<Int<kBlockN>, _1, int64_t, int64_t, int64_t>;
    using TMA_dS8 = decltype(make_tma_copy(
        cute::SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeDS8{}, StrideDS8{}),
        SmemLayoutPdS{}(_, _, _0{})));

    // ## FQ8-V9: TMA bulk-reduce for dQaccum.  MEASURED both ways in v9:
    // atomics + dO-2-stage = 3.67ms vs TMA-reduce + dO-1-stage = 3.13ms; the
    // atomic instruction/L1 pressure costs ~0.5ms here, far more than the dO
    // double-buffering returns.  (L2 red sectors are identical either way.)
    static constexpr bool dQacc_use_TMA = true;
    // For hdim256, we want to slice the dQ MMA (64 x 256 on 2 WGs) into two (64 x 128 on 2 WGs) so that we can
    // do atomic add on one half before doing the other half of the MMA, to reduce register pressure.
    // ## FQ8: the Slice interleave is disabled in v1 (the split-K dQ with its
    // basis rescale replaces it; revisit for perf once numerics are certified).
    static constexpr bool Slice_dQKV_Mma = false;
    static_assert(!(Deterministic && Slice_dQKV_Mma), "Deterministic mode not supported with Slice_dQKV_Mma");

    static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    static constexpr size_t SmemAlignmentdS = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    // Without this SmemAlignment, with hdim 256 we get "misaligned address" error in TMA
    static constexpr size_t SmemAlignmentQKVdO = kHeadDim % 256 == 0 ? 256 : 128;
    static constexpr size_t SmemAlignmentV = !Mma_dP_is_RS ? SmemAlignmentQKVdO : cutlass::detail::alignment_for_swizzle(SmemLayoutV{});
    static_assert(SmemAlignmentP >= 128 && SmemAlignmentdS >= 128, "Require at least 128B alignment");

    // TODO: do we have to worry that smem_dk and smem_dv in the epilogue don't line up w smem_k and smem_v due to alignment?
    using SmemdQacc_t = std::conditional_t<!dQacc_use_TMA, cute::array<ElementAccum, 0>, cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccum>>>;
    // ## FP11-M5: the pass-1 storage keeps its M4 name-and-layout verbatim
    // (renamed TensorStorageP1, re-exposed through the TensorStorage alias
    // below: byte-identical struct).
    struct TensorStorageP1 : cute::aligned_struct<cute::max(SmemAlignmentP, SmemAlignmentdS, SmemAlignmentQKVdO)> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        // ## FQ8-V8: sdQ must sit BEYOND the epilogue's 64KB dK/dV staging window
        // (the kernel unions epilogue storage at offset 0).  fp8 K/V are half of
        // bf16's, so FA3's original slot lands sdQ at [32,96)KB, inside the
        // window: the epilogue's r2s stores race the LAST bulk-reduce's async
        // read of sdQ (observed: final m_block dq corruption).  After
        // k16+v16+q32+do16 = 80KB, sdQ is clear by a 16KB margin.
        SmemdQacc_t smem_dqacc;
        // ## FQ8: gmem twins (K-major operands for dKV/dQ) + transposed exchanges.
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQt>, SmemAlignmentQKVdO> smem_qt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdOt>, SmemAlignmentQKVdO> smem_dot;
        // ## FP11-M16 DO2: the Kt twin buffer is DEAD in pass 1 (V10 removed the
        // in-pass dQ gemm; only descriptor iterators were built on it) and its
        // 16 KB funds the second raw-dO stage.  A zero-length member keeps the
        // sKt tensor construction compiling (never dereferenced).
        std::conditional_t<(FA3FP11_M16_DO2 != 0) || (FA3FP11_M17_GT != 0),
                           cute::array<Element, 0>,
                           cute::array_aligned<Element, cute::cosize_v<SmemLayoutKt>, SmemAlignmentQKVdO>> smem_kt;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_dpsum;
        // ## FQ8: smem_pt = P-hat^T (dV's A source), smem_dst = dS^T (dK's A
        // source), smem_ds = dS natural (dQ's B source); psi cross-WG scratch.
        // 1024-B alignment is LOAD-BEARING: the GMMA descriptor's swizzle is
        // anchored to the physical address, coordinate stores to the tensor base;
        // a base not aligned to the full swizzle period shows up as a uniform
        // XOR-phase between store and read (measured: XOR 0x20 on the m axis).
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdSt>, 1024> smem_pt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdSt>, 1024> smem_dst;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, 1024> smem_ds;
        // ## FP11-F3: 8 slots, one per MMA warp (fq8_psired[thread_idx/32] with
        // NumMmaThreads = 256 -> warp id 0..7).  v10 declared 4: warps 4..7
        // wrote 16 bytes past the array's end into this struct's alignment tail
        // padding (~1 KB of it today, so it never fired, but it was UB and one
        // member-reorder away from corrupting the pipeline barriers).  sizeof
        // (TensorStorage) is 1024-aligned with ~1008 B of tail padding, so the
        // extra 16 B does not change the smem footprint.
        cute::array_aligned<ElementAccum, 8, 16> smem_psired;
    };
    // ## FP11-M5 (pass-0 storage): only what the S/dP recompute consumes.
    // smem_k/v/q/do/kt/lse/dpsum keep the EXACT pass-1 member declarations
    // (same SmemLayout cosizes, same alignments) so load() runs UNCHANGED in
    // both modes (pass 0 loads the Kt twin and dPsum exactly as pass 1 does;
    // both are dead weight here, and keeping them keeps every TMA
    // transaction-byte constant and the producer instruction stream
    // identical: D3's "same pipeline", modulo the flagged Stages_dO=2).
    // Dropped: dqacc/qt/dot/pt/dst/ds/psired (the dV/dK/dS'-store/psi
    // machinery pass 0 compiles out); the delta partials go straight to
    // gmem (no exchange buffer, see the mma_pass0 loop).  ~115 KB total
    // (dO double-buffered); occupancy is register-bound at 1 CTA/SM either
    // way (the 40/232 setmaxnreg split uses 64,512 of the 65,536-reg file).
    struct TensorStorageDelta : cute::aligned_struct<SmemAlignmentQKVdO> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutKt>, SmemAlignmentQKVdO> smem_kt;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128> smem_dpsum;
    };
    using TensorStorage = std::conditional_t<DeltaPass0, TensorStorageDelta, TensorStorageP1>;

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* const ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* const ptr_V;
        ShapeQKV const shape_V;
        StrideQKV const stride_V;
        Element const* const ptr_dO;
        ShapeQKV const shape_dO;
        StrideQKV const stride_dO;
        // ## FQ8: gmem twins [d, seq, head, batch] + per-block fp32 scales
        // (q8s/do8s: one per 64-row m-block; k8s/v8s: one per 32-row n-block,
        // laid out [batch][head][block] contiguous).
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
        // ## V10 (split bwd): pass-1 outputs.
        Element* const ptr_dS8;
        float* const ptr_psi;
        int const nbm;
        int const nbn;
        // ## FP11-M5: pass-0 output (the per-CTA per-row delta partials,
        // [BH, nbn, 2, N] fp32, per warpgroup column half; nullptr in
        // pass-1 mode) and the DBG-build-only
        // instrumentation workspaces (always-present fields, nullptr unless
        // a *_dbg op sets them; no shipped-build code reads them).
        float* const ptr_dltpart;
        float* const ptr_dbg_rows;      // [BH, nbn, 2, N] fp32 (pass-1 dS row-sum partials)
        float* const ptr_dbg_rowsabs;   // [BH, nbn, 2, N] fp32 (sum |dS|)
        unsigned int* const ptr_dbg_cks;  // [2, BH, nbm, nbn] u32 (XOR of P~ / dP~ fragment bits)
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ShapeQKV const shape_V;
        ShapeQKV const shape_dO;
        ElementAccum* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum stride_dQaccum;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_QdO tma_load_Q, tma_load_dO;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        // ## FQ8 (V9: Q/dO twins made in-smem; only the K twin loads from gmem)
        TMA_Kt tma_load_Kt;
        // ## FP11-M17 GT: the Q^T / dO^T gmem-twin descriptors (per m-block TMA).
        TMA_Qt tma_load_Qt;
        TMA_dOt tma_load_dOt;
        ShapeQKV const shape_Qt;
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
        float const softmax_scale, softmax_scale_log2;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        float const softcap_val;
        int const num_batch;
        int* const dq_semaphore;
        int const* const cu_seqlens_q;
        int const* const cu_seqlens_k;
        int const* const seqused_q;
        int const* const seqused_k;
        // ## V10 (split bwd): pass-1 outputs.
        TMA_dS8 tma_store_dS8;
        float* const ptr_psi;
        int const nbm;
        int const nbn;
        // ## FP11-M5: pass-0 output + DBG-only workspaces (see Arguments).
        float* const ptr_dltpart;
        float* const ptr_dbg_rows;
        float* const ptr_dbg_rowsabs;
        unsigned int* const ptr_dbg_cks;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_QdO tma_load_Q = make_tma_copy_A_sm90(
            GmemTiledCopyQdO{},
            mQ,
            SmemLayoutQ{}(_, _, _0{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along N mode for this M load, if any
        Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_dO, args.stride_dO);
        TMA_QdO tma_load_dO = make_tma_copy_A_sm90(
            GmemTiledCopyQdO{},
            mdO,
            SmemLayoutdO{}(_, _, _0{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along N mode for this M load, if any
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mK,
            SmemLayoutK{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for KV
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_V, args.stride_V);
        TMA_V tma_load_V = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mV,
            SmemLayoutV{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for KV
        // ## FQ8-V9: only the K twin still loads from gmem (one-shot per CTA).
        Tensor mKt = make_tensor(make_gmem_ptr(args.ptr_Kt), args.shape_Kt, args.stride_Kt);
        TMA_Kt tma_load_Kt = make_tma_copy_A_sm90(
            cute::SM90_TMA_LOAD{}, mKt, SmemLayoutKt{}, TileShape_TwinN{}, ClusterShape{});
        // ## FP11-M17 GT: Q^T / dO^T gmem twins [d, seq, head, batch] (seq contiguous).
        Tensor mQt = make_tensor(make_gmem_ptr(args.ptr_Qt), args.shape_Qt, args.stride_Qt);
        TMA_Qt tma_load_Qt = make_tma_copy_A_sm90(
            cute::SM90_TMA_LOAD{}, mQt, take<0, 2>(SmemLayoutQt{}), TileShape_TwinM{}, ClusterShape{});
        Tensor mdOt = make_tensor(make_gmem_ptr(args.ptr_dOt), args.shape_dOt, args.stride_dOt);
        TMA_dOt tma_load_dOt = make_tma_copy_A_sm90(
            cute::SM90_TMA_LOAD{}, mdOt, take<0, 2>(SmemLayoutdOt{}), TileShape_TwinM{}, ClusterShape{});
        // ## V10 (split bwd): the dS8 tile-store TMA (de-swizzles SmemLayoutPdS).
        Tensor mDS8 = make_tensor(make_gmem_ptr(args.ptr_dS8),
            make_shape(Int<kBlockMh>{}, Int<kBlockN>{}, args.nbn, args.nbm,
                       args.num_batch * get<2>(args.shape_Q)),
            make_stride(Int<kBlockN>{}, _1{}, int64_t(kBlockMh) * kBlockN,
                        int64_t(kBlockMh) * kBlockN * args.nbn,
                        int64_t(kBlockMh) * kBlockN * args.nbn * args.nbm));
        TMA_dS8 tma_store_dS8 = make_tma_copy(
            cute::SM90_TMA_STORE{}, mDS8, SmemLayoutPdS{}(_, _, _0{}));
        if constexpr (Deterministic) { assert(args.dq_semaphore != nullptr); }
        // Avoid dividing by zero
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        // In the backward, we need to multiply by
        // (1 - tanh^2) * softmax_scale / softcap_val * softcap_val = (1 - tanh^2) * softmax_scale.
        // Instead we multiply by (1 - tanh^2) and multiply dK and dV by params.softmax_scale
        // (the original softmax_scale) at the end.
        return {args.shape_Q, args.shape_K, 
                args.shape_V, args.shape_dO,
                args.ptr_dQaccum, args.shape_dQaccum, args.stride_dQaccum,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_dO, tma_load_K, tma_load_V,
                tma_load_Kt,
                tma_load_Qt, tma_load_dOt,
                args.shape_Qt,
                args.ptr_q8s, args.ptr_k8s, args.ptr_v8s, args.ptr_do8s,
                args.nblk_q, args.nblk_k,
                args.ptr_LSE_log2, args.shape_LSE, args.stride_LSE_log2, args.ptr_dPsum, args.stride_dPsum,
                args.softmax_scale,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.window_size_left, args.window_size_right, attention_chunk_divmod,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.num_batch, args.dq_semaphore,
                args.cu_seqlens_q, args.cu_seqlens_k, args.seqused_q, args.seqused_k,
                tma_store_dS8, args.ptr_psi, args.nbm, args.nbn,
                args.ptr_dltpart, args.ptr_dbg_rows, args.ptr_dbg_rowsabs,
                args.ptr_dbg_cks};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_Kt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_Qt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dOt.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_store_dS8.get_tma_descriptor());
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipeline pipeline_q,
         MainloopPipeline_dO pipeline_do,
         PipelineState& smem_pipe_write,
         PipelineState_dO& smem_pipe_write_do,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t> block_coord
         ) {

        auto [n_block, bidh, bidb] = block_coord;
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb,
            params.window_size_left, params.window_size_right, 0 /*sink_token_length*/);
        // It's possible to have m_block_max <= m_block_min. Loading Q, K can cause illegal memory access.
        if constexpr (Is_causal || Is_local || Varlen) {
            if (m_block_max <= m_block_min) {
                scheduler_prefetch();
                return;
            }
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
        Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});
        // ## FQ8-V9: only the K twin is TMA-loaded here.
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_kt.data()), SmemLayoutKt{});

        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_dO)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_V)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        // ## FQ8-V9: only the K twin remains a gmem tensor.
        auto shape_Kt = make_shape(get<1>(params.shape_K), get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K));
        Tensor mKt = params.tma_load_Kt.get_tma_tensor(shape_Kt)(_, _, bidh_kv, bidb);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, bidh, !is_varlen_q ? bidb : 0);

        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gdO = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gKt = local_tile(mKt, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_0{}, n_block));  // (D, N)
        Tensor gLSE = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mLSE), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)

        Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
        Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
        Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
        Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
        // auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, block_rank_in_cluster, Layout<ClusterShape>{},
        //                                   group_modes<0, 2>(sQ), group_modes<0, 2>(gQ));  // (TMA, k), (TMA, PIPE)
        // auto [tdOgdO, tdOsdO] = tma_partition(params.tma_load_dO, block_rank_in_cluster, Layout<ClusterShape>{},
        //                                   group_modes<0, 2>(sdO), group_modes<0, 2>(gdO));  // (TMA, k), (TMA, PIPE)
        auto block_tma_Q = params.tma_load_Q.get_slice(cluster_local_block_id.y);
        auto block_tma_dO = params.tma_load_dO.get_slice(cluster_local_block_id.y);
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
        Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
        Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
        // ## M16 ABL 64 (timing-only): the twin buffers viewed with the raw layouts as second TMA
        // destinations (models the L2 stream of gmem twins; the data is meaningless).
        // (Only instantiated in the pass-1 mainloop: the DeltaPass0 storage has no twins.)
        auto abl_twins = [&]() {
            if constexpr ((kAbl & 64) && !DeltaPass0) {
                Tensor sQtRaw = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qt.data()), SmemLayoutQtRawAbl{});
                Tensor sdOtRaw = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dot.data()), SmemLayoutdOtRawAbl{});
                return cute::make_tuple(group_modes<0, 3>(block_tma_Q.partition_D(sQtRaw)), group_modes<0, 3>(block_tma_dO.partition_D(sdOtRaw)));
            } else {
                return cute::make_tuple(0, 0);
            }
        }();
        auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));  // (TMA), (TMA)
        auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));  // (TMA), (TMA)
        // ## FQ8-V9: only the one-shot Kt twin partition remains.
        Tensor sKt_x = make_tensor(sKt.data(), make_layout(sKt.layout(), Layout<_1>{}));
        Tensor gKt_x = make_tensor(gKt.data(), make_layout(gKt.layout(), Layout<_1>{}));
        auto [tKtgKt, tKtsKt] = tma_partition(params.tma_load_Kt, _0{}, Layout<_1>{},
                                              group_modes<0, 2>(sKt_x), group_modes<0, 2>(gKt_x));
        auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};

        uint16_t mcast_mask_qdo = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int n = 0; n < size<1>(block_layout); ++n) {
                mcast_mask_qdo |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, _0{}));
            }
        }

        int m_block = m_block_min;

        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            pipeline_q.producer_acquire(smem_pipe_write);
            copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
            if constexpr ((kAbl & 64) && !DeltaPass0) { copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST), tQgQ(_, m_block), get<0>(abl_twins)(_, smem_pipe_write.index() % kStages_qt)); }
            copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
                 gLSE(_, m_block), sLSE(_, smem_pipe_write.index()));
        }

        // // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
        // cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);

        if (lane_predicate) {
            // Copy K tile and V tile from GMEM to SMEM.
            shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
            copy(params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tKgK, tKsK);
            copy(params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tVgV, tVsV);
            // ## FP11-M5: pass 0 skips the (unconsumed) Kt twin load; the
            // barrier byte count above dropped in lockstep.
            if constexpr (kLoadKt) {
                copy(params.tma_load_Kt.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tKtgKt, tKtsKt);
            }

            #pragma unroll (kHeadDim < 256 ? 2 : 1)
            for (; m_block < m_block_max - 1; ++m_block) {
                // If Q and dO have the same number of stages, we can use the same pipeline state variable
                // to reduce registers
                PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
                pipeline_do.producer_acquire(smem_pipe_write_do_cur);
                if constexpr (!(kAbl & 128)) {   // M16 ABL 128 (timing-only): no dO stream
                copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
                }
                if constexpr ((kAbl & 64) && !DeltaPass0) { copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST), tdOgdO(_, m_block), get<1>(abl_twins)(_, smem_pipe_write_do_cur.index() % kStages_dOt)); }
                copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
                     gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
                if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
                ++smem_pipe_write;
                pipeline_q.producer_acquire(smem_pipe_write);
                copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tQgQ(_, m_block + 1), tQsQ(_, smem_pipe_write.index()));
                if constexpr ((kAbl & 64) && !DeltaPass0) { copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST), tQgQ(_, m_block + 1), get<0>(abl_twins)(_, smem_pipe_write.index() % kStages_qt)); }
                copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
                     gLSE(_, m_block + 1), sLSE(_, smem_pipe_write.index()));
            }
        }
        scheduler_prefetch();
        if (lane_predicate) {
            PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
            pipeline_do.producer_acquire(smem_pipe_write_do_cur);
            if constexpr (!(kAbl & 128)) {   // M16 ABL 128 (timing-only): no dO stream
            copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
            }
            if constexpr ((kAbl & 64) && !DeltaPass0) { copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST), tdOgdO(_, m_block), get<1>(abl_twins)(_, smem_pipe_write_do_cur.index() % kStages_dOt)); }
            copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
                 gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
            ++smem_pipe_write;
        }
        if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
    }

    // ## FP11-M17 GT producer: load() plus the Q^T / dO^T gmem twins, TMA-loaded
    // per m-block into the existing smem twin stages on their own TMA pipelines
    // (pass 1 only; DeltaPass0 keeps load() verbatim).  Issue order per m-block
    // mirrors the consumers' release order: dO(m), dOt(m), Q(m+1), Qt(m+1).
    // The Kt one-shot load is gone (kLoadKt is false under GT).  The M16 ABL
    // timing-only bits are not supported on this path (ABL builds are GT=0).
    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load_gt(Params const& params,
            MainloopPipeline pipeline_q,
            MainloopPipeline_dO pipeline_do,
            MainloopPipelineQt pipeline_qt,
            MainloopPipelineDOt pipeline_dot,
            PipelineState& smem_pipe_write,
            PipelineState_dO& smem_pipe_write_do,
            PipelineStateQt& smem_pipe_write_qt,
            PipelineStateDOt& smem_pipe_write_dot,
            SharedStorage &shared_storage,
            SchedulerPrefetch const& scheduler_prefetch,
            cute::tuple<int32_t, int32_t, int32_t> block_coord
            ) {
        static_assert(kAbl == 0, "delta_matching: ABL timing-only bits are not supported with gmem twins");
        if constexpr (DeltaPass0) {
            load(params, pipeline_q, pipeline_do, smem_pipe_write, smem_pipe_write_do,
                 shared_storage, scheduler_prefetch, block_coord);
        } else {
        auto [n_block, bidh, bidb] = block_coord;
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb,
            params.window_size_left, params.window_size_right, 0 /*sink_token_length*/);
        if constexpr (Is_causal || Is_local || Varlen) {
            if (m_block_max <= m_block_min) {
                scheduler_prefetch();
                return;
            }
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

        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_dO)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_V)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        // The twins are [d, seq, head, batch] (seq contiguous), (d, seq) = shape_Q's modes swapped.
        auto shape_Qt = make_shape(get<1>(params.shape_Q), get<0>(params.shape_Q), get<2>(params.shape_Q), get<3>(params.shape_Q));
        auto shape_dOt = make_shape(get<1>(params.shape_dO), get<0>(params.shape_dO), get<2>(params.shape_dO), get<3>(params.shape_dO));
        Tensor mQt = params.tma_load_Qt.get_tma_tensor(shape_Qt)(_, _, bidh, bidb);
        Tensor mdOt = params.tma_load_dOt.get_tma_tensor(shape_dOt)(_, _, bidh, bidb);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, bidh, !is_varlen_q ? bidb : 0);

        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gdO = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gQt = local_tile(mQt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));    // (D, M, _)
        Tensor gdOt = local_tile(mdOt, make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_coord(_0{}, _));  // (D, M, _)
        Tensor gLSE = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mLSE), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)

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
        Tensor tQtgQt = group_modes<0, 3>(block_tma_Qt.partition_S(gQt));      // (TMA, _)
        Tensor tQtsQt = group_modes<0, 3>(block_tma_Qt.partition_D(sQt));      // (TMA, PIPE)
        Tensor tdOtgdOt = group_modes<0, 3>(block_tma_dOt.partition_S(gdOt));
        Tensor tdOtsdOt = group_modes<0, 3>(block_tma_dOt.partition_D(sdOt));
        auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));  // (TMA), (TMA)
        auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));  // (TMA), (TMA)
        auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
        uint16_t const mcast_mask_qdo = 0;

        int m_block = m_block_min;
        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            pipeline_q.producer_acquire(smem_pipe_write);
            copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
            copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
                 gLSE(_, m_block), sLSE(_, smem_pipe_write.index()));
            pipeline_qt.producer_acquire(smem_pipe_write_qt);
            copy(params.tma_load_Qt.with(*pipeline_qt.producer_get_barrier(smem_pipe_write_qt), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tQtgQt(_, m_block), tQtsQt(_, smem_pipe_write_qt.index()));
        }

        if (lane_predicate) {
            shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
            copy(params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tKgK, tKsK);
            copy(params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_KV), 0 /*mcast_mask*/), tVgV, tVsV);

            #pragma unroll 1
            for (; m_block < m_block_max - 1; ++m_block) {
                PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
                pipeline_do.producer_acquire(smem_pipe_write_do_cur);
                copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
                copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
                     gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
                if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
                pipeline_dot.producer_acquire(smem_pipe_write_dot);
                copy(params.tma_load_dOt.with(*pipeline_dot.producer_get_barrier(smem_pipe_write_dot), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_dot.index()));
                ++smem_pipe_write_dot;
                ++smem_pipe_write;
                pipeline_q.producer_acquire(smem_pipe_write);
                copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                     tQgQ(_, m_block + 1), tQsQ(_, smem_pipe_write.index()));
                copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
                     gLSE(_, m_block + 1), sLSE(_, smem_pipe_write.index()));
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
            copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
                 gdPsum(_, m_block), sdPsum(_, smem_pipe_write_do_cur.index()));
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
            pipeline_dot.producer_acquire(smem_pipe_write_dot);
            copy(params.tma_load_dOt.with(*pipeline_dot.producer_get_barrier(smem_pipe_write_dot), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
                 tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_dot.index()));
            ++smem_pipe_write_dot;
            ++smem_pipe_write;
            ++smem_pipe_write_qt;
        }
        if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
        }   // !DeltaPass0
    }

    CUTLASS_DEVICE void
    load_tail_gt(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
                 MainloopPipelineQt pipeline_qt, MainloopPipelineDOt pipeline_dot,
                 PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do,
                 PipelineStateQt& smem_pipe_write_qt, PipelineStateDOt& smem_pipe_write_dot) {
        if (cute::elect_one_sync()) {
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
            if constexpr (!DeltaPass0) {
                pipeline_qt.producer_tail(smem_pipe_write_qt);
                pipeline_dot.producer_tail(smem_pipe_write_dot);
            }
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_write) {
        static_assert(Q_dO_same_stages, "Q and dO must have the same number of stages");
        // Need to copy since pipeline_q.producer_tail(smem_pipe_write) will increment smem_pipe_write
        PipelineState smem_pipe_write_do = smem_pipe_write;
        // Issue the epilogue waits
        if (cute::elect_one_sync()) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do) {
        // Issue the epilogue waits
        if (cute::elect_one_sync()) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    store_dq(Params const& params,
             SharedStorage &shared_storage,
             cute::tuple<int32_t, int32_t, int32_t> block_coord
             ) {
        return;   // ## V10 (split bwd): no dQ flush; pass 2 owns dQ.

        auto [n_block, bidh, bidb] = block_coord;
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left,
            params.window_size_right, 0 /*sink_token_length*/);
        // It's possible to have m_block_max <= m_block_min. Exit early
        // Though if local and deterministic, still need to increment dq semaphore
        if constexpr ((Is_causal || Is_local || Varlen) && !(Is_local && Deterministic)) {
            if (m_block_max <= m_block_min) { return; }
        }

        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});
        static constexpr int dQ_TMA_num_bytes = CUTE_STATIC_V(size<0>(sdQ)) * sizeof(ElementAccum);

        bool const is_varlen = Varlen && params.cu_seqlens_q;
        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gdQaccum_ = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)
        Tensor gdQaccum = cute::flat_divide(gdQaccum_, Int<kBlockM * kHeadDim / NumMmaWarpGroups>{});  // (M * K / WG, WG, _)

        int const num_batch = params.num_batch;
        int const num_head = get<2>(params.shape_Q);
        int *lock_ptr = !Deterministic ? nullptr : params.dq_semaphore + bidb * num_head + bidh;
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
        bool const lane_predicate = cute::elect_one_sync();
        int m_block = m_block_min;
        constexpr int kBlockM = get<0>(TileShape_MNK{});
        constexpr int kBlockN = get<1>(TileShape_MNK{});
        int n_block_global_max = cute::ceil_div(seqlen_info.seqlen_k, kBlockN);
        #pragma unroll 2
        for (; m_block < m_block_max; ++m_block) {
            if constexpr (Deterministic) {
                if constexpr(Is_causal) {
                    int n_block_max_for_m_block = std::min(n_block_global_max, cute::ceil_div((m_block + 1) * kBlockM + seqlen_info.seqlen_k - seqlen_info.seqlen_q, kBlockN));
                    Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head, n_block_max_for_m_block - 1 - n_block);
                } else {
                    Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head, n_block);
                }
            }
            // ## FQ8-V8: WHOLE-TILE flush protocol.  With dQ_swapAB=true each WG's
            // fragments interleave across BOTH sdQ flat halves (a WG owns a
            // d-half, not an m-half), so FA3's per-WG column protocol races.
            // One 288-thread barrier pair (256 consumers + this flush warp)
            // covers the full tile; both columns flush together.
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) /*id*/);  // sdQ full (whole tile)
            if (lane_predicate) {
                #pragma unroll
                for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
                    SM90_BULK_REDUCE_ADD::copy(raw_pointer_cast(sdQ(_, warpgroup_idx).data()), raw_pointer_cast(gdQaccum(_, warpgroup_idx, m_block).data()), dQ_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
                    tma_store_arrive();
                }
                tma_store_wait<0>();
            }
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) /*id*/);  // sdQ empty (whole tile)
            if constexpr (Deterministic) {
                Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
            }
        }
        if constexpr (Is_local && Deterministic) {
            int const m_block_global_max = cute::ceil_div(seqlen_info.seqlen_q, kBlockM);
            #pragma unroll 2
            for (; m_block < m_block_global_max; ++m_block) {
                Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
            }
        }
    }

    // ## FQ8: flash::gemm with an explicit k-atom range [K0, K1): the dQ GEMM
    // splits its contraction at the 32-row K-scale boundary (one fp8 k-atom per
    // scale group) with an accumulator basis-rescale between the halves.
    template <bool zero_init, int wg_wait, bool SwapAB, int K0, int K1,
              typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
    CUTLASS_DEVICE static void fq8_gemm_krange(TiledMma& tiled_mma, Tensor0 const& tCrA, Tensor1 const& tCrB, Tensor2& tCrC) {
        constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
        if constexpr (Is_RS) {
            if constexpr (!SwapAB) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
            else { warpgroup_fence_operand(const_cast<Tensor1 &>(tCrB)); }
        }
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        if constexpr (zero_init) { tiled_mma.accumulate_ = GMMA::ScaleOut::Zero; }
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = K0; k_block < K1; ++k_block) {
            if constexpr (!SwapAB) { cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC); }
            else { cute::gemm(tiled_mma, tCrB(_,_,k_block), tCrA(_,_,k_block), tCrC); }
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
        warpgroup_fence_operand(tCrC);
        if constexpr (Is_RS) {
            if constexpr (!SwapAB) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
            else { warpgroup_fence_operand(const_cast<Tensor1 &>(tCrB)); }
        }
    }

    // ## FQ8-V9: run by warps 2-3 of the load warpgroup (64 threads at the
    // 24-reg producer budget: LDSM.T/STSM + byte_perm only).  Reads the raw
    // K-major Q/dO staging tiles through an MN-major view of the same bytes,
    // writes the wgmma-ready transposed twins.  The raw pipelines are only
    // OBSERVED (consumer_try_wait); the MMA WGs own the releases.
    template <typename SharedStorage>
    CUTLASS_DEVICE void
    transpose_loop(Params const& params,
                   MainloopPipeline pipeline_q, MainloopPipeline_dO pipeline_do,
                   MainloopPipelineQt pipeline_qt, MainloopPipelineDOt pipeline_dot,
                   PipelineState& smem_pipe_read_q, PipelineState_dO& smem_pipe_read_do,
                   PipelineStateQt& smem_pipe_write_qt, PipelineStateDOt& smem_pipe_write_dot,
                   int const thread_idx_t,
                   cute::tuple<int32_t, int32_t, int32_t> block_coord,
                   SharedStorage& shared_storage) {

        auto [n_block, bidh, bidb] = block_coord;
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left,
            params.window_size_right, 0 /*sink_token_length*/);
        if constexpr ((Is_causal || Is_local || Varlen) && !(Is_local && Deterministic)) {
            if (m_block_max <= m_block_min) { return; }
        }

        Tensor sQraw = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQTrans{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qt.data()), SmemLayoutQt{});
        Tensor sdOraw = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdOTrans{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dot.data()), SmemLayoutdOt{});

        S2RTiledCopyQt s2r_tiled_copy;
        R2STiledCopyQt r2s_tiled_copy;
        auto fq8_dcorr = make_layout(
            make_shape(make_shape(_8{}, _2{}, _16{}), Int<kBlockM>{}),
            make_stride(make_stride(_2{}, _1{}, _16{}), Int<kHeadDim>{}));

        // ## FQ8-V9: ALL partitions hoisted out of the m-loop (addresses depend
        // only on (stage, round)); the loop body is pure LDSM/perm/shfl/STSM.
        auto make_src_part = [&](auto const& sFull, int stage, int round) {
            Tensor sSrc = sFull(_, _, stage);
            auto thr = s2r_tiled_copy.get_thread_slice(thread_idx_t + round * NumTransposeThreads);
            Tensor t_ = thr.partition_S(flat_divide(sSrc, LDSM_divide_shape{}));
            // ## FQ8-V9: ILP=2 (32B fragments): measured 7% faster than ILP=1 at
            // the 40-reg budget (memory-level parallelism wins).
            return logical_divide(group_modes<1, rank(t_)>(t_), Shape<Underscore, Int<FA3FP11_M16_TILP>>{});
        };
        auto make_dst_part = [&](auto const& dFull, int stage, int round) {
            Tensor sDstS = dFull(_, _, stage);
            Tensor sDstFix = make_tensor(sDstS.data(), composition(sDstS.layout(), fq8_dcorr));
            auto thr = r2s_tiled_copy.get_thread_slice(thread_idx_t + round * NumTransposeThreads);
            Tensor t_ = thr.partition_D(flat_divide(sDstFix, STSM_divide_shape{}));
            return logical_divide(group_modes<1, rank(t_)>(t_), Shape<Underscore, Int<FA3FP11_M16_TILP>>{});
        };
        // ## FQ8-V9: hoist ONLY stage-0 partitions; stage 1 = stage 0 plus a
        // single pointer offset (the 16KB stage stride is a swizzle-atom
        // multiple, so the address pattern is position-independent per stage).
        auto q_src_r0 = make_src_part(sQraw, 0, 0); auto q_src_r1 = make_src_part(sQraw, 0, 1);
        auto q_dst_r0 = make_dst_part(sQt, 0, 0);   auto q_dst_r1 = make_dst_part(sQt, 0, 1);
        auto do_src_r0 = make_src_part(sdOraw, 0, 0); auto do_src_r1 = make_src_part(sdOraw, 0, 1);
        auto do_dst_r0 = make_dst_part(sdOt, 0, 0);   auto do_dst_r1 = make_dst_part(sdOt, 0, 1);
        static constexpr int kStageElems = kHeadDim * kBlockM;
        auto with_stage = [](auto const& t, int stage) {
            return make_tensor(t.data() + stage * kStageElems, t.layout());
        };

        auto run_round = [&](auto const& tSrc, auto const& tDst) {
            #pragma unroll
            for (int i = 0; i < size<1, 1>(tSrc); ++i) {
                Tensor tTransrSrc = make_fragment_like(tDst(_, make_coord(_, _0{})));
                static_assert(size<0>(tTransrSrc) == 16);
                Tensor tTransrSrc_64 = recast<uint2>(tTransrSrc);
                cute::copy(s2r_tiled_copy, tSrc(_, make_coord(_, i)), tTransrSrc);
                #pragma unroll
                for (int j = 0; j < size(tTransrSrc_64); ++j) {
                    uint32_t upper = tTransrSrc_64[j].x;
                    uint32_t lower = tTransrSrc_64[j].y;
                    tTransrSrc_64[j].x = __byte_perm(upper, lower, 0x6420);
                    tTransrSrc_64[j].y = __byte_perm(upper, lower, 0x7531);
                }
                cute::copy(r2s_tiled_copy, tTransrSrc, tDst(_, make_coord(_, i)));
            }
        };
        auto transpose_q = [&](int src_stage, int dst_stage) {
            run_round(with_stage(q_src_r0, src_stage), with_stage(q_dst_r0, dst_stage));
            run_round(with_stage(q_src_r1, src_stage), with_stage(q_dst_r1, dst_stage));
        };
        // ## FP11-M16 DO2: the raw source stage and the twin destination stage
        // are separate pipeline states (raw 2 stages, twin 1 stage).
        auto transpose_do = [&](int src_stage, int dst_stage) {
            run_round(with_stage(do_src_r0, src_stage), with_stage(do_dst_r0, dst_stage));
            run_round(with_stage(do_src_r1, src_stage), with_stage(do_dst_r1, dst_stage));
        };

        static_assert((kStages == 2 || kStages == 3) && (kStages_dO == 1 || kStages_dO == 2), "FQ8-V9 hoisted transpose assumes Stages in {2,3}, Stages_dO<=2 (stage = pointer offset)");
        auto observe_wait = [](auto& pipeline, auto& state) {
            auto barrier_token = pipeline.consumer_try_wait(state);
            pipeline.consumer_wait(state, barrier_token);
        };

        #pragma unroll 1
        for (int m_block = m_block_min; m_block < m_block_max; ++m_block) {
            // ## FQ8-V9: Q first (measured: dO-first regressed 1-2%).  ONE
            // async-proxy fence covers BOTH transposes (the fence drains the
            // warp's outstanding STS; two per iteration showed ~19K stall
            // samples in the SASS profile); both commits follow it.
            observe_wait(pipeline_q, smem_pipe_read_q);
            if constexpr (!(kAbl & 256)) { pipeline_qt.producer_acquire(smem_pipe_write_qt); }   // M16 ABL 256: no twin protocol
            if constexpr (!(kAbl & 16)) { transpose_q(smem_pipe_read_q.index(), smem_pipe_write_qt.index()); }
            observe_wait(pipeline_do, smem_pipe_read_do);
            if constexpr (!(kAbl & 256)) { pipeline_dot.producer_acquire(smem_pipe_write_dot); }
            if constexpr (!(kAbl & 16)) { transpose_do(smem_pipe_read_do.index(), smem_pipe_write_dot.index()); }
            cutlass::arch::fence_view_async_shared();
            if constexpr (!(kAbl & 256)) { pipeline_qt.producer_commit(smem_pipe_write_qt); pipeline_dot.producer_commit(smem_pipe_write_dot); }
            ++smem_pipe_read_q;
            ++smem_pipe_write_qt;
            ++smem_pipe_read_do;
            ++smem_pipe_write_dot;
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        // We're not currently using this bc we're not using persistent scheduler
        // // Tell producer (warp 0) that smem_k and smem_v are ready
        // cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // ## V10 (split bwd): no dQ barrier seed.
    }

    template <typename SharedStorage, typename FrgTensordKV>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        MainloopPipeline pipeline_q,
        MainloopPipeline_dO pipeline_do,
        MainloopPipelineQt pipeline_qt,
        MainloopPipelineDOt pipeline_dot,
        PipelineState& smem_pipe_read,
        PipelineState_dO& smem_pipe_read_do,
        PipelineStateQt& smem_pipe_read_qt,
        PipelineStateDOt& smem_pipe_read_dot,
        FrgTensordKV& tdKrdK,
        FrgTensordKV& tdVrdV,
        int thread_idx,
        int &work_idx,
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

        int n_block = get<0>(block_coord);
        int bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left,
            params.window_size_right, 0 /*sink_token_length*/);
        // It's possible to have m_block_max <= m_block_min. Exit early
        if constexpr (Is_causal || Is_local || Varlen) {
            if (m_block_max <= m_block_min) { return false; }
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        // ## FQ8: twins are REAL buffers (TMA'd gmem twins), and the transposed
        // exchanges live in their own buffers (smem_pt / smem_dst) written by
        // register scatter; sdS natural is the dQ B-operand, also read as a u16
        // view for packed pair stores.
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qt.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dot.data()), SmemLayoutdOt{});
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_kt.data()), SmemLayoutKt{});
        Tensor sPt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_pt.data()), SmemLayoutPdSt{});
        Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdS{});
        Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
        Tensor sdS16_pi = cute::as_position_independent_swizzle_tensor(cute::recast<uint16_t>(sdS));
        Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dst.data()), SmemLayoutPdSt{});
        Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
        // ## FQ8-V9: the in-smem-transposed twins carry m in the hardware
        // stmatrix interleave pi (per 16-col window: [0,1,8,9,2,3,10,11,...]).
        // The contraction over m is order-invariant when BOTH operands agree,
        // so the P-hat^T / dS^T exchange scatters write m through pi^{-1}.
        auto fq8_pi_m = [](int m) {
            return (m & ~15) | ((m & 6) << 1) | ((m & 8) >> 2) | (m & 1);
        };
        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});
        Tensor sLSEMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});

        static_assert(stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
                      stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
                      size<0>(typename TiledMmaSdP::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                      size<0>(typename TiledMmaSdP::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                      "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        constexpr int MmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        Layout warp_group_thread_layout_dq = make_layout(make_shape(Int<NumMmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaSdP tiled_mma_SdP;
        using TiledMmadP = std::conditional_t<!Mma_dP_is_RS, TiledMmaSdP, TiledMmadPRS>;
        TiledMmadP tiled_mma_dP;
        TiledMmadKV tiled_mma_dKV;
        TiledMmadQ tiled_mma_dQ;

        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dP = tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout_dq(warp_group_idx));

        // ## FQ8: the bf16 STSM PdS tiled-copy is replaced by manual stores (packed
        // u16 pairs for the natural dS tile; byte scatter for the transposed
        // P-hat/dS tiles): fp8 byte pairs do not fit the 16-bit STSM atoms.
        Tensor cS_fq8 = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
        Tensor tScS_fq8 = thread_mma_SdP.partition_C(cS_fq8);
        Tensor tScS_rc = make_tensor(tScS_fq8.data(),
            flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tScS_fq8.layout()));

        R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
        auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);
        // if (thread_idx == 0) { print(sdQ); printf("\n"); print(tdQsdQaccum); printf("\n"); }

        // Allocate "fragments/descriptors"
        // We have to use the templated mma_partition_fragment_AB instead of cute::conditional_return or lambda,
        // because some partition_fragment_A/B don't compile.
        // https://stackoverflow.com/questions/50051473/if-constexpr-in-c17-does-not-work-in-a-non-templated-function
        Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
        Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
        Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
        Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_dP, sV);
        Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);
        Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
        Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
        Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_PdS); print(sP_pi); printf("\n"); print(sPt_pi); printf("\n"); print(tPsP); printf("\n"); print(tdSsdS); printf("\n"); }

        // thread_mma_SdP.partition_C(sLSEMma) has shape ((2, 2, V), MMA_M, MMA_N, PIPE), we only take the col indices
        // or row indices, depending on whether SdP_swapAB.
        Tensor tLSEsLSE = cute::conditional_return<!SdP_swapAB>(
            group_modes<0, 2>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)),  // (2, MMA_M, PIPE)
            group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _, _)));  // (2, V, MMA_N, PIPE)
        Tensor tLSEsdPsum = cute::conditional_return<!SdP_swapAB>(
            group_modes<0, 2>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)),
            group_modes<0, 3>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_, _0{}, _), _0{}, _, _)));
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(sLSEMma); printf("\n"); print(tLSEsLSE); printf("\n"); }
        // If we want to split the stats among the 8 threads that share the same rows.
        static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tLSEsLSE))::value, 8);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int bidh = get<1>(block_coord);
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;

        // For the case where we do atomicAdd directly to gdQaccum instead of using TMA
        bool const is_varlen = Varlen && params.cu_seqlens_q;
        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, bidh, !is_varlen ? bidb : 0);
        Tensor gdQaccum_ = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)
        Tensor gdQaccum = cute::flat_divide(gdQaccum_, Int<kBlockM * kHeadDim / NumMmaWarpGroups>{});  // (M * K / WG, WG, _)
        // We can reuse r2s_thr_copy_dQaccum for this partitioning
        Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(mdQaccum); printf("\n"); print(gdQaccum_); printf("\n"); print(gdQaccum); printf("\n"); print(tdQgdQaccum); printf("\n"); }

        flash::Mask<kBlockM, kBlockN, false /*PackGQA*/, TiledMmaSdP, SdP_swapAB> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod
        );

        int m_block = m_block_min;
        clear(tdKrdK);
        clear(tdVrdV);
        // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

        // ## FP11-M4 (pass-1 interval promotion, the design notes).  The two
        // K-dim=N chains leave the row-long realization; the register ledger
        // (phase-A peak: two 64-reg fragments + tSrS/tdPrdP + working set =
        // the v10 budget; a THIRD live-across fragment measured +0.9 ms of
        // spill traffic) forces a split:
        //   dV: tdVrdV becomes a kFq8PromoteTiles-m-tile WINDOW accumulator
        //   (the wgmma C operand; in-window arithmetic = v10 VERBATIM,
        //   including the clamped f-ratio rescales), promoted at
        //   geometry-fixed boundaries into a persistent TRUE-unit fp32
        //   accumulator in the DEAD 64-KB smem_dqacc member (v10 never
        //   touches smem_dqacc: store_dq returns immediately and the dQ
        //   staging was removed; the epilogue's smem staging union window is
        //   the first 64 KB of TensorStorage while this member sits past the
        //   80-KB mark, so the exit read-back cannot race it).
        //   dK: plan A verbatim: a per-tile TRANSIENT 64-reg temp is the
        //   zero_init wgmma C operand, folded into tdKrdK (now the
        //   persistent TRUE-unit accumulator) with the tile's g basis
        //   immediately: realization = zero-init K=64 tensor-core chains +
        //   one fp32 FMA fold per tile (the g-rescale chain no longer
        //   exists, so the clamped-ratio compounding H-RC is structurally
        //   impossible; the F2/F2b psi-emission machinery, fq8_g_max
        //   included, is kept bit-exactly: it feeds the psi field and dS'
        //   codes, not the accumulation).  The temp lives only from the dK
        //   gemm to its fold, where tSrS/tdPrdP are dead: the proven v9
        //   128+64 envelope.
        // Determinism structural in both chains (fixed order, no atomics,
        // per-thread-owned smem slots).
        //
        // smem_dv overlay: [n=64 keys x d=256] fp32 on smem_dqacc.  Word
        // index  slot(n, d) = n * 256 + (d XOR ((n & 7) << 2))  is an in-row
        // XOR swizzle making the per-thread scalar fp32 zero-init / promote /
        // read-back loops bank-conflict-free BY CONSTRUCTION: at any unroll
        // step a warp's 32 lanes carry d = D0 + lane/4 (+8/128 offsets) and
        // n = N0 + 2*(lane%4), so the unswizzled words n*256 + d collide
        // 4-way on d mod 32 (lane/4 spans only 8 residues); XORing d bits
        // 2..4 with n bits 1..3 spreads the four lane%4 groups over the
        // remaining bank residues.  Verified exhaustively against the real
        // cute fragment map (all 8 warps
        // x 64 unroll steps multiplicity 1; ownership of all 16384 (d, n)
        // cells exactly once, 64 per thread; swizzle bijective in-row and
        // inside the 16384-float member).  Ownership uses the SAME
        // partition_C identity coordinates as the exit epilogue, so every
        // slot is written and read only by its one owning thread: no
        // barriers, no atomics.
        //
        // Addressing, hand strength-reduced (ptxas cannot see through the
        // XOR and would otherwise materialize 64 spilled address temps;
        // measured +600 B consumer spill).  The cute fragment map decomposes
        // EXACTLY as
        //     i = c + 2h + 4k + 32j  ->  n = n0 + c + 8k,  d = d0 + 8h + 128j
        // with (d0, n0) = the thread's first identity coord, n0 in [0,7], and
        // the XOR key (n&7)<<2 depends only on c (8k drops out) and is
        // bit-disjoint from the 128j term, so every slot is one of FOUR
        // per-thread bases plus a compile-time immediate:
        //     slot(i) = base[c][h] + 2048*k + 128*j,
        //     base[c][h] = (n0+c)*256 + ((d0+8h) ^ (((n0+c)&7)<<2)).
        // The k8s n-half at the dK exit is likewise compile-time: n < 32 <=>
        // k < 4 (n0+c <= 7), verified in the sim.
        static_assert(dQacc_use_TMA, "delta_matching: smem_dv overlays the 64-KB smem_dqacc member");
        static_assert(cute::cosize_v<SmemLayoutdQaccum> == kFq8DvOverlayFloats,
                      "delta_matching: the smem_dv overlay must exactly fit smem_dqacc");
        float* const fq8_smem_dv = reinterpret_cast<float*>(shared_storage.tensors.mainloop.smem_dqacc.data());
        float* fq8_dv_base[2][2];
        {
            auto thread_mma_dKV = tiled_mma_dKV.get_thread_slice(thread_idx);
            Tensor fq8_cdKV = cute::make_identity_tensor(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}));
            Tensor fq8_tdKVcdKV = thread_mma_dKV.partition_C(fq8_cdKV);
            CUTE_STATIC_ASSERT_V(size(fq8_tdKVcdKV) == size(tdVrdV));
            static_assert(decltype(size(tdVrdV))::value == 64,
                          "delta_matching: the c/h/k/j decomposition assumes the 64-value dKV C fragment");
            int const fq8_dv_d0 = get<0>(fq8_tdKVcdKV(_0{}));
            int const fq8_dv_n0 = get<1>(fq8_tdKVcdKV(_0{}));
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    fq8_dv_base[c][h] = fq8_smem_dv + (fq8_dv_n0 + c) * kHeadDim
                        + ((fq8_dv_d0 + 8 * h) ^ (((fq8_dv_n0 + c) & 7) << 2));
                }
            }
        }
        // ## Delta-Matching-128: the per-index immediate of the slot map.  d256 (M 128 = 2 atoms per WG, N 64):
        // i = c + 2h + 4k + 32j -> n = n0 + c + 8k, d = d0 + 8h + 128j: 2048*k + 128*j.  hd128 (M 64 = 1 atom per
        // WG, N 128): i = c + 2h + 4k with k in [0, 16) -> n = n0 + c + 8k, d = d0 + 8h: 8*kHeadDim*k = 1024*k
        // (the XOR key (n & 7) << 2 is unchanged: 8k drops out; pitch 128 == 0 mod 32 like 256, same conflict-free
        // argument).  Verified by the development notes (every (d, n) cell owned exactly once).
        auto fq8_dv_off = [](int i) {
            if constexpr (kHeadDim >= 256) { return 2048 * ((i >> 2) & 7) + 128 * ((i >> 5) & 1); }
            else { return 1024 * ((i >> 2) & 15); }
        };
        #pragma unroll
        for (int i = 0; i < size(tdVrdV); ++i) {
            fq8_dv_base[i & 1][(i >> 1) & 1][fq8_dv_off(i)] = 0.0f;
        }

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.pipelines.barrier_KV.wait(work_idx % 2); }

        if constexpr (Mma_dP_is_RS) {
            using SmemCopyAtomV = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
            auto smem_tiled_copy_V = make_tiled_copy_A(SmemCopyAtomV{}, tiled_mma_dP);
            auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
            Tensor tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
            Tensor tdPsV_copy_view = smem_thr_copy_V.partition_S(cute::as_position_independent_swizzle_tensor(sV));
            cute::copy(smem_tiled_copy_V, tdPsV_copy_view, tdPrV_copy_view);
        }

        // ## FQ8: scale bookkeeping.  q8s/do8s: one fp32 per 64-row m-block; k8s/
        // v8s: one per 32-row n-block, this CTA covers blocks {2*n_block, 2*n_block
        // +1} and each warpgroup owns one 32-column half of the SdP tile (WG w ->
        // half w by the tiled-mma N-mode order).  MO: P-hat = 2^8 * P folds into
        // exp2 as +8; the 2^8 rides dS into psi and is divided out downstream.
        int const fq8_bh = get<2>(block_coord) * get<2>(params.shape_Q) + bidh;
        // ## FP11-M16 (native GQA, READ side): the K/V scales are indexed by
        // KV head (b, h_k, nblk_k), exactly as the K/V/Kt TMA tensors already
        // are (bidh_kv).  With h_k == h this is the M10 arithmetic verbatim
        // (divide by 1); with h_k < h the bytes read per query head are the
        // same as from an expanded copy: bitwise by construction.  Every
        // per-query-head workspace (dS8, psi, delta partials) keeps fq8_bh.
        int const fq8_bh_kv = get<2>(block_coord) * get<2>(params.shape_K)
            + params.qhead_per_khead_divmod.divide(bidh);
        float const fq8_k8s_wg = params.ptr_k8s[fq8_bh_kv * params.nblk_k + 2 * n_block + warp_group_idx];
        float const fq8_k8s_g0 = params.ptr_k8s[fq8_bh_kv * params.nblk_k + 2 * n_block];
        float const fq8_k8s_g1 = params.ptr_k8s[fq8_bh_kv * params.nblk_k + 2 * n_block + 1];
        float const fq8_v8s_wg = params.ptr_v8s[fq8_bh_kv * params.nblk_k + 2 * n_block + warp_group_idx];
        float const* fq8_q8s_base = params.ptr_q8s + fq8_bh * params.nblk_q;
        float const* fq8_do8s_base = params.ptr_do8s + fq8_bh * params.nblk_q;
        // ## FQ8-V9: software-pipeline the per-m-block scale loads (global
        // scalar fetches were a top long-scoreboard source).
        // ## Delta-Matching-128: the Q/dO scale block index of an m-block (m-block at d256; m-block / 2 at hd128).
        auto fq8_qblk = [](int m) { if constexpr (kQShift == 0) { return m; } else { return m >> kQShift; } };
        float fq8_q8sc_next = fq8_q8s_base[fq8_qblk(m_block_min)];
        float fq8_do8sc_next = fq8_do8s_base[fq8_qblk(m_block_min)];
        // ## FP11-M4 window state.  dV's per-tile rescale machinery is v10
        // VERBATIM (f_prev basis, clamped ratio, no branches in the tile
        // body: the window is re-zeroed at each boundary, so a window's
        // first tile rescales a zero fragment, value-identical to a
        // zero_init start; f_last is gone because each window's basis is
        // folded at its promote).  At a boundary, f_prev holds the basis of
        // the window's LAST tile, i.e. exactly the units the window fragment
        // is in: promote = smem_dv += fragment * f_prev.  After the promote
        // the cross-window rescale chain and the row-long in-place
        // tensor-core re-round chain no longer exist (the derivation notes: per-window
        // TC error ~1.22e-4-class, cross-window pure fp32).  The dK chain
        // has NO basis-chain state left: its per-tile temp is folded by
        // g_cur = psi_i/2^8 * q8sc(i) at each fold (g_prev / g_last / the
        // g-ratio clamp are gone; fq8_g_max below stays, it feeds the psi
        // emission floor).
        float fq8_f_prev = 1.0f;   // dV window basis (2^-8 * do8sc(i))
        // ## FP11-F2b: running max of the dK basis (0 until the first
        // non-sentinel tile); anchors the psi emission floor.  ROW-GLOBAL
        // (never reset per window): its update sequence is bit-identical to
        // M2/M3's, so the psi field and dS' codes pass 2 consumes are
        // byte-identical (the M4 dq-bitwise guarantee).
        float fq8_g_max = 0.0f;
        float* fq8_psired = shared_storage.tensors.mainloop.smem_psired.data();
        static constexpr float kFq8MoE = 8.0f;          // log2(2^8)
        static constexpr float kFq8MoInv = 1.0f / 256.0f;
        // ## FP11-M4: dK promotion window length in m-tiles.  PINNED at 8
        // (512 query rows): ONE realization class per the design notes, no build knob;
        // the same constant as pass 2's kPromoteTiles.  Boundaries are
        // geometry-fixed at absolute m-tile indices ((m_block & 7) == 7) plus
        // the row tail, so the m-axis window partition is global (independent
        // of the CTA's causal m_block_min; a causal CTA's first window may be
        // short), mirroring pass 2's absolute-nb boundaries.  Skipped
        // (sentinel) tiles are TRANSPARENT exactly as in pass 2: they neither
        // join nor end a window and leave the bases untouched.  (The dV chain
        // is per-tile promoted through the plan-A transient temp: its
        // effective interval is one tile; see the plan-B split note at the
        // accumulator setup.)
        constexpr int kFq8PromoteTiles = 8;
        static_assert((kFq8PromoteTiles & (kFq8PromoteTiles - 1)) == 0,
                      "delta_matching: kFq8PromoteTiles must be a power of two (mask boundary test)");
        // ## P1-C': the same 512-row window counted in stages (8 x 64-row tiles = 4 x 128-row stages).
        constexpr int kFq8PromoteTilesStage = kFq8PromoteTiles / kHalves;
        static_assert(kFq8PromoteTilesStage * kBlockM == 512, "delta_matching: the dV promote window is 512 rows");
        // ## FP11-F2 (ratio-clamp scope, MEASURED on a stress input): with F1 and
        // the zero-tile sentinel alone, non-sentinel psi still spans ~30 decades
        // at cliff-scale inputs, and consecutive >= 8-decade psi collapses along
        // a chain each bind the up-clamp: every bind multiplies the accumulator
        // by 1e8, and the binds COMPOUND (measured: dq/dk = +inf, 0 NaN, dv
        // finite, bad blocks matching the predicted overflow chains).  Repair
        // (FP11-F2b): the dK chain floors the EMITTED psi at 1e-8 x the chain's
        // max basis (fq8_g_max below), so every up-rescale is exact (<= 1e8)
        // and the acc magnitude is bounded by 448*448*64*1e8 ~ 1.3e15; content
        // below the floor is < fp32 epsilon of the chain's dominant tile
        // (every non-floored tile's max |code| is 448 by construction), so
        // flushing it in the e4m3 codes is at least as accurate as v10's
        // clamped mis-scale.  Pass 2 applies the same dominance rule by
        // skipping tiles > 8 decades below its running psi max (fq8_dq_pass2.h).
        // The down-clamp (1e-8) can still bind on an upward basis jump: it
        // over-weights then-negligible old content by <= 1e-8 relative, shrinks
        // acc (never overflows), and does not compound: left as v10.  The dV
        // f-chain basis is do8sc-only (data-bounded ratios ~ O(1)): left as v10.
        auto fq8_clamp_ratio = [](float r) { return fminf(fmaxf(r, 1e-8f), 1e8f); };

        auto bwd_step = [&](int m_block, auto mask_fn) {
            // ## FP11-M4: the plan-A dK temp (written by this tile's
            // zero_init dK gemm, folded at the loop tail) and its per-tile
            // fold basis (0.0f on sentinel tiles: exact no-op fold).
            Tensor fq8_tmpK = make_fragment_like(tdKrdK);
            float fq8_g_fold = 0.0f;
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            consumer_wait(pipeline_q, smem_pipe_read);
            if constexpr (!(kAbl & 8)) {
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
            }
            Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(make_fragment_like(tLSEsLSE(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            if constexpr (!ShuffleLSE) {
                cute::copy(tLSEsLSE(_, smem_pipe_read.index()), tLSErLSE);
            } else {
                #pragma unroll
                for (int i = 0; i < kStatsPerThread; ++i) {
                    // It's ok to read OOB, since we made sure sLSE is large enough and we won't use the OOB values
                    tLSErLSE(i) = tLSEsLSE((thread_idx % 32) / 4 + i * 8, smem_pipe_read.index());
                }
            }
            // ## FQ8: this m-block's row-block scales (uniform over the tile row);
            // consumed from the software pipeline, next iteration's prefetched now.
            float const fq8_q8sc = fq8_q8sc_next;
            float const fq8_do8sc = fq8_do8sc_next;
            if (m_block + 1 < m_block_max) {
                fq8_q8sc_next = fq8_q8s_base[fq8_qblk(m_block + 1)];
                fq8_do8sc_next = fq8_do8s_base[fq8_qblk(m_block + 1)];
            }
            float const fq8_s_descale_log2 = fq8_q8sc * fq8_k8s_wg * float(M_LOG2E);
            float const fq8_dp_descale = fq8_do8sc * fq8_v8s_wg;
            float const fq8_dp_descale_f = fq8_dp_descale * fq8_k8s_wg;
            Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
            consumer_wait(pipeline_do, smem_pipe_read_do_cur);
            if constexpr (!(kAbl & 8)) {
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_dP, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);
            }
            warpgroup_wait<1>();
            // ## FP11-M17 GT: the S gemm (raw Q's ONLY reader: the twin comes from
            // gmem now) has retired and this stage's LSE was copied to registers
            // above: release raw Q here so the producer refills it ~a tile
            // earlier than the M16 release point (after the qt wait).  Pipeline
            // timing only.
            pipeline_q.consumer_release(smem_pipe_read);
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, params.softcap_val); }

            // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
            // dtanh needs to happen before masking, otherwise we get 1 - (-inf)^2 = NaN in the dtanh
            auto dtanh = [&] { if constexpr (Has_softcap) return flash::calculate_dtanh(scores); else return nullptr; }();
            mask_fn(tSrS, m_block);
            if constexpr (!(kAbl & 32)) {
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
                float const lse_scaled = [&] {
                    float const lse_raw = [&] {
                        if constexpr (!ShuffleLSE) return tLSErLSE(mi);
                        else return __shfl_sync(0xffffffff, tLSErLSE(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                    }();
                    return __fmul_rn(lse_raw, float(M_LOG2E));   // ## FP11-M17 LSEK: the host prep's exact multiply
                }();
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
                    // ## FQ8: true S = q8sc(m-block) * k8sc(n-half) * codes; P-hat =
                    // 2^8 * exp(S - lse) folds the MO offset as +8 in exp2 space.
                    // ## FP11-F1 (cliff clamp): in exact arithmetic the argument is
                    // <= kFq8MoE (LSE >= row-max S), and healthy fwd/bwd realization
                    // mismatch is < ~1.5 log2-units, so kFq8MoE + 4 is unreachable
                    // by any healthy input: the fminf is bitwise-invisible there.
                    // At ANY S-vs-LSE inconsistency (the |S|-proportional 226-403
                    // cliff, the development notes) it caps P-hat at 2^12 instead of
                    // exp2f overflowing to +inf at arg >= 128, which was the
                    // mechanized dq/dk NaN: inf dS -> psi = inf -> psi_inv = 0 ->
                    // inf*0 = NaN in the dS' codes feeding the dK and pass-2 dQ
                    // GEMMs.  Masked lanes: fminf(-inf, 12) = -inf -> exp2f = 0.
                    scores(mi, ni) = exp2f(fminf(scores(mi, ni) * fq8_s_descale_log2 - lse_scaled + kFq8MoE,
                                                 kFq8MoE + 4.0f));
                }
            }
            }   // kAbl & 32

            Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(make_fragment_like(tLSEsdPsum(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            if constexpr (!ShuffledPsum) {
                cute::copy(tLSEsdPsum(_, smem_pipe_read_do_cur.index()), tLSErdPsum);
            } else {
                #pragma unroll
                for (int i = 0; i < kStatsPerThread; ++i) {
                    tLSErdPsum(i) = tLSEsdPsum((thread_idx % 32) / 4 + i * 8, smem_pipe_read_do_cur.index());
                }
            }

            warpgroup_wait<0>();
            // ## FP11-M17 GT: the dP gemm (raw dO's only reader) has retired and
            // dPsum was copied to registers above: release raw dO now.
            pipeline_do.consumer_release(smem_pipe_read_do_cur);
            // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
#ifdef FA3FP11_DBG
            // ## FP11-M5-DBG (checksum, the design notes M5 requirement 1): XOR of the
            // realized P~ (= tSrS post-exp2/mask) and raw dP~ (= tdPrdP before
            // the dS overwrite) fragment bits, per (bh, m-tile, n-tile).  XOR
            // is associative/commutative, so warp shfl-fold + one atomicXor
            // per warp per plane is order-free and deterministic.  The SAME
            // two capture points exist in mma_pass0: bit-matching realizations
            // give equal checksums per causal tile.  DBG builds only; the
            // shipped binary contains none of this code.
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
                if ((thread_idx % cutlass::NumThreadsPerWarp) == 0) {
                    int64_t const fq8_nt = int64_t(params.num_batch) * get<2>(params.shape_Q)
                        * params.nbm * params.nbn;
                    int64_t const fq8_ti = (int64_t(fq8_bh) * params.nbm + m_block) * params.nbn + n_block;
                    atomicXor(params.ptr_dbg_cks + fq8_ti, fq8_xp);
                    atomicXor(params.ptr_dbg_cks + fq8_nt + fq8_ti, fq8_xd);
                }
            }
            // ## FP11-M5-DBG (G2 rowsum residual): per-(tile, warpgroup) row
            // sums of the realized dS in TRUE (un-k8s-folded) units:
            //   t(i,ni) = P~ * (dP~_raw * fq8_dp_descale - delta_i)
            // = the register dS line's realized inputs with the k8s fold
            // removed (the fold is a per-32-key column weight that would make
            // the row sum non-zero even in exact arithmetic; removing it is
            // exactly dividing each realized dS element by its k8s_wg, up to
            // one fp32 rounding per element).  Summation: per-thread ni
            // ascending in the rowcol fragment order, quad shfl-tree
            // (Allreduce<4>: offsets 2 then 1, the softmax discipline); the
            // per-WG 32-key half-sums are stored separately at
            // [bh][nb][wg][row] and the test reduces (nb, wg) in fixed
            // ascending order.  Reads only registers the real dS loop is
            // about to consume; runs BEFORE it so raw dP is intact.
            if (params.ptr_dbg_rows != nullptr) {
                flash::SumOp<float> fq8_dbg_sum;
                #pragma unroll
                for (int mi = 0; mi < size<0>(dS); ++mi) {
                    float const fq8_dbg_dsum = [&] {
                        if constexpr (!ShuffledPsum) { return tLSErdPsum(mi); }
                        else { return __shfl_sync(0xffffffff, tLSErdPsum(mi / 8), (mi % 8) * 4 + (thread_idx % 4)); }
                    }();
                    float fq8_ra = 0.f, fq8_rb = 0.f;
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) {
                        float const t = scores(mi, ni) * (dS(mi, ni) * fq8_dp_descale - fq8_dbg_dsum);
                        fq8_ra += t;
                        fq8_rb += fabsf(t);
                    }
                    fq8_ra = flash::Allreduce<4>::run(fq8_ra, fq8_dbg_sum);
                    fq8_rb = flash::Allreduce<4>::run(fq8_rb, fq8_dbg_sum);
                    if ((thread_idx & 3) == 0) {
                        int const fq8_row = get<0>(tScS_rc(mi, _0{}));
                        int64_t const fq8_gi = ((int64_t(fq8_bh) * params.nbn + n_block) * 2
                            + warp_group_idx) * int64_t(seqlen_q) + m_block * kBlockM + fq8_row;
                        params.ptr_dbg_rows[fq8_gi] = fq8_ra;
                        params.ptr_dbg_rowsabs[fq8_gi] = fq8_rb;
                    }
                }
            }
#endif
            if constexpr (!(kAbl & 32)) {
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                float const dP_sum_cur = [&] {
                    if constexpr (!ShuffledPsum) return tLSErdPsum(mi);
                    else return __shfl_sync(0xffffffff, tLSErdPsum(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                }();
                float const dP_sum_cur_f = dP_sum_cur * fq8_k8s_wg;
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) {
                    // ## FQ8: dP_true = codes * do8sc * v8sc(n-half); dS here is
                    // 2^8 * dS_true (the MO factor rides in via scores = 2^8 * P;
                    // psi absorbs it and downstream factors divide it out).
                    // ## FQ8-V9 FOLD: dS' = dS * k8s_wg (the dQ B-operand K-scale
                    // folded into the codes; psi absorbs it, the dK epilogue
                    // divides it back out).  Kills the split-K dQ.
                    dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) * fq8_dp_descale_f - dP_sum_cur_f);
                    if constexpr (Has_softcap) { dS(mi, ni) *= dtanh(mi, ni); }
                }
            }
            }   // kAbl & 32
            // ## FQ8: psi = amax(|dS_tile|)/448 across BOTH warpgroups (the tile is
            // column-split per WG): warp-local amax -> per-warp smem slot -> barrier
            // -> joint max.  Clamp matches the reference (max(amax/448, 1e-30)).
            float fq8_amax = 0.f;
            #pragma unroll
            for (int i = 0; i < size(tdPrdP); ++i) { fq8_amax = fmaxf(fq8_amax, fabsf(tdPrdP(i))); }
            #pragma unroll
            for (int off = 16; off > 0; off /= 2) {
                fq8_amax = fmaxf(fq8_amax, __shfl_xor_sync(0xffffffff, fq8_amax, off));
            }
            if constexpr (!(kAbl & 2)) {
            int const fq8_warp = thread_idx / cutlass::NumThreadsPerWarp;   // 0..7 over MMA threads
            if ((thread_idx % cutlass::NumThreadsPerWarp) == 0) { fq8_psired[fq8_warp] = fq8_amax; }
            cutlass::arch::fence_view_async_shared();
            // ## FQ8-V8: psi gets its OWN barrier id (dQEmptyWG3, unused at 2 WGs);
            // v1 borrowed dQEmptyWG1 which the TMA-dQ protocol now owns.
            cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG3) /*id: free at 2 WGs*/);
            #pragma unroll
            for (int w = 0; w < NumMmaThreads / cutlass::NumThreadsPerWarp; ++w) {
                fq8_amax = fmaxf(fq8_amax, fq8_psired[w]);
            }
            }   // kAbl & 2
            // ## FP11-F2 (psi-floor repair): a tile whose |dS| amax sits below the
            // floor threshold (amax/448 < 1e-30) is fully underflowed or fully
            // masked.  Post-F1 this is still reachable on non-degenerate data:
            // F1 clamps the exp2 argument only from ABOVE; exp2f underflows to
            // subnormal/0 for S - LSE < ~ -93 nats (fast-math FTZ flush at
            // arg < -126; exact-zero underflow at -108.8 nats), first reachable
            // in exactly the cliff's |S| magnitude regime (all 64 rows of an
            // m-block cold against one n-block).  v10 boosted such tiles by psi_inv = 1e30
            // (nonzero garbage codes whenever dS was subnormal-tiny instead of
            // exact 0) and dragged the running dK g-basis / published pass-2 psi
            // to the floor, mis-scaling every OTHER tile's accumulated
            // contribution by up to ~1e-20 whenever a basis chain ENDS on a
            // floor tile (ratio clamps at 1e8 against a true ratio ~1e28).
            // Repair: emit EXACT ZERO dS' codes (psi_inv = 0; the discarded true
            // tile contribution is < 4.48e-28 in dS units, ~17+ orders below the
            // fp32 epsilon of any healthy tile's), skip the g-basis update (a
            // zero tile adds exactly zero in ANY basis; see the FP11-F2 site in
            // the dK rescale below), and publish psi = 0.0f as the pass-2
            // sentinel (pass 2 skips its rescale + basis update on it).  Healthy
            // tiles take the v10 path bit-for-bit.
            bool const fq8_tile_zero = fq8_amax * (1.0f / 448.0f) < 1e-30f;
            // ## FP11-F2b (chain-relative emission floor; see the fq8_clamp_ratio
            // comment): psi is floored at 1e-8 x the dK chain's max basis
            // (rescaled to psi units: g = psi * q8sc/256), so g_cur >= 1e-8 *
            // fq8_g_max always, the up-rescale is exact, and the dK accumulator
            // cannot overflow.  Healthy inputs: the floor sits >= 5 decades
            // below any healthy psi, fmaxf returns psi_raw bit-identically.
            // fq8_g_max = 0 before the first non-sentinel tile: floor inert.
            float const fq8_psi_raw = fmaxf(fq8_amax * (1.0f / 448.0f), 1e-30f);
            float const fq8_psi = fmaxf(fq8_psi_raw, fq8_g_max * (256.0f * 1e-8f) / fq8_q8sc);
            float const fq8_psi_inv = fq8_tile_zero ? 0.0f : 1.0f / fq8_psi;
            // ## V10 (split bwd): publish this tile's psi for pass 2 (0.0f = the
            // FP11-F2 zero-tile sentinel).
            if (thread_idx == 0) {
                params.ptr_psi[(int64_t(fq8_bh) * params.nbm + m_block) * params.nbn + n_block]
                    = fq8_tile_zero ? 0.0f : fq8_psi;
            }

            // ## FQ8: P-hat codes = e4m3(2^8 * P) (scores already carry 2^8; the
            // e4m3 satfinite converter matches the reference rounding).  P-hat is
            // consumed ONLY transposed (dV's A operand): scatter bytes straight
            // into the (n, m) exchange.
            Tensor rP = make_tensor_like<Element>(tSrS);
            flash::convert_type_out(tSrS, rP);
            Tensor rP_rc = make_tensor(rP.data(), scores.layout());
            if constexpr (kStages_dS == 1 && !(kAbl & 4)) {
                // ## V10: the async dS8 tile-store must retire before this
                // iteration's stores overwrite sdS (barrier orders everyone
                // behind the issuing thread's wait).
                if (thread_idx == 0) { cute::tma_store_wait<0>(); }
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
            }
            if constexpr (!(kAbl & 4)) {
            #pragma unroll
            for (int mi = 0; mi < size<0>(rP_rc); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(rP_rc); ++ni) {
                    auto crd = tScS_rc(mi, ni);
                    sPt_pi(get<1>(crd), fq8_pi_m(get<0>(crd)), _0{}) = rP_rc(mi, ni);
                }
            }
            // ## FQ8: dS8 codes = e4m3(dS/psi) (identical to the reference: the 2^8
            // factor cancels between dS and psi).  Natural (m, n) tile: packed u16
            // PAIR stores (the C fragment's even/odd columns are byte-adjacent);
            // transposed (n, m) tile: byte scatter.  The PdS barrier before these
            // stores covers the previous iteration's reads (P-hat scatter above
            // already synced it); the fence + barrier AFTER publish for the gemms.
            #pragma unroll
            for (int i = 0; i < size(tdPrdP); ++i) { tdPrdP(i) *= fq8_psi_inv; }
            Tensor rdS = make_tensor_like<Element>(tdPrdP);
            flash::convert_type_out(tdPrdP, rdS);
            Tensor rdS_rc = make_tensor(rdS.data(), scores.layout());
            #pragma unroll
            for (int mi = 0; mi < size<0>(rdS_rc); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(rdS_rc); ni += 2) {
                    auto crd = tScS_rc(mi, ni);
                    uint16_t const fq8_pack =
                        uint16_t(rdS_rc(mi, ni).storage)
                        | (uint16_t(rdS_rc(mi, ni + 1).storage) << 8);
                    sdS16_pi(get<0>(crd), get<1>(crd) / 2, _0{}) = fq8_pack;
                    auto crd1 = tScS_rc(mi, ni + 1);
                    sdSt_pi(get<1>(crd), fq8_pi_m(get<0>(crd)), _0{}) = rdS_rc(mi, ni);
                    sdSt_pi(get<1>(crd1), fq8_pi_m(get<0>(crd1)), _0{}) = rdS_rc(mi, ni + 1);
                }
            }
            }   // kAbl & 4 (exchange stores)

            if constexpr (!Slice_dQKV_Mma) {
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                // Most cases take this path, except for hdim256 where we want to slice to reduce register pressure
                if constexpr (Mma_dKV_is_RS) {
                    Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_dot.index()), tdVrdV);
                } else {
                    // ## FQ8 v3 (FIFO reorder): the old chain's two mid wait(0)s
                    // drained EVERY in-flight gemm (dV serialized under the dQ
                    // rescales).  New issue order exploits FIFO retire:
                    //   dQ_k0(-1), dV(-1), wait<1>   -> retires dQ_k0, dV flies
                    //   rescale dQ, dQ_k1(-1), dK(-1), wait<1>
                    //                                -> retires dV + dQ_k1, dK flies
                    //   dQ scale + atomics           -> hidden under dK
                    // All accumulator pre-rescales (idle at entry: the loop tail's
                    // wait<0> retired everything) run before any issue.
                    if constexpr (!(kAbl & 4)) {
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
                    }
                    // ## V10 (split bwd): async tile-store of the published
                    // natural dS' codes (sdS is otherwise dead: the dQ gemm
                    // was its only consumer).  Drained at the next
                    // iteration's PdS wait.
                    if (!(kAbl & 4) && thread_idx == 0) {
                        Tensor mdS8 = params.tma_store_dS8.get_tma_tensor(
                            make_shape(Int<kBlockM>{}, Int<kBlockN>{}, params.nbn, params.nbm,
                                       params.num_batch * get<2>(params.shape_Q)));
                        Tensor gdS8 = mdS8(_, _, n_block, m_block, fq8_bh);
                        auto blk_tma_dS8 = params.tma_store_dS8.get_slice(_0{});
                        Tensor tdS8s = blk_tma_dS8.partition_S(sdS(_, _, _0{}));
                        Tensor tdS8g = blk_tma_dS8.partition_D(gdS8);
                        cute::copy(params.tma_store_dS8, tdS8s, tdS8g);
                        cute::tma_store_arrive();
                    }
                    // ## FP11-M4: in-window dV rescale = the v10 sequence
                    // VERBATIM (the window fragment was re-zeroed at the last
                    // boundary, so a window's first tile rescales exact
                    // zeros: value-identical to a zero_init chain start, and
                    // the tile body stays branch-free).
                    float const fq8_f_cur = kFq8MoInv * fq8_do8sc;
                    float const fq8_rf = fq8_clamp_ratio(fq8_f_prev / fq8_f_cur);
                    #pragma unroll
                    for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) *= fq8_rf; }
                    fq8_f_prev = fq8_f_cur;
                    // ## FP11-F2 + M4: a zero tile (psi-floor sentinel, see
                    // the psi site above) contributes exact zeros: its dS'
                    // codes are all zero, so its dK temp gemm below produces
                    // exact zeros and its fold basis is set to 0.0f (an
                    // exact no-op fold): the transparent-skip rule.  For
                    // taken tiles the fold basis is the tile's own
                    // g_cur = psi * 2^-8 * q8sc, and fq8_g_max advances
                    // EXACTLY as in M2/M3 (same predicate, same value, same
                    // order), keeping the psi emission floor, hence the psi
                    // field and dS' codes pass 2 consumes, byte-identical
                    // (the M4 dq-bitwise guarantee).  With the per-tile fold
                    // there is no g-ratio chain at all: the clamped-ratio
                    // compounding mechanism H-RC is
                    // structurally impossible in pass 1, as M3 made it in
                    // pass 2.
                    fq8_g_fold = 0.0f;
                    if (!fq8_tile_zero) {
                        float const fq8_g_cur = fq8_psi * kFq8MoInv * fq8_q8sc;
                        fq8_g_fold = fq8_g_cur;
                        // ## FP11-F2b: advance the floor anchor (row-global;
                        // bit-identical update sequence to M2/M3).
                        fq8_g_max = fmaxf(fq8_g_max, fq8_g_cur);
                    }

                    Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                    static_assert(decltype(size<2>(tdQrK))::value == kBlockN / 32,
                                  "delta_matching: dQ contraction must be kBlockN/32 fp8 k-atoms");
                    Tensor tdVrPt = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
                    Tensor tdVrPt_cur = tdVrPt(_, _, _, _0{});
                    Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
                    Tensor tdKrdS_cur = tdKrdS(_, _, _, _0{});

                    // ## V10 (split bwd): dQ gemm removed; the pass-2 GEMM
                    // consumes the stored dS' codes instead.
                    // ## FQ8-V9: sdOt is produced by the transpose warps.  The dot
                    // wait ALSO proves the transposer is done reading raw dO, and
                    // the dP gemm (raw dO's only MMA reader) retired at the dS
                    // stage: release raw dO here so the 1-stage refill starts a
                    // dV-gemm earlier.
                    if constexpr (!(kAbl & 256)) { consumer_wait(pipeline_dot, smem_pipe_read_dot); }
                    // ## FP11-M16 DO2: the transposed twin is indexed by ITS pipeline's
                    // stage (1 stage under DO2; equal to the raw index in the M10 config).
                    if constexpr (!(kAbl & 1)) {
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdVrPt_cur, tdVrdO(_, _, _, smem_pipe_read_dot.index()), tdVrdV);
                    }
                    // ## FQ8-V9: sQt is produced by the transpose warps.  The qt
                    // wait proves the transposer is done with raw q, and the S gemm
                    // (raw q's only MMA reader) retired at softmax: release raw q
                    // now for deeper producer prefetch.
                    if constexpr (!(kAbl & 256)) { consumer_wait(pipeline_qt, smem_pipe_read_qt); }
                    // ## FP11-M4 plan-A dK: the tile's dK chunk lands in the
                    // TRANSIENT zero_init temp (live only from here to its
                    // fold at the loop tail, where tSrS/tdPrdP are dead: the
                    // v9 128+64 envelope).  FIFO: the dV window gemm flies
                    // under the qt wait + q release + dK issue exactly as
                    // v10; wait<1> retires dV with the dK temp still flying,
                    // drained at the loop tail's wait<0> right before the
                    // fold.
                    if constexpr (!(kAbl & 1)) {
                    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read_qt.index()), fq8_tmpK);
                    }
                    warpgroup_wait<1>();   // retires dV; the dK temp stays in flight
                    if constexpr (!(kAbl & 256)) { pipeline_dot.consumer_release(smem_pipe_read_dot); }  // dV done
                    ++smem_pipe_read_dot;
                    // ## V10: no dQ rescale (psi_mn rides to pass 2 as the
                    // per-tile dS' scale).
                }
                // ## V10 (split bwd): dQ staging removed.

            } else {  // Slice_dQKV_Mma

                static_assert(!(Slice_dQKV_Mma && Mma_dKV_is_RS));
                Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
                Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_dot.index()), tdVrdV);

                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/dQ_swapAB, /*M_slice=*/0>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_dot.index()), tdVrdV);
                Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
                Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
                #pragma unroll
                for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

                Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
                Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read_qt.index()), tdKrdK);
                pipeline_do.consumer_release(smem_pipe_read_do_cur);  // release dO

                flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB, /*M_slice=*/1>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                #pragma unroll
                for (int i = size(tdQrdQ_atomic) / 2;  i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read_qt.index()), tdKrdK);
            }

            warpgroup_wait<0>();
            if constexpr (!(kAbl & 256)) { pipeline_qt.consumer_release(smem_pipe_read_qt); }   // dK issued; sQt free next round
            ++smem_pipe_read_qt;
            ++smem_pipe_read;
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }

            // ## FP11-M4 plan-A dK fold: the wait<0> above retired the dK
            // temp gemm; fold the tile's chunk into the persistent TRUE-unit
            // tdKrdK with the tile's own g basis (sentinel tiles fold exact
            // zeros with basis 0.0f).  One fp32 FMA per element, per tile:
            // no cross-tile tensor-core chain, no ratio chain.
            #pragma unroll
            for (int i = 0; i < size(tdKrdK); ++i) {
                tdKrdK(i) = fmaf(fq8_tmpK(i), fq8_g_fold, tdKrdK(i));
            }

            // ## FP11-M4 dV window boundary: geometry-fixed (absolute m-tile
            // index mask + the row tail), data-independent; the window
            // fragment is idle here (the wait<1> above retired the dV gemm,
            // and the wait<0> everything).  Promote in TRUE units (fold the
            // window basis NOW: this replaces v10's exit-epilogue f_last
            // fold), then re-zero the window so the next window's first tile
            // rescales exact zeros (the branch-free chain start).  f_prev
            // holds exactly the units of the window fragment.  The promote
            // overlaps the next tile's TMA waits.
            if ((m_block & (kFq8PromoteTiles - 1)) == kFq8PromoteTiles - 1
                    || m_block == m_block_max - 1) {
                #pragma unroll
                for (int i = 0; i < size(tdVrdV); ++i) {
                    float* const slot = fq8_dv_base[i & 1][(i >> 1) & 1] + fq8_dv_off(i);
                    *slot = fmaf(tdVrdV(i), fq8_f_prev, *slot);
                    tdVrdV(i) = 0.0f;
                }
            }
        };

        // ## P1-C' (Delta-Matching-128, kHalves == 2): the two-half step.  Per 128-row STAGE: the stage's Q / dO scales, the dV window
        // rescale (one do8sc per stage: the P1-A per-tile rescale of the second 64-row tile by f/f == 1 is an exact no-op, so
        // the wgmma chain is identical), then for each 64-row half h the P1-A tile body VERBATIM on the half's fragments (S / dP
        // gemms on the stage tile's M-atom h, its own LSE / dPsum rows, mask at m64 = 2 m_block + h, exp2, dS line, psi over
        // both WGs' column halves, P~^T / dS'^T / dS' exchange + the [64, kBlockN] TMA store at (n_block, m64), dV window gemm
        // (K = the half's 64 rows of the dOt twin = k-atoms {2h, 2h+1} of the stage twin), the zero-init dK temp gemm on the
        // Qt twin's k-atoms {2h, 2h+1} and its fold), then the stage's pipeline releases and the dV promote boundary
        // (kFq8PromoteTilesStage stages = 512 rows).  dq / dk / dv are bitwise P1-A by construction (same per-tile arithmetic,
        // same order, same slots); only the number of TMA transactions and pipeline round trips per row halves.
        Tensor cS_h = cute::make_identity_tensor(select<0, 1>(TileShape_MhNK{}));
        Tensor tScS_h = thread_mma_SdP.partition_C(cS_h);
        Tensor tScS_rc_h = make_tensor(tScS_h.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tScS_h.layout()));
        flash::Mask<kBlockMh, kBlockN, false /*PackGQA*/, TiledMmaSdP, SdP_swapAB> mask_h(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod);
        // A fragment of the stage tile restricted to M-atom h, re-wrapped with a singleton MMA_M mode (as the forward's slab views).
        auto fq8_half_A = [&](auto const& t, int h, int stage) {
            auto s_ = t(_, h, _, stage);
            return make_tensor(s_.data(), make_layout(get<0>(s_.layout()), Layout<_1, _0>{}, get<1>(s_.layout())));
        };
        auto bwd_step2 = [&](int m_block, auto mask_fn_unused) {
            // (only called when kHalves == 2; the lambda body is type-checked at every head dim, so no static_assert here)
            // ---- stage-level: scales (one Q / dO scale block per 128-row stage), the dV window rescale
            float const fq8_q8sc = fq8_q8sc_next;
            float const fq8_do8sc = fq8_do8sc_next;
            if (m_block + 1 < m_block_max) {
                fq8_q8sc_next = fq8_q8s_base[fq8_qblk(m_block + 1)];
                fq8_do8sc_next = fq8_do8s_base[fq8_qblk(m_block + 1)];
            }
            float const fq8_s_descale_log2 = fq8_q8sc * fq8_k8s_wg * float(M_LOG2E);
            float const fq8_dp_descale = fq8_do8sc * fq8_v8s_wg;
            float const fq8_dp_descale_f = fq8_dp_descale * fq8_k8s_wg;
            consumer_wait(pipeline_q, smem_pipe_read);
            PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
            consumer_wait(pipeline_do, smem_pipe_read_do_cur);
            // A RUNTIME loop over the two halves (one body): every register fragment is indexed by compile-time constants; h
            // only enters smem descriptor arithmetic (the stage tile's M-atom, the twins' k-atoms) and pointer offsets.
            #pragma unroll 1
            for (int h = 0; h < kHalves; ++h) {
                bool const last_h = (h == kHalves - 1);
                int const m64 = m_block * kHalves + h;
                Tensor fq8_tmpK = make_fragment_like(tdKrdK);
                float fq8_g_fold = 0.0f;
                Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<0, 1>(TileShape_MhNK{}));
                Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<0, 1>(TileShape_MhNK{}));
                auto tSrQ_h = fq8_half_A(tSrQ, h, smem_pipe_read.index());
                auto tdPrdO_h = fq8_half_A(tdPrdO, h, smem_pipe_read_do_cur.index());
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ_h, tSrK, tSrS);
                // this half's LSE / dPsum rows (rows kBlockMh * h .. of the stage)
                Tensor sLSEMma_h = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data() + kBlockMh * h), SmemLayoutLSEMmaH{});
                Tensor sdPsumMma_h = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data() + kBlockMh * h), SmemLayoutLSEMmaH{});
                Tensor tLSEsLSE_h = group_modes<0, 2>(thread_mma_SdP.partition_C(sLSEMma_h)(make_coord(_0{}, _, _0{}), _, _0{}, _));
                Tensor tLSEsdPsum_h = group_modes<0, 2>(thread_mma_SdP.partition_C(sdPsumMma_h)(make_coord(_0{}, _, _0{}), _, _0{}, _));
                Tensor tLSErLSE = make_fragment_like(tLSEsLSE_h(_, _0{}));
                cute::copy(tLSEsLSE_h(_, smem_pipe_read.index()), tLSErLSE);
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_dP, tdPrdO_h, tdPrV, tdPrdP);
                warpgroup_wait<1>();
                if (last_h) { pipeline_q.consumer_release(smem_pipe_read); }   // GT: raw Q's only reader (both halves) done
                Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
                mask_h.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m64, n_block);
                #pragma unroll
                for (int mi = 0; mi < size<0>(scores); ++mi) {
                    float const lse_scaled = __fmul_rn(tLSErLSE(mi), float(M_LOG2E));
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ++ni) {
                        scores(mi, ni) = exp2f(fminf(scores(mi, ni) * fq8_s_descale_log2 - lse_scaled + kFq8MoE,
                                                     kFq8MoE + 4.0f));
                    }
                }
                Tensor tLSErdPsum = make_fragment_like(tLSEsdPsum_h(_, _0{}));
                cute::copy(tLSEsdPsum_h(_, smem_pipe_read_do_cur.index()), tLSErdPsum);
                warpgroup_wait<0>();
                if (last_h) { pipeline_do.consumer_release(smem_pipe_read_do_cur); }   // GT: raw dO's only reader done
                Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
#ifdef FA3FP11_DBG
                if (params.ptr_dbg_rows != nullptr) {
                    flash::SumOp<float> fq8_dbg_sum;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(dS); ++mi) {
                        float const fq8_dbg_dsum = tLSErdPsum(mi);
                        float fq8_ra = 0.f, fq8_rb = 0.f;
                        #pragma unroll
                        for (int ni = 0; ni < size<1>(dS); ++ni) {
                            float const t = scores(mi, ni) * (dS(mi, ni) * fq8_dp_descale - fq8_dbg_dsum);
                            fq8_ra += t;
                            fq8_rb += fabsf(t);
                        }
                        fq8_ra = flash::Allreduce<4>::run(fq8_ra, fq8_dbg_sum);
                        fq8_rb = flash::Allreduce<4>::run(fq8_rb, fq8_dbg_sum);
                        if ((thread_idx & 3) == 0) {
                            int const fq8_row = get<0>(tScS_rc_h(mi, _0{}));
                            int64_t const fq8_gi = ((int64_t(fq8_bh) * params.nbn + n_block) * 2
                                + warp_group_idx) * int64_t(seqlen_q) + m64 * kBlockMh + fq8_row;
                            params.ptr_dbg_rows[fq8_gi] = fq8_ra;
                            params.ptr_dbg_rowsabs[fq8_gi] = fq8_rb;
                        }
                    }
                }
#endif
                #pragma unroll
                for (int mi = 0; mi < size<0>(dS); ++mi) {
                    float const dP_sum_cur_f = tLSErdPsum(mi) * fq8_k8s_wg;
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) {
                        dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) * fq8_dp_descale_f - dP_sum_cur_f);
                    }
                }
                float fq8_amax = 0.f;
                #pragma unroll
                for (int i = 0; i < size(tdPrdP); ++i) { fq8_amax = fmaxf(fq8_amax, fabsf(tdPrdP(i))); }
                #pragma unroll
                for (int off = 16; off > 0; off /= 2) {
                    fq8_amax = fmaxf(fq8_amax, __shfl_xor_sync(0xffffffff, fq8_amax, off));
                }
                int const fq8_warp = thread_idx / cutlass::NumThreadsPerWarp;
                if ((thread_idx % cutlass::NumThreadsPerWarp) == 0) { fq8_psired[fq8_warp] = fq8_amax; }
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG3) /*id: free at 2 WGs*/);
                #pragma unroll
                for (int w = 0; w < NumMmaThreads / cutlass::NumThreadsPerWarp; ++w) {
                    fq8_amax = fmaxf(fq8_amax, fq8_psired[w]);
                }
                bool const fq8_tile_zero = fq8_amax * (1.0f / 448.0f) < 1e-30f;
                float const fq8_psi_raw = fmaxf(fq8_amax * (1.0f / 448.0f), 1e-30f);
                float const fq8_psi = fmaxf(fq8_psi_raw, fq8_g_max * (256.0f * 1e-8f) / fq8_q8sc);
                float const fq8_psi_inv = fq8_tile_zero ? 0.0f : 1.0f / fq8_psi;
                if (thread_idx == 0) {
                    params.ptr_psi[(int64_t(fq8_bh) * params.nbm + m64) * params.nbn + n_block]
                        = fq8_tile_zero ? 0.0f : fq8_psi;
                }
                Tensor rP = make_tensor_like<Element>(tSrS);
                flash::convert_type_out(tSrS, rP);
                Tensor rP_rc = make_tensor(rP.data(), scores.layout());
                if (thread_idx == 0) { cute::tma_store_wait<0>(); }
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
                #pragma unroll
                for (int mi = 0; mi < size<0>(rP_rc); ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(rP_rc); ++ni) {
                        auto crd = tScS_rc_h(mi, ni);
                        sPt_pi(get<1>(crd), fq8_pi_m(get<0>(crd)), _0{}) = rP_rc(mi, ni);
                    }
                }
                #pragma unroll
                for (int i = 0; i < size(tdPrdP); ++i) { tdPrdP(i) *= fq8_psi_inv; }
                Tensor rdS = make_tensor_like<Element>(tdPrdP);
                flash::convert_type_out(tdPrdP, rdS);
                Tensor rdS_rc = make_tensor(rdS.data(), scores.layout());
                #pragma unroll
                for (int mi = 0; mi < size<0>(rdS_rc); ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(rdS_rc); ni += 2) {
                        auto crd = tScS_rc_h(mi, ni);
                        uint16_t const fq8_pack =
                            uint16_t(rdS_rc(mi, ni).storage)
                            | (uint16_t(rdS_rc(mi, ni + 1).storage) << 8);
                        sdS16_pi(get<0>(crd), get<1>(crd) / 2, _0{}) = fq8_pack;
                        auto crd1 = tScS_rc_h(mi, ni + 1);
                        sdSt_pi(get<1>(crd), fq8_pi_m(get<0>(crd)), _0{}) = rdS_rc(mi, ni);
                        sdSt_pi(get<1>(crd1), fq8_pi_m(get<0>(crd1)), _0{}) = rdS_rc(mi, ni + 1);
                    }
                }
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
                if (thread_idx == 0) {
                    Tensor mdS8 = params.tma_store_dS8.get_tma_tensor(
                        make_shape(Int<kBlockMh>{}, Int<kBlockN>{}, params.nbn, params.nbm,
                                   params.num_batch * get<2>(params.shape_Q)));
                    Tensor gdS8 = mdS8(_, _, n_block, m64, fq8_bh);
                    auto blk_tma_dS8 = params.tma_store_dS8.get_slice(_0{});
                    Tensor tdS8s = blk_tma_dS8.partition_S(sdS(_, _, _0{}));
                    Tensor tdS8g = blk_tma_dS8.partition_D(gdS8);
                    cute::copy(params.tma_store_dS8, tdS8s, tdS8g);
                    cute::tma_store_arrive();
                }
                // the dV window rescale exactly where (and as often as) P1-A applies it: once per 64-row tile, right before
                // the tile's dV gemm (the second half's factor is f/f == 1.0: an exact no-op, kept for bitwise identity)
                {
                    float const fq8_f_cur = kFq8MoInv * fq8_do8sc;
                    float const fq8_rf = fq8_clamp_ratio(fq8_f_prev / fq8_f_cur);
                    #pragma unroll
                    for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) *= fq8_rf; }
                    fq8_f_prev = fq8_f_cur;
                }
                fq8_g_fold = 0.0f;
                if (!fq8_tile_zero) {
                    float const fq8_g_cur = fq8_psi * kFq8MoInv * fq8_q8sc;
                    fq8_g_fold = fq8_g_cur;
                    fq8_g_max = fmaxf(fq8_g_max, fq8_g_cur);
                }
                Tensor tdVrPt = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
                Tensor tdVrPt_cur = tdVrPt(_, _, _, _0{});
                Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
                Tensor tdKrdS_cur = tdKrdS(_, _, _, _0{});
                if (h == 0) { consumer_wait(pipeline_dot, smem_pipe_read_dot); }
                // dV window gemm: wgmma A = the dOt twin (k-atoms {2h, 2h+1} of the stage), B = P~^T (its two k-atoms)
                {
                    auto tA = tdVrdO(_, _, _, smem_pipe_read_dot.index());
                    warpgroup_fence_operand(tdVrdV);
                    warpgroup_arrive();
                    #pragma unroll
                    for (int kk = 0; kk < 2; ++kk) {
                        cute::gemm(tiled_mma_dKV, tA(_, _, 2 * h + kk), tdVrPt_cur(_, _, kk), tdVrdV);
                        tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::One;
                    }
                    warpgroup_commit_batch();
                    warpgroup_fence_operand(tdVrdV);
                }
                if (h == 0) { consumer_wait(pipeline_qt, smem_pipe_read_qt); }
                // dK temp gemm (zero-init): wgmma A = the Qt twin (k-atoms {2h, 2h+1}), B = dS'^T
                {
                    auto tA = tdKrQ(_, _, _, smem_pipe_read_qt.index());
                    warpgroup_fence_operand(fq8_tmpK);
                    warpgroup_arrive();
                    tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;
                    #pragma unroll
                    for (int kk = 0; kk < 2; ++kk) {
                        cute::gemm(tiled_mma_dKV, tA(_, _, 2 * h + kk), tdKrdS_cur(_, _, kk), fq8_tmpK);
                        tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::One;
                    }
                    warpgroup_commit_batch();
                    warpgroup_fence_operand(fq8_tmpK);
                }
                warpgroup_wait<1>();   // retires dV; the dK temp stays in flight
                if (last_h) { pipeline_dot.consumer_release(smem_pipe_read_dot); }
                warpgroup_wait<0>();
                warpgroup_fence_operand(fq8_tmpK);
                if (last_h) { pipeline_qt.consumer_release(smem_pipe_read_qt); }
                #pragma unroll
                for (int i = 0; i < size(tdKrdK); ++i) {
                    tdKrdK(i) = fmaf(fq8_tmpK(i), fq8_g_fold, tdKrdK(i));
                }
            }
            ++smem_pipe_read_dot;
            ++smem_pipe_read_qt;
            ++smem_pipe_read;
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
            if ((m_block & (kFq8PromoteTilesStage - 1)) == kFq8PromoteTilesStage - 1
                    || m_block == m_block_max - 1) {
                #pragma unroll
                for (int i = 0; i < size(tdVrdV); ++i) {
                    float* const slot = fq8_dv_base[i & 1][(i >> 1) & 1] + fq8_dv_off(i);
                    *slot = fmaf(tdVrdV(i), fq8_f_prev, *slot);
                    tdVrdV(i) = 0.0f;
                }
            }
        };

        // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
        // this helps quite a bit to not have to do causal masking for most of the iterations.
        if constexpr ((Is_causal || Is_local) && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); };
            static constexpr int kBlockM = get<0>(TileShape_MNK{});
            int const m_block_masking_max = ((n_block + 1) * kBlockN - 1 + seqlen_q - seqlen_k - params.window_size_right) / kBlockM + 1;
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < std::min(m_block_max, m_block_masking_max); ++m_block) {
                if constexpr (kHalves == 1) { bwd_step(m_block, mask_fn); } else { bwd_step2(m_block, mask_fn); }
            }
        }

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const m_block_max_before_local_mask = !Is_local || !SeparateMaskingIterations
            ? m_block_max
            : std::min(m_block_max, (n_block * kBlockN + seqlen_q - seqlen_k + params.window_size_left) / kBlockM);

        auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, Is_causal && !SeparateMaskingIterations, Is_local && !SeparateMaskingIterations>(tSrS, m_block, n_block); };
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max_before_local_mask; ++m_block) {
            if constexpr (kHalves == 1) { bwd_step(m_block, mask_fn); } else { bwd_step2(m_block, mask_fn); }
        }

        if constexpr (Is_local && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, false /*Causal_mask*/, Is_local>(tSrS, m_block, n_block); };
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < m_block_max; ++m_block) {
                if constexpr (kHalves == 1) { bwd_step(m_block, mask_fn); } else { bwd_step2(m_block, mask_fn); }
            }
        }

        // ## V10 (split bwd): drain the last m-block's async dS8 tile store
        // (kernel exit must not race the bulk store; non-issuing threads
        // return immediately).
        cute::tma_store_wait<0>();

        // ## FP11-M4: the persistent accumulators already hold TRUE-unit
        // gradients (dK: each tile's g basis folded at its per-tile fold;
        // dV: each window's f basis folded at its boundary promote, and the
        // final bwd_step's tail boundary promoted the last window), so v10's
        // exit fold of f_last / g_last no longer exists: tdKrdK only takes
        // the k8s unfold, and tdVrdV (a dead window fragment here) is
        // overwritten with the smem read-back.  The reference bakes sm into
        // the Q codes, so dK carries no extra softmax_scale (the api passes
        // softmax_scale=1 for the mainloop; the pass-2 kernel's final_scale
        // supplies dQ's sm factor).
        {
            // ## FQ8-V9 FOLD unfold for dK: with dKV_swapAB each WG's dK
            // accumulator covers ALL kBlockN columns (the WG split is along
            // d), and the folded k8s varies per 32-column block: divide per
            // column.  The n-half is compile-time per fragment index
            // (n = n0 + c + 8k with n0 + c <= 7, so n < 32 <=> k < 4;
            // verified in a fragment simulator), matching v10's wg-slice map exactly
            // where v10 evaluated it.
            float const fq8_ik0 = params.softmax_scale / fq8_k8s_g0;
            float const fq8_ik1 = params.softmax_scale / fq8_k8s_g1;
            #pragma unroll
            for (int i = 0; i < size(tdKrdK); ++i) {
                // ## Delta-Matching-128: n = n0 + c + 8k, half = n < kQuantKeys: k < 4 at d256 (k = (i>>2)&7), k < 8 at hd128 (k = (i>>2)&15).
                bool const fq8_half0 = kHeadDim >= 256 ? (((i >> 2) & 7) < 4) : (((i >> 2) & 15) < 8);
                tdKrdK(i) *= fq8_half0 ? fq8_ik0 : fq8_ik1;
            }
        }
        // ## FP11-M4: dV read-back from the smem persistent accumulator.
        // Per-thread-owned slots (the reading thread wrote them): no barrier;
        // and the epilogue's smem staging window (the first 64 KB of the
        // TensorStorage union) does not overlap this member (see the overlay
        // comment at the mainloop entry).
        #pragma unroll
        for (int i = 0; i < size(tdVrdV); ++i) {
            tdVrdV(i) = fq8_dv_base[i & 1][(i >> 1) & 1][fq8_dv_off(i)];
        }

        if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
        ++work_idx;
        return true;
    }

    // ## FP11-M5 PASS 0 (in-kernel matched delta, the design notes): the pass-1-lite
    // consumer.  Everything that determines the realized S~ / dP~ / P~ bits is
    // line-for-line the mma() code above on the SAME smem staging (identical
    // load(), identical SmemLayouts, identical TiledMmaSdP 64x32-per-WG atom
    // split, identical zero-init 8-atom wgmma chains issued in the identical
    // S-then-dP order, identical descale expressions in the identical multiply
    // order, identical LSE copy, identical mask placement, identical exp2 line
    // with the F1 clamp = D4a): wgmma is deterministic on operand bytes, so
    // the per-tile fp32 P~ and raw dP~ fragments here equal pass 1's
    // bit-for-bit (verified by the FA3FP11_DBG checksum mode).  Only the
    // CONSUMPTION differs: instead of dS'/psi/dV/dK, the tile contributes
    //
    //   partial_i^(nb) = 2^-8 * sum_{ni in tile} P~(i,ni) * (dP~(i,ni) * fq8_dp_descale)
    //
    // with fq8_dp_descale = do8sc * v8s_wg (NO k8s: pass 1 multiplies delta by
    // k8s_wg at the dS line, so delta itself is in true dP units, matching
    // both fused_doq2's stale rowsum(dO*O) and the Triton reference delta).  The 2^-8 undoes
    // the MO offset riding in P~ = 2^8*P (exact power-of-two multiply).
    //
    // Summation order (free by the derivation notes item 1; fixed + documented for
    // determinism): per-thread over its 8 fragment columns in ascending
    // rowcol-layout ni (physical columns 0,1,8,9,16,17,24,25 of the WG half,
    // fmaf chain); quad reduction across the 4 lanes sharing the row via
    // flash::Allreduce<4> (shfl_xor offsets 2 then 1: the exact softmax
    // SumOp discipline); each warpgroup's quad leaders store their 32-key
    // half-sum, scaled by the exact 2^-8, straight to the [BH, nbn, 2, N]
    // workspace at [bh][nb][warpgroup][row]; the pass-0.5 reduce
    // (fp11_delta.h) folds the halves and tiles in fp64 in fixed order
    // (n-tile ascending, warpgroup 0 then 1).  Exactly one writer per
    // (row, n-tile, half): deterministic; no atomics, no cross-WG barrier
    // (the iteration record in the loop comment below documents why the
    // in-CTA combine variant was dropped).
    //
    // Causal coverage: this CTA writes tiles m_block_min..m_block_max-1 =
    // exactly the tiles the fixed-order reduce (fp11_delta.h) reads
    // (j <= mb), so the workspace needs no zero-init.
    template <typename SharedStorage>
    CUTLASS_DEVICE bool
    mma_pass0(Params const& params,
              MainloopPipeline pipeline_q,
              MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_read,
              PipelineState_dO& smem_pipe_read_do,
              int thread_idx,
              int &work_idx,
              cute::tuple<int32_t, int32_t, int32_t> block_coord,
              SharedStorage& shared_storage
              ) {
        static_assert(DeltaPass0, "mma_pass0 is the DeltaPass0 consumer");

        int n_block = get<0>(block_coord);
        int bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb, params.window_size_left,
            params.window_size_right, 0 /*sink_token_length*/);
        // It's possible to have m_block_max <= m_block_min. Exit early
        if constexpr (Is_causal || Is_local || Varlen) {
            if (m_block_max <= m_block_min) { return false; }
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sLSEMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});

        constexpr int MmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaSdP tiled_mma_SdP;
        using TiledMmadP = std::conditional_t<!Mma_dP_is_RS, TiledMmaSdP, TiledMmadPRS>;
        TiledMmadP tiled_mma_dP;
        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dP = tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);

        Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
        Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
        Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
        Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_dP, sV);

        // Fragment identity coordinates (row, col) of the 64x64 tile: the row
        // map for the partial-delta reduction (same object mma() scatters by).
        Tensor cS_fq8 = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
        Tensor tScS_fq8 = thread_mma_SdP.partition_C(cS_fq8);
        Tensor tScS_rc = make_tensor(tScS_fq8.data(),
            flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tScS_fq8.layout()));

        Tensor tLSEsLSE = cute::conditional_return<!SdP_swapAB>(
            group_modes<0, 2>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)),  // (2, MMA_M, PIPE)
            group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _, _)));  // (2, V, MMA_N, PIPE)
        static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tLSEsLSE))::value, 8);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int bidh = get<1>(block_coord);
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;

        flash::Mask<kBlockM, kBlockN, false /*PackGQA*/, TiledMmaSdP, SdP_swapAB> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod
        );

        int m_block = m_block_min;

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.pipelines.barrier_KV.wait(work_idx % 2); }

        if constexpr (Mma_dP_is_RS) {
            using SmemCopyAtomV = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
            auto smem_tiled_copy_V = make_tiled_copy_A(SmemCopyAtomV{}, tiled_mma_dP);
            auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
            Tensor tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
            Tensor tdPsV_copy_view = smem_thr_copy_V.partition_S(cute::as_position_independent_swizzle_tensor(sV));
            cute::copy(smem_tiled_copy_V, tdPsV_copy_view, tdPrV_copy_view);
        }

        // ## FQ8: scale bookkeeping, the exact mma() subset (same expressions,
        // same multiply order; the per-tile scales are read at that tile's
        // issue point, which the software pipeline below places one tile
        // ahead: same values, same fp32 arithmetic).
        int const fq8_bh = get<2>(block_coord) * get<2>(params.shape_Q) + bidh;
        // ## FP11-M16 (native GQA, READ side): K/V scales per KV head (see mma()).
        int const fq8_bh_kv = get<2>(block_coord) * get<2>(params.shape_K)
            + params.qhead_per_khead_divmod.divide(bidh);
        float const fq8_k8s_wg = params.ptr_k8s[fq8_bh_kv * params.nblk_k + 2 * n_block + warp_group_idx];
        float const fq8_v8s_wg = params.ptr_v8s[fq8_bh_kv * params.nblk_k + 2 * n_block + warp_group_idx];
        float const* fq8_q8s_base = params.ptr_q8s + fq8_bh * params.nblk_q;
        float const* fq8_do8s_base = params.ptr_do8s + fq8_bh * params.nblk_q;
        static constexpr float kFq8MoE = 8.0f;          // log2(2^8)
        static constexpr float kFq8MoInv = 1.0f / 256.0f;

        // Per-CTA delta-partial base: this CTA's WARPGROUP-HALF column of the
        // [BH, nbn, 2, N] workspace (fp11_delta.h header documents the
        // layout and the fixed-order reduce that consumes it).
        float* const fq8_part_wg = params.ptr_dltpart
            + ((int64_t(fq8_bh) * params.nbn + n_block) * 2 + warp_group_idx)
                * int64_t(seqlen_q);

        // ## FP11-M5 consumer loop (the speed-iteration record, all variants
        // bit-identical on delta -- only scheduling/staging moved):
        //   iter1: straight per-tile loop, in-CTA cross-WG smem exchange +
        //          one 256-thread named barrier per tile, Stages_dO=1:
        //          0.94 ms at B2H8N8192.  The tensor core idles through
        //          every tile's reduce/exchange/barrier tail and the 1-stage
        //          dO refill.
        //   iter2/3: cross-tile software pipeline with double-buffered
        //          loop-carried accumulator fragments: ptxas C7515
        //          ("wgmma serialized: non-wgmma instructions defining
        //          accumulator registers in the pipeline window") -- the
        //          exp2/mask CUDA-core writes into loop-carried wgmma
        //          accumulators while other chains are in flight serialize
        //          the whole async pipe: 1.25 ms.  REVERTED.
        //   iter4 (shipped): the iter1 per-tile structure (per-iteration
        //          fragment locals: no loop-carried accumulator hazard, no
        //          C7515) with the two tail costs removed instead:
        //          (a) Stages_dO=2 (fp11_delta.h; flagged deviation from
        //              pass 1's 1-stage dO: pass 0's per-tile work is ~4x
        //              smaller, so the release->refill latency was exposed;
        //              staging depth changes no realized arithmetic), and
        //          (b) NO in-CTA cross-WG combine: each warpgroup's quad
        //              leaders store their 32-key half-sums (x 2^-8)
        //              STRAIGHT to gmem at [bh][nb][wg][row], and the
        //              pass-0.5 reduce folds the 2 halves in its fixed
        //              order.  Removes the per-tile named barrier + smem
        //              exchange entirely (flagged deviation from the
        //              [BH, nbn, N] workspace shape in the design notes: the
        //              workspace is [BH, nbn, 2, N] = ~128 MB at B2H8
        //              N8192, still transient/allocator-recycled).
        // Summation order (fixed + documented; free by the derivation notes item 1):
        // per-thread fmaf chain over its 8 fragment columns in ascending
        // rowcol-layout ni; quad Allreduce<4> (shfl_xor 2 then 1, the
        // softmax SumOp discipline) across the 4 lanes sharing the row;
        // per-(tile, wg-half) partial scaled by 2^-8 (exact); cross-tile /
        // cross-half combination in the reduce kernel: fp64, n-tile
        // ascending, wg 0 then 1.  Exactly one writer per (row, tile, half):
        // deterministic, no atomics, no barriers.
        static_assert(!SeparateMaskingIterations && !ShuffleLSE && !ShuffledPsum,
                      "mma_pass0's loop assumes the d256 single-mask-loop, unshuffled-stats configuration");
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        auto mask_fn = [&](auto& tSrS_x, int mb) { mask.template apply<true /*Seqlenk_mask*/, Is_causal && !SeparateMaskingIterations, Is_local && !SeparateMaskingIterations>(tSrS_x, mb, n_block); };

        float fq8_q8sc_next = fq8_q8s_base[m_block_min];
        float fq8_do8sc_next = fq8_do8s_base[m_block_min];

        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max; ++m_block) {
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            consumer_wait(pipeline_q, smem_pipe_read);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
            Tensor tLSErLSE = make_fragment_like(tLSEsLSE(_, _0{}));
            cute::copy(tLSEsLSE(_, smem_pipe_read.index()), tLSErLSE);
            // ## FQ8-V9-style software-pipelined scale loads (same values,
            // same fp32 expressions as mma()).
            float const fq8_q8sc = fq8_q8sc_next;
            float const fq8_do8sc = fq8_do8sc_next;
            if (m_block + 1 < m_block_max) {
                fq8_q8sc_next = fq8_q8s_base[m_block + 1];
                fq8_do8sc_next = fq8_do8s_base[m_block + 1];
            }
            float const fq8_s_descale_log2 = fq8_q8sc * fq8_k8s_wg * float(M_LOG2E);
            float const fq8_dp_descale = fq8_do8sc * fq8_v8s_wg;
            Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
            consumer_wait(pipeline_do, smem_pipe_read_do_cur);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_dP, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);
            warpgroup_wait<1>();   // retires this tile's S; its dP still flies

            // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
            mask_fn(tSrS, m_block);
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
                float const lse_scaled = __fmul_rn(tLSErLSE(mi), float(M_LOG2E));
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
                    // ## The mma() exp2 line VERBATIM, F1 clamp included (D4a).
                    scores(mi, ni) = exp2f(fminf(scores(mi, ni) * fq8_s_descale_log2 - lse_scaled + kFq8MoE,
                                                 kFq8MoE + 4.0f));
                }
            }

            warpgroup_wait<0>();   // retires this tile's dP
            // Hand the raw stages back before the reduction so the producer
            // refills under it.  (mma() releases q at its qt wait and do at
            // its dot wait; pass 0 has no transpose warps, so the releases
            // move to the retire point.  Release order has no numeric
            // effect: the S/dP results are already in registers.)
            pipeline_do.consumer_release(smem_pipe_read_do_cur);
            pipeline_q.consumer_release(smem_pipe_read);
            ++smem_pipe_read;
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }

            Tensor dPv = make_tensor(tdPrdP.data(), scores.layout());
#ifdef FA3FP11_DBG
            // ## FP11-M5-DBG checksum: the SAME two capture points as mma()
            // (P~ = tSrS post-exp2/mask; dP~ = raw tdPrdP), same fold, same
            // per-(bh, m-tile, n-tile) slots.  Equal checksums per causal
            // tile == the pass-0 realization is bit-matching pass 1.
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
                if ((thread_idx % cutlass::NumThreadsPerWarp) == 0) {
                    int64_t const fq8_nt = int64_t(params.num_batch) * get<2>(params.shape_Q)
                        * params.nbm * params.nbn;
                    int64_t const fq8_ti = (int64_t(fq8_bh) * params.nbm + m_block) * params.nbn + n_block;
                    atomicXor(params.ptr_dbg_cks + fq8_ti, fq8_xp);
                    atomicXor(params.ptr_dbg_cks + fq8_nt + fq8_ti, fq8_xd);
                }
            }
#endif
            // ## The delta partial (header comment): per-thread fmaf chain
            // over ascending ni, quad Allreduce<4>, quad-leader stores of
            // the 2^-8-scaled warpgroup half-sums straight to gmem.
            flash::SumOp<float> fq8_sum_op;
            CUTE_STATIC_ASSERT_V(size<0>(scores) == _2{});   // 2 rows per thread
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
                float a = 0.f;
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
                    a = fmaf(scores(mi, ni), dPv(mi, ni) * fq8_dp_descale, a);
                }
                a = flash::Allreduce<4>::run(a, fq8_sum_op);
                if ((thread_idx & 3) == 0) {
                    int const fq8_row = get<0>(tScS_rc(mi, _0{}));
                    fq8_part_wg[m_block * kBlockM + fq8_row] = a * kFq8MoInv;
                }
            }
        }

        if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
        ++work_idx;
        return true;
    }

};

} // namespace flash
