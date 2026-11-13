// ## V10 (split bwd) PASS 2: dQ = sm/256 * sum_nb psi[mb,nb] * (dS'8 @ Kt8)
// as a block-scaled all-e4m3 wgmma GEMM.
//
// One CTA per (pair of m-blocks, bh) since P2D: two math warpgroups (one per
// m-block, 128 threads each, the single-warpgroup arithmetic below verbatim)
// share each K^T tile of a 3-stage TMA pipeline fed by a producer warp (one
// transaction barrier per stage covering the K^T tile and both dS' tiles, one
// empty barrier per stage with one arrival per warpgroup).
//
// ## FP11-M3 (interval promotion, the design notes): the fp32 register fragment acc
// is no longer the row-long online chain (v10/M2: ONE psi-ratio-rescaled
// wgmma-in-place chain across up to 128 tiles = a ~14-bit tensor-core re-round
// chain AND a multiplicative rescale chain, both O(row length)).  acc is now a
// REGISTER WINDOW covering kPromoteTiles = 8 tiles (512 keys): within a
// window the v10/M2 arithmetic is kept exactly (psi-ratio rescale between
// taken tiles, in-place wgmma, the M2 F2/F2b skip predicates); at each window
// boundary the window is PROMOTED into a persistent fp32 dQ accumulator in
// SHARED MEMORY:  smem_dq(m,d) += acc(m,d) * psi_w  (psi_w = the window's
// last taken psi, so smem_dq holds TRUE unnormalized units), then acc is
// cleared and the per-window psi tracking reset.  The epilogue reads smem_dq
// and multiplies by final_scale = sm/256 ONLY (psi is already folded per
// window).  Both defect chains are thereby cut to <= kPromoteTiles length;
// cross-window combination is plain fp32 FADD in true units (the derivation notes: the
// per-interval TC error saturates at ~1.22e-4 while the fp32 stage never
// binds).
//
// ## FP11-M3 invariant (why the M2 F2b dominance rule SURVIVES this
// redesign): the F2b skip bounds every taken tile at psi_cur >= 1e-8 *
// psi_max(row so far), so within a window every applied ratio is
// psi_w/psi_cur <= 1e8 EXACTLY (the up-clamp cannot bind in excess, ratios
// telescope to psi_first_taken/psi_last_taken <= 1e8) and |acc| stays bounded
// by ~kPromoteTiles * 448*448*64 * 1e8 ~ 1e16 << fp32 max; ACROSS windows
// there is no rescale at all: the promote is a pure fp32 FADD in true
// (psi-folded) units, which cannot compound.  The compounding-overflow
// mechanism H-RC is therefore structurally impossible
// here, and the skip predicates themselves are byte-identical to M2's (same
// row-global running psi_max), so the SKIP SET is unchanged from M2.
//
// ## Window-boundary / skip interaction (the M3 design decision): a skipped
// tile is TRANSPARENT: it neither joins nor ends a window.  Boundaries are
// fixed by GEOMETRY alone: after every kPromoteTiles-th tile index and at the
// causal tail (nb == nb_end-1), independent of the psi field.  Rationale:
// (1) data-independent promotion boundaries keep pass 2 in ONE realization
// class (the design's consistency law; the design's fixed 128-key boundaries
// are the same principle), where skip-terminated windows would make the
// realized scheme a function of the input's psi field; (2) on healthy inputs
// the predicates never fire (M2 G1: bitwise-invisible), so the two options
// coincide there and transparency is the simpler realization; (3) safety
// does not need boundary-ending: the ratio bound above holds per window
// regardless of which tiles inside it were skipped (skipped tiles contribute
// exact zeros and leave psi_w untouched); (4) cost: exactly
// ceil(nb_end/kPromoteTiles) promotes per row, where skip-ended windows
// could promote up to once per tile on cliff inputs (56% sentinel tiles
// measured on a stress input).  A window whose every tile was skipped has
// psi_w == 0 and acc == 0 and promotes nothing.
//
// ## smem_dq addressing (no atomics, no bank conflicts BY CONSTRUCTION):
// each of the 128 threads owns exactly the 128 (m,d) cells its wgmma C
// fragment holds (the same partition_C identity coordinates the epilogue
// uses; the fragment tiles 64x256 exactly once, verified), so zero-init,
// promote (LDS+FFMA+STS) and the epilogue read are all per-thread-owned
// plain accesses: no read-modify-write races, no atomics, and determinism
// is preserved (single CTA per output row-block, fixed window order,
// per-thread-owned slots).  Layout: row-major with row pitch kDqPitch = 264
// floats (= kBD + 8).  The fragment's access unit is a column pair
// (d, d+1), i.e. 8-byte float2, and at any unrolled step the 16 lanes of a
// half-warp phase touch rows 16w+8rh+{0..3|4..7} x column pairs C0+{0,2,4,6}.
// With the pitch == 8 (mod 32) the float2 start banks are
// (8*a + C0 + 2*c) mod 32 = 16 DISTINCT even residues covering all 32 banks
// per phase: conflict-free under the 64-bit half-warp rule (simulated
// exhaustively over all warps/row-halves/n8/phases; pitch 256 would be
// 4-way conflicted).  Padding columns 256..263 are never touched.
//
// A operand: dS'8 tiles [bh, nbm, nbn, 64, 64] (tile-contiguous, written by
// pass 1's TMA tile-store; k8s and 1/psi are already folded into the codes).
// B operand: the Kt twin (b, h, d, s) = [bh, 256, s], s-contiguous, which is
// K-major for the wgmma B (n=d, k=s) -- the same twin pass 1 loads.
// Causal masking is free: pass 1 wrote zeros above the diagonal in-tile, and
// tiles with nb > mb are never touched.
#pragma once

#include <cstdio>
#include <cstdlib>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

#include "utils.h"

namespace fq8p2 {

using namespace cute;
using Element = cutlass::float_e4m3_t;

constexpr int kBM = 64;      // dQ rows per CTA (= pass-1 kBlockM)
constexpr int kBD = 256;     // head dim
constexpr int kBK = 64;      // contraction depth per tile (= pass-1 kBlockN)
// ## FP11-M3: TMA pipeline depth 2 (v10 shipped 3).  The 67584-B smem_dq
// makes total smem the occupancy limiter: at kStages=3 the CTA needs
// 130048 B -> 1 CTA/SM (v10/M2 ran 2 CTAs/SM: 62464 B, R214), and the
// measured pass-2 cost was 0.674 ms vs 0.416 (occupancy-dominated: halving
// the promote traffic via kPromoteTiles=16 recovered only 0.04 ms).  At
// kStages=2 the CTA is 109568 B -> 2 CTAs/SM again (regs R158 allow 3), and
// the measured cost is 0.462 ms.  Pipeline depth changes NO arithmetic: the
// barrier phase/slot math below is depth-generic.
// ## P2D: 3 stages of (2 dS' tiles + 1 shared K^T tile) + 2 x 67584-B smem_dq = ~209 KB, 1 CTA (2 math warpgroups)
// per SM: the same warps per SM as the v10/M3 2-CTA point, with half the K^T bytes.
constexpr int kStages = 3;
// ## P2W2: kWG row blocks (one math warpgroup each) per CTA share every K^T tile: the K^T bytes pass 2 streams from L2
// (16 KB per 64 x 256 x 64 tile, the dominant traffic) drop by kWG.  Each warpgroup runs the single-warpgroup tile
// sequence of its own row block verbatim (same psi, skips, windows, promotes, epilogue): bitwise by construction.
constexpr int kWG = 2;
// ## P2D: plus one producer warp (TMA issue, decoupled from the math warpgroups through per-stage empty barriers),
// so the two warpgroups drift independently by up to kStages - 1 tiles instead of running in lockstep.
constexpr int kThreads = 128 * kWG + 32;
// ## P2E: L2 prefetch distance (tiles) of the dS' tiles: each dS' byte is read once, from HBM, and the 3-stage smem
// pipeline covers ~2 tiles of compute, less than an HBM round trip; the producer prefetches tile nb + kPfTiles into L2
// (cp.async.bulk.prefetch.tensor) when it issues tile nb.  Cache hint only: no arithmetic change.
constexpr int kPfTiles = 8;

// ## FP11-M3: promotion window length in tiles (C).  PINNED at 8 (512 keys):
// ONE realization class per the design notes (no build knob).  the derivation notes: the
// per-interval TC error e(C) saturates for C >= 4, so C = 8 buys half the
// promote traffic of C = 4 at identical accuracy.
constexpr int kPromoteTiles = 8;
// Power of two: the boundary test is a mask ((nb & (kPromoteTiles-1)) ==
// kPromoteTiles-1); and sane against the pipeline depth: at least one full
// kStages rotation of TMA stages fits inside every window, so the promote
// serialization amortizes over >= one pipeline period.
static_assert(kPromoteTiles > 0 && (kPromoteTiles & (kPromoteTiles - 1)) == 0,
              "delta_matching pass 2: kPromoteTiles must be a power of two (mask boundary test)");
static_assert(kPromoteTiles >= kStages,
              "delta_matching pass 2: promotion window shorter than the TMA pipeline depth");

// smem dQ accumulator row pitch (floats).  kBD + 8 == 8 (mod 32): the unique
// minimal padding making the fragment-owned float2 accesses bank-conflict-free
// (see the header comment); 64 * 264 * 4 B = 67584 B.
constexpr int kDqPitch = kBD + 8;

using TileShapeMNK = Shape<Int<kBM>, Int<kBD>, Int<kBK>>;
using TiledMmaP2 = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, float, TileShapeMNK,
                               GMMA::Major::K, GMMA::Major::K>(),
    Layout<Shape<_1, _1, _1>>{}));

// ## P2B-OVL: the 64 x 256 tile is issued as two 64 x 128 column halves (element-exact: wgmma N-tiling does not
// change any element's k reduction; verified bitwise by the n128 variant), so the rescale / promote of half 1
// overlaps half 0's wgmma.
constexpr int kNH = 2;
constexpr int kBDH = kBD / kNH;
using TileShapeH = Shape<Int<kBM>, Int<kBDH>, Int<kBK>>;
using TiledMmaH = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, float, TileShapeH,
                               GMMA::Major::K, GMMA::Major::K>(),
    Layout<Shape<_1, _1, _1>>{}));

using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GMMA::Major::K, Element, Int<kBM>, Int<kBK>>());
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{}, make_shape(Int<kBM>{}, Int<kBK>{}, Int<kStages>{})));
using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GMMA::Major::K, Element, Int<kBD>, Int<kBK>>());
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{}, make_shape(Int<kBD>{}, Int<kBK>{}, Int<kStages>{})));

struct SharedStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutA> * kWG, 1024> smem_a;   // [wg][stage] (cosize is 1024-B multiple)
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutB>, 1024> smem_b;
    // ## FP11-M3: persistent fp32 dQ accumulator, TRUE unnormalized units
    // (psi folded at promote time), 64 x 264 x 4 = 67584 B per warpgroup.
    cute::array_aligned<float, kWG * kBM * kDqPitch, 1024> smem_dq;
    alignas(16) cutlass::arch::ClusterTransactionBarrier full_bar[kStages];
    alignas(16) cutlass::arch::ClusterBarrier empty_bar[kStages];   // P2D: one arrival per math warpgroup per tile
};

using ShapeA = cute::Shape<Int<kBM>, Int<kBK>, int32_t, int32_t, int32_t>;   // (m, k, nb, mb, bh)
using StrideA = cute::Stride<Int<kBK>, _1, int64_t, int64_t, int64_t>;
using ShapeB = cute::Shape<Int<kBD>, int32_t, int32_t>;                       // (d, s, bh)
using StrideB = cute::Stride<int64_t, _1, int64_t>;
using TmaA = decltype(make_tma_copy(
    cute::SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeA{}, StrideA{}),
    SmemLayoutA{}(_, _, _0{})));
using TmaB = decltype(make_tma_copy(
    cute::SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeB{}, StrideB{}),
    SmemLayoutB{}(_, _, _0{})));

constexpr uint32_t kTxBytesA = kBM * kBK;   // one row block's dS' tile, e4m3 bytes
constexpr uint32_t kTxBytesB = kBD * kBK;   // the shared K^T tile
static_assert(cute::cosize_v<SmemLayoutA> % 1024 == 0, "pass 2: per-warpgroup A stages must keep the swizzle alignment");

struct P2Params {
    TmaA tma_a;
    TmaB tma_b;
    ShapeA shape_a;
    ShapeB shape_b;
    float const* ptr_psi;
    void* dq_ptr;
    int64_t dq_row_stride, dq_head_stride, dq_batch_stride;
    float final_scale;   // softmax_scale / 256 (the 2^8 MO factor)
    int nbm, nbn, h;
    // ## FP11-M16 (native GQA, READ side): the Kt twin is per KV head
    // [b, h_k, d, s]; h_k == h reproduces the M10 index arithmetic exactly.
    int h_k;
    // ## CP ring (batch 1): dq_acc != nullptr: instead of the bf16 dQ, the epilogue adds this block's dQ, rounded to bf16
    // exactly as the bf16 store would, into the fp32 dQ (s_q, h, d) with an IEEE round-to-nearest add (no FTZ): mode 1
    // dq_acc + value, mode 2 0 + value.  Bitwise a zeroed fp32 dQ plus the per-block bf16 dQ added in launch order.
    float* dq_acc;
    int64_t dqa_row_stride, dqa_head_stride;
    int dq_acc_mode;
};

// fp32 a + b, round-to-nearest, denormals kept (the library builds with --use_fast_math, whose FTZ adds would differ
// from torch's fp32 add on denormal results).
__device__ __forceinline__ float add_rn_noftz(float a, float b) {
    float r;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

template <bool Is_causal>
__global__ void __launch_bounds__(kThreads, 1)
fq8_dq_pass2_kernel(CUTE_GRID_CONSTANT P2Params const params) {
    extern __shared__ char smem_raw[];
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_raw);
    int const npair = (params.nbm + kWG - 1) / kWG;
    int const mb0 = (npair - 1 - int(blockIdx.x)) * kWG;   // longest-first (by the CTA's last row block)
    int const wg = __shfl_sync(0xffffffff, int(threadIdx.x) / 128, 0);
    int const mb = mb0 + wg;
    bool const wg_valid = mb < params.nbm;
    int const mb_last = min(mb0 + kWG - 1, params.nbm - 1);
    int const bh = blockIdx.y;
    // nb_end: this warpgroup's tile count; nb_cta: the CTA's (the shared loop trip count, uniform over the CTA).
    int const nb_end = !wg_valid ? 0 : (Is_causal ? (mb + 1) : params.nbn);
    int const nb_cta = Is_causal ? (mb_last + 1) : params.nbn;
    int const tid = threadIdx.x % 128;     // thread index inside the warpgroup (the single-warpgroup kernel's tid)
    int const ctid = threadIdx.x;

    if (ctid == 0) {
        #pragma unroll
        for (int s = 0; s < kStages; ++s) { shared.full_bar[s].init(1); shared.empty_bar[s].init(kWG); }
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    Tensor sB = make_tensor(make_smem_ptr(shared.smem_b.data()), SmemLayoutB{});
    Tensor mA = params.tma_a.get_tma_tensor(params.shape_a);
    Tensor mB = params.tma_b.get_tma_tensor(params.shape_b);
    // ## FP11-M16 (native GQA): B operand indexed by KV head.
    int const bh_kv = (bh / params.h) * params.h_k + (bh % params.h) / (params.h / params.h_k);
    Tensor gB = flat_divide(mB(_, _, bh_kv), Shape<Int<kBD>, Int<kBK>>{})(_, _, _0{}, _);  // (256, 64, nbn)
    auto blkA = params.tma_a.get_slice(_0{});
    auto blkB = params.tma_b.get_slice(_0{});
    Tensor tBgB = group_modes<0, 3>(blkB.partition_S(gB));
    Tensor tBsB = group_modes<0, 3>(blkB.partition_D(sB));


    // One elected thread loads the shared K^T tile and the dS' tile of every row block that needs this tile
    // (causal: the lower row block of a pair stops one tile early).
    auto issue = [&](int tile) {
        {
            int const s = tile % kStages;
            int na = 0;
            #pragma unroll
            for (int w = 0; w < kWG; ++w) {
                int const mbw = mb0 + w;
                na += (mbw < params.nbm && (!Is_causal || tile <= mbw)) ? 1 : 0;
            }
            shared.full_bar[s].arrive_and_expect_tx(kTxBytesB + na * kTxBytesA);
            auto& bar = reinterpret_cast<uint64_t&>(shared.full_bar[s]);
            #pragma unroll
            for (int w = 0; w < kWG; ++w) {
                int const mbw = mb0 + w;
                if (mbw < params.nbm && (!Is_causal || tile <= mbw)) {
                    Tensor sAw = make_tensor(make_smem_ptr(shared.smem_a.data() + w * cute::cosize_v<SmemLayoutA>), SmemLayoutA{});
                    Tensor tAgA = group_modes<0, 3>(blkA.partition_S(mA(_, _, _, mbw, bh)));   // (TMA, nbn)
                    Tensor tAsA = group_modes<0, 3>(blkA.partition_D(sAw));                     // (TMA, stages)
                    copy(params.tma_a.with(bar, 0 /*mcast*/), tAgA(_, tile), tAsA(_, s));
                }
            }
            copy(params.tma_b.with(bar, 0 /*mcast*/), tBgB(_, tile), tBsB(_, s));
        }
    };
    auto prefetch_a = [&](int tile) {
        #pragma unroll
        for (int w = 0; w < kWG; ++w) {
            int const mbw = mb0 + w;
            if (mbw < params.nbm && tile < (Is_causal ? mbw + 1 : params.nbn)) {
                Tensor tAgA = group_modes<0, 3>(blkA.partition_S(mA(_, _, _, mbw, bh)));
                prefetch(params.tma_a, tAgA(_, tile));
            }
        }
    };
    // P2D producer warp (wg == kWG): classified before any math-warpgroup smem tensor / fragment is built; it only
    // touches the barriers, sB and the per-row-block dS' stage tensors of the issue lambda.
    if (wg == kWG) {   // refill slot s once both warpgroups released its previous tile
        if (cute::elect_one_sync()) {
            #pragma unroll 1
            for (int t = kStages; t < kPfTiles && t < nb_cta; ++t) { prefetch_a(t); }
            #pragma unroll 1
            for (int nb = 0; nb < nb_cta; ++nb) {
                if (nb >= kStages) { shared.empty_bar[nb % kStages].wait(((nb / kStages) - 1) & 1); }
                issue(nb);
                if (nb + kPfTiles < nb_cta) { prefetch_a(nb + kPfTiles); }
            }
        }
        return;
    }

    // Math warpgroups only from here (wg < kWG): this warpgroup's dS' stages, its wgmma fragments / descriptors and
    // its smem_dq slots are built after the producer warp's branch, so no tensor ever points past smem_a / smem_dq.
    Tensor sA = make_tensor(make_smem_ptr(shared.smem_a.data() + wg * cute::cosize_v<SmemLayoutA>), SmemLayoutA{});
    TiledMmaH tiled_mma;
    auto wg_mma = tiled_mma.get_thread_slice(tid);
    Tensor tCrA = wg_mma.partition_fragment_A(sA);   // GMMA descriptors (MMA, M, K, stages)
    Tensor tCrB0 = wg_mma.partition_fragment_B(flat_divide(sB, Shape<Int<kBDH>, Int<kBK>>{})(_, _, _0{}, _0{}, _));
    Tensor tCrB1 = wg_mma.partition_fragment_B(flat_divide(sB, Shape<Int<kBDH>, Int<kBK>>{})(_, _, _1{}, _0{}, _));
    Tensor acc0 = partition_fragment_C(tiled_mma, select<0, 1>(TileShapeH{}));   // columns [0, 128)
    Tensor acc1 = partition_fragment_C(tiled_mma, select<0, 1>(TileShapeH{}));   // columns [128, 256)
    clear(acc0);
    clear(acc1);

    // wgmma C fragment coordinates (per half): even/odd i are column-adjacent (d, d+1), so i steps of 2 index
    // 8-byte-aligned float2 slots; half h's column is d + h * kBDH.  The SAME (m, d) map keys smem_dq ownership
    // (promote + zero-init) and the epilogue.
    Tensor cC = cute::make_identity_tensor(make_shape(Int<kBM>{}, Int<kBDH>{}));
    Tensor tCcC = wg_mma.partition_C(cC);
    float* const smem_dq = shared.smem_dq.data() + wg * (kBM * kDqPitch);
    constexpr int kAH = decltype(size(acc0))::value;   // 64 accumulators per thread per half

    // ## FP11-M3: cooperative zero-init of the persistent dQ accumulator, before the pipeline warmup.  Each thread
    // zeroes exactly its own fragment-owned slots: every later access to these slots is by the writing thread itself.
    #pragma unroll
    for (int h = 0; h < kNH; ++h) {
        #pragma unroll
        for (int i = 0; i < kAH; i += 2) {
            int const m = get<0>(tCcC(i));
            int const d = get<1>(tCcC(i)) + h * kBDH;
            *reinterpret_cast<float2*>(&smem_dq[m * kDqPitch + d]) = make_float2(0.0f, 0.0f);
        }
    }
    // FP11-M3 promote of one half: smem_dq(m, d) += acc(m, d) * scale (fp32 FMA, per-thread-owned slots).
    auto promote_h = [&](auto const& a, int coff, float scale) {
        #pragma unroll
        for (int i = 0; i < kAH; i += 2) {
            int const m = get<0>(tCcC(i));
            int const d = get<1>(tCcC(i)) + coff;
            float2* const slot = reinterpret_cast<float2*>(&smem_dq[m * kDqPitch + d]);
            float2 v2 = *slot;
            v2.x = fmaf(a(i), scale, v2.x);
            v2.y = fmaf(a(i + 1), scale, v2.y);
            *slot = v2;
        }
    };

    // ## CP ring, accumulate mode: pull this row block's fp32 dQ (64 rows x 1 KB) into L2 now, so the epilogue's
    // read-modify-write hits L2 instead of exposing an HBM round trip at the CTA's tail (cache hint only).
    if (params.dq_acc != nullptr && params.dq_acc_mode == 1 && wg_valid) {
        float const* const dqa = params.dq_acc + int64_t(bh % params.h) * params.dqa_head_stride + int64_t(mb) * kBM * params.dqa_row_stride;
        #pragma unroll
        for (int l = tid; l < kBM * (kBD / 32); l += 128) {   // 32 floats = one 128-B line
            asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(dqa + int64_t(l / (kBD / 32)) * params.dqa_row_stride + (l % (kBD / 32)) * 32));
        }
    }
    // P2B-OVL: issue one half's two k32 wgmmas and commit, WITHOUT flash::gemm's trailing fence_operand on the
    // in-flight accumulator (that register use made ptxas inject a full wgmma wait before the other half's work).
    auto issue_h = [&](auto& a, auto const& tB, int st, bool zero) {
        warpgroup_fence_operand(a);
        warpgroup_arrive();
        tiled_mma.accumulate_ = zero ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
        #pragma unroll
        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
            cute::gemm(tiled_mma, tCrA(_, _, kb, st), tB(_, _, kb, st), a);
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
    };
    float const* psi_row = params.ptr_psi + (int64_t(bh) * params.nbm + mb) * params.nbn;
    // ## FP11-M3 window state: psi_w = the last TAKEN psi in the CURRENT window (acc's units), 0.0f = window empty.
    // psi_max stays ROW-GLOBAL (the M2 F2b dominance predicate is unchanged: the skip set is bit-identical to M2's).
    // ## P2B-OVL: a closed window's promote is DEFERRED to the next taken tile (or the row end) and interleaved with
    // that tile's two half wgmmas: promote half 0 -> wgmma half 0 (scale_d = 0) -> promote half 1 (overlaps it) ->
    // wgmma half 1.  Nothing touches acc / smem_dq in between (skipped tiles are transparent), so every smem_dq
    // element sees the same FMA sequence in the same order.  Likewise the in-window rescale of half 1 overlaps half
    // 0's wgmma.
    float psi_w = 0.0f;
    float psi_max = 0.0f;
    float psi_prom = 0.0f;   // pending promote scale (0.0f = none pending)
    // psi of tile nb is loaded one iteration early, so its load latency overlaps the barrier wait.
    float psi_pf = nb_end > 0 ? psi_row[0] : 0.0f;
    #pragma unroll 1
    for (int nb = 0; nb < nb_cta; ++nb) {
        int const s = nb % kStages;
        float const psi_cur = nb < nb_end ? psi_pf : 0.0f;   // past this warpgroup's tail: psi 0 = no gemm, no state change
        if (nb + 1 < nb_end) { psi_pf = psi_row[nb + 1]; }
        shared.full_bar[s].wait((nb / kStages) & 1);
        // ## FP11-F2 sentinel / F2b dominance skip (unchanged); a skipped tile is TRANSPARENT to the window.
        if (psi_cur != 0.0f && !(psi_cur < psi_max * 1e-8f)) {
            if (psi_w != 0.0f) {   // not the window's first taken tile: rescale -> gemm, per half
                float ratio = psi_w / psi_cur;
                ratio = fminf(fmaxf(ratio, 1e-8f), 1e8f);
                #pragma unroll
                for (int i = 0; i < kAH; ++i) { acc0(i) *= ratio; }
                issue_h(acc0, tCrB0, s, false);
                warpgroup_wait<1>();   // older groups (acc1's loop-carried def) retired; acc0's stays in flight
                warpgroup_fence_operand(acc1);
                #pragma unroll
                for (int i = 0; i < kAH; ++i) { acc1(i) *= ratio; }
                issue_h(acc1, tCrB1, s, false);
            } else {
                // Window's first taken tile: scale_d = 0 (D = A*B, value-identical to 0 + A*B); the closed window's
                // pending promote reads each half just before that half's wgmma overwrites it.
                bool const pend = psi_prom != 0.0f;
                if (pend) { promote_h(acc0, 0, psi_prom); }
                issue_h(acc0, tCrB0, s, true);
                warpgroup_wait<1>();   // older groups (acc1's loop-carried def) retired; acc0's stays in flight
                warpgroup_fence_operand(acc1);
                if (pend) { promote_h(acc1, kBDH, psi_prom); }
                issue_h(acc1, tCrB1, s, true);
                psi_prom = 0.0f;
            }
            psi_w = psi_cur;
            psi_max = fmaxf(psi_max, psi_cur);
        }
        // Retire this tile's wgmmas at ONE point of every iteration (a no-op on skipped tiles), so ptxas can prove
        // that no wgmma group is outstanding at the loop head (the loop-carried acc defs are retired).
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc0);
        warpgroup_fence_operand(acc1);
        // ## FP11-M10 / P2D release: every thread of this warpgroup has passed wait(nb) (and its gemm retired) before
        // the elected arrive, so the producer's re-arm of slot s can never race a lagging waiter; trip count nb_cta
        // is CTA-uniform.
        cutlass::arch::NamedBarrier::sync(128, 1 + wg);
        if (tid == 0) { shared.empty_bar[s].arrive(); }
        // ## FP11-M3 window boundary: geometry-fixed (every kPromoteTiles-th tile index + the tail), data-independent.
        // The closed window (psi_w != 0) becomes the pending promote; an all-skipped window promotes nothing.
        if (nb < nb_end && ((nb & (kPromoteTiles - 1)) == kPromoteTiles - 1 || nb == nb_end - 1)) {
            if (psi_w != 0.0f) { psi_prom = psi_w; psi_w = 0.0f; }
        }
    }
    if (psi_prom != 0.0f) { promote_h(acc0, 0, psi_prom); promote_h(acc1, kBDH, psi_prom); }

    // Epilogue: smem_dq already holds TRUE unnormalized dQ (psi folded per
    // window); apply final_scale = sm/256 ONLY.  A row whose every tile was
    // a sentinel/skip leaves smem_dq at its zero-init: dq is exactly zero.
    if (!wg_valid) { return; }
    float const fin = params.final_scale;
    int const b_idx = bh / params.h;
    int const h_idx = bh % params.h;
    if (params.dq_acc != nullptr) {   // ## CP ring: fp32 dQ (s_q, h, d) += float(bf16(dQ block)), batch 1
        float* const dqa = params.dq_acc + int64_t(h_idx) * params.dqa_head_stride + int64_t(mb) * kBM * params.dqa_row_stride;
        bool const acc_mode = params.dq_acc_mode == 1;
        // Batches of kSlots float2 slots: all loads of a batch issue before its stores (one round trip per batch
        // instead of a load / store chain per slot); same values, same per-slot arithmetic.
        constexpr int kSlots = 16;
        static_assert(kAH % (2 * kSlots) == 0, "pass 2: dQ slot batches must tile the half fragment");
        #pragma unroll
        for (int h = 0; h < kNH; ++h) {   // P2B-OVL column halves (d offset h * kBDH), as the bf16 epilogue below
            #pragma unroll
            for (int i0 = 0; i0 < kAH; i0 += 2 * kSlots) {
                float2 base[kSlots];
                #pragma unroll
                for (int j = 0; j < kSlots; ++j) {
                    int const m = get<0>(tCcC(i0 + 2 * j));
                    int const d = get<1>(tCcC(i0 + 2 * j)) + h * kBDH;
                    base[j] = acc_mode ? *reinterpret_cast<float2 const*>(dqa + int64_t(m) * params.dqa_row_stride + d)
                                       : make_float2(0.0f, 0.0f);
                }
                #pragma unroll
                for (int j = 0; j < kSlots; ++j) {
                    int const m = get<0>(tCcC(i0 + 2 * j));
                    int const d = get<1>(tCcC(i0 + 2 * j)) + h * kBDH;
                    float2 const v2 = *reinterpret_cast<float2 const*>(&smem_dq[m * kDqPitch + d]);
                    cutlass::bfloat16_t lo(v2.x * fin), hi(v2.y * fin);
                    *reinterpret_cast<float2*>(dqa + int64_t(m) * params.dqa_row_stride + d) =
                        make_float2(add_rn_noftz(base[j].x, float(lo)), add_rn_noftz(base[j].y, float(hi)));
                }
            }
        }
        return;
    }
    auto* dq_base = reinterpret_cast<cutlass::bfloat16_t*>(params.dq_ptr)
        + int64_t(b_idx) * params.dq_batch_stride + int64_t(h_idx) * params.dq_head_stride
        + int64_t(mb) * kBM * params.dq_row_stride;
    // ## P2B-EPI: the bf16 dQ tile is staged through this warpgroup's own smem_dq region and written with coalesced
    // 16-B stores (one full 512-B row per warp instruction).  Same fp32 * fin, same bf16 rounding, same values.
    uint32_t pk[kNH * kAH / 2];
    #pragma unroll
    for (int h = 0; h < kNH; ++h) {
        #pragma unroll
        for (int i = 0; i < kAH; i += 2) {
            int const m = get<0>(tCcC(i));
            int const d = get<1>(tCcC(i)) + h * kBDH;
            float2 const v2 = *reinterpret_cast<float2 const*>(&smem_dq[m * kDqPitch + d]);
            cutlass::bfloat16_t lo(v2.x * fin), hi(v2.y * fin);
            pk[h * (kAH / 2) + i / 2] = (uint32_t(reinterpret_cast<uint16_t const&>(hi)) << 16)
                                      | uint32_t(reinterpret_cast<uint16_t const&>(lo));
        }
    }
    // Staging tile: 64 rows x kStPitchW u32 (kStPitchW == 4 mod 32: conflict-free fragment stores; 528-B rows keep
    // the 16-B read alignment).  It aliases this warpgroup's smem_dq slots: every fp32 read above retires first.
    constexpr int kStPitchW = kBD / 2 + 4;
    static_assert(kBM * kStPitchW <= kBM * kDqPitch, "pass 2: bf16 staging tile must fit the smem_dq region");
    uint32_t* const st = reinterpret_cast<uint32_t*>(smem_dq);
    cutlass::arch::NamedBarrier::sync(128, 1 + wg);
    #pragma unroll
    for (int h = 0; h < kNH; ++h) {
        #pragma unroll
        for (int i = 0; i < kAH; i += 2) {
            int const m = get<0>(tCcC(i));
            int const d = get<1>(tCcC(i)) + h * kBDH;
            st[m * kStPitchW + d / 2] = pk[h * (kAH / 2) + i / 2];
        }
    }
    cutlass::arch::NamedBarrier::sync(128, 1 + wg);
    #pragma unroll
    for (int it = 0; it < kBM / 4; ++it) {
        int const row = it * 4 + tid / 32;
        int const c = tid % 32;   // 16-B chunk of the 512-B row
        uint4 const v = *reinterpret_cast<uint4 const*>(st + row * kStPitchW + c * 4);
        *reinterpret_cast<uint4*>(dq_base + int64_t(row) * params.dq_row_stride + c * 8) = v;
    }
}

}  // namespace fq8p2

// ## FP11 (M2): v11-specific symbol (the parent's run_fq8_dq_pass2 is a weak
// template symbol overlapping the other FA3-derived libraries, and the FP11-F2 sentinel
// makes this function numerically DIVERGENT from v10's: the overlap must be
// impossible by construction, same rationale as run_flash_bwd_fp11_d256).
#if defined(FA3FP11_D256_P2DG) && FA3FP11_D256_P2DG
#include "fq8_dq_pass2_dg.h"   // ## P2DG: the DeepGEMM-class dQ numerics arm (build flag, default off)
#endif

template <bool Is_causal>
void run_fq8_dq_pass2_fp11(Flash_bwd_fp8_params &params, cudaStream_t stream) {
#if defined(FA3FP11_D256_P2DG) && FA3FP11_D256_P2DG
    if (params.dq_acc_ptr == nullptr) {   // the CP ring's in-kernel fp32 dQ accumulation keeps the windowed kernel
        run_fq8_dq_pass2_dg<Is_causal>(params, stream);
        return;
    }
#endif
    using namespace fq8p2;
    int const nbm = cute::ceil_div(params.seqlen_q, kBM);
    int const nbn = cute::ceil_div(params.seqlen_k, kBK);
    int const bh = params.b * params.h;
    ShapeA shape_a = make_shape(Int<kBM>{}, Int<kBK>{}, nbn, nbm, bh);
    StrideA stride_a = make_stride(Int<kBK>{}, _1{}, int64_t(kBM) * kBK,
                                   int64_t(kBM) * kBK * nbn,
                                   int64_t(kBM) * kBK * nbn * nbm);
    Tensor mA = make_tensor(make_gmem_ptr(static_cast<Element const*>(params.ds8_ptr)),
                            shape_a, stride_a);
    TmaA tma_a = make_tma_copy(cute::SM90_TMA_LOAD{}, mA, SmemLayoutA{}(_, _, _0{}));
    ShapeB shape_b = make_shape(Int<kBD>{}, params.seqlen_k, params.b * params.h_k);   // ## FP11-M16: per KV head
    // Explicit Kt strides over the collapsed (b, h_k) mode (the host checks b == 1 or batch stride == h_k * head
    // stride); a contiguous twin gives exactly the former {s_k, 1, 256 s_k}.
    StrideB stride_b = make_stride(params.kt_d_stride, _1{}, params.kt_head_stride);
    Tensor mB = make_tensor(make_gmem_ptr(static_cast<Element const*>(params.kt_ptr)),
                            shape_b, stride_b);
    TmaB tma_b = make_tma_copy(cute::SM90_TMA_LOAD{}, mB, SmemLayoutB{}(_, _, _0{}));
    P2Params p2{tma_a, tma_b, shape_a, shape_b, params.psi_ptr,
                params.dq_ptr, params.dq_row_stride, params.dq_head_stride,
                params.dq_batch_stride,
                params.scale_softmax / 256.0f, nbm, nbn, params.h, params.h_k,
                params.dq_acc_ptr, params.dqa_row_stride, params.dqa_head_stride, params.dq_acc_mode};
    auto* kernel = &fq8p2::fq8_dq_pass2_kernel<Is_causal>;
    constexpr int kSmem = int(sizeof(SharedStorage));
    if (kSmem >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    }
    if (std::getenv("DM_DEBUG")) {
        fprintf(stderr, "[delta_matching pass 2] smem_size=%d grid=(%d,%d) block=%d kPromoteTiles=%d kDqPitch=%d\n",
                kSmem, cute::ceil_div(nbm, kWG), bh, kThreads, kPromoteTiles, kDqPitch);
    }
    dim3 grid(cute::ceil_div(nbm, kWG), bh, 1);
    kernel<<<grid, kThreads, kSmem, stream>>>(p2);
    { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { fprintf(stderr, "[delta_matching pass 2] launch: %s\n", cudaGetErrorString(e_)); exit(1); } }
}
