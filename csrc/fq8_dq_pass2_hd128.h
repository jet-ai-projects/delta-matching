// ## Delta-Matching-128 (M19): PASS 2 (dQ) at head_dim 128.  A copy of fq8_dq_pass2.h (the certified
// d256 pass 2, untouched) with the hd128 geometry: A = the dS' tile pass 1 stores at kBlockN 128 keys
// ([BH, nbm, nbn, 64, 128] tile-contiguous, k8s and 1/psi folded into the codes), B = the Kt twin [bh_kv, 128 d,
// s] (K-major in s), dQ tile 64 rows x 128 d in a 64-register window, kPromoteTiles 4 = 512 keys per window
// (= 16 k32 steps, exactly d256's 8 x 64-key window), row pitch 136 floats (== 8 mod 32: the same conflict-free
// float2 argument as 264).  Smem 2 x (8 + 16) KB + 34.8 KB = ~83 KB -> 2 CTAs/SM.  Numerics: the M3 interval
// promotion + the M2 F2/F2b skip predicates + the M10 unconditional release, line for line.
// ## V10 (split bwd) PASS 2: dQ = sm/256 * sum_nb psi[mb,nb] * (dS'8 @ Kt8)
// as a block-scaled all-e4m3 wgmma GEMM.
//
// One CTA per (m-block, bh); a single math warpgroup (128 threads) consumes a
// 3-stage TMA pipeline (elected-lane issue, one transaction barrier per stage
// covering both operand tiles).
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

namespace fq8p2h128 {

using namespace cute;
using Element = cutlass::float_e4m3_t;

constexpr int kBM = 64;      // dQ rows per CTA (= pass-1 kBlockM)
constexpr int kBD = 128;     // head dim (Delta-Matching-128)
constexpr int kBK = 128;     // contraction depth per tile (= pass-1 kBlockN at hd128)
// ## FP11-M3: TMA pipeline depth 2 (v10 shipped 3).  The 67584-B smem_dq
// makes total smem the occupancy limiter: at kStages=3 the CTA needs
// 130048 B -> 1 CTA/SM (v10/M2 ran 2 CTAs/SM: 62464 B, R214), and the
// measured pass-2 cost was 0.674 ms vs 0.416 (occupancy-dominated: halving
// the promote traffic via kPromoteTiles=16 recovered only 0.04 ms).  At
// kStages=2 the CTA is 109568 B -> 2 CTAs/SM again (regs R158 allow 3), and
// the measured cost is 0.462 ms.  Pipeline depth changes NO arithmetic: the
// barrier phase/slot math below is depth-generic.
// ## M23 candidate B, HALFPSI (a NEW NUMERICS ARM, default off = the certified pass 2 byte for byte): pass 1
// publishes one psi per (tile, warpgroup half) = per 64 keys instead of one per 128-key tile (no cross-warpgroup exchange
// there), so the two 64-key halves of a dS' tile carry different scales.  Pass 2 keeps TWO register accumulators (one per
// half; k-blocks 0-1 and 2-3 of the tile), each with the M2/M3 window logic (F2 sentinel, F2b dominance against the
// row-global psi max, the in-window psi-ratio rescale, the geometry-fixed promote into smem_dq in true units), both
// issued inside one wgmma batch per tile (no extra drain).  The psi field is [BH, nbm, nbn, 2].
#ifndef FA3FP11_HALFPSI
#define FA3FP11_HALFPSI 0
#endif
#ifndef FA3FP11_HALFPSI_OVERLAP   // M23c: the second accumulator's rescale between the two wgmma batches (HALFPSI only)
#define FA3FP11_HALFPSI_OVERLAP 0
#endif
// ## Delta-Matching-128 lever P2-D (M19): at head_dim 128 the smem_dq accumulator is 34816 B (64 rows x 136 floats), not the
// d256 67584 B, and a stage pair is 24576 B, so the CTA is 83968 B at depth 2 and 108544 B at depth 3: BOTH keep 2 CTAs
// per SM (the 116736 B half-SM budget), i.e. depth 3 is free occupancy-wise here where at d256 it cost half the CTAs.
// Depth changes no arithmetic (the barrier phase/slot math is depth-generic), so the outputs must be bitwise.
constexpr int kStages = FA3FP11_HD128_P2ST;

// ## FP11-M3: promotion window length in tiles (C).  PINNED at 8 (512 keys):
// ONE realization class per the design notes (no build knob).  the derivation notes: the
// per-interval TC error e(C) saturates for C >= 4, so C = 8 buys half the
// promote traffic of C = 4 at identical accuracy.
constexpr int kPromoteTiles = 4;   // 4 x 128 keys = 512 keys per window (d256: 8 x 64)
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

using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GMMA::Major::K, Element, Int<kBM>, Int<kBK>>());
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{}, make_shape(Int<kBM>{}, Int<kBK>{}, Int<kStages>{})));
using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GMMA::Major::K, Element, Int<kBD>, Int<kBK>>());
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{}, make_shape(Int<kBD>{}, Int<kBK>{}, Int<kStages>{})));

struct SharedStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutA>, 1024> smem_a;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutB>, 1024> smem_b;
    // ## FP11-M3: persistent fp32 dQ accumulator, TRUE unnormalized units
    // (psi folded at promote time), 64 x 264 x 4 = 67584 B.
    cute::array_aligned<float, kBM * kDqPitch, 1024> smem_dq;
    alignas(16) cutlass::arch::ClusterTransactionBarrier full_bar[kStages];
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

constexpr uint32_t kTxBytes = kBM * kBK + kBD * kBK;   // both tiles, e4m3 bytes

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
};

template <bool Is_causal>
__global__ void __launch_bounds__(128, 1)
fq8_dq_pass2_kernel_hd128(CUTE_GRID_CONSTANT P2Params const params) {
    extern __shared__ char smem_raw[];
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_raw);
    int const mb = params.nbm - 1 - blockIdx.x;   // longest-first
    int const bh = blockIdx.y;
    // ## Delta-Matching-128 FIX (M19): the d256 form `mb + 1` assumes kBM == kBK (64-key tiles).  At hd128 the
    // dS' tiles are 128 keys wide (kBK = 2 kBM): the last tile a causal row block reads is the one holding its last
    // row's diagonal key, nb_end = ((mb + 1) kBM - 1) / kBK + 1 (= mb/2 + 1); pass 1 writes exactly these tiles
    // (m_block_min = 2 nb), and every tile beyond is UNWRITTEN workspace (the nondeterministic NaN dq of an earlier build).
    int const nb_end = Is_causal ? (((mb + 1) * kBM - 1) / kBK + 1) : params.nbn;
    int const tid = threadIdx.x;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < kStages; ++s) { shared.full_bar[s].init(1); }
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    Tensor sA = make_tensor(make_smem_ptr(shared.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared.smem_b.data()), SmemLayoutB{});
    Tensor mA = params.tma_a.get_tma_tensor(params.shape_a);
    Tensor mB = params.tma_b.get_tma_tensor(params.shape_b);
    Tensor gA = mA(_, _, _, mb, bh);                                    // (64, 64, nbn)
    // ## FP11-M16 (native GQA): B operand indexed by KV head.
    int const bh_kv = (bh / params.h) * params.h_k + (bh % params.h) / (params.h / params.h_k);
    Tensor gB = flat_divide(mB(_, _, bh_kv), Shape<Int<kBD>, Int<kBK>>{})(_, _, _0{}, _);  // (256, 64, nbn)
    auto blkA = params.tma_a.get_slice(_0{});
    auto blkB = params.tma_b.get_slice(_0{});
    Tensor tAgA = group_modes<0, 3>(blkA.partition_S(gA));              // (TMA, nbn)
    Tensor tAsA = group_modes<0, 3>(blkA.partition_D(sA));              // (TMA, stages)
    Tensor tBgB = group_modes<0, 3>(blkB.partition_S(gB));
    Tensor tBsB = group_modes<0, 3>(blkB.partition_D(sB));

    TiledMmaP2 tiled_mma;
    auto wg_mma = tiled_mma.get_thread_slice(tid);
    Tensor tCrA = wg_mma.partition_fragment_A(sA);   // GMMA descriptors (MMA, M, K, stages)
    Tensor tCrB = wg_mma.partition_fragment_B(sB);
    Tensor acc = partition_fragment_C(tiled_mma, select<0, 1>(TileShapeMNK{}));
    clear(acc);

    // wgmma C fragment coordinates: even/odd i are column-adjacent (d, d+1),
    // so i steps of 2 index 8-byte-aligned float2 slots.  The SAME (m, d)
    // map keys smem_dq ownership (promote + zero-init) and the epilogue.
    Tensor cC = cute::make_identity_tensor(make_shape(Int<kBM>{}, Int<kBD>{}));
    Tensor tCcC = wg_mma.partition_C(cC);
    float* const smem_dq = shared.smem_dq.data();

    // ## FP11-M3: cooperative zero-init of the persistent dQ accumulator,
    // before the pipeline warmup.  Each thread zeroes exactly its own
    // fragment-owned slots, so no extra __syncthreads is needed anywhere:
    // every later access to these slots is by the writing thread itself.
    #pragma unroll
    for (int i = 0; i < size(acc); i += 2) {
        int const m = get<0>(tCcC(i));
        int const d = get<1>(tCcC(i));
        *reinterpret_cast<float2*>(&smem_dq[m * kDqPitch + d]) = make_float2(0.0f, 0.0f);
    }

    auto issue = [&](int tile) {
        if (tid == 0) {
            int const s = tile % kStages;
            shared.full_bar[s].arrive_and_expect_tx(kTxBytes);
            auto& bar = reinterpret_cast<uint64_t&>(shared.full_bar[s]);
            copy(params.tma_a.with(bar, 0 /*mcast*/), tAgA(_, tile), tAsA(_, s));
            copy(params.tma_b.with(bar, 0 /*mcast*/), tBgB(_, tile), tBsB(_, s));
        }
    };
    #pragma unroll 1
    for (int t = 0; t < (kStages < nb_end ? kStages : nb_end); ++t) { issue(t); }

    float const* psi_row = params.ptr_psi + (int64_t(bh) * params.nbm + mb) * params.nbn;
    // ## FP11-M3 window state: psi_w = the last TAKEN psi in the CURRENT
    // window (acc's units), 0.0f = window empty (no taken tile yet: skip the
    // rescale; acc is zero there anyway).  psi_max stays ROW-GLOBAL: the M2
    // F2b dominance predicate is unchanged, so the skip set is bit-identical
    // to M2's on any psi field.
    float psi_w = 0.0f;
    float psi_max = 0.0f;
    #pragma unroll 1
    for (int nb = 0; nb < nb_end; ++nb) {
        int const s = nb % kStages;
        shared.full_bar[s].wait((nb / kStages) & 1);
        float const psi_cur = psi_row[nb];
        // ## FP11-F2: psi == 0.0f is pass 1's zero-tile sentinel (fully
        // underflowed or fully masked tile; its dS' codes are exact zeros).
        // ## FP11-F2b (dominance skip, M2): also skip any tile whose psi sits
        // > 8 decades below the row's running psi max: its contribution is
        // < fp32 epsilon of the row's dominant tile (every non-sentinel
        // tile's max |code| is 448 by construction), while rescaling toward
        // it would bind the 1e8 up-clamp and compound.  With the skip, every
        // taken in-window ratio is <= psi_max/psi_cur <= 1e8 exactly.
        // Healthy tiles run the v10 sequence bit-for-bit within the window
        // (both predicates are strictly colder than any healthy psi ratio).
        // A skipped tile is TRANSPARENT to the window (see header): it
        // contributes exact zeros and leaves psi_w/psi_max untouched.
        if (psi_cur != 0.0f && !(psi_cur < psi_max * 1e-8f)) {
            if (psi_w != 0.0f) {   // not the window's first taken tile
                float ratio = psi_w / psi_cur;
                ratio = fminf(fmaxf(ratio, 1e-8f), 1e8f);
                #pragma unroll
                for (int i = 0; i < size(acc); ++i) { acc(i) *= ratio; }
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(
                    tiled_mma, tCrA(_, _, _, s), tCrB(_, _, _, s), acc);
            } else {
                // Window's first taken tile: acc's logical value is zero, so
                // start the window's wgmma chain with scale_d = 0 (D = A*B,
                // value-identical to 0 + A*B) instead of rescaling a zero
                // fragment; the promote below then need not re-zero acc.
                // The predicate is warpgroup-uniform (all threads read the
                // same psi_row), so the wgmma stays converged.
                flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(
                    tiled_mma, tCrA(_, _, _, s), tCrB(_, _, _, s), acc);
            }
            psi_w = psi_cur;
            psi_max = fmaxf(psi_max, psi_cur);
        }
        // ## FP11-M10 (HANG FIX): unconditional consumer-release
        // barrier.  This pipeline has NO empty/release mbarrier; v10's safety
        // invariant was IMPLICIT: every iteration ran the warpgroup-collective
        // flash::gemm (wg_wait=0), which cannot complete until all 128 threads
        // issue their part, so thread 0 could never reach issue(nxt) -- the
        // re-arm of slot s = nb % kStages, in THIS iteration -- while another
        // warp still spun in full_bar[s].wait(...) on that slot.  The M2
        // F2/F2b skip predicate above put that collective gemm inside a
        // data-dependent branch: on a SKIPPED tile the iteration contains no
        // warpgroup-collective instruction at all, so thread 0 can
        // arrive_and_expect_tx + TMA-reissue slot s while a descheduled warp
        // (concurrent-load co-residency: NCCL / other CTAs) is still polling
        // that mbarrier's parity.  When the reissued load completes, the
        // parity flips a SECOND time, the lagging warp's try_wait.parity test
        // goes false again, and it is now one full phase rotation behind: at
        // the row tail it waits on a phase nothing will ever complete =
        // permanent spin, wedged CTA, the stream never drains (the observed
        // matched-mode 8-GPU training hang; matched-only because matched
        // delta centers dP so tiny-psi skip tiles are common, while stale psi
        // never trips the predicates and the v10 discipline stays intact).
        // The fix restores the release EXPLICITLY and unconditionally: no
        // thread passes this point until every thread has passed wait(nb), so
        // the re-arm of slot s below can never race a waiter.  Uniform trip
        // count (nb_end is CTA-uniform), zero arithmetic effect (bitwise-
        // invisible on every input); on taken tiles the warps are already
        // converged by wg_wait=0, so the cost is a no-op bar.sync per tile.
        __syncthreads();
        int const nxt = nb + kStages;   // slot s fully consumed (wg_wait=0)
        if (nxt < nb_end) { issue(nxt); }
        // ## FP11-M3 window boundary: geometry-fixed (every kPromoteTiles-th
        // tile index + the tail), data-independent.  Promote the window into
        // smem_dq in TRUE units (fold psi_w NOW: this replaces v10's row-end
        // psi_last fold) and reset the window psi tracking; acc needs no
        // explicit re-zero: the next window's first taken wgmma starts with
        // scale_d = 0.  An all-skipped window (psi_w == 0) promotes nothing.
        // Placed after issue(nxt) so the promote overlaps the next TMA loads.
        if ((nb & (kPromoteTiles - 1)) == kPromoteTiles - 1 || nb == nb_end - 1) {
            if (psi_w != 0.0f) {
                #pragma unroll
                for (int i = 0; i < size(acc); i += 2) {
                    int const m = get<0>(tCcC(i));
                    int const d = get<1>(tCcC(i));
                    float2* const slot = reinterpret_cast<float2*>(&smem_dq[m * kDqPitch + d]);
                    float2 v2 = *slot;
                    v2.x = fmaf(acc(i), psi_w, v2.x);
                    v2.y = fmaf(acc(i + 1), psi_w, v2.y);
                    *slot = v2;
                }
                psi_w = 0.0f;
            }
        }
    }

    // Epilogue: smem_dq already holds TRUE unnormalized dQ (psi folded per
    // window); apply final_scale = sm/256 ONLY.  A row whose every tile was
    // a sentinel/skip leaves smem_dq at its zero-init: dq is exactly zero.
    float const fin = params.final_scale;
    int const b_idx = bh / params.h;
    int const h_idx = bh % params.h;
    auto* dq_base = reinterpret_cast<cutlass::bfloat16_t*>(params.dq_ptr)
        + int64_t(b_idx) * params.dq_batch_stride + int64_t(h_idx) * params.dq_head_stride
        + int64_t(mb) * kBM * params.dq_row_stride;
    // wgmma C fragment: even/odd i are column-adjacent -> packed u32 stores.
    #pragma unroll
    for (int i = 0; i < size(acc); i += 2) {
        int const m = get<0>(tCcC(i));
        int const d = get<1>(tCcC(i));
        float2 const v2 = *reinterpret_cast<float2 const*>(&smem_dq[m * kDqPitch + d]);
        cutlass::bfloat16_t lo(v2.x * fin), hi(v2.y * fin);
        uint32_t const pack = (uint32_t(reinterpret_cast<uint16_t const&>(hi)) << 16)
                            | uint32_t(reinterpret_cast<uint16_t const&>(lo));
        *reinterpret_cast<uint32_t*>(dq_base + int64_t(m) * params.dq_row_stride + d) = pack;
    }
}

}  // namespace fq8p2h128

// ## FP11 (M2): v11-specific symbol (the parent's run_fq8_dq_pass2 is a weak
// template symbol overlapping the other FA3-derived libraries, and the FP11-F2 sentinel
// makes this function numerically DIVERGENT from v10's: the overlap must be
// impossible by construction, same rationale as run_flash_bwd_fp11_d256).
template <bool Is_causal>
void run_fq8_dq_pass2_fp11_d128(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    using namespace fq8p2h128;
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
    StrideB stride_b = make_stride(int64_t(params.seqlen_k), _1{},
                                   int64_t(kBD) * params.seqlen_k);
    Tensor mB = make_tensor(make_gmem_ptr(static_cast<Element const*>(params.kt_ptr)),
                            shape_b, stride_b);
    TmaB tma_b = make_tma_copy(cute::SM90_TMA_LOAD{}, mB, SmemLayoutB{}(_, _, _0{}));
    P2Params p2{tma_a, tma_b, shape_a, shape_b, params.psi_ptr,
                params.dq_ptr, params.dq_row_stride, params.dq_head_stride,
                params.dq_batch_stride,
                params.scale_softmax / 256.0f, nbm, nbn, params.h, params.h_k};
    auto* kernel = &fq8p2h128::fq8_dq_pass2_kernel_hd128<Is_causal>;
    constexpr int kSmem = int(sizeof(SharedStorage));
    if (kSmem >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    }
    if (std::getenv("DM_DEBUG")) {
        fprintf(stderr, "[delta_matching pass 2 hd128] smem_size=%d grid=(%d,%d) block=128 kPromoteTiles=%d kDqPitch=%d\n",
                kSmem, nbm, bh, kPromoteTiles, kDqPitch);
    }
    dim3 grid(nbm, bh, 1);
    kernel<<<grid, 128, kSmem, stream>>>(p2);
    { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) { fprintf(stderr, "[delta_matching pass 2 hd128] launch: %s\n", cudaGetErrorString(e_)); exit(1); } }
}
