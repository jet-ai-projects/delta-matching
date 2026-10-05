/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.   (derived from flash-attention hopper/softmax.h)
 ******************************************************************************
 * ## FQ8F (delta_matching): the FAITHFUL-FORWARD softmax.
 *
 * This replaces FA3's `flash::Softmax<kNRows, Max_offset>` for the fp8 d256
 * recipe forward.  The ONLY thing that changes is the basis on which the
 * softmax probabilities are cast to e4m3, and the bookkeeping that a per-row,
 * per-32-key basis forces on the denominator.  Everything else (the online max
 * recurrence, the deferred quad reduction of the row sum, the natural-log LSE)
 * is upstream's, line for line.
 *
 * WHY.  The deployed Triton realization (the Triton reference implementation)
 * quantizes P per ROW and per 32-KEY BLOCK:
 *     p     = exp(s - m_run)                       # m_run = running row max
 *     ps    = max(rowmax_32(p), 1e-30) / 448
 *     p8    = e4m3(p / ps)                         # == e4m3(448 * exp(s - gm))
 *     acc  += dot(p8, v8) * (ps * v8sc)            # scales OUTSIDE the gemm
 * FA3 instead casts on a CONSTANT basis, P' = 2^8 * exp(s - m_run) (FA3's softmax.h
 * with Max_offset=8).  Both land ~4.6e-2 from exact fp32, but they
 * are DIFFERENTLY wrong: measured 2.77e-2 apart.  The two
 * schemes multiply P by different NON-DYADIC factors before the RTNE cast, so
 * each element lands at a different point inside its e4m3 binade; that, not
 * dynamic range, is the gap (a dyadic change of basis, e.g. 2^8 -> 2^7, is
 * measurably inert at 1.4e-7).  Reproducing the deployed codes is therefore a
 * BIT-LEVEL task: the per-row per-32-key basis has to be reproduced to ~1e-3
 * relative or the gap does not close.
 *
 * WHAT THIS HEADER IMPLEMENTS.  Per row i and per 32-key group g of a 128-key
 * tile, with gm_g = rowmax over group g of the descaled scores S and m the
 * running row max (both in natural units):
 *
 *     nu_g  = max(gm_g, m - B*ln2)                 # B = kPhatFloorB binades
 *     C_g   = RTNE_e4m3( 448 * exp(S - nu_g) )     # the e4m3 codes
 *
 * In log2 units, which is what the code carries (l2 == softmax_scale_log2 ==
 * log2(e) exactly here, because sm is baked into the Q codes so scale_softmax
 * is 1.0):
 *
 *     ms_g  = max(gm_g*l2, m*l2 - B) - log2(448)
 *     C_g   = RTNE_e4m3( exp2(S*l2 - ms_g) )
 *
 * B = infinity is EXACTLY the deployed Triton rule.  Finite B clamps the basis
 * for groups that sit more than 2^-B below the running row max, which bounds
 * the accumulator excursion of the rotation in the mainloop (see there).  Both
 * branches are bounded by 448 -- unclamped C = 448*exp(S-gm_g) <= 448, clamped
 * C = 448*2^B*exp(S-m) <= 448*2^B*2^-B = 448 -- so the +-448 saturating clamp
 * NEVER binds, exactly as on the deployed far path.  Measured convention
 * residual vs the deployed rule, over 64 real/randn/synthetic heads:
 * B=4 -> 3.6e-3 worst, B=8 -> 2.1e-4 worst, B=12 -> 1.6e-5 worst, B=16 -> 6.5e-7.
 *
 * THE CLAMP IS ALSO A CORRECTNESS MECHANISM, not only a range one: a fully
 * masked 32-column group has gm_g = -inf, and it is the clamp that keeps ms_g
 * finite so that exp2(-inf - finite) = 0 instead of NaN.
 *
 * THE DENOMINATOR IS THE SILENT-FAILURE RISK.  The four groups of a tile now
 * carry FOUR DIFFERENT bases, so a plain reduce_sum over the tile is wrong and
 * nothing asserts.  Each group's partial has to be reweighted:
 *     sigma_g = exp2(ms_g - m*l2)      (per row; <= 1/448 always)
 *     l      += sum_g sigma_g * S_g,   S_g = sum of group g's codes PRE-cast
 * which is exactly sum_j exp(S_j - m), i.e. the denominator still comes from
 * UNQUANTIZED p (as in the Triton reference).  Taking it from the e4m3 codes
 * instead costs 5.4e-3 on its own, more than the whole fidelity budget.
 * sigma_g is per-row hence quad-uniform, so upstream's deferred quad_allreduce
 * in finalize() stays valid.
 *
 * NO Max_offset ANYWHERE.  There is no constant 2^8 to divide back out, so
 * upstream's `if constexpr (Max_offset != 0) sum *= 1/2^8` in finalize() is
 * deleted.  The LSE this emits is upstream's formula with no shift: natural
 * log, true units, over the sm-scaled smooth-K-centered scores.  That is the
 * hard contract the frozen v10 backward reads (the original host API ->
 * flash_bwd_fp8v10_launch.h:93 -> mainloop_bwd_sm90_fp8v10.hpp:638,1247).
 *
 * ## V3-TILEPROMOTE (this fork).  finalize() is back to upstream's SCALAR
 * form.  The mainloop no longer leaves the output accumulator in a per-row
 * P-hat basis: it forms each 128-key tile's product in a ZERO-INITIALISED
 * tensor-core accumulator, applies the four per-32-key bases inside that temp
 * (three CUDA-core rho rescales, exactly the ratios the rotation used), and
 * then promotes the whole tile into tOrO with basis_last() on CUDA cores.
 * tOrO therefore always holds the true numerator in exp(-m_run) units, and the
 * cross-tile rescale is upstream's alpha = scores_scale.  basis_last() is
 * retained and is now called ONCE PER TILE (it is the promote's w_3), not once
 * at the end.  Nothing about the codes, ms_[g], its clamp, row_sum or the LSE
 * changes.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"     // fa_src/hopper/utils.h: Allreduce, MaxOp, SumOp, convert_layout_acc_rowcol
// Upstream softmax.h, UNMODIFIED, for its reduction helpers (quad_allreduce_,
// thread_reduce_).  This package deliberately contains no file named softmax.h,
// so the include resolves to fa_src/hopper's copy even though the package dir
// is first on the include path.  flash::Softmax itself is a template that this
// binary never instantiates, so it emits no symbol that could collide with the
// frozen extension's.
#include "softmax.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// ## FQ8F: thread-local reduction over ONE 32-key group, i.e. the column range
// [8*G, 8*G+8) of the rowcol fragment view.  The wgmma C layout gives thread t
// columns 8*(ni>>1) + 2*(t%4) + (ni&1), so column/32 == ni>>3 and a thread's
// low bits never cross a 32-column boundary: the four 8-wide chunks of the
// fragment ARE the four 32-key groups, and the split is free.
// ## Delta-Matching-128 (M19): kCols = rowcol columns per group = 32 / kGroups (8 for the d256
// 32-key groups, 16 for the hd128 64-key groups); the d256 instantiation (kCols = 8) is the M17 loop verbatim.
template <int G, bool zero_init = true, int kCols = 8, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_group_(Tensor<Engine0, Layout0> const& tensor,
                                                     Tensor<Engine1, Layout1>& summary, Operator& op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    static_assert(decltype(size<1>(tensor))::value == 32 || decltype(size<1>(tensor))::value == 64,
                  "FQ8F: rowcol fragment must be 32 wide (kBlockN=128, 2 WGs) or 64 (kBlockN=256, hd128 F3)");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int ni = kCols * G; ni < kCols * G + kCols; ++ni) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
            summary(mi) = (zero_init && ni == kCols * G) ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// kPhatFloorB: the P-hat basis floor, in BINADES below the running row max.
// See the header comment.  12 is the shipped value: convention residual 1.6e-5
// worst measured, an order below the reimplementation floor, with ~15 binades
// of accumulator excursion, which fp32's 8-bit exponent absorbs trivially and
// whose fp32 roundings are damped by exactly the factor that inflated them.
// ## Delta-Matching-128 (M19): kGroups = P-hat groups per 128-key tile (4 x 32 keys at d256 = the certified M17
// realization, textually unchanged below; 2 x 64 keys at hd128 = the Triton reference's [128, 64] blocks).
// ## Delta-Matching-128 F3 (M19): kRowcol = the rowcol fragment width (32 at kBlockN 128, 64 at kBlockN 256), so a group spans
// kRowcol / kGroups columns (8 at d256; 16 at hd128 with 128-key tiles; 16 again at hd128 F3 with 256-key tiles).
template <int kNRows, int kPhatFloorB = 12, int kGroups = 4, int kRowcol = 32>
struct SoftmaxFq8F {
    static_assert(kGroups == 4 || kGroups == 2, "SoftmaxFq8F: kGroups must be 4 (d256, hd128 F3) or 2 (hd128)");
    static constexpr int kColsPerGroup = kRowcol / kGroups;

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    static constexpr float kLog2_448 = 8.8073549220576042f;   // log2(448), the e4m3 finite max
    static constexpr float kFloorB = float(kPhatFloorB);

    TensorT row_max, row_sum;
    // ## FQ8F persistent state (2 * 4 = 8 floats/thread at kNRows=2): the
    // per-row per-group log2 offsets of the tile whose e4m3 codes currently
    // live in tOrP.  The mainloop reads these to build its basis ratios rho,
    // and reads ms_[3] once more at the end for the finalize scale.
    // Register discipline: ms_[] is also the SCRATCH for the four group maxima
    // between max_get_scale_grouped's reduction and its own in-place transform,
    // so the kernel never holds grp_max and ms simultaneously.
    TensorT ms_[kGroups];
    float const softmax_scale_log2;

    CUTLASS_DEVICE SoftmaxFq8F(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};

    // row_max * log2(e), with upstream's -inf guard (a row with no live key yet
    // would otherwise give -inf - -inf = NaN downstream).
    template <bool Check_inf>
    __forceinline__ __device__ float max_l2(int mi) const {
        return Check_inf ? (row_max(mi) == -INFINITY ? 0.f : row_max(mi) * softmax_scale_log2)
                         : row_max(mi) * softmax_scale_log2;
    }

    // Upstream max_get_scale, with the single row reduction replaced by four
    // per-32-key-group reductions (same 32 comparisons per thread; 4
    // quad_allreduce instead of 1), and with the per-group P-hat basis ms_[g]
    // built here, in place, once row_max is known.
    //
    // row_sum *= scores_scale STAYS HERE, exactly where upstream has it
    // (softmax.h:120): nothing the mainloop later folds into an accumulator
    // rescale may ever reach the denominator.
    template <bool Is_first, bool Check_inf = false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale_grouped(Tensor0& acc_s) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        MaxOp<float> max_op;
        // ms_[g] temporarily holds gm_g (natural units).
        if constexpr (kGroups == 4 && kColsPerGroup == 8) {
            flash::thread_reduce_group_<0, /*zero_init=*/true>(scores, ms_[0], max_op);
            flash::thread_reduce_group_<1, /*zero_init=*/true>(scores, ms_[1], max_op);
            flash::thread_reduce_group_<2, /*zero_init=*/true>(scores, ms_[2], max_op);
            flash::thread_reduce_group_<3, /*zero_init=*/true>(scores, ms_[3], max_op);
        } else if constexpr (kGroups == 4) {
            // ## F3: four 64-key groups of 16 rowcol columns each (kBlockN 256 at hd128)
            flash::thread_reduce_group_<0, /*zero_init=*/true, kColsPerGroup>(scores, ms_[0], max_op);
            flash::thread_reduce_group_<1, /*zero_init=*/true, kColsPerGroup>(scores, ms_[1], max_op);
            flash::thread_reduce_group_<2, /*zero_init=*/true, kColsPerGroup>(scores, ms_[2], max_op);
            flash::thread_reduce_group_<3, /*zero_init=*/true, kColsPerGroup>(scores, ms_[3], max_op);
        } else {
            // ## Delta-Matching-128: two 64-key groups of 16 rowcol columns each.
            flash::thread_reduce_group_<0, /*zero_init=*/true, kColsPerGroup>(scores, ms_[0], max_op);
            flash::thread_reduce_group_<1, /*zero_init=*/true, kColsPerGroup>(scores, ms_[1], max_op);
        }
        #pragma unroll
        for (int g = 0; g < kGroups; ++g) { flash::quad_allreduce_(ms_[g], ms_[g], max_op); }

        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            float const tile_max = [&] {
                if constexpr (kGroups == 4) { return fmaxf(fmaxf(ms_[0](mi), ms_[1](mi)), fmaxf(ms_[2](mi), ms_[3](mi))); }
                else { return fmaxf(ms_[0](mi), ms_[1](mi)); }
            }();
            if constexpr (Is_first) {
                row_max(mi) = tile_max;
                scores_scale(mi) = 1.f;
            } else {
                float const scores_max_prev = row_max(mi);
                row_max(mi) = fmaxf(scores_max_prev, tile_max);
                float const scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
            // ms_g = max(gm_g*l2, m*l2 - B) - log2(448), in place.
            float const floor_l2 = max_l2<Check_inf>(mi) - kFloorB;
            #pragma unroll
            for (int g = 0; g < kGroups; ++g) {
                ms_[g](mi) = fmaxf(ms_[g](mi) * softmax_scale_log2, floor_l2) - kLog2_448;
            }
        }
        return scores_scale;
    };

    // Upstream online_softmax, with (a) the per-group clamped basis in place of
    // the constant Max_offset, and (b) the four group partials reweighted onto
    // the running-max basis before they enter row_sum.
    template <bool Is_first, bool Check_inf = false, typename Tensor0>
    __forceinline__ __device__ void online_softmax_grouped(Tensor0& acc_s) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        static_assert(decltype(size<1>(scores))::value == kRowcol, "FQ8F: the rowcol fragment width must match kRowcol");

        // P codes, pre-cast: exp2(S*l2 - ms_g).  Identical FFMA + ex2.approx
        // count to upstream; only which register holds the offset changes, and
        // ni>>3 is a compile-time constant inside the unrolled loop.
        #pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                scores(mi, ni) = exp2f(scores(mi, ni) * softmax_scale_log2 - ms_[ni / kColsPerGroup](mi));
            }
        }
        // Denominator: four group partials, each reweighted by
        // sigma_g = exp2(ms_g - m*l2) onto the running-max basis, so the sum is
        // exactly sum_j exp(S_j - m) over UNQUANTIZED p.  A plain reduce_sum
        // over a tile carrying four different bases would be silently wrong.
        // warp_reduce stays deferred to finalize(); sigma_g is per-row hence
        // quad-uniform, so the deferral is still valid.
        SumOp<float> sum_op;
        TensorT part, acc;
        cute::fill(acc, 0.f);
        auto fold = [&](auto g_c, TensorT const& part_) {
            constexpr int g = decltype(g_c)::value;
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                acc(mi) += exp2f(ms_[g](mi) - max_l2<Check_inf>(mi)) * part_(mi);
            }
        };
        if constexpr (kGroups == 4 && kColsPerGroup == 8) {
            flash::thread_reduce_group_<0, true>(scores, part, sum_op); fold(cute::Int<0>{}, part);
            flash::thread_reduce_group_<1, true>(scores, part, sum_op); fold(cute::Int<1>{}, part);
            flash::thread_reduce_group_<2, true>(scores, part, sum_op); fold(cute::Int<2>{}, part);
            flash::thread_reduce_group_<3, true>(scores, part, sum_op); fold(cute::Int<3>{}, part);
        } else if constexpr (kGroups == 4) {
            flash::thread_reduce_group_<0, true, kColsPerGroup>(scores, part, sum_op); fold(cute::Int<0>{}, part);
            flash::thread_reduce_group_<1, true, kColsPerGroup>(scores, part, sum_op); fold(cute::Int<1>{}, part);
            flash::thread_reduce_group_<2, true, kColsPerGroup>(scores, part, sum_op); fold(cute::Int<2>{}, part);
            flash::thread_reduce_group_<3, true, kColsPerGroup>(scores, part, sum_op); fold(cute::Int<3>{}, part);
        } else {
            flash::thread_reduce_group_<0, true, kColsPerGroup>(scores, part, sum_op); fold(cute::Int<0>{}, part);
            flash::thread_reduce_group_<1, true, kColsPerGroup>(scores, part, sum_op); fold(cute::Int<1>{}, part);
        }
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) { row_sum(mi) = Is_first ? acc(mi) : row_sum(mi) + acc(mi); }
    };

    // ## FQ8F: c_last = v8sc_3 * 2^(ms_3 - m*log2e), the basis of the tile's
    // group-3 product.  ## V3-TILEPROMOTE: this is the promote weight w_3, read
    // once per 128-key tile while ms_[3] and row_max still belong to the tile
    // whose codes were just consumed (max_get_scale_grouped destroys both).
    __forceinline__ __device__ TensorT basis_last(float v8sc_3) const { return basis<kGroups - 1>(v8sc_3); }

    // ## V3-TILEPROMOTE: the promote weight of ANY group,
    //     w_g = v8sc_g * 2^(ms_g - m*l2),
    // i.e. sigma_g * v8sc_g.  G is a template parameter so ms_[G] stays a
    // compile-time register index (a runtime index spills ms_[4] to local).
    // Check_inf is unconditionally true, exactly as basis_last was: the PV
    // chain for tile j runs one step later than tile j's softmax, so the
    // enclosing fwd_step's Check_inf belongs to a DIFFERENT tile and must not
    // be threaded through here.  A still-dead row (row_max == -inf) gets
    // max_l2 = 0 and hence a finite w, so the promote computes 0*finite = 0 and
    // the row exits through finalize's sum == 0 branch to O = 0, LSE = -inf.
    template <int G>
    __forceinline__ __device__ TensorT basis(float v8sc_g) const {
        TensorT c;
        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            c(mi) = v8sc_g * exp2f(ms_[G](mi) - max_l2</*Check_inf=*/true>(mi));
        }
        return c;
    }

    // ## V3-TILEPROMOTE: finalize reverts to UPSTREAM's SCALAR form.  The
    // accumulator tOrO is now kept in true exp(-m_run) units at every tile
    // boundary (the per-row P-hat basis is consumed by promote_o below, inside
    // the tile), so there is no per-row basis left to undo here and
    // final_scale is 1.  The Max_offset unshift stays deleted (no constant
    // basis).  LSE is untouched: natural log, true units, -inf on a fully
    // masked row -- the frozen v10 backward's contract.
    __forceinline__ __device__ TensorT finalize(float const final_scale = 1.f) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            row_sum(mi) = (sum == 0.f || sum != sum)
                ? -INFINITY
                : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };

    // ## V3-TILEPROMOTE: the promote.  acc_o += acc_d * w, per row, on CUDA
    // cores in fp32.  acc_d is the 128-key TILE's product, formed in a
    // ZERO-INITIALISED tensor-core accumulator, so the ~14-bit fp8 wgmma
    // accumulate is committed relative to the TILE partial and never to the
    // whole running numerator.  Same rowcol machinery as rescale_o, one extra
    // operand; w is indexed by mi only (mode 0 of the rowcol view is the row).
    template <typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void promote_o(Tensor1& acc_o, Tensor2 const& acc_d, TensorT const& w) {
        Tensor o_ = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        Tensor d_ = make_tensor(acc_d.data(), flash::convert_layout_acc_rowcol(acc_d.layout()));
        static_assert(CUTE_STATIC_V(size<0>(o_)) == kNRows);
        CUTE_STATIC_ASSERT_V(size<1>(o_) == size<1>(d_));
        #pragma unroll
        for (int mi = 0; mi < size<0>(o_); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(o_); ++ni) {
                o_(mi, ni) = fmaf(d_(mi, ni), w(mi), o_(mi, ni));
            }
        }
    };

    // Upstream rescale_o, unchanged.  The mainloop now calls it three extra
    // times per 128-key tile (once per 32-key group boundary).
    template <typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1& acc_o, TensorT const& scores_scale) {
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
        }
    };

};

}  // namespace flash
