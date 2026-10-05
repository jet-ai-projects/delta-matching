// delta_matching host API: the torch.library ops of the fp8 attention kernels (library `delta_matching`).
//   fwd_fp8          forward on e4m3 codes with per-block fp32 scales -> O (b, s, h, d) bf16, natural-log LSE (b, h, s) fp32
//   bwd_fp8_t        split backward on caller-made transposed twins (Q^T, dO^T): pass 1 = dK / dV and the dS' code tiles,
//                    pass 2 = dQ as a block-scaled e4m3 GEMM; the route the Python API uses
//   bwd_fp8          the same backward with the twins built here (twin_pi)
//   delta_fp8        the in-kernel matched delta: pass 0 on the backward's own fp8 P / dP, then a fixed-order fp64 reduce
//   bwd_fp8_bf16dp   (DM_BF16_DP=1 builds only) the backward with the dP = dO V^T matmul on the raw bf16 operands
//   twin_pi, build_gt, build_hd128, build_bf16dp: helpers and build flags; the *_dbg ops exist in DM_DBG=1 builds.
// Every op checks its inputs, runs on the inputs' device and that device's current stream, and is deterministic.
// Self-attention only (seqlen_q == seqlen_k when causal); head_dim 256, or 128 in a build with the head_dim-128 kernels.
// The module init and the parameter plumbing follow FlashAttention-3 (hopper/flash_api.cpp), Copyright (c) 2024,
// Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao; BSD 3-Clause (THIRD_PARTY_NOTICES.md).

#include "dm_flagset.h"   // stops a build outside the release flag set
#include <Python.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <cmath>

#include "flash_bwd_fp8_params.h"   // Flash_fwd_fp8_params / Flash_bwd_fp8_params (pulls flash.h)
// ## bf16-dP: the build (_buildcfg.py) passes -DFA3FP11_P1R_SAGEDP to every TU; this keeps the host TU compilable alone.
#ifndef FA3FP11_P1R_SAGEDP
#define FA3FP11_P1R_SAGEDP 0
#endif


extern "C" {
/* Dummy empty _C module so the .so is also importable as a Python module.
   torch.ops.load_library() only needs the TORCH_LIBRARY static initializers. */
PyObject* PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "delta_matching._C",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}
}

template <bool Is_causal>
void run_flash_fwd_fp8f_d256(Flash_fwd_fp8_params &params, cudaStream_t stream);
template <bool Is_causal>
void run_flash_bwd_fp11_d256(Flash_bwd_fp8_params &params, cudaStream_t stream);
template <bool Is_causal>
void run_flash_delta_fp11_d256(Flash_bwd_fp8_params &params, cudaStream_t stream);
// ## Delta-Matching-128 (M19): the head_dim 128 instantiations (compiled when FA3FP11_HD128=1;
// the same ops dispatch on head_size, the block-scale contract is [128, 64] = the Triton reference's).
template <bool Is_causal>
void run_flash_fwd_fp8f_d128(Flash_fwd_fp8_params &params, cudaStream_t stream);
template <bool Is_causal>
void run_flash_bwd_fp11_d128(Flash_bwd_fp8_params &params, cudaStream_t stream);
template <bool Is_causal>
void run_flash_delta_fp11_d128(Flash_bwd_fp8_params &params, cudaStream_t stream);
// The block-scale contract per head dim (blocks()): Q/dO rows per scale, K/V keys per scale, pass-1 kBlockN.
inline int fp11_quant_rows_q(int d) { return d >= 256 ? 64 : 128; }
inline int fp11_quant_keys(int d) { return d >= 256 ? 32 : 64; }
inline int fp11_bwd_block_n(int d) { return d >= 256 ? 64 : 128; }
inline bool fp11_hd_supported(int d) { return d == 256 || (FA3FP11_HD128 && d == 128); }
// ## FP11-M17 GT: in-.so twin maker (fp11_twin_kernel.cu): x8 (b, s, h, d) e4m3 with strides
// -> xt8 (b, h, d, s) contiguous e4m3 with the s positions in the pass-1 transposer's pi order.
void fp11_make_twin_pi(void const* x8, void* xt8, int b, int s, int h, int d,
                       int64_t sb, int64_t ss, int64_t sh, cudaStream_t stream);

namespace {

// ---- input checks shared by the ops (host side only; the kernels are unchanged) ----
void dm_check_inputs(char const* op, at::Tensor const& ref, std::initializer_list<at::Tensor> ts) {
    TORCH_CHECK(ref.is_cuda(), "delta_matching ", op, ": inputs must be CUDA tensors");
    for (auto const& t : ts) {
        if (!t.defined()) { continue; }
        TORCH_CHECK(t.is_cuda() && t.device() == ref.device(), "delta_matching ", op, ": every tensor must be on ",
                    ref.device(), " (got one on ", t.device(), ")");
    }
    auto const* dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 9 && dprops->minor == 0, "delta_matching ", op,
                ": needs an sm90 (Hopper) GPU; this device is sm", dprops->major, dprops->minor);
}
void dm_check_codes(char const* op, std::initializer_list<at::Tensor> ts) {
    for (auto const& t : ts) {
        if (!t.defined()) { continue; }
        TORCH_CHECK(t.dim() == 4 && t.scalar_type() == at::kFloat8_e4m3fn, "delta_matching ", op, ": codes must be 4-D e4m3 tensors");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "delta_matching ", op, ": codes must be 16-byte aligned");
        for (int i = 0; i < 3; ++i) {
            TORCH_CHECK(t.stride(i) % 16 == 0, "delta_matching ", op, ": code strides must be 16-byte multiples (dim ", i, ")");
        }
    }
}
void dm_check_scales(char const* op, std::initializer_list<at::Tensor> ts) {
    for (auto const& t : ts) {
        TORCH_CHECK(t.dim() == 3 && t.scalar_type() == at::kFloat, "delta_matching ", op, ": scales must be 3-D fp32 tensors");
    }
}

// ## FQ8F: the faithful recipe fp8 d256 forward on FA3's schedule (per-row,
// per-32-key P-hat basis; PV promoted once per 128-key tile).  Returns
// (O bf16 (b,s,h,d), LSE fp32 (b,h,s), natural log, centered-K convention).
// softmax_scale is passed as 1.0 to the kernel (sm is baked into the Q codes);
// the arg is kept for interface symmetry and ignored.
std::tuple<at::Tensor, at::Tensor> fwd_fp8_core(
    at::Tensor q8,      // (b, s_q, h, d) e4m3 (sm-scaled codes)
    at::Tensor k8,      // (b, s_k, h, d) e4m3 (smooth-K centered codes)
    at::Tensor v8,      // (b, s_k, h, d) e4m3
    at::Tensor q8s,     // (b, h, s_q/64) fp32
    at::Tensor k8s,     // (b, h, s_k/32) fp32
    at::Tensor v8s,     // (b, h, s_k/32) fp32
    double softmax_scale,
    bool is_causal
) {
    c10::cuda::CUDAGuard device_guard(q8.device());
    dm_check_inputs("fwd", q8, {k8, v8, q8s, k8s, v8s});
    dm_check_codes("fwd", {q8, k8, v8});
    dm_check_scales("fwd", {q8s, k8s, v8s});
    auto const sizes = q8.sizes();
    int const batch_size = sizes[0], seqlen_q = sizes[1], num_heads = sizes[2], head_size = sizes[3];
    int const seqlen_k = k8.size(1);
    // ## FP11-M16 (native GQA, READ side): K/V (and their scales) may carry
    // h_k <= h heads; the kernels index them by KV head (h % h_k == 0).  With
    // h_k == h everything is the M10 path.
    int const num_heads_k = k8.size(2);
    TORCH_CHECK(batch_size > 0 && seqlen_q > 0 && seqlen_k > 0 && num_heads > 0 && num_heads_k > 0, "delta_matching fwd: empty input");
    TORCH_CHECK(k8.sizes() == v8.sizes() && k8.size(0) == batch_size && k8.size(3) == head_size,
                "delta_matching fwd: k8 and v8 must be (b, s_k, h_k, d) with q8's batch and head_dim");
    TORCH_CHECK(!is_causal || seqlen_q == seqlen_k, "delta_matching fwd: causal attention needs seqlen_q == seqlen_k (got ",
                seqlen_q, ", ", seqlen_k, ")");
    TORCH_CHECK(v8.size(2) == num_heads_k && num_heads % num_heads_k == 0,
                "delta_matching fwd: k8/v8 heads must divide q8 heads (native GQA)");
    TORCH_CHECK(fp11_hd_supported(head_size), "delta_matching fwd: head_dim must be 256 (or 128 in a Delta-Matching-128 build)");
    TORCH_CHECK(seqlen_q % 128 == 0 && seqlen_k % 128 == 0, "delta_matching fwd: seqlens must be 128-multiples");
    // ## M20: the F3 forward (256-key tiles at hd128, FA3FP11_HD128_BN256=1) mis-handles a partial last key
    // tile (measured on an earlier build: seqlen_k 640 / 896 give a finite O but a backward that is not finite; 768 / 1024 are fine;
    // the 128-key forward is fine at 640 / 896).  Refuse loudly instead of returning a LSE the backward cannot use.
    if (head_size == 128 && FA3FP11_HD128_BN256) {
        TORCH_CHECK(seqlen_k % 256 == 0, "delta_matching fwd (Delta-Matching-128, 256-key forward tiles): seqlen_k must be a 256-multiple (got ", seqlen_k, ")");
    }
    for (auto const& t : {q8, k8, v8}) { TORCH_CHECK(t.stride(-1) == 1); }
    for (auto const& t : {q8s, k8s, v8s}) { TORCH_CHECK(t.is_contiguous()); }
    // ## FQ8F: the two scale-granularity contracts the v10 backward reads
    // (mainloop_bwd_sm90_fp8v10.hpp:1182-1187) and that nothing else asserts.
    // ## Delta-Matching-128: [64, 32] blocks at d256, [128, 64] at hd128.
    int const rq = fp11_quant_rows_q(head_size), rk = fp11_quant_keys(head_size);
    TORCH_CHECK(q8s.size(0) == batch_size && q8s.size(1) == num_heads && q8s.size(2) == seqlen_q / rq, "delta_matching: q8s must be (b, h, s_q/", rq, ")");
    TORCH_CHECK(k8s.size(0) == batch_size && k8s.size(1) == num_heads_k && k8s.size(2) == seqlen_k / rk, "delta_matching: k8s must be (b, h_k, s_k/", rk, ")");
    TORCH_CHECK(v8s.size(0) == batch_size && v8s.size(1) == num_heads_k && v8s.size(2) == seqlen_k / rk, "delta_matching: v8s must be (b, h_k, s_k/", rk, ")");
    auto opts = q8.options();
    at::Tensor o = torch::empty({batch_size, seqlen_q, num_heads, head_size}, opts.dtype(at::kBFloat16));
    at::Tensor lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor tile_count_sem = torch::zeros({1}, opts.dtype(at::kInt));

    Flash_fwd_fp8_params params{};
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;   // ## FP11-M16: native GQA
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q;
    params.seqlen_k_rounded = seqlen_k;
    params.d = head_size;
    params.dv = head_size;
    params.d_rounded = head_size;
    params.dv_rounded = head_size;
    params.scale_softmax = 1.0f;   // sm is baked into the Q codes
    params.is_causal = is_causal;
    params.is_bf16 = true;
    params.q_ptr = q8.data_ptr(); params.k_ptr = k8.data_ptr(); params.v_ptr = v8.data_ptr();
    params.q_row_stride = q8.stride(1); params.q_head_stride = q8.stride(2); params.q_batch_stride = q8.stride(0);
    params.k_row_stride = k8.stride(1); params.k_head_stride = k8.stride(2); params.k_batch_stride = k8.stride(0);
    params.v_row_stride = v8.stride(1); params.v_head_stride = v8.stride(2); params.v_batch_stride = v8.stride(0);
    params.v_dim_stride = v8.stride(3);
    params.o_ptr = o.data_ptr();
    params.o_row_stride = o.stride(1); params.o_head_stride = o.stride(2); params.o_batch_stride = o.stride(0);
    params.softmax_lse_ptr = lse.data_ptr();
    params.tile_count_semaphore = tile_count_sem.data_ptr<int>();
    params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    params.arch = 90;
    params.num_splits = 1;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.q8s_ptr = q8s.data_ptr<float>();
    params.k8s_ptr = k8s.data_ptr<float>();
    params.v8s_ptr = v8s.data_ptr<float>();
    params.nblk_q = q8s.size(2);
    params.nblk_k = k8s.size(2);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (head_size == 128) {
        if (is_causal) { run_flash_fwd_fp8f_d128<true>(params, stream); }   // ## Delta-Matching-128
        else { run_flash_fwd_fp8f_d128<false>(params, stream); }
    } else {
        if (is_causal) { run_flash_fwd_fp8f_d256<true>(params, stream); }
        else { run_flash_fwd_fp8f_d256<false>(params, stream); }
    }
    return {o, lse};
}

std::tuple<at::Tensor, at::Tensor> fwd_fp8(
    at::Tensor q8, at::Tensor k8, at::Tensor v8, at::Tensor q8s, at::Tensor k8s, at::Tensor v8s,
    double softmax_scale, bool is_causal
) {
    return fwd_fp8_core(q8, k8, v8, q8s, k8s, v8s, softmax_scale, is_causal);
}

// ## FQ8-V10 (SPLIT): the 7/7-fp8 d256 backward.  Pass 1 (column-parallel):
// dK/dV + the dS' code tiles and per-tile psi; pass 2 (row-parallel): dQ as a
// block-scaled e4m3 GEMM, register-accumulated (deterministic).  Whole-op
// accounting: the workspace allocations, lse*log2e prep, and both kernel
// launches are inside this op.  Returns (dq, dk, dv) bf16 (b, s, h, d).
// ## FP11-M5: bwd_fp8's whole body lives in this core so the FA3FP11_DBG
// build's bwd_fp8_dbg can drive the identical launch with the debug
// workspaces attached.  The shipped op passes nullptrs (params fields exist
// but no kernel code reads them in the shipped build): host-side refactor
// only, same params, same launches (dq / dk / dv bitwise unchanged).
#ifdef FA3FP11_DBG
// ## DBG: capture of the pass-1 -> pass-2 workspaces (the realized dS' e4m3 codes and the per-tile psi) for
// bwd_fp8_dbg_ds (the row-sum residual AFTER dS quantization).  Null outside that op: every other op is unchanged.
static at::Tensor* g_dbg_ds8 = nullptr;
static at::Tensor* g_dbg_psi = nullptr;
#endif
std::tuple<at::Tensor, at::Tensor, at::Tensor> bwd_fp8_core(
    at::Tensor dout8,   // (b, s_q, h, d) e4m3 codes
    at::Tensor q8,      // (b, s_q, h, d) e4m3
    at::Tensor k8,      // (b, s_k, h, d) e4m3 (smooth-K centered)
    at::Tensor v8,      // (b, s_k, h, d) e4m3
    at::Tensor lse,     // (b, h, s_q) fp32 (natural-log LSE)
    at::Tensor delta,   // (b, h, s_q) fp32 (the recipe delta: stale or matched)
    at::Tensor kt8,     // (b, h, d, s_k) e4m3 transposed K twin, contiguous
    at::Tensor qt8,     // ## FP11-M17 GT: (b, h, d, s_q) e4m3 Q twin (pi-ordered s), contiguous; ignored at GT=0
    at::Tensor dot8,    // ## FP11-M17 GT: (b, h, d, s_q) e4m3 dO twin (pi-ordered s), contiguous; ignored at GT=0
    at::Tensor q8s,     // (b, h, s_q/64) fp32
    at::Tensor k8s,     // (b, h, s_k/32) fp32
    at::Tensor v8s,     // (b, h, s_k/32) fp32
    at::Tensor do8s,    // (b, h, s_q/64) fp32
    double softmax_scale,
    bool is_causal,
    float* dbg_rows_ptr,
    float* dbg_rowsabs_ptr,
    unsigned int* dbg_cks_ptr,
    // ## bf16-dP: the RAW bf16 V / dO for SageBwd's exempt matmul (5) dP = dO V^T.  Undefined on every
    // certified call; required, and required to be absent, exactly as FA3FP11_P1R_SAGEDP dictates.
    at::Tensor vraw = at::Tensor(),
    at::Tensor doraw = at::Tensor()
) {
    c10::cuda::CUDAGuard device_guard(q8.device());
    dm_check_inputs("bwd", q8, {dout8, k8, v8, lse, delta, kt8, qt8, dot8, q8s, k8s, v8s, do8s, vraw, doraw});
    dm_check_codes("bwd", {dout8, q8, k8, v8, kt8, qt8, dot8});
    dm_check_scales("bwd", {q8s, k8s, v8s, do8s});
    auto const sizes = q8.sizes();
    int const batch_size = sizes[0], seqlen_q = sizes[1], num_heads = sizes[2], head_size = sizes[3];
    int const seqlen_k = k8.size(1);
    // ## FP11-M16 (native GQA, READ side): k8/v8/kt8 and k8s/v8s per KV head
    // (h_k <= h, h % h_k == 0); dk/dv stay per QUERY head (the caller folds).
    int const num_heads_k = k8.size(2);
    TORCH_CHECK(batch_size > 0 && seqlen_q > 0 && seqlen_k > 0 && num_heads > 0 && num_heads_k > 0, "delta_matching bwd: empty input");
    TORCH_CHECK(k8.sizes() == v8.sizes() && k8.size(0) == batch_size && k8.size(3) == head_size,
                "delta_matching bwd: k8 and v8 must be (b, s_k, h_k, d) with q8's batch and head_dim");
    TORCH_CHECK(!is_causal || seqlen_q == seqlen_k, "delta_matching bwd: causal attention needs seqlen_q == seqlen_k (got ",
                seqlen_q, ", ", seqlen_k, ")");
    TORCH_CHECK(dout8.sizes() == q8.sizes(), "delta_matching bwd: dout8 must have q8's shape");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && delta.scalar_type() == at::kFloat
                && lse.numel() == int64_t(batch_size) * num_heads * seqlen_q && delta.numel() == lse.numel(),
                "delta_matching bwd: lse and delta must be (b, h, s_q) fp32");
    TORCH_CHECK(int64_t(batch_size) * num_heads <= 65535, "delta_matching bwd: batch * heads must be at most 65535");
    TORCH_CHECK(v8.size(2) == num_heads_k && num_heads % num_heads_k == 0,
                "delta_matching bwd: k8/v8 heads must divide q8 heads (native GQA)");
    TORCH_CHECK(kt8.size(0) == batch_size && kt8.size(1) == num_heads_k && kt8.size(2) == head_size && kt8.size(3) == seqlen_k,
                "delta_matching bwd: kt8 must be (b, h_k, d, s_k)");
    TORCH_CHECK(k8s.size(1) == num_heads_k && v8s.size(1) == num_heads_k && q8s.size(1) == num_heads && do8s.size(1) == num_heads,
                "delta_matching bwd: scale head counts (q/do: h, k/v: h_k)");
    TORCH_CHECK(fp11_hd_supported(head_size), "delta_matching bwd: head_dim must be 256 (or 128 in a Delta-Matching-128 build)");
    int const kbn = fp11_bwd_block_n(head_size);   // ## Delta-Matching-128: pass-1 key block (64 at d256, 128 at hd128)
    TORCH_CHECK(seqlen_q % 64 == 0 && seqlen_k % kbn == 0, "delta_matching bwd: seqlen_q must be a 64-multiple, seqlen_k a ", kbn, "-multiple");
    {
        int const rq = fp11_quant_rows_q(head_size), rk = fp11_quant_keys(head_size);
        TORCH_CHECK(seqlen_q % rq == 0 && q8s.size(2) == seqlen_q / rq && do8s.size(2) == seqlen_q / rq, "delta_matching bwd: q8s/do8s must be (b, h, s_q/", rq, ")");
        TORCH_CHECK(k8s.size(2) == seqlen_k / rk && v8s.size(2) == seqlen_k / rk, "delta_matching bwd: k8s/v8s must be (b, h_k, s_k/", rk, ")");
    }
    for (auto const& t : {dout8, q8, k8, v8}) { TORCH_CHECK(t.stride(-1) == 1); }
    for (auto const& t : {kt8, q8s, k8s, v8s, do8s, lse, delta}) { TORCH_CHECK(t.is_contiguous()); }
    auto opts = q8.options();
    int const seqlen_q_rounded = seqlen_q;    // 64-multiple asserted
    int const head_size_rounded = head_size;

    at::Tensor dq = torch::empty({batch_size, seqlen_q, num_heads, head_size}, opts.dtype(at::kBFloat16));
    at::Tensor dk = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts.dtype(at::kBFloat16));
    at::Tensor dv = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts.dtype(at::kBFloat16));
    // Dummy dq_accum: the split backward has no cross-CTA dQ reduction; the
    // pointer is never dereferenced (plumbing only).
    at::Tensor dq_accum = torch::empty({4}, opts.dtype(at::kFloat));
    // ## FP11-M17 LSEK: the kernels form lse*log2e in-register (bitwise the host prep).
    at::Tensor lse_log2 = lse;
    // The SingleTile schedulers never read the semaphore: no memset launch.
    at::Tensor tile_count_sem = torch::empty({1}, opts.dtype(at::kInt));
    // Pass-1 -> pass-2 workspaces: tile-contiguous dS' codes + per-tile psi.
    // For causal only nb <= mb tiles are written or read: no zeroing needed.
    int const nbm = seqlen_q / 64, nbn = seqlen_k / kbn;
    at::Tensor ds8 = torch::empty({int64_t(batch_size) * num_heads, nbm, nbn, 64, kbn}, opts);   // ## Delta-Matching-128: [64, kbn] tiles
#ifndef FA3FP11_HALFPSI
#define FA3FP11_HALFPSI 0
#endif
    // ## M23 HALFPSI: at head_dim 128 pass 1 publishes one psi per (tile, warpgroup half): [BH, nbm, nbn, 2].
    at::Tensor psi = torch::empty({int64_t(batch_size) * num_heads, nbm, nbn, (FA3FP11_HALFPSI && head_size == 128) ? 2 : 1}, opts.dtype(at::kFloat));
#ifdef FA3FP11_DBG
    if (g_dbg_ds8 != nullptr) { *g_dbg_ds8 = ds8; *g_dbg_psi = psi; }   // ## DBG (bwd_fp8_dbg_ds only)
#endif

    Flash_bwd_fp8_params params{};
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;   // ## FP11-M16: native GQA
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k;
    params.d = head_size;
    params.dv = head_size;
    params.d_rounded = head_size_rounded;
    params.dv_rounded = head_size_rounded;
    params.scale_softmax = float(softmax_scale);
    params.is_causal = is_causal;
    params.is_bf16 = true;
    params.q_ptr = q8.data_ptr(); params.k_ptr = k8.data_ptr(); params.v_ptr = v8.data_ptr();
    params.do_ptr = dout8.data_ptr();
    params.q_row_stride = q8.stride(1); params.q_head_stride = q8.stride(2); params.q_batch_stride = q8.stride(0);
    params.k_row_stride = k8.stride(1); params.k_head_stride = k8.stride(2); params.k_batch_stride = k8.stride(0);
    params.v_row_stride = v8.stride(1); params.v_head_stride = v8.stride(2); params.v_batch_stride = v8.stride(0);
    params.do_row_stride = dout8.stride(1); params.do_head_stride = dout8.stride(2); params.do_batch_stride = dout8.stride(0);
    params.dq_ptr = dq.data_ptr(); params.dk_ptr = dk.data_ptr(); params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(1); params.dq_head_stride = dq.stride(2); params.dq_batch_stride = dq.stride(0);
    params.dk_row_stride = dk.stride(1); params.dk_head_stride = dk.stride(2); params.dk_batch_stride = dk.stride(0);
    params.dv_row_stride = dv.stride(1); params.dv_head_stride = dv.stride(2); params.dv_batch_stride = dv.stride(0);
    params.dq_accum_ptr = dq_accum.data_ptr();
    params.softmax_lse_ptr = lse.data_ptr();
    params.softmax_lse_log2_ptr = lse_log2.data_ptr();
    params.dsoftmax_sum = delta.data_ptr();
    params.tile_count_semaphore = tile_count_sem.data_ptr<int>();
    params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    params.arch = 90;
    // ## FP11-M17 GT: the Q^T / dO^T gmem twins are REAL inputs (pi-ordered s positions).
    TORCH_CHECK(qt8.defined() && dot8.defined(), "delta_matching bwd: the qt8 / dot8 twins are required");
    TORCH_CHECK(qt8.size(0) == batch_size && qt8.size(1) == num_heads && qt8.size(2) == head_size && qt8.size(3) == seqlen_q && qt8.is_contiguous(),
                "delta_matching bwd: qt8 must be (b, h, d, s_q) contiguous");
    TORCH_CHECK(dot8.size(0) == batch_size && dot8.size(1) == num_heads && dot8.size(2) == head_size && dot8.size(3) == seqlen_q && dot8.is_contiguous(),
                "delta_matching bwd: dot8 must be (b, h, d, s_q) contiguous");
    params.qt_ptr = qt8.data_ptr(); params.kt_ptr = kt8.data_ptr(); params.dot_ptr = dot8.data_ptr();
    params.q8s_ptr = q8s.data_ptr<float>(); params.k8s_ptr = k8s.data_ptr<float>();
    params.v8s_ptr = v8s.data_ptr<float>(); params.do8s_ptr = do8s.data_ptr<float>();
    params.nblk_q = q8s.size(2);
    params.nblk_k = k8s.size(2);
    params.ds8_ptr = ds8.data_ptr();
    params.psi_ptr = psi.data_ptr<float>();
#ifndef FA3FP11_HD128_P1X
#define FA3FP11_HD128_P1X 0
#endif
    // ## bf16-dP: the arm's numerics are decided by the BINARY, so a mismatch between the build flag and
    // the call is a hard error in both directions (a conflict is never silently resolved).
#if FA3FP11_P1R_SAGEDP
    TORCH_CHECK(vraw.defined() && doraw.defined(),
                "delta_matching bwd: this library is a bf16-dP build (DM_BF16_DP=1); call bwd_fp8_bf16dp with the raw bf16 v / dO");
    TORCH_CHECK(vraw.scalar_type() == at::kBFloat16 && doraw.scalar_type() == at::kBFloat16,
                "delta_matching bwd (bf16-dP): vraw/doraw must be bf16");
    TORCH_CHECK(vraw.sizes() == v8.sizes(), "delta_matching bwd (bf16-dP): vraw must have v8's shape (b, s_k, h_k, d)");
    TORCH_CHECK(doraw.sizes() == dout8.sizes(), "delta_matching bwd (bf16-dP): doraw must have dout8's shape (b, s_q, h, d)");
    TORCH_CHECK(vraw.stride(-1) == 1 && doraw.stride(-1) == 1, "delta_matching bwd (bf16-dP): vraw/doraw need a contiguous last dim");
    TORCH_CHECK(head_size == 128, "delta_matching bwd (bf16-dP): head_dim 128 only");
    params.vraw_ptr = vraw.data_ptr();
    params.doraw_ptr = doraw.data_ptr();
    params.vraw_row_stride = vraw.stride(1); params.vraw_head_stride = vraw.stride(2); params.vraw_batch_stride = vraw.stride(0);
    params.doraw_row_stride = doraw.stride(1); params.doraw_head_stride = doraw.stride(2); params.doraw_batch_stride = doraw.stride(0);
#else
    TORCH_CHECK(!vraw.defined() && !doraw.defined(),
                "delta_matching bwd: raw bf16 v / dO passed to a library built without DM_BF16_DP=1; use bwd_fp8_t with this library");
#endif
    // ## FP11-M5: DBG-only workspaces (nullptr from the shipped op).
    params.dbg_rows_ptr = dbg_rows_ptr;
    params.dbg_rowsabs_ptr = dbg_rowsabs_ptr;
    params.dbg_cks_ptr = dbg_cks_ptr;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (head_size == 128) {
        if (is_causal) { run_flash_bwd_fp11_d128<true>(params, stream); }   // ## Delta-Matching-128
        else { run_flash_bwd_fp11_d128<false>(params, stream); }
    } else {
        if (is_causal) { run_flash_bwd_fp11_d256<true>(params, stream); }
        else { run_flash_bwd_fp11_d256<false>(params, stream); }
    }
    return {dq, dk, dv};
}

// ## FP11-M17 GT: twins from codes, in-.so (legacy-schema callers only; the
// fast path takes pre-made twins through bwd_fp8_t).  x8 (b, s, h, d) e4m3
// (any strides with d contiguous) -> (b, h, d, s) contiguous, s in pi order.
at::Tensor fp11_twin_pi(at::Tensor x8) {
    TORCH_CHECK(x8.is_cuda(), "delta_matching twin_pi: x8 must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(x8.device());
    TORCH_CHECK(x8.dim() == 4 && x8.stride(3) == 1 && x8.dtype() == at::kFloat8_e4m3fn, "delta_matching twin_pi: (b, s, h, d) e4m3, d contiguous");
    TORCH_CHECK(int64_t(x8.size(0)) * x8.size(2) <= 65535, "delta_matching twin_pi: batch * heads must be at most 65535");
    int const b = x8.size(0), s = x8.size(1), h = x8.size(2), d = x8.size(3);
    TORCH_CHECK(s % 64 == 0 && d % 64 == 0, "delta_matching twin_pi: s and d must be 64-multiples");
    at::Tensor out = torch::empty({b, h, d, s}, x8.options());
    fp11_make_twin_pi(x8.data_ptr(), out.data_ptr(), b, s, h, d,
                      x8.stride(0), x8.stride(1), x8.stride(2),
                      at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> bwd_fp8(
    at::Tensor dout8, at::Tensor q8, at::Tensor k8, at::Tensor v8,
    at::Tensor lse, at::Tensor delta, at::Tensor kt8,
    at::Tensor q8s, at::Tensor k8s, at::Tensor v8s, at::Tensor do8s,
    double softmax_scale, bool is_causal
) {
    // Legacy schema on the GT kernel: make the twins here (bitwise-neutral: the
    // kernel reads identical smem bytes), so callers of this schema run unchanged.
    at::Tensor qt8 = fp11_twin_pi(q8);
    at::Tensor dot8 = fp11_twin_pi(dout8);
    return bwd_fp8_core(dout8, q8, k8, v8, lse, delta, kt8, qt8, dot8, q8s, k8s, v8s,
                        do8s, softmax_scale, is_causal,
                        nullptr, nullptr, nullptr);
}

// ## FP11-M17 GT: the core op on pre-made twins (the L0 accounting op; the
// twins are quantizer outputs like kt8).  At GT=0 the twins are ignored.
std::tuple<at::Tensor, at::Tensor, at::Tensor> bwd_fp8_t(
    at::Tensor dout8, at::Tensor q8, at::Tensor k8, at::Tensor v8,
    at::Tensor lse, at::Tensor delta, at::Tensor kt8, at::Tensor qt8, at::Tensor dot8,
    at::Tensor q8s, at::Tensor k8s, at::Tensor v8s, at::Tensor do8s,
    double softmax_scale, bool is_causal
) {
    return bwd_fp8_core(dout8, q8, k8, v8, lse, delta, kt8, qt8, dot8, q8s, k8s, v8s,
                        do8s, softmax_scale, is_causal,
                        nullptr, nullptr, nullptr);
}

// ## bf16-dP: bwd_fp8_t plus the RAW bf16 V and dO, i.e. SageBwd's single exemption among the
// seven attention matmuls (dP = dO V^T unquantized; S, dV, dQ, dK and the stale delta unchanged).  A SEPARATE op
// so bwd_fp8_t's schema stays unchanged, exactly as delta_fp8 was split out.
std::tuple<at::Tensor, at::Tensor, at::Tensor> bwd_fp8_bf16dp(
    at::Tensor dout8, at::Tensor q8, at::Tensor k8, at::Tensor v8,
    at::Tensor lse, at::Tensor delta, at::Tensor kt8, at::Tensor qt8, at::Tensor dot8,
    at::Tensor q8s, at::Tensor k8s, at::Tensor v8s, at::Tensor do8s,
    at::Tensor vraw, at::Tensor doraw,
    double softmax_scale, bool is_causal
) {
    return bwd_fp8_core(dout8, q8, k8, v8, lse, delta, kt8, qt8, dot8, q8s, k8s, v8s,
                        do8s, softmax_scale, is_causal,
                        nullptr, nullptr, nullptr, vraw, doraw);
}

int64_t fp11_build_gt() { return FA3FP11_M17_GT; }
int64_t fp11_build_bf16dp() { return FA3FP11_P1R_SAGEDP; }   // ## bf16-dP: 1 only in a SageBwd-exemption build
int64_t fp11_build_hd128() { return FA3FP11_HD128; }   // ## Delta-Matching-128: 1 when the hd128 instantiations are compiled in

// ## FP11-M5 (in-kernel matched delta, the design notes): PASS 0 + PASS 0.5.
// delta[b,h,i] = sum_j P~(i,j) * dP~(i,j)*do8sc*v8sc, computed from the
// BACKWARD's own realized fp32 P~ (exp2 recompute, F1-clamped) and raw dP~
// wgmma fragments, bit-matching the pass-1 realization (mma_pass0 header in
// the mainloop).  Output units match what bwd_fp8's dS line consumes: delta
// rides in TRUE dP units (natural-log LSE was already folded into P~; the
// per-32-key k8s is applied by pass 1 AT USE, so it must NOT be in delta),
// exactly the units of the dO producer's stale rowsum(dO*O) and of the Triton
// reference delta.  A SEPARATE op (not a bwd_fp8 flag) so bwd_fp8's schema
// stays unchanged: callers drive bwd_fp8 positionally, and the wrapper already
// composes the delta producer in front of bwd_fp8: this op slots into that same
// composition point.
at::Tensor delta_fp8_core(
    at::Tensor dout8,   // (b, s_q, h, d) e4m3 codes
    at::Tensor q8,      // (b, s_q, h, d) e4m3
    at::Tensor k8,      // (b, s_k, h, d) e4m3 (smooth-K centered)
    at::Tensor v8,      // (b, s_k, h, d) e4m3
    at::Tensor lse,     // (b, h, s_q) fp32 (natural-log LSE)
    at::Tensor kt8,     // (b, h, d, s_k) e4m3 transposed K twin (loaded, unused)
    at::Tensor q8s,     // (b, h, s_q/64) fp32
    at::Tensor k8s,     // (b, h, s_k/32) fp32
    at::Tensor v8s,     // (b, h, s_k/32) fp32
    at::Tensor do8s,    // (b, h, s_q/64) fp32
    bool is_causal,
    unsigned int* dbg_cks_ptr
) {
    c10::cuda::CUDAGuard device_guard(q8.device());
    dm_check_inputs("delta", q8, {dout8, k8, v8, lse, kt8, q8s, k8s, v8s, do8s});
    dm_check_codes("delta", {dout8, q8, k8, v8, kt8});
    dm_check_scales("delta", {q8s, k8s, v8s, do8s});
    auto const sizes = q8.sizes();
    int const batch_size = sizes[0], seqlen_q = sizes[1], num_heads = sizes[2], head_size = sizes[3];
    int const seqlen_k = k8.size(1);
    // ## FP11-M16 (native GQA, READ side): as bwd_fp8.
    int const num_heads_k = k8.size(2);
    TORCH_CHECK(batch_size > 0 && seqlen_q > 0 && seqlen_k > 0 && num_heads > 0 && num_heads_k > 0, "delta_matching delta: empty input");
    TORCH_CHECK(k8.sizes() == v8.sizes() && k8.size(0) == batch_size && k8.size(3) == head_size,
                "delta_matching delta: k8 and v8 must be (b, s_k, h_k, d) with q8's batch and head_dim");
    TORCH_CHECK(!is_causal || seqlen_q == seqlen_k, "delta_matching delta: causal attention needs seqlen_q == seqlen_k (got ",
                seqlen_q, ", ", seqlen_k, ")");
    TORCH_CHECK(dout8.sizes() == q8.sizes(), "delta_matching delta: dout8 must have q8's shape");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.numel() == int64_t(batch_size) * num_heads * seqlen_q,
                "delta_matching delta: lse must be (b, h, s_q) fp32");
    TORCH_CHECK(int64_t(batch_size) * num_heads <= 65535, "delta_matching delta: batch * heads must be at most 65535");
    TORCH_CHECK(v8.size(2) == num_heads_k && num_heads % num_heads_k == 0,
                "delta_matching delta: k8/v8 heads must divide q8 heads (native GQA)");
    TORCH_CHECK(kt8.size(0) == batch_size && kt8.size(1) == num_heads_k && kt8.size(2) == head_size && kt8.size(3) == seqlen_k,
                "delta_matching delta: kt8 must be (b, h_k, d, s_k)");
    TORCH_CHECK(k8s.size(1) == num_heads_k && v8s.size(1) == num_heads_k && q8s.size(1) == num_heads && do8s.size(1) == num_heads,
                "delta_matching delta: scale head counts (q/do: h, k/v: h_k)");
    TORCH_CHECK(fp11_hd_supported(head_size), "delta_matching delta: head_dim must be 256 (or 128 in a Delta-Matching-128 build)");
    int const kbn_d = fp11_bwd_block_n(head_size);   // the pass-1 key block: pass 0 walks the same tiles
    TORCH_CHECK(seqlen_q % 64 == 0 && seqlen_k % kbn_d == 0, "delta_matching delta: seqlen_q must be a 64-multiple, seqlen_k a ",
                kbn_d, "-multiple");
    {
        int const rq = fp11_quant_rows_q(head_size), rk = fp11_quant_keys(head_size);
        TORCH_CHECK(seqlen_q % rq == 0 && q8s.size(2) == seqlen_q / rq && do8s.size(2) == seqlen_q / rq, "delta_matching delta: q8s/do8s must be (b, h, s_q/", rq, ")");
        TORCH_CHECK(k8s.size(2) == seqlen_k / rk && v8s.size(2) == seqlen_k / rk, "delta_matching delta: k8s/v8s must be (b, h_k, s_k/", rk, ")");
    }
    for (auto const& t : {dout8, q8, k8, v8}) { TORCH_CHECK(t.stride(-1) == 1); }
    for (auto const& t : {kt8, q8s, k8s, v8s, do8s, lse}) { TORCH_CHECK(t.is_contiguous()); }
    auto opts = q8.options();
    int const seqlen_q_rounded = seqlen_q;    // 64-multiple asserted
    int const head_size_rounded = head_size;
    int const nbm = seqlen_q / 64, nbn = seqlen_k / 64;

    at::Tensor dlt = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    // Pass-0 -> pass-0.5 workspace: per-CTA per-row per-warpgroup-half
    // partials [BH, nbn, 2, N] fp32 (~128 MB at B2 H8 N8192), transient like
    // the dS' workspace; causal leaves the j > mb tiles unwritten AND unread
    // (no zeroing needed).  The 2-half split (vs D3's [BH, nbn, N]) trades
    // 64 MB of transient workspace for the per-tile cross-warpgroup named
    // barrier (fp11_delta.h header).
    // Only the head_dim-256 pass 0 writes these partials; the head_dim-128 pass 0 folds in registers (it reads and
    // writes nothing but the delta), so it gets a one-element placeholder instead of B*H*N^2/8 bytes.
    at::Tensor part = head_size == 256 ? torch::empty({int64_t(batch_size) * num_heads, nbn, 2, seqlen_q}, opts.dtype(at::kFloat))
                                       : torch::empty({1}, opts.dtype(at::kFloat));
    // Same lse*log2e prep as bwd_fp8 (identical realized fp32 values, so the
    // pass-0 exp2 consumes the very bytes pass 1 will).
    at::Tensor lse_log2 = lse;   // ## FP11-M17 LSEK: in-register multiply in pass 0 (bitwise)
    // The SingleTile schedulers never dereference the semaphore (static tile
    // maps): empty, not zeros, saves the memset launch on the matched path.
    at::Tensor tile_count_sem = torch::empty({1}, opts.dtype(at::kInt));
    at::Tensor dq_accum = torch::empty({4}, opts.dtype(at::kFloat));

    Flash_bwd_fp8_params params{};
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;   // ## FP11-M16: native GQA
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k;
    params.d = head_size;
    params.dv = head_size;
    params.d_rounded = head_size_rounded;
    params.dv_rounded = head_size_rounded;
    params.scale_softmax = 1.0f;   // unused by pass 0 (sm is in the Q codes)
    params.is_causal = is_causal;
    params.is_bf16 = true;
    params.q_ptr = q8.data_ptr(); params.k_ptr = k8.data_ptr(); params.v_ptr = v8.data_ptr();
    params.do_ptr = dout8.data_ptr();
    params.q_row_stride = q8.stride(1); params.q_head_stride = q8.stride(2); params.q_batch_stride = q8.stride(0);
    params.k_row_stride = k8.stride(1); params.k_head_stride = k8.stride(2); params.k_batch_stride = k8.stride(0);
    params.v_row_stride = v8.stride(1); params.v_head_stride = v8.stride(2); params.v_batch_stride = v8.stride(0);
    params.do_row_stride = dout8.stride(1); params.do_head_stride = dout8.stride(2); params.do_batch_stride = dout8.stride(0);
    params.dq_accum_ptr = dq_accum.data_ptr();
    params.softmax_lse_ptr = lse.data_ptr();
    params.softmax_lse_log2_ptr = lse_log2.data_ptr();
    // Dead-value slot: the producer bulk-copies dPsum alongside dO exactly as
    // in pass 1 (identical pipeline transaction bytes); pass 0 never reads
    // it -- it is computing delta.  Any valid (b,h,s) fp32 works.
    params.dsoftmax_sum = lse_log2.data_ptr();
    params.tile_count_semaphore = tile_count_sem.data_ptr<int>();
    params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    params.arch = 90;
    // Dead twin slots exactly as bwd_fp8 (qt/dot never dereferenced); the K
    // twin IS loaded (one-shot 16 KB per CTA, unconsumed) to keep load()'s
    // instruction stream and barrier byte counts pass-1-identical.
    params.qt_ptr = kt8.data_ptr(); params.kt_ptr = kt8.data_ptr(); params.dot_ptr = kt8.data_ptr();
    params.q8s_ptr = q8s.data_ptr<float>(); params.k8s_ptr = k8s.data_ptr<float>();
    params.v8s_ptr = v8s.data_ptr<float>(); params.do8s_ptr = do8s.data_ptr<float>();
    params.nblk_q = q8s.size(2);
    params.nblk_k = k8s.size(2);
    // Descriptor-only dummies: valid device pointers for host-side TMA
    // construction/prefetch; pass 0 issues no dS8 store and no psi publish.
    params.ds8_ptr = part.data_ptr();
    params.psi_ptr = part.data_ptr<float>();
    params.dlt_part_ptr = part.data_ptr<float>();
    params.dlt_ptr = dlt.data_ptr<float>();
    params.dbg_cks_ptr = dbg_cks_ptr;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (head_size == 128) {
        if (is_causal) { run_flash_delta_fp11_d128<true>(params, stream); }   // ## Delta-Matching-128
        else { run_flash_delta_fp11_d128<false>(params, stream); }
    } else {
        if (is_causal) { run_flash_delta_fp11_d256<true>(params, stream); }
        else { run_flash_delta_fp11_d256<false>(params, stream); }
    }
    return dlt;
}

at::Tensor delta_fp8(
    at::Tensor dout8, at::Tensor q8, at::Tensor k8, at::Tensor v8,
    at::Tensor lse, at::Tensor kt8,
    at::Tensor q8s, at::Tensor k8s, at::Tensor v8s, at::Tensor do8s,
    bool is_causal
) {
    return delta_fp8_core(dout8, q8, k8, v8, lse, kt8, q8s, k8s, v8s, do8s,
                          is_causal, nullptr);
}

#ifdef FA3FP11_DBG
// ## FP11-M5 DBG ops (compiled ONLY into the FA3FP11_DBG=1 build; the
// shipped binary has neither the ops nor any kernel code touching the
// workspaces).  bwd_fp8_dbg additionally returns
//   rows    [BH, nbn, 2, s_q] fp32: per-(n-tile, warpgroup) partial row sums
//           of the realized dS in true (un-k8s-folded) units,
//   rowsabs [BH, nbn, 2, s_q] fp32: same partials of |dS|,
//   cks     [2, BH, nbm, nbn] int32: XOR of the P~ (plane 0) / raw dP~
//           (plane 1) fragment bits per tile (zero-initialized; untouched
//           tiles stay 0).
// delta_fp8_dbg returns (delta, cks); the shipped pass-0 kernels write no checksums, so cks stays zero.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> bwd_fp8_dbg(
    at::Tensor dout8, at::Tensor q8, at::Tensor k8, at::Tensor v8,
    at::Tensor lse, at::Tensor delta, at::Tensor kt8,
    at::Tensor q8s, at::Tensor k8s, at::Tensor v8s, at::Tensor do8s,
    double softmax_scale, bool is_causal
) {
    int const b = q8.size(0), sq = q8.size(1), h = q8.size(2);
    int const sk = k8.size(1);
    int const nbm = sq / 64, nbn = sk / fp11_bwd_block_n(q8.size(3));   // pass 1's key block: 64 at d256, 128 at hd128
    auto fopts = q8.options().dtype(at::kFloat);
    // zeros (not empty): causal leaves nb > mb tiles unwritten, and zeros are
    // additive-neutral for the test's fixed-order (nb, wg) reduction.
    at::Tensor rows = torch::zeros({int64_t(b) * h, nbn, 2, sq}, fopts);
    at::Tensor rowsabs = torch::zeros({int64_t(b) * h, nbn, 2, sq}, fopts);
    at::Tensor cks = torch::zeros({2, int64_t(b) * h, nbm, nbn}, q8.options().dtype(at::kInt));
    at::Tensor qt8 = fp11_twin_pi(q8);
    at::Tensor dot8 = fp11_twin_pi(dout8);
    auto [dq, dk, dv] = bwd_fp8_core(
        dout8, q8, k8, v8, lse, delta, kt8, qt8, dot8, q8s, k8s, v8s, do8s,
        softmax_scale, is_causal,
        rows.data_ptr<float>(), rowsabs.data_ptr<float>(),
        reinterpret_cast<unsigned int*>(cks.data_ptr<int>()));
    return {dq, dk, dv, rows, rowsabs, cks};
}

std::tuple<at::Tensor, at::Tensor> delta_fp8_dbg(
    at::Tensor dout8, at::Tensor q8, at::Tensor k8, at::Tensor v8,
    at::Tensor lse, at::Tensor kt8,
    at::Tensor q8s, at::Tensor k8s, at::Tensor v8s, at::Tensor do8s,
    bool is_causal
) {
    int const b = q8.size(0), sq = q8.size(1), h = q8.size(2);
    int const sk = k8.size(1);
    int const nbm = sq / 64, nbn = sk / 64;
    at::Tensor cks = torch::zeros({2, int64_t(b) * h, nbm, nbn}, q8.options().dtype(at::kInt));
    at::Tensor dlt = delta_fp8_core(
        dout8, q8, k8, v8, lse, kt8, q8s, k8s, v8s, do8s, is_causal,
        reinterpret_cast<unsigned int*>(cks.data_ptr<int>()));
    return {dlt, cks};
}

// ## DBG: bwd_fp8_dbg plus the realized dS' of the SAME launch: ds8 [BH, nbm, nbn, 64, kbn] e4m3 codes
// (kbn = 128 at hd128, 64 at d256; natural rows x keys per tile; under causal only the tiles with nb * kbn <= 64 * mb + 63
// are written, the rest is uninitialized memory) and psi [BH, nbm, nbn, 1] fp32 (0 marks a zero tile).  True units:
// dS'(i, j) = code * psi / (2^8 * k8s(j)) (pass 1 stores e4m3(2^8 * P * (dP - delta) * k8s / psi)).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> bwd_fp8_dbg_ds(
    at::Tensor dout8, at::Tensor q8, at::Tensor k8, at::Tensor v8,
    at::Tensor lse, at::Tensor delta, at::Tensor kt8,
    at::Tensor q8s, at::Tensor k8s, at::Tensor v8s, at::Tensor do8s,
    double softmax_scale, bool is_causal
) {
    at::Tensor ds8, psi;
    g_dbg_ds8 = &ds8; g_dbg_psi = &psi;
    auto r = bwd_fp8_dbg(dout8, q8, k8, v8, lse, delta, kt8, q8s, k8s, v8s, do8s, softmax_scale, is_causal);
    g_dbg_ds8 = nullptr; g_dbg_psi = nullptr;
    return {std::get<0>(r), std::get<1>(r), std::get<2>(r), std::get<3>(r), std::get<4>(r), std::get<5>(r), ds8, psi};
}
#endif  // FA3FP11_DBG

}  // namespace

TORCH_LIBRARY(delta_matching, m) {
    m.def("fwd_fp8(Tensor q8, Tensor k8, Tensor v8, Tensor q8s, Tensor k8s, Tensor v8s,"
          " float softmax_scale, bool is_causal) -> (Tensor, Tensor)");
    m.def("bwd_fp8(Tensor dout8, Tensor q8, Tensor k8, Tensor v8, Tensor lse, Tensor delta,"
          " Tensor kt8, Tensor q8s, Tensor k8s, Tensor v8s, Tensor do8s,"
          " float softmax_scale, bool is_causal) -> (Tensor, Tensor, Tensor)");
    m.def("delta_fp8(Tensor dout8, Tensor q8, Tensor k8, Tensor v8, Tensor lse,"
          " Tensor kt8, Tensor q8s, Tensor k8s, Tensor v8s, Tensor do8s,"
          " bool is_causal) -> Tensor");
    // ## FP11-M17 GT
    m.def("bwd_fp8_t(Tensor dout8, Tensor q8, Tensor k8, Tensor v8, Tensor lse, Tensor delta,"
          " Tensor kt8, Tensor qt8, Tensor dot8, Tensor q8s, Tensor k8s, Tensor v8s, Tensor do8s,"
          " float softmax_scale, bool is_causal) -> (Tensor, Tensor, Tensor)");
    m.def("bwd_fp8_bf16dp(Tensor dout8, Tensor q8, Tensor k8, Tensor v8, Tensor lse, Tensor delta,"
          " Tensor kt8, Tensor qt8, Tensor dot8, Tensor q8s, Tensor k8s, Tensor v8s, Tensor do8s,"
          " Tensor vraw, Tensor doraw, float softmax_scale, bool is_causal)"
          " -> (Tensor, Tensor, Tensor)");
    m.def("build_bf16dp() -> int");
    m.def("twin_pi(Tensor x8) -> Tensor");
    m.def("build_gt() -> int");
    m.def("build_hd128() -> int");
#ifdef FA3FP11_DBG
    m.def("bwd_fp8_dbg(Tensor dout8, Tensor q8, Tensor k8, Tensor v8, Tensor lse, Tensor delta,"
          " Tensor kt8, Tensor q8s, Tensor k8s, Tensor v8s, Tensor do8s,"
          " float softmax_scale, bool is_causal) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("delta_fp8_dbg(Tensor dout8, Tensor q8, Tensor k8, Tensor v8, Tensor lse,"
          " Tensor kt8, Tensor q8s, Tensor k8s, Tensor v8s, Tensor do8s,"
          " bool is_causal) -> (Tensor, Tensor)");
    m.def("bwd_fp8_dbg_ds(Tensor dout8, Tensor q8, Tensor k8, Tensor v8, Tensor lse, Tensor delta,"
          " Tensor kt8, Tensor q8s, Tensor k8s, Tensor v8s, Tensor do8s,"
          " float softmax_scale, bool is_causal) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
#endif
}

TORCH_LIBRARY_IMPL(delta_matching, CompositeExplicitAutograd, m) {
    m.impl("build_gt", &fp11_build_gt);
    m.impl("build_bf16dp", &fp11_build_bf16dp);
    m.impl("build_hd128", &fp11_build_hd128);
}

TORCH_LIBRARY_IMPL(delta_matching, CUDA, m) {
    m.impl("fwd_fp8", &fwd_fp8);
    m.impl("bwd_fp8", &bwd_fp8);
    m.impl("delta_fp8", &delta_fp8);
    m.impl("bwd_fp8_t", &bwd_fp8_t);
    m.impl("bwd_fp8_bf16dp", &bwd_fp8_bf16dp);
    m.impl("twin_pi", &fp11_twin_pi);
#ifdef FA3FP11_DBG
    m.impl("bwd_fp8_dbg", &bwd_fp8_dbg);
    m.impl("delta_fp8_dbg", &delta_fp8_dbg);
    m.impl("bwd_fp8_dbg_ds", &bwd_fp8_dbg_ds);
#endif
}
