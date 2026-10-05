// Instantiation TU of the head_dim-128 matched-delta pass 0 (the row-CTA prepass of fp11_delta_v6.h). Its own TU on
// purpose: no fp11_delta.h here (the pass-0.5 reduce kernel is a non-template __global__ with one definition per
// library, and the head_dim-128 pass 0 needs no reduce).
#include "dm_flagset.h"   // stops a build outside the release flag set
#include "fp11_delta_v6.h"

template <bool Is_causal>
void run_flash_delta_fp11_d128(Flash_bwd_fp8_params &params, cudaStream_t stream) {
    run_flash_delta_v6_fp11_hd128<Is_causal>(params, stream);
}
template void run_flash_delta_fp11_d128<true>(Flash_bwd_fp8_params &params, cudaStream_t stream);
template void run_flash_delta_fp11_d128<false>(Flash_bwd_fp8_params &params, cudaStream_t stream);
