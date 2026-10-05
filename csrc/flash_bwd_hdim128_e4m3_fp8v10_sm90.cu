// ## Delta-Matching-128 (M19): instantiation TU for the SPLIT backward at head_dim 128
// (pass 1 = column-parallel dK/dV at kBlockM 64 x kBlockN 128, gmem twins; pass 2 = fq8_dq_pass2_hd128.h).
#include "dm_flagset.h"   // stops a build outside the release flag set
#include "flash_bwd_fp8v10_launch.h"

template void run_flash_bwd_fp11_d128<true>(Flash_bwd_fp8_params &params, cudaStream_t stream);
template void run_flash_bwd_fp11_d128<false>(Flash_bwd_fp8_params &params, cudaStream_t stream);
