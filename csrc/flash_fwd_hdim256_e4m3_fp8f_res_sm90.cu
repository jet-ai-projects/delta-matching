// ## ATTNQAT (delta_matching): instantiation TU for the d256 O' residual pass of the attnqat mode (the shipped d256
// forward with the residual P codes; flash_fwd_fp8f_launch.h run_flash_fwd_fp8f_d256_res).
#include "dm_flagset.h"   // stops a build outside the release flag set
#include "flash_fwd_fp8f_launch.h"

template void run_flash_fwd_fp8f_d256_res<true>(Flash_fwd_fp8_params &params, cudaStream_t stream);
template void run_flash_fwd_fp8f_d256_res<false>(Flash_fwd_fp8_params &params, cudaStream_t stream);
