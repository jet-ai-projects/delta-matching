// ## FQ8F (delta_matching): instantiation TU for the FAITHFUL fp8 d256 forward.
#include "dm_flagset.h"   // stops a build outside the release flag set
#include "flash_fwd_fp8f_launch.h"

template void run_flash_fwd_fp8f_d256<true>(Flash_fwd_fp8_params &params, cudaStream_t stream);
template void run_flash_fwd_fp8f_d256<false>(Flash_fwd_fp8_params &params, cudaStream_t stream);
