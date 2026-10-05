// ## Delta-Matching-128 (M19): instantiation TU for the FAITHFUL fp8 head_dim 128 forward
// (128 x 128 tiles, two 64-key P-hat groups per tile: the Triton reference's [128, 64] block contract).
#include "dm_flagset.h"   // stops a build outside the release flag set
#include "flash_fwd_fp8f_launch.h"

template void run_flash_fwd_fp8f_d128<true>(Flash_fwd_fp8_params &params, cudaStream_t stream);
template void run_flash_fwd_fp8f_d128<false>(Flash_fwd_fp8_params &params, cudaStream_t stream);
