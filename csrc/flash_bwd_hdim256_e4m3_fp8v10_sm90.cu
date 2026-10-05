// ## FQ8-V10: instantiation TU for the SPLIT backward pass 1 (dK/dV only; the
// entire dQ apparatus deleted; dQ moves to a row-parallel pass-2 GEMM).
#include "dm_flagset.h"   // stops a build outside the release flag set
#include "flash_bwd_fp8v10_launch.h"

// ## FP11 (M2): v11-specific symbol name (no weak-symbol overlap with
// other FA3-derived libraries; see the launch header).
template void run_flash_bwd_fp11_d256<true>(Flash_bwd_fp8_params &params, cudaStream_t stream);
template void run_flash_bwd_fp11_d256<false>(Flash_bwd_fp8_params &params, cudaStream_t stream);
