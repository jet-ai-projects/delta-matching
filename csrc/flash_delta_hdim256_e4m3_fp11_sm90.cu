// ## FP11-M5: instantiation TU for the in-kernel matched-delta prepass
// (pass 0 = DeltaPass0 mainloop + pass 0.5 reduce; fp11_delta.h).
#include "dm_flagset.h"   // stops a build outside the release flag set
#include "fp11_delta.h"

template void run_flash_delta_fp11_d256<true>(Flash_bwd_fp8_params &params, cudaStream_t stream);
template void run_flash_delta_fp11_d256<false>(Flash_bwd_fp8_params &params, cudaStream_t stream);

// The head_dim-128 pass 0 is compiled in its own TU (flash_delta_hdim128_e4m3_fp11v5_sm90.cu).
