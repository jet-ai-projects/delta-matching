// ## FQ8 (fa3fp8): params extension shared between the host api and the CUDA TU.
#pragma once
#include "flash.h"

struct Flash_bwd_fp8_params : public Flash_bwd_params {
    void const* qt_ptr;
    void const* kt_ptr;
    void const* dot_ptr;
    float const* q8s_ptr;
    float const* k8s_ptr;
    float const* v8s_ptr;
    float const* do8s_ptr;
    int nblk_q;
    int nblk_k;
    // ## V10 (split bwd): pass-1 outputs for the pass-2 dQ GEMM.  dS' e4m3
    // codes, tile-contiguous [BH, nbm, nbn, 64, 64], plus the per-tile psi
    // scale [BH, nbm, nbn].  nullptr for every other op (zero-init struct).
    void* ds8_ptr;
    float* psi_ptr;
    // ## FP11-M5 (in-kernel matched delta): pass-0 partial workspace
    // [BH, nbn, 2, N] fp32 (per warpgroup column half) and the reduced
    // delta [B, H, N] fp32 (delta_fp8 op
    // only; nullptr in bwd_fp8).  The dbg_* trio is set only by the
    // FA3FP11_DBG build's *_dbg ops (rows/rowsabs: [BH, nbn, 2, N] fp32
    // per-(tile, warpgroup) dS row-sum partials; cks: [2, BH, nbm, nbn] u32
    // XOR checksums of the P~ / dP~ fragment bits).  The shipped build
    // carries the fields but compiles no code that reads them.
    float* dlt_part_ptr;
    float* dlt_ptr;
    float* dbg_rows_ptr;
    float* dbg_rowsabs_ptr;
    unsigned int* dbg_cks_ptr;
    // ## bf16-dP: the RAW (pre-quantization) bf16 V and dO that SageBwd's exempt matmul
    // dP = dO V^T reads instead of the e4m3 codes.  Same logical layouts as v_ptr / do_ptr
    // ((b, s_k, h_k, d) and (b, s_q, h, d), last dim contiguous), so only the row/head/batch strides
    // differ by element size.  nullptr in every non-SAGE op; read only by the P1-R hd128 pass 1
    // compiled with FA3FP11_P1R_SAGEDP=1.
    void const* vraw_ptr;
    void const* doraw_ptr;
    int64_t vraw_row_stride, vraw_head_stride, vraw_batch_stride;
    int64_t doraw_row_stride, doraw_head_stride, doraw_batch_stride;
};

// ## FQ8: fwd params extension (per-block scales).
struct Flash_fwd_fp8_params : public Flash_fwd_params {
    float const* q8s_ptr;
    float const* k8s_ptr;
    float const* v8s_ptr;
    int nblk_q;
    int nblk_k;
};
