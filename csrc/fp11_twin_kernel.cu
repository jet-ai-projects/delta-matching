// ## FP11-M17 GT: twin maker for the legacy-schema bwd_fp8 op (and the DBG op):
// x8 (b, s, h, d) e4m3 with strides (sb, ss, sh, 1) -> xt8 (b, h, d, s) contiguous
// e4m3 whose s positions carry the pass-1 in-smem transposer's stmatrix
// interleave order pi (per 16-position window: position(m) = fq8_pi_m(m) =
// (m & ~15) | ((m & 6) << 1) | ((m & 8) >> 2) | (m & 1)), so a TMA-loaded twin
// tile is byte-identical to what the M16 transposer wrote into smem.  Codes
// only (block scales are untouched).  64 x 64 tiles through smem; coalesced
// 64-B row reads and 64-B row writes (the pi permutation stays inside each
// 16-byte group).  Not the fast path: the API route makes the twins with
// delta_matching/twins.py (twin_pi).
#include "dm_flagset.h"   // stops a build outside the release flag set
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ int fp11_pi_m(int m) {
    return (m & ~15) | ((m & 6) << 1) | ((m & 8) >> 2) | (m & 1);
}

__global__ void fp11_twin_pi_kernel(uint8_t const* __restrict__ x, uint8_t* __restrict__ out,
                                    int s, int h, int d, int64_t sb, int64_t ss, int64_t sh) {
    __shared__ uint8_t tile[64][65];
    int const bh = blockIdx.z;
    int const b = bh / h, hh = bh % h;
    int const s0 = blockIdx.x * 64, d0 = blockIdx.y * 64;
    int const tx = threadIdx.x;          // 0..63
    int const ty = threadIdx.y;          // 0..3
    uint8_t const* src = x + int64_t(b) * sb + int64_t(hh) * sh + d0 + tx;
    #pragma unroll
    for (int r = ty; r < 64; r += 4) {
        tile[r][tx] = src[int64_t(s0 + r) * ss];
    }
    __syncthreads();
    // out[(bh * d + d0 + r) * s + s0 + pi(tx)] = tile[tx][r]
    int const pos = s0 + fp11_pi_m(tx);
    uint8_t* dst = out + (int64_t(bh) * d + d0) * int64_t(s) + pos;
    #pragma unroll
    for (int r = ty; r < 64; r += 4) {
        dst[int64_t(r) * s] = tile[tx][r];
    }
}

// ## VT: the forward's V twin.  x8 (b, s, h, d) e4m3 (d contiguous, 16-byte aligned rows) -> (b, h, d, s) contiguous
// e4m3 holding, per 128-key tile, the bytes the forward's in-smem V transpose (LDSM.T + byte_perm + STSM) wrote:
//   out[b, h, dp, c] = x[b, (c & ~15) | sigma(c & 15), h, (dp & ~15) | tau(dp & 15)]
//   sigma(c) = (c & 1) | ((c & 2) << 2) | ((c & 12) >> 1)   (keys: the A-operand order of permute_Cregs_fp8's P)
//   tau(dp)  = ((dp & 7) << 1) | ((dp & 8) >> 3)            (head dim: the order permute_output_fp8 un-permutes)
// 64 keys x 64 dims per CTA through smem: 16-byte row reads, 16-byte row writes.
__global__ void __launch_bounds__(256) fp11_twin_vt_kernel(uint8_t const* __restrict__ x, uint8_t* __restrict__ out,
                                    int s, int h, int d, int64_t sb, int64_t ss, int64_t sh) {
    __shared__ __align__(16) uint8_t tile[64][80];
    int const bh = blockIdx.z;
    int const b = bh / h, hh = bh % h;
    int const s0 = blockIdx.x * 64, d0 = blockIdx.y * 64;
    int const t = threadIdx.x;
    {
        int const r = t >> 2, c = t & 3;
        uint4 v = *reinterpret_cast<uint4 const*>(x + int64_t(b) * sb + int64_t(hh) * sh + int64_t(s0 + r) * ss + d0 + 16 * c);
        *reinterpret_cast<uint4*>(&tile[r][16 * c]) = v;
    }
    __syncthreads();
    int const dp = t >> 2, c4 = t & 3;
    int const dcol = (dp & ~15) | ((dp & 7) << 1) | ((dp & 8) >> 3);
    uint32_t w[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        uint32_t acc = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int const c = 4 * q + j;
            int const m = (c & 1) | ((c & 2) << 2) | ((c & 12) >> 1);
            acc |= uint32_t(tile[16 * c4 + m][dcol]) << (8 * j);
        }
        w[q] = acc;
    }
    *reinterpret_cast<uint4*>(out + (int64_t(bh) * d + d0 + dp) * int64_t(s) + s0 + 16 * c4) = make_uint4(w[0], w[1], w[2], w[3]);
}

}  // namespace

void fp11_make_twin_vt(void const* x8, void* xt8, int b, int s, int h, int d,
                       int64_t sb, int64_t ss, int64_t sh, cudaStream_t stream) {
    dim3 grid(s / 64, d / 64, b * h);
    fp11_twin_vt_kernel<<<grid, 256, 0, stream>>>(
        static_cast<uint8_t const*>(x8), static_cast<uint8_t*>(xt8), s, h, d, sb, ss, sh);
}

void fp11_make_twin_pi(void const* x8, void* xt8, int b, int s, int h, int d,
                       int64_t sb, int64_t ss, int64_t sh, cudaStream_t stream) {
    dim3 grid(s / 64, d / 64, b * h);
    dim3 block(64, 4, 1);
    fp11_twin_pi_kernel<<<grid, block, 0, stream>>>(
        static_cast<uint8_t const*>(x8), static_cast<uint8_t*>(xt8), s, h, d, sb, ss, sh);
}
