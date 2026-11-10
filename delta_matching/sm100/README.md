# Delta-Matching on Blackwell (sm100)

CuTe DSL kernels for B200 / GB200 (sm_100a), forked from FlashAttention-4's sm100 kernels, with the sm100-mx numerics
(`delta_matching/reference.py`, preset `sm100-mx`, head_dim 128): Q / K / V / dO
as e4m3 codes with a power-of-2 scale per 32 tokens, P and dS as MXFP8 (e4m3 + UE8M0 block scales), block-scaled tcgen05
MMAs. Pure Python + CuTe DSL: nothing to build, kernels are JIT-compiled on first use into the CuTe DSL cache.

```python
from delta_matching.sm100.api import dm_attn_func
o = dm_attn_func(q, k, v, causal=True, mode="matched")   # q (B, N, H, 128) bf16, k / v (B, N, Hk, 128)
o.backward(do)
```
N must be a multiple of 128 and H a multiple of Hk; the softmax scale is D**-0.5. The backward is deterministic by
default (`deterministic=False` or `DM_DETERMINISTIC=0` selects the unordered accumulation; it is not the validated mode).

Requirements (as tested): CUDA 13.0, torch 2.14 (cu130), nvidia-cutlass-dsl 4.8.1, quack-kernels 0.6.5, apache-tvm-ffi,
FlashAttention-4's CuTe DSL package (`flash_attn.cute`) from Dao-AILab/flash-attention at b2edfab9
(`pip install -e flash_attn/cute`). The kernels use FA4's host plumbing; FA4 is not modified (the backward host code is
source-patched at import), so another FA4 revision may need a re-check.

Compile cache: set `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` and `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR`; entries are keyed
on the FA4 sources, every file of this package and the compile-time env knobs.

Tests: `pytest tests/test_sm100.py` (Blackwell GPU; the pure-torch emulation of the numerics is
`delta_matching/reference.py`).
