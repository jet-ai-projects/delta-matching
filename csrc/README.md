# csrc

The CUDA / C++ sources of the `delta_matching` torch library. Every translation unit below is compiled by `setup.py` (and by
`python -m delta_matching.jit`) with the certified flag set in `delta_matching/_buildcfg.py`:

| file | role |
|---|---|
| `delta_matching_api.cpp` | the torch ops (`fwd_fp8`, `bwd_fp8`, `bwd_fp8_t`, `delta_fp8`, `twin_pi`, build flags) and their input checks |
| `flash_fwd_hdim{128,256}_e4m3_fp8f_sm90.cu` | forward instantiations (`flash_fwd_fp8f_launch.h`, `flash_fwd_kernel_sm90_fq8f.h`, `mainloop_fwd_sm90_fp8f.hpp`, `softmax_fq8f.h`) |
| `flash_bwd_hdim{128,256}_e4m3_fp8v10_sm90.cu` | backward instantiations (`flash_bwd_fp8v10_launch.h`; pass 1: `mainloop_bwd_sm90_fp8v10.hpp` at 256, `mainloop_bwd_sm90_fp8r.hpp` at 128; pass 2: `fq8_dq_pass2*.h`) |
| `flash_delta_hdim256_e4m3_fp11_sm90.cu`, `flash_delta_hdim128_e4m3_fp11v5_sm90.cu` | the matched-delta pass 0 (`fp11_delta*.h`) |
| `fp11_twin_kernel.cu` | the transposed operand twins |
| `dm_flagset.h` | generated: stops a build whose flags differ from the release flag set, which these sources were reduced to |

Reading the comments: the sources keep the development history in short tags. `M<n>` is a development milestone, `FP11` /
`FQ8` / `V10` are internal names of the kernel family and its backward, `P1-R` is the head_dim-128 pass-1 collective, `F1`
(exp2 clamp), `F2` (zero-tile psi sentinel) and `F2b` (1e-8 dominance skip) are numerics guards, and "pass 0 / 1 / 2" are the
matched-delta prepass, the dK / dV pass and the dQ pass; `fa3fp8` / `fa3fp8f` name the earlier all-fp8 kernels this one
grew from, and "the design notes" / "the derivation notes" refer to development documents that are not part of this
repository. Macros named `FA3FP11_*` / `FA3FP8F_*` are the build switches;
their certified values are in `_buildcfg.py` and are part of the kernels' numerical identity. Several files are adapted from
FlashAttention-3 (`hopper/`) and keep its copyright notice (BSD 3-Clause, see `THIRD_PARTY_NOTICES.md`).
