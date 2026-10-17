"""JIT build of `delta_matching._C` (the alternative to `pip install .`): `python -m delta_matching.jit` in a source
checkout (the build reads ../csrc and third_party/).  Compiles the same sources with the same flags as setup.py into
DM_JIT_DIR (default delta_matching/_jit) and links delta_matching/_C.so to the result, so `import delta_matching` finds
it.  Env: DM_FA_SRC (the flash-attention checkout, default third_party/flash-attention), MAX_JOBS (parallel nvcc jobs),
DM_DBG=1 (the instrumented variant: row-sum residual and checksum ops; the shipped numerics are unchanged), DM_BF16_DP=1
(the bf16-dP variant, a DIFFERENT numerics arm), DM_LINEINFO=1 (nvcc line tables for profilers; they embed the absolute
source paths of this machine in the library, so they are off by default).  Remove a pip-built `_C.cpython-*.so` from the package directory first:
the loader refuses two libraries."""
import os, sys

import torch  # noqa: F401
from torch.utils.cpp_extension import load

from ._buildcfg import sources, include_dirs, defines, cxx_flags, cuda_flags, write_build_info, PACKAGE_DIR


def main():
    build_dir = os.path.abspath(os.environ.get("DM_JIT_DIR", os.path.join(PACKAGE_DIR, "_jit")))
    os.makedirs(build_dir, exist_ok=True)
    load(name="_C", sources=sources(), extra_include_paths=include_dirs(),
         extra_cflags=cxx_flags() + defines(), extra_cuda_cflags=cuda_flags() + defines(),
         build_directory=build_dir, verbose=True, is_python_module=False)
    so = os.path.join(build_dir, "_C.so")
    if not os.path.isfile(so):
        print(f"JIT build produced no {so}", file=sys.stderr); sys.exit(2)
    dst = os.path.join(PACKAGE_DIR, "_C.so")
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(so, PACKAGE_DIR), dst)
    write_build_info()          # stamped after a successful build only
    print("BUILD-OK", dst, "->", so)


if __name__ == "__main__":
    main()
