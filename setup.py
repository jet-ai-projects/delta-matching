import os, sys
from setuptools import setup

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "delta_matching"))
import _buildcfg as B  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B.write_build_info()   # delta_matching/_build_info.py: the csrc hash + flags of THIS build, for kernel_version() after install
B.clean_ldshared()     # no build-machine RPATH in the library (conda's LDSHARED / LDCXXSHARED would add one)
ext = CUDAExtension(
    name="delta_matching._C",
    sources=[os.path.relpath(s, HERE) for s in B.sources()],   # setuptools refuses absolute source paths in a wheel build
    include_dirs=B.include_dirs(),
    extra_compile_args={"cxx": B.cxx_flags() + B.defines(), "nvcc": B.cuda_flags() + B.defines()},
)
setup(ext_modules=[ext], cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)})
