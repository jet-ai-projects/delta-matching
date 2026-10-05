"""Shared Triton constants of the fp8 producers (e4m3 range and the guarded scale floor)."""
import numpy as _np
import triton.language as tl

_TL_E4M3_MAX = tl.constexpr(448.0)
_TL_TINY = tl.constexpr(1e-30)
_TL_INV_E4M3_MAX = tl.constexpr(float(_np.float32(1.0) / _np.float32(448.0)))
E4M3_MAX = 448.0
