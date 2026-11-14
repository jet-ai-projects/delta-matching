"""Two kinds of tests live here.

  sm90 kernel tests (test_dm_attn.py, test_api.py): the shipped Hopper kernel; they need a Hopper GPU (sm_90) and the
      built extension.  Elsewhere they are skipped with the reason; on a Hopper GPU without a usable extension the session
      stops with one message saying how to build it (instead of one error per test).
  sm100 kernel tests (test_sm100*.py): the Blackwell CuTe DSL kernels (delta_matching/sm100); they skip themselves off
      Blackwell or without nvidia-cutlass-dsl / flash_attn.cute.
"""
import pytest
import torch

SM90_MODULES = ("test_dm_attn.py", "test_api.py")


def _sm90_reason():
    if not torch.cuda.is_available():
        return "needs a CUDA GPU"
    if torch.cuda.get_device_capability(0) != (9, 0):
        return f"sm90 kernel test: needs a Hopper GPU (sm_90); found {torch.cuda.get_device_name(0)}"
    return None


REASON = _sm90_reason()


def pytest_collection_modifyitems(config, items):
    sm90_items = [it for it in items if it.fspath.basename in SM90_MODULES]
    if not sm90_items:
        return
    if REASON:
        skip = pytest.mark.skip(reason=REASON)
        for item in sm90_items:
            item.add_marker(skip)
        return
    try:
        from delta_matching.core import _ops
        _ops()
    except Exception as e:  # noqa: BLE001: any failure to load the library means the tests cannot run
        pytest.exit(f"the delta_matching extension cannot be loaded ({type(e).__name__}: {e}); build it with "
                    "`pip install --no-build-isolation .` or `python -m delta_matching.jit` (README, Install)", returncode=2)
