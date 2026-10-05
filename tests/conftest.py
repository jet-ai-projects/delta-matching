"""Every test here needs a Hopper GPU (sm_90) and the built extension. Without a CUDA GPU, or on another GPU, the tests are
skipped with the reason; with a Hopper GPU but no usable extension the session stops with one message saying how to
build it (instead of one error per test)."""
import pytest
import torch


def _reason():
    if not torch.cuda.is_available():
        return "needs a CUDA GPU"
    if torch.cuda.get_device_capability(0) != (9, 0):
        return f"needs a Hopper GPU (sm_90); found {torch.cuda.get_device_name(0)}"
    return None


REASON = _reason()


def pytest_collection_modifyitems(config, items):
    if REASON:
        skip = pytest.mark.skip(reason=REASON)
        for item in items:
            item.add_marker(skip)
        return
    try:
        from delta_matching.core import _ops
        _ops()
    except Exception as e:  # noqa: BLE001: any failure to load the library means the tests cannot run
        pytest.exit(f"the delta_matching extension cannot be loaded ({type(e).__name__}: {e}); build it with "
                    "`pip install --no-build-isolation .` or `python -m delta_matching.jit` (README, Install)", returncode=2)
