"""
Architecture constants of the current CUDA device, probed once at import.
"""
import torch

__all__ = ["ARCH_MAJOR"]

# 0 without a CUDA device (CPU-only compiles call the launchers directly)
ARCH_MAJOR = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
