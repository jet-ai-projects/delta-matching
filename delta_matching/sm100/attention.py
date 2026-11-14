"""
Delta-Matching attention on Blackwell: picks the kernels for the device
architecture and the head dimension.
"""
from delta_matching.sm100.arch import ARCH_MAJOR

__all__ = ["dm_attn_supported", "dm_attn_func", "dm_attn_debug"]


def dm_attn_supported(head_dim: int) -> bool:
    """
    True when this device runs the sm100 kernels at this head dimension.
    """
    return ARCH_MAJOR == 10 and head_dim in (128, 256)


def dm_attn_func(q, k, v, causal=True, mode="matched", deterministic=None):
    """
    Attention with the Delta-Matching backward: q (B, N, H, D) bf16, k / v (B, N, Hk, D) bf16 with H
    % Hk == 0 and D = 128 or 256 -> o (B, N, H, D) bf16; softmax scale D^-0.5, N a multiple of 128.
    mode "matched" forms the backward's delta from its own fp8 P / dP (pass 0); "stale" uses
    rowsum(dO * O). deterministic None reads DM_DETERMINISTIC (default 1); False selects the
    unordered dQ / dK / dV accumulation.
    """
    match ARCH_MAJOR:
        case 10:
            if q.shape[-1] == 256:
                from delta_matching.sm100.api_hd256 import dm_attn_func_hd256
                return dm_attn_func_hd256(q, k, v, causal, mode, deterministic)
            from delta_matching.sm100.attention_hd128_sm100 import attention_hd128
            return attention_hd128(q, k, v, causal, mode, deterministic)
        case _:
            raise NotImplementedError(f"delta_matching.sm100 supports SM100, not compute capability {ARCH_MAJOR}.x")


def dm_attn_debug(q, k, v, do, causal=True, mode="matched", identity=False, deterministic=None, lockstep=None):
    """
    The delta the backward consumes and the LSE it recomputes P from, (B, H, N) fp32 each ("delta",
    "lse"), from the same producers, forward and delta path as dm_attn_func. identity=True (matched
    mode) also runs the backward's identity build and returns, accumulated in the kernel from its
    own tiles before dS quantization, "rowsum_dS" = sum_j dS_ij and "sum_abs_PdP" = sum_j |P_ij
    dP_ij|, (B, H, N) fp32 each. lockstep: head_dim 256 only (see attention_hd256).
    """
    match ARCH_MAJOR:
        case 10:
            if q.shape[-1] == 256:
                from delta_matching.sm100.api_hd256 import dm_attn_debug_hd256
                return dm_attn_debug_hd256(q, k, v, do, causal, mode, identity, deterministic, lockstep)
            from delta_matching.sm100.attention_hd128_sm100 import attention_debug_hd128
            return attention_debug_hd128(q, k, v, do, causal, mode, identity, deterministic)
        case _:
            raise NotImplementedError(f"delta_matching.sm100 supports SM100, not compute capability {ARCH_MAJOR}.x")
