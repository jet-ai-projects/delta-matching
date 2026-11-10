"""Host entry of the hd256 matched-delta pass 0, MXFP8.
Same plumbing as pass0.py (FA4 _flash_attn_fwd with the kernel class swapped); hd128's pass0.py is untouched.

dm_delta_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal) -> delta (B, H, N) fp32, MXFP8 (sm100-mx-d256) contract:
  q8 / do8 (B, N, H, 256) e4m3, qs / dos (B, H, N/32) fp32;  k8 / v8 (B, N, Hk, 256) e4m3, ks / vs (B, Hk, N/32) fp32;
  every scale an exact power of two per 32 tokens over all D (quant.quant_qkv2 / quant.quant_do_delta);
  lse (B, H, N) fp32 natural log.  P / dP per dm_common_hd256.dm_mx256_*; delta = sum_j P dP / sum_j P (num/den only).
"""
import dataclasses
import threading

import torch
import flash_attn.cute.interface as fi
from flash_attn.cute import cache_utils

from .dm_pass0_hd256_sm100 import DMPass0Hd256Sm100
from .fwd import _fingerprint
from . import fastpath

D = 256
_LOCK = threading.Lock()
FORCE = dict(use_2cta_instrs=False, pack_gqa=False, use_s_ping_pong=False, num_splits=1, use_tma_o=False,
             overlap_sO_sQ=False, kv_stage=2, q_stage=2, tile_m=128, tile_n=128)
REGS_TL = dict(softmax=112, correction=112, other=32)   # 640 threads: per SMSP 4 compute warps + 1 other (4x112+32=480)
BASE_VARIANT = "p3_16_0+ma+h2+pf+sq+ts+tl+es"   # +es: E1 (early S-slot release), bitwise the same values

_CACHES = {}
_CLASSES = {}
_SEL_MEMO = {}


def _variant(exp_variant: str):
    if exp_variant not in _CLASSES:
        _CLASSES[exp_variant] = type(f"DMPass0Hd256Sm100_{exp_variant.replace('+', '_F_')}", (DMPass0Hd256Sm100,),
                                     {"dm_exp_variant": exp_variant})
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            c = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint() / f"pass0hmx_{exp_variant}")
        else:
            c = cache_utils.JITCache()
        _CACHES[exp_variant] = c
    return _CLASSES[exp_variant], _CACHES[exp_variant]


def _aux(t: torch.Tensor, leading_dim: int):
    t.__leading_dim__ = leading_dim
    t.__assumed_align__ = 16 if t.dtype == torch.uint8 else 4
    return t


def variant_for() -> str:
    """The shipped pass-0 variant: delta = num / den (+dnd), the only hd256 MXFP8 delta form."""
    return BASE_VARIANT + "+dnd"


def dm_delta_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal: bool, exp_variant: str | None = None):
    B, N, H, Dq = q8.shape
    assert Dq == D and k8.shape[-1] == D and v8.shape[-1] == D and do8.shape == q8.shape and N % 128 == 0
    assert qs.shape == (B, H, N // 32) and dos.shape == (B, H, N // 32), "Q / dO: one pow2 scale per 32 rows (MXFP8)"
    assert ks.shape == (B, k8.shape[2], N // 32) and vs.shape == ks.shape, "K / V: one pow2 scale per 32 keys (MXFP8)"
    if exp_variant is None:
        exp_variant = variant_for()
    over = dict(FORCE)
    if "+kv3" in exp_variant:   # E4: K/V ring depth 3 (224 KiB raw SMEM); variant token is host-only
        over["kv_stage"] = 3
    aux = [_aux(qs.contiguous(), 2), _aux(ks.contiguous(), 2), _aux(vs.contiguous(), 2), _aux(dos.contiguous(), 2),
           _aux(do8.contiguous().view(torch.uint8), 3), _aux(lse.contiguous(), 2)]
    names = ["qs", "ks", "vs", "dos", "do8", "lse"]
    inp = dict(q8=q8, k8=k8, v8=v8, **{n: t for n, t in zip(names, aux)})
    return fastpath.call("pass0_hd256", inp, (bool(causal), exp_variant, repr(sorted(over.items()))),
                         lambda: _slow(q8, k8, v8, aux, causal, over, exp_variant), cmp=[1])[1]


LAST_CFG = {}


def _slow(q8, k8, v8, aux, causal, over, exp_variant):
    orig = (fi.FlashAttentionForwardSm100, fi._flash_attn_fwd.compile_cache, fi.select_fwd_config)

    def select(inputs):
        try:
            mk = (inputs, exp_variant, tuple(sorted((k, repr(v)) for k, v in over.items())))
            hit = _SEL_MEMO.get(mk)
        except TypeError:
            mk, hit = None, None
        if hit is not None:
            return hit
        cfg = orig[2](inputs)
        cfg = dataclasses.replace(cfg, **{k: v for k, v in over.items() if hasattr(cfg, k)})
        cfg = dataclasses.replace(cfg, registers=dataclasses.replace(cfg.registers, **REGS_TL))
        LAST_CFG["cfg"] = cfg
        if mk is not None:
            _SEL_MEMO[mk] = cfg
        return cfg

    with _LOCK:
        cls, cache = _variant(exp_variant)
        fi.FlashAttentionForwardSm100 = cls
        fi._flash_attn_fwd.compile_cache = fastpath.RecCache(cache)
        fi.select_fwd_config = select
        try:
            o_unused, delta, *_ = fi._flash_attn_fwd(q8, k8, v8, softmax_scale=1.0, causal=causal, return_lse=True,
                                                     pack_gqa=False, aux_tensors=aux)
        finally:
            fi.FlashAttentionForwardSm100, fi._flash_attn_fwd.compile_cache, fi.select_fwd_config = orig
    return (o_unused, delta)
