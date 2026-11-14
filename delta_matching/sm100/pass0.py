"""Host entry of the matched-delta pass 0 (DMPass0Sm100): FA4's _flash_attn_fwd plumbing with the kernel class swapped.

dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal) -> delta (B, H, N) fp32 =
    sum_j P dP / sum_j P with P = exp2(fma(S, (ks*qs)*log2e, -lse*log2e)), dP = (dO8 V8^T) * (vs*dos)   (dm_common contract)
Scales, dO codes and lse travel as aux tensors; delta comes back in the LSE output slot.
"""
import dataclasses
import os
import threading

import torch
import flash_attn.cute.interface as fi
from flash_attn.cute import cache_utils

from .dm_pass0_sm100 import DMPass0Sm100
from .compile_cache_sm100 import compile_env, fingerprint
from . import launch_cache_sm100
from delta_matching.sm100.attention_math_sm100 import b6_m
_env = launch_cache_sm100.env

_LOCK = threading.Lock()
FORCE = dict(use_2cta_instrs=False, pack_gqa=False, use_s_ping_pong=False, num_splits=1, use_tma_o=False,
             overlap_sO_sQ=False, kv_stage=4)
REGS = dict(softmax=208, correction=32, other=64)  # P0: correction warps idle -> registers to softmax
_c3r = _env("DM_P0_C3REGS", "168,96,64").split(",")   # +c3: correction WG computes chunk 3
REGS_C3 = dict(softmax=int(_c3r[0]), correction=int(_c3r[1]), other=int(_c3r[2]))
_tlr = _env("DM_P0_TLREGS", "112,112,32").split(",")   # +tl: 640 threads (96-reg launch), per SMSP 4 compute warps + 1 other
REGS_TL = dict(softmax=int(_tlr[0]), correction=int(_tlr[1]), other=int(_tlr[2]))


def _make_cache():
    if cache_utils.CUTE_DSL_CACHE_ENABLED:
        return cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint() / "pass0")
    return cache_utils.JITCache()


_CACHES = {}
_CLASSES = {}


def _variant(exp_variant: str):
    if exp_variant not in _CLASSES:
        _CLASSES[exp_variant] = type(f"DMPass0Sm100_{exp_variant.replace('+', '_F_')}", (DMPass0Sm100,), {"dm_exp_variant": exp_variant})
    mk = (exp_variant, compile_env())   # memo keyed on every compile-time input
    if mk not in _CACHES:
        c = _make_cache()
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            c = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint() / f"pass0_{exp_variant}")
        _CACHES[mk] = c
    return _CLASSES[exp_variant], _CACHES[mk]


DPDUMP = None
PDUMP = None   # DEBUG: set to a (B, H, N, N) fp32 tensor before dm_delta with exp variant "+pd"


_SEL_MEMO = {}   # host-overhead cut: memoized config selection


def _aux(t: torch.Tensor, leading_dim: int):
    t.__leading_dim__ = leading_dim
    t.__assumed_align__ = 16 if t.dtype == torch.uint8 else 4
    return t


def dm_delta(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal: bool, config_override: dict | None = None,
             exp_variant: str | None = None):
    if exp_variant is None:
        exp_variant = _env("DM_P0_EXP", "p3_16_0+ma+h2+pf+sq+ts+tl")   # +tl two lanes per row, 4 compute warps/SMSP; +ts double-float TwoSum cross-tile sums; +pf scale prefetch; +sq per-stage specialized code; h2 halves; every exp via dm_bwd_p_pair (MUFU), bitwise the backward's P -> rowsum(dS) = 0 at fp32 round-off
    if b6_m() > 0 and "+b6_" not in exp_variant:
        exp_variant = exp_variant + f"+b6_{b6_m()}"
    over = dict(FORCE)
    if config_override:
        over.update(config_override)
    aux = [_aux(qs.contiguous(), 2), _aux(ks.contiguous(), 2), _aux(vs.contiguous(), 2), _aux(dos.contiguous(), 2),
           _aux(do8.contiguous().view(torch.uint8), 3), _aux(lse.contiguous(), 2)]
    if "tr" in exp_variant.split("+")[1:]:
        global TRACE
        TRACE = torch.zeros(64, dtype=torch.int64, device=lse.device)
        aux.append(_aux(TRACE, 0))
    if "pd" in exp_variant.split("+")[1:]:       # DEBUG: (B, H, N, N) fp32 P dump (last aux)
        global PDUMP
        aux.append(_aux(PDUMP, 3))
        aux.append(_aux(DPDUMP, 3))
    if any(f in exp_variant.split("+")[1:] for f in ("tr", "pd")):
        return _dm_delta_slow(q8, k8, v8, aux, causal, over, exp_variant)[1]
    names = ["qs", "ks", "vs", "dos", "do8", "lse"]
    inp = dict(q8=q8, k8=k8, v8=v8, **{n: t for n, t in zip(names, aux)})
    # the O output of pass 0 is an unused scratch buffer: returned to the fast path (as an output) and dropped here
    return launch_cache_sm100.call("pass0", inp, (bool(causal), exp_variant, repr(sorted(over.items())), compile_env()),
                         lambda: _dm_delta_slow(q8, k8, v8, aux, causal, over, exp_variant), cmp=[1])[1]


def _dm_delta_slow(q8, k8, v8, aux, causal, over, exp_variant):
    orig = (fi.FlashAttentionForwardSm100, fi._flash_attn_fwd.compile_cache, fi.select_fwd_config)

    def select(inputs):
        try:
            mk = (inputs, exp_variant, tuple(sorted((k, repr(v)) for k, v in over.items())))
            hit = _SEL_MEMO.get(mk)
        except TypeError:
            mk, hit = None, None
        if hit is not None:
            return hit
        cfg = _select_impl(inputs)
        if mk is not None:
            _SEL_MEMO[mk] = cfg
        return cfg

    def _select_impl(inputs):
        cfg = orig[2](inputs)
        cfg = dataclasses.replace(cfg, **{k: v for k, v in over.items() if hasattr(cfg, k) and k != "registers"})
        regs = over.get("registers", REGS_C3 if "c3" in exp_variant.split("+")[1:] else REGS)
        if "tl" in exp_variant.split("+")[1:]:
            regs = REGS_TL
            assert 4 * regs["softmax"] + regs["other"] <= 480 and regs["softmax"] == regs["correction"], "tl regs"
        for fl in exp_variant.split("+")[1:]:      # +r<softmax>_<correction>: register split (other = 64)
            if fl.startswith("r") and "_" in fl and fl[1:].split("_")[0].isdigit():
                r_ = [int(x) for x in fl[1:].split("_")]
                regs = dict(softmax=r_[0], correction=r_[1], other=r_[2] if len(r_) > 2 else 64)
                # per-SMSP register file: each SMSP holds one warp of each role (softmax0, softmax1, correction, other),
                # so 2*softmax + correction + other must be <= 512 or setmaxnreg.inc never completes (+r216_24 hung).
                if 2 * regs["softmax"] + regs["correction"] + regs["other"] > 512:
                    raise ValueError(f"DM_P0_EXP +{fl}: 2*softmax+correction+other = "
                                     f"{2 * regs['softmax'] + regs['correction'] + regs['other']} > 512 (deadlocks)")
        if regs is not None:
            cfg = dataclasses.replace(cfg, registers=dataclasses.replace(cfg.registers, **regs))
        return cfg

    with _LOCK:
        cls, cache = _variant(exp_variant)
        fi.FlashAttentionForwardSm100 = cls
        fi._flash_attn_fwd.compile_cache = launch_cache_sm100.RecCache(cache)
        fi.select_fwd_config = select
        try:
            o_unused, delta, *_ = fi._flash_attn_fwd(q8, k8, v8, softmax_scale=1.0, causal=causal, return_lse=True,
                                              pack_gqa=False, aux_tensors=aux)
        finally:
            fi.FlashAttentionForwardSm100, fi._flash_attn_fwd.compile_cache, fi.select_fwd_config = orig
    return (o_unused, delta)
