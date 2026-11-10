"""Host entry of the hd256 forward, MXFP8 (sm100-mx at D=256, reference preset sm100-mx-d256; kernel dm_fwd_hd256_mx_sm100.py).

dm_fwd_hd256_mx(q8, qs, k8, kvw, v8, causal) -> (O (B,N,H,256) bf16, LSE (B,H,N) fp32 natural log)
  q8 (B,N,H,256) e4m3 (sm = 2^-4 baked), k8 / v8 (B,N,Hk,256) e4m3; qs (B,H,N/32) fp32 pow2, kvw (B,Hk,N/64) int32 packed
  UE8M0 K/V words (quant.quant_qkv2(q, k, v, 256**-0.5)). N % 128 == 0.
Compile cache: persistent (FLASH_ATTENTION_CUTE_DSL_CACHE_DIR) under fwd._fingerprint(); shapes are dynamic. Under torch
FakeTensorMode the call only compiles (AOT, no GPU) and returns empty fakes.
"""
import os

import torch
import cutlass.cute as cute
from flash_attn.cute import cache_utils
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.testing import is_fake_mode

from .dm_fwd_hd256_mx_sm100 import DMFwdHd256MxSm100

_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None:
        # cache key: the same fingerprint as every other DM kernel = FA4 source fingerprint +
        # ALL delta_matching/sm100/*.py bytes + fwd.compile_env() (DSL lineinfo/opt/debug, SM count, fence settings, ...).
        # (Was: only 4 files and no compile settings -> a change in dm_common*.py / quant*.py or the DSL settings kept the key.)
        from .fwd import _fingerprint
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHE = cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint() / "fwd_hd256")
        else:
            _CACHE = cache_utils.JITCache()
    return _CACHE


# ---------------------------------------------------------------------------------------------------------------------
# sm100-mx.
TRACE = {}   # DEBUG (DM_HD256_FWD_TRACE=1): "buf" = the last call's int64 event buffer (see DMFwdHd256MxSm100.tr)


def dm_fwd_hd256_mx(q8, qs, k8, kvw, v8, causal: bool, kv_stage: int = 4, vs=None):
    """q8 (B,N,H,256) e4m3 (sm baked), k8 / v8 (B,N,Hk,256) e4m3; qs (B,H,N/32) fp32 pow2 (quant.quant_qkv2).
    kvw: (B,Hk,N/64) int32 packed UE8M0 K/V words (fwd.pack_kv_scales), or the K scales ks (B,Hk,N/32) fp32 together
    with vs= (packed here, same words). Returns O (B,N,H,256) bf16, LSE (B,H,N) fp32 natural log."""
    if kvw.dtype != torch.int32:
        from .fwd import pack_kv_scales
        kvw = pack_kv_scales(kvw, vs)
    B, N, H, D = q8.shape
    Hk = k8.shape[2]
    assert D == 256 and N % 128 == 0 and H % Hk == 0
    assert q8.dtype == torch.float8_e4m3fn and k8.dtype == q8.dtype and v8.dtype == q8.dtype
    assert qs.shape == (B, H, N // 32) and kvw.shape == (B, Hk, N // 64) and kvw.dtype == torch.int32
    q8, k8, v8, qs, kvw = (t.contiguous() for t in (q8, k8, v8, qs, kvw))
    o = torch.empty(B, N, H, D, device=q8.device, dtype=torch.bfloat16)
    lse = torch.empty(B, H, N, device=q8.device, dtype=torch.float32)
    sfw = os.environ.get("DM_HD256_MX_SFWAIT", "1") != "0"      # 0 = DEBUG A/B only (pre-C0 unsafe SF overwrite order)
    pc = int(os.environ.get("DM_HD256_MX_PC", "0"))               # DEBUG positive control (0 = off)
    sfb = os.environ.get("DM_HD256_MX_SFBANK", "0") == "1"      # 1 = C5 single-S + SF banks (rejected: slower than C0); default C0
    two = (not sfb) and os.environ.get("DM_HD256_MX_2WG", "1") != "0"   # C6 two softmax WGs by tile parity (default)
    key = ("fwd_hd256_mx", bool(causal), H // Hk, kv_stage, sfw) + ((pc,) if pc else ()) + (("c5",) if sfb else ()) \
        + (() if two else ("1wg",))
    otd = os.environ.get("DM_HD256_ORDT", "1") != "0"           # FW15: teardown fence pair (DEFAULT; 0 = DEBUG pre-fix)
    if not otd:
        key = key + ("noordt",)
    lpt = bool(causal) and os.environ.get("DM_HD256_FWD_LPT", "0") == "1"   # L1: global longest-first CTA order (opt-in: REJECTED, 16/4 8k fwd slower)
    if causal and not lpt:
        key = key + ("nolpt",)
    epi = int(os.environ.get("DM_HD256_FWD_EPI", "2"))   # F1c: SMEM-staged coalesced O epilogue + TMEM prefetch (DEFAULT 2; 1 = F1a, 0 = old)
    if epi != 2:
        key = key + (f"epi{epi}",)
    split = int(os.environ.get("DM_HD256_FWD_SPLIT", "0"))  # C7b: SF at slot+112, QK N112+N16, early S release (2); 0 = C6
    assert split in (0, 2), split
    if split:
        key = key + (f"split{split}",)
    g1 = int(os.environ.get("DM_HD256_FWD_G1", "0"))   # G1 critical-path trims (bitmask, see DMFwdHd256MxSm100); 0 = C6
    if g1:
        key = key + (f"g1_{g1}",)
    c8 = int(os.environ.get("DM_HD256_FWD_C8", "1"))   # C8 column split (implies G1a): 1 = C8 (DEFAULT), 2 = C8b; 0 = C6
    if c8 != 1:
        key = key + (f"c8_{c8}",)
    trace = os.environ.get("DM_HD256_FWD_TRACE", "0") == "1"   # DEBUG: per-event %clock64 trace -> TRACE["buf"]
    dbg = None
    if trace:
        key = key + ("trace",)
        n_m = N // 128
        dbg = TRACE.get("buf")
        if dbg is None or is_fake_mode() or dbg.shape != (n_m * H * B, n_m + 1, 32) or dbg.device != q8.device:
            dbg = torch.zeros(n_m * H * B, n_m + 1, 32, device=q8.device, dtype=torch.int64)   # reused (not re-zeroed) per call
            if not is_fake_mode():
                TRACE["buf"] = dbg
    cache = _cache()
    if key not in cache:
        if os.environ.get("DM_H256_NO_COMPILE", "0") == "1":
            raise RuntimeError(f"DM_H256_NO_COMPILE=1 but {key} is not in the compile cache {getattr(cache, 'cache_path', '?')}"
                               " (compile it ahead of time with the same FLASH_ATTENTION_CUTE_DSL_CACHE_DIR)")
        kern = DMFwdHd256MxSm100(bool(causal), H // Hk, kv_stage=kv_stage, sf_wait=sfw, pc_inj=pc, sf_bank=sfb, two_wg=two, ord_td=otd, lpt=lpt, epi=epi,
                                 split=split, trace=trace, g1=g1, c8=c8)
        args = [to_cute_tensor(t) for t in (q8, k8, v8, o)]
        args += [to_cute_tensor(lse, assumed_align=4), to_cute_tensor(qs, assumed_align=4), to_cute_tensor(kvw, assumed_align=4)]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if trace:
            cache[key] = cute.compile(kern, *args, stream, to_cute_tensor(dbg), options="--enable-tvm-ffi")
        else:
            cache[key] = cute.compile(kern, *args, stream, options="--enable-tvm-ffi")
    if is_fake_mode():
        return o, lse
    u8 = torch.uint8
    cache[key](q8.view(u8), k8.view(u8), v8.view(u8), o, lse, qs, kvw, dbg)   # dbg None unless trace (tvm-ffi slot mDbg)
    return o, lse
