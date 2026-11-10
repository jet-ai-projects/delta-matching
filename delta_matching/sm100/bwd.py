"""Host entry of the sm100-mx backward main pass: FA4's _flash_attn_bwd host code, source-patched (no edits to the
shared FA4 checkout) to run DMBackwardSm100 on fp8 codes, with our delta / LSE instead of FA4's preprocess, scales
passed as aux tensors [qs, ks, vs, dos], bf16 outputs, and no softmax-scale factor on dK (Q codes carry sm)."""
import inspect
import math
import os

import torch
import flash_attn.cute.interface as fi
from flash_attn.cute import cache_utils

from .dm_bwd_sm100 import DMBackwardSm100
from .compile_cache_sm100 import compile_env, fingerprint
from . import launch_cache_sm100
from delta_matching.sm100.attention_math_sm100 import b6_m
_env = launch_cache_sm100.env

LOG2E = math.log2(math.e)


_CACHES = {}


def _dm_bwd_cache():
    """One compile cache per DM_BWD_OPTS string (kernel variant knobs read in DMBackwardSm100.__init__), so A/B
    variants can run in one process."""
    opts = _env("DM_BWD_OPTS", "")
    mk = (opts, compile_env())   # memo keyed on every compile-time input, not only the opts
    if mk not in _CACHES:
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHES[mk] = cache_utils.JITPersistentCache(
                cache_utils.get_cache_path() / "dm_sm100" / fingerprint(opts) / "bwd")
        else:
            _CACHES[mk] = cache_utils.JITCache()
        _CACHES[mk] = launch_cache_sm100.RecCache(_CACHES[mk])
    return _CACHES[mk]


_CTX = {}


# ---- scratch prestaging (E1): FA4's host allocates + zero-fills scratch (dq_accum, dpsum, lse_log2, dK/dV accum,
# semaphores) right before the main kernel. With prestaging, the allocations recorded on the first call of a shape are
# re-created and zeroed on a side stream while pass 0 runs, so only the delta copy stays on the critical path.
_REC = {}          # key -> [(kind, shape, dtype)] of torch.empty / torch.zeros calls in the host (recorded once)
_SIDE = {}
_PFAST = {}       # host fast path: (key, raw stream) -> re-initialiser of a persistent prestaged scratch pool


def _dm_alloc(kind, *shape, dtype=None, device=None, **kw):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = tuple(shape[0])
    pool = _CTX.get("pool")
    if pool:
        t = pool.pop(0)
        assert tuple(t.shape) == tuple(shape) and t.dtype == dtype, (t.shape, shape, t.dtype, dtype)
        return t
    rec = _CTX.get("rec")
    if kind == "z" and dtype == torch.float32 and _fws(_CTX.get("deterministic", True)):
        kind = "e"     # dK/dV accum (GQA): the first group head stores (DM fws), no zero-fill
    if rec is not None:
        rec.append((kind, tuple(shape), dtype))
    return (torch.zeros if kind == "z" else torch.empty)(*shape, dtype=dtype, device=device, **kw)


def _dm_zeros(*a, **k):
    return _dm_alloc("z", *a, **k)


def _dm_empty(*a, **k):
    return _dm_alloc("e", *a, **k)


def _fws(deterministic):
    """dQ first-writer-stores in the kernel (DMBackwardSm100.dm_fws): dq_accum needs no zero-fill."""
    return bool(deterministic) and "nofws" not in _env("DM_BWD_OPTS", "").split(",")


_AUTO_MEMO = {}


def _auto_opts(q8, k8, causal=False):
    """Memoized _auto_opts_impl: pure function of the env knobs and the shape."""
    mk = (_env("DM_BWD_OPTS", ""), _env("DM_B6", ""), _env("DM_NSM", ""),
          q8.shape[1], q8.shape[2], k8.shape[2], bool(causal), q8.device.index)
    r = _AUTO_MEMO.get(mk)
    if r is None:
        r = _AUTO_MEMO[mk] = _auto_opts_impl(q8, k8, causal)
    return r


def _auto_opts_impl(q8, k8, causal=False):
    """Effective DM_BWD_OPTS for this shape. DM gloop (GQA, auto): short sequences spend most of a KV tile on the
    fp32 dK/dV accumulation across the qh query-head CTAs + postprocess; loop the group inside one CTA instead (dK/dV
    in TMEM, stored once). Long causal sequences lose (4x longer tiles -> tail), so only N <= gloopN (default 4096).
    'nogloop' disables, 'gloop' forces."""
    ol = [o for o in _env("DM_BWD_OPTS", "").split(",") if o]
    if b6_m() > 0 and not any(o.startswith("b6=") for o in ol):
        ol = ol + [f"b6={b6_m()}"]   # same DM_B6 as pass 0 (bitwise-identical P)
    # DM dsqw: dS tile amax / scale / e4m3 / stores / exchange on a dedicated warpgroup (bitwise);
    # 'nodsqw' disables. Not combined with the layouts it does not support.
    if "nodsqw" not in ol and not any(o in ol for o in ("dsq", "dsqw", "cwg4", "ilp2", "early_dq")):
        ol = ol + ["dsqw", "dsqr=72:136:48:88"]
    gN = int(next((o[7:] for o in ol if o.startswith("gloopN=")), "4096"))   # GQA long-loop variant only above this N: 4k bwd 0.94x, 8k +0.1..1% slower
    if q8.shape[2] > k8.shape[2] and "gloop" not in ol and "nogloop" not in ol and q8.shape[1] <= gN:
        ol = ol + ["gloop"]
    # DM pers (MHA, auto): persistent grid of 148 CTAs (74 clusters) for short sequences, where the per-CTA
    # launch/prologue/epilogue is a large share of a KV tile; long sequences keep the hardware's dynamic CTA order
    # (static striding loses 4-18% there). 'nopers' disables, explicit 'pers=N' overrides, persN= sets the cut.
    # DM tconv (rw8k, det, auto): PDL tail converter for dQ (tailconv.py) for N <= tconvN (default 16384): 8k e2e -1.0..-1.5%
    # per call (both GQA shapes); measured +0.8..+1.6% at 64k-128k, so long sequences keep the serial postprocess.
    # 'tconv' forces it, 'notconv' disables, tconvN=<N> moves the cut.
    tN = int(next((o[7:] for o in ol if o.startswith("tconvN=")), "16384"))
    if ("tconv" not in ol and "notconv" not in ol and q8.shape[1] <= tN
            and not any(o in ol for o in ("lwc", "trace", "pdyn", "rsdbg", "pdump"))):
        ol = ol + ["tconv"]
    pN = int(next((o[6:] for o in ol if o.startswith("persN=")), "0"))   # off by default: no gain at 8k-64k
    # DM pcN / pdyn REMOVED: the dynamic persistent scheduler no longer compiles, its "+6-8% at causal 8k"
    # was a first-arm timing bias; re-measured 0.998x. Fail loudly instead of silently ignoring.
    if any(o == "pdyn" or o.startswith("pcN=") for o in ol):
        raise ValueError("DM_BWD_OPTS pdyn / pcN= were removed; use pers=N or persN=N")
    if (q8.shape[2] == k8.shape[2] and "nopers" not in ol and not any(o.startswith("pers=") for o in ol)):
        if q8.shape[1] <= pN:
            ol = ol + ["pers=148"]
    return ",".join(ol)


_NSM_MEMO = {}


def _num_sms(q8):
    n = int(_env("DM_NSM", "0"))
    if n:
        return n
    try:
        d = q8.device.index
        if d not in _NSM_MEMO:
            _NSM_MEMO[d] = int(torch.cuda.get_device_properties(q8.device).multi_processor_count)
        return _NSM_MEMO[d]
    except Exception:
        return 148


def _key(q8, k8, causal, deterministic):
    return (tuple(q8.shape), tuple(k8.shape), bool(causal), bool(deterministic), _auto_opts(q8, k8, causal),
            q8.device.index)


def dm_bwd_prestage(q8, k8, lse, causal: bool, deterministic=True):
    """Call BEFORE pass 0 (current stream = the backward's stream): if this shape was seen, allocate + zero the main
    pass's scratch and compute lse_log2 on a side stream; dm_bwd_main then waits on it. Bitwise identical."""
    key = _key(q8, k8, causal, deterministic)
    rec = _REC.get(key)
    if rec is None or _env("DM_E2E", "1") == "0":
        return False
    dev = q8.device.index
    if launch_cache_sm100.ENABLED:
        f = _PFAST.get((key, torch._C._cuda_getCurrentRawStream(dev)))
        if f is not None:
            f(lse)
            return True
    main = torch.cuda.current_stream(dev)
    side = _SIDE.get(dev)
    if side is None:
        side = _SIDE[dev] = torch.cuda.Stream(device=q8.device)
    side.wait_stream(main)          # lse ready; memory freed on main is reusable
    N = lse.shape[-1]
    with torch.cuda.stream(side):
        pool = [(torch.zeros if kind == "z" else torch.empty)(*shape, dtype=dt, device=q8.device)
                for kind, shape, dt in rec]
        dq_accum, dpsum, lse_log2 = pool[0], pool[1], pool[2]
        if not _fws(deterministic):
            dq_accum.zero_()
        dpsum[..., N:].zero_()
        torch.mul(lse, LOG2E, out=lse_log2[..., :N])
        lse_log2[..., N:].zero_()
        ev = torch.cuda.Event()
        ev.record(side)
    for t in pool:
        t.record_stream(main)
    _CTX["staged"] = (key, pool, ev, main)
    if launch_cache_sm100.ENABLED:
        _PFAST[(key, main.cuda_stream)] = _reuse_fn(key, rec, pool, main, side, N, deterministic)
    return True


def _reuse_fn(key, rec, pool, main, side, N, deterministic):
    """Host fast path: persistent scratch per (shape, stream) re-initialised on the side stream with the same ops as a
    fresh prestage (fresh zeros -> zero_()), events reused (each wait is enqueued before the next record)."""
    zs = [t for (kind, _, _), t in zip(rec, pool) if kind == "z"]
    dq_accum, dpsum, lse_log2 = pool[0], pool[1], pool[2]
    if not _fws(deterministic):
        zs.append(dq_accum)
    zs.append(dpsum[..., N:])
    lse_dst, lse_pad = lse_log2[..., :N], lse_log2[..., N:]
    ev_in, ev = torch.cuda.Event(), torch.cuda.Event()
    setst = torch._C._cuda_setStream
    sa = dict(stream_id=side.stream_id, device_index=side.device_index, device_type=side.device_type)
    ma = dict(stream_id=main.stream_id, device_index=main.device_index, device_type=main.device_type)
    staged = (key, pool, ev, main)

    def f(lse):
        ev_in.record(main)
        side.wait_event(ev_in)        # = side.wait_stream(main)
        setst(**sa)
        try:
            for t in zs:
                t.zero_()
            torch.mul(lse, LOG2E, out=lse_dst)
            lse_pad.zero_()
            ev.record(side)
        finally:
            setst(**ma)
        _CTX["staged"] = staged
    return f


def _dm_bwd_preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, *args, fake_mode=False, **kw):
    """Replaces FA4's preprocess: dpsum <- our delta (_CTX), lse_log2 <- lse * log2e, dq_accum <- 0."""
    if fake_mode:
        return
    delta = _CTX["delta"]
    N = lse.shape[-1]
    if _CTX.get("prestaged"):
        torch.cuda.current_stream().wait_event(_CTX["prestaged"])
        dpsum[..., :N].copy_(delta)
        return
    dpsum[..., :N].copy_(delta)
    dpsum[..., N:].zero_()
    lse_log2[..., :N].copy_(lse * LOG2E)
    lse_log2[..., N:].zero_()
    if dq_accum is not None and not _fws(_CTX.get("deterministic", True)):
        dq_accum.zero_()


def _patched_source():
    src = inspect.getsource(fi._flash_attn_bwd)
    subs = [
        ("def _flash_attn_bwd(", "def _dm_flash_attn_bwd("),
        ('assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"',
         'assert q.dtype == torch.float8_e4m3fn, "dm: fp8 codes expected"'),
        ("assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (", "assert True, ("),
        ("out_torch_dtype = q.dtype", "out_torch_dtype = torch.bfloat16"),
        ("dq = torch.empty_like(q)", "dq = _CTX['dq'] if _CTX.get('dq') is not None else torch.empty_like(q, dtype=torch.bfloat16)"),
        ("dk = torch.empty_like(k)", "dk = torch.empty_like(k, dtype=torch.bfloat16)"),
        ("dv = torch.empty_like(v)", "dv = torch.empty_like(v, dtype=torch.bfloat16)"),
        ("    _bwd_preprocess(\n        out, dout,", "    _dm_bwd_preprocess(\n        out, dout,"),
        ("fa_bwd_obj = FlashAttentionBackwardSm100(", "fa_bwd_obj = _DMBackwardSm100("),
        ("_flash_attn_bwd.compile_cache", "_dm_bwd_cache()"),
        ("dk_accum, dk, softmax_scale,", "dk_accum, dk, 1.0,"),
        # rw8k tconv: readiness queue for the tail converter, allocated with the scratch pool (zeroed per call by the
        # prestage side stream, which waits for the previous call's converter) and passed to the kernel as aux[4]
        ("    else:\n        dQ_semaphore = None\n",
         "    else:\n        dQ_semaphore = None\n"
         "    if dQ_semaphore is not None and _dm_tconv():\n"
         "        _dm_tcq = torch.zeros(1 + 2 * (dQ_semaphore.numel() // dQ_semaphore.shape[-1]), dtype=torch.int32, device=device)\n"
         "        _CTX['tcq_cur'] = _dm_tcq\n"
         "        aux_tensors = list(aux_tensors[:4]) + [_dm_tcq] + list(aux_tensors[4:])\n"),
        # rw8k tconv: hand each postprocess call its readiness semaphore (det dQ / dK / dV lock counters)
        ("        _bwd_postprocess_convert(\n            dq_accum, dq,",
         "        _CTX['dm_sem'] = ('q', dQ_semaphore, causal, qhead_per_kvhead, cluster_size * tile_n)\n        _bwd_postprocess_convert(\n            dq_accum, dq,"),
        ("            _bwd_postprocess_convert(\n                dk_accum, dk,",
         "            _CTX['dm_sem'] = ('kv', dK_semaphore, causal, qhead_per_kvhead, cluster_size * tile_n)\n            _bwd_postprocess_convert(\n                dk_accum, dk,"),
        ("            _bwd_postprocess_convert(\n                dv_accum, dv,",
         "            _CTX['dm_sem'] = ('kv', dV_semaphore, causal, qhead_per_kvhead, cluster_size * tile_n)\n            _bwd_postprocess_convert(\n                dv_accum, dv,"),
        ("dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel and not bwd_pack_gqa",
         "dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel and not bwd_pack_gqa and not _dm_gloop()"),
        ("            q.detach(),\n            k.detach(),\n            v.detach(),\n            dout,\n",
         "            q.detach().view(torch.uint8),\n            k.detach().view(torch.uint8),\n"
         "            v.detach().view(torch.uint8),\n            dout.view(torch.uint8),\n"),
    ]
    for a, b in subs:
        if a not in src:
            raise RuntimeError(f"bwd host patch anchor not found: {a!r}")
        src = src.replace(a, b)
    src = src.replace("torch.zeros(", "_dm_zeros(").replace("torch.empty(", "_dm_empty(")
    a = "        config = select_bwd_config(inputs)\n"
    if a not in src:
        raise RuntimeError("bwd host patch anchor not found: select_bwd_config")
    src = src.replace(a, a + "        config = _dm_cfg(config)\n")
    # bf16 for every host-side dtype decision (config, postprocess output); the kernel reads fp8 from the tensors
    a = "dtype = torch2cute_dtype_map[q.dtype]"
    if a in src:
        src = src.replace(a, "dtype = torch2cute_dtype_map[torch.bfloat16]")
    return src


def _dm_cfg(config):
    """bwd config overrides: no pack_gqa; DM_BWD_OPTS spt0 / spt1 force the deterministic dQ lock order."""
    import dataclasses
    over = dict(pack_gqa=False)
    opts = _env("DM_BWD_OPTS", "").split(",")
    if "spt0" in opts:
        over["spt"] = False
    if "spt1" in opts:
        over["spt"] = True
    for o in opts:
        if o.startswith("regs="):           # regs=reduce:compute:load_mma (setmaxnreg split, 2-CTA)
            r, c, l = (int(x) for x in o[5:].split(":"))
            from flash_attn.cute.config import Sm100BwdRegisters
            over["registers"] = Sm100BwdRegisters(reduce=r, compute=c, load=l, mma=l, empty=24)
    return dataclasses.replace(config, **over)


_ns = dict(vars(fi))
def _dm_post_convert(acc, out, *a, **k):
    """FA4 postprocess wrapper: skipped for dQ when the kernel's last writer already wrote bf16 dQ (DM lwc)."""
    sem = _CTX.pop("dm_sem", None)
    if _CTX.get("dq") is not None and out is _CTX["dq"]:
        return
    if _env("DM_RW8K_SEMDUMP", "0") == "1" and sem is not None and sem[1] is not None and not k.get("fake_mode", False):
        _CTX.setdefault("semdump", []).append(sem[1])   # DEBUG (rw8k): readiness counters after the call
    if (sem is not None and sem[1] is not None and sem[0] == "q" and "tconv" in _env("DM_BWD_OPTS", "").split(",")
            and _env("DM_TCONV_OFF", "0") != "1"):   # DM_TCONV_OFF=1: DEBUG, serial FA4 postprocess (isolates bwd main)
        # rw8k tconv: PDL tail converter (tailconv.py): same conversion, each tile as soon as its det counter is final
        from .tailconv import tconv_convert
        kind, sem_t, causal_, G, ctn = sem
        return tconv_convert(sem_t, acc, out, *a, dm_kind=kind, dm_causal=bool(causal_), dm_ctn=int(ctn),
                             dm_G=int(G), queue=_CTX["tcq_cur"], **k)
    return _ORIG_POST(acc, out, *a, **k)


_ORIG_POST = fi._bwd_postprocess_convert
_ns.update(_bwd_postprocess_convert=_dm_post_convert, _CTX=_CTX)
def _dm_gloop():
    """DM_BWD_OPTS gloop (GQA): dK/dV kept in TMEM across the query-head group, stored directly (no dKV postprocess)."""
    return "gloop" in _env("DM_BWD_OPTS", "").split(",")


_ns.update(_dm_gloop=_dm_gloop)


def _dm_tconv():
    """DM_BWD_OPTS tconv (rw8k): PDL tail converter for dQ (tailconv.py)."""
    return "tconv" in _env("DM_BWD_OPTS", "").split(",")


_ns.update(_dm_tconv=_dm_tconv)
_ns.update(_dm_zeros=_dm_zeros, _dm_empty=_dm_empty, _DMBackwardSm100=DMBackwardSm100, _dm_bwd_preprocess=_dm_bwd_preprocess, _dm_bwd_cache=_dm_bwd_cache, _dm_cfg=_dm_cfg)
exec(compile(_patched_source(), fi.__file__ + ":dm_patched_bwd", "exec"), _ns)
_dm_flash_attn_bwd = _ns["_dm_flash_attn_bwd"]

FORCE = dict(pack_gqa=False)


def dm_bwd_main(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal: bool, sm: float, deterministic=True, trace=None):
    """Main backward pass. q8 (B,N,H,D) e4m3 (sm baked), k8/v8 (B,N,Hk,D), do8 (B,N,H,D) e4m3; scales (B,H|Hk,N/32) fp32
    pow2; lse (B,H,N) natural; delta (B,H,N) fp32 in dP units. Returns dq, dk, dv bf16 (dk/dv per KV head)."""
    aux = [qs.contiguous(), ks.contiguous(), vs.contiguous(), dos.contiguous()]
    # DM d3nh precondition (MMA-warp opaque zero = loop counter >> 30): per-tile main-loop trip count < 2^30
    assert (q8.shape[1] // 128 + 1) * max(1, q8.shape[2] // k8.shape[2]) < 2 ** 30, "d3nh: main-loop trip count >= 2^30"
    if "tconv" in _auto_opts(q8, k8, causal).split(",") and deterministic:
        assert trace is None and not any(o in _env("DM_BWD_OPTS", "").split(",") for o in ("lwc", "trace")), \
            "tconv: det only, aux[4] = readiness queue (no trace / lwc)"
    if "lwc" in _env("DM_BWD_OPTS", "").split(","):
        assert trace is None and deterministic and abs(sm - q8.shape[-1] ** -0.5) < 1e-12, "lwc: det, sm = D^-0.5, no trace"
        _CTX["dq"] = torch.empty(q8.shape, dtype=torch.bfloat16, device=q8.device)
        aux.append(_CTX["dq"])
    _ao = _auto_opts(q8, k8, causal)
    if "pdyn" in _ao.split(","):
        # DM pdyn: claim counter for the dynamic persistent scheduler (last aux tensor), zeroed per launch
        _CTX["pdyn_ctr"] = torch.zeros(1, dtype=torch.int32, device=q8.device)
    if trace is not None and "rsdbg" in _env("DM_BWD_OPTS", ""):
        if isinstance(trace, (tuple, list)):   # (rs, pdump (B, H, N, N) fp32) with DM_BWD_OPTS=rsdbg,pdump
            aux.extend(trace)
        else:
            aux.append(trace)          # DEBUG rsdbg: float64 (B, H, N, 2) = (rowsum dS, sum |P dP|) accumulators
        trace = None
    if trace is not None:          # int64 (16,) accumulators of MMA-warp wait cycles (DM_BWD_OPTS must contain 'trace')
        aux.append(trace)
    if "pdyn" in _ao.split(","):
        aux.append(_CTX.pop("pdyn_ctr"))
    _CTX["delta"] = delta
    _CTX["deterministic"] = deterministic
    key = _key(q8, k8, causal, deterministic)
    st = _CTX.pop("staged", None)
    lse_c = lse.contiguous()
    if (st is not None and st[0] == key and trace is None and launch_cache_sm100.ENABLED and "lwc" not in _ao.split(",")
            and "pdyn" not in _ao.split(",")):
        pool, ev, mst = st[1], st[2], st[3]
        N = lse.shape[-1]

        def pre():          # = _dm_bwd_preprocess in prestaged mode
            mst.wait_event(ev)
            pool[1][..., :N].copy_(delta)

        inp = dict(q8=q8, k8=k8, v8=v8, do8=do8, lse=lse_c, delta=delta,
                   **{f"aux{i}": t for i, t in enumerate(aux)}, **{f"pool{i}": t for i, t in enumerate(pool)})

        def slow():
            _CTX["pool"], _CTX["prestaged"] = list(pool), ev
            opts0_ = _env("DM_BWD_OPTS", "")
            os.environ["DM_BWD_OPTS"] = _ao
            pc = fi._bwd_postprocess_convert.compile_cache
            fi._bwd_postprocess_convert.compile_cache = launch_cache_sm100.RecCache(pc)
            from . import tailconv as _tc       # rw8k tconv: record the converter launches too
            if _tc._CACHE is None:
                _tc._CACHE = _tc._cache()
            tcc = _tc._CACHE
            _tc._CACHE = launch_cache_sm100.RecCache(tcc)
            try:
                r = _dm_flash_attn_bwd(q8, k8, v8, do8, do8, lse_c, softmax_scale=sm, causal=causal,
                                       deterministic=deterministic, aux_tensors=aux)[:3]
                assert not _CTX.get("pool"), "prestaged scratch not fully consumed"
                return tuple(r)
            finally:
                fi._bwd_postprocess_convert.compile_cache = pc
                _tc._CACHE = tcc
                os.environ["DM_BWD_OPTS"] = opts0_
                _CTX.pop("pool", None)
                _CTX.pop("prestaged", None)
        def reset():        # verification call: re-initialise the scratch pool the slow path consumed (as a prestage)
            _PFAST[(key, mst.cuda_stream)](lse)
            s2 = _CTX.pop("staged")
            mst.wait_event(s2[2])

        return launch_cache_sm100.call("bwd", inp, (key, _ao, float(sm), bool(deterministic), compile_env()), slow, pre=pre, reset=reset,
                             loose=() if deterministic else ((0, 1, 2) if q8.shape[2] > k8.shape[2] else (0,)))   # nd: dQ unordered; dK/dV only for GQA (MHA nd dK/dV stay bitwise)
    opts0 = _env("DM_BWD_OPTS", "")
    os.environ["DM_BWD_OPTS"] = _ao     # kernel knobs + compile cache follow the effective opts
    if st is not None and st[0] == key and trace is None:
        _CTX["pool"], _CTX["prestaged"] = list(st[1]), st[2]
    elif key not in _REC and not isinstance(q8, torch._subclasses.fake_tensor.FakeTensor):
        _CTX["rec"] = []
    try:
        dq, dk, dv = _dm_flash_attn_bwd(
            q8, k8, v8, do8, do8, lse.contiguous(), softmax_scale=sm, causal=causal,
            deterministic=deterministic, aux_tensors=aux)[:3]
        if _CTX.get("rec") is not None:
            r = _CTX["rec"]
            if len(r) >= 3 and all(x[0] == "e" and x[2] == torch.float32 for x in r[:3]):
                _REC[key] = r
        assert not _CTX.get("pool"), "prestaged scratch not fully consumed"
    finally:
        os.environ["DM_BWD_OPTS"] = opts0
        _CTX.pop("dq", None)
        _CTX.pop("rec", None)
        _CTX.pop("pool", None)
        _CTX.pop("prestaged", None)
    return dq, dk, dv
