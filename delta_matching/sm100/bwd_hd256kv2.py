"""Host entry of the fused paired-CTA hd256 dK/dV kernel KV2.
dm_bwd_hd256kv2(...) has the signature and outputs of bwd_hd256mx.dm_bwd_hd256mx: dQ from DmHd256Mx's dq mode (unchanged),
dK / dV from ONE DmHd256Kv2 launch (replacing the dv + dk modes).  Deterministic; no workspace, no atomics outside dbg.
"""
import torch

from . import bwd_hd256mx as _mx
from .dm_bwd_hd256kv2_sm100 import DmHd256Kv2

D = 256
NUMERICS = "sm100-mx-d256"
DETERMINISTIC = True


_PH = {}


def _ph(dev):
    """per-device dbg placeholders (never written by a dbg=0 build)."""
    k = str(dev)
    if k not in _PH:
        _PH[k] = dict(dump=torch.zeros(1, 1, 1, 4, device=dev, dtype=torch.float32),
                      c=torch.zeros(1, 1, 16, device=dev, dtype=torch.uint8),
                      sb=torch.zeros(1, 1, 4, device=dev, dtype=torch.int32),
                      h=torch.zeros(1, 1, 4, device=dev, dtype=torch.int64))
    return _PH[k]


def dqw_tiles(B, H, n_t, causal):
    """number of (l, q tile, key tile) tiles of the dQ workspace."""
    return B * H * (n_t * (n_t + 1) // 2 if causal else n_t * n_t)


def dqw_alloc(B, H, N, causal, dev):
    T = dqw_tiles(B, H, N // 128, bool(causal))
    return (torch.empty(T, 128, 128, device=dev, dtype=torch.uint8), torch.empty(T, 1, 4, device=dev, dtype=torch.uint8))


def kv2_launch(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, knob="", opts=None, dqw=None, out=None, dqc=None):
    """dqw = (W, Wsb) from dqw_alloc: KV2 also stores the dS codes / raw block exponents for the dQ GEMM (opts must contain dqw).
    dqc = (desc, carry_dk, carry_dv, q1): exact-carry chunk, see DmHd256Kv2 opt dqc; desc =
    int32 device tensor [Q0, n_t, cin, cout] (+ pad), carries fp32 (B, N, Hk, 256); the launch covers q tiles [Q0, q1)."""
    B, N, H, Dq = q8.shape
    Hk = k8.shape[2]
    assert Dq == D and N % 128 == 0 and H % Hk == 0
    dev = q8.device
    causal = bool(causal)
    f8 = torch.float8_e4m3fn
    p = lambda x: x.view(f8).permute(1, 3, 2, 0)        # (B, N, h, D) -> (N, D, h, B)  K-major  # noqa: E731
    tt = lambda x: x.view(f8).permute(3, 1, 2, 0)       # (B, N, h, D) -> (D, N, h, B)  MN-major  # noqa: E731
    if out is not None:      # (dk, dv) contiguous batch slices of the caller's outputs
        dk, dv = out
        assert dk.shape == (B, N, Hk, D) and dk.is_contiguous() and dv.shape == dk.shape and dv.is_contiguous()
    else:
        dk = torch.empty(B, N, Hk, D, device=dev, dtype=torch.bfloat16)
        dv = torch.empty(B, N, Hk, D, device=dev, dtype=torch.bfloat16)
    L = B * H
    big = dbg & 2
    if big or dbg & 24:
        z8 = lambda *s_: torch.zeros(*s_, device=dev, dtype=torch.uint8)  # noqa: E731
        dump = torch.zeros(L, N, N, 4, device=dev, dtype=torch.float32) if big else _ph(dev)["dump"]
        cp_ = z8(L, N, N) if big else _ph(dev)["c"]
        cs_ = z8(L, N, N) if big else _ph(dev)["c"]
        sbp = torch.zeros(L, N, N // 32, device=dev, dtype=torch.int32) if big else _ph(dev)["sb"]
        sbs = torch.zeros_like(sbp) if big else _ph(dev)["sb"]
        hh = torch.zeros(L, N, 4, device=dev, dtype=torch.int64) if dbg & 24 else _ph(dev)["h"]   # 16: [rowsum dS, sum|P dP|] f64
    else:   # production: the dbg tensors are never touched (compiled out) -> shared placeholders, no per-call zeroing
        ph = _ph(dev)
        dump, cp_, cs_, sbp, sbs, hh = ph["dump"], ph["c"], ph["c"], ph["sb"], ph["sb"], ph["h"]
    ot = DmHd256Kv2.DEFAULT_OPTS if opts is None else opts
    assert (dqw is not None) == ("dqw" in ot.split() or "dqt" in ot.split()), (ot, dqw is None)
    if dqw is not None:
        assert not big
        cs_, sbs = dqw
        assert (dqc is not None or cs_.shape == (dqw_tiles(B, H, N // 128, causal), 128, 128)) and sbs.shape == (cs_.shape[0], 1, 4)
    assert (dqc is not None) == ("dqc" in ot.split())
    nt_arg = N // 128
    if dqc is not None:
        desc, ck, cv, nt_arg = dqc
        assert desc.dtype == torch.int32 and desc.numel() >= 4 and ck.dtype == torch.float32 and ck.shape == (B, N, Hk, D)
        assert cv.shape == ck.shape and ck.is_contiguous() and cv.is_contiguous() and 1 <= nt_arg <= N // 128
        dump, cp_, sbp = ck, cv.view(torch.uint8).view(B * N, Hk, D * 4), desc.view(1, 1, -1)
    ten = [p(k8), p(v8), p(q8), p(do8), tt(q8), tt(do8), qs, ks, vs, dos, lse, delta, dk, dv, dump, cp_, cs_, sbp, sbs, hh]
    lds = [1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2]
    kd = dbg & 26
    key = ("kv2", causal, kd, knob, ot)
    _mx._run(key, lambda: DmHd256Kv2(causal, kd, knob, ot), ten, lds, [nt_arg, H, Hk])
    extra = dict(dump=dump, codesP=cp_, codesS=cs_, sbP=sbp, sbS=sbs, hash=hh)
    return dk, dv, extra


_SIDE = {}


def _side(dev):
    k = str(dev)
    if k not in _SIDE:
        _SIDE[k] = torch.cuda.Stream(device=dev)
    return _SIDE[k]


def _s1_mode():
    from .fastpath import env as _env
    v = _env("DM_H256_S1", "kv2first")
    assert v in ("0", "kv2first", "dqfirst"), v
    return v


def dm_bwd_hd256kv2(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, knob="", opts=None):
    """Same contract as bwd_hd256mx.dm_bwd_hd256mx: (dq, dk, dv) [+ dbg dict].  dq = DmHd256Mx dq mode, dk/dv = KV2.
    S1 hand-off proof: at dbg=0 the dq kernel's only global store is dq and KV2's only global
    stores are dk/dv (no TMA stores, no atomics, no global scratch); both only read the ten inputs.
    S1 (DM_H256_S1, default kv2first; dbg=0 only): dq and KV2 are independent (both only read the quantized inputs,
    lse and delta; disjoint outputs) -> dq runs on a side stream concurrently with KV2 on the caller stream, so each fills
    the other's tail SMs. Same kernels, same inputs, same values; the caller stream joins the side stream before returning."""
    mode = _s1_mode() if not dbg else "0"
    if mode != "0" and q8.is_cuda and not _mx._fake() and not torch.cuda.is_current_stream_capturing():  # capture: S1 off
        main = torch.cuda.current_stream(q8.device)
        side = _side(q8.device)
        ready = torch.cuda.Event()
        ready.record(main)                                  # every input of dq (pass 0's delta included) is enqueued on main
        ins = (q8, qs, k8, ks, v8, vs, do8, dos, lse, delta)

        def _dq():
            side.wait_event(ready)
            with torch.cuda.stream(side):
                r = _mx.dm_bwd_hd256mx(*ins, causal, modes=("dq",))
            for t in ins:
                t.record_stream(side)                       # inputs read on side: no reuse before side is done
                t.record_stream(main)                       # and before KV2 on main is done even if they came from another stream
            r[0].record_stream(main)                        # side-allocated dq consumed on main
            return r[0]

        if mode == "dqfirst":
            dq = _dq()
        dk, dv, _ = kv2_launch(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, knob=knob, opts=opts)
        if mode == "kv2first":
            dq = _dq()
        main.wait_stream(side)
        return dq, dk, dv
    r = _mx.dm_bwd_hd256mx(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=dbg & ~16, modes=("dq",))
    dq, d = r[0], (r[3] if dbg & ~16 else {})
    del r                    # the dq-only call's unused (empty) dk / dv buffers go back to the allocator before KV2 allocates
    dk, dv, ex = kv2_launch(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=dbg, knob=knob, opts=opts)
    if dbg:
        d["kv2"] = ex
        return dq, dk, dv, d
    return dq, dk, dv
