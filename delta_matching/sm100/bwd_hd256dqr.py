"""Host entry of the hd256 dqr backward: KV2 (DmHd256Kv2 + opt dqw) computes dK / dV and
stores its dS codes + raw block exponents to a workspace; DmHd256Dqg then forms dQ = dS . K as a pure block-scaled GEMM (no S / dP /
exp recompute).  Same contract and values as bwd_hd256kv2.dm_bwd_hd256kv2 (dq / dk / dv bitwise).  The api DEFAULT since
DM_H256_BWD_IMPL=dqr (DM_H256_BWD_IMPL=kv2 selects the kv2 path everywhere).  Workspace = 16 KiB per visited (q tile, key tile, q head)
+ 4 B, bounded by DM_BWD_WS_LIMIT_GB (alias DM_H256_DQW_GB; GiB, default 8 as on sm90; never exceeded): the call runs in balanced contiguous batch chunks (KV2 + Dqg per chunk, one reused workspace); a single batch element above the budget -> the kv2 path
(dm_bwd_hd256kv2).  Deterministic: one TMEM accumulation chain per dQ element, no atomics."""
import torch

from . import bwd_hd256mx as _mx
from . import bwd_hd256kv2 as _kv
from .dm_bwd_hd256dqg_sm100 import DmHd256Dqg

D = 256
NUMERICS = "sm100-mx-d256"
DETERMINISTIC = True


def _cfg():
    """(scratch cap GiB, Dqg stages, KV2 store mode).  Cap: DM_BWD_WS_LIMIT_GB,
    DM_H256_DQW_GB = the older name, honoured if DM_BWD_WS_LIMIT_GB is unset."""
    from .fastpath import env as _env
    cap = _env("DM_BWD_WS_LIMIT_GB", None)
    cap = _env("DM_H256_DQW_GB", "8") if cap is None else cap
    return float(cap), int(_env("DM_H256_DQG_STAGES", "3")), _env("DM_H256_DQW_MODE", "dqt")


def _groups_mode():
    """DM_BWD_HEAD_GROUPS (as sm90 split.py): "auto" (default) = whole call, else batch chunks, else KV-head-group chunks, whichever
    is the first to fit the cap; "off" = never split heads (batch chunks only); <n> = force n KV-head groups per call."""
    from .fastpath import env as _env
    v = str(_env("DM_BWD_HEAD_GROUPS", "auto")).strip().lower()
    assert v in ("auto", "off") or v.isdigit(), v
    return v


def dqg_launch(W, Wsb, k8, ks, B, N, H, causal, *, knob="", stages=None, wsw=None, out=None, band=None):
    """band = (q0, q1): dQ of q tiles [q0, q1) only from a band-local W (KV2 opt dqc chunk); rows written at their global position."""
    Hk = k8.shape[2]
    st = _cfg()[1] if stages is None else int(stages)
    f8 = torch.float8_e4m3fn
    dq = torch.empty(B, N, H, D, device=k8.device, dtype=torch.bfloat16) if out is None else out
    assert dq.shape == (B, N, H, D) and dq.is_contiguous()
    mA = W.view(f8).permute(2, 1, 0)                       # (T, key, q) -> (q, key, T): MN-major A (queries contiguous)
    mKt = k8.view(f8).permute(3, 1, 2, 0)                  # (B, N, Hk, 256) -> (256, N, Hk, B): MN-major B (as yr / tt)
    ten = [mA, mKt, Wsb, ks, dq]
    lds = [0, 0, 2, 2, 3]
    wsw = _cfg()[2] == "dqt" if wsw is None else bool(wsw)
    bd = band is not None
    key = ("dqg", bool(causal), st, knob, wsw) + (("band",) if bd else ())
    nq, q0 = (band[1] - band[0], band[0]) if bd else (N // 128, 0)
    _mx._run(key, lambda: DmHd256Dqg(bool(causal), st, knob, wsw, bd), ten, lds, [nq, H, Hk, q0], mode="dqg")
    return dq


def _tiles(nt, q0, q1, causal):
    """W tiles of one q head over q tiles [q0, q1) (causal: key tiles 0..ct)."""
    return (q1 * (q1 + 1) - q0 * (q0 + 1)) // 2 if causal else (q1 - q0) * nt


def _ws_bytes(B, h, N, causal):
    return B * h * _tiles(N // 128, 0, N // 128, causal) * (16384 + 4)


def _tmp_bytes(nb, N, nq, ngk, carry):
    """per-chunk temporaries of a head-split chunk (contiguous copies of its nq q heads / ngk kv heads): q8 + do8 + dq (bf16) per q head,
    k8 + v8 + dk + dv (bf16) per kv head, scales / lse / delta; + the fp32 dK / dV carry of its kv heads (lives across the chunks
    of the group set).  Counted against the cap together with the workspace."""
    return nb * N * (nq * (256 * 4 + 16) + ngk * (256 * 6 + 16) + (ngk * 256 * 8 if carry else 0))


def _balanced(n, per):
    nch = -(-n // max(1, min(n, per)))
    sz = [n // nch + (1 if i < n % nch else 0) for i in range(nch)]
    out, a = [], 0
    for z in sz:
        out.append((a, a + z))
        a += z
    return out


def _bands(nt, fits):
    """greedy causal q bands [q0, q1) ascending, each the longest that fits(q0, q1); None if one q tile does not fit."""
    out, q0 = [], 0
    while q0 < nt:
        q1 = q0
        while q1 < nt and fits(q0, q1 + 1):
            q1 += 1
        if q1 == q0:
            return None
        out.append((q0, q1))
        q0 = q1
    nb = len(out)        # same band count, (near) equal causal areas (q1^2 ~ k / nb), if every band still fits
    bd = [0] + [min(nt, max(1, round(nt * (k / nb) ** 0.5))) for k in range(1, nb)] + [nt]
    eq = [(bd[k], bd[k + 1]) for k in range(nb)]
    if all(a < b_ for a, b_ in eq) and all(fits(a, b_) for a, b_ in eq):
        return eq
    return out


def _carry_plan(B, H, Hk, N, causal, cap, mode="1"):
    """exact-carry chunks (KV2 opt dqc) when one KV-head group does not fit: per batch element, a set of G KV groups (largest G that
    works), its q heads hi in ranges [h0, h1) (all q tiles) or, one q head at a time, causal q bands -- order (b, group set, hi, band)
    = the unchunked per-key-tile reduction order (q head asc, q tile asc).  Returns the plan with the fewest chunks, or None."""
    r, nt = H // Hk, N // 128
    best = None
    for G in range(Hk, 0, -1):
        if mode == "heads" and G != Hk:
            continue
        for nh in range(r - 1, 0, -1):     # head ranges, full q range
            if G * nh * _tiles(nt, 0, nt, causal) * 16388 + _tmp_bytes(1, N, G * nh, G, True) <= cap:
                hs = _balanced(r, nh)
                sub = [(h0, h1, 0, nt) for h0, h1 in hs]
                break
        else:
            if not causal or mode == "heads":
                continue
            bands = _bands(nt, lambda a, b_: G * _tiles(nt, a, b_, True) * 16388 + _tmp_bytes(1, N, G, G, True) <= cap)
            if bands is None:
                continue
            sub = [(h, h + 1, q0, q1) for h in range(r) for q0, q1 in bands]
        cand = [dict(b0=b, b1=b + 1, g0=g0, g1=g1, h0=h0, h1=h1, q0=q0, q1=q1)
                for b in range(B) for g0, g1 in _balanced(Hk, G) for h0, h1, q0, q1 in sub]
        if best is None or len(cand) < len(best):
            best = cand
    return best


def plan(B, H, Hk, N, causal):
    """Chunk plan (list of dicts b0 b1 g0 g1 h0 h1 q0 q1: batch range x KV-group range x q-head-in-group range x q-tile band), ascending,
    whose workspace (+ head-split temporaries + carry) fits the cap, or None (-> the kv2 fallback).  Order of preference (sm90 split.py
    convention): whole call, balanced batch chunks, KV-head-group chunks (no carry), then exact-carry chunks (q heads
    of a group / causal q bands).  DM_H256_DQR_CARRY: "heads" (DEFAULT: q-head-range carry only; "1" opt-in); "0" = no carry chunks (kv2 fallback above the KV-group split): at
    the 2 GiB cap no carry plan beat the kv2 path by >= 2% (32k 8/2 q-head carry x1.079 / 0.970 / 1.004 in three sweeps, fwd+bwd +0.6%;
    band plans 0.94-0.98); "heads" = carry only over q-head ranges with ALL KV groups per chunk (no bands; e.g.
    64k 8/2 at 8 GiB x1.05); "1" = every carry class (bands, group subsets).  All three are bitwise (gate classes / dqc / wlock).  Every reduction keeps its unchunked order -> bitwise."""
    from .fastpath import env as _env
    cap = _cfg()[0] * (1 << 30)
    r, nt = H // Hk, N // 128
    full = lambda b0, b1, g0, g1: dict(b0=b0, b1=b1, g0=g0, g1=g1, h0=0, h1=r, q0=0, q1=nt)  # noqa: E731
    gm = _groups_mode()
    if gm.isdigit() and int(gm) < Hk:                  # forced n KV groups per call
        ng = max(1, int(gm))
        if _ws_bytes(1, ng * r, N, causal) + _tmp_bytes(1, N, ng * r, ng, False) > cap:
            return None
        return [full(b, b + 1, g0, g1) for b in range(B) for g0, g1 in _balanced(Hk, ng)]
    per_b = _ws_bytes(1, H, N, causal)
    if per_b <= cap:                                   # whole call or balanced batch chunks (no copies)
        return [full(b0, b1, 0, Hk) for b0, b1 in _balanced(B, int(cap // per_b))]
    if gm == "off":
        return None
    for n in range(Hk - 1, 0, -1):                     # largest KV-group count per chunk that fits
        if _ws_bytes(1, n * r, N, causal) + _tmp_bytes(1, N, n * r, n, False) <= cap:
            return [full(b, b + 1, g0, g1) for b in range(B) for g0, g1 in _balanced(Hk, n)]
    cm = _env("DM_H256_DQR_CARRY", "heads")
    assert cm in ("0", "1", "heads"), cm
    if cm == "0":
        return None
    return _carry_plan(B, H, Hk, N, causal, cap, cm)


def dqw_fits(B, H, N, causal, Hk=None):
    """True if the A2 path runs (a chunk plan fits the cap).  Hk None: batch chunks only (older callers)."""
    if Hk is None:
        return _ws_bytes(1, H, N, causal) <= _cfg()[0] * (1 << 30)
    return plan(B, H, Hk, N, causal) is not None


def batch_chunks(B, H, N, causal):
    """(older callers) balanced contiguous batch chunks whose workspace fits the cap."""
    return _balanced(B, int(_cfg()[0] * (1 << 30) // _ws_bytes(1, H, N, causal)))


_DESC = {}


def _descs(rows, dev):
    """device copy of the chunk descriptors, cached per (device, rows): the plan is a pure function of (shape, cap, env), so a call
    never synchronises the host (a blocking H2D copy inside a call deadlocks harnesses that queue a host-released spin kernel ahead of
    the call -- attnbench).  The first use issues ONE pinned non-blocking copy (stream ordered); the pinned source is kept alive in the
    cache with the device tensor, and both are immutable afterwards."""
    key = (str(dev), tuple(tuple(r) for r in rows))
    if key not in _DESC:
        src = torch.tensor(rows, dtype=torch.int32).pin_memory()
        _DESC[key] = (src, src.to(dev, non_blocking=True))
    return _DESC[key][1]


_CHUNK_HOOK = None      # tests only (positive controls csk / wsk2): hook(chunk index, "chunk" | "dqg", buffers) before each chunk / Dqg


def dm_bwd_hd256dqr(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, *, dbg=0, knob="", opts=None, dqg_knob="", ch=None):
    """(dq, dk, dv) [+ dbg dict as dm_bwd_hd256kv2].  dbg not in (0, 16) or no chunk plan within the cap -> the kv2 path.
    dbg = 16: the SAME plan as dbg 0, every KV2 launch built with dbg bit 16; returns (dq, dk, dv, d) with
    d["ident"] = (B*H, N, 4) int64 rows [rowsum dS, sum|P dP|] as f64 bits (KV2 in-kernel, gathered from the chunk that ran each row)
    and d["plan"] = the chunk plan run (None = kv2 path).
    Chunks (plan()) in order: KV2(+dS workspace) then Dqg per chunk, one reused workspace (stream order = the W hand-off between
    chunks and the carry hand-off between the chunks of a group set).  ch: explicit plan (tests)."""
    B, N, H, _ = q8.shape
    Hk = k8.shape[2]
    r, nt = H // Hk, N // 128
    causal = bool(causal)
    if ch is None:
        ch = None if (dbg & ~16) else plan(B, H, Hk, N, causal)
    idk = dbg & 16                 # identity build of every KV2 launch (debug only; dbg 0 -> idk 0, nothing below changes)
    if ch is None:
        r_ = _kv.dm_bwd_hd256kv2(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=dbg, knob=knob, opts=opts)
        if idk and not (dbg & ~16):
            r_[3]["ident"] = r_[3]["kv2"]["hash"]
            r_[3]["plan"] = None
        return r_
    dev = q8.device
    mode = _cfg()[2]
    base = (_kv.DmHd256Kv2.DEFAULT_OPTS if opts is None else opts) + " " + mode
    Tmax = max((c["b1"] - c["b0"]) * (c["g1"] - c["g0"]) * (c["h1"] - c["h0"]) * _tiles(nt, c["q0"], c["q1"], causal) for c in ch)
    W = torch.empty(Tmax, 128, 128, device=dev, dtype=torch.uint8)
    Wsb = torch.empty(Tmax, 1, 4, device=dev, dtype=torch.uint8)
    dq = torch.empty(B, N, H, D, device=dev, dtype=torch.bfloat16)
    dk = torch.empty(B, N, Hk, D, device=dev, dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    carried = [c for c in ch if not (c["h0"] == 0 and c["h1"] == r and c["q0"] == 0 and c["q1"] == nt)]
    descs = carry = None
    if carried:          # one device copy of every carry chunk's descriptor [Q0, n_t, cin, cout] (stream ordered, pinned source)
        rows = []
        for c in carried:
            rows.append([c["q0"], nt, int(c["h0"] > 0), int(not (c["h1"] == r and c["q1"] == nt)), 0, 0, 0, 0])
        descs = _descs(rows, dev)
        gmax = max(c["g1"] - c["g0"] for c in carried)
        carry = (torch.empty(1, N, gmax, D, device=dev, dtype=torch.float32), torch.empty(1, N, gmax, D, device=dev, dtype=torch.float32))
    ci = ci_all = 0
    ident = torch.zeros(B * H, N, 4, device=dev, dtype=torch.int64) if idk else None
    for c in ch:
        b0, b1, g0, g1, h0, h1, q0, q1 = (c[k_] for k_ in ("b0", "b1", "g0", "g1", "h0", "h1", "q0", "q1"))
        nb, ng, nh = b1 - b0, g1 - g0, h1 - h0
        T = nb * ng * nh * _tiles(nt, q0, q1, causal)
        ws = (W[:T], Wsb[:T])
        if _CHUNK_HOOK is not None:
            _CHUNK_HOOK(ci_all, "chunk", dict(W=W, Wsb=Wsb, carry=carry))
        ci_all += 1
        if ng == Hk and nh == r and q0 == 0 and q1 == nt:     # batch chunk: contiguous batch slices, no copies
            sl = lambda x: x[b0:b1]  # noqa: E731
            ex_ = _kv.kv2_launch(*[sl(x) for x in (q8, qs, k8, ks, v8, vs, do8, dos, lse, delta)], causal, dbg=idk, knob=knob,
                                 opts=base, dqw=ws, out=(sl(dk), sl(dv)))[2]
            if idk:
                ident[b0 * H:b1 * H].copy_(ex_["hash"])
            if _CHUNK_HOOK is not None:
                _CHUNK_HOOK(ci_all - 1, "dqg", dict(W=W, Wsb=Wsb, carry=carry, chunk=c, T=T))
            dqg_launch(*ws, sl(k8), sl(ks), nb, N, H, causal, knob=dqg_knob, wsw=mode == "dqt", out=sl(dq))
            continue
        # head-split chunk: contiguous copies of its heads, group-major (g, hi) -> kernel head h = g_local * nh + hi_local
        hql = [g * r + h for g in range(g0, g1) for h in range(h0, h1)]
        hq = (torch.arange(g0, g1, device=dev)[:, None] * r + torch.arange(h0, h1, device=dev)[None, :]).reshape(-1)  # no H2D copy
        tq = lambda x: x[b0:b1].index_select(2, hq)  # noqa: E731     (B, N, h, D)
        sq = lambda x: x[b0:b1].index_select(1, hq)  # noqa: E731     (B, h, ...)
        k8c, ksc = k8[b0:b1, :, g0:g1].contiguous(), ks[b0:b1, g0:g1].contiguous()
        ins = (tq(q8), sq(qs), k8c, ksc, v8[b0:b1, :, g0:g1].contiguous(), vs[b0:b1, g0:g1].contiguous(), tq(do8), sq(dos), sq(lse),
               sq(delta))
        last = h1 == r and q1 == nt
        if nh == r and q0 == 0 and q1 == nt:      # KV-group chunk: no carry
            dkc, dvc, ex_ = _kv.kv2_launch(*ins, causal, dbg=idk, knob=knob, opts=base, dqw=ws)
        else:
            assert nb == 1
            cc = (carry[0].view(-1)[:N * ng * D].view(1, N, ng, D), carry[1].view(-1)[:N * ng * D].view(1, N, ng, D))
            dkc, dvc, ex_ = _kv.kv2_launch(*ins, causal, dbg=idk, knob=knob, opts=base + " dqc", dqw=ws,
                                           dqc=(descs[ci], cc[0], cc[1], q1))
            ci += 1
        bd = None if (q0 == 0 and q1 == nt) else (q0, q1)
        dqc_ = torch.empty(nb, N, ng * nh, D, device=dev, dtype=torch.bfloat16) if bd else None
        if _CHUNK_HOOK is not None:
            _CHUNK_HOOK(ci_all - 1, "dqg", dict(W=W, Wsb=Wsb, carry=carry, chunk=c, T=T))
        dqc_ = dqg_launch(*ws, k8c, ksc, nb, N, ng * nh, causal, knob=dqg_knob, wsw=mode == "dqt", band=bd, out=dqc_)
        rows_ = slice(q0 * 128, q1 * 128)
        for j, hh in enumerate(hql):
            dq[b0:b1, rows_, hh].copy_(dqc_[:, rows_, j])
            if idk:      # chunk row (b_local, j) of this launch's (nb * ng * nh, N, 4) identity -> global row (b, hh); only its q band
                for bi in range(nb):
                    ident[(b0 + bi) * H + hh, rows_].copy_(ex_["hash"][bi * ng * nh + j, rows_])
        if last:
            dk[b0:b1, :, g0:g1].copy_(dkc)
            dv[b0:b1, :, g0:g1].copy_(dvc)
        del dqc_, dkc, dvc, ins, k8c, ksc
    if idk:
        return dq, dk, dv, dict(ident=ident, plan=ch)
    return dq, dk, dv
