"""The backward on head groups (long context): the backward op materializes its dS' tiles in fp8 over [b*h, s_q, s_k]
(1 GiB at 8k for B1 H16, 64 GiB at 64k), so the call is split per KV-head group or per query head when that workspace
would exceed DM_BWD_WS_LIMIT_GB.  Heads are independent inside the ops (per-query-head dk/dv, fixed-order per-head
reductions), so the concatenation of the per-group results is bitwise the single call's result; in the matched mode the
delta is taken per group as well (pass 0 per group, bitwise the all-heads delta)."""
from __future__ import annotations

import os as _os

import torch

# The backward op materializes its dS' tiles in fp8 over the whole [b*h, s_q, s_k] (1 GiB at
# 8k, 64 GiB at 64k). DM_BWD_HEAD_GROUPS = "auto" (default) calls the op per KV-head group (n_rep query heads) or per
# single query head when the whole-call workspace would exceed DM_BWD_WS_LIMIT_GB (8); "off" never chunks; "kv" /
# "head" force the split. Heads are independent inside the op (per-query-head dk/dv, fixed-order per-head reductions), so
# the concatenation of the per-group results is the single call's result; an 8k call (1 GiB at 16 heads) is not split.
_BWD_GROUPS = _os.environ.get("DM_BWD_HEAD_GROUPS", "auto").strip().lower()
_BWD_WS_LIMIT = float(_os.environ.get("DM_BWD_WS_LIMIT_GB", "8")) * 2 ** 30
# OPT-IN (default 1 = the sequential path above): DM_BWD_STREAMS=K runs the per-
# head-group backward calls round-robin on K CUDA streams, and under the matched delta computes each group's pass 0
# (delta_fp8) on that group's stream right before its pass 1 + pass 2. Heads are independent inside every op and each op
# is deterministic, so the per-group results are bitwise those of the sequential path (tests/test_api.py checks
# it); the streams only let the issue-bound pass 0 and the DRAM-bound pass 2 of one
# group overlap the latency-bound pass 1 of another. Memory: each stream holds its own dS' workspace (B*hp*N*N bytes).
_BWD_STREAMS = max(1, int(_os.environ.get("DM_BWD_STREAMS", "1")))
_bwd_stream_pool = {}


def _bwd_streams(device):
    key = (device.type, device.index)
    if key not in _bwd_stream_pool:
        _bwd_stream_pool[key] = [torch.cuda.Stream(device=device) for _ in range(_BWD_STREAMS)]
    return _bwd_stream_pool[key]


def _bwd_heads_per_call(B, H, Hkv, n_rep, N):
    """Query heads per backward call: H (no split), n_rep (per KV group) or 1 (per head)."""
    ws = float(B) * N * N
    if _BWD_GROUPS in ("off", "0", "1"):
        return H
    if _BWD_GROUPS == "kv":
        return n_rep
    if _BWD_GROUPS == "head":
        return 1
    for hp in (H, n_rep, 1):
        if ws * hp <= _BWD_WS_LIMIT:
            return hp
    return 1


def _bwd_by_head_groups(ops, a, sm, B, H, Hkv, n_rep, N, D, hp, matched_delta=False, causal=True):
    """ops.bwd_fp8_t on hp query heads at a time (hp = n_rep or 1); layouts as bwd_args: dO8/Q8 (B, N, H, D),
    K8kv/V8kv (B, N, Hkv, D), Lse/Dlt (B, H, N), kTtkv (B, Hkv, D, N), Qt/dOt (B, H, D, N), Q8s/dO8s (B, H, nb),
    K8skv/V8skv (B, Hkv, nb); the op returns dq (B, N, h, D), dk/dv (B, N, h, D) per query head.
    matched_delta=True: the group's delta is ops.delta_fp8 on the group (pass 0 per group; Dlt in `a` is ignored), which
    is bitwise the all-heads delta_fp8 (per-head fixed-order reductions). With DM_BWD_STREAMS=K > 1 the groups run
    round-robin on K streams (see the note at _BWD_STREAMS)."""
    dO8, Q8, K8kv, V8kv, Lse, Dlt, kTtkv, Qt, dOt, Q8s, K8skv, V8skv, dO8s = a
    Lse3 = Lse.reshape(B, H, N)
    Dlt3 = Dlt.reshape(B, H, N) if Dlt is not None else None     # None on the per-group matched path (pass 0 per group below)
    assert matched_delta or Dlt3 is not None, "the all-heads delta is required unless the delta is taken per group"
    groups = list(range(0, H, hp))
    K = min(_BWD_STREAMS, len(groups))
    cur = torch.cuda.current_stream(dO8.device)
    streams = _bwd_streams(dO8.device)[:K] if K > 1 else [cur]
    if K > 1:
        ready = torch.cuda.Event(); ready.record(cur)
        for s in streams:
            s.wait_event(ready)
    outs = []
    for i, h0 in enumerate(groups):
        hs = slice(h0, h0 + hp); g = h0 // n_rep; ks = slice(g, g + 1)
        assert (h0 + hp - 1) // n_rep == g, "a call must stay inside one KV group"
        s = streams[i % K]
        with torch.cuda.stream(s):
            dO8_g, Q8_g, K8_g, V8_g = dO8[:, :, hs].contiguous(), Q8[:, :, hs].contiguous(), K8kv[:, :, ks].contiguous(), V8kv[:, :, ks].contiguous()
            Lse_g, kTt_g = Lse3[:, hs].contiguous(), kTtkv[:, ks].contiguous()
            Q8s_g, K8s_g, V8s_g, dO8s_g = Q8s[:, hs].contiguous(), K8skv[:, ks].contiguous(), V8skv[:, ks].contiguous(), dO8s[:, hs].contiguous()
            if matched_delta:
                Dlt_g = ops.delta_fp8(dO8_g, Q8_g, K8_g, V8_g, Lse_g, kTt_g, Q8s_g, K8s_g, V8s_g, dO8s_g, causal).view(B, hp, N)
            else:
                Dlt_g = Dlt3[:, hs].contiguous()
            out = ops.bwd_fp8_t(dO8_g, Q8_g, K8_g, V8_g, Lse_g, Dlt_g, kTt_g, Qt[:, hs].contiguous(),
                                dOt[:, hs].contiguous(), Q8s_g, K8s_g, V8s_g, dO8s_g, sm, causal)
            if K > 1:
                for t in out:
                    t.record_stream(cur)
            outs.append(out)
    if K > 1:
        for s in streams:
            cur.wait_stream(s)
    return tuple(torch.cat([o[i] for o in outs], dim=2) for i in range(3))
