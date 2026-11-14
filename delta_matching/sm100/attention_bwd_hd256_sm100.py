"""
hd256 backward launcher. The fused dK / dV kernel KV2 stores its dS codes and
block exponents to a workspace, then the dQ GEMM kernel Dqg forms dQ = dS K from
them. The workspace is bounded by DM_BWD_WS_LIMIT_GB (GiB, default 8;
DM_H256_DQW_GB is the older name): the call runs as a chunk plan whose workspace
fits, and when no plan fits it takes the kv2 path (KV2 for dK / dV and the Mx2
dQ kernel, run concurrently). Every plan keeps each reduction in its unchunked
order, so all paths give bitwise the same dQ / dK / dV, deterministically and
without any atomics.
"""
import torch
from flash_attn.cute.testing import is_fake_mode

from . import bwd_hd256mx
from . import bwd_hd256kv2
from .dm_bwd_hd256dqg_sm100 import DmHd256Dqg
from .launch_cache_sm100 import env

__all__ = ["attention_bwd_hd256", "plan_backward_chunks", "dqg_launch"]

D = 256
TILE_BYTES = 16384 + 4                  # one workspace tile: 128 x 128 dS codes + 4 block-exponent bytes
DQG_STAGES = 3


def workspace_limit_bytes():
    """
    Workspace cap: DM_BWD_WS_LIMIT_GB, else the older name DM_H256_DQW_GB, else 8 GiB.
    """
    cap = env("DM_BWD_WS_LIMIT_GB", None)
    cap = env("DM_H256_DQW_GB", "8") if cap is None else cap
    return float(cap) * (1 << 30)


def head_groups_setting():
    """
    DM_BWD_HEAD_GROUPS: "auto" (default) splits KV-head groups only when a batch element does not
    fit, "off" never splits heads, a number forces that many KV-head groups per chunk.
    """
    value = str(env("DM_BWD_HEAD_GROUPS", "auto")).strip().lower()
    assert value in ("auto", "off") or value.isdigit(), value
    return value


def dqg_launch(W, Wsb, k8, ks, B, N, H, causal, *, out=None, band=None):
    """
    dQ (B, N, H, 256) bf16 from the workspace (W codes, Wsb block exponents) and K. With band =
    (q_start, q_end), only those q tiles, from a band-local workspace, written at their global rows.
    """
    Hk = k8.shape[2]
    f8 = torch.float8_e4m3fn
    dq = torch.empty(B, N, H, D, device=k8.device, dtype=torch.bfloat16) if out is None else out
    assert dq.shape == (B, N, H, D) and dq.is_contiguous()
    mA = W.view(f8).permute(2, 1, 0)                       # (T, key, q) -> (q, key, T): MN-major A (queries contiguous)
    mKt = k8.view(f8).permute(3, 1, 2, 0)                  # (B, N, Hk, 256) -> (256, N, Hk, B): MN-major B
    tensors = [mA, mKt, Wsb, ks, dq]
    leading_dims = [0, 0, 2, 2, 3]
    has_band = band is not None
    key = ("dqg", bool(causal), DQG_STAGES, "", True) + (("band",) if has_band else ())
    num_q_tiles, q_start = (band[1] - band[0], band[0]) if has_band else (N // 128, 0)
    bwd_hd256mx._run(key, lambda: DmHd256Dqg(bool(causal), DQG_STAGES, "", True, has_band), tensors, leading_dims, [num_q_tiles, H, Hk, q_start], mode="dqg")
    return dq


def num_workspace_tiles(num_q_tiles, q_start, q_end, causal):
    """
    Workspace tiles of one q head over q tiles [q_start, q_end) (causal: the lower triangle).
    """
    return (q_end * (q_end + 1) - q_start * (q_start + 1)) // 2 if causal else (q_end - q_start) * num_q_tiles


def workspace_bytes(B, num_heads, N, causal):
    return B * num_heads * num_workspace_tiles(N // 128, 0, N // 128, causal) * TILE_BYTES


def chunk_copy_bytes(num_batch, N, num_q_heads, num_kv_heads, has_carry):
    """
    Temporaries of a head-split chunk, counted against the cap with the workspace: contiguous copies
    of its q heads (q8, do8, bf16 dq) and kv heads (k8, v8, bf16 dk, dv), scales / lse / delta, and
    the fp32 dK / dV carry of its kv heads.
    """
    return num_batch * N * (num_q_heads * (256 * 4 + 16) + num_kv_heads * (256 * 6 + 16) + (num_kv_heads * 256 * 8 if has_carry else 0))


def balanced_ranges(n, max_per_range):
    """
    [0, n) as contiguous ranges of near-equal size, each at most max_per_range long.
    """
    num_ranges = -(-n // max(1, min(n, max_per_range)))
    sizes = [n // num_ranges + (1 if i < n % num_ranges else 0) for i in range(num_ranges)]
    ranges, start = [], 0
    for size in sizes:
        ranges.append((start, start + size))
        start += size
    return ranges


def causal_bands(num_q_tiles, fits):
    """
    Ascending causal q bands [q_start, q_end), each the longest that fits(q_start, q_end); None if
    one q tile does not fit. When the same number of bands with near-equal causal areas all fit,
    those are returned instead.
    """
    bands, q_start = [], 0
    while q_start < num_q_tiles:
        q_end = q_start
        while q_end < num_q_tiles and fits(q_start, q_end + 1):
            q_end += 1
        if q_end == q_start:
            return None
        bands.append((q_start, q_end))
        q_start = q_end
    num_bands = len(bands)
    bounds = [0] + [min(num_q_tiles, max(1, round(num_q_tiles * (k / num_bands) ** 0.5))) for k in range(1, num_bands)] + [num_q_tiles]
    equal = [(bounds[k], bounds[k + 1]) for k in range(num_bands)]
    if all(a < b for a, b in equal) and all(fits(a, b) for a, b in equal):
        return equal
    return bands


def carry_plan(B, H, Hk, N, causal, cap, carry_mode="1"):
    """
    Exact-carry chunks (KV2 dqc) for when one KV-head group does not fit: per batch element, a set
    of KV groups, its q heads in ranges (all q tiles) or, one q head at a time, causal q bands. The
    chunk order (batch, group set, q head, band) is the unchunked reduction order. Returns the plan
    with the fewest chunks, or None.
    """
    heads_per_group, num_q_tiles = H // Hk, N // 128
    best = None
    for num_groups in range(Hk, 0, -1):
        if carry_mode == "heads" and num_groups != Hk:
            continue
        for heads_per_chunk in range(heads_per_group - 1, 0, -1):
            if num_groups * heads_per_chunk * num_workspace_tiles(num_q_tiles, 0, num_q_tiles, causal) * TILE_BYTES + chunk_copy_bytes(1, N, num_groups * heads_per_chunk, num_groups, True) <= cap:
                sub = [(h_start, h_end, 0, num_q_tiles) for h_start, h_end in balanced_ranges(heads_per_group, heads_per_chunk)]
                break
        else:
            if not causal or carry_mode == "heads":
                continue
            bands = causal_bands(num_q_tiles, lambda a, b, g=num_groups: g * num_workspace_tiles(num_q_tiles, a, b, True) * TILE_BYTES + chunk_copy_bytes(1, N, g, g, True) <= cap)
            if bands is None:
                continue
            sub = [(h, h + 1, q_start, q_end) for h in range(heads_per_group) for q_start, q_end in bands]
        chunks = [dict(b0=b, b1=b + 1, g0=g0, g1=g1, h0=h0, h1=h1, q0=q0, q1=q1)
                  for b in range(B) for g0, g1 in balanced_ranges(Hk, num_groups) for h0, h1, q0, q1 in sub]
        if best is None or len(chunks) < len(best):
            best = chunks
    return best


def plan_backward_chunks(B, H, Hk, N, causal):
    """
    Chunk plan (ascending list of dicts b0 b1 g0 g1 h0 h1 q0 q1: batch range x KV-group range x
    q-head-in-group range x q-tile band) whose workspace, head-split copies and carry fit the cap,
    or None (the kv2 path). Preference: the whole call, balanced batch chunks, KV-head-group chunks,
    then exact-carry chunks. DM_H256_DQR_CARRY selects the carry classes: "heads" (default) q-head
    ranges with all KV groups, "1" also causal bands and group subsets, "0" none.
    """
    cap = workspace_limit_bytes()
    heads_per_group, num_q_tiles = H // Hk, N // 128

    def whole_heads(b0, b1, g0, g1):
        return dict(b0=b0, b1=b1, g0=g0, g1=g1, h0=0, h1=heads_per_group, q0=0, q1=num_q_tiles)

    groups = head_groups_setting()
    if groups.isdigit() and int(groups) < Hk:
        num_groups = max(1, int(groups))
        if workspace_bytes(1, num_groups * heads_per_group, N, causal) + chunk_copy_bytes(1, N, num_groups * heads_per_group, num_groups, False) > cap:
            return None
        return [whole_heads(b, b + 1, g0, g1) for b in range(B) for g0, g1 in balanced_ranges(Hk, num_groups)]
    per_batch = workspace_bytes(1, H, N, causal)
    if per_batch <= cap:
        return [whole_heads(b0, b1, 0, Hk) for b0, b1 in balanced_ranges(B, int(cap // per_batch))]
    if groups == "off":
        return None
    for num_groups in range(Hk - 1, 0, -1):
        if workspace_bytes(1, num_groups * heads_per_group, N, causal) + chunk_copy_bytes(1, N, num_groups * heads_per_group, num_groups, False) <= cap:
            return [whole_heads(b, b + 1, g0, g1) for b in range(B) for g0, g1 in balanced_ranges(Hk, num_groups)]
    carry_mode = env("DM_H256_DQR_CARRY", "heads")
    assert carry_mode in ("0", "1", "heads"), carry_mode
    if carry_mode == "0":
        return None
    return carry_plan(B, H, Hk, N, causal, cap, carry_mode)


CHUNK_DESCRIPTORS = {}


def chunk_descriptors(rows, device):
    """
    Device copy of the carry chunks' descriptors, cached per (device, rows). The plan is a pure
    function of (shape, cap, env), so a call never blocks the host: the first use issues one pinned
    non-blocking copy, and the pinned source stays alive with the device tensor (a blocking copy
    inside a call deadlocks harnesses that queue a host-released kernel ahead of it).
    """
    key = (str(device), tuple(tuple(r) for r in rows))
    if key not in CHUNK_DESCRIPTORS:
        src = torch.tensor(rows, dtype=torch.int32).pin_memory()
        CHUNK_DESCRIPTORS[key] = (src, src.to(device, non_blocking=True))
    return CHUNK_DESCRIPTORS[key][1]


SIDE_STREAMS = {}


def side_stream(device):
    key = str(device)
    if key not in SIDE_STREAMS:
        SIDE_STREAMS[key] = torch.cuda.Stream(device=device)
    return SIDE_STREAMS[key]


def backward_kv2_path(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, identity=False):
    """
    The path without a workspace: KV2 for dK / dV and the Mx2 dQ kernel. Both only read the
    quantized inputs, lse and delta and write disjoint outputs, so dQ runs on a side stream after
    KV2 is enqueued and each fills the other's tail SMs; the caller stream joins the side stream
    before returning. identity runs them serially with the identity build of KV2.
    """
    inputs = (q8, qs, k8, ks, v8, vs, do8, dos, lse, delta)
    if not identity and q8.is_cuda and not is_fake_mode() and not torch.cuda.is_current_stream_capturing():
        main = torch.cuda.current_stream(q8.device)
        side = side_stream(q8.device)
        ready = torch.cuda.Event()
        ready.record(main)                                  # every input of dQ (pass 0's delta included) is enqueued on main
        dk, dv, _ = bwd_hd256kv2.kv2_launch(*inputs, causal)
        side.wait_event(ready)
        with torch.cuda.stream(side):
            side_result = bwd_hd256mx.dm_bwd_hd256mx(*inputs, causal, modes=("dq",))
        dq = side_result[0]
        for t in inputs:
            t.record_stream(side)                           # inputs read on side: no reuse before side is done
            t.record_stream(main)                           # and before KV2 on main is done even if they came from another stream
        dq.record_stream(main)                              # side-allocated dq consumed on main
        del side_result
        main.wait_stream(side)
        return dq, dk, dv
    result = bwd_hd256mx.dm_bwd_hd256mx(*inputs, causal, modes=("dq",))
    dq = result[0]
    del result               # the dq-only call's unused dk / dv buffers go back to the allocator before KV2 allocates
    dk, dv, extra = bwd_hd256kv2.kv2_launch(*inputs, causal, dbg=16 if identity else 0)
    if identity:
        return dq, dk, dv, extra["hash"]
    return dq, dk, dv


def attention_bwd_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, identity=False):
    """
    (dq (B, N, H, 256), dk, dv (B, N, Hk, 256)) bf16. q8 / do8 (B, N, H, 256) e4m3 with qs / dos (B,
    H, N / 32) pow2 scales, k8 / v8 (B, N, Hk, 256) e4m3 with ks / vs (B, Hk, N / 32), lse / delta
    (B, H, N) fp32. identity: also returns (B * H, N, 4) int64 rows whose first two words are the
    fp64 bits of the backward's in-kernel rowsum(dS) and sum |P dP|.
    """
    B, N, H, _ = q8.shape
    Hk = k8.shape[2]
    heads_per_group, num_q_tiles = H // Hk, N // 128
    causal = bool(causal)
    chunks = plan_backward_chunks(B, H, Hk, N, causal)
    if chunks is None:
        return backward_kv2_path(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, identity)
    device = q8.device
    debug_bits = 16 if identity else 0
    opts = bwd_hd256kv2.DmHd256Kv2.DEFAULT_OPTS + " dqt"
    max_tiles = max((c["b1"] - c["b0"]) * (c["g1"] - c["g0"]) * (c["h1"] - c["h0"]) * num_workspace_tiles(num_q_tiles, c["q0"], c["q1"], causal) for c in chunks)
    W = torch.empty(max_tiles, 128, 128, device=device, dtype=torch.uint8)
    Wsb = torch.empty(max_tiles, 1, 4, device=device, dtype=torch.uint8)
    dq = torch.empty(B, N, H, D, device=device, dtype=torch.bfloat16)
    dk = torch.empty(B, N, Hk, D, device=device, dtype=torch.bfloat16)
    dv = torch.empty_like(dk)
    carried = [c for c in chunks if not (c["h0"] == 0 and c["h1"] == heads_per_group and c["q0"] == 0 and c["q1"] == num_q_tiles)]
    descriptors = carry = None
    if carried:          # one device copy of every carry chunk's descriptor [q_start, n_t, carry in, carry out]
        rows = []
        for c in carried:
            rows.append([c["q0"], num_q_tiles, int(c["h0"] > 0), int(not (c["h1"] == heads_per_group and c["q1"] == num_q_tiles)), 0, 0, 0, 0])
        descriptors = chunk_descriptors(rows, device)
        max_groups = max(c["g1"] - c["g0"] for c in carried)
        carry = (torch.empty(1, N, max_groups, D, device=device, dtype=torch.float32), torch.empty(1, N, max_groups, D, device=device, dtype=torch.float32))
    carry_idx = 0
    ident = torch.zeros(B * H, N, 4, device=device, dtype=torch.int64) if identity else None
    for c in chunks:
        b0, b1, g0, g1, h0, h1, q0, q1 = (c[k] for k in ("b0", "b1", "g0", "g1", "h0", "h1", "q0", "q1"))
        num_batch, num_groups, num_heads = b1 - b0, g1 - g0, h1 - h0
        num_tiles = num_batch * num_groups * num_heads * num_workspace_tiles(num_q_tiles, q0, q1, causal)
        workspace = (W[:num_tiles], Wsb[:num_tiles])
        if num_groups == Hk and num_heads == heads_per_group and q0 == 0 and q1 == num_q_tiles:     # batch chunk: slices, no copies
            def batch_slice(x):
                return x[b0:b1]
            extra = bwd_hd256kv2.kv2_launch(*[batch_slice(x) for x in (q8, qs, k8, ks, v8, vs, do8, dos, lse, delta)], causal, dbg=debug_bits, knob="",
                                            opts=opts, dqw=workspace, out=(batch_slice(dk), batch_slice(dv)))[2]
            if identity:
                ident[b0 * H:b1 * H].copy_(extra["hash"])
            dqg_launch(*workspace, batch_slice(k8), batch_slice(ks), num_batch, N, H, causal, out=batch_slice(dq))
            continue
        # head-split chunk: contiguous copies of its heads, group-major (g, hi) -> kernel head g_local * num_heads + hi_local
        q_heads = [g * heads_per_group + h for g in range(g0, g1) for h in range(h0, h1)]
        q_head_index = (torch.arange(g0, g1, device=device)[:, None] * heads_per_group + torch.arange(h0, h1, device=device)[None, :]).reshape(-1)  # no H2D copy

        def take_tokens(x):
            return x[b0:b1].index_select(2, q_head_index)          # (B, N, h, D)

        def take_heads(x):
            return x[b0:b1].index_select(1, q_head_index)          # (B, h, ...)
        k8_chunk, ks_chunk = k8[b0:b1, :, g0:g1].contiguous(), ks[b0:b1, g0:g1].contiguous()
        chunk_inputs = (take_tokens(q8), take_heads(qs), k8_chunk, ks_chunk, v8[b0:b1, :, g0:g1].contiguous(), vs[b0:b1, g0:g1].contiguous(),
                        take_tokens(do8), take_heads(dos), take_heads(lse), take_heads(delta))
        is_last = h1 == heads_per_group and q1 == num_q_tiles
        if num_heads == heads_per_group and q0 == 0 and q1 == num_q_tiles:      # KV-group chunk: no carry
            dk_chunk, dv_chunk, extra = bwd_hd256kv2.kv2_launch(*chunk_inputs, causal, dbg=debug_bits, knob="", opts=opts, dqw=workspace)
        else:
            assert num_batch == 1
            size = N * num_groups * D
            carry_views = (carry[0].view(-1)[:size].view(1, N, num_groups, D), carry[1].view(-1)[:size].view(1, N, num_groups, D))
            dk_chunk, dv_chunk, extra = bwd_hd256kv2.kv2_launch(*chunk_inputs, causal, dbg=debug_bits, knob="", opts=opts + " dqc", dqw=workspace,
                                                                dqc=(descriptors[carry_idx], carry_views[0], carry_views[1], q1))
            carry_idx += 1
        band = None if (q0 == 0 and q1 == num_q_tiles) else (q0, q1)
        dq_chunk = torch.empty(num_batch, N, num_groups * num_heads, D, device=device, dtype=torch.bfloat16) if band else None
        dq_chunk = dqg_launch(*workspace, k8_chunk, ks_chunk, num_batch, N, num_groups * num_heads, causal, band=band, out=dq_chunk)
        rows_ = slice(q0 * 128, q1 * 128)
        for j, head in enumerate(q_heads):
            dq[b0:b1, rows_, head].copy_(dq_chunk[:, rows_, j])
            if identity:      # chunk row (b_local, j) -> global row (b, head); only its q band
                for bi in range(num_batch):
                    ident[(b0 + bi) * H + head, rows_].copy_(extra["hash"][bi * num_groups * num_heads + j, rows_])
        if is_last:
            dk[b0:b1, :, g0:g1].copy_(dk_chunk)
            dv[b0:b1, :, g0:g1].copy_(dv_chunk)
        del dq_chunk, dk_chunk, dv_chunk, chunk_inputs, k8_chunk, ks_chunk
    if identity:
        return dq, dk, dv, ident
    return dq, dk, dv
