"""sm100 (Blackwell, sm_100) Delta-Matching attention at head_dim 256 through the shipped API (delta_matching.sm100.api):
  accuracy      O / dQ / dK / dV against exact attention, no worse than the torch emulation of the spec (preset sm100-mx-d256 of
                delta_matching/reference.py: MXFP8 inputs, P per (row, 32 keys), P^T per (key, 32 queries), dS per (128 x 32),
                num/den delta), matched and stale delta, MHA and GQA, causal and non-causal, odd tile counts;
  determinism   bitwise repeat, also from two caller streams;
  identity      rowsum(dS) = 0 to fp32 round-off, from the backward's own in-kernel pre-quantization dS (dm_attn_debug identity=True);
  lockstep      P / dS / dS-code hashes of the fused dK/dV kernel equal those of the dQ kernel row by row (dm_attn_debug lockstep=True);
  plan classes  the dS-workspace dQ path under every chunk plan class (batch, KV-head groups, q-head carry, causal q bands, the
                above-cap fallback) and the two other backward routes (DM_H256_BWD_IMPL=kv2 / mx) are bitwise the default call.
Needs an sm_100 GPU (B200 / GB200) with the CuTe DSL (nvidia-cutlass-dsl, flash-attn cute); skipped elsewhere.  The first run compiles
the kernels into the CuTe DSL cache (FLASH_ATTENTION_CUTE_DSL_CACHE_DIR), later runs load them (~2 minutes then)."""
import math
import os
from contextlib import contextmanager

import pytest
import torch


def _skip_reason():
    if not torch.cuda.is_available():
        return "needs a CUDA GPU"
    if torch.cuda.get_device_capability(0)[0] != 10:
        return f"sm100 kernel test: needs an sm_100 GPU; found {torch.cuda.get_device_name(0)}"
    try:
        import delta_matching.sm100.api  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"delta_matching.sm100 cannot be imported ({type(e).__name__}: {e})"
    return None


REASON = _skip_reason()
pytestmark = pytest.mark.skipif(REASON is not None, reason=REASON or "")
D = 256
ENV_KNOBS = ("DM_BWD_WS_LIMIT_GB", "DM_H256_DQW_GB", "DM_BWD_HEAD_GROUPS", "DM_H256_DQR_CARRY", "DM_H256_BWD_IMPL")


@contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in ENV_KNOBS}
    try:
        for k in ENV_KNOBS:
            os.environ.pop(k, None)
        for k, v in kw.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _inputs(B, N, H, Hk, family="randn", seed=0):
    """bf16 q, k, v, dO in the flash-attention layout.  ramp / growmax: logits that grow towards key 0 (the forward visits key tiles
    from the highest index down), so the online-softmax rescale fires on many tiles."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, do = (torch.randn(B, N, h, D, device="cuda", generator=g) for h in (H, Hk, Hk, H))
    j = torch.arange(N, device="cuda", dtype=torch.float32)
    if family == "ramp":
        q[..., 0] = 8.0
        k[..., 0] = (28.0 * (1.0 - 2.0 * j / max(N - 1, 1)))[None, :, None]
    elif family == "growmax":
        k[..., 0] = (2.5 * torch.floor(16.0 * (N - 1 - j) / N) - 20.0)[None, :, None]
        q[..., 0] = 2.0 + 12.0 * torch.rand(B, N, H, generator=g, device="cuda")
    elif family == "outliers":
        ch = torch.randint(0, D, (4,), generator=g, device="cuda")
        q[..., ch] *= 30
        k[..., ch] *= 30
        tok = torch.randint(0, N, (max(1, N // 512),), generator=g, device="cuda")
        v[:, tok] *= 100
        do[:, tok] *= 50
    else:
        assert family == "randn", family
    return [x.bfloat16().contiguous() for x in (q, k, v, do)]


def _run(q, k, v, do, causal, mode="matched"):
    from delta_matching.sm100.api import dm_attn_func
    qa, ka, va = (x.clone().requires_grad_() for x in (q, k, v))
    o = dm_attn_func(qa, ka, va, causal=causal, mode=mode, deterministic=True)
    o.backward(do)
    return [o.detach(), qa.grad, ka.grad, va.grad]


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def _bits_equal(xs, ys):
    return all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(xs, ys))


# ------------------------------------------------------------------------------------------------------------------- accuracy
EMU_DIST = {"O": 5e-3, "dQ": 1e-3, "dK": 1.2e-3, "dV": 2e-3}   # rel L2(kernel, emulation), ~3x the measured worst (O: 5e-3)


def _record(tag, case, name, v):
    """DM_TEST_RECORD=<file>: append the measured kernel-vs-emulation distances (to re-derive EMU_DIST)."""
    import json
    import os
    f = os.environ.get("DM_TEST_RECORD")
    if f:
        with open(f, "a") as fh:
            fh.write(json.dumps({"t": tag, "case": case, "out": name, "dist": v}) + "\n")


ACC_CASES = [  # B, N, H, Hk, causal, mode, family
    (1, 1024, 4, 4, True, "matched", "randn"),
    (1, 1024, 8, 2, True, "matched", "ramp"),
    (1, 1152, 8, 2, True, "matched", "growmax"),
    (2, 512, 8, 2, False, "matched", "randn"),
    (1, 1152, 4, 1, False, "matched", "outliers"),
    (1, 1024, 8, 2, True, "stale", "randn"),
    (1, 1024, 4, 4, False, "stale", "randn"),
]


@pytest.mark.parametrize("B,N,H,Hk,causal,mode,family", ACC_CASES)
def test_accuracy_vs_exact_and_spec_emulation(B, N, H, Hk, causal, mode, family):
    """Every output at most 1% (+1e-5) further from exact fp64 attention than the spec's torch emulation (same quantization points,
    same K-centering mean), close to the emulation itself (rel L2 bounds at ~3x the measured worst), and (Gaussian inputs)
    inside the fp8 floor's absolute bounds; LSE equals the emulation's to fp32 round-off (summation order only)."""
    from delta_matching import reference as R
    from delta_matching.sm100.api import dm_attn_debug
    from delta_matching.sm100.quant import quant_qkv2
    q, k, v, do = _inputs(B, N, H, Hk, family, seed=N + H + Hk)
    out = _run(q, k, v, do, causal, mode)
    assert all(torch.isfinite(t.float()).all() for t in out)
    mean = quant_qkv2(q, k, v, D ** -0.5)[7]
    em = R.dm_reference(q, k, v, do, causal, mode, "sm100-mx-d256", q_block=512, k_mean=mean,
                        o_for_stale=out[0] if mode == "stale" else None)
    ex = R.attention_fp32(q, k, v, do, causal, dtype=torch.float64)
    floor = {"O": 6e-2, "dQ": 7.5e-2, "dK": 7.5e-2, "dV": 6e-2}
    for name, got in zip(("O", "dQ", "dK", "dV"), out):
        err, emu_err = _rel(got, ex[name]), _rel(em[name].bfloat16(), ex[name])
        assert err <= 1.01 * emu_err + 1e-5, f"{name}: kernel err {err:.4e} vs spec emulation {emu_err:.4e}"
        dist = _rel(got, em[name].bfloat16())
        _record("hd256", f"{B}_{N}_{H}_{Hk}_{int(causal)}_{mode}_{family}", name, dist)
        assert dist <= EMU_DIST[name], f"{name}: kernel vs spec emulation rel L2 {dist:.3e} > {EMU_DIST[name]:.1e}"
        if family == "randn":     # the structured families sit far above the Gaussian fp8 floor in the emulation too
            assert err <= floor[name], f"{name}: err {err:.4e} above the fp8 floor bound {floor[name]}"
    lse = dm_attn_debug(q, k, v, do, causal=causal, mode=mode)["lse"]
    el = em["LSE"].double()
    assert ((lse.double() - el).abs() <= 1e-5 + 1e-6 * el.abs()).all()       # fp32 round-off at |LSE| up to ~1e3 (outliers)


# ---------------------------------------------------------------------------------------------------------------- determinism
@pytest.mark.parametrize("B,N,H,Hk,causal", [(2, 2048, 8, 2, True), (1, 1152, 16, 4, False)])
def test_bitwise_deterministic(B, N, H, Hk, causal):
    q, k, v, do = _inputs(B, N, H, Hk, seed=3)
    ref = _run(q, k, v, do, causal)
    assert _bits_equal(ref, _run(q, k, v, do, causal))
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(s1):
        a = _run(q, k, v, do, causal)
    with torch.cuda.stream(s2):
        b = _run(q, k, v, do, causal)
    torch.cuda.synchronize()
    assert _bits_equal(ref, a) and _bits_equal(ref, b)


# ------------------------------------------------------------------------------------------------------------------- identity
@pytest.mark.parametrize("family", ["randn", "ramp", "growmax"])
@pytest.mark.parametrize("B,N,H,Hk,causal", [(1, 4096, 8, 2, True), (1, 1152, 4, 1, False), (2, 1024, 16, 4, True)])
def test_zero_sum_identity(B, N, H, Hk, causal, family):
    """max_i |sum_j dS_ij| / sum_j |P_ij dP_ij| <= 1e-5 and its mean <= 2e-7, from the backward's own fp32 dS (pre-quantization)."""
    from delta_matching.sm100.api import dm_attn_debug
    q, k, v, do = _inputs(B, N, H, Hk, family, seed=11)
    d = dm_attn_debug(q, k, v, do, causal=causal, mode="matched", identity=True)
    den = d["sum_abs_PdP"].double()
    live = den > 0
    assert live.float().mean().item() > 0.99
    rel = (d["rowsum_dS"].double().abs() / den.clamp_min(1e-300))[live]
    assert rel.max().item() <= 1e-5 and rel.mean().item() <= 2e-7, (rel.max().item(), rel.mean().item())
    assert torch.isfinite(d["delta"]).all() and torch.isfinite(d["lse"]).all()


# ------------------------------------------------------------------------------------------------------------------- lockstep
@pytest.mark.parametrize("B,N,H,Hk,causal", [(1, 2048, 8, 2, True), (1, 1152, 4, 1, False)])
def test_lockstep_dkdv_and_dq_kernels(B, N, H, Hk, causal):
    """The fused dK/dV kernel (KV2) and the dQ kernel compute bitwise the same P, pre-quantization dS and dS codes + scales
    (per-row order-free hashes of every (query, key) value); the dK kernel agrees with the dQ kernel too."""
    from delta_matching.sm100.api import dm_attn_debug
    q, k, v, do = _inputs(B, N, H, Hk, "ramp", seed=5)
    d = dm_attn_debug(q, k, v, do, causal=causal, mode="matched", lockstep=True)
    kv, dq, dk = d["lockstep_kv2_kernel"], d["lockstep_dq_kernel"], d["lockstep_dk_kernel"]
    for h in ("P_hash", "dS_hash", "code_sf_hash"):
        assert int((kv[h] != dq[h]).sum()) == 0, h
        assert int((kv[h] != 0).sum()) > 0, h
    for h in ("dS_hash", "code_sf_hash"):
        assert int((dk[h] != dq[h]).sum()) == 0, h


# ------------------------------------------------------------------------------------------------------------- plan classes
def _plan_class(B, H, Hk, N, causal):
    from delta_matching.sm100 import bwd_hd256dqr as Q
    p = Q.plan(B, H, Hk, N, causal)
    if p is None:
        return "fallback"
    r, nt = H // Hk, N // 128
    cl = set()
    for c in p:
        if c["q0"] != 0 or c["q1"] != nt:
            cl.add("band")
        elif c["h1"] - c["h0"] != r:
            cl.add("qhead")
        elif c["g1"] - c["g0"] != Hk:
            cl.add("kvgroup")
        else:
            cl.add("batch" if len(p) > 1 else "whole")
    return "+".join(sorted(cl))


def _cap_for(cls, B, H, Hk, N, causal, carry):
    """largest workspace cap (GiB, from a fine geometric grid) whose chunk plan is of class cls."""
    from delta_matching.sm100 import bwd_hd256dqr as Q
    top = Q._ws_bytes(B, H, N, causal) / 2 ** 30
    for i in range(400):
        cap = top * 0.98 ** i
        with _env(DM_BWD_WS_LIMIT_GB=repr(cap), DM_H256_DQR_CARRY=carry):
            if _plan_class(B, H, Hk, N, causal) == cls:
                return cap
    return None


PLAN_SHAPE = (2, 4096, 8, 2)


@pytest.mark.parametrize("causal", [True, False])
def test_plan_classes_bitwise(causal):
    """dQ / dK / dV / O bitwise the default call (whole-call workspace) under every chunk plan the planner emits for this shape, and
    under the other two backward routes.  Each reduction keeps its unchunked order, so every plan is exact, not just close."""
    B, N, H, Hk = PLAN_SHAPE
    q, k, v, do = _inputs(B, N, H, Hk, "randn", seed=21)
    with _env():
        assert _plan_class(B, H, Hk, N, causal) == "whole"
        ref = _run(q, k, v, do, causal)
    arms = [("batch", "heads"), ("kvgroup", "heads"), ("qhead", "heads")] + ([("band", "1")] if causal else [])
    seen = []
    for cls, carry in arms:
        cap = _cap_for(cls, B, H, Hk, N, causal, carry)
        assert cap is not None, f"no cap yields a {cls} plan"
        with _env(DM_BWD_WS_LIMIT_GB=repr(cap), DM_H256_DQR_CARRY=carry):
            assert _bits_equal(ref, _run(q, k, v, do, causal)), f"{cls} plan (cap {cap:.4f} GiB) differs from the default"
        seen.append(cls)
    with _env(DM_BWD_WS_LIMIT_GB="0.001", DM_H256_DQR_CARRY="0"):   # nothing fits: the kv2 route above the cap
        assert _plan_class(B, H, Hk, N, causal) == "fallback"
        assert _bits_equal(ref, _run(q, k, v, do, causal)), "above-cap fallback differs from the default"
    with _env(DM_BWD_HEAD_GROUPS="1"):
        assert _plan_class(B, H, Hk, N, causal) == "kvgroup"
        assert _bits_equal(ref, _run(q, k, v, do, causal)), "DM_BWD_HEAD_GROUPS=1 differs from the default"
    for impl in ("kv2", "mx"):
        with _env(DM_H256_BWD_IMPL=impl):
            assert _bits_equal(ref, _run(q, k, v, do, causal)), f"DM_H256_BWD_IMPL={impl} differs from the default"
    assert len(seen) == len(arms)


def test_input_checks():
    from delta_matching.sm100.api import dm_attn_func
    q, k, v, _ = _inputs(1, 256, 4, 4)
    with pytest.raises(AssertionError):
        dm_attn_func(q[:, :200], k[:, :200], v[:, :200])                  # N % 128 != 0
    with pytest.raises(AssertionError):
        dm_attn_func(q, k, v, mode="exact")
    assert math.isclose(D ** -0.5, 2 ** -4)
