"""sm100 (Blackwell, sm_100a) head_dim 128 kernels (delta_matching/sm100): dm_attn_func against the pure-torch emulation of
their numerics (reference preset sm100-mx) and exact attention, the Delta-Matching identity measured inside the backward,
determinism and strided inputs.  Needs a Blackwell GPU, nvidia-cutlass-dsl and FlashAttention-4's CuTe DSL package
(flash_attn.cute); skipped otherwise.  The first run JIT-compiles the kernels into the CuTe DSL cache (minutes)."""
import importlib.util
import os

import pytest
import torch


def _reason():
    if not torch.cuda.is_available():
        return "needs a CUDA GPU"
    if torch.cuda.get_device_capability(0)[0] != 10:
        return f"sm100 kernel test: needs a Blackwell GPU (sm_100); found {torch.cuda.get_device_name(0)}"
    for m in ("cutlass", "flash_attn.cute"):
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            return f"sm100 kernel test: cannot import {m} ({type(e).__name__}: {e})"
    return None


REASON = _reason()
pytestmark = pytest.mark.skipif(REASON is not None, reason=REASON or "")

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "delta_matching", "reference.py")
_S = importlib.util.spec_from_file_location("dm_reference_sm100_test", _P)
R = importlib.util.module_from_spec(_S)
_S.loader.exec_module(R)

D = 128
BF = torch.bfloat16
NAMES = ("O", "dQ", "dK", "dV")


def _api():
    from delta_matching.sm100 import api
    return api


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def inputs(B, N, H, Hk, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(B, N, h, D, generator=g).to("cuda", BF) for h in (H, Hk, Hk, H)]


def run(q, k, v, do, causal, mode, deterministic=True):
    leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    o = _api().dm_attn_func(*leaves, causal=causal, mode=mode, deterministic=deterministic)
    o.backward(do)
    return {"O": o.detach(), "dQ": leaves[0].grad, "dK": leaves[1].grad, "dV": leaves[2].grad}


# B, N, H, Hk, causal: MHA / GQA / MQA, short sequences (one 128-row tile), odd numbers of 128-row blocks
EMU_DIST = {"O": 3e-4, "dQ": 5e-3, "dK": 5e-3, "dV": 5e-3}   # rel L2(kernel, emulation), ~3x the measured worst


def _record(case, name, v):
    """DM_TEST_RECORD=<file>: append the measured kernel-vs-emulation distances (to re-derive EMU_DIST)."""
    import json
    f = os.environ.get("DM_TEST_RECORD")
    if f:
        with open(f, "a") as fh:
            fh.write(json.dumps({"t": "hd128", "case": case, "out": name, "dist": v}) + "\n")


SHAPES = [(1, 128, 2, 2, True), (2, 256, 4, 1, True), (1, 384, 4, 4, False), (1, 1152, 4, 4, True),
          (2, 1408, 8, 2, False), (1, 2048, 8, 2, True)]


@pytest.mark.parametrize("mode", ["matched", "stale"])
@pytest.mark.parametrize("B,N,H,Hk,causal", SHAPES)
def test_matches_emulation(B, N, H, Hk, causal, mode):
    """Kernel outputs are at most 1% (+1e-5) further from exact attention than the emulation of the sm100-mx numerics (on the
    kernel's own K-centering mean), and close to the emulation itself (rel L2 bounds at ~3x the measured worst)."""
    q, k, v, do = inputs(B, N, H, Hk)
    got = run(q, k, v, do, causal, mode)
    for nm in NAMES:
        assert torch.isfinite(got[nm]).all(), nm
    kmean = _api().quant_qkv2(q, k, v, D ** -0.5)[7]
    em = R.dm_reference(q, k, v, do, causal, mode, "sm100-mx", k_mean=kmean,
                        o_for_stale=got["O"] if mode == "stale" else None)
    ex = R.attention_fp32(q, k, v, do, causal)
    for nm in NAMES:
        e_c, e_e = rel(got[nm], ex[nm]), rel(em[nm], ex[nm])
        assert e_c <= 1.01 * e_e + 1e-5, f"{nm}: err vs exact {e_c:.3e}, emulation {e_e:.3e}"
        dist = rel(got[nm], em[nm])
        _record(f"{B}_{N}_{H}_{Hk}_{int(causal)}_{mode}", nm, dist)
        assert dist <= EMU_DIST[nm], f"{nm}: dist to emulation {dist:.3e} > {EMU_DIST[nm]:.1e}"


@pytest.mark.parametrize("B,N,H,Hk,causal", [(1, 1024, 4, 4, True), (1, 2048, 8, 2, True), (1, 1152, 4, 4, False)])
def test_identity_in_kernel(B, N, H, Hk, causal):
    """Matched mode: the backward's own per-row |sum_j dS_ij| / sum_j |P dP| (accumulated in-kernel before dS quantization,
    debug build with the same arithmetic) is at fp32 round-off."""
    q, k, v, do = inputs(B, N, H, Hk, seed=1)
    d = _api().dm_attn_debug(q, k, v, do, causal=causal, mode="matched", identity=True)
    live = d["sum_abs_PdP"] > 0
    r = (d["rowsum_dS"].double().abs() / d["sum_abs_PdP"].double().clamp_min(1e-300))[live]
    assert r.max().item() <= 1e-5 and r.mean().item() <= 2e-7, (r.max().item(), r.mean().item())
    assert torch.isfinite(d["delta"]).all() and torch.isfinite(d["lse"]).all()


@pytest.mark.parametrize("mode", ["matched", "stale"])
def test_deterministic_and_strided(mode):
    """Repeat calls and q / k / v given as strided views (split from one packed tensor) give bitwise equal outputs."""
    B, N, H, Hk = 1, 2048, 8, 2
    q, k, v, do = inputs(B, N, H, Hk, seed=2)
    a = run(q, k, v, do, True, mode)
    b = run(q, k, v, do, True, mode)
    qkv = torch.cat([q, k, v], dim=2)
    s = run(qkv[:, :, :H], qkv[:, :, H:H + Hk], qkv[:, :, H + Hk:], do, True, mode)
    for nm in NAMES:
        assert torch.equal(a[nm], b[nm]), f"repeat {nm}"
        assert torch.equal(a[nm], s[nm]), f"strided {nm}"


def test_matched_and_stale_share_forward():
    """The forward does not depend on the mode; dV = P^T dO does not use delta, so it is equal too."""
    q, k, v, do = inputs(1, 1024, 4, 4, seed=3)
    m, s = run(q, k, v, do, True, "matched"), run(q, k, v, do, True, "stale")
    assert torch.equal(m["O"], s["O"]) and torch.equal(m["dV"], s["dV"])
