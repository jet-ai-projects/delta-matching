"""The kernel against an fp32 reference (both delta modes, both head dims, MHA and GQA, causal and non-causal), its
determinism, the structural identities between the two modes, the two routes (API vs plain), the
flash-attention-layout API and its argument checks.  Needs a Hopper GPU and the built extension (`pip install .` or `python -m delta_matching.jit`)."""
import math

import pytest
import torch

import delta_matching as dm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
# Relative-norm bounds against fp32 attention.  Every operand (q, k, v, P, dO) is rounded to e4m3, so the error is the
# fp8 rounding floor: on Gaussian inputs O ~0.04, dV ~0.04, dQ / dK ~0.06 (real activations sit lower).  The bounds are
# 2x that; a layout, scale or descale mistake reads 0.5 to 1.4.
TOL = {"O": 8e-2, "dV": 8e-2, "dQ": 1.2e-1, "dK": 1.2e-1}
SHAPES = {  # (B, H, Hk, N)
    "mha": (1, 4, 4, 2048),
    "gqa": (1, 8, 2, 2048),
}


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _inputs(B, H, Hk, N, D, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, N, Hk, D, device="cuda", dtype=torch.bfloat16, generator=g)
    do = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return q, k, v, do


def _reference(q, k, v, do, causal=True):
    """fp32 causal attention on the same bf16 inputs, gradients by autograd (GQA by repeating K / V)."""
    B, N, H, D = q.shape
    Hk = k.shape[2]
    qf = q.float().transpose(1, 2).requires_grad_(True)
    kf = k.float().transpose(1, 2).requires_grad_(True)
    vf = v.float().transpose(1, 2).requires_grad_(True)
    kr = kf.repeat_interleave(H // Hk, dim=1)
    vr = vf.repeat_interleave(H // Hk, dim=1)
    s = (qf @ kr.transpose(-1, -2)) * (D ** -0.5)
    if causal:
        mask = torch.ones(N, N, device=q.device, dtype=torch.bool).tril()
        s = s.masked_fill(~mask, float("-inf"))
    o = torch.softmax(s, dim=-1) @ vr
    o.backward(do.float().transpose(1, 2))
    return (o.transpose(1, 2), qf.grad.transpose(1, 2), kf.grad.transpose(1, 2), vf.grad.transpose(1, 2))


def _run(q, k, v, do, mode, causal=True, fn=dm.dm_attn_func):
    q = q.clone().requires_grad_(True); k = k.clone().requires_grad_(True); v = v.clone().requires_grad_(True)
    o = fn(q, k, v, causal=causal, mode=mode)
    o.backward(do)
    return o, q.grad, k.grad, v.grad


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("mode", ["matched", "stale"])
def test_matches_fp32_reference(D, shape, mode):
    B, H, Hk, N = SHAPES[shape]
    q, k, v, do = _inputs(B, H, Hk, N, D, 0)
    ro, rdq, rdk, rdv = _reference(q, k, v, do)
    o, dq, dk, dv = _run(q, k, v, do, mode)
    assert o.shape == q.shape and dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
    errs = {"O": _rel(o, ro), "dQ": _rel(dq, rdq), "dK": _rel(dk, rdk), "dV": _rel(dv, rdv)}
    print(f"D={D} {shape} {mode}: relative errors vs fp32 {errs}")
    for name, e in errs.items():
        assert math.isfinite(e) and e < TOL[name], f"{name} relative error {e:.4f} exceeds {TOL[name]} ({shape}, D={D}, {mode})"


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("mode", ["matched", "stale"])
def test_deterministic(D, mode):
    B, H, Hk, N = SHAPES["gqa"]
    q, k, v, do = _inputs(B, H, Hk, N, D, 1)
    a = _run(q, k, v, do, mode)
    b = _run(q, k, v, do, mode)
    for x, y, name in zip(a, b, ("O", "dQ", "dK", "dV")):
        assert torch.equal(x, y), f"{name} differs between two identical calls (D={D}, {mode})"


@pytest.mark.parametrize("D", [128, 256])
def test_modes_share_forward_and_dv(D):
    """The two modes differ only in delta: same forward, same dV (dV does not depend on delta)."""
    B, H, Hk, N = SHAPES["gqa"]
    q, k, v, do = _inputs(B, H, Hk, N, D, 2)
    om, dqm, dkm, dvm = _run(q, k, v, do, "matched")
    os_, dqs, dks, dvs = _run(q, k, v, do, "stale")
    assert torch.equal(om, os_) and torch.equal(dvm, dvs)
    assert not torch.equal(dqm, dqs) and not torch.equal(dkm, dks), "the two deltas must give different dQ / dK"


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("mode", ["matched", "stale"])
def test_plain_route_is_bitwise(D, mode):
    """The API route (producer-made twins) and the plain route (quant.py / doq.py producers, twins made by the bwd_fp8 op)
    are the same numerics: bitwise outputs."""
    B, H, Hk, N = SHAPES["gqa"]
    q, k, v, do = _inputs(B, H, Hk, N, D, 3)
    a = _run(q, k, v, do, mode)

    def plain(q, k, v, causal, mode):
        return dm.dm_attn_func_plain(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal, delta_mode=mode)
    b = _run(q, k, v, do, mode, fn=plain)
    for x, y, name in zip(a, b, ("O", "dQ", "dK", "dV")):
        assert torch.equal(x, y), f"{name}: the plain route differs from the twin route (D={D}, {mode})"


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_stale_noncausal_matches_fp32_reference(D, shape):
    B, H, Hk, N = SHAPES[shape]
    q, k, v, do = _inputs(B, H, Hk, N, D, 4)
    ro, rdq, rdk, rdv = _reference(q, k, v, do, causal=False)
    o, dq, dk, dv = _run(q, k, v, do, "stale", causal=False)
    errs = {"O": _rel(o, ro), "dQ": _rel(dq, rdq), "dK": _rel(dk, rdk), "dV": _rel(dv, rdv)}
    print(f"D={D} {shape} stale non-causal: relative errors vs fp32 {errs}")
    for name, e in errs.items():
        assert math.isfinite(e) and e < TOL[name], f"{name} relative error {e:.4f} exceeds {TOL[name]} ({shape}, D={D})"


def test_flash_attn_alias_and_argument_checks():
    B, H, Hk, N = SHAPES["gqa"]
    q, k, v, do = _inputs(B, H, Hk, N, 128, 5)
    a = dm.flash_attn_func(q, k, v, causal=True)
    b = dm.dm_attn_func(q, k, v, causal=True, mode="matched")
    assert torch.equal(a, b)
    with pytest.raises(NotImplementedError):
        dm.flash_attn_func(q, k, v, dropout_p=0.1, causal=True)
    with pytest.raises(NotImplementedError):
        dm.flash_attn_func(q, k, v, causal=True, window_size=(128, 0))
    with pytest.raises(ValueError, match="causal"):
        dm.dm_attn_func(q, k, v, causal=False, mode="matched")     # the matched delta needs causal attention
    with pytest.raises(ValueError, match="mode"):
        dm.dm_attn_func(q, k, v, causal=True, mode="other")
    with pytest.raises(ValueError, match="seqlen"):
        dm.dm_attn_func(q[:, :1000], k[:, :1000], v[:, :1000], causal=True)   # seqlen % 256 == 0 at head_dim 128
    with pytest.raises(TypeError, match="bfloat16"):
        dm.dm_attn_func(q.float(), k.float(), v.float(), causal=True)         # bf16 only


def test_kernel_version_stamp():
    s = dm.kernel_version()
    assert s["fa_src_commit"] == dm.FA_SRC_COMMIT and len(s["code_sha"]) == 12 and s["numerics"]["128"]
