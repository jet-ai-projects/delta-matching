"""CuTe DSL head_dim-256 producers (delta_matching.cute_prod): bitwise the reference ops they replace.

The dO producer and the GQA fold are checked against this package's references; the Q producer and the fold + q-RoPE
backward are checked against the training pipeline's Triton references when that package is importable."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

cp = pytest.importorskip("delta_matching.cute_prod", reason="needs the CuTe DSL (nvidia-cutlass-dsl)")

from delta_matching.producers import doq_m17  # noqa: E402
from delta_matching.twins import twin_pi  # noqa: E402


def _same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _same_f32(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


def _tables(B, N, RD):
    inv = 1.0 / (1e7 ** (torch.arange(0, RD, 2, device="cuda").float() / RD))
    emb = torch.cat([torch.arange(N, device="cuda").float()[:, None] * inv] * 2, -1)
    return (emb.cos().bfloat16()[None].expand(B, -1, -1).contiguous(),
            emb.sin().bfloat16()[None].expand(B, -1, -1).contiguous())


def _stress_dO(B, N, H, D, seed, nonfinite):
    """Per-block magnitudes over 2^-120 .. 2^20, per-element exponent spread (tiny / denormal values in normal
    blocks), -0, and optionally sparse +-Inf / NaN elements and whole NaN / Inf blocks."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    blk_e = torch.randint(-120, 20, (B, N // 64, H, 1), device="cuda", generator=g).float()
    el_e = -torch.randint(0, 40, (B, N, H, D), device="cuda", generator=g).float() \
        * (torch.rand(B, N, H, D, device="cuda", generator=g) < 0.3)
    x = torch.randn(B, N, H, D, device="cuda", generator=g) * torch.exp2(blk_e.repeat_interleave(64, 1)) * torch.exp2(el_e)
    x[torch.rand(B, N, H, D, device="cuda", generator=g) < 0.01] = -0.0
    if nonfinite:
        m = torch.rand(B, N, H, D, device="cuda", generator=g)
        x[m < 2e-5] = float("inf")
        x[(m >= 2e-5) & (m < 4e-5)] = float("-inf")
        x[(m >= 4e-5) & (m < 6e-5)] = float("nan")
        x[0, 64:128, 3] = float("nan")
        x[0, 128:192, 4] = float("inf")
        x[0, 192:256, 5, :128] = float("-inf")
    return x.bfloat16()


@pytest.mark.parametrize("B,N,H", [(2, 8192, 16), (1, 2048, 8), (1, 1024, 4)])
def test_doq_twin_bitwise(B, N, H):
    dO = (torch.randn(B, N, H, 256, device="cuda") * 1e-3).bfloat16()
    dO[0, :64, 0] = 0
    dO[0, 64:128, 1] = -0.0
    dO[0, 128:192, 2] *= 1e-35
    r8, r8s, _, rt = doq_m17(dO, None, H, 64, want_delta=False, want_twin=True)
    got = cp.doq_twin(dO, H, 64)
    assert got is not None
    assert _same_bytes(r8, got[0]) and _same_f32(r8s, got[1]) and _same_bytes(rt, got[2])


@pytest.mark.parametrize("seed,nonfinite", [(100, False), (101, False), (102, True), (103, True)])
@pytest.mark.parametrize("fastdiv", [None, True, False])
def test_doq_twin_stress(seed, nonfinite, fastdiv):
    B, N, H = 1, 4096, 16
    dO = _stress_dO(B, N, H, 256, seed, nonfinite)
    r8, r8s, _, rt = doq_m17(dO, None, H, 64, want_delta=False, want_twin=True)
    got = cp.doq_twin(dO, H, 64, fastdiv=fastdiv)
    assert got is not None
    assert _same_bytes(r8, got[0]) and _same_f32(r8s, got[1]) and _same_bytes(rt, got[2])


def test_twin_matches_twin_pi():
    dO = torch.randn(1, 2048, 8, 256, device="cuda").bfloat16()
    got = cp.doq_twin(dO, 8, 64)
    assert _same_bytes(twin_pi(got[0]), got[2])


@pytest.mark.parametrize("B,N,H,n_rep", [(2, 4096, 16, 4), (1, 8192, 16, 4), (1, 1024, 8, 2), (1, 2048, 4, 1)])
def test_gqa_fold_bitwise(B, N, H, n_rep):
    g = torch.Generator(device="cuda").manual_seed(7)
    dk = torch.randn(B, N, H, 256, device="cuda", generator=g).bfloat16()
    dv = (torch.randn(B, N, H, 256, device="cuda", generator=g) * 1e-30).bfloat16()
    dk[0, :17, 0] = -0.0
    hk = H // n_rep
    ref_k = dk.permute(0, 2, 1, 3).reshape(B, hk, n_rep, N, 256).sum(2)
    ref_v = dv.permute(0, 2, 1, 3).reshape(B, hk, n_rep, N, 256).sum(2)
    ok, ov = cp.gqa_fold(dk, dv, n_rep)
    assert _same_bytes(ref_k, ok) and _same_bytes(ref_v, ov)


def test_off_fast_path_returns_none():
    dO = torch.randn(1, 1024, 8, 128, device="cuda").bfloat16()   # head_dim 128: not a cute_prod shape
    assert cp.doq_twin(dO, 8, 64) is None


def _train_refs():
    try:
        from kernel.dm_port.fused_producers.m13_rope_bwd import rope_bwd_q
        from kernel.dm_port.fused_producers.tier3a_ropeq import ropeq_q
    except ImportError:
        pytest.skip("needs the training pipeline's Triton references (kernel.dm_port)")
    return ropeq_q, rope_bwd_q


@pytest.mark.parametrize("B,N,H,RD,layout", [(2, 8192, 16, 64, "bhnd"), (2, 8192, 16, 64, "bnhd"),
                                             (1, 2048, 8, 128, "bhnd"), (1, 1024, 4, 32, "bnhd")])
def test_ropeq_twin_bitwise(B, N, H, RD, layout):
    ropeq_q, _ = _train_refs()
    q = (torch.randn(B, H, N, 256, device="cuda") * 1.5).bfloat16() if layout == "bhnd" else \
        (torch.randn(B, N, H, 256, device="cuda") * 1.5).bfloat16().transpose(1, 2)
    q[0, 0, :64] = 0
    q[0, 1, 64:128] = -0.0
    q[0, 2, 128:192] *= 1e-30
    q[0, 3, 192:256] *= 1e4
    cos, sin = _tables(B, N, RD)
    r8, r8s = ropeq_q(q, cos, sin, blk=64)
    got = cp.ropeq_twin(q, cos, sin, 64)
    assert got is not None
    assert _same_bytes(r8, got[0]) and _same_f32(r8s, got[1]) and _same_bytes(twin_pi(r8), got[2])


@pytest.mark.parametrize("B,N,H,n_rep", [(2, 4096, 16, 4), (1, 8192, 16, 4)])
def test_fold_rope_bwd_bitwise(B, N, H, n_rep):
    _, rope_bwd_q = _train_refs()
    g = torch.Generator(device="cuda").manual_seed(11)
    dk = torch.randn(B, N, H, 256, device="cuda", generator=g).bfloat16()
    dv = torch.randn(B, N, H, 256, device="cuda", generator=g).bfloat16()
    dq = (torch.randn(B, N, H, 256, device="cuda", generator=g) * 1e-36).bfloat16()   # denormal products: the ftz path
    cos, sin = _tables(B, N, 64)
    got = cp.fold_rope_bwd(dk, dv, dq, cos, sin, n_rep)
    assert got is not None
    dq_pre, ok, ov = got
    hk = H // n_rep
    assert _same_bytes(rope_bwd_q(dq, cos, sin, H), dq_pre)
    assert _same_bytes(dk.permute(0, 2, 1, 3).reshape(B, hk, n_rep, N, 256).sum(2), ok)
    assert _same_bytes(dv.permute(0, 2, 1, 3).reshape(B, hk, n_rep, N, 256).sum(2), ov)
