"""head_dim 256 ring-block backward ops: strided twin views (the ring's half slices of the wire buffer / Q twin) give the
same bytes as contiguous copies, and the fp32 dQ accumulation of bwd_fp8_g (dq_acc mode 2 store, 1 accumulate) is bitwise
a zeroed fp32 dQ plus each block's bf16 dQ added in launch order (torch's fp32 add)."""
import pytest
import torch

from delta_matching.attn import _ops_gt
from delta_matching.core import _quant_native, blocks
from delta_matching.doq import fused_doq2
from delta_matching.twins import twin_pi

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

H, HK, D = 16, 4, 256


def _setup(N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, H, N, D, device="cuda", generator=g).bfloat16()
    k = (torch.randn(1, HK, N, D, device="cuda", generator=g) + 0.5).bfloat16()
    v = torch.randn(1, HK, N, D, device="cuda", generator=g).bfloat16()
    do = torch.randn(1, H, N, D, device="cuda", generator=g).bfloat16()
    ops = _ops_gt()
    BM, _ = blocks(D)
    Q8, Q8s, K8, K8s, V8, V8s, kTt = _quant_native(q, k, v)
    sm = D ** -0.5
    bshd = lambda x, h: x.view(1, h, N, D).permute(0, 2, 1, 3)  # noqa: E731
    O, Lse = ops.fwd_fp8(bshd(Q8, H), bshd(K8, HK), bshd(V8, HK), Q8s.view(1, H, -1), K8s.view(1, HK, -1),
                         V8s.view(1, HK, -1), sm, True)
    dO8, dO8s, Dlt, _ = fused_doq2(do, BM, O, H, want_dof=False)
    c = N // 2
    t = dict(ops=ops, sm=sm, N=N, c=c, BM=BM,
             Q=bshd(Q8, H), Qs=Q8s.view(1, H, -1), dO=bshd(dO8, H), dOs=dO8s.view(1, H, -1),
             K=bshd(K8, HK), V=bshd(V8, HK), Ks=K8s.view(1, HK, -1), Vs=V8s.view(1, HK, -1),
             kTt=kTt.view(1, HK, D, N), Qt=twin_pi(Q8).view(1, H, D, N), dOt=twin_pi(dO8).view(1, H, D, N),
             L=Lse.view(1, H, N).contiguous(), Dl=Dlt.view(1, H, N).contiguous())
    return t


def _blocks(t, contig):
    """The three ring block kinds on one rank: A (all Q vs the first K half), B (the second Q half vs all K), the causal
    diagonal; twins as strided views (contig=False) or contiguous copies."""
    c, BM = t["c"], t["BM"]
    cc = (lambda x: x.contiguous()) if contig else (lambda x: x)
    A = (t["dO"], t["Q"], t["K"][:, :c], t["V"][:, :c], t["L"], t["Dl"], cc(t["kTt"][..., :c]), t["Qt"], t["dOt"],
         t["Qs"], t["Ks"][..., :c // 32].contiguous(), t["Vs"][..., :c // 32].contiguous(), t["dOs"], False)
    B = (t["dO"][:, c:], t["Q"][:, c:], t["K"], t["V"], t["L"][..., c:].contiguous(), t["Dl"][..., c:].contiguous(),
         t["kTt"], cc(t["Qt"][..., c:]), cc(t["dOt"][..., c:]), t["Qs"][..., c // BM:].contiguous(), t["Ks"], t["Vs"],
         t["dOs"][..., c // BM:].contiguous(), False)
    Dg = (t["dO"], t["Q"], t["K"], t["V"], t["L"], t["Dl"], t["kTt"], t["Qt"], t["dOt"], t["Qs"], t["Ks"], t["Vs"],
          t["dOs"], True)
    return A, B, Dg


def _call_t(t, blk):
    *a, causal = blk
    return t["ops"].bwd_fp8_t(*a, t["sm"], causal)


def _call_g(t, blk, dq_acc, mode):
    *a, causal = blk
    return t["ops"].bwd_fp8_g(*a, t["sm"], causal, dq_acc, mode)


@pytest.mark.parametrize("N", [2048, 4096])
def test_strided_twins_bitwise(N):
    t = _setup(N)
    for s, cp in zip(_blocks(t, False), _blocks(t, True)):
        for x, y in zip(_call_t(t, s), _call_t(t, cp)):
            assert torch.equal(x, y)
    # the matched-delta pass 0 on a strided K twin
    c = t["c"]
    a = (t["dO"], t["Q"], t["K"][:, :c], t["V"][:, :c], t["L"])
    sc = (t["Qs"], t["Ks"][..., :c // 32].contiguous(), t["Vs"][..., :c // 32].contiguous(), t["dOs"], False)
    d1 = t["ops"].delta_fp8(*a, t["kTt"][..., :c], *sc)
    d2 = t["ops"].delta_fp8(*a, t["kTt"][..., :c].contiguous(), *sc)
    assert torch.equal(d1, d2)


@pytest.mark.parametrize("N", [2048, 4096])
@pytest.mark.parametrize("first", ["A", "B"])
def test_dq_acc_matches_bf16_sum(N, first):
    t = _setup(N, seed=1)
    c = t["c"]
    A, B, Dg = _blocks(t, False)
    order = [A, B, Dg] if first == "A" else [B, Dg]   # rank 0 has no A block: dq is zeroed and every block accumulates
    ref = torch.zeros(N, H, D, device="cuda", dtype=torch.float32)
    outs = []
    for blk in order:
        dq, dk, dv = _call_t(t, blk)
        (ref[c:] if blk is B else ref).add_(dq[0])
        outs.append((dk, dv))
    acc = torch.full((N, H, D), float("nan"), device="cuda", dtype=torch.float32)
    if first == "B":
        acc.zero_()
    for i, blk in enumerate(order):
        mode = 2 if (i == 0 and first == "A") else 1
        dk, dv = _call_g(t, blk, acc[c:] if blk is B else acc, mode)
        assert torch.equal(dk, outs[i][0]) and torch.equal(dv, outs[i][1])
    assert torch.equal(acc, ref)
    # a dQ that skips the per-block bf16 rounding is not this
    assert not torch.equal(acc, ref + 1e-7 * ref.abs().max())


def test_rejections():
    t = _setup(2048)
    A, _, _ = _blocks(t, False)
    acc = torch.zeros(2048, H, D, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _call_g(t, A, acc, 3)
    with pytest.raises(RuntimeError):
        _call_g(t, A, acc.bfloat16(), 2)
    with pytest.raises(RuntimeError):
        _call_g(t, A, acc[:1024], 2)
    with pytest.raises(RuntimeError):   # 4-byte aligned float2 target
        _call_g(t, A, torch.zeros(2048 * H * D + 2, device="cuda")[1:-1].view(2048, H, D), 2)
    with pytest.raises(RuntimeError):   # overlapping rows (every row the same memory)
        _call_g(t, A, torch.zeros(1, H, D, device="cuda").expand(2048, H, D), 2)
    bad = list(A)
    kt = t["kTt"][..., :t["c"]]
    bad[6] = kt.transpose(2, 3).contiguous().transpose(2, 3)   # s not unit-stride
    with pytest.raises(RuntimeError):
        _call_t(t, tuple(bad))
    bad[6] = torch.empty(1, HK, D, t["c"] + 8, device="cuda", dtype=kt.dtype)[..., :t["c"]]   # d stride not 16-B multiple
    with pytest.raises(RuntimeError):
        _call_t(t, tuple(bad))


