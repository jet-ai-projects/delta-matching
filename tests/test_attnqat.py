"""The attnqat mode (attn-qat's delta from the forward's high-precision second output O', arXiv 2603.00040) against an
fp64 reference that starts from the kernel's own codes, plus its structural identities with the stale mode.  head_dim 256
only (the O' residual pass, fwd_fp8_res).  Needs a Hopper GPU and a build with build_attnqat() == 1."""
import math

import pytest
import torch

import delta_matching as dm
from delta_matching.attn import _ops_gt
from delta_matching.core import blocks
from delta_matching.producers import doq_m17, quant_qkv_m17

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
D = 256


def _ops():
    ops = _ops_gt()
    if not hasattr(ops, "build_attnqat") or int(ops.build_attnqat()) != 1:
        pytest.skip("library without the O' pass")
    return ops


def _inputs(B, H, Hk, N, seed, qscale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(B, H, N, D, device="cuda", generator=g) * qscale).to(torch.bfloat16)
    k = torch.randn(B, Hk, N, D, device="cuda", generator=g).to(torch.bfloat16)
    v = torch.randn(B, Hk, N, D, device="cuda", generator=g).to(torch.bfloat16)
    dO = torch.randn(B, N, H, D, device="cuda", generator=g).to(torch.bfloat16)
    return q, k, v, dO


def _deq(x8, xs, blk):
    return x8.double() * xs.double().repeat_interleave(blk, dim=1)[:, :, None]


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def _fwd_parts(q, k, v, causal):
    ops = _ops()
    B, H, N, _ = q.shape
    Hk = k.shape[1]
    BM, BN = blocks(D)
    Q8, Q8s, K8, K8s, V8, V8s, _, _ = quant_qkv_m17(q, k, v, want_qt=True, blocks=(BM, BN))

    def bshd(x, h):
        return x.view(B, h, N, D).permute(0, 2, 1, 3)
    fa = (bshd(Q8, H), bshd(K8, Hk), bshd(V8, Hk), Q8s.view(B, H, -1), K8s.view(B, Hk, -1), V8s.view(B, Hk, -1),
          D ** -0.5, causal)
    O, lse = ops.fwd_fp8(*fa)
    E, lse2 = ops.fwd_fp8_res(*fa)
    return (Q8, Q8s, K8, K8s, V8, V8s), O, E, lse, lse2


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("qscale", [1.0, 3.0])
def test_oprime_matches_fp64_reference(causal, qscale):
    B, H, Hk, N = 1, 4, 2, 2048
    q, k, v, _ = _inputs(B, H, Hk, N, 7, qscale)
    (Q8, Q8s, K8, K8s, V8, V8s), O, E, lse, lse2 = _fwd_parts(q, k, v, causal)
    assert torch.equal(lse, lse2), "the O' pass must reproduce the forward's softmax statistics bitwise"
    BM, BN = blocks(D)
    Qd, Kd, Vd = _deq(Q8, Q8s, BM), _deq(K8, K8s, BN), _deq(V8, V8s, BN)
    Op = (O.float() + E.float()).to(torch.bfloat16)
    mask = torch.ones(N, N, device="cuda", dtype=torch.bool).tril()
    for bh in range(B * H):
        b, h = divmod(bh, H)
        hk = b * Hk + h // (H // Hk)
        S = Qd[bh] @ Kd[hk].T
        if causal:
            S = S.masked_fill(~mask, float("-inf"))
        Opr = torch.softmax(S, -1) @ Vd[hk]
        e_op, e_o = _rel(Op[b, :, h], Opr), _rel(O[b, :, h], Opr)
        # O' = bf16(O_bf16 + E): two bf16 roundings (~1.1e-3 each) over the two-term P; peaky attention puts O's own
        # P-quantization error near that floor too, so the bound is absolute plus "closer than O"
        assert e_op < 3.5e-3, f"O' vs P V / l: {e_op:.3e}"
        assert e_op < e_o, f"O' ({e_op:.3e}) must be closer to P V / l than O ({e_o:.3e})"


def test_attnqat_shares_the_forward_and_differs_from_stale_only_through_delta():
    B, H, Hk, N = 1, 8, 2, 2048
    q, k, v, dO = _inputs(B, H, Hk, N, 11, 2.0)
    res = {}
    for mode in ("stale", "attnqat"):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        o = dm.dm_attn_func_bhnd(qq, kk, vv, causal=True, delta_mode=mode)
        o.backward(dO)
        res[mode] = (o, qq.grad, kk.grad, vv.grad)
    assert torch.equal(res["stale"][0], res["attnqat"][0]), "attnqat must not change O"
    assert torch.equal(res["stale"][3], res["attnqat"][3]), "dV does not see delta: it must be bitwise stale's"
    # the backward with delta = rowsum(dO * O') by hand through the raw op is the attnqat route, bitwise
    ops = _ops()
    BM, BN = blocks(D)
    (Q8, Q8s, K8, K8s, V8, V8s), O, E, lse, _ = _fwd_parts(q, k, v, True)
    Op = (O.float() + E.float()).to(torch.bfloat16)
    dO8, dO8s, dlt, dOt = doq_m17(dO, Op, H, BM, want_delta=True, want_twin=True)
    ref = (dO.float() * Op.float()).sum(-1).permute(0, 2, 1).reshape(B * H, N)
    assert _rel(dlt, ref) < 1e-5
    Q8_, Q8s_, K8_, K8s_, V8_, V8s_, kTt, Qt = quant_qkv_m17(q, k, v, want_qt=True, blocks=(BM, BN))

    def bshd(x, h):
        return x.view(B, h, N, D).permute(0, 2, 1, 3)
    dq, dk, dv = ops.bwd_fp8_t(bshd(dO8, H), bshd(Q8_, H), bshd(K8_, Hk), bshd(V8_, Hk), lse, dlt.view(B, H, N),
                               kTt.view(B, Hk, D, N), Qt.view(B, H, D, N), dOt.view(B, H, D, N), Q8s_.view(B, H, -1),
                               K8s_.view(B, Hk, -1), V8s_.view(B, Hk, -1), dO8s.view(B, H, -1), D ** -0.5, True)
    rep = H // Hk
    dk = dk.permute(0, 2, 1, 3).reshape(B, Hk, rep, N, D).sum(2)
    assert torch.equal(dq.permute(0, 2, 1, 3), res["attnqat"][1]), "attnqat dq != the stale backward with delta(O')"
    assert torch.equal(dk, res["attnqat"][2]), "attnqat dk != the stale backward with delta(O')"


def test_attnqat_delta_closer_to_the_matched_delta_than_stale():
    """attn-qat's delta removes the P-quantization part of stale's rowsum(dS) violation; the dO-quantization part,
    which only the matched delta removes, remains (the kernel's dP sees the dO codes)."""
    ops = _ops()
    B, H, Hk, N = 1, 8, 2, 2048
    q, k, v, dO = _inputs(B, H, Hk, N, 13, 3.0)
    BM, BN = blocks(D)
    Q8, Q8s, K8, K8s, V8, V8s, kTt, _ = quant_qkv_m17(q, k, v, want_qt=True, blocks=(BM, BN))

    def bshd(x, h):
        return x.view(B, h, N, D).permute(0, 2, 1, 3)
    fa = (bshd(Q8, H), bshd(K8, Hk), bshd(V8, Hk), Q8s.view(B, H, -1), K8s.view(B, Hk, -1), V8s.view(B, Hk, -1),
          D ** -0.5, True)
    O, lse = ops.fwd_fp8(*fa)
    E, _ = ops.fwd_fp8_res(*fa)
    Op = (O.float() + E.float()).to(torch.bfloat16)
    _, _, d_st, _ = doq_m17(dO, O, H, BM, want_delta=True, want_twin=False)
    dO8, dO8s, d_aq, _ = doq_m17(dO, Op, H, BM, want_delta=True, want_twin=False)
    d_mt = ops.delta_fp8(bshd(dO8, H), bshd(Q8, H), bshd(K8, Hk), bshd(V8, Hk), lse.view(B, H, N).contiguous(),
                         kTt.view(B, Hk, D, N), Q8s.view(B, H, -1), K8s.view(B, Hk, -1), V8s.view(B, Hk, -1),
                         dO8s.view(B, H, -1), True).view(B * H, N)
    assert (d_mt - d_aq).norm() < (d_mt - d_st).norm()


def test_attnqat_argument_checks():
    q, k, v, _ = _inputs(1, 2, 2, 512, 0)
    with pytest.raises(NotImplementedError, match="head_dim 256"):
        dm.dm_attn_func_bhnd(q[..., :128].contiguous(), k[..., :128].contiguous(), v[..., :128].contiguous(),
                             causal=True, delta_mode="attnqat")
    with pytest.raises(NotImplementedError, match="plain route"):
        dm.dm_attn_func_plain(q, k, v, causal=True, delta_mode="attnqat")
