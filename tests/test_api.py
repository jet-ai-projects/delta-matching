"""The public API contract: the usage patterns and input errors a user hits first, and the checks of the compiled ops.

Every case here was a failure of the first release candidate: a bare AssertionError on `o.sum().backward()` (the backward
needed a contiguous dO), a Triton crash for tensors on a GPU other than the current one, bare AssertionErrors (no message,
gone under `python -O`) for invalid inputs, and a head_dim-128 matched delta that ended the process (exit(1) in the
launcher) on a seqlen the op did not check.  The long-context head-group split of the backward is checked bitwise against
the single call here as well.
"""
import inspect
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
BF = torch.bfloat16
E4M3 = torch.float8_e4m3fn


def _inputs(B=2, N=1024, H=8, Hk=2, D=128, device="cuda", seed=0, grad=True):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, N, H, D, generator=g).to(device, BF).requires_grad_(grad)
    k = torch.randn(B, N, Hk, D, generator=g).to(device, BF).requires_grad_(grad)
    v = torch.randn(B, N, Hk, D, generator=g).to(device, BF).requires_grad_(grad)
    return q, k, v


def _leaf(*ts):
    return [t.detach().clone().requires_grad_(True) for t in ts]


def _fwd_bwd(q, k, v, go, mode, causal=True):
    from delta_matching import dm_attn_func
    a = _leaf(q, k, v)
    o = dm_attn_func(*a, causal=causal, mode=mode)
    o.backward(go)
    return [o.detach()] + [x.grad for x in a]


@pytest.mark.parametrize("mode", ["matched", "stale"])
@pytest.mark.parametrize("D", [128, 256])
def test_backward_accepts_an_expanded_grad(mode, D):
    """o.sum().backward() hands the backward an expanded (stride-0) dO; it must equal the backward of an explicit ones tensor."""
    from delta_matching import dm_attn_func
    q, k, v = _inputs(D=D)
    a = _leaf(q, k, v); dm_attn_func(*a, causal=True, mode=mode).sum().backward()
    b = _leaf(q, k, v); o = dm_attn_func(*b, causal=True, mode=mode); o.backward(torch.ones_like(o))
    for x, y in zip(a, b):
        assert torch.equal(x.grad, y.grad)


@pytest.mark.parametrize("mode", ["matched", "stale"])
def test_non_contiguous_inputs_match_contiguous(mode):
    """q / k / v as views of one packed qkv tensor (flash-attention's usual call) give the contiguous result bitwise."""
    from delta_matching import dm_attn_func
    B, N, H, Hk, D = 2, 1024, 8, 2, 128
    qkv = torch.randn(B, N, H + 2 * Hk, D, device="cuda", dtype=BF)
    go = torch.randn(B, N, H, D, device="cuda", dtype=BF)
    views = [t.detach().requires_grad_(True) for t in qkv.split([H, Hk, Hk], dim=2)]
    copies = [t.detach().contiguous().requires_grad_(True) for t in views]
    oa = dm_attn_func(*views, causal=True, mode=mode); oa.backward(go)
    ob = dm_attn_func(*copies, causal=True, mode=mode); ob.backward(go)
    assert torch.equal(oa, ob)
    for x, y in zip(views, copies):
        assert torch.equal(x.grad, y.grad)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["matched", "stale"])
def test_tensors_on_a_non_current_device(mode):
    """Tensors on cuda:1 while the current device is cuda:0 run on cuda:1 and match a run with cuda:1 current."""
    q, k, v = _inputs(device="cuda:1", seed=5)
    go = torch.randn(q.shape, device="cuda:1", dtype=BF)
    with torch.cuda.device(1):
        a = _fwd_bwd(q, k, v, go, mode)
    with torch.cuda.device(0):
        b = _fwd_bwd(q, k, v, go, mode)
    torch.cuda.synchronize(1)
    assert all(x.device == torch.device("cuda:1") for x in b)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("mode", ["matched", "stale"])
@pytest.mark.parametrize("groups,streams", [("kv", 1), ("head", 1), ("kv", 2)])
def test_head_group_split_is_bitwise(D, mode, groups, streams, monkeypatch):
    """The long-context split of the backward (per KV group or per query head, sequential or on two streams; in the
    matched mode with the delta taken per group) gives the single call's output and gradients bitwise."""
    import delta_matching.split as S
    q, k, v = _inputs(B=1, N=1024, H=8, Hk=2, D=D, seed=7)
    go = torch.randn(q.shape, device="cuda", dtype=BF)
    monkeypatch.setattr(S, "_BWD_GROUPS", "off")
    ref = _fwd_bwd(q, k, v, go, mode)
    monkeypatch.setattr(S, "_BWD_GROUPS", groups)
    monkeypatch.setattr(S, "_BWD_STREAMS", streams)
    monkeypatch.setattr(S, "_bwd_stream_pool", {})
    got = _fwd_bwd(q, k, v, go, mode)
    for name, x, y in zip(("O", "dQ", "dK", "dV"), ref, got):
        assert torch.equal(x, y), f"{name}: split {groups} x{streams} differs from the single call (D={D}, {mode})"


def test_flash_attn_func_signature_matches_flash_attention():
    flash_attn = pytest.importorskip("flash_attn")
    from delta_matching import flash_attn_func
    ours = list(inspect.signature(flash_attn_func).parameters)
    theirs = list(inspect.signature(flash_attn.flash_attn_func).parameters)
    assert ours[:len(theirs)] == theirs


def _raises(exc, match, fn):
    with pytest.raises(exc, match=match):
        fn()


def test_input_errors_are_clear_exceptions():
    """Invalid inputs raise ValueError / TypeError / NotImplementedError naming the constraint (never a bare assert)."""
    from delta_matching import dm_attn_func, dm_attn_func_bhnd, dm_attn_func_plain, flash_attn_func
    q, k, v = _inputs(B=1, grad=False)
    _raises(ValueError, "causal", lambda: dm_attn_func(q, k, v))                                   # matched + non-causal
    _raises(ValueError, "causal", lambda: flash_attn_func(q, k, v))
    _raises(ValueError, "mode", lambda: dm_attn_func(q, k, v, causal=True, mode="match"))
    _raises(TypeError, "bfloat16", lambda: dm_attn_func(q.half(), k.half(), v.half(), causal=True))
    _raises(TypeError, "bfloat16", lambda: dm_attn_func(q.float(), k.float(), v.float(), causal=True))
    _raises(ValueError, "head_dim", lambda: dm_attn_func(q[..., :64], k[..., :64], v[..., :64], causal=True))
    _raises(ValueError, "256", lambda: dm_attn_func(q[:, :384], k[:, :384], v[:, :384], causal=True))    # hd128: 256-multiple
    _raises(ValueError, "nheads", lambda: dm_attn_func(q[:, :, :7], k, v, causal=True))
    _raises(ValueError, "seqlen", lambda: dm_attn_func(q[:, :512], k, v, causal=True))
    _raises(ValueError, "CUDA", lambda: dm_attn_func(q.cpu(), k.cpu(), v.cpu(), causal=True))
    _raises(ValueError, "4-D", lambda: dm_attn_func(q[0], k[0], v[0], causal=True))
    _raises(ValueError, "shape", lambda: dm_attn_func(q, k, v[:, :, :1], causal=True))
    _raises(NotImplementedError, "softmax_scale", lambda: dm_attn_func(q, k, v, causal=True, softmax_scale=0.05))
    _raises(NotImplementedError, "dropout", lambda: flash_attn_func(q, k, v, 0.1, causal=True))
    _raises(NotImplementedError, "window", lambda: flash_attn_func(q, k, v, causal=True, window_size=(128, 0)))
    _raises(NotImplementedError, "softcap", lambda: flash_attn_func(q, k, v, causal=True, softcap=30.0))
    _raises(TypeError, "unexpected keyword", lambda: flash_attn_func(q, k, v, causal=True, bogus=1))
    # the (B, H, N, D) entry points share the contract
    qb, kb, vb = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    _raises(ValueError, "causal", lambda: dm_attn_func_bhnd(qb, kb, vb, causal=False, delta_mode="matched"))
    _raises(ValueError, "seqlen", lambda: dm_attn_func_plain(qb[:, :, :384], kb[:, :, :384], vb[:, :, :384], causal=True))
    # the default scale passed explicitly is accepted
    dm_attn_func(q, k, v, causal=True, softmax_scale=q.shape[-1] ** -0.5)


def test_head_dim_256_accepts_128_multiples():
    from delta_matching import dm_attn_func
    q, k, v = _inputs(B=1, N=384, H=4, Hk=1, D=256, grad=False)
    assert dm_attn_func(q, k, v, causal=True).shape == q.shape


def _ops():
    from delta_matching.core import _ops
    return _ops()


def _fwd_args(B=1, H=4, Hk=2, N=512, D=128):
    """Valid fwd_fp8 arguments from the package's own producers."""
    from delta_matching.core import blocks
    from delta_matching.producers import quant_qkv_m17
    g = torch.Generator(device="cpu").manual_seed(3)
    q, k, v = (torch.randn(B, h, N, D, generator=g).to("cuda", BF) for h in (H, Hk, Hk))
    Q8, Q8s, K8, K8s, V8, V8s, _, _ = quant_qkv_m17(q, k, v, want_qt=True, blocks=blocks(D))

    def bshd(x, h):
        return x.view(B, h, N, D).permute(0, 2, 1, 3)
    return [bshd(Q8, H), bshd(K8, Hk), bshd(V8, Hk), Q8s.view(B, H, -1), K8s.view(B, Hk, -1), V8s.view(B, Hk, -1),
            D ** -0.5, True]


def test_ops_check_their_inputs():
    """The compiled ops refuse invalid inputs with a RuntimeError naming the problem (before any launch)."""
    ops = _ops()
    args = _fwd_args()
    o, lse = ops.fwd_fp8(*args)                                   # the valid call runs
    assert o.shape == args[0].shape and lse.dtype == torch.float32

    def with_arg(i, x):
        a = list(args); a[i] = x
        return a
    q8 = args[0]
    _raises(RuntimeError, "CUDA", lambda: ops.fwd_fp8(*with_arg(0, q8.cpu())))
    _raises(RuntimeError, "same device|every tensor", lambda: ops.fwd_fp8(*with_arg(3, args[3].cpu())))
    _raises(RuntimeError, "causal", lambda: ops.fwd_fp8(*with_arg(0, q8[:, :256])))            # seqlen_q != seqlen_k
    _raises(RuntimeError, "empty", lambda: ops.fwd_fp8(*with_arg(0, q8[:, :0])))
    _raises(RuntimeError, "e4m3", lambda: ops.fwd_fp8(*with_arg(0, q8.to(BF))))
    _raises(RuntimeError, "fp32", lambda: ops.fwd_fp8(*with_arg(3, args[3].to(BF))))
    buf = torch.empty(q8.numel() + 16, dtype=torch.uint8, device="cuda")                    # a code tensor 1 byte off
    shifted = buf[1:1 + q8.numel()].view(E4M3).view(q8.permute(0, 2, 1, 3).shape).permute(0, 2, 1, 3)
    _raises(RuntimeError, "aligned", lambda: ops.fwd_fp8(*with_arg(0, shifted)))
    _raises(RuntimeError, "CUDA", lambda: ops.twin_pi(torch.zeros(1, 64, 1, 64, dtype=E4M3)))


def test_hd128_delta_rejects_a_bad_seqlen_without_ending_the_process(tmp_path):
    """delta_fp8 at head_dim 128 needs seqlen_k % 128 == 0; it used to exit(1) inside the launcher (killing Python).
    Run in a subprocess so a regression fails this test instead of the whole session."""
    code = r"""
import torch
from delta_matching.core import _ops
ops = _ops()
B, H, Hk, N, D = 1, 2, 1, 192, 128
e = torch.float8_e4m3fn
z = lambda *s, dt=e: torch.zeros(*s, dtype=dt, device="cuda")
f = torch.float32
try:
    ops.delta_fp8(z(B, N, H, D), z(B, N, H, D), z(B, N, Hk, D), z(B, N, Hk, D), z(B, H, N, dt=f), z(B, Hk, D, N),
                  z(B, H, 2, dt=f), z(B, Hk, 3, dt=f), z(B, Hk, 3, dt=f), z(B, H, 2, dt=f), True)
    print("NO-ERROR")
except RuntimeError as ex:
    print("RUNTIME-ERROR", "128-multiple" in str(ex))
"""
    import delta_matching
    env = dict(os.environ)   # the child imports the package this session imported (installed or a source tree), not the cwd's
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(delta_matching.__file__)))
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600, cwd=str(tmp_path))
    assert p.returncode == 0, f"the process ended with {p.returncode}: {p.stderr[-2000:]}"
    assert "RUNTIME-ERROR True" in p.stdout, p.stdout + p.stderr[-2000:]


def test_kernel_version_reports_the_loaded_library():
    import delta_matching as dm
    _ops()
    s = dm.kernel_version()
    assert s["library"] and os.path.isfile(s["library"]) and s["bf16_dp"] is False


def test_torch_compile_matches_eager():
    """torch.compile around a function that calls dm_attn_func gives the eager forward and backward, bit for bit."""
    from delta_matching import dm_attn_func
    q, k, v = _inputs(B=1, N=1024, H=8, Hk=2, D=128, seed=11, grad=False)
    go = torch.randn(1, 1024, 8, 128, device="cuda", dtype=BF)

    def f(q, k, v):
        return dm_attn_func(q * 1.0, k, v, causal=True, mode="matched")
    leaves = [(x.detach().clone().requires_grad_(True) for x in (q, k, v)) for _ in range(2)]
    qa, ka, va = leaves[0]
    oa = torch.compile(f)(qa, ka, va); oa.backward(go)
    qb, kb, vb = leaves[1]
    ob = f(qb, kb, vb); ob.backward(go)
    assert torch.equal(oa, ob) and torch.equal(qa.grad, qb.grad) and torch.equal(ka.grad, kb.grad) and torch.equal(va.grad, vb.grad)


def test_cuda_graph_capture_of_the_forward():
    """After a warm-up call for the shape (Triton compilation, cached tables), the forward can be captured in a CUDA graph;
    the replay equals an eager call."""
    from delta_matching import dm_attn_func
    q, k, v = _inputs(B=1, N=1024, H=8, Hk=2, D=128, seed=12, grad=False)
    dm_attn_func(q, k, v, causal=True)            # warm-up outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = dm_attn_func(q, k, v, causal=True)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(o, dm_attn_func(q, k, v, causal=True))
