"""Forward and backward wall clock of dm_attn_func against flash-attention (FA2 `flash_attn_func` if installed, FA3
`flash_attn_interface.flash_attn_func` if installed) on the same shapes.  Usage: python benchmarks/bench.py [--D 128]
[--N 8192] [--B 2] [--H 16] [--Hk 4] [--iters 20].  Reports the median over iters after 3 warm-ups, in ms per call, for
the forward alone and for forward + backward through autograd (the whole composition, producers included)."""
import argparse, statistics, time

import torch

import delta_matching as dm


def bench(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=128); ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--B", type=int, default=2); ap.add_argument("--H", type=int, default=16); ap.add_argument("--Hk", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(a.B, a.N, a.H, a.D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    k = torch.randn(a.B, a.N, a.Hk, a.D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    v = torch.randn(a.B, a.N, a.Hk, a.D, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
    do = torch.randn(a.B, a.N, a.H, a.D, device="cuda", dtype=torch.bfloat16, generator=g)
    arms = {"delta_matching (matched)": lambda: dm.dm_attn_func(q, k, v, causal=True, mode="matched"),
            "delta_matching (stale)": lambda: dm.dm_attn_func(q, k, v, causal=True, mode="stale")}
    try:
        from flash_attn import flash_attn_func as fa2
        arms["flash-attn 2 (bf16)"] = lambda: fa2(q, k, v, causal=True)
    except Exception:
        pass
    try:
        from flash_attn_interface import flash_attn_func as fa3
        def fa3_out(*a, **kw):   # FA3's interface returns (out, lse) in some versions: one call, first element
            r = fa3(*a, **kw)
            return r[0] if isinstance(r, tuple) else r
        arms["flash-attn 3 (bf16)"] = lambda: fa3_out(q, k, v, causal=True)
    except Exception:
        pass
    print(f"B{a.B} H{a.H} Hk{a.Hk} N{a.N} D{a.D}, median ms over {a.iters}")
    for name, fn in arms.items():
        fwd = bench(lambda: fn(), a.iters)

        def step():
            o = fn(); o.backward(do)
        fb = bench(step, a.iters)
        print(f"  {name:28s} forward {fwd:7.3f}   forward+backward {fb:7.3f}")


if __name__ == "__main__":
    main()
