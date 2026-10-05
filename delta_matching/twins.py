"""The transposed operand twins the backward loads from global memory: Q^T and dO^T as e4m3 codes.

A twin holds the codes of a (B * H, N, D) tensor laid out (B * H, D, N), with the N positions inside every 16-position
window permuted to the order of the kernel's shared-memory tiles: physical position c holds row SIGMA[c],
SIGMA = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]; equivalently position(m) = (m & ~15) | ((m & 6) << 1) |
((m & 8) >> 2) | (m & 1).  The block scales are untouched: a twin is data movement only.  `twin_pi` is a single-store
Triton kernel; `twin_pi_reference` is the torch definition the self-test compares it with.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

SIGMA = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]


def pi_positions(n: int, device=None) -> torch.Tensor:
    """position(m) for m in range(n) (the kernel's fq8_pi_m)."""
    m = torch.arange(n, device=device, dtype=torch.int64)
    return (m & ~15) | ((m & 6) << 1) | ((m & 8) >> 2) | (m & 1)


def twin_pi_reference(x8: torch.Tensor) -> torch.Tensor:
    """[BH, N, D] -> [BH, D, N] with pi-ordered N positions (torch, for tests)."""
    BH, N, D = x8.shape
    pos = pi_positions(N, x8.device)
    out = torch.empty(BH, D, N, device=x8.device, dtype=x8.dtype)
    out.view(torch.uint8)[:, :, pos] = x8.view(torch.uint8).transpose(1, 2)
    return out


@triton.jit
def _twin_pi_kernel(X, XT, N, D: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)
    c = tl.arange(0, BS)
    # source row for physical position c: m = SIGMA[c] (per 16-window), i.e. pi^-1
    m = (c & 1) | ((c & 2) << 2) | ((c & 12) >> 1) | (c & -16)
    rs = pid_s * BS + m
    rd = pid_d * BD + tl.arange(0, BD)
    base = pid_bh.to(tl.int64) * N * D
    x = tl.load(X + base + rs.to(tl.int64)[:, None] * D + rd[None, :])          # [BS, BD] bytes, rows gathered
    ps = pid_s * BS + c
    tl.store(XT + base + rd.to(tl.int64)[:, None] * N + ps[None, :], tl.trans(x))   # contiguous store


def twin_pi(x8: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """x8: [BH, N, D] e4m3 contiguous -> [BH, D, N] e4m3 contiguous, pi-ordered N positions."""
    assert x8.dim() == 3 and x8.is_contiguous() and x8.dtype == torch.float8_e4m3fn, (x8.shape, x8.dtype)
    BH, N, D = x8.shape
    assert N % 64 == 0 and D % 64 == 0
    if out is None:
        out = torch.empty(BH, D, N, device=x8.device, dtype=x8.dtype)
    BS, BD = 64, 64
    _twin_pi_kernel[(N // BS, D // BD, BH)](x8.view(torch.uint8), out.view(torch.uint8), N,
                                             D=D, BS=BS, BD=BD, num_warps=4)
    return out


def selftest(verbose: bool = True) -> bool:
    ok = True
    torch.manual_seed(0)
    for (BH, N, D) in [(32, 8192, 256), (4, 2048, 256), (12, 4096, 256)]:
        x = torch.randn(BH, N, D, device="cuda").to(torch.float8_e4m3fn)
        ref = twin_pi_reference(x)
        got = twin_pi(x)
        e = torch.equal(ref.view(torch.uint8), got.view(torch.uint8))
        ok &= e
        if verbose:
            print(f"twin_pi BH{BH} N{N} D{D}: bitwise={e}")
    return ok


if __name__ == "__main__":
    import time
    print("SELFTEST", "PASS" if selftest() else "FAIL")
    x = torch.randn(32, 8192, 256, device="cuda").to(torch.float8_e4m3fn)

    def bench(f, n=20):
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3
    print(f"twin_pi [32, 8192, 256]: {bench(lambda: twin_pi(x)):.3f} ms")
