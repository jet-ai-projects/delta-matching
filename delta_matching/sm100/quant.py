"""sm100-mx producers: e4m3 codes with one power-of-2 (UE8M0-representable) scale per 32 tokens over all of head_dim.

quant_rows(x, prescale, mean)  x (B, N, H, D) bf16/fp32 (any strides, unit stride on D)
    y = (x - mean[b, h]) * prescale            (fp32; mean optional, (B, H, D) fp32)
    per (b, h, 32-token block):  amax = max |y| = m * 2^E (m in [1, 2), fp32 bits);
        scale = the smallest power of 2 with amax <= 448 * scale  = 2^(E-8) if m <= 1.75 else 2^(E-7)
        (amax == 0 -> scale = 1);  code = rn_e4m3(y / scale)  (exact multiply by 2^-e; never exceeds 448)
    returns codes (B, N, H, D) float8_e4m3fn contiguous, scales (B, H, N // 32) fp32 (exact powers of 2)

The scale exponent biased by 127 is the UE8M0 byte used by the tcgen05 block-scaled MMAs.

every producer is a CuTe DSL kernel (quant_cute.py, compiled with --enable-tvm-ffi), bitwise equal to
the former Triton kernels incl. their fp32 reduction trees.  D (128 or 256) is taken from
the tensors and is a compile-time parameter.  No Triton.
"""
import threading

import torch

from . import quant_cute as qc

BLK = 32
_U8 = torch.uint8
_LOCK = threading.Lock()
_FNS = {}
_CACHE = []


def _cache():
    """Persistent compile cache under the package fingerprint (FA4 + delta_matching/sm100/*.py sources + compile_env)."""
    if not _CACHE:
        from flash_attn.cute import cache_utils
        from .fwd import _fingerprint
        if cache_utils.CUTE_DSL_CACHE_ENABLED:
            _CACHE.append(cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / _fingerprint("quant") / "quant"))
        else:
            _CACHE.append(cache_utils.JITCache())
    return _CACHE[0]


def _fake(dtype, rank):
    import cutlass.cute as cute
    return cute.runtime.make_fake_tensor(dtype, tuple(cute.sym_int() for _ in range(rank)),
                                         stride=tuple(cute.sym_int64() for _ in range(rank)), assumed_align=1)


def _compile(key, obj, sig):
    """sig: list of ('t', cutlass dtype, rank) / ('i',) / ('f',) in call order (stream appended)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Float32
    args = []
    for a in sig:
        if a[0] == "t":
            args.append(_fake(a[1], a[2]))
        elif a[0] == "l":
            args.append(Int64(0))
        elif a[0] == "i":
            args.append(Int32(0))
        else:
            args.append(Float32(0.0))
    # explicit stream argument (current raw stream, passed by the host like Triton's launcher) and Int64 pointers only:
    # no TVM-FFI tensor marshalling / env-stream lookup
    args.append(cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False))
    from . import fastpath
    fastpath.EPOCH[0] += 1
    return cute.compile(obj, *args, options="--enable-tvm-ffi")


def _get(key, make, sig):
    f = _FNS.get(key)
    if f is None:
        with _LOCK:
            f = _FNS.get(key)
            if f is None:
                c = _cache()
                f = c.get(key) if hasattr(c, "get") else None
                if f is None:
                    f = _compile(key, make(), sig)
                    c[key] = f
                _FNS[key] = f
    return f


def _vec(t, ndim=3):
    """16-byte vector loads are legal: base and every outer stride (bytes) multiples of 16."""
    es = t.element_size()
    return t.data_ptr() % 16 == 0 and all((s * es) % 16 == 0 for s in t.stride()[:ndim])


def _tri_aligned(t, nheads):
    """Triton's specialization picks the 16-byte K layout: pointer % 16, the batch / token strides
    % 16 (elements), the head stride % 16 unless nheads == 1 (specialized away)."""
    sb, sn, sh = t.stride()[:3]
    return t.data_ptr() % 16 == 0 and sb % 16 == 0 and sn % 16 == 0 and (nheads == 1 or sh % 16 == 0)


_RAWSTREAM = torch._C._cuda_getCurrentRawStream


def _stream(t):
    """The current CUDA stream of t's device (raw handle), as Triton's launcher used."""
    return _RAWSTREAM(t.get_device())


def _u8(t):
    return t.view(_U8)


TILE_CFG = {128: (2, 1, 0), 256: (4, 1, 1)}   # Tile: (warps per 32-row tile, tiles per CTA, re-convert rows per pass)


def _tile_fn(D, x, o, has_mean, has_pre, w_off, delta, mean_t=None, wb_t=None):
    """Compiled generic tile. Pointers are passed as Int64 addresses (TVM-FFI tensor marshalling costs ~1.5 us per tensor);
    the codes output stays a tensor (device / env-stream anchor)."""
    import cutlass
    vx = _vec(x)
    vo = _vec(o) if delta is not None else False
    xf32 = x.dtype == torch.float32
    nw, tp, rc = TILE_CFG[D]
    key = ("tile", D, vx, has_mean, has_pre, w_off, delta, vo, xf32, nw, tp, rc)
    sig = [("l",), ("l",), ("l",), ("l",), ("l",), ("l",), ("l",)] + [("i",)] * 9 + [("f",)]
    return _get(key, lambda: qc.Tile(D, vx, has_mean, has_pre, w_off, delta, vo, xf32, nw, tp, bool(rc)), sig)


_FAST = {}


def _tile(x, prescale=None, mean=None, o=None, delta=None, y=None, s=None, dl=None, wb=None, w_off=None):
    """Generic 32-row tile launch: codes y, scales s, optional delta dl, optional UE8M0 bytes wb (uint8 1-D view)."""
    B, N, H, D = x.shape
    xp = x.data_ptr()
    op = o.data_ptr() if o is not None else 0
    k = ("t", x.shape, x.stride(), x.dtype, xp & 15, mean is not None, prescale is not None, w_off, delta,
         None if o is None else (o.stride(), o.dtype, op & 15))
    f = _FAST.get(k)
    if f is None:
        f = _FAST[k] = _tile_fn(D, x, o, mean is not None, prescale is not None, w_off, delta)
    so = o.stride() if o is not None else (0, 0, 0)
    sx = x.stride()
    f(xp, op, mean.data_ptr() if mean is not None else 0, y.data_ptr(), s.data_ptr(),
      dl.data_ptr() if dl is not None else 0, wb.data_ptr() if wb is not None else 0, N, H, B, sx[0], sx[1], sx[2],
      so[0], so[1], so[2], float(prescale) if prescale is not None else 1.0, _stream(x))


def quant_rows(x: torch.Tensor, prescale: float = 1.0, mean: torch.Tensor | None = None):
    B, N, H, D = x.shape
    assert N % BLK == 0 and x.stride(-1) == 1
    y = torch.empty(B, N, H, D, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(B, H, N // BLK, device=x.device, dtype=torch.float32)
    _tile(x, prescale=prescale, mean=mean.contiguous() if mean is not None else None, y=y, s=s)
    return y, s


def kmean(k: torch.Tensor) -> torch.Tensor:
    """(B, N, Hk, D) -> (B, Hk, D) fp32 mean over tokens (K centering)."""
    return k.float().mean(dim=1)


def quant_rows_ref(x: torch.Tensor, prescale: float = 1.0, mean: torch.Tensor | None = None):
    """Pure-torch reference of quant_rows (bitwise target)."""
    B, N, H, D = x.shape
    y = x.float()
    if mean is not None:
        y = y - mean[:, None, :, :]
    y = y * prescale
    yb = y.view(B, N // BLK, BLK, H, D)
    amax = yb.abs().amax(dim=(2, 4))                              # (B, N/32, H)
    ab = amax.contiguous().view(torch.int32)
    eb = ((ab >> 23) & 0xFF) - 8 + ((ab & 0x7FFFFF) > 0x600000).to(torch.int32)
    eb = eb.clamp(min=1, max=254)
    eb = torch.where(amax > 0, eb, torch.full_like(eb, 127))
    scale = (eb << 23).view(torch.float32)
    inv = ((254 - eb) << 23).view(torch.float32)
    codes = (yb * inv[:, :, None, :, None]).to(torch.float8_e4m3fn).view(B, N, H, D)
    return codes, scale.permute(0, 2, 1).contiguous()


# ----------------------------------------------------------------------------------------------------------------------
# Fused producers: one launch for Q, K, V (+ packed UE8M0 K/V words for the forward kernel), and a
# deterministic two-stage K mean.  Bitwise equal to quant_rows(...) for the same mean.
# ----------------------------------------------------------------------------------------------------------------------
def _kmp_fn(k):
    import cutlass
    D, Hk = k.shape[-1], k.shape[2]
    vk, kal = _vec(k), _tri_aligned(k, Hk)
    return _get(("kmeanp", D, vk, kal), lambda: qc.KMeanPartial(D, vk, kal),
                [("l",), ("l",)] + [("i",)] * 7)


def kmean_fast(k: torch.Tensor, rows: int = 512) -> torch.Tensor:
    """(B, N, Hk, D) -> (B, Hk, D) fp32 mean over tokens; deterministic (fixed-order partials, no atomics)."""
    import cutlass
    B, N, Hk, D = k.shape
    rows = min(rows, N)
    while N % rows:
        rows //= 2
    assert rows % 32 == 0
    S = N // rows
    part = torch.empty(B * Hk, S, D, device=k.device, dtype=torch.float32)
    f = _kmp_fn(k)
    f(k.data_ptr(), part.data_ptr(), N, Hk, B, k.stride(0), k.stride(1), k.stride(2), rows, _stream(k))
    return (part.sum(dim=1) / N).view(B, Hk, D)


def quant_qkv(q, k, v, sm: float, mean: torch.Tensor | None = None):
    """Fused producer: returns (q8, qs, k8, ks, v8, vs, kvw, mean); kvw (B, Hk, N/64) int32 = forward kernel's
    packed K/V UE8M0 words ([2t] K bytes of 128-key tile t, [2t+1] V bytes).  N % 128 == 0."""
    B, N, H, D = q.shape
    Hk = k.shape[2]
    assert N % 128 == 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    if mean is None:
        mean = kmean_fast(k)
    mean = mean.contiguous()
    dev = q.device
    e4 = torch.float8_e4m3fn
    q8 = torch.empty(B, N, H, D, device=dev, dtype=e4)
    k8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    v8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    qs = torch.empty(B, H, N // 32, device=dev, dtype=torch.float32)
    ks = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    vs = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    kvw = torch.empty(B, Hk, N // 64, device=dev, dtype=torch.int32)
    wb = kvw.view(_U8).view(-1)
    _tile(q, prescale=sm, y=q8, s=qs)
    _tile(k, mean=mean, y=k8, s=ks, wb=wb, w_off=0)
    _tile(v, y=v8, s=vs, wb=wb, w_off=1)
    return q8, qs, k8, ks, v8, vs, kvw, mean


PRODA_NW = {256: 4}      # D=256: warp-pair ProdA (one tile / one K-mean chunk stream per pair); D=128: chunked ProdAC
_CNT = {}
_CNT_KEEP = []


def _prod_fn(q, k, v):
    """The Q/V/K-mean kernel for these inputs (D=128: ProdAC, chunk CTAs; D=256: ProdA, 4 warps), both with the head-level
    mean fold (the K tiles then only quantize)."""
    D, Hk = q.shape[-1], k.shape[2]
    vq, vk, vv, kal = _vec(q), _vec(k), _vec(v), _tri_aligned(k, Hk)
    if D == 128:
        return _get(("prod_ac", D, vq, vk, vv, kal, True), lambda: qc.ProdAC(D, vq, vk, vv, kal, True),
                    [("l",)] * 12 + [("i",)] * 13 + [("f",), ("i",)])
    nw = PRODA_NW[D]
    return _get(("prod_a", D, vq, vk, vv, kal, nw, True), lambda: qc.ProdA(D, vq, vk, vv, kal, nw, True),
                [("l",)] * 11 + [("i",)] * 13 + [("f",), ("i",)])


def _counters(q, st, n):
    """Per-(device, stream) int32 K-mean arrival counters, zero at rest: every counter is reset by the CTA that completes
    it (kernel completion orders that before the next launch on the stream).  A launch cannot stop half-way without a
    sticky CUDA error (device fault -> the context is unusable), and a failed launch never ran; still, any exception from
    a launch re-zeroes the counters (_guard), and DM_CQ_CHECK=1 asserts they are zero after every call (debug).
    Grown-from buffers are kept alive (a captured CUDA graph may point at them)."""
    ck = (q.get_device(), st)
    e = _CNT.get(ck)
    if e is None or e.numel() < n:
        if e is not None:
            _CNT_KEEP.append(e)
        e = _CNT[ck] = torch.zeros(max(n, 1024), device=q.device, dtype=torch.int32)
    return e


_CHECK = __import__("os").environ.get("DM_CQ_CHECK") == "1"     # debug only (read once at import)


def _guard(cnt, launch):
    try:
        launch()
    except BaseException:
        cnt.zero_()
        raise
    if _CHECK and not torch.cuda.is_current_stream_capturing():     # a sync is illegal inside CUDA-graph capture
        torch.cuda.synchronize()
        assert int(cnt.count_nonzero()) == 0, "cutequant: K-mean counters not zero after a call"


def quant_qkv2(q, k, v, sm: float, rows: int = 1024, warps_a: int = 2, warps_b: int = 2, tr: int = 32):
    """Two-launch fused producer.  Returns (q8, qs, k8, ks, v8, vs, kvw, mean); kvw (B, Hk, N/64) int32 packed UE8M0
    words ([2t] K bytes of 128-key tile t, [2t+1] V bytes).  The K mean is (sequential sum of ROWS-row fp32 partial
    sums, each a sum of 64-row chunk sums) / N (div.full).  Bitwise = Triton's _prod_a / _prod_b.
    Launch 1: Q / V tiles + K-mean (chunks or partials; the CTA completing a partial folds its chunks in chunk order, the
    CTA completing a head's last partial folds the partials in order and writes the mean).  Launch 2: K tiles."""
    B, N, H, D = q.shape
    Hk = k.shape[2]
    sq, sk, sv = q.stride(), k.stride(), v.stride()
    qp, kp, vp = q.data_ptr(), k.data_ptr(), v.data_ptr()
    key = (q.shape, k.shape, sq, sk, sv, qp & 15, kp & 15, vp & 15, q.dtype, k.dtype, v.dtype, rows, warps_a, warps_b, tr)
    e = _FAST.get(key)
    if e is None:
        assert N % 128 == 0 and sq[-1] == 1 and sk[-1] == 1 and sv[-1] == 1
        assert (warps_a, warps_b, tr) == (2, 2, 32), "cutequant: only the shipped producer layout (2, 2, 32) is ported"
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        ROWS = rows
        while N % ROWS:
            ROWS //= 2
        assert ROWS % 64 == 0 and ROWS <= 1024, "64-row chunks, <= 16 per partial"
        e = _FAST[key] = (_prod_fn(q, k, v), ROWS, N // ROWS)
    fa, ROWS, S = e
    dev = q.device
    st = _stream(q)
    e4 = torch.float8_e4m3fn
    q8 = torch.empty(B, N, H, D, device=dev, dtype=e4)
    k8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    v8 = torch.empty(B, N, Hk, D, device=dev, dtype=e4)
    qs = torch.empty(B, H, N // 32, device=dev, dtype=torch.float32)
    ks = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    vs = torch.empty(B, Hk, N // 32, device=dev, dtype=torch.float32)
    kvw = torch.empty(B, Hk, N // 64, device=dev, dtype=torch.int32)
    part = torch.empty(B * Hk, S, D, device=dev, dtype=torch.float32)
    mean = torch.empty(B, Hk, D, device=dev, dtype=torch.float32)
    args = (N, H, Hk, B, sq[0], sq[1], sq[2], sk[0], sk[1], sk[2], sv[0], sv[1], sv[2], float(sm), ROWS, st)
    if D == 128:
        chunks = torch.empty(B * Hk * S * (ROWS // 64), D, device=dev, dtype=torch.float32)
        cnt = _counters(q, st, B * Hk * S + B * Hk)
        _guard(cnt, lambda: fa(qp, kp, vp, q8.data_ptr(), v8.data_ptr(), qs.data_ptr(), vs.data_ptr(), kvw.data_ptr(),
                               part.data_ptr(), chunks.data_ptr(), cnt.data_ptr(), mean.data_ptr(), *args))
    else:
        cnt = _counters(q, st, B * Hk)
        _guard(cnt, lambda: fa(qp, kp, vp, q8.data_ptr(), v8.data_ptr(), qs.data_ptr(), vs.data_ptr(), kvw.data_ptr(),
                               part.data_ptr(), cnt.data_ptr(), mean.data_ptr(), *args))
    _tile(k, mean=mean, y=k8, s=ks, wb=kvw, w_off=0)      # K tiles from the final mean (= _prod_b's quantization)
    return q8, qs, k8, ks, v8, vs, kvw, mean


def quant_do_delta(do: torch.Tensor, o: torch.Tensor | None):
    """(dO codes, scales, delta or None); delta = rowsum(dO * O) (stale mode) when o is given.  (The former obar /
    DEQ_DELTA mode was never on the API path and is dropped.)"""
    return _quant_do_delta_stale(do, o)


def _quant_do_delta_stale(do: torch.Tensor, o: torch.Tensor | None):
    """(dO codes, scales, delta or None) in one pass; delta = rowsum(dO * O) fp32 (B, H, N) when o is given."""
    B, N, H, D = do.shape
    assert do.stride(-1) == 1 and (o is None or o.stride(-1) == 1)
    assert do.dtype == torch.bfloat16 and (o is None or (o.dtype == torch.bfloat16 and o.shape == do.shape))
    y = torch.empty(B, N, H, D, device=do.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(B, H, N // BLK, device=do.device, dtype=torch.float32)
    dl = torch.empty(B, H, N, device=do.device, dtype=torch.float32) if o is not None else None
    _tile(do, o=o, delta="stale" if o is not None else None, y=y, s=s, dl=dl)
    return y, s, dl
