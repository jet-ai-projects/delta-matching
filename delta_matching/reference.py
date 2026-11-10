"""Pure-PyTorch reference of Delta-Matching attention (forward + both backward delta modes), for any device.

No Triton, no compiled library, no relative imports: the file can be loaded standalone by path (the sm100 tests do)
and runs on CPU (small shapes) as well as on sm90 / sm100.

  attention_fp32(q, k, v, do, causal)            exact attention in fp32 (fp64 with dtype=torch.float64), autograd grads
  dm_reference(q, k, v, do, causal, mode, preset) emulation of an fp8 Delta-Matching kernel: every quantization point of
                                                  the kernel is reproduced in torch; returns O, LSE, dQ, dK, dV, delta and
                                                  diagnostics (row sums of dS, column sums of dK)
  PRESETS                                         the quantization schemes: "sm90" (the shipped Hopper kernel), "mxfp8"
                                                  (OCP MX block-32 e4m3 with e8m0 scales along every matmul's contraction
                                                  dim, the sm100 tcgen05 block-scaled format), "bf16" (no fp8, a control)

Layout: q (B, N, H, D), k / v (B, N, Hk, D), do (B, N, H, D) - flash-attention layout, any float dtype (bf16 in practice).
Outputs in the same layout, fp32 (the caller rounds to bf16 to compare against a bf16 kernel output).

What the emulation is exact about and what it is not: quantization (scales, rounding, saturation, which tensor is
quantized with which block) is reproduced exactly; fp32 matmul accumulation order (tensor-core MMA trees) and the
hardware exp2 approximation are not, so a correct kernel sits at fp32 round-off * (fp8 code flips this causes) from
the emulation, far below the fp8 floor against exact attention.  The sm100 tests rely on exactly that gap.
"""

import math
from dataclasses import dataclass, field, replace

import torch

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
TINY = 1e-30
INV_E4M3_MAX = float(torch.tensor(1.0 / 448.0, dtype=torch.float32))   # fp32(1/448): the dO producer multiplies by it
LOG2E = 1.4426950408889634


# ----------------------------------------------------------------------------------------------------------------- quant
def e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values to the nearest e4m3 value (RNE, saturating at +-448), returned as fp32."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).to(torch.float32)


def quant_blockrows(x: torch.Tensor, blk: int, mul: bool = False):
    """The sm90 producers' quantizer: x (..., N, D) fp32 -> (codes fp32-valued e4m3, scales (..., N // blk) fp32).
    One scale per block of `blk` rows (all D columns): s = max(absmax, 1e-30) / 448, code = e4m3_rne(x / s) (IEEE
    division, clamped).  Dequantized value = code * s."""
    *lead, N, D = x.shape
    assert N % blk == 0, (N, blk)
    xb = x.reshape(*lead, N // blk, blk * D)
    amax = xb.abs().amax(-1)
    s = amax.clamp_min(TINY) * INV_E4M3_MAX if mul else amax.clamp_min(TINY) / E4M3_MAX
    codes = e4m3_round(xb / s.unsqueeze(-1)).reshape(*lead, N, D)
    return codes, s


def e8m0_scale(amax: torch.Tensor, rounding: str = "ceil") -> torch.Tensor:
    """The power-of-two (e8m0) scale of an MX block from its absmax (fp32 result).
    "ceil":  smallest 2^e with amax / 2^e <= 448 (no saturation; NVIDIA Transformer Engine / cuBLAS MXFP8 recipe).
    "floor": OCP MX v1.0: 2^(floor(log2 amax) - 8) (e4m3 emax = 8); values above 448 * 2^e saturate.
    An all-zero block gets 2^-127 (the smallest e8m0 value)."""
    a = amax.to(torch.float32)
    if rounding == "ceil":
        e = torch.ceil(torch.log2(a / E4M3_MAX))
        # guard log2 rounding: make sure a / 2^e <= 448 and a / 2^(e-1) > 448
        e = torch.where(a / torch.exp2(e) > E4M3_MAX, e + 1, e)
        e = torch.where(a / torch.exp2(e - 1) <= E4M3_MAX, e - 1, e)
    elif rounding == "floor":
        m, ex = torch.frexp(a)            # a = m * 2^ex, m in [0.5, 1): floor(log2 a) = ex - 1, exact
        e = (ex - 1 - 8).to(torch.float32)
    else:
        raise ValueError(rounding)
    e = torch.where(a > 0, e, torch.full_like(e, -127.0)).clamp(-127, 127)
    return torch.exp2(e)


def quant_mx(x: torch.Tensor, axis: int, block: int = 32, rounding: str = "ceil"):
    """MXFP8 (e4m3 elements, one e8m0 scale per `block` consecutive elements along `axis`, the matmul's contraction
    dim).  Returns (codes fp32-valued e4m3, scales fp32 powers of two, shape of x with axis divided by block)."""
    axis = axis % x.dim()
    xt = x.movedim(axis, -1)
    *lead, K = xt.shape
    assert K % block == 0, (K, block)
    xb = xt.reshape(*lead, K // block, block)
    s = e8m0_scale(xb.abs().amax(-1), rounding)
    codes = e4m3_round(xb / s.unsqueeze(-1))
    return codes.reshape(*lead, K).movedim(-1, axis), s.movedim(-1, axis)


def dq_mx(codes: torch.Tensor, s: torch.Tensor, axis: int, block: int = 32) -> torch.Tensor:
    return codes * s.repeat_interleave(block, dim=axis % codes.dim())


def fake_mx(x: torch.Tensor, axis: int, block: int = 32, rounding: str = "ceil") -> torch.Tensor:
    c, s = quant_mx(x, axis, block, rounding)
    return dq_mx(c, s, axis, block)


def pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(amax / 448)) (no saturation, tightest power of two); amax == 0 -> 1."""
    s = e8m0_scale(amax, "ceil")
    return torch.where(amax > 0, s, torch.ones_like(s))


def fake_blockrows_pow2(x: torch.Tensor, blk: int) -> torch.Tensor:
    *lead, N, D = x.shape
    xp = _pad_to(x, blk, 1)
    xb = xp.reshape(*lead, xp.shape[-2] // blk, blk * D)
    s = pow2_scale(xb.abs().amax(-1, keepdim=True))
    return (e4m3_round(xb / s) * s).reshape(xp.shape)[..., :N, :]


def fake_mx2d(x: torch.Tensor, r_: int, c_: int) -> torch.Tensor:
    R_, C = x.shape[-2:]
    xp = _pad_to(x, r_, c_)
    Rp, Cp = xp.shape[-2:]
    t = xp.reshape(*xp.shape[:-2], Rp // r_, r_, Cp // c_, c_)
    s = pow2_scale(t.abs().amax(dim=(-3, -1), keepdim=True))
    return (e4m3_round(t / s) * s).reshape(xp.shape)[..., :R_, :C]


def e4m3_direct(y, mode="rn"):
    """e4m3 code value of 2^y built directly from y (no exp then round), for the candidate "direct code" P:
      "rn"        reference: correctly rounded e4m3(2^y)
      "lin_floor" exponent e = floor(y), mantissa m = floor(8 * frac(y))        (value 2^e (1 + m/8); linear in log domain)
      "lin_round" m = round-half-even(8 * frac(y)), carry into the exponent at m = 8
      "exp_floor" m = floor(8 * (2^frac(y) - 1))  (truncation instead of rounding of the exact mantissa)
    Normal range only for the direct modes (e >= -6); below that the subnormal grid 2^-9 is used with round-to-nearest of
    2^y (codes that small carry no weight in P ~ 2^(k-7) code).  y = -inf -> 0.  Values above 448 saturate."""
    if mode == "rn":
        return e4m3_round(torch.exp2(y))
    e = torch.floor(y)
    f = y - e
    if mode == "lin_floor":
        m = torch.floor(8 * f)
    elif mode == "lin_round":
        m = torch.round(8 * f)
    elif mode == "exp_floor":
        m = torch.floor(8 * (torch.exp2(f) - 1))
    else:
        raise ValueError(mode)
    carry = m >= 8
    e = torch.where(carry, e + 1, e)
    m = torch.where(carry, torch.zeros_like(m), m)
    v = torch.exp2(e) * (1 + m / 8)
    sub = e < -6
    v = torch.where(sub, torch.round(torch.exp2(y) / 2 ** -9) * 2 ** -9, v)
    v = torch.where(torch.isfinite(y), v, torch.zeros_like(v))
    return v.clamp(max=E4M3_MAX)


def code_diff_fraction(y, mode):
    """Fraction of codes (over finite y) where the direct-code mode differs from the correctly rounded e4m3(2^y), and the
    mean signed relative difference (bias) of the direct value."""
    a, b = e4m3_direct(y, mode), e4m3_direct(y, "rn")
    live = torch.isfinite(y) & (b > 0)
    diff = (a != b) & live
    bias = ((a - b) / b.clamp_min(1e-30))[live].mean().item() if live.any() else 0.0
    return diff.sum().item() / max(1, live.sum().item()), bias


def mx_pfwd(P, ctx, blk=32, mode="rn"):
    """the sm100 forward P: x = S log2(e); per (row, blk-key block) k = floor(max x);
    code = e4m3(2^(x - k + 7)), value code * 2^(k - 7).  Independent of the running max (weights are exact powers of 2).
    Returns P-hat in the units of P = exp(S - rowmax)."""
    S = ctx.S
    R_, C = S.shape[-2:]
    x = _pad_to(S, 1, blk, float("-inf")) * LOG2E
    xb = x.reshape(*x.shape[:-1], x.shape[-1] // blk, blk)
    kk = torch.floor(xb.amax(-1, keepdim=True))
    xm = (ctx.rowmax * LOG2E).unsqueeze(-1)                             # (..., R, 1, 1)
    kk = torch.where(torch.isfinite(kk), kk, xm.expand_as(kk))          # blocks with no live key: codes are 0
    ph = e4m3_direct(xb - kk + 7, mode) * torch.exp2(kk - 7 - xm)
    return ph.reshape(x.shape)[..., :C]


def fake_blockrows(x: torch.Tensor, blk: int, mul: bool = False) -> torch.Tensor:
    c, s = quant_blockrows(x, blk, mul)
    *lead, N, D = x.shape
    return (c.reshape(*lead, N // blk, blk, D) * s[..., None, None]).reshape(x.shape)


# ----------------------------------------------------------------------------------------------------------------- spec
@dataclass(frozen=True)
class Preset:
    """A quantization scheme of an fp8 attention kernel.  Every quantization site of the forward and backward:
      q, k, v, do     the producer quantizers of the inputs (k after centering, q after the softmax-scale bake)
      p_fwd           P in the forward's PV matmul;     p_bwd   P in dV = P^T dO
      dp              dP = dO V^T is fp32 out of the MMA (only its inputs are quantized)
      ds              dS in dK = dS^T Q and dQ = dS K
    Each site is a scheme name understood by `_site`: "none" (fp32), "bf16", "blockrows:<rows>", "mx:<axis>", or a
    preset-specific scheme ("sm90_pfwd", "sm90_pbwd", "sm90_ds")."""
    name: str
    q: str
    k: str
    v: str
    do: str
    p_fwd: str
    p_bwd: str
    ds: str
    # separate quantizations of the same tensor for its two uses (MX quantizes along the contraction dim, so the
    # transposed operand of the other matmul gets its own scales): q2 for dK = dS^T Q, k2 for dQ = dS K,
    # v2 unused (V is only in PV and dP), do2 for dV = P^T dO, ds2 for dQ (ds is the dK operand)
    q2: str | None = None
    k2: str | None = None
    do2: str | None = None
    ds2: str | None = None
    vd: str | None = None        # V operand of dP = dO V^T when set (MX: blocked along D); default: v, or k's scheme if q2 is set
    center_k: bool = True        # K centered by its per-head mean before quantization (softmax-invariant)
    delta_norm: bool = False     # matched delta = sum_j P dP / sum_j P (instead of sum_j P dP)
    l_from_codes: bool = False   # forward row sum l from the quantized P used in PV (O a convex combination of V rows)
    delta_from_o: bool = False   # matched delta = dequant(dO codes) . O_fp32 (forward's normalized fp32 output, no pass 0)
    mx_rounding: str = "ceil"
    notes: str = ""


PRESETS: dict[str, Preset] = {}


def _register(p: Preset):
    PRESETS[p.name] = p
    return p


# the shipped sm90 hd128 kernel (Delta-Matching-128 v3); the P / dS sites are filled in from NUMERICS-hd128.md
_register(Preset("sm90", q="blockrows:128", k="blockrows:64", v="blockrows:64", do="blockrows_mul:128",
                 p_fwd="sm90_pfwd", p_bwd="sm90_pbwd", ds="sm90_ds",
                 notes="the Hopper kernel, head_dim 128, branch main : Q / dO 128-row and "
                       "K / V 64-row fp32 block scales; forward P e4m3 with a per (row, 64-key group) basis under a "
                       "descending 256-key-tile online max; backward P as e4m3(2^8 P); dS per 64x128 tile scale psi on "
                       "2^8 ks_j dS; matched delta = sum_j P dP with the unquantized P"))
# MXFP8 everywhere: each operand quantized in blocks of 32 along the contraction dim of the matmul that consumes it.
#   S = Q K^T     contracts D:      q mx:-1, k mx:-1
#   O = P V       contracts keys:   p_fwd mx:-1 (P rows along keys), v mx:-3 (V along seqlen; (B,N,H,D) -> axis 1)
#   dV = P^T dO   contracts queries: p_bwd along queries, do2 along seqlen
#   dP = dO V^T   contracts D:      do mx:-1, v (D-blocked copy) -> k-like; we reuse 'v' only for PV, see _vD
#   dK = dS^T Q   contracts queries: ds along queries, q2 along seqlen
#   dQ = dS K     contracts keys:   ds2 along keys, k2 along seqlen
_register(Preset("mxfp8", q="mx:D", k="mx:D", v="mx:N", do="mx:D", p_fwd="mx:keys", p_bwd="mx:queries",
                 ds="mx:queries", q2="mx:N", k2="mx:N", do2="mx:N", ds2="mx:keys",
                 notes="OCP MX e4m3, block 32 along every contraction dim, e8m0 scales (ceil rounding: no saturation)"))
# sm100 proposal: inputs as sm90, P and dS e4m3 with a UE8M0 scale per row per 32 keys (ceil rounding)
_register(Preset("mxfp8-p", q="blockrows:128", k="blockrows:64", v="blockrows:64", do="blockrows_mul:128",
                 p_fwd="mx:keys", p_bwd="mx:keys", ds="mx:keys",
                 notes="sm90 input quantizers; P (PV, dV) and dS (dK, dQ) MX e4m3 per row per 32 keys"))
PRESETS["mxfp8-all"] = PRESETS["mxfp8"]
# hd256 contract of the shipped sm90 kernel: Q / dO one fp32 scale per 64 rows, K / V one per 32 keys
# (hd128 was [128,64]). Inputs exact to that contract; the P / dS sites reuse the hd128 sm90 model (an approximation until the
# d256 P-hat / dS tiling is analysed) - used as the hd256 ACCURACY FLOOR, not as a bitwise spec.
_register(Preset("sm90-d256", q="blockrows:64", k="blockrows:32", v="blockrows:32", do="blockrows_mul:64",
                 p_fwd="sm90d256_pfwd", p_bwd="sm90_pbwd", ds="sm90d256_ds", delta_norm=True,
                 notes="sm90 hd256: [64,32] fp32 input scales, P-hat per (row, 32-key group) "
                       "under a descending 128-key online max (12-binade floor), P' = 2^8 P, dS psi per 64x64; delta = sum P dP / sum P "
                       "(num/den as at hd128 - rowsum(dS)=0 by the method's definition; measured equal "
                       "to the sm90 no-division form up to rounding); F2b psi floor not modelled; hd256 accuracy floor"))
# info-only: the sm90 delta form without / sum P (breaks rowsum(dS)=0 at fp32 round-off) - never accepted
_register(replace(PRESETS["sm90-d256"], name="sm90-d256-nodiv", delta_norm=False,
                  notes="info only: sm90-d256 with the no-division delta of the sm90 kernel"))
# the one accepted swap - backward P^T for dV as pow2 MX per (key x 32 queries); measured 1.000 vs the floor
_register(replace(PRESETS["sm90-d256"], name="sm90-d256-ptmx", p_bwd="mx:queries:32",
                  notes="sm90-d256 with P^T for dV quantized MX pow2 per (key x 32 queries); accepted swap (ablation 1.000)"))


def make_mx_preset(name, tok_block=32, pow2=True, p_block=32, pt_block=32, ds_block=(32, 32), delta_norm=True,
                    register=True, l_from_codes=False, p_fwd=None, delta_from_o=False, p_code="rn"):
    """the sm100 numerics, with knobs:
      tok_block   Q / K / V / dO: one scale per tok_block tokens over all D (pow2: 2^ceil(log2(amax/448)), else fp32 amax/448)
      p_block     forward P: per (row, p_block keys) block exponent k = floor(max S log2e), code e4m3(2^(x-k+7))
      pt_block    backward P^T for dV: pow2 scale per (key, pt_block queries)
      ds_block    dS (one code set for dK and dQ): pow2 scale per (queries x keys) 2-D block
      delta_norm  matched delta = sum P dP / sum P (True) or sum P dP (False, sm90 style)"""
    inp = f"blockrows_pow2:{tok_block}" if pow2 else f"blockrows:{tok_block}"
    pr = Preset(name, q=inp, k=inp, v=inp, do=inp, p_fwd=p_fwd or (f"mx_pfwd:{p_block}" + ("" if p_code == "rn" else f":{p_code}")), p_bwd=f"mx:queries:{pt_block}",
                ds=f"mx2d:{ds_block[0]}x{ds_block[1]}", delta_norm=delta_norm, l_from_codes=l_from_codes,
                delta_from_o=delta_from_o,
                notes=f"sm100 numerics: inputs {inp}, P fwd per row x {p_block} keys, P^T per key x {pt_block} "
                      f"queries, dS {ds_block[0]}x{ds_block[1]} blocks, delta_norm={delta_norm}")
    return _register(pr) if register else pr


make_mx_preset("mx-v1")
make_mx_preset("mx-v1-nonorm", delta_norm=False)
_register(Preset("bf16", q="bf16", k="bf16", v="bf16", do="bf16", p_fwd="bf16", p_bwd="bf16", ds="bf16",
                 notes="no fp8 anywhere: bf16 operands, fp32 accumulation (the control a cheating candidate matches)"))
_register(Preset("fp32", q="none", k="none", v="none", do="none", p_fwd="none", p_bwd="none", ds="none",
                 center_k=False, notes="no rounding at all (sanity: equals attention_fp32 up to accumulation order)"))


# ----------------------------------------------------------------------------------------------------------------- sites
# Tensors inside the engine are (..., rows, cols): inputs (B, H, N, D); P / dS (B, H, Nq, Nk).
_AXES = {"D": -1, "N": -2, "keys": -1, "queries": -2}


class _Ctx:
    """What a site quantizer may need besides the tensor (row max of the forward, causal mask, ...)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


SITE_FNS = {}     # preset-specific schemes: name -> fn(x, ctx, preset) -> fake-quantized fp32 tensor


def _site(scheme: str, x: torch.Tensor, ctx: _Ctx, preset: Preset) -> torch.Tensor:
    if scheme == "none":
        return x
    if scheme == "bf16":
        return x.to(torch.bfloat16).to(torch.float32)
    if scheme.startswith("blockrows_mul:"):
        return fake_blockrows(x, int(scheme.split(":")[1]), mul=True)
    if scheme.startswith("blockrows:"):
        return fake_blockrows(x, int(scheme.split(":")[1]))
    if scheme.startswith("mx:"):                 # "mx:<axis>[:<block>]"
        parts = scheme.split(":")
        return fake_mx(x, _AXES[parts[1]], int(parts[2]) if len(parts) > 2 else 32, preset.mx_rounding)
    if scheme.startswith("blockrows_pow2:"):     # one 2^ceil(log2(amax/448)) scale per R rows over all D (amax 0 -> 1)
        return fake_blockrows_pow2(x, int(scheme.split(":")[1]))
    if scheme.startswith("mx2d:"):               # one 2^ceil(log2(amax/448)) scale per R x C block of a (rows, cols) tensor
        r_, c_ = (int(t) for t in scheme.split(":")[1].split("x"))
        return fake_mx2d(x, r_, c_)
    if scheme.startswith("mx_pfwd:"):         # "mx_pfwd:<blk>[:<direct-code mode>]"
        parts = scheme.split(":")
        return mx_pfwd(x, ctx, int(parts[1]), parts[2] if len(parts) > 2 else "rn")
    if scheme.startswith("e4m3"):          # plain e4m3 with a fixed power-of-two multiplier: "e4m3*256"
        mul = float(scheme.split("*")[1]) if "*" in scheme else 1.0
        return e4m3_round(x * mul) / mul
    if scheme in SITE_FNS:
        return SITE_FNS[scheme](x, ctx, preset)
    raise ValueError(f"unknown quantization scheme {scheme!r}")


def kmean_sm90(k: torch.Tensor) -> torch.Tensor:
    """Per-head mean over the sequence in the producers' order: fp32 sums of 64-row chunks folded serially, / N.
    k (..., N, D) fp32 -> (..., 1, D).  (The within-chunk 64-row sum is a Triton tree; matches to fp32 round-off.)"""
    N = k.shape[-2]
    if N % 64:
        return k.mean(-2, keepdim=True)
    parts = k.reshape(*k.shape[:-2], N // 64, 64, k.shape[-1]).sum(-2)
    acc = torch.zeros_like(parts[..., 0, :])
    for j in range(parts.shape[-2]):
        acc = acc + parts[..., j, :]
    return (acc / N).unsqueeze(-2)


class _NoTF32:
    def __enter__(self):
        self.a = torch.backends.cuda.matmul.allow_tf32
        self.b = torch.backends.cudnn.allow_tf32
        self.c = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.a
        torch.backends.cudnn.allow_tf32 = self.b
        torch.set_float32_matmul_precision(self.c)


# ----------------------------------------------------------------------------------------------------------------- exact
def attention_fp32(q, k, v, do, causal=True, dtype=torch.float32):
    """Exact attention (no quantization) in fp32 or fp64 on the given inputs, gradients by autograd.  GQA by repeating
    K / V.  Returns dict O, LSE, dQ, dK, dV in the input layout (B, N, H, D)."""
    B, N, H, D = q.shape
    Hk = k.shape[2]
    with _NoTF32():
        qf = q.to(dtype).transpose(1, 2).detach().requires_grad_(True)
        kf = k.to(dtype).transpose(1, 2).detach().requires_grad_(True)
        vf = v.to(dtype).transpose(1, 2).detach().requires_grad_(True)
        kr = kf.repeat_interleave(H // Hk, dim=1)
        vr = vf.repeat_interleave(H // Hk, dim=1)
        s = (qf @ kr.transpose(-1, -2)) * (D ** -0.5)
        if causal:
            mask = torch.ones(N, N, device=q.device, dtype=torch.bool).tril()
            s = s.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(s, -1)
        o = torch.softmax(s, dim=-1) @ vr
        o.backward(do.to(dtype).transpose(1, 2))
    return {"O": o.detach().transpose(1, 2), "LSE": lse.detach(), "dQ": qf.grad.transpose(1, 2),
            "dK": kf.grad.transpose(1, 2), "dV": vf.grad.transpose(1, 2)}


def attention_exact_blocked(q, k, v, do, causal=True, q_block=1024):
    """Exact fp32 attention with O(q_block * N) memory per head (long sequences): the engine with no rounding anywhere
    and the matched delta (= rowsum(dO * O) exactly when nothing is rounded).  Same dict as attention_fp32."""
    r = dm_reference(q, k, v, do, causal, "matched", "fp32", q_block=q_block, round_o=False)
    return {n: r[n] for n in ("O", "LSE", "dQ", "dK", "dV")}


# ----------------------------------------------------------------------------------------------------------------- engine
def dm_reference(q, k, v, do, causal=True, mode="matched", preset="sm90", q_block=1024, aux=False,
                 o_for_stale=None, round_o=True, fwd_s_noise=0.0, bwd_p_noise=0.0, lse_override=None,
                 delta_override=None, k_mean=None):
    """Emulate an fp8 Delta-Matching kernel in torch.

    q (B, N, H, D), k / v (B, N, Hk, D), do (B, N, H, D); mode "matched" | "stale"; preset a PRESETS name or Preset.
    Returns dict O (bf16-rounded, as the kernel stores it), LSE (B, H, N) natural log, dQ, dK, dV (fp32, input layout),
    delta (B, H, N), and with aux=True: rowsum_dS (sum_j dS_ij before dS quantization), rowsum_dSq (after), colsum_dK
    (sum_j dK_j per head, (B, Hk, D)).  o_for_stale: the O the stale delta reads (default: this emulation's bf16 O; a
    test passes the kernel's own O to isolate the backward).
    Memory: O(q_block * N) per (batch, head); causal key ranges are truncated per block."""
    p = PRESETS[preset] if isinstance(preset, str) else preset
    if mode not in ("matched", "stale"):
        raise ValueError(mode)
    B, N, H, D = q.shape
    Hk = k.shape[2]
    rep = H // Hk
    sm = D ** -0.5
    dev = q.device
    f32 = torch.float32
    with _NoTF32():
        qh = q.to(f32).transpose(1, 2) * sm                  # (B, H, N, D), the softmax scale baked into Q
        kh = k.to(f32).transpose(1, 2)
        if p.center_k:
            # k_mean (B, Hk, D): a candidate's own fp32 K-centering mean.  The exact fp32 mean is an implementation detail
            # (summation order); at 128k a 7e-9 difference flipped 102 of 67M K codes by one e4m3 step,
            # so identity cross-checks run the emulation on the candidate's mean (checked separately to fp32 round-off)
            kh = kh - (k_mean.to(f32).reshape(B, Hk, 1, D) if k_mean is not None else kmean_sm90(kh))
        vh = v.to(f32).transpose(1, 2)
        doh = do.to(f32).transpose(1, 2)
        c0 = _Ctx(role="input")
        Qa = _site(p.q, qh, c0, p)                           # operand of S = Q K^T
        Ka = _site(p.k, kh, c0, p)
        _kb = int(p.k.split(":")[1]) if p.k.startswith("blockrows:") else 64      # K scale block (hd128 sm90: 64, hd256: 32)
        kscale = quant_blockrows(kh, _kb)[1].repeat_interleave(_kb, -1)[..., :N] if N % _kb == 0 else None   # (B, Hk, N)
        if p.k.startswith("mx:"):        # MX K carries no per-key scale: the dS' key weighting is 1
            kscale = torch.ones_like(kscale) if kscale is not None else None
        # per-token fp32 input scales of blockrows schemes (B, H|Hk, N), for residual-folding sites (None otherwise)
        def _tokscale(scheme, x):
            if not scheme.startswith(("blockrows:", "blockrows_mul:")):
                return None
            b_ = int(scheme.split(":")[1])
            return quant_blockrows(x, b_, scheme.startswith("blockrows_mul:"))[1].repeat_interleave(b_, -1) if N % b_ == 0 else None
        qscale, vscale, doscale = _tokscale(p.q, qh), _tokscale(p.v, vh), _tokscale(p.do, doh)
        Va = _site(p.v, vh, c0, p)                           # operand of O = P V
        Vd = _site(p.vd or (p.v if p.q2 is None else p.k), vh, c0, p)  # operand of dP = dO V^T (MX: D-blocked like K)
        dOa = _site(p.do, doh, c0, p)                        # operand of dP = dO V^T
        Qb = Qa if p.q2 is None else _site(p.q2, qh, c0, p)  # operand of dK = dS^T Q
        Kb = Ka if p.k2 is None else _site(p.k2, kh, c0, p)  # operand of dQ = dS K
        dOb = dOa if p.do2 is None else _site(p.do2, doh, c0, p)   # operand of dV = P^T dO

        O = torch.empty(B, H, N, D, device=dev, dtype=f32)
        LSE = torch.empty(B, H, N, device=dev, dtype=f32)
        dQ = torch.zeros(B, H, N, D, device=dev, dtype=f32)
        dK = torch.zeros(B, Hk, N, D, device=dev, dtype=f32)
        dV = torch.zeros(B, Hk, N, D, device=dev, dtype=f32)
        delta = torch.empty(B, H, N, device=dev, dtype=f32)
        rs = torch.empty(B, H, N, device=dev, dtype=f32) if aux else None
        rsq = torch.empty(B, H, N, device=dev, dtype=f32) if aux else None
        pdp = torch.empty(B, H, N, device=dev, dtype=torch.float64) if aux else None     # sum_j P dP
        psum = torch.empty(B, H, N, device=dev, dtype=torch.float64) if aux else None    # sum_j P
        pdpa = torch.empty(B, H, N, device=dev, dtype=torch.float64) if aux else None    # sum_j |P dP|
        pdpt = torch.empty(B, H, N, device=dev, dtype=torch.float64) if aux else None    # sum_j P (|dO| . |V_j|): dP term size
        qb = min(q_block, N)
        assert qb % 128 == 0 or qb == N
        kidx = torch.arange(N, device=dev)
        # forward first (the stale delta needs the stored bf16 O)
        for i0 in range(0, N, qb):
            i1 = min(i0 + qb, N)
            nk = i1 if causal else N
            S = Qa[:, :, i0:i1] @ Ka[:, :, :nk].repeat_interleave(rep, 1).transpose(-1, -2)
            if fwd_s_noise:      # diagnostic: forward-only relative perturbation of S (fwd / bwd recompute mismatch)
                S = S * (1 + fwd_s_noise * torch.randn_like(S))
            if causal:
                S = S.masked_fill(kidx[None, :nk] > kidx[i0:i1, None], float("-inf"))
            m = S.amax(-1, keepdim=True)
            P = torch.exp(S - m)
            l = P.sum(-1, keepdim=True)
            ctx = _Ctx(role="p_fwd", rowmax=m, rows=(i0, i1), causal=causal, S=S,
                       vscale=None if vscale is None else vscale[:, :, :nk].repeat_interleave(rep, 1))
            Pq = _site(p.p_fwd, P, ctx, p)
            if p.l_from_codes:
                l = Pq.sum(-1, keepdim=True)
            O[:, :, i0:i1] = (Pq @ Va[:, :, :nk].repeat_interleave(rep, 1)) / l
            LSE[:, :, i0:i1] = (m + torch.log(l)).squeeze(-1)
        if lse_override is not None:   # the backward recomputes P from this LSE (e.g. a candidate's own) instead
            LSE = lse_override.to(f32).reshape(B, H, N).clone()
        Ob = O.to(torch.bfloat16)
        Ost = (o_for_stale.to(f32).transpose(1, 2) if o_for_stale is not None else Ob.to(f32))
        for i0 in range(0, N, qb):
            i1 = min(i0 + qb, N)
            nk = i1 if causal else N
            Kr = Ka[:, :, :nk].repeat_interleave(rep, 1)
            S = Qa[:, :, i0:i1] @ Kr.transpose(-1, -2)
            if causal:
                S = S.masked_fill(kidx[None, :nk] > kidx[i0:i1, None], float("-inf"))
            P = torch.exp(S - LSE[:, :, i0:i1, None])
            if bwd_p_noise:      # diagnostic: elementwise P_bwd * (1 + U(-a, a)) (a cheaper backward exp)
                P = P * (1 + bwd_p_noise * (2 * torch.rand_like(P) - 1))
            ctx = _Ctx(role="p_bwd", lse=LSE[:, :, i0:i1, None], rows=(i0, i1), causal=causal, S=S,
                       doscale=None if doscale is None else doscale[:, :, i0:i1])
            Pq = _site(p.p_bwd, P, ctx, p)
            dVp = Pq.transpose(-1, -2) @ dOb[:, :, i0:i1]                         # (B, H, nk, D)
            dV[:, :, :nk] += dVp.reshape(B, Hk, rep, nk, D).sum(2)
            dP = dOa[:, :, i0:i1] @ Vd[:, :, :nk].repeat_interleave(rep, 1).transpose(-1, -2)
            if aux:
                pdp[:, :, i0:i1] = (P.double() * dP.double()).sum(-1)
                psum[:, :, i0:i1] = P.double().sum(-1)
                pdpa[:, :, i0:i1] = (P.double() * dP.double()).abs().sum(-1)
                pdpt[:, :, i0:i1] = (P.double() * (dOa[:, :, i0:i1].abs() @ Vd[:, :, :nk].abs().repeat_interleave(rep, 1)
                                                   .transpose(-1, -2)).double()).sum(-1)
            if delta_override is not None:
                dl = delta_override.to(f32).reshape(B, H, N)[:, :, i0:i1, None]
            elif mode == "stale":
                dl = (doh[:, :, i0:i1] * Ost[:, :, i0:i1]).sum(-1, keepdim=True)
            elif p.delta_from_o:
                dl = (dOa[:, :, i0:i1].double() * O[:, :, i0:i1].double()).sum(-1, keepdim=True).float()
            else:
                dl = _matched_delta(P, Pq, dP, p)
            delta[:, :, i0:i1] = dl.squeeze(-1)
            dS = _ds_operand(P, Pq, p) * (dP - dl)
            ctx = _Ctx(role="ds", rows=(i0, i1), causal=causal,
                       kscale=None if kscale is None else kscale[:, :, :nk].repeat_interleave(rep, 1).unsqueeze(-2).squeeze(-2),
                       qscale=None if qscale is None else qscale[:, :, i0:i1])
            dSq = _site(p.ds, dS, ctx, p)
            dSq2 = dSq if p.ds2 is None else _site(p.ds2, dS, ctx, p)
            if aux:
                rs[:, :, i0:i1] = dS.sum(-1)
                rsq[:, :, i0:i1] = dSq.sum(-1)
            dKp = dSq.transpose(-1, -2) @ Qb[:, :, i0:i1]
            dK[:, :, :nk] += dKp.reshape(B, Hk, rep, nk, D).sum(2)
            dQ[:, :, i0:i1] = (dSq2 @ Kb[:, :, :nk].repeat_interleave(rep, 1)) * sm
    out = {"O": (Ob if round_o else O).transpose(1, 2), "LSE": LSE, "dQ": dQ.transpose(1, 2), "dK": dK.transpose(1, 2),
           "dV": dV.transpose(1, 2), "delta": delta}
    if aux:
        out.update(rowsum_dS=rs, rowsum_dSq=rsq, colsum_dK=dK.sum(2), sum_PdP=pdp, sum_P=psum, sum_absPdP=pdpa, sum_PdPterms=pdpt)
    return out


def _matched_delta(P, Pq, dP, p: Preset):
    """The matched delta: rowsum of the backward's own (P, dP) pair, P the unquantized fp32 P of the dS line (sm90
    pass 0 sums exactly the operands of pass 1's dS line), accumulated in fp64 and rounded once."""
    fn = SITE_FNS.get(p.name + ":matched_delta")
    if fn is not None:
        return fn(P, Pq, dP, p)
    num = (P.double() * dP.double()).sum(-1, keepdim=True)
    if p.delta_norm:
        num = num / P.double().sum(-1, keepdim=True)
    return num.float()


def _ds_operand(P, Pq, p: Preset):
    """The P that multiplies (dP - delta) in dS: the unquantized fp32 P (the sm90 kernel's pass 1 dS line, and the pass-0
    delta sums the same P, which is what makes rowsum(dS) = delta * (1 - rowsum P) vanish at fp32 round-off)."""
    fn = SITE_FNS.get(p.name + ":ds_p")
    return fn(P, Pq, p) if fn is not None else P


# ------------------------------------------------------------------------------------------------- sm90 site schemes
def _rowblock_absmax(x, blk=32):
    """e4m3 with one fp32 scale absmax / 448 per row per `blk` keys (columns)."""
    *lead, R_, C = x.shape
    xb = x.reshape(*lead, R_, C // blk, blk)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(TINY) / E4M3_MAX
    return (e4m3_round(xb / s) * s).reshape(x.shape)


LN2 = 0.6931471805599453


def _pad_to(x, rows, cols, value=0.0):
    R_, C = x.shape[-2:]
    pr, pc = (-R_) % rows, (-C) % cols
    if pr or pc:
        x = torch.nn.functional.pad(x, (0, pc, 0, pr), value=value)
    return x


def sm90_pfwd(P, ctx, p, tile=256, group=64):
    """The forward's P codes (NUMERICS-hd128.md section 2): keys in 256-key tiles visited in DESCENDING order with an online
    max m_t (the max over tiles >= t); per (row, 64-key group g) the basis nu_g = max(groupmax_g S, m_t - 12 ln2) and
    code = e4m3(448 * exp(S - nu_g)).  P here is exp(S - m) with m the final row max; returns the dequantized P-hat in the
    same units.  (The row sum l is taken from the unquantized P, which the engine already does.)"""
    S = ctx.S
    R_, C = S.shape[-2:]
    Sp = _pad_to(S, 1, tile, float("-inf"))
    T = Sp.shape[-1] // tile
    g = Sp.reshape(*Sp.shape[:-1], T, tile // group, group)
    gm = g.amax(-1)                                                    # (..., R, T, 4)
    tm = gm.amax(-1)                                                   # (..., R, T)
    mt = tm.flip(-1).cummax(-1).values.flip(-1)                        # running max when tile t is processed
    nu = torch.maximum(gm, mt.unsqueeze(-1) - 12 * LN2)                # (..., R, T, 4)
    m = ctx.rowmax.unsqueeze(-1).unsqueeze(-1)                         # (..., R, 1, 1, 1)
    nu = torch.where(torch.isfinite(nu), nu, m.squeeze(-1))            # tiles with no live key (padding, causal tail)
    pf = torch.exp(g - nu.unsqueeze(-1)) * E4M3_MAX
    ph = e4m3_round(pf) / E4M3_MAX * torch.exp(nu.unsqueeze(-1) - m)
    return ph.reshape(Sp.shape)[..., :C]


def sm90_ds(dS, ctx, p, tr=64, tc=128):
    """dS codes of pass 1 (NUMERICS-hd128.md section 3): dS' = 2^8 * ks_j * dS (ks: the 64-key K scale of key j) is
    quantized with one scale psi = max(amax / 448, 1e-30) per 64-row x 128-key tile; dS-hat = e4m3(dS' / psi) * psi /
    (2^8 ks_j).  (The F2b dominance floor, which coarsens tiles 1e-8 below the key block's largest, is not modelled.)"""
    ks = ctx.kscale                                                    # (..., C) per key
    X = dS * ks.unsqueeze(-2)
    R_, C = X.shape[-2:]
    Xp = _pad_to(X, tr, tc)
    Rp, Cp = Xp.shape[-2:]
    t = Xp.reshape(*Xp.shape[:-2], Rp // tr, tr, Cp // tc, tc)
    psi = t.abs().amax(dim=(-3, -1), keepdim=True) * (1.0 / E4M3_MAX)
    zero = psi < TINY
    psi = psi.clamp_min(TINY)
    q = torch.where(zero, torch.zeros_like(t), e4m3_round(t / psi) * psi)
    return q.reshape(Xp.shape)[..., :R_, :C] / ks.unsqueeze(-2)


SITE_FNS["sm90_pfwd"] = sm90_pfwd
SITE_FNS["sm90_pbwd"] = lambda P, ctx, p: e4m3_round(P.clamp(max=16.0) * 256.0) / 256.0     # P' = 2^8 P, fixed scale
SITE_FNS["sm90_ds"] = sm90_ds
# sm90 hd256: P-hat per (row, 32-key group) under a descending 128-key-tile online max with the same
# 12-binade floor (s2.3); dS' psi per 64 x 64 tile (s4.2); backward P' = 2^8 P (s3, cap 2^12) as hd128
SITE_FNS["sm90d256_pfwd"] = lambda P, ctx, p: sm90_pfwd(P, ctx, p, tile=128, group=32)
SITE_FNS["sm90d256_ds"] = lambda dS, ctx, p: sm90_ds(dS, ctx, p, tr=64, tc=64)


# the sm100-mx rules: mx-v1 with dS scaled per (128-query tile x 32 keys)
make_mx_preset("sm100-mx", ds_block=(128, 32), delta_norm=True)
# hd256 (no accuracy trade-off: the [64,32] contract is the floor): sm100-mx's P / dS sites with the [64,32]
# fp32 input scales (Q / dO 64 rows, K / V 32 keys per scale, s = amax / 448), i.e. the hd256 design. Accepted at hd256 only
# where measured at least as accurate as sm90-d256 on every metric.
_d256 = make_mx_preset("sm100-d256", ds_block=(128, 32), delta_norm=True, register=False)
_register(replace(_d256, q="blockrows:64", k="blockrows:32", v="blockrows:32", do="blockrows:64",
                  notes="hd256: [64,32] fp32 input scales + sm100-mx P (row x 32 keys exp) / P^T / dS 128x32 / delta_norm"))
make_mx_preset("sm100-mx-nonorm", ds_block=(128, 32), delta_norm=False)
make_mx_preset("sm100-mx-lq", ds_block=(128, 32), delta_norm=True, l_from_codes=True)
# forward with ~exact P (fp8 code + 2 bf16 residuals ~ fp32), matched delta = dequant(dO8) . O_fp32
make_mx_preset("sm100-mx-x", ds_block=(128, 32), p_fwd="none", delta_from_o=True)
# variant: dS scaled per 32 keys x 32 queries (MXFP8 granularity), everything else sm100-mx
make_mx_preset("sm100-mx-ds32", ds_block=(32, 32), delta_norm=True)
# MXFP8 is THE sm100 spec at both head
# dims.  hd256 spec = the sm100-mx scheme at D=256 (pow2 E8M0 scale per 32 tokens over all D, P per row x 32 keys, P^T per key x 32
# queries, dS 128x32, num/den delta); the exact sm90 [64,32] fp32-scale floor (sm90-d256) is retired as a gate (kept as info).
make_mx_preset("sm100-mx-d256", ds_block=(128, 32), delta_norm=True)
make_mx_preset("sm100-mx-d256-ds32", ds_block=(32, 32), delta_norm=True)


# ------------------------------------------------------------------------------------------ hd256 residual-folding routes
# on sm100 the block-scaled MMA takes only pow2 (UE8M0) scales along the contraction dim.  Split every fp32
# input scale s = 2^e * r (r in [1, 2)): 2^e goes to the MMA scale; r either multiplies post-MMA (scale on a non-contraction
# dim: exact) or is folded into the OTHER operand before its e4m3 rounding (scale on the contraction dim).  Input codes stay
# the floor's; only P (fwd), P^T (dV) and dS codes change.  Emulation of the floor-vs-fold question only (not accepted yet).
def _resid(s):
    m_, _ = torch.frexp(s)
    return 2 * m_                                                       # s / 2^floor(log2 s), exact


def fold_v_pfwd(P, ctx, p, kind="mx"):
    """(1) V's per-32-key residual r_j folded into the P-hat basis: codes of P_ij * r_j with a pow2 scale per (row, 32 keys),
    V codes = the floor's V codes with MMA scale 2^e_j.  kind "mx": the sm100 direct-exponent rule (mx_pfwd on S + ln r);
    "mx": ideal pow2 ceil per (row, 32 keys)."""
    r = _resid(ctx.vscale).unsqueeze(-2)                                # (..., 1, C)
    if kind == "mx":
        c2 = _Ctx(**{**ctx.__dict__, "S": ctx.S + torch.log(r)})
        return mx_pfwd(P, c2, 32) / r
    return fake_mx(_pad_to(P * r, 1, 32), -1, 32)[..., :P.shape[-1]] / r


def fold_do_pbwd(P, ctx, p):
    """(2) dO's per-64-row residual r_i folded into P^T for dV = P^T dO (P^T pow2 per key x 32 queries, the accepted ptmx
    swap); dO codes = the floor's with MMA scale 2^e_i.  The dP = dO V^T side needs no fold (both scales are on
    non-contraction dims: exact post-MMA multiply)."""
    r = _resid(ctx.doscale).unsqueeze(-1)                               # (..., R, 1)
    x = _pad_to(P * r, 32, 1)
    return fake_mx(x, -2, 32)[..., :P.shape[-2], :] / r


def fold_qk_ds(dS, ctx, p, tr=32, tc=32, fold_q=True):
    """(3) dS for dK = dS^T Q (contracts queries: Q's residual) and dQ = dS K (contracts keys: K's residual): one code set of
    dS * ks_j * sq_i (full scales folded, the floor already folds ks_j) with a pow2 scale per tr x tc block; undone exactly
    post-MMA on the non-contraction side.  fold_q=False: info only (pow2 psi without the Q fold - not realisable for dK)."""
    w = ctx.kscale.unsqueeze(-2) * (ctx.qscale.unsqueeze(-1) if fold_q else 1.0)
    return fake_mx2d(dS * w, tr, tc) / w


SITE_FNS["fold_v_mx"] = lambda P, ctx, p: fold_v_pfwd(P, ctx, p, "mx")
SITE_FNS["fold_v_mx"] = lambda P, ctx, p: fold_v_pfwd(P, ctx, p, "mx")
SITE_FNS["fold_do_ptmx"] = fold_do_pbwd
for _r, _c in ((32, 32), (64, 64), (128, 32), (64, 32)):
    SITE_FNS[f"fold_qk_ds:{_r}x{_c}"] = (lambda r_, c_: lambda dS, ctx, p: fold_qk_ds(dS, ctx, p, r_, c_))(_r, _c)
    SITE_FNS[f"pow2_ds_noq:{_r}x{_c}"] = (lambda r_, c_: lambda dS, ctx, p: fold_qk_ds(dS, ctx, p, r_, c_, False))(_r, _c)
