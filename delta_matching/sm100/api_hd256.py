"""hd256 route of dm_attn_func (api.py branches here on q.shape[-1] == 256 before any hd128 code).
Numerics: sm100-mx at D=256.
forward : quant.quant_qkv2(q, k, v, 256**-0.5) (the D-generic quant.py producers) -> fwd_hd256.dm_fwd_hd256_mx
backward: dO -> quant.quant_do_delta(dO, None) -> delta (matched: pass-0 kernel pass0_hd256.dm_delta_hd256, num/den; stale: quant.quant_do_delta(dO, O))
          -> attention_bwd_hd256_sm100.attention_bwd_hd256: fused paired-CTA dK/dV kernel KV2 writing the dS codes to a workspace
          bounded by DM_BWD_WS_LIMIT_GB (default 8 GiB), then the dQ GEMM kernel Dqg; above the cap KV2 + the DmHd256Mx2 dq kernel
          (bitwise the same values).
q (B, N, H, 256) bf16, k / v (B, N, Hk, 256) bf16 -> o (B, N, H, 256) bf16. N % 128 == 0, softmax scale 256^-0.5 (= 2^-4)."""
import torch

from .fwd import pack_kv_scales  # noqa: F401
from .fwd_hd256 import dm_fwd_hd256_mx
from .quant import quant_qkv2, quant_do_delta
from .pass0_hd256 import dm_delta_hd256
from .bwd_hd256mx import dm_bwd_hd256mx
from .bwd_hd256kv2 import kv2_launch
from .attention_bwd_hd256_sm100 import attention_bwd_hd256

D = 256
SM = D ** -0.5     # softmax scale baked into Q by quant_qkv2 (= 2^-4, exact)


def dm_fwd_hd256_from_bf16(q, k, v, causal):
    """(o, lse, saved) for bf16 inputs: the exact forward dm_attn_func runs at hd256."""
    q8, qs, k8, ks, v8, vs, kvw, mean = quant_qkv2(q, k, v, SM)   # kvw == pack_kv_scales(ks, vs)
    o, lse = dm_fwd_hd256_mx(q8, qs, k8, kvw, v8, causal)
    return o, lse, (q8, qs, k8, ks, v8, vs, mean)


def _delta_hd256(q8, qs, k8, ks, v8, vs, lse, o, do, mode, causal):
    """(do8, dos, delta) exactly as the backward consumes them. matched: num/den pass 0 (bitwise the bwd's P/dP)."""
    if mode == "stale":
        return quant_do_delta(do, o)
    do8, dos, _ = quant_do_delta(do, None)
    return do8, dos, dm_delta_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, causal)


class _DMAttnHd256Sm100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, mode, det):
        o, lse, (q8, qs, k8, ks, v8, vs, _) = dm_fwd_hd256_from_bf16(q, k, v, causal)
        ctx.save_for_backward(q8, qs, k8, ks, v8, vs, o, lse)
        ctx.causal, ctx.mode, ctx.det = causal, mode, det
        return o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do):
        q8, qs, k8, ks, v8, vs, o, lse = ctx.saved_tensors
        do8, dos, delta = _delta_hd256(q8, qs, k8, ks, v8, vs, lse, o, do.contiguous(), ctx.mode, ctx.causal)
        dq, dk, dv = attention_bwd_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, ctx.causal)
        return dq, dk, dv, None, None, None


def dm_attn_func_hd256(q, k, v, causal=True, mode="matched", deterministic=None):
    assert q.shape[-1] == D and q.dtype == torch.bfloat16 and q.shape[1] % 128 == 0
    assert mode in ("matched", "stale")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    return _DMAttnHd256Sm100.apply(q, k, v, causal, mode, True)


@torch.no_grad()
def dm_attn_debug_hd256(q, k, v, do, causal=True, mode="matched", identity=None, deterministic=None, lockstep=None):
    """Validator hook at hd256 (same contract as api.dm_attn_debug): delta + LSE the backward consumes; with identity
    (matched mode) also the backward's IN-KERNEL pre-quant rowsum(dS) and sum|P dP|.  These come from the SHIPPED
    dK/dV kernel KV2 (dbg bit 16 build, same arithmetic: the fp32 dS KV2 quantizes into the codes that dK AND dQ consume, fp64
    sums per row), run through the same chunk plan as the production call at the current cap / carry policy."""
    import os
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    o, lse, (q8, qs, k8, ks, v8, vs, _) = dm_fwd_hd256_from_bf16(q, k, v, causal)
    do8, dos, delta = _delta_hd256(q8, qs, k8, ks, v8, vs, lse, o, do.contiguous(), mode, causal)
    out = {"delta": delta.float().contiguous(), "lse": lse.float().contiguous()}
    if identity is None:
        identity = os.environ.get("DM_DEBUG_IDENTITY", "0") == "1"
    if identity and mode == "matched":
        B, N, H, _ = q8.shape
        idt = attention_bwd_hd256(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, identity=True)[3].view(torch.float64).reshape(B, H, N, 4)
        out["rowsum_dS"] = idt[..., 0].float().contiguous()
        out["sum_abs_PdP"] = idt[..., 1].float().contiguous()
    if lockstep is None:
        lockstep = os.environ.get("DM_DEBUG_LOCKSTEP", "0") == "1"
    if lockstep:
        # dbg 8 build (compiled out by default): per query row, from EACH kernel, order-free checksums
        #   [..., 0] P fp32 bits (dv, dq kernels), [..., 1] pre-quant dS fp32 bits (dk, dq), [..., 2] dS e4m3 code + E8M0 SF
        #   (dk, dq): u64 wrapping sums of a mixed u32 hash of (q, k, value); [..., 3] fp64 rowsum of pre-quant dS (dk:
        #   atomic adds -> compare to round-off only; the hashes are the bitwise gate).
        _, _, _, aux = dm_bwd_hd256mx(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=8)
        B, N, H, _ = q8.shape
        for m in ("dq", "dk", "dv"):
            hsh = aux[m]["hash"].view(B, H, N, 4)
            out[f"lockstep_{m}_kernel"] = dict(P_hash=hsh[..., 0].contiguous(), dS_hash=hsh[..., 1].contiguous(),
                                                code_sf_hash=hsh[..., 2].contiguous(),
                                                rowsum_dS=hsh[..., 3].view(torch.float64).contiguous())
        # KV2 (fused dK/dV kernel): P / dS / code|SF hashes from the one KV2 kernel (keys on rows)
        hk = kv2_launch(q8, qs, k8, ks, v8, vs, do8, dos, lse, delta, causal, dbg=8)[2]["hash"].view(B, H, N, 4)
        out["lockstep_kv2_kernel"] = dict(P_hash=hk[..., 0].contiguous(), dS_hash=hk[..., 1].contiguous(),
                                          code_sf_hash=hk[..., 2].contiguous(),
                                          rowsum_dS=hk[..., 3].view(torch.float64).contiguous())
    return out
