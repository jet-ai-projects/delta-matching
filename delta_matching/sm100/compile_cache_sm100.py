"""
Compile-cache keys of the sm100 kernels: every compile-time input outside FA4's
own key (compile_env) and the persistent-cache fingerprint (FA4 sources, this
package's sources, the opts string and compile_env).
"""
import hashlib
from pathlib import Path

from flash_attn.cute import cache_utils

from delta_matching.sm100.launch_cache_sm100 import env
from delta_matching.sm100.tcgen05_fence_sm100 import TCF_IMPL, TCF_MASK

__all__ = ["PACKAGE_DIR", "STATIC_ENV", "compile_env", "fingerprint"]

# every source file of this package is hashed into the fingerprint, so a moved helper can never be served from a stale entry
PACKAGE_DIR = Path(__file__).resolve().parent


STATIC_ENV = []


def compile_env():
    """
    Every compile-time input of the sm100 kernels that neither FA4's compile key (dtypes, head dims,
    causal, GQA ratio, config incl. registers, deterministic, arch) nor the callers' opts strings
    carry, as one string hashed into the persistent-cache fingerprint and the in-process memos:
    DM_ABL, DM_TCF_SKIP, the fence emission settings (DM_TCF_IMPL, DM_TCF_MASK), the pass-0 register
    splits and the CuTe DSL compile settings (its own cache_key_str() without arch, which FA4's key
    has, and without the runtime library path).
    """
    from delta_matching.sm100.pass0 import REGS_C3, REGS_TL
    if not STATIC_ENV:
        try:
            from cutlass.cutlass_dsl import CuTeDSL
            dsl = ";".join(x for x in CuTeDSL._get_dsl().envar.cache_key_str().split(";")
                           if x and not x.startswith(("arch=", "shared_libs=")))
        except Exception as e:  # noqa: BLE001
            dsl = f"unknown:{type(e).__name__}"
        # NOT hashed: the SM count (AOT fakes 148, GB200 reports 152): measured not to change the compiled object (whole .o
        # byte-identical for fwd / pass 0 / bwd), and hashing it would make every GPU run miss
        # the AOT entries (compile inside a lease).
        STATIC_ENV.append(f"dsl={dsl}|tcf_impl={TCF_IMPL}|tcf_mask={TCF_MASK}|"
                           f"p0regs={sorted(REGS_C3.items())}{sorted(REGS_TL.items())}")
    return f"{STATIC_ENV[0]}|abl={env('DM_ABL', '')}|tcf_skip={env('DM_TCF_SKIP', '')}"


def fingerprint(opts=""):
    h = hashlib.sha256(cache_utils._compute_source_fingerprint().encode())
    h.update(opts.encode())   # kernel variant knobs (DM_FWD_OPTS, read in DMForwardSm100.__init__)
    h.update(compile_env().encode())   # compile-time inputs outside FA4's key and the opts
    for f in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:24]
