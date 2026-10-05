"""The build configuration shared by setup.py and the JIT build: sources, include dirs, defines, nvcc flags.
The defines are the certified flag set of the shipped binary (Delta-Matching-128 v3 for head_dim 128, whose head_dim-256 paths
are bitwise the Delta-Matching-256 binary): they are part of the kernel's numerical identity, not tuning knobs.  The csrc/
sources were reduced to this flag set, and csrc/dm_flagset.h stops a build with any other value."""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
try:   # the ninja pip package ships its binary outside PATH; torch's builder only parallelizes across sources when it finds it
    import ninja as _ninja
    if _ninja.BIN_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass
ROOT = os.path.dirname(PACKAGE_DIR)
CSRC = os.path.join(ROOT, "csrc")


def fa_src():
    """The flash-attention checkout the kernels compile against (hopper/ headers + csrc/cutlass): DM_FA_SRC or the
    third_party submodule.  Pinned: flash-attention 0251105a2fb19d2957484b7f023cd8c115286ced, cutlass
    7127592069c2fe01b041e174ba4345ef9b279671."""
    d = os.environ.get("DM_FA_SRC", os.path.join(ROOT, "third_party", "flash-attention"))
    for sub in ("hopper", os.path.join("csrc", "cutlass", "include")):
        if not os.path.isdir(os.path.join(d, sub)):
            raise RuntimeError(f"flash-attention checkout missing {sub} under {d}: run `git submodule update --init "
                               "third_party/flash-attention && git -C third_party/flash-attention submodule update --init "
                               "csrc/cutlass` or set DM_FA_SRC")
    return d


DBG = os.environ.get("DM_DBG", "0") == "1"
BF16_DP = os.environ.get("DM_BF16_DP", "0") == "1"

# the certified flag set (the build of the certified binaries)
FLAGS = {
    "FA3FP11_M16_XCHG": 0, "FA3FP11_M16_FWDSCHED": 0, "FA3FP11_M16_P2V2": 0, "FA3FP11_M16_P0V2": 0, "FA3FP11_M16_DO2": 0,
    "FA3FP11_M16_ER": 0, "FA3FP11_M16_Q3": 0, "FA3FP11_M16_TWINPERM": 0, "FA3FP11_M16_P0V3": 1, "FA3FP11_M16_P2V3": 0,
    "FA3FP11_M17_GT": 1, "FA3FP11_M17_DOTST": 1, "FA3FP11_M17_P0V4": 1, "FA3FP11_M17_LSEK": 1, "FA3FP11_M17_P0MS": 32,
    "FA3FP11_M17_FASKIP": 0, "FA3FP11_M17_FWDIL2": 0, "FA3FP11_M17_FWDG4": 0,
    "FA3FP11_HD128": 1, "FA3FP11_HD128_PVFIRST": 1, "FA3FP11_HD128_BM128": 0, "FA3FP11_HD128_P0NBWG": 2,
    "FA3FP11_HD128_DOTST": 1, "FA3FP11_HD128_BN256": 1, "FA3FP11_HD128_QST": 2, "FA3FP11_HD128_DOST": 1,
    "FA3FP11_HD128_P2ST": 2, "FA3FP11_HD128_P0ST": 3, "FA3FP11_HD128_P1R": 1, "FA3FP11_HD128_P0V5": 1,
    "FA3FP11_P1R_SAGEDP": 1 if BF16_DP else 0,
}
EXTRA_DEFINES = ["-DFA3FP11_M16_P0V2_ATOMN=64", "-DFA3FP11_HD128_P0V6=1", "-DFA3FP11_P1R_MASKDIAG=1", "-DFA3FP11_P1R_RFSKIP=1"]


def defines():
    d = ["-DFLASHATTENTION_DISABLE_SM8x", "-DFLASHATTENTION_DISABLE_SPLIT", "-DFLASHATTENTION_DISABLE_PAGEDKV",
         "-DFLASHATTENTION_DISABLE_APPENDKV", "-DFLASHATTENTION_DISABLE_LOCAL", "-DFLASHATTENTION_DISABLE_SOFTCAP",
         "-DFLASHATTENTION_DISABLE_PACKGQA", "-DFLASHATTENTION_DISABLE_FP16", "-DFLASHATTENTION_DISABLE_VARLEN",
         "-DFLASHATTENTION_DISABLE_CLUSTER", "-DFLASHATTENTION_DISABLE_HDIM64", "-DFLASHATTENTION_DISABLE_HDIM96",
         "-DFLASHATTENTION_DISABLE_HDIM128", "-DFLASHATTENTION_DISABLE_HDIM192", "-DFLASHATTENTION_DISABLE_HDIMDIFF64",
         "-DFLASHATTENTION_DISABLE_HDIMDIFF192",
         # the forward's certified pins
         "-DFA3FP8F_PHAT_FLOOR_B=12", "-DFA3FP8F_TP_GPP=4", "-DFA3FP8F_TP_NSPLIT=2", "-DFA3FP8F_TP_PVFIRST=1"]
    if DBG:
        d.append("-DFA3FP11_DBG")
    d += [f"-D{k}={v}" for k, v in FLAGS.items()]
    d += EXTRA_DEFINES
    return d


def sources():
    s = ["delta_matching_api.cpp", "flash_fwd_hdim256_e4m3_fp8f_sm90.cu", "flash_bwd_hdim256_e4m3_fp8v10_sm90.cu",
         "flash_delta_hdim256_e4m3_fp11_sm90.cu", "fp11_twin_kernel.cu",
         "flash_fwd_hdim128_e4m3_fp8f_sm90.cu", "flash_bwd_hdim128_e4m3_fp8v10_sm90.cu",
         "flash_delta_hdim128_e4m3_fp11v5_sm90.cu"]
    return [os.path.join(CSRC, f) for f in s]


def include_dirs():
    fa = fa_src()
    return [CSRC, os.path.join(fa, "hopper"), os.path.join(fa, "csrc", "cutlass", "include")]


def cuda_flags():
    return ["-O3", "-std=c++17", "--use_fast_math", "-lineinfo", "-DNDEBUG", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DCUTLASS_ENABLE_GDC_FOR_SM90",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=0", "--threads", "4", "-gencode", "arch=compute_90a,code=sm_90a"]


def csrc_sha():
    """A content hash of the kernel sources (file names + bytes, sorted)."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(os.listdir(CSRC)):
        if f.endswith((".cpp", ".cu", ".h", ".hpp")):
            h.update(f.encode()); h.update(open(os.path.join(CSRC, f), "rb").read())
    return h.hexdigest()[:12]


def write_build_info():
    """Stamp this build into the package (read by delta_matching.kernel_version()); rewritten by every build."""
    info = {"code_sha": csrc_sha(), "dbg": DBG, "bf16_dp": BF16_DP, "flags": FLAGS, "extra_defines": EXTRA_DEFINES}
    with open(os.path.join(PACKAGE_DIR, "_build_info.py"), "w") as f:
        f.write("# written by the build (setup.py / delta_matching.jit); not hand-edited\n")
        f.write(f"BUILD_INFO = {info!r}\n")
