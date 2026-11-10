"""
tcgen05 thread-sync fences on the sm100 kernels' cross-thread hand-offs, and
the build-trap guard.
"""
import os
import sys
import types

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from delta_matching.sm100.cute_ptx_sm100 import mbar_wait_parity, mbar_wait_parity_nh, mbar_wait_parity_cluster, mbar_arrive, mbar_arrive_local, mbar_arrive_rank, mbar_arrive_expect_tx

__all__ = ["tcgen05_fence_after", "tcgen05_fence_before", "TCF_IMPL", "fence_state_module", "FENCE_STATE", "FENCE_MODE", "FENCE_STATS", "TCF_MASK", "FenceBuildTrap", "begin_kernel_fences", "save_fence_count", "restore_fence_count", "end_kernel_fences", "raw_method", "wrap_fences", "install_fences", "MARKED_CLASSES", "mark_fenced", "wrap_marked_method", "FENCE_ROLES", "fence_classes", "install_marked_fences", "verify_fences", "TCF_SKIP", "FenceSkipSite", "fence_warp_relay", "RAW_SYNC_WARP", "mbar_wait_parity", "mbar_wait_parity_nh", "mbar_wait_parity_cluster", "mbar_arrive", "mbar_arrive_local", "mbar_arrive_rank", "mbar_arrive_expect_tx"]


TCF_IMPL = os.environ.get("DM_TCF_IMPL", "asm")   # asm | nvvm (tcgen05 fence emission)


@dsl_user_op
def tcgen05_fence_after(*, loc=None, ip=None):
    """
    tcgen05.fence::after_thread_sync: order later tcgen05 ops after a preceding thread
    sync (mbarrier wait).
    """
    if TCF_IMPL == "nvvm":
        from cutlass._mlir.dialects import nvvm
        nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)
        return
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tcgen05_fence_before(*, loc=None, ip=None):
    """
    tcgen05.fence::before_thread_sync: order earlier tcgen05 ops before a following thread
    sync (mbarrier arrive).
    """
    if TCF_IMPL == "nvvm":
        from cutlass._mlir.dialects import nvvm
        nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)
        return
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


# PTX orders one thread's tcgen05 ops (ld / st / mma / cp) against another thread's only through tcgen05.fence::before_thread_sync
# (producer, before the sync) and tcgen05.fence::after_thread_sync (consumer, after it); the FA4-derived pipelines emit no such
# fences. While a kernel body is traced with fences on, every marked pipeline / barrier wait gets fence::after and every arrive
# gets fence::before (on sm_100a both emit no SASS: they only forbid ptxas from moving tcgen05 ops across the sync).
# The wrappers are installed on SHARED classes (cutlass.pipeline, flash_attn.cute.pipeline, cutlass.cute.arch) that survive a
# purge and re-import of delta_matching, so the fence state lives in one process-wide module outside the delta_matching
# namespace (sys.modules["_dm_tcf_state"]); every import unwraps any installed wrapper (this tree's or an older tree's) to the raw
# method and re-wraps it, and end_kernel_fences raises FenceBuildTrap when a kernel marked tcgen05 pipelines but no wrapper fence
# came out. The state module name, its attributes and the _dm_tcf / _dm_tcfm / _dm_tcf_state marks on wrappers are a protocol
# shared with older trees and the test harnesses, so they keep their spelling.
def fence_state_module():
    m = sys.modules.get("_dm_tcf_state")
    if m is None:
        m = types.ModuleType("_dm_tcf_state")
        m.TCF = [False]          # current mode while a kernel body is traced (0 / 1 minimal / 2 blanket)
        m.SKIP = [False]         # FenceSkipSite debug suppression
        m.STATS = {"marks": 0, "fences": 0}   # per kernel trace (reset by begin_kernel_fences): marked objects, wrapper fences
        sys.modules["_dm_tcf_state"] = m
    return m


FENCE_STATE = fence_state_module()


FENCE_MODE = FENCE_STATE.TCF


FENCE_STATS = FENCE_STATE.STATS


TCF_MASK = int(os.environ.get("DM_TCF_MASK", "15"))   # DEBUG bisect: 1 waits(after) 2 releases(before) 4 relay 8 bars


class FenceBuildTrap(RuntimeError):
    """
    A kernel was traced with tcgen05 fences on, but the pipeline fence wrappers emitted
    none (stale binding).
    """


def begin_kernel_fences(on):
    """
    0 = off (no fences), 1 = minimal (default: tcgen05 pipelines / barriers marked with mark_fenced,
    explicit relay sites), 2 = blanket (every mbarrier / named-barrier wait and arrive, every
    sync_warp / elect_one).
    """
    verify_fences()
    FENCE_MODE[0] = int(on)
    FENCE_STATS["marks"] = 0
    FENCE_STATS["fences"] = 0


def save_fence_count():
    """
    Trace time: save / restore the build-trap guard's wrapper-fence tally.
    """
    return int(FENCE_STATS["fences"])


def restore_fence_count(n):
    FENCE_STATS["fences"] = n


def end_kernel_fences():
    """
    Ends a kernel trace. Raises FenceBuildTrap (before cute.compile -> nothing is compiled or
    cached) when fences were on, pipelines were marked, and no wrapper fence came out: the wrappers
    are not bound to this state (BUILD TRAP).
    """
    mode, marks, fences = FENCE_MODE[0], FENCE_STATS["marks"], FENCE_STATS["fences"]
    FENCE_MODE[0] = False
    if (mode and marks and not fences and TCF_MASK == 15 and not os.environ.get("DM_TCF_SKIP")
            and os.environ.get("DM_TCF_NOGUARD", "0") != "1"):
        raise FenceBuildTrap(f"[tcgen05_fence_sm100] BUILD TRAP: kernel traced with FENCE_MODE={mode} and {marks} marked tcgen05 pipelines / "
                           "barriers, but the pipeline fence wrappers emitted 0 fences (wrappers bound to a stale "
                           "delta_matching import?). Refusing to compile a fence-less kernel.")


def raw_method(f):
    """
    Strip installed fence wrappers (this tree's: __wrapped__; older trees': the closure cell of
    the wrapped callable).
    """
    while callable(f) and (getattr(f, "_dm_tcf", False) or getattr(f, "_dm_tcfm", False)):
        g = getattr(f, "__wrapped__", None)
        if g is None:
            code, cl = getattr(f, "__code__", None), getattr(f, "__closure__", None) or ()
            names = code.co_freevars if code is not None else ()
            for n in ("f", "_el"):   # closure names of older trees' wrappers (before __wrapped__)
                if n in names:
                    g = cl[names.index(n)].cell_contents
                    break
        if g is None:
            raise FenceBuildTrap(f"[tcgen05_fence_sm100] cannot unwrap installed fence wrapper {f!r}")
        f = g
    return f


def wrap_fences(f, before, after, mode=2):
    f = raw_method(f)

    def w(*a, **kw):
        if FENCE_MODE[0] >= mode and before:
            tcgen05_fence_before()
            FENCE_STATS["fences"] += 1
        r = f(*a, **kw)
        if FENCE_MODE[0] >= mode and after:
            tcgen05_fence_after()
            FENCE_STATS["fences"] += 1
        return r
    w._dm_tcf = True
    w._dm_tcf_state = FENCE_STATE
    w.__wrapped__ = f
    w.__name__ = getattr(f, "__name__", "w")
    w.__doc__ = f.__doc__
    return w


def install_fences():
    import cutlass.cute.arch as cute_arch
    g = globals()
    for n in ("mbar_wait_parity", "mbar_wait_parity_nh", "mbar_wait_parity_cluster"):
        g[n] = wrap_fences(g[n], False, True)
    for n in ("mbar_arrive", "mbar_arrive_local", "mbar_arrive_rank", "mbar_arrive_expect_tx"):
        g[n] = wrap_fences(g[n], True, False)
    for n, b, a in (("mbarrier_wait", False, True), ("mbarrier_try_wait", False, True),
                    ("mbarrier_conditional_try_wait", False, True), ("mbarrier_test_wait", False, True),
                    ("mbarrier_arrive", True, False), ("mbarrier_arrive_and_expect_tx", True, False),
                    ("cp_async_mbarrier_arrive_noinc", True, False),
                    ("barrier", True, True), ("barrier_arrive", True, False)):
        f = getattr(cute_arch, n, None)
        if f is not None:
            setattr(cute_arch, n, wrap_fences(f, b, a))
    # warp -> elected-leader relays: every lane fences its own tcgen05 ops before the warp sync, so the
    # leader's later arrive publishes the whole warp's ops. elect.sync already requires the full warp, so the added
    # bar.warp.sync is legal wherever elect_one is.
    sync_warp_raw = raw_method(cute_arch.sync_warp)
    RAW_SYNC_WARP[0] = sync_warp_raw
    cute_arch.sync_warp = wrap_fences(sync_warp_raw, True, True)
    elect_one_raw = raw_method(cute_arch.elect_one)

    def elect_one(*a, **kw):
        if FENCE_MODE[0] >= 2:
            tcgen05_fence_before()
            FENCE_STATS["fences"] += 1
            sync_warp_raw()
        return elect_one_raw(*a, **kw)
    elect_one._dm_tcf = True
    elect_one._dm_tcf_state = FENCE_STATE
    elect_one.__wrapped__ = elect_one_raw
    elect_one.__doc__ = elect_one_raw.__doc__
    cute_arch.elect_one = elect_one
    # flash_attn NamedBarrier.arrive_and_wait_unaligned calls nvvm.barrier_cta_sync directly (bypasses cute.arch.barrier)
    try:
        import flash_attn.cute.pipeline as fa_pipeline
        named_barrier = fa_pipeline.NamedBarrier
        for n, b, a in (("arrive_and_wait_unaligned", True, True), ("arrive_unaligned", True, False)):
            f = named_barrier.__dict__.get(n)
            if f is not None:
                setattr(named_barrier, n, wrap_fences(f, b, a))
    except Exception:   # pragma: no cover
        pass


# Minimal mode (FENCE_MODE == 1): fences only on objects marked by mark_fenced (tcgen05 pipelines + barriers that relay
# tcgen05 state). Marking swaps the instance's class to a subclass (like flash_attn's create()); the wrappers check it.
# kinds: "ua" = PipelineUmmaAsync (MMA producer via tcgen05.commit -> thread consumer), "au" = PipelineAsyncUmma
# (thread producer -> MMA consumer, released via tcgen05.commit), "bar" = named barrier.
MARKED_CLASSES = {}


def mark_fenced(kind, *objs):
    for o in objs:
        if o is None:
            continue
        c = type(o)
        k = (c, kind)
        if k not in MARKED_CLASSES:
            MARKED_CLASSES[k] = type(c.__name__, (c,), {"_dm_tcf": kind})
        object.__setattr__(o, "__class__", MARKED_CLASSES[k])
        FENCE_STATS["marks"] += 1
    return objs


def wrap_marked_method(f, role):
    f = raw_method(f)

    def w(self, *a, **kw):
        k = getattr(type(self), "_dm_tcf", None) if FENCE_MODE[0] == 1 and not TCF_SKIP[0] else None
        if k == "bar" and not (TCF_MASK & 8):
            k = None
        before = after = False
        if k is not None:
            if role == "wait":
                after = True
            elif role == "rel":     # thread arrive; tcgen05.commit arrives need no before-fence
                before = not ((k == "ua" and f.__name__.startswith("producer_commit"))
                              or (k == "au" and f.__name__.startswith("consumer_release")))
            elif role == "both":
                before = after = True
        if before and k != "bar" and TCF_MASK & 16 and k == "au":
            before = False   # DEBUG bisect: no before-fence on thread->MMA commits
        if before and k != "bar" and TCF_MASK & 32 and k == "ua":
            before = False   # DEBUG bisect: no before-fence on thread releases of MMA-produced buffers
        if before and (TCF_MASK & 2 or k == "bar"):
            tcgen05_fence_before()
            FENCE_STATS["fences"] += 1
        r = f(self, *a, **kw)
        if after and (TCF_MASK & 1 or k == "bar"):
            tcgen05_fence_after()
            FENCE_STATS["fences"] += 1
        return r
    w._dm_tcfm = True
    w._dm_tcf_state = FENCE_STATE
    w.__wrapped__ = f
    w.__name__ = f.__name__
    return w


FENCE_ROLES = {"consumer_wait": "wait", "producer_acquire": "wait", "consumer_release": "rel", "producer_commit": "rel",
              "arrive": "rel", "arrive_unaligned": "rel", "arrive_w_index": "rel", "wait": "wait",
              "wait_unaligned": "wait", "arrive_and_wait": "both", "arrive_and_wait_unaligned": "both",
              "arrive_and_wait_w_index": "both", "sync": "both"}


def fence_classes():
    import cutlass.pipeline as cutlass_pipeline
    import flash_attn.cute.pipeline as fa_pipeline
    seen = []
    for mod in (cutlass_pipeline, fa_pipeline):
        for n in dir(mod):
            c = getattr(mod, n)
            if not isinstance(c, type) or c in seen:
                continue
            seen.append(c)
            if n.startswith("Pipeline") or n == "NamedBarrier":
                yield c


def install_marked_fences():
    for c in fence_classes():
        for m, r in FENCE_ROLES.items():
            f = c.__dict__.get(m)
            if f is not None and callable(f):
                setattr(c, m, wrap_marked_method(f, r))


def verify_fences():
    """
    True when every installed fence wrapper is bound to this process's fence state (this tree's
    wrappers). Otherwise (another delta_matching import re-wrapped the shared classes, e.g. an
    older tree without the shared state) the wrappers are re-installed from this module, with a
    warning on stderr.
    """
    import cutlass.cute.arch as cute_arch
    ok = all(getattr(c.__dict__[m], "_dm_tcf_state", None) is FENCE_STATE
             for c in fence_classes() for m in FENCE_ROLES if callable(c.__dict__.get(m)))
    ok = ok and all(getattr(getattr(cute_arch, n), "_dm_tcf_state", None) is FENCE_STATE
                    for n in ("sync_warp", "elect_one", "barrier") if hasattr(cute_arch, n))
    if not ok:
        print("[tcgen05_fence_sm100] tcgen05 fence wrappers were not bound to this delta_matching import: re-installed "
              "(build trap guard)", file=sys.stderr, flush=True)
        install_fences()
        install_marked_fences()
    return ok


TCF_SKIP = FENCE_STATE.SKIP


class FenceSkipSite:
    """
    DEBUG bisect: `with FenceSkipSite("id"):` suppresses minimal-mode fences inside if id is
    in env DM_TCF_SKIP.
    """
    def __init__(self, sid):
        self.on = sid in os.environ.get("DM_TCF_SKIP", "").split(",")

    def __enter__(self):
        if self.on:
            TCF_SKIP[0] = True

    def __exit__(self, *a):
        TCF_SKIP[0] = False


def fence_warp_relay():
    """
    Minimal mode: every lane fences its tcgen05 ops, then the warp syncs, before an
    elected leader's arrive.
    """
    if FENCE_MODE[0] == 1 and TCF_MASK & 4:
        tcgen05_fence_before()
        RAW_SYNC_WARP[0]()


RAW_SYNC_WARP = [None]


install_fences()
install_marked_fences()
