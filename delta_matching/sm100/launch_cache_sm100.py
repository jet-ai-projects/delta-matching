"""
Host fast path: record-and-replay of FA4 host entries.

The first call of a (shape, strides, dtypes, options) key goes through FA4's
full host interface (_flash_attn_fwd / _flash_attn_bwd: config selection,
validation, tensor conversion, compile-cache lookup) with the compile caches
wrapped so that every compiled-kernel invocation is RECORDED (compiled fn +
argument structure).  A plan is then built that maps every tensor argument
either to one of the caller's input tensors (same storage, recreated as the same
view of the first input on that storage; the aliasing pattern of the inputs is
part of the key), or to a fresh allocation whose provenance was observed while
recording: torch.empty(_like) / torch.zeros(_like) (hooked during the recording
call; empty results sentinel-filled) and untouched by host code before the first
kernel that uses it.  Later calls with the same key skip the host interface and
just call the recorded compiled functions with the new tensors.  Kernel
launches, kernel arguments' values and launch order are identical, so results
are bitwise identical to the slow path.  Any buffer of other provenance (made by
another op, written by host code, never seen by a kernel) disables the fast path
for that key (WHY records the reason). Disable globally with DM_FASTPATH=0.
"""
import os
import threading

import torch

__all__ = ["ENABLED", "EPOCH", "WHY", "ND_REL", "ND_BLK", "ND_ELEM", "env", "RecCache", "Plan", "key_of", "call"]

ENABLED = os.environ.get("DM_FASTPATH", "1") != "0"
THREAD_STATE = threading.local()
EPOCH = [0]     # bumped by every compile-cache miss / insert and every cute.compile (incl. the CuTe producers, quant.py)

try:                        # os.environ.get without the Mapping machinery (same lookup + decode; ~10x cheaper)
    ENV_DATA, ENV_ENCODE, ENV_DECODE = os.environ._data, os.environ.encodekey, os.environ.decodevalue
except AttributeError:
    ENV_DATA = None
ENV_KEYS = {}


def env(k, d=None):
    """
    == os.environ.get(k, d), read live on every call (tests / debug hooks change the
    env at runtime).
    """
    if ENV_DATA is None:
        return os.environ.get(k, d)
    kb = ENV_KEYS.get(k)
    if kb is None:
        kb = ENV_KEYS[k] = ENV_ENCODE(k)
    v = ENV_DATA.get(kb)
    return d if v is None else ENV_DECODE(v)


class RecCache:
    """
    Wraps an FA4 compile cache; while recording, compiled functions fetched from it are wrapped
    to log calls.
    """

    def __init__(self, cache):
        self.inner_cache = cache

    def __contains__(self, k):
        r = k in self.inner_cache
        if not r:
            EPOCH[0] += 1
        return r

    def __getitem__(self, k):
        fn = self.inner_cache[k]
        rec = getattr(THREAD_STATE, "rec", None)
        if rec is None:
            return fn

        def proxy(*args, **kw):
            rec.log(fn, args, kw)
            return fn(*args, **kw)
        return proxy

    def __setitem__(self, k, v):
        EPOCH[0] += 1
        self.inner_cache[k] = v

    def get(self, k, default=None):
        return self[k] if k in self else default      # a miss bumps EPOCH (via __contains__)

    def __getattr__(self, n):
        return getattr(self.inner_cache, n)


def tensor_leaves(x, out):
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, (tuple, list)):
        for y in x:
            tensor_leaves(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            tensor_leaves(y, out)
    return out


def storage_ptr(t):
    return t.untyped_storage().data_ptr()


SENTINEL = 0xFF          # recording only: torch.empty results are filled with this byte (NaN for float dtypes)


class Recorder:
    def __init__(self, input_ptrs):
        self.calls = []
        self.zero = {}            # storage ptr -> bool (all bytes zero when the first kernel touching it was called)
        self.input_ptrs = input_ptrs
        self.kind = {}            # storage ptr -> "e" / "z": allocated by torch.empty(_like) / torch.zeros(_like)
        self.state = {}           # storage ptr -> "e" (still all-sentinel) / "z" (still all-zero) / False, at first use
        self.keep = []            # hooked allocations stay alive while recording: their pointers can't be reused

    def log(self, fn, args, kw):
        for t in tensor_leaves((args, kw), []):
            p = storage_ptr(t)
            if p in self.input_ptrs or p in self.zero:
                continue
            st = t.untyped_storage()
            b = torch.empty(0, dtype=torch.uint8, device=t.device).set_(st)
            self.zero[p] = bool((b == 0).all().item()) if b.numel() else True
            k = self.kind.get(p)
            if k == "e":
                self.state[p] = "e" if (not b.numel() or bool((b == SENTINEL).all().item())) else False
            elif k == "z":
                self.state[p] = "z" if self.zero[p] else False
        self.calls.append((fn, args, kw))

    def hooks(self):
        """
        Patch torch allocation entry points (recording call only) to learn how each kernel
        buffer was created.
        """
        orig = {n: getattr(torch, n) for n in ("empty", "zeros", "empty_like", "zeros_like")}

        def mk(n, kind):
            o = orig[n]

            def w(*a, **k):
                t = o(*a, **k)
                if isinstance(t, torch.Tensor) and t.is_cuda and type(t) is torch.Tensor:
                    if kind == "e" and t.numel():
                        t.view(-1).view(torch.uint8).fill_(SENTINEL) if t.is_contiguous() else None
                    if kind == "z" or t.is_contiguous():
                        self.kind[storage_ptr(t)] = kind
                        self.keep.append(t)
                return t
            return w
        for n, kind in (("empty", "e"), ("empty_like", "e"), ("zeros", "z"), ("zeros_like", "z")):
            setattr(torch, n, mk(n, kind))
        return orig


def view_spec(t, base):
    """
    Spec to recreate t from base (same storage): (dtype, size, stride, byte offset
    from base's data_ptr).
    """
    return (t.dtype, tuple(t.shape), tuple(t.stride()), t.data_ptr() - base.data_ptr())


def make_view(base, spec):
    dtype, size, stride, boff = spec
    b = base if base.dtype == dtype else base.view(dtype)
    es = b.element_size()
    if boff % es:
        raise RuntimeError("fastpath: misaligned view")
    return torch.as_strided(b, size, stride, b.storage_offset() + boff // es)


class Plan:
    """
    Replays recorded compiled calls. inputs: name -> tensor (caller's tensors); allocs
    recreated per call.
    """

    def __init__(self, rec: Recorder, inputs: dict, ret):
        in_by_ptr = {}
        for name, t in inputs.items():
            if t is not None:
                in_by_ptr.setdefault(storage_ptr(t), (name, t))
        allocs = {}            # storage ptr -> [nbytes, device, zero, returned]
        ret_leaves = tensor_leaves(ret, [])
        ret_ptrs = {storage_ptr(t) for t in ret_leaves}

        def spec_of(t):
            p = storage_ptr(t)
            if p in in_by_ptr:
                name, base = in_by_ptr[p]
                return ("in", name, view_spec(t, base))
            if p not in allocs:
                st = t.untyped_storage()
                k = rec.state.get(p)
                # only buffers of known provenance are replayable: created by a hooked torch.empty(_like) /
                # torch.zeros(_like) and untouched by host code until the first kernel that uses them (re-created the
                # same way). Anything else (made by another torch op, written by host code, never seen by a kernel)
                # makes the plan non-replayable (its bytes would depend on the recording call's data).
                zero = {"z": True, "e": False, False: "modified"}.get(k, "unknown")
                allocs[p] = [st.nbytes(), t.device, zero, p in ret_ptrs, st.data_ptr()]
            a = allocs[p]
            return ("alloc", p, (t.dtype, tuple(t.shape), tuple(t.stride()), t.data_ptr() - a[4]))

        def conv(x):
            if isinstance(x, torch.Tensor):
                return spec_of(x)
            if isinstance(x, tuple) and hasattr(x, "_fields"):
                return ("nt", type(x), tuple(conv(y) for y in x))
            if isinstance(x, tuple):
                return ("tuple", tuple(conv(y) for y in x))
            if isinstance(x, list):
                return ("list", [conv(y) for y in x])
            if isinstance(x, dict):
                return ("dict", {k: conv(v) for k, v in x.items()})
            return ("const", x)

        self.calls = [(fn, conv(args), conv(kw)) for fn, args, kw in rec.calls]
        self.ret = conv(ret)
        # safety: every allocation must be zero-initialised (we zero it) or be an output (overwritten by the kernels)
        self.ok = True
        self.why = []
        for p, (nb, dev, zero, returned, _) in allocs.items():
            if zero == "modified":
                self.ok = False      # host code wrote into a fresh buffer before the kernel: not replayable
                self.why.append(("host-modified", nb))
            elif zero == "unknown":
                self.ok = False
                self.why.append(("unknown-provenance", nb, returned))
            elif zero is None and not returned:
                self.ok = False      # a returned-only tensor never passed to a kernel / unknown provenance
                self.why.append(("unknown-not-returned", nb))
            elif zero is False and not returned:
                self.ok = False
                self.why.append(("nonzero-scratch", nb))
        self.allocs = {p: (nb, dev, zero is True) for p, (nb, dev, zero, returned, _) in allocs.items()}
        self.verified = False
        self.inputs_tpl = {n: (t.dtype, tuple(t.shape), tuple(t.stride())) for n, t in inputs.items() if t is not None}
        self.replay = self.generate_replay()

    def generate_replay(self):
        """
        Specialised replay function: straight-line Python that rebuilds exactly
        the recorded arguments.
        """
        C, T = [], []
        env = {"torch": torch, "make_view": make_view, "C": C, "T": T}
        pre, memo = [], {}
        # allocations: typed directly when one contiguous view covers the whole storage, else a uint8 buffer
        views_by_alloc = {}

        def collect(s):
            k = s[0]
            if k == "alloc":
                views_by_alloc.setdefault(s[1], []).append(s[2])
            elif k in ("tuple", "list"):
                for y in s[1]:
                    collect(y)
            elif k == "nt":
                for y in s[2]:
                    collect(y)
            elif k == "dict":
                for y in s[1].values():
                    collect(y)
        for fn, a, kw in self.calls:
            collect(a), collect(kw)
        collect(self.ret)
        abase = {}
        for n_, (p, (nb, dev, zero)) in enumerate(self.allocs.items()):
            env[f"D{n_}"] = dev
            whole = None
            for (dt, size, stride, boff) in views_by_alloc.get(p, []):
                es = torch.empty(0, dtype=dt).element_size()
                numel = 1
                for x in size:
                    numel *= x
                cstride, acc = [], 1
                for x in reversed(size):
                    cstride.append(acc)
                    acc *= x
                if boff == 0 and numel * es == nb and tuple(reversed(cstride)) == tuple(stride):
                    whole = (dt, size)
                    break
            fn_ = "torch.zeros" if zero else "torch.empty"
            if whole is not None:
                env[f"DT{n_}"] = whole[0]
                pre.append(f"    B{n_} = {fn_}({tuple(whole[1])!r}, dtype=DT{n_}, device=D{n_})")
            else:
                pre.append(f"    B{n_} = {fn_}({nb}, dtype=torch.uint8, device=D{n_})")
            abase[p] = (f"B{n_}", whole)

        def tens(key, expr):
            if key not in memo:
                v = f"v{len(memo)}"
                pre.append(f"    {v} = {expr}")
                memo[key] = v
            return memo[key]

        def gen(s):
            k = s[0]
            if k == "in":
                name, spec = s[1], s[2]
                dt, size, stride, boff = spec
                t_dt, t_size, t_stride = self.inputs_tpl[name]
                base = f"I[{name!r}]"
                if boff == 0 and size == t_size and stride == t_stride:
                    if dt == t_dt:
                        return base
                    if torch.empty(0, dtype=dt).element_size() == torch.empty(0, dtype=t_dt).element_size():
                        C.append(dt)
                        return tens(("in", name, spec), f"{base}.view(C[{len(C) - 1}])")
                C.append(spec)
                return tens(("in", name, spec), f"make_view({base}, C[{len(C) - 1}])")
            if k == "alloc":
                bvar, whole = abase[s[1]]
                dt, size, stride, boff = s[2]
                if whole is not None and boff == 0 and whole == (dt, size):
                    return bvar
                C.append(s[2])
                return tens(("alloc", s[1], s[2]), f"make_view({bvar}, C[{len(C) - 1}])")
            if k == "nt":
                T.append(s[1])
                return f"T[{len(T) - 1}](" + "".join(gen(y) + ", " for y in s[2]) + ")"
            if k == "tuple":
                return "(" + "".join(gen(y) + ", " for y in s[1]) + ")"
            if k == "list":
                return "[" + ", ".join(gen(y) for y in s[1]) + "]"
            if k == "dict":
                return "{" + ", ".join(f"{kk!r}: {gen(v)}" for kk, v in s[1].items()) + "}"
            C.append(s[1])
            return f"C[{len(C) - 1}]"
        body = []
        for i_, (fn, a, kw) in enumerate(self.calls):
            env[f"F{i_}"] = fn
            body.append(f"    F{i_}(*{gen(a)}, **{gen(kw)})")
        ret = gen(self.ret)
        src = "def replay(I):\n" + "\n".join(pre + body) + f"\n    return {ret}\n"
        self.src = src
        exec(compile(src, "<dm_fastpath_plan>", "exec"), env)
        return env["replay"]

    def run(self, inputs: dict):
        return self.replay(inputs)


PLANS = {}
WHY = {}


def key_of(tag, inputs: dict, extra):
    parts = [tag, extra]
    first = {}           # storage ptr -> (index, data_ptr) of the first input on that storage
    for i, (name, t) in enumerate(inputs.items()):   # call sites build the dict in a fixed order; order is in the key
        if t is None:
            parts.append((name, None))
        else:
            dp = t.data_ptr()
            sp = t.untyped_storage().data_ptr()
            f = first.get(sp)
            if f is None:
                f = first[sp] = (i, dp)
            # alias signature: (first input sharing this storage, byte offset from it); plans rebuild aliased inputs
            # as views of that first input, so a call with different aliasing must not hit the same plan
            parts.append((name, t.shape, t.stride(), t.dtype, t.get_device(), dp % 128, f[0], dp - f[1]))
    return tuple(parts)


def flat_return(r):
    return list(r) if isinstance(r, (tuple, list)) else [r]


ND_REL, ND_BLK, ND_ELEM = 2e-4, 5e-4, 1.6e-2     # = the accepted run-to-run limits (whole, 128-row block, element)


def nd_spread(x, y):
    """
    None if y is within the accepted non-deterministic run-to-run limits of x ((B, N, H, D)
    gradients): finite, whole-tensor rel L2 <= 2e-4, every (batch, 128-row block, head) rel L2
    <= 5e-4, every element |y - x| <= 1.6e-2 of its row's max |x| (2 bf16 ulp).  Else the
    first violated criterion.
    """
    xf, yf = x.float(), y.float()
    if not bool(torch.isfinite(yf).all()):
        return ("nonfinite",)
    d = yf - xf
    rel = (d.norm() / xf.norm().clamp_min(1e-30)).item()
    if rel > ND_REL:
        return ("whole", rel)
    if xf.dim() == 4 and xf.shape[1] % 128 == 0:
        B, N, H, D = xf.shape
        xb = xf.view(B, N // 128, 128, H, D).permute(0, 1, 3, 2, 4).reshape(B, N // 128, H, -1)
        db = d.view(B, N // 128, 128, H, D).permute(0, 1, 3, 2, 4).reshape(B, N // 128, H, -1)
        blk = (db.norm(dim=-1) / xb.norm(dim=-1).clamp_min(1e-30)).max().item()
        if blk > ND_BLK:
            return ("block", blk)
        el = (d.abs() / xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)).max().item()
        if el > ND_ELEM:
            return ("elem", el)
    else:
        return ("shape", tuple(xf.shape))
    return None


def verify_plan(k, plan, inputs, slow, pre, reset, cmp, loose):
    """
    First call after recording: the normal host path (its result is returned) AND the replay on the
    same inputs into separate outputs; the plan is enabled only if they agree bitwise (except the
    `loose` outputs, non-deterministic accumulation order: the accepted run-to-run limits,
    nd_spread).  Mismatch or error -> key disabled (WHY).
    """
    ret = slow()
    why = None
    try:
        if reset is not None:
            reset()
        if pre is not None:
            pre()
        rep = plan.run(inputs)
        a, b = flat_return(ret), flat_return(rep)
        idx = range(len(a)) if cmp is None else cmp
        for i in idx:
            x, y = a[i], b[i]
            if x is None and y is None:
                continue
            if x.shape != y.shape or x.dtype != y.dtype:
                why = ("verify-meta", i)
                break
            if i in loose:
                bad = nd_spread(x, y)
                if bad is not None:
                    why = ("verify-loose", i, bad)
                    break
            elif not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)):   # bitwise
                why = ("verify-mismatch", i)
                break
    except Exception as e:
        why = ("verify-error", repr(e))
    if why is None:
        plan.verified = True
        WHY[k] = ["verified"]
    else:
        PLANS[k] = False
        WHY[k] = [why]
        import sys
        print(f"[dm fastpath] plan for {k[0]} disabled: {why}", file=sys.stderr, flush=True)
    return ret


def call(tag, inputs: dict, extra, slow, pre=None, reset=None, cmp=None, loose=()):
    """
    slow(): runs the normal host path (must fetch compiled fns through RecCache-wrapped
    caches) and returns ret. pre(): host work the replay needs right before its kernels;
    reset(): restores input-side scratch the slow path consumed (verification call). cmp:
    indices of ret compared in verification (None: all); loose: indices compared with a
    tolerance (non-deterministic accumulation).
    """
    if not ENABLED or type(next(iter(inputs.values()))) is not torch.Tensor:    # fake / subclass tensors: normal path
        return slow()
    k = key_of(tag, inputs, extra)
    plan = PLANS.get(k)
    if plan is not None:
        if plan is False:
            return slow()
        if not plan.verified:
            return verify_plan(k, plan, inputs, slow, pre, reset, cmp, loose)
        if pre is not None:
            pre()
        return plan.run(inputs)
    if any(isinstance(t, torch._subclasses.fake_tensor.FakeTensor) for t in inputs.values() if t is not None):
        return slow()          # fake tensors (AOT compile): never planned (no plan exists for their keys)
    rec = Recorder({storage_ptr(t) for t in inputs.values() if t is not None})
    THREAD_STATE.rec = rec
    orig = rec.hooks()
    import cutlass.cute as cute
    ccomp = cute.compile

    def compile_counted(*a, **kw):
        EPOCH[0] += 1
        return ccomp(*a, **kw)
    cute.compile = compile_counted
    e0 = EPOCH[0]
    try:
        ret = slow()
    finally:
        cute.compile = ccomp
        for n, f in orig.items():
            setattr(torch, n, f)
        THREAD_STATE.rec = None
    if EPOCH[0] != e0:
        # something was compiled / inserted into a compile cache during this call (cold cache): compiled functions
        # may have been called outside the recorder (FA4's bwd calls a freshly compiled kernel directly) -> no plan
        # from this call; a later all-hit call records.
        WHY[k] = ["compiled-during-recording (will record on a later call)"]
        return ret
    try:
        p = Plan(rec, inputs, ret)
        PLANS[k] = p if (p.ok and p.calls) else False
        WHY[k] = p.why if p.calls else ["no-calls"]
    except Exception as e:
        PLANS[k] = False
        WHY[k] = [repr(e)]
    return ret
