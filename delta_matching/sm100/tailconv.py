"""rw8k tconv: tail converter = FA4's backward postprocess (fp32 accumulator * scale -> bf16), launched with PDL right
after the deterministic backward main kernel, each CTA converting its tile as soon as the tile is FINAL.

Readiness (no new hand-off: the det dQ / dK / dV lock counters the backward already maintains):
  dQ tile (b, h, m), slot r of the 2-CTA pair: final count = number of KV cluster tiles that write it
      = min(ceil(N / ctn), ceil((m + 1) * tile_m / ctn)) causal, ceil(N / ctn) otherwise   (ctn = cluster tile n = 256)
  GQA dK / dV tile (b, hk, n), slot r: final count = G (query heads per KV head)
  (verified on the hardware: 8k / 1152 / 4k non-causal / 2k MHA).
Protocol per converter CTA: thread 0 polls every slot of its tile with ld.acquire.gpu until >= final (bounded: trap after
~2^26 polls, never a silent hang) -> bar.sync (CTA) -> every thread fence.proxy.async.global (the writers' bulk reduce-adds
are async-proxy writes) -> FA4's unchanged conversion code -> griddepcontrol.wait (deferred, before exit).
The writers (backward, tconv knob) execute fence.proxy.async.global before every det semaphore release, and the backward
issues griddepcontrol.launch_dependents after its TMEM allocation (the converter needs no TMEM).
Semaphore reset / ABA: the counters are zeroed per call by the backward host path (prestage side stream, which waits on the
main stream, i.e. after the previous call's converter) before the next backward main kernel; the converter only reads.
Controls (DM_TCONV_CTL, debug): early = poll for final - 1 (reads before the last writer released -> wrong bits, caught);
drop = poll for final + 1 (the flag never arrives -> bounded wait traps).
Same arithmetic and same loads/stores as FA4's postprocess: dQ / dK / dV bitwise equal to the serial postprocess."""
import inspect
import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

import flash_attn.cute.flash_bwd_postprocess as _fpp
from delta_matching.sm100.cute_ptx_sm100 import fence_proxy_async_global, ld_shared_b32, poll_acquire_ge_or_trap, st_shared_b32

@dsl_user_op
def poll_acquire_nz_or_trap(addr: cutlass.Int64, limit: Int32, *, loc=None, ip=None) -> Int32:
    """rw8k tconv: spin with ld.acquire.gpu until the int32 at addr is non-zero (trap after `limit` polls); returns it."""
    return Int32(llvm.inline_asm(
        T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(limit).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .b32 it;\n\tmov.b32 it, 0;\n"
        "DM_PNZ:\n\tld.acquire.gpu.global.b32 $0, [$1];\n\tsetp.ne.s32 p, $0, 0;\n\t@p bra DM_PNZD;\n\t"
        "add.s32 it, it, 1;\n\tsetp.gt.s32 p, it, $2;\n\t@p trap;\n\tnanosleep.u32 256;\n\tbra DM_PNZ;\n"
        "DM_PNZD:\n\t}",
        "=r,l,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


_SRC = inspect.getsource(_fpp)


def _patch(src: str) -> str:
    subs = [
        ("class FlashAttentionBackwardPostprocess:", "class DMTailConv:"),
        # __call__: two extra arguments (readiness counters, kind) before the stream
        ("        mCuTotalMBlocks: Optional[cute.Tensor] = None,\n        # Always keep stream as the last parameter",
         "        mCuTotalMBlocks: Optional[cute.Tensor] = None,\n        mDmSem: Optional[cute.Tensor] = None,\n        mDmQ: Optional[cute.Tensor] = None,\n"
         "        # Always keep stream as the last parameter"),
        ("            tile_sched_params,\n            TileScheduler,\n        ).launch(\n            grid=grid_dim,\n"
         "            block=[self.num_threads, 1, 1],\n            smem=smem_size,\n            stream=stream,\n        )",
         "            tile_sched_params,\n            TileScheduler,\n            mDmSem,\n            mDmQ,\n        ).launch(\n            grid=grid_dim,\n"
         "            block=[self.num_threads, 1, 1],\n            smem=smem_size,\n            stream=stream,\n"
         "            use_pdl=True,\n        )"),
        ("        tile_sched_params: ParamsBase,\n        TileScheduler: cutlass.Constexpr[Callable],\n    ):\n"
         "        # ///////////////////////////////////////////////////////////////////////////////\n"
         "        # Get shared memory buffer",
         "        tile_sched_params: ParamsBase,\n        TileScheduler: cutlass.Constexpr[Callable],\n"
         "        mDmSem: Optional[cute.Tensor],\n        mDmQ: Optional[cute.Tensor],\n    ):\n"
         "        # ///////////////////////////////////////////////////////////////////////////////\n"
         "        # Get shared memory buffer"),
        ("        m_block, head_idx, batch_idx, _ = work_tile.tile_idx\n",
         "        m_block, head_idx, batch_idx, _ = work_tile.tile_idx\n"
         "        cute.arch.griddepcontrol_launch_dependents()   # rw8k: the next kernel may launch now\n"
         "        if const_expr(mDmQ is not None and self.dm_ctl != 'early'):\n"
         "            m_block, head_idx, batch_idx = self._dm_pick(mDmSem, mDmQ, sdQaccum_flat, tidx)\n"
         "        if work_tile.is_valid_tile:\n"
         "            self._dm_ready(mDmSem, m_block, head_idx, batch_idx, tidx, mdQ.shape[1])\n"),
    ]
    # 2-CTA dQ path: the tile goes gmem -> registers -> sdQ (bf16) -> gmem; the fp32 sdQaccum staging buffer is not
    # used there, so size the smem for sdQ only (2x the converter CTAs per SM in bwd main's tail)
    subs += [
        ("        smem_size = max(\n",
         "        smem_size = cute.size_in_bytes(self.dtype, self.sdQ_layout) if const_expr(self.use_2cta_instrs) else max(\n"),
        ("        sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)\n",
         "        if const_expr(self.use_2cta_instrs):\n"
         "            sdQaccum = smem.allocate_tensor(cutlass.Float32, cute.make_layout(cute.size_in_bytes(self.dtype, sdQ_layout) // 4), byte_alignment=1024)\n"
         "        else:\n"
         "            sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)\n"),
    ]
    for a, b in subs:
        if src.count(a) != 1:
            raise RuntimeError(f"tailconv patch anchor not unique/found ({src.count(a)}): {a[:80]!r}")
        src = src.replace(a, b)
    # deferred griddepcontrol.wait at the very end of the kernel (the module ends with the kernel body)
    # deferred griddepcontrol.wait, executed by the LAST CTA of the grid only: the converter grid therefore cannot complete
    # before bwd main has completed and its memory is flushed (stream order for the dK / dV postprocess that follows),
    # while no other converter CTA holds its SM slot until bwd main ends (that blocked the tail fill: rw8k stage 2).
    # Every read of bwd main's output (dq_accum) is ordered by the per-tile ld.acquire of the det counters instead.
    src = src.rstrip("\n") + ("\n        if const_expr(self.dm_gdcwait):\n"
                               "            dm_bx, dm_by, dm_bz = cute.arch.block_idx()\n"
                               "            dm_gx, dm_gy, dm_gz = cute.arch.grid_dim()\n"
                               "            if dm_bx == dm_gx - 1 and dm_by == dm_gy - 1 and dm_bz == dm_gz - 1:\n"
                               "                cute.arch.griddepcontrol_wait()\n")
    return src


_ns = dict(vars(_fpp))
_FN = _fpp.__file__.replace(".py", "_dm_tailconv.py")
_PSRC = _patch(_SRC)
import linecache  # noqa: E402
linecache.cache[_FN] = (len(_PSRC), None, _PSRC.splitlines(True), _FN)   # the DSL's AST pass re-reads the source
exec(compile(_PSRC, _FN, "exec"), _ns)
_Base = _ns["DMTailConv"]


class DMTailConv(_Base):
    """kind 'q': dQ (tile_m rows, causal-dependent final count); kind 'kv': GQA dK / dV (final count = G)."""

    def __init__(self, *a, dm_kind="q", dm_causal=True, dm_ctn=256, dm_G=1, dm_ctl="", **k):
        super().__init__(*a, **k)
        self.dm_kind, self.dm_causal, self.dm_ctn, self.dm_G, self.dm_ctl = dm_kind, dm_causal, dm_ctn, dm_G, dm_ctl
        self.dm_gdcwait = os.environ.get("DM_TCONV_GDCWAIT", "1") == "1"

    @cute.jit
    def _dm_pick(self, mDmSem, mDmQ, sbuf, tidx):
        """Readiness queue: the CTA with linear block id i converts the i-th tile that became final (bwd main appends tile
        ids in completion order), so CTAs dispatched in launch order never wait behind a late tile. Thread 0 polls the
        entry (ld.acquire.gpu, bounded -> trap) and broadcasts it through smem; the det counters are re-checked in
        _dm_ready (the queue is only a scheduling hint)."""
        bx, by, bz = cute.arch.block_idx()
        gx, gy, gz = cute.arch.grid_dim()
        pos = bx + gx * (by + gy * bz)
        saddr = sbuf.iterator.toint()
        if tidx == 0:
            if const_expr(self.dm_ctl == "qdrop"):
                # CONTROL: wait for an entry value no producer ever writes (must trap, not hang)
                poll_acquire_ge_or_trap(mDmQ.iterator.toint() + Int64(4) * Int64(1 + pos), Int32(1 << 30), Int32(1 << 22))
            v0 = poll_acquire_nz_or_trap(mDmQ.iterator.toint() + Int64(4) * Int64(1 + pos), Int32(1 << 24))
            st_shared_b32(saddr, v0)
        cute.arch.barrier()
        t = ld_shared_b32(saddr) - 1
        cute.arch.barrier()
        mt = cute.size(mDmSem, mode=[2])
        nh = cute.size(mDmSem, mode=[1])
        return t % mt, (t // mt) % nh, t // (mt * nh)

    @cute.jit
    def _dm_ready(self, mDmSem, m_block, head_idx, batch_idx, tidx, seqlen):
        if const_expr(mDmSem is not None):
            if const_expr(self.dm_kind == "q"):
                nk = (seqlen + self.dm_ctn - 1) // self.dm_ctn
                fin = nk
                if const_expr(self.dm_causal):
                    fin = cutlass.min(nk, ((m_block + 1) * self.tile_m + self.dm_ctn - 1) // self.dm_ctn)
            else:
                fin = Int32(self.dm_G)
            if const_expr(self.dm_ctl == "early"):
                fin = Int32(0)         # CONTROL: no readiness at all (block-id tile, no queue, no counter wait): reads
                                       # tiles before their last writer released -> must be caught by the bit check
            if const_expr(self.dm_ctl == "drop"):
                fin = fin + 1          # CONTROL: the flag never arrives (must trap, not hang)
            if tidx == 0:
                nh = cute.size(mDmSem, mode=[1])
                nt = cute.size(mDmSem, mode=[2])
                ns = cute.size(mDmSem, mode=[3])
                base = mDmSem.iterator.toint() + Int64(4) * Int64(((batch_idx * nh + head_idx) * nt + m_block) * ns)
                for r in cutlass.range_constexpr(cute.size(mDmSem, mode=[3])):
                    poll_acquire_ge_or_trap(base + Int64(4 * r), fin, Int32(1 << 24))
            cute.arch.barrier()
            fence_proxy_async_global()


def _cache():
    from flash_attn.cute import cache_utils
    from .compile_cache_sm100 import fingerprint
    if cache_utils.CUTE_DSL_CACHE_ENABLED:
        return cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint() / "tconv")
    return cache_utils.JITCache()


_CACHE = None


def _compile(key):
    """Fake-tensor compile (mirrors interface._compile_bwd_postprocess, dense only) + the readiness tensor."""
    from flash_attn.cute.interface import make_fake_bwd_tensors
    (dtype, hdim, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs, cluster_size, arch, hdim_multiple_of,
     accum_order, dm_kind, dm_causal, dm_ctn, dm_G, ctl, _gdc) = key
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum, mScaleP = \
        make_fake_bwd_tensors(dtype, has_gqa=True, varlen_q=False, varlen_k=False)
    from quack.compile_utils import make_fake_tensor as _ft
    mSem = _ft(Int32, (cute.sym_int(), cute.sym_int(), cute.sym_int(), 2), divisibility=1)
    mQ_ = _ft(Int32, (cute.sym_int(),), divisibility=1) if dm_kind == "q" else None
    conv = DMTailConv(dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs=use_2cta_instrs,
                      cluster_size=cluster_size, hdim_multiple_of=hdim_multiple_of, accum_order=accum_order,
                      pack_factor=1, dm_kind=dm_kind, dm_causal=dm_causal, dm_ctn=dm_ctn, dm_G=dm_G, dm_ctl=ctl)
    return cute.compile(conv, mdQaccum, mdQ, cutlass.Float32(0.0), None, None, None, None, mSem, mQ_,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")


def tconv_convert(sem, accum, output, scale, cu_seqlens, seqused, arch, dtype, hdim, block_size, num_threads,
                  atom_layout, swap_ab, use_2cta_instrs=False, cluster_size=1, cu_total_m_blocks=None,
                  sink_tensors=None, hdim_multiple_of=32, accum_order=None, pack_factor=1, *, fake_mode,
                  dm_kind, dm_causal, dm_ctn, dm_G, queue=None):
    """Drop-in for interface._bwd_postprocess_convert with the readiness semaphore and the kind / counts."""
    global _CACHE
    assert cu_seqlens is None and seqused is None and sink_tensors is None and pack_factor == 1, "tconv: dense only"
    assert cu_total_m_blocks is None
    if _CACHE is None:
        _CACHE = _cache()
    key = (dtype, hdim, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs, cluster_size, arch,
           hdim_multiple_of, accum_order, dm_kind, bool(dm_causal), int(dm_ctn), int(dm_G),
           os.environ.get("DM_TCONV_CTL", ""), os.environ.get("DM_TCONV_GDCWAIT", "1"))
    if key not in _CACHE:
        _CACHE[key] = _compile(key)
    if not fake_mode:
        _CACHE[key](accum, output, scale, None, None, None, None, sem, queue)
