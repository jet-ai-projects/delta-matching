"""
dQ tail converter of the head_dim 128 deterministic backward: FA4's backward
postprocess (fp32 accumulator * scale -> bf16, same loads, arithmetic and
stores, so dQ is bitwise the serial postprocess), launched with PDL right after
the backward main kernel; each CTA converts a tile as soon as it is final.

Readiness uses the deterministic dQ lock counters the backward maintains: a dQ
tile (b, h, m), slot r of the 2-CTA pair, is final at min(ceil(N / 256), ceil((m
+ 1) * tile_m / 256)) writes when causal, ceil(N / 256) otherwise. Per CTA,
thread 0 takes the next tile from the backward's completion queue, polls its
counters with ld.acquire.gpu (bounded: traps after about 2^24 polls instead of
hanging), then bar.sync, fence.proxy.async.global on every thread (the writers'
bulk reduce-adds are async-proxy writes), FA4's conversion, and a deferred
griddepcontrol.wait. The writers fence.proxy.async.global before every counter
release; the counters are zeroed per call by the backward host path before the
next main kernel, and the converter only reads them.
"""
import inspect

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

import flash_attn.cute.flash_bwd_postprocess as fa_postprocess
from delta_matching.sm100.cute_ptx_sm100 import fence_proxy_async_global, ld_shared_b32, poll_acquire_ge_or_trap, st_shared_b32


__all__ = ["DqConvertHd128", "COMPILE_CACHE", "make_compile_cache", "convert_dq_tail"]

@dsl_user_op
def poll_acquire_nz_or_trap(addr: cutlass.Int64, limit: Int32, *, loc=None, ip=None) -> Int32:
    """
    Spin with ld.acquire.gpu until the int32 at addr is non-zero (trap after `limit`
    polls); returns it.
    """
    return Int32(llvm.inline_asm(
        T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(limit).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .b32 it;\n\tmov.b32 it, 0;\n"
        "DM_PNZ:\n\tld.acquire.gpu.global.b32 $0, [$1];\n\tsetp.ne.s32 p, $0, 0;\n\t@p bra DM_PNZD;\n\t"
        "add.s32 it, it, 1;\n\tsetp.gt.s32 p, it, $2;\n\t@p trap;\n\tnanosleep.u32 256;\n\tbra DM_PNZ;\n"
        "DM_PNZD:\n\t}",
        "=r,l,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


FA4_POSTPROCESS_SOURCE = inspect.getsource(fa_postprocess)


def patch_postprocess_source(src: str) -> str:
    subs = [
        ("class FlashAttentionBackwardPostprocess:", "class DqConvertBase:"),
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
         "        cute.arch.griddepcontrol_launch_dependents()   # the next kernel may launch now\n"
         "        if const_expr(mDmQ is not None ):\n"
         "            m_block, head_idx, batch_idx = self.pick_ready_tile(mDmSem, mDmQ, sdQaccum_flat, tidx)\n"
         "        if work_tile.is_valid_tile:\n"
         "            self.wait_tile_final(mDmSem, m_block, head_idx, batch_idx, tidx, mdQ.shape[1])\n"),
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
    # deferred griddepcontrol.wait at the very end of the kernel, executed by the last CTA of the grid only: the
    # converter grid cannot complete before the backward main kernel has completed and flushed its memory (stream order
    # for the dK / dV postprocess that follows), while no other converter CTA holds its SM slot until then. Every read of
    # the main kernel's dq_accum is ordered by the per-tile ld.acquire of the deterministic counters instead.
    src = src.rstrip("\n") + ("\n        dm_bx, dm_by, dm_bz = cute.arch.block_idx()\n"
                               "        dm_gx, dm_gy, dm_gz = cute.arch.grid_dim()\n"
                               "        if dm_bx == dm_gx - 1 and dm_by == dm_gy - 1 and dm_bz == dm_gz - 1:\n"
                               "            cute.arch.griddepcontrol_wait()\n")
    return src


PATCHED_NAMESPACE = dict(vars(fa_postprocess))
PATCHED_FILENAME = fa_postprocess.__file__.replace(".py", "_dm_tailconv.py")
PATCHED_SOURCE = patch_postprocess_source(FA4_POSTPROCESS_SOURCE)
import linecache  # noqa: E402
linecache.cache[PATCHED_FILENAME] = (len(PATCHED_SOURCE), None, PATCHED_SOURCE.splitlines(True), PATCHED_FILENAME)   # the DSL's AST pass re-reads the source
exec(compile(PATCHED_SOURCE, PATCHED_FILENAME, "exec"), PATCHED_NAMESPACE)
PatchedPostprocess = PATCHED_NAMESPACE["DqConvertBase"]


class DqConvertHd128(PatchedPostprocess):
    """
    FA4's postprocess with the readiness queue and counters; dm_ctn is the key-tile width of a
    counter increment. The dm_ prefix keeps these attributes apart from the FA4 base class's.
    """

    def __init__(self, *a, dm_causal=True, dm_ctn=256, **k):
        super().__init__(*a, **k)
        self.dm_causal, self.dm_ctn = dm_causal, dm_ctn

    @cute.jit
    def pick_ready_tile(self, mDmSem, mDmQ, sbuf, tidx):
        """
        Readiness queue: the CTA with linear block id i converts the i-th tile that became final
        (bwd main appends tile ids in completion order), so CTAs dispatched in launch order never
        wait behind a late tile. Thread 0 polls the entry (ld.acquire.gpu, bounded -> trap) and
        broadcasts it through smem; the det counters are re-checked in wait_tile_final (the queue is
        only a scheduling hint).
        """
        bx, by, bz = cute.arch.block_idx()
        gx, gy, gz = cute.arch.grid_dim()
        pos = bx + gx * (by + gy * bz)
        saddr = sbuf.iterator.toint()
        if tidx == 0:
            v0 = poll_acquire_nz_or_trap(mDmQ.iterator.toint() + Int64(4) * Int64(1 + pos), Int32(1 << 24))
            st_shared_b32(saddr, v0)
        cute.arch.barrier()
        t = ld_shared_b32(saddr) - 1
        cute.arch.barrier()
        mt = cute.size(mDmSem, mode=[2])
        nh = cute.size(mDmSem, mode=[1])
        return t % mt, (t // mt) % nh, t // (mt * nh)

    @cute.jit
    def wait_tile_final(self, mDmSem, m_block, head_idx, batch_idx, tidx, seqlen):
        if const_expr(mDmSem is not None):
            nk = (seqlen + self.dm_ctn - 1) // self.dm_ctn
            fin = nk
            if const_expr(self.dm_causal):
                fin = cutlass.min(nk, ((m_block + 1) * self.tile_m + self.dm_ctn - 1) // self.dm_ctn)
            if tidx == 0:
                nh = cute.size(mDmSem, mode=[1])
                nt = cute.size(mDmSem, mode=[2])
                ns = cute.size(mDmSem, mode=[3])
                base = mDmSem.iterator.toint() + Int64(4) * Int64(((batch_idx * nh + head_idx) * nt + m_block) * ns)
                for r in cutlass.range_constexpr(cute.size(mDmSem, mode=[3])):
                    poll_acquire_ge_or_trap(base + Int64(4 * r), fin, Int32(1 << 24))
            cute.arch.barrier()
            fence_proxy_async_global()


def make_compile_cache():
    from flash_attn.cute import cache_utils
    from .compile_cache_sm100 import fingerprint
    if cache_utils.CUTE_DSL_CACHE_ENABLED:
        return cache_utils.JITPersistentCache(cache_utils.get_cache_path() / "dm_sm100" / fingerprint() / "tconv")
    return cache_utils.JITCache()


COMPILE_CACHE = None


def compile_converter(key):
    """
    Fake-tensor compile (mirrors interface._compile_bwd_postprocess, dense only) +
    the readiness tensor.
    """
    from flash_attn.cute.interface import make_fake_bwd_tensors
    (dtype, hdim, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs, cluster_size, arch, hdim_multiple_of,
     accum_order, dm_causal, dm_ctn) = key
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum, mScaleP = \
        make_fake_bwd_tensors(dtype, has_gqa=True, varlen_q=False, varlen_k=False)
    from quack.compile_utils import make_fake_tensor as make_fake_tensor
    mSem = make_fake_tensor(Int32, (cute.sym_int(), cute.sym_int(), cute.sym_int(), 2), divisibility=1)
    mQ_ = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1)
    conv = DqConvertHd128(dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs=use_2cta_instrs,
                      cluster_size=cluster_size, hdim_multiple_of=hdim_multiple_of, accum_order=accum_order,
                      pack_factor=1, dm_causal=dm_causal, dm_ctn=dm_ctn)
    return cute.compile(conv, mdQaccum, mdQ, cutlass.Float32(0.0), None, None, None, None, mSem, mQ_,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")


def convert_dq_tail(sem, accum, output, scale, cu_seqlens, seqused, arch, dtype, hdim, block_size, num_threads,
                  atom_layout, swap_ab, use_2cta_instrs=False, cluster_size=1, cu_total_m_blocks=None,
                  sink_tensors=None, hdim_multiple_of=32, accum_order=None, pack_factor=1, *, fake_mode,
                  dm_causal, dm_ctn, queue=None):
    """
    Drop-in for interface._bwd_postprocess_convert with the readiness semaphore and the
    kind / counts.
    """
    global COMPILE_CACHE
    assert cu_seqlens is None and seqused is None and sink_tensors is None and pack_factor == 1, "tconv: dense only"
    assert cu_total_m_blocks is None
    if COMPILE_CACHE is None:
        COMPILE_CACHE = make_compile_cache()
    key = (dtype, hdim, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs, cluster_size, arch,
           hdim_multiple_of, accum_order, bool(dm_causal), int(dm_ctn))
    if key not in COMPILE_CACHE:
        COMPILE_CACHE[key] = compile_converter(key)
    if not fake_mode:
        COMPILE_CACHE[key](accum, output, scale, None, None, None, None, sem, queue)
