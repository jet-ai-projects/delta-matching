"""
tcgen05 MMA issue helpers shared by the sm100 kernels (block-scaled
kind::mxf8f6f4 over all k-steps, inline PTX).
"""
import cutlass  # noqa: F401  (the CuTe DSL preprocessor rewrites helper code into references to the cutlass module)
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm

__all__ = ["descriptor_zero_add", "gemm_ptx_bs"]


def descriptor_zero_add(dz, idx, a_is_tmem):
    """
    PTX that adds the opaque zero operand $idx to the smem descriptor start words (b_lo0, and a0
    when A is an smem descriptor). dz is 0 at run time but not provably 0 to ptxas and varies
    per main-loop iteration, so the per-k-step descriptor words are formed next to each MMA
    instead of being hoisted out of the loop as 64-bit loop invariants (2 uniform registers
    each), which exhausts the MMA warp's uniform registers (UR -> R -> local spills). Same
    descriptors, same MMAs: bitwise.
    """
    if dz is None:
        return ""
    return f"add.u32 b_lo0, b_lo0, ${idx};\n\t" + ("" if a_is_tmem else f"add.u32 a0, a0, ${idx};\n\t")


def gemm_ptx_bs(
    op, acc_tmem_addr, tCrA, tCrB, sA, sB, sfa_base, sfb_base, sfa_map, sfb_map,
    zero_init=False, tA_addr=None, cta_group: int = 1, dz=None, k_range=None,
):
    """
    Block-scaled (kind::mxf8f6f4, UE8M0) MMA over all k-steps of tCrA/tCrB, SS (sA given) or TS (A
    in TMEM at tA_addr). sfa_map / sfb_map: per k-step (column offset added to sfa_base / sfb_base,
    sf byte id 0..3) python tuples. Mirrors flash_attn.cute.blackwell_helpers.gemm_ptx_partial.
    k_range (hd256 dQ split): issue only these k-steps (ascending); zero_init applies to k-step 0
    only.  None = all.
    """
    from flash_attn.cute import mma_sm100_desc as sm100_desc
    from flash_attn.cute.blackwell_helpers import i64_to_i32x2
    is_ts = sA is None
    idesc_plain: int = sm100_desc.mma_op_to_idesc(op)
    nk = cute.size(tCrA.shape[2])
    assert len(sfa_map) == nk and len(sfb_map) == nk

    def desc_base(s, dtype_width, major_mode):
        b = sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, dtype_width, s.layout[0]), s.iterator.type.swizzle_type,
            sm100_desc.Major.K if major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else sm100_desc.Major.MN)
        return i64_to_i32x2(b)

    b_lo, b_hi = desc_base(sB, op.b_dtype.width, op.b_major_mode)
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(nk)]
    start_b = Int32(b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    if is_ts:
        tCrA_layout = cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
        offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(nk)]
        a_arg = Int32(cute.arch.make_warp_uniform(tA_addr))
        a_hi = None
    else:
        a_lo, a_hi = desc_base(sA, op.a_dtype.width, op.a_major_mode)
        offset_a = [cute.crd2idx((0, 0, k), tCrA.layout) for k in range(nk)]
        a_arg = Int32(cute.arch.make_warp_uniform(Int32(a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))))
    pred_str = "p" if isinstance(zero_init, cutlass.Boolean) else "0" if zero_init else "1"
    acc_arg = (Int32(1) - Int32(zero_init)) if isinstance(zero_init, cutlass.Boolean) else Int32(not zero_init)
    input_args = [a_arg.ir_value(), Int32(cute.arch.make_warp_uniform(start_b)).ir_value(), acc_arg.ir_value(),
                  Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
                  Int32(cute.arch.make_warp_uniform(sfa_base)).ir_value(),
                  Int32(cute.arch.make_warp_uniform(sfb_base)).ir_value()]
    if dz is not None:
        input_args.append(Int32(dz).ir_value())
    dzs = descriptor_zero_add(dz, 6, is_ts)
    instr = f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4.block_scale.scale_vec::1X"
    body = ("{\n\t.reg .pred leader_thread;\n\t.reg .pred p;\n\t.reg .b32 tmem_acc, a0, sfa0, sfb0, sfa, sfb;\n\t"
            ".reg .b32 b_lo0, b_lo, b_hi, a_lo, a_hi;\n\t.reg .b64 bdesc, adesc;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            "mov.b32 tmem_acc, $3;\n\tmov.b32 a0, $0;\n\tmov.b32 sfa0, $4;\n\tmov.b32 sfb0, $5;\n\t"
            "mov.b32 b_lo0, $1;\n\t" + dzs +
            f"mov.b32 b_hi, {hex(b_hi)};\n\t"
            + ("" if is_ts else f"mov.b32 a_hi, {hex(a_hi)};\n\t")
            + "setp.ne.b32 p, $2, 0;\n\t")
    for k in (range(nk) if k_range is None else k_range):
        ca, ia = sfa_map[k]
        cb, ib = sfb_map[k]
        idesc = (idesc_plain & ~(0x3 << 4) & ~(0x7 << 29)) | (1 << 23) | ((ia & 3) << 29) | ((ib & 3) << 4)
        pred = pred_str if k == 0 else "1"
        body += (f"add.u32 b_lo, b_lo0, {hex(offset_b[k])};\n\t"
                 "mov.b64 bdesc, {b_lo, b_hi};\n\t"
                 f"add.u32 sfa, sfa0, {hex(ca)};\n\tadd.u32 sfb, sfb0, {hex(cb)};\n\t")
        if is_ts:
            body += f"@leader_thread {instr} [tmem_acc], [a0 + {hex(offset_a[k])}], bdesc, {hex(idesc)}, [sfa], [sfb], {pred};\n\t"
        else:
            body += (f"add.u32 a_lo, a0, {hex(offset_a[k])};\n\tmov.b64 adesc, {{a_lo, a_hi}};\n\t"
                     f"@leader_thread {instr} [tmem_acc], adesc, bdesc, {hex(idesc)}, [sfa], [sfb], {pred};\n\t")
    body += "}\n"
    llvm.inline_asm(None, input_args, body, ",".join(["r"] * len(input_args)),
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
