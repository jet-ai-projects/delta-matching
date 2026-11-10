"""CuTe DSL kernels of the sm100-mx producers.

Bitwise equal to the former Triton kernels of quant.py (_prod_a / _prod_b / _quant_do_delta / _quant_rows / _quant_qkv /
_kmean_partial).  Quantization (amax, E8M0 scale, e4m3 codes) is elementwise or an exact max, so it does not depend on the
thread mapping.  The fp32 reductions (K mean partials, K mean fold, stale delta) reproduce Triton's trees literally (same
grouping, same fma fusion, same shuffle order); every fp32 op of a reduction is inline PTX with an explicit .rn, so neither
LLVM nor ptxas can contract or reassociate it.  SIMT kernels: no TMA / tcgen05 / pipelines (no dm_utils fence wrappers).
D (128 or 256) is a compile-time parameter.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda

NT = 64            # threads per CTA of ProdA (= Triton num_warps 2: its K-mean tree is defined on 2 warps)
NT4 = 128          # Tile at D=256: 4 warps on the same 32-row tile (half the rows per thread -> half the live registers;
                   # codes / scales / bytes are order-free, each row's delta tree stays within one warp's lanes)
TR = 32            # rows per quant tile (one E8M0 block)


def _asm(ret, args, s, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(ret, args, s, cons, has_side_effects=side, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def _v(x, ty, loc, ip):
    return ty(x).ir_value(loc=loc, ip=ip)


# ---------------------------------------------------------------------------------------------------------- PTX helpers
@dsl_user_op
def ld_v4(addr: Int64, *, loc=None, ip=None):
    """16 B global load -> 4 x b32."""
    st = _asm(llvm.StructType.get_literal([T.i32()] * 4), [_v(addr, Int64, loc, ip)],
              "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc=loc, ip=ip)
    return [Int32(llvm.extractvalue(T.i32(), st, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_v4f(addr: Int64, *, loc=None, ip=None):
    st = _asm(llvm.StructType.get_literal([T.f32()] * 4), [_v(addr, Int64, loc, ip)],
              "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l", loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), st, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(addr, Int64, loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def ld_bf16f(addr: Int64, *, loc=None, ip=None) -> Float32:
    """One bf16 (2 B aligned) -> f32 (cvt.f32.bf16, as Triton)."""
    return Float32(_asm(T.f32(), [_v(addr, Int64, loc, ip)],
                        "{\n\t.reg .b16 h;\n\tld.global.b16 h, [$1];\n\tcvt.f32.bf16 $0, h;\n\t}", "=f,l", loc=loc, ip=ip))


def _bf2_op(tag):
    @dsl_user_op
    def f(w: Int32, *, loc=None, ip=None):
        st = _asm(llvm.StructType.get_literal([T.f32()] * 2), [_v(w, Int32, loc, ip)],
                  "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $2;\n\tcvt.f32.bf16 $0, l;\n\tcvt.f32.bf16 $1, h;\n\t}"
                  + (f" // {tag}" if tag else ""), "=f,=f,r", loc=loc, ip=ip)
        return [Float32(llvm.extractvalue(T.f32(), st, [i], loc=loc, ip=ip)) for i in range(2)]
    return f


_BF2 = {}


def bf2(w, tag=""):
    """b32 holding 2 bf16 (lo = lower address) -> (f32 lo, f32 hi) via cvt.f32.bf16 (as Triton).  `tag` makes the asm text
    unique per conversion pass so LLVM cannot CSE a re-conversion with an earlier one (that would keep every converted row
    live and cost occupancy); the PTX instructions are identical."""
    f = _BF2.get(tag)
    if f is None:
        f = _BF2[tag] = _bf2_op(tag)
    return f(w)


def _f2(op, cons="=f,f,f"):
    @dsl_user_op
    def f(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
        return Float32(_asm(T.f32(), [_v(a, Float32, loc, ip), _v(b, Float32, loc, ip)], op + " $0, $1, $2;", cons,
                            loc=loc, ip=ip))
    return f


fadd = _f2("add.rn.f32")
fsub = _f2("sub.rn.f32")
fmul = _f2("mul.rn.f32")
fmax = _f2("max.f32")
fdivfull = _f2("div.full.f32")


# packed f32x2 (sm_100 FADD2 / FMUL2 / FFMA2): each lane is rounded exactly as the scalar .rn op (same values, half the
# instructions; Triton's reductions are packed the same way)
def _pk2(op, nin):
    @dsl_user_op
    def f(*args, loc=None, ip=None):
        ins = []
        for a in args:
            ins += [_v(a[0], Float32, loc, ip), _v(a[1], Float32, loc, ip)]
        regs = ", ".join(f"x{i}" for i in range(nin))
        body = "{\n\t.reg .b64 " + regs + ", z;\n\t"
        for i in range(nin):
            body += f"mov.b64 x{i}, {{${2 + 2 * i}, ${3 + 2 * i}}};\n\t"
        body += f"{op} z, " + regs + ";\n\tmov.b64 {$0, $1}, z;\n\t}"
        st = _asm(llvm.StructType.get_literal([T.f32()] * 2), ins, body, "=f,=f," + ",".join(["f"] * (2 * nin)),
                  loc=loc, ip=ip)
        return (Float32(llvm.extractvalue(T.f32(), st, [0], loc=loc, ip=ip)),
                Float32(llvm.extractvalue(T.f32(), st, [1], loc=loc, ip=ip)))
    return f


add2 = _pk2("add.rn.f32x2", 2)
sub2 = _pk2("sub.rn.f32x2", 2)
mul2 = _pk2("mul.rn.f32x2", 2)
fma2 = _pk2("fma.rn.f32x2", 3)


def tree2(v):
    """tree() on (lo, hi) pairs, packed."""
    while len(v) > 1:
        v = [add2(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)]
    return v[0]


@dsl_user_op
def ffma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(x, Float32, loc, ip) for x in (a, b, c)], "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                        loc=loc, ip=ip))


@dsl_user_op
def fabs(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(a, Float32, loc, ip)], "abs.f32 $0, $1;", "=f,f", loc=loc, ip=ip))


@dsl_user_op
def i2f(n: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(n, Int32, loc, ip)], "cvt.rn.f32.s32 $0, $1;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def shfl_bfly(x: Float32, off: int, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(x, Float32, loc, ip)], f"shfl.sync.bfly.b32 $0, $1, {off}, 31, -1;", "=f,f",
                        side=True, loc=loc, ip=ip))


@dsl_user_op
def e8m0(amax: Float32, *, loc=None, ip=None) -> Int32:
    """eb = ((bits >> 23) & 0xFF) - 8 + ((bits & 0x7FFFFF) > 0x600000), clamped to [1, 254]; 127 unless amax > 0."""
    return Int32(_asm(T.i32(), [_v(amax, Float32, loc, ip)],
                      "{\n\t.reg .b32 a, m, e;\n\t.reg .pred p, q;\n\t"
                      "mov.b32 a, $1;\n\tshr.u32 e, a, 23;\n\tand.b32 e, e, 255;\n\tsub.s32 e, e, 8;\n\t"
                      "and.b32 m, a, 8388607;\n\tsetp.gt.u32 p, m, 6291456;\n\tselp.s32 m, 1, 0, p;\n\tadd.s32 e, e, m;\n\t"
                      "max.s32 e, e, 1;\n\tmin.s32 e, e, 254;\n\t"
                      "setp.gt.f32 q, $1, 0f00000000;\n\tselp.s32 $0, e, 127, q;\n\t}",
                      "=r,f", loc=loc, ip=ip))


@dsl_user_op
def shl23f(e: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(e, Int32, loc, ip)], "shl.b32 $0, $1, 23;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def e4m3x4(a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None) -> Int32:
    """4 fp32 -> 4 e4m3 bytes (byte 0 = a), cvt.rn.satfinite.e4m3x2.f32 (as Triton)."""
    return Int32(_asm(T.i32(), [_v(x, Float32, loc, ip) for x in (a, b, c, d)],
                      "{\n\t.reg .b16 l, h;\n\tcvt.rn.satfinite.e4m3x2.f32 l, $2, $1;\n\t"
                      "cvt.rn.satfinite.e4m3x2.f32 h, $4, $3;\n\tmov.b32 $0, {l, h};\n\t}", "=r,f,f,f,f", loc=loc, ip=ip))


@dsl_user_op
def st_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip)] + [_v(x, Int32, loc, ip) for x in (w0, w1, w2, w3)],
         "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_b32(addr: Int64, w: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip), _v(w, Int32, loc, ip)], "st.global.b32 [$0], $1;", "l,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def st_f32_if(addr: Int64, v: Float32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip), _v(v, Float32, loc, ip), _v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.f32 [$0], $1;\n\t}", "l,f,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def st_v4f_if(addr: Int64, a: Float32, b: Float32, c: Float32, d: Float32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip)] + [_v(x, Float32, loc, ip) for x in (a, b, c, d)] + [_v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $5, 0;\n\t@p st.global.v4.f32 [$0], {$1, $2, $3, $4};\n\t}",
         "l,f,f,f,f,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_u8_if(addr: Int64, v: Int32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip), _v(v, Int32, loc, ip), _v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.s32 p, $2, 0;\n\tcvt.u16.u32 h, $1;\n\t@p st.global.b8 [$0], h;\n\t}",
         "l,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def sts_f32_if(addr: Int32, v: Float32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int32, loc, ip), _v(v, Float32, loc, ip), _v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.shared.f32 [$0], $1;\n\t}", "r,f,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def lds_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(addr, Int32, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", side=True,
                        loc=loc, ip=ip))


@dsl_user_op
def bar_sync(*, loc=None, ip=None):
    _asm(None, [], "bar.sync 0;", "", side=True, loc=loc, ip=ip)


@dsl_user_op
def bar_named(bid: Int32, n: int, *, loc=None, ip=None):
    """bar.sync bid, n (a warp pair's private barrier)."""
    _asm(None, [_v(bid, Int32, loc, ip)], f"bar.sync $0, {n};", "r", side=True, loc=loc, ip=ip)


@dsl_user_op
def smem_base(nbytes: int, *, loc=None, ip=None) -> Int32:
    """32-bit shared address of a static smem buffer (CTA-private)."""
    from cutlass.utils import SmemAllocator
    sm = SmemAllocator()
    t = sm.allocate_tensor(Float32, cute.make_layout(nbytes // 4), byte_alignment=16)
    return Int32(t.iterator.toint())


# ------------------------------------------------------------------------------------------------------- trace helpers
def tree(v):
    """Balanced pairwise tree, level by level: ((v0+v1)+(v2+v3))+... (len a power of 2)."""
    while len(v) > 1:
        v = [fadd(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)]
    return v[0]


def load_row16(addr, vec):
    """16 consecutive bf16 at byte address addr -> 16 f32."""
    if vec:
        w = ld_v4(addr) + ld_v4(addr + Int64(16))
        out = []
        for x in w:
            out += bf2(x)
        return out
    return [ld_bf16f(addr + Int64(2 * j)) for j in range(16)]


def load_row16_f32(addr, vec):
    if vec:
        out = []
        for k in range(4):
            out += ld_v4f(addr + Int64(16 * k))
        return out
    return [ld_f32(addr + Int64(4 * j)) for j in range(16)]


def load_row8(addr, vec):
    if vec:
        out = []
        for x in ld_v4(addr):
            out += bf2(x)
        return out
    return [ld_bf16f(addr + Int64(2 * j)) for j in range(8)]


def row_delta(x, o):
    """Triton's in-thread tree of sum_j x_j * o_j over 16 columns, packed over column pairs exactly as
    its PTX: lane lo = even columns, hi = odd: a_k = fma2(x(4k,4k+1), o(4k,4k+1), x(4k+2,4k+3)*o(4k+2,4k+3)),
    (a0 + a1) + (a2 + a3), then lo + hi."""
    X = [(x[2 * i], x[2 * i + 1]) for i in range(8)]
    O = [(o[2 * i], o[2 * i + 1]) for i in range(8)]
    a = [fma2(X[2 * k], O[2 * k], mul2(X[2 * k + 1], O[2 * k + 1])) for k in range(4)]
    e = add2(add2(a[0], a[1]), add2(a[2], a[3]))
    return fadd(e[0], e[1])


class TileGeo:
    """32-row x D quant tile on 64 threads, Triton layout spt[1,16] tpw[32/(D/16), D/16] wpc[2,1]."""

    def __init__(self, D, tid, nw=2):
        self.D, self.nw = D, nw
        self.tpw1 = D // 16
        self.tpw0 = 32 // self.tpw1
        self.rpt = self.tpw0 * nw
        self.reps = TR // self.rpt
        self.w = tid // 32
        self.lane = tid % 32
        self.l0 = self.lane // self.tpw1
        self.l1 = self.lane % self.tpw1
        self.c0 = self.l1 * 16

    def row(self, rep):
        return self.w * self.tpw0 + self.l0 + rep * self.rpt


def maxtree(a):
    while len(a) > 1:                     # max is exact and order-free (maxNum): a tree only for ILP
        a = [fmax(a[2 * i], a[2 * i + 1]) for i in range(len(a) // 2)]
    return a[0]


def quant_tile(g: TileGeo, row, red, n_tok, h_out, nh_out, bh, row0, mY, mS, mWB, w_off, b, bar=None):
    """row(rep) -> 16 f32 (centred / prescaled) of this thread's row `rep`; called twice per row (amax pass, code pass)
    so only the packed raw loads stay live (register pressure = occupancy; the recomputation is the same ops, same
    values).  Stores codes, the scale and (w_off not None) the UE8M0 byte; returns the scale.
    red: smem address of 2 floats (cross-warp amax)."""
    m = maxtree([maxtree([fabs(v) for v in row(rep, "a")]) for rep in range(g.reps)])
    for off in (16, 8, 4, 2, 1):
        m = fmax(m, shfl_bfly(m, off))
    sts_f32_if(red + g.w * 4, m, Int32(g.lane == 0))
    (bar_sync if bar is None else bar)()
    m = maxtree([lds_f32(red + 4 * i) for i in range(g.nw)])
    eb = e8m0(m)
    scale = shl23f(eb)
    inv = shl23f(Int32(254) - eb)
    blk = row0 // TR
    y0 = mY + ((Int64(b) * Int64(n_tok) + Int64(row0 + g.row(0))) * Int64(nh_out) + Int64(h_out)) * Int64(g.D) + Int64(g.c0)
    ystep = Int64(nh_out) * (g.rpt * g.D)
    for rep in range(g.reps):
        r = row(rep, "c")
        y = []
        for i in range(8):
            y += list(mul2((r[2 * i], r[2 * i + 1]), (inv, inv)))
        wds = [e4m3x4(y[4 * k], y[4 * k + 1], y[4 * k + 2], y[4 * k + 3]) for k in range(4)]
        st_v4(y0, *wds)
        y0 = y0 + ystep
    first = Int32(g.w == 0) * Int32(g.lane == 0)
    st_f32_if(mS + (Int64(bh) * Int64(n_tok // TR) + Int64(blk)) * 4, scale, first)
    if w_off is not None:
        baddr = mWB + (Int64(bh) * Int64(n_tok // 64) + Int64(2 * (blk // 4) + w_off)) * 4 + Int64(blk % 4)
        st_u8_if(baddr, eb, first)
    return scale


def load_tile_raw(g: TileGeo, mX, b, h, row0, sb, sn, sh, vec, f32=False):
    """Issue every load of the thread's rows (all in flight before any use). Returns per rep: 8 b32 words (bf16 pairs,
    16 B-vector path) or 16 f32 (scalar / fp32 path)."""
    es = 4 if f32 else 2
    a0 = mX + (Int64(b) * Int64(sb) + Int64(h) * Int64(sh) + Int64(row0 + g.row(0)) * Int64(sn) + Int64(g.c0)) * es
    step = Int64(sn) * (g.rpt * es)            # rows of rep r = row(0) + r * rpt: strength-reduced addresses
    addrs = [a0]
    for rep in range(1, g.reps):
        addrs.append(addrs[-1] + step)
    if f32 or not vec:
        return [load_row16_f32(a, vec) if f32 else load_row16(a, vec) for a in addrs]
    return [ld_v4(a) + ld_v4(a + Int64(16)) for a in addrs]


def conv_row(r, packed, tag=""):
    """Raw row -> 16 f32 (cvt.f32.bf16 of each half; exact)."""
    if not packed:
        return r
    out = []
    for x in r:
        out += bf2(x, tag)
    return out


def row_fn(raw, packed, mu=None, pre=None, recon=True):
    """rep -> 16 f32 = ((x - mu) * pre) with the same ops as Triton (sub.rn, mul.rn).  recon=False: every pass uses the
    same asm text, so LLVM CSEs the conversions (converted rows stay live: fewer instructions, more registers)."""
    def f(rep, tag=""):
        x = conv_row(raw[rep], packed, tag if recon else "")
        if mu is not None:
            x = [v for i in range(8) for v in sub2((x[2 * i], x[2 * i + 1]), (mu[2 * i], mu[2 * i + 1]))]
        if pre is not None:
            x = [v for i in range(8) for v in mul2((x[2 * i], x[2 * i + 1]), (pre, pre))]
        return x
    return f


class ChunkGeo:
    """K-mean partial chunk: `rows` rows x D columns on nw warps.
    Lanes l1 own 8 columns each (D/8 lanes); G = groups of rows = 32*nw / (D/8); a group g = w * tpw0 + l0 holds
    reps = rows / G rows: Triton-ALIGNED row r = rep * G + g, UNALIGNED (one thread per column, contiguous) r = g * reps + rep."""

    def __init__(self, D, nw, rows, tid, aligned):
        self.D, self.nw, self.rows, self.aligned = D, nw, rows, aligned
        self.tpw1 = D // 8
        self.tpw0 = 32 // self.tpw1
        self.G = self.tpw0 * nw
        self.reps = rows // self.G
        self.w = tid // 32
        self.lane = tid % 32
        self.l0 = self.lane // self.tpw1
        self.l1 = self.lane % self.tpw1
        self.c0 = self.l1 * 8
        self.g = self.w * self.tpw0 + self.l0

    def row(self, rep):
        return rep * self.G + self.g if self.aligned else self.g * self.reps + rep


def chunk_colsum(cg: ChunkGeo, mK, b, h, row0, sb, sn, sh, vec, red, warp_order, bar=None):
    """Column sums of one chunk in Triton's tree; returns 8 f32 (valid in warp 0, lanes with l0 == 0).
    red: smem address of nw * D floats."""
    a0 = mK + (Int64(b) * Int64(sb) + Int64(h) * Int64(sh) + Int64(cg.c0) + Int64(row0 + cg.row(0)) * Int64(sn)) * 2
    step = Int64(sn) * (2 * (cg.G if cg.aligned else 1))      # row(rep + 1) - row(rep)
    addrs = [a0]
    for rep in range(1, cg.reps):
        addrs.append(addrs[-1] + step)
    if vec:
        # groups of <= 16 reps (16 B loads in flight together); a balanced tree over 2^k leaves = the tree of its two
        # halves' trees, so per-group trees combined by tree() give exactly T_reps
        gs = min(cg.reps, 16)
        parts = []
        for g0 in range(0, cg.reps, gs):
            raw = [ld_v4(a) for a in addrs[g0:g0 + gs]]
            sg = []
            for wj in range(4):                  # one column pair at a time: only the packed words stay live
                sg += list(tree2([tuple(bf2(raw[r][wj])) for r in range(gs)]))
            parts.append(sg)
        s = []
        for j in range(4):
            s += list(tree2([(pt[2 * j], pt[2 * j + 1]) for pt in parts]))
    else:
        vals = [load_row8(a, vec) for a in addrs]
        s = [tree([vals[rep][j] for rep in range(cg.reps)]) for j in range(8)]
    if cg.tpw0 == 2:
        s = [fadd(v, shfl_bfly(v, 16)) for v in s]
    if cg.nw == 1:
        return s
    # cross-warp: every warp's lanes with l0 == 0 publish their 8 column sums, warp 0 combines in warp_order
    pub = Int32(cg.l0 == 0)
    sync = bar_sync if bar is None else bar
    for j in range(8):
        sts_f32_if(red + (cg.w * cg.D + cg.c0 + j) * 4, s[j], pub)
    sync()
    out = []
    for j in range(8):
        ws = [lds_f32(red + (wi * cg.D + cg.c0 + j) * 4) for wi in range(cg.nw)]
        out.append(warp_order(ws))
    sync()
    return out


def warps2(ws):
    return fadd(ws[0], ws[1])


# ---------------------------------------------------------------------------------------------------------------- kernels
class ProdA:
    """= _prod_a_kernel (used at D=256): 1-D grid [K-mean partials (ROWS rows) | Q tile groups | V tile groups]; CTA = nw/2
    warp pairs (tile roles: one 32-row tile per pair; partial role: one 64-row chunk stream per pair, chunk totals folded
    in chunk order by pair 0).  headfold: the CTA completing a head's last partial also writes the mean."""

    def __init__(self, D, vq, vk, vv, kal, nw=4, headfold=False):
        self.headfold = headfold
        assert D in (128, 256) and nw in (2, 4, 8)
        self.D, self.vq, self.vk, self.vv, self.kal, self.nw = D, vq, vk, vv, kal, nw

    @cute.jit
    def __call__(self, mQ: Int64, mK: Int64, mV: Int64, mQ8: Int64, mV8: Int64,
                 mQS: Int64, mVS: Int64, mWB: Int64, mP: Int64, mCnt: Int64, mM: Int64,
                 N: Int32, H: Int32, Hk: Int32, B: Int32, sqb: Int32, sqn: Int32, sqh: Int32, skb: Int32, skn: Int32,
                 skh: Int32, svb: Int32, svn: Int32, svh: Int32, sm: Float32, ROWS: Int32, stream: cuda.CUstream):
        ntl = N // TR
        P = self.nw // 2
        total = (B * H * ntl) // P + (B * Hk * ntl) // P + B * Hk * (N // ROWS)
        self.kernel(mQ, mK, mV, mQ8, mV8, mQS, mVS, mWB, mP, mCnt, mM, N, H, Hk, B, sqb, sqn, sqh, skb, skn, skh, svb, svn, svh,
                    sm, ROWS).launch(grid=[total, 1, 1], block=[32 * self.nw, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: Int64, mK: Int64, mV: Int64, mQ8: Int64, mV8: Int64,
               mQS: Int64, mVS: Int64, mWB: Int64, mP: Int64, mCnt: Int64, mM: Int64,
               N: Int32, H: Int32, Hk: Int32, B: Int32, sqb: Int32, sqn: Int32, sqh: Int32, skb: Int32, skn: Int32,
               skh: Int32, svb: Int32, svn: Int32, svh: Int32, sm: Float32, ROWS: Int32):
        # CTA = P = nw/2 warp pairs.  Tile roles: each pair quantizes its own 32-row tile (per-thread work = the 2-warp
        # design), pair-private smem + named barrier.  Partial role: P chunk streams (below).
        D = self.D
        P = self.nw // 2
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        red = smem_base(self.nw * D * 4)
        cbuf = smem_base(16 * D * 4) if const_expr(self.nw > 2) else red
        pp = tid // 64
        ntl = N // TR
        nq = (B * H * ntl) // P
        nv = (B * Hk * ntl) // P
        S = N // ROWS
        npart = B * Hk * S
        pid = bid - npart
        if bid < npart:
            pid = pid + nq + nv + npart
        if pid < nq:
            g = TileGeo(D, tid % 64, 2)
            t = pid * P + pp
            bh = t // ntl
            b = bh // H
            h = bh % H
            row0 = (t % ntl) * TR
            raw = load_tile_raw(g, mQ, b, h, row0, sqb, sqn, sqh, self.vq)
            quant_tile(g, row_fn(raw, self.vq, None, sm), red + pp * 16, N, h, H, bh, row0, mQ8, mQS,
                       None, None, b, bar=pair_bar(pp, P))
        elif pid < nq + nv:
            g = TileGeo(D, tid % 64, 2)
            t = (pid - nq) * P + pp
            bh = t // ntl
            b = bh // Hk
            h = bh % Hk
            row0 = (t % ntl) * TR
            raw = load_tile_raw(g, mV, b, h, row0, svb, svn, svh, self.vv)
            quant_tile(g, row_fn(raw, self.vv), red + pp * 16, N, h, Hk, bh, row0, mV8, mVS,
                       mWB, 1, b, bar=pair_bar(pp, P))
        else:
            p = pid - nq - nv
            bh = p // S
            s = p % S
            b = bh // Hk
            h = bh % Hk
            if const_expr(self.nw > 2):
                # P = nw/2 warp pairs; pair pp reduces chunks c = P i + pp with the exact 2-warp tree (own smem region,
                # own named barrier), publishes the chunk total to cbuf[c]; then pair 0 folds cbuf[0..NCH) in chunk order:
                # acc = ((0 + c0) + c1) + ...  -> the same sum as the sequential loop.  NCH = ROWS / 64 <= 16.
                cg = ChunkGeo(D, 2, 64, tid % 64, self.kal)
                pub = Int32(cg.w == 0) * Int32(cg.l0 == 0)
                nch = ROWS // 64
                for ci in cutlass.range((nch + P - 1) // P, unroll=1):
                    c = ci * P + pp
                    if c < nch:
                        tot = chunk_colsum(cg, mK, b, h, s * ROWS + c * 64, skb, skn, skh, self.vk,
                                           red + pp * (2 * D * 4), warps2, bar=pair_bar(pp, P))
                        for j in cutlass.range_constexpr(8):
                            sts_f32_if(cbuf + (c * D + cg.c0 + j) * 4, tot[j], pub)
                bar_sync()
                acc = cute.make_rmem_tensor(8, Float32)
                for j in cutlass.range_constexpr(8):
                    acc[j] = Float32(0.0)
                for c in cutlass.range(nch, unroll=1):
                    for j in cutlass.range_constexpr(8):
                        acc[j] = fadd(acc[j], lds_f32(cbuf + (c * D + cg.c0 + j) * 4))
                pa = mP + ((Int64(bh) * Int64(S) + Int64(s)) * Int64(D) + Int64(cg.c0)) * 4
                ok = pub * Int32(pp == 0)
                st_v4f_if(pa, acc[0], acc[1], acc[2], acc[3], ok)
                st_v4f_if(pa + Int64(16), acc[4], acc[5], acc[6], acc[7], ok)
                if const_expr(self.headfold):
                    # the CTA completing head bh's last partial folds the mean (R2 order, div.full); counters zero at rest
                    fence_gpu()
                    bar_sync()
                    if tid == 0:
                        old2 = atom_add_gpu(mCnt + Int64(bh) * 4, Int32(1)) + Int32(1)
                        sts_i32(cbuf, old2, Int32(1))
                    bar_sync()
                    if lds_i32(cbuf) == S:
                        fence_gpu()
                        if tid < 64:
                            fold_head(mP, mM, bh, S, D, tid, N)
                        st_b32_if(mCnt + Int64(bh) * 4, Int32(0), Int32(tid == 0))
            else:
                cg = ChunkGeo(D, 2, 64, tid, self.kal)
                acc = cute.make_rmem_tensor(8, Float32)
                for j in cutlass.range_constexpr(8):
                    acc[j] = Float32(0.0)
                for ci in cutlass.range(ROWS // 64, unroll=1):
                    tot = chunk_colsum(cg, mK, b, h, s * ROWS + ci * 64, skb, skn, skh, self.vk, red, warps2)
                    for j in cutlass.range_constexpr(8):
                        acc[j] = fadd(acc[j], tot[j])
                pa = mP + ((Int64(bh) * Int64(S) + Int64(s)) * Int64(D) + Int64(cg.c0)) * 4
                ok = Int32(cg.w == 0) * Int32(cg.l0 == 0)
                st_v4f_if(pa, acc[0], acc[1], acc[2], acc[3], ok)
                st_v4f_if(pa + Int64(16), acc[4], acc[5], acc[6], acc[7], ok)


def pair_bar(pp, P):
    """Barrier of warp pair pp: CTA-wide bar 0 when the CTA is one pair, else the pair's named barrier 1 + pp."""
    if P == 1:
        return bar_sync
    return lambda: bar_named(Int32(1) + pp, 64)


class Tile:
    """Generic 32-row quant tile, grid (N/32, B*H): = _quant_do_delta (delta None / "stale") and, with a mean / prescale /
    UE8M0 byte, = _quant_rows and the per-tensor tiles of _quant_qkv."""

    def __init__(self, D, vx, has_mean=False, has_pre=False, w_off=None, delta=None, vo=False, xf32=False, nw=2, tp=1,
                 recon=True):
        """nw warps per 32-row tile, tp tiles per CTA (each tile its own warps, smem slot and named barrier)."""
        assert D in (128, 256) and delta in (None, "stale") and nw in (2, 4) and tp in (1, 2, 4)
        self.nw, self.tp, self.recon = nw, tp, recon
        self.D, self.vx, self.has_mean, self.has_pre, self.w_off, self.delta, self.vo, self.xf32 = (
            D, vx, has_mean, has_pre, w_off, delta, vo, xf32)

    @cute.jit
    def __call__(self, mX: Int64, mO: Int64, mMean: Int64, mY: Int64, mS: Int64,
                 mDL: Int64, mWB: Int64, N: Int32, H: Int32, B: Int32, sxb: Int32, sxn: Int32, sxh: Int32,
                 sob: Int32, son: Int32, soh: Int32, pre: Float32, stream: cuda.CUstream):
        self.kernel(mX, mO, mMean, mY, mS, mDL, mWB, N, H, sxb, sxn, sxh, sob, son, soh, pre).launch(
            grid=[N // (TR * self.tp), B * H, 1], block=[32 * self.nw * self.tp, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: Int64, mO: Int64, mMean: Int64, mY: Int64, mS: Int64,
               mDL: Int64, mWB: Int64, N: Int32, H: Int32, sxb: Int32, sxn: Int32, sxh: Int32,
               sob: Int32, son: Int32, soh: Int32, pre: Float32):
        D = self.D
        tid, _, _ = cute.arch.thread_idx()
        pid_n, bh, _ = cute.arch.block_idx()
        red = smem_base(4 * 4 * self.tp)
        tt = tid // (32 * self.nw)
        g = TileGeo(D, tid % (32 * self.nw), self.nw)
        b = bh // H
        h = bh % H
        row0 = (pid_n * self.tp + tt) * TR
        bar = bar_sync if const_expr(self.tp == 1) else (lambda: bar_named(Int32(1) + tt, 32 * self.nw))
        tile_body(g, self, mX, mO, mMean, mY, mS, mDL, mWB, N, H, b, h, bh, row0,
                  sxb, sxn, sxh, sob, son, soh, pre, red + tt * 16, bar)


def tile_body(g, cfg, mX, mO, mMean, mY, mS, mDL, mWB, N, H, b, h, bh, row0, sxb, sxn, sxh, sob, son, soh, pre, red, bar):
    D = cfg.D
    raw = load_tile_raw(g, mX, b, h, row0, sxb, sxn, sxh, cfg.vx, cfg.xf32)
    rawo = load_tile_raw(g, mO, b, h, row0, sob, son, soh, cfg.vo) if cfg.delta is not None else None
    mu = load_mean(mMean, bh, D, g) if cfg.has_mean else None
    row = row_fn(raw, cfg.vx and not cfg.xf32, mu, pre if cfg.has_pre else None, cfg.recon)
    if cfg.delta is not None:     # delta first: O's words die before the quantization passes (registers); independent values
        tile_delta(g, row, row_fn(rawo, cfg.vo, recon=cfg.recon), mDL, bh, N, row0)
    quant_tile(g, row, red, N, h, H, bh, row0, mY, mS, mWB if cfg.w_off is not None else None, cfg.w_off, b, bar=bar)


def load_mean(ma, bh, D, g):
    ma = ma + (Int64(bh) * Int64(D) + Int64(g.c0)) * 4
    return [ld_f32(ma + Int64(4 * j)) for j in range(16)]


def tile_delta(g, xrow, orow, dl, bh, N, row0):
    """Stale delta per row: R3 in-thread tree, then shfl.bfly over the D/16 column lanes (largest offset first);
    lane l1 == 0 stores."""
    for rep in range(g.reps):
        t = row_delta(xrow(rep, "d"), orow(rep, "o"))
        off = g.tpw1 // 2
        while off >= 1:
            t = fadd(t, shfl_bfly(t, off))
            off //= 2
        st_f32_if(dl + (Int64(bh) * Int64(N) + Int64(row0 + g.row(rep))) * 4, t, Int32(g.l1 == 0))   # 1 store / row


def warps4_02_13(ws):
    """Triton's 4-warp smem + shfl.bfly 2, 1 combine: (w0 + w2) + (w1 + w3)."""
    return fadd(fadd(ws[0], ws[2]), fadd(ws[1], ws[3]))


class KMeanPartial:
    """= _kmean_partial_kernel (kmean_fast; not on the shipped path): grid (B*Hk, N/ROWS), 128 threads, 32-row chunks."""

    def __init__(self, D, vk, kal):
        self.D, self.vk, self.kal = D, vk, kal

    @cute.jit
    def __call__(self, mK: Int64, mP: Int64, N: Int32, Hk: Int32, B: Int32, skb: Int32, skn: Int32,
                 skh: Int32, ROWS: Int32, stream: cuda.CUstream):
        self.kernel(mK, mP, N, Hk, skb, skn, skh, ROWS).launch(grid=[B * Hk, N // ROWS, 1], block=[128, 1, 1],
                                                               stream=stream)

    @cute.kernel
    def kernel(self, mK: Int64, mP: Int64, N: Int32, Hk: Int32, skb: Int32, skn: Int32, skh: Int32,
               ROWS: Int32):
        D = self.D
        tid, _, _ = cute.arch.thread_idx()
        bh, s, _ = cute.arch.block_idx()
        red = smem_base(4 * D * 4)
        S = N // ROWS
        b = bh // Hk
        h = bh % Hk
        if const_expr(self.kal):
            cg = ChunkGeo(D, 4, 32, tid, True)
            acc = cute.make_rmem_tensor(8, Float32)
            for j in cutlass.range_constexpr(8):
                acc[j] = Float32(0.0)
            for ci in cutlass.range(ROWS // 32, unroll=1):
                tot = chunk_colsum(cg, mK, b, h, s * ROWS + ci * 32, skb, skn, skh, self.vk, red,
                                   warps4_02_13)
                for j in cutlass.range_constexpr(8):
                    acc[j] = fadd(acc[j], tot[j])
            pa = mP + ((Int64(bh) * Int64(S) + Int64(s)) * Int64(D) + Int64(cg.c0)) * 4
            ok = Int32(cg.w == 0) * Int32(cg.l0 == 0)
            st_v4f_if(pa, acc[0], acc[1], acc[2], acc[3], ok)
            st_v4f_if(pa + Int64(16), acc[4], acc[5], acc[6], acc[7], ok)
        else:
            # Triton UNALIGNED layout spt[1,1] tpw[1,32] wpc[1,4]: whole columns per thread (tid + 128 j)
            acc = cute.make_rmem_tensor(D // 128, Float32)
            for j in cutlass.range_constexpr(D // 128):
                acc[j] = Float32(0.0)
            for ci in cutlass.range(ROWS // 32, unroll=1):
                tot = kmp_u_colsum(mK, b, h, s * ROWS + ci * 32, skb, skn, skh, tid, D)
                for j in cutlass.range_constexpr(D // 128):
                    acc[j] = fadd(acc[j], tot[j])
            pa = mP + (Int64(bh) * Int64(S) + Int64(s)) * Int64(D) * 4
            for j in cutlass.range_constexpr(D // 128):
                st_f32_if(pa + Int64(tid + 128 * j) * 4, acc[j], Int32(1))


def kmp_u_colsum(mK, b, h, row0, sb, sn, sh, tid, D):
    """UNALIGNED _kmean_partial chunk (32 rows): D=128 (1 column / thread) Triton packs ROW pairs into f32x2 ->
    T16(even rows) + T16(odd rows) (kmp_D128_Hk4_pad8.ptx:423-524); D=256 (2 columns / thread) packs column pairs ->
    plain T32 per column."""
    out = []
    for j in range(D // 128):
        base = Int64(b) * Int64(sb) + Int64(h) * Int64(sh) + Int64(tid + 128 * j)
        v = [ld_bf16f(mK + (base + Int64(row0 + r) * Int64(sn)) * 2) for r in range(32)]
        if D == 128:
            out.append(fadd(tree(v[0::2]), tree(v[1::2])))
        else:
            out.append(tree(v))
    return out


def warps4_01_23(ws):
    """UNALIGNED (contiguous row groups): T32 = (T8(w0) + T8(w1)) + (T8(w2) + T8(w3))."""
    return fadd(fadd(ws[0], ws[1]), fadd(ws[2], ws[3]))


# ------------------------------------------------------------------------------------------- ProdA, chunked K mean (dev)
@dsl_user_op
def fence_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", side=True, loc=loc, ip=ip)


@dsl_user_op
def atom_add_gpu(addr: Int64, v: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [_v(addr, Int64, loc, ip), _v(v, Int32, loc, ip)],
                      "atom.add.acq_rel.gpu.global.s32 $0, [$1], $2;", "=r,l,r", side=True, loc=loc, ip=ip))


@dsl_user_op
def ld_cg_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(addr, Int64, loc, ip)], "ld.global.cg.f32 $0, [$1];", "=f,l", side=True,
                        loc=loc, ip=ip))


@dsl_user_op
def sts_i32(addr: Int32, v: Int32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int32, loc, ip), _v(v, Int32, loc, ip), _v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.shared.b32 [$0], $1;\n\t}", "r,r,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def lds_i32(addr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [_v(addr, Int32, loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", side=True, loc=loc, ip=ip))


@dsl_user_op
def st_b32_if(addr: Int64, w: Int32, pred: Int32, *, loc=None, ip=None):
    _asm(None, [_v(addr, Int64, loc, ip), _v(w, Int32, loc, ip), _v(pred, Int32, loc, ip)],
         "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}", "l,r,r", side=True,
         loc=loc, ip=ip)


class ProdAC:
    """= _prod_a_kernel with every CTA a 2-warp unit (Triton's CTA shape): 1-D grid [K-mean CHUNKS | Q tiles | V tiles].
    A chunk CTA reduces one 64-row chunk of one partial with the exact 2-warp tree (R1), stores the chunk total, and
    counts in (fence + atomic); the CTA that completes a partial folds its NCH chunk totals in CHUNK ORDER,
    acc = ((0 + c0) + c1) + ... = Triton's sequential loop, independent of arrival order, and resets the counter."""

    def __init__(self, D, vq, vk, vv, kal, headfold=False):
        """headfold: the CTA completing a head's last partial also folds the S partials (R2 order) and writes the mean
        (div.full by N) - ProdB then only quantizes."""
        assert D in (128, 256)
        self.D, self.vq, self.vk, self.vv, self.kal, self.headfold = D, vq, vk, vv, kal, headfold

    @cute.jit
    def __call__(self, mQ: Int64, mK: Int64, mV: Int64, mQ8: Int64, mV8: Int64, mQS: Int64, mVS: Int64, mWB: Int64,
                 mP: Int64, mC: Int64, mCnt: Int64, mM: Int64, N: Int32, H: Int32, Hk: Int32, B: Int32, sqb: Int32, sqn: Int32,
                 sqh: Int32, skb: Int32, skn: Int32, skh: Int32, svb: Int32, svn: Int32, svh: Int32, sm: Float32,
                 ROWS: Int32, stream: cuda.CUstream):
        ntl = N // TR
        total = B * H * ntl + B * Hk * ntl + B * Hk * (N // ROWS) * (ROWS // 64)
        self.kernel(mQ, mK, mV, mQ8, mV8, mQS, mVS, mWB, mP, mC, mCnt, mM, N, H, Hk, B, sqb, sqn, sqh, skb, skn, skh,
                    svb, svn, svh, sm, ROWS).launch(grid=[total, 1, 1], block=[64, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: Int64, mK: Int64, mV: Int64, mQ8: Int64, mV8: Int64, mQS: Int64, mVS: Int64, mWB: Int64,
               mP: Int64, mC: Int64, mCnt: Int64, mM: Int64, N: Int32, H: Int32, Hk: Int32, B: Int32, sqb: Int32, sqn: Int32,
               sqh: Int32, skb: Int32, skn: Int32, skh: Int32, svb: Int32, svn: Int32, svh: Int32, sm: Float32,
               ROWS: Int32):
        D = self.D
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        red = smem_base(2 * D * 4)
        flag = smem_base(16)
        ntl = N // TR
        nq = B * H * ntl
        nv = B * Hk * ntl
        S = N // ROWS
        NCH = ROWS // 64
        nck = B * Hk * S * NCH
        pid = bid - nck
        if bid < nck:
            pid = pid + nq + nv + nck
        if pid < nq:
            g = TileGeo(D, tid, 2)
            bh = pid // ntl
            b = bh // H
            h = bh % H
            row0 = (pid % ntl) * TR
            raw = load_tile_raw(g, mQ, b, h, row0, sqb, sqn, sqh, self.vq)
            quant_tile(g, row_fn(raw, self.vq, None, sm), red, N, h, H, bh, row0, mQ8, mQS, None, None, b)
        elif pid < nq + nv:
            g = TileGeo(D, tid, 2)
            p = pid - nq
            bh = p // ntl
            b = bh // Hk
            h = bh % Hk
            row0 = (p % ntl) * TR
            raw = load_tile_raw(g, mV, b, h, row0, svb, svn, svh, self.vv)
            quant_tile(g, row_fn(raw, self.vv), red, N, h, Hk, bh, row0, mV8, mVS, mWB, 1, b)
        else:
            cid = pid - nq - nv
            part = cid // NCH
            c = cid % NCH
            bh = part // S
            s = part % S
            b = bh // Hk
            h = bh % Hk
            cg = ChunkGeo(D, 2, 64, tid, self.kal)
            tot = chunk_colsum(cg, mK, b, h, s * ROWS + c * 64, skb, skn, skh, self.vk, red, warps2)
            ca = mC + ((Int64(part) * Int64(NCH) + Int64(c)) * Int64(D) + Int64(cg.c0)) * 4
            pub = Int32(cg.w == 0) * Int32(cg.l0 == 0)
            st_v4f_if(ca, tot[0], tot[1], tot[2], tot[3], pub)
            st_v4f_if(ca + Int64(16), tot[4], tot[5], tot[6], tot[7], pub)
            fence_gpu()
            bar_sync()
            if tid == 0:
                old = atom_add_gpu(mCnt + Int64(part) * 4, Int32(1)) + Int32(1)
                sts_i32(flag, old, Int32(1))
            bar_sync()
            if lds_i32(flag) == NCH:
                fence_gpu()
                fold_partial(mC, mP, part, NCH, D, tid)
                st_b32_if(mCnt + Int64(part) * 4, Int32(0), Int32(tid == 0))
                if const_expr(self.headfold):
                    # second level: counters [B*Hk*S + bh] count completed partials of head bh
                    fence_gpu()
                    bar_sync()
                    if tid == 0:
                        old2 = atom_add_gpu(mCnt + (Int64(B) * Int64(Hk) * Int64(S) + Int64(bh)) * 4, Int32(1)) + Int32(1)
                        sts_i32(flag + 4, old2, Int32(1))
                    bar_sync()
                    if lds_i32(flag + 4) == S:
                        fence_gpu()
                        fold_head(mP, mM, bh, S, D, tid, N)
                        st_b32_if(mCnt + (Int64(B) * Int64(Hk) * Int64(S) + Int64(bh)) * 4, Int32(0), Int32(tid == 0))


@cute.jit
def fold_head(mP: Int64, mM: Int64, bh: Int32, S: Int32, D: cutlass.Constexpr, tid: Int32, N: Int32):
    """mean[bh] = div.full((0 + P[bh,0]) + P[bh,1] + ..., N)  (R2 order; PART rows of other CTAs read through L2)."""
    acc = cute.make_rmem_tensor(D // 64, Float32)
    for j in cutlass.range_constexpr(D // 64):
        acc[j] = Float32(0.0)
    base = mP + (Int64(bh) * Int64(S) * Int64(D) + Int64(tid)) * 4
    for sg in cutlass.range(S // 8, unroll=1):
        v = [[ld_cg_f32(base + ((Int64(sg) * 8 + Int64(i)) * Int64(D) + Int64(64 * j)) * 4) for j in range(D // 64)]
             for i in range(8)]
        for i in cutlass.range_constexpr(8):
            for j in cutlass.range_constexpr(D // 64):
                acc[j] = fadd(acc[j], v[i][j])
    for s in cutlass.range(S - (S // 8) * 8, unroll=1):
        for j in cutlass.range_constexpr(D // 64):
            acc[j] = fadd(acc[j], ld_cg_f32(base + ((Int64(S // 8) * 8 + Int64(s)) * Int64(D) + Int64(64 * j)) * 4))
    nf = i2f(N)
    for j in cutlass.range_constexpr(D // 64):
        st_f32_if(mM + (Int64(bh) * Int64(D) + Int64(tid + 64 * j)) * 4, fdivfull(acc[j], nf), Int32(1))


@cute.jit
def fold_partial(mC: Int64, mP: Int64, part: Int32, NCH: Int32, D: cutlass.Constexpr, tid: Int32):
    acc = cute.make_rmem_tensor(D // 64, Float32)
    for j in cutlass.range_constexpr(D // 64):
        acc[j] = Float32(0.0)
    base = mC + (Int64(part) * Int64(NCH) * Int64(D) + Int64(tid)) * 4
    for cc in cutlass.range(NCH, unroll=1):
        for j in cutlass.range_constexpr(D // 64):
            acc[j] = fadd(acc[j], ld_cg_f32(base + (Int64(cc) * Int64(D) + Int64(64 * j)) * 4))
    for j in cutlass.range_constexpr(D // 64):
        st_f32_if(mP + (Int64(part) * Int64(D) + Int64(tid + 64 * j)) * 4, acc[j], Int32(1))
