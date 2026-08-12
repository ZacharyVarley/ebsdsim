/*
 * Galerkin expand: w_out = Q @ F (compact out_rank stride).
 *
 * After galerkin_build_uniq_shared_bicgstab_f16 leaves orthonormal Q in the first `kept`
 * columns of w_q (layout batch * n * in_rank + beam * in_rank + col)
 * and F is uploaded / written as complex64, this kernel writes W to w_out
 * with out_rank stride so intensity_fused can use params.rank = out_rank:
 *   W[i, 0:n_out) = sum_{j < kept} Q[i, j] * F[j, 0:n_out)
 *   n_out = min(out_rank, kept)   // GPU Lyapunov path: out_rank := kept/m
 *
 * F layouts (selected by f_col_stride):
 *   f_col_stride == 0 → rectangular host packing [batch][max_rank][out_rank]
 *                       (legacy rectangular host packing; default / backward compatible)
 *   f_col_stride  > 0 → col stride = f_col_stride
 *                       (gpu Lyapunov: set f_col_stride = max_rank = MAX_M for square)
 *
 * Q rows are staged in private registers before any write so reads from w_q
 * never race with writes to w_out (separate buffers).
 *
 * Bindings (group 0):
 *   0  params
 *   1  w_q     (read)        — Q from Galerkin solve (in_rank stride)
 *   2  galerkin_f   (read)        — F factors (see layouts above)
 *   3  stats   (read)        — kept in bits 20–24 of stats[batch]
 *   4  w_out   (read_write)  — expanded W (out_rank stride) for intensity
 *
 * Params packing: 8 × u32
 *   batch_count, n, in_rank, out_rank, max_rank, f_col_stride, _pad1, _pad2
 */

struct Params {
    batch_count: u32,
    n: u32,
    in_rank: u32,
    out_rank: u32,
    max_rank: u32,
    f_col_stride: u32, // 0 → out_rank (legacy rectangular); else explicit col stride
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> w_q: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> galerkin_f: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> stats: array<u32>;
@group(0) @binding(4) var<storage, read_write> w_out: array<vec2<f32>>;

const WG: u32 = 256u;
const MAX_RANK: u32 = 16u;

fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cx_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let batch = workgroup_id.x;
    let lid = local_id.x;
    if (batch >= params.batch_count) { return; }

    let n = params.n;
    let in_rank = params.in_rank;
    let out_rank = min(params.out_rank, MAX_RANK);
    let max_rank = min(params.max_rank, MAX_RANK);
    if (n == 0u || in_rank == 0u || out_rank == 0u) { return; }

    let st = stats[batch];
    var kept = 0u;
    if (st != 0xFFFFFFFFu) {
        kept = (st >> 20u) & 0x1Fu;
    }
    if (kept > in_rank) { kept = in_rank; }
    if (kept > max_rank) { kept = max_rank; }

    let qbase = batch * n * in_rank;
    let obase = batch * n * out_rank;
    // f_col_stride == 0 → legacy rectangular [max_rank][out_rank]
    // f_col_stride  > 0 → that stride (gpu Lyapunov: pass max_rank for square)
    let f_cs = select(out_rank, params.f_col_stride, params.f_col_stride != 0u);
    let fbase = batch * max_rank * f_cs;
    let n_out = min(out_rank, kept);

    for (var i = lid; i < n; i = i + WG) {
        var qrow: array<vec2<f32>, MAX_RANK>;
        // Only load kept columns (not always MAX_RANK=16); unread slots unused below.
        for (var j = 0u; j < kept; j = j + 1u) {
            qrow[j] = w_q[qbase + i * in_rank + j];
        }

        for (var c = 0u; c < out_rank; c = c + 1u) {
            var acc = vec2<f32>(0.0, 0.0);
            if (c < n_out) {
                for (var j = 0u; j < kept; j = j + 1u) {
                    let fjc = galerkin_f[fbase + j * f_cs + c];
                    acc = cx_add(acc, cx_mul(qrow[j], fjc));
                }
            }
            w_out[obase + i * out_rank + c] = acc;
        }
    }
}
