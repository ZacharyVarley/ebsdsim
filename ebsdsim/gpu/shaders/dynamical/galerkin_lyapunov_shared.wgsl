/*
 * Galerkin Lyapunov projection (standalone dispatch).
 *
 * After galerkin_build_uniq_shared_bicgstab_f16 writes H = Qᴴ A Q and b = Qᴴ e0, this kernel
 * solves the continuous Lyapunov equation on the m×m pencil:
 *
 *   H Y + Y Hᴴ + b bᴴ = 0
 *
 * via the Kronecker system (f32 Gauss–Jordan)
 *
 *   (I ⊗ H + conj(H) ⊗ I) vec(Y) = -vec(b bᴴ)
 *
 * then Hermitianizes Y and factors Y ≈ F Fᴴ by Cholesky (tiny diagonal floor).
 * Output rank = m (= kept). No eig truncation.
 *
 * One workgroup per batch element. Workgroup size 256 (matches the solve
 * shader; N×(N+1) ≤ 1332 augmented entries at m=6 benefit from wider WG).
 * MAX_M = 6 shared-memory path (product default krylov; fits 32 KiB WG).
 * For krylov > 6 use galerkin_lyapunov_implicit.wgsl (implicit Kronecker, MAX_M=16).
 * Workgroup footprint @ MAX_M=6, WG=256 (~13.6 KB):
 *   g_K  36²×8 = 10368; g_rhs/H/Y/F 4×36×8 = 1152; g_b 48;
 *   scalars ~28; g_scratch + g_scratch_u 2048 → ~13600 B (margin ~18 KB).
 * h_ld in params is smith H leading dim (storage = krylovRank / params.rank) — NOT workgroup size.
 *
 * Kronecker indexing (column-major vec; must match the numpy Lyapunov oracle):
 *   vec(Y)[i + j*m] = Y[i,j]          (column-major flatten)
 *   g_K[row * N + col]                (row-major over N×N, N = m²)
 *   (I⊗H):     row=i+c*m, col=k+c*m,  += H[i,k]
 *   (conj(H)⊗I): row=i+j*m, col=i+ell*m, += conj(H[j,ell])
 *
 * Bindings (group 0):
 *   0  params   — batch_count, h_ld, f_ld, _pad
 *   1  galerkin_h    — [batch][h_ld][h_ld] c64 from Galerkin solve
 *   2  galerkin_b    — [batch][h_ld]       c64
 *   3  stats    — read kept (bits 20–24); write galerkin_status into bits 25–26
 *   4  galerkin_f    — out [batch][f_ld][f_ld] c64, f_ld = MAX_M
 *
 * Params packing: 4 × u32
 *   batch_count, h_ld, f_ld, _pad
 *
 * galerkin_status (logical):
 *   bit0 = singular pencil
 *   bit1 = Cholesky floor engaged
 * Packed into stats[batch] as bit25 / bit26 (OR'd; kept bits preserved).
 *
 * Do NOT fold into the Galerkin solve kernel — Kronecker needs ~10.6 KB at
 * m=6 and the solve WG has almost no free shared memory. Standalone
 * dispatch; this shared path caps MAX_M=6.
 *
 * After this kernel, galerkin_expand.wgsl forms W = Q F for intensity.
 */

struct Params {
    batch_count: u32,
    h_ld: u32,   // leading dim of galerkin_h / galerkin_b (smith params.rank / host krylovRank)
    f_ld: u32,   // leading dim of galerkin_f (= MAX_M = 6)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> galerkin_h: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> galerkin_b: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> stats: array<u32>;
@group(0) @binding(4) var<storage, read_write> galerkin_f: array<vec2<f32>>;

const WG: u32 = 256u;
const MAX_M: u32 = 6u;
const MAX_N2: u32 = 36u;            // MAX_M²
const MAX_K: u32 = 1296u;           // MAX_N2²
const PIVOT_EPS: f32 = 1.0e-20;
const CHOL_REL_FLOOR: f32 = 1.0e-7;
const CHOL_ABS_FLOOR: f32 = 1.0e-20;

const STATUS_SINGULAR: u32 = 1u;
const STATUS_CHOL_FLOOR: u32 = 2u;

// Workgroup memory (~12.1 KB peak for K + scratch; fits 32 KiB devices)
var<workgroup> g_K: array<vec2<f32>, MAX_K>;       // 10368 B
var<workgroup> g_rhs: array<vec2<f32>, MAX_N2>;    // 288 B
var<workgroup> g_H: array<vec2<f32>, MAX_N2>;      // 288 B (m×m, row-major ld=m)
var<workgroup> g_Y: array<vec2<f32>, MAX_N2>;      // 288 B
var<workgroup> g_F: array<vec2<f32>, MAX_N2>;      // 288 B
var<workgroup> g_b: array<vec2<f32>, MAX_M>;       // 48 B
var<workgroup> g_m: u32;
var<workgroup> g_N: u32;
var<workgroup> g_bad: u32;
var<workgroup> g_status: u32;
var<workgroup> g_piv: u32;
var<workgroup> g_piv_mag: f32;
var<workgroup> g_max_diag: f32;
var<workgroup> g_scratch: array<f32, WG>;           // reductions
var<workgroup> g_scratch_u: array<u32, WG>;

fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cx_conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

fn cx_abs2(a: vec2<f32>) -> f32 {
    return a.x * a.x + a.y * a.y;
}

fn cx_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = cx_abs2(b);
    if (d <= 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let inv = 1.0 / d;
    return vec2<f32>((a.x * b.x + a.y * b.y) * inv, (a.y * b.x - a.x * b.y) * inv);
}

fn cx_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b;
}

fn cx_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

fn cx_neg(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.x, -a.y);
}

fn cx_scale(a: vec2<f32>, s: f32) -> vec2<f32> {
    return vec2<f32>(a.x * s, a.y * s);
}

/// Row-major H index with leading dim = m (packed into g_H densely).
fn h_idx(i: u32, j: u32, m: u32) -> u32 {
    return i * m + j;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let batch = workgroup_id.x;
    let lid = local_id.x;
    if (batch >= params.batch_count) { return; }

    let h_ld = max(params.h_ld, 1u);
    let f_ld = min(max(params.f_ld, 1u), MAX_M);

    // --- read kept from stats ---
    if (lid == 0u) {
        g_status = 0u;
        g_bad = 0u;
        let st = stats[batch];
        var kept = 0u;
        if (st != 0xFFFFFFFFu) {
            kept = (st >> 20u) & 0x1Fu;
        }
        if (kept > MAX_M) { kept = MAX_M; }
        if (kept > h_ld) { kept = h_ld; }
        g_m = kept;
        g_N = kept * kept;
    }
    workgroupBarrier();
    let m = workgroupUniformLoad(&g_m);
    let N = workgroupUniformLoad(&g_N);

    // Clear F output for this batch (full f_ld×f_ld tile)
    let fbase = batch * f_ld * f_ld;
    for (var t = lid; t < f_ld * f_ld; t = t + WG) {
        galerkin_f[fbase + t] = vec2<f32>(0.0, 0.0);
    }

    if (m == 0u) {
        workgroupBarrier();
        return;
    }

    // --- load H (m×m) and b (m) into workgroup ---
    let hbase = batch * h_ld * h_ld;
    let bbase = batch * h_ld;
    for (var t = lid; t < m * m; t = t + WG) {
        let i = t / m;
        let j = t - i * m;
        g_H[t] = galerkin_h[hbase + i * h_ld + j];
    }
    for (var t = lid; t < m; t = t + WG) {
        g_b[t] = galerkin_b[bbase + t];
    }
    // Clear K and rhs
    for (var t = lid; t < N * N; t = t + WG) {
        g_K[t] = vec2<f32>(0.0, 0.0);
    }
    for (var t = lid; t < N; t = t + WG) {
        g_rhs[t] = vec2<f32>(0.0, 0.0);
    }
    workgroupBarrier();

    // --- build RHS = -vec(b bᴴ), column-major ---
    for (var t = lid; t < N; t = t + WG) {
        let j = t / m;
        let i = t - j * m;
        g_rhs[t] = cx_neg(cx_mul(g_b[i], cx_conj(g_b[j])));
    }

    // --- (I ⊗ H): for each Y-column c, block-diag H ---
    // Parallelize over (c, i, k) flattened as c*m*m + i*m + k.
    // Each (row,col) is written exactly once in this phase (no atomics).
    let n_ih = m * m * m;
    for (var t = lid; t < n_ih; t = t + WG) {
        let c = t / (m * m);
        let rem = t - c * m * m;
        let i = rem / m;
        let k = rem - i * m;
        let row = i + c * m;
        let col = k + c * m;
        g_K[row * N + col] = g_H[h_idx(i, k, m)];
    }
    workgroupBarrier();

    // --- (conj(H) ⊗ I) ---
    // Parallelize over (j, ell, i); each (row,col) unique for fixed structure
    for (var t = lid; t < n_ih; t = t + WG) {
        let j = t / (m * m);
        let rem = t - j * m * m;
        let ell = rem / m;
        let i = rem - ell * m;
        let row = i + j * m;
        let col = i + ell * m;
        let hc = cx_conj(g_H[h_idx(j, ell, m)]);
        g_K[row * N + col] = cx_add(g_K[row * N + col], hc);
    }
    workgroupBarrier();

    // =====================================================================
    // Gauss–Jordan on [K | rhs]
    // Pivot search is a parallel tree-reduce; elimination strides all lanes
    // over the N×(N+1) augmented matrix (O(N) barriers total, not O(N²)).
    // Uniform break via workgroupUniformLoad(&g_bad).
    // =====================================================================
    for (var k = 0u; k < N; k = k + 1u) {
        // --- find pivot row: max |K[i,k]| for i >= k ---
        if (lid == 0u) {
            g_piv = k;
            g_piv_mag = 0.0;
        }
        workgroupBarrier();

        // Each thread scans a strided subset of candidate rows
        var local_piv = k;
        var local_mag = 0.0;
        for (var i = k + lid; i < N; i = i + WG) {
            let mag = sqrt(cx_abs2(g_K[i * N + k]));
            if (mag > local_mag) {
                local_mag = mag;
                local_piv = i;
            }
        }
        g_scratch[lid] = local_mag;
        g_scratch_u[lid] = local_piv;
        workgroupBarrier();

        // Tree reduce max; requires WG to be a power of two
        var stride = WG >> 1u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) {
                let o = lid + stride;
                if (g_scratch[o] > g_scratch[lid]) {
                    g_scratch[lid] = g_scratch[o];
                    g_scratch_u[lid] = g_scratch_u[o];
                }
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) {
            g_piv_mag = g_scratch[0];
            g_piv = g_scratch_u[0];
            if (g_piv_mag < PIVOT_EPS) {
                g_bad = 1u;
                g_status = g_status | STATUS_SINGULAR;
            }
        }
        workgroupBarrier();

        let bad = workgroupUniformLoad(&g_bad);
        if (bad == 1u) { break; }

        let piv = workgroupUniformLoad(&g_piv);

        // --- swap rows k ↔ piv ---
        if (piv != k) {
            for (var j = lid; j < N; j = j + WG) {
                let tmp = g_K[k * N + j];
                g_K[k * N + j] = g_K[piv * N + j];
                g_K[piv * N + j] = tmp;
            }
            if (lid == 0u) {
                let tr = g_rhs[k];
                g_rhs[k] = g_rhs[piv];
                g_rhs[piv] = tr;
            }
        }
        workgroupBarrier();

        // --- scale pivot row so K[k,k] = 1 ---
        let akk = g_K[k * N + k];
        for (var j = lid; j < N; j = j + WG) {
            g_K[k * N + j] = cx_div(g_K[k * N + j], akk);
        }
        if (lid == 0u) {
            g_rhs[k] = cx_div(g_rhs[k], akk);
        }
        workgroupBarrier();

        // --- eliminate other rows: cooperative over N×(N+1) augmented entries ---
        // Cache factors[i] = K[i,k] first so parallel writes to column k cannot race
        // with factor reads. Reuse g_Y (idle until post-GJ reshape) as scratch.
        // Equivalence: for fixed k the maps row_i -= factor_i * row_k (i≠k) commute
        // because each update only reads the scaled pivot row and its own factor;
        // serial-per-row vs all-rows-parallel is therefore identical. With K[k,k]=1
        // after scaling, the j=k update naturally zeros the pivot column.
        for (var i = lid; i < N; i = i + WG) {
            g_Y[i] = g_K[i * N + k];
        }
        workgroupBarrier();

        let aug_n = N * (N + 1u);
        for (var idx = lid; idx < aug_n; idx = idx + WG) {
            let i = idx / (N + 1u);
            let j = idx - i * (N + 1u);
            if (i == k) { continue; }
            let factor = g_Y[i];
            let f2 = cx_abs2(factor);
            if (f2 <= 0.0) {
                if (j == k) {
                    g_K[i * N + k] = vec2<f32>(0.0, 0.0);
                }
                continue;
            }
            if (j < N) {
                g_K[i * N + j] = cx_sub(g_K[i * N + j], cx_mul(factor, g_K[k * N + j]));
            } else {
                g_rhs[i] = cx_sub(g_rhs[i], cx_mul(factor, g_rhs[k]));
            }
        }
        workgroupBarrier();
    }

    let bad_final = workgroupUniformLoad(&g_bad);
    if (bad_final == 1u) {
        // Singular: leave F zeros; write status
        if (lid == 0u) {
            let st = stats[batch];
            if (st != 0xFFFFFFFFu) {
                // clear previous status bits 25–26 then OR
                var st2 = st & ~(0x3u << 25u);
                st2 = st2 | ((g_status & 0x3u) << 25u);
                stats[batch] = st2;
            }
        }
        workgroupBarrier();
        return;
    }

    // --- reshape rhs → Y (column-major), Hermitianize ---
    for (var t = lid; t < N; t = t + WG) {
        let j = t / m;
        let i = t - j * m;
        g_Y[h_idx(i, j, m)] = g_rhs[t];
    }
    workgroupBarrier();

    // Hermitianize: only threads with i <= j write both (i,j) and (j,i)
    for (var t = lid; t < N; t = t + WG) {
        let i = t / m;
        let j = t - i * m;
        if (j < i) { continue; }
        let a = g_Y[h_idx(i, j, m)];
        let b = g_Y[h_idx(j, i, m)];
        let re = 0.5 * (a.x + b.x);
        let im = 0.5 * (a.y - b.y);
        g_Y[h_idx(i, j, m)] = vec2<f32>(re, im);
        g_Y[h_idx(j, i, m)] = vec2<f32>(re, -im);
    }
    workgroupBarrier();

    // --- Cholesky Y ≈ F Fᴴ (lower triangular) ---
    for (var t = lid; t < N; t = t + WG) {
        g_F[t] = vec2<f32>(0.0, 0.0);
    }
    if (lid == 0u) {
        g_max_diag = 0.0;
        for (var i = 0u; i < m; i = i + 1u) {
            let d = abs(g_Y[h_idx(i, i, m)].x);
            if (d > g_max_diag) { g_max_diag = d; }
        }
    }
    workgroupBarrier();
    let max_diag = workgroupUniformLoad(&g_max_diag);
    let floor_v = max(CHOL_ABS_FLOOR, CHOL_REL_FLOOR * max_diag);

    // Serial Cholesky over k (m ≤ 6); inner products parallelized lightly
    for (var k = 0u; k < m; k = k + 1u) {
        // Diagonal: serial on lid 0 (depends on previous columns)
        if (lid == 0u) {
            var acc = g_Y[h_idx(k, k, m)].x;
            for (var j = 0u; j < k; j = j + 1u) {
                acc = acc - cx_abs2(g_F[h_idx(k, j, m)]);
            }
            if (acc < floor_v) {
                acc = floor_v;
                g_status = g_status | STATUS_CHOL_FLOOR;
            }
            if (acc < PIVOT_EPS) {
                g_status = g_status | STATUS_CHOL_FLOOR;
                g_F[h_idx(k, k, m)] = vec2<f32>(0.0, 0.0);
            } else {
                g_F[h_idx(k, k, m)] = vec2<f32>(sqrt(acc), 0.0);
            }
        }
        workgroupBarrier();

        let dkk = g_F[h_idx(k, k, m)];
        let dkk2 = cx_abs2(dkk);
        // Off-diagonal column k: rows i = k+1 .. m-1
        // Parallelize over i
        for (var i = k + 1u + lid; i < m; i = i + WG) {
            if (dkk2 <= 0.0) {
                g_F[h_idx(i, k, m)] = vec2<f32>(0.0, 0.0);
            } else {
                var s = g_Y[h_idx(i, k, m)];
                for (var j = 0u; j < k; j = j + 1u) {
                    s = cx_sub(s, cx_mul(g_F[h_idx(i, j, m)], cx_conj(g_F[h_idx(k, j, m)])));
                }
                g_F[h_idx(i, k, m)] = cx_div(s, dkk);
            }
        }
        workgroupBarrier();
    }

    // --- write F to galerkin_f [batch][f_ld][f_ld] ---
    for (var t = lid; t < m * m; t = t + WG) {
        let i = t / m;
        let j = t - i * m;
        galerkin_f[fbase + i * f_ld + j] = g_F[t];
    }

    // --- pack galerkin_status into stats bits 25–26 ---
    if (lid == 0u) {
        let st = stats[batch];
        if (st != 0xFFFFFFFFu) {
            var st2 = st & ~(0x3u << 25u);
            st2 = st2 | ((g_status & 0x3u) << 25u);
            stats[batch] = st2;
        }
    }
    workgroupBarrier();
}
