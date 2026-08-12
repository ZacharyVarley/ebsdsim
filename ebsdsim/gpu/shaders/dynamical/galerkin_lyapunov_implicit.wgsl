/*
 * Galerkin Lyapunov projection — implicit Kronecker (high Krylov ranks, m = 7..16).
 *
 * Solves the same continuous Lyapunov equation as galerkin_lyapunov_shared.wgsl:
 *
 *   H Y + Y Hᴴ + b bᴴ = 0
 *
 * via the Kronecker system
 *
 *   (I ⊗ H + conj(H) ⊗ I) vec(Y) = -vec(b bᴴ)
 *
 * but K = I⊗H + conj(H)⊗I is **never materialized**.  Instead the solve uses
 * CGNE (CG on the normal equations KᴴK x = Kᴴ b) on the m²-vector, where each
 * operator@vec is evaluated implicitly as two m×m matmuls:
 *
 *   K  vec(V) = vec(H V + V Hᴴ)         (forward Lyapunov operator)
 *   Kᴴ vec(V) = vec(Hᴴ V + V H)          (adjoint)
 *
 * — O(m³) per matvec and zero K VRAM.  This replaces the storage-backed
 * Gauss–Jordan path (galerkin_lyapunov_storage.wgsl) whose O(m⁶) GJ work over a
 * materialized m²×m² c64 K made rank 8+ practically unusable, and avoids
 * BiCGSTAB's breakdown sensitivity.
 *
 * CGNE is chosen over BiCGSTAB because the Lyapunov Kronecker operator is
 * non-Hermitian but KᴴK is Hermitian SPD (K nonsingular for stable H), so CG
 * cannot break down.  cond(KᴴK) = cond(K)² ≈ 64–676 for the dissipative Galerkin
 * pencils (cond(K) ≈ 8–26), so convergence in ≤ ~50 iterations is expected.
 *
 * No preconditioner: the Jacobi diagonal of KᴴK is |H[i,i]+conj(H[j,j])|²
 * which is already O(1) for dissipative H; CGNE self-preconditions via the
 * Krylov subspace.  (A diagonal preconditioner would need KᴴK's diagonal,
 * which costs an extra O(m³) per application and was not worth it here.)
 *
 * Operator identity (column-major vec, vec(V)[i + j*m] = V[i,j]):
 *   (I ⊗ H) vec(V)       = vec(H V)         (block-diag H per Y-column)
 *   (conj(H) ⊗ I) vec(V) = vec(V Hᴴ)        (H̄_{j,ℓ} on V[i,ℓ], summed over ℓ)
 *   => K vec(V) = vec(H V + V Hᴴ)           matches scipy.solve_continuous_lyapunov
 *   Kᴴ = I⊗Hᴴ + Hᵀ⊗I  =>  Kᴴ vec(V) = vec(Hᴴ V + V H)
 *
 * One workgroup per batch element. Workgroup size 256 (N=m² ≤ 256 so each
 * lane owns ≤1 vector entry at max rank; shared-mem still fits 32 KiB).
 * MAX_M = 16 (matches GALERKIN_MAX_RANK).
 *
 * Workgroup footprint @ MAX_M=16, WG=256 (~19.8 KB, fits 32 KiB):
 *   g_H            16²·8   = 2048
 *   g_b            16·8    = 128
 *   g_Y,g_F        2·16²·8 = 4096
 *   CGNE vecs      5·16²·8 = 10240  (x, r, p, w, Kp)  + 1 (Kᴴb staging in g_rhs)
 *   g_rhs          16²·8   = 2048
 *   scalars + g_re_re      ~1100
 *   total                  ≈ 19800 B
 * m=8: much smaller; shared path preferred for m ≤ 6.
 *
 * Kronecker indexing (must match the shared path and the numpy Lyapunov oracle):
 *   vec(Y)[i + j*m] = Y[i,j]          (column-major flatten)
 *   RHS = -vec(b bᴴ) = -outer(b, conj(b)) flattened column-major.
 *
 * Bindings (group 0) — identical to the shared path (no galerkin_k scratch):
 *   0  params   — batch_count, h_ld, f_ld, _pad (k_n2 ignored)
 *   1  galerkin_h    — [batch][h_ld][h_ld] c64 from Galerkin solve
 *   2  galerkin_b    — [batch][h_ld]       c64
 *   3  stats    — read kept (bits 20–24); write galerkin_status into bits 25–26
 *   4  galerkin_f    — out [batch][f_ld][f_ld] c64
 *
 * galerkin_status (logical):
 *   bit0 = singular pencil (CG failed / Kᴴb ≈ 0 with nonzero RHS)
 *   bit1 = Cholesky near-singular pivot dropped (column zeroed)
 * Packed into stats[batch] as bit25 / bit26 (OR'd; kept bits preserved).
 *
 * Cholesky near-singular handling: the continuous-Lyapunov solution Y is
 * Hermitian PSD for stable H, but can be near-singular (low rank) when b
 * has little projection onto weakly-damped modes.  When a Cholesky pivot
 * acc = Y[k,k] - Σ|F[k,j]|² falls below floor = max(1e-20, 1e-7·max_diag),
 * the entire column k is zeroed (F[k,k]=0 ⇒ F[i,k]=0) instead of flooring
 * the diagonal to a tiny value.  Flooring the diagonal would divide the
 * off-diagonal entries by ~√floor, amplifying them to O(Y/floor) and
 * blowing up F Fᴴ.  Zeroing the column mirrors the eigh oracle's
 * eigenvalue clip (ev ≥ 0) and keeps F Fᴴ bounded.
 */

struct Params {
    batch_count: u32,
    h_ld: u32,   // leading dim of galerkin_h / galerkin_b (smith MAX_RANK, typically 16)
    f_ld: u32,   // leading dim of galerkin_f (= MAX_M for this path, ≤ 16)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> galerkin_h: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> galerkin_b: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> stats: array<u32>;
@group(0) @binding(4) var<storage, read_write> galerkin_f: array<vec2<f32>>;

const WG: u32 = 256u;
const MAX_M: u32 = 16u;
const MAX_N2: u32 = 256u;           // MAX_M²
const CHOL_REL_FLOOR: f32 = 1.0e-7;
const CHOL_ABS_FLOOR: f32 = 1.0e-20;

// CGNE controls
const CG_MAX_ITER: u32 = 512u;      // > 2·m² for m=16 (256); ample for cond²≤700
const CG_TOL: f32 = 1.0e-6;         // relative residual ‖r‖/‖Kᴴb‖
const CG_BREAK_EPS: f32 = 1.0e-30;

const STATUS_SINGULAR: u32 = 1u;
const STATUS_CHOL_FLOOR: u32 = 2u;

// ---- workgroup memory ----
var<workgroup> g_H: array<vec2<f32>, MAX_N2>;      // m×m row-major, ld=m
var<workgroup> g_b: array<vec2<f32>, MAX_M>;
// CGNE vectors (length N = m², column-major vecs of m×m matrices)
var<workgroup> g_x: array<vec2<f32>, MAX_N2>;      // solution (y-space; no precond)
var<workgroup> g_r: array<vec2<f32>, MAX_N2>;      // residual
var<workgroup> g_p: array<vec2<f32>, MAX_N2>;      // search direction
var<workgroup> g_w: array<vec2<f32>, MAX_N2>;      // KᴴK p
var<workgroup> g_Kp: array<vec2<f32>, MAX_N2>;     // K p (staging for KᴴK p)
var<workgroup> g_rhs: array<vec2<f32>, MAX_N2>;    // Kᴴ b (CGNE rhs) / matvec staging
var<workgroup> g_Y: array<vec2<f32>, MAX_N2>;      // Cholesky source
var<workgroup> g_F: array<vec2<f32>, MAX_N2>;
var<workgroup> g_m: u32;
var<workgroup> g_N: u32;
var<workgroup> g_bad: u32;
var<workgroup> g_status: u32;
var<workgroup> g_max_diag: f32;
var<workgroup> g_re_re: array<f32, WG>;
var<workgroup> g_dot: f32;          // real scalar reduction result (Hermitian inner products)
var<workgroup> g_rnorm2: f32;
var<workgroup> g_bnorm2: f32;
var<workgroup> g_rr: f32;           // <r,r> current
var<workgroup> g_alpha: f32;
var<workgroup> g_beta: f32;

// ---- complex helpers (c64 = vec2<f32> = (re, im)) ----

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

/// Row-major H index with leading dim = m (packed densely in g_H).
fn h_idx(i: u32, j: u32, m: u32) -> u32 {
    return i * m + j;
}

/// Tree-reduce the real per-thread accumulators in g_re_re into g_re_re[0].
fn reduce_real(lid: u32) {
    var stride = WG >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            let o = lid + stride;
            g_re_re[lid] = g_re_re[lid] + g_re_re[o];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
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
        if (kept > f_ld) { kept = f_ld; }
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
    workgroupBarrier();

    // =====================================================================
    // Build the original Lyapunov RHS = -vec(b bᴴ) (column-major) into g_rhs
    // (temporarily), then form the CGNE rhs = Kᴴ (-vec(b bᴴ)) into g_rhs.
    // CGNE solves KᴴK x = Kᴴ rhs_orig.
    // =====================================================================
    for (var t = lid; t < N; t = t + WG) {
        let j = t / m;
        let i = t - j * m;
        g_rhs[t] = cx_neg(cx_mul(g_b[i], cx_conj(g_b[j])));
    }
    workgroupBarrier();

    // rhs_cgne = Kᴴ g_rhs  →  store back into g_rhs.
    // Kᴴ vec(V) = vec(Hᴴ V + V H);  Hᴴ[a,b] = conj(H[b,a]).
    // (Derivation: Kᴴ = I⊗Hᴴ + Hᵀ⊗I; the Hᵀ⊗I term contributes
    //  sum_l H[l,j] V[i,l] = (V H)[i,j] — note H, not Hᵀ, in the matmul.)
    for (var t = lid; t < N; t = t + WG) {
        let j = t / m;
        let i = t - j * m;
        var hv = vec2<f32>(0.0, 0.0);   // Hᴴ V
        var vh = vec2<f32>(0.0, 0.0);   // V H
        for (var k = 0u; k < m; k = k + 1u) {
            // (Hᴴ V)[i,j] = sum_k Hᴴ[i,k] V[k,j] = sum_k conj(H[k,i]) V[k,j]
            hv = cx_add(hv, cx_mul(cx_conj(g_H[h_idx(k, i, m)]), g_rhs[k + j * m]));
            // (V H)[i,j] = sum_l V[i,l] H[l,j]
            vh = cx_add(vh, cx_mul(g_rhs[i + k * m], g_H[h_idx(k, j, m)]));
        }
        g_Kp[t] = cx_add(hv, vh);       // reuse g_Kp as staging
    }
    workgroupBarrier();
    for (var t = lid; t < N; t = t + WG) {
        g_rhs[t] = g_Kp[t];
    }
    workgroupBarrier();

    // =====================================================================
    // CGNE: solve KᴴK x = rhs_cgne.  KᴴK is Hermitian SPD (K nonsingular).
    //   r = rhs_cgne (x=0);  p = r;
    //   w = KᴴK p;  alpha = <r,r>/<p,w>;  x += alpha p;  r -= alpha w;
    //   beta = <r_new,r_new>/<r_old,r_old>;  p = r_new + beta p;
    // =====================================================================
    for (var t = lid; t < N; t = t + WG) {
        g_x[t] = vec2<f32>(0.0, 0.0);
        g_r[t] = g_rhs[t];
        g_p[t] = g_rhs[t];
    }

    // ||rhs_cgne||² for relative stopping.
    var bpart = 0.0;
    for (var t = lid; t < N; t = t + WG) {
        bpart = bpart + cx_abs2(g_rhs[t]);
    }
    g_re_re[lid] = bpart;
    workgroupBarrier();
    reduce_real(lid);
    if (lid == 0u) {
        g_bnorm2 = g_re_re[0];
        g_rr = g_re_re[0];   // <r,r> = ||rhs||² initially
        if (g_bnorm2 < CG_BREAK_EPS) {
            g_bad = 2u;      // zero RHS → x = 0
        }
    }
    workgroupBarrier();
    let bnorm2 = workgroupUniformLoad(&g_bnorm2);

    if (bnorm2 >= CG_BREAK_EPS) {
        for (var iter = 0u; iter < CG_MAX_ITER; iter = iter + 1u) {
            // --- w = KᴴK p :  Kp = K p (g_Kp), then w = Kᴴ Kp (g_w) ---
            // K vec(V) = vec(H V + V Hᴴ)
            for (var t = lid; t < N; t = t + WG) {
                let j = t / m;
                let i = t - j * m;
                var hv = vec2<f32>(0.0, 0.0);
                var vh = vec2<f32>(0.0, 0.0);
                for (var k = 0u; k < m; k = k + 1u) {
                    hv = cx_add(hv, cx_mul(g_H[h_idx(i, k, m)], g_p[k + j * m]));
                    vh = cx_add(vh, cx_mul(g_p[i + k * m], cx_conj(g_H[h_idx(j, k, m)])));
                }
                g_Kp[t] = cx_add(hv, vh);
            }
            workgroupBarrier();
            // w = Kᴴ Kp = vec(Hᴴ Kp + Kp H)
            for (var t = lid; t < N; t = t + WG) {
                let j = t / m;
                let i = t - j * m;
                var hv = vec2<f32>(0.0, 0.0);
                var vh = vec2<f32>(0.0, 0.0);
                for (var k = 0u; k < m; k = k + 1u) {
                    hv = cx_add(hv, cx_mul(cx_conj(g_H[h_idx(k, i, m)]), g_Kp[k + j * m]));
                    vh = cx_add(vh, cx_mul(g_Kp[i + k * m], g_H[h_idx(k, j, m)]));
                }
                g_w[t] = cx_add(hv, vh);
            }
            workgroupBarrier();

            // --- alpha = <r,r> / <p,w>  (Hermitian inner products, real) ---
            var pw = 0.0;
            for (var t = lid; t < N; t = t + WG) {
                // <p,w> = sum conj(p[t]) w[t]  (real for Hermitian KᴴK)
                let pp = g_p[t];
                let cp = vec2<f32>(pp.x, -pp.y);
                let d = cx_mul(cp, g_w[t]);
                pw = pw + d.x;   // imag part ≈ 0
            }
            g_re_re[lid] = pw;
            workgroupBarrier();
            reduce_real(lid);
            if (lid == 0u) { g_dot = g_re_re[0]; }
            workgroupBarrier();
            let pw_s = workgroupUniformLoad(&g_dot);
            if (pw_s < CG_BREAK_EPS) {
                // KᴴK p ≈ 0 → search direction lost; stop with current x.
                if (lid == 0u) { g_bad = 1u; g_status = g_status | STATUS_SINGULAR; }
                workgroupBarrier();
                break;
            }
            let rr_old = workgroupUniformLoad(&g_rr);
            let alpha = rr_old / pw_s;
            if (lid == 0u) { g_alpha = alpha; }
            workgroupBarrier();
            let a = workgroupUniformLoad(&g_alpha);

            // --- x += a p ;  r -= a w ---
            for (var t = lid; t < N; t = t + WG) {
                g_x[t] = cx_add(g_x[t], cx_scale(g_p[t], a));
                g_r[t] = cx_sub(g_r[t], cx_scale(g_w[t], a));
            }
            workgroupBarrier();

            // --- <r_new,r_new> ---
            var rr_new = 0.0;
            for (var t = lid; t < N; t = t + WG) {
                rr_new = rr_new + cx_abs2(g_r[t]);
            }
            g_re_re[lid] = rr_new;
            workgroupBarrier();
            reduce_real(lid);
            if (lid == 0u) { g_rnorm2 = g_re_re[0]; }
            workgroupBarrier();
            let rnorm2 = workgroupUniformLoad(&g_rnorm2);
            if (sqrt(rnorm2) <= CG_TOL * sqrt(bnorm2)) {
                break;
            }

            // --- beta = <r_new,r_new> / <r_old,r_old> ;  p = r_new + beta p ---
            let beta = rnorm2 / max(rr_old, CG_BREAK_EPS);
            if (lid == 0u) { g_beta = beta; g_rr = rnorm2; }
            workgroupBarrier();
            let b = workgroupUniformLoad(&g_beta);
            for (var t = lid; t < N; t = t + WG) {
                g_p[t] = cx_add(g_r[t], cx_scale(g_p[t], b));
            }
            workgroupBarrier();
        }

        if (workgroupUniformLoad(&g_bad) == 1u) {
            if (lid == 0u) { g_status = g_status | STATUS_SINGULAR; }
            workgroupBarrier();
        }
    }

    // =====================================================================
    // Extract Y from x (column-major vec), Hermitianize, Cholesky → F.
    // (Identical to the shared path, except the near-singular pivot branch
    //  zeros the whole column instead of flooring the diagonal — see below.)
    // =====================================================================
    for (var t = lid; t < N; t = t + WG) {
        let j = t / m;
        let i = t - j * m;
        g_Y[h_idx(i, j, m)] = g_x[t];
    }
    workgroupBarrier();

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

    for (var k = 0u; k < m; k = k + 1u) {
        if (lid == 0u) {
            var acc = g_Y[h_idx(k, k, m)].x;
            for (var j = 0u; j < k; j = j + 1u) {
                acc = acc - cx_abs2(g_F[h_idx(k, j, m)]);
            }
            if (acc < floor_v) {
                // Near-singular pivot: drop this component entirely (zero the
                // whole column) rather than flooring the diagonal to a tiny
                // value, which would amplify off-diagonals to O(Y/floor) and
                // blow up F Fᴴ.  Mirrors the eigh path's eigenvalue clip ≥ 0.
                acc = 0.0;
                g_status = g_status | STATUS_CHOL_FLOOR;
            }
            g_F[h_idx(k, k, m)] = vec2<f32>(sqrt(max(acc, 0.0)), 0.0);
        }
        workgroupBarrier();

        let dkk = g_F[h_idx(k, k, m)];
        let dkk2 = cx_abs2(dkk);
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
