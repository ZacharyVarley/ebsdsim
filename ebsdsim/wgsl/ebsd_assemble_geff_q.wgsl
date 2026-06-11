/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

struct AssembleGeffQParams {
    batch_count: u32,
    n: u32,
    use_sigma: u32,
    _pad0: u32,
    mu_shift: f32,
    eps: f32,
    _pad1: f32,
    _pad2: f32,
}

fn cx_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x + b.x, a.y + b.y);
}

fn cx_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x - b.x, a.y - b.y);
}

@group(0) @binding(0) var<uniform> params: AssembleGeffQParams;
@group(0) @binding(1) var<storage, read> v_aa: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> d_a: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> sigma: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> idx_a: array<u32>;
@group(0) @binding(5) var<storage, read_write> q_values: array<f32>;
@group(0) @binding(6) var<storage, read_write> q_minus_a: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> q_plus_a: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> e0: array<vec2<f32>>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let b = wg_id.x;
    if (b >= params.batch_count) { return; }

    let mat_base = b * params.n * params.n;
    let vec_base = b * params.n;
    var fro_sq = 0.0;
    var diag_re_sum = 0.0;
    var inc_local = 0u;

    for (var i: u32 = 0u; i < params.n; i = i + 1u) {
        if (idx_a[vec_base + i] == 0u) {
            inc_local = i;
        }
        e0[vec_base + i] = vec2<f32>(0.0, 0.0);
    }
    e0[vec_base + inc_local] = vec2<f32>(1.0, 0.0);

    for (var i: u32 = 0u; i < params.n; i = i + 1u) {
        for (var j: u32 = 0u; j < params.n; j = j + 1u) {
            let mi = mat_base + i * params.n + j;
            var value = v_aa[mi];
            if (i == j) {
                value = cx_add(value, d_a[vec_base + i]);
            }
            if (params.use_sigma != 0u) {
                value = cx_sub(value, sigma[mi]);
            }
            q_plus_a[mi] = value;
            fro_sq = fro_sq + value.x * value.x + value.y * value.y;
            if (i == j) {
                diag_re_sum = diag_re_sum + value.x;
            }
        }
    }

    let n_f = max(f32(params.n), 1.0);
    let half_mu = 0.5 * params.mu_shift;
    let fro_per_dim = sqrt(max(fro_sq, 0.0)) / n_f;
    let mean_decay = (-diag_re_sum) / n_f + half_mu;
    let q = sqrt(max(fro_per_dim * max(mean_decay, params.eps), params.eps));
    let q_plus = q + half_mu;
    let q_minus = q - half_mu;
    q_values[b * 4u + 0u] = q;
    q_values[b * 4u + 1u] = q_plus;
    q_values[b * 4u + 2u] = q_minus;
    q_values[b * 4u + 3u] = sqrt(2.0 * q);

    for (var i: u32 = 0u; i < params.n; i = i + 1u) {
        for (var j: u32 = 0u; j < params.n; j = j + 1u) {
            let mi = mat_base + i * params.n + j;
            let g = q_plus_a[mi];
            if (i == j) {
                q_minus_a[mi] = vec2<f32>(q_plus - g.x, -g.y);
                q_plus_a[mi] = vec2<f32>(q_minus + g.x, g.y);
            } else {
                q_minus_a[mi] = vec2<f32>(-g.x, -g.y);
                q_plus_a[mi] = g;
            }
        }
    }
}
