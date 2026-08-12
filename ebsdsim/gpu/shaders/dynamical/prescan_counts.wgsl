const WG_SIZE: u32 = 256u;

struct PrescanCountParams {
    batch_count: u32,
    n_g: u32,
    _pad0: u32,
    _pad1: u32,
    bethe_c_cutoff: f32,
    dbdiff_sg_cutoff: f32,
    bethe_c_strong: f32,
    bethe_c_weak: f32,
}

fn dot_metric(a: vec3<f32>, m0: vec3<f32>, m1: vec3<f32>, m2: vec3<f32>, b: vec3<f32>) -> f32 {
    let mb = vec3<f32>(
        dot(m0, b),
        dot(m1, b),
        dot(m2, b),
    );
    return dot(a, mb);
}

@group(0) @binding(0) var<uniform> params: PrescanCountParams;
@group(0) @binding(1) var<storage, read> kvecs: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> hkl: array<vec3<i32>>;
@group(0) @binding(3) var<storage, read> metric: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read> coupling: array<f32>;
@group(0) @binding(5) var<storage, read> reflection_dbdiff: array<u32>;
// row_counts[row] = vec4u(strong, weak, candidate, 0)
@group(0) @binding(6) var<storage, read_write> row_counts: array<vec4<u32>>;

var<workgroup> strong_counts: array<u32, WG_SIZE>;
var<workgroup> weak_counts: array<u32, WG_SIZE>;
var<workgroup> candidate_counts: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let row = wg_id.x;
    if (row >= params.batch_count) {
        return;
    }

    let k = kvecs[row];
    let m0 = metric[0];
    let m1 = metric[1];
    let m2 = metric[2];

    var strong_local = 0u;
    var weak_local = 0u;
    var candidate_local = 0u;
    for (var g = lid; g < params.n_g; g = g + WG_SIZE) {
        let h = vec3<f32>(f32(hkl[g].x), f32(hkl[g].y), f32(hkl[g].z));
        let kpg = k + h;
        let tkpg = 2.0 * k + h;
        let xnom = -dot_metric(h, m0, m1, m2, tkpg);
        let q1 = sqrt(max(dot_metric(kpg, m0, m1, m2, kpg), 0.0));
        let klen = sqrt(max(dot_metric(k, m0, m1, m2, k), 0.0));
        let kpg_k = dot_metric(kpg, m0, m1, m2, k);
        let xden = 2.0 * q1 * kpg_k / (q1 * klen + 1e-16);
        var sg = xnom / (xden + 1e-16);
        if (g == 0u) {
            sg = 0.0;
        }

        let abs_sg = abs(sg);
        let ratio = abs_sg / max(coupling[g], 1e-12);
        var candidate = false;
        if (reflection_dbdiff[g] != 0u) {
            candidate = abs_sg <= params.dbdiff_sg_cutoff;
        } else {
            candidate = ratio <= params.bethe_c_cutoff;
        }
        var is_strong = candidate && ratio <= params.bethe_c_strong;
        let is_weak = candidate && ratio > params.bethe_c_strong && ratio <= params.bethe_c_weak;
        if (g == 0u) {
            candidate = true;
            is_strong = true;
        }

        if (candidate) {
            candidate_local = candidate_local + 1u;
        }
        if (is_strong) {
            strong_local = strong_local + 1u;
        }
        if (is_weak) {
            weak_local = weak_local + 1u;
        }
    }

    strong_counts[lid] = strong_local;
    weak_counts[lid] = weak_local;
    candidate_counts[lid] = candidate_local;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if (lid < stride) {
            strong_counts[lid] = strong_counts[lid] + strong_counts[lid + stride];
            weak_counts[lid] = weak_counts[lid] + weak_counts[lid + stride];
            candidate_counts[lid] = candidate_counts[lid] + candidate_counts[lid + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lid == 0u) {
        row_counts[row] = vec4<u32>(strong_counts[0], weak_counts[0], candidate_counts[0], 0u);
    }
}