/*
 * Grid-strided copy of ebsdsim/wgsl/ebsd_excitation_score.wgsl.
 * Identical math; a bounded launch loops over (batch*n_g) so large k-tiles never
 * exceed the 65535 workgroups-per-dimension limit. Backward compatible: with a
 * full-size launch the stride loop runs exactly once per thread.
 */

const WG_SIZE: u32 = 256u;
const SCORE_WORST: f32 = 0x1.fffffep+127f;

struct ExcitationScoreParams {
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
    let mb = vec3<f32>(dot(m0, b), dot(m1, b), dot(m2, b));
    return dot(a, mb);
}

@group(0) @binding(0) var<uniform> params: ExcitationScoreParams;
@group(0) @binding(1) var<storage, read> kvecs: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> hkl: array<vec3<i32>>;
@group(0) @binding(3) var<storage, read> metric: array<vec3<f32>>;
@group(0) @binding(4) var<storage, read> coupling: array<f32>;
@group(0) @binding(5) var<storage, read> reflection_dbdiff: array<u32>;
@group(0) @binding(6) var<storage, read_write> sg_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> score_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> candidate_out: array<u32>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let total = params.batch_count * params.n_g;
    let stride = nwg.x * WG_SIZE;
    let m0 = metric[0];
    let m1 = metric[1];
    let m2 = metric[2];
    for (var linear = gid.x; linear < total; linear = linear + stride) {
        let b = linear / params.n_g;
        let g = linear - b * params.n_g;
        let k = kvecs[b];
        let h = vec3<f32>(f32(hkl[g].x), f32(hkl[g].y), f32(hkl[g].z));

        let kpg = k + h;
        let tkpg = 2.0 * k + h;
        let xnom = -dot_metric(h, m0, m1, m2, tkpg);
        let q1 = sqrt(max(dot_metric(kpg, m0, m1, m2, kpg), 0.0));
        let klen = sqrt(max(dot_metric(k, m0, m1, m2, k), 0.0));
        let kpg_k = dot_metric(kpg, m0, m1, m2, k);
        let xden = 2.0 * q1 * kpg_k / (q1 * klen + 1e-16);
        var sg = xnom / (xden + 1e-16);
        if (g == 0u) { sg = 0.0; }

        let idx = b * params.n_g + g;
        sg_out[idx] = sg;

        let abs_sg = abs(sg);
        let ratio = abs_sg / max(coupling[g], 1e-12);
        var candidate = false;
        if (reflection_dbdiff[g] != 0u) {
            candidate = abs_sg <= params.dbdiff_sg_cutoff;
        } else {
            candidate = ratio <= params.bethe_c_cutoff;
        }
        if (g == 0u) { candidate = true; }
        candidate_out[idx] = select(0u, 1u, candidate);
        if (candidate) {
            score_out[idx] = select(ratio, 0.0, g == 0u);
        } else {
            score_out[idx] = SCORE_WORST;
        }
    }
}
