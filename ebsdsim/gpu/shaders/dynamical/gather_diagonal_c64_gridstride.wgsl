/*
 * Grid-strided copy of ebsdsim/wgsl/ebsd_gather_diagonal_c64.wgsl.
 * Identical math; bounded launch loops over (batch*n) so large k-tiles never
 * exceed the 65535 workgroups-per-dimension limit.
 */

const WG_SIZE: u32 = 256u;

struct GatherDiagonalParams {
    batch_count: u32,
    n_g: u32,
    n: u32,
    mode: u32,
    diag_imag: f32,
    mlambda: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: GatherDiagonalParams;
@group(0) @binding(1) var<storage, read> sg: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let total = params.batch_count * params.n;
    let stride = nwg.x * WG_SIZE;
    for (var linear = gid.x; linear < total; linear = linear + stride) {
        let b = linear / params.n;
        let g = idx[linear];
        let s = sg[b * params.n_g + g];
        if (params.mode == 0u) {
            let raw_re = 2.0 * s / params.mlambda;
            let raw_im = params.diag_imag;
            out[linear] = vec2<f32>(-3.141592653589793 * params.mlambda * raw_im, 3.141592653589793 * params.mlambda * raw_re);
        } else if (params.mode == 1u) {
            let raw_re = 2.0 * s;
            let raw_im = 1.0 / params.diag_imag;
            out[linear] = vec2<f32>(-3.141592653589793 * raw_im, 3.141592653589793 * raw_re);
        } else {
            out[linear] = vec2<f32>(0.0, 2.0 * 3.141592653589793 * s);
        }
    }
}
