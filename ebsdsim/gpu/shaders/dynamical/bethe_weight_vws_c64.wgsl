const WG_SIZE: u32 = 256u;

struct BetheWeightParams {
    batch_count: u32,
    n_a: u32,
    n_w: u32,
    _pad0: u32,
    eps: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cx_conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

@group(0) @binding(0) var<uniform> params: BetheWeightParams;
@group(0) @binding(1) var<storage, read> v_sw: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> d_w: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> weighted_vws: array<vec2<f32>>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    let elems_per_batch = params.n_w * params.n_a;
    let total = params.batch_count * elems_per_batch;
    if (linear >= total) { return; }

    let b = linear / elems_per_batch;
    let rem = linear - b * elems_per_batch;
    let w = rem / params.n_a;
    let a_col = rem - w * params.n_a;

    let d = d_w[b * params.n_w + w];
    let mag2 = max(d.x * d.x + d.y * d.y, params.eps * params.eps);
    let dinv = vec2<f32>(d.x / mag2, -d.y / mag2);
    let vsw = v_sw[b * params.n_a * params.n_w + a_col * params.n_w + w];
    weighted_vws[linear] = cx_mul(cx_conj(vsw), dinv);
}
