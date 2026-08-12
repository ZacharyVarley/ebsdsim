const TILE_M: u32 = 8u;
const TILE_N: u32 = 8u;

struct GemmC64Params {
    batch_count: u32,
    m: u32,
    n: u32,
    k: u32,
    a_stride: u32,
    b_stride: u32,
    c_stride: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    alpha: vec2<f32>,
    beta: vec2<f32>,
}

fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cx_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x + b.x, a.y + b.y);
}

@group(0) @binding(0) var<uniform> params: GemmC64Params;
@group(0) @binding(1) var<storage, read> a: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> c: array<vec2<f32>>;

@compute @workgroup_size(TILE_N, TILE_M, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let batch = wg_id.z;
    if (batch >= params.batch_count) { return; }

    let row = wg_id.y * TILE_M + lid.y;
    let col = wg_id.x * TILE_N + lid.x;
    if (row >= params.m || col >= params.n) { return; }

    let a_base = batch * params.a_stride;
    let b_base = batch * params.b_stride;
    let c_base = batch * params.c_stride;

    var acc = vec2<f32>(0.0, 0.0);
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        let av = a[a_base + row * params.lda + kk];
        let bv = b[b_base + kk * params.ldb + col];
        acc = cx_add(acc, cx_mul(av, bv));
    }

    let ci = c_base + row * params.ldc + col;
    let old = c[ci];
    c[ci] = cx_add(cx_mul(params.alpha, acc), cx_mul(params.beta, old));
}
