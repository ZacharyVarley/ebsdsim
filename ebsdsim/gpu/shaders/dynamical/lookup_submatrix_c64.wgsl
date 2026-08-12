const WG_SIZE: u32 = 256u;

struct LookupSubmatrixParams {
    batch_count: u32,
    n_rows: u32,
    n_cols: u32,
    table_size: u32,
    offset: i32,
    prefactor: vec2<f32>,
    zero_diagonal: u32,
    _pad0: u32,
}

fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@group(0) @binding(0) var<uniform> params: LookupSubmatrixParams;
@group(0) @binding(1) var<storage, read> idx_rows: array<u32>;
@group(0) @binding(2) var<storage, read> idx_cols: array<u32>;
@group(0) @binding(3) var<storage, read> hkl_hash: array<i32>;
@group(0) @binding(4) var<storage, read> table: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    let elems_per_batch = params.n_rows * params.n_cols;
    let total = params.batch_count * elems_per_batch;
    if (linear >= total) { return; }

    let b = linear / elems_per_batch;
    let rem = linear - b * elems_per_batch;
    let row = rem / params.n_cols;
    let col = rem - row * params.n_cols;

    if (params.zero_diagonal != 0u && row == col) {
        out[linear] = vec2<f32>(0.0, 0.0);
        return;
    }

    let row_ref = idx_rows[b * params.n_rows + row];
    let col_ref = idx_cols[b * params.n_cols + col];
    let dh = hkl_hash[row_ref] - hkl_hash[col_ref] + params.offset;
    if (dh < 0 || u32(dh) >= params.table_size) {
        out[linear] = vec2<f32>(0.0, 0.0);
        return;
    }
    out[linear] = cx_mul(params.prefactor, table[u32(dh)]);
}
