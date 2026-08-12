const WG_SIZE: u32 = 256u;

struct HashDiffParams {
    batch_count: u32,
    n: u32,
    table_size: u32,
    _pad0: u32,
    offset: i32,
    _pad1: i32,
    _pad2: i32,
    _pad3: i32,
}

@group(0) @binding(0) var<uniform> params: HashDiffParams;
@group(0) @binding(1) var<storage, read> idx_a: array<u32>;
@group(0) @binding(2) var<storage, read> hkl_hash: array<i32>;
@group(0) @binding(3) var<storage, read_write> dh_out: array<u32>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    let elems_per_batch = params.n * params.n;
    let total = params.batch_count * elems_per_batch;
    if (linear >= total) { return; }

    let b = linear / elems_per_batch;
    let rem = linear - b * elems_per_batch;
    let i = rem / params.n;
    let j = rem - i * params.n;
    let idx_base = b * params.n;
    let hi = hkl_hash[idx_a[idx_base + i]];
    let hj = hkl_hash[idx_a[idx_base + j]];
    let dh = hj - hi + params.offset;
    if (dh >= 0 && u32(dh) < params.table_size) {
        dh_out[linear] = u32(dh);
    } else {
        dh_out[linear] = 0u;
    }
}
