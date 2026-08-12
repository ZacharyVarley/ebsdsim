const WG_SIZE: u32 = 256u;

struct GatherSghParams {
    batch_count: u32,
    n: u32,
    table_size: u32,
    site: u32,
}

@group(0) @binding(0) var<uniform> params: GatherSghParams;
@group(0) @binding(1) var<storage, read> dh: array<u32>;
@group(0) @binding(2) var<storage, read> sgh_tables: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> sgh_out: array<vec2<f32>>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    let total = params.batch_count * params.n * params.n;
    if (linear >= total) { return; }
    let key = dh[linear];
    if (key < params.table_size) {
        sgh_out[linear] = sgh_tables[params.site * params.table_size + key];
    } else {
        sgh_out[linear] = vec2<f32>(0.0, 0.0);
    }
}
