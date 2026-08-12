const WG_SIZE: u32 = 256u;

struct OutputWritebackParams {
    batch_count: u32,
    n_sites: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: OutputWritebackParams;
@group(0) @binding(1) var<storage, read> chunk_values: array<f32>;
@group(0) @binding(2) var<storage, read> output_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    let total = params.batch_count * params.n_sites;
    if (linear >= total) { return; }
    let b = linear / params.n_sites;
    let site = linear - b * params.n_sites;
    output[output_indices[b] * params.n_sites + site] = chunk_values[linear];
}
