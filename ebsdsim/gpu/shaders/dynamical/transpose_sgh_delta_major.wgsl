// One-time crystal-level Sgh transpose.
// Input:  site-major  src[site * table_size + delta]
// Output: delta-major dst[delta * n_sites + site]
// Dispatch: ceil(table_size*n_sites / 256), 1, 1

struct Params {
    // x=table_size, y=n_sites, z/w=unused
    dims: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    let table_size = params.dims.x;
    let n_sites = params.dims.y;
    let total = table_size * n_sites;
    if (linear >= total) {
        return;
    }
    let site = linear / table_size;
    let delta = linear - site * table_size;
    dst[delta * n_sites + site] = src[linear];
}
