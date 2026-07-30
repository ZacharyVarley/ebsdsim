// Fused hash + intensity contract + optional global writeback.
//
// Reorders the contraction as
//   sum_(i,j) Re[S_ij * sum_r conj(w_i,r) w_j,r]
// so each Sgh coefficient is loaded once per (i,j,site), rather than once per
// (i,j,site,rank). No dh matrix is materialized.
//
// Dispatch: (batch_count, 1, 1), @workgroup_size(256)
// Limits: n <= 2048 (matches Smith iterative ceiling), rank <= 16, n_sites <= 16.
// Shared: 3*2048*i32 + 16*256*f32 = 40960 B (< 48 KiB).
// Params packing: <4I4i4I4f

struct Params {
    // x=batch_count, y=n, z=rank, w=n_sites
    dims: vec4<u32>,
    // x=stride_h, y=stride_k, z=offset, w=table_size
    hash: vec4<i32>,
    // x=global_output_site_stride, y=write_global, z=sgh_delta_major, w=unused
    out: vec4<u32>,
    // x=amplitude, y/z/w=unused
    phys: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> idx_a: array<u32>;
@group(0) @binding(2) var<storage, read> hkl: array<i32>;
@group(0) @binding(3) var<storage, read> w_stack: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> sgh_tables: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> output_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> intensities: array<f32>;
@group(0) @binding(7) var<storage, read_write> global_output: array<f32>;

const WG_SIZE: u32 = 256u;
const MAX_N: u32 = 2048u;
const MAX_RANK: u32 = 16u;
const MAX_SITES: u32 = 16u;

var<workgroup> local_h: array<i32, 2048>;
var<workgroup> local_k: array<i32, 2048>;
var<workgroup> local_l: array<i32, 2048>;
var<workgroup> partial: array<f32, 4096>; // MAX_SITES * WG_SIZE

fn load_selected_hkl(batch_base: u32, i: u32) -> vec3<i32> {
    let global_ref = idx_a[batch_base + i];
    let base = global_ref * 4u;
    return vec3<i32>(hkl[base], hkl[base + 1u], hkl[base + 2u]);
}

fn diff_index(i: u32, j: u32) -> i32 {
    // Preserve the production contract sign: h_j - h_i + offset.
    let dh = local_h[j] - local_h[i];
    let dk = local_k[j] - local_k[i];
    let dl = local_l[j] - local_l[i];
    return dh * params.hash.x + dk * params.hash.y + dl + params.hash.z;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let b = workgroup_id.x;
    let lid = local_id.x;
    let batch_count = params.dims.x;
    let n = params.dims.y;
    let rank = params.dims.z;
    let n_sites = params.dims.w;
    let table_size = u32(params.hash.w);

    if (b >= batch_count || n > MAX_N || rank > MAX_RANK || n_sites > MAX_SITES) {
        return;
    }

    let idx_base = b * n;
    for (var i = lid; i < n; i = i + WG_SIZE) {
        let hv = load_selected_hkl(idx_base, i);
        local_h[i] = hv.x;
        local_k[i] = hv.y;
        local_l[i] = hv.z;
    }
    workgroupBarrier();

    var accum: array<f32, 16>;
    for (var s = 0u; s < MAX_SITES; s = s + 1u) {
        accum[s] = 0.0;
    }

    let pair_count = n * n;
    for (var linear = lid; linear < pair_count; linear = linear + WG_SIZE) {
        let i = linear / n;
        let j = linear - i * n;
        let wi_base = (b * n + i) * rank;
        let wj_base = (b * n + j) * rank;

        // Gram element C_ij = sum_r conj(w_i,r) * w_j,r.
        var gram_re = 0.0;
        var gram_im = 0.0;
        for (var r = 0u; r < rank; r = r + 1u) {
            let wi = w_stack[wi_base + r];
            let wj = w_stack[wj_base + r];
            gram_re = gram_re + wi.x * wj.x + wi.y * wj.y;
            gram_im = gram_im + wi.x * wj.y - wi.y * wj.x;
        }

        let ti = diff_index(i, j);
        if (ti >= 0 && u32(ti) < table_size) {
            let table_offset = u32(ti);
            for (var s = 0u; s < n_sites; s = s + 1u) {
                // Delta-major layout makes all site values for one pair contiguous.
                // Keep site-major as the compatibility path.
                var sh_index = s * table_size + table_offset;
                if (params.out.z != 0u) {
                    sh_index = table_offset * n_sites + s;
                }
                let sh = sgh_tables[sh_index];
                accum[s] = accum[s] + sh.x * gram_re - sh.y * gram_im;
            }
        }
    }

    // Reduce every site in parallel using the same eight barriers.
    for (var s = 0u; s < n_sites; s = s + 1u) {
        partial[s * WG_SIZE + lid] = accum[s];
    }
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if (lid < stride) {
            for (var s = 0u; s < n_sites; s = s + 1u) {
                let p = s * WG_SIZE + lid;
                partial[p] = partial[p] + partial[p + stride];
            }
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lid == 0u) {
        let local_out_base = b * n_sites;
        let global_k = output_indices[b];
        let global_out_base = global_k * params.out.x;
        for (var s = 0u; s < n_sites; s = s + 1u) {
            let value = params.phys.x * partial[s * WG_SIZE];
            intensities[local_out_base + s] = value;
            // Caller owns the per-batch write; optional global amp-add when out.y != 0.
            if (params.out.y != 0u) {
                global_output[global_out_base + s] =
                    global_output[global_out_base + s] + value;
            }
        }
    }
}
