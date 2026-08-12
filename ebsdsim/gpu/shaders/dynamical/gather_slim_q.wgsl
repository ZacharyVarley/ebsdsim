// Fused gather_diagonal + smith_slim_q (one WG per k).
// Writes d_a from sg[idx], then assembles e0 / q_values like ebsd_smith_slim_q.
//
// Dispatch: (batch_count, 1, 1), @workgroup_size(256)
// Params packing: <4I4i8f
//   dims: batch, n, table_size, n_g
//   hash: stride_h, stride_k, offset, mode (0=bloch)
//   phys: pref.re, pref.im, mu_shift, eps, diag_imag, mlambda, pad, pad

struct Params {
    dims: vec4<u32>,
    hash: vec4<i32>,
    phys: vec4<f32>,
    gather: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> idx_a: array<u32>;
@group(0) @binding(2) var<storage, read> sg: array<f32>;
@group(0) @binding(3) var<storage, read> hkl: array<i32>;
@group(0) @binding(4) var<storage, read> diff_table: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> d_a: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> q_values: array<f32>;
@group(0) @binding(7) var<storage, read_write> e0: array<vec2<f32>>;

const WG_SIZE: u32 = 256u;
const PI: f32 = 3.141592653589793;

var<workgroup> reduce_fro: array<f32, 256>;
var<workgroup> reduce_decay: array<f32, 256>;
var<workgroup> incident_slot: atomic<u32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn norm2(z: vec2<f32>) -> f32 {
    return z.x * z.x + z.y * z.y;
}

fn table_index(hi: vec3<i32>, hj: vec3<i32>) -> i32 {
    let d = hi - hj;
    return d.x * params.hash.x + d.y * params.hash.y + d.z + params.hash.z;
}

fn load_hkl(global_ref: u32) -> vec3<i32> {
    let base = global_ref * 4u;
    return vec3<i32>(hkl[base], hkl[base + 1u], hkl[base + 2u]);
}

fn gather_diag(s: f32) -> vec2<f32> {
    let mode = u32(params.hash.w);
    let diag_imag = params.gather.x;
    let mlambda = params.gather.y;
    if (mode == 0u) {
        let raw_re = 2.0 * s / mlambda;
        let raw_im = diag_imag;
        return vec2<f32>(-PI * mlambda * raw_im, PI * mlambda * raw_re);
    } else if (mode == 1u) {
        let raw_re = 2.0 * s;
        let raw_im = 1.0 / diag_imag;
        return vec2<f32>(-PI * raw_im, PI * raw_re);
    }
    return vec2<f32>(0.0, 2.0 * PI * s);
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
    let table_size = params.dims.z;
    let n_g = params.dims.w;

    if (b >= batch_count) {
        return;
    }

    if (lid == 0u) {
        atomicStore(&incident_slot, 0xffffffffu);
    }
    workgroupBarrier();

    let idx_base = b * n;
    let sg_base = b * n_g;
    let pref = params.phys.xy;

    // Gather d_a and locate incident beam.
    for (var i = lid; i < n; i = i + WG_SIZE) {
        let g = idx_a[idx_base + i];
        d_a[idx_base + i] = gather_diag(sg[sg_base + g]);
        if (g == 0u) {
            atomicMin(&incident_slot, i);
        }
    }
    workgroupBarrier();

    var slot = atomicLoad(&incident_slot);
    if (slot == 0xffffffffu) {
        slot = 0u;
    }

    for (var i = lid; i < n; i = i + WG_SIZE) {
        e0[idx_base + i] = select(
            vec2<f32>(0.0, 0.0),
            vec2<f32>(1.0, 0.0),
            i == slot,
        );
    }

    var fro_sum = 0.0;
    var fro_corr = 0.0;
    var decay_sum = 0.0;
    var decay_corr = 0.0;

    let total = n * n;
    for (var linear = lid; linear < total; linear = linear + WG_SIZE) {
        let i = linear / n;
        let j = linear - i * n;
        var gij: vec2<f32>;

        if (i == j) {
            gij = d_a[idx_base + i];
            let y_decay = (-gij.x) - decay_corr;
            let t_decay = decay_sum + y_decay;
            decay_corr = (t_decay - decay_sum) - y_decay;
            decay_sum = t_decay;
        } else {
            let hi = load_hkl(idx_a[idx_base + i]);
            let hj = load_hkl(idx_a[idx_base + j]);
            let ti = table_index(hi, hj);
            var u = vec2<f32>(0.0, 0.0);
            if (ti >= 0 && u32(ti) < table_size) {
                u = diff_table[u32(ti)];
            }
            gij = cmul(pref, u);
        }

        let term = norm2(gij);
        let y_fro = term - fro_corr;
        let t_fro = fro_sum + y_fro;
        fro_corr = (t_fro - fro_sum) - y_fro;
        fro_sum = t_fro;
    }

    reduce_fro[lid] = fro_sum;
    reduce_decay[lid] = decay_sum;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if (lid < stride) {
            reduce_fro[lid] = reduce_fro[lid] + reduce_fro[lid + stride];
            reduce_decay[lid] = reduce_decay[lid] + reduce_decay[lid + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lid == 0u) {
        let n_f = f32(n);
        let fro_per_dim = sqrt(max(reduce_fro[0], 0.0)) / n_f;
        let mean_decay = reduce_decay[0] / n_f + 0.5 * params.phys.z;
        let eps = params.phys.w;
        let q = sqrt(max(fro_per_dim * max(mean_decay, eps), eps));
        let q_base = b * 4u;
        q_values[q_base] = q;
        q_values[q_base + 1u] = q + 0.5 * params.phys.z;
        q_values[q_base + 2u] = q - 0.5 * params.phys.z;
        q_values[q_base + 3u] = sqrt(2.0 * q);
    }
}
