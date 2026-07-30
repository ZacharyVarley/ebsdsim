/*
 * Unique-Δ shared BiCGSTAB Smith.
 *
 * Same LLL pack ladder: resident / unique-Δ / tile. Krylov is BiCGSTAB on
 * A = S Q- S (Hermitian products) for odd U.
 *
 * stats bit28 = unique-Δ (resident or segment), bit29 = BiCGSTAB,
 * bit30 = unique-segment TILE, bit31 = dense TILE, 0xFFFFFFFF = fail.
 * Host string-replaces MAX_N / MAX_PACK / MAX_UNIQ / BPT for beam buckets.
 */

enable f16;

const WG: u32 = 256u;
const MAX_N: u32 = 384u;
const MAX_PACK: u32 = 10200u;
const MAX_UNIQ: u32 = 8600u;
const BPT: u32 = 2u;
// Dense/unique-seg bitset may cover up to this AABB length (global meta).
const MAX_PLEN_CAP: u32 = 65536u;
const MAX_UWORDS: u32 = 2048u; // ceil(MAX_PLEN_CAP / 32)
// Resident unique-Δ still packs bitset+prefix into shared spack tail; that
// layout only fits ~20k plen words (625 u32) alongside MAX_UNIQ values.
const MAX_PLEN_SHARED_META: u32 = 20000u;
const MAX_UWORDS_SHARED: u32 = 625u;
const META_BITS_BASE: u32 = MAX_UNIQ;
// Two u32 words → six exact f16 chunks → three vec2<f16> slots (shared meta).
const META_BITS_SLOTS: u32 = 939u;
const META_PREFIX_BASE: u32 = META_BITS_BASE + META_BITS_SLOTS;
// Unique-segment TILE window. With global bitset/prefix, SEG may be MAX_PACK
// (entire shared spack is values). Host usually sets this to the bucket MAX_PACK.
const UNIQUE_SEG_TILE: u32 = 8600u;
// Set to 1u via host replace to force segment TILE even when ν fits.
const FORCE_UNIQUE_SEG_TILE: u32 = 0u;
// Global compacted unique values (one-time fill; segment loads are coherent).
const MAX_NU_CAP: u32 = 16384u;
// 1 = load segments from uniq_vals (fast); 0 = legacy rescan plen (slow).
const USE_GLOBAL_UNIQ_VALS: u32 = 1u;
// Set to 1u to force dense AABB TILE (skip unique paths) for benchmarks.
const FORCE_DENSE_TILE: u32 = 0u;

struct Params {
    batch_count: u32,
    n: u32,
    rank: u32,
    max_iter: u32,
    atol: f32,
    table_size: u32,
    offset: i32,
    stride_h: i32,
    stride_k: i32,
    _pad0: u32,
    pref: vec2<f32>,
    bl00: f32, bl01: f32, bl02: f32,
    bl10: f32, bl11: f32, bl12: f32,
    bl20: f32, bl21: f32, bl22: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> hkl: array<i32>;
@group(0) @binding(2) var<storage, read> table: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> idx_a: array<u32>;
@group(0) @binding(4) var<storage, read> d_a: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> e0: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> q_values: array<f32>;
@group(0) @binding(7) var<storage, read_write> w_stack: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> stats: array<u32>;
// One-time build scratch only: batch_count * MAX_UWORDS atomic words.
@group(0) @binding(9) var<storage, read_write> uniq_bits: array<atomic<u32>>;
// batch * MAX_NU_CAP compacted U(Δ) for unique-segment TILE.
@group(0) @binding(10) var<storage, read_write> uniq_vals: array<vec2<f16>>;
// Per batch: [0..MAX_UWORDS) plain bitwords, [MAX_UWORDS..2*MAX_UWORDS) prefixes.
@group(0) @binding(11) var<storage, read_write> uniq_meta: array<u32>;

var<workgroup> spack: array<vec2<f16>, MAX_PACK>;
var<workgroup> ss: array<i32, MAX_N>;
var<workgroup> sord: array<u32, MAX_N>;
var<workgroup> sp: array<vec2<f32>, MAX_N>;
var<workgroup> red_re: array<f32, WG>;
var<workgroup> red_im: array<f32, WG>;
var<workgroup> geom: array<f32, 32>;
var<workgroup> g_nu: u32;
var<workgroup> g_mode: u32; // 0=resident, 1=uniq, 2=tiled safety fallback

fn cx_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return a + b; }
fn cx_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return a - b; }
fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn cx_scale(a: vec2<f32>, s: f32) -> vec2<f32> { return a * s; }
fn cx_conj(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x, -a.y); }
fn cx_abs2(a: vec2<f32>) -> f32 { return a.x * a.x + a.y * a.y; }
fn cx_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = max(cx_abs2(b), 1.0e-30);
    return vec2<f32>((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

// Callers issue reductions back to back with no barrier in between, so the
// leading fence is required: without it a fast subgroup can overwrite red_re[0]
// while a slow one is still reading the previous reduction's result.
fn reduce_sum_f(lid: u32, val: f32) -> f32 {
    workgroupBarrier();
    red_re[lid] = val;
    workgroupBarrier();
    var stride = WG / 2u;
    loop {
        if (lid < stride) { red_re[lid] = red_re[lid] + red_re[lid + stride]; }
        workgroupBarrier();
        if (stride <= 1u) { break; }
        stride = stride / 2u;
    }
    return red_re[0];
}

fn reduce_sum_c(lid: u32, val: vec2<f32>) -> vec2<f32> {
    workgroupBarrier();  // see reduce_sum_f
    red_re[lid] = val.x;
    red_im[lid] = val.y;
    workgroupBarrier();
    var stride = WG / 2u;
    loop {
        if (lid < stride) {
            red_re[lid] = red_re[lid] + red_re[lid + stride];
            red_im[lid] = red_im[lid] + red_im[lid + stride];
        }
        workgroupBarrier();
        if (stride <= 1u) { break; }
        stride = stride / 2u;
    }
    return vec2<f32>(red_re[0], red_im[0]);
}

fn load_u(idx: u32) -> vec2<f32> {
    let v = spack[idx];
    return vec2<f32>(f32(v.x), f32(v.y));
}

fn store_bitword(wi: u32, word: u32) {
    let c0 = f16(word & 0x7ffu);
    let c1 = f16((word >> 11u) & 0x7ffu);
    let c2 = f16((word >> 22u) & 0x3ffu);
    let base = META_BITS_BASE + (wi >> 1u) * 3u;
    if ((wi & 1u) == 0u) {
        spack[base] = vec2<f16>(c0, c1);
        let bridge = spack[base + 1u];
        spack[base + 1u] = vec2<f16>(c2, bridge.y);
    } else {
        let bridge = spack[base + 1u];
        spack[base + 1u] = vec2<f16>(bridge.x, c0);
        spack[base + 2u] = vec2<f16>(c1, c2);
    }
}

fn load_bitword(wi: u32) -> u32 {
    let base = META_BITS_BASE + (wi >> 1u) * 3u;
    var c0: u32;
    var c1: u32;
    var c2: u32;
    if ((wi & 1u) == 0u) {
        let a = spack[base];
        let b = spack[base + 1u];
        c0 = u32(f32(a.x)); c1 = u32(f32(a.y)); c2 = u32(f32(b.x));
    } else {
        let a = spack[base + 1u];
        let b = spack[base + 2u];
        c0 = u32(f32(a.y)); c1 = u32(f32(b.x)); c2 = u32(f32(b.y));
    }
    return c0 | (c1 << 11u) | (c2 << 22u);
}

fn store_prefix(wi: u32, prefix: u32) {
    spack[META_PREFIX_BASE + wi] = vec2<f16>(
        f16(prefix & 0xffu), f16((prefix >> 8u) & 0xffu)
    );
}

fn load_prefix(wi: u32) -> u32 {
    let p = spack[META_PREFIX_BASE + wi];
    return u32(f32(p.x)) | (u32(f32(p.y)) << 8u);
}

fn unique_rank(idx: u32) -> u32 {
    let wi = idx >> 5u;
    let bit = idx & 31u;
    let word = load_bitword(wi);
    let prefix = load_prefix(wi);
    return prefix + countOneBits(word & ((1u << bit) - 1u));
}

fn unique_rank_g(batch: u32, idx: u32) -> u32 {
    let wi = idx >> 5u;
    let bit = idx & 31u;
    let base = batch * (MAX_UWORDS * 2u);
    let word = uniq_meta[base + wi];
    let prefix = uniq_meta[base + MAX_UWORDS + wi];
    return prefix + countOneBits(word & ((1u << bit) - 1u));
}

fn inv3(
    a00: f32, a01: f32, a02: f32,
    a10: f32, a11: f32, a12: f32,
    a20: f32, a21: f32, a22: f32,
) -> array<f32, 9> {
    let c00 = a11 * a22 - a12 * a21;
    let c01 = a02 * a21 - a01 * a22;
    let c02 = a01 * a12 - a02 * a11;
    let c10 = a12 * a20 - a10 * a22;
    let c11 = a00 * a22 - a02 * a20;
    let c12 = a02 * a10 - a00 * a12;
    let c20 = a10 * a21 - a11 * a20;
    let c21 = a01 * a20 - a00 * a21;
    let c22 = a00 * a11 - a01 * a10;
    // Preserve sign: max(det, eps) breaks orientation-reversing LLL bases (det < 0)
    // and blows up Minv → empty pack → diagonal-only BiCGSTAB (iters = rank).
    let det = a00 * c00 + a01 * c10 + a02 * c20;
    let inv_d = 1.0 / (sign(det) * max(abs(det), 1.0e-20));
    return array<f32, 9>(
        c00 * inv_d, c01 * inv_d, c02 * inv_d,
        c10 * inv_d, c11 * inv_d, c12 * inv_d,
        c20 * inv_d, c21 * inv_d, c22 * inv_d,
    );
}

fn beam_hkl(n_base: u32, i: u32) -> vec3<f32> {
    // Persistent hkl is i32 vec3 packed as vec4 (stride 4), matching production buffers.
    let ref_i = idx_a[n_base + i];
    let base = ref_i * 4u;
    return vec3<f32>(
        f32(hkl[base]),
        f32(hkl[base + 1u]),
        f32(hkl[base + 2u]),
    );
}

fn build_smith_iterative_geometry(n: u32, n_base: u32) {
    var mx = 0.0; var my = 0.0; var mz = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let h = beam_hkl(n_base, i);
        mx = mx + h.x; my = my + h.y; mz = mz + h.z;
    }
    let inv_n = 1.0 / f32(n);
    mx = mx * inv_n; my = my * inv_n; mz = mz * inv_n;

    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0;
    var c11 = 0.0; var c12 = 0.0; var c22 = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let h = beam_hkl(n_base, i);
        let dx = h.x - mx; let dy = h.y - my; let dz = h.z - mz;
        c00 = c00 + dx * dx; c01 = c01 + dx * dy; c02 = c02 + dx * dz;
        c11 = c11 + dy * dy; c12 = c12 + dy * dz; c22 = c22 + dz * dz;
    }
    c00 = c00 * inv_n + 1.0e-9;
    c01 = c01 * inv_n; c02 = c02 * inv_n;
    c11 = c11 * inv_n + 1.0e-9; c12 = c12 * inv_n;
    c22 = c22 * inv_n + 1.0e-9;

    let bli = inv3(
        params.bl00, params.bl01, params.bl02,
        params.bl10, params.bl11, params.bl12,
        params.bl20, params.bl21, params.bl22,
    );
    let b00 = bli[0]; let b01 = bli[3]; let b02 = bli[6];
    let b10 = bli[1]; let b11 = bli[4]; let b12 = bli[7];
    let b20 = bli[2]; let b21 = bli[5]; let b22 = bli[8];

    let t00 = c00*b00 + c01*b01 + c02*b02;
    let t01 = c00*b10 + c01*b11 + c02*b12;
    let t02 = c00*b20 + c01*b21 + c02*b22;
    let t10 = c01*b00 + c11*b01 + c12*b02;
    let t11 = c01*b10 + c11*b11 + c12*b12;
    let t12 = c01*b20 + c11*b21 + c12*b22;
    let t20 = c02*b00 + c12*b01 + c22*b02;
    let t21 = c02*b10 + c12*b11 + c22*b12;
    let t22 = c02*b20 + c12*b21 + c22*b22;

    let g00 = b00*t00 + b01*t10 + b02*t20;
    let g01 = b00*t01 + b01*t11 + b02*t21;
    let g02 = b00*t02 + b01*t12 + b02*t22;
    let g10 = b10*t00 + b11*t10 + b12*t20;
    let g11 = b10*t01 + b11*t11 + b12*t21;
    let g12 = b10*t02 + b11*t12 + b12*t22;
    let g20 = b20*t00 + b21*t10 + b22*t20;
    let g21 = b20*t01 + b21*t11 + b22*t21;
    let g22 = b20*t02 + b21*t12 + b22*t22;

    var u00 = 1.0; var u01 = 0.0; var u02 = 0.0;
    var u10 = 0.0; var u11 = 1.0; var u12 = 0.0;
    var u20 = 0.0; var u21 = 0.0; var u22 = 1.0;
    let delta = 0.99;
    var k = 1;
    var it = 0;
    loop {
        if (k >= 3 || it >= 200) { break; }
        it = it + 1;

        // Size-reduce row k against j = k-1 .. 0 (recompute GSO mu each time)
        var j = k - 1;
        loop {
            // GSO: Bs rows
            var bs00 = u00; var bs01 = u01; var bs02 = u02;
            var bs10 = u10; var bs11 = u11; var bs12 = u12;
            var bs20 = u20; var bs21 = u21; var bs22 = u22;
            let ip0 = bs00*(g00*bs00+g01*bs01+g02*bs02) + bs01*(g10*bs00+g11*bs01+g12*bs02) + bs02*(g20*bs00+g21*bs01+g22*bs02);
            let mu10 = (u10*(g00*bs00+g01*bs01+g02*bs02) + u11*(g10*bs00+g11*bs01+g12*bs02) + u12*(g20*bs00+g21*bs01+g22*bs02)) / max(ip0, 1.0e-30);
            bs10 = bs10 - mu10 * bs00; bs11 = bs11 - mu10 * bs01; bs12 = bs12 - mu10 * bs02;
            let ip1 = bs10*(g00*bs10+g01*bs11+g02*bs12) + bs11*(g10*bs10+g11*bs11+g12*bs12) + bs12*(g20*bs10+g21*bs11+g22*bs12);
            let mu20 = (u20*(g00*bs00+g01*bs01+g02*bs02) + u21*(g10*bs00+g11*bs01+g12*bs02) + u22*(g20*bs00+g21*bs01+g22*bs02)) / max(ip0, 1.0e-30);
            let mu21 = (u20*(g00*bs10+g01*bs11+g02*bs12) + u21*(g10*bs10+g11*bs11+g12*bs12) + u22*(g20*bs10+g21*bs11+g22*bs12)) / max(ip1, 1.0e-30);
            var mu: f32;
            if (k == 1) { mu = mu10; }
            else if (j == 0) { mu = mu20; }
            else { mu = mu21; }
            let q = round(mu);
            if (abs(q) > 0.0) {
                if (k == 1) {
                    u10 = u10 - q * u00; u11 = u11 - q * u01; u12 = u12 - q * u02;
                } else if (j == 0) {
                    u20 = u20 - q * u00; u21 = u21 - q * u01; u22 = u22 - q * u02;
                } else {
                    u20 = u20 - q * u10; u21 = u21 - q * u11; u22 = u22 - q * u12;
                }
            }
            if (j == 0) { break; }
            j = j - 1;
        }

        // Lovasz on GSO lengths
        var bs00 = u00; var bs01 = u01; var bs02 = u02;
        var bs10 = u10; var bs11 = u11; var bs12 = u12;
        var bs20 = u20; var bs21 = u21; var bs22 = u22;
        let ip0 = bs00*(g00*bs00+g01*bs01+g02*bs02) + bs01*(g10*bs00+g11*bs01+g12*bs02) + bs02*(g20*bs00+g21*bs01+g22*bs02);
        let mu10 = (u10*(g00*bs00+g01*bs01+g02*bs02) + u11*(g10*bs00+g11*bs01+g12*bs02) + u12*(g20*bs00+g21*bs01+g22*bs02)) / max(ip0, 1.0e-30);
        bs10 = bs10 - mu10 * bs00; bs11 = bs11 - mu10 * bs01; bs12 = bs12 - mu10 * bs02;
        let ip1 = bs10*(g00*bs10+g01*bs11+g02*bs12) + bs11*(g10*bs10+g11*bs11+g12*bs12) + bs12*(g20*bs10+g21*bs11+g22*bs12);
        let mu20 = (u20*(g00*bs00+g01*bs01+g02*bs02) + u21*(g10*bs00+g11*bs01+g12*bs02) + u22*(g20*bs00+g21*bs01+g22*bs02)) / max(ip0, 1.0e-30);
        let mu21 = (u20*(g00*bs10+g01*bs11+g02*bs12) + u21*(g10*bs10+g11*bs11+g12*bs12) + u22*(g20*bs10+g21*bs11+g22*bs12)) / max(ip1, 1.0e-30);
        bs20 = bs20 - mu20 * bs00 - mu21 * bs10;
        bs21 = bs21 - mu20 * bs01 - mu21 * bs11;
        bs22 = bs22 - mu20 * bs02 - mu21 * bs12;
        let ip2 = bs20*(g00*bs20+g01*bs21+g02*bs22) + bs21*(g10*bs20+g11*bs21+g12*bs22) + bs22*(g20*bs20+g21*bs21+g22*bs22);

        var lovasz_ok: bool;
        if (k == 1) {
            lovasz_ok = ip1 >= (delta - mu10 * mu10) * ip0;
        } else {
            lovasz_ok = ip2 >= (delta - mu21 * mu21) * ip1;
        }
        if (lovasz_ok) {
            k = k + 1;
        } else {
            if (k == 1) {
                let z0 = u00; let z1 = u01; let z2 = u02;
                u00 = u10; u01 = u11; u02 = u12;
                u10 = z0; u11 = z1; u12 = z2;
            } else {
                let z0 = u10; let z1 = u11; let z2 = u12;
                u10 = u20; u11 = u21; u12 = u22;
                u20 = z0; u21 = z1; u22 = z2;
            }
            k = max(k - 1, 1);
        }
    }

    let m00 = u00*b00 + u01*b10 + u02*b20;
    let m01 = u00*b01 + u01*b11 + u02*b21;
    let m02 = u00*b02 + u01*b12 + u02*b22;
    let m10 = u10*b00 + u11*b10 + u12*b20;
    let m11 = u10*b01 + u11*b11 + u12*b21;
    let m12 = u10*b02 + u11*b12 + u12*b22;
    let m20 = u20*b00 + u21*b10 + u22*b20;
    let m21 = u20*b01 + u21*b11 + u22*b21;
    let m22 = u20*b02 + u21*b12 + u22*b22;
    geom[0] = m00; geom[1] = m01; geom[2] = m02;
    geom[3] = m10; geom[4] = m11; geom[5] = m12;
    geom[6] = m20; geom[7] = m21; geom[8] = m22;
    let mi = inv3(m00, m01, m02, m10, m11, m12, m20, m21, m22);
    for (var t = 0u; t < 9u; t = t + 1u) { geom[9u + t] = mi[t]; }

    let h0 = beam_hkl(n_base, 0u);
    var lo0 = 1e9; var lo1 = 1e9; var lo2 = 1e9;
    var hi0 = -1e9; var hi1 = -1e9; var hi2 = -1e9;
    for (var i = 0u; i < n; i = i + 1u) {
        let h = beam_hkl(n_base, i);
        let dx = h.x - h0.x; let dy = h.y - h0.y; let dz = h.z - h0.z;
        let c0 = round(m00*dx + m01*dy + m02*dz);
        let c1 = round(m10*dx + m11*dy + m12*dz);
        let c2 = round(m20*dx + m21*dy + m22*dz);
        lo0 = min(lo0, c0); lo1 = min(lo1, c1); lo2 = min(lo2, c2);
        hi0 = max(hi0, c0); hi1 = max(hi1, c1); hi2 = max(hi2, c2);
    }
    let e0 = hi0 - lo0 + 1.0;
    let e1 = hi1 - lo1 + 1.0;
    let e2 = hi2 - lo2 + 1.0;
    let d0 = 2.0 * e0 - 1.0;
    let d1 = 2.0 * e1 - 1.0;
    let d2 = 2.0 * e2 - 1.0;
    let plen_f = d0 * d1 * d2;
    geom[18] = lo0; geom[19] = lo1; geom[20] = lo2;
    geom[21] = e0; geom[22] = e1; geom[23] = e2;
    geom[24] = d0; geom[25] = d1; geom[26] = d2;
    geom[27] = (e0 - 1.0) * (d1 * d2) + (e1 - 1.0) * d2 + (e2 - 1.0);
    geom[28] = plen_f;
    // Any positive pack is solvable via resident or tiled shared fills.
    if (plen_f > 0.0 && plen_f < 1.0e7) {
        geom[29] = 1.0;
    } else {
        geom[29] = 0.0;
    }
}

fn zero_w_stack(lid: u32, wbase: u32, n_elems: u32) {
    for (var i = lid; i < n_elems; i = i + WG) {
        w_stack[wbase + i] = vec2<f32>(0.0, 0.0);
    }
}

@compute @workgroup_size(WG, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let batch = wg_id.x;
    if (batch >= params.batch_count) { return; }
    let n = params.n;
    let wbase = batch * n * params.rank;
    let n_w_elems = n * params.rank;
    if (n == 0u) { return; }
    if (n > MAX_N) {
        zero_w_stack(lid, wbase, n_w_elems);
        if (lid == 0u) { stats[batch] = 0xFFFFFFFFu; }
        return;
    }
    let n_base = batch * n;

    for (var i = lid; i < n; i = i + WG) {
        sord[i] = i;
    }
    workgroupBarrier();

    if (lid == 0u) {
        build_smith_iterative_geometry(n, n_base);
    }
    workgroupBarrier();

    if (lid == 0u) {
        // Always publish pack length (even on reject) for diagnostics
        stats[params.batch_count + batch] = u32(min(geom[28], 4294967040.0));
    }
    workgroupBarrier();

    if (geom[29] < 0.5) {
        zero_w_stack(lid, wbase, n_w_elems);
        if (lid == 0u) { stats[batch] = 0xFFFFFFFFu; }
        return;
    }

    let lo0 = geom[18]; let lo1 = geom[19]; let lo2 = geom[20];
    let e0v = geom[21]; let e1v = geom[22]; let e2v = geom[23];
    let d0 = geom[24]; let d1 = geom[25]; let d2 = geom[26];
    let off = i32(geom[27]);
    let plen = u32(geom[28]);
    let m00 = geom[0]; let m01 = geom[1]; let m02 = geom[2];
    let m10 = geom[3]; let m11 = geom[4]; let m12 = geom[5];
    let m20 = geom[6]; let m21 = geom[7]; let m22 = geom[8];
    let mi00 = geom[9]; let mi01 = geom[10]; let mi02 = geom[11];
    let mi10 = geom[12]; let mi11 = geom[13]; let mi12 = geom[14];
    let mi20 = geom[15]; let mi21 = geom[16]; let mi22 = geom[17];

    let h0 = beam_hkl(n_base, 0u);
    for (var i = lid; i < n; i = i + WG) {
        let h = beam_hkl(n_base, i);
        let dx = h.x - h0.x; let dy = h.y - h0.y; let dz = h.z - h0.z;
        let c0 = round(m00*dx + m01*dy + m02*dz) - lo0;
        let c1 = round(m10*dx + m11*dy + m12*dz) - lo1;
        let c2 = round(m20*dx + m21*dy + m22*dz) - lo2;
        ss[i] = i32(c0 * (d1 * d2) + c1 * d2 + c2);
    }
    workgroupBarrier();

    // Odd-even sort by ss
    for (var phase = 0u; phase < n; phase = phase + 1u) {
        let odd = phase & 1u;
        for (var i = lid; i < n / 2u; i = i + WG) {
            let a = 2u * i + odd;
            let b = a + 1u;
            if (b < n && ss[a] > ss[b]) {
                let ts = ss[a]; ss[a] = ss[b]; ss[b] = ts;
                let to = sord[a]; sord[a] = sord[b]; sord[b] = to;
            }
        }
        workgroupBarrier();
    }

    let pref = params.pref;
    let resident = plen <= MAX_PACK;
    if (lid == 0u) {
        g_nu = 0u;
        g_mode = select(2u, 0u, resident); // tiled only if unique cannot fit
    }
    workgroupBarrier();

    if (resident) {
        if (lid == 0u) { g_mode = 0u; }
        for (var i = lid; i < plen; i = i + WG) {
            let iz = i % u32(d2);
            let t = i / u32(d2);
            let iy = t % u32(d1);
            let ix = t / u32(d1);
            let gx = f32(ix) - (e0v - 1.0);
            let gy = f32(iy) - (e1v - 1.0);
            let gz = f32(iz) - (e2v - 1.0);
            let dhx = round(mi00*gx + mi01*gy + mi02*gz);
            let dhy = round(mi10*gx + mi11*gy + mi12*gz);
            let dhz = round(mi20*gx + mi21*gy + mi22*gz);
            let dhash = i32(dhx) * params.stride_h + i32(dhy) * params.stride_k + i32(dhz) + params.offset;
            if (dhash >= 0 && u32(dhash) < params.table_size) {
                let u = table[u32(dhash)];
                let v = vec2<f32>(pref.x * u.x - pref.y * u.y, pref.x * u.y + pref.y * u.x);
                spack[i] = vec2<f16>(f16(v.x), f16(v.y));
            } else {
                spack[i] = vec2<f16>(0.0h, 0.0h);
            }
        }
        workgroupBarrier();
    } else if (plen <= MAX_PLEN_CAP) {
        let nwords = (plen + 31u) >> 5u;
        let bits_base = batch * MAX_UWORDS;
        let meta_base = batch * (MAX_UWORDS * 2u);
        for (var wi = lid; wi < nwords; wi = wi + WG) {
            atomicStore(&uniq_bits[bits_base + wi], 0u);
        }
        workgroupBarrier();

        for (var i = lid; i < n; i = i + WG) {
            let si = ss[i];
            for (var j = 0u; j < n; j = j + 1u) {
                if (j == i) { continue; }
                let idx = u32(si - ss[j] + off);
                if (idx < plen) {
                    atomicOr(&uniq_bits[bits_base + (idx >> 5u)], 1u << (idx & 31u));
                }
            }
        }
        workgroupBarrier();

        // Serialize bitset → global plain words + exclusive prefixes (once per k).
        if (lid == 0u) {
            var nu = 0u;
            for (var wi = 0u; wi < nwords; wi = wi + 1u) {
                let word = atomicLoad(&uniq_bits[bits_base + wi]);
                uniq_meta[meta_base + wi] = word;
                uniq_meta[meta_base + MAX_UWORDS + wi] = nu;
                nu = nu + countOneBits(word);
            }
            g_nu = nu;
            if (FORCE_DENSE_TILE != 0u) {
                g_mode = 2u;
            } else if (FORCE_UNIQUE_SEG_TILE != 0u) {
                g_mode = select(2u, 3u, nu <= MAX_NU_CAP);
            } else if (nu <= MAX_UNIQ && plen <= MAX_PLEN_SHARED_META) {
                g_mode = 1u; // resident unique (shared meta + values)
            } else if (nu <= MAX_NU_CAP) {
                // Global meta frees shared for a MAX_PACK-sized value window.
                let useg_p = (nu + MAX_PACK - 1u) / MAX_PACK;
                let dense_p = (plen + MAX_PACK - 1u) / MAX_PACK;
                g_mode = select(2u, 3u, useg_p <= dense_p);
            } else {
                g_mode = 2u;
            }
        }
        workgroupBarrier();

        if (g_mode == 1u) {
            // Mirror global meta into shared packing for unique_rank / load_u.
            let nwords_s = min(nwords, MAX_UWORDS_SHARED);
            for (var wi = lid; wi < nwords_s; wi = wi + WG) {
                store_bitword(wi, uniq_meta[meta_base + wi]);
                store_prefix(wi, uniq_meta[meta_base + MAX_UWORDS + wi]);
            }
            workgroupBarrier();
            for (var gi = lid; gi < plen; gi = gi + WG) {
                let wi = gi >> 5u;
                let bit = gi & 31u;
                let word = load_bitword(wi);
                let flag = 1u << bit;
                if ((word & flag) == 0u) { continue; }
                let rank = unique_rank(gi);
                let iz = gi % u32(d2);
                let tt = gi / u32(d2);
                let iy = tt % u32(d1);
                let ix = tt / u32(d1);
                let gx = f32(ix) - (e0v - 1.0);
                let gy = f32(iy) - (e1v - 1.0);
                let gz = f32(iz) - (e2v - 1.0);
                let dhx = round(mi00*gx + mi01*gy + mi02*gz);
                let dhy = round(mi10*gx + mi11*gy + mi12*gz);
                let dhz = round(mi20*gx + mi21*gy + mi22*gz);
                let dhash = i32(dhx) * params.stride_h + i32(dhy) * params.stride_k + i32(dhz) + params.offset;
                if (dhash >= 0 && u32(dhash) < params.table_size) {
                    let u = table[u32(dhash)];
                    let v = vec2<f32>(pref.x * u.x - pref.y * u.y, pref.x * u.y + pref.y * u.x);
                    spack[rank] = vec2<f16>(f16(v.x), f16(v.y));
                } else {
                    spack[rank] = vec2<f16>(0.0h, 0.0h);
                }
            }
            workgroupBarrier();
        }

        if (g_mode == 3u) {
            if (g_nu > MAX_NU_CAP) {
                if (lid == 0u) { g_mode = 2u; }
                workgroupBarrier();
            } else if (USE_GLOBAL_UNIQ_VALS != 0u) {
                let vbase = batch * MAX_NU_CAP;
                for (var gi = lid; gi < plen; gi = gi + WG) {
                    let wi = gi >> 5u;
                    let bit = gi & 31u;
                    let word = uniq_meta[meta_base + wi];
                    let flag = 1u << bit;
                    if ((word & flag) == 0u) { continue; }
                    let rank = unique_rank_g(batch, gi);
                    let iz = gi % u32(d2);
                    let tt = gi / u32(d2);
                    let iy = tt % u32(d1);
                    let ix = tt / u32(d1);
                    let gx = f32(ix) - (e0v - 1.0);
                    let gy = f32(iy) - (e1v - 1.0);
                    let gz = f32(iz) - (e2v - 1.0);
                    let dhx = round(mi00*gx + mi01*gy + mi02*gz);
                    let dhy = round(mi10*gx + mi11*gy + mi12*gz);
                    let dhz = round(mi20*gx + mi21*gy + mi22*gz);
                    let dhash = i32(dhx) * params.stride_h + i32(dhy) * params.stride_k
                        + i32(dhz) + params.offset;
                    if (dhash >= 0 && u32(dhash) < params.table_size) {
                        let u = table[u32(dhash)];
                        let v = vec2<f32>(
                            pref.x * u.x - pref.y * u.y,
                            pref.x * u.y + pref.y * u.x
                        );
                        uniq_vals[vbase + rank] = vec2<f16>(f16(v.x), f16(v.y));
                    } else {
                        uniq_vals[vbase + rank] = vec2<f16>(0.0h, 0.0h);
                    }
                }
                workgroupBarrier();
            }
        }
    } else {
        if (lid == 0u) { g_mode = 2u; }
        workgroupBarrier();
    }
    let mode = g_mode;

    let q = q_values[batch * 4u];
    let q1 = q_values[batch * 4u + 1u];
    let scale = q_values[batch * 4u + 3u];
    let two_q = 2.0 * q;

    var n_own = 0u;
    var my_i: array<u32, BPT>;
    for (var t = 0u; t < BPT; t = t + 1u) {
        let idx = lid + t * WG;
        if (idx < n) {
            my_i[n_own] = idx;
            n_own = n_own + 1u;
        }
    }

    var yd: array<vec2<f32>, BPT>;
    var rd: array<vec2<f32>, BPT>;
    var rhatd: array<vec2<f32>, BPT>;
    var pd: array<vec2<f32>, BPT>;
    var vd: array<vec2<f32>, BPT>;
    var sres: array<vec2<f32>, BPT>;
    var td: array<vec2<f32>, BPT>;
    var Apd: array<vec2<f32>, BPT>;
    var Dd: array<vec2<f32>, BPT>;
    var wd: array<vec2<f32>, BPT>;
    var scl: array<vec2<f32>, BPT>;
    var orig: array<u32, BPT>;

    for (var t = 0u; t < n_own; t = t + 1u) {
        let i = my_i[t];
        let o = sord[i];
        orig[t] = o;
        Dd[t] = d_a[n_base + o];
        let diag = cx_sub(vec2<f32>(q1, 0.0), Dd[t]);
        let rr0 = max(cx_abs2(diag), 1.0e-30);
        let rad = sqrt(sqrt(rr0));
        let inv_rad = 1.0 / max(rad, 1.0e-15);
        let half_ang = 0.5 * atan2(diag.y, diag.x);
        scl[t] = vec2<f32>(inv_rad * cos(-half_ang), inv_rad * sin(-half_ang));
        wd[t] = cx_scale(e0[n_base + o], scale);
        yd[t] = vec2<f32>(0.0, 0.0);
    }

    var total_it = 0u;
    var any_fail = 0u;
    for (var col = 0u; col < params.rank; col = col + 1u) {
        for (var t = 0u; t < n_own; t = t + 1u) {
            yd[t] = vec2<f32>(0.0, 0.0);
            rd[t] = cx_mul(scl[t], wd[t]);
            rhatd[t] = rd[t];
            pd[t] = vec2<f32>(0.0, 0.0);
            vd[t] = vec2<f32>(0.0, 0.0);
        }
        var rho_old = vec2<f32>(1.0, 0.0);
        var alpha = vec2<f32>(1.0, 0.0);
        var omega = vec2<f32>(1.0, 0.0);
        var it = 0u;
        var col_ok = 0u;

        {
            var r0_loc = 0.0;
            for (var t = 0u; t < n_own; t = t + 1u) {
                r0_loc = r0_loc + cx_abs2(rd[t]);
            }
            if (sqrt(reduce_sum_f(lid, r0_loc)) <= params.atol) {
                it = 0u;
                col_ok = 1u;
            } else {
                for (var kk = 1u; kk <= params.max_iter; kk = kk + 1u) {
                    var rho_loc = vec2<f32>(0.0, 0.0);
                    for (var t = 0u; t < n_own; t = t + 1u) {
                        rho_loc = cx_add(rho_loc, cx_mul(cx_conj(rhatd[t]), rd[t]));
                    }
                    let rho = reduce_sum_c(lid, rho_loc);
                    if (cx_abs2(rho) < 1.0e-30) { it = kk; col_ok = 0u; break; }
                    let beta = cx_mul(cx_div(rho, rho_old), cx_div(alpha, omega));

                    for (var t = 0u; t < n_own; t = t + 1u) {
                        let pwo = cx_sub(pd[t], cx_mul(omega, vd[t]));
                        pd[t] = cx_add(rd[t], cx_mul(beta, pwo));
                        sp[my_i[t]] = cx_mul(scl[t], pd[t]);
                    }
                    workgroupBarrier();

            for (var t = 0u; t < n_own; t = t + 1u) {
                Apd[t] = vec2<f32>(0.0, 0.0);
            }
            if (mode == 0u) {
                for (var t = 0u; t < n_own; t = t + 1u) {
                    let i = my_i[t];
                    var acc = vec2<f32>(0.0, 0.0);
                    let si = ss[i];
                    for (var j = 0u; j < n; j = j + 1u) {
                        if (j == i) { continue; }
                        let idx = u32(si - ss[j] + off);
                        if (idx < plen) {
                            acc = cx_add(acc, cx_mul(load_u(idx), sp[j]));
                        }
                    }
                    let diag = cx_sub(vec2<f32>(q1, 0.0), Dd[t]);
                    let qpv = cx_sub(cx_mul(diag, sp[i]), acc);
                    Apd[t] = cx_mul(scl[t], qpv);
                }
            } else if (mode == 1u) {
                for (var t = 0u; t < n_own; t = t + 1u) {
                    let i = my_i[t];
                    var acc = vec2<f32>(0.0, 0.0);
                    let si = ss[i];
                    for (var j = 0u; j < n; j = j + 1u) {
                        if (j == i) { continue; }
                        let idx = u32(si - ss[j] + off);
                        acc = cx_add(acc, cx_mul(load_u(unique_rank(idx)), sp[j]));
                    }
                    let diag = cx_sub(vec2<f32>(q1, 0.0), Dd[t]);
                    let qpv = cx_sub(cx_mul(diag, sp[i]), acc);
                    Apd[t] = cx_mul(scl[t], qpv);
                }

            } else if (mode == 3u) {
                // Unique-segment TILE: bitset/prefix in global uniq_meta; values
                // from uniq_vals into a shared window up to MAX_PACK.
                var seg0 = 0u;
                let vbase = batch * MAX_NU_CAP;
                loop {
                    if (seg0 >= g_nu) { break; }
                    var seglen = UNIQUE_SEG_TILE;
                    if (seglen > MAX_PACK) { seglen = MAX_PACK; }
                    if (seglen > (g_nu - seg0)) { seglen = g_nu - seg0; }
                    if (USE_GLOBAL_UNIQ_VALS != 0u) {
                        for (var i = lid; i < seglen; i = i + WG) {
                            spack[i] = uniq_vals[vbase + seg0 + i];
                        }
                    } else {
                        for (var gi = lid; gi < plen; gi = gi + WG) {
                            let wi = gi >> 5u;
                            let bit = gi & 31u;
                            let word = load_bitword(wi);
                            let flag = 1u << bit;
                            if ((word & flag) == 0u) { continue; }
                            let rank = unique_rank_g(batch, gi);
                            if (rank < seg0 || (rank - seg0) >= seglen) { continue; }
                            let iz = gi % u32(d2);
                            let tt = gi / u32(d2);
                            let iy = tt % u32(d1);
                            let ix = tt / u32(d1);
                            let gx = f32(ix) - (e0v - 1.0);
                            let gy = f32(iy) - (e1v - 1.0);
                            let gz = f32(iz) - (e2v - 1.0);
                            let dhx = round(mi00*gx + mi01*gy + mi02*gz);
                            let dhy = round(mi10*gx + mi11*gy + mi12*gz);
                            let dhz = round(mi20*gx + mi21*gy + mi22*gz);
                            let dhash = i32(dhx) * params.stride_h + i32(dhy) * params.stride_k
                                + i32(dhz) + params.offset;
                            if (dhash >= 0 && u32(dhash) < params.table_size) {
                                let u = table[u32(dhash)];
                                let v = vec2<f32>(
                                    pref.x * u.x - pref.y * u.y,
                                    pref.x * u.y + pref.y * u.x
                                );
                                spack[rank - seg0] = vec2<f16>(f16(v.x), f16(v.y));
                            } else {
                                spack[rank - seg0] = vec2<f16>(0.0h, 0.0h);
                            }
                        }
                    }
                    workgroupBarrier();
                    for (var t = 0u; t < n_own; t = t + 1u) {
                        let i = my_i[t];
                        var acc = Apd[t];
                        let si = ss[i];
                        for (var j = 0u; j < n; j = j + 1u) {
                            if (j == i) { continue; }
                            let idx = u32(si - ss[j] + off);
                            let rank = unique_rank_g(batch, idx);
                            if (rank >= seg0 && (rank - seg0) < seglen) {
                                acc = cx_add(acc, cx_mul(load_u(rank - seg0), sp[j]));
                            }
                        }
                        Apd[t] = acc;
                    }
                    workgroupBarrier();
                    seg0 = seg0 + seglen;
                }
                for (var t = 0u; t < n_own; t = t + 1u) {
                    let i = my_i[t];
                    let diag = cx_sub(vec2<f32>(q1, 0.0), Dd[t]);
                    let qpv = cx_sub(cx_mul(diag, sp[i]), Apd[t]);
                    Apd[t] = cx_mul(scl[t], qpv);
                }

            } else {
                var tile0 = 0u;
                loop {
                    if (tile0 >= plen) { break; }
                    var tlen = MAX_PACK;
                    let remain = plen - tile0;
                    if (remain < tlen) { tlen = remain; }
                    for (var i = lid; i < tlen; i = i + WG) {
                        let gi = tile0 + i;
                        let iz = gi % u32(d2);
                        let tt = gi / u32(d2);
                        let iy = tt % u32(d1);
                        let ix = tt / u32(d1);
                        let gx = f32(ix) - (e0v - 1.0);
                        let gy = f32(iy) - (e1v - 1.0);
                        let gz = f32(iz) - (e2v - 1.0);
                        let dhx = round(mi00*gx + mi01*gy + mi02*gz);
                        let dhy = round(mi10*gx + mi11*gy + mi12*gz);
                        let dhz = round(mi20*gx + mi21*gy + mi22*gz);
                        let dhash = i32(dhx) * params.stride_h + i32(dhy) * params.stride_k
                            + i32(dhz) + params.offset;
                        if (dhash >= 0 && u32(dhash) < params.table_size) {
                            let u = table[u32(dhash)];
                            let v = vec2<f32>(
                                pref.x * u.x - pref.y * u.y,
                                pref.x * u.y + pref.y * u.x
                            );
                            spack[i] = vec2<f16>(f16(v.x), f16(v.y));
                        } else {
                            spack[i] = vec2<f16>(0.0h, 0.0h);
                        }
                    }
                    workgroupBarrier();
                    for (var t = 0u; t < n_own; t = t + 1u) {
                        let i = my_i[t];
                        var acc = Apd[t];
                        let si = ss[i];
                        for (var j = 0u; j < n; j = j + 1u) {
                            if (j == i) { continue; }
                            let idx = u32(si - ss[j] + off);
                            if (idx >= tile0 && (idx - tile0) < tlen) {
                                acc = cx_add(acc, cx_mul(load_u(idx - tile0), sp[j]));
                            }
                        }
                        Apd[t] = acc;
                    }
                    workgroupBarrier();
                    tile0 = tile0 + tlen;
                }
                for (var t = 0u; t < n_own; t = t + 1u) {
                    let i = my_i[t];
                    let diag = cx_sub(vec2<f32>(q1, 0.0), Dd[t]);
                    let qpv = cx_sub(cx_mul(diag, sp[i]), Apd[t]);
                    Apd[t] = cx_mul(scl[t], qpv);
                }
            }
            workgroupBarrier();

                    for (var t = 0u; t < n_own; t = t + 1u) {
                        vd[t] = Apd[t];
                    }

                    var rhat_v_loc = vec2<f32>(0.0, 0.0);
                    for (var t = 0u; t < n_own; t = t + 1u) {
                        rhat_v_loc = cx_add(rhat_v_loc, cx_mul(cx_conj(rhatd[t]), vd[t]));
                    }
                    let rhat_v = reduce_sum_c(lid, rhat_v_loc);
                    if (cx_abs2(rhat_v) < 1.0e-30) { it = kk; col_ok = 0u; break; }
                    alpha = cx_div(rho, rhat_v);

                    for (var t = 0u; t < n_own; t = t + 1u) {
                        yd[t] = cx_add(yd[t], cx_mul(alpha, pd[t]));
                        sres[t] = cx_sub(rd[t], cx_mul(alpha, vd[t]));
                    }
                    var s_loc = 0.0;
                    for (var t = 0u; t < n_own; t = t + 1u) {
                        s_loc = s_loc + cx_abs2(sres[t]);
                    }
                    let s_n = sqrt(reduce_sum_f(lid, s_loc));
                    it = kk;
                    if (s_n != s_n || s_n > 1.0e20) {
                        for (var t = 0u; t < n_own; t = t + 1u) {
                            yd[t] = vec2<f32>(0.0, 0.0);
                        }
                        col_ok = 0u;
                        break;
                    }
                    if (s_n <= params.atol) { col_ok = 1u; break; }

                    for (var t = 0u; t < n_own; t = t + 1u) {
                        sp[my_i[t]] = cx_mul(scl[t], sres[t]);
                    }
                    workgroupBarrier();

            for (var t = 0u; t < n_own; t = t + 1u) {
                Apd[t] = vec2<f32>(0.0, 0.0);
            }
            if (mode == 0u) {
                for (var t = 0u; t < n_own; t = t + 1u) {
                    let i = my_i[t];
                    var acc = vec2<f32>(0.0, 0.0);
                    let si = ss[i];
                    for (var j = 0u; j < n; j = j + 1u) {
                        if (j == i) { continue; }
                        let idx = u32(si - ss[j] + off);
                        if (idx < plen) {
                            acc = cx_add(acc, cx_mul(load_u(idx), sp[j]));
                        }
                    }
                    let diag = cx_sub(vec2<f32>(q1, 0.0), Dd[t]);
                    let qpv = cx_sub(cx_mul(diag, sp[i]), acc);
                    Apd[t] = cx_mul(scl[t], qpv);
                }
            } else if (mode == 1u) {
                for (var t = 0u; t < n_own; t = t + 1u) {
                    let i = my_i[t];
                    var acc = vec2<f32>(0.0, 0.0);
                    let si = ss[i];
                    for (var j = 0u; j < n; j = j + 1u) {
                        if (j == i) { continue; }
                        let idx = u32(si - ss[j] + off);
                        acc = cx_add(acc, cx_mul(load_u(unique_rank(idx)), sp[j]));
                    }
                    let diag = cx_sub(vec2<f32>(q1, 0.0), Dd[t]);
                    let qpv = cx_sub(cx_mul(diag, sp[i]), acc);
                    Apd[t] = cx_mul(scl[t], qpv);
                }

            } else if (mode == 3u) {
                // Unique-segment TILE: bitset/prefix in global uniq_meta; values
                // from uniq_vals into a shared window up to MAX_PACK.
                var seg0 = 0u;
                let vbase = batch * MAX_NU_CAP;
                loop {
                    if (seg0 >= g_nu) { break; }
                    var seglen = UNIQUE_SEG_TILE;
                    if (seglen > MAX_PACK) { seglen = MAX_PACK; }
                    if (seglen > (g_nu - seg0)) { seglen = g_nu - seg0; }
                    if (USE_GLOBAL_UNIQ_VALS != 0u) {
                        for (var i = lid; i < seglen; i = i + WG) {
                            spack[i] = uniq_vals[vbase + seg0 + i];
                        }
                    } else {
                        for (var gi = lid; gi < plen; gi = gi + WG) {
                            let wi = gi >> 5u;
                            let bit = gi & 31u;
                            let word = load_bitword(wi);
                            let flag = 1u << bit;
                            if ((word & flag) == 0u) { continue; }
                            let rank = unique_rank_g(batch, gi);
                            if (rank < seg0 || (rank - seg0) >= seglen) { continue; }
                            let iz = gi % u32(d2);
                            let tt = gi / u32(d2);
                            let iy = tt % u32(d1);
                            let ix = tt / u32(d1);
                            let gx = f32(ix) - (e0v - 1.0);
                            let gy = f32(iy) - (e1v - 1.0);
                            let gz = f32(iz) - (e2v - 1.0);
                            let dhx = round(mi00*gx + mi01*gy + mi02*gz);
                            let dhy = round(mi10*gx + mi11*gy + mi12*gz);
                            let dhz = round(mi20*gx + mi21*gy + mi22*gz);
                            let dhash = i32(dhx) * params.stride_h + i32(dhy) * params.stride_k
                                + i32(dhz) + params.offset;
                            if (dhash >= 0 && u32(dhash) < params.table_size) {
                                let u = table[u32(dhash)];
                                let v = vec2<f32>(
                                    pref.x * u.x - pref.y * u.y,
                                    pref.x * u.y + pref.y * u.x
                                );
                                spack[rank - seg0] = vec2<f16>(f16(v.x), f16(v.y));
                            } else {
                                spack[rank - seg0] = vec2<f16>(0.0h, 0.0h);
                            }
                        }
                    }
                    workgroupBarrier();
                    for (var t = 0u; t < n_own; t = t + 1u) {
                        let i = my_i[t];
                        var acc = Apd[t];
                        let si = ss[i];
                        for (var j = 0u; j < n; j = j + 1u) {
                            if (j == i) { continue; }
                            let idx = u32(si - ss[j] + off);
                            let rank = unique_rank_g(batch, idx);
                            if (rank >= seg0 && (rank - seg0) < seglen) {
                                acc = cx_add(acc, cx_mul(load_u(rank - seg0), sp[j]));
                            }
                        }
                        Apd[t] = acc;
                    }
                    workgroupBarrier();
                    seg0 = seg0 + seglen;
                }
                for (var t = 0u; t < n_own; t = t + 1u) {
                    let i = my_i[t];
                    let diag = cx_sub(vec2<f32>(q1, 0.0), Dd[t]);
                    let qpv = cx_sub(cx_mul(diag, sp[i]), Apd[t]);
                    Apd[t] = cx_mul(scl[t], qpv);
                }

            } else {
                var tile0 = 0u;
                loop {
                    if (tile0 >= plen) { break; }
                    var tlen = MAX_PACK;
                    let remain = plen - tile0;
                    if (remain < tlen) { tlen = remain; }
                    for (var i = lid; i < tlen; i = i + WG) {
                        let gi = tile0 + i;
                        let iz = gi % u32(d2);
                        let tt = gi / u32(d2);
                        let iy = tt % u32(d1);
                        let ix = tt / u32(d1);
                        let gx = f32(ix) - (e0v - 1.0);
                        let gy = f32(iy) - (e1v - 1.0);
                        let gz = f32(iz) - (e2v - 1.0);
                        let dhx = round(mi00*gx + mi01*gy + mi02*gz);
                        let dhy = round(mi10*gx + mi11*gy + mi12*gz);
                        let dhz = round(mi20*gx + mi21*gy + mi22*gz);
                        let dhash = i32(dhx) * params.stride_h + i32(dhy) * params.stride_k
                            + i32(dhz) + params.offset;
                        if (dhash >= 0 && u32(dhash) < params.table_size) {
                            let u = table[u32(dhash)];
                            let v = vec2<f32>(
                                pref.x * u.x - pref.y * u.y,
                                pref.x * u.y + pref.y * u.x
                            );
                            spack[i] = vec2<f16>(f16(v.x), f16(v.y));
                        } else {
                            spack[i] = vec2<f16>(0.0h, 0.0h);
                        }
                    }
                    workgroupBarrier();
                    for (var t = 0u; t < n_own; t = t + 1u) {
                        let i = my_i[t];
                        var acc = Apd[t];
                        let si = ss[i];
                        for (var j = 0u; j < n; j = j + 1u) {
                            if (j == i) { continue; }
                            let idx = u32(si - ss[j] + off);
                            if (idx >= tile0 && (idx - tile0) < tlen) {
                                acc = cx_add(acc, cx_mul(load_u(idx - tile0), sp[j]));
                            }
                        }
                        Apd[t] = acc;
                    }
                    workgroupBarrier();
                    tile0 = tile0 + tlen;
                }
                for (var t = 0u; t < n_own; t = t + 1u) {
                    let i = my_i[t];
                    let diag = cx_sub(vec2<f32>(q1, 0.0), Dd[t]);
                    let qpv = cx_sub(cx_mul(diag, sp[i]), Apd[t]);
                    Apd[t] = cx_mul(scl[t], qpv);
                }
            }
            workgroupBarrier();

                    for (var t = 0u; t < n_own; t = t + 1u) {
                        td[t] = Apd[t];
                    }

                    var tt_loc = vec2<f32>(0.0, 0.0);
                    var ts_loc = vec2<f32>(0.0, 0.0);
                    for (var t = 0u; t < n_own; t = t + 1u) {
                        tt_loc = cx_add(tt_loc, cx_mul(cx_conj(td[t]), td[t]));
                        ts_loc = cx_add(ts_loc, cx_mul(cx_conj(td[t]), sres[t]));
                    }
                    let tt = reduce_sum_c(lid, tt_loc);
                    if (cx_abs2(tt) < 1.0e-30) { col_ok = 0u; break; }
                    let ts = reduce_sum_c(lid, ts_loc);
                    omega = cx_div(ts, tt);

                    for (var t = 0u; t < n_own; t = t + 1u) {
                        yd[t] = cx_add(yd[t], cx_mul(omega, sres[t]));
                        rd[t] = cx_sub(sres[t], cx_mul(omega, td[t]));
                    }
                    var r_loc = 0.0;
                    for (var t = 0u; t < n_own; t = t + 1u) {
                        r_loc = r_loc + cx_abs2(rd[t]);
                    }
                    let r_n = sqrt(reduce_sum_f(lid, r_loc));
                    if (r_n != r_n || r_n > 1.0e20) {
                        for (var t = 0u; t < n_own; t = t + 1u) {
                            yd[t] = vec2<f32>(0.0, 0.0);
                        }
                        col_ok = 0u;
                        break;
                    }
                    if (r_n <= params.atol) { col_ok = 1u; break; }
                    if (cx_abs2(omega) < 1.0e-30) { col_ok = 0u; break; }
                    rho_old = rho;
                }
            }
        }
        total_it = total_it + it;
        if (col_ok == 0u) { any_fail = 1u; }

        var bad_loc = 0.0;
        for (var t = 0u; t < n_own; t = t + 1u) {
            var xd = cx_mul(scl[t], yd[t]);
            if (xd.x != xd.x || xd.y != xd.y || abs(xd.x) > 1.0e3 || abs(xd.y) > 1.0e3) {
                xd = vec2<f32>(0.0, 0.0);
                bad_loc = 1.0;
            }
            w_stack[wbase + orig[t] * params.rank + col] = xd;
            wd[t] = cx_sub(cx_scale(xd, two_q), wd[t]);
        }
        if (reduce_sum_f(lid, bad_loc) > 0.0) { any_fail = 1u; }
        workgroupBarrier();
    }

    if (lid == 0u) {
        if (any_fail != 0u) {
            stats[batch] = 0xFFFFFFFFu;
        } else {
            var st = total_it | 0x20000000u;
            if (mode == 1u || mode == 3u) { st = st | 0x10000000u; }
            if (mode == 3u) { st = st | 0x40000000u; } // unique-segment TILE
            if (mode == 2u) { st = st | 0x80000000u; }
            stats[batch] = st;
        }
        if (mode == 1u || mode == 3u) {
            stats[params.batch_count + batch] = g_nu;
        } else {
            stats[params.batch_count + batch] = plen;
        }
    }
}
