/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

const WG_SIZE: u32 = 64u;

struct IntensityContractParams {
    batch_count: u32,
    n: u32,
    rank: u32,
    n_sites: u32,
    table_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    amplitude: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cx_conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

@group(0) @binding(0) var<uniform> params: IntensityContractParams;
@group(0) @binding(1) var<storage, read> dh: array<u32>;
@group(0) @binding(2) var<storage, read> sgh_tables: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> w_stack: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> intensities: array<f32>;

var<workgroup> partial: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let b = wg_id.x;
    let site = wg_id.y;
    if (b >= params.batch_count || site >= params.n_sites) { return; }

    let dh_base = b * params.n * params.n;
    let w_base = b * params.n * params.rank;
    let table_base = site * params.table_size;
    var total = 0.0;

    let work_items = params.rank * params.n;
    for (var item: u32 = lid; item < work_items; item = item + WG_SIZE) {
        let r = item / params.n;
        let i = item - r * params.n;
        var t = vec2<f32>(0.0, 0.0);
        for (var j: u32 = 0u; j < params.n; j = j + 1u) {
            let key = dh[dh_base + i * params.n + j];
            var sgh = vec2<f32>(0.0, 0.0);
            if (key < params.table_size) {
                sgh = sgh_tables[table_base + key];
            }
            let wj = w_stack[w_base + j * params.rank + r];
            t = t + cx_mul(sgh, wj);
        }
        let wi = w_stack[w_base + i * params.rank + r];
        total = total + cx_mul(cx_conj(wi), t).x;
    }
    partial[lid] = total;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if (lid < stride) {
            partial[lid] = partial[lid] + partial[lid + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }

    if (lid == 0u) {
        intensities[b * params.n_sites + site] = params.amplitude * partial[0];
    }
}
