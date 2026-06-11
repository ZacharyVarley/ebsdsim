/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

const WG_SIZE: u32 = 256u;

struct PackWParams {
    batch_count: u32,
    n: u32,
    rank: u32,
    iter: u32,
    q_offset: u32,
    apply_scale: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: PackWParams;
@group(0) @binding(1) var<storage, read> q_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> w: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> w_stack: array<vec2<f32>>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    let total = params.batch_count * params.n;
    if (linear >= total) { return; }
    let b = linear / params.n;
    let row = linear - b * params.n;
    var value = w[linear];
    if (params.apply_scale != 0u) {
        let s = q_values[b * 4u + params.q_offset];
        value = vec2<f32>(value.x * s, value.y * s);
        w[linear] = value;
    }
    w_stack[b * params.n * params.rank + row * params.rank + params.iter] = value;
}
