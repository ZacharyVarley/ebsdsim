/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

const WG_SIZE: u32 = 64u;

struct GemvC64Params {
    batch_count: u32,
    n: u32,
    a_stride: u32,
    x_stride: u32,
    out_stride: u32,
    q_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cx_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x + b.x, a.y + b.y);
}

@group(0) @binding(0) var<uniform> params: GemvC64Params;
@group(0) @binding(1) var<storage, read> a: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> x: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec2<f32>>;

var<workgroup> partial: array<vec2<f32>, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let row = wg_id.x;
    let batch = wg_id.y;
    if (batch >= params.batch_count || row >= params.n) { return; }

    let a_base = batch * params.a_stride + row * params.n;
    let x_base = batch * params.x_stride;
    var acc = vec2<f32>(0.0, 0.0);
    for (var col: u32 = lid; col < params.n; col = col + WG_SIZE) {
        acc = cx_add(acc, cx_mul(a[a_base + col], x[x_base + col]));
    }
    partial[lid] = acc;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if (lid < stride) {
            partial[lid] = cx_add(partial[lid], partial[lid + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }

    if (lid == 0u) {
        // NOTE: out = A @ x. The Cayley iteration's RHS is (q_minus*I + G_eff) @ w,
        // and q_plus_a is assembled with q_minus already added to its diagonal in
        // ebsd-assemble-geff-q.wgsl, so this kernel must NOT add q*x[row] again.
        out[batch * params.out_stride + row] = partial[0];
    }
}
