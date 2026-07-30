// Exact Bethe top-k selection by two radix-select thresholds.
//
// Ordering matches numeric f32 ascending, then smaller reflection index.
// NaNs are treated as +infinity. +0 and -0 are normalized to the same key.
//
// Dispatch: (batch_count, 1, 1), @workgroup_size(256)
// Bindings follow PipelineCache.dispatch_with_params conventions.

struct Params {
    // x=batch_count, y=n_g, z=n_strong, w=n_weak
    dims: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> scores: array<f32>;
@group(0) @binding(2) var<storage, read> candidate_mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> idx_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> idx_w: array<u32>;
@group(0) @binding(5) var<storage, read_write> selected_flags: array<u32>;

const WG_SIZE: u32 = 256u;
const KEY_INF: u32 = 0xffffffffu;

var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> rs_prefix_mask: u32;
var<workgroup> rs_prefix_value: u32;
var<workgroup> rs_k: u32;
var<workgroup> rs_threshold: u32;
var<workgroup> rs_cutoff_index: u32;

var<workgroup> strong_threshold: u32;
var<workgroup> strong_cutoff: u32;
var<workgroup> total_threshold: u32;
var<workgroup> total_cutoff: u32;

fn ordered_key(x_in: f32) -> u32 {
    var x = x_in;

    // Production argmin effectively ignores NaNs; map them to worst.
    if (x != x) {
        return KEY_INF;
    }

    // Numeric comparison considers -0 and +0 equal. Tie must then use index.
    if (x == 0.0) {
        x = 0.0;
    }

    let bits = bitcast<u32>(x);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits ^ 0x80000000u;
}

fn candidate_key(base: u32, g: u32) -> u32 {
    if (candidate_mask[base + g] == 0u) {
        return KEY_INF;
    }
    return ordered_key(scores[base + g]);
}

// Find the exact key of the target_count-th candidate (1-based count), then
// find the reflection-index cutoff among equal-key candidates. Every invocation
// in the workgroup must call this function in uniform control flow.
fn radix_select_threshold(base: u32, n_g: u32, target_count: u32, lid: u32) {
    if (lid == 0u) {
        rs_prefix_mask = 0u;
        rs_prefix_value = 0u;
        rs_k = target_count - 1u; // zero-based rank within current prefix
        rs_threshold = KEY_INF;
        rs_cutoff_index = 0xffffffffu;
    }
    workgroupBarrier();

    // Most-significant byte first. Four passes select one exact f32 key.
    for (var radix_pass = 0u; radix_pass < 4u; radix_pass = radix_pass + 1u) {
        atomicStore(&hist[lid], 0u);
        workgroupBarrier();

        let shift = 24u - 8u * radix_pass;
        let mask_snapshot = rs_prefix_mask;
        let value_snapshot = rs_prefix_value;

        for (var g = lid; g < n_g; g = g + WG_SIZE) {
            if (candidate_mask[base + g] != 0u) {
                let key = ordered_key(scores[base + g]);
                if ((key & mask_snapshot) == value_snapshot) {
                    let digit = (key >> shift) & 0xffu;
                    atomicAdd(&hist[digit], 1u);
                }
            }
        }
        workgroupBarrier();

        if (lid == 0u) {
            var before = 0u;
            var chosen = 255u;
            var d = 0u;
            loop {
                let count = atomicLoad(&hist[d]);
                if (rs_k < before + count) {
                    chosen = d;
                    rs_k = rs_k - before;
                    break;
                }
                before = before + count;
                if (d == 255u) {
                    break;
                }
                d = d + 1u;
            }

            rs_prefix_value = rs_prefix_value | (chosen << shift);
            rs_prefix_mask = rs_prefix_mask | (0xffu << shift);
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        rs_threshold = rs_prefix_value;

        // rs_k is now the zero-based rank among candidates with this exact key.
        var remaining = rs_k + 1u;
        var g = 0u;
        loop {
            if (g >= n_g) {
                break;
            }
            if (candidate_mask[base + g] != 0u &&
                ordered_key(scores[base + g]) == rs_threshold) {
                remaining = remaining - 1u;
                if (remaining == 0u) {
                    rs_cutoff_index = g;
                    break;
                }
            }
            g = g + 1u;
        }
    }
    workgroupBarrier();
}

fn is_selected_by_threshold(
    base: u32,
    g: u32,
    threshold: u32,
    cutoff_index: u32,
) -> bool {
    if (candidate_mask[base + g] == 0u) {
        return false;
    }
    let key = ordered_key(scores[base + g]);
    return key < threshold || (key == threshold && g <= cutoff_index);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let b = workgroup_id.x;
    let lid = local_id.x;
    let batch_count = params.dims.x;
    let n_g = params.dims.y;
    let n_strong = params.dims.z;
    let n_weak = params.dims.w;
    let n_total = n_strong + n_weak;

    if (b >= batch_count) {
        return;
    }

    let base = b * n_g;

    for (var g = lid; g < n_g; g = g + WG_SIZE) {
        selected_flags[base + g] = 0u;
    }
    workgroupBarrier();

    if (n_strong > 0u) {
        radix_select_threshold(base, n_g, n_strong, lid);
        if (lid == 0u) {
            strong_threshold = rs_threshold;
            strong_cutoff = rs_cutoff_index;
        }
    } else if (lid == 0u) {
        strong_threshold = 0u;
        strong_cutoff = 0u;
    }
    workgroupBarrier();

    if (n_total > n_strong) {
        radix_select_threshold(base, n_g, n_total, lid);
        if (lid == 0u) {
            total_threshold = rs_threshold;
            total_cutoff = rs_cutoff_index;
        }
    } else if (lid == 0u) {
        total_threshold = strong_threshold;
        total_cutoff = strong_cutoff;
    }
    workgroupBarrier();

    // Mark selected flags in parallel. Strong and weak membership is exact;
    // output ordering is produced by the ascending serial scan below.
    for (var g = lid; g < n_g; g = g + WG_SIZE) {
        let in_strong = n_strong > 0u && is_selected_by_threshold(
            base, g, strong_threshold, strong_cutoff
        );
        let in_total = n_total > 0u && is_selected_by_threshold(
            base, g, total_threshold, total_cutoff
        );
        if (in_total) {
            selected_flags[base + g] = 1u;
        }
    }
    workgroupBarrier();

    // n_g is only ~3.5k. One ascending scan is cheaper than another parallel
    // sort and guarantees the exact required ascending-index output order.
    if (lid == 0u) {
        var sa = 0u;
        var sw = 0u;
        var g = 0u;
        loop {
            if (g >= n_g) {
                break;
            }

            let in_strong = n_strong > 0u && is_selected_by_threshold(
                base, g, strong_threshold, strong_cutoff
            );
            let in_total = n_total > 0u && is_selected_by_threshold(
                base, g, total_threshold, total_cutoff
            );

            if (in_strong) {
                idx_a[b * n_strong + sa] = g;
                sa = sa + 1u;
            } else if (in_total) {
                idx_w[b * n_weak + sw] = g;
                sw = sw + 1u;
            }
            g = g + 1u;
        }
    }
}
