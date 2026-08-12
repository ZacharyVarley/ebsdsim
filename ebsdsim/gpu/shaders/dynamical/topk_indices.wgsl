const WG_SIZE: u32 = 64u;
const SCORE_WORST: f32 = 0x1.fffffep+127f;

struct TopKParams {
    batch_count: u32,
    n_g: u32,
    n_strong: u32,
    n_weak: u32,
}

@group(0) @binding(0) var<uniform> params: TopKParams;
@group(0) @binding(1) var<storage, read> scores: array<f32>;
@group(0) @binding(2) var<storage, read> candidate_mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> selected_flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> idx_a: array<u32>;
@group(0) @binding(5) var<storage, read_write> idx_w: array<u32>;

var<workgroup> best_scores: array<f32, WG_SIZE>;
var<workgroup> best_indices: array<u32, WG_SIZE>;

fn better(score_a: f32, index_a: u32, score_b: f32, index_b: u32) -> bool {
    if (score_a < score_b) { return true; }
    if (score_a > score_b) { return false; }
    return index_a < index_b;
}

fn rank_sort_key(_rank: u32, candidate: u32, ratio: f32) -> f32 {
    // Match Python _select_beams: rank all candidate beams together by score,
    // then split top n_strong as A and next n_weak as W.
    if (candidate == 0u) {
        return SCORE_WORST;
    }
    return ratio;
}

fn insert_sorted_a(base: u32, count: u32, value: u32) {
    var pos = count;
    loop {
        if (pos == 0u) { break; }
        let prev_index = pos - 1u;
        let prev = idx_a[base + prev_index];
        if (prev <= value) { break; }
        idx_a[base + pos] = prev;
        pos = prev_index;
    }
    idx_a[base + pos] = value;
}

fn insert_sorted_w(base: u32, count: u32, value: u32) {
    var pos = count;
    loop {
        if (pos == 0u) { break; }
        let prev_index = pos - 1u;
        let prev = idx_w[base + prev_index];
        if (prev <= value) { break; }
        idx_w[base + pos] = prev;
        pos = prev_index;
    }
    idx_w[base + pos] = value;
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let b = wg_id.x;
    if (b >= params.batch_count) { return; }

    let n_sw = params.n_strong + params.n_weak;
    let score_base = b * params.n_g;
    let flag_base = b * params.n_g;
    let idx_a_base = b * params.n_strong;
    let idx_w_base = b * params.n_weak;

    for (var g: u32 = lid; g < params.n_g; g = g + WG_SIZE) {
        selected_flags[flag_base + g] = 0u;
    }
    workgroupBarrier();

    for (var rank: u32 = 0u; rank < n_sw; rank = rank + 1u) {
        var best_score = SCORE_WORST;
        var best_index = 0xffffffffu;
        for (var g: u32 = lid; g < params.n_g; g = g + WG_SIZE) {
            let used = selected_flags[flag_base + g] != 0u;
            if (used) { continue; }
            let candidate = candidate_mask[score_base + g];
            let ratio = scores[score_base + g];
            let s = rank_sort_key(rank, candidate, ratio);
            if (better(s, g, best_score, best_index)) {
                best_score = s;
                best_index = g;
            }
        }
        best_scores[lid] = best_score;
        best_indices[lid] = best_index;
        workgroupBarrier();

        var stride = WG_SIZE / 2u;
        loop {
            if (lid < stride) {
                let other_score = best_scores[lid + stride];
                let other_index = best_indices[lid + stride];
                if (better(other_score, other_index, best_scores[lid], best_indices[lid])) {
                    best_scores[lid] = other_score;
                    best_indices[lid] = other_index;
                }
            }
            workgroupBarrier();
            if (stride == 1u) { break; }
            stride = stride / 2u;
        }

        if (lid == 0u) {
            var row_best = best_indices[0];
            if (row_best == 0xffffffffu) {
                row_best = 0u;
            }
            selected_flags[flag_base + row_best] = 1u;
            if (rank < params.n_strong) {
                insert_sorted_a(idx_a_base, rank, row_best);
            } else {
                insert_sorted_w(idx_w_base, rank - params.n_strong, row_best);
            }
        }
        workgroupBarrier();
    }
}
