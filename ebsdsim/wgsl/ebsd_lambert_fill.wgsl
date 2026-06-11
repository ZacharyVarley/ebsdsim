struct Params {
    side: u32,
    viewCount: u32,
    planeSize: u32,
    isCentro: u32,
    threadsPerRow: u32,
}

@group(0) @binding(0) var<storage, read> sheets: array<f32>;
@group(0) @binding(1) var<storage, read> nhSrcX: array<f32>;
@group(0) @binding(2) var<storage, read> nhSrcY: array<f32>;
@group(0) @binding(3) var<storage, read> nhFromSh: array<u32>;
@group(0) @binding(4) var<storage, read> shSrcX: array<f32>;
@group(0) @binding(5) var<storage, read> shSrcY: array<f32>;
@group(0) @binding(6) var<storage, read> shFromSh: array<u32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

fn sampleSheet(viewBase: u32, srcX: f32, srcY: f32, useSh: bool) -> f32 {
    let sideMinusOne = f32(max(params.side, 1u) - 1u);
    let fx = clamp((srcX * 0.5 + 0.5) * sideMinusOne, 0.0, sideMinusOne);
    let fy = clamp((srcY * 0.5 + 0.5) * sideMinusOne, 0.0, sideMinusOne);
    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, params.side - 1u);
    let y1 = min(y0 + 1u, params.side - 1u);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let sheetBase = viewBase + select(0u, params.planeSize, useSh);
    let row0 = y0 * params.side;
    let row1 = y1 * params.side;
    let i00 = sheetBase + row0 + x0;
    let i10 = sheetBase + row0 + x1;
    let i01 = sheetBase + row1 + x0;
    let i11 = sheetBase + row1 + x1;
    let v00 = sheets[i00];
    let v10 = sheets[i10];
    let v01 = sheets[i01];
    let v11 = sheets[i11];
    let v0 = v00 * (1.0 - tx) + v10 * tx;
    let v1 = v01 * (1.0 - tx) + v11 * tx;
    return v0 * (1.0 - ty) + v1 * ty;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x + gid.y * params.threadsPerRow;
    let viewStride = params.planeSize * 2u;
    let total = params.viewCount * viewStride;
    if (flat >= total) {
        return;
    }
    let view = flat / viewStride;
    let rem = flat - view * viewStride;
    let hemisphere = rem / params.planeSize;
    let pixel = rem - hemisphere * params.planeSize;
    var srcX = nhSrcX[pixel];
    var srcY = nhSrcY[pixel];
    var fromSh = false;
    if (params.isCentro == 0u) {
        if (hemisphere == 0u) {
            fromSh = nhFromSh[pixel] != 0u;
        } else {
            srcX = shSrcX[pixel];
            srcY = shSrcY[pixel];
            fromSh = shFromSh[pixel] != 0u;
        }
    }
    let viewBase = view * viewStride;
    out[flat] = sampleSheet(viewBase, srcX, srcY, params.isCentro == 0u && fromSh);
}
