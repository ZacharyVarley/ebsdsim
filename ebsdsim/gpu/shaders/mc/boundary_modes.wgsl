const PI: f32 = 3.14159265359;

struct LambertStruct {
    x: f32,
    y: f32,
}

struct McParams {
    n_trajectories: u32,
    starting_E_keV: f32,
    n_exit_energy_bins: u32,
    n_exit_depth_bins: u32,
    n_exit_direction_bins: u32,
    binsize_exit_energy: f32,
    binsize_exit_depth: f32,
    atom_num: f32,
    rho_gcc: f32,
    atomic_weight_A: f32,
    n_max_steps: u32,
    sigma_deg: f32,
    omega_deg: f32,
    depth_mode: u32,
    e_min_keV: f32,
    exit_model: u32,
    depth_metric: u32,
    binning_mode: u32,
}

fn rosca_lambert(pt: vec3<f32>) -> LambertStruct {
    var ret: LambertStruct;
    let factor = sqrt(max(2.0 * (1.0 - abs(pt.z)), 0.0));
    let big = select(pt.y, pt.x, abs(pt.y) <= abs(pt.x));
    let sml = select(pt.x, pt.y, abs(pt.y) <= abs(pt.x));
    let sign_big = select(-1.0, 1.0, big >= 0.0);
    let simpler_term = sign_big * factor * (2.0 / sqrt(8.0));
    let arctan_term = sign_big * factor * atan2(sml * sign_big, abs(big)) * (2.0 * sqrt(2.0) / PI);
    ret.x = select(arctan_term, simpler_term, abs(pt.y) <= abs(pt.x));
    ret.y = select(simpler_term, arctan_term, abs(pt.y) <= abs(pt.x));
    return ret;
}

fn lfsr113_bits(z: vec4<u32>) -> vec4<u32> {
    var out = z;
    var b = ((out.x << 6u) ^ out.x) >> 13u;
    out.x = ((out.x & 4294967294u) << 18u) ^ b;
    b = ((out.y << 2u) ^ out.y) >> 27u;
    out.y = ((out.y & 4294967288u) << 2u) ^ b;
    b = ((out.z << 13u) ^ out.z) >> 21u;
    out.z = ((out.z & 4294967280u) << 7u) ^ b;
    b = ((out.w << 3u) ^ out.w) >> 12u;
    out.w = ((out.w & 4294967168u) << 13u) ^ b;
    return out;
}

fn rand01(seed: ptr<function, vec4<u32>>) -> f32 {
    *seed = lfsr113_bits(*seed);
    let u = f32((*seed).x ^ (*seed).y ^ (*seed).z ^ (*seed).w) * 2.3283064370807974e-10;
    return max(u, 1.0e-12);
}

fn nearest_bin(x: f32, step: f32) -> i32 {
    return i32(floor(x / step + 0.5));
}

fn floor_bin(x: f32, step: f32) -> i32 {
    return i32(floor(x / step));
}

fn joy_luo_deds(E_keV: f32, atom_num: f32, A: f32, J_keV: f32) -> f32 {
    return -0.00785 * (atom_num / (A * E_keV)) * log(1.166 * E_keV / J_keV + 0.9911);
}

fn sigma_total(E_keV: f32, atom_num: f32) -> f32 {
    let alpha = 3.4e-3 * pow(atom_num, 0.66667) / E_keV;
    var rel = ((511.0 + E_keV) / (1024.0 + E_keV));
    rel = rel * rel;
    return (5.21 * 602.2) * atom_num * atom_num / (E_keV * E_keV) * (4.0 * PI / (alpha * (1.0 + alpha))) * rel;
}

fn mfp_nm(E_keV: f32, atom_num: f32, A: f32, rho_gcc: f32) -> f32 {
    return A * 1.0e7 / (rho_gcc * sigma_total(E_keV, atom_num));
}

fn sample_phi(E_keV: f32, atom_num: f32, u: f32) -> f32 {
    let alpha = 3.4e-3 * pow(atom_num, 0.66667) / E_keV;
    return acos(1.0 - ((2.0 * alpha * u) / (1.0 + alpha - u)));
}

fn scatter_dir(c_old: vec3<f32>, phi: f32, psi: f32) -> vec3<f32> {
    let cos_phi = cos(phi);
    let sin_phi = sin(phi);
    if (abs(c_old.z) > 0.99999) {
        let signz = select(-1.0, 1.0, c_old.z > 0.0);
        return normalize(vec3<f32>(sin_phi * cos(psi), sin_phi * sin(psi), signz * cos_phi));
    }
    let dsq = sqrt(1.0 - c_old.z * c_old.z);
    let dsqi = 1.0 / dsq;
    let cos_psi = cos(psi);
    let sin_psi = sin(psi);
    return normalize(vec3<f32>(
        sin_phi * (c_old.x * c_old.z * cos_psi - c_old.y * sin_psi) * dsqi + c_old.x * cos_phi,
        sin_phi * (c_old.y * c_old.z * cos_psi + c_old.x * sin_psi) * dsqi + c_old.y * cos_phi,
        -sin_phi * cos_psi * dsq + c_old.z * cos_phi,
    ));
}

fn lambert_to_bin(dir: vec3<f32>, n_dir: u32) -> vec2<i32> {
    let lambert = rosca_lambert(dir);
    return vec2<i32>(
        i32((lambert.x * 0.499999 + 0.5) * f32(n_dir)),
        i32((lambert.y * 0.499999 + 0.5) * f32(n_dir)),
    );
}

fn record_exit(
    E0_keV: f32,
    E_exit_keV: f32,
    depth_value_nm: f32,
    dir: vec3<f32>,
    nE: i32,
    nD: i32,
    nDir: i32,
    dEbin: f32,
    dzbin: f32,
    depth_mode: u32,
    binning_mode: u32,
) {
    let k = select(floor_bin(E0_keV - E_exit_keV, dEbin), nearest_bin(E0_keV - E_exit_keV, dEbin), binning_mode == 0u);
    let zvalue = select(log(depth_value_nm + 1.0), depth_value_nm, depth_mode == 0u);
    let l = select(floor_bin(zvalue, dzbin), nearest_bin(zvalue, dzbin), binning_mode == 0u);
    let pq = lambert_to_bin(dir, u32(nDir));
    let p = pq.x;
    let q = pq.y;
    if (k >= 0 && k < nE && l >= 0 && l < nD && p >= 0 && p < nDir && q >= 0 && q < nDir) {
        let idx = (((k * nD + l) * nDir + p) * nDir + q);
        atomicAdd(&acc4d[u32(idx)], 1u);
    }
}

@group(0) @binding(0) var<storage, read_write> acc4d: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> seeds: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: McParams;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n_trajectories) {
        return;
    }

    var seed = seeds[gid.x];
    let sigma_rad = params.sigma_deg * PI / 180.0;
    let omega_rad = params.omega_deg * PI / 180.0;
    let J = ((9.76 * params.atom_num) + (58.5 * pow(params.atom_num, -0.19))) * 1.0e-3;

    var pos = vec3<f32>(0.0, 0.0, 0.0);
    var dir = normalize(vec3<f32>(
        sin(sigma_rad) * cos(omega_rad),
        sin(sigma_rad) * sin(omega_rad),
        cos(sigma_rad),
    ));
    var energy = params.starting_E_keV;

    let L0 = -mfp_nm(energy, params.atom_num, params.atomic_weight_A, params.rho_gcc) * log(rand01(&seed));
    let deds0 = joy_luo_deds(energy, params.atom_num, params.atomic_weight_A, J);
    pos = pos + L0 * dir;
    energy = energy + L0 * params.rho_gcc * deds0;
    if (energy <= params.e_min_keV) {
        return;
    }

    let nE = i32(params.n_exit_energy_bins);
    let nD = i32(params.n_exit_depth_bins);
    let nDir = i32(params.n_exit_direction_bins);

    for (var step_index: u32 = 0u; step_index < params.n_max_steps; step_index = step_index + 1u) {
        let deds = joy_luo_deds(energy, params.atom_num, params.atomic_weight_A, J);
        let L = -mfp_nm(energy, params.atom_num, params.atomic_weight_A, params.rho_gcc) * log(rand01(&seed));
        let phi = sample_phi(energy, params.atom_num, rand01(&seed));
        let psi = 2.0 * PI * rand01(&seed);
        let dir_new = scatter_dir(dir, phi, psi);

        if (params.exit_model == 0u) {
            var escape_path = 0.0;
            if (abs(dir_new.z) > 1.0e-5) {
                escape_path = abs(pos.z / dir_new.z);
            }
            pos = pos + L * dir_new;
            energy = energy + L * params.rho_gcc * deds;
            dir = dir_new;
            if (pos.z <= 0.0) {
                record_exit(
                    params.starting_E_keV,
                    max(energy, 0.0),
                    escape_path,
                    dir,
                    nE,
                    nD,
                    nDir,
                    params.binsize_exit_energy,
                    params.binsize_exit_depth,
                    params.depth_mode,
                    params.binning_mode,
                );
                return;
            }
        } else {
            if (dir_new.z < -1.0e-12 && pos.z > 0.0) {
                let x_exit = pos.z / (-dir_new.z);
                if (L >= x_exit) {
                    let E_exit = energy + x_exit * params.rho_gcc * deds;
                    let depth_value = select(pos.z, x_exit, params.depth_metric == 0u);
                    record_exit(
                        params.starting_E_keV,
                        max(E_exit, 0.0),
                        depth_value,
                        dir_new,
                        nE,
                        nD,
                        nDir,
                        params.binsize_exit_energy,
                        params.binsize_exit_depth,
                        params.depth_mode,
                        params.binning_mode,
                    );
                    return;
                }
            }
            pos = pos + L * dir_new;
            energy = energy + L * params.rho_gcc * deds;
            dir = dir_new;
        }

        if (energy <= params.e_min_keV) {
            return;
        }
    }
}
