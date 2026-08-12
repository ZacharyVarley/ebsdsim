# WGSL shaders

Host wrappers live under `ebsdsim/gpu/` and mirror these folders:

| Shader folder | Host |
|---|---|
| `dynamical/` | `gpu/dynamical/` (LU–Smith stages + Smith) |
| `lu/` | `gpu/lu.py` |
| `mc/` | `gpu/monte_carlo.py` |
| `lambert/` | `gpu/raster.py` |

Load shaders with `ebsdsim.gpu.pipelines.load_wgsl("dynamical/excitation_score.wgsl")`.

## Conventions

- Single `main` compute entry per file.
- Uniform params in a struct (or tightly packed bytes matching the host `struct.pack`).
- Storage bindings use `@binding(N) var<storage, read|read_write>`; the host infers read-only via `infer_storage_read_only`.
- Keep type suffixes in filenames (`_c64`, `_f32`, `_f16`) — they carry information. Do not re-add an `ebsd_` prefix; the package path already scopes them.

## Add a kernel checklist

1. Drop the `.wgsl` into the matching folder under `gpu/shaders/`.
2. Add a host wrapper in the matching stage module (`gpu/dynamical/score.py`, `smith.py`, …) that packs uniforms and calls `PipelineCache.dispatch_with_params` / `PersistentSubmitter`.
3. Add a test under `tests/gpu/` (mark `@pytest.mark.gpu`).
4. If the kernel needs `shader-f16`, ensure `require_gpu(required_features=("shader-f16",))` on that path.
