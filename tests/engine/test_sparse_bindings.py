"""CPU tests for sparse WebGPU storage binding assignment."""

from __future__ import annotations

from ebsdsim.gpu.batch import ResourceBinding, binding_indices_for, resource_binding_index
from ebsdsim.gpu.pipelines import infer_storage_read_only, resolve_storage_read_only


def test_positional_binding_indices_unchanged():
    resources = ["a", "b", "c"]
    assert binding_indices_for(resources) == [1, 2, 3]
    assert resource_binding_index(resources[0], 0) == 1


def test_sparse_binding_indices_skip_reserved_slots():
    resources = [
        "hkl",
        "table",
        ResourceBinding("galerkin_h", binding=14),
        ResourceBinding("galerkin_b", binding=15),
    ]
    assert binding_indices_for(resources) == [1, 2, 14, 15]


def test_infer_storage_read_only_contiguous_compat():
    wgsl = """
    @group(0) @binding(1) var<storage, read> a: array<f32>;
    @group(0) @binding(2) var<storage, read_write> b: array<f32>;
    @group(0) @binding(3) var<storage, read> c: array<f32>;
    """
    assert infer_storage_read_only(wgsl, 3) == [True, False, True]


def test_infer_storage_read_only_sparse_bindings():
    wgsl = """
    @group(0) @binding(1) var<storage, read> a: array<f32>;
    @group(0) @binding(11) var<storage, read_write> meta: array<u32>;
    @group(0) @binding(14) var<storage, read_write> galerkin_h: array<vec2<f32>>;
    @group(0) @binding(15) var<storage, read_write> galerkin_b: array<vec2<f32>>;
    """
    # Contiguous assumption would skip 14/15 when n_storage_bindings=4.
    assert infer_storage_read_only(wgsl, 4) == [True, False, False, False]
    indices = [1, 11, 14, 15]
    assert infer_storage_read_only(wgsl, 4, binding_indices=indices) == [
        True,
        False,
        False,
        False,
    ]
    resolved = resolve_storage_read_only(
        wgsl, 4, None, binding_indices=indices
    )
    assert resolved == [True, False, False, False]
