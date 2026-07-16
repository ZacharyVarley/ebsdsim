"""Diff-lookup geometry and coupling tests."""

from __future__ import annotations

import importlib.resources

import numpy as np

from ebsdsim.integrate import surrogate_to_multi_voltage_mc
from ebsdsim.lookup import (
    BuildLookupOptions,
    LookupPrefetcher,
    _wk_scatter_factors_per_row,
    build_diff_lookup,
    build_diff_lookup_from_geometry,
    build_lookup_cache,
    coupling_scale_from_diff_candidates,
    coupling_scale_streaming,
    prepare_diff_lookup_geometry,
)
from ebsdsim.surrogate import infer_direct_exp_from_cell_rebinned
from ebsdsim.structure import build_cell_from_cif_path

_DMIN = 0.05


def _gan_cell():
    path = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif")
    return build_cell_from_cif_path(str(path))


def test_geometry_lookup_matches_full_build():
    cell = _gan_cell()
    geom = prepare_diff_lookup_geometry(cell, _DMIN)
    opts = BuildLookupOptions(voltage_kv=15.5, dmin=_DMIN)
    full = build_diff_lookup(cell, opts)
    fast = build_diff_lookup_from_geometry(geom, opts)
    for field in (
        "hkl",
        "hkl_hash",
        "hkl_diff",
        "diff_table",
        "coupling",
        "reflection_dbdiff",
        "dbdiff_table",
        "mlambda",
        "diag_imag",
        "n_dbdiff",
    ):
        a = getattr(full, field)
        b = getattr(fast, field)
        if np.issubdtype(np.asarray(a).dtype, np.floating):
            assert np.max(np.abs(np.asarray(a) - np.asarray(b))) < 1e-5, field
        else:
            assert np.array_equal(a, b), field


def test_lookup_cache_matches_single_builds():
    cell = _gan_cell()
    geom = prepare_diff_lookup_geometry(cell, _DMIN)
    direct = infer_direct_exp_from_cell_rebinned(
        cell=cell,
        sigma_deg=70.0,
        beam_kv=20.0,
        energy_binwidth_keV=4.0,
        n_energy_bins=5,
    )
    mc = surrogate_to_multi_voltage_mc(direct, 20.0)
    cache = build_lookup_cache(geom, mc.voltages_kv, _DMIN, "bloch")
    for vkv in mc.voltages_kv:
        vkv = float(vkv)
        if vkv <= 0:
            continue
        single = build_diff_lookup_from_geometry(
            geom, BuildLookupOptions(voltage_kv=vkv, dmin=_DMIN)
        )
        cached = cache.get(vkv)
        assert np.allclose(single.diff_table, cached.diff_table, rtol=0, atol=1e-5)
        assert np.allclose(single.coupling, cached.coupling, rtol=0, atol=1e-5)


def test_structure_factor_assembly_matches_scalar_loop():
    cell = _gan_cell()
    geom = prepare_diff_lookup_geometry(cell, _DMIN)
    sf_re, sf_im = _wk_scatter_factors_per_row(
        geom.row_g_sq,
        geom.species_z,
        geom.species_thermal_sigma,
        19.5,
        absorption=True,
        wk_opts={"include_core": True, "include_phonon": True},
    )
    n_diff = geom.row_g_sq.size
    n_sites = geom.site_occ.size
    f_re_o = np.zeros(n_diff)
    f_im_o = np.zeros(n_diff)
    gp_re_o = np.zeros(n_diff)
    gp_im_o = np.zeros(n_diff)
    for row in range(n_diff):
        for site in range(n_sites):
            sp = int(geom.site_species[site])
            occ = geom.site_occ[site] or 1.0
            pr = geom.row_phase_re[row, site]
            pi = geom.row_phase_im[row, site]
            sf_re_v = sf_re[sp, row]
            sf_im_v = sf_im[sp, row]
            f_re_o[row] += occ * pr * sf_re_v
            f_im_o[row] += occ * pi * sf_re_v
            gp_re_o[row] += occ * pr * sf_im_v
            gp_im_o[row] += occ * pi * sf_im_v

    lu = build_diff_lookup_from_geometry(geom, BuildLookupOptions(voltage_kv=19.5, dmin=_DMIN))
    pre = geom.pre
    for row in range(1, min(n_diff, 50)):
        h = int(geom.row_table_hash[row])
        ucg_re = float(lu.diff_table[h * 2])
        ucg_im = float(lu.diff_table[h * 2 + 1])
        assert np.isclose(ucg_re, pre * (f_re_o[row] - gp_im_o[row]), rtol=0, atol=1e-4)
        assert np.isclose(ucg_im, pre * (f_im_o[row] + gp_re_o[row]), rtol=0, atol=1e-4)


def test_wk_scatter_uses_unique_g_sq_only():
    cell = _gan_cell()
    geom = prepare_diff_lookup_geometry(cell, _DMIN)
    n_unique = len(np.unique(geom.row_g_sq))
    assert n_unique < geom.row_g_sq.size
    calls: list[int] = []

    def _counting_wk_array(g, z, thermal_sigma, voltage_kv, opts=None):
        calls.append(int(np.asarray(g).size))
        from ebsdsim.wk import scatter_factor_wk_array as real_wk

        return real_wk(g, z, thermal_sigma, voltage_kv, opts)

    import ebsdsim.lookup as lookup_mod

    original = lookup_mod.scatter_factor_wk_array
    lookup_mod.scatter_factor_wk_array = _counting_wk_array
    try:
        build_diff_lookup_from_geometry(geom, BuildLookupOptions(voltage_kv=19.5, dmin=_DMIN))
    finally:
        lookup_mod.scatter_factor_wk_array = original
    assert calls == [n_unique] * len(geom.species_z)


def test_lookup_prefetcher_matches_single_build():
    cell = _gan_cell()
    geom = prepare_diff_lookup_geometry(cell, _DMIN)
    prefetcher = LookupPrefetcher(geom, _DMIN, "bloch")
    try:
        prefetcher.prefetch(19.5)
        prefetched = prefetcher.get(19.5)
        direct = build_diff_lookup_from_geometry(
            geom, BuildLookupOptions(voltage_kv=19.5, dmin=_DMIN)
        )
        assert np.allclose(prefetched.diff_table, direct.diff_table, rtol=0, atol=1e-5)
        assert np.allclose(prefetched.coupling, direct.coupling, rtol=0, atol=1e-5)
    finally:
        prefetcher.close()


def test_coupling_candidates_matches_streaming():
    cell = _gan_cell()
    lu = build_diff_lookup(cell, BuildLookupOptions(voltage_kv=19.5, dmin=_DMIN))
    stream = coupling_scale_streaming(
        lu.hkl_hash, lu.diff_table, lu.dbdiff_table, lu.offset, lu.mlambda, 512
    )
    cand = coupling_scale_from_diff_candidates(
        lu.hkl_hash,
        lu.hkl_diff,
        lu.diff_table,
        lu.dbdiff_table,
        lu.offset,
        lu.stride_h,
        lu.stride_k,
        lu.mlambda,
    )
    assert np.max(np.abs(stream - cand)) < 1e-5
