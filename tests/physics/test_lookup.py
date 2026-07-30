"""Diff-lookup geometry and coupling tests."""

from __future__ import annotations

import importlib.resources

import numpy as np
from ebsdsim.crystal.build import build_cell_from_cif_path
from ebsdsim.energy.surrogate import infer_direct_exp_from_cell_rebinned
from ebsdsim.engine.integrate import surrogate_to_multi_voltage_mc
from ebsdsim.physics.lookup import (
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
        from ebsdsim.physics.scattering import scatter_factor_wk_array as real_wk

        return real_wk(g, z, thermal_sigma, voltage_kv, opts)

    import ebsdsim.physics.lookup as lookup_mod

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


def test_phase_sum_ignores_padded_orbit_slots():
    """Unequal multiplicities must not count zero-padded (0,0,0) as real atoms.

    Regression: dense stacking padded short orbits with zeros; summing cos/sin
    over the pad treated each zero as an atom at the origin and inflated |U|.
    """
    from ebsdsim.api import Atom, Cell, Material

    cell = Material(
        cell=Cell(
            a=5.64871185,
            b=5.64871185,
            c=12.08748583,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
            space_group=166,
        ),
        atoms=[
            Atom("B", 0.01013122, 0.50506561, 0.30845848, b_iso=0.5),
            Atom("B", 0.10535026, 0.21070052, 0.88677738, b_iso=0.5),
            Atom("B", 0.0, 0.0, 0.5, b_iso=0.5),
            Atom("C", 0.0, 0.0, 0.38118845, b_iso=0.5),
        ],
    ).to_simulation_cell()
    mults = [len(p) for p in cell.positions]
    assert max(mults) > min(mults)

    geom = prepare_diff_lookup_geometry(cell, _DMIN)
    # Scalar reference over real orbit members only.
    h = geom.hkl_diff.reshape(-1, 3).astype(np.float64)
    for site, positions in enumerate(cell.positions):
        xyz = np.asarray(positions, dtype=np.float64)
        ang = -2 * np.pi * (h @ xyz.T)
        expect_re = np.cos(ang).sum(axis=1)
        expect_im = np.sin(ang).sum(axis=1)
        assert np.allclose(geom.row_phase_re[:, site], expect_re, rtol=0, atol=1e-9)
        assert np.allclose(geom.row_phase_im[:, site], expect_im, rtol=0, atol=1e-9)

    lu = build_diff_lookup_from_geometry(geom, BuildLookupOptions(voltage_kv=20.0, dmin=_DMIN))
    # Web reference (ebsdsim-web) coupling ≈ 0.0236 for this crystal @ 20 kV / dmin=0.05.
    assert abs(float(lu.coupling[0]) - 0.023617) < 5e-4
