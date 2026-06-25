Simulation parameters
=====================

All parameters below are keyword arguments to :func:`~ebsdsim.master_pattern` and
:func:`~ebsdsim.master_pattern_from_cif`. Defaults match the API.

What drives the pattern
-----------------------

**Beam and specimen** -- ``voltage_kv``, ``sigma_deg``, ``omega_deg``,
``energy_binwidth_keV``

**Monte Carlo** -- ``mc_backend``, ``mc_auto_stop``, ``mc_relative_tol``,
``n_trajectories``

**Dynamical diffraction** -- ``bethe_c_strong``, ``bethe_c_weak``,
``bethe_c_cutoff``, ``dbdiff_sg_cutoff``, ``rank``, ``dmin``,
``exact_slow_cpu``, ``verbosity``

**Raster** -- ``halfw`` (Lambert half-width; side length ``1 + 2*halfw``)

**Structure** -- lattice, sites, ``b_iso``

Beam and Monte Carlo
--------------------

+-------------------------+-----------+---------------------------------------------+
| Parameter               | Default   | Role                                        |
+=========================+===========+=============================================+
| ``voltage_kv``          | ``20.0``  | Nominal beam energy (kV)                    |
| ``sigma_deg``           | ``70.0``  | Specimen tilt (degrees)                     |
| ``omega_deg``           | ``0.0``   | Azimuthal sample rotation (degrees)         |
| ``energy_binwidth_keV`` | ``1.0``   | Energy-loss bin width (keV)                 |
| ``relative_image_stop`` | ``0.01``  | Stop bins when delta-image/image is low     |
| ``marginal_coverage``   | ``1.0``   | Fraction of MC energy bins to integrate     |
| ``mc_backend``          | surrogate | ``"gpu"`` for full boundary Monte Carlo     |
| ``mc_auto_stop``        | ``True``  | GPU MC: stop when fits converge             |
| ``mc_relative_tol``     | ``0.01``  | GPU MC convergence tolerance                |
| ``n_trajectories``      | 1048576   | GPU MC budget when ``mc_auto_stop=False``   |
+-------------------------+-----------+---------------------------------------------+

Dynamical diffraction
---------------------

The solve uses the Bloch formulation. Bethe perturbation theory ranks reflections
(``bethe_c_*`` cutoffs).

+----------------------+-----------+-----------------------------------------------+
| Parameter            | Default   | Role                                          |
+======================+===========+===============================================+
| ``bethe_c_strong``   | ``20.0``  | Strong-beam excitation-score threshold        |
| ``bethe_c_weak``     | ``40.0``  | Weak-beam threshold band                      |
| ``bethe_c_cutoff``   | ``200.0`` | Upper cutoff; beams above are excluded        |
| ``dbdiff_sg_cutoff`` | ``1.0``   | Double-diffraction excitation admission       |
| ``rank``             | ``20``    | Smith / Lyapunov truncation rank              |
| ``dmin``             | ``0.05``  | Minimum *d*-spacing (nm)                      |
| ``halfw``            | ``250``   | Lambert half-width (501 by 501 at default)    |
| ``exact_slow_cpu``   | ``False`` | Full-rank CPU Lyapunov (``numpy.linalg.eig``) |
| ``verbosity``        | ``0``     | Progress: 0 silent, 1 bins, 2 chunks          |
+----------------------+-----------+-----------------------------------------------+

Display scaling: :meth:`~ebsdsim.MasterPattern.lambert_data` with
``normalize="minmax"`` or ``normalize="robust"``.

Debye-Waller factor
-------------------

When ``b_iso`` is omitted on an :class:`~ebsdsim.Atom`, or when a CIF lacks
``_atom_site_B_iso_or_equiv`` / ``_atom_site_U_iso_or_equiv``, ebsdsim defaults to
**0.5 A^2** (stored as 0.005 nm^2). Override per site when better values are known:

.. code-block:: python

   es.Atom("Ga", x=1/3, y=2/3, z=0.0, b_iso=0.45)

Saved ``.npz`` layout
---------------------

+------------------------+------------------------------------------------------+
| Array                  | Meaning                                              |
+========================+======================================================+
| ``fundamental_sector`` | Raw FS intensities ``(E, S, n_k)``                   |
| ``fundamental_kij``    | Lambert indices per FS pixel                         |
| ``fundamental_khat``   | Unit directions per FS pixel                         |
| ``pg_operators``       | Point-group rotation matrices                        |
| ``fs_normals``         | Fundamental-sector bounding normals                  |
| ``bin_voltages_kv``    | Dynamical voltage per energy bin                     |
| ``bin_weights``        | MC weight per energy bin                             |
| ``site_weights``       | Site marginal weights                                |
| ``meta_json``          | Simulation metadata (UTF-8 JSON)                     |
+------------------------+------------------------------------------------------+

Offline loading uses :mod:`ebsdsim.mploader` (NumPy only). See
:doc:`quickstart` and ``examples/02_save_and_load.ipynb``.
