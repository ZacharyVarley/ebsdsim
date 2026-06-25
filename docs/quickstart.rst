Quick start
===========

Install
-------

.. code-block:: bash

   pip install ebsdsim

A WebGPU-capable GPU and driver are required to run simulations (not needed to
build or read the docs).

From a CIF file
---------------

Bundled presets ``"GaN.cif"`` and ``"Ni.cif"`` ship with the package, or pass any
filesystem path:

.. code-block:: python

   import ebsdsim as es

   mp = es.master_pattern_from_cif(
       "GaN.cif",
       voltage_kv=20.0,
       halfw=250,              # 501×501 Lambert raster
       dmin=0.05,              # nm, reflection cutoff
       energy_binwidth_keV=1.0,
   )
   print(mp.pattern.shape)     # (501, 501) north hemisphere, raw intensities
   mp.save("GaN-master-pattern.npz")

From a manual material
----------------------

.. code-block:: python

   import ebsdsim as es

   ni = es.Material(
       cell=es.Cell(a=3.52, b=3.52, c=3.52, space_group="Fm-3m"),
       atoms=[es.Atom("Ni", x=0.0, y=0.0, z=0.0)],
       name="Ni",
   )
   mp = es.master_pattern(ni, voltage_kv=20.0, halfw=250)

Display scaling
---------------

Simulation results are never min–max normalized. For display:

.. code-block:: python

   disp, axes = mp.lambert_data(normalize="robust")
   nh = disp[0, 0, 0]  # energy- and site-integrated north hemisphere

Offline loading (NumPy only)
----------------------------

The standalone :mod:`ebsdsim.mploader` module reads ``.npz`` files without
installing the full GPU stack:

.. code-block:: python

   from ebsdsim.mploader import load_master_pattern, to_uint8, save_png_gray

   mp = load_master_pattern("GaN-master-pattern.npz")
   disp, _ = mp.lambert_data(normalize="minmax")
   save_png_gray(to_uint8(disp[0, 0, 0]), "preview.png")

Further reading
---------------

* `Installation <installation.html>`_
* `Simulation parameters <parameters.html>`_
* `GitHub repository <https://github.com/ZacharyVarley/ebsdsim>`_
* `Contributing <https://github.com/ZacharyVarley/ebsdsim/blob/main/CONTRIBUTING.md>`_
* `Changelog <https://github.com/ZacharyVarley/ebsdsim/blob/main/CHANGELOG.md>`_
* :doc:`api` — full API reference
