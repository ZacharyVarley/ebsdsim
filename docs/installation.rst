Installation
============

Requirements
------------

- Python 3.10 – 3.13
- NumPy ≥ 1.21, `wgpu <https://pypi.org/project/wgpu/>`_ ≥ 0.29
- WebGPU adapter to run simulations

CIF files are standardized to the International Tables setting on load via the
vendored ``cif_reader`` package (dual-origin groups use origin choice 2).
GPU setup — drivers, headless Linux, cloud VMs — follows the
`wgpu-py installation guide <https://wgpu-py.readthedocs.io/en/stable/start.html#install-with-pip>`_.

Loading saved ``.npz`` files via :mod:`ebsdsim.io.load` needs NumPy (and the
shared Lambert helpers in :mod:`ebsdsim.lambert`).

PyPI
----

.. code-block:: bash

   pip install ebsdsim

If ``import ebsdsim`` works but simulation fails with a GPU error, see
`wgpu-py platform requirements <https://wgpu-py.readthedocs.io/en/stable/start.html#platform-requirements>`_.

From source
-----------

.. code-block:: bash

   git clone https://github.com/ZacharyVarley/ebsdsim.git
   cd ebsdsim
   pip install -e ".[dev,docs]"

See `CONTRIBUTING.md <https://github.com/ZacharyVarley/ebsdsim/blob/main/CONTRIBUTING.md>`_
on GitHub for development and releases.
