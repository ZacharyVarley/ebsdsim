Installation
============

Requirements
------------

- Python 3.10 – 3.13
- NumPy ≥ 1.21, `PyCifRW <https://pypi.org/project/PyCifRW/>`_ ≥ 5.0,
  `wgpu <https://pypi.org/project/wgpu/>`_ ≥ 0.29
- WebGPU adapter to run simulations

CIF files are read with PyCifRW (Hermann–Mauguin symbols and mixed-case tags are
accepted). GPU setup — drivers, headless Linux, cloud VMs — follows the
`wgpu-py installation guide <https://wgpu-py.readthedocs.io/en/stable/start.html#install-with-pip>`_.

Loading saved ``.npz`` files via :mod:`ebsdsim.mploader` needs only NumPy.

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
