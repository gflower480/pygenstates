Installation
============

``pygenstates`` is distributed on PyPI and can be installed with ``pip``:

.. code-block:: bash

   pip install pygenstates

A virtual environment is recommended for normal use. This keeps the package and
its numerical dependencies separate from the system Python installation.

On Windows:

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate
   pip install pygenstates

On macOS or Linux:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install pygenstates

Links
-----

* `PyPI <https://pypi.org/project/pygenstates/>`_
* `GitHub <https://github.com/gflower480/pygenstates>`_

General Requirements
--------------------

``pygenstates`` requires Python 3.10 or newer.

The core package installs the following runtime dependencies:

* ``numpy``
* ``scipy``
* ``scikit-fem``

Example notebooks
-----------------

The repository includes worked example notebooks in the ``examples`` folder.
They additionally use:

* ``ipykernel``
* ``matplotlib``
* ``notebook``

From a local checkout of the repository, install these optional dependencies
with:

.. code-block:: bash

   pip install -e ".[examples]"

Documentation Requirements
--------------------------

The documentation is built with Sphinx and the Read the Docs theme. From a
local checkout, install the documentation dependencies and build the HTML pages
with:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html

The documentation build depends on:

* ``sphinx>=7``
* ``sphinx-rtd-theme>=2``
