pygenstates
===========

``pygenstates`` solves time-independent Schrodinger eigenvalue problems on
continuous spatial domains, with optional coupling to finite-dimensional
discrete bases. It is intended for Hamiltonians that can be represented on a
rectangular grid, including models used in circuit QED and related quantum
systems.

The package provides two solver entry points and one helper function:

``eigensolver``
   Solves an n-dimensional continuous problem of the form:

   .. math::

      H = -\sum_i k_{ii}\frac{d^2}{dx_i^2}
          -\sum_{i<j} k_{ij}\frac{d^2}{dx_i dx_j}
          + U(x_i)

   with eigenvalue equation:

   .. math::

      H|\Psi\rangle = E|\Psi\rangle

   Returning solutions of the form:

   .. math::

      \Psi(x_0,\ldots)=\langle x_0,\ldots|\Psi\rangle

   The default backend uses finite differences. A finite-element backend is also
   available for one-, two-, and three-dimensional problems.

``Ceigensolver``
   Solves a continuous problem coupled to a discrete basis, such as a qubit or
   other N-level subsystem. This has Hamiltonian

   .. math::

      H = H_0(x) + H_1 + H_c

   where:

   .. math::

      H_0 = -\sum_i k_{ii}\frac{d^2}{dx_i^2}
            -\sum_{i<j} k_{ij}\frac{d^2}{dx_i dx_j}
            + U(x_i)

   .. math::

      H_1 = \sum_n E_n |n\rangle\langle n|

   .. math::

      H_c = \sum_i\sum_{n>m}
            \left(-kc_{i,n,m}\frac{d}{dx_i}+vc_{i,n,m}x_i\right)
            |n\rangle\langle m| + \mathrm{h.c.}

   And the eigenvalue problem:

   .. math::

      H|\Psi(x_i)\rangle = E|\Psi(x_i)\rangle

   Returning solutions of the form:

   .. math::

      \Psi_n(x_0,\ldots)=\langle n,x_0,\ldots|\Psi\rangle

   And subject to the normalisation condition:

   .. math::

      \int \sum_n |\Psi_n(x)|^2\,dx = 1

``available_methods``
   Lists the supported numerical backends.

The :doc:`worked_examples` page gives worked examples ranging from harmonic
oscillators to coupled circuit models. The :doc:`usage_notes` page summarizes
input conventions and practical solver notes. The :doc:`api` page documents
the function signatures and return values.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage_notes
   worked_examples
   api
