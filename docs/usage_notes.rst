Usage Notes
===========

By default, the eigensolvers search for eigenvalues near zero using
``sigma=0``. Use the ``sigma`` keyword argument to change the energy around
which eigenvalues are requested.

Complex absorbing potentials
-----------------------------

By default, both solvers use Hermitian sparse eigensolvers and assume that the
input Hamiltonian is Hermitian. Complex absorbing potentials can be included by
adding the imaginary term to the potential function ``U`` and setting
``nonHermitian=True`` in ``eigensolver`` or ``Ceigensolver``. This switches the
backend to a non-Hermitian eigenvalue solve.

Performance tips
----------------

To discretize the Hamiltonian operator, the supplied potential function ``U`` is
evaluated many times. For good performance, ``U`` should be written in a
vectorized NumPy style. Python branching with ``if`` statements and explicit
``for`` loops can substantially increase solve time. Where branching is needed,
prefer NumPy operations such as ``np.where``. Avoid using ``np.vectorize`` to
make a scalar Python function appear vectorized; it still evaluates through
Python-level looping.

Although ``pygenstates`` can construct finite-difference problems in arbitrary
dimension, dimensions above three can become expensive quickly. With ``x`` grid
points in each direction, the number of grid points grows like
:math:`O(x^n)`.

Using ``which='SA'`` and ``sigma=None`` requests the smallest algebraic
eigenvalues and uses SciPy's standard (and typically fastest) Hermitian sparse 
eigensolver mode. To target a different energy while still using this mode, add
a constant offset to the supplied potential ``U`` so that the desired energy 
region is shifted near zero.

Backends
--------

The default backend is ``"finite_difference"``. It supports arbitrary
continuous dimension, subject to sparse-matrix size and available memory.

The ``"FEM"`` backend uses ``scikit-fem`` and supports one-, two-, and
three-dimensional problems. It accepts ``intorder`` to control the finite-element
integration order.

SciPy Eigensolver Options
-------------------------

The solvers discretize the differential operator into a sparse matrix and pass
that matrix to SciPy's sparse eigensolvers. For more control over the solve,
additional keyword arguments can be passed through to SciPy. See the
`SciPy eigsh API page <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh>`_
for more details.

Common options include ``sigma``, ``which``, ``tol``, ``maxiter``, and ``ncv``.
Use ``Enum`` to choose the number of eigenpairs; do not pass SciPy's ``k``
argument directly.
