"""
Generic Schrodinger eigensolver interface.

This module dispatches to either the finite-difference backend or the
scikit-fem backend while keeping the public call format consistent.
"""

import warnings

from . import eigensolver_fd as _fd
from . import eigensolver_fem as _fem


__all__ = ["eigensolver", "Ceigensolver", "available_methods"]


_METHODS = ("finite_difference", "FEM")


def _normalise_method(method):
    if method is None:
        return "finite_difference"
    if method in _METHODS:
        return method
    raise ValueError("Unknown method {!r}. Choose 'finite_difference' or 'FEM'.".format(method))


def _backend_for(method, N):
    backend = _normalise_method(method)
    if backend == "FEM" and len(N) > 3:
        warnings.warn(
            "method='FEM' supports only 1D, 2D, or 3D problems; defaulting back to method='finite_difference'.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "finite_difference"
    return backend


def available_methods():
    """
    Return the supported method names.

    Returns
    -------
    tuple[str, str]
        ``("finite_difference", "FEM")``.
    """
    return _METHODS


def eigensolver(U, N=[], domain=[], k_diag=[1], k_cross=[], method="finite_difference",
                Enum=1, vals_only=False, intorder=None, nonHermitian=False,
                **eigsh_kwargs):
    """
    Solve a time-independent Schrodinger eigenvalue problem.

    The Hamiltonian convention is::

        H = -sum_i k_ii d^2/dx_i^2 -sum_{i<j} k_ij d^2/(dx_i dx_j) + U(x_i)

    and the solver returns eigenpairs satisfying ``H psi = E psi`` on a
    rectangular domain with zero Dirichlet boundary conditions.

    Parameters
    ----------
    U : callable
        Potential function. It must accept one array argument per coordinate and
        return either a scalar or an array broadcastable to the grid shape.
        Examples are ``U(x)``, ``U(x, y)``, and ``U(x, y, z)``.
    N : sequence of int
        Number of grid points in each spatial dimension, including boundary
        points. Each entry must be at least 3.
    domain : sequence of tuple(float, float)
        Bounds for each coordinate, with the same length as ``N``.
        For example, ``[(-5, 5)]`` or ``[(-3, 3), (-3, 3)]``.
    k_diag : sequence of float or complex
        Diagonal kinetic coefficients ``k_ii``, one per spatial dimension.
    k_cross : dict or scalar or None, optional
        Mixed-derivative coefficients. Use ``{(i, j): value}`` for the
        coefficient multiplying ``-d^2/(dx_i dx_j)``. In 2D, a scalar is
        accepted as shorthand for ``{(0, 1): value}``. Use ``None`` or ``[]``
        for no mixed derivative terms.
    method : {"finite_difference", "FEM"}, optional
        Numerical backend. ``"finite_difference"`` is the default and supports
        arbitrary dimension. ``"FEM"`` uses scikit-fem and supports 1D, 2D, and
        3D; if requested above 3D, the solver warns and falls back to finite
        differences.
    Enum : int, optional
        Number of eigenvalues/eigenvectors to compute.
    vals_only : bool, optional
        If True, return only eigenvalues and coordinate grids.
    intorder : int or None, optional
        Integration order passed to ``skfem.Basis`` for FEM solves. Ignored by
        the finite-difference backend. If None, scikit-fem chooses its default.
    nonHermitian : bool, optional
        If False, use the Hermitian solver ``scipy.sparse.linalg.eigsh``.
        Complex Hermitian-safe coefficients are allowed. If the potential has a
        nonzero imaginary part, or the Hamiltonian is otherwise non-Hermitian,
        set this to True to use ``scipy.sparse.linalg.eigs``.
    **eigsh_kwargs
        Extra keyword arguments passed to SciPy's sparse eigensolver, such as
        ``sigma``, ``which``, ``tol``, ``maxiter``, or ``ncv``. Use ``Enum``
        rather than passing ``k`` directly. If no eigensolver options are given,
        the default is ``sigma=0``.

    Returns
    -------
    vals : ndarray
        Eigenvalues sorted by real part, then imaginary part.
    vecs : ndarray
        Eigenvectors reshaped onto the full grid, including zero boundary
        values. Shape is ``(Enum, *N)``. Returned only when
        ``vals_only=False``.
    xlists : list[ndarray]
        Coordinate arrays, one per spatial dimension.

    Notes
    -----
    With ``vals_only=False`` the return value is ``(vals, vecs, xlists)``.
    With ``vals_only=True`` the return value is ``(vals, xlists)``.
    """
    backend = _backend_for(method, N)
    if backend == "finite_difference":
        return _fd.eigensolver(
            U,
            N=N,
            domain=domain,
            k_diag=k_diag,
            k_cross=k_cross,
            Enum=Enum,
            vals_only=vals_only,
            nonHermitian=nonHermitian,
            **eigsh_kwargs,
        )

    return _fem.eigensolver(
        U,
        N=N,
        domain=domain,
        k_diag=k_diag,
        k_cross=k_cross,
        Enum=Enum,
        vals_only=vals_only,
        intorder=intorder,
        nonHermitian=nonHermitian,
        **eigsh_kwargs,
    )


def Ceigensolver(U, H1, N=[], domain=[], k_diag=[1], k_cross=[],
                 k_coup=None, v_coup=None, method="finite_difference", Enum=1,
                 vals_only=False, intorder=None, nonHermitian=False,
                 **eigsh_kwargs):
    """
    Solve a continuous Schrodinger problem coupled to a discrete basis.

    The state has components ``Psi_m(x)`` in an ``M``-dimensional discrete
    basis. The Hamiltonian is assembled as::

        H = H0(x) + H1 + Hc

    where::

        H0 = -sum_i k_ii d^2/dx_i^2 -sum_{i<j} k_ij d^2/(dx_i dx_j) + U(x_i)

        H1 = sum_n E_n |n><n|

        Hc = sum_i sum_{n>m} (-k_c[i,n,m] d/dx_i + v_c[i,n,m] x_i)|n><m| 
            + Hermitian conjugate

    Parameters
    ----------
    U : callable
        Potential function for the continuous coordinates.
    H1 : array_like, shape (M, M)
        Matrix acting on the discrete basis. It must be Hermitian unless
        ``nonHermitian=True``.
    N : sequence of int
        Number of grid points in each spatial dimension.
    domain : sequence of tuple(float, float)
        Bounds for each coordinate, with the same length as ``N``.
    k_diag : sequence of float or complex
        Diagonal kinetic coefficients ``k_ii``, one per spatial dimension.
    k_cross : dict or scalar or None, optional
        Mixed-derivative coefficients for the continuous Hamiltonian ``H0``.
        Uses the same convention as ``eigensolver``:
        ``{(i, j): value}`` multiplies ``-d^2/(dx_i dx_j)``.
    method : {"finite_difference", "FEM"}, optional
        Numerical backend.
    Enum : int, optional
        Number of eigenvalues/eigenvectors to compute.
    vals_only : bool, optional
        If True, return only eigenvalues and coordinate grids.
    intorder : int or None, optional
        Integration order passed to ``skfem.Basis`` for FEM solves.
    nonHermitian : bool, optional
        If True, use SciPy's non-Hermitian sparse eigensolver.
    **eigsh_kwargs
        Extra keyword arguments passed to SciPy's sparse eigensolver.
    k_coup : scalar, matrix, dict, list, or None, optional
        Derivative coupling coefficients. A scalar creates nearest-neighbour
        couplings between adjacent discrete levels along spatial axis 0. The
        upper off-diagonal derivative block is ``-coeff * d/dx`` and the lower
        block uses ``conj(coeff) * d/dx`` so that the full derivative coupling
        is Hermitian. An ``M x M`` matrix supplies the discrete coupling matrix
        along axis 0. A dict ``{axis: scalar_or_matrix}`` chooses the spatial
        derivative axis, allowing different coupling matrices for different
        continuous coordinates. A list of pair dictionaries, such as
        ``[{(0, 1): value}, {(1, 2): value}]``, supplies physical pair
        coefficients for each continuous coordinate axis and fills the
        Hermitian-conjugate entries automatically. In Hermitian mode, derivative
        coupling matrices must be skew-Hermitian because ``d/dx`` is
        anti-Hermitian.
    v_coup : scalar, matrix, dict, list, or None, optional
        Linear position coupling coefficients. Input forms match ``k_coup``.
        A scalar creates nearest-neighbour position couplings along axis 0. In
        Hermitian mode, position coupling matrices must be Hermitian. Pair
        dictionaries fill conjugate entries automatically.

    Returns
    -------
    vals : ndarray
        Eigenvalues sorted by real part, then imaginary part.
    vecs : ndarray
        Coupled eigenvectors on the full grid. Shape is ``(Enum, M, *N)``.
        Returned only when ``vals_only=False``.
    xlists : list[ndarray]
        Coordinate arrays, one per spatial dimension.

    Notes
    -----
    With ``vals_only=False`` the return value is ``(vals, vecs, xlists)``.
    With ``vals_only=True`` the return value is ``(vals, xlists)``.
    """
    backend = _backend_for(method, N)
    if backend == "finite_difference":
        return _fd.Ceigensolver(
            U,
            H1,
            N=N,
            domain=domain,
            k_diag=k_diag,
            k_cross=k_cross,
            k_coup=k_coup,
            v_coup=v_coup,
            Enum=Enum,
            vals_only=vals_only,
            nonHermitian=nonHermitian,
            **eigsh_kwargs,
        )

    return _fem.Ceigensolver(
        U,
        H1,
        N=N,
        domain=domain,
        k_diag=k_diag,
        k_cross=k_cross,
        k_coup=k_coup,
        v_coup=v_coup,
        Enum=Enum,
        vals_only=vals_only,
        intorder=intorder,
        nonHermitian=nonHermitian,
        **eigsh_kwargs,
    )
