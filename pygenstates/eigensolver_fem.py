import numpy as np
import skfem as fem
from skfem.helpers import dot, grad
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh



@fem.BilinearForm
def Mform(u, v, w):
    return u * v


def _k_cross_matrix(k_cross, dim, complex_ok=True, conjugate_pairs=True):
    """Return a symmetric off-diagonal matrix of mixed derivative coefficients."""
    dtype = complex if complex_ok else float
    K = np.zeros((dim, dim), dtype=dtype)
    if k_cross is None:
        return K

    if isinstance(k_cross, dict):
        items = [(key[0], key[1], value) for key, value in k_cross.items()]
    else:
        arr = np.asarray(k_cross, dtype=dtype)
        if arr.size == 0:
            return K
        if arr.ndim != 0 or dim != 2:
            raise ValueError("k_cross must be empty/None, a dict {(i, j): value}, or a scalar in 2D.")
        items = [(0, 1, float(arr))]

    for i, j, value in items:
        i = int(i)
        j = int(j)
        if i == j:
            raise ValueError("k_cross only stores off-diagonal mixed derivative coefficients.")
        if i < 0 or i >= dim or j < 0 or j >= dim:
            raise ValueError("k_cross index out of range for the problem dimension.")
        if K[i, j] != 0.0 and not np.isclose(K[i, j], value):
            raise ValueError("Conflicting duplicate entries in k_cross.")
        K[i, j] = value
        K[j, i] = np.conjugate(value) if conjugate_pairs else value

    return K


def _add_scaled_symmetric_operator(base, op, coeff, nonHermitian=False):
    if nonHermitian or np.isrealobj(coeff) or np.isclose(np.imag(coeff), 0.0):
        return base + coeff * op

    op = op.tocsr()
    upper = sp.triu(op, k=1, format="csr")
    diag = sp.diags(op.diagonal(), offsets=0, format="csr")
    hermitian_op = coeff * upper + np.conjugate(coeff) * upper.getH() + np.real(coeff) * diag
    return base + hermitian_op


def _potential_for_form(U, coords, nonHermitian=False):
    V = np.asarray(U(*coords), dtype=complex)
    if not nonHermitian:
        if np.any(np.abs(np.imag(V)) > 1e-12):
            raise ValueError("Complex potentials make the Hamiltonian non-Hermitian; set nonHermitian=True to use eigs.")
        return np.real(V)
    return V


def _validate_square_matrix(mat, name):
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return mat


def _nearest_neighbor_coupling(M, coeff, derivative=False):
    mat = np.zeros((M, M), dtype=np.result_type(coeff, complex))
    for m in range(1, M):
        if derivative:
            mat[m - 1, m] = -coeff
            mat[m, m - 1] = np.conjugate(coeff)
        else:
            mat[m - 1, m] = coeff
            mat[m, m - 1] = np.conjugate(coeff)
    return mat


def _coupling_from_pairs(pairs, M, name, derivative=False):
    mat = np.zeros((M, M), dtype=complex)
    seen = set()
    for key, coeff in pairs.items():
        try:
            row, col = key
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} pair keys must be two-index tuples.") from exc

        row = int(row)
        col = int(col)
        if row == col:
            raise ValueError(f"{name} pair dictionaries only describe off-diagonal couplings.")
        if row < 0 or row >= M or col < 0 or col >= M:
            raise ValueError(f"{name} pair index out of range for discrete dimension {M}.")

        pair = frozenset((row, col))
        if pair in seen:
            raise ValueError(f"{name} pair dictionaries must not include both directions of a coupling.")
        seen.add(pair)

        if derivative:
            mat[row, col] = -coeff
            mat[col, row] = np.conjugate(coeff)
        else:
            mat[row, col] = coeff
            mat[col, row] = np.conjugate(coeff)

    return mat


def _is_pair_dict(value):
    return isinstance(value, dict) and all(isinstance(key, tuple) and len(key) == 2 for key in value)


def _coupling_matrices(coup, dim, M, name, derivative=False, nonHermitian=False):
    if coup is None:
        return {}
    if isinstance(coup, (list, tuple)):
        if len(coup) == 0:
            return {}
        if len(coup) > dim:
            raise ValueError(f"{name} has entries for more axes than the continuous dimension {dim}.")
        items = enumerate(coup)
    elif _is_pair_dict(coup):
        items = [(0, coup)]
    elif isinstance(coup, dict):
        items = coup.items()
    else:
        items = [(0, coup)]

    mats = {}
    for axis, value in items:
        if value is None:
            continue
        axis = int(axis)
        if axis < 0 or axis >= dim:
            raise ValueError(f"{name} axis {axis} is out of range for dimension {dim}.")

        if _is_pair_dict(value):
            mat = _coupling_from_pairs(value, M, name, derivative=derivative)
        else:
            arr = np.asarray(value)
            if arr.ndim == 0:
                mat = _nearest_neighbor_coupling(M, arr.item(), derivative=derivative)
            else:
                mat = _validate_square_matrix(arr, name)
                if mat.shape != (M, M):
                    raise ValueError(f"{name} matrices must have shape {(M, M)}.")

        if nonHermitian:
            pass
        elif derivative:
            if not np.allclose(mat.conj().T, -mat):
                raise ValueError(f"{name} derivative coupling matrices must be skew-Hermitian.")
        elif not np.allclose(mat.conj().T, mat):
            raise ValueError(f"{name} position coupling matrices must be Hermitian.")
        mats[axis] = sp.csr_matrix(mat)

    return mats


def _solve_eigenproblem(A, Enum, M=None, nonHermitian=False, **solver_kwargs):
    solver_kwargs = dict(solver_kwargs) if solver_kwargs else {"sigma": 0}
    if nonHermitian:
        return eigs(A, k=Enum, M=M, **solver_kwargs)
    return eigsh(A, k=Enum, M=M, **solver_kwargs)


def _sort_eigenpairs(vals, vecs):
    idx = np.lexsort((np.imag(vals), np.real(vals)))
    return vals[idx], vecs[:, idx]


def _basis_from_grid(xlists, intorder=None):
    basis_kwargs = {} if intorder is None else {"intorder": intorder}
    if len(xlists) == 2:
        m = fem.MeshQuad.init_tensor(*xlists)
        return fem.Basis(m, fem.ElementQuad1(), **basis_kwargs)
    if len(xlists) == 3:
        m = fem.MeshHex.init_tensor(*xlists)
        return fem.Basis(m, fem.ElementHex1(), **basis_kwargs)
    if len(xlists) == 1:
        m = fem.MeshLine(xlists[0])
        return fem.Basis(m, fem.ElementLineP1(), **basis_kwargs)
    raise ValueError("This implementation supports only 1D, 2D, or 3D.")


def _derivative_form(axis):
    @fem.BilinearForm
    def Dform(u, v, w):
        return v * grad(u)[axis]
    return Dform


def _position_form(axis):
    @fem.BilinearForm
    def Xform(u, v, w):
        return w.x[axis] * u * v
    return Xform


def _mixed_form(i, j):
    @fem.BilinearForm
    def Cform(u, v, w):
        gu = grad(u)
        gv = grad(v)
        return 0.5 * (gu[j] * gv[i] + gu[i] * gv[j])
    return Cform


def eigensolver(U, N=[], domain=[], k_diag=[1], k_cross=[],Enum=1, vals_only=False,
                intorder=None, nonHermitian=False, **eigsh_kwargs):
    """
    Solve a time-independent Schrodinger eigenvalue problem using scikit-fem.

    The Hamiltonian convention is::

        H = -sum_i k_ii d^2/dx_i^2
            -sum_{i<j} k_ij d^2/(dx_i dx_j)
            + U(x_i)

    The solve is performed on a 1D, 2D, or 3D finite-element mesh with zero
    Dirichlet boundary conditions.

    Parameters
    ----------
    U : callable
        Potential function. It must accept one array argument per coordinate and
        return either a scalar or an array broadcastable to quadrature points.
    N : sequence of int
        Number of grid points in each spatial dimension.
    domain : sequence of tuple(float, float)
        Bounds for each coordinate, with the same length as ``N``.
    k_diag : sequence of float or complex
        Diagonal kinetic coefficients ``k_ii``, one per spatial dimension.
    k_cross : dict, scalar, or None, optional
        Mixed-derivative coefficients ``k_ij``. Use ``{(i, j): value}`` for the
        coefficient multiplying ``-d^2/(dx_i dx_j)``. In 2D, a scalar is
        accepted as shorthand for ``{(0, 1): value}``. Use ``None`` or ``[]``
        for no mixed derivative terms.
    Enum : int, optional
        Number of eigenvalues/eigenvectors to compute.
    vals_only : bool, optional
        If True, return only eigenvalues and coordinate grids.
    intorder : int or None, optional
        Integration order passed to ``skfem.Basis``. If None, scikit-fem chooses
        its default integration order.
    nonHermitian : bool, optional
        If True, allow complex potentials and genuinely non-Hermitian terms and
        use ``scipy.sparse.linalg.eigs`` instead of ``eigsh``.
    **eigsh_kwargs
        Extra keyword arguments passed to SciPy's sparse eigensolver, such as
        ``sigma``, ``which``, ``tol``, ``maxiter``, or ``ncv``. Use ``Enum``
        rather than passing ``k`` directly. Do not pass ``M``; the FEM mass
        matrix is built internally. If no options are supplied, the default is
        ``sigma=0``.

    Returns
    -------
    vals : ndarray
        Eigenvalues sorted by real part, then imaginary part.
    vecs : ndarray
        Eigenvectors reshaped onto the full grid. Shape is ``(Enum, *N)`` for
        2D and 3D, and ``(Enum, N[0])`` for 1D. Returned only when
        ``vals_only=False``.
    xlists : list[ndarray]
        Coordinate arrays, one per spatial dimension.

    Notes
    -----
    With ``vals_only=False`` the return value is ``(vals, vecs, xlists)``.
    With ``vals_only=True`` the return value is ``(vals, xlists)``.
    """
    dim = len(N)
    if dim != len(domain):
        raise ValueError("N and domain must have the same length.")
    if dim < 1 or dim > 3:
        raise ValueError("This implementation supports only 1D, 2D, or 3D.")
    if len(k_diag) != dim:
        raise ValueError("k_diag must have one entry per dimension.")
    Kcross = _k_cross_matrix(k_cross, dim, conjugate_pairs=not nonHermitian)
    if "k" in eigsh_kwargs:
        raise ValueError("Use Enum to set the number of eigenpairs, not eigsh k.")
    if "M" in eigsh_kwargs:
        raise ValueError("The FEM mass matrix is built internally; do not pass eigsh M.")
    
    xlists = [np.linspace(domain[i][0],domain[i][1],N[i]) for i in range(len(N))]

    basis = _basis_from_grid(xlists, intorder=intorder)

    # Dirichlet boundary DOFs
    D = basis.get_dofs()

    hform_decorator = fem.BilinearForm(dtype=complex) if nonHermitian else fem.BilinearForm

    @hform_decorator
    def Hform(u, v, w):
        gu = grad(u)
        gv = grad(v)
        if dim == 1:
            V = _potential_for_form(U, (w.x[0],), nonHermitian=nonHermitian)
        elif dim == 2:
            V = _potential_for_form(U, w.x, nonHermitian=nonHermitian)
        elif dim == 3:
            V = _potential_for_form(U, w.x, nonHermitian=nonHermitian)

        kin = 0.0
        for i in range(dim):
            kin = kin + k_diag[i] * gu[i] * gv[i]
        return kin + V * u * v
    

    H = Hform.assemble(basis)
    for i in range(dim):
        for j in range(i + 1, dim):
            coeff = Kcross[i, j]
            if coeff != 0.0:
                Cop = _mixed_form(i, j).assemble(basis)
                H = _add_scaled_symmetric_operator(H, Cop, coeff, nonHermitian=nonHermitian)
    M = Mform.assemble(basis)

    # Condense both matrices consistently
    Acond, Bcond, _, I = fem.condense(H, M, D=D)

    # Solve generalized eigenproblem
    # eigsh needs k < matrix size
    n_free = Acond.shape[0]
    if Enum >= n_free:
        raise ValueError(f"Enum must be smaller than number of free DOFs ({n_free}).")
    if nonHermitian and Enum >= n_free - 1:
        raise ValueError(f"Enum must be smaller than free DOFs - 1 ({n_free - 1}) when nonHermitian=True.")
    vals, vecs = _solve_eigenproblem(Acond, Enum, M=Bcond, nonHermitian=nonHermitian, **eigsh_kwargs)

    vals, vecs = _sort_eigenpairs(vals, vecs)

    if vals_only:
        return vals, xlists

    # Expand back to full vectors
    full_vecs = np.zeros((vecs.shape[1], basis.N), dtype=vecs.dtype)
    full_vecs[:, I] = vecs.T

    # Reshape onto tensor grid
    if dim == 1:
        return vals, full_vecs, xlists

    else:
        coords = basis.doflocs  # shape (dim, basis.N)

        # find integer grid indices for each dof
        inds = []
        for d in range(dim):
            xd = np.asarray(xlists[d])
            cd = coords[d]
            ind = np.searchsorted(xd, cd)
            ind = np.clip(ind, 0, len(xd) - 1)

            # fix any searchsorted off-by-one issues
            left = np.maximum(ind - 1, 0)
            choose_left = np.abs(cd - xd[left]) < np.abs(cd - xd[ind])
            ind[choose_left] = left[choose_left]

            inds.append(ind)

        vecs_out = np.zeros((vecs.shape[1], *N), dtype=vecs.dtype)

        if dim == 2:
            for j in range(vecs.shape[1]):
                vecs_out[j, inds[0], inds[1]] = full_vecs[j]

        elif dim == 3:
            for j in range(vecs.shape[1]):
                vecs_out[j, inds[0], inds[1], inds[2]] = full_vecs[j]

    
    #Normalise
    hs = [x[1] - x[0] for x in xlists]
    dV = np.prod(hs)
    for j in range(Enum):
        norm = np.sqrt(np.sum(np.abs(vecs_out[j])**2) * dV)
        if norm > 0:
            vecs_out[j] /= norm
    return vals, vecs_out, xlists


def Ceigensolver(U, H1, N=[], domain=[], k_diag=[1], k_cross=[],
                 k_coup=None, v_coup=None, Enum=1, vals_only=False,
                 intorder=None, nonHermitian=False, **eigsh_kwargs):
    """
    Solve a continuous-discrete Schrodinger problem using scikit-fem.

    The Hamiltonian is assembled as::

        H = H0(x) + H1 + Hc

    where::

        H0 = -sum_i k_ii d^2/dx_i^2
             -sum_{i<j} k_ij d^2/(dx_i dx_j)
             + U(x_i)

        H1 = sum_n E_n |n><n|

        Hc = sum_i sum_{n>m} (-k_c[i,n,m] d/dx_i + v_c[i,n,m] x_i)
             |n><m| + Hermitian conjugate

    The returned wavefunctions have components ``Psi_m(x0, x1, ...)`` in the
    discrete basis.

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
    k_cross : dict, scalar, or None, optional
        Mixed-derivative coefficients for ``H0``. Uses the same convention as
        ``eigensolver``: ``{(i, j): value}`` multiplies
        ``-d^2/(dx_i dx_j)``.
    k_coup : scalar, matrix, dict, list, or None, optional
        Derivative coupling coefficients. A scalar creates nearest-neighbour
        couplings between adjacent discrete levels along spatial axis 0. An
        ``M x M`` matrix supplies the full discrete coupling matrix along axis
        0. A dict ``{axis: scalar_or_matrix}`` chooses the spatial derivative
        axis. A list of pair dictionaries, such as
        ``[{(0, 1): value}, {(1, 2): value}]``, supplies physical pair
        coefficients for each continuous coordinate axis and fills conjugate
        entries automatically. In Hermitian mode, derivative coupling matrices
        must be skew-Hermitian because ``d/dx`` is anti-Hermitian.
    v_coup : scalar, matrix, dict, list, or None, optional
        Linear position coupling coefficients. Input forms match ``k_coup``.
        In Hermitian mode, position coupling matrices must be Hermitian.
    Enum : int, optional
        Number of eigenvalues/eigenvectors to compute.
    vals_only : bool, optional
        If True, return only eigenvalues and coordinate grids.
    intorder : int or None, optional
        Integration order passed to ``skfem.Basis``. If None, scikit-fem chooses
        its default integration order.
    nonHermitian : bool, optional
        If True, allow complex potentials and genuinely non-Hermitian terms and
        use ``scipy.sparse.linalg.eigs`` instead of ``eigsh``.
    **eigsh_kwargs
        Extra keyword arguments passed to SciPy's sparse eigensolver, such as
        ``sigma``, ``which``, ``tol``, ``maxiter``, or ``ncv``. Use ``Enum``
        rather than passing ``k`` directly. Do not pass ``M``; the FEM mass
        matrix is built internally. If no options are supplied, the default is
        ``sigma=0``.

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
    dim = len(N)
    if dim != len(domain):
        raise ValueError("N and domain must have the same length.")
    if dim < 1 or dim > 3:
        raise ValueError("This implementation supports only 1D, 2D, or 3D.")
    if len(k_diag) != dim:
        raise ValueError("k_diag must have one entry per dimension.")
    if "k" in eigsh_kwargs:
        raise ValueError("Use Enum to set the number of eigenpairs, not eigsh k.")
    if "M" in eigsh_kwargs:
        raise ValueError("The FEM mass matrix is built internally; do not pass eigsh M.")

    H1 = _validate_square_matrix(H1, "H1")
    if not nonHermitian and not np.allclose(H1.conj().T, H1):
        raise ValueError("H1 must be Hermitian.")
    Mdisc = H1.shape[0]
    Kcross = _k_cross_matrix(k_cross, dim, conjugate_pairs=not nonHermitian)
    Kcoup = _coupling_matrices(k_coup, dim, Mdisc, "k_coup", derivative=True, nonHermitian=nonHermitian)
    Vcoup = _coupling_matrices(v_coup, dim, Mdisc, "v_coup", derivative=False, nonHermitian=nonHermitian)

    xlists = [np.linspace(domain[i][0],domain[i][1],N[i]) for i in range(len(N))]
    basis = _basis_from_grid(xlists, intorder=intorder)
    D = basis.get_dofs()

    hform_decorator = fem.BilinearForm(dtype=complex) if nonHermitian else fem.BilinearForm

    @hform_decorator
    def Hform(u, v, w):
        gu = grad(u)
        gv = grad(v)
        if dim == 1:
            V = _potential_for_form(U, (w.x[0],), nonHermitian=nonHermitian)
        elif dim == 2:
            V = _potential_for_form(U, w.x, nonHermitian=nonHermitian)
        elif dim == 3:
            V = _potential_for_form(U, w.x, nonHermitian=nonHermitian)

        kin = 0.0
        for i in range(dim):
            kin = kin + k_diag[i] * gu[i] * gv[i]
        return kin + V * u * v

    H0 = Hform.assemble(basis)
    for i in range(dim):
        for j in range(i + 1, dim):
            coeff = Kcross[i, j]
            if coeff != 0.0:
                Cop = _mixed_form(i, j).assemble(basis)
                H0 = _add_scaled_symmetric_operator(H0, Cop, coeff, nonHermitian=nonHermitian)
    Mmass = Mform.assemble(basis)
    Acond, Bcond, _, I = fem.condense(H0, Mmass, D=D)

    n_free = Acond.shape[0]
    n_total = Mdisc * n_free
    if Enum >= n_total:
        raise ValueError(f"Enum must be smaller than number of free DOFs ({n_total}).")
    if nonHermitian and Enum >= n_total - 1:
        raise ValueError(f"Enum must be smaller than free DOFs - 1 ({n_total - 1}) when nonHermitian=True.")

    Idisc = sp.eye(Mdisc, format="csr")
    H = (
        sp.kron(Idisc, Acond, format="csr") +
        sp.kron(sp.csr_matrix(H1), Bcond, format="csr")
    )
    Btotal = sp.kron(Idisc, Bcond, format="csr")

    for axis, mat in Kcoup.items():
        Dop = _derivative_form(axis).assemble(basis).tocsr()[I][:, I]
        H = H + sp.kron(mat, Dop, format="csr")
    for axis, mat in Vcoup.items():
        Xop = _position_form(axis).assemble(basis).tocsr()[I][:, I]
        H = H + sp.kron(mat, Xop, format="csr")

    vals, vecs = _solve_eigenproblem(H, Enum, M=Btotal, nonHermitian=nonHermitian, **eigsh_kwargs)

    vals, vecs = _sort_eigenpairs(vals, vecs)

    if vals_only:
        return vals, xlists

    reduced = vecs.T.reshape((Enum, Mdisc, n_free), order="C")
    full_vecs = np.zeros((Enum, Mdisc, basis.N), dtype=vecs.dtype)
    full_vecs[:, :, I] = reduced

    if dim == 1:
        vecs_out = full_vecs.reshape((Enum, Mdisc, N[0]), order="C")
    else:
        coords = basis.doflocs
        inds = []
        for d in range(dim):
            xd = np.asarray(xlists[d])
            cd = coords[d]
            ind = np.searchsorted(xd, cd)
            ind = np.clip(ind, 0, len(xd) - 1)
            left = np.maximum(ind - 1, 0)
            choose_left = np.abs(cd - xd[left]) < np.abs(cd - xd[ind])
            ind[choose_left] = left[choose_left]
            inds.append(ind)

        vecs_out = np.zeros((Enum, Mdisc, *N), dtype=vecs.dtype)
        if dim == 2:
            for j in range(Enum):
                for m in range(Mdisc):
                    vecs_out[j, m, inds[0], inds[1]] = full_vecs[j, m]
        elif dim == 3:
            for j in range(Enum):
                for m in range(Mdisc):
                    vecs_out[j, m, inds[0], inds[1], inds[2]] = full_vecs[j, m]

    hs = [x[1] - x[0] for x in xlists]
    dV = np.prod(hs)
    for j in range(Enum):
        norm = np.sqrt(np.sum(np.abs(vecs_out[j])**2) * dV)
        if norm > 0:
            vecs_out[j] /= norm

    return vals, vecs_out, xlists
