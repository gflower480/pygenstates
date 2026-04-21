import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh


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


def _potential_values(U, grids, shape, nonHermitian=False):
    Vvals = np.asarray(U(*grids), dtype=complex)
    if Vvals.shape == ():
        Vvals = np.full(shape, Vvals.item(), dtype=complex)
    if Vvals.size != int(np.prod(shape)):
        raise ValueError("U must return a scalar or an array matching the interior grid shape.")
    if not nonHermitian:
        if np.any(np.abs(np.imag(Vvals)) > 1e-12):
            raise ValueError("Complex potentials make the Hamiltonian non-Hermitian; set nonHermitian=True to use eigs.")
        Vvals = np.real(Vvals)
    return Vvals.reshape(-1)


def _kron_all(mats):
    out = mats[0]
    for mat in mats[1:]:
        out = sp.kron(out, mat, format="csr")
    return out


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


def _coupling_matrices(coup, dim, M, name, derivative=False, nonHermitian=False):
    if coup is None:
        return {}
    if isinstance(coup, dict):
        items = coup.items()
    else:
        items = [(0, coup)]

    mats = {}
    for axis, value in items:
        axis = int(axis)
        if axis < 0 or axis >= dim:
            raise ValueError(f"{name} axis {axis} is out of range for dimension {dim}.")

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


def _solve_eigenproblem(A, Enum, nonHermitian=False, **solver_kwargs):
    solver_kwargs = dict(solver_kwargs) if solver_kwargs else {"sigma": 0}
    if nonHermitian:
        return eigs(A, k=Enum, **solver_kwargs)
    return eigsh(A, k=Enum, **solver_kwargs)


def _sort_eigenpairs(vals, vecs):
    idx = np.lexsort((np.imag(vals), np.real(vals)))
    return vals[idx], vecs[:, idx]


def eigensolver(U, N=[], domain=[], k_diag=[1], k_cross=[],
                   Enum=1, vals_only=False, nonHermitian=False, **eigsh_kwargs):
    """
    Solve the time-independent Schrodinger equation

        (-sum_i k_i d^2/dx_i^2 + sum_{i<j} c_ij d^2/(dx_i dx_j) + U) psi = E psi

    on an n-dimensional box using finite differences on a tensor-product grid.

    Parameters
    ----------
    U : callable
        Potential function. Should accept coordinates as separate arguments:
            1D: U(x)
            2D: U(x, y)
            3D: U(x, y, z)
        and return an array of matching shape.
    N : list[int]
        Number of grid points in each dimension.
    domain : list[tuple[float, float]]
        Domain bounds in each dimension, same length as N.
    k_diag : list[float]
        Diagonal kinetic coefficients in each dimension.
        The operator is:
            -sum_i k_diag[i] * d^2/dx_i^2 + U
    k_cross : list
        Mixed derivative coefficients c_ij. Empty/None means no cross terms.
        Accepted forms are:
            dict {(i, j): value}
            scalar in 2D
    Enum : int
        Number of eigenvalues/eigenvectors to compute.
    vals_only : bool
        If True, return only eigenvalues.
    nonHermitian : bool
        If True, allow complex potentials and genuinely non-Hermitian terms and
        use scipy eigs instead of eigsh. Complex Hermitian-safe couplings are
        supported on the default eigsh path.
    **eigsh_kwargs
        Optional keyword arguments passed through to scipy.sparse.linalg.eigsh/eigs,
        such as sigma, which, tol, maxiter, or ncv. If no eigsh options are
        supplied, sigma=0 is used to preserve the previous default.

    Returns
    -------
    vals : ndarray
        Eigenvalues.
    vecs_out : ndarray
        Eigenvectors reshaped onto the full tensor grid, with boundary values zero.
        Shape:
            (Enum, *N)
        Returned only when vals_only=False.
    xlists : list[ndarray]
        Coordinate arrays in each dimension.
    """
    dim = len(N)

    if dim != len(domain):
        raise ValueError("N and domain must have the same length.")
    if dim < 1:
        raise ValueError("This implementation needs at least one dimension.")
    if len(k_diag) != dim:
        raise ValueError("k_diag must have one entry per dimension.")
    Kcross = _k_cross_matrix(k_cross, dim, conjugate_pairs=not nonHermitian)
    if any(n < 3 for n in N):
        raise ValueError("Each N[i] must be at least 3 to allow interior points.")
    if "k" in eigsh_kwargs:
        raise ValueError("Use Enum to set the number of eigenpairs, not eigsh k.")

    xlists = [np.linspace(domain[i][0], domain[i][1], N[i]) for i in range(dim)]
    hs = [x[1] - x[0] for x in xlists]

    # ------------------------------------------------------------------
    # Matrix build
    # ------------------------------------------------------------------
    # Interior sizes (Dirichlet boundary values are fixed to zero)
    Ni = [n - 2 for n in N]
    n_free = int(np.prod(Ni))

    if Enum >= n_free:
        raise ValueError(f"Enum must be smaller than number of free DOFs ({n_free}).")
    if nonHermitian and Enum >= n_free - 1:
        raise ValueError(f"Enum must be smaller than free DOFs - 1 ({n_free - 1}) when nonHermitian=True.")

    # 1D second-difference matrices on interior points:
    #   (-d^2/dx^2) ~ (2u_i - u_{i-1} - u_{i+1}) / h^2
    Ls = []
    Ds = []
    Is = []
    xints = []

    for d in range(dim):
        nd = Ni[d]
        hd = hs[d]

        main = 2.0 * np.ones(nd) / hd**2
        off = -1.0 * np.ones(nd - 1) / hd**2

        Ld = sp.diags(
            diagonals=[off, main, off],
            offsets=[-1, 0, 1],
            shape=(nd, nd),
            format="csr"
        )

        Doff_p = 0.5 * np.ones(nd - 1) / hd
        Doff_m = -0.5 * np.ones(nd - 1) / hd
        Dd = sp.diags(
            diagonals=[Doff_m, Doff_p],
            offsets=[-1, 1],
            shape=(nd, nd),
            format="csr"
        )

        Id = sp.eye(nd, format="csr")

        Ls.append(Ld)
        Ds.append(Dd)
        Is.append(Id)
        xints.append(xlists[d][1:-1])

    # Build kinetic operator by Kronecker sum, plus optional mixed derivatives.
    T = sp.csr_matrix((n_free, n_free))
    for d in range(dim):
        mats = [Ls[d] if axis == d else Is[axis] for axis in range(dim)]
        T = T + k_diag[d] * _kron_all(mats)

    for i in range(dim):
        for j in range(i + 1, dim):
            coeff = Kcross[i, j]
            if coeff != 0.0:
                mats = [
                    Ds[axis] if axis == i or axis == j else Is[axis]
                    for axis in range(dim)
                ]
                T = _add_scaled_symmetric_operator(T, _kron_all(mats), coeff, nonHermitian=nonHermitian)

    grids = np.meshgrid(*xints, indexing="ij")
    Vvals = _potential_values(U, grids, Ni, nonHermitian=nonHermitian)

    Vmat = sp.diags(Vvals, offsets=0, format="csr")
    H = T + Vmat

    # ------------------------------------------------------------------
    # Eigensolve
    # ------------------------------------------------------------------
    # This is a standard eigenproblem, no mass matrix.
    vals, vecs = _solve_eigenproblem(H, Enum, nonHermitian=nonHermitian, **eigsh_kwargs)

    # ------------------------------------------------------------------
    # Postprocess
    # ------------------------------------------------------------------
    vals, vecs = _sort_eigenpairs(vals, vecs)

    if vals_only:
        return vals, xlists

    # Fill full grid with zeros on the boundary.
    vecs_out = np.zeros((Enum, *N), dtype=vecs.dtype)
    interior = (slice(None),) + (slice(1, -1),) * dim
    vecs_out[interior] = vecs.T.reshape((Enum, *Ni), order='C')

    #Normalise
    dV = np.prod(hs)
    for j in range(Enum):
        norm = np.sqrt(np.sum(np.abs(vecs_out[j])**2) * dV)
        if norm > 0:
            vecs_out[j] /= norm

    return vals, vecs_out, xlists


def Ceigensolver(U, H1, N=[], domain=[], k_diag=[1], k_cross=[],
                 k_coup=None, v_coup=None, Enum=1, vals_only=False,
                 nonHermitian=False, **eigsh_kwargs):
    """
    Solve a coupled continuous-discrete Schrodinger problem using finite differences.

        H = H0(x) + H1 + Hc

    H1 is an MxM matrix acting on the discrete basis. The returned wavefunctions
    have components Psi_m(x0, x1, ...), with shape (Enum, M, *N).

    k_coup gives derivative couplings and v_coup gives linear position couplings.
    Each may be None, a scalar, an MxM matrix, or a dict {axis: scalar_or_matrix}.
    Scalar derivative couplings create nearest-neighbour skew-Hermitian matrices;
    scalar position couplings create nearest-neighbour Hermitian matrices.
    Complex Hermitian-safe couplings are supported on the default eigsh path.
    If nonHermitian=True, complex potentials and non-Hermitian coupling matrices
    are allowed and scipy eigs is used instead of eigsh.
    """
    dim = len(N)

    if dim != len(domain):
        raise ValueError("N and domain must have the same length.")
    if dim < 1:
        raise ValueError("This implementation needs at least one dimension.")
    if len(k_diag) != dim:
        raise ValueError("k_diag must have one entry per dimension.")
    if any(n < 3 for n in N):
        raise ValueError("Each N[i] must be at least 3 to allow interior points.")
    if "k" in eigsh_kwargs:
        raise ValueError("Use Enum to set the number of eigenpairs, not eigsh k.")

    H1 = _validate_square_matrix(H1, "H1")
    if not nonHermitian and not np.allclose(H1.conj().T, H1):
        raise ValueError("H1 must be Hermitian.")
    Mdisc = H1.shape[0]
    Kcross = _k_cross_matrix(k_cross, dim, conjugate_pairs=not nonHermitian)
    Kcoup = _coupling_matrices(k_coup, dim, Mdisc, "k_coup", derivative=True, nonHermitian=nonHermitian)
    Vcoup = _coupling_matrices(v_coup, dim, Mdisc, "v_coup", derivative=False, nonHermitian=nonHermitian)

    xlists = [np.linspace(domain[i][0], domain[i][1], N[i]) for i in range(dim)]
    hs = [x[1] - x[0] for x in xlists]

    Ni = [n - 2 for n in N]
    n_free = int(np.prod(Ni))
    n_total = Mdisc * n_free

    if Enum >= n_total:
        raise ValueError(f"Enum must be smaller than number of free DOFs ({n_total}).")
    if nonHermitian and Enum >= n_total - 1:
        raise ValueError(f"Enum must be smaller than free DOFs - 1 ({n_total - 1}) when nonHermitian=True.")

    Ls = []
    Ds_1d = []
    Is = []
    xints = []

    for d in range(dim):
        nd = Ni[d]
        hd = hs[d]

        main = 2.0 * np.ones(nd) / hd**2
        off = -1.0 * np.ones(nd - 1) / hd**2
        Ls.append(sp.diags([off, main, off], [-1, 0, 1], shape=(nd, nd), format="csr"))

        Doff_p = 0.5 * np.ones(nd - 1) / hd
        Doff_m = -0.5 * np.ones(nd - 1) / hd
        Ds_1d.append(sp.diags([Doff_m, Doff_p], [-1, 1], shape=(nd, nd), format="csr"))

        Is.append(sp.eye(nd, format="csr"))
        xints.append(xlists[d][1:-1])

    H0 = sp.csr_matrix((n_free, n_free))
    for d in range(dim):
        mats = [Ls[d] if axis == d else Is[axis] for axis in range(dim)]
        H0 = H0 + k_diag[d] * _kron_all(mats)

    for i in range(dim):
        for j in range(i + 1, dim):
            coeff = Kcross[i, j]
            if coeff != 0.0:
                mats = [Ds_1d[axis] if axis == i or axis == j else Is[axis] for axis in range(dim)]
                H0 = _add_scaled_symmetric_operator(H0, _kron_all(mats), coeff, nonHermitian=nonHermitian)

    grids = np.meshgrid(*xints, indexing="ij")
    Vvals = _potential_values(U, grids, Ni, nonHermitian=nonHermitian)
    H0 = H0 + sp.diags(Vvals, offsets=0, format="csr")

    Idisc = sp.eye(Mdisc, format="csr")
    Isp = sp.eye(n_free, format="csr")
    H = sp.kron(Idisc, H0, format="csr") + sp.kron(sp.csr_matrix(H1), Isp, format="csr")

    derivative_ops = {}
    position_ops = {}
    for axis in sorted(set(Kcoup) | set(Vcoup)):
        mats = [Ds_1d[axis] if d == axis else Is[d] for d in range(dim)]
        derivative_ops[axis] = _kron_all(mats)
        position_ops[axis] = sp.diags(grids[axis].reshape(-1), offsets=0, format="csr")

    for axis, mat in Kcoup.items():
        H = H + sp.kron(mat, derivative_ops[axis], format="csr")
    for axis, mat in Vcoup.items():
        H = H + sp.kron(mat, position_ops[axis], format="csr")

    vals, vecs = _solve_eigenproblem(H, Enum, nonHermitian=nonHermitian, **eigsh_kwargs)

    vals, vecs = _sort_eigenpairs(vals, vecs)

    if vals_only:
        return vals, xlists

    vecs_out = np.zeros((Enum, Mdisc, *N), dtype=vecs.dtype)
    interior = (slice(None), slice(None)) + (slice(1, -1),) * dim
    vecs_out[interior] = vecs.T.reshape((Enum, Mdisc, *Ni), order="C")

    dV = np.prod(hs)
    for j in range(Enum):
        norm = np.sqrt(np.sum(np.abs(vecs_out[j])**2) * dV)
        if norm > 0:
            vecs_out[j] /= norm

    return vals, vecs_out, xlists
