import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigs


def guided_modes_1DTE(prm, k0, h):
    """
    Computes the effective permittivities of TE polarized guided eigenmodes.

    Solves the eigenvalue problem derived from the discretized Helmholtz equation:

        (d²/dx² + k0² * eps(x)) * Ey = eps_eff * k0² * Ey

    The operator is discretized on a uniform grid using central differences,
    yielding a tridiagonal matrix eigenvalue problem M * v = eps_eff * v.

    All dimensions are in µm.

    Args:
        prm (numpy.ndarray): Dielectric permittivity distribution along x, shape (n,).
        k0 (float): Free-space wavenumber in µm⁻¹.
        h (float): Spatial grid spacing in µm.

    Returns:
        eff_eps (numpy.ndarray): Effective permittivities of all eigenmodes,
            sorted in descending order of real part.
        guided (numpy.ndarray): Corresponding field distributions (Ey),
            shape (n, n), each column is one eigenmode.
    """

    n = len(prm)

    # tridiagonal operator: central-difference Laplacian + k0² * eps(x), normalised by k0²
    main = -2 * np.ones(n) / (k0 * h)**2 + prm
    sec = np.ones(n-1) / (k0 * h)**2
    M = np.diag(sec, -1) + np.diag(main, 0) + np.diag(sec, 1)
    M = np.around(M, 6)

    eff_eps, guided = np.linalg.eig(M)
    idx = np.argsort(-eff_eps.real)  # sort by descending real part

    return eff_eps[idx], guided[:, idx]


def guided_modes_2D(prm, k0, h, numb):
    """
    Computes the effective permittivities of quasi-TE polarized guided eigenmodes
    in a 2D waveguide cross-section.

    Solves the eigenvalue problem derived from the discretized 2D Helmholtz equation:

        (d²/dx² + d²/dy² + k0² * eps(x,y)) * Ez = eps_eff * k0² * Ez

    The operator is discretized on a uniform grid using central differences,
    yielding a sparse pentadiagonal matrix eigenvalue problem M * v = eps_eff * v.
    Spurious couplings across column boundaries are explicitly removed.

    All dimensions are in µm.

    Args:
        prm (numpy.ndarray): Dielectric permittivity distribution in the xy-plane,
            shape (m, n).
        k0 (float): Free-space wavenumber in µm⁻¹.
        h (float): Spatial grid spacing in µm (uniform in x and y).
        numb (int): Number of eigenmodes to compute.

    Returns:
        eff_eps (numpy.ndarray): Effective permittivities of the computed eigenmodes,
            shape (numb,).
        guided (numpy.ndarray): Corresponding field distributions,
            shape (numb, m, n), each entry along axis 0 is one eigenmode.
    """

    m, n = prm.shape
    count = m * n  # total number of grid points
    # flatten permittivity in column-major order to match sparse matrix layout
    prm_flat = prm.ravel(order='F')

    # build pentadiagonal operator: central-difference 2D Laplacian + k0² * eps, normalised by k0²
    main = -2 * np.ones(count) * 2 / h**2 + k0**2 * prm_flat
    ex = np.ones(count) / h**2
    data = np.array([ex, ex, main, ex, ex])
    offsets = np.array([-m, -1, 0, 1, m])
    M = sps.spdiags(data, offsets, count, count, format='csc')

    # remove spurious off-diagonal couplings introduced across column boundaries by the ±1 diagonals
    # (column-major ordering connects last row of col i to first row of col i+1)
    for i in range(1, n):
        idx = i * m
        M[idx-1, idx] = 0
        M[idx, idx-1] = 0

    eig_vals, eig_vecs = eigs(M, k=numb, which='LM')
    eff_eps = eig_vals / k0**2  # convert eigenvalues back to effective permittivity

    # reshape eigenvectors from flat column-major vectors to 2D field distributions
    guided = np.zeros((len(eff_eps), m, n), dtype=eig_vecs.dtype)
    for i in range(len(eff_eps)):
        guided[i, :, :] = np.reshape(eig_vecs[:, i], (m, n), order='F')

    return eff_eps, guided
