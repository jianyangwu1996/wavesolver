import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve


def waveguide(xa, xb, Nx, n_cladding, n_core):
    """
    Generates the refractive index distribution of a step-index slab waveguide
    centered at the origin, with refractive index n_core in the core region
    and n_cladding in the surrounding cladding.

    All lengths must be specified in µm.

    Args:
        xa (float): Total width of the simulation window in µm.
        xb (float): Width of the waveguide core in µm.
        Nx (int): Number of transverse grid points.
        n_cladding (float): Refractive index of the cladding region.
        n_core (float): Refractive index of the core region.

    Returns:
        n (numpy.ndarray): Refractive index distribution, shape (Nx,).
        x (numpy.ndarray): Transverse coordinate vector in µm, shape (Nx,).
    """

    # generate transverse coordinate vector centered at origin, ranging from -xa/2 to xa/2
    x = np.linspace(-xa / 2, xa / 2, Nx)
    n = np.ones(Nx) * n_cladding
    # set core index for all grid points within the core region (|x| <= xb/2)
    idx_core = ((x <= xb / 2) & (x >= -xb / 2))
    n[idx_core] = n_core

    return n, x


def gauss(xa, Nx, w):
    """
    Generates a Gaussian field distribution centered at the origin.

    The field profile is defined as:

        v(x) = exp(-x^2 / w^2)

    All lengths must be specified in µm.

    Args:
        xa (float): Total width of the simulation window in µm.
        Nx (int): Number of transverse grid points.
        w (float): 1/e half-width of the Gaussian field in µm.

    Returns:
        v (numpy.ndarray): Gaussian field distribution, shape (Nx,).
        x (numpy.ndarray): Transverse coordinate vector in µm, shape (Nx,).
    """

    x = np.linspace(-xa / 2, xa / 2, Nx)
    v = np.exp(-x ** 2 / w ** 2)  # Gaussian field amplitude profile

    return v, x


def beamprop_cn(v_in, lam, dx, n, nd, z_end, dz, output_step=1):
    """Propagates an initial field over a given distance using the Crank-Nicolson
    finite-difference scheme applied to the paraxial wave equation in an
    inhomogeneous refractive index distribution.

    The paraxial wave equation is discretised as:

        (I - 0.5 * dz_out * L) * v^(i) = (I + 0.5 * dz_out * L) * v^(i-1)

    where L is the finite-difference operator containing the diffraction and
    phase terms:

        L_jj = -2 * alpha + i * W_j,    L_{j,j±1} = alpha
        alpha = i / (2 * k_ref * dx^2)
        W_j   = (k_j^2 - k_ref^2) / (2 * k_ref)

    All lengths must be specified in µm.

    Args:
        v_in (numpy.ndarray): Initial complex field distribution, shape (Nx,).
        lam (float): Free-space wavelength in µm.
        dx (float): Transverse grid spacing in µm.
        n (numpy.ndarray): Refractive index distribution, shape (Nx,).
        nd (float): Reference refractive index for the slowly-varying envelope.
        z_end (float): Total propagation distance in µm.
        dz (float): Longitudinal step size in µm.
        output_step (int): Number of steps between successive field outputs.

    Returns:
        v (numpy.ndarray): Propagated complex field, shape (Nx, len(z)).
        z (numpy.ndarray): z-coordinates of field output snapshots in µm, shape (len(z),).
    """

    dz_out = dz * output_step  # effective output step size
    z = np.arange(0, z_end, dz_out)

    N = len(v_in)
    k = 2 * np.pi / lam * n  # local wavenumber at each grid point
    # reference wavenumber for the slowly-varying envelope approximation
    k_mean = 2 * np.pi / lam * nd
    # phase correction term W_j = (k_j^2 - k_ref^2) / (2 * k_ref)
    W = (k ** 2 - k_mean ** 2) / (2 * k_mean)
    # off-diagonal coefficient alpha = i / (2 * k_ref * dx^2), encodes transverse diffraction
    sec = (1j / (2 * k_mean * dx ** 2)) * np.ones(N)
    # main diagonal: -2*alpha accounts for the second-order finite difference
    main = -2 * sec + 1j * W
    data = np.array([sec, main, sec])
    offsets = np.array([-1, 0, 1])
    L = sps.dia_array((data, offsets), shape=(N, N)).tocsc()

    eye = sps.eye(N)
    # left-hand side matrix A = I - 0.5 * dz_out * L  (implicit part)
    A = eye - 0.5 * dz_out * L
    # right-hand side matrix B = I + 0.5 * dz_out * L  (explicit part)
    B = eye + 0.5 * dz_out * L

    v = np.zeros((N, len(z)), dtype=complex)
    v[:, 0] = v_in
    # march forward in z: solve A * v[:,i] = B * v[:,i-1]
    for i in range(1, len(z)):
        v[:, i] = spsolve(A, B @ v[:, i-1])

    return v, z
