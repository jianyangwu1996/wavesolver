import numpy as np
import time


def fdtd_1d(eps_rel, dx, time_span, source_frequency, source_position,
            source_pulse_length):
    """
    Computes the temporal evolution of a pulsed excitation using the
    1D FDTD method on a Yee grid with PEC boundary conditions.

    The electric and magnetic fields are updated via the leapfrog scheme:

        Ez^{n+1}_{j} = Ez^{n}_{j} + (dt / eps0 / eps_j / dx) * (Hy^{n}_{j} - Hy^{n}_{j-1})
        Hy^{n+1}_{j} = Hy^{n}_{j} + (dt / mu0 / dx) * (Ez^{n+1}_{j+1} - Ez^{n+1}_{j})

    A soft current source injects a Gaussian-modulated sinusoidal pulse:

        j(t) = exp(-i * 2*pi*f * t) * exp(-(t / tau)^2)

    centered at t0 = 3 * source_pulse_length. After the simulation, Hy is
    interpolated onto the Ez grid (in both time and space) to allow direct
    computation of the Poynting vector S = Ez x Hy.

    All quantities must be specified in SI units.

    Args:
        eps_rel (numpy.ndarray): Relative permittivity distribution, shape (Nx,).
        dx (float): Spatial grid spacing in meters. Should satisfy dx <= lambda / 20.
        time_span (float): Total simulation time in seconds.
        source_frequency (float): Carrier frequency of the current source in Hz.
        source_position (float): Spatial position of the current source in meters.
        source_pulse_length (float): 1/e half-width of the Gaussian temporal envelope in seconds.

    Returns:
        Ez (numpy.ndarray): Z-component of the electric field E(x, t), shape (Nt+1, Nx).
            Each row corresponds to one time step.
        Hy (numpy.ndarray): Y-component of the magnetic field H(x, t), shape (Nt+1, Nx),
            interpolated onto the Ez grid for Poynting vector computation.
        x (numpy.ndarray): Spatial coordinates in meters, shape (Nx,).
        t (numpy.ndarray): Time coordinates in seconds, shape (Nt+1,).
    """

    # --- Physical constants ---
    c = 2.99792458e8  # speed of light [m/s]
    mu0 = 4 * np.pi * 1e-7  # vacuum permeability [H/m]
    eps0 = 1 / (mu0 * c ** 2)  # vacuum permittivity [F/m]
    dt = dx / (2 * c)
    # Update coefficients for Ez and Hy leapfrog equations
    e = dt / eps0  # E-field update prefactor [m/(F·s) -> V·m/A]
    m = dt / mu0  # H-field update prefactor [m/(H·s) -> A·m/V]

    # --- Grid setup ---
    Nx = eps_rel.size
    x_span = 18e-6
    x = np.linspace(-x_span / 2, x_span / 2, Nx)
    Nt = int(round(time_span / dt))
    t = np.arange(Nt + 1) * dt

    # --- Field arrays (PEC boundary: Ez=0 at x[0] and x[-1]) ---
    Ez = np.zeros((Nt + 1, Nx), dtype='complex64')
    Hy = np.zeros((Nt + 1, Nx - 1), dtype='complex64')

    # --- Source parameters ---
    t0 = 3 * source_pulse_length
    source_ind = int(round((source_position - x[0]) / dx))

    # --- Leapfrog time-stepping ---
    for n in range(0, Nt):
        # 1) Update Ez at interior nodes (PEC: Ez[0] and Ez[-1] remain zero)
        Ez[n + 1, 1:-1] = (Ez[n, 1:-1]
                           + e / dx * (Hy[n, 1:] - Hy[n, :-1]) / eps_rel[1:-1])

        # 2) Inject soft current source at half time step n+1/2
        #    j(t) = exp(-i*2π*f*t) * exp(-(t/τ)²)
        t_source = (n + 0.5) * dt - t0
        j_source = (np.exp(-1j * 2 * np.pi * source_frequency * t_source)  # carrier
                    * np.exp(-(t_source / source_pulse_length) ** 2))  # envelope
        Ez[n + 1, source_ind] -= e / eps_rel[source_ind] * j_source

        # 3) Update Hy from the updated Ez (standard Yee leapfrog)
        Hy[n + 1, :] = Hy[n, :] + m / dx * (Ez[n + 1, 1:] - Ez[n + 1, :-1])

    # --- Interpolate Hy onto the Ez grid for Poynting vector S = Ez × Hy ---
    # Hy in the Yee scheme lives at half-integer spatial (j+1/2) and temporal (n+1/2) nodes.
    # interpolate Hy in temporal domain
    Hy[1:, :] = 0.5 * (Hy[:-1, :] + Hy[1:, :])
    # additional two columns with zeros at two edges of boundary.
    Hy = np.pad(Hy, ((0, 0), (1, 1)), 'edge')
    # interpolate Hy in spatial domain
    Hy = 0.5 * (Hy[:, 1:] + Hy[:, :-1])

    return Ez, Hy, x, t


def fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz, field_component, z_index, output_step, dt=None):
    """
    Computes the temporal evolution of a pulsed spatially extended current
    source using the 3D FDTD method on a Yee grid with PEC boundary conditions.

    The six field components (Ex, Ey, Ez, Hx, Hy, Hz) are updated via the
    standard Yee leapfrog scheme. The current source is a Gaussian-modulated
    sinusoidal pulse evaluated at half time steps:

        j(t) = exp(-i * 2*pi*f * t) * exp(-((t - t0) / tau)^2)

    centered at t0 = 3 * tau. Permittivity is averaged at shared Yee-cell
    faces to handle material interfaces. The selected field component is
    interpolated onto a common spatial grid and stored as a z-slice every
    output_step time steps.

    All quantities must be specified in SI units.

    Args:
        eps_rel (numpy.ndarray): Relative permittivity distribution, shape (Nx, Ny, Nz).
        dr (float): Uniform grid spacing in meters. Should satisfy dr <= lambda / 20.
            If this condition is violated, dr is clamped automatically.
        time_span (float): Total simulation time in seconds.
        freq (float): Center frequency of the current source in Hz.
        tau (float): 1/e half-width of the Gaussian temporal envelope in seconds.
        jx, jy, jz (numpy.ndarray): the current source density, shape (Nx, Ny, Nz).
        field_component (str): Field component to store, one of 'ex', 'ey', 'ez', 'hx', 'hy', 'hz'.
        z_index (int): Z-index of the output slice.
        output_step (int): Number of time steps between successive field outputs.
        dt (float, optional): Time step in seconds. Defaults to dr / (2 * c) if None.

    Returns:
        F (numpy.ndarray): Z-slices of the selected field component, shape (Nt, Nx, Ny).
            F[0] stores the initial field (zeros at t=0); subsequent frames
            are written from F[1] onward. Time varies along the first axis.
        t (numpy.ndarray): Time coordinates of the field output in seconds, shape (Nt,).
    """

    # --- Physical constants ---
    c = 2.99792458e8  # speed of light [m/s]
    mu0 = 4 * np.pi * 1e-7  # vacuum permeability [H/m]
    eps0 = 1 / (mu0 * c ** 2)  # vacuum permittivity [F/m]

    # --- Grid spacing: clamp dr to lambda/20 if Courant condition is violated ---
    lam = c / freq
    if dr > lam / 20:
        import warnings
        warnings.warn(f"dr={dr} exceeds lambda/20, clamping to {lam / 20:.3e}")
        dr = lam / 20

    if dt is None:
        dt = dr / (2 * c)

    Niter = int(round(time_span / dt))
    t = np.linspace(0, time_span, Niter // output_step + 1)
    Nt = len(t)

    # --- Allocate Yee-grid field arrays ---
    # E and H components are offset by half a cell in their respective directions
    Nx, Ny, Nz = eps_rel.shape
    Ex = np.zeros((Nx - 1, Ny, Nz), dtype='complex64')
    Ey = np.zeros((Nx, Ny - 1, Nz), dtype='complex64')
    Ez = np.zeros((Nx, Ny, Nz - 1), dtype='complex64')
    Hx = np.zeros((Nx, Ny - 1, Nz - 1), dtype='complex64')
    Hy = np.zeros((Nx - 1, Ny, Nz - 1), dtype='complex64')
    Hz = np.zeros((Nx - 1, Ny - 1, Nz), dtype='complex64')

    # --- Precompute face-averaged inverse permittivity for each E component ---
    epsx_rec = (1 / eps_rel[:-1, :, :] + 1 / eps_rel[1:, :, :]) / 2
    epsx_rec = epsx_rec[0:Nx - 1, 1:Ny - 1, 1:Nz - 1]
    epsy_rec = (1 / eps_rel[:, :-1, :] + 1 / eps_rel[:, 1:, :]) / 2
    epsy_rec = epsy_rec[1:Nx - 1, 0:Ny - 1, 1:Nz - 1]
    epsz_rec = (1 / eps_rel[:, :, :-1] + 1 / eps_rel[:, :, 1:]) / 2
    epsz_rec = epsz_rec[1:Nx - 1, 1:Ny - 1, 0:Nz - 1]

    # --- Interpolate current source onto Yee E-field nodes ---
    # Each current component is averaged to align with its corresponding E node
    jx = ((jx[:-1, :, :] + jx[1:, :, :]) / 2)
    jy = ((jy[:, :-1, :] + jy[:, 1:, :]) / 2)
    jz = ((jz[:, :, :-1] + jz[:, :, 1:]) / 2)

    F = np.zeros((Nt, Nx, Ny), dtype='complex64')
    count = 0

    # --- Leapfrog time-stepping ---
    for n in range(Niter):
        # Current source envelope evaluated at half time step n+1/2
        t_source = dt * (n + 1 / 2) - 3 * tau
        jx_n = jx * np.exp(-2j * np.pi * freq * t_source) * np.exp(-(t_source / tau) ** 2)
        jy_n = jy * np.exp(-2j * np.pi * freq * t_source) * np.exp(-(t_source / tau) ** 2)
        jz_n = jz * np.exp(-2j * np.pi * freq * t_source) * np.exp(-(t_source / tau) ** 2)

        # 1) Update E fields from curl(H) - J (interior nodes only; PEC on boundary)
        Ex[0:Nx - 1, 1:Ny - 1, 1:Nz - 1] += (dt / eps0 * epsx_rec *
                                             ((Hz[0:Nx - 1, 1:Ny - 1, 1:Nz - 1] -
                                               Hz[0:Nx - 1, 0:Ny - 2, 1:Nz - 1]) / dr -
                                              (Hy[0:Nx - 1, 1:Ny - 1, 1:Nz - 1] -
                                               Hy[0:Nx - 1, 1:Ny - 1, 0:Nz - 2]) / dr -
                                              jx_n[0:Nx - 1, 1:Ny - 1, 1:Nz - 1]))
        Ey[1:Nx - 1, 0:Ny - 1, 1:Nz - 1] += (dt / eps0 * epsy_rec *
                                             ((Hx[1:Nx - 1, 0:Ny - 1, 1:Nz - 1] -
                                               Hx[1:Nx - 1, 0:Ny - 1, 0:Nz - 2]) / dr -
                                              (Hz[1:Nx - 1, 0:Ny - 1, 1:Nz - 1] -
                                               Hz[0:Nx - 2, 0:Ny - 1, 1:Nz - 1]) / dr -
                                              jy_n[1:Nx - 1, 0:Ny - 1, 1:Nz - 1]))
        Ez[1:Nx - 1, 1:Ny - 1, 0:Nz - 1] += (dt / eps0 * epsz_rec *
                                             ((Hy[1:Nx - 1, 1:Ny - 1, 0:Nz - 1] -
                                               Hy[0:Nx - 2, 1:Ny - 1, 0:Nz - 1]) / dr -
                                              (Hx[1:Nx - 1, 1:Ny - 1, 0:Nz - 1] -
                                               Hx[1:Nx - 1, 0:Ny - 2, 0:Nz - 1]) / dr -
                                              jz_n[1:Nx - 1, 1:Ny - 1, 0:Nz - 1]))

        # Capture H field before update for temporal averaging (H components only)
        # temp holds H_n, which will be averaged with H_n+1 after the update
        # to approximate H at integer time steps (co-located with E).
        if field_component == 'hx':
            temp = Hx[:, 0:Ny - 1, 0:Nz - 1].copy()
        elif field_component == 'hy':
            temp = Hy[0:Nx - 1, :, 0:Nz - 1].copy()
        elif field_component == 'hz':
            temp = Hz[0:Nx - 1, 0:Ny - 1, :].copy()

        # Update H fields from curl(E)
        Hx[1:Nx - 1, 0:Ny - 1, 0:Nz - 1] += (
                dt / mu0 * ((Ey[1:Nx - 1, 0:Ny - 1, 1:Nz] - Ey[1:Nx - 1, 0:Ny - 1, 0:Nz - 1]) / dr -
                            (Ez[1:Nx - 1, 1:Ny, 0:Nz - 1] - Ez[1:Nx - 1, 0:Ny - 1, 0:Nz - 1]) / dr))
        Hy[0:Nx - 1, 1:Ny - 1, 0:Nz - 1] += (
                dt / mu0 * ((Ez[1:Nx, 1:Ny - 1, 0:Nz - 1] - Ez[0:Nx - 1, 1:Ny - 1, 0:Nz - 1]) / dr -
                            (Ex[0:Nx - 1, 1:Ny - 1, 1:Nz] - Ex[0:Nx - 1, 1:Ny - 1, 0:Nz - 1]) / dr))
        Hz[0:Nx - 1, 0:Ny - 1, 1:Nz - 1] += (
            (dt / mu0 * (Ex[0:Nx - 1, 1:Ny, 1:Nz - 1] - Ex[0:Nx - 1, 0:Ny - 1, 1:Nz - 1]) / dr -
             (Ey[1:Nx, 0:Ny - 1, 1:Nz - 1] - Ey[0:Nx - 1, 0:Ny - 1, 1:Nz - 1]) / dr))

        # Store output slice every output_step steps
        if (n + 1) % output_step == 0:
            count += 1
            # Interpolate selected field component onto a common Nx x Ny grid.
            # E fields need 1D spatial interpolation along their offset axis;
            # H fields additionally need temporal averaging using temp (H_n)
            # and the just-updated H_n+1 to approximate H at integer time steps.
            if field_component == 'ex':
                res = Ex[0:Nx - 1, :, :]
                res = np.pad(res, ((1, 1), (0, 0), (0, 0)), 'edge')
                res = (res[:-1, ...] + res[1:, ...]) * 0.5
            elif field_component == 'ey':
                res = Ey[:, 0:Ny - 1, :]
                res = np.pad(res, ((0, 0), (1, 1), (0, 0)), 'edge')
                res = (res[:, :-1, :] + res[:, 1:, :]) * 0.5
            elif field_component == 'ez':
                res = Ez[:, :, 0:Nz - 1]
                res = np.pad(res, ((0, 0), (0, 0), (1, 1)), 'edge')
                res = (res[..., :-1] + res[..., 1:]) * 0.5
            elif field_component == 'hx':
                res = (Hx[:, 0:Ny - 1, 0:Nz - 1] + temp) * 0.5
                res = np.pad(res, ((0, 0), (1, 1), (1, 1)), 'edge')
                res = (res[:, :-1, :] + res[:, 1:, :]) * 0.5
                res = (res[..., :-1] + res[..., 1:]) * 0.5
            elif field_component == 'hy':
                res = (Hy[0:Nx - 1, :, 0:Nz - 1] + temp) * 0.5
                res = np.pad(res, ((1, 1), (0, 0), (1, 1)), 'edge')
                res = (res[:-1, ...] + res[1:, ...]) * 0.5
                res = (res[..., :-1] + res[..., 1:]) * 0.5
            elif field_component == 'hz':
                res = (Hz[0:Nx - 1, 0:Ny - 1, :] + temp) * 0.5
                res = np.pad(res, ((1, 1), (1, 1), (0, 0)), 'edge')
                res = (res[:-1, ...] + res[1:, ...]) * 0.5
                res = (res[:, :-1, :] + res[:, 1:, :]) * 0.5

            F[count, ...] = res[..., z_index]

    return F, t
