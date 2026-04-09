import numpy as np


def transfer_matrix(thickness, epsilon, polarisation, wavelength, kz):
    """Computes the transfer matrix for a given stratified medium.

    For each layer i, the local transfer matrix is:

    m_i = [[ cos(kx_i * d_i),          sin(kx_i * d_i) / (q_i * kx_i) ],
           [ -q_i * kx_i * sin(kx_i * d_i),  cos(kx_i * d_i)          ]]

    where kx_i = sqrt(k0^2 * eps_i - kz^2) is the normal wavevector component,
    and q_i = 1 (TE) or q_i = 1/eps_i (TM).

    The system transfer matrix is the cumulative product:

    The system matrix is the cumulative product M = m_N @ ... @ m_1.

    All dimensions are in µm.

    Args:
        thickness (array_like): Thicknesses of the layers in µm.
        epsilon (array_like): Relative dielectric permittivity of each layer.
        polarisation (str): Polarisation state, either 'TE' or 'TM'.
        wavelength (float): Free-space wavelength in µm.
        kz (float): Transverse wavevector component in µm⁻¹.

    Returns:
        numpy.ndarray: Transfer matrix M of shape (2, 2).
    """
    
    M = np.eye(2, 2, dtype='complex')

    # q encodes the boundary condition difference between polarisations:
    # TE: continuity of Ey and dEy/dx  -> q = 1
    # TM: continuity of Hy and (1/ε)dHy/dx -> q = 1/ε
    if polarisation == 'TE':
        q = np.ones(len(thickness), dtype='complex')
    elif polarisation == 'TM':
        q = 1 / epsilon
    
    k0 = 2 * np.pi / wavelength  # free-space wavenumber [µm⁻¹]
    # normal wavevector per layer; complex to handle evanescent fields
    kx = np.sqrt(k0**2 * epsilon - kz**2).astype('complex')
    cos = np.cos(kx * thickness)
    sin = np.sin(kx * thickness)
    kq = q * kx

    # accumulate layer matrices from left (entrance) to right (exit)
    for kqi, cosi, sini in zip(kq, cos, sin):
        mi = np.array([[cosi, sini/kqi],
                       [-kqi * sini, cosi]])
        M = mi @ M  # left-multiply to propagate through stack
    
    return M


def spectrum(thickness, epsilon, polarisation, wavelength, angle_inc, n_in, n_out):
    """
    Computes the reflection and transmission spectra of a stratified medium.

    For each wavelength, the transfer matrix M is computed and the amplitude
    coefficients are extracted via:

        t = 2 * kq_in / N
        r = (kq_in * M[1,1] - kq_out * M[0,0] - 1j * (M[1,0] + kq_in * kq_out * M[0,1])) / N

    where N = kq_in * M[1,1] + kq_out * M[0,0] + 1j * (M[1,0] - kq_in * kq_out * M[0,1])
    and kq = q * kx, with q = 1 (TE) or q = 1/eps (TM).

    Energy transmittance and reflectance are:

        T = Re(kq_out) / Re(kq_in) * |t|^2
        R = |r|^2

    All dimensions are in µm.

    Args:
        thickness (array_like): Thicknesses of the layers in µm.
        epsilon (array_like): Relative dielectric permittivity of each layer.
        polarisation (str): Polarisation state, either 'TE' or 'TM'.
        wavelength (array_like): Free-space wavelengths in µm.
        angle_inc (float): Angle of incidence in degrees.
        n_in (float): Refractive index of the input medium.
        n_out (float): Refractive index of the output medium.

    Returns:
        t (numpy.ndarray): Complex transmission amplitude, shape (len(wavelength),).
        r (numpy.ndarray): Complex reflection amplitude, shape (len(wavelength),).
        T (numpy.ndarray): Energy transmittance, shape (len(wavelength),).
        R (numpy.ndarray): Energy reflectance, shape (len(wavelength),).
        """
    
    epsilon_in = n_in**2    # permittivity of input medium
    epsilon_out = n_out**2  # permittivity of output medium

    # q encodes polarisation-dependent boundary condition
    if polarisation == 'TE':
        q_in, q_out = 1, 1
    elif polarisation == 'TM':
        q_in, q_out = 1/epsilon_in, 1/epsilon_out
    
    k0 = 2*np.pi/wavelength  # free-space wavenumber [µm^-1]
    kz = k0 * n_in * np.sin(np.deg2rad(angle_inc))  # transverse wavevector, conserved across all layers (Snell's law)
    k_in = np.sqrt(k0**2 * epsilon_in - kz**2)    # normal wavevector in input medium
    k_out = np.sqrt(k0**2 * epsilon_out - kz**2)  # normal wavevector in output medium
    kq_in = q_in * k_in     # combined factor for input medium
    kq_out = q_out * k_out  # combined factor for output medium

    t = np.ones(len(wavelength)).astype('complex')
    r = np.ones(len(wavelength)).astype('complex')

    for i, (kqi_in, kqi_out) in enumerate(zip(kq_in, kq_out)):
        M = transfer_matrix(thickness, epsilon, polarisation, wavelength[i], kz[i])
        # numerator of r, derived from matching boundary conditions at input and output interfaces
        nume = kqi_in * M[1, 1] - kqi_out * M[0, 0] - 1j * (M[1, 0] + kqi_in * kqi_out * M[0, 1])
        # common denominator for both t and r
        N = kqi_in * M[1, 1] + kqi_out * M[0, 0] + 1j * (M[1, 0] - kqi_in * kqi_out * M[0, 1])
        t[i] = 2 * kqi_in / N  # transmission amplitude
        r[i] = nume / N        # reflection amplitude
    # energy coefficients: Re(kq) ratio accounts for beam cross-section change at oblique incidence
    T = (q_out * k_out).real / (q_in * k_in).real * np.abs(t)**2
    R = np.abs(r)**2
    
    return t, r, T, R


def field(thickness, epsilon, polarisation, wavelength, kz, n_in, n_out, Nx, l_in, l_out):
    """
    Computes the field distribution inside a stratified medium.

    The field is normalised such that the transmitted field amplitude equals 1.
    The medium is built from output side to input side internally (reversed order),
    and the result is flipped back before returning.

    The state vector is propagated layer by layer:

        [[f(x)    ]   =   M(x) @ [[1      ]
         [df/dx(x)]]              [1j*kq_out]]

    where the initial vector represents a unit-amplitude outgoing wave
    at the output side.

    All dimensions are in µm.

    Args:
        thickness (array_like): Thicknesses of the layers in µm.
        epsilon (array_like): Relative dielectric permittivity of each layer.
        polarisation (str): Polarisation state, either 'TE' or 'TM'.
        wavelength (float): Free-space wavelength in µm.
        kz (float): Transverse wavevector component in µm^-1.
        n_in (float): Refractive index of the input medium.
        n_out (float): Refractive index of the output medium.
        Nx (int): Number of spatial points for field output.
        l_in (float): Thickness of input medium region to include in output in µm.
        l_out (float): Thickness of output medium region to include in output in µm.

    Returns:
        f (numpy.ndarray): Complex field distribution, shape (Nx,).
        index (numpy.ndarray): Refractive index distribution, shape (Nx,).
        x (numpy.ndarray): Spatial coordinates in µm, shape (Nx,).
    """

    epsilon_in = n_in ** 2
    epsilon_out = n_out ** 2

    # append input and output media to the layer stack
    # reverse the stack: propagation is computed from output side to input side
    # so that the initial condition (unit transmitted field) is set at index 0
    epsilon = np.concatenate(([epsilon_in], epsilon, [epsilon_out]))[::-1].copy()
    thickness = np.concatenate(([l_in], thickness, [l_out]))[::-1].copy()

    if polarisation == 'TE':
        q = np.ones(len(epsilon)).astype('complex')
    elif polarisation == 'TM':
        q = 1 / epsilon

    k0 = 2 * np.pi / wavelength  # free-space wavenumber [µm^-1]
    kx = np.sqrt(k0**2 * epsilon - kz**2)  # normal wavevector per layer
    kq = kx * q
    kq_out = kq[0]

    # initial state vector: unit-amplitude plane wave propagating in output medium
    # [field amplitude, derivative] = [1, i * kq_out]
    f_vector = np.array([[1], [1j * kq_out]])
    M = np.eye(2, dtype='complex')  # cumulative transfer matrix, starts as identity

    x = np.linspace(0, np.sum(thickness), Nx)  # spatial coordinate along reversed stack
    index = np.ones(Nx)  # refractive index at each output point
    f = np.ones(Nx).astype('complex')  # field amplitude at each output point
    layer = 0   # current layer index
    layer_below = 0.0   # accumulated thickness up to current layer boundary
    low = 0.0   # x-coordinate of last matrix update point

    for i in range(len(x)):
        if x[i] - layer_below > thickness[layer]:
            # x[i] has crossed into the next layer: first propagate M to the layer boundary
            layer_upper = layer_below + thickness[layer]
            cos = np.cos(kx[layer] * (layer_upper - low))
            sin = np.sin(kx[layer] * (layer_upper - low))
            m = np.array([[cos, -sin / kq[layer]],
                          [kq[layer] * sin, cos]])
            M = m @ M   # update M to the layer boundary
            low = layer_upper   # reset reference point to layer boundary
            layer_below += thickness[layer]
            layer += 1

        # propagate M from low to current point x[i] within the current layer
        cos = np.cos(kx[layer] * (x[i] - low))
        sin = np.sin(kx[layer] * (x[i] - low))
        m = np.array([[cos, -sin / kq[layer]],
                      [kq[layer] * sin, cos]])
        M = m @ M
        f[i] = (M @ f_vector)[0, 0]  # extract field amplitude (first component of state vector)
        index[i] = np.sqrt(epsilon[layer])  # refractive index at this point
        low = x[i]  # update reference point to current x

    # flip back to physical order: input side on the left, output side on the right
    f = f[::-1]
    index = index[::-1]
    return f, index, x


def bragg(n1, n2, d1, d2, N):
    """Generates the layer parameters of a Bragg mirror.

        Constructs a periodic stack of N bilayers, each consisting of two layers
        with refractive indices n1, n2 and thicknesses d1, d2. The stack starts
        at the incidence side with layer 1 and is terminated by layer 2:

            [n1, n2, n1, n2, ..., n1, n2]  (N periods)

        For a quarter-wave Bragg mirror at target wavelength lam0, the optimal
        thicknesses are:

            d1 = lam0 / (4 * n1)
            d2 = lam0 / (4 * n2)

        Args:
            n1, n2 (float or complex): Refractive indices of the layers of one period.
            d1, d2 (float): Thicknesses of layers of one layer in µm.
            N (int): Number of periods.

        Returns:
            epsilon (numpy.ndarray): Permittivity of each layer (length 2*N).
            thickness (numpy.ndarray): Thickness of each layer in µm (length 2*N).
        """

    # allocate arrays for 2*N layers
    epsilon = np.zeros(2*N, dtype=np.array(n1).dtype)
    thickness = np.zeros(2*N, dtype=np.array(d1).dtype)

    # fill alternating layers using slice notation: even indices -> layer 1, odd -> layer 2
    epsilon[::2] = n1**2
    epsilon[1::2] = n2**2
    thickness[::2] = d1
    thickness[1::2] = d2

    return epsilon, thickness
