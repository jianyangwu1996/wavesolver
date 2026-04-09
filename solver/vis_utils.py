import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def time_animation(x, f, index, steps, periods):
    """
    Animation of a quasi-stationary field.

    Args:
        x (numpy.ndarray): Spatial coordinates in µm, shape (Nx,).
        f (numpy.ndarray): Complex field distribution, shape (Nx,).
        index (numpy.ndarray): Refractive index distribution, shape (Nx,).
        steps (int): Total number of animation frames.
        periods (int): Number of oscillation periods over the animation.

    Returns:
        ani (matplotlib.animation.FuncAnimation): Animation object.
    """

    freq = periods / (steps - 1)

    # precompute normalised field to avoid redundant calculation in every frame
    f_normalised = f.real / f.real.max() * index.max()

    fig = plt.figure()
    line_f, = plt.plot(x, f_normalised, label='EM field')
    plt.plot(x, index, label='refr. index')
    plt.xlim(0, x[-1])
    plt.xlabel('x [µm]')
    plt.ylim(-1.1 * np.abs(f_normalised).max(), 1.1 * f_normalised.max())
    plt.ylabel('normalized field (real part)')
    plt.legend()

    def update(n):
        # apply time-harmonic phase factor for frame n; real part gives physical field
        line_f.set_ydata((f_normalised * np.exp(-2.0j * np.pi * freq * n)).real)
        return line_f

    ani = animation.FuncAnimation(fig, update, frames=steps)

    return ani


class Fdtd1DAnimation(animation.TimedAnimation):
    """Animation of the 1D FDTD fields Ez and Z0*Hy over time.

    Renders two subplots (Ez and Z0*Hy vs. x) that update frame by frame.
    The magnetic field is scaled by the vacuum impedance Z0 so that both
    fields share the same unit (V/m) and are directly comparable in amplitude.

    Args:
        x (numpy.ndarray): Spatial coordinates in meters, shape (Nx,).
        t (numpy.ndarray): Time coordinates in seconds, shape (Nt,).
        Ez (numpy.ndarray): Z-component of the electric field E(x, t),
            shape (Nt, Nx). Each row corresponds to one time step.
        Hy (numpy.ndarray): Y-component of the magnetic field H(x, t),
            shape (Nt, Nx). Each row corresponds to one time step.
        x_interface (float, optional): Spatial position of a dielectric
            interface to mark with a dashed vertical line. Default: None.
        step (float, optional): Physical time interval between animation
            frames in seconds. Default: 2e-15 / 25.
        fps (int, optional): Frames per second of the animation. Default: 25.
    """

    def __init__(self, x, t, Ez, Hy, x_interface=None, step=2e-15 / 25, fps=25):
        # constants
        c = 2.99792458e8  # speed of light [m/s]
        mu0 = 4 * np.pi * 1e-7  # vacuum permeability [Vs/(Am)]
        eps0 = 1 / (mu0 * c ** 2)  # vacuum permittivity [As/(Vm)]
        Z0 = np.sqrt(mu0 / eps0)  # vacuum impedance [Ohm]
        self.Ez = Ez
        self.Z0Hy = Z0 * Hy  # scale Hy to same unit as Ez for direct comparison
        self.x = x
        self.ct = c * t

        # index step between consecutive frames
        self.frame_step = int(round(step / (t[1] - t[0])))

        # set up initial plot
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        vmax = max(np.max(np.abs(Ez)), np.max(np.abs(Hy)) * Z0) * 1e6
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.4})
        self.line_E, = ax[0].plot(x * 1e6, self.E_at_step(0),
                                  color=colors[0], label='$\\Re\\{E_z\\}$')
        self.line_H, = ax[1].plot(x * 1e6, self.H_at_step(0),
                                  color=colors[1], label='$Z_0\\Re\\{H_y\\}$')
        # Mark dielectric interface position if provided
        if x_interface is not None:
            for a in ax:
                a.axvline(x_interface * 1e6, ls='--', color='k')
        for a in ax:
            a.set_xlim(x[[0, -1]] * 1e6)
            a.set_ylim(np.array([-1.1, 1.1]) * vmax)
        ax[0].set_ylabel('$\\Re\\{E_z\\}$ [µV/m]')
        ax[1].set_ylabel('$Z_0\\Re\\{H_y\\}$ [µV/m]')
        self.text_E = ax[0].set_title('')
        self.text_H = ax[1].set_title('')
        ax[1].set_xlabel('$x$ [µm]')
        super().__init__(fig, interval=1000 / fps, blit=False)

    def E_at_step(self, n):
        """Returns the real part of Ez at time step n, scaled to µV/m."""
        return self.Ez[n, :].real * 1e6

    def H_at_step(self, n):
        """Returns the real part of Z0*Hy at time step n, scaled to µV/m."""
        return self.Z0Hy[n, :].real * 1e6

    def new_frame_seq(self):
        return iter(range(0, self.ct.size, self.frame_step))

    def _init_draw(self):
        """Clears all artists before the animation starts."""
        self.line_E.set_ydata(self.x * np.nan)
        self.line_H.set_ydata(self.x * np.nan)
        self.text_E.set_text('')
        self.text_H.set_text('')

    def _draw_frame(self, framedata):
        i = framedata
        self.line_E.set_ydata(self.E_at_step(i))
        self.line_H.set_ydata(self.H_at_step(i))
        self.text_E.set_text(
            'Electric field, $ct = {0:1.2f}$µm'.format(self.ct[i] * 1e6))
        self.text_H.set_text(
            'Magnetic field, $ct = {0:1.2f}$µm'.format(self.ct[i] * 1e6))
        self._drawn_artists = [self.line_E, self.line_H,
                               self.text_E, self.text_H]


class Fdtd3DAnimation(animation.TimedAnimation):
    """
    Animation of a selected field component from the 3D FDTD simulation.
    Renders a single animated pseudocolor plot of a z-slice of the chosen
    field component over time.

    Args:
        x (numpy.ndarray): X-axis coordinates in meters, shape (Nx,).
        y (numpy.ndarray): Y-axis coordinates in meters, shape (Ny,).
        t (numpy.ndarray): Time coordinates in seconds, shape (Nt,).
        field (numpy.ndarray): Z-slices of the field component to animate, shape (Nt, Nx, Ny).
        title_str (str): Base string for the plot title. The current ct value
            is appended automatically at each frame.
        cb_label (str): Label for the colorbar.
        rel_color_range (float): Colormap range as a fraction of the maximum
            field magnitude. Use values < 1 to saturate and enhance contrast.
        fps (int, optional): Frames per second of the animation. Default: 25.
    """

    def __init__(self, x, y, t, field, title_str, cb_label, rel_color_range, fps=25):
        # --- Physical constants ---
        c = 2.99792458e8  # speed of light [m/s]
        self.ct = c * t  # optical path length [m]

        self.fig = plt.figure()
        self.F = field
        color_range = rel_color_range * np.max(np.abs(field))
        phw = 0.5 * (x[1] - x[0])  # pixel half-width
        extent = ((x[0] - phw) * 1e6, (x[-1] + phw) * 1e6,
                  (y[-1] + phw) * 1e6, (y[0] - phw) * 1e6)

        # Display the first frame as initial plot state
        self.mapable = plt.imshow(self.F[0, :, :].real.T,
                                  vmin=-color_range, vmax=color_range,
                                  extent=extent)
        cb = plt.colorbar(self.mapable)
        plt.gca().invert_yaxis()
        self.title_str = title_str
        self.text = plt.title('')
        plt.xlabel('x position [µm]')
        plt.ylabel('y position [µm]')
        cb.set_label(cb_label)
        super().__init__(self.fig, interval=1000 / fps, blit=False)

    def new_frame_seq(self):
        return iter(range(self.ct.size))

    def _init_draw(self):
        self.mapable.set_array(np.nan * self.F[0, :, :].real.T)
        self.text.set_text('')

    def _draw_frame(self, frame_data):
        i = frame_data
        self.mapable.set_array(self.F[i, :, :].real.T)
        self.text.set_text(self.title_str
                           + ', $ct$ = {0:1.2f}µm'.format(self.ct[i] * 1e6))
        self._drawn_artists = [self.mapable, self.text]
