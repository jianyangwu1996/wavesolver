# wavesolver
 
A collection of numerical solvers for computational photonics and electromagnetic wave simulation, implemented in Python with NumPy and SciPy.
 
光子学与电磁波数值仿真工具集，基于 Python / NumPy / SciPy 实现。
 
---
 
## Modules
 
| Module | Method | Description |
|---|---|---|
| `tmm_solver` | Transfer Matrix Method | Reflection / transmission spectra and field distributions in multilayer stacks |
| `waveguide_mode_solver` | Eigenmode Analysis | Guided mode profiles and effective permittivities for 1D TE and 2D quasi-TE waveguides |
| `bpm_solver_cn` | Beam Propagation (Crank–Nicolson) | Paraxial field propagation in inhomogeneous waveguide structures |
| `fdtd_solver` | FDTD (1D / 3D) | Time-domain EM field propagation with Yee grid and PEC boundary conditions |
| `vis_utils` | — | Shared visualisation utilities (field snapshots, animations) |
 
---
 
## Methods Overview
 
### Transfer Matrix Method — `tmm_solver.py`
 
Assembles the system transfer matrix as the ordered product of per-layer matrices:
 
$$M = \prod_{i=N}^{1} m_i, \quad m_i = \begin{pmatrix} \cos(k_{x,i} d_i) & \frac{\sin(k_{x,i} d_i)}{q_i k_{x,i}} \\ -q_i k_{x,i} \sin(k_{x,i} d_i) & \cos(k_{x,i} d_i) \end{pmatrix}$$
 
Computes complex reflection / transmission coefficients and intracavity field distributions for TE / TM polarisation. Includes a Bragg mirror builder for quarter-wave stacks.
 
### Waveguide Mode Solver — `waveguide_mode_solver.py`
 
Discretises the Helmholtz eigenvalue problem on a uniform grid:
 
$$\left(\frac{d^2}{dx^2} + k_0^2 \varepsilon(x)\right) E_y = \varepsilon_\text{eff}\, k_0^2\, E_y$$
 
Solves for guided mode profiles and effective permittivities in 1D (dense eigensolver) and 2D (sparse ARPACK eigensolver) geometries. Spurious inter-column couplings in the 2D pentadiagonal operator are explicitly removed.
 
### Beam Propagation — `bpm_solver_cn.py`
 
Propagates an initial field using the Crank–Nicolson discretisation of the paraxial wave equation:
 
$$\left(I - \frac{\Delta z}{2} L\right) v^{(i)} = \left(I + \frac{\Delta z}{2} L\right) v^{(i-1)}$$
 
The finite-difference operator $L$ encodes both diffraction (transverse Laplacian) and phase accumulation ($k^2 - k_\text{ref}^2$). The implicit scheme is unconditionally stable and solved via sparse LU factorisation at each step.
 
### FDTD — `fdtd_solver.py`
 
Implements the leapfrog update scheme on a staggered Yee grid:
 
$$E^{n+1}_{z,j} = E^{n}_{z,j} + \frac{\Delta t}{\varepsilon_0 \varepsilon_j \Delta x} \left( H^{n}_{y,j} - H^{n}_{y,j-1} \right)$$
 
Supports 1D single-interface problems (Fresnel validation) and 3D TM-polarised propagation with Gaussian line-source excitation.
 
---
 
## Project Structure
 
```
wavesolver/
├── solver/                       # Numerical solver modules
│   ├── tmm_solver.py             # Transfer matrix method
│   ├── waveguide_mode_solver.py  # 1D/2D eigenmode solver
│   ├── bpm_solver_cn.py          # Crank–Nicolson beam propagation
│   ├── fdtd_solver.py            # 1D/3D FDTD engine
│   └── vis_utils.py              # Visualisation utilities
├── demo/                         # Jupyter notebook demonstrations
│   ├── tmm_demo.ipynb            # TMM demonstration
│   ├── waveguide_mode_demo.ipynb # Mode solver & BPM demonstration
│   └── fdtdd_demo.ipynb          # FDTD demonstration
├── animation/                    # FDTD field evolution animations
├── .gitignore
└── README.md
```
 
---
 
## Author
 
Jianyang Wu · MSc Medical Photonics
 
