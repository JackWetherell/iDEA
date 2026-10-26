# <img src="docs/logo.png" alt="" width="70"/> iDEA (interacting Dynamic Electrons Approach)  

Exploring exact solutions and practical approximations in many-electron quantum mechanics.

![pip](https://img.shields.io/pypi/v/iDEA-latest)
![tag](https://img.shields.io/github/v/tag/iDEA-org/iDEA)
[![Documentation Status](https://readthedocs.org/projects/idea-interacting-dynamic-electrons-approach/badge/?version=latest)](https://idea-interacting-dynamic-electrons-approach.readthedocs.io/en/latest/?badge=latest)
![Issues](https://img.shields.io/github/issues/iDEA-org/iDEA)
![Pull Requests](https://img.shields.io/github/issues-pr/iDEA-org/iDEA)

<p>iDEA (the interacting dynamic electrons approach) is a comprehensive software library that enables users to explore and understand the intricacies of many-body quantum mechanics. Developed at the <a href="https://www.york.ac.uk/">University of York</a> and the <a href="https://www.polytechnique.edu/en">&Eacute;cole polytechnique</a>, iDEA is written in Python and offers both exact and approximate approaches to quantum mechanics. With its focus on reproducibility, interactivity, and simplicity, iDEA has been used in a variety of research projects to gain insights into fundamental theories, such as density functional theory and many-body perturbation theory, as well as in educational contexts, such as <a href="https://www.coursera.org/learn/density-functional-theory">Coursera online courses</a>.</p>
 
<p>One of the main goals of iDEA is to help users understand when popular approximations used in practical quantum theory calculations may be unreliable and why. By using iDEA to study a variety of systems, researchers can identify the circumstances in which these approximations are least secure and develop more advanced methods for use in materials science.</p>

<!--**iDEA (interacting Dynamic Electrons Approach) is a high-performance, user friendly, free software framework in python for state-of-the-art research, experiments, testing and education in many-body quantum physics with a focus on reproducibility, interactivity and simplicity.** -->

[Homepage](https://idea-org.github.io/)

[View on GitHub](https://github.com/iDEA-org/iDEA)

![demo](demo.gif)

## Installation

### User

To install the [latest version of the iDEA code](https://pypi.org/project/iDEA-latest/):

`pip install iDEA-latest`

### Developer

If you would like to develop iDEA, first fork this git repository, and then clone from there.

Add the upstream repository: `git remote add upstream https://github.com/iDEA-org/iDEA.git`

Optionally, create a virtual environment first:

```
python -m venv .venv
source .venv/bin/activate
```

Install locally with dev dependencies: `pip install -e ".[dev]"`

### Code Quality

iDEA uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, installed automatically with the dev dependencies. Before submitting a pull request, please run:

```
ruff check iDEA/ tests/ benchmarking/
ruff format iDEA/ tests/ benchmarking/
```

### Testing

To run unit tests: `pytest -v`

The many-body perturbation theory tests in `tests/test_mbpt.py` check the GW implementation against results computed independently of its imaginary time and frequency grids (sums over states built from the random phase approximation, and brute-force frequency integration), as well as particle-number conservation and starting-point independence of self-consistent GW. They take about a minute.

The exact Green's function tests in `tests/test_green_function.py` check the exact $G$ and $\Sigma_{xc}$ against their sum rules (normalisation, density matrix, first moment, Galitskii–Migdal energy), against direct inversion of Dyson's equation, and against the non-interacting limit, for one, two and three electrons. They take about a minute.

The GPU tests in `tests/test_gpu.py` check that the GPU and CPU produce the same results for interacting ground-state and time-dependent calculations. They are skipped automatically unless CuPy is installed (see [GPU Acceleration](#gpu-acceleration)). On an Intel i9 / RTX 4090 these take roughly 26 minutes.

### Benchmarking

The `benchmarking` directory contains scripts that measure how the exact interacting solver scales on the CPU compared to the GPU:

```
python benchmarking/gpu.py      # ground state, grid sizes 50-450
python benchmarking/gpu_td.py   # time-dependent propagation, grid sizes 50-300
```

Each script detects your CPU and GPU, prints them along with an estimated runtime calibrated on your machine, and then writes a plot of wall-clock time and peak memory against grid size to `benchmarking/gpu_scaling.png` and `benchmarking/gpu_td_scaling.png` respectively.

On an Intel i9 / RTX 4090 these take roughly 11 minutes and 2 minutes. Most of that is the CPU points, whose cost grows steeply with grid size — the GPU sweep is a small fraction of the total. Without CuPy installed both scripts still run, recording the CPU points and skipping the GPU ones.

## Documentation

For full details of usage please see our [tutorial](https://github.com/iDEA-org/iDEA/blob/master/tutorial/tutorial.ipynb). The full API documentation is available at [readthedocs](https://idea-interacting-dynamic-electrons-approach.readthedocs.io/en/latest/).

## Features

Some of iDEA's features:
- Exact solution of the many-electron problem by solving the static and time-dependent Schrödinger equation, including exact exchange and correlation.
- Exact solutions which approach the degree of exchange and correlation in realistic systems.
- Free choice of external potential that may be time-dependent, on an arbitrarily dense spatial grid, for any number of electron with any spin configuration.
- Implementation of various approximate methods (established and novel) for comparison, including:
    - Non-interacting electrons
    - Hartree theory
    - Restricted and unrestricted Hartree-Fock
    - The Local Density Approximation (LDA)
    - Hybrid functionals
    - Many-body perturbation theory in the GW approximation: one-shot (G0W0), partially self-consistent (GW0) and fully self-consistent (GW) (see [Many-Body Perturbation Theory](#many-body-perturbation-theory))
- The exact Green's function and exact exchange-correlation self-energy of the interacting system, in real and imaginary time and frequency, with videos of them (see [Exact Green's Function and Self-Energy](#exact-greens-function-and-self-energy)).
- Implementation of all common observables.
- Reverse-engineering to solve potential inversion, from exact Kohn-Sham DFT and beyond.
- Fully parallelised using OpenBLAS.
- GPU acceleration of the exact interacting solver (see [GPU Acceleration](#gpu-acceleration)).

## Example

In order to solve the Schrödinger equation for the two electron atom for the ground-state charge density and total energy:

```
import iDEA
system = iDEA.system.systems.atom
ground_state = iDEA.methods.interacting.solve(system, k=0)
n = iDEA.observables.density(system, state=ground_state)
E = ground_state.energy

import matplotlib.pyplot as plt
print(E)
plt.plot(system.x, n, 'k-')
plt.show()
```

## GPU Acceleration

Solving the many-electron Schrödinger equation exactly is by far the most demanding thing iDEA does: the size of the problem grows exponentially with the number of electrons, and even two electrons on a fine grid means working with matrices with tens of billions of elements. To make this fast, iDEA can offload this heavy linear algebra to an NVIDIA GPU using [CuPy](https://cupy.dev/).

GPU acceleration is available for the **exact interacting solver** (`iDEA.methods.interacting`), for both:
- **Ground-state and excited-state calculations** — building the many-body Hamiltonian and solving the eigenproblem on the GPU.
- **Time-dependent calculations** — propagating the many-body wavefunction through time on the GPU.

Using it is as simple as adding `GPU=True`:

```python
import iDEA

system = iDEA.system.systems.atom

# Solve for the ground state on the GPU.
ground_state = iDEA.methods.interacting.solve(system, k=0, GPU=True)

# Propagate in time on the GPU.
evolution = iDEA.methods.interacting.propagate(system, ground_state, v_ptrb, t, GPU=True)
```

Everything else about your workflow stays the same — the results are returned as ordinary NumPy arrays, agree with the CPU implementation to machine precision, and all observables can be computed as usual. If you don't have a GPU, simply leave `GPU=False` (the default) and the calculation runs on the CPU.

A few things to be aware of:
- You will need an NVIDIA GPU with CUDA, and the [CuPy](https://docs.cupy.dev/en/stable/install.html) package installed. It is an optional dependency, so it is not installed by default — use `pip install -e ".[gpu]"` (or `pip install cupy-cudaxxx` directly, matching your CUDA version).
- The **approximate methods** (non-interacting, Hartree, Hartree-Fock, LDA, hybrids) run on the CPU only. This is by design: they work with small single-particle matrices for which a GPU offers no benefit — they are already fast.
- The speedup grows with system size: the finer the grid and the more electrons, the more the GPU helps.

You can measure the performance benefit on your own hardware using the scripts in the `benchmarking` directory — see [Benchmarking](#benchmarking).

## Many-Body Perturbation Theory

iDEA can solve Hedin's equations in the GW approximation to give the ground-state one-body Green's function $G$ of the interacting system, from which the charge density, self-energy and screened interaction follow:

```python
import iDEA

system = iDEA.system.systems.atom

# Fully self-consistent GW, starting from Hartree-Fock.
state = iDEA.methods.mbpt.solve(system)

# One-shot G0W0 (or partially self-consistent GW0) from a chosen starting point.
state = iDEA.methods.mbpt.solve(system, mode="G0W0", initial="lda")

n = iDEA.observables.density(system, state=state)  # n(x) = G(x,x;tau=0^-)
E = state.energy                                    # Galitskii-Migdal total energy
```

The returned `iDEA.state.QuasiParticleState` contains everything computed along the way, on the real-space grid and on imaginary time (`state.tau`) and imaginary frequency (`state.omega`) grids. For each spin channel (`state.up`, `state.down`) it holds the Green's function `G` / `G_iw`, the starting-point Green's function `G0` / `G0_iw`, and the exchange `sigma_x`, correlation `sigma_c` / `sigma_c_iw`, exchange-correlation `sigma_xc_iw` and total `sigma_iw` self-energies. It also holds the polarisability `state.P` / `state.P_iw` and the screened interaction `state.W_iw` (with its correlation part `state.W_c` / `state.W_c_iw`). `state.grid` provides the Fourier transforms between imaginary time and frequency, and interpolation to any imaginary time or frequency.

The implementation follows the GW space-time method of Rojas, Godby and Needs: polarisabilities and self-energies are formed in real space and imaginary time, while the screened interaction and Dyson's equation are solved on the imaginary frequency axis. Instead of uniform grids, which need a time step smaller than the inverse of the largest single-particle energy on the real-space grid, it uses imaginary time and frequency grids optimised for decaying exponentials. Each grid holds roughly 50–100 points, and the Fourier transforms between them are accurate to about $10^{-10}$. The chemical potential is kept in the centre of the quasiparticle gap with the Hedin shift, as in the original iDEA code; this does not change self-consistent (GW and GW0) results. Each building block of Hedin's equations is also available on its own: `non_interacting_green_function`, `polarisability`, `screened_interaction`, `exchange_self_energy`, `correlation_self_energy` and `dyson_equation`.

A worked example, which compares GW with exact and Hartree-Fock results and steps through Hedin's equations one at a time, is given in [tutorial/example_mbpt.ipynb](tutorial/example_mbpt.ipynb).

## Exact Green's Function and Self-Energy

iDEA can also construct the exact one-body Green's function $G(x,x')$ and the exact exchange-correlation self-energy $\Sigma_{xc}(x,x')$ of the interacting system, from the exact eigenstates of the $N-1$, $N$ and $N+1$ electron systems:

```python
import numpy as np
import iDEA

# The one-dimensional hydrogen atom.
x = np.linspace(-12, 12, 61)
system = iDEA.system.System(x, -1.0 / (abs(x) + 1.0), iDEA.interactions.softened_interaction(x), "u")
ground_state = iDEA.methods.interacting.solve(system, k=0)

# Exact G and Sigma_xc in pole form, from the eigenstates of the N-1, N and N+1 electron systems.
green = iDEA.methods.interacting.green_function(system, ground_state)

omega = np.linspace(-1.0, 2.0, 150)
G = iDEA.observables.green_function(system, green, omega=omega, eta=0.05)  # G(x,x';omega), shape (150, 61, 61)
sigma_xc = iDEA.observables.self_energy(system, green, omega=omega)  # Sigma_xc(x,x';omega)
sigma_c = iDEA.observables.self_energy(system, green, t=np.linspace(0, 20, 200), part="c")  # Sigma_c(x,x';t)
sigma_iw = iDEA.observables.self_energy(system, green, nu=np.linspace(0, 3, 50))  # Sigma_xc(x,x';mu+i*nu)
A = iDEA.observables.spectral_function(system, green, omega, diagonal=True)  # A(x,x;omega)
E = iDEA.observables.galitskii_migdal_energy(system, green)  # equals ground_state.energy

# Videos of G and Sigma_xc as functions of (x,x'), one frame per frequency.
iDEA.plotting.animate_green_function(system, green, omega=omega, file_name="G_omega.gif")
iDEA.plotting.animate_self_energy(system, green, omega=omega, spin="down", file_name="Sigma_xc_omega.gif")
```

The returned `iDEA.state.GreenFunction` holds, for each spin channel, the Lehmann representation of $G$ (pole energies $E_0-E^{N-1}_k$ and $E^{N+1}_k-E_0$ with their Dyson orbitals) and the exact self-energy in the same form: the static exchange part $\Sigma_x$, which is the Fock operator of the exact density matrix, and the poles and residues of $\Sigma_c$. These are computed by block Lanczos (Löwdin partitioning) on the poles of $G$. From them `iDEA.observables` evaluates $G$ and $\Sigma$ in real time (time-ordered, retarded or advanced), at real or complex frequency, and on the imaginary time and frequency axes used by `iDEA.methods.mbpt`, so exact and GW results can be compared directly. `green_function`, `self_energy` and `galitskii_migdal_energy` also accept a `QuasiParticleState`, and `self_energy(..., method="dyson")` gives $\Sigma_{xc}$ by direct inversion of Dyson's equation as an independent check.

The $N\pm1$ electron Hamiltonians are built in a basis of ordered Slater determinants. Sectors up to `dense_max` determinants are diagonalised completely. Larger sectors use a block Krylov method seeded with the $N\pm1$ electron states $\hat\psi^{(\dagger)}(x)|\Psi_0\rangle$, which keeps the sum rules (normalisation, density matrix, first moment and Galitskii–Migdal energy) exact. Both run on the GPU with `GPU=True`. The residuals of these sum rules are reported in `green.info`.

A worked example for the one-dimensional hydrogen atom is given in [tutorial/example_green_function.ipynb](tutorial/example_green_function.ipynb). It checks that the exact self-energy is free of self-interaction, $(v_H + \Sigma_{xc}(\omega))\,\phi = 0$ for the occupied orbital $\phi$, and compares it with GW. [tutorial/green_function_hydrogen.py](tutorial/green_function_hydrogen.py) runs the same calculation from the command line on larger grids and writes the videos.

## Tutorial

We provide a [tutorial](https://github.com/iDEA-org/iDEA/blob/master/tutorial/tutorial.ipynb) where you can learn how to use the iDEA code in your research and teaching projects.

## Papers You Can Reproduce With iDEA

1. "Advantageous nearsightedness of many-body perturbation theory contrasted with Kohn-Sham density functional theory", J. Wetherell, M. J. P. Hodgson, L. Talirz, and R. W. Godby, Physical Review B 99 045129 (2019).
[paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.045129), [reprint](https://jackwetherell.github.io/files/nearsightedness.pdf), [preprint](https://arxiv.org/pdf/1812.02661.pdf), [code](https://github.com/JackWetherell/advantageous-nearsightedness).

More coming soon...

The development and applications of the iDEA code from 2010 to 2021 is documented [here](https://www-users.york.ac.uk/~rwg3/idea.html).

## Teaching

iDEA can be used to create teaching content, visualisations and expositions. For example, see the following [YouTube video created using iDEA](https://www.youtube.com/watch?v=JaSVguMFA-M&ab_channel=JackWetherell).

iDEA was used to create teaching content for the [Density Functional Theory MOOC on Coursera](https://www.coursera.org/learn/density-functional-theory).

## Developers

Dr. Jack Wetherell, Dr. Matt Hodgson and Dr. Leopold Talirz.

<div style="display:flex;">
    <img src="docs/photos.png" style="width:40%">
</div>

<!--img src="docs/dev.png" alt="" width="500"/-->

## Contributors

We thank all of the developers, PhD students, master's students, summer project interns and researchers for thier key contributions to iDEA:

Sean Adamson, Jacob Chapman, Thomas Durrant, Razak Elmaslmane, Mike Entwistle, Rex Godby, Matt Hodgson, Piers Lillystone, Aaron Long, Robbie Oliver, James Ramsden, Ewan Richardson, Paul Sharp, Matthew Smith, Leopold Talirz and Jack Wetherell. 

## Getting Involved

To get involved:
- Raising issues and pull requests here is greatly appreciated!
- We can add any papers that can be fully reproduced by iDEA to our dedicated page by sending your open access paper to jack.wetherell@gmail.com.
- We provide a [template](https://github.com/iDEA-org/iDEA-project-template) to get you started!

## Dependencies

iDEA supports `python 3.8+` along with the following dependences:
```
numpy >= "1.22.3"
scipy >= "1.8.0"
matplotlib >= "3.5.1"
jupyterlab >= "3.3.2"
tqdm >= "4.64.0"
pytest >= "8.3.0"
```

<img src="docs/logos.png" alt="" width="200"/>
