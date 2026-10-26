"""Contains all many-body perturbation theory (MBPT) functionality and solvers.

The ground state of the interacting system is described by the one-body Green's function G, which is found by
solving Hedin's equations [1] within the GW approximation, in which the vertex function is neglected:

| P(x,x';t) = sum_s G_s(x,x';t) G_s(x',x;-t),             (irreducible polarisability)
| W = v + v P W,                                          (screened interaction)
| Sigma_xc,s(x,x';t) = -G_s(x,x';t) W(x,x';t),            (exchange-correlation self-energy)
| G_s = G0_s + G0_s (Sigma_s - V_s) G_s,                  (Dyson equation)

where t is imaginary time, s is spin, Sigma_s = v_h + Sigma_xc,s is the total self-energy and V_s is the mean-field
potential of the starting point (which is removed to avoid double counting).

The equations are solved with the GW space-time method of Rojas, Godby and Needs [2, 3]: the products (P and Sigma)
are formed in real space and imaginary time, where they are local, while W and the Dyson equation are solved on the
imaginary frequency axis, where they are local. Fourier transforms connect the two.

Three levels of self-consistency are available:

| G0W0: One-shot. The self-energy is built from the Green's function of the starting point, and Dyson's equation is solved once.
| GW0: G is iterated to self-consistency with W = W0 kept fixed (partially self-consistent).
| GW: G and W are both iterated to self-consistency (fully self-consistent), which is conserving and independent of the starting point.

Conventions: The calculation is performed at zero temperature using real-valued imaginary-time functions,
G(x,x';t) = -<T psi(x,t) psi^dagger(x',0)>, with G(iw) = int dt e^(iwt) G(t). All quantities are measured relative to
a chemical potential mu lying within the quasiparticle gap of each spin channel, so every imaginary-time quantity is a
superposition of decaying exponentials exp(-e|t|). The density matrix is G(x,x';t=0^-). In the convention of [2, 3],
G(x,x';it) = iG(x,x';t). All two-point quantities are stored as kernels A(x,x'), so the operator they represent acts
as int A(x,x') f(x') dx', and on the grid as np.dot(A, f) * dx.

Imaginary time and frequency grids: Rather than using uniform grids and fast Fourier transforms, which would require
an imaginary time step smaller than 1/(the largest single-particle energy on the grid), the imaginary time and frequency
grids are chosen to be optimal for representing decaying exponentials exp(-e|t|), with e between the smallest and
largest single-particle excitation energy, using a pivoted QR decomposition (as in the discrete Lehmann representation
[4]). The Fourier transforms between these grids are least-squares fits that are exact for this family of functions
(as in minimax and least-squares space-time methods [5, 6]), giving transforms that are accurate to the grid tolerance
with only O(50) imaginary time and O(100) imaginary frequency points.

The chemical potential: At zero temperature the chemical potential must lie within the quasiparticle gap. The Hedin
shift [7] (as in the original iDEA MBPT code) moves the chemical potential to the centre of the quasiparticle gap at
each iteration, which is equivalent to shifting the self-energy by a constant. For self-consistent calculations (GW and
GW0) this does not change the converged Green's function. For G0W0 it aligns the chemical potentials of G0 and G,
which is necessary when the starting point has a very different quasiparticle gap (e.g. a non-interacting starting
point).

References:
1. L. Hedin, New method for calculating the one-particle Green's function with application to the electron-gas problem. Phys. Rev. 139, A796 (1965). DOI: [10.1103/PhysRev.139.A796](https://doi.org/10.1103/PhysRev.139.A796)
2. H. N. Rojas, R. W. Godby, and R. J. Needs, Space-time method for ab initio calculations of self-energies and dielectric response functions of solids. Phys. Rev. Lett. 74, 1827 (1995). DOI: [10.1103/PhysRevLett.74.1827](https://doi.org/10.1103/PhysRevLett.74.1827)
3. M. M. Rieger, L. Steinbeck, I. D. White, H. N. Rojas, and R. W. Godby, The GW space-time method for the self-energy of large systems. Comput. Phys. Commun. 117, 211 (1999). DOI: [10.1016/S0010-4655(98)00174-X](https://doi.org/10.1016/S0010-4655(98)00174-X)
4. J. Kaye, K. Chen, and O. Parcollet, Discrete Lehmann representation of imaginary time Green's functions. Phys. Rev. B 105, 235115 (2022). DOI: [10.1103/PhysRevB.105.235115](https://doi.org/10.1103/PhysRevB.105.235115)
5. M. Kaltak, J. Klimes, and G. Kresse, Low scaling algorithms for the random phase approximation: Imaginary time and Laplace transformations. J. Chem. Theory Comput. 10, 2498 (2014). DOI: [10.1021/ct5001268](https://doi.org/10.1021/ct5001268)
6. P. Liu, M. Kaltak, J. Klimes, and G. Kresse, Cubic scaling GW: Towards fast quasiparticle calculations. Phys. Rev. B 94, 165109 (2016). DOI: [10.1103/PhysRevB.94.165109](https://doi.org/10.1103/PhysRevB.94.165109)
7. J. Wetherell, Electron correlation and screening in model nanostructures. PhD thesis, University of York (2018).
8. A. Stan, N. E. Dahlen, and R. van Leeuwen, Levels of self-consistency in the GW approximation. J. Chem. Phys. 130, 114105 (2009). DOI: [10.1063/1.3089567](https://doi.org/10.1063/1.3089567)
"""

import warnings
from typing import Union

import numpy as np
import scipy.linalg as spla

import iDEA.methods.non_interacting
import iDEA.observables
import iDEA.state
import iDEA.system

name = "mbpt"


kinetic_energy_operator = iDEA.methods.non_interacting.kinetic_energy_operator
external_potential_operator = iDEA.methods.non_interacting.external_potential_operator


def _composite_gauss_legendre(start: float, end: float, first_width: float, points: int = 24) -> np.ndarray:
    r"""
    Construct a composite Gauss-Legendre grid, with panels that double in width, which resolves functions that vary
    on all scales between first_width and end - start.

    | Args:
    |     start: float, Start of the interval.
    |     end: float, End of the interval.
    |     first_width: float, Width of the first panel.
    |     points: int, Number of Gauss-Legendre points per panel. (default = 24)

    | Returns:
    |     grid: np.ndarray, Grid points.
    """
    nodes = np.polynomial.legendre.leggauss(points)[0]
    edges = [start]
    width = first_width
    while edges[-1] < end:
        edges.append(min(edges[-1] + width, end))
        width *= 2.0
    panels = [0.5 * (upper - lower) * nodes + 0.5 * (upper + lower) for lower, upper in zip(edges[:-1], edges[1:])]
    return np.concatenate(panels)


def _select_points(A: np.ndarray, tol: float) -> np.ndarray:
    r"""
    Select the rows of A that span its row space to within a relative tolerance, using a pivoted QR decomposition.

    | Args:
    |     A: np.ndarray, Matrix, indexed as A[candidate_point,decay_rate].
    |     tol: float, Relative tolerance.

    | Returns:
    |     indices: np.ndarray, Sorted indices of the selected rows.
    """
    R, pivots = spla.qr(A.T, mode="r", pivoting=True)
    diagonal = abs(np.diag(R))
    rank = int(np.sum(diagonal > tol * diagonal[0]))
    return np.sort(pivots[:rank])


def _fit(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""
    Least-squares fit of the linear map X that best satisfies X A = B.

    | Args:
    |     A: np.ndarray, Input values, indexed as A[input_point,decay_rate].
    |     B: np.ndarray, Target values, indexed as B[output_point,decay_rate].

    | Returns:
    |     X: np.ndarray, Linear map, indexed as X[output_point,input_point].
    """
    return np.linalg.lstsq(A.T, B.T, rcond=1e-14)[0].T


def _decay_kernels(rates: np.ndarray, tau: np.ndarray, omega: np.ndarray) -> tuple:
    r"""
    Compute the decaying exponentials exp(-e t) for t > 0, and their one-sided cosine and sine transforms.

    | Args:
    |     rates: np.ndarray, Decay rates e.
    |     tau: np.ndarray, Imaginary times t > 0.
    |     omega: np.ndarray, Imaginary frequencies w.

    | Returns:
    |     E, C, S: np.ndarray, exp(-e t), int_0^inf cos(w t) exp(-e t) dt = e/(e^2 + w^2), and int_0^inf sin(w t) exp(-e t) dt = w/(e^2 + w^2).
    """
    E = np.exp(-np.outer(tau, rates))
    denominator = rates[None, :] ** 2 + omega[:, None] ** 2
    C = rates[None, :] / denominator
    S = omega[:, None] / denominator
    return E, C, S


class SpaceTimeGrid:
    r"""
    Imaginary time and imaginary frequency grids, and the Fourier transforms between them.

    At zero temperature, every imaginary-time quantity in the GW approximation is a superposition of decaying
    exponentials exp(-e|t|), where the decay rates e are excitation energies measured relative to the chemical
    potential, and lie between energy_min and energy_max. The grid points are chosen with a pivoted QR decomposition
    to optimally represent these functions (as in the discrete Lehmann representation), and the Fourier transforms
    are least-squares fits that are exact for this family of functions, to within the given tolerance.

    Imaginary-time quantities are stored on the two-sided grid tau (negative times followed by positive times), as
    f[tau,...]. Imaginary-frequency quantities are stored on the non-negative grid omega, as F[omega,...], with
    F(-iw) = F(iw)^* for real f(t).

    The Fourier transforms are defined as F(iw) = int dt e^(iwt) f(t) and f(t) = (1/2pi) int dw e^(-iwt) F(iw).
    """

    def __init__(self, energy_min: float, energy_max: float, tol: float = 1e-10):
        r"""
        Imaginary time and imaginary frequency grids, and the Fourier transforms between them.

        | Args:
        |     energy_min: float, Smallest decay rate (excitation energy relative to the chemical potential) to represent.
        |     energy_max: float, Largest decay rate (excitation energy relative to the chemical potential) to represent.
        |     tol: float, Relative tolerance of the representation. (default = 1e-10)
        """
        if not 0.0 < energy_min < energy_max:
            raise ValueError(f"Expected 0 < energy_min < energy_max, got {energy_min} and {energy_max} instead.")
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.tol = tol

        # Select the imaginary time and frequency points that best represent the decaying exponentials.
        rates = np.geomspace(energy_min, energy_max, 1500)
        tau_candidates = _composite_gauss_legendre(0.0, np.log(1.0 / tol) / energy_min, 0.25 / energy_max)
        omega_candidates = np.concatenate(([0.0], _composite_gauss_legendre(0.0, 1e3 * energy_max, 0.25 * energy_min)))
        E, C, S = _decay_kernels(rates, tau_candidates, omega_candidates)
        tau = tau_candidates[_select_points(E, tol)]
        omega = np.unique(
            np.concatenate(([0.0], omega_candidates[_select_points(C, tol)], omega_candidates[_select_points(S, tol)]))
        )
        self.tau = np.concatenate((-tau[::-1], tau))
        self.omega = omega

        # Fit the Fourier transforms on a dense set of decay rates.
        self._rates = np.geomspace(energy_min, energy_max, 4500)
        E, C, S = _decay_kernels(self._rates, tau, omega)
        self._cos = _fit(E, C)
        self._sin = _fit(E, S)
        self._inverse_cos = _fit(C, E)
        self._inverse_sin = _fit(S, E)
        self._zero = _fit(E, np.ones((1, self._rates.shape[0])))[0]

        # Fit the quadrature for int_0^inf f(t) dt, for products of two functions (with up to double the decay rate).
        product_rates = np.geomspace(energy_min, 2.0 * energy_max, 4500)
        self._quadrature = _fit(np.exp(-np.outer(tau, product_rates)), 1.0 / product_rates[None, :])[0]

    @property
    def tau_positive(self) -> np.ndarray:
        r"""Positive imaginary times."""
        return self.tau[self.tau.shape[0] // 2 :]

    def split(self, f: np.ndarray) -> tuple:
        r"""
        Split an imaginary-time quantity into its values at positive and negative times.

        | Args:
        |     f: np.ndarray, Imaginary-time quantity, indexed as f[tau,...].

        | Returns:
        |     f_plus, f_minus: np.ndarray, Values f(+t) and f(-t) for t in tau_positive, each indexed as f[t,...].
        """
        L = self.tau.shape[0] // 2
        return f[L:], f[:L][::-1]

    def join(self, f_plus: np.ndarray, f_minus: np.ndarray) -> np.ndarray:
        r"""
        Join the values of an imaginary-time quantity at positive and negative times.

        | Args:
        |     f_plus: np.ndarray, Values f(+t) for t in tau_positive, indexed as f_plus[t,...].
        |     f_minus: np.ndarray, Values f(-t) for t in tau_positive, indexed as f_minus[t,...].

        | Returns:
        |     f: np.ndarray, Imaginary-time quantity, indexed as f[tau,...].
        """
        return np.concatenate((f_minus[::-1], f_plus), axis=0)

    def fourier_transform(self, f: np.ndarray, omega: np.ndarray = None, even: bool = False) -> np.ndarray:
        r"""
        Fourier transform from imaginary time to imaginary frequency.

        .. math:: F(i\omega) = \int d\tau e^{i\omega\tau} f(\tau)

        | Args:
        |     f: np.ndarray, Imaginary-time quantity, indexed as f[tau,...].
        |     omega: np.ndarray, Imaginary frequencies to compute the transform at [If None this will be self.omega]. (default = None)
        |     even: bool, Set to True if f(t) = f(-t), in which case only positive times are used and a real result is returned. (default = False)

        | Returns:
        |     F: np.ndarray, Imaginary-frequency quantity, indexed as F[omega,...].
        """
        if omega is None:
            cos, sin = self._cos, self._sin
        else:
            E, C, S = _decay_kernels(self._rates, self.tau_positive, np.asarray(omega, dtype=float))
            cos, sin = _fit(E, C), _fit(E, S)
        f_plus, f_minus = self.split(f)
        shape = (cos.shape[0],) + f.shape[1:]
        if even:
            return (2.0 * cos @ f_plus.reshape(f_plus.shape[0], -1)).reshape(shape)
        F = np.empty((cos.shape[0], int(np.prod(f.shape[1:]))), dtype=complex)
        F.real = cos @ (f_plus + f_minus).reshape(f_plus.shape[0], -1)
        F.imag = sin @ (f_plus - f_minus).reshape(f_plus.shape[0], -1)
        return F.reshape(shape)

    def inverse_fourier_transform(self, F: np.ndarray, tau: np.ndarray = None, even: bool = False) -> np.ndarray:
        r"""
        Fourier transform from imaginary frequency to imaginary time.

        .. math:: f(\tau) = \frac{1}{2\pi} \int d\omega e^{-i\omega\tau} F(i\omega)

        | Args:
        |     F: np.ndarray, Imaginary-frequency quantity on the grid self.omega, indexed as F[omega,...].
        |     tau: np.ndarray, Imaginary times to compute the transform at, where tau = 0 is taken as 0^- [If None this will be self.tau]. (default = None)
        |     even: bool, Set to True if f(t) = f(-t), in which case F(iw) is real and only its real part is used. (default = False)

        | Returns:
        |     f: np.ndarray, Imaginary-time quantity, indexed as f[tau,...].
        """
        if tau is None:
            inverse_cos, inverse_sin = self._inverse_cos, self._inverse_sin
            sign = np.sign(self.tau)
            shape = self.tau.shape + F.shape[1:]
        else:
            tau = np.asarray(tau, dtype=float)
            E, C, S = _decay_kernels(self._rates, self.tau_positive, self.omega)
            E_target = np.exp(-np.outer(abs(tau), self._rates))
            inverse_cos, inverse_sin = _fit(C, E_target), _fit(S, E_target)
            sign = np.where(tau > 0.0, 1.0, -1.0)
            shape = tau.shape + F.shape[1:]
        F = F.reshape(F.shape[0], -1)
        f_even = 0.5 * (inverse_cos @ F.real)
        if tau is None:
            f_even = np.concatenate((f_even[::-1], f_even), axis=0)
        if even:
            return f_even.reshape(shape)
        f_odd = 0.5 * (inverse_sin @ F.imag)
        if tau is None:
            f_odd = np.concatenate((f_odd[::-1], f_odd), axis=0)
        return (f_even + sign[:, None] * f_odd).reshape(shape)

    def interpolate(self, f: np.ndarray, tau: Union[float, np.ndarray]) -> np.ndarray:
        r"""
        Interpolate an imaginary-time quantity to arbitrary imaginary times.

        | Args:
        |     f: np.ndarray, Imaginary-time quantity, indexed as f[tau,...].
        |     tau: float or np.ndarray, Imaginary time(s) to interpolate to, where tau = 0 is taken as 0^- (the equal-time limit, e.g. G(x,x';0^-) is the density matrix).

        | Returns:
        |     f: np.ndarray, Interpolated quantity, indexed as f[...] if tau is a float, or f[tau,...] if tau is an array.
        """
        f_plus, f_minus = self.split(f)
        if np.ndim(tau) == 0:
            if tau == 0.0:
                return np.tensordot(self._zero, f_minus, axes=(0, 0))
            return self.interpolate(f, np.array([tau]))[0]
        tau = np.asarray(tau, dtype=float)
        E = np.exp(-np.outer(self.tau_positive, self._rates))
        weights = _fit(E, np.exp(-np.outer(abs(tau), self._rates)))
        f_plus = f_plus.reshape(f_plus.shape[0], -1)
        f_minus = f_minus.reshape(f_minus.shape[0], -1)
        values = np.where((tau > 0.0)[:, None], weights @ f_plus, weights @ f_minus)
        return values.reshape(tau.shape + f.shape[1:])

    def integrate(self, f: np.ndarray) -> np.ndarray:
        r"""
        Integrate an imaginary-time quantity over all imaginary time.

        .. math:: \int_{-\infty}^{\infty} d\tau f(\tau)

        | Args:
        |     f: np.ndarray, Imaginary-time quantity, indexed as f[tau,...].

        | Returns:
        |     integral: np.ndarray, Integral, indexed as integral[...].
        """
        f_plus, f_minus = self.split(f)
        return np.tensordot(self._quadrature, f_plus + f_minus, axes=(0, 0))

    def __str__(self):
        return f"iDEA.methods.mbpt.SpaceTimeGrid: {self.tau.shape[0]} imaginary times in [{self.tau[0]:.3e},{self.tau[-1]:.3e}], {self.omega.shape[0]} imaginary frequencies in [{self.omega[0]:.3e},{self.omega[-1]:.3e}]"


def non_interacting_green_function(
    s: iDEA.system.System, grid: SpaceTimeGrid, energies: np.ndarray, orbitals: np.ndarray, mu: float
) -> tuple:
    r"""
    Compute the Green's function of non-interacting (or mean-field) particles from their orbitals and energies.

    .. math:: G(x,x';\tau) = -\sum_{\varepsilon_n > \mu} \phi_n(x) \phi_n(x') e^{-(\varepsilon_n - \mu)\tau} \theta(\tau) + \sum_{\varepsilon_n < \mu} \phi_n(x) \phi_n(x') e^{-(\varepsilon_n - \mu)\tau} \theta(-\tau)

    .. math:: G(x,x';i\omega) = \sum_n \frac{\phi_n(x) \phi_n(x')}{i\omega + \mu - \varepsilon_n}

    | Args:
    |     s: iDEA.system.System, System object.
    |     grid: iDEA.methods.mbpt.SpaceTimeGrid, Imaginary time and frequency grids.
    |     energies: np.ndarray, Single-particle energies, indexed as energies[orbital_number].
    |     orbitals: np.ndarray, Single-particle orbitals, indexed as orbitals[space,orbital_number].
    |     mu: float, Chemical potential (orbitals with energies below mu are occupied).

    | Returns:
    |     G: np.ndarray, Green's function, indexed as G[tau,space,space].
    |     G_iw: np.ndarray, Green's function, indexed as G_iw[omega,space,space].
    """
    g, g_iw = _scalar_green_functions(grid, energies - mu)
    G = _expand(orbitals, g)
    G_iw = np.empty((grid.omega.shape[0],) + G.shape[1:], dtype=complex)
    G_iw.real = _expand(orbitals, g_iw.real)
    G_iw.imag = _expand(orbitals, g_iw.imag)
    return G, G_iw


def _scalar_green_functions(grid: SpaceTimeGrid, e: np.ndarray) -> tuple:
    r"""
    Compute the Green's functions of single levels, with energies e measured relative to the chemical potential.

    .. math:: g_n(\tau) = -\theta(\tau)\theta(e_n) e^{-e_n\tau} + \theta(-\tau)\theta(-e_n) e^{-e_n\tau}, \quad g_n(i\omega) = \frac{1}{i\omega - e_n}

    | Args:
    |     grid: iDEA.methods.mbpt.SpaceTimeGrid, Imaginary time and frequency grids.
    |     e: np.ndarray, Energies relative to the chemical potential, indexed as e[orbital_number].

    | Returns:
    |     g: np.ndarray, Green's functions in imaginary time, indexed as g[tau,orbital_number].
    |     g_iw: np.ndarray, Green's functions in imaginary frequency, indexed as g_iw[omega,orbital_number].
    """
    exponent = -np.outer(grid.tau_positive, abs(e))
    g_plus = np.where(e > 0.0, -np.exp(exponent), 0.0)
    g_minus = np.where(e < 0.0, np.exp(exponent), 0.0)
    g_iw = 1.0 / (1.0j * grid.omega[:, None] - e[None, :])
    return grid.join(g_plus, g_minus), g_iw


def _expand(orbitals: np.ndarray, g: np.ndarray) -> np.ndarray:
    r"""
    Expand diagonal quantities in a basis of orbitals: A[k] = sum_n orbitals[:,n] g[k,n] orbitals[:,n]^T.

    | Args:
    |     orbitals: np.ndarray, Orbitals, indexed as orbitals[space,orbital_number].
    |     g: np.ndarray, Diagonal quantities, indexed as g[k,orbital_number].

    | Returns:
    |     A: np.ndarray, Expanded quantities, indexed as A[k,space,space].
    """
    return (orbitals[None, :, :] * g[:, None, :]) @ orbitals.T


def polarisability(s: iDEA.system.System, grid: SpaceTimeGrid, up_G: np.ndarray, down_G: np.ndarray) -> tuple:
    r"""
    Compute the irreducible polarisability in the GW approximation (the independent-particle polarisability of G).

    .. math:: P(x,x';\tau) = \sum_\sigma G_\sigma(x,x';\tau) G_\sigma(x',x;-\tau)

    | Args:
    |     s: iDEA.system.System, System object.
    |     grid: iDEA.methods.mbpt.SpaceTimeGrid, Imaginary time and frequency grids.
    |     up_G: np.ndarray, Green's function of up electrons, indexed as up_G[tau,space,space].
    |     down_G: np.ndarray, Green's function of down electrons, indexed as down_G[tau,space,space].

    | Returns:
    |     P: np.ndarray, Polarisability, indexed as P[tau,space,space].
    |     P_iw: np.ndarray, Polarisability, indexed as P_iw[omega,space,space].
    """
    P_plus = 0.0
    for G in (up_G, down_G):
        G_plus, G_minus = grid.split(G)
        P_plus = P_plus + G_plus * np.swapaxes(G_minus, 1, 2)
    P_plus = 0.5 * (P_plus + np.swapaxes(P_plus, 1, 2))
    P = grid.join(P_plus, P_plus)
    P_iw = grid.fourier_transform(P, even=True)
    return P, P_iw


def screened_interaction(s: iDEA.system.System, grid: SpaceTimeGrid, P_iw: np.ndarray) -> tuple:
    r"""
    Compute the screened interaction from the polarisability.

    .. math:: W(i\omega) = v + v P(i\omega) W(i\omega) = [1 - v P(i\omega)]^{-1} v

    As W contains the instantaneous bare interaction v (a delta function in imaginary time), only the
    correlation part W_c = W - v is represented in imaginary time.

    | Args:
    |     s: iDEA.system.System, System object.
    |     grid: iDEA.methods.mbpt.SpaceTimeGrid, Imaginary time and frequency grids.
    |     P_iw: np.ndarray, Polarisability, indexed as P_iw[omega,space,space].

    | Returns:
    |     W_c: np.ndarray, Correlation part of the screened interaction, indexed as W_c[tau,space,space].
    |     W_c_iw: np.ndarray, Correlation part of the screened interaction, indexed as W_c_iw[omega,space,space].
    |     W_iw: np.ndarray, Screened interaction, indexed as W_iw[omega,space,space].
    """
    v = s.v_int
    diagonal = np.arange(s.x.shape[0])
    vP = v[None, :, :] @ P_iw
    vP *= s.dx**2
    vPv = vP @ v[None, :, :]
    vP *= -1.0
    vP[:, diagonal, diagonal] += 1.0
    W_c_iw = np.linalg.solve(vP, vPv)
    W_c_iw += np.swapaxes(W_c_iw, 1, 2)
    W_c_iw *= 0.5
    W_c = grid.inverse_fourier_transform(W_c_iw, even=True)
    W_iw = W_c_iw + v[None, :, :]
    return W_c, W_c_iw, W_iw


def exchange_self_energy(s: iDEA.system.System, p: np.ndarray) -> np.ndarray:
    r"""
    Compute the exchange self-energy (the Fock exchange operator) from a density matrix.

    .. math:: \Sigma_x(x,x') = -\rho(x,x') v(x,x')

    | Args:
    |     s: iDEA.system.System, System object.
    |     p: np.ndarray, Density matrix of the spin channel, indexed as p[space,space].

    | Returns:
    |     sigma_x: np.ndarray, Exchange self-energy, indexed as sigma_x[space,space].
    """
    return iDEA.observables.exchange_potential(s, p)


def correlation_self_energy(s: iDEA.system.System, grid: SpaceTimeGrid, G: np.ndarray, W_c: np.ndarray) -> tuple:
    r"""
    Compute the correlation self-energy in the GW approximation.

    .. math:: \Sigma_c(x,x';\tau) = -G(x,x';\tau) W_c(x,x';\tau)

    | Args:
    |     s: iDEA.system.System, System object.
    |     grid: iDEA.methods.mbpt.SpaceTimeGrid, Imaginary time and frequency grids.
    |     G: np.ndarray, Green's function, indexed as G[tau,space,space].
    |     W_c: np.ndarray, Correlation part of the screened interaction, indexed as W_c[tau,space,space].

    | Returns:
    |     sigma_c: np.ndarray, Correlation self-energy, indexed as sigma_c[tau,space,space].
    |     sigma_c_iw: np.ndarray, Correlation self-energy, indexed as sigma_c_iw[omega,space,space].
    """
    sigma_c = -G * W_c
    sigma_c_iw = grid.fourier_transform(sigma_c)
    return sigma_c, sigma_c_iw


def dyson_equation(
    s: iDEA.system.System, grid: SpaceTimeGrid, H: np.ndarray, sigma_c_iw: np.ndarray, mu: float
) -> tuple:
    r"""
    Solve Dyson's equation for the Green's function.

    .. math:: G(i\omega) = [i\omega + \mu - h - \Sigma_c(i\omega)]^{-1}

    where h is a static (mean-field) Hamiltonian that contains all static parts of the self-energy. To transform G to
    imaginary time accurately, G is split into the Green's function of h, which is known analytically in imaginary
    time, and the remainder, which decays rapidly with frequency and is transformed numerically. (In practice the
    whole of G is transformed numerically, and the transform of the Green's function of h is then corrected to its
    analytic value in the eigenbasis of h.)

    | Args:
    |     s: iDEA.system.System, System object.
    |     grid: iDEA.methods.mbpt.SpaceTimeGrid, Imaginary time and frequency grids.
    |     H: np.ndarray, Static Hamiltonian h (kinetic, external, Hartree and exchange) as a matrix acting on the grid.
    |     sigma_c_iw: np.ndarray, Correlation self-energy, indexed as sigma_c_iw[omega,space,space].
    |     mu: float, Chemical potential.

    | Returns:
    |     G: np.ndarray, Green's function, indexed as G[tau,space,space].
    |     G_iw: np.ndarray, Green's function, indexed as G_iw[omega,space,space].
    """
    diagonal = np.arange(s.x.shape[0])
    A = sigma_c_iw * (-s.dx)
    A -= H[None, :, :]
    A[:, diagonal, diagonal] += (1.0j * grid.omega + mu)[:, None]
    G_iw = np.linalg.inv(A)
    G_iw /= s.dx
    energies, orbitals = spla.eigh(H)
    g, g_iw = _scalar_green_functions(grid, energies - mu)
    G = grid.inverse_fourier_transform(G_iw)
    G += _expand(orbitals / np.sqrt(s.dx), g - grid.inverse_fourier_transform(g_iw))
    return G, G_iw


def quasiparticle_states(s: iDEA.system.System, H: np.ndarray, sigma_c_iw: np.ndarray, count: int) -> tuple:
    r"""
    Compute the quasiparticle energies and orbitals in the static approximation, in which the correlation self-energy
    is evaluated at the chemical potential (iw = 0), i.e. the eigenstates of h + Sigma_c(iw=0).

    | Args:
    |     s: iDEA.system.System, System object.
    |     H: np.ndarray, Static Hamiltonian h (kinetic, external, Hartree and exchange) as a matrix acting on the grid.
    |     sigma_c_iw: np.ndarray, Correlation self-energy at iw = 0, indexed as sigma_c_iw[space,space].
    |     count: int, Number of electrons in the spin channel.

    | Returns:
    |     energies: np.ndarray, Quasiparticle energies, indexed as energies[orbital_number].
    |     orbitals: np.ndarray, Quasiparticle orbitals, indexed as orbitals[space,orbital_number].
    |     occupations: np.ndarray, Quasiparticle occupations, indexed as occupations[orbital_number].
    """
    sigma_c = 0.5 * (sigma_c_iw.real + sigma_c_iw.real.T)
    energies, orbitals = spla.eigh(H + sigma_c * s.dx)
    occupations = np.zeros(shape=energies.shape)
    occupations[:count] = 1.0
    return energies, orbitals / np.sqrt(s.dx), occupations


def _midgap(energies: np.ndarray, count: int, half_gap: float) -> tuple:
    r"""
    Compute the chemical potential at the centre of the gap of a spin channel.

    | Args:
    |     energies: np.ndarray, Sorted (quasi)particle energies of the spin channel.
    |     count: int, Number of electrons in the spin channel.
    |     half_gap: float, Distance below the lowest level to place the chemical potential, if the channel is empty.

    | Returns:
    |     mu: float, Chemical potential.
    |     half_gap: float, Half of the gap.
    """
    if count == 0:
        return energies[0] - half_gap, half_gap
    if count >= energies.shape[0]:
        raise ValueError("Cannot place the chemical potential: every state of the spin channel is occupied.")
    homo, lumo = energies[count - 1], energies[count]
    if lumo - homo <= 1e-8:
        raise ValueError(
            f"Cannot place the chemical potential: the HOMO ({homo:.6f}) and LUMO ({lumo:.6f}) are degenerate."
        )
    return 0.5 * (homo + lumo), 0.5 * (lumo - homo)


def _starting_point(s: iDEA.system.System, initial, silent: bool) -> iDEA.state.SingleBodyState:
    r"""
    Compute the mean-field starting point.

    | Args:
    |     s: iDEA.system.System, System object.
    |     initial: str or iDEA.state.SingleBodyState, Name of a mean-field method, or a solved state.
    |     silent: bool, Set to true to prevent printing.

    | Returns:
    |     state: iDEA.state.SingleBodyState, Starting point.
    """
    if isinstance(initial, iDEA.state.SingleBodyState):
        state = initial
    else:
        methods = {
            "non_interacting": iDEA.methods.non_interacting,
            "hartree": iDEA.methods.hartree,
            "hartree_fock": iDEA.methods.hartree_fock,
            "lda": iDEA.methods.lda,
            "hybrid": iDEA.methods.hybrid,
        }
        if initial not in methods:
            raise ValueError(f"initial must be one of {list(methods)} or an iDEA.state.SingleBodyState, got {initial}.")
        state = methods[initial].solve(s, silent=silent)
    for spin, count in ((state.up, s.up_count), (state.down, s.down_count)):
        if np.max(abs(np.imag(spin.orbitals))) > 1e-10:
            raise ValueError("The starting point must have real orbitals.")
        occupied = np.argsort(spin.energies, kind="stable")[:count]
        if not np.allclose(np.sort(occupied), np.nonzero(spin.occupations)[0]):
            raise ValueError("The starting point must be a ground state, with the lowest energy orbitals occupied.")
    return state


def total_energy(s: iDEA.system.System, state: iDEA.state.QuasiParticleState) -> float:
    r"""
    Compute the total energy using the Galitskii-Migdal formula.

    .. math:: E = \sum_\sigma \mathrm{Tr}[h_0 \rho_\sigma] + \frac{1}{2} \sum_\sigma \mathrm{Tr}[(v_h + \Sigma_{x,\sigma}) \rho_\sigma] + \frac{1}{2} \sum_\sigma \int d\tau \mathrm{Tr}[\Sigma_{c,\sigma}(\tau) G_\sigma(-\tau)]

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.QuasiParticleState, State.

    | Returns:
    |     E: float, Total energy.
    """
    H0 = kinetic_energy_operator(s) + external_potential_operator(s)
    E = 0.0
    for spin in (state.up, state.down):
        p = state.grid.interpolate(spin.G, 0.0)
        E += np.sum(H0 * p.T) * s.dx
        E += 0.5 * np.dot(state.v_h, np.diag(p)) * s.dx
        E += 0.5 * np.sum(spin.sigma_x * p.T) * s.dx**2
        sigma_c_plus, sigma_c_minus = state.grid.split(spin.sigma_c)
        G_plus, G_minus = state.grid.split(spin.G)
        trace_plus = np.einsum("tij,tji->t", sigma_c_plus, G_minus)
        trace_minus = np.einsum("tij,tji->t", sigma_c_minus, G_plus)
        E += 0.5 * state.grid.integrate(state.grid.join(trace_plus, trace_minus)) * s.dx**2
    return float(E)


def solve(
    s: iDEA.system.System,
    mode: str = "GW",
    initial: Union[str, iDEA.state.SingleBodyState] = "hartree_fock",
    hedin_shift: bool = True,
    mixing: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 100,
    grid: SpaceTimeGrid = None,
    grid_tol: float = 1e-10,
    silent: bool = False,
) -> iDEA.state.QuasiParticleState:
    r"""
    Solves Hedin's equations in the GW approximation for the ground state of the given system.

    | Args:
    |     s: iDEA.system.System, System object.
    |     mode: str, Level of self-consistency: 'G0W0' (one-shot), 'GW0' (self-consistent G, fixed W) or 'GW' (fully self-consistent). (default = 'GW')
    |     initial: str or iDEA.state.SingleBodyState, Mean-field starting point: one of 'non_interacting', 'hartree', 'hartree_fock', 'lda', 'hybrid', or a solved ground state. (default = 'hartree_fock')
    |     hedin_shift: bool, Keep the chemical potential at the centre of the quasiparticle gap by shifting the self-energy by a constant. This does not change the self-consistent result (GW and GW0). For G0W0, it aligns the chemical potentials of G0 and G (Hedin shift). (default = True)
    |     mixing: float, Mixing parameter for the self-consistent Green's function. (default = 1.0)
    |     tol: float, Tolerance of convergence of the density. (default = 1e-8)
    |     max_iter: int, Maximum number of self-consistent iterations. (default = 100)
    |     grid: iDEA.methods.mbpt.SpaceTimeGrid, Imaginary time and frequency grids [If None these will be constructed from the starting point]. (default = None)
    |     grid_tol: float, Tolerance of the imaginary time and frequency grids, if constructed. (default = 1e-10)
    |     silent: bool, Set to true to prevent printing. (default = False)

    | Returns:
    |     state: iDEA.state.QuasiParticleState, Solved state.
    """
    if mode not in ("G0W0", "GW0", "GW"):
        raise ValueError(f"mode must be one of 'G0W0', 'GW0' or 'GW', got {mode} instead.")
    if s.count == 0:
        raise ValueError("The system must contain at least one electron.")

    # Compute the mean-field starting point, and the chemical potential of each spin channel.
    start = _starting_point(s, initial, silent)
    counts = (s.up_count, s.down_count)
    starts = (start.up, start.down)
    half_gaps = [_midgap(spin.energies, count, 0.0)[1] for spin, count in zip(starts, counts) if count > 0]
    empty_half_gap = min(half_gaps)
    mus0 = [_midgap(spin.energies, count, empty_half_gap)[0] for spin, count in zip(starts, counts)]
    mus = list(mus0)

    # Construct the imaginary time and frequency grids, to represent all excitations of the starting point.
    if grid is None:
        energy_min = 0.25 * empty_half_gap
        energy_max = 3.0 * max(np.max(abs(spin.energies - mu)) for spin, mu in zip(starts, mus))
        grid = SpaceTimeGrid(energy_min, energy_max, grid_tol)

    # Construct the Green's functions of the starting point.
    G0s = [non_interacting_green_function(s, grid, spin.energies, spin.orbitals, mu) for spin, mu in zip(starts, mus)]
    restricted = s.up_count == s.down_count and np.allclose(G0s[0][0], G0s[1][0], rtol=0.0, atol=1e-8)
    spins = (0,) if restricted else (0, 1)
    Gs = [G0s[0][0], G0s[0][0] if restricted else G0s[1][0]]

    # Run the self-consistent algorithm.
    H0 = kinetic_energy_operator(s) + external_potential_operator(s)
    W_c = None
    convergence = 1.0
    for iteration in range(max_iter):
        # Static self-energy (Hartree and exchange) from the density matrix G(0^-).
        ps = [grid.interpolate(G, 0.0) for G in Gs]
        n = np.diag(ps[0]) + np.diag(ps[1])
        v_h = iDEA.observables.hartree_potential(s, n)
        sigma_xs = [exchange_self_energy(s, p) for p in ps]
        Hs = [H0 + np.diag(v_h) + sigma_x * s.dx for sigma_x in sigma_xs]

        # Screened interaction (only computed once for G0W0 and GW0).
        if W_c is None or mode == "GW":
            P, P_iw = polarisability(s, grid, Gs[0], Gs[1])
            W_c, W_c_iw, W_iw = screened_interaction(s, grid, P_iw)

        # Correlation self-energy, the chemical potential, and Dyson's equation.
        results = []
        for i in spins:
            sigma_c, sigma_c_iw = correlation_self_energy(s, grid, Gs[i], W_c)
            energies, orbitals, occupations = quasiparticle_states(s, Hs[i], sigma_c_iw[0], counts[i])
            if hedin_shift:
                mus[i] = _midgap(energies, counts[i], empty_half_gap)[0]
            elif np.sum(energies < mus[i]) != counts[i]:
                raise RuntimeError(
                    f"iDEA.methods.{name}.solve: the chemical potential ({mus[i]:.6f}) lies outside the quasiparticle gap. "
                    "Use hedin_shift=True, or a starting point with a quasiparticle gap closer to that of GW."
                )
            G, G_iw = dyson_equation(s, grid, Hs[i], sigma_c_iw, mus[i])
            results.append((G, G_iw, sigma_c, sigma_c_iw, energies, orbitals, occupations))
        if restricted:
            results.append(results[0])
            mus[1] = mus[0]

        # Test for convergence.
        n_new = np.diag(grid.interpolate(results[0][0], 0.0)) + np.diag(grid.interpolate(results[1][0], 0.0))
        convergence = np.sum(abs(n_new - n)) * s.dx
        if silent is False:
            print(
                rf"iDEA.methods.{name}.solve: iteration = {iteration + 1}, convergence = {convergence:.5}, tolerance = {tol:.5}",
                end="\r",
            )
        if mode == "G0W0" or convergence < tol:
            break

        # Perform mixing.
        Gs = [mixing * result[0] + (1.0 - mixing) * G for result, G in zip(results[: len(spins)], Gs)]
        if restricted:
            Gs.append(Gs[0])
    if silent is False:
        print()
    if mode != "G0W0" and convergence >= tol:
        warnings.warn(
            f"iDEA.methods.{name}.solve: not converged after {max_iter} iterations (convergence = {convergence:.3e}).",
            stacklevel=2,
        )
    energy_scale = min(np.min(abs(result[4] - mu)) for result, mu in zip(results, mus))
    if energy_scale < 2.0 * grid.energy_min:
        warnings.warn(
            f"iDEA.methods.{name}.solve: the quasiparticle gap is small compared to the smallest energy represented by the grid, results may be inaccurate. Consider constructing a grid with a smaller energy_min.",
            stacklevel=2,
        )

    # Construct the quasiparticle state.
    state = iDEA.state.QuasiParticleState()
    state.tau = grid.tau
    state.omega = grid.omega
    state.grid = grid
    state.mode = mode
    state.v_h = v_h
    state.P = P
    state.P_iw = P_iw
    state.W_c = W_c
    state.W_c_iw = W_c_iw
    state.W_iw = W_iw
    sigma_h_iw = np.diag(v_h) / s.dx
    for i, spin in zip(spins, (state.up, state.down)):
        G, G_iw, sigma_c, sigma_c_iw, energies, orbitals, occupations = results[i]
        spin.mu = mus[i]
        spin.mu0 = mus0[i]
        spin.G = G
        spin.G_iw = G_iw
        spin.G0 = G0s[i][0]
        spin.G0_iw = G0s[i][1]
        spin.sigma_x = sigma_xs[i]
        spin.sigma_c = sigma_c
        spin.sigma_c_iw = sigma_c_iw
        spin.sigma_xc_iw = sigma_xs[i][None, :, :] + sigma_c_iw
        spin.sigma_iw = sigma_h_iw[None, :, :] + spin.sigma_xc_iw
        spin.energies = energies
        spin.orbitals = orbitals
        spin.occupations = occupations
        spin.occupied = np.nonzero(occupations)[0]
    if restricted:
        for attribute, value in vars(state.up).items():
            setattr(state.down, attribute, value)
    state.energy = total_energy(s, state)

    return state
