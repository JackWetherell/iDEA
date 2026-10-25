"""Defines the structures to describe the system states"""

import copy
import pickle
from abc import ABC as Interface

import numpy as np

import iDEA.utilities

__all__ = [
    "State",
    "States",
    "ManyBodyState",
    "ManyBodyStates",
    "SingleBodyState",
    "QuasiParticleState",
    "GreenFunction",
    "Evolution",
    "ManyBodyEvolution",
    "SingleBodyEvolution",
]


class State(Interface):
    """Interface class representing a static state."""


class States(Interface):
    """Interface class representing a group of static states."""


class Evolution(Interface):
    """Interface class representing a time-dependent evolution of a state."""


class ManyBodyState(State):
    """State of interacting particles."""

    def __init__(
        self,
        space: np.ndarray = None,
        spin: np.ndarray = None,
        full=None,
        energy=None,
    ):
        r"""
        State of particles in a many-body state.

        This is described by a spatial part
        .. math:: \psi(x_1,x_2,\dots,x_N)
        on the spatial grid, and a spin
        part on the spin grid
        .. math:: \chi(\sigma_1,\sigma_2,\dots,\sigma_N).
        These are NOT necessarily antisymmetric states,
        they can be combined using the antisymmetrisation operaration to produce the full
        wavefunction
        .. math:: \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N).

        | Args:
        |     space: np.ndarray, Spatial part of the wavefunction on the spatial grid \psi(x_1,x_2,\dots,x_N). (default = None)
        |     spin: np.ndarray, Spin part of the wavefunction on the spin grid \chi(\sigma_1,\sigma_2,\dots,\sigma_N). (default = None)
        |     full: np.ndarray, Total antisymmetrised wavefunction \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N). (default = None)
        |     energy: float, Total energy of the state.
        """
        if space is None:
            self.space = iDEA.utilities.ArrayPlaceholder()
        else:
            self.space = space
        if spin is None:
            self.spin = iDEA.utilities.ArrayPlaceholder()
        else:
            self.spin = spin
        if full is None:
            self.full = iDEA.utilities.ArrayPlaceholder()
        else:
            self.full = full
        if energy is None:
            self.energy = 0.0
        else:
            self.energy = energy


class ManyBodyStates(States):
    """A collection of many-body states over a range of energy levels from a single system of interacting particles."""

    def __init__(self, spaces: np.ndarray = None, spins: np.ndarray = None, fulls=None, energies=None):
        r"""
        Collection of many-body states indexed by energy level n.

        Each state is described by a spatial part
        .. math:: \psi(x_1,x_2,\dots,x_N,n)
        on the spatial grid, and a spin part on the spin grid
        .. math:: \chi(\sigma_1,\sigma_2,\dots,\sigma_N,n).
        These are NOT necessarily antisymmetric states;
        they can be combined using the antisymmetrisation operation to produce the full
        wavefunction
        .. math:: \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N,n).

        | Args:
        |     spaces: np.ndarray, Spatial part of each wavefunction on the spatial grid \psi(x_1,x_2,\dots,x_N,n). (default = None)
        |     spins: np.ndarray, Spin part of each wavefunction on the spin grid \chi(\sigma_1,\sigma_2,\dots,\sigma_N,n). (default = None)
        |     fulls: np.ndarray, Total antisymmetrised wavefunction of each state \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N,n). (default = None)
        |     energies: np.ndarray, Total energy of each state, indexed as energies[n]. (default = None)
        """
        if spaces is None:
            self.spaces = iDEA.utilities.ArrayPlaceholder()
        else:
            self.spaces = spaces
        if spins is None:
            self.spins = iDEA.utilities.ArrayPlaceholder()
        else:
            self.spins = spins
        if fulls is None:
            self.fulls = iDEA.utilities.ArrayPlaceholder()
        else:
            self.fulls = fulls
        if energies is None:
            self.energies = iDEA.utilities.ArrayPlaceholder()
        else:
            self.energies = energies


class SingleBodyState(State):
    r"""
    State of particles in a single-body state.

    This is described by three arrays for each spin channel:

    | up.energies: np.ndarray, Array of single-body energies, indexed as energies[orbital_number].
    | up.orbitals: np.ndarray, Array of single-body orbitals, indexed as orbitals[space,orbital_number].
    | up.occupations: np.ndarray, Array of single-body occupations, indexed as occupations[orbital_number].
    | up.occupied: np.ndarray, Indices of up.occupations that are non-zero, to indicate occupied orbitals.

    | down.energies: np.ndarray, Array of single-body energies, indexed as energies[orbital_number].
    | down.orbitals: np.ndarray, Array of single-body orbitals, indexed as orbitals[space,orbital_number].
    | down.occupations: np.ndarray, Array of single-body occupations, indexed as occupations[orbital_number].
    | down.occupied: np.ndarray, Indices of down.occupations that are non-zero, to indicate occupied orbitals.
    """

    def __init__(self):
        self.up = iDEA.utilities.Container()
        self.down = iDEA.utilities.Container()

        self.up.energies = iDEA.utilities.ArrayPlaceholder()
        self.up.orbitals = iDEA.utilities.ArrayPlaceholder()
        self.up.occupations = iDEA.utilities.ArrayPlaceholder()
        self.up.occupied = iDEA.utilities.ArrayPlaceholder()

        self.down.energies = iDEA.utilities.ArrayPlaceholder()
        self.down.orbitals = iDEA.utilities.ArrayPlaceholder()
        self.down.occupations = iDEA.utilities.ArrayPlaceholder()
        self.down.occupied = iDEA.utilities.ArrayPlaceholder()


class QuasiParticleState(State):
    r"""
    State of interacting particles described by the one-body Green's function, as given by many-body perturbation theory.

    The central quantity is the Green's function on the real-space grid and the imaginary-time grid, which is real-valued:

    .. math:: G_\sigma(x,x';\tau) = -\langle T_\tau \hat{\psi}_\sigma(x,\tau) \hat{\psi}^\dagger_\sigma(x',0) \rangle

    and its Fourier transform onto the imaginary-frequency grid:

    .. math:: G_\sigma(x,x';i\omega) = \int d\tau e^{i\omega\tau} G_\sigma(x,x';\tau)

    The equal-time limit gives the density matrix:

    .. math:: \rho_\sigma(x,x') = G_\sigma(x,x';\tau=0^-)

    In the convention of Rojas, Godby and Needs, G(x,x';i tau) = iG(x,x';tau).

    All two-point quantities are stored as kernels A(x,x'), so the operator they represent acts as .. math:: \int A(x,x')f(x')dx'.
    Imaginary-time quantities are indexed as A[tau,space,space] on the grid tau, and imaginary-frequency
    quantities as A_iw[omega,space,space] on the grid omega. Static quantities are indexed as A[space,space].

    | tau: np.ndarray, Imaginary-time grid (negative and positive times, ascending), indexed as tau[time].
    | omega: np.ndarray, Imaginary-frequency grid (non-negative frequencies), indexed as omega[frequency].
    | grid: iDEA.methods.mbpt.SpaceTimeGrid, Grid object, providing the Fourier transforms between tau and omega.
    | mode: str, Level of self-consistency used to compute the state ('G0W0', 'GW0' or 'GW').
    | energy: float, Total energy, computed with the Galitskii-Migdal formula.
    | v_h: np.ndarray, Hartree potential, indexed as v_h[space].
    | P: np.ndarray, Irreducible polarisability (summed over spin), indexed as P[tau,space,space].
    | P_iw: np.ndarray, Irreducible polarisability (summed over spin), indexed as P_iw[omega,space,space].
    | W_c: np.ndarray, Correlation part of the screened interaction W - v, indexed as W_c[tau,space,space].
    | W_c_iw: np.ndarray, Correlation part of the screened interaction W - v, indexed as W_c_iw[omega,space,space].
    | W_iw: np.ndarray, Screened interaction W = v + W_c, indexed as W_iw[omega,space,space].

    For each spin channel (up and down):

    | up.mu: float, Chemical potential that G is measured relative to (lies within the quasiparticle gap).
    | up.G: np.ndarray, Green's function, indexed as G[tau,space,space].
    | up.G_iw: np.ndarray, Green's function, indexed as G_iw[omega,space,space].
    | up.G0: np.ndarray, Green's function of the mean-field starting point, indexed as G0[tau,space,space].
    | up.G0_iw: np.ndarray, Green's function of the mean-field starting point, indexed as G0_iw[omega,space,space].
    | up.mu0: float, Chemical potential that G0 is measured relative to.
    | up.sigma_x: np.ndarray, Exchange self-energy (static), indexed as sigma_x[space,space].
    | up.sigma_c: np.ndarray, Correlation self-energy, indexed as sigma_c[tau,space,space].
    | up.sigma_c_iw: np.ndarray, Correlation self-energy, indexed as sigma_c_iw[omega,space,space].
    | up.sigma_xc_iw: np.ndarray, Exchange-correlation self-energy, indexed as sigma_xc_iw[omega,space,space].
    | up.sigma_iw: np.ndarray, Total self-energy (Hartree + exchange + correlation), indexed as sigma_iw[omega,space,space].
    | up.energies: np.ndarray, Quasiparticle energies in the static approximation (self-energy evaluated at the chemical potential), indexed as energies[orbital_number].
    | up.orbitals: np.ndarray, Quasiparticle orbitals in the static approximation, indexed as orbitals[space,orbital_number].
    | up.occupations: np.ndarray, Occupations of the quasiparticle orbitals, indexed as occupations[orbital_number].
    | up.occupied: np.ndarray, Indices of up.occupations that are non-zero, to indicate occupied orbitals.
    """

    def __init__(self):
        self.tau = iDEA.utilities.ArrayPlaceholder()
        self.omega = iDEA.utilities.ArrayPlaceholder()
        self.grid = None
        self.mode = None
        self.energy = 0.0
        self.v_h = iDEA.utilities.ArrayPlaceholder()
        self.P = iDEA.utilities.ArrayPlaceholder()
        self.P_iw = iDEA.utilities.ArrayPlaceholder()
        self.W_c = iDEA.utilities.ArrayPlaceholder()
        self.W_c_iw = iDEA.utilities.ArrayPlaceholder()
        self.W_iw = iDEA.utilities.ArrayPlaceholder()

        self.up = iDEA.utilities.Container()
        self.down = iDEA.utilities.Container()
        for spin in (self.up, self.down):
            spin.mu = 0.0
            spin.mu0 = 0.0
            spin.G = iDEA.utilities.ArrayPlaceholder()
            spin.G_iw = iDEA.utilities.ArrayPlaceholder()
            spin.G0 = iDEA.utilities.ArrayPlaceholder()
            spin.G0_iw = iDEA.utilities.ArrayPlaceholder()
            spin.sigma_x = iDEA.utilities.ArrayPlaceholder()
            spin.sigma_c = iDEA.utilities.ArrayPlaceholder()
            spin.sigma_c_iw = iDEA.utilities.ArrayPlaceholder()
            spin.sigma_xc_iw = iDEA.utilities.ArrayPlaceholder()
            spin.sigma_iw = iDEA.utilities.ArrayPlaceholder()
            spin.energies = iDEA.utilities.ArrayPlaceholder()
            spin.orbitals = iDEA.utilities.ArrayPlaceholder()
            spin.occupations = iDEA.utilities.ArrayPlaceholder()
            spin.occupied = iDEA.utilities.ArrayPlaceholder()


class GreenFunction(State):
    r"""
    Exact one-body Green's function and self-energy of an interacting state, in Lehmann (pole) form.

    The Green's function of each spin channel is a sum over the poles of the (N-1) and (N+1)-electron systems:

    .. math:: G_\sigma(x,x';z) = \sum_k \frac{d_k(x) d_k^*(x')}{z - \varepsilon_k}

    where the removal poles are :math:`\varepsilon_k = E_0^N - E_k^{N-1}` with Dyson orbitals
    :math:`d_k(x) = \langle \Psi_k^{N-1} | \hat{\psi}_\sigma(x) | \Psi_0^N \rangle` (occupation 1), and the addition
    poles are :math:`\varepsilon_k = E_k^{N+1} - E_0^N` with Dyson orbitals
    :math:`d_k(x) = \langle \Psi_0^N | \hat{\psi}_\sigma(x) | \Psi_k^{N+1} \rangle` (occupation 0). This is the same
    structure as a SingleBodyState (energies, orbitals and occupations), so the same code evaluates mean-field and
    exact Green's functions.

    The exchange-correlation self-energy, defined by
    :math:`G^{-1}(z) = z - h_0 - v_h - \Sigma_{xc}(z)`, is stored as a static exchange part and a sum over poles:

    .. math:: \Sigma_{xc,\sigma}(x,x';z) = \Sigma_{x,\sigma}(x,x') + \sum_p \frac{r_p(x) r_p^*(x')}{z - \zeta_p}

    Each self-energy pole has a character :math:`s_p \in [-1,1]`, the weight with which it is built from addition
    (+1) and removal (-1) poles of G. Its sign gives the time ordering of the pole: particle-like if :math:`s_p > 0`,
    hole-like if :math:`s_p < 0`.

    All two-point quantities are stored as kernels A(x,x'), so the operator they represent acts as
    .. math:: \int A(x,x') f(x') dx
    and the orbitals are normalised so that :math:`\sum_k d_k(x) d_k^*(x') = \delta(x-x')`.

    | mu: float, Chemical potential at the centre of the gap, (E_0^{N+1} - E_0^{N-1})/2, over both spin channels (None if there is no gap).
    | energy: float, Total energy of the interacting state.
    | v_h: np.ndarray, Hartree potential, indexed as v_h[space].
    | info: dict, Details of the calculation (solvers, sector dimensions and sum-rule residuals).

    For each spin channel (up and down):

    | up.mu: float, Chemical potential (the same for both spin channels).
    | up.energies: np.ndarray, Pole energies, indexed as energies[pole].
    | up.orbitals: np.ndarray, Dyson orbitals, indexed as orbitals[space,pole].
    | up.occupations: np.ndarray, Occupations of the poles (1 for removal, 0 for addition), indexed as occupations[pole].
    | up.sigma_x: np.ndarray, Exchange self-energy (static), indexed as sigma_x[space,space].
    | up.sigma_energies: np.ndarray, Correlation self-energy pole energies, indexed as sigma_energies[pole].
    | up.sigma_orbitals: np.ndarray, Correlation self-energy pole residue vectors, indexed as sigma_orbitals[space,pole].
    | up.sigma_character: np.ndarray, Character of the self-energy poles (> 0 particle-like, < 0 hole-like), indexed as sigma_character[pole].
    """

    def __init__(self):
        self.mu = None
        self.energy = 0.0
        self.v_h = iDEA.utilities.ArrayPlaceholder()
        self.info = {}

        self.up = iDEA.utilities.Container()
        self.down = iDEA.utilities.Container()
        for spin in (self.up, self.down):
            spin.mu = None
            spin.energies = iDEA.utilities.ArrayPlaceholder()
            spin.orbitals = iDEA.utilities.ArrayPlaceholder()
            spin.occupations = iDEA.utilities.ArrayPlaceholder()
            spin.sigma_x = iDEA.utilities.ArrayPlaceholder()
            spin.sigma_energies = iDEA.utilities.ArrayPlaceholder()
            spin.sigma_orbitals = iDEA.utilities.ArrayPlaceholder()
            spin.sigma_character = iDEA.utilities.ArrayPlaceholder()


class ManyBodyEvolution(Evolution):
    r"""
    Time-dependent evolution of particles in a many-body state.

    In addition to the arrays defined within the initial ManyBodyState, this state is described by three additional arrays:

    | td_space: np.ndarray, Spatial part of the wavefunction on the spatial grid \psi(t,x_1,x_2,\dots,x_N).
    | v_ptrb: np.ndarray, Perturbation potential that this time-dependence was driven by. indexed as v_ptrb[space] if static, and v_ptrb[time,space] if dynamic.
    | t: np.ndarray, Time grid used during evolution.
    """

    def __init__(self, initial_state: ManyBodyState):
        self.space = copy.deepcopy(initial_state.space)
        self.spin = copy.deepcopy(initial_state.spin)
        self.full = copy.deepcopy(initial_state.full)
        self.td_space = iDEA.utilities.ArrayPlaceholder()
        self.v_ptrb = iDEA.utilities.ArrayPlaceholder()
        self.t = iDEA.utilities.ArrayPlaceholder()


class SingleBodyEvolution(Evolution):
    r"""
    Time-dependent evolution of particles in a single-body state.

    In addition to the arrays defined within the initial SingleBodyState, this state is described by four additional arrays:

    | up.td_orbitals: np.ndarray, Array of single-body time-dependend orbitals, indexed as orbitals[time,space,orbital_number].
    | down.td_orbital: np.ndarray, Array of single-body time-dependend orbitals, indexed as orbitals[time,space,orbital_number].
    | v_ptrb: np.ndarray, Perturbation potential that this time-dependence was driven by. indexed as v_ptrb[space] if static, and v_ptrb[time,space] if dynamic.
    | t: np.ndarray, Time grid used during evolution.

    In this case, only the occupied time-dependent orbitals are stored, as only these are propigated.
    """

    def __init__(self, initial_state: SingleBodyState):
        self.up = copy.deepcopy(initial_state.up)
        self.down = copy.deepcopy(initial_state.down)
        self.up.td_orbitals = iDEA.utilities.ArrayPlaceholder()
        self.down.td_orbitals = iDEA.utilities.ArrayPlaceholder()
        self.v_ptrb = iDEA.utilities.ArrayPlaceholder()
        self.t = iDEA.utilities.ArrayPlaceholder()


def save_many_body_state(state: ManyBodyState, file_name: str) -> None:
    r"""
    Save a many body state to a system file.

    | Args:
    |     state: iDEA.state.ManyBodyState, State object to save.
    |     file_name: str, file name.
    """
    with open(file_name, "wb") as f:
        pickle.dump(state, f)


def save_single_body_state(state: SingleBodyState, file_name: str) -> None:
    r"""
    Save a single body state to a system file.

    | Args:
    |     state: iDEA.state.SingleBodyState, State object to save.
    |     file_name: str, file name.
    """
    with open(file_name, "wb") as f:
        pickle.dump(state, f)


def load_many_body_state(file_name: str) -> ManyBodyState:
    r"""
    Load a many body state from an system file.

    | Args:
    |     file_name: str, file name.

    | Returns
    |     system: iDEA.state.ManyBodyState, Loaded State object.
    """
    return pickle.load(open(file_name, "rb"))


def load_single_body_state(file_name: str) -> SingleBodyState:
    r"""
    Load a single body state from an system file.

    | Args:
    |     file_name: str, file name.

    | Returns
    |     system: iDEA.state.SingleBodyState, Loaded State object.
    """
    return pickle.load(open(file_name, "rb"))
