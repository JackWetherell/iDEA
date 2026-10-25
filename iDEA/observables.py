import copy
import itertools
import string
from typing import Union

import numpy as np

import iDEA.methods.interacting
import iDEA.methods.non_interacting
import iDEA.state
import iDEA.system


def observable(
    s: iDEA.system.System,
    observable_operator: np.ndarray,
    state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None,
    evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None,
    return_spins: bool = False,
) -> Union[float, np.ndarray]:
    r"""
    Compute an observable based on a given operator and state or evolution.

    | Args:
    |     s: iDEA.system.System, System object.
    |     observable_operator: np.ndarray, Obserbable operator.
    |     state: iDEA.state.SingleBodyState, iDEA.state.ManyBodyState or iDEA.state.QuasiParticleState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
    |     return_spins: bool, True to also return the spin observables: total, up, down. (default = False)

    | Returns:
    |     observable: float or np.ndarray, Observable.
    """
    if state is not None and isinstance(state, iDEA.state.ManyBodyState):
        raise NotImplementedError()

    if state is not None and isinstance(state, iDEA.state.QuasiParticleState):
        p, up_p, down_p = density_matrix(s, state=state, return_spins=True)
        up_O = np.sum(observable_operator * up_p.T) * s.dx
        down_O = np.sum(observable_operator * down_p.T) * s.dx
        O = up_O + down_O
        if return_spins:
            return O, up_O, down_O
        else:
            return O

    if state is not None and isinstance(state, iDEA.state.SingleBodyState):
        up_O = 0.0
        for i in range(state.up.orbitals.shape[1]):
            up_O += (
                np.vdot(
                    state.up.orbitals[:, i],
                    np.dot(observable_operator, state.up.orbitals[:, i]),
                )
                * state.up.occupations[i]
                * s.dx
            )
        down_O = 0.0
        for i in range(state.down.orbitals.shape[1]):
            down_O += (
                np.vdot(
                    state.down.orbitals[:, i],
                    np.dot(observable_operator, state.down.orbitals[:, i]),
                )
                * state.down.occupations[i]
                * s.dx
            )
        O = up_O + down_O
        if return_spins:
            return O, up_O, down_O
        else:
            return O

    if evolution is not None and isinstance(evolution, iDEA.state.ManyBodyEvolution):
        raise NotImplementedError()

    if evolution is not None and isinstance(evolution, iDEA.state.SingleBodyEvolution):
        up_O = np.zeros(shape=evolution.t.shape, dtype=complex)
        for i, I in enumerate(evolution.up.occupied):
            for j, _ti in enumerate(evolution.t):
                up_O[j] += (
                    np.vdot(
                        evolution.up.td_orbitals[j, :, i],
                        np.dot(observable_operator, evolution.up.td_orbitals[j, :, i]),
                    )
                    * evolution.up.occupations[I]
                    * s.dx
                )
        down_O = np.zeros(shape=evolution.t.shape, dtype=complex)
        for i, I in enumerate(evolution.down.occupied):
            for j, _ti in enumerate(evolution.t):
                down_O[j] += (
                    np.vdot(
                        evolution.down.td_orbitals[j, :, i],
                        np.dot(observable_operator, evolution.down.td_orbitals[j, :, i]),
                    )
                    * evolution.down.occupations[I]
                    * s.dx
                )
        O = up_O + down_O
        if return_spins:
            return O.real, up_O.real, down_O.real
        else:
            return O.real

    else:
        raise AttributeError("State or Evolution must be provided.")


def density(
    s: iDEA.system.System,
    state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None,
    evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None,
    time_indices: np.ndarray = None,
    return_spins: bool = False,
) -> np.ndarray:
    r"""
    Compute the charge density of a non_interacting state.

    For a quasiparticle state the density is given by the Green's function in the equal-time limit:

    .. math:: n_\sigma(x) = G_\sigma(x,x;\tau=0^-)

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, iDEA.state.ManyBodyState or iDEA.state.QuasiParticleState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
    |     time_indices: np.ndarray, Time indices to compute observable if given evolution. If None will perform for all time indices. (default = None)
    |     return_spins: bool, True to also return the spin densities: total, up, down. (default = False)

    | Returns:
    |     n: np.ndarray, Charge density, or evolution of charge density.
    """
    if state is not None and isinstance(state, iDEA.state.QuasiParticleState):
        p, up_p, down_p = density_matrix(s, state=state, return_spins=True)
        up_n = np.diag(up_p).copy()
        down_n = np.diag(down_p).copy()
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if state is not None and isinstance(state, iDEA.state.ManyBodyState):
        spin_densities = np.zeros(shape=(s.x.shape[0], 2))
        for i in range(s.x.shape[0]):
            for k in range(2):
                spin_densities[i, k] = np.sum(abs(state.full[i, k, ...]) ** 2) * s.dx ** (s.count - 1) * s.count
        up_n = spin_densities[:, 0]
        down_n = spin_densities[:, 1]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if state is not None and isinstance(state, iDEA.state.SingleBodyState):
        up_n = np.zeros(shape=s.x.shape[0])
        down_n = np.zeros(shape=s.x.shape[0])
        for i in range(state.up.orbitals.shape[1]):
            up_n += abs(state.up.orbitals[:, i]) ** 2 * state.up.occupations[i]
        for i in range(state.down.orbitals.shape[1]):
            down_n += abs(state.down.orbitals[:, i]) ** 2 * state.down.occupations[i]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if evolution is not None and isinstance(evolution, iDEA.state.ManyBodyEvolution):
        if time_indices is None:
            time_indices = np.array(range(evolution.t.shape[0]))
        spin_densities = np.zeros(shape=(time_indices.shape[0], s.x.shape[0], 2))
        for j, ti in enumerate(time_indices):
            l = string.ascii_lowercase[: s.count]
            L = string.ascii_uppercase[: s.count]
            st = l + "," + L + "->" + "".join([i for sub in list(zip(l, L)) for i in sub])
            full = np.einsum(st, evolution.td_space[ti, ...], evolution.spin)
            L = list(zip(list(range(0, s.count * 2, 2)), list(range(1, s.count * 2, 2))))
            perms = itertools.permutations(list(range(s.count)))
            full_copy = copy.deepcopy(full)
            full = np.zeros_like(full_copy)
            for p in perms:
                indices = list(itertools.chain(*[L[e] for e in p]))
                full += iDEA.methods.interacting._permutation_parity(p) * np.moveaxis(
                    full_copy, list(range(s.count * 2)), indices
                )
            full = full / np.sqrt(np.sum(abs(full) ** 2) * s.dx**s.count)
            for i in range(s.x.shape[0]):
                for k in range(2):
                    spin_densities[j, i, k] = np.sum(abs(full[i, k, ...]) ** 2) * s.dx ** (s.count - 1) * s.count
        up_n = spin_densities[:, :, 0]
        down_n = spin_densities[:, :, 1]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if evolution is not None and isinstance(evolution, iDEA.state.SingleBodyEvolution):
        if time_indices is None:
            time_indices = np.array(range(evolution.t.shape[0]))
        up_n = np.zeros(shape=(time_indices.shape[0], s.x.shape[0]))
        for i, I in enumerate(evolution.up.occupied):
            for j, ti in enumerate(time_indices):
                up_n[j, :] += abs(evolution.up.td_orbitals[ti, :, i]) ** 2 * evolution.up.occupations[I]
        down_n = np.zeros(shape=(time_indices.shape[0], s.x.shape[0]))
        for i, I in enumerate(evolution.down.occupied):
            for j, ti in enumerate(time_indices):
                down_n[j, :] += abs(evolution.down.td_orbitals[ti, :, i]) ** 2 * evolution.down.occupations[I]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    else:
        raise AttributeError("State or Evolution must be provided.")


def density_matrix(
    s: iDEA.system.System,
    state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None,
    evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None,
    time_indices: np.ndarray = None,
    return_spins: bool = False,
) -> np.ndarray:
    r"""
    Compute the charge density matrix of a non_interacting state.

    For a quasiparticle state the density matrix is given by the Green's function in the equal-time limit:

    .. math:: \rho_\sigma(x,x') = G_\sigma(x,x';\tau=0^-)

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, iDEA.state.ManyBodyState or iDEA.state.QuasiParticleState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
    |     time_indices: np.ndarray, Time indices to compute observable if given evolution. If None will perform for all time indices. (default = None)
    |     return_spins: bool, True to also return the spin density matrices: total, up, down. (default = False)

    | Returns:
    |     p: np.ndarray, Charge density matrix, or evolution of charge density matrix.
    """
    if state is not None and isinstance(state, iDEA.state.QuasiParticleState):
        up_p = state.grid.interpolate(state.up.G, 0.0)
        down_p = state.grid.interpolate(state.down.G, 0.0)
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if state is not None and isinstance(state, iDEA.state.ManyBodyState):
        tosum = list(range(2, s.count * 2))
        spin_p = (
            np.tensordot(state.full, state.full.conj(), axes=(tosum, tosum)).diagonal(axis1=1, axis2=3)
            * s.dx ** (s.count - 1)
            * s.count
        )
        up_p = spin_p[:, :, 0]
        down_p = spin_p[:, :, 1]
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if state is not None and isinstance(state, iDEA.state.SingleBodyState):
        up_p = np.zeros(shape=s.x.shape * 2)
        down_p = np.zeros(shape=s.x.shape * 2)
        for i in range(state.up.orbitals.shape[1]):
            up_p += (
                np.tensordot(state.up.orbitals[:, i], state.up.orbitals[:, i].conj(), axes=0) * state.up.occupations[i]
            )
        for i in range(state.down.orbitals.shape[1]):
            down_p += (
                np.tensordot(state.down.orbitals[:, i], state.down.orbitals[:, i].conj(), axes=0)
                * state.down.occupations[i]
            )
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if evolution is not None and isinstance(evolution, iDEA.state.ManyBodyEvolution):
        if time_indices is None:
            time_indices = np.array(range(evolution.t.shape[0]))
        tosum = list(range(2, s.count * 2))
        spin_density_matrices = np.zeros(shape=(time_indices.shape[0], s.x.shape[0], s.x.shape[0], 2), dtype=complex)
        for j, ti in enumerate(time_indices):
            l = string.ascii_lowercase[: s.count]
            L = string.ascii_uppercase[: s.count]
            st = l + "," + L + "->" + "".join([i for sub in list(zip(l, L)) for i in sub])
            full = np.einsum(st, evolution.td_space[ti, ...], evolution.spin)
            L = list(zip(list(range(0, s.count * 2, 2)), list(range(1, s.count * 2, 2))))
            perms = itertools.permutations(list(range(s.count)))
            full_copy = copy.deepcopy(full)
            full = np.zeros_like(full_copy)
            for p in perms:
                indices = list(itertools.chain(*[L[e] for e in p]))
                full += iDEA.methods.interacting._permutation_parity(p) * np.moveaxis(
                    full_copy, list(range(s.count * 2)), indices
                )
            full = full / np.sqrt(np.sum(abs(full) ** 2) * s.dx**s.count)
            spin_density_matrices[j, :, :, :] = (
                np.tensordot(full, full.conj(), axes=(tosum, tosum)).diagonal(axis1=1, axis2=3)
                * s.dx ** (s.count - 1)
                * s.count
            )
        up_p = spin_density_matrices[:, :, :, 0]
        down_p = spin_density_matrices[:, :, :, 1]
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if evolution is not None and isinstance(evolution, iDEA.state.SingleBodyEvolution):
        if time_indices is None:
            time_indices = np.array(range(evolution.t.shape[0]))
        up_p = np.zeros(shape=(time_indices.shape[0], s.x.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.up.occupied):
            for j, ti in enumerate(time_indices):
                up_p[j, :] += (
                    np.tensordot(
                        evolution.up.td_orbitals[ti, :, i],
                        evolution.up.td_orbitals[ti, :, i].conj(),
                        axes=0,
                    )
                    * evolution.up.occupations[I]
                )
        down_p = np.zeros(shape=(time_indices.shape[0], s.x.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.down.occupied):
            for j, ti in enumerate(time_indices):
                down_p[j, :] += (
                    np.tensordot(
                        evolution.down.td_orbitals[ti, :, i],
                        evolution.down.td_orbitals[ti, :, i].conj(),
                        axes=0,
                    )
                    * evolution.down.occupations[I]
                )
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    else:
        raise AttributeError("State or Evolution must be provided.")


def kinetic_energy(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState = None,
    evolution: iDEA.state.SingleBodyEvolution = None,
) -> Union[float, np.ndarray]:
    r"""
    Compute the kinetic energy of a non_interacting state.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState or iDEA.state.QuasiParticleState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution, Evolution. (default = None)

    | Returns:
    |     E_k: float or np.ndarray, Kinetic energy, or evolution of kinetic energy.
    """
    if state is not None and isinstance(state, iDEA.state.ManyBodyState):
        K = iDEA.methods.interacting.kinetic_energy_operator(s)
        return observable(s, K, state=state)

    if state is not None and isinstance(state, iDEA.state.QuasiParticleState):
        K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
        return observable(s, K, state=state)

    if state is not None and isinstance(state, iDEA.state.SingleBodyState):
        K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
        return observable(s, K, state=state)

    if evolution is not None and isinstance(evolution, iDEA.state.ManyBodyEvolution):
        K = iDEA.methods.interacting.kinetic_energy_operator(s)
        return observable(s, K, evolution=evolution)

    if evolution is not None and isinstance(evolution, iDEA.state.SingleBodyEvolution):
        K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
        return observable(s, K, evolution=evolution)

    else:
        raise AttributeError("State or Evolution must be provided.")


def external_potential(s: iDEA.system.System) -> np.ndarray:
    r"""
    Compute the external potential.

    | Args:
    |     s: iDEA.system.System, System object.

    | Returns:
    |     v_ext: np.ndarray, External potential of the system.
    """
    return s.v_ext


def external_energy(s: iDEA.system.System, n: np.ndarray, v_ext: np.ndarray) -> Union[float, np.ndarray]:
    r"""
    Compute the external energy from a density and external potential.

    Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density of the system.
    |     v_ext: np.ndarray, External potential of the system.

    Returns:
    |     E_ext: float or np.ndarray, External energy, or evolution of external energy.
    """
    if len(n.shape) == 1:
        E_ext = np.dot(n, v_ext) * s.dx
        return E_ext

    elif len(n.shape) == 2:
        E_ext = np.zeros(shape=n.shape[0])
        for j in range(E_ext.shape[0]):
            E_ext[j] = np.dot(n[j, :], v_ext[:]) * s.dx
        return E_ext

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {n.shape} instead.")


def hartree_potential(s: iDEA.system.System, n: np.ndarray) -> np.ndarray:
    r"""
    Compute the Hartree potential from a density.

    | Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density of the system.

    | Returns:
    |     v_h: np.ndarray, Hartree potential, or evolution of Hartree potential.
    """
    if len(n.shape) == 1:
        v_h = np.dot(n, s.v_int) * s.dx
        return v_h

    elif len(n.shape) == 2:
        v_h = np.zeros_like(n)
        for j in range(v_h.shape[0]):
            v_h[j, :] = np.dot(n[j, :], s.v_int[:, :]) * s.dx
        return v_h

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {n.shape} instead.")


def hartree_energy(s: iDEA.system.System, n: np.ndarray, v_h: np.ndarray) -> Union[float, np.ndarray]:
    r"""
    Compute the Hartree energy from a density and Hartree potential.

    | Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density of the system.
    |     v_h: np.ndarray, Hartree potential of the system.


    | Returns:
    |     E_h: float or np.ndarray, Hartree energy, or evolution of Hartree energy.
    """
    if len(n.shape) == 1:
        E_h = 0.5 * np.dot(n, v_h) * s.dx
        return E_h

    elif len(n.shape) == 2:
        E_h = np.zeros(shape=n.shape[0])
        for j in range(E_h.shape[0]):
            E_h[j] = 0.5 * np.dot(n[j, :], v_h[j, :]) * s.dx
        return E_h

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {n.shape} instead.")


def exchange_potential(s: iDEA.system.System, p: np.ndarray) -> np.ndarray:
    r"""
    Compute the exchange potential from a density matrix.

    | Args:
    |     s: iDEA.system.System, System object.
    |     p: np.ndarray, Density matrix of the system.

    | Returns:
    |     v_x: np.ndarray, Exchange potential, or evolution of exchange potential.
    """
    if len(p.shape) == 2:
        v_x = -p * s.v_int
        return v_x

    elif len(p.shape) == 3:
        v_x = np.zeros_like(p)
        for j in range(v_x.shape[0]):
            v_x[j, :, :] = -p[j, :, :] * s.v_int[:, :]
        return v_x

    else:
        raise AttributeError(f"Expected array of shape 2 or 3, got {p.shape} instead.")


def exchange_energy(s: iDEA.system.System, p: np.ndarray, v_x: np.ndarray) -> Union[float, np.ndarray]:
    r"""
    Compute the exchange energy from a density matrix and exchange potential.

    | Args:
    |     s: iDEA.system.System, System object.
    |     p: np.ndarray, Density matrix of the system.
    |     v_x: np.ndarray, Exchange potential of the system.

    | Returns:
    |     E_x: float or np.ndarray, Exchange energy, or evolution of exchange energy.
    """
    if len(p.shape) == 2:
        E_x = 0.5 * np.tensordot(p, v_x, axes=2) * s.dx * s.dx
        return E_x

    elif len(p.shape) == 3:
        E_x = np.zeros(shape=p.shape[0], dtype=complex)
        for j in range(E_x.shape[0]):
            E_x[j] = 0.5 * np.tensordot(p[j, :, :].T, v_x[j, :, :], axes=2) * s.dx * s.dx
        return E_x.real

    else:
        raise AttributeError(f"Expected array of shape 3 or 3, got {p.shape} instead.")


def single_particle_energy(s: iDEA.system.System, state: iDEA.state.SingleBodyState) -> float:
    r"""
    Compute the single particle energy of a single particle state.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State.

    | Returns:
    |     E: float, Single particle energy.
    """
    return np.sum(state.up.energies[:] * state.up.occupations[:]) + np.sum(
        state.down.energies[:] * state.down.occupations[:]
    )


def _one_axis(**axes) -> tuple:
    r"""
    Check that exactly one of the given axes is provided, and return its name and values.

    | Args:
    |     **axes: np.ndarray or None, Candidate axes.

    | Returns:
    |     name: str, Name of the provided axis.
    |     values: np.ndarray, Values of the provided axis (at least 1D).
    """
    provided = {name: values for name, values in axes.items() if values is not None}
    if len(provided) != 1:
        raise ValueError(f"Exactly one of {', '.join(axes)} must be given, got {list(provided) or 'none'}.")
    name, values = provided.popitem()
    return name, np.atleast_1d(np.asarray(values))


def _green_function_poles(s: iDEA.system.System, state, spin: str, **kwargs) -> tuple:
    r"""
    Get the pole representation of the Green's function of a spin channel.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.GreenFunction, iDEA.state.ManyBodyState or iDEA.state.SingleBodyState, State.
    |     spin: str, Spin channel ('up' or 'down').
    |     **kwargs: Arguments passed to iDEA.methods.interacting.green_function if state is a ManyBodyState.

    | Returns:
    |     channel: iDEA.utilities.Container, Pole energies, orbitals and occupations of the spin channel (and the self-energy, if available).
    |     mu: float, Chemical potential (None if there is no gap).
    """
    if spin not in ("up", "down"):
        raise ValueError(f"spin must be 'up' or 'down', got {spin} instead.")
    if isinstance(state, iDEA.state.ManyBodyState):
        state = iDEA.methods.interacting.green_function(s, state, silent=True, **kwargs)
    if isinstance(state, iDEA.state.GreenFunction):
        return getattr(state, spin), state.mu
    if isinstance(state, iDEA.state.SingleBodyState):
        occupied = np.concatenate([c.energies[c.occupations > 0] for c in (state.up, state.down)])
        empty = np.concatenate([c.energies[c.occupations < 1] for c in (state.up, state.down)])
        mu = None
        if occupied.shape[0] > 0 and empty.shape[0] > 0 and np.min(empty) > np.max(occupied):
            mu = 0.5 * (np.max(occupied) + np.min(empty))
        return getattr(state, spin), mu
    raise TypeError(f"Expected a GreenFunction, ManyBodyState or SingleBodyState, got {type(state)} instead.")


def _pole_sum(orbitals: np.ndarray, factors: np.ndarray) -> np.ndarray:
    r"""
    Sum over poles: A[k] = sum_n orbitals[:,n] factors[k,n] orbitals[:,n]^dagger.

    | Args:
    |     orbitals: np.ndarray, Pole orbitals, indexed as orbitals[space,pole].
    |     factors: np.ndarray, Pole factors, indexed as factors[k,pole].

    | Returns:
    |     A: np.ndarray, Two-point quantity, indexed as A[k,space,space].
    """
    A = np.empty((factors.shape[0],) + (orbitals.shape[0],) * 2, dtype=np.result_type(orbitals, factors))
    for k in range(factors.shape[0]):
        A[k] = (orbitals * factors[k]) @ orbitals.conj().T
    return A


def _pole_factors(
    energies: np.ndarray, particle: np.ndarray, axis: str, values: np.ndarray, eta: float, kind: str, mu: float
) -> np.ndarray:
    r"""
    Compute the factor of each pole at each point of an axis.

    Real time t (time-ordered, retarded or advanced, with t = 0 taken as 0^-), real frequency omega (with
    broadening eta), complex frequency z, imaginary frequency nu (z = mu + i nu) or imaginary time tau (in the
    convention of iDEA.methods.mbpt, with tau = 0 taken as 0^-).

    | Args:
    |     energies: np.ndarray, Pole energies, indexed as energies[pole].
    |     particle: np.ndarray, Weight of each pole on the particle (addition) side, between 0 and 1, indexed as particle[pole].
    |     axis: str, 't', 'omega', 'z', 'nu' or 'tau'.
    |     values: np.ndarray, Points on the axis.
    |     eta: float, Broadening for real frequencies.
    |     kind: str, 'time-ordered', 'retarded' or 'advanced' (for t and omega).
    |     mu: float, Chemical potential (for nu and tau).

    | Returns:
    |     factors: np.ndarray, Pole factors, indexed as factors[point,pole].
    """
    if kind not in ("time-ordered", "retarded", "advanced"):
        raise ValueError(f"kind must be 'time-ordered', 'retarded' or 'advanced', got {kind} instead.")
    if axis in ("nu", "tau") and mu is None:
        raise ValueError("The imaginary axis requires a chemical potential, but this state has no gap.")
    e = energies[None, :]
    if axis == "t":
        t = values[:, None]
        phase = np.exp(-1.0j * e * t)
        forward = np.where(t > 0.0, 1.0, 0.0)
        if kind == "time-ordered":
            return -1.0j * phase * (forward * particle[None, :] - (1.0 - forward) * (1.0 - particle[None, :]))
        if kind == "retarded":
            return -1.0j * phase * forward
        return 1.0j * phase * (1.0 - forward)
    if axis == "omega":
        w = values[:, None]
        if kind == "time-ordered":
            return particle[None, :] / (w - e + 1.0j * eta) + (1.0 - particle[None, :]) / (w - e - 1.0j * eta)
        sign = 1.0 if kind == "retarded" else -1.0
        return 1.0 / (w - e + sign * 1.0j * eta)
    if axis == "z":
        return 1.0 / (values[:, None] - e)
    if axis == "nu":
        return 1.0 / (mu + 1.0j * values[:, None] - e)
    if axis == "tau":
        tau = values[:, None]
        decay = np.exp(-abs(e - mu) * abs(tau))
        return np.where(tau > 0.0, -particle[None, :] * decay, (1.0 - particle[None, :]) * decay)
    raise ValueError(f"Unknown axis {axis}.")


def _energy_filter(energies: np.ndarray, mu: float, cutoff: float) -> np.ndarray:
    r"""
    Smooth energy filter of the poles about the chemical potential (or zero, if there is no gap).

    | Args:
    |     energies: np.ndarray, Pole energies, indexed as energies[pole].
    |     mu: float, Chemical potential (or None).
    |     cutoff: float, Width of the filter.

    | Returns:
    |     weights: np.ndarray, Weights, indexed as weights[pole].
    """
    reference = 0.0 if mu is None else mu
    return np.exp(-0.5 * ((energies - reference) / cutoff) ** 2)


def green_function(
    s: iDEA.system.System,
    state,
    t: np.ndarray = None,
    omega: np.ndarray = None,
    z: np.ndarray = None,
    nu: np.ndarray = None,
    tau: np.ndarray = None,
    eta: float = 0.05,
    kind: str = "time-ordered",
    spin: str = "up",
    cutoff: float = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the one-body Green's function of a spin channel on a time or frequency axis.

    .. math:: G(x,x';t) = -i \langle T \hat{\psi}(x,t) \hat{\psi}^\dagger(x',0) \rangle, \qquad G(x,x';z) = \sum_k \frac{d_k(x) d_k^*(x')}{z - \varepsilon_k}

    Exactly one axis must be given: real time t, real frequency omega (broadened by eta), complex frequency z,
    imaginary frequency nu (z = mu + i nu, as in iDEA.methods.mbpt) or imaginary time tau (G(tau) = -<T psi(tau)
    psi^dagger(0)>, as in iDEA.methods.mbpt). The equal-time limit is taken as t = 0^-, so that G(t=0) = i rho and
    G(tau=0) = rho, the density matrix.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.GreenFunction (exact, from iDEA.methods.interacting.green_function), iDEA.state.ManyBodyState (the exact Green's function is computed), iDEA.state.SingleBodyState (mean-field) or iDEA.state.QuasiParticleState (GW, imaginary axis only).
    |     t: np.ndarray, Real times. (default = None)
    |     omega: np.ndarray, Real frequencies. (default = None)
    |     z: np.ndarray, Complex frequencies. (default = None)
    |     nu: np.ndarray, Imaginary frequencies, measured from the chemical potential. (default = None)
    |     tau: np.ndarray, Imaginary times. (default = None)
    |     eta: float, Broadening of real-frequency quantities. (default = 0.05)
    |     kind: str, 'time-ordered', 'retarded' or 'advanced' (for t and omega). (default = 'time-ordered')
    |     spin: str, Spin channel, 'up' or 'down'. (default = 'up')
    |     cutoff: float, If given, weight each pole by exp(-(e - mu)^2 / (2 cutoff^2)), a smooth energy filter about the chemical potential. This band-limits the result for visualisation (the highest grid energies otherwise dominate real-time plots); the result is then no longer exact. (default = None)
    |     **kwargs: Arguments passed to iDEA.methods.interacting.green_function if state is a ManyBodyState.

    | Returns:
    |     G: np.ndarray, Green's function, indexed as G[point,space,space].
    """
    axis, values = _one_axis(t=t, omega=omega, z=z, nu=nu, tau=tau)
    if isinstance(state, iDEA.state.QuasiParticleState):
        channel = getattr(state, spin)
        if axis == "nu":
            return state.grid.fourier_transform(channel.G, omega=values)
        if axis == "tau":
            return state.grid.interpolate(channel.G, values)
        raise NotImplementedError("A QuasiParticleState is only available on the imaginary axis (nu or tau).")
    channel, mu = _green_function_poles(s, state, spin, **kwargs)
    particle = 1.0 - np.asarray(channel.occupations, dtype=float)
    factors = _pole_factors(channel.energies, particle, axis, values, eta, kind, mu)
    if cutoff is not None:
        factors = factors * _energy_filter(channel.energies, mu, cutoff)[None, :]
    return _pole_sum(channel.orbitals, factors)


def self_energy(
    s: iDEA.system.System,
    state,
    t: np.ndarray = None,
    omega: np.ndarray = None,
    z: np.ndarray = None,
    nu: np.ndarray = None,
    tau: np.ndarray = None,
    eta: float = 0.05,
    kind: str = "time-ordered",
    part: str = "xc",
    method: str = "poles",
    spin: str = "up",
    cutoff: float = None,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the self-energy of a spin channel on a time or frequency axis.

    .. math:: G^{-1}(z) = z - h_0 - v_h - \Sigma_{xc}(z), \qquad \Sigma_{xc}(z) = \Sigma_x + \Sigma_c(z)

    The exchange part is static, so in the time domain it is a delta function, and only the correlation part
    (part = 'c') is available on the t and tau axes. The axes are as in green_function. With method = 'poles' the
    self-energy is evaluated from its pole representation (see iDEA.state.GreenFunction); with method = 'dyson' it
    is found by inverting Dyson's equation at each frequency (complex, real or imaginary frequencies only).

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.GreenFunction, iDEA.state.ManyBodyState (the exact Green's function is computed) or iDEA.state.QuasiParticleState (GW, imaginary axis only).
    |     t, omega, z, nu, tau: np.ndarray, Axis (exactly one must be given; not needed for part = 'x'). (default = None)
    |     eta: float, Broadening of real-frequency quantities. (default = 0.05)
    |     kind: str, 'time-ordered', 'retarded' or 'advanced' (for t and omega). (default = 'time-ordered')
    |     part: str, 'xc' (exchange-correlation), 'c' (correlation) or 'x' (exchange, static). (default = 'xc')
    |     method: str, 'poles' or 'dyson'. (default = 'poles')
    |     spin: str, Spin channel, 'up' or 'down'. (default = 'up')
    |     cutoff: float, If given (method = 'poles' only), weight each correlation pole by exp(-(e - mu)^2 / (2 cutoff^2)), a smooth energy filter for visualisation; the result is then no longer exact. (default = None)
    |     **kwargs: Arguments passed to iDEA.methods.interacting.green_function if state is a ManyBodyState.

    | Returns:
    |     sigma: np.ndarray, Self-energy, indexed as sigma[point,space,space] (or sigma[space,space] for part = 'x').
    """
    if part not in ("xc", "c", "x"):
        raise ValueError(f"part must be 'xc', 'c' or 'x', got {part} instead.")
    if method not in ("poles", "dyson"):
        raise ValueError(f"method must be 'poles' or 'dyson', got {method} instead.")
    if isinstance(state, iDEA.state.ManyBodyState):
        state = iDEA.methods.interacting.green_function(s, state, silent=True, **kwargs)
    if isinstance(state, iDEA.state.SingleBodyState):
        raise TypeError(
            "The self-energy is not defined for a SingleBodyState; use a GreenFunction or QuasiParticleState."
        )
    channel = getattr(state, spin)
    if part == "x":
        return np.array(channel.sigma_x)
    axis, values = _one_axis(t=t, omega=omega, z=z, nu=nu, tau=tau)
    if part == "xc" and axis in ("t", "tau"):
        raise ValueError("The exchange self-energy is a delta function in time; use part = 'c' on the t and tau axes.")
    static = channel.sigma_x if part == "xc" else 0.0

    if isinstance(state, iDEA.state.QuasiParticleState):
        if axis == "nu":
            return state.grid.fourier_transform(channel.sigma_c, omega=values) + static
        if axis == "tau":
            return state.grid.interpolate(channel.sigma_c, values)
        raise NotImplementedError("A QuasiParticleState is only available on the imaginary axis (nu or tau).")
    if not isinstance(state, iDEA.state.GreenFunction):
        raise TypeError(f"Expected a GreenFunction, ManyBodyState or QuasiParticleState, got {type(state)} instead.")

    if method == "dyson":
        if axis in ("t", "tau"):
            raise ValueError("method = 'dyson' is only available on frequency axes (omega, z or nu).")
        if axis == "omega":
            shift = {"time-ordered": 0.0, "retarded": 1.0j * eta, "advanced": -1.0j * eta}[kind]
            frequencies = values + shift
        elif axis == "nu":
            frequencies = state.mu + 1.0j * values
        else:
            frequencies = values
        G = green_function(s, state, eta=eta, kind=kind, spin=spin, **{axis: values})
        h = iDEA.methods.non_interacting.hamiltonian(s)[0] + np.diag(state.v_h)
        identity = np.eye(s.x.shape[0])
        sigma = np.array([(w * identity - h - np.linalg.inv(g * s.dx)) / s.dx for w, g in zip(frequencies, G)])
        return sigma if part == "xc" else sigma - channel.sigma_x

    particle = np.where(channel.sigma_character > 0, 1.0, 0.0)
    if axis == "tau":
        particle = np.where(channel.sigma_energies > state.mu, 1.0, 0.0)
    factors = _pole_factors(channel.sigma_energies, particle, axis, values, eta, kind, state.mu)
    if cutoff is not None:
        factors = factors * _energy_filter(channel.sigma_energies, state.mu, cutoff)[None, :]
    return _pole_sum(channel.sigma_orbitals, factors) + static


def spectral_function(
    s: iDEA.system.System,
    state,
    omega: np.ndarray,
    eta: float = 0.05,
    spin: str = "up",
    diagonal: bool = False,
    **kwargs,
) -> np.ndarray:
    r"""
    Compute the spectral function of a spin channel, broadened by Lorentzians of width eta.

    .. math:: A(x,x';\omega) = \frac{i}{2\pi}[G^R(\omega) - G^{R\dagger}(\omega)] = \sum_k d_k(x) d_k^*(x') \frac{\eta/\pi}{(\omega - \varepsilon_k)^2 + \eta^2}

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.GreenFunction, iDEA.state.ManyBodyState (the exact Green's function is computed) or iDEA.state.SingleBodyState (mean-field).
    |     omega: np.ndarray, Real frequencies.
    |     eta: float, Broadening. (default = 0.05)
    |     spin: str, Spin channel, 'up' or 'down'. (default = 'up')
    |     diagonal: bool, Return only the local spectral function A(x,x;omega). (default = False)
    |     **kwargs: Arguments passed to iDEA.methods.interacting.green_function if state is a ManyBodyState.

    | Returns:
    |     A: np.ndarray, Spectral function, indexed as A[omega,space,space] (or A[omega,space] if diagonal).
    """
    channel, _ = _green_function_poles(s, state, spin, **kwargs)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    lorentzians = (eta / np.pi) / ((omega[:, None] - channel.energies[None, :]) ** 2 + eta**2)
    if diagonal:
        return lorentzians @ (abs(channel.orbitals) ** 2).T
    return _pole_sum(channel.orbitals, lorentzians)


def galitskii_migdal_energy(s: iDEA.system.System, state, **kwargs) -> float:
    r"""
    Compute the total energy from the Green's function with the Galitskii-Migdal formula. In Lehmann form this is
    a sum over the removal poles of each spin channel:

    .. math:: E = \frac{1}{2} \sum_\sigma \sum_{m} \langle d_m | h_0 + \varepsilon_m | d_m \rangle

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.GreenFunction, iDEA.state.ManyBodyState (the exact Green's function is computed) or iDEA.state.QuasiParticleState (GW).
    |     **kwargs: Arguments passed to iDEA.methods.interacting.green_function if state is a ManyBodyState.

    | Returns:
    |     E: float, Total energy.
    """
    if isinstance(state, iDEA.state.QuasiParticleState):
        from iDEA.methods import mbpt

        return mbpt.total_energy(s, state)
    h0 = iDEA.methods.non_interacting.hamiltonian(s)[0]
    E = 0.0
    for spin in ("up", "down"):
        channel, _ = _green_function_poles(s, state, spin, **kwargs)
        removal = channel.occupations > 0
        d = channel.orbitals[:, removal]
        weights = channel.occupations[removal]
        E += 0.5 * s.dx * np.real(np.sum(weights * np.sum(d.conj() * (h0 @ d), axis=0)))
        E += 0.5 * s.dx * np.real(np.sum(weights * channel.energies[removal] * np.sum(abs(d) ** 2, axis=0)))
    return float(E)


def _placeholder(
    s: iDEA.system.System,
    state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None,
    evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None,
    return_spins: bool = False,
) -> Union[float, np.ndarray]:
    r"""
    Placeholer function. Use this as a template when constructing observable methods.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
    |     return_spins: bool, True to also return the spin placeholer: total, up, down. (default = False)

    | Returns:
    |     observable: float or np.ndarray, Placeholer.
    """
    if state is not None and isinstance(state, iDEA.state.SingleBodyState):
        raise NotImplementedError()
    if state is not None and isinstance(state, iDEA.state.ManyBodyState):
        raise NotImplementedError()
    if evolution is not None and isinstance(evolution, iDEA.state.SingleBodyEvolution):
        raise NotImplementedError()
    if evolution is not None and isinstance(evolution, iDEA.state.ManyBodyEvolution):
        raise NotImplementedError()
    else:
        raise AttributeError("State or Evolution must be provided.")
