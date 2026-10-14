"""
Tests for the approximate (single-body) methods and reverse engineering.

These complement the analytic many-body tests in test_manybody.py by checking that:
1. The Hartree total energy removes the Hartree double counting exactly once, E = sum(eps) - E_H.
2. Every single-body method converges on a small two-electron atom and conserves particle number.
3. Reverse engineering a non-interacting density recovers the original external potential.
"""

import numpy as np
import pytest

import iDEA


@pytest.fixture(scope="module")
def atom():
    x = np.linspace(-10, 10, 101)
    v_ext = -2.0 / (np.abs(x) + 1.0)
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons="ud")


def test_hartree_total_energy(atom):
    state = iDEA.methods.hartree.solve(atom, tol=1e-10, silent=True)
    n = iDEA.observables.density(atom, state)
    v_h = iDEA.observables.hartree_potential(atom, n)
    E_sp = iDEA.observables.single_particle_energy(atom, state)
    E_h = iDEA.observables.hartree_energy(atom, n, v_h)
    E_kin_ext = E_sp - 2.0 * E_h  # sum(eps) = T + V_ext + 2 E_H
    assert iDEA.methods.hartree.total_energy(atom, state) == pytest.approx(E_sp - E_h, abs=1e-12)
    assert iDEA.methods.hartree.total_energy(atom, state) == pytest.approx(E_kin_ext + E_h, abs=1e-12)


@pytest.mark.parametrize("method", iDEA.iterate_sb_methods, ids=lambda m: m.name)
def test_single_body_methods_converge(atom, method):
    state = method.solve(atom, tol=1e-8, silent=True)
    n = iDEA.observables.density(atom, state)
    assert np.sum(n) * atom.dx == pytest.approx(atom.count, abs=1e-8)
    assert np.isfinite(method.total_energy(atom, state))


def test_hartree_fock_two_electron_singlet(atom):
    # For a closed-shell two-electron singlet, exchange cancels exactly half the Hartree energy.
    state = iDEA.methods.hartree_fock.solve(atom, restricted=True, tol=1e-10, silent=True)
    n = iDEA.observables.density(atom, state)
    p, up_p, down_p = iDEA.observables.density_matrix(atom, state, return_spins=True)
    E_h = iDEA.observables.hartree_energy(atom, n, iDEA.observables.hartree_potential(atom, n))
    E_x = iDEA.observables.exchange_energy(atom, up_p, iDEA.observables.exchange_potential(atom, up_p))
    E_x += iDEA.observables.exchange_energy(atom, down_p, iDEA.observables.exchange_potential(atom, down_p))
    assert E_x == pytest.approx(-0.5 * E_h, rel=1e-8)


def test_reverse_engineering_round_trip(atom):
    state = iDEA.methods.non_interacting.solve(atom, silent=True)
    n = iDEA.observables.density(atom, state)
    s_fict = iDEA.reverse_engineering.reverse(
        atom, n, iDEA.methods.non_interacting, v_guess=0.5 * atom.v_ext, mu=5.0, pe=0.1, tol=1e-6, silent=True
    )
    # The potential is defined up to a constant; compare where the density is non-negligible.
    mask = n > 1e-3
    dv = s_fict.v_ext[mask] - atom.v_ext[mask]
    assert np.max(np.abs(dv - np.mean(dv))) < 1e-2
