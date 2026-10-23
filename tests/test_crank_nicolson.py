"""
iDEA Crank-Nicolson propagation testing platform.

Tests iDEA.methods.interacting.propagate(method="cn") on the CPU against the
matrix exponential propagator (method="expm"). Crank-Nicolson is second order in
dt, so the two do not agree to machine precision: these tests check that the
difference is small, that it falls by 4x when dt is halved (for static and
time-dependent perturbations, and for two and three electrons), that the
propagation is unitary, and that the many-body perturbation diagonal it builds by
broadcasting matches the kron-built operator.

The GPU (cuDSS) path is checked against this CPU path in tests/test_gpu.py.

Systems are kept small (at most a few thousand basis states) so the module runs
in seconds.
"""

import numpy as np
import pytest

import iDEA.interactions
import iDEA.methods.interacting
import iDEA.observables
import iDEA.system

T_LENGTH = 1.0


def make_system(N: int = 40, spin: str = "uu") -> iDEA.system.System:
    """Build a small softened-interaction QHO system on an N-point grid."""
    x = np.linspace(-10, 10, N)
    v_ext = 0.5 * (0.25**2) * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons=spin)


def make_perturbation(s: iDEA.system.System, t: np.ndarray, kind: str) -> np.ndarray:
    """A uniform field that is either switched on suddenly ("static") or oscillates in time ("td")."""
    if kind == "static":
        return np.tile(-0.1 * s.x, (t.shape[0], 1))
    return -0.1 * np.sin(2.0 * t)[:, None] * s.x[None, :]


def norms(s: iDEA.system.System, td_space: np.ndarray) -> np.ndarray:
    """Norm of the many-body wavefunction at each time index."""
    return np.array([np.sum(abs(td_space[j, ...]) ** 2) * s.dx**s.count for j in range(td_space.shape[0])])


def cn_error(s, state, steps: int, kind: str) -> float:
    """Largest difference between the CN and expm wavefunctions over all time indices."""
    t = np.linspace(0, T_LENGTH, steps + 1)
    v_ptrb = make_perturbation(s, t, kind)
    expm = iDEA.methods.interacting.propagate(s, state, v_ptrb, t, method="expm")
    cn = iDEA.methods.interacting.propagate(s, state, v_ptrb, t, method="cn")
    return float(np.max(abs(expm.td_space - cn.td_space)))


@pytest.fixture(scope="module")
def two_electron():
    s = make_system(N=40, spin="uu")
    return s, iDEA.methods.interacting.solve(s, k=0)


@pytest.fixture(scope="module")
def three_electron():
    s = make_system(N=12, spin="uud")
    return s, iDEA.methods.interacting.solve(s, k=0)


class TestPerturbationDiagonal:
    """The broadcast diagonal must equal the many-body operator built from kron products."""

    @pytest.mark.parametrize("spin", ["u", "uu", "uud"])
    def test_matches_kron(self, spin):
        s = make_system(N=8, spin=spin)
        V = iDEA.methods.interacting.external_potential_operator(s)
        diagonal = iDEA.methods.interacting._perturbation_diagonal(np, s.v_ext, s.count)
        assert np.allclose(diagonal, V.diagonal(), rtol=1e-14, atol=1e-14)


class TestCrankNicolson:
    """iDEA.methods.interacting.propagate(method="cn") against method="expm"."""

    @pytest.mark.parametrize("kind", ["static", "td"])
    def test_conserves_norm(self, two_electron, kind):
        s, state = two_electron
        t = np.linspace(0, T_LENGTH, 21)
        evolution = iDEA.methods.interacting.propagate(s, state, make_perturbation(s, t, kind), t, method="cn")
        assert np.allclose(norms(s, evolution.td_space), 1.0, rtol=1e-12, atol=1e-12)

    def test_initial_state(self, two_electron):
        s, state = two_electron
        t = np.linspace(0, T_LENGTH, 11)
        evolution = iDEA.methods.interacting.propagate(s, state, make_perturbation(s, t, "static"), t, method="cn")
        assert np.array_equal(evolution.td_space[0, ...], state.space)

    def test_agrees_with_expm(self, two_electron):
        """At dt = 0.025 the two differ by ~5e-6 in the wavefunction and ~1e-6 in the density."""
        s, state = two_electron
        t = np.linspace(0, T_LENGTH, 41)
        v_ptrb = make_perturbation(s, t, "static")
        expm = iDEA.methods.interacting.propagate(s, state, v_ptrb, t, method="expm")
        cn = iDEA.methods.interacting.propagate(s, state, v_ptrb, t, method="cn")
        assert np.allclose(cn.td_space, expm.td_space, rtol=0, atol=2e-5)
        n_expm = iDEA.observables.density(s, evolution=expm)
        n_cn = iDEA.observables.density(s, evolution=cn)
        assert np.allclose(n_cn, n_expm, rtol=0, atol=5e-6)

    @pytest.mark.parametrize("system", ["two_electron", "three_electron"])
    @pytest.mark.parametrize("kind", ["static", "td"])
    def test_second_order(self, request, system, kind):
        """Halving dt must reduce the difference from expm by 4x."""
        s, state = request.getfixturevalue(system)
        ratio = cn_error(s, state, 20, kind) / cn_error(s, state, 40, kind)
        assert 3.8 < ratio < 4.2, f"error ratio {ratio:.3f} is not second order"

    def test_invalid_method(self, two_electron):
        s, state = two_electron
        t = np.linspace(0, T_LENGTH, 3)
        with pytest.raises(ValueError):
            iDEA.methods.interacting.propagate(s, state, make_perturbation(s, t, "static"), t, method="rk4")
