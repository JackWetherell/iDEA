"""
iDEA many-body perturbation theory (GW) testing platform.

Tests iDEA.methods.mbpt, which solves Hedin's equations in the GW approximation using the space-time method, against
results computed independently of the imaginary time and frequency grids:

- The imaginary time/frequency Fourier transforms are tested on random superpositions of decaying exponentials.
- One-shot G0W0 is tested against a sum-over-states (Lehmann) construction, in which the screened interaction is
  built from the eigenstates of the random phase approximation (Casida's equations), and the density is found by
  brute-force quadrature of the Green's function over imaginary frequency.
- The Galitskii-Migdal correlation energy is tested against its closed-form sum-over-states expression.
- Self-consistent GW is tested for conservation of particle number, and independence of the starting point.

The systems are deliberately small (a few tens of grid points) so that the whole module runs in about a minute.
"""

import numpy as np
import pytest
import scipy.linalg as spla

import iDEA

GRID_TOL = 1e-10


def make_atom(electrons: str = "ud", points: int = 41, length: float = 16.0, Z: float = 2.0) -> iDEA.system.System:
    """Build a small softened-interaction atom."""
    x = np.linspace(-0.5 * length, 0.5 * length, points)
    v_ext = -Z / (abs(x) + 1.0)
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons)


def lehmann_g0w0(s: iDEA.system.System, state: iDEA.state.SingleBodyState, mu: float) -> dict:
    r"""
    Construct G0, P0, W_c and Sigma_c for a closed-shell starting point using sums over states, with the screened
    interaction built from the eigenstates of the random phase approximation (Casida's equations).
    """
    e = state.up.energies - mu
    phi = state.up.orbitals
    occupied = e < 0.0
    pairs = [(i, a) for i in np.nonzero(occupied)[0] for a in np.nonzero(~occupied)[0]]
    rho = np.array([phi[:, i] * phi[:, a] for i, a in pairs]).T
    D = np.array([e[a] - e[i] for i, a in pairs])
    K = 2.0 * rho.T @ s.v_int @ rho * s.dx**2
    omega2, Z = spla.eigh(np.diag(np.sqrt(D)) @ (np.diag(D) + 2.0 * K) @ np.diag(np.sqrt(D)))
    Omega = np.sqrt(omega2)
    rho_s = np.sqrt(2.0) * rho @ (np.diag(np.sqrt(D)) @ Z / np.sqrt(Omega)[None, :])
    w = s.v_int @ rho_s * s.dx

    def P0(omega):
        return np.einsum("ip,kp,jp->kij", rho, -4.0 * D[None, :] / (omega[:, None] ** 2 + D[None, :] ** 2), rho)

    def W_c(omega):
        return np.einsum("is,ks,js->kij", w, -2.0 * Omega[None, :] / (omega[:, None] ** 2 + Omega[None, :] ** 2), w)

    def sigma_c(omega):
        out = np.zeros((omega.shape[0],) + s.v_int.shape, dtype=complex)
        for n in range(e.shape[0]):
            shift = Omega if occupied[n] else -Omega
            out += np.einsum(
                "i,j,is,ks,js->kij", phi[:, n], phi[:, n], w, 1.0 / (1.0j * omega[:, None] - e[n] + shift), w
            )
        return out

    return {"P0": P0, "W_c": W_c, "sigma_c": sigma_c, "rho": rho, "D": D, "Omega": Omega, "w": w}


@pytest.fixture(scope="module")
def atom():
    return make_atom()


@pytest.fixture(scope="module")
def atom_hf(atom):
    return iDEA.methods.hartree_fock.solve(atom, silent=True)


@pytest.fixture(scope="module")
def atom_gw(atom):
    return iDEA.methods.mbpt.solve(atom, mode="GW", silent=True)


@pytest.fixture(scope="module")
def atom_exact(atom):
    # The ground state of two opposite-spin electrons is the lowest eigenstate of the spatial Hamiltonian (a singlet).
    # A dense eigensolver is used, so that the result is deterministic and the random state of the iterative
    # eigensolver used by other tests is left untouched.
    energies, wavefunctions = spla.eigh(iDEA.methods.interacting.hamiltonian(atom).toarray(), subset_by_index=[0, 0])
    wavefunction = wavefunctions[:, 0].reshape(atom.x.shape * 2)
    density = 2.0 * np.sum(wavefunction**2, axis=1) / np.sum(wavefunction**2) / atom.dx
    return energies[0], density


class TestSpaceTimeGrid:
    r"""Tests of the imaginary time and frequency grids and Fourier transforms."""

    grid = iDEA.methods.mbpt.SpaceTimeGrid(0.05, 500.0, GRID_TOL)
    rng = np.random.default_rng(0)

    def random_function(self, number: int = 200):
        """Random two-sided superposition of decaying exponentials, f(t) = sum_n c_n exp(-e_n |t|) (separately for t > 0 and t < 0)."""
        rates = np.exp(self.rng.uniform(np.log(0.05), np.log(500.0), (2, number)))
        coefficients = self.rng.normal(size=(2, number))
        return rates, coefficients

    def evaluate(self, rates, coefficients, tau=None, omega=None):
        """Exact values of a random function in imaginary time, or its Fourier transform in imaginary frequency."""
        (rates_plus, rates_minus), (c_plus, c_minus) = rates, coefficients
        if tau is not None:
            f_plus = np.exp(-np.outer(abs(tau), rates_plus)) @ c_plus
            f_minus = np.exp(-np.outer(abs(tau), rates_minus)) @ c_minus
            return np.where(tau > 0, f_plus, f_minus)[:, None]
        F = (c_plus / (rates_plus - 1.0j * omega[:, None])).sum(axis=1)
        F += (c_minus / (rates_minus + 1.0j * omega[:, None])).sum(axis=1)
        return F[:, None]

    def test_grid_size(self):
        assert self.grid.tau.shape[0] < 150
        assert self.grid.omega.shape[0] < 150
        assert self.grid.omega[0] == 0.0
        assert np.all(np.diff(self.grid.tau) > 0)
        assert np.allclose(self.grid.tau, -self.grid.tau[::-1])

    def test_fourier_transform(self):
        omega = np.array([0.0, 0.3, 7.0, 1e4])
        for _ in range(10):
            rates, coefficients = self.random_function()
            f = self.evaluate(rates, coefficients, tau=self.grid.tau)
            F = self.evaluate(rates, coefficients, omega=self.grid.omega)
            assert np.max(abs(self.grid.fourier_transform(f) - F)) < 1e-8 * np.max(abs(F))
            F = self.evaluate(rates, coefficients, omega=omega)
            assert np.max(abs(self.grid.fourier_transform(f, omega=omega) - F)) < 1e-8 * np.max(abs(F))

    def test_inverse_fourier_transform(self):
        tau = np.array([-30.0, -0.01, 0.0, 1e-6, 0.2, 5.0])
        for _ in range(10):
            rates, coefficients = self.random_function()
            f = self.evaluate(rates, coefficients, tau=self.grid.tau)
            F = self.evaluate(rates, coefficients, omega=self.grid.omega)
            assert np.max(abs(self.grid.inverse_fourier_transform(F) - f)) < 1e-7 * np.max(abs(f))
            f = self.evaluate(rates, coefficients, tau=tau)
            assert np.max(abs(self.grid.inverse_fourier_transform(F, tau=tau) - f)) < 1e-7 * np.max(abs(f))

    def test_interpolate_and_integrate(self):
        tau = np.array([-30.0, -0.01, 0.0, 1e-6, 0.2, 5.0])
        for _ in range(10):
            rates, coefficients = self.random_function()
            f = self.evaluate(rates, coefficients, tau=self.grid.tau)
            scale = np.max(abs(f))
            assert abs(self.grid.interpolate(f, 0.0)[0] - coefficients[1].sum()) < 1e-10 * scale
            assert (
                np.max(abs(self.grid.interpolate(f, tau) - self.evaluate(rates, coefficients, tau=tau))) < 1e-9 * scale
            )
            integral = np.sum(coefficients / rates)
            assert abs(self.grid.integrate(f)[0] - integral) < 1e-9 * np.sum(abs(coefficients / rates))

    def test_green_function(self, atom, atom_hf):
        mu = 0.5 * (atom_hf.up.energies[0] + atom_hf.up.energies[1])
        e, phi = atom_hf.up.energies, atom_hf.up.orbitals
        G, G_iw = iDEA.methods.mbpt.non_interacting_green_function(atom, self.grid, e, phi, mu)
        p = iDEA.observables.density_matrix(atom, atom_hf, return_spins=True)[1]
        assert np.allclose(self.grid.interpolate(G, 0.0), p, atol=1e-10)
        assert np.allclose(
            self.grid.interpolate(G, np.array([1e-14]))[0], p - np.eye(atom.x.shape[0]) / atom.dx, atol=1e-8
        )
        assert np.max(abs(self.grid.fourier_transform(G) - G_iw)) < 1e-8 * np.max(abs(G_iw))
        assert np.max(abs(self.grid.inverse_fourier_transform(G_iw) - G)) < 1e-7 * np.max(abs(G))


class TestNonInteractingLimit:
    r"""Without interactions GW must reduce to the non-interacting solution."""

    @pytest.mark.parametrize("mode", ["G0W0", "GW"])
    def test_non_interacting(self, mode):
        s = make_atom("ud")
        s.v_int = 0.0 * s.v_int
        state = iDEA.methods.mbpt.solve(s, mode=mode, initial="non_interacting", silent=True)
        reference = iDEA.methods.non_interacting.solve(s, silent=True)
        assert np.allclose(iDEA.observables.density(s, state), iDEA.observables.density(s, reference), atol=1e-9)
        assert state.energy == pytest.approx(iDEA.methods.non_interacting.total_energy(s, reference), abs=1e-9)
        assert np.max(abs(state.up.sigma_iw)) < 1e-12
        assert np.allclose(state.up.G, state.up.G0, atol=1e-8)


@pytest.fixture(scope="module")
def g0w0(atom):
    return iDEA.methods.mbpt.solve(atom, mode="G0W0", initial="hartree_fock", hedin_shift=False, silent=True)


@pytest.fixture(scope="module")
def reference(atom, atom_hf, g0w0):
    return lehmann_g0w0(atom, atom_hf, g0w0.up.mu0)


class TestG0W0:
    r"""Tests of one-shot G0W0 against sums over states."""

    def test_polarisability(self, g0w0, reference):
        P0 = reference["P0"](g0w0.omega)
        assert np.max(abs(g0w0.P_iw - P0)) < 1e-8 * np.max(abs(P0))
        assert np.max(abs(g0w0.P - g0w0.grid.inverse_fourier_transform(P0, even=True))) < 1e-7 * np.max(abs(g0w0.P))

    def test_screened_interaction(self, atom, g0w0, reference):
        W_c = reference["W_c"](g0w0.omega)
        assert np.max(abs(g0w0.W_c_iw - W_c)) < 1e-8 * np.max(abs(W_c))
        assert np.allclose(g0w0.W_iw - g0w0.W_c_iw, atom.v_int[None, :, :], atol=1e-12)

    def test_self_energy(self, atom, atom_hf, g0w0, reference):
        sigma_c = reference["sigma_c"](g0w0.omega)
        assert np.max(abs(g0w0.up.sigma_c_iw - sigma_c)) < 1e-8 * np.max(abs(sigma_c))
        p = iDEA.observables.density_matrix(atom, atom_hf, return_spins=True)[1]
        assert np.allclose(g0w0.up.sigma_x, -p * atom.v_int, atol=1e-10)
        assert np.allclose(g0w0.up.sigma_xc_iw, g0w0.up.sigma_x[None, :, :] + g0w0.up.sigma_c_iw)
        v_h = iDEA.observables.hartree_potential(atom, iDEA.observables.density(atom, atom_hf))
        assert np.allclose(g0w0.v_h, v_h, atol=1e-10)

    def test_density(self, atom, atom_hf, g0w0, reference):
        # Brute-force quadrature of G(iw) - G0(iw) over imaginary frequency with the sum-over-states self-energy.
        mu = g0w0.up.mu
        H = iDEA.methods.mbpt.kinetic_energy_operator(atom) + iDEA.methods.mbpt.external_potential_operator(atom)
        H += np.diag(g0w0.v_h) + g0w0.up.sigma_x * atom.dx
        nodes, weights = np.polynomial.legendre.leggauss(300)
        theta = 0.25 * np.pi * (nodes + 1.0)
        omega = np.tan(theta)
        jacobian = 0.25 * np.pi * weights / np.cos(theta) ** 2
        identity = np.eye(atom.x.shape[0])
        A = (1.0j * omega + mu)[:, None, None] * identity - H[None, :, :]
        G = np.linalg.inv(A - reference["sigma_c"](omega) * atom.dx)
        G0 = np.linalg.inv(A)
        p_hf = iDEA.observables.density_matrix(atom, atom_hf, return_spins=True)[1]
        p_reference = p_hf + np.tensordot(jacobian, (G - G0).real, axes=(0, 0)) / (np.pi * atom.dx)
        n, up_n, down_n = iDEA.observables.density(atom, g0w0, return_spins=True)
        assert np.allclose(up_n, np.diag(p_reference), atol=1e-9)
        assert np.allclose(up_n, down_n)
        # G0W0 does not conserve particle number.
        assert abs(np.sum(n) * atom.dx - 2.0) > 1e-6

    def test_galitskii_migdal_correlation_energy(self, atom, g0w0, reference):
        # 1/2 sum_s int dt Tr[Sigma_c[G0](t) G0(-t)] = -1/2 int dw/2pi Tr[W_c(iw) P0(iw)], in closed form.
        overlap = reference["w"].T @ reference["rho"] * atom.dx
        D, Omega = reference["D"], reference["Omega"]
        E_c_reference = -0.5 * np.sum(2.0 * overlap**2 * 2.0 / (Omega[:, None] + D[None, :]))
        sigma_c, _ = iDEA.methods.mbpt.correlation_self_energy(atom, g0w0.grid, g0w0.up.G0, g0w0.W_c)
        sigma_c_plus, sigma_c_minus = g0w0.grid.split(sigma_c)
        G0_plus, G0_minus = g0w0.grid.split(g0w0.up.G0)
        trace = g0w0.grid.join(
            np.einsum("tij,tji->t", sigma_c_plus, G0_minus), np.einsum("tij,tji->t", sigma_c_minus, G0_plus)
        )
        E_c = 2.0 * 0.5 * g0w0.grid.integrate(trace) * atom.dx**2
        assert E_c == pytest.approx(E_c_reference, rel=1e-8)
        assert E_c < 0.0


class TestSelfConsistentGW:
    r"""Tests of partially and fully self-consistent GW."""

    def test_particle_number(self, atom, atom_gw):
        n, up_n, down_n = iDEA.observables.density(atom, atom_gw, return_spins=True)
        assert np.sum(up_n) * atom.dx == pytest.approx(1.0, abs=1e-8)
        assert np.sum(down_n) * atom.dx == pytest.approx(1.0, abs=1e-8)
        state = iDEA.methods.mbpt.solve(atom, mode="GW0", silent=True)
        assert np.sum(iDEA.observables.density(atom, state)) * atom.dx == pytest.approx(2.0, abs=1e-8)

    @pytest.mark.parametrize("initial", ["non_interacting", "hartree", "hartree_state"])
    def test_starting_point_independence(self, atom, atom_gw, initial):
        if initial == "hartree_state":
            initial = iDEA.methods.hartree.solve(atom, silent=True)
        state = iDEA.methods.mbpt.solve(atom, mode="GW", initial=initial, tol=1e-10, silent=True)
        assert np.allclose(iDEA.observables.density(atom, state), iDEA.observables.density(atom, atom_gw), atol=1e-7)
        assert state.energy == pytest.approx(atom_gw.energy, abs=1e-7)

    def test_hedin_shift_does_not_change_self_consistent_result(self, atom, atom_gw):
        state = iDEA.methods.mbpt.solve(atom, mode="GW", hedin_shift=False, silent=True)
        assert np.allclose(iDEA.observables.density(atom, state), iDEA.observables.density(atom, atom_gw), atol=1e-7)
        assert state.energy == pytest.approx(atom_gw.energy, abs=1e-7)

    def test_self_consistency(self, atom, atom_gw):
        # The converged G satisfies Dyson's equation with the self-energy built from itself.
        state = atom_gw
        H = iDEA.methods.mbpt.kinetic_energy_operator(atom) + iDEA.methods.mbpt.external_potential_operator(atom)
        p = iDEA.observables.density_matrix(atom, state, return_spins=True)[1]
        assert np.allclose(state.up.sigma_x, -p * atom.v_int, atol=1e-8)
        P, P_iw = iDEA.methods.mbpt.polarisability(atom, state.grid, state.up.G, state.down.G)
        assert np.allclose(P_iw, state.P_iw, atol=1e-7)
        identity = np.eye(atom.x.shape[0])
        A = (1.0j * state.omega + state.up.mu)[:, None, None] * identity - H[None, :, :] - state.up.sigma_iw * atom.dx
        assert np.allclose(np.linalg.inv(A) / atom.dx, state.up.G_iw, atol=1e-10)

    def test_accuracy(self, atom, atom_hf, atom_gw, atom_exact):
        E_exact, n_exact = atom_exact
        n_gw = iDEA.observables.density(atom, atom_gw)
        n_hf = iDEA.observables.density(atom, atom_hf)
        assert np.sum(abs(n_gw - n_exact)) < np.sum(abs(n_hf - n_exact))
        # GW over-correlates, but is much closer to the exact energy than Hartree-Fock.
        E_hf = iDEA.methods.hartree_fock.total_energy(atom, atom_hf)
        assert atom_gw.energy < E_exact < E_hf
        assert abs(atom_gw.energy - E_exact) < 0.5 * abs(E_hf - E_exact)
        assert atom_gw.energy == pytest.approx(iDEA.methods.mbpt.total_energy(atom, atom_gw))

    def test_quasiparticle_energies(self, atom, atom_gw):
        # The static quasiparticle energies straddle the chemical potential.
        assert atom_gw.up.energies[0] < atom_gw.up.mu < atom_gw.up.energies[1]
        assert np.array_equal(atom_gw.up.occupied, [0])
        assert np.sum(atom_gw.up.orbitals[:, 0] ** 2) * atom.dx == pytest.approx(1.0)


class TestSpin:
    r"""Tests of spin-polarised and open-shell systems."""

    def test_spin_polarised(self):
        s = make_atom("uu")
        state = iDEA.methods.mbpt.solve(s, mode="GW", silent=True)
        n, up_n, down_n = iDEA.observables.density(s, state, return_spins=True)
        assert np.sum(up_n) * s.dx == pytest.approx(2.0, abs=1e-8)
        assert np.max(abs(down_n)) < 1e-10
        assert np.max(abs(state.grid.split(state.down.G)[1])) < 1e-9
        assert state.up.mu != state.down.mu

    def test_open_shell(self):
        s = make_atom("uud", Z=3.0)
        state = iDEA.methods.mbpt.solve(s, mode="GW", silent=True)
        n, up_n, down_n = iDEA.observables.density(s, state, return_spins=True)
        assert np.sum(up_n) * s.dx == pytest.approx(2.0, abs=1e-8)
        assert np.sum(down_n) * s.dx == pytest.approx(1.0, abs=1e-8)
        assert state.down.G is not state.up.G


class TestInterface:
    r"""Tests of the interface and observables."""

    def test_state(self, atom, atom_gw):
        state = atom_gw
        assert isinstance(state, iDEA.state.QuasiParticleState)
        assert state.mode == "GW"
        L, K, N = state.tau.shape[0], state.omega.shape[0], atom.x.shape[0]
        for array in (state.up.G, state.up.G0, state.up.sigma_c, state.P, state.W_c):
            assert array.shape == (L, N, N)
            assert np.isrealobj(array)
        for array in (state.up.G_iw, state.up.G0_iw, state.up.sigma_c_iw, state.up.sigma_xc_iw, state.up.sigma_iw):
            assert array.shape == (K, N, N)
        for array in (state.P_iw, state.W_c_iw, state.W_iw):
            assert array.shape == (K, N, N)
        assert np.allclose(state.P, state.P[::-1])
        assert np.allclose(state.W_c, state.W_c[::-1])

    def test_observables(self, atom, atom_gw):
        p, up_p, down_p = iDEA.observables.density_matrix(atom, atom_gw, return_spins=True)
        n = iDEA.observables.density(atom, atom_gw)
        assert np.allclose(np.diag(p), n)
        assert np.allclose(up_p, up_p.T, atol=1e-10)
        K = iDEA.methods.mbpt.kinetic_energy_operator(atom)
        assert iDEA.observables.kinetic_energy(atom, atom_gw) == pytest.approx(np.sum(K * p.T) * atom.dx)
        assert iDEA.observables.observable(atom, np.eye(atom.x.shape[0]), state=atom_gw) == pytest.approx(2.0, abs=1e-8)

    def test_invalid_arguments(self, atom):
        with pytest.raises(ValueError):
            iDEA.methods.mbpt.solve(atom, mode="GW2", silent=True)
        with pytest.raises(ValueError):
            iDEA.methods.mbpt.solve(atom, initial="kssce", silent=True)
        # Without the Hedin shift, the chemical potential of a non-interacting starting point lies outside the
        # quasiparticle gap.
        with pytest.raises(RuntimeError):
            iDEA.methods.mbpt.solve(atom, mode="G0W0", initial="non_interacting", hedin_shift=False, silent=True)
