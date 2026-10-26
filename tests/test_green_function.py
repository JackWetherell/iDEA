"""
Tests for the exact one-body Green's function and self-energy (iDEA.methods.interacting.green_function), their
evaluation (iDEA.observables), and the plotting of two-point quantities (iDEA.plotting).

The exact Green's function is built from the eigenstates of the (N-1) and (N+1)-electron systems, so it must satisfy
a set of exact identities on any grid: completeness of the Dyson orbitals, the density matrix as the removal part,
the first moment of the spectral function (the static self-energy is the Fock operator of the exact density matrix),
and the Galitskii-Migdal energy. The tests use small grids so that every sector can also be diagonalised completely.
"""

import numpy as np
import pytest

import iDEA

GPU_AVAILABLE = True
try:
    import cupy  # noqa: F401
except ImportError:
    GPU_AVAILABLE = False


def make_system(points: int, length: float, electrons: str, Z: float, interacting: bool = True) -> iDEA.system.System:
    x = np.linspace(-length, length, points)
    v_int = iDEA.interactions.softened_interaction(x)
    if not interacting:
        v_int = np.zeros_like(v_int)
    return iDEA.system.System(x, -Z / (abs(x) + 1.0), v_int, electrons)


def solve(s: iDEA.system.System, **kwargs) -> tuple:
    state = iDEA.methods.interacting.solve(s, k=0)
    return state, iDEA.methods.interacting.green_function(s, state, silent=True, **kwargs)


@pytest.fixture(scope="module")
def hydrogen():
    s = make_system(15, 7.0, "u", 1.0)
    return (s,) + solve(s)


@pytest.fixture(scope="module")
def helium():
    s = make_system(11, 5.0, "ud", 2.0)
    return (s,) + solve(s)


@pytest.fixture(scope="module")
def triplet():
    s = make_system(11, 6.0, "uu", 2.0)
    return (s,) + solve(s)


@pytest.fixture(scope="module")
def lithium():
    s = make_system(9, 5.0, "uud", 3.0)
    return (s,) + solve(s)


Z = np.array([0.3 + 0.4j, -1.0 + 0.2j, 2.0 + 1.0j, -0.5 - 0.7j])


class TestSumRules:
    @pytest.mark.parametrize("name", ["hydrogen", "helium", "triplet", "lithium"])
    def test_sum_rules(self, name, request):
        s, state, green = request.getfixturevalue(name)
        info = green.info
        assert info["rayleigh_residual"] < 1e-10
        assert info["galitskii_migdal_error"] < 1e-10
        for spin in ("up", "down"):
            assert info[f"{spin}_completeness_error"] < 1e-10
            assert info[f"{spin}_first_moment_error"] < 1e-9
        assert abs(green.energy - state.energy) < 1e-8

    def test_field_operators(self, helium):
        s, state, _ = helium
        interacting = iDEA.methods.interacting
        psi = interacting.spin_ordered_wavefunction(s, state)
        up, down, up_index, down_index = interacting._determinants(s.x.shape[0], 1, 1)
        X = psi[tuple(np.concatenate((up[up_index], down[down_index]), axis=1).T)]
        _, up_p, _ = iDEA.observables.density_matrix(s, state=state, return_spins=True)
        removal = interacting._field_sources(s.x.shape[0], X, 1, 1, "up", create=False)
        addition = interacting._field_sources(s.x.shape[0], X, 1, 1, "up", create=True)
        # <c_i Psi|c_j Psi> = gamma(j,i) dx and <c_i^dagger Psi|c_j^dagger Psi> = delta_ij - gamma(i,j) dx.
        assert np.allclose(removal.T @ removal, up_p.T * s.dx, atol=1e-12)
        assert np.allclose(addition.T @ addition, np.eye(s.x.shape[0]) - up_p * s.dx, atol=1e-12)

    @pytest.mark.parametrize("name", ["helium", "triplet"])
    def test_density_matrix(self, name, request):
        s, state, green = request.getfixturevalue(name)
        _, up_p, down_p = iDEA.observables.density_matrix(s, state=state, return_spins=True)
        for spin, p in (("up", up_p), ("down", down_p)):
            G = iDEA.observables.green_function(s, green, t=np.array([0.0]), spin=spin)[0]
            assert np.allclose(G, 1.0j * p, atol=1e-12)
            if green.mu is not None:
                G_tau = iDEA.observables.green_function(s, green, tau=np.array([0.0]), spin=spin)[0]
                assert np.allclose(G_tau, p, atol=1e-12)

    def test_time_jump(self, helium):
        s, _, green = helium
        G = iDEA.observables.green_function(s, green, t=np.array([0.0, 1e-13]))
        assert np.allclose(G[1] - G[0], -1.0j * np.eye(s.x.shape[0]) / s.dx, atol=1e-8)

    @pytest.mark.parametrize("name", ["hydrogen", "helium", "lithium"])
    def test_galitskii_migdal(self, name, request):
        s, state, green = request.getfixturevalue(name)
        assert abs(iDEA.observables.galitskii_migdal_energy(s, green) - state.energy) < 1e-10


class TestNonInteracting:
    def test_non_interacting_limit(self):
        s = make_system(11, 5.0, "ud", 2.0, interacting=False)
        _, green = solve(s)
        mean_field = iDEA.methods.non_interacting.solve(s, silent=True)
        for spin in ("up", "down"):
            sigma = iDEA.observables.self_energy(s, green, z=Z, spin=spin)
            assert np.max(abs(sigma)) < 1e-10
            G = iDEA.observables.green_function(s, green, z=Z, spin=spin)
            G0 = iDEA.observables.green_function(s, mean_field, z=Z, spin=spin)
            assert np.allclose(G, G0, atol=1e-9)

    def test_consistency_with_mbpt(self):
        s = make_system(11, 5.0, "ud", 2.0, interacting=False)
        _, green = solve(s)
        mean_field = iDEA.methods.non_interacting.solve(s, silent=True)
        grid = iDEA.methods.mbpt.SpaceTimeGrid(0.1, 300.0)
        G, G_iw = iDEA.methods.mbpt.non_interacting_green_function(
            s, grid, mean_field.up.energies, mean_field.up.orbitals, green.mu
        )
        assert np.allclose(iDEA.observables.green_function(s, green, nu=grid.omega), G_iw, atol=1e-9)
        assert np.allclose(iDEA.observables.green_function(s, green, tau=grid.tau), G, atol=1e-9)


class TestSolvers:
    def test_exhausted_krylov_is_exact(self, helium):
        s, state, green = helium
        krylov = iDEA.methods.interacting.green_function(s, state, solver="krylov", krylov_dim=10**6, silent=True)
        for spin in ("up", "down"):
            for quantity in (iDEA.observables.green_function, iDEA.observables.self_energy):
                assert np.allclose(quantity(s, green, z=Z, spin=spin), quantity(s, krylov, z=Z, spin=spin), atol=1e-9)

    def test_truncated_krylov_moments(self, helium):
        s, state, green = helium
        blocks = 3
        krylov = iDEA.methods.interacting.green_function(
            s, state, solver="krylov", krylov_dim=blocks * s.x.shape[0], silent=True
        )
        assert krylov.info["up_completeness_error"] < 1e-10
        assert krylov.info["up_first_moment_error"] < 1e-9
        for j in range(2 * blocks):
            exact = (green.up.orbitals * green.up.energies**j) @ green.up.orbitals.T
            approximate = (krylov.up.orbitals * krylov.up.energies**j) @ krylov.up.orbitals.T
            assert np.max(abs(exact - approximate)) < 1e-8 * np.max(abs(exact))

    def test_invalid_solver(self, helium):
        s, state, _ = helium
        with pytest.raises(ValueError):
            iDEA.methods.interacting.green_function(s, state, solver="magic", silent=True)


class TestSelfEnergy:
    @pytest.mark.parametrize("name", ["helium", "triplet", "lithium"])
    def test_poles_match_dyson(self, name, request):
        s, _, green = request.getfixturevalue(name)
        omega = np.linspace(-2.0, 2.0, 5)
        for spin in ("up", "down"):
            poles = iDEA.observables.self_energy(s, green, z=Z, spin=spin)
            dyson = iDEA.observables.self_energy(s, green, z=Z, spin=spin, method="dyson")
            assert np.allclose(poles, dyson, atol=1e-9)
            poles = iDEA.observables.self_energy(s, green, omega=omega, kind="retarded", spin=spin)
            dyson = iDEA.observables.self_energy(s, green, omega=omega, kind="retarded", spin=spin, method="dyson")
            assert np.allclose(poles, dyson, atol=1e-9)

    def test_matches_loewdin_partitioning(self, helium):
        # Reference: complete the Dyson orbitals to an orthogonal matrix and partition the diagonal matrix of poles.
        s, _, green = helium
        V = green.up.orbitals * np.sqrt(s.dx)
        E = green.up.energies
        Q, _ = np.linalg.qr(V.T, mode="complete")
        W = Q[:, s.x.shape[0] :].T
        U = V @ np.diag(E) @ W.T
        D = W @ np.diag(E) @ W.T
        for z in Z:
            reference = U @ np.linalg.solve(z * np.eye(D.shape[0]) - D, U.T) / s.dx
            sigma_c = iDEA.observables.self_energy(s, green, z=np.array([z]), part="c")[0]
            assert np.allclose(sigma_c, reference, atol=1e-9)

    def test_exchange_is_fock(self, helium):
        s, state, green = helium
        _, up_p, _ = iDEA.observables.density_matrix(s, state=state, return_spins=True)
        sigma_x = iDEA.observables.self_energy(s, green, part="x")
        assert np.allclose(sigma_x, iDEA.observables.exchange_potential(s, up_p), atol=1e-9)

    @pytest.mark.parametrize("name", ["helium", "triplet"])
    def test_time_ordering(self, name, request):
        # Near each isolated self-energy pole, the time-ordered self-energy from Dyson's equation has an imaginary
        # part of the opposite sign to the pole's character.
        s, _, green = request.getfixturevalue(name)
        for spin in ("up", "down"):
            channel = getattr(green, spin)
            checked = 0
            for p in range(channel.sigma_energies.shape[0]):
                zeta = channel.sigma_energies[p]
                separation = min(
                    np.min(abs(np.delete(channel.sigma_energies, p) - zeta), initial=np.inf),
                    np.min(abs(channel.energies - zeta)),
                )
                weight = np.linalg.norm(channel.sigma_orbitals[:, p])
                if separation < 1e-6 or weight < 1e-6 or checked >= 20:
                    continue
                sigma = iDEA.observables.self_energy(
                    s, green, omega=np.array([zeta]), eta=1e-3 * separation, method="dyson", spin=spin
                )[0]
                v = channel.sigma_orbitals[:, p] / weight
                assert np.sign(np.imag(v.conj() @ sigma @ v)) == -np.sign(channel.sigma_character[p])
                checked += 1
            assert checked > 0

    def test_hydrogen_self_interaction(self, hydrogen):
        # For one electron the exact self-energy cancels the Hartree potential on the occupied orbital, at every
        # frequency, and every self-energy pole is particle-like.
        s, state, green = hydrogen
        removal = green.up.occupations > 0
        phi = green.up.orbitals[:, removal][:, 0]
        assert abs(green.up.energies[removal][0] - state.energy) < 1e-10
        sigma_x = iDEA.observables.self_energy(s, green, part="x")
        assert np.allclose((green.v_h * phi + sigma_x @ phi * s.dx), 0.0, atol=1e-9)
        sigma_c = iDEA.observables.self_energy(s, green, z=Z, part="c")
        assert np.allclose(sigma_c @ phi * s.dx, 0.0, atol=1e-9)
        for spin in ("up", "down"):
            assert np.allclose(getattr(green, spin).sigma_character, 1.0)

    def test_imaginary_time_matches_fourier_transform(self, helium):
        s, _, green = helium
        grid = iDEA.methods.mbpt.SpaceTimeGrid(0.05, 400.0)
        G_iw = iDEA.observables.green_function(s, green, nu=grid.omega)
        G_tau = iDEA.observables.green_function(s, green, tau=grid.tau)
        assert np.allclose(grid.inverse_fourier_transform(G_iw), G_tau, atol=1e-6)

    def test_quasiparticle_state(self, helium):
        s, _, _ = helium
        qp = iDEA.methods.mbpt.solve(s, mode="G0W0", silent=True)
        G = iDEA.observables.green_function(s, qp, nu=qp.omega)
        assert np.allclose(G, qp.up.G_iw, atol=1e-6)
        sigma = iDEA.observables.self_energy(s, qp, nu=qp.omega)
        assert np.allclose(sigma, qp.up.sigma_xc_iw, atol=1e-6)

    def test_errors(self, helium, triplet):
        s, _, green = helium
        with pytest.raises(ValueError):
            iDEA.observables.self_energy(s, green, t=np.array([1.0]))
        with pytest.raises(ValueError):
            iDEA.observables.green_function(s, green, t=np.array([1.0]), omega=np.array([1.0]))
        s, _, green = triplet
        assert green.mu is None
        with pytest.raises(ValueError):
            iDEA.observables.green_function(s, green, nu=np.array([1.0]))


class TestSpectralFunction:
    def test_normalisation(self, helium):
        s, _, green = helium
        omega = np.linspace(-60.0, 60.0, 60001)
        A = iDEA.observables.spectral_function(s, green, omega, eta=0.05, diagonal=True)
        total = np.trapezoid(A, omega, axis=0) if hasattr(np, "trapezoid") else np.trapz(A, omega, axis=0)
        assert np.allclose(total * s.dx, 1.0, atol=5e-3)

    def test_matches_retarded_green_function(self, helium):
        s, _, green = helium
        omega = np.linspace(-2.0, 2.0, 7)
        G = iDEA.observables.green_function(s, green, omega=omega, kind="retarded")
        A = iDEA.observables.spectral_function(s, green, omega)
        assert np.allclose(A, (1.0j / (2.0 * np.pi)) * (G - np.conj(np.swapaxes(G, 1, 2))), atol=1e-12)


class TestPlotting:
    def test_animations(self, hydrogen, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        s, _, green = hydrogen
        animation = iDEA.plotting.animate_green_function(
            s, green, omega=np.linspace(-1.0, 1.0, 3), file_name=str(tmp_path / "G.gif"), fps=5
        )
        assert (tmp_path / "G.gif").stat().st_size > 0
        assert animation is not None
        iDEA.plotting.animate_self_energy(
            s, green, t=np.linspace(0.0, 2.0, 3), spin="down", cutoff=2.0, file_name=str(tmp_path / "S.gif"), fps=5
        )
        assert (tmp_path / "S.gif").stat().st_size > 0
        fig = iDEA.plotting.plot_two_point(s.x, iDEA.observables.self_energy(s, green, part="x"), parts=("real",))
        assert len(fig.axes) >= 1
        with pytest.raises(ValueError):
            iDEA.plotting.animate(s.x, np.zeros((2, 3, 3)), np.arange(2))


@pytest.mark.skipif(not GPU_AVAILABLE, reason="cupy is not installed")
class TestGPU:
    @pytest.mark.parametrize("solver", ["dense", "krylov"])
    def test_gpu_matches_cpu(self, helium, solver):
        s, state, green = helium
        gpu = iDEA.methods.interacting.green_function(s, state, solver=solver, GPU=True, silent=True)
        for spin in ("up", "down"):
            assert np.allclose(
                iDEA.observables.green_function(s, green, z=Z, spin=spin),
                iDEA.observables.green_function(s, gpu, z=Z, spin=spin),
                atol=1e-9,
            )
            assert np.allclose(
                iDEA.observables.self_energy(s, green, z=Z, spin=spin),
                iDEA.observables.self_energy(s, gpu, z=Z, spin=spin),
                atol=1e-8,
            )
