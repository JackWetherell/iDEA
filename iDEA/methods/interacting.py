"""Contains all interacting functionality and solvers."""

import copy
import functools
import itertools
import math
import string
import warnings
from typing import Union

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from tqdm import tqdm

import iDEA.methods.non_interacting
import iDEA.state
import iDEA.system

name = "interacting"


def kinetic_energy_operator(s: iDEA.system.System, GPU: bool = False):
    r"""
    Compute many-particle kinetic energy operator as a matrix.

    This is built using a given number of finite differences to represent the second derivative.
    The number of differences taken is defined in s.stencil.

    | Args:
    |     s: iDEA.system.System, System object.
    |     GPU: bool, Compute on GPU using cupy. If false will use scipy on CPU. (default = False)

    | Returns:
    |     K: sparse matrix, Kintetic energy operator.
    """
    if GPU:
        import cupyx.scipy.sparse as csps

        sp = csps
        fmt = "csr"
    else:
        sp = sps
        fmt = "dia"
    k = sps.csr_matrix(iDEA.methods.non_interacting.kinetic_energy_operator(s))
    k = sp.csr_matrix(k) if GPU else sps.dia_matrix(k)
    I = sp.identity(s.x.shape[0], format=fmt)
    partial_operators = lambda A, B, k, n: (A if i + k == n - 1 else B for i in range(n))
    fold_partial_operators = lambda f, po: functools.reduce(lambda acc, val: f(val, acc, format=fmt), po)
    generate_terms = lambda f, A, B, n: (fold_partial_operators(f, partial_operators(A, B, k, n)) for k in range(n))
    K = functools.reduce(lambda a, b: a + b, generate_terms(sp.kron, k, I, s.count))
    return K


def external_potential_operator(s: iDEA.system.System, GPU: bool = False):
    r"""
    Compute many-particle external potential energy operator as a matrix.

    | Args:
    |     s: iDEA.system.System, System object.
    |     GPU: bool, Compute on GPU using cupy. If false will use scipy on CPU. (default = False)

    | Returns:
    |     Vext: sparse matrix, External potential operator.
    """
    if GPU:
        import cupyx.scipy.sparse as csps

        sp = csps
        fmt = "csr"
    else:
        sp = sps
        fmt = "dia"
    vext = sps.csr_matrix(iDEA.methods.non_interacting.external_potential_operator(s))
    vext = sp.csr_matrix(vext) if GPU else sps.dia_matrix(vext)
    I = sp.identity(s.x.shape[0], format=fmt)
    partial_operators = lambda A, B, k, n: (A if i + k == n - 1 else B for i in range(n))
    fold_partial_operators = lambda f, po: functools.reduce(lambda acc, val: f(val, acc, format=fmt), po)
    generate_terms = lambda f, A, B, n: (fold_partial_operators(f, partial_operators(A, B, k, n)) for k in range(n))
    Vext = functools.reduce(lambda a, b: a + b, generate_terms(sp.kron, vext, I, s.count))
    return Vext


def hamiltonian(s: iDEA.system.System, GPU: bool = False):
    r"""
    Compute the many-body Hamiltonian.

    | Args:
    |     s: iDEA.system.System, System object.
    |     GPU: bool, Compute on GPU using cupy. If false will use scipy on CPU. (default = False)

    | Returns:
    |     H: sparse matrix, Hamiltonian.
    """
    if GPU:
        import cupy as cp
        import cupyx.scipy.sparse as csps

        sp = csps
        xp = cp
        fmt = "csr"
    else:
        sp = sps
        xp = np
        fmt = "dia"

    # Construct the non-interacting part of the many-body Hamiltonian
    h = sps.csr_matrix(iDEA.methods.non_interacting.hamiltonian(s)[0])
    h = sp.csr_matrix(h) if GPU else sps.dia_matrix(h)
    I = sp.identity(s.x.shape[0], format=fmt)
    partial_operators = lambda A, B, k, n: (A if i + k == n - 1 else B for i in range(n))
    fold_partial_operators = lambda f, po: functools.reduce(lambda acc, val: f(val, acc, format=fmt), po)
    generate_terms = lambda f, A, B, n: (fold_partial_operators(f, partial_operators(A, B, k, n)) for k in range(n))
    H0 = functools.reduce(lambda a, b: a + b, generate_terms(sp.kron, h, I, s.count))

    # Add the interaction part of the many-body Hamiltonian
    symbols = string.ascii_lowercase + string.ascii_uppercase
    if s.count > 1:
        indices = ",".join(["".join(c) for c in itertools.combinations(symbols[: s.count], 2)])
        v_int = xp.asarray(s.v_int) if GPU else s.v_int
        U = xp.log(xp.einsum(indices + "->" + symbols[: s.count], *(xp.exp(v_int),) * int(s.count * (s.count - 1) / 2)))
        U = sp.diags(U.reshape(H0.shape[0]), format=fmt)
    else:
        U = 0.0

    # Construct the total many-body Hamiltonian
    H = H0 + U

    return H


def total_energy(s: iDEA.system.System, state: iDEA.state.ManyBodyState) -> float:
    r"""
    Compute the total energy of an interacting state.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.ManyBodyState, State.

    | Returns:
    |     E: float, Total energy.
    """
    return state.energy


def _permutation_parity(p):
    r"""
    Compute the permulation paritiy of a given permutation.

    | Args:
    |     p: tuple, Permutation.

    | Returns:
    |     parity: float, Permutation parity.
    """
    p = list(p)
    parity = 1
    for i in range(0, len(p) - 1):
        if p[i] != i:
            parity *= -1
            mn = min(range(i, len(p)), key=p.__getitem__)
            p[i], p[mn] = p[mn], p[i]
    return parity


def antisymmetrize(s, spaces, spins, energies):
    r"""
    Antisymmetrize the solution to the Schrodinger equation.

    | Args:
    |     s: iDEA.system.System, System object.
    |     spaces: np.ndarray, Spatial parts of the wavefunction.
    |     spins: np.ndarray, Spin parts of the wavefunction.
    |     energies: np.ndarray, Energies.

    | Returns:
    |     fulls: np.ndarray, Full anantisymmetrized wavefunction.
    |     spaces: np.ndarray, Spatial parts of the wavefunction.
    |     spins: np.ndarray, Spin parts of the wavefunction.
    |     energies: np.ndarray, Energies.

    """
    # Perform antisymmetrization.
    l = string.ascii_lowercase[: s.count]
    L = string.ascii_uppercase[: s.count]
    st = l + "Y," + L + "Y->" + "".join([i for sub in list(zip(l, L)) for i in sub]) + "Y"
    fulls = np.einsum(st, spaces, spins)
    L = list(zip(list(range(0, s.count * 2, 2)), list(range(1, s.count * 2, 2))))
    perms = itertools.permutations(list(range(s.count)))
    fulls_copy = copy.deepcopy(fulls)
    fulls = np.zeros_like(fulls)
    for p in perms:
        indices = list(itertools.chain(*[L[e] for e in p]))
        fulls += _permutation_parity(p) * np.moveaxis(fulls_copy, list(range(s.count * 2)), indices)

    # Filter out zeros.
    allowed_fulls = []
    allowed_energies = []
    allowed_spaces = []
    allowed_spins = []
    for n in range(fulls.shape[-1]):
        if np.allclose(fulls[..., n], np.zeros(fulls.shape[:-1])):
            pass
        else:
            allowed_fulls.append(fulls[..., n])
            allowed_energies.append(energies[n])
            allowed_spaces.append(spaces[..., n])
            allowed_spins.append(spins[..., n])
    fulls = np.moveaxis(np.array(allowed_fulls), 0, -1)
    spaces = np.moveaxis(np.array(allowed_spaces), 0, -1)
    spins = np.moveaxis(np.array(allowed_spins), 0, -1)
    energies = np.array(allowed_energies)

    # Normalise.
    for k in range(fulls.shape[-1]):
        fulls[..., k] = fulls[..., k] / np.sqrt(np.sum(abs(fulls[..., k]) ** 2) * s.dx**s.count)

    # Filter out duplicates.
    allowed_fulls = []
    allowed_energies = []
    for n in range(fulls.shape[-1] - 1):
        if np.allclose(abs(fulls[..., n]), abs(fulls[..., n + 1])):
            pass
        else:
            allowed_fulls.append(fulls[..., n])
            allowed_energies.append(energies[n])
    allowed_fulls.append(fulls[..., -1])
    allowed_energies.append(energies[-1])
    fulls = np.moveaxis(np.array(allowed_fulls), 0, -1)
    spaces = spaces[..., : fulls.shape[-1]]
    spins = spins[..., : fulls.shape[-1]]
    energies = np.array(allowed_energies)

    return fulls, spaces, spins, energies


def _estimate_level(s: iDEA.system.System, k: int) -> int:
    r"""
    Estimate the solution to the Schrodinger equation needed to eachive given antisymetric energy state.

    | Args:
    |     s: iDEA.system.System, System object.
    |     k: int, Target energy state.

    | Returns:
    |     level: int, Extimate of level of excitement.
    """
    return (abs(s.up_count - s.down_count) + 1) ** 2 * s.count * (k + 1)


def solve(
    s: iDEA.system.System, H: np.ndarray = None, k: int = 0, level=None, GPU=False
) -> Union[iDEA.state.ManyBodyState, iDEA.state.ManyBodyStates]:
    r"""
    Solves the interacting Schrodinger equation of the given system.

    | Args:
    |     s: iDEA.system.System, System object.
    |     H: np.ndarray, Hamiltonian [If None this will be computed from s]. (default = None)
    |     k: int, Energy state to solve for. If -1 will return all states. (default = 0, the ground-state)
    |     level: int. Max level of excitation to use when solving the Schrodinger equation.
    |     GPU: bool, Solve on GPU using cupy. If false will use scipy on CPU.

    | Returns:
    |     state: iDEA.state.ManyBodyState or iDEA.state.ManyBodyStates, Solved state, or collection of all solved states if k=-1.
    """
    # Construct the Hamiltonian.
    if H is None:
        H = hamiltonian(s, GPU=GPU)

    # Estimate the level of excitation.
    if level is None:
        if k == -1:
            raise ValueError("level must be provided explicitly when k=-1 (all states).")
        level = _estimate_level(s, k)

    # Solve the many-body Schrodinger equation.
    if GPU:
        import cupy as cp
        import cupyx.scipy.sparse as csps
        import cupyx.scipy.sparse.linalg as cspsla

        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
        print(f"iDEA.methods.interacting.solve: solving eigenproblem on GPU: {name}...")
        energies, spaces = cspsla.eigsh(csps.csr_matrix(H), k=level, which="SA")
        energies = energies.get()
        spaces = spaces.get()
    else:
        print("iDEA.methods.interacting.solve: solving eigenproblem on CPU...")
        energies, spaces = spsla.eigsh(H.tocsr(), k=level, which="SA")

    # Reshape and normalise the solutions.
    spaces = spaces.reshape((s.x.shape[0],) * s.count + (spaces.shape[-1],))
    for j in range(spaces.shape[-1]):
        spaces[..., j] = spaces[..., j] / np.sqrt(np.sum(abs(spaces[..., j]) ** 2) * s.dx**s.count)

    # Construct the spin part.
    symbols = string.ascii_lowercase + string.ascii_uppercase
    u = np.array([1, 0])
    d = np.array([0, 1])
    spin_state = tuple([u if spin == "u" else d for spin in s.electrons])
    spin = np.einsum(",".join(symbols[: s.count]) + "->" + "".join(symbols[: s.count]), *spin_state)
    spins = np.zeros(shape=((2,) * s.count + (spaces.shape[-1],)))
    for i in range(spaces.shape[-1]):
        spins[..., i] = spin

    # Antisymmetrize.
    fulls, spaces, spins, energies = antisymmetrize(s, spaces, spins, energies)

    # Populate the state.
    if k != -1:
        state = iDEA.state.ManyBodyState()
        state.space = spaces[..., k]
        state.spin = spins[..., k]
        state.full = fulls[..., k]
        state.energy = energies[k]
        return state
    else:
        states = iDEA.state.ManyBodyStates()
        states.spaces = spaces
        states.spins = spins
        states.fulls = fulls
        states.energies = energies
        return states


def _expm_multiply_gpu(A, v):
    r"""
    Compute the action of the matrix exponential, expm(A) @ v, on the GPU.

    cupy does not provide a sparse expm_multiply, so when it is unavailable this falls back to
    scaling with a truncated Taylor series: expm(A) @ v = (expm(A/m))^m @ v, where m is chosen
    such that the infinity norm of A/m is at most 1, guaranteeing rapid convergence of the
    series for each substep without catastrophic cancellation.

    | Args:
    |     A: cupyx.scipy.sparse matrix, Matrix.
    |     v: cp.ndarray, Vector.

    | Returns:
    |     w: cp.ndarray, Result of expm(A) @ v.
    """
    import cupy as cp
    import cupyx.scipy.sparse as csps
    import cupyx.scipy.sparse.linalg as cspsla

    if hasattr(cspsla, "expm_multiply"):
        return cspsla.expm_multiply(A, v)

    A = csps.csr_matrix(A)
    B = csps.csr_matrix((cp.abs(A.data), A.indices, A.indptr), shape=A.shape)
    norm = float(B.sum(axis=1).max())
    m = int(norm) + 1

    w = v.astype(complex)
    for _ in range(m):
        term = w
        for k in range(1, 60):
            term = A.dot(term) / (m * k)
            w = w + term
            if float(cp.linalg.norm(term)) <= 1e-16 * float(cp.linalg.norm(w)):
                break
    return w


def propagate_step(
    s: iDEA.system.System,
    evolution: iDEA.state.ManyBodyEvolution,
    H: sps.dia_matrix,
    v_ptrb: np.ndarray,
    j: int,
    dt: float,
    objs: tuple,
    GPU: bool = False,
) -> iDEA.state.ManyBodyEvolution:
    r"""
    Propagate a many body state forward in time, one time-step, due to a local pertubation.

    | Args:
    |     s: iDEA.system.System, System object.
    |     evolution: iDEA.state.ManyBodyEvolution, time-dependent evolution.
    |     H: np.ndarray, Static Hamiltonian [If None this will be computed from s]. (default = None)
    |     v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
    |     j: int, Time index.
    |     dt: float, Time-step.
    |     objs: tuple. Tuple of objects needed to construct many-body operator (I, generate_terms).
    |     GPU: bool, Propagate on GPU using cupy. If false will use scipy on CPU. (default = False)

    | Returns:
    |     evolution: iDEA.state.ManyBodyEvolution, time-dependent evolution one time-step evolved.
    """
    if GPU:
        import cupy as cp
        import cupyx.scipy.sparse as csps

        # Construct the pertubation potential.
        vptrb = csps.diags(cp.asarray(v_ptrb[j, :]), format="csr")
        terms = objs[1](csps.kron, vptrb, objs[0], s.count)
        Vptrb = functools.reduce(lambda a, b: a + b, terms)

        # Contruct the perturbed Hamiltonian.
        Hp = H + Vptrb

        # Evolve.
        wavefunction = cp.asarray(evolution.td_space[j - 1, ...].reshape(s.x.shape[0] ** s.count))
        wavefunction = _expm_multiply_gpu(-1.0j * dt * Hp, wavefunction)
        evolution.td_space[j, ...] = cp.asnumpy(wavefunction).reshape((s.x.shape[0],) * s.count)

        return evolution

    # Construct the pertubation potential.
    vptrb = sps.dia_matrix(np.diag(v_ptrb[j, :]))
    terms = objs[1](sps.kron, vptrb, objs[0], s.count)
    Vptrb = sps.dia_matrix((s.x.shape[0] ** s.count,) * 2, dtype=float)
    for term in terms:
        Vptrb += term

    # Contruct the perturbed Hamiltonian.
    Hp = H + Vptrb

    # Evolve.
    wavefunction = evolution.td_space[j - 1, ...].reshape(s.x.shape[0] ** s.count)
    wavefunction = spsla.expm_multiply(-1.0j * dt * Hp, wavefunction)
    evolution.td_space[j, ...] = wavefunction.reshape((s.x.shape[0],) * s.count)

    return evolution


def propagate(
    s: iDEA.system.System,
    state: iDEA.state.ManyBodyState,
    v_ptrb: np.ndarray,
    t: np.ndarray,
    H: sps.dia_matrix = None,
    GPU: bool = False,
) -> iDEA.state.ManyBodyEvolution:
    r"""
    Propagate a many body state forward in time due to a local pertubation.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.ManyBodyState, State to be propigated.
    |     v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
    |     t: np.ndarray, Grid of time values.
    |     H: np.ndarray, Static Hamiltonian [If None this will be computed from s]. (default = None)
    |     GPU: bool, Propagate on GPU using cupy. If false will use scipy on CPU. (default = False)

    | Returns:
    |     evolution: iDEA.state.ManyBodyEvolution, Solved time-dependent evolution.
    """
    # Construct the unperturbed Hamiltonian.
    if H is None:
        H = hamiltonian(s, GPU=GPU)

    # Choose the sparse linear algebra backend.
    if GPU:
        import cupy as cp
        import cupyx.scipy.sparse as csps

        if not csps.issparse(H):
            H = csps.csr_matrix(sps.csr_matrix(H))
        device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
        print(f"iDEA.methods.interacting.propagate: propagating state on GPU: {device_name}...")
        sp = csps
        fmt = "csr"
    else:
        sp = sps
        fmt = "dia"

    # Compute timestep.
    dt = t[1] - t[0]

    # Initilise time-dependent wavefunction.
    evolution = iDEA.state.ManyBodyEvolution(initial_state=state)
    evolution.td_space = np.zeros(shape=t.shape + state.space.shape, dtype=complex)
    evolution.td_space[0, ...] = copy.deepcopy(evolution.space)

    # Construct objects needed to update potential.
    I = sp.identity(s.x.shape[0], format=fmt)
    partial_operators = lambda A, B, k, n: (A if i + k == n - 1 else B for i in range(n))
    fold_partial_operators = lambda f, po: functools.reduce(lambda acc, val: f(val, acc, format=fmt), po)
    generate_terms = lambda f, A, B, n: (fold_partial_operators(f, partial_operators(A, B, k, n)) for k in range(n))
    objs = (I, generate_terms)

    # Propagate.
    for j, _ti in enumerate(tqdm(t, desc="iDEA.methods.interacting.propagate: propagating state")):
        if j != 0:
            propagate_step(s, evolution, H, v_ptrb, j, dt, objs, GPU=GPU)

    # Populate the many-body time-dependent evolution.
    evolution.v_ptrb = v_ptrb
    evolution.t = t

    return evolution


def _comb_table(x_count: int, n: int) -> np.ndarray:
    r"""
    Compute a table of binomial coefficients C(a,k) for a = 0..x_count and k = 0..n.

    | Args:
    |     x_count: int, Number of grid points.
    |     n: int, Largest k.

    | Returns:
    |     table: np.ndarray, Binomial coefficients, indexed as table[a,k].
    """
    return np.array([[math.comb(a, k) for k in range(n + 1)] for a in range(x_count + 1)], dtype=np.int64)


def _combination_ranks(combinations: np.ndarray, x_count: int) -> np.ndarray:
    r"""
    Compute the colexicographic rank of sorted combinations (the combinatorial number system).

    .. math:: \mathrm{rank}(a_1 < a_2 < \dots < a_n) = \sum_k \binom{a_k}{k}

    | Args:
    |     combinations: np.ndarray, Sorted combinations of grid points, indexed as combinations[combination,slot].
    |     x_count: int, Number of grid points.

    | Returns:
    |     ranks: np.ndarray, Ranks, indexed as ranks[combination].
    """
    n = combinations.shape[1]
    ranks = np.zeros(combinations.shape[0], dtype=np.int64)
    if n == 0:
        return ranks
    table = _comb_table(x_count, n)
    for k in range(n):
        ranks += table[combinations[:, k], k + 1]
    return ranks


def _combinations(x_count: int, n: int) -> np.ndarray:
    r"""
    Compute all sorted combinations of n grid points, in colexicographic order (so that row r has rank r).

    | Args:
    |     x_count: int, Number of grid points.
    |     n: int, Number of electrons.

    | Returns:
    |     combinations: np.ndarray, Sorted combinations, indexed as combinations[combination,slot].
    """
    if n == 0:
        return np.zeros((1, 0), dtype=np.int64)
    combinations = np.array(list(itertools.combinations(range(x_count), n)), dtype=np.int64)
    return combinations[np.argsort(_combination_ranks(combinations, x_count), kind="stable")]


def _determinants(x_count: int, up_count: int, down_count: int) -> tuple:
    r"""
    Enumerate the Slater determinants of a sector with a fixed number of up and down electrons.

    The determinant with up sites a_1 < ... and down sites b_1 < ... is
    c^dagger_{a_1 up} ... c^dagger_{b_1 down} ... |0>, and has index rank(a) * C(x_count, down_count) + rank(b).

    | Args:
    |     x_count: int, Number of grid points.
    |     up_count: int, Number of up electrons.
    |     down_count: int, Number of down electrons.

    | Returns:
    |     up: np.ndarray, Up combinations, indexed as up[combination,slot].
    |     down: np.ndarray, Down combinations, indexed as down[combination,slot].
    |     up_index: np.ndarray, Index of the up combination of each determinant, indexed as up_index[determinant].
    |     down_index: np.ndarray, Index of the down combination of each determinant, indexed as down_index[determinant].
    """
    up = _combinations(x_count, up_count)
    down = _combinations(x_count, down_count)
    up_index = np.repeat(np.arange(up.shape[0]), down.shape[0])
    down_index = np.tile(np.arange(down.shape[0]), up.shape[0])
    return up, down, up_index, down_index


def sector_isometry(x_count: int, up_count: int, down_count: int) -> sps.csr_matrix:
    r"""
    Compute the isometry from the Slater determinant basis of a sector to the product space of iDEA's
    many-body Hamiltonian.

    Electrons are ordered with all up electrons first. Each column is the normalised spatial function of a
    determinant, antisymmetric within each spin group.

    | Args:
    |     x_count: int, Number of grid points.
    |     up_count: int, Number of up electrons.
    |     down_count: int, Number of down electrons.

    | Returns:
    |     P: scipy.sparse.csr_matrix, Isometry, of shape (x_count**(up_count + down_count), number of determinants).
    """
    count = up_count + down_count
    up, down, up_index, down_index = _determinants(x_count, up_count, down_count)
    columns = np.arange(up_index.shape[0])
    norm = 1.0 / np.sqrt(math.factorial(up_count) * math.factorial(down_count))
    rows, cols, values = [], [], []
    for up_permutation in itertools.permutations(range(up_count)):
        for down_permutation in itertools.permutations(range(down_count)):
            coordinates = np.concatenate(
                (up[up_index][:, list(up_permutation)], down[down_index][:, list(down_permutation)]), axis=1
            )
            sign = _permutation_parity(up_permutation) * _permutation_parity(down_permutation)
            rows.append(np.ravel_multi_index(tuple(coordinates.T), (x_count,) * count))
            cols.append(columns)
            values.append(np.full(columns.shape[0], sign * norm))
    return sps.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(x_count**count, columns.shape[0]),
    )


def sector_hamiltonian(s: iDEA.system.System, up_count: int, down_count: int) -> sps.csr_matrix:
    r"""
    Compute the many-body Hamiltonian of the system with a given number of up and down electrons, in the Slater
    determinant basis of that sector (see sector_isometry).

    | Args:
    |     s: iDEA.system.System, System object (provides the grid, external potential, interaction and stencil).
    |     up_count: int, Number of up electrons.
    |     down_count: int, Number of down electrons.

    | Returns:
    |     H: scipy.sparse.csr_matrix, Hamiltonian in the determinant basis.
    """
    if up_count + down_count == 0:
        return sps.csr_matrix(np.zeros((1, 1)))
    sector = iDEA.system.System(s.x, s.v_ext, s.v_int, "u" * up_count + "d" * down_count, stencil=s.stencil)
    P = sector_isometry(s.x.shape[0], up_count, down_count)
    H = sps.csr_matrix(hamiltonian(sector))
    return sps.csr_matrix(P.T @ H @ P)


def spin_ordered_wavefunction(s: iDEA.system.System, state: iDEA.state.ManyBodyState) -> np.ndarray:
    r"""
    Compute the spatial wavefunction of a many-body state with all up electrons first,
    .. math:: \psi(x_1,\dots,x_{N_\uparrow};y_1,\dots,y_{N_\downarrow}),
    which is antisymmetric within each spin group and normalised on the grid as :math:`\sum |\psi|^2 = 1`.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.ManyBodyState, State (only state.full is used).

    | Returns:
    |     psi: np.ndarray, Spin-ordered wavefunction, indexed as psi[x_1,...,y_1,...].
    """
    index = []
    for i in range(s.count):
        index += [slice(None), 0 if i < s.up_count else 1]
    psi = state.full[tuple(index)] * np.sqrt(math.comb(s.count, s.up_count)) * s.dx ** (s.count / 2)
    norm = np.sum(abs(psi) ** 2)
    if abs(norm - 1.0) > 1e-8:
        raise ValueError(f"The spin-ordered wavefunction is not normalised (norm = {norm}).")
    return psi


def _field_sources(x_count: int, X: np.ndarray, up_count: int, down_count: int, spin: str, create: bool) -> np.ndarray:
    r"""
    Apply the field operators c_{i,spin} (create = False) or c^dagger_{i,spin} (create = True) at every grid point
    to a state given in the determinant basis.

    | Args:
    |     x_count: int, Number of grid points.
    |     X: np.ndarray, State in the determinant basis of the (up_count, down_count) sector, indexed as X[determinant].
    |     up_count: int, Number of up electrons of the state.
    |     down_count: int, Number of down electrons of the state.
    |     spin: str, Spin of the field operator ('up' or 'down').
    |     create: bool, True for the creation operator, False for the annihilation operator.

    | Returns:
    |     sources: np.ndarray, c_i|X> or c^dagger_i|X> in the determinant basis of the target sector, indexed as sources[determinant,i].
    """
    shift = 1 if create else -1
    target = (up_count + shift, down_count) if spin == "up" else (up_count, down_count + shift)
    large, small = (target, (up_count, down_count)) if create else ((up_count, down_count), target)
    up, down, up_index, down_index = _determinants(x_count, *large)
    large_index = up_index * down.shape[0] + down_index
    small_down_dim = math.comb(x_count, small[1])
    sources = np.zeros((math.comb(x_count, target[0]) * math.comb(x_count, target[1]), x_count), dtype=X.dtype)
    group = up if spin == "up" else down
    for k in range(group.shape[1]):
        rest_ranks = _combination_ranks(np.delete(group, k, axis=1), x_count)
        if spin == "up":
            sites = up[up_index, k]
            small_index = rest_ranks[up_index] * small_down_dim + down_index
            sign = (-1) ** k
        else:
            sites = down[down_index, k]
            small_index = up_index * small_down_dim + rest_ranks[down_index]
            sign = (-1) ** (large[0] + k)
        if create:
            sources[large_index, sites] = sign * X[small_index]
        else:
            sources[small_index, sites] = sign * X[large_index]
    return sources


def _backend(GPU: bool) -> tuple:
    r"""
    Return the array and sparse modules to use.

    | Args:
    |     GPU: bool, Use cupy on the GPU. If false will use numpy and scipy on the CPU.

    | Returns:
    |     xp, sp: modules, Dense and sparse array modules.
    """
    if GPU:
        import cupy as cp
        import cupyx.scipy.sparse as csps

        return cp, csps
    return np, sps


def _to_numpy(a):
    r"""Copy an array to the host, if it is a cupy array."""
    return a.get() if hasattr(a, "get") else a


def _block_krylov(apply, B, max_dim: int, tol: float, scale: float, xp=np, orthonormal: bool = False) -> tuple:
    r"""
    Construct an orthonormal basis Q of the block Krylov space span{B, AB, A^2B, ...} of a Hermitian operator A,
    and the projection T = Q^dagger A Q (block Rayleigh-Ritz).

    Each new block is orthogonalised twice against every previous vector (classical Gram-Schmidt, CGS2), and
    directions with singular values below tol * scale are deflated. The construction stops when max_dim vectors are
    reached or the Krylov space is exhausted, in which case the eigenvalues of T are exact. Since B lies in the
    Krylov space, the moments B^dagger A^k B are reproduced exactly for k up to twice the number of blocks minus one.

    | Args:
    |     apply: callable, Action of A on a block of vectors.
    |     B: array, Starting block, indexed as B[dimension,vector].
    |     max_dim: int, Maximum number of Krylov vectors.
    |     tol: float, Relative deflation tolerance.
    |     scale: float, Estimate of the norm of A.
    |     xp: module, Dense array module (numpy or cupy). (default = numpy)
    |     orthonormal: bool, Set to True if the columns of B are already orthonormal, in which case they are used as the first block unchanged. (default = False)

    | Returns:
    |     T: array, Projected operator Q^dagger A Q, indexed as T[vector,vector].
    |     Q: array, Orthonormal Krylov basis, indexed as Q[dimension,vector].
    """
    dimension = B.shape[0]
    max_dim = min(max_dim, dimension)
    if orthonormal:
        first = B[:, :max_dim]
    else:
        U, singular_values, _ = xp.linalg.svd(B, full_matrices=False)
        first = U[:, singular_values > tol * singular_values[0]][:, :max_dim]
    Q = xp.zeros((dimension, max_dim), dtype=B.dtype)
    AQ = xp.zeros((dimension, max_dim), dtype=B.dtype)
    start, end = 0, first.shape[1]
    Q[:, :end] = first
    while True:
        AQ[:, start:end] = apply(Q[:, start:end])
        if end >= max_dim:
            break
        W = AQ[:, start:end]
        for _ in range(2):
            W = W - Q[:, :end] @ (Q[:, :end].conj().T @ W)
        U, singular_values, _ = xp.linalg.svd(W, full_matrices=False)
        new = U[:, singular_values > tol * scale][:, : max_dim - end]
        if new.shape[1] == 0:
            break
        start, end = end, end + new.shape[1]
        Q[:, start:end] = new
    Q = Q[:, :end]
    T = Q.conj().T @ AQ[:, :end]
    return 0.5 * (T + T.conj().T), Q


def _sector_poles(
    H: sps.csr_matrix, sources: np.ndarray, solver: str, dense_max: int, krylov_dim: int, tol: float, GPU: bool
) -> tuple:
    r"""
    Compute the poles of a sector and the overlaps of its eigenstates with a block of source states.

    | Args:
    |     H: scipy.sparse.csr_matrix, Hamiltonian of the sector in its determinant basis.
    |     sources: np.ndarray, Source states in the determinant basis, indexed as sources[determinant,i].
    |     solver: str, 'dense' (complete diagonalisation), 'krylov' (block Krylov) or 'auto' (dense if the sector dimension is at most dense_max).
    |     dense_max: int, Largest sector dimension to diagonalise completely when solver = 'auto'.
    |     krylov_dim: int, Maximum dimension of the block Krylov space [If None this will be 30 blocks of sources].
    |     tol: float, Relative deflation tolerance of the block Krylov method.
    |     GPU: bool, Solve on GPU using cupy.

    | Returns:
    |     energies: np.ndarray, Energies of the sector eigenstates, indexed as energies[state].
    |     overlaps: np.ndarray, Overlaps <state|source_i>, indexed as overlaps[state,i].
    |     info: dict, Solver details.
    """
    dimension = H.shape[0]
    if solver == "auto":
        solver = "dense" if dimension <= dense_max else "krylov"
    if solver not in ("dense", "krylov"):
        raise ValueError(f"solver must be one of 'auto', 'dense' or 'krylov', got {solver} instead.")
    xp, sp = _backend(GPU)
    B = xp.asarray(sources)
    if solver == "dense":
        energies, states = xp.linalg.eigh(xp.asarray(H.toarray()))
        overlaps = states.conj().T @ B
        info = {"solver": "dense", "dimension": dimension, "krylov_dim": dimension}
    else:
        if krylov_dim is None:
            krylov_dim = 30 * sources.shape[1]
        H_device = sp.csr_matrix(H) if GPU else H
        scale = float(abs(H).sum(axis=1).max())
        T, Q = _block_krylov(lambda V: H_device @ V, B, krylov_dim, tol, scale, xp=xp)
        energies, vectors = xp.linalg.eigh(T)
        overlaps = vectors.conj().T @ (Q.conj().T @ B)
        info = {"solver": "krylov", "dimension": dimension, "krylov_dim": int(T.shape[0])}
    return _to_numpy(energies), _to_numpy(overlaps), info


def _self_energy_poles(energies: np.ndarray, amplitudes: np.ndarray, character: np.ndarray, tol: float) -> tuple:
    r"""
    Compute the pole representation of the self-energy from the pole representation of G.

    The Green's function G(z) = V (z - E)^{-1} V^dagger, with V V^dagger = 1 (completeness), is the projection onto
    the grid of the resolvent of the diagonal matrix E. Block Lanczos on E starting from V^dagger partitions E into
    the static Hamiltonian h_inf (the first block), and the remaining space, giving
    .. math:: G^{-1}(z) = z - h_\infty - T_{12} (z - T_{22})^{-1} T_{21}
    so that the correlation self-energy has poles at the eigenvalues of T_22 (Loewdin partitioning). The character
    of each self-energy pole is its weight on addition (+1) minus removal (-1) poles of G, which sets its time ordering.

    | Args:
    |     energies: np.ndarray, Poles of G, indexed as energies[pole].
    |     amplitudes: np.ndarray, Dyson amplitudes on the grid (normalised as discrete vectors), indexed as amplitudes[i,pole].
    |     character: np.ndarray, +1 for addition and -1 for removal poles, indexed as character[pole].
    |     tol: float, Relative deflation tolerance.

    | Returns:
    |     h_inf: np.ndarray, Static part of G^{-1}, the first moment of the spectral function, indexed as h_inf[i,j].
    |     sigma_energies: np.ndarray, Self-energy poles, indexed as sigma_energies[pole].
    |     sigma_amplitudes: np.ndarray, Self-energy residue vectors (discrete), indexed as sigma_amplitudes[i,pole].
    |     sigma_character: np.ndarray, Character of each self-energy pole, indexed as sigma_character[pole].
    |     completeness: float, Largest deviation of V V^dagger from the identity.
    """
    x_count = amplitudes.shape[0]
    overlap = amplitudes @ amplitudes.conj().T
    completeness = float(np.max(abs(overlap - np.eye(x_count))))
    w, U = np.linalg.eigh(overlap)
    V = (U * (1.0 / np.sqrt(w))) @ U.conj().T @ amplitudes
    scale = float(np.max(abs(energies)))
    T, Q = _block_krylov(
        lambda Y: energies[:, None] * Y, V.conj().T, energies.shape[0], tol, scale, xp=np, orthonormal=True
    )
    h_inf = T[:x_count, :x_count]
    zeta, Z = np.linalg.eigh(T[x_count:, x_count:])
    sigma_amplitudes = T[:x_count, x_count:] @ Z
    Y = Q[:, x_count:] @ Z
    sigma_character = np.sum(character[:, None] * abs(Y) ** 2, axis=0)

    # Within degenerate clusters of poles, choose the basis that diagonalises the character.
    start = 0
    while start < zeta.shape[0]:
        end = start + 1
        while end < zeta.shape[0] and zeta[end] - zeta[end - 1] <= 1e-9 * (1.0 + abs(zeta[end])):
            end += 1
        if end - start > 1:
            Y_c = Y[:, start:end]
            c, R = np.linalg.eigh(Y_c.conj().T @ (character[:, None] * Y_c))
            sigma_character[start:end] = c
            sigma_amplitudes[:, start:end] = sigma_amplitudes[:, start:end] @ R
        start = end
    return h_inf, zeta, sigma_amplitudes, sigma_character, completeness


def green_function(
    s: iDEA.system.System,
    state: iDEA.state.ManyBodyState,
    solver: str = "auto",
    dense_max: int = 6000,
    krylov_dim: int = None,
    tol: float = 1e-10,
    self_energy: bool = True,
    GPU: bool = False,
    silent: bool = False,
) -> iDEA.state.GreenFunction:
    r"""
    Compute the exact one-body Green's function, and exchange-correlation self-energy, of an interacting state.

    The Green's function is built from the exact eigenstates of the (N-1) and (N+1)-electron systems (the Lehmann
    representation). Each sector is diagonalised in its Slater determinant basis: completely (solver = 'dense'),
    giving every pole exactly, or with a block Krylov method seeded by the states c_i|Psi> and c^dagger_i|Psi>
    (solver = 'krylov'), which keeps every sum rule (and so the static part of the self-energy) exact and converges
    the poles as krylov_dim grows (all poles are exact once the Krylov space is exhausted). The self-energy is then
    found in pole form from the Green's function (see iDEA.state.GreenFunction).

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.ManyBodyState, Interacting state (usually the ground state from solve).
    |     solver: str, 'dense', 'krylov' or 'auto' (dense if the sector dimension is at most dense_max). (default = 'auto')
    |     dense_max: int, Largest sector dimension to diagonalise completely when solver = 'auto'. (default = 6000)
    |     krylov_dim: int, Maximum dimension of each block Krylov space [If None this will be 30 times the number of grid points]. (default = None)
    |     tol: float, Relative deflation tolerance of the block Krylov method. (default = 1e-10)
    |     self_energy: bool, Also compute the self-energy. (default = True)
    |     GPU: bool, Diagonalise the sectors on GPU using cupy. If false will use numpy and scipy on CPU. (default = False)
    |     silent: bool, Set to true to prevent printing. (default = False)

    | Returns:
    |     green_function: iDEA.state.GreenFunction, Green's function and self-energy in pole form.
    """
    x_count = s.x.shape[0]
    if not np.allclose(s.v_int, s.v_int.T):
        raise ValueError("The interaction v_int must be symmetric.")
    up_count, down_count = s.up_count, s.down_count

    # The state in the determinant basis, and its energy as a Rayleigh quotient.
    psi = spin_ordered_wavefunction(s, state)
    up, down, up_index, down_index = _determinants(x_count, up_count, down_count)
    coordinates = np.concatenate((up[up_index], down[down_index]), axis=1)
    X = psi[tuple(coordinates.T)] * np.sqrt(math.factorial(up_count) * math.factorial(down_count))
    H = sector_hamiltonian(s, up_count, down_count)
    HX = H @ X
    E0 = float(np.real(np.vdot(X, HX)))
    residual = float(np.linalg.norm(HX - E0 * X))
    if residual > 1e-6:
        raise ValueError(f"The state is not an eigenstate of the Hamiltonian (residual {residual:.2e}).")
    if residual > 1e-8:
        warnings.warn(f"The state is an eigenstate only to a residual of {residual:.2e}.", stacklevel=2)

    # The poles of each spin channel.
    green = iDEA.state.GreenFunction()
    green.energy = E0
    green.info = {"rayleigh_residual": residual, "sectors": []}
    hamiltonians = {}
    channels = {}
    for spin, count in (("up", up_count), ("down", down_count)):
        energies, amplitudes, character = [], [], []
        for create in (False, True):
            if (not create and count == 0) or (create and count == x_count):
                continue
            shift = 1 if create else -1
            sector = (up_count + shift, down_count) if spin == "up" else (up_count, down_count + shift)
            if sector not in hamiltonians:
                hamiltonians[sector] = sector_hamiltonian(s, *sector)
            sources = _field_sources(x_count, X, up_count, down_count, spin, create)
            if sum(sector) == 0:
                sector_energies, overlaps = np.zeros(1), sources.copy()
                info = {"solver": "vacuum", "dimension": 1, "krylov_dim": 1}
            else:
                if not silent:
                    print(
                        f"iDEA.methods.interacting.green_function: {'adding' if create else 'removing'} a spin-{spin} "
                        f"electron, sector dimension {hamiltonians[sector].shape[0]}..."
                    )
                sector_energies, overlaps, info = _sector_poles(
                    hamiltonians[sector], sources, solver, dense_max, krylov_dim, tol, GPU
                )
            info.update({"spin": spin, "sector": "addition" if create else "removal", "electrons": sector})
            green.info["sectors"].append(info)
            weights = np.sum(abs(overlaps) ** 2, axis=1)
            keep = weights > 1e-14
            if create:
                energies.append(sector_energies[keep] - E0)
                amplitudes.append(overlaps[keep].conj().T)
                character.append(np.ones(np.count_nonzero(keep)))
            else:
                energies.append(E0 - sector_energies[keep])
                amplitudes.append(overlaps[keep].T)
                character.append(-np.ones(np.count_nonzero(keep)))
        energies = np.concatenate(energies)
        order = np.argsort(energies, kind="stable")
        channels[spin] = (
            energies[order],
            np.concatenate(amplitudes, axis=1)[:, order],
            np.concatenate(character)[order],
        )

    # The chemical potential at the centre of the gap, over both spin channels.
    removal = [e[c < 0] for e, _, c in channels.values()]
    addition = [e[c > 0] for e, _, c in channels.values()]
    removal = np.concatenate(removal) if removal else np.zeros(0)
    addition = np.concatenate(addition) if addition else np.zeros(0)
    if removal.shape[0] > 0 and addition.shape[0] > 0 and np.min(addition) > np.max(removal):
        green.mu = 0.5 * (np.min(addition) + np.max(removal))
    else:
        green.mu = None

    # Store the Green's function, with Dyson orbitals normalised as kernels.
    n = np.zeros(x_count)
    density_matrices = {}
    for spin in ("up", "down"):
        energies, amplitudes, character = channels[spin]
        container = getattr(green, spin)
        container.mu = green.mu
        container.energies = energies
        container.orbitals = amplitudes / np.sqrt(s.dx)
        container.occupations = (character < 0).astype(float)
        removal_orbitals = container.orbitals[:, character < 0]
        density_matrices[spin] = removal_orbitals @ removal_orbitals.conj().T
        n += np.real(np.diag(density_matrices[spin]))
    green.v_h = np.dot(n, s.v_int) * s.dx

    # The self-energy in pole form.
    gm_energy = 0.0
    h0 = iDEA.methods.non_interacting.hamiltonian(s)[0]
    for spin in ("up", "down"):
        energies, amplitudes, character = channels[spin]
        container = getattr(green, spin)
        removal_amplitudes = amplitudes[:, character < 0]
        gm_energy += 0.5 * np.real(
            np.sum(removal_amplitudes.conj() * (h0 @ removal_amplitudes))
            + np.sum(energies[character < 0] * np.sum(abs(removal_amplitudes) ** 2, axis=0))
        )
        if not self_energy:
            continue
        h_inf, zeta, sigma_amplitudes, sigma_character, completeness = _self_energy_poles(
            energies, amplitudes, character, tol
        )
        container.sigma_x = (h_inf - h0 - np.diag(green.v_h)) / s.dx
        container.sigma_energies = zeta
        container.sigma_orbitals = sigma_amplitudes / np.sqrt(s.dx)
        container.sigma_character = sigma_character
        fock = -density_matrices[spin] * s.v_int
        green.info[f"{spin}_completeness_error"] = completeness
        green.info[f"{spin}_first_moment_error"] = float(np.max(abs(container.sigma_x - fock)))
        ambiguous = abs(sigma_character) < 1e-6
        if np.any(ambiguous):
            warnings.warn(
                f"{np.count_nonzero(ambiguous)} {spin} self-energy pole(s) have no definite time ordering "
                "(|character| < 1e-6); they are ordered by their position relative to mu.",
                stacklevel=2,
            )
            reference = 0.0 if green.mu is None else green.mu
            container.sigma_character[ambiguous] = np.where(zeta[ambiguous] > reference, 1e-6, -1e-6)
    green.info["galitskii_migdal_error"] = float(abs(gm_energy - E0))
    return green
