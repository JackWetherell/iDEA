"""Model Context Protocol (MCP) server exposing iDEA to AI agents.

Run with ``idea-mcp`` (or ``python -m iDEA.mcp_server``) after installing the optional extra:
``pip install "iDEA-latest[mcp]"``.

iDEA objects (systems, states and evolutions) hold large numpy arrays, so the server keeps them in an in-memory
registry and hands the client short IDs (``sys_1``, ``state_1``, ``evo_1``) that later tool calls refer to. Tools
return JSON summaries with (optionally downsampled) arrays.
"""

import argparse
import ast
import contextlib
import copy
import functools
import io
import itertools
import sys
from typing import Any

import numpy as np

import iDEA

try:  # mcp >= 2.0
    from mcp.server.mcpserver import Image
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # pragma: no cover - mcp 1.x
    try:
        from mcp.server.fastmcp import FastMCP as _Server
        from mcp.server.fastmcp import Image
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError as e:
        raise ImportError('The iDEA MCP server requires the "mcp" package: pip install "iDEA-latest[mcp]"') from e

__all__ = ["server", "main", "eval_expression", "ToolError"]


INSTRUCTIONS = """\
iDEA (interacting Dynamic Electrons Approach) solves the exact many-electron Schrodinger equation for 1D model
systems and compares it with approximate methods (non-interacting, Hartree, Hartree-Fock, LDA, hybrid, KS-SCE).

Conventions: Hartree atomic units; a 1D real-space grid; electrons are given as a spin string, e.g. "ud" (one up,
one down) or "uud"; interactions are softened Coulomb 1/(|x-x'|+a).

Typical workflow:
1. create_system (or load_builtin_system) -> system_id
2. solve(system_id, method) -> state_id and total energy
3. compute_observables(state_id, [...]) for densities, energies and potentials
4. optionally propagate(state_id, perturbation) for time dependence, or reverse_engineer(state_id) to find the
   exact Kohn-Sham potential
5. plot([...]) to look at the results

The exact "interacting" method costs ~points**N in memory and time: keep points <= ~300 for 2 electrons and
<= ~80 for 3 electrons. Approximate methods are cheap.
"""

server = _Server(name="iDEA", instructions=INSTRUCTIONS)

METHODS = {
    "interacting": "Exact many-body solution (exact exchange and correlation). Cost grows as points**N.",
    "non_interacting": "Non-interacting electrons in v_ext (also the Kohn-Sham solver).",
    "hartree": "Hartree theory (self-consistent mean field, no exchange).",
    "hartree_fock": "Hartree-Fock (exact exchange, no correlation).",
    "lda": "Local density approximation, parameterised from the 1D homogeneous electron gas.",
    "hybrid": "Hybrid functional: alpha * exact exchange + (1 - alpha) * LDA exchange-correlation.",
    "kssce": "Kohn-Sham strictly-correlated electrons (strong-correlation limit).",
}

OBSERVABLES = [
    "density",
    "spin_densities",
    "external_potential",
    "kinetic_energy",
    "external_energy",
    "hartree_potential",
    "hartree_energy",
    "exchange_potential",
    "exchange_energy",
    "lda_xc_potential",
    "lda_xc_energy",
    "orbitals",
]

_SAFE_FUNCTIONS = {
    name: getattr(np, name)
    for name in [
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "log10",
        "sqrt",
        "abs",
        "sign",
        "heaviside",
        "where",
        "minimum",
        "maximum",
    ]
}
_SAFE_CONSTANTS = {"pi": np.pi, "e": np.e}
_BIN_OPS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
}
_UNARY_OPS = {ast.USub: np.negative, ast.UAdd: np.positive}
_COMPARE_OPS = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}

_REGISTRY: dict[str, dict[str, Any]] = {}
_COUNTER = itertools.count(1)


def _tool(func):
    """Register func as an MCP tool.

    iDEA prints progress to stdout, which would corrupt the stdio transport, so it is redirected to stderr. Errors
    are re-raised as ToolError so that their message reaches the client instead of a generic failure.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                return func(*args, **kwargs)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return server.tool()(wrapper)


def eval_expression(expr: str, **variables: np.ndarray) -> np.ndarray:
    r"""
    Safely evaluate a numpy expression such as "-2/(abs(x)+1)" or "0.1*x*(t>0)".

    Only arithmetic, comparisons, numeric literals, the given variables, pi, e and a whitelist of numpy functions
    are allowed; anything else (attribute access, imports, names, keywords) raises ValueError.

    | Args:
    |     expr: str, Expression to evaluate.
    |     variables: np.ndarray, Arrays that the expression may reference by name (e.g. x, t).

    | Returns:
    |     value: np.ndarray, Evaluated expression.
    """
    if len(expr) > 1000:
        raise ValueError("Expression is too long (max 1000 characters).")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression {expr!r}: {e}") from e

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            if node.id in _SAFE_CONSTANTS:
                return _SAFE_CONSTANTS[node.id]
            raise ValueError(f"Unknown name {node.id!r}; allowed variables: {sorted(variables)} and pi, e.")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE_OPS:
            return _COMPARE_OPS[type(node.ops[0])](ev(node.left), ev(node.comparators[0])).astype(float)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _SAFE_FUNCTIONS
            and not node.keywords
        ):
            return _SAFE_FUNCTIONS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(
            f"Unsupported syntax in expression {expr!r}. Use arithmetic, comparisons and the functions "
            f"{sorted(_SAFE_FUNCTIONS)}."
        )

    with np.errstate(all="ignore"):
        value = np.asarray(ev(tree), dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Expression {expr!r} produced non-finite values on the grid.")
    return value


def _store(kind: str, obj: Any, **meta: Any) -> str:
    prefix = {"system": "sys", "state": "state", "evolution": "evo"}[kind]
    object_id = f"{prefix}_{next(_COUNTER)}"
    _REGISTRY[object_id] = {"kind": kind, "obj": obj, **meta}
    return object_id


def _get(object_id: str, kind: str | None = None) -> dict[str, Any]:
    if object_id not in _REGISTRY:
        raise ValueError(f"Unknown id {object_id!r}. Known ids: {sorted(_REGISTRY) or 'none'}.")
    entry = _REGISTRY[object_id]
    if kind is not None and entry["kind"] != kind:
        raise ValueError(f"{object_id!r} is a {entry['kind']}, expected a {kind}.")
    return entry


def _method(name: str):
    if name not in METHODS:
        raise ValueError(f"Unknown method {name!r}; choose from {sorted(METHODS)}.")
    return getattr(iDEA.methods, name)


def _array(a: np.ndarray, max_points: int | None = None) -> list:
    """JSON-friendly copy of a 1D/2D array, strided along the first axis down to max_points samples."""
    a = np.asarray(a)
    if max_points is not None and a.shape[0] > max_points:
        a = a[:: int(np.ceil(a.shape[0] / max_points))]
    if np.iscomplexobj(a):
        a = np.abs(a)
    return np.round(a.astype(float), 10).tolist()


def _system_summary(s: iDEA.system.System) -> dict[str, Any]:
    return {
        "electrons": s.electrons,
        "count": s.count,
        "up_count": s.up_count,
        "down_count": s.down_count,
        "points": int(s.x.shape[0]),
        "x_min": float(s.x[0]),
        "x_max": float(s.x[-1]),
        "dx": float(s.dx),
        "stencil": s.stencil,
        "v_ext_min": float(np.min(s.v_ext)),
        "many_body_dimension": int(s.x.shape[0]) ** s.count,
    }


def _total_energy(method: str, s: iDEA.system.System, state: Any, alpha: float) -> float:
    if method == "interacting":
        return float(state.energy)
    if method == "hybrid":
        return float(iDEA.methods.hybrid.total_energy(s, state, alpha=alpha))
    return float(_method(method).total_energy(s, state))


@_tool
def list_methods() -> dict[str, Any]:
    """List the solvers (methods) iDEA provides, the observables compute_observables can return, and the
    functions allowed in potential expressions."""
    return {
        "methods": METHODS,
        "observables": OBSERVABLES,
        "expression_functions": sorted(_SAFE_FUNCTIONS),
        "expression_constants": sorted(_SAFE_CONSTANTS),
    }


@_tool
def list_builtin_systems() -> dict[str, Any]:
    """List iDEA's built-in model systems (load one with load_builtin_system)."""
    return {
        "qho": {"description": "Two same-spin electrons ('uu') in a harmonic well, omega = 0.25, x in [-10, 10]."}
        | _system_summary(iDEA.system.systems.qho),
        "atom": {"description": "Two-electron ('ud') softened atom, v_ext = -2/(|x|+1), x in [-20, 20]."}
        | _system_summary(iDEA.system.systems.atom),
    }


@_tool
def load_builtin_system(name: str) -> dict[str, Any]:
    """Load one of iDEA's built-in systems ('qho' or 'atom') and return its system_id."""
    if name not in ("qho", "atom"):
        raise ValueError("name must be 'qho' or 'atom'.")
    s = copy.deepcopy(getattr(iDEA.system.systems, name))
    return {"system_id": _store("system", s, label=name)} | _system_summary(s)


@_tool
def create_system(
    electrons: str,
    v_ext: str,
    x_min: float = -10.0,
    x_max: float = 10.0,
    points: int = 100,
    interaction_strength: float = 1.0,
    softening: float = 1.0,
    stencil: int = 13,
    label: str | None = None,
) -> dict[str, Any]:
    """Define a 1D model system and return its system_id.

    Args:
        electrons: Spin of each electron, e.g. "ud" (two electrons, opposite spin), "uu", "uud".
        v_ext: External potential as a numpy expression in x (atomic units), e.g. "0.5*0.25**2*x**2",
            "-2/(abs(x)+1)" or "-1/(abs(x-3)+1) - 1/(abs(x+3)+1)". See list_methods for allowed functions.
        x_min, x_max, points: Uniform real-space grid. The grid must be wide enough for the density to decay.
        interaction_strength: Prefactor of the softened Coulomb interaction 1/(|x-x'|+softening); 0 for none.
        softening: Softening parameter of the interaction.
        stencil: Finite-difference stencil for the kinetic energy (3, 5, 7, 9, 11 or 13).
        label: Optional human-readable name used in plots.
    """
    if points < 10:
        raise ValueError("points must be at least 10.")
    if not (x_max > x_min):
        raise ValueError("x_max must be greater than x_min.")
    x = np.linspace(x_min, x_max, points)
    v = np.broadcast_to(eval_expression(v_ext, x=x), x.shape).copy()
    v_int = iDEA.interactions.softened_interaction(x, strength=interaction_strength, softening=softening)
    try:
        s = iDEA.system.System(x, v, v_int, electrons, stencil=stencil)
    except AssertionError as e:
        raise ValueError(str(e)) from e
    return {"system_id": _store("system", s, label=label or v_ext)} | _system_summary(s)


@_tool
def describe(object_id: str | None = None) -> dict[str, Any]:
    """Describe a stored system, state or evolution, or list every stored object when object_id is omitted."""
    if object_id is None:
        return {k: {kk: vv for kk, vv in v.items() if kk != "obj"} for k, v in _REGISTRY.items()}
    entry = _get(object_id)
    info = {k: v for k, v in entry.items() if k != "obj"}
    if entry["kind"] == "system":
        info |= _system_summary(entry["obj"])
    return info


@_tool
def solve(
    system_id: str,
    method: str = "interacting",
    k: int = 0,
    restricted: bool = False,
    mixing: float = 0.5,
    tol: float = 1e-10,
    alpha: float = 0.8,
    gpu: bool = False,
    allow_large: bool = False,
) -> dict[str, Any]:
    """Solve a system for its k-th energy eigenstate (k=0 is the ground state) with the chosen method.

    Args:
        system_id: System to solve.
        method: One of interacting (exact), non_interacting, hartree, hartree_fock, lda, hybrid, kssce.
        k: Excitation index (0 = ground state).
        restricted: Restricted (spin-paired) calculation, single-body methods only.
        mixing, tol: Self-consistency mixing and convergence tolerance, single-body methods only.
        alpha: Fraction of exact exchange, hybrid only.
        gpu: Use an NVIDIA GPU via CuPy, interacting only.
        allow_large: Permit interacting calculations with a many-body dimension above 5e6.
    """
    s = _get(system_id, "system")["obj"]
    module = _method(method)
    if method == "interacting":
        dim = s.x.shape[0] ** s.count
        if dim > 5e6 and not allow_large:
            raise ValueError(
                f"Many-body dimension points**N = {dim:.3g} is very large; reduce points or set allow_large=True."
            )
        state = module.solve(s, k=k, GPU=gpu)
    else:
        kwargs = {"k": k, "restricted": restricted, "mixing": mixing, "tol": tol, "silent": True}
        if method == "hybrid":
            kwargs["alpha"] = alpha
        state = module.solve(s, **kwargs)

    energy = _total_energy(method, s, state, alpha)
    state_id = _store("state", state, system_id=system_id, method=method, k=k, alpha=alpha, energy=energy)
    n = iDEA.observables.density(s, state=state)
    result = {
        "state_id": state_id,
        "method": method,
        "k": k,
        "total_energy": energy,
        "electron_count": float(np.sum(n) * s.dx),
    }
    if method != "interacting":
        for spin in ("up", "down"):
            channel = getattr(state, spin)
            occupied = list(channel.occupied)
            n_show = max(occupied, default=-1) + 3
            result[f"{spin}_orbital_energies"] = _array(channel.energies[:n_show])
            result[f"{spin}_occupations"] = _array(channel.occupations[:n_show])
    return result


@_tool
def compute_observables(
    state_id: str,
    quantities: list[str] | None = None,
    max_points: int = 200,
) -> dict[str, Any]:
    """Compute observables of a solved state. Arrays are returned on the grid "x", downsampled to max_points.

    Args:
        state_id: State from solve or reverse_engineer.
        quantities: Any of density, spin_densities, external_potential, kinetic_energy, external_energy,
            hartree_potential, hartree_energy, exchange_potential, exchange_energy, lda_xc_potential, lda_xc_energy,
            orbitals (single-body states only). Defaults to density and the energies.
        max_points: Maximum number of grid points per returned array.
    """
    entry = _get(state_id, "state")
    s = _get(entry["system_id"], "system")["obj"]
    state = entry["obj"]
    if quantities is None:
        quantities = ["density", "kinetic_energy", "external_energy", "hartree_energy", "exchange_energy"]
    unknown = sorted(set(quantities) - set(OBSERVABLES))
    if unknown:
        raise ValueError(f"Unknown quantities {unknown}; choose from {OBSERVABLES}.")

    n, up_n, down_n = iDEA.observables.density(s, state=state, return_spins=True)
    out: dict[str, Any] = {"x": _array(s.x, max_points), "total_energy": entry["energy"]}
    p = None
    if {"exchange_potential", "exchange_energy"} & set(quantities):
        p = iDEA.observables.density_matrix(s, state=state)
    for q in quantities:
        try:
            if q == "density":
                out[q] = _array(n, max_points)
            elif q == "spin_densities":
                out["up_density"] = _array(up_n, max_points)
                out["down_density"] = _array(down_n, max_points)
            elif q == "external_potential":
                out[q] = _array(s.v_ext, max_points)
            elif q == "kinetic_energy":
                out[q] = float(iDEA.observables.kinetic_energy(s, state=state))
            elif q == "external_energy":
                out[q] = float(iDEA.observables.external_energy(s, n, s.v_ext))
            elif q == "hartree_potential":
                out[q] = _array(iDEA.observables.hartree_potential(s, n), max_points)
            elif q == "hartree_energy":
                out[q] = float(iDEA.observables.hartree_energy(s, n, iDEA.observables.hartree_potential(s, n)))
            elif q == "exchange_potential":
                out[q] = _array(np.diag(iDEA.observables.exchange_potential(s, p)), max_points)
                out["exchange_potential_note"] = "Diagonal of the non-local exchange operator v_x(x, x')."
            elif q == "exchange_energy":
                out[q] = float(iDEA.observables.exchange_energy(s, p, iDEA.observables.exchange_potential(s, p)))
            elif q == "lda_xc_potential":
                out[q] = _array(iDEA.methods.lda.exchange_correlation_potential(s, n), max_points)
            elif q == "lda_xc_energy":
                out[q] = float(iDEA.methods.lda.exchange_correlation_energy(s, n))
            elif q == "orbitals":
                if entry["method"] == "interacting":
                    raise ValueError("orbitals are only defined for single-body (non-interacting/mean-field) states.")
                for spin in ("up", "down"):
                    channel = getattr(state, spin)
                    out[f"{spin}_occupied_orbitals"] = _array(channel.orbitals[:, channel.occupied], max_points)
        except NotImplementedError:
            out.setdefault("unavailable", {})[q] = f"iDEA does not implement {q} for {entry['method']} states."
    return out


@_tool
def propagate(
    state_id: str,
    v_ptrb: str,
    t_max: float,
    steps: int = 200,
    method: str | None = None,
    restricted: bool = False,
    gpu: bool = False,
    max_points: int = 100,
    max_times: int = 50,
) -> dict[str, Any]:
    """Propagate a state in time under a perturbing potential v_ptrb(x, t) added to v_ext.

    Args:
        state_id: Initial state (from solve).
        v_ptrb: Perturbation as a numpy expression in x and t, e.g. "-0.1*x" (sudden electric field) or
            "0.05*x*sin(0.5*t)".
        t_max: Final time (atomic units); the time grid is linspace(0, t_max, steps).
        steps: Number of time points.
        method: Method to propagate with; defaults to the method the state was solved with.
        restricted: Restricted propagation (single-body methods).
        gpu: Use the GPU (interacting only).
        max_points, max_times: Downsampling of the returned density evolution.

    Returns an evolution_id plus the dipole moment d(t) = integral x n(x,t) dx and the sampled density n(t, x).
    """
    entry = _get(state_id, "state")
    s = _get(entry["system_id"], "system")["obj"]
    method = method or entry["method"]
    module = _method(method)
    if (method == "interacting") != (entry["method"] == "interacting"):
        raise ValueError(
            "Interacting states must be propagated with 'interacting', and single-body states with a single-body method."
        )
    if not 2 <= steps <= 20000:
        raise ValueError("steps must be between 2 and 20000.")
    t = np.linspace(0.0, t_max, steps)
    v = eval_expression(v_ptrb, x=s.x[np.newaxis, :], t=t[:, np.newaxis])
    v = np.broadcast_to(v, (t.shape[0], s.x.shape[0])).copy()
    if method == "interacting":
        evolution = module.propagate(s, entry["obj"], v, t, GPU=gpu)
    elif method == "hybrid":
        evolution = module.propagate(s, entry["obj"], v, t, restricted=restricted, alpha=entry["alpha"])
    else:
        evolution = module.propagate(s, entry["obj"], v, t, restricted=restricted)
    n_t = iDEA.observables.density(s, evolution=evolution)
    dipole = np.sum(n_t * s.x[np.newaxis, :], axis=1) * s.dx
    evolution_id = _store("evolution", evolution, system_id=entry["system_id"], state_id=state_id, method=method)
    times = np.unique(np.linspace(0, steps - 1, min(steps, max_times)).astype(int))
    return {
        "evolution_id": evolution_id,
        "method": method,
        "t": _array(t[times]),
        "dipole": _array(dipole[times]),
        "x": _array(s.x, max_points),
        "density": [_array(n_t[j], max_points) for j in times],
        "electron_count_final": float(np.sum(n_t[-1]) * s.dx),
    }


@_tool
def reverse_engineer(
    target_state_id: str,
    method: str = "non_interacting",
    mu: float = 1.0,
    pe: float = 0.1,
    tol: float = 1e-8,
    max_points: int = 200,
) -> dict[str, Any]:
    """Find the fictitious potential that makes `method` reproduce the target state's density.

    With method="non_interacting" and an interacting target this is the exact Kohn-Sham potential; the returned
    v_hxc = v_ks - v_ext is the exact Hartree-exchange-correlation potential. The fictitious system and its
    state are stored and their ids returned.

    Args:
        target_state_id: State whose density should be reproduced (typically an interacting ground state).
        method: Single-body method used for the fictitious system (non_interacting, hartree, hartree_fock, lda,
            hybrid).
        mu, pe: Parameters of the iterative update v -> v + mu*(n**pe - n_target**pe).
        tol: Convergence tolerance on integral |n - n_target| dx. Tighter values take longer.
        max_points: Downsampling of the returned arrays.
    """
    entry = _get(target_state_id, "state")
    system_id = entry["system_id"]
    s = _get(system_id, "system")["obj"]
    if method in ("interacting", "kssce"):
        raise ValueError("reverse_engineer requires a single-body method other than kssce.")
    module = _method(method)
    target_n = iDEA.observables.density(s, state=entry["obj"])
    s_fict = iDEA.reverse_engineering.reverse(s, target_n, module, mu=mu, pe=pe, tol=tol, silent=True)
    state = module.solve(s_fict, silent=True)
    n = iDEA.observables.density(s_fict, state=state)
    fict_id = _store("system", s_fict, label=f"{method} reverse-engineered from {target_state_id}", parent=system_id)
    energy = _total_energy(method, s_fict, state, 0.8)
    state_id = _store("state", state, system_id=fict_id, method=method, k=0, alpha=0.8, energy=energy)
    return {
        "fictitious_system_id": fict_id,
        "fictitious_state_id": state_id,
        "x": _array(s.x, max_points),
        "v_fictitious": _array(s_fict.v_ext, max_points),
        "v_hxc" if method == "non_interacting" else "v_correction": _array(s_fict.v_ext - s.v_ext, max_points),
        "density_error": float(np.sum(np.abs(n - target_n)) * s.dx),
        "up_orbital_energies": _array(state.up.energies[: s.up_count + 2]),
        "down_orbital_energies": _array(state.down.energies[: s.down_count + 2]),
    }


@_tool
def plot(object_ids: list[str], title: str | None = None, time_index: int = -1, save_path: str | None = None) -> Image:
    """Plot stored objects on one figure and return a PNG image.

    Systems are drawn as their external potential (dashed), states as their density, and evolutions as their
    density at time_index (default: final time). Optionally also writes the PNG to save_path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax_n = plt.subplots(figsize=(7, 4.5), dpi=110)
    ax_v = None
    for i, object_id in enumerate(object_ids):
        color = f"C{i % 10}"
        entry = _get(object_id)
        if entry["kind"] == "system":
            s = entry["obj"]
            ax_v = ax_v or ax_n.twinx()
            ax_v.plot(s.x, s.v_ext, "--", color=color, label=f"v_ext {object_id} ({entry.get('label', '')})")
            continue
        s = _get(entry["system_id"], "system")["obj"]
        if entry["kind"] == "state":
            n = iDEA.observables.density(s, state=entry["obj"])
            ax_n.plot(s.x, n, color=color, label=f"n {object_id} ({entry['method']})")
        else:
            evolution = entry["obj"]
            n = iDEA.observables.density(s, evolution=evolution, time_indices=np.array([time_index]))[0]
            ax_n.plot(s.x, n, color=color, label=f"n {object_id} ({entry['method']}, t={evolution.t[time_index]:.3g})")
    ax_n.set_xlabel("x (a.u.)")
    ax_n.set_ylabel("density n(x)")
    handles, labels = ax_n.get_legend_handles_labels()
    if ax_v is not None:
        ax_v.set_ylabel("potential (Ha)")
        h, lab = ax_v.get_legend_handles_labels()
        handles, labels = handles + h, labels + lab
    if handles:
        ax_n.legend(handles, labels, fontsize=8, loc="best")
    if title:
        ax_n.set_title(title)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    if save_path:
        with open(save_path, "wb") as f:
            f.write(buffer.getvalue())
    return Image(data=buffer.getvalue(), format="png")


@_tool
def save(object_id: str, path: str) -> dict[str, Any]:
    """Save a stored system or state to a file with iDEA's own save functions (load it in Python with
    iDEA.system.load_system / iDEA.state.load_many_body_state / iDEA.state.load_single_body_state)."""
    entry = _get(object_id)
    obj = entry["obj"]
    if entry["kind"] == "system":
        iDEA.system.save_system(obj, path)
        loader = "iDEA.system.load_system"
    elif isinstance(obj, iDEA.state.ManyBodyState):
        iDEA.state.save_many_body_state(obj, path)
        loader = "iDEA.state.load_many_body_state"
    elif isinstance(obj, iDEA.state.SingleBodyState):
        iDEA.state.save_single_body_state(obj, path)
        loader = "iDEA.state.load_single_body_state"
    else:
        raise ValueError("Only systems and states can be saved.")
    return {"saved": path, "load_with": loader}


@_tool
def clear(object_id: str | None = None) -> dict[str, Any]:
    """Delete one stored object, or every stored object when object_id is omitted, to free memory."""
    if object_id is None:
        removed = sorted(_REGISTRY)
        _REGISTRY.clear()
    else:
        _get(object_id)
        del _REGISTRY[object_id]
        removed = [object_id]
    return {"removed": removed}


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: run the iDEA MCP server."""
    parser = argparse.ArgumentParser(prog="idea-mcp", description="Run the iDEA MCP server.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transports.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports.")
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        server.run("stdio")
    elif hasattr(server, "settings"):  # mcp 1.x configures HTTP via settings
        server.settings.host, server.settings.port = args.host, args.port
        server.run(args.transport)
    else:
        server.run(args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
