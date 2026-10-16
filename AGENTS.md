# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, Copilot, etc.) working with **iDEA** — the interacting
Dynamic Electrons Approach. It covers both *using* iDEA in a workflow and *developing* this repository.

## What iDEA is

A Python library for exact and approximate many-electron quantum mechanics in **one dimension**. It solves the
static and time-dependent many-body Schrödinger equation exactly (including all exchange and correlation) and
compares the result with approximate methods: non-interacting electrons, Hartree, Hartree-Fock, LDA, hybrid
functionals and KS-SCE. It can also reverse-engineer the exact Kohn-Sham potential of a density. It is used for
research into density functional theory / many-body perturbation theory and for teaching.

Reach for iDEA when a task involves model atoms or molecules, Kohn-Sham / exchange-correlation potentials,
benchmarking a density-functional approximation against an exact answer, or the time evolution of a few
interacting electrons.

## Install

Python **3.10+** is required.

```bash
pip install iDEA-latest            # users
pip install "iDEA-latest[mcp]"     # + MCP server for agents
pip install -e ".[dev,mcp]"        # developing this repo
pip install -e ".[gpu]"            # optional NVIDIA GPU support (CuPy) for the exact solver
```

## Conventions

- **Hartree atomic units** throughout (energies in Hartree, lengths in Bohr, ħ = mₑ = e = 1).
- One spatial dimension on a uniform grid `x`; `dx` is derived from it.
- Electrons are given as a spin string: `"ud"` = one up + one down, `"uu"` = two up, `"uud"`, … .
- Interaction `v_int` is a 2D array `v_int[x, x']`, normally the softened Coulomb interaction
  `iDEA.interactions.softened_interaction(x)` = 1/(|x − x′| + 1).
- Time-dependent perturbations are arrays indexed `v_ptrb[time, space]`.
- Single-body states: `state.up` / `state.down` each have `energies[n]`, `orbitals[space, n]`, `occupations[n]`,
  `occupied`. Many-body states: `state.space`, `state.spin`, `state.full`, `state.energy`.
- Every method module in `iDEA.methods` exposes `solve(s, k=0, ...)` and `propagate(s, state, v_ptrb, t, ...)`;
  `iDEA.iterate_methods` lists them so you can loop over methods generically.

## Python API cheat sheet

```python
import numpy as np
import iDEA

# Define a system (or use the built-ins iDEA.system.systems.atom / .qho).
x = np.linspace(-20, 20, 300)
s = iDEA.system.System(x, -2.0 / (abs(x) + 1.0), iDEA.interactions.softened_interaction(x), "ud")

# Exact ground state.
gs = iDEA.methods.interacting.solve(s, k=0)        # GPU=True to use CuPy
n = iDEA.observables.density(s, state=gs)
E = gs.energy

# Approximate methods (single-body, self-consistent).
lda = iDEA.methods.lda.solve(s, silent=True)
E_lda = iDEA.methods.lda.total_energy(s, lda)
n_lda = iDEA.observables.density(s, state=lda)

# Exact Kohn-Sham potential that reproduces the exact density.
s_ks = iDEA.reverse_engineering.reverse(s, n, iDEA.methods.non_interacting, silent=True)
v_hxc = s_ks.v_ext - s.v_ext

# Time evolution under a perturbation (here a sudden electric field).
t = np.linspace(0, 5, 500)
v_ptrb = -0.1 * np.tile(x, (t.shape[0], 1))       # v_ptrb[time, space]
evo = iDEA.methods.interacting.propagate(s, gs, v_ptrb, t)
n_t = iDEA.observables.density(s, evolution=evo)   # n_t[time, space]

# Persistence.
iDEA.system.save_system(s, "atom.sys")
iDEA.state.save_many_body_state(gs, "atom_gs.mbs")
```

The [tutorial notebook](tutorial/tutorial.ipynb) and the
[API docs](https://idea-interacting-dynamic-electrons-approach.readthedocs.io/en/latest/) go further.

### Cost of the exact solver

The exact (`interacting`) solver works in a space of dimension `points**N`, so memory and time grow
exponentially with the number of electrons. Rough guide: ≤ ~300 points for 2 electrons and ≤ ~80 points for
3 electrons on a workstation; use `GPU=True` for larger problems. The approximate methods are cheap at any
grid size. Prefer small grids for exploration and refine once the workflow works.

## MCP server

`iDEA/mcp_server.py` is a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes the
workflow above as tools, so agents can run calculations without writing Python.

Run it: `idea-mcp` (stdio, the default) or `idea-mcp --transport streamable-http --port 8000`.
`python -m iDEA.mcp_server` also works.

Register it with a client:

- **Claude Code** — this repo ships a project-scoped [`.mcp.json`](.mcp.json), so the server is offered
  automatically when Claude Code is opened here. Elsewhere: `claude mcp add idea -- idea-mcp`.
- **Any stdio MCP client** (Claude Desktop, Cursor, Codex, …):
  ```json
  {"mcpServers": {"idea": {"command": "idea-mcp"}}}
  ```

Tools:

| Tool | Purpose |
|---|---|
| `list_methods` | Available methods, observables and functions allowed in potential expressions |
| `list_builtin_systems` / `load_builtin_system` | iDEA's built-in `atom` and `qho` systems |
| `create_system` | Define a system from a spin string and a `v_ext` expression in `x` (e.g. `"-2/(abs(x)+1)"`) |
| `solve` | Solve with any method → `state_id`, total energy, orbital energies |
| `compute_observables` | Densities, potentials and energy components of a state |
| `propagate` | Time evolution under `v_ptrb(x, t)` (e.g. `"-0.1*x"`) → dipole d(t) and n(x, t) |
| `reverse_engineer` | Exact Kohn-Sham (or other fictitious) potential reproducing a state's density |
| `plot` | PNG of potentials and densities of stored objects |
| `describe`, `save`, `clear` | Inspect, persist (to iDEA's own file format) and free stored objects |

Objects live in server memory and are referred to by IDs (`sys_1`, `state_2`, `evo_3`). Potentials are given
as numpy expressions that are parsed with a whitelist (arithmetic, comparisons, `pi`, `e` and functions such as
`sin`, `exp`, `abs`, `sqrt`, `tanh`, `where`); arbitrary Python is rejected.

Typical session: `create_system` → `solve(method="interacting")` → `solve(method="lda")` →
`compute_observables` on both → `reverse_engineer` the exact state → `plot` all of them.

## Developing this repository

Layout:

```
iDEA/
  system.py              System class, built-in systems, save/load
  interactions.py        softened Coulomb interactions
  state.py               single-/many-body states and evolutions, save/load
  observables.py         density, density matrix, energies, potentials
  methods/               interacting, non_interacting, hartree, hartree_fock, lda, hybrid, kssce
  reverse_engineering.py reverse(), reverse_propagation()
  mcp_server.py          MCP server (optional, needs the "mcp" extra)
tests/                   pytest suite
benchmarking/            CPU vs GPU scaling scripts
tutorial/                Jupyter tutorials
docs/                    Sphinx docs (docs/_build is generated output — do not hand-edit)
```

Commands (run before every commit):

```bash
ruff check iDEA/ tests/ benchmarking/
ruff format iDEA/ tests/ benchmarking/
pytest -v -k "not Long"        # fast suite (~1 min); TestLong takes much longer
```

- `tests/test_gpu.py` is skipped without CuPy and `tests/test_mcp_server.py` without `mcp`.
- Line length is 120; ruff targets Python 3.10 (`X | Y` unions and builtin generics are expected).
- Import order in `iDEA/__init__.py` matters: `lda` must be imported before `hybrid`.
- Keep public functions documented in the existing docstring style (`| Args:` / `| Returns:` blocks), since
  Sphinx builds the API docs from them.
- Approximate methods are CPU-only by design; only `iDEA.methods.interacting` takes `GPU=`.
- Don't commit generated plots, pickled systems/states, or benchmark outputs.
