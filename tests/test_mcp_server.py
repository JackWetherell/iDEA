"""Tests for the iDEA MCP server (iDEA/mcp_server.py). Skipped unless the optional "mcp" extra is installed."""

import asyncio

import numpy as np
import pytest

pytest.importorskip("mcp")

import iDEA  # noqa: E402
from iDEA import mcp_server  # noqa: E402

POINTS = 60


@pytest.fixture(scope="module")
def atom():
    """Small softened two-electron atom, solved exactly."""
    system = mcp_server.create_system("ud", "-2/(abs(x)+1)", x_min=-10, x_max=10, points=POINTS, label="atom")
    state = mcp_server.solve(system["system_id"], "interacting")
    return system, state


def test_expression_accepts_physics():
    x = np.linspace(-1, 1, 5)
    assert np.allclose(mcp_server.eval_expression("-2/(abs(x)+1)", x=x), -2 / (abs(x) + 1))
    assert np.allclose(
        mcp_server.eval_expression("0.5*x**2*(x>0) + sin(pi*x)", x=x), 0.5 * x**2 * (x > 0) + np.sin(np.pi * x)
    )


@pytest.mark.parametrize(
    "expr", ["__import__('os')", "x.__class__", "open('f')", "np.sin(x)", "y", "10**10**10", "x if x else 1"]
)
def test_expression_rejects_unsafe(expr):
    with pytest.raises(ValueError):
        mcp_server.eval_expression(expr, x=np.ones(3))


def test_create_system_validation():
    with pytest.raises(mcp_server.ToolError):
        mcp_server.create_system("ux", "x**2")
    with pytest.raises(mcp_server.ToolError):
        mcp_server.create_system("ud", "x**2", stencil=4)


@pytest.mark.parametrize("method", ["non_interacting", "hartree", "hartree_fock", "lda", "hybrid"])
def test_solve_matches_direct_api(atom, method):
    system, _ = atom
    result = mcp_server.solve(system["system_id"], method)
    s = mcp_server._get(system["system_id"], "system")["obj"]
    module = getattr(iDEA.methods, method)
    expected = module.total_energy(s, module.solve(s, silent=True))
    assert result["total_energy"] == pytest.approx(expected)
    assert result["electron_count"] == pytest.approx(2.0)


def test_interacting_matches_direct_api(atom):
    system, state = atom
    s = mcp_server._get(system["system_id"], "system")["obj"]
    assert state["total_energy"] == pytest.approx(iDEA.methods.interacting.solve(s, k=0).energy)


def test_interacting_size_guard():
    system = mcp_server.create_system("uud", "x**2", points=200)
    with pytest.raises(mcp_server.ToolError, match="allow_large"):
        mcp_server.solve(system["system_id"], "interacting")


def test_observables(atom):
    _, state = atom
    out = mcp_server.compute_observables(
        state["state_id"], ["density", "spin_densities", "hartree_energy", "kinetic_energy"], max_points=30
    )
    assert len(out["density"]) == len(out["x"]) == 30
    assert out["hartree_energy"] > 0
    assert "kinetic_energy" in out["unavailable"]


def test_propagate(atom):
    _, state = atom
    out = mcp_server.propagate(state["state_id"], "-0.1*x", t_max=1.0, steps=20, max_times=5)
    assert out["evolution_id"].startswith("evo_")
    assert len(out["t"]) == len(out["dipole"]) == len(out["density"]) == 5
    assert out["electron_count_final"] == pytest.approx(2.0)
    assert out["dipole"][-1] > out["dipole"][0]


def test_reverse_engineer(atom):
    _, state = atom
    out = mcp_server.reverse_engineer(state["state_id"], tol=1e-6)
    assert out["density_error"] < 1e-6
    assert mcp_server.describe(out["fictitious_system_id"])["kind"] == "system"


def test_plot_save_and_clear(atom, tmp_path):
    system, state = atom
    image = mcp_server.plot([system["system_id"], state["state_id"]], save_path=str(tmp_path / "plot.png"))
    assert image.data.startswith(b"\x89PNG")
    assert (tmp_path / "plot.png").exists()
    saved = mcp_server.save(state["state_id"], str(tmp_path / "state.pkl"))
    assert saved["load_with"] == "iDEA.state.load_many_body_state"
    extra = mcp_server.load_builtin_system("qho")["system_id"]
    assert mcp_server.clear(extra) == {"removed": [extra]}
    with pytest.raises(mcp_server.ToolError):
        mcp_server.describe(extra)


def test_tools_registered():
    tools = asyncio.run(mcp_server.server.list_tools())
    names = {tool.name for tool in tools}
    assert {"create_system", "solve", "compute_observables", "propagate", "reverse_engineer", "plot"} <= names
