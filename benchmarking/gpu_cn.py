"""Benchmark Crank-Nicolson (cuDSS) against the exponential propagators of the iDEA interacting solver.

Compares four ways of propagating the same many-body state, for the QHO with two
electrons under a suddenly applied uniform field:

    CPU expm    scipy expm_multiply             propagate(GPU=False)               (CPU default)
    GPU Taylor  scaled truncated Taylor series  propagate(GPU=True)                (GPU default)
    CPU CN      Crank-Nicolson, SuperLU         propagate(GPU=False, method="cn")
    GPU CN      Crank-Nicolson, cuDSS           propagate(GPU=True, method="cn")

Two sweeps are run, each written to its own figure:

1. Grid size (gpu_cn_scaling.png). For each N every propagator advances the
   ground state over the same time grid. Crank-Nicolson pays a one-off
   factorisation and then a cheap step, while the exponentials pay the same cost
   every step; the two are separated by also timing a one-step run. Panels: total
   time, time per step, one-off setup, peak memory, density error against CPU
   expm, and the speedup of GPU CN.

2. Time step (gpu_cn_dt.png). At a fixed N every propagator runs to the same
   final time with a range of dt. The field is static, so the exact final state
   does not depend on dt and CPU expm at the smallest dt is the reference.
   Crank-Nicolson's error falls as dt^2 while its cost grows as 1/dt; the
   work-precision panel shows which propagator reaches a given accuracy soonest.

CPU CN is capped at CPU_CN_MAX_N: SuperLU's fill-in with the default 13-point
stencil makes its factorisation take minutes beyond that. It is here as the CPU
reference for the same scheme, not as a contender.

The ground state is solved once per grid size (on the GPU when available) and
the Hamiltonian is built once per backend, so only propagation is timed. The GPU
points are skipped without cupy, and the GPU CN points without nvmath-python.

Usage (from repo root):
    python benchmarking/gpu_cn.py

Output:
    benchmarking/gpu_cn_scaling.png
    benchmarking/gpu_cn_dt.png
"""

import contextlib
import gc
import io
import os
import time

import _bench
import matplotlib.pyplot as plt
import numpy as np

import iDEA.methods.interacting
import iDEA.observables

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    import nvmath.sparse.advanced  # noqa: F401

    HAS_NVMATH = True
except Exception:
    HAS_NVMATH = False


# Grid-size sweep.
GRID_SIZES = list(range(50, 401, 50))
TIME_LENGTH = 1.0
TIME_STEPS = 101
CPU_CN_MAX_N = 200
# Time indices at which densities are compared against CPU expm.
ERROR_TIME_INDICES = np.linspace(0, TIME_STEPS - 1, 11).astype(int)

# Time-step sweep.
DT_SWEEP_N = 150
DT_SWEEP_TIME = 1.0
DT_VALUES = [0.04, 0.02, 0.01, 0.005, 0.0025]

# key, legend label, on the GPU, propagate method, colour, line style
METHODS = [
    ("cpu_expm", "CPU expm", False, "expm", _bench.COLOR_CPU, "o-"),
    ("gpu_taylor", "GPU Taylor", True, "expm", _bench.COLOR_GPU, "o-"),
    ("cpu_cn", "CPU CN (SuperLU)", False, "cn", _bench.COLOR_CPU, "s--"),
    ("gpu_cn", "GPU CN (cuDSS)", True, "cn", _bench.COLOR_GPU, "s--"),
]
CN_KEYS = ("cpu_cn", "gpu_cn")


def available(key: str, N: int) -> bool:
    """Whether this propagator can run on this machine at grid size N."""
    if key == "gpu_taylor":
        return HAS_CUPY
    if key == "gpu_cn":
        return HAS_CUPY and HAS_NVMATH
    if key == "cpu_cn":
        return N <= CPU_CN_MAX_N
    return True


def quiet(fn):
    """Run fn with iDEA's progress bars and messages suppressed."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn()


def hamiltonians(s):
    """The many-body Hamiltonian on each available backend."""
    H = {False: iDEA.methods.interacting.hamiltonian(s, GPU=False)}
    if HAS_CUPY:
        H[True] = iDEA.methods.interacting.hamiltonian(s, GPU=True)
    return H


def propagate(s, state, v_ptrb, t, H, gpu, method):
    """Propagate quietly, returning (evolution, wall-clock seconds)."""
    t0 = time.perf_counter()
    evolution = quiet(lambda: iDEA.methods.interacting.propagate(s, state, v_ptrb, t, H=H[gpu], GPU=gpu, method=method))
    if gpu:
        cp.cuda.Device().synchronize()
    return evolution, time.perf_counter() - t0


def timed_propagate(s, state, v_ptrb, t, H, gpu, method, time_indices=None):
    """Propagate, returning (density at time_indices, seconds, peak memory in GB: GPU VRAM if gpu else CPU RAM).

    Only the propagation is timed. The density is computed afterwards (it antisymmetrises the wavefunction at every
    requested time index, which would otherwise swamp the time per step), and is None if no time_indices are given.
    """
    gc.collect()
    probe = _bench.peak_vram_gb if gpu else _bench.peak_rss_gb
    (evolution, elapsed), mem = probe(lambda: propagate(s, state, v_ptrb, t, H, gpu, method))
    if time_indices is None:
        return None, elapsed, mem
    return iDEA.observables.density(s, evolution=evolution, time_indices=time_indices), elapsed, mem


def warm_up():
    """Run each GPU propagator once on a tiny system, so library loading and kernel compilation are not timed."""
    if not HAS_CUPY:
        return
    print("Warming up the GPU...", flush=True)
    s = _bench.make_qho(20)
    state = quiet(lambda: iDEA.methods.interacting.solve(s, k=0))
    t = np.linspace(0, 0.02, 3)
    v_ptrb = _bench.make_perturbation(s, t)
    H = hamiltonians(s)
    for key, _label, gpu, method, _colour, _style in METHODS:
        if gpu and available(key, 20):
            propagate(s, state, v_ptrb, t, H, gpu, method)


def max_error(n, n_ref):
    """Largest absolute difference between two densities (nan if either run failed)."""
    return float("nan") if n is None or n_ref is None else float(np.max(abs(n - n_ref)))


def nan_if_zero(x):
    """Errors of exactly zero (the reference against itself) cannot be shown on a log axis."""
    return float("nan") if x == 0 else x


def plot_methods(ax, x, results, field, keys=None):
    """Plot one result field against x for each propagator (or those in keys) that has any data."""
    for key, label, gpu, _method, colour, style in METHODS:
        if keys is not None and key not in keys:
            continue
        ys = [r[field] for r in results[key]]
        if np.all(np.isnan(ys)):
            continue
        device = hw.gpu_label() if gpu else hw.cpu_label()
        ax.plot(x, ys, style, color=colour, label=f"{label} ({device})")


hw = _bench.Hardware()

t = np.linspace(0, TIME_LENGTH, TIME_STEPS)
steps = TIME_STEPS - 1
dt = t[1] - t[0]

_bench.print_preamble(
    title="interacting solver, Crank-Nicolson (cuDSS) vs exponential propagation",
    hw=hw,
    sweep=GRID_SIZES,
    detail=(
        f"QHO, 2 electrons (uu), {steps} timesteps of dt={dt:.3g}, sudden uniform field; "
        f"then dt = {', '.join(str(d) for d in DT_VALUES)} at N={DT_SWEEP_N}"
    ),
)
if HAS_CUPY and not HAS_NVMATH:
    print("  nvmath-python (cuDSS) not installed - GPU CN points will be skipped: pip install -e '.[gpu]'\n")

warm_up()


# ----------------------------------------------------------------------------------------------------------------------
# Sweep 1: grid size.
# ----------------------------------------------------------------------------------------------------------------------
nan_row = {"total": float("nan"), "per_step": float("nan"), "setup": float("nan"), "mem": float("nan")}
scaling = {key: [] for key, *_ in METHODS}

for N in GRID_SIZES:
    print(f"\n  N={N}  (Hamiltonian size {N**2}x{N**2}, {steps} timesteps)", flush=True)
    s = _bench.make_qho(N)
    v_ptrb = _bench.make_perturbation(s, t)
    try:
        state = quiet(lambda system=s: iDEA.methods.interacting.solve(system, k=0, GPU=HAS_CUPY))
        H = hamiltonians(s)
    except Exception as exc:
        print(f"    ground state FAILED: {exc}")
        for key in scaling:
            scaling[key].append(dict(nan_row, error=float("nan"), density=None))
        continue

    for key, label, gpu, method, _colour, _style in METHODS:
        if not available(key, N):
            scaling[key].append(dict(nan_row, error=float("nan"), density=None))
            continue
        try:
            # A one-step run measures the one-off setup (the factorisation for CN) plus one step.
            _, t_one, _ = timed_propagate(s, state, v_ptrb[:2], t[:2], H, gpu, method)
            density, t_full, mem = timed_propagate(s, state, v_ptrb, t, H, gpu, method, ERROR_TIME_INDICES)
            per_step = (t_full - t_one) / (steps - 1)
            if per_step <= 0:
                # Below timing resolution: the step is lost in the run-to-run noise of the one-off setup.
                per_step = float("nan")
            setup = t_one if np.isnan(per_step) else max(t_one - per_step, 0.0)
            row = {"total": t_full, "per_step": per_step, "setup": setup, "mem": mem}
            print(
                f"    {label:17s} total {t_full:8.2f} s   per step {per_step * 1e3:9.2f} ms   "
                f"setup {row['setup']:7.2f} s   peak {'VRAM' if gpu else 'RAM '} {mem:6.2f} GB",
                flush=True,
            )
        except Exception as exc:
            print(f"    {label:17s} FAILED: {exc}", flush=True)
            row, density = dict(nan_row), None
        scaling[key].append(dict(row, density=density))
        if HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()

    reference = scaling["cpu_expm"][-1]["density"]
    for key in scaling:
        scaling[key][-1]["error"] = max_error(scaling[key][-1]["density"], reference)
        scaling[key][-1]["density"] = None
    del state, H
    gc.collect()


# ----------------------------------------------------------------------------------------------------------------------
# Summary.
# ----------------------------------------------------------------------------------------------------------------------
def break_even(cn, other):
    """Steps after which CN's one-off setup is paid back by its cheaper step (inf if its step is not cheaper)."""
    if np.isnan(cn["setup"]) or np.isnan(other["per_step"]) or other["per_step"] <= cn["per_step"]:
        return float("inf")
    return cn["setup"] / (other["per_step"] - cn["per_step"])


print()
print("  Time per step (ms), one-off CN setup (s), and steps for GPU CN to break even")
print(
    f"  {'N':>5} {'CPU expm':>10} {'GPU Taylor':>11} {'CPU CN':>9} {'GPU CN':>9} {'GPU CN setup':>13}"
    f" {'vs CPU expm':>12} {'vs GPU Taylor':>14}"
)
for i, N in enumerate(GRID_SIZES):
    row = {key: scaling[key][i] for key in scaling}
    per_step = [f"{row[k]['per_step'] * 1e3:.2f}" if not np.isnan(row[k]["per_step"]) else "-" for k in scaling]
    setup = f"{row['gpu_cn']['setup']:.2f}" if not np.isnan(row["gpu_cn"]["setup"]) else "-"
    be = [break_even(row["gpu_cn"], row[k]) for k in ("cpu_expm", "gpu_taylor")]
    be = ["-" if np.isnan(row["gpu_cn"]["setup"]) else ("never" if np.isinf(b) else f"{b:.0f}") for b in be]
    print(
        f"  {N:5d} {per_step[0]:>10} {per_step[1]:>11} {per_step[2]:>9} {per_step[3]:>9} {setup:>13} {be[0]:>12} {be[1]:>14}"
    )
print(flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Figure 1: grid-size sweep.
# ----------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
(ax_total, ax_step, ax_setup), (ax_mem, ax_err, ax_speed) = axes

plot_methods(ax_total, GRID_SIZES, scaling, "total")
ax_total.set_title(f"Total propagation time ({steps} steps, including setup)")
ax_total.set_ylabel("Time (s)")

plot_methods(ax_step, GRID_SIZES, scaling, "per_step")
ax_step.set_title("Time per step (excluding setup)")
ax_step.set_ylabel("Time (s)")

plot_methods(ax_setup, GRID_SIZES, scaling, "setup", keys=CN_KEYS)
ax_setup.set_title("One-off Crank-Nicolson setup (factorisation)")
ax_setup.set_ylabel("Time (s)")

plot_methods(ax_mem, GRID_SIZES, scaling, "mem")
ax_mem.set_title("Peak memory (CPU RAM / GPU VRAM)")
ax_mem.set_ylabel("Memory (GB)")

plot_methods(ax_err, GRID_SIZES, scaling, "error", keys=("gpu_taylor", "cpu_cn", "gpu_cn"))
ax_err.set_title(f"Max density error vs CPU expm (dt={dt:.3g})")
ax_err.set_ylabel("max |n - n_ref|")

for key, colour in (("cpu_expm", _bench.COLOR_CPU), ("gpu_taylor", _bench.COLOR_GPU)):
    speedup = [o["total"] / c["total"] for o, c in zip(scaling[key], scaling["gpu_cn"])]
    if not np.all(np.isnan(speedup)):
        label = dict((k, lbl) for k, lbl, *_ in METHODS)[key]
        ax_speed.plot(GRID_SIZES, speedup, "s--", color=colour, label=f"over {label}")
ax_speed.axhline(1.0, color="grey", lw=0.8)
ax_speed.set_title(f"GPU CN speedup in total time ({steps} steps)")
ax_speed.set_ylabel("Speedup (x)")

for ax in axes.flat:
    ax.set_xlabel("Number of grid points")
    if ax is not ax_mem:
        ax.set_yscale("log")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

fig.suptitle(
    f"iDEA interacting propagation - Crank-Nicolson vs exponential, scaling with grid size (QHO, 2 electrons, {steps} timesteps)"
)
plt.tight_layout()
out_scaling = os.path.join(os.path.dirname(__file__), "gpu_cn_scaling.png")
plt.savefig(out_scaling, dpi=150, bbox_inches="tight")
print(f"Figure saved to {out_scaling}")


# ----------------------------------------------------------------------------------------------------------------------
# Sweep 2: time step.
# ----------------------------------------------------------------------------------------------------------------------
print(f"\n  dt sweep at N={DT_SWEEP_N} to t={DT_SWEEP_TIME}", flush=True)
s = _bench.make_qho(DT_SWEEP_N)
try:
    state = quiet(lambda: iDEA.methods.interacting.solve(s, k=0, GPU=HAS_CUPY))
    H = hamiltonians(s)
except Exception as exc:
    print(f"    ground state FAILED: {exc}")
    state = None
dt_sweep = {key: [] for key, *_ in METHODS}
for dt_value in DT_VALUES:
    t_dt = np.linspace(0, DT_SWEEP_TIME, int(round(DT_SWEEP_TIME / dt_value)) + 1)
    v_dt = _bench.make_perturbation(s, t_dt)
    for key, label, gpu, method, _colour, _style in METHODS:
        if state is None or not available(key, DT_SWEEP_N):
            dt_sweep[key].append({"time": float("nan"), "final": None})
            continue
        try:
            density, elapsed, _ = timed_propagate(s, state, v_dt, t_dt, H, gpu, method, np.array([t_dt.shape[0] - 1]))
            dt_sweep[key].append({"time": elapsed, "final": density[0]})
            print(f"    dt={dt_value:<7g} {label:17s} {elapsed:8.2f} s", flush=True)
        except Exception as exc:
            print(f"    dt={dt_value:<7g} {label:17s} FAILED: {exc}", flush=True)
            dt_sweep[key].append({"time": float("nan"), "final": None})

# The field is static, so expm is exact for any dt: its smallest-dt run is the reference for every run.
reference = dt_sweep["cpu_expm"][int(np.argmin(DT_VALUES))]["final"]
for key in dt_sweep:
    for r in dt_sweep[key]:
        r["error"] = nan_if_zero(max_error(r["final"], reference))


# ----------------------------------------------------------------------------------------------------------------------
# Figure 2: time-step sweep.
# ----------------------------------------------------------------------------------------------------------------------
fig, (ax_err, ax_time, ax_wp) = plt.subplots(1, 3, figsize=(18, 5.5))

plot_methods(ax_err, DT_VALUES, dt_sweep, "error")
for key in CN_KEYS:
    errors = np.array([r["error"] for r in dt_sweep[key]])
    if not np.all(np.isnan(errors)):
        i = int(np.nanargmax(errors))
        guide = errors[i] * (np.array(DT_VALUES) / DT_VALUES[i]) ** 2
        ax_err.plot(DT_VALUES, guide, ":", color="grey", label="dt^2")
        break
ax_err.set_title(f"Final density error vs dt (t={DT_SWEEP_TIME})")
ax_err.set_xlabel("dt")
ax_err.set_ylabel("max |n - n_ref|")

plot_methods(ax_time, DT_VALUES, dt_sweep, "time")
ax_time.set_title(f"Time to solution (t={DT_SWEEP_TIME}, including setup)")
ax_time.set_xlabel("dt")
ax_time.set_ylabel("Time (s)")

for key, label, gpu, _method, colour, style in METHODS:
    errors = [r["error"] for r in dt_sweep[key]]
    times = [r["time"] for r in dt_sweep[key]]
    if np.all(np.isnan(errors)) or np.all(np.isnan(times)):
        continue
    device = hw.gpu_label() if gpu else hw.cpu_label()
    ax_wp.plot(times, errors, style, color=colour, label=f"{label} ({device})")
ax_wp.set_title("Work-precision (each point is one dt)")
ax_wp.set_xlabel("Time to solution (s)")
ax_wp.set_ylabel("max |n - n_ref|")

for ax in (ax_err, ax_time, ax_wp):
    ax.set_xscale("log")
    ax.set_yscale("log")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

fig.suptitle(f"iDEA interacting propagation - accuracy and cost vs time step (QHO, 2 electrons, N={DT_SWEEP_N})")
plt.tight_layout()
out_dt = os.path.join(os.path.dirname(__file__), "gpu_cn_dt.png")
plt.savefig(out_dt, dpi=150, bbox_inches="tight")
print(f"Figure saved to {out_dt}")

plt.show()
