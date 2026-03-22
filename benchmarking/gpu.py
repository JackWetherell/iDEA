"""Benchmark GPU vs CPU scaling of the iDEA interacting solver.

Solves the QHO ground state for grid sizes 50–500 (steps of 50) and records
wall-clock time and memory usage for both the CPU and GPU paths.

Usage (from repo root):
    python benchmarking/gpu.py

Output:
    benchmarking/gpu_scaling.png
"""

import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil

import iDEA.interactions
import iDEA.methods.interacting
import iDEA.system


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qho(N: int) -> iDEA.system.System:
    """Return a QHO system with N grid points (mirrors iDEA.system.systems.qho)."""
    x = np.linspace(-10, 10, N)
    v_ext = 0.5 * (0.25 ** 2) * x ** 2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, "uu")


# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

GRID_SIZES = list(range(50, 501, 50))   # [50, 100, 150, ..., 500]

cpu_times = []
cpu_mems  = []
gpu_times = []
gpu_mems  = []

# ---------------------------------------------------------------------------
# CPU pass
# ---------------------------------------------------------------------------

proc = psutil.Process(os.getpid())

print("=== CPU benchmarks ===")
for N in GRID_SIZES:
    print(f"  N={N}  (Hamiltonian size {N**2}×{N**2})")
    s = make_qho(N)
    gc.collect()
    mem_before = proc.memory_info().rss

    t0 = time.perf_counter()
    try:
        state = iDEA.methods.interacting.solve(s, k=0, GPU=False)
        t1 = time.perf_counter()
        mem_after = proc.memory_info().rss
        cpu_times.append(t1 - t0)
        cpu_mems.append((mem_after - mem_before) / 1024 ** 3)
        del state
    except Exception as exc:
        t1 = time.perf_counter()
        print(f"    FAILED: {exc}")
        cpu_times.append(float("nan"))
        cpu_mems.append(float("nan"))

    gc.collect()

# ---------------------------------------------------------------------------
# GPU pass
# ---------------------------------------------------------------------------

print("\n=== GPU benchmarks ===")
try:
    import cupy as cp
    _has_cupy = True
except ImportError:
    print("  cupy not available – skipping GPU benchmarks")
    _has_cupy = False

for N in GRID_SIZES:
    if not _has_cupy:
        gpu_times.append(float("nan"))
        gpu_mems.append(float("nan"))
        continue

    print(f"  N={N}  (Hamiltonian size {N**2}×{N**2})")
    s = make_qho(N)
    cp.get_default_memory_pool().free_all_blocks()
    free_before, _ = cp.cuda.Device().mem_info()

    t0 = time.perf_counter()
    try:
        state = iDEA.methods.interacting.solve(s, k=0, GPU=True)
        t1 = time.perf_counter()
        free_after, _ = cp.cuda.Device().mem_info()
        gpu_times.append(t1 - t0)
        gpu_mems.append((free_before - free_after) / 1024 ** 3)
        del state
    except Exception as exc:
        t1 = time.perf_counter()
        print(f"    FAILED: {exc}")
        gpu_times.append(float("nan"))
        gpu_mems.append(float("nan"))

    cp.get_default_memory_pool().free_all_blocks()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COLOR_TIME = "steelblue"
COLOR_MEM  = "coral"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, times, mems, title in (
    (ax1, cpu_times, cpu_mems, "CPU (Intel i9)"),
    (ax2, gpu_times, gpu_mems, "GPU (RTX 4090)"),
):
    ax_mem = ax.twinx()

    (line_t,) = ax.plot(GRID_SIZES, times, "o-",  color=COLOR_TIME, label="Solve time (s)")
    (line_m,) = ax_mem.plot(GRID_SIZES, mems,  "s--", color=COLOR_MEM,  label="Memory (GB)")

    ax.set_title(title)
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Solve time (s)", color=COLOR_TIME)
    ax.tick_params(axis="y", labelcolor=COLOR_TIME)

    ax_mem.set_ylabel("Memory usage (GB)", color=COLOR_MEM)
    ax_mem.tick_params(axis="y", labelcolor=COLOR_MEM)

    ax.legend(handles=[line_t, line_m], loc="upper left")

fig.suptitle("iDEA interacting solver – scaling with grid size (QHO, 2 electrons, ground state)")
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "gpu_scaling.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")
plt.show()
