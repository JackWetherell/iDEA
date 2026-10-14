"""
Generate every figure, table and number quoted in paper/main.tex.

Usage (from the repository root):

    python paper/scripts/make_figures.py            # everything
    python paper/scripts/make_figures.py fig1 fig3  # selected parts

Outputs are written to paper/figures/. Intermediate results are cached in
paper/figures/cache/ so that individual parts can be re-run cheaply; delete the
cache directory to force a full recalculation.
"""

import json
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import iDEA  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "paper", "figures")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "tests"))

# Validated categorical palette (colour-blind safe on a white background); the
# exact result is always black. Line styles provide a second, colour-free cue.
EXACT = dict(color="black", lw=1.6, ls="-", label="Exact")
METHOD_STYLE = {
    "non_interacting": dict(color="#2a78d6", ls=(0, (1, 1.2)), label="Non-interacting"),
    "hartree": dict(color="#eb6834", ls=(0, (4, 1.5)), label="Hartree"),
    "hartree_fock": dict(color="#1baf7a", ls="-", label="Hartree-Fock"),
    "lda": dict(color="#eda100", ls=(0, (6, 1.5, 1, 1.5)), label="LDA"),
    "hybrid": dict(color="#e87ba4", ls=(0, (2.5, 1.2)), label=r"Hybrid ($\alpha=0.8$)"),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
COLUMN = 3.4  # inches, single column of the elsarticle 5p layout

NUMBERS = {}


def number(name, value, fmt):
    NUMBERS[name] = fmt.format(value)


def cached(name, fn):
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    t0 = time.time()
    data = fn()
    data["_seconds"] = time.time() - t0
    np.savez(path, **data)
    print(f"  computed {name} in {data['_seconds']:.1f} s", flush=True)
    return data


def atom_system(num_points=181, length=24.0, electrons="uu"):
    """Softened 1D atom with nuclear charge 2: two spin-polarised electrons."""
    x = np.linspace(-length / 2, length / 2, num_points)
    v_ext = -2.0 / (np.abs(x) + 1.0)
    return iDEA.system.System(x, v_ext, iDEA.interactions.softened_interaction(x), electrons=electrons)


# --------------------------------------------------------------------------------------
# Validation and timing (Table 1)
# --------------------------------------------------------------------------------------
def validation():
    import test_manybody
    import test_time_evolution

    rows = {}

    def manybody():
        out = {}
        for spin in ["ud", "uu"]:
            p = test_manybody.short_params
            ref = test_manybody.Groundstate.get_analytic_groundstate(p, spin)
            calc = test_manybody.Groundstate.get_iDEA_groundstate(p, spin)
            dx = p["length"] / (p["num_points"] - 1)
            overlap = abs(np.sum(np.conj(ref.wavefunction) * calc.wavefunction)) * dx**2
            norm = np.sqrt(np.sum(abs(ref.wavefunction) ** 2) * np.sum(abs(calc.wavefunction) ** 2)) * dx**2
            out[spin] = np.array(
                [
                    calc.total_energy,
                    abs(calc.total_energy - ref.total_energy),
                    np.max(abs(calc.density - ref.density)),
                    1.0 - overlap / norm,
                ]
            )
        return out

    mb = cached("validation_manybody", manybody)
    for spin in ["ud", "uu"]:
        rows[spin] = mb[spin]

    def td():
        p = test_time_evolution.short_params
        ref = test_time_evolution.Groundstate.get_analytic_groundstate(p, "u")
        calc = test_time_evolution.Groundstate.get_iDEA_groundstate(p, "u")
        return {"max_dn": np.max(abs(calc.density - ref.density))}

    tdv = cached("validation_td", td)

    def hpt():
        # Harmonic-potential theorem: lowest dipole excitation of interacting
        # electrons in a harmonic well is exactly omega.
        omega = 0.25
        x = np.linspace(-10, 10, 151)
        s = iDEA.system.System(x, 0.5 * omega**2 * x**2, iDEA.interactions.softened_interaction(x), electrons="uu")
        e0 = iDEA.methods.interacting.solve(s, k=0).energy
        e1 = iDEA.methods.interacting.solve(s, k=1).energy
        return {"omega": omega, "gap": e1 - e0}

    h = cached("validation_hpt", hpt)

    def hpt_dynamic():
        # Dynamic harmonic-potential theorem: under a uniform field E(t) the interacting density
        # translates rigidly and its centre of mass obeys the classical equation x'' = -w^2 x - E(t).
        omega, e0, w_drive = 0.25, 0.02, 0.4
        x = np.linspace(-10, 10, 101)
        s = iDEA.system.System(x, 0.5 * omega**2 * x**2, iDEA.interactions.softened_interaction(x), electrons="uu")
        gs = iDEA.methods.interacting.solve(s, k=0)
        out = {}
        for tag, steps in [("", 601), ("_half", 1201)]:
            t = np.linspace(0, 30, steps)
            v_ptrb = e0 * np.sin(w_drive * t)[:, None] * x[None, :]
            n = iDEA.observables.density(s, evolution=iDEA.methods.interacting.propagate(s, gs, v_ptrb, t))
            x_cm = n @ x * s.dx / s.count
            exact = -e0 / (omega**2 - w_drive**2) * (np.sin(w_drive * t) - w_drive / omega * np.sin(omega * t))
            out["err" + tag] = np.max(abs(x_cm - exact))
            out["dt" + tag] = t[1] - t[0]
        out["amp"] = np.max(abs(exact))
        return out

    hd = cached("validation_hpt_dynamic", hpt_dynamic)

    def convergence():
        energies, dxs = [], []
        for num_points in [121, 181, 241]:
            s = atom_system(num_points)
            energies.append(iDEA.methods.interacting.solve(s, k=0).energy)
            dxs.append(s.dx)
        return {"E": np.array(energies), "dx": np.array(dxs)}

    conv = cached("validation_convergence", convergence)

    def timing():
        out = {}
        for electrons, grids in [("uu", [60, 100, 140, 180, 220]), ("uuu", [20, 30, 40])]:
            times = []
            for num_points in grids:
                x = np.linspace(-10, 10, num_points)
                s = iDEA.system.System(
                    x, 0.5 * 0.25**2 * x**2, iDEA.interactions.softened_interaction(x), electrons=electrons
                )
                t0 = time.time()
                iDEA.methods.interacting.solve(s, k=0)
                times.append(time.time() - t0)
            out[electrons] = np.array([grids, times])
        return out

    tim = cached("validation_timing", timing)

    # numbers
    for spin in ["ud", "uu"]:
        E, dE, dn, infid = rows[spin]
        tag = "Singlet" if spin == "ud" else "Triplet"
        number(f"E{tag}", E, "{:.6f}")
        number(f"dE{tag}", dE, "{:.1e}")
        number(f"dn{tag}", dn, "{:.1e}")
        number(f"inf{tag}", max(infid, 0.0), "{:.1e}")
    number("dnTD", float(tdv["max_dn"]), "{:.1e}")
    number("hptOmega", float(h["omega"]), "{:.2f}")
    number("hptGap", float(h["gap"]), "{:.8f}")
    number("hptErr", abs(float(h["gap"]) - float(h["omega"])), "{:.1e}")
    number("hptdErr", float(hd["err"]), "{:.1e}")
    number("hptdAmp", float(hd["amp"]), "{:.2f}")
    number("hptdDt", float(hd["dt"]), "{:.2f}")
    number("hptdErrHalf", float(hd["err_half"]), "{:.1e}")
    number("hptdDtHalf", float(hd["dt_half"]), "{:.3f}")
    for i, tag in enumerate(["A", "B", "C"]):
        number(f"convE{tag}", float(conv["E"][i]), "{:.5f}")
        number(f"convdx{tag}", float(conv["dx"][i]), "{:.3f}")

    ncpu = os.cpu_count()
    with open(os.path.join(OUT, "table_validation.tex"), "w") as f:
        f.write("% Generated by paper/scripts/make_figures.py -- do not edit by hand.\n")
        f.write("\\begin{table}[t]\n\\centering\\footnotesize\n")
        f.write("\\caption{Validation of the exact solver and indicative cost. ")
        f.write(
            "(a) Two electrons in a harmonic well ($\\omega=0.3275$, $N_x=231$, $L=20$) "
            "compared with the separated centre-of-mass/relative-coordinate reference "
            "on the same grid: absolute energy error, maximum density error and "
            "wavefunction infidelity $1-|\\langle\\Psi_\\mathrm{ref}|\\Psi\\rangle|$. "
            "(b) Lowest dipole excitation energy versus the harmonic-potential theorem, "
            "and the time-dependent density error over 500 steps. "
            "(c) Grid convergence of the ground-state energy of the atom in Fig.~\\ref{fig:gs}. "
            f"(d) Wall time of \\texttt{{interacting.solve}} on {ncpu} CPU cores of the test machine.}}\n"
        )
        f.write("\\label{tab:validation}\n\\begin{tabular}{@{}lll@{}}\n\\toprule\n")
        f.write("\\multicolumn{3}{@{}l}{(a) Two-electron ground state} \\\\\n")
        f.write(" & singlet (\\texttt{ud}) & triplet (\\texttt{uu}) \\\\\n\\midrule\n")
        f.write(f"$E$ (Ha) & {NUMBERS['ESinglet']} & {NUMBERS['ETriplet']} \\\\\n")
        f.write(f"$|\\Delta E|$ & {sci(NUMBERS['dESinglet'])} & {sci(NUMBERS['dETriplet'])} \\\\\n")
        f.write(f"$\\max|\\Delta n|$ & {sci(NUMBERS['dnSinglet'])} & {sci(NUMBERS['dnTriplet'])} \\\\\n")
        f.write(f"infidelity & {sci(NUMBERS['infSinglet'])} & {sci(NUMBERS['infTriplet'])} \\\\\n\\midrule\n")
        f.write("\\multicolumn{3}{@{}l}{(b) Excitations and dynamics} \\\\\n")
        f.write(
            f"$E_1-E_0$ vs $\\omega={NUMBERS['hptOmega']}$ & {NUMBERS['hptGap']} & "
            f"error {sci(NUMBERS['hptErr'])} \\\\\n"
        )
        f.write(f"TD density, $\\max|\\Delta n(x,t)|$ & {sci(NUMBERS['dnTD'])} & \\\\\n")
        f.write(
            f"driven HPT, $\\max|\\Delta x_\\mathrm{{cm}}|$, $\\delta t={NUMBERS['hptdDt']}$ & "
            f"{sci(NUMBERS['hptdErr'])} & (amplitude {NUMBERS['hptdAmp']}) \\\\\n"
            f"\\quad $\\delta t={NUMBERS['hptdDtHalf']}$ & {sci(NUMBERS['hptdErrHalf'])} & \\\\\n\\midrule\n"
        )
        f.write("\\multicolumn{3}{@{}l}{(c) Grid convergence, atom (\\texttt{uu})} \\\\\n")
        for tag in ["A", "B", "C"]:
            f.write(f"$\\delta x={NUMBERS['convdx' + tag]}$ & $E={NUMBERS['convE' + tag]}$ & \\\\\n")
        f.write("\\midrule\n\\multicolumn{3}{@{}l}{(d) Ground-state solve time} \\\\\n")
        for electrons, label in [("uu", "$N=2$"), ("uuu", "$N=3$")]:
            grids, times = tim[electrons]
            grid_cells = ", ".join(str(int(g)) for g in grids)
            time_cells = ", ".join(f"{t:.1f}" for t in times)
            f.write(f"{label}: $N_x$ & \\multicolumn{{2}}{{l}}{{{grid_cells}}} \\\\\n")
            f.write(f"\\quad time (s) & \\multicolumn{{2}}{{l}}{{{time_cells}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    for electrons in ["uu", "uuu"]:
        grids, times = tim[electrons]
        number(f"timeMax{'Two' if electrons == 'uu' else 'Three'}", float(times[-1]), "{:.0f}")
        number(f"gridMax{'Two' if electrons == 'uu' else 'Three'}", int(grids[-1]), "{:d}")


def sci(s):
    """Format '1.2e-11' as LaTeX scientific notation."""
    mantissa, exponent = s.split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


# --------------------------------------------------------------------------------------
# Figure 1: ground-state densities of all methods
# --------------------------------------------------------------------------------------
def fig1():
    s = atom_system()

    def compute():
        gs = iDEA.methods.interacting.solve(s, k=0)
        out = {"n_exact": iDEA.observables.density(s, state=gs), "E_exact": gs.energy}
        for m in iDEA.iterate_sb_methods:
            st = m.solve(s, mixing=0.2, tol=1e-10, silent=True)
            out["n_" + m.name] = iDEA.observables.density(s, state=st)
            out["E_" + m.name] = m.total_energy(s, st)
        return out

    d = cached("fig1", compute)
    n0 = d["n_exact"]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 3.3), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    axes[0].plot(s.x, n0, **EXACT)
    for m in iDEA.iterate_sb_methods:
        st = METHOD_STYLE[m.name]
        axes[0].plot(s.x, d["n_" + m.name], color=st["color"], ls=st["ls"], label=st["label"])
        axes[1].plot(s.x, d["n_" + m.name] - n0, color=st["color"], ls=st["ls"])
        dn = np.sum(abs(d["n_" + m.name] - n0)) * s.dx
        number("E" + m.name.replace("_", ""), float(d["E_" + m.name]), "{:.4f}")
        number("dn" + m.name.replace("_", ""), dn, "{:.3f}")
    axes[1].axhline(0, color="0.6", lw=0.5)
    axes[1].set_yscale("symlog", linthresh=1e-3, linscale=0.6)
    axes[1].set_yticks([-1e-1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1])
    number("Eexact", float(d["E_exact"]), "{:.4f}")
    number("fignx", s.x.shape[0], "{:d}")
    number("figdx", s.dx, "{:.3f}")
    axes[0].set_ylabel(r"$n(x)$")
    axes[1].set_ylabel(r"$n(x)-n_\mathrm{exact}(x)$")
    axes[1].set_xlabel(r"$x$ (a.u.)")
    axes[0].set_xlim(-8, 8)
    axes[0].legend(loc="upper left", ncol=1, handlelength=2.6)
    for ax, lab in zip(axes, "ab"):
        ax.set_title(f"({lab})", loc="left", fontsize=8, pad=2)
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(OUT, "fig1_densities.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 2: exact Kohn-Sham and xc potentials by reverse engineering
# --------------------------------------------------------------------------------------
def fig2():
    s = atom_system()
    s2 = atom_system(electrons="ud")

    def compute():
        d1 = dict(np.load(os.path.join(CACHE, "fig1.npz")))
        n = d1["n_exact"]
        ks = iDEA.reverse_engineering.reverse(s, n, iDEA.methods.non_interacting, mu=1.0, pe=0.1, tol=1e-8, silent=True)
        v_h = iDEA.observables.hartree_potential(s, n)
        out = {
            "v_ks": ks.v_ext,
            "v_h": v_h,
            "v_xc": ks.v_ext - s.v_ext - v_h,
            "v_xc_lda": iDEA.methods.lda.exchange_correlation_potential(s, n),
        }
        st = iDEA.methods.non_interacting.solve(ks, silent=True)
        out["dn_re"] = np.sum(abs(iDEA.observables.density(ks, state=st) - n)) * s.dx
        gs2 = iDEA.methods.interacting.solve(s2, k=0)
        n2 = iDEA.observables.density(s2, state=gs2)
        ks2 = iDEA.reverse_engineering.reverse(
            s2, n2, iDEA.methods.non_interacting, mu=1.0, pe=0.1, tol=1e-8, silent=True
        )
        v_h2 = iDEA.observables.hartree_potential(s2, n2)
        out.update({"n2": n2, "E2": gs2.energy, "v_h2": v_h2, "v_xc2": ks2.v_ext - s2.v_ext - v_h2})
        return out

    d = cached("fig2", compute)

    n1 = dict(np.load(os.path.join(CACHE, "fig1.npz")))["n_exact"]
    n2 = d["n2"]

    def ip_gauge(system, v_ks, energy):
        """Constant that makes the highest occupied KS eigenvalue equal minus the exact ionisation energy."""
        ks = iDEA.system.System(system.x, v_ks, system.v_int, electrons=system.electrons)
        st = iDEA.methods.non_interacting.solve(ks, silent=True)
        eps = np.concatenate([st.up.energies[st.up.occupied], st.down.energies[st.down.occupied]])
        ion = iDEA.system.System(system.x, system.v_ext, system.v_int, electrons="u")
        e_ion = iDEA.methods.non_interacting.solve(ion, silent=True).up.energies[0]  # one remaining electron
        return (energy - e_ion) - np.max(eps), e_ion - energy

    c1, ip1 = ip_gauge(s, d["v_ks"], float(dict(np.load(os.path.join(CACHE, "fig1.npz")))["E_exact"]))
    c2, ip2 = ip_gauge(s2, d["v_ks"] * 0 + (d["v_xc2"] + s2.v_ext + d["v_h2"]), float(d["E2"]))
    v_ks1, v_xc1 = d["v_ks"] + c1, d["v_xc"] + c1
    v_xc2 = d["v_xc2"] + c2
    v_x2 = -0.5 * d["v_h2"]  # exact exchange of a two-electron singlet
    v_c2 = v_xc2 - v_x2

    def mask(v, n, floor=1e-5):
        # The inverted potential is undetermined where the density is negligible.
        return np.where(n > floor, v, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 3.6), sharex=True)
    ax = axes[0]
    ax.plot(s.x, s.v_ext, color="0.55", lw=1.0, ls=(0, (4, 1.5)), label=r"$v_\mathrm{ext}$")
    ax.plot(s.x, mask(v_ks1, n1), color="black", lw=1.6, label=r"$v_\mathrm{KS}$ (exact)")
    ax.plot(s.x, mask(v_xc1, n1), color="#2a78d6", lw=1.4, label=r"$v_\mathrm{xc}$ (exact)")
    ax.plot(s.x, d["v_xc_lda"], color="#eda100", ls=(0, (6, 1.5, 1, 1.5)), label=r"$v_\mathrm{xc}^\mathrm{LDA}[n]$")
    ax.set_ylabel("potential (Ha)")
    ax.legend(loc="lower right", handlelength=2.6)
    ax.text(0.02, 0.06, "(a) triplet", transform=ax.transAxes)

    ax = axes[1]
    ax.plot(s2.x, mask(v_xc2, n2), color="#2a78d6", lw=1.4, label=r"$v_\mathrm{xc}$ (exact)")
    ax.plot(s2.x, v_x2, color="#eb6834", ls=(0, (4, 1.5)), label=r"$v_\mathrm{x}=-v_\mathrm{H}/2$")
    ax.plot(s2.x, mask(v_c2, n2), color="#1baf7a", lw=1.2, label=r"$v_\mathrm{c}=v_\mathrm{xc}-v_\mathrm{x}$")
    ax.set_xlabel(r"$x$ (a.u.)")
    ax.set_ylabel("potential (Ha)")
    ax.set_xlim(-8, 8)
    ax.legend(loc="lower right", handlelength=2.6)
    ax.text(0.02, 0.06, "(b) singlet", transform=ax.transAxes)
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(OUT, "fig2_potentials.pdf"))
    plt.close(fig)

    number("dnRE", sci(f"{float(d['dn_re']):.1e}").strip("$"), "{}")
    number("Esinglet", float(d["E2"]), "{:.4f}")
    number("ipTriplet", float(ip1), "{:.3f}")
    number("ipSinglet", float(ip2), "{:.3f}")
    core = n2 > 1e-3
    number("vcMin", float(np.min(v_c2[core])), "{:.3f}")
    number("vcMax", float(np.max(v_c2[core])), "{:.3f}")
    for xi, tag in [(0.0, "Zero"), (-4.0, "Four"), (-8.0, "Eight")]:
        i = np.argmin(abs(s.x - xi))
        number("vxcExact" + tag, float(v_xc1[i]), "{:.3f}")
        number("vxcLDA" + tag, float(d["v_xc_lda"][i]), "{:.3f}")
        number("vxcAsym" + tag, float(-1.0 / (abs(s.x[i]) + 1.0)), "{:.3f}")
        number("nAt" + tag, sci(f"{n1[i]:.0e}").strip("$"), "{}")


# --------------------------------------------------------------------------------------
# Figure 3: driven dynamics
# --------------------------------------------------------------------------------------
TD_FIELD = 0.05
TD_PULSE = 1.0
TD_TMAX = 100.0
TD_STEPS = 1001


def fig3():
    s = atom_system(num_points=121)
    t = np.linspace(0, TD_TMAX, TD_STEPS)
    v_ptrb = np.zeros((t.shape[0], s.x.shape[0]))
    # Short uniform-field pulse, then free evolution. The step ending at t_j uses v_ptrb[j], so the
    # field acts over exactly TD_PULSE when it is set for 0 < t_j <= TD_PULSE.
    v_ptrb[(t > 0) & (t <= TD_PULSE + 1e-9), :] = TD_FIELD * s.x

    def compute():
        out = {"t": t}
        gs = iDEA.methods.interacting.solve(s, k=0)
        t0 = time.time()
        ev = iDEA.methods.interacting.propagate(s, gs, v_ptrb, t)
        out["time_exact"] = time.time() - t0
        n = iDEA.observables.density(s, evolution=ev)
        out["d_exact"] = n @ s.x * s.dx
        for m in iDEA.iterate_sb_methods[1:]:
            st = m.solve(s, mixing=0.2, tol=1e-10, silent=True)
            ev = m.propagate(s, st, v_ptrb, t)
            n = iDEA.observables.density(s, evolution=ev)
            out["d_" + m.name] = n @ s.x * s.dx
        return out

    d = cached("fig3", compute)
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    ax.plot(t, d["d_exact"] - d["d_exact"][0], **EXACT)
    for m in iDEA.iterate_sb_methods[1:]:
        st = METHOD_STYLE[m.name]
        y = d["d_" + m.name]
        ax.plot(t, y - y[0], color=st["color"], ls=st["ls"], label=st["label"])
    ax.set_xlabel(r"$t$ (a.u.)")
    ax.set_ylabel(r"$d(t)-d(0)$")
    ax.set_xlim(0, TD_TMAX)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22), handlelength=2.4)
    fig.savefig(os.path.join(OUT, "fig3_dynamics.pdf"))
    plt.close(fig)
    number("tdField", TD_FIELD, "{:.2f}")
    number("tdTmax", TD_TMAX, "{:.0f}")
    number("tdPulse", TD_PULSE, "{:.0f}")
    number("tdSteps", TD_STEPS - 1, "{:d}")
    number("tdNx", s.x.shape[0], "{:d}")
    number("tdTime", float(d["time_exact"]), "{:.0f}")
    number("tdDt", float(t[1] - t[0]), "{:.1f}")
    number("tdAmp", float(np.max(abs(d["d_exact"] - d["d_exact"][0]))), "{:.2f}")


PARTS = {"validation": validation, "fig1": fig1, "fig2": fig2, "fig3": fig3}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(PARTS)
    numbers_json = os.path.join(CACHE, "numbers.json")
    if os.path.exists(numbers_json):
        with open(numbers_json) as f:
            NUMBERS.update(json.load(f))
    for name in selected:
        print(f"[{name}]", flush=True)
        PARTS[name]()
    with open(numbers_json, "w") as f:
        json.dump(NUMBERS, f, indent=1, sort_keys=True)
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        f.write("% Generated by paper/scripts/make_figures.py -- do not edit by hand.\n")
        for key, value in sorted(NUMBERS.items()):
            f.write(f"\\newcommand{{\\num{key}}}{{{value}}}\n")
    print("done", flush=True)
