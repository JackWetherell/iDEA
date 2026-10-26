"""
Figures and numbers for the technical report on the exact Green's function and self-energy (report/latex).

Computes the exact Green's function and self-energy of the one-dimensional hydrogen atom, and its GW
approximations, and writes every figure of the report to report/latex/figures/ together with values.tex, which
defines the numbers quoted in the text. Re-running with a finer grid (for example on a GPU) and rebuilding the
report updates both:

    python report/make_figures.py                                       # nx = 61, a few minutes on a laptop
    python report/make_figures.py --nx 151 --dense-max 30000 --gpu      # production grid
    make -C report/latex
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

import iDEA  # noqa: E402

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latex", "figures")
ARROWS = {"up": r"\uparrow", "down": r"\downarrow"}
WIDTH = 6.6

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nx", type=int, default=61, help="Number of grid points for hydrogen.")
    parser.add_argument("--length", type=float, default=12.0, help="Half-width of the box for hydrogen.")
    parser.add_argument("--solver", default="auto", choices=["auto", "dense", "krylov"], help="Sector solver.")
    parser.add_argument("--dense-max", type=int, default=6000, help="Largest sector to diagonalise completely.")
    parser.add_argument("--gpu", action="store_true", help="Diagonalise the sectors on the GPU (cupy).")
    parser.add_argument("--eta", type=float, default=0.05, help="Broadening of real-frequency quantities.")
    return parser.parse_args()


def heatmap(ax, x, data, cmap="RdBu_r", symmetric=True, vmax=None, extent=None, **kwargs):
    """Draw a two-point quantity (or a space-time map) with a colour bar."""
    extent = [x[0], x[-1], x[0], x[-1]] if extent is None else extent
    if vmax is None:
        vmax = np.max(abs(data)) if symmetric else np.max(data)
    vmax = max(vmax, 1e-300)
    norm = TwoSlopeNorm(0.0, -vmax, vmax) if symmetric else None
    image = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        vmin=None if symmetric else 0.0,
        vmax=None if symmetric else vmax,
        aspect="auto",
        **kwargs,
    )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return image


def save(fig, name):
    fig.savefig(os.path.join(FIGURES, name))
    plt.close(fig)
    print(f"wrote {name}")


def spectral_functions(s, green, eta):
    omega = np.linspace(-1.0, 1.5, 500)
    fig, axes = plt.subplots(
        1, 3, figsize=(WIDTH, 2.3), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 0.8]}
    )
    for ax, spin in zip(axes[:2], ("up", "down")):
        A = iDEA.observables.spectral_function(s, green, omega, eta=eta, spin=spin, diagonal=True)
        heatmap(ax, s.x, A.T, cmap="magma", symmetric=False, vmax=np.percentile(A, 99.5),
                extent=[omega[0], omega[-1], s.x[0], s.x[-1]])  # fmt: skip
        ax.axvline(green.mu, color="w", ls="--", lw=0.7)
        ax.set_xlabel(r"$\omega$")
        ax.set_title(rf"$A_{ARROWS[spin]}(x,x;\omega)$")
        axes[2].plot(omega, A.sum(axis=1) * s.dx, label=f"${ARROWS[spin]}$", lw=1.0)
    axes[0].set_ylabel("$x$")
    axes[2].axvline(green.mu, color="0.5", ls="--", lw=0.7)
    axes[2].set_xlabel(r"$\omega$")
    axes[2].set_title(r"$\int A(x,x;\omega)\,dx$")
    axes[2].set_yscale("log")
    axes[2].legend()
    save(fig, "spectral_functions.pdf")


def self_screening(s, green, qp, phi):
    nu = np.linspace(0.0, 3.0, 150)
    fig, ax = plt.subplots(figsize=(3.3, 2.4), layout="constrained")
    exact = [np.linalg.norm(S @ phi) * s.dx for S in iDEA.observables.self_energy(s, green, nu=nu, part="c")]
    ax.semilogy(nu, np.maximum(exact, 1e-18), "k-", label="exact")
    residuals = {}
    for mode, state in qp.items():
        phi_gw = state.up.orbitals[:, state.up.occupations > 0][:, 0]
        sigma_c = iDEA.observables.self_energy(s, state, nu=nu, part="c")
        residuals[mode] = [np.linalg.norm(S @ phi_gw) * s.dx for S in sigma_c]
        ax.semilogy(nu, residuals[mode], label=mode.replace("0", r"$_0$"))
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\Vert \Sigma_{c\uparrow}(\mu+i\nu)\,\phi \Vert$")
    ax.legend(ncol=2)
    save(fig, "self_screening.pdf")
    return float(np.max(exact)), {mode: float(np.max(r)) for mode, r in residuals.items()}


def static_correlation(s, green, gw):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.0), layout="constrained")
    depths = {}
    for row, (name, state) in enumerate((("exact", green), ("GW", gw))):
        for col, spin in enumerate(("up", "down")):
            sigma = np.real(iDEA.observables.self_energy(s, state, nu=np.array([0.0]), part="c", spin=spin)[0])
            depths[(name, spin)] = float(np.min(sigma))
            heatmap(axes[row, col], s.x, sigma)
            axes[row, col].set_title(rf"{name} $\mathrm{{Re}}\,\Sigma_{{c{ARROWS[spin]}}}(x,x';\mu+i0)$")
            axes[row, col].set_xlabel("$x'$")
            axes[row, col].set_ylabel("$x$")
    save(fig, "static_correlation.pdf")
    return depths


def injected_electron(s, green):
    t = np.linspace(1e-6, 25.0, 251)
    i0 = np.argmin(abs(s.x))
    near = abs(s.x) < 3
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), layout="constrained")
    for ax, spin, ls in zip(axes, ("up", "down"), ("--", "-")):
        G = iDEA.observables.green_function(s, green, t=t, spin=spin, cutoff=0.5)[:, :, i0]
        heatmap(ax, s.x, abs(G), cmap="magma", symmetric=False, vmax=np.percentile(abs(G), 99.5),
                extent=[s.x[0], s.x[-1], t[0], t[-1]])  # fmt: skip
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        ax.set_title(rf"$|G_{ARROWS[spin]}(x,0;t)|$")
        fraction = np.sum(abs(G[:, near]) ** 2, axis=1) / np.sum(abs(G) ** 2, axis=1)
        axes[2].plot(t, fraction, ls, color="k", lw=1.0, label=f"${ARROWS[spin]}$")
    axes[2].set_xlabel("$t$")
    axes[2].set_title(r"weight within $|x|<3$")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()
    save(fig, "injected_electron.pdf")


def frequency_frames(s, green, eta, bound):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.0), layout="constrained")
    G = iDEA.observables.green_function(s, green, omega=np.array([bound]), eta=eta, spin="down")[0]
    sigma = iDEA.observables.self_energy(s, green, omega=np.array([0.5]), eta=eta, spin="down")[0]
    panels = (
        (np.real(G), rf"$\mathrm{{Re}}\,G_\downarrow(x,x';\omega={bound:.3f})$"),
        (np.imag(G), rf"$\mathrm{{Im}}\,G_\downarrow(x,x';\omega={bound:.3f})$"),
        (np.real(sigma), r"$\mathrm{Re}\,\Sigma_{xc\downarrow}(x,x';\omega=0.5)$"),
        (np.imag(sigma), r"$\mathrm{Im}\,\Sigma_{xc\downarrow}(x,x';\omega=0.5)$"),
    )
    for ax, (data, title) in zip(axes.ravel(), panels):
        heatmap(ax, s.x, data)
        ax.set_title(title)
        ax.set_xlabel("$x'$")
        ax.set_ylabel("$x$")
    save(fig, "frequency_frames.pdf")


def krylov_convergence():
    """Block Krylov against complete diagonalisation for a two-electron atom (N+1 sectors of dimension 4410)."""
    x = np.linspace(-6.0, 6.0, 21)
    s = iDEA.system.System(x, -2.0 / (abs(x) + 1.0), iDEA.interactions.softened_interaction(x), "ud")
    state = iDEA.methods.interacting.solve(s, k=0)
    exact = iDEA.methods.interacting.green_function(s, state, solver="dense", silent=True)
    omega = np.linspace(-1.5, 2.5, 600)
    eta = 0.05
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), layout="constrained")
    trace = lambda g: iDEA.observables.spectral_function(s, g, omega, eta=eta, diagonal=True).sum(axis=1) * s.dx  # noqa: E731
    axes[0].semilogy(omega, trace(exact), "k-", lw=1.4, label="dense")
    ks = np.arange(0, 16)
    discrete = lambda g: (g.up.orbitals * np.sqrt(s.dx), g.up.energies)  # noqa: E731
    V, E = discrete(exact)
    moments = [(V * E**k) @ V.T for k in ks]
    errors = {}
    for blocks, ls in ((2, ":"), (4, "--"), (8, "-.")):
        krylov = iDEA.methods.interacting.green_function(
            s, state, solver="krylov", krylov_dim=blocks * x.shape[0], silent=True
        )
        axes[0].semilogy(omega, trace(krylov), ls, lw=1.0, label=f"$m={blocks}$")
        W, F = discrete(krylov)
        errors[blocks] = [np.max(abs((W * F**k) @ W.T - M)) / np.max(abs(M)) for k, M in zip(ks, moments)]
        axes[1].semilogy(ks, np.maximum(errors[blocks], 1e-17), "o" + ls, ms=3, lw=0.8, label=f"$m={blocks}$")
        axes[1].axvline(2 * blocks - 1 + 0.5, color="0.7", lw=0.6)
    axes[0].set_xlabel(r"$\omega$")
    axes[0].set_title(r"$\int A_\uparrow(x,x;\omega)\,dx$, two electrons")
    axes[0].set_ylim(1e-3, None)
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("moment $k$")
    axes[1].set_title(r"relative error of $\sum_n d_n d_n^\dagger \varepsilon_n^k$")
    axes[1].legend()
    save(fig, "krylov_convergence.pdf")
    dimension = max(info["dimension"] for info in exact.info["sectors"])
    return dimension, errors


def main():
    args = parse_arguments()
    os.makedirs(FIGURES, exist_ok=True)
    values = {}

    # The one-dimensional hydrogen atom: exact, and in the GW approximation.
    x = np.linspace(-args.length, args.length, args.nx)
    s = iDEA.system.System(x, -1.0 / (abs(x) + 1.0), iDEA.interactions.softened_interaction(x), "u")
    state = iDEA.methods.interacting.solve(s, k=0)
    start = time.time()
    green = iDEA.methods.interacting.green_function(
        s, state, solver=args.solver, dense_max=args.dense_max, GPU=args.gpu, silent=True
    )
    values["TimeExact"] = f"{time.time() - start:.0f}"
    qp = {mode: iDEA.methods.mbpt.solve(s, mode=mode, silent=True) for mode in ("G0W0", "GW0", "GW")}
    phi = green.up.orbitals[:, green.up.occupations > 0][:, 0]
    addition = (green.down.occupations == 0) & (green.down.energies < 0)
    bound = float(green.down.energies[addition][0]) if np.any(addition) else float(green.mu)

    info = green.info
    values.update(
        {
            "Nx": f"{args.nx}",
            "Length": f"{args.length:g}",
            "Dx": f"{s.dx:.2f}",
            "Ezero": f"{green.energy:.6f}",
            "Mu": f"{green.mu:.4f}",
            "Hminus": f"{bound:.4f}",
            "EAffinity": f"{-bound:.4f}",
            "DimUp": f"{info['sectors'][1]['dimension']:,}".replace(",", "{,}"),
            "DimDown": f"{info['sectors'][2]['dimension']:,}".replace(",", "{,}"),
            "PolesUp": f"{green.up.energies.shape[0]:,}".replace(",", "{,}"),
            "PolesDown": f"{green.down.energies.shape[0]:,}".replace(",", "{,}"),
            "Rayleigh": f"{info['rayleigh_residual']:.0e}",
            "Completeness": f"{max(info['up_completeness_error'], info['down_completeness_error']):.0e}",
            "FirstMoment": f"{max(info['up_first_moment_error'], info['down_first_moment_error']):.0e}",
            "GMError": f"{info['galitskii_migdal_error']:.0e}",
            "SigmaXResidual": f"{np.max(abs(green.v_h * phi + green.up.sigma_x @ phi * s.dx)):.0e}",
        }
    )
    for mode, name in (("G0W0", "GzWz"), ("GW0", "GWz"), ("GW", "GW")):
        values[f"E{name}"] = f"{qp[mode].energy:.6f}"
        removal = qp[mode].up.energies[qp[mode].up.occupations > 0]
        values[f"Eps{name}"] = f"{np.max(removal):.4f}"

    spectral_functions(s, green, args.eta)
    exact_residual, gw_residuals = self_screening(s, green, qp, phi)
    values["ScreeningExact"] = f"{exact_residual:.0e}"
    for mode, name in (("G0W0", "GzWz"), ("GW0", "GWz"), ("GW", "GW")):
        values[f"Screening{name}"] = f"{gw_residuals[mode]:.1e}"
    depths = static_correlation(s, green, qp["GW"])
    for (name, spin), depth in depths.items():
        values[f"Depth{name.capitalize() if name == 'exact' else name}{spin.capitalize()}"] = f"{depth:.4f}"
    injected_electron(s, green)
    frequency_frames(s, green, args.eta, bound)
    dimension, errors = krylov_convergence()
    values["KrylovDim"] = f"{dimension:,}".replace(",", "{,}")

    # Numbers quoted in the text, as LaTeX macros (scientific notation as a times ten to the b).
    def latex(value):
        if "e" in value and value.replace("e", "").replace("-", "").replace("+", "").replace(".", "").isdigit():
            mantissa, exponent = value.split("e")
            return rf"{mantissa}\times10^{{{int(exponent)}}}"
        return value

    with open(os.path.join(FIGURES, "values.tex"), "w") as f:
        f.write("% Generated by report/make_figures.py; do not edit.\n")
        for key, value in values.items():
            f.write(rf"\newcommand{{\val{key}}}{{{latex(value)}}}" + "\n")
    print("wrote values.tex")
    for key, value in values.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
