"""
Exact Green's function and self-energy of the one-dimensional hydrogen atom, compared with GW.

Computes G(x,x') and Sigma_xc(x,x') exactly from the eigenstates of the 0, 1 and 2-electron systems, renders videos
of them over real time, real frequency and imaginary frequency, and compares the exact self-energy with G0W0, GW0
and self-consistent GW (iDEA.methods.mbpt). For one electron the exact self-energy is free of self-interaction:
(v_h + Sigma_xc(i nu)) phi = 0 for the occupied orbital phi at every frequency, whereas GW is not (self-screening).

Usage (the defaults run in a few minutes on a laptop; larger grids are best run with --gpu):

    python tutorial/green_function_hydrogen.py --nx 101 --out hydrogen
    python tutorial/green_function_hydrogen.py --nx 161 --dense-max 30000 --gpu --out hydrogen_161

Outputs, in the --out directory: pole data (green_function.npz), sum-rule residuals and timings (info.json),
static figures (.png) and videos (.mp4 or .gif).
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import iDEA  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nx", type=int, default=101, help="Number of grid points.")
    parser.add_argument("--length", type=float, default=15.0, help="Half-width of the box.")
    parser.add_argument("--solver", default="auto", choices=["auto", "dense", "krylov"], help="Sector solver.")
    parser.add_argument("--dense-max", type=int, default=6000, help="Largest sector to diagonalise completely.")
    parser.add_argument("--krylov-dim", type=int, default=None, help="Maximum block Krylov dimension.")
    parser.add_argument("--gpu", action="store_true", help="Diagonalise the sectors on the GPU (cupy).")
    parser.add_argument("--eta", type=float, default=0.05, help="Broadening of real-frequency quantities.")
    parser.add_argument("--cutoff", type=float, default=1.0, help="Energy filter for the real-time videos.")
    parser.add_argument("--t-max", type=float, default=30.0, help="Largest real time.")
    parser.add_argument("--omega-min", type=float, default=-1.0, help="Smallest real frequency.")
    parser.add_argument("--omega-max", type=float, default=2.0, help="Largest real frequency.")
    parser.add_argument("--nu-max", type=float, default=3.0, help="Largest imaginary frequency.")
    parser.add_argument("--frames", type=int, default=150, help="Frames per video.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second.")
    parser.add_argument("--format", default="mp4", choices=["mp4", "gif"], help="Video format.")
    parser.add_argument("--no-gw", action="store_true", help="Skip the GW comparison.")
    parser.add_argument("--no-videos", action="store_true", help="Skip the videos.")
    parser.add_argument("--out", default="green_function_hydrogen", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.out, exist_ok=True)
    path = lambda name: os.path.join(args.out, name)  # noqa: E731
    timings = {}

    # The one-dimensional hydrogen atom.
    x = np.linspace(-args.length, args.length, args.nx)
    s = iDEA.system.System(x, -1.0 / (abs(x) + 1.0), iDEA.interactions.softened_interaction(x), "u")

    # Exact ground state, Green's function and self-energy.
    start = time.time()
    state = iDEA.methods.interacting.solve(s, k=0)
    green = iDEA.methods.interacting.green_function(
        s, state, solver=args.solver, dense_max=args.dense_max, krylov_dim=args.krylov_dim, GPU=args.gpu
    )
    timings["exact"] = time.time() - start
    phi = green.up.orbitals[:, green.up.occupations > 0][:, 0]
    print(f"E0 = {green.energy:.10f}, removal pole = {green.up.energies[green.up.occupations > 0][0]:.10f}")
    print(f"mu = {green.mu:.6f}, exact Green's function in {timings['exact']:.1f} s")
    data = {"x": x, "mu": green.mu, "energy": green.energy, "v_h": green.v_h}
    for spin in ("up", "down"):
        channel = getattr(green, spin)
        for name in ("energies", "orbitals", "occupations", "sigma_x", "sigma_energies", "sigma_orbitals"):
            data[f"{spin}_{name}"] = getattr(channel, name)
        data[f"{spin}_sigma_character"] = channel.sigma_character
    np.savez_compressed(path("green_function.npz"), **data)

    # Spectral functions.
    omega = np.linspace(args.omega_min, args.omega_max, 600)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, spin in zip(axes, ("up", "down")):
        A = iDEA.observables.spectral_function(s, green, omega, eta=args.eta, spin=spin, diagonal=True)
        image = ax.imshow(
            A.T,
            origin="lower",
            aspect="auto",
            extent=[omega[0], omega[-1], x[0], x[-1]],
            cmap="magma",
            vmax=np.percentile(A, 99.5),
        )
        ax.axvline(green.mu, color="w", ls="--", lw=0.8)
        ax.set_xlabel(r"$\omega$")
        ax.set_title(rf"$A_{{\{spin}arrow}}(x,x;\omega)$")
        fig.colorbar(image, ax=ax)
    axes[0].set_ylabel("$x$")
    fig.savefig(path("spectral_function.png"), dpi=150)
    plt.close(fig)

    # Comparison with GW: self-interaction (self-screening) of the occupied orbital, and total energies.
    nu = np.linspace(0.0, args.nu_max, 200)
    report = {"exact": {"energy": green.energy}}
    if not args.no_gw:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        residual = [np.linalg.norm(S @ phi) * s.dx for S in iDEA.observables.self_energy(s, green, nu=nu, part="c")]
        ax.semilogy(nu, np.maximum(residual, 1e-18), "k-", label="exact")
        for mode in ("G0W0", "GW0", "GW"):
            start = time.time()
            qp = iDEA.methods.mbpt.solve(s, mode=mode, silent=True)
            timings[mode] = time.time() - start
            phi_gw = qp.up.orbitals[:, qp.up.occupations > 0][:, 0]
            sigma_c = iDEA.observables.self_energy(s, qp, nu=nu, part="c")
            ax.semilogy(nu, [np.linalg.norm(S @ phi_gw) * s.dx for S in sigma_c], label=mode)
            report[mode] = {"energy": qp.energy, "mu": qp.up.mu}
            print(f"{mode}: E = {qp.energy:.6f} (exact {green.energy:.6f})")
            if not args.no_videos and mode == "GW":
                iDEA.plotting.animate_self_energy(
                    s,
                    qp,
                    nu=np.linspace(0.0, args.nu_max, args.frames),
                    file_name=path(f"Sigma_xc_up_nu_GW.{args.format}"),
                    fps=args.fps,
                )
        ax.set_xlabel(r"$\nu$")
        ax.set_ylabel(r"$\Vert \Sigma_c(i\nu)\,\phi \Vert$")
        ax.set_title("Self-screening of the occupied orbital")
        ax.legend()
        fig.savefig(path("self_screening.png"), dpi=150)
        plt.close(fig)

    # Videos.
    if not args.no_videos:
        start = time.time()
        t = np.linspace(0.0, args.t_max, args.frames)
        omega = np.linspace(args.omega_min, args.omega_max, args.frames)
        nu = np.linspace(0.0, args.nu_max, args.frames)
        for spin in ("up", "down"):
            iDEA.plotting.animate_green_function(
                s,
                green,
                t=t,
                spin=spin,
                cutoff=args.cutoff,
                demodulate=True,
                file_name=path(f"G_{spin}_t.{args.format}"),
                fps=args.fps,
            )
            iDEA.plotting.animate_green_function(
                s,
                green,
                omega=omega,
                spin=spin,
                eta=args.eta,
                file_name=path(f"G_{spin}_omega.{args.format}"),
                fps=args.fps,
            )
            iDEA.plotting.animate_self_energy(
                s,
                green,
                omega=omega,
                spin=spin,
                eta=args.eta,
                file_name=path(f"Sigma_xc_{spin}_omega.{args.format}"),
                fps=args.fps,
            )
            iDEA.plotting.animate_self_energy(
                s,
                green,
                t=t,
                spin=spin,
                cutoff=args.cutoff,
                file_name=path(f"Sigma_c_{spin}_t.{args.format}"),
                fps=args.fps,
            )
        iDEA.plotting.animate_self_energy(
            s, green, nu=nu, file_name=path(f"Sigma_xc_up_nu_exact.{args.format}"), fps=args.fps
        )
        timings["videos"] = time.time() - start

    info = {key: value for key, value in green.info.items()}
    info.update({"nx": args.nx, "length": args.length, "timings": timings, "comparison": report})
    with open(path("info.json"), "w") as f:
        json.dump(info, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"Wrote results to {args.out}/")


if __name__ == "__main__":
    main()
