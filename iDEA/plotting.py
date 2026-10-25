"""Contains plotting functionality: snapshots and videos of two-point quantities such as G(x,x';t) and Sigma(x,x';omega)."""

import os

import numpy as np

import iDEA.methods.interacting
import iDEA.observables
import iDEA.state
import iDEA.system

__all__ = [
    "plot_two_point",
    "animate",
    "animate_green_function",
    "animate_self_energy",
]

_PARTS = {
    "real": (np.real, "RdBu_r", True),
    "imag": (np.imag, "RdBu_r", True),
    "abs": (np.abs, "viridis", False),
    "phase": (np.angle, "twilight", False),
}

_LABELS = {"t": "t", "omega": r"\omega", "z": "z", "nu": r"i\nu", "tau": r"\tau"}


def _check_parts(parts: tuple) -> tuple:
    r"""
    Check the requested parts of a complex quantity.

    | Args:
    |     parts: tuple, Parts to plot, from 'real', 'imag', 'abs' and 'phase'.

    | Returns:
    |     parts: tuple, Parts to plot.
    """
    parts = (parts,) if isinstance(parts, str) else tuple(parts)
    for part in parts:
        if part not in _PARTS:
            raise ValueError(f"parts must be chosen from {list(_PARTS)}, got {part} instead.")
    return parts


def _colour_norm(values: np.ndarray, part: str, norm: str, vmax: float, percentile: float):
    r"""
    Construct the colour normalisation for one part of a quantity.

    | Args:
    |     values: np.ndarray, Values of the part (all frames).
    |     part: str, Part being plotted.
    |     norm: str, 'linear' or 'symlog'.
    |     vmax: float, Largest magnitude of the colour scale [If None this is the given percentile of the magnitudes].
    |     percentile: float, Percentile of the magnitudes used for the colour scale.

    | Returns:
    |     norm: matplotlib.colors.Normalize, Colour normalisation.
    """
    import matplotlib.colors

    if part == "phase":
        return matplotlib.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    if vmax is None:
        vmax = float(np.percentile(abs(values), percentile))
    if vmax <= 0.0:
        vmax = 1.0
    symmetric = _PARTS[part][2]
    vmin = -vmax if symmetric else 0.0
    if norm == "linear":
        return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if norm == "symlog":
        return matplotlib.colors.SymLogNorm(linthresh=1e-2 * vmax, vmin=vmin, vmax=vmax)
    raise ValueError(f"norm must be 'linear' or 'symlog', got {norm} instead.")


def _figure(x: np.ndarray, parts: tuple, trace: bool, title: str):
    r"""
    Construct a figure with one heatmap per part and, optionally, a trace panel underneath.

    | Args:
    |     x: np.ndarray, Grid.
    |     parts: tuple, Parts to plot.
    |     trace: bool, Add a trace panel.
    |     title: str, Figure title.

    | Returns:
    |     fig: matplotlib.figure.Figure, Figure.
    |     axes: list, Heatmap axes.
    |     trace_axis: matplotlib.axes.Axes, Trace axis (or None).
    """
    from matplotlib.figure import Figure

    columns = len(parts)
    fig = Figure(figsize=(4.2 * columns, 4.6 + (1.6 if trace else 0.0)), layout="constrained")
    grid = fig.add_gridspec(2 if trace else 1, columns, height_ratios=[4, 1.2] if trace else [1])
    axes = [fig.add_subplot(grid[0, i]) for i in range(columns)]
    for ax, part in zip(axes, parts):
        ax.set_xlabel("$x'$")
        ax.set_ylabel("$x$")
        ax.set_title(part)
        ax.set_aspect("equal")
    trace_axis = fig.add_subplot(grid[1, :]) if trace else None
    if title is not None:
        fig.suptitle(title)
    return fig, axes, trace_axis


def plot_two_point(
    x: np.ndarray,
    data: np.ndarray,
    parts: tuple = ("real", "imag"),
    title: str = None,
    norm: str = "linear",
    vmax: float = None,
    percentile: float = 100.0,
    cmap: str = None,
):
    r"""
    Plot a snapshot of a two-point quantity A(x,x'), such as a Green's function, self-energy or density matrix.

    | Args:
    |     x: np.ndarray, Grid.
    |     data: np.ndarray, Two-point quantity, indexed as data[space,space].
    |     parts: tuple, Parts of the (complex) quantity to plot, from 'real', 'imag', 'abs' and 'phase'. (default = ('real', 'imag'))
    |     title: str, Title. (default = None)
    |     norm: str, Colour normalisation, 'linear' or 'symlog'. (default = 'linear')
    |     vmax: float, Largest magnitude of the colour scale [If None this is set by percentile]. (default = None)
    |     percentile: float, Percentile of the magnitudes used for the colour scale. (default = 100)
    |     cmap: str, Colour map [If None a diverging map is used for real and imaginary parts]. (default = None)

    | Returns:
    |     fig: matplotlib.figure.Figure, Figure.
    """
    parts = _check_parts(parts)
    fig, axes, _ = _figure(x, parts, False, title)
    extent = [x[0], x[-1], x[0], x[-1]]
    for ax, part in zip(axes, parts):
        values = _PARTS[part][0](data)
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap=cmap or _PARTS[part][1],
            norm=_colour_norm(values, part, norm, vmax, percentile),
            interpolation="nearest",
        )
        fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def animate(
    x: np.ndarray,
    data: np.ndarray,
    frame_values: np.ndarray,
    frame_label: str = "t",
    parts: tuple = ("real", "imag"),
    title: str = None,
    norm: str = "linear",
    vmax: float = None,
    percentile: float = 99.5,
    cmap: str = None,
    trace: np.ndarray = None,
    trace_label: str = None,
    file_name: str = None,
    fps: int = 15,
    dpi: int = 100,
):
    r"""
    Animate a two-point quantity A(x,x') over a time or frequency axis, e.g. G(x,x';t) or Sigma(x,x';omega).

    Each frame shows the requested parts of A side by side, with a colour scale that is fixed over all frames. An
    optional trace (a 1D quantity on the same axis, such as the spectral function) is shown underneath, with a
    marker at the current frame. The video is saved if a file name is given: '.mp4' (requires ffmpeg) or '.gif'.
    The animation object is returned, so in a notebook it can be displayed with
    IPython.display.HTML(animation.to_jshtml()).

    | Args:
    |     x: np.ndarray, Grid.
    |     data: np.ndarray, Two-point quantity on each frame, indexed as data[frame,space,space].
    |     frame_values: np.ndarray, Value of the time or frequency at each frame, indexed as frame_values[frame].
    |     frame_label: str, Name of the axis, e.g. 't', 'omega', 'nu' or 'tau' (rendered with LaTeX if a Greek letter). (default = 't')
    |     parts: tuple, Parts of the (complex) quantity to plot, from 'real', 'imag', 'abs' and 'phase'. (default = ('real', 'imag'))
    |     title: str, Title. (default = None)
    |     norm: str, Colour normalisation, 'linear' or 'symlog'. (default = 'linear')
    |     vmax: float, Largest magnitude of the colour scale [If None this is set by percentile over all frames]. (default = None)
    |     percentile: float, Percentile of the magnitudes (over all frames) used for the colour scale. (default = 99.5)
    |     cmap: str, Colour map [If None a diverging map is used for real and imaginary parts]. (default = None)
    |     trace: np.ndarray, 1D quantity to plot underneath, indexed as trace[frame]. (default = None)
    |     trace_label: str, Label of the trace. (default = None)
    |     file_name: str, File to save the video to ('.mp4' or '.gif'). (default = None)
    |     fps: int, Frames per second. (default = 15)
    |     dpi: int, Resolution of the saved video. (default = 100)

    | Returns:
    |     animation: matplotlib.animation.FuncAnimation, Animation.
    """
    import matplotlib.animation

    parts = _check_parts(parts)
    data = np.asarray(data)
    frame_values = np.asarray(frame_values)
    if data.ndim != 3 or data.shape[1:] != (x.shape[0], x.shape[0]):
        raise ValueError(f"data must be indexed as data[frame,space,space], got shape {data.shape} instead.")
    if frame_values.shape[0] != data.shape[0]:
        raise ValueError("frame_values must have one value per frame.")
    fig, axes, trace_axis = _figure(x, parts, trace is not None, title)
    extent = [x[0], x[-1], x[0], x[-1]]
    label = _LABELS.get(frame_label, frame_label)
    images = []
    for ax, part in zip(axes, parts):
        values = _PARTS[part][0](data)
        image = ax.imshow(
            values[0],
            origin="lower",
            extent=extent,
            cmap=cmap or _PARTS[part][1],
            norm=_colour_norm(values, part, norm, vmax, percentile),
            interpolation="nearest",
            animated=True,
        )
        fig.colorbar(image, ax=ax, shrink=0.8)
        images.append((image, values))
    marker = None
    if trace is not None:
        trace_axis.plot(np.real(frame_values), np.asarray(trace), "k-", lw=1.0)
        trace_axis.set_xlabel(f"${label}$")
        if trace_label is not None:
            trace_axis.set_ylabel(trace_label)
        marker = trace_axis.axvline(np.real(frame_values[0]), color="tab:red", lw=1.0)
    stamp = fig.text(0.01, 0.99, "", va="top", ha="left")

    def update(frame):
        for image, values in images:
            image.set_data(values[frame])
        stamp.set_text(f"${label} = {np.real(frame_values[frame]):.3f}$")
        if marker is not None:
            marker.set_xdata([np.real(frame_values[frame])] * 2)
        return [image for image, _ in images] + [stamp] + ([marker] if marker is not None else [])

    animation = matplotlib.animation.FuncAnimation(fig, update, frames=data.shape[0], interval=1000.0 / fps, blit=False)
    if file_name is not None:
        extension = os.path.splitext(file_name)[1].lower()
        if extension == ".gif":
            writer = matplotlib.animation.PillowWriter(fps=fps)
        elif extension in (".mp4", ".mov", ".mkv"):
            if not matplotlib.animation.FFMpegWriter.isAvailable():
                raise RuntimeError("Saving to video requires ffmpeg; install it, or save as '.gif' instead.")
            writer = matplotlib.animation.FFMpegWriter(fps=fps)
        else:
            raise ValueError(f"file_name must end in '.mp4', '.mov', '.mkv' or '.gif', got {file_name} instead.")
        animation.save(file_name, writer=writer, dpi=dpi)
    return animation


def _axis(t, omega, nu, tau) -> tuple:
    r"""Return the name and values of the single given axis."""
    provided = {
        name: values for name, values in (("t", t), ("omega", omega), ("nu", nu), ("tau", tau)) if values is not None
    }
    if len(provided) != 1:
        raise ValueError(f"Exactly one of t, omega, nu or tau must be given, got {list(provided) or 'none'}.")
    name, values = provided.popitem()
    return name, np.atleast_1d(np.asarray(values))


def animate_green_function(
    s: iDEA.system.System,
    state,
    t: np.ndarray = None,
    omega: np.ndarray = None,
    nu: np.ndarray = None,
    tau: np.ndarray = None,
    eta: float = 0.05,
    kind: str = "time-ordered",
    spin: str = "up",
    demodulate: bool = False,
    cutoff: float = None,
    trace: bool = True,
    title: str = None,
    **kwargs,
):
    r"""
    Animate the Green's function G(x,x') of a spin channel over a time or frequency axis.

    On the real-frequency axis the trace underneath is the total spectral function, Tr A(omega). On the real-time
    axis it is |Tr G(t)|, and on the imaginary axis Re Tr G.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.GreenFunction, iDEA.state.SingleBodyState, iDEA.state.QuasiParticleState (nu or tau only) or iDEA.state.ManyBodyState (the exact Green's function is computed).
    |     t, omega, nu, tau: np.ndarray, Axis (exactly one must be given; each value is a frame).
    |     eta: float, Broadening for real frequencies. (default = 0.05)
    |     kind: str, 'time-ordered', 'retarded' or 'advanced'. (default = 'time-ordered')
    |     spin: str, Spin channel, 'up' or 'down'. (default = 'up')
    |     demodulate: bool, On the real-time axis, multiply by exp(i mu t) to remove the overall phase rotation. (default = False)
    |     cutoff: float, Smooth energy filter of the poles about mu, to band-limit real-time videos (see iDEA.observables.green_function). (default = None)
    |     trace: bool, Show a trace underneath. (default = True)
    |     title: str, Title [If None a default is used]. (default = None)
    |     **kwargs: Arguments passed to animate (parts, norm, vmax, percentile, cmap, file_name, fps, dpi).

    | Returns:
    |     animation: matplotlib.animation.FuncAnimation, Animation.
    """
    axis, values = _axis(t, omega, nu, tau)
    if isinstance(state, iDEA.state.ManyBodyState):
        state = iDEA.methods.interacting.green_function(s, state, silent=True)
    data = iDEA.observables.green_function(s, state, eta=eta, kind=kind, spin=spin, cutoff=cutoff, **{axis: values})
    if demodulate and axis == "t":
        if state.mu is None:
            raise ValueError("demodulate requires a chemical potential, but this state has no gap.")
        data = data * np.exp(1.0j * state.mu * values)[:, None, None]
    trace_values, trace_label = None, None
    if trace:
        if axis == "omega" and not isinstance(state, iDEA.state.QuasiParticleState):
            A = iDEA.observables.spectral_function(s, state, values, eta=eta, spin=spin, diagonal=True)
            trace_values, trace_label = np.sum(A, axis=1) * s.dx, r"Tr $A(\omega)$"
        elif axis == "t":
            trace_values, trace_label = abs(np.trace(data, axis1=1, axis2=2)) * s.dx, "|Tr $G(t)$|"
        else:
            trace_values, trace_label = np.real(np.trace(data, axis1=1, axis2=2)) * s.dx, "Re Tr $G$"
    if title is None:
        arrow = r"\uparrow" if spin == "up" else r"\downarrow"
        title = rf"$G_{{{arrow}}}(x,x';{_LABELS[axis]})$"
    return animate(
        s.x, data, values, frame_label=axis, title=title, trace=trace_values, trace_label=trace_label, **kwargs
    )


def animate_self_energy(
    s: iDEA.system.System,
    state,
    t: np.ndarray = None,
    omega: np.ndarray = None,
    nu: np.ndarray = None,
    tau: np.ndarray = None,
    eta: float = 0.05,
    kind: str = "time-ordered",
    part: str = None,
    spin: str = "up",
    cutoff: float = None,
    trace: bool = True,
    title: str = None,
    **kwargs,
):
    r"""
    Animate the self-energy Sigma(x,x') of a spin channel over a time or frequency axis.

    By default the exchange-correlation self-energy is shown on frequency axes, and the correlation self-energy on
    time axes (the exchange part is a delta function in time). The trace underneath is Re Tr Sigma.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.GreenFunction, iDEA.state.QuasiParticleState (nu or tau only) or iDEA.state.ManyBodyState (the exact Green's function is computed).
    |     t, omega, nu, tau: np.ndarray, Axis (exactly one must be given; each value is a frame).
    |     eta: float, Broadening for real frequencies. (default = 0.05)
    |     kind: str, 'time-ordered', 'retarded' or 'advanced'. (default = 'time-ordered')
    |     part: str, 'xc' or 'c' [If None, 'xc' on frequency axes and 'c' on time axes]. (default = None)
    |     spin: str, Spin channel, 'up' or 'down'. (default = 'up')
    |     cutoff: float, Smooth energy filter of the correlation poles about mu, to band-limit real-time videos (see iDEA.observables.self_energy). (default = None)
    |     trace: bool, Show a trace underneath. (default = True)
    |     title: str, Title [If None a default is used]. (default = None)
    |     **kwargs: Arguments passed to animate (parts, norm, vmax, percentile, cmap, file_name, fps, dpi).

    | Returns:
    |     animation: matplotlib.animation.FuncAnimation, Animation.
    """
    axis, values = _axis(t, omega, nu, tau)
    if isinstance(state, iDEA.state.ManyBodyState):
        state = iDEA.methods.interacting.green_function(s, state, silent=True)
    if part is None:
        part = "c" if axis in ("t", "tau") else "xc"
    cutoff_kwargs = {} if cutoff is None else {"cutoff": cutoff}
    data = iDEA.observables.self_energy(
        s, state, eta=eta, kind=kind, part=part, spin=spin, **cutoff_kwargs, **{axis: values}
    )
    trace_values = np.real(np.trace(data, axis1=1, axis2=2)) * s.dx if trace else None
    if title is None:
        arrow = r"\uparrow" if spin == "up" else r"\downarrow"
        title = rf"$\Sigma_{{{part},{arrow}}}(x,x';{_LABELS[axis]})$"
    return animate(
        s.x,
        data,
        values,
        frame_label=axis,
        title=title,
        trace=trace_values,
        trace_label=r"Re Tr $\Sigma$" if trace else None,
        **kwargs,
    )
