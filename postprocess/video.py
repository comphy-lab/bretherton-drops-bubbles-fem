"""Render an axisymmetric animation of the co-moving void bubble from pyoomph VTU output.

What is drawn
-------------
One tube, drawn about its axis, with a different field either side:

* **above the axis** the velocity magnitude ``|u|`` in the **bubble frame**;
* **below the axis** ``log10 Phi`` with ``Phi = 2 mu E:E`` the viscous
  dissipation rate per unit volume, ``E`` the rate-of-strain tensor.

The interface is mirrored about the axis so the bubble reads as one object, and
the void is simply absent from the liquid mesh, so it renders as background
rather than as a masked field. The interface is the mesh boundary adjacent to
the void - it is extracted as the boundary edge chain of the triangulation, not
chased as an isosurface, because in this formulation the mesh nodes *are* the
interface.

Why one row, not two
--------------------
The parent Basilisk renderer (``postProcess/Video-bretherton.py``) needs two
rows: the whole domain, plus a window travelling with the bubble, because there
the bubble translates through a 16-radius domain. Here the calculation is done
in a frame moving with the bubble, so the bubble is stationary in the mesh and
a travelling window would be the same window twice. The single panel is the
visible consequence of the co-moving formulation, so it is labelled as such in
the figure rather than silently dropped, and the axial ticks are kept so a
viewer can confirm the bubble does not drift.

The window is fixed for the same reason the parent's is: the tube wall is
rigid, so it must occupy the same pixels in every frame. Axes limits, ticks,
figure size and colour limits are computed once in :func:`main` and passed to
every worker, and the frame PNG dimensions are checked for uniformity before
encoding.

Velocities are in the bubble frame
----------------------------------
This is not a detail. In the bubble frame the wall moves at ``-U`` and the far
field is Poiseuille shifted by ``-U``; a viewer who reads the colour scale as a
laboratory velocity will conclude the wall is moving backwards through a
stationary bubble, which is exactly what is being displayed but not what is
being computed about. The frame is therefore stated in the figure caption line
of every frame.

Dissipation
-----------
``Phi`` is computed from the velocity gradient of the P2 (``triangle6``)
velocity field, not from any quantity written by the solver. For each element
the isoparametric Jacobian is evaluated at the six nodal parametric points and
inverted, giving an exact elementwise gradient of the quadratic field; the
element contributions are then averaged at shared nodes. With ``mu = 1``:

    Phi = 2 ( E_rr^2 + E_zz^2 + E_tt^2 + 2 E_rz^2 )

    E_rr = du_r/dr,  E_zz = du_z/dz,  E_rz = (du_r/dz + du_z/dr)/2,
    E_tt = u_r/r     (hoop term; replaced by du_r/dr where r < ``R_EPS``,
                      which is its limit on the axis since u_r vanishes there
                      linearly)

The reconstruction is checked rather than trusted: ``--check`` reports
``rms(div u)/rms(grad u)`` from the same reconstructed gradients, which is zero
for the exact incompressible solution and is therefore a direct measure of how
much the nodal averaging has corrupted the gradient. If that ratio is large the
honest response is ``--lower pressure``, which draws the solver's own pressure
field below the axis and labels it as pressure.

Nondimensionalisation is ``sigma = mu = R0 = 1``; lengths are in ``R0``, so
``Ca_in`` is the imposed mean inlet speed.

Typography
----------
mathtext, not ``text.usetex``. LaTeX spawns a subprocess per unique string and
deadlocks under ``multiprocessing``, which is how frames are generated here.

Usage
-----
    python -m postprocess.video OUTPUT_DIR --out media/bubble.mp4
    python -m postprocess.video OUTPUT_DIR --stills-only --stills-dir media/

``OUTPUT_DIR`` is a pyoomph output directory: the one holding ``liquid.pvd``
and ``liquid/liquid_*.vtu``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

# Parallel-safe typography: see the module docstring.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

R_EPS = 1e-9
"""Radius below which the hoop strain u_r/r is replaced by its axial limit."""

BOUNDARY_TOL = 1e-7
"""Absolute tolerance for recognising a node as sitting on a pinned boundary.

The wall, inlet and outlet mesh nodes are held by ``DirichletBC(mesh_x=True,
mesh_y=True)``, so they are at their nominal coordinates to round-off; only the
interface and the interior move. A loose tolerance would swallow the first film
element into the "wall" class and silently truncate the interface.
"""

CMAP_SPEED = "Blues"
CMAP_DISS = "hot_r"
CMAP_PRESSURE = "RdBu_r"

# Font sizes for a wide, short panel. The publication-plots defaults (label 40,
# ticks 30) are set for a 12-19 inch square; on a ~2 inch tall strip a 40 pt
# label is a third of the panel height, so they are scaled to the panel, not to
# the figure width.
FS_TITLE = 22
FS_SUB = 15
FS_LABEL = 19
FS_TICK = 14
FS_CBAR_LABEL = 17
FS_CBAR_TICK = 12
FS_CAPTION = 13


# --------------------------------------------------------------------- reading


@dataclass(frozen=True)
class Frame:
    """One VTU snapshot: the liquid mesh and its P2 fields."""

    time: float
    index: int
    points: np.ndarray  # (N, 2) as (r, z)
    cells: np.ndarray  # (M, 6) triangle6 connectivity, VTK ordering
    velocity: np.ndarray  # (N, 2) as (u_r, u_z), bubble frame
    pressure: np.ndarray  # (N,)


def read_series(output_dir: Path) -> list[tuple[float, Path]]:
    """``(time, vtu path)`` pairs in numeric time order.

    ``liquid.pvd`` is the only place the solver records the physical time of
    each file, so it is preferred. Ordering is numeric on the recorded time, not
    lexicographic on the filename: ``liquid_000010.vtu`` sorts before
    ``liquid_000009.vtu`` under several plausible glob orders, and a video built
    that way runs backwards in places without ever failing.
    """
    pvd = output_dir / "liquid.pvd"
    if pvd.is_file():
        root = ET.parse(pvd).getroot()
        series = []
        for ds in root.iter("DataSet"):
            rel = ds.get("file")
            if rel is None:
                continue
            series.append((float(ds.get("timestep", "nan")), (output_dir / rel).resolve()))
        series = [(t, p) for t, p in series if p.is_file()]
        if series:
            return sorted(series, key=lambda item: item[0])

    # Fall back to the files themselves, still ordered numerically, with the
    # file index standing in for a time that was never recorded.
    files = sorted(
        (output_dir / "liquid").glob("liquid_*.vtu"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not files:
        raise SystemExit(f"no VTU output found under {output_dir}")
    print(
        f"warning: no readable {pvd.name}; frame index is used in place of time",
        file=sys.stderr,
    )
    return [(float(i), p) for i, p in enumerate(files)]


def load_frame(index: int, time: float, path: Path) -> Frame:
    import meshio

    mesh = meshio.read(path)
    blocks = [cb for cb in mesh.cells if cb.type == "triangle6"]
    if not blocks:
        raise SystemExit(f"{path} has no triangle6 cells; this reader assumes P2 output")
    cells = np.vstack([cb.data for cb in blocks]).astype(np.int64)

    pts = np.asarray(mesh.points, dtype=float)[:, :2]  # (x=r, y=z)
    vel = np.asarray(mesh.point_data["velocity"], dtype=float)[:, :2]
    pres = np.asarray(mesh.point_data["pressure"], dtype=float).ravel()
    return Frame(time=time, index=index, points=pts, cells=cells, velocity=vel, pressure=pres)


# ------------------------------------------------------------------- P2 fields

# VTK quadratic triangle: corners 0,1,2 then midsides 3=(0,1), 4=(1,2), 5=(2,0).
# Parametric coordinates of the six nodes.
_NODE_XI = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
)

_SUBTRIS = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
"""Four straight sub-triangles per P2 element, so a quadratic field is drawn
piecewise-linearly on the midside nodes instead of being collapsed to its
corner values."""


def _dshape(xi: np.ndarray) -> np.ndarray:
    """Derivatives of the six P2 shape functions at one parametric point.

    Returns ``(6, 2)`` holding ``dN_i/dxi_a``.
    """
    x, y = float(xi[0]), float(xi[1])
    l0 = 1.0 - x - y
    return np.array(
        [
            [-(4.0 * l0 - 1.0), -(4.0 * l0 - 1.0)],
            [4.0 * x - 1.0, 0.0],
            [0.0, 4.0 * y - 1.0],
            [4.0 * (l0 - x), -4.0 * x],
            [4.0 * y, 4.0 * x],
            [-4.0 * y, 4.0 * (l0 - y)],
        ]
    )


def nodal_velocity_gradient(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Nodal ``du_c/dx_b`` from the P2 velocity field, plus the averaging count.

    Returns ``grad`` of shape ``(N, 2, 2)`` with ``grad[n, b, c] = du_c/dx_b``
    (``x_0 = r``, ``x_1 = z``; ``u_0 = u_r``, ``u_1 = u_z``), and the number of
    element contributions averaged at each node. Nodes with no contribution -
    which cannot happen for a conforming mesh but would otherwise propagate
    silently - are returned as NaN.
    """
    coords = frame.points[frame.cells]  # (M, 6, 2)
    uvals = frame.velocity[frame.cells]  # (M, 6, 2)

    acc = np.zeros((frame.points.shape[0], 2, 2))
    count = np.zeros(frame.points.shape[0])

    for k in range(6):
        dN = _dshape(_NODE_XI[k])  # (6, 2)
        # jac[e, a, b] = dx_b/dxi_a
        jac = np.einsum("ia,eib->eab", dN, coords)
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        # A degenerate element contributes nothing rather than an infinity.
        ok = np.abs(det) > 1e-14
        # dudxi[e, a, c] = du_c/dxi_a
        dudxi = np.einsum("ia,eic->eac", dN, uvals[ok])
        grad_e = np.einsum("eba,eac->ebc", np.linalg.inv(jac[ok]), dudxi)
        np.add.at(acc, frame.cells[ok, k], grad_e)
        np.add.at(count, frame.cells[ok, k], 1.0)

    bad = count == 0
    count[bad] = np.nan
    return acc / count[:, None, None], count


def strain_invariants(frame: Frame) -> dict[str, np.ndarray]:
    """Nodal strain-rate components, ``2 E:E`` and the incompressibility residual."""
    grad, _ = nodal_velocity_gradient(frame)
    r = frame.points[:, 0]
    u_r = frame.velocity[:, 0]

    e_rr = grad[:, 0, 0]  # du_r/dr
    e_zz = grad[:, 1, 1]  # du_z/dz
    e_rz = 0.5 * (grad[:, 1, 0] + grad[:, 0, 1])  # (du_r/dz + du_z/dr)/2
    # Hoop term. u_r vanishes linearly on the axis, so u_r/r -> du_r/dr there.
    with np.errstate(divide="ignore", invalid="ignore"):
        e_tt = np.where(r > R_EPS, u_r / np.where(r > R_EPS, r, 1.0), e_rr)

    phi = 2.0 * (e_rr**2 + e_zz**2 + e_tt**2 + 2.0 * e_rz**2)
    div = e_rr + e_zz + e_tt
    scale = np.sqrt(np.nansum(grad**2, axis=(1, 2)) + e_tt**2)
    return {"phi": phi, "div": div, "grad_norm": scale}


def divergence_check(frame: Frame) -> float:
    """``rms(div u) / rms(|grad u|)`` from the reconstructed nodal gradients.

    Zero for the exact solution, so any excess measures the error introduced by
    the elementwise gradient and its nodal averaging. This is the number that
    decides whether the lower half may honestly be labelled "dissipation".
    """
    inv = strain_invariants(frame)
    denom = float(np.sqrt(np.nanmean(inv["grad_norm"] ** 2)))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(inv["div"] ** 2)) / denom)


def lower_field(frame: Frame, kind: str) -> np.ndarray:
    """Nodal values for the half below the axis."""
    if kind == "pressure":
        return frame.pressure
    inv = strain_invariants(frame)
    phi = inv["phi"]
    floor = np.nanmax(phi) * 1e-12 if np.isfinite(np.nanmax(phi)) else 1e-30
    return np.log10(np.maximum(phi, max(floor, 1e-30)))


def speed(frame: Frame) -> np.ndarray:
    return np.hypot(frame.velocity[:, 0], frame.velocity[:, 1])


# ------------------------------------------------------------------- interface


def boundary_edges(cells: np.ndarray) -> list[tuple[int, int, int]]:
    """``(corner, midside, corner)`` triples for edges owned by exactly one element."""
    local = ((0, 3, 1), (1, 4, 2), (2, 5, 0))
    seen: dict[frozenset, tuple[tuple[int, int, int], int]] = {}
    for a, m, b in local:
        for ea, em, eb in zip(cells[:, a], cells[:, m], cells[:, b]):
            key = frozenset((int(ea), int(eb)))
            if key in seen:
                trip, n = seen[key]
                seen[key] = (trip, n + 1)
            else:
                seen[key] = ((int(ea), int(em), int(eb)), 1)
    return [trip for trip, n in seen.values() if n == 1]


def interface_polyline(frame: Frame, r_tube: float, l_domain: float) -> np.ndarray:
    """The liquid/void interface as an ordered ``(K, 2)`` array of ``(z, r)``.

    The interface is what is left of the mesh boundary once the axis, wall,
    inlet and outlet are removed - it is a boundary of the liquid domain, not an
    isosurface of a captured field. The edge chain is walked from an endpoint so
    that a cap which folds back in ``z`` is still drawn as one continuous curve;
    sorting on ``z`` would quietly cut the corner.
    """
    pts = frame.points
    keep: list[tuple[int, int, int]] = []
    for a, m, b in boundary_edges(frame.cells):
        ra, rb = pts[a, 0], pts[b, 0]
        za, zb = pts[a, 1], pts[b, 1]
        on_axis = ra < BOUNDARY_TOL and rb < BOUNDARY_TOL
        on_wall = abs(ra - r_tube) < BOUNDARY_TOL and abs(rb - r_tube) < BOUNDARY_TOL
        on_inlet = za < BOUNDARY_TOL and zb < BOUNDARY_TOL
        on_outlet = abs(za - l_domain) < BOUNDARY_TOL and abs(zb - l_domain) < BOUNDARY_TOL
        if not (on_axis or on_wall or on_inlet or on_outlet):
            keep.append((a, m, b))
    if not keep:
        return np.empty((0, 2))

    adj: dict[int, list[tuple[int, int]]] = {}
    for a, m, b in keep:
        adj.setdefault(a, []).append((m, b))
        adj.setdefault(b, []).append((m, a))

    ends = [n for n, nb in adj.items() if len(nb) == 1]
    start = min(ends, key=lambda n: pts[n, 1]) if ends else min(adj, key=lambda n: pts[n, 1])

    order = [start]
    visited = {start}
    node = start
    while True:
        nxt = [(m, b) for m, b in adj.get(node, []) if b not in visited]
        if not nxt:
            break
        m, b = nxt[0]
        order.extend([m, b])
        visited.add(b)
        node = b

    chain = pts[np.array(order, dtype=np.int64)]
    return np.column_stack([chain[:, 1], chain[:, 0]])  # (z, r)


# --------------------------------------------------------------------- drawing


def subdivided_triangulation(frame: Frame, mirror: bool) -> Triangulation:
    """Plotting triangulation with four straight sub-triangles per P2 element."""
    tris = frame.cells[:, _SUBTRIS].reshape(-1, 3)
    z = frame.points[:, 1]
    r = frame.points[:, 0] * (-1.0 if mirror else 1.0)
    return Triangulation(z, r, tris)


@dataclass(frozen=True)
class Layout:
    """Everything that must be identical in every frame."""

    r_tube: float
    l_domain: float
    fig_w: float
    fig_h: float
    rect: tuple[float, float, float, float]
    ylim: float
    zticks: tuple[float, ...]
    dpi: int
    ca_in: float
    lower: str
    vmin: float
    vmax: float
    dmin: float
    dmax: float
    # Per-frame (Ca_in, Ca_b) for a continuation walk. ``None`` for the ordinary
    # case, where Ca_in is fixed and the frame's own time is the varying label.
    trail: tuple[tuple[float, float], ...] | None = None


def make_layout(
    *, r_tube, l_domain, ca_in, lower, limits, dpi, fig_w, trail=None
) -> Layout:
    ylim = 1.06 * r_tube
    left, right = 0.055, 0.865
    panel_w_in = fig_w * (right - left)
    panel_h_in = panel_w_in * (2.0 * ylim) / l_domain
    top_in, bottom_in = 1.00, 1.05
    fig_h = panel_h_in + top_in + bottom_in
    rect = (left, bottom_in / fig_h, right - left, panel_h_in / fig_h)
    step = 1.0 if l_domain <= 12 else 2.0
    zticks = tuple(np.arange(0.0, l_domain + 1e-9, step))
    return Layout(
        r_tube=r_tube, l_domain=l_domain, fig_w=fig_w, fig_h=fig_h, rect=rect,
        ylim=ylim, zticks=zticks, dpi=dpi, ca_in=ca_in, lower=lower,
        vmin=limits["vmin"], vmax=limits["vmax"],
        dmin=limits["dmin"], dmax=limits["dmax"],
        trail=trail,
    )


def draw_frame(frame: Frame, lay: Layout, dest: Path) -> Path:
    fig = plt.figure(figsize=(lay.fig_w, lay.fig_h))
    ax = fig.add_axes(lay.rect)

    upper = speed(frame)
    lower = lower_field(frame, lay.lower)
    cmap_lower = CMAP_PRESSURE if lay.lower == "pressure" else CMAP_DISS

    ax.tripcolor(
        subdivided_triangulation(frame, mirror=False), upper,
        shading="gouraud", cmap=CMAP_SPEED, vmin=lay.vmin, vmax=lay.vmax,
        rasterized=True,
    )
    ax.tripcolor(
        subdivided_triangulation(frame, mirror=True), lower,
        shading="gouraud", cmap=cmap_lower, vmin=lay.dmin, vmax=lay.dmax,
        rasterized=True,
    )

    line = interface_polyline(frame, lay.r_tube, lay.l_domain)
    if len(line):
        ax.plot(line[:, 0], line[:, 1], color="black", lw=1.6, zorder=6)
        ax.plot(line[:, 0], -line[:, 1], color="black", lw=1.6, zorder=6)

    # The wall is rigid and is drawn at +/- R_tube, never at a data extreme.
    for sgn in (1.0, -1.0):
        ax.plot([0.0, lay.l_domain], [sgn * lay.r_tube] * 2, color="dimgrey",
                lw=4.0, solid_capstyle="butt", zorder=7)
    ax.axhline(0.0, color="grey", lw=0.7, ls=(0, (6, 6)), zorder=5)

    ax.set_xlim(0.0, lay.l_domain)
    ax.set_ylim(-lay.ylim, lay.ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(lay.zticks)
    ax.set_yticks([-lay.r_tube, 0.0, lay.r_tube])
    ax.set_yticklabels([r"$-R_t$", r"$0$", r"$R_t$"])
    ax.set_xlabel(r"$z/R_0$", fontsize=FS_LABEL, labelpad=4)
    ax.set_ylabel(r"$r/R_0$", fontsize=FS_LABEL, labelpad=4)
    ax.tick_params(which="both", direction="out", width=1.2, labelsize=FS_TICK, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Label what varies along the sequence. A relaxation holds Ca_in fixed and
    # advances in time, so the time is the informative half. A continuation walk
    # advances in Ca_in and every frame is a converged steady state, so its time
    # is a meaningless output counter and printing it would imply an evolution
    # that is not there.
    if lay.trail is not None:
        ca_in_f, ca_b_f = lay.trail[frame.index]
        caption = rf"$Ca_{{\rm in}} = {ca_in_f:.3f}$,   $Ca_b = {ca_b_f:.3f}$"
    else:
        caption = rf"$Ca_{{\rm in}} = {lay.ca_in:g}$,   $t = {frame.time:5.2f}$"
    fig.text(
        0.5, 1.0 - 0.30 / lay.fig_h, caption,
        ha="center", va="center", fontsize=FS_TITLE,
    )
    # No explanatory prose is drawn on the frames. Vatsal's standing instruction
    # is that rendered output carries no text beyond what identifies the plotted
    # quantities - axis labels, ticks and colourbar labels. The frame is in the
    # bubble frame with the wall at -U, the lower half is the dissipation (or
    # pressure) field, and sigma = mu = R0 = 1; all of that belongs in the
    # caption of whatever document embeds the video, not burned into the pixels.

    x0, y0, w, h = lay.rect
    cax_u = fig.add_axes([x0 + w + 0.018, y0 + 0.52 * h, 0.010, 0.46 * h])
    cax_l = fig.add_axes([x0 + w + 0.018, y0 + 0.02 * h, 0.010, 0.46 * h])
    cb_u = fig.colorbar(
        plt.cm.ScalarMappable(cmap=CMAP_SPEED, norm=plt.Normalize(lay.vmin, lay.vmax)),
        cax=cax_u,
    )
    cb_l = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_lower, norm=plt.Normalize(lay.dmin, lay.dmax)),
        cax=cax_l,
    )
    cb_u.set_label(r"$|u|$", fontsize=FS_CBAR_LABEL, labelpad=6)
    cb_l.set_label(
        r"$p$" if lay.lower == "pressure" else r"$\log_{10}\Phi$",
        fontsize=FS_CBAR_LABEL, labelpad=6,
    )
    for cb in (cb_u, cb_l):
        cb.ax.tick_params(labelsize=FS_CBAR_TICK)

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=lay.dpi)
    plt.close(fig)
    return dest


def _render_one(item: tuple[int, float, str], lay: Layout, outdir: str) -> str | None:
    index, time, path = item
    try:
        frame = load_frame(index, time, Path(path))
        return str(draw_frame(frame, lay, Path(outdir) / f"frame-{index:05d}.png"))
    # SystemExit, not just Exception: load_frame raises SystemExit for a snapshot
    # without triangle6 cells, and SystemExit derives from BaseException, so it
    # would escape this handler and kill the pool worker rather than being
    # counted as one failed frame.
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - one bad snapshot must not kill the video
        print(f"  frame {index} failed: {exc}", file=sys.stderr)
        return None


# ------------------------------------------------------------ colour limits, QA


def colour_limits(series, lower_kind: str, n_probe: int = 10) -> dict[str, float]:
    """Fixed colour limits for the whole sequence, from a spread of frames.

    Fixed limits are the point: with per-frame autoscaling a change in
    brightness means the normalisation moved, not that the physics did.
    """
    idx = np.unique(np.linspace(0, len(series) - 1, min(n_probe, len(series))).astype(int))
    speeds, lows = [], []
    for i in idx:
        t, path = series[int(i)]
        frame = load_frame(int(i), t, path)
        speeds.append(np.nanpercentile(speed(frame), 99.8))
        vals = lower_field(frame, lower_kind)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lows.append(np.nanpercentile(vals, [1.0, 99.8]))
    if not speeds or not lows:
        raise SystemExit("could not establish colour limits")
    dmax = float(np.max([v[1] for v in lows]))
    dmin = float(np.min([v[0] for v in lows]))
    if lower_kind != "pressure":
        # Five decades shows the film and the caps together; more washes the
        # film detail out.
        dmin = max(dmin, dmax - 5.0)
    return {"vmin": 0.0, "vmax": float(np.max(speeds)), "dmin": dmin, "dmax": dmax}


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


# --------------------------------------------------------------------- driving


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Animate the co-moving void bubble.")
    ap.add_argument("output_dir", type=Path, help="pyoomph output directory (holds liquid.pvd)")
    ap.add_argument("--out", type=Path, default=None, help="MP4 path")
    ap.add_argument("--frames-dir", type=Path, default=None, help="where PNG frames are written")
    ap.add_argument("--stills-dir", type=Path, default=None, help="where the three stills go")
    ap.add_argument("--stills", type=int, default=3, help="number of stills (0 disables)")
    ap.add_argument("--still-times", type=str, default=None,
                    help="comma-separated times for the stills; the nearest recorded "
                         "snapshot is used. Overrides --stills. Useful because the "
                         "t = 0 snapshot is the seeded state with u = 0 everywhere, "
                         "which is a true initial condition but a poor 'early' still.")
    ap.add_argument("--stills-only", action="store_true", help="skip frames and encoding")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4, help="render workers (shared host: keep small)")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--fig-width", type=float, default=14.5, help="figure width in inches")
    ap.add_argument("--ca-in", type=float, default=0.05, help="Ca_in, for the title only")
    ap.add_argument("--skip-first", type=int, default=0,
                    help="drop this many leading snapshots. A continuation run has to "
                         "reach its starting state by a transient first, and those "
                         "frames are not converged; including them would put a "
                         "relaxation at the front of a walk along the steady curve.")
    ap.add_argument("--ca-trail", type=Path, default=None,
                    help="JSON with a 'trail' list of {ca_in, ca_b}, one entry per "
                         "retained frame. Switches the caption from (Ca_in, t) to "
                         "(Ca_in, Ca_b), which is what varies in a continuation walk.")
    ap.add_argument("--lower", choices=("dissipation", "pressure"), default="dissipation",
                    help="field drawn below the axis")
    ap.add_argument("--check", action="store_true",
                    help="report rms(div u)/rms(grad u) for the gradient reconstruction")
    args = ap.parse_args(argv)

    if args.workers < 1:
        ap.error("--workers must be at least 1")
    if args.fps < 1:
        ap.error("--fps must be at least 1")

    if args.skip_first < 0:
        ap.error("--skip-first must not be negative")

    output_dir = args.output_dir.resolve()
    series = read_series(output_dir)
    if args.skip_first:
        if args.skip_first >= len(series):
            ap.error(f"--skip-first {args.skip_first} leaves no snapshots of {len(series)}")
        print(f"dropping the first {args.skip_first} of {len(series)} snapshots", flush=True)
        series = series[args.skip_first:]
    print(f"{len(series)} snapshots, t = {series[0][0]:g} to {series[-1][0]:g}", flush=True)

    trail = None
    if args.ca_trail is not None:
        raw = json.loads(args.ca_trail.read_text())["trail"]
        if len(raw) != len(series):
            # Silently zipping to the shorter of the two would mislabel every
            # frame after the first mismatch, and the video would still look fine.
            ap.error(
                f"--ca-trail has {len(raw)} entries but {len(series)} snapshots are "
                "retained; the captions would not correspond to the frames"
            )
        trail = tuple((float(e["ca_in"]), float(e["ca_b"])) for e in raw)

    probe = load_frame(0, series[0][0], series[0][1])
    r_tube = float(np.max(probe.points[:, 0]))
    l_domain = float(np.max(probe.points[:, 1]))
    print(f"R_tube = {r_tube:.6g}, L_domain = {l_domain:.6g}", flush=True)

    if args.check:
        for i in np.unique(np.linspace(0, len(series) - 1, 5).astype(int)):
            t, path = series[int(i)]
            frame = load_frame(int(i), t, path)
            print(f"  t = {t:6.3f}: rms(div u)/rms(grad u) = {divergence_check(frame):.3e}",
                  flush=True)

    limits = colour_limits(series, args.lower)
    print(
        f"colour limits: |u| {limits['vmin']:.4g} -> {limits['vmax']:.4g}, "
        f"{args.lower} {limits['dmin']:.4g} -> {limits['dmax']:.4g}", flush=True,
    )

    lay = make_layout(
        r_tube=r_tube, l_domain=l_domain, ca_in=args.ca_in, lower=args.lower,
        limits=limits, dpi=args.dpi, fig_w=args.fig_width, trail=trail,
    )

    out_mp4 = args.out or (output_dir / "bubble.mp4")
    frames_dir = args.frames_dir or (output_dir / "frames")
    stills_dir = args.stills_dir or out_mp4.parent

    if args.still_times:
        wanted = [float(v) for v in args.still_times.split(",") if v.strip()]
        times = np.array([t for t, _ in series])
        picks = np.array([int(np.argmin(np.abs(times - w))) for w in wanted])
    elif args.stills:
        picks = np.unique(np.linspace(0, len(series) - 1, args.stills).astype(int))
    else:
        picks = np.empty(0, dtype=int)

    if picks.size:
        names = ["early", "middle", "late"] if len(picks) == 3 else [f"s{i}" for i in range(len(picks))]
        stills_dir.mkdir(parents=True, exist_ok=True)
        for name, i in zip(names, picks):
            t, path = series[int(i)]
            dest = stills_dir / f"still-{name}.png"
            draw_frame(load_frame(int(i), t, path), lay, dest)
            print(f"wrote {dest} (t = {t:g})", flush=True)

    if args.stills_only:
        return 0

    # Frames are always regenerated: reusing them silently mixes output from
    # different colour limits or solver versions into one video.
    if frames_dir.is_dir():
        for old in frames_dir.glob("frame-*.png"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    items = [(i, t, str(p)) for i, (t, p) in enumerate(series)]
    with Pool(args.workers) as pool:
        made = pool.map(partial(_render_one, lay=lay, outdir=str(frames_dir)), items)
    made = [Path(m) for m in made if m]
    print(f"rendered {len(made)} of {len(items)} frames", flush=True)
    if not made:
        raise SystemExit("no frames were rendered")
    if len(made) != len(items):
        # ffmpeg's image2 demuxer stops at the first gap in the numbering, so a
        # partial render does not fail - it silently produces a short video that
        # looks perfectly well formed. Refuse rather than encode a truncated
        # animation whose last frame is not the last state.
        raise SystemExit(
            f"only {len(made)} of {len(items)} frames rendered; the numbering has "
            "gaps and ffmpeg would stop at the first one, silently producing a "
            "truncated animation"
        )

    sizes = {png_size(p) for p in made}
    if len(sizes) != 1:
        raise SystemExit(f"frame sizes are not uniform: {sorted(sizes)}")
    width, height = sizes.pop()
    print(f"frame size {width}x{height}", flush=True)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    encode = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame-%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out_mp4),
    ]
    if shutil.which("ffmpeg") is None:
        print(
            "ffmpeg is not on PATH. The frames are complete and numerically\n"
            "ordered; encode them with:\n\n  " + " ".join(encode) + "\n",
            file=sys.stderr,
        )
        return 2
    run = subprocess.run(encode, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        print(run.stderr[-2000:], file=sys.stderr)
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out_mp4} ({out_mp4.stat().st_size / 1e6:.2f} MB, "
          f"{len(made)} frames at {args.fps} fps = {len(made) / args.fps:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
