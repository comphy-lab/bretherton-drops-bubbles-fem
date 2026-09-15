"""Streamlines around the re-entrant rear cap, in the frame of the bubble.

Inputs are ../data/rear-velocity-cab9.8.csv.gz and
../data/rear-interface-cab9.8.csv, both exported from one converged solution --
the sweep row at Ca_in = 4, Ca_b = 9.795. Outputs are the PDF and PNG beside
this script. No solver and no simulation dump are needed.

The frame matters and is the whole point. In the laboratory frame every
streamline near the bubble simply follows the flow; it is only in the bubble
frame, where the bubble is stationary and the wall runs rearward at -U_b, that a
closed recirculation can appear at all. Far behind the bubble the profile there
is Poiseuille minus U_b, which is negative at every radius, so any forward-moving
liquid is a local structure and not the far field intruding.

The reversed-flow region is shaded rather than left to the streamlines alone.
The eddy turns over at a few per cent of the through-flow speed, so at any line
density that renders the wake legible the closed loop is easy to miss.
"""

from pathlib import Path
import csv
import gzip

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"

FS_LABEL, FS_TICK, FS_LEGEND = 36, 26, 22
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Computer Modern Roman"],
    "text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}",
    "pdf.fonttype": 42, "axes.linewidth": 2.6,
})
C_REVERSED, C_INTERFACE, C_WALL = "#e5f0e3", "#111111", "0.25"

with gzip.open(DATA / "rear-velocity-cab9.8.csv.gz", "rt") as fh:
    rows = list(csv.DictReader(fh))
# streamplot requires exactly equally spaced axes; the CSV stores six decimals,
# so the round trip leaves ragged spacing that trips its check. Rebuild the axes
# from their endpoints and count, and index by position rather than by value.
z_vals = sorted({float(r["z_over_R0"]) for r in rows})
r_vals = sorted({float(r["r_over_R0"]) for r in rows})
z = np.linspace(z_vals[0], z_vals[-1], len(z_vals))
r = np.linspace(r_vals[0], r_vals[-1], len(r_vals))
UZ = np.full((r.size, z.size), np.nan)
UR = np.full((r.size, z.size), np.nan)
zi = {v: k for k, v in enumerate(z_vals)}
ri = {v: k for k, v in enumerate(r_vals)}
for row in rows:
    if int(row["in_liquid"]):
        i, j = ri[float(row["r_over_R0"])], zi[float(row["z_over_R0"])]
        UZ[i, j], UR[i, j] = float(row["uz"]), float(row["ur"])

with (DATA / "rear-interface-cab9.8.csv").open() as fh:
    iface = np.array([[float(x["z_over_R0"]), float(x["r_over_R0"])]
                      for x in csv.DictReader(fh)])
r_tube = float(r.max())
z_tip = float(iface[0, 0])
assert np.isfinite(UZ).any(), "no liquid nodes read"

# The rear cap is the leading arc of the polyline, up to where it reaches the
# film; drawing the whole bubble would put the front cap off the panel.
in_window = (iface[:, 0] >= z[0] - 0.05) & (iface[:, 0] <= z[-1] + 0.05)
cap = iface[in_window]

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2),
                         gridspec_kw={"width_ratios": [1.45, 1]})
windows = [(z[0], z[-1], 0.0, r_tube), (z_tip - 0.30, z_tip + 0.06, 0.0, 0.42 * r_tube)]
densities = [1.5, 3.2]

for ax, (z0, z1, r0, r1), dens in zip(axes, windows, densities):
    ax.contourf(z, r, np.where(np.isfinite(UZ) & (UZ > 0), 1.0, np.nan),
                levels=[0.5, 1.5], colors=[C_REVERSED], zorder=1)
    ax.streamplot(z, r, UZ, UR, density=dens, color="0.35", linewidth=1.3,
                  arrowsize=1.5, broken_streamlines=False, zorder=2)
    ax.plot(cap[:, 0], cap[:, 1], color=C_INTERFACE, lw=3.0, zorder=4)
    ax.axhline(r_tube, color=C_WALL, lw=3.0, zorder=3)
    ax.axhline(0.0, color="0.75", lw=1.2, ls=":", zorder=3)
    ax.set_xlim(z0, z1)
    ax.set_ylim(r0, r1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$z/R_0$", fontsize=FS_LABEL, labelpad=12)
    ax.set_ylabel(r"$r/R_0$", fontsize=FS_LABEL, labelpad=12)
    ax.tick_params(which="both", direction="out", width=2.6, labelsize=FS_TICK, pad=8)
    ax.tick_params(which="major", length=12)
    ax.tick_params(which="minor", length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2.6)

axes[0].set_yticks([0.0, 0.5 * r_tube, r_tube])
axes[0].set_yticklabels([r"$0$", r"$R_t/2$", r"$R_t$"])
axes[1].set_yticks([0.0, 0.2 * r_tube, 0.4 * r_tube])
axes[1].set_yticklabels([r"$0$", r"$0.2R_t$", r"$0.4R_t$"])

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"rear-flow.{ext}", dpi=300, bbox_inches="tight",
                pad_inches=0.06)
plt.close(fig)

fwd = np.isfinite(UZ) & (UZ > 0)
print(f"wrote rear-flow.pdf and .png; {int(fwd.sum())} reversed-flow nodes, "
      f"peak {np.nanmax(np.where(fwd, UZ, np.nan)):+.4f} against a far-field "
      f"axis value of {UZ[0, 0]:+.4f}")
