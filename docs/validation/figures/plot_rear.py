"""Bubble shapes and the re-entrant rear cap.

Inputs are the compact CSVs in ../data. Outputs are the PDF and PNG beside this
script. No simulation dumps and no solver are needed.

The left panel shows converged profiles, one per decade, each shifted so its
rear tip sits at the origin; without that shift the bubbles sit at different
stations in their own windows and the rear caps cannot be compared. The right
panel is the indentation of the rear cap against capillary number, at two film
resolutions, because near onset the indentation is small enough that it has to
be shown to be a property of the solution rather than of the mesh.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
R_TUBE = 0.7   # R_t/R_0 for the whole campaign

# Font sizes follow the house scale, adjusted for the width at which the figure
# is embedded. The companion report renders a 12-inch figure at \textwidth with
# 28 pt labels; these are 15.5 inches wide, so the same on-page size is
# 28 * 15.5 / 12 = 36 pt, and the tick and legend sizes scale with it.
FS_LABEL, FS_TICK, FS_LEGEND = 36, 26, 22

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Computer Modern Roman"],
    "text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}",
    "pdf.fonttype": 42, "axes.linewidth": 2.6,
})

profiles = []
for path in sorted((DATA / "profiles").glob("profile-cab*.csv"),
                   key=lambda p: float(p.stem.split("cab")[1])):
    with path.open() as fh:
        pts = np.array([[float(r["z_over_R0"]), float(r["r_over_R0"])]
                        for r in csv.DictReader(fh)])
    profiles.append((float(path.stem.split("cab")[1]), pts))
assert profiles, "no vendored profiles found"

fig, (ax, ind) = plt.subplots(1, 2, figsize=(15.5, 6.4),
                              gridspec_kw={"width_ratios": [1.25, 1]})
for a in (ax, ind):
    a.tick_params(which="both", direction="out", width=2.6, labelsize=FS_TICK, pad=8)
    a.tick_params(which="major", length=12)
    a.tick_params(which="minor", length=6)
    for spine in a.spines.values():
        spine.set_linewidth(2.6)

colours = plt.cm.viridis(np.linspace(0.08, 0.9, len(profiles)))
for (ca_b, pts), colour in zip(profiles, colours):
    z, r = pts[:, 0] - pts[:, 0].min(), pts[:, 1]
    # Mirrored about the axis so the bubble reads as one object rather than as
    # a curve that happens to start and end on r = 0.
    ax.plot(np.concatenate([z, z[::-1]]), np.concatenate([r, -r[::-1]]),
            color=colour, lw=2.0, label=rf"$Ca_b={ca_b:.4g}$")
ax.axhline(R_TUBE, color="0.25", lw=1.8)
ax.axhline(-R_TUBE, color="0.25", lw=1.8)
ax.axhline(0.0, color="0.75", lw=1.0, ls=":")
ax.set_xlim(-0.12, 2.1)
ax.set_ylim(-1.12 * R_TUBE, 1.12 * R_TUBE)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$(z-z_{\mathrm{rear}})/R_0$", fontsize=FS_LABEL, labelpad=12)
ax.set_ylabel(r"$r/R_0$", fontsize=FS_LABEL, labelpad=12)
ax.set_yticks([-R_TUBE, 0.0, R_TUBE])
ax.set_yticklabels([r"$-R_t$", r"$0$", r"$R_t$"])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
          fontsize=19, frameon=False, handlelength=1.6, columnspacing=1.4)
ax.annotate("", xy=(1.62, 0.82 * R_TUBE), xytext=(0.95, 0.82 * R_TUBE),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color="0.3"))

with (DATA / "rear-cap.csv").open() as fh:
    sweep = list(csv.DictReader(fh))
ca_s = np.array([float(r["Ca_b"]) for r in sweep])
d_s = np.array([float(r["indentation_over_Rt"]) for r in sweep])
with (DATA / "tail-onset.csv").open() as fh:
    onset = list(csv.DictReader(fh))
ca_o = np.array([float(r["Ca_b"]) for r in onset])
d3 = np.array([float(r["indentation_n3"]) for r in onset])
d6 = np.array([float(r["indentation_n6"]) for r in onset])

# Taylor's transition: on the axis far from the bubble the liquid changes
# direction in the bubble frame when Ca_b/Ca_in = 2, i.e. m = 1/2.
m = np.array([float(r["m"]) for r in sweep])
cross = np.interp(0.5, m, np.log(ca_s))
ind.axvline(np.exp(cross), color="#D55E00", lw=1.8, ls="--", zorder=2)
ind.text(np.exp(cross) * 0.90, 2e-3, r"$m=\tfrac12$", rotation=90,
         ha="right", va="center", fontsize=FS_TICK, color="#D55E00")

pos = d_s > 0
ind.plot(ca_s[pos], d_s[pos], color="#009E73", lw=2.0, marker="o", ms=6.5,
         mfc="white", mew=1.5, zorder=4, label=r"sweep, $n_{\mathrm{film}}=3$")
ind.plot(ca_o[d3 > 0], d3[d3 > 0], ls="none", marker="^", ms=8,
         color="#009E73", mfc="none", mew=1.5, zorder=5,
         label=r"onset scan, $n_{\mathrm{film}}=3$")
ind.plot(ca_o[d6 > 0], d6[d6 > 0], ls="none", marker="v", ms=8,
         color="#0072B2", mfc="none", mew=1.5, zorder=5,
         label=r"onset scan, $n_{\mathrm{film}}=6$")
ind.axvspan(1.024, 1.060, color="0.85", zorder=1)
ind.set(xscale="log", yscale="log", xlim=(0.5, 22), ylim=(1e-7, 0.45))
ind.set_xlabel(r"$Ca_b=\mu U_b/\sigma$", fontsize=FS_LABEL, labelpad=12)
ind.set_ylabel(r"$\delta_{\mathrm{tail}}/R_t$", fontsize=FS_LABEL, labelpad=12)
ind.xaxis.set_major_locator(LogLocator(base=10))
ind.xaxis.set_minor_formatter(NullFormatter())
ind.legend(loc="lower right", fontsize=19, frameon=False, handlelength=1.6)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"rear-cap.{ext}", dpi=300, bbox_inches="tight",
                pad_inches=0.06)
plt.close(fig)
print("wrote rear-cap.pdf and .png")
