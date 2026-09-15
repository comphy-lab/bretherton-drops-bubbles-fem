"""Azimuthal growth rates for the void-bubble validation report.

Inputs are the compact CSV in ../data. Outputs are the PDF and PNG beside this
script. No simulation dumps and no solver are needed.

The plotted quantity is Re(lambda) R_t / U_b. Raw lambda is in units of
sigma/(mu R_0) and its meaning changes across five decades; the scaled rate
does not. Zero is invariant.
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

C_K = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}
MARK = {0: "o", 1: "s", 2: "D"}


def read(name):
    with (DATA / name).open() as fh:
        return list(csv.DictReader(fh))


rows = read("s14-azimuthal-growth.csv")
assert len(rows) == 99
assert all(r["label"] == "stable" for r in rows)

FS_LABEL, FS_TICK, FS_LEGEND = 36, 26, 22
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Computer Modern Roman"],
    "text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}",
    "pdf.fonttype": 42, "axes.linewidth": 2.6,
})

fig, ax = plt.subplots(figsize=(10.4, 7.4))
ax.tick_params(which="both", direction="out", width=2.6, labelsize=FS_TICK, pad=8)
ax.tick_params(which="major", length=12)
ax.tick_params(which="minor", length=6)
ax.minorticks_on()
for spine in ax.spines.values():
    spine.set_linewidth(2.6)

ax.axhline(0.0, color="0.35", lw=1.8, zorder=1)

for k in (0, 1, 2):
    prod = [r for r in rows if int(r["azimuthal_k"]) == k and r["role"] == "production"]
    fine = [r for r in rows if int(r["azimuthal_k"]) == k and r["role"] == "fine"]
    prod.sort(key=lambda r: float(r["ca_b"]))
    fine.sort(key=lambda r: float(r["ca_b"]))
    ax.plot(
        [float(r["ca_b"]) for r in prod],
        [float(r["growth_over_rate"]) for r in prod],
        color=C_K[k], lw=2.0, marker=MARK[k], ms=7.0, mfc="white", mew=1.6,
        zorder=5, label=rf"$k={k}$",
    )
    if fine:
        ax.scatter(
            [float(r["ca_b"]) for r in fine],
            [float(r["growth_over_rate"]) for r in fine],
            s=70, marker=MARK[k], facecolors="none", edgecolors=C_K[k],
            linewidths=1.8, zorder=6,
        )

ax.set(xscale="log", xlim=(6e-5, 25), ylim=(-12.4, 0.5))
ax.set_xlabel(r"$Ca_b=\mu U_b/\sigma$", fontsize=FS_LABEL, labelpad=12)
ax.set_ylabel(r"$\mathrm{Re}(\lambda)\,R_t/U_b$", fontsize=FS_LABEL, labelpad=12)
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.legend(loc="lower right", fontsize=FS_LEGEND, frameon=False, handlelength=1.8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"azimuthal-growth.{ext}", dpi=300,
                bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print("wrote azimuthal-growth.pdf and .png")
