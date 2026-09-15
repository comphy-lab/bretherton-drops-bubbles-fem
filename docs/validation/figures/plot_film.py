"""Deposited-film comparison for the void-bubble validation report.

Inputs are the compact CSVs in ../data. Outputs are the PDF and PNG beside this
script. No simulation dumps and no solver are needed.

Every deviation plotted here is recomputed from the raw ``H`` and ``Ca_b``
columns against one definition of the reference law. That is deliberate. The
companion Basilisk table carries its own deviation column computed with the
conventional rounded prefactor 1.34, whereas the unrounded Bretherton
coefficient is 0.643*3^{2/3} = 1.3375. The two differ by about 0.2% in the
thin-film limit, which is a fifth of the smallest difference this figure is
used to discuss, so mixing the conventions would put a spurious constant offset
between the two codes.
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

# Colourblind-safe, and shared with the companion report so the two read as a
# pair: orange for the empirical fit, blue for Basilisk, green for this solver.
C_FIT, C_VOF, C_FEM = "#D55E00", "#0072B2", "#009E73"
C_LIT_EDGE, C_LIT_FILL = "#9ac295", "#e5f0e3"


def read(name):
    with (DATA / name).open() as fh:
        return list(csv.DictReader(fh))


def bretherton(ca):
    """0.643 (3 Ca)^{2/3}. Unrounded, for the reason given in the module docstring."""
    return 0.643 * (3.0 * np.asarray(ca, dtype=float)) ** (2.0 / 3.0)


def aussillous_quere(ca):
    b = bretherton(ca)
    return b / (1.0 + 2.5 * b)


fem = read("film-sweep.csv")
ca_f = np.array([float(r["Ca_b"]) for r in fem])
h_f = np.array([float(r["H"]) for r in fem])

vof = read("companion-basilisk.csv")
ca_v = np.array([float(r["Ca_b"]) for r in vof])
h_v = np.array([float(r["H"]) for r in vof])
limited = np.array([str(r["outlet_limited"]).lower() == "true" for r in vof])

lit = read("giavedoni-saita-1997.csv")
ca_g = np.array([float(r["Ca"]) for r in lit])
h_g = np.array([float(r["H"]) for r in lit])

assert np.all(ca_f > 0) and np.all(h_f > 0) and len(ca_f) == 25
# The vendored reference column must agree with the function used here, or the
# figure and the table in the report would be drawn from different laws.
assert np.allclose([float(r["H_AQ"]) for r in fem], aussillous_quere(ca_f), rtol=1e-9)

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

fig, (ax, res) = plt.subplots(1, 2, figsize=(15.5, 7.4),
                              gridspec_kw={"width_ratios": [1.15, 1]})
for a in (ax, res):
    a.tick_params(which="both", direction="out", width=2.6, labelsize=FS_TICK, pad=8)
    a.tick_params(which="major", length=12)
    a.tick_params(which="minor", length=6)
    a.minorticks_on()
    for spine in a.spines.values():
        spine.set_linewidth(2.6)

curve = np.geomspace(5e-5, 25, 600)
ax.plot(curve, aussillous_quere(curve), color=C_FIT, lw=2.8, zorder=2,
        label=r"Aussillous--Qu\'er\'e")
small = np.geomspace(5e-5, 0.1, 300)
ax.plot(small, bretherton(small), "--", color="0.45", lw=2.2, zorder=2,
        label=r"Bretherton $0.643\,(3Ca_b)^{2/3}$")
ax.scatter(ca_g, h_g, s=190, marker="s", facecolors=C_LIT_FILL,
           edgecolors=C_LIT_EDGE, linewidths=1.5, zorder=3,
           label=r"Giavedoni \& Saita (1997)")
ax.scatter(ca_v[~limited], h_v[~limited], s=90, marker="s", color=C_VOF,
           edgecolors="white", linewidths=0.8, zorder=4,
           label=r"Companion Basilisk")
if limited.any():
    ax.scatter(ca_v[limited], h_v[limited], s=110, marker="s", facecolors="none",
               edgecolors=C_VOF, linewidths=1.8, zorder=4,
               label="Companion, outlet-limited")
ax.plot(ca_f, h_f, color=C_FEM, lw=2.0, marker="o", ms=6.5, mfc="white",
        mew=1.5, zorder=5, label="This solver")

ax.set(xscale="log", yscale="log", xlim=(6e-5, 25), ylim=(2e-3, 0.5))
ax.set_xlabel(r"$Ca_b=\mu U_b/\sigma$", fontsize=FS_LABEL, labelpad=12)
ax.set_ylabel(r"$H=h_{\mathrm{flat}}/R_t$", fontsize=FS_LABEL, labelpad=12)
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())


res.axhline(0.0, color=C_FIT, lw=2.2, zorder=2)
res.scatter(ca_g, 100 * (h_g / aussillous_quere(ca_g) - 1), s=170, marker="s",
            facecolors=C_LIT_FILL, edgecolors=C_LIT_EDGE, linewidths=1.4, zorder=3)
res.scatter(ca_v[~limited], 100 * (h_v[~limited] / aussillous_quere(ca_v[~limited]) - 1),
            s=80, marker="s", color=C_VOF, edgecolors="white", linewidths=0.8,
            zorder=4)
if limited.any():
    res.scatter(ca_v[limited], 100 * (h_v[limited] / aussillous_quere(ca_v[limited]) - 1),
                s=100, marker="s", facecolors="none", edgecolors=C_VOF,
                linewidths=1.8, zorder=4)
res.plot(ca_f, 100 * (h_f / aussillous_quere(ca_f) - 1), color=C_FEM, lw=2.0,
         marker="o", ms=5.5, mfc="white", mew=1.4, zorder=5)
res.set(xscale="log", xlim=(6e-5, 25), ylim=(-12, 12))
res.set_xlabel(r"$Ca_b=\mu U_b/\sigma$", fontsize=FS_LABEL, labelpad=12)
res.set_ylabel(r"$100\,(H-H_T)/H_T$", fontsize=FS_LABEL, labelpad=12)
res.xaxis.set_major_locator(LogLocator(base=10))
res.xaxis.set_minor_formatter(NullFormatter())


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
           ncol=3, fontsize=FS_LEGEND, frameon=False, handlelength=1.8,
           columnspacing=2.0)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"film-thickness-vs-ca.{ext}", dpi=300,
                bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print("wrote film-thickness-vs-ca.pdf and .png")
