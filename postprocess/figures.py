"""Render the comparison and convergence figures from the committed receipts.

Every plotted quantity is read directly from a receipt JSON (or the interface
CSV it points at), or from the parent Basilisk campaign's baseline CSV. No
value is fitted, smoothed, interpolated or extrapolated here.

Conventions (see AGENTS.md): sigma = mu = R0 = 1, R_tube/R0 = 0.7, film
thicknesses are reported as h/R_tube, and the reference laws are evaluated at
the *measured* Ca_b, never at the imposed Ca_in.

Usage
-----
    python postprocess/figures.py --outdir <dir>
    python postprocess/figures.py --list
    python postprocess/figures.py --only film_vs_cab deviation_vs_cab
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECEIPTS = REPO_ROOT / "receipts"
DEFAULT_PARENT_CSV = (
    REPO_ROOT.parent
    / "bretherton-drops-bubbles"
    / "docs"
    / "Newtonian-Validation"
    / "data"
    / "baseline-film.csv"
)

R_TUBE = 0.7  # R_tube / R0, project convention

# Colours: FEM (this work) vs parent Basilisk campaign, plus reference laws.
C_FEM = "#1f4e9c"
C_PARENT = "#d1622b"
C_BRETH = "#111111"
C_AQ = "#5c5c5c"
C_CA_HI = "#8b1a1a"
C_CA_LO = "#1f6f4e"

LBL_FEM = r"pyoomph FEM (void bubble, steady co-moving)"
LBL_PARENT = (
    r"Basilisk VoF (gas, $\mu_g/\mu_\ell = 10^{-2}$, transient lab frame)"
)


# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------
def init_style(usetex: bool = True) -> None:
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
    matplotlib.rcParams["text.usetex"] = bool(usetex)
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.linewidth"] = 1.5
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def fonts(width_inches: float, multipanel: bool = False) -> dict:
    """Font sizes scaled from the skill's 12-inch reference figure."""
    s = width_inches / 12.0
    if multipanel:
        s *= 1.12
    return {
        "label": 40.0 * s,
        "tick": 30.0 * s,
        "legend": 30.0 * s,
        "annot": 26.0 * s,
        "lw": 3.0 * s,
        "ms": 12.0 * s,
        "pad": 10.0 * s,
        "labelpad": 15.0 * s,
    }


def style_axes(ax, f: dict) -> None:
    ax.tick_params(which="both", direction="out", width=1.5 * f["lw"] / 3.0 * 2,
                   labelsize=f["tick"], pad=f["pad"])
    ax.tick_params(which="major", length=12.0 * f["lw"] / 3.0)
    ax.tick_params(which="minor", length=6.0 * f["lw"] / 3.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5 * f["lw"] / 3.0 * 2)
    ax.minorticks_on()


def clean_log_axis(axis, ticks, labels=None) -> None:
    """Explicit ticks, no minor-tick labels (they collide over short ranges)."""
    axis.set_major_locator(LogLocator(base=10.0))
    axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))
    axis.set_minor_formatter(NullFormatter())
    if ticks is not None:
        axis.set_ticks(ticks)
        if labels is not None:
            axis.set_ticklabels(labels)


def save(fig, outdir: Path, name: str, dpi_png: int) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in ((".png", dpi_png), (".pdf", 300)):
        p = outdir / f"{name}{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------
def load_receipt(receipts: Path, name: str) -> dict:
    with (receipts / name).open() as fh:
        return json.load(fh)


def load_parent(parent_csv: Path) -> list[dict]:
    with parent_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [{k: (float(v) if k != "case" else v) for k, v in r.items()} for r in rows]


def bretherton(ca: np.ndarray | float):
    """h/R_tube from Bretherton, evaluated at the measured Ca_b."""
    return 0.643 * (3.0 * np.asarray(ca, dtype=float)) ** (2.0 / 3.0)


def aussillous_quere(ca: np.ndarray | float):
    """h/R_tube from Aussillous-Quere: Bretherton value divided by 1 + 2.5 x."""
    x = bretherton(ca)
    return x / (1.0 + 2.5 * x)


def sweep_rows(receipts: Path) -> list[dict]:
    rows = load_receipt(receipts, "sweep.json")["results"]
    return sorted(rows, key=lambda r: r["ca_b"])


def interface_xy(receipts: Path, row: dict) -> np.ndarray:
    """Load the (z, r) interface polyline referenced by a sweep row."""
    rel = row.get("interface_profile")
    if not rel:
        raise KeyError(
            f"sweep row ca_in={row['ca_in']} has no 'interface_profile' field"
        )
    path = Path(rel)
    if not path.is_absolute():
        path = REPO_ROOT / rel
    return np.loadtxt(path, delimiter=",", skiprows=1)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_film_vs_cab(receipts: Path, parent_csv: Path, outdir: Path, dpi: int):
    """h/R_tube against the measured Ca_b, log-log, both codes plus laws."""
    fem = sweep_rows(receipts)
    par = load_parent(parent_csv)

    ca_f = np.array([r["ca_b"] for r in fem])
    h_f = np.array([r["h_flat_over_r_tube"] for r in fem])
    ca_p = np.array([r["Ca_b"] for r in par])
    h_p = np.array([r["b_flat_over_R"] for r in par])

    lo = 0.8 * min(ca_f.min(), ca_p.min())
    hi = 1.3 * max(ca_f.max(), ca_p.max())
    ca_line = np.logspace(math.log10(lo), math.log10(hi), 400)

    w = 7.0
    f = fonts(w)
    fig, ax = plt.subplots(figsize=(w, w * 0.82))

    ax.plot(ca_line, bretherton(ca_line), "--", color=C_BRETH, lw=f["lw"] * 0.8,
            label=r"Bretherton, $h/R_t = 0.643\,(3Ca_b)^{2/3}$", zorder=1)
    ax.plot(ca_line, aussillous_quere(ca_line), "-", color=C_AQ, lw=f["lw"] * 0.8,
            label=r"Aussillous--Qu\'er\'e", zorder=1)
    ax.plot(ca_p, h_p, "s", mfc="none", mec=C_PARENT, ms=f["ms"] * 1.8,
            mew=f["lw"] * 0.55, ls="none", label=LBL_PARENT, zorder=3)
    ax.plot(ca_f, h_f, "o", color=C_FEM, ms=f["ms"] * 0.85, mec="k",
            mew=f["lw"] * 0.2, ls="none", label=LBL_FEM, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Ca_b = \mu U_b/\sigma$", fontsize=f["label"],
                  labelpad=f["labelpad"])
    ax.set_ylabel(r"$h/R_t$", fontsize=f["label"], labelpad=f["labelpad"])
    style_axes(ax, f)
    ax.legend(loc="upper left", frameon=False, fontsize=f["legend"] * 0.62,
              handlelength=2.0, borderpad=0.2, labelspacing=0.35)
    clean_log_axis(ax.xaxis, [2e-3, 1e-2, 5e-2],
                   [r"$2\times10^{-3}$", r"$10^{-2}$", r"$5\times10^{-2}$"])
    clean_log_axis(ax.yaxis, [2e-2, 5e-2, 1e-1],
                   [r"$2\times10^{-2}$", r"$5\times10^{-2}$", r"$10^{-1}$"])
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.7 * min(h_f.min(), h_p.min()), 3.2 * max(h_f.max(), h_p.max()))
    fig.tight_layout()
    return save(fig, outdir, "film_vs_cab", dpi)


def fig_deviation_vs_cab(receipts: Path, parent_csv: Path, outdir: Path, dpi: int):
    """Relative deviation from Aussillous-Quere, per cent, for both codes."""
    fem = sweep_rows(receipts)
    par = load_parent(parent_csv)

    ca_f = np.array([r["ca_b"] for r in fem])
    d_f = 100.0 * np.array([r["deviation_from_aussillous_quere"] for r in fem])
    ca_p = np.array([r["Ca_b"] for r in par])
    d_p = np.array([r["deviation_from_taylor_percent"] for r in par])

    w = 7.0
    f = fonts(w)
    fig, ax = plt.subplots(figsize=(w, w * 0.82))

    ax.axhline(0.0, color=C_AQ, ls=":", lw=f["lw"] * 0.7, zorder=1)
    ax.plot(ca_p, d_p, "s--", mfc=C_PARENT, color=C_PARENT, ms=f["ms"], mec="k",
            mew=f["lw"] * 0.25, lw=f["lw"] * 0.6, label=LBL_PARENT, zorder=3)
    ax.plot(ca_f, d_f, "o-", color=C_FEM, ms=f["ms"] * 1.1, mec="k",
            mew=f["lw"] * 0.25, lw=f["lw"] * 0.6, label=LBL_FEM, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel(r"$Ca_b = \mu U_b/\sigma$", fontsize=f["label"],
                  labelpad=f["labelpad"])
    ax.set_ylabel(r"$100\,(h - h_{\mathrm{AQ}})/h_{\mathrm{AQ}}$",
                  fontsize=f["label"], labelpad=f["labelpad"])
    style_axes(ax, f)
    # Limits derive from the data, never hardcoded. An earlier version fixed the
    # range to the original five-point sweep, and when the sweep was extended to
    # sixteen points spanning four decades of Ca_b, eleven of them were silently
    # clipped off the axes while the figure still looked perfectly well formed.
    lo_x, hi_x = min(ca_f.min(), ca_p.min()), max(ca_f.max(), ca_p.max())
    ax.set_xlim(lo_x / 1.6, hi_x * 1.6)
    # math.floor/ceil already return int; the redundant int() calls tripped RUF046.
    decades = [10.0 ** e for e in range(
        math.floor(math.log10(lo_x)), math.ceil(math.log10(hi_x)) + 1)]
    decades = [t for t in decades if lo_x / 1.6 <= t <= hi_x * 1.6]
    clean_log_axis(ax.xaxis, decades,
                   [r"$10^{%d}$" % round(math.log10(t)) for t in decades])
    span = max(abs(min(d_f.min(), d_p.min())), abs(max(d_f.max(), d_p.max())))
    ax.set_ylim(-1.55 * span, 1.35 * span)
    ax.legend(loc="upper left", frameon=False, fontsize=f["legend"] * 0.58,
              handlelength=2.4, borderpad=0.2, labelspacing=0.35)
    ax.text(0.03, 0.04, r"$h_{\mathrm{AQ}}$ evaluated at the measured $Ca_b$",
            transform=ax.transAxes, fontsize=f["annot"] * 0.8, color="0.25")
    fig.tight_layout()
    return save(fig, outdir, "deviation_vs_cab", dpi)


def fig_window_independence(receipts: Path, parent_csv: Path, outdir: Path, dpi: int):
    """Pad independence of h, and the Poiseuille / traction-free outlet gap."""
    s5 = load_receipt(receipts, "s5_window.json")
    ref_pad = s5["reference_pad"]

    w = 11.0
    f = fonts(w / 2.0, multipanel=True)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(w, w * 0.45))

    # (a) relative change vs the reference pad, symmetric windows
    dropped = []
    for ca, colour in ((0.05, C_CA_HI), (0.005, C_CA_LO)):
        for outlet, ls, mk in (("poiseuille", "-", "o"),
                               ("traction_free", "--", "^")):
            sel = [c for c in s5["convergence"]
                   if c["ca_in"] == ca and c["outlet"] == outlet
                   and c["kind"] == "symmetric"]
            sel.sort(key=lambda c: c["pad_front"])
            pad = np.array([c["pad_front"] for c in sel])
            val = np.abs(np.array([c["h_flat_over_r_tube_rel_change"]
                                   for c in sel]))
            keep = val > 0.0
            dropped += [round(p, 1) for p in pad[~keep]]
            lbl = (r"$Ca_{\mathrm{in}}=%g$, %s"
                   % (ca, "Poiseuille" if outlet == "poiseuille"
                      else "traction free"))
            axa.plot(pad[keep], val[keep], ls, marker=mk, color=colour,
                     ms=f["ms"] * 0.8, mec="k", mew=f["lw"] * 0.2,
                     lw=f["lw"] * 0.6, label=lbl)
    axa.axvline(ref_pad, color="0.5", ls=":", lw=f["lw"] * 0.6)
    axa.text(ref_pad, 0.93, r" reference pad", transform=axa.get_xaxis_transform(),
             ha="right", va="top", rotation=90, fontsize=f["annot"] * 0.8,
             color="0.35")
    axa.set_yscale("log")
    axa.set_xlabel(r"pad $/R_t$", fontsize=f["label"], labelpad=f["labelpad"])
    axa.set_ylabel(r"$|h - h_{\mathrm{ref}}|/h_{\mathrm{ref}}$",
                   fontsize=f["label"], labelpad=f["labelpad"])
    axa.set_xlim(0.5, ref_pad + 0.5)
    axa.margins(y=0.12)
    style_axes(axa, f)
    axa.legend(loc="lower left", frameon=False, fontsize=f["legend"] * 0.5,
               handlelength=2.2, borderpad=0.2, labelspacing=0.3)
    axa.set_title(r"(a) window independence, each outlet alone",
                  fontsize=f["annot"], pad=8)

    # (b) Poiseuille vs traction-free gap at matched pad
    for ca, colour in ((0.05, C_CA_HI), (0.005, C_CA_LO)):
        sel = [g for g in s5["outlet_gap"] if g["ca_in"] == ca]
        sel.sort(key=lambda g: g["pad"])
        pad = np.array([g["pad"] for g in sel])
        gap = np.abs(np.array([g["h_flat_over_r_tube_rel_gap"] for g in sel]))
        axb.plot(pad, gap, "-o", color=colour, ms=f["ms"] * 0.8, mec="k",
                 mew=f["lw"] * 0.2, lw=f["lw"] * 0.6,
                 label=r"$Ca_{\mathrm{in}}=%g$" % ca)
    axb.set_yscale("log")
    axb.set_xlabel(r"pad $/R_t$", fontsize=f["label"], labelpad=f["labelpad"])
    axb.set_ylabel(r"$|h_{\mathrm{Pois}} - h_{\mathrm{tf}}|/h_{\mathrm{tf}}$",
                   fontsize=f["label"], labelpad=f["labelpad"])
    axb.set_xlim(0.5, ref_pad + 0.5)
    axb.margins(y=0.12)
    style_axes(axb, f)
    axb.legend(loc="upper right", frameon=False, fontsize=f["legend"] * 0.55,
               handlelength=2.0, borderpad=0.2, labelspacing=0.3)
    axb.set_title(r"(b) outlet condition gap at matched pad",
                  fontsize=f["annot"], pad=8)

    offs = {}
    for g in s5["outlet_gap"]:
        if g["pad"] >= 2.0:
            offs.setdefault(g["ca_in"], []).append(g["pressure_abs_offset"])
    note = ", ".join(
        r"$Ca_{\mathrm{in}}=%g$: $%.4f$" % (ca, float(np.mean(v)))
        for ca, v in sorted(offs.items(), reverse=True))
    fig.text(0.5, -0.02,
             r"pad-independent outlet pressure datum offset (pad $\geq 2$): "
             + note,
             ha="center", fontsize=f["annot"] * 0.75, color="0.25")
    if dropped:
        fig.text(0.5, -0.075,
                 r"(a) omits pad $=%s$: identically zero by construction, "
                 r"not representable on a logarithmic axis"
                 % ", ".join(str(int(p)) for p in sorted(set(dropped))),
                 ha="center", fontsize=f["annot"] * 0.75, color="0.25")
    fig.tight_layout()
    return save(fig, outdir, "window_independence", dpi)


def fig_mesh_convergence(receipts: Path, parent_csv: Path, outdir: Path, dpi: int):
    """Relative film change against each refinement knob, at both Ca_in."""
    s6 = load_receipt(receipts, "s6_mesh.json")
    res = s6["results"]

    def spread(ca, knob):
        for e in s6["extrapolations"]:
            if e["ca_in"] == ca and e["knob"] == knob:
                return e["h_flat_over_r_tube"]["spread"]
        return None

    w = 11.0
    f = fonts(w / 2.0, multipanel=True)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(w, w * 0.45))

    for ca, colour, mk in ((0.05, C_CA_HI, "o"), (0.005, C_CA_LO, "^")):
        # film_elements knob: bulk resolution held at the baseline value
        sel = [r for r in res
               if r["ca_in"] == ca and r.get("knob") == "film_elements"]
        sel.sort(key=lambda r: r["film_elements"])
        n = np.array([r["film_elements"] for r in sel])
        h = np.array([r["h_flat_over_r_tube"] for r in sel])
        axa.plot(n, (h - h[-1]) / h[-1] * 1e5, "-", marker=mk, color=colour,
                 ms=f["ms"] * 0.8, mec="k", mew=f["lw"] * 0.2, lw=f["lw"] * 0.6,
                 label=r"$Ca_{\mathrm{in}}=%g$" % ca)

        # bulk resolution knob: the baseline point is the film_elements=3 run
        base = [r for r in sel if r["film_elements"] == 3.0]
        sel_b = [r for r in res
                 if r["ca_in"] == ca and r.get("knob") == "bulk_resolution"]
        sel_b = sel_b + base
        sel_b.sort(key=lambda r: -r["bulk_resolution_over_r0"])
        d = np.array([r["bulk_resolution_over_r0"] for r in sel_b])
        hb = np.array([r["h_flat_over_r_tube"] for r in sel_b])
        axb.plot(d, (hb - hb[-1]) / hb[-1] * 1e5, "-", marker=mk, color=colour,
                 ms=f["ms"] * 0.8, mec="k", mew=f["lw"] * 0.2, lw=f["lw"] * 0.6,
                 label=r"$Ca_{\mathrm{in}}=%g$" % ca)

    axa.set_xlabel(r"elements across the film", fontsize=f["label"],
                   labelpad=f["labelpad"])
    axa.set_xticks([2, 3, 4, 6])
    axb.set_xlabel(r"$\Delta/R_0$", fontsize=f["label"], labelpad=f["labelpad"])
    axb.set_xticks([0.06, 0.10, 0.16])
    axb.invert_xaxis()
    for ax, knob, tag in ((axa, "film_elements", "(a)"), (axb, "bulk_resolution", "(b)")):
        ax.axhline(0.0, color="0.6", ls=":", lw=f["lw"] * 0.5)
        ax.set_ylabel(r"$10^{5}\,(h - h_{\mathrm{finest}})/h_{\mathrm{finest}}$",
                      fontsize=f["label"], labelpad=f["labelpad"])
        style_axes(ax, f)
        ax.legend(loc="best", frameon=False, fontsize=f["legend"] * 0.55,
                  handlelength=2.0, borderpad=0.2, labelspacing=0.3)
        s_hi, s_lo = spread(0.05, knob), spread(0.005, knob)
        ax.margins(y=0.14)
        ax.set_title(
            tag + r" relative spread: $%.1f\times10^{-5}$ at $Ca_{\mathrm{in}}=0.05$,"
            r" $%.1f\times10^{-5}$ at $Ca_{\mathrm{in}}=0.005$"
            % (s_hi * 1e5, s_lo * 1e5),
            fontsize=f["annot"] * 0.72, pad=8)
    fig.tight_layout()
    return save(fig, outdir, "mesh_convergence", dpi)


def fig_interface_profiles(receipts: Path, parent_csv: Path, outdir: Path, dpi: int):
    """Converged interface shapes, mirrored about the axis, with tube walls.

    Orientation follows the solver geometry: ``z_rear = pad_rear * r_tube`` and
    ``z_front = z_rear + bubble_length`` (problems/geometry.py), so larger z is
    the advancing front cap and the bubble translates towards +z. In the
    bubble frame the wall therefore moves rearward, towards -z.
    """
    fem = sweep_rows(receipts)
    profiles = [(r, interface_xy(receipts, r)) for r in fem]

    w = 9.6
    f = fonts(w * 0.62)
    zmin = min(p[:, 0].min() for _, p in profiles)
    zmax = max(p[:, 0].max() for _, p in profiles)

    # Two panels rather than a magnified inset. The sweep spans a hundredfold
    # range of film thickness, from h/R_t = 0.0029 to 0.313, and no linear zoom
    # resolves both ends at once: an equal-aspect inset sized for the thin films
    # loses the thick ones off the bottom, and one sized for the thick films
    # renders the thin ones as a single line on the wall. The lower panel plots
    # the gap R_t - r(z) logarithmically instead, which separates every case.
    fig, (ax, axg) = plt.subplots(
        2, 1, figsize=(w, w * 0.60), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})

    cmap = plt.get_cmap("viridis")
    n = len(profiles)
    for i, (row, prof) in enumerate(profiles):
        colour = cmap(0.08 + 0.78 * i / max(n - 1, 1))
        z, r = prof[:, 0], prof[:, 1]
        ax.plot(z, r, "-", color=colour, lw=f["lw"] * 0.6,
                label=r"$Ca_b = %.4f$" % row["ca_b"])
        ax.plot(z, -r, "-", color=colour, lw=f["lw"] * 0.6)
        gap = R_TUBE - r
        keep = gap > 0
        axg.plot(z[keep], gap[keep], "-", color=colour, lw=f["lw"] * 0.6)

    for sgn in (+1.0, -1.0):
        ax.plot([zmin - 0.3, zmax + 0.3], [sgn * R_TUBE, sgn * R_TUBE],
                "-", color="0.15", lw=f["lw"] * 0.9, zorder=5)
    ax.axhline(0.0, color="0.75", ls=":", lw=f["lw"] * 0.4, zorder=0)
    ax.text(zmax + 0.25, R_TUBE + 0.035, r"$r = \pm R_t = \pm 0.7\,R_0$",
            ha="right", va="bottom", fontsize=f["annot"] * 0.85, color="0.15")

    axg.set_yscale("log")
    axg.set_ylabel(r"$(R_t - r)/R_0$", fontsize=f["label"] * 0.8,
                   labelpad=f["labelpad"] * 0.6)
    # Short enough to sit inside the panel: a longer caption right-aligned at the
    # frame ran off the left edge and over the y-label.
    axg.text(0.985, 0.08, "plateau = reported film; rear dip = capillary ripple",
             transform=axg.transAxes, ha="right",
             fontsize=f["annot"] * 0.78, color="0.35")
    style_axes(axg, f)
    axg.tick_params(labelsize=f["tick"] * 0.8)

    # orientation: bubble advances towards +z, so the wall moves rearward here
    ax.annotate("", xy=(zmin + 1.55, 0.83), xytext=(zmin + 0.35, 0.83),
                arrowprops={"arrowstyle": "-|>", "color": "0.15",
                            "lw": f["lw"] * 0.45, "shrinkA": 0, "shrinkB": 0})
    ax.text(zmin + 0.35, 0.885, r"bubble motion, $+z$", ha="left", va="bottom",
            fontsize=f["annot"] * 0.85, color="0.15")
    ax.text(zmin, -0.80, r"rear cap", ha="center", va="top",
            fontsize=f["annot"] * 0.85, color="0.2")
    ax.text(zmax - 0.35, -0.80, r"front (advancing) cap", ha="right", va="top",
            fontsize=f["annot"] * 0.85, color="0.2")
    ax.text(zmin + 0.35, -1.0, r"wall moves rearward ($-z$) in this frame",
            ha="left", va="bottom", fontsize=f["annot"] * 0.8, color="0.45")

    ax.set_xlim(zmin - 0.35, zmax + 0.3)
    ax.set_ylim(-1.05, 1.0)
    ax.set_aspect("equal", adjustable="box")
    axg.set_xlabel(r"$z/R_0$", fontsize=f["label"], labelpad=f["labelpad"] * 0.6)
    ax.set_ylabel(r"$r/R_0$", fontsize=f["label"], labelpad=f["labelpad"] * 0.6)
    style_axes(ax, f)
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=f["legend"] * 0.62, handlelength=1.6, labelspacing=0.4)
    fig.tight_layout()
    return save(fig, outdir, "interface_profiles", dpi)


def fig_inertia(receipts: Path, parent_csv: Path, outdir: Path, dpi: int):
    """Relative film change against Re = La Ca_b; three points only."""
    s8 = load_receipt(receipts, "s8_inertia.json")
    corr = sorted(s8["inertial_correction"], key=lambda c: c["reynolds"])

    re = np.array([0.0] + [c["reynolds"] for c in corr])
    dh = np.array([0.0] + [c["h_rel_change_vs_stokes"] for c in corr]) * 1e3
    la = [0.0] + [c["laplace"] for c in corr]
    slope_la1 = corr[0]["h_change_per_unit_reynolds"] * 1e3

    w = 6.5
    f = fonts(w)
    fig, ax = plt.subplots(figsize=(w, w * 0.8))

    xs = np.array([0.0, re.max() * 1.12])
    ax.plot(xs, slope_la1 * xs, "--", color="0.55", lw=f["lw"] * 0.6, zorder=1,
            label=r"receipt slope at $La=1$ ($%.4f$ per unit $Re$), not a fit"
                  % (slope_la1 * 1e-3))
    ax.plot(re, dh, "o", color=C_FEM, ms=f["ms"], mec="k", mew=f["lw"] * 0.25,
            ls="none", zorder=3,
            label=r"pyoomph FEM, $Ca_{\mathrm{in}}=%g$" % s8["ca_in"])

    # Hand-placed to avoid collisions at the three La values the receipt
    # currently carries; anything else falls back to a generic offset rather
    # than raising, so a wider La sweep still plots.
    offsets = {0.0: (14, 2), 1.0: (12, 13), 10.0: (-14, 14)}
    aligns = {0.0: "left", 1.0: "left", 10.0: "right"}
    for x, y, lval in zip(re, dh, la):
        txt = r"$La=0$ (Stokes)" if lval == 0.0 else r"$La=%g$" % lval
        ax.annotate(txt, (x, y), textcoords="offset points",
                    xytext=offsets.get(lval, (12, 10)), ha=aligns.get(lval, "left"),
                    fontsize=f["annot"] * 0.8, color="0.2")
    ax.annotate(r"parent campaign", xy=(re[1], dh[1]),
                xytext=(re[1] + 0.09, dh[1] - 0.55), ha="left", va="center",
                fontsize=f["annot"] * 0.8, color=C_PARENT,
                arrowprops={"arrowstyle": "->", "color": C_PARENT,
                            "lw": f["lw"] * 0.45, "shrinkA": 2, "shrinkB": 6})

    ax.set_xlim(-0.03, re.max() * 1.18)
    ax.set_ylim(dh.min() * 1.20, 0.42)
    ax.set_xlabel(r"$Re = La\,Ca_b$", fontsize=f["label"], labelpad=f["labelpad"])
    ax.set_ylabel(r"$10^{3}\,(h - h_{\mathrm{Stokes}})/h_{\mathrm{Stokes}}$",
                  fontsize=f["label"], labelpad=f["labelpad"])
    style_axes(ax, f)
    ax.legend(loc="lower left", frameon=False, fontsize=f["legend"] * 0.55,
              handlelength=2.0, borderpad=0.2, labelspacing=0.3)
    fig.tight_layout()
    return save(fig, outdir, "inertia", dpi)


FIGURES = {
    "film_vs_cab": fig_film_vs_cab,
    "deviation_vs_cab": fig_deviation_vs_cab,
    "window_independence": fig_window_independence,
    "mesh_convergence": fig_mesh_convergence,
    "interface_profiles": fig_interface_profiles,
    "inertia": fig_inertia,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS,
                    help="directory holding the receipt JSON files")
    ap.add_argument("--parent-csv", type=Path, default=DEFAULT_PARENT_CSV,
                    help="baseline-film.csv from the parent Basilisk campaign")
    ap.add_argument("--outdir", type=Path, required=False,
                    default=REPO_ROOT / "figures",
                    help="directory to write <name>.png and <name>.pdf into")
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=None,
                    help="render only these figures (default: all)")
    ap.add_argument("--dpi", type=int, default=165,
                    help="raster resolution of the PNG output")
    ap.add_argument("--no-usetex", action="store_true",
                    help="fall back to mathtext if no LaTeX install is present")
    ap.add_argument("--list", action="store_true", help="list figure names and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name, fn in sorted(FIGURES.items()):
            print(f"{name:22s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    init_style(usetex=not args.no_usetex)
    names = args.only or list(FIGURES)
    for name in names:
        paths = FIGURES[name](args.receipts, args.parent_csv, args.outdir, args.dpi)
        print(f"{name}: " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
