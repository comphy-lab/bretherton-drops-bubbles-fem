"""Run or collect the portable Herrada--Yu--Stone figure 2 cases.

Configuration and coordinate helpers intentionally import no pyoomph. The
solver is imported only after a case has passed the pure contract checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOI = "10.1017/jfm.2023.123"
VHAT = 3.3
RT = VHAT ** (-1.0 / 3.0)


def read_case(path: Path) -> dict:
    case = json.loads(path.read_text(encoding="utf-8"))
    validate_case(case)
    return case


def validate_case(case: dict) -> None:
    """Reject a case that does not represent the paper's figure 2 mapping."""
    required = {
        "schema", "case_id", "geometry", "nondimensionalization", "physics",
        "tolerances", "mesh", "output", "completion", "boundaries", "closure", "solve",
    }
    missing = sorted(required - case.keys())
    if missing:
        raise ValueError(f"case is missing required fields: {', '.join(missing)}")
    if case["schema"] != "pyoomph-case-v1":
        raise ValueError("figure 2 cases must use schema pyoomph-case-v1")
    if case["geometry"] != {
        "kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "co-moving"
    }:
        raise ValueError("geometry must be axisymmetric (r,z) in the co-moving frame")
    p = case["physics"]
    expected = {
        "bond": 0.0, "reynolds": 0.0, "laplace": 0.0, "vhat": VHAT,
        "r0": 1.0, "r_tube": RT,
    }
    for key, value in expected.items():
        if not math.isclose(float(p.get(key, math.nan)), value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{key} must be {value!r} for the figure 2 mapping")
    cal = float(p.get("cal_paper", math.nan))
    if cal not in (-0.001, -0.02):
        raise ValueError("cal_paper must be -0.001 or -0.02")
    if not math.isclose(float(p.get("ca_in", math.nan)), abs(cal), rel_tol=0, abs_tol=1e-15):
        raise ValueError("ca_in must equal abs(cal_paper) under the Bo=0 axial reflection")
    if p.get("interior") != "void":
        raise ValueError("the available solver requires the documented void approximation")
    if not math.isclose(float(p.get("z_rear", math.nan)), 5 * RT, rel_tol=1e-12):
        raise ValueError("z_rear must map the paper's upper boundary to 5 R_t")
    if not math.isclose(float(p.get("l_domain", math.nan)), 15 * RT, rel_tol=1e-12):
        raise ValueError("l_domain must equal the paper's H/R=15")
    if case["boundaries"].get("outlet") != "poiseuille":
        raise ValueError("primary figure 2 cases require the audited Poiseuille outlet")
    if case["closure"].get("frame_speed") != "tip":
        raise ValueError("the figure 2 coordinate anchor requires the rear-tip closure")
    if case["solve"].get("mode") != "steady":
        raise ValueError("an accepted figure 2 result requires stationary polish")
    if case["completion"] != {"kind": "steady", "value": "stationary_polish"}:
        raise ValueError("completion must require stationary_polish")


def paper_coordinates(z: list[float], r: list[float], r_tube: float) -> tuple[list[float], list[float]]:
    if len(z) != len(r) or not z:
        raise ValueError("z and r must be non-empty arrays of equal length")
    z0 = min(z)
    return ([(z0 - value) / r_tube for value in z], [1.0 - value / r_tube for value in r])


def bretherton(ca_b: float) -> float:
    return 0.643 * (3.0 * ca_b) ** (2.0 / 3.0)


def aussillous_quere(ca_b: float) -> float:
    x = ca_b ** (2.0 / 3.0)
    return 1.34 * x / (1.0 + 3.35 * x)


def acceptance_errors(result: dict, gates: dict) -> list[str]:
    errors: list[str] = []
    if result.get("route") not in ("transient+stationary", "stationary"):
        errors.append("solver did not finish with a stationary polish")
    finite = ("ca_b", "h_flat_over_r_tube", "check_void_volume_error",
              "check_flux_error", "check_film_flatness")
    missing = False
    for name in finite:
        try:
            if not math.isfinite(float(result[name])):
                errors.append(f"{name} is not finite")
                missing = True
        except (KeyError, TypeError, ValueError):
            errors.append(f"{name} is missing")
            missing = True
    if missing:
        return errors
    if float(result["ca_b"]) <= 0 or not 0 < float(result["h_flat_over_r_tube"]) < 1:
        errors.append("solved speed and film thickness must be physical positive fractions")
    for field, gate in (
        ("check_void_volume_error", "max_volume_error"),
        ("check_flux_error", "max_flux_error"),
        ("check_film_flatness", "max_film_flatness"),
    ):
        if abs(float(result[field])) > float(gates[gate]):
            errors.append(f"{field} exceeds {gate}")
    if int(result.get("check_central_nodes", 0)) < int(gates["min_central_nodes"]):
        errors.append("too few central interface nodes for a film measurement")
    return errors


def append_progress(path: Path, stage: str, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time_unix": time.time(), "stage": stage, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_interface(path: Path, z: list[float], r: list[float], r_tube: float) -> None:
    zhat, hhat = paper_coordinates(z, r, r_tube)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("z_solver", "r_solver", "zhat_paper", "hhat_paper"))
        writer.writerows(zip(z, r, zhat, hhat, strict=True))


def plot_profiles(rows: list[tuple[dict, Path]], output_dir: Path, name: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}",
        "mathtext.fontset": "cm", "axes.linewidth": 1.4,
    })
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colours = {0.001: "#1f4e9c", 0.02: "#b24736"}
    for summary, interface in rows:
        with interface.open(newline="", encoding="utf-8") as stream:
            data = list(csv.DictReader(stream))
        zhat = [float(row["zhat_paper"]) for row in data]
        hhat = [float(row["hhat_paper"]) for row in data]
        ca = abs(float(summary["cal_paper"]))
        label = rf"$Ca_l={-ca:g}$ ({summary['resolution']})"
        ax.plot(zhat, hhat, lw=2.0 if summary["resolution"] == "baseline" else 1.2,
                ls="-" if summary["resolution"] == "baseline" else "--",
                color=colours[ca], label=label)
        if summary["resolution"] == "baseline":
            ca_b = float(summary["ca_b"])
            ax.axhline(bretherton(ca_b), color=colours[ca], lw=1.2, ls=":",
                       label=rf"Bretherton at $|Ca_b|={ca_b:.4g}$")
            if ca == 0.02:
                ax.axhline(aussillous_quere(ca_b), color=colours[ca], lw=1.2, ls="-.",
                           label=rf"Aussillous--Qu\'er\'e at $|Ca_b|={ca_b:.4g}$")
    ax.set_xlabel(r"$\hat z$")
    ax.set_ylabel(r"$\hat h_b$")
    ax.set_xlim(-5.0, 0.0)
    ax.set_ylim(0.0, 0.15)
    ax.tick_params(which="both", direction="out", width=1.4, labelsize=11)
    ax.minorticks_on()
    ax.legend(frameon=False, fontsize=8, ncol=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for suffix, dpi in (("pdf", 300), ("png", 300)):
        target = output_dir / f"{name}.{suffix}"
        fig.savefig(target, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        made.append(str(target))
    plt.close(fig)
    return made


def run_case(case_path: Path, output_dir: Path) -> int:
    # Production attempts pre-create an empty runtime directory. Refuse any
    # populated destination, and claim the empty directory exactly once.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".started").open("x", encoding="utf-8") as stream:
        stream.write("single-case attempt\n")
    progress = output_dir / "progress.jsonl"
    case: dict = {}
    try:
        case = read_case(case_path)
        append_progress(progress, "initialising", case_id=case["case_id"])
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from runners.run_case import build, record_case

        problem = build({**case, "schema": "pyoomph-case-v2"})
        problem.set_linear_solver("superlu")
        problem.set_c_compiler("system")
        problem.set_output_directory(str(output_dir / "solver"))
        problem.initialise()
        audit = problem.audit_constraints()
        append_progress(progress, "initialised", constraint_audit=audit)
        solve = case["solve"]
        started = time.time()
        route = problem.solve_to_steady(
            relax_time=float(solve["relax_time"]), start_step=float(solve["start_step"]),
            temporal_error=float(solve["temporal_error"]),
        )
        append_progress(progress, "stationary_polish_complete", route=route)
        result = record_case(problem, output_dir=output_dir, route=route,
                             wall=time.time() - started, audit=audit)
        z_raw, r_raw = problem.interface_polyline()
        write_interface(output_dir / "interface.csv", list(z_raw), list(r_raw), RT)
        problem.save_state(str(output_dir / "state.dump"), quiet=True)
        errors = acceptance_errors(result, case["acceptance"])
        if errors:
            raise RuntimeError("; ".join(errors))
        summary = {
            **result, "case_id": case["case_id"], "cal_paper": case["physics"]["cal_paper"],
            "vhat": VHAT, "bond": 0.0, "reynolds": 0.0,
            "resolution": case["reproduction"]["resolution"],
            "interface_profile": "interface.csv",
            "source": {"doi": DOI, "figure": 2},
            "model": {"interior": "void", "outlet": "poiseuille",
                      "paper_outlet": "static_pressure", "axial_mapping": "reflection"},
            "accepted": True,
        }
        write_json(output_dir / "summary.json", summary)
        append_progress(progress, "outputs_written", files=["summary.json", "interface.csv", "state.dump"])
        plots = plot_profiles([(summary, output_dir / "interface.csv")], output_dir, "profile")
        append_progress(progress, "complete", plots=plots)
        write_json(output_dir / "status.json", {"status": "complete", "case_id": case["case_id"]})
        return 0
    except Exception as exc:  # noqa: BLE001 - status receipt must cover solver failures
        write_json(output_dir / "status.json", {
            "status": "failed", "case_id": case.get("case_id"),
            "error": f"{type(exc).__name__}: {exc}",
        })
        append_progress(progress, "failed", error=f"{type(exc).__name__}: {exc}")
        print(f"failed: {exc}", file=sys.stderr)
        return 1


def collect(results: list[Path], output_dir: Path) -> int:
    rows: list[tuple[dict, Path]] = []
    for result_dir in results:
        summary_path = result_dir / "summary.json"
        interface_path = result_dir / "interface.csv"
        if not summary_path.is_file() or not interface_path.is_file():
            raise ValueError(f"{result_dir} is not an accepted result directory")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("accepted") is not True or summary.get("source", {}).get("doi") != DOI:
            raise ValueError(f"{result_dir} is not an accepted figure 2 result")
        rows.append((summary, interface_path))
    if not rows:
        raise ValueError("--results requires at least one accepted result directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    plots = plot_profiles(rows, output_dir, "herradaEtAl_figure2")
    baseline = {abs(float(s["cal_paper"])): s for s, _ in rows if s["resolution"] == "baseline"}
    observations: dict[str, dict] = {}
    if 0.001 in baseline:
        s = baseline[0.001]
        ratio = float(s["h_flat_over_r_tube"]) / bretherton(float(s["ca_b"]))
        observations["small_ca_near_bretherton"] = {"ratio": ratio, "passes": abs(ratio - 1) <= 0.15}
    if 0.02 in baseline:
        s = baseline[0.02]
        h = float(s["h_flat_over_r_tube"])
        ca_b = float(s["ca_b"])
        db = abs(h - bretherton(ca_b))
        daq = abs(h - aussillous_quere(ca_b))
        observations["higher_ca_below_bretherton_nearer_aq"] = {
            "below_bretherton": h < bretherton(ca_b), "distance_bretherton": db,
            "distance_aussillous_quere": daq, "passes": h < bretherton(ca_b) and daq < db,
        }
    write_json(output_dir / "collection.json", {
        "source": {"doi": DOI, "figure": 2}, "cases": [s["case_id"] for s, _ in rows],
        "plots": plots, "observations": observations,
    })
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--case", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    gather = commands.add_parser("collect")
    gather.add_argument("--results", required=True, nargs="+", type=Path)
    gather.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_case(args.case, args.output_dir)
    return collect(args.results, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
