"""S13: does the rightmost eigenvalue match an independent dynamic measurement?

S12 reports an ``m = 0`` pair ``-0.1232 +/- 0.1578 i`` at ``Ca_in = 0.05`` on a
coarse mesh. S9, on a different mesh, displaced that state and recovered a
residual 39 times larger than ``exp(-0.1232 * 91.44)`` predicts. Different
meshes and S9's ``temporal_error = 1e-4`` floor can explain that without the
eigensolver being wrong - or S12's "rightmost" is not the rightmost.

This stage uses **S12's state and S12's mesh**, marches with output, and fits
the decay envelope and period against the eigenvalue. Two probes:

1. An eigenfunction perturbation of the ``m = 0`` pair. If the fitted growth
   rate and period disagree with the eigensolver, the campaign's numbers are
   solver output, not evidence, and the stage fails.
2. S9's neighbouring-``Ca_in`` displacement on the same mesh, so the 39x gap
   is measured rather than argued. A mismatch here with a matching
   eigenfunction march is the temporal-floor / mixed-mode explanation, not a
   failure of the eigensolver.

The stage can fail. Agreement of Newton residuals is not a pass.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eigen_helpers import (
    advective_rate,
    build_stability_problem,
    film_grid,
    fit_exponential_envelope,
    resample_film,
    shift_sets,
    two_set_rightmost,
)


def snapshot(problem):
    buf = io.BytesIO()
    problem.save_state(buf, quiet=True)
    buf.seek(0)
    return buf


def restore(problem, buf):
    buf.seek(0)
    problem.load_state(buf, quiet=True)


def observables(problem, z_grid, r0, u0, h0):
    z, r = problem.interface_polyline()
    film = resample_film(z, r, z_grid)
    checks = problem.consistency()
    return {
        "time": float(problem.get_current_time(as_float=True)),
        "u": checks.frame_speed,
        "h": checks.h_flat,
        "du": checks.frame_speed - u0,
        "dh": checks.h_flat - h0,
        "film_max": float(np.max(np.abs(film - r0))),
    }


def march(problem, horizon: float, chunks: int, temporal_error: float,
          z_grid, r0, u0, h0, startstep: float) -> list[dict]:
    series = [observables(problem, z_grid, r0, u0, h0)]
    chunk = horizon / chunks
    t0 = float(problem.get_current_time(as_float=True))
    for i in range(chunks):
        problem.run(
            t0 + (i + 1) * chunk,
            startstep=min(startstep, chunk / 20.0),
            outstep=False,
            temporal_error=temporal_error,
            maxstep=chunk,
        )
        series.append(observables(problem, z_grid, r0, u0, h0))
        print(
            f"    t={series[-1]['time']:.4g}  du={series[-1]['du']:+.3e}  "
            f"film={series[-1]['film_max']:.3e}",
            flush=True,
        )
    return series


def fit_series(series: list[dict], key: str) -> dict:
    times = np.array([row["time"] for row in series], dtype=float)
    values = np.array([row[key] for row in series], dtype=float)
    times = times - times[0]
    return fit_exponential_envelope(times, values)


def compare_fit(fit: dict, real: float, imag: float, *, rtol: float) -> dict:
    if not fit.get("ok"):
        return {"agrees": False, "reason": fit.get("reason")}
    sigma_err = abs(fit["sigma"] - real) / max(abs(real), 1e-12)
    omega = fit.get("omega")
    omega_err = None
    if omega is not None and abs(imag) > 1e-12:
        omega_err = abs(omega - abs(imag)) / abs(imag)
    agrees = sigma_err <= rtol and (omega_err is None or omega_err <= rtol)
    return {
        "agrees": agrees,
        "sigma_rel_error": sigma_err,
        "omega_rel_error": omega_err,
        "fit": fit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, default=0.05)
    parser.add_argument("--film-elements", type=float, default=2.0)
    parser.add_argument("--bulk-resolution", type=float, default=0.22)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--periods", type=float, default=4.0)
    parser.add_argument("--chunks", type=int, default=40)
    parser.add_argument("--temporal-error", type=float, default=1e-6)
    parser.add_argument("--s9-temporal-error", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=0.20)
    parser.add_argument("--output-root", type=Path, default=ROOT / "receipts")
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "s13_eigen_dynamic.json")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    problem = build_stability_problem(
        args.ca,
        film_elements=args.film_elements,
        bulk_resolution=args.bulk_resolution,
        output_dir=args.output_root / f"s13_ca{args.ca:g}",
        remesh=False,
    )
    problem.solve_to_steady()
    checks = problem.consistency()
    geom = problem.geometry
    rate = advective_rate(checks.frame_speed, geom.r_tube)
    print(
        f"base: ndof={problem.ndof()} Ca_b={checks.frame_speed:.6f} "
        f"h/Rt={checks.h_flat / geom.r_tube:.6f} rate={rate:.4g}",
        flush=True,
    )

    eigen = two_set_rightmost(problem, 0, n=args.n, rate=rate)
    if eigen["agreed"] is None:
        payload = {
            "check": "s13_eigen_dynamic",
            "passed": False,
            "reason": "two shift sets did not agree on the m=0 rightmost eigenvalue",
            "ca_in": args.ca,
            "ca_b": checks.frame_speed,
            "ndof": problem.ndof(),
            "eigen": {k: eigen[k] for k in eigen if not k.startswith("_")},
        }
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("FAIL two shift sets disagree on the rightmost m=0 eigenvalue", flush=True)
        return 1

    lam = eigen["agreed"]
    print(
        f"m=0 rightmost {lam['real']:+.6g} {lam['imag']:+.6g}i  "
        f"(two shift sets agree)",
        flush=True,
    )
    period = 2.0 * np.pi / abs(lam["imag"]) if abs(lam["imag"]) > 1e-12 else 40.0
    horizon = args.periods * period

    z0, r0_poly = problem.interface_polyline()
    z_grid = film_grid(z0, r0_poly)
    r_eq = resample_film(z0, r0_poly, z_grid)
    u_eq, h_eq = checks.frame_speed, checks.h_flat
    saved = snapshot(problem)

    # --- probe 1: eigenfunction of the agreed rightmost ---
    # Re-solve one shift so get_last_eigenvectors() holds that pair, then
    # perturb the rightmost index. sort=True puts the most negative first.
    sample = eigen["_private_a"][0]
    problem.solve_eigenproblem(args.n, azimuthal_m=0, shift=sample["shift"])
    values = problem.get_last_eigenvalues()
    idx = int(max(range(len(values)), key=lambda i: values[i].real))
    dt = problem.perturb_by_eigenfunction(
        eigenmode=idx, desired_initial_residuals=1e-3)
    startstep = float(dt) if dt is not None else min(1e-3, horizon / 200.0)
    print(f"eigen-perturb march, horizon={horizon:.4g}, dt~{startstep:.3g}", flush=True)
    eigen_series = march(
        problem, horizon, args.chunks, args.temporal_error,
        z_grid, r_eq, u_eq, h_eq, startstep,
    )
    eigen_fit_u = compare_fit(fit_series(eigen_series, "du"), lam["real"], lam["imag"], rtol=args.rtol)
    eigen_fit_film = compare_fit(fit_series(eigen_series, "film_max"), lam["real"], 0.0, rtol=args.rtol)
    # film_max is an envelope already; its slope should match Re(lambda), period is from U.
    print(
        f"  eigen fit: sigma_rel={eigen_fit_u.get('sigma_rel_error')} "
        f"omega_rel={eigen_fit_u.get('omega_rel_error')} agrees={eigen_fit_u['agrees']}",
        flush=True,
    )

    # --- probe 2: S9-style displacement on this same mesh ---
    restore(problem, saved)
    problem.go_to_param(Ca_in=args.ca * 1.05, max_step=0.25 * args.ca)
    zp, rp = problem.interface_polyline()
    imposed = float(np.max(np.abs(resample_film(zp, rp, z_grid) - r_eq)))
    problem.get_global_parameter("Ca_in").value = args.ca
    print(
        f"S9-style imposed film displacement {imposed / geom.r_tube:.3e} R_t; "
        f"marching {horizon:.4g} at temporal_error={args.temporal_error}",
        flush=True,
    )
    s9_series = march(
        problem, horizon, args.chunks, args.temporal_error,
        z_grid, r_eq, u_eq, h_eq, min(1e-3, horizon / 200.0),
    )
    s9_fit = compare_fit(fit_series(s9_series, "du"), lam["real"], lam["imag"], rtol=args.rtol)
    s9_end = s9_series[-1]
    predicted = imposed * np.exp(lam["real"] * horizon)
    measured = s9_end["film_max"]
    ratio = measured / predicted if predicted else float("nan")
    print(
        f"  S9-style: residual {measured:.3e} vs predicted {predicted:.3e} "
        f"(ratio {ratio:.2g}); fit agrees={s9_fit['agrees']}",
        flush=True,
    )

    # Predicted residual after S9's 91.44 time units, for the discrepancy note.
    s9_horizon = 91.43795271542417
    predicted_s9 = np.exp(lam["real"] * s9_horizon)
    measured_s9_fraction = 0.000493669243600488

    passed = bool(eigen["agreed"]) and eigen_fit_u["agrees"]
    if not passed:
        print("FAIL eigenfunction march does not match the eigensolver", flush=True)
    elif not s9_fit["agrees"]:
        print(
            "NOTE S9-style displacement does not match the rightmost pair "
            "(mixed modes or a finite-amplitude leftover); eigensolver still validated",
            flush=True,
        )

    payload = {
        "check": "s13_eigen_dynamic",
        "passed": passed,
        "note": (
            "Same state and mesh as S12. The eigensolver is validated only if "
            "the eigenfunction march's fitted decay and period agree with the "
            "rightmost m=0 eigenvalue. The S9-style probe records the 39x "
            "discrepancy on this mesh; it is not a second pass condition."
        ),
        "ca_in": args.ca,
        "ca_b": checks.frame_speed,
        "h_over_r_tube": checks.h_flat / geom.r_tube,
        "ndof": problem.ndof(),
        "pressure_datum": "integral",
        "remesh": False,
        "film_elements": args.film_elements,
        "bulk_resolution": args.bulk_resolution,
        "advective_rate": rate,
        "shift_sets": {"a": shift_sets(rate)[0], "b": shift_sets(rate)[1]},
        "rightmost": lam,
        "horizon": horizon,
        "period_from_eigenvalue": period,
        "temporal_error": args.temporal_error,
        "eigenfunction_fit_u": eigen_fit_u,
        "eigenfunction_fit_film_envelope": eigen_fit_film,
        "s9_style_fit_u": s9_fit,
        "s9_style_imposed_over_r_tube": imposed / geom.r_tube,
        "s9_style_residual": measured,
        "s9_style_predicted_residual": predicted,
        "s9_style_residual_ratio": ratio,
        "s9_discrepancy_on_original_receipts": {
            "predicted_fraction_after_91.44": float(predicted_s9),
            "s9_measured_recovery_fraction": measured_s9_fraction,
            "ratio": float(measured_s9_fraction / predicted_s9),
        },
        "eigenfunction_series": eigen_series,
        "s9_style_series": s9_series,
        "eigen_two_set": {k: eigen[k] for k in eigen if not k.startswith("_")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
