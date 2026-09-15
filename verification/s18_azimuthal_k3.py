"""S18: Campaign S remainder — azimuthal_k=3 on the committed 25-state sweep.

S14 labelled k=0,1,2 on every production row and k=3 only at the top.
This stage walks the same states at k=3, including the high-Ca fine
rows, the three-state spurious-mode check, and one high-Ca pad-double.

A crossing is a bracket at two film resolutions. This stage does not
report a critical Ca_b as a point, does not interpret "unstable" as
breakup, and does not extrapolate a crossing that sits outside the sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eigen_helpers import (
    CONTINUATION_SEED,
    FRESH_CA_LIMIT,
    SWEEP_CA,
    advective_rate,
    build_stability_problem,
    climb_with_forced_remesh,
    load_sweep,
    remesh_ceiling_for_film,
    two_set_rightmost,
)
from problems.bretherton_bubble import VoidBubbleProblem
from runners.run_case import record_case
from s14_azimuthal_sweep import (
    against_receipt,
    calibrate_count,
    crossings,
    finalise_mode,
    public_eigen,
    spectrum_row,
    spurious_check,
    upsert_row,
    write_payload,
)


def solve_fresh(ca_in: float, *, film_elements: float, bulk_resolution: float,
                output_root: Path, pad_rear: float, pad_front: float,
                remesh: bool) -> tuple[VoidBubbleProblem, dict]:
    outdir = output_root / (
        f"s18_ca{ca_in:g}_n{film_elements:g}_pad{pad_rear:g}x{pad_front:g}")
    problem = build_stability_problem(
        ca_in, film_elements=film_elements, bulk_resolution=bulk_resolution,
        output_dir=outdir, remesh=remesh, pad_rear=pad_rear, pad_front=pad_front,
    )
    route = problem.solve_to_steady()
    row = record_case(problem, output_dir=outdir, route=route, wall=0.0, audit={})
    return problem, row


def continue_and_remesh(problem: VoidBubbleProblem, ca_in: float,
                        output_root: Path, film_elements: float,
                        pad_rear: float, pad_front: float,
                        remesh_ceiling: float | None = None) -> tuple[VoidBubbleProblem, dict]:
    outdir = output_root / (
        f"s18_ca{ca_in:g}_n{film_elements:g}_pad{pad_rear:g}x{pad_front:g}")
    outdir.mkdir(parents=True, exist_ok=True)
    problem.set_output_directory(str(outdir))
    ndof_before = problem.ndof()
    start = problem.current_ca_in()
    ceiling = remesh_ceiling if remesh_ceiling is not None else remesh_ceiling_for_film(
        film_elements, pad_rear)
    climb = climb_with_forced_remesh(problem, ca_in, remesh_ceiling=ceiling)
    remeshed = [step for step in climb if step <= ceiling + 1e-12]
    print(
        f"    climbed {start:g} -> {ca_in:g} remesh_at={remeshed} ceiling={ceiling:g}",
        flush=True,
    )
    row = record_case(problem, output_dir=outdir, route="continuation+forced_remesh",
                      wall=0.0, audit={})
    row["ndof_before_remesh"] = ndof_before
    row["forced_remesh"] = bool(remeshed) and abs(ca_in - remeshed[-1]) < 1e-12
    row["climb"] = climb
    row["remesh_stations"] = remeshed
    row["remesh_ceiling"] = ceiling
    return problem, row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, nargs="+", default=list(reversed(SWEEP_CA)))
    parser.add_argument("--modes", type=int, nargs="+", default=[3])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--film-elements", type=float, default=3.0)
    parser.add_argument("--film-elements-fine", type=float, default=6.0)
    parser.add_argument("--bulk-resolution", type=float, default=0.10)
    parser.add_argument("--pad-rear", type=float, default=1.5)
    parser.add_argument("--pad-front", type=float, default=3.0)
    parser.add_argument("--calibrate-ca", type=float, nargs="+",
                        default=[0.05, 0.5, 6.0])
    parser.add_argument("--spurious-ca", type=float, nargs="+",
                        default=[0.05, 0.5, 6.0])
    parser.add_argument("--pad-double-ca", type=float, default=6.0)
    parser.add_argument("--sweep", type=Path, default=ROOT / "receipts" / "sweep.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "receipts")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "receipts" / "s18_azimuthal_k3.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    refs = load_sweep(args.sweep)
    payload = {
        "check": "s18_azimuthal_k3",
        "passed": False,
        "note": (
            "azimuthal_k=3 labels for the committed 25-state sweep. "
            "S14 already labelled k=0,1,2 and spotted k=3 only at the top. "
            "A crossing is a bracket at two film resolutions, never a point "
            "value. pressure_datum is integral throughout."
        ),
        "pressure_datum": "integral",
        "modes_requested": args.modes,
        "n_eigenvalues": args.n,
        "rows": [],
        "calibrations": [],
        "spurious_checks": [],
        "pad_doubling": None,
        "crossings": [],
    }
    if args.resume and args.out.is_file():
        payload = json.loads(args.out.read_text(encoding="utf-8"))

    done = {(r["ca_in"], r.get("role"), r.get("film_elements"))
            for r in payload["rows"] if r.get("modes")}

    production_states: dict[float, VoidBubbleProblem] = {}
    continuation_problem = None
    continuation_fine = None

    def want(ca, role, film):
        return (ca, role, film) not in done

    for ca in args.ca:
        roles = [("production", args.film_elements)]
        if ca >= 1.0 - 1e-12:
            roles.append(("fine", args.film_elements_fine))
        for role, film in roles:
            if not want(ca, role, film):
                print(f"skip Ca_in={ca:g} {role} n={film} (resume)", flush=True)
                continue
            print(f"== Ca_in={ca:g} role={role} film_elements={film} ==", flush=True)
            try:
                if ca <= FRESH_CA_LIMIT + 1e-12:
                    problem, solved = solve_fresh(
                        ca, film_elements=film, bulk_resolution=args.bulk_resolution,
                        output_root=args.output_root, pad_rear=args.pad_rear,
                        pad_front=args.pad_front, remesh=True,
                    )
                else:
                    holder = continuation_problem if role == "production" else continuation_fine
                    if holder is None:
                        seed, _ = solve_fresh(
                            CONTINUATION_SEED, film_elements=film,
                            bulk_resolution=args.bulk_resolution,
                            output_root=args.output_root, pad_rear=args.pad_rear,
                            pad_front=args.pad_front, remesh=True,
                        )
                        holder = seed
                    problem, solved = continue_and_remesh(
                        holder, ca, args.output_root, film, args.pad_rear, args.pad_front)
                    if role == "production":
                        continuation_problem = problem
                    else:
                        continuation_fine = problem
                if role == "production":
                    production_states[ca] = problem
                    if abs(ca - args.pad_double_ca) < 1e-12:
                        rate = advective_rate(problem.frame_speed(), problem.geometry.r_tube)
                        spot = two_set_rightmost(problem, 3, n=args.n, rate=rate)
                        payload["k3_spot_top_row"] = public_eigen(spot)
                        if payload["k3_spot_top_row"].get("_winner"):
                            payload["k3_spot_top_row"] = finalise_mode(
                                payload["k3_spot_top_row"], rate, problem)
                spec = spectrum_row(problem, n=args.n, modes=args.modes)
                spec["role"] = role
                spec["route"] = solved.get("route")
                spec["forced_remesh"] = solved.get("forced_remesh")
                spec["ndof_before_remesh"] = solved.get("ndof_before_remesh")
                spec["receipt_comparison"] = against_receipt(spec, refs.get(ca))
                upsert_row(payload, spec)
                print(
                    f"  Ca_b={spec['ca_b']:.6g} ndof={spec['ndof']} "
                    f"labels={spec['labels']} vs receipt "
                    f"dCa_b={spec['receipt_comparison'].get('rel_diff_ca_b')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - a failed row is untested
                upsert_row(payload, {
                    "ca_in": ca,
                    "role": role,
                    "film_elements": film,
                    "labels": {f"k{k}": "untested" for k in args.modes},
                    "label_reason": f"{type(exc).__name__}: {exc}",
                    "pressure_datum": "integral",
                    "remesh_flag": True,
                })
                print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)
            write_payload(args.out, payload)

        if (ca in args.calibrate_ca and ca in production_states
                and not any(abs(c["ca_in"] - ca) < 1e-12 for c in payload["calibrations"])):
            problem = production_states[ca]
            rate = advective_rate(problem.frame_speed(), problem.geometry.r_tube)
            print(f"-- count calibration at Ca_in={ca:g} --", flush=True)
            payload["calibrations"].append({
                "ca_in": ca,
                "ca_b": problem.frame_speed(),
                "ndof": problem.ndof(),
                **calibrate_count(problem, args.modes, [8, 16], rate),
            })
            write_payload(args.out, payload)

        if (ca in args.spurious_ca and ca in production_states
                and not any(abs(c["ca_in"] - ca) < 1e-12 for c in payload["spurious_checks"])):
            print(f"-- spurious-mode check at Ca_in={ca:g} --", flush=True)
            payload["spurious_checks"].append(
                spurious_check(production_states[ca], args.modes, n=20))
            write_payload(args.out, payload)

        pad_done = payload.get("pad_doubling") and not payload["pad_doubling"].get("failed")
        if abs(ca - args.pad_double_ca) < 1e-12 and ca in production_states and not pad_done:
            print(f"-- pad doubling at Ca_in={ca:g} --", flush=True)
            try:
                doubled, _ = solve_fresh(
                    min(ca, CONTINUATION_SEED), film_elements=args.film_elements,
                    bulk_resolution=args.bulk_resolution,
                    output_root=args.output_root,
                    pad_rear=2.0 * args.pad_rear, pad_front=2.0 * args.pad_front,
                    remesh=True,
                )
                if ca > FRESH_CA_LIMIT + 1e-12:
                    doubled, _ = continue_and_remesh(
                        doubled, ca, args.output_root, args.film_elements,
                        2.0 * args.pad_rear, 2.0 * args.pad_front)
                spec = spectrum_row(doubled, n=args.n, modes=args.modes)
                spec["role"] = "pad_doubled"
                base = next(r for r in payload["rows"]
                            if r["ca_in"] == ca and r.get("role") == "production")
                payload["pad_doubling"] = {
                    "ca_in": ca,
                    "base": {"ca_b": base["ca_b"], "h_over_r_tube": base["h_over_r_tube"],
                             "labels": base["labels"], "ndof": base["ndof"]},
                    "doubled": {"ca_b": spec["ca_b"], "h_over_r_tube": spec["h_over_r_tube"],
                                "labels": spec["labels"], "ndof": spec["ndof"],
                                "modes": spec["modes"]},
                    "rel_diff_h": abs(spec["h_over_r_tube"] / base["h_over_r_tube"] - 1.0),
                    "labels_agree": spec["labels"] == base["labels"],
                }
            except Exception as exc:  # noqa: BLE001
                payload["pad_doubling"] = {"ca_in": ca, "failed": f"{type(exc).__name__}: {exc}"}
            write_payload(args.out, payload)

    if payload.get("pad_doubling") is None or payload["pad_doubling"].get("failed"):
        for ca, ceiling in ((args.pad_double_ca, 2.5), (5.0, 2.5), (3.0, 3.0)):
            base = next((r for r in payload["rows"]
                         if abs(r["ca_in"] - ca) < 1e-12
                         and r.get("role") == "production" and r.get("modes")), None)
            if base is None:
                continue
            print(f"-- pad doubling retry Ca_in={ca:g} remesh_ceiling={ceiling} --",
                  flush=True)
            try:
                doubled, _ = solve_fresh(
                    min(ca, CONTINUATION_SEED), film_elements=args.film_elements,
                    bulk_resolution=args.bulk_resolution,
                    output_root=args.output_root,
                    pad_rear=2.0 * args.pad_rear, pad_front=2.0 * args.pad_front,
                    remesh=True,
                )
                if ca > FRESH_CA_LIMIT + 1e-12:
                    doubled, _ = continue_and_remesh(
                        doubled, ca, args.output_root, args.film_elements,
                        2.0 * args.pad_rear, 2.0 * args.pad_front,
                        remesh_ceiling=ceiling)
                spec = spectrum_row(doubled, n=args.n, modes=args.modes)
                payload["pad_doubling"] = {
                    "ca_in": ca,
                    "remesh_ceiling": ceiling,
                    "base": {"ca_b": base["ca_b"], "h_over_r_tube": base["h_over_r_tube"],
                             "labels": base["labels"], "ndof": base["ndof"]},
                    "doubled": {"ca_b": spec["ca_b"], "h_over_r_tube": spec["h_over_r_tube"],
                                "labels": spec["labels"], "ndof": spec["ndof"],
                                "modes": spec["modes"]},
                    "rel_diff_h": abs(spec["h_over_r_tube"] / base["h_over_r_tube"] - 1.0),
                    "labels_agree": spec["labels"] == base["labels"],
                }
                write_payload(args.out, payload)
                break
            except Exception as exc:  # noqa: BLE001
                payload["pad_doubling"] = {
                    "ca_in": ca, "remesh_ceiling": ceiling,
                    "failed": f"{type(exc).__name__}: {exc}",
                }
                write_payload(args.out, payload)

    payload["crossings"] = crossings(payload["rows"], args.modes)

    requested = set(args.ca)
    have = {r["ca_in"] for r in payload["rows"] if r.get("role") == "production"}
    labelled = all(
        r.get("labels") and all(r["labels"].get(f"k{k}") in {"stable", "unstable", "untested"}
                                for k in args.modes)
        for r in payload["rows"] if r.get("role") == "production"
    )
    every_row = requested <= have
    bad_crossing = any(
        c["bracketed_at_two_resolutions"] is False for c in payload["crossings"]
    )
    accepted_without_agreement = any(
        m.get("label") in {"stable", "unstable"} and not m.get("two_set_agreement", True)
        for r in payload["rows"] for m in r.get("modes", [])
    )
    pad_ok = bool(payload.get("pad_doubling") and not payload["pad_doubling"].get("failed"))
    payload["passed"] = bool(
        every_row and labelled and not bad_crossing
        and not accepted_without_agreement and pad_ok
    )
    if not every_row:
        print(f"FAIL missing production rows: {sorted(requested - have)}", flush=True)
    if bad_crossing:
        print("FAIL a crossing is not bracketed at two film resolutions", flush=True)
    if accepted_without_agreement:
        print("FAIL a stable/unstable label was accepted without two-set agreement", flush=True)
    if not pad_ok:
        print("FAIL no successful high-Ca pad-doubling check", flush=True)
    write_payload(args.out, payload)
    print(f"wrote {args.out}  passed={payload['passed']}", flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
