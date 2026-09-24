"""S14: Campaign S — linear stability labels for the committed 25-state sweep.

Every row of ``receipts/sweep.json`` gets a label in {stable, unstable, untested}
for azimuthal wavenumbers 0, 1 and 2. The receipt is this stage's, not a merge
into the committed sweep.

What this stage does, and refuses to do:

- Re-solves on ``pressure_datum='integral'`` (the committed sweep used fixation).
- Re-solves the eight continuation rows fresh where S10 says that works
  (``Ca_in <= 2``) or continues from the ``Ca_in = 0.7`` seed and **forces a
  remesh at every committed sweep station**. A single ``0.7 → 6`` jump plus
  remesh interpolates from a stretched mesh and Newton residuals go to
  ``inf``. Those eight all have ``ndof = 11131`` in the committed receipt.
  They run at two film resolutions.
- Scales the shift-invert centres on each row's own ``U_b / R_t``. A row is
  accepted only when two different shift sets agree on the rightmost eigenvalue.
- Re-checks the spurious-mode hazard (no near-zero, no unexpected positive) at
  three states, not just S12's one.
- Pad-doubles one high-``Ca`` state.
- Brackets any sign change at two film resolutions. It does not report a
  critical ``Ca_b`` as a point, does not interpret "unstable" as breakup, and
  does not extrapolate a crossing that sits outside the sweep.

``untested`` is a valid label. Silence is not.
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
    remesh_ceiling_for_film,
    eigenvector_support,
    label_from_growth,
    load_sweep,
    rightmost_sample,
    shift_sets,
    two_set_rightmost,
)
from problems.bretherton_bubble import VoidBubbleProblem
from runners.run_case import record_case


def public_eigen(result: dict) -> dict:
    agreed = result["agreed"]
    label, reason = ("untested", "shift_sets_disagree")
    support = None
    if agreed is not None:
        # Support of the winning vector, if we still have a private sample.
        private = list(result.get("_private_a") or []) + list(
            result.get("_private_b") or [])
        winner = None
        if private:
            winner = min(
                private,
                key=lambda s: (
                    abs(s["real"] - agreed["real"]),
                    abs(abs(s["imag"]) - abs(agreed["imag"])),
                ),
            )
        return {
            "azimuthal_k": result["azimuthal_k"],
            "rightmost_real": agreed["real"],
            "rightmost_imag": agreed["imag"],
            "growth_over_rate": None,
            "label": label,
            "label_reason": reason,
            "two_set_agreement": True,
            "spread_a": result["consensus_a"]["spread"] if result["consensus_a"] else None,
            "spread_b": result["consensus_b"]["spread"] if result["consensus_b"] else None,
            "failures": result["failures"],
            "support": support,
            "_winner": winner,
        }
    return {
        "azimuthal_k": result["azimuthal_k"],
        "rightmost_real": None,
        "rightmost_imag": None,
        "growth_over_rate": None,
        "label": "untested",
        "label_reason": "shift_sets_disagree" if not result["failures"] else "eigensolve_failed",
        "two_set_agreement": False,
        "failures": result["failures"],
        "samples_a": result["samples_a"],
        "samples_b": result["samples_b"],
    }


def finalise_mode(entry: dict, rate: float, problem: VoidBubbleProblem) -> dict:
    """Label a mode, screening the winning eigenpair on its residual.

    The screen is ``is_physical``: M has to act on the vector at all, and
    ``||J v - lambda M v|| / ||lambda M v||`` has to be small. The earlier
    screen asked instead where the eigenvector's amplitude sat and rejected
    anything living on pressure or the global multipliers. For this pencil M is
    nonzero only on the kinematic-condition rows against the interface
    mesh-position columns, so every genuine mode is pressure-dominated and that
    screen rejected the physics; see verification/README.md.

    A winner with no residual verdict at all (``physical is None``: the pencil
    was not captured, or its size did not match the vector) is ``untested``.
    Growth alone is the unscreened acceptance this screen exists to remove.
    """
    winner = entry.pop("_winner", None)
    if entry["rightmost_real"] is None:
        return entry
    label, reason = label_from_growth(entry["rightmost_real"], rate)
    physical = winner.get("physical") if winner is not None else None
    if winner is not None:
        screen = winner.get("residual_screen")
        entry["residual_screen"] = screen
        entry["physical"] = physical
        # Support stays in the receipt as a diagnostic, never as a verdict.
        # The live pencil's description matches the vector's numbering.
        entry["support"] = eigenvector_support(
            problem, winner["vector"],
            dof_description=winner.get("dof_description"))
    if physical is False:
        label, reason = "untested", "eigenpair_residual_too_large"
    elif physical is None:
        label, reason = "untested", "residual_screen_unavailable"
    entry["label"] = label
    entry["label_reason"] = reason
    entry["growth_over_rate"] = entry["rightmost_real"] / max(rate, 1e-12)
    entry["two_set_agreement"] = True
    return entry


def solve_fresh(ca_in: float, *, film_elements: float, bulk_resolution: float,
                output_root: Path, pad_rear: float, pad_front: float,
                remesh: bool) -> tuple[VoidBubbleProblem, dict]:
    outdir = output_root / f"s14_ca{ca_in:g}_n{film_elements:g}_pad{pad_rear:g}x{pad_front:g}"
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
    outdir = output_root / f"s14_ca{ca_in:g}_n{film_elements:g}_pad{pad_rear:g}x{pad_front:g}"
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


def upsert_row(payload: dict, spec: dict) -> None:
    key = (float(spec["ca_in"]), spec.get("role"), spec.get("film_elements"))
    for index, row in enumerate(payload["rows"]):
        if (float(row["ca_in"]), row.get("role"), row.get("film_elements")) == key:
            payload["rows"][index] = spec
            return
    payload["rows"].append(spec)


def spectrum_row(problem: VoidBubbleProblem, *, n: int, modes: list[int]) -> dict:
    checks = problem.consistency()
    rate = advective_rate(checks.frame_speed, problem.geometry.r_tube)
    modes_out = []
    for k in modes:
        raw = two_set_rightmost(problem, k, n=n, rate=rate)
        entry = public_eigen(raw)
        modes_out.append(finalise_mode(entry, rate, problem))
        print(
            f"    k={k}: {modes_out[-1]['label']:9s}  "
            f"Re={modes_out[-1]['rightmost_real']}  "
            f"{modes_out[-1]['label_reason']}",
            flush=True,
        )
    labels = {f"k{k}": next(m["label"] for m in modes_out if m["azimuthal_k"] == k)
              for k in modes}
    return {
        "ca_in": problem.current_ca_in(),
        "ca_b": checks.frame_speed,
        "h_over_r_tube": checks.h_flat / problem.geometry.r_tube,
        "ndof": problem.ndof(),
        "pressure_datum": problem.pressure_datum,
        "remesh_flag": problem.remesh,
        "film_elements": problem.film_elements,
        "bulk_resolution": problem.bulk_resolution,
        "pad_rear": problem.geometry.pad_rear,
        "pad_front": problem.geometry.pad_front,
        "advective_rate": rate,
        "shift_sets": {"a": shift_sets(rate)[0], "b": shift_sets(rate)[1]},
        "modes": modes_out,
        "labels": labels,
        "check_void_volume_error": checks.volume_error,
        "check_stagnant_film_error": checks.stagnant_film_error,
    }


def against_receipt(row: dict, ref: dict | None) -> dict:
    if ref is None:
        return {"compared": False}
    return {
        "compared": True,
        "rel_diff_ca_b": abs(row["ca_b"] / ref["ca_b"] - 1.0),
        "rel_diff_h": abs(row["h_over_r_tube"] / ref["h_flat_over_r_tube"] - 1.0),
        "ref_ca_b": ref["ca_b"],
        "ref_h_over_r_tube": ref["h_flat_over_r_tube"],
        "ref_ndof": ref["ndof"],
        "ref_route": ref["route"],
    }


def spurious_check(problem: VoidBubbleProblem, modes: list[int], n: int = 20) -> dict:
    checks = problem.consistency()
    rate = advective_rate(checks.frame_speed, problem.geometry.r_tube)
    out = []
    for k in modes:
        sample = rightmost_sample(problem, k, n, 0.0)
        out.append({
            "azimuthal_k": k,
            "n": n,
            "rightmost_real": sample["real"],
            "rightmost_imag": sample["imag"],
            "positive_count": sample["positive_count"],
            "near_zero_count": sample["near_zero_count"],
            "physical": sample["physical"],
            "residual_screen": sample["residual_screen"],
            "spectrum": sample["spectrum"],
        })
    return {
        "ca_in": problem.current_ca_in(),
        "ca_b": checks.frame_speed,
        "ndof": problem.ndof(),
        "rate": rate,
        "modes": out,
        "hazard_absent": all(m["positive_count"] == 0 and m["near_zero_count"] == 0 for m in out),
    }


def calibrate_count(problem: VoidBubbleProblem, modes: list[int],
                    counts: list[int], rate: float) -> dict:
    rows = []
    required = None
    for k in modes:
        by_n = {}
        for n in counts:
            raw = two_set_rightmost(problem, k, n=n, rate=rate)
            by_n[n] = {
                "agreed": raw["agreed"],
                "failures": raw["failures"],
            }
        agreed_ns = [n for n, v in by_n.items() if v["agreed"] is not None]
        min_n = min(agreed_ns) if agreed_ns else None
        rows.append({"azimuthal_k": k, "min_reliable_count": min_n, "by_n": {
            str(n): ({"real": v["agreed"]["real"], "imag": v["agreed"]["imag"]} if v["agreed"] else None)
            for n, v in by_n.items()
        }})
        if min_n is not None:
            required = min_n if required is None else max(required, min_n)
    return {"required_eigenvalue_count": required, "modes": rows}


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def crossings(rows: list[dict], modes: list[int]) -> list[dict]:
    """Sign-change brackets on the production resolution, plus a fine pair if present."""
    found = []
    primary = [r for r in rows if r.get("role") == "production" and "ca_b" in r]
    primary.sort(key=lambda r: r["ca_b"])
    for k in modes:
        key = f"k{k}"
        for left, right in zip(primary, primary[1:]):
            a, b = left["labels"][key], right["labels"][key]
            if {a, b} == {"stable", "unstable"}:
                fine = [
                    r for r in rows
                    if r.get("role") == "fine" and "ca_b" in r
                    and r["ca_in"] in {left["ca_in"], right["ca_in"]}
                ]
                found.append({
                    "azimuthal_k": k,
                    "bracket_ca_b": [left["ca_b"], right["ca_b"]],
                    "bracket_ca_in": [left["ca_in"], right["ca_in"]],
                    "labels": [a, b],
                    "fine_resolutions": [
                        {"ca_in": r["ca_in"], "ca_b": r["ca_b"],
                         "film_elements": r["film_elements"], "labels": r["labels"]}
                        for r in fine
                    ],
                    "bracketed_at_two_resolutions": (
                        len(fine) == 2
                        and {r["labels"][key] for r in fine}
                        == {"stable", "unstable"}
                    ),
                })
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, nargs="+", default=list(reversed(SWEEP_CA)))
    parser.add_argument("--modes", type=int, nargs="+", default=[0, 1, 2])
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
                        default=ROOT / "receipts" / "s14_azimuthal_sweep.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    refs = load_sweep(args.sweep)
    payload = {
        "check": "s14_azimuthal_sweep",
        "passed": False,
        "note": (
            "Linear stability labels for the committed 25-state sweep. "
            "azimuthal_k is the wavenumber; do not call it m in the report. "
            "pressure_datum is integral throughout. Continuation rows are "
            "re-solved fresh up to Ca_in=2 or remeshed. A crossing is a "
            "bracket at two film resolutions, never a point value."
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

    # High Ca first: that is where a crossing is expected and ndof is smallest.
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
                        print(f"  k=3 spot: {payload['k3_spot_top_row'].get('label')}", flush=True)
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
                # Fresh starts fail by Ca_in = 4. Seed at 0.7 on the doubled
                # window and continue, same as the production high-Ca route.
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

    # Doubled pads remesh only through Ca_in=2.5; remesh at 3 interpolates
    # a bulk node by nearest-neighbour and Newton residuals go to inf.
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
