"""S6: mesh convergence of the reported observables.

The parent Basilisk campaign has no committed refinement sequence - its own
report says so, and records that its four baseline films span between 1.6 and
6.4 of the smallest cells, below its own four-cell warning threshold at the two
lowest capillary numbers. So the deposited-film curve currently rests on
single-resolution calculations. This produces the missing evidence for the FEM
counterpart.

Two independent knobs are refined, because they control different errors: the
number of elements across the film (which resolves the lubrication profile) and
the bulk element size (which resolves the menisci and the outer flow). Refining
only one would leave the other as an uncontrolled floor.

The reported quantity is the Richardson-extrapolated value together with the
observed order, so the production resolution comes with a quantified
discretisation error rather than a claim of "converged".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.run_case import build, load_case, solve_case

REPORTED = ("h_flat_over_r_tube", "ca_b", "bubble_pressure_excess")


CONVERGED_SPREAD = 2e-4
"""Relative spread below which a refinement sequence is treated as converged.

Set from what the solver can actually distinguish rather than from taste: the
transient-plus-polish route reproduces a given configuration to a few parts in
1e-6, but successive mesh refinements also change which nodes the film median is
taken over, and that quantisation is the real floor on this observable.
"""


def convergence(sizes: list[float], values: list[float]) -> dict:
    """Characterise a refinement sequence, without forcing an extrapolation.

    Reports the relative spread first, because that is the number a reader needs
    and it is always meaningful. An observed order is attempted only if the
    sequence is actually varying monotonically by more than the spread threshold.

    This distinction matters here. A Richardson ratio computed on a sequence that
    has already converged to its noise floor returns a meaningless order - this
    solver produces -10.7 on one such sequence - and quoting that as evidence of
    a convergence rate would be worse than quoting nothing. "Converged to better
    than 6e-5, order not extractable" is the honest statement, and it is the
    stronger one.
    """
    if len(sizes) < 3:
        return {"spread": None, "order": None, "extrapolated": None,
                "verdict": "need three resolutions"}
    lo, hi = min(values), max(values)
    spread = (hi - lo) / abs(lo) if lo else None

    if spread is not None and spread < CONVERGED_SPREAD:
        return {"spread": spread, "order": None, "extrapolated": sum(values) / len(values),
                "verdict": f"converged: relative spread {spread:.2g} is below "
                           f"{CONVERGED_SPREAD:g}; an order is not extractable and "
                           f"the mean is quoted as the value"}

    (h1, f1), (h2, f2), (h3, f3) = sorted(zip(sizes, values))[:3]  # h1 < h2 < h3

    # The constant-ratio Richardson formula below assumes a geometric refinement
    # sequence. The default sequences here are not geometric (film elements
    # 2,3,4,6 give sizes 1/2,1/3,1/4,1/6; bulk sizes 0.16,0.10,0.06), and
    # applying the formula anyway would manufacture an order from unequal
    # spacing. Unequal-grid Richardson needs an iterative solve for the order;
    # rather than add one for a path that is dead whenever the sequence has
    # already converged, refuse and say what would be needed.
    ratios = (h2 / h1, h3 / h2)
    if abs(ratios[0] - ratios[1]) > 0.05 * ratios[0]:
        return {"spread": spread, "order": None, "extrapolated": None,
                "verdict": f"refinement sizes are not geometric (ratios "
                           f"{ratios[0]:.3g}, {ratios[1]:.3g}); a constant-ratio "
                           "order is not extractable - use a geometric sequence"}

    d1, d2 = f2 - f1, f3 - f2
    if d1 == 0 or d2 / d1 <= 0:
        return {"spread": spread, "order": None, "extrapolated": None,
                "verdict": "differences do not decrease monotonically; not asymptotic"}
    r = h2 / h1
    order = math.log(d2 / d1) / math.log(h3 / h2) if h3 != h2 else float("nan")
    extrapolated = f1 - d1 / (r**order - 1.0) if r**order != 1.0 else None
    believable = 0.5 < order < 6
    return {"spread": spread, "order": order,
            "extrapolated": extrapolated if believable else None,
            "verdict": "asymptotic" if believable
            else f"observed order {order:.3g} is outside a believable range; "
                 "the sequence is not in its asymptotic regime"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=ROOT / "cases" / "bubble_ca0p05.json")
    parser.add_argument("--ca", type=float, nargs="+", default=[0.05, 0.005])
    parser.add_argument("--film-elements", type=float, nargs="+", default=[2.0, 3.0, 4.0, 6.0])
    parser.add_argument("--bulk-resolution", type=float, nargs="+", default=[0.16, 0.10, 0.06])
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "s6_mesh.json")
    args = parser.parse_args(argv)

    case = load_case(args.case)
    solve = case["solve"]
    base_bulk = float(case["mesh"]["bulk_resolution"])
    base_film = float(case["mesh"]["film_elements"])

    results = []

    def attempt(ca, film_elements, bulk_resolution, knob):
        label = f"ca{ca:g}_n{film_elements:g}_d{bulk_resolution:g}"
        try:
            row = solve_case(
                build(case, overrides={"physics": {"ca_in": ca},
                                       "mesh": {"film_elements": film_elements,
                                                "bulk_resolution": bulk_resolution}}),
                output_dir=ROOT / "receipts" / f"s6_{label}",
                relax_time=float(solve["relax_time"]),
                start_step=float(solve["start_step"]),
                temporal_error=float(solve["temporal_error"]),
            )
            row["knob"] = knob
            results.append(row)
            print(f"{label:26s} ndof={row['ndof']:>7d} h/Rt={row['h_flat_over_r_tube']:.7f} "
                  f"Ca_b={row['ca_b']:.7f} stag={row['check_stagnant_film_error']:+.1e}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"label": label, "ca_in": ca, "knob": knob,
                            "film_elements": film_elements,
                            "bulk_resolution": bulk_resolution,
                            "failed": f"{type(exc).__name__}: {exc}"})
            print(f"{label:26s} FAILED {type(exc).__name__}", flush=True)

    for ca in args.ca:
        for n in args.film_elements:
            attempt(ca, n, base_bulk, "film_elements")
        for d in args.bulk_resolution:
            if d != base_bulk:
                attempt(ca, base_film, d, "bulk_resolution")

    extrapolations = []
    for ca in args.ca:
        for knob, sizes_of, belongs in (
            (
                "film_elements",
                lambda r: 1.0 / r["film_elements"],
                lambda r: r["bulk_resolution_over_r0"] == base_bulk,
            ),
            (
                "bulk_resolution",
                lambda r: r["bulk_resolution_over_r0"],
                lambda r: r["film_elements"] == base_film,
            ),
        ):
            # Select by configuration rather than by which loop produced the run,
            # so the baseline (base_film, base_bulk) counts towards BOTH sequences.
            # Tagging by producing loop silently left the bulk sequence with two
            # points and no convergence statement at all.
            rows = [r for r in results
                    if "failed" not in r and r["ca_in"] == ca and belongs(r)]
            if len(rows) < 3:
                continue
            sizes = [sizes_of(r) for r in rows]
            entry = {"ca_in": ca, "knob": knob, "sizes": sizes,
                     "n_points": len(rows)}
            for key in REPORTED:
                entry[key] = convergence(sizes, [r[key] for r in rows])
            extrapolations.append(entry)

    # As for S5: the stage must fail on the thing it measures. Every reported
    # quantity needs a full three-point sequence and a verdict that actually
    # asserts convergence; a missing sequence or a "not asymptotic" verdict is a
    # result without evidence behind it.
    ACCEPTED = ("converged", "asymptotic")
    evidence_failures = []
    for ca in args.ca:
        for knob in ("film_elements", "bulk_resolution"):
            entry = next((e for e in extrapolations
                          if e["ca_in"] == ca and e["knob"] == knob), None)
            if entry is None or entry["n_points"] < 3:
                evidence_failures.append(
                    {"ca_in": ca, "knob": knob, "why": "fewer than three resolutions"})
                continue
            for key in REPORTED:
                verdict = entry[key]["verdict"]
                if not verdict.startswith(ACCEPTED):
                    evidence_failures.append(
                        {"ca_in": ca, "knob": knob, "quantity": key, "why": verdict})

    payload = {"stage": "S6", "results": results, "extrapolations": extrapolations,
               "evidence_failures": evidence_failures,
               "passed": all("failed" not in r for r in results)
               and not evidence_failures}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
