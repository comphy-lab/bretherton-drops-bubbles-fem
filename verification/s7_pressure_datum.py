"""S7: does the pressure datum matter, and does the fixation actually work?

With Poiseuille imposed at both ends, every boundary carries a prescribed normal
velocity and the system has an exact gauge freedom: shifting the liquid pressure,
the void pressure ``P`` and the inlet/outlet enforced-velocity multipliers
together by a constant changes no residual. A datum has to remove it.

An adversarial diagnostic pass found a near-null Jacobian mode at
``sigma_min/sigma_max ~ 4e-11`` supported entirely on pressure dofs, ``globals/P``
and those multipliers - the fingerprint of exactly this gauge - **while
``with_pressure_fixation()`` was active**. That is a claim about the solver, so it
is settled by measurement here rather than by reading the source.

The test is not "does it converge" - a gauge mode is harmless to a direct solver
and to every observable we report, because ``consistency()`` already references
the void pressure to the mean outlet pressure. The test is whether the three
datum choices give the *same physical answers*. If they do, the gauge is
confirmed harmless and the choice is free. If they do not, one of them is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.bretherton_bubble import VoidBubbleProblem
from problems.geometry import CapsuleGeometry
from runners.run_case import solve_case

# Gauge-invariant: these must agree across datum choices.
INVARIANT = ("h_flat_over_r_tube", "ca_b", "bubble_pressure_gauge")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, default=0.05)
    parser.add_argument("--data", nargs="+", default=["fixation", "integral", "none"])
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "s7_pressure_datum.json")
    args = parser.parse_args(argv)

    results = []
    for datum in args.data:
        geom = CapsuleGeometry.for_capillary_number(args.ca, pad_rear=3.0, pad_front=3.0)
        problem = VoidBubbleProblem(
            geometry=geom, ca_in=args.ca, laplace=0.0, film_elements=3.0,
            bulk_resolution=0.10, closure="centroid", outlet="poiseuille",
            pressure_datum=datum, remesh=True,
        )
        try:
            row = solve_case(problem, output_dir=ROOT / "receipts" / f"s7_{datum}",
                             relax_time=8.0, start_step=1e-4, temporal_error=1e-4)
            row["pressure_datum"] = datum
            results.append(row)
            print(f"{datum:10s} h/Rt={row['h_flat_over_r_tube']:.9f} "
                  f"Ca_b={row['ca_b']:.9f} P-p_out={row['bubble_pressure_gauge']:.7f}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"pressure_datum": datum, "failed": f"{type(exc).__name__}: {exc}"})
            print(f"{datum:10s} FAILED {type(exc).__name__}: {exc}", flush=True)

    ok = [r for r in results if "failed" not in r]
    spread = {}
    if len(ok) >= 2:
        for key in INVARIANT:
            values = [r[key] for r in ok]
            spread[key] = max(values) / min(values) - 1.0 if min(values) else None

    # `all({})` is True, so an empty spread would have passed this stage while
    # proving nothing. Be explicit about what success requires:
    #   - both real datum mechanisms converged;
    #   - a spread was computed for every gauge-invariant quantity, and each is
    #     within tolerance;
    #   - if "none" was requested, it FAILED. That is the measured expectation,
    #     and it is the only evidence that the datum is doing anything at all;
    #     a "none" that converged would mean the gauge is being removed by
    #     something else and the whole argument needs revisiting.
    converged = {r["pressure_datum"] for r in ok}
    mechanisms_ok = {"fixation", "integral"} <= converged
    spread_ok = bool(spread) and all(
        v is not None and abs(v) < args.rtol for v in spread.values()
    ) and set(spread) == set(INVARIANT)
    none_ok = "none" not in args.data or "none" not in converged

    payload = {
        "stage": "S7", "ca_in": args.ca, "rtol": args.rtol, "results": results,
        "relative_spread_across_data": spread,
        "mechanisms_converged": sorted(converged),
        "checks": {"both_mechanisms_converged": mechanisms_ok,
                   "spread_within_tolerance": spread_ok,
                   "no_datum_fails_as_expected": none_ok},
        "passed": mechanisms_ok and spread_ok and none_ok,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"spread": spread, "passed": payload["passed"]}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
