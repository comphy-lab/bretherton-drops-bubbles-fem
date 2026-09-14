"""S8: does inertia matter, and is the non-inertial frame term exercised at all?

Two jobs.

**It exercises a code path nothing else reaches.** At ``La = 0`` the solver uses
``StokesEquations`` and there is no frame-acceleration term. Only ``La > 0``
selects ``NavierStokesEquations`` with
``gravity = vector(0,-1)*partial_t(U)``, the fictitious force that makes the
co-moving frame consistent while ``U`` is still changing. Without this stage
that term would be written, plausible and never executed. It appears only in the
transient, since a stationary solve zeroes time derivatives.

**It quantifies the inertial correction.** The parent Basilisk campaign fixes
``La = 1`` and never sweeps it; its report raises inertia only as future work.
The question this answers is whether ``La = 1`` was doing anything at all.

Expect the correction to scale with ``Re = La Ca_b``, which is well below one
over the range of interest, so the film should move by a fraction of a percent.
The measured result is the point: if it moved by several percent, the parent's
unswept ``La`` would be a confounder in every comparison, and it is not.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, default=0.05)
    parser.add_argument("--la", type=float, nargs="+", default=[0.0, 1.0, 10.0])
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "s8_inertia.json")
    args = parser.parse_args(argv)

    if 0.0 not in args.la:
        raise SystemExit("La = 0 must be included: it is the Stokes reference")

    results = []
    for la in sorted(args.la):
        geom = CapsuleGeometry.for_capillary_number(args.ca, pad_rear=3.0, pad_front=3.0)
        problem = VoidBubbleProblem(
            geometry=geom, ca_in=args.ca, laplace=la, film_elements=3.0,
            bulk_resolution=0.10, closure="centroid", outlet="poiseuille",
        )
        try:
            row = solve_case(problem, output_dir=ROOT / "receipts" / f"s8_la{la:g}",
                             relax_time=8.0, start_step=1e-4, temporal_error=1e-4)
            row["reynolds"] = la * row["ca_b"]
            results.append(row)
            print(f"La={la:<7g} Re={row['reynolds']:<10.4g} "
                  f"h/Rt={row['h_flat_over_r_tube']:.7f} Ca_b={row['ca_b']:.7f} "
                  f"[{row['route']}]", flush=True)
        except Exception as exc:  # noqa: BLE001 - a failed point is the result
            results.append({"laplace": la, "failed": f"{type(exc).__name__}: {exc}"})
            print(f"La={la:<7g} FAILED {type(exc).__name__}", flush=True)

    ok = [r for r in results if "failed" not in r]
    stokes = next((r for r in ok if r["laplace"] == 0.0), None)
    correction = []
    if stokes:
        for r in ok:
            if r["laplace"] == 0.0:
                continue
            dh = r["h_flat_over_r_tube"] / stokes["h_flat_over_r_tube"] - 1.0
            correction.append({
                "laplace": r["laplace"],
                "reynolds": r["reynolds"],
                "h_rel_change_vs_stokes": dh,
                "ca_b_rel_change_vs_stokes": r["ca_b"] / stokes["ca_b"] - 1.0,
                # If the correction is linear in Re this is constant across La,
                # which is the check that it is inertia and not a solver artefact.
                "h_change_per_unit_reynolds": dh / r["reynolds"] if r["reynolds"] else None,
            })

    payload = {"stage": "S8", "ca_in": args.ca, "results": results,
               "inertial_correction": correction,
               "passed": bool(ok) and len(ok) == len(results)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(correction, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
