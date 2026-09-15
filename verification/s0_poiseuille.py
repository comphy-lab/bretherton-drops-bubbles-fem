"""S0: steady Poiseuille flow in the bare window, against the analytic solution.

This is the cheapest test in the suite and the only one with an exact answer.
Steady flow in a straight circular tube is Poiseuille; a P2 velocity space
represents that parabola exactly on straight-sided elements, so both the flux and
the pressure drop must come out at solver precision, not at discretisation error.

What it can catch that the full problem cannot:

* a wrong coordinate system or a mis-scaled axisymmetric measure - the flux would
  be out by a factor of ``2 pi`` or by an ``r`` weighting;
* an ``EnforcedDirichlet`` that has been silently deactivated by a strong
  ``DirichletBC`` on the same component - the inflow profile would be wrong;
* a missing pressure datum when both ends carry prescribed normal velocity - the
  solve would fail on a singular Jacobian.

None of these announce themselves once a free surface and two global unknowns are
in the system, which is why this runs first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.bretherton_bubble import PoiseuilleTubeProblem


def run(outlet: str, resolution: float, ca: float, length: float, rtol: float) -> dict:
    problem = PoiseuilleTubeProblem(
        ca_in=ca, l_domain=length, bulk_resolution=resolution, outlet=outlet
    )
    problem.set_output_directory(str(ROOT / "receipts" / f"s0_{outlet}_{resolution:g}"))
    problem.initialise()
    problem.solve()

    flux = problem.measured_flux()
    drop = problem.measured_pressure_drop()
    result = {
        "outlet": outlet,
        "resolution": resolution,
        "ca_in": ca,
        "l_domain": length,
        "flux": flux,
        "flux_expected": problem.expected_flux,
        "flux_rel_error": flux / problem.expected_flux - 1.0,
        "pressure_drop": drop,
        "pressure_drop_expected": problem.expected_pressure_drop,
        "pressure_drop_rel_error": drop / problem.expected_pressure_drop - 1.0,
    }
    # Mass conservation is exact for both outlets and is held to `rtol`.
    #
    # The pressure drop is exact only for the Poiseuille outlet. The
    # do-nothing outlet sets the whole traction vector to zero, which Poiseuille
    # violates: its radial traction mu*du_z/dr is nonzero at the outflow face. The
    # result is a localised outlet disturbance carrying a fixed ABSOLUTE pressure
    # deficit (measured: 0.0663 at Ca = 0.05, unchanged from L = 8 to L = 16), so
    # the relative error falls as 1/L. Holding that variant to `rtol` would be a
    # test failing for a physical reason, so it is recorded instead, and what is
    # asserted is the property that distinguishes an artefact from a bug: the
    # deficit must be independent of window length.
    result["passed"] = abs(result["flux_rel_error"]) < rtol and (
        outlet != "poiseuille" or abs(result["pressure_drop_rel_error"]) < rtol
    )
    result["pressure_drop_abs_deficit"] = (
        result["pressure_drop_expected"] - result["pressure_drop"]
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=float, nargs="+", default=[0.12, 0.06])
    parser.add_argument("--ca", type=float, default=0.05)
    parser.add_argument("--length", type=float, nargs="+", default=[8.0, 16.0])
    parser.add_argument("--deficit-rtol", type=float, default=0.05,
                        help="tolerance on the traction-free deficit's window independence")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--outlet", nargs="+", default=["poiseuille", "traction_free"])
    parser.add_argument("--out", type=Path,
                        default=ROOT / "receipts" / "s0.json",
                        help="receipt path. An ordinary run writes one, like every "
                             "other stage; the JSON is printed either way.")
    args = parser.parse_args(argv)

    results = [
        run(outlet, resolution, args.ca, length, args.rtol)
        for outlet in args.outlet
        for resolution in args.resolution
        for length in args.length
    ]

    # The traction-free pressure drop is deliberately not held to `rtol`: the
    # do-nothing outlet is an inexact boundary condition and carries a real
    # deficit. What distinguishes that artefact from a code error is that the
    # deficit is ABSOLUTE - independent of window length, so the relative error
    # falls as 1/L. Assert exactly that, otherwise the traction-free rows are
    # recorded without any claim attached and the distinction is asserted in
    # prose rather than measured.
    deficit_checks = []
    for resolution in args.resolution:
        rows = [r for r in results
                if r["outlet"] == "traction_free" and r["resolution"] == resolution]
        if len(rows) < 2:
            continue
        deficits = [r["pressure_drop_abs_deficit"] for r in rows]
        spread = (max(deficits) - min(deficits)) / abs(min(deficits))
        deficit_checks.append({
            "resolution": resolution,
            "lengths": [r["l_domain"] for r in rows],
            "deficits": deficits,
            "relative_spread": spread,
            "passed": spread < args.deficit_rtol,
        })
    deficit_required = "traction_free" in args.outlet and len(args.length) > 1

    payload = {"stage": "S0", "rtol": args.rtol, "results": results,
               "deficit_window_independence": deficit_checks,
               "passed": all(r["passed"] for r in results)
               and (not deficit_required
                    or (bool(deficit_checks) and all(c["passed"] for c in deficit_checks)))}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
