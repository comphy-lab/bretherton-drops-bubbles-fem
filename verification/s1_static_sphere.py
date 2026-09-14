"""S1: an unconfined void must relax to a sphere with a Laplace pressure of 2.

The seeded capsule is not an equilibrium shape: its cylindrical body has one
principal curvature (``1/R_b``) where its caps have two (``2/R_b``), so surface
tension drives it towards the sphere of equal volume. In the nondimensionalisation
``sigma = R0 = 1`` that sphere has radius exactly 1 and the pressure jump across
its surface is exactly 2.

This is run in a *wide* tube, ``R_tube = 1.5``, so the void is unconfined and the
analytic answer applies. It deliberately does not test the Bretherton film; it
tests the machinery that the film calculation is built on, all at once and
against a number known in closed form:

* the surface-tension term and its sign,
* the free-surface kinematic condition and the mesh motion it drives,
* the fixed-volume constraint and the sign of the void pressure ``P``,
* the axisymmetric measure on a curved interface.

A confined capsule cannot be used for this: in a narrow tube a static bubble has
no analytic equilibrium, so there would be nothing to check against. That is why
the tube is widened rather than the flow switched off in the production geometry.
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

from problems.bretherton_bubble import VoidBubbleProblem
from problems.geometry import CapsuleGeometry


def run(*, r_tube: float, r_body: float, resolution: float, ca: float,
        rtol: float, transient: float) -> dict:
    geom = CapsuleGeometry(
        r0=1.0, r_tube=r_tube, r_body=r_body, z_rear=2.0,
        l_domain=2.0 + CapsuleGeometry.bubble_length(1.0, r_body) + 2.0,
    )
    problem = VoidBubbleProblem(
        geometry=geom, ca_in=ca, laplace=0.0, film_elements=2.0,
        bulk_resolution=resolution, closure="centroid", outlet="poiseuille",
        remesh=False,
    )
    problem.set_output_directory(str(ROOT / "receipts" / f"s1_{resolution:g}"))
    problem.initialise()
    audit = problem.audit_constraints()

    if transient > 0:
        problem.run(transient, startstep=1e-3, outstep=transient / 4, temporal_error=1e-4)
    problem.solve(max_newton_iterations=40)

    z, r = problem.interface_polyline()
    # Distance of every interface node from the void centroid: constant, and
    # equal to 1, if and only if the shape is the unit sphere.
    centre = 0.5 * (float(z.min()) + float(z.max()))
    radii = [math.hypot(float(ri), float(zi) - centre) for zi, ri in zip(z, r)]
    laplace_jump = problem.bubble_pressure() - problem.outlet_mean_pressure()

    result = {
        "r_tube": r_tube,
        "r_body_seed": r_body,
        "resolution": resolution,
        "sphere_radius_mean": sum(radii) / len(radii),
        "sphere_radius_max_dev": max(abs(x - 1.0) for x in radii),
        "laplace_jump": laplace_jump,
        "laplace_jump_expected": 2.0,
        "laplace_rel_error": laplace_jump / 2.0 - 1.0,
        "frame_speed": problem.frame_speed(),
        "void_volume_audit": audit,
        "volume_error": problem.observable("void_volume", "liquid/interface")
        / geom.volume
        - 1.0,
    }
    result["passed"] = (
        abs(result["laplace_rel_error"]) < rtol
        and result["sphere_radius_max_dev"] < rtol
        and abs(result["volume_error"]) < 1e-6
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-tube", type=float, default=1.5)
    parser.add_argument("--r-body", type=float, default=0.9)
    parser.add_argument("--resolution", type=float, nargs="+", default=[0.12, 0.06])
    parser.add_argument("--ca", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--transient", type=float, default=0.0,
                        help="relax for this long before the stationary solve")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    results = []
    for resolution in args.resolution:
        try:
            results.append(run(r_tube=args.r_tube, r_body=args.r_body,
                               resolution=resolution, ca=args.ca, rtol=args.rtol,
                               transient=args.transient))
        except Exception as exc:  # noqa: BLE001 - a failure here is the result
            results.append({"resolution": resolution, "passed": False,
                            "error": f"{type(exc).__name__}: {exc}"})
    payload = {"stage": "S1", "rtol": args.rtol, "results": results,
               "passed": all(r["passed"] for r in results)}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
