"""S11: where does the rear cap stop being convex, and is that onset resolved?

At low capillary number the rear cap is convex and carries the capillary ripple
where it joins the film. Above ``Ca_b`` of order one it folds back: the on-axis
rear tip moves *ahead* of the rearmost point of the interface, so the trailing
surface encloses an axisymmetric indentation. This stage measures that
indentation,

    delta_tail = z(on-axis tip) - min z over the rear cap,

and brackets the capillary number at which it becomes non-zero.

**The point of the stage is the resolution pair, not the onset value.** Near
onset the indentation is a few parts in 10^5 of ``R_t``, which is small enough
that it could as easily be a property of the mesh as of the solution. Running
every capillary number at two film resolutions is what distinguishes those: if
the two disagree by a factor of several the indentation is at the level of the
discretisation and the onset cannot be located there, whereas once they agree
the depth is a solution feature. Reporting a single-resolution onset would be
reporting the mesh.

The stage is deliberately not a pass/fail on the onset value, because no
reference value exists to fail against. It fails in three ways: if the two
resolutions disagree where the indentation is large enough that they should
not, so the *resolved* part of the curve is not resolved after all; if any case
fails to solve; or if no point reaches an indentation large enough to compare
resolutions at all, since a stage with nothing to compare has produced no
evidence and must not report a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.bretherton_bubble import VoidBubbleProblem
from problems.geometry import CapsuleGeometry


def measure(ca_in: float, film_elements: float, output_root: Path) -> dict:
    geom = CapsuleGeometry.for_capillary_number(
        ca_in, pad_rear=1.5, pad_front=3.0, seed_from_bubble_speed=True)
    problem = VoidBubbleProblem(
        geometry=geom, ca_in=ca_in, laplace=0.0, film_elements=film_elements,
        bulk_resolution=0.10, closure="centroid", outlet="poiseuille",
    )
    problem.set_output_directory(str(output_root / f"s11_ca{ca_in:g}_n{film_elements:g}"))
    problem.initialise()
    problem.audit_constraints()
    problem.solve_to_steady()
    checks = problem.consistency()

    # The polyline is walked from the rear tip along the element chain, so the
    # rear cap is its leading portion up to where the radius first reaches the
    # film. A z-sort would be useless here by construction: the whole quantity
    # being measured is that r(z) has become multivalued.
    z, r = problem.interface_polyline()
    film_radius = 0.9 * (geom.r_tube - checks.h_flat)
    reached = np.argmax(r > film_radius)
    stop = int(reached) if reached else len(r)
    return {
        "ca_in": ca_in,
        "film_elements": film_elements,
        "film_element_size_over_r0": problem.film_element_size() / geom.r0,
        "bulk_resolution_over_r0": problem.bulk_resolution / geom.r0,
        "ca_b": checks.frame_speed,
        "h_over_r_tube": checks.h_flat / geom.r_tube,
        "indentation_over_r_tube": float(z[0] - z[:stop].min()) / geom.r_tube,
        "check_void_volume_error": checks.volume_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, nargs="+",
                        default=[0.4788, 0.485, 0.49, 0.495, 0.5, 0.51, 0.53, 0.56, 0.60],
                        help="Ca_in values bracketing the onset")
    parser.add_argument("--film-elements", type=float, nargs=2, default=[3.0, 6.0],
                        help="the two film resolutions to compare")
    parser.add_argument("--resolved-above", type=float, default=4e-4,
                        help="indentation, over R_t, above which the two "
                             "resolutions are required to agree")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="relative agreement required there")
    parser.add_argument("--output-root", type=Path, default=ROOT / "receipts",
                        help="directory to hold per-case solver output")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "receipts" / "s11_rear_cap.json")
    args = parser.parse_args(argv)
    if len(set(args.film_elements)) != 2:
        parser.error("--film-elements needs two distinct resolutions; equal ones "
                     "compare a case with itself")
    args.output_root.mkdir(parents=True, exist_ok=True)

    coarse, fine = sorted(args.film_elements)
    rows: list[dict] = []
    for ca_in in sorted(args.ca):
        pair = {}
        for n in (coarse, fine):
            try:
                pair[n] = measure(ca_in, n, args.output_root)
            except Exception as exc:  # noqa: BLE001 - a failed point is a result
                pair[n] = {"ca_in": ca_in, "film_elements": n,
                           "failed": f"{type(exc).__name__}: {exc}"}
                print(f"Ca_in={ca_in:<7g} n={n:g} FAILED {type(exc).__name__}", flush=True)
        if any("failed" in p for p in pair.values()):
            rows.append({"ca_in": ca_in, "cases": list(pair.values())})
            continue
        d_c, d_f = (pair[coarse]["indentation_over_r_tube"],
                    pair[fine]["indentation_over_r_tube"])
        row = {
            "ca_in": ca_in,
            "ca_b": pair[coarse]["ca_b"],
            "h_over_r_tube": pair[coarse]["h_over_r_tube"],
            "indentation_coarse": d_c,
            "indentation_fine": d_f,
            "convex_at_both": d_c == 0.0 and d_f == 0.0,
            "relative_difference": abs(d_c / d_f - 1.0) if d_f > 0 else None,
            "cases": [pair[coarse], pair[fine]],
        }
        rows.append(row)
        print(f"Ca_b={row['ca_b']:<9.5f} n={coarse:g} {d_c:.3e}  n={fine:g} {d_f:.3e}"
              + ("  convex at both" if row["convex_at_both"] else ""), flush=True)

    # Bracket: the largest Ca_b that is convex at both resolutions, and the
    # smallest that is indented at both.
    convex = [r["ca_b"] for r in rows if r.get("convex_at_both")]
    # "Indented at both" has to mean both, not just the finer mesh. A row with a
    # convex coarse cap and a barely non-zero fine one is exactly the ambiguous
    # case the bracket exists to exclude, and testing only the fine resolution
    # would place the bracket's upper end on it.
    indented = [r["ca_b"] for r in rows
                if r.get("indentation_coarse", 0.0) > 0.0
                and r.get("indentation_fine", 0.0) > 0.0]
    bracket = [max(convex) if convex else None, min(indented) if indented else None]

    # Where the indentation is large enough to be resolved, the two resolutions
    # must agree; that is the only falsifiable claim this stage makes.
    checked = [r for r in rows
               if max(r.get("indentation_coarse", 0.0),
                      r.get("indentation_fine", 0.0)) > args.resolved_above
               and r.get("relative_difference") is not None]
    bad = [r for r in checked if r["relative_difference"] > args.tolerance]
    for r in bad:
        print(f"FAIL Ca_b={r['ca_b']:.5g}: resolutions differ by "
              f"{100*r['relative_difference']:.1f}% where the indentation is "
              f"{max(r['indentation_coarse'], r['indentation_fine']):.2e}", flush=True)

    # A stage that solved nothing, or that lost cases on the way, has produced
    # no evidence; reporting that as a pass is the vacuous pass this suite is
    # built to avoid. Both conditions are recorded separately from the
    # resolution disagreement so the receipt says which one fired.
    failed = [c for r in rows for c in r.get("cases", []) if "failed" in c]
    if failed:
        print(f"FAIL {len(failed)} case(s) did not solve", flush=True)
    if not checked:
        print("FAIL no point reached an indentation large enough to compare "
              "resolutions; the stage has no evidence to offer", flush=True)
    passed = not bad and not failed and bool(checked)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "check": "s11_rear_cap",
        "passed": passed,
        "n_failed_cases": len(failed),
        "note": (
            "indentation_over_r_tube is the axial distance by which the on-axis "
            "rear tip leads the rearmost point of the interface, over R_t. It is "
            "zero for a convex cap and positive once the cap folds back. The "
            "onset is reported as a bracket rather than a value because between "
            "the bracketing capillary numbers the indentation is at the level of "
            "the discretisation; the pass condition is agreement between the two "
            "resolutions where the indentation is large enough to be resolved."
        ),
        "onset_bracket_ca_b": bracket,
        "resolved_above": args.resolved_above,
        "tolerance": args.tolerance,
        "n_resolution_checks": len(checked),
        "results": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nonset bracketed between Ca_b = {bracket[0]} and {bracket[1]}; "
          f"{len(checked)} resolved point(s) checked against the two resolutions")
    print(f"wrote {args.out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
