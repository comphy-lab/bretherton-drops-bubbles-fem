"""S10: does the answer depend on how it was reached, or on what was seeded?

The sweep is produced by two routes. Up to ``Ca_in = 0.7`` each point is solved
from its own seeded capsule, independently of its neighbours; above that the
points come from an ascending continuation ladder, because a fresh transient
costs a full relaxation per point where a continued step costs under a second.
That switch is about cost, not capability - this stage measures a fresh solve
succeeding as high as ``Ca_in = 2``. A continued point inherits its
predecessor's state, so a systematic error at one capillary number would
propagate along the curve without ever announcing itself.

Three questions follow, and none of them is answered by the residual norm.

**Do the routes agree where both exist?** ``Ca_in`` in 1 to 2 is reachable both
ways. If the continued and fresh answers differ there, the continued half of the
curve is carrying its history.

**Is the ladder path-dependent?** Walking up to ``Ca_in = 6`` and back down
should return the same states. A difference would mean the continuation is
tracking something other than the branch - a drifting mesh, or a hysteretic
solution.

**Does the answer depend on how long the bubble is?** The void volume is held at
``4/3 pi R0^3`` throughout, so nothing here varies it. What varies is the
confinement: changing ``R_tube`` at fixed
``R0`` changes the bubble length by nearly a factor of two while leaving the
film's own geometry in units of ``R_tube`` alone, so any movement in ``h/R_t``
is a finite-bubble-length effect rather than the Landau-Levich film the
comparison assumes. Measured, that movement is 1.3e-3 at ``Ca_b = 0.011`` and
1.9e-4 at ``Ca_b = 0.585``, and it is *resolved*: doubling the film elements
moves each confinement by under 1e-5 and leaves the difference at 1.33e-3. So
the residual is physics, not discretisation - the shorter bubble's film has not
quite reached its plateau - and the length tolerance is set above it rather
than pretending it is zero. It also bounds how much of any parent comparison
can be attributed to a difference in ``R_0/R_t``.

Each is a falsifiable statement about the curve, not a restatement of how well
Newton converged.
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

# The production film resolution, and the base of the refinement comparison.
BASE_FILM_ELEMENTS = 3.0


def fresh(ca: float, *, r_tube: float, tag: str, output_root: Path,
          film_elements: float = BASE_FILM_ELEMENTS) -> tuple[VoidBubbleProblem, CapsuleGeometry]:
    geom = CapsuleGeometry.for_capillary_number(
        ca, r_tube=r_tube, pad_rear=1.5, pad_front=3.0, seed_from_bubble_speed=True)
    problem = VoidBubbleProblem(
        geometry=geom, ca_in=ca, laplace=0.0, film_elements=film_elements,
        bulk_resolution=0.10, closure="centroid", outlet="poiseuille",
    )
    problem.set_output_directory(str(output_root / f"s10_{tag}"))
    problem.initialise()
    problem.audit_constraints()
    problem.solve_to_steady()
    return problem, geom


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap", type=float, nargs="+", default=[1.0, 1.5, 2.0],
                        help="Ca_in values reachable by both routes")
    parser.add_argument("--anchor", type=float, default=0.7,
                        help="Ca_in of the fresh solve the ladder starts from")
    parser.add_argument("--ladder", type=float, nargs="+", default=[1.0, 2.0, 4.0, 6.0],
                        help="ascending ladder, retraced downward for hysteresis")
    parser.add_argument("--length-ca", type=float, nargs="+", default=[0.01, 0.3],
                        help="Ca_in at the two extremes of the AQ deviation")
    parser.add_argument("--r-tube", type=float, nargs=2, default=[0.7, 0.5],
                        help="two confinements, hence two bubble lengths at fixed R0")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="relative agreement required of the route comparisons")
    parser.add_argument("--refine-to", type=float, default=6.0,
                        help="film elements for the refined repeat of the "
                             "bubble-length comparison")
    parser.add_argument("--self-convergence-tolerance", type=float, default=1e-4,
                        help="how far each confinement may move under refinement "
                             "before its bubble-length comparison stops being "
                             "evidence about physics")
    parser.add_argument("--length-tolerance", type=float, default=2e-3,
                        help="separate, looser bound for the bubble-length comparison, "
                             "which carries a resolved physical difference rather than "
                             "a numerical one")
    parser.add_argument("--output-root", type=Path, default=ROOT / "receipts",
                        help="directory to hold per-case solver output. The default "
                             "is gitignored and fine locally; a dispatched run "
                             "places it on a registered working volume instead.")
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "s10_route.json")
    args = parser.parse_args(argv)
    if len(set(args.r_tube)) != 2:
        parser.error("--r-tube needs two distinct radii; equal ones compare a case "
                     "with itself and report perfect agreement")
    if len(set(args.ladder)) < 2:
        parser.error("--ladder needs at least two distinct values to retrace")
    if args.refine_to <= BASE_FILM_ELEMENTS:
        # Otherwise self_convergence is zero by construction and the receipt
        # records a refinement comparison that was never refined - which reads
        # exactly like a converged result.
        parser.error(f"--refine-to must exceed the base {BASE_FILM_ELEMENTS:g} film "
                     "elements, or there is no refinement to report")

    args.output_root.mkdir(parents=True, exist_ok=True)

    out: dict = {"check": "s10_route", "tolerance": args.tolerance,
                 "length_tolerance": args.length_tolerance}
    r_tube = args.r_tube[0]

    # --------------------------------------------------- fresh versus continued
    overlap = []
    problem, geom = fresh(args.anchor, r_tube=r_tube, tag="overlap_anchor",
                            output_root=args.output_root)
    for target in sorted(args.overlap):
        problem.go_to_param(Ca_in=target, max_step=0.25)
        cont = problem.consistency()
        other, other_geom = fresh(target, r_tube=r_tube, tag=f"overlap_fresh{target:g}",
                                    output_root=args.output_root)
        direct = other.consistency()
        row = {
            "ca_in": target,
            "continued_ca_b": cont.frame_speed,
            "fresh_ca_b": direct.frame_speed,
            "continued_h_over_r_tube": cont.h_flat / geom.r_tube,
            "fresh_h_over_r_tube": direct.h_flat / other_geom.r_tube,
            "rel_diff_ca_b": abs(cont.frame_speed / direct.frame_speed - 1.0),
            "rel_diff_h": abs((cont.h_flat / geom.r_tube) / (direct.h_flat / other_geom.r_tube) - 1.0),
        }
        overlap.append(row)
        print(f"OVERLAP  Ca_in={target:<6g} dCa_b={row['rel_diff_ca_b']:.2e} "
              f"dh={row['rel_diff_h']:.2e}", flush=True)
    out["overlap_fresh_vs_continued"] = overlap

    # ------------------------------------------------------------- hysteresis
    up: dict[float, tuple[float, float]] = {}
    down: dict[float, tuple[float, float]] = {}
    problem, geom = fresh(args.anchor, r_tube=r_tube, tag="hysteresis",
                            output_root=args.output_root)
    ladder = sorted(args.ladder)
    for target in ladder:
        problem.go_to_param(Ca_in=target, max_step=0.25)
        c = problem.consistency()
        up[target] = (c.frame_speed, c.h_flat / geom.r_tube)
    for target in sorted(ladder[:-1], reverse=True):
        problem.go_to_param(Ca_in=target, max_step=0.25)
        c = problem.consistency()
        down[target] = (c.frame_speed, c.h_flat / geom.r_tube)
    hysteresis = []
    for target in sorted(down):
        row = {
            "ca_in": target,
            "up_ca_b": up[target][0], "down_ca_b": down[target][0],
            "up_h_over_r_tube": up[target][1], "down_h_over_r_tube": down[target][1],
            "rel_diff_ca_b": abs(down[target][0] / up[target][0] - 1.0),
            "rel_diff_h": abs(down[target][1] / up[target][1] - 1.0),
        }
        hysteresis.append(row)
        print(f"RETRACE  Ca_in={target:<6g} dCa_b={row['rel_diff_ca_b']:.2e} "
              f"dh={row['rel_diff_h']:.2e}", flush=True)
    out["hysteresis_up_vs_down"] = hysteresis

    # -------------------------------------------------- bubble-length dependence
    length = []
    for ca in sorted(args.length_ca):
        rows = []
        for rt in args.r_tube:
            # Both confinements are also run at doubled film resolution, because
            # the two carry different absolute film thicknesses at the same
            # element count; without that the comparison could report a mesh
            # difference as a bubble-length difference.
            by_n = {}
            for n in (BASE_FILM_ELEMENTS, args.refine_to):
                p, g = fresh(ca, r_tube=rt, tag=f"length_ca{ca:g}_rt{rt:g}_n{n:g}",
                             output_root=args.output_root, film_elements=n)
                c = p.consistency()
                by_n[n] = (c.h_flat / rt, c.frame_speed)
            rows.append({
                "r_tube": rt, "r0_over_r_tube": g.r0 / rt,
                "bubble_length": g.z_front - g.z_rear,
                "ca_b": by_n[BASE_FILM_ELEMENTS][1],
                "h_over_r_tube": by_n[BASE_FILM_ELEMENTS][0],
                "h_over_r_tube_refined": by_n[args.refine_to][0],
                "film_elements_refined": args.refine_to,
                "self_convergence": abs(
                    by_n[args.refine_to][0] / by_n[BASE_FILM_ELEMENTS][0] - 1.0),
            })
        row = {
            "ca_in": ca, "cases": rows,
            "rel_diff_h": abs(rows[0]["h_over_r_tube"] / rows[1]["h_over_r_tube"] - 1.0),
            "rel_diff_ca_b": abs(rows[0]["ca_b"] / rows[1]["ca_b"] - 1.0),
            "length_ratio": rows[1]["bubble_length"] / rows[0]["bubble_length"],
            "rel_diff_h_refined": abs(
                rows[0]["h_over_r_tube_refined"] / rows[1]["h_over_r_tube_refined"] - 1.0),
        }
        length.append(row)
        print(f"LENGTH   Ca_in={ca:<6g} length x{row['length_ratio']:.2f} "
              f"dh={row['rel_diff_h']:.2e} (refined {row['rel_diff_h_refined']:.2e}, "
              f"self-conv {max(r['self_convergence'] for r in rows):.1e}) "
              f"dCa_b={row['rel_diff_ca_b']:.2e}", flush=True)
    out["bubble_length_dependence"] = length

    worst_route = max(
        [r["rel_diff_h"] for r in overlap]
        + [r["rel_diff_ca_b"] for r in overlap]
        + [r["rel_diff_ca_b"] for r in hysteresis]
        + [r["rel_diff_h"] for r in hysteresis]
    )
    worst_length = max(r["rel_diff_h"] for r in length)
    # The claim that the bubble-length difference is physics rests entirely on
    # it surviving refinement, so that has to be a pass condition too. Without
    # this the stage would accept a badly under-resolved pair whose agreement
    # happened to fall inside the tolerance.
    worst_self = max(c["self_convergence"] for r in length for c in r["cases"])
    worst_length_refined = max(r["rel_diff_h_refined"] for r in length)
    out["worst_route_disagreement"] = worst_route
    out["worst_bubble_length_difference"] = worst_length
    out["worst_bubble_length_difference_refined"] = worst_length_refined
    out["worst_self_convergence"] = worst_self
    out["self_convergence_tolerance"] = args.self_convergence_tolerance
    out["passed"] = not (worst_route > args.tolerance
                         or worst_length > args.length_tolerance
                         or worst_length_refined > args.length_tolerance
                         or worst_self > args.self_convergence_tolerance)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print(f"worst route disagreement {worst_route:.2e} against {args.tolerance:.1e}; "
          f"worst bubble-length difference {worst_length:.2e} (refined "
          f"{worst_length_refined:.2e}) against {args.length_tolerance:.1e}; "
          f"worst self-convergence {worst_self:.2e} against "
          f"{args.self_convergence_tolerance:.1e}", flush=True)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
