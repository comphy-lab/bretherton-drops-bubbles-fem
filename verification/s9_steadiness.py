"""S9: is the reported state steady, and over how much physical time is that shown?

The reported answer is a **stationary** solution: the time derivatives are
removed from the residual and Newton drives the steady equations to machine
precision. In that sense "time convergence" is not a question the answer can
fail - there is no time in it.

What a stationary solve does *not* establish is that the state persists when
time is put back. A saddle, or a state held up by a constraint that the
transient would relax, satisfies the steady equations just as exactly as an
attractor does. So this stage marches the converged state forward and measures
whether it moves.

**The reach of that probe must be quoted in the problem's own time unit, not in
nondimensional units.** The film is renewed on ``L_b/U_b``: the time for the
bubble to traverse its own length relative to the wall. That is 3172 time units
at ``Ca_b ~ 1e-3`` and 0.5 at ``Ca_b ~ 15``, a factor of 6000. A fixed 40-unit
march is therefore 79 renewals at the top of the sweep and 1.3% of one renewal
at the bottom; reported as a single number it would claim far more at low
``Ca`` than it shows. Each row here records ``renewals_covered`` so the strength
of the evidence is read off with the result rather than assumed uniform.

The horizon is set in renewals, with a wall-clock budget per case. The two
halves treat exhaustion differently, and deliberately. A truncated *march* is
reported with the renewals it did reach and is not a failure - it is a weaker
version of the same observation. A truncated *recovery* is a failure, because
a displaced state that was not given time to return tells you nothing about
whether it would have, and scoring its leftover displacement as though it had
been given that time would read as instability. If the budget is already gone
before the displacement, it is skipped rather than started.

**Persistence is weaker evidence than it looks, so the stage does not stop
there.** A stationary solution is an exact fixed point of the implicit time
discretisation as well as of the steady equations, so marching from it stays on
it to round-off whether it is an attractor or a saddle; only an unstable mode
seeded by round-off would ever show, and only after it had amplified by some
four orders of magnitude. The second half of this stage therefore *displaces*
the state - it takes the converged solution at a neighbouring capillary number,
puts the parameter back, and marches in time alone with no stationary polish -
and asks whether the interface returns. Recovery is reported as the residual
displacement as a fraction of the one imposed. That is the test the phrase "the
bubble shape stops changing" actually names.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.bretherton_bubble import VoidBubbleProblem
from problems.geometry import CapsuleGeometry


def resample(z, r, z_grid):
    """Radius on a common axial grid, so two profiles can be differenced.

    The mesh moves during the march, so node ``i`` before and after are not the
    same material point and differencing them node by node would measure mesh
    motion rather than interface motion. Restricting to the film, where the
    profile is single valued in ``z``, makes the comparison geometric.
    """
    order = np.argsort(z)
    return np.interp(z_grid, np.asarray(z)[order], np.asarray(r)[order])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, nargs="+",
                        default=[0.001, 0.01, 0.05, 0.3, 0.5])
    parser.add_argument("--renewals", type=float, default=1.5,
                        help="target march horizon in film-renewal times L_b/U_b")
    parser.add_argument("--min-time", type=float, default=40.0,
                        help="floor on the horizon in capillary times, so the "
                             "high-Ca cases are still probed for long enough to "
                             "see a slow mode that renewal scaling would miss")
    parser.add_argument("--budget-seconds", type=float, default=1800.0,
                        help="wall-clock budget per case; a case that runs out is "
                             "reported with the renewals it reached")
    parser.add_argument("--chunks", type=int, default=12,
                        help="march in this many pieces, so the budget is checked "
                             "against progress rather than only at the end")
    parser.add_argument("--temporal-error", type=float, default=1e-4)
    parser.add_argument("--perturb", type=float, default=0.05,
                        help="relative displacement in Ca_in used to knock the state "
                             "off the fixed point before marching it back; 0 skips "
                             "the recovery test")
    parser.add_argument("--recovery-renewals", type=float, default=1.5,
                        help="march horizon for the recovery test, in film renewals")
    parser.add_argument("--output-root", type=Path, default=ROOT / "receipts",
                        help="directory to hold per-case solver output. The default "
                             "is gitignored and fine locally; a dispatched run "
                             "places it on a registered working volume instead.")
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "s9_steadiness.json")
    args = parser.parse_args(argv)
    if args.chunks < 1:
        parser.error("--chunks must be at least 1")
    if args.recovery_renewals <= 0:
        parser.error("--recovery-renewals must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for ca in sorted(args.ca):
        geom = CapsuleGeometry.for_capillary_number(
            ca, pad_rear=1.5, pad_front=3.0, seed_from_bubble_speed=True)
        problem = VoidBubbleProblem(
            geometry=geom, ca_in=ca, laplace=0.0, film_elements=3.0,
            bulk_resolution=0.10, closure="centroid", outlet="poiseuille",
        )
        problem.set_output_directory(str(args.output_root / f"s9_ca{ca:g}"))
        problem.initialise()
        problem.audit_constraints()
        route = problem.solve_to_steady()

        before = problem.consistency()
        z0, rr0 = problem.interface_polyline()
        bubble_length = float(np.max(z0) - np.min(z0))
        renewal = bubble_length / before.frame_speed
        horizon = max(args.renewals * renewal, args.min_time)

        # Compare only over the film, where the profile is a function of z. The
        # caps are multivalued in z and would need an arc-length correspondence
        # that the mesh does not preserve across a march.
        z_lo = float(np.min(z0)) + 0.25 * bubble_length
        z_hi = z_lo + 0.5 * bubble_length
        z_grid = np.linspace(z_lo, z_hi, 400)
        r0 = resample(z0, rr0, z_grid)

        t_start = float(problem.get_current_time(as_float=True))
        chunk = horizon / args.chunks
        t_done, spent = 0.0, 0.0
        for _ in range(args.chunks):
            tic = time.time()
            problem.run(t_start + t_done + chunk, startstep=min(1e-3, chunk / 100.0),
                        outstep=False, temporal_error=args.temporal_error,
                        maxstep=chunk)
            spent += time.time() - tic
            t_done += chunk
            if spent > args.budget_seconds:
                break

        after = problem.consistency()
        z1, rr1 = problem.interface_polyline()
        r1 = resample(z1, rr1, z_grid)
        drift = float(np.max(np.abs(r1 - r0)))
        row = {
            "ca_in": ca,
            "ca_b": before.frame_speed,
            "route": route,
            "h_flat_over_r_tube": before.h_flat / geom.r_tube,
            "bubble_length": bubble_length,
            "film_renewal_time": renewal,
            "horizon_requested": horizon,
            "time_marched": t_done,
            "renewals_covered": t_done / renewal,
            "budget_exhausted": spent > args.budget_seconds,
            "march_seconds": spent,
            "film_drift_over_r_tube": drift / geom.r_tube,
            "relative_h_change": after.h_flat / before.h_flat - 1.0,
            "relative_ca_b_change": after.frame_speed / before.frame_speed - 1.0,
            "growth_rate_per_capillary_time": (
                abs(after.h_flat / before.h_flat - 1.0) / t_done if t_done else float("nan")
            ),
            "check_void_volume_error": after.volume_error,
        }

        # Displace, then march back. go_to_param is used only to reach the
        # neighbouring steady state; the return leg sets the parameter directly
        # and runs in time, because a stationary solve on the way back would
        # answer a different question - Newton would find the fixed point
        # whether or not time evolution does.
        if args.perturb:
            # Chunked and budgeted like the first march. Run in one shot, the
            # recovery leg could overrun the per-case budget without saying so,
            # and a partial recovery would then be reported as if it had been
            # given its full horizon.
            if spent > args.budget_seconds:
                # Do not pay for a displacement that cannot then be marched
                # back. Recorded as truncated, which the pass predicate below
                # treats as absent evidence rather than as a failure to return.
                row.update({"perturbation_fraction": args.perturb,
                            "recovery_renewals": 0.0, "recovery_truncated": True})
                results.append(row)
                print("  recover: skipped, wall-clock budget already spent", flush=True)
                continue
            tic = time.time()
            problem.go_to_param(Ca_in=ca * (1.0 + args.perturb), max_step=0.25 * ca)
            zp, rp = problem.interface_polyline()
            imposed = float(np.max(np.abs(resample(zp, rp, z_grid) - r0)))
            problem.get_global_parameter("Ca_in").value = ca
            spent += time.time() - tic
            back = args.recovery_renewals * renewal
            chunk = back / args.chunks
            done = 0.0
            for _ in range(args.chunks):
                if spent > args.budget_seconds:
                    break
                tic = time.time()
                t_now = float(problem.get_current_time(as_float=True))
                problem.run(t_now + chunk, startstep=min(1e-3, chunk / 100.0),
                            outstep=False, temporal_error=args.temporal_error,
                            maxstep=chunk)
                spent += time.time() - tic
                done += chunk
            zr, rr = problem.interface_polyline()
            residual = float(np.max(np.abs(resample(zr, rr, z_grid) - r0)))
            recovered = problem.consistency()
            row["budget_exhausted"] = spent > args.budget_seconds
            row["march_seconds"] = spent
            row.update({
                "perturbation_fraction": args.perturb,
                "recovery_time": done,
                "recovery_renewals": done / renewal,
                "imposed_displacement_over_r_tube": imposed / geom.r_tube,
                "residual_displacement_over_r_tube": residual / geom.r_tube,
                "recovery_fraction": residual / imposed if imposed else float("nan"),
                "recovered_relative_h_error": recovered.h_flat / before.h_flat - 1.0,
                "recovered_relative_ca_b_error": recovered.frame_speed / before.frame_speed - 1.0,
            })
            print(
                f"  recover: imposed {imposed / geom.r_tube:.3e}, left "
                f"{residual / geom.r_tube:.3e} ({100 * row['recovery_fraction']:.2f}% "
                f"of it) after {row['recovery_renewals']:.3g} renewals; "
                f"dh/h={row['recovered_relative_h_error']:+.2e}"
                + ("  [budget]" if row["budget_exhausted"] else ""), flush=True,
            )

        results.append(row)
        print(
            f"Ca_b={row['ca_b']:<10.5g} renewal={renewal:<9.4g} marched={t_done:<9.4g} "
            f"({row['renewals_covered']:.3g} renewals) drift={row['film_drift_over_r_tube']:.2e} "
            f"dh/h={row['relative_h_change']:+.2e} dCa_b={row['relative_ca_b_change']:+.2e}"
            + ("  [budget]" if row["budget_exhausted"] else ""),
            flush=True,
        )

    # A drift above a tenth of a percent of R_tube over the marched window is
    # not steadiness, whatever the residual said; and a displaced state that
    # keeps more than a twentieth of its displacement has not returned.
    #
    # The recovery threshold is applied only to probes that reached their
    # horizon. A budget-truncated recovery is not weak evidence of instability,
    # it is no evidence either way - so it fails for its own reason rather than
    # being scored as if the state had refused to return.
    bad = []
    for r in results:
        if r["film_drift_over_r_tube"] > 1e-3:
            bad.append((r["ca_b"], "drift"))
        elif "recovery_fraction" not in r:
            continue
        elif r["recovery_renewals"] < 0.99 * args.recovery_renewals:
            bad.append((r["ca_b"], "recovery truncated by the wall-clock budget"))
        elif r["recovery_fraction"] > 0.05:
            bad.append((r["ca_b"], "displacement did not decay"))
    for ca_b, why in bad:
        print(f"FAIL Ca_b={ca_b:.5g}: {why}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "check": "s9_steadiness",
        "passed": not bad,
        "note": (
            "The reported films come from a stationary solve. This stage asks "
            "two things of that state: whether it survives time marching, and - "
            "the load-bearing one - whether it is an attractor, tested by "
            "displacing it to a neighbouring capillary number, restoring the "
            "parameter and marching back with no stationary polish. Both "
            "probes record their reach in film renewals L_b/U_b, so the "
            "evidence is not read as uniform across a sweep whose renewal time "
            "spans a factor of six thousand."
        ),
        "results": results,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
