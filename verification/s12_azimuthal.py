"""S12: can we ask whether the axisymmetric state is the state a bubble adopts?

Every result in this repository is an axisymmetric steady solution. Nothing in
S0 to S11 asks whether a physical bubble would actually adopt it: S9 displaces
the state and marches it back, but a time march on an axisymmetric mesh can only
see axisymmetric perturbations. A bubble is free to break that symmetry, and the
buoyant literature reports it doing so through an ``|m| = 1`` mode.

pyoomph can answer this without a three-dimensional calculation. Perturbations
are expanded as ``eps U^(m)(r,z) exp(i m phi + lambda_m t)``, which turns a
three-dimensional stability problem into a two-dimensional eigenproblem on the
mesh we already have. This stage establishes that the machinery works *on this
class of problem* and that its answer is numerically converged, so that a later
campaign can quote a critical capillary number.

**The trap this stage exists to record.** pyoomph refuses angular eigensolving
when the pressure datum is a fixation:

    RuntimeError: Do not use pressure_fixation with angular eigensolving.
    Use [Navier]StokesEquation(...).with_pressure_integral_constraint(problem)
    instead...

``pressure_datum="fixation"`` is this solver's default and is what the committed
sweep used, so the whole approach appears blocked until one reads the message.
It is not: ``pressure_datum="integral"`` is exactly the alternative named, and
``verification/s7_pressure_datum.py`` already measured the two mechanisms
agreeing on every reported observable to 2.6e-14. The switch therefore costs
nothing, which is a fact worth having had in advance rather than assumed.

**What this stage does and does not claim.** It does not locate a critical
capillary number; that is a campaign, not a check. It claims three things, each
falsifiable: the eigensolver runs for ``m = 0, 1, 2`` on a moving mesh with a
free surface; some tested eigenvalue count resolves the rightmost eigenvalue
independently of the spectral shift, so it is a property of the operator rather
than of the solver call; and a bubble deep in the Bretherton regime is stable to
all three modes, which it had better be.

Measured, and not guessable in advance: ``m = 0`` needs at least **8**
eigenvalues. At four, one shift out of three converges to a cluster that does
not contain the rightmost and reports ``-0.174`` where the answer is ``-0.123``
-- a 41% error, silently, with no warning from the solver. ``m = 1`` and
``m = 2`` are already converged at four. Any campaign must request at least
eight and re-run this calibration if the mesh or the state changes.
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


def build(ca_in: float, film_elements: float, bulk: float, output_root: Path):
    geom = CapsuleGeometry.for_capillary_number(
        ca_in, pad_rear=1.5, pad_front=3.0, seed_from_bubble_speed=True)
    problem = VoidBubbleProblem(
        geometry=geom, ca_in=ca_in, laplace=0.0, film_elements=film_elements,
        bulk_resolution=bulk, closure="centroid", outlet="poiseuille",
        # Not the default. See the module docstring: angular eigensolving is
        # refused with a pressure fixation, and S7 shows the swap is free.
        pressure_datum="integral", remesh=False,
    )
    problem.set_output_directory(str(output_root / f"s12_ca{ca_in:g}"))
    # Must precede initialise(): it changes code generation.
    problem.setup_for_stability_analysis(azimuthal_stability=True)
    problem.initialise()
    problem.audit_constraints()
    problem.solve_to_steady()
    return problem, geom


def rightmost(problem, m: int, n: int, shift: float) -> complex:
    values, _ = problem.solve_eigenproblem(n, azimuthal_m=m, shift=shift)
    return max(values, key=lambda z: z.real)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, default=0.05,
                        help="Ca_in of the reference state; the default sits deep "
                             "in the Bretherton regime, where stability is not in "
                             "doubt and a positive growth rate would be a defect")
    parser.add_argument("--modes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--counts", type=int, nargs="+", default=[4, 8, 16],
                        help="eigenvalue counts, for the convergence check")
    parser.add_argument("--shifts", type=float, nargs="+", default=[0.0, -0.05, 0.05],
                        help="spectral shifts, for the convergence check")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="agreement required of the rightmost eigenvalue across "
                             "eigenvalue counts and shifts")
    parser.add_argument("--film-elements", type=float, default=2.0)
    parser.add_argument("--bulk-resolution", type=float, default=0.22)
    parser.add_argument("--output-root", type=Path, default=ROOT / "receipts")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "receipts" / "s12_azimuthal.json")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    problem, geom = build(args.ca, args.film_elements, args.bulk_resolution,
                          args.output_root)
    checks = problem.consistency()
    print(f"base state: ndof={problem.ndof()}, Ca_b={checks.frame_speed:.6f}, "
          f"h/R_t={checks.h_flat / geom.r_tube:.6f}", flush=True)

    rows, failures = [], []
    for m in args.modes:
        samples = []
        for n in args.counts:
            for shift in args.shifts:
                try:
                    lam = rightmost(problem, m, n, shift)
                except Exception as exc:  # noqa: BLE001 - a refusal is the result
                    failures.append(f"m={m} n={n} shift={shift}: "
                                    f"{type(exc).__name__}: {exc}")
                    print(f"  m={m} n={n} shift={shift:+.3g} FAILED "
                          f"{type(exc).__name__}", flush=True)
                    continue
                samples.append({"n": n, "shift": shift,
                                "real": lam.real, "imag": lam.imag})
        if not samples:
            continue
        # A shift-invert Krylov solve returns eigenvalues near the shift. Asked
        # for too few, it can converge to a cluster that does not contain the
        # rightmost one and report a more negative value with no warning. The
        # consensus is therefore taken from the largest count tested, and the
        # useful output is the smallest count at which every shift agrees with
        # it: that is the calibration a campaign needs, and it is not
        # guessable in advance.
        best_n = max(x["n"] for x in samples)
        consensus = max(x["real"] for x in samples if x["n"] == best_n)
        scale = max(abs(consensus), 1e-300)
        by_count = {}
        for n in sorted({x["n"] for x in samples}):
            at_n = [x["real"] for x in samples if x["n"] == n]
            worst = max(abs(v - consensus) / scale for v in at_n)
            by_count[n] = worst
        reliable = [n for n, err in by_count.items() if err <= args.rtol]
        row = {"m": m,
               "rightmost_real": consensus,
               "rightmost_imag": next(x["imag"] for x in samples
                                      if x["n"] == best_n and x["real"] == consensus),
               "consensus_from_count": best_n,
               "relative_error_by_count": by_count,
               "min_reliable_count": min(reliable) if reliable else None,
               "samples": samples}
        rows.append(row)
        detail = ", ".join(f"n={n}: {err:.1e}" for n, err in by_count.items())
        print(f"  m={m}: rightmost Re(lambda) = {consensus:+.6g}  [{detail}]  "
              f"min reliable count = {row['min_reliable_count']}", flush=True)

    # Three falsifiable claims, reported separately so a failure says which.
    # A mode that raised on some calls but succeeded on others would otherwise
    # produce a row and pass, which is the vacuous pass this suite exists to
    # avoid: a refusal is a result and must fail the stage.
    ran = len(rows) == len(args.modes) and not failures
    converged = all(r["min_reliable_count"] is not None for r in rows)
    stable = all(r["rightmost_real"] < 0.0 for r in rows)
    if failures:
        print(f"FAIL {len(failures)} eigensolve call(s) raised; a refusal is a "
              "result and is not tolerated", flush=True)
    elif len(rows) != len(args.modes):
        print("FAIL the eigensolver did not run for every requested mode", flush=True)
    if rows and not converged:
        print("FAIL no tested eigenvalue count resolves the rightmost eigenvalue "
              "independently of the shift; raise --counts", flush=True)
    if rows and not stable:
        print("FAIL a mode is unstable deep in the Bretherton regime, where the "
              "physical state is certainly stable", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "check": "s12_azimuthal",
        "passed": ran and converged and stable,
        "eigensolver_ran_for_every_mode": ran,
        "rightmost_resolved_at_some_tested_count": converged,
        "required_eigenvalue_count": (max(r["min_reliable_count"] for r in rows)
                                      if converged and rows else None),
        "reference_state_stable": stable,
        "note": (
            "Establishes that azimuthal linear stability is available for this "
            "moving-mesh free-surface formulation and that its rightmost "
            "eigenvalue is a property of the operator rather than of the solver "
            "call, and calibrates how many eigenvalues must be requested for "
            "that to hold. It does NOT locate a critical capillary number. pressure_datum must be 'integral': "
            "pyoomph refuses angular eigensolving with a pressure fixation, and "
            "s7_pressure_datum.py shows the two mechanisms agree to 2.6e-14."
        ),
        "ca_in": args.ca,
        "ca_b": checks.frame_speed,
        "h_over_r_tube": checks.h_flat / geom.r_tube,
        "ndof": problem.ndof(),
        "rtol": args.rtol,
        "failures": failures,
        "results": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0 if (ran and converged and stable) else 1


if __name__ == "__main__":
    raise SystemExit(main())
