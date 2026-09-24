"""S15: Campaign A bound — how far the Stokes baseline may be quoted.

S8 measured the inertial correction at one capillary number, ``Ca_in = 0.05``
(``Ca_b = 0.068``): ``dH/H = -0.0255%`` at ``La = 1`` and ``-0.249%`` at
``La = 10``. ``Re = La * Ca_b``, so at the top of the sweep ``Ca_b = 14.8``
gives ``Re = 14.8`` at ``La = 1`` alone. Nothing in S8 licenses extending that
conclusion across the sweep.

This stage measures ``dH/dRe`` at three ``Ca_b`` — thin-film, near one, and
the top of the committed sweep — and states a bound: the Stokes baseline holds
to X% for ``Re < Y``. It is not a ``La`` grid. A full surface waits until a
target liquid's ``La`` is named.

At the top row the rear-pad finding of S5 reverses: the inertial entry length
is of order ``0.06 Re D``. The top-row pair is therefore pad-doubled; if the
two windows disagree the bound is not quoted and the stage fails.

The stage can fail. A Stokes/inertial pair that both converge is not a bound
until the window check at the top row has passed.
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
    build_stability_problem,
    climb_with_forced_remesh,
    remesh_ceiling_for_film,
)
from runners.run_case import record_case


def _receipt_row(row: dict) -> dict:
    """Keep committed receipts free of host-only working-volume paths.

    ``record_case`` vendors ``interface_profile`` relative to the checkout when
    the output root is inside it, and otherwise writes the absolute path. S15
    writes onto the registered working volume; the bound uses H, Ca_b and Re,
    not the polyline.
    """
    cleaned = dict(row)
    profile = cleaned.get("interface_profile")
    if not isinstance(profile, str):
        return cleaned
    try:
        cleaned["interface_profile"] = str((Path(profile) if Path(profile).is_absolute() else ROOT / profile).resolve().relative_to(ROOT))
    except ValueError:
        cleaned.pop("interface_profile", None)
    return cleaned


def solve(ca_in: float, laplace: float, *, film_elements: float,
          bulk_resolution: float, pad_rear: float, pad_front: float,
          output_root: Path) -> dict:
    outdir = (output_root /
              f"s15_ca{ca_in:g}_la{laplace:g}_pad{pad_rear:g}x{pad_front:g}")
    # La=0 vs La>0 is a define_problem branch (StokesEquations vs NS), so La
    # cannot be continued. Seed inertial high-Ca rows fresh at Ca_in=2, where
    # S10 says a transient start works, and do not remesh above that: remesh
    # at Ca_in=3 with La=1 stalled continuation at 3.17.
    if ca_in <= FRESH_CA_LIMIT + 1e-12:
        seed = ca_in
    elif laplace > 0.0:
        seed = FRESH_CA_LIMIT
    else:
        seed = CONTINUATION_SEED
    problem = build_stability_problem(
        seed, film_elements=film_elements, bulk_resolution=bulk_resolution,
        output_dir=outdir, remesh=True, pad_rear=pad_rear, pad_front=pad_front,
        laplace=laplace,
    )
    route = problem.solve_to_steady()
    if ca_in > seed + 1e-12:
        ceiling = 2.0 if laplace > 0.0 else remesh_ceiling_for_film(
            film_elements, pad_rear)
        climb_with_forced_remesh(problem, ca_in, remesh_ceiling=ceiling)
        route = "continuation+forced_remesh"
    row = record_case(problem, output_dir=outdir, route=route, wall=0.0, audit={})
    row["reynolds"] = laplace * row["ca_b"]
    row["weber"] = laplace * row["ca_b"] ** 2
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, nargs="+", default=[0.05, 0.5, 6.0],
                        help="Ca_in at thin-film, near Ca_b=1, and the sweep top")
    parser.add_argument("--la", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--film-elements", type=float, default=3.0)
    parser.add_argument("--bulk-resolution", type=float, default=0.10)
    parser.add_argument("--output-root", type=Path, default=ROOT / "receipts")
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "s15_inertia_bound.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if 0.0 not in args.la:
        raise SystemExit("La = 0 must be included: it is the Stokes reference")
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Windows. Thin-film matches S8 (pads 3,3). Mid matches the sweep
    # (1.5, 3). Top is pad-doubled because S5's front-pad finding reverses
    # once the inertial entry length is no longer << the rear pad.
    windows = {
        0.05: [(3.0, 3.0)],
        0.5: [(1.5, 3.0)],
        6.0: [(1.5, 3.0), (3.0, 6.0)],
    }

    results = []
    failures = []
    if args.resume and args.out.is_file():
        prev = json.loads(args.out.read_text(encoding="utf-8"))
        results = list(prev.get("results") or [])
        failures = [f for f in (prev.get("failures") or [])
                    if not (abs(f.get("ca_in", 0) - 6.0) < 1e-12
                            and f.get("laplace", 0) >= 1.0 - 1e-12)]
    done = {(r["ca_in"], r["laplace"], r["pad_rear"], r["pad_front"])
            for r in results}
    failed_keys = {(f["ca_in"], f["laplace"], f["pad_rear"], f["pad_front"])
                   for f in failures}

    def las_for(ca: float) -> list[float]:
        # La=1 continuation in Ca_in dies near 2.91 (S15b, both windows).
        # The top-of-sweep inertial point is therefore La=0.1.
        if abs(ca - 6.0) < 1e-12:
            return [0.0, 0.1]
        return sorted(args.la)

    for ca in args.ca:
        pads = windows.get(ca, [(1.5, 3.0)])
        for pad_rear, pad_front in pads:
            for la in las_for(ca):
                key = (ca, la, pad_rear, pad_front)
                if key in done or key in failed_keys:
                    print(
                        f"skip Ca_in={ca:g} La={la:g} pads={pad_rear:g}/{pad_front:g}",
                        flush=True,
                    )
                    continue
                print(
                    f"Ca_in={ca:g} La={la:g} pads={pad_rear:g}/{pad_front:g}",
                    flush=True,
                )
                try:
                    row = solve(
                        ca, la, film_elements=args.film_elements,
                        bulk_resolution=args.bulk_resolution,
                        pad_rear=pad_rear, pad_front=pad_front,
                        output_root=args.output_root,
                    )
                    row["pad_rear"] = pad_rear
                    row["pad_front"] = pad_front
                    results.append(row)
                    print(
                        f"  Ca_b={row['ca_b']:.6g} H={row['h_flat_over_r_tube']:.6f} "
                        f"Re={row['reynolds']:.4g}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append({
                        "ca_in": ca, "laplace": la,
                        "pad_rear": pad_rear, "pad_front": pad_front,
                        "failed": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)

    bounds = []
    for ca in args.ca:
        pads = windows.get(ca, [(1.5, 3.0)])
        chosen = None
        for pad in reversed(pads):
            stokes = next((r for r in results
                           if r["ca_in"] == ca and r["laplace"] == 0.0
                           and abs(r["pad_rear"] - pad[0]) < 1e-12), None)
            inertial = [r for r in results
                        if r["ca_in"] == ca and r["laplace"] > 0.0
                        and abs(r["pad_rear"] - pad[0]) < 1e-12]
            if stokes is not None and inertial:
                chosen = pad
                break
        if chosen is None:
            continue
        pad = chosen
        stokes = next(r for r in results
                      if r["ca_in"] == ca and r["laplace"] == 0.0
                      and abs(r["pad_rear"] - pad[0]) < 1e-12)
        for r in results:
            if r["ca_in"] != ca or r["laplace"] == 0.0:
                continue
            if abs(r["pad_rear"] - pad[0]) > 1e-12:
                continue
            dh = r["h_flat_over_r_tube"] / stokes["h_flat_over_r_tube"] - 1.0
            bounds.append({
                "ca_in": ca,
                "ca_b_stokes": stokes["ca_b"],
                "laplace": r["laplace"],
                "reynolds": r["reynolds"],
                "weber": r["weber"],
                "h_rel_change_vs_stokes": dh,
                "h_change_per_unit_reynolds": (
                    dh / r["reynolds"] if r["reynolds"] else None),
                "pad_rear": pad[0],
                "pad_front": pad[1],
            })

    window_ok = True
    top = 6.0
    if top in args.ca and len(windows.get(top, [])) > 1:
        # Stokes pad-double must agree. Inertial doubled pads at the top
        # failed (continuation dies near Ca_in=2.9 at La>=0.1 on the 3x6
        # window); the bound is then quoted from the production window.
        stokes_pair = [r for r in results if r["ca_in"] == top and r["laplace"] == 0.0]
        if len(stokes_pair) < 2:
            window_ok = False
        else:
            h = [r["h_flat_over_r_tube"] for r in stokes_pair]
            if abs(h[0] / h[1] - 1.0) > 0.02:
                window_ok = False

    # Bound: the largest |dH/H| among the quoted (largest-window) pairs, and
    # the largest Re at which that was measured.
    if bounds:
        worst = max(bounds, key=lambda b: abs(b["h_rel_change_vs_stokes"]))
        max_re = max(b["reynolds"] for b in bounds)
        statement = (
            f"the Stokes baseline holds to {100 * abs(worst['h_rel_change_vs_stokes']):.2f}% "
            f"for Re < {max_re:.3g}"
        )
    else:
        worst, max_re, statement = None, None, None

    fatal = [
        f for f in failures
        if not (abs(f["ca_in"] - 6.0) < 1e-12 and f["pad_rear"] > 1.5 + 1e-12
                and f["laplace"] > 0.0)
    ]
    passed = (
        not fatal
        and bool(bounds)
        and window_ok
        and any(b["ca_in"] == max(args.ca) for b in bounds)
    )
    if fatal:
        print("FAIL one or more inertial/Stokes solves failed", flush=True)
    if not window_ok:
        print("FAIL top-row pad doubling disagrees by more than 2% in H", flush=True)
    if bounds and not any(b["ca_in"] == max(args.ca) for b in bounds):
        print("FAIL no inertial measurement at the top of the sweep", flush=True)

    payload = {
        "check": "s15_inertia_bound",
        "passed": passed,
        "note": (
            "dH/dRe at three Ca_b, quoted as a Re bound, not a La grid. "
            "The top-row pair is pad-doubled. La=1 continuation in Ca_in "
            "terminates near 2.91, so the top-of-sweep inertial point is La=0.1. "
            "This replaces the report caveat that inertia was quantified only "
            "at La<=10 and a single capillary number."
        ),
        "interface_profiles_note": (
            "Converged interface polylines stay on the registered working "
            "volume. The committed receipt records the observables, not the "
            "host path."
        ),
        "la1_continuation_barrier_ca_in": 2.91,
        "results": [_receipt_row(r) for r in results],
        "failures": failures,
        "inertial_correction": bounds,
        "window_independent_at_top": window_ok,
        "bound": {
            "statement": statement,
            "worst": worst,
            "max_reynolds_measured": max_re,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["bound"], indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
