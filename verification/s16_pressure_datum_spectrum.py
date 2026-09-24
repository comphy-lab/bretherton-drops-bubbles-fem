"""S16: does the pressure datum move the axisymmetric spectrum?

S7 shows ``fixation`` and ``integral`` agree on every reported base-state
observable to 2.6e-14. The claim that the datum has no dynamics, and therefore
cannot move eigenvalues, was unmeasured. Angular eigensolving is refused under
fixation, so the check is the cheap three-way comparison the handoff named:

1. the plain axisymmetric eigenproblem (``azimuthal_m=None``) under fixation;
2. the same call under integral;
3. ``azimuthal_m=0`` under integral, which is the pencil Campaign S used.

All three rightmost eigenvalues must agree. ``pressure_datum`` is an argument;
the class default stays ``fixation``.
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
    _eig_close,
    advective_rate,
    build_stability_problem,
    two_set_rightmost,
)


LEGS = (
    {
        "name": "fixation_m_none",
        "pressure_datum": "fixation",
        "azimuthal_stability": False,
        "azimuthal_k": None,
    },
    {
        "name": "integral_m_none",
        "pressure_datum": "integral",
        "azimuthal_stability": False,
        "azimuthal_k": None,
    },
    {
        "name": "integral_k0",
        "pressure_datum": "integral",
        "azimuthal_stability": True,
        "azimuthal_k": 0,
    },
)


def public_eigen(eigen: dict) -> dict:
    return {k: eigen[k] for k in eigen if not k.startswith("_")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca", type=float, default=0.05)
    parser.add_argument("--film-elements", type=float, default=2.0)
    parser.add_argument("--bulk-resolution", type=float, default=0.22)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output-root", type=Path, default=ROOT / "receipts")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "receipts" / "s16_pressure_datum_spectrum.json")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    legs = []
    agreed = []
    for spec in LEGS:
        outdir = args.output_root / f"s16_{spec['name']}_ca{args.ca:g}"
        problem = build_stability_problem(
            args.ca,
            film_elements=args.film_elements,
            bulk_resolution=args.bulk_resolution,
            output_dir=outdir,
            pressure_datum=spec["pressure_datum"],
            remesh=False,
            azimuthal_stability=spec["azimuthal_stability"],
        )
        problem.solve_to_steady()
        checks = problem.consistency()
        rate = advective_rate(checks.frame_speed, problem.geometry.r_tube)
        eigen = two_set_rightmost(
            problem, spec["azimuthal_k"], n=args.n, rate=rate,
            rtol=args.rtol, atol=args.atol,
        )
        row = {
            "name": spec["name"],
            "pressure_datum": spec["pressure_datum"],
            "azimuthal_stability": spec["azimuthal_stability"],
            "azimuthal_k": spec["azimuthal_k"],
            "ca_b": checks.frame_speed,
            "h_over_r_tube": checks.h_flat / problem.geometry.r_tube,
            "ndof": problem.ndof(),
            "eigen": public_eigen(eigen),
        }
        legs.append(row)
        agreed.append(eigen["agreed"])
        label = spec["name"]
        if eigen["agreed"] is None:
            print(f"{label}: FAIL two shift sets disagree", flush=True)
        else:
            lam = eigen["agreed"]
            print(
                f"{label}: ndof={problem.ndof()} Ca_b={checks.frame_speed:.9f} "
                f"rightmost {lam['real']:+.6g} {lam['imag']:+.6g}i",
                flush=True,
            )

    have_all = all(lam is not None for lam in agreed)
    spectra_agree = have_all and all(
        _eig_close(agreed[0], lam, rtol=args.rtol, atol=args.atol) for lam in agreed[1:]
    )
    base_ok = False
    if legs:
        ca_vals = [row["ca_b"] for row in legs]
        h_vals = [row["h_over_r_tube"] for row in legs]
        ca_spread = max(ca_vals) / min(ca_vals) - 1.0
        h_spread = max(h_vals) / min(h_vals) - 1.0
        base_ok = ca_spread <= 1e-10 and h_spread <= 1e-10
    else:
        ca_spread = h_spread = None

    passed = have_all and spectra_agree and base_ok
    if not have_all:
        print("FAIL a leg has no two-set consensus", flush=True)
    elif not spectra_agree:
        print("FAIL the three rightmost eigenvalues do not agree", flush=True)
    elif not base_ok:
        print("FAIL base-state Ca_b or H disagree across datums", flush=True)
    else:
        print("PASS three spectra agree; datum does not move the axisymmetric pencil",
              flush=True)

    payload = {
        "check": "s16_pressure_datum_spectrum",
        "passed": passed,
        "note": (
            "Three-way axisymmetric spectrum: azimuthal_m=None under fixation, "
            "azimuthal_m=None under integral, and azimuthal_m=0 under integral. "
            "pressure_datum is an argument; the class default is unchanged."
        ),
        "ca_in": args.ca,
        "n": args.n,
        "rtol": args.rtol,
        "atol": args.atol,
        "base_state_ca_b_spread": ca_spread,
        "base_state_h_spread": h_spread,
        "spectra_agree": spectra_agree,
        "legs": legs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
