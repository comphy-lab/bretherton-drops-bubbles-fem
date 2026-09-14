"""S5: is the answer independent of the window length?

This is the claim the whole co-moving formulation rests on. In a circular tube
the fully developed solution is exactly Poiseuille, so imposing it at both ends
is the infinite-tube condition provided the cap disturbance has decayed by the
time it reaches them. "Provided" is the part that has to be measured.

The pads behind and ahead of the bubble are varied at fixed mesh size, and the
converged film thickness, bubble speed and void pressure are compared. Two
things make this a test rather than a demonstration:

* the pads are varied *independently*, because the two ends are not alike - the
  rear meniscus carries the capillary ripple and the film relaxation, the front
  sees only the Poiseuille approach - so a symmetric study could hide a rear-pad
  dependence behind an already-converged front;
* the Poiseuille and traction-free outlets are compared at each pad. If the
  window really is effectively infinite, the difference between the two outlet
  conditions must vanish faster than either converges on its own. If it does
  not, the exactness claim is wrong, and this is where it would show.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.run_case import build, load_case, solve_case

REPORTED = ("h_flat_over_r_tube", "ca_b", "bubble_pressure_excess")

OUTLET_COMPARABLE = ("h_flat_over_r_tube", "ca_b")
"""Quantities the two outlet conditions may legitimately be compared on.

The pressure excess is excluded, and the reason is a result rather than an
excuse. It is referenced to the mean outlet pressure, so comparing it between a
Poiseuille outlet and a traction-free one compares two different pressure data,
not two window lengths. The do-nothing outlet drives the whole traction vector
to zero, which Poiseuille violates through its radial component, and
``s0_poiseuille.py`` measures the resulting offset independently in the bare
tube: a fixed 0.066 at ``Ca = 0.05``, unchanged with window length.

The signature is unmistakable in the data. Across pads 1, 2 and 5 at
``Ca = 0.05`` the pressure gap sits at -1.37%, -1.66%, -1.65% - flat, where a
window effect would collapse - and 1.65% of the measured excess of about 3.84 is
0.063, the S0 deficit. So the cross-outlet pressure gap is reported with the
implied absolute offset attached, for comparison against S0, and the *window*
independence of the pressure is tested where it belongs: against pad, within a
single outlet condition, where it holds to 1.7e-6.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=ROOT / "cases" / "bubble_ca0p05.json")
    parser.add_argument("--ca", type=float, nargs="+", default=[0.05, 0.005])
    parser.add_argument("--pads", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--symmetric-only", action="store_true",
                        help="skip the one-at-a-time pad variation (not recommended: the "
                             "front and rear pads converge at very different rates, and a "
                             "symmetric-only study cannot separate them)")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="maximum accepted relative change from the reference pad")
    parser.add_argument("--outlet-gap-tol", type=float, default=1e-6,
                        help="maximum accepted outlet-condition gap at the reference pad")
    parser.add_argument("--outlets", nargs="+", default=["poiseuille", "traction_free"])
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "s5_window.json")
    args = parser.parse_args(argv)

    case = load_case(args.case)
    solve = case["solve"]
    reference_pad = max(args.pads)

    configs = []
    for ca in args.ca:
        for outlet in args.outlets:
            for pad in args.pads:
                configs.append((ca, outlet, pad, pad, "symmetric"))
            if not args.symmetric_only:
                for pad in args.pads:
                    if pad != reference_pad:
                        configs.append((ca, outlet, pad, reference_pad, "rear_only"))
                        configs.append((ca, outlet, reference_pad, pad, "front_only"))

    results = []
    for ca, outlet, pad_rear, pad_front, kind in configs:
        case_variant = json.loads(json.dumps(case))
        case_variant["boundaries"]["outlet"] = outlet
        overrides = {
            "physics": {"ca_in": ca, "pad_rear": pad_rear, "pad_front": pad_front},
            "mesh": {},
        }
        label = f"ca{ca:g}_{outlet}_r{pad_rear:g}_f{pad_front:g}"
        try:
            row = solve_case(
                build(case_variant, overrides=overrides),
                output_dir=ROOT / "receipts" / f"s5_{label}",
                relax_time=float(solve["relax_time"]),
                start_step=float(solve["start_step"]),
                temporal_error=float(solve["temporal_error"]),
            )
            row["kind"] = kind
            results.append(row)
            print(
                f"{label:34s} h/Rt={row['h_flat_over_r_tube']:.7f} Ca_b={row['ca_b']:.7f} "
                f"P_excess={row['bubble_pressure_excess']:.5f} "
                f"stag={row['check_stagnant_film_error']:+.1e}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"label": label, "ca_in": ca, "outlet": outlet,
                            "pad_rear": pad_rear, "pad_front": pad_front, "kind": kind,
                            "failed": f"{type(exc).__name__}: {exc}"})
            print(f"{label:34s} FAILED {type(exc).__name__}", flush=True)

    # Convergence: fractional change of each reported quantity relative to the
    # largest symmetric pad, at the same Ca and outlet.
    convergence = []
    for ca in args.ca:
        for outlet in args.outlets:
            ref = next(
                (r for r in results if r.get("kind") == "symmetric"
                 and r.get("ca_in") == ca and r.get("outlet") == outlet
                 and r.get("pad_rear") == reference_pad and "failed" not in r),
                None,
            )
            if ref is None:
                continue
            for r in results:
                if "failed" in r or r.get("ca_in") != ca or r.get("outlet") != outlet:
                    continue
                convergence.append(
                    {
                        "ca_in": ca, "outlet": outlet, "kind": r["kind"],
                        "pad_rear": r["pad_rear"], "pad_front": r["pad_front"],
                        **{f"{k}_rel_change": r[k] / ref[k] - 1.0 for k in REPORTED},
                    }
                )

    # Outlet-condition difference at matched pads: the direct test of exactness.
    outlet_gap = []
    if {"poiseuille", "traction_free"} <= set(args.outlets):
        for ca in args.ca:
            for pad in args.pads:
                def find(o, ca=ca, pad=pad):
                    # Loop variables bound as defaults: the closure is called in
                    # the same iteration today, but late binding here would be a
                    # silent mis-pairing of outlets if that ever changed.
                    return next(
                        (r for r in results if "failed" not in r and r.get("ca_in") == ca
                         and r.get("outlet") == o and r.get("kind") == "symmetric"
                         and r.get("pad_rear") == pad), None)
                a, b = find("poiseuille"), find("traction_free")
                if a and b:
                    entry = {"ca_in": ca, "pad": pad,
                             **{f"{k}_rel_gap": b[k] / a[k] - 1.0 for k in REPORTED}}
                    # Absolute pressure offset implied by the gap, so it can be
                    # compared directly with the S0 traction-free deficit.
                    entry["pressure_abs_offset"] = (
                        b["bubble_pressure_excess"] - a["bubble_pressure_excess"]
                    )
                    outlet_gap.append(entry)

    # The stage must be able to fail on what it measures, not merely on whether
    # the solver crashed. Three conditions, each of which a genuinely
    # window-dependent result would violate:
    #   - every configuration solved;
    #   - at the second-largest pad and above, every reported quantity sits
    #     within `--tol` of its value at the reference pad;
    #   - at the reference pad the two outlet conditions agree to
    #     `--outlet-gap-tol` on the flow observables, which is the direct test
    #     that the window is effectively infinite rather than merely
    #     self-consistent. Pressure is excluded from this one comparison for the
    #     reason given at OUTLET_COMPARABLE; its window independence is still
    #     tested, by the pad criterion above.
    pads_sorted = sorted(args.pads)
    converged_pads = set(pads_sorted[1:]) if len(pads_sorted) > 1 else set(pads_sorted)
    tolerance_failures = [
        c for c in convergence
        if c["pad_rear"] in converged_pads and c["pad_front"] in converged_pads
        and any(abs(c[f"{k}_rel_change"]) > args.tol for k in REPORTED)
    ]
    gap_failures = [
        g for g in outlet_gap
        if g["pad"] == reference_pad
        and any(abs(g[f"{k}_rel_gap"]) > args.outlet_gap_tol for k in OUTLET_COMPARABLE)
    ]
    # Both criteria can be satisfied vacuously, and a stage that passes without
    # comparing anything is worse than no stage. Require the comparisons to have
    # actually happened: at least one non-reference symmetric pad for every
    # (Ca, outlet), and a matched two-outlet comparison at the reference pad.
    compared_pads = {
        (c["ca_in"], c["outlet"])
        for c in convergence
        if c["kind"] == "symmetric" and c["pad_rear"] != reference_pad
    }
    missing_pad_comparison = [
        {"ca_in": ca, "outlet": outlet}
        for ca in args.ca for outlet in args.outlets
        if (ca, outlet) not in compared_pads
    ]
    gap_required = {"poiseuille", "traction_free"} <= set(args.outlets)
    gap_compared = [g["ca_in"] for g in outlet_gap if g["pad"] == reference_pad]
    missing_gap_comparison = gap_required and set(gap_compared) != set(args.ca)

    payload = {"stage": "S5", "reference_pad": reference_pad, "results": results,
               "convergence": convergence, "outlet_gap": outlet_gap,
               "tolerance": args.tol, "outlet_gap_tolerance": args.outlet_gap_tol,
               "tolerance_failures": tolerance_failures, "gap_failures": gap_failures,
               "missing_pad_comparison": missing_pad_comparison,
               "outlet_exactness_compared": gap_required and not missing_gap_comparison,
               "passed": all("failed" not in r for r in results)
               and not tolerance_failures and not gap_failures
               and not missing_pad_comparison
               and gap_required and not missing_gap_comparison}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
