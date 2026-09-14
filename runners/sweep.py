"""Sweep ``Ca_in`` and write the ``h(Ca)`` table with its checks.

Each capillary number is solved from its own Aussillous-Quere-seeded capsule
rather than continued from its neighbour. Continuation (``go_to_param``) is
available and would be the natural choice for an expensive solve, but each case
here costs a few seconds, and independent solves are worth more than the saving:
a continued sweep inherits its predecessor's state, so a systematic error in one
point propagates silently along the curve, whereas independent points can
disagree and thereby reveal it.
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

# The parent Basilisk campaign's swept values, so the curves are comparable
# point by point rather than only in shape.
PARENT_CA = (0.002, 0.005, 0.01, 0.02, 0.05)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=ROOT / "cases" / "bubble_ca0p05.json")
    parser.add_argument("--ca", type=float, nargs="+", default=list(PARENT_CA))
    parser.add_argument("--la", type=float, default=None)
    parser.add_argument("--film-elements", type=float, default=None)
    parser.add_argument("--bulk-resolution", type=float, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "sweep.json")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="directory to hold per-case solver output; defaults to "
                             "receipts/ inside the checkout, which is only appropriate "
                             "for local exploration. A dispatched run must place output "
                             "on a registered working volume, outside the source tree.")
    parser.add_argument("--pad-rear", type=float, default=None)
    parser.add_argument("--pad-front", type=float, default=None)
    args = parser.parse_args(argv)

    output_root = args.output_root or (ROOT / "receipts")
    output_root.mkdir(parents=True, exist_ok=True)

    case = load_case(args.case)
    solve = case["solve"]
    results = []
    for ca in sorted(args.ca, reverse=True):
        overrides: dict = {"physics": {"ca_in": ca}, "mesh": {}}
        if args.la is not None:
            overrides["physics"]["laplace"] = args.la
        if args.pad_rear is not None:
            overrides["physics"]["pad_rear"] = args.pad_rear
        if args.pad_front is not None:
            overrides["physics"]["pad_front"] = args.pad_front
        if args.film_elements is not None:
            overrides["mesh"]["film_elements"] = args.film_elements
        if args.bulk_resolution is not None:
            overrides["mesh"]["bulk_resolution"] = args.bulk_resolution
        try:
            results.append(
                solve_case(
                    build(case, overrides=overrides),
                    output_dir=output_root / f"sweep_ca{ca:g}",
                    relax_time=float(solve["relax_time"]),
                    start_step=float(solve["start_step"]),
                    temporal_error=float(solve["temporal_error"]),
                )
            )
            r = results[-1]
            print(
                f"Ca_in={ca:<8g} Ca_b={r['ca_b']:.6g} h/Rt={r['h_flat_over_r_tube']:.6f} "
                f"AQ={r['aussillous_quere_at_ca_b']:.6f} "
                f"dev={r['deviation_from_aussillous_quere']:+.3%} "
                f"stag={r['check_stagnant_film_error']:+.2e} "
                f"vol={r['check_void_volume_error']:+.1e} [{r['route']}]",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - a failed point is a result
            results.append({"ca_in": ca, "failed": f"{type(exc).__name__}: {exc}"})
            print(f"Ca_in={ca:<8g} FAILED {type(exc).__name__}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"sweep": "ca_in", "results": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if all("failed" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
