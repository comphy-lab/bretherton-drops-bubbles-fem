"""Sweep ``Ca_in`` and write the ``h(Ca)`` table with its checks.

Two modes, and the default is the stricter one.

By default each capillary number is solved from its own Aussillous-Quere-seeded
capsule, independently of its neighbours. That costs a few seconds per case and
is worth more than the saving continuation would give: a continued sweep
inherits its predecessor's state, so a systematic error in one point propagates
silently along the curve, whereas independent points can disagree and thereby
reveal it.

``--continuation`` solves the lowest capillary number fresh and then walks up
through the global parameter, because independence stops being affordable
before it stops being possible. Seeded from the self-consistent bubble speed, a
fresh transient start still reaches ``Ca_in = 2`` - measured, and it agrees with
the continued answer there to 8e-5 in the film - but it costs a full transient
per point and has been observed to fail by ``Ca_in = 4``, while a continued step
costs under a second. Read the resulting curve knowing that its points are no
longer independent evidence of one another; ``verification/s10_route.py`` is
what checks that the two routes agree where both work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.run_case import build, continue_case, load_case, solve_case

# The parent Basilisk campaign's swept values, so the curves are comparable
# point by point rather than only in shape.
PARENT_CA = (0.002, 0.005, 0.01, 0.02, 0.05)


def write_results(out: Path, results: list[dict], *, merge: bool) -> None:
    """Write the receipt, refusing to silently discard rows this run did not solve.

    The two modes produce disjoint halves of the same curve, and both default to
    the same path. Writing unconditionally therefore means a continuation run
    replaces a 25-row committed receipt with its own 9 rows, leaving a file that
    is internally consistent, obviously smaller, and easy not to notice. Merge
    on ``ca_in`` when asked; otherwise stop and say which rows would be lost.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"sweep": "ca_in"}
    rows = sorted(results, key=lambda r: r["ca_in"])
    if out.is_file():
        existing = json.loads(out.read_text(encoding="utf-8"))
        kept = {r["ca_in"]: r for r in existing.get("results", [])}
        incoming = {r["ca_in"] for r in results}
        lost = sorted(set(kept) - incoming)
        if lost and not merge:
            raise SystemExit(
                f"{out} holds {len(lost)} row(s) this run did not solve "
                f"({', '.join(f'{c:g}' for c in lost)}). Pass --merge to combine "
                "them with these results, or --out to write elsewhere."
            )
        # Carry the existing top-level keys across either way. They hold the
        # provenance, route and extraction notes, which are written by hand and
        # are not recoverable from a re-run; dropping them on the replace path
        # would quietly strip a receipt of everything that explains it.
        payload.update({k: v for k, v in existing.items() if k != "results"})
        if merge:
            kept.update({r["ca_in"]: r for r in results})
            rows = [kept[c] for c in sorted(kept)]
    payload["results"] = rows
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")


def run_continuation(case: dict, args, output_root: Path, solve: dict) -> list[dict]:
    """Ascending ladder: one fresh solve, then continuation for the rest.

    Ascending rather than descending because the cheap, well-conditioned end is
    the low-Ca one for a fresh start; continuation then walks up into the range
    where a fresh start fails.
    """
    order = sorted(args.ca)
    overrides: dict = {"physics": {"ca_in": order[0]}, "mesh": {}}
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

    outdir = output_root / f"cont_ca{order[0]:g}"
    try:
        # build() is inside the try as well: it meshes and assembles, so a bad
        # case document or a geometry the mesher rejects raises here rather than
        # in solve_case, and leaving it outside would lose the whole run's
        # receipt to exactly the failure this handler exists to record.
        problem = build(case, overrides=overrides,
                        seed_from_bubble_speed=args.seed_from_bubble_speed)
        row = solve_case(problem, output_dir=outdir,
                         relax_time=float(solve["relax_time"]),
                         start_step=float(solve["start_step"]),
                         temporal_error=float(solve["temporal_error"]))
    except Exception as exc:  # noqa: BLE001 - the seed failing is a result too
        # Same contract as every other point: record it and let the receipt be
        # written. An unhandled seed failure would lose the whole run's output.
        print(f"Ca_in={order[0]:<8g} FAILED {type(exc).__name__}; seed solve, "
              "ladder cannot start", flush=True)
        return [{"ca_in": order[0], "failed": f"{type(exc).__name__}: {exc}"}]
    results = [row]
    report(row)

    for ca in order[1:]:
        try:
            row = continue_case(problem, output_dir=output_root / f"cont_ca{ca:g}",
                                target_ca=ca)
            results.append(row)
            report(row)
        except Exception as exc:  # noqa: BLE001 - a failed step ends the ladder
            results.append({"ca_in": ca, "failed": f"{type(exc).__name__}: {exc}"})
            print(f"Ca_in={ca:<8g} FAILED {type(exc).__name__}; ladder stops here",
                  flush=True)
            break
    return results


def report(r: dict) -> None:
    print(
        f"Ca_in={r['ca_in']:<8g} Ca_b={r['ca_b']:.6g} h/Rt={r['h_flat_over_r_tube']:.6f} "
        f"AQ={r['aussillous_quere_at_ca_b']:.6f} "
        f"dev={r['deviation_from_aussillous_quere']:+.3%} "
        f"stag={r['check_stagnant_film_error']:+.2e} "
        f"vol={r['check_void_volume_error']:+.1e} [{r['route']}]",
        flush=True,
    )


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
    parser.add_argument("--merge", action="store_true",
                        help="merge these results into an existing receipt on "
                             "ca_in rather than replacing it. The committed sweep "
                             "is built this way: the independent ladder first, "
                             "then the continuation ladder merged into it.")
    parser.add_argument("--continuation", action="store_true",
                        help="solve the lowest Ca fresh, then continue upward in the "
                             "Ca_in global parameter. Fresh starts still work at "
                             "Ca_in = 2 when seeded from the bubble speed, and fail "
                             "by Ca_in = 4; use this above about 1 for cost, since "
                             "each continued step takes under a second.")
    parser.add_argument("--seed-from-bubble-speed", action="store_true",
                        help="seed the capsule from the self-consistent Ca_b rather "
                             "than from Ca_in; matters once the two differ by more "
                             "than a few per cent")
    parser.add_argument("--pad-rear", type=float, default=None)
    parser.add_argument("--pad-front", type=float, default=None)
    args = parser.parse_args(argv)

    output_root = args.output_root or (ROOT / "receipts")
    output_root.mkdir(parents=True, exist_ok=True)

    case = load_case(args.case)
    solve = case["solve"]
    results = []
    if args.continuation:
        results = run_continuation(case, args, output_root, solve)
        write_results(args.out, results, merge=args.merge)
        return 0 if all("failed" not in r for r in results) else 1

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
                    build(case, overrides=overrides,
                          seed_from_bubble_speed=args.seed_from_bubble_speed),
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

    write_results(args.out, results, merge=args.merge)
    return 0 if all("failed" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
