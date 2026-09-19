"""S17: why does La=1 continuation die near Ca_in=2.91?

S15b's log already shows automatic RemeshWhen during go_to_param, element
expansion ~1.8, and the continuation step collapsing to 1e-8 at
Ca_in=2.9085. A fresh seed at Ca_in=2, La=1 with remesh off dies in the
transient start (dt < 1e-12). This stage therefore seeds with remesh on,
then sets do_call_remeshing_when_necessary=False and climbs. If that
passes 2.91, the S15 barrier is remesh wreck on the climb. If the step
still collapses, the barrier survives without RemeshWhen.

Do not coarsen the mesh.
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

from eigen_helpers import build_stability_problem
from runners.run_case import record_case


def _public_row(row: dict) -> dict:
    cleaned = dict(row)
    profile = cleaned.get("interface_profile")
    if isinstance(profile, str):
        try:
            cleaned["interface_profile"] = str(Path(profile).resolve().relative_to(ROOT))
        except ValueError:
            cleaned.pop("interface_profile", None)
    return cleaned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=float, default=2.0)
    parser.add_argument("--target", type=float, default=4.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--la", type=float, default=1.0)
    parser.add_argument("--film-elements", type=float, default=3.0)
    parser.add_argument("--bulk-resolution", type=float, default=0.10)
    parser.add_argument("--pad-rear", type=float, default=1.5)
    parser.add_argument("--pad-front", type=float, default=3.0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "receipts" / "s17_la1_barrier.json")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    outdir = (args.output_root /
              f"s17_la{args.la:g}_pad{args.pad_rear:g}x{args.pad_front:g}")
    problem = build_stability_problem(
        args.seed, film_elements=args.film_elements,
        bulk_resolution=args.bulk_resolution, output_dir=outdir,
        remesh=True, pad_rear=args.pad_rear, pad_front=args.pad_front,
        laplace=args.la, pressure_datum="integral",
        azimuthal_stability=False,
    )
    route = problem.solve_to_steady()
    seed_row = _public_row(record_case(
        problem, output_dir=outdir, route=route, wall=0.0, audit={}))
    problem.do_call_remeshing_when_necessary = False
    print(
        f"seed Ca_in={args.seed:g} La={args.la:g} ndof={problem.ndof()} "
        f"Ca_b={seed_row['ca_b']:.6f} remesh_on_seed=True "
        f"do_call_remeshing_when_necessary=False",
        flush=True,
    )

    stations = []
    last_ok = args.seed
    failure = None
    failure_ca_in = None
    ca = args.seed
    while ca + args.step / 2.0 < args.target:
        nxt = min(round(ca + args.step, 10), args.target)
        try:
            problem.go_to_param(Ca_in=nxt, max_step=args.step)
            row = _public_row(record_case(
                problem, output_dir=outdir, route="continuation_no_remesh",
                wall=0.0, audit={}))
            stations.append({
                "ca_in": problem.current_ca_in(),
                "ca_b": row["ca_b"],
                "h_over_r_tube": row["h_flat_over_r_tube"],
                "ndof": problem.ndof(),
            })
            last_ok = problem.current_ca_in()
            print(
                f"  ok Ca_in={last_ok:.4g} Ca_b={row['ca_b']:.6f} "
                f"H={row['h_flat_over_r_tube']:.6f}",
                flush=True,
            )
            ca = nxt
        except Exception as exc:  # noqa: BLE001
            failure = {
                "target_ca_in": nxt,
                "last_ok_ca_in": last_ok,
                "failed": f"{type(exc).__name__}: {exc}",
            }
            # The live parameter at death, which is where the branch actually
            # ends; the target is only where the step was aiming.
            failure_ca_in = problem.current_ca_in()
            print(f"  FAIL at Ca_in->{nxt:g} after {last_ok:.6g}: {exc}",
                  flush=True)
            break

    passed_barrier = last_ok > 2.91 + 1e-6 and failure is None
    classified = (
        "remesh_wreck" if last_ok > 2.91 + 1e-6
        else "still_blocked_without_remesh"
    )
    payload = {
        "check": "s17_la1_barrier",
        "passed": failure is None or last_ok > args.seed + 1e-12,
        "classification": classified,
        "passed_s15_barrier_without_remesh": last_ok > 2.91 + 1e-6,
        "note": (
            "Seed at Ca_in=2 La=1 with remesh on, then climb with "
            "do_call_remeshing_when_necessary=False. If it passes 2.91 the "
            "S15 barrier is remesh wreck on the climb; if not, the barrier "
            "survives without RemeshWhen."
        ),
        "remesh_on_seed": True,
        "do_call_remeshing_when_necessary_on_climb": False,
        "seed": seed_row,
        "stations": stations,
        "last_ok_ca_in": last_ok,
        "failure": failure,
        "failure_ca_in": failure_ca_in,
        "ndof_constant_on_climb": len({s["ndof"] for s in stations}) <= 1,
        "passed_barrier": passed_barrier,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out} classification={classified}", flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
