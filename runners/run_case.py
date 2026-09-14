"""Solve one portable case to steady state and record the result with its checks.

Every number this writes is accompanied by the checks that could have falsified
it: the void-volume error, the inlet/outlet flux mismatch, the stagnant-film
identity residual, and the flatness scatter of the central film. A film thickness
reported without those is not evidence, because the failure modes of this
formulation are quiet ones - a silently static interface, a deactivated enforced
boundary condition, a bubble that never reached steady state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from postprocess.film import aussillous_quere, bretherton
from problems.bretherton_bubble import VoidBubbleProblem
from problems.geometry import geometry_from_case


def load_case(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pyoomph-case-v2":
        raise ValueError(f"{path} is not a pyoomph-case-v2 document")
    if payload.get("physics", {}).get("interior") != "void":
        raise ValueError("this runner only accepts a void bubble interior")
    return payload


def build(case: dict, *, overrides: dict | None = None) -> VoidBubbleProblem:
    physics = dict(case["physics"])
    mesh = dict(case["mesh"])
    physics.update((overrides or {}).get("physics", {}))
    mesh.update((overrides or {}).get("mesh", {}))

    geometry = geometry_from_case(physics)
    # A case that seeds its capsule the legacy way, by `rb0_frac` with an explicit
    # `z_rear` and `l_domain`, carries no pads for `geometry_from_case` to read.
    # Re-apply them here so a pad override means the same thing for every case
    # shape; silently ignoring it would make a window study on such a case
    # produce identical rows and look converged.
    pads = {k: physics[k] for k in ("pad_rear", "pad_front") if k in physics}
    if pads and "rb0_frac" in physics:
        geometry = geometry.with_pads(
            pad_rear=float(pads.get("pad_rear", geometry.pad_rear)),
            pad_front=float(pads.get("pad_front", geometry.pad_front)),
        )

    return VoidBubbleProblem(
        geometry=geometry,
        ca_in=float(physics["ca_in"]),
        laplace=float(physics["laplace"]),
        film_elements=float(mesh["film_elements"]),
        bulk_resolution=float(mesh["bulk_resolution"]),
        closure=case["closure"]["frame_speed"],
        outlet=case["boundaries"]["outlet"],
        remesh=bool(mesh.get("remeshing", True)),
        newton_tolerance=float(case["tolerances"]["newton"]),
    )


def solve_case(
    problem: VoidBubbleProblem,
    *,
    output_dir: Path,
    relax_time: float,
    start_step: float,
    temporal_error: float,
    mode: str = "steady",
) -> dict:
    """Initialise, audit, solve, and return the result together with its checks.

    ``mode`` is honoured rather than assumed. A case that asks for ``transient``
    and silently receives a steady solve would report a steady answer under a
    transient label, and the difference is not cosmetic: the stationary polish
    moves the film by nearly 20% away from the transient state.
    """
    if mode not in ("steady", "transient"):
        raise ValueError(f"unsupported solve mode {mode!r}; expected 'steady' or 'transient'")
    problem.set_output_directory(str(output_dir))
    problem.initialise()
    audit = problem.audit_constraints()

    started = time.time()
    if mode == "steady":
        route = problem.solve_to_steady(
            relax_time=relax_time, start_step=start_step, temporal_error=temporal_error
        )
    else:
        problem.run(relax_time, startstep=start_step, outstep=relax_time / 8,
                    temporal_error=temporal_error, maxstep=relax_time / 8)
        route = "transient"
    wall = time.time() - started

    checks = problem.consistency()
    film = problem.measure_film()
    geom = problem.geometry
    ca_b = checks.frame_speed

    return {
        "ca_in": problem.ca_in,
        "laplace": problem.laplace,
        "r_tube": geom.r_tube,
        "pad_rear": geom.pad_rear,
        "pad_front": geom.pad_front,
        "film_elements": problem.film_elements,
        # Mesh spacing is reported as Delta/R0, the convention the comparison
        # plots use. R0 is 1 in the working nondimensionalisation, so this is a
        # naming guarantee rather than a rescaling - but a case with R0 != 1
        # would otherwise report raw lengths under an R0-normalised label.
        "bulk_resolution_over_r0": problem.bulk_resolution / geom.r0,
        "film_element_size_over_r0": problem.film_element_size() / geom.r0,
        "closure": problem.closure,
        "outlet": problem.outlet,
        "ndof": problem.ndof(),
        "route": route,
        "wall_seconds": wall,
        # Results.
        "ca_b": ca_b,
        "speed_ratio": ca_b / problem.ca_in,
        "h_flat": checks.h_flat,
        "h_flat_over_r_tube": checks.h_flat / geom.r_tube,
        "h_min_over_r_tube": film.h_min / geom.r_tube,
        "bubble_pressure_gauge": checks.bubble_pressure,
        "bubble_pressure_excess": checks.bubble_pressure_excess,
        "bubble_length": float(film.bubble_length),
        # Reference laws, evaluated at the MEASURED bubble speed, as the parent does.
        "aussillous_quere_at_ca_b": aussillous_quere(ca_b),
        "bretherton_at_ca_b": bretherton(ca_b),
        "deviation_from_aussillous_quere": (checks.h_flat / geom.r_tube)
        / aussillous_quere(ca_b)
        - 1.0,
        # Checks that could have failed.
        "check_void_volume_error": checks.volume_error,
        "check_flux_error": checks.flux_error,
        "check_stagnant_film_error": checks.stagnant_film_error,
        "check_film_flatness": film.flatness,
        "check_initial_constraint_audit": audit,
        "check_central_nodes": film.n_central,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--ca", type=float, default=None, help="override Ca_in")
    parser.add_argument("--la", type=float, default=None, help="override La")
    parser.add_argument("--pad", type=float, default=None, help="override both pads")
    parser.add_argument("--film-elements", type=float, default=None)
    parser.add_argument("--bulk-resolution", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    case = load_case(args.case)
    overrides: dict = {"physics": {}, "mesh": {}}
    if args.ca is not None:
        overrides["physics"]["ca_in"] = args.ca
    if args.la is not None:
        overrides["physics"]["laplace"] = args.la
    if args.pad is not None:
        overrides["physics"]["pad_rear"] = args.pad
        overrides["physics"]["pad_front"] = args.pad
    if args.film_elements is not None:
        overrides["mesh"]["film_elements"] = args.film_elements
    if args.bulk_resolution is not None:
        overrides["mesh"]["bulk_resolution"] = args.bulk_resolution

    problem = build(case, overrides=overrides)
    solve = case["solve"]
    result = solve_case(
        problem,
        output_dir=args.output_dir or (ROOT / "receipts" / case["case_id"]),
        relax_time=float(solve["relax_time"]),
        start_step=float(solve["start_step"]),
        temporal_error=float(solve["temporal_error"]),
        mode=solve.get("mode", "steady"),
    )
    result["case_id"] = case["case_id"]
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
