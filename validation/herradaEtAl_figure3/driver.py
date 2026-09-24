"""Trace the Herrada--Yu--Stone figure 3 response with native arclength continuation."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "pyoomph-case-v1"
BENCHMARK_SCHEMA = "herrada-figure3-v1"
DOI = "10.1017/jfm.2023.123"
BASE_CASE_ID = "herradaEtAl_figure3_bo1p56"
BASE_DOMAIN = {"domain_over_r": 15.0, "upper_tip_over_r": 10.0}
BASE_MESH = {"film_elements": 4.0, "bulk_resolution": 0.08}
CONVERGENCE_VARIANTS = {
    "mesh-medium": {
        "case_id": f"{BASE_CASE_ID}_mesh_medium",
        "domain": BASE_DOMAIN,
        "mesh": {"film_elements": 6.0, "bulk_resolution": 0.08 * 4.0 / 6.0},
    },
    "mesh-fine": {
        "case_id": f"{BASE_CASE_ID}_mesh_fine",
        "domain": BASE_DOMAIN,
        "mesh": {"film_elements": 8.0, "bulk_resolution": 0.04},
    },
    "domain20-fine": {
        "case_id": f"{BASE_CASE_ID}_domain20_fine",
        "domain": {"domain_over_r": 20.0, "upper_tip_over_r": 12.5},
        "mesh": {"film_elements": 8.0, "bulk_resolution": 0.04},
    },
    "domain25-fine": {
        "case_id": f"{BASE_CASE_ID}_domain25_fine",
        "domain": {"domain_over_r": 25.0, "upper_tip_over_r": 15.0},
        "mesh": {"film_elements": 8.0, "bulk_resolution": 0.04},
    },
}
SAMPLING_CAPS = {"sampling-half": 0.01, "sampling-quarter": 0.005,
                 "sampling-quarter-mesh-medium": 0.005,
                 "sampling-quarter-domain20-fine": 0.005,
                 "sampling-quarter-domain25-fine": 0.005,
                 "sampling-eighth-hook": 0.0025,
                 "sampling-eighth-hook-mesh-medium": 0.0025,
                 "sampling-eighth-hook-domain20-fine": 0.0025,
                 "sampling-eighth-hook-domain25-fine": 0.0025}
SAMPLING_SOURCE_VARIANTS = {
    "sampling-half": "mesh-fine",
    "sampling-quarter": "mesh-fine",
    "sampling-quarter-mesh-medium": "mesh-medium",
    "sampling-quarter-domain20-fine": "domain20-fine",
    "sampling-quarter-domain25-fine": "domain25-fine",
    "sampling-eighth-hook": "mesh-fine",
    "sampling-eighth-hook-mesh-medium": "mesh-medium",
    "sampling-eighth-hook-domain20-fine": "domain20-fine",
    "sampling-eighth-hook-domain25-fine": "domain25-fine",
}
# Each local probe starts at a checkpoint accepted by its own base trace.
LOCAL_SAMPLING_STATIONS = {
    "sampling-eighth-hook": 80,
    "sampling-eighth-hook-mesh-medium": 77,
    "sampling-eighth-hook-domain20-fine": 73,
    "sampling-eighth-hook-domain25-fine": 70,
}
FULL_SAMPLING_STATIONS = {
    "sampling-quarter-mesh-medium": 77,
    "sampling-quarter-domain20-fine": 73,
    "sampling-quarter-domain25-fine": 70,
}
LOCAL_COMPLETION = {"kind": "event",
                    "value": "first_ca_reversal_plus_10_stations_and_descending_l_b_le_6p84"}
BRANCH_FIELDS = (
    "station", "bond", "ca", "bubble_length_over_r", "signed_ca_b",
    "volume_error", "flux_error", "min_gap", "axial_span_over_r",
)


def case_digest(case: dict) -> str:
    encoded = json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_case(path: Path, expected_mode: str | None = None) -> dict:
    case = json.loads(path.read_text(encoding="utf-8"))
    validate_case(case, expected_mode)
    return case


def validate_case(case: dict, expected_mode: str | None = None) -> None:
    required = {
        "schema", "benchmark_schema", "benchmark_mode", "case_id", "source", "geometry",
        "nondimensionalization", "physics", "mesh", "solve", "tolerances", "boundaries",
        "closure", "output", "completion", "acceptance",
    }
    missing = sorted(required - case.keys())
    if missing:
        raise ValueError(f"case is missing required fields: {', '.join(missing)}")
    if case["schema"] != SCHEMA or case["benchmark_schema"] != BENCHMARK_SCHEMA:
        raise ValueError(f"case must use {SCHEMA} with benchmark_schema={BENCHMARK_SCHEMA!r}")
    if case["benchmark_mode"] not in ("run", "smoke"):
        raise ValueError("benchmark_mode must be 'run' or 'smoke'")
    if expected_mode is not None and case["benchmark_mode"] != expected_mode:
        raise ValueError(f"{expected_mode} requires a case with mode={expected_mode!r}")
    if case["source"] != {"doi": DOI, "figure": 3}:
        raise ValueError("source must identify Herrada et al. figure 3")

    if case["geometry"] != {
        "kind": "axisymmetric", "coordinates": ["r", "z"], "frame": "co-moving"
    }:
        raise ValueError("geometry must be axisymmetric (r,z) in the co-moving frame")
    if case["nondimensionalization"] != {
        "surface_tension": 1.0, "liquid_viscosity": 1.0,
        "volume_equivalent_radius": 1.0,
    }:
        raise ValueError("nondimensionalization must set sigma=mu=R0=1")

    def finite(mapping: dict, name: str, *, positive: bool = False) -> float:
        try:
            value = float(mapping[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(value) or (positive and value <= 0):
            raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
        return value

    p = case["physics"]
    expected = {
        "reynolds": 0.0, "laplace": 0.0, "bond": 1.56, "vhat": 3.3, "r0": 1.0,
        "seed_ca": 0.02,
    }
    for name, value in expected.items():
        if not math.isclose(float(p.get(name, math.nan)), value, rel_tol=1e-12,
                            abs_tol=1e-12):
            raise ValueError(f"{name} must be {value!r} for the figure 3 mapping")
    rt = 3.3 ** (-1.0 / 3.0)
    if not math.isclose(float(p.get("r_tube", math.nan)), rt, rel_tol=1e-12):
        raise ValueError("r_tube must equal vhat^(-1/3)")
    if p.get("interior") != "void" or p.get("allow_signed_flow") is not True:
        raise ValueError("figure 3 requires the explicit signed-flow void approximation")
    if case["boundaries"] != {"inlet": "poiseuille", "outlet": "poiseuille"}:
        raise ValueError("figure 3 cases require two-ended Poiseuille conditions")
    if case["closure"] != {"frame_speed": "front_tip"}:
        raise ValueError("the paper coordinate requires the front-tip closure")
    if case["mesh"].get("template") != "LiquidAnnulusMesh":
        raise ValueError("mesh template must be LiquidAnnulusMesh")
    film_elements = finite(case["mesh"], "film_elements", positive=True)
    if film_elements < 1:
        raise ValueError("film_elements must be at least one")
    bulk_resolution = finite(case["mesh"], "bulk_resolution", positive=True)
    if not isinstance(case["mesh"].get("remeshing"), bool):
        raise TypeError("remeshing must be boolean")
    finite(case["tolerances"], "newton", positive=True)
    finite(case["tolerances"], "temporal_error", positive=True)
    finite(case["output"], "cadence", positive=True)
    if case["output"].get("write_states") is not True:
        raise ValueError("output.write_states must be true")
    for name in ("max_volume_error", "max_flux_error", "containment_epsilon"):
        finite(case["acceptance"], name, positive=True)

    if case["benchmark_mode"] == "run":
        convergence = case.get("convergence")
        if convergence is None:
            variant = {
                "case_id": BASE_CASE_ID,
                "domain": BASE_DOMAIN,
                "mesh": BASE_MESH,
            }
        else:
            if not isinstance(convergence, dict) or set(convergence) != {
                "study", "baseline_case_id", "variant"
            }:
                raise ValueError("convergence metadata must declare study, baseline_case_id and variant")
            if convergence["study"] != "figure3_numerical_convergence":
                raise ValueError("convergence study must be figure3_numerical_convergence")
            if convergence["baseline_case_id"] != BASE_CASE_ID:
                raise ValueError(f"convergence baseline_case_id must be {BASE_CASE_ID!r}")
            try:
                variant = CONVERGENCE_VARIANTS[convergence["variant"]]
            except (KeyError, TypeError) as exc:
                raise ValueError("convergence variant is not a declared figure 3 variant") from exc

        sampling = case.get("sampling")
        if sampling is not None:
            if not isinstance(sampling, dict) or sampling.get("variant") not in SAMPLING_CAPS:
                raise ValueError("sampling metadata must name a declared refinement")
            source_variant = SAMPLING_SOURCE_VARIANTS[sampling["variant"]]
            if convergence is None or convergence["variant"] != source_variant:
                raise ValueError(f"sampling refinement requires the {source_variant} variant")
            if sampling != {
                "study": "figure3_continuation_sampling",
                "baseline_case_id": CONVERGENCE_VARIANTS[source_variant]["case_id"],
                "variant": sampling.get("variant"),
            }:
                raise ValueError("sampling metadata must name a declared refinement")
            variant = dict(variant, case_id=f"{BASE_CASE_ID}_{sampling['variant'].replace('-', '_')}")
            if case["continuation"]["max_ds"] != SAMPLING_CAPS[sampling["variant"]]:
                raise ValueError("sampling max_ds does not match its declared cap")
        if case["case_id"] != variant["case_id"]:
            raise ValueError(f"case_id must be {variant['case_id']!r} for this figure 3 variant")
        for name, value in variant["domain"].items():
            if not math.isclose(float(p.get(name, math.nan)), value,
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{name} must be {value!r} for this figure 3 variant")
        for name, actual in (("film_elements", film_elements),
                             ("bulk_resolution", bulk_resolution)):
            value = variant["mesh"][name]
            if not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{name} must be {value!r} for this figure 3 variant")

        cont = case.get("continuation", {})
        if not finite(cont, "initial_ds") < 0:
            raise ValueError("initial_ds must point towards decreasing signed Ca")
        min_ds = finite(cont, "min_ds", positive=True)
        max_ds = finite(cont, "max_ds", positive=True)
        if min_ds > max_ds:
            raise ValueError("continuation requires 0 < min_ds <= max_ds")
        if not min_ds <= abs(float(cont["initial_ds"])) <= max_ds:
            raise ValueError("initial_ds must lie between min_ds and max_ds in magnitude")
        if not isinstance(cont.get("max_steps"), int) or not isinstance(cont.get("min_turns"), int):
            raise ValueError("max_steps and min_turns must be integers")
        if cont["max_steps"] <= 0 or cont["min_turns"] < 2:
            raise ValueError("continuation needs a finite positive budget and at least two turns")
        if (sampling is not None and sampling["variant"] in LOCAL_SAMPLING_STATIONS
                and cont["max_steps"] != 200):
            raise ValueError("local sampling max_steps must be 200")
        if (sampling is not None and sampling["variant"] in FULL_SAMPLING_STATIONS
                and cont["max_steps"] != 2000):
            raise ValueError("full-branch sampling max_steps must be 2000")
        if not math.isclose(finite(cont, "stop_ca"), -0.02, rel_tol=0, abs_tol=1e-12):
            raise ValueError("stop_ca must be -0.02 for the declared completion event")
        finite(cont, "bond_step", positive=True)
        if not isinstance(cont.get("max_newton_iterations"), int) or cont["max_newton_iterations"] <= 0:
            raise ValueError("max_newton_iterations must be a positive integer")
        for name in ("relax_time", "start_step"):
            finite(case["solve"], name, positive=True)
        if case["solve"].get("mode") != "continuation":
            raise ValueError("production solve mode must be continuation")
        expected_completion = LOCAL_COMPLETION if sampling is not None and sampling[
            "variant"] in LOCAL_SAMPLING_STATIONS else {
            "kind": "event", "value": "ca_le_-0p02_and_at_least_2_turns"
        }
        if case["completion"] != expected_completion:
            raise ValueError("production completion event is not explicit")
    else:
        if "convergence" in case:
            raise ValueError("smoke cases cannot declare production convergence variants")
        for name, value in BASE_DOMAIN.items():
            if not math.isclose(float(p.get(name, math.nan)), value,
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{name} must be {value!r} for the figure 3 smoke mapping")
        smoke = case.get("smoke", {})
        if finite(smoke, "ca") >= 0 or not isinstance(smoke.get("steps"), int) or smoke["steps"] <= 0:
            raise ValueError("smoke requires a negative Ca and a positive step count")
        finite(smoke, "dt", positive=True)
        finite(smoke, "temporal_error", positive=True)
        if case["solve"].get("mode") != "transient":
            raise ValueError("smoke solve mode must be transient")
        if case["completion"] != {"kind": "event", "value": "bounded_transient_steps"}:
            raise ValueError("smoke completion event is not explicit")


def turning_candidates(ca_values: list[float], atol: float = 1e-12) -> list[dict]:
    """Return chronological parameter reversals without assigning stability."""
    turns: list[dict] = []
    for index in range(1, len(ca_values) - 1):
        before = ca_values[index] - ca_values[index - 1]
        after = ca_values[index + 1] - ca_values[index]
        if abs(before) > atol and abs(after) > atol and before * after < 0:
            turns.append({"station": index, "ca": ca_values[index]})
    return turns


def trace_complete(rows: list[dict], stop_ca: float, min_turns: int) -> bool:
    return bool(rows) and float(rows[-1]["ca"]) <= stop_ca and len(
        turning_candidates([float(row["ca"]) for row in rows])
    ) >= min_turns


def local_trace_complete(rows: list[dict]) -> bool:
    if len(rows) < 12:
        return False
    turns = turning_candidates([float(row["ca"]) for row in rows])
    if not turns or int(rows[-1]["station"]) - turns[0]["station"] < 10:
        return False
    last_length = float(rows[-1]["bubble_length_over_r"])
    previous_length = float(rows[-2]["bubble_length_over_r"])
    return math.isfinite(last_length) and last_length <= 6.84 and last_length < previous_length


def case_trace_complete(rows: list[dict], case: dict) -> bool:
    if case["completion"] == LOCAL_COMPLETION:
        return local_trace_complete(rows)
    cont = case["continuation"]
    return trace_complete(rows, float(cont["stop_ca"]), int(cont["min_turns"]))


def trace_outcome(rows: list[dict], *, stop_ca: float, min_turns: int,
                  steps_taken: int, max_steps: int, failure: str | None = None,
                  case: dict | None = None) -> str:
    if failure is not None:
        return "failed"
    complete = (case_trace_complete(rows, case) if case is not None
                else trace_complete(rows, stop_ca, min_turns))
    if complete:
        return "complete"
    if steps_taken >= max_steps:
        return "incomplete"
    return "running"


def validate_resume(restart: dict, case: dict) -> None:
    if restart.get("schema") != "herrada-figure3-restart-v2":
        raise ValueError("resume requires figure 3 restart v2 (meridional length); keep legacy runs separate")
    if restart.get("case_sha256") != case_digest(case):
        raise ValueError("resume case hash does not match the selected case")
    if restart.get("phase") != "trace":
        raise ValueError("resume is only supported for an established arclength trace")
    if not isinstance(restart.get("station"), int) or restart["station"] < 0:
        raise ValueError("resume station must be a non-negative integer")
    try:
        next_ds = float(restart["next_ds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resume next_ds must be finite and non-zero") from exc
    if not math.isfinite(next_ds) or next_ds == 0:
        raise ValueError("resume next_ds must be finite and non-zero")
    for name in ("state_path", "branches_path", "run_directory"):
        if not isinstance(restart.get(name), str) or not restart[name]:
            raise ValueError(f"resume {name} must be a non-empty path")
    for name in ("state_sha256", "branches_prefix_sha256"):
        digest = restart.get(name, "")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError(f"resume {name} must be a SHA-256 digest")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def branch_digest(rows: list[dict]) -> str:
    normalized = [{key: int(row[key]) if key == "station" else float(row[key])
                   for key in BRANCH_FIELDS} for row in rows]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_sampling_seed(seed: dict, case: dict) -> tuple[dict, list[dict], Path]:
    """Authenticate a pre-fold checkpoint from the sampling case's own base trace."""
    validate_case(case, "run")
    if "sampling" not in case or seed.get("schema") != "herrada-figure3-sampling-seed-v1":
        raise ValueError("refine requires a sampling case and sampling seed v1")
    for name in ("run_directory", "case_path", "branches_path", "state_path", "progress_path"):
        if not isinstance(seed.get(name), str) or not seed[name]:
            raise ValueError(f"seed {name} must be a non-empty path")
    for name in ("case_sha256", "branches_sha256", "state_sha256", "progress_sha256"):
        value = seed.get(name, "")
        if not isinstance(value, str) or len(value) != 64 or any(
            c not in "0123456789abcdef" for c in value
        ):
            raise ValueError(f"seed {name} must be a SHA-256 digest")
    source = Path(seed["run_directory"])
    source_case = read_case(source / seed["case_path"], "run")
    source_variant = SAMPLING_SOURCE_VARIANTS[case["sampling"]["variant"]]
    accepted_case = read_case(HERE / "cases" / f"{source_variant}.json", "run")
    if source_case != accepted_case or case_digest(source_case) != seed["case_sha256"]:
        raise ValueError(f"seed source case must match the accepted {source_variant} case and hash")
    normalized = copy.deepcopy(case)
    normalized.pop("sampling")
    normalized["case_id"] = source_case["case_id"]
    normalized["continuation"]["max_ds"] = source_case["continuation"]["max_ds"]
    if case["completion"] == LOCAL_COMPLETION:
        normalized["completion"] = source_case["completion"]
        normalized["continuation"]["initial_ds"] = source_case["continuation"]["initial_ds"]
        normalized["continuation"]["max_steps"] = source_case["continuation"]["max_steps"]
    if normalized != source_case:
        raise ValueError("sampling may change only case identity, metadata, step cap and local endpoint")
    state, branches = source / seed["state_path"], source / seed["branches_path"]
    if file_digest(state) != seed["state_sha256"]:
        raise ValueError("seed state hash mismatch")
    if file_digest(branches) != seed["branches_sha256"]:
        raise ValueError("seed branch table hash mismatch")
    rows = read_branches(branches)
    station = seed.get("station")
    turns = turning_candidates([float(row["ca"]) for row in rows])
    if (type(station) is not int or station < 1 or station >= len(rows)
            or not turns or station >= turns[0]["station"]):
        raise ValueError("seed must precede the first reversal on branch III")
    sampling_variant = case["sampling"]["variant"]
    required_station = (LOCAL_SAMPLING_STATIONS.get(sampling_variant)
                        or FULL_SAMPLING_STATIONS.get(sampling_variant))
    if required_station is not None and station != required_station:
        raise ValueError(f"sampling seed must use saved {source_variant} station {required_station}")
    if seed["state_path"] != f"checkpoints/station_{station:05d}.dump":
        raise ValueError("seed state must be the saved checkpoint for its station")
    next_ds = float(seed.get("next_ds", math.nan))
    if not math.isfinite(next_ds) or next_ds == 0:
        raise ValueError("seed next_ds must be finite and non-zero")
    progress_path = source / seed["progress_path"]
    if file_digest(progress_path) != seed["progress_sha256"]:
        raise ValueError("seed progress hash mismatch")
    progress = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    matching = [event for event in progress if event.get("stage") == "station_accepted"
                and event.get("station") == station]
    if (len(matching) != 1 or float(matching[0].get("next_ds", math.nan)) != next_ds
            or float(matching[0].get("ca", math.nan)) != float(rows[station]["ca"])):
        raise ValueError("seed next_ds and ca must match the unique accepted progress station")
    return source_case, rows, state


def validate_loaded_seed(row: dict, source_row: dict) -> None:
    """Reject a state that does not reproduce the recorded physical observables."""
    for name in ("bond", "ca", "bubble_length_over_r", "volume_error"):
        if not math.isclose(float(row[name]), float(source_row[name]),
                            rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(f"loaded seed {name} disagrees with its source station")


def atomic_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".started").open("x", encoding="utf-8") as stream:
        stream.write("single-case attempt\n")


def _require_one_rank() -> None:
    from pyoomph.generic.mpi import get_mpi_nproc

    if get_mpi_nproc() > 1:
        raise RuntimeError("figure 3 continuation must run on exactly one MPI rank")


def seed_geometry(case: dict):
    from postprocess.film import aussillous_quere
    from problems.geometry import CapsuleGeometry

    p = case["physics"]
    rt = float(p["r_tube"])
    body = rt * (1.0 - aussillous_quere(float(p["seed_ca"])))
    length = CapsuleGeometry.bubble_length(float(p["r0"]), body)
    front = float(p["upper_tip_over_r"]) * rt
    return CapsuleGeometry(r0=float(p["r0"]), r_tube=rt, r_body=body,
                           z_rear=front - length,
                           l_domain=float(p["domain_over_r"]) * rt)


def build_problem(case: dict, *, ca: float, bond: float,
                  pressure_datum: str = "fixation"):
    from problems.bretherton_bubble import VoidBubbleProblem

    problem = VoidBubbleProblem(
        geometry=seed_geometry(case), ca_in=ca, laplace=0.0, bond=bond,
        allow_signed_flow=True, film_elements=float(case["mesh"]["film_elements"]),
        bulk_resolution=float(case["mesh"]["bulk_resolution"]), closure="front_tip",
        outlet="poiseuille", remesh=bool(case["mesh"].get("remeshing", True)),
        pressure_datum=pressure_datum,
        newton_tolerance=float(case["tolerances"]["newton"]),
    )
    problem.set_linear_solver("superlu")
    problem.set_c_compiler("system")
    problem.continuation_data_in_states = True
    problem.set_arclength_inner_product("ndof")
    return problem


def meridional_length(z, r, *, quadrature_order: int = 16) -> float:
    """Integrate equation (2.3) along ordered quadratic interface elements.

    Coordinates alternate endpoint, midpoint, endpoint, as returned by
    ``interface_polyline``. This is a planar arc length, without the
    axisymmetric surface-area factor. Re-entrant portions retain curve order.
    """
    import numpy as np

    points = np.column_stack((z, r)).astype(float)
    if len(points) < 3 or len(points) % 2 != 1 or not np.isfinite(points).all():
        raise ValueError("length requires finite ordered quadratic interface nodes")
    xi, weights = np.polynomial.legendre.leggauss(quadrature_order)
    a, middle, b = points[:-2:2], points[1:-1:2], points[2::2]
    derivative = (xi[:, None, None] * (a - 2 * middle + b)[None, :, :]
                  + 0.5 * (b - a)[None, :, :])
    return float(np.sum(weights[:, None] * np.linalg.norm(derivative, axis=2)))


def extract_row(problem, station: int, acceptance: dict) -> tuple[dict, tuple[list, list]]:
    z, r = problem.interface_polyline()
    z_values, r_values = list(map(float, z)), list(map(float, r))
    geom = problem.geometry
    volume = problem.observable("void_volume", "liquid/interface")
    inlet = problem.observable("inlet_flux", "liquid/inlet")
    outlet = problem.observable("outlet_flux", "liquid/outlet")
    flux_scale = max(abs(inlet), abs(outlet), 1e-300)
    row = {
        "station": station,
        "bond": problem.current_bond(),
        "ca": problem.current_ca_in(),
        "bubble_length_over_r": meridional_length(z_values, r_values) / geom.r_tube,
        "axial_span_over_r": (max(z_values) - min(z_values)) / geom.r_tube,
        "signed_ca_b": problem.frame_speed(),
        "volume_error": volume / geom.volume - 1.0,
        "flux_error": (outlet - inlet) / flux_scale,
        "min_gap": min(geom.r_tube - value for value in r_values),
    }
    if not all(math.isfinite(float(value)) for value in row.values()):
        raise RuntimeError("accepted branch row contains a non-finite value")
    if abs(row["volume_error"]) > float(acceptance["max_volume_error"]):
        raise RuntimeError("void-volume error exceeds its acceptance gate")
    if abs(row["flux_error"]) > float(acceptance["max_flux_error"]):
        raise RuntimeError("inlet/outlet flux mismatch exceeds its acceptance gate")
    eps = float(acceptance["containment_epsilon"])
    if row["min_gap"] <= eps or min(z_values) <= eps or max(z_values) >= geom.l_domain - eps:
        raise RuntimeError("bubble is not geometrically contained in the tube window")
    return row, (z_values, r_values)


def write_profile(output: Path, station: int, problem, profile: tuple[list, list],
                  case: dict) -> None:
    z_values, r_values = profile
    rt = problem.geometry.r_tube
    tip = float(case["physics"]["upper_tip_over_r"])
    path = output / "profiles" / f"station_{station:05d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("z_solver", "r_solver", "z_paper", "gap_over_r"))
        writer.writerows((z, r, z / rt - tip, 1.0 - r / rt)
                         for z, r in zip(z_values, r_values, strict=True))


def write_branches(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=BRANCH_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_branches(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if rows and "axial_span_over_r" not in rows[0]:
        raise ValueError("legacy axial-length table: remeasure saved profiles before plotting or resuming")
    if [int(row["station"]) for row in rows] != list(range(len(rows))):
        raise ValueError("resume branches are not in chronological station order")
    return rows


def checkpoint(problem, output: Path, case: dict, rows: list[dict], next_ds: float,
               parent_run: str | None) -> None:
    station = int(rows[-1]["station"])
    state = output / "checkpoints" / f"station_{station:05d}.dump"
    state.parent.mkdir(parents=True, exist_ok=True)
    problem.save_state(str(state), quiet=True)
    atomic_json(output / "restart.json", {
        "schema": "herrada-figure3-restart-v2", "case_sha256": case_digest(case),
        "phase": "trace", "station": station, "next_ds": next_ds,
        "state_path": str(state.relative_to(output)), "branches_path": "branches.csv",
        "state_sha256": file_digest(state), "branches_prefix_sha256": branch_digest(rows),
        "run_directory": str(output.resolve()), "parent_run": parent_run,
    })


def run_trace(case: dict, output: Path, resume_path: Path | None,
              seed_path: Path | None = None) -> int:
    if resume_path is not None and seed_path is not None:
        raise ValueError("resume and a new sampling study are distinct operations")
    if "sampling" in case and resume_path is None and seed_path is None:
        raise ValueError("sampling cases require refine --seed or an exact-case resume")
    prepare_output(output)
    atomic_json(output / "case.json", case)
    progress: list[dict] = []

    def event(stage: str, **fields) -> None:
        progress.append({"time_unix": time.time(), "stage": stage, **fields})
        atomic_jsonl(output / "progress.jsonl", progress)

    cont = case["continuation"]
    parent_run: str | None = None
    rows: list[dict] = []
    steps_taken = 0
    failure: str | None = None
    event("initialising", resume=None if resume_path is None else str(resume_path))
    try:
        _require_one_rank()
        if seed_path is not None:
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
            _, source_rows, state = validate_sampling_seed(seed, case)
            parent_run = str(Path(seed["run_directory"]).resolve())
            atomic_json(output / "sampling-seed.json", seed)
            problem = build_problem(case, ca=float(case["physics"]["seed_ca"]), bond=0.0)
            problem.set_output_directory(str(output / "solver"))
            problem.initialise()
            problem.load_state(str(state), quiet=True)
            row, profile = extract_row(problem, 0, case["acceptance"])
            validate_loaded_seed(row, source_rows[seed["station"]])
            rows = [row]
            next_ds = math.copysign(min(abs(float(seed["next_ds"])),
                                        float(cont["max_ds"])), float(seed["next_ds"]))
            write_profile(output, 0, problem, profile, case)
            write_branches(output / "branches.csv", rows)
            checkpoint(problem, output, case, rows, next_ds, parent_run)
            event("sampling_started", station=0, source_station=seed["station"],
                  parent_run=parent_run, next_ds=next_ds,
                  degrees_of_freedom=problem.ndof())
        elif resume_path is None:
            problem = build_problem(case, ca=float(case["physics"]["seed_ca"]), bond=0.0)
            problem.set_output_directory(str(output / "solver"))
            problem.initialise()
            problem.audit_constraints()
            event("initialised", bond=0.0, ca=case["physics"]["seed_ca"])
            solve = case["solve"]
            problem.solve_to_steady(
                relax_time=float(solve["relax_time"]),
                start_step=float(solve["start_step"]),
                temporal_error=float(case["tolerances"]["temporal_error"]),
            )
            problem.go_to_param(Bo=float(case["physics"]["bond"]),
                                max_step=float(cont["bond_step"]))
            row, profile = extract_row(problem, 0, case["acceptance"])
            rows = [row]
            write_profile(output, 0, problem, profile, case)
            write_branches(output / "branches.csv", rows)
            problem.reset_arc_length_parameters()
            problem.set_arclength_inner_product("ndof")
            next_ds = float(cont["initial_ds"])
            checkpoint(problem, output, case, rows, next_ds, parent_run)
            event("trace_started", station=0, bond=row["bond"], ca=row["ca"])
        else:
            restart = json.loads(resume_path.read_text(encoding="utf-8"))
            validate_resume(restart, case)
            source_run = Path(restart["run_directory"])
            state = source_run / restart["state_path"]
            source_branches = source_run / restart["branches_path"]
            if not state.is_file() or not source_branches.is_file():
                raise ValueError("resume state or branch table is missing")
            if file_digest(state) != restart["state_sha256"]:
                raise ValueError("resume state hash does not match the accepted checkpoint")
            all_rows = read_branches(source_branches)
            station = int(restart["station"])
            if station < 0 or len(all_rows) <= station:
                raise ValueError("resume station is absent from the branch table")
            rows = all_rows[:station + 1]
            if int(rows[-1]["station"]) != station:
                raise ValueError("resume station does not match the validated branch prefix")
            if branch_digest(rows) != restart["branches_prefix_sha256"]:
                raise ValueError("resume branch prefix differs from the checkpoint")
            write_branches(output / "branches.csv", rows)
            problem = build_problem(case, ca=float(case["physics"]["seed_ca"]), bond=0.0)
            problem.set_output_directory(str(output / "solver"))
            problem.initialise()
            problem.load_state(str(state), quiet=True)
            next_ds = float(restart["next_ds"])
            parent_run = str(source_run.resolve())
            steps_taken = station
            event("resumed", station=station, parent_run=parent_run)

        while steps_taken < int(cont["max_steps"]) and not case_trace_complete(rows, case):
            next_ds = problem.arclength_continuation(
                "Ca_in", next_ds, max_ds=float(cont["max_ds"]),
                min_ds=float(cont["min_ds"]),
                max_newton_iterations=int(cont["max_newton_iterations"]),
            )
            station = int(rows[-1]["station"]) + 1
            row, profile = extract_row(problem, station, case["acceptance"])
            candidate_rows = [*rows, row]
            write_profile(output, station, problem, profile, case)
            write_branches(output / "branches.csv", candidate_rows)
            checkpoint(problem, output, case, candidate_rows, next_ds, parent_run)
            rows = candidate_rows
            steps_taken += 1
            event("station_accepted", station=station, ca=row["ca"], next_ds=next_ds,
                  degrees_of_freedom=problem.ndof())
    except Exception as exc:  # noqa: BLE001 - preserve the last accepted checkpoint
        failure = f"{type(exc).__name__}: {exc}"
        event("step_failed", error=failure)

    outcome = trace_outcome(rows, stop_ca=float(cont["stop_ca"]),
                            min_turns=int(cont["min_turns"]),
                            steps_taken=steps_taken, max_steps=int(cont["max_steps"]),
                            failure=failure, case=case)
    status = {
        "status": outcome, "case_id": case["case_id"], "accepted_steps": steps_taken,
        "last_station": int(rows[-1]["station"]) if rows else None,
        "last_ca": float(rows[-1]["ca"]) if rows else None,
        "last_bubble_length_over_r": (float(rows[-1]["bubble_length_over_r"])
                                      if rows else None),
        "candidate_turns": turning_candidates([float(row["ca"]) for row in rows]),
        "completion_scope": ("local_first_ca_fold" if case["completion"] == LOCAL_COMPLETION
                             else "full_figure3_branch"),
        "candidate_length_turns": ([
            {"station": turn["station"], "bubble_length_over_r": turn["ca"]}
            for turn in turning_candidates([float(row["bubble_length_over_r"]) for row in rows])]
            if case["completion"] == LOCAL_COMPLETION else None),
        "error": failure,
        "restart": "restart.json" if (output / "restart.json").is_file() else None,
        "parent_run": parent_run,
    }
    atomic_json(output / "status.json", status)
    return 0 if outcome == "complete" else 2


def run_smoke(case: dict, output: Path, max_steps: int | None) -> int:
    smoke = case["smoke"]
    configured = int(smoke["steps"])
    if max_steps is not None and not 0 < max_steps <= configured:
        raise ValueError("--max-steps may only reduce the smoke case's configured budget")
    steps = max_steps or configured
    prepare_output(output)
    atomic_json(output / "case.json", case)
    progress = [{"time_unix": time.time(), "stage": "initialising"}]
    atomic_jsonl(output / "progress.jsonl", progress)
    try:
        _require_one_rank()
        problem = build_problem(case, ca=float(smoke["ca"]),
                                bond=float(case["physics"]["bond"]))
        problem.set_output_directory(str(output / "solver"))
        problem.initialise()
        audit = problem.audit_constraints()
        dt = float(smoke["dt"])
        problem.run(steps * dt, startstep=dt, maxstep=dt, outstep=steps * dt,
                    temporal_error=float(smoke["temporal_error"]))
        row, profile = extract_row(problem, 0, case["acceptance"])
        write_profile(output, 0, problem, profile, case)
        write_branches(output / "branches.csv", [row])
        progress.append({"time_unix": time.time(), "stage": "complete", "steps": steps})
        atomic_jsonl(output / "progress.jsonl", progress)
        atomic_json(output / "status.json", {
            "status": "complete", "kind": "bounded_forcing_smoke",
            "case_id": case["case_id"], "steps": steps, "dt": dt,
            "physical_time": float(problem.get_current_time()),
            "degrees_of_freedom": problem.ndof(), "final_state": row,
            "initial_constraint_audit": audit,
            "claims": ["finite state", "void-volume gate", "flux gate", "geometric containment"],
        })
        return 0
    except Exception as exc:  # noqa: BLE001 - smoke failures require a status receipt
        error = f"{type(exc).__name__}: {exc}"
        progress.append({"time_unix": time.time(), "stage": "failed", "error": error})
        atomic_jsonl(output / "progress.jsonl", progress)
        atomic_json(output / "status.json", {
            "status": "failed", "kind": "bounded_forcing_smoke",
            "case_id": case["case_id"], "error": error,
        })
        return 2


def plot_branch(case: dict, output: Path) -> int:
    rows = read_branches(output / "branches.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}",
    })
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot([float(row["ca"]) for row in rows],
            [float(row["bubble_length_over_r"]) for row in rows], "k-", lw=1.5)
    ax.set_xlabel(r"$Ca_l$")
    ax.set_ylabel(r"$L_b/R$")
    ax.tick_params(which="both", direction="out")
    ax.minorticks_on()
    for suffix, dpi in (("pdf", 300), ("png", 300)):
        fig.savefig(output / f"herradaEtAl_figure3_branch.{suffix}",
                    bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("run", "refine", "smoke", "plot"):
        command = commands.add_parser(name)
        command.add_argument("--case", required=True, type=Path)
        command.add_argument("--output-dir", required=True, type=Path)
        if name == "run":
            command.add_argument("--resume", type=Path)
        if name == "refine":
            command.add_argument("--seed", required=True, type=Path)
        if name == "smoke":
            command.add_argument("--max-steps", type=int)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    case = read_case(args.case, "run" if args.command in ("run", "refine") else
                     "smoke" if args.command == "smoke" else None)
    if args.command == "run":
        return run_trace(case, args.output_dir, args.resume)
    if args.command == "refine":
        return run_trace(case, args.output_dir, None, args.seed)
    if args.command == "smoke":
        return run_smoke(case, args.output_dir, args.max_steps)
    return plot_branch(case, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
