"""Remesh one retained terminal state, then use the unchanged bounded polish.

The failed parent, model, case, environment and restart packet are bound before
native assembly. Interpolation deliberately changes the state and its layout.
Case qualification uses the original finite-gas gates plus a separate cold-load
check; the stronger geometric predictions are reported independently.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_pressure_diagnostic as diagnostic

require = diagnostic.require
sha = diagnostic.sha
write_json = diagnostic.write_json
SIDES = ((0, 1, 3), (1, 2, 4), (2, 0, 5))


def implementation_records(head):
    """Bind this entry point and the reused inspection implementation to Git."""
    records = {}
    for relative in ("runners/run_pressure_remesh.py", "runners/run_pressure_diagnostic.py"):
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"nonregular implementation: {relative}")
        committed = hashlib.sha256(diagnostic.git(ROOT, "show", f"{head}:{relative}")).hexdigest()
        require(sha(path) == committed, f"implementation differs from commit: {relative}")
        records[relative] = {"path": str(path), "sha256": committed}
    return records


def preflight(args):
    """Validate bytes and metadata without importing the native solver."""
    from runners import run_paper_case as runner

    require(args.source.resolve(strict=True) == ROOT, "runner must use its own isolated component")
    require(
        Path(diagnostic.__file__).resolve() == ROOT / "runners/run_pressure_diagnostic.py"
        and Path(runner.__file__).resolve() == ROOT / "runners/run_paper_case.py",
        "inspection or case runner imported from another component",
    )
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    head = diagnostic.source_contract(ROOT, manifest)
    require(runner.METHOD_FILES == diagnostic.METHOD_FILES, "unexpected method-file contract")
    require(
        manifest["pyoomph"]["commit"] == diagnostic.PYOOMPH
        and manifest["pyoomph"]["native_core"]["sha256"] == diagnostic.CORE,
        "bound environment differs",
    )
    require(manifest["azimuthal_stability"] is True, "AZ layout required")
    require(sha(args.case) == manifest["case_sha256"], "case bytes differ")
    environment = runner.pyoomph_revision()
    runner.require_pinned_environment(environment)
    require(environment["native_core"]["sha256"] == diagnostic.CORE, "installed core differs")
    require(
        environment["interpreter"]["resolved_executable"]
        == manifest["pyoomph"]["interpreter"]["resolved_executable"],
        "interpreter differs",
    )
    case = runner.read_case(args.case)
    require(case["physics"]["problem"] == "finite_gas_bubble", "finite-gas case required")
    require(case["mesh"]["remeshing"] is True, "unchanged case must permit remeshing")
    component = runner.component_revision(ROOT)
    runner.require_clean_source(component)
    require(component["head"] == head, "source changed during preflight")
    implementation = implementation_records(head)
    restart = runner.validate_restart(
        args.packet,
        case,
        sha(args.case),
        component=component,
        environment=environment,
        azimuthal_stability=True,
    )
    require(Path(restart["parent_manifest"]["path"]) == manifest_path, "packet binds another run")
    require(Path(restart["state"]["path"]) == args.state.resolve(strict=True), "state differs")
    require(
        restart["stage"]["kind"] == "completed_polish"
        and restart["stage"]["bond"] == case["physics"]["bond"],
        "terminal target checkpoint required",
    )
    records = [
        restart["packet"],
        restart["parent_manifest"],
        restart["state"],
        *restart["meshes"],
        {"path": str(args.case.resolve()), "sha256": sha(args.case)},
    ]
    final = manifest_path.parent / "checkpoints/final.dump"
    if final.exists():
        records.append({"path": str(final), "sha256": sha(final)})
    return case, restart, environment, component, records, implementation


def verify_source(component, implementation):
    from runners import run_paper_case as runner

    require(runner.component_revision(ROOT) == component, "source changed since preflight")
    for record in implementation.values():
        require(sha(record["path"]) == record["sha256"], "bound implementation changed")


def verify_inputs(records):
    require(diagnostic.parents_unchanged(records), "parent inputs changed")


def run_manifest(args, case, restart, environment, component, implementation):
    """Describe this intervention using its actual canonical entry point."""
    from runners.run_paper_case import CASE_FIELDS_USED

    script = Path(__file__).resolve()
    record = {"path": str(script), "sha256": sha(script)}
    return {
        "schema": "pyoomph-runner-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "case_id": case["case_id"],
        "case_path": str(args.case.resolve()),
        "case_file": str(args.case.resolve()),
        "case_sha256": sha(args.case),
        "case_fields_used": list(CASE_FIELDS_USED),
        "case_fields_scope": "unchanged finite-gas construction, polish and canonical case gates",
        "units": "nondimensional (sigma=mu_liquid=R0=1)",
        "nondimensionalization": case["nondimensionalization"],
        "argv": [sys.executable, str(script), *sys.argv[1:]],
        "runner": record,
        "runner_path": record["path"],
        "runner_sha256": record["sha256"],
        "component_commit": component["commit"],
        "component": component,
        "pyoomph": environment,
        "azimuthal_stability": True,
        "implementation": implementation,
        "restart": restart,
        "intervention": {
            "kind": "retained-terminal-remesh-and-polish",
            "explicit_force_remesh_calls": 1,
            "polish": "unchanged problems.paper_bubble.polish_confined_state",
            "initial_relaxation": 1.0,
            "stage_update_limit": min(40, case["solve"]["max_newton_iterations"]),
            "total_update_limit": 160,
            "polish_remesh_limit": 3,
            "changes": "mesh interpolation and stationary repolish; no equation, case or tolerance change",
            "parent_acceptance_changed": False,
        },
        "qualification_scope": "one finite-gas stationary case; no spectrum or paper-comparison claim",
    }


def port_adjacent_geometry(lower, bulk, np):
    """Measure all three midsides of each triangle uniquely adjoining the port."""
    require(np.all(bulk["element_types"] == 9), "triangle6 bulk required")
    require(np.all(lower["element_types"] == 2), "line3 port required")
    geom = np.column_stack((bulk["coordinate_x"], bulk["coordinate_y"]))
    port = np.column_stack((lower["coordinate_x"], lower["coordinate_y"]))
    cells, edges = bulk["connectivity"], lower["connectivity"]
    require(cells.ndim == 2 and cells.shape[1] == 6, "triangle6 connectivity required")
    require(edges.ndim == 2 and edges.shape[1] == 3, "line3 connectivity required")
    require(len(cells) > 0 and len(edges) > 0, "empty port or bulk")
    require(
        np.issubdtype(cells.dtype, np.integer)
        and np.issubdtype(edges.dtype, np.integer)
        and np.min(cells) >= 0
        and np.max(cells) < len(geom)
        and np.min(edges) >= 0
        and np.max(edges) < len(port),
        "invalid native connectivity",
    )
    mapped = []
    for index, point in enumerate(port):
        matches = np.flatnonzero(np.max(np.abs(geom - point), axis=1) < 1e-12)
        require(len(matches) == 1, "port node lacks a unique bulk match")
        node = int(matches[0])
        mapped.append(node)
        for name in ("pressure", "velocity_x", "velocity_y"):
            require(
                abs(lower[name][index] - bulk[name][node]) < 1e-13,
                f"port/bulk field mismatch: {name}",
            )
    adjacency = {}
    for index, cell in enumerate(cells):
        require(len({int(v) for v in cell}) == 6, "degenerate triangle6")
        for side in SIDES:
            key = frozenset(int(cell[i]) for i in side)
            adjacency.setdefault(key, []).append(index)
    records = []
    for index, edge in enumerate(edges):
        neighbours = adjacency.get(frozenset(mapped[int(v)] for v in edge), [])
        require(len(neighbours) == 1, "port edge lacks exactly one adjacent triangle")
        cell_index = neighbours[0]
        points = geom[cells[cell_index]]
        departures = np.array([points[m] - (points[a] + points[b]) / 2 for a, b, m in SIDES])
        records.append(
            {
                "port_edge": index,
                "bulk_triangle": cell_index,
                "midside_departures_r_z": departures.tolist(),
                "max_departure_l2": float(np.max(np.linalg.norm(departures, axis=1))),
            }
        )
    deviations = np.array([row["midside_departures_r_z"] for row in records])
    return {
        "method": "native type9 triangle6 vertices(0,1,2), midsides(3,4,5); unique line3 adjacency",
        "port_edges": len(records),
        "adjacent_triangles": len({r["bulk_triangle"] for r in records}),
        "max_midside_departure_l2": max(row["max_departure_l2"] for row in records),
        "max_abs_radial_departure": float(np.max(np.abs(deviations[:, :, 0]))),
        "max_abs_axial_departure": float(np.max(np.abs(deviations[:, :, 1]))),
        "edges": records,
    }


def canonical_checks(case, metrics):
    """The five finite-gas predicates in the unchanged paper_bubble.run_model."""
    tol = case["tolerances"]
    required = (
        "stationary_residual_linf",
        "lower_pressure_rms",
        "lower_flux",
        "upper_flux",
        "gas_volume",
        "surface_volume",
        "target_volume",
    )
    require(all(math.isfinite(metrics[key]) for key in required), "nonfinite canonical metric")
    require(metrics["gas_volume"] > 0 and metrics["target_volume"] > 0, "nonpositive gas volume")
    return {
        "stationary": metrics["stationary_residual_linf"] <= tol["newton"],
        "lower_static_pressure": metrics["lower_pressure_rms"] <= tol["pressure_boundary_absolute"],
        "flux": abs(metrics["lower_flux"] - metrics["upper_flux"]) <= tol["flux_absolute"],
        "volume": abs(metrics["gas_volume"] / metrics["target_volume"] - 1)
        <= tol["volume_relative"],
        "volume_integrals": abs(metrics["surface_volume"] / metrics["gas_volume"] - 1)
        <= tol["volume_relative"],
    }


def check_loaded_state(signature, restart, case, pins, expected_pins=None):
    """Refuse changed fields, positions, masks, layout, forcing or actual pins."""
    require(signature["bc_cache_marker"] == restart["stage"]["bc_mode"], "loaded BC cache differs")
    require(
        signature["parameters"]
        == {"Bo": case["physics"]["bond"], "Ca_in": case["physics"]["ca_in"]},
        "loaded forcing differs",
    )
    expected = restart["roundtrip"]["after"]
    require(signature["layout"] == restart["roundtrip"]["layout"], "loaded layout differs")
    require(signature["arrays"] == expected["arrays"], "loaded fields/positions/masks differ")
    for key in ("volume", "frame_speed", "time"):
        require(signature[key] == expected[key], f"loaded {key} differs")
    if expected_pins is not None:
        require(pins == expected_pins, "cold-load actual pins differ")


def configuration(problem, case, model):
    require(problem.case == case, "native case changed")
    require(
        asdict(problem.geometry) == asdict(model.case_geometry(case)), "native geometry changed"
    )
    require(
        problem.bulk_resolution == case["mesh"]["bulk_resolution"]
        and problem.film_elements == case["mesh"]["film_elements"],
        "mesh controls changed",
    )
    require(
        problem.newton_solver_tolerance == case["tolerances"]["newton"], "Newton tolerance changed"
    )
    parameters = {name: float(problem.get_global_parameter(name).value) for name in ("Bo", "Ca_in")}
    require(
        parameters == {"Bo": case["physics"]["bond"], "Ca_in": case["physics"]["ca_in"]},
        "intervention changed forcing",
    )
    return {
        "global_parameters": parameters,
        "physics": case["physics"],
        "geometry": asdict(problem.geometry),
        "boundaries": case["boundaries"],
        "pressure": case["pressure"],
        "closure": case["closure"],
        "mesh": case["mesh"],
        "tolerances": case["tolerances"],
        "solve": case["solve"],
    }


def differing_keys(before, after, prefix=""):
    """Name changed JSON signature members without omitting any full record."""
    if isinstance(before, dict) and isinstance(after, dict):
        changed = []
        for key in sorted(before.keys() | after.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                changed.append(name)
            else:
                changed.extend(differing_keys(before[key], after[key], name))
        return changed
    return [] if before == after else [prefix]


def sample(problem, model, case, output, phase, np):
    """Sample one phase and prove that observations preserve that phase's state."""
    before = model._stationary_state_signature(problem, True)
    pins = diagnostic.actual_pins(problem)
    controls = configuration(problem, case, model)
    lower, lower_meta = diagnostic.extract(problem, "liquid/lower", np)
    bulk, bulk_meta = diagnostic.extract(problem, "liquid", np)
    area = float(problem.observable("area", "liquid/lower"))
    require(math.isfinite(area) and area > 0, "invalid lower area")
    moment = float(problem.observable("pressure_squared", "liquid/lower"))
    require(math.isfinite(moment) and moment >= 0, "invalid native pressure-squared integral")
    volume = float(problem.observable("volume", "gas"))
    require(volume > 0, "nonpositive native gas volume")
    metrics = {
        "stationary_residual_linf": model._stationary_residual(problem),
        "lower_area": area,
        "lower_pressure_mean": float(problem.observable("pressure", "liquid/lower")) / area,
        "lower_pressure_rms": math.sqrt(moment / area),
        "frame_speed": problem.frame_speed(),
        "gas_pressure_mean": float(problem.observable("pressure", "gas")) / volume,
        "gas_volume": volume,
        "surface_volume": float(problem.observable("surface_volume", "liquid/interface")),
        "target_volume": problem.geometry.volume,
        "lower_flux": float(problem.observable("flux", "liquid/lower")),
        "upper_flux": float(problem.observable("flux", "liquid/upper")),
        "gas_volume_multiplier": float(
            problem.get_ode("gas_volume").get_value("pressure", as_float=True)
        ),
    }
    require(all(math.isfinite(value) for value in metrics.values()), "nonfinite sampled observable")
    independent = diagnostic.integrate_lower(lower, area, np)
    require(
        math.isclose(
            metrics["lower_pressure_rms"], independent["pressure_rms"], rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(
            metrics["lower_pressure_mean"],
            independent["pressure_mean"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        "native and independent pressure moments disagree",
    )
    geometry = port_adjacent_geometry(lower, bulk, np)
    z, r = (np.asarray(value).copy() for value in problem.interface_polyline())
    after = model._stationary_state_signature(problem, True)
    after_pins = diagnostic.actual_pins(problem)
    if before != after or pins != after_pins:
        directory = output / "samples"
        directory.mkdir(exist_ok=True)
        write_json(
            directory / f"{phase}-preservation-failure.json",
            {
                "schema": "private-pressure-remesh-preservation-failure-v1",
                "phase": phase,
                "status": "failed",
                "before": before,
                "after": after,
                "actual_pins": {"before": pins, "after": after_pins},
                "differing_signature_keys": differing_keys(before, after),
                "differing_pin_keys": differing_keys(pins, after_pins),
                "sampling_preserved_state": before == after,
                "sampling_preserved_actual_pins": pins == after_pins,
                "numerical_acceptance": False,
            },
        )
    require(before == after, "sampling changed native state")
    require(pins == after_pins, "sampling changed actual pins")
    data = io.BytesIO()
    np.savez_compressed(
        data,
        **{f"lower_{key}": value for key, value in lower.items()},
        **{f"bulk_{key}": value for key, value in bulk.items()},
        interface_z=z,
        interface_r=r,
    )
    directory = output / "samples"
    directory.mkdir(exist_ok=True)
    diagnostic.atomic(directory / f"{phase}.npz", data.getvalue())
    result = {
        "phase": phase,
        "metrics": metrics,
        "independent_pressure": independent,
        "port_adjacent_geometry": geometry,
        "configuration": controls,
        "signature": before,
        "actual_pins": pins,
        "sampling_preserved_state": True,
        "lower_metadata": lower_meta,
        "bulk_metadata": bulk_meta,
    }
    write_json(directory / f"{phase}.json", result)
    return result


def retain_portable_sample(problem, model, output, phase):
    """Retain a dump and its meshes without calling it a qualified restart."""
    from pyoomph.output.states import copy_state_file

    from runners.run_paper_case import state_bc_cache_marker, state_mesh_references

    signature = model._stationary_state_signature(problem, True)
    pins = diagnostic.actual_pins(problem)
    directory = output / "states" / phase
    directory.mkdir(parents=True, exist_ok=False)
    transfer = output / "solver" / f"{phase}.dump"
    require(not transfer.exists(), "sample transfer path occupied")
    saved_setting = problem.eigen_data_in_states
    try:
        problem.eigen_data_in_states = True
        problem.save_state(str(transfer), quiet=True)
        references = state_mesh_references(transfer)
        source_hashes = [sha((transfer.parent / path).resolve()) for path in references]
        state = directory / "state.dump"
        copy_state_file(str(transfer), str(state), copy_mesh_files=True)
        meshes = []
        for reference in state_mesh_references(state):
            path = (state.parent / reference).resolve(strict=True)
            require(
                path.is_relative_to(directory) and path.is_file() and not path.is_symlink(),
                "sample mesh escaped portable custody",
            )
            meshes.append({"stored_path": reference, "path": str(path), "sha256": sha(path)})
        require([mesh["sha256"] for mesh in meshes] == source_hashes, "sample mesh copy differs")
        require(
            state_bc_cache_marker(state) == signature["bc_cache_marker"],
            "saved sample BC marker differs",
        )
    finally:
        problem.eigen_data_in_states = saved_setting
    require(
        signature == model._stationary_state_signature(problem, True),
        "state saving changed native fields",
    )
    require(pins == diagnostic.actual_pins(problem), "state saving changed actual pins")
    record = {
        "schema": "private-pressure-remesh-observation-state-v1",
        "phase": phase,
        "state": {"path": str(state), "sha256": sha(state), "format": "portable_file"},
        "mesh_reference_base": str(directory),
        "meshes": meshes,
        "signature": signature,
        "actual_pins": pins,
        "numerical_acceptance": False,
        "scope": "portable observation dump; no cold-load or stationary qualification claim",
    }
    write_json(directory / "retention.json", record)
    return record


def remesh_and_polish(problem, model, case, output, observe):
    """Make one explicit intervention; leave all retry budgets with the model."""
    before = observe("before")
    require(problem.force_remesh() is True, "explicit remesh did not replace a mesh")
    remeshed = observe("after-remesh")
    polish = model.polish_confined_state(
        problem, case, output, "terminal_remesh", initial_relaxation=1.0
    )
    require(polish["status"] == "converged", "bounded polish did not converge")
    after = observe("after-polish")
    return {
        "before": before,
        "after_remesh": remeshed,
        "after_polish": after,
        "polish": polish,
        "explicit_force_remesh_calls": 1,
        "state_preserved_across_intervention": False,
    }


def causal_predictions(result):
    before, after = (result[name]["metrics"] for name in ("before", "after_polish"))
    absolute_speed_change = abs(after["frame_speed"] - before["frame_speed"])
    # At zero reference speed the relative diagnostic is undefined and fails honestly.
    relative_speed_change = (
        absolute_speed_change / abs(before["frame_speed"]) if before["frame_speed"] else None
    )
    gas_change = abs(after["gas_pressure_mean"] - before["gas_pressure_mean"])
    fresh = result["after_remesh"]["port_adjacent_geometry"]["max_midside_departure_l2"]
    polished = result["after_polish"]["port_adjacent_geometry"]["max_midside_departure_l2"]
    return {
        "scope": "mechanism diagnostics; original scientific acceptance gates are unchanged",
        "relative_frame_speed_change": relative_speed_change,
        "absolute_frame_speed_change": absolute_speed_change,
        "gas_pressure_mean_absolute_change": gas_change,
        "checks": {
            "pressure_below_1e_minus_8": after["lower_pressure_rms"] < 1e-8,
            "fresh_midsides_at_1e_minus_12": fresh <= 1e-12,
            "polished_midsides_at_1e_minus_12": polished <= 1e-12,
            "relative_frame_speed_change_below_1e_minus_6": relative_speed_change is not None
            and relative_speed_change < 1e-6,
            "gas_pressure_change_below_1e_minus_5": gas_change < 1e-5,
        },
    }


def execute(args, prepared, output):
    """Restore, intervene and cold-load; import the solver only after provenance."""
    import numpy as np

    from problems import paper_bubble as model
    from runners import run_paper_case as runner

    case, restart, environment, component, records, implementation = prepared
    require(
        Path(model.__file__).resolve() == ROOT / "problems/paper_bubble.py",
        "model imported elsewhere",
    )

    def emit(event, **payload):
        runner.append_progress(output, {"event": event, "case_id": case["case_id"], **payload})

    def stable_inputs():
        verify_inputs(records)
        verify_source(component, implementation)

    def configure(problem, directory):
        problem.set_output_directory(str(directory))
        problem.set_linear_solver("superlu")
        problem.set_c_compiler("system")
        problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        stable_inputs()
        problem.initialise()

    stable_inputs()
    emit("native-load", status="starting")
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as problem:
        configure(problem, output / "solver")
        problem.load_state(
            str(args.state.resolve()),
            quiet=True,
            additional_info={"exchange_msh_file": restart["mesh_exchange"]},
        )
        loaded = model._stationary_state_signature(problem, True)
        loaded_pins = diagnostic.actual_pins(problem)
        check_loaded_state(loaded, restart, case, loaded_pins)
        preparation = model._prepare_stationary_bc_state(problem, True)
        require(
            loaded_pins == diagnostic.actual_pins(problem), "BC preparation changed actual pins"
        )
        write_json(
            output / "parent-load.json",
            {
                "loaded": loaded,
                "actual_pins": loaded_pins,
                "bc_preparation": preparation,
                "full_parent_fields_restored": True,
            },
        )

        def observe(phase):
            stable_inputs()
            value = sample(problem, model, case, output, phase, np)
            if phase != "after-polish":
                value["retained_state"] = retain_portable_sample(problem, model, output, phase)
            stable_inputs()
            emit(
                "observed",
                phase=phase,
                metrics=value["metrics"],
                max_port_midside_departure=value["port_adjacent_geometry"][
                    "max_midside_departure_l2"
                ],
            )
            return value

        result = remesh_and_polish(problem, model, case, output, observe)
        checkpoint = model._checkpoint_stationary_state(
            problem, case, output, restart["stage"]["bond_index"], True
        )
        packet_path = output / checkpoint["resume_packet"]
        final_restart = runner.validate_restart(
            packet_path,
            case,
            sha(args.case),
            component=component,
            environment=environment,
            azimuthal_stability=True,
        )
        final_pins = diagnostic.actual_pins(problem)
        require(
            final_pins == result["after_polish"]["actual_pins"], "checkpoint changed actual pins"
        )
        check_loaded_state(
            model._stationary_state_signature(problem, True), final_restart, case, final_pins
        )
        stable_inputs()
    # A fresh problem checks the actual saved state; no solve or remesh is permitted here.
    emit("cold-load", status="starting", checkpoint=checkpoint)
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as cold:
        configure(cold, output / "cold-solver")
        runner.verify_restart_artifacts(final_restart)
        cold.load_state(
            final_restart["state"]["path"],
            quiet=True,
            additional_info={"exchange_msh_file": final_restart["mesh_exchange"]},
        )
        cold_signature = model._stationary_state_signature(cold, True)
        cold_pins = diagnostic.actual_pins(cold)
        check_loaded_state(cold_signature, final_restart, case, cold_pins, final_pins)
        cold_preparation = model._prepare_stationary_bc_state(cold, True)
        require(
            cold_pins == diagnostic.actual_pins(cold), "cold BC preparation changed actual pins"
        )
        cold_sample = sample(cold, model, case, output, "cold-load", np)
        cold_checks = canonical_checks(case, cold_sample["metrics"])
        write_json(
            output / "cold-load.json",
            {
                "verified": True,
                "loaded": cold_signature,
                "actual_pins": cold_pins,
                "bc_preparation": cold_preparation,
                "canonical_checks": cold_checks,
                "scope": "fresh native load: complete fields, positions, masks, actual pins, layout, forcing and stationary gates",
            },
        )
    stable_inputs()
    checks = canonical_checks(case, result["after_polish"]["metrics"])
    accepted = all(checks.values()) and all(cold_checks.values())
    result.update(
        schema="private-pressure-remesh-result-v1",
        canonical_checks=checks,
        cold_load={"verified": True, "canonical_checks": cold_checks},
        canonical_gate_authority="unchanged problems/paper_bubble.py:run_model finite-gas predicates",
        causal_predictions=causal_predictions(result),
        checkpoint=checkpoint,
        parents_unchanged=True,
        parent_acceptance_changed=False,
        numerical_acceptance=accepted,
        scientific_status="checks_passed" if accepted else "checks_failed",
    )
    write_json(output / "result.json", result)
    write_json(
        output / "summary.json",
        {
            "status": "completed" if accepted else "failed",
            "completion_reached": accepted,
            "scientific_status": result["scientific_status"],
            "checks": checks,
            "cold_load_verified": True,
            "cold_load_checks": cold_checks,
            **result["after_polish"]["metrics"],
            "checkpoint": checkpoint["resume_packet"],
            "parent_acceptance_changed": False,
            "scope": "one stationary case; no paper-comparison claim",
        },
    )
    require(accepted, "original scientific or cold-load gates failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="metadata preflight only; no native import/output"
    )
    args = parser.parse_args(argv)
    os.umask(0o077)
    prepared = preflight(args)
    case, restart, environment, component, records, implementation = prepared
    require(not args.out.is_symlink(), "output must not be a symlink")
    output = args.out.resolve()
    require(
        not output.exists() or (output.is_dir() and not any(output.iterdir())),
        "output must be new or empty",
    )
    for parent in (ROOT, Path(restart["parent_manifest"]["path"]).parent):
        require(not output.is_relative_to(parent), "output overlaps source/parent runtime")
    if args.check:
        print(
            json.dumps(
                {
                    "verified": True,
                    "native_imported": False,
                    "inputs": records,
                    "implementation": implementation,
                },
                allow_nan=False,
            )
        )
        return 0
    actual = run_manifest(args, case, restart, environment, component, implementation)
    preparation = {
        "schema": "private-pressure-remesh-preparation-v1",
        "inputs": records,
        "runner": actual["runner"],
        "implementation": implementation,
        "case": case,
        "component": component,
        "environment": environment,
        "intervention": actual["intervention"],
        "mesh_exchange": restart["mesh_exchange"],
        "solver": "superlu",
        "compiler": "system",
        "parent_acceptance_changed": False,
        "status": "prepared",
    }
    json.dumps(actual, allow_nan=False)
    json.dumps(preparation, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)
    write_json(output / "run-manifest.json", actual)
    from runners import run_paper_case as runner

    runner.append_progress(
        output,
        {
            "event": "prepared",
            "case_id": case["case_id"],
            "runner": actual["runner"],
            "component_commit": component["commit"],
        },
        first=True,
    )
    try:
        write_json(output / "preparation.json", preparation)
        result = execute(args, prepared, output)
        verify_inputs(records)
        verify_source(component, implementation)
        # Bind all controlled result, sample, polish and portable-state artifacts.
        # Generated solver code/logs stay in the native raw output tree.
        files = {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.relative_to(output).parts[0] not in {"solver", "cold-solver"}
            and path.name != "progress.jsonl"
        }
        write_json(
            output / "completion.json",
            {
                "status": "complete",
                "files": files,
                "numerical_acceptance": result["numerical_acceptance"],
                "parent_acceptance_changed": False,
                "parents_unchanged": True,
                "scope": actual["qualification_scope"],
            },
        )
        runner.append_progress(
            output,
            {
                "event": "final",
                "case_id": case["case_id"],
                "status": "complete",
                "numerical_acceptance": True,
                "parent_acceptance_changed": False,
            },
        )
        return 0
    except BaseException as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "parents_unchanged": diagnostic.parents_unchanged(records),
            "numerical_acceptance": False,
            "parent_acceptance_changed": False,
        }
        write_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", "case_id": case["case_id"], **failure})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
