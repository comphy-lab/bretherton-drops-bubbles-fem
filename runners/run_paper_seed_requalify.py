"""Create a stationary-only descendant with checked native endtime restoration.

The failed producer remains failed. No solve, evolution, remesh, branch,
spectrum, Jacobian or dynamic restart certificate is supplied by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_cold_history as cold

model, runner, diagnostic, remesh = cold.model, cold.runner, cold.diagnostic, cold.remesh
require, sha, write_json = cold.require, cold.sha, cold.write_json
DIAGNOSTIC_SOURCE = "a354f6d364eace3e1c9f397b11e14737c83a98b5"
ENTRYPOINT = "runners/run_paper_seed_requalify.py"
SCHEMA = "stationary-seed-requalification-v1"


def record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "nonregular immutable input")
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def bind_diagnostic(runtime, prepared):
    """Verify the actual diagnostic completion and every direct immutable input."""
    runtime = runtime.resolve(strict=True)
    require(runtime.is_dir() and not (runtime / "failure.json").exists(), "diagnostic failed")
    completion_path = runtime / "completion.json"
    complete = runner._read_json(completion_path)
    require(
        complete["status"] == "complete" and complete["numerical_acceptance"] is True,
        "diagnostic did not complete",
    )
    inventory = complete["files"]
    require(
        set(inventory)
        == {
            str(p.relative_to(runtime))
            for p in runtime.rglob("*")
            if p.is_file() and p.name not in ("completion.json", "progress.jsonl")
        },
        "diagnostic completion inventory differs",
    )
    records = [record(completion_path)]
    for relative, digest in inventory.items():
        path = runtime / relative
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "unsafe diagnostic inventory path",
        )
        require(
            path.resolve(strict=True).is_relative_to(runtime), "diagnostic file escapes runtime"
        )
        item = record(path)
        require(item["sha256"] == digest, "diagnostic completion hash differs")
        records.append(item)
    manifest = runner._read_json(runtime / "run-manifest.json")
    require(
        manifest["component_commit"] == manifest["component"]["commit"] == DIAGNOSTIC_SOURCE,
        "diagnostic source differs",
    )
    runner.require_clean_source(manifest["component"])
    cold.seed.validate_environment(manifest["pyoomph"])
    require(manifest["pyoomph"] == prepared["environment"], "diagnostic environment differs")
    require(
        manifest["case_sha256"] == prepared["restart"]["case_sha256"], "diagnostic case differs"
    )
    case_record = record(manifest["case_path"])
    require(case_record["sha256"] == manifest["case_sha256"], "diagnostic case bytes differ")
    records.append(case_record)
    for key in ("packet", "parent_manifest", "state", "meshes", "stage"):
        require(
            manifest["restart"][key] == prepared["restart"][key], f"diagnostic parent {key} differs"
        )
    producer = hashlib.sha256(
        diagnostic.git(ROOT, "show", f"{DIAGNOSTIC_SOURCE}:{cold.ENTRYPOINT}")
    ).hexdigest()
    require(
        manifest["runner"]["sha256"] == manifest["runner_sha256"] == producer
        and Path(manifest["runner"]["path"]).name == Path(cold.ENTRYPOINT).name,
        "diagnostic producer differs",
    )
    require(
        manifest["intervention"]
        == {
            "kind": "cold-load-run-endtime-metadata-only",
            "api": cold.API,
            "solve_calls": 0,
            "remesh_calls": 0,
            "time_evolution": False,
            "parent_acceptance_changed": False,
        },
        "diagnostic intervention differs",
    )
    preparation = runner._read_json(runtime / "preparation.json")
    remesh.verify_inputs(preparation["inputs"])
    require(
        {
            (r["path"], r["sha256"])
            for r in prepared["records"]
            if r["sha256"] != prepared["restart"]["case_sha256"]
        }
        <= {(r["path"], r["sha256"]) for r in preparation["inputs"]},
        "diagnostic direct parent inputs differ",
    )
    records.extend(preparation["inputs"])
    for relative, current in prepared["implementation"].items():
        if relative == ENTRYPOINT:
            continue
        digest = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{DIAGNOSTIC_SOURCE}:{relative}")
        ).hexdigest()
        require(
            current["sha256"]
            == digest
            == manifest["implementation"][relative]["sha256"]
            == preparation["implementation"][relative]["sha256"],
            f"diagnostic frozen helper differs: {relative}",
        )
    for relative, current in prepared["component"]["sources"].items():
        previous = manifest["component"]["sources"][relative]
        require(
            previous["committed_matches"] is True
            and current["sha256"] == previous["sha256"] == previous["committed_sha256"],
            "diagnostic model source differs",
        )
    saved = Path(prepared["restart"]["state"]["path"]).read_bytes()
    expected = prepared["restart"]["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
    snapshots = {}
    for phase in ("before", "after", "after-sampling"):
        retention = runner._read_json(runtime / f"{phase}-retention.json")
        require(
            retention["state"]
            == {**record(runtime / f"{phase}.dump"), "format": "memory_snapshot"},
            "diagnostic retained state identity differs",
        )
        remesh.verify_inputs([retention["state"], *retention["meshes"]])
        records.extend(retention["meshes"])
        raw = (runtime / f"{phase}.dump").read_bytes()
        base = Path(retention["mesh_reference_base"])
        require(
            set(retention["mesh_exchange"])
            == {ref for _, _, ref in cold.serialized_metadata(raw)["meshes"]},
            "diagnostic retained mesh inventory differs",
        )
        for mesh in retention["meshes"]:
            require(
                retention["mesh_exchange"][mesh["stored_path"]] == mesh["path"]
                and sha(base / mesh["stored_path"]) == mesh["sha256"],
                "diagnostic original/retained mesh differs",
            )
            records.append(record(base / mesh["stored_path"]))
        snapshots[phase] = (raw, base)
    before, base = snapshots["before"]
    after, after_base = snapshots["after"]
    require(base == after_base, "diagnostic snapshot bases differ")
    proof = cold.verify_snapshots(
        saved, before, after, Path(prepared["restart"]["mesh_reference_base"]), base, expected
    )
    mechanism = runner._read_json(runtime / "mechanism.json")
    require(
        mechanism == {**proof, "api": cold.API, "parent_acceptance_changed": False},
        "diagnostic mechanism receipt differs",
    )
    require(snapshots["after-sampling"][0] == after, "diagnostic sampling changed bytes")
    observations = [
        runner._read_json(runtime / f"{p}-observation.json") for p in ("before", "after")
    ]
    require(
        observations[0]["actual_pins"]
        == observations[1]["actual_pins"]
        == prepared["sample"]["actual_pins"],
        "diagnostic pins differ",
    )
    for phase, observation in zip(("before", "after"), observations, strict=True):
        raw, observation_base = snapshots[phase]
        require(
            observation["signature"]["serialized_state"]
            == runner.snapshot_bc_cache_signature(raw, mesh_reference_base=observation_base),
            "diagnostic observed fingerprint differs",
        )
        remesh.check_loaded_state(
            observation["signature"],
            prepared["restart"],
            prepared["case"],
            observation["actual_pins"],
            prepared["sample"]["actual_pins"],
        )
        require(
            observation["time_history"]
            == {
                k: prepared["metadata"][k]
                for k in ("time", "dts", "unsteady_steps", "suggested_next_dt")
            },
            "diagnostic live time history differs",
        )
    require(
        observations[1]["restored_endtime"] == prepared["metadata"]["endtime"]
        and observations[1]["signature"]["serialized_state"]["payload_sha256"] == expected,
        "diagnostic restored state differs",
    )
    sample = runner._read_json(runtime / "samples/cold-load.json")
    require(
        sample["sampling_preserved_state"] is True
        and sample["actual_pins"] == prepared["sample"]["actual_pins"]
        and sample["signature"]["serialized_state"]["payload_sha256"] == expected,
        "diagnostic sampled state differs",
    )
    remesh.check_loaded_state(
        sample["signature"],
        prepared["restart"],
        prepared["case"],
        sample["actual_pins"],
        prepared["sample"]["actual_pins"],
    )
    checks = remesh.canonical_checks(prepared["case"], sample["metrics"])
    gates = runner._read_json(runtime / "numerical-gates.json")
    result = runner._read_json(runtime / "result.json")
    summary = runner._read_json(runtime / "summary.json")
    require(
        all(checks.values())
        and gates["checks"] == result["checks"] == checks
        and gates["sampling_preserved_complete_state"] is True
        and result["mechanism_verified"] is True
        and result["numerical_acceptance"] is True
        and result["parent_status"] == "failed"
        and result["parent_acceptance_changed"] is False
        and summary["status"] == "completed"
        and summary["completion_reached"] is True,
        "diagnostic cold gates or terminal receipts differ",
    )
    return {
        "manifest": record(runtime / "run-manifest.json"),
        "completion": record(completion_path),
        "source": DIAGNOSTIC_SOURCE,
    }, records


def preflight(args):
    prepared = cold.preflight(args)
    diagnostic.git(
        ROOT, "merge-base", "--is-ancestor", DIAGNOSTIC_SOURCE, prepared["component"]["commit"]
    )
    entry = record(ROOT / ENTRYPOINT)
    committed = hashlib.sha256(
        diagnostic.git(ROOT, "show", f"{prepared['component']['commit']}:{ENTRYPOINT}")
    ).hexdigest()
    require(entry["sha256"] == committed, "requalification entrypoint is not committed")
    prepared["implementation"][ENTRYPOINT] = entry
    lineage, records = bind_diagnostic(args.diagnostic_runtime, prepared)
    prepared["records"].extend(records)
    prepared["lineage"] = {
        "schema": SCHEMA,
        "scope": "stationary-only descendant",
        "original_failed_producer": prepared["restart"]["parent_manifest"],
        "original_packet": prepared["restart"]["packet"],
        "original_state": prepared["restart"]["state"],
        "completed_diagnostic": lineage,
        "parent_status": "failed",
        "parent_acceptance_changed": False,
        "consumer": {
            "module": ENTRYPOINT,
            "callable": "restore_checked",
            "sha256": entry["sha256"],
            "saved_endtime": prepared["metadata"]["endtime"],
            "api": cold.API,
        },
    }
    cold.stable_inputs(prepared)
    return prepared


def snapshot(problem):
    setting = problem.eigen_data_in_states
    try:
        problem.eigen_data_in_states = True
        return problem._snapshot_state()
    finally:
        problem.eigen_data_in_states = setting


def prepare_bc(problem, expected_history, expected_pins):
    before = model._stationary_state_signature(problem, True)
    receipt = model._prepare_stationary_bc_state(problem, True)
    after = model._stationary_state_signature(problem, True)

    def physical(signature):
        return {
            **{
                k: v
                for k, v in signature.items()
                if k not in ("serialized_state", "bc_cache_marker")
            },
            "payload_sha256": signature["serialized_state"]["payload_sha256"],
        }

    require(
        receipt["verified"] is True
        and physical(before) == physical(after)
        and after["serialized_state"]["payload_sha256"] == expected_history,
        "BC preparation changed full stationary state/history",
    )
    require(diagnostic.actual_pins(problem) == expected_pins, "BC preparation changed actual pins")
    return receipt


def restore_checked(problem, case, restart, expected_pins, expected_history):
    """Anchor-compatible typed load; preserve raw proof even on refusal.

    Repeated warm calls may retain the exact writer scalar. A changed finite
    value is refused; only an unset writer may receive the parsed saved value.
    """
    runner.verify_restart_artifacts(restart)
    saved = Path(restart["state"]["path"]).read_bytes()
    metadata = cold.serialized_metadata(saved)
    expected = metadata["endtime"]
    require(
        isinstance(expected, float)
        and math.isfinite(expected)
        and expected == case["solve"]["relax_time"],
        "saved endtime differs from consumer case",
    )
    require(
        runner.snapshot_bc_cache_signature(
            saved, mesh_reference_base=Path(restart["mesh_reference_base"])
        )["payload_sha256"]
        == expected_history,
        "saved complete history differs",
    )
    base = Path(
        getattr(problem, "_seed_requalification_receipts_directory", problem.get_output_directory())
    )
    base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="typed-load-", dir=base))
    problem.load_state(
        restart["state"]["path"],
        quiet=True,
        additional_info={"exchange_msh_file": restart["mesh_exchange"]},
    )
    before = cold.retain_snapshot(problem, output, "before")
    changed = False
    try:
        native_metadata = cold.serialized_metadata(before)
        write_json(
            output / "load-metadata.json",
            {
                "saved_endtime": expected,
                "saved_offset": metadata["endtime_offset"],
                "native_offset": native_metadata["endtime_offset"],
                "native_serialized_endtime": repr(native_metadata["endtime"]),
                "native_loaded_endtime": repr(vars(problem).get("_loaded_run_statement_endtime")),
                "native_writer_endtime": repr(vars(problem).get("_run_statement_endtime")),
                "saved_state": restart["state"],
                "expected_history": expected_history,
                "before": record(output / "before.dump"),
            },
        )
        loaded = vars(problem).get("_loaded_run_statement_endtime")
        require(
            isinstance(loaded, float) and struct.pack("<d", loaded) == struct.pack("<d", expected),
            "native loaded endtime differs from serialized scalar",
        )
        require("_run_statement_endtime" in vars(problem), "native writer endtime unavailable")
        writer = problem._run_statement_endtime
        if writer is None:
            offset = cold.serialized_metadata(before)["endtime_offset"]
            # This byte prediction is compared in memory, never written as state.
            prediction = before[:offset] + struct.pack("<d", expected) + before[offset + 8 :]
            require(
                cold.normalized_bytes(prediction, Path.cwd())
                == cold.normalized_bytes(saved, Path(restart["mesh_reference_base"])),
                "nonendtime bytes differ before restoration",
            )
            cold.restore_endtime(problem, saved, before, case)
            changed = True
        else:
            require(
                isinstance(writer, float)
                and struct.pack("<d", writer) == struct.pack("<d", expected),
                "retained writer endtime differs from saved scalar",
            )
    finally:
        after = cold.retain_snapshot(problem, output, "after")
        write_json(
            output / "raw-intervention.json",
            {
                "applied": changed,
                "before": record(output / "before.dump"),
                "after": record(output / "after.dump"),
            },
        )
    if changed:
        proof = cold.verify_snapshots(
            saved, before, after, Path(restart["mesh_reference_base"]), Path.cwd(), expected_history
        )
    else:
        require(before == after, "equal writer load changed raw bytes")
        require(
            cold.normalized_bytes(after, Path.cwd())
            == cold.normalized_bytes(saved, Path(restart["mesh_reference_base"])),
            "complete native bytes differ from saved",
        )
        proof = {"verified": True, "only_changed_slot": None}
    intervention = {
        **proof,
        "kind": "typed-native-endtime-restoration",
        "applied": changed,
        "saved_endtime": expected,
        "saved_offset": metadata["endtime_offset"],
        "native_offset": cold.serialized_metadata(before)["endtime_offset"],
        "before": record(output / "before.dump"),
        "after": record(output / "after.dump"),
        "api": cold.API,
    }
    write_json(output / "intervention.json", intervention)
    signature = model._stationary_state_signature(problem, True)
    pins = diagnostic.actual_pins(problem)
    remesh.check_loaded_state(signature, restart, case, pins, expected_pins)
    require(
        signature["serialized_state"]["payload_sha256"] == expected_history,
        "cold-loaded complete history differs",
    )
    time_history = cold.check_time_history(problem, metadata)
    try:
        preparation = prepare_bc(problem, expected_history, pins)
    finally:
        cold.retain_snapshot(problem, output, "after-bc")
    return {
        "verified": True,
        "loaded": signature,
        "actual_pins": pins,
        "bc_preparation": preparation,
        "time_history": time_history,
        "intervention": intervention,
    }


def configure(problem, directory):
    problem.set_output_directory(str(directory))
    problem._seed_requalification_receipts_directory = directory.parent / (
        directory.name + "-loads"
    )
    problem.set_linear_solver("superlu")
    problem.set_c_compiler("system")
    problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
    problem.initialise()


def sample_checked(problem, case, output, phase, np):
    before = cold.retain_snapshot(problem, output, phase + "-before-sampling")
    try:
        sample = remesh.sample(problem, model, case, output, phase, np)
    finally:
        after = cold.retain_snapshot(problem, output, phase + "-after-sampling")
    require(before == after, "five-gate sampling changed complete native bytes")
    checks = remesh.canonical_checks(case, sample["metrics"])
    require(all(checks.values()), "original five stationary gates failed")
    return checks


def execute(args, prepared, output):
    import numpy as np

    case, restart = prepared["case"], prepared["restart"]
    expected = restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
    pins = prepared["sample"]["actual_pins"]
    cold.stable_inputs(prepared)
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as problem:
        configure(problem, output / "solver")
        receipt = restore_checked(problem, case, restart, pins, expected)
        write_json(output / "parent-cold-load.json", receipt)
        checks = sample_checked(problem, case, output, "requalified", np)
        before = cold.retain_snapshot(problem, output, "before-checkpoint")
        try:
            checkpoint = model._checkpoint_stationary_state(
                problem, case, output, restart["stage"]["bond_index"], True
            )
        finally:
            after = cold.retain_snapshot(problem, output, "after-checkpoint")
        require(
            cold.normalized_bytes(before, Path.cwd()) == cold.normalized_bytes(after, Path.cwd()),
            "checkpoint changed complete state",
        )
        require(diagnostic.actual_pins(problem) == pins, "checkpoint changed actual pins")
        packet = output / checkpoint["resume_packet"]
        final = runner.validate_restart(
            packet,
            case,
            sha(args.case),
            component=prepared["component"],
            environment=prepared["environment"],
            azimuthal_stability=True,
        )
        require(
            final["parent_manifest"] == record(output / "run-manifest.json")
            and final["stage"]["kind"] == "completed_polish"
            and final["stage"]["bond_index"] == restart["stage"]["bond_index"]
            and Path(final["state"]["path"]).is_relative_to(output)
            and final["roundtrip"]["after"]["serialized_state"]["payload_sha256"] == expected,
            "descendant packet identity or history differs",
        )
        require(
            cold.normalized_bytes(
                Path(final["state"]["path"]).read_bytes(), Path(final["mesh_reference_base"])
            )
            == cold.normalized_bytes(
                Path(restart["state"]["path"]).read_bytes(), Path(restart["mesh_reference_base"])
            ),
            "descendant portable state differs from original",
        )
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as fresh:
        configure(fresh, output / "cold-solver")
        receipt = restore_checked(fresh, case, final, pins, expected)
        cold_checks = sample_checked(fresh, case, output, "cold-load", np)
        write_json(
            output / "cold-load.json",
            {
                **receipt,
                "checks": cold_checks,
                "packet": final["packet"],
                "parent_manifest": final["parent_manifest"],
            },
        )
    cold.stable_inputs(prepared)
    runner.verify_restart_artifacts(final)
    return {
        "schema": SCHEMA,
        "numerical_acceptance": True,
        "checks": checks,
        "cold_load_checks": cold_checks,
        "cold_load_verified": True,
        "checkpoint": checkpoint,
        "lineage": prepared["lineage"],
        "parent_status": "failed",
        "parent_acceptance_changed": False,
        "branch_identity_verified": False,
        "eigenanalysis_performed": False,
        "jacobian_certificate": False,
        "dynamic_restart_certificate": False,
        "full_comparison_verified": False,
    }


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "diagnostic-runtime", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    require(not args.out.is_symlink(), "output must not be a symlink")
    output = runner.validate_output_path(args.out, ROOT)
    for evidence in (args.manifest.resolve().parent, args.diagnostic_runtime.resolve()):
        require(
            not output.is_relative_to(evidence) and not evidence.is_relative_to(output),
            "output overlaps immutable evidence",
        )
    if args.check:
        print('{"verified": true, "native_imported": false, "output_created": false}')
        return 0
    actual = remesh.run_manifest(
        args,
        prepared["case"],
        prepared["restart"],
        prepared["environment"],
        prepared["component"],
        prepared["implementation"],
    )
    entry = prepared["implementation"][ENTRYPOINT]
    actual.update(
        runner=entry,
        runner_path=entry["path"],
        runner_sha256=entry["sha256"],
        argv=[sys.executable, str(ROOT / ENTRYPOINT), *arguments],
        lineage=prepared["lineage"],
        case_fields_scope="unchanged construction and five gates; metadata-only descendant",
        qualification_scope="stationary-only descendant; full history and fresh typed cold load",
        intervention={
            "kind": "stationary-seed-requalification",
            "solve_calls": 0,
            "remesh_calls": 0,
            "time_evolution": False,
            "api": cold.API,
            "parent_acceptance_changed": False,
        },
    )
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    runner.create_run_manifest(output, actual)
    runner.append_progress(output, {"event": "prepared", "runner": entry}, first=True)
    try:
        write_json(
            output / "preparation.json",
            {
                "inputs": prepared["records"],
                "implementation": prepared["implementation"],
                "lineage": prepared["lineage"],
            },
        )
        write_json(
            output / "schema.json",
            {
                "schema": SCHEMA,
                "scope": actual["qualification_scope"],
                "consumer": prepared["lineage"]["consumer"],
            },
        )
        result = execute(args, prepared, output)
        cold.stable_inputs(prepared)
        write_json(output / "result.json", result)
        write_json(
            output / "summary.json",
            {
                **result,
                "status": "completed",
                "scientific_status": "checks_passed",
                "completion_reached": True,
            },
        )
        write_json(
            output / "completion.json",
            {
                "schema": SCHEMA,
                "status": "complete",
                "numerical_acceptance": True,
                "files": {
                    str(p.relative_to(output)): sha(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file() and p.name != "progress.jsonl"
                },
            },
        )
        runner.append_progress(output, {"event": "final", "numerical_acceptance": True})
        return 0
    except BaseException as error:
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "numerical_acceptance": False,
            "inputs_unchanged": diagnostic.parents_unchanged(prepared["records"]),
            "parent_status": "failed",
            "parent_acceptance_changed": False,
        }
        write_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", **failure})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
