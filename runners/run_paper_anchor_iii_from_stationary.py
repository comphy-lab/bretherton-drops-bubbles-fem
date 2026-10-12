"""Continue one qualified stationary III candidate to the exact Ca=-0.0038 sample.

The three declared steps and original stationary gates are fixed. Native history
is preserved without an endtime intervention. No branch or spectrum is certified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_anchor as anchor
from runners import run_paper_cold_history as history

model, runner, diagnostic, remesh = anchor.model, anchor.runner, anchor.diagnostic, anchor.remesh
require, sha, write_json = diagnostic.require, diagnostic.sha, diagnostic.write_json
ENTRYPOINT = "runners/run_paper_anchor_iii_from_stationary.py"
SCHEMA = "stationary-III-anchor-from-stationary-v1"
KIND = "stationary_III_candidate_from_stationary"
SCOPE = "stationary III candidate at exact Ca=-0.0038; no branch identity, spectrum or paper claim"
PARENT_SOURCE = "958f7b6a142be27ba5779609ebadf1bdc0316e6d"
PARENT_CASE_SHA = "f340071e262f107a417a308dd46739465e5d88d9973bd8947b3cba9870bbbc23"
PARENT_COMPLETION_SHA = "52e8d48e00127213db99b2cbfd24f0e715ddbeb6f73dccfe5ce6705523c09ded"
PARENT_STATE_SHA = "2286652f331ea6ee398727e3503931c5a7638a8e7b0c9f87478915a312680d51"
FROZEN_BASE = "6e23bf3fb281a8b070a3d3b11b927be58b3d56de"
PARENT_CA, TARGET_CA = -0.00315, -0.0038
LADDER = (-0.0035, -0.00375, -0.0038)
OLD_HELPERS = (
    "runners/run_paper_anchor.py",
    "runners/run_pressure_remesh.py",
    "runners/run_pressure_diagnostic.py",
)
NEW_HELPERS = ("runners/run_paper_cold_history.py", "runners/run_paper_seed_i.py")
HELPERS = (*OLD_HELPERS, *NEW_HELPERS)
GATES = anchor.GATES
NO_CLAIMS = {
    "branch_identity_verified": False,
    "eigenanalysis_performed": False,
    "jacobian_certificate": False,
    "dynamic_restart_certificate": False,
    "full_comparison_verified": False,
}


def regular_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular authority required")
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def relative_path(value):
    require(isinstance(value, str) and value, "artifact path must be text")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe artifact path")
    return path


def physical_signature(value):
    return {
        **{
            key: item
            for key, item in value.items()
            if key not in {"serialized_state", "bc_cache_marker"}
        },
        "payload_sha256": value["serialized_state"]["payload_sha256"],
    }


def completion_records(runtime, files):
    require(isinstance(files, dict), "completion inventory required")
    records = []
    for name, expected in files.items():
        record = regular_record(runtime / relative_path(name))
        require(
            Path(record["path"]).is_relative_to(runtime) and record["sha256"] == expected,
            "completion artifact differs or escaped runtime",
        )
        records.append(record)
    return records


def bound(record, runtime, files):
    actual = regular_record(record["path"])
    path = Path(actual["path"])
    require(
        path.is_relative_to(runtime)
        and actual["sha256"] == record["sha256"]
        and files.get(str(path.relative_to(runtime))) == actual["sha256"],
        "direct proof unbound or outside producer custody",
    )
    return actual


def source_contract(parent):
    component = runner.component_revision(ROOT)
    runner.require_clean_source(component)
    runner.require_clean_source(parent["component"])
    require(
        parent["component_commit"] == parent["component"]["commit"] == PARENT_SOURCE,
        "only the qualified 958 III source is admitted",
    )
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", PARENT_SOURCE, component["commit"])
    require(
        set(parent["component"]["sources"]) == set(component["sources"]), "method inventory differs"
    )
    for name, old in parent["component"]["sources"].items():
        expected = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{PARENT_SOURCE}:{name}")
        ).hexdigest()
        require(
            old["committed_matches"] is True
            and expected
            == old["sha256"]
            == old["committed_sha256"]
            == component["sources"][name]["sha256"]
            == sha(ROOT / relative_path(name)),
            "frozen method changed: " + name,
        )
    implementation = {}
    for name in (*HELPERS, ENTRYPOINT):
        actual = regular_record(ROOT / name)
        expected = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{component['commit']}:{name}")
        ).hexdigest()
        require(actual["sha256"] == expected, "implementation differs from committed source")
        if name in OLD_HELPERS:
            old = parent["implementation"][name]
            require(
                old["sha256"]
                == expected
                == hashlib.sha256(
                    diagnostic.git(ROOT, "show", f"{PARENT_SOURCE}:{name}")
                ).hexdigest(),
                "qualified III helper changed",
            )
        elif name in NEW_HELPERS:
            require(
                expected
                == hashlib.sha256(
                    diagnostic.git(ROOT, "show", f"{FROZEN_BASE}:{name}")
                ).hexdigest(),
                "reviewed history helper changed",
            )
        implementation[name] = actual
    return component, implementation


def parent_evidence(runtime, parent, restart, case):
    require(
        parent.get("schema") == "pyoomph-runner-manifest-v1"
        and Path(parent["runner"]["path"]).name == "run_paper_anchor.py"
        and parent["anchor_stage"] == {"kind": "stationary_III_anchor"}
        and parent["parent_acceptance_changed"] is False
        and parent["azimuthal_stability"] is True,
        "qualified III producer discriminator differs",
    )
    completion_path = runtime / "completion.json"
    require(
        sha(completion_path) == PARENT_COMPLETION_SHA
        and restart["state"]["sha256"] == PARENT_STATE_SHA,
        "the exact qualified III completion/state is required",
    )
    require(not (runtime / "failure.json").exists(), "parent has a failure receipt")
    complete = runner._read_json(completion_path)
    require(
        complete.get("status") == "complete" and complete.get("numerical_acceptance") is True,
        "parent is not numerically qualified",
    )
    files = complete["files"]
    require(
        {"run-manifest.json", "result.json", "summary.json", "cold-load.json"} <= set(files),
        "parent qualification files unbound",
    )
    records = [regular_record(completion_path), *completion_records(runtime, files)]
    result, summary, cold = [
        runner._read_json(runtime / name)
        for name in ("result.json", "summary.json", "cold-load.json")
    ]
    checks = (
        result["canonical_checks"],
        result["cold_load_checks"],
        summary["checks"],
        summary["cold_load_checks"],
        cold["canonical_checks"],
    )
    require(
        result["schema"] == "private-paper-stationary-anchor-v1"
        and result["numerical_acceptance"] is True
        and result["completion_reached"] is True
        and result["parent_acceptance_changed"] is False
        and summary["completion_reached"] is True
        and summary["cold_load_verified"] is True
        and cold["verified"] is True
        and all(
            set(item) == GATES and all(value is True for value in item.values()) for item in checks
        )
        and all(remesh.canonical_checks(case, summary).values()),
        "parent original warm/cold numerical gates failed",
    )
    for value in (result, summary):
        require(
            value["branch_identity_verified"] is False
            and value["eigenanalysis_performed"] is False,
            "parent claims branch or eigen qualification",
        )
    require(
        summary["ca_in"] == result["ca_in"] == PARENT_CA
        and Path(restart["parent_manifest"]["path"]) == runtime / "run-manifest.json"
        and runtime / relative_path(summary["checkpoint"]) == Path(restart["packet"]["path"]),
        "parent checkpoint identity differs",
    )
    records.extend(
        bound(item, runtime, files)
        for item in (restart["packet"], restart["state"], *restart["meshes"])
    )
    expected = restart["roundtrip"]["after"]
    saved = Path(restart["state"]["path"]).read_bytes()
    require(
        history.serialized_metadata(saved)["endtime"] is None,
        "qualified III unset writer marker differs",
    )
    require(
        runner.snapshot_bc_cache_signature(
            saved, mesh_reference_base=Path(restart["mesh_reference_base"])
        )["payload_sha256"]
        == expected["serialized_state"]["payload_sha256"]
        and physical_signature(cold["loaded"]) == physical_signature(expected)
        and physical_signature(result["terminal"]["after_polish"]["signature"])
        == physical_signature(expected)
        and cold["actual_pins"] == result["terminal"]["after_polish"]["actual_pins"]
        and cold["bc_preparation"]["verified"] is True,
        "parent full fields/history/pins or portable state differs",
    )
    remesh.check_loaded_state(cold["loaded"], restart, case, cold["actual_pins"])
    return cold, records


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "runner must use its isolated source")
    for module, name in (
        (anchor, OLD_HELPERS[0]),
        (model, "problems/paper_bubble.py"),
        (runner, "runners/run_paper_case.py"),
        (remesh, OLD_HELPERS[1]),
        (diagnostic, OLD_HELPERS[2]),
        (history, NEW_HELPERS[0]),
        (history.seed, NEW_HELPERS[1]),
    ):
        require(Path(module.__file__).resolve() == ROOT / name, "helper imported elsewhere")
    parent_path = Path(regular_record(args.manifest)["path"])
    require(parent_path.name == "run-manifest.json", "canonical parent manifest required")
    parent = runner._read_json(parent_path)
    component, implementation = source_contract(parent)
    environment = runner.pyoomph_revision()
    runner.require_pinned_environment(environment)
    runner.require_pinned_environment(parent["pyoomph"])
    require(
        environment["commit"] == parent["pyoomph"]["commit"] == diagnostic.PYOOMPH
        and environment["native_core"]["sha256"]
        == parent["pyoomph"]["native_core"]["sha256"]
        == diagnostic.CORE
        and environment["interpreter"]["resolved_executable"]
        == parent["pyoomph"]["interpreter"]["resolved_executable"],
        "parent/current pin, core or interpreter differs",
    )
    case_path = Path(regular_record(parent["case_path"])["path"])
    require(
        parent["case_file"] == str(case_path)
        and sha(case_path) == parent["case_sha256"] == PARENT_CASE_SHA,
        "qualified III case identity differs",
    )
    original, target = runner.read_case(case_path), runner.read_case(args.case)
    contract = anchor.transition_contract(original, target)
    require(
        original["physics"]["ca_in"] == PARENT_CA
        and target["physics"]["ca_in"] == TARGET_CA
        and original["physics"]["problem"] == "finite_gas_bubble"
        and original["physics"]["bond"] == 1.56
        and original["solve"]["max_newton_iterations"]
        == target["solve"]["max_newton_iterations"]
        == 40,
        "declared III target or inherited stage budget differs",
    )
    values = anchor.read_ladder(args.ca_ladder, original, target)
    require(values == list(LADDER), "only the three fixed monotone III steps are admitted")
    restart = runner.validate_restart(
        args.packet,
        original,
        sha(case_path),
        component=component,
        environment=environment,
        azimuthal_stability=True,
    )
    require(
        restart["parent_manifest"] == regular_record(parent_path)
        and Path(restart["state"]["path"]) == args.state.resolve(strict=True)
        and restart["state"]["format"] == "portable_file"
        and restart["stage"]["kind"] == "completed_polish"
        and restart["stage"]["bond_index"] == 16
        and restart["stage"]["bond"] == 1.56
        and restart["stage"]["ca_in"] == PARENT_CA,
        "qualified III portable target packet differs",
    )
    cold, records = parent_evidence(parent_path.parent, parent, restart, original)
    records.extend(
        (regular_record(case_path), regular_record(args.case), regular_record(args.ca_ladder))
    )
    contract.update(
        parent_case=regular_record(case_path),
        target_case=regular_record(args.case),
        ca_ladder={**regular_record(args.ca_ladder), "values": values},
        maximum_ca_steps=3,
        per_mesh_stage_newton_cap=40,
        per_polish_total_newton_cap=160,
        per_polish_remesh_bound=3,
        total_newton_update_bound=640,
        explicit_terminal_remesh_calls=1,
    )
    return {
        "parent": original,
        "target": target,
        "restart": restart,
        "parent_cold": cold,
        "component": component,
        "environment": environment,
        "implementation": implementation,
        "records": records,
        "contract": contract,
        "values": values,
    }


def manifest(case_path, case, prepared, stage, argv):
    actual = anchor.manifest(case_path, case, prepared, stage)
    own = prepared["implementation"][ENTRYPOINT]
    actual.update(
        runner=own,
        runner_path=own["path"],
        runner_sha256=own["sha256"],
        argv=[sys.executable, str(ROOT / ENTRYPOINT), *argv],
        qualification_schema=SCHEMA,
        qualification_scope=SCOPE,
        branch_identity_verified=False,
        eigenanalysis_performed=False,
    )
    return actual


def restore_checked(problem, case, restart, expected_pins, expected_history):
    """Use the established III loader; retain its unchanged marker and full history."""
    base = Path(problem._stationary_iii_receipts_directory)
    base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="stationary-load-", dir=base))
    saved_metadata = history.serialized_metadata(Path(restart["state"]["path"]).read_bytes())
    require(saved_metadata["endtime"] is None, "III writer marker must remain unset")
    receipt = anchor.restore_checked(problem, case, restart, expected_pins, expected_history)
    time_history = history.check_time_history(problem, saved_metadata)
    raw = history.retain_snapshot(problem, output, "after-bc")
    fingerprint = runner.snapshot_bc_cache_signature(raw, mesh_reference_base=Path.cwd())
    require(
        fingerprint == receipt["bc_preparation"]["after"]["serialized_state"]
        and fingerprint["payload_sha256"] == expected_history
        and history.serialized_metadata(raw)["endtime"] is None,
        "III restoration changed full history or writer marker",
    )
    return {
        **receipt,
        "schema": "preserved-stationary-III-history-load-v1",
        "history_intervention": False,
        "writer_endtime": None,
        "time_history": time_history,
        "raw_after_bc": regular_record(output / "after-bc.dump"),
        "retention": regular_record(output / "after-bc-retention.json"),
        "scope": "stationary full-history preservation; no dynamic restart certificate",
    }


def validate_load(receipt, restart, case, pins, expected, runtime, files):
    require(
        receipt["schema"] == "preserved-stationary-III-history-load-v1"
        and receipt["verified"] is True
        and receipt["history_intervention"] is False
        and receipt["writer_endtime"] is None
        and receipt["actual_pins"] == pins,
        "III load type, preservation or pins differ",
    )
    remesh.check_loaded_state(receipt["loaded"], restart, case, pins, pins)
    metadata = history.serialized_metadata(Path(restart["state"]["path"]).read_bytes())
    require(
        metadata["endtime"] is None
        and receipt["time_history"]
        == {key: metadata[key] for key in ("time", "dts", "unsteady_steps", "suggested_next_dt")},
        "III time/dt history or marker differs",
    )
    records = [
        bound(receipt["raw_after_bc"], runtime, files),
        bound(receipt["retention"], runtime, files),
    ]
    retention = runner._read_json(Path(receipt["retention"]["path"]))
    state_path = Path(receipt["raw_after_bc"]["path"])
    require(
        retention["state"] == {**regular_record(state_path), "format": "memory_snapshot"}
        and [item["stored_path"] for item in retention["meshes"]]
        == runner.state_mesh_references(state_path)
        and retention["mesh_exchange"]
        == {item["stored_path"]: item["path"] for item in retention["meshes"]},
        "III raw state/retained mesh closure differs",
    )
    base = Path(retention["mesh_reference_base"])
    records.extend(bound(item, runtime, files) for item in retention["meshes"])
    require(
        all(sha(base / item["stored_path"]) == item["sha256"] for item in retention["meshes"]),
        "III snapshot mesh content differs",
    )
    fingerprint = runner.snapshot_bc_cache_signature(
        state_path.read_bytes(), mesh_reference_base=base
    )
    preparation = receipt["bc_preparation"]
    require(
        preparation["verified"] is True
        and preparation["method"] == "actions_before_stationary_solve(force_reassign_eqs=False)"
        and preparation["before"] == receipt["loaded"]
        and physical_signature(preparation["before"]) == physical_signature(preparation["after"])
        and fingerprint == preparation["after"]["serialized_state"]
        and fingerprint["payload_sha256"]
        == expected
        == receipt["loaded"]["serialized_state"]["payload_sha256"]
        and history.serialized_metadata(state_path.read_bytes())["endtime"] is None,
        "III BC preparation or complete raw history differs",
    )
    return records


def polish_bounds(polish):
    return polish.get("status") == "converged" and all(
        type(polish.get(key)) is int and 0 <= polish[key] <= maximum
        for key, maximum in (("stage_updates", 40), ("updates_total", 160), ("remeshes_total", 3))
    )


def stage(problem, prepared, previous, index, value, output, argv, np):
    current = anchor.stage_case(prepared["target"], value, index, index == len(LADDER))
    directory = output / "stages" / f"ca_{index:03d}"
    (directory / "solver").mkdir(parents=True)
    case_path = directory / "case.json"
    write_json(case_path, current)
    write_json(
        directory / "run-manifest.json",
        manifest(
            case_path,
            current,
            prepared,
            {"kind": "stationary_III_monotone_step", "index": index, "ca_in": value},
            argv,
        ),
    )
    runner.append_progress(
        directory, {"event": "prepared", "case_id": current["case_id"]}, first=True
    )
    transition = anchor.adopt_case(problem, previous, current, directory)
    polish = model.polish_confined_state(
        problem, current, directory, "ca_step", initial_relaxation=1.0
    )
    require(polish_bounds(polish), "III stage exceeds inherited polish contract")
    sample = remesh.sample(problem, model, current, directory, "after-polish", np)
    checkpoint = model._checkpoint_stationary_state(problem, current, directory, 16, True)
    restart = runner.validate_restart(
        directory / checkpoint["resume_packet"],
        current,
        sha(case_path),
        component=prepared["component"],
        environment=prepared["environment"],
        azimuthal_stability=True,
    )
    require(sample["actual_pins"] == diagnostic.actual_pins(problem), "III checkpoint changed pins")
    record = {
        "index": index,
        "ca_in": value,
        "case": regular_record(case_path),
        "transition": transition,
        "polish": polish,
        "checkpoint": checkpoint,
        "resume_packet_sha256": restart["packet"]["sha256"],
        "canonical_checks_observed": remesh.canonical_checks(current, sample["metrics"]),
        "numerical_acceptance": False,
        "scope": "provisional monotone III stage",
    }
    write_json(directory / "stage.json", record)
    return current, record


def execute(args, prepared, output, argv):
    import numpy as np

    parent, target = prepared["parent"], prepared["target"]
    restart = prepared["restart"]
    stages = []
    with model.build_problem(parent, initial_bond=parent["physics"]["bond"]) as problem:
        problem._stationary_iii_receipts_directory = output / "loads"
        anchor.configure(problem, output, prepared)
        restored = restore_checked(
            problem,
            parent,
            restart,
            prepared["parent_cold"]["actual_pins"],
            restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
        )
        observation = output / "parent-observation"
        observation.mkdir()
        initial = remesh.sample(problem, model, parent, observation, "cold-load", np)
        require(
            all(remesh.canonical_checks(parent, initial["metrics"]).values()),
            "III parent fresh-load gates failed",
        )
        write_json(output / "parent-load.json", restored)
        previous = parent
        for index, value in enumerate(prepared["values"], 1):
            anchor.stable_inputs(prepared)
            previous, record = stage(problem, prepared, previous, index, value, output, argv, np)
            stages.append(record)
        require(problem.case == target, "terminal III live case differs")
        terminal = output / "terminal"
        (terminal / "solver").mkdir(parents=True)
        write_json(
            terminal / "run-manifest.json",
            manifest(args.case, target, prepared, {"kind": "stationary_III_terminal_remesh"}, argv),
        )
        runner.append_progress(
            terminal, {"event": "prepared", "case_id": target["case_id"]}, first=True
        )

        def observe(phase):
            anchor.stable_inputs(prepared)
            if phase == "after-remesh":
                remesh.normalize_pressure_dummies(problem, model, terminal, np)
            sampled = remesh.sample(problem, model, target, terminal, phase, np)
            if phase != "after-polish":
                sampled["retained_state"] = remesh.retain_portable_sample(
                    problem, model, terminal, phase
                )
            return sampled

        terminal_result = remesh.remesh_and_polish(problem, model, target, terminal, observe)
        require(polish_bounds(terminal_result["polish"]), "III terminal polish bound differs")
        checkpoint = model._checkpoint_stationary_state(problem, target, output, 16, True)
        final = runner.validate_restart(
            output / checkpoint["resume_packet"],
            target,
            sha(args.case),
            component=prepared["component"],
            environment=prepared["environment"],
            azimuthal_stability=True,
        )
        pins = diagnostic.actual_pins(problem)
        require(
            pins == terminal_result["after_polish"]["actual_pins"],
            "III target checkpoint changed pins",
        )
        anchor.stable_inputs(prepared)
    cold_output = output / "target-cold"
    cold_output.mkdir()
    with model.build_problem(target, initial_bond=target["physics"]["bond"]) as fresh:
        fresh._stationary_iii_receipts_directory = cold_output / "loads"
        anchor.configure(fresh, cold_output, prepared)
        loaded = restore_checked(
            fresh,
            target,
            final,
            pins,
            final["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
        )
        sampled = remesh.sample(fresh, model, target, cold_output, "cold-load", np)
        cold_checks = remesh.canonical_checks(target, sampled["metrics"])
        loaded["canonical_checks"] = cold_checks
        write_json(output / "cold-load.json", loaded)
    checks = remesh.canonical_checks(target, terminal_result["after_polish"]["metrics"])
    require(
        all(checks.values()) and all(cold_checks.values()),
        "III original target warm/cold gates failed",
    )
    result = {
        "schema": SCHEMA,
        "stages": stages,
        "terminal": terminal_result,
        "checkpoint": checkpoint,
        "ca_in": TARGET_CA,
        "canonical_checks": checks,
        "cold_load_checks": cold_checks,
        "cold_load_verified": True,
        "completion_reached": True,
        "numerical_acceptance": True,
        "anchor_accepted": True,
        "parent_acceptance_changed": False,
        "parents_unchanged": True,
        "scope": SCOPE,
        **NO_CLAIMS,
    }
    anchor.stable_inputs(prepared)
    write_json(output / "result.json", result)
    write_json(
        output / "summary.json",
        {
            "status": "completed",
            "completion_reached": True,
            "scientific_status": "checks_passed",
            "checks": checks,
            "cold_load_checks": cold_checks,
            "cold_load_verified": True,
            **terminal_result["after_polish"]["metrics"],
            "checkpoint": checkpoint["resume_packet"],
            "ca_in": TARGET_CA,
            "anchor_accepted": True,
            "parent_acceptance_changed": False,
            **NO_CLAIMS,
        },
    )
    return result


def inventory(output):
    files = {}
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), "completion custody contains a symlink")
        if (
            path.is_file()
            and "solver" not in path.relative_to(output).parts
            and path.name not in {"progress.jsonl", "completion.json"}
        ):
            files[str(path.relative_to(output))] = sha(path)
    return files


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "ca-ladder", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="metadata only; no native import or output"
    )
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    output = runner.validate_output_path(args.out, ROOT)
    require(
        not output.is_relative_to(Path(prepared["restart"]["parent_manifest"]["path"]).parent),
        "output overlaps qualified III parent",
    )
    if args.check:
        print(
            json.dumps(
                {"verified": True, "native_imported": False, "contract": prepared["contract"]},
                allow_nan=False,
            )
        )
        return 0
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(
        output / "run-manifest.json",
        manifest(args.case, prepared["target"], prepared, {"kind": KIND}, arguments),
    )
    runner.append_progress(
        output, {"event": "prepared", "case_id": prepared["target"]["case_id"]}, first=True
    )
    try:
        write_json(
            output / "preparation.json",
            {
                "contract": prepared["contract"],
                "inputs": prepared["records"],
                "implementation": prepared["implementation"],
                "numerical_acceptance": False,
            },
        )
        execute(args, prepared, output, arguments)
        anchor.stable_inputs(prepared)
        write_json(
            output / "completion.json",
            {
                "schema": SCHEMA,
                "status": "complete",
                "numerical_acceptance": True,
                "anchor_accepted": True,
                "parent_acceptance_changed": False,
                "scope": SCOPE,
                "files": inventory(output),
            },
        )
        runner.append_progress(
            output, {"event": "final", "status": "complete", "numerical_acceptance": True}
        )
        return 0
    except (Exception, KeyboardInterrupt, SystemExit) as error:  # noqa: BLE001 - preserve terminal failures
        write_json(
            output / "failure.json",
            {
                "schema": SCHEMA,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "numerical_acceptance": False,
                "anchor_accepted": False,
                "parent_acceptance_changed": False,
                **NO_CLAIMS,
            },
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


def admit_terminal(runtime, parent, restart, case, summary, cold, files, completion):
    """Replay exactly this terminal producer's original qualified-III admission."""
    require(not (runtime / "failure.json").exists(), "III terminal has a failure receipt")
    require(
        set(completion)
        == {
            "schema",
            "status",
            "numerical_acceptance",
            "anchor_accepted",
            "parent_acceptance_changed",
            "scope",
            "files",
        }
        and completion["schema"] == SCHEMA
        and completion["scope"] == SCOPE
        and completion["numerical_acceptance"] is True
        and completion["anchor_accepted"] is True
        and completion["parent_acceptance_changed"] is False
        and inventory(runtime) == files,
        "III terminal completion type or inventory differs",
    )
    require(
        Path(parent["runner"]["path"]).name == Path(ENTRYPOINT).name
        and parent["anchor_stage"] == {"kind": KIND}
        and parent["qualification_schema"] == SCHEMA,
        "exact III terminal producer required",
    )
    seed = parent["parent_restart"]
    admitted = preflight(
        SimpleNamespace(
            source=ROOT,
            case=Path(parent["case_path"]),
            manifest=Path(seed["parent_manifest"]["path"]),
            packet=Path(seed["packet"]["path"]),
            state=Path(seed["state"]["path"]),
            ca_ladder=Path(parent["case_transition_contract"]["ca_ladder"]["path"]),
        )
    )
    historical = {
        **admitted["restart"],
        "compatibility": {
            **admitted["restart"]["compatibility"],
            "current_commit": parent["component_commit"],
        },
    }
    preparation = runner._read_json(runtime / "preparation.json")
    require(
        historical == seed
        and admitted["target"] == case
        and parent["case_transition_contract"] == preparation["contract"] == admitted["contract"]
        and preparation["inputs"] == admitted["records"]
        and preparation["implementation"] == parent["implementation"]
        and preparation["numerical_acceptance"] is False
        and set(parent["implementation"]) == set(admitted["implementation"])
        and all(
            item["sha256"] == parent["implementation"][name]["sha256"]
            for name, item in admitted["implementation"].items()
        ),
        "III immutable parent admission or helper contract differs",
    )
    result = runner._read_json(runtime / "result.json")
    require(
        set(result)
        == {
            "schema",
            "stages",
            "terminal",
            "checkpoint",
            "ca_in",
            "canonical_checks",
            "cold_load_checks",
            "cold_load_verified",
            "completion_reached",
            "numerical_acceptance",
            "anchor_accepted",
            "parent_acceptance_changed",
            "parents_unchanged",
            "scope",
            *NO_CLAIMS,
        }
        and result["schema"] == SCHEMA
        and result["scope"] == SCOPE
        and result["ca_in"] == case["physics"]["ca_in"] == TARGET_CA
        and result["completion_reached"] is True
        and result["numerical_acceptance"] is True
        and result["anchor_accepted"] is True
        and result["parents_unchanged"] is True
        and result["parent_acceptance_changed"] is False
        and result["cold_load_verified"] is True,
        "III terminal result type or acceptance differs",
    )
    for value in (parent, result, summary):
        require(
            all(value.get(name, False) is False for name in NO_CLAIMS),
            "unestablished III branch, dynamics or spectrum claim",
        )
    records = list(admitted["records"])
    records.extend(
        bound(item, runtime, files)
        for item in (restart["packet"], restart["state"], *restart["meshes"])
    )
    require(
        restart["parent_manifest"] == regular_record(runtime / "run-manifest.json")
        and runtime / relative_path(result["checkpoint"]["resume_packet"])
        == Path(restart["packet"]["path"]),
        "III terminal checkpoint lineage differs",
    )
    stages = result["stages"]
    require(len(stages) == len(LADDER), "exact three III stages required")
    for index, (value, row) in enumerate(zip(LADDER, stages, strict=True), 1):
        directory = runtime / "stages" / f"ca_{index:03d}"
        case_path = directory / "case.json"
        current = anchor.stage_case(case, value, index, index == len(LADDER))
        stage_manifest = runner._read_json(directory / "run-manifest.json")
        require(
            row["index"] == index
            and row["ca_in"] == value
            and row["numerical_acceptance"] is False
            and row == runner._read_json(directory / "stage.json")
            and runner.read_case(case_path) == current
            and row["case"] == regular_record(case_path)
            and stage_manifest["anchor_stage"]
            == {"kind": "stationary_III_monotone_step", "index": index, "ca_in": value}
            and all(
                stage_manifest[name] == parent[name]
                for name in (
                    "component",
                    "component_commit",
                    "implementation",
                    "parent_restart",
                    "case_transition_contract",
                    "qualification_schema",
                    "runner",
                )
            )
            and polish_bounds(row["polish"])
            and row["polish"] == runner._read_json(directory / "polish/ca_step/polish.json")
            and row["transition"] == runner._read_json(directory / "transition.json")
            and row["transition"]["verified"] is True,
            "III provisional stage proof differs",
        )
        packet = runner.validate_restart(
            directory / relative_path(row["checkpoint"]["resume_packet"]),
            current,
            sha(case_path),
            component=admitted["component"],
            environment=admitted["environment"],
            azimuthal_stability=True,
        )
        sample = runner._read_json(directory / "samples/after-polish.json")
        require(
            packet["parent_manifest"] == regular_record(directory / "run-manifest.json")
            and packet["packet"]["sha256"] == row["resume_packet_sha256"]
            and packet["stage"]["kind"] == "completed_polish"
            and packet["state"]["format"] == "portable_file"
            and sample["sampling_preserved_state"] is True
            and physical_signature(sample["signature"])
            == physical_signature(packet["roundtrip"]["after"])
            and row["canonical_checks_observed"]
            == remesh.canonical_checks(current, sample["metrics"]),
            "III stage packet or sampled history differs",
        )
        records.extend(
            bound(item, runtime, files)
            for item in (packet["packet"], packet["state"], *packet["meshes"])
        )
    terminal = result["terminal"]
    require(
        type(terminal["explicit_force_remesh_calls"]) is int
        and terminal["explicit_force_remesh_calls"] == 1
        and polish_bounds(terminal["polish"])
        and terminal["polish"]
        == runner._read_json(runtime / "terminal/polish/terminal_remesh/polish.json"),
        "III terminal remesh/polish bound differs",
    )
    warm = terminal["after_polish"]
    require(
        warm == runner._read_json(runtime / "terminal/samples/after-polish.json")
        and warm["sampling_preserved_state"] is True
        and all(result["canonical_checks"].values())
        and all(result["cold_load_checks"].values())
        and result["canonical_checks"]
        == summary["checks"]
        == remesh.canonical_checks(case, warm["metrics"])
        and result["cold_load_checks"] == summary["cold_load_checks"] == cold["canonical_checks"]
        and cold["actual_pins"] == warm["actual_pins"]
        and physical_signature(warm["signature"])
        == physical_signature(restart["roundtrip"]["after"]),
        "III target warm/cold gates or checkpoint state differ",
    )
    expected = restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
    records.extend(
        validate_load(cold, restart, case, warm["actual_pins"], expected, runtime, files)
    )
    parent_load = runner._read_json(runtime / "parent-load.json")
    records.extend(
        validate_load(
            parent_load,
            admitted["restart"],
            admitted["parent"],
            admitted["parent_cold"]["actual_pins"],
            admitted["restart"]["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
            runtime,
            files,
        )
    )
    initial_sample = runner._read_json(runtime / "parent-observation/samples/cold-load.json")
    final_sample = runner._read_json(runtime / "target-cold/samples/cold-load.json")
    require(
        initial_sample["sampling_preserved_state"] is True
        and initial_sample["actual_pins"] == parent_load["actual_pins"]
        and physical_signature(initial_sample["signature"])
        == physical_signature(parent_load["loaded"])
        and all(remesh.canonical_checks(admitted["parent"], initial_sample["metrics"]).values())
        and final_sample["sampling_preserved_state"] is True
        and remesh.canonical_checks(case, final_sample["metrics"]) == result["cold_load_checks"]
        and final_sample["actual_pins"] == cold["actual_pins"]
        and physical_signature(final_sample["signature"]) == physical_signature(cold["loaded"]),
        "III fresh sampled load proof differs",
    )
    return records


if __name__ == "__main__":
    raise SystemExit(main())
