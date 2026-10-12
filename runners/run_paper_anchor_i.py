"""Continue a qualified fresh negative-flow seed to one stationary I candidate.

Only case identity and imposed Ca change along an explicit ascending ladder.
Intermediate observations remain provisional. Qualification requires the
unchanged terminal remesh, pressure-history audit and fresh complete-state load.
This driver makes no branch-identity, spectrum or full-comparison claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_anchor as anchor  # noqa: E402
from runners import run_paper_seed_i as seed  # noqa: E402
from runners import run_paper_seed_recovery as recovery  # noqa: E402
from runners import run_paper_seed_requalify as requalification  # noqa: E402

model, runner, diagnostic, remesh = anchor.model, anchor.runner, anchor.diagnostic, anchor.remesh
require, sha, write_json = diagnostic.require, diagnostic.sha, diagnostic.write_json
PARENT_SOURCE = "0eed7987acf76febe65f044250fe61217e688ae9"
PARENT_CASE_SHA = "f9f0e44ff573c1c9cb37ad22d85b2ea3b5625e45dd0c64ddf854749708a9ae1d"
PARENT_CA, TARGET_CA, MAX_CA_STEPS = -0.02, -0.0016, 8
GATES = anchor.GATES
ENTRYPOINT = "runners/run_paper_anchor_i.py"
PRODUCER = "runners/run_paper_seed_requalify.py"


def transition_contract(parent, target):
    contract = anchor.transition_contract(parent, target)
    require(
        parent["physics"]["ca_in"] == PARENT_CA
        and target["physics"]["ca_in"] == TARGET_CA
        and parent["physics"]["bond"] == 1.56
        and parent["physics"]["problem"] == "finite_gas_bubble",
        "ascending I continuation requires the declared negative-flow seed and target",
    )
    return contract


def read_ladder(path, parent, target):
    transition_contract(parent, target)
    ladder = json.loads(Path(path).read_text())
    require(
        isinstance(ladder, dict)
        and set(ladder) == {"schema", "values"}
        and ladder["schema"] == "herrada-paper-ca-ladder-v1",
        "explicit Ca ladder schema required",
    )
    values = ladder["values"]
    require(
        isinstance(values, list) and 1 <= len(values) <= MAX_CA_STEPS,
        "Ca ladder exceeds its step bound",
    )
    require(
        all(
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            for value in values
        ),
        "Ca ladder values must be finite numbers",
    )
    require(values[-1] == TARGET_CA, "Ca ladder does not end at the declared target")
    require(
        all(left < right for left, right in pairwise([PARENT_CA, *values])),
        "Ca ladder must increase strictly from its parent",
    )
    return values


def check_parent_producer(parent_manifest):
    require(
        parent_manifest["component_commit"]
        == parent_manifest["component"]["commit"]
        == PARENT_SOURCE,
        "unexpected qualified seed source",
    )
    expected = hashlib.sha256(
        diagnostic.git(ROOT, "show", f"{PARENT_SOURCE}:{PRODUCER}")
    ).hexdigest()
    record = parent_manifest.get("runner", {})
    implementation = parent_manifest.get("implementation", {}).get(PRODUCER, {})
    require(
        Path(record.get("path", "")).name == "run_paper_seed_requalify.py"
        and record.get("path") == parent_manifest.get("runner_path") == implementation.get("path")
        and expected
        == record.get("sha256")
        == parent_manifest.get("runner_sha256")
        == implementation.get("sha256")
        and parent_manifest.get("intervention")
        == {
            "kind": "stationary-seed-requalification",
            "solve_calls": 0,
            "remesh_calls": 0,
            "time_evolution": False,
            "api": requalification.cold.API,
            "parent_acceptance_changed": False,
        }
        and parent_manifest.get("lineage", {}).get("schema") == requalification.SCHEMA
        and parent_manifest["lineage"].get("parent_status") == "failed"
        and parent_manifest["lineage"].get("parent_acceptance_changed") is False
        and parent_manifest["lineage"].get("consumer")
        == {
            "module": PRODUCER,
            "callable": "restore_checked",
            "sha256": expected,
            "saved_endtime": 8.0,
            "api": requalification.cold.API,
        }
        and parent_manifest.get("azimuthal_stability") is True,
        "parent producer, hash or typed requalification differs from the reviewed descendant",
    )


def validate_parent_lineage(parent_manifest):
    """Recheck the immutable failed producer and completed diagnostic without native imports."""
    lineage = parent_manifest["lineage"]
    prepared = requalification.preflight(
        SimpleNamespace(
            source=ROOT,
            case=Path(parent_manifest["case_path"]),
            manifest=Path(lineage["original_failed_producer"]["path"]),
            packet=Path(lineage["original_packet"]["path"]),
            state=Path(lineage["original_state"]["path"]),
            diagnostic_runtime=Path(lineage["completed_diagnostic"]["manifest"]["path"]).parent,
        )
    )
    require(
        prepared["lineage"] == lineage, "requalified parent lineage differs from direct evidence"
    )
    return prepared


def qualified_parent(parent_path, parent, restart):
    """Require terminal qualification, with all direct artifacts hash-bound."""
    runtime = parent_path.parent
    require(not (runtime / "failure.json").exists(), "parent has a failure receipt")
    completion_path = runtime / "completion.json"
    require(
        completion_path.is_file() and not completion_path.is_symlink(),
        "qualified terminal parent completion required",
    )
    completion = json.loads(completion_path.read_text())
    require(
        completion.get("status") == "complete"
        and completion.get("schema") == requalification.SCHEMA
        and completion.get("numerical_acceptance") is True,
        "parent lacks numerical acceptance",
    )
    files = completion.get("files", {})
    require(
        isinstance(files, dict)
        and {
            "run-manifest.json",
            "summary.json",
            "cold-load.json",
            "result.json",
            "parent-cold-load.json",
            "schema.json",
        }
        <= set(files),
        "parent qualification files unbound",
    )
    records = [{"path": str(completion_path), "sha256": sha(completion_path)}]
    for relative, digest in files.items():
        path = runtime / relative
        require(
            not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(runtime)
            and sha(path) == digest,
            "parent completion artifact differs",
        )
        records.append({"path": str(path.resolve()), "sha256": digest})
    summary = json.loads((runtime / "summary.json").read_text())
    cold = json.loads((runtime / "cold-load.json").read_text())
    result = json.loads((runtime / "result.json").read_text())
    parent_cold = json.loads((runtime / "parent-cold-load.json").read_text())
    parent_manifest = json.loads(parent_path.read_text())
    schema = json.loads((runtime / "schema.json").read_text())
    require(
        summary.get("completion_reached") is True
        and summary.get("cold_load_verified") is True
        and set(summary.get("checks", {})) == set(summary.get("cold_load_checks", {})) == GATES
        and all(value is True for value in summary["checks"].values())
        and all(value is True for value in summary["cold_load_checks"].values())
        and cold.get("verified") is True
        and cold.get("checks") == summary["cold_load_checks"]
        and result.get("schema") == requalification.SCHEMA
        and result.get("numerical_acceptance") is True
        and result.get("parent_status") == "failed"
        and result.get("parent_acceptance_changed") is False
        and all(
            result.get(key) is False
            for key in (
                "branch_identity_verified",
                "eigenanalysis_performed",
                "jacobian_certificate",
                "dynamic_restart_certificate",
                "full_comparison_verified",
            )
        )
        and result.get("lineage") == summary.get("lineage") == parent_manifest.get("lineage")
        and parent_cold.get("verified") is True
        and all(
            receipt.get("bc_preparation", {}).get("verified") is True
            and receipt.get("intervention", {}).get("verified") is True
            and receipt["intervention"].get("kind") == "typed-native-endtime-restoration"
            and receipt["intervention"].get("saved_endtime") == parent["solve"]["relax_time"]
            and receipt["intervention"].get("api") == requalification.cold.API
            for receipt in (parent_cold, cold)
        ),
        "parent scientific, typed restoration or BC-preparation gates failed",
    )
    require(
        schema.get("schema") == summary.get("schema") == requalification.SCHEMA
        and schema.get("consumer") == parent_manifest["lineage"]["consumer"]
        and schema.get("scope") == parent_manifest["qualification_scope"]
        and summary.get("status") == "completed"
        and all(summary.get(key) == value for key, value in result.items())
        and result.get("cold_load_verified") is True
        and result.get("checks") == summary["checks"]
        and result.get("cold_load_checks") == summary["cold_load_checks"]
        and cold.get("packet") == restart["packet"]
        and cold.get("parent_manifest") == restart["parent_manifest"]
        and restart["parent_manifest"] == {"path": str(parent_path), "sha256": sha(parent_path)}
        and (runtime / result["checkpoint"]["resume_packet"]).resolve()
        == Path(restart["packet"]["path"])
        and files.get(str(Path(restart["packet"]["path"]).relative_to(runtime)))
        == restart["packet"]["sha256"]
        and all(
            Path(record["path"]).is_relative_to(runtime)
            and files.get(str(Path(record["path"]).relative_to(runtime))) == record["sha256"]
            for record in (restart["state"], *restart["meshes"])
        ),
        "parent schema, checkpoint or manifest lineage differs",
    )
    original = validate_parent_lineage(parent_manifest)
    expected_pins = original["sample"]["actual_pins"]
    expected_history = original["restart"]["roundtrip"]["after"]["serialized_state"][
        "payload_sha256"
    ]
    for receipt, saved in ((parent_cold, original["restart"]), (cold, restart)):
        remesh.check_loaded_state(
            receipt["loaded"], saved, parent, receipt["actual_pins"], expected_pins
        )
        require(
            receipt["loaded"]["serialized_state"]["payload_sha256"] == expected_history
            and receipt.get("time_history")
            == {
                key: original["metadata"][key]
                for key in ("time", "dts", "unsteady_steps", "suggested_next_dt")
            },
            "parent cold-load pins or complete time history differs",
        )
    require(
        cold["loaded"]["serialized_state"]["payload_sha256"]
        == restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
        and requalification.cold.normalized_bytes(
            Path(restart["state"]["path"]).read_bytes(), Path(restart["mesh_reference_base"])
        )
        == requalification.cold.normalized_bytes(
            Path(original["restart"]["state"]["path"]).read_bytes(),
            Path(original["restart"]["mesh_reference_base"]),
        ),
        "parent cold-load complete history differs",
    )
    records.extend(original["records"])
    return cold, records


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "runner must use its own isolated component")
    modules = (
        (model, "problems/paper_bubble.py"),
        (runner, "runners/run_paper_case.py"),
        (diagnostic, "runners/run_pressure_diagnostic.py"),
        (remesh, "runners/run_pressure_remesh.py"),
        (anchor, "runners/run_paper_anchor.py"),
        (seed, "runners/run_paper_seed_i.py"),
        (recovery, "runners/run_paper_seed_recovery.py"),
        (requalification.cold, "runners/run_paper_cold_history.py"),
        (requalification, PRODUCER),
    )
    for module, relative in modules:
        require(
            Path(module.__file__).resolve() == ROOT / relative,
            "helper imported from another component",
        )
    parent_path = args.manifest.resolve(strict=True)
    parent_manifest = json.loads(parent_path.read_text())
    check_parent_producer(parent_manifest)
    component = runner.component_revision(ROOT)
    runner.require_clean_source(component)
    runner.require_clean_source(parent_manifest["component"])
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", PARENT_SOURCE, component["commit"])
    old_sources = parent_manifest["component"]["sources"]
    require(set(old_sources) == set(component["sources"]), "parent method inventory differs")
    implementation = {}
    paths = set(component["sources"]) | {relative for _, relative in modules} | {ENTRYPOINT}
    for relative in sorted(paths):
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "unsafe implementation path",
        )
        path = ROOT / relative
        committed = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{component['commit']}:{relative}")
        ).hexdigest()
        require(
            path.is_file() and not path.is_symlink() and sha(path) == committed,
            "implementation differs from commit",
        )
        if relative != ENTRYPOINT:
            frozen = hashlib.sha256(
                diagnostic.git(ROOT, "show", f"{PARENT_SOURCE}:{relative}")
            ).hexdigest()
            require(committed == frozen, f"frozen method changed: {relative}")
        if relative in old_sources:
            old = old_sources[relative]
            require(
                old["committed_matches"] is True
                and old["sha256"] == old["committed_sha256"] == committed,
                f"parent method changed: {relative}",
            )
        implementation[relative] = {"path": str(path), "sha256": committed}
    environment = runner.pyoomph_revision()
    seed.validate_environment(environment)
    runner.require_pinned_environment(parent_manifest["pyoomph"])
    require(
        parent_manifest["pyoomph"]["commit"] == diagnostic.PYOOMPH
        and parent_manifest["pyoomph"]["native_core"]["sha256"] == diagnostic.CORE
        and environment["interpreter"]["resolved_executable"]
        == parent_manifest["pyoomph"]["interpreter"]["resolved_executable"],
        "parent source/core or interpreter differs",
    )
    parent_case_path = Path(parent_manifest["case_path"]).resolve(strict=True)
    require(
        sha(parent_case_path) == parent_manifest["case_sha256"] == PARENT_CASE_SHA,
        "parent construction differs from the declared seed case",
    )
    parent, target = runner.read_case(parent_case_path), runner.read_case(args.case)
    contract = transition_contract(parent, target)
    values = read_ladder(args.ca_ladder, parent, target)
    restart = runner.validate_restart(
        args.packet,
        parent,
        sha(parent_case_path),
        component=component,
        environment=environment,
        azimuthal_stability=True,
    )
    require(
        Path(restart["parent_manifest"]["path"]) == parent_path
        and Path(restart["state"]["path"]) == args.state.resolve(strict=True)
        and restart["stage"]["kind"] == "completed_polish"
        and restart["stage"]["bond"] == parent["physics"]["bond"],
        "qualified terminal parent packet/state required",
    )
    cold, qualification_records = qualified_parent(parent_path, parent, restart)
    records = [
        restart["packet"],
        restart["parent_manifest"],
        restart["state"],
        *restart["meshes"],
        *qualification_records,
        {"path": str(parent_case_path), "sha256": sha(parent_case_path)},
        {"path": str(args.case.resolve()), "sha256": sha(args.case)},
        {"path": str(args.ca_ladder.resolve()), "sha256": sha(args.ca_ladder)},
    ]
    contract.update(
        parent_case={"path": str(parent_case_path), "sha256": sha(parent_case_path)},
        target_case={"path": str(args.case.resolve()), "sha256": sha(args.case)},
        ca_ladder={
            "path": str(args.ca_ladder.resolve()),
            "sha256": sha(args.ca_ladder),
            "schema": "herrada-paper-ca-ladder-v1",
            "values": values,
            "direction": "ascending",
        },
        maximum_ca_steps=MAX_CA_STEPS,
        per_mesh_stage_newton_cap=min(40, target["solve"]["max_newton_iterations"]),
        per_polish_total_newton_cap=160,
        total_newton_update_bound=160 * (len(values) + 1),
        per_polish_remesh_bound=3,
        explicit_terminal_remesh_calls=1,
    )
    return {
        "parent": parent,
        "target": target,
        "restart": restart,
        "parent_cold": cold,
        "environment": environment,
        "component": component,
        "implementation": implementation,
        "records": records,
        "contract": contract,
        "values": values,
    }


def manifest(case_path, case, prepared, stage):
    script = Path(__file__).resolve()
    record = {"path": str(script), "sha256": sha(script)}
    return {
        "schema": "pyoomph-runner-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "case_path": str(case_path.resolve()),
        "case_file": str(case_path.resolve()),
        "case_sha256": sha(case_path),
        "case_id": case["case_id"],
        "case_fields_used": list(runner.CASE_FIELDS_USED),
        "case_fields_scope": "all fields validated/bound; qualified seed restore and explicit ascending Ca stages",
        "units": "nondimensional (sigma=mu_liquid=R0=1)",
        "nondimensionalization": case["nondimensionalization"],
        "argv": [sys.executable, str(script), *sys.argv[1:]],
        "runner": record,
        "runner_path": record["path"],
        "runner_sha256": record["sha256"],
        "component_commit": prepared["component"]["commit"],
        "component": prepared["component"],
        "pyoomph": prepared["environment"],
        "azimuthal_stability": True,
        "implementation": prepared["implementation"],
        "anchor_stage": stage,
        "case_transition_contract": prepared["contract"],
        "construction_case": prepared["contract"]["parent_case"],
        "parent_restart": prepared["restart"],
        "parent_acceptance_changed": False,
        "qualification_scope": "stationary Ca=-0.0016 candidate for Branch I; no branch identity, spectrum or full-comparison claim",
    }


def execute(args, prepared, output):
    import numpy as np

    parent, target, restart = prepared["parent"], prepared["target"], prepared["restart"]
    cold_parent, stages = prepared["parent_cold"], []

    def emit(event, **payload):
        runner.append_progress(output, {"event": event, "case_id": target["case_id"], **payload})

    with model.build_problem(parent, initial_bond=parent["physics"]["bond"]) as problem:
        anchor.configure(problem, output, prepared)
        restored = requalification.restore_checked(
            problem,
            parent,
            restart,
            cold_parent["actual_pins"],
            cold_parent["loaded"]["serialized_state"]["payload_sha256"],
        )
        parent_output = output / "parent-observation"
        parent_output.mkdir()
        observed = remesh.sample(problem, model, parent, parent_output, "cold-load", np)
        require(
            all(remesh.canonical_checks(parent, observed["metrics"]).values()),
            "parent cold-load gates failed",
        )
        write_json(output / "parent-load.json", restored)
        emit("parent-restored", complete_fields_pins_history_verified=True)
        previous = parent
        for index, value in enumerate(prepared["values"], 1):
            anchor.stable_inputs(prepared)
            current = anchor.stage_case(target, value, index, index == len(prepared["values"]))
            stage_output = output / "stages" / f"ca_{index:03d}"
            (stage_output / "solver").mkdir(parents=True)
            case_path = stage_output / "case.json"
            write_json(case_path, current)
            write_json(
                stage_output / "run-manifest.json",
                manifest(
                    case_path,
                    current,
                    prepared,
                    {"kind": "ascending_ca_continuation", "index": index, "ca_in": value},
                ),
            )
            runner.append_progress(
                stage_output, {"event": "prepared", "case_id": current["case_id"]}, first=True
            )
            transition = anchor.adopt_case(problem, previous, current, stage_output)
            emit("ca-transition", index=index, ca_in=value)
            polish = model.polish_confined_state(
                problem,
                current,
                stage_output,
                "ca_step",
                initial_relaxation=1.0,
            )
            require(polish["status"] == "converged", "Ca stage polish failed")
            observed = remesh.sample(problem, model, current, stage_output, "after-polish", np)
            checkpoint = model._checkpoint_stationary_state(
                problem,
                current,
                stage_output,
                len(model.stationary_bond_schedule(current)) - 1,
                True,
            )
            stage_restart = runner.validate_restart(
                stage_output / checkpoint["resume_packet"],
                current,
                sha(case_path),
                component=prepared["component"],
                environment=prepared["environment"],
                azimuthal_stability=True,
            )
            require(
                diagnostic.actual_pins(problem) == observed["actual_pins"],
                "continuation checkpoint changed actual pins",
            )
            record = {
                "index": index,
                "ca_in": value,
                "case": {"path": str(case_path), "sha256": sha(case_path)},
                "transition": transition,
                "polish": polish,
                "checkpoint": checkpoint,
                "resume_packet_sha256": stage_restart["packet"]["sha256"],
                "canonical_checks_observed": remesh.canonical_checks(current, observed["metrics"]),
                "numerical_acceptance": False,
                "qualification_scope": "provisional stationary continuation stage; terminal qualification pending",
            }
            write_json(stage_output / "stage.json", record)
            stages.append(record)
            previous = current
            anchor.stable_inputs(prepared)
        require(problem.case == target, "terminal live case differs from target")
        terminal = output / "terminal"
        (terminal / "solver").mkdir(parents=True)
        write_json(
            terminal / "run-manifest.json",
            manifest(args.case, target, prepared, {"kind": "terminal_remesh"}),
        )
        runner.append_progress(
            terminal, {"event": "prepared", "case_id": target["case_id"]}, first=True
        )

        def observe(phase):
            anchor.stable_inputs(prepared)
            if phase == "after-remesh":
                remesh.normalize_pressure_dummies(problem, model, terminal, np)
            observed = remesh.sample(problem, model, target, terminal, phase, np)
            if phase != "after-polish":
                observed["retained_state"] = remesh.retain_portable_sample(
                    problem,
                    model,
                    terminal,
                    phase,
                )
            return observed

        terminal_result = remesh.remesh_and_polish(problem, model, target, terminal, observe)
        checkpoint = model._checkpoint_stationary_state(
            problem,
            target,
            output,
            len(model.stationary_bond_schedule(target)) - 1,
            True,
        )
        final_restart = runner.validate_restart(
            output / checkpoint["resume_packet"],
            target,
            sha(args.case),
            component=prepared["component"],
            environment=prepared["environment"],
            azimuthal_stability=True,
        )
        final_pins = diagnostic.actual_pins(problem)
        require(
            final_pins == terminal_result["after_polish"]["actual_pins"],
            "target checkpoint changed actual pins",
        )
        anchor.stable_inputs(prepared)
    emit("target-cold-load", status="starting")
    cold_output = output / "target-cold"
    cold_output.mkdir()
    with model.build_problem(target, initial_bond=target["physics"]["bond"]) as cold:
        anchor.configure(cold, cold_output, prepared)
        cold_receipt = requalification.restore_checked(
            cold,
            target,
            final_restart,
            final_pins,
            final_restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
        )
        observed = remesh.sample(cold, model, target, cold_output, "cold-load", np)
        cold_checks = remesh.canonical_checks(target, observed["metrics"])
        cold_receipt["canonical_checks"] = cold_checks
        write_json(output / "cold-load.json", cold_receipt)
    checks = remesh.canonical_checks(target, terminal_result["after_polish"]["metrics"])
    accepted = all(checks.values()) and all(cold_checks.values())
    result = {
        "schema": "private-paper-stationary-anchor-v1",
        "stages": stages,
        "terminal": terminal_result,
        "ca_in": TARGET_CA,
        "canonical_checks": checks,
        "cold_load_checks": cold_checks,
        "cold_load_verified": True,
        "completion_reached": accepted,
        "numerical_acceptance": accepted,
        "causal_predictions": remesh.causal_predictions(terminal_result),
        "checkpoint": checkpoint,
        "parent_acceptance_changed": False,
        "parents_unchanged": True,
        "branch_identity_verified": False,
        "eigenanalysis_performed": False,
    }
    anchor.stable_inputs(prepared)
    write_json(output / "result.json", result)
    write_json(
        output / "summary.json",
        {
            "status": "completed" if accepted else "failed",
            "completion_reached": accepted,
            "scientific_status": "checks_passed" if accepted else "checks_failed",
            "checks": checks,
            "cold_load_checks": cold_checks,
            "cold_load_verified": True,
            **terminal_result["after_polish"]["metrics"],
            "checkpoint": checkpoint["resume_packet"],
            "ca_in": TARGET_CA,
            "branch_identity_verified": False,
            "eigenanalysis_performed": False,
        },
    )
    require(accepted, "target original scientific or cold-load gates failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "ca-ladder", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="metadata validation only; no native import/output"
    )
    args = parser.parse_args(argv)
    os.umask(0o077)
    prepared = preflight(args)
    require(not args.out.is_symlink(), "output must not be a symlink")
    output = runner.validate_output_path(args.out, ROOT)
    require(
        not output.is_relative_to(Path(prepared["restart"]["parent_manifest"]["path"]).parent),
        "output overlaps parent runtime",
    )
    if args.check:
        print(
            json.dumps(
                {"verified": True, "native_imported": False, "contract": prepared["contract"]},
                allow_nan=False,
            )
        )
        return 0
    actual = manifest(args.case, prepared["target"], prepared, {"kind": "stationary_I_candidate"})
    json.dumps(actual, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
    write_json(output / "run-manifest.json", actual)
    runner.append_progress(
        output,
        {
            "event": "prepared",
            "case_id": prepared["target"]["case_id"],
            "runner": actual["runner"],
            "component_commit": prepared["component"]["commit"],
        },
        first=True,
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
        result = execute(args, prepared, output)
        anchor.stable_inputs(prepared)
        files = {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and "solver" not in path.relative_to(output).parts
            and path.name != "progress.jsonl"
        }
        write_json(
            output / "completion.json",
            {
                "status": "complete",
                "files": files,
                "numerical_acceptance": result["numerical_acceptance"],
                "parent_acceptance_changed": False,
                "scope": actual["qualification_scope"],
            },
        )
        runner.append_progress(
            output, {"event": "final", "status": "complete", "numerical_acceptance": True}
        )
        return 0
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "parents_unchanged": diagnostic.parents_unchanged(prepared["records"]),
            "numerical_acceptance": False,
            "parent_acceptance_changed": False,
        }
        write_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", **failure})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
