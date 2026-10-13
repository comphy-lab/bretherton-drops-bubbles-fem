"""Qualify one saved finite-gas fold candidate without another equilibrium solve.

The failed producer and cumulative work remain immutable inputs. This is a new
qualification phase with a fresh Problem, two bounded m=0 searches, exact
same-instance restoration and a fresh cold check. Metadata checking imports no
pyoomph. A clean source, external seal and active admission are required
before native work; this driver cannot issue that admission.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_finite_gas_fold as fold

model, runner, diagnostic, remesh, history, modes, approach = (
    fold.model, fold.runner, fold.diagnostic, fold.remesh, fold.history, fold.modes, fold.approach
)
gate, regular, sha, write_json = fold.gate, fold.regular, fold.sha, fold.write_json
BASE = "905c56d802ae85a3c10143f2f9f13bbc07f29ea1"
ENTRYPOINT = "runners/run_paper_finite_gas_fold_qualify.py"
TEST = "tests/test_paper_finite_gas_fold_qualify.py"
SCHEMA = "saved-finite-gas-fold-qualification-only-v1"
RECOVERY_KIND = "saved-finite-gas-fold-qualification-recovery-v1"
PRELAUNCH_COMMIT = "9cfe77adaf0500fb48f98b2bfb9c2ec27bb8ad77"
PRELAUNCH_TERMINAL_SHA = "0cc9c66c8462607e601dea70470ac56c66e9b04f032380f34492601b215b4bba"
PRELAUNCH_ARCHIVE_SHA = "0045537f71e07eee9e692d1ea03ae115e8c0fa771b5b83c3f3dbf37c0cb75de6"
ORIGINAL_QUALIFICATION_DEADLINE = "2026-10-13T21:45:00+00:00"
RECOVERY_MAX_WALL_SECONDS = 2700
PARENT_HASHES = {
    "runtime/run-manifest.json": "a35ed73d95748fffe21e86fbb467604d362019979f322ebb5c0b0412153a3436",
    "runtime/result.json": "aeae11edf4ae0f31f9d5a57eb510fd7932666c21dc30a3748dc3df8711671640",
    "runtime/completion.json": "83fd964b5f84c657ef86501c2a506d990e46fee2a41f435bd221fe0d667d77a3",
    "lineage-terminal.json": "c7520911c856dcec38fc24d66ca175ea0b7216d711d266f24a3669280ce5e7f4",
    "prepared.json": "cb3f51799134fda39687e021cc619806df6e83c8ed1ec5c145b5d885bb3b5158",
    "supervisor-result.json": "4174db4382810a8751c76d699464c0c86ad4abd21ce541018e02e59165c36f32",
}
DEBITS = {"augmented_solve_calls": 2, "completed_augmented_updates": 6,
          "eigenproblem_calls": 0, "physical_remeshes": 0}
NEW_CAPS = {"augmented_solve_calls": 0, "completed_augmented_updates": 0,
            "stationary_solve_calls": 0, "time_solve_calls": 0,
            "eigenproblem_calls": 2, "physical_remeshes": 0}
AGGREGATE_CAPS = {"augmented_solve_calls": 2, "completed_augmented_updates": 23,
                  "eigenproblem_calls": 2, "physical_remeshes": 0}
SOLVE_APIS = ("solve", "run", "solve_auxiliary_residual", "arclength_continuation",
              "activate_bifurcation_tracking", "steady_newton_solve", "unsteady_newton_solve",
              "adaptive_unsteady_newton_solve", "doubly_adaptive_unsteady_newton_solve")


def read_bound(record, records):
    actual = regular(record["path"])
    gate(actual["sha256"] == record["sha256"], "bound_input_digest_differs", actual["path"])
    records.append(actual)
    return runner._read_json(Path(actual["path"]))


def failed_closure(runtime, records):
    """Bind a terminal failed producer without promoting its acceptance."""
    completion = regular(runtime / "completion.json")
    complete = runner._read_json(Path(completion["path"]))
    gate(complete["status"] == "inconclusive" and complete["numerical_acceptance"] is False,
         "failed_parent_required")
    gate(approach.iii.inventory(runtime) == complete["files"], "failed_parent_inventory_differs")
    records.extend([completion, *approach.iii.completion_records(runtime, complete["files"])])
    result = runner._read_json(runtime / "result.json")
    gate(all(complete[key] == result[key] for key in result), "failed_parent_result_closure_differs")
    return result


def source_contract(parent, metadata_only):
    current = runner.component_revision(ROOT)
    runner.require_clean_source(parent["component"])
    gate(parent["component_commit"] == parent["component"]["commit"] == BASE,
         "saved_candidate_source_differs")
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", BASE, current["head"])
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", PRELAUNCH_COMMIT, current["head"])
    names = diagnostic.git(ROOT, "ls-tree", "-r", "--name-only", BASE).decode().splitlines()
    frozen = {name for name in names if name.endswith(".py")
              or name in ("pyproject.toml", "uv.lock", ".python-version")}
    allowed = {fold.ENTRYPOINT, ENTRYPOINT, TEST}
    files = diagnostic.git(ROOT, "ls-files", "--cached", "--others", "--exclude-standard").decode().splitlines()
    gate(not {name for name in files if name.endswith(".py")} - frozen - allowed,
         "unbound_extra_source")
    implementation = {}
    for name in sorted(frozen | allowed):
        actual = regular(ROOT / name)
        if name not in allowed:
            expected = hashlib.sha256(diagnostic.git(ROOT, "show", f"{BASE}:{name}")).hexdigest()
            gate(actual["sha256"] == expected, "frozen_candidate_source_changed", name)
        elif name == fold.ENTRYPOINT:
            expected = hashlib.sha256(diagnostic.git(ROOT, "show", f"{PRELAUNCH_COMMIT}:{name}")).hexdigest()
            gate(actual["sha256"] == expected, "reviewed_qualification_helper_changed")
        implementation[name] = actual
    for name, item in parent["implementation"].items():
        expected = hashlib.sha256(diagnostic.git(ROOT, "show", f"{BASE}:{name}")).hexdigest()
        gate(item["sha256"] == expected and regular(item["path"])["sha256"] == expected,
             "saved_producer_source_custody_changed", name)
    if not metadata_only:
        runner.require_clean_source(current)
        for name in allowed:
            expected = hashlib.sha256(diagnostic.git(ROOT, "show", f"{current['commit']}:{name}")).hexdigest()
            gate(implementation[name]["sha256"] == expected, "qualification_source_not_committed", name)
    return current, implementation


def lineage_contract(attempt, parent, result, records):
    prepared = read_bound(regular(attempt / "prepared.json"), records)
    authority = read_bound(prepared["lineage_declaration"], records)
    terminal = read_bound(regular(attempt / "lineage-terminal.json"), records)
    supervisor = read_bound(regular(attempt / "supervisor-result.json"), records)
    gate(supervisor["native_execution_terminal"] is True and supervisor["root_exit_observed"] is True
         and supervisor["terminal_inventory_verified"] is True
         and supervisor["numerical_acceptance"] is False, "parent_not_terminal")
    prior = authority["prior_records"]
    old_result = read_bound(prior["result"], records)
    old_parent = read_bound(prior["manifest"], records)
    old_runtime = Path(prior["result"]["path"]).parent
    gate(failed_closure(old_runtime, records) == old_result, "prior_failed_parent_differs")
    for key in ("completion", "preparation", "supervisor", "archive", "release"):
        read_bound(prior[key], records)
    historical_review = regular(authority["historical_f84_review"]["path"])
    gate(historical_review["sha256"] == authority["historical_f84_review"]["sha256"],
         "historical_method_review_custody_changed")
    records.append(historical_review)
    prior_work = {"augmented_solve_calls": 1, "completed_augmented_updates": 3,
                  "eigenproblem_calls": 0, "physical_remeshes": 0}
    gate(authority["source_commit"] == BASE and authority["prior_work"] == old_result["work"] == prior_work
         and terminal["prior_work"] == prior_work and terminal["new_work"] == result["work"] == prior_work
         and terminal["new_work_recorded"] is True and terminal["cumulative_work"] == DEBITS
         and authority["aggregate_work_caps"] == terminal["aggregate_caps"] == AGGREGATE_CAPS
         and authority["prior_one_call_cap_spent"] is True
         and authority["old_failure_or_acceptance_changed"] is False
         and terminal["old_failure_or_acceptance_changed"] is False,
         "cumulative_work_or_caps_differs")
    gate(old_result["numerical_acceptance"] is False and old_result["original_full_state_restored"] is True
         and old_parent["parent_restart"]["packet"] == parent["parent_restart"]["packet"]
         and old_parent["fold_contract"]["limits"] == parent["fold_contract"]["limits"] == fold.LIMITS,
         "prior_failure_or_stationary_lineage_differs")
    deadline = prepared["deadline_utc"]
    gate(deadline == authority["prospective_deadline_utc"], "verification_deadline_differs")
    return {"prior": prior, "authority": regular(prepared["lineage_declaration"]["path"]),
            "terminal": regular(attempt / "lineage-terminal.json"),
            "old_deadline_utc": authority["old_deadline_utc"], "deadline_utc": deadline,
            "cumulative_work": DEBITS, "aggregate_caps": AGGREGATE_CAPS,
            "new_work_caps": NEW_CAPS, "old_failure_or_acceptance_changed": False,
            "historical_method_review": historical_review,
            "fresh_strategy_review_passed": False, "fresh_final_comparison_review_passed": False}


def saved_candidate(attempt, parent, result, prepared, records):
    runtime = attempt / "runtime"
    case_path = runtime / "fold/case.json"
    case_record = regular(case_path)
    case = model.read_case(case_path)
    gate(case == fold.measured_case(prepared["case"], result["ca_fold"]),
         "saved_fold_case_changed")
    packet = runtime / "fold/checkpoints/bond_016/resume.json"
    restart = runner.validate_restart(packet, case, case_record["sha256"], component=prepared["component"],
                                      environment=prepared["environment"], azimuthal_stability=True)
    records.extend(regular(item["path"]) for item in
                   (restart["packet"], restart["state"], restart["parent_manifest"], *restart["meshes"]))
    warm = runner._read_json(runtime / "fold/samples/warm.json")
    checks = remesh.canonical_checks(case, warm["metrics"])
    gate(set(checks) == modes.GATES and all(checks.values()) and warm["sampling_preserved_state"] is True,
         "saved_fold_warm_gates_failed", checks)
    gate(approach.iii.physical_signature(warm["signature"])
         == approach.iii.physical_signature(restart["roundtrip"]["after"]),
         "saved_fold_checkpoint_differs")
    solved_path = runtime / "snapshots/003-solved-fold.dump"
    solved = runner._read_json(runtime / "snapshots/003-solved-fold.json")
    gate(solved["mode_state"]["mode"] == 0 and solved["mode_state"]["actual_pins"] == warm["actual_pins"]
         and solved["mode_state"]["parameters"] == restart["roundtrip"]["after"]["parameters"],
         "saved_fold_pins_parameters_or_mode_differs")
    references = [reference for _, _, reference in history.serialized_metadata(solved_path.read_bytes())["meshes"]]
    fold.canonical_mesh_exchange(references, attempt, restart["meshes"])
    stream_hash = hashlib.sha256(history.normalized_bytes(solved_path.read_bytes(), attempt)).hexdigest()
    gate(stream_hash == solved["full_native_stream"]
         == solved["mode_state"]["serialized_state"]["full_stream_sha256"]
         == hashlib.sha256(history.normalized_bytes(Path(restart["state"]["path"]).read_bytes(),
                                                    Path(restart["mesh_reference_base"]))).hexdigest(),
         "saved_fold_complete_stream_differs")
    layout = runner._read_json(runtime / "layout.json")
    types = np.asarray(layout["types"], dtype="<i8")
    identity = hashlib.sha256(json.dumps(layout["names"], separators=(",", ":")).encode()
                              + types.tobytes()).hexdigest()
    gate(types.shape == (fold.N,) and identity == restart["roundtrip"]["layout"]["dof_layout_sha256"]
         and layout["names"] == solved["mode_state"]["dof_names"]
         and set(layout["globals"]) == set(modes.GLOBALS)
         and all(len(layout["globals"][name]) == 1 and layout["globals"][name][0] not in layout["forced"]
                 for name in modes.GLOBALS), "saved_fold_layout_or_globals_differs")
    q = np.load(runtime / "null.npy", allow_pickle=False)
    gate(q.shape == (fold.N,) and q.dtype == np.dtype("float64") and np.isfinite(q).all()
         and float(np.linalg.norm(q)) > 0, "saved_fold_null_invalid")
    actions = runner._read_json(runtime / "null-actions.json")
    gate(result["stationary_fold_candidate_passed"] is True and result["direct_dynamic_null_passed"] is True
         and actions == result["null_actions"] and actions["stationary_null_passed"] is True
         and actions["direct_dynamic_null_passed"] is True
         and actions["steady_residual_linf"] <= case["tolerances"]["newton"]
         and actions["stationary_null_scaled_frobenius"] <= fold.LIMITS["null"]
         and actions["dynamic_null_scaled_frobenius"] <= fold.LIMITS["null"]
         and actions["mass_action_scaled_max_entry"] >= fold.LIMITS["mass"]
         and actions["null_pencil_diagnostic"]["pencil_qualified"] is True,
         "saved_fold_direct_null_gate_failed")
    records.append(case_record)
    return {"case": case, "case_record": case_record, "restart": restart, "warm": warm,
            "warm_checks": checks, "solved_signature": solved, "stream_sha256": stream_hash,
            "null": regular(runtime / "null.npy"), "layout": layout,
            "direct_null": regular(runtime / "null-actions.json")}


def admission_contract(admission, prepared, *, now=None):
    """Validate a separately issued active-phase receipt; never manufacture one."""
    records = []
    required = prepared["required_seal"]
    phase_kind = prepared.get("phase_kind", SCHEMA)
    gate(admission.get("schema") == phase_kind + "-admission"
         and admission.get("native_execution_authorized") is True
         and admission.get("execution_seal") == required
         and admission.get("released_parent_process_resumed") is False,
         "qualification_admission_differs")
    tracker = read_bound(admission["active_tracker"], records)
    gate(tracker.get("schema") == phase_kind + "-active-tracker"
         and tracker.get("status") == "running" and tracker.get("phase_kind") == phase_kind
         and tracker.get("output") == str(prepared["output"])
         and tracker.get("source_commit") == prepared["component"]["commit"]
         and tracker.get("deadline_utc") == prepared["lineage"]["deadline_utc"]
         and tracker.get("cumulative_work") == DEBITS
         and tracker.get("new_work_caps") == NEW_CAPS
         and tracker.get("reservation", {}).get("state") == "active"
         and isinstance(tracker.get("reservation", {}).get("id"), str)
         and bool(tracker["reservation"]["id"])
         and tracker.get("phase_authorized") is True,
         "qualification_active_tracker_differs")
    if phase_kind == RECOVERY_KIND:
        expected = {"recovery_phase": prepared["lineage"]["recovery_phase"],
                    "original_deadline_utc": prepared["lineage"]["original_qualification_deadline_utc"],
                    "prospective_declared_at": prepared["lineage"]["prospective_declared_at"]}
        gate(all(admission.get(key) == value and tracker.get(key) == value for key, value in expected.items()),
             "recovery_active_admission_binding_differs")
    observed = datetime.fromisoformat(tracker["checked_at"])
    current = datetime.now(UTC) if now is None else now
    gate(observed.tzinfo is not None and 0 <= (current - observed).total_seconds() <= 300,
         "qualification_active_tracker_stale")
    return records


def inherited_deadline(requested, lineage, *, now=None):
    gate(requested == lineage["deadline_utc"], "qualification_deadline_reset_refused")
    deadline = datetime.fromisoformat(requested)
    current = datetime.now(UTC) if now is None else now
    gate(deadline.tzinfo is not None and current < deadline, "inherited_qualification_deadline_expired")
    return deadline


def utc_timestamp(value, label):
    gate(isinstance(value, str), label)
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise fold.Refusal(label, value) from error
    gate(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, label)
    return stamp


def predecessor_contract(original_contract, records):
    """Bind the expired, never-started phase and its immutable archival inventory."""
    archive_path = ROOT.parents[1] / "scratch/figure3/stability/full-comparison-2026-10-12" / \
        "recovery-aa922-20261013/qualification-prelaunch-archive.json"
    archive_ref = regular(archive_path)
    gate(archive_ref["sha256"] == PRELAUNCH_ARCHIVE_SHA, "exact_prelaunch_archive_required")
    archive = read_bound(archive_ref, records)
    old_root = Path(archive["source"])
    terminal_ref = regular(old_root / "attempts/attempt-0001/execution-terminal.json")
    gate(terminal_ref["sha256"] == PRELAUNCH_TERMINAL_SHA, "exact_zero_work_prelaunch_terminal_required")
    terminal = read_bound(terminal_ref, records)
    gate(terminal.get("schema") == "stone-qualification-prelaunch-terminal-v1"
         and terminal.get("status") == "failed" and terminal.get("source_commit") == PRELAUNCH_COMMIT
         and terminal.get("original_deadline_utc") == ORIGINAL_QUALIFICATION_DEADLINE
         and terminal.get("owned_processes_terminal") is True and terminal.get("native_pid_absent") is True
         and terminal.get("matching_processes") == []
         and terminal.get("unit") == {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead",
                                     "MainPID": "0", "ControlGroup": ""}
         and all(terminal.get(key) is False for key in ("scientific_result_accepted", "numerical_acceptance",
                 "native_execution_performed", "native_execution_started", "numerical_work_started",
                 "prior_failures_or_acceptance_changed"))
         and all(type(terminal.get(key)) is int and terminal[key] == 0 for key in
                 ("new_completed_updates", "new_physical_remeshes", "new_eigen_calls"))
         and terminal.get("cumulative_prior_augmented_calls") == DEBITS["augmented_solve_calls"]
         and terminal.get("cumulative_prior_augmented_updates") == DEBITS["completed_augmented_updates"],
         "prelaunch_not_proved_terminal_with_zero_work")
    gate(terminal["absent_execution_files"] == ["runtime", "preflight.json", "launch.json", "launch-audit.json",
         "activation.json", "dispatch.json", "unit-command.json", "runner.log", "supervisor-result.json"]
         and all(not (old_root / "attempts/attempt-0001" / name).exists()
                 and not (old_root / "attempts/attempt-0001" / name).is_symlink()
                 for name in terminal["absent_execution_files"]), "expired_prelaunch_execution_appeared")
    files = archive["files"]
    gate(archive.get("simulation") == terminal["simulation"] and archive.get("attempt") == terminal["attempt"]
         and archive.get("file_count") == len(files) == 26
         and archive.get("bytes") == sum(item["bytes"] for item in files.values()) == 3336679
         and archive.get("source_deleted") is False and archive.get("destination")
         and archive.get("protocol_files_verified_at") == archive.get("verified_at")
         and archive.get("bounded_terminal_evidence") == {
             "terminal_kind": "root-failed-execution-readback", "reported_numerical_acceptance": False,
             "execution_terminal_sha256": PRELAUNCH_TERMINAL_SHA}, "prelaunch_archival_custody_differs")
    for name, row in files.items():
        gate(not Path(name).is_absolute() and ".." not in Path(name).parts, "prelaunch_inventory_path_invalid")
        item = regular(old_root / name)
        gate(item["sha256"] == row["sha256"] and Path(item["path"]).stat().st_size == row["bytes"],
             "prelaunch_immutable_inventory_changed", name)
        records.append(item)
    old_seal_ref = regular(old_root / "launch/attempt-0001/execution-seal.json")
    old_seal = read_bound(old_seal_ref, records)
    old_prepared = read_bound(regular(old_root / "attempts/attempt-0001/prepared.json"), records)
    gate(old_seal.get("schema") == SCHEMA + "-execution-seal"
         and old_seal.get("source_commit") == old_prepared["source_commit"] == PRELAUNCH_COMMIT
         and old_seal.get("contract") == original_contract
         and old_prepared["execution_seal"] == old_seal_ref
         and old_prepared["deadline_utc"] == ORIGINAL_QUALIFICATION_DEADLINE
         and old_prepared["source_files"] == old_seal["source_files"], "expired_phase_bindings_changed")
    old_run = read_bound(old_prepared["contract"], records)
    old_source = Path(old_run["runner"]["path"]).parents[1]
    for name, expected in old_seal["source_files"].items():
        committed = hashlib.sha256(diagnostic.git(ROOT, "show", f"{PRELAUNCH_COMMIT}:{name}")).hexdigest()
        item = regular(old_source / name)
        gate(committed == expected == item["sha256"], "expired_phase_source_changed", name)
        records.append(item)
    for item in old_seal["inputs"]:
        actual = regular(item["path"])
        gate(actual["sha256"] == item["sha256"], "expired_phase_input_changed", actual["path"])
        records.append(actual)
    return {"terminal": terminal_ref, "archive": archive_ref, "execution_seal": old_seal_ref,
            "source_commit": PRELAUNCH_COMMIT, "source_checkout": str(old_source), "root": str(old_root),
            "terminal_checked_at": terminal["checked_at"], "archive_verified_at": archive["verified_at"],
            "original_deadline_utc": ORIGINAL_QUALIFICATION_DEADLINE,
            "status": "failed", "new_work": {name: 0 for name in NEW_CAPS}}


def recovery_requirements(original_contract, component, implementation, environment, output, records):
    """Describe required root evidence; never issue a phase or select a new clock."""
    predecessor = predecessor_contract(original_contract, records)
    return {"schema": RECOVERY_KIND + "-declaration", "phase_kind": RECOVERY_KIND,
            "predecessor": predecessor, "original_phase_contract": original_contract,
            "output": str(output),
            "source_commit": component["commit"],
            "source_files": {name: item["sha256"] for name, item in implementation.items()},
            "pyoomph_commit": environment["commit"], "native_core_sha256": environment["native_core"]["sha256"],
            "inputs": approach.coarse.unique_records(records),
            "prior_cumulative_work": DEBITS, "new_work_caps": NEW_CAPS, "aggregate_caps": AGGREGATE_CAPS,
            "spectral_request": {"azimuthal_m": 0, "shifts": fold.LIMITS["shifts"], "neigen": 8, "max_retries": 0},
            "maximum_wall_seconds": RECOVERY_MAX_WALL_SECONDS, "cold_load_required": True,
            "original_failures_and_acceptance_preserved": True, "released_parent_process_resumed": False,
            "old_unserialized_continuation_buffer_restored": False}


def recovery_phase_contract(declaration, reference, requirements, lineage, *, now=None):
    gate(set(declaration) == set(requirements) | {"issued_by", "declared_at", "deadline_utc", "phase_authorized"}
         and all(declaration[key] == value for key, value in requirements.items())
         and declaration["issued_by"] == "root/Kelvin" and declaration["phase_authorized"] is True,
         "prospective_recovery_declaration_differs")
    declared = utc_timestamp(declaration["declared_at"], "recovery_declaration_utc_required")
    deadline = utc_timestamp(declaration["deadline_utc"], "recovery_deadline_utc_required")
    predecessor = requirements["predecessor"]
    terminal = utc_timestamp(predecessor["terminal_checked_at"], "predecessor_terminal_utc_required")
    archived = utc_timestamp(predecessor["archive_verified_at"], "predecessor_archive_utc_required")
    current = datetime.now(UTC) if now is None else now
    gate(max(terminal, archived) < declared <= current < deadline
         and 0 < (deadline - declared).total_seconds() <= RECOVERY_MAX_WALL_SECONDS,
         "recovery_phase_not_prospective_or_finite")
    gate(lineage["deadline_utc"] == predecessor["original_deadline_utc"] == ORIGINAL_QUALIFICATION_DEADLINE,
         "original_qualification_deadline_changed")
    return {**lineage, "phase_kind": RECOVERY_KIND, "recovery_phase": reference,
            "original_qualification_deadline_utc": ORIGINAL_QUALIFICATION_DEADLINE,
            "prospective_deadline_utc": declaration["deadline_utc"], "deadline_utc": declaration["deadline_utc"],
            "prospective_declared_at": declaration["declared_at"], "expired_predecessor": predecessor,
            "predecessor_new_work": predecessor["new_work"], "released_parent_process_resumed": False,
            "source_alone_authorizes_recovery": False}


def preflight(args):
    gate(args.source.resolve(strict=True) == ROOT and Path(fold.__file__).resolve() == ROOT / fold.ENTRYPOINT,
         "isolated_qualification_source_required")
    attempt = args.parent_attempt.resolve(strict=True)
    gate(not any(path.is_symlink() for path in (args.parent_attempt, *args.parent_attempt.parents)),
         "parent_attempt_symlink")
    records = []
    for relative, expected in PARENT_HASHES.items():
        item = regular(attempt / relative)
        gate(item["sha256"] == expected, "exact_failed_parent_required", relative)
        records.append(item)
    runtime = attempt / "runtime"
    parent = runner._read_json(runtime / "run-manifest.json")
    result = failed_closure(runtime, records)
    gate(parent["qualification_schema"] == fold.SCHEMA and result["schema"] == fold.SCHEMA
         and result["original_full_state_restored"] is False
         and result["refusal"]["label"] == "full_fields_pins_history_or_controls_restore_failed"
         and result["refusal"]["evidence"] == "checkpoint-restoration"
         and result["work"]["eigenproblem_calls"] == 0, "saved_failed_fold_parent_required")
    contract = parent["fold_contract"]
    original_args = SimpleNamespace(source=args.source, case=Path(contract["case"]["path"]),
        manifest=Path(contract["parent_manifest"]["path"]), packet=Path(contract["restart"]["path"]),
        state=Path(contract["state"]["path"]), spectrum=Path(contract["spectral_closure"]["path"]).parent,
        out=args.out, seal=None, check=True)
    prepared = fold.preflight(original_args)
    gate(prepared["contract"] == contract and prepared["restart"]["packet"] == parent["parent_restart"]["packet"],
         "original_stationary_spectral_contract_differs")
    component, implementation = source_contract(parent, args.check)
    records.extend(prepared["records"])
    records.extend(regular(item["path"]) for item in parent["implementation"].values())
    lineage = lineage_contract(attempt, parent, result, records)
    prepared.update(component=component, implementation=implementation)
    candidate = saved_candidate(attempt, parent, result, prepared, records)
    gate(regular(args.case) == candidate["case_record"] and model.read_case(args.case) == candidate["case"],
         "qualification_case_binding_differs")
    records = approach.coarse.unique_records(records)
    phase_contract = {"schema": SCHEMA + "-contract", "parent_attempt": str(attempt),
        "failed_parent": {name: regular(attempt / name) for name in PARENT_HASHES},
        "candidate_case": candidate["case_record"], "candidate_restart": candidate["restart"]["packet"],
        "candidate_state": candidate["restart"]["state"], "candidate_null": candidate["null"],
        "candidate_complete_stream_sha256": candidate["stream_sha256"], "direct_null": candidate["direct_null"],
        "original_limits": fold.LIMITS, "lineage": lineage, "new_work_caps": NEW_CAPS,
        "released_parent_process_resumed": False, "original_parent_restoration_passed": False,
        "original_parent_numerical_acceptance": False, **fold.NO_CLAIMS}
    recovery_path = getattr(args, "recovery_phase", None)
    phase_kind = SCHEMA
    if recovery_path is not None:
        requirements = recovery_requirements(phase_contract, component, implementation, prepared["environment"],
                                             prepared["output"], records)
        recovery_ref = regular(recovery_path)
        declaration = read_bound(recovery_ref, records)
        lineage = recovery_phase_contract(declaration, recovery_ref, requirements, lineage)
        phase_kind = RECOVERY_KIND
        phase_contract = {**phase_contract, "schema": RECOVERY_KIND + "-contract", "phase_kind": phase_kind,
                          "lineage": lineage, "recovery_phase": recovery_ref,
                          "original_qualification_deadline_utc": ORIGINAL_QUALIFICATION_DEADLINE,
                          "prospective_deadline_utc": lineage["prospective_deadline_utc"]}
    output = prepared["output"]
    prior_manifest = runner._read_json(Path(lineage["prior"]["manifest"]["path"]))
    protected = {attempt, Path(lineage["prior"]["result"]["path"]).parent.parent,
                 Path(parent["runner"]["path"]).parents[1], Path(prior_manifest["runner"]["path"]).parents[1]}
    gate(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in protected),
         "qualification_output_overlaps_parent_or_source")
    if recovery_path is not None:
        predecessor = lineage["expired_predecessor"]
        gate(all(not output.is_relative_to(Path(path)) and not Path(path).is_relative_to(output)
                 for path in (predecessor["root"], predecessor["source_checkout"])),
             "recovery_output_overlaps_expired_phase")
    records = approach.coarse.unique_records(records)
    prepared.update(origin=prepared["restart"], restart=candidate["restart"], case=candidate["case"],
                    candidate=candidate, parent_fold=parent, parent_result=result, lineage=lineage,
                    phase_kind=phase_kind, contract=phase_contract, records=records,
                    source_seals_pending=[], native_execution_authorized=False)
    seal = {"schema": phase_kind + "-execution-seal", "source_commit": component["commit"],
            "source_files": {name: item["sha256"] for name, item in implementation.items()},
            "pyoomph_commit": prepared["environment"]["commit"],
            "native_core_sha256": prepared["environment"]["native_core"]["sha256"],
            "interpreter": prepared["environment"]["interpreter"]["resolved_executable"],
            "inputs": list(records), "contract": phase_contract}
    prepared["required_seal"] = seal
    pending = prepared["source_seals_pending"]
    if component["clean"] is not True:
        pending.append("clean committed qualification source pending")
    if args.seal is None:
        pending.append("external exact qualification seal pending")
    else:
        gate(read_bound(regular(args.seal), records) == seal, "qualification_execution_seal_differs")
    if args.admission is None:
        pending.append("external source review and fresh active-phase admission pending")
    elif args.check and args.admission.exists():
        admission = read_bound(regular(args.admission), records)
        records.extend(admission_contract(admission, prepared))
    prepared["input_seal"] = regular(args.seal) if args.seal is not None else None
    if not args.check:
        gate(not pending, "native_qualification_not_admitted", pending)
        inherited_deadline(args.deadline, lineage)
        # The held process supplies fresh progress for the existing launcher.
        # Only its separately issued active receipt can admit native work.
    return prepared


def await_active_admission(prepared, path, deadline_text, *, clock=None, sleep=None):
    """Wait in the held process; never activate a reservation or reset its deadline."""
    clock = (lambda: datetime.now(UTC)) if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    gate(path is not None, "qualification_admission_path_required")
    path = Path(path)
    while True:
        deadline = inherited_deadline(deadline_text, prepared["lineage"], now=clock())
        gate(not any(item.is_symlink() for item in (path, *path.parents)), "qualification_admission_symlink")
        if path.exists():
            break
        sleep(min(1.0, max(0.0, (deadline - clock()).total_seconds())))
    records = []
    admission = read_bound(regular(path), records)
    records.extend(admission_contract(admission, prepared, now=clock()))
    fold.stable_inputs(prepared)
    inherited_deadline(deadline_text, prepared["lineage"], now=clock())
    admission_contract(admission, prepared, now=clock())
    prepared["records"] = approach.coarse.unique_records([*prepared["records"], *records])
    prepared["active_admission"] = regular(path)
    fold.stable_inputs(prepared)
    prepared["native_execution_authorized"] = True


class SolveDenial:
    """Block equilibrium, time evolution and continuation before initialisation."""
    def __init__(self, problem):
        self.problem, self.attempts = problem, []
        for name in SOLVE_APIS:
            gate(callable(getattr(problem, name, None)), "required_solve_denial_api_missing", name)
            setattr(problem, name, self.reject(name))

    def reject(self, name):
        def denied(*args, **kwargs):
            self.attempts.append(name)
            gate(False, "solve_forbidden_in_saved_candidate_qualification", name)
        return denied


@contextlib.contextmanager
def bound_checkpoint_loads(problem, meshes):
    """Apply the canonical exchange to both loads in the unchanged checkpoint writer."""
    present, original = "load_state" in vars(problem), problem.load_state
    def load(state, **options):
        return fold.load_with_bound_meshes(problem, state, meshes, loader=original, **options)
    problem.load_state = load
    try:
        yield
    finally:
        if present:
            problem.load_state = original
        else:
            del problem.load_state


class QualificationAdapter(fold.Adapter):
    def check(self):
        super().check()
        p = self.problem
        gate(p.get_bifurcation_tracking_mode() == "" and p.ndof() == fold.N
             and p._bifurcation_reactivation_after_adaptation is None
             and int(p._azimuthal_mode_param_m.value) == 0, "qualification_active_tracker_or_mode")
        gate(float(p.get_global_parameter("Ca_in").value) == self.prepared["case"]["physics"]["ca_in"]
             and float(p.get_global_parameter("Bo").value) == self.prepared["case"]["physics"]["bond"],
             "qualification_forcing_changed")
        gate(all(self.work[name] == 0 for name in
                 ("augmented_solve_calls", "completed_augmented_updates", "physical_remeshes")),
             "qualification_forbidden_work_recorded")
        gate(DEBITS["eigenproblem_calls"] + self.work["eigenproblem_calls"] <= AGGREGATE_CAPS["eigenproblem_calls"],
             "qualification_cumulative_eigen_cap_exceeded")
        gate(not self.solve_denial.attempts, "qualification_forbidden_solve_attempt", self.solve_denial.attempts)
        gate(self.core.get_detect_inverted_elements() is True and p._inversion_remesh_threshold == 0,
             "qualification_inversion_detection_policy_changed")
        gate(fold.owned_template_signature(p) == self.template_policy,
             "qualification_template_registration_changed")


class NativeSession:
    """Caller-owned handles and completed teardown, independent of native flags."""
    def __init__(self):
        self.problem = self.adapter = None
        self.release_attempted = self.release_verified = False

    def release(self):
        if self.problem is not None and not self.release_attempted:
            self.release_attempted = True
            self.problem.release()
            self.release_verified = True

    def receipt(self):
        return {"problem_constructed": self.problem is not None, "release_attempted": self.release_attempted,
                "release_completed": self.release_verified, "verified": self.release_verified}


def initialise(session, problem_type, prepared, output, core, ode_type, hook_type, deadline):
    session.problem = problem = fold.build_bridged(problem_type, prepared["case"])
    denial = SolveDenial(problem)
    problem.set_output_directory(str(output / "solver"))
    problem.set_linear_solver("superlu")
    problem.set_c_compiler("system")
    problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
    problem.set_eigensolver("spectra")
    session.adapter = adapter = QualificationAdapter(
        problem, prepared, output, core, ode_type, hook_type, None, deadline
    )
    adapter.solve_denial = denial
    with adapter.mesh_guard.administrative("fresh-compatible-initialisation") as event:
        problem.initialise()
        event["verified"] = True
    gate(not problem.is_distributed(), "serial_qualification_required")
    problem.get_eigen_solver().use_pardiso = False
    problem.get_eigen_solver().max_retries = 0
    problem._stationary_iii_receipts_directory = output / "loads"
    with adapter.mesh_guard.administrative("saved-candidate-load") as event:
        candidate = prepared["candidate"]
        loaded = approach.iii.restore_checked(problem, prepared["case"], prepared["restart"],
            candidate["warm"]["actual_pins"],
            prepared["restart"]["roundtrip"]["after"]["serialized_state"]["payload_sha256"])
        saved_mode = candidate["solved_signature"]["mode_state"]
        for name, value in saved_mode["flags"].items():
            setattr(problem, name, value)
        for index, steady in enumerate(saved_mode["steady"]):
            stepper = problem.time_stepper_pt(index)
            stepper.make_steady() if steady else stepper.undo_make_steady()
        event["verified"] = True
    adapter.bound_meshes = [regular(item._get_template()._meshfile) for item in problem._meshtemplate_list]
    gate([item["sha256"] for item in adapter.bound_meshes]
         == [item["sha256"] for item in prepared["restart"]["meshes"]], "loaded_candidate_mesh_content_differs")
    adapter.template_policy = fold.owned_template_signature(problem)
    adapter.check()
    write_json(output / "candidate-load.json", loaded)


def manifest(case_path, prepared, arguments):
    actual = fold.manifest(case_path, prepared["case"], prepared, arguments)
    own = prepared["implementation"][ENTRYPOINT]
    actual.update(runner=own, runner_path=own["path"], runner_sha256=own["sha256"],
                  argv=[sys.executable, own["path"], *arguments],
                  qualification_schema=prepared.get("phase_kind", SCHEMA),
                  parent_fold=prepared["contract"]["failed_parent"], qualification_contract=prepared["contract"],
                  released_parent_process_resumed=False)
    return actual


def new_result(prepared=None):
    lineage = {} if prepared is None else prepared["lineage"]
    phase_kind = SCHEMA if prepared is None else prepared.get("phase_kind", SCHEMA)
    return {"schema": phase_kind, "phase_kind": phase_kind, "status": "inconclusive", "numerical_acceptance": False,
            "original_qualification_deadline_utc": ORIGINAL_QUALIFICATION_DEADLINE,
            "prospective_deadline_utc": lineage.get("prospective_deadline_utc"),
            "prospective_declared_at": lineage.get("prospective_declared_at"),
            "recovery_phase": lineage.get("recovery_phase"), "expired_predecessor": lineage.get("expired_predecessor"),
            "source_alone_authorizes_recovery": False,
            "released_parent_process_resumed": False, "same_instance_baseline_restored": False,
            "original_parent_restoration_passed": False, "original_parent_numerical_acceptance": False,
            "old_unserialized_continuation_buffer_restored": False,
            "continuation_scope": "Fresh admitted owner captures/restores its own continuation state; "
                                  "the old un-serialized current buffer is not recovered.",
            "work": dict(NEW_CAPS, eigenproblem_calls=0), "prior_cumulative_work": DEBITS,
            "cumulative_work": dict(DEBITS), **fold.NO_CLAIMS}


def record_refusal(result, key, error):
    result.update(status="inconclusive", numerical_acceptance=False)
    result.setdefault(key, refusal(error))


def update_work(result, *adapters):
    work = {name: sum(adapter.work[name] for adapter in adapters if adapter is not None) for name in DEBITS}
    gate(all(work[name] >= result.get("work", {}).get(name, 0) for name in DEBITS),
         "qualification_work_debit_regressed")
    result["work"] = {**work, "stationary_solve_calls": 0, "time_solve_calls": 0}
    result["cumulative_work"] = {name: DEBITS[name] + work[name] for name in DEBITS}


def persist_result(output, result, label):
    """Retain spent work before cold setup, cleanup or final input checks can fail."""
    try:
        gate(Path(label).name == label, "invalid_qualification_receipt_label", label)
        directory = output / "phase-receipts"
        directory.mkdir(exist_ok=True)
        write_json(directory / (label + ".json"), {
            "schema": result.get("phase_kind", SCHEMA) + "-phase-receipt", "stage": label, "result": result,
        })
    except BaseException as error:  # noqa: BLE001 - keep in-memory debits if the sink refuses
        record_refusal(result, "receipt_write_failure", error)


def qualify_same_instance(adapter, prepared, output, arguments, result):
    from scipy import sparse

    p = adapter.problem
    original = adapter.snapshot("fresh-candidate-baseline")
    adapter.fixed = original["fixed"]
    candidate = prepared["candidate"]
    gate(original["signature"]["full_native_stream"] == candidate["stream_sha256"],
         "fresh_candidate_complete_history_differs")
    gate(original["signature"]["meshes"] == candidate["solved_signature"]["meshes"],
         "fresh_candidate_actual_mesh_graph_differs")
    result.update(saved_candidate_continuation=candidate["solved_signature"]["continuation"],
                  fresh_baseline_continuation=original["signature"]["continuation"])
    try:
        adapter.check()
        adapter.restore(original, "initial-candidate-full-roundtrip")
        fold_output = output / "fold"
        (fold_output / "solver").mkdir(parents=True)
        case_path = fold_output / "case.json"
        write_json(case_path, prepared["case"])
        write_json(fold_output / "run-manifest.json", manifest(case_path, prepared, arguments))
        runner.append_progress(fold_output, {"event": "prepared", "case_id": prepared["case"]["case_id"]}, first=True)
        warm = remesh.sample(p, model, prepared["case"], fold_output, "warm", np)
        checks = remesh.canonical_checks(prepared["case"], warm["metrics"])
        gate(set(checks) == modes.GATES and all(checks.values()), "qualification_warm_gates_failed", checks)
        gate(warm["actual_pins"] == candidate["warm"]["actual_pins"]
             and approach.iii.physical_signature(warm["signature"])
             == approach.iii.physical_signature(candidate["warm"]["signature"]),
             "qualification_candidate_fields_pins_or_history_differs")
        q = np.load(candidate["null"]["path"], allow_pickle=False)
        residual, stationary = adapter.probe("stationary-assembly", adapter.stationary_matrix)
        jacobian, mass, layout = adapter.probe("dynamic-assembly", adapter.eigen_matrix)
        gate(layout == candidate["layout"], "qualification_live_layout_differs")
        for name, matrix in (("Jstat", stationary), ("J", jacobian), ("M", mass)):
            sparse.save_npz(output / (name + ".npz"), matrix)
        np.save(output / "residual.npy", residual)
        np.save(output / "null.npy", q)
        write_json(output / "layout.json", layout)
        actions = fold.null_actions(residual, stationary, jacobian, mass, q, layout, prepared["case"])
        write_json(output / "null-actions.json", actions)
        gate(actions["stationary_null_passed"] and actions["direct_dynamic_null_passed"],
             "qualification_recomputed_null_action_failed", actions)
        result.update(stationary_fold_candidate_passed=True, direct_dynamic_null_passed=True,
                      null_actions=actions, warm_checks=checks)
        with adapter.mesh_guard.administrative("qualification-checkpoint") as event, \
                bound_checkpoint_loads(p, adapter.bound_meshes):
            checkpoint = model._checkpoint_stationary_state(p, prepared["case"], fold_output, 16, True)
        gate(event["callbacks"] == ["actions_after_adapt", "actions_after_remeshing"] * 2,
             "qualification_checkpoint_callback_inventory_differs", event)
        restart = runner.validate_restart(fold_output / checkpoint["resume_packet"], prepared["case"], sha(case_path),
            component=prepared["component"], environment=prepared["environment"], azimuthal_stability=True)
        gate(approach.iii.physical_signature(warm["signature"])
             == approach.iii.physical_signature(restart["roundtrip"]["after"]),
             "qualification_checkpoint_changed_candidate")
        adapter.restore(original, "qualification-checkpoint-restoration")
        event["verified"] = True
        pairs = []
        for index, shift in enumerate(fold.LIMITS["shifts"]):
            values, vectors = adapter.probe(f"eigen-{index}", lambda shift=shift: adapter.solve_spectrum(shift))
            rows = fold.returned_null_metrics(values, vectors, jacobian, mass, q, layout, prepared["case"])
            pairs.append({"shift": shift, "pairs": rows})
            write_json(output / f"shift-{index}.json", pairs[-1])
        matched = any(row["returned_real_null_matches"] for search in pairs for row in search["pairs"])
        result.update(returned_pairs=pairs, returned_real_null_matched=matched, fold_checkpoint=checkpoint,
                      cold_restart=restart, status="candidate_ready_for_cold_qualification")
    except BaseException as error:  # noqa: BLE001 - retain partial evidence and restore after interruption
        record_refusal(result, "refusal", error)
    finally:
        update_work(result, adapter)
        persist_result(output, result, "searches-before-final-restoration")
        try:
            adapter.restore(original, "fresh-baseline-final-restoration")
            result["same_instance_baseline_restored"] = True
        except BaseException as error:  # noqa: BLE001 - preserve any restoration failure
            record_refusal(result, "restoration_failure", error)
        result.update(restoration=adapter.receipts, physical_mesh_checks=adapter.mesh_guard.checks,
                      mesh_callbacks=adapter.mesh_guard.events, forbidden_solve_attempts=adapter.solve_denial.attempts,
                      physical_after_callbacks_observed=adapter.mesh_guard.physical_callbacks,
                      remesh_or_adapt_requests_observed=adapter.mesh_guard.requests)
        update_work(result, adapter)
    return result


def refusal(error):
    return {"label": getattr(error, "label", "native_qualification_refusal"),
            "error_type": type(error).__name__, "error": str(error), "evidence": getattr(error, "evidence", None)}


def native_bindings(prepared):
    """Import only after active admission; software failure controls replace this boundary."""
    from pyoomph import _pyoomph_core as core
    from pyoomph.generic.problem import GenericProblemHooks
    from pyoomph.meshes.mesh import ODEStorageMesh
    from pyoomph.meshes.remesher import RemesherViaRecreation

    from problems.paper_mesh import PaperTwoPhaseMesh

    gate(Path(core.__file__).resolve() == Path(prepared["environment"]["native_core"]["path"]).resolve()
         and sha(core.__file__) == diagnostic.CORE, "qualification_native_core_binding_changed")
    problem_type = model._native_problem_types()[0]
    return core, problem_type, ODEStorageMesh, GenericProblemHooks, PaperTwoPhaseMesh, RemesherViaRecreation


def release_session(session, result, label):
    try:
        session.release()
    except BaseException as error:  # noqa: BLE001 - teardown failure cannot replace the initiating refusal
        record_refusal(result, "release_failure" if label == "same_instance" else "cold_release_failure", error)
    result[label + "_released"] = session.release_verified
    result[label + "_cleanup"] = session.receipt()


def restore_detection(core, before, result, label):
    result["inversion_detection_global_restored"] = False
    try:
        core.set_detect_inverted_elements(before)
        gate(core.get_detect_inverted_elements() == before, "qualification_global_inversion_restore_failed")
        result["inversion_detection_global_restored"] = True
    except BaseException as error:  # noqa: BLE001 - retain cleanup failure and the already spent searches
        record_refusal(result, label + "_inversion_restoration_failure", error)


def same_instance_lane(session, prepared, output, arguments, deadline, bindings, result):
    core, problem_type, ode_type, hook_type, template_type, remesher_type = bindings
    before = core.get_detect_inverted_elements()
    try:
        core.set_detect_inverted_elements(True)
        initialise(session, problem_type, prepared, output, core, ode_type, hook_type, deadline)
        write_json(output / "candidate-template-ownership.json",
                   fold.inspect_inversion_templates(session.problem, template_type, remesher_type))
        qualify_same_instance(session.adapter, prepared, output, arguments, result)
    except BaseException as error:  # noqa: BLE001 - preserve every native refusal
        record_refusal(result, "refusal", error)
    finally:
        update_work(result, session.adapter)
        persist_result(output, result, "same-instance-before-release")
        release_session(session, result, "same_instance")
        restore_detection(core, before, result, "same_instance")
        update_work(result, session.adapter)
        persist_result(output, result, "same-instance-after-release")


def cold_lane(session, same_session, prepared, output, deadline, bindings, result):
    core, problem_type, ode_type, hook_type, template_type, remesher_type = bindings
    before = core.get_detect_inverted_elements()
    try:
        cold_output = output / "fresh-fold-cold"
        cold_output.mkdir()
        gate(datetime.now(UTC) < deadline, "inherited_qualification_deadline_expired")
        core.set_detect_inverted_elements(True)
        cold_prepared = {**prepared, "restart": result["cold_restart"]}
        initialise(session, problem_type, cold_prepared, cold_output, core, ode_type, hook_type, deadline)
        write_json(cold_output / "candidate-template-ownership.json",
                   fold.inspect_inversion_templates(session.problem, template_type, remesher_type))
        sampled = remesh.sample(session.problem, model, prepared["case"], cold_output, "cold-load", np)
        checks = remesh.canonical_checks(prepared["case"], sampled["metrics"])
        gate(set(checks) == modes.GATES and all(checks.values()), "qualification_cold_gates_failed", checks)
        gate(sampled["actual_pins"] == prepared["candidate"]["warm"]["actual_pins"]
             and approach.iii.physical_signature(sampled["signature"])
             == approach.iii.physical_signature(prepared["candidate"]["warm"]["signature"]),
             "qualification_cold_fields_pins_or_history_differs")
        session.adapter.check()
        gate(all(value == 0 for value in session.adapter.work.values()), "qualification_cold_performed_work")
        result.update(cold_checks=checks, cold_load_verified=True,
                      numerical_acceptance=bool(result["returned_real_null_matched"]),
                      status="complete" if result["returned_real_null_matched"] else "inconclusive")
    except BaseException as error:  # noqa: BLE001 - retain cold setup and qualification failures
        record_refusal(result, "cold_refusal", error)
        result["cold_load_verified"] = False
    finally:
        update_work(result, same_session.adapter, session.adapter)
        persist_result(output, result, "cold-before-release")
        release_session(session, result, "cold")
        restore_detection(core, before, result, "cold")
        update_work(result, same_session.adapter, session.adapter)


def execute(prepared, output, arguments, deadline, *, result=None):
    gate(prepared["native_execution_authorized"] is True, "metadata_cannot_authorise_native_qualification")
    result = new_result(prepared) if result is None else result
    same_session, cold_session = NativeSession(), NativeSession()
    try:
        fold.stable_inputs(prepared)
        gate(datetime.now(UTC) < deadline
             and deadline == utc_timestamp(prepared["lineage"]["deadline_utc"],
                                           "qualification_deadline_utc_required"),
             "inherited_qualification_deadline_expired")
        bindings = native_bindings(prepared)
        same_instance_lane(same_session, prepared, output, arguments, deadline, bindings, result)
        if result["status"] == "candidate_ready_for_cold_qualification":
            cold_lane(cold_session, same_session, prepared, output, deadline, bindings, result)
    except BaseException as error:  # noqa: BLE001 - never replace the shared monotonic result
        record_refusal(result, "refusal", error)
    finally:
        update_work(result, same_session.adapter, cold_session.adapter)
        result.pop("cold_restart", None)
        try:
            fold.stable_inputs(prepared)
        except BaseException as error:  # noqa: BLE001 - final custody failure preserves native expenditure
            record_refusal(result, "final_input_refusal", error)
        persist_result(output, result, "native-phase-return")
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "parent-attempt", "case", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("seal", "admission", "recovery-phase"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--deadline")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    if args.check:
        print(json.dumps({"verified": True, "native_imported": False, "native_execution_authorized": False,
                          "contract": prepared["contract"], "required_seal": prepared["required_seal"],
                          "source_seals_pending": prepared["source_seals_pending"]}, allow_nan=False))
        return 0
    output = prepared["output"]
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    case_path = output / "case.json"
    write_json(case_path, prepared["case"])
    write_json(output / "run-manifest.json", manifest(case_path, prepared, arguments))
    runner.append_progress(output, {"event": "prepared", "native_execution_authorized": False,
                                   "active_admission_pending": True}, first=True)
    write_json(output / "preparation.json", {"contract": prepared["contract"], "inputs": prepared["records"],
               "source_files": prepared["implementation"], "deadline_utc": args.deadline})
    result = new_result(prepared)
    try:
        await_active_admission(prepared, args.admission, args.deadline)
        write_json(output / "active-admission.json", prepared["active_admission"])
        runner.append_progress(output, {"event": "active-admission-verified", "native_execution_authorized": True})
        execute(prepared, output, arguments, datetime.fromisoformat(args.deadline), result=result)
    except BaseException as error:  # noqa: BLE001 - preserve terminal evidence after interruption
        record_refusal(result, "refusal", error)
        write_json(output / "failure.json", result)
    persist_result(output, result, "driver-return")
    try:
        write_json(output / "result.json", result)
    except BaseException as error:  # noqa: BLE001 - prior exclusive phase receipts retain the spent work
        record_refusal(result, "result_write_failure", error)
        write_json(output / "failure-result-write.json", result)
    write_json(output / "completion.json", {**result, "files": approach.iii.inventory(output)})
    runner.append_progress(output, {"event": "final", "status": result["status"],
                                    "numerical_acceptance": result["numerical_acceptance"]})
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
