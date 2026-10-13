"""Run a distinct post-C1 phase with carried numerical debt and new wall authority.

The expired phase stays terminal and unqualified. Its retained seed, numerical
methods, counters, C1 preparation, remesher selection and qualification remain
bound to the published source. Only this phase's explicit wall authority is new.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_volume_owned_resume as old

model, runner, history, diagnostic, remesh, seed = old.model, old.runner, old.history, old.diagnostic, old.remesh, old.seed
owned, c1 = old.owned, old.c1
require, sha, write_json, bind = old.require, old.sha, old.write_json, old.bind
ENTRYPOINT = "runners/run_paper_volume_post_c1.py"
BASE = "22fc160c1aa9914f760b47a573b9d2180cbbedab"
SCHEMA = "herrada-volume-post-C1-phase-v1"
SIMULATION = "paper-model-volume3p8-post-c1-medium"
OLD_SIMULATION = "paper-model-volume3p8-owned-resume-medium"
DEBITS, RESOURCES, NO_CLAIMS = old.DEBITS, old.RESOURCES, old.NO_CLAIMS
WALL_MAX = 21600
FIRST_RELEASE_NEW_UPDATES = 37
EXPIRED_CONFIG_SHA = "48f576e2ba9866be48cc0dc53908994f20025e0f491f004aa2b70acd5ade95c1"
TERMINAL_SHA = "d022dcd45a54b6ad4fcba9e5783b116a8c764d7c9e9fd5df02b14687d9e4d119"
ARCHIVE_SHA = "5ad9916abdfb4f9370d7f90499fac62a0cd7a1cfd2ad9b9654b282259d7e714a"
PROBE_CONFIG_SHA, PROBE_SHA, PROBE_COMPLETION_SHA = old.PROBE_CONFIG_SHA, old.PROBE_SHA, old.PROBE_COMPLETION_SHA
LOAD_SHA, STATE_SHA = old.LOAD_SHA, old.STATE_SHA


def utc_now():
    return datetime.now(UTC)


def utc(value):
    require(isinstance(value, str), "phase UTC timestamp must be a string")
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), "phase timestamps must be explicit UTC")
    return parsed


def phase_authority(configuration, now=None):
    now = utc_now() if now is None else now
    phase = configuration["phase"]
    require(configuration["schema"] == SCHEMA + "-configuration"
            and configuration["simulation"] == SIMULATION
            and configuration["resource_policy"] == RESOURCES,
            "post-C1 configuration or physical resource policy differs")
    require(phase["kind"] == "new_bounded_post_C1_phase"
            and isinstance(phase["phase_id"], str)
            and re.fullmatch(r"[a-zA-Z0-9._-]{1,96}", phase["phase_id"])
            and phase["phase_id"].startswith(SIMULATION + "-")
            and type(phase["wall_max_seconds"]) is int and phase["wall_max_seconds"] == WALL_MAX
            and phase["carried_debits"] == DEBITS
            and all(type(value) is int for value in phase["carried_debits"].values())
            and type(phase["bo_zero_new_update_cap"]) is int
            and phase["bo_zero_new_update_cap"] == FIRST_RELEASE_NEW_UPDATES,
            "new phase identity, carried debt or declared bounds differ")
    declared, deadline = utc(phase["declared_at_utc"]), utc(phase["deadline_utc"])
    require(utc(old.DEADLINE) <= declared <= now and 0 < (deadline - declared).total_seconds() <= WALL_MAX,
            "new phase must be declared after original expiry within its 21600s authority")
    remaining = (deadline - now).total_seconds()
    require(math.isfinite(remaining) and 0 < remaining <= WALL_MAX, "post-C1 absolute phase deadline reached")
    return copy.deepcopy(phase), remaining


class PostC1PhaseBudget(old.RetainedPhaseBudget):
    """Carry frozen hook semantics with an instance-owned new phase clock."""

    def __init__(self, authority, emit, stages, clock=time.monotonic, wall_clock=utc_now):
        self.authority, self.wall_clock = copy.deepcopy(authority), wall_clock
        self.deadline = utc(authority["deadline_utc"])
        remaining = (self.deadline - wall_clock()).total_seconds()
        require(0 < remaining <= WALL_MAX, "post-C1 absolute phase deadline reached")
        seed.PhaseBudget.__init__(self, remaining, emit, clock)
        self.updates, self.remeshes, self.remesh_starts = 3, 1, 1
        self.stages, self.resume_pending = copy.deepcopy(stages), True

    def check_wall(self):
        require(self.wall_clock() < self.deadline, "post-C1 absolute phase deadline reached")
        seed.PhaseBudget.check_wall(self)

    def before_update(self):
        self.check_wall()
        self.audit()
        require(self.label != "bond_000" or self.updates - DEBITS["updates_total"] < FIRST_RELEASE_NEW_UPDATES,
                "post-C1 Bo=0 phase-budget refusal: 37 new completed updates exhausted")
        seed.PhaseBudget.before_update(self)

    def record(self):
        return {**seed.PhaseBudget.record(self), "absolute_deadline_utc": self.authority["deadline_utc"],
                "declared_phase": self.authority, "original_phase_deadline_utc": old.DEADLINE,
                "original_phase_debits": DEBITS, "wall_max_seconds": WALL_MAX,
                "remaining_wall_seconds": max(0, (self.deadline - self.wall_clock()).total_seconds()),
                "new_wall_authority_allocated": True, "fresh_numerical_budget_allocated": False,
                "bo_zero_new_update_cap": FIRST_RELEASE_NEW_UPDATES}


def validate_terminal(terminal):
    require(terminal["simulation"] == OLD_SIMULATION and terminal["attempt"] == "attempt-0003"
            and terminal["status"] == "failed" and terminal["scientific_result_accepted"] is False
            and terminal["original_deadline_utc"] == old.DEADLINE and terminal["original_debits"] == DEBITS
            and terminal["new_completed_update_events"] == terminal["new_physical_remesh_events"] == 0
            and terminal["completed_Newton_updates_recorded"] == 0 and terminal["new_wall_authority"] is False
            and terminal["outcome"] == "original_wall_timeout_during_first_stationary_solve"
            and terminal["gas_C1_preparation_reached"] is True
            and terminal["load_restore_verification"]["verified"] is True,
            "expired original outcome, carry or unqualified status differs")


def bind_native_proof(configuration, prepared):
    records = prepared["inputs"]
    terminal_path = bind(configuration["expired_terminal"], records)
    archive_path = bind(configuration["expired_archive"], records)
    require(sha(terminal_path) == TERMINAL_SHA and sha(archive_path) == ARCHIVE_SHA,
            "reviewed native terminal or archive receipt differs")
    terminal, archive = runner._read_json(terminal_path), runner._read_json(archive_path)
    validate_terminal(terminal)
    require(utc(configuration["phase"]["declared_at_utc"]) >= utc(terminal["observed_at"]),
            "new phase precedes original terminal readback")
    require(archive["simulation"] == OLD_SIMULATION and archive["attempt"] == "attempt-0003"
            and archive["file_count"] == len(archive["files"]) == 173
            and archive["bytes"] == sum(row["bytes"] for row in archive["files"].values()) == 109299163
            and archive["source_deleted"] is False and archive["verified_at"]
            and archive["protocol_files_verified_at"], "complete retained archive proof differs")
    original, archived_destination = Path(archive["source"]), Path(archive["destination"])
    require(original.name == archived_destination.name == OLD_SIMULATION, "original archive identity differs")
    for name, row in archive["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe original archive path")
        path = bind({"path": str(original / name), "sha256": row["sha256"]}, records)
        require(path.stat().st_size == row["bytes"], "original archive byte count differs")
    attempt = original / "attempts/attempt-0003"
    runtime = attempt / "runtime"
    manifest = runner._read_json(runtime / "run-manifest.json")
    require(manifest["component_commit"] == BASE and manifest["restart_from"]["sha256"] == STATE_SHA
            and manifest["original_failure"] == prepared["original_failure"], "native proof source or original failure differs")
    load = runner._read_json(runtime / "retained-load/load.json")
    require(load["verified"] is True and load["full_stream_load_proof"]["verified"] is True
            and load["strict_post_preparation_stream_gate_unchanged"] is True
            and load["original_snapshot_gate_unchanged"] is True
            and load["snapshot_restore"]["verified"] is True and load["source_snapshot_stationary"] is False,
            "native C1 load/restore proof differs")
    directory = runtime / "retained-load"
    before, after, later = ((directory / name).read_bytes() for name in
                           ("after-endtime.dump", "after-prepared.dump", "after-pinned-reads.dump"))
    proof, _ = c1.verify_gas_preparation(before, after, Path(prepared["restart"]["meshes"][0]["path"]))
    require(proof["half_sum_values_verified"] == 82257 and load["preparation"]["verified"] is True
            and load["preparation"]["inactivity"]["verified"] is True
            and load["preparation"]["flags_before"] == load["preparation"]["flags_prepared"]
            and history.normalized_bytes(after, attempt) == history.normalized_bytes(later, attempt),
            "native prepared baseline or inactive flags differ")
    progress = [json.loads(line) for line in (runtime / "progress.jsonl").read_text().splitlines()]
    require([row["event"] for row in progress] == ["prepared", "gas_c1_dummy_prepared", "polish_budget_carried"]
            and (progress[-1]["carried_updates"], progress[-1]["carried_physical_remeshes"],
                 progress[-1]["stage_updates_at_entry"]) == (3, 1, 3), "native completed-work ledger differs")
    require(not prepared["output"].startswith(str(original) + "/")
            and not original.is_relative_to(Path(prepared["output"])), "new output overlaps expired run evidence")
    return {"terminal": configuration["expired_terminal"], "archive": configuration["expired_archive"],
            "retained_local_source": {"path": str(original), "file_count": archive["file_count"],
                                      "bytes": archive["bytes"], "hashes_verified_now": True},
            "archive_custody": {"receipt": configuration["expired_archive"],
                                "destination": str(archived_destination),
                                "verified_at": archive["verified_at"],
                                "protocol_files_verified_at": archive["protocol_files_verified_at"],
                                "historical_root_verified_receipt": True,
                                "destination_verified_now": False},
            "original_source": BASE, "original_deadline_utc": old.DEADLINE, "original_status": "failed",
            "parent_is_unqualified": True, "carried_debits": DEBITS, "new_completed_updates": 0,
            "native_C1_and_restore_verified": True, "original_failures_retained": True,
            "native_attempt_manifest": {"path": str(runtime / "run-manifest.json"), "sha256": sha(runtime / "run-manifest.json")},
            "C1_half_sum_values_verified": proof["half_sum_values_verified"]}


def preflight(args):
    records = []
    configuration_path = bind({"path": str(args.configuration), "sha256": sha(args.configuration)}, records)
    configuration = runner._read_json(configuration_path)
    authority, remaining = phase_authority(configuration)
    prepared = admit_seed(args, configuration, records)
    lineage = bind_native_proof(configuration, prepared)
    prepared.update(configuration=configuration, phase={**authority, "lineage": lineage,
                    "caps": seed.BUDGET, "fresh_numerical_budget_allocated": False,
                    "new_wall_authority_allocated": True}, lineage=lineage,
                    remaining_wall_seconds=min(remaining, (utc(authority["deadline_utc"]) - utc_now()).total_seconds()),
                    parent_is_unqualified=True, first_release_new_update_cap=FIRST_RELEASE_NEW_UPDATES)
    require(prepared["remaining_wall_seconds"] > 0, "post-C1 absolute phase deadline reached")
    return prepared


stable_inputs = old.stable_inputs
load_retained = old.load_retained



def source_records(parent, metadata_only):
    component = runner.component_revision(ROOT)
    require(parent["component_commit"] == parent["component"]["commit"] == old.BASE,
            "retained producer source differs")
    runner.require_clean_source(parent["component"])
    require(set(component["sources"]) == set(parent["component"]["sources"]), "frozen method inventory differs")
    names = set(component["sources"]) | {seed.ENTRYPOINT, old.ENTRYPOINT}
    for module_name, module in tuple(sys.modules.items()):
        if module_name.startswith(("runners.", "problems.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(ROOT), "helper imported from another component")
            names.add(str(path.relative_to(ROOT)))
    implementation = {}
    additions = {ENTRYPOINT}
    for name in sorted(names | additions):
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), "regular implementation file required")
        digest = sha(path)
        if name not in additions:
            frozen = runner._git(ROOT, "show", f"{BASE}:{name}")
            require(frozen is not None and hashlib.sha256(frozen).hexdigest() == digest,
                    "frozen 22fc helper or method changed: " + name)
        if name in parent["component"]["sources"]:
            require(parent["component"]["sources"][name]["sha256"] == digest, "parent method differs")
        implementation[name] = {"path": str(path), "sha256": digest}
    if not metadata_only:
        runner.require_clean_source(component)
        for name, record in implementation.items():
            committed = runner._git(ROOT, "show", f"{component['commit']}:{name}")
            require(committed is not None and hashlib.sha256(committed).hexdigest() == record["sha256"],
                    "implementation differs from clean committed source")
    return component, implementation


def admit_seed(args, new_configuration, records):
    require(args.source.resolve(strict=True) == ROOT, "resume must use its isolated source")
    configuration_path = bind(new_configuration["expired_resume_configuration"], records)
    require(sha(configuration_path) == EXPIRED_CONFIG_SHA, "expired resume configuration differs")
    configuration = runner._read_json(configuration_path)
    require(configuration["schema"] == old.SCHEMA + "-configuration"
            and configuration["deadline_utc"] == old.DEADLINE and configuration["resource_policy"] == RESOURCES
            and re.fullmatch(r"[a-zA-Z0-9._-]{1,96}", configuration["phase_id"]), "resume configuration or authority differs")
    cfg_path = bind(configuration["probe_configuration"], records)
    require(sha(cfg_path) == PROBE_CONFIG_SHA, "accepted probe configuration differs")
    cfg = runner._read_json(cfg_path)
    probe_path = bind(configuration["probe_summary"], records)
    require(sha(probe_path) == PROBE_SHA, "accepted ownership probe differs")
    probe = runner._read_json(probe_path)
    completion_path = bind({"path": str(probe_path.parent / "completion.json"), "sha256": PROBE_COMPLETION_SHA}, records)
    completion = runner._read_json(completion_path)
    require(completion["schema"] == "volume-remesher-ownership-probe-v1"
            and completion["status"] == "complete" and completion["numerical_acceptance"] is False,
            "probe completion differs")
    for name, digest in completion["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe probe completion path")
        bind({"path": str(probe_path.parent / name), "sha256": digest}, records)
    probe_attempt = probe_path.parent.parent
    for name in ("runner.log", "supervisor-result.json", "dispatch.json", "run-contract.json"):
        bind({"path": str(probe_attempt / name), "sha256": sha(probe_attempt / name)}, records)
    for item in [*cfg["bindings"].values(), *cfg["source_files"].values(), *cfg["api_files"].values()]:
        bind(item, records)
    for name, digest in cfg["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe probe worker path")
        bind({"path": str(cfg_path.parent / name), "sha256": digest}, records)
    snapshot = cfg["snapshot"]
    state = bind(snapshot["state"], records)
    require(sha(state) == STATE_SHA and cfg["bindings"]["snapshot"] == snapshot["state"]
            and snapshot["format"] == "memory_snapshot", "only the exact c7 after-quality snapshot is admitted")
    case_path = bind(cfg["bindings"]["case"], records)
    require(sha(case_path) == seed.CASE_SHA and args.case.resolve(strict=True) == case_path, "exact retained case required")
    case = model.read_case(case_path)
    require(case["physics"]["vhat"] == 3.8 and case["physics"]["ca_in"] == 0.02
            and case["physics"]["bond"] == 1.56, "volume resume case differs")
    parent_path = Path(cfg["bindings"]["run_manifest"]["path"])
    parent = runner._read_json(parent_path)
    component, implementation = source_records(parent, args.check)
    environment = runner.pyoomph_revision()
    runner.require_pinned_environment(environment)
    require(environment["commit"] == model.PYOOMPH_COMMIT
            and environment["native_core"]["sha256"] == diagnostic.CORE
            and environment["interpreter"]["resolved_executable"]
            == parent["pyoomph"]["interpreter"]["resolved_executable"], "native source/core/interpreter differs")
    bind(environment["native_core"], records)
    require(cfg["source"]["commit"] == old.BASE and cfg["deadline_utc"] == old.DEADLINE
            and cfg["phase_debits"] == DEBITS
            and cfg["phase_caps"] == {"total_updates": 160, "physical_remeshes": 3, "per_mesh_stage_updates": 40},
            "accepted probe phase authority differs")
    budget = runner._read_json(Path(cfg["bindings"]["phase_budget"]["path"]))
    polish = runner._read_json(Path(cfg["bindings"]["polish"]["path"]))
    failure = runner._read_json(Path(cfg["bindings"]["failure"]["path"]))
    projection = runner._read_json(Path(cfg["bindings"]["projection"]["path"]))
    load_path = parent_path.parent / "seed-quality-remesh/load.json"
    bind({"path": str(load_path), "sha256": LOAD_SHA}, records)
    require(all(budget[key] == DEBITS[key] for key in ("updates_total", "physical_remeshes_total", "physical_remesh_starts"))
            and budget["physical_remesh_pending"] is False and budget["budget_hooks_agree"] is True
            and budget["budget"] == {**seed.BUDGET, "wall_seconds": 21600}
            and len(budget["stages"]) == 1 and budget["stages"][0]["label"] == "bond_000"
            and (polish["updates_total"], polish["remeshes_total"], polish["stage_updates"]) == (3, 1, 3)
            and polish["rollback_verified"] is True
            and failure["status"] == "failed" and failure["numerical_acceptance"] is False
            and failure["error"] == "quality phase requires one shared two-phase remesher",
            "retained phase debit, guard failure or rollback differs")
    require(projection["after_sha256"] == STATE_SHA and projection["snapshot_restore"]["verified"] is True
            and {key: value for key, value in projection["after"]["signature"].items() if key != "serialized_state"}
            == snapshot["expected_signature"] and projection["after"]["pins"] == snapshot["expected_pins"],
            "raw c7 projection proof differs")
    bind({"path": str(parent_path.parent / "seed-quality-remesh/before.dump"),
          "sha256": projection["before_sha256"]}, records)
    require(probe["schema"] == "volume-remesher-ownership-probe-v1" and probe["administrative_probe_complete"] is True
            and probe["load"]["verified"] is True and probe["load"]["full_stream_load_proof"]["verified"] is True
            and probe["interpretation"] == "single_owned_physical_remesher_repeated"
            and probe["ownership"]["raw_count"] == 2 and probe["ownership"]["unique_count"] == 1
            and probe["ownership"]["registration_unchanged"] is True
            and probe["zero_operation_guard"]["violations"] == []
            and probe["phase_debits_unchanged"] == DEBITS
            and probe["absolute_deadline_utc"] == old.DEADLINE, "ownership probe proof differs")
    saved = state.read_bytes()
    metadata = history.serialized_metadata(saved)
    require(json.loads(json.dumps(metadata)) == snapshot["metadata"] and metadata["endtime"] == 8.0
            and metadata["time"] == case["solve"]["relax_time"]
            and runner.snapshot_bc_cache_signature(saved, mesh_reference_base=Path(snapshot["mesh_reference_base"]))[
                "payload_sha256"] == snapshot["payload_sha256"]
            and runner.state_bc_cache_marker(state) == "init", "c7 complete byte/history/BC proof differs")
    require(history.verify_snapshots(saved, (probe_path.parent / "load-before-endtime.dump").read_bytes(),
                                    (probe_path.parent / "load-after-endtime.dump").read_bytes(),
                                    Path(snapshot["mesh_reference_base"]), probe_attempt,
                                    snapshot["payload_sha256"])["verified"] is True,
            "authentic probe full-stream witnesses differ")
    meshes = snapshot["meshes"]
    require(len(meshes) == 1 and runner.state_mesh_references(state) == [item["stored_path"] for item in meshes],
            "c7 mesh closure differs")
    for mesh in meshes:
        mesh_path = bind(mesh, records)
        require(mesh_path == (Path(snapshot["mesh_reference_base"]) / mesh["stored_path"]).resolve(strict=True),
                "c7 mesh reference differs")
    require(snapshot["expected_signature"]["layout"]["ndof"] == 159597
            and snapshot["expected_signature"]["parameters"] == {"Ca_in": 0.02, "Bo": 0.0},
            "c7 fields/layout/forcing differs")
    original_phase_path = Path(cfg["bindings"]["original_phase"]["path"])
    original_phase = seed.phase_contract(original_phase_path)
    original_packet = bind(original_phase["restart_packet"], records)
    original_parent = runner._read_json(Path(runner._read_json(original_packet)["parent_manifest"]["path"]))
    original_restart = runner.validate_restart(original_packet, case, seed.CASE_SHA,
                                               component=original_parent["component"], environment=environment,
                                               azimuthal_stability=True)
    terminal_path = bind(original_phase["original_terminal"], records)
    original_records, original_failure = seed.failed_inputs(original_restart, runner._read_json(terminal_path))
    records.extend(original_records)
    require(original_failure == parent["original_failure"] == failure["original_failure"],
            "original 9b failure debit was changed or transferred")
    compiler = owned.PinnedSelection(Path(cfg["api_files"]["generic_problem"]["path"]))
    require(not any(item.is_symlink() for item in (args.out, *args.out.parents)),
            "resume output contains a symlink")
    output = runner.validate_output_path(args.out, ROOT)
    for evidence in (parent_path.parent.parent, probe_path.parent.parent):
        require(not output.is_relative_to(evidence) and not evidence.is_relative_to(output),
                "resume output overlaps the seed or probe attempt")
    wall = (utc(new_configuration["phase"]["deadline_utc"]) - utc_now()).total_seconds()
    restart = {"schema": SCHEMA + "-raw-projection-admission", "packet": cfg["bindings"]["projection"],
               "packet_kind": "projection_receipt", "parent_manifest": cfg["bindings"]["run_manifest"],
               "state": {**snapshot["state"], "format": "memory_snapshot"},
               "meshes": meshes, "mesh_reference_base": snapshot["mesh_reference_base"],
               "mesh_exchange": {item["stored_path"]: item["path"] for item in meshes},
               "stage": {"kind": "retained_after_quality_projection", "bond_index": 0,
                         "bond": 0.0, "ca_in": 0.02, "bc_mode": "init"}}
    return {"case": case, "case_path": str(case_path), "component": component, "environment": environment,
            "implementation": implementation, "inputs": records, "output": str(output), "restart": restart,
            "original_failure": original_failure, "configuration": configuration,
            "phase": {"kind": "resume_existing_phase", "phase_id": configuration["phase_id"],
                      "original_phase": cfg["bindings"]["original_phase"], "deadline_utc": old.DEADLINE,
                      "caps": seed.BUDGET, "retained_debits": DEBITS, "fresh_numerical_budget_allocated": False},
            "remaining_wall_seconds": wall, "retained_stages": budget["stages"], "probe": probe,
            "snapshot_contract": snapshot, "compiler": compiler, "schedule": model.stationary_bond_schedule(case),
            "source_seal_pending": component["clean"] is not True,
            "native_execution_authorized": not args.check and component["clean"] is True}


def execute(prepared, output):
    require(prepared["native_execution_authorized"] is True, "metadata inspection cannot authorise native resume")
    stable_inputs(prepared)
    phase = PostC1PhaseBudget(prepared["phase"],
                                lambda event, **values: runner.append_progress(output, {"event": event, **values}),
                                prepared["retained_stages"])
    import numpy as np
    from pyoomph.generic.problem import GenericProblemHooks
    from pyoomph.meshes.mesh import ODEStorageMesh
    from pyoomph.meshes.remesher import RemesherViaRecreation

    cls = owned.owned_problem_type(model._native_problem_types()[0], GenericProblemHooks,
                                   prepared["compiler"], RemesherViaRecreation, output)
    case, stages, receipts = prepared["case"], [], []
    try:
        with cls(case, initial_bond=0.0) as problem:
            seed.configure(problem, output)
            seed.attach_budget(problem, phase, GenericProblemHooks)
            directory = output / "retained-load"
            directory.mkdir()
            loaded = load_retained(problem, prepared, directory, phase, ODEStorageMesh)
            _, owner, ownership = owned.canonicalise(problem, problem._domains_remesh_on_inversion,
                                                     prepared["compiler"], RemesherViaRecreation)
            require(len(ownership["raw_entries"]) == 2 and owner._cnt == 1,
                    "fresh c7 ownership does not match the accepted alias mechanism")
            write_json(directory / "ownership.json", ownership)
            for index, bond in enumerate(prepared["schedule"]):
                phase.check_wall()
                problem.get_global_parameter("Bo").value = bond
                polish = seed.polish_stage(problem, case, output, phase, f"bond_{index:03d}", 0.5 if index == 0 else 1.0)
                if index == 0:
                    require(polish["status"] == "converged" and phase.updates - DEBITS["updates_total"] <= FIRST_RELEASE_NEW_UPDATES,
                            "post-C1 Bo=0 phase-budget refusal")
                    phase.emit("bo_zero_first_release_passed", new_completed_updates=phase.updates - DEBITS["updates_total"],
                               parent_is_unqualified=True)
                checkpoint = model._checkpoint_stationary_state(problem, case, output, index, True)
                stages.append({"bond_index": index, "bond": bond, "polish": polish,
                               "checkpoint": checkpoint, "phase_budget": phase.record()})
                stable_inputs(prepared)
            require(stages[-1]["bond"] == case["physics"]["bond"] == 1.56, "no complete Bond endpoint")
            warm = remesh.sample(problem, model, case, output, "warm-endpoint", np)
            checks = remesh.canonical_checks(case, warm["metrics"])
            receipts = getattr(problem, "_owned_remesh_receipts", [])
        cold = seed.qualify_cold(case, checkpoint, warm, output, prepared, phase, np)
        accepted = all(checks.values()) and all(cold["checks"].values())
        result = {"schema": SCHEMA, "status": "completed" if accepted else "failed",
                  "completion_reached": accepted, "numerical_acceptance": accepted,
                  "stages": stages, "checks": checks, "cold_load": cold, "cold_load_checks": cold["checks"],
                  "cold_load_verified": True, "retained_load": loaded, "ownership": ownership,
                  "physical_remesh_receipts": receipts, "phase_budget": phase.record(),
                  "original_failure": prepared["original_failure"], "lineage": prepared["lineage"],
                  "parent_is_unqualified": True, "first_release_new_update_cap": FIRST_RELEASE_NEW_UPDATES, "checkpoint": checkpoint["resume_packet"],
                  "ca_in": 0.02, "bond": 1.56, **NO_CLAIMS, **warm["metrics"]}
        stable_inputs(prepared)
        phase.check_wall()
        write_json(output / "result.json", result)
        write_json(output / "summary.json", result)
        require(accepted, "unchanged warm/cold five-gate qualification failed")
        return result
    except BaseException:
        write_json(output / "partial-result.json", {"schema": SCHEMA, "completion_reached": False,
                                                    "numerical_acceptance": False, "checkpointed_stages": stages,
                                                    "phase_budget": phase.record(), **NO_CLAIMS})
        raise
    finally:
        write_json(output / "phase-budget.json", phase.record())


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "case", "configuration", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    if args.check:
        print(json.dumps({"verified": True, "native_imported": False, "native_execution_authorized": False,
                          "source_seal_pending": prepared["source_seal_pending"], "inputs": prepared["inputs"],
                          "retained_debits": DEBITS, "remaining": {"total_updates": 157, "physical_remeshes": 2,
                                                                    "mesh_stage_updates": 37},
                          "absolute_deadline_utc": prepared["phase"]["deadline_utc"], "remaining_wall_seconds": prepared["remaining_wall_seconds"],
                          "resource_policy": RESOURCES, "seed_is_stationary": False,
                          "original_failure": prepared["original_failure"], "lineage": prepared["lineage"],
                          "new_wall_authority_allocated": True, "parent_is_unqualified": True, **NO_CLAIMS}, allow_nan=False))
        return 0
    output = Path(prepared["output"])
    stable_inputs(prepared)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    actual = seed.manifest(prepared, arguments)
    own = prepared["implementation"][ENTRYPOINT]
    actual.update(runner=own, runner_path=own["path"], runner_sha256=own["sha256"],
                  argv=[sys.executable, own["path"], *arguments], qualification_schema=SCHEMA,
                  resource_policy=RESOURCES, phase_resume=prepared["phase"], lineage=prepared["lineage"],
                  parent_is_unqualified=True, **NO_CLAIMS)
    runner.create_run_manifest(output, actual)
    runner.append_progress(output, {"event": "prepared", "retained_debits": DEBITS, "absolute_deadline_utc": prepared["phase"]["deadline_utc"]}, first=True)
    try:
        write_json(output / "preparation.json", {key: value for key, value in prepared.items()
                                                if key not in {"compiler", "case"}})
        execute(prepared, output)
        return 0
    except BaseException as error:
        write_json(output / "failure.json", {"schema": SCHEMA, "status": "failed", "error": str(error),
                                             "error_type": type(error).__name__, "completion_reached": False,
                                             "numerical_acceptance": False,
                                             "inputs_unchanged": diagnostic.parents_unchanged(prepared["inputs"]),
                                             "original_failure": prepared["original_failure"], "lineage": prepared["lineage"],
                                             "new_wall_authority_allocated": True, "parent_is_unqualified": True,
                                             "failure_class": "phase-budget-refusal" if any(word in str(error).lower()
                                                 for word in ("budget", "deadline")) else "native-or-contract-refusal",
                                             "stationary_nonexistence_claimed": False, **NO_CLAIMS})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

