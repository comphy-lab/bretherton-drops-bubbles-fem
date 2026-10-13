"""Resume one retained projected volume seed under its existing phase authority.

The remesher selection is canonicalised only after proving one owned physical
two-phase remesher. The existing numerical caps, damping, rollback, Bond ladder
and qualification gates are unchanged. No transient or forced seed remesh is
repeated, and the original absolute deadline cannot be renewed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import paper_owned_remesher as owned
from runners import run_paper_volume_seed_remesh as seed

model, runner, history, diagnostic, remesh = seed.model, seed.runner, seed.history, seed.diagnostic, seed.remesh
require, sha, write_json = seed.require, seed.sha, seed.write_json
ENTRYPOINT = "runners/run_paper_volume_owned_resume.py"
HELPER = "runners/paper_owned_remesher.py"
BASE = "c671377ef15934ed53f0d39192fcbc44d13d0a48"
SCHEMA = "herrada-volume-owned-remesher-resume-v1"
DEADLINE = "2026-10-13T04:41:19Z"
DEBITS = {"updates_total": 3, "physical_remeshes_total": 1, "physical_remesh_starts": 1, "stage_updates": 3}
RESOURCES = {"memory_max_bytes": 32 * 1024**3, "memory_swap_max_bytes": 0}
PROBE_CONFIG_SHA = "eab58111dcd056cc72ec664fd6a6022c68d0aa3baeff62d8c020a88a7546860a"
PROBE_SHA = "e93b5736433952843a2fc5671494afc5462996eab9a4a627a4dc76c87979fdf9"
PROBE_COMPLETION_SHA = "1f3ee72f6ed5e3c2ad35d7818050ac97c8ba63ffb2f8dac4e22e4e6a9da49b1c"
LOAD_SHA = "78f128d2d8e138f72868688d5fdcf1ec5b782a01a5e6acff80836bfc709e72bb"
STATE_SHA = "c7abf9947924ea963b87ea7148cb6f23c1d440e39b11211c0a456122cb116cf3"
NO_CLAIMS = {"branch_identity_verified": False, "eigenanalysis_performed": False,
             "spatial_certificate": False, "dynamic_certificate": False, "full_comparison_verified": False,
             "actual_failed_callback_identities_recovered": False}


def utc_now():
    return datetime.now(UTC)


def remaining_wall(now=None):
    remaining = (datetime.fromisoformat(DEADLINE) - (utc_now() if now is None else now)).total_seconds()
    require(remaining > 0, "original absolute phase deadline reached")
    return remaining


class RetainedPhaseBudget(seed.PhaseBudget):
    """Retain the lost work debit and hard deadline; never allocate another phase."""

    def __init__(self, wall_seconds, emit, stages, clock=time.monotonic, wall_clock=utc_now):
        self.wall_clock = wall_clock
        require(0 < wall_seconds <= 21600, "remaining wall authority differs")
        super().__init__(min(wall_seconds, remaining_wall(wall_clock())), emit, clock)
        self.updates, self.remeshes, self.remesh_starts = 3, 1, 1
        self.stages = copy.deepcopy(stages)
        self.resume_pending = True

    def check_wall(self):
        remaining_wall(self.wall_clock())
        super().check_wall()

    def carry(self, hook):
        if not self.resume_pending:
            return super().carry(hook)
        require(hook.__class__.__module__ == model.__name__
                and hook.__class__.__qualname__ == "polish_confined_state.<locals>.Budget",
                "unknown or ambiguous polish budget hook")
        require(self.current is None and hook.active is True
                and hook.updates == hook.remeshes == hook.stage_updates == 0
                and not hook.trials and not hook.remeshing and not hook.converged,
                "new polish budget was already used")
        self.check_wall()
        self.audit()
        hook.updates, hook.remeshes, hook.stage_updates = self.updates, self.remeshes, 3
        self.current, self.resume_pending = hook, False
        item = {"label": self.label, "carried_updates": self.updates,
                "carried_physical_remeshes": self.remeshes, "stage_updates_at_entry": 3,
                "resumed_existing_mesh_stage": True}
        self.stages.append(item)
        self.emit("polish_budget_carried", **item)
        require(hook.remaining_updates() == 37, "retained first-stage update allowance differs")

    def record(self):
        return {**super().record(), "absolute_deadline_utc": DEADLINE, "original_wall_seconds": 21600,
                "remaining_wall_seconds": max(0, (datetime.fromisoformat(DEADLINE)
                                                   - self.wall_clock()).total_seconds()),
                "original_phase_debits": DEBITS, "fresh_numerical_budget_allocated": False}


def bind(record, records):
    path = seed.bound_file(record, Path.cwd())
    actual = {"path": str(path), "sha256": sha(path)}
    records.append(actual)
    return path


def source_records(parent, metadata_only):
    component = runner.component_revision(ROOT)
    require(parent["component_commit"] == parent["component"]["commit"] == BASE,
            "retained producer source differs")
    runner.require_clean_source(parent["component"])
    require(set(component["sources"]) == set(parent["component"]["sources"]), "frozen method inventory differs")
    names = set(component["sources"]) | {seed.ENTRYPOINT}
    for module_name, module in tuple(sys.modules.items()):
        if module_name.startswith(("runners.", "problems.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(ROOT), "helper imported from another component")
            names.add(str(path.relative_to(ROOT)))
    implementation = {}
    for name in sorted(names | {ENTRYPOINT, HELPER}):
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), "regular implementation file required")
        digest = sha(path)
        if name not in {ENTRYPOINT, HELPER}:
            frozen = runner._git(ROOT, "show", f"{BASE}:{name}")
            require(frozen is not None and hashlib.sha256(frozen).hexdigest() == digest,
                    "frozen c671 helper or method changed: " + name)
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


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "resume must use its isolated source")
    records = []
    configuration_path = bind({"path": str(args.configuration), "sha256": sha(args.configuration)}, records)
    configuration = runner._read_json(configuration_path)
    require(configuration["schema"] == SCHEMA + "-configuration"
            and configuration["deadline_utc"] == DEADLINE and configuration["resource_policy"] == RESOURCES
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
    require(cfg["source"]["commit"] == BASE and cfg["deadline_utc"] == DEADLINE
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
            and probe["absolute_deadline_utc"] == DEADLINE, "ownership probe proof differs")
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
    wall = remaining_wall()
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
                      "original_phase": cfg["bindings"]["original_phase"], "deadline_utc": DEADLINE,
                      "caps": seed.BUDGET, "retained_debits": DEBITS, "fresh_numerical_budget_allocated": False},
            "remaining_wall_seconds": wall, "retained_stages": budget["stages"], "probe": probe,
            "snapshot_contract": snapshot, "compiler": compiler, "schedule": model.stationary_bond_schedule(case),
            "source_seal_pending": component["clean"] is not True,
            "native_execution_authorized": not args.check and component["clean"] is True}


def stable_inputs(prepared):
    seed.stable_inputs(prepared)


def load_retained(problem, prepared, directory, phase, ode_type):
    """Retain both flat views and refuse any physical or full-stream change."""
    import numpy as np

    restart, case, expected = prepared["restart"], prepared["case"], prepared["probe"]
    runner.verify_restart_artifacts(restart)
    saved = Path(restart["state"]["path"]).read_bytes()
    problem.load_state(restart["state"]["path"], quiet=True, additional_info={"exchange_msh_file": restart["mesh_exchange"]})
    before = seed.snapshot(problem)
    intervention = history.restore_endtime(problem, saved, before, case)
    after = seed.snapshot(problem)
    proof = history.verify_snapshots(saved, before, after, Path(restart["mesh_reference_base"]),
                                    Path.cwd(), prepared["snapshot_contract"]["payload_sha256"])
    for label, raw in (("before-endtime", before), ("after-endtime", after)):
        diagnostic.atomic(directory / (label + ".dump"), raw)
    # Match the accepted probe: its first flat capture follows only the load proof.
    first, flat_first = owned.pure_read(problem, ode_type, phase.check_wall)
    physical_after_first = owned.physical_history(problem, ode_type, phase.check_wall)
    second, flat_second = owned.pure_read(problem, ode_type, phase.check_wall)
    physical_after_second = owned.physical_history(problem, ode_type, phase.check_wall)
    # Preserve all actual flat values even on a refused representation/physical check.
    for label, flat in (("first", flat_first), ("subsequent", flat_second)):
        with (directory / ("flat-pinned-" + label + ".npz")).open("xb") as stream:
            np.savez_compressed(stream, pinned_values_with_positions=flat)
    observation = {"first_read": first, "subsequent_read": second,
                   "first_read_follows_full_stream_load_proof": proof["verified"],
                   "physical_after_first_read": physical_after_first, "physical_after_subsequent_read": physical_after_second,
                   "flat_hashes_equal": np.array_equal(flat_first, flat_second),
                   "flat_arrays_are_separately_retained": True, "representation_difference_waived": False}
    write_json(directory / "pinned-read-observation.json", observation)
    require(first == expected["load"]["pure_loaded"] and second == expected["inspection_before"]
            == expected["inspection_after"], "actual load/read views differ from the accepted probe")
    require(physical_after_first == physical_after_second
            == {key: first[key] for key in ("data_history_sha256", "history_values_with_repeats")},
            "first flat read changed physical pinned values, fields, histories or pin flags")
    after_reads = seed.snapshot(problem)
    diagnostic.atomic(directory / "after-pinned-reads.dump", after_reads)
    require(history.normalized_bytes(after, Path.cwd()) == history.normalized_bytes(after_reads, Path.cwd()),
            "pinned read observations changed complete serialized physical history")
    frozen_state = seed.state_record(problem)
    restored = seed.restore(problem, frozen_state)  # Original strict signature/full-stream gate.
    record = {"verified": True, "full_stream_load_proof": proof, "endtime_restoration": intervention,
              "pinned_read_observation": observation, "snapshot_restore": restored,
              "original_snapshot_gate_unchanged": True, "source_snapshot_stationary": False}
    write_json(directory / "load.json", record)
    return record


def execute(prepared, output):
    require(prepared["native_execution_authorized"] is True, "metadata inspection cannot authorise native resume")
    stable_inputs(prepared)
    phase = RetainedPhaseBudget(prepared["remaining_wall_seconds"],
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
                  "original_failure": prepared["original_failure"], "checkpoint": checkpoint["resume_packet"],
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
                          "absolute_deadline_utc": DEADLINE, "remaining_wall_seconds": prepared["remaining_wall_seconds"],
                          "resource_policy": RESOURCES, "seed_is_stationary": False,
                          "original_failure": prepared["original_failure"], **NO_CLAIMS}, allow_nan=False))
        return 0
    output = Path(prepared["output"])
    stable_inputs(prepared)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    actual = seed.manifest(prepared, arguments)
    own = prepared["implementation"][ENTRYPOINT]
    actual.update(runner=own, runner_path=own["path"], runner_sha256=own["sha256"],
                  argv=[sys.executable, own["path"], *arguments], qualification_schema=SCHEMA,
                  resource_policy=RESOURCES, phase_resume=prepared["phase"], **NO_CLAIMS)
    runner.create_run_manifest(output, actual)
    runner.append_progress(output, {"event": "prepared", "retained_debits": DEBITS, "absolute_deadline_utc": DEADLINE}, first=True)
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
                                             "original_failure": prepared["original_failure"], **NO_CLAIMS})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
