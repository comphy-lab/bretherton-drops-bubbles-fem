"""Probe one retained target state with one physical remesh and bounded polish.

Loading the rejected parent establishes restart fidelity only. The cumulative
Newton/remesh debits and original absolute deadline remain in force. Only fresh
warm and cold five-gate measurements can qualify this new stationary endpoint.
``--check`` performs metadata inspection without importing the native solver.
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
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems import paper_bubble as model
from runners import paper_owned_remesher as owned
from runners import run_paper_case as runner
from runners import run_paper_volume_seed_remesh as seed
from runners import run_pressure_remesh as remesh

require, sha, write_json = seed.require, seed.sha, seed.write_json
ENTRYPOINT = "runners/run_paper_volume_terminal_remesh.py"
SCHEMA = "herrada-volume-terminal-remesh-v1"
BASE_SOURCE = "d6fb3ad1eb59f9313a86f504d94eaf4ffa9dcd82"
DEBIT_FIELDS = ("updates_total", "physical_remeshes_total", "physical_remesh_starts")
PARENT_HASHES = {
    "manifest": "91b48a256cb90ec62504db139ddc2c52eb1c4641f933afb7f79cc2e519d56404",
    "result": "f6ac70abb6e985f101c752ebbc50a3fd96c41c0d4dd097fa807c09d1e808fb2b",
    "budget": "44916cac9542e8609caf80a674feacbeeb6fe17a2b70c24d4dd4298b44f37127",
    "supervisor": "9d4e5635b79875889705850725e1ca302333b1bd53ada9af7f11fe7508fb727d",
    "packet": "1a699c543911385c663a44cbbe9114ff0e7ff8264c9b96a552bf2fa6fad9d465",
    "warm": "9c1118a2f9287dc23218a30a5d6d4bffda39bc4a079fb2036b0fc303d2a22928",
    "cold": "f1f28453dd8dbdee8ff8c28f0a009a787fa9c4c5b5784c6d4b0385bef4cd7684",
}
NO_CLAIMS = {
    "parent_acceptance_changed": False,
    "parent_is_unqualified": True,
    "lineage_aggregate_caps_reset": False,
    "fresh_numerical_budget_allocated": False,
    "new_wall_authority_allocated": False,
    "eigenanalysis_performed": False,
    "branch_identity_verified": False,
    "geometry_mechanism_established": False,
    "spatial_certificate": False,
    "dynamic_certificate": False,
    "stationary_existence_certified": False,
    "stationary_nonexistence_claimed": False,
    "unique_stationary_state_certified": False,
    "undamped_basin_certified": False,
    "full_comparison_verified": False,
}


def utc_now():
    return datetime.now(UTC)


def remaining_wall(deadline_utc, now=None):
    remaining = (datetime.fromisoformat(deadline_utc) - (utc_now() if now is None else now)).total_seconds()
    require(remaining > 0, "original absolute phase deadline reached")
    return remaining


class TerminalPhaseBudget(seed.PhaseBudget):
    """Reuse the unchanged hook protocol while retaining every cumulative debit."""

    def __init__(self, authority, emit, stages, clock=time.monotonic, wall_clock=utc_now):
        self.authority = copy.deepcopy(authority)
        self.wall_clock = wall_clock
        super().__init__(min(authority["wall_budget_seconds"],
                             remaining_wall(authority["deadline_utc"], wall_clock())), emit, clock)
        self.updates, self.remeshes, self.remesh_starts = (authority["debits"][key] for key in DEBIT_FIELDS)
        self.stages = copy.deepcopy(stages)
        self.label = "terminal_remesh"

    def check_wall(self):
        remaining_wall(self.authority["deadline_utc"], self.wall_clock())
        super().check_wall()

    def before_remesh(self, remeshers):
        if self.active:
            require(self.remesh_starts < self.authority["debits"]["physical_remesh_starts"] + 1,
                    "original physical-remesh start allowance exhausted")
        super().before_remesh(remeshers)

    def carry(self, hook):
        debits = self.authority["debits"]
        require((self.remeshes, self.remesh_starts)
                == (debits["physical_remeshes_total"] + 1, debits["physical_remesh_starts"] + 1)
                and not self.pending_remesh,
                "polish requires the single completed terminal remesh")
        super().carry(hook)
        require(hook.remaining_updates() == min(seed.BUDGET["per_mesh_stage_updates"],
                                                seed.BUDGET["total_updates"] - self.updates),
                "stationary polish reset the original aggregate update allowance")

    def record(self):
        return {**super().record(), **NO_CLAIMS,
                "absolute_deadline_utc": self.authority["deadline_utc"],
                "remaining_wall_seconds": max(0, (datetime.fromisoformat(self.authority["deadline_utc"])
                                                   - self.wall_clock()).total_seconds()),
                "retained_parent_debits": dict(self.authority["debits"]),
                "scope": "fixed_target_terminal_remesh_polish_warm_and_cold_qualification",
                "original_failure_debits_are_separate": False,
                "new_stage_requires_completed_physical_remesh": True}


def validate_resources(resources):
    """Validate a private logical-CPU declaration without embedding fleet IDs."""
    require(isinstance(resources, dict) and set(resources)
            == {"cpus", "cpu_set", "memory_gib", "memory_swap_max_bytes"},
            "resource policy must declare logical cpus and explicit memory limits")
    require(type(resources["cpus"]) is int and resources["cpus"] > 0
            and isinstance(resources["cpu_set"], str)
            and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", resources["cpu_set"]),
            "invalid private logical-CPU mask")
    cpus = [int(value) for value in resources["cpu_set"].split(",")]
    require(len(cpus) == resources["cpus"] and cpus == sorted(set(cpus)),
            "private CPU mask count, order or uniqueness differs")
    require(type(resources["memory_gib"]) is int and resources["memory_gib"] > 0
            and type(resources["memory_swap_max_bytes"]) is int
            and resources["memory_swap_max_bytes"] == 0,
            "private memory or zero-swap declaration differs")


def parent_authority(budget, configuration):
    """Derive authority from the hash-bound parent; declarations cannot extend it."""
    debits = {key: budget[key] for key in DEBIT_FIELDS}
    require(all(type(value) is int and value >= 0 for value in debits.values())
            and debits["updates_total"] < seed.BUDGET["total_updates"]
            and debits["physical_remeshes_total"] <= debits["physical_remesh_starts"]
            < seed.BUDGET["physical_remeshes"], "retained debits leave no bounded terminal probe")
    require(isinstance(configuration["carried_debits"], dict)
            and set(configuration["carried_debits"]) == set(DEBIT_FIELDS)
            and all(type(configuration["carried_debits"][key]) is int
                    and configuration["carried_debits"][key] == value for key, value in debits.items()),
            "private carried debits differ from the hash-bound parent")
    deadline_utc = budget["absolute_deadline_utc"]
    require(isinstance(deadline_utc, str) and configuration["deadline_utc"] == deadline_utc,
            "private deadline differs from the hash-bound parent")
    deadline = datetime.fromisoformat(deadline_utc)
    require(deadline.tzinfo is not None and deadline.utcoffset().total_seconds() == 0,
            "retained absolute deadline must declare UTC")
    require(all(type(budget["budget"][key]) is int and budget["budget"][key] == value
                for key, value in seed.BUDGET.items()), "retained aggregate caps differ")
    wall = budget["budget"]["wall_seconds"]
    require(type(wall) in (int, float) and math.isfinite(wall) and wall > 0,
            "retained wall budget must be finite and positive")
    remaining = {"total_updates": seed.BUDGET["total_updates"] - debits["updates_total"],
                 "physical_remeshes": min(1, seed.BUDGET["physical_remeshes"]
                                           - debits["physical_remesh_starts"])}
    require(isinstance(configuration["remaining_allowance"], dict)
            and set(configuration["remaining_allowance"]) == set(remaining)
            and all(type(configuration["remaining_allowance"][key]) is int
                    and configuration["remaining_allowance"][key] == value for key, value in remaining.items()),
            "private remaining allowance differs from the hash-bound parent")
    return {"deadline_utc": deadline_utc, "debits": debits, "wall_budget_seconds": wall,
            "aggregate_caps": dict(seed.BUDGET), "remaining": remaining}


def validate_parent(case, parent, result, budget, supervisor, packet, warm, cold, configuration):
    """Reject another target, a qualified parent, reset debits or incomplete custody."""
    require(parent["component_commit"] == parent["component"]["commit"] == BASE_SOURCE,
            "terminal parent source differs")
    runner.require_clean_source(parent["component"])
    require(parent["case_sha256"] == seed.CASE_SHA and parent["azimuthal_stability"] is True,
            "terminal parent case or layout differs")
    require(case["physics"]["ca_in"] == 0.02 and case["physics"]["bond"] == 1.56
            and case["physics"]["vhat"] == 3.8 and case["mesh"]["remeshing"] is True,
            "fixed target case differs")
    require(result["status"] == "failed" and result["completion_reached"] is False
            and result["numerical_acceptance"] is False and result["cold_load_verified"] is True,
            "parent must remain rejected after its fresh cold measurement")
    expected_checks = {"stationary": True, "lower_static_pressure": False,
                       "flux": True, "volume": True, "volume_integrals": True}
    require(result["checks"] == result["cold_load_checks"] == expected_checks,
            "parent failure is not the retained pressure predicate")
    require(warm["metrics"] == cold["metrics"] and result["cold_load"]["sample"] == cold,
            "retained warm and cold measurements differ")
    require(remesh.canonical_checks(case, warm["metrics"]) == expected_checks,
            "retained canonical gate measurements differ")
    authority = parent_authority(budget, configuration)
    require(budget["physical_remesh_pending"] is False,
            "retained physical-remesh status differs")
    require(all(result["phase_budget"][key] == debit for key, debit in authority["debits"].items())
            and result["phase_budget"]["absolute_deadline_utc"] == authority["deadline_utc"],
            "retained debits or aggregate caps differ")
    require(supervisor["status"] == "inconclusive"
            and supervisor["administrative_complete"] is True
            and supervisor["owned_processes_terminal"] is True
            and supervisor["live_process_claim"] is False
            and supervisor["terminal_protocol_verified"] is True
            and supervisor["reported_numerical_acceptance"] is False
            and supervisor["parent_acceptance_changed"] is False,
            "terminal parent custody has not closed")
    require(packet["stage"] == {"kind": "completed_polish", "bond_index": 16,
                                "bond": 1.56, "ca_in": 0.02, "bc_mode": "init"}
            and packet["state"]["format"] == "portable_file"
            and result["checkpoint"] == "checkpoints/bond_016/resume.json",
            "only the retained fixed-target portable checkpoint is admitted")
    return authority


def source_records(parent, metadata_only):
    """Freeze all imported methods/helpers; execution requires their real commit."""
    component = runner.component_revision(ROOT)
    require(set(component["sources"]) == set(parent["component"]["sources"]),
            "frozen method inventory differs")
    names = set(component["sources"]) | {ENTRYPOINT}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("runners.", "problems.")) and getattr(module, "__file__", None):
            path = Path(module.__file__)
            require(not path.is_symlink() and path.resolve().is_relative_to(ROOT),
                    "terminal-remesh helper imported from another source checkout")
            names.add(str(path.resolve().relative_to(ROOT)))
    records = {}
    for relative in sorted(names):
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), "implementation must be regular")
        digest = sha(path)
        if relative != ENTRYPOINT:
            frozen = runner._git(ROOT, "show", f"{BASE_SOURCE}:{relative}")
            require(frozen is not None and hashlib.sha256(frozen).hexdigest() == digest,
                    "frozen terminal-remesh helper changed: " + relative)
        if relative in parent["component"]["sources"]:
            require(parent["component"]["sources"][relative]["sha256"] == digest,
                    "terminal parent method changed: " + relative)
        records[relative] = {"path": str(path), "sha256": digest}
    if not metadata_only:
        runner.require_clean_source(component)
        for relative, record in records.items():
            committed = runner._git(ROOT, "show", f"{component['commit']}:{relative}")
            require(committed is not None and hashlib.sha256(committed).hexdigest() == record["sha256"],
                    "terminal-remesh implementation differs from committed source")
    return component, records


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT and Path(__file__).resolve() == ROOT / ENTRYPOINT,
            "terminal remesh must use its own isolated source")
    inputs = []

    def bind(record):
        path = seed.bound_file(record, Path.cwd())
        inputs.append({"path": str(path), "sha256": sha(path)})
        return path

    cfg_path = bind({"path": str(args.configuration), "sha256": sha(args.configuration)})
    cfg = runner._read_json(cfg_path)
    require(cfg["schema"] == SCHEMA + "-configuration"
            and re.fullmatch(r"[a-zA-Z0-9._-]{1,96}", cfg["phase_id"]),
            "terminal-remesh configuration or resource authority differs")
    validate_resources(cfg["resource_policy"])
    require(set(cfg["parent_records"]) == set(PARENT_HASHES), "terminal parent binding inventory differs")
    paths = {}
    for name, digest in PARENT_HASHES.items():
        require(cfg["parent_records"][name]["sha256"] == digest, "terminal parent artifact differs")
        paths[name] = bind(cfg["parent_records"][name])
    parent = runner._read_json(paths["manifest"])
    case_path = bind({"path": str(args.case), "sha256": seed.CASE_SHA})
    case = model.read_case(case_path)
    data = {name: runner._read_json(path) for name, path in paths.items()}
    authority = validate_parent(case, parent, data["result"], data["budget"], data["supervisor"],
                                data["packet"], data["warm"], data["cold"], cfg)
    runtime = paths["manifest"].parent
    require(paths["result"] == runtime / "result.json"
            and paths["budget"] == runtime / "phase-budget.json"
            and paths["supervisor"] == runtime.parent / "supervisor-result.json"
            and paths["packet"] == runtime / data["result"]["checkpoint"]
            and paths["warm"] == runtime / "samples/warm-endpoint.json"
            and paths["cold"] == runtime / "cold/samples/endpoint.json",
            "terminal parent files do not belong to one attempt")
    environment = runner.pyoomph_revision()
    runner.require_pinned_environment(environment)
    require(environment["native_core"]["sha256"] == remesh.diagnostic.CORE
            and environment["interpreter"]["resolved_executable"]
            == parent["pyoomph"]["interpreter"]["resolved_executable"],
            "terminal-remesh native core or interpreter differs")
    component, implementation = source_records(parent, args.check)
    restart = runner.validate_restart(paths["packet"], case, seed.CASE_SHA,
                                      component=component, environment=environment,
                                      azimuthal_stability=True)
    require(Path(restart["parent_manifest"]["path"]) == paths["manifest"], "restart binds another parent")
    for record in (restart["state"], *restart["meshes"]):
        bind(record)
    api = bind(cfg["api_problem"])
    require(sha(api) == seed.API["python_problem_sha256"], "pinned Python remesh source differs")
    compiler = owned.PinnedSelection(api)
    output = runner.validate_output_path(args.out.resolve(), ROOT)
    parent_run = runtime.parents[2]
    require(not output.is_relative_to(parent_run) and not parent_run.is_relative_to(output),
            "probe output overlaps retained parent evidence")
    wall_remaining = max(0, (datetime.fromisoformat(authority["deadline_utc"]) - utc_now()).total_seconds())
    if not args.check:
        remaining_wall(authority["deadline_utc"])
    return {"case": case, "case_path": str(case_path), "component": component,
            "environment": environment, "implementation": implementation, "inputs": inputs,
            "configuration": cfg, "authority": authority, "compiler": compiler, "restart": restart,
            "parent_result": data["result"], "retained_stages": data["budget"]["stages"],
            "remaining_wall_seconds": wall_remaining, "output": str(output),
            "native_execution_authorized": not args.check and component["clean"] is True,
            "source_seal_pending": component["clean"] is not True, **NO_CLAIMS}


def stable_inputs(prepared):
    seed.stable_inputs(prepared)
    require(runner.component_revision(ROOT) == prepared["component"], "source changed after admission")


def inspect_owned(problem, prepared, remesher_type, template_type):
    """Inspect the live ownership chain, including the active template's leaf."""
    before = owned.registration(problem)
    _, owner, receipt = owned.canonicalise(problem, problem._domains_remesh_on_inversion,
                                           prepared["compiler"], remesher_type)
    active = owner.template
    require(type(active) is template_type and active._get_template() is active,
            "active remesh template type differs or its override is not terminal")
    require(all(type(template) is template_type and template._get_template() is active
                for template in [*problem._meshtemplate_list, *problem._domains_remesh_on_inversion]),
            "original or inversion template differs from the same owned active leaf")
    require(before == owned.registration(problem), "ownership inspection changed registration")
    receipt.update(active_template_is_terminal=True, exact_template_type_verified=True)
    return receipt


def terminal_intervention(problem, prepared, output, phase, np, remesher_type, template_type):
    """One physical generation followed by the unchanged fixed-target polish."""
    case = prepared["case"]
    directory = output / "retained-load"
    directory.mkdir()
    loaded = seed.load_exact(problem, case, prepared["restart"], directory)
    ownership = inspect_owned(problem, prepared, remesher_type, template_type)
    write_json(directory / "ownership.json", ownership)
    before = remesh.sample(problem, model, case, output, "before", np)
    require(before["metrics"] == prepared["parent_result"]["cold_load"]["sample"]["metrics"],
            "fixed-target load changed the retained measured state")
    inherited = tuple(prepared["authority"]["debits"][key] for key in DEBIT_FIELDS)
    projected = (inherited[0], inherited[1] + 1, inherited[2] + 1)
    require((phase.updates, phase.remeshes, phase.remesh_starts) == inherited,
            "load or observations consumed numerical work")
    require(problem.force_remesh() is True and (phase.updates, phase.remeshes, phase.remesh_starts)
            == projected and not phase.pending_remesh,
            "explicit terminal remesh did not complete exactly one physical generation")
    normalization = remesh.normalize_pressure_dummies(problem, model, output, np)
    after_remesh = remesh.sample(problem, model, case, output, "after-remesh", np)
    require((phase.updates, phase.remeshes, phase.remesh_starts) == projected,
            "post-remesh representation preparation consumed numerical work")
    polish = seed.polish_stage(problem, case, output, phase, "terminal_remesh", 1.0)
    require(polish["status"] == "converged", "unchanged bounded stationary polish did not converge")
    inspect_owned(problem, prepared, remesher_type, template_type)
    checkpoint = model._checkpoint_stationary_state(problem, case, output, 16, True)
    warm = remesh.sample(problem, model, case, output, "warm-endpoint", np)
    return {"loaded": loaded, "ownership": ownership, "before": before,
            "after_remesh": after_remesh, "dummy_pressure_normalization": normalization,
            "polish": polish, "checkpoint": checkpoint, "warm": warm,
            "explicit_force_remesh_calls": 1,
            "state_preserved_across_intervention": False}


def execute(prepared, output):
    require(prepared["native_execution_authorized"] is True,
            "metadata inspection cannot authorise native execution")
    stable_inputs(prepared)
    phase = TerminalPhaseBudget(prepared["authority"],
                                lambda event, **values: runner.append_progress(output, {"event": event, **values}),
                                prepared["retained_stages"])
    import numpy as np
    from pyoomph.generic.problem import GenericProblemHooks
    from pyoomph.meshes.remesher import RemesherViaRecreation

    from problems.paper_mesh import PaperTwoPhaseMesh

    cls = owned.owned_problem_type(model._native_problem_types()[0], GenericProblemHooks,
                                  prepared["compiler"], RemesherViaRecreation, output)
    intervention = None
    try:
        with cls(prepared["case"], initial_bond=1.56) as problem:
            seed.configure(problem, output)
            seed.attach_budget(problem, phase, GenericProblemHooks)
            intervention = terminal_intervention(problem, prepared, output, phase, np,
                                                 RemesherViaRecreation, PaperTwoPhaseMesh)
            receipts = getattr(problem, "_owned_remesh_receipts", [])
            require(len(receipts) == 1 and receipts[0]["returned"] is True,
                    "terminal probe has more than one physical-remesh receipt")
        warm = intervention["warm"]
        checks = remesh.canonical_checks(prepared["case"], warm["metrics"])
        debits_before_cold = (phase.updates, phase.remeshes, phase.remesh_starts)
        cold = seed.qualify_cold(prepared["case"], intervention["checkpoint"], warm,
                                 output, prepared, phase, np)
        require((phase.updates, phase.remeshes, phase.remesh_starts) == debits_before_cold,
                "cold qualification consumed numerical work")
        accepted = all(checks.values()) and all(cold["checks"].values())
        phase.audit()
        phase.check_wall()
        stable_inputs(prepared)
        result = {"schema": SCHEMA, "status": "completed" if accepted else "failed",
                  "completion_reached": accepted, "numerical_acceptance": accepted,
                  "checks": checks, "cold_load_checks": cold["checks"], "cold_load": cold,
                  "cold_load_verified": True, "intervention": intervention,
                  "physical_remesh_receipts": receipts, "phase_budget": phase.record(),
                  "checkpoint": intervention["checkpoint"]["resume_packet"],
                  "parent_records": prepared["configuration"]["parent_records"],
                  "retained_parent_lineage": prepared["parent_result"]["lineage"],
                  **NO_CLAIMS, **warm["metrics"]}
        write_json(output / "result.json", result)
        write_json(output / "summary.json", result)
        require(accepted, "unchanged warm/cold five-gate qualification failed")
        return result
    except BaseException:
        write_json(output / "partial-result.json", {"schema": SCHEMA, "numerical_acceptance": False,
                                                    "completion_reached": False,
                                                    "intervention": intervention,
                                                    "phase_budget": phase.record(), **NO_CLAIMS})
        raise
    finally:
        write_json(output / "phase-budget.json", phase.record())


def manifest(prepared, arguments):
    own = prepared["implementation"][ENTRYPOINT]
    return {"schema": "pyoomph-runner-manifest-v1", "created_at": utc_now().isoformat(),
            "case_id": prepared["case"]["case_id"], "case_path": prepared["case_path"],
            "case_file": prepared["case_path"], "case_sha256": seed.CASE_SHA,
            "case_fields_used": list(model.CASE_FIELDS_USED),
            "component": prepared["component"], "component_commit": prepared["component"]["commit"],
            "pyoomph": prepared["environment"], "runner": own, "runner_path": own["path"],
            "runner_sha256": own["sha256"], "argv": [sys.executable, own["path"], *arguments],
            "azimuthal_stability": True, "implementation": prepared["implementation"],
            "units": "nondimensional (sigma=mu_liquid=R0=1)",
            "nondimensionalization": prepared["case"]["nondimensionalization"],
            "restart_from": {**prepared["restart"]["state"],
                             "parent_manifest": prepared["restart"]["parent_manifest"],
                             "packet": prepared["restart"]["packet"], "qualified_stationary": False},
            "intervention": {"kind": "fixed_target_terminal_remesh_and_stationary_polish",
                             "explicit_force_remesh_calls": 1, "initial_relaxation": 1.0,
                             "original_aggregate_caps": seed.BUDGET,
                             "retained_parent_debits": prepared["authority"]["debits"],
                             "absolute_deadline_utc": prepared["authority"]["deadline_utc"],
                             "new_updates_maximum": prepared["authority"]["remaining"]["total_updates"],
                             "new_physical_remeshes_maximum": prepared["authority"]["remaining"]["physical_remeshes"]},
            "resource_policy": prepared["configuration"]["resource_policy"],
            "qualification_schema": SCHEMA, **NO_CLAIMS}


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
        print(json.dumps({"verified": True, "native_imported": False,
                          "native_execution_authorized": False,
                          "source_seal_pending": prepared["source_seal_pending"],
                          "retained_parent_debits": prepared["authority"]["debits"],
                          "remaining": prepared["authority"]["remaining"],
                          "absolute_deadline_utc": prepared["authority"]["deadline_utc"],
                          "remaining_wall_seconds": prepared["remaining_wall_seconds"],
                          "inputs": prepared["inputs"], **NO_CLAIMS}, allow_nan=False))
        return 0
    stable_inputs(prepared)
    output = Path(prepared["output"])
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    runner.create_run_manifest(output, manifest(prepared, arguments))
    runner.append_progress(output, {"event": "prepared",
                                    "retained_parent_debits": prepared["authority"]["debits"],
                                    "absolute_deadline_utc": prepared["authority"]["deadline_utc"]}, first=True)
    try:
        write_json(output / "preparation.json", {key: value for key, value in prepared.items()
                                                if key not in {"compiler", "case"}})
        execute(prepared, output)
        return 0
    except BaseException as error:
        try:
            inputs_unchanged = remesh.diagnostic.parents_unchanged(prepared["inputs"])
        except (OSError, ValueError, TypeError, KeyError) as check_error:
            inputs_unchanged = {"verified": False, "error": str(check_error),
                                "error_type": type(check_error).__name__}
        write_json(output / "failure.json", {"schema": SCHEMA, "status": "failed",
                                             "error": str(error), "error_type": type(error).__name__,
                                             "completion_reached": False, "numerical_acceptance": False,
                                             "parent_records": prepared["configuration"]["parent_records"],
                                             "inputs_unchanged": inputs_unchanged,
                                             **NO_CLAIMS})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
