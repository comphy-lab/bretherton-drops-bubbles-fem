"""Observe three declared finite-gas III candidate controls and their m=0 searches.

Each stationary stage has its own portable checkpoint, unchanged five gates
and fresh cold-load proof. A later numerical refusal preserves accepted stages
and is inconclusive about a fold. No automatic retry or ladder extension runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_iii_coarse_cold as coarse

iii, model, runner, diagnostic, remesh = (
    coarse.iii, coarse.model, coarse.runner, coarse.diagnostic, coarse.remesh
)
require, sha, write_json = diagnostic.require, diagnostic.sha, diagnostic.write_json
regular_record, relative_path = iii.regular_record, iii.relative_path
ENTRYPOINT = "runners/run_paper_iii_fold_approach.py"
SCHEMA = "stationary-III-approach-phase-v1"
STAGE_SCHEMA = "stationary-III-approach-stage-v1"
REQUEST_SCHEMA = "paper-returned-III-approach-m0-request-v1"
SEAL_SCHEMA = "stationary-III-approach-execution-seal-v1"
KIND = "cold_qualified_stationary_III_approach_stage"
BASE = "83e94bd4d9294fa8bf1dd9339958b0858aee27c4"
PARENT_MANIFEST_SHA = "09a43f8592a7a2fe3125bb63fd40a75aa437db91d8344fb0321d95c1b0fcc9ee"
PARENT_COMPLETION_SHA = "2d002f3d6a1cef4e8f6d9e06b9e694d474eca73c9508635f78c5fc632b7e8cf9"
PARENT_PACKET_SHA = "9e5aa571bfaebe7c661efc5a49920f158cbbcb6908b0b278f9a111d96d023ba9"
PARENT_CA = -0.0038
LADDER = (-0.00381, -0.00382, -0.00383)
MAX_UPDATES, MAX_REMESHES, MAX_CALLS, MAX_WALL = 160, 3, 6, 21600
HELPERS = (*coarse.HELPERS, ENTRYPOINT)
SCOPE = "three fixed controls on inherited III candidate lineage; no fold location or branch certificate"
NO_CLAIMS = {
    **{key: value for key, value in iii.NO_CLAIMS.items() if key != "eigenanalysis_performed"},
    "fold_location_certified": False,
    "fold_bracket_certified": False,
    "SP_certificate": False,
    "finite_spectrum_complete": False,
    "global_dominance_certified": False,
    "unique_family_identity_certified": False,
}


def is_producer(parent):
    return Path(parent.get("runner", {}).get("path", "")).name == Path(ENTRYPOINT).name


def replace_once(text, old, new):
    require(text.count(old) == 1, "frozen selector source differs")
    return text.replace(old, new)


def routed_modes_source(original):
    """Bind six exact selector edits; every other byte remains the BASE source."""
    text = original.decode()
    text = replace_once(
        text,
        "def request_contract(case, shifts, neigen, *, metadata_only=False, coarse_iii=False):\n",
        "def request_contract(case, shifts, neigen, *, metadata_only=False, coarse_iii=False, approach_iii=False):\n"
        "    if approach_iii:\n"
        "        require(not coarse_iii, \"approach and original coarse request types cannot be combined\")\n"
        "        from runners import run_paper_iii_fold_approach as approach\n\n"
        "        return approach.request_contract(case, shifts, neigen)\n",
    )
    text = replace_once(
        text,
        "    if coarse_cold.is_producer(parent):\n",
        "    if Path(parent.get(\"runner\", {}).get(\"path\", \"\")).name == \"run_paper_iii_fold_approach.py\":\n"
        "        from runners import run_paper_iii_fold_approach as approach\n\n"
        "        return approach.consumer_preflight(args, parent, case)\n"
        "    if coarse_cold.is_producer(parent):\n",
    )
    for heading in ("def parent_load_contract(prepared):", "def restore_parent(problem, prepared, output):"):
        start = text.index(heading)
        end = text.find("\ndef ", start + len(heading))
        selected = text[start:end]
        selected = replace_once(
            selected,
            'if prepared["case"]["physics"]["ca_in"] == anchor_iii.TARGET_CA:'
            if "parent_load" in heading else 'if case["physics"]["ca_in"] == anchor_iii.TARGET_CA:',
            'if (prepared.get("request", {}).get("schema") == "paper-returned-III-approach-m0-request-v1"\n'
            '            or prepared["case"]["physics"]["ca_in"] == anchor_iii.TARGET_CA):'
            if "parent_load" in heading else
            'if (prepared.get("request", {}).get("schema") == "paper-returned-III-approach-m0-request-v1"\n'
            '            or case["physics"]["ca_in"] == anchor_iii.TARGET_CA):',
        )
        text = text[:start] + selected + text[end:]
    text = replace_once(
        text,
        '    expected = request_contract(prepared["case"], SHIFTS, NEIGEN, coarse_iii=coarse)\n',
        '    approach = request.get("schema") == "paper-returned-III-approach-m0-request-v1"\n'
        '    expected = request_contract(prepared["case"], SHIFTS, NEIGEN, coarse_iii=coarse, approach_iii=approach)\n'
        '    if approach:\n'
        '        require(prepared.get("execution_authorized") is True\n'
        '                and prepared.get("restart", {}).get("schema") == runner.RESUME_SCHEMA,\n'
        '                "approach execution requires an admitted completed-polish packet and execution seal")\n',
    )
    text = replace_once(
        text,
        '    if prepared["request"]["schema"] == coarse_cold.REQUEST_SCHEMA:\n',
        '    if prepared["request"]["schema"] in {coarse_cold.REQUEST_SCHEMA, "paper-returned-III-approach-m0-request-v1"}:\n',
    )
    return text.encode()


def source_contract(parent, *, metadata_only):
    current = runner.component_revision(ROOT)
    runner.require_clean_source(parent["component"])
    require(parent["component_commit"] == parent["component"]["commit"] == BASE,
            "only the qualified coarse III source is admitted")
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", BASE, current["head"])
    require(set(current["sources"]) == set(parent["component"]["sources"]),
            "method inventory differs")
    for name, record in parent["component"]["sources"].items():
        expected = hashlib.sha256(diagnostic.git(ROOT, "show", f"{BASE}:{name}")).hexdigest()
        require(record["committed_matches"] is True
                and record["sha256"] == record["committed_sha256"] == expected
                == current["sources"][name]["sha256"], "frozen numerical method changed: " + name)
    implementation = {name: regular_record(ROOT / name) for name in HELPERS}
    for name in coarse.HELPERS:
        original = diagnostic.git(ROOT, "show", f"{BASE}:{name}")
        require(parent["implementation"][name]["sha256"] == hashlib.sha256(original).hexdigest(),
                "coarse producer helper declaration differs")
        expected = routed_modes_source(original) if name == "runners/run_paper_modes.py" else original
        require((ROOT / name).read_bytes() == expected, "undeclared helper/selector edit: " + name)
    if not metadata_only:
        runner.require_clean_source(current)
        for name, record in implementation.items():
            require(hashlib.sha256(diagnostic.git(ROOT, "show", f"{current['commit']}:{name}")).hexdigest()
                    == record["sha256"], "implementation is not sealed by the source commit")
    return current, implementation


def read_parent(manifest_path, packet_path, state_path):
    manifest_path = Path(regular_record(manifest_path)["path"])
    runtime = manifest_path.parent
    require(manifest_path.name == "run-manifest.json"
            and sha(manifest_path) == PARENT_MANIFEST_SHA
            and sha(runtime / "completion.json") == PARENT_COMPLETION_SHA
            and sha(packet_path) == PARENT_PACKET_SHA, "exact typed coarse parent required")
    parent = runner._read_json(manifest_path)
    require(coarse.is_producer(parent), "coarse cold producer required")
    case_path = Path(regular_record(parent["case_path"])["path"])
    case = model.read_case(case_path)
    require(sha(case_path) == parent["case_sha256"] == coarse.CASE_SHA
            and case["physics"]["ca_in"] == PARENT_CA, "original coarse case differs")
    origin = parent["origin"]
    original = coarse.origin_evidence(SimpleNamespace(
        manifest=Path(origin["manifest"]["path"]), packet=Path(origin["packet"]["path"]),
        state=Path(origin["state"]["path"]), case=Path(origin["case"]["path"]),
        archive=Path(origin["archive"]["path"])), case)
    restart = coarse.read_packet(packet_path, runtime, case, parent, original)
    require(Path(restart["state"]["path"]) == Path(state_path).resolve(strict=True),
            "coarse parent state differs")
    summary, cold, records = coarse.admit_observation(runtime, parent, restart, case, original)
    records = coarse.unique_records([*records, *original["records"], regular_record(case_path)])
    return {"manifest": parent, "case": case, "case_path": case_path, "restart": restart,
            "summary": summary, "cold": cold, "records": records}


def request_contract(case, shifts, neigen):
    from runners import run_paper_modes as modes

    case = model.validate_case(case)
    p = case["physics"]
    require(p["ca_in"] in LADDER and p["problem"] == "finite_gas_bubble"
            and p["bond"] == 1.56 and p["domain_over_r"] == 15 and p["vhat"] == 3.3
            and p["r0"] == 1 and tuple(shifts) == modes.SHIFTS and neigen == modes.NEIGEN,
            "only the three declared m0 approach requests are admitted")
    return {"schema": REQUEST_SCHEMA, "branch_candidate": "III inherited candidate lineage",
            "ca_in": p["ca_in"], "modes": [0], "shifts": list(modes.SHIFTS),
            "neigen": modes.NEIGEN, "eigenproblem_calls": 2,
            "paper_time_factor": p["r_tube"] / p["r0"], "limits": modes.LIMITS,
            "reference_metadata_only": False, "scope": modes.SCOPE,
            "finite_spectrum_complete": False, "global_dominance_certified": False}


def control_contract(original, target, values):
    contract = iii.anchor.transition_contract(original, target)
    require(original["physics"]["ca_in"] == PARENT_CA
            and target["physics"]["ca_in"] == LADDER[-1]
            and values == list(LADDER)
            and original["solve"]["max_newton_iterations"] == 40,
            "only the three fixed approach controls and inherited solver caps are admitted")
    return {**contract, "values": values, "maximum_ca_steps": 3,
            "helper_per_mesh_stage_newton_cap": 40, "helper_per_polish_newton_cap": 160,
            "helper_per_polish_remesh_cap": 3,
            "phase_newton_update_cap": MAX_UPDATES, "phase_physical_remesh_cap": MAX_REMESHES,
            "phase_eigenproblem_call_cap": MAX_CALLS, "phase_wall_seconds_cap": MAX_WALL,
            "explicit_terminal_remesh_calls": 0, "automatic_retry_or_extension": False,
            "m0_requests": [request_contract(iii.anchor.stage_case(target, value, index, index == 3),
                                             (-0.001, 0.001), 8)
                            for index, value in enumerate(LADDER, 1)]}


def seal_contract(purpose, component, implementation, environment, records, contract):
    return {**coarse.seal_contract(purpose, component, implementation, environment, records),
            "schema": SEAL_SCHEMA, "control_contract": contract}


def execution_seal(args, component, implementation, environment, records, contract, purpose):
    expected = seal_contract(purpose, component, implementation, environment, records, contract)
    pending = []
    if component["clean"] is not True:
        pending.append("clean committed source seal pending")
    record = regular_record(args.seal) if getattr(args, "seal", None) is not None else None
    if record is None:
        pending.append("external source/input/control execution seal pending")
    else:
        require(runner._read_json(Path(record["path"])) == expected, "execution seal differs")
    if not args.check:
        runner.require_clean_source(component)
        require(not pending, "native execution requires the committed source and external execution seal")
    return expected, pending, record


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "runner must use its isolated source")
    helper_origins()
    parent = read_parent(args.manifest, args.packet, args.state)
    component, implementation = source_contract(parent["manifest"], metadata_only=args.check)
    environment = runner.pyoomph_revision()
    core = coarse.environment_contract(environment, parent["manifest"]["pyoomph"])
    target = model.read_case(args.case)
    values = iii.anchor.read_ladder(args.ca_ladder, parent["case"], target)
    contract = control_contract(parent["case"], target, values)
    contract.update(parent_case=regular_record(parent["case_path"]), target_case=regular_record(args.case),
                    ca_ladder=regular_record(args.ca_ladder))
    records = coarse.unique_records([*parent["records"], regular_record(args.case),
                                     regular_record(args.ca_ladder), core])
    seal, pending, record = execution_seal(args, component, implementation, environment, records,
                                           contract, "approach-producer")
    return {"parent": parent["case"], "target": target, "restart": parent["restart"],
            "parent_cold": parent["cold"], "parent_manifest": parent["manifest"],
            "component": component, "implementation": implementation, "environment": environment,
            "contract": contract, "values": values,
            "records": coarse.unique_records([*records, *([record] if record else [])]),
            "required_seal": seal, "input_seal": record, "source_seals_pending": pending,
            "execution_authorized": not args.check and not pending}


def helper_origins():
    for module, name in ((coarse, coarse.ENTRYPOINT), (iii, iii.ENTRYPOINT),
                         (model, "problems/paper_bubble.py"), (runner, "runners/run_paper_case.py"),
                         (diagnostic, "runners/run_pressure_diagnostic.py"),
                         (remesh, "runners/run_pressure_remesh.py"),
                         (iii.anchor, "runners/run_paper_anchor.py"),
                         (iii.history, "runners/run_paper_cold_history.py")):
        require(Path(module.__file__).resolve() == ROOT / name, "helper imported from another source")


def manifest(case_path, case, prepared, stage, argv, incoming):
    actual = iii.anchor.manifest(case_path, case, prepared, stage)
    own = prepared["implementation"][ENTRYPOINT]
    actual.update(runner=own, runner_path=own["path"], runner_sha256=own["sha256"],
                  argv=[sys.executable, own["path"], *argv], qualification_schema=STAGE_SCHEMA,
                  qualification_scope=SCOPE, input_seal=prepared["input_seal"],
                  incoming_restart=incoming, **NO_CLAIMS)
    return actual


class PhaseBudget:
    """Count completed work once; restored-state notifications are not remeshes."""

    def __init__(self, progress=None):
        self.updates = self.remeshes = self.calls = 0
        self.before_solve_calls = 0
        self.progress = progress

    def emit(self, event, **details):
        if self.progress is not None:
            runner.append_progress(self.progress, {"event": event, "phase_work": self.receipt(), **details})

    def before_step(self):
        require(self.updates < MAX_UPDATES, "approach phase exhausted its Newton update cap")

    def after_step(self):
        self.updates += 1
        self.emit("phase-newton-step-completed")
        require(self.before_solve_calls > 0, "native before-solve campaign guard was not dispatched")
        require(self.updates <= MAX_UPDATES, "approach phase exceeded its Newton update cap")

    def before_remesh(self):
        require(self.remeshes < MAX_REMESHES, "approach phase exhausted its physical remesh cap")
        self.remeshes += 1
        self.emit("phase-remesh-started")

    def start_search(self):
        require(self.calls < MAX_CALLS, "approach phase exhausted its eigenproblem call cap")
        self.calls += 1
        self.emit("phase-spectral-search-started")

    def receipt(self):
        return {"completed_newton_updates": self.updates, "physical_remesh_calls_started": self.remeshes,
                "eigenproblem_calls_started": self.calls}


@contextmanager
def phase_guard(problem, budget):
    from pyoomph.generic.problem import GenericProblemHooks

    with install_phase_guard(problem, budget, GenericProblemHooks):
        yield


@contextmanager
def install_phase_guard(problem, budget, hook_type):
    """Install the supported dispatch guard; a plain hook type permits offline tests."""
    class Guard(hook_type):
        active = True
        before_solve_seen = False

        def actions_after_newton_step(self):
            if self.active:
                budget.after_step()
                require(self.before_solve_seen, "this native instance did not dispatch its campaign guard")

        def actions_before_remeshing(self, active_remeshers):
            if self.active:
                budget.before_remesh()

    guard = Guard()
    problem += guard
    old_instance = problem.__dict__.get("actions_before_newton_solve")
    had_instance = "actions_before_newton_solve" in problem.__dict__
    original = problem.actions_before_newton_solve

    def before_solve():
        # The original dispatch first lets the unchanged local polish hook set
        # its allowance. This stricter campaign ceiling applies afterwards.
        original()
        budget.before_step()
        budget.before_solve_calls += 1
        guard.before_solve_seen = True
        problem.max_newton_iterations = min(problem.max_newton_iterations, MAX_UPDATES - budget.updates)
        budget.emit("phase-before-solve-guard", actual_solve_update_allowance=problem.max_newton_iterations)

    problem.actions_before_newton_solve = before_solve
    try:
        yield
    finally:
        guard.active = False
        if had_instance:
            problem.actions_before_newton_solve = old_instance
        else:
            del problem.actions_before_newton_solve


def check_deadline(deadline):
    require(datetime.now(UTC) < deadline, "approach phase reached its declared wall deadline")


def qualify_stage(prepared, previous, incoming, pins, index, output, argv, budget, deadline):
    import numpy as np

    check_deadline(deadline)
    iii.anchor.stable_inputs(prepared)
    current = iii.anchor.stage_case(prepared["target"], LADDER[index - 1], index, index == 3)
    directory = output / "stages" / f"ca_{index:03d}"
    (directory / "solver").mkdir(parents=True, exist_ok=False)
    case_path = directory / "case.json"
    write_json(case_path, current)
    write_json(directory / "run-manifest.json",
               manifest(case_path, current, prepared, {"kind": KIND, "index": index,
                                                      "ca_in": current["physics"]["ca_in"]}, argv, incoming))
    runner.append_progress(directory, {"event": "prepared", "index": index}, first=True)
    before_work = budget.receipt()
    with model.build_problem(previous, initial_bond=previous["physics"]["bond"]) as problem:
        problem._stationary_iii_receipts_directory = directory / "incoming-loads"
        iii.anchor.configure(problem, directory, prepared)
        require(not problem.is_distributed(), "serial approach required")
        loaded = iii.restore_checked(problem, previous, incoming, pins,
                                     incoming["roundtrip"]["after"]["serialized_state"]["payload_sha256"])
        write_json(directory / "incoming-load.json", loaded)
        observation = directory / "incoming-observation"
        observation.mkdir()
        initial = remesh.sample(problem, model, previous, observation, "cold-load", np)
        require(all(remesh.canonical_checks(previous, initial["metrics"]).values()),
                "incoming fresh stationary gates failed")
        transition = iii.anchor.adopt_case(problem, previous, current, directory)
        check_deadline(deadline)
        with phase_guard(problem, budget):
            polish = model.polish_confined_state(problem, current, directory, "ca_step", initial_relaxation=1.0)
        require(iii.polish_bounds(polish), "inherited polish bounds differ")
        warm = remesh.sample(problem, model, current, directory, "after-polish", np)
        checks = remesh.canonical_checks(current, warm["metrics"])
        require(set(checks) == iii.GATES and all(checks.values()), "warm stationary gates failed")
        checkpoint = model._checkpoint_stationary_state(problem, current, directory, 16, True)
        restart = runner.validate_restart(directory / checkpoint["resume_packet"], current, sha(case_path),
                                          component=prepared["component"], environment=prepared["environment"],
                                          azimuthal_stability=True)
        actual_pins = diagnostic.actual_pins(problem)
        require(actual_pins == warm["actual_pins"]
                and iii.physical_signature(warm["signature"])
                == iii.physical_signature(restart["roundtrip"]["after"]),
                "checkpoint changed complete warm state or pins")
    cold_output = directory / "fresh-cold"
    cold_output.mkdir()
    check_deadline(deadline)
    with model.build_problem(current, initial_bond=current["physics"]["bond"]) as fresh:
        fresh._stationary_iii_receipts_directory = cold_output / "loads"
        iii.anchor.configure(fresh, cold_output, prepared)
        cold = iii.restore_checked(fresh, current, restart, actual_pins,
                                   restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"])
        sampled = remesh.sample(fresh, model, current, cold_output, "cold-load", np)
        cold_checks = remesh.canonical_checks(current, sampled["metrics"])
        require(set(cold_checks) == iii.GATES and all(cold_checks.values()), "fresh cold stationary gates failed")
        require(sampled["actual_pins"] == actual_pins
                and iii.physical_signature(sampled["signature"]) == iii.physical_signature(cold["loaded"]),
                "cold observations changed complete state/history or pins")
        cold["canonical_checks"] = cold_checks
        write_json(directory / "cold-load.json", cold)
    record = {"schema": STAGE_SCHEMA, "index": index, "ca_in": current["physics"]["ca_in"],
              "case": regular_record(case_path), "transition": transition, "polish": polish,
              "checkpoint": checkpoint, "resume_packet_sha256": restart["packet"]["sha256"],
              "canonical_checks": checks, "cold_load_checks": cold_checks, "cold_load_verified": True,
              "completion_reached": True, "numerical_acceptance": True, "parent_acceptance_changed": False,
              "work_before": before_work, "work_after": budget.receipt(), "scope": SCOPE, **NO_CLAIMS}
    record["eigenanalysis_performed"] = False
    write_json(directory / "result.json", record)
    write_json(directory / "summary.json", {"schema": STAGE_SCHEMA, "status": "completed",
               "completion_reached": True, "numerical_acceptance": True, "cold_load_verified": True,
               "checks": checks, "cold_load_checks": cold_checks,
               "checkpoint": checkpoint["resume_packet"], "ca_in": current["physics"]["ca_in"],
               **warm["metrics"], "scope": SCOPE, **NO_CLAIMS})
    iii.anchor.stable_inputs(prepared)
    write_json(directory / "completion.json", {"schema": STAGE_SCHEMA, "status": "complete",
               "numerical_acceptance": True, "parent_acceptance_changed": False,
               "files": iii.inventory(directory), "scope": SCOPE, **NO_CLAIMS})
    return current, restart, actual_pins, regular_record(directory / "completion.json")


def admit_stage(runtime, parent, restart, case):
    runtime = Path(runtime).resolve(strict=True)
    index = parent["anchor_stage"]["index"]
    require(type(index) is int and 1 <= index <= 3 and is_producer(parent)
            and parent["qualification_schema"] == STAGE_SCHEMA
            and parent["anchor_stage"] == {"kind": KIND, "index": index, "ca_in": LADDER[index - 1]}
            and case["physics"]["ca_in"] == LADDER[index - 1]
            and not (runtime / "failure.json").exists(), "accepted approach stage producer required")
    completion = runner._read_json(runtime / "completion.json")
    files = completion["files"]
    require(completion["schema"] == STAGE_SCHEMA and completion["status"] == "complete"
            and completion["numerical_acceptance"] is True and iii.inventory(runtime) == files,
            "accepted approach stage closure differs")
    records = [regular_record(runtime / "completion.json"), *iii.completion_records(runtime, files)]
    result, summary, cold, warm = [runner._read_json(runtime / name) for name in
                                 ("result.json", "summary.json", "cold-load.json", "samples/after-polish.json")]
    require(result["schema"] == summary["schema"] == STAGE_SCHEMA
            and result["numerical_acceptance"] is summary["numerical_acceptance"] is True
            and result["completion_reached"] is summary["completion_reached"] is True
            and result["cold_load_verified"] is summary["cold_load_verified"] is True
            and result["ca_in"] == summary["ca_in"] == case["physics"]["ca_in"]
            and result["case"] == regular_record(runtime / "case.json")
            and result["polish"] == runner._read_json(runtime / "polish/ca_step/polish.json")
            and iii.polish_bounds(result["polish"])
            and result["transition"] == runner._read_json(runtime / "transition.json")
            and result["transition"]["verified"] is True, "stage polish/control proof differs")
    before, after = result["work_before"], result["work_after"]
    require(all(type(row[key]) is int and 0 <= row[key] <= bound for row in (before, after)
                for key, bound in (("completed_newton_updates", MAX_UPDATES),
                                   ("physical_remesh_calls_started", MAX_REMESHES),
                                   ("eigenproblem_calls_started", MAX_CALLS)))
            and after["completed_newton_updates"] - before["completed_newton_updates"]
            == result["polish"]["updates_total"]
            and after["physical_remesh_calls_started"] - before["physical_remesh_calls_started"]
            == result["polish"]["remeshes_total"]
            and before["eigenproblem_calls_started"] == after["eigenproblem_calls_started"] == 2 * (index - 1),
            "accepted stage does not respect the shared campaign budget")
    checks = remesh.canonical_checks(case, warm["metrics"])
    require(set(checks) == iii.GATES and all(checks.values())
            and result["canonical_checks"] == summary["checks"] == checks
            and set(result["cold_load_checks"]) == iii.GATES
            and all(result["cold_load_checks"].values())
            and result["cold_load_checks"] == summary["cold_load_checks"] == cold["canonical_checks"]
            and warm["sampling_preserved_state"] is True
            and warm["configuration"] == coarse.declared_configuration(case)
            and cold["actual_pins"] == warm["actual_pins"]
            and restart["parent_manifest"] == regular_record(runtime / "run-manifest.json")
            and restart["stage"]["kind"] == "completed_polish"
            and restart["state"]["format"] == "portable_file"
            and restart["packet"]["sha256"] == result["resume_packet_sha256"]
            and runtime / relative_path(summary["checkpoint"]) == Path(restart["packet"]["path"])
            and summary["checkpoint"] == result["checkpoint"]["resume_packet"]
            and iii.physical_signature(warm["signature"])
            == iii.physical_signature(restart["roundtrip"]["after"]), "stage warm/cold history/pin gates differ")
    for value in (parent, completion, result, summary):
        require(all(value.get(name, False) is False for name in NO_CLAIMS), "stage contains unestablished claims")
    records.extend(iii.bound(record, runtime, files) for record in
                   (restart["packet"], restart["state"], *restart["meshes"]))
    records.extend(iii.validate_load(cold, restart, case, warm["actual_pins"],
                                    restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
                                    runtime, files))
    sampled = runner._read_json(runtime / "fresh-cold/samples/cold-load.json")
    require(sampled["sampling_preserved_state"] is True
            and sampled["configuration"] == warm["configuration"]
            and sampled["actual_pins"] == cold["actual_pins"]
            and iii.physical_signature(sampled["signature"]) == iii.physical_signature(cold["loaded"])
            and remesh.canonical_checks(case, sampled["metrics"]) == result["cold_load_checks"],
            "fresh cold sample differs")
    return summary, cold, coarse.unique_records(records)


def consumer_preflight(args, parent, case):
    require(args.source.resolve(strict=True) == ROOT, "consumer must use its isolated source")
    helper_origins()
    require(is_producer(parent) and parent["case_fields_used"] == model.CASE_FIELDS_USED
            and parent["azimuthal_stability"] is True and parent["case_sha256"] == sha(args.case)
            and model.read_case(Path(parent["case_path"])) == case
            and parent["case_id"] == case["case_id"], "approach stage case/layout differs")
    contract = parent["case_transition_contract"]
    source_parent = read_parent(parent["parent_restart"]["parent_manifest"]["path"],
                                parent["parent_restart"]["packet"]["path"],
                                parent["parent_restart"]["state"]["path"])
    target = model.read_case(Path(contract["target_case"]["path"]))
    values = iii.anchor.read_ladder(Path(contract["ca_ladder"]["path"]), source_parent["case"], target)
    expected = control_contract(source_parent["case"], target, values)
    expected.update(parent_case=regular_record(source_parent["case_path"]),
                    target_case=regular_record(contract["target_case"]["path"]),
                    ca_ladder=regular_record(contract["ca_ladder"]["path"]))
    require(contract == expected, "stage immutable phase control contract differs")
    index = parent["anchor_stage"]["index"]
    require(type(index) is int and 1 <= index <= 3
            and case == iii.anchor.stage_case(target, LADDER[index - 1], index, index == 3),
            "stage case differs from its declared Ca transition")
    component, implementation = source_contract(source_parent["manifest"], metadata_only=args.check)
    runner.require_clean_source(parent["component"])
    require(parent["component_commit"] == parent["component"]["commit"]
            and parent["implementation"] == implementation
            and {name: row["sha256"] for name, row in parent["component"]["sources"].items()}
            == {name: row["sha256"] for name, row in component["sources"].items()},
            "stage source/implementation differs")
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", parent["component_commit"], component["head"])
    environment = runner.pyoomph_revision()
    core = coarse.environment_contract(environment, parent["pyoomph"])
    restart = runner.validate_restart(args.packet, case, sha(args.case), component=component,
                                      environment=environment, azimuthal_stability=True)
    require(Path(restart["state"]["path"]) == args.state.resolve(strict=True), "stage consumer state differs")
    _summary, cold, records = admit_stage(args.manifest.resolve(strict=True).parent, parent, restart, case)
    records = coarse.unique_records([*records, *source_parent["records"], core,
                                     regular_record(contract["target_case"]["path"]),
                                     regular_record(contract["ca_ladder"]["path"])])
    seal, pending, record = execution_seal(args, component, implementation, environment, records,
                                           contract, "approach-m0-consumer")
    return {"case": case, "request": request_contract(case, args.shifts, args.neigen),
            "restart": restart, "cold": cold, "component": component,
            "implementation": implementation, "environment": environment,
            "records": coarse.unique_records([*records, *([record] if record else [])]),
            "required_seal": seal, "source_seals_pending": pending,
            "execution_authorized": not args.check and not pending}


def spectra_for_stage(prepared, stage_record, output, index, budget, deadline):
    """Use the unchanged two-shift search and comparison; emit no figures."""
    from runners import run_paper_modes as modes

    directory = Path(stage_record["path"]).parent
    case_path = directory / "case.json"
    stage_parent = runner._read_json(directory / "run-manifest.json")
    case = model.read_case(case_path)
    result = runner._read_json(directory / "result.json")
    restart = runner.validate_restart(directory / result["checkpoint"]["resume_packet"], case,
                                      sha(case_path), component=prepared["component"],
                                      environment=prepared["environment"], azimuthal_stability=True)
    _summary, cold, records = admit_stage(directory, stage_parent, restart, case)
    request = request_contract(case, modes.SHIFTS, modes.NEIGEN)
    consumer = {**prepared, "case": case, "restart": restart, "cold": cold,
                "request": request, "records": coarse.unique_records([*prepared["records"], *records])}
    spectrum = output / "spectra" / f"ca_{index:03d}"
    spectrum.mkdir(parents=True, exist_ok=False)
    write_json(spectrum / "run-manifest.json", {
        **modes.manifest(SimpleNamespace(case=case_path), consumer, []),
        "driver": prepared["implementation"][ENTRYPOINT], "accepted_stationary_stage": stage_record,
        "phase_control_contract": prepared["contract"]})
    runner.append_progress(spectrum, {"event": "prepared", "index": index}, first=True)
    require(modes.selected_modes(consumer) == (0,), "only m0 searches are admitted")
    searches = []
    for shift_index, shift in enumerate(modes.SHIFTS):
        check_deadline(deadline)
        iii.anchor.stable_inputs(prepared)
        budget.start_search()
        runner.append_progress(spectrum, {"event": "search-started", "mode": 0,
                                          "shift": shift, "phase_work": budget.receipt()})
        searches.append(modes.spectral_stage(consumer, spectrum / f"m0-shift-{shift_index}", 0, shift))
    comparison = modes.compare_searches(*searches, 0)
    actual = {"schema": "paper-returned-modes-result-v1", "request": request,
              "mode_comparisons": [comparison], "numerical_acceptance": comparison["matched_returned_envelope"],
              "scientific_status": "qualified_returned_envelopes_at_one_anchor"
              if comparison["matched_returned_envelope"] else "inconclusive_returned_envelopes",
              "scope": modes.SCOPE, "finite_spectrum_complete": False,
              "global_dominance_certified": False, "parent_acceptance_changed": False}
    write_json(spectrum / "result.json", actual)
    write_json(spectrum / "summary.json", {"status": "completed", "completion_reached": True,
               "scientific_status": actual["scientific_status"], "numerical_acceptance": actual["numerical_acceptance"],
               "scope": modes.SCOPE, "finite_spectrum_complete": False, "global_dominance_certified": False})
    write_json(spectrum / "completion.json", {"status": "complete", "files": iii.inventory(spectrum),
               "numerical_acceptance": actual["numerical_acceptance"], "parent_acceptance_changed": False})
    return regular_record(spectrum / "completion.json")


def run_stages(prepared, output, argv, budget, deadline):
    require(prepared.get("execution_authorized") is True, "metadata admission cannot authorise native work")
    accepted, spectra = [], []
    previous, incoming, pins = prepared["parent"], prepared["restart"], prepared["parent_cold"]["actual_pins"]
    error = None
    next_index = 1
    try:
        for index in range(1, 4):
            next_index = index
            previous, incoming, pins, record = qualify_stage(
                prepared, previous, incoming, pins, index, output, argv, budget, deadline)
            accepted.append(record)
            write_json(output / f"accepted-stage-{index:03d}.json", record)
            spectra.append(spectra_for_stage(prepared, record, output, index, budget, deadline))
            write_json(output / f"completed-spectrum-{index:03d}.json", spectra[-1])
        next_index = None
    except (Exception, KeyboardInterrupt, SystemExit) as failure:  # noqa: BLE001 - retain a bounded partial phase
        error = {"index": next_index, "ca_in": LADDER[next_index - 1],
                 "error_type": type(failure).__name__, "error": str(failure)}
    result = {"schema": SCHEMA, "status": "complete" if error is None else "inconclusive",
              "accepted_stages": accepted, "completed_spectra": spectra, "terminal_refusal": error,
              "completion_reached": error is None, "stationary_stages_accepted": len(accepted),
              "spectral_searches_completed": 2 * len(spectra), "work": budget.receipt(),
              "numerical_acceptance": False, "parent_acceptance_changed": False,
              "numerical_acceptance_scope": "individual stationary and returned-spectrum receipts only",
              "parents_unchanged": diagnostic.parents_unchanged(prepared["records"]),
              "scope": SCOPE, "eigenanalysis_performed": budget.calls > 0, **NO_CLAIMS}
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "ca-ladder", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--deadline", help="UTC ISO timestamp; hard unit wall cap remains a launch requirement")
    parser.add_argument("--check", action="store_true", help="metadata only; no native import or output")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    output = runner.validate_output_path(args.out, ROOT)
    require(not args.out.is_symlink()
            and not output.is_relative_to(Path(prepared["restart"]["parent_manifest"]["path"]).parent),
            "output overlaps the qualified parent or a symlink")
    if args.check:
        print(json.dumps({"verified": True, "native_imported": False, "native_execution_authorized": False,
                          "contract": prepared["contract"], "required_seal": prepared["required_seal"],
                          "source_seals_pending": prepared["source_seals_pending"]}, allow_nan=False))
        return 0
    require(args.deadline is not None, "native phase requires a declared UTC wall deadline")
    deadline = datetime.fromisoformat(args.deadline)
    require(deadline.tzinfo is not None and 0 < (deadline - datetime.now(UTC)).total_seconds() <= MAX_WALL,
            "deadline must be future, explicit and at most six hours away")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(output / "run-manifest.json", manifest(args.case, prepared["target"], prepared,
               {"kind": "bounded_stationary_III_approach_phase"}, arguments, prepared["restart"]))
    diagnostic.atomic(output / "input-seal.json", Path(prepared["input_seal"]["path"]).read_bytes())
    write_json(output / "preparation.json", {"contract": prepared["contract"], "inputs": prepared["records"],
               "implementation": prepared["implementation"], "deadline": deadline.isoformat(),
               "native_execution_authorized": True, "hard_wall_cap_required": True})
    runner.append_progress(output, {"event": "prepared"}, first=True)
    result = run_stages(prepared, output, arguments, PhaseBudget(output), deadline)
    write_json(output / "completion.json", {**result, "files": iii.inventory(output)})
    runner.append_progress(output, {"event": "final", "status": result["status"], "work": result["work"]})
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
