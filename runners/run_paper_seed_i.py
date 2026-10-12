"""Qualify one fresh negative-flow finite-gas seed along its declared Bo ladder.

The declared transient is preparatory relaxation only. Stationary ladder stages
retain their own case, manifest and portable checkpoint. Final qualification
uses the reviewed terminal remesh/polish and a fresh complete-state cold load;
no Ca continuation, eigenanalysis or branch-identity claim is made.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems import paper_bubble as model
from runners import run_paper_anchor as anchor
from runners import run_paper_case as runner
from runners import run_pressure_diagnostic as diagnostic
from runners import run_pressure_remesh as remesh

FROZEN_SOURCE = "958f7b6a142be27ba5779609ebadf1bdc0316e6d"
BASE_CASE = "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json"
TARGET_CA = -0.02
require, sha, write_json = diagnostic.require, diagnostic.sha, diagnostic.write_json


def validate_case(case, baseline):
    case, baseline = model.validate_case(case), model.validate_case(baseline)
    require(
        set(remesh.differing_keys(baseline, case)) == {"case_id", "physics.ca_in"},
        "fresh seed may change only case_id and Ca",
    )
    require(case["physics"]["ca_in"] == TARGET_CA, "fresh seed requires Ca=-0.02")
    return case


def validate_schedule(case, schedule):
    require(
        schedule == model.stationary_bond_schedule(case),
        "Bo order differs from the declared schedule",
    )
    require(
        len(schedule) <= case["solve"]["max_bond_steps"] + 1
        and schedule[0] == case["solve"]["initial_bond"]
        and schedule[-1] == case["physics"]["bond"],
        "Bo schedule exceeds its declared bounds",
    )
    return schedule


def validate_environment(environment):
    runner.require_pinned_environment(environment)
    require(
        environment["commit"] == diagnostic.PYOOMPH
        and environment["native_core"]["sha256"] == diagnostic.CORE,
        "native source/core differs",
    )


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "runner must use its own isolated source")
    for module, relative in (
        (model, "problems/paper_bubble.py"),
        (runner, "runners/run_paper_case.py"),
        (remesh, "runners/run_pressure_remesh.py"),
        (anchor, "runners/run_paper_anchor.py"),
        (diagnostic, "runners/run_pressure_diagnostic.py"),
    ):
        require(
            Path(module.__file__).resolve() == ROOT / relative,
            "helper imported from another component",
        )
    component = runner.component_revision(ROOT)
    runner.require_clean_source(component)
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", FROZEN_SOURCE, component["commit"])
    implementation = {}
    paths = set(component["sources"]) | {
        BASE_CASE,
        "runners/run_paper_anchor.py",
        "runners/run_pressure_remesh.py",
        "runners/run_pressure_diagnostic.py",
        "runners/run_paper_seed_i.py",
    }
    for relative in sorted(paths):
        path = ROOT / relative
        committed = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{component['commit']}:{relative}")
        ).hexdigest()
        require(
            path.is_file() and not path.is_symlink() and sha(path) == committed,
            "implementation differs from commit",
        )
        if relative != "runners/run_paper_seed_i.py":
            frozen = hashlib.sha256(
                diagnostic.git(ROOT, "show", f"{FROZEN_SOURCE}:{relative}")
            ).hexdigest()
            require(committed == frozen, f"frozen method changed: {relative}")
        implementation[relative] = {"path": str(path), "sha256": committed}
    environment = runner.pyoomph_revision()
    validate_environment(environment)
    case = validate_case(runner.read_case(args.case), runner.read_case(ROOT / BASE_CASE))
    schedule = validate_schedule(case, model.stationary_bond_schedule(case))
    records = [{"path": str(args.case.resolve()), "sha256": sha(args.case)}]
    return {
        "case": case,
        "schedule": schedule,
        "component": component,
        "implementation": implementation,
        "environment": environment,
        "records": records,
        "bounds": {
            "preparation_endtime": case["solve"]["relax_time"],
            "preparation_startstep": case["solve"]["start_step"],
            "preparation_maxstep": case["solve"]["relax_time"] / 8,
            "preparation_transient_newton_cap": 10,
            "maximum_bo_updates": case["solve"]["max_bond_steps"],
            "actual_bo_updates": len(schedule) - 1,
            "per_mesh_stage_newton_cap": min(40, case["solve"]["max_newton_iterations"]),
            "per_polish_total_newton_cap": 160,
            "per_polish_remesh_cap": 3,
            "stationary_total_newton_cap": 160 * (len(schedule) + 1),
            "explicit_terminal_remesh_calls": 1,
        },
    }


def stable_inputs(prepared):
    remesh.verify_inputs(prepared["records"])
    remesh.verify_source(prepared["component"], prepared["implementation"])


def stage_case(case, index, schedule):
    stage = copy.deepcopy(case)
    stage["physics"]["bond"] = schedule[index]
    if index != len(schedule) - 1:
        stage["case_id"] = f"{case['case_id']}_bo_{index:03d}"
    stage = model.validate_case(stage)
    require(
        model.stationary_bond_schedule(stage) == schedule[: index + 1], "stage Bo schedule differs"
    )
    return stage


def manifest(case_path, case, prepared, stage):
    script = Path(__file__).resolve()
    record = {"path": str(script), "sha256": sha(script)}
    return {
        "schema": "pyoomph-runner-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "case_id": case["case_id"],
        "case_path": str(case_path.resolve()),
        "case_file": str(case_path.resolve()),
        "case_sha256": sha(case_path),
        "case_fields_used": list(runner.CASE_FIELDS_USED),
        "case_fields_scope": "all fields validated/bound; fresh construction, declared preparatory transient and Bo ladder, stationary qualification",
        "units": "nondimensional (sigma=mu_liquid=R0=1)",
        "nondimensionalization": case["nondimensionalization"],
        "argv": [sys.executable, str(script), *sys.argv[1:]],
        "runner": record,
        "runner_path": record["path"],
        "runner_sha256": record["sha256"],
        "component": prepared["component"],
        "component_commit": prepared["component"]["commit"],
        "pyoomph": prepared["environment"],
        "azimuthal_stability": True,
        "implementation": prepared["implementation"],
        "fresh_seed_stage": stage,
        "target_case": prepared["records"][0],
        "declared_bo_schedule": prepared["schedule"],
        "runtime_bounds": prepared["bounds"],
        "parent_state_used": False,
        "qualification_scope": "one stationary negative-flow seed; no branch identity, eigenanalysis or transient validation",
    }


def check_forcing(problem, case):
    parameters = {name: float(problem.get_global_parameter(name).value) for name in ("Bo", "Ca_in")}
    require(
        problem.case == case and parameters == {"Bo": case["physics"]["bond"], "Ca_in": TARGET_CA},
        "live seed forcing or declared stage case differs",
    )
    return parameters


def adopt_bond(problem, previous, current, output):
    require(
        set(remesh.differing_keys(previous, current)) == {"case_id", "physics.bond"},
        "Bo transition changed undeclared case fields",
    )
    check_forcing(problem, previous)
    before, pins = model._stationary_state_signature(problem, True), diagnostic.actual_pins(problem)
    write_json(
        output / "transition-prepared.json",
        {"before": before, "actual_pins": pins, "from_case": previous, "to_case": current},
    )
    problem.get_global_parameter("Bo").value = float(current["physics"]["bond"])
    problem.case = copy.deepcopy(current)
    after, after_pins = (
        model._stationary_state_signature(problem, True),
        diagnostic.actual_pins(problem),
    )
    differences = remesh.differing_keys(before, after)
    checks = {
        "actual_pins_preserved": pins == after_pins,
        "current_fields_layout_and_geometry_preserved": set(differences)
        <= {"parameters.Bo", "serialized_state.sha256", "serialized_state.payload_sha256"},
        "forcing_matches_declared_case": after["parameters"]
        == {"Bo": current["physics"]["bond"], "Ca_in": TARGET_CA},
    }
    receipt = {
        "verified": all(checks.values()),
        "checks": checks,
        "before": before,
        "after": after,
        "actual_pins": {"before": pins, "after": after_pins},
        "differing_signature_keys": differences,
        "numerical_acceptance": False,
    }
    write_json(output / "transition.json", receipt)
    require(all(checks.values()), "Bo transition changed undeclared forcing, fields or pins")
    return receipt


def preparatory_relaxation(problem, case, output):
    check_forcing(problem, case)
    solve = case["solve"]
    before = remesh.retain_portable_sample(problem, model, output, "before-relaxation")
    require(before["signature"]["time"] == 0.0, "fresh preparation must start at time zero")
    require(problem.max_newton_iterations == 10, "pinned preparatory Newton default differs")
    arguments = {
        "startstep": solve["start_step"],
        "outstep": case["output"]["cadence"],
        "maxstep": solve["relax_time"] / 8,
        "temporal_error": case["tolerances"]["temporal_error"],
    }
    write_json(
        output / "relaxation-prepared.json",
        {
            "status": "prepared",
            "endtime": solve["relax_time"],
            "arguments": arguments,
            "before_state": before,
            "transient_newton_cap": problem.max_newton_iterations,
            "scope": "preparatory relaxation only; no Fig7 dynamics or transient validation",
        },
    )
    problem.run(solve["relax_time"], **arguments)
    check_forcing(problem, case)
    after = remesh.retain_portable_sample(problem, model, output, "after-relaxation")
    require(
        math.isclose(after["signature"]["time"], solve["relax_time"], rel_tol=0, abs_tol=1e-12),
        "declared preparatory endtime not reached",
    )
    receipt = {
        "status": "completed",
        "endtime": solve["relax_time"],
        "arguments": arguments,
        "before_state": before,
        "after_state": after,
        "numerical_acceptance": False,
        "scope": "preparatory relaxation only; stationary and target gates remain pending",
    }
    write_json(output / "relaxation.json", receipt)
    return receipt


def stationary_stage(problem, case, output, index, prepared, np):
    check_forcing(problem, case)
    polish = model.polish_confined_state(
        problem,
        case,
        output,
        "seed" if index == 0 else "bond_step",
        initial_relaxation=0.5 if index == 0 else 1.0,
    )
    require(polish["status"] == "converged", "stationary Bo stage did not converge")
    observed = remesh.sample(problem, model, case, output, "after-polish", np)
    checkpoint = model._checkpoint_stationary_state(problem, case, output, index, True)
    packet = runner.validate_restart(
        output / checkpoint["resume_packet"],
        case,
        sha(output / "case.json"),
        component=prepared["component"],
        environment=prepared["environment"],
        azimuthal_stability=True,
    )
    record = {
        "index": index,
        "bond": case["physics"]["bond"],
        "polish": polish,
        "checkpoint": checkpoint,
        "packet_sha256": packet["packet"]["sha256"],
        "canonical_checks_observed": remesh.canonical_checks(case, observed["metrics"]),
        "qualification_scope": "stationary ladder stage; final target acceptance pending",
    }
    write_json(output / "stage.json", record)
    return record


def execute(args, prepared, output):
    import numpy as np

    case, schedule = prepared["case"], prepared["schedule"]
    stages, cases, paths = [], [], []
    for index in range(len(schedule)):
        current = stage_case(case, index, schedule)
        stage_output = output / "stages" / f"bond_{index:03d}"
        (stage_output / "solver").mkdir(parents=True)
        path = stage_output / "case.json"
        write_json(path, current)
        write_json(
            stage_output / "run-manifest.json",
            manifest(
                path,
                current,
                prepared,
                {"kind": "stationary_bo_stage", "index": index, "bond": schedule[index]},
            ),
        )
        runner.append_progress(
            stage_output, {"event": "prepared", "case_id": current["case_id"]}, first=True
        )
        cases.append(current)
        paths.append(stage_output)
    runner.append_progress(
        output, {"event": "native-start", "case_id": case["case_id"], "initial_bond": schedule[0]}
    )
    with model.build_problem(cases[0], initial_bond=schedule[0]) as problem:
        anchor.configure(problem, output, prepared)
        runner.append_progress(
            output,
            {
                "event": "preparatory-relaxation-starting",
                "endtime": case["solve"]["relax_time"],
                "bond": schedule[0],
                "transient_validation_claimed": False,
            },
        )
        preparatory_relaxation(problem, cases[0], paths[0])
        runner.append_progress(
            output, {"event": "preparatory-relaxation-completed", "numerical_acceptance": False}
        )
        for index, current in enumerate(cases):
            stable_inputs(prepared)
            if index:
                adopt_bond(problem, cases[index - 1], current, paths[index])
            stages.append(stationary_stage(problem, current, paths[index], index, prepared, np))
            stable_inputs(prepared)
            runner.append_progress(
                output,
                {
                    "event": "stationary_bo_stage",
                    "index": index,
                    "bond": schedule[index],
                    "packet": str(
                        paths[index].relative_to(output) / stages[-1]["checkpoint"]["resume_packet"]
                    ),
                },
            )
        check_forcing(problem, case)
        terminal = output / "terminal"
        (terminal / "solver").mkdir(parents=True)
        write_json(
            terminal / "run-manifest.json",
            manifest(args.case, case, prepared, {"kind": "terminal_remesh"}),
        )
        runner.append_progress(
            terminal, {"event": "prepared", "case_id": case["case_id"]}, first=True
        )

        def observe(phase):
            stable_inputs(prepared)
            if phase == "after-remesh":
                remesh.normalize_pressure_dummies(problem, model, terminal, np)
            observed = remesh.sample(problem, model, case, terminal, phase, np)
            if phase != "after-polish":
                observed["retained_state"] = remesh.retain_portable_sample(
                    problem, model, terminal, phase
                )
            return observed

        runner.append_progress(
            output, {"event": "terminal-remesh-starting", "bond": case["physics"]["bond"]}
        )
        terminal_result = remesh.remesh_and_polish(problem, model, case, terminal, observe)
        checkpoint = model._checkpoint_stationary_state(
            problem, case, output, len(schedule) - 1, True
        )
        final_restart = runner.validate_restart(
            output / checkpoint["resume_packet"],
            case,
            sha(args.case),
            component=prepared["component"],
            environment=prepared["environment"],
            azimuthal_stability=True,
        )
        final_pins = diagnostic.actual_pins(problem)
        require(
            final_pins == terminal_result["after_polish"]["actual_pins"],
            "final checkpoint changed actual pins",
        )
        stable_inputs(prepared)
    runner.append_progress(output, {"event": "target-cold-load-starting"})
    cold_output = output / "cold"
    cold_output.mkdir()
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as cold:
        anchor.configure(cold, cold_output, prepared)
        cold_receipt = anchor.restore_checked(
            cold,
            case,
            final_restart,
            final_pins,
            final_restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
        )
        cold_sample = remesh.sample(cold, model, case, cold_output, "cold-load", np)
        cold_checks = remesh.canonical_checks(case, cold_sample["metrics"])
        cold_receipt["canonical_checks"] = cold_checks
        write_json(output / "cold-load.json", cold_receipt)
    checks = remesh.canonical_checks(case, terminal_result["after_polish"]["metrics"])
    accepted = all(checks.values()) and all(cold_checks.values())
    result = {
        "schema": "private-fresh-negative-paper-seed-v1",
        "stages": stages,
        "terminal": terminal_result,
        "checks": checks,
        "cold_load_checks": cold_checks,
        "cold_load_verified": True,
        "causal_predictions": remesh.causal_predictions(terminal_result),
        "checkpoint": checkpoint,
        "numerical_acceptance": accepted,
        "completion_reached": accepted,
        "branch_identity_verified": False,
        "eigenanalysis_performed": False,
        "transient_validation_claimed": False,
    }
    stable_inputs(prepared)
    write_json(output / "result.json", result)
    write_json(
        output / "summary.json",
        {
            "status": "completed" if accepted else "failed",
            "scientific_status": "checks_passed" if accepted else "checks_failed",
            "completion_reached": accepted,
            "checks": checks,
            "cold_load_checks": cold_checks,
            "cold_load_verified": True,
            **terminal_result["after_polish"]["metrics"],
            "ca_in": TARGET_CA,
            "bond": case["physics"]["bond"],
            "checkpoint": checkpoint["resume_packet"],
            "branch_identity_verified": False,
            "transient_validation_claimed": False,
            "eigenanalysis_performed": False,
        },
    )
    require(accepted, "fresh seed target or cold-load gates failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="metadata preflight; no native import/output"
    )
    args = parser.parse_args(argv)
    os.umask(0o077)
    prepared = preflight(args)
    require(not args.out.is_symlink(), "output must not be a symlink")
    output = runner.validate_output_path(args.out, ROOT)
    if args.check:
        print(
            json.dumps(
                {
                    "verified": True,
                    "native_imported": False,
                    "bounds": prepared["bounds"],
                    "bo_schedule": prepared["schedule"],
                },
                allow_nan=False,
            )
        )
        return 0
    actual = manifest(args.case, prepared["case"], prepared, {"kind": "fresh_negative_seed"})
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
            "case_id": prepared["case"]["case_id"],
            "runner": actual["runner"],
            "component_commit": prepared["component"]["commit"],
        },
        first=True,
    )
    try:
        write_json(
            output / "preparation.json",
            {
                "inputs": prepared["records"],
                "implementation": prepared["implementation"],
                "bounds": prepared["bounds"],
                "bo_schedule": prepared["schedule"],
                "numerical_acceptance": False,
            },
        )
        result = execute(args, prepared, output)
        stable_inputs(prepared)
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
            "inputs_unchanged": diagnostic.parents_unchanged(prepared["records"]),
            "numerical_acceptance": False,
        }
        write_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", **failure})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
