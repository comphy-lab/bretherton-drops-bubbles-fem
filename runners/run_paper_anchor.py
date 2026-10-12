"""Continue one qualified paper-model state to the declared stationary III anchor.

Parent restoration uses its original case and the unchanged restart validator.
Only case identity and imposed Ca may change along an explicit bounded ladder.
Every stage retains its case, provenance and portable checkpoint; target case
qualification follows a terminal remesh/polish and an independent fresh load.
No eigenproblem, transient or automatic continuation ladder is executed.
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
from itertools import pairwise
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems import paper_bubble as model
from runners import run_paper_case as runner
from runners import run_pressure_diagnostic as diagnostic
from runners import run_pressure_remesh as remesh

PARENT_SOURCE = "2b985bddd203df127826f55329e0bd7efa617580"
PARENT_CASE_SHA = "d264591bdaf7ba1864e0c4dd64655ca6fbeeb87029f8cc1af6b03df710401b14"
TARGET_CA = -0.00315
MAX_CA_STEPS = 8
GATES = {"stationary", "lower_static_pressure", "flux", "volume", "volume_integrals"}
require, sha, write_json = diagnostic.require, diagnostic.sha, diagnostic.write_json


def transition_contract(parent, target):
    """Use a canonical deep diff; changes to any other case member are refused."""
    parent, target = model.validate_case(parent), model.validate_case(target)
    changes = remesh.differing_keys(parent, target)
    require(
        set(changes) == {"physics.ca_in", "case_id"},
        "case transition may change only Ca and case_id",
    )
    return {
        "schema": "herrada-paper-ca-transition-v1",
        "changed_fields": changes,
        "parent_ca": parent["physics"]["ca_in"],
        "target_ca": target["physics"]["ca_in"],
        "parent_case_id": parent["case_id"],
        "target_case_id": target["case_id"],
    }


def read_ladder(path, parent, target):
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
            not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
            for v in values
        ),
        "Ca ladder values must be finite numbers",
    )
    require(
        values[-1] == target["physics"]["ca_in"], "Ca ladder does not end at the declared target"
    )
    route = [parent["physics"]["ca_in"], *values]
    require(
        all(a > b for a, b in pairwise(route)),
        "Ca ladder must decrease strictly from its parent",
    )
    return values


def preflight(args):
    """Bind the qualified parent, target, ladder and unchanged implementation."""
    require(args.source.resolve(strict=True) == ROOT, "runner must use its own isolated component")
    for module, relative in (
        (model, "problems/paper_bubble.py"),
        (runner, "runners/run_paper_case.py"),
        (diagnostic, "runners/run_pressure_diagnostic.py"),
        (remesh, "runners/run_pressure_remesh.py"),
    ):
        require(
            Path(module.__file__).resolve() == ROOT / relative,
            "helper imported from another component",
        )
    parent_path = args.manifest.resolve(strict=True)
    parent_manifest = json.loads(parent_path.read_text())
    require(
        parent_manifest["component_commit"]
        == parent_manifest["component"]["commit"]
        == PARENT_SOURCE,
        "unexpected qualified parent source",
    )
    component = runner.component_revision(ROOT)
    runner.require_clean_source(component)
    runner.require_clean_source(parent_manifest["component"])
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", PARENT_SOURCE, component["commit"])
    parent_sources = parent_manifest["component"]["sources"]
    require(set(parent_sources) == set(component["sources"]), "parent method inventory differs")
    for relative, old in parent_sources.items():
        expected = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{PARENT_SOURCE}:{relative}")
        ).hexdigest()
        require(
            old["committed_matches"] is True
            and expected
            == old["sha256"]
            == old["committed_sha256"]
            == component["sources"][relative]["sha256"]
            == sha(ROOT / relative),
            f"parent method changed: {relative}",
        )
    implementation = {}
    for relative in (
        "runners/run_paper_anchor.py",
        "runners/run_pressure_remesh.py",
        "runners/run_pressure_diagnostic.py",
    ):
        path = ROOT / relative
        digest = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{component['commit']}:{relative}")
        ).hexdigest()
        require(
            path.is_file() and not path.is_symlink() and sha(path) == digest,
            "implementation differs from Git",
        )
        implementation[relative] = {"path": str(path), "sha256": digest}
        if relative != "runners/run_paper_anchor.py":
            require(
                digest == parent_manifest["implementation"][relative]["sha256"],
                "reviewed remesh helper changed",
            )
    environment = runner.pyoomph_revision()
    runner.require_pinned_environment(environment)
    runner.require_pinned_environment(parent_manifest["pyoomph"])
    require(
        environment["commit"] == diagnostic.PYOOMPH
        and environment["native_core"]["sha256"]
        == parent_manifest["pyoomph"]["native_core"]["sha256"]
        == diagnostic.CORE,
        "native core or source pin differs",
    )
    require(
        environment["interpreter"]["resolved_executable"]
        == parent_manifest["pyoomph"]["interpreter"]["resolved_executable"],
        "interpreter differs",
    )
    parent_case_path = Path(parent_manifest["case_path"]).resolve(strict=True)
    require(sha(parent_case_path) == parent_manifest["case_sha256"], "parent case bytes differ")
    require(
        sha(parent_case_path) == PARENT_CASE_SHA,
        "parent construction differs from the declared medium case",
    )
    parent, target = runner.read_case(parent_case_path), runner.read_case(args.case)
    contract = transition_contract(parent, target)
    require(
        parent["physics"]["ca_in"] == 0.02
        and target["physics"]["ca_in"] == TARGET_CA
        and parent["physics"]["bond"] == 1.56
        and parent["physics"]["problem"] == "finite_gas_bubble",
        "this phase requires the declared III anchor",
    )
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
        and Path(restart["state"]["path"]) == args.state.resolve(strict=True),
        "parent packet binds another run/state",
    )
    require(
        restart["stage"]["kind"] == "completed_polish"
        and restart["stage"]["bond"] == parent["physics"]["bond"],
        "qualified terminal parent required",
    )
    runtime = parent_path.parent
    completion = json.loads((runtime / "completion.json").read_text())
    require(
        completion["status"] == "complete" and completion["numerical_acceptance"] is True,
        "parent lacks numerical acceptance",
    )
    records = [
        restart["packet"],
        restart["parent_manifest"],
        restart["state"],
        *restart["meshes"],
        {"path": str(parent_case_path), "sha256": sha(parent_case_path)},
        {"path": str(args.case.resolve()), "sha256": sha(args.case)},
        {"path": str(args.ca_ladder.resolve()), "sha256": sha(args.ca_ladder)},
        {"path": str(runtime / "completion.json"), "sha256": sha(runtime / "completion.json")},
    ]
    require(
        {"run-manifest.json", "summary.json", "cold-load.json"} <= set(completion["files"]),
        "parent qualification files unbound",
    )
    for relative, digest in completion["files"].items():
        path = runtime / relative
        require(
            not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and not path.is_symlink()
            and sha(path) == digest,
            "parent completion artifact differs",
        )
        records.append({"path": str(path.resolve()), "sha256": digest})
    summary = json.loads((runtime / "summary.json").read_text())
    cold = json.loads((runtime / "cold-load.json").read_text())
    require(
        summary["completion_reached"] is True
        and summary["cold_load_verified"] is True
        and set(summary["checks"]) == GATES
        and set(summary["cold_load_checks"]) == GATES
        and all(v is True for v in summary["checks"].values())
        and all(v is True for v in summary["cold_load_checks"].values())
        and cold["verified"] is True,
        "parent scientific or cold-load gates failed",
    )
    contract.update(
        parent_case={"path": str(parent_case_path), "sha256": sha(parent_case_path)},
        target_case={"path": str(args.case.resolve()), "sha256": sha(args.case)},
        ca_ladder={
            "path": str(args.ca_ladder.resolve()),
            "sha256": sha(args.ca_ladder),
            "values": values,
        },
        maximum_ca_steps=MAX_CA_STEPS,
        total_newton_update_bound=160 * (len(values) + 1),
        per_polish_remesh_bound=3,
        explicit_terminal_remesh_calls=1,
    )
    return {
        "parent": parent,
        "target": target,
        "restart": restart,
        "environment": environment,
        "component": component,
        "implementation": implementation,
        "records": records,
        "contract": contract,
        "values": values,
        "parent_cold": cold,
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
        "case_fields_scope": "all sections validated and bound; parent construction, explicit Ca transitions and stationary gates; no transient or Bond traversal",
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
        "qualification_scope": "stationary Ca=-0.00315 candidate for Branch III; no branch-identity, spectrum or full-comparison claim",
    }


def stable_inputs(prepared):
    remesh.verify_inputs(prepared["records"])
    remesh.verify_source(prepared["component"], prepared["implementation"])


def restore_checked(problem, case, restart, expected_pins, expected_history):
    runner.verify_restart_artifacts(restart)
    problem.load_state(
        restart["state"]["path"],
        quiet=True,
        additional_info={"exchange_msh_file": restart["mesh_exchange"]},
    )
    loaded = model._stationary_state_signature(problem, True)
    pins = diagnostic.actual_pins(problem)
    remesh.check_loaded_state(loaded, restart, case, pins, expected_pins)
    require(
        loaded["serialized_state"]["payload_sha256"] == expected_history,
        "cold-loaded complete history differs",
    )
    preparation = model._prepare_stationary_bc_state(problem, True)
    require(pins == diagnostic.actual_pins(problem), "cold BC preparation changed actual pins")
    return {"verified": True, "loaded": loaded, "actual_pins": pins, "bc_preparation": preparation}


def stage_case(target, value, index, final):
    case = copy.deepcopy(target)
    case["physics"]["ca_in"] = float(value)
    if not final:
        case["case_id"] = f"{target['case_id']}_ca_{index:03d}"
    return model.validate_case(case)


def adopt_case(problem, previous, current, output):
    """Record the actual parameter change and matching declared case adoption."""
    contract = transition_contract(previous, current)
    require(problem.case == previous, "live prior case differs")
    before = model._stationary_state_signature(problem, True)
    pins = diagnostic.actual_pins(problem)
    require(
        before["parameters"]
        == {"Ca_in": previous["physics"]["ca_in"], "Bo": previous["physics"]["bond"]},
        "live prior forcing differs",
    )
    write_json(
        output / "transition-prepared.json",
        {"contract": contract, "before": before, "actual_pins": pins},
    )
    problem.get_global_parameter("Ca_in").value = float(current["physics"]["ca_in"])
    problem.case = copy.deepcopy(current)
    after = model._stationary_state_signature(problem, True)
    after_pins = diagnostic.actual_pins(problem)
    differences = remesh.differing_keys(before, after)
    checks = {
        "forcing_matches_declared_case": after["parameters"]
        == {"Ca_in": current["physics"]["ca_in"], "Bo": current["physics"]["bond"]},
        "actual_pins_preserved": pins == after_pins,
        "current_fields_layout_and_geometry_preserved": set(differences)
        <= {"parameters.Ca_in", "serialized_state.sha256", "serialized_state.payload_sha256"},
        "live_case_matches_declared_case": problem.case == current,
    }
    receipt = {
        "verified": all(checks.values()),
        "contract": contract,
        "checks": checks,
        "before": before,
        "after": after,
        "actual_pins": {"before": pins, "after": after_pins},
        "differing_signature_keys": differences,
        "numerical_acceptance": False,
    }
    write_json(output / "transition.json", receipt)
    require(all(checks.values()), "case adoption changed undeclared forcing, fields or pins")
    return receipt


def configure(problem, output, prepared):
    problem.set_output_directory(str(output / "solver"))
    problem.set_linear_solver("superlu")
    problem.set_c_compiler("system")
    problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
    stable_inputs(prepared)
    problem.initialise()


def execute(args, prepared, output):
    import numpy as np

    parent, target = prepared["parent"], prepared["target"]
    restart, cold_parent = prepared["restart"], prepared["parent_cold"]

    def emit(event, **payload):
        runner.append_progress(output, {"event": event, "case_id": target["case_id"], **payload})

    stages = []
    with model.build_problem(parent, initial_bond=parent["physics"]["bond"]) as problem:
        configure(problem, output, prepared)
        restored = restore_checked(
            problem,
            parent,
            restart,
            cold_parent["actual_pins"],
            cold_parent["loaded"]["serialized_state"]["payload_sha256"],
        )
        parent_output = output / "parent-observation"
        parent_output.mkdir()
        parent_sample = remesh.sample(problem, model, parent, parent_output, "cold-load", np)
        require(
            all(remesh.canonical_checks(parent, parent_sample["metrics"]).values()),
            "parent cold-load gates failed",
        )
        write_json(output / "parent-load.json", restored)
        emit("parent-restored", complete_fields_pins_history_verified=True)
        previous = parent
        for index, value in enumerate(prepared["values"], 1):
            stable_inputs(prepared)
            current = stage_case(target, value, index, index == len(prepared["values"]))
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
                    {"kind": "ca_continuation", "index": index, "ca_in": value},
                ),
            )
            runner.append_progress(
                stage_output, {"event": "prepared", "case_id": current["case_id"]}, first=True
            )
            adopt_case(problem, previous, current, stage_output)
            emit(
                "ca-transition",
                index=index,
                ca_in=value,
                stage=str(stage_output.relative_to(output)),
            )
            polish = model.polish_confined_state(
                problem, current, stage_output, "ca_step", initial_relaxation=1.0
            )
            require(polish["status"] == "converged", "Ca stage polish failed")
            sample = remesh.sample(problem, model, current, stage_output, "after-polish", np)
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
            record = {
                "index": index,
                "ca_in": value,
                "case": {"path": str(case_path), "sha256": sha(case_path)},
                "polish": polish,
                "checkpoint": checkpoint,
                "resume_packet_sha256": stage_restart["packet"]["sha256"],
                "canonical_checks_observed": remesh.canonical_checks(current, sample["metrics"]),
                "qualification_scope": "stationary continuation stage; target acceptance pending",
            }
            write_json(stage_output / "stage.json", record)
            stages.append(record)
            previous = current
            stable_inputs(prepared)
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
            stable_inputs(prepared)
            if phase == "after-remesh":
                remesh.normalize_pressure_dummies(problem, model, terminal, np)
            value = remesh.sample(problem, model, target, terminal, phase, np)
            if phase != "after-polish":
                value["retained_state"] = remesh.retain_portable_sample(
                    problem, model, terminal, phase
                )
            return value

        terminal_result = remesh.remesh_and_polish(problem, model, target, terminal, observe)
        checkpoint = model._checkpoint_stationary_state(
            problem, target, output, len(model.stationary_bond_schedule(target)) - 1, True
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
        stable_inputs(prepared)
    emit("target-cold-load", status="starting")
    with model.build_problem(target, initial_bond=target["physics"]["bond"]) as cold:
        cold_output = output / "target-cold"
        cold_output.mkdir()
        configure(cold, cold_output, prepared)
        cold_receipt = restore_checked(
            cold,
            target,
            final_restart,
            final_pins,
            final_restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
        )
        cold_sample = remesh.sample(cold, model, target, cold_output, "cold-load", np)
        cold_checks = remesh.canonical_checks(target, cold_sample["metrics"])
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
    stable_inputs(prepared)
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
    actual = manifest(args.case, prepared["target"], prepared, {"kind": "stationary_III_anchor"})
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
