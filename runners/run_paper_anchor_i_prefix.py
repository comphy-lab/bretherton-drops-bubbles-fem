"""Continue an independently qualified prefix under its remaining original budgets.

Only the original unfinished Ca suffix is traversed. Terminal qualification and
all five original predicates are retained; no branch or spectrum is certified.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_prefix_qualify as prefix

model, runner, diagnostic, remesh = prefix.model, prefix.runner, prefix.diagnostic, prefix.remesh
anchor, cold, requalification = prefix.anchor, prefix.cold, prefix.requalification
require, sha, write_json = prefix.require, prefix.sha, prefix.write_json
ENTRYPOINT = prefix.CONTINUATION
SCHEMA = "stationary-I-anchor-from-prefix-v1"
SCOPE = (
    "stationary I candidate from qualified prefix; no branch identity, spectrum or full comparison"
)


def inventory_records(runtime, files):
    require(isinstance(files, dict), "completion file inventory required")
    require(prefix.completion_inventory(runtime) == files, "prefix completion inventory differs")
    return [prefix.regular_record(runtime / prefix.relative_path(name)) for name in sorted(files)]


def validate_typed_receipt(receipt, restart, case, pins, expected, runtime):
    require(
        receipt.get("verified") is True
        and receipt.get("bc_preparation", {}).get("verified") is True
        and receipt.get("intervention", {}).get("verified") is True
        and receipt["intervention"].get("kind") == "typed-native-endtime-restoration"
        and receipt["intervention"].get("api") == cold.API
        and receipt["intervention"].get("saved_endtime") == case["solve"]["relax_time"] == 8.0
        and receipt["loaded"]["serialized_state"]["payload_sha256"] == expected
        and receipt["actual_pins"] == pins,
        "prefix typed restoration, BC or full history differs",
    )
    remesh.check_loaded_state(receipt["loaded"], restart, case, receipt["actual_pins"], pins)
    metadata = cold.serialized_metadata(Path(restart["state"]["path"]).read_bytes())
    require(
        receipt["time_history"]
        == {key: metadata[key] for key in ("time", "dts", "unsteady_steps", "suggested_next_dt")},
        "prefix live time history differs",
    )
    raw = {}
    bases = {}
    for phase in ("before", "after"):
        record = prefix.bound_record(receipt["intervention"][phase])
        path = Path(record["path"])
        require(
            path.is_relative_to(runtime) and "solver" not in path.relative_to(runtime).parts,
            "raw typed proof outside bound prefix custody",
        )
        retention = runner._read_json(path.parent / (phase + "-retention.json"))
        require(
            retention["state"] == {**record, "format": "memory_snapshot"},
            "typed snapshot retention differs",
        )
        base = Path(retention["mesh_reference_base"])
        for mesh in retention["meshes"]:
            prefix.bound_record(mesh)
            require(
                Path(mesh["path"]).is_relative_to(runtime)
                and sha(base / mesh["stored_path"]) == mesh["sha256"]
                and retention["mesh_exchange"][mesh["stored_path"]] == mesh["path"],
                "typed snapshot retained mesh differs",
            )
        raw[phase], bases[phase] = path.read_bytes(), base
    require(bases["before"] == bases["after"], "typed snapshot reference base changed")
    saved = Path(restart["state"]["path"]).read_bytes()
    if receipt["intervention"].get("applied") is True:
        proof = cold.verify_snapshots(
            saved,
            raw["before"],
            raw["after"],
            Path(restart["mesh_reference_base"]),
            bases["after"],
            expected,
        )
        require(
            proof["only_changed_slot"] == receipt["intervention"]["only_changed_slot"],
            "typed intervention slot differs",
        )
    else:
        require(
            receipt["intervention"].get("applied") is False
            and raw["before"] == raw["after"]
            and cold.normalized_bytes(raw["after"], bases["after"])
            == cold.normalized_bytes(saved, Path(restart["mesh_reference_base"])),
            "equal-writer typed restoration changed complete bytes",
        )


def qualified_prefix(args):
    manifest_path = args.manifest.resolve(strict=True)
    runtime = manifest_path.parent
    require(not (runtime / "failure.json").exists(), "prefix qualification failed")
    producer = runner._read_json(manifest_path)
    require(
        producer.get("qualification_schema") == prefix.SCHEMA
        and Path(producer["runner"]["path"]).name == "run_paper_prefix_qualify.py"
        and producer.get("intervention")
        == {
            "kind": "stationary-continuation-prefix-qualification",
            "solve_calls": 0,
            "remesh_calls": 0,
            "time_evolution": False,
            "api": cold.API,
            "original_acceptance_changed": False,
        },
        "qualified-prefix producer/schema differs",
    )
    lineage = producer["lineage"]
    selected = lineage["completed_prefix"][-1]
    original_args = SimpleNamespace(
        source=ROOT,
        case=Path(selected["case"]["path"]),
        original_attempt_terminal=Path(lineage["original_attempt"]["terminal"]["path"]),
        original_manifest=Path(lineage["original_attempt"]["manifest"]["path"]),
        stage=Path(selected["stage"]["path"]),
        manifest=Path(selected["manifest"]["path"]),
        packet=Path(selected["packet"]["path"]),
        state=Path(selected["state"]["path"]),
        original_ca_ladder=Path(lineage["remaining_ladder"]["original"]["path"]),
        custody=Path(lineage["original_attempt"]["custody"]["path"]),
        budget=Path(lineage["budget"]["witness"]["path"]),
    )
    admitted = prefix.preflight(original_args)
    require(lineage == admitted["lineage"], "qualified-prefix lineage differs from direct evidence")
    component, environment, implementation = prefix.source_contract(
        runner._read_json(original_args.original_manifest), ENTRYPOINT
    )
    runner.require_clean_source(producer["component"])
    require(
        producer["component_commit"] == producer["component"]["commit"],
        "prefix producer commit differs",
    )
    diagnostic.git(
        ROOT, "merge-base", "--is-ancestor", producer["component_commit"], component["commit"]
    )
    for relative, declaration in producer["implementation"].items():
        require(
            relative in implementation
            and declaration["sha256"] == implementation[relative]["sha256"],
            "qualified-prefix helper/producer source differs",
        )
    require(
        producer["runner_sha256"]
        == producer["runner"]["sha256"]
        == implementation[prefix.ENTRYPOINT]["sha256"]
        and producer["runner_path"] == producer["runner"]["path"]
        and producer["case_sha256"] == selected["case"]["sha256"]
        and producer["case_path"] == producer["case_file"] == selected["case"]["path"]
        and producer["original_acceptance_changed"] is False
        and producer["azimuthal_stability"] is True
        and producer["pyoomph"] == environment,
        "qualified-prefix producer bindings differ",
    )
    completion_path = runtime / "completion.json"
    completion = runner._read_json(completion_path)
    require(
        set(completion)
        == {
            "schema",
            "status",
            "numerical_acceptance",
            "stationary_prefix_accepted",
            "anchor_accepted",
            "original_acceptance_changed",
            "files",
        }
        and completion.get("schema") == prefix.SCHEMA
        and completion.get("status") == "complete"
        and completion.get("numerical_acceptance") is True
        and completion.get("stationary_prefix_accepted") is True
        and completion.get("anchor_accepted") is False
        and completion.get("original_acceptance_changed") is False,
        "prefix-only completion required",
    )
    files = completion["files"]
    require(
        {
            "run-manifest.json",
            "preparation.json",
            "schema.json",
            "result.json",
            "summary.json",
            "cold-load.json",
            "parent-cold-load.json",
            "first-zero-solve-guard.json",
            "second-zero-solve-guard.json",
        }
        <= set(files),
        "prefix qualification proofs unbound",
    )
    records = inventory_records(runtime, files) + [prefix.regular_record(completion_path)]
    preparation = runner._read_json(runtime / "preparation.json")
    require(
        preparation["lineage"] == lineage
        and preparation["inputs"] == admitted["records"]
        and preparation["implementation"] == producer["implementation"]
        and preparation["numerical_acceptance"] is False,
        "prefix qualification admission differs",
    )
    schema = runner._read_json(runtime / "schema.json")
    require(
        schema
        == {
            "schema": prefix.SCHEMA,
            "scope": producer["qualification_scope"],
            "consumer": lineage["consumer"],
        },
        "prefix schema contract differs",
    )
    result = runner._read_json(runtime / "result.json")
    summary = runner._read_json(runtime / "summary.json")
    require(
        set(result) == prefix.RESULT_FIELDS
        and set(summary)
        == prefix.RESULT_FIELDS | {"status", "completion_reached", "scientific_status"}
        and result["schema"] == prefix.SCHEMA
        and result["numerical_acceptance"] is True
        and result["stationary_prefix_accepted"] is True
        and result["anchor_accepted"] is False
        and result["cold_load_verified"] is True
        and result["original_attempt_status"] == "failed"
        and result["original_acceptance_changed"] is False
        and result["lineage"] == lineage
        and all(result.get(key) is value for key, value in prefix.NO_CLAIMS.items())
        and all(summary.get(key) == value for key, value in result.items())
        and summary["status"] == "completed"
        and summary["completion_reached"] is True
        and set(result["checks"]) == set(result["cold_load_checks"]) == prefix.GATES
        and all(
            value is True
            for value in (*result["checks"].values(), *result["cold_load_checks"].values())
        ),
        "prefix stationary-only acceptance differs",
    )
    case = admitted["case"]
    restart = runner.validate_restart(
        args.packet,
        case,
        selected["case"]["sha256"],
        component=component,
        environment=environment,
        azimuthal_stability=True,
    )
    require(
        restart["parent_manifest"] == prefix.regular_record(manifest_path)
        and restart["state"]["path"] == str(args.state.resolve(strict=True))
        and restart["stage"] == admitted["restart"]["stage"]
        and (runtime / prefix.relative_path(result["checkpoint"]["resume_packet"])).resolve(
            strict=True
        )
        == Path(restart["packet"]["path"])
        and cold.normalized_bytes(
            Path(restart["state"]["path"]).read_bytes(), Path(restart["mesh_reference_base"])
        )
        == cold.normalized_bytes(
            Path(admitted["restart"]["state"]["path"]).read_bytes(),
            Path(admitted["restart"]["mesh_reference_base"]),
        ),
        "prefix descendant state/checkpoint lineage differs",
    )
    for record in (restart["packet"], restart["state"], *restart["meshes"]):
        path = Path(record["path"])
        require(
            path.is_relative_to(runtime)
            and files.get(str(path.relative_to(runtime))) == record["sha256"],
            "prefix descendant artifact unbound",
        )
    expected = admitted["restart"]["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
    pins = admitted["sample"]["actual_pins"]
    first = runner._read_json(runtime / "parent-cold-load.json")
    second = runner._read_json(runtime / "cold-load.json")
    validate_typed_receipt(first, admitted["restart"], case, pins, expected, runtime)
    validate_typed_receipt(second, restart, case, pins, expected, runtime)
    require(
        second["checks"] == result["cold_load_checks"]
        and second["packet"] == restart["packet"]
        and second["parent_manifest"] == restart["parent_manifest"],
        "prefix second fresh-load binding differs",
    )
    for name in ("first-zero-solve-guard.json", "second-zero-solve-guard.json"):
        guard = runner._read_json(runtime / name)
        require(
            guard
            == {
                "installed_before_initialise": True,
                "methods": [
                    "solve",
                    "run",
                    "force_remesh",
                    "solve_eigenproblem",
                    "steady_newton_solve",
                    "unsteady_newton_solve",
                ],
                "violations": [],
                "solve_calls": 0,
                "physical_remesh_calls": 0,
                "time_evolution": False,
            },
            "prefix zero-solve guard differs",
        )
    target = runner.read_case(args.case)
    require(
        target == admitted["target"]
        and prefix.regular_record(args.case) == lineage["remaining_ladder"]["target"],
        "original physical target changed",
    )
    contract = anchor.transition_contract(case, target)
    charge = lineage["budget"]["charge"]
    u, d = charge["charged_updates"], charge["charged_remeshes"]
    contract.update(
        schema="herrada-paper-prefix-ca-transition-v1",
        parent_case=selected["case"],
        target_case=prefix.regular_record(args.case),
        qualified_prefix=prefix.regular_record(manifest_path),
        original_ladder=lineage["remaining_ladder"]["original"],
        values=[-0.002, -0.0016],
        original_indices=[4, 5],
        carried={"updates": u, "remeshes": d},
        per_mesh_stage_newton_cap=40,
        per_mesh_stage_newton_cap_first_resumed=40 - u,
        per_polish_total_newton_cap=160,
        per_polish_total_newton_cap_first_resumed=160 - u,
        per_polish_remesh_bound=3,
        per_polish_remesh_bound_first_resumed=3 - d,
        total_newton_update_bound=480 - u,
        explicit_terminal_remesh_calls=1,
        closed_stage_allowance_transferable=False,
    )
    return {
        **admitted,
        "case": target,
        "parent": case,
        "target": target,
        "restart": restart,
        "parent_cold": second,
        "contract": contract,
        "prefix_runtime": str(runtime),
        "records": prefix.unique_records(
            [*admitted["records"], *records, prefix.regular_record(args.case)]
        ),
        "component": component,
        "environment": environment,
        "implementation": implementation,
    }


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "runner must use its isolated component")
    require(
        Path(prefix.__file__).resolve() == ROOT / prefix.ENTRYPOINT,
        "prefix helper imported elsewhere",
    )
    prepared = qualified_prefix(args)
    prefix.stable_inputs(prepared)
    return prepared


def deactivate_on_callback_failure(callback):
    """Make rollback notifications inert before preserving a callback's error."""

    @wraps(callback)
    def guarded(self, *args, **kwargs):
        try:
            return callback(self, *args, **kwargs)
        except BaseException:
            self.active = False
            raise

    return guarded


def carry_hook_type(hook_base):
    """Use the pinned hook base at execution; importing this module remains native-free."""

    class CarryBudget(hook_base):
        def __init__(self, problem, updates, remeshes, journal):
            super().__init__()
            require(
                type(updates) is int and 0 <= updates < 40,
                "carried first-mesh update allowance exhausted",
            )
            require(
                type(remeshes) is int and 0 <= remeshes < 3, "carried remesh allowance exhausted"
            )
            self.problem = problem
            self.updates, self.remeshes = updates, remeshes
            self.prior_ids = {id(hook) for hook in problem._hooks}
            self.journal = Path(journal)
            require(
                not self.journal.exists() and not self.journal.is_symlink(),
                "carry journal must be fresh",
            )
            self.active, self.seeded, self.remesh_pending = True, False, False
            self.counter = None
            self.close_error = None
            self.new_updates = self.new_remeshes = self.sequence = 0
            self.observed_updates, self.observed_remeshes = updates, remeshes

        def record(self, event, *, applied=None, phase="pre-inner-Budget-callback"):
            counter = self.counter
            observed = (
                None
                if counter is None
                else {
                    "updates": counter.updates,
                    "stage_updates": counter.stage_updates,
                    "remeshes": counter.remeshes,
                    "remaining_updates": counter.remaining_updates(),
                    "relaxation": counter.relaxation,
                }
            )
            record = {
                "schema": "stationary-prefix-budget-carry-event-v1",
                "sequence": self.sequence,
                "event": event,
                "observation_phase": phase,
                "observed_budget": observed,
                "carried": {"updates": self.updates, "remeshes": self.remeshes},
                "independently_applied_new_updates": self.new_updates,
                "independently_completed_new_remeshes": self.new_remeshes,
                "applied_event": applied,
                "active": self.active,
            }
            fd = os.open(
                self.journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "w") as stream:
                require(
                    stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "carry journal must be regular"
                )
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.sequence += 1

        def counters(self):
            require(self.active and self.seeded, "carry callback preceded its single seed")
            counter = self.counter
            require(
                type(counter.updates)
                is type(counter.stage_updates)
                is type(counter.remeshes)
                is int
                and self.updates <= counter.updates <= 160
                and self.remeshes <= counter.remeshes <= 3
                and 0 <= counter.stage_updates <= 40
                and counter.updates >= self.observed_updates
                and counter.remeshes >= self.observed_remeshes,
                "carried Budget counters are not monotone or bounded",
            )
            self.observed_updates, self.observed_remeshes = counter.updates, counter.remeshes
            return counter

        @deactivate_on_callback_failure
        def actions_before_newton_solve(self):
            if not self.active:
                return
            if not self.seeded:
                appended = [
                    hook
                    for hook in self.problem._hooks
                    if id(hook) not in self.prior_ids and hook is not self
                ]
                require(len(appended) == 1, "one newly appended native Budget required")
                counter = appended[0]
                kind = type(counter)
                require(
                    kind.__qualname__ == "polish_confined_state.<locals>.Budget"
                    and kind.__module__ == model.__name__
                    and kind.__init__.__code__.co_filename == str(model.__file__)
                    and counter.active is True
                    and counter.updates == counter.stage_updates == counter.remeshes == 0
                    and counter.trials == []
                    and counter.remeshing is False
                    and counter.converged is False
                    and counter.relaxation == 1.0
                    and self.problem._hooks.index(self) < self.problem._hooks.index(counter),
                    "new Budget identity, order or initial state differs",
                )
                self.counter = counter
                counter.updates = counter.stage_updates = self.updates
                counter.remeshes = self.remeshes
                self.seeded = True
                require(
                    counter.remaining_updates() == min(40 - self.updates, 160 - self.updates) > 0
                    and 3 - self.remeshes > 0,
                    "carried remaining allowance differs",
                )
                self.record("seed", phase="seed-before-inner-Budget-before-solve")
            counter = self.counters()
            require(
                counter.updates == self.updates + self.new_updates
                and counter.remeshes == self.remeshes + self.new_remeshes,
                "independent carry accounting differs before solve",
            )
            self.record("before_solve")

        @deactivate_on_callback_failure
        def actions_after_newton_step(self):
            if not self.active:
                return
            counter = self.counters()
            require(
                counter.updates == self.updates + self.new_updates, "pre-inner step count differs"
            )
            self.new_updates += 1
            self.record(
                "after_step",
                applied="one native update applied; inner Budget increment follows this callback",
            )

        @deactivate_on_callback_failure
        def actions_before_remeshing(self, active_remeshers):
            if not self.active:
                return
            counter = self.counters()
            require(
                not self.remesh_pending and counter.remeshes == self.remeshes + self.new_remeshes,
                "carry remesh sequence differs",
            )
            require(counter.remeshes < 3, "carried remesh allowance exhausted")
            self.remesh_pending = True
            self.record("before_remesh")

        @deactivate_on_callback_failure
        def actions_after_remeshing(self):
            if not self.active:
                return
            counter = self.counters()
            if self.remesh_pending and counter.active is True and counter.remeshing is True:
                require(
                    counter.remeshes == self.remeshes + self.new_remeshes,
                    "pre-inner remesh count differs",
                )
                self.new_remeshes += 1
                applied = (
                    "one physical remesh completed; inner Budget increment follows this callback"
                )
            else:
                applied = "load/rollback notification; no physical remesh counted"
            self.remesh_pending = False
            self.record("after_remesh", applied=applied)

        def close(self, *, failed):
            self.active = False
            try:
                self.record(
                    "helper_failed" if failed else "helper_returned", phase="post-helper-finally"
                )
            except BaseException as error:
                if not failed:
                    raise
                # A failed journal close must not replace the polish's primary error.
                self.close_error = error

        def receipt(self, polish):
            require(not self.active and self.seeded, "carry did not finish its single seeded slot")
            require(
                polish["updates_total"] == self.updates + self.new_updates
                and polish["remeshes_total"] == self.remeshes + self.new_remeshes
                and polish["trials"][0]["remaining_updates"] == 40 - self.updates,
                "carried polish final accounting differs",
            )
            return {
                "carried": {"updates": self.updates, "remeshes": self.remeshes},
                "new_spent": {"updates": self.new_updates, "remeshes": self.new_remeshes},
                "aggregate_spent": {
                    "updates": polish["updates_total"],
                    "remeshes": polish["remeshes_total"],
                },
                "first_solve_kwarg_limit_before_callback": 40,
                "first_bound_trial_remaining_updates": 40 - self.updates,
                "counter_seeded_once": True,
                "deactivated": True,
                "journal": prefix.regular_record(self.journal),
                "journal_counter_observations": "before inner Budget callback; applied events independently labelled",
            }

    return CarryBudget


def polish_with_carry(problem, case, directory, charge):
    from pyoomph.generic.problem import GenericProblemHooks

    carry = carry_hook_type(GenericProblemHooks)(
        problem, charge["updates"], charge["remeshes"], directory / "carry-budget.jsonl"
    )
    problem += carry
    failed = True
    try:
        polish = model.polish_confined_state(
            problem, case, directory, "ca_step", initial_relaxation=1.0
        )
        failed = False
    finally:
        carry.close(failed=failed)
    return polish, carry.receipt(polish)


def run_manifest(args, prepared, argv, *, case_path=None, case=None, stage=None):
    actual = prefix.manifest(
        args, prepared, ENTRYPOINT, argv, case_path=case_path, case=case, scope=SCOPE
    )
    actual.update(
        anchor_stage=stage or {"kind": "stationary_I_candidate_from_prefix"},
        case_transition_contract=prepared["contract"],
        construction_case=prepared["contract"]["parent_case"],
        parent_restart=prepared["restart"],
        qualified_prefix=prepared["contract"]["qualified_prefix"],
        original_acceptance_changed=False,
    )
    return actual


def execute(args, prepared, output, argv):
    import numpy as np

    parent, target, restart = prepared["parent"], prepared["target"], prepared["restart"]
    stages = []
    with model.build_problem(parent, initial_bond=parent["physics"]["bond"]) as problem:
        anchor.configure(problem, output, prepared)
        problem._seed_requalification_receipts_directory = output / "typed-loads"
        loaded = requalification.restore_checked(
            problem,
            parent,
            restart,
            prepared["parent_cold"]["actual_pins"],
            restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
        )
        observation = output / "parent-observation"
        observation.mkdir()
        parent_checks = requalification.sample_checked(
            problem, parent, observation, "cold-load", np
        )
        require(all(parent_checks.values()), "qualified prefix cold-load gates failed")
        write_json(output / "parent-load.json", loaded)
        previous = parent
        for index, value in zip(
            prepared["contract"]["original_indices"], prepared["contract"]["values"], strict=True
        ):
            prefix.stable_inputs(prepared)
            current = anchor.stage_case(target, value, index, index == 5)
            directory = output / "stages" / f"ca_{index:03d}"
            (directory / "solver").mkdir(parents=True)
            case_path = directory / "case.json"
            write_json(case_path, current)
            write_json(
                directory / "run-manifest.json",
                run_manifest(
                    args,
                    prepared,
                    argv,
                    case_path=case_path,
                    case=current,
                    stage={
                        "kind": "ascending_ca_continuation_from_prefix",
                        "index": index,
                        "ca_in": value,
                    },
                ),
            )
            runner.append_progress(
                directory, {"event": "prepared", "case_id": current["case_id"]}, first=True
            )
            transition = anchor.adopt_case(problem, previous, current, directory)
            if index == 4:
                polish, carried = polish_with_carry(
                    problem, current, directory, prepared["contract"]["carried"]
                )
            else:
                polish = model.polish_confined_state(
                    problem, current, directory, "ca_step", initial_relaxation=1.0
                )
                carried = None
            require(polish["status"] == "converged", "remaining original Ca stage failed")
            observed = remesh.sample(problem, model, current, directory, "after-polish", np)
            checkpoint = model._checkpoint_stationary_state(
                problem, current, directory, len(model.stationary_bond_schedule(current)) - 1, True
            )
            stage_restart = runner.validate_restart(
                directory / checkpoint["resume_packet"],
                current,
                sha(case_path),
                component=prepared["component"],
                environment=prepared["environment"],
                azimuthal_stability=True,
            )
            require(
                diagnostic.actual_pins(problem) == observed["actual_pins"],
                "continuation checkpoint changed pins",
            )
            row = {
                "index": index,
                "ca_in": value,
                "case": prefix.regular_record(case_path),
                "transition": transition,
                "polish": polish,
                "checkpoint": checkpoint,
                "resume_packet_sha256": stage_restart["packet"]["sha256"],
                "carry": carried,
                "canonical_checks_observed": remesh.canonical_checks(current, observed["metrics"]),
                "numerical_acceptance": False,
                "qualification_scope": "provisional remaining original stage; terminal qualification pending",
            }
            write_json(directory / "stage.json", row)
            stages.append(row)
            previous = current
            prefix.stable_inputs(prepared)
        require(problem.case == target, "terminal case differs from original target")
        terminal = output / "terminal"
        (terminal / "solver").mkdir(parents=True)
        write_json(
            terminal / "run-manifest.json",
            run_manifest(args, prepared, argv, stage={"kind": "terminal_remesh_from_prefix"}),
        )
        runner.append_progress(
            terminal, {"event": "prepared", "case_id": target["case_id"]}, first=True
        )

        def observe(phase):
            prefix.stable_inputs(prepared)
            if phase == "after-remesh":
                remesh.normalize_pressure_dummies(problem, model, terminal, np)
            sample = remesh.sample(problem, model, target, terminal, phase, np)
            if phase != "after-polish":
                sample["retained_state"] = remesh.retain_portable_sample(
                    problem, model, terminal, phase
                )
            return sample

        terminal_result = remesh.remesh_and_polish(problem, model, target, terminal, observe)
        checkpoint = model._checkpoint_stationary_state(
            problem, target, output, len(model.stationary_bond_schedule(target)) - 1, True
        )
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
            pins == terminal_result["after_polish"]["actual_pins"], "target checkpoint changed pins"
        )
        prefix.stable_inputs(prepared)
    cold_output = output / "target-cold"
    cold_output.mkdir()
    with model.build_problem(target, initial_bond=target["physics"]["bond"]) as fresh:
        anchor.configure(fresh, cold_output, prepared)
        fresh._seed_requalification_receipts_directory = cold_output / "typed-loads"
        loaded = requalification.restore_checked(
            fresh,
            target,
            final,
            pins,
            final["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
        )
        observed = remesh.sample(fresh, model, target, cold_output, "cold-load", np)
        cold_checks = remesh.canonical_checks(target, observed["metrics"])
        loaded["canonical_checks"] = cold_checks
        write_json(output / "cold-load.json", loaded)
    checks = remesh.canonical_checks(target, terminal_result["after_polish"]["metrics"])
    require(
        all(checks.values()) and all(cold_checks.values()),
        "original target warm or cold gates failed",
    )
    charge = prepared["contract"]["carried"]
    spent = (
        stages[0]["carry"]["new_spent"]["updates"]
        + stages[1]["polish"]["updates_total"]
        + terminal_result["polish"]["updates_total"]
    )
    require(
        spent <= prepared["contract"]["total_newton_update_bound"],
        "remaining original update bound exceeded",
    )
    result = {
        "schema": SCHEMA,
        "stages": stages,
        "terminal": terminal_result,
        "ca_in": prefix.TARGET_CA,
        "canonical_checks": checks,
        "cold_load_checks": cold_checks,
        "cold_load_verified": True,
        "completion_reached": True,
        "numerical_acceptance": True,
        "anchor_accepted": True,
        "causal_predictions": remesh.causal_predictions(terminal_result),
        "checkpoint": checkpoint,
        "original_acceptance_changed": False,
        "parents_unchanged": True,
        "qualified_prefix": prepared["contract"]["qualified_prefix"],
        "lineage": prepared["lineage"],
        "carried": charge,
        "new_updates_total": spent,
        "total_newton_update_bound": prepared["contract"]["total_newton_update_bound"],
        **prefix.NO_CLAIMS,
    }
    prefix.stable_inputs(prepared)
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
            "ca_in": prefix.TARGET_CA,
            "anchor_accepted": True,
            "original_acceptance_changed": False,
            **prefix.NO_CLAIMS,
        },
    )
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="metadata only; no native imports/output"
    )
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    output = prefix.output_path(args.out, prepared, additional=(prepared["prefix_runtime"],))
    if args.check:
        print(
            json.dumps(
                {
                    "verified": True,
                    "native_imported": False,
                    "output_created": False,
                    "contract": prepared["contract"],
                },
                allow_nan=False,
            )
        )
        return 0
    actual = run_manifest(args, prepared, arguments)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    runner.create_run_manifest(output, actual)
    runner.append_progress(output, {"event": "prepared", "runner": actual["runner"]}, first=True)
    try:
        write_json(
            output / "preparation.json",
            {
                "contract": prepared["contract"],
                "lineage": prepared["lineage"],
                "inputs": prepared["records"],
                "implementation": prepared["implementation"],
                "numerical_acceptance": False,
            },
        )
        execute(args, prepared, output, arguments)
        prefix.stable_inputs(prepared)
        write_json(
            output / "completion.json",
            {
                "schema": SCHEMA,
                "status": "complete",
                "numerical_acceptance": True,
                "anchor_accepted": True,
                "original_acceptance_changed": False,
                "qualification_schema": prefix.SCHEMA,
                "scope": SCOPE,
                "files": prefix.completion_inventory(output, exclude_solver=True),
            },
        )
        runner.append_progress(
            output, {"event": "final", "status": "complete", "numerical_acceptance": True}
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "numerical_acceptance": False,
            "anchor_accepted": False,
            "original_acceptance_changed": False,
            "parents_unchanged": diagnostic.parents_unchanged(prepared["records"]),
        }
        write_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", **failure})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
