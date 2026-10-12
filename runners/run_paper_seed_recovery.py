"""Test bounded damping recovery in only the initial negative-flow Bo=0 polish.

The fresh preparation, equations, target gates and later Bo polishes remain
unchanged. Recorded native residuals, rather than an additional assembly in a
Newton callback, determine the factor for the next update. Contraction ratios
are a hypothesis discriminator; they are not a Jacobian certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems import paper_bubble as model
from runners import run_paper_anchor as anchor
from runners import run_paper_case as runner
from runners import run_paper_seed_i as seed
from runners import run_pressure_diagnostic as diagnostic
from runners import run_pressure_remesh as remesh

require, sha, write_json = diagnostic.require, diagnostic.sha, diagnostic.write_json
FROZEN = "e5bc7126a274e8e4cfd946c5c32c0ad83f042304"
POLICY = {
    "initial_omega": 0.5,
    "entry_omega_max": 0.125,
    "ratio_window": 3,
    "relative_ratio_band": 0.10,
    "omega_max": 1.0,
    "fullstep_growth_stop": 2.0,
    "stage_update_cap": 40,
    "total_update_cap": 160,
    "remesh_cap": 3,
    "prefix_relative_allowance": 1e-5,
    "prefix_absolute_allowance": 1e-9,
    "prediction_first_recovery_by_trial2_update": 12,
    "prediction_total_updates_max": 40,
    "prediction_remeshes_max": 2,
}
API = {
    "pyoomph_commit": diagnostic.PYOOMPH,
    "residual_getter_copy": "src/problem.hpp:1360-1365; src/nanobind/problem.cpp:1337-1338",
    "before_step_dispatch": "src/nanobind/problem.cpp:138-140,688-689",
    "before_step_base_noop": "src/thirdparty/oomph-lib/include/problem.h:1188",
    "residual_order": "src/thirdparty/oomph-lib/include/problem.cc:9235,9297,9436,9452-9456",
    "live_factor": "src/nanobind/problem.cpp:821-825; problem.cc:9421",
    "inversion_recovery": "pyoomph/generic/problem.py:4056-4104",
    "snapshot_restore": "pyoomph/generic/problem.py:10769-10813",
    "callback_residual_assembly_performed": False,
}


class RecoveryStop(RuntimeError):
    """A failed recovery prediction that requires the original full rollback."""


class RecoveryPolicy:
    """Deterministic next-update policy, independent of native solver imports."""

    def __init__(self, tolerance):
        self.omega = POLICY["initial_omega"]
        self.tolerance = tolerance
        self.previous = None
        self.ratios = []
        self.failed_remeshes = 0
        self.ever_recovered = False
        self.frozen = False
        self.decisions = []

    def reset_window(self):
        self.previous = None
        self.ratios.clear()

    def initial_residual(self, value):
        require(math.isfinite(value) and value >= 0, "invalid native initial residual")
        self.previous = value

    def observe(self, residual, applied):
        require(applied == self.omega, "native damping differs from the recorded policy")
        require(math.isfinite(residual) and residual >= 0, "invalid native updated residual")
        ratio = residual / self.previous if self.previous else None
        event = {
            "residual_linf": residual,
            "applied_omega": applied,
            "ratio": ratio,
            "next_omega": self.omega,
            "decision": "hold",
            "growth_frozen": self.frozen,
        }
        self.previous = residual
        if residual <= self.tolerance:
            event["decision"] = "converged"
        elif applied == 1 and ratio is not None and ratio > POLICY["fullstep_growth_stop"]:
            event["decision"] = "hard-stop-fullstep-growth"
            self.decisions.append(event)
            raise RecoveryStop("full-step residual grew more than twofold")
        elif self.ever_recovered and ratio is not None and ratio > 1:
            self.omega *= 0.5
            self.frozen = True
            self.ratios.clear()
            event.update(
                decision="halve-and-freeze-growth", next_omega=self.omega, growth_frozen=True
            )
        elif ratio is not None:
            self.ratios.append(ratio)
            self.ratios = self.ratios[-POLICY["ratio_window"] :]
            expected = 1 - applied
            eligible = self.ever_recovered or (
                self.failed_remeshes > 0 and applied <= POLICY["entry_omega_max"]
            )
            if (
                eligible
                and not self.frozen
                and applied < 1
                and len(self.ratios) == 3
                and all(
                    abs(value - expected) <= POLICY["relative_ratio_band"] * expected
                    for value in self.ratios
                )
            ):
                self.omega = min(2 * applied, POLICY["omega_max"])
                self.ever_recovered = True
                self.ratios.clear()
                event.update(decision="double-for-next-update", next_omega=self.omega)
        else:
            self.ratios.clear()
        self.decisions.append(event)
        return event

    def before_failed_remesh(self):
        if self.ever_recovered:
            raise RecoveryStop("new inversion after damping escalation")

    def remeshed(self, *, failed):
        if failed:
            self.failed_remeshes += 1
            self.omega *= 0.5
        self.frozen = False
        self.reset_window()


def reference_prefix(log_text, trial_count):
    """Read only the final polish trials, keeping printed precision explicit."""
    starts = list(re.finditer(r"Initial Maximum residuals ([-+0-9.eE]+)", log_text))
    require(len(starts) >= trial_count, "failed log lacks the declared polish prefix")
    chosen = starts[-trial_count:]
    trials = []
    for index, start in enumerate(chosen):
        end = chosen[index + 1].start() if index + 1 < len(chosen) else len(log_text)
        rows = [
            [0, float(start.group(1))],
            *[
                [int(step), float(value)]
                for step, value in re.findall(
                    r"Newton Step (\d+): Maximum residuals ([-+0-9.eE]+)",
                    log_text[start.end() : end],
                )
            ],
        ]
        require(
            all(math.isfinite(value) and value > 0 for _, value in rows), "invalid prefix residual"
        )
        trials.append(dict(rows))
    return trials


def prefix_comparison(reference, trial, update, residual):
    expected = reference[trial].get(update) if trial < len(reference) else None
    allowance = (
        (POLICY["prefix_absolute_allowance"] + POLICY["prefix_relative_allowance"] * abs(expected))
        if expected is not None
        else None
    )
    return {
        "trial": trial,
        "update": update,
        "observed": residual,
        "reference": expected,
        "allowance": allowance,
        "passed": expected is not None and abs(residual - expected) <= allowance,
        "scope": "numerical prefix comparison through first escalation; no bitwise replay claim",
    }


def preflight(args):
    """Reuse the frozen case/environment checks and bind this actual producer separately."""
    require(
        Path(seed.__file__).resolve() == ROOT / "runners/run_paper_seed_i.py",
        "seed helper imported elsewhere",
    )
    prepared = seed.preflight(args)
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", FROZEN, prepared["component"]["commit"])
    for relative, record in prepared["implementation"].items():
        require(
            hashlib.sha256(diagnostic.git(ROOT, "show", f"{FROZEN}:{relative}")).hexdigest()
            == record["sha256"],
            "frozen seed implementation changed",
        )
    relative = "runners/run_paper_seed_recovery.py"
    path = Path(__file__).resolve()
    require(path == ROOT / relative, "recovery entrypoint differs")
    digest = hashlib.sha256(
        diagnostic.git(ROOT, "show", f"{prepared['component']['commit']}:{relative}")
    ).hexdigest()
    require(
        path.is_file() and not path.is_symlink() and sha(path) == digest,
        "recovery entrypoint differs from the committed source",
    )
    prepared["implementation"][relative] = {"path": str(path), "sha256": digest}
    failed = args.failed_runtime.resolve(strict=True)
    paths = [
        failed / "run-manifest.json",
        failed / "failure.json",
        failed / "stages/bond_000/polish/seed/polish.json",
        failed.parent / "runner.log",
    ]
    old_manifest, failure, polish = [json.loads(path.read_text()) for path in paths[:3]]
    require(
        old_manifest["component_commit"] == FROZEN
        and old_manifest["case_sha256"] == sha(args.case),
        "failed observation has different source/case",
    )
    runner.require_clean_source(old_manifest["component"])
    seed.validate_environment(old_manifest["pyoomph"])
    require(
        old_manifest["component"]["commit"] == FROZEN
        and old_manifest["azimuthal_stability"] is True
        and old_manifest["case_fields_used"] == model.CASE_FIELDS_USED,
        "failed observation has different method inventory/layout",
    )
    require(
        old_manifest["runner"]["sha256"]
        == hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{FROZEN}:runners/run_paper_seed_i.py")
        ).hexdigest(),
        "failed producer differs from the frozen source",
    )
    require(
        failure["status"] == "failed"
        and failure["numerical_acceptance"] is False
        and polish["status"] == "failed"
        and polish["rollback_verified"] is True,
        "failed seed/rollback receipt differs",
    )
    require(
        polish["updates_total"] == 40
        and polish["remeshes_total"] == 2
        and [trial["relaxation"] for trial in polish["trials"]] == [0.5, 0.25, 0.125],
        "reference failure is not the reviewed bounded polish",
    )
    for item in paths:
        require(item.is_file() and not item.is_symlink(), "failed input must be a regular file")
        prepared["records"].append({"path": str(item), "sha256": sha(item)})
    prepared["reference_prefix"] = reference_prefix(paths[-1].read_text(), len(polish["trials"]))
    prepared["recovery_policy"] = POLICY
    require(
        prepared["bounds"]["per_mesh_stage_newton_cap"] == 40
        and prepared["bounds"]["per_polish_total_newton_cap"] == 160
        and prepared["bounds"]["per_polish_remesh_cap"] == 3,
        "original budgets differ",
    )
    return prepared


def manifest(case_path, case, prepared, stage, argv):
    value = seed.manifest(case_path, case, prepared, stage)
    own = prepared["implementation"]["runners/run_paper_seed_recovery.py"]
    value.update(
        runner=own,
        runner_path=own["path"],
        runner_sha256=own["sha256"],
        argv=[sys.executable, own["path"], *argv],
        seed_recovery_method={
            "policy": POLICY,
            "api": API,
            "scope": "initial Bo=0 seed polish only; later methods unchanged",
        },
        failed_observation=prepared["records"][1:],
    )
    return value


def build_recovery_problem(case):
    """The original problem with only the formerly empty before-step callback supplied."""
    base = model._native_problem_types()[0]

    class RecoveryProblem(base):
        _seed_recovery_budget = None

        def actions_before_newton_step(self):
            # The inherited oomph hook is empty at a186. No residual assembly is performed here.
            budget = self._seed_recovery_budget
            if budget is not None and budget.active:
                budget.before_update()

    return RecoveryProblem(case, initial_bond=case["physics"]["bond"])


def restore_full(problem, snapshot, signature, pins, flags, steady, settings):
    """Restore the original complete state and every altered method setting."""
    problem._seed_recovery_budget = None
    problem.eigen_data_in_states = True
    problem._restore_state(snapshot)
    for name, value in flags.items():
        setattr(problem, name, value)
    for index, was_steady in enumerate(steady):
        stepper = problem.time_stepper_pt(index)
        stepper.make_steady() if was_steady else stepper.undo_make_steady()
    problem._domains_to_remesh.clear()
    problem._remesh_requested_inside_solve = False
    problem.invalidate_cached_mesh_data()
    require(
        model._stationary_state_signature(problem, True) == signature
        and diagnostic.actual_pins(problem) == pins,
        "complete recovery rollback failed",
    )
    for name, value in settings.items():
        setattr(problem, name, value)
    require(
        all(getattr(problem, name) == value for name, value in settings.items()),
        "method settings did not restore",
    )
    return {
        "verified": True,
        "full_fields_pins_history_forcing_and_layout": True,
        "method_settings_restored": True,
    }


def polish_seed(problem, case, output, prepared):
    """Keep the original 40/160/3 recovery skeleton with one next-factor policy."""
    from pyoomph.generic.problem import GenericProblemHooks

    require(
        case["physics"]["bond"] == 0 and case["physics"]["ca_in"] == -0.02,
        "experimental polish is restricted to the initial Bo=0 seed",
    )
    require(problem._seed_recovery_budget is None, "recovery hook already active")
    directory = output / "polish/seed-recovery"
    directory.mkdir(parents=True, exist_ok=False)
    journal = directory / "progress.jsonl"
    runner.append_progress(
        directory, {"event": "prepared", "policy": POLICY, "api": API}, first=True
    )

    def emit(event, **values):
        runner.append_progress(directory, {"event": event, **values})

    settings = {
        name: getattr(problem, name)
        for name in (
            "max_newton_iterations",
            "newton_relaxation_factor",
            "newton_solver_tolerance",
            "inversion_remesh_max_retries",
            "eigen_data_in_states",
        )
    }
    flags = {
        name: getattr(problem, name)
        for name in (
            "_last_bc_setting",
            "_first_step",
            "_taken_already_an_unsteady_step",
            "_last_step_was_stationary",
        )
    }
    steady = [problem.time_stepper_pt(i).is_steady() for i in range(problem.ntime_stepper())]
    signature = model._stationary_state_signature(problem, True)
    pins = diagnostic.actual_pins(problem)
    problem.eigen_data_in_states = True
    try:
        snapshot = problem._snapshot_state()
        diagnostic.atomic(directory / "before.dump", snapshot)
    except Exception:
        problem.eigen_data_in_states = settings["eigen_data_in_states"]
        raise
    record = {
        "method": "bounded native-history damping recovery",
        "policy": POLICY,
        "api": API,
        "checkpoint": str((directory / "before.dump").relative_to(output)),
        "checkpoint_sha256": hashlib.sha256(snapshot).hexdigest(),
        "checkpoint_format": "memory_snapshot",
        "mesh_reference_base": str(Path.cwd()),
        "status": "running",
        "initial_residual_linf": model._stationary_residual(problem),
        "trials": [],
        "prefix_comparisons": [],
        "prediction_is_jacobian_certificate": False,
    }

    class Budget(GenericProblemHooks):
        def __init__(self):
            super().__init__()
            self.active = True
            self.updates = self.stage_updates = self.remeshes = 0
            self.converged = self.remeshing = self.remesh_after_convergence = False
            self.policy = RecoveryPolicy(case["tolerances"]["newton"])
            self.pending = []
            self.processed = 0
            self.trial = -1
            self.step = 0
            self.first_recovery = None

        def remaining(self):
            return min(40 - self.stage_updates, 160 - self.updates)

        def consume(self):
            values = list(
                problem.get_last_residual_convergence()
            )  # Native copy; no assembly/hooks.
            if not values:
                return
            if self.processed == 0:
                self.policy.initial_residual(float(values[0]))
                self.processed = 1
                comparison = prefix_comparison(
                    prepared["reference_prefix"], self.trial, 0, float(values[0])
                )
                if not self.policy.ever_recovered:
                    record["prefix_comparisons"].append(comparison)
                emit("trial-initial-residual", trial=self.trial, residual=float(values[0]))
            require(len(values) >= self.processed, "native residual history shrank within a trial")
            while self.processed < len(values):
                require(self.pending, "native history has no matching applied update")
                update = self.pending.pop(0)
                residual = float(values[self.processed])
                self.processed += 1
                prior = self.policy.ever_recovered
                emit(
                    "native-updated-residual",
                    **update,
                    residual_linf=residual if math.isfinite(residual) else None,
                    residual_finite=math.isfinite(residual),
                )
                try:
                    decision = self.policy.observe(residual, update["applied_omega"])
                except RecoveryStop:
                    emit("recovery-hard-stop", **{**update, **self.policy.decisions[-1]})
                    raise
                emit("residual-observed", **{**update, **decision})
                if not prior:
                    record["prefix_comparisons"].append(
                        prefix_comparison(
                            prepared["reference_prefix"],
                            self.trial,
                            update["trial_update"],
                            residual,
                        )
                    )
                if decision["decision"] == "double-for-next-update" and self.first_recovery is None:
                    self.first_recovery = {
                        "trial": self.trial,
                        "trial_update": update["trial_update"],
                        "total_updates": update["total_updates"],
                    }

        def before_update(self):
            self.consume()
            require(self.remaining() > 0, "original Newton update budget exhausted")
            problem.newton_relaxation_factor = self.policy.omega
            require(
                problem.newton_relaxation_factor == self.policy.omega, "live Newton factor differs"
            )

        def actions_before_newton_solve(self):
            if not self.active:
                return
            require(self.remaining() > 0, "original Newton update budget exhausted")
            self.converged = False
            self.trial += 1
            self.step = self.processed = 0
            self.pending.clear()
            self.policy.reset_window()
            problem.max_newton_iterations = self.remaining()
            problem.newton_relaxation_factor = self.policy.omega
            trial = {
                "index": self.trial,
                "relaxation": self.policy.omega,
                "remaining_updates": self.remaining(),
                "stage_updates_before": self.stage_updates,
                "updates_total_before": self.updates,
                "remeshes_before": self.remeshes,
            }
            record["trials"].append(trial)
            emit("trial-started", **trial)

        def actions_after_newton_step(self):
            if not self.active:
                return
            self.updates += 1
            self.stage_updates += 1
            self.step += 1
            require(
                self.stage_updates <= 40 and self.updates <= 160, "original update bounds exceeded"
            )
            update = {
                "trial": self.trial,
                "trial_update": self.step,
                "total_updates": self.updates,
                "stage_updates": self.stage_updates,
                "applied_omega": float(problem.newton_relaxation_factor),
            }
            self.pending.append(update)
            emit("update-applied", **update)

        def actions_after_newton_solve(self):
            if self.active:
                self.consume()
                self.converged = True

        def abandon_unobserved(self):
            for update in self.pending:
                emit(
                    "updated-residual-unavailable",
                    **update,
                    reason="native assembly/solve raised before recording this residual",
                )
            self.pending.clear()

        def actions_before_remeshing(self, active_remeshers):
            if not self.active:
                return
            self.abandon_unobserved()
            require(self.remeshes < 3, "original three-remesh budget exhausted")
            if not self.converged:
                self.policy.before_failed_remesh()
            self.remeshing = True
            self.remesh_after_convergence = self.converged
            self.converged = False
            emit(
                "remesh-started",
                failed_inversion=not self.remesh_after_convergence,
                updates_total=self.updates,
                stage_updates=self.stage_updates,
            )

        def actions_after_remeshing(self):
            if self.active and self.remeshing:
                self.remeshing = False
                self.remeshes += 1
                if self.remesh_after_convergence:
                    self.stage_updates = 0
                self.policy.remeshed(failed=not self.remesh_after_convergence)
                self.remesh_after_convergence = False
                emit(
                    "remesh-finished",
                    remeshes_total=self.remeshes,
                    next_omega=self.policy.omega,
                    stage_updates=self.stage_updates,
                )

    counter = Budget()
    problem += counter
    problem._seed_recovery_budget = counter
    problem.inversion_remesh_max_retries = 3
    try:
        while counter.remaining() > 0:
            previous_remeshes = counter.remeshes
            problem.solve(
                max_newton_iterations=counter.remaining(),
                newton_relaxation_factor=counter.policy.omega,
                do_not_set_IC=True,
            )
            residual = model._stationary_residual(problem)
            seed.check_forcing(problem, case)
            if residual <= case["tolerances"]["newton"]:
                record.update(status="converged", residual_linf=residual)
                break
            require(
                counter.remeshes != previous_remeshes,
                "returned solve is not stationary and has no evidenced remesh",
            )
        require(record["status"] == "converged", "original Newton budget exhausted")
    except Exception as error:
        try:
            counter.consume()
        except Exception as history_error:  # noqa: BLE001 - residual inspection cannot prevent rollback
            emit(
                "failed-history-inspection",
                error_type=type(history_error).__name__,
                error=str(history_error),
            )
        counter.abandon_unobserved()
        record.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            rollback_verified=False,
        )
        counter.active = False
        problem._seed_recovery_budget = None
        try:
            forensic = problem._snapshot_state()
            diagnostic.atomic(directory / "unaccepted-current.dump", forensic)
            record["forensic_state"] = {
                "path": "unaccepted-current.dump",
                "sha256": hashlib.sha256(forensic).hexdigest(),
                "format": "memory_snapshot",
                "mesh_reference_base": str(Path.cwd()),
                "numerical_acceptance": False,
                "scope": "state after solver exception, before outer rollback; native recovery may already have restored an internal snapshot",
            }
        except Exception as forensic_error:  # noqa: BLE001 - this observation cannot prevent rollback
            record["forensic_state"] = {"retained": False, "error": str(forensic_error)}
        restored = restore_full(problem, snapshot, signature, pins, flags, steady, settings)
        record.update(rollback_verified=True, rollback=restored)
        raise
    finally:
        counter.active = False
        problem._seed_recovery_budget = None
        for name, value in settings.items():
            setattr(problem, name, value)
        record.update(
            updates_total=counter.updates,
            stage_updates=counter.stage_updates,
            remeshes_total=counter.remeshes,
            first_recovery=counter.first_recovery,
            policy_decisions=counter.policy.decisions,
            journal={"path": str(journal.relative_to(output)), "sha256": sha(journal)},
        )
        record["hypothesis_predictions"] = {
            "prefix_within_declared_allowance": bool(record["prefix_comparisons"])
            and all(row["passed"] for row in record["prefix_comparisons"]),
            "first_recovery_by_trial2_update12": counter.first_recovery is not None
            and counter.first_recovery["trial"] == 2
            and counter.first_recovery["trial_update"] <= 12,
            "converged_within40_total_and_two_remeshes": record["status"] == "converged"
            and counter.updates <= 40
            and counter.remeshes <= 2,
            "fullstep_each_observed_ratio_at_most0p1": any(
                row["applied_omega"] == 1 and row["ratio"] is not None
                for row in counter.policy.decisions
            )
            and all(
                row["ratio"] <= 0.1
                for row in counter.policy.decisions
                if row["applied_omega"] == 1 and row["ratio"] is not None
            ),
            "scope": "mechanism predictions; separate from target five-gate acceptance",
        }
        write_json(directory / "polish.json", record)
    return record


def stationary_stage(problem, case, output, index, prepared, np):
    if index:
        return seed.stationary_stage(problem, case, output, index, prepared, np)
    seed.check_forcing(problem, case)
    polish = polish_seed(problem, case, output, prepared)
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


def execute(args, prepared, output, argv):
    """Replay the frozen seed route, substituting only its first stationary polish."""
    import numpy as np

    case, schedule = prepared["case"], prepared["schedule"]
    stages, cases, paths = [], [], []
    for index in range(len(schedule)):
        current = seed.stage_case(case, index, schedule)
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
                argv,
            ),
        )
        runner.append_progress(
            stage_output, {"event": "prepared", "case_id": current["case_id"]}, first=True
        )
        cases.append(current)
        paths.append(stage_output)
    runner.append_progress(output, {"event": "native-start", "initial_bond": schedule[0]})
    with build_recovery_problem(cases[0]) as problem:
        anchor.configure(problem, output, prepared)
        seed.preparatory_relaxation(problem, cases[0], paths[0])
        for index, current in enumerate(cases):
            seed.stable_inputs(prepared)
            if index:
                seed.adopt_bond(problem, cases[index - 1], current, paths[index])
            stages.append(stationary_stage(problem, current, paths[index], index, prepared, np))
            seed.stable_inputs(prepared)
            runner.append_progress(
                output, {"event": "stationary_bo_stage", "index": index, "bond": schedule[index]}
            )
        seed.check_forcing(problem, case)
        terminal = output / "terminal"
        (terminal / "solver").mkdir(parents=True)
        write_json(
            terminal / "run-manifest.json",
            manifest(args.case, case, prepared, {"kind": "terminal_remesh"}, argv),
        )
        runner.append_progress(
            terminal, {"event": "prepared", "case_id": case["case_id"]}, first=True
        )

        def observe(phase):
            seed.stable_inputs(prepared)
            if phase == "after-remesh":
                remesh.normalize_pressure_dummies(problem, model, terminal, np)
            observed = remesh.sample(problem, model, case, terminal, phase, np)
            if phase != "after-polish":
                observed["retained_state"] = remesh.retain_portable_sample(
                    problem, model, terminal, phase
                )
            return observed

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
        seed.stable_inputs(prepared)
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
        "schema": "paper-seed-recovery-result-v1",
        "stages": stages,
        "terminal": terminal_result,
        "checks": checks,
        "cold_load_checks": cold_checks,
        "cold_load_verified": True,
        "checkpoint": checkpoint,
        "numerical_acceptance": accepted,
        "completion_reached": accepted,
        "initial_polish_predictions": stages[0]["polish"]["hypothesis_predictions"],
        "predictions_define_numerical_acceptance": False,
        "branch_identity_verified": False,
        "eigenanalysis_performed": False,
        "transient_validation_claimed": False,
    }
    seed.stable_inputs(prepared)
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
            "ca_in": seed.TARGET_CA,
            "bond": case["physics"]["bond"],
            "checkpoint": checkpoint["resume_packet"],
            "branch_identity_verified": False,
            "eigenanalysis_performed": False,
            "transient_validation_claimed": False,
        },
    )
    require(accepted, "fresh seed target or cold-load gates failed")
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "failed-runtime", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    require(not args.out.is_symlink(), "output must not be a symlink")
    output = runner.validate_output_path(args.out, ROOT)
    require(
        not output.is_relative_to(args.failed_runtime.resolve()), "output overlaps failed evidence"
    )
    if args.check:
        print(
            json.dumps(
                {
                    "verified": True,
                    "native_imported": False,
                    "bounds": prepared["bounds"],
                    "policy": POLICY,
                    "api": API,
                    "bo_schedule": prepared["schedule"],
                }
            )
        )
        return 0
    actual = manifest(
        args.case, prepared["case"], prepared, {"kind": "fresh_negative_seed_recovery"}, arguments
    )
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(output / "run-manifest.json", actual)
    runner.append_progress(output, {"event": "prepared", "runner": actual["runner"]}, first=True)
    try:
        write_json(
            output / "preparation.json",
            {
                "inputs": prepared["records"],
                "implementation": prepared["implementation"],
                "bounds": prepared["bounds"],
                "policy": POLICY,
                "api": API,
                "numerical_acceptance": False,
            },
        )
        result = execute(args, prepared, output, arguments)
        seed.stable_inputs(prepared)
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
            output, {"event": "final", "numerical_acceptance": result["numerical_acceptance"]}
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
