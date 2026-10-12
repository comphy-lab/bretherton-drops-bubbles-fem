"""Remesh one retained transient seed before the unchanged stationary polish.

The seed is an initial guess, never a stationary certificate. A separate phase
owns one shared Newton/remesh budget through the seed, Bond ladder and warm/cold
qualification. The original failed attempt remains a bound, immutable input.
``--check`` reads metadata and source only; it imports no native solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems import paper_bubble as model  # noqa: E402
from runners import run_paper_case as runner  # noqa: E402
from runners import run_paper_cold_history as history  # noqa: E402
from runners import run_pressure_diagnostic as diagnostic  # noqa: E402
from runners import run_pressure_remesh as remesh  # noqa: E402

require, sha, write_json = diagnostic.require, diagnostic.sha, runner._exclusive_json
ENTRYPOINT = "runners/run_paper_volume_seed_remesh.py"
BASE_SOURCE = "6e23bf3fb281a8b070a3d3b11b927be58b3d56de"
PARENT_SOURCE = "9b77001fde2a56e93a85c70395055ff1e9443557"
CASE_SHA = "63f4fc6a3612cffb28805c6c181d4739f79d226827cf3e8d664890a7268372b0"
PHASE_SCHEMA = "herrada-volume-seed-quality-remesh-phase-v1"
BUDGET = {"per_mesh_stage_updates": 40, "total_updates": 160, "physical_remeshes": 3}
FAILURE_HASHES = {
    "runtime/run-manifest.json": "c125f7a3b6a822f01bc98e4ef1a9238a5bcd50989820c5c9974b9cd87d648740",
    "runtime/polish/seed/polish.json": (
        "6394b630e9401cd872fc425d4e8b9ec723b672128707b57dbd31af521cc1d5ea"
    ),
    "runtime/polish/seed/before.dump": (
        "8116f81ef74dedb4fe2bb1f95853c7356119677e2562441a662abfcf2c6463f9"
    ),
    "runtime/solver/_gmsh/PaperTwoPhaseMesh.msh": (
        "e47728235718948683735df487f94a987a404a4b6acb9d9185957667d0d0ca4b"
    ),
    "supervisor-result.json": "429207d992093568400cb47f28389814abf641cf93190e6f13c833a7b549ceb5",
    "runner.log": "8c9bc8b6405faef29b7a70e0de13c3684c6d8e6655f1c2c818e6846274dd65cc",
}
API = {
    "pyoomph_commit": model.PYOOMPH_COMMIT,
    "python_problem_sha256": "b75d6a810daddb3e8f0853c60fea780b33220c6c09cf819e5f219f5af056018f",
    "hook_registration": "pyoomph/generic/problem.py:3459-3478",
    "physical_remesh_hooks": "pyoomph/generic/problem.py:10030-10032,10156-10158",
    "load_notification": "pyoomph/generic/problem.py:10747",
    "inversion_restore": "pyoomph/generic/problem.py:4056-4104",
    "before_update": "src/nanobind/problem.cpp:138-140,688-689; problem.cc:9235",
    "before_update_base_noop": "src/thirdparty/oomph-lib/include/problem.h:1188",
    "endtime_loader_writer": "pyoomph/generic/problem.py:10341-10349",
}
SETTINGS = (
    "max_newton_iterations",
    "newton_relaxation_factor",
    "newton_solver_tolerance",
    "inversion_remesh_max_retries",
    "eigen_data_in_states",
)
FLAGS = (
    "_last_bc_setting",
    "_first_step",
    "_taken_already_an_unsteady_step",
    "_last_step_was_stationary",
)


def bound_file(record, base):
    """Reject symlinks in every existing path component before resolving it."""
    require(
        isinstance(record, dict) and isinstance(record.get("path"), str),
        "bound artifact requires a path and digest",
    )
    path = Path(record["path"]).expanduser()
    path = path if path.is_absolute() else base / path
    require(
        not any(item.is_symlink() for item in (path, *path.parents)),
        "bound artifact contains a symlink",
    )
    return runner._bound_file(record, base)


def phase_contract(path):
    path = Path(path)
    phase = runner._read_json(path)
    require(phase.get("schema") == PHASE_SCHEMA, "unexpected seed-quality phase schema")
    require(
        isinstance(phase.get("phase_id"), str)
        and re.fullmatch(r"[a-zA-Z0-9._-]{1,96}", phase["phase_id"]),
        "invalid new phase id",
    )
    require(phase.get("kind") == "new_informative_phase", "a fresh phase must be declared")
    require(
        phase.get("budget_scope") == "seed_bond_ladder_warm_and_cold_qualification",
        "phase budget must include the ladder and qualification",
    )
    budget = phase.get("fresh_budget", {})
    require(
        isinstance(budget, dict) and set(budget) == {*BUDGET, "wall_seconds"},
        "fresh phase requires explicit finite resource bounds",
    )
    require(
        all(type(budget[key]) is int and budget[key] == value for key, value in BUDGET.items()),
        "fresh phase changed the frozen numerical caps",
    )
    wall = budget["wall_seconds"]
    require(
        type(wall) in (int, float) and math.isfinite(wall) and wall > 0,
        "fresh phase requires a positive finite wall budget",
    )
    return phase


def failed_inputs(restart, terminal):
    """Admit only the retained original producer and its still-failed receipts."""
    attempt = Path(restart["parent_manifest"]["path"]).parent.parent
    records = []
    for relative, digest in FAILURE_HASHES.items():
        path = bound_file({"path": str(attempt / relative), "sha256": digest}, attempt)
        records.append({"path": str(path), "sha256": digest})
    parent = runner._read_json(attempt / "runtime/run-manifest.json")
    polish = runner._read_json(attempt / "runtime/polish/seed/polish.json")
    supervisor = runner._read_json(attempt / "supervisor-result.json")
    producer_sha = parent["component"]["sources"]["runners/run_paper_case.py"]["sha256"]
    require(
        parent["component_commit"] == parent["component"]["commit"] == PARENT_SOURCE,
        "original seed producer source differs",
    )
    require(
        parent["runner"]["sha256"] == parent["runner_sha256"] == producer_sha
        and parent["runner"]["path"] == parent["runner_path"]
        and Path(parent["runner_path"]).name == "run_paper_case.py",
        "original seed producer entrypoint differs",
    )
    require(
        supervisor.get("status") == "failed"
        and supervisor.get("completion_reached") is False
        and supervisor.get("cleanup_unresolved") is False,
        "original producer must remain terminal and unqualified",
    )
    require(
        polish.get("status") == "failed"
        and polish.get("rollback_verified") is True
        and (polish.get("updates_total"), polish.get("stage_updates"), polish.get("remeshes_total"))
        == (40, 40, 1),
        "original failure debit differs",
    )
    require(
        [trial.get("relaxation") for trial in polish["trials"]] == [0.5, 0.25]
        and [trial.get("remaining_updates") for trial in polish["trials"]] == [40, 37]
        and polish["parameters"] == {"Ca_in": 0.02, "Bo": 0.0},
        "original bounded polish failure differs",
    )
    require(
        terminal.get("schema") == "stone-volume-primary-terminal-v1"
        and terminal.get("physical_acceptance") is False
        and terminal.get("target_bond_reached") is False
        and terminal.get("polish") == polish
        and terminal.get("combined_deadline_utc") == "2026-10-12T23:36:09Z",
        "original terminal history differs",
    )
    require(
        all(
            terminal["artifacts"].get(str(attempt / relative)) == FAILURE_HASHES[relative]
            for relative in (
                "runtime/run-manifest.json",
                "runtime/polish/seed/polish.json",
                "runtime/polish/seed/before.dump",
                "supervisor-result.json",
            )
        ),
        "terminal custody hashes differ",
    )
    return records, {
        "status": "failed",
        "updates": 40,
        "physical_remeshes": 1,
        "stationary_endpoint_exists": False,
        "historical_deadline_utc": terminal["combined_deadline_utc"],
        "budget_transferred_to_new_phase": False,
    }


def source_records(component, parent):
    """Bind every imported helper; keep the original model/runner wholly unchanged."""
    records = {}
    required = {ENTRYPOINT}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("runners.", "problems.")) and getattr(module, "__file__", None):
            path = Path(module.__file__)
            require(
                path.resolve().is_relative_to(ROOT) and not path.is_symlink(),
                "seed-quality helper imported from another checkout",
            )
            required.add(str(path.resolve().relative_to(ROOT)))
    for relative in sorted(required | set(component["sources"])):
        path = ROOT / relative
        committed = runner._git(ROOT, "show", f"{component['head']}:{relative}")
        digest = hashlib.sha256(committed).hexdigest() if committed is not None else None
        require(
            digest is not None and path.is_file() and not path.is_symlink() and sha(path) == digest,
            f"uncommitted seed-quality implementation: {relative}",
        )
        if relative != ENTRYPOINT:
            frozen = runner._git(ROOT, "show", f"{BASE_SOURCE}:{relative}")
            require(
                frozen is not None and hashlib.sha256(frozen).hexdigest() == digest,
                f"frozen seed-quality helper changed: {relative}",
            )
        if relative in parent["component"]["sources"]:
            require(
                parent["component"]["sources"][relative]["sha256"] == digest,
                f"original producer method changed: {relative}",
            )
        records[relative] = {"path": str(path), "sha256": digest}
    return records


def preflight(args):
    """Bind the new phase and original seed before output or native imports."""
    require(args.source.resolve(strict=True) == ROOT, "runner must use its own isolated source")
    require(Path(__file__).resolve() == ROOT / ENTRYPOINT, "entrypoint custody differs")
    phase_path = bound_file({"path": str(args.phase), "sha256": sha(args.phase)}, Path.cwd())
    phase = phase_contract(phase_path)
    case_path = bound_file({"path": str(args.case), "sha256": CASE_SHA}, Path.cwd())
    case = model.read_case(case_path)
    component, environment = runner.component_revision(), runner.pyoomph_revision()
    runner.require_clean_source(component)
    runner.require_pinned_environment(environment)
    api_path = Path(metadata.distribution("pyoomph").locate_file("pyoomph/generic/problem.py"))
    bound_file({"path": str(api_path), "sha256": API["python_problem_sha256"]}, Path.cwd())
    packet_path = bound_file(phase.get("restart_packet"), phase_path.parent)
    restart = runner.validate_restart(
        packet_path,
        case,
        CASE_SHA,
        component=component,
        environment=environment,
        azimuthal_stability=True,
    )
    require(
        restart["state"]["format"] == "memory_snapshot"
        and restart["stage"]
        == {
            "kind": "pending_polish",
            "bond_index": 0,
            "bond": 0.0,
            "ca_in": 0.02,
            "bc_mode": "init",
        },
        "quality remesh requires the original pending Bo=0 seed",
    )
    terminal_path = bound_file(phase.get("original_terminal"), phase_path.parent)
    records, old_failure = failed_inputs(restart, runner._read_json(terminal_path))
    require(
        restart["state"]["sha256"] == FAILURE_HASHES["runtime/polish/seed/before.dump"]
        and len(restart["meshes"]) == 1
        and restart["meshes"][0]["sha256"]
        == FAILURE_HASHES["runtime/solver/_gmsh/PaperTwoPhaseMesh.msh"],
        "retained seed or mesh differs",
    )
    saved = Path(restart["state"]["path"]).read_bytes()
    seed_metadata = history.serialized_metadata(saved)
    require(
        seed_metadata["time"] == seed_metadata["endtime"] == case["solve"]["relax_time"]
        and runner.state_bc_cache_marker(Path(restart["state"]["path"])) == "init",
        "retained seed control/history differs",
    )
    original = Path(restart["parent_manifest"]["path"])
    implementation = source_records(component, runner._read_json(original))
    records.extend(
        [
            {"path": str(path), "sha256": sha(path)}
            for path in (phase_path, packet_path, terminal_path, case_path, api_path)
        ]
    )
    native_core = environment["native_core"]
    bound_file(native_core, ROOT)
    records.append(dict(native_core))
    require(
        not any(item.is_symlink() for item in (args.out, *args.out.parents)),
        "new output contains a symlink",
    )
    output = runner.validate_output_path(args.out, ROOT)
    attempt = original.parent.parent
    require(
        not output.is_relative_to(attempt) and not attempt.is_relative_to(output),
        "new output overlaps original evidence",
    )
    return {
        "phase": phase,
        "case": case,
        "case_path": str(case_path),
        "output": str(output),
        "restart": restart,
        "component": component,
        "environment": environment,
        "implementation": implementation,
        "inputs": records,
        "original_failure": old_failure,
        "seed_metadata": seed_metadata,
        "seed_stream": runner.snapshot_bc_cache_signature(
            saved, mesh_reference_base=Path(restart["mesh_reference_base"])
        ),
        "seed_is_stationary": False,
        "schedule": model.stationary_bond_schedule(case),
    }


def stable_inputs(prepared):
    for record in (*prepared["inputs"], *prepared["implementation"].values()):
        bound_file(record, ROOT)


def manifest(prepared, arguments):
    own = prepared["implementation"][ENTRYPOINT]
    return {
        "schema": "pyoomph-runner-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "case_file": prepared["case_path"],
        "case_path": prepared["case_path"],
        "case_id": prepared["case"]["case_id"],
        "case_sha256": CASE_SHA,
        "case_fields_used": list(model.CASE_FIELDS_USED),
        "component": prepared["component"],
        "component_commit": prepared["component"]["commit"],
        "pyoomph": prepared["environment"],
        "runner": own,
        "runner_path": own["path"],
        "runner_sha256": own["sha256"],
        "argv": [sys.executable, own["path"], *arguments],
        "units": "nondimensional (sigma=mu_liquid=R0=1)",
        "nondimensionalization": prepared["case"]["nondimensionalization"],
        "azimuthal_stability": True,
        "implementation": prepared["implementation"],
        "new_phase": prepared["phase"],
        "original_failure": prepared["original_failure"],
        "restart_from": {
            **prepared["restart"]["state"],
            "parent_manifest": prepared["restart"]["parent_manifest"],
            "packet": prepared["restart"]["packet"],
            "qualified_stationary": False,
        },
        "api": API,
    }


class PhaseBudget:
    """Carry, then audit, the frozen helper's counters before its first read."""

    def __init__(self, wall_seconds, emit, clock=time.monotonic):
        self.clock, self.started, self.wall_seconds = clock, clock(), wall_seconds
        self.emit = emit
        self.updates = self.remeshes = 0
        self.remesh_starts = 0
        self.pending_remesh = False
        self.current = None
        self.label = "pre_polish_quality_remesh"
        self.active = True
        self.stages = []

    def check_wall(self):
        require(self.clock() - self.started < self.wall_seconds, "new phase wall budget exhausted")

    def carry(self, hook):
        require(
            hook.__class__.__module__ == model.__name__
            and hook.__class__.__qualname__ == "polish_confined_state.<locals>.Budget",
            "unknown or ambiguous polish budget hook",
        )
        require(self.current is None or not self.current.active, "two polish budgets are active")
        require(
            hook.active is True
            and hook.updates == hook.remeshes == hook.stage_updates == 0
            and not hook.trials
            and not hook.remeshing
            and not hook.converged,
            "new polish budget was already used",
        )
        self.check_wall()
        require(self.updates < BUDGET["total_updates"], "new phase Newton budget exhausted")
        hook.updates, hook.remeshes = self.updates, self.remeshes
        self.current = hook
        item = {
            "label": self.label,
            "carried_updates": self.updates,
            "carried_physical_remeshes": self.remeshes,
            "stage_updates_at_entry": 0,
        }
        self.stages.append(item)
        self.emit("polish_budget_carried", **item)

    def audit(self):
        if self.current is not None:
            require(
                (self.current.updates, self.current.remeshes) == (self.updates, self.remeshes),
                "native hooks and carried polish budget disagree",
            )
        require(
            0 <= self.updates <= 160 and 0 <= self.remeshes <= 3,
            "new phase resource cap was exceeded",
        )

    def before_update(self):
        self.check_wall()
        self.audit()
        require(
            self.current is not None and self.current.active,
            "Newton update outside the bound frozen polish",
        )
        require(
            self.updates < 160 and self.current.stage_updates < 40,
            "new phase or mesh-stage Newton budget exhausted",
        )

    def before_remesh(self, remeshers):
        if not self.active:
            return
        self.check_wall()
        self.audit()
        require(len(remeshers) == 1, "quality phase requires one shared two-phase remesher")
        require(not self.pending_remesh, "nested physical remesh")
        require(self.remeshes < 3, "new phase exhausted its three physical remeshes")
        self.pending_remesh = True
        self.remesh_starts += 1
        self.emit(
            "physical_remesh_started", label=self.label, physical_remeshes_before=self.remeshes
        )

    def after_remesh(self):
        if not self.active or not self.pending_remesh:
            return  # load_state emits this notification without a physical remesh.
        if self.current is not None and not self.current.active:
            self.pending_remesh = False  # A failed remake followed by helper rollback.
            return
        self.pending_remesh = False
        self.remeshes += 1
        self.emit("physical_remesh_completed", label=self.label, physical_remeshes=self.remeshes)

    def record(self):
        return {
            "updates_total": self.updates,
            "physical_remeshes_total": self.remeshes,
            "physical_remesh_starts": self.remesh_starts,
            "physical_remesh_pending": self.pending_remesh,
            "budget": {**BUDGET, "wall_seconds": self.wall_seconds},
            "wall_seconds": self.clock() - self.started,
            "stages": [dict(stage) for stage in self.stages],
            "budget_hooks_agree": self.current is None
            or (self.current.updates, self.current.remeshes) == (self.updates, self.remeshes),
            "scope": "seed_bond_ladder_warm_and_cold_qualification",
            "original_failure_debits_are_separate": True,
        }


def budget_problem_type(base, hooks_base):
    """Change hook registration and its empty before-update callback only."""

    class SharedBudgetProblem(base):
        _quality_phase = None

        def __iadd__(self, other):
            if self._quality_phase is not None and isinstance(other, hooks_base):
                self._quality_phase.carry(other)
            return super().__iadd__(other)

        def actions_before_newton_step(self):
            super().actions_before_newton_step()
            if self._quality_phase is not None:
                self._quality_phase.before_update()

    return SharedBudgetProblem


def attach_budget(problem, phase, hooks_base):
    class Guard(hooks_base):
        def actions_before_newton_solve(self):
            phase.check_wall()
            phase.audit()

        def actions_after_newton_step(self):
            phase.updates += 1
            phase.emit(
                "newton_update",
                label=phase.label,
                phase_updates=phase.updates,
                phase_physical_remeshes=phase.remeshes,
            )

        def actions_before_remeshing(self, remeshers):
            phase.before_remesh(remeshers)

        def actions_after_remeshing(self):
            phase.after_remesh()

    require(
        problem._quality_phase is None and isinstance(problem._hooks, list),
        "unexpected native hook registration layout",
    )
    guard = Guard()
    problem += guard
    require(
        problem._hooks[-1] is guard and guard.get_problem() is problem,
        "phase guard registration failed",
    )
    problem._quality_phase = phase


def snapshot(problem):
    setting = problem.eigen_data_in_states
    try:
        problem.eigen_data_in_states = True
        return problem._snapshot_state()
    finally:
        problem.eigen_data_in_states = setting


def state_record(problem):
    raw = snapshot(problem)
    return {
        "raw": raw,
        "signature": model._stationary_state_signature(problem, True),
        "pins": diagnostic.actual_pins(problem),
        "flags": {name: getattr(problem, name) for name in FLAGS},
        "settings": {name: getattr(problem, name) for name in SETTINGS},
        "steady": [problem.time_stepper_pt(i).is_steady() for i in range(problem.ntime_stepper())],
        "metadata": history.serialized_metadata(raw),
    }


def restore(problem, saved):
    """Restore controls as well as the complete serialized native state."""
    problem._restore_state(saved["raw"])
    for name, value in {**saved["flags"], **saved["settings"]}.items():
        setattr(problem, name, value)
    for index, was_steady in enumerate(saved["steady"]):
        stepper = problem.time_stepper_pt(index)
        stepper.make_steady() if was_steady else stepper.undo_make_steady()
    problem._domains_to_remesh.clear()
    problem._remesh_requested_inside_solve = False
    problem.invalidate_cached_mesh_data()
    after = state_record(problem)
    require(
        after["signature"] == saved["signature"]
        and after["pins"] == saved["pins"]
        and after["flags"] == saved["flags"]
        and after["settings"] == saved["settings"]
        and after["steady"] == saved["steady"]
        and history.normalized_bytes(after["raw"], Path.cwd())
        == history.normalized_bytes(saved["raw"], Path.cwd()),
        "complete quality-remesh rollback or round trip failed",
    )
    return {"verified": True, "full_stream_and_actual_pins": True, "controls_restored": True}


def load_exact(problem, case, restart, directory):
    """Restore the evidenced endtime scalar, then compare the complete stream."""
    runner.verify_restart_artifacts(restart)
    saved = Path(restart["state"]["path"]).read_bytes()
    problem.load_state(
        restart["state"]["path"],
        quiet=True,
        additional_info={"exchange_msh_file": restart["mesh_exchange"]},
    )
    before = snapshot(problem)
    endtime = history.restore_endtime(problem, saved, before, case)
    after = snapshot(problem)
    expected = runner.snapshot_bc_cache_signature(
        saved, mesh_reference_base=Path(restart["mesh_reference_base"])
    )["payload_sha256"]
    stream = history.verify_snapshots(
        saved, before, after, Path(restart["mesh_reference_base"]), Path.cwd(), expected
    )
    loaded = state_record(problem)
    require(
        loaded["signature"]["parameters"]
        == {"Ca_in": restart["stage"]["ca_in"], "Bo": restart["stage"]["bond"]}
        and loaded["signature"]["bc_cache_marker"] == restart["stage"]["bc_mode"],
        "loaded seed forcing or BC cache differs",
    )
    restored = restore(problem, loaded)
    record = {
        "verified": True,
        "stream": stream,
        "signature": loaded["signature"],
        "actual_pins": loaded["pins"],
        "snapshot_restore": restored,
        "restored_writer_endtime": endtime,
        "stationary_certificate_claimed_by_load": False,
    }
    write_json(directory / "load.json", record)
    return record


def remesh_seed(problem, case, directory, phase):
    """One projected seed, with complete before/after custody and failure rollback."""
    before = state_record(problem)
    diagnostic.atomic(directory / "before.dump", before["raw"])
    record = {
        "status": "running",
        "initial_guess_is_stationary": False,
        "before": {key: value for key, value in before.items() if key != "raw"},
        "before_sha256": hashlib.sha256(before["raw"]).hexdigest(),
        "physical_remesh_debit": 1,
        "same_state_claimed_after_projection": False,
    }
    try:
        require(phase.updates == phase.remeshes == 0, "pre-polish intervention was already used")
        require(
            problem.force_remesh() is True and phase.remeshes == 1 and not phase.pending_remesh,
            "pre-polish remesh did not complete exactly once",
        )
        after = state_record(problem)
        diagnostic.atomic(directory / "after.dump", after["raw"])
        keys = (
            "time",
            "output_step",
            "continue_step",
            "compression",
            "dts",
            "unsteady_steps",
            "suggested_next_dt",
            "endtime",
        )
        require(
            all(before["metadata"][key] == after["metadata"][key] for key in keys),
            "pre-polish projection changed native time/history controls",
        )
        require(
            before["signature"]["parameters"] == after["signature"]["parameters"]
            and before["signature"]["frame_speed"] == after["signature"]["frame_speed"]
            and before["signature"]["layout"]["dof_names"]
            == after["signature"]["layout"]["dof_names"]
            and before["flags"] == after["flags"]
            and before["settings"] == after["settings"]
            and before["steady"] == after["steady"],
            "projection changed forcing, ODE or controls",
        )
        require(
            before["metadata"]["meshes"] != after["metadata"]["meshes"],
            "quality intervention did not retain a new mesh reference",
        )
        require(
            all(
                (Path.cwd() / ref).resolve().is_relative_to(directory.parent)
                for _, _, ref in after["metadata"]["meshes"]
            ),
            "projected seed mesh escaped the new run",
        )
        residual = model._stationary_residual(problem)
        record.update(
            status="projected_initial_guess",
            stationary_residual_linf=residual,
            after={key: value for key, value in after.items() if key != "raw"},
            after_sha256=hashlib.sha256(after["raw"]).hexdigest(),
            snapshot_restore=restore(problem, after),
            numerical_acceptance=False,
        )
        return record
    except BaseException as error:
        phase.active = False
        phase.pending_remesh = False
        record.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            rollback_verified=False,
            numerical_acceptance=False,
        )
        record["rollback"] = restore(problem, before)
        record["rollback_verified"] = True
        raise
    finally:
        write_json(directory / "projection.json", record)


def polish_stage(problem, case, output, phase, label, relaxation):
    phase.label = label
    phase.check_wall()
    phase.audit()
    require(phase.updates < 160, "new phase Newton budget exhausted")
    before = state_record(problem)
    try:
        result = model.polish_confined_state(
            problem, case, output, label, initial_relaxation=relaxation
        )
    except BaseException:
        phase.active = False
        phase.pending_remesh = False
        rollback = restore(problem, before)
        write_json(output / f"{label}-full-rollback.json", rollback)
        raise
    finally:
        phase.audit()
        phase.emit(
            "polish_finished",
            label=label,
            phase_updates=phase.updates,
            phase_physical_remeshes=phase.remeshes,
        )
    phase.check_wall()
    return result


def configure(problem, output):
    problem.set_output_directory(str(output / "solver"))
    problem.set_linear_solver("superlu")
    problem.set_c_compiler("system")
    problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
    problem.initialise()


def qualify_cold(case, checkpoint, warm, output, prepared, phase, np):
    phase.check_wall()
    restart = runner.validate_restart(
        output / checkpoint["resume_packet"],
        case,
        CASE_SHA,
        component=prepared["component"],
        environment=prepared["environment"],
        azimuthal_stability=True,
    )
    cold_output = output / "cold"
    cold_output.mkdir()
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as cold:
        configure(cold, cold_output)
        load = load_exact(cold, case, restart, cold_output)

        def physical(signature):
            return {
                **signature,
                "serialized_state": {
                    "payload_sha256": signature["serialized_state"]["payload_sha256"],
                    "bc_cache_marker": signature["serialized_state"]["bc_cache_marker"],
                },
            }

        require(
            physical(load["signature"]) == physical(warm["signature"])
            and load["actual_pins"] == warm["actual_pins"],
            "cold load changed endpoint fields, pins or history",
        )
        sample = remesh.sample(cold, model, case, cold_output, "endpoint", np)
        checks = remesh.canonical_checks(case, sample["metrics"])
    phase.check_wall()
    return {"verified": True, "load": load, "sample": sample, "checks": checks}


def execute(prepared, output):
    """No transient fallback or fitted route; each failure propagates once."""
    case = prepared["case"]
    phase = PhaseBudget(
        prepared["phase"]["fresh_budget"]["wall_seconds"],
        lambda event, **values: runner.append_progress(output, {"event": event, **values}),
    )
    import numpy as np
    from pyoomph.generic.problem import GenericProblemHooks

    problem_type = budget_problem_type(model._native_problem_types()[0], GenericProblemHooks)
    stages = []
    try:
        with problem_type(case, initial_bond=0.0) as problem:
            configure(problem, output)
            attach_budget(problem, phase, GenericProblemHooks)
            seed_directory = output / "seed-quality-remesh"
            seed_directory.mkdir()
            load_exact(problem, case, prepared["restart"], seed_directory)
            projection = remesh_seed(problem, case, seed_directory, phase)
            for index, bond in enumerate(prepared["schedule"]):
                phase.check_wall()
                problem.get_global_parameter("Bo").value = bond
                label = f"bond_{index:03d}"
                polish = polish_stage(
                    problem, case, output, phase, label, 0.5 if index == 0 else 1.0
                )
                checkpoint = model._checkpoint_stationary_state(problem, case, output, index, True)
                stages.append(
                    {
                        "bond_index": index,
                        "bond": bond,
                        "polish": polish,
                        "checkpoint": checkpoint,
                        "phase_budget": phase.record(),
                    }
                )
                stable_inputs(prepared)
            require(
                stages[-1]["bond"] == case["physics"]["bond"] == 1.56,
                "new phase has no complete Bo=1.56 endpoint",
            )
            warm = remesh.sample(problem, model, case, output, "warm-endpoint", np)
            warm_checks = remesh.canonical_checks(case, warm["metrics"])
            phase.check_wall()
        cold = qualify_cold(case, checkpoint, warm, output, prepared, phase, np)
        accepted = all(warm_checks.values()) and all(cold["checks"].values())
        result = {
            "schema": "herrada-volume-seed-quality-remesh-result-v1",
            "status": "completed" if accepted else "failed",
            "completion_reached": accepted,
            "numerical_acceptance": accepted,
            "scientific_status": "checks_passed" if accepted else "checks_failed",
            "projection": projection,
            "stages": stages,
            "checks": warm_checks,
            "cold_load": cold,
            "cold_load_checks": cold["checks"],
            "cold_load_verified": True,
            "original_failure": prepared["original_failure"],
            "new_phase_budget": phase.record(),
            "checkpoint": checkpoint["resume_packet"],
            "ca_in": 0.02,
            "bond": 1.56,
            "branch_identity_verified": False,
            "eigenanalysis_performed": False,
            "transient_validation_claimed": False,
            **warm["metrics"],
        }
        phase.audit()
        stable_inputs(prepared)
        phase.check_wall()
        write_json(output / "result.json", result)
        write_json(output / "summary.json", result)
        require(accepted, "new endpoint failed an unchanged warm or cold qualification gate")
        return result
    finally:
        write_json(output / "phase-budget.json", phase.record())


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "case", "phase", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    if args.check:
        print(
            json.dumps(
                {
                    "verified": True,
                    "native_imported": False,
                    "new_phase": prepared["phase"],
                    "original_failure": prepared["original_failure"],
                    "seed_is_stationary": False,
                    "schedule": prepared["schedule"],
                },
                allow_nan=False,
            )
        )
        return 0
    stable_inputs(prepared)
    output = Path(prepared["output"])
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    runner.create_run_manifest(output, manifest(prepared, arguments))
    runner.append_progress(
        output, {"event": "prepared", "phase_id": prepared["phase"]["phase_id"]}, first=True
    )
    try:
        write_json(
            output / "preparation.json",
            {key: value for key, value in prepared.items() if key != "case"},
        )
        result = execute(prepared, output)
        stable_inputs(prepared)
        runner.append_progress(
            output, {"event": "final", "completion_reached": result["completion_reached"]}
        )
        return 0
    except BaseException as error:
        failure = {
            "status": "failed",
            "completion_reached": False,
            "numerical_acceptance": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "inputs_unchanged": diagnostic.parents_unchanged(prepared["inputs"]),
            "original_failure": prepared["original_failure"],
        }
        write_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", **failure})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
