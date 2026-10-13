"""Offline contracts for distinct post-C1 wall authority and carried work."""

from __future__ import annotations

import ast
import copy
import importlib.abc
import inspect
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from runners import run_paper_volume_post_c1 as driver

ROOT = Path(__file__).resolve().parents[1]
DECLARED = datetime(2026, 10, 13, 5, tzinfo=UTC)
NOW = DECLARED + timedelta(hours=1)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith("pyoomph."):
            raise AssertionError("post-C1 software tests forbid native imports")


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch):
    assert not any(name.startswith("pyoomph") for name in sys.modules)
    monkeypatch.setattr(sys, "meta_path", [NoNative(), *sys.meta_path])

    def refuse(*args, **kwargs):
        raise AssertionError("post-C1 software tests forbid native construction")

    monkeypatch.setattr(driver.model, "_native_problem_types", refuse)
    monkeypatch.setattr(driver.model, "build_problem", refuse)


def configuration():
    return {
        "schema": driver.SCHEMA + "-configuration",
        "simulation": driver.SIMULATION,
        "resource_policy": copy.deepcopy(driver.RESOURCES),
        "phase": {
            "kind": "new_bounded_post_C1_phase",
            "phase_id": driver.SIMULATION + "-attempt-0001",
            "declared_at_utc": DECLARED.isoformat(),
            "deadline_utc": (DECLARED + timedelta(seconds=21600)).isoformat(),
            "wall_max_seconds": 21600,
            "carried_debits": copy.deepcopy(driver.DEBITS),
            "bo_zero_new_update_cap": 37,
        },
    }


@pytest.mark.parametrize("seconds", [1, 21600])
def test_positive_declared_duration_up_to_six_hours(seconds):
    cfg = configuration()
    cfg["phase"]["deadline_utc"] = (DECLARED + timedelta(seconds=seconds)).isoformat()
    phase, remaining = driver.phase_authority(cfg, DECLARED)
    assert remaining == seconds
    phase["carried_debits"]["updates_total"] = 100
    assert cfg["phase"]["carried_debits"]["updates_total"] == 3


@pytest.mark.parametrize("key,value", [
    ("kind", "resume_existing_phase"),
    ("phase_id", "owned-resume-attempt-0004"),
    ("wall_max_seconds", 21601),
    ("wall_max_seconds", True),
    ("bo_zero_new_update_cap", 40),
    ("carried_debits", {**driver.DEBITS, "updates_total": 0}),
    ("carried_debits", {**driver.DEBITS, "physical_remesh_starts": True}),
    ("declared_at_utc", "2026-10-13T05:00:00"),
    ("declared_at_utc", "2026-10-13T05:00:00+01:00"),
    ("declared_at_utc", "2026-10-13T04:00:00Z"),
    ("declared_at_utc", "2026-10-13T07:00:00Z"),
    ("deadline_utc", "2026-10-13T05:00:00Z"),
    ("deadline_utc", "2026-10-13T06:00:00Z"),
    ("deadline_utc", "2026-10-13T11:00:01Z"),
])
def test_phase_declaration_refuses_clock_or_debt_changes(key, value):
    cfg = configuration()
    cfg["phase"][key] = value
    with pytest.raises(ValueError):
        driver.phase_authority(cfg, NOW)


def test_new_clock_does_not_revive_the_expired_driver():
    old_deadline = driver.old.DEADLINE
    phase = driver.PostC1PhaseBudget(configuration()["phase"], lambda *a, **k: None, [],
                                    clock=lambda: 0, wall_clock=lambda: NOW)
    phase.check_wall()
    assert driver.old.DEADLINE == old_deadline == "2026-10-13T04:41:19Z"
    with pytest.raises(ValueError, match="original absolute phase deadline"):
        driver.old.remaining_wall(NOW)
    with pytest.raises(ValueError, match="original absolute phase deadline"):
        driver.old.RetainedPhaseBudget(100, lambda *a, **k: None, [], wall_clock=lambda: NOW)


@pytest.mark.parametrize("clock_kind", ["UTC", "monotonic"])
def test_phase_refuses_either_expired_clock(clock_kind):
    wall, monotonic = [NOW], [1000]
    phase = driver.PostC1PhaseBudget(configuration()["phase"], lambda *a, **k: None, [],
                                    clock=lambda: monotonic[0], wall_clock=lambda: wall[0])
    phase.check_wall()
    if clock_kind == "UTC":
        wall[0] = datetime.fromisoformat(phase.authority["deadline_utc"])
    else:
        monotonic[0] += phase.wall_seconds
    with pytest.raises(ValueError, match="deadline|wall budget"):
        phase.check_wall()


class Hooks:
    def get_problem(self):
        return self.problem


class FakeProblem:
    def __init__(self):
        self._quality_phase, self._hooks = None, []

    def __iadd__(self, hook):
        hook.problem = self
        self._hooks.append(hook)
        return self

    def actions_before_newton_step(self):
        pass


def frozen_polish_hook(problem):
    # Run the unchanged counter class with a Python fake base, without its
    # native import, problem construction, residual evaluation or solve.
    body = ast.parse(inspect.getsource(driver.model.polish_confined_state)).body[0].body
    counter = next(item for item in body if isinstance(item, ast.ClassDef) and item.name == "Budget")
    scope = {"__name__": driver.model.__name__, "GenericProblemHooks": Hooks,
             "initial_relaxation": 0.5, "stage_limit": 40, "problem": problem}
    exec(compile(ast.Module(body=[counter], type_ignores=[]), "<frozen-fake-polish-hook>", "exec"), scope)  # noqa: S102
    scope["Budget"].__qualname__ = "polish_confined_state.<locals>.Budget"
    return scope["Budget"]()


@pytest.fixture
def carried():
    events = []
    phase = driver.PostC1PhaseBudget(configuration()["phase"],
                                    lambda event, **values: events.append({"event": event, **values}),
                                    [{"label": "bond_000", "carried_updates": 0,
                                      "carried_physical_remeshes": 1, "stage_updates_at_entry": 0}],
                                    clock=lambda: 0, wall_clock=lambda: NOW)
    phase.label = "bond_000"
    problem = driver.seed.budget_problem_type(FakeProblem, Hooks)()
    driver.seed.attach_budget(problem, phase, Hooks)
    hook = frozen_polish_hook(problem)
    problem += hook
    return SimpleNamespace(phase=phase, problem=problem, hook=hook, events=events)


def update(bundle):
    bundle.problem.actions_before_newton_step()
    for hook in bundle.problem._hooks:
        callback = getattr(hook, "actions_after_newton_step", None)
        if callback:
            callback()


def quality_remesh(bundle):
    bundle.hook.converged = True
    for hook in bundle.problem._hooks:
        hook.actions_before_remeshing([object()])
    for hook in bundle.problem._hooks:
        hook.actions_after_remeshing()


def test_exact_carry_and_37_first_stage_allowance(carried):
    phase, hook = carried.phase, carried.hook
    assert (phase.updates, phase.remeshes, phase.remesh_starts, hook.stage_updates) == (3, 1, 1, 3)
    assert (hook.updates, hook.remeshes, hook.remaining_updates()) == (3, 1, 37)
    assert phase.record()["fresh_numerical_budget_allocated"] is False
    assert phase.record()["new_wall_authority_allocated"] is True
    assert phase.record()["original_phase_deadline_utc"] == driver.old.DEADLINE
    assert phase.record()["budget_hooks_agree"] is True
    assert len(phase.stages) == 2 and phase.stages[-1]["resumed_existing_mesh_stage"] is True
    assert driver.load_retained is driver.old.load_retained
    assert driver.stable_inputs is driver.old.stable_inputs


def test_bo_zero_37_new_updates_remain_bounded_after_quality_reset(carried):
    for _ in range(5):
        update(carried)
    quality_remesh(carried)
    assert carried.hook.stage_updates == 0
    assert (carried.phase.remeshes, carried.phase.remesh_starts) == (2, 2)
    for _ in range(32):
        update(carried)
    assert carried.phase.updates == 40 and carried.hook.stage_updates == 32
    with pytest.raises(ValueError, match="Bo=0 phase-budget refusal: 37"):
        update(carried)
    assert carried.phase.updates == 40
    assert sum(row["event"] == "newton_update" for row in carried.events) == 37


def test_later_bond_stage_keeps_frozen_per_mesh_cap(carried):
    update(carried)
    carried.hook.active = False
    carried.phase.label = "bond_001"
    hook = frozen_polish_hook(carried.problem)
    carried.problem += hook
    carried.hook = hook
    assert (hook.updates, hook.remeshes, hook.stage_updates) == (4, 1, 0)
    for _ in range(40):
        update(carried)
    with pytest.raises(ValueError, match="mesh-stage Newton budget"):
        update(carried)
    assert carried.phase.updates == 44 and hook.stage_updates == 40


def test_later_stage_cannot_cross_aggregate_160_updates(carried):
    carried.phase.label = "bond_002"
    carried.phase.updates = carried.hook.updates = 159
    update(carried)
    assert carried.hook.stage_updates == 4
    with pytest.raises(ValueError, match="Newton budget"):
        update(carried)
    assert carried.phase.updates == 160


def test_only_two_more_physical_remeshes_and_no_load_notification_debit(carried):
    carried.phase.after_remesh()
    assert (carried.phase.remeshes, carried.phase.remesh_starts) == (1, 1)
    quality_remesh(carried)
    quality_remesh(carried)
    with pytest.raises(ValueError, match="three physical remeshes"):
        quality_remesh(carried)
    assert (carried.phase.remeshes, carried.phase.remesh_starts) == (3, 3)
    assert carried.phase.record()["budget_hooks_agree"] is True


def terminal():
    return {"simulation": driver.OLD_SIMULATION, "attempt": "attempt-0003", "status": "failed",
            "scientific_result_accepted": False, "original_deadline_utc": driver.old.DEADLINE,
            "original_debits": copy.deepcopy(driver.DEBITS), "new_completed_update_events": 0,
            "new_physical_remesh_events": 0, "completed_Newton_updates_recorded": 0,
            "new_wall_authority": False, "outcome": "original_wall_timeout_during_first_stationary_solve",
            "gas_C1_preparation_reached": True, "load_restore_verification": {"verified": True}}


def test_original_terminal_is_admitted_only_as_failed_and_unqualified():
    driver.validate_terminal(terminal())


@pytest.mark.parametrize("key,value", [
    ("status", "completed"), ("scientific_result_accepted", True),
    ("new_completed_update_events", 1), ("new_physical_remesh_events", 1),
    ("completed_Newton_updates_recorded", 1), ("new_wall_authority", True),
    ("original_debits", {**driver.DEBITS, "stage_updates": 0}),
])
def test_completed_work_or_original_acceptance_tampering_refuses(key, value):
    record = terminal()
    record[key] = value
    with pytest.raises(ValueError, match="expired original outcome"):
        driver.validate_terminal(record)


@pytest.mark.parametrize("error,expected", [
    ("post-C1 absolute phase deadline reached", "phase-budget-refusal"),
    ("new phase or mesh-stage Newton budget exhausted", "phase-budget-refusal"),
    ("unchanged warm/cold five-gate qualification failed", "native-or-contract-refusal"),
])
def test_failure_record_never_claims_stationary_nonexistence(tmp_path, monkeypatch, error, expected):
    output = tmp_path / "runtime"
    own = {"path": str(ROOT / driver.ENTRYPOINT), "sha256": driver.sha(ROOT / driver.ENTRYPOINT)}
    prepared = {"output": str(output), "implementation": {driver.ENTRYPOINT: own}, "inputs": [],
                "phase": configuration()["phase"], "lineage": {"parent_is_unqualified": True},
                "original_failure": {"status": "failed"}}
    monkeypatch.setattr(driver, "preflight", lambda args: prepared)
    monkeypatch.setattr(driver, "stable_inputs", lambda prepared: None)
    monkeypatch.setattr(driver.seed, "manifest", lambda *args: {})

    def refuse(*args):
        raise RuntimeError(error)

    monkeypatch.setattr(driver, "execute", refuse)
    with pytest.raises(RuntimeError, match=error):
        driver.main(["--source", str(ROOT), "--case", "unused", "--configuration", "unused",
                     "--out", str(output)])
    record = driver.runner._read_json(output / "failure.json")
    assert record["failure_class"] == expected
    assert record["stationary_nonexistence_claimed"] is False
    assert record["parent_is_unqualified"] is True
    assert record["completion_reached"] is record["numerical_acceptance"] is False


@pytest.fixture
def admitted(tmp_path):
    path = os.environ.get("STONE_VOLUME_POST_C1_CONFIGURATION")
    if path is None:
        pytest.skip("actual admission requires the retained external configuration")
    path = Path(path)
    cfg = driver.runner._read_json(path)
    expired = driver.runner._read_json(Path(cfg["expired_resume_configuration"]["path"]))
    probe = driver.runner._read_json(Path(expired["probe_configuration"]["path"]))
    args = SimpleNamespace(source=ROOT, case=Path(probe["bindings"]["case"]["path"]),
                           configuration=path, out=tmp_path / "never-native-runtime", check=True)
    return SimpleNamespace(args=args, prepared=driver.preflight(args), cfg=cfg)


def test_actual_admission_binds_c7_c1_restore_and_all_173_retained_sources(admitted):
    prepared = admitted.prepared
    hashes = {record["sha256"] for record in prepared["inputs"]}
    assert {driver.STATE_SHA, driver.TERMINAL_SHA, driver.ARCHIVE_SHA,
            "491d72487740950ac9fa89228f6e2d78863dd42e3c314c96cf995c4d8023dcf8",
            "88fc95f34c9d651ab1965f793bf1404fcc6a81a99c587e49a497508e660b186c"} <= hashes
    archive = driver.runner._read_json(Path(admitted.cfg["expired_archive"]["path"]))
    inputs = {record["path"]: record["sha256"] for record in prepared["inputs"]}
    assert all(inputs[str(Path(archive["source"]) / name)] == row["sha256"]
               for name, row in archive["files"].items())
    assert prepared["lineage"]["retained_local_source"]["file_count"] == 173
    assert prepared["lineage"]["archive_custody"]["destination_verified_now"] is False
    assert not any(path.startswith(archive["destination"] + "/") for path in inputs)
    assert prepared["lineage"]["C1_half_sum_values_verified"] == 82257
    assert prepared["restart"]["state"]["sha256"] == driver.STATE_SHA
    assert prepared["parent_is_unqualified"] is True
    assert prepared["native_execution_authorized"] is False
    assert prepared["source_seal_pending"] is (prepared["component"]["clean"] is not True)
    assert prepared["phase"]["carried_debits"] == driver.DEBITS
    assert prepared["original_failure"]["budget_transferred_to_new_phase"] is False
    assert prepared["schedule"][-1] == 1.56 and prepared["schedule"][0] == 0
    assert not admitted.args.out.exists()


def test_actual_hash_binding_refuses_changed_terminal_or_completed_work_ledger(admitted):
    records = admitted.prepared["inputs"]
    chosen = [record for record in records if record["sha256"] in
              {driver.TERMINAL_SHA, driver.ARCHIVE_SHA, driver.STATE_SHA}
              or record["path"].endswith("attempt-0003/runtime/progress.jsonl")]
    assert len(chosen) >= 4
    for record in chosen:
        with pytest.raises(ValueError, match="digest|hash"):
            driver.bind({**record, "sha256": "0" * 64}, [])


def test_frozen_22fc_source_and_original_entrypoint_are_unchanged():
    for name in (driver.old.ENTRYPOINT, driver.old.PREPARATION_HELPER,
                 driver.old.HELPER, driver.seed.ENTRYPOINT, "problems/paper_bubble.py", "uv.lock"):
        assert (ROOT / name).read_bytes() == driver.runner._git(ROOT, "show", f"{driver.BASE}:{name}")
    assert driver.old.DEADLINE == "2026-10-13T04:41:19Z"
