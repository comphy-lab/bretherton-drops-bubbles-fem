"""Software controls for damping decisions, original budgets and full rollback."""

from __future__ import annotations

import json
import subprocess
import sys
from types import ModuleType

import pytest

from runners import run_paper_seed_recovery as recovery


@pytest.fixture
def case():
    value = recovery.model.read_case(recovery.ROOT / recovery.seed.BASE_CASE)
    value["case_id"] = "software_negative_seed"
    value["physics"]["ca_in"] = -0.02
    value["physics"]["bond"] = 0.0
    return value


def test_metadata_import_does_not_import_native_solver():
    program = """
import importlib.abc
import sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyoomph' or fullname.startswith('pyoomph.'):
            raise AssertionError('native import during metadata preparation')
sys.meta_path.insert(0, NoNative())
from runners import run_paper_seed_recovery
assert 'pyoomph' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=recovery.ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def lowered_policy():
    policy = recovery.RecoveryPolicy(1e-8)
    policy.remeshed(failed=True)
    policy.remeshed(failed=True)
    assert policy.omega == 0.125
    return policy


def observe_ratios(policy, values, start=1.0):
    policy.initial_residual(start)
    residual, rows = start, []
    for ratio in values:
        residual *= ratio
        rows.append(policy.observe(residual, policy.omega))
    return rows, residual


def test_recovery_requires_three_ratios_and_resets_window_after_each_factor_change():
    policy = lowered_policy()
    rows, residual = observe_ratios(policy, [0.875] * 3)
    assert [row["decision"] for row in rows] == ["hold", "hold", "double-for-next-update"]
    assert policy.omega == 0.25
    assert policy.ratios == []
    rows, residual = observe_ratios(policy, [0.75] * 3, residual)
    assert policy.omega == 0.5
    rows, residual = observe_ratios(policy, [0.5] * 3, residual)
    assert policy.omega == 1.0
    assert policy.observe(residual * 0.05, 1.0)["next_omega"] == 1.0


@pytest.mark.parametrize("ratio", [0.70, 0.99, 1.1])
def test_nonmatching_ratios_do_not_trigger_recovery(ratio):
    policy = lowered_policy()
    rows, _ = observe_ratios(policy, [ratio] * 3)
    assert policy.omega == 0.125
    assert all(row["decision"] == "hold" for row in rows)


def test_growth_after_escalation_halves_and_freezes_until_a_new_mesh():
    policy = lowered_policy()
    _, residual = observe_ratios(policy, [0.875] * 3)
    decision = policy.observe(residual * 1.1, 0.25)
    assert decision["decision"] == "halve-and-freeze-growth"
    assert policy.omega == 0.125 and policy.frozen
    observe_ratios(policy, [0.875] * 4)
    assert policy.omega == 0.125
    policy.remeshed(failed=False)
    assert not policy.frozen and policy.ratios == [] and policy.previous is None


def test_fullstep_growth_and_new_inversion_after_recovery_are_hard_stops():
    policy = lowered_policy()
    observe_ratios(policy, [0.875] * 3)
    with pytest.raises(recovery.RecoveryStop, match="new inversion"):
        policy.before_failed_remesh()
    policy.omega = 1.0
    policy.initial_residual(0.01)
    with pytest.raises(recovery.RecoveryStop, match="twofold"):
        policy.observe(0.021, 1.0)


@pytest.mark.parametrize("residual,applied", [(float("nan"), 0.125), (-1.0, 0.125), (0.8, 0.25)])
def test_nonfinite_negative_or_unrecorded_damping_refused(residual, applied):
    policy = lowered_policy()
    policy.initial_residual(1.0)
    with pytest.raises(ValueError):
        policy.observe(residual, applied)


def test_ratio_recovery_is_not_active_before_failed_inversion_remesh():
    policy = recovery.RecoveryPolicy(1e-8)
    observe_ratios(policy, [0.5] * 3)
    assert policy.omega == 0.5 and not policy.ever_recovered
    policy.remeshed(failed=True)
    observe_ratios(policy, [0.75] * 3)
    assert policy.omega == 0.25 and not policy.ever_recovered


def test_reference_prefix_uses_only_final_polish_trials_and_fixed_allowance():
    text = """Initial Maximum residuals 9
Newton Step 1: Maximum residuals 8
Initial Maximum residuals .2
Newton Step 1: Maximum residuals .1
Initial Maximum residuals .3
Newton Step 1: Maximum residuals .25
Initial Maximum residuals .4
Newton Step 1: Maximum residuals .35
"""
    prefix = recovery.reference_prefix(text, 3)
    assert prefix == [{0: 0.2, 1: 0.1}, {0: 0.3, 1: 0.25}, {0: 0.4, 1: 0.35}]
    assert recovery.prefix_comparison(prefix, 2, 1, 0.35 + 1e-7)["passed"]
    assert not recovery.prefix_comparison(prefix, 2, 1, 0.36)["passed"]
    assert not recovery.prefix_comparison(prefix, 2, 2, 0.1)["passed"]


def test_manifest_uses_this_entrypoint_and_declares_changed_method(case):
    prepared = {
        "component": {"commit": "software"},
        "environment": {},
        "bounds": {},
        "schedule": [0.0],
        "records": [{"path": "software-case", "sha256": "case"}],
        "implementation": {
            "runners/run_paper_seed_recovery.py": {
                "path": str(recovery.ROOT / "runners/run_paper_seed_recovery.py"),
                "sha256": "own",
            }
        },
    }
    path = recovery.ROOT / recovery.seed.BASE_CASE
    actual = recovery.manifest(path, case, prepared, {"kind": "software-stage"}, ["--check"])
    assert actual["runner"]["path"].endswith("run_paper_seed_recovery.py")
    assert actual["argv"][1] == actual["runner"]["path"]
    assert actual["seed_recovery_method"]["policy"]["stage_update_cap"] == 40
    assert actual["seed_recovery_method"]["api"]["callback_residual_assembly_performed"] is False


class FakeStepper:
    def __init__(self):
        self.steady = False

    def is_steady(self):
        return self.steady

    def make_steady(self):
        self.steady = True

    def undo_make_steady(self):
        self.steady = False


class FakeProblem:
    """A small callback-order fixture, never a native simulation substitute."""

    def __init__(self, fault=None):
        self.fault = fault
        self.max_newton_iterations = 10
        self.newton_relaxation_factor = 1.0
        self.newton_solver_tolerance = 1e-8
        self.inversion_remesh_max_retries = 3
        self.eigen_data_in_states = False
        self._last_bc_setting = "init"
        self._first_step = False
        self._taken_already_an_unsteady_step = True
        self._last_step_was_stationary = False
        self._seed_recovery_budget = None
        self._domains_to_remesh = set()
        self._remesh_requested_inside_solve = False
        self.value = 0.0
        self.residual = 0.001
        self.stepper = FakeStepper()
        self.history = []
        self.hooks = []

    def __iadd__(self, hook):
        self.hooks.append(hook)
        return self

    def ntime_stepper(self):
        return 1

    def time_stepper_pt(self, index):
        return self.stepper

    def _snapshot_state(self):
        return json.dumps({"value": self.value, "residual": self.residual}).encode()

    def _restore_state(self, payload):
        restored = json.loads(payload)
        self.value, self.residual = restored["value"], restored["residual"]
        if self.fault == "rollback":
            self.value += 1

    def invalidate_cached_mesh_data(self):
        pass

    def get_last_residual_convergence(self):
        return list(self.history)

    def solve(self, **kwargs):
        budget = self.hooks[-1]
        for trial in range(3):
            self.history = []
            budget.actions_before_newton_solve()
            limit = self.max_newton_iterations
            for update in range(limit):
                budget.before_update()
                if not self.history:
                    self.history.append(self.residual)
                self.value += 1
                budget.actions_after_newton_step()
                if trial < 2 and update == (1 if trial == 0 else 5):
                    budget.actions_before_remeshing([])
                    budget.actions_after_remeshing()
                    self.residual = 0.01
                    break
                ratio = 0.99 if self.fault else 1 - self.newton_relaxation_factor
                self.residual *= ratio
                self.history.append(self.residual)
                if self.residual < self.newton_solver_tolerance:
                    budget.actions_after_newton_solve()
                    return
            else:
                raise RuntimeError("software original stage cap exhausted")
        raise RuntimeError("software original stage cap exhausted")


@pytest.mark.parametrize("fault", [None, "budget", "rollback"])
def test_initial_polish_preserves_original_caps_and_rolls_back_failure(
    tmp_path, case, monkeypatch, fault
):
    # Supply a Python-only hooks base; importing the real pyoomph core is forbidden in this lane.
    fake = ModuleType("pyoomph.generic.problem")
    fake.GenericProblemHooks = type("SoftwareHooks", (), {})
    monkeypatch.setitem(sys.modules, "pyoomph.generic.problem", fake)
    problem = FakeProblem(fault)
    original_settings = {
        name: getattr(problem, name)
        for name in ("max_newton_iterations", "newton_relaxation_factor", "eigen_data_in_states")
    }
    monkeypatch.setattr(recovery.model, "_stationary_residual", lambda problem: problem.residual)
    monkeypatch.setattr(
        recovery.model,
        "_stationary_state_signature",
        lambda problem, az: {"field": problem.value, "residual": problem.residual},
    )
    monkeypatch.setattr(recovery.diagnostic, "actual_pins", lambda problem: {"software-pins": True})
    monkeypatch.setattr(recovery.seed, "check_forcing", lambda problem, case: {})
    prepared = {"reference_prefix": [{}, {}, {}]}
    if fault:
        with pytest.raises((RuntimeError, ValueError)):
            recovery.polish_seed(problem, case, tmp_path, prepared)
    else:
        result = recovery.polish_seed(problem, case, tmp_path, prepared)
        assert result["status"] == "converged"
        assert result["updates_total"] <= 40
        assert result["remeshes_total"] == 2
        assert result["prediction_is_jacobian_certificate"] is False
    receipt = json.loads((tmp_path / "polish/seed-recovery/polish.json").read_text())
    if fault == "budget":
        assert receipt["updates_total"] == 40
        assert receipt["stage_updates"] == 40
        assert receipt["remeshes_total"] == 2
        assert receipt["rollback_verified"] is True
        assert problem.value == 0.0 and problem.residual == 0.001
    if fault == "rollback":
        assert receipt["rollback_verified"] is False
    assert problem._seed_recovery_budget is None
    assert original_settings == {name: getattr(problem, name) for name in original_settings}
    assert not problem.stepper.is_steady()
    events = [
        json.loads(line)
        for line in (tmp_path / "polish/seed-recovery/progress.jsonl").read_text().splitlines()
    ]
    assert sum(event["event"] == "update-applied" for event in events) == receipt["updates_total"]


def test_later_stationary_stages_use_original_helper_only(case, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        recovery.seed, "stationary_stage", lambda *args: calls.append(args[3]) or {"original": True}
    )
    monkeypatch.setattr(
        recovery,
        "polish_seed",
        lambda *args: pytest.fail("experimental polish used after the first stage"),
    )
    assert recovery.stationary_stage(object(), case, tmp_path, 1, {}, object()) == {
        "original": True
    }
    assert calls == [1]
