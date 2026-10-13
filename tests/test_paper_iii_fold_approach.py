"""Offline admission, shared-budget and partial-result software contracts."""

from __future__ import annotations

import ast
import copy
import importlib.abc
import json
import sys
from datetime import UTC, datetime, timedelta

import pytest

from runners import run_paper_iii_fold_approach as approach
from runners import run_paper_modes as modes


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith(("pyoomph.", "_pyoomph")):
            raise AssertionError("native import forbidden in approach software tests")


@pytest.fixture(autouse=True)
def no_native():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


@pytest.fixture
def cases():
    parent = approach.model.read_case(approach.ROOT / "validation/herradaEtAl_paper_model/cases/iii-ca-m0p0038-medium.json")
    target = copy.deepcopy(parent)
    target["case_id"] = "software_III_approach"
    target["physics"]["ca_in"] = approach.LADDER[-1]
    return parent, target


def test_declared_controls_keep_physics_and_bound_six_m0_calls(cases):
    parent, target = cases
    contract = approach.control_contract(parent, target, list(approach.LADDER))
    assert set(contract["changed_fields"]) == {"physics.ca_in", "case_id"}
    assert contract["phase_newton_update_cap"] == 160
    assert contract["phase_physical_remesh_cap"] == 3
    assert contract["explicit_terminal_remesh_calls"] == 0
    assert [row["ca_in"] for row in contract["m0_requests"]] == list(approach.LADDER)
    assert sum(row["eigenproblem_calls"] for row in contract["m0_requests"]) == 6
    assert all(row["modes"] == [0] and row["shifts"] == [-.001, .001]
               and row["neigen"] == 8 for row in contract["m0_requests"])


@pytest.mark.parametrize("section,field,value", [
    ("physics", "gas_viscosity_ratio", 2e-5),
    ("physics", "bond", 1.57),
    ("tolerances", "newton", 1e-6),
    ("mesh", "film_elements", 8),
    ("solve", "max_newton_iterations", 41),
])
def test_control_admission_refuses_undeclared_physics_or_solver_edits(cases, section, field, value):
    parent, target = cases
    target[section][field] = value
    with pytest.raises((ValueError, RuntimeError)):
        approach.control_contract(parent, target, list(approach.LADDER))


@pytest.mark.parametrize("values", [[-.00381, -.00383], [-.00381, -.00382, -.00384],
                                   [-.00381, -.00382, -.00383, -.00384]])
def test_ladder_cannot_be_retried_or_extended(cases, values):
    with pytest.raises((ValueError, RuntimeError)):
        approach.control_contract(*cases, values)


class Hooks:
    def actions_before_newton_solve(self):
        pass

    def actions_after_newton_step(self):
        pass

    def actions_before_remeshing(self, active_remeshers):
        pass

    def actions_after_remeshing(self):
        pass


class Dispatch:
    """A solver-free dispatch with the pinned source's hook ordering."""
    def __init__(self):
        self.hooks = []
        self.max_newton_iterations = 40
        self.newton_solver_tolerance = 1e-8
        self.newton_relaxation_factor = 1.0

    def __iadd__(self, hook):
        self.hooks.append(hook)
        return self

    def actions_before_newton_solve(self):
        for hook in self.hooks:
            hook.actions_before_newton_solve()

    def completed_step(self):
        for hook in self.hooks:
            hook.actions_after_newton_step()

    def before_remesh(self):
        for hook in self.hooks:
            hook.actions_before_remeshing([])

    def load_notification(self):
        for hook in self.hooks:
            hook.actions_after_remeshing()


def test_campaign_guard_runs_after_the_unchanged_local_allowance():
    budget = approach.PhaseBudget()
    budget.updates = 150
    problem = Dispatch()

    class LocalPolish(Hooks):
        completed = 0

        def actions_before_newton_solve(self):
            problem.max_newton_iterations = 40

        def actions_after_newton_step(self):
            self.completed += 1

    local = LocalPolish()
    original_function = problem.actions_before_newton_solve.__func__
    with approach.install_phase_guard(problem, budget, Hooks):
        problem += local  # The real helper attaches its local Budget afterwards.
        problem.actions_before_newton_solve()
        assert problem.max_newton_iterations == 10
        for _ in range(problem.max_newton_iterations):
            problem.completed_step()
        assert local.completed == 10 and budget.updates == 160
        with pytest.raises((ValueError, RuntimeError), match="exhausted"):
            problem.actions_before_newton_solve()
        assert budget.updates == 160 and local.completed == 10
        assert problem.newton_solver_tolerance == 1e-8
        assert problem.newton_relaxation_factor == 1.0
    assert "actions_before_newton_solve" not in problem.__dict__
    assert problem.actions_before_newton_solve.__func__ is original_function
    problem.completed_step()
    assert budget.updates == 160  # Detached guard cannot debit a later notification.


def test_shared_remesh_start_cap_ignores_load_notifications():
    budget, problem = approach.PhaseBudget(), Dispatch()
    with approach.install_phase_guard(problem, budget, Hooks):
        for _ in range(3):
            problem.load_notification()
        assert budget.remeshes == 0
        for _ in range(3):
            problem.before_remesh()
            problem.load_notification()
        with pytest.raises((ValueError, RuntimeError), match="exhausted"):
            problem.before_remesh()
    assert budget.remeshes == 3
    assert budget.receipt()["physical_remesh_calls_started"] == 3


def test_eigen_call_cap_is_not_reset_between_controls():
    budget = approach.PhaseBudget()
    for _ in range(6):
        budget.start_search()
    with pytest.raises((ValueError, RuntimeError), match="exhausted"):
        budget.start_search()
    assert budget.calls == 6


def test_missing_native_before_solve_dispatch_refuses_on_first_completed_step():
    budget = approach.PhaseBudget()
    with pytest.raises((ValueError, RuntimeError), match="not dispatched"):
        budget.after_step()
    assert budget.updates == 1


@pytest.mark.parametrize("refused_index", [1, 2, 3])
def test_early_refusal_retains_accepted_prefix_without_retry(tmp_path, monkeypatch, refused_index):
    output = tmp_path / "software-phase"
    output.mkdir()
    attempted = []
    sealed = []

    def qualify(prepared, previous, incoming, pins, index, output, argv, budget, deadline):
        attempted.append(index)
        budget.before_solve_calls += 1
        for _ in range(4):
            budget.after_step()
        if index == refused_index:
            failure = output / f"software-failed-control-{index}.json"
            approach.write_json(failure, {"software_only": True, "error": "fixed-Ca Newton refusal"})
            raise RuntimeError("fixed-Ca Newton refusal is not a fold certificate")
        stage = output / "stages" / f"ca_{index:03d}"
        stage.mkdir(parents=True)
        approach.write_json(stage / "completion.json", {"software_only": True, "index": index})
        record = approach.regular_record(stage / "completion.json")
        sealed.append(record)
        return previous, incoming, pins, record

    def spectrum(prepared, record, output, index, budget, deadline):
        budget.start_search()
        budget.start_search()
        path = output / f"software-spectrum-{index}.json"
        approach.write_json(path, {"software_only": True, "calls": 2})
        return approach.regular_record(path)

    monkeypatch.setattr(approach, "qualify_stage", qualify)
    monkeypatch.setattr(approach, "spectra_for_stage", spectrum)
    prepared = {"parent": {}, "restart": {}, "parent_cold": {"actual_pins": {}}, "records": [],
                "execution_authorized": True}
    result = approach.run_stages(prepared, output, [], approach.PhaseBudget(),
                                 datetime.now(UTC) + timedelta(seconds=60))
    assert attempted == list(range(1, refused_index + 1))
    assert result["status"] == "inconclusive"
    assert result["stationary_stages_accepted"] == refused_index - 1
    assert result["spectral_searches_completed"] == 2 * (refused_index - 1)
    assert result["work"]["completed_newton_updates"] == 4 * refused_index
    assert result["work"]["eigenproblem_calls_started"] == 2 * (refused_index - 1)
    assert result["terminal_refusal"]["ca_in"] == approach.LADDER[refused_index - 1]
    assert result["fold_location_certified"] is result["fold_bracket_certified"] is False
    assert result["parents_unchanged"] is True
    assert all(approach.regular_record(item["path"]) == item for item in sealed)
    assert all(json.loads((output / f"accepted-stage-{index:03d}.json").read_text()) == item
               for index, item in enumerate(sealed, 1))


def test_exact_selector_patch_leaves_all_numerical_functions_unchanged():
    original = approach.diagnostic.git(approach.ROOT, "show", f"{approach.BASE}:runners/run_paper_modes.py")
    actual = (approach.ROOT / "runners/run_paper_modes.py").read_bytes()
    assert actual == approach.routed_modes_source(original)
    functions = lambda value: {node.name: ast.dump(node, include_attributes=False)
                               for node in ast.parse(value).body if isinstance(node, ast.FunctionDef)}
    before, after = functions(original), functions(actual)
    assert set(before) == set(after)
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {"request_contract", "preflight", "parent_load_contract",
                       "restore_parent", "selected_modes", "main"}
    for name in ("spectral_stage", "compare_searches", "diagnose_pair", "take_state", "restore_state"):
        assert before[name] == after[name]


def test_approach_mode_selector_requires_execution_seal_and_completed_polish(cases):
    case = approach.iii.anchor.stage_case(cases[1], approach.LADDER[0], 1, False)
    prepared = {"case": case, "request": modes.request_contract(case, modes.SHIFTS, 8, approach_iii=True),
                "restart": {"schema": approach.runner.RESUME_SCHEMA}, "execution_authorized": True}
    assert modes.selected_modes(prepared) == (0,)
    prepared["execution_authorized"] = False
    with pytest.raises((ValueError, RuntimeError), match="execution seal"):
        modes.selected_modes(prepared)
    prepared["execution_authorized"] = True
    prepared["restart"]["schema"] = approach.coarse.PACKET_SCHEMA
    with pytest.raises((ValueError, RuntimeError), match="completed-polish"):
        modes.selected_modes(prepared)


def test_approach_requests_use_the_existing_preserved_history_loader(cases, tmp_path, monkeypatch):
    case = approach.iii.anchor.stage_case(cases[1], approach.LADDER[0], 1, False)
    prepared = {"case": case, "request": approach.request_contract(case, modes.SHIFTS, 8),
                "restart": {}, "cold": {"actual_pins": {"sha256": "software-pins"},
                "loaded": {"serialized_state": {"payload_sha256": "software-history"}}}}
    calls = []
    monkeypatch.setattr(modes.anchor_iii, "restore_checked", lambda *args: calls.append(args) or {"software_only": True})
    monkeypatch.setattr(modes.anchor, "restore_checked", lambda *args: pytest.fail("plain loader selected"))
    problem = Dispatch()
    assert modes.restore_parent(problem, prepared, tmp_path) == {"software_only": True}
    assert calls[0][1] == case and calls[0][-1] == "software-history"
    assert problem._stationary_iii_receipts_directory == tmp_path / "stationary-loads"


def test_untyped_producer_is_refused_without_native(tmp_path):
    parent = {"runner": {"path": "untyped.py"}, "anchor_stage": {"index": 1},
              "qualification_schema": approach.STAGE_SCHEMA}
    with pytest.raises((ValueError, RuntimeError), match="producer required"):
        approach.admit_stage(tmp_path, parent, {}, {"physics": {"ca_in": approach.LADDER[0]}})


def test_metadata_admission_cannot_run_native_stages(tmp_path):
    with pytest.raises((ValueError, RuntimeError), match="cannot authorise"):
        approach.run_stages({"execution_authorized": False}, tmp_path, [], approach.PhaseBudget(),
                            datetime.now(UTC) + timedelta(seconds=60))
    assert not list(tmp_path.iterdir())
