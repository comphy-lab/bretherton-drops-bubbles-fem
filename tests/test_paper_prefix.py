"""Software contracts only; synthetic fixtures never count as native evidence."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_paper_cold_history as byte_fixtures
import test_paper_model as model_fixtures

from runners import run_paper_anchor_i_prefix as continuation
from runners import run_paper_prefix_qualify as prefix

ROOT = Path(__file__).resolve().parents[1]


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"pyoomph", "_pyoomph"}:
            raise AssertionError("native imports forbidden in prefix software tests")


@pytest.fixture(autouse=True)
def deny_native_imports():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def test_standalone_imports_do_not_load_native(tmp_path):
    code = """
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyoomph', '_pyoomph'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
from runners import run_paper_prefix_qualify, run_paper_anchor_i_prefix
assert not {'pyoomph', '_pyoomph'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_production_check_retains_clean_committed_source_gate():
    revision = prefix.runner.component_revision(ROOT)
    if not revision["clean"]:
        with pytest.raises(ValueError, match="clean, committed"):
            prefix.runner.require_clean_source(revision)


def log_bytes():
    return (
        b"older stage\nSTATIONARY SOLVE. Parameters: Bo=1.56, Ca_in=-0.002, azimuthal_m=0.0\n"
        b"\nInitial Maximum residuals 0.001\n"
        b"Time to set up CRDoubleMatrix Jacobian : 1s\n"
        b"Time for linear solver (ndof=10): 2s\n"
        b"Newton Step 1: Maximum residuals 0.004\n"
        b"Time to set up CRDoubleMatrix Jacobian : 1s\n"
    )


def test_charge_counts_applied_and_conservative_inflight_without_time():
    raw = log_bytes()
    charge = prefix.derive_charge(raw)
    assert charge["charged_updates"] == 2
    assert charge["charged_remeshes"] == 0
    assert charge["window_end"] == len(raw)
    assert charge["window_sha256"] == hashlib.sha256(raw[charge["window_start"] :]).hexdigest()
    assert prefix.derive_charge(raw.replace(b"2s", b"999999s"))["charged_updates"] == 2


@pytest.mark.parametrize(
    "fault",
    [
        "extra_solve",
        "extra_update",
        "remesh",
        "inversion",
        "terminal",
        "parameters",
        "unfinished_tail",
        "missing_update",
    ],
)
def test_uncertified_charge_is_refused(fault):
    raw = log_bytes()
    if fault == "extra_solve":
        raw += raw
    elif fault == "extra_update":
        raw += b"Newton Step 2: Maximum residuals 0.1\n"
    elif fault == "remesh":
        raw += b"REMESHING\n"
    elif fault == "inversion":
        raw += b"INVERTED ELEMENT\n"
    elif fault == "terminal":
        raw += b"Reached max. number of iterations\n"
    elif fault == "parameters":
        raw = raw.replace(b"Bo=1.56", b"Bo=1.57")
    elif fault == "unfinished_tail":
        raw += b"Time for linear solver (ndof=10): 2s\n"
    else:
        raw = raw.replace(b"Newton Step 1: Maximum residuals 0.004\n", b"")
    with pytest.raises(ValueError):
        prefix.derive_charge(raw)


def test_bound_files_refuse_alias_drift_and_sizes(tmp_path):
    original = tmp_path / "original.json"
    original.write_text("{}")
    record = prefix.regular_record(original)
    alias = tmp_path / "alias.json"
    alias.symlink_to(original)
    with pytest.raises(ValueError, match="regular"):
        prefix.regular_record(alias)
    with pytest.raises(ValueError):
        prefix.bound_record({**record, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="size"):
        prefix.bound_record({**record, "bytes": 10})
    assert prefix.unique_records([record, record]) == [record]


def test_completion_binds_raw_load_proofs_and_rejects_symlink(tmp_path):
    (tmp_path / "typed-loads").mkdir()
    (tmp_path / "typed-loads/before.dump").write_bytes(b"software-only")
    (tmp_path / "progress.jsonl").write_text("{}\n")
    assert set(prefix.completion_inventory(tmp_path, exclude_solver=True)) == {
        "typed-loads/before.dump"
    }
    (tmp_path / "alias.dump").symlink_to(tmp_path / "typed-loads/before.dump")
    with pytest.raises(ValueError, match="symlink"):
        prefix.completion_inventory(tmp_path)


@pytest.fixture
def pinned_dispatch(monkeypatch):
    """Compile only pinned Python definitions, with no package/native import."""
    distribution = importlib.metadata.distribution("pyoomph")
    path = Path(distribution.locate_file("pyoomph/generic/problem.py"))
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "b75d6a810daddb3e8f0853c60fea780b33220c6c09cf819e5f219f5af056018f"
    )
    tree = ast.parse(path.read_text())
    hook = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GenericProblemHooks"
    )
    problem = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Problem"
    )
    names = {
        "_solve_with_inversion_remesh",
        "_perform_pending_remesh",
        "actions_before_newton_solve",
        "actions_after_newton_step",
        "actions_after_newton_solve",
        "last_newton_step_failed",
        "actions_before_remeshing",
        "actions_after_remeshing",
    }
    methods = [
        node for node in problem.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    standalone = ast.ClassDef(
        name="Problem", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            hook,
            standalone,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "InvertedElementRemeshRequest": type("InvertedElementRemeshRequest", (RuntimeError,), {}),
        "MeshFromTemplate1d": type("MeshFromTemplate1d", (), {}),
        "MeshFromTemplate2d": type("MeshFromTemplate2d", (), {}),
        "MeshFromTemplate3d": type("MeshFromTemplate3d", (), {}),
        "InterfaceMesh": type("InterfaceMesh", (), {}),
        "ODEStorageMesh": type("ODEStorageMesh", (), {}),
        "MeshFromTemplateBase": type("MeshFromTemplateBase", (), {}),
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - exact hash-bound pinned definitions, no imports
    native_free = types.ModuleType("pyoomph.generic.problem")
    native_free.Problem = namespace["Problem"]
    native_free.GenericProblemHooks = namespace["GenericProblemHooks"]
    monkeypatch.setitem(sys.modules, "pyoomph", types.ModuleType("pyoomph"))
    monkeypatch.setitem(sys.modules, "pyoomph.generic", types.ModuleType("pyoomph.generic"))
    monkeypatch.setitem(sys.modules, "pyoomph.generic.problem", native_free)
    fake = model_fixtures.confined_polish_standin.__wrapped__(monkeypatch)
    original_init = fake.__init__

    def initialise_fake(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._equation_system._after_remeshing = lambda: None

    def force_remesh(self, domains):
        assert domains == self._domains_remesh_on_inversion
        assert self.fields == [1.0, 2.0]
        self.actions_before_remeshing(["shared_liquid_gas_template"])
        self.events.append("remesh_both_phases")
        self.residual = 1e-3
        self.actions_after_remeshing()

    fake.__init__ = initialise_fake
    fake.actions_before_remeshing = native_free.Problem.actions_before_remeshing
    fake.actions_after_remeshing = native_free.Problem.actions_after_remeshing
    fake.reapply_boundary_conditions = lambda self: None
    fake._sync_static_condensation_rules = lambda self: None
    fake.force_remesh = force_remesh
    return fake, native_free


def finite_case():
    return prefix.model.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")


def restorable_problem_state(problem):
    return copy.deepcopy(
        {
            "dofs": problem.get_current_dofs(),
            "interface": problem.interface_polyline(),
            "parameters": {name: value.value for name, value in problem.parameters.items()},
            "volume": problem.observable("volume", "gas"),
            "frame_speed": problem.frame_speed(),
            "residual": problem.residual,
            "flags": {
                name: getattr(problem, name)
                for name in (
                    "_last_bc_setting",
                    "_first_step",
                    "_taken_already_an_unsteady_step",
                    "_last_step_was_stationary",
                )
            },
            "steppers_steady": [
                problem.time_stepper_pt(index).is_steady()
                for index in range(problem.ntime_stepper())
            ],
            "settings": {
                name: getattr(problem, name)
                for name in (
                    "max_newton_iterations",
                    "newton_relaxation_factor",
                    "newton_solver_tolerance",
                    "inversion_remesh_max_retries",
                    "eigen_data_in_states",
                )
            },
        }
    )


def require_inert_carry_during_rollback(problem):
    original_load = problem.load_state

    def load(*args, **kwargs):
        carry = next(hook for hook in problem._hooks if type(hook).__name__ == "CarryBudget")
        assert carry.active is False, "callback must deactivate before rollback starts"
        original_load(*args, **kwargs)
        # The frozen helper must also restore flags/steppers after load notifications.
        problem._last_bc_setting = "software-load-notification"
        problem._first_step = True
        problem._taken_already_an_unsteady_step = False
        problem._last_step_was_stationary = True
        problem.steady = True

    problem.load_state = load


def assert_original_error_and_verified_rollback(
    problem, directory, initial, error, message, *, error_type=ValueError
):
    assert type(error) is error_type
    assert str(error) == message
    record = json.loads((directory / "polish/ca_step/polish.json").read_text())
    assert record["status"] == "failed"
    assert record["error"] == error_type.__name__ + ": " + message
    assert record["rollback_verified"] is True
    assert restorable_problem_state(problem) == initial
    assert not next(hook for hook in problem._hooks if type(hook).__name__ == "CarryBudget").active


def test_carry_debits_first_trial_and_labels_pre_inner_counts(
    pinned_dispatch, tmp_path, monkeypatch
):
    fake, _ = pinned_dispatch
    problem = fake(updates_needed=2)
    fsyncs = []
    original_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (fsyncs.append(fd), original_fsync(fd))[1])
    polish, receipt = continuation.polish_with_carry(
        problem, finite_case(), tmp_path, {"updates": 3, "remeshes": 0}
    )
    assert problem.events[0] == ("trial", 1.0, 37)
    assert polish["updates_total"] == 5 and receipt["new_spent"]["updates"] == 2
    journal = [
        json.loads(line) for line in (tmp_path / "carry-budget.jsonl").read_text().splitlines()
    ]
    assert len(fsyncs) == len(journal)
    assert [row["event"] for row in journal].count("seed") == 1
    first_step = next(row for row in journal if row["event"] == "after_step")
    assert first_step["observed_budget"]["updates"] == 3
    assert first_step["independently_applied_new_updates"] == 1
    assert first_step["observation_phase"] == "pre-inner-Budget-callback"
    assert "increment follows" in first_step["applied_event"]
    assert journal[-1]["observation_phase"] == "post-helper-finally"
    assert journal[-1]["observed_budget"]["updates"] == 5 and journal[-1]["active"] is False


def test_remaining_total_157_and_later_slot_original_limits(pinned_dispatch, tmp_path):
    fake, _ = pinned_dispatch
    first = tmp_path / "first"
    first.mkdir()
    problem = fake(updates_needed=[37, 40, 40, 40], quality_remesh=3)
    polish, receipt = continuation.polish_with_carry(
        problem, finite_case(), first, {"updates": 3, "remeshes": 0}
    )
    assert polish["updates_total"] == 160 and receipt["new_spent"]["updates"] == 157
    assert polish["remeshes_total"] == 3
    problem.updates_needed = 2
    problem.quality_remesh = 0
    second = tmp_path / "second"
    second.mkdir()
    later = prefix.model.polish_confined_state(
        problem, finite_case(), second, "later", initial_relaxation=1.0
    )
    assert later["trials"][0]["remaining_updates"] == 40
    assert later["updates_total"] == 2 and later["remeshes_total"] == 0
    assert len((first / "carry-budget.jsonl").read_text().splitlines()) == 1 + 4 + 157 + 6 + 1


def test_inversion_retry_preserves_debit_and_load_notifications(pinned_dispatch, tmp_path):
    fake, _ = pinned_dispatch
    problem = fake(["inversion"], state_notifications=True)
    polish, receipt = continuation.polish_with_carry(
        problem, finite_case(), tmp_path, {"updates": 3, "remeshes": 1}
    )
    assert [row for row in problem.events if isinstance(row, tuple)] == [
        ("trial", 1.0, 37),
        ("trial", 0.5, 36),
    ]
    assert polish["updates_total"] == 6 and polish["remeshes_total"] == 2
    assert receipt["new_spent"] == {"updates": 3, "remeshes": 1}
    journal = [
        json.loads(line) for line in (tmp_path / "carry-budget.jsonl").read_text().splitlines()
    ]
    assert sum(row["event"] == "seed" for row in journal) == 1
    assert any(
        row["applied_event"] == "load/rollback notification; no physical remesh counted"
        for row in journal
    )


def test_remesh_charge_refuses_third_physical_retry(pinned_dispatch, tmp_path):
    fake, _ = pinned_dispatch
    problem = fake(["inversion"] * 3)
    with pytest.raises(ValueError, match="remesh allowance"):
        continuation.polish_with_carry(
            problem, finite_case(), tmp_path, {"updates": 3, "remeshes": 1}
        )
    assert problem.events.count("remesh_both_phases") == 2
    assert json.loads((tmp_path / "polish/ca_step/polish.json").read_text())["status"] == "failed"
    assert not next(hook for hook in problem._hooks if type(hook).__name__ == "CarryBudget").active


@pytest.mark.parametrize(
    "method", ["actions_after_newton_step", "actions_after_remeshing", "actions_before_remeshing"]
)
def test_preseed_callbacks_refuse(pinned_dispatch, tmp_path, method):
    fake, _ = pinned_dispatch
    problem = fake()
    initial = restorable_problem_state(problem)
    require_inert_carry_during_rollback(problem)

    def preseed(**kwargs):
        carry = next(hook for hook in problem._hooks if type(hook).__name__ == "CarryBudget")
        if method == "actions_before_remeshing":
            getattr(carry, method)([])
        else:
            getattr(carry, method)()

    problem.solve = preseed
    message = "carry callback preceded its single seed"
    with pytest.raises(ValueError) as caught:
        continuation.polish_with_carry(
            problem, finite_case(), tmp_path, {"updates": 3, "remeshes": 0}
        )
    assert_original_error_and_verified_rollback(problem, tmp_path, initial, caught.value, message)


@pytest.mark.parametrize("u,d", [(40, 0), (3, 3), (True, 0), (-1, 0)])
def test_exhausted_or_invalid_charge_refuses_before_solve(pinned_dispatch, tmp_path, u, d):
    fake, _ = pinned_dispatch
    problem = fake()
    with pytest.raises(ValueError):
        continuation.polish_with_carry(
            problem, finite_case(), tmp_path, {"updates": u, "remeshes": d}
        )
    assert problem.events == []


@pytest.mark.parametrize("fault", ["zero", "two", "foreign", "nonzero", "trials"])
def test_budget_identification_refuses_before_any_update(pinned_dispatch, tmp_path, fault):
    fake, base = pinned_dispatch
    problem = fake()
    initial = restorable_problem_state(problem)
    require_inert_carry_during_rollback(problem)
    solve = problem.solve

    def corrupt(**kwargs):
        budget = next(
            hook
            for hook in problem._hooks
            if type(hook).__qualname__ == "polish_confined_state.<locals>.Budget"
        )
        if fault == "zero":
            problem._hooks.remove(budget)
        elif fault == "two":
            problem._hooks.append(copy.copy(budget))
        elif fault == "foreign":
            other = type("Budget", (base.GenericProblemHooks,), {})()
            vars(other).update(vars(budget))
            problem._hooks[problem._hooks.index(budget)] = other
        elif fault == "nonzero":
            budget.updates = 1
        else:
            budget.trials.append({"unexpected": True})
        return solve(**kwargs)

    problem.solve = corrupt
    with pytest.raises(ValueError) as caught:
        continuation.polish_with_carry(
            problem, finite_case(), tmp_path, {"updates": 3, "remeshes": 0}
        )
    assert not any(isinstance(event, tuple) for event in problem.events)
    message = (
        "one newly appended native Budget required"
        if fault in {"zero", "two"}
        else "new Budget identity, order or initial state differs"
    )
    assert_original_error_and_verified_rollback(problem, tmp_path, initial, caught.value, message)


@pytest.mark.parametrize(
    "fault", ["updates_backwards", "remeshes_below_debit", "negative_mesh_updates"]
)
def test_carry_monotonic_refusal_preserves_error_and_rolls_back(pinned_dispatch, tmp_path, fault):
    fake, _ = pinned_dispatch
    problem = fake(updates_needed=3)
    initial = restorable_problem_state(problem)
    require_inert_carry_during_rollback(problem)
    solve = problem.solve

    def corrupt(**kwargs):
        carry = next(hook for hook in problem._hooks if type(hook).__name__ == "CarryBudget")
        after_step = carry.actions_after_newton_step
        calls = 0

        def step():
            nonlocal calls
            calls += 1
            if calls == 3:
                problem.fields, problem.pins = [9.0, 8.0], [1, 0]
                problem.parameters["Ca_in"].value = 99.0
                problem.volume, problem.speed, problem.residual = 7.0, 8.0, 1.0
                if fault == "updates_backwards":
                    carry.counter.updates = carry.updates
                elif fault == "remeshes_below_debit":
                    carry.counter.remeshes = carry.remeshes - 1
                else:
                    carry.counter.stage_updates = -1
            return after_step()

        carry.actions_after_newton_step = step
        return solve(**kwargs)

    problem.solve = corrupt
    message = "carried Budget counters are not monotone or bounded"
    with pytest.raises(ValueError) as caught:
        continuation.polish_with_carry(
            problem, finite_case(), tmp_path, {"updates": 3, "remeshes": 1}
        )
    assert_original_error_and_verified_rollback(problem, tmp_path, initial, caught.value, message)


def test_callback_journal_io_failure_preserves_primary_and_rollback(
    pinned_dispatch, tmp_path, monkeypatch
):
    fake, _ = pinned_dispatch
    problem = fake()
    initial = restorable_problem_state(problem)
    require_inert_carry_during_rollback(problem)
    primary = OSError("software journal fsync refused")
    secondary = OSError("software journal close also refused")
    calls = 0

    def fsync(fd):
        nonlocal calls
        calls += 1
        raise primary if calls == 1 else secondary

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError) as caught:
        continuation.polish_with_carry(
            problem, finite_case(), tmp_path, {"updates": 3, "remeshes": 0}
        )
    assert caught.value is primary
    assert_original_error_and_verified_rollback(
        problem, tmp_path, initial, caught.value, str(primary), error_type=OSError
    )
    carry = next(hook for hook in problem._hooks if type(hook).__name__ == "CarryBudget")
    assert carry.close_error is secondary


@pytest.fixture
def qualification_lane(tmp_path, monkeypatch):
    """Exercise the reviewed typed loader with deliberately synthetic opaque bytes."""
    monkeypatch.chdir(tmp_path)
    original = tmp_path / "software-original"
    original.mkdir()
    mesh = original / "software.msh"
    mesh.write_bytes(b"synthetic mesh; no FEM evidence")
    case = finite_case()
    case["physics"]["ca_in"] = -0.003
    case["case_id"] = "software-prefix-only"
    case_path = original / "case.json"
    write(case_path, case)
    state = original / "state.dump"
    state.write_bytes(byte_fixtures.dump(str(mesh), endtime=8.0))
    pins = {"sha256": "software-pins", "coverage": "software only"}
    problems = []

    class Problem:
        def __init__(self):
            self.index = len(problems)
            self.eigen_data_in_states = False
            self._run_statement_endtime = None
            self._loaded_run_statement_endtime = None
            self._suggested_next_dt = 0.125
            self.raw = state.read_bytes()
            self.initialised = False
            self.timestepper = SimpleNamespace(
                time_pt=lambda: SimpleNamespace(ndt=lambda: 3, dt=lambda i: [0.1, 0.2, 0.3][i]),
                get_num_unsteady_steps_done=lambda: 12,
            )
            problems.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def forbidden(self, *args, **kwargs):
            raise AssertionError("no native solve/remesh in fixture")

        solve = run = force_remesh = solve_eigenproblem = steady_newton_solve = (
            unsteady_newton_solve
        ) = forbidden

        def set_output_directory(self, path):
            self.output = path

        def get_output_directory(self):
            return self.output

        def set_linear_solver(self, name):
            assert name == "superlu"

        def set_c_compiler(self, name):
            assert name == "system"

        def setup_for_stability_analysis(self, **kwargs):
            assert kwargs == {"azimuthal_stability": True, "analytic_hessian": False}

        def initialise(self):
            assert all(
                getattr(self, name).__name__ == "refuse"
                for name in (
                    "solve",
                    "run",
                    "force_remesh",
                    "solve_eigenproblem",
                    "steady_newton_solve",
                    "unsteady_newton_solve",
                )
            )
            self.initialised = True

        def load_state(self, path, **kwargs):
            assert self.initialised
            self.raw = Path(path).read_bytes()
            self._loaded_run_statement_endtime = prefix.cold.serialized_metadata(self.raw)[
                "endtime"
            ]

        def _snapshot_state(self):
            import struct

            offset = prefix.cold.serialized_metadata(self.raw)["endtime_offset"]
            scalar = (
                math.nan if self._run_statement_endtime is None else self._run_statement_endtime
            )
            return self.raw[:offset] + struct.pack("<d", scalar) + self.raw[offset + 8 :]

        def get_current_time(self, **kwargs):
            return 2.5

    def signature(problem, az):
        assert az is True
        return {
            "bc_cache_marker": "init",
            "parameters": {"Bo": 1.56, "Ca_in": -0.003},
            "layout": {"ndof": 6, "mode_setup": "software-az"},
            "arrays": {"software": "unchanged"},
            "volume": 4.1887902047863905,
            "frame_speed": -0.003,
            "time": 2.5,
            "serialized_state": byte_fixtures.fingerprint(problem._snapshot_state(), tmp_path),
        }

    initial = Problem()
    problems.clear()
    initial._run_statement_endtime = 8.0
    saved_signature = signature(initial, True)
    producer_path = original / "run-manifest.json"
    producer_path.write_text("{}")
    packet_path = original / "resume.json"
    packet = {
        "schema": prefix.runner.RESUME_SCHEMA,
        "parent_manifest": prefix.regular_record(producer_path),
        "case_sha256": prefix.sha(case_path),
        "azimuthal_stability": True,
        "state": {**prefix.regular_record(state), "format": "portable_file"},
        "mesh_reference_base": str(original),
        "meshes": [{**prefix.regular_record(mesh), "stored_path": str(mesh)}],
        "stage": {
            "kind": "completed_polish",
            "bond_index": 16,
            "bond": 1.56,
            "ca_in": -0.003,
            "bc_mode": "init",
        },
        "roundtrip": {
            "verified": True,
            "after": saved_signature,
            "layout": saved_signature["layout"],
        },
    }
    write(packet_path, packet)
    restart = {
        **packet,
        "packet": prefix.regular_record(packet_path),
        "mesh_exchange": {str(mesh): str(mesh)},
    }
    metrics = {
        "stationary_residual_linf": 1e-12,
        "lower_pressure_rms": 0.0,
        "lower_flux": 1.0,
        "upper_flux": 1.0,
        "gas_volume": 4.1887902047863905,
        "surface_volume": 4.1887902047863905,
        "target_volume": 4.1887902047863905,
    }
    failure = {"instance": None, "gate": None, "sampling_mutation": False}

    def sample(problem, model, current, output, phase, array_module):
        assert array_module is np and current == case
        values = dict(metrics)
        if problem.index == failure["instance"]:
            faults = {
                "stationary": ("stationary_residual_linf", 1.0),
                "lower_static_pressure": ("lower_pressure_rms", 1.0),
                "flux": ("lower_flux", 2.0),
                "volume": ("gas_volume", 5.0),
                "volume_integrals": ("surface_volume", 5.0),
            }
            key, value = faults[failure["gate"]]
            values[key] = value
        if failure["sampling_mutation"]:
            problem.raw = problem.raw.replace(b"seven histories", b"eight histories")
        value = {
            "metrics": values,
            "actual_pins": pins,
            "signature": signature(problem, True),
            "sampling_preserved_state": True,
        }
        write(output / "samples" / (phase + ".json"), value)
        return value

    def checkpoint(problem, current, output, index, az):
        assert current == case and index == 16 and az is True
        directory = output / "checkpoints/bond_016"
        directory.mkdir(parents=True)
        new_mesh = directory / "relocated-longer-name.msh"
        new_mesh.write_bytes(mesh.read_bytes())
        new_state = directory / "state.dump"
        new_state.write_bytes(byte_fixtures.dump(str(new_mesh), endtime=8.0))
        final = {
            **copy.deepcopy(packet),
            "parent_manifest": prefix.regular_record(output / "run-manifest.json"),
            "state": {**prefix.regular_record(new_state), "format": "portable_file"},
            "mesh_reference_base": str(directory),
            "meshes": [{**prefix.regular_record(new_mesh), "stored_path": str(new_mesh)}],
        }
        final["roundtrip"]["after"]["serialized_state"] = byte_fixtures.fingerprint(
            new_state.read_bytes(), directory
        )
        write(directory / "resume.json", final)
        return {"resume_packet": "checkpoints/bond_016/resume.json"}

    def validate_packet(path, *args, **kwargs):
        value = json.loads(Path(path).read_text())
        return {
            **value,
            "packet": prefix.regular_record(path),
            "mesh_exchange": {mesh["stored_path"]: mesh["path"] for mesh in value["meshes"]},
        }

    monkeypatch.setattr(prefix.model, "build_problem", lambda *a, **k: Problem())
    monkeypatch.setattr(prefix.model, "_stationary_state_signature", signature)
    monkeypatch.setattr(prefix.model, "_prepare_stationary_bc_state", lambda *a: {"verified": True})
    monkeypatch.setattr(prefix.diagnostic, "actual_pins", lambda p: pins)
    monkeypatch.setattr(prefix.remesh, "sample", sample)
    monkeypatch.setattr(prefix.model, "_checkpoint_stationary_state", checkpoint)
    monkeypatch.setattr(prefix.runner, "validate_restart", validate_packet)
    prepared = {
        "case": case,
        "restart": restart,
        "sample": {"actual_pins": pins},
        "metadata": prefix.cold.serialized_metadata(state.read_bytes()),
        "records": [
            prefix.regular_record(case_path),
            prefix.regular_record(state),
            prefix.regular_record(mesh),
        ],
        "component": {"commit": "1" * 40},
        "environment": {},
        "implementation": {prefix.ENTRYPOINT: prefix.regular_record(ROOT / prefix.ENTRYPOINT)},
        "lineage": {"schema": prefix.SCHEMA, "scope": "software-only prefix", "consumer": {}},
        "original_runtime": str(original),
        "custody_root": str(tmp_path / "software-archive-not-local"),
    }
    monkeypatch.setattr(
        prefix, "stable_inputs", lambda p: prefix.remesh.verify_inputs(p["records"])
    )
    output = tmp_path / "software-qualified"
    args = SimpleNamespace(case=case_path)
    output.mkdir()
    write(output / "run-manifest.json", {"software_only": True})
    return SimpleNamespace(
        args=args,
        prepared=prepared,
        output=output,
        problems=problems,
        failure=failure,
        restart=restart,
        state=state,
        mesh=mesh,
        case=case,
        pins=pins,
    )


def test_prefix_two_fresh_typed_loads_and_portable_same_state(qualification_lane):
    b = qualification_lane
    original = b.state.read_bytes()
    result = prefix.execute(b.args, b.prepared, b.output)
    assert len(b.problems) == 2 and b.problems[0] is not b.problems[1]
    assert all(
        problem.initialised
        and problem._loaded_run_statement_endtime == problem._run_statement_endtime == 8.0
        for problem in b.problems
    )
    assert result["stationary_prefix_accepted"] is True and result["anchor_accepted"] is False
    assert result["original_acceptance_changed"] is False
    assert all(result[key] is False for key in prefix.NO_CLAIMS)
    assert b.state.read_bytes() == original
    packet = json.loads((b.output / result["checkpoint"]["resume_packet"]).read_text())
    final = Path(packet["state"]["path"])
    assert prefix.cold.normalized_bytes(
        final.read_bytes(), final.parent
    ) == prefix.cold.normalized_bytes(original, b.state.parent)
    inventory = prefix.completion_inventory(b.output)
    assert any(
        name.startswith("typed-loads/") and name.endswith("before.dump") for name in inventory
    )
    assert any(
        name.startswith("cold-typed-loads/") and name.endswith("after.dump") for name in inventory
    )


@pytest.mark.parametrize("instance", [0, 1])
@pytest.mark.parametrize("gate", sorted(prefix.GATES))
def test_each_original_warm_or_cold_gate_refuses(qualification_lane, instance, gate):
    b = qualification_lane
    b.failure.update(instance=instance, gate=gate)
    with pytest.raises(ValueError, match="five stationary gates"):
        prefix.execute(b.args, b.prepared, b.output)
    assert not (b.output / "completion.json").exists()
    assert (b.output / "typed-loads").is_dir()


def test_sampling_cannot_change_full_history(qualification_lane):
    b = qualification_lane
    b.failure["sampling_mutation"] = True
    with pytest.raises(ValueError, match="sampling changed complete"):
        prefix.execute(b.args, b.prepared, b.output)
    assert not (b.output / "completion.json").exists()


def test_zero_solve_guard_is_installed_before_initialise(qualification_lane):
    b = qualification_lane
    problem = prefix.model.build_problem(b.case, initial_bond=1.56)
    prefix.configure_guarded(problem, b.output / "solver-extra", b.output / "extra-loads")
    assert problem.initialised
    for name in (
        "solve",
        "run",
        "force_remesh",
        "solve_eigenproblem",
        "steady_newton_solve",
        "unsteady_newton_solve",
    ):
        with pytest.raises(RuntimeError, match="zero-solve"):
            getattr(problem, name)()


def test_native_initialisation_violation_cannot_be_accepted(qualification_lane, monkeypatch):
    b = qualification_lane
    configure = prefix.requalification.configure

    def violate(problem, directory):
        problem.solve()
        configure(problem, directory)

    monkeypatch.setattr(prefix.requalification, "configure", violate)
    with pytest.raises(RuntimeError, match="refused solve"):
        prefix.execute(b.args, b.prepared, b.output)
    assert not (b.output / "completion.json").exists()


def test_missing_terminal_budget_never_becomes_admission(tmp_path):
    with pytest.raises(ValueError, match="budget witness required"):
        prefix.bind_budget(None, tmp_path, {}, {})


@pytest.mark.parametrize("relative", ["../state.dump", "/state.dump", "", "."])
def test_unsafe_completion_and_lineage_paths_refuse(relative):
    with pytest.raises(ValueError, match="relative path"):
        prefix.relative_path(relative)


def test_current_typed_I_gate_refuses_new_prefix_producer(tmp_path):
    source = (
        ROOT.parent / "bretherton-drops-bubbles-fem-paper-spectrum-typed/runners/run_paper_modes.py"
    )
    if not source.is_file():
        pytest.skip("isolated typed-spectrum source not materialised")
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "qualified_i_parent"
    )
    namespace = {"require": prefix.require, "json": json, "Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - bounded repository-owned admission definition
    write(tmp_path / "result.json", {})
    write(tmp_path / "preparation.json", {})
    parent = {
        "schema": "pyoomph-runner-manifest-v1",
        "runner": {"path": "runners/run_paper_anchor_i_prefix.py"},
        "anchor_stage": {"kind": "stationary_I_candidate_from_prefix"},
    }
    with pytest.raises(ValueError, match="producer/schema"):
        namespace["qualified_i_parent"](
            tmp_path,
            {},
            parent,
            {},
            {},
            {},
            {"result.json": "software", "preparation.json": "software"},
        )


@pytest.fixture
def completed_qualification(qualification_lane, tmp_path, monkeypatch):
    """Produce a synthetic completed-prefix record for consumer refusal tests."""
    b = qualification_lane
    original = Path(b.prepared["original_runtime"])
    metadata_files = {}
    for name in ("stage.json", "terminal.json", "custody.json", "ladder.json", "budget.json"):
        write(original / name, {"software_only": True})
        metadata_files[name] = prefix.regular_record(original / name)
    target = copy.deepcopy(b.case)
    target["case_id"] = "software-target-only"
    target["physics"]["ca_in"] = -0.0016
    target_path = original / "target.json"
    write(target_path, target)
    b.prepared["target"] = target
    b.prepared["component"] = {
        "head": "1" * 40,
        "commit": "1" * 40,
        "clean": True,
        "status": [],
        "sources": {},
        "declared_commit": None,
    }
    b.prepared["implementation"] = {
        relative: prefix.regular_record(ROOT / relative)
        for relative in (prefix.ENTRYPOINT, continuation.ENTRYPOINT)
    }
    selected = {
        "case": prefix.regular_record(b.args.case),
        "stage": metadata_files["stage.json"],
        "manifest": b.restart["parent_manifest"],
        "packet": b.restart["packet"],
        "state": b.restart["state"],
    }
    b.prepared["lineage"].update(
        original_attempt={
            "manifest": b.restart["parent_manifest"],
            "terminal": metadata_files["terminal.json"],
            "custody": metadata_files["custody.json"],
        },
        completed_prefix=[selected],
        remaining_ladder={
            "original": metadata_files["ladder.json"],
            "target": prefix.regular_record(target_path),
        },
        budget={
            "witness": metadata_files["budget.json"],
            "charge": {"charged_updates": 2, "charged_remeshes": 0},
        },
    )
    output = tmp_path / "software-completed-qualification"
    args = {
        "source": ROOT,
        "case": b.args.case,
        "original_attempt_terminal": original / "terminal.json",
        "original_manifest": original / "run-manifest.json",
        "stage": original / "stage.json",
        "manifest": original / "run-manifest.json",
        "packet": original / "resume.json",
        "state": b.state,
        "original_ca_ladder": original / "ladder.json",
        "custody": original / "custody.json",
        "budget": original / "budget.json",
        "out": output,
    }
    monkeypatch.setattr(prefix, "preflight", lambda args: b.prepared)
    arguments = [
        item for key, value in args.items() for item in ("--" + key.replace("_", "-"), str(value))
    ]
    assert prefix.main(arguments) == 0
    monkeypatch.setattr(
        prefix,
        "source_contract",
        lambda parent, entry: (
            b.prepared["component"],
            b.prepared["environment"],
            b.prepared["implementation"],
        ),
    )
    monkeypatch.setattr(prefix.diagnostic, "git", lambda *args: b"")
    b.consumer_args = SimpleNamespace(
        source=ROOT,
        case=target_path,
        manifest=output / "run-manifest.json",
        packet=output / "checkpoints/bond_016/resume.json",
        state=output / "checkpoints/bond_016/state.dump",
        out=tmp_path / "software-anchor-output",
    )
    b.completed_output = output
    return b


def resign_completion(runtime):
    path = runtime / "completion.json"
    value = json.loads(path.read_text())
    value["files"] = prefix.completion_inventory(runtime)
    write(path, value)


def test_completed_prefix_admission_has_exact_suffix_and_debited_caps(completed_qualification):
    b = completed_qualification
    prepared = continuation.preflight(b.consumer_args)
    assert prepared["contract"]["values"] == [-0.002, -0.0016]
    assert prepared["contract"]["original_indices"] == [4, 5]
    assert prepared["contract"]["per_mesh_stage_newton_cap_first_resumed"] == 38
    assert prepared["contract"]["per_polish_total_newton_cap_first_resumed"] == 158
    assert prepared["contract"]["per_polish_remesh_bound_first_resumed"] == 3
    assert prepared["contract"]["total_newton_update_bound"] == 478
    assert prepared["contract"]["closed_stage_allowance_transferable"] is False
    assert not b.consumer_args.out.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "producer",
        "intervention",
        "expanded_claim",
        "cold_gate",
        "pins",
        "history",
        "bc",
        "guard",
        "target",
        "original_acceptance",
    ],
)
def test_prefix_consumer_refuses_forged_completed_evidence(completed_qualification, fault):
    b = completed_qualification
    runtime = b.completed_output
    if fault in {"producer", "intervention", "original_acceptance"}:
        path = runtime / "run-manifest.json"
        value = json.loads(path.read_text())
        if fault == "producer":
            value["runner"]["path"] = "runners/run_paper_cold_history.py"
        elif fault == "intervention":
            value["intervention"]["solve_calls"] = 1
        else:
            value["original_acceptance_changed"] = True
    elif fault in {"expanded_claim", "cold_gate"}:
        path = runtime / "result.json"
        value = json.loads(path.read_text())
        if fault == "expanded_claim":
            value["branch_identity_verified"] = True
        else:
            value["cold_load_checks"]["stationary"] = False
    elif fault in {"pins", "history", "bc"}:
        path = runtime / "cold-load.json"
        value = json.loads(path.read_text())
        if fault == "pins":
            value["actual_pins"]["sha256"] = "changed software pins"
        elif fault == "history":
            value["time_history"]["dts"] = [0.1, 0.2, 9.0]
        else:
            value["bc_preparation"]["verified"] = False
    elif fault == "guard":
        path = runtime / "second-zero-solve-guard.json"
        value = json.loads(path.read_text())
        value["installed_before_initialise"] = False
    else:
        path = b.consumer_args.case
        value = json.loads(path.read_text())
        value["mesh"]["bulk_resolution"] /= 2
    write(path, value)
    resign_completion(runtime)
    with pytest.raises(ValueError):
        continuation.preflight(b.consumer_args)
    assert not b.consumer_args.out.exists()


@pytest.fixture
def terminal_bundle(tmp_path):
    attempt = tmp_path / "paper-model-anchor-i-medium/attempts/attempt-0001"
    runtime = attempt / "runtime"
    runtime.mkdir(parents=True)
    events = [
        {"MESSAGE": "Service reached runtime time limit. Stopping."},
        {"MESSAGE": "Failed with result 'timeout'."},
    ]
    journal = attempt / "execution-terminal-journal.jsonl"
    journal.write_text("".join(json.dumps(event) + "\n" for event in events))
    value = {
        "schema": "stone-run-execution-terminal-v1",
        "status": "failed",
        "scientific_result_accepted": False,
        "original_deadline_extended": False,
        "unit": "stone-paper-model-anchor-i-medium-attempt-0001.service",
        "live_run_root_candidates": [],
        "selected_prefix_index": 3,
        "unfinished_stage_index": 4,
        "native_failure_receipt_present": False,
        "native_completion_receipt_present": False,
        "supervisor_result_present": False,
        "recorded_pids_absent": [11, 12],
        "terminal_journal_events": events,
        "journal_sha256": prefix.sha(journal),
    }
    path = attempt / "execution-terminal.json"
    write(path, value)
    return path, runtime, value


def test_truthful_timeout_receipt_preserves_absent_native_failure(terminal_bundle):
    path, runtime, value = terminal_bundle
    actual, records = prefix.bind_terminal(path, runtime)
    assert actual == value and len(records) == 2
    assert not (runtime / "failure.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("scientific_result_accepted", True),
        ("original_deadline_extended", True),
        ("native_failure_receipt_present", True),
        ("native_completion_receipt_present", True),
        ("supervisor_result_present", True),
        ("selected_prefix_index", 4),
        ("unfinished_stage_index", 5),
        ("live_run_root_candidates", [11]),
        ("recorded_pids_absent", []),
    ],
)
def test_unreconciled_or_relabelled_original_refuses(terminal_bundle, field, value):
    path, runtime, receipt = terminal_bundle
    receipt[field] = value
    write(path, receipt)
    with pytest.raises(ValueError, match="truthful terminal"):
        prefix.bind_terminal(path, runtime)


def test_remote_custody_receipt_rechecks_full_retained_source_without_local_mount(tmp_path):
    run_root = tmp_path / "software-retained-tree"
    run_root.mkdir()
    files = {}
    for name in ("AGENTS.md", "CLAUDE.md", "README.md"):
        path = run_root / name
        path.write_text("software-only protocol fixture\n")
        files[name] = {"sha256": prefix.sha(path), "bytes": path.stat().st_size}
    receipt = {
        "schema": "figure10-next-phase-archive-v1",
        "simulation": "paper-model-anchor-i-medium",
        "attempt": "attempt-0001",
        "source": str(run_root),
        "destination": "/nonexistent-software-remote/archive",
        "source_deleted": False,
        "verified_at": "2026-10-12T13:36:18Z",
        "protocol_files_verified_at": "2026-10-12T13:36:18Z",
        "file_count": 3,
        "bytes": sum(file["bytes"] for file in files.values()),
        "files": files,
    }
    path = tmp_path / "software-custody.json"
    write(path, receipt)
    _, records = prefix.bind_custody(path, run_root)
    assert len(records) == 4
    (run_root / "extra.dump").write_bytes(b"unbound")
    with pytest.raises(ValueError, match="inventory"):
        prefix.bind_custody(path, run_root)
