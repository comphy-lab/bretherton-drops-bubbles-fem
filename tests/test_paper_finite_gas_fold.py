"""Offline contracts for typed fold admission, callbacks and exact restoration."""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.abc
import json
import sys
import weakref
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from runners import run_paper_finite_gas_fold as fold


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith(("pyoomph.", "_pyoomph")):
            raise AssertionError("native import denied in fold software tests")


@pytest.fixture(autouse=True)
def no_native():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def bank_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(fold, "N", 3)
    types = np.array([0, 1, 2], dtype="<i8")
    names = ["liquid/mesh_x", "globals/U", "gas_volume/pressure"]
    digest = fold.hashlib.sha256(json.dumps(names, separators=(",", ":")).encode() + types.tobytes()).hexdigest()
    arrays = {"free_dofs": {"sha256": "software-fields"}, "positional_dof_mask": {"sha256": "software-mask"},
              "pinned_values_with_positions": {"sha256": "software-pins"}}
    expected = {"arrays": arrays, "layout": {"dof_layout_sha256": digest}, "parameters": {"Ca_in": fold.CA, "Bo": 1.56},
                "time": 8., "frame_speed": -.004, "volume": 4.2}
    for index, shift in enumerate(fold.LIMITS["shifts"]):
        directory = tmp_path / f"m0-shift-{index}"
        directory.mkdir()
        write(directory / "layout.json", {"types": types.tolist(), "names": names,
              "globals": {"globals/U": [1], "gas_volume/pressure": [2]}, "forced": []})
        write(directory / "base-before.json", {**{key: value for key, value in expected.items() if key not in {"arrays", "layout"}},
              "arrays": {modern: arrays[old] for modern, old in
                         (("free_values", "free_dofs"), ("positional_mask", "positional_dof_mask"),
                          ("pinned_values", "pinned_values_with_positions"))}})
        write(directory / "status.json", {"mode": 0, "shift": shift, "base_restoration_verified": True})
        np.savez(directory / "eigenpairs.npz", values=np.zeros(8, dtype=complex), vectors=np.ones((8, 3), dtype=complex),
                 modes=np.zeros(8, dtype=np.int32), types=types, names=np.asarray(names), forced=np.asarray([], dtype=np.int64))
    return tmp_path, {"roundtrip": {"after": expected}}


def test_own_bank_layout_and_fields_are_required(bank_bundle):
    banks, records = fold.bank_contract(*bank_bundle)
    assert len(banks) == 2 and len(records) == 8
    assert [row["shift"] for row in banks] == [-.001, .001]


@pytest.mark.parametrize("file,key,value,label", [
    ("base-before.json", "parameters", {"Ca_in": -.00382, "Bo": 1.56}, "bank_scalar"),
    ("base-before.json", "time", 9, "bank_scalar"),
    ("status.json", "mode", 1, "bank_mode"),
    ("status.json", "shift", 0, "bank_mode"),
    ("layout.json", "types", [1, 0, 2], "bank_full_layout"),
    ("layout.json", "names", ["gas/mesh_x", "globals/U", "gas_volume/pressure"], "bank_full_layout"),
])
def test_bank_tampering_is_refused(bank_bundle, file, key, value, label):
    directory, restart = bank_bundle
    path = directory / "m0-shift-0" / file
    data = json.loads(path.read_text())
    data[key] = value
    write(path, data)
    with pytest.raises(fold.Refusal, match=label):
        fold.bank_contract(directory, restart)


def test_bank_cannot_borrow_a_vector_with_foreign_incidence(bank_bundle):
    directory, restart = bank_bundle
    path = directory / "m0-shift-0/eigenpairs.npz"
    with np.load(path, allow_pickle=False) as bank:
        data = {name: bank[name].copy() for name in bank.files}
    data["types"] = np.array([2, 1, 0])
    np.savez(path, **data)
    with pytest.raises(fold.Refusal, match="bank_array_shape"):
        fold.bank_contract(directory, restart)


def test_bank_input_mutation_is_detected_after_admission(bank_bundle):
    _banks, records = fold.bank_contract(*bank_bundle)
    with open(records[0]["path"], "ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="parent inputs changed"):
        fold.remesh.verify_inputs(records)


@pytest.fixture
def case():
    value = fold.model.read_case(fold.ROOT / "validation/herradaEtAl_paper_model/cases/iii-ca-m0p0038-medium.json")
    value["physics"]["ca_in"] = fold.CA
    return value


def test_measured_case_changes_only_identity_and_tracked_parameter(case):
    old = copy.deepcopy(case)
    current = fold.measured_case(case, fold.CA - 1e-5)
    assert case == old
    assert set(fold.remesh.differing_keys(case, current)) == {"case_id", "physics.ca_in"}
    assert current["tolerances"] == old["tolerances"]
    assert current["physics"]["gas_viscosity_ratio"] == old["physics"]["gas_viscosity_ratio"]


@pytest.mark.parametrize("ca", [float("nan"), float("inf"), fold.CA - 3e-5])
def test_measured_case_refuses_nonfinite_or_nonlocal_candidate(case, ca):
    with pytest.raises(fold.Refusal, match="ca_locality"):
        fold.measured_case(case, ca)


def test_fold_call_omits_unsupported_keywords_and_keeps_m0_setup():
    called = []
    problem = SimpleNamespace(_azimuthal_mode_param_m=SimpleNamespace(value=0),
                              activate_bifurcation_tracking=lambda *a, **k: called.append((a, k)))
    seed = np.array([1., 2.])
    fold.activate_fold(problem, seed)
    args, kwargs = called[0]
    assert args == ("Ca_in", "fold")
    assert kwargs == {"blocksolve": False, "eigenvector": seed, "eigenvector_scaling": "auto"}
    assert "azimuthal_mode" not in kwargs and "eigenvalue_for_branch_tracking" not in kwargs
    problem._azimuthal_mode_param_m.value = 1
    with pytest.raises(fold.Refusal, match="nonzero_fold_mode"):
        fold.activate_fold(problem, seed)


def test_captured_scientific_class_is_bridged_before_construction_once():
    events = []

    class Core:
        def actions_before_newton_step(self):
            events.append("base-before")

    Core.__module__ = "pyoomph._pyoomph_core"

    def factory():
        events.append("factory")

        class Scientific(Core):
            def __init__(self):
                events.append("constructor")
                assert fold.BEFORE_STEP in vars(type(self))

        return Scientific, object

    captured, bridge = fold.construct_type(SimpleNamespace(Problem=Core), factory)
    problem = captured()
    monitor = SimpleNamespace(before_iteration=lambda: events.append("monitor-before"))
    bridge.attach(problem, monitor)
    problem.actions_before_newton_step()
    bridge.detach(problem, monitor)
    assert events == ["factory", "constructor", "monitor-before", "base-before"]
    with pytest.raises(fold.Refusal, match="before_native_release"):
        bridge.restore_after_release(native_session_released=False)
    bridge.restore_after_release(native_session_released=True)
    assert fold.BEFORE_STEP not in vars(captured)


def test_class_bridge_refuses_a_scientific_before_step_override():
    class Core:
        def actions_before_newton_step(self):
            pass

    Core.__module__ = "pyoomph._pyoomph_core"

    class Scientific(Core):
        def actions_before_newton_step(self):
            pass

    with pytest.raises(fold.Refusal, match="already_overridden"):
        fold.BeforeStepBridge(Scientific, Core)


def test_update_budget_allows_twenty_updates_then_refuses_without_reset():
    seen = []
    adapter = SimpleNamespace(work={"completed_augmented_updates": 0}, check_newton=seen.append)
    monitor = fold.UpdateMonitor(adapter)
    monitor.before_solve()
    for _ in range(20):
        monitor.before_iteration()
        monitor.after_update()
    with pytest.raises(fold.Refusal, match="budget_or_sequence"):
        monitor.before_iteration()
    assert monitor.updates == adapter.work["completed_augmented_updates"] == 20
    with pytest.raises(fold.Refusal, match="automatic_augmented_retry"):
        monitor.before_solve()


def test_missing_before_step_dispatch_refuses_the_first_update():
    adapter = SimpleNamespace(work={}, check_newton=lambda _: None)
    monitor = fold.UpdateMonitor(adapter)
    monitor.before_solve()
    with pytest.raises(fold.Refusal, match="callback_or_update"):
        monitor.after_update()
    assert monitor.updates == adapter.work["completed_augmented_updates"] == 1


def test_zero_completed_update_has_callback_coverage_without_fabricated_work():
    seen = []
    monitor = fold.UpdateMonitor(SimpleNamespace(work={"completed_augmented_updates": 0}, check_newton=seen.append))
    monitor.before_solve()
    monitor.before_iteration()
    monitor.finalise()
    assert monitor.updates == 0 and monitor.starts == 1 and monitor.before_iterations == 1
    assert seen == ["before_solve", "before_iteration", "after_solve"]


def test_solve_cap_and_search_cap_are_fixed():
    assert fold.LIMITS["updates"] == 20
    assert fold.LIMITS["shifts"] == [-.001, .001]
    assert fold.LIMITS["neigen"] == 8
    tree = ast.parse((fold.ROOT / fold.ENTRYPOINT).read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_candidate")
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "solve"]
    assert len(calls) == 1
    assert {key.arg: ast.literal_eval(key.value) for key in calls[0].keywords} == {
        "spatial_adapt": 0, "max_newton_iterations": 20, "do_not_set_IC": True,
        "newton_relaxation_factor": 1.0, "globally_convergent_newton": False}


def test_solver_owner_cannot_be_rebound_as_restorable_state():
    problem = SimpleNamespace()

    class Solver:
        def __init__(self):
            self._problem_wr = weakref.ref(problem)

        @property
        def problem(self):
            return self._problem_wr()

    # SimpleNamespace is not weak-referenceable; use an ordinary record.
    class Problem:
        pass

    problem = Problem()
    solver = Solver()
    problem.get_eigen_solver = lambda: solver
    owner = fold.solver_owner_signature(problem)
    with pytest.raises(fold.Refusal, match="owner_in_restorable"):
        fold.restore_solver_scalars(problem, {"_problem_wr": "untrusted"}, owner)
    foreign = Problem()
    solver._problem_wr = weakref.ref(foreign)
    with pytest.raises(fold.Refusal, match="foreign_eigensolver"):
        fold.solver_owner_signature(problem)


class Continuation:
    def __init__(self):
        self.current = np.array([3., 4.])
        self.derivative = np.zeros(0)
        self.dp, self.theta = 1., 1.
        self._last_arclength_parameter = None
        self._solve_in_arclength_conti = None
        self._arclength_inner_product = "ndof"
        self.minimum_arclength_ds = 1e-6
        self.max_newton_iterations = 20
        self.newton_relaxation_factor = 1.
        self.newton_solver_tolerance = 1e-8
        self._parked_continuation_data = None

    def ndof(self):
        return 2

    def get_arclength_dof_derivative_vector(self):
        return self.derivative

    def get_arclength_dof_current_vector(self):
        return self.current

    def get_arc_length_parameter_derivative(self):
        return self.dp

    def get_arc_length_theta_sqr(self):
        return self.theta

    def _set_arc_length_parameter_derivative(self, value):
        self.dp = value

    def _set_arc_length_theta_sqr(self, value):
        self.theta = value

    def _update_dof_vectors_for_continuation(self, derivative, current):
        self.derivative, self.current = derivative.copy(), current.copy()

    def reset_arc_length_parameters(self):
        self.derivative = np.zeros(0)  # The pinned reset retains current.


@pytest.mark.parametrize("inner_product", [None, "ndof"])
def test_continuation_controls_and_vectors_restore_exactly(inner_product):
    problem = Continuation()
    problem._arclength_inner_product = inner_product
    expected = fold.capture_continuation(problem)
    assert expected["controls"]["_arclength_inner_product"] == inner_product
    assert problem._arclength_inner_product == inner_product
    problem.current[:] = -9
    problem.derivative = np.array([8., 9.])
    problem.dp, problem.theta = 7., 8.
    problem._arclength_inner_product = "ndof" if inner_product is None else None
    problem._last_arclength_parameter = "Bo"
    problem._solve_in_arclength_conti = "Bo"
    problem.minimum_arclength_ds = .25
    problem.max_newton_iterations = 37
    problem.newton_relaxation_factor = .5
    problem.newton_solver_tolerance = 1e-4
    fold.restore_continuation(problem, expected)
    assert fold.continuation_signature(fold.capture_continuation(problem)) == fold.continuation_signature(expected)
    assert problem._arclength_inner_product == inner_product


@pytest.mark.parametrize("inner_product", ["l2", "mass"])
def test_unsupported_metric_is_refused_with_its_actual_control_value(inner_product):
    problem = Continuation()
    problem._arclength_inner_product = inner_product
    with pytest.raises(fold.Refusal, match="unsupported_continuation_controls") as refusal:
        fold.capture_continuation(problem)
    assert refusal.value.evidence["_arclength_inner_product"] == inner_product
    assert problem._arclength_inner_product == inner_product


@pytest.mark.parametrize("field,value,label", [
    ("_solve_in_arclength_conti", "Ca_in", "unsupported_continuation_controls"),
    ("_last_arclength_parameter", "Bo", "unsupported_continuation_controls"),
    ("_parked_continuation_data", ([], 1., 1.), "unsupported_parked_continuation_data"),
])
def test_active_parked_and_foreign_continuation_states_remain_refused(field, value, label):
    problem = Continuation()
    problem._arclength_inner_product = None
    setattr(problem, field, value)
    with pytest.raises(fold.Refusal, match=label):
        fold.capture_continuation(problem)


def test_original_empty_continuation_cannot_be_falsely_reported_restored():
    problem = Continuation()
    problem.current = np.zeros(0)
    expected = fold.capture_continuation(problem)
    problem.current = np.array([3., 4.])
    with pytest.raises(fold.Refusal, match="current_empty_restore_unsupported"):
        fold.restore_continuation(problem, expected)


def test_physical_mesh_request_refuses_even_when_administrative_load_is_allowed():
    problem = SimpleNamespace(_remesh_requested_inside_solve=False, _domains_to_remesh=set(),
                              do_call_remeshing_when_necessary=False, _meshdict={})
    guard = fold.RemeshGuard(problem)
    with guard.administrative("software-load"):
        guard.callback("actions_after_remeshing")
    assert guard.physical_callbacks == guard.requests == 0
    with pytest.raises(fold.Refusal, match="physical_mesh_callback"):
        guard.callback("actions_after_remeshing")
    with pytest.raises(fold.Refusal, match="physical_remesh_or_adapt"):
        guard.reject("force_remesh")()


@pytest.fixture
def analytic_null(monkeypatch, case):
    monkeypatch.setattr(fold, "N", 3)
    stationary = sparse.diags([0., 2., 3.], format="csr")
    jacobian, mass = -stationary, sparse.diags([1., 0., 0.], format="csr")
    layout = {"forced": [], "globals": {"globals/U": [1], "gas_volume/pressure": [2]}}
    return np.zeros(3), stationary, jacobian, mass, np.array([1., 0., 0.]), layout, case


def test_exact_null_uses_expected_assembly_sign_and_retains_global_rows(analytic_null):
    metrics = fold.null_actions(*analytic_null)
    assert metrics["full_J_plus_Jstat_actual_nonzeros"] == 0
    assert metrics["stationary_null_passed"] is metrics["direct_dynamic_null_passed"] is True
    assert metrics["mass_action_scaled_max_entry"] == 1.
    assert set(metrics["null_pencil_diagnostic"]["constraint_actions"]) == set(fold.modes.GLOBALS)


def test_static_only_null_does_not_pass_dynamic_gate(analytic_null):
    residual, stationary, jacobian, _mass, q, layout, case = analytic_null
    mass = sparse.diags([0., 1., 0.], format="csr")
    metrics = fold.null_actions(residual, stationary, jacobian, mass, q, layout, case)
    assert metrics["stationary_null_passed"] is True
    assert metrics["direct_dynamic_null_passed"] is False


def test_returned_nearzero_requires_real_root_exact_overlap_and_constraint_gates(analytic_null):
    _r, _s, j, m, q, layout, case = analytic_null
    rows = fold.returned_null_metrics(np.array([0j, 1e-5j]), np.array([q, q]), j, m, q, layout, case)
    assert rows[0]["returned_real_null_matches"] is True
    assert rows[1]["returned_real_null_matches"] is False
    other = np.array([0., 1., 0.])
    rows = fold.returned_null_metrics(np.array([0j]), np.array([other]), j, m, q, layout, case)
    assert rows[0]["returned_real_null_matches"] is False


def test_fixed_identity_preserves_connectivity_pins_and_addresses_but_allows_values_to_move():
    node = {"values": {"equations": [0], "pins": [False], "history": "initial"},
            "positions": {"equations": [1], "pins": [False], "history": "initial"},
            "position_pins": [False], "lagrangian": "fixed"}
    meshes = {"liquid": {"nodes": [node], "elements": [], "connectivity": "unchanged"}}
    baseline = fold.fixed_identity(meshes, np.array([0, 1]), ["velocity", "mesh"], np.array([False, True]))
    node["values"]["history"] = "new solved fields"
    node["positions"]["history"] = "moved Eulerian positions"
    assert fold.fixed_identity(meshes, np.array([0, 1]), ["velocity", "mesh"], np.array([False, True])) == baseline
    node["positions"]["pins"] = [True]
    assert fold.fixed_identity(meshes, np.array([0, 1]), ["velocity", "mesh"], np.array([False, True])) != baseline


def test_metadata_cannot_authorise_native_execution(tmp_path):
    with pytest.raises(fold.Refusal, match="metadata_cannot_authorise"):
        fold.execute({"native_execution_authorized": False}, tmp_path, [], None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("pin_drift", [False, True])
@pytest.mark.parametrize("historical_mesh_violation", [False, True])
def test_same_instance_restore_verifies_pins_and_restores_controls(monkeypatch, pin_drift, historical_mesh_violation):
    class Solver:
        def __init__(self, problem):
            self._problem_wr = weakref.ref(problem)
            self.matrix_manipulators = []
            self.max_retries = 0
            self.use_pardiso = False

        @property
        def problem(self):
            return self._problem_wr()

    class Problem(Continuation):
        def __init__(self):
            super().__init__()
            self.solver = Solver(self)
            self.parameters = {"Ca_in": SimpleNamespace(value=fold.CA), "Bo": SimpleNamespace(value=1.56)}
            self.pins = np.array([7., 8.])
            self._azimuthal_mode_param_m = SimpleNamespace(value=0)
            self.eigen_data_in_states = self.continuation_data_in_states = False
            self.case = {"software_only": True}
            self.residual = ""
            self.steady = True
            self._domains_to_remesh = set()
            self._remesh_requested_inside_solve = False
            self.do_call_remeshing_when_necessary = False
            self._meshdict = {}
            self._meshtemplate_list = []
            for name in fold.FLAGS:
                setattr(self, name, False)

        def get_eigen_solver(self):
            return self.solver

        def get_bifurcation_tracking_mode(self):
            return ""

        def actions_before_stationary_solve(self, **kwargs):
            pass

        def _restore_state(self, raw):
            assert raw == b"software-only full-state payload"
            self.pins = np.array([7., 99. if pin_drift else 8.])

        def get_global_parameter(self, name):
            return self.parameters[name]

        def time_stepper_pt(self, index):
            return SimpleNamespace(make_steady=lambda: setattr(self, "steady", True),
                                   undo_make_steady=lambda: setattr(self, "steady", False))

        def _set_solved_residual(self, value, *args):
            self.residual = value

        def invalidate_cached_mesh_data(self):
            pass

    problem = Problem()
    owner = fold.solver_owner_signature(problem)
    state = {"mode_state": {"actual_pins": fold.array_record(problem.pins)},
             "solver_owner": owner, "solver": {"max_retries": 0, "use_pardiso": False},
             "continuation_data_in_states": False, "scalar_fields": {"newton_solver_tolerance": 1e-8},
             "selected_residual": ""}
    native_signature = {"mode": 0, "flags": {name: False for name in fold.FLAGS},
                        "parameters": {"Ca_in": fold.CA, "Bo": 1.56}, "steady": [True],
                        "eigen_data_in_states": False}
    saved = {"signature": state, "native": {"signature": native_signature,
             "eigendata": {name: None for name in fold.EIGEN_FIELDS}},
             "payload": b"software-only full-state payload", "manipulators": [],
             "continuation": fold.capture_continuation(problem), "case": {"software_only": True}, "fixed": {}}
    problem.pins = np.array([-1., -2.])
    problem.newton_solver_tolerance = 1e-4
    problem.residual = "wrong"
    adapter = object.__new__(fold.Adapter)
    adapter.problem, adapter.prepared, adapter.receipts = problem, {}, []
    adapter.mesh_guard = fold.RemeshGuard(problem)
    if historical_mesh_violation:
        adapter.mesh_guard.requests = adapter.mesh_guard.physical_callbacks = 1
        problem._domains_to_remesh = {"liquid"}
        problem._remesh_requested_inside_solve = True
    adapter.signature = lambda: {"signature": {**state, "mode_state": {"actual_pins": fold.array_record(problem.pins)},
                                               "scalar_fields": {"newton_solver_tolerance": problem.newton_solver_tolerance},
                                               "selected_residual": problem.residual}, "fixed": {}}
    monkeypatch.setattr(fold, "stable_inputs", lambda _: None)
    if pin_drift:
        with pytest.raises(fold.Refusal, match="fields_pins_history_or_controls_restore_failed"):
            adapter.restore(saved, "software-load")
        assert adapter.receipts == []
    else:
        adapter.restore(saved, "software-load")
        assert adapter.receipts[0]["verified"] is True
        assert problem.newton_solver_tolerance == 1e-8 and problem.residual == ""
        assert not problem._domains_to_remesh and problem._remesh_requested_inside_solve is False
        assert adapter.mesh_guard.requests == adapter.mesh_guard.physical_callbacks == int(historical_mesh_violation)
        assert adapter.receipts[0]["phase_mesh_violation_precedes_restore"] is historical_mesh_violation
        assert all(value == 0 for value in adapter.receipts[0]["physical_mesh_event_deltas"].values())


@pytest.fixture
def candidate_harness(monkeypatch, case, tmp_path):
    """Synthetic three-DOF candidate with real adapter and guard orchestration."""
    monkeypatch.setattr(fold, "N", 3)
    stationary = sparse.diags([0., 2., 3.], format="csr")
    jacobian, mass = -stationary, sparse.diags([1., 0., 0.], format="csr")
    null = np.array([1., 0., 0.])
    layout = {"types": np.array([0, 1, 2], dtype=np.int64),
              "names": ["liquid/mesh_x", "globals/U", "gas_volume/pressure"],
              "forced": [], "globals": {"globals/U": [1], "gas_volume/pressure": [2]}}
    output = tmp_path / "candidate"
    output.mkdir()
    bank = tmp_path / "own-bank.npz"
    np.savez(bank, values=np.zeros(8, dtype=complex), vectors=np.tile(null, (8, 1)).astype(complex))
    layout_path = tmp_path / "own-layout.json"
    write(layout_path, {key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in layout.items()})
    state_path = tmp_path / "admitted-state.dump"
    state_path.write_bytes(b"synthetic history; no native evidence")
    prepared = {"case": copy.deepcopy(case), "component": {}, "environment": {},
                "restart": {"state": {"path": str(state_path)}},
                "contract": {"banks": [{"bank": fold.regular(bank), "layout": fold.regular(layout_path)}]}}

    class Solver:
        def __init__(self, problem):
            self._problem_wr = weakref.ref(problem)
            self.matrix_manipulators = []
            self.max_retries, self.use_pardiso = 0, False

        @property
        def problem(self):
            return self._problem_wr()

        def get_J_M_n_and_type(self):
            return jacobian.copy(), mass.copy(), 3, False

    class Problem(Continuation):
        def __init__(self):
            super().__init__()
            self.current = np.array([3., 4., 5.])
            self.fields, self.pins = np.array([3., 4., 5.]), np.array([7., 8.])
            self.parameters = {"Ca_in": SimpleNamespace(value=fold.CA), "Bo": SimpleNamespace(value=1.56)}
            self.solver = Solver(self)
            self.case, self.residual, self.tracking = copy.deepcopy(case), "", ""
            self.steady = True
            self.eigen_data_in_states = self.continuation_data_in_states = False
            self._azimuthal_mode_param_m = SimpleNamespace(value=0)
            self._domains_to_remesh, self._meshdict = set(), {}
            self._remesh_requested_inside_solve = False
            template = type("PaperTwoPhaseMesh", (), {})()
            self._domains_remesh_on_inversion, self._meshtemplate_list = {template}, [template]
            self._inversion_remesh_threshold, self.inversion_remesh_max_retries = 0, 3
            self.inversion_reports = 7  # A previous solve's count, reset at this solve.
            self.request_quality_remesh = self.invert_during_solve = False
            self.restoration_effect = None
            for name in fold.FLAGS:
                setattr(self, name, False)
            for name in fold.EIGEN_FIELDS:
                setattr(self, name, None)
            for name in ("force_remesh", "remesh_if_necessary", "remesh_handler_during_continuation", "_adapt",
                         "adapt", "_adapt_with_interfacial_errors", "custom_adapt", "actions_before_remeshing",
                         "actions_before_adapt", "_perform_pending_remesh", "actions_after_adapt", "actions_after_remeshing"):
                setattr(self, name, lambda *a, **k: None)

        def ndof(self):
            return 7 if self.tracking else 3

        def get_eigen_solver(self):
            return self.solver

        def get_global_parameter(self, name):
            return self.parameters[name]

        def get_bifurcation_tracking_mode(self):
            return self.tracking

        def activate_bifurcation_tracking(self, *args, **kwargs):
            assert args == ("Ca_in", "fold") and kwargs["blocksolve"] is False
            assert np.array_equal(kwargs["eigenvector"], null)
            self.tracking = "fold"

        def deactivate_bifurcation_tracking(self):
            self.tracking = ""

        def _get_inversion_reports(self):
            return self.inversion_reports

        def _set_globally_convergent_newton_method(self, value):
            assert value is False

        def actions_before_stationary_solve(self, **kwargs):
            pass

        def actions_before_eigen_solve(self):
            pass

        def setup_forced_zero_dof_list_for_eigenproblems(self):
            pass

        def ntime_stepper(self):
            return 1

        def time_stepper_pt(self, index):
            return SimpleNamespace(make_steady=lambda: setattr(self, "steady", True),
                                   undo_make_steady=lambda: setattr(self, "steady", False))

        def _set_solved_residual(self, value, *args):
            self.residual = value

        def invalidate_cached_mesh_data(self):
            pass

        def payload(self):
            return json.dumps({"fields": self.fields.tolist(), "pins": self.pins.tolist(),
                               "ca_in": self.parameters["Ca_in"].value}).encode()

        def _restore_state(self, raw):
            saved = json.loads(raw)
            self.fields, self.pins = np.array(saved["fields"]), np.array(saved["pins"])
            self.parameters["Ca_in"].value = saved["ca_in"]
            self.actions_after_adapt()
            self.actions_after_remeshing()
            if self.restoration_effect == "pending":
                self._domains_to_remesh.add("gas")
            if self.restoration_effect == "counter":
                self.adapter.mesh_guard.requests += 1

        def load_state(self, path):
            self._restore_state(Path(path).read_bytes())

        def assemble_jacobian(self, **kwargs):
            assert kwargs == {"with_residual": True}
            return np.zeros(3), stationary.copy()

        def get_last_eigenvectors(self):
            return self._last_eigenvectors

        def solve(self, **kwargs):
            assert kwargs == {"spatial_adapt": 0, "max_newton_iterations": 20, "do_not_set_IC": True,
                              "newton_relaxation_factor": 1.0, "globally_convergent_newton": False}
            self.inversion_reports = 0
            self.monitor.before_solve()
            self.monitor.before_iteration()
            self.fields[:] = [-2., 0., 0.]
            self.parameters["Ca_in"].value = fold.CA - 1e-5
            self._last_eigenvectors = np.array([null])
            self.monitor.after_update()
            self.monitor.before_iteration()
            if self.request_quality_remesh:
                self._domains_to_remesh.add("liquid")
            if self.invert_during_solve:
                self.inversion_reports = 1
                raise RuntimeError("synthetic inverted element")

    class SoftwareAdapter(fold.Adapter):
        def canonicalise_continuation(self):
            pass

        def signature(self):
            p = self.problem
            native_signature = {"mode": 0, "flags": {name: getattr(p, name) for name in fold.FLAGS},
                                "parameters": {name: parameter.value for name, parameter in p.parameters.items()},
                                "steady": [p.steady], "eigen_data_in_states": p.eigen_data_in_states}
            eigendata = fold.copy_eigendata(p)
            continuation = fold.capture_continuation(p)
            signature = {"mode_state": {**native_signature, "free": fold.array_record(p.fields),
                                        "actual_pins": fold.array_record(p.pins)},
                         "solver_owner": fold.solver_owner_signature(p),
                         "solver": {"max_retries": p.solver.max_retries, "use_pardiso": p.solver.use_pardiso},
                         "scalar_fields": {name: getattr(p, name) for name in fold.CONTROLS},
                         "continuation_data_in_states": p.continuation_data_in_states,
                         "selected_residual": p.residual, "eigendata": fold.eigen_signature(eigendata),
                         "continuation": fold.continuation_signature(continuation)}
            return {"signature": signature, "native": {"signature": native_signature, "eigendata": eigendata},
                    "payload": p.payload(), "continuation": continuation, "manipulators": [],
                    "case": copy.deepcopy(p.case), "fixed": {"synthetic_fixed_layout": True}}

        def check_newton(self, event):
            self.check()

        @contextlib.contextmanager
        def update_session(self, monitor):
            self.problem.monitor = monitor
            try:
                yield monitor
                monitor.finalise()
            finally:
                self.problem.monitor = None

        def solve_spectrum(self, shift):
            self.check()
            assert shift in fold.LIMITS["shifts"] and self.work["eigenproblem_calls"] < 2
            self.work["eigenproblem_calls"] += 1
            return np.zeros(8, dtype=complex), np.tile(null, (8, 1)).astype(complex)

    problem = Problem()
    adapter = SoftwareAdapter(problem, prepared, output, SimpleNamespace(get_detect_inverted_elements=lambda: True),
                              object, object, None, datetime.now(UTC) + timedelta(minutes=5))
    problem.adapter = adapter
    checkpoint_calls, layout_calls = [], []

    def stationary_signature(problem, *args):
        return {"parameters": {name: parameter.value for name, parameter in problem.parameters.items()},
                "fields": problem.fields.tolist(), "bc_cache_marker": "init", "layout": {"ndof": 3}}

    def raw_layout(problem, n, mode):
        assert n == 3 and mode == 0 and isinstance(layout["types"], np.ndarray)
        layout_calls.append("ndarray layout")
        return copy.deepcopy(layout)

    def checkpoint(problem, value, directory, index, azimuthal):
        assert adapter.mesh_guard.context == "fold-checkpoint-roundtrip"
        assert index == 16 and azimuthal is True and problem.case == value
        checkpoint_calls.append(value["physics"]["ca_in"])
        payload = problem.payload()
        state = directory / "portable-state.dump"
        state.write_bytes(payload)
        problem.load_state(state)
        problem._restore_state(payload)
        resume = directory / "resume.json"
        write(resume, {"synthetic": True})
        return {"resume_packet": "resume.json"}

    monkeypatch.setattr(fold, "stable_inputs", lambda _: None)
    monkeypatch.setattr(fold, "manifest", lambda *args: {"synthetic": True})
    monkeypatch.setattr(fold.modes, "mode_layout", raw_layout)
    monkeypatch.setattr(fold.model, "_stationary_state_signature", stationary_signature)
    monkeypatch.setattr(fold.model, "_checkpoint_stationary_state", checkpoint)
    monkeypatch.setattr(fold.diagnostic, "actual_pins", lambda p: fold.array_record(p.pins))
    monkeypatch.setattr(fold.history, "serialized_metadata", lambda _: {"synthetic": True})
    monkeypatch.setattr(fold.history, "check_time_history", lambda *args: {"synthetic_unchanged": True})
    monkeypatch.setattr(fold.remesh, "sample", lambda p, *args: {
        "metrics": {}, "signature": stationary_signature(p), "actual_pins": fold.array_record(p.pins)})
    monkeypatch.setattr(fold.remesh, "canonical_checks", lambda *args: {name: True for name in fold.modes.GATES})
    monkeypatch.setattr(fold.approach.iii, "physical_signature", lambda value: value)
    monkeypatch.setattr(fold.runner, "validate_restart", lambda *args, **kwargs: {
        "roundtrip": {"after": stationary_signature(problem)}})
    return SimpleNamespace(adapter=adapter, prepared=prepared, output=output, layout=layout,
                           checkpoint_calls=checkpoint_calls, layout_calls=layout_calls)


def test_execute_preserves_typed_control_refusal_evidence_and_releases_fake_problem(candidate_harness, monkeypatch):
    h = candidate_harness
    problem = h.adapter.problem
    problem._arclength_inner_product = "mass"
    evidence = {name: getattr(problem, name) for name in fold.CONTROLS}
    releases = []
    for name in ("set_output_directory", "set_linear_solver", "set_c_compiler",
                 "setup_for_stability_analysis", "set_eigensolver", "initialise"):
        setattr(problem, name, lambda *args, **kwargs: None)
    problem.is_distributed = lambda: False
    problem.release = lambda: releases.append("released")
    h.adapter.canonicalise_continuation = lambda: fold.capture_continuation(problem)
    core_path = h.output / "synthetic-core.so"
    core_path.write_bytes(b"software-only placeholder; no native library")
    core = SimpleNamespace(__file__=str(core_path), get_detect_inverted_elements=lambda: True,
                           set_detect_inverted_elements=lambda value: None)
    # Cached synthetic modules replace the runtime imports; no native code is loaded.
    modules = {name: ModuleType(name) for name in ("pyoomph", "pyoomph.generic", "pyoomph.generic.problem",
               "pyoomph.meshes", "pyoomph.meshes.mesh")}
    modules["pyoomph"]._pyoomph_core = core
    modules["pyoomph.generic.problem"].GenericProblemHooks = object
    modules["pyoomph.meshes.mesh"].ODEStorageMesh = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    bridge = SimpleNamespace(verify_problem=lambda p: None,
                             restore_after_release=lambda **kwargs: {"verified": True})
    monkeypatch.setattr(fold, "sha", lambda path: fold.diagnostic.CORE)
    monkeypatch.setattr(fold, "construct_type", lambda *args: (type(problem), bridge))
    monkeypatch.setattr(fold, "build_bridged", lambda *args: problem)
    monkeypatch.setattr(fold, "Adapter", lambda *args: h.adapter)
    monkeypatch.setattr(fold.approach.iii, "restore_checked", lambda *args: {"synthetic": True})
    h.prepared.update(native_execution_authorized=True, environment={"native_core": {"path": str(core_path)}},
                      cold={"actual_pins": {}, "loaded": {"serialized_state": {"payload_sha256": "synthetic"}}})
    result = fold.execute(h.prepared, h.output, [], h.adapter.deadline)
    assert result["refusal"]["label"] == "unsupported_continuation_controls"
    assert result["refusal"]["evidence"] == evidence
    assert json.loads((h.output / "result.json").read_text())["refusal"]["evidence"] == evidence
    assert result["status"] == "inconclusive" and result["numerical_acceptance"] is False
    assert releases == ["released"] and h.adapter.work["augmented_solve_calls"] == 0
    assert all(result[name] is False for name in fold.NO_CLAIMS)


def test_main_preserves_typed_outer_refusal_evidence(monkeypatch, case, tmp_path):
    output = tmp_path / "runtime"
    prepared = {"case": case, "output": output, "contract": {}, "records": [], "implementation": {}}
    evidence = {"_arclength_inner_product": "mass", "_solve_in_arclength_conti": None}
    monkeypatch.setattr(fold, "preflight", lambda args: prepared)
    monkeypatch.setattr(fold, "manifest", lambda *args: {"synthetic": True})

    def refuse(*args):
        raise fold.Refusal("unsupported_continuation_controls", evidence)

    monkeypatch.setattr(fold, "execute", refuse)
    arguments = [item for name in ("source", "case", "manifest", "packet", "state", "spectrum", "out")
                 for item in ("--" + name, str(tmp_path / name))]
    arguments.extend(["--deadline", (datetime.now(UTC) + timedelta(minutes=1)).isoformat()])
    assert fold.main(arguments) == 1
    for name in ("failure.json", "completion.json"):
        result = json.loads((output / name).read_text())
        assert result["refusal"]["evidence"] == evidence
        assert result["status"] == "inconclusive" and result["numerical_acceptance"] is False
        assert all(result[claim] is False for claim in fold.NO_CLAIMS)


def test_candidate_orchestration_normalizes_ndarray_layout_and_scopes_both_checkpoint_loads(candidate_harness):
    h = candidate_harness
    original = h.adapter.signature()
    result = fold.run_candidate(h.adapter, h.prepared, h.output, [])
    assert result["status"] == "candidate_ready_for_cold_qualification", result
    assert result["stationary_fold_candidate_passed"] is result["direct_dynamic_null_passed"] is True
    assert result["returned_real_null_matched"] is result["original_full_state_restored"] is True
    assert h.adapter.signature()["signature"] == original["signature"]
    assert h.layout_calls == ["ndarray layout", "ndarray layout"]
    assert isinstance(h.layout["types"], np.ndarray)  # The helper's original object is untouched.
    assert json.loads((h.output / "layout.json").read_text())["types"] == [0, 1, 2]
    assert h.checkpoint_calls == [fold.CA - 1e-5]
    event = next(row for row in result["mesh_callbacks"] if row["label"] == "fold-checkpoint-roundtrip")
    assert event["callbacks"] == ["actions_after_adapt", "actions_after_remeshing"] * 2
    assert event["verified"] is True
    assert result["physical_after_callbacks_observed"] == result["remesh_or_adapt_requests_observed"] == 0
    assert result["work"]["augmented_solve_calls"] == 1 and result["work"]["eigenproblem_calls"] == 2
    assert result["augmented_solve"]["inversion_reports_in_previous_solve_call_at_entry"] == 7
    assert result["augmented_solve"]["inversion_reports_in_augmented_solve_call"] == 0


def test_pending_quality_request_refuses_candidate_but_final_restoration_is_truthfully_successful(candidate_harness):
    h = candidate_harness
    original = h.adapter.signature()
    h.adapter.problem.request_quality_remesh = True
    result = fold.run_candidate(h.adapter, h.prepared, h.output, [])
    assert result["status"] == "inconclusive" and result["refusal"]["label"] == "physical_remesh_pending_flag"
    assert result["original_full_state_restored"] is True and "restoration_failure" not in result
    assert h.adapter.signature()["signature"] == original["signature"]
    receipt = result["restoration"][-1]
    assert receipt["leg"] == "original-final-restoration" and receipt["verified"] is True
    assert receipt["pending_request_metadata_before"]["_domains_to_remesh"] == ["liquid"]
    assert receipt["pending_request_metadata_after"] == {"_domains_to_remesh": [], "_remesh_requested_inside_solve": False}
    assert receipt["phase_mesh_violation_precedes_restore"] is True
    assert all(value == 0 for value in receipt["physical_mesh_event_deltas"].values())
    assert result["remesh_or_adapt_requests_observed"] == h.adapter.mesh_guard.requests == 1
    assert h.adapter.mesh_guard.flag_events and not h.checkpoint_calls
    assert result["work"]["eigenproblem_calls"] == result["physical_remeshes_performed"] == 0


@pytest.mark.parametrize("effect,label", [("pending", "physical_remesh_pending_during_restore"),
                                          ("counter", "physical_mesh_event_during_restore")])
def test_restoration_cannot_certify_new_mesh_violations(candidate_harness, effect, label):
    h = candidate_harness
    saved = h.adapter.signature()
    h.adapter.problem.restoration_effect = effect
    with pytest.raises(fold.Refusal, match=label):
        h.adapter.restore(saved, "software-new-restore-event")
    assert h.adapter.receipts == [] and h.adapter.mesh_guard.events[-1]["verified"] is False


def test_augmented_inversion_count_is_current_call_and_cannot_be_hidden_by_previous_count(candidate_harness):
    h = candidate_harness
    h.adapter.problem.invert_during_solve = True
    result = fold.run_candidate(h.adapter, h.prepared, h.output, [])
    assert result["refusal"]["label"] == "inverted_element_during_augmented_solve"
    assert result["augmented_solve"]["inversion_reports_in_previous_solve_call_at_entry"] == 7
    assert result["augmented_solve"]["inversion_reports_in_augmented_solve_call"] == 1
    assert result["original_full_state_restored"] is True and not h.checkpoint_calls


def test_candidate_refuses_unexpected_checkpoint_callback_inventory(candidate_harness, monkeypatch):
    h = candidate_harness
    helper = fold.model._checkpoint_stationary_state

    def extra_callback(*args):
        checkpoint = helper(*args)
        h.adapter.problem.actions_after_adapt()
        return checkpoint

    monkeypatch.setattr(fold.model, "_checkpoint_stationary_state", extra_callback)
    result = fold.run_candidate(h.adapter, h.prepared, h.output, [])
    assert result["refusal"]["label"] == "fold_checkpoint_callback_inventory_differs"
    assert result["original_full_state_restored"] is True and result["work"]["eigenproblem_calls"] == 0
    event = next(row for row in result["mesh_callbacks"] if row["label"] == "fold-checkpoint-roundtrip")
    assert event["verified"] is False and len(event["callbacks"]) == 5


def test_external_execution_seal_is_unchanged_when_its_witness_joins_run_inventory(bank_bundle, case, monkeypatch):
    spectrum, restart = bank_bundle
    stationary = spectrum.parent / (spectrum.name + "-stationary")
    stationary.mkdir()
    case_path, manifest_path = stationary / "case.json", stationary / "run-manifest.json"
    write(case_path, case)
    state_path, packet_path = stationary / "state.dump", stationary / "resume.json"
    state_path.write_bytes(b"synthetic admitted state")
    write(packet_path, {"synthetic": True})
    write(stationary / "completion.json", {"synthetic": True})
    write(spectrum / "completion.json", {"synthetic": True})
    core_path = stationary / "synthetic-core.so"
    core_path.write_bytes(b"metadata only; never imported")
    component = {"commit": "synthetic-clean-commit", "clean": True}
    environment = {"commit": fold.diagnostic.PYOOMPH,
                   "native_core": {"path": str(core_path), "sha256": fold.diagnostic.CORE},
                   "interpreter": {"resolved_executable": "synthetic-metadata-interpreter"}}
    parent = {"anchor_stage": {"kind": fold.approach.KIND, "index": 1, "ca_in": fold.CA},
              "case_sha256": fold.sha(case_path), "pyoomph": environment}
    write(manifest_path, parent)
    restart = copy.deepcopy(restart)
    restart.update(state=fold.regular(state_path), packet=fold.regular(packet_path), meshes=[])
    restart["roundtrip"]["layout"] = {"ndof": 3}
    write(spectrum / "result.json", {"request": fold.approach.request_contract(case, fold.LIMITS["shifts"], 8),
          "numerical_acceptance": True, "mode_comparisons": [{"matched_returned_envelope": True}]})
    write(spectrum / "run-manifest.json", {"component_commit": fold.BASE, "case_sha256": fold.sha(case_path),
          "parent_restart": {"packet": restart["packet"]}, "pyoomph": environment})
    monkeypatch.setattr(fold.approach, "is_producer", lambda _: True)
    monkeypatch.setattr(fold, "source_contract", lambda *args: (component, {fold.ENTRYPOINT: fold.regular(fold.ROOT / fold.ENTRYPOINT)}))
    monkeypatch.setattr(fold.runner, "pyoomph_revision", lambda: environment)
    monkeypatch.setattr(fold.approach.coarse, "environment_contract", lambda *args: fold.regular(core_path))
    monkeypatch.setattr(fold.runner, "validate_restart", lambda *args, **kwargs: restart)
    monkeypatch.setattr(fold.approach, "admit_stage", lambda *args: ({}, {}, [fold.regular(state_path)]))
    monkeypatch.setattr(fold, "closure", lambda path: ({}, [fold.regular(path / "completion.json")]))
    arguments = SimpleNamespace(source=fold.ROOT, case=case_path, manifest=manifest_path, packet=packet_path,
                                state=state_path, spectrum=spectrum, check=True, seal=None,
                                out=spectrum.parent / (spectrum.name + "-metadata-output"))
    unsigned = fold.preflight(arguments)
    original_seal = copy.deepcopy(unsigned["required_seal"])
    seal_path = stationary / "external-seal.json"
    write(seal_path, original_seal)
    original_bytes = seal_path.read_bytes()
    arguments.seal = seal_path
    bound = fold.preflight(arguments)
    assert bound["required_seal"] == original_seal == json.loads(seal_path.read_text())
    assert seal_path.read_bytes() == original_bytes
    assert bound["records"][-1] == bound["input_seal"] == fold.regular(seal_path)
    assert len(bound["records"]) == len(bound["required_seal"]["inputs"]) + 1
    assert bound["records"] is not bound["required_seal"]["inputs"]
    assert not arguments.out.exists() and bound["native_execution_authorized"] is False


def test_all_scientific_claim_flags_are_closed_and_no_void_campaign_is_imported():
    assert fold.NO_CLAIMS and all(value is False for value in fold.NO_CLAIMS.values())
    assert "saddle_node_nondegeneracy_certified" in fold.NO_CLAIMS
    text = (fold.ROOT / fold.ENTRYPOINT).read_text()
    assert "CAMPAIGN_DEBIT" not in text and "90101" not in text
    assert "VoidBubbleProblem" not in text


def test_gate_exception_cannot_be_swallowed_by_native_RuntimeError_recovery():
    assert issubclass(fold.Refusal, Exception) and not issubclass(fold.Refusal, RuntimeError)


@pytest.mark.parametrize("bc_changes", [False, True])
def test_second_stationary_hook_must_preserve_full_state_and_order(monkeypatch, bc_changes):
    state = {"bc_cache_marker": "init", "layout": {"ndof": 3, "order": "original"}}
    calls = []

    def before_stationary(**kwargs):
        calls.append(kwargs)
        if bc_changes and len(calls) == 2:
            state["layout"]["order"] = "renumbered"

    problem = SimpleNamespace(actions_before_stationary_solve=before_stationary)
    monkeypatch.setattr(fold.model, "_stationary_state_signature", lambda *_: copy.deepcopy(state))
    if bc_changes:
        with pytest.raises(fold.Refusal, match="stationary_BC_not_idempotent"):
            fold.stationary_bc_idempotence(problem)
    else:
        receipt = fold.stationary_bc_idempotence(problem)
        assert receipt["complete_signature_equal"] is True and receipt["bc_cache_marker"] == "init"
    assert calls == [{"force_reassign_eqs": False}, {"force_reassign_eqs": False}]


def test_bridge_keeps_cached_builder_validation_path_and_rejects_a_different_class(monkeypatch, case):
    class Scientific:
        pass

    calls = []
    monkeypatch.setattr(fold.model, "build_problem", lambda value, **options: calls.append((value, options)) or Scientific())
    assert type(fold.build_bridged(Scientific, case)) is Scientific
    assert calls == [(case, {"initial_bond": 1.56})]
    with pytest.raises(fold.Refusal, match="cached_scientific_class_identity"):
        fold.build_bridged(type("Foreign", (), {}), case)


@pytest.mark.parametrize("failure", [False, True])
def test_inversion_wrapper_is_empty_only_during_augmented_scope_and_controls_restore(failure):
    PaperTwoPhaseMesh = type("PaperTwoPhaseMesh", (), {})
    template = PaperTwoPhaseMesh()
    calls = []
    problem = SimpleNamespace(_domains_remesh_on_inversion={template}, _meshtemplate_list=[template],
              _inversion_remesh_threshold=0, inversion_remesh_max_retries=3,
              max_newton_iterations=40, newton_relaxation_factor=1., newton_solver_tolerance=1e-8,
              _get_inversion_reports=lambda: 7, deactivate_bifurcation_tracking=lambda: calls.append("deactivate"),
              _set_globally_convergent_newton_method=lambda value: calls.append(("global", value)))
    core = SimpleNamespace(get_detect_inverted_elements=lambda: True)
    receipt = {}
    try:
        with fold.augmented_controls(problem, core, receipt):
            assert problem._domains_remesh_on_inversion == set() and problem.inversion_remesh_max_retries == 0
            assert core.get_detect_inverted_elements() is True
            problem.max_newton_iterations = 20
            problem.newton_solver_tolerance = 1e-3
            if failure:
                raise fold.Refusal("software-callback-refusal")
    except fold.Refusal:
        assert failure
    assert problem._domains_remesh_on_inversion == {template} and problem.inversion_remesh_max_retries == 3
    assert problem.max_newton_iterations == 40 and problem.newton_solver_tolerance == 1e-8
    assert receipt["original_solve_controls_restored"] is True and receipt["original_inversion_templates_restored"] is True
    assert receipt["inversion_reports_in_previous_solve_call_at_entry"] == 7
    assert receipt["inversion_reports_in_augmented_solve_call"] == 7
    assert receipt["inversion_counter_semantics"] == "reset at each native solve call; not cumulative"
    assert calls == ["deactivate", ("global", False)]


def test_new_case_adoption_never_assigns_the_tracked_parameter(monkeypatch, case):
    live = fold.CA - 1e-5
    signature = {"parameters": {"Ca_in": live, "Bo": 1.56}, "fields": "software-fields"}
    problem = SimpleNamespace(case=copy.deepcopy(case))
    monkeypatch.setattr(fold.model, "_stationary_state_signature", lambda *_: copy.deepcopy(signature))
    monkeypatch.setattr(fold.diagnostic, "actual_pins", lambda _: {"sha256": "software-pins"})
    new, receipt = fold.adopt_measured_case(problem, case, live)
    assert problem.case == new and new["physics"]["ca_in"] == live
    assert receipt["native_parameter_assignments"] == 0 and receipt["complete_signature_equal"] is True


def test_cold_qualification_uses_new_case_receipt_directory_and_new_history(monkeypatch, case, tmp_path):
    case = fold.measured_case(case, fold.CA - 1e-5)
    signature = {"fields": "new-fold-history", "serialized_state": {"payload_sha256": "new-fold-payload"}}
    pins = {"sha256": "new-fold-pins"}
    calls = []

    class Cold:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_output_directory(self, value):
            pass

        def set_linear_solver(self, value):
            pass

        def set_c_compiler(self, value):
            pass

        def setup_for_stability_analysis(self, **kwargs):
            assert kwargs == {"azimuthal_stability": True, "analytic_hessian": False}

        def initialise(self):
            pass

    cold = Cold()
    monkeypatch.setattr(fold, "build_bridged", lambda cls, value: calls.append(("build", value)) or cold)

    def restore(problem, value, restart, actual_pins, payload):
        assert problem._stationary_iii_receipts_directory == tmp_path / "loads"
        assert value == case and actual_pins == pins and payload == "new-fold-payload"
        calls.append(("restore", payload))
        return {"loaded": signature}

    monkeypatch.setattr(fold.approach.iii, "restore_checked", restore)
    monkeypatch.setattr(fold.remesh, "sample", lambda *a: {"metrics": {}, "signature": signature, "actual_pins": pins})
    monkeypatch.setattr(fold.remesh, "canonical_checks", lambda c, m: {name: True for name in fold.modes.GATES})
    restart = {"roundtrip": {"after": signature}}
    result = {"fold_case": case, "cold_plan": {"restart": restart, "pins": pins}}
    assert all(fold.qualify_fold_cold(Cold, {}, result, tmp_path).values())
    assert calls == [("build", case), ("restore", "new-fold-payload")]
