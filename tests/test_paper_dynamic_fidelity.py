"""Native-denied software contracts; all processes/steps/buffers are synthetic."""

from __future__ import annotations

import contextlib
import copy
import importlib.abc
import json
import os
import stat
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith("pyoomph.") or "_pyoomph" in fullname:
            raise AssertionError("native import denied in prospective fidelity tests")


guard = NoNative()
sys.meta_path.insert(0, guard)
try:
    from test_paper_dynamic_capture import Problem as RawProblem
    from test_paper_dynamic_jump import Template

    from runners import paper_dynamic_fidelity as fidelity
    from runners import paper_dynamic_jump_runtime as runtime
    from runners import run_paper_dynamic_fidelity as cli
finally:
    sys.meta_path.remove(guard)


@pytest.fixture(autouse=True)
def deny_native():
    blocker = NoNative()
    sys.meta_path.insert(0, blocker)
    try:
        yield
        assert not any(name == "pyoomph" or name.startswith("pyoomph.") or "_pyoomph" in name for name in sys.modules)
    finally:
        sys.meta_path.remove(blocker)


def store(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")
    return fidelity.reference(path)


def initial_witness():
    return {"schema": "synthetic-not-native", "time": 2.0, "counter": 3, "dts": [0.5, 0.5],
        "data": [{"name": "synthetic/nodal", "kind": "nodal", "fields": ["value"],
                  "rows": [[float(i)] for i in range(7)], "pins": [False], "equations": [0]}],
        "parameters": {"Ca_in": 0.01, "Bo": 0.5}, "census": {"synthetic": True},
        "layout": {"ndof": 1}, "mode": {"base": True}, "flags": runtime.DYNAMIC_FLAGS.copy(),
        "geometry": {"connectivity": [0]}, "steppers": [{"type": "synthetic-primary", "order": 2,
          "ntstorage": 7, "nprev_values": 2, "is_steady": False, "time": 2.0, "counter": 3,
          "dts": [0.5, 0.5], "time_history": [2.0, 1.5, 1.0], "weights": {}}],
        "policy": {"all_parameters": {"Ca_in": 0.01, "Bo": 0.5}, "serialization": {"output_step": 2}}}


# Source-derived API presence at a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618.
# Python Problem plus src/nanobind/problem.cpp; literals do not depend on
# EvolutionGuard.REQUIRED. These stubs establish software behaviour only.
PYOOMPH_A186_API = frozenset({
    '_adapt',
    '_adapt_with_interfacial_errors',
    '_apply_deferred_initial_refinement',
    '_arc_length_step',
    '_compute_arclength_tangent',
    '_load_state',
    '_perform_pending_remesh',
    '_solve_eigenproblem_helper',
    '_solve_normal_mode_eigenproblem',
    '_solve_normal_mode_eigenproblem_impl',
    '_solve_with_inversion_remesh',
    '_start_bifurcation_tracking',
    '_start_orbit_tracking',
    'actions_after_remeshing',
    'actions_before_newton_solve',
    'actions_before_remeshing',
    'activate_bifurcation_tracking',
    'activate_eigenbranch_tracking',
    'adapt',
    'adaptive_unsteady_newton_solve',
    'arclength_continuation',
    'assemble_eigenproblem_matrices',
    'assign_initial_values_impulsive',
    'custom_adapt',
    'deactivate_bifurcation_tracking',
    'deflated_solve_by_eigenperturbation',
    'doubly_adaptive_unsteady_newton_solve',
    'find_bifurcation_via_eigenvalues',
    'force_remesh',
    'initialise',
    'initialise_dt',
    'load_state',
    'newton_solve',
    'perturb_by_eigenfunction',
    'precice_run',
    'refine_eigenfunction',
    'refine_uniformly',
    'remesh_handler_during_continuation',
    'remesh_if_necessary',
    'run',
    'set_current_pinned_values',
    'set_current_time',
    'set_eigenfunction_as_dofs',
    'set_history_dofs',
    'set_initial_condition',
    'set_max_refinement_level',
    'shift_time_values',
    'solve',
    'solve_auxiliary_residual',
    'solve_eigenproblem',
    'steady_newton_solve',
    'unrefine_uniformly',
    'unsteady_newton_solve',
})


class CallbackProblem:
    def __init__(self, *, zero=False, fault=None):
        self.witness = initial_witness()
        self.zero, self.fault = zero, fault
        self.calls = []
        self._initialised = True
        for name, value in runtime.DYNAMIC_FLAGS.items():
            setattr(self, name, value)
        self._runmode = "new"
        self.initial_adaption_steps = 0
        self.project_initial_conditions = False
        self.inversion_remesh_max_retries = 0
        self._remesh_requested_inside_solve = False

    def is_initialised(self):
        return self._initialised

    def get_scaling(self, name):
        assert name == "temporal"
        return 1.0

    def solve(self, **kwargs):
        self.calls.append(("solve", kwargs))
        self._solve_with_inversion_remesh(
            lambda: self.unsteady_newton_solve(kwargs["timestep"], kwargs["shift_values"]))
        self._perform_pending_remesh()
        # The high-level pinned solve increments the counter after its native
        # primitive succeeds; a failed attempt remains charged without this.
        self.witness["counter"] += 1
        self.witness["steppers"][0]["counter"] = self.witness["counter"]

    def _solve_with_inversion_remesh(self, callback):
        assert self.inversion_remesh_max_retries == 0
        return callback()

    def _perform_pending_remesh(self):
        assert self._remesh_requested_inside_solve is False
        return False

    def unsteady_newton_solve(self, dt, shift_values):
        assert shift_values is True
        # Embedded ordinary stepping shifts time/history before the virtual
        # Newton callbacks. This synthetic leaf contains no native arrays.
        old = self.witness["dts"][:]
        self.witness["time"] += dt
        self.witness["dts"] = [dt, old[0]]
        stepper = self.witness["steppers"][0]
        stepper.update(time=self.witness["time"], dts=self.witness["dts"][:])
        if self.fault == "dt_flatten":
            self.witness["dts"][1] = dt
        self.actions_before_newton_solve()
        self.actions_before_newton_step()
        if self.fault == "interrupted":
            raise RuntimeError("synthetic interrupted native update")
        if not self.zero:
            self.witness["data"][0]["rows"][0][0] += 0.125
            self.actions_after_newton_step()
        self.actions_after_newton_solve()

    def steady_newton_solve(self):
        self.calls.append("stationary")

    def solve_eigenproblem(self):
        self.calls.append("eigen")

    def force_remesh(self):
        self.calls.append("remesh")

    def adapt(self):
        self.calls.append("adapt")

    def assign_initial_values_impulsive(self):
        self.calls.append("fill")

    def actions_before_remeshing(self):
        self.calls.append("before_remesh")

    def actions_after_remeshing(self):
        self.calls.append("after_remesh")

    def actions_before_newton_solve(self):
        self.calls.append("before_solve")

    def actions_before_newton_step(self):
        self.calls.append("before_step")

    def actions_after_newton_step(self):
        self.calls.append("after_step")

    def actions_after_newton_solve(self):
        self.calls.append("after_solve")

    def output(self):
        self.calls.append("output")
        self.witness["policy"]["serialization"]["output_step"] += 1
        if self.fault == "output_history":
            self.witness["data"][0]["rows"][2][0] += 1

    def load_state(self):
        self.calls.append("load")


def synthetic_api_method(name):
    def call(self, *args, **kwargs):
        self.calls.append(name)
    return call


for _api in PYOOMPH_A186_API:
    if not hasattr(CallbackProblem, _api):
        setattr(CallbackProblem, _api, synthetic_api_method(_api))


def selected():
    return {"dt_solver": 0.25, "max_newton_iterations": 3, "max_newton_updates": 3, "newton_tolerance": 1e-6}


def selected_solve_kwargs():
    value = selected()
    return {"timestep": value["dt_solver"], "temporal_error": None, "spatial_adapt": 0,
        "shift_values": True, "do_not_set_IC": True, "max_newton_iterations": value["max_newton_iterations"],
        "newton_solver_tolerance": value["newton_tolerance"]}


def test_unissued_native_entry_refuses_before_import_or_factory(tmp_path):
    with pytest.raises(ValueError, match="metadata cannot authorise"):
        cli.execute({"native_execution_authorized": False}, tmp_path, lambda *_: None)


def test_expired_clock_refuses_even_a_previously_authorised_native_entry(tmp_path):
    _now, _identity, _tracker, _admission, prepared = authority(tmp_path)
    prepared.update(native_execution_authorized=True, pending=[])
    actual = datetime.now(UTC)
    prepared["phase"]["declared_at"] = (actual - timedelta(minutes=2)).isoformat()
    prepared["phase"]["deadline_utc"] = (actual - timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError, match="window"):
        cli.execute(prepared, tmp_path, lambda *_: None)


@pytest.mark.parametrize("key", sorted(fidelity.CONTROL_KEYS))
def test_each_unissued_control_is_an_actual_native_gate(key):
    value = selected()
    value[key] = None
    assert fidelity.controls(value, unissued=True)[1] == [key]
    with pytest.raises(ValueError, match="unissued"):
        fidelity.controls(value)


@pytest.mark.parametrize("name", ["steady_newton_solve", "solve_eigenproblem", "force_remesh", "adapt",
                                   "assign_initial_values_impulsive", "solve"])
def test_guard_denies_hidden_preparation_and_restores_instance_methods(name):
    problem = CallbackProblem()
    own = dict(vars(problem))
    events = []
    with pytest.raises(ValueError, match="prohibits"), fidelity.EvolutionGuard(problem, {}, lambda *x: events.append(x)).installed():
        getattr(problem, name)()
    assert vars(problem) == own and not problem.calls
    assert any(event[0] == "forbidden_native_entry" for event in events)


def test_step_scope_cannot_admit_a_hidden_load_or_second_solve():
    problem = CallbackProblem()
    guarded = fidelity.EvolutionGuard(problem, {}, lambda *_: None)
    with pytest.raises(ValueError, match="additional guarded entry"), guarded.installed(), guarded.declared("step", selected=selected()):
        problem.solve(**selected_solve_kwargs())
        problem.solve(**selected_solve_kwargs())
    assert len([v for v in problem.calls if isinstance(v, tuple)]) == 1
    problem = CallbackProblem()
    guarded = fidelity.EvolutionGuard(problem, {}, lambda *_: None)
    with pytest.raises(ValueError, match="prohibits load_state"), guarded.installed(), guarded.declared("step", selected=selected()):
        problem.load_state()
    assert not problem.calls


def test_partial_guard_installation_is_also_restored():
    problem = CallbackProblem()
    problem.steady_newton_solve = None
    own = dict(vars(problem))
    with pytest.raises(ValueError, match="required guarded API absent"), fidelity.EvolutionGuard(problem, {}, lambda *_: None).installed():
        pass
    assert vars(problem) == own


@pytest.mark.parametrize("zero", [False, True])
def test_sole_step_charges_once_preserves_old_dt_and_next_output(tmp_path, monkeypatch, zero):
    problem = CallbackProblem(zero=zero)
    events = []
    capture = lambda p: copy.deepcopy(p.witness)
    monkeypatch.setattr(fidelity, "retain", lambda p, c, out, index, w: {"accepted_step_index": index, "witness": copy.deepcopy(w)})
    guarded = fidelity.EvolutionGuard(problem, {}, lambda *x: events.append(x))
    with guarded.installed():
        got = fidelity.sole_step(problem, capture, guarded, selected(), datetime(2099, 1, 1, tzinfo=UTC),
                                 tmp_path, lambda *x: events.append(x), lambda: None)
    assert got["next_witness"]["dts"] == [0.25, 0.5] and got["next_witness"]["counter"] == 4
    assert got["next_witness"]["policy"]["serialization"]["output_step"] == 3
    assert got["work"]["solve_calls"] == 1 and got["work"]["newton_updates"] == (0 if zero else 1)
    assert got["callbacks"]["solve_attempts"] == 1
    assert len([v for v in problem.calls if isinstance(v, tuple)]) == 1
    charge = json.loads((tmp_path / "step-charge.json").read_text())
    assert charge["time_step_calls_charged"] == 1 and charge["retries"] == 0
    with pytest.raises(FileExistsError):
        fidelity.sole_step(problem, capture, guarded, selected(), datetime(2099, 1, 1, tzinfo=UTC),
                           tmp_path, lambda *_: None, lambda: None)


@pytest.mark.parametrize("fault", ["interrupted", "dt_flatten", "output_history"])
def test_failures_keep_exclusive_charge_and_do_not_retry(tmp_path, monkeypatch, fault):
    problem = CallbackProblem(fault=fault)
    monkeypatch.setattr(fidelity, "retain", lambda *_: {})
    guarded = fidelity.EvolutionGuard(problem, {}, lambda *_: None)
    with pytest.raises((ValueError, RuntimeError)), guarded.installed():
        fidelity.sole_step(problem, lambda p: copy.deepcopy(p.witness), guarded, selected(),
                           datetime(2099, 1, 1, tzinfo=UTC), tmp_path, lambda *_: None, lambda: None)
    assert (tmp_path / "step-charge.json").is_file() and (tmp_path / "step-work.json").is_file()
    assert len([v for v in problem.calls if isinstance(v, tuple)]) == 1
    if fault == "interrupted":
        receipt = json.loads((tmp_path / "step-work.json").read_text())
        assert receipt["callbacks"]["unmatched_possible_update"] is True
        assert receipt["resumable"] is False


def dump(reference):
    from test_paper_dynamic_jump import full_witness, synthetic_dump
    return synthetic_dump(reference, full_witness(-0.004))


@pytest.mark.parametrize("equal_bytes", [True, False])
def test_canonical_load_preserves_or_reconstructs_only_bound_owner(tmp_path, equal_bytes):
    live = tmp_path / "live.msh"
    live.write_bytes(b"mesh-one")
    problem = RawProblem(live)
    capture = problem.adapter
    original = problem._meshtemplate_list[0]
    active = original._get_template()
    target = tmp_path / "checkpoint" / "bound.msh"
    target.parent.mkdir()
    target.write_bytes(b"mesh-one" if equal_bytes else b"mesh-two")
    state = target.parent / "state.dump"
    state.write_bytes(dump("bound.msh"))
    calls = []

    def load(_path, **kwargs):
        calls.append(kwargs)
        if not equal_bytes:
            new = Template(problem)
            new._meshfile, new.remesher = str(target), active.remesher
            new.remesher.template = new
            original._template_override = new
            problem._domains_remesh_on_inversion = {original, new}
            for name in ("gas", "liquid"):
                problem._meshdict[name]._templatemesh = new

    problem.load_state = load
    loader = fidelity.LoadAdapter(problem, capture, SimpleNamespace(administrative_load=contextlib.nullcontext), lambda *_: None)
    meshes = [{**fidelity.reference(target), "stored_path": "bound.msh"}]
    loader.guard.forcing = {"Ca_in": -0.004, "Bo": 1.56}
    loader.load(fidelity.reference(state), meshes)
    assert calls[0]["additional_info"]["exchange_msh_file"] == {"bound.msh": "../live.msh" if equal_bytes else "bound.msh"}
    assert problem._meshtemplate_list[0] is original
    assert problem._meshtemplate_list[0]._get_template().remesher is active.remesher
    assert (problem._meshtemplate_list[0]._get_template() is active) is equal_bytes
    with pytest.raises(ValueError, match="additional native load"):
        loader.load(fidelity.reference(state), meshes)


def test_canonical_load_refuses_changed_mesh_before_load(tmp_path):
    live = tmp_path / "live.msh"
    live.write_bytes(b"old")
    problem = RawProblem(live)
    state = tmp_path / "state.dump"
    state.write_bytes(dump("live.msh"))
    ref = {**fidelity.reference(live), "stored_path": "live.msh"}
    live.write_bytes(b"new")
    loader = fidelity.LoadAdapter(problem, problem.adapter, SimpleNamespace(), lambda *_: None)
    with pytest.raises(ValueError, match="bytes changed"):
        loader.load(fidelity.reference(state), [ref])
    assert loader.calls == 0


def authority(tmp_path):
    now = datetime(2030, 1, 1, tzinfo=UTC)
    requirements = {"role": "cold", "source": "synthetic", "caps": fidelity.CAPS}
    phase = {"schema": fidelity.SCHEMA + "-phase", "requirements": requirements, "issued_by": "root",
             "declared_at": (now - timedelta(seconds=30)).isoformat(),
             "deadline_utc": (now + timedelta(seconds=300)).isoformat(), "phase_authorized": True}
    phase_ref = store(tmp_path / "phase.json", phase)
    identity = {"pid": 123, "uid": 1000, "host": "synthetic-host", "start_time_ticks": 900,
                "process_group_id": 123, "session_id": 123, "cwd": str(tmp_path), "cgroup": ["synthetic"], "argv": ["synthetic-worker"]}
    launch_audit = store(tmp_path / "launch-audit.json", {"synthetic": True})
    launch = store(tmp_path / "launch.json", {"root_identity": identity, "launch_audit": launch_audit})
    dispatcher = store(tmp_path / "dispatcher.json", {"authority_verified": True, "status": "active",
                  "reservation_id": "synthetic-reservation", "generation_id": "synthetic-generation", "cpus": [2, 4]})
    tracker = {"schema": fidelity.SCHEMA + "-active-tracker", "phase_kind": fidelity.SCHEMA,
               "status": "running", "role": "cold", "phase": phase_ref, "phase_authorized": True,
               "output": str(tmp_path / "runtime"), "source_commit": "synthetic-commit", "deadline_utc": phase["deadline_utc"],
               "caps": fidelity.CAPS, "checked_at": now.isoformat(),
               "reservation": {"state": "active", "id": "synthetic-reservation", "generation_id": "synthetic-generation", "generation_sha256": "synthetic-fingerprint"},
               "active_dispatcher": dispatcher, "launch": launch, "launch_audit": launch_audit,
               "current_root_identity": identity, "cpus": [2, 4]}
    tracker_ref = store(tmp_path / "tracker.json", tracker)
    admission = {"schema": fidelity.SCHEMA + "-admission", "native_execution_authorized": True,
                 "phase": phase_ref, "requirements": requirements, "active_tracker": tracker_ref}
    prepared = {"phase": phase, "phase_ref": phase_ref, "requirements": requirements, "book": cli.Book(),
                "output": tmp_path / "runtime", "source": {"commit": "synthetic-commit", "files": {}},
                "warm": {"native_process": {**identity, "pid": 122, "start_time_ticks": 899}}}
    return now, identity, tracker, admission, prepared


@pytest.mark.parametrize("fault", ["stale", "inactive", "identity", "affinity", "same_process", "dispatcher", "audit"])
def test_official_active_admission_refuses_real_gate_drift(tmp_path, fault):
    now, identity, tracker, admission, prepared = authority(tmp_path)
    assert cli.admission_contract(admission, prepared, now=now, identity=identity, affinity={123: [2, 4]}) == identity
    # A new Book is required because changing an already observed authority is
    # itself a rejection; these controls exercise the specific intended gate.
    prepared["book"] = cli.Book()
    affinity = {123: [2, 4]}
    if fault == "stale":
        tracker["checked_at"] = (now - timedelta(seconds=301)).isoformat()
    elif fault == "inactive":
        tracker["reservation"]["state"] = "released"
    elif fault == "identity":
        identity = {**identity, "start_time_ticks": 901}
    elif fault == "affinity":
        affinity[123] = [2]
    elif fault == "same_process":
        prepared["warm"]["native_process"] = identity
    elif fault == "dispatcher":
        tracker["active_dispatcher"] = store(tmp_path / "dispatcher-wrong.json", {"authority_verified": False})
    else:
        tracker["launch_audit"] = store(tmp_path / "launch-audit-wrong.json", {"synthetic": "different"})
    admission["active_tracker"] = store(tmp_path / "tracker.json", tracker)
    with pytest.raises(ValueError):
        cli.admission_contract(admission, prepared, now=now, identity=identity, affinity=affinity)


def test_phase_clock_is_exact_utc_and_never_extended(tmp_path):
    now, _identity, _tracker, _admission, prepared = authority(tmp_path)
    assert cli.phase_contract(prepared["phase"], prepared["requirements"], now=now) > now
    expired = copy.deepcopy(prepared["phase"])
    expired["deadline_utc"] = now.isoformat()
    with pytest.raises(ValueError, match="window"):
        cli.phase_contract(expired, prepared["requirements"], now=now)
    foreign = copy.deepcopy(prepared["requirements"])
    foreign["role"] = "warm"
    with pytest.raises(ValueError, match="exact requirements"):
        cli.phase_contract(prepared["phase"], foreign, now=now)


def test_held_wait_does_not_import_native_or_reset_deadline(tmp_path):
    now, _identity, _tracker, _admission, prepared = authority(tmp_path)
    prepared["pending"] = []
    timestamps = iter((now, now + timedelta(seconds=301)))
    sleeps = []
    with pytest.raises(ValueError, match="window"):
        cli.await_admission(prepared, tmp_path / "unissued.json", clock=lambda: next(timestamps), sleep=sleeps.append)
    assert sleeps == [1.0] and prepared.get("native_execution_authorized") is not True


def test_missing_admission_refuses_without_output_or_native_import(tmp_path, monkeypatch, capsys):
    prepared = {"pending": ["root phase declaration unissued"], "phase": None, "output": tmp_path / "runtime"}
    monkeypatch.setattr(cli, "preflight", lambda args: prepared)
    assert cli.main(["--source", str(cli.ROOT), "--bindings", str(tmp_path / "missing.json"), "--role", "warm", "--execute"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "refused" and result["native_imported"] is False
    assert not prepared["output"].exists()


def test_authority_read_cannot_use_a_separately_reopened_json(tmp_path, monkeypatch):
    ref = store(tmp_path / "dispatcher.json", {"authority_verified": False})
    calls = []
    def swapped_reader(path):
        calls.append(path)
        return {"authority_verified": True}
    monkeypatch.setattr(cli.ordinary, "_read_json", swapped_reader)
    assert cli.Book().read(ref) == {"authority_verified": False}
    assert calls == []


def test_authority_hash_and_parsed_bytes_are_one_snapshot(tmp_path, monkeypatch):
    ref = store(tmp_path / "dispatcher.json", {"authority_verified": False})
    original = cli.json.loads
    def replace_on_parse(raw, **kwargs):
        (tmp_path / "dispatcher.json").write_text('{"authority_verified":true}\n')
        return original(raw, **kwargs)
    monkeypatch.setattr(cli.json, "loads", replace_on_parse)
    book = cli.Book()
    assert book.read(ref) == {"authority_verified": False}
    with pytest.raises(ValueError, match="artifact bytes changed"):
        book.stable()


def test_charge_syncs_file_then_parent_before_returning(tmp_path, monkeypatch):
    sync = os.fsync
    trace = []
    def observed(descriptor):
        trace.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
        sync(descriptor)
    monkeypatch.setattr(fidelity.os, "fsync", observed)
    fidelity.write_new(tmp_path / "step-charge.json", {"time_step_calls_charged": 1})
    assert trace == ["file", "directory"]
