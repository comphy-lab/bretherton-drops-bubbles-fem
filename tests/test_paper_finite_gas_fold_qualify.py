"""Solver-denied contracts for saved-candidate admission and template retention."""

from __future__ import annotations

import ast
import importlib.abc
import io
import json
import os
import struct
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from runners import run_paper_finite_gas_fold_qualify as qualify

fold = qualify.fold
SOURCE_INITIALISE = qualify.initialise


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith(("pyoomph.", "_pyoomph")):
            raise AssertionError("native import denied in qualification software tests")


@pytest.fixture(autouse=True)
def no_native():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return fold.regular(path)


def state(reference):
    def string(value):
        raw = value.encode("ascii")
        return struct.pack("<q", len(raw)) + raw
    return (string("pyoomph_dump") + string("0.1.6") + string("global")
            + struct.pack("<dqqqq", 8., 0, 0, 0, 1) + string(reference) + struct.pack("<qq", 1, 3)
            + struct.pack("<qdqdd", 1, .1, 4, .1, float("nan")) + string("EOF_pyoomph"))


class Template:
    def __init__(self, problem, mesh):
        self.problem, self._meshfile, self._template_override, self.remesher = problem, str(mesh), None, None

    def get_problem(self):
        return self.problem

    def _get_template(self):
        return self if self._template_override is None else self._template_override


@pytest.fixture
def loader(tmp_path, monkeypatch):
    """Model only the pinned string-comparison and append-only registration."""
    monkeypatch.chdir(tmp_path)
    live = tmp_path / "live/mesh.msh"
    copy = tmp_path / "portable/mesh.msh"
    live.parent.mkdir()
    copy.parent.mkdir()
    live.write_bytes(b"software mesh fixture\n")
    copy.write_bytes(live.read_bytes())
    p = SimpleNamespace(_domains_to_remesh=set(), _domains_remesh_on_inversion=set())
    original, active = Template(p, live), Template(p, live)
    original._template_override = active
    owner = SimpleNamespace(template=active, _cnt=3)
    original.remesher = active.remesher = owner
    p._meshtemplate_list = [original]
    p._domains_remesh_on_inversion.update((original, active))
    p._meshdict = {name: SimpleNamespace(_templatemesh=active) for name in ("liquid", "gas")}
    calls = []

    def load(value, *, additional_info=None, **options):
        base = Path.cwd() if isinstance(value, io.BytesIO) else Path(value).parent
        payload = value.getvalue() if isinstance(value, io.BytesIO) else Path(value).read_bytes()
        reference = fold.history.serialized_metadata(payload)["meshes"][0][2]
        found = (additional_info or {}).get("exchange_msh_file", {}).get(reference, reference)
        current = os.path.relpath(original._get_template()._meshfile, base)
        if found != current:
            replacement = Template(p, base / found)
            replacement.remesher = owner
            owner.template = replacement
            original._template_override = replacement
        for mesh in p._meshdict.values():
            mesh._templatemesh = original._get_template()
        p._domains_remesh_on_inversion.add(original._get_template())
        calls.append({"found": found, "current": current, "options": options})
        return True

    p.load_state = load
    portable = copy.parent / "state.dump"
    portable.write_bytes(state("mesh.msh"))
    memory = state(os.path.relpath(live, tmp_path))
    return SimpleNamespace(problem=p, live=live, copy=copy, original=original, active=active,
                           owner=owner, portable=portable, memory=memory, calls=calls,
                           bound=[fold.regular(live)])


def test_loader_regression_preserves_registry_and_owner_for_both_paths(loader):
    h = loader
    before = fold.owned_template_signature(h.problem)
    fold.load_with_bound_meshes(h.problem, h.portable, h.bound, quiet=True)
    fold.load_with_bound_meshes(h.problem, h.memory, h.bound, quiet=True)
    assert fold.owned_template_signature(h.problem) == before
    assert h.original._get_template() is h.active and h.owner.template is h.active
    assert len(h.problem._domains_remesh_on_inversion) == 2
    assert all(row["found"] == row["current"] for row in h.calls)
    # The uncorrected portable spelling reproduces the registration defect.
    h.problem.load_state(str(h.portable), quiet=True)
    assert len(h.problem._domains_remesh_on_inversion) == 3
    h.problem.load_state(io.BytesIO(h.memory), quiet=True)
    assert len(h.problem._domains_remesh_on_inversion) == 4


@pytest.mark.parametrize("mutation", ["missing_copy", "corrupt_copy", "symlink_copy", "corrupt_live"])
def test_mesh_exchange_refuses_before_loader_call(loader, mutation):
    h = loader
    if mutation == "missing_copy":
        h.copy.unlink()
    elif mutation == "corrupt_copy":
        h.copy.write_bytes(b"different mesh")
    elif mutation == "symlink_copy":
        h.copy.unlink()
        h.copy.symlink_to(h.live)
    else:
        h.live.write_bytes(b"changed live mesh")
    with pytest.raises(fold.Refusal):
        fold.load_with_bound_meshes(h.problem, h.portable, h.bound)
    assert not h.calls and len(h.problem._domains_remesh_on_inversion) == 2


def test_reference_base_symlink_is_refused(loader, tmp_path):
    link = tmp_path / "portable-alias"
    link.symlink_to(loader.copy.parent, target_is_directory=True)
    with pytest.raises(fold.Refusal, match="regular_mesh_reference_base_required"):
        fold.canonical_mesh_exchange(["mesh.msh"], link, loader.bound)


def test_checkpoint_wrapper_covers_memory_and_portable_and_restores_binding(loader):
    h = loader
    original_load = h.problem.load_state
    with qualify.bound_checkpoint_loads(h.problem, h.bound):
        h.problem.load_state(str(h.portable), ignore_outstep=True)
        h.problem.load_state(io.BytesIO(h.memory), quiet=True)
    assert h.problem.load_state is original_load
    assert h.owner.template is h.active and len(h.problem._domains_remesh_on_inversion) == 2


def test_differences_name_every_changed_leaf_and_summarise_large_values():
    left = {"inversion_templates": [["mesh", 1], ["mesh", 2]], "flags": {"stationary": True},
            "history": "a" * 10000, "gone": 3}
    right = {"inversion_templates": [["mesh", 1], ["mesh", 9], ["mesh", 3]],
             "flags": {"stationary": False}, "history": "b" * 10000, "new": 4}
    rows = fold.signature_mismatches(left, right)
    assert {row["path"] for row in rows} == {"inversion_templates[1][1]", "inversion_templates[2]",
        "flags.stationary", "history", "gone", "new"}
    assert len(json.dumps(rows)) < 1200
    assert next(row for row in rows if row["path"] == "history")["actual"]["encoded_bytes"] == 10002


def test_all_equilibrium_and_time_solve_routes_are_denied():
    calls = []
    p = SimpleNamespace(**{name: lambda: calls.append("native") for name in qualify.SOLVE_APIS})
    denial = qualify.SolveDenial(p)
    for name in qualify.SOLVE_APIS:
        with pytest.raises(fold.Refusal, match="solve_forbidden"):
            getattr(p, name)()
    assert denial.attempts == list(qualify.SOLVE_APIS) and calls == []


def test_execute_cannot_import_native_from_metadata():
    with pytest.raises(fold.Refusal, match="metadata_cannot_authorise"):
        qualify.execute({"native_execution_authorized": False}, None, [], datetime.now(UTC))


@pytest.mark.parametrize("tracking,reactivation,mode", [("fold", None, 0), ("", "pending", 0), ("", None, 1)])
def test_live_bifurcation_tracker_or_nonzero_mode_is_refused(monkeypatch, tracking, reactivation, mode):
    adapter = object.__new__(qualify.QualificationAdapter)
    adapter.problem = SimpleNamespace(get_bifurcation_tracking_mode=lambda: tracking,
        ndof=lambda: fold.N, _bifurcation_reactivation_after_adaptation=reactivation,
        _azimuthal_mode_param_m=SimpleNamespace(value=mode))
    monkeypatch.setattr(fold.Adapter, "check", lambda self: None)
    with pytest.raises(fold.Refusal, match="qualification_active_tracker_or_mode"):
        adapter.check()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_corrupt_exact_parent_refuses_before_environment_inspection(tmp_path, monkeypatch, missing):
    args = SimpleNamespace(source=qualify.ROOT, parent_attempt=tmp_path, out=tmp_path / "out")
    if not missing:
        write(tmp_path / "runtime/run-manifest.json", {"corrupted": True})
    monkeypatch.setattr(qualify.runner, "pyoomph_revision", lambda: pytest.fail("environment reached"))
    with pytest.raises(fold.Refusal):
        qualify.preflight(args)


def test_source_contract_binds_all_source_and_refuses_changed_physics(tmp_path, monkeypatch):
    relative = "problems/scientific.py"
    baseline = b"static_pressure = True\n"
    (tmp_path / "problems").mkdir()
    (tmp_path / relative).write_bytes(baseline)
    for name in (qualify.ENTRYPOINT, qualify.TEST, fold.ENTRYPOINT):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# software source fixture\n")
    current = {"commit": qualify.BASE, "head": qualify.BASE, "clean": True}
    parent = {"component": current, "component_commit": qualify.BASE, "implementation": {}}
    names = [relative, fold.ENTRYPOINT, qualify.ENTRYPOINT, qualify.TEST]
    def git(root, *argv):
        if argv[0] == "ls-tree":
            return (relative + "\n" + fold.ENTRYPOINT + "\n").encode()
        if argv[0] == "ls-files":
            return ("\n".join(names) + "\n").encode()
        if argv[0] == "show":
            if argv[1] == qualify.PRELAUNCH_COMMIT + ":" + fold.ENTRYPOINT:
                return (tmp_path / fold.ENTRYPOINT).read_bytes()
            return baseline
        return b""
    monkeypatch.setattr(qualify, "ROOT", tmp_path)
    monkeypatch.setattr(qualify.runner, "component_revision", lambda root: current)
    monkeypatch.setattr(qualify.diagnostic, "git", git)
    component, implementation = qualify.source_contract(parent, True)
    assert component == current and set(implementation) == set(names)
    (tmp_path / relative).write_text("static_pressure = False\n")
    with pytest.raises(fold.Refusal, match="frozen_candidate_source_changed"):
        qualify.source_contract(parent, True)


def test_inherited_deadline_cannot_be_reset_or_reused_after_expiry():
    lineage = {"deadline_utc": "2026-10-13T21:45:00+00:00"}
    now = datetime(2026, 10, 13, 20, tzinfo=UTC)
    assert qualify.inherited_deadline(lineage["deadline_utc"], lineage, now=now).hour == 21
    with pytest.raises(fold.Refusal, match="deadline_reset_refused"):
        qualify.inherited_deadline("2026-10-14T21:45:00+00:00", lineage, now=now)
    with pytest.raises(fold.Refusal, match="deadline_expired"):
        qualify.inherited_deadline(lineage["deadline_utc"], lineage, now=now + timedelta(hours=2))


@pytest.fixture
def admitted(tmp_path):
    now = datetime(2026, 10, 13, 20, tzinfo=UTC)
    output = tmp_path / "new-output"
    component = {"commit": "software-committed-source"}
    seal = {"schema": qualify.SCHEMA + "-execution-seal", "source_files": {"driver.py": "software-hash"}}
    prepared = {"required_seal": seal, "output": output, "component": component,
                "lineage": {"deadline_utc": "2026-10-13T21:45:00+00:00"}}
    tracker = {"schema": qualify.SCHEMA + "-active-tracker", "status": "running", "phase_kind": qualify.SCHEMA,
        "output": str(output), "source_commit": component["commit"],
        "deadline_utc": prepared["lineage"]["deadline_utc"], "cumulative_work": qualify.DEBITS,
        "new_work_caps": qualify.NEW_CAPS, "reservation": {"state": "active", "id": "software-reservation"},
        "phase_authorized": True, "checked_at": now.isoformat()}
    admission = {"schema": qualify.SCHEMA + "-admission", "native_execution_authorized": True,
                 "execution_seal": seal, "released_parent_process_resumed": False,
                 "active_tracker": write(tmp_path / "tracker.json", tracker)}
    return admission, prepared, tracker, now


def test_external_admission_requires_active_tracker_and_exact_seal(admitted):
    admission, prepared, _tracker, now = admitted
    assert len(qualify.admission_contract(admission, prepared, now=now)) == 1
    assert prepared.get("native_execution_authorized") is None


@pytest.mark.parametrize("key,value", [("status", "terminal"), ("phase_authorized", False),
    ("source_commit", "other-source"), ("cumulative_work", {**qualify.DEBITS, "augmented_solve_calls": 0}),
    ("new_work_caps", {**qualify.NEW_CAPS, "augmented_solve_calls": 1}),
    ("reservation", {"state": "released", "id": "software-reservation"}),
    ("deadline_utc", "2026-10-14T21:45:00+00:00")])
def test_active_tracker_mutations_are_refused_even_with_fresh_digest(admitted, key, value):
    admission, prepared, tracker, now = admitted
    tracker[key] = value
    admission["active_tracker"] = write(Path(admission["active_tracker"]["path"]), tracker)
    with pytest.raises(fold.Refusal, match="active_tracker_differs"):
        qualify.admission_contract(admission, prepared, now=now)


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "stale", "future"])
def test_missing_corrupt_or_old_tracker_is_refused(admitted, mutation):
    admission, prepared, _tracker, now = admitted
    path = Path(admission["active_tracker"]["path"])
    if mutation == "missing":
        path.unlink()
    elif mutation == "corrupt":
        path.write_text("{}")
    elif mutation == "stale":
        now += timedelta(seconds=301)
    else:
        now -= timedelta(seconds=1)
    with pytest.raises(fold.Refusal):
        qualify.admission_contract(admission, prepared, now=now)


def test_an_external_admission_cannot_reset_source_or_parent_identity(admitted):
    admission, prepared, _tracker, now = admitted
    admission["released_parent_process_resumed"] = True
    with pytest.raises(fold.Refusal, match="qualification_admission_differs"):
        qualify.admission_contract(admission, prepared, now=now)


def test_failed_closure_refuses_missing_or_corrupt_candidate_authority(tmp_path):
    runtime = tmp_path / "failed"
    result = {"status": "inconclusive", "numerical_acceptance": False, "work": qualify.DEBITS}
    write(runtime / "result.json", result)
    write(runtime / "fold/case.json", {"case_id": "software-case"})
    (runtime / "null.npy").write_bytes(b"software null fixture")
    write(runtime / "completion.json", {**result, "files": qualify.approach.iii.inventory(runtime)})
    assert qualify.failed_closure(runtime, []) == result
    for path in (runtime / "fold/case.json", runtime / "null.npy"):
        raw = path.read_bytes()
        path.write_bytes(b"corrupted")
        with pytest.raises(fold.Refusal, match="inventory_differs"):
            qualify.failed_closure(runtime, [])
        path.write_bytes(raw)
        path.unlink()
        with pytest.raises(fold.Refusal, match="inventory_differs"):
            qualify.failed_closure(runtime, [])
        path.write_bytes(raw)


def test_entrypoint_has_no_candidate_solve_or_old_execute_route():
    tree = ast.parse(Path(qualify.__file__).read_text())
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert not calls & {"run_candidate", "activate_fold", "canonicalise_continuation", "solve", "run",
                        "activate_bifurcation_tracking", "arclength_continuation", "steady_newton_solve"}
    assert qualify.NEW_CAPS["augmented_solve_calls"] == qualify.NEW_CAPS["completed_augmented_updates"] == 0
    assert qualify.NEW_CAPS["stationary_solve_calls"] == qualify.NEW_CAPS["time_solve_calls"] == 0
    assert qualify.NEW_CAPS["eigenproblem_calls"] == 2


def test_check_mode_produces_no_output_and_cannot_authorise_execution(tmp_path, monkeypatch, capsys):
    output = tmp_path / "must-remain-absent"
    prepared = {"contract": {"new_work_caps": qualify.NEW_CAPS}, "required_seal": {},
                "source_seals_pending": ["external admission pending"]}
    monkeypatch.setattr(qualify, "preflight", lambda args: prepared)
    monkeypatch.setattr(qualify, "execute", lambda *args: pytest.fail("native path invoked"))
    assert qualify.main(["--source", str(qualify.ROOT), "--parent-attempt", str(tmp_path),
                         "--case", str(tmp_path / "case.json"), "--out", str(output), "--check"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["native_imported"] is False and receipt["native_execution_authorized"] is False
    assert not output.exists()


def test_held_process_waits_for_external_active_admission_without_native_import(admitted, tmp_path, monkeypatch):
    admission, prepared, _tracker, now = admitted
    prepared.update(records=[], native_execution_authorized=False)
    path = tmp_path / "external-admission.json"
    checks = []
    monkeypatch.setattr(fold, "stable_inputs", lambda value: checks.append(value["native_execution_authorized"]))
    sleeps = []
    def deliver(seconds):
        sleeps.append(seconds)
        assert prepared["native_execution_authorized"] is False
        write(path, admission)
    qualify.await_active_admission(prepared, path, prepared["lineage"]["deadline_utc"],
                                   clock=lambda: now, sleep=deliver)
    assert sleeps == [1.0] and checks and not any(checks)
    assert prepared["native_execution_authorized"] is True
    assert prepared["active_admission"] == fold.regular(path)
    assert len(prepared["records"]) == 2


def test_held_wait_expires_at_inherited_deadline_without_authorising(admitted, tmp_path):
    _admission, prepared, _tracker, now = admitted
    prepared.update(records=[], native_execution_authorized=False)
    clock = [now.replace(hour=21, minute=44, second=59)]
    def advance(seconds):
        clock[0] += timedelta(seconds=seconds)
    with pytest.raises(fold.Refusal, match="inherited_qualification_deadline_expired"):
        qualify.await_active_admission(prepared, tmp_path / "absent.json", prepared["lineage"]["deadline_utc"],
                                       clock=lambda: clock[0], sleep=advance)
    assert prepared["native_execution_authorized"] is False
    assert clock[0].isoformat() == prepared["lineage"]["deadline_utc"]


def test_source_change_during_held_wait_refuses_before_native_admission(admitted, tmp_path, monkeypatch):
    admission, prepared, _tracker, now = admitted
    prepared.update(records=[], native_execution_authorized=False)
    path = tmp_path / "admission.json"
    write(path, admission)
    def changed(value):
        raise fold.Refusal("software-source-changed")
    monkeypatch.setattr(fold, "stable_inputs", changed)
    with pytest.raises(fold.Refusal, match="software-source-changed"):
        qualify.await_active_admission(prepared, path, prepared["lineage"]["deadline_utc"], clock=lambda: now)
    assert prepared["native_execution_authorized"] is False


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    """Exercise Python custody/cleanup with primitive objects; no native bindings exist."""
    output = tmp_path / "software-runtime"
    output.mkdir()
    deadline = datetime.now(UTC) + timedelta(minutes=5)
    trace = []
    faults = {}

    class Core:
        def __init__(self):
            self.enabled = False
            self.restores = 0

        def get_detect_inverted_elements(self):
            return self.enabled

        def set_detect_inverted_elements(self, enabled):
            trace.append(("detection", enabled))
            if not enabled:
                self.restores += 1
                if faults.get("restore_detection") == self.restores:
                    raise RuntimeError("software inversion cleanup failure")
            self.enabled = enabled

    class Problem:
        def __init__(self, label):
            self.label = label
            self._released = False

        def release(self):
            trace.append(("release", self.label))
            self._released = True  # Exact native ordering: this marker cannot prove successful cleanup.
            if faults.get("release") == self.label:
                raise RuntimeError("software teardown failure")

    core = Core()
    sessions = []
    def initialise(session, problem_type, selected, selected_output, native_core, *args):
        label = "cold" if selected_output.name == "fresh-fold-cold" else "same"
        trace.append(("initialise", label, native_core.enabled))
        assert native_core.enabled is True
        if faults.get("construct") == label:
            raise RuntimeError("software construction failure")
        session.problem = Problem(label)
        session.adapter = SimpleNamespace(work={name: 0 for name in qualify.DEBITS}, check=lambda: None)
        sessions.append(session)
        if faults.get("initialise") == label:
            raise RuntimeError("software load failure")

    def same(adapter, prepared, selected_output, arguments, result):
        assert core.enabled is True
        adapter.work["eigenproblem_calls"] = 2
        result.update(status="candidate_ready_for_cold_qualification", returned_real_null_matched=True,
                      cold_restart={"software": True}, restoration=[{"software_restoration": True}],
                      same_instance_baseline_restored=True)

    warm = {"actual_pins": {"software": True}, "signature": {"software": True}}
    prepared = {"native_execution_authorized": True, "lineage": {"deadline_utc": deadline.isoformat()},
                "case": {"software": True}, "candidate": {"warm": warm}}
    monkeypatch.setattr(qualify, "native_bindings", lambda value: (core, object, object, object, object, object))
    monkeypatch.setattr(qualify, "initialise", initialise)
    monkeypatch.setattr(qualify, "qualify_same_instance", same)
    monkeypatch.setattr(fold, "stable_inputs", lambda value: None)
    monkeypatch.setattr(fold, "inspect_inversion_templates", lambda *args: {"software": True})
    monkeypatch.setattr(qualify.approach.iii, "physical_signature", lambda value: value)
    def sample(*args):
        trace.append(("cold-sample", core.enabled))
        assert core.enabled is True
        return {**warm, "metrics": {"software": True}}
    monkeypatch.setattr(qualify.remesh, "sample", sample)
    monkeypatch.setattr(qualify.remesh, "canonical_checks", lambda *args: {name: True for name in qualify.modes.GATES})
    return SimpleNamespace(output=output, deadline=deadline, prepared=prepared, core=core, trace=trace,
                           faults=faults, sessions=sessions)


def assert_spent_searches(result):
    assert result["work"] == {**qualify.DEBITS, "augmented_solve_calls": 0, "completed_augmented_updates": 0,
                             "eigenproblem_calls": 2, "stationary_solve_calls": 0, "time_solve_calls": 0}
    assert result["cumulative_work"] == {**qualify.DEBITS, "eigenproblem_calls": 2}


def test_same_and_cold_detection_are_scoped_and_exactly_restored(orchestration):
    h = orchestration
    result = qualify.execute(h.prepared, h.output, [], h.deadline)
    assert result["status"] == "complete" and result["same_instance_released"] and result["cold_released"]
    assert h.trace == [("detection", True), ("initialise", "same", True), ("release", "same"),
                      ("detection", False), ("detection", True), ("initialise", "cold", True),
                      ("cold-sample", True), ("release", "cold"), ("detection", False)]
    assert h.core.enabled is False and result["inversion_detection_global_restored"] is True
    assert_spent_searches(result)
    assert json.loads((h.output / "phase-receipts/native-phase-return.json").read_text())["result"]["work"] == result["work"]
    assert result["old_unserialized_continuation_buffer_restored"] is False


@pytest.mark.parametrize("label", ["same", "cold"])
def test_load_and_teardown_failures_keep_both_and_never_prove_release(orchestration, label):
    h = orchestration
    h.faults.update(initialise=label, release=label)
    result = qualify.execute(h.prepared, h.output, [], h.deadline)
    assert result["status"] == "inconclusive" and result["numerical_acceptance"] is False
    initiating, cleanup = ("refusal", "release_failure") if label == "same" else ("cold_refusal", "cold_release_failure")
    release_label = "same_instance" if label == "same" else "cold"
    assert result[initiating]["error"] == "software load failure"
    assert result[cleanup]["error"] == "software teardown failure"
    assert result[release_label + "_released"] is False
    assert result[release_label + "_cleanup"] == {"problem_constructed": True, "release_attempted": True,
                                                  "release_completed": False, "verified": False}
    assert h.sessions[-1].problem._released is True
    assert h.core.enabled is False
    if label == "cold":
        assert_spent_searches(result)


def test_missing_constructed_handle_does_not_claim_verified_release(orchestration):
    h = orchestration
    h.faults["construct"] = "same"
    result = qualify.execute(h.prepared, h.output, [], h.deadline)
    assert result["same_instance_released"] is False
    assert result["same_instance_cleanup"] == {"problem_constructed": False, "release_attempted": False,
                                                "release_completed": False, "verified": False}


def test_actual_initialise_retains_handle_before_fallible_configuration(orchestration, monkeypatch):
    h = orchestration
    problem = SimpleNamespace(**{name: lambda *args, **kwargs: None for name in qualify.SOLVE_APIS})
    def refuse(*args):
        raise RuntimeError("software output configuration failure")
    problem.set_output_directory = refuse
    monkeypatch.setattr(fold, "build_bridged", lambda *args: problem)
    session = qualify.NativeSession()
    with pytest.raises(RuntimeError, match="output configuration failure"):
        SOURCE_INITIALISE(session, object, h.prepared, h.output, h.core, object, object, h.deadline)
    assert session.problem is problem and session.adapter is None


@pytest.mark.parametrize("lane", ["same", "cold"])
def test_saved_candidate_initialise_skips_default_adaptation(orchestration, monkeypatch, lane):
    h = orchestration
    names = set(qualify.SOLVE_APIS) | {
        "set_output_directory", "set_linear_solver", "set_c_compiler",
        "setup_for_stability_analysis", "set_eigensolver", "force_remesh",
        "remesh_if_necessary", "remesh_handler_during_continuation", "_adapt", "adapt",
        "_adapt_with_interfacial_errors", "custom_adapt", "actions_before_remeshing",
        "actions_before_adapt", "actions_after_remeshing", "actions_after_adapt",
        "_perform_pending_remesh",
    }
    problem = SimpleNamespace(**{name: lambda *args, **kwargs: None for name in names},
                              initial_adaption_steps=None, max_refinement_level=8,
                              project_initial_conditions=False, _meshdict={})
    def initialise():
        # Pinned Problem.initialise uses the default refinement limit when unset.
        if problem.initial_adaption_steps is None:
            problem.initial_adaption_steps = problem.max_refinement_level
        for _ in range(problem.initial_adaption_steps):
            problem._adapt()
        raise RuntimeError("native-free initialisation boundary")
    problem.initialise = initialise
    def adapter(problem, *args):
        guard = fold.RemeshGuard(problem)
        guard.install()
        return SimpleNamespace(mesh_guard=guard)
    monkeypatch.setattr(fold, "build_bridged", lambda *args: problem)
    monkeypatch.setattr(qualify, "QualificationAdapter", adapter)
    session = qualify.NativeSession()
    with pytest.raises(RuntimeError, match="native-free initialisation boundary"):
        SOURCE_INITIALISE(session, object, h.prepared, h.output / lane, h.core,
                          object, object, h.deadline)
    assert problem.initial_adaption_steps == 0 and session.adapter.mesh_guard.requests == 0
    with pytest.raises(fold.Refusal, match="physical_remesh_or_adapt_request"):
        problem._adapt()
    assert session.adapter.mesh_guard.requests == 1


def test_cold_directory_failure_retains_durable_search_debits(orchestration):
    h = orchestration
    (h.output / "fresh-fold-cold").mkdir()
    result = qualify.execute(h.prepared, h.output, [], h.deadline)
    assert result["status"] == "inconclusive" and result["cold_refusal"]["error_type"] == "FileExistsError"
    assert result["same_instance_released"] is True and result["same_instance_baseline_restored"] is True
    assert result["restoration"] == [{"software_restoration": True}]
    assert_spent_searches(result)
    assert json.loads((h.output / "phase-receipts/same-instance-before-release.json").read_text())["result"]["work"] == result["work"]


def test_final_input_change_retains_search_and_cleanup_evidence(orchestration, monkeypatch):
    h = orchestration
    calls = []
    def stable(value):
        calls.append(value)
        if len(calls) == 2:
            raise fold.Refusal("software-bound-input-changed")
    monkeypatch.setattr(fold, "stable_inputs", stable)
    result = qualify.execute(h.prepared, h.output, [], h.deadline)
    assert result["status"] == "inconclusive" and result["numerical_acceptance"] is False
    assert result["final_input_refusal"]["label"] == "software-bound-input-changed"
    assert result["same_instance_released"] and result["cold_released"]
    assert_spent_searches(result)


@pytest.mark.parametrize("lane", [1, 2])
def test_global_cleanup_failure_keeps_known_work_and_reports_unverified_policy(orchestration, lane):
    h = orchestration
    h.faults["restore_detection"] = lane
    result = qualify.execute(h.prepared, h.output, [], h.deadline)
    label = "same_instance" if lane == 1 else "cold"
    assert result["status"] == "inconclusive" and result["numerical_acceptance"] is False
    assert result["inversion_detection_global_restored"] is False
    assert result[label + "_inversion_restoration_failure"]["error"] == "software inversion cleanup failure"
    assert_spent_searches(result)


def test_postsearch_receipt_failure_keeps_durable_work_and_prevents_cold(orchestration, monkeypatch):
    h = orchestration
    original = qualify.write_json
    def write_result(path, value):
        if path.name == "same-instance-after-release.json":
            raise OSError("software result sink unavailable")
        return original(path, value)
    monkeypatch.setattr(qualify, "write_json", write_result)
    result = qualify.execute(h.prepared, h.output, [], h.deadline)
    assert result["status"] == "inconclusive" and result["receipt_write_failure"]["error_type"] == "OSError"
    assert not any(row[0] == "cold-sample" for row in h.trace)
    assert_spent_searches(result)
    assert json.loads((h.output / "phase-receipts/same-instance-before-release.json").read_text())["result"]["work"] == result["work"]


@pytest.mark.parametrize("change", ["detect", "threshold", "registration"])
def test_operation_check_refuses_changed_inversion_policy(monkeypatch, change):
    p = SimpleNamespace(get_bifurcation_tracking_mode=lambda: "", ndof=lambda: fold.N,
        _bifurcation_reactivation_after_adaptation=None, _azimuthal_mode_param_m=SimpleNamespace(value=0),
        get_global_parameter=lambda name: SimpleNamespace(value={"Ca_in": -0.0038, "Bo": 1.56}[name]),
        _inversion_remesh_threshold=1 if change == "threshold" else 0)
    adapter = object.__new__(qualify.QualificationAdapter)
    adapter.problem = p
    adapter.prepared = {"case": {"physics": {"ca_in": -.0038, "bond": 1.56}}}
    adapter.work = {name: 0 for name in qualify.DEBITS}
    adapter.solve_denial = SimpleNamespace(attempts=[])
    adapter.core = SimpleNamespace(get_detect_inverted_elements=lambda: change != "detect")
    adapter.template_policy = {"software_owned_template": 1}
    monkeypatch.setattr(fold.Adapter, "check", lambda self: None)
    monkeypatch.setattr(fold, "owned_template_signature", lambda problem: {
        "software_owned_template": 2 if change == "registration" else 1})
    with pytest.raises(fold.Refusal, match="qualification_(inversion_detection_policy|template_registration)_changed"):
        adapter.check()


def test_main_supplies_held_progress_before_admission_and_preserves_outer_fallback(tmp_path, monkeypatch):
    output = tmp_path / "software-output"
    case = tmp_path / "case.json"
    write(case, {"software": True})
    deadline = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    prepared = {"output": output, "case": {"software": True}, "contract": {}, "records": [],
                "implementation": {}, "lineage": {"deadline_utc": deadline}, "native_execution_authorized": False}
    monkeypatch.setattr(qualify, "preflight", lambda args: prepared)
    monkeypatch.setattr(qualify, "manifest", lambda *args: {"software": True})
    def active(value, path, deadline_text):
        rows = [json.loads(row) for row in (output / "progress.jsonl").read_text().splitlines()]
        assert rows[0]["event"] == "prepared" and rows[0]["native_execution_authorized"] is False
        assert rows[0]["active_admission_pending"] is True
        assert (output / "preparation.json").is_file()
        assert value["native_execution_authorized"] is False and deadline_text == deadline
        value.update(native_execution_authorized=True, active_admission={"software": True})
    monkeypatch.setattr(qualify, "await_active_admission", active)
    def unexpected(*args, result):
        assert prepared["native_execution_authorized"] is True
        qualify.update_work(result, SimpleNamespace(work={**qualify.DEBITS, "augmented_solve_calls": 0,
            "completed_augmented_updates": 0, "eigenproblem_calls": 2}))
        result.update(restoration=[{"software": True}], same_instance_released=True)
        raise RuntimeError("software late escape")
    monkeypatch.setattr(qualify, "execute", unexpected)
    assert qualify.main(["--source", str(qualify.ROOT), "--parent-attempt", str(tmp_path), "--case", str(case),
                         "--out", str(output), "--deadline", deadline, "--admission", str(tmp_path / "active.json")]) == 1
    result = json.loads((output / "completion.json").read_text())
    assert result["refusal"]["error"] == "software late escape" and result["numerical_acceptance"] is False
    assert result["same_instance_released"] is True and result["restoration"] == [{"software": True}]
    assert_spent_searches(result)


@pytest.fixture
def recovery_declaration(tmp_path):
    """A synthetic future phase for checking predicates; never an operational declaration."""
    now = datetime(2099, 1, 1, 12, tzinfo=UTC)
    predecessor = {"terminal_checked_at": "2026-10-13T21:47:41.605879+00:00",
                   "archive_verified_at": "2026-10-13T21:47:47.208982Z",
                   "original_deadline_utc": qualify.ORIGINAL_QUALIFICATION_DEADLINE,
                   "status": "failed", "new_work": {name: 0 for name in qualify.NEW_CAPS}}
    requirements = {"schema": qualify.RECOVERY_KIND + "-declaration", "phase_kind": qualify.RECOVERY_KIND,
                    "predecessor": predecessor, "source_commit": "software-source", "source_files": {"driver": "software"},
                    "output": "software-new-owner-runtime",
                    "inputs": [{"path": "software-only", "sha256": "software"}],
                    "prior_cumulative_work": qualify.DEBITS, "new_work_caps": qualify.NEW_CAPS,
                    "aggregate_caps": qualify.AGGREGATE_CAPS,
                    "spectral_request": {"azimuthal_m": 0, "shifts": [-.001, .001], "neigen": 8, "max_retries": 0},
                    "maximum_wall_seconds": 2700, "cold_load_required": True,
                    "original_failures_and_acceptance_preserved": True,
                    "released_parent_process_resumed": False, "old_unserialized_continuation_buffer_restored": False}
    declaration = {**requirements, "issued_by": "root/Kelvin", "phase_authorized": True,
                   "declared_at": (now - timedelta(seconds=1)).isoformat(),
                   "deadline_utc": (now + timedelta(seconds=2699)).isoformat()}
    reference = write(tmp_path / "software-recovery-declaration.json", declaration)
    lineage = {"deadline_utc": qualify.ORIGINAL_QUALIFICATION_DEADLINE, "cumulative_work": qualify.DEBITS}
    return declaration, reference, requirements, lineage, now


def test_recovery_is_a_separate_prospective_clock_with_unchanged_work(recovery_declaration):
    declaration, reference, requirements, lineage, now = recovery_declaration
    recovered = qualify.recovery_phase_contract(declaration, reference, requirements, lineage, now=now)
    assert lineage["deadline_utc"] == qualify.ORIGINAL_QUALIFICATION_DEADLINE
    assert recovered["original_qualification_deadline_utc"] == lineage["deadline_utc"]
    assert recovered["deadline_utc"] == recovered["prospective_deadline_utc"] == declaration["deadline_utc"]
    assert recovered["phase_kind"] == qualify.RECOVERY_KIND and recovered["recovery_phase"] == reference
    assert recovered["source_alone_authorizes_recovery"] is False
    assert recovered["released_parent_process_resumed"] is False
    assert recovered["predecessor_new_work"] == {name: 0 for name in qualify.NEW_CAPS}
    result = qualify.new_result({"phase_kind": qualify.RECOVERY_KIND, "lineage": recovered})
    assert result["schema"] == qualify.RECOVERY_KIND
    assert result["original_qualification_deadline_utc"] == qualify.ORIGINAL_QUALIFICATION_DEADLINE
    assert result["prospective_deadline_utc"] == declaration["deadline_utc"]
    assert result["expired_predecessor"]["status"] == "failed"
    assert result["old_unserialized_continuation_buffer_restored"] is False


@pytest.mark.parametrize("key,value", [
    ("phase_kind", qualify.SCHEMA), ("issued_by", "not-root"), ("phase_authorized", False),
    ("source_commit", "unreviewed-source"), ("source_files", {"driver": "changed"}), ("output", "another-runtime"),
    ("inputs", []), ("prior_cumulative_work", {**qualify.DEBITS, "eigenproblem_calls": 1}),
    ("new_work_caps", {**qualify.NEW_CAPS, "augmented_solve_calls": 1}),
    ("aggregate_caps", {**qualify.AGGREGATE_CAPS, "eigenproblem_calls": 3}),
    ("spectral_request", {"azimuthal_m": 1, "shifts": [-.001, .001], "neigen": 8, "max_retries": 0}),
    ("maximum_wall_seconds", 2701), ("released_parent_process_resumed", True),
    ("old_unserialized_continuation_buffer_restored", True),
])
def test_recovery_rejects_work_source_inputs_caps_or_owner_changes(recovery_declaration, key, value):
    declaration, reference, requirements, lineage, now = recovery_declaration
    declaration = {**declaration, key: value}
    with pytest.raises(fold.Refusal, match="prospective_recovery_declaration_differs"):
        qualify.recovery_phase_contract(declaration, reference, requirements, lineage, now=now)


@pytest.mark.parametrize("timing", ["before_terminal", "before_archive", "future_declaration", "expired", "too_long", "not_utc"])
def test_recovery_clock_requires_later_root_declaration_and_a_finite_future_window(recovery_declaration, timing):
    declaration, reference, requirements, lineage, now = recovery_declaration
    declaration = dict(declaration)
    if timing == "before_terminal":
        now = datetime(2026, 10, 13, 22, tzinfo=UTC)
        declaration.update(declared_at="2026-10-13T21:47:40+00:00", deadline_utc="2026-10-13T22:30:00+00:00")
    elif timing == "before_archive":
        now = datetime(2026, 10, 13, 22, tzinfo=UTC)
        declaration.update(declared_at="2026-10-13T21:47:45+00:00", deadline_utc="2026-10-13T22:30:00+00:00")
    elif timing == "future_declaration":
        declaration["declared_at"] = (now + timedelta(seconds=1)).isoformat()
    elif timing == "expired":
        now += timedelta(seconds=2700)
    elif timing == "too_long":
        declaration["deadline_utc"] = (now + timedelta(seconds=2700)).isoformat()
    else:
        declaration["deadline_utc"] = "2099-01-01T12:30:00+01:00"
    with pytest.raises(fold.Refusal, match="recovery_(phase_not_prospective_or_finite|deadline_utc_required)"):
        qualify.recovery_phase_contract(declaration, reference, requirements, lineage, now=now)


def test_default_expired_phase_cannot_acquire_the_recovery_clock(recovery_declaration):
    declaration, reference, requirements, lineage, now = recovery_declaration
    with pytest.raises(fold.Refusal, match="qualification_deadline_reset_refused"):
        qualify.inherited_deadline(declaration["deadline_utc"], lineage, now=now)
    with pytest.raises(fold.Refusal, match="inherited_qualification_deadline_expired"):
        qualify.inherited_deadline(lineage["deadline_utc"], lineage, now=now)
    changed = {**lineage, "deadline_utc": "2026-10-13T22:45:00+00:00"}
    with pytest.raises(fold.Refusal, match="original_qualification_deadline_changed"):
        qualify.recovery_phase_contract(declaration, reference, requirements, changed, now=now)


def test_missing_or_mutated_exact_predecessor_receipt_refuses_before_new_clock(tmp_path, monkeypatch):
    # Source paths remain inside a synthetic project; only an exact archive digest can open the bundle.
    source = tmp_path / "project/repos/recovery"
    source.mkdir(parents=True)
    monkeypatch.setattr(qualify, "ROOT", source)
    with pytest.raises(fold.Refusal, match="regular_authority_required"):
        qualify.predecessor_contract({}, [])
    archive = source.parents[1] / "scratch/figure3/stability/full-comparison-2026-10-12" / \
        "recovery-aa922-20261013/qualification-prelaunch-archive.json"
    write(archive, {"source": str(tmp_path / "old")})
    with pytest.raises(fold.Refusal, match="exact_prelaunch_archive_required"):
        qualify.predecessor_contract({}, [])
    monkeypatch.setattr(qualify, "PRELAUNCH_ARCHIVE_SHA", fold.sha(archive))
    with pytest.raises(fold.Refusal, match="regular_authority_required"):
        qualify.predecessor_contract({}, [])
    write(tmp_path / "old/attempts/attempt-0001/execution-terminal.json", {"native_execution_started": True})
    with pytest.raises(fold.Refusal, match="exact_zero_work_prelaunch_terminal_required"):
        qualify.predecessor_contract({}, [])


@pytest.mark.parametrize("key,value", [("new_eigen_calls", 1), ("new_completed_updates", 1),
    ("new_physical_remeshes", 1), ("native_execution_started", True), ("owned_processes_terminal", False)])
def test_predecessor_semantics_refuse_spent_or_unsettled_work_even_with_bound_test_digest(tmp_path, monkeypatch, key, value):
    source = tmp_path / "project/repos/recovery"
    source.mkdir(parents=True)
    monkeypatch.setattr(qualify, "ROOT", source)
    old_root = tmp_path / "old"
    archive = source.parents[1] / "scratch/figure3/stability/full-comparison-2026-10-12" / \
        "recovery-aa922-20261013/qualification-prelaunch-archive.json"
    write(archive, {"source": str(old_root)})
    monkeypatch.setattr(qualify, "PRELAUNCH_ARCHIVE_SHA", fold.sha(archive))
    terminal = {"schema": "stone-qualification-prelaunch-terminal-v1", "status": "failed",
        "source_commit": qualify.PRELAUNCH_COMMIT, "original_deadline_utc": qualify.ORIGINAL_QUALIFICATION_DEADLINE,
        "owned_processes_terminal": True, "native_pid_absent": True, "matching_processes": [],
        "unit": {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead", "MainPID": "0", "ControlGroup": ""},
        "scientific_result_accepted": False, "numerical_acceptance": False, "native_execution_performed": False,
        "native_execution_started": False, "numerical_work_started": False, "prior_failures_or_acceptance_changed": False,
        "new_completed_updates": 0, "new_physical_remeshes": 0, "new_eigen_calls": 0,
        "cumulative_prior_augmented_calls": 2, "cumulative_prior_augmented_updates": 6}
    terminal[key] = value
    record = write(old_root / "attempts/attempt-0001/execution-terminal.json", terminal)
    monkeypatch.setattr(qualify, "PRELAUNCH_TERMINAL_SHA", record["sha256"])
    with pytest.raises(fold.Refusal, match="prelaunch_not_proved_terminal_with_zero_work"):
        qualify.predecessor_contract({}, [])


@pytest.mark.parametrize("mutation", ["old_schema", "wrong_reference", "wrong_original_clock", "wrong_declared_at"])
def test_recovery_active_admission_cannot_reuse_the_old_phase_or_another_declaration(admitted, mutation):
    admission, prepared, tracker, now = admitted
    recovery_ref = {"path": "software-root-declaration", "sha256": "software-exact-digest"}
    prepared["phase_kind"] = qualify.RECOVERY_KIND
    prepared["lineage"].update(recovery_phase=recovery_ref,
        original_qualification_deadline_utc=qualify.ORIGINAL_QUALIFICATION_DEADLINE,
        prospective_declared_at=now.isoformat())
    expected = {"recovery_phase": recovery_ref, "original_deadline_utc": qualify.ORIGINAL_QUALIFICATION_DEADLINE,
                "prospective_declared_at": now.isoformat()}
    admission.update(schema=qualify.RECOVERY_KIND + "-admission", **expected)
    tracker.update(schema=qualify.RECOVERY_KIND + "-active-tracker", phase_kind=qualify.RECOVERY_KIND, **expected)
    if mutation == "old_schema":
        admission["schema"] = qualify.SCHEMA + "-admission"
    elif mutation == "wrong_reference":
        admission["recovery_phase"] = {**recovery_ref, "sha256": "forged"}
    elif mutation == "wrong_original_clock":
        tracker["original_deadline_utc"] = "2026-10-13T22:45:00+00:00"
    else:
        tracker["prospective_declared_at"] = (now - timedelta(seconds=1)).isoformat()
    admission["active_tracker"] = write(Path(admission["active_tracker"]["path"]), tracker)
    with pytest.raises(fold.Refusal, match="(qualification_admission|recovery_active_admission_binding)_differs"):
        qualify.admission_contract(admission, prepared, now=now)
