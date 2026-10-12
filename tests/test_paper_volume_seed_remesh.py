"""Seed-quality runner contracts with synthetic streams and fake dispatch only."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from problems import paper_bubble as model
from runners import run_paper_case as runner
from runners import run_paper_volume_seed_remesh as driver

ROOT = Path(__file__).resolve().parents[1]


class Hooks:
    """Fake of the pinned no-op hook interface; no native module is loaded."""

    def __init__(self):
        self._problem = None

    def get_problem(self):
        return self._problem

    def actions_after_initialise(self):
        pass

    def actions_before_newton_solve(self):
        pass

    def actions_after_newton_step(self):
        pass

    def actions_after_newton_solve(self):
        pass

    def actions_before_remeshing(self, remeshers):
        pass

    def actions_after_remeshing(self):
        pass


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("software tests must not construct native problems")

    monkeypatch.setattr(model, "_native_problem_types", forbidden)
    monkeypatch.setattr(model, "build_problem", forbidden)
    assert not any(name.startswith("pyoomph") for name in sys.modules)
    package = ModuleType("pyoomph")
    generic = ModuleType("pyoomph.generic")
    problem = ModuleType("pyoomph.generic.problem")
    problem.GenericProblemHooks = Hooks
    for name, module in (
        ("pyoomph", package),
        ("pyoomph.generic", generic),
        ("pyoomph.generic.problem", problem),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def fake_dump(reference, state, *, endtime=8.0, marker="init", remeshes=0):
    def integer(value):
        return struct.pack("<q", value)

    def string(value):
        return integer(len(value)) + value.encode("ascii")

    payload = json.dumps(state, sort_keys=True).encode()
    return (
        string("pyoomph_dump")
        + string("0.1.6")
        + string("global")
        + struct.pack("<d", 8.0)
        + integer(9)
        + integer(0)
        + integer(6)
        + integer(1)
        + string(reference)
        + integer(1)
        + integer(remeshes)
        + integer(2)
        + struct.pack("<dd", 1, 1)
        + integer(15)
        + struct.pack("<dd", 4, float("nan") if endtime is None else endtime)
        + integer(len(payload))
        + payload
        + runner._bc_cache_trailer(marker)
    )


class FakeProblem:
    """Bounded fake Newton/remesh events and complete synthetic state rollback."""

    def __init__(self, mesh, output, *, needed=2, failures=(), quality=0, remesh_fault=None):
        self.reference, self.output = str(mesh), output
        self.needed = list(needed) if isinstance(needed, list) else needed
        self.failures = list(failures)
        self.quality, self.remesh_fault = quality, remesh_fault
        self.max_newton_iterations = 40
        self.newton_relaxation_factor = 1.0
        self.newton_solver_tolerance = 1e-8
        self.inversion_remesh_max_retries = 3
        self.eigen_data_in_states = False
        self._last_bc_setting = "init"
        self._first_step = False
        self._taken_already_an_unsteady_step = True
        self._last_step_was_stationary = False
        self._run_statement_endtime = 8.0
        self._loaded_run_statement_endtime = None
        self._domains_to_remesh = set()
        self._remesh_requested_inside_solve = False
        self._hooks = []
        self.fields, self.pins = [1.0, 2.0], [False, True]
        self.parameters = {"Bo": SimpleNamespace(value=0.0), "Ca_in": SimpleNamespace(value=0.02)}
        self.residual, self.volume, self.speed = 3e-4, 4.0, 0.01
        self.steady = False
        self.native_updates = self.physical_remeshes = self.loads = self.remesher_counter = 0
        self.trials = []

    def __iadd__(self, hook):
        hook._problem = self
        self._hooks.append(hook)
        hook.actions_after_initialise()
        return self

    def actions_before_newton_step(self):
        pass

    def ntime_stepper(self):
        return 1

    def time_stepper_pt(self, index):
        return SimpleNamespace(
            is_steady=lambda: self.steady,
            make_steady=lambda: setattr(self, "steady", True),
            undo_make_steady=lambda: setattr(self, "steady", False),
        )

    def get_global_parameter(self, name):
        return self.parameters[name]

    def get_current_dofs(self):
        return self.fields, self.pins

    def interface_polyline(self):
        return [0.0, 1.0], [0.0, self.fields[0]]

    def observable(self, name, domain):
        assert (name, domain) == ("volume", "gas")
        return self.volume

    def frame_speed(self):
        return self.speed

    def _get_inversion_reports(self):
        return 0

    def signature(self):
        state = {
            "fields": self.fields,
            "pins": self.pins,
            "parameters": {key: value.value for key, value in self.parameters.items()},
            "volume": self.volume,
            "frame_speed": self.speed,
            "bc_cache_marker": self._last_bc_setting,
            "time": 8.0,
            "layout": {"dof_names": ["liquid/velocity_phi", "gas/velocity_phi"]},
        }
        state["serialized_state"] = runner.snapshot_bc_cache_signature(
            self._snapshot_state(), mesh_reference_base=Path.cwd()
        )
        return copy.deepcopy(state)

    def _snapshot_state(self):
        state = {
            "fields": self.fields,
            "pins": self.pins,
            "residual": self.residual,
            "volume": self.volume,
            "speed": self.speed,
            "parameters": {key: value.value for key, value in self.parameters.items()},
        }
        return fake_dump(
            self.reference,
            state,
            endtime=self._run_statement_endtime,
            marker=self._last_bc_setting,
            remeshes=self.remesher_counter,
        )

    def load_state(self, value, **kwargs):
        raw = value.read() if hasattr(value, "read") else Path(value).read_bytes()
        metadata = driver.history.serialized_metadata(raw)
        offset = metadata["endtime_offset"] + 8
        length = struct.unpack("<q", raw[offset : offset + 8])[0]
        state = json.loads(raw[offset + 8 : offset + 8 + length])
        self.reference = metadata["meshes"][0][2]
        self.fields, self.pins = state["fields"], state["pins"]
        self.residual, self.volume, self.speed = state["residual"], state["volume"], state["speed"]
        for key, number in state["parameters"].items():
            self.parameters[key].value = number
        self._last_bc_setting = runner.snapshot_bc_cache_signature(raw)["bc_cache_marker"]
        self._loaded_run_statement_endtime = metadata["endtime"]
        end = metadata["meshes"][0][1]
        self.remesher_counter = struct.unpack("<q", raw[end + 8 : end + 16])[0]
        self.loads += 1
        for hook in self._hooks:
            hook.actions_after_remeshing()

    def _restore_state(self, raw):
        self.load_state(io.BytesIO(raw))

    def invalidate_cached_mesh_data(self):
        pass

    def force_remesh(self):
        if self.remesh_fault == "absent":
            return False
        for hook in self._hooks:
            hook.actions_before_remeshing(["one_shared_two_phase_template"])
        if self.remesh_fault == "incomplete":
            self.fields = [9.0, 9.0]
            raise RuntimeError("fake mesh generation failure")
        self.physical_remeshes += 1
        self.remesher_counter += 1
        mesh = self.output / "solver" / f"quality-{self.physical_remeshes}.msh"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(f"synthetic projected mesh {self.physical_remeshes}".encode())
        self.reference, self.fields, self.residual = str(mesh), [1.5, 2.5], 1e-3
        if self.remesh_fault == "forcing":
            self.parameters["Bo"].value = 99
        for hook in self._hooks:
            hook.actions_after_remeshing()
        return True

    def solve(self, **kwargs):
        self.max_newton_iterations = kwargs["max_newton_iterations"]
        self.newton_relaxation_factor = kwargs["newton_relaxation_factor"]
        for _ in range(self.inversion_remesh_max_retries + 1):
            checkpoint = self._snapshot_state()
            for hook in self._hooks:
                hook.actions_before_newton_solve()
            self.trials.append((self.newton_relaxation_factor, self.max_newton_iterations))
            failure = self.failures.pop(0) if self.failures else None
            needed = (
                1
                if failure
                else (self.needed.pop(0) if isinstance(self.needed, list) else self.needed)
            )
            for _ in range(min(needed, self.max_newton_iterations)):
                self.actions_before_newton_step()
                self.native_updates += 1
                for hook in self._hooks:
                    hook.actions_after_newton_step()
            if failure == "inversion":
                self._restore_state(checkpoint)
                self.force_remesh()
                continue
            if failure or needed > self.max_newton_iterations:
                self.fields, self.pins, self.residual = [9.0, 10.0], [True, False], 0.1
                raise RuntimeError("fake Newton failure or iteration budget exhaustion")
            self.residual = 0.0
            for hook in self._hooks:
                hook.actions_after_newton_solve()
            if self.quality:
                self.quality -= 1
                self.force_remesh()
            return
        raise RuntimeError("fake inversion retry exhaustion")


@pytest.fixture
def fake_problem(tmp_path, monkeypatch):
    mesh = tmp_path / "original.msh"
    mesh.write_bytes(b"synthetic original mesh")
    output = tmp_path / "new-runtime"
    output.mkdir()
    problem_type = driver.budget_problem_type(FakeProblem, Hooks)
    monkeypatch.setattr(model, "_stationary_residual", lambda p: p.residual)
    monkeypatch.setattr(model, "_stationary_state_signature", lambda p, az: p.signature())
    monkeypatch.setattr(driver.diagnostic, "actual_pins", lambda p: {"pins": p.pins.copy()})
    case = model.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")

    def create(**kwargs):
        events = []
        phase = driver.PhaseBudget(
            10000, lambda event, **values: events.append({"event": event, **values})
        )
        problem = problem_type(mesh, output, **kwargs)
        driver.attach_budget(problem, phase, Hooks)
        return SimpleNamespace(
            problem=problem, phase=phase, output=output, case=case, events=events
        )

    return create


def test_one_pre_remesh_is_carried_before_counter_reads_and_bond_carry_persists(fake_problem):
    bundle = fake_problem(needed=[34, 22, 4], quality=1)
    p, phase = bundle.problem, bundle.phase
    assert p.force_remesh() is True and phase.remeshes == 1
    seed = model.polish_confined_state(p, bundle.case, bundle.output, "seed")
    assert seed["status"] == "converged" and seed["updates_total"] == 56
    assert seed["remeshes_total"] == phase.remeshes == p.physical_remeshes == 2
    assert p.trials == [(0.5, 40), (0.5, 40)]
    phase.label = "bond_001"
    p.parameters["Bo"].value = 0.1
    bond = model.polish_confined_state(
        p, bundle.case, bundle.output, "bond_001", initial_relaxation=1.0
    )
    phase.audit()
    assert bond["updates_total"] == phase.updates == p.native_updates == 60
    assert bond["remeshes_total"] == 2 and p.trials[-1] == (1.0, 40)
    assert phase.stages[-1]["carried_updates"] == 56
    assert phase.stages[-1]["carried_physical_remeshes"] == 2


def test_pre_remesh_leaves_at_most_two_additional_physical_remeshes(fake_problem):
    bundle = fake_problem(failures=["inversion"] * 3)
    p, phase = bundle.problem, bundle.phase
    p.force_remesh()
    with pytest.raises(ValueError, match="three physical remeshes"):
        model.polish_confined_state(p, bundle.case, bundle.output, "seed")
    phase.audit()
    receipt = runner._read_json(bundle.output / "polish/seed/polish.json")
    assert receipt["status"] == "failed" and receipt["rollback_verified"] is True
    assert receipt["updates_total"] == phase.updates == p.native_updates == 3
    assert receipt["remeshes_total"] == phase.remeshes == p.physical_remeshes == 3
    assert p.trials == [(0.5, 40), (0.25, 39), (0.125, 38)]


def test_load_notifications_consume_no_remesh_and_renew_no_stage(fake_problem):
    bundle = fake_problem()
    p, phase = bundle.problem, bundle.phase
    p.force_remesh()
    before = p._snapshot_state()
    p._restore_state(before)
    p._restore_state(before)
    assert phase.remeshes == p.physical_remeshes == 1
    model.polish_confined_state(p, bundle.case, bundle.output, "seed")
    stage_updates = phase.current.stage_updates
    p._restore_state(p._snapshot_state())
    phase.audit()
    assert phase.current.stage_updates == stage_updates == 2
    assert phase.remeshes == 1 and phase.updates == 2


def test_shared_160_update_cap_never_resets_at_a_new_bond(fake_problem):
    bundle = fake_problem(needed=40)
    p, phase = bundle.problem, bundle.phase
    p.force_remesh()
    for index in range(4):
        driver.polish_stage(p, bundle.case, bundle.output, phase, f"bond_{index:03d}", 1.0)
    assert phase.updates == p.native_updates == 160
    with pytest.raises(ValueError, match="Newton budget exhausted"):
        driver.polish_stage(p, bundle.case, bundle.output, phase, "bond_004", 1.0)
    assert p.native_updates == 160 and len(phase.stages) == 4
    assert phase.remeshes == 1 and not (bundle.output / "polish/bond_004").exists()


def test_40_update_stage_failure_keeps_debits_and_full_rollback_without_retry(fake_problem):
    bundle = fake_problem(needed=50)
    p, phase = bundle.problem, bundle.phase
    p.force_remesh()
    before = driver.state_record(p)
    with pytest.raises(RuntimeError, match="budget exhaustion"):
        driver.polish_stage(p, bundle.case, bundle.output, phase, "seed", 0.5)
    assert phase.updates == p.native_updates == 40 and phase.remeshes == 1
    assert p.signature() == before["signature"]
    assert runner._read_json(bundle.output / "seed-full-rollback.json")["verified"] is True
    assert p.trials == [(0.5, 40)]


@pytest.mark.parametrize("fault", ["absent", "incomplete", "forcing"])
def test_quality_failure_preserves_full_seed_and_is_not_retried(fake_problem, fault):
    bundle = fake_problem(remesh_fault=fault)
    p, phase = bundle.problem, bundle.phase
    directory = bundle.output / "seed-quality-remesh"
    directory.mkdir()
    before = driver.state_record(p)
    with pytest.raises((ValueError, RuntimeError)):
        driver.remesh_seed(p, bundle.case, directory, phase)
    assert p.signature() == before["signature"]
    receipt = runner._read_json(directory / "projection.json")
    assert receipt["status"] == "failed" and receipt["rollback_verified"] is True
    assert receipt["numerical_acceptance"] is False and phase.updates == 0
    assert phase.remesh_starts <= 1 and not phase.pending_remesh


def test_projection_is_unqualified_with_complete_stream_pin_and_history_receipts(fake_problem):
    bundle = fake_problem()
    directory = bundle.output / "seed-quality-remesh"
    directory.mkdir()
    before = bundle.problem.signature()
    result = driver.remesh_seed(bundle.problem, bundle.case, directory, bundle.phase)
    assert (
        result["status"] == "projected_initial_guess"
        and result["snapshot_restore"]["verified"] is True
    )
    assert (
        result["numerical_acceptance"] is False
        and result["same_state_claimed_after_projection"] is False
    )
    assert result["before"]["metadata"]["dts"] == result["after"]["metadata"]["dts"] == [1.0, 1.0]
    assert result["before"]["pins"] == result["after"]["pins"]
    assert bundle.problem.signature()["fields"] != before["fields"]
    assert bundle.phase.remeshes == bundle.problem.physical_remeshes == 1
    assert (directory / "before.dump").is_file() and (directory / "after.dump").is_file()


def test_wall_and_missing_hook_refuse_before_an_update(fake_problem):
    bundle = fake_problem()
    clock = [0.0]
    phase = driver.PhaseBudget(10, lambda *a, **kw: None, clock=lambda: clock[0])
    clock[0] = 10
    with pytest.raises(ValueError, match="wall budget exhausted"):
        phase.before_update()
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        bundle.phase.carry(Hooks())
    assert bundle.problem.native_updates == 0


def phase_packet(tmp_path, **budget):
    path = tmp_path / "phase.json"
    value = {
        "schema": driver.PHASE_SCHEMA,
        "phase_id": "new-seed-quality-phase",
        "kind": "new_informative_phase",
        "budget_scope": "seed_bond_ladder_warm_and_cold_qualification",
        "fresh_budget": {**driver.BUDGET, "wall_seconds": 21600, **budget},
    }
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize(
    "key,value",
    [
        ("total_updates", 200),
        ("physical_remeshes", 4),
        ("per_mesh_stage_updates", True),
        ("wall_seconds", 0),
        ("wall_seconds", float("inf")),
    ],
)
def test_new_phase_has_finite_caps_and_no_old_allowance(tmp_path, key, value):
    path, _ = phase_packet(tmp_path, **{key: value})
    with pytest.raises(ValueError):
        driver.phase_contract(path)


def test_check_and_failure_protocol_import_no_native_and_preserve_original(tmp_path, monkeypatch):
    path, phase = phase_packet(tmp_path)
    original = tmp_path / "original-failed.json"
    original.write_text(json.dumps({"status": "failed", "updates": 40, "physical_remeshes": 1}))
    original_bytes = original.read_bytes()
    source = {"path": str(Path(driver.__file__)), "sha256": driver.sha(Path(driver.__file__))}
    out = tmp_path / "new-runtime"
    prepared = {
        "phase": phase,
        "case": {"case_id": "fake-contract-case", "nondimensionalization": {}},
        "case_path": str(path),
        "output": str(out),
        "component": {"commit": "a" * 40},
        "environment": {"commit": model.PYOOMPH_COMMIT},
        "implementation": {driver.ENTRYPOINT: source},
        "inputs": [{"path": str(original), "sha256": hashlib.sha256(original_bytes).hexdigest()}],
        "restart": {
            "state": {"path": str(original), "sha256": driver.sha(original)},
            "parent_manifest": {},
            "packet": {},
        },
        "original_failure": json.loads(original_bytes),
        "schedule": [0.0, 1.56],
    }
    monkeypatch.setattr(driver, "preflight", lambda args: prepared)

    def failed(prepared, output):
        assert (output / runner.MANIFEST).is_file()
        events = (output / runner.PROGRESS).read_text().splitlines()
        assert len(events) == 1 and json.loads(events[0])["event"] == "prepared"
        raise RuntimeError("fake bounded native failure")

    monkeypatch.setattr(driver, "execute", failed)
    args = ["--source", str(ROOT), "--case", str(path), "--phase", str(path), "--out", str(out)]
    assert driver.main([*args, "--check"]) == 0 and not out.exists()
    with pytest.raises(RuntimeError, match="bounded native failure"):
        driver.main(args)
    failure = runner._read_json(out / "failure.json")
    actual = runner._read_json(out / runner.MANIFEST)
    assert actual["runner"] == source and actual["restart_from"]["qualified_stationary"] is False
    assert actual["new_phase"]["fresh_budget"]["total_updates"] == 160
    assert failure["original_failure"]["updates"] == 40 and failure["inputs_unchanged"] is True
    assert failure["completion_reached"] is False and original.read_bytes() == original_bytes


def test_symlink_artifact_is_refused_even_with_matching_resolved_bytes(tmp_path):
    original = tmp_path / "data.json"
    original.write_bytes(b"original software input")
    alias = tmp_path / "alias.json"
    alias.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        driver.bound_file({"path": str(alias), "sha256": driver.sha(original)}, tmp_path)


def test_old_model_runner_and_bond_schedule_are_wholly_unchanged():
    for relative in (
        "problems/paper_bubble.py",
        "runners/run_paper_case.py",
        "problems/paper_mesh.py",
    ):
        assert (ROOT / relative).read_bytes() == runner._git(
            ROOT, "show", f"{driver.BASE_SOURCE}:{relative}"
        )
    case = model.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")
    schedule = model.stationary_bond_schedule(case)
    assert len(schedule) == 17 and schedule[-1] == 1.56 and schedule[8] == 0.7999999999999999


@pytest.fixture
def failed_bundle(tmp_path, monkeypatch):
    attempt = tmp_path / "original-attempt"
    runtime = attempt / "runtime"
    state = runtime / "polish/seed/before.dump"
    state.parent.mkdir(parents=True)
    reference = "runtime/solver/_gmsh/PaperTwoPhaseMesh.msh"
    mesh = attempt / reference
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"synthetic original mesh")
    state.write_bytes(fake_dump(reference, {"not": "native fields"}))
    producer = "7" * 64
    manifest = {
        "component_commit": driver.PARENT_SOURCE,
        "component": {
            "commit": driver.PARENT_SOURCE,
            "sources": {"runners/run_paper_case.py": {"sha256": producer}},
        },
        "runner": {"path": "/synthetic/source/run_paper_case.py", "sha256": producer},
        "runner_path": "/synthetic/source/run_paper_case.py",
        "runner_sha256": producer,
    }
    polish = {
        "status": "failed",
        "rollback_verified": True,
        "updates_total": 40,
        "stage_updates": 40,
        "remeshes_total": 1,
        "parameters": {"Ca_in": 0.02, "Bo": 0.0},
        "trials": [
            {"relaxation": 0.5, "remaining_updates": 40},
            {"relaxation": 0.25, "remaining_updates": 37},
        ],
    }
    supervisor = {"status": "failed", "completion_reached": False, "cleanup_unresolved": False}
    (attempt / "runner.log").write_bytes(b"software fake log only")
    terminal = {
        "schema": "stone-volume-primary-terminal-v1",
        "physical_acceptance": False,
        "target_bond_reached": False,
        "polish": copy.deepcopy(polish),
        "combined_deadline_utc": "2026-10-12T23:36:09Z",
        "artifacts": {},
    }
    restart = {"parent_manifest": {"path": str(runtime / runner.MANIFEST)}}

    def rewrite():
        for path, value in (
            (runtime / runner.MANIFEST, manifest),
            (runtime / "polish/seed/polish.json", polish),
            (attempt / "supervisor-result.json", supervisor),
        ):
            path.write_text(json.dumps(value))
        hashes = {relative: driver.sha(attempt / relative) for relative in driver.FAILURE_HASHES}
        monkeypatch.setattr(driver, "FAILURE_HASHES", hashes)
        terminal["artifacts"] = {
            str(attempt / relative): digest for relative, digest in hashes.items()
        }

    rewrite()
    return SimpleNamespace(
        attempt=attempt,
        restart=restart,
        manifest=manifest,
        polish=polish,
        supervisor=supervisor,
        terminal=terminal,
        rewrite=rewrite,
    )


def test_strict_original_failure_admission_has_separate_fresh_phase_debits(failed_bundle):
    bundle = failed_bundle
    originals = {str(p): p.read_bytes() for p in bundle.attempt.rglob("*") if p.is_file()}
    records, failure = driver.failed_inputs(bundle.restart, bundle.terminal)
    assert len(records) == 6 and failure["updates"] == 40 and failure["physical_remeshes"] == 1
    assert failure["stationary_endpoint_exists"] is False
    assert failure["budget_transferred_to_new_phase"] is False
    assert originals == {str(p): p.read_bytes() for p in bundle.attempt.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "fault", ["producer", "source", "budget", "completed", "rollback", "deadline"]
)
def test_wrong_failed_producer_or_old_phase_claim_is_rejected(failed_bundle, fault):
    bundle = failed_bundle
    if fault == "producer":
        bundle.manifest["runner"]["sha256"] = "0" * 64
    elif fault == "source":
        bundle.manifest["component_commit"] = "0" * 40
    elif fault == "budget":
        bundle.polish["updates_total"] = 39
        bundle.terminal["polish"] = copy.deepcopy(bundle.polish)
    elif fault == "completed":
        bundle.supervisor.update(status="completed", completion_reached=True)
    elif fault == "rollback":
        bundle.polish["rollback_verified"] = False
        bundle.terminal["polish"] = copy.deepcopy(bundle.polish)
    else:
        bundle.terminal["combined_deadline_utc"] = "2026-10-13T23:36:09Z"
    bundle.rewrite()
    with pytest.raises(ValueError):
        driver.failed_inputs(bundle.restart, bundle.terminal)


@pytest.mark.parametrize("fault", [None, "fields", "pins", "footer"])
def test_cold_seed_load_changes_only_evidenced_endtime_and_refuses_other_payload_faults(
    fake_problem,
    monkeypatch,
    fault,
):
    bundle = fake_problem()
    p = bundle.problem
    state = bundle.output / "saved.dump"
    state.write_bytes(p._snapshot_state())
    parent = bundle.output / "parent.json"
    parent.write_text("{}")
    packet = bundle.output / "packet.json"
    packet.write_text("{}")
    mesh = Path(p.reference)
    restart = {
        "state": {"path": str(state), "sha256": driver.sha(state)},
        "parent_manifest": {"path": str(parent), "sha256": driver.sha(parent)},
        "packet": {"path": str(packet), "sha256": driver.sha(packet)},
        "meshes": [{"path": str(mesh), "sha256": driver.sha(mesh)}],
        "mesh_reference_base": str(bundle.output),
        "mesh_exchange": {p.reference: p.reference},
        "stage": {"kind": "pending_polish", "ca_in": 0.02, "bond": 0.0, "bc_mode": "init"},
    }
    original_load = p.load_state

    def corrupt(value, **kwargs):
        original_load(value, **kwargs)
        if fault == "fields":
            p.fields[0] = 99
        elif fault == "pins":
            p.pins[0] = True

    monkeypatch.setattr(p, "load_state", corrupt)
    if fault == "footer":
        state.write_bytes(state.read_bytes()[:-1])
        restart["state"]["sha256"] = driver.sha(state)
    p._run_statement_endtime = None
    if fault is None:
        receipt = driver.load_exact(p, bundle.case, restart, bundle.output)
        assert receipt["verified"] and receipt["stream"]["verified"]
        assert receipt["restored_writer_endtime"] == 8.0
        assert receipt["stationary_certificate_claimed_by_load"] is False
        assert receipt["snapshot_restore"]["full_stream_and_actual_pins"] is True
        assert bundle.phase.updates == bundle.phase.remeshes == 0
    else:
        with pytest.raises(ValueError):
            driver.load_exact(p, bundle.case, restart, bundle.output)
        assert not (bundle.output / "load.json").exists()
    assert p.native_updates == p.physical_remeshes == 0


@pytest.mark.parametrize("fault", ["phase", "case"])
def test_preflight_wrong_phase_or_case_is_refused_before_environment_and_output(
    tmp_path,
    monkeypatch,
    fault,
):
    phase_path, phase = phase_packet(tmp_path)
    if fault == "phase":
        phase["kind"] = "reuse_unused_old_allowance"
        phase_path.write_text(json.dumps(phase))
    case_path = tmp_path / "fake-case.json"
    case_path.write_text("{}")
    out = tmp_path / "never-created/runtime"
    args = SimpleNamespace(source=ROOT, case=case_path, phase=phase_path, out=out)

    def no_environment():
        raise AssertionError("invalid phase/case must be refused before environment checks")

    monkeypatch.setattr(runner, "pyoomph_revision", no_environment)
    with pytest.raises(ValueError):
        driver.preflight(args)
    assert not out.parent.exists()


@pytest.mark.parametrize(
    "fault", [None, "stationary", "lower_static_pressure", "flux", "volume", "volume_integrals"]
)
def test_full_fake_bond_route_and_fresh_cold_load_require_each_unchanged_gate(
    fake_problem,
    monkeypatch,
    fault,
):
    bundle = fake_problem()
    original = bundle.output.parent / "retained.dump"
    original.write_bytes(bundle.problem._snapshot_state())
    original_bytes = original.read_bytes()
    packet = bundle.output.parent / "original-packet.json"
    packet.write_text("{}")
    parent = bundle.output.parent / "original-manifest.json"
    parent.write_text("{}")
    original_mesh = Path(bundle.problem.reference)

    def restart_for(state, manifest, packet_path, stage):
        refs = runner.state_mesh_references(state)
        meshes = [
            {"stored_path": ref, "path": ref, "sha256": driver.sha(Path(ref))} for ref in refs
        ]
        return {
            "state": {"path": str(state), "sha256": driver.sha(state), "format": "memory_snapshot"},
            "parent_manifest": {"path": str(manifest), "sha256": driver.sha(manifest)},
            "packet": {"path": str(packet_path), "sha256": driver.sha(packet_path)},
            "meshes": meshes,
            "mesh_exchange": {entry["stored_path"]: entry["path"] for entry in meshes},
            "mesh_reference_base": str(state.parent),
            "stage": stage,
        }

    initial_stage = {
        "kind": "pending_polish",
        "bond_index": 0,
        "ca_in": 0.02,
        "bond": 0.0,
        "bc_mode": "init",
    }
    initial = restart_for(original, parent, packet, initial_stage)
    _, phase_packet_value = phase_packet(bundle.output.parent)
    schedule = model.stationary_bond_schedule(bundle.case)
    own = {"path": str(Path(driver.__file__)), "sha256": driver.sha(Path(driver.__file__))}
    prepared = {
        "case": bundle.case,
        "phase": phase_packet_value,
        "restart": initial,
        "schedule": schedule,
        "component": {"commit": "c" * 40},
        "environment": {},
        "implementation": {driver.ENTRYPOINT: own},
        "inputs": [],
        "original_failure": {"status": "failed", "updates": 40, "physical_remeshes": 1},
    }
    constructed = []

    class FakeContextProblem(FakeProblem):
        def __init__(self, case, *, initial_bond):
            super().__init__(original_mesh, bundle.output)
            self.parameters["Bo"].value = initial_bond
            self._run_statement_endtime = None
            constructed.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(model, "_native_problem_types", lambda: (FakeContextProblem, None))
    monkeypatch.setattr(
        model, "build_problem", lambda case, **kwargs: FakeContextProblem(case, **kwargs)
    )
    monkeypatch.setattr(driver, "configure", lambda *args: None)

    def checkpoint(p, case, output, index, az):
        assert az and p.residual == 0.0 and p.parameters["Bo"].value == schedule[index]
        directory = output / "checkpoints" / f"bond_{index:03d}"
        directory.mkdir(parents=True)
        state = directory / "state.dump"
        state.write_bytes(p._snapshot_state())
        p._restore_state(state.read_bytes())  # A load notification must consume no remesh.
        packet_path = directory / "resume.json"
        stage = {
            "kind": "completed_polish",
            "bond_index": index,
            "ca_in": 0.02,
            "bond": schedule[index],
            "bc_mode": p._last_bc_setting,
        }
        packet_path.write_text(json.dumps(restart_for(state, parent, packet, stage)))
        return {
            "bond_index": index,
            "bond": schedule[index],
            "resume_packet": str(packet_path.relative_to(output)),
        }

    monkeypatch.setattr(model, "_checkpoint_stationary_state", checkpoint)
    monkeypatch.setattr(
        runner, "validate_restart", lambda path, *args, **kwargs: runner._read_json(path)
    )

    def sample(p, model, case, output, phase, np):
        values = {
            "stationary_residual_linf": p.residual,
            "lower_pressure_rms": 0.0,
            "lower_flux": 1.0,
            "upper_flux": 1.0,
            "gas_volume": p.volume,
            "surface_volume": p.volume,
            "target_volume": p.volume,
        }
        if phase == "endpoint" and fault:
            fields = {
                "stationary": "stationary_residual_linf",
                "lower_static_pressure": "lower_pressure_rms",
                "flux": "upper_flux",
                "volume": "gas_volume",
                "volume_integrals": "surface_volume",
            }
            values[fields[fault]] = 9.0
        return {
            "signature": p.signature(),
            "actual_pins": {"pins": p.pins.copy()},
            "metrics": values,
        }

    monkeypatch.setattr(driver.remesh, "sample", sample)
    runner.append_progress(bundle.output, {"event": "prepared"}, first=True)
    if fault is None:
        result = driver.execute(prepared, bundle.output)
        assert result["completion_reached"] and all(result["checks"].values())
        assert all(result["cold_load_checks"].values())
    else:
        with pytest.raises(ValueError, match="qualification gate"):
            driver.execute(prepared, bundle.output)
        result = runner._read_json(bundle.output / "summary.json")
        assert not result["completion_reached"] and not result["cold_load_checks"][fault]
    assert len(constructed) == 2 and constructed[0] is not constructed[1]
    assert len(result["stages"]) == 17 and result["bond"] == 1.56
    assert constructed[0].trials[0] == (0.5, 40)
    assert all(omega == 1.0 for omega, _ in constructed[0].trials[1:])
    assert constructed[0].native_updates == 34 and constructed[1].native_updates == 0
    assert result["new_phase_budget"]["updates_total"] == 34
    assert result["new_phase_budget"]["physical_remeshes_total"] == 1
    assert result["new_phase_budget"]["budget_hooks_agree"] is True
    assert result["stages"][0]["phase_budget"]["stages"] == result["new_phase_budget"]["stages"][:1]
    assert result["original_failure"]["updates"] == 40 and original.read_bytes() == original_bytes
