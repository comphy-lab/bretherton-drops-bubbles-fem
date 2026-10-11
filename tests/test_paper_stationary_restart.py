"""Stationary restart orchestration and refusal tests; no native solve executes."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from problems import paper_bubble
from runners import run_paper_case as runner

PARENT_COMMIT = "acdcf57ec39124f0d418d5e1ab9ed4a8f84699b7"
ROOT = Path(__file__).resolve().parents[1]


def dump_header(reference):
    def integer(number):
        return number.to_bytes(8, "little", signed=True)

    def string(value):
        return integer(len(value)) + value.encode("ascii")

    return (string("pyoomph_dump") + string("0.1.6") + string("global")
            + struct.pack("<d", 8.0) + integer(9) + integer(0) + integer(6)
            + integer(1) + string(reference) + integer(1) + integer(2)
            + b"software fixture: no native fields" + string("EOF_pyoomph"))


@pytest.fixture
def restart_bundle(tmp_path, monkeypatch):
    """Real Git source identity with synthetic dump/environment; software evidence only."""
    case_path = ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json"
    case = paper_bubble.read_case(case_path)
    current = runner.component_revision()
    current.update(commit=PARENT_COMMIT, head=PARENT_COMMIT, clean=True, status=[])
    source_records = {}
    for relative in current["sources"]:
        raw = runner._git(ROOT, "show", f"{PARENT_COMMIT}:{relative}")
        assert raw is not None
        digest = hashlib.sha256(raw).hexdigest()
        source_records[relative] = {"sha256": digest, "committed_sha256": digest,
                                    "committed_matches": True}
    component = {"commit": PARENT_COMMIT, "head": PARENT_COMMIT, "clean": True,
                 "status": [], "sources": source_records}
    environment = {"available": True, "version": "software_standin",
        "commit": paper_bubble.PYOOMPH_COMMIT, "url": "https://github.com/comphy-lab/pyoomph",
        "direct_url": {"vcs_info": {"vcs": "git", "commit_id": paper_bubble.PYOOMPH_COMMIT}},
        "native_core": {"path": "software-native-core.so", "sha256": "3" * 64}}
    attempt = tmp_path / "parent-attempt"
    runtime = attempt / "runtime"
    state = runtime / "polish/bond_018/before.dump"
    state.parent.mkdir(parents=True)
    reference = "runtime/solver/_gmsh/PaperTwoPhaseMesh_REMESH_000001.msh"
    mesh = attempt / reference
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"Gmsh software fixture only")
    state.write_bytes(dump_header(reference))
    retained_case = attempt / "case.json"
    retained_case.write_bytes(case_path.read_bytes())
    manifest = {"schema": "pyoomph-runner-manifest-v1", "component_commit": PARENT_COMMIT,
                "component": component, "pyoomph": copy.deepcopy(environment), "case_id": case["case_id"],
                "case_path": str(retained_case), "case_file": str(retained_case),
                "case_sha256": runner.sha256_file(retained_case),
                "case_fields_used": paper_bubble.CASE_FIELDS_USED, "azimuthal_stability": True}
    manifest_path = runtime / runner.MANIFEST
    manifest_path.write_text(json.dumps(manifest))
    packet = {"schema": runner.RESUME_SCHEMA,
        "parent_manifest": {"path": str(manifest_path), "sha256": runner.sha256_file(manifest_path)},
        "case_sha256": runner.sha256_file(retained_case), "azimuthal_stability": True,
        "state": {"path": str(state), "sha256": runner.sha256_file(state), "format": "memory_snapshot"},
        "mesh_reference_base": str(attempt),
        "meshes": [{"stored_path": reference, "path": str(mesh), "sha256": runner.sha256_file(mesh)}],
        "stage": {"kind": "pending_polish", "bond_index": 9,
                  "bond": paper_bubble.stationary_bond_schedule(case)[9],
                  "ca_in": 0.02, "bc_mode": "stationary"}}
    packet_path = tmp_path / "resume.json"
    packet_path.write_text(json.dumps(packet))
    monkeypatch.setattr(runner, "component_revision", lambda: copy.deepcopy(current))
    monkeypatch.setattr(runner, "pyoomph_revision", lambda: copy.deepcopy(environment))

    def no_native(*args, **kwargs):
        raise AssertionError("software restart tests must never execute native solves")

    monkeypatch.setattr(runner, "run_model", no_native)
    return SimpleNamespace(case=case, case_path=retained_case, current=current,
        environment=environment, manifest=manifest, manifest_path=manifest_path,
        packet=packet, packet_path=packet_path, state=state, mesh=mesh)


def admit(bundle):
    return runner.validate_restart(bundle.packet_path, bundle.case,
        runner.sha256_file(bundle.case_path), component=bundle.current,
        environment=bundle.environment, azimuthal_stability=True)


def write_packet(bundle):
    bundle.packet_path.write_text(json.dumps(bundle.packet))


def rewrite_parent(bundle):
    bundle.manifest_path.write_text(json.dumps(bundle.manifest))
    bundle.packet["parent_manifest"]["sha256"] = runner.sha256_file(bundle.manifest_path)
    write_packet(bundle)


def test_incomplete_original_parent_is_admitted_with_explicit_stationary_compatibility(restart_bundle):
    bundle = restart_bundle
    assert not (bundle.manifest_path.parent / "summary.json").exists()
    admitted = admit(bundle)
    assert admitted["mesh_exchange"] == {bundle.packet["meshes"][0]["stored_path"]: str(bundle.mesh)}
    assert admitted["compatibility"]["parent_commit"] == PARENT_COMMIT
    assert admitted["compatibility"]["physics_ast_sha256"]
    assert admitted["stage"]["bond_index"] == 9


@pytest.mark.parametrize("fault", ["state_bytes", "mesh_bytes", "mesh_missing", "parent_bytes",
    "parent_dirty", "unknown_source", "parent_source_hash", "case_bytes", "case_id",
    "case_sha", "azimuthal_packet", "azimuthal_parent", "core_bytes", "mesh_base",
    "mesh_reference", "stage_index", "stage_bond", "stage_ca", "completed_without_receipt"])
def test_restart_tampering_refuses_before_any_new_output(restart_bundle, tmp_path, fault):
    bundle = restart_bundle
    if fault == "state_bytes":
        bundle.state.write_bytes(bundle.state.read_bytes() + b"changed")
    elif fault == "mesh_bytes":
        bundle.mesh.write_bytes(b"changed")
    elif fault == "mesh_missing":
        bundle.mesh.unlink()
    elif fault == "parent_bytes":
        bundle.manifest_path.write_bytes(b"changed")
    elif fault == "parent_dirty":
        bundle.manifest["component"].update(clean=False, commit=PARENT_COMMIT + "-dirty")
        rewrite_parent(bundle)
    elif fault == "unknown_source":
        bundle.manifest["component"].update(commit="0" * 40, head="0" * 40)
        bundle.manifest["component_commit"] = "0" * 40
        rewrite_parent(bundle)
    elif fault == "parent_source_hash":
        bundle.manifest["component"]["sources"]["problems/paper_mesh.py"]["sha256"] = "0" * 64
        rewrite_parent(bundle)
    elif fault == "case_bytes":
        bundle.case_path.write_bytes(bundle.case_path.read_bytes() + b" ")
    elif fault == "case_id":
        bundle.manifest["case_id"] = "other_case"
        rewrite_parent(bundle)
    elif fault == "case_sha":
        bundle.packet["case_sha256"] = "0" * 64
    elif fault == "azimuthal_packet":
        bundle.packet["azimuthal_stability"] = False
    elif fault == "azimuthal_parent":
        bundle.manifest["azimuthal_stability"] = False
        rewrite_parent(bundle)
    elif fault == "core_bytes":
        bundle.manifest["pyoomph"]["native_core"]["sha256"] = "0" * 64
        rewrite_parent(bundle)
    elif fault == "mesh_base":
        bundle.packet["mesh_reference_base"] = str(bundle.state.parent)
    elif fault == "mesh_reference":
        bundle.packet["meshes"][0]["stored_path"] = "wrong.msh"
    elif fault == "stage_index":
        bundle.packet["stage"]["bond_index"] = True
    elif fault == "stage_bond":
        bundle.packet["stage"]["bond"] = 0.9  # Exact original ladder value is required.
    elif fault == "stage_ca":
        bundle.packet["stage"]["ca_in"] = -0.02
    else:
        bundle.packet["stage"]["kind"] = "completed_polish"
    write_packet(bundle)
    out = tmp_path / "never-created/output"
    with pytest.raises(SystemExit) as error:
        runner.main([str(bundle.case_path), "--out", str(out), "--azimuthal-stability",
                     "--restart-from", str(bundle.packet_path)])
    assert error.value.code == 2 and not out.parent.exists()


def test_restart_check_is_read_only_and_manifest_binds_official_lineage(
    restart_bundle, tmp_path, monkeypatch,
):
    bundle = restart_bundle
    out = tmp_path / "new/output"
    arguments = [str(bundle.case_path), "--out", str(out), "--azimuthal-stability",
                 "--restart-from", str(bundle.packet_path)]
    assert runner.main([*arguments, "--check"]) == 0
    assert not out.parent.exists()

    def software_execute(case, directory, *, azimuthal_stability, restart):
        assert case == bundle.case and directory == out and azimuthal_stability is True
        manifest = json.loads((out / runner.MANIFEST).read_text())
        lineage = manifest["restart_from"]
        assert lineage["path"] == str(bundle.state)
        assert lineage["sha256"] == runner.sha256_file(bundle.state)
        assert lineage["parent_manifest"] == restart["parent_manifest"]
        assert lineage["packet_sha256"] == runner.sha256_file(bundle.packet_path)
        assert lineage["parent_component_commit"] == PARENT_COMMIT
        assert lineage["parent_native_core_sha256"] == "3" * 64
        assert len((out / runner.PROGRESS).read_text().splitlines()) == 1
        return {"status": "completed", "completion_reached": True,
                "scientific_status": "software_fixture_only"}

    monkeypatch.setattr(runner, "run_model", software_execute)
    assert runner.main(arguments) == 0


def test_physics_or_polish_changes_are_incompatible_even_with_clean_metadata(restart_bundle):
    bundle = restart_bundle
    original = (ROOT / "problems/paper_bubble.py").read_bytes()
    changed = original.replace(b"self.target = target", b"self.target = target + 1")
    assert changed != original
    assert runner._physics_contract(original) != runner._physics_contract(changed)
    changed_polish = original.replace(b"min(40, case", b"min(80, case")
    assert runner._physics_contract(original) != runner._physics_contract(changed_polish)
    current = copy.deepcopy(bundle.current)
    current["sources"]["problems/paper_mesh.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="supporting method"):
        runner._parent_source_contract(bundle.manifest, current)


@pytest.mark.parametrize("bad", [b"", b"not a state", dump_header("fixture.msh")[:-1]])
def test_header_refuses_truncated_or_nonstate_files_without_native_import(tmp_path, bad):
    path = tmp_path / "bad.dump"
    path.write_bytes(bad)
    with pytest.raises(ValueError):
        runner.state_mesh_references(path)


@pytest.mark.parametrize("stage_kind,index", [("pending_polish", 0), ("pending_polish", 9),
                                             ("completed_polish", 9), ("completed_polish", 16)])
def test_resume_skips_seed_selects_bound_polish_and_finishes_unchanged_checks(
    tmp_path, monkeypatch, stage_kind, index,
):
    """A Python orchestration stand-in; the real native smoke remains a separate run."""
    case = paper_bubble.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")
    schedule = paper_bubble.stationary_bond_schedule(case)
    calls = []
    parameters = {"Ca_in": SimpleNamespace(value=0.02), "Bo": SimpleNamespace(value=0.0)}
    geometry = paper_bubble.case_geometry(case)

    class SoftwareProblem:
        def __init__(self):
            self.geometry = geometry

        def set_output_directory(self, path):
            calls.append("output")

        def set_linear_solver(self, value):
            assert value == "superlu"

        def set_c_compiler(self, value):
            assert value == "system"

        def setup_for_stability_analysis(self, **kwargs):
            calls.append("azimuthal_setup")

        def initialise(self):
            calls.append("initialise")

        def run(self, *args, **kwargs):
            raise AssertionError("a stationary resume must not regenerate its transient seed")

        def get_global_parameter(self, name):
            return parameters[name]

        def observable(self, name, domain="liquid"):
            if name in {"volume", "surface_volume"}:
                return geometry.volume if domain == "gas" or name == "surface_volume" else 1.0
            return 1.0 if name == "area" else 0.0

        def frame_speed(self):
            return 0.01

        def get_ode(self, domain):
            return SimpleNamespace(get_value=lambda *args, **kwargs: 0.0)

        def interface_polyline(self):
            return np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 0.5, 0.0])

        def save_state(self, path, **kwargs):
            path = Path(path)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"software-only final-state fixture")

    @contextmanager
    def build(*args, **kwargs):
        assert kwargs["initial_bond"] == case["solve"]["initial_bond"]
        yield SoftwareProblem()

    def load(problem, same_case, output, restart, azimuthal_stability):
        assert same_case == case and azimuthal_stability is True
        calls.append("load_verified")
        parameters["Bo"].value = restart["stage"]["bond"]

    def polish(problem, same_case, output, label, *, initial_relaxation=0.5):
        assert same_case == case and same_case["solve"]["max_newton_iterations"] == 40
        calls.append(("polish", parameters["Bo"].value, initial_relaxation, label))
        return {"status": "software_fixture_only"}

    def checkpoint(problem, same_case, output, bond_index, azimuthal_stability):
        calls.append(("checkpoint", bond_index))
        return {"bond_index": bond_index}

    package = ModuleType("pyoomph")
    package.__path__ = []
    outputs = ModuleType("pyoomph.output")
    outputs.__path__ = []
    states = ModuleType("pyoomph.output.states")
    states.copy_state_file = lambda source, destination, **kwargs: Path(destination).write_bytes(
        Path(source).read_bytes())
    for name, module in (("pyoomph", package), ("pyoomph.output", outputs),
                         ("pyoomph.output.states", states)):
        monkeypatch.setitem(__import__("sys").modules, name, module)
    monkeypatch.setattr(paper_bubble, "build_problem", build)
    monkeypatch.setattr(paper_bubble, "_load_stationary_restart", load)
    monkeypatch.setattr(paper_bubble, "polish_confined_state", polish)
    monkeypatch.setattr(paper_bubble, "_checkpoint_stationary_state", checkpoint)
    monkeypatch.setattr(paper_bubble, "_stationary_residual", lambda problem: 0.0)
    runner.create_run_manifest(tmp_path, {"purpose": "stationary resume software fixture only"})
    restart = {"stage": {"kind": stage_kind, "bond_index": index, "bond": schedule[index]}}
    summary = paper_bubble.run_model(case, tmp_path, azimuthal_stability=True, restart=restart)
    polishes = [event for event in calls if isinstance(event, tuple) and event[0] == "polish"]
    expected = schedule[index if stage_kind == "pending_polish" else index + 1:]
    assert [event[1] for event in polishes] == expected
    assert [event[2] for event in polishes] == [0.5 if index == 0 and i == 0 else 1.0
                                             for i in range(len(expected))]
    assert calls[:4] == ["output", "azimuthal_setup", "initialise", "load_verified"]
    assert summary["bond"] == 1.56 and summary["completion_reached"] is True
    assert all(summary["checks"].values()) and summary["stationary_residual_linf"] == 0.0


def test_loaded_layout_refuses_missing_phase_phi_before_resumed_polish():
    problem = SimpleNamespace(is_normal_mode_stability_set_up=lambda: "azimuthal",
        _azimuthal_mode_param_m=SimpleNamespace(value=0), ndof=lambda: 2,
        get_dof_description=lambda: ([0, 1], ["liquid/velocity_phi", "gas/pressure"]))
    with pytest.raises(ValueError, match="azimuthal velocity layout"):
        paper_bubble.stationary_layout(problem, True)


@pytest.fixture
def state_problem():
    """Minimal full-state API stand-in with explicit save/load fault injection."""
    class StateProblem:
        def __init__(self):
            self.parameters = {"Ca_in": SimpleNamespace(value=0.02), "Bo": SimpleNamespace(value=0.0)}
            self.fields = np.asarray([1.0, 2.0])
            self.positional_mask = np.asarray([False, True], dtype=bool)
            self.pinned_values = np.asarray([7.0, 8.0, 0.0])
            self.eigen_data_in_states = False
            self._first_step = False
            self._taken_already_an_unsteady_step = True
            self._last_step_was_stationary = True
            self._last_bc_setting = "stationary"
            self._azimuthal_mode_param_m = SimpleNamespace(value=0)
            self.steady = False
            self.loads = []
            self.loaded_parameters = None
            self.corrupt_restore = False
            self.corrupt_pin_load = False
            self.corrupt_pin_restore = False
            self.mesh = None

        def get_global_parameter(self, name):
            return self.parameters[name]

        def get_current_dofs(self):
            return self.fields, self.positional_mask

        def get_current_pinned_values(self, with_positions):
            assert with_positions is True
            return self.pinned_values

        def interface_polyline(self):
            return np.asarray([0.0, 1.0]), np.asarray([0.0, 0.5])

        def observable(self, name, domain="liquid"):
            return 4.0

        def frame_speed(self):
            return 0.01

        def is_normal_mode_stability_set_up(self):
            return "azimuthal"

        def ndof(self):
            return 2

        def get_dof_description(self):
            return [0, 1], ["liquid/velocity_phi", "gas/velocity_phi"]

        def ntime_stepper(self):
            return 1

        def time_stepper_pt(self, index):
            return SimpleNamespace(is_steady=lambda: self.steady,
                make_steady=lambda: setattr(self, "steady", True),
                undo_make_steady=lambda: setattr(self, "steady", False))

        def _snapshot_state(self):
            assert self.eigen_data_in_states is True
            return json.dumps({"fields": self.fields.tolist(),
                               "positional_mask": self.positional_mask.tolist(),
                               "pinned_values": self.pinned_values.tolist(),
                               "parameters": {k: v.value for k, v in self.parameters.items()}}).encode()

        def _restore_state(self, snapshot):
            import io

            self.load_state(io.BytesIO(snapshot))

        def load_state(self, source, **kwargs):
            self.loads.append((source, kwargs))
            if hasattr(source, "read"):
                record = json.loads(source.read())
                self.fields = np.asarray(record["fields"])
                self.positional_mask = np.asarray(record["positional_mask"], dtype=bool)
                self.pinned_values = np.asarray(record["pinned_values"])
                for key, value in record["parameters"].items():
                    self.parameters[key].value = value
                if self.corrupt_pin_restore:
                    self.pinned_values[0] += 1
            elif self.loaded_parameters:
                for key, value in self.loaded_parameters.items():
                    self.parameters[key].value = value
            if not hasattr(source, "read") and self.corrupt_pin_load:
                self.pinned_values[0] += 1
            if self.corrupt_restore:
                self.fields[0] += 1
            self._last_step_was_stationary = False

        def save_state(self, path, **kwargs):
            assert self.eigen_data_in_states is True
            Path(path).write_bytes(dump_header(str(self.mesh)))

    return StateProblem


def test_native_load_gate_records_bound_meshes_az_forcing_and_snapshot_restore(
    restart_bundle, tmp_path, monkeypatch, state_problem,
):
    admitted = admit(restart_bundle)
    output = tmp_path / "new-native-standin"
    output.mkdir()
    runner.append_progress(output, {"event": "start"}, first=True)
    problem = state_problem()
    problem.loaded_parameters = {"Ca_in": 0.02, "Bo": admitted["stage"]["bond"]}
    paper_bubble._load_stationary_restart(problem, restart_bundle.case, output, admitted, True)
    receipt = json.loads((output / "restart-load.json").read_text())
    assert receipt["verified"] is True and receipt["parameters"] == problem.loaded_parameters
    assert receipt["snapshot_restore"]["before"] == receipt["snapshot_restore"]["after"]
    array_receipt = receipt["snapshot_restore"]["before"]["arrays"]
    assert array_receipt["positional_dof_mask"]["dtype"] == "bool"
    assert array_receipt["pinned_values_with_positions"]["shape"] == [3]
    assert array_receipt["pinned_values_with_positions"]["sha256"] == hashlib.sha256(
        problem.pinned_values.tobytes()).hexdigest()
    assert receipt["snapshot_restore"]["verified"] is True
    assert receipt["layout"]["dof_names"] == ["liquid/velocity_phi", "gas/velocity_phi"]
    assert problem.loads[0][1]["additional_info"] == {"exchange_msh_file": admitted["mesh_exchange"]}
    assert len(problem.loads) == 2 and problem.eigen_data_in_states is False
    assert not problem.steady


@pytest.mark.parametrize("fault", ["forcing", "snapshot_restore", "pinned_restore", "changed_artifact"])
def test_native_load_gate_refuses_mismatch_before_verified_receipt(
    restart_bundle, tmp_path, state_problem, fault,
):
    admitted = admit(restart_bundle)
    output = tmp_path / "native-refusal"
    output.mkdir()
    runner.append_progress(output, {"event": "start"}, first=True)
    problem = state_problem()
    original_fields = problem.fields.copy()
    original_mask = problem.positional_mask.copy()
    original_profile = [value.copy() for value in problem.interface_polyline()]
    problem.loaded_parameters = {"Ca_in": 0.02, "Bo": admitted["stage"]["bond"]}
    if fault == "forcing":
        problem.loaded_parameters["Bo"] = 0.0
    elif fault == "snapshot_restore":
        problem.corrupt_restore = True
    elif fault == "pinned_restore":
        problem.corrupt_pin_restore = True
    else:
        restart_bundle.mesh.write_bytes(b"altered after admission")
    with pytest.raises(ValueError):
        paper_bubble._load_stationary_restart(problem, restart_bundle.case, output, admitted, True)
    assert not (output / "restart-load.json").exists()
    assert len((output / runner.PROGRESS).read_text().splitlines()) == 1
    assert problem.eigen_data_in_states is False
    if fault == "changed_artifact":
        assert problem.loads == []
    if fault == "pinned_restore":
        assert np.array_equal(problem.fields, original_fields)
        assert np.array_equal(problem.positional_mask, original_mask)
        assert all(np.array_equal(a, b) for a, b in zip(
            problem.interface_polyline(), original_profile, strict=True))
        assert problem.pinned_values[0] == 8.0


@pytest.mark.parametrize("corruption", [None, "free_dofs", "pinned_values"])
def test_post_polish_checkpoint_is_exclusive_relocated_and_verified_before_progress(
    tmp_path, monkeypatch, state_problem, corruption,
):
    import sys

    case = paper_bubble.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json")
    problem = state_problem()
    problem.corrupt_restore = corruption == "free_dofs"
    problem.corrupt_pin_load = corruption == "pinned_values"
    original_fields = problem.fields.copy()
    original_mask = problem.positional_mask.copy()
    original_profile = [value.copy() for value in problem.interface_polyline()]
    (tmp_path / "solver").mkdir()
    problem.mesh = tmp_path / "solver/original.msh"
    problem.mesh.write_bytes(b"Gmsh checkpoint software fixture")
    manifest = {"case_sha256": "3" * 64, "purpose": "checkpoint software fixture"}
    runner.create_run_manifest(tmp_path, manifest)
    runner.append_progress(tmp_path, {"event": "start"}, first=True)
    package = ModuleType("pyoomph")
    package.__path__ = []
    outputs = ModuleType("pyoomph.output")
    outputs.__path__ = []
    states = ModuleType("pyoomph.output.states")

    def copy_state(source, destination, **kwargs):
        destination = Path(destination)
        (destination.parent / "original.msh").write_bytes(problem.mesh.read_bytes())
        destination.write_bytes(dump_header("original.msh"))

    states.copy_state_file = copy_state
    for name, module in (("pyoomph", package), ("pyoomph.output", outputs),
                         ("pyoomph.output.states", states)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(paper_bubble, "_stationary_residual", lambda problem: 1e-12)
    if corruption is not None:
        with pytest.raises(ValueError, match="same-state round trip"):
            paper_bubble._checkpoint_stationary_state(problem, case, tmp_path, 0, True)
        assert not (tmp_path / "checkpoints/bond_000/resume.json").exists()
        assert len((tmp_path / runner.PROGRESS).read_text().splitlines()) == 1
        if corruption == "pinned_values":
            assert np.array_equal(problem.fields, original_fields)
            assert np.array_equal(problem.positional_mask, original_mask)
            assert all(np.array_equal(a, b) for a, b in zip(
                problem.interface_polyline(), original_profile, strict=True))
            assert problem.pinned_values[0] == 8.0
    else:
        result = paper_bubble._checkpoint_stationary_state(problem, case, tmp_path, 0, True)
        packet_path = tmp_path / result["resume_packet"]
        packet = json.loads(packet_path.read_text())
        assert packet["roundtrip"]["verified"] is True
        assert packet["stage"]["kind"] == "completed_polish" and packet["stage"]["bond_index"] == 0
        assert packet["meshes"][0]["sha256"] == runner.sha256_file(problem.mesh)
        assert Path(packet["meshes"][0]["path"]).parent == packet_path.parent
        before = packet_path.read_bytes()
        with pytest.raises(FileExistsError):
            paper_bubble._checkpoint_stationary_state(problem, case, tmp_path, 0, True)
        assert packet_path.read_bytes() == before
        events = [json.loads(line) for line in (tmp_path / runner.PROGRESS).read_text().splitlines()]
        assert [event["event"] for event in events] == ["start", "stationary_checkpoint"]
    assert problem.eigen_data_in_states is False and not problem.steady
    assert problem._last_step_was_stationary is True
