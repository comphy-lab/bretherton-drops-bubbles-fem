"""Software contracts only: synthetic states never establish native qualification."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.abc
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from runners import run_paper_iii_coarse_cold as cold
from runners import run_paper_modes as modes


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith(("pyoomph.", "_pyoomph")):
            raise AssertionError("native import forbidden in coarse software controls")


@pytest.fixture(autouse=True)
def no_native():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def state_bytes(reference, *, endtime=None, marker="init"):
    def integer(value):
        return struct.pack("<q", value)
    def string(value):
        return integer(len(value.encode())) + value.encode()
    return (string("pyoomph_dump") + string("0.1.6") + string("global")
            + struct.pack("<dqqqq", 8.0, 9, 0, 6, 1) + string(str(reference)) + integer(0)
            + integer(2) + struct.pack("<ddqdd", 1.0, 1.0, 15, 4.0,
                                       float("nan") if endtime is None else endtime)
            + b"synthetic opaque software history; not a native solution"
            + cold.runner._bc_cache_trailer(marker))


def extracted_walker():
    tree = ast.parse((cold.ROOT / "problems/bretherton_bubble.py").read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "VoidBubbleProblem")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                  and node.name == "interface_polyline")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "software_walker", "exec"), namespace)  # noqa: S102 - repo-owned method only; no native class import
    return namespace["interface_polyline"]


class Node:
    def __init__(self, z, r):
        self.point = np.asarray([r, z], dtype=float)
    def x(self, index):
        return self.point[index]


def walker(nodes, segments):
    elements = []
    for indices in segments:
        selected = [nodes[index] for index in indices]
        elements.append(SimpleNamespace(nnode=lambda: 3,
                                        node_pt=lambda index, selected=selected: selected[index]))
    problem = SimpleNamespace(get_mesh=lambda name: SimpleNamespace(elements=lambda: elements))
    return np.column_stack(extracted_walker()(problem))


def test_raw_walker_keeps_reentrant_chain_and_repeated_endpoint_identity():
    points = np.asarray([[0.123456789012345, 0], [0.8, 0.4], [0.6, 0.7], [1.1, 0.6],
                         [1.234567890123456, 0]])
    nodes = [Node(*point) for point in points]
    # Both elements refer to the same physical endpoint, in scrambled orientation.
    actual = walker(nodes, [(4, 3, 2), (0, 1, 2)])
    assert np.array_equal(actual, points)
    assert len(np.unique(actual, axis=0)) == len(points)
    assert actual[0, 0] != round(actual[0, 0], 12)


def test_raw_walker_rejects_rounded_key_collision_and_invalid_chains():
    nodes = [Node(0, 0), Node(0.5, 0.3), Node(1.12345678901230, 0.5),
             Node(1.12345678901231, 0.5), Node(1.5, 0.3), Node(2, 0)]
    with pytest.raises(RuntimeError, match="rounded connectivity key"):
        walker(nodes, [(0, 1, 2), (3, 4, 5)])
    nodes = [Node(index, 0.2) for index in range(6)]
    with pytest.raises(RuntimeError, match="endpoints"):
        walker(nodes, [(0, 1, 2), (3, 4, 5)])
    with pytest.raises(RuntimeError, match="endpoints"):
        walker(nodes, [(0, 1, 2), (2, 3, 0)])


def test_raw_walker_linear_probe_halves_and_respects_conjugation():
    points = np.asarray([[0.123456789012345, 0], [0.4, 0.35], [0.7, 0.45],
                         [1.0, 0.35], [1.234567890123456, 0]])
    nodes = [Node(*point) for point in points]
    segments = [(2, 3, 4), (2, 1, 0)]
    _arc, normals, weights = modes.geometry(points)
    direction = np.asarray([[0.0003, 0], [0.0005 + 0.0007j, 0.0008 + 0.0002j],
                            [0.0002 - 0.0005j, 0.0007 + 0.0009j],
                            [-0.0003 + 0.0002j, 0.0006 - 0.0008j], [0.0004, 0]])
    def probe(vector, epsilon):
        components = []
        for component in (vector.real, vector.imag):
            for node, point, delta in zip(nodes, points, component, strict=True):
                node.point[:] = (point + epsilon * delta)[::-1]
            components.append(np.sum((walker(nodes, segments) - points) * normals, axis=1) / epsilon)
        for node, point in zip(nodes, points, strict=True):
            node.point[:] = point[::-1]
        return components[0] + 1j * components[1]
    full, half = probe(direction, modes.LIMITS["epsilon"]), probe(direction, modes.LIMITS["epsilon"] / 2)
    norm = lambda value: np.sqrt(np.sum(weights * abs(value) ** 2))
    assert norm(full - half) / norm(full) < 1e-8
    np.testing.assert_allclose(probe(direction.conj(), modes.LIMITS["epsilon"]), full.conj(), rtol=1e-8, atol=1e-12)
    assert np.array_equal(walker(nodes, segments), points)


def test_only_two_diagnostic_method_ast_exceptions_and_no_spectral_changes():
    for name in ("problems/bretherton_bubble.py", "problems/paper_bubble.py"):
        original = cold.diagnostic.git(cold.ROOT, "show", f"{cold.BASE}:{name}")
        actual = (cold.ROOT / name).read_bytes()
        assert cold.same_ast(cold.repaired_method(name, original), actual)
        assert not cold.same_ast(cold.repaired_method(name, original), actual + b"\nwrong_physics = True\n")
    original = cold.diagnostic.git(cold.ROOT, "show", f"{cold.BASE}:runners/run_paper_modes.py")
    actual = (cold.ROOT / "runners/run_paper_modes.py").read_bytes()
    cold.frozen_spectral_contract(original, actual)
    with pytest.raises(ValueError, match="gates changed"):
        cold.frozen_spectral_contract(original, actual.replace(b'"halving": 1e-4', b'"halving": 1e-3'))
    with pytest.raises(ValueError, match="function changed"):
        cold.frozen_spectral_contract(original, actual.replace(b"solver.use_pardiso = False", b"solver.use_pardiso = True"))


@pytest.fixture
def case():
    value = cold.model.read_case(cold.ROOT / "validation/herradaEtAl_paper_model/cases/iii-ca-m0p0038-medium.json")
    value["case_id"] = "software_only_coarse_observation"
    return value


@pytest.mark.parametrize("alias", [False, True])
def test_consumer_refuses_outputs_inside_sealed_origin(case, tmp_path, monkeypatch, alias):
    origin = tmp_path / "sealed-origin"
    origin.mkdir()
    origin_manifest = origin / "run-manifest.json"
    write(origin_manifest, {"software_only": True})
    case_path = tmp_path / "case.json"
    write(case_path, case)
    named_origin = origin
    if alias:
        named_origin = tmp_path / "origin-alias"
        named_origin.symlink_to(origin, target_is_directory=True)
    fields = {name: {"path": str(case_path)} for name in ("packet", "state", "case", "archive")}
    fields["manifest"] = {"path": str(origin_manifest)}
    parent = {"case_sha256": cold.sha(case_path), "case_id": case["case_id"],
              "case_fields_used": cold.model.CASE_FIELDS_USED, "azimuthal_stability": True,
              "case_path": str(case_path), "case_file": str(case_path), "origin": fields}
    monkeypatch.setattr(cold, "CASE_SHA", cold.sha(case_path))
    # Closed-origin integrity has separate controls; exercise its authenticated path here.
    monkeypatch.setattr(cold, "origin_evidence", lambda *args: {"origin": fields})
    monkeypatch.setattr(cold, "source_contract", lambda *args, **kwargs: pytest.fail("source admission reached"))
    args = SimpleNamespace(case=case_path, out=named_origin / "new-output", check=True)
    before = origin_manifest.read_bytes()
    with pytest.raises(ValueError, match="output overlaps the sealed stationary origin"):
        cold.consumer_preflight(args, parent, case)
    assert not args.out.exists() and origin_manifest.read_bytes() == before


@pytest.mark.parametrize("coarse", [False, True])
def test_request_executor_calls_exactly_declared_searches(case, coarse, tmp_path, monkeypatch):
    calls = []
    request = modes.request_contract(case, modes.SHIFTS, 8, coarse_iii=coarse)
    prepared = {"case": case, "request": request, "execution_authorized": True,
                "restart": {"schema": cold.PACKET_SCHEMA}}
    def search(prepared, output, mode, shift):
        calls.append((mode, shift))
        return {"mode": mode, "shift": shift, "software_fixture": True}
    monkeypatch.setattr(modes, "spectral_stage", search)
    monkeypatch.setattr(modes, "compare_searches", lambda first, second, mode: {"mode": mode, "matched_returned_envelope": True})
    monkeypatch.setattr(modes, "plot_returned", lambda searches, case, out: {"software_fixture": True})
    modes.runner.append_progress(tmp_path, {"event": "software-prepared"}, first=True)
    result = modes.execute(prepared, tmp_path)
    assert calls == [(mode, shift) for mode in ((0,) if coarse else (0, 1)) for shift in modes.SHIFTS]
    assert request["eigenproblem_calls"] == len(calls) == (2 if coarse else 4)
    assert result["parent_acceptance_changed"] is False


@pytest.mark.parametrize("fault", ["modes", "shifts", "neigen", "calls", "seal", "packet"])
def test_coarse_request_refuses_fabricated_readiness_or_wider_search(case, fault):
    prepared = {"case": case, "request": modes.request_contract(case, modes.SHIFTS, 8, coarse_iii=True),
                "execution_authorized": True, "restart": {"schema": cold.PACKET_SCHEMA}}
    if fault == "seal":
        prepared["execution_authorized"] = False
    elif fault == "packet":
        prepared["restart"]["schema"] = cold.runner.RESUME_SCHEMA
    else:
        key = "eigenproblem_calls" if fault == "calls" else fault
        prepared["request"][key] = {"modes": [0, 1], "shifts": [-0.001, 0.002], "neigen": 16, "calls": 4}[fault]
    with pytest.raises(ValueError):
        modes.selected_modes(prepared)
    wrong = copy.deepcopy(case)
    wrong["physics"]["ca_in"] = cold.iii.PARENT_CA
    with pytest.raises(ValueError, match="exact typed III"):
        modes.request_contract(wrong, modes.SHIFTS, 8, coarse_iii=True)


def test_execution_requires_external_seal_and_dirty_inspection_remains_unready(tmp_path):
    component = {"commit": "a" * 40, "head": "a" * 40, "clean": True,
                 "sources": {"method.py": {"sha256": "source"}}}
    environment = {"commit": cold.diagnostic.PYOOMPH, "native_core": {"sha256": cold.diagnostic.CORE},
                   "interpreter": {"resolved_executable": "/software-only/python"}}
    records = []
    for name in ("case.json", "packet.json", "archive.json"):
        path = tmp_path / name
        write(path, {"software_only": name})
        records.append(cold.regular_record(path))
    args = SimpleNamespace(check=False, seal=None)
    with pytest.raises(ValueError, match="external source/case/packet/archive"):
        cold.execution_seal(args, component, {}, environment, records, "producer")
    args.check = True
    dirty = {**component, "clean": False, "commit": "a" * 40 + "-dirty"}
    expected, pending, record = cold.execution_seal(args, dirty, {}, environment, records, "producer")
    assert record is None and len(pending) == 2 and expected["source_commit"].endswith("-dirty")
    args.seal = tmp_path / "seal.json"
    write(args.seal, cold.seal_contract("producer", component, {}, environment, records))
    args.check = False
    _, pending, _ = cold.execution_seal(args, component, {}, environment, records, "producer")
    assert pending == []
    with pytest.raises(ValueError, match="execution seal differs"):
        cold.execution_seal(args, dirty, {}, environment, records, "producer")
    for record in records:
        path = Path(record["path"])
        saved = path.read_bytes()
        path.write_bytes(saved + b" ")
        mutated = [cold.regular_record(item["path"]) for item in records]
        with pytest.raises(ValueError, match="execution seal differs"):
            cold.execution_seal(args, component, {}, environment, mutated, "producer")
        path.write_bytes(saved)


@pytest.mark.parametrize("entrypoint", ["solve", "solve_eigenproblem", "force_remesh", "run", "linear"])
def test_zero_work_guard_rejects_work_and_restores_methods(entrypoint):
    solver = SimpleNamespace(idname="superlu", solve_serial=lambda: "original linear method", solve_distributed=lambda: None)
    problem = SimpleNamespace(solve=lambda: "original solve", solve_eigenproblem=lambda: None,
                              force_remesh=lambda: None, run=lambda: None, get_la_solver=lambda: solver)
    original = problem.solve
    with pytest.raises(RuntimeError, match="prohibits"), cold.zero_work(problem) as (_, guard):
        guard()
        (solver.solve_serial if entrypoint == "linear" else getattr(problem, entrypoint))()
    assert problem.solve is original and solver.solve_serial() == "original linear method"


@pytest.fixture
def synthetic_run(tmp_path, case, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "mesh.msh").write_bytes(b"synthetic mesh; not a native mesh")
    (origin / "state.dump").write_bytes(state_bytes("mesh.msh"))
    write(origin / "resume.json", {"software_only": True})
    write(origin / "run-manifest.json", {"software_only": True})
    case_path = origin / "case.json"
    write(case_path, case)
    pins = {"sha256": "software-only-pins"}
    names = ["globals/U", "liquid/velocity_phi", "gas/velocity_phi"]
    types = np.zeros(316989, dtype="<i8")
    z = np.linspace(0, 1, 1481)
    r = 0.5 * np.sin(np.pi * z)
    r[-1] = 0
    made = []
    class FakeProblem:
        def __init__(self):
            self._azimuthal_mode_param_m = SimpleNamespace(value=0)
            self.eigen_data_in_states = False
            self._last_bc_setting = "init"
            self._first_step = False
            self._taken_already_an_unsteady_step = True
            self._last_step_was_stationary = False
            self._last_eigenvalues = self._last_eigenvectors = None
            self._last_eigenvalues_m = self._last_eigenvalues_k = None
            self._suggested_next_dt = 4.0
            self._fields = np.zeros(316989)
            self._mask = np.zeros(316989, dtype=bool)
            self._pinned = np.zeros(3)
            self.reference = origin / "mesh.msh"
            self.solver = SimpleNamespace(idname="superlu", solve_serial=lambda: None, solve_distributed=lambda: None)
            self.timestepper = SimpleNamespace(time_pt=lambda: SimpleNamespace(ndt=lambda: 2, dt=lambda index: 1.0),
                                               get_num_unsteady_steps_done=lambda: 15)
            made.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def ndof(self):
            return 316989
        def get_dof_description(self):
            return types, names
        def is_normal_mode_stability_set_up(self):
            return "azimuthal"
        def get_current_dofs(self):
            return self._fields, self._mask
        def get_current_pinned_values(self, positions):
            return self._pinned
        def interface_polyline(self):
            return z, r
        def get_global_parameter(self, name):
            return SimpleNamespace(value={"Ca_in": cold.iii.TARGET_CA, "Bo": 1.56}[name])
        def observable(self, name, domain):
            return 4.0
        def frame_speed(self):
            return -0.003
        def get_current_time(self, **kwargs):
            return 8.0
        def _snapshot_state(self):
            return state_bytes(self.reference)
        def load_state(self, path, **kwargs):
            self.reference = Path(path).parent / "mesh.msh"
        def _restore_state(self, snapshot):
            if getattr(self, "bad_restore", False):
                self._fields[0] = 1.0
        def actions_before_stationary_solve(self, **kwargs):
            pass
        def invalidate_cached_mesh_data(self):
            pass
        def ntime_stepper(self):
            return 0
        def is_distributed(self):
            return False
        def get_la_solver(self):
            return self.solver
    baseline = FakeProblem()
    made.clear()
    signature = cold.model._stationary_state_signature(baseline, True)
    mesh = {**cold.regular_record(origin / "mesh.msh"), "stored_path": "mesh.msh"}
    restart = {"schema": cold.runner.RESUME_SCHEMA, "packet": cold.regular_record(origin / "resume.json"),
               "parent_manifest": cold.regular_record(origin / "run-manifest.json"),
               "state": {**cold.regular_record(origin / "state.dump"), "format": "portable_file"},
               "mesh_reference_base": str(origin), "meshes": [mesh],
               "mesh_exchange": {"mesh.msh": mesh["path"]},
               "stage": {"kind": "completed_polish", "bond_index": 16, "bond": 1.56,
                         "ca_in": cold.iii.TARGET_CA, "bc_mode": "init"},
               "roundtrip": {"after": signature, "layout": signature["layout"]}}
    monkeypatch.setattr(cold, "CASE_SHA", cold.sha(case_path))
    monkeypatch.setattr(cold, "COARSE_STATE_SHA", restart["state"]["sha256"])
    monkeypatch.setattr(cold, "COARSE_MESH_SHA", mesh["sha256"])
    monkeypatch.setattr(cold, "COARSE_PAYLOAD", signature["serialized_state"]["payload_sha256"])
    monkeypatch.setattr(cold, "COARSE_STREAM", hashlib.sha256(cold.history.normalized_bytes((origin / "state.dump").read_bytes(), origin)).hexdigest())
    monkeypatch.setattr(cold, "COARSE_PINS", pins["sha256"])
    monkeypatch.setattr(cold, "COARSE_MASK", signature["arrays"]["positional_dof_mask"]["sha256"])
    monkeypatch.setattr(cold, "COARSE_LAYOUT", signature["layout"]["dof_layout_sha256"])
    monkeypatch.setattr(cold.diagnostic, "actual_pins", lambda problem: copy.deepcopy(pins))
    monkeypatch.setattr(cold.model, "build_problem", lambda *args, **kwargs: FakeProblem())
    monkeypatch.setattr(cold.iii.anchor, "configure", lambda *args: None)
    monkeypatch.setattr(cold, "stable_inputs", lambda *args: None)
    monkeypatch.setattr(cold.remesh, "configuration", lambda problem, case, model: cold.declared_configuration(case))
    samples = []
    def sample(problem, model, case, directory, phase, np):
        value = {"phase": phase, "sampling_preserved_state": True, "actual_pins": copy.deepcopy(pins),
                 "configuration": cold.declared_configuration(case),
                 "signature": model._stationary_state_signature(problem, True),
                 "metrics": {"stationary_residual_linf": 1e-10, "lower_pressure_rms": 1e-10,
                             "gas_volume": 4.0, "surface_volume": 4.0, "target_volume": 4.0,
                             "lower_flux": 0.01, "upper_flux": 0.01}}
        samples.append(value)
        write(directory / "samples/cold-load.json", value)
        return value
    monkeypatch.setattr(cold.remesh, "sample", sample)
    output = tmp_path / "observed"
    output.mkdir()
    write(origin / "input-seal.json", {"software_only": True})
    prepared = {"case": case, "restart": restart, "warm": {"actual_pins": pins},
                "execution_authorized": True, "origin": {"software_only": True},
                "input_seal": cold.regular_record(origin / "input-seal.json"),
                "component": {"commit": "a" * 40}, "environment": {}, "compatibility": {},
                "implementation": {cold.ENTRYPOINT: cold.regular_record(cold.ROOT / cold.ENTRYPOINT)}}
    args = SimpleNamespace(case=case_path)
    parent = cold.manifest(args, prepared, [])
    write(output / "run-manifest.json", parent)
    return SimpleNamespace(args=args, prepared=prepared, parent=parent, output=output,
                           original={"restart": restart, "warm": {"actual_pins": pins},
                                     "origin": prepared["origin"]}, made=made, samples=samples,
                           sample=sample, baseline=baseline)


def complete_synthetic(b):
    result = cold.execute(b.args, b.prepared, b.output)
    write(b.output / "completion.json", {"schema": cold.SCHEMA, "status": "complete",
                                         "numerical_acceptance": True, "files": cold.iii.inventory(b.output),
                                         "scope": cold.SCOPE, **cold.NO_CLAIMS})
    packet = cold.read_packet(b.output / result["checkpoint"], b.output, b.prepared["case"], b.parent, b.original)
    return result, packet


def test_two_fresh_loads_full_restore_and_strict_typed_admission(synthetic_run):
    b = synthetic_run
    result, packet = complete_synthetic(b)
    summary, _, _ = cold.admit_observation(b.output, b.parent, packet, b.prepared["case"], b.original)
    assert len(b.made) == 2 and b.made[0] is not b.made[1]
    assert len(b.samples) == 2 and result["work_counts"] == cold.ZERO_WORK
    assert packet["stage"]["kind"] == cold.KIND
    assert packet["origin_completed_polish"] == b.prepared["restart"]["packet"]
    assert summary["fresh_cold_loads"] == 2
    for flag in cold.NO_CLAIMS:
        assert result[flag] is summary[flag] is packet[flag] is False
    assert b.prepared["restart"]["stage"]["kind"] == "completed_polish"


@pytest.mark.parametrize("fault", ["second_gate", "restore", "sample_drift"])
def test_warm_readiness_cannot_hide_failed_second_load_or_restore(synthetic_run, monkeypatch, fault):
    b = synthetic_run
    original_build = cold.model.build_problem
    def build(*args, **kwargs):
        problem = original_build(*args, **kwargs)
        if fault == "restore" and len(b.made) == 2:
            problem.bad_restore = True
        return problem
    monkeypatch.setattr(cold.model, "build_problem", build)
    def bad_sample(problem, *args):
        value = b.sample(problem, *args)
        if len(b.samples) == 2:
            if fault == "second_gate":
                value["metrics"]["stationary_residual_linf"] = 1e-5
            if fault == "sample_drift":
                problem._fields[0] = 1.0
        return value
    monkeypatch.setattr(cold.remesh, "sample", bad_sample)
    with pytest.raises((ValueError, modes.RestorationFailure)):
        cold.execute(b.args, b.prepared, b.output)
    assert len(b.made) == 2 and not (b.output / "result.json").exists()
    assert not (b.output / "completion.json").exists()


@pytest.mark.parametrize("fault", ["warm_only", "second_pins", "restoration", "packet_kind", "native_stream", "fine_mesh"])
def test_cold_admission_refuses_altered_evidence(synthetic_run, fault):
    b = synthetic_run
    result, packet = complete_synthetic(b)
    if fault == "warm_only":
        result["fresh_loads"] = result["fresh_loads"][:1]
        write(b.output / "result.json", result)
    elif fault == "second_pins":
        path = b.output / "fresh-2/load.json"
        value = cold.runner._read_json(path)
        value["actual_pins"]["sha256"] = "wrong pins"
        write(path, value)
    elif fault == "restoration":
        path = b.output / "fresh-2/restoration.json"
        value = cold.runner._read_json(path)
        value["after"]["flags"]["_first_step"] = True
        write(path, value)
    elif fault == "packet_kind":
        packet["stage"]["kind"] = "completed_polish"
        write(b.output / result["checkpoint"], packet)
    elif fault == "native_stream":
        Path(packet["state"]["path"]).write_bytes(state_bytes("mesh.msh", endtime=8))
    else:
        Path(packet["meshes"][0]["path"]).write_bytes(b"wrong synthetic fine mesh")
    completion = cold.runner._read_json(b.output / "completion.json")
    completion["files"] = cold.iii.inventory(b.output)
    write(b.output / "completion.json", completion)
    with pytest.raises(ValueError):
        checked = cold.read_packet(b.output / result["checkpoint"], b.output, b.prepared["case"], b.parent, b.original)
        cold.admit_observation(b.output, b.parent, checked, b.prepared["case"], b.original)


def test_metadata_inspection_cannot_construct_native_problem(synthetic_run, monkeypatch):
    b = synthetic_run
    b.prepared["execution_authorized"] = False
    monkeypatch.setattr(cold.model, "build_problem", lambda *args, **kwargs: pytest.fail("Problem construction was reached"))
    with pytest.raises(ValueError, match="metadata inspection"):
        cold.execute(b.args, b.prepared, b.output)
    assert not b.made


@pytest.mark.parametrize("fault", ["dofs", "layout", "mask", "ca", "bond", "history", "pins", "interface", "bc"])
def test_coarse_identity_refuses_fine_or_changed_state(synthetic_run, fault):
    b = synthetic_run
    signature = copy.deepcopy(b.prepared["restart"]["roundtrip"]["after"])
    pins = copy.deepcopy(b.prepared["warm"]["actual_pins"])
    if fault == "dofs":
        signature["layout"]["ndof"] = 542833
    elif fault == "layout":
        signature["layout"]["dof_layout_sha256"] = "wrong layout"
    elif fault == "mask":
        signature["arrays"]["positional_dof_mask"]["sha256"] = "wrong mask"
    elif fault == "ca":
        signature["parameters"]["Ca_in"] = cold.iii.PARENT_CA
    elif fault == "bond":
        signature["parameters"]["Bo"] = 1.55
    elif fault == "history":
        signature["serialized_state"]["payload_sha256"] = "changed history"
    elif fault == "pins":
        pins["sha256"] = "wrong pins"
    elif fault == "interface":
        signature["arrays"]["interface_z"]["shape"] = [2591]
    else:
        signature["bc_cache_marker"] = "transient"
    with pytest.raises(ValueError, match="coarse state/layout/forcing/history/pins"):
        cold.coarse_identity(signature, pins)


def test_stationary_signature_keeps_legacy_hash_and_complete_raw_stream(synthetic_run):
    b = synthetic_run
    before = b.baseline._snapshot_state()
    signature = cold.model._stationary_state_signature(b.baseline, True)
    raw_z, raw_r = b.baseline.interface_polyline()
    for name, raw in (("interface_z", raw_z), ("interface_r", raw_r)):
        legacy = np.asarray([round(float(value), 12) for value in raw])
        assert not np.array_equal(legacy, raw)
        assert signature["arrays"][name] == modes.array_record(legacy)
    assert signature["arrays"]["free_dofs"] == modes.array_record(b.baseline._fields)
    assert cold.runner.snapshot_bc_cache_signature(before, mesh_reference_base=cold.ROOT)[
        "payload_sha256"] == signature["serialized_state"]["payload_sha256"]
    assert cold.history.serialized_metadata(before)["endtime"] is None
    assert b.baseline._snapshot_state() == before

