"""Software contract fixtures; no synthetic output is native evidence."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import test_paper_cold_history as fixtures
from test_paper_cold_history import FakeProblem, fingerprint

from runners import run_paper_seed_requalify as requal

cold = requal.cold
parent_bundle = fixtures.bundle


class Problem(FakeProblem):
    def set_output_directory(self, directory):
        self.output = directory

    def get_output_directory(self):
        return self.output


@pytest.fixture
def lane(parent_bundle, monkeypatch):
    b = parent_bundle
    b.current.update(commit=requal.DIAGNOSTIC_SOURCE, head=requal.DIAGNOSTIC_SOURCE)
    problems = []

    def build(*args, **kwargs):
        problem = Problem(b, b.args.out)
        problem.eigen_data_in_states = True
        problems.append(problem)
        return problem

    def signature(problem, az):
        value = copy.deepcopy(b.signature)
        value["serialized_state"] = fingerprint(problem._snapshot_state(), Path.cwd())
        return value

    def sample(problem, model, case, output, phase, np):
        value = {
            "metrics": b.metrics,
            "signature": signature(problem, True),
            "actual_pins": b.pins,
            "sampling_preserved_state": True,
        }
        directory = output / "samples"
        directory.mkdir(exist_ok=True)
        cold.write_json(directory / (phase + ".json"), value)
        return value

    monkeypatch.setattr(requal.model, "build_problem", build)
    monkeypatch.setattr(requal.model, "_stationary_state_signature", signature)
    monkeypatch.setattr(requal.diagnostic, "actual_pins", lambda p: b.pins)
    monkeypatch.setattr(requal.model, "_prepare_stationary_bc_state", lambda *a: {"verified": True})
    monkeypatch.setattr(requal.remesh, "sample", sample)
    b.args.out = b.runtime.parent / "diagnostic"
    original_arguments = [p for k, v in vars(b.args).items() for p in ("--" + k, str(v))]
    assert cold.main(original_arguments) == 0
    b.args.diagnostic_runtime = b.args.out
    b.args.out = b.runtime.parent / "descendant"
    b.problems = problems

    def checkpoint(problem, case, output, index, az):
        # Simulate relocation and manifest binding, never a native checkpoint.
        directory = output / "checkpoints" / f"bond_{index:03d}"
        directory.mkdir(parents=True)
        mesh = directory / b.mesh.name
        mesh.write_bytes(b.mesh.read_bytes())
        state = directory / "state.dump"
        state.write_bytes(b.state.read_bytes())
        packet = copy.deepcopy(b.packet)
        packet.update(
            parent_manifest=requal.record(output / "run-manifest.json"),
            state={**requal.record(state), "format": "portable_file"},
            mesh_reference_base=str(directory),
            meshes=[{**requal.record(mesh), "stored_path": mesh.name}],
        )
        requal.write_json(directory / "resume.json", packet)
        return {"resume_packet": str((directory / "resume.json").relative_to(output))}

    monkeypatch.setattr(requal.model, "_checkpoint_stationary_state", checkpoint)
    return b


def arguments(b):
    return [p for k, v in vars(b.args).items() for p in ("--" + k.replace("_", "-"), str(v))]


def resign(runtime):
    path = runtime / "completion.json"
    value = json.loads(path.read_text())
    value["files"] = {
        str(p.relative_to(runtime)): requal.sha(p)
        for p in runtime.rglob("*")
        if p.is_file() and p.name not in ("completion.json", "progress.jsonl")
    }
    path.write_text(json.dumps(value))


def test_check_uses_real_preflight_without_native_or_output(lane, monkeypatch):
    b = lane
    monkeypatch.setattr(requal, "execute", lambda *a: pytest.fail("execute in --check"))
    assert requal.main([*arguments(b), "--check"]) == 0
    assert not b.args.out.exists()
    assert "pyoomph" not in sys.modules


@pytest.mark.parametrize(
    "fault", ["source", "state", "hash", "input", "gate", "helper", "producer"]
)
def test_diagnostic_binding_refuses_forged_or_changed_evidence(lane, fault):
    b = lane
    runtime = b.args.diagnostic_runtime
    if fault == "hash":
        (runtime / "after.dump").write_bytes(b"changed")
    elif fault == "state":
        path = runtime / "run-manifest.json"
        value = json.loads(path.read_text())
        value["restart"]["state"]["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        resign(runtime)
    elif fault in ("source", "helper", "producer"):
        path = runtime / "run-manifest.json"
        value = json.loads(path.read_text())
        if fault == "source":
            value["component_commit"] = "0" * 40
        elif fault == "producer":
            value["runner_sha256"] = "0" * 64
        else:
            value["implementation"][cold.ENTRYPOINT]["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        resign(runtime)
    elif fault == "input":
        b.mesh.write_bytes(b"changed immutable parent mesh")
    else:
        path = runtime / "samples/cold-load.json"
        value = json.loads(path.read_text())
        value["metrics"]["lower_pressure_rms"] = 1.0
        path.write_text(json.dumps(value))
        resign(runtime)
    with pytest.raises(ValueError):
        requal.main([*arguments(b), "--check"])
    assert not b.args.out.exists()


def test_descendant_uses_new_packet_and_second_fresh_typed_load(lane):
    b = lane
    assert requal.main(arguments(b)) == 0
    assert len(b.problems) == 3  # One diagnostic, parent load, independent descendant load.
    receipt = json.loads((b.args.out / "cold-load.json").read_text())
    assert Path(receipt["packet"]["path"]).is_relative_to(b.args.out)
    assert receipt["parent_manifest"] == requal.record(b.args.out / "run-manifest.json")
    assert receipt["intervention"]["applied"] is True
    assert receipt["intervention"]["saved_offset"] != receipt["intervention"]["native_offset"]
    result = json.loads((b.args.out / "result.json").read_text())
    assert result["schema"] == requal.SCHEMA and result["numerical_acceptance"]
    assert result["parent_status"] == "failed" and not result["dynamic_restart_certificate"]
    completion = json.loads((b.args.out / "completion.json").read_text())
    for relative, digest in completion["files"].items():
        assert requal.sha(b.args.out / relative) == digest
    assert json.loads((b.runtime / "failure.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "fault",
    [
        "loaded_scalar",
        "writer_scalar",
        "history",
        "bc_history",
        "bc_pins",
        "secondary",
        "packet",
        "sampling_error",
    ],
)
def test_failure_retains_raw_proof_and_never_completes(lane, monkeypatch, fault):
    b = lane
    build = requal.model.build_problem

    def faulty_build(*a, **k):
        problem = build(*a, **k)
        if fault == "writer_scalar":
            problem._run_statement_endtime = 3.0
        if fault == "history" or (fault == "secondary" and len(b.problems) == 3):
            problem.payload += b"corrupted history"
        if fault == "loaded_scalar":
            load = problem.load_state

            def wrong_load(*a, **k):
                load(*a, **k)
                problem._loaded_run_statement_endtime = 3.0

            problem.load_state = wrong_load
        return problem

    monkeypatch.setattr(requal.model, "build_problem", faulty_build)
    if fault in ("bc_history", "bc_pins"):

        def prepare(problem, az):
            if fault == "bc_history":
                problem.payload += b"BC changed history"
            else:
                monkeypatch.setattr(requal.diagnostic, "actual_pins", lambda p: {"changed": True})
            return {"verified": True}

        monkeypatch.setattr(requal.model, "_prepare_stationary_bc_state", prepare)
    if fault == "packet":
        checkpoint = requal.model._checkpoint_stationary_state

        def wrong_packet(*a):
            value = checkpoint(*a)
            path = a[2] / value["resume_packet"]
            packet = json.loads(path.read_text())
            packet["parent_manifest"] = b.packet["parent_manifest"]
            path.write_text(json.dumps(packet))
            return value

        monkeypatch.setattr(requal.model, "_checkpoint_stationary_state", wrong_packet)
    if fault == "sampling_error":

        def broken_sample(problem, *a):
            problem.payload += b"failed observation retained"
            raise ValueError("sampling failed")

        monkeypatch.setattr(requal.remesh, "sample", broken_sample)
    with pytest.raises(ValueError):
        requal.main(arguments(b))
    failure = json.loads((b.args.out / "failure.json").read_text())
    assert not failure["numerical_acceptance"] and failure["inputs_unchanged"]
    assert not (b.args.out / "completion.json").exists()
    assert list(b.args.out.rglob("before.dump")) and list(b.args.out.rglob("after.dump"))
    if fault == "sampling_error":
        assert (
            b"failed observation retained"
            in (b.args.out / "requalified-after-sampling.dump").read_bytes()
        )


def test_warm_typed_load_requires_equality_without_reassignment(lane):
    b = lane
    prepared = requal.preflight(b.args)
    problem = requal.model.build_problem()
    requal.configure(problem, b.args.out / "warm-solver")
    expected = b.signature["serialized_state"]["payload_sha256"]
    first = requal.restore_checked(problem, prepared["case"], prepared["restart"], b.pins, expected)
    second = requal.restore_checked(
        problem, prepared["case"], prepared["restart"], b.pins, expected
    )
    assert first["intervention"]["applied"] and not second["intervention"]["applied"]
    problem._run_statement_endtime = 3.0
    with pytest.raises(ValueError, match="writer endtime differs"):
        requal.restore_checked(problem, prepared["case"], prepared["restart"], b.pins, expected)
    assert problem._run_statement_endtime == 3.0


def test_standalone_import_has_no_native_side_effects(tmp_path):
    program = f"""
import importlib.abc, sys
sys.path.insert(0, {str(requal.ROOT)!r})
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if 'pyoomph' in fullname:
            raise AssertionError('native import forbidden')
sys.meta_path.insert(0, NoNative())
from runners import run_paper_seed_requalify
assert 'pyoomph' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
