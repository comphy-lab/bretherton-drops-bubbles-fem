"""Software controls only: synthetic state bytes never establish native evidence."""

from __future__ import annotations

import copy
import importlib.abc
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from runners import run_paper_cold_history as cold


def dump(reference, endtime=2.5, payload=b"opaque fields, pins, seven histories, globals"):
    def integer(value):
        return struct.pack("<q", value)

    def string(value):
        return integer(len(value)) + value.encode("ascii")

    return (
        string("pyoomph_dump")
        + string("0.1.6")
        + string("global")
        + struct.pack("<dqqqq", 2.5, 9, 0, 6, 2)
        + string(reference)
        + integer(1)
        + integer(4)
        + string("")
        + integer(0)
        + integer(3)
        + struct.pack("<dddqdd", 0.1, 0.2, 0.3, 12, 0.125, endtime)
        + payload
        + cold.runner._bc_cache_trailer("init")
    )


def fingerprint(snapshot, base):
    return cold.runner.snapshot_bc_cache_signature(snapshot, mesh_reference_base=base)


@pytest.fixture
def scalar_bundle(tmp_path):
    mesh = tmp_path / "parent.msh"
    mesh.write_bytes(b"synthetic mesh")
    other = tmp_path / "cold-longer-name.msh"
    other.write_bytes(mesh.read_bytes())
    saved = dump(mesh.name)
    before = dump(other.name, float("nan"))
    after = dump(other.name)
    return SimpleNamespace(
        saved=saved,
        before=before,
        after=after,
        base=tmp_path,
        expected=fingerprint(saved, tmp_path)["payload_sha256"],
    )


def test_independent_parser_follows_variable_mesh_entries_and_dt_count(scalar_bundle):
    metadata = cold.serialized_metadata(scalar_bundle.saved)
    assert metadata["dts"] == [0.1, 0.2, 0.3]
    assert metadata["unsteady_steps"] == 12
    assert metadata["suggested_next_dt"] == 0.125
    assert metadata["endtime"] == 2.5
    offset = metadata["endtime_offset"]
    assert scalar_bundle.saved[offset : offset + 8] == struct.pack("<d", 2.5)
    assert len(metadata["meshes"]) == 1  # The second template deliberately has no mesh file.


def test_loader_writer_asymmetry_requires_real_scalar_restoration(scalar_bundle):
    bundle = scalar_bundle
    problem = SimpleNamespace(_loaded_run_statement_endtime=2.5, _run_statement_endtime=None)
    assert fingerprint(bundle.before, bundle.base)["payload_sha256"] != bundle.expected
    assert (
        cold.restore_endtime(problem, bundle.saved, bundle.before, {"solve": {"relax_time": 2.5}})
        == 2.5
    )
    assert vars(problem) == {"_loaded_run_statement_endtime": 2.5, "_run_statement_endtime": 2.5}
    receipt = cold.verify_snapshots(
        bundle.saved, bundle.before, bundle.after, bundle.base, bundle.base, bundle.expected
    )
    assert receipt["verified"] and receipt["after"]["payload_sha256"] == bundle.expected


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_loaded",
        "missing_loaded",
        "missing_writer",
        "writer_set",
        "wrong_case",
        "cold_finite",
        "saved_nan",
    ],
)
def test_wrong_or_unavailable_metadata_refuses_without_assigning(scalar_bundle, fault):
    bundle = scalar_bundle
    problem = SimpleNamespace(_loaded_run_statement_endtime=2.5, _run_statement_endtime=None)
    case = {"solve": {"relax_time": 2.5}}
    saved, before = bundle.saved, bundle.before
    if fault == "wrong_loaded":
        problem._loaded_run_statement_endtime = 3.0
    elif fault == "missing_loaded":
        del problem._loaded_run_statement_endtime
    elif fault == "missing_writer":
        del problem._run_statement_endtime
    elif fault == "writer_set":
        problem._run_statement_endtime = 2.5
    elif fault == "wrong_case":
        case["solve"]["relax_time"] = 3.0
    elif fault == "cold_finite":
        before = bundle.after
    else:
        saved = dump("parent.msh", float("nan"))
    original = vars(problem).copy()
    with pytest.raises(ValueError):
        cold.restore_endtime(problem, saved, before, case)
    assert vars(problem) == original


@pytest.mark.parametrize(
    "fault",
    [
        "history",
        "dt",
        "footer",
        "extra_byte",
        "scalar",
        "wrong_hash",
        "unchanged",
        "parent_history",
        "mesh",
    ],
)
def test_full_byte_gate_refuses_nonendtime_changes_and_wrong_parent(scalar_bundle, fault):
    b = scalar_bundle
    saved, after, expected = b.saved, b.after, b.expected
    if fault == "history":
        after = after.replace(b"seven histories", b"eight histories")
    elif fault == "dt":
        after = after.replace(struct.pack("<d", 0.2), struct.pack("<d", 0.25), 1)
    elif fault == "footer":
        after = after[:-1] + b"X"
    elif fault == "extra_byte":
        after = after[:-19] + b"X" + after[-19:]
    elif fault == "scalar":
        after = dump("cold-longer-name.msh", 3.0)
    elif fault == "wrong_hash":
        expected = "0" * 64
    elif fault == "unchanged":
        after = b.before
    elif fault == "parent_history":
        saved = saved.replace(b"seven histories", b"eight histories")
    else:
        (b.base / "cold-longer-name.msh").write_bytes(b"changed mesh")
    with pytest.raises(ValueError):
        cold.verify_snapshots(saved, b.before, after, b.base, b.base, expected)


@pytest.mark.parametrize(
    "bad",
    [b"", b"not a state", dump("mesh.msh")[:-1], dump("mesh.msh").replace(b"0.1.6", b"0.1.5")],
)
def test_unsupported_or_truncated_serialization_refuses(bad):
    with pytest.raises(ValueError):
        cold.serialized_metadata(bad)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith("pyoomph.") or "_pyoomph" in fullname:
            raise AssertionError("native solver import is forbidden in this software lane")


def test_standalone_metadata_import_does_not_import_native_solver(tmp_path):
    program = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(cold.ROOT)!r})
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyoomph' or fullname.startswith('pyoomph.') or '_pyoomph' in fullname:
            raise AssertionError('native import during metadata preparation')
sys.meta_path.insert(0, NoNative())
from runners import run_paper_cold_history
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


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Actual case/restart validators with synthetic native environment and dump."""
    monkeypatch.setattr(sys, "meta_path", [NoNative(), *sys.meta_path])
    case = cold.runner.read_case(cold.ROOT / cold.seed.BASE_CASE)
    case["case_id"] = "software-cold-history"
    case["physics"]["ca_in"] = cold.seed.TARGET_CA
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    runtime = tmp_path / "failed-runtime"
    index = len(cold.model.stationary_bond_schedule(case)) - 1
    directory = runtime / "checkpoints" / f"bond_{index:03d}"
    directory.mkdir(parents=True)
    mesh = directory / "parent.msh"
    mesh.write_bytes(b"synthetic mesh")
    state = directory / "state.dump"
    state.write_bytes(dump(mesh.name, case["solve"]["relax_time"]))
    current = cold.runner.component_revision(cold.ROOT)
    current.update(clean=True, commit="a" * 40, head="a" * 40, status=[])
    parent_component = copy.deepcopy(current)
    parent_component.update(commit=cold.PARENT_SOURCE, head=cold.PARENT_SOURCE)
    paths = {*current["sources"], *cold.HELPERS, cold.seed.BASE_CASE}
    implementation = {
        relative: {"path": str(cold.ROOT / relative), "sha256": cold.sha(cold.ROOT / relative)}
        for relative in paths
    }
    environment = {
        "available": True,
        "version": "software-only",
        "url": "https://github.com/comphy-lab/pyoomph",
        "commit": cold.diagnostic.PYOOMPH,
        "direct_url": {"vcs_info": {"vcs": "git", "commit_id": cold.diagnostic.PYOOMPH}},
        "native_core": {"path": "software-core.so", "sha256": cold.diagnostic.CORE},
        "interpreter": {"resolved_executable": str(Path(sys.executable).resolve())},
    }
    producer = implementation["runners/run_paper_seed_recovery.py"]
    parent = {
        "schema": "pyoomph-runner-manifest-v1",
        "component_commit": cold.PARENT_SOURCE,
        "component": parent_component,
        "implementation": implementation,
        "runner": producer,
        "runner_sha256": producer["sha256"],
        "runner_path": producer["path"],
        "case_path": str(case_path),
        "case_file": str(case_path),
        "case_sha256": cold.sha(case_path),
        "case_id": case["case_id"],
        "case_fields_used": cold.runner.CASE_FIELDS_USED,
        "azimuthal_stability": True,
        "pyoomph": environment,
    }
    manifest = runtime / "run-manifest.json"
    manifest.write_text(json.dumps(parent))
    signature = {
        "arrays": {"all_fields": "software-current-values"},
        "layout": {"ndof": 3},
        "parameters": {"Ca_in": case["physics"]["ca_in"], "Bo": case["physics"]["bond"]},
        "time": 2.5,
        "volume": 1.0,
        "frame_speed": 0.2,
        "bc_cache_marker": "init",
        "serialized_state": fingerprint(state.read_bytes(), directory),
    }
    metrics = {
        "stationary_residual_linf": 0.0,
        "lower_pressure_rms": 0.0,
        "lower_flux": 1.0,
        "upper_flux": 1.0,
        "gas_volume": 1.0,
        "surface_volume": 1.0,
        "target_volume": 1.0,
    }
    pins = {"sha256": "software-actual-pins"}
    sample = {
        "sampling_preserved_state": True,
        "metrics": metrics,
        "signature": signature,
        "actual_pins": pins,
    }
    sample_path = runtime / "terminal/samples/after-polish.json"
    sample_path.parent.mkdir(parents=True)
    sample_path.write_text(json.dumps(sample))
    failure = {
        "status": "failed",
        "error_type": "ValueError",
        "error": "cold-loaded complete history differs",
        "numerical_acceptance": False,
        "inputs_unchanged": True,
    }
    (runtime / "failure.json").write_text(json.dumps(failure))
    packet = {
        "schema": cold.runner.RESUME_SCHEMA,
        "azimuthal_stability": True,
        "case_sha256": cold.sha(case_path),
        "parent_manifest": {"path": str(manifest), "sha256": cold.sha(manifest)},
        "state": {"path": str(state), "sha256": cold.sha(state), "format": "portable_file"},
        "mesh_reference_base": str(directory),
        "meshes": [{"stored_path": mesh.name, "path": str(mesh), "sha256": cold.sha(mesh)}],
        "stage": {
            "kind": "completed_polish",
            "bond_index": index,
            "bond": case["physics"]["bond"],
            "ca_in": case["physics"]["ca_in"],
            "bc_mode": "init",
        },
        "roundtrip": {"verified": True, "after": signature, "layout": signature["layout"]},
    }
    packet_path = directory / "resume.json"
    packet_path.write_text(json.dumps(packet))
    monkeypatch.setattr(
        cold.runner, "component_revision", lambda root=cold.ROOT: copy.deepcopy(current)
    )
    monkeypatch.setattr(cold.runner, "pyoomph_revision", lambda: copy.deepcopy(environment))

    def git(root, *args):
        if args[0] == "merge-base":
            return b""
        assert args[0] == "show"
        _, relative = args[1].split(":", 1)
        return (cold.ROOT / relative).read_bytes()

    monkeypatch.setattr(cold.diagnostic, "git", git)
    # Only the duplicated AST compatibility check is substituted; source_records above is real.
    monkeypatch.setattr(cold.runner, "_parent_source_contract", lambda *args: {"software": True})
    args = SimpleNamespace(
        source=cold.ROOT,
        case=case_path,
        manifest=manifest,
        packet=packet_path,
        state=state,
        out=tmp_path / "new/output",
    )
    return SimpleNamespace(
        args=args,
        current=current,
        parent=parent,
        packet=packet,
        sample=sample,
        signature=signature,
        metrics=metrics,
        pins=pins,
        runtime=runtime,
        mesh=mesh,
        state=state,
        git=git,
    )


def arguments(bundle):
    return [part for key, value in vars(bundle.args).items() for part in ("--" + key, str(value))]


def test_check_is_metadata_only_and_creates_no_output(bundle, monkeypatch):
    monkeypatch.setattr(cold, "execute", lambda *args: pytest.fail("execute during --check"))
    before = {str(path): path.read_bytes() for path in bundle.runtime.rglob("*") if path.is_file()}
    assert cold.main([*arguments(bundle), "--check"]) == 0
    assert not bundle.args.out.parent.exists()
    assert before == {
        str(path): path.read_bytes() for path in bundle.runtime.rglob("*") if path.is_file()
    }
    assert "pyoomph" not in sys.modules


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "helper",
        "producer",
        "parent",
        "case",
        "state",
        "mesh",
        "failure",
        "completion",
        "wrong_packet",
    ],
)
def test_parent_source_case_or_artifact_drift_refuses_before_output(bundle, monkeypatch, fault):
    b = bundle
    if fault == "source":
        b.current.update(clean=False, commit="a" * 40 + "-dirty")
    elif fault == "helper":

        def changed_git(root, *args):
            raw = b.git(root, *args)
            if args == ("show", "a" * 40 + ":runners/run_pressure_remesh.py"):
                return raw + b"changed"
            return raw

        monkeypatch.setattr(cold.diagnostic, "git", changed_git)
    elif fault == "producer":
        b.parent["runner_sha256"] = "0" * 64
    elif fault == "parent":
        b.parent["component_commit"] = "0" * 40
    elif fault == "case":
        b.args.case.write_bytes(b.args.case.read_bytes() + b" ")
    elif fault == "state":
        b.state.write_bytes(b.state.read_bytes() + b"changed")
    elif fault == "mesh":
        b.mesh.write_bytes(b"changed")
    elif fault == "failure":
        (b.runtime / "failure.json").write_text('{"status":"complete"}')
    elif fault == "completion":
        (b.runtime / "completion.json").write_text("{}")
    else:
        b.args.state = b.mesh
    if fault in {"producer", "parent"}:
        b.args.manifest.write_text(json.dumps(b.parent))
        b.packet["parent_manifest"]["sha256"] = cold.sha(b.args.manifest)
        b.args.packet.write_text(json.dumps(b.packet))
    with pytest.raises((ValueError, KeyError)):
        cold.main([*arguments(b), "--check"])
    assert not b.args.out.parent.exists()


class FakeProblem:
    """Python loader/writer contract fixture; solve/remesh operations are forbidden."""

    def __init__(self, bundle, output):
        self.bundle = bundle
        self.eigen_data_in_states = False
        self._run_statement_endtime = None
        self._loaded_run_statement_endtime = None
        self._suggested_next_dt = 0.125
        self.payload = b"opaque fields, pins, seven histories, globals"
        self.reference = str(bundle.mesh)
        time = SimpleNamespace(ndt=lambda: 3, dt=lambda index: [0.1, 0.2, 0.3][index])
        self.timestepper = SimpleNamespace(
            time_pt=lambda: time, get_num_unsteady_steps_done=lambda: 12
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def set_output_directory(self, *args):
        pass

    set_linear_solver = set_output_directory
    set_c_compiler = set_output_directory

    def setup_for_stability_analysis(self, **kwargs):
        assert kwargs == {"azimuthal_stability": True, "analytic_hessian": False}

    def initialise(self):
        pass

    def load_state(self, *args, **kwargs):
        # Model the pinned reader: only the loaded attribute receives the saved scalar.
        self._loaded_run_statement_endtime = cold.serialized_metadata(
            self.bundle.state.read_bytes()
        )["endtime"]

    def _snapshot_state(self):
        assert self.eigen_data_in_states is True
        endtime = self._run_statement_endtime
        return dump(self.reference, float("nan") if endtime is None else endtime, self.payload)

    def get_current_time(self, **kwargs):
        return 2.5

    def solve(self, *args, **kwargs):
        pytest.fail("diagnostic solved or evolved the fixture")

    force_remesh = solve


@pytest.mark.parametrize("fault", [None, "other_byte", "sampling", "sampling_error", "gate"])
def test_execution_retains_snapshots_and_failures_without_evolution(bundle, monkeypatch, fault):
    b = bundle
    fake = FakeProblem(b, b.args.out)
    monkeypatch.setattr(cold.model, "build_problem", lambda *args, **kwargs: fake)

    def signature(problem, az):
        assert az is True
        value = copy.deepcopy(b.signature)
        value["serialized_state"] = fingerprint(problem._snapshot_state(), Path.cwd())
        return value

    monkeypatch.setattr(cold.model, "_stationary_state_signature", signature)
    monkeypatch.setattr(cold.diagnostic, "actual_pins", lambda problem: b.pins)
    # The signature helper normally enables this option around its own native snapshot.
    fake.eigen_data_in_states = True
    if fault == "other_byte":
        fake.payload += b"changed history"

    def sample(*args):
        if fault == "sampling":
            fake.payload += b"changed by observation"
        if fault == "sampling_error":
            fake.payload += b"failed observation"
            raise ValueError("software sampling failure")
        metrics = b.metrics.copy()
        if fault == "gate":
            metrics["lower_pressure_rms"] = 1.0
        return {"metrics": metrics}

    monkeypatch.setattr(cold.remesh, "sample", sample)
    if fault:
        with pytest.raises(ValueError):
            cold.main(arguments(b))
        failure = json.loads((b.args.out / "failure.json").read_text())
        assert failure["numerical_acceptance"] is False
        assert failure["inputs_unchanged"] is True
        assert not (b.args.out / "completion.json").exists()
        if fault == "sampling_error":
            assert b"failed observation" in (b.args.out / "after-sampling.dump").read_bytes()
    else:
        assert cold.main(arguments(b)) == 0
        summary = json.loads((b.args.out / "summary.json").read_text())
        assert summary["mechanism_verified"] and summary["numerical_acceptance"]
        assert summary["parent_status"] == "failed" and not summary["parent_acceptance_changed"]
        manifest = json.loads((b.args.out / "run-manifest.json").read_text())
        assert manifest["runner_path"] == str(cold.ROOT / cold.ENTRYPOINT)
        assert manifest["runner_sha256"] == cold.sha(cold.ROOT / cold.ENTRYPOINT)
        assert manifest["argv"][1] == manifest["runner_path"]
        complete = json.loads((b.args.out / "completion.json").read_text())
        assert {"before.dump", "after.dump", "after-sampling.dump"} <= complete["files"].keys()
    assert (b.args.out / "before.dump").is_file() and (b.args.out / "after.dump").is_file()
    assert math.isnan(
        struct.unpack(
            "<d",
            (b.args.out / "before.dump").read_bytes()[
                cold.serialized_metadata((b.args.out / "before.dump").read_bytes())[
                    "endtime_offset"
                ] : cold.serialized_metadata((b.args.out / "before.dump").read_bytes())[
                    "endtime_offset"
                ]
                + 8
            ],
        )[0]
    )
    assert json.loads((b.runtime / "failure.json").read_text())["status"] == "failed"
