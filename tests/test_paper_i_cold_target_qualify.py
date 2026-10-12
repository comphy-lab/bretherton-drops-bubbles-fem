"""Software-only exact I cold-target qualification controls."""

from __future__ import annotations

import copy
import importlib.abc
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_paper_modes as original

from runners import run_paper_i_cold_target_qualify as target
from runners import run_paper_modes as modes

case = original.case
typed_i_bundle = original.typed_i_bundle


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith(("pyoomph.", "_pyoomph")):
            raise AssertionError("native import forbidden in I cold-target controls")


@pytest.fixture(autouse=True)
def no_native():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.mark.parametrize("index,ca", [(4, -0.002), (5, -0.0016)])
def test_exact_types_preserve_failed_suffix_remesh_scope(index, ca):
    identity = target.stage_type(index)
    assert identity["index"] == index and identity["ca_in"] == ca
    assert identity["suffix_terminal_scope"] == target.OOM_SCOPE
    assert identity["suffix_terminal_remesh_executed"] is True
    assert identity["selected_stage_remesh_attempted"] is (index == 5)
    assert "no spatial or spectral certificate" in identity["scope"]


@pytest.mark.parametrize("index", [0, 3, 6, True, "5"])
def test_other_stages_are_refused(index):
    with pytest.raises(ValueError):
        target.stage_type(index)


@pytest.fixture(params=[4, 5])
def qualified_bundle(typed_i_bundle, request, monkeypatch):
    b = typed_i_bundle
    index = request.param
    identity = target.stage_type(index)
    b.case["physics"]["ca_in"] = identity["ca_in"]
    b.cold["loaded"]["parameters"]["Ca_in"] = identity["ca_in"]
    b.restart["roundtrip"]["after"] = copy.deepcopy(b.cold["loaded"])
    b.restart["stage"].update(
        kind="completed_polish", ca_in=identity["ca_in"], bond=1.56, bond_index=16
    )
    b.restart["state"]["format"] = "portable_file"
    b.restart["meshes"][0]["stored_path"] = "mesh.msh"
    original_root = b.runtime.parent / (b.runtime.name + "-retained-original")
    original_root.mkdir()
    for name in ("state.dump", "mesh.msh", "packet.json"):
        (original_root / name).write_bytes((b.runtime / name).read_bytes())
    write(original_root / "run-manifest.json", {"software_original_failed": True})
    saved = copy.deepcopy(b.restart)
    saved.update(
        state={**target.record(original_root / "state.dump"), "format": "portable_file"},
        packet=target.record(original_root / "packet.json"),
        meshes=[{**target.record(original_root / "mesh.msh"), "stored_path": "mesh.msh"}],
        mesh_reference_base=str(original_root),
        mesh_exchange={"mesh.msh": str(original_root / "mesh.msh")},
        parent_manifest=target.record(original_root / "run-manifest.json"),
        compatibility={"current_commit": "software-source"},
    )
    case_path = original_root / "case.json"
    write(case_path, b.case)
    lineage = {
        "type": identity,
        "selected": {
            "case": target.record(case_path),
            "restart": saved,
            "pins": b.cold["actual_pins"],
        },
        "failed_suffix": {
            "manifest": target.record(original_root / "run-manifest.json"),
            "terminal": {"path": "software-OOM"},
            "custody": {"path": "software-archive"},
            "status": "failed",
            "accepted": False,
            "scope": target.OOM_SCOPE,
            "oom_at_stage": 5,
        },
        "new_budget": {"newton_updates": 0, "physical_remeshes": 0, "time_evolution": False},
    }
    implementation = {
        name: {"path": "/producer/" + name, "sha256": modes.sha(modes.ROOT / name)}
        for name in (*modes.HELPERS, *modes.PREFIX_I_HELPERS, target.ENTRYPOINT)
    }
    parent = {
        "runner": {"path": "/producer/" + target.ENTRYPOINT},
        "component_commit": "software-source",
        "lineage": lineage,
        "implementation": implementation,
        "qualification_schema": identity["schema"],
        "anchor_stage": {"kind": identity["kind"]},
        "original_acceptance_changed": False,
        **target.NO_CLAIMS,
    }
    write(b.runtime / "run-manifest.json", parent)
    b.restart["parent_manifest"] = target.record(b.runtime / "run-manifest.json")
    admitted = {
        "case": b.case,
        "restart": saved,
        "sample": {"actual_pins": b.cold["actual_pins"]},
        "lineage": lineage,
        "records": [],
        "implementation": implementation,
    }
    monkeypatch.setattr(target, "preflight", lambda *args: copy.deepcopy(admitted))
    write(
        b.runtime / "preparation.json",
        {
            "lineage": lineage,
            "inputs": [],
            "implementation": implementation,
            "numerical_acceptance": False,
        },
    )

    def typed(directory, receipt):
        directory.mkdir(parents=True)
        raw = (b.runtime / "state.dump").read_bytes()
        offset = target.cold.serialized_metadata(raw)["endtime_offset"]
        receipt["intervention"].update(applied=True, only_changed_slot=[offset, offset + 8])
        for phase in ("before", "after", "after-bc"):
            path = directory / (phase + ".dump")
            path.write_bytes(
                raw[:offset] + struct.pack("<d", float("nan")) + raw[offset + 8 :]
                if phase == "before"
                else raw
            )
            mesh = directory / (phase + "-meshes/000.msh")
            mesh.parent.mkdir()
            mesh.write_bytes((b.runtime / "mesh.msh").read_bytes())
            write(
                directory / (phase + "-retention.json"),
                {
                    "state": {**target.record(path), "format": "memory_snapshot"},
                    "mesh_reference_base": str(b.runtime),
                    "meshes": [{**target.record(mesh), "stored_path": "mesh.msh"}],
                    "mesh_exchange": {"mesh.msh": str(mesh)},
                },
            )
            if phase != "after-bc":
                receipt["intervention"][phase] = target.record(path)
        receipt["bc_preparation"] = {
            "verified": True,
            "method": "actions_before_stationary_solve(force_reassign_eqs=False)",
            "before": copy.deepcopy(receipt["loaded"]),
            "after": copy.deepcopy(receipt["loaded"]),
        }
        return receipt

    first = typed(b.runtime / "typed-loads/first", copy.deepcopy(b.cold))
    second = typed(b.runtime / "typed-loads/second", copy.deepcopy(b.cold))
    second.update(
        packet=b.restart["packet"],
        parent_manifest=b.restart["parent_manifest"],
        qualification_type=identity,
        checks=b.summary["checks"],
        canonical_checks=b.summary["checks"],
    )
    metrics = {
        name: value
        for name, value in b.summary.items()
        if name
        in {
            "stationary_residual_linf",
            "lower_pressure_rms",
            "lower_flux",
            "upper_flux",
            "gas_volume",
            "surface_volume",
            "target_volume",
        }
    }
    for phase, receipt in (("prefix", first), ("cold-load", second)):
        write(
            b.runtime / "samples" / (phase + ".json"),
            {
                "signature": receipt["loaded"],
                "actual_pins": receipt["actual_pins"],
                "sampling_preserved_state": True,
                "metrics": metrics,
            },
        )
        for suffix in ("before-sampling", "after-sampling"):
            (b.runtime / (phase + "-" + suffix + ".dump")).write_bytes(
                (b.runtime / "state.dump").read_bytes()
            )
    guard = {
        "installed_before_initialise": True,
        "methods": [
            "solve",
            "run",
            "force_remesh",
            "solve_eigenproblem",
            "steady_newton_solve",
            "unsteady_newton_solve",
        ],
        "violations": [],
        "solve_calls": 0,
        "physical_remesh_calls": 0,
        "time_evolution": False,
    }
    for name in ("first-zero-solve-guard.json", "second-zero-solve-guard.json"):
        write(b.runtime / name, guard)
    result = {
        "schema": identity["schema"],
        "type": identity,
        "lineage": lineage,
        "stationary_stage_accepted": True,
        "anchor_accepted": False,
        "original_attempt_status": "failed",
        "original_acceptance_changed": False,
        "numerical_acceptance": True,
        "cold_load_verified": True,
        "checks": b.summary["checks"],
        "cold_load_checks": b.summary["cold_load_checks"],
        "checkpoint": {"resume_packet": "packet.json"},
        **target.NO_CLAIMS,
    }
    summary = {**b.summary, **result, "ca_in": identity["ca_in"], "checkpoint": "packet.json"}
    write(b.runtime / "summary.json", summary)
    for phase in (
        "prefix-before-sampling",
        "prefix-after-sampling",
        "cold-load-before-sampling",
        "cold-load-after-sampling",
        "before-checkpoint",
        "after-checkpoint",
    ):
        path = b.runtime / (phase + ".dump")
        path.write_bytes((b.runtime / "state.dump").read_bytes())
        mesh = b.runtime / (phase + "-meshes/000.msh")
        mesh.parent.mkdir()
        mesh.write_bytes((b.runtime / "mesh.msh").read_bytes())
        write(
            b.runtime / (phase + "-retention.json"),
            {
                "state": {**target.record(path), "format": "memory_snapshot"},
                "mesh_reference_base": str(b.runtime),
                "meshes": [{**target.record(mesh), "stored_path": "mesh.msh"}],
                "mesh_exchange": {"mesh.msh": str(mesh)},
            },
        )
    return SimpleNamespace(
        runtime=b.runtime,
        case=b.case,
        parent=parent,
        restart=b.restart,
        first=first,
        second=second,
        result=result,
        summary=summary,
        identity=identity,
        admitted=admitted,
    )


def publish(b):
    for name, value in (
        ("run-manifest.json", b.parent),
        ("result.json", b.result),
        ("parent-cold-load.json", b.first),
        ("cold-load.json", b.second),
    ):
        write(b.runtime / name, value)
    completion = {
        "schema": b.identity["schema"],
        "status": "complete",
        "numerical_acceptance": True,
        "type": b.identity,
        "stationary_stage_accepted": True,
        "anchor_accepted": False,
        "original_acceptance_changed": False,
        "files": target.prefix.completion_inventory(b.runtime),
    }
    write(b.runtime / "completion.json", completion)
    return completion


def test_complete_two_fresh_typed_load_admission_preserves_portable_state(qualified_bundle):
    b = qualified_bundle
    completion = publish(b)
    records = target.admit(
        b.runtime, b.case, b.parent, b.restart, b.summary, b.second, completion["files"], completion
    )
    assert records and b.result["anchor_accepted"] is False
    assert target.cold.normalized_bytes(
        Path(b.restart["state"]["path"]).read_bytes(), Path(b.restart["mesh_reference_base"])
    ) == target.cold.normalized_bytes(
        Path(b.admitted["restart"]["state"]["path"]).read_bytes(),
        Path(b.admitted["restart"]["mesh_reference_base"]),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "history",
        "pins",
        "writer",
        "guard",
        "sample",
        "gate",
        "type",
        "ancestor_acceptance",
        "portable",
    ],
)
def test_qualification_refuses_altered_history_pins_predicates_scope_or_original_acceptance(
    qualified_bundle, fault
):
    b = qualified_bundle
    if fault == "history":
        b.second["loaded"]["serialized_state"]["payload_sha256"] = "changed"
    elif fault == "pins":
        b.first["actual_pins"] = {"sha256": "changed"}
    elif fault == "writer":
        b.second["intervention"]["saved_endtime"] = None
    elif fault == "guard":
        path = b.runtime / "second-zero-solve-guard.json"
        guard = modes.runner._read_json(path)
        guard["solve_calls"] = 1
        write(path, guard)
    elif fault == "sample":
        path = b.runtime / "cold-load-after-sampling.dump"
        path.write_bytes(path.read_bytes() + b"different pressure history")
    elif fault == "gate":
        b.result["cold_load_checks"] = {**b.result["cold_load_checks"], "volume": False}
    elif fault == "type":
        b.second["qualification_type"] = target.stage_type(5 if b.identity["index"] == 4 else 4)
    elif fault == "ancestor_acceptance":
        b.parent["original_acceptance_changed"] = True
    else:
        path = Path(b.restart["state"]["path"])
        path.write_bytes(path.read_bytes() + b"different opaque history")
    completion = publish(b)
    with pytest.raises((ValueError, OSError)):
        target.admit(
            b.runtime,
            b.case,
            b.parent,
            b.restart,
            b.summary,
            b.second,
            completion["files"],
            completion,
        )


def test_new_exact_I_requests_use_typed_loader_and_carry_scope(qualified_bundle, monkeypatch):
    b = qualified_bundle
    assert modes.request_contract(b.case, modes.SHIFTS, modes.NEIGEN)["branch_candidate"] == "I"
    prepared = {
        "case": b.case,
        "implementation": {
            target.ENTRYPOINT: {},
            modes.requalification.ENTRYPOINT: {"sha256": "frozen-loader"},
        },
        "restart": b.restart,
        "cold": b.second,
    }
    problem, calls = SimpleNamespace(), []
    monkeypatch.setattr(
        modes.requalification,
        "restore_checked",
        lambda *args: calls.append(args) or {"verified": True},
    )
    assert modes.restore_parent(problem, prepared, b.runtime) == {"verified": True}
    contract = modes.parent_load_contract(prepared)
    assert contract["kind"] == "typed-I-cold-target-stationary-only"
    assert (
        contract["qualification_type"] == b.identity
        and contract["spatial_convergence_certificate"] is False
    )
    assert (
        len(calls) == 1
        and problem._seed_requalification_receipts_directory == b.runtime / "typed-loads"
    )


def test_consumer_dispatches_exact_new_producer_without_old_terminal_remesh_route(
    qualified_bundle, monkeypatch
):
    b = qualified_bundle
    publish(b)
    calls = []
    monkeypatch.setattr(target, "admit", lambda *args: calls.append(args) or [])
    monkeypatch.setattr(
        modes, "qualified_i_parent", lambda *args: pytest.fail("old I terminal-remesh route used")
    )
    monkeypatch.setattr(
        modes,
        "qualified_prefix_i_parent",
        lambda *args: pytest.fail("old prefix I terminal-remesh route used"),
    )
    summary, cold, records = modes.qualified_parent(
        b.runtime, b.case, parent=b.parent, restart=b.restart
    )
    assert calls and records and summary["anchor_accepted"] is False
    assert cold["qualification_type"] == b.identity


def test_guarded_executor_uses_two_distinct_instances_without_new_solves(
    qualified_bundle, monkeypatch
):
    b = qualified_bundle
    prepared = {**b.admitted, "component": {}, "environment": {}}
    monkeypatch.chdir(b.runtime)
    write(b.runtime / "run-manifest.json", b.parent)
    b.restart["parent_manifest"] = target.record(b.runtime / "run-manifest.json")
    for name in (
        "parent-cold-load.json",
        "cold-load.json",
        "first-zero-solve-guard.json",
        "second-zero-solve-guard.json",
    ):
        (b.runtime / name).unlink(missing_ok=True)
    instances, loads = [], []

    class Problem:
        def __init__(self):
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def initialise(self):
            assert "ZeroSolveGuard" in self.solve.__qualname__

        def solve(self, *args):
            pytest.fail("solve attempted")

        run = force_remesh = solve_eigenproblem = steady_newton_solve = unsteady_newton_solve = (
            solve
        )
        set_output_directory = set_linear_solver = set_c_compiler = lambda *args: None
        setup_for_stability_analysis = lambda *args, **kwargs: None

    monkeypatch.setattr(target.model, "build_problem", lambda *args, **kwargs: Problem())
    monkeypatch.setattr(target.prefix, "stable_inputs", lambda *args: None)
    monkeypatch.setattr(target.runner, "verify_restart_artifacts", lambda *args: None)
    monkeypatch.setattr(target.runner, "validate_restart", lambda *args, **kwargs: b.restart)
    monkeypatch.setattr(target.diagnostic, "actual_pins", lambda *args: b.second["actual_pins"])
    monkeypatch.setattr(
        target.model, "_checkpoint_stationary_state", lambda *args: {"resume_packet": "packet.json"}
    )

    def restore(problem, case, packet, pins, expected):
        loads.append((problem, packet))
        return copy.deepcopy(b.first if len(loads) == 1 else b.second)

    monkeypatch.setattr(target.prefix.requalification, "restore_checked", restore)

    def retain(problem, output, phase):
        raw = Path(b.restart["state"]["path"]).read_bytes()
        (output / (phase + ".dump")).write_bytes(raw)
        return raw

    monkeypatch.setattr(target.cold, "retain_snapshot", retain)
    sample = modes.runner._read_json(b.runtime / "samples/prefix.json")
    monkeypatch.setattr(target.remesh, "sample", lambda *args: copy.deepcopy(sample))
    result = target.execute(
        SimpleNamespace(case=Path(b.admitted["lineage"]["selected"]["case"]["path"])),
        prepared,
        b.runtime,
    )
    assert len(instances) == 2 and instances[0] is not instances[1]
    assert loads[0][0] is instances[0] and loads[1][0] is instances[1]
    assert result["type"] == b.identity and result["stationary_stage_accepted"] is True
    assert result["anchor_accepted"] is False and all(
        result[name] is False for name in target.NO_CLAIMS
    )
    for name in ("first-zero-solve-guard.json", "second-zero-solve-guard.json"):
        assert modes.runner._read_json(b.runtime / name)["solve_calls"] == 0


def test_new_source_route_requires_exact_type_and_typed_I_pairing(qualified_bundle, monkeypatch):
    b = qualified_bundle
    sources = {
        name: {
            "sha256": modes.sha(modes.ROOT / name),
            "committed_sha256": modes.sha(modes.ROOT / name),
            "committed_matches": True,
        }
        for name in modes.runner.METHOD_FILES
    }
    component = {
        "commit": "a" * 40,
        "head": "a" * 40,
        "clean": True,
        "status": [],
        "sources": sources,
    }
    parent = copy.deepcopy(b.parent)
    parent.update(
        component=component,
        component_commit=component["commit"],
        runner={
            "path": "/producer/" + target.ENTRYPOINT,
            "sha256": modes.sha(modes.ROOT / target.ENTRYPOINT),
        },
        runner_path="/producer/" + target.ENTRYPOINT,
        runner_sha256=modes.sha(modes.ROOT / target.ENTRYPOINT),
    )

    def source(root, *args):
        return (
            b"" if args[0] == "merge-base" else (modes.ROOT / args[1].split(":", 1)[1]).read_bytes()
        )

    monkeypatch.setattr(modes.diagnostic, "git", source)
    monkeypatch.setattr(modes.runner, "_git", source)
    assert target.ENTRYPOINT in modes.source_contract(parent, component, typed_i=True)
    with pytest.raises(ValueError, match="typed I request"):
        modes.source_contract(parent, component)
    parent["qualification_schema"] = "old-producer-type"
    with pytest.raises(ValueError, match="kind/schema"):
        modes.source_contract(parent, component, typed_i=True)


def test_ca_m0p002_cannot_enter_the_old_I_producer_route(tmp_path, case):
    case["physics"]["ca_in"] = -0.002
    case_path = tmp_path / "case.json"
    manifest = tmp_path / "run-manifest.json"
    write(case_path, case)
    write(manifest, {"runner": {"path": "/old-producer/runners/run_paper_anchor_i_prefix.py"}})
    with pytest.raises(ValueError, match="exact new cold-target"):
        modes.preflight(
            SimpleNamespace(
                source=modes.ROOT,
                manifest=manifest,
                case=case_path,
                shifts=modes.SHIFTS,
                neigen=modes.NEIGEN,
                check=True,
            )
        )
