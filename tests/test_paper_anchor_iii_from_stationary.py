"""No-native controls for the exact stationary III continuation and typed load."""

from __future__ import annotations

import copy
import importlib.abc
import json
import struct
import sys
from types import SimpleNamespace

import pytest

from runners import run_paper_anchor_iii_from_stationary as iii
from runners import run_paper_modes as modes


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith(("pyoomph.", "_pyoomph")):
            raise AssertionError("native import forbidden in stationary III software tests")


@pytest.fixture(autouse=True)
def no_native():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def state_bytes(marker="init"):
    def integer(value):
        return struct.pack("<q", value)

    def string(value):
        return integer(len(value)) + value.encode()

    return (
        string("pyoomph_dump")
        + string("0.1.6")
        + string("global")
        + struct.pack("<dqqqq", 8.0, 9, 0, 6, 1)
        + string("mesh.msh")
        + integer(0)
        + integer(2)
        + struct.pack("<ddqdd", 1.0, 1.0, 15, 4.0, float("nan"))
        + b"opaque software fields and histories"
        + iii.runner._bc_cache_trailer(marker)
    )


@pytest.fixture
def parent_bundle(tmp_path, monkeypatch):
    parent_case = iii.runner.read_case(
        iii.ROOT / "validation/herradaEtAl_paper_model/cases/iii-ca-m0p0038-medium.json"
    )
    parent_case["case_id"] = "software_qualified_iii"
    parent_case["physics"]["ca_in"] = iii.PARENT_CA
    case_path = tmp_path / "case.json"
    write(case_path, parent_case)
    target = copy.deepcopy(parent_case)
    target["case_id"] = "software_exact_iii"
    target["physics"]["ca_in"] = iii.TARGET_CA
    target_path = tmp_path.parent / (tmp_path.name + "-target.json")
    write(target_path, target)
    ladder_path = tmp_path.parent / (tmp_path.name + "-ladder.json")
    write(ladder_path, {"schema": "herrada-paper-ca-ladder-v1", "values": list(iii.LADDER)})
    mesh = tmp_path / "mesh.msh"
    mesh.write_bytes(b"software-only mesh")
    state = tmp_path / "state.dump"
    state.write_bytes(state_bytes())
    signature = {
        "arrays": {"all": "software fields/masks/positions"},
        "layout": {"ndof": 2},
        "parameters": {"Ca_in": iii.PARENT_CA, "Bo": 1.56},
        "time": 8.0,
        "volume": 4.1887902047863905,
        "frame_speed": -0.00298,
        "bc_cache_marker": "init",
        "serialized_state": iii.runner.snapshot_bc_cache_signature(
            state.read_bytes(), mesh_reference_base=tmp_path
        ),
    }
    pins = {"sha256": "software-pins"}
    metrics = {
        "stationary_residual_linf": 1e-10,
        "lower_pressure_rms": 1e-10,
        "lower_flux": 0.01,
        "upper_flux": 0.01,
        "gas_volume": 4.1887902047863905,
        "surface_volume": 4.1887902047863905,
        "target_volume": 4.1887902047863905,
    }
    checks = {name: True for name in iii.GATES}
    environment = {
        "available": True,
        "commit": iii.diagnostic.PYOOMPH,
        "url": "https://github.com/comphy-lab/pyoomph",
        "direct_url": {"vcs_info": {"vcs": "git", "commit_id": iii.diagnostic.PYOOMPH}},
        "native_core": {"path": "/software/core.so", "sha256": iii.diagnostic.CORE},
        "interpreter": {"resolved_executable": "/software/python"},
    }
    sources = {
        name: {
            "sha256": iii.sha(iii.ROOT / name),
            "committed_sha256": iii.sha(iii.ROOT / name),
            "committed_matches": True,
        }
        for name in iii.runner.METHOD_FILES
    }
    component = {
        "head": "a" * 40,
        "commit": "a" * 40,
        "clean": True,
        "status": [],
        "sources": sources,
    }
    old = {**component, "head": iii.PARENT_SOURCE, "commit": iii.PARENT_SOURCE}
    implementation = {
        name: {"path": "/producer/" + name, "sha256": iii.sha(iii.ROOT / name)}
        for name in iii.OLD_HELPERS
    }
    manifest = {
        "schema": "pyoomph-runner-manifest-v1",
        "component_commit": iii.PARENT_SOURCE,
        "component": old,
        "pyoomph": environment,
        "implementation": implementation,
        "runner": implementation[iii.OLD_HELPERS[0]],
        "anchor_stage": {"kind": "stationary_III_anchor"},
        "parent_acceptance_changed": False,
        "azimuthal_stability": True,
        "case_path": str(case_path),
        "case_file": str(case_path),
        "case_sha256": iii.sha(case_path),
    }
    restart = {
        "state": {**iii.regular_record(state), "format": "portable_file"},
        "meshes": [{**iii.regular_record(mesh), "stored_path": "mesh.msh"}],
        "mesh_reference_base": str(tmp_path),
        "mesh_exchange": {"mesh.msh": str(mesh)},
        "stage": {
            "kind": "completed_polish",
            "bond_index": 16,
            "bond": 1.56,
            "ca_in": iii.PARENT_CA,
            "bc_mode": "init",
        },
        "roundtrip": {"after": signature, "layout": signature["layout"], "verified": True},
        "compatibility": {
            "current_commit": component["commit"],
            "parent_commit": iii.PARENT_SOURCE,
        },
    }
    packet_path = tmp_path / "packet.json"
    write(packet_path, {"software_only": True})
    restart["packet"] = iii.regular_record(packet_path)
    cold = {
        "verified": True,
        "loaded": copy.deepcopy(signature),
        "actual_pins": pins,
        "canonical_checks": checks,
        "bc_preparation": {
            "verified": True,
            "method": "actions_before_stationary_solve(force_reassign_eqs=False)",
            "before": copy.deepcopy(signature),
            "after": copy.deepcopy(signature),
        },
    }
    result = {
        "schema": "private-paper-stationary-anchor-v1",
        "numerical_acceptance": True,
        "completion_reached": True,
        "parent_acceptance_changed": False,
        "ca_in": iii.PARENT_CA,
        "canonical_checks": checks,
        "cold_load_checks": checks,
        "branch_identity_verified": False,
        "eigenanalysis_performed": False,
        "terminal": {"after_polish": {"signature": copy.deepcopy(signature), "actual_pins": pins}},
    }
    summary = {
        **metrics,
        "ca_in": iii.PARENT_CA,
        "completion_reached": True,
        "cold_load_verified": True,
        "checks": checks,
        "cold_load_checks": checks,
        "checkpoint": packet_path.name,
        "branch_identity_verified": False,
        "eigenanalysis_performed": False,
    }
    b = SimpleNamespace(
        runtime=tmp_path,
        parent=parent_case,
        target=target,
        case_path=case_path,
        target_path=target_path,
        ladder_path=ladder_path,
        state=state,
        signature=signature,
        pins=pins,
        metrics=metrics,
        manifest=manifest,
        restart=restart,
        cold=cold,
        result=result,
        summary=summary,
        component=component,
        environment=environment,
    )
    monkeypatch.setattr(iii, "PARENT_CASE_SHA", iii.sha(case_path))
    monkeypatch.setattr(iii, "PARENT_STATE_SHA", iii.sha(state))
    monkeypatch.setattr(iii.runner, "component_revision", lambda root: copy.deepcopy(component))
    monkeypatch.setattr(iii.runner, "pyoomph_revision", lambda: copy.deepcopy(environment))

    def git(root, *args):
        if args[0] == "merge-base":
            return b""
        return (iii.ROOT / args[1].split(":", 1)[1]).read_bytes()

    monkeypatch.setattr(iii.diagnostic, "git", git)
    monkeypatch.setattr(
        iii.runner, "validate_restart", lambda *args, **kwargs: copy.deepcopy(restart)
    )

    def publish():
        for name, item in {
            "run-manifest.json": manifest,
            "result.json": result,
            "summary.json": summary,
            "cold-load.json": cold,
        }.items():
            write(tmp_path / name, item)
        restart["parent_manifest"] = iii.regular_record(tmp_path / "run-manifest.json")
        files = {
            str(path.relative_to(tmp_path)): iii.sha(path)
            for path in tmp_path.rglob("*")
            if path.is_file() and path.name != "completion.json"
        }
        write(
            tmp_path / "completion.json",
            {"status": "complete", "numerical_acceptance": True, "files": files},
        )
        monkeypatch.setattr(iii, "PARENT_COMPLETION_SHA", iii.sha(tmp_path / "completion.json"))

    b.publish = publish
    b.args = SimpleNamespace(
        source=iii.ROOT,
        case=target_path,
        ca_ladder=ladder_path,
        manifest=tmp_path / "run-manifest.json",
        packet=packet_path,
        state=state,
        out=tmp_path.parent / (tmp_path.name + "-output"),
        check=True,
    )
    publish()
    return b


def test_exact_parent_and_fixed_ladder_metadata_admission(parent_bundle):
    b = parent_bundle
    prepared = iii.preflight(b.args)
    assert prepared["values"] == [-0.0035, -0.00375, -0.0038]
    assert prepared["contract"]["total_newton_update_bound"] == 640
    assert prepared["contract"]["per_mesh_stage_newton_cap"] == 40
    assert prepared["contract"]["per_polish_remesh_bound"] == 3
    assert iii.history.serialized_metadata(b.state.read_bytes())["endtime"] is None
    assert not b.args.out.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "kind",
        "source",
        "ca",
        "geometry",
        "core",
        "acceptance",
        "pressure",
        "cold_gate",
        "history",
        "pins",
        "marker",
        "finite_marker",
        "ladder",
        "budget",
    ],
)
def test_parent_target_and_history_refusals(parent_bundle, fault):
    b = parent_bundle
    field_faults = {
        "kind": (b.manifest["anchor_stage"], "kind", "stationary_I_candidate"),
        "source": (b.manifest, "component_commit", "0" * 40),
        "core": (b.environment["native_core"], "sha256", "changed"),
        "acceptance": (b.result, "numerical_acceptance", False),
        "pressure": (b.summary, "lower_pressure_rms", 1.0),
        "history": (b.cold["loaded"]["serialized_state"], "payload_sha256", "changed"),
        "pins": (b.cold, "actual_pins", {"sha256": "changed"}),
        "marker": (b.restart["state"], "sha256", "0" * 64),
    }
    if fault in field_faults:
        record, key, value = field_faults[fault]
        record[key] = value
    elif fault in {"ca", "geometry", "budget"}:
        value = copy.deepcopy(b.target)
        if fault == "ca":
            value["physics"]["ca_in"] = -0.0039
        elif fault == "geometry":
            value["physics"]["domain_over_r"], value["physics"]["upper_tip_over_r"] = 20.0, 40 / 3
        else:
            value["solve"]["max_newton_iterations"] = 41
        write(b.target_path, value)
    elif fault == "cold_gate":
        b.cold["canonical_checks"] = {**b.cold["canonical_checks"], "stationary": False}
    elif fault == "finite_marker":
        raw = b.state.read_bytes()
        offset = iii.history.serialized_metadata(raw)["endtime_offset"]
        b.state.write_bytes(raw[:offset] + struct.pack("<d", 8.0) + raw[offset + 8 :])
        b.restart["state"]["sha256"] = iii.sha(b.state)
    else:
        write(b.ladder_path, {"schema": "herrada-paper-ca-ladder-v1", "values": [-0.0036, -0.0038]})
    b.publish()
    with pytest.raises(ValueError):
        iii.preflight(b.args)
    assert not b.args.out.exists()


@pytest.mark.parametrize(
    "field,value",
    [("stage_updates", 41), ("updates_total", 161), ("remeshes_total", 4), ("status", "failed")],
)
def test_original_polish_limits_are_not_relaxed(field, value):
    record = {"status": "converged", "stage_updates": 1, "updates_total": 1, "remeshes_total": 0}
    record[field] = value
    assert not iii.polish_bounds(record)


def test_new_target_request_and_old_requests_are_exact():
    case = iii.runner.read_case(
        iii.ROOT / "validation/herradaEtAl_paper_model/cases/iii-ca-m0p0038-medium.json"
    )
    assert modes.request_contract(case, modes.SHIFTS, modes.NEIGEN)["branch_candidate"] == "III"
    for ca, branch in ((-0.0016, "I"), (-0.00315, "III")):
        old = copy.deepcopy(case)
        old["physics"]["ca_in"] = ca
        assert modes.request_contract(old, modes.SHIFTS, modes.NEIGEN)["branch_candidate"] == branch
    wrong = copy.deepcopy(case)
    wrong["physics"]["ca_in"] = -0.0039
    with pytest.raises(ValueError):
        modes.request_contract(wrong, modes.SHIFTS, modes.NEIGEN)


def test_exact_III_loader_is_selected_without_I_marker_assignment(tmp_path, monkeypatch):
    case = {"physics": {"ca_in": iii.TARGET_CA}}
    prepared = {
        "case": case,
        "restart": {"packet": {"path": "software-packet"}},
        "cold": {
            "actual_pins": {},
            "loaded": {"serialized_state": {"payload_sha256": "software-history"}},
        },
        "implementation": {iii.ENTRYPOINT: {"sha256": "software-loader"}},
    }
    problem = SimpleNamespace()
    calls = []

    def restore(*args):
        calls.append(args)
        return {"software_only": True}

    monkeypatch.setattr(iii, "restore_checked", restore)
    monkeypatch.setattr(
        modes.requalification, "restore_checked", lambda *args: pytest.fail("I restoration used")
    )
    assert modes.restore_parent(problem, prepared, tmp_path) == {"software_only": True}
    assert modes.parent_load_contract(prepared)["kind"] == "preserved-stationary-III-history"
    assert problem._stationary_iii_receipts_directory == tmp_path / "stationary-loads"
    assert len(calls) == 1


@pytest.mark.parametrize("fault", [None, "history", "pin", "writer", "time", "bc"])
def test_preserved_III_load_raw_receipt_without_endtime_assignment(
    parent_bundle, monkeypatch, fault
):
    b = parent_bundle
    output = b.runtime.parent / (b.runtime.name + "-loads")
    output.mkdir()
    problem = SimpleNamespace(_stationary_iii_receipts_directory=output)
    base_receipt = {
        "verified": True,
        "loaded": copy.deepcopy(b.signature),
        "actual_pins": b.pins,
        "bc_preparation": copy.deepcopy(b.cold["bc_preparation"]),
    }
    monkeypatch.setattr(iii.anchor, "restore_checked", lambda *args: copy.deepcopy(base_receipt))
    metadata = iii.history.serialized_metadata(b.state.read_bytes())
    monkeypatch.setattr(
        iii.history,
        "check_time_history",
        lambda *args: {
            key: metadata[key] for key in ("time", "dts", "unsteady_steps", "suggested_next_dt")
        },
    )
    monkeypatch.chdir(b.runtime)

    def retain(problem, directory, phase):
        raw = b.state.read_bytes()
        path = directory / (phase + ".dump")
        path.write_bytes(raw)
        meshes = directory / (phase + "-meshes")
        meshes.mkdir()
        retained = meshes / "000.msh"
        retained.write_bytes((b.runtime / "mesh.msh").read_bytes())
        write(
            directory / (phase + "-retention.json"),
            {
                "state": {**iii.regular_record(path), "format": "memory_snapshot"},
                "mesh_reference_base": str(b.runtime),
                "meshes": [{**iii.regular_record(retained), "stored_path": "mesh.msh"}],
                "mesh_exchange": {"mesh.msh": str(retained)},
            },
        )
        return raw

    monkeypatch.setattr(iii.history, "retain_snapshot", retain)
    receipt = iii.restore_checked(
        problem, b.parent, b.restart, b.pins, b.signature["serialized_state"]["payload_sha256"]
    )
    assert receipt["history_intervention"] is False and receipt["writer_endtime"] is None
    assert not hasattr(problem, "_run_statement_endtime")
    if fault == "history":
        receipt["loaded"]["serialized_state"]["payload_sha256"] = "changed"
    elif fault == "pin":
        receipt["actual_pins"] = {"sha256": "changed"}
    elif fault == "writer":
        receipt["writer_endtime"] = 8.0
    elif fault == "time":
        receipt["time_history"]["unsteady_steps"] = 0
    elif fault == "bc":
        receipt["bc_preparation"]["verified"] = False
    files = {
        str(path.relative_to(output)): iii.sha(path) for path in output.rglob("*") if path.is_file()
    }
    if fault is None:
        records = iii.validate_load(
            receipt,
            b.restart,
            b.parent,
            b.pins,
            b.signature["serialized_state"]["payload_sha256"],
            output,
            files,
        )
        assert len(records) == 3
    else:
        with pytest.raises(ValueError):
            iii.validate_load(
                receipt,
                b.restart,
                b.parent,
                b.pins,
                b.signature["serialized_state"]["payload_sha256"],
                output,
                files,
            )


def test_exact_execution_has_three_provisional_steps_and_two_distinct_loads(
    parent_bundle, monkeypatch
):
    b = parent_bundle
    prepared = iii.preflight(b.args)
    output = b.args.out
    output.mkdir()
    write(
        output / "run-manifest.json",
        iii.manifest(b.args.case, b.target, prepared, {"kind": iii.KIND}, []),
    )
    events, instances = [], []

    class Problem:
        def __init__(self, case):
            self.case = copy.deepcopy(case)
            self.identity = len(instances)
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(iii.model, "build_problem", lambda case, **kwargs: Problem(case))
    monkeypatch.setattr(
        iii.anchor,
        "configure",
        lambda problem, *args: events.append(("configure", problem.identity)),
    )
    monkeypatch.setattr(iii.anchor, "stable_inputs", lambda *args: None)

    def load(problem, case, restart, pins, payload):
        events.append(("load", problem.identity, case["physics"]["ca_in"]))
        return {"verified": True, "software_only": True}

    monkeypatch.setattr(iii, "restore_checked", load)

    def step(problem, prepared, previous, index, value, output, argv, np):
        events.append(("stage", index, value))
        problem.case = iii.anchor.stage_case(prepared["target"], value, index, index == 3)
        return problem.case, {"index": index, "ca_in": value, "numerical_acceptance": False}

    monkeypatch.setattr(iii, "stage", step)

    def sample(problem, model, case, directory, phase, np):
        sig = copy.deepcopy(b.signature)
        sig["parameters"]["Ca_in"] = case["physics"]["ca_in"]
        return {
            "metrics": copy.deepcopy(b.metrics),
            "signature": sig,
            "actual_pins": b.pins,
            "sampling_preserved_state": True,
        }

    monkeypatch.setattr(iii.remesh, "sample", sample)
    monkeypatch.setattr(
        iii.remesh, "normalize_pressure_dummies", lambda *args: events.append(("normalize", 1))
    )
    monkeypatch.setattr(
        iii.remesh,
        "retain_portable_sample",
        lambda *args: {"numerical_acceptance": False, "software_only": True},
    )

    def terminal(problem, model, case, directory, observe):
        return {
            "before": observe("before"),
            "after_remesh": observe("after-remesh"),
            "after_polish": observe("after-polish"),
            "explicit_force_remesh_calls": 1,
            "polish": {
                "status": "converged",
                "stage_updates": 1,
                "updates_total": 1,
                "remeshes_total": 0,
            },
        }

    monkeypatch.setattr(iii.remesh, "remesh_and_polish", terminal)
    monkeypatch.setattr(
        iii.model,
        "_checkpoint_stationary_state",
        lambda *args: {"resume_packet": "software-final-packet.json"},
    )
    final = copy.deepcopy(b.restart)
    final["roundtrip"]["after"]["parameters"]["Ca_in"] = iii.TARGET_CA
    monkeypatch.setattr(iii.runner, "validate_restart", lambda *args, **kwargs: final)
    monkeypatch.setattr(iii.diagnostic, "actual_pins", lambda *args: b.pins)
    result = iii.execute(b.args, prepared, output, [])
    assert len(instances) == 2 and instances[0] is not instances[1]
    assert [event for event in events if event[0] == "stage"] == [
        ("stage", 1, -0.0035),
        ("stage", 2, -0.00375),
        ("stage", 3, -0.0038),
    ]
    assert [event for event in events if event[0] == "load"] == [
        ("load", 0, -0.00315),
        ("load", 1, -0.0038),
    ]
    assert events.count(("normalize", 1)) == 1
    assert all(row["numerical_acceptance"] is False for row in result["stages"])
    assert result["numerical_acceptance"] is True
    assert all(result[key] is False for key in iii.NO_CLAIMS)


@pytest.fixture
def terminal_bundle(parent_bundle, monkeypatch):
    """Retain a complete software-only terminal closure, including raw load proofs."""
    b = parent_bundle
    prepared = iii.preflight(b.args)
    runtime = b.args.out
    runtime.mkdir()
    parent = iii.manifest(b.args.case, b.target, prepared, {"kind": iii.KIND}, [])
    write(runtime / "run-manifest.json", parent)
    write(
        runtime / "preparation.json",
        {
            "contract": prepared["contract"],
            "inputs": prepared["records"],
            "implementation": prepared["implementation"],
            "numerical_acceptance": False,
        },
    )
    packets = {str(b.args.packet.resolve()): copy.deepcopy(b.restart)}
    polish = {"status": "converged", "stage_updates": 1, "updates_total": 1, "remeshes_total": 0}

    def packet(directory, case):
        state, mesh, packet_path = (
            directory / name for name in ("state.dump", "mesh.msh", "packet.json")
        )
        state.write_bytes(state_bytes())
        mesh.write_bytes(b.state.with_name("mesh.msh").read_bytes())
        write(packet_path, {"software_only": True})
        signature = copy.deepcopy(b.signature)
        signature["parameters"]["Ca_in"] = case["physics"]["ca_in"]
        signature["serialized_state"] = iii.runner.snapshot_bc_cache_signature(
            state.read_bytes(), mesh_reference_base=directory
        )
        restart = copy.deepcopy(b.restart)
        restart.update(
            state={**iii.regular_record(state), "format": "portable_file"},
            meshes=[{**iii.regular_record(mesh), "stored_path": "mesh.msh"}],
            mesh_reference_base=str(directory),
            mesh_exchange={"mesh.msh": str(mesh)},
            packet=iii.regular_record(packet_path),
            parent_manifest=iii.regular_record(directory / "run-manifest.json"),
            roundtrip={"after": signature, "layout": signature["layout"], "verified": True},
        )
        restart["stage"]["ca_in"] = case["physics"]["ca_in"]
        packets[str(packet_path.resolve())] = restart
        sample = {
            "signature": copy.deepcopy(signature),
            "metrics": copy.deepcopy(b.metrics),
            "actual_pins": copy.deepcopy(b.pins),
            "sampling_preserved_state": True,
        }
        return restart, sample

    stages = []
    for index, value in enumerate(iii.LADDER, 1):
        directory = runtime / "stages" / f"ca_{index:03d}"
        current = iii.anchor.stage_case(b.target, value, index, index == len(iii.LADDER))
        case_path = directory / "case.json"
        write(case_path, current)
        write(
            directory / "run-manifest.json",
            iii.manifest(
                case_path,
                current,
                prepared,
                {"kind": "stationary_III_monotone_step", "index": index, "ca_in": value},
                [],
            ),
        )
        restart, sample = packet(directory, current)
        row = {
            "index": index,
            "ca_in": value,
            "case": iii.regular_record(case_path),
            "transition": {"verified": True},
            "polish": copy.deepcopy(polish),
            "checkpoint": {"resume_packet": "packet.json"},
            "resume_packet_sha256": restart["packet"]["sha256"],
            "canonical_checks_observed": iii.remesh.canonical_checks(current, sample["metrics"]),
            "numerical_acceptance": False,
        }
        write(directory / "stage.json", row)
        write(directory / "transition.json", row["transition"])
        write(directory / "polish/ca_step/polish.json", polish)
        write(directory / "samples/after-polish.json", sample)
        stages.append(row)
    final, warm = packet(runtime, b.target)
    write(runtime / "terminal/samples/after-polish.json", warm)
    write(runtime / "terminal/polish/terminal_remesh/polish.json", polish)

    def load(restart, directory):
        directory.mkdir()
        raw = directory / "after-bc.dump"
        raw.write_bytes(state_bytes())
        mesh = directory / "mesh.msh"
        mesh.write_bytes(b.state.with_name("mesh.msh").read_bytes())
        retention_path = directory / "after-bc-retention.json"
        write(
            retention_path,
            {
                "state": {**iii.regular_record(raw), "format": "memory_snapshot"},
                "mesh_reference_base": str(directory),
                "meshes": [{**iii.regular_record(mesh), "stored_path": "mesh.msh"}],
                "mesh_exchange": {"mesh.msh": str(mesh)},
            },
        )
        signature = copy.deepcopy(restart["roundtrip"]["after"])
        metadata = iii.history.serialized_metadata(raw.read_bytes())
        return {
            "schema": "preserved-stationary-III-history-load-v1",
            "verified": True,
            "history_intervention": False,
            "writer_endtime": None,
            "actual_pins": copy.deepcopy(b.pins),
            "loaded": signature,
            "bc_preparation": {
                "verified": True,
                "method": "actions_before_stationary_solve(force_reassign_eqs=False)",
                "before": copy.deepcopy(signature),
                "after": copy.deepcopy(signature),
            },
            "time_history": {
                key: metadata[key] for key in ("time", "dts", "unsteady_steps", "suggested_next_dt")
            },
            "raw_after_bc": iii.regular_record(raw),
            "retention": iii.regular_record(retention_path),
        }

    parent_load = load(b.restart, runtime / "parent-load-proof")
    write(runtime / "parent-load.json", parent_load)
    initial = {**copy.deepcopy(warm), "signature": copy.deepcopy(b.signature)}
    write(runtime / "parent-observation/samples/cold-load.json", initial)
    cold = load(final, runtime / "target-load-proof")
    checks = iii.remesh.canonical_checks(b.target, warm["metrics"])
    cold["canonical_checks"] = checks
    write(runtime / "cold-load.json", cold)
    write(runtime / "target-cold/samples/cold-load.json", warm)
    result = {
        "schema": iii.SCHEMA,
        "stages": stages,
        "terminal": {
            "explicit_force_remesh_calls": 1,
            "polish": polish,
            "after_polish": warm,
        },
        "checkpoint": {"resume_packet": "packet.json"},
        "ca_in": iii.TARGET_CA,
        "canonical_checks": checks,
        "cold_load_checks": checks,
        "cold_load_verified": True,
        "completion_reached": True,
        "numerical_acceptance": True,
        "anchor_accepted": True,
        "parent_acceptance_changed": False,
        "parents_unchanged": True,
        "scope": iii.SCOPE,
        **iii.NO_CLAIMS,
    }
    summary = {
        **copy.deepcopy(b.summary),
        "ca_in": iii.TARGET_CA,
        "checks": checks,
        "cold_load_checks": checks,
        **iii.NO_CLAIMS,
    }
    write(runtime / "summary.json", summary)
    monkeypatch.setattr(
        iii.runner,
        "validate_restart",
        lambda path, *args, **kwargs: copy.deepcopy(packets[str(path.resolve())]),
    )
    return SimpleNamespace(
        runtime=runtime,
        parent=parent,
        case=b.target,
        restart=final,
        result=result,
        summary=summary,
        cold=cold,
        packets=packets,
    )


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "initial_signature",
        "initial_pins",
        "warm_detached",
        "warm_preservation",
        "stage_manifest",
    ],
)
def test_complete_terminal_admission_binds_original_parent_and_sample_proofs(
    terminal_bundle, fault
):
    b = terminal_bundle
    initial_path = b.runtime / "parent-observation/samples/cold-load.json"
    warm_path = b.runtime / "terminal/samples/after-polish.json"
    if fault in {"initial_signature", "initial_pins"}:
        initial = iii.runner._read_json(initial_path)
        if fault == "initial_signature":
            initial["signature"]["arrays"]["all"] = "different parent fields"
        else:
            initial["actual_pins"]["sha256"] = "different parent pins"
        write(initial_path, initial)
    elif fault == "warm_detached":
        warm = iii.runner._read_json(warm_path)
        warm["signature"]["arrays"]["all"] = "detached warm fields"
        write(warm_path, warm)
    elif fault == "warm_preservation":
        b.result["terminal"]["after_polish"]["sampling_preserved_state"] = False
        write(warm_path, b.result["terminal"]["after_polish"])
    elif fault == "stage_manifest":
        key = str((b.runtime / "stages/ca_001/packet.json").resolve())
        b.packets[key]["parent_manifest"] = iii.regular_record(b.runtime / "run-manifest.json")
    write(b.runtime / "result.json", b.result)
    completion = {
        "schema": iii.SCHEMA,
        "status": "complete",
        "numerical_acceptance": True,
        "anchor_accepted": True,
        "parent_acceptance_changed": False,
        "scope": iii.SCOPE,
        "files": iii.inventory(b.runtime),
    }
    write(b.runtime / "completion.json", completion)
    args = (
        b.runtime,
        b.parent,
        b.restart,
        b.case,
        b.summary,
        b.cold,
        completion["files"],
        completion,
    )
    if fault is None:
        records = iii.admit_terminal(*args)
        assert records and all(len(record["sha256"]) == 64 for record in records)
    else:
        with pytest.raises(ValueError):
            iii.admit_terminal(*args)


def test_new_consumer_requires_exact_producer_type(parent_bundle, monkeypatch):
    b = parent_bundle
    parent = {
        "component_commit": b.component["commit"],
        "component": b.component,
        "runner": {
            "path": "/producer/" + iii.ENTRYPOINT,
            "sha256": iii.sha(iii.ROOT / iii.ENTRYPOINT),
        },
        "runner_path": "/producer/" + iii.ENTRYPOINT,
        "runner_sha256": iii.sha(iii.ROOT / iii.ENTRYPOINT),
        "anchor_stage": {"kind": iii.KIND},
        "qualification_schema": iii.SCHEMA,
        "implementation": {
            name: {"path": "/producer/" + name, "sha256": iii.sha(iii.ROOT / name)}
            for name in (*modes.HELPERS, *iii.NEW_HELPERS, iii.ENTRYPOINT)
        },
    }
    monkeypatch.setattr(
        modes.runner,
        "_git",
        lambda root, command, item: (iii.ROOT / item.split(":", 1)[1]).read_bytes(),
    )
    result = modes.source_contract(parent, b.component, typed_iii=True)
    assert iii.ENTRYPOINT in result
    with pytest.raises(ValueError, match="must be paired"):
        modes.source_contract(parent, b.component)
    parent["anchor_stage"]["kind"] = "stationary_III_anchor"
    with pytest.raises(ValueError, match="kind/schema"):
        modes.source_contract(parent, b.component, typed_iii=True)
