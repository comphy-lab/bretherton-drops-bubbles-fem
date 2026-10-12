"""Software-only ascending continuation, provenance and qualification controls."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_paper_seed_requalify as requal_fixtures

from runners import run_paper_anchor_i as lane

cold_bundle = requal_fixtures.parent_bundle


@pytest.fixture
def cases():
    parent = lane.model.read_case(lane.ROOT / lane.seed.BASE_CASE)
    parent["case_id"] = "software_fresh_negative_seed"
    parent["physics"]["ca_in"] = lane.PARENT_CA
    target = copy.deepcopy(parent)
    target["case_id"] = "software_stationary_I_candidate"
    target["physics"]["ca_in"] = lane.TARGET_CA
    return parent, target


@pytest.mark.parametrize(
    "values",
    [
        [],
        [-0.02, -0.0016],
        [-0.01, -0.01, -0.0016],
        [-0.01, -0.015, -0.0016],
        [-0.021, -0.0016],
        [-0.01, -0.0015],
        [-0.01, True, -0.0016],
        [float("nan"), -0.0016],
        [-0.01, 0.0, -0.0016],
        [-0.019, -0.018, -0.017, -0.016, -0.015, -0.014, -0.013, -0.012, -0.0016],
    ],
)
def test_bad_ascending_ladder_is_refused(cases, tmp_path, values):
    parent, target = cases
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"schema": "herrada-paper-ca-ladder-v1", "values": values}))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        lane.read_ladder(path, parent, target)
    assert path.read_bytes() == before


def test_declared_ladder_and_each_truthful_stage(cases, tmp_path):
    parent, target = cases
    values = [-0.01, -0.005, -0.003, -0.002, -0.0016]
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"schema": "herrada-paper-ca-ladder-v1", "values": values}))
    assert lane.read_ladder(path, parent, target) == values
    previous = parent
    for index, value in enumerate(values, 1):
        current = lane.anchor.stage_case(target, value, index, index == len(values))
        assert current["physics"]["ca_in"] == value
        assert set(lane.remesh.differing_keys(previous, current)) == {"case_id", "physics.ca_in"}
        previous = current
    assert previous == target
    assert parent["physics"]["ca_in"] == -0.02


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("physics", "bond", 1.57),
        ("mesh", "film_elements", 8.0),
        ("mesh", "bulk_resolution", 0.04),
        ("tolerances", "pressure_boundary_absolute", 1e-5),
        ("solve", "max_newton_iterations", 41),
    ],
)
def test_other_case_fields_cannot_change(cases, section, key, value):
    parent, target = cases
    target[section][key] = value
    with pytest.raises(ValueError, match="only Ca and case_id"):
        lane.transition_contract(parent, target)


def test_positive_seed_and_other_endpoint_are_refused(cases):
    parent, target = cases
    parent["physics"]["ca_in"] = 0.02
    with pytest.raises(ValueError, match="declared negative-flow"):
        lane.transition_contract(parent, target)
    parent["physics"]["ca_in"] = -0.02
    target["physics"]["ca_in"] = -0.00315
    with pytest.raises(ValueError, match="declared negative-flow"):
        lane.transition_contract(parent, target)


def signature(ca=-0.02):
    payload = hashlib.sha256(repr(ca).encode()).hexdigest()
    return {
        "bc_cache_marker": "init",
        "parameters": {"Bo": 1.56, "Ca_in": ca},
        "layout": {"ndof": 6},
        "arrays": {"free_dofs": {"sha256": "software-fields"}},
        "volume": 1.0,
        "frame_speed": -0.02,
        "time": 8.0,
        "serialized_state": {"sha256": payload, "payload_sha256": payload},
    }


def bind_completion(runtime, files):
    (runtime / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema": lane.requalification.SCHEMA,
                "numerical_acceptance": True,
                "files": {name: lane.sha(runtime / name) for name in files},
            }
        )
    )


def producer_manifest():
    digest = hashlib.sha256(
        lane.diagnostic.git(lane.ROOT, "show", f"{lane.PARENT_SOURCE}:{lane.PRODUCER}")
    ).hexdigest()
    record = {"path": str(lane.ROOT / lane.PRODUCER), "sha256": digest}
    return {
        "component_commit": lane.PARENT_SOURCE,
        "component": {"commit": lane.PARENT_SOURCE},
        "runner": record,
        "runner_path": record["path"],
        "runner_sha256": digest,
        "implementation": {lane.PRODUCER: record},
        "intervention": {
            "kind": "stationary-seed-requalification",
            "solve_calls": 0,
            "remesh_calls": 0,
            "time_evolution": False,
            "api": lane.requalification.cold.API,
            "parent_acceptance_changed": False,
        },
        "lineage": {
            "schema": lane.requalification.SCHEMA,
            "parent_status": "failed",
            "parent_acceptance_changed": False,
            "consumer": {
                "module": lane.PRODUCER,
                "callable": "restore_checked",
                "sha256": digest,
                "saved_endtime": 8.0,
                "api": lane.requalification.cold.API,
            },
        },
        "azimuthal_stability": True,
        "qualification_scope": "stationary-only software fixture",
    }


@pytest.fixture
def parent_bundle(cases, tmp_path, monkeypatch):
    parent, target = cases
    runtime = tmp_path / "parent"
    runtime.mkdir()
    checks = dict.fromkeys(lane.GATES, True)
    history = {"time": 8.0, "dts": [0.1, 0.1], "unsteady_steps": 12, "suggested_next_dt": 0.1}
    cold = {
        "verified": True,
        "loaded": signature(),
        "actual_pins": {"sha256": "software-pins"},
        "checks": checks,
        "bc_preparation": {"verified": True},
        "time_history": history,
        "intervention": {
            "verified": True,
            "kind": "typed-native-endtime-restoration",
            "saved_endtime": 8.0,
            "api": lane.requalification.cold.API,
        },
    }
    manifest = producer_manifest()
    (runtime / "run-manifest.json").write_text(json.dumps(manifest))
    (runtime / "resume.json").write_text("{}")
    (runtime / "state.dump").write_bytes(b"software-only complete state")
    restart = {
        "stage": {"kind": "completed_polish", "bond": 1.56, "bc_mode": "init"},
        "roundtrip": {"after": signature(), "layout": signature()["layout"]},
        "packet": {
            "path": str(runtime / "resume.json"),
            "sha256": lane.sha(runtime / "resume.json"),
        },
        "parent_manifest": {
            "path": str(runtime / "run-manifest.json"),
            "sha256": lane.sha(runtime / "run-manifest.json"),
        },
        "state": {"path": str(runtime / "state.dump"), "sha256": lane.sha(runtime / "state.dump")},
        "mesh_reference_base": str(runtime),
        "meshes": [],
    }
    original_state = tmp_path / "original.dump"
    original_state.write_bytes((runtime / "state.dump").read_bytes())
    original = {
        "restart": {**copy.deepcopy(restart), "state": {"path": str(original_state)}},
        "sample": {"actual_pins": copy.deepcopy(cold["actual_pins"])},
        "metadata": {**history, "endtime": 8.0},
        "records": [{"path": str(original_state), "sha256": lane.sha(original_state)}],
    }
    # Admission orchestration uses synthetic bytes; native parsing is covered by helper tests.
    monkeypatch.setattr(lane, "validate_parent_lineage", lambda value: original)
    monkeypatch.setattr(lane.requalification.cold, "normalized_bytes", lambda data, base: data)
    result = {
        "schema": lane.requalification.SCHEMA,
        "numerical_acceptance": True,
        "checks": checks,
        "cold_load_checks": checks,
        "cold_load_verified": True,
        "checkpoint": {"resume_packet": "resume.json"},
        "lineage": manifest["lineage"],
        "parent_status": "failed",
        "parent_acceptance_changed": False,
        "branch_identity_verified": False,
        "eigenanalysis_performed": False,
        "jacobian_certificate": False,
        "dynamic_restart_certificate": False,
        "full_comparison_verified": False,
    }
    files = {
        "run-manifest.json": manifest,
        "summary.json": {
            **copy.deepcopy(result),
            "status": "completed",
            "completion_reached": True,
        },
        "cold-load.json": {
            **cold,
            "packet": restart["packet"],
            "parent_manifest": restart["parent_manifest"],
        },
        "parent-cold-load.json": copy.deepcopy(cold),
        "result.json": result,
        "schema.json": {
            "schema": lane.requalification.SCHEMA,
            "scope": manifest["qualification_scope"],
            "consumer": manifest["lineage"]["consumer"],
        },
    }
    for name, value in files.items():
        (runtime / name).write_text(json.dumps(value))
    bound = {*files, "resume.json", "state.dump"}
    bind_completion(runtime, bound)
    return SimpleNamespace(
        runtime=runtime,
        parent=parent,
        target=target,
        cold=files["cold-load.json"],
        files=files,
        bound=bound,
        restart=restart,
        original=original,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_completion",
        "unbound_cold",
        "unbound_state",
        "failed_gate",
        "cold_false",
        "changed_bytes",
        "history",
        "wrong_forcing",
        "bc_false",
        "typed_false",
        "pins",
        "parent_history",
        "dt_history",
        "schema_consumer",
        "packet_lineage",
        "checkpoint",
        "result_gate",
        "lineage",
        "failure_receipt",
        "expanded_claim",
        "portable_history",
        "not_accepted",
        "old_schema",
        "unknown_schema",
    ],
)
def test_failed_or_unbound_parent_is_refused(parent_bundle, fault):
    bundle = parent_bundle
    if fault == "missing_completion":
        (bundle.runtime / "completion.json").unlink()
    elif fault == "unbound_cold":
        bind_completion(bundle.runtime, bundle.bound - {"cold-load.json"})
    elif fault == "unbound_state":
        bind_completion(bundle.runtime, bundle.bound - {"state.dump"})
    elif fault == "changed_bytes":
        (bundle.runtime / "summary.json").write_text("{}")
    elif fault == "not_accepted":
        path = bundle.runtime / "completion.json"
        value = json.loads(path.read_text())
        value["numerical_acceptance"] = False
        path.write_text(json.dumps(value))
    else:
        if fault == "failed_gate":
            bundle.files["summary.json"]["checks"] = {
                **dict.fromkeys(lane.GATES, True),
                "lower_static_pressure": False,
            }
        elif fault == "cold_false":
            bundle.files["cold-load.json"]["verified"] = False
        elif fault == "history":
            bundle.files["cold-load.json"]["loaded"]["serialized_state"]["payload_sha256"] = (
                "changed"
            )
        elif fault == "wrong_forcing":
            bundle.files["cold-load.json"]["loaded"]["parameters"]["Ca_in"] = -0.01
        elif fault == "bc_false":
            bundle.files["parent-cold-load.json"]["bc_preparation"]["verified"] = False
        elif fault == "typed_false":
            bundle.files["cold-load.json"]["intervention"]["api"] = {}
        elif fault == "pins":
            bundle.files["cold-load.json"]["actual_pins"] = {"sha256": "changed"}
        elif fault == "parent_history":
            bundle.files["parent-cold-load.json"]["loaded"]["serialized_state"][
                "payload_sha256"
            ] = "changed"
        elif fault == "dt_history":
            bundle.files["cold-load.json"]["time_history"]["dts"] = [0.2]
        elif fault == "schema_consumer":
            bundle.files["schema.json"]["consumer"] = {"callable": "untyped_load"}
        elif fault == "packet_lineage":
            bundle.files["cold-load.json"]["packet"] = {"path": "another"}
        elif fault == "checkpoint":
            bundle.files["result.json"]["checkpoint"] = {"resume_packet": "state.dump"}
        elif fault == "result_gate":
            bundle.files["result.json"]["checks"] = dict.fromkeys(lane.GATES, False)
        elif fault == "lineage":
            bundle.files["result.json"]["lineage"] = {"parent_status": "complete"}
        elif fault == "failure_receipt":
            (bundle.runtime / "failure.json").write_text("{}")
        elif fault == "expanded_claim":
            bundle.files["result.json"]["dynamic_restart_certificate"] = True
        elif fault == "portable_history":
            (bundle.runtime / "state.dump").write_bytes(b"different complete history")
            bundle.restart["state"]["sha256"] = lane.sha(bundle.runtime / "state.dump")
        elif fault == "old_schema":
            bundle.files["result.json"]["schema"] = "private-fresh-negative-paper-seed-v1"
        elif fault == "unknown_schema":
            bundle.files["result.json"]["schema"] = "unreviewed-seed-result-v1"
        for name, value in bundle.files.items():
            (bundle.runtime / name).write_text(json.dumps(value))
        bind_completion(bundle.runtime, bundle.bound)
    before = {str(p): p.read_bytes() for p in bundle.runtime.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        lane.qualified_parent(bundle.runtime / "run-manifest.json", bundle.parent, bundle.restart)
    assert {str(p): p.read_bytes() for p in bundle.runtime.rglob("*") if p.is_file()} == before


def test_qualified_parent_records_every_bound_byte(parent_bundle):
    bundle = parent_bundle
    cold, records = lane.qualified_parent(
        bundle.runtime / "run-manifest.json", bundle.parent, bundle.restart
    )
    assert cold == bundle.cold
    assert len(records) == len(bundle.bound) + 1 + len(bundle.original["records"])
    assert lane.diagnostic.parents_unchanged(records) is True


def test_recovery_predictions_are_not_additional_target_acceptance_gates(parent_bundle):
    bundle = parent_bundle
    result = bundle.files["result.json"]
    result["initial_polish_predictions"] = {"checks": {"software_diagnostic": False}}
    result["predictions_define_numerical_acceptance"] = False
    bundle.files["summary.json"].update(result)
    (bundle.runtime / "summary.json").write_text(json.dumps(bundle.files["summary.json"]))
    (bundle.runtime / "result.json").write_text(json.dumps(result))
    bind_completion(bundle.runtime, bundle.bound)
    cold, _ = lane.qualified_parent(
        bundle.runtime / "run-manifest.json", bundle.parent, bundle.restart
    )
    assert cold["checks"] == dict.fromkeys(lane.GATES, True)


@pytest.mark.parametrize("fault", [None, "core", "interpreter", "dirty", "producer"])
def test_preflight_binds_parent_case_clean_source_and_exact_environment(
    parent_bundle,
    tmp_path,
    monkeypatch,
    fault,
):
    bundle = parent_bundle
    parent_case, target_case, ladder = [
        tmp_path / name
        for name in (
            "parent-case.json",
            "target-case.json",
            "ladder.json",
        )
    ]
    parent_case.write_text(json.dumps(bundle.parent))
    target_case.write_text(json.dumps(bundle.target))
    ladder.write_text(
        json.dumps({"schema": "herrada-paper-ca-ladder-v1", "values": [-0.01, -0.0016]})
    )
    component = lane.runner.component_revision(lane.ROOT)
    component.update(
        commit=lane.PARENT_SOURCE,
        head=lane.PARENT_SOURCE,
        clean=True,
        status=[],
        declared_commit=None,
    )
    current = {**component, "commit": "f" * 40, "head": "f" * 40}
    if fault == "dirty":
        current["clean"] = False
    environment = {
        "available": True,
        "commit": lane.diagnostic.PYOOMPH,
        "url": "https://software.invalid",
        "direct_url": {"vcs_info": {"vcs": "git", "commit_id": lane.diagnostic.PYOOMPH}},
        "native_core": {"path": "/software-only/core.so", "sha256": lane.diagnostic.CORE},
        "interpreter": {"resolved_executable": "/software-only/python"},
    }
    parent_manifest = {
        **producer_manifest(),
        "component": component,
        "pyoomph": copy.deepcopy(environment),
        "case_path": str(parent_case),
        "case_sha256": lane.sha(parent_case),
    }
    if fault == "producer":
        parent_manifest["runner_sha256"] = "wrong"
    if fault == "core":
        environment["native_core"]["sha256"] = "wrong"
    if fault == "interpreter":
        environment["interpreter"]["resolved_executable"] = "/different/python"
    path = bundle.runtime / "run-manifest.json"
    bundle.files["run-manifest.json"] = parent_manifest
    path.write_text(json.dumps(parent_manifest))
    bundle.files["cold-load.json"]["parent_manifest"] = {
        "path": str(path),
        "sha256": lane.sha(path),
    }
    (bundle.runtime / "cold-load.json").write_text(json.dumps(bundle.files["cold-load.json"]))
    bind_completion(bundle.runtime, bundle.bound)
    state, packet = bundle.runtime / "state.dump", bundle.runtime / "resume.json"
    state.write_bytes(b"software-only complete state")
    packet.write_text("{}")
    restart = {
        **bundle.restart,
        "stage": {"kind": "completed_polish", "bond": 1.56, "bc_mode": "init"},
        "parent_manifest": {"path": str(path), "sha256": lane.sha(path)},
        "packet": {"path": str(packet), "sha256": lane.sha(packet)},
        "state": {"path": str(state), "sha256": lane.sha(state)},
        "meshes": [],
    }
    git = lane.diagnostic.git

    def read_git(root, *args):
        if args[:2] == ("merge-base", "--is-ancestor"):
            assert args[2:] == (lane.PARENT_SOURCE, current["commit"])
            return b""
        if args[0] == "show" and args[1].startswith(current["commit"] + ":"):
            return (lane.ROOT / args[1].partition(":")[2]).read_bytes()
        return git(root, *args)

    def validate(path, case, digest, **kwargs):
        assert path == packet and case == bundle.parent and digest == lane.sha(parent_case)
        assert kwargs == {
            "component": current,
            "environment": environment,
            "azimuthal_stability": True,
        }
        return restart

    monkeypatch.setattr(lane, "PARENT_CASE_SHA", lane.sha(parent_case))
    monkeypatch.setattr(lane.runner, "component_revision", lambda root: current)
    monkeypatch.setattr(lane.runner, "pyoomph_revision", lambda: environment)
    monkeypatch.setattr(lane.runner, "validate_restart", validate)
    monkeypatch.setattr(lane.diagnostic, "git", read_git)
    args = SimpleNamespace(
        source=lane.ROOT,
        case=target_case,
        manifest=path,
        packet=packet,
        state=state,
        ca_ladder=ladder,
        out=tmp_path / "unused-output",
    )
    if fault:
        with pytest.raises(ValueError):
            lane.preflight(args)
    else:
        prepared = lane.preflight(args)
        assert prepared["parent"] == bundle.parent and prepared["target"] == bundle.target
        assert prepared["contract"]["ca_ladder"]["direction"] == "ascending"
        assert prepared["contract"]["total_newton_update_bound"] == 480
        assert prepared["contract"]["per_polish_remesh_bound"] == 3
        assert prepared["implementation"][lane.ENTRYPOINT]["sha256"] == lane.sha(
            lane.ROOT / lane.ENTRYPOINT
        )
        assert lane.diagnostic.parents_unchanged(prepared["records"]) is True
    assert not args.out.exists()


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "source",
        "old_source",
        "producer",
        "old_producer",
        "hash",
        "solve",
        "remesh",
        "api",
        "lineage",
        "consumer",
        "endtime",
    ],
)
def test_only_the_hash_bound_typed_requalification_producer_is_allowed(fault):
    value = producer_manifest()
    if fault == "source":
        value["component_commit"] = "0" * 40
    elif fault == "old_source":
        value["component_commit"] = value["component"]["commit"] = (
            lane.requalification.cold.PARENT_SOURCE
        )
    elif fault == "producer":
        value["runner"]["path"] = str(lane.ROOT / "runners/run_paper_anchor.py")
    elif fault == "old_producer":
        value["runner"]["path"] = value["runner_path"] = str(
            lane.ROOT / "runners/run_paper_seed_recovery.py"
        )
    elif fault == "hash":
        value["runner_sha256"] = "0" * 64
    elif fault == "solve":
        value["intervention"]["solve_calls"] = 1
    elif fault == "remesh":
        value["intervention"]["remesh_calls"] = 1
    elif fault == "api":
        value["intervention"]["api"] = {}
    elif fault == "lineage":
        value["lineage"]["parent_status"] = "complete"
    elif fault == "consumer":
        value["lineage"]["consumer"]["callable"] = "load_state"
    elif fault == "endtime":
        value["lineage"]["consumer"]["saved_endtime"] = 9.0
    if fault:
        with pytest.raises(ValueError):
            lane.check_parent_producer(value)
    else:
        lane.check_parent_producer(value)


@pytest.mark.parametrize("changed_lineage", [False, True])
def test_lineage_preflight_uses_original_direct_pointers(tmp_path, monkeypatch, changed_lineage):
    lineage = producer_manifest()["lineage"]
    for key, name in (
        ("original_failed_producer", "failed/manifest.json"),
        ("original_packet", "failed/packet.json"),
        ("original_state", "failed/state.dump"),
    ):
        lineage[key] = {"path": str(tmp_path / name), "sha256": "direct-evidence"}
    lineage["completed_diagnostic"] = {
        "manifest": {"path": str(tmp_path / "diagnostic/run-manifest.json")}
    }
    manifest = {"lineage": lineage, "case_path": str(tmp_path / "original-case.json")}
    expected = copy.deepcopy(lineage)
    if changed_lineage:
        expected["original_state"]["sha256"] = "different"

    def preflight(args):
        assert vars(args) == {
            "source": lane.ROOT,
            "case": tmp_path / "original-case.json",
            "manifest": tmp_path / "failed/manifest.json",
            "packet": tmp_path / "failed/packet.json",
            "state": tmp_path / "failed/state.dump",
            "diagnostic_runtime": tmp_path / "diagnostic",
        }
        return {"lineage": expected, "records": [{"path": "direct-record"}]}

    monkeypatch.setattr(lane.requalification, "preflight", preflight)
    if changed_lineage:
        with pytest.raises(ValueError, match="lineage differs"):
            lane.validate_parent_lineage(manifest)
    else:
        assert lane.validate_parent_lineage(manifest)["records"] == [{"path": "direct-record"}]


def test_generated_typed_descendant_passes_actual_consumer_check(cold_bundle, monkeypatch):
    # Reuse the fake native writer to generate real producer receipt schemas.
    # The complete admission path parses synthetic portable bytes, never pyoomph.
    bundle = requal_fixtures.lane.__wrapped__(cold_bundle, monkeypatch)
    bundle.current.update(commit=lane.PARENT_SOURCE, head=lane.PARENT_SOURCE)
    assert lane.requalification.main(requal_fixtures.arguments(bundle)) == 0
    runtime = bundle.args.out
    result = json.loads((runtime / "result.json").read_text())
    packet = runtime / result["checkpoint"]["resume_packet"]
    restart = lane.runner.validate_restart(
        packet,
        lane.runner.read_case(bundle.args.case),
        lane.sha(bundle.args.case),
        component=bundle.current,
        environment=lane.runner.pyoomph_revision(),
        azimuthal_stability=True,
    )
    target = copy.deepcopy(lane.runner.read_case(bundle.args.case))
    target["case_id"] = "software-I-target"
    target["physics"]["ca_in"] = lane.TARGET_CA
    case_path = runtime.parent / "target.json"
    lane.write_json(case_path, target)
    ladder = runtime.parent / "ladder.json"
    lane.write_json(
        ladder,
        {
            "schema": "herrada-paper-ca-ladder-v1",
            "values": [-0.01, -0.005, -0.003, -0.002, -0.0016],
        },
    )
    monkeypatch.setattr(lane, "PARENT_CASE_SHA", lane.sha(bundle.args.case))
    monkeypatch.setattr(lane, "execute", lambda *args: pytest.fail("native consumer execution"))
    output = runtime.parent / "unused-consumer-output"
    builds = len(bundle.problems)
    args = [
        item
        for key, value in {
            "source": lane.ROOT,
            "case": case_path,
            "manifest": runtime / "run-manifest.json",
            "packet": packet,
            "state": Path(restart["state"]["path"]),
            "ca-ladder": ladder,
            "out": output,
        }.items()
        for item in ("--" + key, str(value))
    ]
    before = {str(path): lane.sha(path) for path in runtime.parent.rglob("*") if path.is_file()}
    assert lane.main([*args, "--check"]) == 0
    assert not output.exists() and len(bundle.problems) == builds
    assert "pyoomph" not in sys.modules
    assert before == {
        str(path): lane.sha(path) for path in runtime.parent.rglob("*") if path.is_file()
    }


def prepared_fixture(bundle):
    return {
        "parent": bundle.parent,
        "target": bundle.target,
        "values": [-0.01, -0.0016],
        "parent_cold": bundle.cold,
        "restart": {
            **bundle.restart,
            "parent_manifest": {
                "path": str(bundle.runtime / "run-manifest.json"),
            },
        },
        "environment": {},
        "component": {"commit": "software-only"},
        "implementation": {},
        "records": [
            {
                "path": str(bundle.runtime / "summary.json"),
                "sha256": lane.sha(bundle.runtime / "summary.json"),
            }
        ],
        "contract": {"parent_case": {"path": "/software-only/parent-case.json"}},
    }


def test_every_manifest_binds_the_new_entrypoint(parent_bundle, tmp_path):
    prepared = prepared_fixture(parent_bundle)
    case_path = tmp_path / "target.json"
    case_path.write_text(json.dumps(prepared["target"]))
    for stage in (
        {"kind": "stationary_I_candidate"},
        {"kind": "ascending_ca_continuation"},
        {"kind": "terminal_remesh"},
    ):
        manifest = lane.manifest(case_path, prepared["target"], prepared, stage)
        assert manifest["runner_path"] == str(lane.ROOT / lane.ENTRYPOINT)
        assert manifest["runner"] == {
            "path": manifest["runner_path"],
            "sha256": lane.sha(lane.ROOT / lane.ENTRYPOINT),
        }
        assert manifest["runner_sha256"] == manifest["runner"]["sha256"]
        assert manifest["case_sha256"] == lane.sha(case_path)
        assert manifest["anchor_stage"] == stage
        assert "Ca=-0.0016" in manifest["qualification_scope"]
        assert lane.anchor.TARGET_CA == -0.00315


def arguments(bundle, case, output, check=False):
    values = [
        "--source",
        str(lane.ROOT),
        "--case",
        str(case),
        "--manifest",
        str(bundle.runtime / "run-manifest.json"),
        "--packet",
        "/software-only/resume.json",
        "--state",
        "/software-only/state.dump",
        "--ca-ladder",
        "/software-only/ladder.json",
        "--out",
        str(output),
    ]
    return values + (["--check"] if check else [])


def test_check_creates_no_output_and_never_executes(parent_bundle, tmp_path, monkeypatch):
    prepared = prepared_fixture(parent_bundle)
    monkeypatch.setattr(lane, "preflight", lambda args: prepared)
    monkeypatch.setattr(lane, "execute", lambda *args: pytest.fail("native execution forbidden"))
    output = tmp_path / "check-output"
    assert lane.main(arguments(parent_bundle, tmp_path / "case.json", output, True)) == 0
    assert not output.exists()


def test_import_uses_no_native_solver():
    probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; from runners import run_paper_anchor_i; "
                "assert not any(n == 'pyoomph' or n.startswith('pyoomph.') "
                "or n.startswith('_pyoomph_core') for n in sys.modules)"
            ),
        ],
        cwd=lane.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_failure_retains_manifest_and_parent_bytes(parent_bundle, tmp_path, monkeypatch):
    prepared = prepared_fixture(parent_bundle)
    case_path = tmp_path / "target.json"
    case_path.write_text(json.dumps(prepared["target"]))
    monkeypatch.setattr(lane, "preflight", lambda args: prepared)

    def fail(*args):
        raise RuntimeError("software stationary budget exhausted")

    monkeypatch.setattr(lane, "execute", fail)
    output = tmp_path / "failed-output"
    before = (parent_bundle.runtime / "summary.json").read_bytes()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        lane.main(arguments(parent_bundle, case_path, output))
    failure = json.loads((output / "failure.json").read_text())
    assert failure["parents_unchanged"] is True and failure["numerical_acceptance"] is False
    assert (parent_bundle.runtime / "summary.json").read_bytes() == before
    assert (output / "run-manifest.json").exists() and not (output / "completion.json").exists()


@pytest.mark.parametrize("fault", [None, "cold_pressure", "parent_restore", "target_restore"])
def test_ascending_orchestration_uses_one_terminal_remesh_and_original_gates(
    parent_bundle,
    tmp_path,
    monkeypatch,
    fault,
):
    prepared = prepared_fixture(parent_bundle)
    prepared["values"] = [-0.01, -0.005, -0.003, -0.002, -0.0016]
    case_path = tmp_path / "target.json"
    case_path.write_text(json.dumps(prepared["target"]))
    output = tmp_path / "output"
    output.mkdir()
    for name, payload in (
        (
            "run-manifest.json",
            lane.manifest(
                case_path, prepared["target"], prepared, {"kind": "stationary_I_candidate"}
            ),
        ),
    ):
        lane.write_json(output / name, payload)
    lane.runner.append_progress(output, {"event": "software-prepared"}, first=True)
    calls, packets, builds = [], {}, []
    pins = {"sha256": "software-pins"}

    class Problem:
        def __init__(self, case):
            self.case = copy.deepcopy(case)
            self.parameters = {
                "Bo": SimpleNamespace(value=1.56),
                "Ca_in": SimpleNamespace(value=case["physics"]["ca_in"]),
            }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_global_parameter(self, name):
            return self.parameters[name]

        def force_remesh(self):
            calls.append("force-remesh")
            return True

    def build(case, **kwargs):
        builds.append(case["physics"]["ca_in"])
        return Problem(case)

    def sample(problem, model, case, directory, phase, np):
        provisional = directory.name.startswith("ca_")
        failure = provisional or (fault == "cold_pressure" and directory.name == "target-cold")
        metrics = {
            "stationary_residual_linf": 1e-9,
            "lower_pressure_rms": 5e-6 if failure else 1e-9,
            "frame_speed": -0.001,
            "gas_pressure_mean": 1.0,
            "lower_max_divergence": 1e-10,
            "lower_flux": case["physics"]["ca_in"],
            "upper_flux": case["physics"]["ca_in"],
            "gas_volume": 1.0,
            "surface_volume": 1.0,
            "target_volume": 1.0,
        }
        return {
            "metrics": metrics,
            "actual_pins": pins,
            "port_adjacent_geometry": {"max_midside_departure_l2": 0.0},
        }

    def checkpoint(problem, case, directory, index, azimuthal):
        path = directory / "software-resume.json"
        lane.write_json(path, {"software_only": True})
        packets[path] = {
            "packet": {"sha256": lane.sha(path)},
            "roundtrip": {"after": signature(case["physics"]["ca_in"])},
        }
        return {"resume_packet": path.name}

    def validate(path, case, digest, **kwargs):
        manifest = json.loads((path.parent / "run-manifest.json").read_text())
        assert manifest["runner_path"] == str(lane.ROOT / lane.ENTRYPOINT)
        assert manifest["case_sha256"] == digest
        assert manifest["case_id"] == case["case_id"]
        return packets[path]

    def polish(*args, **kwargs):
        assert kwargs == {"initial_relaxation": 1.0}
        calls.append("polish")
        return {"status": "converged"}

    monkeypatch.setattr(lane.model, "build_problem", build)
    monkeypatch.setattr(lane.anchor, "configure", lambda *args: None)
    monkeypatch.setattr(lane.anchor, "stable_inputs", lambda *args: None)

    def restore(problem, case, restart, expected_pins, expected_history):
        expected_directory = output if case == prepared["parent"] else output / "target-cold"
        assert (
            problem._seed_requalification_receipts_directory == expected_directory / "typed-loads"
        )
        assert "solver" not in problem._seed_requalification_receipts_directory.parts
        assert expected_pins == pins
        assert (
            expected_history
            == signature(case["physics"]["ca_in"])["serialized_state"]["payload_sha256"]
        )
        calls.append("typed-restore")
        if (fault == "parent_restore" and case == prepared["parent"]) or (
            fault == "target_restore" and case == prepared["target"]
        ):
            raise ValueError("typed full-history restore refused")
        return {"verified": True}

    monkeypatch.setattr(
        lane.anchor, "restore_checked", lambda *args: pytest.fail("untyped restore forbidden")
    )
    monkeypatch.setattr(lane.requalification, "restore_checked", restore)
    monkeypatch.setattr(
        lane.model,
        "_stationary_state_signature",
        lambda problem, *args: signature(problem.parameters["Ca_in"].value),
    )
    monkeypatch.setattr(lane.diagnostic, "actual_pins", lambda problem: pins)
    monkeypatch.setattr(lane.model, "polish_confined_state", polish)
    monkeypatch.setattr(lane.model, "_checkpoint_stationary_state", checkpoint)
    monkeypatch.setattr(lane.runner, "validate_restart", validate)
    monkeypatch.setattr(lane.remesh, "sample", sample)
    monkeypatch.setattr(
        lane.remesh, "retain_portable_sample", lambda *args: {"software_only": True}
    )
    monkeypatch.setattr(
        lane.remesh, "normalize_pressure_dummies", lambda *args: calls.append("dummy-history-audit")
    )
    args = SimpleNamespace(case=case_path)
    if fault in ("parent_restore", "target_restore"):
        with pytest.raises(ValueError, match="typed full-history restore refused"):
            lane.execute(args, prepared, output)
        assert not (output / "result.json").exists()
        assert not (output / "summary.json").exists()
        assert not (output / "cold-load.json").exists()
        if fault == "parent_restore":
            assert calls == ["typed-restore"]
            assert builds == [-0.02]
            assert not (output / "stages").exists()
            return
    elif fault == "cold_pressure":
        with pytest.raises(ValueError, match="original scientific or cold-load"):
            lane.execute(args, prepared, output)
    else:
        result = lane.execute(args, prepared, output)
        assert result["numerical_acceptance"] is True
        assert result["branch_identity_verified"] is result["eigenanalysis_performed"] is False
        assert len(result["stages"]) == 5
        assert all(stage["numerical_acceptance"] is False for stage in result["stages"])
        assert all(
            not stage["canonical_checks_observed"]["lower_static_pressure"]
            for stage in result["stages"]
        )
    assert calls.count("force-remesh") == calls.count("dummy-history-audit") == 1
    assert calls.count("polish") == 6
    assert calls.count("typed-restore") == 2
    assert builds == [-0.02, -0.0016]
    if fault != "target_restore":
        assert json.loads((output / "summary.json").read_text())["completion_reached"] is (
            fault is None
        )
