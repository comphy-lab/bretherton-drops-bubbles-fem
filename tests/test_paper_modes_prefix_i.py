"""Synthetic terminal-prefix admission controls; no native evidence or execution."""

from __future__ import annotations

import copy
import importlib.abc
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import test_paper_modes as original

from runners import run_paper_modes as modes

case = original.case
typed_i_bundle = original.typed_i_bundle


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "pyoomph" or fullname.startswith("_pyoomph"):
            raise AssertionError("native imports forbidden in prefix-I spectrum admission tests")


@pytest.fixture(autouse=True)
def deny_native():
    guard = NoNative()
    sys.meta_path.insert(0, guard)
    yield
    sys.meta_path.remove(guard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def prefix_i_bundle(typed_i_bundle, monkeypatch):
    b = typed_i_bundle
    runtime, case = b.runtime, b.case
    target = runtime / "case.json"
    write(target, case)
    producer = "b0c9b609146e7b4881b0c7e1894216718b0be9d7"
    consumer = "7d3b3980c6ad22989620106bd76ac368cd931c44"
    sources = {
        name: {
            "sha256": modes.sha(modes.ROOT / name),
            "committed_sha256": modes.sha(modes.ROOT / name),
            "committed_matches": True,
        }
        for name in modes.runner.METHOD_FILES
    }
    component = {
        "commit": producer,
        "head": producer,
        "clean": True,
        "status": [],
        "sources": sources,
    }
    environment = {
        "available": True,
        "commit": modes.diagnostic.PYOOMPH,
        "url": "https://github.com/comphy-lab/pyoomph",
        "direct_url": {"vcs_info": {"vcs": "git", "commit_id": modes.diagnostic.PYOOMPH}},
        "native_core": {"path": "/software/core.so", "sha256": modes.diagnostic.CORE},
    }
    implementation = {
        name: {"path": "/producer/" + name, "sha256": modes.sha(modes.ROOT / name)}
        for name in (*modes.runner.METHOD_FILES, *modes.HELPERS, *modes.PREFIX_I_HELPERS)
    }
    b.parent.update(
        runner={
            "path": "/producer/runners/run_paper_anchor_i_prefix.py",
            "sha256": implementation[modes.anchor_i_prefix.ENTRYPOINT]["sha256"],
        },
        runner_path="/producer/runners/run_paper_anchor_i_prefix.py",
        runner_sha256=implementation[modes.anchor_i_prefix.ENTRYPOINT]["sha256"],
        anchor_stage={"kind": "stationary_I_candidate_from_prefix"},
        qualification_schema=modes.prefix_qualification.SCHEMA,
        original_acceptance_changed=False,
        component_commit=producer,
        component=component,
        pyoomph=environment,
        azimuthal_stability=True,
        case_fields_used=list(modes.runner.CASE_FIELDS_USED),
        implementation=implementation,
    )
    b.parent.pop("parent_acceptance_changed")
    seed = copy.deepcopy(b.parent["parent_restart"])
    seed["compatibility"].update(
        parent_commit=producer,
        current_commit=producer,
        physics_ast_sha256="software-identical-physics",
    )
    b.parent["parent_restart"] = seed
    parent_case = copy.deepcopy(case)
    parent_case["physics"]["ca_in"] = -0.003
    parent_case["case_id"] = "software-qualified-prefix-only"
    qsignature = copy.deepcopy(b.cold["loaded"])
    qsignature["parameters"]["Ca_in"] = -0.003
    seed["roundtrip"] = {"after": qsignature, "layout": qsignature["layout"]}
    seed["stage"] = {"kind": "completed_polish", "bc_mode": "stationary"}
    seed["state"] = copy.deepcopy(b.restart["state"])
    seed["meshes"] = copy.deepcopy(b.restart["meshes"])
    seed["mesh_reference_base"] = str(runtime)
    seed["state"]["format"] = "portable_file"
    seed["roundtrip"]["after"]["serialized_state"] = copy.deepcopy(
        b.cold["loaded"]["serialized_state"]
    )
    qualified_prefix = {
        "path": str(runtime / "software-prefix/run-manifest.json"),
        "sha256": "software-prefix-hash",
    }
    contract = {
        "parent_case": {"path": "/software/prefix-case.json", "sha256": "software-case"},
        "qualified_prefix": qualified_prefix,
        "values": [-0.002, -0.0016],
        "original_indices": [4, 5],
        "carried": {"updates": 2, "remeshes": 0},
        "total_newton_update_bound": 478,
    }
    lineage = {
        "schema": modes.prefix_qualification.SCHEMA,
        "original_attempt": {"accepted": False, "acceptance_changed": False},
        "selected_index": 3,
        "scope": "software-only lineage",
    }
    admitted = {
        "restart": copy.deepcopy(seed),
        "parent": parent_case,
        "target": copy.deepcopy(case),
        "parent_cold": {"actual_pins": copy.deepcopy(b.cold["actual_pins"])},
        "contract": copy.deepcopy(contract),
        "lineage": copy.deepcopy(lineage),
        "records": [],
        "implementation": {
            name: {**record, "path": str(modes.ROOT / name)}
            for name, record in implementation.items()
        },
        "component": {**component, "commit": consumer, "head": consumer},
        "environment": environment,
    }
    admitted["restart"]["compatibility"]["current_commit"] = consumer
    b.parent.update(
        case_transition_contract=contract,
        construction_case=contract["parent_case"],
        qualified_prefix=qualified_prefix,
        lineage=lineage,
    )
    b.restart["state"]["format"] = "portable_file"
    b.restart["stage"]["kind"] = "completed_polish"
    for mesh in b.restart["meshes"]:
        mesh["stored_path"] = Path(mesh["path"]).name

    def typed_proof(directory, receipt):
        directory.mkdir(parents=True)
        state = Path(b.restart["state"]["path"])
        raw = state.read_bytes()
        offset = modes.requalification.cold.serialized_metadata(raw)["endtime_offset"]
        receipt["intervention"].update(applied=True, only_changed_slot=[offset, offset + 8])
        for phase in ("before", "after", "after-bc"):
            path = directory / (phase + ".dump")
            path.write_bytes(
                raw[:offset] + struct.pack("<d", float("nan")) + raw[offset + 8 :]
                if phase == "before"
                else raw
            )
            retained_mesh = directory / (phase + "-meshes/000.msh")
            retained_mesh.parent.mkdir()
            retained_mesh.write_bytes((runtime / "mesh.msh").read_bytes())
            retention = {
                "state": {**modes.regular_record(path), "format": "memory_snapshot"},
                "mesh_reference_base": str(runtime),
                "meshes": [{**modes.regular_record(retained_mesh), "stored_path": "mesh.msh"}],
                "mesh_exchange": {"mesh.msh": str(retained_mesh)},
            }
            write(directory / (phase + "-retention.json"), retention)
            if phase != "after-bc":
                receipt["intervention"][phase] = modes.regular_record(path)
        receipt["bc_preparation"] = {
            "verified": True,
            "method": "actions_before_stationary_solve(force_reassign_eqs=False)",
            "before": copy.deepcopy(receipt["loaded"]),
            "after": copy.deepcopy(receipt["loaded"]),
        }
        return receipt

    b.cold = typed_proof(runtime / "target-cold/typed-loads/software", b.cold)
    parent_load = copy.deepcopy(b.cold)
    parent_load["loaded"] = qsignature
    parent_load = typed_proof(runtime / "typed-loads/software", parent_load)
    b.parent_load = parent_load
    warm = {
        "metrics": {
            key: value
            for key, value in b.summary.items()
            if key
            in {
                "stationary_residual_linf",
                "lower_pressure_rms",
                "lower_flux",
                "upper_flux",
                "gas_volume",
                "surface_volume",
                "target_volume",
            }
        },
        "signature": copy.deepcopy(b.cold["loaded"]),
        "actual_pins": copy.deepcopy(b.cold["actual_pins"]),
        "sampling_preserved_state": True,
    }
    after_remesh = copy.deepcopy(warm)
    terminal = {
        "before": copy.deepcopy(after_remesh),
        "after_remesh": after_remesh,
        "after_polish": warm,
        "explicit_force_remesh_calls": 1,
        "polish": {
            "status": "converged",
            "stage_updates": 1,
            "updates_total": 1,
            "remeshes_total": 0,
        },
    }

    def portable_observation(observed, phase):
        directory = runtime / "terminal/states" / phase
        directory.mkdir(parents=True)
        state, mesh = directory / "state.dump", directory / "mesh.msh"
        state.write_bytes(Path(b.restart["state"]["path"]).read_bytes())
        mesh.write_bytes((runtime / "mesh.msh").read_bytes())
        observed["retained_state"] = {
            "schema": "private-pressure-remesh-observation-state-v1",
            "phase": phase,
            "numerical_acceptance": False,
            "state": {**modes.regular_record(state), "format": "portable_file"},
            "meshes": [{**modes.regular_record(mesh), "stored_path": "mesh.msh"}],
            "mesh_reference_base": str(directory),
            "signature": copy.deepcopy(observed["signature"]),
            "actual_pins": copy.deepcopy(observed["actual_pins"]),
            "scope": "software-only portable observation",
        }

    portable_observation(terminal["before"], "before")
    portable_observation(terminal["after_remesh"], "after-remesh")
    # The actual producer cwd is its attempt. A content-identical retained mesh
    # supplies the authority for the synthetic memory snapshot's relative path.
    (runtime.parent / "mesh.msh").write_bytes((runtime / "mesh.msh").read_bytes())
    normalization = runtime / "terminal/normalization"
    normalization.mkdir(parents=True)
    for phase in ("before", "after"):
        (normalization / (phase + ".dump")).write_bytes(
            Path(b.restart["state"]["path"]).read_bytes()
        )
    with (normalization / "dummy-histories.npz").open("wb") as stream:
        np.savez_compressed(stream, before=np.zeros(4), after=np.zeros(4))
    checks = {
        name: True
        for name in (
            "physical_serialized_state_preserved",
            "normalized_snapshot_restored",
            "actual_pins_preserved",
            "signature_changes_confined_to_dummy_storage",
            "all_dummy_histories_equal_vertex_pressure_averages",
        )
    }
    b.audit = {
        "schema": "private-pressure-dummy-normalization-v1",
        "phase": "after-remesh",
        "verified": True,
        "numerical_acceptance": False,
        "checks": checks,
        "method": "liquid/gas mesh._interpolate_hanging_values; no solve, renumber or pin change",
        "source": {
            "pyoomph_commit": modes.diagnostic.PYOOMPH,
            "binding": "src/nanobind/mesh.cpp:1349",
            "histories": "src/elements_hanging.cpp:1143",
        },
        "before": copy.deepcopy(warm["signature"]),
        "after": copy.deepcopy(warm["signature"]),
        "actual_pins": {
            "before": copy.deepcopy(b.cold["actual_pins"]),
            "after": copy.deepcopy(b.cold["actual_pins"]),
        },
        "differing_signature_keys": [],
        "slots": [
            {
                "mesh": mesh,
                "node": 2,
                "value_index": 0,
                "equation_number": -1,
                "master_nodes": [0, 1],
                "history_offset": offset,
                "history_count": 2,
            }
            for mesh, offset in (("liquid", 0), ("gas", 2))
        ],
        "differences": [],
        "files": {
            name: modes.sha(normalization / name)
            for name in ("before.dump", "after.dump", "dummy-histories.npz")
        },
    }
    stages = []
    b.stage_files = {}
    for index, ca in ((4, -0.002), (5, -0.0016)):
        directory = runtime / "stages" / f"ca_{index:03d}"
        directory.mkdir(parents=True)
        journal = directory / "carry-budget.jsonl"
        journal.write_text('{"software_only":true}\n')
        carry = (
            {
                "carried": {"updates": 2, "remeshes": 0},
                "counter_seeded_once": True,
                "deactivated": True,
                "first_bound_trial_remaining_updates": 38,
                "new_spent": {"updates": 1, "remeshes": 0},
                "journal": modes.regular_record(journal),
            }
            if index == 4
            else None
        )
        current = modes.anchor.stage_case(case, ca, index, index == 5)
        case_path = directory / "case.json"
        write(case_path, current)
        signature = copy.deepcopy(warm["signature"])
        signature["parameters"]["Ca_in"] = ca
        state, mesh = directory / "state.dump", directory / "mesh.msh"
        state.write_bytes(Path(b.restart["state"]["path"]).read_bytes())
        mesh.write_bytes((runtime / "mesh.msh").read_bytes())
        manifest = {
            **copy.deepcopy(b.parent),
            "case_path": str(case_path),
            "case_file": str(case_path),
            "case_sha256": modes.sha(case_path),
            "case_id": current["case_id"],
            "anchor_stage": {
                "kind": "ascending_ca_continuation_from_prefix",
                "index": index,
                "ca_in": ca,
            },
        }
        manifest_path = directory / "run-manifest.json"
        write(manifest_path, manifest)
        packet = {
            "schema": modes.runner.RESUME_SCHEMA,
            "parent_manifest": modes.regular_record(manifest_path),
            "case_sha256": modes.sha(case_path),
            "azimuthal_stability": True,
            "state": {**modes.regular_record(state), "format": "portable_file"},
            "meshes": [{**modes.regular_record(mesh), "stored_path": "mesh.msh"}],
            "mesh_reference_base": str(directory),
            "stage": {
                "kind": "completed_polish",
                "bond_index": len(modes.model.stationary_bond_schedule(current)) - 1,
                "bond": 1.56,
                "ca_in": ca,
                "bc_mode": "stationary",
            },
            "roundtrip": {"verified": True, "after": signature, "layout": signature["layout"]},
        }
        packet_path = directory / "packet.json"
        write(packet_path, packet)
        row = {
            "index": index,
            "ca_in": ca,
            "case": modes.regular_record(case_path),
            "transition": {"verified": True, "software_only": True},
            "checkpoint": {"resume_packet": "packet.json"},
            "resume_packet_sha256": modes.sha(packet_path),
            "numerical_acceptance": False,
            "polish": {
                "status": "converged",
                "stage_updates": 1 if index == 5 else 3,
                "updates_total": 1 if index == 5 else 3,
                "remeshes_total": 0,
                "trials": [{"remaining_updates": 38 if index == 4 else 40}],
            },
            "carry": carry,
            "canonical_checks_observed": modes.remesh.canonical_checks(current, warm["metrics"]),
            "qualification_scope": "software-only provisional stage",
        }
        stages.append(row)
        b.stage_files[index] = {
            "case.json": current,
            "run-manifest.json": manifest,
            "packet.json": packet,
            "transition.json": row["transition"],
            "polish/ca_step/polish.json": row["polish"],
            "samples/after-polish.json": {
                "metrics": copy.deepcopy(warm["metrics"]),
                "signature": signature,
                "actual_pins": copy.deepcopy(warm["actual_pins"]),
                "sampling_preserved_state": True,
            },
        }
    b.result = {
        "schema": modes.anchor_i_prefix.SCHEMA,
        "stages": stages,
        "terminal": terminal,
        "ca_in": case["physics"]["ca_in"],
        "canonical_checks": copy.deepcopy(b.summary["checks"]),
        "cold_load_checks": copy.deepcopy(b.summary["cold_load_checks"]),
        "cold_load_verified": True,
        "completion_reached": True,
        "numerical_acceptance": True,
        "anchor_accepted": True,
        "causal_predictions": {},
        "checkpoint": {"resume_packet": "packet.json"},
        "original_acceptance_changed": False,
        "parents_unchanged": True,
        "qualified_prefix": qualified_prefix,
        "lineage": lineage,
        "carried": {"updates": 2, "remeshes": 0},
        "new_updates_total": 3,
        "total_newton_update_bound": 478,
        **modes.prefix_qualification.NO_CLAIMS,
    }
    b.summary = {
        "status": "completed",
        "completion_reached": True,
        "scientific_status": "checks_passed",
        "checks": b.result["canonical_checks"],
        "cold_load_checks": b.result["cold_load_checks"],
        "cold_load_verified": True,
        **warm["metrics"],
        "checkpoint": "packet.json",
        "ca_in": modes.anchor_i.TARGET_CA,
        "anchor_accepted": True,
        "original_acceptance_changed": False,
        **modes.prefix_qualification.NO_CLAIMS,
    }
    b.preparation = {
        "contract": contract,
        "lineage": lineage,
        "inputs": admitted["records"],
        "implementation": implementation,
        "numerical_acceptance": False,
    }
    b.admitted = admitted
    b.calls = []

    def prefix_admission(args):
        b.calls.append(args)
        assert args.source == modes.ROOT and args.case == target
        assert args.manifest == Path(seed["parent_manifest"]["path"])
        assert args.packet == Path(seed["packet"]["path"])
        assert args.state == Path(seed["state"]["path"])
        return admitted

    monkeypatch.setattr(modes.anchor_i_prefix, "preflight", prefix_admission)
    monkeypatch.setattr(
        modes.anchor_i,
        "preflight",
        lambda *args: pytest.fail("old I/failed-seed route must not consume prefix terminal"),
    )

    def source_compatibility(parent, current):
        assert parent["component_commit"] == producer
        assert parent["component"] == component and current == admitted["component"]
        return {
            "parent_commit": producer,
            "current_commit": consumer,
            "physics_ast_sha256": "software-identical-physics",
        }

    monkeypatch.setattr(modes.runner, "_parent_source_contract", source_compatibility)

    def publish():
        for name, value in {
            "run-manifest.json": b.parent,
            "summary.json": b.summary,
            "cold-load.json": b.cold,
            "result.json": b.result,
            "preparation.json": b.preparation,
            "parent-load.json": b.parent_load,
            "terminal/normalization/receipt.json": b.audit,
            "terminal/polish/terminal_remesh/polish.json": b.result["terminal"]["polish"],
            "parent-observation/samples/cold-load.json": {
                **warm,
                "signature": copy.deepcopy(b.parent_load["loaded"]),
                "actual_pins": copy.deepcopy(b.parent_load["actual_pins"]),
            },
            "target-cold/samples/cold-load.json": {
                **warm,
                "metrics": copy.deepcopy(warm["metrics"]),
                "signature": copy.deepcopy(b.cold["loaded"]),
                "actual_pins": copy.deepcopy(b.cold["actual_pins"]),
            },
        }.items():
            write(runtime / name, value)
        for stage in b.result["stages"]:
            write(runtime / "stages" / f"ca_{stage['index']:03d}" / "stage.json", stage)
            for name, record in b.stage_files[stage["index"]].items():
                write(runtime / "stages" / f"ca_{stage['index']:03d}" / name, record)
        for phase in ("before", "after_remesh", "after_polish"):
            write(
                runtime / "terminal/samples" / (phase.replace("_", "-") + ".json"),
                {
                    key: value
                    for key, value in b.result["terminal"][phase].items()
                    if key != "retained_state"
                },
            )
            if phase != "after_polish":
                write(
                    runtime / "terminal/states" / phase.replace("_", "-") / "retention.json",
                    b.result["terminal"][phase]["retained_state"],
                )
        b.restart["parent_manifest"] = modes.regular_record(runtime / "run-manifest.json")
        files = modes.prefix_qualification.completion_inventory(runtime, exclude_solver=True)
        write(
            runtime / "completion.json",
            {
                "schema": modes.anchor_i_prefix.SCHEMA,
                "status": "complete",
                "numerical_acceptance": True,
                "anchor_accepted": True,
                "original_acceptance_changed": False,
                "qualification_schema": modes.prefix_qualification.SCHEMA,
                "scope": modes.anchor_i_prefix.SCOPE,
                "files": files,
            },
        )

    b.publish = publish
    b.publish()
    return b


def test_exact_terminal_prefix_i_is_admitted_with_distinct_consumer_commit(prefix_i_bundle):
    b = prefix_i_bundle
    summary, cold, records = modes.qualified_parent(
        b.runtime, b.case, parent=b.parent, restart=b.restart
    )
    assert summary == b.summary and cold == b.cold
    assert len(b.calls) == 1
    assert any(item["path"].endswith("terminal/normalization/receipt.json") for item in records)
    assert any(item["path"].endswith("after-bc.dump") for item in records)
    assert b.parent["component_commit"] != b.admitted["restart"]["compatibility"]["current_commit"]


def test_pressure_audit_admits_distinct_before_signature_and_later_raw_dump(prefix_i_bundle):
    b = prefix_i_bundle
    before = b.audit["before"]["serialized_state"]
    before["sha256"] = "a" * 64
    before["payload_sha256"] = "b" * 64
    b.audit["differing_signature_keys"] = modes.remesh.differing_keys(
        b.audit["before"], b.audit["after"]
    )
    b.publish()
    directory = b.runtime / "terminal/normalization"
    raw_before = modes.runner.snapshot_bc_cache_signature(
        (directory / "before.dump").read_bytes(), mesh_reference_base=b.runtime.parent
    )
    assert raw_before != before
    assert raw_before["bc_cache_marker"] == before["bc_cache_marker"]
    summary, cold, _ = modes.qualified_parent(b.runtime, b.case, parent=b.parent, restart=b.restart)
    assert summary == b.summary and cold == b.cold


@pytest.mark.parametrize("phase", ["before", "after"])
def test_pressure_audit_still_refuses_changed_raw_cache_marker(prefix_i_bundle, phase):
    b = prefix_i_bundle
    path = b.runtime / "terminal/normalization" / (phase + ".dump")
    raw = path.read_bytes()
    marker = modes.runner.snapshot_bc_cache_signature(raw)["bc_cache_marker"]
    replacement = "init" if marker != "init" else "stationary"
    trailer = modes.runner._bc_cache_trailer(marker)
    path.write_bytes(raw[: -len(trailer)] + modes.runner._bc_cache_trailer(replacement))
    b.audit["files"][path.name] = modes.sha(path)
    b.publish()
    with pytest.raises(ValueError, match="pressure audit"):
        modes.qualified_parent(b.runtime, b.case, parent=b.parent, restart=b.restart)


@pytest.mark.parametrize(
    "fault",
    [
        "producer",
        "kind",
        "result_schema",
        "prefix_only",
        "false_anchor",
        "original_acceptance",
        "lineage",
        "contract",
        "current_commit",
        "physics_contract",
        "helper",
        "cold_history",
        "cold_pins",
        "cold_bc",
        "cold_gate",
        "warm_gate",
        "audit_failed",
        "audit_pins",
        "audit_slot",
        "audit_difference",
        "claimed_dynamic",
        "claimed_completeness",
        "stage_debit",
        "phase_bound",
        "missing_audit",
    ],
)
def test_terminal_prefix_i_refuses_forged_evidence(prefix_i_bundle, fault):
    b = prefix_i_bundle
    if fault == "producer":
        b.parent["runner"]["path"] = "/producer/runners/run_paper_prefix_qualify.py"
    elif fault == "kind":
        b.parent["anchor_stage"]["kind"] = "ascending_ca_continuation_from_prefix"
    elif fault == "result_schema":
        b.result["schema"] = modes.prefix_qualification.SCHEMA
    elif fault == "prefix_only":
        b.result["stationary_prefix_accepted"] = True
        b.result["anchor_accepted"] = False
    elif fault == "false_anchor":
        b.result["anchor_accepted"] = False
    elif fault == "original_acceptance":
        b.result["original_acceptance_changed"] = True
    elif fault == "lineage":
        b.result["lineage"] = {"schema": "forged"}
    elif fault == "contract":
        b.preparation["contract"] = {}
    elif fault == "current_commit":
        b.parent["parent_restart"]["compatibility"]["current_commit"] = "unrelated"
    elif fault == "physics_contract":
        b.parent["parent_restart"]["compatibility"]["physics_ast_sha256"] = "changed"
    elif fault == "helper":
        b.preparation["implementation"] = {}
    elif fault == "cold_history":
        b.cold["time_history"]["dts"] = [9.0]
    elif fault == "cold_pins":
        b.cold["actual_pins"] = {"sha256": "changed"}
    elif fault == "cold_bc":
        b.cold["bc_preparation"]["verified"] = False
    elif fault == "cold_gate":
        b.result["cold_load_checks"]["stationary"] = False
    elif fault == "warm_gate":
        b.result["terminal"]["after_polish"]["metrics"]["lower_pressure_rms"] = 1.0
    elif fault == "audit_failed":
        b.audit["checks"]["physical_serialized_state_preserved"] = False
    elif fault == "audit_pins":
        b.audit["actual_pins"]["after"] = {"sha256": "changed"}
    elif fault == "audit_slot":
        b.audit["slots"][0]["equation_number"] = 1
    elif fault == "audit_difference":
        b.audit["differences"] = [{"slot": 0, "history_index": 0, "before": 0.0, "after": 1.0}]
    elif fault == "claimed_dynamic":
        b.result["dynamic_restart_certificate"] = True
    elif fault == "claimed_completeness":
        b.result["finite_spectrum_complete"] = True
    elif fault == "stage_debit":
        b.result["stages"][0]["carry"]["first_bound_trial_remaining_updates"] = 40
    elif fault == "phase_bound":
        b.result["new_updates_total"] = 479
    b.publish()
    if fault == "missing_audit":
        path = b.runtime / "completion.json"
        completion = json.loads(path.read_text())
        completion["files"].pop("terminal/normalization/receipt.json")
        write(path, completion)
    with pytest.raises(ValueError):
        modes.qualified_parent(b.runtime, b.case, parent=b.parent, restart=b.restart)


@pytest.mark.parametrize(
    "fault",
    [
        "stage_case",
        "stage_manifest",
        "stage_packet",
        "stage_transition",
        "stage_polish",
        "stage_sample",
        "stage_history",
        "negative_debit",
        "terminal_retention",
        "terminal_portable_history",
        "terminal_mesh",
        "bc_method",
        "bc_before",
        "bc_after",
        "bc_raw_history",
        "bc_mesh_exchange",
        "bc_mesh_content",
        "initial_cold_sample",
    ],
)
def test_terminal_prefix_i_binds_direct_stage_and_cold_proofs(prefix_i_bundle, fault):
    b = prefix_i_bundle
    stage = b.result["stages"][0]
    direct = b.stage_files[4]
    if fault == "stage_case":
        direct["case.json"]["physics"]["ca_in"] = -0.0021
    elif fault == "stage_manifest":
        direct["run-manifest.json"]["qualified_prefix"] = {}
    elif fault == "stage_packet":
        direct["packet.json"]["stage"]["ca_in"] = -0.0021
    elif fault == "stage_transition":
        direct["transition.json"] = {"verified": False}
    elif fault == "stage_polish":
        direct["polish/ca_step/polish.json"] = {"status": "failed"}
    elif fault == "stage_sample":
        direct["samples/after-polish.json"]["sampling_preserved_state"] = False
    elif fault == "stage_history":
        direct["samples/after-polish.json"]["signature"] = copy.deepcopy(
            direct["samples/after-polish.json"]["signature"]
        )
        direct["samples/after-polish.json"]["signature"]["serialized_state"]["payload_sha256"] = (
            "changed"
        )
    elif fault == "negative_debit":
        stage["polish"]["stage_updates"] = stage["polish"]["updates_total"] = 1
        stage["carry"]["new_spent"]["updates"] = -1
        b.result["new_updates_total"] = 1
    elif fault == "terminal_retention":
        b.result["terminal"]["after_remesh"]["retained_state"]["numerical_acceptance"] = True
    elif fault == "terminal_portable_history":
        path = Path(b.result["terminal"]["after_remesh"]["retained_state"]["state"]["path"])
        path.write_bytes(path.read_bytes().replace(b"all opaque fields", b"bad opaque fields"))
        b.result["terminal"]["after_remesh"]["retained_state"]["state"]["sha256"] = modes.sha(path)
    elif fault == "terminal_mesh":
        b.result["terminal"]["after_remesh"]["retained_state"]["meshes"][0]["stored_path"] = (
            "unbound.msh"
        )
    elif fault == "bc_method":
        b.cold["bc_preparation"]["method"] = "set _last_bc_setting"
    elif fault == "bc_before":
        b.cold["bc_preparation"]["before"]["time"] = 9.0
    elif fault == "bc_after":
        b.cold["bc_preparation"]["after"]["serialized_state"]["payload_sha256"] = "changed"
    elif fault in {"bc_raw_history", "bc_mesh_exchange", "bc_mesh_content"}:
        directory = Path(b.cold["intervention"]["after"]["path"]).parent
        path = directory / "after-bc-retention.json"
        retention = json.loads(path.read_text())
        if fault == "bc_raw_history":
            raw = directory / "after-bc.dump"
            raw.write_bytes(raw.read_bytes().replace(b"all opaque fields", b"bad opaque fields"))
            retention["state"]["sha256"] = modes.sha(raw)
        elif fault == "bc_mesh_exchange":
            retention["mesh_exchange"] = {}
        else:
            mesh = Path(retention["meshes"][0]["path"])
            mesh.write_bytes(b"changed software mesh")
            retention["meshes"][0]["sha256"] = modes.sha(mesh)
        write(path, retention)
    b.publish()
    if fault == "initial_cold_sample":
        path = b.runtime / "parent-observation/samples/cold-load.json"
        sample = json.loads(path.read_text())
        sample["metrics"]["lower_pressure_rms"] = 1.0
        write(path, sample)
        completion_path = b.runtime / "completion.json"
        completion = json.loads(completion_path.read_text())
        completion["files"] = modes.prefix_qualification.completion_inventory(
            b.runtime, exclude_solver=True
        )
        write(completion_path, completion)
    with pytest.raises(ValueError):
        modes.qualified_parent(b.runtime, b.case, parent=b.parent, restart=b.restart)


def test_intermediate_stage_is_provisional_with_unchanged_terminal_acceptance(prefix_i_bundle):
    b = prefix_i_bundle
    metrics = b.stage_files[4]["samples/after-polish.json"]["metrics"]
    metrics["lower_pressure_rms"] = 1.0
    current = b.stage_files[4]["case.json"]
    b.result["stages"][0]["canonical_checks_observed"] = modes.remesh.canonical_checks(
        current, metrics
    )
    b.publish()
    _, cold, _ = modes.qualified_parent(b.runtime, b.case, parent=b.parent, restart=b.restart)
    assert all(cold["canonical_checks"].values())
    assert b.result["stages"][0]["numerical_acceptance"] is False
    assert b.result["stages"][0]["canonical_checks_observed"]["lower_static_pressure"] is False


@pytest.mark.parametrize("fault", ["extra", "object", "integer", "nonfinite", "shape", "unequal"])
def test_pressure_history_archive_refuses_nonproducer_layouts(tmp_path, fault):
    path = tmp_path / "histories.npz"
    arrays = {"before": np.zeros(4), "after": np.zeros(4)}
    if fault == "extra":
        arrays["arbitrary"] = np.zeros(1)
    elif fault == "object":
        arrays["before"] = np.array([{}], dtype=object)
    elif fault == "integer":
        arrays["before"] = np.zeros(4, dtype=np.int64)
    elif fault == "nonfinite":
        arrays["before"][0] = np.nan
    elif fault == "shape":
        arrays["before"] = np.zeros((2, 2))
    else:
        arrays["before"] = np.zeros(3)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        modes.pressure_history_arrays(path)


@pytest.mark.parametrize(
    "fault", [None, "qualification_helper", "continuation_helper", "method", "ancestry"]
)
def test_prefix_i_source_contract_binds_exact_producer_helpers_and_distinct_commits(
    monkeypatch, fault
):
    producer = "b0c9b609146e7b4881b0c7e1894216718b0be9d7"
    consumer = "7d3b3980c6ad22989620106bd76ac368cd931c44"
    sources = {
        name: {
            "sha256": modes.sha(modes.ROOT / name),
            "committed_sha256": modes.sha(modes.ROOT / name),
            "committed_matches": True,
        }
        for name in modes.runner.METHOD_FILES
    }
    current = {
        "commit": consumer,
        "head": consumer,
        "clean": True,
        "status": [],
        "sources": sources,
    }
    old = {**current, "commit": producer, "head": producer}
    implementation = {
        name: {"path": "/producer/" + name, "sha256": modes.sha(modes.ROOT / name)}
        for name in (*modes.HELPERS, *modes.PREFIX_I_HELPERS)
    }
    path = "/producer/runners/run_paper_anchor_i_prefix.py"
    parent = {
        "component_commit": producer,
        "component": old,
        "runner": {
            "path": path,
            "sha256": implementation[modes.anchor_i_prefix.ENTRYPOINT]["sha256"],
        },
        "runner_path": path,
        "runner_sha256": implementation[modes.anchor_i_prefix.ENTRYPOINT]["sha256"],
        "implementation": implementation,
        "anchor_stage": {"kind": "stationary_I_candidate_from_prefix"},
        "qualification_schema": modes.prefix_qualification.SCHEMA,
    }
    calls = []

    def git(root, *args):
        calls.append(args)
        if args[0] == "merge-base":
            if fault == "ancestry":
                raise ValueError("producer is not a consumer ancestor")
            return b""
        commit, name = args[1].split(":", 1)
        changed = {
            "qualification_helper": modes.prefix_qualification.ENTRYPOINT,
            "continuation_helper": modes.anchor_i_prefix.ENTRYPOINT,
            "method": "problems/paper_bubble.py",
        }.get(fault)
        raw = (modes.ROOT / name).read_bytes()
        return (
            raw + b"\n# changed producer bytes\n" if commit == producer and name == changed else raw
        )

    monkeypatch.setattr(modes.diagnostic, "git", git)
    monkeypatch.setattr(modes.runner, "_git", git)
    if fault is not None:
        with pytest.raises(ValueError):
            modes.source_contract(parent, current, typed_i=True)
        return
    result = modes.source_contract(parent, current, typed_i=True)
    assert (
        modes.anchor_i_prefix.ENTRYPOINT in result
        and modes.prefix_qualification.ENTRYPOINT in result
    )
    assert ("merge-base", "--is-ancestor", producer, consumer) in calls
    with pytest.raises(ValueError, match="declared typed I request"):
        modes.source_contract(parent, current, typed_i=False)
