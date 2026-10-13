"""Inspect the returned m=0/m=1 spectrum of one qualified two-phase anchor.

Four cold-loaded searches retain the actual Spectra J/M pencils, every returned
pair and physical interface fields. Partial searches establish no complete
finite spectrum, unique continuation identity or global spectral dominance.
Metadata preflight imports no native solver and creates no output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import struct
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems import paper_bubble as model
from runners import run_paper_anchor as anchor
from runners import run_paper_anchor_i as anchor_i
from runners import run_paper_anchor_i_prefix as anchor_i_prefix
from runners import run_paper_anchor_iii_from_stationary as anchor_iii
from runners import run_paper_case as runner
from runners import run_paper_iii_coarse_cold as coarse_cold
from runners import run_paper_prefix_qualify as prefix_qualification
from runners import run_paper_seed_requalify as requalification
from runners import run_pressure_diagnostic as diagnostic
from runners import run_pressure_remesh as remesh

require = diagnostic.require
sha = diagnostic.sha
write_json = diagnostic.write_json
MODES = (0, 1)
SHIFTS = (-0.001, 0.001)
NEIGEN = 8
ANCHORS = {-0.0016: "I", -0.00315: "III", anchor_iii.TARGET_CA: "III"}
GLOBALS = ("globals/U", "gas_volume/pressure")
EIGENDATA_FIELDS = (
    "_last_eigenvalues",
    "_last_eigenvectors",
    "_last_eigenvalues_m",
    "_last_eigenvalues_k",
)
HELPERS = (
    "runners/run_pressure_diagnostic.py",
    "runners/run_pressure_remesh.py",
    "runners/run_paper_anchor.py",
)
I_HELPERS = (
    anchor_i.ENTRYPOINT,
    "runners/run_paper_seed_i.py",
    "runners/run_paper_seed_recovery.py",
    "runners/run_paper_cold_history.py",
    requalification.ENTRYPOINT,
    "runners/run_paper_case.py",
)
PREFIX_I_HELPERS = (*I_HELPERS, prefix_qualification.ENTRYPOINT, anchor_i_prefix.ENTRYPOINT)
UNESTABLISHED_CLAIMS = {
    **prefix_qualification.NO_CLAIMS,
    "finite_spectrum_complete": False,
    "global_dominance_certified": False,
    "unique_family_identity_certified": False,
}
GATES = {"stationary", "lower_static_pressure", "flux", "volume", "volume_integrals"}
LIMITS = {
    "backward": 1e-7,
    "mass_action": 1e-8,
    "forced_coefficient": 1e-12,
    "two_shift_complex": 1e-7,
    "normal_overlap": 0.99,
    "epsilon": 1e-4,
    "halving": 1e-4,
    "normal_rank_relative": 1e-10,
}
API = {
    "pyoomph_commit": diagnostic.PYOOMPH,
    "spectra_pencils": "pyoomph/solvers/spectra.py:231,309,413",
    "normal_mode_numbering": "pyoomph/generic/problem.py:8632-8704",
    "full_snapshot_restore": "pyoomph/generic/problem.py:10769-10813",
    "full_eigen_snapshot_stream": "pyoomph/generic/problem.py:10442-10508",
    "cached_eigendata": "pyoomph/generic/problem.py:1142-1145,6343-6360",
    "official_field_incidence": "pyoomph/solvers/generic.py:EigenMatrixManipulatorBase",
    "kinematic_mass": "pyoomph/equations/navier_stokes.py:811-818",
    "gas_volume_constraint": "pyoomph/equations/ALE.py:596-614",
    "pencil_sign": "J=-dR/dU; Jv=lambda Mv",
}
SCOPE = "returned physical families at one anchor; no completeness or global dominance"


class RestorationFailure(RuntimeError):
    """A state or pin mismatch that must terminate the native search."""


def regular_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "input must be a regular non-symlink file")
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def relative_path(value):
    require(isinstance(value, str), "artifact path must be text")
    path = Path(value)
    require(
        isinstance(value, str) and value and not path.is_absolute() and ".." not in path.parts,
        "unsafe artifact path",
    )
    return path


def request_contract(case, shifts, neigen, *, metadata_only=False, coarse_iii=False):
    case = model.validate_case(case)
    p = case["physics"]
    reference = metadata_only and p["ca_in"] == 0.02
    require(p["ca_in"] in ANCHORS or reference, "only the declared stationary anchors are allowed")
    require(
        p["problem"] == "finite_gas_bubble"
        and p["bond"] == 1.56
        and p["domain_over_r"] == 15
        and p["vhat"] == 3.3
        and p["r0"] == 1,
        "anchor geometry or formulation differs",
    )
    require(tuple(shifts) == SHIFTS and neigen == NEIGEN, "bounded mode/shift/pair request differs")
    require(not coarse_iii or p["ca_in"] == anchor_iii.TARGET_CA,
            "coarse m0 request requires the exact typed III cold observation")
    return {
        "schema": coarse_cold.REQUEST_SCHEMA if coarse_iii else "paper-returned-modes-request-v1",
        "branch_candidate": ANCHORS.get(p["ca_in"], "III positive reference; metadata only"),
        "ca_in": p["ca_in"],
        "modes": [0] if coarse_iii else list(MODES),
        "shifts": list(SHIFTS),
        "neigen": NEIGEN,
        "eigenproblem_calls": 2 if coarse_iii else 4,
        "paper_time_factor": p["r_tube"] / p["r0"],
        "limits": LIMITS,
        "reference_metadata_only": reference,
        "scope": SCOPE,
        "finite_spectrum_complete": False,
        "global_dominance_certified": False,
    }


def qualified_parent(runtime, case, *, parent=None, restart=None):
    """Verify the producer's complete five-gate and fresh cold-load evidence."""
    if parent is not None and coarse_cold.is_producer(parent):
        require(restart is not None, "typed coarse cold packet required")
        return coarse_cold.admit_observation(runtime, parent, restart, case)
    runtime = Path(runtime).resolve(strict=True)
    completion_path = runtime / "completion.json"
    completion = json.loads(completion_path.read_text())
    require(
        completion.get("status") == "complete" and completion.get("numerical_acceptance") is True,
        "parent lacks numerical acceptance",
    )
    files = completion.get("files", {})
    require(
        isinstance(files, dict)
        and {"run-manifest.json", "summary.json", "cold-load.json"} <= set(files),
        "parent qualification files are unbound",
    )
    records = [regular_record(completion_path)]
    for name, digest in files.items():
        path = runtime / relative_path(name)
        require(
            path.resolve(strict=True).is_relative_to(runtime), "completion escaped parent runtime"
        )
        record = regular_record(path)
        require(record["sha256"] == digest, "parent completion artifact differs")
        records.append(record)
    summary = json.loads((runtime / "summary.json").read_text())
    cold = json.loads((runtime / "cold-load.json").read_text())
    checks = (
        summary.get("checks", {}),
        summary.get("cold_load_checks", {}),
        cold.get("canonical_checks", {}),
    )
    require(
        summary.get("completion_reached") is True
        and summary.get("cold_load_verified") is True
        and cold.get("verified") is True
        and all(set(value) == GATES and all(v is True for v in value.values()) for value in checks),
        "parent scientific or cold-load gates failed",
    )
    require(
        summary.get("ca_in") == case["physics"]["ca_in"]
        and ("bond" not in summary or summary["bond"] == case["physics"]["bond"]),
        "parent qualification belongs to different forcing",
    )
    require(
        all(remesh.canonical_checks(case, summary).values()),
        "parent metrics fail the unchanged scientific gates",
    )
    require(isinstance(cold.get("actual_pins"), dict), "parent cold load lacks actual pins")
    require(
        isinstance(cold.get("loaded", {}).get("serialized_state", {}).get("payload_sha256"), str),
        "parent cold load lacks complete history identity",
    )
    require(
        cold["loaded"]["parameters"]
        == {"Ca_in": case["physics"]["ca_in"], "Bo": case["physics"]["bond"]},
        "parent cold-load forcing differs",
    )
    if case["physics"]["ca_in"] == anchor_i.TARGET_CA:
        require(parent is not None and restart is not None, "typed I parent and packet required")
        if is_prefix_i_producer(parent):
            records.extend(
                qualified_prefix_i_parent(
                    runtime, case, parent, restart, summary, cold, files, completion
                )
            )
        else:
            require(
                completion.get("parent_acceptance_changed") is False,
                "I completion changed parent acceptance",
            )
            records.extend(qualified_i_parent(runtime, case, parent, restart, summary, cold, files))
    if case["physics"]["ca_in"] == anchor_iii.TARGET_CA:
        require(
            parent is not None and restart is not None,
            "exact typed III producer and packet required",
        )
        records.extend(
            anchor_iii.admit_terminal(
                runtime, parent, restart, case, summary, cold, files, completion
            )
        )
        result = runner._read_json(runtime / "result.json")
        terminal = result["terminal"]
        records.extend(bind_pressure_history_audit(runtime, terminal, files))
        for phase in ("before", "after_remesh"):
            records.extend(
                bind_portable_observation(terminal[phase], phase.replace("_", "-"), runtime, files)
            )
    return summary, cold, records


def qualified_i_parent(runtime, case, parent, restart, summary, cold, files):
    """Bind a completed I descendant to its admitted requalified seed, not a failed run."""
    require(not (runtime / "failure.json").exists(), "I parent has a failure receipt")
    require(
        {"result.json", "preparation.json"} <= set(files), "I admission/result files are unbound"
    )
    result = json.loads((runtime / "result.json").read_text())
    preparation = json.loads((runtime / "preparation.json").read_text())
    require(
        parent.get("schema") == "pyoomph-runner-manifest-v1"
        and Path(parent.get("runner", {}).get("path", "")).name == "run_paper_anchor_i.py"
        and parent.get("anchor_stage") == {"kind": "stationary_I_candidate"}
        and parent.get("parent_acceptance_changed") is False
        and result.get("schema") == "private-paper-stationary-anchor-v1"
        and result.get("completion_reached") is True
        and result.get("numerical_acceptance") is True
        and result.get("cold_load_verified") is True
        and result.get("canonical_checks") == summary["checks"]
        and result.get("cold_load_checks") == summary["cold_load_checks"]
        and result.get("ca_in") == case["physics"]["ca_in"]
        and result.get("parent_acceptance_changed") is False
        and result.get("parents_unchanged") is True
        and result.get("branch_identity_verified") is False
        and result.get("eigenanalysis_performed") is False
        and summary.get("status") == "completed"
        and summary.get("branch_identity_verified") is False
        and summary.get("eigenanalysis_performed") is False,
        "I producer/schema or qualification differs",
    )
    for record in (restart["packet"], restart["state"], *restart["meshes"]):
        path = Path(record["path"])
        require(
            path.is_relative_to(runtime)
            and files.get(str(path.relative_to(runtime))) == record["sha256"],
            "I terminal packet/state/mesh is unbound",
        )
    require(
        restart["parent_manifest"] == regular_record(runtime / "run-manifest.json")
        and (runtime / relative_path(result["checkpoint"]["resume_packet"])).resolve(strict=True)
        == Path(restart["packet"]["path"]),
        "I terminal checkpoint lineage differs",
    )
    seed_restart = parent["parent_restart"]
    contract = parent["case_transition_contract"]
    # The I runner admits the qualified seed first; only that seed's explicit
    # original_failed_producer pointer enters the failed-only recovery preflight.
    admitted = anchor_i.preflight(
        SimpleNamespace(
            source=ROOT,
            case=Path(parent["case_path"]),
            manifest=Path(seed_restart["parent_manifest"]["path"]),
            packet=Path(seed_restart["packet"]["path"]),
            state=Path(seed_restart["state"]["path"]),
            ca_ladder=Path(contract["ca_ladder"]["path"]),
        )
    )
    # validate_restart recomputes the consumer revision at each admission.
    # Compare the historical receipt at the I producer's revision; all packet,
    # source-contract and direct-artifact fields still have to match exactly.
    historical_restart = {
        **admitted["restart"],
        "compatibility": {
            **admitted["restart"]["compatibility"],
            "current_commit": parent["component_commit"],
        },
    }
    require(
        historical_restart == seed_restart
        and admitted["target"] == case
        and admitted["contract"] == contract == preparation.get("contract")
        and parent.get("construction_case") == contract["parent_case"]
        and preparation.get("inputs") == admitted["records"]
        and preparation.get("numerical_acceptance") is False
        and preparation.get("implementation") == parent["implementation"],
        "I seed admission or lineage differs from direct evidence",
    )
    expected = requalification.cold.serialized_metadata(Path(restart["state"]["path"]).read_bytes())
    require(
        expected["endtime"] == case["solve"]["relax_time"] == 8.0
        and runner.snapshot_bc_cache_signature(
            Path(restart["state"]["path"]).read_bytes(),
            mesh_reference_base=Path(restart["mesh_reference_base"]),
        )["payload_sha256"]
        == cold["loaded"]["serialized_state"]["payload_sha256"]
        == restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
        and cold["actual_pins"] == result["terminal"]["after_polish"]["actual_pins"]
        and cold.get("bc_preparation", {}).get("verified") is True
        and cold.get("intervention", {}).get("verified") is True
        and cold["intervention"].get("kind") == "typed-native-endtime-restoration"
        and cold["intervention"].get("saved_endtime") == expected["endtime"]
        and cold["intervention"].get("api") == requalification.cold.API
        and cold.get("time_history")
        == {key: expected[key] for key in ("time", "dts", "unsteady_steps", "suggested_next_dt")},
        "I typed restoration/BC preparation or time history differs",
    )
    remesh.check_loaded_state(cold["loaded"], restart, case, cold["actual_pins"])
    return admitted["records"]


def bound_parent_file(record, runtime, files):
    """Bind each direct proof pointer to the producer's immutable completion."""
    actual = regular_record(record["path"])
    path = Path(actual["path"])
    require(
        path.is_relative_to(runtime)
        and actual["sha256"] == record["sha256"]
        and files.get(str(path.relative_to(runtime))) == actual["sha256"],
        "prefix I direct proof is unbound or escaped its runtime",
    )
    return actual


def physical_signature(signature):
    """Keep full opaque history; only BC-cache labels and raw mesh paths are separate."""
    return {
        **{
            key: value
            for key, value in signature.items()
            if key not in {"serialized_state", "bc_cache_marker"}
        },
        "payload_sha256": signature["serialized_state"]["payload_sha256"],
    }


def refuse_unestablished_claims(value):
    require(
        all(value.get(key, False) is False for key in UNESTABLISHED_CLAIMS),
        "prefix I producer claims an unestablished branch, dynamic or spectral certificate",
    )


def original_polish_bounds(polish):
    return polish.get("status") == "converged" and all(
        type(polish.get(key)) is int and 0 <= polish[key] <= limit
        for key, limit in (("stage_updates", 40), ("updates_total", 160), ("remeshes_total", 3))
    )


def pressure_history_arrays(path):
    """Read only the producer's two plain little-endian f64 NPY arrays; no pickle."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        require(
            len(archive.namelist()) == 2 and set(archive.namelist()) == {"before.npy", "after.npy"},
            "pressure-history archive inventory differs",
        )
        for phase in ("before", "after"):
            stream = io.BytesIO(archive.read(phase + ".npy"))
            require(stream.read(6) == b"\x93NUMPY", "pressure-history array format differs")
            version = stream.read(2)
            require(version in {b"\x01\x00", b"\x02\x00"}, "unsupported pressure-history header")
            width = 2 if version == b"\x01\x00" else 4
            length = int.from_bytes(stream.read(width), "little")
            require(0 < length <= 65536, "pressure-history header size differs")
            header = ast.literal_eval(stream.read(length).decode("latin1"))
            require(
                set(header) == {"descr", "fortran_order", "shape"}
                and header["descr"] == "<f8"
                and header["fortran_order"] is False
                and isinstance(header["shape"], tuple)
                and len(header["shape"]) == 1
                and type(header["shape"][0]) is int
                and header["shape"][0] > 0,
                "pressure-history dtype/layout differs",
            )
            raw = stream.read()
            require(len(raw) == 8 * header["shape"][0], "pressure-history payload size differs")
            values = [value[0] for value in struct.iter_unpack("<d", raw)]
            require(
                all(math.isfinite(value) for value in values), "nonfinite dummy-pressure history"
            )
            result[phase] = values
    require(len(result["before"]) == len(result["after"]), "pressure-history lengths differ")
    return result


def bind_pressure_history_audit(runtime, terminal, files):
    """Bind the original native full-history normalization proof and its slot changes."""
    directory = runtime / "terminal/normalization"
    receipt_path = directory / "receipt.json"
    records = [bound_parent_file(regular_record(receipt_path), runtime, files)]
    receipt = runner._read_json(receipt_path)
    expected_checks = {
        "physical_serialized_state_preserved",
        "normalized_snapshot_restored",
        "actual_pins_preserved",
        "signature_changes_confined_to_dummy_storage",
        "all_dummy_histories_equal_vertex_pressure_averages",
    }
    require(
        receipt.get("schema") == "private-pressure-dummy-normalization-v1"
        and receipt.get("phase") == "after-remesh"
        and receipt.get("verified") is True
        and receipt.get("numerical_acceptance") is False
        and set(receipt.get("checks", {})) == expected_checks
        and all(value is True for value in receipt["checks"].values())
        and receipt.get("method")
        == "liquid/gas mesh._interpolate_hanging_values; no solve, renumber or pin change"
        and receipt.get("source")
        == {
            "pyoomph_commit": diagnostic.PYOOMPH,
            "binding": "src/nanobind/mesh.cpp:1349",
            "histories": "src/elements_hanging.cpp:1143",
        },
        "prefix I pressure-dummy full-history audit failed or differs",
    )
    refuse_unestablished_claims(receipt)
    permitted = {
        "arrays.pinned_values_with_positions.sha256",
        "serialized_state.sha256",
        "serialized_state.payload_sha256",
    }
    changed = remesh.differing_keys(receipt["before"], receipt["after"])
    require(
        changed == receipt["differing_signature_keys"]
        and set(changed) <= permitted
        and receipt["actual_pins"]["before"]
        == receipt["actual_pins"]["after"]
        == terminal["after_remesh"]["actual_pins"]
        and physical_signature(receipt["after"])
        == physical_signature(terminal["after_remesh"]["signature"]),
        "pressure audit changed physical fields/pins or belongs to another remesh",
    )
    require(
        set(receipt["files"]) == {"before.dump", "after.dump", "dummy-histories.npz"},
        "pressure audit raw proofs missing",
    )
    for name, digest in receipt["files"].items():
        records.append(
            bound_parent_file({"path": str(directory / name), "sha256": digest}, runtime, files)
        )
    # The producer's snapshots use its attempt cwd. Retained observation meshes
    # bind those original references even though solver files are excluded from completion.
    retained = terminal["after_remesh"]["retained_state"]
    require(
        retained["phase"] == "after-remesh" and retained["numerical_acceptance"] is False,
        "remesh observation cannot be an accepted anchor",
    )
    authority = {item["sha256"] for item in retained["meshes"]}
    records.extend(
        bound_parent_file(item, runtime, files) for item in (retained["state"], *retained["meshes"])
    )
    after_references = runner.state_mesh_references(directory / "after.dump")
    for phase in ("before", "after"):
        raw_path = directory / (phase + ".dump")
        raw = raw_path.read_bytes()
        for _, _, reference in requalification.cold.serialized_metadata(raw)["meshes"]:
            path = (runtime.parent / reference).resolve(strict=True)
            record = regular_record(path)
            require(
                record["sha256"] in authority,
                "pressure snapshot mesh lacks retained content authority",
            )
            records.append(record)
        fingerprint = runner.snapshot_bc_cache_signature(raw, mesh_reference_base=runtime.parent)
        if phase == "after":
            require(
                fingerprint == receipt[phase]["serialized_state"],
                "pressure audit raw snapshot and signature differ",
            )
        else:
            # The producer records the before signature before taking before.dump.
            require(
                fingerprint["bc_cache_marker"] == receipt[phase]["bc_cache_marker"]
                and runner.state_mesh_references(raw_path) == after_references,
                "pressure audit before snapshot cache or mesh references differ",
            )
    values = pressure_history_arrays(directory / "dummy-histories.npz")
    offset, differences = 0, []
    require(
        receipt["slots"] and {item["mesh"] for item in receipt["slots"]} == {"liquid", "gas"},
        "both phase dummy-pressure incidence required",
    )
    for index, item in enumerate(receipt["slots"]):
        count = item["history_count"]
        require(
            type(count) is int
            and count > 0
            and item["history_offset"] == offset
            and type(item["equation_number"]) is int
            and item["equation_number"] < 0
            and type(item["node"]) is int
            and item["node"] >= 0
            and type(item["value_index"]) is int
            and item["value_index"] >= 0
            and isinstance(item["master_nodes"], list)
            and len(item["master_nodes"]) == 2
            and all(type(node) is int and node >= 0 for node in item["master_nodes"])
            and offset + count <= len(values["before"]),
            "pressure audit slot/history incidence differs",
        )
        for history in range(count):
            before, after = values["before"][offset + history], values["after"][offset + history]
            if struct.pack("<d", before) != struct.pack("<d", after):
                differences.append(
                    {"slot": index, "history_index": history, "before": before, "after": after}
                )
        offset += count
    require(
        offset == len(values["before"]) and differences == receipt["differences"],
        "pressure audit history changes differ from retained arrays",
    )
    return records


def bind_typed_proof_inventory(receipt, runtime, files):
    records, fingerprints, bases = [], {}, {}
    directory = Path(receipt["intervention"]["after"]["path"]).parent
    for phase in ("before", "after", "after-bc"):
        record = bound_parent_file(
            receipt["intervention"][phase]
            if phase != "after-bc"
            else regular_record(directory / "after-bc.dump"),
            runtime,
            files,
        )
        records.append(record)
        path = Path(record["path"])
        retention_path = path.parent / (phase + "-retention.json")
        records.append(bound_parent_file(regular_record(retention_path), runtime, files))
        retained = runner._read_json(retention_path)
        require(
            set(retained) == {"state", "mesh_reference_base", "meshes", "mesh_exchange"}
            and retained["state"] == {**record, "format": "memory_snapshot"}
            and [item["stored_path"] for item in retained["meshes"]]
            == runner.state_mesh_references(path)
            and retained["mesh_exchange"]
            == {item["stored_path"]: item["path"] for item in retained["meshes"]},
            "typed snapshot retention or mesh exchange differs",
        )
        records.extend(bound_parent_file(item, runtime, files) for item in retained["meshes"])
        bases[phase] = Path(retained["mesh_reference_base"]).resolve(strict=True)
        for item in retained["meshes"]:
            reference = regular_record(bases[phase] / item["stored_path"])
            require(reference["sha256"] == item["sha256"], "typed retained mesh content differs")
            records.append(reference)
        fingerprints[phase] = runner.snapshot_bc_cache_signature(
            path.read_bytes(), mesh_reference_base=bases[phase]
        )
    preparation = receipt["bc_preparation"]
    require(
        len(set(bases.values())) == 1
        and preparation["verified"] is True
        and preparation["method"] == "actions_before_stationary_solve(force_reassign_eqs=False)"
        and preparation["before"] == receipt["loaded"]
        and physical_signature(preparation["before"]) == physical_signature(preparation["after"])
        and fingerprints["after"] == preparation["before"]["serialized_state"]
        and fingerprints["after-bc"] == preparation["after"]["serialized_state"],
        "BC preparation signatures or full native history differ",
    )
    return records


def bind_prefix_stage(runtime, stage, case, parent, admitted, files):
    """Bind each provisional suffix row to its actual source, case and checkpoint."""
    index = stage["index"]
    directory = runtime / "stages" / f"ca_{index:03d}"
    case_path, manifest_path = directory / "case.json", directory / "run-manifest.json"
    records = [
        bound_parent_file(regular_record(path), runtime, files)
        for path in (case_path, manifest_path)
    ]
    current = anchor.stage_case(case, stage["ca_in"], index, index == 5)
    manifest = runner._read_json(manifest_path)
    require(
        runner.read_case(case_path) == current
        and stage["case"] == regular_record(case_path)
        and manifest["anchor_stage"]
        == {
            "kind": "ascending_ca_continuation_from_prefix",
            "index": index,
            "ca_in": stage["ca_in"],
        }
        and all(
            manifest[key] == parent[key]
            for key in (
                "schema",
                "runner",
                "runner_path",
                "runner_sha256",
                "component_commit",
                "component",
                "pyoomph",
                "implementation",
                "lineage",
                "qualification_schema",
                "case_transition_contract",
                "construction_case",
                "parent_restart",
                "qualified_prefix",
                "original_acceptance_changed",
                "azimuthal_stability",
            )
        ),
        "remaining I stage case or producer lineage differs",
    )
    restart = runner.validate_restart(
        directory / relative_path(stage["checkpoint"]["resume_packet"]),
        current,
        sha(case_path),
        component=admitted["component"],
        environment=admitted["environment"],
        azimuthal_stability=True,
    )
    require(
        restart["parent_manifest"] == regular_record(manifest_path)
        and restart["state"]["format"] == "portable_file"
        and restart["stage"]["kind"] == "completed_polish"
        and restart["stage"]["bond_index"] == len(model.stationary_bond_schedule(current)) - 1
        and stage["resume_packet_sha256"] == restart["packet"]["sha256"],
        "remaining I stage checkpoint differs",
    )
    records.extend(
        bound_parent_file(item, runtime, files)
        for item in (restart["packet"], restart["state"], *restart["meshes"])
    )
    transition_path = directory / "transition.json"
    sample_path = directory / "samples/after-polish.json"
    polish_path = directory / "polish/ca_step/polish.json"
    records.extend(
        bound_parent_file(regular_record(path), runtime, files)
        for path in (transition_path, sample_path, polish_path)
    )
    sample = runner._read_json(sample_path)
    require(
        stage["transition"] == runner._read_json(transition_path)
        and stage["transition"]["verified"] is True
        and stage["polish"] == runner._read_json(polish_path)
        and sample["sampling_preserved_state"] is True
        and physical_signature(sample["signature"])
        == physical_signature(restart["roundtrip"]["after"])
        and stage["canonical_checks_observed"]
        == remesh.canonical_checks(current, sample["metrics"]),
        "remaining I stage transition, polish or sampled state differs",
    )
    # Intermediate observations remain provisional; only terminal warm/cold
    # checks determine the unchanged anchor acceptance.
    return records


def bind_portable_observation(observed, phase, runtime, files):
    retained = observed["retained_state"]
    path = runtime / "terminal/states" / phase / "retention.json"
    records = [bound_parent_file(regular_record(path), runtime, files)]
    require(
        retained == runner._read_json(path)
        and retained["schema"] == "private-pressure-remesh-observation-state-v1"
        and retained["phase"] == phase
        and retained["numerical_acceptance"] is False
        and retained["state"]["format"] == "portable_file"
        and retained["signature"] == observed["signature"]
        and retained["actual_pins"] == observed["actual_pins"],
        "terminal retained observation differs",
    )
    records.extend(
        bound_parent_file(item, runtime, files) for item in (retained["state"], *retained["meshes"])
    )
    state = Path(retained["state"]["path"])
    base = Path(retained["mesh_reference_base"]).resolve(strict=True)
    require(
        base == state.parent
        and [item["stored_path"] for item in retained["meshes"]]
        == runner.state_mesh_references(state)
        and all(
            (base / item["stored_path"]).resolve(strict=True) == Path(item["path"])
            for item in retained["meshes"]
        )
        and runner.snapshot_bc_cache_signature(state.read_bytes(), mesh_reference_base=base)[
            "payload_sha256"
        ]
        == retained["signature"]["serialized_state"]["payload_sha256"],
        "terminal portable observation mesh or full history differs",
    )
    return records


def qualified_prefix_i_parent(runtime, case, parent, restart, summary, cold, files, completion):
    """Admit exactly a terminal prefix-descended I candidate, preserving the old route."""
    require(not (runtime / "failure.json").exists(), "prefix I parent has a failure receipt")
    require(
        set(completion)
        == {
            "schema",
            "status",
            "numerical_acceptance",
            "anchor_accepted",
            "original_acceptance_changed",
            "qualification_schema",
            "scope",
            "files",
        }
        and completion["schema"] == anchor_i_prefix.SCHEMA
        and completion["anchor_accepted"] is True
        and completion["original_acceptance_changed"] is False
        and completion["qualification_schema"] == prefix_qualification.SCHEMA
        and completion["scope"] == anchor_i_prefix.SCOPE
        and {
            "result.json",
            "preparation.json",
            "parent-load.json",
            "terminal/normalization/receipt.json",
        }
        <= set(files),
        "prefix I terminal completion/schema is absent or differs",
    )
    require(
        prefix_qualification.completion_inventory(runtime, exclude_solver=True) == files,
        "prefix I complete proof inventory differs",
    )
    result = runner._read_json(runtime / "result.json")
    preparation = runner._read_json(runtime / "preparation.json")
    expected_result_fields = {
        "schema",
        "stages",
        "terminal",
        "ca_in",
        "canonical_checks",
        "cold_load_checks",
        "cold_load_verified",
        "completion_reached",
        "numerical_acceptance",
        "anchor_accepted",
        "causal_predictions",
        "checkpoint",
        "original_acceptance_changed",
        "parents_unchanged",
        "qualified_prefix",
        "lineage",
        "carried",
        "new_updates_total",
        "total_newton_update_bound",
        *prefix_qualification.NO_CLAIMS,
    }
    require(
        set(result) == expected_result_fields
        and parent.get("schema") == "pyoomph-runner-manifest-v1"
        and is_prefix_i_producer(parent)
        and parent.get("anchor_stage") == {"kind": "stationary_I_candidate_from_prefix"}
        and parent.get("qualification_schema") == prefix_qualification.SCHEMA
        and parent.get("original_acceptance_changed") is False
        and result.get("schema") == anchor_i_prefix.SCHEMA
        and result.get("anchor_accepted") is True
        and result.get("numerical_acceptance") is True
        and result.get("completion_reached") is True
        and result.get("cold_load_verified") is True
        and result.get("original_acceptance_changed") is False
        and result.get("parents_unchanged") is True
        and result.get("ca_in") == case["physics"]["ca_in"] == anchor_i.TARGET_CA,
        "prefix I producer/result is not a qualified terminal candidate",
    )
    for value in (parent, result, summary, completion):
        refuse_unestablished_claims(value)
    prefix_restart = parent["parent_restart"]
    admitted = anchor_i_prefix.preflight(
        SimpleNamespace(
            source=ROOT,
            case=Path(parent["case_path"]),
            manifest=Path(prefix_restart["parent_manifest"]["path"]),
            packet=Path(prefix_restart["packet"]["path"]),
            state=Path(prefix_restart["state"]["path"]),
        )
    )
    historical_restart = {
        **admitted["restart"],
        "compatibility": {
            **admitted["restart"]["compatibility"],
            "current_commit": parent["component_commit"],
        },
    }
    contract = admitted["contract"]
    require(
        historical_restart == prefix_restart
        and admitted["target"] == case
        and parent["case_transition_contract"] == preparation["contract"] == contract
        and parent["lineage"] == preparation["lineage"] == result["lineage"] == admitted["lineage"]
        and parent["construction_case"] == contract["parent_case"]
        and parent["qualified_prefix"] == result["qualified_prefix"] == contract["qualified_prefix"]
        and preparation["inputs"] == admitted["records"]
        and preparation["numerical_acceptance"] is False
        and preparation["implementation"] == parent["implementation"]
        and set(parent["implementation"]) == set(admitted["implementation"])
        and all(
            parent["implementation"][name]["sha256"] == declaration["sha256"]
            for name, declaration in admitted["implementation"].items()
        )
        and result["carried"] == contract["carried"] == {"updates": 2, "remeshes": 0}
        and result["total_newton_update_bound"] == contract["total_newton_update_bound"] == 478
        and type(result["new_updates_total"]) is int
        and 0 <= result["new_updates_total"] <= 478,
        "prefix I immutable admission, source contract, lineage or carried bounds differ",
    )
    records = list(admitted["records"])
    for record in (restart["packet"], restart["state"], *restart["meshes"]):
        records.append(bound_parent_file(record, runtime, files))
    require(
        restart["parent_manifest"] == regular_record(runtime / "run-manifest.json")
        and (runtime / relative_path(result["checkpoint"]["resume_packet"])).resolve(strict=True)
        == Path(restart["packet"]["path"])
        and restart["state"]["format"] == "portable_file"
        and restart["stage"]["kind"] == "completed_polish",
        "prefix I terminal checkpoint lineage differs",
    )
    terminal = result["terminal"]
    require(len(result["stages"]) == 2, "only the remaining original I stages are admissible")
    new_updates = 0
    for index, value, stage in zip((4, 5), (-0.002, -0.0016), result["stages"], strict=True):
        path = runtime / "stages" / f"ca_{index:03d}" / "stage.json"
        records.append(bound_parent_file(regular_record(path), runtime, files))
        require(
            runner._read_json(path) == stage
            and stage["index"] == index
            and stage["ca_in"] == value
            and stage["numerical_acceptance"] is False
            and original_polish_bounds(stage["polish"]),
            "remaining I stage identity or original polish bound differs",
        )
        records.extend(bind_prefix_stage(runtime, stage, case, parent, admitted, files))
        refuse_unestablished_claims(stage)
        if index == 4:
            carry = stage["carry"]
            require(
                carry["carried"] == {"updates": 2, "remeshes": 0}
                and carry["counter_seeded_once"] is True
                and carry["deactivated"] is True
                and carry["first_bound_trial_remaining_updates"] == 38
                and stage["polish"]["trials"][0]["remaining_updates"] == 38
                and stage["polish"]["updates_total"] >= 2
                and all(
                    type(carry["new_spent"][key]) is int and carry["new_spent"][key] >= 0
                    for key in ("updates", "remeshes")
                )
                and carry["new_spent"]["updates"] == stage["polish"]["updates_total"] - 2
                and carry["new_spent"]["remeshes"] == stage["polish"]["remeshes_total"],
                "first I stage lost its certified debit",
            )
            records.append(bound_parent_file(carry["journal"], runtime, files))
            new_updates += carry["new_spent"]["updates"]
        else:
            require(
                stage["carry"] is None, "later I stage must retain its original independent budget"
            )
            new_updates += stage["polish"]["updates_total"]
    for phase in ("before", "after_remesh", "after_polish"):
        observed = terminal[phase]
        name = phase.replace("_", "-")
        path = runtime / "terminal/samples" / (name + ".json")
        records.append(bound_parent_file(regular_record(path), runtime, files))
        require(
            runner._read_json(path)
            == {key: item for key, item in observed.items() if key != "retained_state"}
            and observed["sampling_preserved_state"] is True,
            "terminal phase observation differs from its direct sample",
        )
        if phase != "after_polish":
            records.extend(bind_portable_observation(observed, name, runtime, files))
    warm = terminal["after_polish"]
    terminal_polish_path = runtime / "terminal/polish/terminal_remesh/polish.json"
    records.append(bound_parent_file(regular_record(terminal_polish_path), runtime, files))
    require(
        terminal["polish"] == runner._read_json(terminal_polish_path),
        "terminal I polish differs from its direct receipt",
    )
    expected_summary = {
        "status": "completed",
        "completion_reached": True,
        "scientific_status": "checks_passed",
        "checks": result["canonical_checks"],
        "cold_load_checks": result["cold_load_checks"],
        "cold_load_verified": True,
        **warm["metrics"],
        "checkpoint": result["checkpoint"]["resume_packet"],
        "ca_in": anchor_i.TARGET_CA,
        "anchor_accepted": True,
        "original_acceptance_changed": False,
        **prefix_qualification.NO_CLAIMS,
    }
    require(
        summary == expected_summary
        and set(result["canonical_checks"]) == set(result["cold_load_checks"]) == GATES
        and result["canonical_checks"] == remesh.canonical_checks(case, warm["metrics"])
        and all(
            value is True
            for value in (
                *result["canonical_checks"].values(),
                *result["cold_load_checks"].values(),
            )
        )
        and type(terminal["explicit_force_remesh_calls"]) is int
        and terminal["explicit_force_remesh_calls"] == 1
        and original_polish_bounds(terminal["polish"])
        and new_updates + terminal["polish"]["updates_total"] == result["new_updates_total"]
        and physical_signature(warm["signature"])
        == physical_signature(restart["roundtrip"]["after"]),
        "prefix I terminal remesh, warm gates or full state differs",
    )
    records.extend(bind_pressure_history_audit(runtime, terminal, files))
    expected = restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
    pins = warm["actual_pins"]
    anchor_i_prefix.validate_typed_receipt(cold, restart, case, pins, expected, runtime)
    records.extend(bind_typed_proof_inventory(cold, runtime, files))
    cold_sample_path = runtime / "target-cold/samples/cold-load.json"
    records.append(bound_parent_file(regular_record(cold_sample_path), runtime, files))
    cold_sample = runner._read_json(cold_sample_path)
    require(
        cold_sample["sampling_preserved_state"] is True
        and cold_sample["actual_pins"] == pins
        and physical_signature(cold_sample["signature"]) == physical_signature(cold["loaded"])
        and remesh.canonical_checks(case, cold_sample["metrics"])
        == result["cold_load_checks"]
        == cold["canonical_checks"],
        "prefix I fresh cold sample/gates/pins/history differs",
    )
    parent_load = runner._read_json(runtime / "parent-load.json")
    initial_expected = admitted["restart"]["roundtrip"]["after"]["serialized_state"][
        "payload_sha256"
    ]
    anchor_i_prefix.validate_typed_receipt(
        parent_load,
        admitted["restart"],
        admitted["parent"],
        admitted["parent_cold"]["actual_pins"],
        initial_expected,
        runtime,
    )
    records.extend(bind_typed_proof_inventory(parent_load, runtime, files))
    initial_sample_path = runtime / "parent-observation/samples/cold-load.json"
    records.append(bound_parent_file(regular_record(initial_sample_path), runtime, files))
    initial_sample = runner._read_json(initial_sample_path)
    require(
        initial_sample["sampling_preserved_state"] is True
        and initial_sample["actual_pins"] == parent_load["actual_pins"]
        and physical_signature(initial_sample["signature"])
        == physical_signature(parent_load["loaded"])
        and all(remesh.canonical_checks(admitted["parent"], initial_sample["metrics"]).values()),
        "prefix I initial fresh-load observation differs or fails original gates",
    )
    return records


def is_prefix_i_producer(parent):
    return Path(parent.get("runner", {}).get("path", "")).name == "run_paper_anchor_i_prefix.py"


def source_contract(parent, component, *, metadata_reference=False, typed_i=False, typed_iii=False):
    """Check exact parent/current method inventories without fixing a parent revision."""
    runner.require_clean_source(component)
    runner.require_clean_source(parent["component"])
    commit = parent["component_commit"]
    require(commit == parent["component"]["commit"], "parent source declarations differ")
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", commit, component["commit"])
    old_sources, current = parent["component"]["sources"], component["sources"]
    require(
        set(old_sources) == set(current) and set(runner.METHOD_FILES) <= set(current),
        "method-file inventories differ",
    )
    for name, record in old_sources.items():
        path = ROOT / relative_path(name)
        old = hashlib.sha256(diagnostic.git(ROOT, "show", f"{commit}:{name}")).hexdigest()
        require(
            record["committed_matches"] is True
            and old
            == record["sha256"]
            == record["committed_sha256"]
            == current[name]["sha256"]
            == sha(path),
            "parent/current method differs: " + name,
        )
    implementation = {}
    exact_iii = (
        Path(parent.get("runner", {}).get("path", "")).name == Path(anchor_iii.ENTRYPOINT).name
    )
    require(exact_iii == typed_iii, "exact III target and producer must be paired")
    if exact_iii:
        require(
            parent.get("anchor_stage") == {"kind": anchor_iii.KIND}
            and parent.get("qualification_schema") == anchor_iii.SCHEMA,
            "exact III producer kind/schema differs",
        )
    prefix_i = is_prefix_i_producer(parent)
    require(not prefix_i or typed_i, "prefix I producer requires the declared typed I request")
    if prefix_i:
        require(
            parent.get("anchor_stage") == {"kind": "stationary_I_candidate_from_prefix"}
            and parent.get("qualification_schema") == prefix_qualification.SCHEMA,
            "prefix I producer kind/schema differs",
        )
    helpers = (
        (*HELPERS, *anchor_iii.NEW_HELPERS, anchor_iii.ENTRYPOINT)
        if exact_iii
        else (*HELPERS, *PREFIX_I_HELPERS)
        if prefix_i
        else (*HELPERS, *I_HELPERS)
        if typed_i
        else HELPERS
    )
    for name in (*helpers, "runners/run_paper_modes.py"):
        path = ROOT / name
        digest = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{component['commit']}:{name}")
        ).hexdigest()
        require(regular_record(path)["sha256"] == digest, "implementation differs from Git")
        implementation[name] = {"path": str(path), "sha256": digest}
        if name in helpers:
            # The older positive reference predates the anchor helper; it is never executed here.
            old = runner._git(ROOT, "show", f"{commit}:{name}")
            require(
                old is not None or metadata_reference and name == "runners/run_paper_anchor.py",
                "parent lacks a supporting helper",
            )
            if old is not None:
                require(hashlib.sha256(old).hexdigest() == digest, "supporting helper changed")
            if name in parent.get("implementation", {}):
                require(
                    parent["implementation"][name]["sha256"] == digest,
                    "parent helper declaration differs",
                )
            if typed_i or typed_iii:
                declaration = parent.get("implementation", {}).get(name, {})
                require(
                    declaration.get("sha256") == digest
                    and Path(declaration.get("path", ""))
                    == Path(parent["runner"]["path"]).parent.parent / name,
                    "I supporting helper is unbound or differs",
                )
    producer = parent["runner"]
    producer_path = Path(producer["path"])
    if typed_i:
        expected = "run_paper_anchor_i_prefix.py" if prefix_i else "run_paper_anchor_i.py"
        require(producer_path.name == expected, "unsupported typed I producer")
    require(
        producer_path.name
        in {"run_paper_anchor.py", "run_pressure_remesh.py", "run_paper_seed_i.py"}
        or producer_path.name.startswith("run_paper_anchor_"),
        "unsupported parent producer",
    )
    producer_name = "runners/" + producer_path.name
    producer_bytes = diagnostic.git(ROOT, "show", f"{commit}:{producer_name}")
    require(
        hashlib.sha256(producer_bytes).hexdigest() == producer["sha256"],
        "parent producer does not match its committed source",
    )
    require(
        parent["runner_path"] == producer["path"] and parent["runner_sha256"] == producer["sha256"],
        "parent runner declarations differ",
    )
    return implementation


def environment_contract(environment, parent_environment):
    for value in (environment, parent_environment):
        runner.require_pinned_environment(value)
        require(
            value["commit"] == diagnostic.PYOOMPH
            and value["native_core"]["sha256"] == diagnostic.CORE,
            "native core/source pin differs",
        )
    require(
        environment["url"] == parent_environment["url"]
        and environment["interpreter"]["resolved_executable"]
        == parent_environment["interpreter"]["resolved_executable"],
        "native environment differs",
    )


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "runner must use its own component")
    for module, name in (
        (model, "problems/paper_bubble.py"),
        (runner, "runners/run_paper_case.py"),
        (diagnostic, HELPERS[0]),
        (remesh, HELPERS[1]),
        (anchor, HELPERS[2]),
        (anchor_i, anchor_i.ENTRYPOINT),
        (anchor_i_prefix, anchor_i_prefix.ENTRYPOINT),
        (anchor_iii, anchor_iii.ENTRYPOINT),
        (prefix_qualification, prefix_qualification.ENTRYPOINT),
        (requalification, requalification.ENTRYPOINT),
        (requalification.cold, "runners/run_paper_cold_history.py"),
    ):
        require(
            Path(module.__file__).resolve() == ROOT / name, "helper imported from another component"
        )
    parent_path = args.manifest.resolve(strict=True)
    require(parent_path.name == "run-manifest.json", "canonical actual parent manifest required")
    parent = json.loads(parent_path.read_text())
    case = model.read_case(args.case)
    if coarse_cold.is_producer(parent):
        require(Path(coarse_cold.__file__).resolve() == ROOT / coarse_cold.ENTRYPOINT,
                "coarse helper imported from another component")
        return coarse_cold.consumer_preflight(args, parent, case)
    request = request_contract(case, args.shifts, args.neigen, metadata_only=args.check)
    require(
        sha(args.case) == parent["case_sha256"] and parent["case_id"] == case["case_id"],
        "parent belongs to a different case",
    )
    require(
        parent["case_fields_used"] == model.CASE_FIELDS_USED
        and parent["azimuthal_stability"] is True,
        "parent case inventory/AZ layout differs",
    )
    parent_case = Path(parent["case_path"])
    require(
        parent["case_file"] == str(parent_case)
        and regular_record(parent_case)["sha256"] == sha(args.case)
        and model.read_case(parent_case) == case,
        "parent case bytes differ",
    )
    component = runner.component_revision(ROOT)
    implementation = source_contract(
        parent,
        component,
        metadata_reference=request["reference_metadata_only"],
        typed_i=request["branch_candidate"] == "I",
        typed_iii=case["physics"]["ca_in"] == anchor_iii.TARGET_CA,
    )
    environment = runner.pyoomph_revision()
    environment_contract(environment, parent["pyoomph"])
    restart = runner.validate_restart(
        args.packet,
        case,
        sha(args.case),
        component=component,
        environment=environment,
        azimuthal_stability=True,
    )
    require(
        Path(restart["parent_manifest"]["path"]) == parent_path
        and Path(restart["state"]["path"]) == args.state.resolve(strict=True),
        "packet binds another manifest/state",
    )
    require(
        restart["stage"]["kind"] == "completed_polish"
        and restart["stage"]["ca_in"] == case["physics"]["ca_in"]
        and restart["stage"]["bond"] == case["physics"]["bond"],
        "qualified terminal packet required",
    )
    summary, cold, records = qualified_parent(
        parent_path.parent, case, parent=parent, restart=restart
    )
    require(
        (parent_path.parent / relative_path(summary["checkpoint"])).resolve(strict=True)
        == args.packet.resolve(strict=True),
        "qualified checkpoint differs from the selected packet",
    )
    require(
        cold["loaded"]["serialized_state"]["payload_sha256"]
        == restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
        "parent cold-loaded history differs from the packet",
    )
    records.extend(
        [
            restart["packet"],
            restart["parent_manifest"],
            restart["state"],
            *restart["meshes"],
            regular_record(args.case),
            regular_record(parent_case),
        ]
    )
    return {
        "case": case,
        "request": request,
        "restart": restart,
        "cold": cold,
        "component": component,
        "implementation": implementation,
        "environment": environment,
        "records": records,
    }


def stable_inputs(prepared):
    remesh.verify_inputs(prepared["records"])
    remesh.verify_source(prepared["component"], prepared["implementation"])


def manifest(args, prepared, argv):
    own = prepared["implementation"]["runners/run_paper_modes.py"]
    return {
        "schema": "pyoomph-runner-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "case_path": str(args.case.resolve()),
        "case_file": str(args.case.resolve()),
        "case_sha256": sha(args.case),
        "case_id": prepared["case"]["case_id"],
        "case_fields_used": model.CASE_FIELDS_USED,
        "case_fields_scope": "qualified stationary case restored in full; one bounded spectral request",
        "units": "nondimensional (sigma=mu_liquid=R0=1)",
        "nondimensionalization": prepared["case"]["nondimensionalization"],
        "argv": [sys.executable, own["path"], *argv],
        "runner": own,
        "runner_path": own["path"],
        "runner_sha256": own["sha256"],
        "component_commit": prepared["component"]["commit"],
        "component": prepared["component"],
        "pyoomph": prepared["environment"],
        "implementation": prepared["implementation"],
        "azimuthal_stability": True,
        "mode_request": prepared["request"],
        "parent_restart": prepared["restart"],
        "parent_load_contract": parent_load_contract(prepared),
        "parent_acceptance_changed": False,
        "qualification_scope": SCOPE,
        "api_source": API,
        "finite_spectrum_complete": False,
        "global_dominance_certified": False,
    }


def array_record(value):
    import numpy as np

    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def full_mode_snapshot_signature(
    snapshot, *, mesh_reference_base, mode, bc_marker, mesh_authority=None
):
    """Bind the whole 0.1.6 stream, including all eigen/continuation/history bytes.

    Only length-prefixed mesh-reference strings are replaced by their admitted
    content identities. Compression flags and every other header/body/trailer
    byte, including the serialized BC label, remain unchanged and hashed.
    Parent meshes outside CWD require the exact restart path/hash authority.
    """
    require(isinstance(snapshot, bytes), "full memory snapshot bytes required")
    require(not isinstance(mode, bool) and mode in MODES, "unsupported snapshot mode")
    require(
        bc_marker in {"init", "stationary", "transient", "eigen"}, "unknown native BC cache marker"
    )
    base = Path(mesh_reference_base).resolve(strict=True)
    require(base.is_dir(), "snapshot mesh base must be a directory")
    authority = {}
    if mesh_authority is not None:
        require(
            isinstance(mesh_authority, (list, tuple)) and mesh_authority, "empty mesh authority"
        )
        for record in mesh_authority:
            verified = regular_record(record["path"])
            require(verified["sha256"] == record["sha256"], "snapshot mesh authority bytes changed")
            authority[verified["path"]] = verified["sha256"]
    parts, cursor, identities = [], 0, []
    entries = runner._state_mesh_entries(io.BytesIO(snapshot))
    require(entries, "mode snapshot lacks referenced meshes")
    for start, end, reference in entries:
        path = base / reference
        verified = regular_record(path)
        resolved = Path(verified["path"])
        if mesh_authority is None:
            require(resolved.is_relative_to(base), "snapshot mesh reference escaped its base")
        else:
            require(
                authority.get(verified["path"]) == verified["sha256"],
                "snapshot mesh reference escaped its bound authority",
            )
        identity = ("sha256:" + verified["sha256"]).encode("ascii")
        parts.extend((snapshot[cursor:start], struct.pack("<q", len(identity)), identity))
        cursor = end
        identities.append(verified["sha256"])
    parts.append(snapshot[cursor:])  # includes full eigen/continuation blocks and the intact footer
    normalised = b"".join(parts)
    return {
        "schema": "pyoomph-full-mode-snapshot-v1",
        "native_format": "0.1.6/global",
        "mode": mode,
        "bc_cache_marker": bc_marker,
        "full_stream_sha256": hashlib.sha256(normalised).hexdigest(),
        "normalised_byte_count": len(normalised),
        "mesh_content_sha256": identities,
        "scope": "complete serialized history/eigen/BC/continuation stream; only admitted mesh names normalised",
    }


def copy_eigendata(problem):
    """Copy serialized values/vectors/m labels and the separately retained k cache."""
    import numpy as np

    result = {}
    for name in EIGENDATA_FIELDS:
        value = getattr(problem, name)
        require(
            value is None or isinstance(value, np.ndarray), "unsupported native eigen cache type"
        )
        if value is not None:
            require(np.isfinite(value).all(), "nonfinite native eigen cache")
        result[name] = None if value is None else value.copy()
    return result


def eigendata_signature(data):
    require(set(data) == set(EIGENDATA_FIELDS), "incomplete saved eigen cache")
    return {
        name: None if data[name] is None else array_record(data[name]) for name in EIGENDATA_FIELDS
    }


def restore_eigendata(problem, saved):
    """Restore exactly the captured cache, including a genuinely empty pre-solve state.

    At a186 the loader reads nonzero eigen blocks but leaves existing arrays
    intact for a zero-count block. The pinned solver assigns all four caches;
    its snapshot writer serializes values/vectors/m labels and omits k, which
    is captured and restored separately here.
    """
    eigendata_signature(saved)
    for name, value in saved.items():
        setattr(problem, name, None if value is None else value.copy())
    require(
        eigendata_signature(copy_eigendata(problem)) == eigendata_signature(saved),
        "saved eigen cache did not restore exactly",
    )


def take_state(problem, *, mesh_authority=None):
    """Capture mode-aware fields, actual pins and the complete native history stream."""
    import numpy as np

    setting = problem.eigen_data_in_states
    flags = {
        name: getattr(problem, name)
        for name in ("_first_step", "_taken_already_an_unsteady_step", "_last_step_was_stationary")
    }
    steady = [problem.time_stepper_pt(i).is_steady() for i in range(problem.ntime_stepper())]
    try:
        problem.eigen_data_in_states = True
        payload = problem._snapshot_state()
    finally:
        problem.eigen_data_in_states = setting
    require(isinstance(payload, bytes), "full memory snapshot required")
    eigendata = copy_eigendata(problem)
    types, names = problem.get_dof_description()
    arrays = {
        "free_values": np.asarray(problem.get_current_dofs()[0]).copy(),
        "positional_mask": np.asarray(problem.get_current_dofs()[1]).copy(),
        "pinned_values": np.asarray(problem.get_current_pinned_values(True)).copy(),
        "interface": np.column_stack(problem.interface_polyline()),
        "dof_types": np.asarray(types, dtype=np.int64).copy(),
    }
    require(all(np.isfinite(value).all() for value in arrays.values()), "nonfinite physical state")
    names = [str(name) for name in names]
    signature = {
        "arrays": {key: array_record(value) for key, value in arrays.items()},
        "dof_names": names,
        "actual_pins": diagnostic.actual_pins(problem),
        "serialized_state": full_mode_snapshot_signature(
            payload,
            mesh_reference_base=Path.cwd(),
            mode=int(problem._azimuthal_mode_param_m.value),
            bc_marker=problem._last_bc_setting,
            mesh_authority=mesh_authority,
        ),
        "last_eigendata": eigendata_signature(eigendata),
        "parameters": {
            name: float(problem.get_global_parameter(name).value) for name in ("Ca_in", "Bo")
        },
        "mode": int(problem._azimuthal_mode_param_m.value),
        "volume": problem.observable("volume", "gas"),
        "frame_speed": problem.frame_speed(),
        "time": float(problem.get_current_time(dimensional=False, as_float=True)),
        "bc_cache_marker": problem._last_bc_setting,
        "steady": steady,
        "flags": flags,
        "eigen_data_in_states": setting,
    }
    return {
        "signature": signature,
        "payload": payload,
        "arrays": arrays,
        "names": names,
        "eigendata": eigendata,
        "mesh_authority": mesh_authority,
        "raw_snapshot_sha256": hashlib.sha256(payload).hexdigest(),
    }


def restore_state(problem, saved):
    """Restore through the native loader and refuse any field, pin or history drift."""
    signature = saved["signature"]
    try:
        problem._azimuthal_mode_param_m.value = signature["mode"]
        if signature["mode"] == 0:
            problem.actions_before_stationary_solve(force_reassign_eqs=True)
        else:
            problem.actions_before_eigen_solve(force_reassign_eqs=True)
        problem.eigen_data_in_states = True
        problem._restore_state(saved["payload"])
        restore_eigendata(problem, saved["eigendata"])
        problem._azimuthal_mode_param_m.value = signature["mode"]
        for name, value in signature["flags"].items():
            setattr(problem, name, value)
        for i, was_steady in enumerate(signature["steady"]):
            stepper = problem.time_stepper_pt(i)
            stepper.make_steady() if was_steady else stepper.undo_make_steady()
        problem.eigen_data_in_states = signature["eigen_data_in_states"]
        problem.invalidate_cached_mesh_data()
        after = (
            take_state(problem)
            if saved.get("mesh_authority") is None
            else take_state(problem, mesh_authority=saved["mesh_authority"])
        )
        if after["signature"] != signature:
            raise RestorationFailure("complete state/pins/layout/forcing/history did not restore")
        return {"verified": True, "before": signature, "after": after["signature"]}
    except RestorationFailure:
        raise
    except Exception as error:
        raise RestorationFailure("native full-state restoration failed") from error


def save_state_receipt(directory, name, saved):
    import numpy as np

    diagnostic.atomic(directory / (name + ".dump"), saved["payload"])
    with (directory / (name + ".npz")).open("xb") as stream:
        np.savez_compressed(
            stream,
            **saved["arrays"],
            dof_names=np.asarray(saved["names"], dtype=str),
            **{name: value for name, value in saved["eigendata"].items() if value is not None},
        )
    write_json(
        directory / (name + ".json"),
        {**saved["signature"], "raw_snapshot_sha256": saved["raw_snapshot_sha256"]},
    )


def validate_layout(types, names, incidence, forced, selectors, n, mode):
    """Use actual equation incidence, including boundary-owned bulk fields."""
    import numpy as np

    types = np.asarray(types, dtype=np.int64)
    require(
        types.shape == (n,) and np.all(types >= 0) and np.all(types < len(names)),
        "incomplete mode DOF descriptions",
    )
    require(mode in MODES, "unsupported mode")
    for indices in incidence.values():
        require(
            all(isinstance(i, int) and 0 <= i < n for i in indices),
            "invalid native equation incidence",
        )
    require(
        all(len(incidence[name]) == 1 for name in GLOBALS),
        "global equation incidence is not unique",
    )
    named = {item for item in selectors if isinstance(item, str)}
    global_rows = {incidence[name][0] for name in GLOBALS}
    require(
        (global_rows <= set(forced) and set(GLOBALS) <= named)
        if mode
        else (not global_rows & set(forced) and not set(GLOBALS) & named),
        "mode-dependent pressure/frame constraints differ",
    )
    for phase in ("liquid", "gas"):
        require(incidence[phase + "/pressure"], "bulk pressure equation incidence is empty")
        require(
            set(incidence[phase + "/pressure"]) - set(forced), "bulk pressure removed by forcing"
        )
        if mode:
            require(
                incidence[phase + "/velocity_phi"], "phase azimuthal velocity incidence is empty"
            )
    require(incidence["liquid/interface/_kin_bc"], "free-surface kinematic incidence is empty")
    if mode:
        require(
            incidence["liquid/interface/_lagr_velo_connvelocity_phi"],
            "interface azimuthal velocity connection is absent",
        )
    require(all(0 <= i < n for i in forced), "forced equation is outside the trial layout")
    return {
        "types": types,
        "names": names,
        "incidence": incidence,
        "forced": sorted(forced),
        "selectors": sorted(selectors, key=str),
        "globals": {name: incidence[name] for name in GLOBALS},
        "field_counts": {name: len(indices) for name, indices in incidence.items()},
    }


def mode_layout(problem, n, mode):
    from pyoomph.solvers.generic import EigenMatrixManipulatorBase

    resolver = EigenMatrixManipulatorBase(problem)
    fields = [*GLOBALS, "liquid/interface/_kin_bc", "liquid/interface/_lagr_velo_connvelocity_phi"]
    for phase in ("liquid", "gas"):
        fields.extend(
            phase + "/" + field
            for field in (
                "pressure",
                "velocity_x",
                "velocity_y",
                "velocity_phi",
                "mesh_x",
                "mesh_y",
            )
        )
    incidence = {
        name: sorted(int(i) for i in resolver.resolve_equations_by_name(name)) for name in fields
    }
    selectors = problem.setup_forced_zero_dof_list_for_eigenproblems()
    forced = sorted(int(i) for i in problem.dof_strings_to_global_equations(selectors))
    types, names = problem.get_dof_description()
    return validate_layout(
        types, [str(name) for name in names], incidence, forced, selectors, n, mode
    )


def verify_pencil(jacobian, mass, layout):
    import numpy as np

    require(jacobian.shape == mass.shape and mass.shape[0] == mass.shape[1], "nonsquare pencil")
    require(np.isfinite(jacobian.data).all() and np.isfinite(mass.data).all(), "nonfinite pencil")
    j, m = jacobian.tocsr(copy=True), mass.tocsr(copy=True)
    j.eliminate_zeros()
    m.eliminate_zeros()
    rows = np.flatnonzero(np.diff(m.indptr) > 0)
    require(len(rows), "empty mass support")
    for index in layout["forced"]:
        row = j.getrow(index)
        require(
            row.nnz == 1
            and int(row.indices[0]) == index
            and row.data[0] == 1
            and m.getrow(index).nnz == 0,
            "actual forced row is not identity-J/zero-M",
        )
    for name in GLOBALS:
        require(
            m.getrow(layout["globals"][name][0]).nnz == 0,
            "global constraint carries unexpected mass",
        )
    return {
        "n": mass.shape[0],
        "J_nnz": j.nnz,
        "M_nnz": m.nnz,
        "mass_rows": rows.tolist(),
        "mass_columns": np.unique(m.indices).tolist(),
        "forced_rows_verified": True,
        "mass_support_exact_zeros_only": True,
        "finite_spectrum_complete": False,
    }


def validate_returned(values, vectors, labels, n, mode):
    import numpy as np

    require(
        values.shape == (NEIGEN,)
        and vectors.shape == (NEIGEN, n)
        and labels.shape == (NEIGEN,)
        and np.all(labels == mode),
        "returned count, labels or trial layout differs",
    )


def scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def diagnose_pair(value, vector, jacobian, mass, layout, factor):
    """Retain diagnostics even when a returned root fails physical qualification."""
    import numpy as np

    result = {
        "lambda_real": scalar(value.real),
        "lambda_imag": scalar(value.imag),
        "paper_growth": scalar(factor * value.real),
        "paper_frequency": scalar(-factor * value.imag),
        "rejection_reasons": [],
        "pencil_qualified": False,
    }
    if not np.isfinite(value) or not np.isfinite(vector).all():
        result["rejection_reasons"].append("nonfinite returned eigenvalue/vector")
        return result
    vnorm = float(np.linalg.norm(vector))
    jnorm, mnorm = float(np.linalg.norm(jacobian.data)), float(np.linalg.norm(mass.data))
    mass_scale = float(np.max(abs(mass.data), initial=0))
    jv, mv = jacobian @ vector, mass @ vector
    residual = float(np.linalg.norm(jv - value * mv))
    backward = residual / max((jnorm + abs(value) * mnorm) * vnorm, 1e-300)
    mass_action = float(np.linalg.norm(mv)) / max(mass_scale * vnorm, 1e-300)
    forced = float(np.linalg.norm(vector[layout["forced"]])) / max(vnorm, 1e-300)
    actions = {}
    for name, indices in layout["globals"].items():
        row = jacobian.getrow(indices[0])
        action = abs((row @ vector)[0])
        actions[name] = {
            "absolute_J_action": float(action),
            "relative_J_action": float(action / max(np.linalg.norm(row.data) * vnorm, 1e-300)),
            "coefficient_abs": float(abs(vector[indices[0]])),
        }
    result.update(
        vector_norm_l2=vnorm,
        absolute_residual_l2=residual,
        backward_residual_l2=backward,
        mass_action_ratio=mass_action,
        forced_coefficient_ratio=forced,
        constraint_actions=actions,
    )
    if vnorm == 0 or not all(math.isfinite(v) for v in (backward, mass_action, forced)):
        result["rejection_reasons"].append("invalid eigenvector/diagnostic")
    if backward > LIMITS["backward"]:
        result["rejection_reasons"].append("original-pencil backward residual")
    if mass_action <= LIMITS["mass_action"]:
        result["rejection_reasons"].append("unresolved zero/near-null mass action")
    if forced > LIMITS["forced_coefficient"]:
        result["rejection_reasons"].append("forced coefficients")
    if any(v["relative_J_action"] > LIMITS["backward"] for v in actions.values()):
        result["rejection_reasons"].append("algebraic constraint action")
    result["pencil_qualified"] = not result["rejection_reasons"]
    return result


def geometry(points):
    """Normals from quadratic-chain derivatives; common nodal 2*pi*r*ds weights."""
    import numpy as np

    require(
        points.ndim == 2
        and points.shape[1] == 2
        and len(points) >= 3
        and len(points) % 2 == 1
        and np.isfinite(points).all(),
        "invalid quadratic interface chain",
    )
    require(
        abs(points[0, 1]) <= 1e-7
        and abs(points[-1, 1]) <= 1e-7
        and points[0, 0] < points[-1, 0]
        and np.min(points[:, 1]) >= -1e-10,
        "interface must run rear-to-front between axis tips",
    )
    tangents = np.zeros_like(points)
    for first in range(0, len(points) - 2, 2):
        a, mid, b = points[first : first + 3]
        tangents[first : first + 3] += np.asarray(
            [-1.5 * a + 2 * mid - 0.5 * b, -0.5 * a + 0.5 * b, 0.5 * a - 2 * mid + 1.5 * b]
        )
    norms = np.linalg.norm(tangents, axis=1)
    require(np.all(norms > 1e-12), "degenerate quadratic-chain tangent")
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) / norms[:, None]
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    require(np.all(ds > 0), "duplicate interface node")
    weights = np.zeros(len(points))
    weights[:-1] += math.pi * np.maximum(points[:-1, 1], 0) * ds
    weights[1:] += math.pi * np.maximum(points[1:, 1], 0) * ds
    require(np.any(weights > 0), "zero surface measure")
    return np.r_[0.0, np.cumsum(ds)], normals, weights


def normal_samples(problem, vector, mode, *, mesh_authority=None):
    import numpy as np

    saved = (
        take_state(problem)
        if mesh_authority is None
        else take_state(problem, mesh_authority=mesh_authority)
    )
    base = saved["arrays"]["interface"]
    epsilon = LIMITS["epsilon"]
    motions = []
    for step in (epsilon, epsilon / 2):
        parts = []
        for part in (vector.real, vector.imag):
            try:
                problem.perturb_dofs(step * part)
                points = np.column_stack(problem.interface_polyline())
                require(
                    points.shape == base.shape and np.isfinite(points).all(),
                    "probe changed interface chain",
                )
                parts.append((points - base) / step)
            finally:
                restore_state(problem, saved)
        motions.append(parts[0] + 1j * parts[1])
    arclength, normals, weights = geometry(base)
    eta, half = (np.sum(motion * normals, axis=1) for motion in motions)
    norm = math.sqrt(float(np.sum(weights * abs(eta) ** 2)))
    require(math.isfinite(norm) and norm > 1e-14, "no resolved physical interface-normal action")
    halving = math.sqrt(float(np.sum(weights * abs(eta - half) ** 2))) / norm
    require(halving <= LIMITS["halving"], "normal epsilon-halving gate")
    eta /= norm
    columns = np.column_stack((eta.real, eta.imag)) * np.sqrt(weights)[:, None]
    _, singular, _ = np.linalg.svd(columns, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * LIMITS["normal_rank_relative"]))
    display = eta / float(np.max(abs(eta)))
    fraction, energy = arclength / arclength[-1], weights * abs(eta) ** 2
    translation = abs(np.sum(weights * np.conjugate(eta) * normals[:, 1])) / math.sqrt(
        float(np.sum(weights * normals[:, 1] ** 2))
    )
    return {
        "mode": mode,
        "epsilon": epsilon,
        "halving_relative_surface_l2": halving,
        "raw_normal_surface_l2": norm,
        "real_surface_span_dimension": rank,
        "real_surface_span_singular_values": singular.tolist(),
        "normalization": "unit 2*pi*r*ds surface L2; arbitrary eigenvector phase",
        "fourier_convention": f"Re[eta(s)*exp(lambda*t+i*{mode}*phi)]",
        "display_scale": "maximum complex normal amplitude is one; linear illustration only",
        "normal_method": "ordered quadratic-chain tangent normals; nodal trapezoidal surface weights",
        "front_cap_energy_fraction": float(np.sum(energy[fraction >= 0.8])),
        "rear_cap_energy_fraction": float(np.sum(energy[fraction <= 0.2])),
        "transverse_translation_overlap": float(translation),
        "state_restoration_verified": True,
        "samples": [
            {
                "s": float(s),
                "z": float(p[0]),
                "r": float(p[1]),
                "normal_z": float(n[0]),
                "normal_r": float(n[1]),
                "weight_surface": float(w),
                "eta_real": float(e.real),
                "eta_imag": float(e.imag),
                "display_eta_real": float(d.real),
                "display_eta_imag": float(d.imag),
            }
            for s, p, n, w, e, d in zip(
                arclength, base, normals, weights, eta, display, strict=True
            )
        ],
    }


def compare_normals(first, second, mode):
    import numpy as np

    a, b = first["samples"], second["samples"]
    require(
        len(a) == len(b)
        and all(
            x["z"] == y["z"] and x["r"] == y["r"] and x["weight_surface"] == y["weight_surface"]
            for x, y in zip(a, b, strict=True)
        ),
        "cross-shift base geometry differs",
    )
    weights = np.sqrt([row["weight_surface"] for row in a])
    eta = [
        np.asarray([complex(row["eta_real"], row["eta_imag"]) for row in group]) for group in (a, b)
    ]
    if mode == 1:
        overlap = float(abs(np.vdot(weights * eta[0], weights * eta[1])))
        return {"passed": overlap >= LIMITS["normal_overlap"], "complex_normal_overlap": overlap}
    bases, dimensions = [], []
    for value in eta:
        columns = np.column_stack((value.real, value.imag)) * weights[:, None]
        u, singular, _ = np.linalg.svd(columns, full_matrices=False)
        keep = singular > singular[0] * LIMITS["normal_rank_relative"]
        bases.append(u[:, keep])
        dimensions.append(int(sum(keep)))
    cosines = np.linalg.svd(bases[0].T @ bases[1], compute_uv=False)
    return {
        "passed": dimensions[0] == dimensions[1]
        and bool(np.all(cosines >= LIMITS["normal_overlap"])),
        "real_span_dimensions": dimensions,
        "principal_cosines": cosines.tolist(),
        "scope": "physical surface span only; no native invariant-subspace identity",
    }


def returned_families(pairs, mode):
    """Retain every possible conjugate relation, without choosing a unique family."""
    require(mode in MODES, "unsupported family sector")
    qualified = [p for p in pairs if p["physical_qualified"]]
    families = []
    for pair in qualified:
        value = complex(pair["lambda_real"], pair["lambda_imag"])
        real = abs(value.imag) <= LIMITS["two_shift_complex"]
        partners = (
            [
                p["index"]
                for p in qualified
                if p["index"] != pair["index"]
                and abs(complex(p["lambda_real"], p["lambda_imag"]) - value.conjugate())
                <= LIMITS["two_shift_complex"]
            ]
            if not real
            else []
        )
        families.append(
            {
                "representative": pair["index"],
                "kind": "numerically_real" if real else "oscillatory",
                "possible_conjugate_partners": partners,
                "conjugate_complete_returned": real or bool(partners),
                "conjugate_requirement_satisfied": mode == 1 or real or bool(partners),
                "conjugate_requirement": "within-sector returned partner for oscillatory m=0"
                if mode == 0
                else "no within-sector conjugate requirement for m=+1",
                "negative_sector": {"mode": -1, "searched": False, "conjugacy_verified": False},
                "paper_growth": pair["paper_growth"],
                "unique_identity_certified": False,
            }
        )
    return families


def compare_searches(first, second, mode):
    physical = [
        [p for p in search["pairs"] if p["physical_qualified"]] for search in (first, second)
    ]
    pairings = []
    for a in physical[0]:
        for b in physical[1]:
            distance = abs(
                complex(a["lambda_real"], a["lambda_imag"])
                - complex(b["lambda_real"], b["lambda_imag"])
            )
            if distance <= LIMITS["two_shift_complex"]:
                normal = compare_normals(a["normal"], b["normal"], mode)
                if normal["passed"]:
                    pairings.append(
                        {
                            "left": a["index"],
                            "right": b["index"],
                            "complex_rate_difference": distance,
                            "normal_comparison": normal,
                        }
                    )
    leaders = [
        [p for p in group if p["paper_growth"] == max(q["paper_growth"] for q in group)]
        if group
        else []
        for group in physical
    ]
    eligible = bool(
        leaders[0]
        and leaders[1]
        and any(
            match["left"] in {p["index"] for p in leaders[0]}
            and match["right"] in {p["index"] for p in leaders[1]}
            for match in pairings
        )
    )
    unknown = []
    for side, search in enumerate((first, second)):
        top = leaders[side][0]["paper_growth"] if leaders[side] else None
        for pair in search["pairs"]:
            if not pair["physical_qualified"] and (
                pair["paper_growth"] is None or top is None or pair["paper_growth"] >= top
            ):
                unknown.append(
                    {
                        "shift": search["shift"],
                        "index": pair["index"],
                        "paper_growth": pair["paper_growth"],
                        "reasons": pair["rejection_reasons"],
                    }
                )
    complete_conjugates = all(
        any(
            f["representative"] == p["index"] and f["conjugate_complete_returned"]
            for f in returned_families(physical[side], mode)
        )
        for side, group in enumerate(leaders)
        for p in group
    )
    conjugate_requirement_satisfied = all(
        any(
            f["representative"] == p["index"] and f["conjugate_requirement_satisfied"]
            for f in returned_families(physical[side], mode)
        )
        for side, group in enumerate(leaders)
        for p in group
    )
    return {
        "mode": mode,
        "admissible_pairings": pairings,
        "returned_envelope": [
            {
                "shift": search["shift"],
                "paper_growth": group[0]["paper_growth"] if group else None,
                "candidate_indices": [p["index"] for p in group],
            }
            for search, group in zip((first, second), leaders, strict=True)
        ],
        "unqualified_higher_or_unresolved_returns": unknown,
        "matched_returned_envelope": eligible and not unknown and conjugate_requirement_satisfied,
        "returned_envelope_conjugate_complete": complete_conjugates,
        "conjugate_requirement_satisfied": conjugate_requirement_satisfied,
        "negative_sector": {"mode": -1, "searched": False, "conjugacy_verified": False},
        "finite_spectrum_complete": False,
        "global_dominance_certified": False,
        "unique_family_identity_certified": False,
        "scope": SCOPE,
    }


def parent_load_contract(prepared):
    if prepared["case"]["physics"]["ca_in"] == anchor_iii.TARGET_CA:
        return {
            "kind": "preserved-stationary-III-history",
            "module": anchor_iii.ENTRYPOINT,
            "callable": "restore_checked",
            "sha256": prepared["implementation"][anchor_iii.ENTRYPOINT]["sha256"],
            "packet": prepared["restart"]["packet"],
            "payload_sha256": prepared["cold"]["loaded"]["serialized_state"]["payload_sha256"],
        }
    typed = prepared["case"]["physics"]["ca_in"] == anchor_i.TARGET_CA
    module = requalification.ENTRYPOINT if typed else "runners/run_paper_anchor.py"
    return {
        "kind": "typed-I-stationary-descendant" if typed else "plain-anchor",
        "module": module,
        "callable": "restore_checked",
        "sha256": prepared["implementation"][module]["sha256"],
        "packet": prepared["restart"]["packet"],
        "payload_sha256": prepared["cold"]["loaded"]["serialized_state"]["payload_sha256"],
    }


def restore_parent(problem, prepared, output):
    case, restart, cold = prepared["case"], prepared["restart"], prepared["cold"]
    restore = anchor.restore_checked
    if case["physics"]["ca_in"] == anchor_iii.TARGET_CA:
        problem._stationary_iii_receipts_directory = output / "stationary-loads"
        restore = anchor_iii.restore_checked
    if case["physics"]["ca_in"] == anchor_i.TARGET_CA:
        problem._seed_requalification_receipts_directory = output / "typed-loads"
        restore = requalification.restore_checked
    return restore(
        problem,
        case,
        restart,
        cold["actual_pins"],
        cold["loaded"]["serialized_state"]["payload_sha256"],
    )


def spectral_stage(prepared, output, mode, shift):
    import numpy as np
    from scipy import sparse

    case = prepared["case"]
    output.mkdir(exist_ok=False)
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as problem:
        problem.set_output_directory(str(output / "solver"))
        problem.set_linear_solver("superlu")
        problem.set_c_compiler("system")
        problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        problem.set_eigensolver("spectra")
        stable_inputs(prepared)
        problem.initialise()
        require(not problem.is_distributed(), "serial full-pencil search required")
        loaded = restore_parent(problem, prepared, output)
        metrics = remesh.sample(problem, model, case, output, "cold-load", np)["metrics"]
        require(
            all(remesh.canonical_checks(case, metrics).values()), "cold-loaded base gates failed"
        )
        mesh_authority = prepared["restart"]["meshes"]
        base = take_state(problem, mesh_authority=mesh_authority)
        save_state_receipt(output, "base-before", base)
        write_json(output / "parent-load.json", loaded)
        problem._azimuthal_mode_param_m.value = mode
        solver = problem.get_eigen_solver()
        require(getattr(solver, "idname", None) == "spectra", "actual eigensolver differs")
        solver.use_pardiso = False
        original = solver.solve
        captured = []

        def retain_pencil(*args, **kwargs):
            result = original(*args, **kwargs)
            require(len(result) == 4, "pinned Spectra matrix return contract differs")
            values, vectors, j, m = result
            layout = mode_layout(problem, j.shape[0], mode)
            captured.append(
                (
                    np.asarray(values).copy(),
                    np.asarray(vectors).copy(),
                    j.tocsr(copy=True),
                    m.tocsr(copy=True),
                    layout,
                )
            )
            return result

        solver.solve = retain_pencil
        try:
            values, vectors = problem.solve_eigenproblem(
                NEIGEN, shift=shift, azimuthal_m=mode, quiet=True
            )
            values, vectors = np.asarray(values).copy(), np.asarray(vectors).copy()
            labels = np.asarray(problem.get_last_eigenmodes_m()).copy()
        finally:
            solver.solve = original
        require(len(captured) == 1, "one original pencil per scalar-mode solve required")
        raw_values, raw_vectors, j, m, layout = captured[0]
        sparse.save_npz(output / "J.npz", j)
        sparse.save_npz(output / "M.npz", m)
        with (output / "eigenpairs.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                values=values,
                vectors=vectors,
                modes=labels,
                original_values=raw_values,
                original_vectors=raw_vectors,
                types=layout["types"],
                names=np.asarray(layout["names"], dtype=str),
                forced=np.asarray(layout["forced"], dtype=np.int64),
            )
        validate_returned(values, vectors, labels, j.shape[0], mode)
        require(
            np.array_equal(values, raw_values, equal_nan=True)
            and np.array_equal(vectors, raw_vectors, equal_nan=True),
            "returned coefficients differ from the actual Spectra pencil solve",
        )
        now = mode_layout(problem, j.shape[0], mode)
        require(
            np.array_equal(layout["types"], now["types"])
            and layout["names"] == now["names"]
            and layout["incidence"] == now["incidence"]
            and layout["forced"] == now["forced"],
            "native trial layout changed after original-pencil capture",
        )
        support = verify_pencil(j, m, layout)
        write_json(
            output / "layout.json",
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in layout.items()},
        )
        mode_state = take_state(problem, mesh_authority=mesh_authority)
        save_state_receipt(output, "mode-state", mode_state)
        pairs = []
        try:
            for index, (value, vector) in enumerate(zip(values, vectors, strict=True)):
                record = diagnose_pair(
                    value, vector, j, m, layout, prepared["request"]["paper_time_factor"]
                )
                record.update(index=index, mode=mode, physical_qualified=False)
                if record["pencil_qualified"]:
                    try:
                        normal = normal_samples(
                            problem, vector, mode, mesh_authority=mesh_authority
                        )
                        record.update(
                            normal=normal,
                            geometric_normal_action=normal["raw_normal_surface_l2"],
                            physical_qualified=True,
                            normal_file=f"normal-{index:02d}.json",
                        )
                        write_json(output / record["normal_file"], normal)
                    except ValueError as error:
                        record["rejection_reasons"].append(str(error))
                pairs.append(record)
                write_json(output / f"pair-{index:02d}.json", record)
        finally:
            restored = restore_state(problem, base)
            write_json(output / "base-restoration.json", restored)
            save_state_receipt(
                output, "base-after", take_state(problem, mesh_authority=mesh_authority)
            )
        stage = {
            "mode": mode,
            "shift": shift,
            "pairs": pairs,
            "families": returned_families(pairs, mode),
            "support": support,
            "pencil_origin": "matrices returned by the actual Spectra.solve call",
            "base_restoration_verified": True,
            "backend": {
                "id": solver.idname,
                "use_pardiso": solver.use_pardiso,
                "infinite_eigenvalue_cutoff": solver.infinite_eigenvalue_cutoff,
                "tol": solver.tol,
                "maxit": solver.maxit,
                "max_retries": solver.max_retries,
            },
            "backend_limit": "native backend filters Ritz values; unreturned roots are unobserved",
            "finite_spectrum_complete": False,
            "global_dominance_certified": False,
        }
        stable_inputs(prepared)
        write_json(output / "status.json", stage)
        return stage


def plot_returned(searches, case, output):
    """Plot finite returned rates and their physical envelopes at this one Ca."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
    paths = [output / "returned-rates.pdf", output / "returned-rates.png"]
    require(all(not path.exists() for path in paths), "returned-rate plot already exists")
    with matplotlib.rc_context(settings):
        selected = tuple(searches)
        require(selected in {(0,), MODES}, "unsupported returned-mode plot request")
        figure, panels = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4.5),
                                     constrained_layout=True, squeeze=False)
        axes = panels.ravel()
        for mode, axis in zip(selected, axes, strict=True):
            for shift_index, search in enumerate(searches[mode]):
                physical = []
                for pair in search["pairs"]:
                    rate = pair["paper_growth"]
                    if rate is None:
                        continue
                    qualified = pair["physical_qualified"]
                    oscillatory = abs(pair["lambda_imag"] or 0) > LIMITS["two_shift_complex"]
                    marker = "s" if oscillatory else "o"
                    axis.scatter(
                        shift_index,
                        rate,
                        marker=marker if qualified else "x",
                        color="C0" if qualified else "0.5",
                        s=42,
                    )
                    if qualified:
                        physical.append(rate)
                if physical:
                    axis.scatter(
                        shift_index,
                        max(physical),
                        marker="D",
                        facecolors="none",
                        edgecolors="black",
                        s=110,
                    )
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_title(f"$m={mode}$", fontsize=16)
            axis.set_xticks([0, 1], [r"$-10^{-3}$", r"$+10^{-3}$"])
            axis.set_xlabel(r"$\sigma$", fontsize=16)
            axis.set_ylabel(r"$\widehat\omega_i$", fontsize=16)
            axis.tick_params(direction="in", labelsize=12, top=True, right=True)
            axis.set_xlim(-0.35, 1.35)
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker="o", linestyle="none", color="C0", label="Physical, real"),
            Line2D([], [], marker="s", linestyle="none", color="C0", label="Physical, oscillatory"),
            Line2D([], [], marker="x", linestyle="none", color="0.5", label="Unqualified"),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markerfacecolor="none",
                color="black",
                label="Physical returned envelope",
            ),
        ]
        axes[0].legend(handles=handles, fontsize=9, loc="best")
        figure.suptitle(
            rf"$Ca_l={case['physics']['ca_in']:.8g},\quad Bo={case['physics']['bond']:.4g}$",
            fontsize=16,
        )
        try:
            for path in paths:
                figure.savefig(path, bbox_inches="tight", dpi=300)
        finally:
            plt.close(figure)
    return {
        "files": [regular_record(path) for path in paths],
        "scope": "one anchor, two searched shifts; no curve interpolation or complete spectrum",
        "numerical_uncertainty": "backward checks and two-shift differences in result.json; no continuum error bar",
    }


def selected_modes(prepared):
    request = prepared["request"]
    coarse = request.get("schema") == coarse_cold.REQUEST_SCHEMA
    expected = request_contract(prepared["case"], SHIFTS, NEIGEN, coarse_iii=coarse)
    require(request == expected, "execution request differs from the validated bounded subtype")
    if coarse:
        require(prepared.get("execution_authorized") is True
                and prepared.get("restart", {}).get("schema") == coarse_cold.PACKET_SCHEMA,
                "coarse execution requires an admitted cold packet and execution seal")
    return tuple(expected["modes"])


def execute(prepared, output):
    selected = selected_modes(prepared)
    result = {
        "schema": "paper-returned-modes-result-v1",
        "request": prepared["request"],
        "mode_comparisons": [],
        "scope": SCOPE,
        "finite_spectrum_complete": False,
        "global_dominance_certified": False,
        "parent_acceptance_changed": False,
    }
    all_searches = {}
    for mode in selected:
        searches = []
        for index, shift in enumerate(SHIFTS):
            runner.append_progress(
                output,
                {
                    "event": "search-started",
                    "mode": mode,
                    "shift": shift,
                    "requested_pairs": NEIGEN,
                },
            )
            searches.append(
                spectral_stage(prepared, output / f"m{mode}-shift-{index}", mode, shift)
            )
        all_searches[mode] = searches
        result["mode_comparisons"].append(compare_searches(*searches, mode))
    result["numerical_acceptance"] = all(
        row["matched_returned_envelope"] for row in result["mode_comparisons"]
    )
    result["scientific_status"] = (
        "qualified_returned_envelopes_at_one_anchor"
        if result["numerical_acceptance"]
        else "inconclusive_returned_envelopes"
    )
    write_json(output / "result.json", result)
    write_json(
        output / "summary.json",
        {
            "status": "completed",
            "completion_reached": True,
            "scientific_status": result["scientific_status"],
            "numerical_acceptance": result["numerical_acceptance"],
            "finite_spectrum_complete": False,
            "global_dominance_certified": False,
            "scope": SCOPE,
        },
    )
    scientific_files = [regular_record(output / name) for name in ("result.json", "summary.json")]
    try:
        plot = {
            "schema": "paper-returned-modes-plot-v1",
            "status": "complete",
            "scientific_results": scientific_files,
            **plot_returned(all_searches, prepared["case"], output),
        }
    except Exception as error:  # noqa: BLE001 - renderer failures cannot erase completed searches
        plot = {
            "schema": "paper-returned-modes-plot-v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "scientific_results": scientific_files,
            "scientific_results_preserved": True,
            "native_replay_required": False,
        }
        runner.append_progress(
            output,
            {
                "event": "plot-failed",
                "error_type": plot["error_type"],
                "error": plot["error"],
                "scientific_results_preserved": True,
            },
        )
        print(
            f"Plot failed; scientific results retained: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
    require(diagnostic.parents_unchanged(scientific_files), "plotting changed scientific results")
    write_json(output / "plot-receipt.json", plot)
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shifts", nargs=2, type=float, required=True)
    parser.add_argument("--neigen", type=int, default=NEIGEN)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    require(not args.out.is_symlink(), "output must not be a symlink")
    output = runner.validate_output_path(args.out, ROOT)
    require(
        not output.is_relative_to(Path(prepared["restart"]["parent_manifest"]["path"]).parent),
        "output overlaps the parent runtime",
    )
    if args.check:
        print(
            json.dumps({"verified": True, "native_imported": False, "request": prepared["request"],
                        "native_execution_authorized": False,
                        "source_seals_pending": prepared.get("source_seals_pending", []),
                        "required_seal": prepared.get("required_seal")})
        )
        return 0
    actual = manifest(args, prepared, arguments)
    if prepared["request"]["schema"] == coarse_cold.REQUEST_SCHEMA:
        actual["execution_seal"] = regular_record(args.seal)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if "execution_seal" in actual:
        diagnostic.atomic(output / "input-seal.json", Path(actual["execution_seal"]["path"]).read_bytes())
    write_json(output / "run-manifest.json", actual)
    runner.append_progress(
        output,
        {"event": "prepared", "case_id": prepared["case"]["case_id"], "runner": actual["runner"]},
        first=True,
    )
    write_json(
        output / "preparation.json",
        {"inputs": prepared["records"], "request": prepared["request"], "api_source": API},
    )
    try:
        result = execute(prepared, output)
        stable_inputs(prepared)
        files = {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and "solver" not in path.relative_to(output).parts
            and path.name != "progress.jsonl"
        }
        write_json(
            output / "completion.json",
            {
                "status": "complete",
                "files": files,
                "numerical_acceptance": result["numerical_acceptance"],
                "parent_acceptance_changed": False,
                "scope": SCOPE,
            },
        )
        runner.append_progress(
            output,
            {
                "event": "final",
                "scientific_status": result["scientific_status"],
                "numerical_acceptance": result["numerical_acceptance"],
            },
        )
        return 0
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "parents_unchanged": diagnostic.parents_unchanged(prepared["records"]),
            "numerical_acceptance": False,
            "parent_acceptance_changed": False,
            "scope": SCOPE,
        }
        write_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", **failure})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
