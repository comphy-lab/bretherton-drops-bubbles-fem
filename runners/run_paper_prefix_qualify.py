"""Qualify one retained stationary prefix with two fresh, zero-solve loads.

The original attempt and its provisional stages remain unaccepted. Qualification
certifies the saved stationary state and its full storage history only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_anchor as anchor
from runners import run_paper_seed_requalify as requalification

model, runner, diagnostic, remesh = anchor.model, anchor.runner, anchor.diagnostic, anchor.remesh
cold = requalification.cold
require, sha, write_json = diagnostic.require, diagnostic.sha, diagnostic.write_json
SCHEMA = "stationary-continuation-prefix-qualification-v1"
ENTRYPOINT = "runners/run_paper_prefix_qualify.py"
CONTINUATION = "runners/run_paper_anchor_i_prefix.py"
ORIGINAL_SOURCE = "1ffa038002499337c768b326da8b6b10b5f16267"
ORIGINAL_PRODUCER = "runners/run_paper_anchor_i.py"
ORIGINAL_VALUES = [-0.01, -0.005, -0.003, -0.002, -0.0016]
SELECTED_INDEX = 3
TARGET_CA = -0.0016
GATES = anchor.GATES
SEMANTIC_SHA = {
    "problem.cc": "1d36032bc5131452e5c74aaa4155b021e3e45f89448495e31504015bf62252ee",
    "logging.cpp": "1971725114788026794c34d1ea5d8342a217f36ac189075e6e42715d673395c8",
    "logging.py": "8d7598c2da8aaa30c968b151aa6aeac87d9c4f7b062103098d9c81ba101d3c6a",
    "paper_bubble.py": "d3e73c4c28d700ccd7e2196a430c11e9f72c14a873279b48599f43482988343c",
    "problem.py": "b75d6a810daddb3e8f0853c60fea780b33220c6c09cf819e5f219f5af056018f",
}
HELPERS = (
    "runners/run_paper_anchor.py",
    ORIGINAL_PRODUCER,
    "runners/run_paper_seed_i.py",
    "runners/run_paper_seed_recovery.py",
    "runners/run_paper_cold_history.py",
    "runners/run_paper_seed_requalify.py",
    "runners/run_pressure_diagnostic.py",
    "runners/run_pressure_remesh.py",
)
NO_CLAIMS = {
    "branch_identity_verified": False,
    "eigenanalysis_performed": False,
    "jacobian_certificate": False,
    "dynamic_restart_certificate": False,
    "full_comparison_verified": False,
}
RESULT_FIELDS = {
    "schema",
    "numerical_acceptance",
    "stationary_prefix_accepted",
    "anchor_accepted",
    "checks",
    "cold_load_checks",
    "cold_load_verified",
    "checkpoint",
    "lineage",
    "original_attempt_status",
    "original_acceptance_changed",
    *NO_CLAIMS,
}


def regular_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular immutable file required")
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def relative_path(value):
    require(isinstance(value, str), "relative path must be a string")
    path = Path(value)
    require(
        value and not path.is_absolute() and ".." not in path.parts and path != Path("."),
        "unsafe relative path",
    )
    return path


def bound_record(record):
    require(
        isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")),
        "invalid bound file record",
    )
    actual = regular_record(record["path"])
    require(actual == {key: record[key] for key in ("path", "sha256")}, "bound bytes/path differ")
    require(
        "bytes" not in record or Path(actual["path"]).stat().st_size == record["bytes"],
        "bound file size differs",
    )
    return actual


def unique_records(records):
    result = {}
    for item in records:
        record = bound_record(item)
        require(
            record["path"] not in result or result[record["path"]] == record,
            "conflicting immutable file binding",
        )
        result[record["path"]] = record
    return [result[key] for key in sorted(result)]


def portable_signature(value):
    return {
        **value,
        "serialized_state": {
            key: item for key, item in value["serialized_state"].items() if key != "sha256"
        },
    }


def source_contract(parent, entrypoint):
    """Require the original methods and helpers byte-identical to the published source."""
    component = runner.component_revision(ROOT)
    runner.require_clean_source(component)
    runner.require_clean_source(parent["component"])
    require(
        parent["component_commit"] == parent["component"]["commit"] == ORIGINAL_SOURCE,
        "unexpected original I source",
    )
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", ORIGINAL_SOURCE, component["commit"])
    old_sources = parent["component"]["sources"]
    require(set(old_sources) == set(component["sources"]), "method inventory changed")
    paths = set(old_sources) | set(HELPERS) | {ENTRYPOINT, entrypoint}
    implementation = {}
    for relative in sorted(paths):
        path = ROOT / relative_path(relative)
        current = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{component['commit']}:{relative}")
        ).hexdigest()
        require(regular_record(path)["sha256"] == current, "implementation differs from commit")
        if relative not in {ENTRYPOINT, CONTINUATION}:
            old = hashlib.sha256(
                diagnostic.git(ROOT, "show", f"{ORIGINAL_SOURCE}:{relative}")
            ).hexdigest()
            require(current == old, f"frozen method/helper changed: {relative}")
            declaration = old_sources.get(relative) or parent.get("implementation", {}).get(
                relative
            )
            require(
                isinstance(declaration, dict) and declaration.get("sha256") == old,
                "original method/helper declaration differs",
            )
            if relative in old_sources:
                require(
                    declaration["committed_matches"] is True
                    and declaration["committed_sha256"] == old,
                    "original method was not committed",
                )
        implementation[relative] = {"path": str(path), "sha256": current}
    for module, relative in (
        (model, "problems/paper_bubble.py"),
        (runner, "runners/run_paper_case.py"),
        (diagnostic, "runners/run_pressure_diagnostic.py"),
        (remesh, "runners/run_pressure_remesh.py"),
        (anchor, "runners/run_paper_anchor.py"),
        (cold, "runners/run_paper_cold_history.py"),
        (requalification, "runners/run_paper_seed_requalify.py"),
    ):
        require(Path(module.__file__).resolve() == ROOT / relative, "helper imported elsewhere")
    environment = runner.pyoomph_revision()
    cold.seed.validate_environment(environment)
    cold.seed.validate_environment(parent["pyoomph"])
    require(
        environment["commit"] == parent["pyoomph"]["commit"] == diagnostic.PYOOMPH
        and environment["native_core"]["sha256"]
        == parent["pyoomph"]["native_core"]["sha256"]
        == diagnostic.CORE
        and environment["interpreter"]["resolved_executable"]
        == parent["pyoomph"]["interpreter"]["resolved_executable"],
        "original interpreter, source pin or native core differs",
    )
    return component, environment, implementation


def bind_custody(path, run_root):
    receipt = runner._read_json(path)
    require(
        receipt.get("schema") == "figure10-next-phase-archive-v1"
        and receipt.get("simulation") == "paper-model-anchor-i-medium"
        and receipt.get("attempt") == "attempt-0001"
        and receipt.get("source") == str(run_root)
        and receipt.get("source_deleted") is False
        and isinstance(receipt.get("verified_at"), str)
        and receipt.get("protocol_files_verified_at") == receipt["verified_at"],
        "terminal tree custody differs",
    )
    destination = Path(receipt["destination"])
    require(destination.is_absolute() and destination != run_root, "independent archive required")
    files = receipt["files"]
    generated = receipt.get("verifier_generated_files", {})
    require(
        isinstance(files, dict)
        and {"AGENTS.md", "CLAUDE.md", "README.md"} <= set(files)
        and receipt["file_count"] == len(files)
        and receipt["bytes"] == sum(value["bytes"] for value in files.values()),
        "archive inventory/protocol binding differs",
    )
    records = [regular_record(path)]
    # The archive can be on another fleet host. Its root-verified receipt is
    # custody authority; this consumer independently rechecks the retained source.
    for root in (run_root,):
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        permitted = set(files)
        require(actual == permitted, "terminal tree inventory differs")
        for relative in sorted(permitted):
            declaration = files.get(relative) or generated[relative]
            file = root / relative_path(relative)
            item = regular_record(file)
            require(
                file.resolve().is_relative_to(root)
                and item["sha256"] == declaration["sha256"]
                and file.stat().st_size == declaration["bytes"],
                "terminal/archive bytes differ",
            )
            records.append(item)
    return receipt, records


def bind_terminal(path, runtime):
    require(
        path.resolve(strict=True) == runtime.parent / "execution-terminal.json",
        "terminal receipt belongs elsewhere",
    )
    terminal = runner._read_json(path)
    require(
        terminal.get("schema") == "stone-run-execution-terminal-v1"
        and terminal.get("status") == "failed"
        and terminal.get("scientific_result_accepted") is False
        and terminal.get("original_deadline_extended") is False
        and terminal.get("unit") == "stone-paper-model-anchor-i-medium-attempt-0001.service"
        and terminal.get("live_run_root_candidates") == []
        and terminal.get("selected_prefix_index") == SELECTED_INDEX
        and terminal.get("unfinished_stage_index") == SELECTED_INDEX + 1
        and all(
            terminal.get(key) is False
            for key in (
                "native_failure_receipt_present",
                "native_completion_receipt_present",
                "supervisor_result_present",
            )
        )
        and isinstance(terminal.get("recorded_pids_absent"), list)
        and len(terminal["recorded_pids_absent"]) >= 2
        and all(type(pid) is int and pid > 0 for pid in terminal["recorded_pids_absent"]),
        "original attempt lacks the declared truthful terminal outcome",
    )
    require(
        not any((runtime / name).exists() for name in ("failure.json", "completion.json"))
        and not (runtime.parent / "supervisor-result.json").exists()
        and not (runtime / "terminal").exists(),
        "original terminal acceptance or explicit remesh status changed",
    )
    journal = runtime.parent / "execution-terminal-journal.jsonl"
    require(sha(journal) == terminal["journal_sha256"], "terminal journal bytes differ")
    entries = [json.loads(line) for line in journal.read_text().splitlines() if line]
    events = terminal["terminal_journal_events"]
    require(
        len(events) == 2
        and all(
            any(all(entry.get(key) == value for key, value in event.items()) for entry in entries)
            for event in events
        )
        and "Service reached runtime time limit" in events[0]["MESSAGE"]
        and "Failed with result 'timeout'" in events[1]["MESSAGE"],
        "terminal timeout events differ",
    )
    return terminal, [regular_record(path), regular_record(journal)]


def derive_charge(raw):
    candidates = []
    for match in re.finditer(rb"(?m)^STATIONARY SOLVE\. Parameters: ([^\n]+)$", raw):
        parameters = {}
        for item in match.group(1).decode("ascii").split(", "):
            name, value = item.split("=", 1)
            require(name not in parameters, "duplicate stationary parameter")
            parameters[name] = float(value)
        if parameters.get("Ca_in") == -0.002:
            require(
                parameters == {"Bo": 1.56, "Ca_in": -0.002, "azimuthal_m": 0.0},
                "unfinished solve parameters differ",
            )
            candidates.append(match)
    require(len(candidates) == 1, "one unfinished stage-4 solve window required")
    start = candidates[0].start()
    window = raw[start:]
    require(window.count(b"STATIONARY SOLVE") == 1, "unfinished window contains another solve")
    steps = re.findall(rb"(?m)^\s*Newton Step (\d+): Maximum residuals[^\n]*$", window)
    require(
        steps == [b"1"]
        and window.count(b"Initial Maximum residuals") == 1
        and b"REMESHING" not in window
        and b"INVERTED ELEMENT" not in window
        and not any(
            term in window for term in (b"Reached max.", b"exceeded", b"Newton solve failure")
        )
        and window.count(b"Time to set up CRDoubleMatrix Jacobian") == 2
        and window.count(b"Time for linear solver (ndof=") == 1
        and window.rstrip().splitlines()[-1].startswith(b"Time to set up CRDoubleMatrix Jacobian"),
        "unfinished log does not certify the reviewed single-update/no-remesh bound",
    )
    return {
        "window_start": start,
        "window_end": len(raw),
        "window_sha256": hashlib.sha256(window).hexdigest(),
        "applied_updates": 1,
        "possible_inflight_updates": 1,
        "charged_updates": 2,
        "charged_remeshes": 0,
    }


def bind_budget(path, runtime, terminal_record, custody_record):
    """A root witness must bind the independently checked unfinished log window."""
    require(path is not None, "certified terminal budget witness required")
    witness = runner._read_json(path)
    log = runtime / "solver/_pyoomph_logfile.txt"
    charge = derive_charge(log.read_bytes())
    # The exact root witness schema is checked here, rather than accepting a scalar claim.
    require(
        witness.get("schema") == "stone-anchor-i-terminal-budget-v1",
        "terminal budget schema differs",
    )
    require(
        bound_record(witness.get("terminal")) == terminal_record
        and bound_record(witness.get("custody")) == custody_record
        and bound_record(witness.get("log")) == regular_record(log)
        and {key: witness["charge"].get(key) for key in charge} == charge
        and witness["charge"].get("stationary_entries") == 1
        and witness["charge"].get("initial_residual_entries") == 1
        and witness["charge"].get("printed_step_indices") == [1]
        and witness["charge"].get("terminal_marker_present") is False
        and witness["charge"].get("factorisation_in_flight") is True
        and witness.get("source_commit") == ORIGINAL_SOURCE
        and witness.get("pyoomph_commit") == diagnostic.PYOOMPH
        and witness.get("core_sha256") == diagnostic.CORE
        and witness.get("selected_prefix_index") == SELECTED_INDEX
        and witness.get("pending_stage_index") == SELECTED_INDEX + 1
        and witness.get("original_acceptance_changed") is False
        and witness.get("original_deadline_extended") is False,
        "terminal budget witness differs from its direct evidence",
    )
    pending = runtime / "stages/ca_004"
    require(
        (pending / "polish/ca_step/before.dump").is_file()
        and not (pending / "polish/ca_step/polish.json").exists()
        and not (pending / "stage.json").exists()
        and runner._read_json(pending / "transition.json")["verified"] is True,
        "unfinished stage-4 status differs",
    )
    require(
        bound_record(witness.get("pending_before"))
        == regular_record(pending / "polish/ca_step/before.dump")
        and bound_record(witness.get("pending_transition"))
        == regular_record(pending / "transition.json"),
        "unfinished state/transition binding differs",
    )
    listing = []
    for file in sorted(pending.rglob("*")):
        require(not file.is_symlink(), "unfinished tree contains a symlink")
        item = {
            "relative_path": str(file.relative_to(pending)),
            "kind": "file" if file.is_file() else "directory",
        }
        if file.is_file():
            item.update(sha256=sha(file), bytes=file.stat().st_size)
        listing.append(item)
    require(
        witness.get("stage4_directory_listing") == listing, "unfinished directory listing differs"
    )
    meshes = []
    for file in sorted(runtime.rglob("*.msh")):
        match = re.search(r"_REMESH_(\d+)", file.name)
        index = int(match.group(1)) if match else None
        require(index is None or index <= 2, "unfinished stage has an uncharged remesh")
        meshes.append(
            {
                **regular_record(file),
                "bytes": file.stat().st_size,
                "relative_path": str(file.relative_to(runtime)),
                "remesh_index": index,
            }
        )
    require(witness.get("mesh_inventory") == meshes, "unfinished mesh inventory differs")
    semantics = witness.get("source_semantics")
    require(
        isinstance(semantics, list) and len(semantics) == 5,
        "pinned callback/flush source binding required",
    )
    source_records = [bound_record(item) for item in semantics]
    require(
        {Path(item["path"]).name for item in source_records} == set(SEMANTIC_SHA)
        and all(item["sha256"] == SEMANTIC_SHA[Path(item["path"]).name] for item in source_records)
        and next(item for item in source_records if Path(item["path"]).name == "paper_bubble.py")[
            "sha256"
        ]
        == sha(ROOT / "problems/paper_bubble.py"),
        "callback/flush source inventory differs",
    )
    require(
        witness.get("remaining_caps")
        == {
            "first_mesh_updates": 38,
            "first_polish_total_updates": 158,
            "first_polish_remeshes": 3,
            "later_mesh_updates": 40,
            "later_polish_total_updates": 160,
            "later_polish_remeshes": 3,
            "total_new_phase_update_bound": 478,
        },
        "remaining budget differs from the carried charge",
    )
    return witness, charge, [regular_record(path), regular_record(log), *source_records]


def bind_original(args, component, environment, implementation):
    original_path = args.original_manifest.resolve(strict=True)
    original = runner._read_json(original_path)
    runtime = original_path.parent
    require(
        runtime.name == "runtime"
        and runtime.parent.name == "attempt-0001"
        and runtime.parent.parent.name == "attempts"
        and runtime.parent.parent.parent.name == "paper-model-anchor-i-medium",
        "original attempt identity differs",
    )
    run_root = runtime.parent.parent.parent
    target_path = Path(original["case_path"])
    target = runner.read_case(target_path)
    require(
        original["schema"] == "pyoomph-runner-manifest-v1"
        and original["anchor_stage"] == {"kind": "stationary_I_candidate"}
        and original["parent_acceptance_changed"] is False
        and original["azimuthal_stability"] is True
        and original["component_commit"] == original["component"]["commit"] == ORIGINAL_SOURCE
        and original["case_sha256"] == sha(target_path)
        and target["physics"]["ca_in"] == TARGET_CA
        and target["physics"]["bond"] == 1.56
        and target["physics"]["problem"] == "finite_gas_bubble",
        "original I construction differs",
    )
    old_runner = original["runner"]
    expected = hashlib.sha256(
        diagnostic.git(ROOT, "show", f"{ORIGINAL_SOURCE}:{ORIGINAL_PRODUCER}")
    ).hexdigest()
    require(
        Path(old_runner["path"]).name == "run_paper_anchor_i.py"
        and old_runner["path"] == original["runner_path"]
        and old_runner["sha256"] == original["runner_sha256"] == expected,
        "original producer binding differs",
    )
    preparation_path = runtime / "preparation.json"
    preparation = runner._read_json(preparation_path)
    contract = original["case_transition_contract"]
    require(
        preparation["contract"] == contract
        and preparation["implementation"] == original["implementation"]
        and preparation["numerical_acceptance"] is False,
        "original admission/implementation differs",
    )
    ladder_path = args.original_ca_ladder.resolve(strict=True)
    require(
        contract["ca_ladder"]["path"] == str(ladder_path)
        and contract["ca_ladder"]["sha256"] == sha(ladder_path)
        and runner._read_json(ladder_path)
        == {"schema": "herrada-paper-ca-ladder-v1", "values": ORIGINAL_VALUES}
        and contract["ca_ladder"]["values"] == ORIGINAL_VALUES
        and contract["per_mesh_stage_newton_cap"] == 40
        and contract["per_polish_total_newton_cap"] == 160
        and contract["per_polish_remesh_bound"] == 3
        and contract["explicit_terminal_remesh_calls"] == 1,
        "original ladder/numerical bounds differ",
    )
    terminal, terminal_records = bind_terminal(args.original_attempt_terminal, runtime)
    custody, custody_records = bind_custody(args.custody, run_root)
    budget, charge, budget_records = bind_budget(
        args.budget,
        runtime,
        regular_record(args.original_attempt_terminal),
        regular_record(args.custody),
    )
    records = [
        regular_record(original_path),
        regular_record(preparation_path),
        regular_record(target_path),
        regular_record(ladder_path),
    ]
    records.extend(preparation["inputs"])
    records.extend(terminal_records + custody_records + budget_records)
    prefix = []
    selected = None
    for index in range(1, SELECTED_INDEX + 1):
        directory = runtime / "stages" / f"ca_{index:03d}"
        stage_path = directory / "stage.json"
        stage = runner._read_json(stage_path)
        case_path = directory / "case.json"
        case = runner.read_case(case_path)
        manifest_path = directory / "run-manifest.json"
        manifest = runner._read_json(manifest_path)
        require(
            stage["index"] == index
            and stage["ca_in"] == ORIGINAL_VALUES[index - 1]
            and case == anchor.stage_case(target, stage["ca_in"], index, False)
            and stage["case"] == regular_record(case_path)
            and stage["numerical_acceptance"] is False
            and stage["polish"]["status"] == "converged"
            and manifest["component_commit"] == ORIGINAL_SOURCE
            and manifest["component"] == original["component"]
            and manifest["implementation"] == original["implementation"]
            and manifest["case_transition_contract"] == contract
            and manifest["parent_restart"] == original["parent_restart"]
            and manifest["anchor_stage"]
            == {"kind": "ascending_ca_continuation", "index": index, "ca_in": stage["ca_in"]},
            "completed prefix stage/lineage differs",
        )
        packet_path = directory / relative_path(stage["checkpoint"]["resume_packet"])
        restart = runner.validate_restart(
            packet_path,
            case,
            sha(case_path),
            component=component,
            environment=environment,
            azimuthal_stability=True,
        )
        require(
            restart["stage"]
            == {
                "kind": "completed_polish",
                "bond_index": 16,
                "bond": 1.56,
                "ca_in": stage["ca_in"],
                "bc_mode": "init",
            }
            and restart["state"]["format"] == "portable_file"
            and restart["parent_manifest"] == regular_record(manifest_path)
            and stage["resume_packet_sha256"] == restart["packet"]["sha256"],
            "completed prefix packet differs",
        )
        sample_path = directory / "samples/after-polish.json"
        sample = runner._read_json(sample_path)
        checks = remesh.canonical_checks(case, sample["metrics"])
        require(
            set(checks) == GATES
            and all(value is True for value in checks.values())
            and stage["canonical_checks_observed"] == checks
            and sample["sampling_preserved_state"] is True
            and portable_signature(sample["signature"])
            == portable_signature(restart["roundtrip"]["after"])
            and stage["polish"] == runner._read_json(directory / "polish/ca_step/polish.json")
            and stage["polish"]["updates_total"] <= 160
            and stage["polish"]["stage_updates"] <= 40
            and stage["polish"]["remeshes_total"] <= 3,
            "provisional prefix observations/limits differ",
        )
        transition = runner._read_json(directory / "transition.json")
        require(
            transition == stage["transition"] and transition["verified"] is True,
            "prefix transition differs",
        )
        row = {
            "index": index,
            "ca_in": stage["ca_in"],
            "stage": regular_record(stage_path),
            "case": regular_record(case_path),
            "manifest": regular_record(manifest_path),
            "packet": restart["packet"],
            "state": restart["state"],
            "meshes": restart["meshes"],
            "updates": stage["polish"]["updates_total"],
            "remeshes": stage["polish"]["remeshes_total"],
            "original_acceptance": False,
        }
        terminal_row = terminal["completed_provisional_stages"][index - 1]
        require(
            terminal_row
            == {
                "index": index,
                "ca_in": stage["ca_in"],
                "stage_sha256": row["stage"]["sha256"],
                "polish_status": "converged",
                "updates": row["updates"],
                "remeshes": row["remeshes"],
                "acceptance": False,
            },
            "terminal completed-prefix account differs",
        )
        prefix.append(row)
        records.extend(
            [restart["packet"], restart["parent_manifest"], restart["state"], *restart["meshes"]]
        )
        records.extend(
            regular_record(directory / relative)
            for relative in (
                "case.json",
                "stage.json",
                "transition.json",
                "polish/ca_step/polish.json",
                "samples/after-polish.json",
                "samples/after-polish.npz",
            )
        )
        if index == SELECTED_INDEX:
            selected = {"case": case, "restart": restart, "sample": sample}
    require(
        len(terminal["completed_provisional_stages"]) == SELECTED_INDEX,
        "later completed prefix unhandled",
    )
    require(
        budget["completed_prefix_updates"] == sum(row["updates"] for row in prefix)
        and budget["completed_prefix_polish_remeshes"] == sum(row["remeshes"] for row in prefix),
        "terminal prefix budget account differs",
    )
    require(
        args.case.resolve(strict=True) == Path(prefix[-1]["case"]["path"])
        and args.stage.resolve(strict=True) == Path(prefix[-1]["stage"]["path"])
        and args.manifest.resolve(strict=True) == Path(prefix[-1]["manifest"]["path"])
        and args.packet.resolve(strict=True) == Path(prefix[-1]["packet"]["path"])
        and args.state.resolve(strict=True) == Path(prefix[-1]["state"]["path"]),
        "selected inputs differ from the terminal prefix",
    )
    saved = Path(selected["restart"]["state"]["path"]).read_bytes()
    metadata = cold.serialized_metadata(saved)
    fingerprint = runner.snapshot_bc_cache_signature(
        saved, mesh_reference_base=Path(selected["restart"]["mesh_reference_base"])
    )
    require(
        metadata["endtime"] == selected["case"]["solve"]["relax_time"] == 8.0
        and {key: value for key, value in fingerprint.items() if key != "sha256"}
        == {
            key: value
            for key, value in selected["restart"]["roundtrip"]["after"]["serialized_state"].items()
            if key != "sha256"
        },
        "saved complete state/history differs",
    )
    lineage = {
        "schema": SCHEMA,
        "original_attempt": {
            "manifest": regular_record(original_path),
            "terminal": regular_record(args.original_attempt_terminal),
            "custody": regular_record(args.custody),
            "status": terminal["status"],
            "accepted": False,
            "acceptance_changed": False,
        },
        "completed_prefix": prefix,
        "selected_index": SELECTED_INDEX,
        "remaining_ladder": {
            "original": regular_record(ladder_path),
            "values": ORIGINAL_VALUES[SELECTED_INDEX:],
            "original_indices": [4, 5],
            "target": regular_record(target_path),
        },
        "budget": {
            "witness": regular_record(args.budget),
            "charge": charge,
            "completed_updates": sum(row["updates"] for row in prefix),
            "completed_remeshes": sum(row["remeshes"] for row in prefix),
            "closed_allowance_transferable": False,
        },
        "consumer": {
            "module": "runners/run_paper_seed_requalify.py",
            "callable": "restore_checked",
            "sha256": implementation["runners/run_paper_seed_requalify.py"]["sha256"],
            "saved_endtime": metadata["endtime"],
            "api": cold.API,
        },
        "original_acceptance_changed": False,
        "scope": "stationary prefix and full storage history; no terminal-anchor acceptance",
    }
    return {
        **selected,
        "target": target,
        "metadata": metadata,
        "lineage": lineage,
        "records": unique_records(records),
        "component": component,
        "environment": environment,
        "implementation": implementation,
        "original_runtime": str(runtime),
        "original_run_root": str(run_root),
        "custody_root": custody["destination"],
        "budget_witness": budget,
    }


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "runner must use its isolated component")
    original = runner._read_json(args.original_manifest)
    component, environment, implementation = source_contract(original, ENTRYPOINT)
    prepared = bind_original(args, component, environment, implementation)
    stable_inputs(prepared)
    return prepared


def stable_inputs(prepared):
    remesh.verify_inputs(prepared["records"])
    remesh.verify_source(prepared["component"], prepared["implementation"])


def output_path(path, prepared, additional=()):
    require(not path.is_symlink(), "output must not be a symlink")
    output = runner.validate_output_path(path, ROOT)
    require(not output.exists(), "new output must be absent")
    for source in (
        prepared.get("original_run_root", prepared["original_runtime"]),
        prepared["custody_root"],
        *additional,
    ):
        evidence = Path(source).resolve()
        require(
            not output.is_relative_to(evidence) and not evidence.is_relative_to(output),
            "output overlaps immutable evidence",
        )
    return output


def manifest(args, prepared, entrypoint, argv, *, case_path=None, case=None, scope=None):
    case_path = args.case if case_path is None else case_path
    case = prepared["case"] if case is None else case
    entry = prepared["implementation"][entrypoint]
    return {
        "schema": "pyoomph-runner-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "case_path": str(case_path.resolve()),
        "case_file": str(case_path.resolve()),
        "case_sha256": sha(case_path),
        "case_id": case["case_id"],
        "case_fields_used": list(runner.CASE_FIELDS_USED),
        "case_fields_scope": "unchanged stationary case, complete restart and original five predicates",
        "units": "nondimensional (sigma=mu_liquid=R0=1)",
        "nondimensionalization": case["nondimensionalization"],
        "argv": [sys.executable, str(ROOT / entrypoint), *argv],
        "runner": entry,
        "runner_path": entry["path"],
        "runner_sha256": entry["sha256"],
        "component_commit": prepared["component"]["commit"],
        "component": prepared["component"],
        "pyoomph": prepared["environment"],
        "azimuthal_stability": True,
        "implementation": prepared["implementation"],
        "lineage": prepared["lineage"],
        "qualification_scope": scope or prepared["lineage"]["scope"],
        "qualification_schema": SCHEMA,
        "original_acceptance_changed": False,
    }


class ZeroSolveGuard:
    """Install refusal stubs before initialise; retain explicit coverage and violations."""

    def __init__(self, problem):
        self.installed, self.violations = [], []
        for name in (
            "solve",
            "run",
            "force_remesh",
            "solve_eigenproblem",
            "steady_newton_solve",
            "unsteady_newton_solve",
        ):
            require(callable(getattr(problem, name, None)), f"zero-solve guard API missing: {name}")

            def refuse(*args, _name=name, **kwargs):
                self.violations.append(_name)
                raise RuntimeError(f"zero-solve prefix qualification refused {_name}")

            setattr(problem, name, refuse)
            self.installed.append(name)

    def receipt(self):
        require(not self.violations, "zero-solve guard was violated")
        return {
            "installed_before_initialise": True,
            "methods": self.installed,
            "violations": [],
            "solve_calls": 0,
            "physical_remesh_calls": 0,
            "time_evolution": False,
        }


def configure_guarded(problem, directory, typed_directory):
    guard = ZeroSolveGuard(problem)
    requalification.configure(problem, directory)
    problem._seed_requalification_receipts_directory = typed_directory
    return guard


def execute(args, prepared, output):
    import numpy as np

    case, restart = prepared["case"], prepared["restart"]
    expected = restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
    pins = prepared["sample"]["actual_pins"]
    stable_inputs(prepared)
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as problem:
        guard = configure_guarded(problem, output / "solver", output / "typed-loads")
        first = requalification.restore_checked(problem, case, restart, pins, expected)
        write_json(output / "parent-cold-load.json", first)
        checks = requalification.sample_checked(problem, case, output, "prefix", np)
        before = cold.retain_snapshot(problem, output, "before-checkpoint")
        try:
            checkpoint = model._checkpoint_stationary_state(
                problem, case, output, restart["stage"]["bond_index"], True
            )
        finally:
            after = cold.retain_snapshot(problem, output, "after-checkpoint")
        require(
            cold.normalized_bytes(before, Path.cwd()) == cold.normalized_bytes(after, Path.cwd()),
            "checkpoint changed complete state/history",
        )
        require(diagnostic.actual_pins(problem) == pins, "checkpoint changed actual pins")
        final = runner.validate_restart(
            output / checkpoint["resume_packet"],
            case,
            sha(args.case),
            component=prepared["component"],
            environment=prepared["environment"],
            azimuthal_stability=True,
        )
        require(
            final["parent_manifest"] == regular_record(output / "run-manifest.json")
            and final["stage"] == restart["stage"]
            and Path(final["state"]["path"]).is_relative_to(output)
            and final["roundtrip"]["after"]["serialized_state"]["payload_sha256"] == expected
            and cold.normalized_bytes(
                Path(final["state"]["path"]).read_bytes(), Path(final["mesh_reference_base"])
            )
            == cold.normalized_bytes(
                Path(restart["state"]["path"]).read_bytes(), Path(restart["mesh_reference_base"])
            ),
            "portable descendant differs from the retained prefix",
        )
        write_json(output / "first-zero-solve-guard.json", guard.receipt())
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as fresh:
        guard = configure_guarded(fresh, output / "cold-solver", output / "cold-typed-loads")
        second = requalification.restore_checked(fresh, case, final, pins, expected)
        cold_checks = requalification.sample_checked(fresh, case, output, "cold-load", np)
        write_json(
            output / "cold-load.json",
            {
                **second,
                "checks": cold_checks,
                "packet": final["packet"],
                "parent_manifest": final["parent_manifest"],
            },
        )
        write_json(output / "second-zero-solve-guard.json", guard.receipt())
    stable_inputs(prepared)
    runner.verify_restart_artifacts(final)
    return {
        "schema": SCHEMA,
        "numerical_acceptance": True,
        "stationary_prefix_accepted": True,
        "anchor_accepted": False,
        "checks": checks,
        "cold_load_checks": cold_checks,
        "cold_load_verified": True,
        "checkpoint": checkpoint,
        "lineage": prepared["lineage"],
        "original_attempt_status": "failed",
        "original_acceptance_changed": False,
        **NO_CLAIMS,
    }


def completion_inventory(output, *, exclude_solver=False):
    result = {}
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), "completion contains a symlink")
        if path.is_file() and path.name not in ("progress.jsonl", "completion.json"):
            relative = path.relative_to(output)
            if exclude_solver and "solver" in relative.parts:
                continue
            result[str(relative)] = sha(path)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "case",
        "original-attempt-terminal",
        "original-manifest",
        "stage",
        "manifest",
        "packet",
        "state",
        "original-ca-ladder",
        "custody",
        "out",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--budget", type=Path)
    result.add_argument(
        "--check", action="store_true", help="metadata only; no native imports or output"
    )
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    output = output_path(args.out, prepared)
    if args.check:
        print(
            json.dumps(
                {
                    "verified": True,
                    "native_imported": False,
                    "output_created": False,
                    "lineage": prepared["lineage"],
                },
                allow_nan=False,
            )
        )
        return 0
    actual = manifest(args, prepared, ENTRYPOINT, arguments)
    actual["intervention"] = {
        "kind": "stationary-continuation-prefix-qualification",
        "solve_calls": 0,
        "remesh_calls": 0,
        "time_evolution": False,
        "api": cold.API,
        "original_acceptance_changed": False,
    }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    runner.create_run_manifest(output, actual)
    runner.append_progress(output, {"event": "prepared", "runner": actual["runner"]}, first=True)
    try:
        write_json(
            output / "preparation.json",
            {
                "inputs": prepared["records"],
                "implementation": prepared["implementation"],
                "lineage": prepared["lineage"],
                "numerical_acceptance": False,
            },
        )
        write_json(
            output / "schema.json",
            {
                "schema": SCHEMA,
                "scope": actual["qualification_scope"],
                "consumer": prepared["lineage"]["consumer"],
            },
        )
        result = execute(args, prepared, output)
        stable_inputs(prepared)
        write_json(output / "result.json", result)
        write_json(
            output / "summary.json",
            {
                **result,
                "status": "completed",
                "completion_reached": True,
                "scientific_status": "prefix_checks_passed",
            },
        )
        write_json(
            output / "completion.json",
            {
                "schema": SCHEMA,
                "status": "complete",
                "numerical_acceptance": True,
                "stationary_prefix_accepted": True,
                "anchor_accepted": False,
                "original_acceptance_changed": False,
                "files": completion_inventory(output),
            },
        )
        runner.append_progress(
            output, {"event": "final", "stationary_prefix_accepted": True, "anchor_accepted": False}
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "numerical_acceptance": False,
            "stationary_prefix_accepted": False,
            "anchor_accepted": False,
            "original_acceptance_changed": False,
            "inputs_unchanged": diagnostic.parents_unchanged(prepared["records"]),
        }
        write_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", **failure})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
