"""Qualify two exact retained I targets by fresh loads without solving.

The failed suffix and both original attempts remain failed and unaccepted.
The selected stationary state receives no spatial or spectral certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_anchor_i_prefix as suffix
from runners import run_paper_prefix_qualify as prefix

model, runner, diagnostic, remesh = prefix.model, prefix.runner, prefix.diagnostic, prefix.remesh
require, sha, write_json = prefix.require, prefix.sha, prefix.write_json
record, bound, unique = prefix.regular_record, prefix.bound_record, prefix.unique_records
cold = prefix.cold
ENTRYPOINT = "runners/run_paper_i_cold_target_qualify.py"
SOURCE = "b0c9b609146e7b4881b0c7e1894216718b0be9d7"
TERMINAL_SHA = "f03240849ffd2404f51f4c91ab8be87c1599ea2e73bc022f57362f718f478c96"
CUSTODY_SHA = "9bed88e148cde777660195c64e8a255e4af7a1192190f597bbbfa7c009721d5d"
TARGETS = {4: -0.002, 5: -0.0016}
SELECTED = {
    4: {
        "case": "c98ed9ca4b5c86d9af5cb91298b0d118b8c912765bd7783947e654232fce394b",
        "packet": "9e9992b52cf456d958dd956b5b6a44a5778726d54f17c35c5f8d848b0a3e5978",
        "state": "0ca03a835c082b9d24da6c3c2cb788d03335dc32db8482c8b0d73ac3cf53486a",
    },
    5: {
        "case": "d0e8182aea56cc5e3515d755d7d935ab799cf324485da89ad36c93108b97fdf0",
        "packet": "b500e1508972cafd4c114e4380297aa4c12c11171d6dd12457a8c18668de7ec4",
        "state": "d2c033a5aa93c925abffb1663b7e1d56aa0b1ade860506fb3487bc27bcd4ebee",
    },
}
OOM_SCOPE = "terminal remesh executed, follow-up polish OOM before first update; no spatial-convergence certificate"
MESH_TAG = "retained-350214-DOF-I-mesh-built-at-Ca--0.02"
NO_CLAIMS = {
    **prefix.NO_CLAIMS,
    "spatial_convergence_certificate": False,
    "mesh_certificate": False,
    "paper_agreement_certificate": False,
}


def stage_type(index):
    require(type(index) is int and index in TARGETS, "only exact retained I stages 4 and 5")
    return {
        "schema": f"stationary-I-cold-target-stage-{index}-qualification-v1",
        "kind": f"stationary_I_cold_target_stage_{index}",
        "index": index,
        "ca_in": TARGETS[index],
        "scope": f"stationary-only retained I stage {index} at Ca={TARGETS[index]}; two fresh zero-solve loads; no spatial or spectral certificate",
        "suffix_terminal_scope": OOM_SCOPE,
        "suffix_terminal_remesh_executed": True,
        "selected_stage_remesh_attempted": index == 5,
        "mesh_tag": MESH_TAG,
    }


def audit_module():
    # Reuse only the unchanged metadata/retained-proof binders. No native API.
    from runners import run_paper_modes

    require(
        Path(run_paper_modes.__file__).resolve() == ROOT / "runners/run_paper_modes.py",
        "metadata proof helper imported elsewhere",
    )
    return run_paper_modes


def bind_archive(path, run_root, runtime):
    require(sha(path) == CUSTODY_SHA, "exact failed suffix custody receipt required")
    receipt = runner._read_json(path)
    require(
        receipt["schema"] == "figure10-next-phase-archive-v1"
        and receipt["simulation"] == "paper-model-anchor-i-from-prefix-medium"
        and receipt["source"] == str(run_root)
        and receipt["attempt"] == "attempt-0001"
        and receipt["file_count"] == len(receipt["files"]) == 142
        and receipt["bytes"]
        == sum(item["bytes"] for item in receipt["files"].values())
        == 364779278
        and receipt["source_deleted"] is False
        and receipt["verified_at"] == receipt["protocol_files_verified_at"],
        "failed suffix archive authority differs",
    )
    records, files = [record(path)], {}
    for name, declaration in receipt["files"].items():
        source = run_root / prefix.relative_path(name)
        item = record(source)
        require(
            item["sha256"] == declaration["sha256"]
            and source.stat().st_size == declaration["bytes"],
            "archived source bytes differ",
        )
        records.append(item)
        if source.is_relative_to(runtime):
            files[str(source.relative_to(runtime))] = item["sha256"]
    # Later root receipts outside the sealed runtime are separate authorities.
    actual = {str(path.relative_to(runtime)) for path in runtime.rglob("*") if path.is_file()}
    require(actual == set(files), "failed runtime inventory differs from archive")
    require(not any(path.is_symlink() for path in runtime.rglob("*")), "failed runtime symlink")
    return receipt, files, records


def bind_oom(path, runtime, files):
    require(sha(path) == TERMINAL_SHA, "exact suffix OOM receipt required")
    value = runner._read_json(path)
    require(
        value["schema"] == "stone-execution-terminal-v1"
        and value["status"] == "failed"
        and value["scientific_result_accepted"] is False
        and value["outputs_preserved"] is True
        and value["no_new_counter_allowance"] is True
        and value["unit"] == "stone-paper-model-anchor-i-from-prefix-medium-attempt-0001.service"
        and "Result=oom-kill\n" in value["unit_status"]
        and "MemoryPeak=51539607552\n" in value["unit_status"]
        and value["after_remesh_ndof"] == 1650666
        and value["post_remesh_updates_logged"] == 0
        and len(value["pids_absent"]) == 2
        and all(type(pid) is int and pid > 0 for pid in value["pids_absent"]),
        "suffix OOM outcome differs",
    )
    require(
        not any(
            (runtime / name).exists()
            for name in (
                "failure.json",
                "completion.json",
                "result.json",
                "summary.json",
                "target-cold",
                "terminal/polish/terminal_remesh/polish.json",
            )
        ),
        "failed suffix acquired terminal acceptance or follow-up polish",
    )
    observations = {}
    for phase in ("before", "after-remesh"):
        sample = runner._read_json(runtime / "terminal/samples" / (phase + ".json"))
        retained = runner._read_json(runtime / "terminal/states" / phase / "retention.json")
        observations[phase.replace("-", "_")] = {**sample, "retained_state": retained}
    methods = audit_module()
    records = [record(path)]
    for phase, observed in observations.items():
        records.extend(
            methods.bind_portable_observation(observed, phase.replace("_", "-"), runtime, files)
        )
    require(
        observations["before"]["signature"]["layout"]["ndof"] == 350214
        and observations["after_remesh"]["signature"]["layout"]["ndof"]
        == value["after_remesh_ndof"],
        "terminal remesh was not retained",
    )
    records.extend(methods.bind_pressure_history_audit(runtime, observations, files))
    return value, observations, records


def replay_suffix(parent):
    seed = parent["parent_restart"]
    prepared = suffix.preflight(
        SimpleNamespace(
            source=ROOT,
            case=Path(parent["case_path"]),
            manifest=Path(seed["parent_manifest"]["path"]),
            packet=Path(seed["packet"]["path"]),
            state=Path(seed["state"]["path"]),
        )
    )
    historical = {
        **prepared["restart"],
        "compatibility": {**prepared["restart"]["compatibility"], "current_commit": SOURCE},
    }
    require(
        parent["lineage"] == prepared["lineage"]
        and parent["case_transition_contract"] == prepared["contract"]
        and parent["parent_restart"] == historical,
        "failed suffix original admission differs",
    )
    return prepared


def current_source(parent, prepared):
    require(
        parent["component_commit"] == parent["component"]["commit"] == SOURCE,
        "exact suffix source required",
    )
    runner.require_clean_source(parent["component"])
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", SOURCE, prepared["component"]["commit"])
    implementation = dict(prepared["implementation"])
    for name in (prefix.ENTRYPOINT, suffix.ENTRYPOINT):
        require(
            implementation[name]["sha256"]
            == hashlib.sha256(diagnostic.git(ROOT, "show", f"{SOURCE}:{name}")).hexdigest(),
            "frozen I qualification/helper changed",
        )
    for name in (ENTRYPOINT, "runners/run_paper_modes.py"):
        item = record(ROOT / name)
        require(
            item["sha256"]
            == hashlib.sha256(
                diagnostic.git(ROOT, "show", f"{prepared['component']['commit']}:{name}")
            ).hexdigest(),
            "new implementation differs from committed source",
        )
        implementation[name] = item
    return implementation


def bind_stages(runtime, parent, prepared, files):
    methods, rows = audit_module(), {}
    records = []
    for index, ca in TARGETS.items():
        directory = runtime / "stages" / f"ca_{index:03d}"
        row = runner._read_json(directory / "stage.json")
        require(
            row["index"] == index
            and row["ca_in"] == ca
            and row["numerical_acceptance"] is False
            and row["polish"]["status"] == "converged"
            and row["polish"]["updates_total"] == (9 if index == 4 else 6)
            and row["polish"]["remeshes_total"] == 0
            and all(value is True for value in row["canonical_checks_observed"].values()),
            "exact retained stage gates/counters differ",
        )
        records.extend(
            methods.bind_prefix_stage(runtime, row, prepared["target"], parent, prepared, files)
        )
        records.append(record(directory / "stage.json"))
        if index == 4:
            carry = row["carry"]
            require(
                carry["carried"] == {"updates": 2, "remeshes": 0}
                and carry["new_spent"] == {"updates": 7, "remeshes": 0}
                and carry["aggregate_spent"] == {"updates": 9, "remeshes": 0}
                and carry["deactivated"] is True
                and carry["counter_seeded_once"] is True,
                "historical stage-4 carry differs",
            )
            records.append(bound(carry["journal"]))
        else:
            require(row["carry"] is None, "stage 5 has no carried debit")
        rows[index] = row
    return rows, records


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "isolated qualification source required")
    identity = stage_type(args.index)
    manifest = Path(record(args.suffix_manifest)["path"])
    runtime = manifest.parent
    run_root = runtime.parent.parent.parent
    require(
        runtime.name == "runtime"
        and runtime.parent.name == "attempt-0001"
        and run_root.name == "paper-model-anchor-i-from-prefix-medium"
        and manifest.name == "run-manifest.json",
        "exact failed suffix attempt required",
    )
    parent = runner._read_json(manifest)
    require(
        parent["anchor_stage"] == {"kind": "stationary_I_candidate_from_prefix"}
        and Path(parent["runner"]["path"]).name == "run_paper_anchor_i_prefix.py"
        and parent["original_acceptance_changed"] is False,
        "failed suffix producer type differs",
    )
    admitted = replay_suffix(parent)
    implementation = current_source(parent, admitted)
    preparation = runner._read_json(runtime / "preparation.json")
    require(
        preparation["contract"] == parent["case_transition_contract"]
        and preparation["lineage"] == parent["lineage"]
        and preparation["implementation"] == parent["implementation"]
        and preparation["numerical_acceptance"] is False
        and preparation["inputs"] == admitted["records"],
        "failed suffix preparation differs",
    )
    custody, files, records = bind_archive(args.custody, run_root, runtime)
    outcome, terminal, proof = bind_oom(args.terminal, runtime, files)
    records.extend(proof)
    rows, proof = bind_stages(runtime, parent, admitted, files)
    records.extend(proof)
    directory = runtime / "stages" / f"ca_{args.index:03d}"
    case_path = directory / "case.json"
    require(
        args.case.resolve(strict=True) == case_path
        and sha(case_path) == SELECTED[args.index]["case"],
        "exact selected case required",
    )
    case = runner.read_case(case_path)
    packet = directory / prefix.relative_path(rows[args.index]["checkpoint"]["resume_packet"])
    restart = runner.validate_restart(
        packet,
        case,
        sha(case_path),
        component=admitted["component"],
        environment=admitted["environment"],
        azimuthal_stability=True,
    )
    require(
        restart["packet"]["sha256"] == SELECTED[args.index]["packet"]
        and restart["state"]["sha256"] == SELECTED[args.index]["state"]
        and restart["parent_manifest"] == record(directory / "run-manifest.json")
        and restart["stage"]
        == {
            "kind": "completed_polish",
            "bond_index": 16,
            "bond": 1.56,
            "ca_in": identity["ca_in"],
            "bc_mode": "init",
        },
        "exact selected portable checkpoint required",
    )
    metadata = cold.serialized_metadata(Path(restart["state"]["path"]).read_bytes())
    sample = runner._read_json(directory / "samples/after-polish.json")
    require(
        metadata["endtime"] == case["solve"]["relax_time"] == 8.0
        and prefix.portable_signature(sample["signature"])
        == prefix.portable_signature(restart["roundtrip"]["after"])
        and all(remesh.canonical_checks(case, sample["metrics"]).values()),
        "saved I history/forcing/signature or five predicates differ",
    )
    if args.index == 5:
        require(
            prefix.portable_signature(terminal["before"]["signature"])
            == prefix.portable_signature(sample["signature"])
            and terminal["before"]["retained_state"]["state"]["sha256"]
            == restart["state"]["sha256"],
            "OOM pre-remesh state differs from stage 5",
        )
    lineage = {
        "type": identity,
        "selected": {
            "case": record(case_path),
            "stage": record(directory / "stage.json"),
            "restart": restart,
            "pins": sample["actual_pins"],
        },
        "failed_suffix": {
            "manifest": record(manifest),
            "terminal": record(args.terminal),
            "custody": record(args.custody),
            "status": outcome["status"],
            "accepted": False,
            "scope": OOM_SCOPE,
            "oom_at_stage": 5,
        },
        "original_lineage": parent["lineage"],
        "historical_counters": {
            str(index): {
                "updates": row["polish"]["updates_total"],
                "remeshes": row["polish"]["remeshes_total"],
                "carry": row["carry"],
            }
            for index, row in rows.items()
        },
        "new_budget": {"newton_updates": 0, "physical_remeshes": 0, "time_evolution": False},
        "consumer": {
            "module": prefix.requalification.ENTRYPOINT,
            "callable": "restore_checked",
            "sha256": implementation[prefix.requalification.ENTRYPOINT]["sha256"],
            "saved_endtime": 8.0,
            "api": cold.API,
        },
        "original_acceptance_changed": False,
    }
    prepared = {
        **admitted,
        "case": case,
        "restart": restart,
        "sample": sample,
        "lineage": lineage,
        "implementation": implementation,
        "records": unique([*admitted["records"], *records, record(manifest)]),
        "original_runtime": str(runtime),
        "original_run_root": str(run_root),
        "custody_root": custody["destination"],
    }
    prefix.stable_inputs(prepared)
    return prepared


def execute(args, prepared, output):
    import numpy as np

    case, restart = prepared["case"], prepared["restart"]
    expected = restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
    pins = prepared["sample"]["actual_pins"]
    identity = prepared["lineage"]["type"]
    prefix.stable_inputs(prepared)
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as problem:
        guard = prefix.configure_guarded(problem, output / "solver", output / "typed-loads")
        first = prefix.requalification.restore_checked(problem, case, restart, pins, expected)
        write_json(output / "parent-cold-load.json", first)
        checks = prefix.requalification.sample_checked(problem, case, output, "prefix", np)
        before = cold.retain_snapshot(problem, output, "before-checkpoint")
        try:
            checkpoint = model._checkpoint_stationary_state(
                problem, case, output, restart["stage"]["bond_index"], True
            )
        finally:
            after = cold.retain_snapshot(problem, output, "after-checkpoint")
        require(
            cold.normalized_bytes(before, Path.cwd()) == cold.normalized_bytes(after, Path.cwd())
            and diagnostic.actual_pins(problem) == pins,
            "checkpoint changed full state/history or pins",
        )
        final = runner.validate_restart(
            output / checkpoint["resume_packet"],
            case,
            sha(args.case),
            component=prepared["component"],
            environment=prepared["environment"],
            azimuthal_stability=True,
        )
        require(
            final["parent_manifest"] == record(output / "run-manifest.json")
            and final["stage"] == restart["stage"]
            and Path(final["state"]["path"]).is_relative_to(output)
            and final["roundtrip"]["after"]["serialized_state"]["payload_sha256"] == expected
            and cold.normalized_bytes(
                Path(final["state"]["path"]).read_bytes(), Path(final["mesh_reference_base"])
            )
            == cold.normalized_bytes(
                Path(restart["state"]["path"]).read_bytes(), Path(restart["mesh_reference_base"])
            ),
            "portable descendant differs from retained I target",
        )
        write_json(output / "first-zero-solve-guard.json", guard.receipt())
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as fresh:
        require(fresh is not problem, "two genuinely fresh cold-load instances required")
        guard = prefix.configure_guarded(fresh, output / "cold-solver", output / "cold-typed-loads")
        second = prefix.requalification.restore_checked(fresh, case, final, pins, expected)
        cold_checks = prefix.requalification.sample_checked(fresh, case, output, "cold-load", np)
        write_json(
            output / "cold-load.json",
            {
                **second,
                "checks": cold_checks,
                "canonical_checks": cold_checks,
                "packet": final["packet"],
                "parent_manifest": final["parent_manifest"],
                "qualification_type": identity,
            },
        )
        write_json(output / "second-zero-solve-guard.json", guard.receipt())
    prefix.stable_inputs(prepared)
    runner.verify_restart_artifacts(final)
    return {
        "schema": identity["schema"],
        "type": identity,
        "numerical_acceptance": True,
        "stationary_stage_accepted": True,
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


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "suffix-manifest", "terminal", "custody", "out"):
        value.add_argument("--" + name, type=Path, required=True)
    value.add_argument("--index", type=int, choices=tuple(TARGETS), required=True)
    value.add_argument("--check", action="store_true")
    return value


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    output = prefix.output_path(args.out, prepared)
    identity = prepared["lineage"]["type"]
    if args.check:
        print(
            json.dumps(
                {
                    "verified": True,
                    "native_imported": False,
                    "output_created": False,
                    "type": identity,
                },
                allow_nan=False,
            )
        )
        return 0
    manifest = prefix.manifest(args, prepared, ENTRYPOINT, arguments, scope=identity["scope"])
    manifest.update(
        qualification_schema=identity["schema"],
        anchor_stage={"kind": identity["kind"]},
        intervention={
            "kind": "stationary-I-cold-target-qualification",
            **prepared["lineage"]["new_budget"],
            "api": cold.API,
        },
        **NO_CLAIMS,
    )
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    runner.create_run_manifest(output, manifest)
    runner.append_progress(output, {"event": "prepared", "type": identity}, first=True)
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
            output / "schema.json", {"type": identity, "consumer": prepared["lineage"]["consumer"]}
        )
        result = execute(args, prepared, output)
        prefix.stable_inputs(prepared)
        sample = runner._read_json(output / "samples/prefix.json")
        write_json(output / "result.json", result)
        write_json(
            output / "summary.json",
            {
                **sample["metrics"],
                **result,
                "status": "completed",
                "completion_reached": True,
                "scientific_status": "stationary_only_checks_passed",
                "ca_in": identity["ca_in"],
                "checkpoint": result["checkpoint"]["resume_packet"],
            },
        )
        write_json(
            output / "completion.json",
            {
                "schema": identity["schema"],
                "status": "complete",
                "numerical_acceptance": True,
                "stationary_stage_accepted": True,
                "anchor_accepted": False,
                "original_acceptance_changed": False,
                "type": identity,
                "files": prefix.completion_inventory(output),
            },
        )
        runner.append_progress(
            output,
            {
                "event": "final",
                "stationary_stage_accepted": True,
                "anchor_accepted": False,
                "type": identity,
            },
        )
        return 0
    except BaseException as error:
        write_json(
            output / "failure.json",
            {
                "schema": identity["schema"],
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "numerical_acceptance": False,
                "stationary_stage_accepted": False,
                "anchor_accepted": False,
                "original_acceptance_changed": False,
                **NO_CLAIMS,
            },
        )
        raise


def snapshot_proof(runtime, phase, files):
    methods = audit_module()
    path = runtime / (phase + ".dump")
    own = methods.bound_parent_file(record(path), runtime, files)
    retention_path = runtime / (phase + "-retention.json")
    records = [own, methods.bound_parent_file(record(retention_path), runtime, files)]
    retained = runner._read_json(retention_path)
    require(
        retained["state"] == {**own, "format": "memory_snapshot"}
        and [item["stored_path"] for item in retained["meshes"]]
        == runner.state_mesh_references(path)
        and retained["mesh_exchange"]
        == {item["stored_path"]: item["path"] for item in retained["meshes"]},
        "snapshot full mesh closure differs",
    )
    base = Path(retained["mesh_reference_base"])
    for item in retained["meshes"]:
        records.append(methods.bound_parent_file(item, runtime, files))
        require(
            sha(base / item["stored_path"]) == item["sha256"],
            "snapshot referenced mesh content differs",
        )
    raw = path.read_bytes()
    return (
        raw,
        cold.normalized_bytes(raw, base),
        runner.snapshot_bc_cache_signature(raw, mesh_reference_base=base),
        records,
    )


def admit(runtime, case, parent, restart, summary, second, files, completion):
    """Replay this exact zero-solve producer without accepting its failed ancestor."""
    require(
        Path(parent["runner"]["path"]).name == Path(ENTRYPOINT).name,
        "exact cold-target producer required",
    )
    lineage = parent["lineage"]
    identity = stage_type(lineage["type"]["index"])
    ancestor = lineage["failed_suffix"]
    admitted = preflight(
        SimpleNamespace(
            source=ROOT,
            index=identity["index"],
            case=Path(lineage["selected"]["case"]["path"]),
            suffix_manifest=Path(ancestor["manifest"]["path"]),
            terminal=Path(ancestor["terminal"]["path"]),
            custody=Path(ancestor["custody"]["path"]),
        )
    )
    historical = {
        **admitted["lineage"],
        "selected": {
            **admitted["lineage"]["selected"],
            "restart": {
                **admitted["restart"],
                "compatibility": {
                    **admitted["restart"]["compatibility"],
                    "current_commit": parent["component_commit"],
                },
            },
        },
    }
    require(
        identity == lineage["type"] == second["qualification_type"]
        and case == admitted["case"]
        and parent["qualification_schema"] == identity["schema"]
        and parent["anchor_stage"] == {"kind": identity["kind"]}
        and lineage == historical
        and parent["original_acceptance_changed"] is False,
        "cold-target type or original admission differs",
    )
    require(
        not (runtime / "failure.json").exists()
        and prefix.completion_inventory(runtime) == files
        and completion["schema"] == identity["schema"]
        and completion["status"] == "complete"
        and completion["numerical_acceptance"] is True
        and completion["type"] == identity
        and completion["stationary_stage_accepted"] is True
        and completion["anchor_accepted"] is False
        and completion["original_acceptance_changed"] is False,
        "cold-target completion differs",
    )
    preparation = runner._read_json(runtime / "preparation.json")
    require(
        preparation["lineage"] == lineage
        and preparation["inputs"] == admitted["records"]
        and preparation["implementation"] == parent["implementation"]
        and preparation["numerical_acceptance"] is False,
        "cold-target preparation differs",
    )
    result = runner._read_json(runtime / "result.json")
    require(
        result["schema"] == identity["schema"]
        and result["type"] == identity
        and result["lineage"] == lineage
        and result["stationary_stage_accepted"] is True
        and result["anchor_accepted"] is False
        and result["original_attempt_status"] == "failed"
        and result["original_acceptance_changed"] is False
        and result["numerical_acceptance"] is True
        and result["cold_load_verified"] is True
        and summary["stationary_stage_accepted"] is True
        and summary["type"] == identity
        and summary["lineage"] == lineage
        and summary["anchor_accepted"] is False,
        "stationary-only result differs",
    )
    require(
        all(value.get(name) is False for value in (parent, result, summary) for name in NO_CLAIMS),
        "cold target claims an unestablished certificate",
    )
    methods = audit_module()
    first = runner._read_json(runtime / "parent-cold-load.json")
    expected = admitted["restart"]["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
    pins = admitted["sample"]["actual_pins"]
    records = list(admitted["records"])
    for receipt, packet in ((first, admitted["restart"]), (second, restart)):
        suffix.validate_typed_receipt(receipt, packet, case, pins, expected, runtime)
        records.extend(methods.bind_typed_proof_inventory(receipt, runtime, files))
    require(
        Path(first["intervention"]["after"]["path"]).parent
        != Path(second["intervention"]["after"]["path"]).parent
        and second["packet"] == restart["packet"]
        and second["parent_manifest"] == restart["parent_manifest"],
        "two distinct fresh loads or descendant packet differ",
    )
    require(
        restart["parent_manifest"] == record(runtime / "run-manifest.json")
        and restart["stage"] == admitted["restart"]["stage"]
        and cold.normalized_bytes(
            Path(restart["state"]["path"]).read_bytes(), Path(restart["mesh_reference_base"])
        )
        == cold.normalized_bytes(
            Path(admitted["restart"]["state"]["path"]).read_bytes(),
            Path(admitted["restart"]["mesh_reference_base"]),
        ),
        "portable descendant changed full state/history",
    )
    for item in (restart["packet"], restart["state"], *restart["meshes"]):
        records.append(methods.bound_parent_file(item, runtime, files))
    require(
        runtime / prefix.relative_path(result["checkpoint"]["resume_packet"])
        == Path(restart["packet"]["path"]),
        "cold-target result checkpoint differs",
    )
    checkpoint_snapshots = [
        snapshot_proof(runtime, phase, files) for phase in ("before-checkpoint", "after-checkpoint")
    ]
    require(
        checkpoint_snapshots[0][1]
        == checkpoint_snapshots[1][1]
        == cold.normalized_bytes(
            Path(restart["state"]["path"]).read_bytes(), Path(restart["mesh_reference_base"])
        ),
        "native checkpoint full history changed",
    )
    for _, _, _, items in checkpoint_snapshots:
        records.extend(items)
    expected_guard = {
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
        require(
            runner._read_json(runtime / name) == expected_guard,
            "cold-target zero-solve guard differs",
        )
    for phase, receipt in (("prefix", first), ("cold-load", second)):
        sample = runner._read_json(runtime / "samples" / (phase + ".json"))
        require(
            sample["sampling_preserved_state"] is True
            and sample["actual_pins"] == pins
            and methods.physical_signature(sample["signature"])
            == methods.physical_signature(receipt["loaded"]),
            "fresh cold-target sampled state differs",
        )
        before = snapshot_proof(runtime, phase + "-before-sampling", files)
        after = snapshot_proof(runtime, phase + "-after-sampling", files)
        records.extend([*before[3], *after[3]])
        require(
            before[0] == after[0]
            and before[2] == after[2] == sample["signature"]["serialized_state"],
            "cold-target sampling changed full native history",
        )
        checks = remesh.canonical_checks(case, sample["metrics"])
        require(
            set(checks) == prefix.GATES
            and all(checks.values())
            and checks == result["checks" if phase == "prefix" else "cold_load_checks"],
            "ten original cold-target predicates differ",
        )
    require(
        second["checks"] == second["canonical_checks"] == summary["cold_load_checks"]
        and summary["checks"] == result["checks"]
        and second["checks"] == result["cold_load_checks"],
        "cold-target summary gates differ",
    )
    return records


if __name__ == "__main__":
    raise SystemExit(main())
