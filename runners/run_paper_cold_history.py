"""Diagnose a cold-load serialization mismatch without evolving the saved state.

Only the saved native run-statement endtime is restored. Complete snapshots,
the unchanged full-history fingerprint and the original five case predicates
are separate evidence. The failed producer remains unqualified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems import paper_bubble as model  # noqa: E402 - standalone entrypoint path bootstrap
from runners import run_paper_case as runner  # noqa: E402
from runners import run_paper_seed_i as seed  # noqa: E402
from runners import run_pressure_diagnostic as diagnostic  # noqa: E402
from runners import run_pressure_remesh as remesh  # noqa: E402

require, sha, write_json = diagnostic.require, diagnostic.sha, diagnostic.write_json
PARENT_SOURCE = "cc2dcfe644169f7c2a4ca8353f5743dc287afbd1"
ENTRYPOINT = "runners/run_paper_cold_history.py"
HELPERS = (
    "runners/run_paper_anchor.py",
    "runners/run_paper_seed_i.py",
    "runners/run_paper_seed_recovery.py",
    "runners/run_pressure_remesh.py",
    "runners/run_pressure_diagnostic.py",
)
API = {
    "pyoomph_commit": diagnostic.PYOOMPH,
    "defaults": "pyoomph/generic/problem.py:1108-1115",
    "header": "pyoomph/generic/problem.py:10218-10243; output/states.py:241-273",
    "dt_history": "pyoomph/generic/problem.py:10315-10335",
    "endtime_loader_writer": "pyoomph/generic/problem.py:10341-10349",
    "float_encoding": "pyoomph/output/states.py:136-141",
    "snapshot": "pyoomph/generic/problem.py:10769-10813",
}


def serialized_metadata(snapshot):
    """Independently parse the pinned prefix; never search for a scalar's bytes.

    No array, pickle or native code is decoded. All following serialized data
    remain opaque and participate in the complete byte comparisons.
    """
    stream = io.BytesIO(snapshot)

    def read(count):
        value = stream.read(count)
        require(len(value) == count, "truncated native metadata")
        return value

    def integer():
        return struct.unpack("<q", read(8))[0]

    def scalar():
        return struct.unpack("<d", read(8))[0]

    def string():
        length = integer()
        require(0 <= length <= 1024 * 1024, "invalid native metadata string")
        return read(length).decode("ascii")

    require(
        (string(), string(), string()) == ("pyoomph_dump", "0.1.6", "global"),
        "unsupported native metadata format",
    )
    require(snapshot.endswith(struct.pack("<q", 11) + b"EOF_pyoomph"), "missing native footer")
    time = scalar()
    output_step, continue_step, compression = integer(), integer(), integer()
    count = integer()
    require(1 <= count <= 128, "invalid native template count")
    meshes = []
    for _ in range(count):
        start = stream.tell()
        reference = string()
        end = stream.tell()
        if reference:
            require("\0" not in reference and Path(reference).suffix == ".msh", "invalid mesh name")
            meshes.append((start, end, reference))
        remesher = integer()
        require(remesher in (0, 1), "invalid native remesher flag")
        if remesher:
            require(integer() >= 0, "invalid native remesher counter")
    count = integer()
    require(1 <= count <= 128, "invalid native dt count")
    dts = [scalar() for _ in range(count)]
    steps = integer()
    suggested = scalar()
    offset = stream.tell()
    endtime = scalar()
    require(math.isfinite(time), "nonfinite saved time")
    require(all(math.isfinite(dt) and dt > 0 for dt in dts), "invalid saved dt history")
    require(steps >= 0, "invalid unsteady step count")
    require(
        math.isnan(suggested) or (math.isfinite(suggested) and suggested > 0), "invalid next dt"
    )
    return {
        "time": time,
        "output_step": output_step,
        "continue_step": continue_step,
        "compression": compression,
        "dts": dts,
        "unsteady_steps": steps,
        "suggested_next_dt": None if math.isnan(suggested) else suggested,
        "endtime": None if math.isnan(endtime) else endtime,
        "endtime_offset": offset,
        "meshes": meshes,
    }


def normalized_bytes(snapshot, base):
    """Replace only mesh-reference strings with content identities, including the footer."""
    parts, cursor = [], 0
    for start, end, reference in serialized_metadata(snapshot)["meshes"]:
        path = (base / reference).resolve(strict=True)
        require(path.is_file() and not path.is_symlink(), "missing snapshot mesh")
        identity = ("sha256:" + sha(path)).encode("ascii")
        parts.extend((snapshot[cursor:start], struct.pack("<q", len(identity)), identity))
        cursor = end
    parts.append(snapshot[cursor:])
    return b"".join(parts)


def restore_endtime(problem, saved, before, case):
    """Guard the actual loader/writer asymmetry before assigning one metadata attribute."""
    parent, cold = serialized_metadata(saved), serialized_metadata(before)
    expected = parent["endtime"]
    require(
        isinstance(expected, float)
        and math.isfinite(expected)
        and expected == case["solve"]["relax_time"],
        "saved endtime differs from declared preparation",
    )
    require(
        "_loaded_run_statement_endtime" in vars(problem)
        and "_run_statement_endtime" in vars(problem),
        "native endtime attributes unavailable",
    )
    loaded = problem._loaded_run_statement_endtime
    require(
        isinstance(loaded, float) and struct.pack("<d", loaded) == struct.pack("<d", expected),
        "native loaded endtime differs from independently parsed scalar",
    )
    require(problem._run_statement_endtime is None, "fresh native writer endtime is not unset")
    require(cold["endtime"] is None, "fresh snapshot endtime is not NaN")
    problem._run_statement_endtime = loaded
    return expected


def verify_snapshots(saved, before, after, parent_base, cold_base, expected_hash):
    """Require the sole scalar change and strict complete parent equality."""
    old, new = serialized_metadata(before), serialized_metadata(after)
    offset = old["endtime_offset"]
    require(new["endtime_offset"] == offset, "native metadata slot moved")
    require(
        len(before) == len(after)
        and before[:offset] == after[:offset]
        and before[offset + 8 :] == after[offset + 8 :]
        and before[offset : offset + 8] != after[offset : offset + 8],
        "snapshots differ outside the exact endtime scalar",
    )
    parent = serialized_metadata(saved)
    parent_offset = parent["endtime_offset"]
    require(
        after[offset : offset + 8] == saved[parent_offset : parent_offset + 8],
        "restored endtime bytes differ from parent",
    )
    require(
        normalized_bytes(after, cold_base) == normalized_bytes(saved, parent_base),
        "complete mesh-normalized native bytes differ from parent",
    )
    fingerprint = runner.snapshot_bc_cache_signature(after, mesh_reference_base=cold_base)
    require(fingerprint["payload_sha256"] == expected_hash, "cold-loaded complete history differs")
    return {"verified": True, "only_changed_slot": [offset, offset + 8], "after": fingerprint}


def source_records(parent, component):
    """Bind the exact recovery producer, unchanged methods/helpers and committed diagnostic."""
    require(
        parent["component_commit"] == parent["component"]["commit"] == PARENT_SOURCE,
        "unexpected recovery parent source",
    )
    runner.require_clean_source(parent["component"])
    runner.require_clean_source(component)
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", PARENT_SOURCE, component["commit"])
    old = parent["component"]["sources"]
    require(set(old) == set(component["sources"]), "parent method inventory differs")
    records = {}
    for relative in (*old, *HELPERS, seed.BASE_CASE, ENTRYPOINT):
        path = ROOT / relative
        digest = hashlib.sha256(
            diagnostic.git(ROOT, "show", f"{component['commit']}:{relative}")
        ).hexdigest()
        require(
            path.is_file() and not path.is_symlink() and sha(path) == digest, "source bytes drift"
        )
        if relative != ENTRYPOINT:
            original = hashlib.sha256(
                diagnostic.git(ROOT, "show", f"{PARENT_SOURCE}:{relative}")
            ).hexdigest()
            require(digest == original, f"parent method/helper changed: {relative}")
            if relative in old:
                require(
                    old[relative]["committed_matches"] is True
                    and old[relative]["sha256"] == old[relative]["committed_sha256"] == original,
                    "parent method provenance differs",
                )
            else:
                require(
                    parent["implementation"][relative]["sha256"] == original,
                    "parent helper provenance differs",
                )
        records[relative] = {"path": str(path), "sha256": digest}
    producer = records["runners/run_paper_seed_recovery.py"]["sha256"]
    require(
        parent["runner"]["sha256"] == parent["runner_sha256"] == producer
        and Path(parent["runner"]["path"]).name == "run_paper_seed_recovery.py",
        "parent recovery producer differs",
    )
    return records


def preflight(args):
    """Bind one failed recovery packet without importing pyoomph or creating output."""
    require(args.source.resolve(strict=True) == ROOT, "runner must use its own isolated component")
    for module, relative in (
        (model, "problems/paper_bubble.py"),
        (runner, "runners/run_paper_case.py"),
        (seed, "runners/run_paper_seed_i.py"),
        (diagnostic, "runners/run_pressure_diagnostic.py"),
        (remesh, "runners/run_pressure_remesh.py"),
    ):
        require(Path(module.__file__).resolve() == ROOT / relative, "helper imported elsewhere")
    manifest_path = args.manifest.resolve(strict=True)
    parent = runner._read_json(manifest_path)
    component = runner.component_revision(ROOT)
    implementation = source_records(parent, component)
    environment = runner.pyoomph_revision()
    seed.validate_environment(environment)
    seed.validate_environment(parent["pyoomph"])
    require(
        environment["interpreter"]["resolved_executable"]
        == parent["pyoomph"]["interpreter"]["resolved_executable"],
        "parent interpreter differs",
    )
    case = runner.read_case(args.case)
    seed.validate_case(case, runner.read_case(ROOT / seed.BASE_CASE))
    require(sha(args.case) == parent["case_sha256"], "original case bytes differ")
    restart = runner.validate_restart(
        args.packet,
        case,
        sha(args.case),
        component=component,
        environment=environment,
        azimuthal_stability=True,
    )
    require(
        Path(restart["parent_manifest"]["path"]) == manifest_path
        and Path(restart["state"]["path"]) == args.state.resolve(strict=True),
        "packet binds another parent/state",
    )
    require(
        restart["stage"]["kind"] == "completed_polish"
        and restart["stage"]["bond_index"] == len(model.stationary_bond_schedule(case)) - 1
        and restart["state"]["format"] == "portable_file",
        "final portable target checkpoint required",
    )
    runtime = manifest_path.parent
    require(
        Path(restart["packet"]["path"]).parent
        == runtime / "checkpoints" / f"bond_{restart['stage']['bond_index']:03d}",
        "packet is not the terminal recovery checkpoint",
    )
    failure_path = runtime / "failure.json"
    failure = runner._read_json(failure_path)
    require(
        failure["status"] == "failed"
        and failure["numerical_acceptance"] is False
        and failure["error_type"] == "ValueError"
        and failure["error"] == "cold-loaded complete history differs"
        and failure["inputs_unchanged"] is True,
        "parent is not the retained complete-history failure",
    )
    require(
        not any(
            (runtime / name).exists()
            for name in ("completion.json", "result.json", "summary.json", "cold-load.json")
        ),
        "parent terminal status differs",
    )
    sample_path = runtime / "terminal/samples/after-polish.json"
    sample = runner._read_json(sample_path)
    require(sample["sampling_preserved_state"] is True, "parent sample did not preserve state")
    require(
        all(remesh.canonical_checks(case, sample["metrics"]).values()), "parent five gates fail"
    )
    expected = restart["roundtrip"]["after"]
    require(
        sample["signature"]["arrays"] == expected["arrays"]
        and sample["signature"]["layout"] == expected["layout"]
        and sample["signature"]["serialized_state"]["payload_sha256"]
        == expected["serialized_state"]["payload_sha256"],
        "parent sample and checkpoint differ",
    )
    saved = Path(restart["state"]["path"]).read_bytes()
    metadata = serialized_metadata(saved)
    require(metadata["endtime"] == case["solve"]["relax_time"], "saved preparation endtime differs")
    fingerprint = runner.snapshot_bc_cache_signature(
        saved, mesh_reference_base=Path(restart["mesh_reference_base"])
    )
    require(
        fingerprint["payload_sha256"] == expected["serialized_state"]["payload_sha256"],
        "parent dump differs from hash-bound complete history",
    )
    records = [restart["packet"], restart["parent_manifest"], restart["state"], *restart["meshes"]]
    for path in (args.case, failure_path, sample_path, Path(parent["case_path"])):
        require(path.is_file() and not path.is_symlink(), "nonregular parent input")
        records.append({"path": str(path.resolve()), "sha256": sha(path)})
    final = runtime / "checkpoints/final.dump"
    if final.exists():
        require(final.is_file() and not final.is_symlink(), "nonregular parent final dump")
        require(
            normalized_bytes(final.read_bytes(), final.parent)
            == normalized_bytes(saved, Path(restart["mesh_reference_base"])),
            "parent final and checkpoint dump differ",
        )
        records.append({"path": str(final), "sha256": sha(final)})
    return {
        "case": case,
        "restart": restart,
        "component": component,
        "environment": environment,
        "implementation": implementation,
        "records": records,
        "sample": sample,
        "metadata": metadata,
    }


def stable_inputs(prepared):
    remesh.verify_inputs(prepared["records"])
    remesh.verify_source(prepared["component"], prepared["implementation"])


def check_time_history(problem, saved):
    """Compare live time-step storage with independently parsed native metadata."""
    time = problem.timestepper.time_pt()
    actual = {
        "time": float(problem.get_current_time(dimensional=True, as_float=True)),
        "dts": [float(time.dt(i)) for i in range(time.ndt())],
        "unsteady_steps": problem.timestepper.get_num_unsteady_steps_done(),
        "suggested_next_dt": problem._suggested_next_dt,
    }
    require(
        actual == {key: saved[key] for key in actual}, "live time/dt history differs from parent"
    )
    return actual


def retain_snapshot(problem, output, phase):
    """Retain unmodified native bytes and copies of each referenced mesh."""
    setting = problem.eigen_data_in_states
    try:
        problem.eigen_data_in_states = True
        snapshot = problem._snapshot_state()
    finally:
        problem.eigen_data_in_states = setting
    path = output / f"{phase}.dump"
    diagnostic.atomic(path, snapshot)
    meshes = []
    directory = output / f"{phase}-meshes"
    directory.mkdir()
    for index, reference in enumerate(runner.state_mesh_references(path)):
        original = (Path.cwd() / reference).resolve(strict=True)
        retained = directory / f"{index:03d}.msh"
        diagnostic.atomic(retained, original.read_bytes())
        require(sha(original) == sha(retained), "snapshot mesh retention differs")
        meshes.append({"stored_path": reference, "path": str(retained), "sha256": sha(retained)})
    write_json(
        output / f"{phase}-retention.json",
        {
            "state": {"path": str(path), "sha256": sha(path), "format": "memory_snapshot"},
            "mesh_reference_base": str(Path.cwd()),
            "meshes": meshes,
            "mesh_exchange": {mesh["stored_path"]: mesh["path"] for mesh in meshes},
        },
    )
    return snapshot


def execute(args, prepared, output):
    import numpy as np

    case, restart = prepared["case"], prepared["restart"]
    stable_inputs(prepared)
    saved = Path(restart["state"]["path"]).read_bytes()
    with model.build_problem(case, initial_bond=case["physics"]["bond"]) as problem:
        problem.set_output_directory(str(output / "solver"))
        problem.set_linear_solver("superlu")
        problem.set_c_compiler("system")
        problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        problem.initialise()
        runner.verify_restart_artifacts(restart)
        problem.load_state(
            restart["state"]["path"],
            quiet=True,
            additional_info={"exchange_msh_file": restart["mesh_exchange"]},
        )
        before = retain_snapshot(problem, output, "before")
        signature = model._stationary_state_signature(problem, True)
        pins = diagnostic.actual_pins(problem)
        times = check_time_history(problem, prepared["metadata"])
        write_json(
            output / "before-observation.json",
            {
                "signature": signature,
                "actual_pins": pins,
                "time_history": times,
            },
        )
        expected = restart["roundtrip"]["after"]["serialized_state"]["payload_sha256"]
        # This comparison intentionally excludes no history bytes. The prior failure must recur.
        require(
            signature["serialized_state"]["payload_sha256"] != expected,
            "original full-history mismatch did not recur",
        )
        restored = restore_endtime(problem, saved, before, case)
        after = retain_snapshot(problem, output, "after")
        after_signature = model._stationary_state_signature(problem, True)
        after_pins = diagnostic.actual_pins(problem)
        write_json(
            output / "after-observation.json",
            {
                "signature": after_signature,
                "actual_pins": after_pins,
                "restored_endtime": restored,
                "time_history": check_time_history(problem, prepared["metadata"]),
            },
        )
        mechanism = verify_snapshots(
            saved, before, after, Path(restart["mesh_reference_base"]), Path.cwd(), expected
        )
        require(
            {k: v for k, v in signature.items() if k != "serialized_state"}
            == {k: v for k, v in after_signature.items() if k != "serialized_state"}
            and pins == after_pins,
            "metadata restoration changed current state/pins",
        )
        remesh.check_loaded_state(
            after_signature, restart, case, after_pins, prepared["sample"]["actual_pins"]
        )
        require(
            after_signature["serialized_state"]["payload_sha256"] == expected,
            "cold-loaded complete history differs",
        )
        mechanism.update(api=API, parent_acceptance_changed=False)
        write_json(output / "mechanism.json", mechanism)
        # Observables temporarily assemble a stationary residual; sample proves complete preservation.
        try:
            sample = remesh.sample(problem, model, case, output, "cold-load", np)
        finally:
            sampled = retain_snapshot(problem, output, "after-sampling")
        require(sampled == after, "five-gate sampling changed complete native snapshot")
        require(problem._run_statement_endtime == restored, "sampling changed restored metadata")
        checks = remesh.canonical_checks(case, sample["metrics"])
        write_json(
            output / "numerical-gates.json",
            {
                "checks": checks,
                "sampling_preserved_complete_state": True,
                "scope": "original five predicates on this cold-loaded observation only",
                "parent_acceptance_changed": False,
            },
        )
        require(all(checks.values()), "original cold-load numerical gates failed")
    stable_inputs(prepared)
    return {
        "mechanism_verified": True,
        "checks": checks,
        "numerical_acceptance": True,
        "parent_status": "failed",
        "parent_acceptance_changed": False,
        "branch_identity_verified": False,
        "eigenanalysis_performed": False,
        "jacobian_certificate": False,
        "full_comparison_verified": False,
    }


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    require(not args.out.is_symlink(), "output must not be a symlink")
    output = runner.validate_output_path(args.out, ROOT)
    runtime = Path(prepared["restart"]["parent_manifest"]["path"]).parent
    require(
        not output.is_relative_to(runtime) and not runtime.is_relative_to(output),
        "output overlaps failed evidence",
    )
    if args.check:
        print('{"verified": true, "native_imported": false, "output_created": false}')
        return 0
    actual = remesh.run_manifest(
        args,
        prepared["case"],
        prepared["restart"],
        prepared["environment"],
        prepared["component"],
        prepared["implementation"],
    )
    actual.update(
        runner=prepared["implementation"][ENTRYPOINT],
        runner_path=str(ROOT / ENTRYPOINT),
        runner_sha256=prepared["implementation"][ENTRYPOINT]["sha256"],
        case_fields_scope="unchanged finite-gas construction and original five case gates; no polish",
        argv=[sys.executable, str(ROOT / ENTRYPOINT), *arguments],
        intervention={
            "kind": "cold-load-run-endtime-metadata-only",
            "api": API,
            "solve_calls": 0,
            "remesh_calls": 0,
            "time_evolution": False,
            "parent_acceptance_changed": False,
        },
        qualification_scope="one cold-load diagnostic and original case gates; parent remains failed",
    )
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    runner.create_run_manifest(output, actual)
    runner.append_progress(output, {"event": "prepared", "runner": actual["runner"]}, first=True)
    try:
        write_json(
            output / "preparation.json",
            {
                "inputs": prepared["records"],
                "implementation": prepared["implementation"],
                "api": API,
                "parent_status": "failed",
                "numerical_acceptance": False,
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
                "scientific_status": "checks_passed",
                "completion_reached": True,
            },
        )
        files = {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "progress.jsonl"
        }
        write_json(
            output / "completion.json",
            {
                "status": "complete",
                "files": files,
                "numerical_acceptance": True,
                "scope": actual["qualification_scope"],
            },
        )
        runner.append_progress(output, {"event": "final", **result})
        return 0
    except BaseException as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "inputs_unchanged": diagnostic.parents_unchanged(prepared["records"]),
            "numerical_acceptance": False,
            "parent_acceptance_changed": False,
        }
        write_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", **failure})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
