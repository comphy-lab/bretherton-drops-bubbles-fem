"""Invoke one checksum-bound worker from a private run configuration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import re
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PYTHON = ROOT.parent / "bretherton-drops-bubbles-fem/.venv/bin/python"
SAVED_INTERFACE = "saved-paper-spectrum-v1"
PACKET_SCHEMA = "stone-paper-saved-spectrum-packet-v1"
CHILD_SCHEMA = "stone-paper-saved-spectrum-output-v1"
PRODUCER = "bf704cc07c964daa800abcc09b1d6de611ae80fb"
PIN = "a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618"
CORE = "af7334e81f38453d6852581b6c208e93a0648c6887a6766b0fe63acd2ab51c0d"
CLOSURE = {"saved_search.py", "admission.py", "numerics.py", "geometry_volume.py"}
INPUT_ROLES = {
    "j",
    "m",
    "layout",
    "eigen_plus",
    "eigen_minus",
    "case",
    "native_manifest",
    "native_completion",
    "preparation",
    "reference",
    "gas_geometry",
    "liquid_geometry",
    "volume_w4",
    "volume_identity",
    "volume_provenance",
    "volume_restoration",
    "strategy",
    "core",
    "source_modes",
    "source_model",
    "source_bubble",
    "source_spectra",
    *(f"pair_plus_{index:02d}" for index in range(8)),
}
NO_CLAIMS = {
    "scientific_acceptance": False,
    "numerical_acceptance": False,
    "finite_spectrum_complete": False,
    "global_dominance_certified": False,
    "physical_mode_qualification": False,
    "native_imports": False,
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class NativeDeny(importlib.abc.MetaPathFinder):
    """Keep the envelope metadata-only; the isolated child has its own guard."""

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith(("pyoomph.", "_pyoomph")):
            self.attempts.append(fullname)
            raise ImportError("native solver import denied: " + fullname)

    def check(self):
        cached = [
            name
            for name in sys.modules
            if name == "pyoomph" or name.startswith(("pyoomph.", "_pyoomph"))
        ]
        require(not cached and not self.attempts, "native-import invariant failed")


def regular_path(value, *, directory=False, missing=False):
    path = Path(os.path.abspath(value))
    require(
        all(not item.is_symlink() for item in (path, *path.parents)), "symlink in authority path"
    )
    if missing and not path.exists():
        require(path.parent.is_dir(), "output parent must already exist")
        return path
    mode = path.stat().st_mode
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode), "nonregular authority path")
    return path


def relative_path(value):
    require(isinstance(value, str) and value, "relative authority path must be text")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe relative authority path")
    return path


def file_record(path, expected=None, size=None):
    path = regular_path(path)
    if expected is not None:
        require(
            isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected), "invalid SHA-256"
        )
    if size is not None:
        require(type(size) is int and size >= 0, "invalid artifact size")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), "opened authority is nonregular")
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "authority changed during read",
    )
    require(expected is None or checksum == expected, "artifact checksum differs")
    require(size is None or after.st_size == size, "artifact size differs")
    return {"path": str(path), "sha256": checksum, "bytes": after.st_size}


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def json_object(raw):
    def nonfinite(value):
        raise ValueError("nonfinite JSON number: " + value)

    value = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=nonfinite)
    require(isinstance(value, dict), "JSON authority must be an object")
    json.dumps(value, allow_nan=False)
    return value


def read_json(path):
    path = regular_path(path)
    before = file_record(path)
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == before["sha256"], "JSON authority changed during read"
    )
    return json_object(raw)


def same_json(first, second):
    return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )


def descriptor(record, base, *, contained=False):
    require(
        isinstance(record, dict) and set(record) == {"path", "sha256"},
        "path/hash descriptor differs",
    )
    require(isinstance(record["path"], str) and record["path"], "authority path must be text")
    path = Path(record["path"])
    if not path.is_absolute():
        path = base / relative_path(record["path"])
    actual = file_record(path, record["sha256"])
    require(
        not contained or Path(actual["path"]).is_relative_to(base),
        "authority escaped configuration root",
    )
    return actual


def metadata_helpers(guard):
    guard.check()
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from problems import paper_bubble as model
    from runners import run_paper_case as runner

    require(
        Path(model.__file__).resolve() == ROOT / "problems/paper_bubble.py"
        and Path(runner.__file__).resolve() == ROOT / "runners/run_paper_case.py",
        "metadata helpers imported from another component",
    )
    guard.check()
    return model, runner


def source_metadata(guard):
    model, runner = metadata_helpers(guard)
    component = runner.component_revision(ROOT)
    runner.require_clean_source(component)
    own = file_record(__file__)
    committed = runner._git(ROOT, "show", component["commit"] + ":runners/run_bound_worker.py")
    require(
        committed is not None and hashlib.sha256(committed).hexdigest() == own["sha256"],
        "wrapper bytes differ from clean component commit",
    )
    environment = runner.pyoomph_revision()
    runner.require_pinned_environment(environment)
    require(
        environment["commit"] == PIN and environment["native_core"]["sha256"] == CORE,
        "environment source/core differs",
    )
    require(
        sys.executable == str(CANONICAL_PYTHON)
        and environment["interpreter"]["resolved_executable"] == str(CANONICAL_PYTHON.resolve()),
        "outer interpreter is not the canonical shared interpreter",
    )
    guard.check()
    return model, runner, component, environment, own


def empty_output(path, prepared=None):
    path = regular_path(path, directory=True)
    require(not any(path.iterdir()), "parent runtime must exist and be empty; reuse is refused")
    require(not path.is_relative_to(ROOT), "output overlaps component source")
    if prepared is not None:
        require(
            not path.is_relative_to(prepared["root"]) and not prepared["root"].is_relative_to(path),
            "output overlaps configuration or packet custody",
        )
        require(
            all(not Path(item["path"]).is_relative_to(path) for item in prepared["records"]),
            "output contains a bound original input",
        )
    return path


def saved_bindings(args, guard):
    config_path = regular_path(args.configuration)
    root = config_path.parent
    config = read_json(config_path)
    require(
        set(config) == {"case", "entrypoint", "files", "worker_interface", "packet"}
        and config["worker_interface"] == SAVED_INTERFACE,
        "saved worker configuration schema differs",
    )
    case_path = regular_path(args.case)
    require(
        isinstance(config["case"], str) and regular_path(config["case"]) == case_path,
        "case selector differs from configuration",
    )
    dispatch_case = read_json(case_path)
    configuration = descriptor(dispatch_case.get("configuration"), case_path.parent)
    require(Path(configuration["path"]) == config_path, "dispatch case binds another configuration")
    packet_record = descriptor(config["packet"], root, contained=True)
    packet_path = Path(packet_record["path"])
    packet = read_json(packet_path)
    require(
        packet.get("schema") == PACKET_SCHEMA
        and set(packet)
        == {
            "schema",
            "authority",
            "environment",
            "inputs",
            "source_closure",
            "model",
            "target",
            "gates",
            "resource_caps",
            "algorithm",
            "reference_rectangles",
            "protected_roots",
        },
        "saved packet schema differs",
    )
    authority = packet["authority"]
    require(
        authority.get("source_commit") == PRODUCER
        and authority.get("pyoomph_commit") == PIN
        and authority.get("native_core_sha256") == CORE
        and authority.get("pencil_sign") == "J=-dR/dU; Jv=lambda Mv",
        "saved producer authority differs",
    )
    require(
        packet["environment"]["interpreter"] == str(CANONICAL_PYTHON)
        and packet["environment"]["resolved_interpreter"] == str(CANONICAL_PYTHON.resolve()),
        "packet interpreter substitution is refused",
    )
    require(
        set(packet["source_closure"]) == CLOSURE and isinstance(config["files"], dict),
        "four-file closure required",
    )
    records = [file_record(case_path), configuration, packet_record]
    closure_files = {}
    for name, declaration in packet["source_closure"].items():
        require(set(declaration) == {"sha256", "bytes"}, "source closure declaration differs")
        item = file_record(packet_path.parent / name, declaration["sha256"], declaration["bytes"])
        require(
            Path(item["path"]).is_relative_to(root), "source closure escaped configuration root"
        )
        closure_files[str(Path(item["path"]).relative_to(root))] = item["sha256"]
        records.append(item)
    require(config["files"] == closure_files, "configuration source closure differs")
    worker = regular_path(root / relative_path(config["entrypoint"]))
    require(
        worker == packet_path.parent / "saved_search.py",
        "only the packet saved_search.py entrypoint is allowed",
    )
    require(set(packet["inputs"]) == INPUT_ROLES, "exactly the thirty bound input roles required")
    witness = {}
    for role, declaration in packet["inputs"].items():
        require(
            set(declaration) == {"path", "sha256", "bytes", "kind"}
            and declaration["kind"] in {"npz", "json", "source", "binary", "text"},
            "input declaration differs",
        )
        require(isinstance(declaration["path"], str), "input path must be text")
        path = Path(declaration["path"])
        if not path.is_absolute():
            path = packet_path.parent / relative_path(declaration["path"])
        item = file_record(path, declaration["sha256"], declaration["bytes"])
        witness[role] = item
        records.append(item)
    physical_case = {key: value for key, value in dispatch_case.items() if key != "configuration"}
    require(
        same_json(physical_case, packet["model"])
        and same_json(read_json(witness["case"]["path"]), packet["model"]),
        "dispatch/original case differs from packet model",
    )
    model, runner, component, environment, own = source_metadata(guard)
    require(
        same_json(model.validate_case(physical_case), physical_case),
        "original physical case validation differs",
    )
    prepared = {
        "root": root,
        "config": config,
        "packet": packet,
        "packet_record": packet_record,
        "worker": worker,
        "case": dispatch_case,
        "physical_case": physical_case,
        "records": records,
        "witness": witness,
        "model": model,
        "runner": runner,
        "component": component,
        "environment": environment,
        "own": own,
    }
    prepared["output"] = empty_output(args.output, prepared)
    return prepared


def child_argv(prepared, *, check):
    return [
        sys.executable,
        "-I",
        str(prepared["worker"]),
        "--packet",
        prepared["packet_record"]["path"],
        "--packet-sha256",
        prepared["packet_record"]["sha256"],
        "--output",
        str(prepared["output"] / "result"),
        "--check" if check else "--execute",
        "--shift",
        "0.001",
    ]


def stable_bindings(prepared, guard):
    guard.check()
    for item in prepared["records"]:
        require(
            file_record(item["path"], item["sha256"], item["bytes"]) == item,
            "bound authority changed",
        )
    _, _, component, environment, own = source_metadata(guard)
    require(
        component == prepared["component"]
        and environment == prepared["environment"]
        and own == prepared["own"],
        "wrapper source or environment changed",
    )


def stable_bytes_after_child(prepared, guard):
    """Rehash execution bindings without claiming fresh post-run Git cleanliness."""
    guard.check()
    for item in prepared["records"]:
        require(
            file_record(item["path"], item["sha256"], item["bytes"]) == item,
            "bound authority changed",
        )
    _, runner = metadata_helpers(guard)
    for relative, source in prepared["component"]["sources"].items():
        file_record(ROOT / relative_path(relative), source["sha256"])
    own = file_record(__file__)
    environment = runner.pyoomph_revision()
    runner.require_pinned_environment(environment)
    require(
        own == prepared["own"]
        and environment == prepared["environment"]
        and sys.executable == str(CANONICAL_PYTHON)
        and environment["interpreter"]["resolved_executable"] == str(CANONICAL_PYTHON.resolve()),
        "wrapper source or environment changed",
    )
    guard.check()


def metadata_check(prepared, guard):
    done = subprocess.run(
        child_argv(prepared, check=True), check=False, capture_output=True, text=True
    )
    require(done.returncode == 0, "isolated child metadata check failed: " + done.stderr)
    checked = json_object(done.stdout)
    require(
        checked.get("status") == "metadata_checked_no_execution"
        and checked.get("packet_sha256") == prepared["packet_record"]["sha256"]
        and checked.get("source_closure") == prepared["packet"]["source_closure"]
        and checked.get("input_witness") == prepared["witness"]
        and checked.get("output_absent") is True,
        "child metadata authority differs",
    )
    require(
        checked.get("native_imports") is False
        and type(checked.get("factorisations")) is int
        and checked["factorisations"] == 0
        and type(checked.get("eigensolves")) is int
        and checked["eigensolves"] == 0,
        "child metadata imported native solver or executed numerics",
    )
    target = prepared["packet"]["target"]
    require(
        checked.get("n") == target["n"]
        and checked.get("J_nnz") == target["j_nnz"]
        and checked.get("M_nnz") == target["m_nnz"]
        and checked.get("mass_rows") == target["mass_rows"]
        and checked.get("mass_columns") == target["mass_columns"]
        and checked.get("globals_preserved") == target["globals"]
        and checked.get("forced") == target["forced"]
        and checked.get("model_hash") == prepared["witness"]["case"]["sha256"]
        and checked.get("time_factor") == target["time_factor"]
        and checked.get("pencil_sign") == prepared["packet"]["authority"]["pencil_sign"],
        "child checked target metadata differs",
    )
    empty_output(prepared["output"], prepared)
    stable_bindings(prepared, guard)
    return checked


def child_completion(prepared):
    output = regular_path(prepared["output"] / "result", directory=True)
    completion = read_json(output / "completion.json")
    require(
        set(completion)
        == {
            "schema",
            "status",
            "files",
            "scientific_acceptance",
            "finite_spectrum_complete",
            "global_dominance_certified",
            "native_imports",
        }
        and completion["schema"] == CHILD_SCHEMA
        and all(
            completion[key] is False
            for key in (
                "scientific_acceptance",
                "finite_spectrum_complete",
                "global_dominance_certified",
                "native_imports",
            )
        ),
        "child sealed completion schema or claims differ",
    )
    files = completion["files"]
    require(
        isinstance(files, dict)
        and {"result.json", "output-hashes.json", "progress.jsonl"} <= set(files)
        and "completion.json" not in files,
        "child completion inventory differs",
    )
    actual = {}
    for path in output.iterdir():
        item = file_record(path)
        actual[path.name] = item
    require(set(actual) == {*files, "completion.json"}, "unexpected or unsealed child output")
    for name, declaration in files.items():
        require(
            relative_path(name).name == name
            and isinstance(declaration, dict)
            and set(declaration)
            in ({"sha256", "bytes"}, {"sha256", "bytes", "partial_unsealed_file"}),
            "child file declaration differs",
        )
        require(
            actual[name]["sha256"] == declaration["sha256"]
            and actual[name]["bytes"] == declaration["bytes"],
            "child sealed output hash differs",
        )
        require(
            "partial_unsealed_file" not in declaration
            or declaration["partial_unsealed_file"] is True,
            "child partial artifact flag differs",
        )
    hashes = read_json(output / "output-hashes.json")
    require(
        hashes == {key: value for key, value in files.items() if key != "output-hashes.json"},
        "child hash table differs",
    )
    result = read_json(output / "result.json")
    require(
        isinstance(result.get("status"), str)
        and result["status"] == completion["status"]
        and result.get("post_input_hashes_verified") is True
        and result.get("no_native_import") is True
        and result.get("second_shift_automatic") is False,
        "child terminal result authority differs",
    )
    for key in (
        "scientific_acceptance",
        "finite_spectrum_complete",
        "global_dominance_certified",
        "physical_mode_qualification",
        "absence_claim_allowed",
        "exact_root_exclusion_certified",
    ):
        require(result.get(key, False) is False, "child claims an unestablished certificate")
    lines = (output / "progress.jsonl").read_text().splitlines()
    require(lines, "child final progress missing")
    first = json_object(lines[0])
    require(
        first.get("event") == "prepared"
        and first.get("packet_sha256") == prepared["packet_record"]["sha256"],
        "child progress belongs to another packet",
    )
    final = json_object(lines[-1])
    require(
        final.get("event") == "final"
        and final.get("status") == result["status"]
        and final.get("result_sha256") == actual["result.json"]["sha256"],
        "child final progress/result linkage differs",
    )
    return result, actual


def parent_manifest(prepared, arguments):
    case, own = prepared["case"], prepared["own"]
    return {
        "schema": "pyoomph-runner-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "case_file": prepared["records"][0]["path"],
        "case_path": prepared["records"][0]["path"],
        "case_sha256": prepared["records"][0]["sha256"],
        "case_id": case["case_id"],
        "case_fields_used": [*prepared["model"].CASE_FIELDS_USED, "configuration"],
        "units": "nondimensional (sigma=mu_liquid=R0=1)",
        "nondimensionalization": case["nondimensionalization"],
        "argv": [sys.executable, own["path"], *arguments],
        "runner": {"path": own["path"], "sha256": own["sha256"]},
        "runner_path": own["path"],
        "runner_sha256": own["sha256"],
        "component_commit": prepared["component"]["commit"],
        "component": prepared["component"],
        "pyoomph": prepared["environment"],
        "worker_interface": SAVED_INTERFACE,
        "packet": prepared["packet_record"],
        "saved_producer_commit": PRODUCER,
        "child_argv": child_argv(prepared, check=False),
        "qualification_scope": "administrative saved-spectrum envelope; no physical or spectral certificate",
    }


def seal_parent(prepared, child_files):
    output = prepared["output"]
    names = {
        "run-manifest.json",
        "progress.jsonl",
        "child-process.json",
        "result.json",
        "summary.json",
    }
    files = {}
    for path in output.iterdir():
        if path.name == "result":
            require(path == regular_path(path, directory=True), "child output directory differs")
        elif path.name == "monitor.jsonl":
            regular_path(path)
        else:
            require(path.name in names, "unexpected parent envelope artifact")
            files[path.name] = file_record(path)["sha256"]
    require(set(files) == names, "parent envelope artifacts missing")
    for name, expected in child_files.items():
        actual = file_record(output / "result" / name, expected["sha256"], expected["bytes"])
        files["result/" + name] = actual["sha256"]
    require(
        {path.name for path in (output / "result").iterdir()} == set(child_files),
        "child sealed directory inventory changed",
    )
    prepared["runner"]._exclusive_json(
        output / "completion.json",
        {
            "schema": "saved-paper-spectrum-envelope-completion-v1",
            "status": "complete",
            "files": files,
            "completion_reached": True,
            **NO_CLAIMS,
            "excluded_mutable_administrative_files": ["monitor.jsonl"],
            "scope": "listed immutable envelope and fully sealed child artifacts; not full-runtime immutability",
        },
    )


def execute_saved(prepared, checked, arguments, guard):
    output, runner = prepared["output"], prepared["runner"]
    stable_bindings(prepared, guard)
    runner.create_run_manifest(output, parent_manifest(prepared, arguments))
    runner.append_progress(
        output,
        {
            "event": "prepared",
            "case_id": prepared["case"]["case_id"],
            "packet_sha256": prepared["packet_record"]["sha256"],
            "child_output": "result",
            "native_imports": False,
        },
        first=True,
    )
    done = None
    try:
        done = subprocess.run(
            child_argv(prepared, check=False), check=False, capture_output=True, text=True
        )
        process = {
            "returncode": done.returncode,
            "argv": child_argv(prepared, check=False),
            "stdout": done.stdout,
            "stderr": done.stderr,
        }
        runner._exclusive_json(output / "child-process.json", process)
        require(done.returncode == 0, "isolated child exited nonzero: " + str(done.returncode))
        result, child_files = child_completion(prepared)
        transcript = json_object(done.stdout)
        require(
            transcript.get("exit_code") == 0
            and transcript.get("completion") == child_files["completion.json"]["path"]
            and transcript.get("completion_sha256") == child_files["completion.json"]["sha256"],
            "child terminal stdout differs from its sealed completion",
        )
        stable_bytes_after_child(prepared, guard)
        parent_result = {
            "schema": "saved-paper-spectrum-envelope-result-v1",
            "status": "completed",
            "completion_reached": True,
            "administrative_complete": True,
            "child_status": result["status"],
            "child_completion": child_files["completion.json"],
            "child_result": child_files["result.json"],
            "packet": prepared["packet_record"],
            "metadata_check": checked,
            **NO_CLAIMS,
        }
        runner._exclusive_json(output / "result.json", parent_result)
        runner._exclusive_json(
            output / "summary.json",
            {
                "status": "completed",
                "completion_reached": True,
                "scientific_status": "unqualified_saved_spectrum",
                "child_status": result["status"],
                "administrative_complete": True,
                **NO_CLAIMS,
            },
        )
        runner.append_progress(
            output,
            {
                "event": "final",
                "status": "completed",
                "exit_status": 0,
                "case_id": prepared["case"]["case_id"],
                "child_status": result["status"],
                **NO_CLAIMS,
            },
        )
        seal_parent(prepared, child_files)
        return 0
    except (Exception, KeyboardInterrupt, SystemExit) as error:  # noqa: BLE001 - retain truthful terminal failure
        failure = {
            "schema": "saved-paper-spectrum-envelope-failure-v1",
            "status": "failed",
            "completion_reached": False,
            "administrative_complete": False,
            "error_type": type(error).__name__,
            "error": str(error),
            **NO_CLAIMS,
        }
        if done is not None:
            failure["child_returncode"] = done.returncode
            failure["child_stdout"] = done.stdout
            failure["child_stderr"] = done.stderr
        runner._exclusive_json(output / "failure.json", failure)
        runner.append_progress(output, {"event": "failure", **failure})
        return (
            130
            if isinstance(error, KeyboardInterrupt)
            else done.returncode
            if done is not None and done.returncode > 0
            else 1
        )


def saved_main(args, arguments):
    guard = NativeDeny()
    sys.meta_path.insert(0, guard)
    try:
        guard.check()
        prepared = saved_bindings(args, guard)
        checked = metadata_check(prepared, guard)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "metadata_checked_no_execution",
                        "worker_interface": SAVED_INTERFACE,
                        "packet": prepared["packet_record"],
                        "child": checked,
                        "parent_output_written": False,
                        "native_imports": False,
                    },
                    allow_nan=False,
                )
            )
            return 0
        return execute_saved(prepared, checked, arguments, guard)
    finally:
        sys.meta_path.remove(guard)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    config = json.loads(args.configuration.read_text())
    if "worker_interface" in config:
        require(config["worker_interface"] == SAVED_INTERFACE, "unsupported worker interface")
        return saved_main(args, arguments)
    root = args.configuration.resolve().parent
    if args.case.resolve() != Path(config["case"]).resolve():
        raise ValueError("case selector differs from configuration")
    worker = (root / config["entrypoint"]).resolve(strict=True)
    if not worker.is_relative_to(root) or not worker.is_file():
        raise ValueError("entry point must be a regular file inside the run root")
    expected = config["files"].get(str(worker.relative_to(root)))
    if expected is None or digest(worker) != expected:
        raise ValueError("entry point checksum differs from configuration")
    argv = [
        sys.executable,
        str(worker),
        "--case",
        str(args.case),
        "--configuration",
        str(args.configuration),
        "--output",
        str(args.output),
    ]
    if args.preflight_only:
        argv.append("--preflight-only")
    return subprocess.run(argv, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
