"""Inspect or separately admit the zero-solve before-step marker role.

Metadata cannot authorise native entry. Marker execution requires exact root
phase and official active admission. Full warm/cold execution remains refused.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems import paper_bubble as model
from runners import paper_dynamic_callback_protocol as callback_protocol
from runners import paper_dynamic_fidelity as fidelity
from runners import paper_dynamic_jump_runtime as runtime
from runners import paper_dynamic_marker as marker
from runners import run_paper_case as ordinary
from runners import run_paper_cold_history as history

require = fidelity.require
BASE_CAPTURE_COMMIT = "687ef8ee72ba4f50ee23fbeb15217ecdf6e6c3d8"
PIN = model.PYOOMPH_COMMIT
FACTORIES = ("validate_case", "case_geometry", "_bulk_viscous_stress", "_native_problem_types", "build_problem")


class Book:
    def __init__(self):
        self.records = {}

    def file(self, ref):
        path, _raw, record = fidelity.read_bound(ref)
        require(str(path) not in self.records or self.records[str(path)] == record, "conflicting input authority")
        self.records[str(path)] = record
        return path

    def read(self, ref):
        path, raw, record = fidelity.read_bound(ref)
        def reject_constant(value):
            raise ValueError(f"nonfinite JSON value: {value}")
        def unique_object(pairs):
            value = {}
            for key, child in pairs:
                require(key not in value, "duplicate JSON authority key: " + key)
                value[key] = child
            return value
        result = json.loads(raw.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)
        require(isinstance(result, dict), "JSON authority must be an object")
        require(str(path) not in self.records or self.records[str(path)] == record, "conflicting input authority")
        self.records[str(path)] = record
        return result

    def stable(self):
        for ref in list(self.records.values()):
            self.file(ref)


def utc(value):
    require(isinstance(value, str), "explicit UTC timestamp required")
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "UTC clock required")
    return result


def parent_contract(bindings, book):
    require(bindings.get("schema") == "stone-dynamic-fidelity-parent-byte-audit-v1", "parent binding schema differs")
    require(bindings["source"]["expected_published_commit"] == BASE_CAPTURE_COMMIT, "published capture base differs")
    for ref in bindings["inputs"]:
        book.file(ref)
    parent = bindings["parent"]
    case = model.validate_case(book.read(parent["case"]))
    require(case["physics"]["problem"] == "finite_gas_bubble" and parent["azimuthal_stability"] is True,
            "unchanged finite-gas base case required")
    resume = book.read(parent["resume"])
    require(resume.get("schema") == "herrada-paper-stationary-resume-v1"
            and resume["roundtrip"]["verified"] is True and resume["stage"]["kind"] == "completed_polish"
            and resume["case_sha256"] == parent["case"]["sha256"]
            and resume["state"]["sha256"] == parent["state"]["sha256"], "stationary parent provenance differs")
    manifest = book.read(resume["parent_manifest"])
    root = Path(resume["parent_manifest"]["path"]).parent
    refs = {Path(ref["path"]).name: ref for ref in bindings["inputs"] if Path(ref["path"]).parent == root}
    summary, completion = book.read(refs["summary.json"]), book.read(refs["completion.json"])
    require(summary.get("numerical_acceptance") is True and summary.get("cold_load_verified") is True
            and completion.get("numerical_acceptance") is True and completion.get("status") == "complete",
            "parent native stationary qualification is absent")
    for key in ("checks", "cold_load_checks"):
        require(set(summary[key]) == {"stationary", "volume", "volume_integrals", "flux", "lower_static_pressure"}
                and all(value is True for value in summary[key].values()), "parent stationary gate failed")
    require(manifest["case_sha256"] == parent["case"]["sha256"] and manifest["azimuthal_stability"] is True,
            "parent manifest belongs to another case/layout")
    book.file(parent["state"])
    for mesh in parent["meshes"]:
        book.file(mesh)
    raw = Path(parent["state"]["path"]).read_bytes()
    parsed = history.serialized_metadata(raw)
    require(parsed["time"] == bindings["decoded"]["clock"]["time"]
            and parsed["dts"] == bindings["decoded"]["clock"]["dts"]
            and parsed["unsteady_steps"] == bindings["decoded"]["clock"]["counter"]
            and parsed["output_step"] == bindings["decoded"]["clock"]["output_step"], "saved parent clock changed")
    require(ordinary.state_bc_cache_marker(Path(parent["state"]["path"])) in {"init", "stationary", "transient"},
            "zero-eigen/continuation serialized parent BC policy required")
    require(parsed["endtime"] == case["solve"]["relax_time"], "parent endpoint metadata differs")
    return {"case": case, "parent": parent, "resume": resume, "manifest": manifest, "metadata": parsed}


def factory_contract(bindings, book):
    old = bindings["factory_compatibility"]["parent_file"]
    text = book.file(old).read_text()
    current = (ROOT / "problems/paper_bubble.py").read_text()
    old_tree, new_tree = ast.parse(text), ast.parse(current)
    for name in FACTORIES:
        a = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        require(ast.get_source_segment(text, a) == ast.get_source_segment(current, b), "parent factory differs: " + name)
    frozen = {ref["relative_path"]: ref for ref in bindings["source"]["files"]}
    for name in ("runners/paper_dynamic_capture.py", "runners/paper_dynamic_state.py",
                 "runners/paper_dynamic_jump_runtime.py", "problems/paper_mesh.py", "pyproject.toml", "uv.lock"):
        require(fidelity.digest(ROOT / name) == frozen[name]["sha256"], "published capture/model dependency differs: " + name)


def environment_contract(bindings, receipt_ref, book, observed):
    ordinary.require_pinned_environment(observed)
    expected = bindings["environment"]
    require(observed["commit"] == expected["pin"] == PIN
            and observed["native_core"] == expected["native_core"], "installed native core/source differs")
    require(observed["interpreter"]["executable"] == expected["interpreter"], "selected interpreter differs")
    if receipt_ref is None:
        return ["fresh environment receipt unissued"]
    receipt = book.read(receipt_ref)
    historical = book.read(expected["receipt"])
    require(receipt.get("schema") == "pyoomph-environment-v1" and receipt.get("compatible") is True
            and receipt["checked_compiler"] == "system" and receipt["checked_solver"] == "superlu"
            and receipt["checks"]["compiler"]["status"] == receipt["checks"]["solver"]["status"] == "passed"
            and receipt["pyoomph"]["resolved_commit"] == PIN
            and receipt["pyoomph"]["core_sha256"] == observed["native_core"]["sha256"]
            and receipt["interpreter"]["path"] == observed["interpreter"]["executable"]
            and receipt["mpi"]["compiled"] is False, "fresh serial pinned environment evidence differs")
    require(utc(receipt["created_at"]) > utc(historical["created_at"])
            and utc(receipt["created_at"]) <= datetime.now(UTC), "historical environment receipt cannot qualify this phase")
    return []


def warm_contract(ref, book, selected, source):
    result = book.read(ref)
    require(result.get("schema") == fidelity.SCHEMA + "-result" and result.get("role") == "warm"
            and result.get("status") == "complete_role" and result.get("native_imported") is True
            and result.get("controls") == selected and result.get("source") == source,
            "warm role result does not bind the same published source/operator")
    require(result.get("time_step_calls_charged") == 1 and result.get("native_process") is not None,
            "warm native call/process proof is absent")
    for key in ("initial_row", "next_row"):
        row = result[key]
        for name in ("state", "mesh_inventory", "physical_witness", "profile", "numeric_cross_check"):
            book.file(row[name])
        physical = book.read(row["physical_witness"])
        require(physical["state"] == row["state"], "warm retained witness belongs to another state")
        runtime.validate_witness(physical["witness"])
        inventory = book.read(row["mesh_inventory"])
        require(inventory["state"] == row["state"] and inventory["mesh_reference_base"] == str(Path(row["state"]["path"]).parent),
                "warm mesh inventory differs")
        for mesh in inventory["meshes"]:
            book.file(mesh)
    return result


def preflight(args, *, component=None, observed=None, committed_reader=None):
    require(args.role == "marker", "full warm/cold execution and metadata remain refused pending separate wiring/review")
    return marker.preflight(args, sys.modules[__name__], component=component, observed=observed,
                            committed_reader=committed_reader)


def phase_contract(phase, requirements, *, now=None):
    return marker.phase_contract(phase, requirements, now=now)


def process_identity():
    # Parse the documented /proc stat fields after its parenthesised comm.
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": os.getpid(), "uid": os.getuid(), "host": os.uname().nodename,
            "start_time_ticks": int(fields[19]), "process_group_id": int(fields[2]), "session_id": int(fields[3]),
            "cwd": str(Path("/proc/self/cwd").resolve()), "cgroup": Path("/proc/self/cgroup").read_text().splitlines(),
            "argv": [v.decode() for v in Path("/proc/self/cmdline").read_bytes().split(b"\0") if v]}


def admission_contract(admission, prepared, *, now=None, identity=None, affinity=None):
    current = datetime.now(UTC) if now is None else now
    deadline = phase_contract(prepared["phase"], prepared["requirements"], now=current)
    marker.admission_header_contract(admission, prepared)
    require(admission.get("schema") == marker.SCHEMA + "-admission"
            and admission.get("marker_authorized") is True and admission.get("role") == "marker"
            and admission.get("deadline_utc") == prepared["phase"]["deadline_utc"]
            and admission.get("native_execution_authorized") is True
            and callback_protocol.same(admission.get("phase"), prepared["phase_ref"])
            and callback_protocol.same(admission.get("requirements"), prepared["requirements"]), "native admission differs")
    tracker = prepared["book"].read(admission["active_tracker"])
    require(tracker.get("schema") == marker.SCHEMA + "-active-tracker" and tracker.get("phase_kind") == marker.SCHEMA
            and tracker.get("status") == "running" and tracker.get("role") == prepared["requirements"]["role"]
            and tracker.get("phase") == prepared["phase_ref"] and tracker.get("phase_authorized") is True
            and tracker.get("output") == str(prepared["output"]) and tracker.get("source_commit") == prepared["source"]["commit"]
            and utc(tracker["deadline_utc"]) == deadline and callback_protocol.same(callback_protocol.caps_contract(tracker.get("caps")), prepared["requirements"]["caps"]),
            "active tracker differs from the exact fidelity role")
    require(0 <= (current - utc(tracker["checked_at"])).total_seconds() <= 300, "active tracker is stale")
    reservation = tracker["reservation"]
    require(reservation.get("state") == "active" and all(isinstance(reservation.get(k), str) and reservation[k]
            for k in ("id", "generation_id", "generation_sha256")), "official active reservation proof missing")
    dispatcher = prepared["book"].read(tracker["active_dispatcher"])
    require(dispatcher.get("authority_verified") is True and dispatcher.get("status") == "active"
            and dispatcher.get("reservation_id") == reservation["id"]
            and dispatcher.get("generation_id") == reservation["generation_id"], "official active dispatcher differs")
    launch = prepared["book"].read(tracker["launch"])
    prepared["book"].file(tracker["launch_audit"])
    require(launch["launch_audit"] == tracker["launch_audit"], "official launch audit differs")
    actual = process_identity() if identity is None else identity
    expected = callback_protocol.process_contract(tracker["current_root_identity"])
    callback_protocol.process_contract(actual)
    require(callback_protocol.same(actual, expected), "admitted native worker process identity differs")
    require(all(callback_protocol.same(expected[k], launch["root_identity"][k]) for k in expected if k != "host"), "official launch identity differs")
    cpus = tracker["cpus"]
    require(isinstance(cpus, list) and cpus and all(type(c) is int and c >= 0 for c in cpus)
            and len(set(cpus)) == len(cpus) and callback_protocol.same(dispatcher.get("cpus"), cpus), "official CPU set differs")
    if affinity is None:
        affinity = {int(path.name): sorted(os.sched_getaffinity(int(path.name))) for path in Path("/proc/self/task").iterdir()}
    require(type(affinity) is dict and affinity and all(type(tid) is int and tid > 0
            and type(value) is list and callback_protocol.same(value, sorted(cpus)) for tid, value in affinity.items()),
            "actual worker thread affinity differs")
    warm = prepared.get("warm")
    if warm is not None:
        previous = warm["native_process"]
        require((actual["host"], actual["pid"], actual["start_time_ticks"])
                != (previous["host"], previous["pid"], previous["start_time_ticks"]), "cold role reused the warm process")
    prepared["book"].stable()
    for ref in prepared["source"]["files"].values():
        fidelity.bound(ref)
    return actual


def await_admission(prepared, path, *, clock=None, sleep=None):
    require(path is not None and prepared["phase"] is not None and not prepared["pending"], "exact native phase/admission unissued")
    clock = (lambda: datetime.now(UTC)) if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    path = Path(path).absolute()
    while not path.exists():
        current = clock()
        end = phase_contract(prepared["phase"], prepared["requirements"], now=current)
        require(not any(p.is_symlink() for p in (path, *path.parents)), "admission alias refused")
        sleep(min(1.0, (end - current).total_seconds()))
    admission_ref = fidelity.reference(path)
    admission = prepared["book"].read(admission_ref)
    actual = admission_contract(admission, prepared, now=clock())
    prepared["native_execution_authorized"] = True
    prepared["native_process"] = actual
    prepared["admission_ref"] = admission_ref
    prepared["admission"] = admission


def execute(prepared, output, emit):
    require(prepared["requirements"]["role"] == "marker", "full warm/cold execution remains refused")
    return marker.execute(prepared, output, emit, sys.modules[__name__])


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("bindings", "source"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--role", choices=("marker", *fidelity.ROLES), required=True)
    for name in ("controls", "environment", "warm-result", "out", "phase", "admission"):
        result.add_argument("--" + name, type=Path)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    os.umask(0o077)
    try:
        prepared = preflight(args)
        if not args.check:
            require(not prepared["pending"] and prepared["phase"] is not None and prepared["output"] is not None
                    and args.admission is not None, "native marker requirements are unissued")
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "refused", "refusal": str(error), "native_imported": False,
                          "native_execution_authorized": False, "ready_for_dispatch": False}))
        return 2
    if args.check:
        print(json.dumps({"schema": marker.SCHEMA + "-metadata-check", "requirements": prepared["requirements"],
              "pending": prepared["pending"], "native_imported": False, "native_execution_authorized": False,
              "ready_for_dispatch": False}, indent=2, allow_nan=False))
        return 0
    output = prepared["output"]
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    counter = 1
    fidelity.write_new(output / "receipt-0001-prepared.json", {"event": "prepared", "role": "marker",
        "native_execution_authorized": False, "active_admission_pending": True,
        "requirements": prepared["requirements"], "phase": prepared["phase_ref"]})
    prepared["raw_events"] = []

    def emit(label, value):
        nonlocal counter
        counter += 1
        authority = prepared["marker_authority"].snapshot()
        event = {"schema": marker.SCHEMA + "-raw-event", "role": "marker", "sequence": counter,
                 "event": label, "authority": authority, "value": value}
        ref = fidelity.write_new(output / (f"receipt-{counter:04d}-" + label + ".json"), event)
        prepared["raw_events"].append(ref)
        if label == "before_step_cpp_marker_received":
            prepared["raw_marker_deliveries"] = value["raw_marker_deliveries"]
        if label == "before_step_cpp_descriptor_returned":
            prepared["normal_descriptor_return"] = value["normal_descriptor_return"]
        return ref

    result = marker.initial_result(prepared)
    try:
        await_admission(prepared, args.admission)
        prepared["marker_authority"] = marker.bind_authority(prepared, sys.modules[__name__])
        result["native_process"] = prepared["native_process"]
        result["admission"] = prepared["admission_ref"]
        fidelity.write_new(output / "run-manifest.json", {**result, "native_execution_authorized": True,
                     "environment": prepared["environment"]})
        result["native_entry_attempted"] = True
        result.update(execute(prepared, output, emit))
        result["status"] = "complete_marker"
    except BaseException as error:  # noqa: BLE001 - preserve charges/raw evidence; never retry native work
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    marker.retain_result_state(result, prepared)
    if result["status"] == "failed_marker":
        fidelity.write_new(output / "failure.json", result)
    fidelity.write_new(output / "result.json", result)
    return 0 if result["status"] == "complete_marker" else 1


if __name__ == "__main__":
    raise SystemExit(main())
