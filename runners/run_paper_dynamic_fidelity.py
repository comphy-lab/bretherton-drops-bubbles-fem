"""Inspect or separately admit one unchanged-case native fidelity role.

Metadata inspection cannot authorise native entry. Warm and cold roles require
distinct processes, exact published source/input/control/clock declarations,
and an official active launch receipt. Neither role admits a production jump.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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
from runners import paper_dynamic_capture as native_capture
from runners import paper_dynamic_fidelity as fidelity
from runners import paper_dynamic_jump_runtime as runtime
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
        result = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
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


def preflight(args, *, component=None, observed=None):
    require(args.source.resolve(strict=True) == ROOT, "use the explicitly selected isolated fidelity checkout")
    fidelity.no_cached_foreign_helpers(ROOT)
    book = Book()
    bindings_ref = fidelity.reference(args.bindings.absolute())
    bindings = book.read(bindings_ref)
    parent = parent_contract(bindings, book)
    factory_contract(bindings, book)
    selected = {key: None for key in fidelity.CONTROL_KEYS} if args.controls is None else book.read(fidelity.reference(args.controls.absolute()))
    selected, missing = fidelity.controls(selected, unissued=True)
    component = ordinary.component_revision(ROOT) if component is None else component
    observed = ordinary.pyoomph_revision() if observed is None else observed
    pending = ["numerical control unissued: " + key for key in missing]
    try:
        ordinary.require_clean_source(component)
    except ValueError:
        pending.append("prospective driver is not clean published source")
    source = {"commit": component["head"], "root": str(ROOT), "files": fidelity.source_files(ROOT)}
    if component.get("clean") is True:
        for name, ref in source["files"].items():
            committed = ordinary._git(ROOT, "show", "HEAD:" + name)
            require(committed is not None and hashlib.sha256(committed).hexdigest() == ref["sha256"],
                    "prospective source closure is not committed: " + name)
    receipt_ref = None if args.environment is None else fidelity.reference(args.environment.absolute())
    pending.extend(environment_contract(bindings, receipt_ref, book, observed))
    warm = None
    if args.role == "cold":
        if args.warm_result is None:
            pending.append("completed warm-role result unissued")
        elif not missing:
            warm = warm_contract(fidelity.reference(args.warm_result.absolute()), book, selected, source)
    else:
        require(args.warm_result is None, "warm role cannot consume another warm result")
    output = None
    if args.out is None:
        pending.append("fidelity output unissued")
    else:
        output = args.out.absolute()
        require(not any(p.is_symlink() for p in (output, *output.parents)), "output alias refused")
        for ref in [*book.records.values(), *source["files"].values()]:
            path = Path(ref["path"])
            require(not path.is_relative_to(output) and not output.is_relative_to(path.parent)
                    and not output.is_relative_to(ROOT), "output overlaps input or source authority")
    requirements = {"schema": fidelity.SCHEMA + "-requirements", "role": args.role, "bindings": bindings_ref,
        "source": source, "environment": observed, "fresh_environment": receipt_ref, "controls": selected,
        "caps": fidelity.CAPS, "case": parent["case"], "output": str(output) if output is not None else None,
        "warm_result": None if args.warm_result is None else fidelity.reference(args.warm_result.absolute()),
        "inputs": sorted(book.records.values(), key=lambda row: row["path"]),
        "policy": {"history_fill": False, "clock_reset": False, "forcing_change": False,
                   "production_jump": False, "synthetic_buffer_stress": False, "exact_next_transition": True}}
    phase = phase_ref = None
    if args.phase is None:
        pending.append("root phase declaration unissued")
    elif not pending:
        phase_ref = fidelity.reference(args.phase.absolute())
        phase = book.read(phase_ref)
        phase_contract(phase, requirements)
    return {**parent, "bindings": bindings, "bindings_ref": bindings_ref, "book": book, "controls": selected,
            "component": component, "source": source, "environment": observed, "warm": warm,
            "output": output, "phase": phase, "phase_ref": phase_ref,
            "requirements": requirements, "pending": pending, "native_execution_authorized": False}


def phase_contract(phase, requirements, *, now=None):
    require(isinstance(phase, dict) and set(phase) == {"schema", "requirements", "issued_by", "declared_at", "deadline_utc", "phase_authorized"}
            and phase["schema"] == fidelity.SCHEMA + "-phase" and phase["requirements"] == requirements
            and phase["issued_by"] == "root" and phase["phase_authorized"] is True, "root phase differs from exact requirements")
    current = datetime.now(UTC) if now is None else now
    start, end = utc(phase["declared_at"]), utc(phase["deadline_utc"])
    require(start <= current < end and end > start, "prospective finite phase window is not current")
    return end


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
    require(admission.get("schema") == fidelity.SCHEMA + "-admission"
            and admission.get("native_execution_authorized") is True
            and admission.get("phase") == prepared["phase_ref"]
            and admission.get("requirements") == prepared["requirements"], "native admission differs")
    tracker = prepared["book"].read(admission["active_tracker"])
    require(tracker.get("schema") == fidelity.SCHEMA + "-active-tracker" and tracker.get("phase_kind") == fidelity.SCHEMA
            and tracker.get("status") == "running" and tracker.get("role") == prepared["requirements"]["role"]
            and tracker.get("phase") == prepared["phase_ref"] and tracker.get("phase_authorized") is True
            and tracker.get("output") == str(prepared["output"]) and tracker.get("source_commit") == prepared["source"]["commit"]
            and utc(tracker["deadline_utc"]) == deadline and tracker.get("caps") == fidelity.CAPS,
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
    expected = tracker["current_root_identity"]
    require(all(actual[k] == expected[k] for k in actual), "admitted native worker process identity differs")
    require(all(expected[k] == launch["root_identity"][k] for k in expected if k != "host"), "official launch identity differs")
    cpus = tracker["cpus"]
    require(isinstance(cpus, list) and cpus and all(type(c) is int and c >= 0 for c in cpus)
            and len(set(cpus)) == len(cpus) and dispatcher.get("cpus") == cpus, "official CPU set differs")
    if affinity is None:
        affinity = {int(path.name): sorted(os.sched_getaffinity(int(path.name))) for path in Path("/proc/self/task").iterdir()}
    require(affinity and all(value == sorted(cpus) for value in affinity.values()), "actual worker thread affinity differs")
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
    admission = prepared["book"].read(fidelity.reference(path))
    actual = admission_contract(admission, prepared, now=clock())
    prepared["native_execution_authorized"] = True
    prepared["native_process"] = actual


def execute(prepared, output, emit):
    require(prepared["native_execution_authorized"] is True and prepared["phase"] is not None and not prepared["pending"],
            "metadata cannot authorise native fidelity entry")
    phase_contract(prepared["phase"], prepared["requirements"])
    prepared["book"].stable()
    for ref in prepared["source"]["files"].values():
        fidelity.bound(ref)
    # Every native import follows actual root-issued phase and active admission.
    from pyoomph import _pyoomph_core as core
    prepared["native_imported"] = True
    from pyoomph.meshes.mesh import ODEStorageMesh
    from pyoomph.meshes.remesher import RemesherViaRecreation

    from problems.paper_mesh import PaperTwoPhaseMesh

    require(fidelity.reference(Path(core.__file__).absolute()) == prepared["environment"]["native_core"], "imported native core differs")
    before_detection = core.get_detect_inverted_elements()
    selected = prepared["controls"]
    deadline = utc(prepared["phase"]["deadline_utc"])

    def stable():
        require(datetime.now(UTC) < deadline, "finite native fidelity deadline elapsed")
        prepared["book"].stable()
        for ref in prepared["source"]["files"].values():
            fidelity.bound(ref)

    case = prepared["case"]
    forcing = {"Ca_in": case["physics"]["ca_in"], "Bo": case["physics"]["bond"]}
    types = native_capture.NativeTypes(model._native_problem_types()[0], core.MultiTimeStepper,
                ODEStorageMesh, PaperTwoPhaseMesh, RemesherViaRecreation)
    capture = native_capture.Capture(types)
    core.set_detect_inverted_elements(True)
    try:
        with model.build_problem(case, initial_bond=case["physics"]["bond"]) as problem:
            problem.set_output_directory(str(output / "solver"))
            problem.set_linear_solver("superlu")
            problem.set_c_compiler("system")
            problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
            policy = native_capture.prepare_initialisation(problem, types)
            emit("initialisation_policy", policy)
            problem.eigen_data_in_states, problem.continuation_data_in_states = True, False
            problem.do_call_remeshing_when_necessary = False
            problem.inversion_remesh_max_retries = 0
            guard = fidelity.EvolutionGuard(problem, forcing, emit)
            with guard.installed():
                stable()
                with guard.declared("initialise"):
                    problem.initialise()
                require(not problem.is_distributed(), "serial native fidelity role required")
                parent = prepared["parent"]
                expected = None
                if prepared["warm"] is not None:
                    row = prepared["warm"]["initial_row"]
                    inventory = prepared["book"].read(row["mesh_inventory"])
                    physical = prepared["book"].read(row["physical_witness"])
                    parent = {"state": row["state"], "meshes": inventory["meshes"]}
                    expected = runtime.validate_witness(physical["witness"])
                loader = fidelity.LoadAdapter(problem, capture, guard, emit)
                loader.load(parent["state"], parent["meshes"])
                original = prepared["book"].file(parent["state"]).read_bytes()
                fidelity.endtime_restore(problem, original, case, output, emit)
                if expected is None:
                    fidelity.cross_check_live(problem, capture, original, emit, "original_parent_numeric_cross_check")
                    initial = fidelity.register_policy(problem, capture, emit)
                else:
                    runtime.restore_loaded_policy(problem, expected, fidelity.capture_module_hooks(capture))
                    initial = fidelity.cross_check_live(problem, capture, original, emit, "cold_initial_numeric_cross_check")
                    require(initial == expected, "fresh cold complete history/policy differs")
                initial_row = fidelity.retain(problem, capture, output, 0, initial)
                require(capture(problem) == initial, "initial retention changed native state")
                stable()
                next_state = fidelity.sole_step(problem, capture, guard, selected, deadline, output, emit, stable)
                if prepared["warm"] is not None:
                    warm_after = prepared["book"].read(prepared["warm"]["next_row"]["physical_witness"])["witness"]
                    require(next_state["next_witness"] == warm_after, "warm/cold next transition differs")
                    emit("exact_next_transition_match", {"full_witness_equal": True})
                stable()
                return {"initial_row": initial_row, "next_row": next_state["row"], "work": next_state["work"],
                        "callbacks": next_state["callbacks"], "time_step_calls_charged": 1,
                        "load_calls": loader.calls, "forbidden_entry_attempts": guard.denied,
                        "native_imported": True, "exact_warm_cold_next_transition": prepared["warm"] is not None}
    finally:
        core.set_detect_inverted_elements(before_detection)
        require(core.get_detect_inverted_elements() == before_detection, "inversion detection cleanup differs")


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("bindings", "source"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--role", choices=fidelity.ROLES, required=True)
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
                    and args.admission is not None, "native fidelity requirements are unissued")
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "refused", "refusal": str(error), "native_imported": False,
                          "native_execution_authorized": False, "ready_for_dispatch": False}))
        return 2
    if args.check:
        print(json.dumps({"schema": fidelity.SCHEMA + "-metadata-check", "requirements": prepared["requirements"],
              "pending": prepared["pending"], "native_imported": False, "native_execution_authorized": False,
              "ready_for_dispatch": False}, indent=2, allow_nan=False))
        return 0
    output = prepared["output"]
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    counter = 0

    def emit(label, value):
        nonlocal counter
        counter += 1
        return fidelity.write_new(output / (f"receipt-{counter:04d}-" + label + ".json"), value)

    emit("prepared", {"event": "prepared", "native_execution_authorized": False, "active_admission_pending": True,
                     "requirements": prepared["requirements"], "phase": prepared["phase_ref"]})
    result = {"schema": fidelity.SCHEMA + "-result", "role": args.role, "status": "failed",
              "source": prepared["source"], "controls": prepared["controls"], "phase": prepared["phase_ref"],
              "native_imported": False, "numerical_acceptance": False, "production_jump_qualified": False,
              "nonflat_physical_history_qualified": False, "synthetic_buffer_stress_performed": False}
    try:
        await_admission(prepared, args.admission)
        result["native_process"] = prepared["native_process"]
        fidelity.write_new(output / "run-manifest.json", {**result, "native_execution_authorized": True,
                     "requirements": prepared["requirements"], "environment": prepared["environment"]})
        result["native_entry_attempted"] = True
        result.update(execute(prepared, output, emit))
        result["status"] = "complete_role"
    except BaseException as error:  # noqa: BLE001 - preserve interrupted charges without retrying native work
        result["native_imported"] = prepared.get("native_imported", False)
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        if (output / "step-charge.json").exists():
            result["time_step_calls_charged"] = 1
        fidelity.write_new(output / "failure.json", result)
    fidelity.write_new(output / "result.json", result)
    return 0 if result["status"] == "complete_role" else 1


if __name__ == "__main__":
    raise SystemExit(main())
