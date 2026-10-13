"""Check the four-role jump study or refuse unqualified native execution.

The study, parent and checkpoint contracts are read without pyoomph imports or
output creation. The guarded runtime kernel is separate; a production native
adapter and its listed verification gates are still required for dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import paper_dynamic_jump as study
from runners import run_paper_case as stationary

IMPLEMENTATION = (
    "runners/run_paper_dynamic_jump.py",
    "runners/paper_dynamic_jump.py",
    "runners/paper_dynamic_jump_runtime.py",
)
require = study.require


def source_closure(root, packet, book):
    """Keep all baseline Python methods frozen and bind the complete added path."""
    root = root.resolve(strict=True)
    revision = stationary.component_revision(root)
    head = revision["head"]
    require(head is not None and stationary._git(root, "merge-base", "--is-ancestor", study.FEM_BASE,
                                                head) is not None,
            "isolated source must descend from the recorded FEM base")
    names = stationary._git(root, "ls-tree", "-r", "--name-only", study.FEM_BASE)
    require(names is not None, "frozen source tree unavailable")
    baseline = {name for name in names.decode().splitlines()
                if name.endswith(".py") or name in ("pyproject.toml", "uv.lock", ".python-version")}
    require(not baseline.intersection(IMPLEMENTATION), "jump path already exists at the recorded base")
    files = {}
    for name in sorted(baseline | set(IMPLEMENTATION)):
        path = root / name
        require(path.is_file() and not any(item.is_symlink() for item in (path, *path.parents)),
                "source closure requires regular files: " + name)
        digest = stationary.sha256_file(path)
        if name in baseline:
            original = stationary._git(root, "show", f"{study.FEM_BASE}:{name}")
            require(original is not None and hashlib.sha256(original).hexdigest() == digest,
                    "frozen physical model/helper changed: " + name)
        record = {"path": str(path), "sha256": digest}
        book.file(record, "source " + name)
        files[name] = digest
    actual = {str(path.relative_to(root)) for folder in ("problems", "runners")
              for path in (root / folder).rglob("*.py")}
    require(actual <= baseline | set(IMPLEMENTATION), "unsealed additional model/runner module")
    closure = {"schema": "stone-figure7-jump-source-v1", "fem_base_commit": study.FEM_BASE,
               "implementation_commit": head, "pyoomph_pin": study.PYOOMPH_PIN, "files": files}
    pending = []
    if revision["clean"] is not True:
        pending.append("clean committed isolated source pending")
    declared = packet["source"].get("isolated_published_implementation_commit")
    if declared is None:
        pending.append("published implementation commit unbound")
    else:
        require(declared == head, "published implementation commit differs from this checkout")
    seal = packet["source"].get("source_seal")
    if seal is None:
        pending.append("exact source closure seal unbound")
    else:
        require(book.json(seal, "source closure seal") == closure, "source closure seal differs")
    return revision, closure, pending


def deadline_contract(value):
    require(isinstance(value, str), "finite UTC deadline is unbound")
    try:
        deadline = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("deadline must be an explicit UTC timestamp") from error
    require(deadline.tzinfo is not None and deadline.utcoffset().total_seconds() == 0,
            "deadline must be an explicit UTC timestamp")
    require(deadline > datetime.now(UTC), "registered deadline has elapsed")
    return deadline.isoformat()


def validate_dynamic_resume(reference, study_reference, packet, plan, role, book):
    """Bind actual physical histories, the old horizon and the cumulative ledgers."""
    from runners import paper_dynamic_jump_runtime as runtime
    from runners import run_paper_cold_history as history

    resumed = book.json(reference, "dynamic resume packet")
    direction = role["direction"]
    spec = packet["directions"][direction]
    binding = plan["bindings"][direction]
    require(resumed.get("schema") == "stone-figure7-dynamic-resume-v2"
            and resumed.get("study") == study_reference
            and resumed.get("role_id") == role["role_id"]
            and resumed.get("parent_certificate") == spec["parent_certificate"]
            and resumed.get("target_case") == binding["target_case"]
            and resumed.get("dt_hat") == role["dt_hat"]
            and resumed.get("dt_solver") == role["dt_solver"]
            and resumed.get("horizon_hat") == role["horizon_hat"]
            and resumed.get("max_steps") == role["max_steps"], "dynamic resume changed study/role/control/horizon")
    index = study.integer(resumed.get("accepted_step_index"), "dynamic accepted-step index")
    require(index <= role["max_steps"], "resume index extends the registered horizon")
    inventory, exchange = study.mesh_inventory(resumed.get("mesh_inventory"), resumed.get("state"), book)
    state = book.file(resumed["state"], "dynamic state")
    metadata = history.serialized_metadata(state.read_bytes())
    require(resumed.get("native_metadata") == {
        key: metadata[key] for key in ("time", "dts", "unsteady_steps", "suggested_next_dt")},
        "dynamic metadata differs from the actual saved clock/dts/counter")
    require(metadata["unsteady_steps"] == index and metadata["suggested_next_dt"] == role["dt_solver"]
            and all(dt == role["dt_solver"] for dt in metadata["dts"]),
            "dynamic counter or fixed dt drifted")
    physical = book.json(resumed.get("physical_witness"), "dynamic full physical witness")
    require(physical.get("state") == resumed["state"], "dynamic physical witness belongs to another state")
    witness = runtime.validate_witness(physical.get("witness"))
    require(witness["time"] == metadata["time"] and witness["dts"] == metadata["dts"]
            and witness["counter"] == index
            and witness["parameters"] == {"Ca_in": role["target_ca"], "Bo": study.MODEL["bond"]}
            and witness["flags"] == runtime.DYNAMIC_FLAGS
            and all(s["is_steady"] is False for s in witness["steppers"])
            and resumed.get("steppers") == witness["steppers"]
            and resumed.get("solve_flags") == witness["flags"],
            "dynamic witness is missing saved physical history or stepping policy")
    ledger = book.json(resumed.get("step_ledger"), "native step ledger")
    rows = ledger.get("rows")
    require(ledger.get("study") == study_reference and ledger.get("role_id") == role["role_id"]
            and isinstance(rows, list) and len(rows) == index + 1, "dynamic native step ledger is incomplete")
    clock = 0.0
    for step, row in enumerate(rows):
        require(isinstance(row, dict)
                and type(row.get("accepted_step_index")) is int and type(row.get("counter")) is int
                and type(row.get("t_hat")) in (int, float) and type(row.get("native_time")) in (int, float)
                and row.get("accepted_step_index") == step
                and row.get("t_hat") == step * role["dt_hat"]
                and row.get("native_time") == clock
                and row.get("counter") == step
                and row.get("dt_solver") == role["dt_solver"], "native clock/grid/counter ledger drifted")
        require(row.get("field_qualification") == (
            "stationary_geometry_prejump_fluid_fields" if step == 0 else "converged_native_transient"),
            "interpolated or pre-jump fluid fields cannot qualify an accepted transient row")
        retained = study.retained_row(row, role, packet["thresholds"], book)
        if step == index:
            require(retained == witness and row.get("mesh_inventory") == resumed.get("mesh_inventory")
                    and row.get("physical_witness") == resumed.get("physical_witness"),
                    "last retained row and dynamic checkpoint closure differ")
        if step < index:
            clock += role["dt_solver"]
    require(clock == metadata["time"] and rows[-1]["state"] == resumed["state"],
            "dynamic checkpoint is not the last accepted native row")
    work = runtime.validate_work(resumed.get("work"), packet["controls"], packet["thresholds"], index)
    require(resumed.get("pyoomph_pin") == study.PYOOMPH_PIN
            and resumed.get("source") == packet["source"]
            and resumed.get("native_core") == binding["parent"]["manifest"]["pyoomph"]["native_core"],
            "dynamic source/environment policy changed")
    return {**resumed, "mesh_exchange": exchange, "meshes": inventory["meshes"],
            "witness": witness, "work": work, "rows": rows}


def preflight(options):
    require(options.source.resolve(strict=True) == ROOT, "use the explicitly selected isolated source")
    book = study.Artifacts()
    study_path = options.packet.expanduser().absolute()
    study_reference = {"path": str(study_path), "sha256": stationary.sha256_file(study_path)}
    packet = book.json(study_reference, "study")
    plan = study.build_plan(packet, book=book)
    for name, module in list(sys.modules.items()):
        if not name.startswith(("runners.", "problems.")):
            continue
        filename = getattr(module, "__file__", None)
        require(filename is not None and Path(filename).resolve(strict=True)
                == ROOT / (name.replace(".", "/") + ".py"), "foreign cached helper import: " + name)
    require(options.role in study.ROLES, "only the four registered roles are admitted")
    role = next(row for row in plan["roles"] if row["role_id"] == options.role)
    component, closure, pending = source_closure(ROOT, packet, book)
    environment = stationary.pyoomph_revision()
    try:
        stationary.require_pinned_environment(environment)
    except ValueError:
        pending.append("installed pinned native-core/environment authority unavailable")
    else:
        # The unchanged-case policy is deliberately reused before target adoption.
        for binding in plan["bindings"].values():
            parent = binding["parent"]
            refs = parent["certificate"]["artifacts"]
            checked = stationary.validate_restart(
                Path(refs["restart_packet"]["path"]), parent["case"], refs["case"]["sha256"],
                component=component, environment=environment,
                azimuthal_stability=book.json(refs["restart_packet"])["azimuthal_stability"])
            require(checked["mesh_exchange"] == parent["mesh_exchange"], "parent mesh exchange changed")
            parent["validated_restart"] = checked
    deadline = deadline_contract(options.deadline) if options.deadline is not None else None
    if deadline is None:
        pending.append("finite UTC execution deadline unbound")
    resumed = None
    if options.resume_packet is not None:
        ref = {"path": str(options.resume_packet.expanduser().absolute()),
               "sha256": stationary.sha256_file(options.resume_packet.expanduser())}
        resumed = validate_dynamic_resume(ref, study_reference, packet, plan, role, book)
    output = None
    if options.out is not None:
        raw = options.out.expanduser().absolute()
        require(not any(path.is_symlink() for path in (raw, *raw.parents)), "output symlink")
        output = stationary.validate_output_path(raw, ROOT)
        for path in book.records:
            authority = Path(path)
            require(not authority.is_relative_to(output) and not output.is_relative_to(authority.parent),
                    "output overlaps input authority")
    required_seal = {"schema": "stone-figure7-jump-execution-seal-v1", "study": study_reference,
                     "role": role, "source": closure, "environment": environment,
                     "deadline": deadline, "output": str(output) if output is not None else None,
                     "mode": "resume_jump" if resumed is not None else "start_jump",
                     "inputs": sorted(book.records.values(), key=lambda row: row["path"])}
    if options.seal is None:
        pending.append("exact source/input/role/deadline execution seal unbound")
    else:
        ref = {"path": str(options.seal.expanduser().absolute()),
               "sha256": stationary.sha256_file(options.seal.expanduser())}
        require(book.json(ref, "execution seal") == required_seal, "execution seal differs")
    book.verify()
    return {"packet": packet, "study": study_reference, "plan": plan, "role": role,
            "resume": resumed, "book": book, "required_seal": required_seal,
            "source_seals_pending": pending, "native_execution_authorized": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("packet", type=Path, help="immutable four-role study packet")
    result.add_argument("--source", type=Path, required=True, help="recorded isolated checkout")
    result.add_argument("--role", required=True, choices=tuple(study.ROLES))
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="native-free admission; no output")
    modes.add_argument("--start-jump", action="store_true", help="refuse until native adapter is qualified")
    modes.add_argument("--resume-jump", action="store_true", help="refuse until native adapter is qualified")
    result.add_argument("--resume-packet", type=Path, help="saved dynamic packet; also checkable with --check")
    result.add_argument("--out", type=Path, help="fresh attempt output outside all input authorities")
    result.add_argument("--deadline", help="explicit finite UTC deadline")
    result.add_argument("--seal", type=Path, help="exact external execution seal")
    return result


def main(argv=None):
    options = parser().parse_args(argv)
    try:
        require(not options.start_jump or options.resume_packet is None, "fresh jump cannot consume a dynamic resume")
        require(not options.resume_jump or options.resume_packet is not None, "resume requires a dynamic packet")
        prepared = preflight(options)
        result = {key: prepared["plan"][key] for key in (
            "schema", "software_admission_passed", "roles", "qualification_claim", "native_verification_gates")}
        result.update(selected_role=options.role, ready_for_dispatch=False,
                      native_execution_authorized=False, native_imported=False,
                      source_seals_pending=prepared["source_seals_pending"],
                      required_execution_seal=prepared["required_seal"])
        if not options.check:
            result.update(status="refused", refusal="production native adapter is unqualified")
            print(json.dumps(result, indent=2, allow_nan=False))
            return 2
        result["status"] = "checked"
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
        print(json.dumps({"status": "refused", "software_admission_passed": False,
                          "ready_for_dispatch": False, "native_execution_authorized": False,
                          "native_imported": False, "refusal": str(error)}, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
