"""Cold-observe the retained coarse III state twice without solving or remeshing.

This type qualifies only its two fresh stationary observations. The original
provisional stage and the fine stationary parent's acceptance remain unchanged.
Metadata inspection cannot authorise native execution; execution requires a
clean committed source and an exact external input seal.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_anchor_iii_from_stationary as iii

model, runner, diagnostic, remesh, history = (
    iii.model, iii.runner, iii.diagnostic, iii.remesh, iii.history
)
require, sha, write_json = diagnostic.require, diagnostic.sha, diagnostic.write_json
regular_record, relative_path = iii.regular_record, iii.relative_path
ENTRYPOINT = "runners/run_paper_iii_coarse_cold.py"
SCHEMA = "stationary-III-coarse-cold-observation-v1"
PACKET_SCHEMA = "stationary-III-coarse-cold-packet-v1"
KIND = "cold_qualified_stationary_observation"
REQUEST_SCHEMA = "paper-returned-III-coarse-m0-request-v1"
SEAL_SCHEMA = "stationary-III-coarse-execution-seal-v1"
BASE = "30aa43e28f8e481735c8a5164e3cab39cd16e5d2"
CASE_SHA = "f620e9c79ecdb3d50813d35568ac5582ac893d4fc9e2fa8b577c9715f8605cfb"
ARCHIVE_SHA = "4ddeb02bdd1f255422c2911770b9cd7f1e98051e656acd3b2559ea493107ab31"
COARSE_STATE_SHA = "5a755242ef31c1016c114a9531a381da5222b3142c0b09a535f63adbb775f12c"
COARSE_MESH_SHA = "cd32763fce24e5062108354aff87d60747ba7d4357e949014069ec8536150327"
COARSE_PAYLOAD = "457daaef59ff913ff72a1c9109e6597f35d0972726ec6d31c00f948240f100ba"
COARSE_STREAM = "04adb889e21d5ae3f1bb38544a1101ca65640f7a3025a39c027587e6413328ca"
COARSE_PINS = "8c8c0ae735c1ed3a2dc8442b6139fb20f1f273f228e63990ea4bc091e0ca099d"
COARSE_MASK = "516009d750d409b19ef58fe33dde02ccb88cc6268501945ff3b8a7102ea2598f"
COARSE_LAYOUT = "f9433a8582120d2e59c08407c49548cfe0cd87be7b843657a4cb9f67857ee5fe"
SEALED_ORIGIN = {
    "run-manifest.json": "781a5d53ad32bc46ec2a75e60bd679aa0f694f637c5d2d57247f1f97e28f6e21",
    "completion.json": "0c7165af48ac47c5e79aed69d8f14d09e926e14e2fdc1deb95d987bdf4e77b7c",
    "checkpoints/bond_016/resume.json": "baa32dbdc113fcbcf0db97c8b96fc27d918a567fad348c0e25727e45f2c870ed",
    "checkpoints/bond_016/state.dump": "f9bef6c1227854e5b6981dd07ac011d734cd97cc558cea4c4c9ea12220ce0f87",
    "checkpoints/bond_016/PaperTwoPhaseMesh_REMESH_000004.msh": "6c2b0703403c81bc3c104eb6cf3a3a8e0efb9f45db8f03e342311bf4f5077cd9",
    "stages/ca_003/checkpoints/bond_016/resume.json": "d51773ed7f417c0471931aa5f1532a126b090e7721d48164d544a0b5011557b3",
    "stages/ca_003/checkpoints/bond_016/state.dump": COARSE_STATE_SHA,
    "stages/ca_003/checkpoints/bond_016/PaperTwoPhaseMesh_REMESH_000003.msh": COARSE_MESH_SHA,
}
HELPERS = (*iii.HELPERS, iii.ENTRYPOINT, ENTRYPOINT, "runners/run_paper_modes.py")
NO_CLAIMS = {
    **iii.NO_CLAIMS,
    "anchor_accepted": False,
    "parent_acceptance_changed": False,
    "original_stage_acceptance_changed": False,
    "SP_certificate": False,
    "finite_spectrum_complete": False,
    "global_dominance_certified": False,
    "unique_family_identity_certified": False,
}
SCOPE = "two fresh cold stationary observations of the retained III coarse mesh; no solve or remesh"
ZERO_WORK = {
    "nonlinear_solves": 0,
    "eigen_solves": 0,
    "remeshes": 0,
    "time_advances": 0,
    "linear_factorizations": 0,
}


def is_producer(parent):
    return Path(parent.get("runner", {}).get("path", "")).name == Path(ENTRYPOINT).name


def unique_records(records):
    by_path = {}
    for record in records:
        value = regular_record(record["path"])
        require(value["sha256"] == record["sha256"], "bound input changed")
        require(by_path.get(value["path"], value) == value, "conflicting input identities")
        by_path[value["path"]] = value
    return [by_path[path] for path in sorted(by_path)]


def repaired_method(name, original):
    """Specify the two permitted diagnostic AST changes, including their exact scope."""
    text = original.decode()
    if name == "problems/bretherton_bubble.py":
        old = "        def key(node):\n            return (round(node.x(0), 12), round(node.x(1), 12))"
        new = '''        raw_coordinates = {}

        def key(node):
            raw = (node.x(0), node.x(1))
            rounded = (round(raw[0], 12), round(raw[1], 12))
            if raw_coordinates.setdefault(rounded, raw) != raw:
                raise RuntimeError("distinct interface coordinates share a rounded connectivity key")
            return rounded'''
        old_points = "        pts = np.array([[z, r] for r, z in chain], dtype=float)"
        new_points = '''        pts = np.array([[raw_coordinates[node][1], raw_coordinates[node][0]]
                        for node in chain], dtype=float)'''
        require(text.count(old) == text.count(old_points) == 1, "legacy walker source differs")
        return text.replace(old, new).replace(old_points, new_points).encode()
    require(name == "problems/paper_bubble.py", "unsupported diagnostic exception")
    old = "    dofs, positional_dofs = problem.get_current_dofs()\n    z, r = problem.interface_polyline()"
    new = old + '''
    # Preserve the legacy interface hash representation; native positions remain raw.
    z, r = (np.asarray([round(float(value), 12) for value in axis]) for axis in (z, r))'''
    require(text.count(old) == 1, "legacy stationary signature source differs")
    return text.replace(old, new).encode()


def same_ast(first, second):
    return ast.dump(ast.parse(first), include_attributes=False) == ast.dump(
        ast.parse(second), include_attributes=False
    )


def frozen_spectral_contract(original, current):
    before, after = ast.parse(original), ast.parse(current)
    edited = {"request_contract", "qualified_parent", "preflight", "execute", "plot_returned", "main"}
    functions = {node.name: node for node in after.body if isinstance(node, ast.FunctionDef)}
    for node in before.body:
        if isinstance(node, ast.FunctionDef) and node.name not in edited:
            require(
                node.name in functions
                and ast.dump(node, include_attributes=False)
                == ast.dump(functions[node.name], include_attributes=False),
                "frozen spectral/restore function changed: " + node.name,
            )
    for name in ("MODES", "SHIFTS", "NEIGEN", "GATES", "LIMITS", "GLOBALS", "EIGENDATA_FIELDS"):
        def assignment(tree, name=name):
            return [node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name
                            for target in node.targets)]
        a, b = assignment(before), assignment(after)
        require(len(a) == len(b) == 1 and same_ast(ast.unparse(a[0]), ast.unparse(b[0])),
                "frozen spectral request/gates changed: " + name)


def source_contract(parent, *, metadata_only):
    current = runner.component_revision(ROOT)
    runner.require_clean_source(parent["component"])
    require(parent["component_commit"] == parent["component"]["commit"] == BASE,
            "only the sealed III stationary source is admitted")
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", BASE, current["head"])
    require(set(current["sources"]) == set(parent["component"]["sources"]),
            "method inventories differ")
    exceptions = {}
    for name, record in parent["component"]["sources"].items():
        old = diagnostic.git(ROOT, "show", f"{BASE}:{name}")
        digest = hashlib.sha256(old).hexdigest()
        actual = (ROOT / relative_path(name)).read_bytes()
        require(record["committed_matches"] is True
                and record["sha256"] == record["committed_sha256"] == digest,
                "origin method declaration differs")
        if name in {"problems/bretherton_bubble.py", "problems/paper_bubble.py"}:
            require(same_ast(repaired_method(name, old), actual),
                    "diagnostic repair changed another method AST: " + name)
            exceptions[name] = {"origin_sha256": digest, "current_sha256": sha(ROOT / name)}
        else:
            require(actual == old, "frozen method changed: " + name)
    implementation = {name: regular_record(ROOT / name) for name in HELPERS}
    for name in (*iii.HELPERS, iii.ENTRYPOINT):
        expected = hashlib.sha256(diagnostic.git(ROOT, "show", f"{BASE}:{name}")).hexdigest()
        require(implementation[name]["sha256"] == expected, "frozen III helper changed")
        require(parent["implementation"][name]["sha256"] == expected,
                "origin III helper declaration differs")
    frozen_spectral_contract(
        diagnostic.git(ROOT, "show", f"{BASE}:runners/run_paper_modes.py"),
        (ROOT / "runners/run_paper_modes.py").read_bytes(),
    )
    if not metadata_only:
        runner.require_clean_source(current)
        for name, record in implementation.items():
            require(hashlib.sha256(diagnostic.git(ROOT, "show", f"{current['commit']}:{name}")).hexdigest()
                    == record["sha256"], "implementation is not sealed by the source commit")
    return current, implementation, {
        "schema": "III-coarse-diagnostic-source-compatibility-v1",
        "origin_commit": BASE,
        "current_commit": current["commit"],
        "diagnostic_exceptions": exceptions,
        "native_state_bytes_modified": False,
        "physics_and_spectral_gates_changed": False,
    }


def environment_contract(current, parent):
    for value in (current, parent):
        runner.require_pinned_environment(value)
        require(value["commit"] == diagnostic.PYOOMPH
                and value["native_core"]["sha256"] == diagnostic.CORE,
                "source pin or native core differs")
    require(current["url"] == parent["url"]
            and current["interpreter"]["resolved_executable"]
            == parent["interpreter"]["resolved_executable"], "native interpreter differs")
    core = regular_record(current["native_core"]["path"])
    require(core["sha256"] == diagnostic.CORE, "native core file changed")
    return core


def coarse_identity(signature, pins):
    require(signature["layout"]["ndof"] == 316989
            and signature["layout"]["dof_layout_sha256"] == COARSE_LAYOUT
            and signature["layout"]["mode_setup"] == "azimuthal"
            and signature["layout"]["base_mode"] == 0
            and signature["arrays"]["positional_dof_mask"]["sha256"] == COARSE_MASK
            and signature["arrays"]["interface_z"]["shape"]
            == signature["arrays"]["interface_r"]["shape"] == [1481]
            and signature["serialized_state"]["payload_sha256"] == COARSE_PAYLOAD
            and signature["parameters"] == {"Ca_in": iii.TARGET_CA, "Bo": 1.56}
            and signature["bc_cache_marker"] == "init"
            and pins["sha256"] == COARSE_PINS, "coarse state/layout/forcing/history/pins differs")


def archive_contract(path, runtime, records):
    record = regular_record(path)
    require(record["sha256"] == ARCHIVE_SHA, "sealed archive receipt differs")
    archive = runner._read_json(Path(record["path"]))
    require(archive["schema"] == "figure10-next-phase-archive-v1"
            and archive["source_deleted"] is False
            and archive["verified_at"] and archive["protocol_files_verified_at"]
            and archive["file_count"] == len(archive["files"]), "archive receipt is incomplete")
    source = Path(archive["source"]).resolve(strict=True)
    require(runtime.is_relative_to(source), "archive belongs to another stationary origin")
    for item in records:
        authority = Path(item["path"])
        if authority.is_relative_to(source):
            retained = archive["files"].get(str(authority.relative_to(source)), {})
            require(retained.get("sha256") == item["sha256"]
                    and retained.get("bytes") == authority.stat().st_size,
                    "archive does not bind the stationary origin input")
    return record


def origin_evidence(args, case):
    """Recheck the exact sealed stationary closure, independently of N1 spectra."""
    from runners import run_paper_modes as modes

    manifest_path = Path(regular_record(args.manifest)["path"])
    runtime = manifest_path.parent
    require(manifest_path.name == "run-manifest.json", "canonical fine stationary manifest required")
    records = []
    for name, digest in SEALED_ORIGIN.items():
        record = regular_record(runtime / name)
        require(record["sha256"] == digest, "sealed III origin differs: " + name)
        records.append(record)
    parent = runner._read_json(manifest_path)
    require(parent["schema"] == "pyoomph-runner-manifest-v1"
            and Path(parent["runner"]["path"]).name == Path(iii.ENTRYPOINT).name
            and parent["qualification_schema"] == iii.SCHEMA
            and parent["anchor_stage"] == {"kind": iii.KIND}
            and parent["component_commit"] == BASE
            and parent["case_fields_used"] == model.CASE_FIELDS_USED
            and parent["azimuthal_stability"] is True
            and parent["parent_acceptance_changed"] is False,
            "fine stationary origin type differs")
    require(parent["case_sha256"] == sha(args.case) == CASE_SHA
            and parent["case_file"] == parent["case_path"]
            and model.read_case(Path(parent["case_path"])) == case,
            "exact unchanged III case required")
    records.extend((regular_record(args.case), regular_record(parent["case_path"])))
    completion = runner._read_json(runtime / "completion.json")
    files = completion["files"]
    require(not (runtime / "failure.json").exists()
            and completion["schema"] == iii.SCHEMA
            and completion["status"] == "complete"
            and completion["numerical_acceptance"] is True
            and completion["anchor_accepted"] is True
            and completion["parent_acceptance_changed"] is False
            and iii.inventory(runtime) == files, "fine stationary closure is not complete")
    records.extend(iii.completion_records(runtime, files))
    preparation = runner._read_json(runtime / "preparation.json")
    require(preparation["implementation"] == parent["implementation"]
            and preparation["contract"] == parent["case_transition_contract"]
            and preparation["numerical_acceptance"] is False, "origin preparation differs")
    records.extend(preparation["inputs"])
    # These are historical packets checked at their sealed producer revision.
    # The current diagnostic-only source compatibility is checked separately.
    def historical_packet(path, selected_case, selected_sha):
        return runner.validate_restart(path, selected_case, selected_sha,
                                       component=parent["component"], environment=parent["pyoomph"],
                                       azimuthal_stability=True)

    fine = historical_packet(runtime / "checkpoints/bond_016/resume.json", case, CASE_SHA)
    coarse = historical_packet(args.packet, case, CASE_SHA)
    require(Path(coarse["packet"]["path"]) == runtime / "stages/ca_003/checkpoints/bond_016/resume.json"
            and Path(coarse["state"]["path"]) == args.state.resolve(strict=True)
            and coarse["state"]["sha256"] == COARSE_STATE_SHA
            and [mesh["sha256"] for mesh in coarse["meshes"]] == [COARSE_MESH_SHA],
            "only the original post-polish coarse checkpoint is admitted")
    result, summary, cold = [runner._read_json(runtime / name)
                             for name in ("result.json", "summary.json", "cold-load.json")]
    require(result["schema"] == iii.SCHEMA and result["numerical_acceptance"] is True
            and result["completion_reached"] is True and result["cold_load_verified"] is True
            and result["parent_acceptance_changed"] is False and result["parents_unchanged"] is True
            and summary["completion_reached"] is True and summary["cold_load_verified"] is True
            and summary["ca_in"] == result["ca_in"] == iii.TARGET_CA
            and fine["roundtrip"]["layout"]["ndof"] == 542833,
            "fine stationary acceptance differs")
    for value in (parent, result, summary):
        require(all(value.get(name, False) is False for name in iii.NO_CLAIMS),
                "stationary origin contains unestablished spectral/dynamic claims")
    for value in (result["canonical_checks"], result["cold_load_checks"], summary["checks"],
                  summary["cold_load_checks"], cold["canonical_checks"]):
        require(set(value) == iii.GATES and all(item is True for item in value.values()),
                "fine stationary five-gate closure failed")
    warm = result["terminal"]["after_polish"]
    require(warm == runner._read_json(runtime / "terminal/samples/after-polish.json")
            and warm["sampling_preserved_state"] is True
            and remesh.canonical_checks(case, warm["metrics"]) == result["canonical_checks"]
            and remesh.canonical_checks(case, summary) == summary["checks"]
            and iii.physical_signature(warm["signature"])
            == iii.physical_signature(fine["roundtrip"]["after"]), "fine warm observation differs")
    records.extend(iii.validate_load(cold, fine, case, warm["actual_pins"],
                                    fine["roundtrip"]["after"]["serialized_state"]["payload_sha256"],
                                    runtime, files))
    records.extend(modes.bind_pressure_history_audit(runtime, result["terminal"], files))
    for phase in ("before", "after_remesh"):
        records.extend(modes.bind_portable_observation(result["terminal"][phase],
                                                      phase.replace("_", "-"), runtime, files))
    require(len(result["stages"]) == len(iii.LADDER), "original III stage count differs")
    for index, (value, row) in enumerate(zip(iii.LADDER, result["stages"], strict=True), 1):
        directory = runtime / "stages" / f"ca_{index:03d}"
        selected_case = iii.anchor.stage_case(case, value, index, index == len(iii.LADDER))
        require(row == runner._read_json(directory / "stage.json")
                and row["numerical_acceptance"] is False and row["index"] == index
                and row["ca_in"] == value
                and model.read_case(directory / "case.json") == selected_case
                and row["polish"] == runner._read_json(directory / "polish/ca_step/polish.json")
                and iii.polish_bounds(row["polish"]), "original provisional polish differs")
        packet = historical_packet(directory / relative_path(row["checkpoint"]["resume_packet"]),
                                   selected_case, sha(directory / "case.json"))
        sample = runner._read_json(directory / "samples/after-polish.json")
        require(packet["packet"]["sha256"] == row["resume_packet_sha256"]
                and sample["sampling_preserved_state"] is True
                and iii.physical_signature(sample["signature"])
                == iii.physical_signature(packet["roundtrip"]["after"])
                and row["canonical_checks_observed"] == remesh.canonical_checks(selected_case, sample["metrics"]),
                "stage checkpoint/observation differs")
    observed = runner._read_json(runtime / "stages/ca_003/samples/after-polish.json")
    coarse_identity(observed["signature"], observed["actual_pins"])
    require(set(remesh.canonical_checks(case, observed["metrics"])) == iii.GATES
            and all(remesh.canonical_checks(case, observed["metrics"]).values())
            and coarse["roundtrip"]["snapshot_restore"]["verified"] is True
            and coarse["stage"]["kind"] == "completed_polish"
            and result["terminal"]["before"]["retained_state"]["numerical_acceptance"] is False
            and result["terminal"]["before"]["retained_state"]["state"]["sha256"] == COARSE_STATE_SHA,
            "coarse warm gates/original restoration or false acceptance differs")
    saved = Path(coarse["state"]["path"]).read_bytes()
    metadata = history.serialized_metadata(saved)
    require(metadata["endtime"] is None and metadata["time"] == 8
            and metadata["dts"] == [1, 1] and metadata["unsteady_steps"] == 15
            and metadata["suggested_next_dt"] == 4
            and hashlib.sha256(history.normalized_bytes(saved, Path(coarse["mesh_reference_base"]))).hexdigest()
            == COARSE_STREAM, "coarse complete native history/endtime differs")
    records = unique_records(records)
    archive = archive_contract(args.archive, runtime, records)
    records.append(archive)
    return {
        "parent": parent, "restart": coarse, "fine_restart": fine, "warm": observed,
        "records": unique_records(records),
        "origin": {"manifest": regular_record(manifest_path), "packet": coarse["packet"],
                   "state": coarse["state"], "case": regular_record(args.case), "archive": archive,
                   "fine_packet": fine["packet"], "fine_state": fine["state"],
                   "fine_numerical_acceptance": True, "original_stage_numerical_acceptance": False,
                   "original_observation_numerical_acceptance": False},
    }


def seal_contract(purpose, component, implementation, environment, records):
    return {
        "schema": SEAL_SCHEMA, "purpose": purpose,
        "source_commit": component["commit"],
        "source_files": {**{name: item["sha256"] for name, item in component["sources"].items()},
                         **{name: item["sha256"] for name, item in implementation.items()}},
        "pyoomph_commit": environment["commit"], "native_core": environment["native_core"]["sha256"],
        "interpreter": environment["interpreter"]["resolved_executable"],
        "inputs": unique_records(records),
    }


def execution_seal(args, component, implementation, environment, records, purpose):
    expected = seal_contract(purpose, component, implementation, environment, records)
    pending = []
    if component["clean"] is not True:
        pending.append("clean committed source seal pending")
    seal_path = getattr(args, "seal", None)
    record = None
    if seal_path is None:
        pending.append("external source/case/packet/archive execution seal pending")
    else:
        record = regular_record(seal_path)
        require(runner._read_json(Path(record["path"])) == expected, "execution seal differs from actual inputs")
    if not args.check:
        runner.require_clean_source(component)
        require(not pending, "native execution requires the external source/case/packet/archive seal")
    return expected, pending, record


def preflight(args):
    require(args.source.resolve(strict=True) == ROOT, "runner must use its own isolated source")
    for module, name in ((iii, iii.ENTRYPOINT), (iii.anchor, iii.OLD_HELPERS[0]),
                         (model, "problems/paper_bubble.py"), (runner, "runners/run_paper_case.py"),
                         (diagnostic, iii.OLD_HELPERS[2]), (remesh, iii.OLD_HELPERS[1]),
                         (history, iii.NEW_HELPERS[0]), (history.seed, iii.NEW_HELPERS[1])):
        require(Path(module.__file__).resolve() == ROOT / name, "helper imported from another component")
    case = model.read_case(args.case)
    require(sha(args.case) == CASE_SHA and case["physics"]["ca_in"] == iii.TARGET_CA,
            "only the exact retained III case is admitted")
    original = origin_evidence(args, case)
    component, implementation, compatibility = source_contract(original["parent"], metadata_only=args.check)
    environment = runner.pyoomph_revision()
    core = environment_contract(environment, original["parent"]["pyoomph"])
    records = unique_records([*original["records"], core])
    seal, pending, seal_record = execution_seal(args, component, implementation, environment, records, "producer")
    return {**original, "case": case, "component": component, "implementation": implementation,
            "environment": environment, "compatibility": compatibility, "sealed_records": records,
            "records": unique_records([*records, *([seal_record] if seal_record else [])]),
            "required_seal": seal, "source_seals_pending": pending, "input_seal": seal_record,
            "execution_authorized": not args.check and not pending}


def stable_inputs(prepared):
    remesh.verify_inputs(prepared["records"])
    remesh.verify_source(prepared["component"], prepared["implementation"])


@contextmanager
def zero_work(problem):
    """Reject solve, factorisation, evolution and actual remesh entrypoints."""
    counts, restored = dict(ZERO_WORK), []
    def forbid(owner, name, category):
        if not callable(getattr(owner, name, None)):
            return
        previous = owner.__dict__.get(name)
        present = name in owner.__dict__
        def denied(*args, **kwargs):
            counts[category] += 1
            raise RuntimeError("cold observation prohibits " + name)
        setattr(owner, name, denied)
        restored.append((owner, name, present, previous))
    try:
        for category, names in {
            "nonlinear_solves": ("solve", "solve_stationary", "actions_before_newton_solve"),
            "eigen_solves": ("solve_eigenproblem",),
            "remeshes": ("force_remesh", "actions_before_remeshing"),
            "time_advances": ("run", "run_with_adaptive_timestepping", "solve_unsteady"),
        }.items():
            for name in names:
                forbid(problem, name, category)
        def guard_linear_solver():
            solver = problem.get_la_solver()
            require(getattr(solver, "idname", None) == "superlu"
                    and callable(getattr(solver, "solve_serial", None)), "actual coarse observation solver differs")
            for name in ("solve_serial", "solve_distributed"):
                forbid(solver, name, "linear_factorizations")
        yield counts, guard_linear_solver
        require(counts == ZERO_WORK, "cold observation performed forbidden work")
    finally:
        for owner, name, present, previous in reversed(restored):
            if present:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)


def copy_inputs(prepared, output):
    directory = output / "inputs"
    directory.mkdir()
    original = prepared["restart"]
    state = directory / "state.dump"
    diagnostic.atomic(state, Path(original["state"]["path"]).read_bytes())
    meshes = []
    for mesh in original["meshes"]:
        name = relative_path(mesh["stored_path"])
        require(len(name.parts) == 1, "coarse portable input mesh must be local")
        target = directory / name
        diagnostic.atomic(target, Path(mesh["path"]).read_bytes())
        meshes.append({**mesh, **regular_record(target)})
    diagnostic.atomic(directory / "origin-resume.json", Path(original["packet"]["path"]).read_bytes())
    require(sha(state) == COARSE_STATE_SHA and [item["sha256"] for item in meshes] == [COARSE_MESH_SHA],
            "private coarse input copy differs")
    return {**original, "state": {**regular_record(state), "format": "portable_file"},
            "meshes": meshes, "mesh_reference_base": str(directory),
            "mesh_exchange": {item["stored_path"]: item["path"] for item in meshes}}


def manifest(args, prepared, argv):
    own = prepared["implementation"][ENTRYPOINT]
    return {
        "schema": "pyoomph-runner-manifest-v1", "created_at": datetime.now(UTC).isoformat(),
        "case_path": str(args.case.resolve()), "case_file": str(args.case.resolve()),
        "case_sha256": sha(args.case), "case_id": prepared["case"]["case_id"],
        "case_fields_used": model.CASE_FIELDS_USED, "azimuthal_stability": True,
        "runner": own, "runner_path": own["path"], "runner_sha256": own["sha256"],
        "argv": [sys.executable, own["path"], *argv],
        "component": prepared["component"], "component_commit": prepared["component"]["commit"],
        "pyoomph": prepared["environment"], "implementation": prepared["implementation"],
        "qualification_schema": SCHEMA, "anchor_stage": {"kind": KIND},
        "origin": prepared["origin"], "source_compatibility": prepared["compatibility"],
        "input_seal": prepared["input_seal"], "qualification_scope": SCOPE, **NO_CLAIMS,
    }


def declared_configuration(case):
    return {"global_parameters": {"Bo": case["physics"]["bond"], "Ca_in": case["physics"]["ca_in"]},
            "geometry": asdict(model.case_geometry(case)),
            **{name: case[name] for name in ("physics", "boundaries", "pressure", "closure",
                                           "mesh", "tolerances", "solve")}}


def execute(args, prepared, output):
    require(prepared.get("execution_authorized") is True, "metadata inspection cannot authorise native execution")
    stable_inputs(prepared)
    import numpy as np

    from runners import run_paper_modes as modes

    case = prepared["case"]
    copied = copy_inputs(prepared, output)
    rows, instances = [], []
    last_load = last_sample = None
    for index in (1, 2):
        directory = output / f"fresh-{index}"
        directory.mkdir()
        stable_inputs(prepared)
        with model.build_problem(case, initial_bond=case["physics"]["bond"]) as problem:
            require(all(problem is not previous for previous in instances), "a cold instance was reused")
            instances.append(problem)
            with zero_work(problem) as (counts, guard_linear_solver):
                problem._stationary_iii_receipts_directory = directory / "stationary-loads"
                iii.anchor.configure(problem, directory, prepared)
                require(not problem.is_distributed(), "serial coarse observation required")
                guard_linear_solver()
                loaded = iii.restore_checked(problem, case, copied,
                                             prepared["warm"]["actual_pins"], COARSE_PAYLOAD)
                controls = remesh.configuration(problem, case, model)
                require(controls == declared_configuration(case), "cold observation controls differ")
                before = modes.take_state(problem, mesh_authority=copied["meshes"])
                modes.save_state_receipt(directory, "before", before)
                sample = remesh.sample(problem, model, case, directory, "cold-load", np)
                checks = remesh.canonical_checks(case, sample["metrics"])
                require(set(checks) == iii.GATES and all(checks.values()), "fresh coarse cold gates failed")
                require(before["signature"] == modes.take_state(problem, mesh_authority=copied["meshes"])["signature"],
                        "cold observations changed the complete native state")
                restored = modes.restore_state(problem, before)
                after = modes.take_state(problem, mesh_authority=copied["meshes"])
                require(remesh.configuration(problem, case, model) == controls
                        and sample["configuration"] == controls, "cold restore or sampling changed native controls")
                modes.save_state_receipt(directory, "after", after)
                write_json(directory / "restoration.json", restored)
                coarse_identity(loaded["loaded"], loaded["actual_pins"])
                require(iii.physical_signature(loaded["loaded"])
                        == iii.physical_signature(sample["signature"]), "cold sampled state differs")
                loaded["canonical_checks"] = checks
                write_json(directory / "load.json", loaded)
            rows.append({"fresh_instance": index, "load": regular_record(directory / "load.json"),
                         "sample": regular_record(directory / "samples/cold-load.json"),
                         "restoration": regular_record(directory / "restoration.json"),
                         "snapshot_base": str(Path.cwd()), "configuration": controls, "work_counts": counts})
            last_load, last_sample = loaded, sample
        stable_inputs(prepared)
    packet_path = output / "checkpoints/coarse/resume.json"
    packet_path.parent.mkdir(parents=True)
    packet = {
        "schema": PACKET_SCHEMA, "qualification_schema": SCHEMA,
        "parent_manifest": regular_record(output / "run-manifest.json"),
        "case_sha256": sha(args.case), "azimuthal_stability": True,
        "state": copied["state"], "meshes": copied["meshes"],
        "mesh_reference_base": copied["mesh_reference_base"],
        "stage": {**copied["stage"], "kind": KIND},
        "roundtrip": {"kind": "cold_observation_full_snapshot_restore", "verified": True,
                      "after": last_load["loaded"], "layout": last_load["loaded"]["layout"]},
        "origin_completed_polish": prepared["restart"]["packet"],
        "fresh_loads": rows, "input_seal": prepared["input_seal"], **NO_CLAIMS,
    }
    write_json(packet_path, packet)
    write_json(output / "cold-load.json", last_load)
    result = {"schema": SCHEMA, "completion_reached": True, "numerical_acceptance": True,
              "cold_load_verified": True, "fresh_loads": rows, "work_counts": dict(ZERO_WORK),
              "canonical_checks": last_load["canonical_checks"],
              "cold_load_checks": last_load["canonical_checks"],
              "checkpoint": str(packet_path.relative_to(output)), "scope": SCOPE, **NO_CLAIMS}
    write_json(output / "result.json", result)
    write_json(output / "summary.json", {
        "schema": SCHEMA, "status": "completed", "completion_reached": True,
        "numerical_acceptance": True, "cold_load_verified": True, "fresh_cold_loads": 2,
        "checks": last_load["canonical_checks"], "cold_load_checks": last_load["canonical_checks"],
        "checkpoint": result["checkpoint"], "ca_in": case["physics"]["ca_in"],
        "bond": case["physics"]["bond"], **last_sample["metrics"], "scope": SCOPE, **NO_CLAIMS,
    })
    return result


def read_packet(path, runtime, case, parent, original):
    """Read only this cold-observation schema; the generic resume reader is unchanged."""
    path = Path(regular_record(path)["path"])
    packet = runner._read_json(path)
    require(packet["schema"] == PACKET_SCHEMA and packet["qualification_schema"] == SCHEMA
            and packet["case_sha256"] == CASE_SHA and packet["azimuthal_stability"] is True
            and packet["parent_manifest"] == regular_record(runtime / "run-manifest.json")
            and packet["stage"] == {**original["restart"]["stage"], "kind": KIND}
            and packet["roundtrip"]["kind"] == "cold_observation_full_snapshot_restore"
            and packet["roundtrip"]["verified"] is True
            and packet["roundtrip"]["layout"] == packet["roundtrip"]["after"]["layout"]
            and packet["origin_completed_polish"] == original["restart"]["packet"]
            and packet["input_seal"] == parent["input_seal"], "coarse observation packet type or origin differs")
    require(all(packet.get(name) is False for name in NO_CLAIMS), "cold packet overclaims acceptance")
    state = runner._bound_file(packet["state"], path.parent)
    base = Path(packet["mesh_reference_base"]).resolve(strict=True)
    require(path == runtime / "checkpoints/coarse/resume.json" and state.is_relative_to(runtime)
            and packet["state"]["format"] == "portable_file" and base == state.parent
            and packet["state"]["sha256"] == COARSE_STATE_SHA,
            "coarse packet does not bind its own unchanged input dump")
    references = runner.state_mesh_references(state)
    require(len(references) == len(packet["meshes"]) == 1, "coarse packet mesh closure differs")
    meshes = []
    for reference, record in zip(references, packet["meshes"], strict=True):
        mesh = runner._bound_file(record, path.parent)
        require(record["stored_path"] == reference and mesh == (base / relative_path(reference)).resolve(strict=True)
                and mesh.is_relative_to(runtime) and record["sha256"] == COARSE_MESH_SHA,
                "coarse packet mesh reference/content differs")
        meshes.append({**record, "path": str(mesh)})
    require(history.serialized_metadata(state.read_bytes())["endtime"] is None
            and hashlib.sha256(history.normalized_bytes(state.read_bytes(), base)).hexdigest() == COARSE_STREAM,
            "coarse copied native stream or writer marker differs")
    return {**packet, "packet": regular_record(path), "state": {**packet["state"], "path": str(state)},
            "meshes": meshes, "mesh_exchange": {item["stored_path"]: item["path"] for item in meshes}}


def validate_witness(row, restart, case, pins, runtime, files):
    """Check raw complete streams and the field/layout/geometry restore witnesses."""
    import numpy as np

    from runners import run_paper_modes as modes

    records = []
    load_path = Path(iii.bound(row["load"], runtime, files)["path"])
    directory = load_path.parent
    require(directory == runtime / f"fresh-{row['fresh_instance']}"
            and load_path.name == "load.json" and row["work_counts"] == ZERO_WORK,
            "fresh load identity or zero-work counters differ")
    receipt = runner._read_json(load_path)
    coarse_identity(receipt["loaded"], receipt["actual_pins"])
    records.extend(iii.validate_load(receipt, restart, case, pins, COARSE_PAYLOAD, runtime, files))
    sample_path = Path(iii.bound(row["sample"], runtime, files)["path"])
    restoration_path = Path(iii.bound(row["restoration"], runtime, files)["path"])
    require(sample_path == directory / "samples/cold-load.json"
            and restoration_path == directory / "restoration.json", "fresh observation proof paths differ")
    sample, restored = runner._read_json(sample_path), runner._read_json(restoration_path)
    checks = remesh.canonical_checks(case, sample["metrics"])
    require(set(checks) == iii.GATES and all(checks.values())
            and receipt["canonical_checks"] == checks and sample["sampling_preserved_state"] is True
            and sample["configuration"] == row["configuration"] == declared_configuration(case)
            and sample["actual_pins"] == pins
            and iii.physical_signature(sample["signature"]) == iii.physical_signature(receipt["loaded"]),
            "fresh coarse sampled cold predicates/state differ")
    signatures = []
    for phase in ("before", "after"):
        paths = [directory / (phase + suffix) for suffix in (".json", ".dump", ".npz")]
        records.extend(iii.bound(regular_record(path), runtime, files) for path in paths)
        declared = runner._read_json(paths[0])
        raw = paths[1].read_bytes()
        require(declared["raw_snapshot_sha256"] == hashlib.sha256(raw).hexdigest()
                and history.serialized_metadata(raw)["endtime"] is None
                and runner.state_bc_cache_marker(paths[1]) == declared["bc_cache_marker"]
                == sample["signature"]["bc_cache_marker"]
                and declared["serialized_state"] == modes.full_mode_snapshot_signature(
                    raw, mesh_reference_base=Path(row["snapshot_base"]), mode=0,
                    bc_marker=declared["bc_cache_marker"], mesh_authority=restart["meshes"])
                and runner.snapshot_bc_cache_signature(raw, mesh_reference_base=Path(row["snapshot_base"]))[
                    "payload_sha256"] == COARSE_PAYLOAD,
                "complete native restore witness/history differs")
        signature = {key: value for key, value in declared.items() if key != "raw_snapshot_sha256"}
        require(signature["actual_pins"] == pins and signature["mode"] == 0
                and signature["parameters"] == {"Ca_in": iii.TARGET_CA, "Bo": 1.56}
                and signature["time"] == sample["signature"]["time"]
                and signature["volume"] == sample["signature"]["volume"]
                and signature["frame_speed"] == sample["signature"]["frame_speed"],
                "native restore scalar controls differ")
        with np.load(paths[2], allow_pickle=False) as arrays:
            for name, record in signature["arrays"].items():
                require(modes.array_record(arrays[name]) == record, "restore field/position/layout array differs")
            require(arrays["dof_names"].tolist() == signature["dof_names"]
                    == restart["roundtrip"]["layout"]["dof_names"], "restore field names differ")
            digest = hashlib.sha256(json.dumps(signature["dof_names"], separators=(",", ":")).encode()
                                    + np.asarray(arrays["dof_types"], dtype="<i8").tobytes()).hexdigest()
            require(digest == COARSE_LAYOUT, "restore DOF incidence/layout differs")
            for actual, legacy in (("free_values", "free_dofs"), ("positional_mask", "positional_dof_mask"),
                                   ("pinned_values", "pinned_values_with_positions")):
                require(modes.array_record(arrays[actual]) == sample["signature"]["arrays"][legacy],
                        "restore fields do not match the cold sample")
            interface = arrays["interface"]
            require(interface.shape == (1481, 2), "restore ordered interface differs")
            for column, name in enumerate(("interface_z", "interface_r")):
                legacy = np.asarray([round(float(value), 12) for value in interface[:, column]])
                require(modes.array_record(legacy) == sample["signature"]["arrays"][name],
                        "raw interface witness differs from its legacy signature")
        signatures.append(signature)
    require(restored["verified"] is True and restored["before"] == restored["after"]
            == signatures[0] == signatures[1], "complete native snapshot restoration failed")
    records.extend((regular_record(load_path), regular_record(sample_path), regular_record(restoration_path)))
    return receipt, sample, records


def admit_observation(runtime, parent, restart, case, original=None):
    runtime = Path(runtime).resolve(strict=True)
    require(is_producer(parent) and parent["schema"] == "pyoomph-runner-manifest-v1"
            and parent["qualification_schema"] == SCHEMA and parent["anchor_stage"] == {"kind": KIND}
            and not (runtime / "failure.json").exists(), "qualified coarse cold producer required")
    if original is None:
        origin = parent["origin"]
        original = origin_evidence(SimpleNamespace(manifest=Path(origin["manifest"]["path"]),
                                                   packet=Path(origin["packet"]["path"]),
                                                   state=Path(origin["state"]["path"]),
                                                   case=Path(origin["case"]["path"]),
                                                   archive=Path(origin["archive"]["path"])), case)
    completion = runner._read_json(runtime / "completion.json")
    files = completion["files"]
    require(completion["schema"] == SCHEMA and completion["status"] == "complete"
            and completion["numerical_acceptance"] is True and iii.inventory(runtime) == files
            and parent["origin"] == original["origin"], "coarse completion inventory/origin differs")
    records = [regular_record(runtime / "completion.json"), *iii.completion_records(runtime, files)]
    result, summary, cold = [runner._read_json(runtime / name)
                             for name in ("result.json", "summary.json", "cold-load.json")]
    for value in (parent, completion, result, summary):
        require(all(value.get(name) is False for name in NO_CLAIMS), "coarse observation overclaims acceptance")
    rows = result["fresh_loads"]
    require(result["schema"] == summary["schema"] == SCHEMA
            and result["numerical_acceptance"] is summary["numerical_acceptance"] is True
            and result["completion_reached"] is summary["completion_reached"] is True
            and result["cold_load_verified"] is summary["cold_load_verified"] is True
            and summary["fresh_cold_loads"] == 2 and result["work_counts"] == ZERO_WORK
            and len(rows) == 2 and [row["fresh_instance"] for row in rows] == [1, 2]
            and restart["fresh_loads"] == rows
            and runtime / relative_path(summary["checkpoint"]) == Path(restart["packet"]["path"])
            and summary["checkpoint"] == result["checkpoint"], "exact two-fresh-load qualification required")
    for record in (restart["packet"], restart["state"], *restart["meshes"]):
        records.append(iii.bound(record, runtime, files))
    last_receipt = last_sample = None
    for row in rows:
        last_receipt, last_sample, proof = validate_witness(row, restart, case,
                                                          original["warm"]["actual_pins"], runtime, files)
        records.extend(proof)
    checks = remesh.canonical_checks(case, last_sample["metrics"])
    require(cold == last_receipt and cold["loaded"] == restart["roundtrip"]["after"]
            and result["canonical_checks"] == result["cold_load_checks"]
            == summary["checks"] == summary["cold_load_checks"] == checks
            and all(summary[key] == value for key, value in last_sample["metrics"].items())
            and summary["ca_in"] == iii.TARGET_CA and summary["bond"] == 1.56,
            "coarse summary/packet differs from the second fresh native observation")
    return summary, cold, unique_records(records)


def consumer_preflight(args, parent, case):
    from runners import run_paper_modes as modes

    require(sha(args.case) == parent["case_sha256"] == CASE_SHA
            and parent["case_id"] == case["case_id"]
            and parent["case_fields_used"] == model.CASE_FIELDS_USED
            and parent["azimuthal_stability"] is True
            and model.read_case(Path(parent["case_path"])) == case
            and parent["case_file"] == parent["case_path"], "coarse consumer case/layout differs")
    origin = parent["origin"]
    original = origin_evidence(SimpleNamespace(manifest=Path(origin["manifest"]["path"]),
                                               packet=Path(origin["packet"]["path"]),
                                               state=Path(origin["state"]["path"]),
                                               case=Path(origin["case"]["path"]),
                                               archive=Path(origin["archive"]["path"])), case)
    require(not args.out.expanduser().resolve().is_relative_to(Path(original["origin"]["manifest"]["path"]).parent),
            "coarse consumer output overlaps the sealed stationary origin")
    component, implementation, compatibility = source_contract(original["parent"], metadata_only=args.check)
    runner.require_clean_source(parent["component"])
    require(parent["component_commit"] == parent["component"]["commit"], "producer source declarations differ")
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", parent["component_commit"], component["head"])
    require({name: item["sha256"] for name, item in component["sources"].items()}
            == {name: item["sha256"] for name, item in parent["component"]["sources"].items()}
            and {name: item["sha256"] for name, item in implementation.items()}
            == {name: item["sha256"] for name, item in parent["implementation"].items()}
            and parent["source_compatibility"] == {**compatibility, "current_commit": parent["component_commit"]},
            "coarse producer/current implementation differs")
    own = parent["runner"]
    require(own == parent["implementation"][ENTRYPOINT] and parent["runner_path"] == own["path"]
            and parent["runner_sha256"] == own["sha256"]
            and hashlib.sha256(diagnostic.git(ROOT, "show", f"{parent['component_commit']}:{ENTRYPOINT}")).hexdigest()
            == own["sha256"], "coarse producer is not bound to its committed source")
    environment = runner.pyoomph_revision()
    core = environment_contract(environment, parent["pyoomph"])
    producer_records = unique_records([*original["records"], core])
    producer_seal = regular_record(parent["input_seal"]["path"])
    runtime = args.manifest.resolve(strict=True).parent
    require(producer_seal == parent["input_seal"]
            and sha(runtime / "input-seal.json") == producer_seal["sha256"]
            and runner._read_json(runtime / "input-seal.json") == seal_contract(
                "producer", parent["component"], parent["implementation"], parent["pyoomph"], producer_records),
            "producer source/case/packet/archive seal differs")
    restart = read_packet(args.packet, runtime, case, parent, original)
    require(Path(restart["state"]["path"]) == args.state.resolve(strict=True), "coarse consumer state differs")
    _summary, cold, records = admit_observation(runtime, parent, restart, case, original)
    records = unique_records([*records, *producer_records, producer_seal,
                              regular_record(args.case), regular_record(parent["case_path"])])
    seal, pending, seal_record = execution_seal(args, component, implementation, environment, records, "m0-consumer")
    return {"case": case, "request": modes.request_contract(case, args.shifts, args.neigen, coarse_iii=True),
            "restart": restart, "cold": cold, "component": component, "implementation": implementation,
            "environment": environment, "records": unique_records([*records, *([seal_record] if seal_record else [])]),
            "required_seal": seal, "source_seals_pending": pending,
            "execution_authorized": not args.check and not pending}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "archive", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    output = runner.validate_output_path(args.out, ROOT)
    require(not output.is_relative_to(Path(prepared["origin"]["manifest"]["path"]).parent.parent),
            "output overlaps the sealed stationary origin")
    if args.check:
        print(json.dumps({"verified": True, "native_imported": False, "native_execution_authorized": False,
                          "source_seals_pending": prepared["source_seals_pending"],
                          "required_seal": prepared["required_seal"], "coarse_dofs": 316989,
                          "fresh_cold_loads_requested": 2, "work_budget": ZERO_WORK}, allow_nan=False))
        return 0
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    diagnostic.atomic(output / "input-seal.json", Path(prepared["input_seal"]["path"]).read_bytes())
    write_json(output / "run-manifest.json", manifest(args, prepared, arguments))
    runner.append_progress(output, {"event": "prepared", "case_id": prepared["case"]["case_id"]}, first=True)
    try:
        write_json(output / "preparation.json", {"inputs": prepared["records"],
                                                "source_compatibility": prepared["compatibility"],
                                                "numerical_acceptance": False})
        execute(args, prepared, output)
        stable_inputs(prepared)
        write_json(output / "completion.json", {"schema": SCHEMA, "status": "complete",
                                                "numerical_acceptance": True, "files": iii.inventory(output),
                                                "scope": SCOPE, **NO_CLAIMS})
        return 0
    except (Exception, KeyboardInterrupt, SystemExit) as error:  # noqa: BLE001 - retain every terminal failure
        write_json(output / "failure.json", {"schema": SCHEMA, "status": "inconclusive_cold_observation",
                                             "error_type": type(error).__name__, "error": str(error),
                                             "numerical_acceptance": False, **NO_CLAIMS})
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
