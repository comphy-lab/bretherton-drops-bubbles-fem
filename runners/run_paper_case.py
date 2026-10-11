"""Run one paper-model case with exclusive output and recorded provenance.

``--check`` validates the case and output custody without importing the native
solver or creating output. Execution requires committed source and the pinned
pyoomph distribution, and writes its manifest before the first progress event.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

# Validation must also leave the component free of import-generated files.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.paper_bubble import (
    CASE_FIELDS_USED,
    PYOOMPH_COMMIT,
    read_case,
    stationary_bond_schedule,
)

MANIFEST = "run-manifest.json"
PROGRESS = "progress.jsonl"
METHOD_FILES = (
    "runners/run_paper_case.py",
    "problems/paper_bubble.py",
    "problems/paper_mesh.py",
    "pyproject.toml",
    "uv.lock",
)
RESUME_SCHEMA = "herrada-paper-stationary-resume-v1"
PHYSICS_CONTRACT_NODES = {
    "PYOOMPH_COMMIT", "CASE_SECTIONS", "CASE_FIELDS_USED", "PROBLEMS", "MATERIAL_RATIOS",
    "MIN_SEED_CYLINDER_OVER_R0", "_number", "validate_case", "read_case", "case_geometry",
    "phase_force_density", "analytic_tube_fields", "bulk_velocity_rms", "_bulk_viscous_stress",
    "_native_problem_types", "build_problem", "_stationary_residual", "polish_confined_state",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> bytes | None:
    """Read Git metadata with argument arrays and without index refresh writes."""
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            capture_output=True, check=False, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def component_revision(root: Path = ROOT) -> dict:
    """Record the actual checkout, including untracked or altered method files.

    This function may describe dirty source for software diagnostics; only
    ``require_clean_source`` authorises production execution.
    """
    root = root.resolve()
    top = _git(root, "rev-parse", "--show-toplevel")
    head_bytes = _git(root, "rev-parse", "--verify", "HEAD")
    status_bytes = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    head = head_bytes.decode("ascii").strip() if head_bytes else None
    own_checkout = bool(top) and Path(os.fsdecode(top).strip()).resolve() == root
    if not own_checkout or not head or len(head) != 40:
        head = None
    status = os.fsdecode(status_bytes).splitlines() if status_bytes is not None else None

    paths = sorted(set(METHOD_FILES) | {
        str(path.relative_to(root)) for path in (root / "problems").glob("*.py")
    })
    sources = {}
    for relative in paths:
        path = root / relative
        regular = path.is_file() and not path.is_symlink()
        actual_hash = sha256_file(path) if regular else None
        committed = _git(root, "show", f"HEAD:{relative}") if head else None
        committed_hash = hashlib.sha256(committed).hexdigest() if committed is not None else None
        sources[relative] = {
            "sha256": actual_hash,
            "committed_sha256": committed_hash,
            "committed_matches": regular and committed_hash == actual_hash,
        }

    commit_file = root / "COMMIT"
    declared_commit = commit_file.read_text(encoding="utf-8").strip() if commit_file.exists() else None
    clean = bool(head) and status == [] and all(
        source["committed_matches"] for source in sources.values()
    ) and (declared_commit is None or declared_commit == head)
    return {
        "commit": head if clean else f"{head}-dirty" if head else "unknown",
        "head": head,
        "clean": clean,
        "status": status,
        "declared_commit": declared_commit,
        "sources": sources,
    }


def require_clean_source(revision: dict) -> None:
    if not isinstance(revision, dict):
        raise ValueError("production source provenance must be an object")  # noqa: TRY004
    head = revision.get("head")
    if (not isinstance(head, str) or len(head) != 40
            or any(character not in "0123456789abcdef" for character in head)
            or revision.get("clean") is not True or revision.get("commit") != head):
        raise ValueError("production execution requires a clean, committed component and method files")


def pyoomph_revision() -> dict:
    """Inspect distribution metadata and native bytes without importing pyoomph."""
    revision = {
        "available": False,
        "version": None,
        "url": None,
        "commit": None,
        "direct_url": None,
        "native_core": None,
        "interpreter": {
            "executable": sys.executable,
            "resolved_executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
    }
    try:
        distribution = metadata.distribution("pyoomph")
        revision["available"] = True
        revision["version"] = distribution.version
        direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
        if not isinstance(direct_url, dict):
            raise TypeError("pyoomph has no direct_url.json object provenance")
        revision["direct_url"] = direct_url
        revision["url"] = direct_url.get("url")
        vcs = direct_url.get("vcs_info", {})
        revision["commit"] = vcs.get("commit_id") if isinstance(vcs, dict) else None
        candidates = [
            Path(distribution.locate_file(item))
            for item in distribution.files or []
            if item.name.startswith("_pyoomph_core") and item.suffix in (".so", ".pyd", ".dylib")
        ]
        if len(candidates) != 1 or not candidates[0].is_file():
            raise ValueError("pyoomph must identify exactly one installed native core")
        native = candidates[0].resolve()
        revision["native_core"] = {"path": str(native), "sha256": sha256_file(native)}
    except (metadata.PackageNotFoundError, OSError, TypeError, ValueError) as exc:
        revision["error"] = str(exc)
    return revision


def require_pinned_environment(revision: dict) -> None:
    if not isinstance(revision, dict):
        raise ValueError("pyoomph environment provenance must be an object")  # noqa: TRY004
    direct_url = revision.get("direct_url") or {}
    vcs = direct_url.get("vcs_info", {}) if isinstance(direct_url, dict) else {}
    native = revision.get("native_core") or {}
    if revision.get("error") or revision.get("available") is not True:
        raise ValueError(f"cannot establish installed pyoomph provenance: {revision.get('error', 'absent')}")
    if (revision.get("commit") != PYOOMPH_COMMIT or not isinstance(vcs, dict)
            or vcs.get("vcs") != "git"):
        raise ValueError(f"installed pyoomph must have exact commit {PYOOMPH_COMMIT}")
    if (not revision.get("url") or not isinstance(native, dict)
            or not native.get("path") or not native.get("sha256")):
        raise ValueError("installed pyoomph provenance requires its URL and native-core digest")


def state_mesh_references(path: Path) -> list[str]:
    """Read only the bounded mesh header of the exact pinned 0.1.6 dump format.

    This is the header layout in a186e07e's output/states.py:241-273. It uses
    no native imports or pickle decoding, and also refuses a missing footer.
    Memory snapshots store mesh names relative to the writer's working
    directory; portable file dumps store them relative to their own directory.
    """
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size

        def integer():
            value = stream.read(8)
            if len(value) != 8:
                raise ValueError("truncated restart state header")
            return int.from_bytes(value, "little", signed=True)

        def string():
            length = integer()
            if not 0 <= length <= 1024 * 1024 or length > size - stream.tell():
                raise ValueError("invalid restart state string length")
            try:
                return stream.read(length).decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError("restart state header must be ASCII") from exc

        footer = b"EOF_pyoomph"
        if size < len(footer) + 8:
            raise ValueError("truncated restart state")
        stream.seek(-len(footer) - 8, os.SEEK_END)
        if integer() != len(footer) or stream.read(len(footer)) != footer:
            raise ValueError("restart state has no complete pyoomph footer")
        stream.seek(0)
        if (string(), string(), string()) != ("pyoomph_dump", "0.1.6", "global"):
            raise ValueError("restart requires the pinned global 0.1.6 state format")
        if len(stream.read(struct.calcsize("<d"))) != 8:
            raise ValueError("truncated restart state time")
        for _ in range(3):
            integer()  # output step, continue-section step, numpy compression.
        count = integer()
        if not 1 <= count <= 128:
            raise ValueError("invalid restart mesh-template count")
        references = []
        for _ in range(count):
            reference = string()
            if reference:
                if "\0" in reference or Path(reference).suffix != ".msh":
                    raise ValueError("restart template must reference a Gmsh mesh")
                references.append(reference)
            has_remesher = integer()
            if has_remesher not in (0, 1):
                raise ValueError("invalid restart remesher flag")
            if has_remesher and integer() < 0:
                raise ValueError("invalid restart remesher counter")
        return references


def _read_json(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")  # noqa: TRY004 - input refusal contract
    return value


def _bound_file(record: dict, base: Path) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("resume artifact requires a path and SHA-256")  # noqa: TRY004
    path = Path(record["path"]).expanduser()
    path = path if path.is_absolute() else base / path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"resume artifact is missing or not a regular file: {path}")
    path = path.resolve(strict=True)
    if record.get("sha256") != sha256_file(path):
        raise ValueError(f"resume artifact digest mismatch: {path}")
    return path


def _physics_contract(source: bytes) -> str:
    tree = ast.parse(source)
    selected = []
    names = set()
    for node in tree.body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
        if name in PHYSICS_CONTRACT_NODES:
            names.add(name)
            selected.append(ast.dump(node, include_attributes=False))
    if names != PHYSICS_CONTRACT_NODES:
        raise ValueError("parent source lacks the complete stationary physics/layout contract")
    return hashlib.sha256("\n".join(selected).encode()).hexdigest()


def _parent_source_contract(parent: dict, current: dict) -> dict:
    component = parent.get("component", {})
    require_clean_source(component)
    commit = component["commit"]
    if (parent.get("component_commit") != commit or component.get("status") != []
            or component.get("declared_commit") not in (None, commit)):
        raise ValueError("parent manifest has inconsistent or dirty source provenance")
    if _git(ROOT, "merge-base", "--is-ancestor", commit, current["head"]) is None:
        raise ValueError("parent component must be a known committed ancestor of the current source")
    sources = component.get("sources", {})
    if not isinstance(sources, dict) or set(sources) != set(current.get("sources", {})):
        raise ValueError("parent method-file inventory differs from the current source")
    for relative, record in sources.items():
        source = _git(ROOT, "show", f"{commit}:{relative}")
        digest = hashlib.sha256(source).hexdigest() if source is not None else None
        if (not isinstance(record, dict) or record.get("committed_matches") is not True
                or digest is None or record.get("sha256") != digest
                or record.get("committed_sha256") != digest):
            raise ValueError(f"parent method bytes are not established by Git: {relative}")
        if (relative not in {"runners/run_paper_case.py", "problems/paper_bubble.py"}
                and current["sources"][relative].get("sha256") != digest):
            raise ValueError(f"stationary restart changed a supporting method: {relative}")
    parent_model = _git(ROOT, "show", f"{commit}:problems/paper_bubble.py")
    current_model = (ROOT / "problems/paper_bubble.py").read_bytes()
    contract = _physics_contract(parent_model)
    if contract != _physics_contract(current_model):
        raise ValueError("stationary restart changed the equations, layout or bounded polish")
    return {"schema": "herrada-stationary-physics-layout-v1", "parent_commit": commit,
            "current_commit": current["commit"], "physics_ast_sha256": contract,
            "unchanged_nodes": sorted(PHYSICS_CONTRACT_NODES),
            "scope": "stationary orchestration only; no dynamic continuation claim"}


def validate_restart(packet_path: Path, case: dict, case_sha256: str, *,
                     component: dict, environment: dict, azimuthal_stability: bool) -> dict:
    """Bind an incomplete parent without fabricating its final completion.

    A packet describes either a retained pre-polish memory snapshot or one of
    this runner's verified portable post-polish checkpoints. All artifact and
    source checks precede output creation and native assembly.
    """
    packet_path = packet_path.expanduser().resolve(strict=True)
    packet = _read_json(packet_path)
    if packet.get("schema") != RESUME_SCHEMA or case["physics"]["problem"] != "finite_gas_bubble":
        raise ValueError("restart requires a finite-gas stationary resume packet")
    parent_path = _bound_file(packet.get("parent_manifest"), packet_path.parent)
    if parent_path.name != MANIFEST:
        raise ValueError("restart requires the canonical parent run-manifest.json")
    parent = _read_json(parent_path)
    if (parent.get("schema") != "pyoomph-runner-manifest-v1"
            or parent.get("case_sha256") != case_sha256
            or parent.get("case_id") != case["case_id"]
            or packet.get("case_sha256") != case_sha256
            or parent.get("case_fields_used") != list(CASE_FIELDS_USED)):
        raise ValueError("restart parent belongs to a different or unqualified case")
    parent_case_path = parent.get("case_path")
    if not isinstance(parent_case_path, str) or parent.get("case_file") != parent_case_path:
        raise ValueError("restart parent lacks a canonical case path")
    parent_case = _bound_file({"path": parent_case_path, "sha256": case_sha256}, parent_path.parent)
    if read_case(parent_case) != case:
        raise ValueError("restart parent case bytes or validated fields differ")
    if (parent.get("azimuthal_stability") is not azimuthal_stability
            or packet.get("azimuthal_stability") is not azimuthal_stability):
        raise ValueError("restart azimuthal layout differs from the selected run")
    parent_environment = parent.get("pyoomph", {})
    require_pinned_environment(parent_environment)
    if (parent_environment.get("direct_url", {}).get("vcs_info", {}).get("commit_id") != PYOOMPH_COMMIT
            or parent_environment["native_core"]["sha256"] != environment["native_core"]["sha256"]
            or parent_environment.get("url") != environment.get("url")):
        raise ValueError("restart native-core bytes or source authority differ from the parent")
    compatibility = _parent_source_contract(parent, component)
    state = _bound_file(packet.get("state"), packet_path.parent)
    runtime = parent_path.parent
    if not state.is_relative_to(runtime):
        raise ValueError("restart state must remain in its parent run")
    state_format = packet["state"].get("format")
    expected_base = runtime.parent if state_format == "memory_snapshot" else state.parent
    if state_format not in {"memory_snapshot", "portable_file"}:
        raise ValueError("restart must declare its snapshot path convention")
    reference_base = packet.get("mesh_reference_base")
    if not isinstance(reference_base, str) or Path(reference_base).resolve() != expected_base:
        raise ValueError("restart mesh reference base differs from the snapshot path convention")
    references = state_mesh_references(state)
    meshes = packet.get("meshes")
    if not references or not isinstance(meshes, list) or len(meshes) != len(references):
        raise ValueError("restart must bind every referenced mesh")
    exchange = {}
    for reference, mesh in zip(references, meshes, strict=True):
        mesh_path = _bound_file(mesh, packet_path.parent)
        if (mesh.get("stored_path") != reference
                or mesh_path != (expected_base / reference).resolve()
                or not mesh_path.is_relative_to(runtime)):
            raise ValueError("restart mesh does not match the saved template reference")
        exchange[reference] = str(mesh_path)
    stage = packet.get("stage", {})
    schedule = stationary_bond_schedule(case)
    index = stage.get("bond_index") if isinstance(stage, dict) else None
    if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(schedule)
            or stage.get("kind") not in {"pending_polish", "completed_polish"}
            or isinstance(stage.get("bond"), bool) or stage.get("bond") != schedule[index]
            or isinstance(stage.get("ca_in"), bool) or stage.get("ca_in") != case["physics"]["ca_in"]
            or stage.get("bc_mode") not in {"transient", "stationary"}):
        raise ValueError("restart stage differs from the unchanged case Bond schedule")
    if stage["kind"] == "completed_polish" and stage["bc_mode"] != "stationary":
        raise ValueError("a completed polish must retain stationary boundary conditions")
    if (state_format == "memory_snapshot" and (stage["kind"] != "pending_polish"
            or len(state.relative_to(runtime).parts) != 3
            or state.relative_to(runtime).parts[0] != "polish" or state.name != "before.dump")):
        raise ValueError("memory restart must bind a retained pre-polish snapshot")
    if stage["kind"] == "completed_polish" and (
            not isinstance(packet.get("roundtrip"), dict)
            or packet["roundtrip"].get("verified") is not True):
        raise ValueError("completed restart requires its verified post-polish round-trip receipt")
    return {**packet, "packet": {"path": str(packet_path), "sha256": sha256_file(packet_path)},
            "parent_manifest": {"path": str(parent_path), "sha256": sha256_file(parent_path)},
            "state": {**packet["state"], "path": str(state)},
            "meshes": [{**mesh, "path": exchange[mesh["stored_path"]]} for mesh in meshes],
            "mesh_exchange": exchange, "compatibility": compatibility}


def verify_restart_artifacts(restart: dict) -> None:
    """Recheck bound bytes immediately before loading an already admitted state."""
    for record in (restart["packet"], restart["parent_manifest"], restart["state"], *restart["meshes"]):
        _bound_file(record, Path("/"))


def validate_output_path(path: Path, root: Path = ROOT) -> Path:
    out = path.expanduser().resolve()
    if out.is_relative_to(root.resolve()):
        raise ValueError("run output must be outside the component repository")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("run output is occupied; use a new or empty directory")
    for parent in out.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ValueError("run output has a parent that is not a directory")
            break
    return out


def _exclusive_json(path: Path, payload: Mapping) -> None:
    contents = (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"output is not a regular file: {path.name}")
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def create_run_manifest(out: Path, manifest: dict) -> None:
    """Write the first run artifact exclusively, preserving an existing file."""
    _exclusive_json(out / MANIFEST, manifest)


def append_progress(out: Path, payload: dict, *, first: bool = False) -> None:
    event = {"timestamp": datetime.now(UTC).isoformat(), **payload}
    contents = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW
    if first:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(out / PROGRESS, flags, 0o600)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def run_model(case: dict, output_path: Path, *, azimuthal_stability: bool = False,
              restart: dict | None = None) -> dict:
    """Import the native execution path only after the manifest and start event."""
    from problems.paper_bubble import run_model as execute

    options = {}
    if azimuthal_stability:
        options["azimuthal_stability"] = True
    if restart is not None:
        options["restart"] = restart
    return execute(case, output_path, **options)


def exit_status(summary: Mapping) -> int:
    return 0 if summary.get("status") == "completed" and summary.get("completion_reached") is True else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("case", type=Path, help="portable paper-model case JSON")
    parser.add_argument("--out", type=Path, help="new or empty output directory outside the component")
    parser.add_argument("--check", action="store_true", help="validate without native execution or output")
    parser.add_argument("--azimuthal-stability", action="store_true",
                        help="prepare azimuthal fields before initialisation for later mode analysis")
    parser.add_argument("--restart-from", type=Path,
                        help="verified stationary resume packet; always uses a new output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    options = parser.parse_args(arguments)
    if options.out is None and not options.check:
        parser.error("--out is required for execution")
    try:
        case_path = options.case.expanduser().resolve(strict=True)
        raw_case = case_path.read_bytes()
        case = read_case(case_path)
        if case_path.read_bytes() != raw_case:
            raise ValueError("case changed during validation")
        out = validate_output_path(options.out) if options.out is not None else None
        if options.check and options.restart_from is None:
            print(json.dumps({"status": "checked", "case_id": case["case_id"]}))
            return 0
        component = component_revision()
        require_clean_source(component)
        environment = pyoomph_revision()
        require_pinned_environment(environment)
        restart = None
        if options.restart_from is not None:
            restart = validate_restart(options.restart_from, case,
                hashlib.sha256(raw_case).hexdigest(), component=component, environment=environment,
                azimuthal_stability=options.azimuthal_stability)
        if options.check:
            print(json.dumps({"status": "checked", "case_id": case["case_id"],
                              "restart": restart}, allow_nan=False))
            return 0
        runner_record = {"path": str(Path(__file__).resolve()),
                         "sha256": sha256_file(Path(__file__))}
        manifest = {
            "schema": "pyoomph-runner-manifest-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "case_file": str(case_path),
            "case_path": str(case_path),
            "case_sha256": hashlib.sha256(raw_case).hexdigest(),
            "case_id": case["case_id"],
            "case_fields_used": list(CASE_FIELDS_USED),
            "units": "nondimensional (sigma=mu_liquid=R0=1)",
            "nondimensionalization": case["nondimensionalization"],
            "argv": [sys.executable, str(Path(__file__).resolve()), *arguments],
            "runner": runner_record,
            "runner_path": runner_record["path"],
            "runner_sha256": runner_record["sha256"],
            "component_commit": component["commit"],
            "component": component,
            "pyoomph": environment,
            "azimuthal_stability": options.azimuthal_stability,
        }
        if restart is not None:
            manifest["restart"] = restart
            manifest["restart_from"] = {"path": restart["state"]["path"],
                "sha256": restart["state"]["sha256"], "parent_manifest": restart["parent_manifest"],
                "packet_sha256": restart["packet"]["sha256"],
                "parent_case_sha256": restart["case_sha256"],
                "parent_component_commit": restart["compatibility"]["parent_commit"],
                "parent_pyoomph_commit": PYOOMPH_COMMIT,
                "parent_native_core_sha256": environment["native_core"]["sha256"]}
        # Check serialisability and custody before creating even a directory.
        json.dumps(manifest, allow_nan=False)
        out = validate_output_path(out)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    try:
        if not out.exists():
            out.mkdir(parents=True, exist_ok=False)
        create_run_manifest(out, manifest)
    except (ValueError, OSError) as exc:
        print(f"cannot create run manifest: {exc}", file=sys.stderr)
        return 1

    started = time.monotonic()
    try:
        append_progress(out, {"event": "start", "case_id": case["case_id"],
                              "azimuthal_stability": options.azimuthal_stability}, first=True)
        execution_options = {}
        if options.azimuthal_stability:
            execution_options["azimuthal_stability"] = True
        if restart is not None:
            execution_options["restart"] = restart
        summary = run_model(case, out, **execution_options)
        if not isinstance(summary, Mapping):
            raise TypeError("model execution must return a summary mapping")
        _exclusive_json(out / "summary.json", summary)
        result = exit_status(summary)
        append_progress(out, {
            "event": "final", "case_id": case["case_id"],
            "status": "completed" if result == 0 else "incomplete",
            "exit_status": result, "summary": "summary.json",
            "wall_seconds": time.monotonic() - started,
        })
        return result
    # Every native failure must leave a terminal event, including interruption.
    except (Exception, KeyboardInterrupt, SystemExit) as exc:  # noqa: BLE001
        try:
            append_progress(out, {
                "event": "failure", "case_id": case["case_id"],
                "error_type": type(exc).__name__, "error": str(exc),
                "wall_seconds": time.monotonic() - started,
            })
        except OSError as progress_error:
            print(f"cannot record failure: {progress_error}", file=sys.stderr)
        print(f"paper-model execution failed: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
