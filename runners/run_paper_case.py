"""Run one paper-model case with exclusive output and recorded provenance.

``--check`` validates the case and output custody without importing the native
solver or creating output. Execution requires committed source and the pinned
pyoomph distribution, and writes its manifest before the first progress event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
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

from problems.paper_bubble import CASE_FIELDS_USED, PYOOMPH_COMMIT, read_case

MANIFEST = "run-manifest.json"
PROGRESS = "progress.jsonl"
METHOD_FILES = (
    "runners/run_paper_case.py",
    "problems/paper_bubble.py",
    "problems/paper_mesh.py",
    "pyproject.toml",
    "uv.lock",
)


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


def run_model(case: dict, output_path: Path) -> dict:
    """Import the native execution path only after the manifest and start event."""
    from problems.paper_bubble import run_model as execute

    return execute(case, output_path)


def exit_status(summary: Mapping) -> int:
    return 0 if summary.get("status") == "completed" and summary.get("completion_reached") is True else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("case", type=Path, help="portable paper-model case JSON")
    parser.add_argument("--out", type=Path, help="new or empty output directory outside the component")
    parser.add_argument("--check", action="store_true", help="validate without native execution or output")
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
        if options.check:
            print(json.dumps({"status": "checked", "case_id": case["case_id"]}))
            return 0
        component = component_revision()
        require_clean_source(component)
        environment = pyoomph_revision()
        require_pinned_environment(environment)
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
        }
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
        append_progress(out, {"event": "start", "case_id": case["case_id"]}, first=True)
        summary = run_model(case, out)
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
