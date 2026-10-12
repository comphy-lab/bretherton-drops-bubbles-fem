"""Inspect retained pressure fields without solving or changing the physical state.

The model and checkpoint are hash-bound before native restoration. Native and
independent axisymmetric pressure moments are extracted with full state and pin
preservation checks. Completion certifies inspection only, not case acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
SOURCE = "9b77001fde2a56e93a85c70395055ff1e9443557"
PYOOMPH = "a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618"
CORE = "af7334e81f38453d6852581b6c208e93a0648c6887a6766b0fe63acd2ab51c0d"
ROOT = Path(__file__).resolve().parents[1]
METHOD_FILES = (
    "runners/run_paper_case.py",
    "problems/paper_bubble.py",
    "problems/paper_mesh.py",
    "pyproject.toml",
    "uv.lock",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def atomic(path, payload):
    """Create a private artifact exclusively; completion binds finished files."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "output must be regular")
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    atomic(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def source_contract(root, manifest):
    """Bind unchanged methods to both the retained parent and current commit."""
    require(
        manifest["component_commit"] == SOURCE == manifest["component"]["commit"],
        "unexpected bound model commit",
    )
    head = git(root, "rev-parse", "HEAD").decode().strip()
    git(root, "merge-base", "--is-ancestor", SOURCE, head)
    require(not git(root, "status", "--porcelain", "--untracked-files=no"), "tracked source dirty")
    require(
        set(METHOD_FILES) <= set(manifest["component"]["sources"]),
        "parent lacks required method files",
    )
    for relative, record in manifest["component"]["sources"].items():
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "unsafe method path",
        )
        committed = hashlib.sha256(git(root, "show", f"{SOURCE}:{relative}")).hexdigest()
        current = hashlib.sha256(git(root, "show", f"{head}:{relative}")).hexdigest()
        require(
            record["committed_matches"] is True
            and committed
            == record["sha256"]
            == record["committed_sha256"]
            == current
            == sha(root / relative),
            f"source mismatch: {relative}",
        )
    return head


def preflight(args):
    root = args.source.resolve(strict=True)
    require(root == ROOT, "diagnostic must run from its own isolated component")
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    head = source_contract(root, manifest)
    require(
        manifest["pyoomph"]["commit"] == PYOOMPH
        and manifest["pyoomph"]["native_core"]["sha256"] == CORE,
        "bound environment differs",
    )
    require(manifest["azimuthal_stability"] is True, "AZ layout required")
    require(sha(args.case) == manifest["case_sha256"], "case bytes differ")
    # The imported runner/model have just been checked against the retained manifest.
    sys.path.insert(0, str(root))
    from runners import run_paper_case as runner

    require(
        Path(runner.__file__).resolve() == root / "runners/run_paper_case.py",
        "runner imported from another component",
    )
    require(
        runner.METHOD_FILES == METHOD_FILES,
        "unexpected method-file contract",
    )
    environment = runner.pyoomph_revision()  # metadata only, no native import
    runner.require_pinned_environment(environment)
    require(environment["native_core"]["sha256"] == CORE, "installed core differs")
    require(
        environment["interpreter"]["resolved_executable"]
        == manifest["pyoomph"]["interpreter"]["resolved_executable"],
        "interpreter differs",
    )
    case = runner.read_case(args.case)
    component = runner.component_revision(root)
    runner.require_clean_source(component)
    require(component["head"] == head, "source commit changed during preflight")
    runner_path = Path(__file__).resolve()
    require(
        sha(runner_path)
        == hashlib.sha256(git(root, "show", f"{head}:{runner_path.relative_to(root)}")).hexdigest(),
        "diagnostic runner must match the current committed source",
    )
    restart = runner.validate_restart(
        args.packet,
        case,
        sha(args.case),
        component=component,
        environment=environment,
        azimuthal_stability=True,
    )
    require(Path(restart["parent_manifest"]["path"]) == manifest_path, "packet binds another run")
    require(
        Path(restart["state"]["path"]) == args.state.resolve(strict=True),
        "state differs from packet",
    )
    require(
        restart["stage"]["kind"] == "completed_polish"
        and restart["stage"]["bond"] == case["physics"]["bond"],
        "terminal target checkpoint required",
    )
    records = [
        restart["packet"],
        restart["parent_manifest"],
        restart["state"],
        *restart["meshes"],
        {"path": str(args.case.resolve()), "sha256": sha(args.case)},
    ]
    # Also protect the retained final dump where present, without claiming byte equivalence.
    final = manifest_path.parent / "checkpoints/final.dump"
    if final.exists():
        records.append({"path": str(final), "sha256": sha(final)})
    return case, restart, environment, component, records


def verify_current_source(component, runner_record):
    from runners import run_paper_case as runner

    require(runner.component_revision(ROOT) == component, "source changed since preflight")
    require(
        sha(runner_record["path"]) == runner_record["sha256"],
        "diagnostic runner changed since preflight",
    )


def parents_unchanged(records):
    try:
        return all(sha(record["path"]) == record["sha256"] for record in records)
    except OSError:
        return False


def run_manifest(args, case, restart, environment, component):
    """Describe this inspection separately from the retained stationary run."""
    from runners.run_paper_case import CASE_FIELDS_USED

    script = Path(__file__).resolve()
    runner_record = {"path": str(script), "sha256": sha(script)}
    return {
        "schema": "pyoomph-runner-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "case_id": case["case_id"],
        "case_sha256": sha(args.case),
        "case_path": str(args.case.resolve()),
        "case_file": str(args.case.resolve()),
        "case_fields_used": list(CASE_FIELDS_USED),
        "case_fields_scope": "validated case and construction; no completion gate evaluated",
        "units": "nondimensional (sigma=mu_liquid=R0=1)",
        "nondimensionalization": case["nondimensionalization"],
        "argv": [sys.executable, str(script), *sys.argv[1:]],
        "runner": runner_record,
        "runner_path": runner_record["path"],
        "runner_sha256": runner_record["sha256"],
        "component_commit": component["commit"],
        "component": component,
        "pyoomph": environment,
        "azimuthal_stability": True,
        "inspection_parent": {
            "component_commit": SOURCE,
            "manifest": restart["parent_manifest"],
            "packet": restart["packet"],
            "state": restart["state"],
            "source_compatibility": restart["compatibility"],
        },
        "diagnostic_claim": {
            "kind": "retained-state-pressure-inspection",
            "solve_performed": False,
            "numerical_acceptance": False,
            "parent_acceptance_changed": False,
            "completion": "independent inspection completion; no stationary case qualification",
        },
    }


def progress(stream, event, case_id, **payload):
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        "case_id": case_id,
        **payload,
    }
    stream.write((json.dumps(record, allow_nan=False) + "\n").encode())
    stream.flush()
    os.fsync(stream.fileno())


def extract(problem, domain, np):
    cache = problem.get_cached_mesh_data(
        domain,
        nondimensional=True,
        tesselate_tri=False,
        eigenvector=None,
        history_index=0,
        discontinuous=False,
    )
    require(cache is not None and not cache.mesh.is_mesh_distributed(), "serial mesh required")
    fields = {}
    for name in ("coordinate_x", "coordinate_y", "pressure", "velocity_x", "velocity_y"):
        require(name in cache.nodal_field_inds, f"missing native nodal field {domain}/{name}")
        value = np.asarray(cache.get_data(name)).copy()
        require(value.ndim == 1 and np.isfinite(value).all(), f"invalid field {name}")
        fields[name] = value
    fields["connectivity"] = np.asarray(cache.elem_indices).copy()
    fields["element_types"] = np.asarray(cache.elem_types).copy()
    return fields, {
        "nodal_field_indices": cache.nodal_field_inds,
        "elemental_field_indices": cache.elemental_field_inds,
        "tesselate_tri": False,
        "nondimensional": True,
    }


def actual_pins(problem):
    """Fingerprint actual Data/position pin flags, including interface and ODE data."""
    from pyoomph.meshes.mesh import ODEStorageMesh

    digest = hashlib.sha256()
    count = 0

    def add(label, flags):
        nonlocal count
        count += len(flags)
        digest.update(json.dumps([label, flags], separators=(",", ":")).encode())

    def walk(mesh, name):
        for index in range(0 if isinstance(mesh, ODEStorageMesh) else mesh.nnode()):
            node = mesh.node_pt(index)
            add([name, "node", index], [bool(node.is_pinned(i)) for i in range(node.nvalue())])
            add(
                [name, "position", index],
                [bool(node.position_is_pinned(i)) for i in range(node.ndim())],
            )
        for index in range(mesh.nelement()):
            element = mesh.element_pt(index)
            for slot in range(element.ninternal_data()):
                data = element.internal_data_pt(slot)
                add(
                    [name, "internal", index, slot],
                    [bool(data.is_pinned(i)) for i in range(data.nvalue())],
                )
        for child, interface in sorted(getattr(mesh, "_interfacemeshes", {}).items()):
            walk(interface, name + "/" + child)

    for name, mesh in sorted(problem._meshdict.items()):
        walk(mesh, name)
    return {
        "sha256": digest.hexdigest(),
        "flags_count_with_shared_node_repeats": count,
        "coverage": "all bulk/interface/ODE node, position and element internal Data pins",
    }


def integrate_lower(fields, native_area, np):
    """Exact three-point quadrature of r*p^k on straight native P1 pressure edges."""
    r, z, p = (fields[n] for n in ("coordinate_x", "coordinate_y", "pressure"))
    require(len(r) > 1 and np.min(r) >= -1e-12, "invalid radial coordinates")
    require(np.ptp(z) <= 1e-10 * max(1.0, float(np.max(np.abs(z)))), "lower port not planar")
    totals = np.zeros(5)  # area, p integral, p² integral, length, unweighted p² integral
    edges, degrees = set(), {}
    for row, kind in zip(fields["connectivity"], fields["element_types"], strict=True):
        require(int(kind) in (1, 2), f"unsupported line element type {kind}")
        n = 2 if int(kind) == 1 else 3
        indices = [int(v) for v in row[:n]]
        require(all(0 <= v < len(r) for v in indices), "invalid element connectivity")
        a, b = indices[0], indices[-1]
        edge = tuple(sorted((a, b)))
        require(a != b and edge not in edges, "duplicate or degenerate line")
        edges.add(edge)
        for node in (a, b):
            degrees[node] = degrees.get(node, 0) + 1
        length = math.hypot(float(r[b] - r[a]), float(z[b] - z[a]))
        require(length > 0, "zero length edge")
        if n == 3:
            m = indices[1]
            require(
                abs(r[m] - (r[a] + r[b]) / 2) <= 1e-10 and abs(z[m] - (z[a] + z[b]) / 2) <= 1e-10,
                "curved/nonuniform quadratic edge unsupported",
            )
            require(
                abs(p[m] - (p[a] + p[b]) / 2) <= 1e-10 * max(1.0, abs(p[a]), abs(p[b])),
                "pressure trace is not P1 on native edge",
            )
        for t, weight in (
            (0.5 - math.sqrt(3 / 5) / 2, 5 / 18),
            (0.5, 4 / 9),
            (0.5 + math.sqrt(3 / 5) / 2, 5 / 18),
        ):
            radius, pressure = r[a] * (1 - t) + r[b] * t, p[a] * (1 - t) + p[b] * t
            ds = length * weight
            totals += ds * np.array(
                [
                    2 * math.pi * radius,
                    2 * math.pi * radius * pressure,
                    2 * math.pi * radius * pressure**2,
                    1,
                    pressure**2,
                ]
            )
    endpoints = [node for node, degree in degrees.items() if degree == 1]
    require(
        len(endpoints) == 2 and all(v in (1, 2) for v in degrees.values()),
        "port is not a simple chain",
    )
    # Connectedness plus total radial span rejects disjoint chains and overlapping segments.
    reached, pending = set(), [endpoints[0]]
    adjacency = {node: [] for node in degrees}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node)
            pending.extend(adjacency[node])
    require(reached == set(degrees), "disconnected port topology")
    require(abs(totals[3] - np.ptp(r)) <= 1e-10, "overlapping/nonradial port edges")
    require(
        totals[0] > 0 and math.isclose(float(totals[0]), native_area, rel_tol=1e-9, abs_tol=1e-12),
        "independent area disagrees with native measure",
    )
    return {
        "area": float(totals[0]),
        "pressure_mean": float(totals[1] / totals[0]),
        "pressure_rms": math.sqrt(float(totals[2] / totals[0])),
        "unweighted_length_rms": math.sqrt(float(totals[4] / totals[3])),
        "unweighted_node_rms": float(np.sqrt(np.mean(p * p))),
        "max_abs_pressure": float(np.max(np.abs(p))),
        "median_abs_pressure": float(np.median(np.abs(p))),
        "method": "native edge topology; straight geometry; P1 pressure; 3-point Gauss; 2*pi*r once",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "case", "manifest", "packet", "state", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="provenance only; no native imports/output"
    )
    args = parser.parse_args()
    os.umask(0o077)
    case, restart, environment, component, records = preflight(args)
    require(not args.out.is_symlink(), "output must not be a symlink")
    output = args.out.resolve()
    require(
        not output.exists() or (output.is_dir() and not any(output.iterdir())),
        "output must be new or empty",
    )
    for parent in (args.source.resolve(), Path(restart["parent_manifest"]["path"]).parent):
        require(not output.is_relative_to(parent), "output overlaps source/runtime")
    if args.check:
        print(json.dumps({"verified": True, "native_imported": False, "inputs": records}))
        return
    actual = run_manifest(args, case, restart, environment, component)
    manifest = {
        "schema": "private-pressure-diagnostic-v1",
        "status": "prepared",
        "created_at": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "inputs": records,
        "case": case,
        "component": component,
        "environment": environment,
        "mesh_exchange": restart["mesh_exchange"],
        "solver": "superlu",
        "compiler": "system",
        "inspection_parent": actual["inspection_parent"],
        "diagnostic_claim": actual["diagnostic_claim"],
        "configuration_authority": "hash-bound problems/paper_bubble.py:run_model",
        "operations": "build/initialise/load/BC preparation/residual/sample only",
    }
    json.dumps(actual, allow_nan=False)
    json.dumps(manifest, allow_nan=False)
    if not output.exists():
        output.mkdir(parents=True, mode=0o700)
    # Guarded attempts may supply an empty directory created under a wider umask.
    directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(directory_fd, 0o700)
    finally:
        os.close(directory_fd)
    write_json(output / "run-manifest.json", actual)
    descriptor = os.open(
        output / "progress.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    events = os.fdopen(descriptor, "wb")
    try:
        progress(
            events,
            "prepared",
            case["case_id"],
            component_commit=component["commit"],
            runner=actual["runner"],
            inspection_parent_commit=SOURCE,
        )
        write_json(output / "diagnostic-manifest.json", manifest)
        import numpy as np

        from problems import paper_bubble as model

        require(
            Path(model.__file__).resolve() == ROOT / "problems/paper_bubble.py",
            "model imported from another component",
        )
        progress(events, "native-load", case["case_id"], status="starting")
        with model.build_problem(case, initial_bond=case["physics"]["bond"]) as problem:
            problem.set_output_directory(str(output / "solver"))
            problem.set_linear_solver("superlu")
            problem.set_c_compiler("system")
            problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
            for record in records:
                require(sha(record["path"]) == record["sha256"], "input changed since preflight")
            verify_current_source(component, actual["runner"])
            problem.initialise()
            problem.load_state(
                str(args.state.resolve()),
                quiet=True,
                additional_info={"exchange_msh_file": restart["mesh_exchange"]},
            )
            loaded = model._stationary_state_signature(problem, True)
            require(
                loaded["bc_cache_marker"] == restart["stage"]["bc_mode"], "literal BC cache differs"
            )
            require(
                loaded["parameters"]
                == {"Bo": case["physics"]["bond"], "Ca_in": case["physics"]["ca_in"]},
                "loaded forcing differs",
            )
            require(loaded["layout"] == restart["roundtrip"]["layout"], "loaded layout differs")
            require(
                loaded["arrays"] == restart["roundtrip"]["after"]["arrays"],
                "restored values/positions/masks differ from qualified retained checkpoint",
            )
            loaded_pins = actual_pins(problem)
            preparation = model._prepare_stationary_bc_state(problem, True)
            before = model._stationary_state_signature(problem, True)
            before_pins = actual_pins(problem)
            require(loaded_pins == before_pins, "BC preparation changed actual pins")
            residual = model._stationary_residual(problem)
            area = float(problem.observable("area", "liquid/lower"))
            require(area > 0, "nonpositive native area")
            native = {
                "area": area,
                "pressure_mean": float(problem.observable("pressure", "liquid/lower")) / area,
                "pressure_rms": math.sqrt(
                    max(float(problem.observable("pressure_squared", "liquid/lower")) / area, 0.0)
                ),
            }
            lower, lower_meta = extract(problem, "liquid/lower", np)
            bulk, bulk_meta = extract(problem, "liquid", np)
            independent = integrate_lower(lower, area, np)
            after = model._stationary_state_signature(problem, True)
            after_pins = actual_pins(problem)
            require(before == after, "sampling/residual changed full native state")
            require(before_pins == after_pins, "sampling/residual changed actual pins")
            for record in records:
                require(sha(record["path"]) == record["sha256"], "parent input changed")
            verify_current_source(component, actual["runner"])
            progress(
                events,
                "extracted",
                case["case_id"],
                full_state_preserved=True,
                actual_pins_preserved=True,
                parents_unchanged=True,
            )
            order = np.lexsort((lower["coordinate_y"], lower["coordinate_x"]))
            names = ("coordinate_x", "coordinate_y", "pressure", "velocity_y", "velocity_x")
            samples = {
                "columns": ["r", "z", "p", "axial_velocity", "radial_velocity"],
                "native_node_index": order.tolist(),
                "rows": np.column_stack([lower[n][order] for n in names]).tolist(),
            }
            data = io.BytesIO()
            np.savez_compressed(
                data,
                **{f"lower_{k}": v for k, v in lower.items()},
                **{f"bulk_{k}": v for k, v in bulk.items()},
            )
            atomic(output / "samples.npz", data.getvalue())
            write_json(output / "lower-nodes.json", samples)
            write_json(
                output / "result.json",
                {
                    "schema": "private-pressure-diagnostic-result-v1",
                    "native": native,
                    "independent": independent,
                    "stationary_residual_linf": residual,
                    "loaded_signature": loaded,
                    "bc_preparation": preparation,
                    "before": before,
                    "after": after,
                    "actual_pins": {
                        "loaded": loaded_pins,
                        "before": before_pins,
                        "after": after_pins,
                    },
                    "parents_unchanged": True,
                    "lower_metadata": lower_meta,
                    "bulk_metadata": bulk_meta,
                    "limitations": [
                        "No gradient/divergence estimate or stress observable evaluated.",
                        "Bulk coordinates, velocity and original element connectivity retained for later estimates.",
                        "No solve, transient, remesh or eigenanalysis; no scientific gate changed.",
                        "No isolated causality or traction-sign conclusion follows from these samples.",
                    ],
                },
            )
        write_json(
            output / "completion.json",
            {
                "status": "complete",
                "files": {
                    name: sha(output / name)
                    for name in (
                        "run-manifest.json",
                        "diagnostic-manifest.json",
                        "samples.npz",
                        "lower-nodes.json",
                        "result.json",
                    )
                },
                "diagnostic_claim": actual["diagnostic_claim"],
            },
        )
        progress(
            events,
            "final",
            case["case_id"],
            status="complete",
            numerical_acceptance=False,
            parent_acceptance_changed=False,
        )
    except BaseException as exc:
        progress(
            events,
            "failure",
            case["case_id"],
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "parents_unchanged": parents_unchanged(records),
            },
        )
        raise
    finally:
        events.close()


if __name__ == "__main__":
    main()
