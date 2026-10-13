"""Locate one bounded finite-gas singular equilibrium and test its dynamic null.

The admitted stationary III candidate supplies its own same-layout m0 seed.
One fixed-mesh augmented solve and two searches are permitted. A returned null
is a fold candidate, without a nondegeneracy, family or completeness certificate.
Metadata checking imports no native solver and creates no run output.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import resource
import sys
import weakref
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_iii_fold_approach as approach
from runners import run_paper_modes as modes

model, runner, diagnostic, remesh, history = (
    approach.model, approach.runner, approach.diagnostic, approach.remesh, approach.iii.history
)
sha, write_json = diagnostic.sha, diagnostic.write_json
BASE = "3e1afbd91e1e5518bc481502a76dce6fa8695857"
ENTRYPOINT = "runners/run_paper_finite_gas_fold.py"
SCHEMA = "finite-gas-same-instance-fold-candidate-v1"
N, CA = 316989, -0.00381
LIMITS = {"updates": 20, "ca_distance": 2e-5, "null": 1e-7, "mass": 1e-8,
          "root": 1e-4, "overlap": .999, "forced": 1e-12,
          "shifts": [-.001, .001], "neigen": 8, "wall_seconds": 21600}
NO_CLAIMS = {"saddle_node_nondegeneracy_certified": False, "branch_identity_verified": False,
             "unique_family_identity_certified": False, "finite_spectrum_complete": False,
             "global_dominance_certified": False, "continuum_accuracy_certified": False,
             "paper_agreement_certified": False, "parent_acceptance_changed": False}
FLAGS = ("_first_step", "_taken_already_an_unsteady_step", "_last_step_was_stationary")
EIGEN_FIELDS = modes.EIGENDATA_FIELDS
CONTROLS = ("_last_arclength_parameter", "_solve_in_arclength_conti", "_arclength_inner_product",
            "minimum_arclength_ds", "max_newton_iterations", "newton_relaxation_factor",
            "newton_solver_tolerance")
TANGENT_APIS = ("get_arclength_dof_derivative_vector", "get_arclength_dof_current_vector",
               "get_arc_length_parameter_derivative", "get_arc_length_theta_sqr",
               "_set_arc_length_parameter_derivative", "_set_arc_length_theta_sqr",
               "_update_dof_vectors_for_continuation", "reset_arc_length_parameters")


class Refusal(Exception):
    def __init__(self, label, evidence=None):
        super().__init__(label)
        self.label, self.evidence = label, evidence


def gate(condition, label, evidence=None):
    if not condition:
        raise Refusal(label, evidence)


def digest(path):
    return sha(path)


def regular(path):
    path = Path(path)
    gate(path.is_file() and not any(item.is_symlink() for item in (path, *path.parents)),
         "regular_authority_required", str(path))
    return {"path": str(path.resolve(strict=True)), "sha256": sha(path)}


def canonical_mesh_exchange(references, source_base, bound_meshes):
    """Use the pinned loader's spelling only for proven identical regular files."""
    source_base = Path(source_base)
    gate(source_base.is_dir() and not any(item.is_symlink() for item in (source_base, *source_base.parents)),
         "regular_mesh_reference_base_required", str(source_base))
    source_base = source_base.resolve(strict=True)
    gate(len(references) == len(bound_meshes) and bool(references),
         "mesh_reference_count_differs")
    exchange = {}
    for reference, bound in zip(references, bound_meshes, strict=True):
        gate(isinstance(reference, str) and bool(reference) and "\0" not in reference
             and Path(reference).suffix == ".msh" and reference not in exchange,
             "invalid_or_duplicate_mesh_reference", reference)
        source, target = regular(source_base / reference), regular(bound["path"])
        gate(target["sha256"] == bound["sha256"] == source["sha256"],
             "mesh_reference_content_differs", {"source": source, "target": target})
        # MeshTemplate._define_state_file compares this exact relative spelling
        # before deciding whether to replace its active template and remesher.
        exchange[reference] = os.path.relpath(target["path"], source_base)
    return exchange


def owned_template_signature(problem):
    """Keep template/owner identities; mesh reconstruction has no pointer identity."""
    graph = template_registration(problem)
    graph["domains"] = {name: {"template_id": row["template_id"]}
                        for name, row in graph["domains"].items()}
    return graph


def load_with_bound_meshes(problem, state, bound_meshes, *, loader=None, **options):
    """Retain the active template for memory and portable administrative loads.

    This never removes inversion registrations or assigns a BC cache marker.
    A fresh candidate load must first establish its own compatible template.
    """
    if isinstance(state, bytes):
        payload, base, stream = state, Path.cwd(), io.BytesIO(state)
    elif isinstance(state, io.BytesIO):
        payload, base, stream = state.getvalue(), Path.cwd(), io.BytesIO(state.getvalue())
    else:
        record = regular(state)
        stream = record["path"]
        payload, base = Path(stream).read_bytes(), Path(stream).parent
    references = [reference for _, _, reference in history.serialized_metadata(payload)["meshes"]]
    actual = [regular(item._get_template()._meshfile) for item in problem._meshtemplate_list]
    gate(actual == [{"path": str(Path(item["path"]).resolve(strict=True)), "sha256": item["sha256"]}
                    for item in bound_meshes], "bound_live_mesh_changed")
    exchange = canonical_mesh_exchange(references, base, bound_meshes)
    extra = dict(options.pop("additional_info", {}))
    gate(set(extra) <= {"exchange_msh_file"}, "unsupported_mesh_load_information")
    # The checked byte identities supersede an absolute-path exchange from a
    # portable packet, which would trigger replacement at this exact pin.
    extra["exchange_msh_file"] = exchange
    before = owned_template_signature(problem)
    load = problem.load_state if loader is None else loader
    result = load(stream, additional_info=extra, **options)
    after = owned_template_signature(problem)
    gate(before == after, "administrative_load_changed_template_ownership",
         signature_mismatches(before, after))
    return result


def signature_mismatches(expected, actual, path=""):
    """Name every changed signature leaf without writing native arrays."""
    missing = object()

    def compact(value):
        if value is missing:
            return {"present": False}
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str) and len(value) <= 160:
            return value
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return {"type": type(value).__name__, "sha256": hashlib.sha256(raw).hexdigest(),
                "encoded_bytes": len(raw)}

    def walk(left, right, here):
        if left == right:
            return []
        if isinstance(left, dict) and isinstance(right, dict):
            return [item for key in sorted(set(left) | set(right))
                    for item in walk(left.get(key, missing), right.get(key, missing),
                                     f"{here}.{key}" if here else str(key))]
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return [item for index in range(max(len(left), len(right)))
                    for item in walk(left[index] if index < len(left) else missing,
                                     right[index] if index < len(right) else missing,
                                     f"{here}[{index}]")]
        return [{"path": here, "expected": compact(left), "actual": compact(right)}]

    return walk(expected, actual, path)


def required(problem, names):
    for name in names:
        gate(callable(getattr(problem, name, None)), "required_api_missing", name)


def source_contract(parent, metadata_only):
    current = runner.component_revision(ROOT)
    runner.require_clean_source(parent["component"])
    gate(parent["component_commit"] == parent["component"]["commit"] == BASE,
         "stationary_source_differs")
    diagnostic.git(ROOT, "merge-base", "--is-ancestor", BASE, current["head"])
    baseline = diagnostic.git(ROOT, "ls-tree", "-r", "--name-only", BASE).decode().splitlines()
    frozen = {}
    for name in baseline:
        if name.endswith(".py") or name in ("pyproject.toml", "uv.lock", ".python-version"):
            expected = hashlib.sha256(diagnostic.git(ROOT, "show", f"{BASE}:{name}")).hexdigest()
            actual = regular(ROOT / name)
            gate(actual["sha256"] == expected, "frozen_source_changed", name)
            frozen[name] = actual
    implementation = {**frozen, ENTRYPOINT: regular(ROOT / ENTRYPOINT)}
    for name, item in parent["implementation"].items():
        gate(name in frozen and item["sha256"] == frozen[name]["sha256"],
             "producer_helper_changed", name)
        gate(regular(item["path"])["sha256"] == item["sha256"], "producer_helper_custody_changed", name)
    if not metadata_only:
        runner.require_clean_source(current)
        gate(hashlib.sha256(diagnostic.git(ROOT, "show", f"{current['commit']}:{ENTRYPOINT}")).hexdigest()
             == implementation[ENTRYPOINT]["sha256"], "driver_not_committed")
    return current, implementation


def closure(directory):
    directory = Path(directory).resolve(strict=True)
    completion_path = directory / "completion.json"
    complete = runner._read_json(completion_path)
    gate(complete["status"] == "complete" and complete["numerical_acceptance"] is True,
         "input_closure_not_accepted")
    gate(approach.iii.inventory(directory) == complete["files"], "input_closure_inventory_differs")
    return complete, [regular(completion_path), *approach.iii.completion_records(directory, complete["files"])]


def bank_contract(directory, restart):
    records, banks = [], []
    expected = restart["roundtrip"]["after"]
    for index, shift in enumerate(LIMITS["shifts"]):
        folder = directory / f"m0-shift-{index}"
        layout = runner._read_json(folder / "layout.json")
        base = runner._read_json(folder / "base-before.json")
        status = runner._read_json(folder / "status.json")
        gate(status["mode"] == 0 and status["shift"] == shift
             and status["base_restoration_verified"] is True, "bank_mode_or_restoration_differs")
        for modern, saved in (("free_values", "free_dofs"), ("positional_mask", "positional_dof_mask"),
                              ("pinned_values", "pinned_values_with_positions")):
            gate(base["arrays"][modern] == expected["arrays"][saved], "bank_state_differs", modern)
        gate(base["parameters"] == expected["parameters"] and base["time"] == expected["time"]
             and base["frame_speed"] == expected["frame_speed"] and base["volume"] == expected["volume"],
             "bank_scalar_state_differs")
        types = np.asarray(layout["types"], dtype="<i8")
        layout_sha = hashlib.sha256(json.dumps(layout["names"], separators=(",", ":")).encode()
                                    + types.tobytes()).hexdigest()
        gate(types.shape == (N,) and layout_sha == expected["layout"]["dof_layout_sha256"],
             "bank_full_layout_differs")
        gate(set(layout["globals"]) == set(modes.GLOBALS)
             and all(len(rows) == 1 for rows in layout["globals"].values()), "bank_globals_differs")
        bank = regular(folder / "eigenpairs.npz")
        with np.load(bank["path"], allow_pickle=False) as arrays:
            gate(arrays["values"].shape == (8,) and arrays["vectors"].shape == (8, N)
                 and np.isfinite(arrays["values"]).all() and np.isfinite(arrays["vectors"]).all()
                 and arrays["modes"].shape == (8,) and np.all(arrays["modes"] == 0)
                 and arrays["names"].tolist() == layout["names"]
                 and np.array_equal(arrays["types"], types)
                 and arrays["forced"].tolist() == layout["forced"],
                 "bank_array_shape_or_values_differs")
        records.extend(regular(folder / name) for name in ("layout.json", "base-before.json", "status.json", "eigenpairs.npz"))
        banks.append({"shift": shift, "index": index, "bank": bank, "layout": regular(folder / "layout.json")})
    return banks, records


def preflight(args):
    gate(args.source.resolve(strict=True) == ROOT, "isolated_source_required")
    for module, name in ((model, "problems/paper_bubble.py"), (runner, "runners/run_paper_case.py"),
                         (modes, "runners/run_paper_modes.py"), (approach, approach.ENTRYPOINT)):
        gate(Path(module.__file__).resolve() == ROOT / name, "foreign_helper_import", name)
    manifest_path = Path(regular(args.manifest)["path"])
    parent = runner._read_json(manifest_path)
    case = model.read_case(args.case)
    gate(manifest_path.name == "run-manifest.json" and approach.is_producer(parent)
         and parent["anchor_stage"] == {"kind": approach.KIND, "index": 1, "ca_in": CA}
         and case["physics"]["problem"] == "finite_gas_bubble" and case["physics"]["ca_in"] == CA
         and case["physics"]["bond"] == 1.56 and case["physics"]["vhat"] == 3.3
         and case["physics"]["domain_over_r"] == 15
         and sha(args.case) == parent["case_sha256"], "exact_stationary_III_first_point_required")
    component, implementation = source_contract(parent, args.check)
    environment = runner.pyoomph_revision()
    core = approach.coarse.environment_contract(environment, parent["pyoomph"])
    restart = runner.validate_restart(args.packet, case, sha(args.case), component=component,
                                      environment=environment, azimuthal_stability=True)
    gate(Path(restart["state"]["path"]) == args.state.resolve(strict=True)
         and restart["roundtrip"]["layout"]["ndof"] == N, "first_point_restart_differs")
    _summary, cold, records = approach.admit_stage(manifest_path.parent, parent, restart, case)
    spectrum = args.spectrum.resolve(strict=True)
    _complete, spectral_records = closure(spectrum)
    spectral = runner._read_json(spectrum / "result.json")
    spectral_manifest = runner._read_json(spectrum / "run-manifest.json")
    gate(spectral_manifest["component_commit"] == BASE
         and spectral_manifest["case_sha256"] == sha(args.case)
         and spectral_manifest["parent_restart"]["packet"] == restart["packet"]
         and spectral_manifest["pyoomph"]["commit"] == diagnostic.PYOOMPH
         and spectral_manifest["pyoomph"]["native_core"]["sha256"] == diagnostic.CORE,
         "spectral_producer_or_stationary_lineage_differs")
    requested = approach.request_contract(case, LIMITS["shifts"], 8)
    gate(spectral["request"] == requested and spectral["numerical_acceptance"] is True
         and len(spectral["mode_comparisons"]) == 1
         and spectral["mode_comparisons"][0]["matched_returned_envelope"] is True,
         "first_point_m0_envelope_required")
    banks, witnesses = bank_contract(spectrum, restart)
    records = approach.coarse.unique_records([*records, *spectral_records, *witnesses, core,
                                               regular(args.case), regular(manifest_path)])
    contract = {"schema": SCHEMA + "-contract", "base_ndof": N, "initial_ca": CA,
                "case": regular(args.case), "parent_manifest": regular(manifest_path),
                "restart": restart["packet"], "state": restart["state"], "meshes": restart["meshes"],
                "stationary_closure": regular(manifest_path.parent / "completion.json"),
                "spectral_closure": regular(spectrum / "completion.json"), "banks": banks,
                "limits": LIMITS, "steady_residual_limit": case["tolerances"]["newton"],
                "augmented_solve_calls": 1, "physical_remeshes": 0, "same_instance_searches": 2,
                "seed_bank": 0, "seed_index": 0, "candidate_scope": "singular equilibrium and dynamic null only",
                **NO_CLAIMS}
    seal = {"schema": SCHEMA + "-execution-seal", "source_commit": component["commit"],
            "source_files": {name: item["sha256"] for name, item in implementation.items()},
            "pyoomph_commit": environment["commit"], "native_core_sha256": environment["native_core"]["sha256"],
            "interpreter": environment["interpreter"]["resolved_executable"], "inputs": list(records), "contract": contract}
    pending = []
    if component["clean"] is not True:
        pending.append("clean committed source pending")
    input_seal = regular(args.seal) if getattr(args, "seal", None) is not None else None
    if input_seal is None:
        pending.append("external exact source/input execution seal pending")
    else:
        gate(runner._read_json(Path(input_seal["path"])) == seal, "execution_seal_differs")
        records.append(input_seal)
    if not args.check:
        gate(not pending, "native_execution_not_admitted", pending)
    output = runner.validate_output_path(args.out, ROOT)
    gate(not any(path.is_symlink() for path in (args.out, *args.out.parents)), "output_symlink")
    gate(not output.is_relative_to(manifest_path.parent) and not output.is_relative_to(spectrum)
         and not manifest_path.parent.is_relative_to(output) and not spectrum.is_relative_to(output),
         "output_overlaps_input_authority")
    return {"case": case, "parent": parent, "restart": restart, "cold": cold,
            "component": component, "implementation": implementation, "environment": environment,
            "records": records, "contract": contract, "required_seal": seal,
            "input_seal": input_seal, "source_seals_pending": pending, "output": output,
            "native_execution_authorized": not args.check and not pending}


def stable_inputs(prepared):
    remesh.verify_inputs(prepared["records"])
    remesh.verify_source(prepared["component"], prepared["implementation"])


def measured_case(original, value):
    gate(np.isfinite(value) and abs(value - CA) <= LIMITS["ca_distance"], "ca_locality_exceeded")
    current = copy.deepcopy(original)
    current["case_id"] = original["case_id"] + "_fold_candidate"
    current["physics"]["ca_in"] = float(value)
    current = model.validate_case(current)
    gate(set(remesh.differing_keys(original, current)) <= {"case_id", "physics.ca_in"},
         "measured_case_changed_physics")
    return current


def manifest(case_path, case, prepared, arguments):
    own = prepared["implementation"][ENTRYPOINT]
    return {"schema": "pyoomph-runner-manifest-v1", "created_at": datetime.now(UTC).isoformat(),
            "case_path": str(case_path.resolve()), "case_file": str(case_path.resolve()),
            "case_sha256": sha(case_path), "case_id": case["case_id"], "case_fields_used": model.CASE_FIELDS_USED,
            "nondimensionalization": case["nondimensionalization"], "units": "sigma=mu_liquid=R0=1",
            "runner": own, "runner_path": own["path"], "runner_sha256": own["sha256"],
            "argv": [sys.executable, own["path"], *arguments], "component": prepared["component"],
            "component_commit": prepared["component"]["commit"], "pyoomph": prepared["environment"],
            "implementation": prepared["implementation"], "azimuthal_stability": True,
            "parent_restart": prepared["restart"], "fold_contract": prepared["contract"],
            "qualification_schema": SCHEMA, "input_seal": prepared["input_seal"], **NO_CLAIMS}




BEFORE_STEP = "actions_before_newton_step"
OBSERVER = "_stone_newton_observer"


def forced_payload(problem):
    """Use the admitted full-state policy and restore both actual flags on error."""
    eigen_setting = problem.eigen_data_in_states
    continuation_setting = problem.continuation_data_in_states
    try:
        problem.eigen_data_in_states = True
        problem.continuation_data_in_states = True
        return problem._snapshot_state()
    finally:
        problem.eigen_data_in_states = eigen_setting
        problem.continuation_data_in_states = continuation_setting


def array_record(value):
    value = np.asarray(value)
    gate(value.dtype.kind != 'O' and np.isfinite(value).all(), 'nonfinite_or_opaque_state_array')
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())  # noqa: C408 - preserve reviewed restoration expressions


class RemeshGuard:
    """Monotonic physical counters and individually labelled administrative events."""
    def __init__(self, problem):
        self.problem = problem
        self.physical_callbacks = 0
        self.requests = 0
        self.before_remesh_count = 0
        self.flag_events = []
        self.checks = []
        self.events = []
        self.pending_remesh_calls = []
        self.context = None
        self.physical_fixed = None

    @contextlib.contextmanager
    def administrative(self, label):
        gate(self.context is None and isinstance(label, str) and bool(label), 'invalid_restore_context')
        self.context = label
        event = dict(label=label, callbacks=[], verified=False)  # noqa: C408 - preserve reviewed restoration expressions
        self.events.append(event)
        try:
            yield event
        finally:
            self.context = None
            self.arm_mesh_requests()

    def callback(self, name):
        if self.context is None:
            self.physical_callbacks += 1
            gate(False, 'physical_mesh_callback', name)
        self.events[-1]['callbacks'].append(name)

    def reject(self, name):
        def rejected(*args, **kwargs):
            self.requests += 1
            if name == 'actions_before_remeshing': self.before_remesh_count += 1
            gate(False, 'physical_remesh_or_adapt_request', name)
        return rejected

    def guarded_pending_remesh(self):
        """Record the unconditional stationary-solve epilogue without clearing flags."""
        p = self.problem
        flag = p._remesh_requested_inside_solve
        domains = len(p._domains_to_remesh)
        benign = flag is False and domains == 0
        row = dict(label=self.context, call=len(self.pending_remesh_calls)+1,  # noqa: C408 - preserve reviewed restoration expressions
                   requested=flag if isinstance(flag, bool) else repr(flag),
                   pending_domains=domains, no_op=benign)
        self.pending_remesh_calls.append(row)
        if benign:
            return False
        self.requests += 1
        gate(False, 'physical_remesh_or_adapt_request',
             dict(api='_perform_pending_remesh', **row))

    def install(self):
        p = self.problem
        # Administrative load calls AFTER notifications, never these request paths.
        for name in ('force_remesh', 'remesh_if_necessary', 'remesh_handler_during_continuation',
                     '_adapt', 'adapt', '_adapt_with_interfacial_errors', 'custom_adapt', 'actions_before_remeshing', 'actions_before_adapt'):
            gate(callable(getattr(p, name, None)), 'missing_physical_request_api', name)
            setattr(p, name, self.reject(name))
        gate(callable(getattr(p, '_perform_pending_remesh', None)),
             'missing_physical_request_api', '_perform_pending_remesh')
        p._perform_pending_remesh = self.guarded_pending_remesh
        for name in ('actions_after_remeshing', 'actions_after_adapt'):
            original = getattr(p, name)
            def notified(*args, _name=name, _original=original, **kwargs):
                self.callback(_name)
                return _original(*args, **kwargs)
            setattr(p, name, notified)
        p.do_call_remeshing_when_necessary = False
        self.arm_mesh_requests()

    def check_physical_flags(self, label):
        p = self.problem
        requested = p._remesh_requested_inside_solve is not False
        domains = len(p._domains_to_remesh)
        row = dict(label=label, automatic_remesh=bool(p.do_call_remeshing_when_necessary),  # noqa: C408 - preserve reviewed restoration expressions
                   before_remesh_count=self.before_remesh_count, raising_request_count=self.requests,
                   pending_request=requested, pending_domains=domains,
                   physical_after_callback_count=self.physical_callbacks,
                   administrative_after_callbacks=sum(len(e['callbacks']) for e in self.events),
                   administrative_remesh_callbacks=sum(e['callbacks'].count('actions_after_remeshing') for e in self.events))
        self.checks.append(row)
        if requested or domains:
            self.requests += 1
            self.flag_events.append(row)
            gate(False, 'physical_remesh_pending_flag', row)
        gate(not p.do_call_remeshing_when_necessary and self.before_remesh_count == 0
             and self.requests == 0 and self.physical_callbacks == 0,
             'physical_remesh_guard_failed', row)

    def arm_mesh_requests(self):
        # Native reconstruction may replace mesh objects; re-arm the new instances.
        for path, mesh in walk_meshes(self.problem):
            if getattr(mesh, '_stone_fold_guard', None) is self: continue
            for name in ('adapt_by_elemental_errors', '_adapt_select', '_adapt_execute', '_adapt_finalise'):
                if callable(getattr(mesh, name, None)):
                    try:
                        setattr(mesh, name, self.reject(path+'/'+name))
                    except Exception as exc:  # noqa: BLE001 - retain partial evidence and restore on every failure
                        gate(False, 'mesh_request_guard_binding_failed', dict(mesh=path, api=name, error=str(exc)))  # noqa: C408 - preserve reviewed restoration expressions
            mesh._stone_fold_guard = self
        for template in getattr(self.problem, '_meshtemplate_list', []):
            remesher = getattr(template, 'remesher', None)
            if remesher is not None and getattr(remesher, '_stone_fold_guard', None) is not self:
                remesher.remesh = self.reject(type(remesher).__name__+'.remesh')
                remesher._stone_fold_guard = self


def copy_eigendata(problem):
    result = {}
    for name in EIGEN_FIELDS:
        value = getattr(problem, name)
        gate(value is None or isinstance(value, np.ndarray), 'unsupported_eigen_cache', name)
        if value is not None: array_record(value)
        result[name] = None if value is None else value.copy()
    return result


def eigen_signature(data):
    gate(set(data) == set(EIGEN_FIELDS), 'incomplete_eigen_cache')
    return {name: None if value is None else array_record(value) for name, value in data.items()}


def walk_meshes(problem):
    def walk(name, mesh):
        yield name, mesh
        for child, interface in sorted(getattr(mesh, '_interfacemeshes', {}).items()):
            yield from walk(name+'/'+child, interface)
    for name, mesh in sorted(problem._meshdict.items()):
        yield from walk(name, mesh)


def actual_mesh_signature(problem, ode_type, *, fold_base_ndof=None):
    """Full precision node order, history, connectivity, actual pins/equation addresses.

    No C++ pointer is used as an identity in the recorded signature.
    """
    expected_ndof = problem.ndof()
    if fold_base_ndof is not None:
        # Preserve the old complete-coverage predicate on every unaugmented
        # call. During a fold update the extra vector/parameter DOFs belong to
        # the handler, not mesh Data, and are checked as live augmented values.
        gate(problem.get_bifurcation_tracking_mode() == 'fold'
             and problem.ndof() == 2*fold_base_ndof+1,
             'unsupported_augmented_mesh_address_coverage')
        handler = problem.assembly_handler_pt()
        gate(callable(getattr(handler, 'get_base_ndof', None))
             and handler.get_base_ndof() == fold_base_ndof,
             'missing_fold_base_address_coverage_hook')
        expected_ndof = fold_base_ndof
    meshes = {}
    covered = set()
    for name, mesh in walk_meshes(problem):
        nodes, elements = [], []
        for index in range(0 if isinstance(mesh, ode_type) else mesh.nnode()):
            node = mesh.node_pt(index)
            def data_record(data):
                equations = [int(data.eqn_number(i)) for i in range(data.nvalue())]
                covered.update(e for e in equations if e >= 0)
                history = [[data.value_at_t(t, i) for i in range(data.nvalue())] for t in range(data.ntstorage())]
                return dict(equations=equations, pins=[bool(data.is_pinned(i)) for i in range(data.nvalue())],  # noqa: C408 - preserve reviewed restoration expressions
                            history=array_record(np.asarray(history, dtype=np.float64)))
            nodes.append(dict(values=data_record(node), positions=data_record(node.variable_position_pt()),  # noqa: C408 - preserve reviewed restoration expressions
                              position_pins=[bool(node.position_is_pinned(i)) for i in range(node.ndim())],
                              lagrangian=array_record([node.x_lagr(i) for i in range(node.ndim())])))
        for index in range(mesh.nelement()):
            element = mesh.element_pt(index)
            internal = []
            for slot in range(element.ninternal_data()):
                data = element.internal_data_pt(slot)
                equations = [int(data.eqn_number(i)) for i in range(data.nvalue())]
                covered.update(e for e in equations if e >= 0)
                internal.append(dict(equations=equations,  # noqa: C408 - preserve reviewed restoration expressions
                    pins=[bool(data.is_pinned(i)) for i in range(data.nvalue())],
                    history=array_record(np.asarray([[data.value_at_t(t, i) for i in range(data.nvalue())]
                        for t in range(data.ntstorage())], dtype=np.float64))))
            elements.append(internal)
        record = dict(nodes=nodes, elements=elements)  # noqa: C408 - preserve reviewed restoration expressions
        if not isinstance(mesh, ode_type):
            conn, stride = mesh.get_element_node_indices()
            record['connectivity'] = dict(indices=array_record(conn), stride=int(stride))  # noqa: C408 - preserve reviewed restoration expressions
            record['nodal_state'] = [array_record(v) for v in mesh.save_nodal_state()]
            # Interface element state is covered by Data history and the full stream.
            if '/' not in name:
                record['elemental_state'] = [array_record(v) for v in mesh.save_elemental_state()]
        meshes[name] = record
    gate(covered == set(range(expected_ndof)), 'incomplete_equation_address_map',
         dict(covered=len(covered), ndof=expected_ndof))  # noqa: C408 - preserve reviewed restoration expressions
    return meshes


def forcing_signature(problem):
    result = []
    for manip in problem.get_eigen_solver().matrix_manipulators:
        gate(hasattr(manip, 'doflist') and callable(getattr(manip, '_resolve_zeromap', None)),
             'unsupported_eigen_matrix_manipulator', type(manip).__name__)
        result.append(dict(kind=type(manip).__module__+'.'+type(manip).__qualname__,  # noqa: C408 - preserve reviewed restoration expressions
                           selectors=sorted(manip.doflist, key=str), rows=sorted(map(int, manip._resolve_zeromap()))))
    return result


def solver_owner_signature(problem):
    """Validate the pinned weakref owner; this witness is never assigned as state."""
    solver=problem.get_eigen_solver()
    attributes=vars(solver)
    gate('problem' not in attributes, 'unsupported_eigensolver_owner_shadow', 'problem')
    reference=attributes.get('_problem_wr')
    typename=type(reference).__module__+'.'+type(reference).__qualname__
    gate(type(reference) is weakref.ReferenceType,
         'unsupported_eigensolver_owner_reference', dict(attribute='_problem_wr',type=typename))  # noqa: C408 - preserve reviewed restoration expressions
    gate(reference.__callback__ is None, 'unsupported_eigensolver_owner_callback')
    target=reference()
    gate(target is not None, 'dead_eigensolver_problem_owner')
    gate(target is problem, 'foreign_eigensolver_problem_owner')
    gate(solver.problem is target, 'eigensolver_problem_getter_mismatch')
    return dict(reference_type=typename,alive=True,target_is_same_problem=True,  # noqa: C408 - preserve reviewed restoration expressions
                getter_agrees=True,problem_python_id=id(problem),solver_python_id=id(solver))


def solver_signature(problem):
    """Capture all pinned Spectra scalar state, including contribution/cache markers."""
    fields = {}
    solver_owner_signature(problem)
    for name, value in vars(problem.get_eigen_solver()).items():
        if name in ('_problem_wr', 'matrix_manipulators'): continue
        if isinstance(value, np.generic): value = value.item()
        gate(value is None or isinstance(value, (str, bool, int, float)),
             'unsupported_eigensolver_cache_attribute', name)
        if isinstance(value, float): gate(np.isfinite(value), 'nonfinite_eigensolver_attribute', name)
        fields[name] = value
    return fields


def restore_solver_scalars(problem, fields, expected_owner):
    """Keep ownership intact while restoring only validated scalar/cache fields."""
    gate(solver_owner_signature(problem)==expected_owner, 'eigensolver_owner_identity_changed')
    gate(not {'_problem_wr','problem','matrix_manipulators'} & set(fields),
         'owner_in_restorable_solver_state')
    for name,value in fields.items():
        gate(value is None or isinstance(value,(str,bool,int,float)),
             'unsupported_eigensolver_cache_attribute',name)
        if isinstance(value,float):gate(np.isfinite(value),'nonfinite_eigensolver_attribute',name)
    solver=problem.get_eigen_solver()
    for name,value in fields.items():setattr(solver,name,value)
    gate(solver_owner_signature(problem)==expected_owner, 'eigensolver_owner_identity_changed')


def array_identity(v):
    v = np.asarray(v)
    return dict(shape=list(v.shape), dtype=str(v.dtype),  # noqa: C408 - preserve reviewed restoration expressions
                sha256=hashlib.sha256(v.tobytes()).hexdigest())


def capture_continuation(p):
    """Capture exported vectors and controls without solving or normalising them."""
    required(p, TANGENT_APIS)
    n = p.ndof()
    derivative = np.asarray(p.get_arclength_dof_derivative_vector()).copy()
    current = np.asarray(p.get_arclength_dof_current_vector()).copy()
    for name, value in (('Dof_derivative', derivative), ('Dof_current', current)):
        gate(value.ndim == 1 and len(value) in (0, n)
             and value.dtype.kind == 'f' and np.isfinite(value).all(),
             'invalid_continuation_vector', dict(field=name, shape=list(value.shape), ndof=n))  # noqa: C408 - preserve reviewed restoration expressions
    dp = float(p.get_arc_length_parameter_derivative())
    theta = float(p.get_arc_length_theta_sqr())
    gate(np.isfinite(dp) and np.isfinite(theta) and theta > 0,
         'invalid_continuation_scalar', dict(parameter_derivative=str(dp), theta_sqr=str(theta)))  # noqa: C408 - preserve reviewed restoration expressions
    controls = {}
    for name in CONTROLS:
        gate(hasattr(p, name), 'missing_required_continuation_control', name)
        value = getattr(p, name)
        gate(value is None or isinstance(value, (str, int, float, bool)),
             'opaque_continuation_control', name)
        if isinstance(value, float): gate(np.isfinite(value), 'nonfinite_continuation_control', name)
        controls[name] = value
    gate(controls['_last_arclength_parameter'] in (None, 'Ca_in')
         and controls['_solve_in_arclength_conti'] is None
         and controls['_arclength_inner_product'] in (None, 'ndof'),
         'unsupported_continuation_controls', controls)
    gate(hasattr(p, '_parked_continuation_data') and p._parked_continuation_data is None,
         'unsupported_parked_continuation_data')
    return dict(derivative=derivative, current=current, parameter_derivative=dp,  # noqa: C408 - preserve reviewed restoration expressions
                theta_sqr=theta, controls=controls, parked_continuation_data=None)


def continuation_signature(state):
    return dict(dof_derivative=array_identity(state['derivative']),  # noqa: C408 - preserve reviewed restoration expressions
                dof_current=array_identity(state['current']),
                parameter_derivative=state['parameter_derivative'], theta_sqr=state['theta_sqr'],
                controls=state['controls'], parked_continuation_data=None)


def restore_continuation(p, saved):
    """Restore what is exported; fail when an empty current buffer cannot be restored.

    reset_arc_length_parameters() leaves Dof_current allocated. No empty setter is
    exported at a186. A fresh Problem can start empty but changes the admitted
    same-instance/owner identity; it is not an administrative restore here.
    """
    required(p, TANGENT_APIS)
    derivative, current = saved['derivative'], saved['current']
    live_current = np.asarray(p.get_arclength_dof_current_vector())
    live_derivative = np.asarray(p.get_arclength_dof_derivative_vector())
    if np.array_equal(live_current, current) and np.array_equal(live_derivative, derivative):
        pass  # No setter is needed to preserve already exact empty/nonempty buffers.
    elif len(derivative) == 0:
        if len(current):
            p._update_dof_vectors_for_continuation(np.zeros_like(current), current)
        p.reset_arc_length_parameters()
    else:
        gate(len(current) == len(derivative), 'continuation_current_restore_unsupported',
             dict(field='Dof_current', saved_length=len(current), derivative_length=len(derivative),  # noqa: C408 - preserve reviewed restoration expressions
                  reason='pinned setter requires equally sized ndof-long vectors'))
        p._update_dof_vectors_for_continuation(derivative, current)
    p._set_arc_length_parameter_derivative(saved['parameter_derivative'])
    p._set_arc_length_theta_sqr(saved['theta_sqr'])
    for name, value in saved['controls'].items(): setattr(p, name, value)
    p._parked_continuation_data = None
    gate(len(current) or len(live_current) == 0, 'continuation_current_empty_restore_unsupported',
         dict(field='Dof_current', saved_length=0, live_length=len(live_current),  # noqa: C408 - preserve reviewed restoration expressions
              reason='reset leaves current buffer; native setter rejects empty vectors',
              fresh_problem='changes the admitted Problem/solver owner identity; not used'))
    gate(continuation_signature(capture_continuation(p)) == continuation_signature(saved),
         'continuation_tangent_restoration_failed')


@contextlib.contextmanager
def install_update_monitor(adapter, monitor):
    """Attach a real GenericProblemHooks subclass; preserve every existing callback.

    A Python GateError propagates through NB_OVERRIDE and is not either native
    exception caught by arc_length_step_solve_helper. _request_newton_abort is
    deliberately unused: it becomes NewtonSolverError and asks arclength to retry.
    """
    p = adapter.problem
    required(p, ('actions_before_newton_step', '_newton_abort_requested',
                 'arclength_continuation', '_retune_arclength_theta'))
    gate(p._newton_abort_requested() is False, 'pending_native_newton_abort')
    gate(isinstance(p._hooks, list), 'missing_required_newton_hook_list')
    base = adapter.hook_type
    for name in ('actions_before_newton_solve', 'actions_after_newton_step', 'actions_after_parameter_increase'):
        gate(callable(getattr(base, name, None)), 'missing_required_newton_hook', name)
    original_hooks = list(p._hooks)
    bridge = adapter.before_step_bridge
    bridge.verify_problem(p)
    class Hook(base):
        def actions_before_newton_solve(self): monitor.before_solve()
        def actions_after_newton_step(self): monitor.after_update()
        def actions_after_parameter_increase(self, name): monitor.after_parameter_increase(name)
    hook = Hook()
    installed = False
    observer_attached = False
    try:
        p += hook
        gate(p._hooks == original_hooks+[hook], 'newton_hook_registration_failed')
        # Check before other user hooks can change the updated configuration.
        p._hooks[:] = [hook]+original_hooks
        bridge.attach(p, monitor)
        observer_attached = True
        installed = True
        yield monitor
        monitor.finalise()
    finally:
        unchanged = p._hooks == ([hook]+original_hooks if installed else original_hooks+[hook])
        p._hooks[:] = original_hooks
        if observer_attached: bridge.detach(p, monitor)
        gate(unchanged, 'newton_hook_inventory_changed')
        gate(p._newton_abort_requested() is False, 'pending_native_newton_abort')


class BeforeStepBridge:
    """Install once before construction; restore only after native release."""
    def __init__(self, problem_type, core_type):
        owners = [base for base in problem_type.__mro__ if BEFORE_STEP in vars(base)]
        gate(BEFORE_STEP not in vars(problem_type) and owners == [core_type],
             'before_step_already_overridden', [base.__module__+'.'+base.__qualname__ for base in owners])
        base_before = vars(core_type).get(BEFORE_STEP)
        gate(core_type.__module__ == 'pyoomph._pyoomph_core'
             and callable(base_before) and getattr(problem_type, BEFORE_STEP) is base_before,
             'before_step_base_binding_changed')
        self.problem_type, self.base_before = problem_type, base_before
        self.restored = False

        def _stone_actions_before_newton_step(problem):
            self.verify_problem(problem)
            observer = getattr(problem, OBSERVER, None)
            gate(observer is not None, 'newton_iteration_outside_update_session')
            observer.before_iteration()
            return base_before(problem)

        self.function = _stone_actions_before_newton_step
        setattr(problem_type, BEFORE_STEP, self.function)
        gate(vars(problem_type).get(BEFORE_STEP) is self.function,
             'constructor_class_bridge_install_failed')
        self.receipt = dict(schema='stone-before-step-original-class-bridge-v1',  # noqa: C408 - preserve reviewed restoration expressions
            class_bridge_installed_before_construction=True,
            scientific_type_unchanged=True,
            problem_type=problem_type.__module__+'.'+problem_type.__qualname__,
            mro=[base.__module__+'.'+base.__qualname__ for base in problem_type.__mro__],
            function_name=self.function.__name__, function_python_id=id(self.function),
            base_binding_owner=core_type.__module__+'.'+core_type.__qualname__,
            after_step_route='unchanged GenericProblemHooks first in existing hook list')

    def verify_problem(self, problem):
        gate(not self.restored and type(problem) is self.problem_type,
             'constructor_class_bridge_type_changed')
        gate(vars(self.problem_type).get(BEFORE_STEP) is self.function,
             'constructor_class_bridge_changed')
        gate(BEFORE_STEP not in vars(problem), 'instance_shadow_present')

    def attach(self, problem, observer):
        self.verify_problem(problem)
        gate(OBSERVER not in vars(problem), 'newton_observer_already_attached')
        setattr(problem, OBSERVER, observer)

    def detach(self, problem, observer):
        unchanged = vars(problem).get(OBSERVER) is observer
        if OBSERVER in vars(problem):
            delattr(problem, OBSERVER)
        self.verify_problem(problem)
        gate(unchanged, 'newton_observer_changed')

    def restore_after_release(self, *, native_session_released):
        gate(native_session_released is True, 'class_bridge_restore_before_native_release')
        gate(not self.restored and vars(self.problem_type).get(BEFORE_STEP) is self.function,
             'constructor_class_bridge_changed')
        delattr(self.problem_type, BEFORE_STEP)
        gate(BEFORE_STEP not in vars(self.problem_type)
             and getattr(self.problem_type, BEFORE_STEP) is self.base_before,
             'class_bridge_restoration_failed')
        self.restored = True
        return dict(schema='stone-before-step-class-restoration-v1',  # noqa: C408 - preserve reviewed restoration expressions
            native_session_released=True, original_inherited_binding_restored=True,
            scientific_type_unchanged=True)


def fixed_identity(meshes, types, names, positional):
    stable = {}
    for name, mesh in meshes.items():
        stable[name] = {"connectivity": mesh.get("connectivity"),
                        "nodes": [{"values": {key: node["values"][key] for key in ("equations", "pins")},
                                   "positions": {key: node["positions"][key] for key in ("equations", "pins")},
                                   "position_pins": node["position_pins"], "lagrangian": node["lagrangian"]}
                                  for node in mesh["nodes"]],
                        "elements": [[{key: data[key] for key in ("equations", "pins")} for data in element]
                                     for element in mesh["elements"]]}
    return {"meshes": stable, "types": array_record(types), "names": list(names),
            "positional": array_record(positional)}


def unit(vector):
    vector = np.asarray(vector)
    gate(vector.shape == (N,) and np.isfinite(vector).all(), "invalid_base_vector")
    length = float(np.linalg.norm(vector))
    gate(length > 0 and np.isfinite(length), "zero_or_nonfinite_vector_norm")
    return vector / length


def real_seed(vector):
    vector = np.asarray(vector, dtype=complex)
    gate(vector.shape == (N,) and np.isfinite(vector).all(), "invalid_seed_vector")
    index = int(np.argmax(np.abs(vector)))
    gate(abs(vector[index]) > 0, "zero_seed")
    rotation = np.conjugate(vector[index]) / abs(vector[index])
    return unit((vector * rotation).real), {"rotation_real": float(rotation.real),
                                          "rotation_imag": float(rotation.imag), "pivot_index": index}


class UpdateMonitor:
    def __init__(self, adapter):
        self.a = adapter
        self.starts = self.updates = self.before_iterations = 0

    def before_solve(self):
        self.starts += 1
        gate(self.starts == 1, "automatic_augmented_retry_refused")
        self.a.check_newton("before_solve")

    def before_iteration(self):
        self.before_iterations += 1
        gate(self.before_iterations == self.updates + 1 and self.updates < LIMITS["updates"],
             "augmented_update_budget_or_sequence_refused")
        self.a.check_newton("before_iteration")

    def after_update(self):
        self.updates += 1
        self.a.work["completed_augmented_updates"] = self.updates
        gate(self.updates <= LIMITS["updates"] and self.before_iterations >= self.updates,
             "augmented_callback_or_update_budget_exceeded")
        self.a.check_newton("after_update")

    def after_parameter_increase(self, name):
        gate(False, "unrequested_arclength_callback", name)

    def finalise(self):
        gate(self.starts == 1 and self.before_iterations >= self.updates, "augmented_callback_coverage_missing")
        self.a.check_newton("after_solve")


class Adapter:
    def __init__(self, problem, prepared, output, core, ode_type, hook_type, bridge, deadline):
        self.problem, self.prepared, self.output = problem, prepared, output
        self.core, self.ode_type, self.hook_type, self.before_step_bridge = core, ode_type, hook_type, bridge
        self.deadline = deadline
        self.work = {"augmented_solve_calls": 0, "completed_augmented_updates": 0,
                     "eigenproblem_calls": 0, "physical_remeshes": 0}
        self.receipts = []
        self.mesh_guard = RemeshGuard(problem)
        self.mesh_guard.install()
        self.fixed = None
        self.serial = 0

    def check(self):
        gate(datetime.now(UTC) < self.deadline, "phase_wall_deadline_reached")
        stable_inputs(self.prepared)
        self.mesh_guard.check_physical_flags("phase")

    def signature(self):
        p = self.problem
        gate(p.get_bifurcation_tracking_mode() == "" and p.ndof() == N, "base_snapshot_during_augmentation")
        native = modes.take_state(p, mesh_authority=self.prepared["restart"]["meshes"])
        raw = forced_payload(p)
        continuation = capture_continuation(p)
        scalar_fields = {name: getattr(p, name) for name in
                         ("_bifurcation_tracking_parameter_name", "_bifurcation_eigenvector_scaling",
                          "do_call_remeshing_when_necessary", "inversion_remesh_max_retries",
                          "_inversion_remesh_threshold")}
        types, names = p.get_dof_description()
        meshes = actual_mesh_signature(p, self.ode_type)
        signature = {"mode_state": native["signature"], "full_native_stream": hashlib.sha256(
            history.normalized_bytes(raw, Path.cwd())).hexdigest(),
            "continuation": continuation_signature(continuation), "scalar_fields": scalar_fields,
            "continuation_data_in_states": p.continuation_data_in_states,
            "solver": solver_signature(p), "solver_owner": solver_owner_signature(p),
            "manipulators": forcing_signature(p), "meshes": meshes,
            "selected_residual": p._get_solved_residual(),
            "inversion_templates": sorted((type(item).__name__, id(item)) for item in p._domains_remesh_on_inversion),
            "use_static_condensation": bool(p.use_static_condensation)}
        return {"signature": signature, "native": native, "payload": raw,
                "continuation": continuation, "manipulators": list(p.get_eigen_solver().matrix_manipulators),
                "case": copy.deepcopy(p.case),
                "fixed": fixed_identity(meshes, np.asarray(types, dtype=np.int64), names,
                                        p.get_current_dofs()[1])}

    def snapshot(self, label):
        stable_inputs(self.prepared)
        self.mesh_guard.check_physical_flags(label)
        saved = self.signature()
        directory = self.output / "snapshots"
        directory.mkdir(exist_ok=True)
        self.serial += 1
        stem = f"{self.serial:03d}-{label}"
        diagnostic.atomic(directory / (stem + ".dump"), saved["payload"])
        write_json(directory / (stem + ".json"), saved["signature"])
        return saved

    def restore(self, saved, label):
        p, expected = self.problem, saved["signature"]
        guard = self.mesh_guard
        counter_names = ("requests", "physical_callbacks", "before_remesh_count")
        counters_before = {name: getattr(guard, name) for name in counter_names}
        pending_before = {"_domains_to_remesh": sorted(map(str, p._domains_to_remesh)),
                          "_remesh_requested_inside_solve": p._remesh_requested_inside_solve
                          if isinstance(p._remesh_requested_inside_solve, bool)
                          else repr(p._remesh_requested_inside_solve)}
        if p.get_bifurcation_tracking_mode():
            p.deactivate_bifurcation_tracking()
        gate(solver_owner_signature(p) == expected["solver_owner"], "solver_owner_changed")
        with guard.administrative(label) as event:
            # These are request metadata, not fields or physical mesh history.
            # Retain the refused phase events in the monotonic guard counters.
            p._domains_to_remesh.clear()
            p._remesh_requested_inside_solve = False
            p._azimuthal_mode_param_m.value = saved["native"]["signature"]["mode"]
            p.actions_before_stationary_solve(force_reassign_eqs=True)
            p.eigen_data_in_states = p.continuation_data_in_states = True
            bound_meshes = getattr(self, "bound_meshes", None)
            if bound_meshes is None:
                p._restore_state(saved["payload"])
            else:
                load_with_bound_meshes(p, saved["payload"], bound_meshes, quiet=True)
            for name, value in saved["native"]["eigendata"].items():
                setattr(p, name, None if value is None else value.copy())
            p._azimuthal_mode_param_m.value = saved["native"]["signature"]["mode"]
            for name, value in saved["native"]["signature"]["flags"].items():
                setattr(p, name, value)
            for name, value in saved["native"]["signature"]["parameters"].items():
                p.get_global_parameter(name).value = value
            for index, steady in enumerate(saved["native"]["signature"]["steady"]):
                stepper = p.time_stepper_pt(index)
                stepper.make_steady() if steady else stepper.undo_make_steady()
            p.eigen_data_in_states = saved["native"]["signature"]["eigen_data_in_states"]
            p.continuation_data_in_states = expected["continuation_data_in_states"]
            for name, value in expected["scalar_fields"].items():
                setattr(p, name, value)
            p.get_eigen_solver().matrix_manipulators[:] = saved["manipulators"]
            restore_solver_scalars(p, expected["solver"], expected["solver_owner"])
            restore_continuation(p, saved["continuation"])
            p._set_solved_residual(expected["selected_residual"], True, True)
            p.case = copy.deepcopy(saved["case"])
            p.invalidate_cached_mesh_data()
            actual = self.signature()
            differences = signature_mismatches(expected, actual["signature"])
            fixed_differences = signature_mismatches(saved["fixed"], actual["fixed"], "fixed")
            if (differences or fixed_differences) and hasattr(self, "output"):
                gate(Path(label).name == label, "invalid_restoration_label", label)
                write_json(self.output / ("restoration-" + label + ".json"), {
                    "schema": "finite-gas-fold-restoration-difference-v1", "leg": label,
                    "verified": False, "signature_differences": differences,
                    "fixed_differences": fixed_differences,
                })
            gate(actual["signature"] == expected, "full_fields_pins_history_or_controls_restore_failed", label)
            gate(actual["fixed"] == saved["fixed"], "fixed_layout_restore_failed", label)
        counters_after = {name: getattr(guard, name) for name in counter_names}
        deltas = {name: counters_after[name] - counters_before[name] for name in counter_names}
        pending_after = {"_domains_to_remesh": sorted(map(str, p._domains_to_remesh)),
                         "_remesh_requested_inside_solve": p._remesh_requested_inside_solve
                         if isinstance(p._remesh_requested_inside_solve, bool)
                         else repr(p._remesh_requested_inside_solve)}
        gate(all(value == 0 for value in deltas.values()), "physical_mesh_event_during_restore", deltas)
        gate(p._remesh_requested_inside_solve is False and not p._domains_to_remesh,
             "physical_remesh_pending_during_restore", pending_after)
        gate(not p.do_call_remeshing_when_necessary, "automatic_remesh_enabled_after_restore")
        stable_inputs(self.prepared)
        event["verified"] = True
        self.receipts.append({"leg": label, "verified": True, "complete_fields_pins_history_continuation": True,
                              "pending_request_metadata_before": pending_before,
                              "pending_request_metadata_after": pending_after,
                              "pending_request_metadata_restored": True,
                              "physical_guard_counters_before": counters_before,
                              "physical_guard_counters_after": counters_after,
                              "physical_mesh_event_deltas": deltas,
                              "phase_mesh_violation_precedes_restore": any(counters_before.values())
                              or bool(pending_before["_domains_to_remesh"])
                              or pending_before["_remesh_requested_inside_solve"] is not False})

    def probe(self, label, operation):
        saved = self.snapshot(label + "-before")
        try:
            return operation()
        finally:
            self.restore(saved, label)

    def canonicalise_continuation(self):
        p = self.problem
        before, raw = capture_continuation(p), forced_payload(p)
        if len(before["current"]) == 0:
            current = np.asarray(p.get_current_dofs()[0]).copy()
            derivative = before["derivative"] if len(before["derivative"]) else np.zeros(N)
            p._update_dof_vectors_for_continuation(derivative, current)
            if len(before["derivative"]) == 0:
                p.reset_arc_length_parameters()
                p._set_arc_length_parameter_derivative(before["parameter_derivative"])
                p._set_arc_length_theta_sqr(before["theta_sqr"])
        after = capture_continuation(p)
        old, new = continuation_signature(before), continuation_signature(after)
        gate(len(after["current"]) == N
             and {key: value for key, value in old.items() if key != "dof_current"}
             == {key: value for key, value in new.items() if key != "dof_current"},
             "continuation_canonicalisation_changed_controls")
        gate(forced_payload(p) == raw, "continuation_canonicalisation_changed_raw_history")
        expected_current = array_record(p.get_current_dofs()[0]) if not len(before["current"]) else old["dof_current"]
        gate(new["dof_current"] == expected_current, "continuation_current_buffer_differs")
        record = {"before": old, "after": new, "original_current_empty": len(before["current"]) == 0,
                  "original_empty_current_restored": False, "full_raw_payload_equal": True,
                  "scope": "explicit current-buffer preparation; original physical history unchanged"}
        write_json(self.output / "continuation-preparation.json", record)

    def check_newton(self, event):
        self.check()
        p = self.problem
        gate(p.get_bifurcation_tracking_mode() == "fold" and p.ndof() == 2 * N + 1,
             "augmented_mode_or_ndof_changed")
        ca = float(p.get_global_parameter("Ca_in").value)
        gate(np.isfinite(ca) and abs(ca - CA) <= LIMITS["ca_distance"], "ca_locality_exceeded")
        gate(float(p.get_global_parameter("Bo").value) == 1.56
             and int(p._azimuthal_mode_param_m.value) == 0, "undeclared_forcing_or_mode_changed")
        free, positional = p.get_current_dofs()
        gate(np.asarray(free).shape == (2 * N + 1,) and np.isfinite(free).all()
             and np.isfinite(p.get_current_pinned_values(True)).all(), "nonfinite_augmented_state")
        gate(float(free[N]) == ca and np.isfinite(np.asarray(free)[N + 1:]).all(),
             "augmented_parameter_slot_or_null_block_differs")
        types, names = p.get_dof_description()
        types = np.asarray(types, dtype=np.int64)
        gate(types.shape == (2 * N + 1,) and np.all(types[N:] == -1), "augmented_incidence_changed")
        meshes = actual_mesh_signature(p, self.ode_type, fold_base_ndof=N)
        gate(fixed_identity(meshes, types[:N], names, np.asarray(positional)[:N]) == self.fixed,
             "base_mesh_layout_pins_or_addresses_changed")
        gate(p.newton_solver_tolerance == self.prepared["case"]["tolerances"]["newton"],
             "newton_tolerance_changed")
        gate(p.newton_relaxation_factor == self.relaxation
             and p._get_solved_residual() == self.selected_residual
             and self.core.get_detect_inverted_elements() is True and p._inversion_remesh_threshold == 0
             and not p._domains_remesh_on_inversion and p.inversion_remesh_max_retries == 0,
             "newton_damping_residual_or_inversion_policy_changed")
        runner.append_progress(self.output, {"event": "augmented-" + event, "work": self.work, "ca_in": ca,
               "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "inversion_reports_in_augmented_solve_call": p._get_inversion_reports(),
               "use_static_condensation": bool(p.use_static_condensation),
               "quality_remesh_domains_requested": sorted(map(str, p._domains_to_remesh))})

    def stationary_matrix(self):
        p = self.problem
        p.actions_before_stationary_solve()
        for index in range(p.ntime_stepper()):
            p.time_stepper_pt(index).make_steady()
        residual, jacobian = p.assemble_jacobian(with_residual=True)
        return np.asarray(residual).copy(), jacobian.copy()

    def eigen_matrix(self):
        p = self.problem
        for index in range(p.ntime_stepper()):
            p.time_stepper_pt(index).make_steady()
        p._azimuthal_mode_param_m.value = 0
        p.actions_before_eigen_solve()
        p.setup_forced_zero_dof_list_for_eigenproblems()
        j, m, n, complex_matrix = p.get_eigen_solver().get_J_M_n_and_type()
        gate(n == N and not complex_matrix, "m0_pencil_shape_or_type_differs")
        layout = modes.mode_layout(p, n, 0)
        layout = {key: value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in layout.items()}
        modes.verify_pencil(j, m, layout)
        gate(all(j.getrow(layout["globals"][name][0]).nnz > 0 for name in modes.GLOBALS),
             "coupled_global_constraint_row_missing")
        return j.copy(), m.copy(), layout

    def solve_spectrum(self, shift):
        self.check()
        gate(self.work["eigenproblem_calls"] < 2, "spectrum_call_cap_exceeded")
        solver = self.problem.get_eigen_solver()
        gate(solver.max_retries == 0 and solver.use_pardiso is False, "spectral_retry_policy_changed")
        self.work["eigenproblem_calls"] += 1
        values, vectors = self.problem.solve_eigenproblem(8, shift=shift, azimuthal_m=0, quiet=True)
        labels = np.asarray(self.problem.get_last_eigenmodes_m())
        directory = self.output / "search-returns"
        directory.mkdir(exist_ok=True)
        with (directory / f"call-{self.work['eigenproblem_calls']:02d}.npz").open("xb") as stream:
            np.savez_compressed(stream, values=np.asarray(values), vectors=np.asarray(vectors), labels=labels)
        gate(np.asarray(values).shape == (8,), "unresolved_search_conditioning",
             {"shift": shift, "returned_count": len(values), "requested_count": 8})
        gate(np.asarray(values).shape == (8,) and np.asarray(vectors).shape == (8, N)
             and labels.shape == (8,) and np.all(labels == 0), "spectrum_count_layout_or_labels_changed")
        return np.asarray(values).copy(), np.asarray(vectors).copy()

    def update_session(self, monitor):
        return install_update_monitor(self, monitor)


def activate_fold(problem, seed):
    gate(int(problem._azimuthal_mode_param_m.value) == 0, "nonzero_fold_mode")
    problem.activate_bifurcation_tracking("Ca_in", "fold", blocksolve=False,
                                          eigenvector=seed, eigenvector_scaling="auto")


def null_actions(residual, stationary, jacobian, mass, q, layout, case):
    q = unit(q)
    gate(np.asarray(residual).shape == (N,) and np.isfinite(residual).all(), "invalid_stationary_residual")
    gate(stationary.shape == jacobian.shape == mass.shape == (N, N), "null_pencil_dimension_differs")
    gate(all(np.isfinite(matrix.data).all() and not np.iscomplexobj(matrix.data)
             for matrix in (stationary, jacobian, mass)), "nonfinite_or_complex_m0_pencil")
    jnorm, snorm = float(np.linalg.norm(jacobian.data)), float(np.linalg.norm(stationary.data))
    mass_scale = float(np.max(np.abs(mass.data), initial=0))
    gate(jnorm > 0 and snorm > 0 and mass_scale > 0, "degenerate_operator_scale")
    difference = (jacobian + stationary).tocsr()
    free = np.ones(N, dtype=bool)
    free[layout["forced"]] = False
    diagnostic = modes.diagnose_pair(0j, q, jacobian, mass, layout, case["physics"]["r_tube"])
    metrics = {"steady_residual_linf": float(np.max(np.abs(residual))),
               "stationary_null_scaled_frobenius": float(np.linalg.norm(stationary @ q) / snorm),
               "dynamic_null_scaled_frobenius": float(np.linalg.norm(jacobian @ q) / jnorm),
               "mass_action_scaled_max_entry": float(np.linalg.norm(mass @ q) / mass_scale),
               "expected_sign": "J_eigen=-J_stationary", "nonforced_sign_difference_scaled":
               float(np.linalg.norm(difference[free].data) / snorm),
               "nonforced_sign_action_scaled": float(np.linalg.norm((difference @ q)[free]) / snorm),
               "full_J_plus_Jstat_actual_nonzeros": int(np.count_nonzero(difference.data)),
               "forced_row_count": len(layout["forced"]),
               "forced_coefficient_gate_vacuous": not layout["forced"],
               "null_pencil_diagnostic": diagnostic, "unit_null_l2": float(np.linalg.norm(q))}
    stationary_passed = (metrics["steady_residual_linf"] <= case["tolerances"]["newton"]
                          and metrics["stationary_null_scaled_frobenius"] <= LIMITS["null"])
    dynamic_passed = (metrics["dynamic_null_scaled_frobenius"] <= LIMITS["null"]
                      and metrics["mass_action_scaled_max_entry"] >= LIMITS["mass"]
                      and diagnostic["pencil_qualified"] is True)
    metrics.update(stationary_null_passed=stationary_passed, direct_dynamic_null_passed=dynamic_passed)
    return metrics


def returned_null_metrics(values, vectors, jacobian, mass, q, layout, case):
    rows = []
    for index, (value, vector) in enumerate(zip(values, vectors, strict=True)):
        v = unit(vector)
        overlap = float(min(1.0, abs(np.vdot(unit(q), v))))
        diagnostic = modes.diagnose_pair(value, v, jacobian, mass, layout, case["physics"]["r_tube"])
        passed = (value.imag == 0 and abs(value) < LIMITS["root"] and overlap >= LIMITS["overlap"]
                  and diagnostic["pencil_qualified"] is True)
        physical_types = {i for i, name in enumerate(layout.get("names", []))
                          if name.startswith(("liquid/", "gas/")) and name.rsplit("/", 1)[-1]
                          in {"velocity_x", "velocity_y", "velocity_phi", "mesh_x", "mesh_y", "pressure"}}
        physical = np.isin(np.asarray(layout.get("types", [])), list(physical_types))
        restricted_overlap = None
        if physical.shape == (N,) and np.any(physical):
            a, b = np.linalg.norm(q[physical]), np.linalg.norm(v[physical])
            if a > 0 and b > 0:
                restricted_overlap = float(min(1., abs(np.vdot(q[physical], v[physical])) / (a * b)))
        rows.append({"index": index, "lambda_real": float(value.real), "lambda_imag": float(value.imag),
                     "coefficient_overlap": overlap, "original_pencil_diagnostic": diagnostic,
                     "physical_field_overlap_without_multipliers_diagnostic_only": restricted_overlap,
                     "returned_real_null_matches": bool(passed)})
    return rows


def construct_type(core, factory):
    problem_type = factory()[0]
    bridge = BeforeStepBridge(problem_type, core.Problem)
    return problem_type, bridge


def build_bridged(problem_type, case):
    problem = model.build_problem(case, initial_bond=case["physics"]["bond"])
    gate(type(problem) is problem_type, "cached_scientific_class_identity_changed")
    return problem


def stationary_bc_idempotence(problem, full_capture=None):
    before = model._stationary_state_signature(problem, True)
    full_before = full_capture() if full_capture is not None else None
    problem.actions_before_stationary_solve(force_reassign_eqs=False)
    first = model._stationary_state_signature(problem, True)
    full_first = full_capture() if full_capture is not None else None
    problem.actions_before_stationary_solve(force_reassign_eqs=False)
    second = model._stationary_state_signature(problem, True)
    full_second = full_capture() if full_capture is not None else None
    gate(before == first == second and full_before == full_first == full_second, "stationary_BC_not_idempotent")
    return {"stationary_calls": 2, "complete_signature_equal": True,
            "bc_cache_marker": second["bc_cache_marker"], "layout": second["layout"]}


def template_registration(problem):
    """Read the reviewed owned-template graph without changing registration."""
    originals = list(problem._meshtemplate_list)
    domains = sorted(problem._meshdict.items())
    inversion = list(problem._domains_remesh_on_inversion)
    templates = []
    for item in [*originals, *inversion, *(getattr(mesh, "_templatemesh", None) for _, mesh in domains)]:
        if item is not None and not any(item is prior for prior in templates):
            templates.append(item)
    rows = []
    for template in templates:
        remesher = getattr(template, "remesher", None)
        owner = None
        try:
            owner = template.get_problem()
        except (AttributeError, RuntimeError, TypeError):
            pass
        rows.append({"template_id": id(template), "type": type(template).__module__ + "." + type(template).__qualname__,
                     "override_id": id(template._template_override) if getattr(template, "_template_override", None) is not None else None,
                     "owned": owner is problem, "remesher_id": id(remesher) if remesher is not None else None,
                     "remesher_template_id": id(getattr(remesher, "template", None)) if remesher is not None else None,
                     "counter": getattr(remesher, "_cnt", None)})
    return {"original_templates": [id(item) for item in originals],
            "inversion_templates": sorted(id(item) for item in inversion),
            "quality_templates": sorted(id(item) for item in problem._domains_to_remesh),
            "domains": {name: {"mesh_id": id(mesh), "template_id": id(getattr(mesh, "_templatemesh", None))}
                        for name, mesh in domains}, "templates": sorted(rows, key=lambda row: row["template_id"])}


def inspect_inversion_templates(problem, template_type, remesher_type):
    """Admit one owned physical remesher's original/current template aliases."""
    before = template_registration(problem)
    originals = list(problem._meshtemplate_list)
    domains = set(problem._domains_remesh_on_inversion)
    label = "finite_gas_inversion_template_policy_differs"
    gate(len(originals) == 1 and type(originals[0]) is template_type, label, before)
    original = originals[0]
    active = original._get_template()
    owner = getattr(active, "remesher", None)
    gate(type(active) is template_type and active._get_template() is active and type(owner) is remesher_type
         and owner.template is active, label, before)
    gate(domains == {original, active} and all(type(item) is template_type for item in domains), label, before)
    gate(all(name in problem._meshdict and problem._meshdict[name]._templatemesh is active
             for name in ("liquid", "gas")), label, before)
    gate(all(item.get_problem() is problem and item.remesher is owner
             and all(item.has_domain(name) for name in ("liquid", "gas"))
             and (item is active or item is original and item._get_template() is active)
             for item in domains), label, before)
    after = template_registration(problem)
    gate(before == after, "template_inspection_changed_registration", {"before": before, "after": after})
    return {"registration_before": before, "registration_after_inspection": after,
            "registration_changed_by_inspection": False, "original_template_id": id(original),
            "active_template_id": id(active), "remesher_id": id(owner), "registered_alias_count": len(domains),
            "one_owned_physical_remesher": True, "current_liquid_and_gas_template_verified": True}


@contextlib.contextmanager
def augmented_controls(problem, core, record, template_type, remesher_type):
    saved = {name: getattr(problem, name) for name in
             ("max_newton_iterations", "newton_relaxation_factor", "newton_solver_tolerance",
              "inversion_remesh_max_retries")}
    domains = set(problem._domains_remesh_on_inversion)
    policy = {"detect_inverted_elements": core.get_detect_inverted_elements(),
              "inversion_remesh_threshold": problem._inversion_remesh_threshold,
              "inversion_remesh_max_retries": problem.inversion_remesh_max_retries,
              "registration": template_registration(problem)}
    record["inversion_template_policy_before"] = policy
    gate(policy["detect_inverted_elements"] is True and policy["inversion_remesh_threshold"] == 0,
         "finite_gas_inversion_template_policy_differs", policy)
    record["inversion_template_ownership"] = inspect_inversion_templates(problem, template_type, remesher_type)
    record.update(original_inversion_templates=[type(template).__name__ for template in domains],
                  inversion_reports_in_previous_solve_call_at_entry=problem._get_inversion_reports(),
                  inversion_counter_semantics="reset at each native solve call; not cumulative",
                  globally_convergent_newton_requested=False, relaxation_requested=1.0,
                  original_solve_controls=saved, inversion_detection_retained=True,
                  inversion_wrapper_domains_empty_only_inside_augmented_scope=True)
    problem._domains_remesh_on_inversion = set()
    problem.inversion_remesh_max_retries = 0
    try:
        yield
    finally:
        problem.deactivate_bifurcation_tracking()
        for name, value in saved.items():
            setattr(problem, name, value)
        problem._set_globally_convergent_newton_method(False)
        problem._domains_remesh_on_inversion = domains
        after = template_registration(problem)
        before = record["inversion_template_ownership"]["registration_before"]
        record["inversion_template_registration_after"] = after
        record["original_template_ownership_graph_restored"] = (
            {key: value for key, value in before.items() if key != "quality_templates"}
            == {key: value for key, value in after.items() if key != "quality_templates"})
        record.update(inversion_reports_in_augmented_solve_call=problem._get_inversion_reports(),
                      original_inversion_templates_restored=problem._domains_remesh_on_inversion == domains,
                      original_solve_controls_restored=all(getattr(problem, name) == value for name, value in saved.items()),
                      globally_convergent_newton_restore_value=False,
                      globally_convergent_newton_restore_setter_called=True)
        gate(record["original_solve_controls_restored"] and record["original_inversion_templates_restored"]
             and record["original_template_ownership_graph_restored"]
             and core.get_detect_inverted_elements() is True, "augmented_controls_restoration_failed", record)


def adopt_measured_case(problem, original, value):
    case = measured_case(original, value)
    transition = approach.iii.anchor.transition_contract(original, case)
    before = model._stationary_state_signature(problem, True)
    pins = diagnostic.actual_pins(problem)
    problem.case = copy.deepcopy(case)
    after = model._stationary_state_signature(problem, True)
    gate(before == after and pins == diagnostic.actual_pins(problem)
         and after["parameters"] == {"Ca_in": case["physics"]["ca_in"], "Bo": 1.56},
         "measured_case_declaration_changed_native_state")
    return case, {"transition": transition, "complete_signature_equal": True, "actual_pins_equal": True,
                  "native_parameter_assignments": 0}


def run_candidate(adapter, prepared, output, arguments):
    from scipy import sparse

    p = adapter.problem
    adapter.canonicalise_continuation()
    original = adapter.snapshot("original-admitted")
    adapter.fixed = original["fixed"]
    adapter.restore(original, "initial-full-state-roundtrip")
    adapter.selected_residual = original["signature"]["selected_residual"]
    adapter.relaxation = p.newton_relaxation_factor
    result = {"schema": SCHEMA, "status": "inconclusive", "numerical_acceptance": False,
              "stationary_fold_candidate_passed": False, "direct_dynamic_null_passed": False,
              "returned_real_null_matched": False, "work": adapter.work, **NO_CLAIMS}
    solve_record = {}
    try:
        bc_record = stationary_bc_idempotence(p, lambda: adapter.signature()["signature"])
        write_json(output / "stationary-BC-idempotence.json", bc_record)
        seed_j, seed_m, seed_layout = adapter.probe("seed-pencil", adapter.eigen_matrix)
        bank = prepared["contract"]["banks"][0]
        saved_layout = runner._read_json(Path(bank["layout"]["path"]))
        gate(seed_layout == saved_layout, "live_seed_layout_differs_from_own_bank")
        with np.load(bank["bank"]["path"], allow_pickle=False) as saved:
            vector, value = saved["vectors"][0].copy(), saved["values"][0]
        seed_diagnostic = modes.diagnose_pair(value, vector, seed_j, seed_m, seed_layout,
                                               prepared["case"]["physics"]["r_tube"])
        gate(seed_diagnostic["pencil_qualified"] is True, "saved_seed_live_pencil_gate_failed", seed_diagnostic)
        seed, rotation = real_seed(vector)
        write_json(output / "seed.json", {"bank": bank, "index": 0, "rotation": rotation,
                                          "live_pencil_diagnostic": seed_diagnostic,
                                          "identified_fold_family": False})
        del seed_j, seed_m, vector
        monitor = UpdateMonitor(adapter)
        try:
            adapter.check()
            with augmented_controls(p, adapter.core, solve_record, adapter.template_type, adapter.remesher_type):
                activate_fold(p, seed)
                gate(not p._domains_remesh_on_inversion, "inversion_wrapper_rearmed_during_activation")
                adapter.work["augmented_solve_calls"] += 1
                with adapter.update_session(monitor):
                    p.solve(spatial_adapt=0, max_newton_iterations=20, do_not_set_IC=True,
                            newton_relaxation_factor=1.0, globally_convergent_newton=False)
                q = np.asarray(p.get_last_eigenvectors()[0]).copy()
                gate(q.shape == (N,) and np.all(q.imag == 0), "tracked_null_not_real_base_vector")
                q = unit(q.real)
                ca_fold = float(p.get_global_parameter("Ca_in").value)
        finally:
            solve_record.update(completed_updates=monitor.updates, before_iterations=monitor.before_iterations,
                                callback_solve_starts=monitor.starts,
                                update_outcome="zero_completed_updates" if monitor.updates == 0 else "completed_updates")
            result["augmented_solve"] = solve_record
            write_json(output / "augmented-solve.json", solve_record)
        gate(p.ndof() == N, "base_ndof_not_restored_after_tracking")
        gate(p._bifurcation_reactivation_after_adaptation is None,
             "pending_bifurcation_reactivation_before_cache_retirement")
        tracked_cache = eigen_signature(copy_eigendata(p))
        p.invalidate_eigendata()
        empty_cache = copy_eigendata(p)
        gate(all(isinstance(empty_cache[name], np.ndarray) and empty_cache[name].size == 0
                 and empty_cache[name].dtype == np.dtype(np.complex128)
                 for name in ("_last_eigenvalues", "_last_eigenvectors"))
             and empty_cache["_last_eigenvalues_m"] is None
             and empty_cache["_last_eigenvalues_k"] is None,
             "tracked_eigen_cache_retirement_failed")
        result["tracked_eigen_cache_retirement"] = {
            "before": tracked_cache, "after": eigen_signature(empty_cache),
            "native_method": "invalidate_eigendata", "pending_reactivation": False,
            "tracked_null_retained_separately": True}
        original_history = history.serialized_metadata(Path(prepared["restart"]["state"]["path"]).read_bytes())
        result["preserved_time_dt_step_counts"] = history.check_time_history(p, original_history)
        case, adoption = adopt_measured_case(p, prepared["case"], ca_fold)
        fold = output / "fold"
        (fold / "solver").mkdir(parents=True)
        case_path = fold / "case.json"
        write_json(case_path, case)
        gate(model.read_case(case_path)["physics"]["ca_in"] == float(p.get_global_parameter("Ca_in").value),
             "measured_case_JSON_parameter_roundtrip_failed")
        write_json(fold / "case-adoption.json", adoption)
        write_json(fold / "run-manifest.json", manifest(case_path, case, prepared, arguments))
        runner.append_progress(fold, {"event": "prepared", "case_id": case["case_id"]}, first=True)
        warm = remesh.sample(p, model, case, fold, "warm", np)
        checks = remesh.canonical_checks(case, warm["metrics"])
        gate(set(checks) == modes.GATES and all(checks.values()), "fold_original_warm_gates_failed", checks)
        np.save(output / "null.npy", q)
        solved = adapter.snapshot("solved-fold")
        residual, stationary = adapter.probe("stationary-assembly", adapter.stationary_matrix)
        jacobian, mass, layout = adapter.probe("dynamic-assembly", adapter.eigen_matrix)
        for name, matrix in (("Jstat", stationary), ("J", jacobian), ("M", mass)):
            sparse.save_npz(output / (name + ".npz"), matrix)
        np.save(output / "residual.npy", residual)
        write_json(output / "layout.json", layout)
        actions = null_actions(residual, stationary, jacobian, mass, q, layout, case)
        write_json(output / "null-actions.json", actions)
        result.update(ca_fold=ca_fold, stationary_fold_candidate_passed=actions["stationary_null_passed"],
                      direct_dynamic_null_passed=actions["direct_dynamic_null_passed"], null_actions=actions)
        gate(actions["stationary_null_passed"], "stationary_null_action_failed", actions)
        gate(actions["direct_dynamic_null_passed"],
             "static_algebraic_or_dynamic_null_action_failed", actions)
        with adapter.mesh_guard.administrative("fold-checkpoint-roundtrip") as event:
            checkpoint = model._checkpoint_stationary_state(p, case, fold, 16, True)
        gate(event["callbacks"] == ["actions_after_adapt", "actions_after_remeshing"] * 2,
             "fold_checkpoint_callback_inventory_differs", event)
        restart = runner.validate_restart(fold / checkpoint["resume_packet"], case, sha(case_path),
                                          component=prepared["component"], environment=prepared["environment"],
                                          azimuthal_stability=True)
        gate(approach.iii.physical_signature(warm["signature"])
             == approach.iii.physical_signature(restart["roundtrip"]["after"]), "fold_checkpoint_changed_state")
        adapter.restore(solved, "checkpoint-restoration")
        event["verified"] = True
        pairs = []
        for index, shift in enumerate(LIMITS["shifts"]):
            values, vectors = adapter.probe(f"eigen-{index}", lambda shift=shift: adapter.solve_spectrum(shift))
            with (output / f"shift-{index}.npz").open("xb") as stream:
                np.savez_compressed(stream, values=values, vectors=vectors)
            rows = returned_null_metrics(values, vectors, jacobian, mass, q, layout, case)
            pairs.append({"shift": shift, "pairs": rows})
            write_json(output / f"shift-{index}.json", pairs[-1])
        result.update(returned_pairs=pairs, returned_real_null_matched=any(
            row["returned_real_null_matches"] for search in pairs for row in search["pairs"]),
                      fold_checkpoint=checkpoint, fold_case=case, warm_checks=checks,
                      cold_plan={"case": regular(case_path), "restart": restart, "pins": warm["actual_pins"]})
        result["status"] = "candidate_ready_for_cold_qualification"
    except BaseException as error:  # noqa: BLE001 - retain partial evidence and restore on every failure
        label = getattr(error, "label", "native_or_adapter_refusal")
        if solve_record.get("inversion_reports_in_augmented_solve_call", 0) > 0:
            label = "inverted_element_during_augmented_solve"
        result.update(status="inconclusive", refusal={"label": getattr(error, "label", "native_or_adapter_refusal"),
                      "error_type": type(error).__name__, "error": str(error),
                      "evidence": getattr(error, "evidence", None)})
        result["refusal"]["label"] = label
    finally:
        try:
            adapter.restore(original, "original-final-restoration")
            result["original_full_state_restored"] = True
        except BaseException as error:  # noqa: BLE001 - retain partial evidence and restore on every failure
            result.update(status="inconclusive", original_full_state_restored=False,
                          restoration_failure={"error_type": type(error).__name__, "error": str(error)})
        result.update(restoration=adapter.receipts, physical_mesh_checks=adapter.mesh_guard.checks,
                      mesh_callbacks=adapter.mesh_guard.events, work=adapter.work,
                      physical_remeshes_performed=0,
                      physical_after_callbacks_observed=adapter.mesh_guard.physical_callbacks,
                      remesh_or_adapt_requests_observed=adapter.mesh_guard.requests,
                      pending_remesh_checks=adapter.mesh_guard.pending_remesh_calls)
    return result


def execute(prepared, output, arguments, deadline):
    gate(prepared["native_execution_authorized"] is True, "metadata_cannot_authorise_native_execution")
    stable_inputs(prepared)
    from pyoomph import _pyoomph_core as core
    from pyoomph.generic.problem import GenericProblemHooks
    from pyoomph.meshes.mesh import ODEStorageMesh
    from pyoomph.meshes.remesher import RemesherViaRecreation

    from problems.paper_mesh import PaperTwoPhaseMesh

    gate(Path(core.__file__).resolve() == Path(prepared["environment"]["native_core"]["path"]).resolve()
         and sha(core.__file__) == diagnostic.CORE, "native_core_binding_changed")
    problem_type, bridge = construct_type(core, model._native_problem_types)
    result, problem, adapter = None, None, None
    before_detection = core.get_detect_inverted_elements()
    try:
        problem = build_bridged(problem_type, prepared["case"])
        bridge.verify_problem(problem)
        problem.set_output_directory(str(output / "solver"))
        problem.set_linear_solver("superlu")
        problem.set_c_compiler("system")
        problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        problem.set_eigensolver("spectra")
        problem.initialise()
        gate(not problem.is_distributed(), "serial_fold_required")
        problem.get_eigen_solver().use_pardiso = False
        problem.get_eigen_solver().max_retries = 0
        gate(problem._inversion_remesh_threshold == 0, "inversion_escalation_not_disabled")
        problem._set_globally_convergent_newton_method(False)
        adapter = Adapter(problem, prepared, output, core, ODEStorageMesh, GenericProblemHooks, bridge, deadline)
        adapter.template_type, adapter.remesher_type = PaperTwoPhaseMesh, RemesherViaRecreation
        problem._stationary_iii_receipts_directory = output / "loads"
        with adapter.mesh_guard.administrative("original-load"):
            loaded = approach.iii.restore_checked(problem, prepared["case"], prepared["restart"],
                                                  prepared["cold"]["actual_pins"],
                                                  prepared["cold"]["loaded"]["serialized_state"]["payload_sha256"])
        observation = output / "original-observation"
        observation.mkdir()
        sampled = remesh.sample(problem, model, prepared["case"], observation, "cold-load", np)
        gate(all(remesh.canonical_checks(prepared["case"], sampled["metrics"]).values()),
             "original_fresh_load_gates_failed")
        write_json(output / "original-load.json", loaded)
        core.set_detect_inverted_elements(True)
        gate(problem.newton_relaxation_factor == 1.0, "original_relaxation_not_one")
        result = run_candidate(adapter, prepared, output, arguments)
        write_json(output / "candidate-result.json", result)
    except BaseException as error:  # noqa: BLE001 - retain partial evidence and restore on every failure
        result = {"schema": SCHEMA, "status": "inconclusive", "numerical_acceptance": False,
                  "refusal": {"label": getattr(error, "label", "native_construction_or_admission_refusal"),
                              "error_type": type(error).__name__, "error": str(error),
                              "evidence": getattr(error, "evidence", None)}, **NO_CLAIMS}
    finally:
        released = problem is None
        try:
            if problem is not None:
                problem.release()
                released = True
        except BaseException as error:  # noqa: BLE001 - retain native release failure
            if result is None:
                result = {"schema": SCHEMA, "numerical_acceptance": False, **NO_CLAIMS}
            result.update(status="inconclusive", release_failure=str(error))
        finally:
            core.set_detect_inverted_elements(before_detection)
            if released:
                bridge_record = bridge.restore_after_release(native_session_released=True)
            else:
                bridge_record = {"verified": False, "native_release_verified": False,
                                 "class_bridge_restored": False}
            write_json(output / "class-bridge-restoration.json", bridge_record)
    if result["status"] == "candidate_ready_for_cold_qualification":
        cold_output = output / "fresh-fold-cold"
        cold_output.mkdir()
        try:
            result.update(cold_checks=qualify_fold_cold(problem_type, prepared, result, cold_output),
                          cold_load_verified=True, numerical_acceptance=bool(result["returned_real_null_matched"]),
                          status="complete" if result["returned_real_null_matched"] else "inconclusive")
        except BaseException as error:  # noqa: BLE001 - preserve candidate even when cold qualification refuses
            result.update(status="inconclusive", numerical_acceptance=False, cold_load_verified=False,
                          cold_refusal={"label": getattr(error, "label", "cold_qualification_refusal"),
                                        "error_type": type(error).__name__, "error": str(error)})
    result.pop("cold_plan", None)
    stable_inputs(prepared)
    write_json(output / "result.json", result)
    return result


def qualify_fold_cold(problem_type, prepared, result, cold_output):
    plan = result["cold_plan"]
    with build_bridged(problem_type, result["fold_case"]) as cold:
        cold.set_output_directory(str(cold_output / "solver"))
        cold.set_linear_solver("superlu")
        cold.set_c_compiler("system")
        cold.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        cold.initialise()
        cold._stationary_iii_receipts_directory = cold_output / "loads"
        loaded = approach.iii.restore_checked(cold, result["fold_case"], plan["restart"], plan["pins"],
                                              plan["restart"]["roundtrip"]["after"]["serialized_state"]["payload_sha256"])
        sampled = remesh.sample(cold, model, result["fold_case"], cold_output, "cold-load", np)
        checks = remesh.canonical_checks(result["fold_case"], sampled["metrics"])
        gate(set(checks) == modes.GATES and all(checks.values()), "fold_fresh_cold_gates_failed")
        gate(sampled["actual_pins"] == plan["pins"]
             and approach.iii.physical_signature(sampled["signature"])
             == approach.iii.physical_signature(loaded["loaded"])
             == approach.iii.physical_signature(plan["restart"]["roundtrip"]["after"]),
             "fold_cold_fields_pins_or_new_checkpoint_history_differs")
        loaded["canonical_checks"] = checks
        write_json(cold_output / "cold-load.json", loaded)
        return checks


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "case", "manifest", "packet", "state", "spectrum", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--deadline")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    if args.check:
        print(json.dumps({"verified": True, "native_imported": False, "native_execution_authorized": False,
                          "contract": prepared["contract"], "required_seal": prepared["required_seal"],
                          "source_seals_pending": prepared["source_seals_pending"]}, allow_nan=False))
        return 0
    gate(args.deadline is not None, "absolute_deadline_required")
    deadline = datetime.fromisoformat(args.deadline)
    gate(deadline.tzinfo is not None and 0 < (deadline - datetime.now(UTC)).total_seconds() <= LIMITS["wall_seconds"],
         "phase_deadline_expired_or_unbounded")
    output = prepared["output"]
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(output / "run-manifest.json", manifest(args.case, prepared["case"], prepared, arguments))
    runner.append_progress(output, {"event": "prepared"}, first=True)
    write_json(output / "preparation.json", {"contract": prepared["contract"], "inputs": prepared["records"],
               "source_files": prepared["implementation"], "deadline_utc": deadline.isoformat()})
    try:
        result = execute(prepared, output, arguments, deadline)
    except BaseException as error:  # noqa: BLE001 - retain partial evidence and restore on every failure
        result = {"schema": SCHEMA, "status": "inconclusive", "numerical_acceptance": False,
                  "refusal": {"label": getattr(error, "label", "native_or_restoration_refusal"),
                              "error_type": type(error).__name__, "error": str(error),
                              "evidence": getattr(error, "evidence", None)}, **NO_CLAIMS}
        write_json(output / "failure.json", result)
    write_json(output / "completion.json", {**result, "files": approach.iii.inventory(output)})
    runner.append_progress(output, {"event": "final", "status": result["status"],
                                    "numerical_acceptance": result["numerical_acceptance"]})
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
