"""Native-free contracts for the four conserved-volume finite-gas jumps.

Artifact consistency establishes a software contract. It does not establish
the numerical meaning of a fold, a transfer norm or an endpoint certificate.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from problems import paper_bubble as model
from runners import run_paper_case as stationary

PYOOMPH_PIN = "a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618"
FEM_BASE = "905c56d802ae85a3c10143f2f9f13bbc07f29ea1"
RATIO = 0.6716793862139855
MODEL = {
    "problem": "finite_gas_bubble", "pressure": "static",
    "gravity": "explicit_downward_both_phases", "stress_jump": "ordinary_laplace",
    "bond": 1.56, "vhat": 3.3, "domain_over_r": 15.0,
    "upper_tip_over_r": 10.0, "r0": 1.0, "r_tube": RATIO,
    "reynolds": 0.0, "gas_density_ratio": 0.000949,
    "gas_viscosity_ratio": 0.000018, "frame": "front_tip",
    "ports": "paper_mixed", "volume": "fixed_pressure_constraint",
    "surface_tension": 1.0, "liquid_viscosity": 1.0,
}
ROLES = {d + "_dt_hat_" + str(dt): (d, dt)
         for d in ("I_to_III", "III_to_I") for dt in (50, 25)}
PROFILE_TIMES = {"I_to_III": [31, 220, 430, 1000], "III_to_I": [10, 1018, 2005, 2990]}
TRACE_ENDS = {"I_to_III": 1207, "III_to_I": 3013}
THRESHOLDS = {
    "volume_relative", "surface_volume_relative", "remesh_volume_relative",
    "cumulative_remesh_volume_relative", "transient_newton_residual",
    "remesh_geometry_profile_norm", "remesh_history_profile_norm",
    "flux_absolute", "static_port_pressure_absolute", "pair_L_over_R",
    "pair_Ca_b", "pair_profile_norm", "endpoint_stationary_residual",
    "endpoint_window_L_drift_over_R", "endpoint_window_Ca_b_drift",
    "endpoint_profile_norm",
}
PROFILE_NORMS = {
    "remesh_geometry_profile_norm", "remesh_history_profile_norm",
    "pair_profile_norm", "endpoint_profile_norm",
}
FOLD_GATES = {
    "stationary_fold_candidate_passed", "direct_dynamic_null_passed",
    "returned_real_null_matched", "cold_load_verified", "numerical_acceptance",
    "branch_identity_qualified",
}
UNCERTAINTY_SOURCES = {
    "fold_location", "solver_residual", "warm_cold_variability",
    "spatial_discretisation", "model_sensitivity",
}
STARTUP = "stationary_impulsive_history_BDF1_then_BDF2"
ACTIVATION = "undo_make_steady_on_all_actual_steady_steppers_before_jump"
FLAGS = ("_first_step", "_taken_already_an_unsteady_step", "_last_step_was_stationary")

# These gates require a production adapter and native evidence. They are not
# JSON switches; native commands refuse while this source has these gaps.
UNESTABLISHED_NATIVE_GATES = (
    "actual fold load and changed-control next-step qualification",
    "complete native Data/position/ODE history and stepper witness adapter",
    "native Newton callback bridge and cumulative update/inversion accounting",
    "owned two-phase quality/inversion remesh and distinctive old-history mapping norms",
    "portable dynamic checkpoint and uninterrupted/cold-resume equivalence",
    "separate-copy endpoint qualification and common-time pair comparison",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, *, positive=False, signed=False):
    require(type(value) in (int, float) and math.isfinite(value), label + " must be finite")
    if not signed:
        require(value > 0 if positive else value >= 0, label + " is out of bounds")
    return value


def integer(value, label, *, minimum=1):
    require(type(value) is int and value >= minimum, label + " must be a bounded integer")
    return value


def nonempty(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + " is unbound")
    return value


def same_model(value):
    return (isinstance(value, dict) and value == MODEL
            and all(type(value[key]) in (int, float) for key, expected in MODEL.items()
                    if type(expected) in (int, float)))


def read_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON value: " + value)

    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=constant)
    require(isinstance(value, dict), "JSON authority must be an object")
    return value


class Artifacts:
    """Hash regular authorities without deserialising native state or arrays."""

    def __init__(self):
        self.records = {}

    def file(self, reference, label="artifact"):
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
                label + " requires a bound path and SHA-256")
        value, digest = reference["path"], reference["sha256"]
        require(isinstance(value, str) and bool(value) and isinstance(digest, str)
                and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                label + " is unbound")
        path = Path(value)
        require(path.is_absolute() and path.is_file()
                and not any(item.is_symlink() for item in (path, *path.parents)),
                label + " must be an absolute regular authority")
        path = path.resolve(strict=True)
        require(stationary.sha256_file(path) == digest, label + " digest changed")
        record = {"path": str(path), "sha256": digest}
        previous = self.records.get(str(path))
        require(previous is None or previous == record, label + " has conflicting identities")
        self.records[str(path)] = record
        return path

    def json(self, reference, label="artifact"):
        path = self.file(reference, label)
        result = read_json(path)
        self.file(reference, label)  # The parsed bytes must still be the bound bytes.
        return result

    def verify(self):
        for record in list(self.records.values()):
            self.file(record)


def case_contract(case, ca):
    case = model.validate_case(case)
    require(all(case["physics"].get(key) == MODEL[key] for key in (
        "problem", "bond", "vhat", "domain_over_r", "upper_tip_over_r", "r0", "r_tube",
        "reynolds", "gas_density_ratio", "gas_viscosity_ratio")), "parent physical model changed")
    require(case["physics"]["ca_in"] == ca, "case and measured control disagree")
    return case


def target_case(parent, ca, direction):
    result = copy.deepcopy(parent)
    result["case_id"] = parent["case_id"] + "_" + direction + "_jump"
    result["physics"]["ca_in"] = number(ca, "target Ca", signed=True)
    return case_contract(result, ca)


def mesh_inventory(reference, state, book):
    inventory = book.json(reference, "mesh inventory")
    require(inventory.get("state") == state, "mesh inventory belongs to another state")
    base = Path(nonempty(inventory.get("mesh_reference_base"), "mesh reference base"))
    state_path = book.file(state, "state")
    require(base.is_absolute() and base.resolve(strict=True) == state_path.parent,
            "jump certificates require a portable state mesh convention")
    references = stationary.state_mesh_references(state_path)
    meshes = inventory.get("meshes")
    require(references and isinstance(meshes, list) and len(meshes) == len(references),
            "every native mesh reference must be bound")
    exchange = {}
    for saved, record in zip(references, meshes, strict=True):
        require(isinstance(record, dict) and set(record) == {"stored_path", "path", "sha256"},
                "mesh inventory row is unbound")
        path = book.file({k: record[k] for k in ("path", "sha256")}, "mesh")
        require(record["stored_path"] == saved and path == (base / saved).resolve(strict=True)
                and path.is_relative_to(state_path.parent), "portable mesh reference escaped its custody")
        require(saved not in exchange, "duplicate saved mesh reference")
        exchange[saved] = str(path)
    return inventory, exchange


def state_witness(reference, state, ca, book):
    from runners import paper_dynamic_jump_runtime as runtime
    from runners import run_paper_cold_history as history

    physical = book.json(reference, "full physical/history witness")
    require(physical.get("state") == state, "physical/history witness belongs to a different state")
    witness = runtime.validate_witness(physical.get("witness"))
    metadata = history.serialized_metadata(book.file(state, "state").read_bytes())
    require(witness["time"] == metadata["time"] and witness["dts"] == metadata["dts"]
            and witness["counter"] == metadata["unsteady_steps"]
            and witness["parameters"] == {"Ca_in": ca, "Bo": MODEL["bond"]},
            "actual saved metadata and complete physical witness disagree")
    return witness


def retained_row(row, role, thresholds, book):
    """Bind every raw row's portable meshes, physical history and diagnostics."""
    from runners import paper_dynamic_jump_runtime as runtime
    from runners import run_paper_cold_history as history

    require(isinstance(row, dict), "retained native row required")
    index = integer(row.get("accepted_step_index"), "retained native step", minimum=0)
    require(index <= role["max_steps"] and integer(row.get("counter"), "retained step counter", minimum=0) == index
            and number(row.get("t_hat"), "retained native t_hat") == index * role["dt_hat"]
            and number(row.get("dt_solver"), "retained native dt", positive=True) == role["dt_solver"],
            "retained row changed the native grid/control")
    number(row.get("native_time"), "retained native clock")
    state = row.get("state")
    mesh_inventory(row.get("mesh_inventory"), state, book)
    witness = state_witness(row.get("physical_witness"), state, role["target_ca"], book)
    metadata = history.serialized_metadata(book.file(state, "retained native state").read_bytes())
    require(witness["time"] == row.get("native_time") and witness["counter"] == index
            and all(dt == role["dt_solver"] for dt in witness["dts"])
            and metadata["suggested_next_dt"] == role["dt_solver"]
            and witness["bc_cache"] == "transient"
            and all(stepper["is_steady"] is False for stepper in witness["steppers"]),
            "retained state history/clock/stepping policy differs")
    qualification = "stationary_geometry_prejump_fluid_fields" if index == 0 else "converged_native_transient"
    require(row.get("field_qualification") == qualification, "retained physical-field qualification differs")
    metrics = row.get("metrics")
    require(isinstance(metrics, dict), "retained native diagnostics required")
    if index == 0:
        runtime.volume_metrics(metrics, thresholds)
        number(metrics.get("meridional_L_over_R"), "initial curved meridional L/R", positive=True)
        number(metrics.get("Ca_b"), "initial frame speed", signed=True)
    else:
        require(witness["flags"] == runtime.DYNAMIC_FLAGS, "retained state would repeat BDF startup")
        runtime.accepted_metrics(metrics, role, thresholds)
    book.file(row.get("profile"), "retained native profile")
    return witness


def fold_parent(reference, branch, book):
    cert = book.json(reference, "parent certificate")
    require(cert.get("schema") == "stone-jump-parent-v1", "missing actual-fold certificate")
    require(cert.get("status") == "complete" and cert.get("branch") == branch,
            "unqualified fold parent")
    require(same_model(cert.get("model")) and cert.get("pyoomph_pin") == PYOOMPH_PIN,
            "fold model changed")
    require(set(cert.get("checks", {})) == FOLD_GATES
            and all(v is True for v in cert["checks"].values()), "fold gates missing")
    refs = cert.get("artifacts", {})
    required = {"case", "manifest", "fold_result", "restart_packet", "state", "mesh_inventory",
                "physical_signature", "uncertainty_evidence"}
    require(isinstance(refs, dict) and set(refs) == required, "fold artifacts are incomplete")
    for name in sorted(required):
        book.file(refs[name], "fold " + name)
    raw = book.json(refs["fold_result"], "fold result")
    require(raw.get("status") == "complete"
            and all(raw.get(name) is True for name in FOLD_GATES - {"branch_identity_qualified"}),
            "direct fold result is unqualified")
    measured = number(cert.get("ca_fold"), "measured fold", signed=True)
    require(raw.get("ca_fold") == measured, "fold result and measured control disagree")
    case = case_contract(book.json(refs["case"], "parent case"), measured)
    require(raw.get("fold_case") == case, "direct fold result belongs to a different case")
    manifest = book.json(refs["manifest"], "parent manifest")
    require(manifest.get("case_sha256") == refs["case"]["sha256"]
            and manifest.get("case_id") == case["case_id"]
            and manifest.get("pyoomph", {}).get("commit") == PYOOMPH_PIN,
            "parent manifest case or solver authority differs")
    stationary.require_clean_source(manifest.get("component"))
    require(manifest.get("component_commit") == manifest["component"]["commit"],
            "parent source authority is inconsistent")
    stationary.require_pinned_environment(manifest.get("pyoomph"))
    book.file(manifest["pyoomph"]["native_core"], "parent native core")
    inventory, exchange = mesh_inventory(refs["mesh_inventory"], refs["state"], book)
    witness = state_witness(refs["physical_signature"], refs["state"], measured, book)
    restart = book.json(refs["restart_packet"], "unchanged parent restart")
    require(restart.get("schema") == stationary.RESUME_SCHEMA
            and restart.get("case_sha256") == refs["case"]["sha256"]
            and restart.get("parent_manifest") == refs["manifest"]
            and restart.get("state") == {**refs["state"], "format": "portable_file"}
            and restart.get("meshes") == inventory["meshes"]
            and restart.get("mesh_reference_base") == inventory["mesh_reference_base"]
            and restart.get("stage", {}).get("kind") == "completed_polish"
            and restart.get("stage", {}).get("ca_in") == measured
            and restart.get("stage", {}).get("bond") == MODEL["bond"]
            and restart.get("roundtrip", {}).get("verified") is True,
            "parent restart must bind the unchanged completed fold case")
    require(restart["stage"].get("bc_mode") == witness["bc_cache"], "parent BC mode witness differs")
    branch_evidence = book.json(cert.get("branch_connection_evidence"), "branch connection evidence")
    require(branch_evidence.get("branch") == branch and branch_evidence.get("qualified") is True
            and branch_evidence.get("fold_result") == refs["fold_result"]
            and branch_evidence.get("case") == refs["case"] and same_model(branch_evidence.get("model")),
            "fold branch connection is unbound")
    qualification = book.json(cert.get("native_load_and_next_step_qualification"),
                              "parent native load and next-step qualification")
    require(qualification.get("state") == refs["state"]
            and qualification.get("physical_signature") == refs["physical_signature"]
            and qualification.get("pyoomph_pin") == PYOOMPH_PIN
            and qualification.get("case") == refs["case"]
            and qualification.get("native_core") == manifest["pyoomph"]["native_core"]
            and qualification.get("full_history_load_passed") is True
            and qualification.get("unaugmented_base_mode_passed") is True
            and qualification.get("transient_next_step_passed") is True,
            "parent native load or next-step evidence is unqualified")
    uncertainty = cert.get("uncertainty", {})
    require(book.json(refs["uncertainty_evidence"], "fold uncertainty") == uncertainty,
            "fold uncertainty differs from direct evidence")
    require(uncertainty.get("origin") == "measured_total_fold_uncertainty"
            and uncertainty.get("physical_and_numerical_sources_bound") is True,
            "fold uncertainty is not qualified")
    nonempty(uncertainty.get("method_and_error_budget"), "total fold uncertainty method")
    sources = uncertainty.get("sources", {})
    require(isinstance(sources, dict) and set(sources) == UNCERTAINTY_SOURCES,
            "total fold uncertainty sources are unbound")
    for name, source in sources.items():
        require(isinstance(source, dict) and type(source.get("included")) is bool,
                "fold uncertainty source is unbound: " + name)
        nonempty(source.get("justification"), "fold uncertainty source justification")
        evidence = source.get("evidence")
        require(isinstance(evidence, list) and evidence, "fold uncertainty source evidence missing")
        for record in evidence:
            book.file(record, "fold uncertainty source evidence")
    interval = uncertainty.get("ca_interval")
    require(isinstance(interval, list) and len(interval) == 2, "missing finite fold envelope")
    lo, hi = (number(x, "fold envelope", signed=True) for x in interval)
    require(lo < hi and lo <= measured <= hi, "invalid fold uncertainty envelope")
    width = max(measured - lo, hi - measured)
    require(width > 0 and uncertainty.get("radius") == width,
            "uncertainty radius does not match envelope")
    return {"certificate": cert, "reference": reference, "case": case,
            "manifest": manifest, "mesh_exchange": exchange, "uncertainty_radius": width,
            "witness": witness, "state": refs["state"]}


def endpoint(reference, branch, ca, thresholds, book):
    cert = book.json(reference, "endpoint certificate")
    require(cert.get("schema") == "stone-jump-endpoint-v1" and cert.get("branch") == branch
            and cert.get("ca_in") == ca and cert.get("qualified") is True
            and same_model(cert.get("model")) and cert.get("pyoomph_pin") == PYOOMPH_PIN,
            "target stationary branch is unqualified")
    refs = cert.get("artifacts", {})
    require(isinstance(refs, dict) and set(refs) == {
        "case", "state", "mesh_inventory", "physical_signature", "qualification"},
        "endpoint artifacts are incomplete")
    for name, record in refs.items():
        book.file(record, "endpoint " + name)
    case_contract(book.json(refs["case"], "endpoint case"), ca)
    mesh_inventory(refs["mesh_inventory"], refs["state"], book)
    state_witness(refs["physical_signature"], refs["state"], ca, book)
    direct = book.json(refs["qualification"], "endpoint qualification")
    require(direct.get("status") == "complete" and direct.get("numerical_acceptance") is True
            and direct.get("branch") == branch and direct.get("ca_in") == ca
            and direct.get("state") == refs["state"]
            and direct.get("physical_signature") == refs["physical_signature"],
            "direct endpoint qualification differs")
    residual = number(direct.get("stationary_residual"), "endpoint stationary residual")
    require(residual <= thresholds["endpoint_stationary_residual"], "endpoint residual exceeds budget")
    return cert


def bracket(t_hat, dt_hat):
    lo, hi = math.floor(t_hat / dt_hat), math.ceil(t_hat / dt_hat)
    return {"requested_t_hat": t_hat, "lower_step": lo, "upper_step": hi,
            "lower_t_hat": lo * dt_hat, "upper_t_hat": hi * dt_hat,
            "exact_native_time": lo == hi, "interpolated": False}


def build_plan(packet, *, book=None):
    book = Artifacts() if book is None else book
    require(packet.get("schema") == "stone-figure7-jump-plan-v1", "unknown packet schema")
    require(packet.get("execution") == {"mode": "plan_only", "native_execution_authorized": False},
            "the study packet cannot authorise native execution")
    require(same_model(packet.get("model")), "uncontrolled gravity, model or units change")
    source = packet.get("source", {})
    require(source.get("pyoomph_pin") == PYOOMPH_PIN and source.get("fem_base_commit") == FEM_BASE,
            "source pin changed")
    controls = packet.get("controls", {})
    require(isinstance(controls, dict) and set(controls) == {
        "allow_profile_interpolation", "do_not_set_IC", "max_newton_iterations_per_step",
        "max_total_newton_updates", "remesh", "spatial_adapt", "startup",
        "startup_registration_receipt", "stepper_activation", "temporal_error",
        "temporal_scale", "time_support"}, "every registered control must be explicit")
    require(controls.get("startup") == STARTUP and controls.get("stepper_activation") == ACTIVATION,
            "unregistered physical history or stepper activation")
    require(number(controls.get("temporal_scale"), "temporal scale", positive=True) == 1.0
            and controls.get("temporal_error") is None
            and type(controls.get("spatial_adapt")) is int and controls["spatial_adapt"] == 0
            and controls.get("do_not_set_IC") is True, "time or adaptation policy changed")
    require(controls.get("time_support") == "native_grid_and_two_sided_brackets"
            and controls.get("allow_profile_interpolation") is False, "unsupported profile time policy")
    remesh = controls.get("remesh", {})
    require(isinstance(remesh, dict) and set(remesh) == {
        "automatic_postsolve_quality_remesh", "final_time_remesh", "history", "interpolator",
        "max_inversion_retries_per_step", "max_per_run", "quality_timing", "topology"},
        "every registered remesh control must be explicit")
    require(remesh.get("interpolator") == "existing_InternalInterpolator"
            and remesh.get("history") == "transfer_all_history_no_flatten_after_quality_remesh"
            and remesh.get("topology") == "unchanged", "uncontrolled remesh policy")
    require(remesh.get("quality_timing") == "between_steps_before_next_solve"
            and remesh.get("automatic_postsolve_quality_remesh") is False
            and remesh.get("final_time_remesh") is False, "unregistered quality remesh timing")
    integer(remesh.get("max_per_run"), "max remeshes", minimum=0)
    integer(remesh.get("max_inversion_retries_per_step"), "max inversion retries", minimum=0)
    integer(controls.get("max_newton_iterations_per_step"), "max Newton iterations per step")
    integer(controls.get("max_total_newton_updates"), "total Newton-update bound")
    thresholds = packet.get("thresholds", {})
    require(set(thresholds) == THRESHOLDS, "acceptance predicates are not fully preregistered")
    for name, value in thresholds.items():
        number(value, name, positive=True)
    evidence = book.json(packet.get("threshold_evidence"), "threshold evidence")
    require(evidence.get("thresholds") == thresholds
            and evidence.get("preregistered_before_trajectory") is True,
            "threshold evidence does not bind the preregistration")
    require(evidence.get("length_estimator") == "existing_curved_interface_meridional_length_over_R"
            and evidence.get("profile_frame") == "front_tip", "geometry estimator or frame is unbound")
    norms = evidence.get("norms", {})
    require(isinstance(norms, dict) and set(norms) == PROFILE_NORMS, "profile/history norms are unbound")
    for name, definition in norms.items():
        nonempty(definition, name + " definition")
    startup = book.json(controls.get("startup_registration_receipt"), "startup registration")
    require(startup.get("startup") == STARTUP and startup.get("stepper_activation") == ACTIVATION
            and number(startup.get("temporal_scale"), "registered temporal scale", positive=True) == 1.0
            and startup.get("preregistered_before_trajectory") is True,
            "startup was not preregistered")
    directions = packet.get("directions", {})
    require(set(directions) == {"I_to_III", "III_to_I"}, "direction scope changed")
    bindings, values = {}, {}
    for direction, spec in directions.items():
        initial, target = direction.split("_to_")
        parent = fold_parent(spec.get("parent_certificate"), initial, book)
        cert, width = parent["certificate"], parent["uncertainty_radius"]
        offset = spec.get("offset", {})
        require(offset.get("origin") == "measured_fold_uncertainty"
                and offset.get("preregistered_before_trajectory") is True,
                "epsilon must be fixed before trajectory")
        factor = number(offset.get("factor"), "uncertainty multiplier", positive=True)
        require(factor > 1, "offset must exceed the full fold uncertainty")
        delta = factor * width
        cap = number(offset.get("maximum_delta"), "maximum epsilon", positive=True)
        require(math.isfinite(delta) and delta <= cap, "unbounded epsilon")
        expected = cert["ca_fold"] + (delta if initial == "I" else -delta)
        require(number(spec.get("target_ca"), "target Ca", signed=True) == expected,
                "target Ca is not the preregistered fold offset")
        registered = book.json(offset.get("registration_receipt"), "offset registration")
        for key in ("factor", "maximum_delta", "target_ca"):
            number(registered.get(key), "registered " + key, positive=key != "target_ca", signed=key == "target_ca")
        require(all(registered.get(key) == value for key, value in {
            "direction": direction, "target_ca": expected, "factor": factor,
            "maximum_delta": cap, "parent_certificate": spec["parent_certificate"]}.items())
            and registered.get("preregistered_before_trajectory") is True,
            "offset registration does not bind this parent/control")
        lo, hi = cert["uncertainty"]["ca_interval"]
        require(expected > hi if initial == "I" else expected < lo, "target is inside fold uncertainty")
        horizon = number(spec.get("horizon_hat"), "finite horizon", positive=True)
        require(horizon % 50 == 0, "pair horizon must lie on both native grids")
        window = number(spec.get("final_window_hat"), "finite final window", positive=True)
        require(window % 50 == 0 and window <= horizon, "final window is off the common time grid")
        endpoint_cert = endpoint(spec.get("endpoint_certificate"), target, expected, thresholds, book)
        budget = spec.get("endpoint_qualification", {})
        for key in ("max_stationary_calls", "max_newton_updates", "max_remeshes"):
            integer(budget.get(key), "endpoint " + key, minimum=0)
        require(budget.get("fresh_cold_load_required") is True, "endpoint cold qualification is required")
        values[direction] = {"direction": direction, "target_ca": expected, "delta": delta,
                             "horizon_hat": horizon, "final_window_hat": window}
        bindings[direction] = {"parent": parent, "endpoint": endpoint_cert,
                               "target_case": target_case(parent["case"], expected, direction)}
    roles = packet.get("roles", [])
    require(isinstance(roles, list) and len(roles) == 4 and all(isinstance(r, dict) for r in roles)
            and {r.get("role_id") for r in roles} == set(ROLES),
            "only the four required roles are admitted")
    rows = []
    for role in roles:
        direction, dt = ROLES[role["role_id"]]
        require(role.get("direction") == direction and type(role.get("dt_hat")) is int
                and role["dt_hat"] == dt, "role time step changed")
        require(role.get("dt_solver") == RATIO * dt, "solver time mapping changed")
        count = integer(role.get("max_steps"), "max_steps")
        spec = values[direction]
        require(count * dt == spec["horizon_hat"], "step count does not match finite horizon")
        require(type(role.get("output_every_steps")) is int and role["output_every_steps"] == 1,
                "record every accepted native step")
        require(role.get("requested_profile_times_hat") == PROFILE_TIMES[direction],
                "Figure 7 profile request scope changed")
        require(role.get("scalar_comparison_interval_hat") == [0, TRACE_ENDS[direction]]
                and spec["horizon_hat"] >= TRACE_ENDS[direction], "full reference trace lacks native support")
        support = [bracket(t, dt) for t in PROFILE_TIMES[direction]]
        require(all(row["upper_step"] <= count for row in support), "reference time has no two-sided support")
        rows.append({"role_id": role["role_id"], "dt_hat": dt, "dt_solver": RATIO * dt,
                     "max_steps": count, "time_support": support, **spec})
    book.verify()
    return {"schema": "stone-figure7-software-plan-result-v1", "software_admission_passed": True,
            "ready_for_dispatch": False, "native_execution_authorized": False, "native_imported": False,
            "qualification_claim": "bounded software contract only", "roles": rows,
            "native_verification_gates": list(UNESTABLISHED_NATIVE_GATES), "bindings": bindings}
