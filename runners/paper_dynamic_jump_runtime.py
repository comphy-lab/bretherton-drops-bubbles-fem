"""Guarded jump sequencing, independent of a native solver import.

There is deliberately no default witness/transfer/output adapter. Callers must
provide complete raw Data histories and independently measured transfer errors.
The command-line runner refuses native dispatch until those adapters and the
native restart/next-step contracts have been qualified.
"""

from __future__ import annotations

import contextlib
import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from runners import paper_dynamic_jump as study

require, number, integer = study.require, study.number, study.integer
DYNAMIC_FLAGS = dict(zip(study.FLAGS, (False, True, False), strict=True))
BASE_MODE = {"bifurcation": "", "reactivation": None, "custom_assembler": None,
             "azimuthal": 0, "solved_residual": ""}
WORK_KEYS = {"newton_updates", "solve_calls", "physical_remeshes", "inversion_retries",
             "newton_step_updates", "cumulative_transfer_relative", "transfer_ledger"}
WITNESS_KEYS = {"schema", "ndof", "data", "geometry", "layout", "parameters", "time", "dts",
                "counter", "steppers", "flags", "bc_cache", "mode", "pending_quality_remesh", "census", "policy"}


def validate_pending_remesh(policy):
    require(isinstance(policy, dict) and set(policy) == {"schema", "owner", "domains", "requested"}
            and policy["schema"] == "stone-jump-owned-quality-remesh-v1"
            and policy["owner"] == "shared_primary_template"
            and policy["domains"] == ["gas", "liquid"]
            and type(policy["requested"]) is bool, "portable owned quality-remesh policy required")
    return copy.deepcopy(policy)


def native_capture_contract():
    """Specify the source census and its still-unestablished native qualification."""
    return {"schema": "stone-jump-native-capture-obligations-v1", "pyoomph_pin": study.PYOOMPH_PIN,
            "production_adapter_implemented": True, "native_adapter_qualified": False,
            "data_census": ["recursive bulk, interface and ODE meshes", "nodal values and positions",
                            "element internal Data and global Data", "pinned and copy/alias values"],
            "history": "every native Data column at every actual ntstorage slot, including predictors",
            "stepper_binding": "bind each Data/position block to its actual stepper and clock; prove complete coverage",
            "geometry": "per-domain node order, connectivity/stride, Lagrangian and current coordinates",
            "policy": "live parameters, solve flags, BC cache, base mode and owned pending quality request",
            "qualification": "exact cold load plus unchanged next step, then distinctive-history remesh transfer"}


def owned_quality_templates(problem, hooks):
    from runners.run_paper_finite_gas_fold import inspect_inversion_templates

    inspect_inversion_templates(problem, hooks.template_type, hooks.remesher_type)
    original = problem._meshtemplate_list[0]
    active = original._get_template()
    pending = problem._domains_to_remesh
    require(isinstance(pending, set) and len(pending) <= 1
            and all(item is original or item is active for item in pending),
            "foreign/additional pending quality domain")
    return original, active


def pending_remesh_policy(problem, hooks):
    """Record the one owned request without serialising process-local object IDs."""
    owned_quality_templates(problem, hooks)
    return {"schema": "stone-jump-owned-quality-remesh-v1", "owner": "shared_primary_template",
            "domains": ["gas", "liquid"], "requested": bool(problem._domains_to_remesh)}


def restore_pending_remesh(problem, saved, hooks):
    """Restore only a saved request for the proved shared liquid/gas owner.

    At the pinned revision RemeshWhen adds its mesh's owned template to this
    queue (equations/generic.py:594-595). State loading does not retain the
    request. The caller must first prove the complete loaded physical state.
    An unexpected request is refused; no queue is replaced or cleared here.
    """
    saved = validate_pending_remesh(saved)
    _, active = owned_quality_templates(problem, hooks)
    live = pending_remesh_policy(problem, hooks)
    require(not live["requested"] or saved["requested"], "load introduced an unrecorded quality request")
    if saved["requested"] and not live["requested"]:
        problem._domains_to_remesh.add(active)
    require(pending_remesh_policy(problem, hooks) == saved, "owned quality request restoration failed")


def validate_witness(witness):
    """Require raw histories/pins/positions with complete equation coverage.

    This schema expresses the adapter's obligation. A schema-valid JSON object
    does not prove that a native adapter actually visited every native Data.
    """
    require(isinstance(witness, dict) and set(witness) == WITNESS_KEYS
            and witness["schema"] == "stone-jump-complete-physical-v3", "full physical witness required")
    ndof = integer(witness["ndof"], "native ndof")
    data = witness["data"]
    require(isinstance(data, list) and data and all(isinstance(row, dict) for row in data),
            "all nodal/internal/ODE and position histories are required")
    names, kinds, covered = set(), set(), set()
    for block in data:
        name = study.nonempty(block.get("name"), "native Data name")
        require(name not in names, "duplicate native Data block")
        names.add(name)
        kinds.add(block.get("kind"))
        rows, pins, equations, fields = (block.get(key) for key in ("rows", "pins", "equations", "fields"))
        require(isinstance(rows, list) and len(rows) in (5, 7) and isinstance(fields, list) and fields
                and all(isinstance(f, str) and f for f in fields)
                and isinstance(pins, list) and len(pins) == len(fields)
                and all(type(p) is bool for p in pins)
                and isinstance(equations, list) and len(equations) == len(fields)
                and all(type(e) is int and e < ndof for e in equations),
                "incomplete raw Data history/pin/address block")
        for row in rows:
            require(isinstance(row, list) and len(row) == len(fields), "missing native history column")
            for value in row:
                number(value, "native history value", signed=True)
        covered.update(e for e in equations if e >= 0)
    require(kinds <= {"nodal", "position", "internal", "ode"}
            and covered == set(range(ndof)), "incomplete native field/position/ODE equation coverage")
    geometry, layout = witness["geometry"], witness["layout"]
    require(isinstance(geometry, dict) and {"nodes", "connectivity", "lagrangian"} <= set(geometry)
            and isinstance(layout, dict) and layout.get("ndof") == ndof,
            "raw geometry/connectivity/lagrangian and layout witness required")
    require(isinstance(geometry["nodes"], list) and geometry["nodes"]
            and isinstance(geometry["lagrangian"], list)
            and len(geometry["nodes"]) == len(geometry["lagrangian"]), "raw native coordinates required")
    for points in (geometry["nodes"], geometry["lagrangian"]):
        for point in points:
            require(isinstance(point, list) and len(point) == 2, "raw axisymmetric coordinates required")
            for value in point:
                number(value, "raw coordinate", signed=True)
    require(isinstance(geometry["connectivity"], list) and geometry["connectivity"]
            and all(type(i) is int and 0 <= i < len(geometry["nodes"]) for i in geometry["connectivity"]),
            "raw native connectivity required")
    require(isinstance(layout.get("dof_names"), list) and layout["dof_names"]
            and all(isinstance(name, str) and name for name in layout["dof_names"])
            and isinstance(layout.get("types"), list) and len(layout["types"]) == ndof
            and all(type(i) is int and 0 <= i < len(layout["dof_names"]) for i in layout["types"]),
            "complete raw native DOF description required")
    require(witness["mode"] == BASE_MODE, "augmented, foreign-assembler or non-base mode refused")
    require(isinstance(witness["parameters"], dict) and set(witness["parameters"]) == {"Ca_in", "Bo"},
            "complete physical global control witness required")
    for name, value in witness["parameters"].items():
        number(value, name, signed=name == "Ca_in")
    number(witness["time"], "native clock", signed=True)
    integer(witness["counter"], "native step counter", minimum=0)
    require(isinstance(witness["dts"], list) and len(witness["dts"]) >= 2, "full native dt vector required")
    for dt in witness["dts"]:
        number(dt, "native dt", positive=True)
    steppers = witness["steppers"]
    require(isinstance(steppers, list) and steppers, "every actual native stepper must be recorded")
    for stepper in steppers:
        require(isinstance(stepper, dict) and {"type", "order", "ntstorage", "is_steady"} <= set(stepper)
                and isinstance(stepper["type"], str) and stepper["type"]
                and stepper["order"] == 2 and stepper["ntstorage"] in (5, 7)
                and type(stepper["is_steady"]) is bool, "unsupported native stepper/history layout")
    require(set(witness["flags"]) == set(study.FLAGS)
            and all(type(v) is bool for v in witness["flags"].values()), "actual solve flags required")
    require(witness["bc_cache"] in {"init", "stationary", "transient"}, "actual BC cache marker required")
    validate_pending_remesh(witness["pending_quality_remesh"])
    from runners.paper_dynamic_capture import validate_census
    validate_census(witness["census"], data, steppers)
    if witness["census"]["ownership"]["origin"] == "exact_pinned_primary_factory_chain":
        require(steppers[0]["time"] == witness["time"] and steppers[0]["dts"] == witness["dts"]
                and steppers[0]["counter"] == witness["counter"], "primary stepper clock/counter witness differs")
        require(isinstance(geometry.get("domains"), dict)
                and set(geometry["domains"]) == set(witness["census"]["enumerated_meshes"]),
                "native geometry/census mesh inventory differs")
    require(isinstance(witness["policy"], dict) and {"all_parameters", "default_scheme"} <= set(witness["policy"])
            and set(witness["policy"]) <= {"all_parameters", "default_scheme", "serialization"}
            and witness["policy"]["default_scheme"] == "BDF2"
            and isinstance(witness["policy"]["all_parameters"], dict)
            and all(witness["policy"]["all_parameters"].get(name) == value for name, value in witness["parameters"].items()),
            "complete live forcing/default stepping policy required")
    for value in witness["policy"]["all_parameters"].values():
        number(value, "native global parameter", signed=True)
    if witness["census"]["ownership"]["origin"] == "exact_pinned_primary_factory_chain":
        serialization = witness["policy"].get("serialization")
        require(isinstance(serialization, dict) and set(serialization) == {"output_step", "continue_step",
                "suggested_next_dt", "run_endtime", "remesher_counter"}, "complete serialization policy required")
        for key in ("output_step", "continue_step", "remesher_counter"):
            integer(serialization[key], "native " + key, minimum=0)
        for key in ("suggested_next_dt", "run_endtime"):
            if serialization[key] is not None:
                number(serialization[key], "native " + key, positive=key == "suggested_next_dt", signed=key == "run_endtime")
    return copy.deepcopy(witness)


def physical(witness, *, current_only=False, omit=()):
    result = copy.deepcopy(witness)
    for key in omit:
        result.pop(key)
    for stepper in result.get("steppers", []):
        stepper.pop("is_steady")  # Flags/weight banks are checked separately.
        stepper.pop("weights", None)
        for key in omit:
            stepper.pop(key, None)
        if "time" in omit or "dts" in omit:
            stepper.pop("time_history", None)
    for view in result.get("geometry", {}).get("domains", {}).values():
        view.pop("nodal_buffer", None)
        view.pop("elemental_buffer", None)
    if current_only:
        for block in result["data"]:
            block["rows"] = block["rows"][:1]
    return result


def stepper_registry(witness):
    """Separate immutable registry fields from the declared clock/weight changes."""
    dynamic = {"time", "time_history", "dts", "counter", "weights"}
    return [{key: value for key, value in stepper.items() if key not in dynamic}
            for stepper in witness["steppers"]]


def history_times(current, dts):
    result = [current]
    for dt in dts:
        current -= dt  # Pinned Time::time(t) uses this subtraction order.
        result.append(current)
    return result


def clock_transition(before, after, *, dt=None):
    require(stepper_registry(before) == stepper_registry(after), "clock transition changed stepper registry")
    for old, new in zip(before["steppers"], after["steppers"], strict=True):
        for key in ("time", "dts", "counter", "time_history", "weights"):
            require((key in old) == (key in new), "clock transition changed stepper witness coverage")
        if "time" in new:
            require(new["time"] == after["time"] and new["counter"] == old["counter"],
                    "clock transition changed actual counter/time policy")
            require(new["dts"] == (old["dts"] if dt is None else [dt] * len(old["dts"])),
                    "clock transition changed unregistered stored dts")
            require(new["time_history"] == history_times(new["time"], new["dts"]),
                    "native clock history is not source-consistent")
        if dt is None:
            require(old.get("weights") == new.get("weights"), "origin reset changed stepper weights")
        elif "weights" in new:
            banks = copy.deepcopy(old["weights"])
            nt = old["ntstorage"]
            require(set(banks) == {"BDF1", "BDF2", "Newmark2"}
                    and all(len(bank) == 3 and all(len(row) == nt for row in bank) for bank in banks.values()),
                    "complete pinned weight banks required for dt fill")
            # Exact a186 set_weights formulas, with the frozen default beta1=beta2=1/2.
            banks["BDF1"][1] = [1 / dt, -1 / dt] + [0.0] * (nt - 2)
            banks["BDF2"][1] = [1 / dt + 1 / (dt + dt), -(dt + dt) / (dt * dt),
                                 dt / ((dt + dt) * dt)] + [0.0] * (nt - 3)
            acceleration = [0.0] * nt
            acceleration[0], acceleration[1] = 2 / (0.5 * dt * dt), -2 / (0.5 * dt * dt)
            acceleration[3], acceleration[4] = -2 / (dt * 0.5), (0.5 - 1) / 0.5
            velocity = [0.0] * nt
            velocity[0], velocity[1] = 0.5 * dt * acceleration[0], 0.5 * dt * acceleration[1]
            velocity[3] = 1 + 0.5 * dt * acceleration[3]
            velocity[4] = dt * (1 - 0.5) + 0.5 * dt * acceleration[4]
            banks["Newmark2"][1], banks["Newmark2"][2] = velocity, acceleration
            require(new["weights"] == banks, "impulsive dt fill has inconsistent native weight banks")


def output_transition(before, after):
    """Permit the one proved bookkeeping increment; all other fields stay exact."""
    expected = copy.deepcopy(before)
    serialization = expected["policy"].get("serialization")
    require(isinstance(serialization, dict), "output serialization counter witness required")
    current = integer(serialization.get("output_step"), "native output index", minimum=0)
    serialization["output_step"] = current + 1
    require(after == expected, "native output changed physical state/history or unregistered bookkeeping")
    return current + 1


def capture(problem, hooks):
    witness = validate_witness(hooks.capture(problem))
    require(witness["time"] == problem.get_current_time(dimensional=False, as_float=True)
            and witness["counter"] == problem.timestepper.get_num_unsteady_steps_done(),
            "witness does not describe the actual native clock/counter")
    clock = problem.timestepper.time_pt()
    require(witness["dts"] == [float(clock.dt(i)) for i in range(clock.ndt())],
            "witness does not describe every actual stored dt")
    require(witness["parameters"] == {name: float(problem.get_global_parameter(name).value)
                                      for name in ("Ca_in", "Bo")}, "witness does not describe live forcing")
    require(len(witness["steppers"]) == problem.ntime_stepper()
            and [s["is_steady"] for s in witness["steppers"]]
            == [problem.time_stepper_pt(i).is_steady() for i in range(problem.ntime_stepper())],
            "witness omitted an actual native stepper")
    require(witness["flags"] == {name: getattr(problem, name) for name in study.FLAGS},
            "witness does not describe actual solve flags")
    require(witness["bc_cache"] == problem._last_bc_setting, "witness does not describe actual BC cache")
    require(witness["policy"]["all_parameters"] == {name: float(problem.get_global_parameter(name).value)
            for name in sorted(problem.get_global_parameter_names())}
            and witness["policy"]["default_scheme"] == problem.default_timestepping_scheme,
            "witness does not describe all live forcing/stepping controls")
    require(witness["pending_quality_remesh"] == pending_remesh_policy(problem, hooks),
            "witness does not describe the actual owned quality request")
    require(problem.get_bifurcation_tracking_mode() == ""
            and problem._bifurcation_reactivation_after_adaptation is None
            and problem.get_custom_assembler() is None and problem._get_solved_residual() == ""
            and problem._azimuthal_mode_param_m is not None
            and float(problem._azimuthal_mode_param_m.value) == 0.0,
            "active or pending bifurcation tracker refused")
    require(float(problem.get_scaling("temporal")) == 1.0, "temporal mapping would be applied twice")
    return witness


def restore_loaded_policy(problem, expected, hooks):
    """Restore only bound solve/steady metadata absent from the native stream."""
    loaded = capture(problem, hooks)
    require(physical(loaded, omit=("flags", "pending_quality_remesh"))
            == physical(expected, omit=("flags", "pending_quality_remesh")),
            "load changed actual physical histories/time/dts/counter")
    original_flags = [s["is_steady"] for s in loaded["steppers"]]
    for i, saved in enumerate(expected["steppers"]):
        stepper = problem.time_stepper_pt(i)
        if saved["is_steady"] != stepper.is_steady():
            stepper.make_steady() if saved["is_steady"] else stepper.undo_make_steady()
        require(physical(capture(problem, hooks), omit=("flags",))
                == physical(loaded, omit=("flags",)), "loaded stepper restoration changed physical history")
    for name, value in expected["flags"].items():
        setattr(problem, name, value)
    before_request = capture(problem, hooks)
    restore_pending_remesh(problem, expected["pending_quality_remesh"], hooks)
    restored = capture(problem, hooks)
    require(physical(before_request, omit=("pending_quality_remesh",))
            == physical(restored, omit=("pending_quality_remesh",)),
            "quality request restoration changed physical history")
    require(restored == expected, "bound native history/stepper policy restoration failed")
    return {"loaded_steady_flags": original_flags,
            "restored_steady_flags": [s["is_steady"] for s in restored["steppers"]]}


def activate_unsteady(problem, hooks):
    """Activate every actual steady stepper after the explicit positive-dt fill."""
    before = capture(problem, hooks)
    for i in range(problem.ntime_stepper()):
        stepper = problem.time_stepper_pt(i)
        if stepper.is_steady():
            stepper.undo_make_steady()
    after = capture(problem, hooks)
    require(physical(before) == physical(after), "stepper activation changed physical state/history")
    require(all(s["is_steady"] is False for s in after["steppers"]), "stepper activation failed")
    return {"before_steady_flags": [s["is_steady"] for s in before["steppers"]],
            "after_steady_flags": [s["is_steady"] for s in after["steppers"]]}


def prepare_transient(problem, hooks):
    before = capture(problem, hooks)
    problem.actions_before_transient_solve(force_reassign_eqs=True)
    after = capture(problem, hooks)
    require(physical(before, omit=("bc_cache",)) == physical(after, omit=("bc_cache",))
            and after["bc_cache"] == "transient", "transient hook changed fields/pins/history/layout")


def start_jump(problem, parent, target, role, controls, hooks):
    """Load the unchanged parent first; then create one physical jump origin."""
    require(problem.case == parent["case"], "construct the unchanged parent case first")
    expected = validate_witness(parent["witness"])
    require(expected["pending_quality_remesh"]["requested"] is False,
            "fold parent has a pending quality request; startup transfer is unqualified")
    with hooks.load_session(problem):
        problem.load_state(parent["state"]["path"], quiet=True,
                           additional_info={"exchange_msh_file": parent["mesh_exchange"]})
    restoration = restore_loaded_policy(problem, expected, hooks)
    loaded = capture(problem, hooks)
    require(loaded == expected, "faithful complete parent load failed")
    hooks.record("unchanged_parent_load", {"witness": loaded, "policy_restoration": restoration})
    problem.set_current_time(0.0, dimensional=False)
    clock_reset = capture(problem, hooks)
    require(physical(loaded, omit=("time",)) == physical(clock_reset, omit=("time",))
            and clock_reset["time"] == 0.0, "fresh time-origin change altered physical state")
    clock_transition(loaded, clock_reset)
    dt = number(role["dt_solver"], "explicit jump dt", positive=True)
    problem.assign_initial_values_impulsive(dt)
    filled = capture(problem, hooks)
    require(physical(clock_reset, current_only=True, omit=("dts",))
            == physical(filled, current_only=True, omit=("dts",))
            and all(value == dt for value in filled["dts"]), "impulsive fill overwrote current fold fields")
    clock_transition(clock_reset, filled, dt=dt)
    for block in filled["data"]:
        rows = block["rows"]
        require(rows[1] == rows[0] and rows[2] == rows[0]
                and all(v == 0 for row in rows[3:5] for v in row)
                and (len(rows) == 5 or all(v == 0 for v in rows[5]) and rows[6] == rows[0]),
                "stationary physical history or derivative/predictor fill failed")
    # The native impulsive fill cannot change the Python scheduling suggestion.
    # Bind this fresh jump to its already-registered fixed dt, including t0 saves.
    problem._suggested_next_dt = dt
    suggested = capture(problem, hooks)
    before, after = copy.deepcopy(filled), copy.deepcopy(suggested)
    for witness in (before, after):
        witness["policy"].get("serialization", {}).pop("suggested_next_dt", None)
    require(before == after, "fixed next-dt registration changed physical state/history")
    activation = activate_unsteady(problem, hooks)
    before_counter = capture(problem, hooks)
    problem.timestepper.set_num_unsteady_steps_done(0)
    after_counter = capture(problem, hooks)
    require(physical(before_counter, omit=("counter",)) == physical(after_counter, omit=("counter",))
            and after_counter["counter"] == 0, "fresh counter reset changed physical history")
    hooks.record("explicit_stationary_history_startup", {"witness": after_counter, "activation": activation})
    prepare_transient(problem, hooks)
    require(target == study.target_case(parent["case"], role["target_ca"], role["direction"]),
            "target equation case changed more than case ID/Ca")
    before = capture(problem, hooks)
    problem.get_global_parameter("Ca_in").value = role["target_ca"]
    problem.case = copy.deepcopy(target)
    problem.actions_after_change_in_global_parameter("Ca_in")
    problem.invalidate_eigendata()
    problem.invalidate_cached_mesh_data()
    after = capture(problem, hooks)
    old, new = physical(before), physical(after)
    old["parameters"]["Ca_in"] = new["parameters"]["Ca_in"]
    old["policy"]["all_parameters"]["Ca_in"] = new["policy"]["all_parameters"]["Ca_in"]
    require(old == new and new["parameters"] == {"Ca_in": role["target_ca"], "Bo": study.MODEL["bond"]}
            and problem.case == target, "live Ca adoption changed fields/pins/history/geometry")
    problem.do_call_remeshing_when_necessary = False
    problem.inversion_remesh_max_retries = controls["remesh"]["max_inversion_retries_per_step"]
    require(capture(problem, hooks) == after, "remesh control registration changed physical state")
    hooks.record("t0_geometry_prejump_fluid_fields", after)
    return after


def resume_jump(problem, resumed, role, controls, hooks):
    """Restore a dynamic state with no origin reset, jump, fill or BDF startup."""
    require(problem.case == resumed["target_case"], "construct the saved changed-control case")
    expected = validate_witness(resumed["witness"])
    require(expected["flags"] == DYNAMIC_FLAGS and expected["counter"] == resumed["accepted_step_index"]
            and all(s["is_steady"] is False for s in expected["steppers"])
            and expected["parameters"] == {"Ca_in": role["target_ca"], "Bo": study.MODEL["bond"]}
            and all(dt == role["dt_solver"] for dt in expected["dts"]), "dynamic stepping policy drifted")
    with hooks.load_session(problem):
        problem.load_state(resumed["state"]["path"], quiet=True,
                           additional_info={"exchange_msh_file": resumed["mesh_exchange"]})
    restoration = restore_loaded_policy(problem, expected, hooks)
    restored = capture(problem, hooks)
    problem.do_call_remeshing_when_necessary = False
    problem.inversion_remesh_max_retries = controls["remesh"]["max_inversion_retries_per_step"]
    require(capture(problem, hooks) == expected, "resume remesh policy changed physical state")
    hooks.record("dynamic_history_policy_restored", {"witness": restored, "policy_restoration": restoration})
    return restored


def validate_work(work, controls, thresholds, index):
    require(isinstance(work, dict) and set(work) == WORK_KEYS, "cumulative dynamic work ledger required")
    require(integer(work["solve_calls"], "solve calls", minimum=0) == index,
            "resume solve count differs from accepted-step ledger")
    updates = integer(work["newton_updates"], "completed Newton updates", minimum=0)
    step_updates = work["newton_step_updates"]
    require(isinstance(step_updates, list) and len(step_updates) == index,
            "completed updates for every physical step are required")
    for count in step_updates:
        require(integer(count, "completed updates in physical step", minimum=0)
                <= controls["max_newton_iterations_per_step"], "physical-step Newton-update budget exceeded")
    require(sum(step_updates) == updates, "cumulative Newton updates differ from the physical-step ledger")
    remeshes = integer(work["physical_remeshes"], "physical remeshes", minimum=0)
    require(updates <= controls["max_total_newton_updates"]
            and remeshes <= controls["remesh"]["max_per_run"], "finite cumulative work budget exhausted")
    retries = work["inversion_retries"]
    require(isinstance(retries, list) and len(retries) == index, "per-step inversion ledger required")
    for retry in retries:
        require(integer(retry, "inversion retries", minimum=0)
                <= controls["remesh"]["max_inversion_retries_per_step"], "inversion retry budget exceeded")
    transfers = work["transfer_ledger"]
    require(isinstance(transfers, list) and len(transfers) == remeshes, "all physical remesh transfers required")
    cumulative = 0.0
    for row in transfers:
        require(isinstance(row, dict) and row.get("field_qualification") == "interpolated_unqualified_before_next_step",
                "remesh transfer cannot qualify a trajectory sample")
        require(row.get("phase") in {"between_steps", "fixed_step"}, "unknown physical transfer phase")
        before, after = validate_witness(row.get("before")), validate_witness(row.get("after"))
        for key in ("time", "dts", "counter", "parameters", "flags", "steppers", "mode"):
            require(before[key] == after[key], "retained transfer changed time/history policy or forcing")
        require(0 <= before["counter"] <= index and row.get("native_time") == before["time"],
                "retained transfer is outside the physical-step ledger")
        first, second = row.get("before_metrics"), row.get("after_metrics")
        require(isinstance(first, dict) and isinstance(second, dict), "both remesh-side volume evaluations required")
        volume_metrics(first, thresholds)
        volume_metrics(second, thresholds)
        mapping_errors(before, after, row.get("mapping_errors"), thresholds)
        jump = number(row.get("volume_change_relative"), "absolute remesh volume change")
        require(jump == abs(second["gas_volume"] - first["gas_volume"]) / (4 * math.pi / 3)
                and jump <= thresholds["remesh_volume_relative"], "single remesh volume budget exceeded")
        cumulative += jump
    require(number(work["cumulative_transfer_relative"], "cumulative absolute transfer") == cumulative
            and cumulative <= thresholds["cumulative_remesh_volume_relative"],
            "cumulative absolute remesh transfer budget exceeded")
    return copy.deepcopy(work)


def empty_work():
    return {"newton_updates": 0, "solve_calls": 0, "physical_remeshes": 0, "inversion_retries": [],
            "newton_step_updates": [], "cumulative_transfer_relative": 0.0, "transfer_ledger": []}


class UpdateBudget:
    """A qualified native callback bridge must call these around every update."""

    def __init__(self, work, controls, deadline):
        self.work, self.controls, self.deadline = work, controls, deadline
        self.starts = self.started = self.completed = self.in_call = 0
        self.initial_checks = self.zero_update_solves = 0
        self.initial_residual_pending = self.finished = False
        self.solves_completed = 0
        self.solve_open = False

    def check(self):
        require(self.deadline.tzinfo is not None and datetime.now(UTC) < self.deadline,
                "finite dynamic deadline elapsed")

    def before_solve(self):
        self.check()
        require(not self.finished and not self.solve_open and self.started == self.completed,
                "interrupted Newton update requires qualified recovery")
        self.starts += 1
        self.in_call = 0
        require(self.starts <= 1 + self.controls["remesh"]["max_inversion_retries_per_step"],
                "native inversion/solve retry budget exceeded")
        self.solve_open = True

    def before_update(self):
        self.check()
        require(self.solve_open and self.started == self.completed, "missing native Newton callback")
        require(not self.finished and self.completed < self.controls["max_newton_iterations_per_step"]
                and self.work["newton_updates"] < self.controls["max_total_newton_updates"],
                "finite Newton-update budget exhausted")
        self.started += 1

    def before_initial_residual(self):
        """Reserve a possible first update; native initial convergence may cancel it."""
        require(self.in_call == 0 and not self.initial_residual_pending, "duplicate initial residual callback")
        self.before_update()
        self.initial_residual_pending = True
        self.initial_checks += 1

    def accept_initial_convergence(self):
        require(self.initial_residual_pending and self.in_call == 0
                and self.started == self.completed + 1, "unqualified zero-update native convergence")
        self.started -= 1
        self.initial_residual_pending = False
        self.zero_update_solves += 1

    def after_update(self):
        require(self.solve_open and self.started == self.completed + 1,
                "unmatched native Newton update callback")
        self.completed += 1
        self.in_call += 1
        self.initial_residual_pending = False
        self.work["newton_updates"] += 1

    def after_solve(self):
        require(self.solve_open and self.started == self.completed, "incomplete native Newton solve callback")
        self.solve_open = False
        self.solves_completed += 1

    def finish(self):
        require(not self.finished and not self.solve_open and self.starts >= 1
                and self.solves_completed == self.starts and self.started == self.completed,
                "incomplete native Newton callback ledger")
        self.work["inversion_retries"].append(self.starts - 1)
        self.work["newton_step_updates"].append(self.completed)
        self.finished = True

    def callback_receipt(self):
        return {"solve_attempts": self.starts, "completed_updates": self.completed,
                "completed_solve_callbacks": self.solves_completed,
                "initial_residual_checks": self.initial_checks, "zero_update_solves": self.zero_update_solves,
                "unmatched_possible_update": self.started != self.completed,
                "counts_qualified_for_resume": self.started == self.completed and not self.solve_open
                and self.solves_completed == self.starts}


class NewtonCallbackBridge:
    """Wrap the pinned ordinary-Newton callbacks without importing native code.

    Embedded problem.cc:9235 calls the before-step hook before the initial
    residual; 9313-9317 can then finish with no update. A completed update
    reaches the after-step hook at 9436, before the next residual check.
    Native failures before that callback have uncertain physical update
    completion and remain inadmissible for retries or resume.
    """

    CALLBACKS = ("actions_before_newton_solve", "actions_before_newton_step",
                 "actions_after_newton_step", "actions_after_newton_solve")

    def __init__(self, problem, budget, record):
        self.problem, self.budget, self.record = problem, budget, record
        self.solve_open = self.iteration_open = False
        self.iterations = 0

    @contextlib.contextmanager
    def installed(self):
        original = {name: getattr(self.problem, name) for name in self.CALLBACKS}

        def before_solve():
            require(not self.solve_open, "interrupted native solve requires qualified recovery")
            self.budget.before_solve()
            self.solve_open, self.iterations = True, 0
            return original["actions_before_newton_solve"]()

        def before_step():
            require(self.solve_open and not self.iteration_open, "unmatched before-Newton-step callback")
            if self.iterations == 0:
                self.budget.before_initial_residual()
            else:
                self.budget.before_update()
            self.iteration_open = True
            self.iterations += 1
            return original["actions_before_newton_step"]()

        def after_step():
            require(self.solve_open and self.iteration_open, "unmatched after-Newton-step callback")
            # Account first: an original after-step hook may itself fail after
            # the native DOF update has already completed.
            self.budget.after_update()
            self.iteration_open = False
            return original["actions_after_newton_step"]()

        def after_solve():
            require(self.solve_open and self.iterations > 0, "unmatched after-Newton-solve callback")
            if self.iteration_open:
                require(self.iterations == 1, "native convergence has an unmatched later update")
                self.budget.accept_initial_convergence()
                self.iteration_open = False
            result = original["actions_after_newton_solve"]()
            self.budget.after_solve()
            self.solve_open = False
            self.record("native_newton_call_completed", self.budget.callback_receipt())
            return result

        wrapped = dict(zip(self.CALLBACKS, (before_solve, before_step, after_step, after_solve), strict=True))
        for name, callback in wrapped.items():
            setattr(self.problem, name, callback)
        try:
            yield
            require(not self.solve_open and not self.iteration_open, "native solve did not complete its callbacks")
        except BaseException:
            self.record("interrupted_native_newton_call", {**self.budget.callback_receipt(),
                         "solve_completed": not self.solve_open, "native_recovery_qualified": False,
                         "cumulative_work": copy.deepcopy(self.budget.work)})
            raise
        finally:
            unchanged = all(getattr(self.problem, name) is callback for name, callback in wrapped.items())
            for name, callback in original.items():
                setattr(self.problem, name, callback)
            require(unchanged, "native Newton callback inventory changed")


def volume_metrics(metrics, thresholds):
    v0 = 4 * math.pi / 3  # R0=1; the axisymmetric measure is already in each observable.
    gas = number(metrics.get("gas_volume"), "native gas volume", positive=True)
    surface = number(metrics.get("surface_volume"), "independent surface volume", positive=True)
    require(abs(gas / v0 - 1) <= thresholds["volume_relative"]
            and abs(surface - gas) / v0 <= thresholds["surface_volume_relative"],
            "conserved volume or independent surface-volume gate failed")
    require(metrics.get("inverted_elements") is False and metrics.get("interface_conformity") is True,
            "inverted/nonconforming native geometry refused")
    return gas


def accepted_metrics(metrics, role, thresholds):
    volume_metrics(metrics, thresholds)
    number(metrics.get("meridional_L_over_R"), "curved meridional L/R", positive=True)
    number(metrics.get("Ca_b"), "frame speed Ca_b", signed=True)
    number(metrics.get("volume_multiplier"), "volume multiplier", signed=True)
    number(metrics.get("gas_pressure"), "gas pressure diagnostic", signed=True)
    residual = number(metrics.get("transient_newton_residual"), "actual BDF residual")
    require(residual <= thresholds["transient_newton_residual"], "transient BDF residual budget exceeded")
    expected = math.pi * study.RATIO**2 * role["target_ca"]
    upper = number(metrics.get("upper_lab_axial_flux"), "upper lab-frame axial flux", signed=True)
    lower = number(metrics.get("lower_lab_axial_flux"), "lower lab-frame axial flux", signed=True)
    require(abs(upper - expected) <= thresholds["flux_absolute"]
            and abs(upper - lower) <= thresholds["flux_absolute"], "actual changed forcing/port flux gate failed")
    pressure = number(metrics.get("lower_static_pressure_residual"), "static pressure port residual")
    require(pressure <= thresholds["static_port_pressure_absolute"], "lower static pressure port gate failed")


def mapping_errors(before, after, errors, thresholds):
    require(isinstance(errors, dict), "independently measured physical-history mapping errors required")
    expected_fields = {field for block in before["data"] for field in block["fields"]}
    after_fields = {field for block in after["data"] for field in block["fields"]}
    expected_slots = {}
    for block in before["data"]:
        for field in block["fields"]:
            expected_slots.setdefault(field, set()).update(range(len(block["rows"])))
    expected_slots = {field: sorted(slots) for field, slots in expected_slots.items()}
    require(errors.get("mapped_fields") == sorted(expected_fields) == sorted(after_fields)
            and errors.get("mapped_history_slots") == expected_slots
            and errors.get("topology_unchanged") is True
            and errors.get("interface_conformity") is True,
            "complete named field/position/internal/ODE history transfer mapping required")
    require(number(errors.get("geometry_norm"), "measured geometry transfer norm")
            <= thresholds["remesh_geometry_profile_norm"]
            and number(errors.get("history_norm"), "measured physical-history transfer norm")
            <= thresholds["remesh_history_profile_norm"], "remesh mapping norm budget exceeded")


@dataclass(frozen=True)
class RuntimeHooks:
    capture: Callable
    load_session: Callable
    update_session: Callable
    diagnostics: Callable
    transfer_errors: Callable
    record: Callable
    retain: Callable
    template_type: type
    remesher_type: type


class RemeshGuard:
    """Matched quality transfers; load rebuilds are administrative.

    A failed native step needs a qualified restoration/transfer adapter. Its
    callback is refused here rather than mistaken for a quality transfer.
    """

    def __init__(self, problem, hooks, work, thresholds, controls):
        self.problem, self.hooks, self.work = problem, hooks, work
        self.thresholds, self.controls = thresholds, controls
        self.phase, self.pending = None, None

    @contextlib.contextmanager
    def installed(self):
        """Wrap the actual Python callbacks; restore their exact bindings on exit."""
        problem = self.problem
        before, after = problem.actions_before_remeshing, problem.actions_after_remeshing

        def guarded_before(active_remeshers):
            self.before(active_remeshers)
            return before(active_remeshers)

        def guarded_after():
            result = after()
            self.after()
            return result

        problem.actions_before_remeshing = guarded_before
        problem.actions_after_remeshing = guarded_after
        try:
            yield
        finally:
            unchanged = problem.actions_before_remeshing is guarded_before and problem.actions_after_remeshing is guarded_after
            problem.actions_before_remeshing, problem.actions_after_remeshing = before, after
            require(unchanged, "native remesh callback inventory changed")

    @contextlib.contextmanager
    def scope(self, phase):
        require(self.phase is None and self.pending is None, "nested/unmatched remesh scope")
        self.phase = phase
        try:
            yield
            require(self.pending is None, "physical remesh has no matched after callback")
        finally:
            self.phase = None

    def owned(self):
        # Reuse the same read-only proof as capture and cold restoration.
        _, active = owned_quality_templates(self.problem, self.hooks)
        require(len(self.problem._domains_to_remesh) == 1, "missing owned pending quality request")
        return active.remesher

    def before(self, active_remeshers):
        require(self.phase in {"between_steps", "fixed_step"} and self.pending is None,
                "physical remesh outside an admitted step")
        require(self.phase == "between_steps", "native inversion recovery transfer adapter is unqualified")
        owner = self.owned()
        require(isinstance(active_remeshers, list) and len(active_remeshers) == 1
                and active_remeshers[0] is owner, "foreign/additional active remesher")
        require(self.work["physical_remeshes"] < self.controls["remesh"]["max_per_run"],
                "finite physical remesh budget exhausted")
        witness = capture(self.problem, self.hooks)
        metrics = self.hooks.diagnostics(self.problem)
        volume_metrics(metrics, self.thresholds)
        self.pending = (witness, metrics)
        self.hooks.record("before_physical_remesh", {"phase": self.phase, "witness": witness, "metrics": metrics})

    def after(self):
        if self.phase == "administrative_load":
            require(self.pending is None, "load/rebuild mixed with physical remesh")
            self.hooks.record("administrative_load_remesh_callback", {})
            return
        require(self.pending is not None, "unmatched physical after-remesh callback")
        from runners.run_paper_finite_gas_fold import inspect_inversion_templates

        inspect_inversion_templates(self.problem, self.hooks.template_type, self.hooks.remesher_type)
        before, first = self.pending
        after = capture(self.problem, self.hooks)
        second = self.hooks.diagnostics(self.problem)
        volume_metrics(second, self.thresholds)
        for key in ("time", "dts", "counter", "parameters", "flags", "steppers", "mode"):
            require(before[key] == after[key], "remesh changed physical time/history policy or forcing")
        errors = self.hooks.transfer_errors(before, after)
        mapping_errors(before, after, errors, self.thresholds)
        jump = abs(second["gas_volume"] - first["gas_volume"]) / (4 * math.pi / 3)
        require(jump <= self.thresholds["remesh_volume_relative"]
                and self.work["cumulative_transfer_relative"] + jump
                <= self.thresholds["cumulative_remesh_volume_relative"], "absolute remesh volume transfer budget exceeded")
        row = {"phase": self.phase, "native_time": before["time"], "before": before, "after": after,
               "before_metrics": first, "after_metrics": second, "mapping_errors": errors,
               "volume_change_relative": jump, "field_qualification": "interpolated_unqualified_before_next_step"}
        self.work["physical_remeshes"] += 1
        self.work["cumulative_transfer_relative"] += jump
        self.work["transfer_ledger"].append(row)
        self.pending = None
        self.hooks.record("unqualified_physical_remesh_transfer", row)


def fixed_steps(problem, role, controls, thresholds, hooks, deadline, *,
                accepted_step_index=0, work=None, retained_rows=None, remesh_guard=None):
    """Take the original finite native grid; retain each converged row before transfer.

    Endpoint and pair qualification remain separate operations. The last row is
    never followed by a trajectory remesh or stationary polish.
    """
    require(controls["startup"] == study.STARTUP, "unregistered startup policy")
    integer(accepted_step_index, "accepted-step index", minimum=0)
    require(accepted_step_index <= role["max_steps"], "horizon extension refused")
    work = validate_work(empty_work() if work is None else work, controls, thresholds, accepted_step_index)
    rows = [] if retained_rows is None else copy.deepcopy(retained_rows)
    require(len(rows) == accepted_step_index + 1 if accepted_step_index else not rows,
            "resume must retain the original native rows")
    clock = 0.0
    output_origin = integer(rows[0].get("next_output_index"), "retained output origin", minimum=0) if rows else None
    for index, row in enumerate(rows):
        require(row.get("accepted_step_index") == index and row.get("native_time") == clock
                and row.get("next_output_index") == output_origin + index,
                "retained native row order/clock drifted")
        study.retained_row(row, role, thresholds, study.Artifacts())
        clock += role["dt_solver"]
    require(problem.do_call_remeshing_when_necessary is False
            and problem.inversion_remesh_max_retries == controls["remesh"]["max_inversion_retries_per_step"],
            "registered remesh controls drifted")
    if remesh_guard is not None:
        require(remesh_guard.problem is problem and remesh_guard.hooks is hooks and remesh_guard.work == work,
                "remesh work ledger must contain the restored cumulative counts")
        remesh_guard.work = work
    if accepted_step_index:
        restored = capture(problem, hooks)
        require(restored["time"] == rows[-1]["native_time"]
                and restored["counter"] == accepted_step_index
                and restored["policy"]["serialization"]["output_step"] == rows[-1]["next_output_index"],
                "resume does not match its last retained row/output index")
    session = remesh_guard.installed() if remesh_guard is not None else deny_physical_remesh(problem)
    with session:
        return _advance_steps(problem, role, controls, thresholds, hooks, deadline,
                              accepted_step_index, work, rows, remesh_guard)


@contextlib.contextmanager
def deny_physical_remesh(problem):
    """Refuse before physical mutation when no transfer adapter was provided."""
    before, after = problem.actions_before_remeshing, problem.actions_after_remeshing

    def denied(*args, **kwargs):
        raise ValueError("native history-transfer adapter is unqualified")

    problem.actions_before_remeshing = problem.actions_after_remeshing = denied
    try:
        yield
    finally:
        unchanged = problem.actions_before_remeshing is denied and problem.actions_after_remeshing is denied
        problem.actions_before_remeshing, problem.actions_after_remeshing = before, after
        require(unchanged, "native remesh callback inventory changed")


def _advance_steps(problem, role, controls, thresholds, hooks, deadline, accepted_step_index,
                   work, rows, remesh_guard):
    target = copy.deepcopy(problem.case)
    if accepted_step_index == 0:
        initial = capture(problem, hooks)
        require(initial["counter"] == 0 and initial["time"] == 0.0, "fresh jump origin drifted")
        row = {"accepted_step_index": 0, "t_hat": 0, "native_time": 0.0, "counter": 0,
               "dt_solver": role["dt_solver"], "field_qualification": "stationary_geometry_prejump_fluid_fields",
               "metrics": hooks.diagnostics(problem)}
        row["next_output_index"] = initial["policy"]["serialization"]["output_step"]
        retained = hooks.retain(problem, row, initial)
        require(isinstance(retained, dict) and all(retained.get(k) == v for k, v in row.items())
                and capture(problem, hooks) == initial, "retaining the initial row changed its qualification or state")
        require(study.retained_row(retained, role, thresholds, study.Artifacts()) == initial,
                "retained initial physical history differs")
        rows.append(retained)
    for index in range(accepted_step_index + 1, role["max_steps"] + 1):
        budget = UpdateBudget(work, controls, deadline)
        budget.check()
        require(work["newton_updates"] < controls["max_total_newton_updates"], "finite Newton-update budget exhausted")
        if index > 1 and problem.remeshing_necessary():
            require(remesh_guard is not None, "native history-transfer adapter is unqualified")
            with remesh_guard.scope("between_steps"):
                problem.remesh_if_necessary()
        before = capture(problem, hooks)
        require(before["counter"] == index - 1 and all(s["is_steady"] is False for s in before["steppers"])
                and before["parameters"] == {"Ca_in": role["target_ca"], "Bo": study.MODEL["bond"]},
                "native counter/stepper/forcing policy drifted")
        if index > 1:
            require(before["flags"] == DYNAMIC_FLAGS, "resume/step would repeat BDF startup")
        hooks.record("failure_recovery_state", {"next_step": index, "witness": before})
        expected_clock = before["time"] + role["dt_solver"]
        scope = remesh_guard.scope("fixed_step") if remesh_guard is not None else contextlib.nullcontext()
        with scope, hooks.update_session(problem, budget):
            problem.solve(timestep=role["dt_solver"], temporal_error=None, spatial_adapt=0,
                          shift_values=True, do_not_set_IC=True,
                          max_newton_iterations=controls["max_newton_iterations_per_step"],
                          newton_solver_tolerance=thresholds["transient_newton_residual"])
        budget.finish()
        after = capture(problem, hooks)
        require(after["time"] == expected_clock and all(dt == role["dt_solver"] for dt in after["dts"])
                and after["counter"] == index and after["flags"] == DYNAMIC_FLAGS
                and after["parameters"] == {"Ca_in": role["target_ca"], "Bo": study.MODEL["bond"]}
                and all(s["is_steady"] is False for s in after["steppers"])
                and problem.case == target,
                "native fixed clock/dt/counter or BDF policy drifted")
        require(problem.do_call_remeshing_when_necessary is False
                and problem.inversion_remesh_max_retries == controls["remesh"]["max_inversion_retries_per_step"],
                "automatic postsolve quality remesh or inversion policy changed")
        metrics = hooks.diagnostics(problem)
        accepted_metrics(metrics, role, thresholds)
        work["solve_calls"] += 1
        validate_work(work, controls, thresholds, index)
        row = {"accepted_step_index": index, "t_hat": index * role["dt_hat"],
               "native_time": after["time"], "dt_solver": role["dt_solver"], "counter": index,
               "field_qualification": "converged_native_transient", "metrics": metrics}
        # retain() owns the portable state/profile and complete mesh bindings.
        # Neither it nor output() may evolve or flatten the physical histories.
        # Save the accepted row AFTER the permitted output bookkeeping step.
        # A cold load then starts at the same next output index as a warm run.
        problem.output()
        written = capture(problem, hooks)
        row["next_output_index"] = output_transition(after, written)
        retained = hooks.retain(problem, row, written)
        require(isinstance(retained, dict) and all(retained.get(k) == v for k, v in row.items())
                and capture(problem, hooks) == written, "retaining the converged row changed qualification or physical state")
        require(study.retained_row(retained, role, thresholds, study.Artifacts()) == written,
                "retained accepted physical history differs")
        rows.append(retained)
    return {"status": "horizon_reached_endpoint_unqualified", "numerical_acceptance": False,
            "endpoint_qualified": False, "pair_qualified": False, "rows": rows, "work": work,
            "terminal_witness": capture(problem, hooks)}
