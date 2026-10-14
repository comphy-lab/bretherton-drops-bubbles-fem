"""Caller-bound marker authority and isolated native type ownership.

No native import occurs here. Descriptor fixtures can test these Python
contracts only; their nanobind-shaped names do not establish C++ dispatch.
Full warm/cold execution remains refused. The marker source interface grants
no native authority; the official caller binds actual core and source bytes.
"""
from __future__ import annotations

import inspect
import json
import math
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FunctionType, MappingProxyType, MethodType

PYOOMPH_PIN = "a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618"
SCHEMA = "stone-before-step-cpp-marker-v1"
HOOK = "actions_before_newton_step"
DRIVER_INTERFACE_SUPPORTED = False
MARKER_DRIVER_INTERFACE_SUPPORTED = True
MARKER_CAPS = MappingProxyType({
    "problem_constructions": 1, "constructor_compiler_availability_passes": 1,
    "fem_equation_compilations": 0, "cpp_virtual_marker_calls": 1,
    "initialisation_calls": 0, "load_calls": 0, "dof_array_accesses": 0,
    "capture_calls": 0, "state_output_calls": 0, "matrix_assemblies": 0,
    "factorisations": 0, "time_step_calls": 0, "newton_updates": 0,
    "stationary_solves": 0, "eigensolves": 0, "continuation_steps": 0,
    "remeshes": 0, "refinements": 0, "inversion_retries": 0, "automatic_retries": 0,
})
MARKER_CONTROLS = MappingProxyType({
    "schema": SCHEMA + "-controls", "selected_compiler": "system",
    "linear_solver_default": "superlu", "constructor_compiler_availability_passes": 1,
    "empty_constructor_metadata_permitted": True, "fem_equation_compilation_permitted": False,
    "private_compiler_default_override_permitted": False,
    "historical_time_step_charge_spent": 1, "new_time_step_allowance": 0,
})
MARKER_POLICY = MappingProxyType({
    "marker_only": True, "positive_ndof_required": False, "history_fill": False,
    "clock_reset": False, "forcing_change": False, "production_jump": False,
    "synthetic_buffer_stress": False, "exact_next_transition": False,
    "equation_initialisation": False, "restart_or_state_read": False,
    "readiness_transferable": False,
})
_AUTHORITY_TOKEN = object()
_READY_TOKEN = object()
_CLASS_OWNERS = {}
_OWNER_LOCK = threading.RLock()
_MISSING = object()


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    """Preserve JSON types: true/1/1.0 are three different authorities."""
    def check(item):
        kind = type(item)
        if kind is dict:
            require(all(type(key) is str for key in item), "authority keys must be strings")
            for child in item.values():
                check(child)
        elif kind is list:
            for child in item:
                check(child)
        else:
            require(item is None or kind in (str, int, bool, float), "plain JSON authority required")
            require(kind is not float or math.isfinite(item), "finite authority required")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def same(left, right):
    return canonical(left) == canonical(right)


def exact_values(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), "exact " + label + " keys required")
    require(all(type(value[key]) is type(expected[key]) and value[key] == expected[key]
                for key in expected), "exact typed " + label + " values required")
    return json.loads(canonical(value))


def caps_contract(value):
    return exact_values(value, MARKER_CAPS, "marker caps")


def controls_contract(value):
    return exact_values(value, MARKER_CONTROLS, "marker controls")


def utc(value):
    require(type(value) is str, "explicit UTC timestamp required")
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "UTC timestamp required")
    return result


def reference_contract(value):
    require(type(value) is dict and set(value) == {"path", "sha256"}
            and type(value["path"]) is str and Path(value["path"]).is_absolute()
            and type(value["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", value["sha256"]),
            "exact bound byte reference required")
    return value


def process_contract(value):
    numbers = {"pid", "uid", "start_time_ticks", "process_group_id", "session_id"}
    strings = {"host", "cwd"}
    require(type(value) is dict and set(value) == numbers | strings | {"cgroup", "argv"},
            "exact actual process identity required")
    require(all(type(value[key]) is int and value[key] >= 0 for key in numbers)
            and value["pid"] > 0 and value["start_time_ticks"] > 0,
            "canonical process identity integers required")
    require(all(type(value[key]) is str and value[key] for key in strings)
            and Path(value["cwd"]).is_absolute(), "actual host/cwd required")
    require(type(value["argv"]) is list and value["argv"]
            and all(type(v) is str and v for v in value["argv"])
            and type(value["cgroup"]) is list
            and all(type(v) is str for v in value["cgroup"]), "actual argv/cgroup required")
    return value


def authority_frame_contract(frame, now):
    require(type(frame) is dict and set(frame) == {"requirements", "phase", "phase_ref",
            "admission", "admission_ref", "native_process"}, "complete admitted marker frame required")
    req, phase, admission = frame["requirements"], frame["phase"], frame["admission"]
    require(type(req) is dict and req.get("schema") == SCHEMA + "-requirements"
            and req.get("role") == "marker" and req.get("source_pin") == PYOOMPH_PIN,
            "exact marker requirements required")
    require(set(req) == {"schema", "role", "source_pin", "bindings", "source", "environment", "fresh_environment",
            "controls", "controls_ref", "caps", "case", "case_ref", "output", "warm_result", "inputs", "policy",
            "inherited_time_step_charge_spent", "new_time_step_allowance", "callback_protocol"},
            "complete closed marker requirement keys required")
    caps_contract(req["caps"])
    controls_contract(req["controls"])
    require(type(req.get("inherited_time_step_charge_spent")) is int
            and req["inherited_time_step_charge_spent"] == 1
            and type(req.get("new_time_step_allowance")) is int and req["new_time_step_allowance"] == 0,
            "spent historical charge and zero new timestep allowance required")
    source = req.get("source")
    require(type(source) is dict and type(source.get("commit")) is str
            and re.fullmatch("[0-9a-f]{40}", source["commit"])
            and type(source.get("root")) is str and Path(source["root"]).is_absolute()
            and type(source.get("files")) is dict and source["files"], "bound published source required")
    require(type(req.get("output")) is str and Path(req["output"]).is_absolute(), "bound marker output required")
    exact_values(req["policy"], MARKER_POLICY, "marker policy")
    exact_values(req["callback_protocol"], {"schema": SCHEMA, "type_forwarding_before_construction": True,
                 "same_instance_dispatch_only": True, "marker_interface_supported": True,
                 "full_fidelity_interface_supported": False}, "marker callback protocol")
    require(req["warm_result"] is None and type(req["inputs"]) is list, "no warm/state dependency permitted")
    for ref in [req[key] for key in ("bindings", "fresh_environment", "controls_ref", "case_ref")] + req["inputs"]:
        reference_contract(ref)
    for ref in source["files"].values():
        reference_contract(ref)
    require(type(req["environment"]) is dict and req["environment"].get("available") is True
            and req["environment"].get("commit") == PYOOMPH_PIN, "exact pinned available environment required")
    reference_contract(req["environment"]["native_core"])
    case = req["case"]
    require(type(case) is dict and type(case.get("physics")) is dict, "exact unchanged scalar case required")
    physical = case["physics"]
    require(physical.get("problem") == "finite_gas_bubble" and all(type(physical.get(key)) in (int, float)
            and physical[key] == expected for key, expected in {"ca_in": -.002, "bond": 1.56,
                "vhat": 3.3, "domain_over_r": 15}.items())
            and same(case.get("pressure"), {"field": "static", "reference": "lower_static_zero",
                "body_force": "explicit_both_phases"}), "unchanged I scalar formulation required")
    require(type(phase) is dict and set(phase) == {"schema", "requirements", "issued_by", "declared_at",
            "deadline_utc", "phase_authorized"} and phase["schema"] == SCHEMA + "-phase"
            and phase["issued_by"] == "root" and phase["phase_authorized"] is True
            and same(phase["requirements"], req), "exact root-authorised marker phase required")
    reference_contract(frame["phase_ref"])
    reference_contract(frame["admission_ref"])
    require(type(admission) is dict and set(admission) == {"schema", "native_execution_authorized",
            "marker_authorized", "phase", "requirements", "active_tracker", "role", "deadline_utc"}
            and admission["schema"] == SCHEMA + "-admission"
            and admission["native_execution_authorized"] is True and admission["marker_authorized"] is True
            and admission["role"] == "marker" and same(admission["phase"], frame["phase_ref"])
            and same(admission["requirements"], req)
            and admission["deadline_utc"] == phase["deadline_utc"], "exact official active marker admission required")
    reference_contract(admission["active_tracker"])
    process_contract(frame["native_process"])
    require(Path(req["output"]) == Path(frame["native_process"]["cwd"]) / "runtime/fidelity",
            "marker output must be exact actual attempt/runtime child")
    require(isinstance(now, datetime) and now.tzinfo is not None
            and now.utcoffset().total_seconds() == 0, "current actual UTC clock required")
    start, deadline = utc(phase["declared_at"]), utc(phase["deadline_utc"])
    require(start <= now < deadline and 0 < (deadline - start).total_seconds() <= 2700,
            "finite root marker deadline expired or invalid")
    return frame


@dataclass(frozen=True)
class _AdmittedMarkerAuthority:
    encoded: bytes
    refresh: object
    clock: object
    token: object

    def snapshot(self):
        require(self.token is _AUTHORITY_TOKEN, "foreign marker authority")
        return json.loads(self.encoded)

    def require_current(self):
        require(self.token is _AUTHORITY_TOKEN and callable(self.refresh) and callable(self.clock),
                "validated caller marker authority required")
        frame = self.refresh()
        authority_frame_contract(frame, self.clock())
        require(canonical(frame) == self.encoded, "immutable admitted marker binding changed")
        return json.loads(self.encoded)


def bind_admitted_marker_authority(frame, refresh, *, clock=None):
    """Seal only after the official caller has checked live admission and bytes.

    refresh must repeat that full check and return the same complete frame.
    This trusted-call-path boundary is not an adversarial Python sandbox.
    """
    require(callable(refresh), "complete official admission revalidation required")
    clock = (lambda: datetime.now(UTC)) if clock is None else clock
    require(callable(clock), "current clock required")
    authority_frame_contract(frame, clock())
    authority = _AdmittedMarkerAuthority(canonical(frame), refresh, clock, _AUTHORITY_TOKEN)
    authority.require_current()
    return authority


def require_authority(value):
    require(type(value) is _AdmittedMarkerAuthority and value.token is _AUTHORITY_TOKEN,
            "immutable validated caller marker authority required")
    return value.require_current()


class DurableMarkerCharges:
    def __init__(self, output, authority, write_new, read_bound):
        frame = require_authority(authority)
        self.output = Path(output)
        require(str(self.output) == frame["requirements"]["output"] and callable(write_new) and callable(read_bound),
                "charges must use exact admitted output and durable byte primitives")
        self.authority, self.write_new, self.read_bound = authority, write_new, read_bound
        self.attempted = set()
        self.references = {}
        self._reference_snapshots, self._value_snapshots = {}, {}

    def charge(self, name, local):
        require(name in {"construction", "marker"} and name not in self.attempted,
                "one-shot marker action already spent; no retry")
        frame = require_authority(self.authority)
        self.attempted.add(name)  # Even a failed receipt write cannot be retried.
        key = "problem_constructions_charged" if name == "construction" else "cpp_virtual_marker_calls_charged"
        value = {"schema": SCHEMA + "-" + name + "-charge", "role": "marker", key: 1,
                 "authority": frame, "local_identity": local, "retry_permitted": False,
                 "new_time_step_allowance": 0, "inherited_time_step_charge_spent": 1}
        target = self.output / (name + "-charge.json")
        ref = self.write_new(target, value)
        reference_contract(ref)
        require(ref["path"] == str(target), "charge must be retained at its exact admitted action path")
        _path, raw, actual = self.read_bound(ref)
        require(same(actual, ref) and same(json.loads(raw), value), "durable exclusive charge bytes differ")
        self.references[name] = ref
        self._reference_snapshots[name], self._value_snapshots[name] = canonical(ref), canonical(value)
        require_authority(self.authority)  # No native entry after an expired/changed write.
        return ref

    def stable(self):
        frame = require_authority(self.authority)
        for name, ref in self.references.items():
            require(canonical(ref) == self._reference_snapshots[name]
                    and ref["path"] == str(self.output / (name + "-charge.json")),
                    "immutable retained action charge reference changed")
            _path, raw, actual = self.read_bound(ref)
            value = json.loads(raw)
            require(same(actual, ref) and value.get("schema") == SCHEMA + "-" + name + "-charge"
                    and same(value.get("authority"), frame) and canonical(value) == self._value_snapshots[name],
                    "retained marker charge bytes/authority/local identity changed")


class BeforeStepTypeForwarding:
    """Exclude overlapping owners; retain uncertain native lifetime to process exit.

    Construction invokes the declared class directly, never an arbitrary
    factory returning an unproven preexisting object. The caller binds this
    exact class to the source-owned PaperBubbleProblem before native entry.
    """
    def __init__(self, problem_type, authority, charges, record):
        require(isinstance(problem_type, type) and callable(record), "declared Problem class/recorder required")
        require_authority(authority)
        self.problem_type, self.authority, self.charges, self.record = problem_type, authority, charges, record
        self.live, self.released, self.release_completed = {}, set(), set()
        self.construction_attempted = self.marker_attempted = False
        self.raw_marker_deliveries = 0
        self.normal_descriptor_return = False
        self.installed = self.scope_closed = False
        self.original_own = vars(problem_type).get(HOOK, _MISSING)
        self.parent_default = self._parent_descriptor()
        self.inherited_default = inspect.getattr_static(problem_type, HOOK)
        require(callable(self.inherited_default), "native before-step default is absent")

        def python_forwarder(problem):
            self.record("unexpected_before_step_forwarder", {"instance_id": id(problem)})
            raise ValueError("marker-only role refuses before-step dispatch outside its charged instance marker")
        self.shim = python_forwarder

    def _parent_descriptor(self):
        for parent in self.problem_type.__mro__[1:]:
            if HOOK in vars(parent):
                return vars(parent)[HOOK]
        return _MISSING

    def __enter__(self):
        require_authority(self.authority)
        with _OWNER_LOCK:
            require(not self.installed and not self.scope_closed and not self.construction_attempted
                    and self.problem_type not in _CLASS_OWNERS, "exact class already owned or owner reused")
            require(vars(self.problem_type).get(HOOK, _MISSING) is self.original_own
                    and self._parent_descriptor() is self.parent_default
                    and inspect.getattr_static(self.problem_type, HOOK) is self.inherited_default,
                    "foreign before-step descriptor changed before installation")
            _CLASS_OWNERS[self.problem_type] = self
            try:
                setattr(self.problem_type, HOOK, self.shim)
            except BaseException:
                del _CLASS_OWNERS[self.problem_type]
                raise
            self.installed = True
        return self

    def require_live_owner(self):
        require(self.installed and not self.scope_closed and _CLASS_OWNERS.get(self.problem_type) is self
                and vars(self.problem_type).get(HOOK, _MISSING) is self.shim,
                "live exclusive class owner/shim required")

    def construct_case(self, case):
        self.require_live_owner()
        frame = require_authority(self.authority)
        require(not self.construction_attempted and not self.live
                and same(case, frame["requirements"]["case"]), "one fresh exact case construction required")
        self.construction_attempted = True
        self.charges.charge("construction", {"owner_id": id(self), "problem_type_id": id(self.problem_type)})
        self.require_live_owner()
        self.charges.stable()
        require_authority(self.authority)
        try:
            problem = self.problem_type(case, initial_bond=case["physics"]["bond"])
            self.live[id(problem)] = problem
            require(type(problem) is self.problem_type, "declared constructor returned a foreign type")
            self.record("problem_constructed", {"owner_id": id(self), "problem_type_id": id(self.problem_type),
                        "instance_id": id(problem), "arbitrary_factory_freshness_claim": False})
            require_authority(self.authority)
            return problem
        except BaseException:
            self.record("construction_lifetime_unresolved", {"registered_instances": len(self.live),
                        "shim_retained_until_process_exit": True, "native_destruction_proved": False})
            raise

    def release_instance(self, problem):
        require(self.installed and _CLASS_OWNERS.get(self.problem_type) is self
                and self.live.get(id(problem)) is problem and type(problem) is self.problem_type,
                "release only the exact registered native instance")
        require(id(problem) not in self.released, "explicit release was already attempted")
        self.released.add(id(problem))
        record_error = None
        try:
            self.record("problem_release_entered", {"instance_id": id(problem)})
        except BaseException as error:  # noqa: BLE001 - preserve interruption/IO failure while still releasing
            record_error = error
        problem.release()
        self.release_completed.add(id(problem))
        try:
            self.record("problem_release_returned", {"instance_id": id(problem), "release_returned": True,
                        "native_destruction_proved": False, "shim_retained_until_process_exit": True})
        finally:
            if record_error is not None:
                raise record_error

    def _restore_unconstructed(self):
        with _OWNER_LOCK:
            require(not self.construction_attempted and not self.live and self.installed
                    and _CLASS_OWNERS.get(self.problem_type) is self
                    and self._parent_descriptor() is self.parent_default
                    and vars(self.problem_type).get(HOOK, _MISSING) is self.shim,
                    "foreign/live callback cannot be restored")
            # All refusal checks precede mutation.
            if self.original_own is _MISSING:
                delattr(self.problem_type, HOOK)
            else:
                setattr(self.problem_type, HOOK, self.original_own)
            del _CLASS_OWNERS[self.problem_type]
            self.installed = False

    def __exit__(self, exc_type, exc, traceback):
        self.scope_closed = True
        if self.construction_attempted:
            self.record("before_step_shim_retained", {"registered_instances": len(self.live),
                        "construction_attempted": True, "shim_retained_until_process_exit":
                            _CLASS_OWNERS.get(self.problem_type) is self and vars(self.problem_type).get(HOOK) is self.shim,
                        "native_destruction_proved": False})
        else:
            self._restore_unconstructed()
        return False


@dataclass(frozen=True)
class _CppMarkerReadiness:
    problem: object
    owner: BeforeStepTypeForwarding
    descriptor: object
    authority: _AdmittedMarkerAuthority
    charge: bytes
    token: object


def qualify_cpp_before_step_marker(problem, owner, cpp_problem_base, descriptor, record):
    owner.require_live_owner()
    require_authority(owner.authority)
    require(owner.live.get(id(problem)) is problem and type(problem) is owner.problem_type
            and id(problem) not in owner.released, "same fresh unreleased native instance required")
    kind = type(descriptor)
    require(isinstance(cpp_problem_base, type) and cpp_problem_base in type(problem).__mro__
            and vars(cpp_problem_base).get(HOOK) is descriptor
            and kind.__module__ == "nanobind" and kind.__name__ in {"nb_func", "nb_method", "nb_bound_method"}
            and not isinstance(descriptor, (FunctionType, MethodType)), "exact caller-bound native class descriptor required")
    require(not owner.marker_attempted, "marker allowance already spent; no retry")
    owner.marker_attempted = True
    local = {"owner_id": id(owner), "problem_type_id": id(type(problem)), "instance_id": id(problem),
             "descriptor_id": id(descriptor), "core_problem_type_id": id(cpp_problem_base)}
    charge = owner.charges.charge("marker", local)
    owner.charges.stable()
    had_own, original = HOOK in vars(problem), vars(problem).get(HOOK)
    observed = 0

    def marker():
        nonlocal observed
        observed += 1
        owner.raw_marker_deliveries = observed  # Preserve delivery before any fallible record write.
        record("before_step_cpp_marker_received", {**local, "raw_marker_deliveries": observed})

    setattr(problem, HOOK, marker)
    try:
        owner.require_live_owner()
        require_authority(owner.authority)
        owner.charges.stable()
        record("before_step_cpp_descriptor_entered", local)
        descriptor(problem)
        owner.normal_descriptor_return = True
        record("before_step_cpp_descriptor_returned", {**local, "normal_descriptor_return": True,
                    "raw_marker_deliveries": observed})
        require_authority(owner.authority)
        owner.charges.stable()
        require(observed == 1 and getattr(problem, HOOK) is marker,
                "actual descriptor must deliver exactly one unchanged instance marker")
        return _CppMarkerReadiness(problem, owner, descriptor, owner.authority, canonical(charge), _READY_TOKEN)
    finally:
        require(vars(problem).get(HOOK, _MISSING) is marker, "foreign instance callback changed; no restoration mutation")
        if had_own:
            setattr(problem, HOOK, original)
        else:
            delattr(problem, HOOK)


def require_dispatch_marker_readiness(problem, readiness):
    require(type(readiness) is _CppMarkerReadiness and readiness.token is _READY_TOKEN
            and readiness.problem is problem and readiness.owner.live.get(id(problem)) is problem
            and id(problem) not in readiness.owner.released
            and readiness.authority is readiness.owner.authority, "same-instance opaque dispatch readiness required")
    readiness.owner.require_live_owner()
    require_authority(readiness.authority)
    readiness.owner.charges.stable()
    require(canonical(readiness.owner.charges.references.get("marker")) == readiness.charge,
            "retained marker charge binding changed")
    # No ndof or solution-array operation belongs to marker-only readiness.
    return {"same_instance_dispatch": True, "full_newton_sequence_qualified": False,
            "positive_ndof_required": False, "compiled_header_provenance": "unestablished"}


def require_cpp_marker_readiness(problem, readiness):
    raise ValueError("full loaded warm/cold callback readiness remains refused pending separate wiring/review")
