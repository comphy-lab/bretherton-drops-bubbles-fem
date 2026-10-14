"""Native/scientific-denied controls; every descriptor/process is synthetic.

The nanobind-shaped Python classes below are cache/dispatch contract fixtures.
They are not native descriptors, C++ calls, real admissions or physics evidence.
"""
from __future__ import annotations

import copy
import importlib.abc
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DENIED_IMPORT_ATTEMPTS = []


class NoScientific(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"numpy", "scipy", "pyoomph"} or fullname.startswith(("numpy.", "scipy.", "pyoomph.", "_pyoomph")):
            DENIED_IMPORT_ATTEMPTS.append(fullname)
            raise AssertionError("scientific/native import denied in source controls: " + fullname)


BLOCKER = NoScientific()
sys.meta_path.insert(0, BLOCKER)
from runners import paper_dynamic_callback_protocol as protocol
from runners import paper_dynamic_fidelity as fidelity
from runners import paper_dynamic_marker as marker
from runners import run_paper_dynamic_fidelity as driver
from runners import run_paper_dynamic_fidelity_adapter as bridge


def store(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return fidelity.write_new(path, value)


class nb_bound_method:
    __module__ = "nanobind"  # Explicitly synthetic shape, not native evidence.

    def __init__(self, descriptor, instance):
        self.descriptor, self.instance = descriptor, instance

    def __call__(self):
        return self.descriptor(self.instance)


class nb_func:
    __module__ = "nanobind"  # Explicitly synthetic shape, not native evidence.

    def __get__(self, instance, owner):
        return self if instance is None else nb_bound_method(self, instance)

    def __call__(self, instance):
        instance.synthetic_descriptor_calls += 1
        if instance.fault == "miss":
            return
        callback = getattr(instance, protocol.HOOK)
        if type(callback).__module__ == "nanobind":
            instance.synthetic_default_calls += 1
            return
        callback()
        if instance.fault == "duplicate":
            callback()
        if instance.fault == "throw":
            raise RuntimeError("synthetic descriptor failure after delivery")
        return


class SyntheticCore:
    actions_before_newton_step = nb_func()


def fresh_problem_type(fault=None, retained=None):
    class SyntheticProblem(SyntheticCore):
        def __init__(self, case, *, initial_bond):
            self.case, self.initial_bond = copy.deepcopy(case), initial_bond
            self.fault = fault
            self.synthetic_descriptor_calls = self.synthetic_default_calls = self.release_calls = 0
            self.ndof_calls = 0
            self._initialised = False
            if retained is not None:
                retained.append(self)
                raise RuntimeError("synthetic constructor retains instance then raises")

        def ndof(self):
            self.ndof_calls += 1
            raise AssertionError("marker readiness must not call ndof")

        def release(self):
            self.release_calls += 1
            if self.fault == "release":
                raise RuntimeError("synthetic release failure")

    return SyntheticProblem


class Fixture:
    def __init__(self, root):
        self.root = root
        self.output = root / "attempt/runtime/fidelity"
        self.output.mkdir(parents=True)
        self.now = datetime(2026, 10, 14, 12, tzinfo=UTC)
        self.case = {"physics": {"problem": "finite_gas_bubble", "ca_in": -.002, "bond": 1.56,
                     "vhat": 3.3, "domain_over_r": 15}, "pressure": {"field": "static", "reference": "lower_static_zero",
                     "body_force": "explicit_both_phases"}, "synthetic_fixture": True}
        ref = {"path": str(root / "synthetic-input.json"), "sha256": "b" * 64}
        req = {"schema": protocol.SCHEMA + "-requirements", "role": "marker", "source_pin": protocol.PYOOMPH_PIN,
               "caps": dict(protocol.MARKER_CAPS), "controls": dict(protocol.MARKER_CONTROLS),
               "inherited_time_step_charge_spent": 1, "new_time_step_allowance": 0,
               "source": {"commit": "a" * 40, "root": str(ROOT), "files": {"synthetic.py": {
                   "path": str(root / "synthetic.py"), "sha256": "b" * 64}}},
               "output": str(self.output), "policy": dict(marker.POLICY), "case": self.case,
               "bindings": ref, "fresh_environment": ref, "controls_ref": ref, "case_ref": ref,
               "warm_result": None, "inputs": [ref],
               "environment": {"available": True, "commit": protocol.PYOOMPH_PIN, "native_core": ref},
               "callback_protocol": {"schema": protocol.SCHEMA, "type_forwarding_before_construction": True,
                   "same_instance_dispatch_only": True, "marker_interface_supported": True,
                   "full_fidelity_interface_supported": False}}
        phase = {"schema": protocol.SCHEMA + "-phase", "requirements": copy.deepcopy(req), "issued_by": "root",
                 "declared_at": self.now.isoformat(), "deadline_utc": (self.now + timedelta(seconds=100)).isoformat(),
                 "phase_authorized": True}
        phase_ref = {"path": str(root / "synthetic-phase.json"), "sha256": "c" * 64}
        admission = {"schema": protocol.SCHEMA + "-admission", "native_execution_authorized": True,
                     "marker_authorized": True, "phase": phase_ref, "requirements": copy.deepcopy(req),
                     "active_tracker": {"path": str(root / "synthetic-active-tracker.json"), "sha256": "d" * 64},
                     "role": "marker", "deadline_utc": phase["deadline_utc"]}
        self.frame = {"requirements": req, "phase": phase, "phase_ref": phase_ref, "admission": admission,
                      "admission_ref": {"path": str(root / "synthetic-admission.json"), "sha256": "e" * 64},
                      "native_process": {"pid": 123, "uid": 1000, "host": "synthetic-host", "start_time_ticks": 456,
                          "process_group_id": 123, "session_id": 123, "cwd": str(root / "attempt"),
                          "cgroup": ["synthetic-unit"], "argv": ["/synthetic/python", "/synthetic/bridge.py", "--role", "marker"]}}
        self.events = []

    def authority(self):
        return protocol.bind_admitted_marker_authority(self.frame, lambda: self.frame, clock=lambda: self.now)

    def owner(self, problem_type):
        authority = self.authority()
        charges = protocol.DurableMarkerCharges(self.output, authority, fidelity.write_new, fidelity.read_bound)
        owner = protocol.BeforeStepTypeForwarding(problem_type, authority, charges,
                                                 lambda label, value: self.events.append((label, copy.deepcopy(value))))
        return owner


class SourceControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stone-marker-source-synthetic-")
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()
        self.assertEqual(DENIED_IMPORT_ATTEMPTS, [])
        self.assertFalse(any(name in {"numpy", "scipy", "pyoomph"} or name.startswith(
            ("numpy.", "scipy.", "pyoomph.", "_pyoomph")) for name in sys.modules))

    def test_cap_aliases_missing_extra_refuse_before_charge(self):
        self.assertEqual(protocol.caps_contract(dict(protocol.MARKER_CAPS)), dict(protocol.MARKER_CAPS))
        for name, expected in protocol.MARKER_CAPS.items():
            for value in (bool(expected), float(expected), -1):
                with self.subTest(name=name, value=value):
                    candidate = dict(protocol.MARKER_CAPS)
                    candidate[name] = value
                    with self.assertRaises(ValueError):
                        protocol.caps_contract(candidate)
        for candidate in ({}, {**dict(protocol.MARKER_CAPS), "extra": 0}):
            with self.assertRaises(ValueError):
                protocol.caps_contract(candidate)
        self.assertFalse((self.fixture.output / "marker-charge.json").exists())

    def test_controls_have_strict_integer_boolean_string_types(self):
        for name, value in protocol.MARKER_CONTROLS.items():
            candidate = dict(protocol.MARKER_CONTROLS)
            candidate[name] = 1 if type(value) is bool else True
            with self.subTest(name=name), self.assertRaises(ValueError):
                protocol.controls_contract(candidate)

    def test_false_expired_mismatched_unauthorised_authority_refused(self):
        for key in ("requirements", "phase", "admission"):
            value = copy.deepcopy(self.fixture.frame)
            value[key] = False
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError, TypeError)):
                protocol.bind_admitted_marker_authority(value, lambda value=value: value, clock=lambda: self.fixture.now)
        for key in ("phase_authorized",):
            value = copy.deepcopy(self.fixture.frame)
            value["phase"][key] = False
            with self.assertRaises(ValueError):
                protocol.bind_admitted_marker_authority(value, lambda value=value: value, clock=lambda: self.fixture.now)
        for key in ("marker_authorized", "native_execution_authorized"):
            for alias in (False, 1):
                value = copy.deepcopy(self.fixture.frame)
                value["admission"][key] = alias
                with self.assertRaises(ValueError):
                    protocol.bind_admitted_marker_authority(value, lambda value=value: value, clock=lambda: self.fixture.now)
        self.fixture.now += timedelta(seconds=100)
        with self.assertRaises(ValueError):
            self.fixture.authority()

    def test_nested_phase_caps_alias_is_not_equal(self):
        self.fixture.frame["phase"]["requirements"]["caps"]["problem_constructions"] = True
        with self.assertRaises(ValueError):
            self.fixture.authority()

    def test_authority_original_mutation_and_snapshot_mutation(self):
        authority = self.fixture.authority()
        snapshot = authority.snapshot()
        snapshot["requirements"]["caps"]["problem_constructions"] = 99
        authority.require_current()
        self.fixture.frame["native_process"]["argv"].append("changed")
        with self.assertRaises(ValueError):
            authority.require_current()

    def test_direct_dict_authority_does_not_install_type(self):
        cls = fresh_problem_type()
        original = protocol.HOOK in vars(cls)
        with self.assertRaises(ValueError):
            protocol.BeforeStepTypeForwarding(cls, {"marker_authorized": True}, None, lambda *_: None)
        self.assertEqual(protocol.HOOK in vars(cls), original)

    def test_exclusive_class_owner_refuses_overlap_before_mutation(self):
        cls = fresh_problem_type()
        one = self.fixture.owner(cls)
        two = self.fixture.owner(cls)
        with one:
            problem = one.construct_case(self.fixture.case)
            with self.assertRaises(ValueError):
                two.__enter__()
            self.assertIs(vars(cls)[protocol.HOOK], one.shim)
            self.assertIs(one.live[id(problem)], problem)
        self.assertIs(vars(cls)[protocol.HOOK], one.shim)

    def test_changed_descriptor_refused_before_install_and_restore_mutation(self):
        cls = fresh_problem_type()
        owner = self.fixture.owner(cls)
        foreign = lambda *_: None
        setattr(cls, protocol.HOOK, foreign)
        with self.assertRaises(ValueError):
            owner.__enter__()
        self.assertIs(vars(cls)[protocol.HOOK], foreign)
        another = fresh_problem_type()
        owner = self.fixture.owner(another)
        owner.__enter__()
        setattr(another, protocol.HOOK, foreign)
        with self.assertRaises(ValueError):
            owner.__exit__(None, None, None)
        self.assertIs(vars(another)[protocol.HOOK], foreign)

    def test_unconstructed_owner_restores_exact_original_descriptor(self):
        cls = fresh_problem_type()
        with self.fixture.owner(cls):
            self.assertIn(protocol.HOOK, vars(cls))
        self.assertNotIn(protocol.HOOK, vars(cls))

    def test_inherited_descriptor_change_refuses_before_restoration_mutation(self):
        class Parent(SyntheticCore):
            actions_before_newton_step = nb_func()
        class Child(Parent):
            pass
        owner = self.fixture.owner(Child)
        owner.__enter__()
        Parent.actions_before_newton_step = lambda *_: None
        with self.assertRaises(ValueError):
            owner.__exit__(None, None, None)
        self.assertIs(vars(Child)[protocol.HOOK], owner.shim)

    def test_constructor_retained_then_raises_stays_owned_even_empty_registry(self):
        retained = []
        cls = fresh_problem_type(retained=retained)
        owner = self.fixture.owner(cls)
        with self.assertRaises(RuntimeError), owner:
            owner.construct_case(self.fixture.case)
        self.assertEqual(len(retained), 1)
        self.assertEqual(owner.live, {})
        self.assertTrue(owner.installed)
        self.assertIs(vars(cls)[protocol.HOOK], owner.shim)
        self.assertTrue((self.fixture.output / "construction-charge.json").exists())
        self.assertFalse((self.fixture.output / "marker-charge.json").exists())

    def test_arbitrary_factory_input_is_not_freshness_certificate(self):
        cls = fresh_problem_type()
        with self.fixture.owner(cls) as owner:
            with self.assertRaises(ValueError):
                owner.construct_case(lambda: cls(self.fixture.case, initial_bond=1.56))
            self.assertFalse(owner.construction_attempted)
        self.assertFalse((self.fixture.output / "construction-charge.json").exists())

    def test_charge_is_exclusive_complete_and_immutable_before_descriptor(self):
        cls = fresh_problem_type()
        with self.fixture.owner(cls) as owner:
            problem = owner.construct_case(self.fixture.case)
            with self.assertRaises(ValueError):
                owner.construct_case(self.fixture.case)
            charge = json.loads((self.fixture.output / "construction-charge.json").read_bytes())
            self.assertEqual(charge["authority"], self.fixture.frame)
            self.assertFalse(charge["retry_permitted"])
            (self.fixture.output / "construction-charge.json").write_text("changed")
            with self.assertRaises(ValueError):
                protocol.qualify_cpp_before_step_marker(problem, owner, SyntheticCore,
                    vars(SyntheticCore)[protocol.HOOK], lambda *_: None)
            self.assertEqual(problem.synthetic_descriptor_calls, 0)

    def test_mutated_construction_charge_reference_refuses_before_descriptor(self):
        cls = fresh_problem_type()
        with self.fixture.owner(cls) as owner:
            problem = owner.construct_case(self.fixture.case)
            owner.charges.references["construction"]["path"] = str(self.root / "foreign-charge.json")
            with self.assertRaises(ValueError):
                protocol.qualify_cpp_before_step_marker(problem, owner, SyntheticCore,
                    vars(SyntheticCore)[protocol.HOOK], lambda *_: None)
            self.assertEqual(problem.synthetic_descriptor_calls, 0)

    def test_charge_write_failure_and_expired_write_permit_no_native_entry_or_retry(self):
        cls = fresh_problem_type()
        owner = self.fixture.owner(cls)
        def failed_write(path, value):
            raise OSError("synthetic durable writer failure")
        owner.charges.write_new = failed_write
        with self.assertRaises(OSError), owner:
            owner.construct_case(self.fixture.case)
        self.assertTrue(owner.installed)
        self.assertIn("construction", owner.charges.attempted)
        cls = fresh_problem_type()
        owner = self.fixture.owner(cls)
        def expired_write(path, value):
            result = fidelity.write_new(path, value)
            self.fixture.now += timedelta(seconds=100)
            return result
        owner.charges.write_new = expired_write
        with self.assertRaises(ValueError), owner:
            owner.construct_case(self.fixture.case)
        self.assertEqual(owner.live, {})

    def test_one_synthetic_dispatch_same_instance_no_ndof_and_explicit_release(self):
        cls = fresh_problem_type()
        with self.fixture.owner(cls) as owner:
            problem = owner.construct_case(self.fixture.case)
            readiness = protocol.qualify_cpp_before_step_marker(problem, owner, SyntheticCore,
                vars(SyntheticCore)[protocol.HOOK], lambda label, value: self.fixture.events.append((label, value)))
            self.assertTrue(protocol.require_dispatch_marker_readiness(problem, readiness)["same_instance_dispatch"])
            self.assertEqual(problem.synthetic_descriptor_calls, 1)
            self.assertEqual(problem.ndof_calls, 0)
            with self.assertRaises(ValueError):
                protocol.qualify_cpp_before_step_marker(problem, owner, SyntheticCore,
                    vars(SyntheticCore)[protocol.HOOK], lambda *_: None)
            with self.assertRaises(ValueError):
                protocol.require_dispatch_marker_readiness(object(), readiness)
            with self.assertRaises(ValueError):
                protocol.require_dispatch_marker_readiness(problem, {"same_instance_dispatch": True})
            owner.release_instance(problem)
            self.assertEqual(problem.release_calls, 1)
            with self.assertRaises(ValueError):
                protocol.require_dispatch_marker_readiness(problem, readiness)
        self.assertTrue(owner.installed)
        self.assertIs(vars(cls)[protocol.HOOK], owner.shim)
        self.assertNotIn(protocol.HOOK, vars(problem))

    def test_python_callable_and_foreign_descriptor_refused_before_marker_charge(self):
        cls = fresh_problem_type()
        with self.fixture.owner(cls) as owner:
            problem = owner.construct_case(self.fixture.case)
            for candidate in (lambda p: getattr(p, protocol.HOOK)(), nb_func()):
                with self.assertRaises(ValueError):
                    protocol.qualify_cpp_before_step_marker(problem, owner, SyntheticCore, candidate, lambda *_: None)
            self.assertFalse(owner.marker_attempted)
            self.assertFalse((self.fixture.output / "marker-charge.json").exists())

    def test_missed_duplicate_throwing_descriptor_preserves_raw_charge_and_instance_restoration(self):
        for fault, expected in (("miss", 0), ("duplicate", 2), ("throw", 1)):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(prefix="stone-marker-fault-synthetic-") as raw:
                fixture = Fixture(Path(raw))
                cls = fresh_problem_type(fault=fault)
                events = []
                with fixture.owner(cls) as owner:
                    problem = owner.construct_case(fixture.case)
                    with self.assertRaises((ValueError, RuntimeError)):
                        protocol.qualify_cpp_before_step_marker(problem, owner, SyntheticCore,
                            vars(SyntheticCore)[protocol.HOOK], lambda label, value, events=events: events.append((label, value)))
                    self.assertTrue(owner.marker_attempted)
                    self.assertTrue((fixture.output / "marker-charge.json").exists())
                    self.assertEqual(sum(label == "before_step_cpp_marker_received" for label, _ in events), expected)
                    self.assertNotIn(protocol.HOOK, vars(problem))

    def test_full_warm_cold_and_unsupported_marker_execute_refuse_before_import(self):
        args = driver.parser().parse_args(["--bindings", "/synthetic/bindings", "--source", str(ROOT), "--role", "warm", "--check"])
        with self.assertRaises(ValueError):
            driver.preflight(args)
        with patch.object(protocol, "MARKER_DRIVER_INTERFACE_SUPPORTED", False), self.assertRaises(ValueError):
            driver.execute({"requirements": {"role": "marker"}}, self.fixture.output, lambda *_: None)
        with patch.object(protocol, "MARKER_DRIVER_INTERFACE_SUPPORTED", True), self.assertRaises(ValueError):
            driver.execute({"requirements": {"role": "marker"}, "native_execution_authorized": False,
                            "pending": []}, self.fixture.output, lambda *_: None)
        with self.assertRaises(ValueError):
            protocol.require_cpp_marker_readiness(object(), object())

    def test_book_duplicate_and_nonfinite_authority_refused(self):
        for name, raw in [("duplicate", b'{"phase_authorized":false,"phase_authorized":true}'),
                          ("nonfinite", b'{"value":NaN}')]:
            path = self.root / (name + ".json")
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                driver.Book().read(fidelity.reference(path))

    def test_constructor_policy_counts_allowed_pass_denies_extra_and_evolution(self):
        class SyntheticCompilerBase:
            @classmethod
            def available_compilers(cls):
                return {"system": 9}

            def compile(self):
                raise AssertionError("must not reach compiler implementation")

        class SyntheticSystemCompiler(SyntheticCompilerBase):
            compiler_id = "system"

        cls = fresh_problem_type()
        cls.get_ccompiler = lambda self: SyntheticSystemCompiler()
        with self.fixture.owner(cls) as owner:
            policy = marker._ConstructorPolicy(owner, SyntheticCompilerBase, SyntheticSystemCompiler,
                                                {"ndof"}, lambda *_: None)
            policy.install()
            self.assertEqual(SyntheticCompilerBase.available_compilers(), {"system": 9})
            with self.assertRaises(ValueError):
                SyntheticCompilerBase.available_compilers()
            with self.assertRaises(ValueError):
                SyntheticCompilerBase().compile()
            policy.close()

    def test_result_policy_keeps_historical_charge_and_all_scientific_claims_false(self):
        prepared = {"requirements": self.fixture.frame["requirements"], "source": self.fixture.frame["requirements"]["source"],
                    "controls": dict(protocol.MARKER_CONTROLS), "phase_ref": self.fixture.frame["phase_ref"]}
        result = marker.initial_result(prepared)
        self.assertEqual(result["status"], "failed_marker")
        self.assertEqual(result["inherited_time_step_charge_spent"], 1)
        self.assertEqual(result["new_time_step_allowance"], 0)
        self.assertEqual(result["time_step_calls_charged"], 0)
        for key in marker.CLAIMS:
            self.assertIs(result[key], False)
        self.assertIs(result["native_destruction_proved"], False)

    def test_release_recorder_failure_still_releases_owned_instance(self):
        cls = fresh_problem_type()
        with self.fixture.owner(cls) as owner:
            problem = owner.construct_case(self.fixture.case)
            def record(label, value):
                if label == "problem_release_entered":
                    raise OSError("synthetic record failure")
            owner.record = record
            with self.assertRaises(OSError):
                owner.release_instance(problem)
            self.assertEqual(problem.release_calls, 1)
            self.assertIn(id(problem), owner.release_completed)
        self.assertIs(vars(cls)[protocol.HOOK], owner.shim)

    def test_delivered_marker_record_failure_keeps_raw_delivery_before_failed_write(self):
        cls = fresh_problem_type()
        with self.fixture.owner(cls) as owner:
            problem = owner.construct_case(self.fixture.case)
            def record(label, value):
                if label == "before_step_cpp_marker_received":
                    raise OSError("synthetic raw recorder failure")
            with self.assertRaises(OSError):
                protocol.qualify_cpp_before_step_marker(problem, owner, SyntheticCore,
                    vars(SyntheticCore)[protocol.HOOK], record)
            self.assertEqual(owner.raw_marker_deliveries, 1)
            self.assertFalse(owner.normal_descriptor_return)
            self.assertNotIn(protocol.HOOK, vars(problem))
            self.assertTrue((self.fixture.output / "marker-charge.json").exists())

    def test_declared_marker_body_writes_charge_evidence_raw_sequence_without_native_import(self):
        class SyntheticCompilerBase:
            @classmethod
            def available_compilers(cls):
                return {"system": 9}

            def compile(self):
                raise AssertionError("FEM compiler must not be reached")

        # This source-shaped class is still an explicit Python fixture.
        system = type("SystemCCompiler", (SyntheticCompilerBase,), {
            "__module__": "pyoomph.generic.ccompiler", "compiler_id": "system"})
        defaults = {"value": None}
        solver = SimpleNamespace(set_default_linear_solver=lambda value: defaults.update(value=value),
                                 get_default_linear_solver=lambda: defaults["value"])
        cls = fresh_problem_type()
        original = cls.__init__
        def construct(problem, case, *, initial_bond):
            original(problem, case, initial_bond=initial_bond)
            problem._lasolver = solver.get_default_linear_solver()
            SyntheticCompilerBase.available_compilers()
        cls.__init__ = construct
        cls.get_ccompiler = lambda problem: system()
        authority = self.fixture.authority()
        prepared = {"marker_authority": authority, "case": self.fixture.case}
        events = []
        def emit(label, value):
            event = {"schema": protocol.SCHEMA + "-raw-event", "role": "marker", "sequence": len(events) + 2,
                     "event": label, "authority": authority.snapshot(), "value": value}
            ref = store(self.fixture.output / (f"receipt-{len(events) + 2:04d}-" + label + ".json"), event)
            events.append(ref)
            return ref
        result = marker._run_declared_marker(prepared, self.fixture.output, emit, driver,
            SimpleNamespace(Problem=SyntheticCore), cls, SyntheticCompilerBase, system, solver)
        self.assertTrue(result["same_instance_dispatch"])
        self.assertEqual(result["selected_compiler"], {"module": "pyoomph.generic.ccompiler",
            "class": "SystemCCompiler", "compiler_id": "system"})
        self.assertEqual(result["linear_solver_default"], "superlu")
        self.assertEqual(result["constructor_compiler_availability_passes"], 1)
        evidence = json.loads((self.fixture.output / "marker-evidence.json").read_bytes())
        self.assertEqual(set(evidence["local_identity"]), {"owner_id", "problem_type_id", "instance_id", "descriptor_id"})
        rows = [json.loads(Path(ref["path"]).read_bytes()) for ref in events]
        received = [row for row in rows if row["event"] == "before_step_cpp_marker_received"]
        self.assertEqual(len(received), 1)
        self.assertEqual(set(received[0]["value"]), {"owner_id", "problem_type_id", "instance_id", "descriptor_id",
            "core_problem_type_id", "raw_marker_deliveries"})
        self.assertEqual(received[0]["value"]["raw_marker_deliveries"], 1)
        self.assertEqual([row["sequence"] for row in rows], list(range(2, len(rows) + 2)))
        problem = next(iter(prepared["marker_owner"].live.values()))
        self.assertEqual(problem.ndof_calls, 0)
        self.assertEqual(problem.release_calls, 1)
        self.assertIs(vars(cls)[protocol.HOOK], prepared["marker_owner"].shim)
        self.assertNotIn("native_imported", prepared)

    def test_full_official_admission_contract_retains_exact_tracker_launch_process_affinity_gates(self):
        frame = copy.deepcopy(self.fixture.frame)
        source = frame["requirements"]["source"]
        source["files"] = {"runner.py": fidelity.reference(ROOT / "runners/run_paper_dynamic_fidelity.py")}
        frame["phase"]["requirements"] = copy.deepcopy(frame["requirements"])
        frame["admission"]["requirements"] = copy.deepcopy(frame["requirements"])
        artifact = lambda name: {"path": str(self.root / (name + ".json")), "sha256": "b" * 64}
        reservation = {"state": "active", "id": "synthetic-reservation", "generation_id": "synthetic-generation",
                       "generation_sha256": "f" * 64}
        tracker = {"schema": protocol.SCHEMA + "-active-tracker", "phase_kind": protocol.SCHEMA, "status": "running",
                   "role": "marker", "phase": frame["phase_ref"], "phase_authorized": True,
                   "output": str(self.fixture.output), "source_commit": source["commit"], "deadline_utc": frame["phase"]["deadline_utc"],
                   "caps": dict(protocol.MARKER_CAPS), "checked_at": self.fixture.now.isoformat(), "reservation": reservation,
                   "active_dispatcher": artifact("dispatcher"), "launch": artifact("launch"), "launch_audit": artifact("audit"),
                   "current_root_identity": frame["native_process"], "cpus": [6, 38]}
        dispatcher = {"authority_verified": True, "status": "active", "reservation_id": reservation["id"],
                      "generation_id": reservation["generation_id"], "cpus": [6, 38]}
        launch = {"launch_audit": tracker["launch_audit"], "root_identity": frame["native_process"]}
        table = {frame["admission"]["active_tracker"]["path"]: tracker,
                 tracker["active_dispatcher"]["path"]: dispatcher, tracker["launch"]["path"]: launch}
        prepared = {"phase": frame["phase"], "phase_ref": frame["phase_ref"], "requirements": frame["requirements"],
                    "output": self.fixture.output, "source": source,
                    "book": SimpleNamespace(read=lambda ref: table[ref["path"]], file=lambda ref: None, stable=lambda: None)}
        kwargs = {"now": self.fixture.now, "identity": frame["native_process"], "affinity": {123: [6, 38]}}
        self.assertEqual(driver.admission_contract(frame["admission"], prepared, **kwargs), frame["native_process"])
        for key, replacement in [("pid", 123.0), ("argv", ["/synthetic/changed"]), ("cwd", str(self.root))]:
            identity = copy.deepcopy(frame["native_process"])
            identity[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.admission_contract(frame["admission"], prepared, **{**kwargs, "identity": identity})
        for value in ([6.0, 38.0], [6], [38, 6]):
            with self.subTest(affinity=value), self.assertRaises(ValueError):
                driver.admission_contract(frame["admission"], prepared, **{**kwargs, "affinity": {123: value}})
        tracker["caps"]["time_step_calls"] = False
        with self.assertRaises(ValueError):
            driver.admission_contract(frame["admission"], prepared, **kwargs)
        tracker["caps"]["time_step_calls"] = 0
        dispatcher["cpus"] = [6.0, 38.0]
        with self.assertRaises(ValueError):
            driver.admission_contract(frame["admission"], prepared, **kwargs)


class MetadataAndBridgeControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stone-marker-metadata-synthetic-")
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.attempt = self.root / "attempt"
        self.runtime = self.attempt / "runtime"
        self.runtime.mkdir(parents=True)
        source_case = ROOT / 'validation/herradaEtAl_paper_model/cases/h15-bo1p56.json'
        case = json.loads(source_case.read_bytes())
        case['case_id'] = 'synthetic_marker_i_input'
        case['physics']['ca_in'] = -.002
        self.case_ref = store(self.inputs / "case.json", case)
        core_path = self.inputs / "synthetic-core.so"
        core_path.write_bytes(b"synthetic core bytes, never imported")
        core_ref = fidelity.reference(core_path)
        self.observed = {"available": True, "commit": protocol.PYOOMPH_PIN, "native_core": core_ref,
                         "interpreter": {"executable": "/synthetic/python"}}
        package = Path(os.environ['STONE_MARKER_SOURCE_FIXTURES'])
        python_sources = {}
        for name in marker.PYTHON_SOURCE_HASHES:
            destination = self.inputs / "source-text" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((package / name).read_bytes())
            python_sources[name] = fidelity.reference(destination)
        receipt = {"schema": "pyoomph-environment-v1", "compatible": True, "checked_compiler": "system",
                   "checked_solver": "superlu", "checks": {"compiler": {"status": "passed"}, "solver": {"status": "passed"}},
                   "pyoomph": {"resolved_commit": protocol.PYOOMPH_PIN, "core_sha256": core_ref["sha256"]},
                   "interpreter": {"path": "/synthetic/python"}, "mpi": {"compiled": False}, "synthetic_fixture": True}
        self.receipt_ref = store(self.inputs / "environment.json", receipt)
        binding = {"schema": protocol.SCHEMA + "-bindings", "case": self.case_ref,
                   "environment": {"pin": protocol.PYOOMPH_PIN, "native_core": core_ref, "interpreter": "/synthetic/python",
                       "receipt": self.receipt_ref, "python_sources": python_sources}, "inherited_time_step_charge_spent": 1}
        self.binding_ref = store(self.inputs / "bindings.json", binding)
        self.controls_ref = store(self.inputs / "controls.json", dict(protocol.MARKER_CONTROLS))
        self.component = {"head": "a" * 40, "commit": "a" * 40, "clean": True}
        self.values = ["--source", str(ROOT), "--bindings", self.binding_ref["path"], "--role", "marker",
                       "--controls", self.controls_ref["path"], "--environment", self.receipt_ref["path"],
                       "--out", str(self.runtime / "fidelity"), "--check"]

    def tearDown(self):
        self.temp.cleanup()
        self.assertEqual(DENIED_IMPORT_ATTEMPTS, [])

    def metadata(self, values=None):
        args = driver.parser().parse_args(self.values if values is None else values)
        with patch.object(driver.ordinary, "require_pinned_environment", lambda value: None), \
             patch.object(driver, "parent_contract", side_effect=AssertionError("marker cannot read a parent dump")), \
             patch.object(driver.history, "serialized_metadata", side_effect=AssertionError("marker cannot decode arrays")):
            return driver.preflight(args, component=self.component, observed=self.observed,
                                    committed_reader=lambda name: (ROOT / name).read_bytes())

    def test_real_marker_metadata_branches_before_parent_and_is_write_free(self):
        prepared = self.metadata()
        self.assertEqual(prepared["requirements"]["role"], "marker")
        self.assertIs(prepared["native_execution_authorized"], False)
        self.assertEqual(prepared["requirements"]["caps"], dict(protocol.MARKER_CAPS))
        self.assertNotIn("parent", prepared)
        self.assertEqual(list(self.runtime.iterdir()), [])
        self.assertFalse(any(Path(ref["path"]).suffix == ".dump" for ref in prepared["requirements"]["inputs"]))

    def test_marker_phase_alias_changed_source_and_old_binding_refuse(self):
        prepared = self.metadata()
        now = datetime.now(UTC)
        phase = {"schema": protocol.SCHEMA + "-phase", "requirements": copy.deepcopy(prepared["requirements"]),
                 "issued_by": "root", "declared_at": now.isoformat(),
                 "deadline_utc": (now + timedelta(seconds=30)).isoformat(), "phase_authorized": True}
        phase["requirements"]["caps"]["time_step_calls"] = False
        with self.assertRaises(ValueError):
            driver.phase_contract(phase, prepared["requirements"], now=now)
        phase["requirements"] = copy.deepcopy(prepared["requirements"])
        phase["requirements"]["source"]["commit"] = "b" * 40
        with self.assertRaises(ValueError):
            driver.phase_contract(phase, prepared["requirements"], now=now)

    def test_same_pid_marker_bridge_writes_canonical_files_before_delegate(self):
        prepared = self.metadata()
        prepared["pending"] = []
        prepared["phase"] = {"synthetic_fixture": True}
        prepared["phase_ref"] = {"path": str(self.inputs / "phase.json"), "sha256": "c" * 64}
        observed = []
        def delegated(values):
            observed.append((os.getpid(), list(sys.argv)))
            manifest = json.loads((self.runtime / "run-manifest.json").read_bytes())
            progress = [json.loads(row) for row in (self.runtime / "progress.jsonl").read_text().splitlines()]
            self.assertIs(manifest["marker_only"], True)
            self.assertIs(manifest["equations_initialised"], False)
            self.assertNotIn("restart_from", manifest)
            self.assertIsNone(manifest["qualified_stationary_parent"])
            self.assertEqual(progress[0]["event"], "prepared")
            self.assertFalse((self.runtime / "fidelity").exists())
            return 0
        fake = SimpleNamespace(parser=driver.parser, preflight=lambda args: prepared,
                               fidelity=fidelity, ordinary=driver.ordinary, main=delegated)
        arguments = ["--case", self.case_ref["path"], "--source", str(ROOT), "--bindings", self.binding_ref["path"],
                     "--runtime", str(self.runtime), "--role", "marker", "--phase", str(self.inputs / "phase.json"),
                     "--admission", str(self.inputs / "admission.json"), "--execute"]
        pid, argv = os.getpid(), list(sys.argv)
        with patch.object(bridge.Path, "cwd", return_value=self.attempt):
            self.assertEqual(bridge.main(arguments, driver=fake), 0)
        self.assertEqual(observed, [(pid, argv)])
        with patch.object(bridge.Path, "cwd", return_value=self.attempt), self.assertRaises(ValueError):
            bridge.main(arguments, driver=fake)

    def test_marker_bridge_metadata_no_files_and_wrong_case_refused(self):
        prepared = self.metadata()
        fake = SimpleNamespace(parser=driver.parser, preflight=lambda args: prepared,
                               fidelity=fidelity, ordinary=driver.ordinary, main=lambda _: self.fail("metadata delegated"))
        arguments = ["--case", self.case_ref["path"], "--source", str(ROOT), "--bindings", self.binding_ref["path"],
                     "--runtime", str(self.runtime), "--role", "marker", "--check"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(bridge.main(arguments, driver=fake), 0)
        self.assertEqual(list(self.runtime.iterdir()), [])
        Path(self.case_ref["path"]).write_text('{"case_id":"changed"}')
        with self.assertRaises(ValueError):
            bridge.main(arguments, driver=fake)


if __name__ == "__main__":
    unittest.main(verbosity=2)
