"""W1 controls using exact getter/emitter source and synthetic leaves only.

No native or scientific package is imported. Compiler classes and descriptor
calls are Python fixtures, including their deliberately source-shaped names.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("marker_recording_test_support", Path(__file__).with_name("test_paper_dynamic_marker_source.py"))
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
protocol, marker, driver, fidelity = support.protocol, support.marker, support.driver, support.fidelity


def exact_getter(base, cached):
    path = Path(os.environ["STONE_MARKER_SOURCE_FIXTURES"]) / "pyoomph/__init__.py"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "1aa7d956d78da567084566bcd4acf34c340d9900107ead4bd9fc650df13c4b7a"
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "get_default_c_compiler")
    original_import = builtins.__import__
    def synthetic_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "generic.ccompiler" and level == 1:
            return SimpleNamespace(BaseCCompiler=base)
        if name.startswith(("pyoomph", "numpy", "scipy")):
            raise AssertionError("native/scientific import denied")
        return original_import(name, globals, locals, fromlist, level)
    namespace = {"__builtins__": {**vars(builtins), "__import__": synthetic_import},
                 "_default_c_compiler": "system", "_resolved_default_c_compiler": "system" if cached else None}
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - exact hash-bound getter AST with synthetic import leaf
    return namespace["get_default_c_compiler"]


def exact_emitter(prepared, output):
    path = ROOT / "runners/run_paper_dynamic_fidelity.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "e14801e977931980a07a4fd5984590237aad5e9c62e25c11df25b948ff5da5a4"
    main = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    emit = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "emit")
    wrapper = ast.FunctionDef(name="make_emitter", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
        kw_defaults=[], defaults=[]), body=[ast.Assign(targets=[ast.Name(id="counter", ctx=ast.Store())], value=ast.Constant(value=1)),
        emit, ast.Return(value=ast.Name(id="emit", ctx=ast.Load()))], decorator_list=[])
    namespace = {"prepared": prepared, "output": output, "fidelity": fidelity, "marker": marker}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), "exec"), namespace)  # noqa: S102 - exact hash-bound emitter AST, no native imports
    return namespace["make_emitter"]()


class RecordingFailureControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stone-marker-W1-synthetic-")
        self.fixture = support.Fixture(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()
        self.assertEqual(support.DENIED_IMPORT_ATTEMPTS, [])
        self.assertFalse(any(name in {"numpy", "scipy", "pyoomph"} or name.startswith(
            ("numpy.", "scipy.", "pyoomph.", "_pyoomph")) for name in sys.modules))

    def run_body(self, *, fault_event=None, after_write=False, backend_fault=False, cached=False, callback_fault=None):
        entered = []
        class SyntheticCompilerBase:
            @classmethod
            def available_compilers(cls):
                entered.append(True)
                if backend_fault:
                    raise RuntimeError("synthetic original availability backend failure")
                return {"system": 9}

            def compile(self):
                raise AssertionError("FEM compile denied")

        system = type("SystemCCompiler", (SyntheticCompilerBase,), {
            "__module__": "pyoomph.generic.ccompiler", "compiler_id": "system"})
        getter = exact_getter(SyntheticCompilerBase, cached)
        defaults = {"value": None}
        solver = SimpleNamespace(set_default_linear_solver=lambda value: defaults.update(value=value),
                                 get_default_linear_solver=lambda: defaults["value"])
        cls = support.fresh_problem_type(fault=callback_fault)
        original = cls.__init__
        def construct(problem, case, *, initial_bond):
            original(problem, case, initial_bond=initial_bond)
            problem._lasolver = solver.get_default_linear_solver()
            # Execute the unchanged pinned getter, including its exception catch.
            problem.synthetic_selected_name = getter()
        cls.__init__ = construct
        cls.get_ccompiler = lambda problem: system()
        prepared = {"marker_authority": self.fixture.authority(), "case": self.fixture.case,
                    "requirements": self.fixture.frame["requirements"], "source": self.fixture.frame["requirements"]["source"],
                    "controls": dict(protocol.MARKER_CONTROLS), "phase_ref": self.fixture.frame["phase_ref"], "raw_events": []}
        emit = exact_emitter(prepared, self.fixture.output)
        original_write = fidelity.write_new
        failed = []
        def write(path, value):
            if fault_event is not None and value.get("event") == fault_event and not failed:
                failed.append(str(path))
                if after_write:
                    original_write(path, value)
                raise OSError("synthetic raw-event write/reference failure")
            return original_write(path, value)
        error = None
        returned = None
        with patch.object(fidelity, "write_new", write):
            try:
                returned = marker._run_declared_marker(prepared, self.fixture.output, emit, driver,
                    SimpleNamespace(Problem=support.SyntheticCore), cls, SyntheticCompilerBase, system, solver)
            except (ValueError, RuntimeError, OSError) as caught:
                error = caught
        result = marker.initial_result(prepared)
        if returned is not None:
            result.update(returned)
            result["status"] = "complete_marker"
        marker.retain_result_state(result, prepared)
        problem = next(iter(prepared["marker_owner"].live.values()))
        return prepared, result, problem, entered, error, failed

    def assert_refused_before_marker(self, result, problem):
        self.assertEqual(result["status"], "failed_marker")
        self.assertTrue(result["constructor_policy_failed"])
        self.assertFalse(result["constructor_policy_verified"])
        self.assertTrue(result["constructor_policy_errors"])
        self.assertEqual(result["problem_constructions_charged"], 1)
        self.assertEqual(result["cpp_virtual_marker_calls_charged"], 0)
        self.assertFalse((self.fixture.output / "marker-charge.json").exists())
        self.assertEqual(problem.synthetic_descriptor_calls, 0)
        self.assertEqual(problem.release_calls, 1)
        self.assertTrue(result["release_returned"])
        self.assertTrue(result["shim_retained_until_process_exit"])
        self.assertEqual(problem.ndof_calls, 0)
        self.assertEqual(result["inherited_time_step_charge_spent"], 1)
        self.assertEqual(result["new_time_step_allowance"], 0)
        for key in marker.CLAIMS:
            self.assertIs(result[key], False)

    def test_exact_getter_swallowed_prewrite_record_failure_is_sticky_before_marker(self):
        prepared, result, problem, entered, error, failed = self.run_body(fault_event="constructor_compiler_availability_pass")
        self.assertIsInstance(error, ValueError)
        self.assertEqual(problem.synthetic_selected_name, "system")
        self.assert_refused_before_marker(result, problem)
        self.assertEqual(result["constructor_compiler_availability_passes"], 1)
        self.assertEqual(result["constructor_compiler_backend_entries"], 0)
        self.assertEqual(result["constructor_compiler_backend_returns"], 0)
        self.assertEqual(entered, [])
        self.assertFalse(Path(failed[0]).exists())
        self.assertEqual(json.loads(Path(result["raw_events"][0]["path"]).read_bytes())["sequence"], 3)
        with self.assertRaises(ValueError):
            prepared["constructor_policy"].require_healthy()

    def test_exact_getter_swallowed_after_file_before_reference_failure_preserves_evidence(self):
        _prepared, result, problem, entered, _error, failed = self.run_body(
            fault_event="constructor_compiler_availability_pass", after_write=True)
        self.assert_refused_before_marker(result, problem)
        self.assertEqual(entered, [])
        self.assertTrue(Path(failed[0]).is_file())
        orphan = json.loads(Path(failed[0]).read_bytes())
        self.assertEqual(orphan["sequence"], 2)
        self.assertEqual(orphan["value"]["constructor_compiler_availability_passes"], 1)
        self.assertNotIn(str(Path(failed[0])), {ref["path"] for ref in result["raw_events"]})
        self.assertEqual(result["constructor_compiler_backend_entries"], 0)

    def test_exact_getter_swallowed_original_backend_failure_is_sticky(self):
        _prepared, result, problem, entered, _error, _failed = self.run_body(backend_fault=True)
        self.assert_refused_before_marker(result, problem)
        self.assertEqual(problem.synthetic_selected_name, "system")
        self.assertEqual(len(entered), 1)
        self.assertEqual(result["constructor_compiler_backend_entries"], 1)
        self.assertEqual(result["constructor_compiler_backend_returns"], 0)
        self.assertTrue(any(value["stage"] == "compiler_availability" for value in result["constructor_policy_errors"]))

    def test_returned_backend_record_failure_keeps_return_count_but_refuses_marker(self):
        _prepared, result, problem, entered, _error, _failed = self.run_body(
            fault_event="constructor_compiler_availability_returned")
        self.assert_refused_before_marker(result, problem)
        self.assertEqual(len(entered), 1)
        self.assertEqual(result["constructor_compiler_backend_entries"], 1)
        self.assertEqual(result["constructor_compiler_backend_returns"], 1)

    def test_body_and_verification_record_failure_paths_refuse_before_marker(self):
        for event in ("problem_constructed", "constructor_policy_verified"):
            with self.subTest(event=event), tempfile.TemporaryDirectory(prefix="stone-marker-W1-body-synthetic-") as raw:
                previous = self.fixture
                self.fixture = support.Fixture(Path(raw))
                try:
                    _prepared, result, problem, entered, _error, _failed = self.run_body(fault_event=event)
                    self.assert_refused_before_marker(result, problem)
                    self.assertEqual(len(entered), 1)
                finally:
                    self.fixture = previous

    def test_normal_and_cached_zero_paths_remain_matched_and_same_instance(self):
        for cached, count in ((False, 1), (True, 0)):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory(prefix="stone-marker-W1-normal-synthetic-") as raw:
                previous = self.fixture
                self.fixture = support.Fixture(Path(raw))
                try:
                    _prepared, result, problem, entered, error, _failed = self.run_body(cached=cached)
                    self.assertIsNone(error)
                    self.assertEqual(result["status"], "complete_marker")
                    self.assertFalse(result["constructor_policy_failed"])
                    self.assertEqual(result["constructor_policy_errors"], [])
                    self.assertTrue(result["constructor_policy_verified"])
                    self.assertEqual(result["constructor_compiler_availability_passes"], count)
                    self.assertEqual(result["constructor_compiler_backend_entries"], count)
                    self.assertEqual(result["constructor_compiler_backend_returns"], count)
                    self.assertEqual(len(entered), count)
                    self.assertEqual(problem.synthetic_descriptor_calls, 1)
                    self.assertEqual(problem.release_calls, 1)
                    self.assertEqual(result["raw_marker_deliveries"], 1)
                    self.assertEqual(result["cpp_virtual_marker_calls_charged"], 1)
                    rows = [json.loads(Path(ref["path"]).read_bytes()) for ref in result["raw_events"]]
                    self.assertEqual([row["sequence"] for row in rows], list(range(2, len(rows) + 2)))
                finally:
                    self.fixture = previous

    def test_original_marker_callback_failure_and_release_failure_are_sticky_with_spent_charge(self):
        for fault in ("throw", "release"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(prefix="stone-marker-W1-callback-synthetic-") as raw:
                previous = self.fixture
                self.fixture = support.Fixture(Path(raw))
                try:
                    _prepared, result, problem, _entered, _error, _failed = self.run_body(callback_fault=fault)
                    self.assertEqual(result["status"], "failed_marker")
                    self.assertTrue(result["constructor_policy_failed"])
                    self.assertFalse(result["constructor_policy_verified"])
                    self.assertEqual(result["constructor_compiler_backend_returns"], 1)
                    self.assertEqual(result["cpp_virtual_marker_calls_charged"], 1)
                    self.assertEqual(result["raw_marker_deliveries"], 1)
                    self.assertEqual(problem.release_calls, 1)
                    self.assertTrue(result["shim_retained_until_process_exit"])
                    self.assertIsNotNone(result["marker_charge"])
                finally:
                    self.fixture = previous


if __name__ == "__main__":
    unittest.main(verbosity=2)
