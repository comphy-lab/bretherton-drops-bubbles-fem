"""Native-denied marker manifest regression against the official consumer.

The injected driver and all run/launch/receipt data are synthetic. These
checks establish manifest compatibility only; no solver or native probe runs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
HELPER_SELECTOR = os.environ.get("STONE_PYOOMPH_PROJECT_HELPER")
if not HELPER_SELECTOR:
    raise unittest.SkipTest("Set STONE_PYOOMPH_PROJECT_HELPER to run the exact consumer controls")
ROOT = Path(__file__).resolve().parents[1]
IMPORT_ATTEMPTS = []


class NoScientific(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"numpy", "scipy", "pyoomph"} or fullname.startswith(("numpy.", "scipy.", "pyoomph.", "_pyoomph")):
            IMPORT_ATTEMPTS.append(fullname)
            raise AssertionError("scientific/native import denied")


sys.meta_path.insert(0, NoScientific())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = load_module("synthetic_marker_manifest_bridge", ROOT / "runners/run_paper_dynamic_fidelity_adapter.py")
HELPER_PATH = Path(HELPER_SELECTOR).resolve(strict=True)
HELPER_SHA256 = "16a56cdd86be5ebf0cfdb60db864ddd6dc6ad854959b0ff88f0ac3878dbece17"
assert hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() == HELPER_SHA256
official = load_module("source_bound_official_pyoomph_project", HELPER_PATH)
PIN = "a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618"


def reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)


class CanonicalManifestControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stone-marker-manifest-synthetic-")
        self.root = Path(self.temp.name)
        self.component = self.root / "component"
        (self.component / "runners").mkdir(parents=True)
        self.entry = self.component / bridge.ENTRYPOINT
        self.entry.write_bytes((ROOT / bridge.ENTRYPOINT).read_bytes())
        self.attempt = self.root / "attempt"
        self.runtime = self.attempt / "runtime"
        self.runtime.mkdir(parents=True)
        self.case_path = self.root / "synthetic-case.json"
        self.case = {"case_id": "synthetic_marker_case", "physics": {"ca_in": -.002}}
        write_new(self.case_path, self.case)
        self.case_ref = reference(self.case_path)
        source = {"commit": "a" * 40, "files": {bridge.ENTRYPOINT: reference(self.entry)}}
        self.prepared = {"case_ref": self.case_ref, "case": self.case, "pending": [],
            "requirements": {"role": "marker", "synthetic_fixture": True}, "source": source,
            "component": {"commit": "a" * 40}, "environment": {"commit": PIN},
            "controls": {"synthetic_fixture": True}, "phase": {"synthetic_fixture": True},
            "phase_ref": {"path": str(self.root / "synthetic-phase.json"), "sha256": "b" * 64},
            "book": SimpleNamespace(file=lambda value: None)}
        self.pid_and_argv = []
        def parser():
            result = argparse.ArgumentParser()
            for name in ("source", "bindings", "role", "out", "controls", "environment", "warm-result", "phase", "admission"):
                result.add_argument("--" + name)
            result.add_argument("--execute", action="store_true")
            result.add_argument("--check", action="store_true")
            return result
        def delegate(values):
            self.pid_and_argv.append((os.getpid(), list(sys.argv)))
            self.assertTrue((self.runtime / "run-manifest.json").is_file())
            self.assertTrue((self.runtime / "progress.jsonl").is_file())
            return 0
        self.driver = SimpleNamespace(parser=parser, preflight=lambda args: self.prepared,
            fidelity=SimpleNamespace(reference=reference, bound=lambda value: None, write_new=write_new),
            ordinary=SimpleNamespace(read_case=lambda path: json.loads(path.read_text()), CASE_FIELDS_USED=("case_id", "physics")),
            main=delegate)
        args = ["--case", str(self.case_path), "--source", str(self.component), "--bindings", str(self.root / "synthetic-bindings.json"),
            "--runtime", str(self.runtime), "--role", "marker", "--phase", str(self.root / "synthetic-phase.json"),
            "--admission", str(self.root / "synthetic-admission.json"), "--execute"]
        with patch.object(bridge, "ROOT", self.component), patch.object(bridge.Path, "cwd", return_value=self.attempt):
            self.assertEqual(bridge.main(args, driver=self.driver), 0)
        self.manifest = json.loads((self.runtime / "run-manifest.json").read_bytes())

    def tearDown(self):
        self.temp.cleanup()
        self.assertEqual(IMPORT_ATTEMPTS, [])
        self.assertEqual(hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest(), HELPER_SHA256)

    def consume(self, *, launch=None):
        values = ["validate-run-output", "--run", str(self.runtime)]
        if launch is not None:
            values.extend(["--launch-manifest", str(launch)])
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(official.subprocess, "run", side_effect=AssertionError("host/Git command denied")), \
             redirect_stdout(output), redirect_stderr(errors):
            code = official.main(values)
        return code, output.getvalue(), errors.getvalue()

    def test_actual_consumer_accepts_marker_without_restart_claim_and_preserves_scope(self):
        code, output, errors = self.consume()
        self.assertEqual(code, 0, errors)
        result = json.loads(output)
        self.assertIn("provenance agreement only", result["claim"])
        self.assertNotIn("restart_from", self.manifest)
        self.assertIsNone(self.manifest["qualified_stationary_parent"])
        self.assertIs(self.manifest["marker_only"], True)
        for key in ("azimuthal_stability", "equations_initialised", "stability_layout_initialised",
                    "scientific_acceptance", "production_jump_qualified", "nonflat_physical_history_qualified",
                    "native_execution_authorized"):
            self.assertIs(self.manifest[key], False)
        self.assertEqual(self.manifest["inherited_time_step_charge_spent"], 1)
        self.assertEqual(self.manifest["new_time_step_allowance"], 0)
        self.assertFalse(any(path.suffix == ".dump" for path in self.root.rglob("*")))
        self.assertEqual(self.pid_and_argv, [(os.getpid(), list(sys.argv))])

    def test_actual_consumer_and_launch_receipt_match_accept_without_helper_changes(self):
        receipt = self.root / "synthetic-environment.json"
        write_new(receipt, {"schema": "pyoomph-environment-v1", "compatible": True,
                           "pyoomph": {"resolved_commit": PIN}})
        launch = self.root / "synthetic-launch.json"
        write_new(launch, {"schema": "pyoomph-run-manifest-v1", "created_at": "2026-10-14T00:00:00+00:00",
            "case_id": self.case["case_id"], "case": self.case_ref, "runner": reference(self.entry),
            "fem_commit": self.manifest["component_commit"], "pyoomph_commit": PIN,
            "core_sha256": "c" * 64, "environment_receipt": reference(receipt)})
        code, output, errors = self.consume(launch=launch)
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output)["launch_manifest"], reference(launch))

    def test_original_null_restart_shape_still_refused_by_actual_consumer(self):
        path = self.runtime / "run-manifest.json"
        value = {**self.manifest, "restart_from": None}
        path.write_text(json.dumps(value))
        before = path.read_bytes()
        code, _output, errors = self.consume()
        self.assertEqual(code, 2)
        self.assertIn("restart_from must record the parent state's path and SHA-256", errors)
        self.assertEqual(path.read_bytes(), before)

    def test_ordinary_malformed_restart_claims_still_refused(self):
        ordinary = copy.deepcopy(self.manifest)
        ordinary.pop("marker_only")
        ordinary["fidelity_role"] = "ordinary_synthetic"
        for restart in (None, {}, {"path": "/synthetic/parent.dump"}, {"path": "/synthetic/parent.dump", "sha256": "bad"}):
            with self.subTest(restart=restart):
                (self.runtime / "run-manifest.json").write_text(json.dumps({**ordinary, "restart_from": restart}))
                code, _output, errors = self.consume()
                self.assertEqual(code, 2)
                self.assertIn("restart_from must record the parent state's path and SHA-256", errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
