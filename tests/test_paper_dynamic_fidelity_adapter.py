"""NumPy/native-denied entry bridge controls; no original arrays or host actions."""
import hashlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]


class NoScientific(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('numpy', 'scipy', 'pyoomph') or fullname.startswith(('numpy.', 'scipy.', 'pyoomph.', '_pyoomph')):
            raise AssertionError('scientific imports denied in bridge tests')


import_guard = NoScientific()
sys.meta_path.insert(0, import_guard)
spec = importlib.util.spec_from_file_location('synthetic_official_bridge', HERE / 'runners/run_paper_dynamic_fidelity_adapter.py')
bridge = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(bridge)
finally:
    sys.meta_path.remove(import_guard)


def reference(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class BridgeTests(unittest.TestCase):
    def test_actual_driver_import_defers_all_scientific_packages(self):
        code = '''import importlib.abc
import json
import sys
from pathlib import Path
class NoScientific(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('numpy', 'scipy', 'pyoomph') or fullname.startswith(('numpy.', 'scipy.', 'pyoomph.', '_pyoomph')):
            raise AssertionError('pre-admission scientific import attempted: ' + fullname)
sys.meta_path.insert(0, NoScientific())
from runners import run_paper_dynamic_fidelity_adapter as bridge
driver = bridge.selected_driver(bridge.ROOT)
assert driver.parser() is not None
case = json.loads(Path('validation/herradaEtAl_paper_model/cases/h15-bo1p56.json').read_text())
case['physics']['ca_in'] = -.002
assert driver.model.validate_case(case)['physics']['ca_in'] == -.002
assert 'postprocess.film' not in sys.modules
assert not any(name in ('numpy', 'scipy', 'pyoomph') or name.startswith(('numpy.', 'scipy.', 'pyoomph.', '_pyoomph')) for name in sys.modules)
print('real driver imports without scientific packages')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=HERE,
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'real driver imports without scientific packages')

    def setUp(self):
        self.import_guard = NoScientific()
        sys.meta_path.insert(0, self.import_guard)
        self.temp = tempfile.TemporaryDirectory(prefix='stone-fidelity-bridge-synthetic-')
        self.root = Path(self.temp.name) / 'component'
        (self.root / 'runners').mkdir(parents=True)
        self.entry = self.root / bridge.ENTRYPOINT
        self.entry.write_bytes((HERE / bridge.ENTRYPOINT).read_bytes())
        self.attempt = Path(self.temp.name) / 'attempt'
        self.runtime = self.attempt / 'runtime'
        self.runtime.mkdir(parents=True)
        self.case = Path(self.temp.name) / 'case.json'
        self.case.write_text(json.dumps({'case_id': 'synthetic_case', 'physics': {'ca_in': -.002}}))
        self.arguments = ['--case', str(self.case), '--source', str(self.root), '--bindings', str(Path(self.temp.name) / 'bindings.json'),
            '--runtime', str(self.runtime), '--role', 'warm', '--phase', str(Path(self.temp.name) / 'phase.json'),
            '--admission', str(Path(self.temp.name) / 'admission.json'), '--execute']
        self.trace = []
        selected_case = json.loads(self.case.read_text())
        self.prepared = {'parent': {'case': reference(self.case), 'state': {'path': '/synthetic/state', 'sha256': 'b' * 64}},
            'case': selected_case, 'book': SimpleNamespace(file=lambda ref: self.trace.append('case-bound')),
            'pending': [], 'phase': {'synthetic_only': True}, 'phase_ref': {'path': '/synthetic/phase', 'sha256': 'c' * 64},
            'requirements': {'synthetic_only': True}, 'source': {'commit': 'a' * 40, 'files': {bridge.ENTRYPOINT: reference(self.entry)}},
            'component': {'commit': 'a' * 40}, 'environment': {'commit': 'a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618'}}
        def write_new(path, value):
            with path.open('x') as stream:
                json.dump(value, stream)
        def main(values):
            self.trace.append(('same-pid-entry', os.getpid(), list(sys.argv)))
            progress = [json.loads(x) for x in (self.runtime / 'progress.jsonl').read_text().splitlines()]
            manifest = json.loads((self.runtime / 'run-manifest.json').read_text())
            self.assertEqual(progress[0]['event'], 'prepared')
            self.assertFalse(progress[0]['native_execution_authorized'])
            self.assertEqual(manifest['runner'], reference(self.entry))
            self.assertEqual(manifest['case_sha256'], reference(self.case)['sha256'])
            self.assertFalse((self.runtime / 'fidelity').exists())
            self.assertIn(str(self.runtime / 'fidelity'), values)
            return 0
        self.driver = SimpleNamespace(parser=bridge.parser, preflight=lambda args: self.prepared,
            fidelity=SimpleNamespace(reference=reference, bound=lambda ref: self.trace.append('bridge-bound'), write_new=write_new),
            ordinary=SimpleNamespace(read_case=lambda path: json.loads(path.read_text()), CASE_FIELDS_USED=('case_id', 'physics')),
            main=main)
        # The underlying driver has --out, not the bridge's --runtime/--case selector.
        import argparse
        def parser():
            p = argparse.ArgumentParser()
            for name in ('source', 'bindings', 'out', 'role', 'controls', 'environment', 'warm-result', 'phase', 'admission'):
                p.add_argument('--' + name)
            p.add_argument('--check', action='store_true')
            p.add_argument('--execute', action='store_true')
            return p
        self.driver.parser = parser
        self.patches = [patch.object(bridge, 'ROOT', self.root), patch.object(bridge.Path, 'cwd', lambda: self.attempt)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        sys.meta_path.remove(self.import_guard)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_exact_case_official_files_precede_same_pid_call_and_preserve_argv(self):
        pid, arguments = os.getpid(), list(sys.argv)
        self.assertEqual(bridge.main(self.arguments, driver=self.driver), 0)
        self.assertEqual(self.trace[-1], ('same-pid-entry', pid, arguments))
        self.assertEqual(self.trace[:2], ['case-bound', 'bridge-bound'])

    def test_changed_case_refuses_before_progress_or_delegation(self):
        self.case.write_text('{"case_id":"changed"}')
        with self.assertRaisesRegex(ValueError, 'case selector differs'):
            bridge.main(self.arguments, driver=self.driver)
        self.assertFalse((self.runtime / 'progress.jsonl').exists())

    def test_unissued_phase_refuses_before_official_files(self):
        self.prepared['pending'] = ['actual root phase unissued']
        with self.assertRaisesRegex(ValueError, 'phase and future admission'):
            bridge.main(self.arguments, driver=self.driver)
        self.assertFalse((self.runtime / 'run-manifest.json').exists())

    def test_occupied_child_or_runtime_receipt_never_reused(self):
        (self.runtime / 'fidelity').mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh fidelity child'):
            bridge.main(self.arguments, driver=self.driver)

    def test_metadata_check_writes_no_runtime_or_child_files(self):
        values = [*self.arguments[:-1], '--check']
        self.assertEqual(bridge.main(values, driver=self.driver), 0)
        self.assertEqual(list(self.runtime.iterdir()), [])

    def test_cold_manifest_names_the_actual_retained_warm_initial_state(self):
        retained = {'path': '/synthetic/retained-warm-initial.dump', 'sha256': 'd' * 64}
        self.prepared['warm'] = {'initial_row': {'state': retained}}
        arguments = list(self.arguments)
        arguments[arguments.index('--role') + 1] = 'cold'
        self.assertEqual(bridge.main(arguments, driver=self.driver), 0)
        manifest = json.loads((self.runtime / 'run-manifest.json').read_text())
        self.assertEqual(manifest['restart_from'], retained)
        self.assertEqual(manifest['qualified_stationary_parent'], self.prepared['parent']['state'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
