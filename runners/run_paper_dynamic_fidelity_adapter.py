"""Official one-case entry bridge for the separately admitted fidelity driver.

Delegation stays in this PID and retains the kernel argv used by the official
adapter audit. Metadata, canonical provenance and fresh progress precede the
unchanged driver's native admission. This bridge issues no native permission.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = 'runners/run_paper_dynamic_fidelity_adapter.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def selected_driver(root):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    driver = importlib.import_module('runners.run_paper_dynamic_fidelity')
    require(Path(driver.__file__).resolve() == root / 'runners/run_paper_dynamic_fidelity.py',
            'same-component fidelity driver required')
    return driver


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('case', 'source', 'bindings', 'runtime'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--role', choices=('marker', 'warm', 'cold'), required=True)
    for name in ('controls', 'environment', 'warm-result', 'phase', 'admission'):
        result.add_argument('--' + name, type=Path)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--execute', action='store_true')
    return result


def driver_arguments(args, child):
    values = ['--source', str(args.source), '--bindings', str(args.bindings),
              '--role', args.role, '--out', str(child)]
    for name in ('controls', 'environment', 'warm_result', 'phase', 'admission'):
        value = getattr(args, name)
        if value is not None:
            values += ['--' + name.replace('_', '-'), str(value)]
    return [*values, '--check' if args.check else '--execute']


def case_selector(path, prepared, driver):
    reference = driver.fidelity.reference(path.absolute())
    original = prepared['case_ref'] if prepared['requirements']['role'] == 'marker' else prepared['parent']['case']
    require(all(reference[key] == original[key] for key in ('path', 'sha256')),
            'official case selector differs from the exact bound parent')
    prepared['book'].file(original)
    require(driver.ordinary.read_case(Path(reference['path'])) == prepared['case'],
            'official case fields differ from the unchanged fidelity case')
    return reference


def runtime_contract(runtime, child, *, working_directory=None):
    current = Path.cwd().absolute() if working_directory is None else Path(working_directory).absolute()
    require(runtime.is_absolute() and runtime == current / 'runtime' and runtime.is_dir()
            and not any(path.is_symlink() for path in (runtime, *runtime.parents)),
            'official runtime must be the existing canonical child of this attempt cwd')
    require(child == runtime / 'fidelity' and not child.exists() and not child.is_symlink(),
            'driver owns one fresh fidelity child; occupied output cannot be reused')
    require(not (runtime / 'progress.jsonl').exists() and not (runtime / 'run-manifest.json').exists(),
            'official progress/provenance already exists; no role retry')


def prepared_progress(path, value):
    raw = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv=None, *, driver=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    os.umask(0o077)
    require(args.source.resolve(strict=True) == ROOT, 'entry bridge must run from its selected component')
    child = args.runtime.absolute() / 'fidelity'
    driver = selected_driver(ROOT) if driver is None else driver
    values = driver_arguments(args, child)
    options = driver.parser().parse_args(values)
    prepared = driver.preflight(options)
    case = case_selector(args.case, prepared, driver)
    if args.check:
        print(json.dumps({'schema': 'stone-dynamic-fidelity-official-bridge-metadata-v1',
            'requirements': prepared['requirements'], 'pending': prepared['pending'],
            'case': case, 'native_imported': False, 'native_execution_authorized': False,
            'source_alone_authorizes_native': False, 'official_runtime': str(args.runtime),
            'driver_output': str(child)}, allow_nan=False))
        return 0
    require(not prepared['pending'] and prepared['phase'] is not None and args.admission is not None,
            'exact fidelity phase and future admission path required')
    runtime_contract(args.runtime.absolute(), child)
    bridge = prepared['source']['files'][ENTRYPOINT]
    require(bridge['path'] == str(ROOT / ENTRYPOINT), 'published component bridge identity differs')
    driver.fidelity.bound(bridge)
    manifest = {'schema': 'pyoomph-runner-manifest-v1', 'created_at': datetime.now(UTC).isoformat(),
        'case_file': case['path'], 'case_path': case['path'], 'case_sha256': case['sha256'],
        'case_id': prepared['case']['case_id'], 'case_fields_used': list(driver.ordinary.CASE_FIELDS_USED),
        'component_commit': prepared['source']['commit'], 'component': prepared['component'],
        'pyoomph': prepared['environment'], 'runner': bridge, 'runner_path': bridge['path'],
        'runner_sha256': bridge['sha256'], 'argv': [sys.executable, bridge['path'], *arguments],
        'azimuthal_stability': False,
        'restart_from': None,
        'qualified_stationary_parent': None,
        'marker_only': True, 'equations_initialised': False, 'stability_layout_initialised': False,
        'constructor_policy': prepared['controls'],
        'inherited_time_step_charge_spent': 1, 'new_time_step_allowance': 0,
        'fidelity_role': args.role, 'fidelity_output': str(child), 'requirements': prepared['requirements'],
        'phase': prepared['phase_ref'], 'native_execution_authorized': False,
        'production_jump_qualified': False, 'nonflat_physical_history_qualified': False,
        'scientific_acceptance': False}
    driver.fidelity.write_new(args.runtime / 'run-manifest.json', manifest)
    prepared_progress(args.runtime / 'progress.jsonl', {'event': 'prepared', 'role': args.role,
        'native_execution_authorized': False, 'active_admission_pending': True,
        'requirements': prepared['requirements'], 'phase': prepared['phase_ref'], 'driver_output': str(child)})
    # Direct call preserves PID and /proc argv. The unchanged driver creates its
    # own child/prepared receipt and waits for the official root active admission.
    return driver.main(values)


if __name__ == '__main__':
    raise SystemExit(main())
