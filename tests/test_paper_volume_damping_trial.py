"""Damping-trial controls with pure fakes and bound metadata; no native imports."""
from __future__ import annotations

import ast
import copy
import importlib.abc
import inspect
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_paper_volume_seed_remesh import FakeProblem, Hooks

from runners import run_paper_volume_damping_trial as driver

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 13, 10, tzinfo=UTC)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyoomph' or fullname.startswith('pyoomph.'):
            raise AssertionError('native import forbidden in damping software tests')


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    assert not any(name.startswith('pyoomph') for name in sys.modules)
    monkeypatch.setattr(sys, 'meta_path', [NoNative(), *sys.meta_path])
    def refuse(*args, **kwargs):
        raise AssertionError('native construction forbidden')
    monkeypatch.setattr(driver.model, '_native_problem_types', refuse)
    monkeypatch.setattr(driver.model, 'build_problem', refuse)


def configuration(template=False):
    return {'schema': driver.SCHEMA + '-configuration', 'simulation': driver.SIMULATION,
            'resource_policy': copy.deepcopy(driver.RESOURCES),
            'phase': {'kind': 'new_numerical_method_trial_same_c7_baseline',
                      'phase_id': driver.SIMULATION + '-attempt-0001',
                      'declared_at_utc': None if template else (NOW - timedelta(hours=1)).isoformat(),
                      'deadline_utc': None if template else (NOW + timedelta(hours=5)).isoformat(),
                      'wall_max_seconds': 21600, 'carried_debits': copy.deepcopy(driver.DEBITS2),
                      'bo_zero_new_update_cap': 30, 'numerical_policy': copy.deepcopy(driver.POLICY)}}


def frozen_counter(problem):
    body = ast.parse(inspect.getsource(driver.model.polish_confined_state)).body[0].body
    counter = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == 'Budget')
    scope = {'__name__': driver.model.__name__, 'GenericProblemHooks': Hooks,
             'stage_limit': 40, 'initial_relaxation': .25, 'problem': problem}
    exec(compile(ast.Module(body=[counter], type_ignores=[]), '<frozen-fake-counter>', 'exec'), scope)  # noqa: S102
    scope['Budget'].__qualname__ = 'polish_confined_state.<locals>.Budget'
    return scope['Budget']()


@pytest.fixture
def factory(tmp_path):
    def create(policy=True):
        mesh, output = tmp_path / 'synthetic.msh', tmp_path / 'runtime'
        mesh.write_bytes(b'synthetic fixture, not native evidence')
        output.mkdir(exist_ok=True)
        events = []
        phase = driver.TrialPhaseBudget(configuration()['phase'],
            lambda event, **values: events.append({'event': event, **values}), [],
            clock=lambda: 0, wall_clock=lambda: NOW)
        cls = driver.seed.budget_problem_type(FakeProblem, Hooks)
        problem = cls(mesh, output)
        release = None
        if policy:
            release = driver.damping_policy_type(Hooks)(phase, lambda p: p.residual)
            problem += release  # Before attach_budget; no unknown-hook carry.
            phase.policy = release
        driver.seed.attach_budget(problem, phase, Hooks)
        phase.label = 'bond_000'
        counter = frozen_counter(problem)
        problem += counter
        for hook in problem._hooks:
            hook.actions_before_newton_solve()
        return SimpleNamespace(p=problem, phase=phase, policy=release, counter=counter,
                               events=events, cls=cls, output=output)
    return create


def step(bundle, residual):
    # The native binding captures the class callback, not an instance method.
    bundle.cls.actions_before_newton_step(bundle.p)
    factor = bundle.p.newton_relaxation_factor
    bundle.p.residual = residual
    reconstruction_factor = bundle.p.newton_relaxation_factor
    for hook in bundle.p._hooks:
        hook.actions_after_newton_step()
    assert reconstruction_factor == factor  # Release follows dof reconstruction.
    return factor


def remesh(bundle, quality):
    bundle.counter.converged = quality
    for hook in bundle.p._hooks:
        hook.actions_before_remeshing([object()])
    for hook in bundle.p._hooks:
        hook.actions_after_remeshing()
    # The owned wrapper records this only after all after-remesh callbacks.
    bundle.p._owned_remesh_receipts = [{'returned': True, 'post_convergence_quality_remesh': quality}]


def release(bundle):
    start = bundle.p.residual
    for i in range(1, 7):
        assert step(bundle, start * .75 ** i) == .25
    assert bundle.policy.released and bundle.p.newton_relaxation_factor == 1.


def test_stage_is_new_and_lineage_carry_is_exact(factory):
    b = factory(policy=False)
    assert (b.phase.updates, b.phase.remeshes, b.phase.remesh_starts, b.counter.stage_updates) == (40, 2, 2, 0)
    assert b.counter.remaining_updates() == 40
    record = b.phase.record()
    assert record['fresh_numerical_budget_allocated'] is record['new_mesh_stage_declared'] is True
    assert record['fresh_numerical_budget_scope'] == 'per_mesh_stage_newton_updates_only'
    assert record['resumed_existing_mesh_stage'] is record['lineage_aggregate_caps_reset'] is False
    assert b.events[0]['stage_updates_at_entry'] == 0


def test_31st_new_bo_zero_step_refuses_after_quality_reset(factory):
    b = factory(policy=False)
    for _ in range(10):
        step(b, 1.)
    remesh(b, quality=True)
    assert b.counter.stage_updates == 0
    for _ in range(20):
        step(b, 1.)
    with pytest.raises(driver.TrialRefusal, match='trial_cap_30_exhausted'):
        step(b, 1.)
    assert b.phase.updates == b.counter.updates == 70 and b.counter.stage_updates == 20


def test_later_stage_keeps_40_and_aggregate_160(factory):
    b = factory(policy=False)
    b.counter.active = False
    b.phase.label = 'bond_001'
    b.counter = frozen_counter(b.p)
    b.p += b.counter
    for _ in range(40):
        step(b, 1.)
    with pytest.raises(ValueError, match='mesh-stage'):
        step(b, 1.)
    b.counter.stage_updates = 0
    b.phase.updates = b.counter.updates = 159
    step(b, 1.)
    with pytest.raises(ValueError, match='Newton budget'):
        step(b, 1.)


def test_only_one_physical_remesh_remains(factory):
    b = factory(policy=False)
    b.phase.after_remesh()
    assert b.phase.remeshes == 2
    remesh(b, quality=True)
    assert (b.phase.remeshes, b.phase.remesh_starts) == (3, 3)
    with pytest.raises(ValueError, match='remesh-start budget'):
        remesh(b, quality=True)


def test_registration_order_and_dynamic_factor_after_budget_entry(factory):
    b = factory()
    assert b.p._hooks[0] is b.policy and b.p._hooks[-1] is b.counter
    assert not hasattr(Hooks, 'actions_before_newton_step')
    assert b.policy.used_factor is None and b.p.newton_relaxation_factor == .25
    release(b)
    assert b.policy.events[0]['trial_update'] == 6
    assert b.phase.updates == b.counter.updates == 46
    assert step(b, 1e-9) == 1.
    b.policy.finish_stage()
    assert b.policy.record()['full_step_predicate_exercised'] is True
    assert len(b.policy.events) == 1


def test_unknown_policy_registration_after_attach_refuses(factory):
    b = factory(policy=False)
    policy = driver.damping_policy_type(Hooks)(b.phase, lambda p: p.residual)
    with pytest.raises(ValueError, match='unknown or ambiguous'):
        b.p += policy


def test_window_does_not_release_above_entry_and_deadline_is_debited(factory):
    b = factory()
    entry = b.p.residual
    step(b, entry * 1000.)
    for i in range(1, 20):
        step(b, entry * 1000. * .75 ** i)
    assert b.policy.pending_refusal['name'] == 'no_fixed_rate_regime_within_20_updates'
    assert not b.policy.released and b.phase.updates == b.counter.updates == 60
    with pytest.raises(driver.TrialRefusal, match='within_20'):
        step(b, entry)


@pytest.mark.parametrize('refusal', ['growth', 'six_full'])
def test_full_step_refusal_is_deferred_until_after_debit(factory, refusal):
    b = factory()
    release(b)
    if refusal == 'growth':
        step(b, b.p.residual * 2)
        name = 'full_step_residual_growth'
        expected = 47
    else:
        for _ in range(6):
            step(b, b.p.residual * .9)
        name = 'full_step_tolerance_not_reached_within_6_updates'
        expected = 52
    assert b.phase.updates == b.counter.updates == expected
    with pytest.raises(driver.TrialRefusal, match=name):
        b.policy.finish_stage()  # Even if no further native step was requested.


def test_roundoff_growth_below_tolerance_and_early_tolerance_are_allowed(factory):
    b = factory()
    release(b)
    step(b, 1e-9)
    step(b, 2e-9)
    assert b.policy.pending_refusal is None
    early = factory()
    step(early, 1e-9)
    early.policy.finish_stage()
    assert early.policy.initial_regime_discharged and not early.policy.full_step_predicate_exercised
    assert early.policy.record()['H1_control_not_exercised'] is True


@pytest.mark.parametrize('fault', ['residual', 'interrupt', 'event'])
def test_policy_callback_failure_never_skips_a_completed_update(factory, fault):
    b = factory()
    if fault == 'event':
        release_value = b.p.residual
        for i in range(1, 6):
            step(b, release_value * .75 ** i)
        def fail(*args, **kwargs):
            if args[0] == 'damping_released':
                raise OSError('synthetic event refusal')
        b.phase.emit = fail
    else:
        def fail(*args):
            raise KeyboardInterrupt('synthetic interrupt') if fault == 'interrupt' else RuntimeError('synthetic residual refusal')
        # Entry sampling is permitted; refusal occurs only in the after-step read.
        b.cls.actions_before_newton_step(b.p)
        b.policy.residual = fail
    for hook in b.p._hooks:
        if fault == 'event':
            b.p.residual = release_value * .75 ** 6
        hook.actions_after_newton_step()
    assert b.phase.updates == b.counter.updates == 40 + b.policy.new_updates
    assert b.policy.pending_refusal['name'] == 'policy_callback_refusal'
    with pytest.raises(driver.TrialRefusal, match='policy_callback_refusal'):
        b.cls.actions_before_newton_step(b.p)


@pytest.mark.parametrize('full', [False, True])
def test_inversion_waits_for_owned_receipt_before_next_solve(factory, full):
    b = factory()
    if full:
        release(b)
        step(b, b.p.residual * .9)
    else:
        step(b, b.p.residual * .9)
    remesh(b, quality=False)  # This must return; no policy exception before the receipt.
    name = 'full_step_left_mesh_validity' if full else 'damped_step_left_mesh_validity_at_0.25'
    with pytest.raises(driver.TrialRefusal, match=name):
        b.policy.actions_before_newton_solve()
    assert b.policy.pending_refusal['owned_remesh_receipt_preserved'] is True
    assert b.phase.remeshes == b.counter.remeshes == 3


def test_quality_reset_preserves_discharged_initial_deadline_and_total_cap(factory):
    b = factory()
    release(b)
    remesh(b, quality=True)
    assert not b.policy.released and b.policy.initial_regime_discharged
    assert b.policy.new_updates == 6 and b.policy.mesh_updates == 0
    for hook in b.p._hooks:
        hook.actions_before_newton_solve()
    for _ in range(24):
        step(b, 1.)
    assert b.policy.pending_refusal is None  # No new cumulative20 permission or unmet deadline.
    with pytest.raises(driver.TrialRefusal, match='trial_cap_30_exhausted'):
        step(b, 1.)


def test_inversion_after_last_remesh_is_named_without_inventing_a_receipt(factory):
    b = factory()
    release(b)
    remesh(b, quality=True)
    for hook in b.p._hooks:
        hook.actions_before_newton_solve()
    release(b)
    step(b, b.p.residual * .9)
    with pytest.raises(ValueError, match='remesh-start budget'):
        remesh(b, quality=False)
    assert b.policy.pending_refusal['name'] == 'full_step_left_mesh_validity'
    assert b.policy.pending_refusal['owned_remesh_receipt_required'] is False
    windows = b.policy.mesh_index
    b.counter.active = False
    for hook in b.p._hooks:
        hook.actions_after_remeshing()  # The frozen helper's rollback notification.
    assert b.policy.mesh_index == windows and b.phase.remeshes == 3


def test_policy_refusal_uses_frozen_full_snapshot_restore(factory, monkeypatch):
    import numpy as np
    b = factory()
    b.counter.active = False
    b.phase.first_carry = False
    saved = driver.seed.snapshot(b.p)
    settings = {name: getattr(b.p, name) for name in driver.seed.SETTINGS}
    flags = {name: getattr(b.p, name) for name in driver.seed.FLAGS}
    monkeypatch.setattr(driver.model, '_stationary_residual', lambda p: p.residual)
    monkeypatch.setattr(driver.model, '_stationary_state_signature', lambda p, az: p.signature())
    monkeypatch.setattr(driver.diagnostic, 'actual_pins', lambda p: {'pins': p.pins.copy()})
    tree = ast.parse(inspect.getsource(driver.model.polish_confined_state))
    function = tree.body[0]
    function.body = [node for node in function.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    scope = {**driver.model.__dict__, 'np': np, 'GenericProblemHooks': Hooks,
             '_stationary_residual': lambda p: p.residual}
    exec(compile(tree, '<frozen-fake-polish>', 'exec'), scope)  # noqa: S102
    monkeypatch.setattr(driver.model, 'polish_confined_state', scope['polish_confined_state'])
    def solve(**kwargs):
        for hook in b.p._hooks:
            hook.actions_before_newton_solve()
        b.cls.actions_before_newton_step(b.p)
        b.p.fields, b.p.pins = [9., 10.], [True, False]
        b.policy.residual = lambda p: (_ for _ in ()).throw(RuntimeError('synthetic read refusal'))
        for hook in b.p._hooks:
            hook.actions_after_newton_step()
        b.cls.actions_before_newton_step(b.p)
    b.p.solve = solve
    case = driver.model.read_case(ROOT / 'validation/herradaEtAl_paper_model/cases/h15-bo1p56-volume3p8.json')
    with pytest.raises(driver.TrialRefusal):
        driver.seed.polish_stage(b.p, case, b.output, b.phase, 'bond_000', .25)
    assert driver.history.normalized_bytes(saved, Path.cwd()) == driver.history.normalized_bytes(driver.seed.snapshot(b.p), Path.cwd())
    assert all(getattr(b.p, name) == value for name, value in {**settings, **flags}.items())
    assert driver.runner._read_json(b.output / 'bond_000-full-rollback.json')['verified'] is True
    assert b.phase.updates == b.phase.current.updates == 41


def test_refusal_on_returned_solve_last_step_restores_before_checkpoint(factory, monkeypatch):
    b = factory()
    monkeypatch.setattr(driver.model, '_stationary_state_signature', lambda p, az: p.signature())
    monkeypatch.setattr(driver.diagnostic, 'actual_pins', lambda p: {'pins': p.pins.copy()})
    saved = driver.seed.snapshot(b.p)
    def returned(*args):
        b.cls.actions_before_newton_step(b.p)
        b.p.fields, b.p.pins = [9., 10.], [True, False]
        b.policy.residual = lambda p: (_ for _ in ()).throw(RuntimeError('synthetic last-step refusal'))
        for hook in b.p._hooks:
            hook.actions_after_newton_step()
        b.counter.active = False
        return {'status': 'converged'}
    monkeypatch.setattr(driver.seed, 'polish_stage', returned)
    with pytest.raises(driver.TrialRefusal, match='policy_callback_refusal'):
        driver.polish_trial_stage(b.p, {}, b.output, b.phase, b.policy)
    assert driver.history.normalized_bytes(saved, Path.cwd()) == driver.history.normalized_bytes(driver.seed.snapshot(b.p), Path.cwd())
    assert b.policy.post_solve_rollback['verified'] is True
    assert b.phase.updates == b.counter.updates == 41


@pytest.mark.parametrize('change', ['naive', 'too_long', 'expired', 'debt', 'policy', 'undeclared_native'])
def test_declared_clock_and_policy_changes_refuse(change):
    cfg = configuration()
    if change == 'naive':
        cfg['phase']['declared_at_utc'] = '2026-10-13T09:00:00'
    elif change == 'too_long':
        cfg['phase']['deadline_utc'] = (NOW + timedelta(hours=6)).isoformat()
    elif change == 'expired':
        cfg['phase']['deadline_utc'] = NOW.isoformat()
    elif change == 'debt':
        cfg['phase']['carried_debits']['updates_total'] = 0
    elif change == 'policy':
        cfg['phase']['numerical_policy']['consecutive_ratios'] = 5
    else:
        cfg = configuration(template=True)
    with pytest.raises(ValueError):
        driver.phase_authority(cfg, False, (NOW - timedelta(hours=2)).isoformat(), NOW)


def test_old_clocks_and_source_are_preserved():
    for name in driver.runner.component_revision(ROOT)['sources']:
        assert (ROOT / name).read_bytes() == driver.runner._git(ROOT, 'show', f'{driver.METHODS_BASE}:{name}')
    assert (ROOT / driver.previous.ENTRYPOINT).read_bytes() == driver.runner._git(ROOT, 'show', f'{driver.BASE}:{driver.previous.ENTRYPOINT}')
    assert driver.old.DEADLINE == '2026-10-13T04:41:19Z'
    with pytest.raises(ValueError):
        driver.old.remaining_wall(NOW)


@pytest.mark.parametrize('clock', ['absolute', 'monotonic'])
def test_independent_trial_deadline_refusals(clock):
    wall, tick = [NOW], [0]
    phase = driver.TrialPhaseBudget(configuration()['phase'], lambda *a, **k: None, [],
        clock=lambda: tick[0], wall_clock=lambda: wall[0])
    if clock == 'absolute':
        wall[0] = datetime.fromisoformat(phase.authority['deadline_utc'])
    else:
        tick[0] = phase.wall_seconds
    with pytest.raises(ValueError, match='deadline|wall budget'):
        phase.check_wall()


@pytest.mark.parametrize('wrapped', [False, True])
def test_failure_json_preserves_policy_refusal_and_all_no_claims(tmp_path, monkeypatch, wrapped):
    own = {'path': str(ROOT / driver.ENTRYPOINT), 'sha256': driver.sha(ROOT / driver.ENTRYPOINT)}
    prepared = {'output': str(tmp_path / 'runtime'), 'phase': configuration()['phase'], 'lineage': {},
                'implementation': {driver.ENTRYPOINT: own}, 'inputs': [], 'original_failure': {},
                'trial_policy_receipt': {'trial_refusal': {'name': 'full_step_residual_growth'}}}
    monkeypatch.setattr(driver, 'preflight', lambda args: prepared)
    monkeypatch.setattr(driver, 'stable_inputs', lambda prepared: None)
    monkeypatch.setattr(driver.seed, 'manifest', lambda *args: {})
    def refuse(*args):
        raise RuntimeError('OomphException') if wrapped else driver.TrialRefusal('full_step_residual_growth')
    monkeypatch.setattr(driver, 'execute', refuse)
    with pytest.raises(RuntimeError):
        driver.main(['--source', str(ROOT), '--case', 'unused', '--configuration', 'unused',
                     '--out', str(tmp_path / 'runtime')])
    failure = driver.runner._read_json(tmp_path / 'runtime/failure.json')
    assert failure['failure_class'] == 'phase-budget-refusal'
    assert failure['trial_refusal']['name'] == 'full_step_residual_growth'
    assert failure['numerical_acceptance'] is failure['completion_reached'] is False
    assert all(failure[name] == value for name, value in driver.FLAGS.items())


@pytest.fixture
def admitted(tmp_path):
    path = os.environ.get('STONE_VOLUME_DAMPING_CONFIGURATION')
    if path is None:
        pytest.skip('actual metadata admission needs external prospective configuration')
    cfg = driver.runner._read_json(Path(path))
    expired = driver.runner._read_json(Path(cfg['expired_resume_configuration']['path']))
    probe = driver.runner._read_json(Path(expired['probe_configuration']['path']))
    args = SimpleNamespace(source=ROOT, case=Path(probe['bindings']['case']['path']),
                           configuration=Path(path), out=tmp_path / 'never-native', check=True)
    return driver.preflight(args)


def test_actual_seed_c1_prior_failure_custody_and_template_admission(admitted):
    assert admitted['restart']['state']['sha256'] == driver.STATE_SHA
    assert admitted['lineage']['original']['C1_half_sum_values_verified'] == 82257
    assert admitted['prior']['prior_stage_updates'] == 40
    assert admitted['phase_declaration_materialised'] is admitted['native_execution_authorized'] is False
    assert admitted['parent_is_unqualified'] is True and admitted['fresh_numerical_budget_allocated'] is True
    assert not Path(admitted['output']).exists()
    for record in admitted['prior']['prior_stage_receipts'].values():
        with pytest.raises(ValueError):
            driver.bind({**record, 'sha256': '0' * 64}, [])


@pytest.mark.parametrize('fault', ['counter', 'acceptance', 'restore', 'supervisor'])
def test_actual_prior_receipt_semantic_tampering_refuses(admitted, fault):
    runtime = Path(admitted['prior']['runtime'])
    values = [driver.runner._read_json(path) for path in (runtime / 'failure.json', runtime / 'phase-budget.json',
        runtime / 'polish/bond_000/polish.json', runtime / 'bond_000-full-rollback.json', runtime.parent / 'supervisor-result.json')]
    if fault == 'counter':
        values[1]['updates_total'] = 0
    elif fault == 'acceptance':
        values[0]['numerical_acceptance'] = True
    elif fault == 'restore':
        values[3]['controls_restored'] = False
    else:
        values[4]['owned_processes_terminal'] = False
    with pytest.raises(ValueError):
        driver.validate_prior_receipts(*values)


def test_bound_printed_ratio_replay_releases_at_reviewed_window(admitted, factory):
    log = (Path(admitted['prior']['runtime']).parent / 'runner.log').read_text()
    part = log.split('Initial Maximum residuals ')[-1]
    values = re.findall(r'Newton Step \d+: Maximum residuals ([0-9.e+-]+)', part)
    assert len(values) == 34
    b = factory()
    b.p.residual = float(part.splitlines()[0])
    for value in values:
        step(b, float(value))
        if b.policy.released:
            break
    assert b.policy.events[0]['trial_update'] == 15
    assert all(abs(r - .75) <= .03 * .75 for r in b.policy.events[0]['ratios'])
