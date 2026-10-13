"""One declared damping-policy trial at the unchanged c7 baseline.

Allocate a new mesh-stage allowance while retaining the closed lineage's
aggregate work. A numerical refusal makes no stationary-nonexistence claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import run_paper_volume_post_c1 as previous

old, seed = previous.old, previous.seed
model, runner, history, diagnostic, remesh = previous.model, previous.runner, previous.history, previous.diagnostic, previous.remesh
owned, c1 = previous.owned, previous.c1
require, sha, write_json, bind = previous.require, previous.sha, previous.write_json, previous.bind
ENTRYPOINT = 'runners/run_paper_volume_damping_trial.py'
BASE = 'ae17f237295c6093a0de447e51104783f250f76e'
METHODS_BASE = previous.BASE
SCHEMA = 'herrada-volume-damping-trial-v1'
SIMULATION = 'paper-model-volume3p8-damping-trial-medium'
PRIOR_SIMULATION = previous.SIMULATION
DEBITS = previous.DEBITS
DEBITS2 = {'updates_total': 40, 'physical_remeshes_total': 2, 'physical_remesh_starts': 2,
           'prior_stage_updates': 40}
RESOURCES, NO_CLAIMS = previous.RESOURCES, {**previous.NO_CLAIMS,
    'stationary_nonexistence_claimed': False, 'undamped_basin_certified': False,
    'stationary_existence_certified': False, 'unique_stationary_state_certified': False}
WALL_MAX, TRIAL_CAP = 21600, 30
POLICY = {'initial_relaxation': .25, 'released_relaxation': 1., 'ratio_target': .75,
          'ratio_relative_tolerance': .03, 'consecutive_ratios': 6, 'initial_regime_deadline_updates': 20,
          'full_step_update_limit': 6, 'bo_zero_new_update_cap': TRIAL_CAP,
          'initial_regime_deadline_scope': 'cumulative_until_first_release_or_stationary_tolerance',
          'stationary_tolerance': 1e-8,
          'hypothesis_scope': 'numerical_control_on_one_discrete_state_only'}
FLAGS = {'fresh_numerical_budget_allocated': True,
         'fresh_numerical_budget_scope': 'per_mesh_stage_newton_updates_only',
         'new_mesh_stage_declared': True, 'numerical_method_changed': True,
         'resumed_existing_mesh_stage': False, 'prior_mesh_stage_exhausted': True,
         'lineage_aggregate_caps_reset': False, 'parent_is_unqualified': True, **NO_CLAIMS}
EXPIRED_CONFIG_SHA = previous.EXPIRED_CONFIG_SHA
PROBE_CONFIG_SHA, PROBE_SHA, PROBE_COMPLETION_SHA = previous.PROBE_CONFIG_SHA, previous.PROBE_SHA, previous.PROBE_COMPLETION_SHA
LOAD_SHA, STATE_SHA = previous.LOAD_SHA, previous.STATE_SHA


def utc_now():
    return datetime.now(UTC)


utc = previous.utc
stable_inputs, load_retained = previous.stable_inputs, previous.load_retained


def phase_authority(configuration, metadata_only, terminal_time, now=None):
    now = utc_now() if now is None else now
    phase = configuration['phase']
    require(configuration['schema'] == SCHEMA + '-configuration'
            and configuration['simulation'] == SIMULATION and configuration['resource_policy'] == RESOURCES,
            'damping-trial configuration or resource policy differs')
    require(phase['kind'] == 'new_numerical_method_trial_same_c7_baseline'
            and isinstance(phase['phase_id'], str)
            and re.fullmatch(r'[a-zA-Z0-9._-]{1,96}', phase['phase_id'])
            and phase['phase_id'].startswith(SIMULATION + '-')
            and phase['carried_debits'] == DEBITS2
            and all(type(value) is int for value in phase['carried_debits'].values())
            and type(phase['wall_max_seconds']) is int and phase['wall_max_seconds'] == WALL_MAX
            and type(phase['bo_zero_new_update_cap']) is int and phase['bo_zero_new_update_cap'] == TRIAL_CAP
            and phase['numerical_policy'] == POLICY
            and all(type(phase['numerical_policy'][key]) is type(value) for key, value in POLICY.items()),
            'damping trial declaration, carried work or policy differs')
    if phase['declared_at_utc'] is None and phase['deadline_utc'] is None:
        require(metadata_only, 'native damping trial requires a materialised UTC declaration')
        return copy.deepcopy(phase), None
    declared, deadline = utc(phase['declared_at_utc']), utc(phase['deadline_utc'])
    require(utc(terminal_time) < declared <= now
            and 0 < (deadline - declared).total_seconds() <= WALL_MAX,
            'trial declaration must follow terminal readback and have at most 21600 seconds')
    remaining = (deadline - now).total_seconds()
    require(0 < remaining <= WALL_MAX, 'damping-trial absolute phase deadline reached')
    return copy.deepcopy(phase), remaining


class TrialRefusal(RuntimeError):
    pass


class TrialPhaseBudget(seed.PhaseBudget):
    """Retain aggregate debt and explicitly declare one new trial stage."""

    def __init__(self, authority, emit, stages, clock=time.monotonic, wall_clock=utc_now):
        self.authority, self.wall_clock = copy.deepcopy(authority), wall_clock
        self.deadline = utc(authority['deadline_utc'])
        remaining = (self.deadline - wall_clock()).total_seconds()
        require(0 < remaining <= WALL_MAX, 'damping-trial absolute phase deadline reached')
        super().__init__(remaining, emit, clock)
        self.updates, self.remeshes, self.remesh_starts = 40, 2, 2
        self.stages, self.first_carry = copy.deepcopy(stages), True
        self.policy = None
        self.bo_zero_updates, self.trial_stage_updates = 0, 0

    def check_wall(self):
        require(self.wall_clock() < self.deadline, 'damping-trial absolute phase deadline reached')
        super().check_wall()

    def carry(self, hook):
        if not self.first_carry:
            return super().carry(hook)
        require(hook.__class__.__module__ == model.__name__
                and hook.__class__.__qualname__ == 'polish_confined_state.<locals>.Budget',
                'unknown or ambiguous polish budget hook')
        require(self.current is None and hook.active is True
                and hook.updates == hook.remeshes == hook.stage_updates == 0
                and not hook.trials and not hook.remeshing and not hook.converged,
                'new trial polish hook was already used')
        self.check_wall()
        self.audit()
        hook.updates, hook.remeshes, hook.stage_updates = 40, 2, 0
        self.current, self.first_carry = hook, False
        item = {'label': self.label, 'carried_updates': 40, 'carried_physical_remeshes': 2,
                'stage_updates_at_entry': 0, 'resumed_existing_mesh_stage': False,
                'new_mesh_stage_declared': True, 'numerical_method_changed': True}
        self.stages.append(item)
        self.emit('polish_budget_carried', **item)
        require(hook.remaining_updates() == 40, 'new trial mesh-stage allowance differs')

    def before_update(self):
        self.check_wall()
        self.audit()
        if self.policy is not None:
            self.policy.refuse_pending()
        if self.label == 'bond_000' and self.updates - DEBITS2['updates_total'] >= TRIAL_CAP:
            if self.policy is not None:
                self.policy.pending('trial_cap_30_exhausted')
                self.policy.refuse_pending()
            raise TrialRefusal('trial_cap_30_exhausted')
        if self.policy is not None:
            self.policy.before_step()
        super().before_update()

    def before_remesh(self, remeshers):
        require(self.remesh_starts < 3, 'lineage physical remesh-start budget exhausted')
        super().before_remesh(remeshers)

    def finish_bo_zero(self):
        self.bo_zero_updates = self.updates - DEBITS2['updates_total']
        self.trial_stage_updates = self.current.stage_updates
        require(self.bo_zero_updates <= TRIAL_CAP, 'trial_cap_30_exhausted')

    def record(self):
        bo_zero = self.updates - 40 if self.label == 'bond_000' else self.bo_zero_updates
        stage_updates = self.current.stage_updates if self.label == 'bond_000' and self.current else self.trial_stage_updates
        return {**super().record(), **FLAGS,
                'lineage_updates_total': self.updates, 'lineage_physical_remeshes_total': self.remeshes,
                'lineage_physical_remesh_starts': self.remesh_starts,
                'prior_mesh_stage_updates': 40, 'trial_stage_updates': stage_updates,
                'trial_new_bo_zero_updates': bo_zero, 'bo_zero_new_update_cap': TRIAL_CAP,
                'original_phase_debits': DEBITS2, 'absolute_deadline_utc': self.authority['deadline_utc'],
                'declared_phase': self.authority, 'trial_schedule': POLICY, 'new_wall_authority_allocated': True,
                'remaining_wall_seconds': max(0, (self.deadline - self.wall_clock()).total_seconds())}


def damping_policy_type(hooks_base):
    class DampingRelease(hooks_base):
        """Observe after updates; defer refusals until completed work is debited."""

        def __init__(self, phase, residual=model._stationary_residual):
            super().__init__()
            self.phase, self.residual = phase, residual
            self.events, self.observations, self.callback_errors, self.remesh_observations = [], [], [], []
            self.pending_refusal, self.remeshing = None, False
            self.new_updates, self.mesh_index = 0, 0
            self.initial_regime_discharged = self.full_step_predicate_exercised = False
            self.reset_mesh()

        def reset_mesh(self):
            self.entry_residual = self.previous_residual = self.used_factor = None
            self.mesh_updates = self.full_updates = 0
            self.ratios, self.released = [], False

        def enabled(self):
            return (self.phase.label == 'bond_000' and self.phase.current is not None
                    and self.phase.current.active)

        def pending(self, name, **values):
            if self.pending_refusal is None:
                self.pending_refusal = {'name': name, 'trial_update': self.new_updates,
                                        'mesh_index': self.mesh_index, **values}

        def refuse_pending(self):
            if self.pending_refusal is not None:
                if self.pending_refusal.get('owned_remesh_receipt_required'):
                    receipts = getattr(self.get_problem(), '_owned_remesh_receipts', [])
                    require(receipts and receipts[-1]['returned'] is True
                            and receipts[-1]['post_convergence_quality_remesh'] is False,
                            'trial refusal lacks the owned inversion-remesh receipt')
                    self.pending_refusal['owned_remesh_receipt_preserved'] = True
                raise TrialRefusal(self.pending_refusal['name'])

        def actions_before_newton_solve(self):
            self.refuse_pending()
            self.used_factor = None  # The frozen Budget applies the factor after this callback.

        def before_step(self):
            self.refuse_pending()
            if not self.enabled():
                return
            problem = self.get_problem()
            self.used_factor = float(problem.newton_relaxation_factor)
            require(self.used_factor == (1. if self.released else .25), 'trial damping factor changed outside policy')
            if self.entry_residual is None:
                value = float(self.residual(problem))
                require(math.isfinite(value) and value >= 0, 'nonfinite trial entry residual')
                self.entry_residual = self.previous_residual = value

        def event(self, event, **values):
            record = {'event': event, **values}
            self.events.append(record)
            self.phase.emit(event, **values)

        def actions_after_newton_step(self):
            if not self.enabled():
                return
            # This hook precedes Guard and Budget. Never propagate an error here:
            # the native class-level before-step path checks it after both debits.
            self.new_updates += 1
            self.mesh_updates += 1
            try:
                problem = self.get_problem()
                value = float(self.residual(problem))
                require(math.isfinite(value) and value >= 0, 'nonfinite trial residual')
                prior = self.previous_residual
                ratio = value / prior if prior and prior > 0 else None
                observation = {'trial_update': self.new_updates, 'mesh_update': self.mesh_updates,
                               'mesh_index': self.mesh_index, 'residual_linf': value, 'ratio': ratio,
                               'factor_used': self.used_factor, 'lineage_update_after_debit': self.phase.updates + 1}
                self.observations.append(observation)
                if value <= POLICY['stationary_tolerance']:
                    self.initial_regime_discharged = True
                if self.used_factor == 1.:
                    self.full_updates += 1
                    self.full_step_predicate_exercised = True
                    if prior is not None and value > prior and value > POLICY['stationary_tolerance']:
                        self.pending('full_step_residual_growth', residual_linf=value, previous_residual=prior)
                    if self.full_updates >= 6 and value > POLICY['stationary_tolerance']:
                        self.pending('full_step_tolerance_not_reached_within_6_updates', residual_linf=value)
                else:
                    self.ratios.append(ratio)
                    self.ratios = self.ratios[-6:]
                    steady_ratio = (len(self.ratios) == 6 and all(r is not None and abs(r - .75) <= .03 * .75
                                                               for r in self.ratios))
                    if not self.released and steady_ratio and value < self.entry_residual:
                        problem.newton_relaxation_factor = 1.
                        self.released = True
                        self.initial_regime_discharged = True
                        self.event('damping_released', trial_update=self.new_updates, mesh_update=self.mesh_updates,
                                   mesh_index=self.mesh_index, residual_linf=value,
                                   ratios=list(self.ratios), factor_before=.25, factor_after=1.)
                    if not self.initial_regime_discharged and self.new_updates >= 20:
                        self.pending('no_fixed_rate_regime_within_20_updates')
                self.previous_residual = value
            except BaseException as error:  # noqa: BLE001 - debit the completed native step before refusing
                self.callback_errors.append({'error_type': type(error).__name__, 'error': str(error),
                                             'trial_update': self.new_updates})
                self.pending('policy_callback_refusal', error_type=type(error).__name__, error=str(error))

        def actions_before_remeshing(self, active_remeshers):
            if not self.enabled():
                return
            self.remeshing = True
            self.inversion = not self.phase.current.converged
            self.remesh_factor = self.used_factor
            if self.inversion and self.phase.remesh_starts >= 3:
                name = 'full_step_left_mesh_validity' if self.remesh_factor == 1. else 'damped_step_left_mesh_validity_at_0.25'
                self.pending(name, physical_remesh_refused_by_lineage_cap=True,
                             owned_remesh_receipt_required=False)

        def actions_after_remeshing(self):
            if not self.remeshing:
                return  # Loading/restoring a snapshot is not a physical remesh.
            self.remeshing = False
            if not self.phase.pending_remesh or not self.phase.current.active:
                return  # A refused/incomplete remake followed by rollback is not a new mesh.
            inversion, factor = self.inversion, self.remesh_factor
            self.remesh_observations.append({'mesh_index_before': self.mesh_index, 'factor_used': factor,
                                            'inversion': inversion, 'released_before_remesh': self.released,
                                            'trial_update': self.new_updates})
            self.mesh_index += 1
            self.reset_mesh()
            if inversion:
                name = 'full_step_left_mesh_validity' if factor == 1. else 'damped_step_left_mesh_validity_at_0.25'
                self.pending_refusal = {'name': name, 'trial_update': self.new_updates,
                                        'mesh_index': self.mesh_index, 'owned_remesh_receipt_required': True}
            # Raising here would skip OwnedRemeshProblem's physical receipt.

        def finish_stage(self):
            self.refuse_pending()  # Also catches refusal on the last step of a returned solve.
            require(self.new_updates == self.phase.updates - 40, 'policy and frozen completed-update debits disagree')
            if self.new_updates == 0:
                self.initial_regime_discharged = self.residual(self.get_problem()) <= POLICY['stationary_tolerance']

        def record(self):
            return {'trial_schedule': POLICY, 'trial_new_bo_zero_updates_observed': self.new_updates,
                    'peak_observed_residual_linf': max((row['residual_linf'] for row in self.observations), default=None),
                    'damping_released_events': list(self.events), 'residual_observations': list(self.observations),
                    'physical_remesh_observations': list(self.remesh_observations),
                    'trial_refusal': self.pending_refusal, 'callback_errors': list(self.callback_errors),
                    'initial_regime_deadline_updates': 20, 'initial_regime_deadline_discharged': self.initial_regime_discharged,
                    'full_step_predicate_exercised': self.full_step_predicate_exercised,
                    'H1_control_not_exercised': not self.full_step_predicate_exercised,
                    'post_solve_policy_full_restore': getattr(self, 'post_solve_rollback', None),
                    'mesh_window_updates': self.mesh_updates, 'full_updates_after_release': self.full_updates,
                    'hypothesis_scope': POLICY['hypothesis_scope'], **NO_CLAIMS}

    return DampingRelease


def validate_prior_receipts(failure, budget, polish, rollback, supervisor):
    require(failure['status'] == polish['status'] == 'failed'
            and failure['numerical_acceptance'] is False and failure['completion_reached'] is False
            and failure['stationary_nonexistence_claimed'] is False and failure['parent_is_unqualified'] is True
            and failure['inputs_unchanged'] is True and polish['rollback_verified'] is True
            and rollback == {'controls_restored': True, 'full_stream_and_actual_pins': True, 'verified': True},
            'prior failed stage, strict full restore or unqualified status differs')
    require(all(type(budget[key]) is int and budget[key] == value
                for key, value in DEBITS2.items() if key != 'prior_stage_updates')
            and budget['budget_hooks_agree'] is True and budget['physical_remesh_pending'] is False
            and budget['fresh_numerical_budget_allocated'] is False
            and budget['original_phase_debits'] == DEBITS and budget['bo_zero_new_update_cap'] == 37
            and (polish['updates_total'], polish['remeshes_total'], polish['stage_updates']) == (40, 2, 40)
            and all(type(polish[key]) is int for key in ('updates_total', 'remeshes_total', 'stage_updates')),
            'prior lineage or exhausted stage debit differs')
    require([(row['relaxation'], row['remaining_updates'], row['stage_updates_before'], row['updates_total_before'])
             for row in polish['trials']] == [(.5, 37, 3, 3), (.25, 34, 6, 6)]
            and polish['parameters'] == {'Ca_in': .02, 'Bo': 0.}, 'prior damping trials or forcing differ')
    require(supervisor['status'] == supervisor['phase_status'] == 'inconclusive'
            and supervisor['adapter_exit_code'] == 0 and supervisor['administrative_complete'] is True
            and supervisor['native_execution_complete'] is False and supervisor['owned_processes_terminal'] is True
            and supervisor['terminal_protocol_verified'] is True and supervisor['manifest_validation'] is True
            and supervisor['reported_numerical_acceptance'] is False
            and supervisor['native_failure'] == failure and supervisor['phase_budget'] == budget,
            'prior native supervisor or terminal receipt differs')


def bind_prior(configuration, records):
    roots = {}
    for key, expected in PRIOR_ROOT_HASHES.items():
        path = bind(configuration[key], records)
        require(sha(path) == expected, 'reviewed prior root receipt differs: ' + key)
        roots[key] = runner._read_json(path)
    terminal, whole, archive, release = (roots[key] for key in
        ('prior_terminal', 'prior_whole_run_terminal', 'prior_archive', 'prior_release'))
    require(terminal['schema'] == 'stone-volume-post-C1-terminal-root-v1'
            and terminal['native_pid_absent'] is terminal['supervisor_pid_absent'] is terminal['cgroup_absent'] is True
            and terminal['properties']['MainPID'] == '0' and terminal['properties']['ActiveState'] == 'failed'
            and terminal['reported_numerical_acceptance'] is False and terminal['total_updates'] == 40
            and terminal['total_physical_remeshes'] == terminal['physical_remesh_starts'] == 2,
            'prior root terminal lineage differs')
    original = Path(archive['source'])
    require(original.name == PRIOR_SIMULATION and whole['run_path'] == str(original)
            and whole['owned_cwd_candidates'] == []
            and all(item['properties']['MainPID'] == '0' and item['properties']['ActiveState'] in ('inactive', 'failed')
                    for item in whole['units']), 'prior whole-run idle readback differs')
    require(archive['simulation'] == PRIOR_SIMULATION and archive['attempt'] == 'attempt-0002'
            and archive['file_count'] == len(archive['files']) == 98
            and archive['bytes'] == sum(item['bytes'] for item in archive['files'].values()) == 80096993
            and archive['source_deleted'] is False and archive['protocol_files_verified_at'],
            'prior verified cumulative archive differs')
    for name, item in archive['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe prior archive path')
        path = bind({'path': str(original / name), 'sha256': item['sha256']}, records)
        require(path.stat().st_size == item['bytes'], 'prior retained local file bytes differ')
    for name, item in terminal['files'].items():
        require(archive['files'][name] == item, 'terminal and cumulative archive bindings disagree')
    require(release['returncode'] == 0 and '--require-idle' in release['argv']
            and '--require-state-closeout' in release['argv']
            and whole['generation_sha256'] in release['argv']
            and release['stdout'].startswith('RELEASED ' + original.parent.name + '/' + PRIOR_SIMULATION + '\n'),
            'prior reservation release differs')
    runtime = original / 'attempts/attempt-0002/runtime'
    failure, budget, polish, rollback, supervisor = (runner._read_json(path) for path in
        (runtime / 'failure.json', runtime / 'phase-budget.json', runtime / 'polish/bond_000/polish.json',
         runtime / 'bond_000-full-rollback.json', runtime.parent / 'supervisor-result.json'))
    validate_prior_receipts(failure, budget, polish, rollback, supervisor)
    require(not any((runtime / name).exists() for name in ('result.json', 'summary.json', 'completion.json')),
            'prior failed parent unexpectedly qualified')
    manifest = runner._read_json(runtime / 'run-manifest.json')
    require(manifest['component_commit'] == BASE and manifest['restart_from']['sha256'] == STATE_SHA,
            'prior source or primary c7 seed differs')
    load = runner._read_json(runtime / 'retained-load/load.json')
    require(load['verified'] is True and load['snapshot_restore']['verified'] is True
            and load['strict_post_preparation_stream_gate_unchanged'] is True
            and load['original_snapshot_gate_unchanged'] is True
            and load['preparation']['half_sum_values_verified'] == 82257,
            'prior c7 C1 preparation or strict restore differs')
    progress = [json.loads(line) for line in (runtime / 'progress.jsonl').read_text().splitlines()]
    updates = [row for row in progress if row['event'] == 'newton_update']
    require([row['phase_updates'] for row in updates] == list(range(4, 41))
            and sum(row['event'] == 'physical_remesh_completed' for row in progress) == 1,
            'prior completed work progress differs')
    return {'runtime': str(runtime), 'run_root': str(original), 'terminal_time': terminal['verified_at'],
            'roots': {key: configuration[key] for key in PRIOR_ROOT_HASHES},
            'prior_stage_receipts': {name: {'path': str(original / name), **item}
                                     for name, item in terminal['files'].items()},
            'prior_trials': polish['trials'], 'prior_stage_updates': 40, 'prior_stages': budget['stages'],
            'prior_phase_deadline_utc': budget['absolute_deadline_utc'],
            'prior_archive_custody': {'historical_root_verified_receipt': True,
                                     'destination_verified_now': False, 'receipt': configuration['prior_archive']},
            'fixed_factor_extrapolation_from_discarded_iterate': roots['numerical_control_evidence'][
                'additional_updates_if_fixed_ratio_persists'], 'resumable_position': False, **FLAGS}


def preflight(args):
    records = []
    path = bind({'path': str(args.configuration), 'sha256': sha(args.configuration)}, records)
    configuration = runner._read_json(path)
    prior = bind_prior(configuration, records)
    authority, remaining = phase_authority(configuration, args.check, prior['terminal_time'])
    prepared = admit_seed(args, configuration, records)
    historical = copy.deepcopy(configuration)
    historical['phase']['declared_at_utc'] = prior['terminal_time']
    original_lineage = previous.bind_native_proof(historical, prepared)
    output = Path(prepared['output'])
    run_root = Path(prior['run_root'])
    require(not output.is_relative_to(run_root) and not run_root.is_relative_to(output),
            'trial output overlaps prior evidence')
    require(prepared['case']['tolerances']['newton'] == POLICY['stationary_tolerance'],
            'frozen stationary tolerance differs')
    materialised = authority['deadline_utc'] is not None
    prepared.update(configuration=configuration, phase={**authority, **FLAGS, 'new_wall_authority_allocated': materialised}, remaining_wall_seconds=remaining,
                    prior=prior, retained_stages=prior['prior_stages'], new_wall_authority_allocated=materialised,
                    lineage={'original': original_lineage, 'prior': prior, **FLAGS},
                    native_execution_authorized=materialised and not args.check and prepared['component']['clean'] is True,
                    phase_declaration_materialised=materialised, first_release_new_update_cap=TRIAL_CAP, **FLAGS)
    return prepared


def source_records(parent, metadata_only):
    component = runner.component_revision(ROOT)
    require(parent["component_commit"] == parent["component"]["commit"] == old.BASE,
            "retained producer source differs")
    runner.require_clean_source(parent["component"])
    require(set(component["sources"]) == set(parent["component"]["sources"]), "frozen method inventory differs")
    names = set(component["sources"]) | {seed.ENTRYPOINT, old.ENTRYPOINT}
    names.update(str(path.relative_to(ROOT)) for path in (ROOT / 'problems').glob('*.py'))
    for module_name, module in tuple(sys.modules.items()):
        if module_name.startswith(("runners.", "problems.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(ROOT), "helper imported from another component")
            names.add(str(path.relative_to(ROOT)))
    implementation = {}
    additions = {ENTRYPOINT}
    for name in sorted(names | additions):
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), "regular implementation file required")
        digest = sha(path)
        if name not in additions:
            frozen = runner._git(ROOT, "show", f"{BASE if name == previous.ENTRYPOINT else METHODS_BASE}:{name}")
            require(frozen is not None and hashlib.sha256(frozen).hexdigest() == digest,
                    "frozen source helper or method changed: " + name)
        if name in parent["component"]["sources"]:
            require(parent["component"]["sources"][name]["sha256"] == digest, "parent method differs")
        implementation[name] = {"path": str(path), "sha256": digest}
    if not metadata_only:
        runner.require_clean_source(component)
        for name, record in implementation.items():
            committed = runner._git(ROOT, "show", f"{component['commit']}:{name}")
            require(committed is not None and hashlib.sha256(committed).hexdigest() == record["sha256"],
                    "implementation differs from clean committed source")
    return component, implementation


def admit_seed(args, new_configuration, records):
    require(args.source.resolve(strict=True) == ROOT, "resume must use its isolated source")
    configuration_path = bind(new_configuration["expired_resume_configuration"], records)
    require(sha(configuration_path) == EXPIRED_CONFIG_SHA, "expired resume configuration differs")
    configuration = runner._read_json(configuration_path)
    require(configuration["schema"] == old.SCHEMA + "-configuration"
            and configuration["deadline_utc"] == old.DEADLINE and configuration["resource_policy"] == RESOURCES
            and re.fullmatch(r"[a-zA-Z0-9._-]{1,96}", configuration["phase_id"]), "resume configuration or authority differs")
    cfg_path = bind(configuration["probe_configuration"], records)
    require(sha(cfg_path) == PROBE_CONFIG_SHA, "accepted probe configuration differs")
    cfg = runner._read_json(cfg_path)
    probe_path = bind(configuration["probe_summary"], records)
    require(sha(probe_path) == PROBE_SHA, "accepted ownership probe differs")
    probe = runner._read_json(probe_path)
    completion_path = bind({"path": str(probe_path.parent / "completion.json"), "sha256": PROBE_COMPLETION_SHA}, records)
    completion = runner._read_json(completion_path)
    require(completion["schema"] == "volume-remesher-ownership-probe-v1"
            and completion["status"] == "complete" and completion["numerical_acceptance"] is False,
            "probe completion differs")
    for name, digest in completion["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe probe completion path")
        bind({"path": str(probe_path.parent / name), "sha256": digest}, records)
    probe_attempt = probe_path.parent.parent
    for name in ("runner.log", "supervisor-result.json", "dispatch.json", "run-contract.json"):
        bind({"path": str(probe_attempt / name), "sha256": sha(probe_attempt / name)}, records)
    for item in [*cfg["bindings"].values(), *cfg["source_files"].values(), *cfg["api_files"].values()]:
        bind(item, records)
    for name, digest in cfg["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe probe worker path")
        bind({"path": str(cfg_path.parent / name), "sha256": digest}, records)
    snapshot = cfg["snapshot"]
    state = bind(snapshot["state"], records)
    require(sha(state) == STATE_SHA and cfg["bindings"]["snapshot"] == snapshot["state"]
            and snapshot["format"] == "memory_snapshot", "only the exact c7 after-quality snapshot is admitted")
    case_path = bind(cfg["bindings"]["case"], records)
    require(sha(case_path) == seed.CASE_SHA and args.case.resolve(strict=True) == case_path, "exact retained case required")
    case = model.read_case(case_path)
    require(case["physics"]["vhat"] == 3.8 and case["physics"]["ca_in"] == 0.02
            and case["physics"]["bond"] == 1.56, "volume resume case differs")
    parent_path = Path(cfg["bindings"]["run_manifest"]["path"])
    parent = runner._read_json(parent_path)
    component, implementation = source_records(parent, args.check)
    environment = runner.pyoomph_revision()
    runner.require_pinned_environment(environment)
    require(environment["commit"] == model.PYOOMPH_COMMIT
            and environment["native_core"]["sha256"] == diagnostic.CORE
            and environment["interpreter"]["resolved_executable"]
            == parent["pyoomph"]["interpreter"]["resolved_executable"], "native source/core/interpreter differs")
    bind(environment["native_core"], records)
    require(cfg["source"]["commit"] == old.BASE and cfg["deadline_utc"] == old.DEADLINE
            and cfg["phase_debits"] == DEBITS
            and cfg["phase_caps"] == {"total_updates": 160, "physical_remeshes": 3, "per_mesh_stage_updates": 40},
            "accepted probe phase authority differs")
    budget = runner._read_json(Path(cfg["bindings"]["phase_budget"]["path"]))
    polish = runner._read_json(Path(cfg["bindings"]["polish"]["path"]))
    failure = runner._read_json(Path(cfg["bindings"]["failure"]["path"]))
    projection = runner._read_json(Path(cfg["bindings"]["projection"]["path"]))
    load_path = parent_path.parent / "seed-quality-remesh/load.json"
    bind({"path": str(load_path), "sha256": LOAD_SHA}, records)
    require(all(budget[key] == DEBITS[key] for key in ("updates_total", "physical_remeshes_total", "physical_remesh_starts"))
            and budget["physical_remesh_pending"] is False and budget["budget_hooks_agree"] is True
            and budget["budget"] == {**seed.BUDGET, "wall_seconds": 21600}
            and len(budget["stages"]) == 1 and budget["stages"][0]["label"] == "bond_000"
            and (polish["updates_total"], polish["remeshes_total"], polish["stage_updates"]) == (3, 1, 3)
            and polish["rollback_verified"] is True
            and failure["status"] == "failed" and failure["numerical_acceptance"] is False
            and failure["error"] == "quality phase requires one shared two-phase remesher",
            "retained phase debit, guard failure or rollback differs")
    require(projection["after_sha256"] == STATE_SHA and projection["snapshot_restore"]["verified"] is True
            and {key: value for key, value in projection["after"]["signature"].items() if key != "serialized_state"}
            == snapshot["expected_signature"] and projection["after"]["pins"] == snapshot["expected_pins"],
            "raw c7 projection proof differs")
    bind({"path": str(parent_path.parent / "seed-quality-remesh/before.dump"),
          "sha256": projection["before_sha256"]}, records)
    require(probe["schema"] == "volume-remesher-ownership-probe-v1" and probe["administrative_probe_complete"] is True
            and probe["load"]["verified"] is True and probe["load"]["full_stream_load_proof"]["verified"] is True
            and probe["interpretation"] == "single_owned_physical_remesher_repeated"
            and probe["ownership"]["raw_count"] == 2 and probe["ownership"]["unique_count"] == 1
            and probe["ownership"]["registration_unchanged"] is True
            and probe["zero_operation_guard"]["violations"] == []
            and probe["phase_debits_unchanged"] == DEBITS
            and probe["absolute_deadline_utc"] == old.DEADLINE, "ownership probe proof differs")
    saved = state.read_bytes()
    metadata = history.serialized_metadata(saved)
    require(json.loads(json.dumps(metadata)) == snapshot["metadata"] and metadata["endtime"] == 8.0
            and metadata["time"] == case["solve"]["relax_time"]
            and runner.snapshot_bc_cache_signature(saved, mesh_reference_base=Path(snapshot["mesh_reference_base"]))[
                "payload_sha256"] == snapshot["payload_sha256"]
            and runner.state_bc_cache_marker(state) == "init", "c7 complete byte/history/BC proof differs")
    require(history.verify_snapshots(saved, (probe_path.parent / "load-before-endtime.dump").read_bytes(),
                                    (probe_path.parent / "load-after-endtime.dump").read_bytes(),
                                    Path(snapshot["mesh_reference_base"]), probe_attempt,
                                    snapshot["payload_sha256"])["verified"] is True,
            "authentic probe full-stream witnesses differ")
    meshes = snapshot["meshes"]
    require(len(meshes) == 1 and runner.state_mesh_references(state) == [item["stored_path"] for item in meshes],
            "c7 mesh closure differs")
    for mesh in meshes:
        mesh_path = bind(mesh, records)
        require(mesh_path == (Path(snapshot["mesh_reference_base"]) / mesh["stored_path"]).resolve(strict=True),
                "c7 mesh reference differs")
    require(snapshot["expected_signature"]["layout"]["ndof"] == 159597
            and snapshot["expected_signature"]["parameters"] == {"Ca_in": 0.02, "Bo": 0.0},
            "c7 fields/layout/forcing differs")
    original_phase_path = Path(cfg["bindings"]["original_phase"]["path"])
    original_phase = seed.phase_contract(original_phase_path)
    original_packet = bind(original_phase["restart_packet"], records)
    original_parent = runner._read_json(Path(runner._read_json(original_packet)["parent_manifest"]["path"]))
    original_restart = runner.validate_restart(original_packet, case, seed.CASE_SHA,
                                               component=original_parent["component"], environment=environment,
                                               azimuthal_stability=True)
    terminal_path = bind(original_phase["original_terminal"], records)
    original_records, original_failure = seed.failed_inputs(original_restart, runner._read_json(terminal_path))
    records.extend(original_records)
    require(original_failure == parent["original_failure"] == failure["original_failure"],
            "original 9b failure debit was changed or transferred")
    compiler = owned.PinnedSelection(Path(cfg["api_files"]["generic_problem"]["path"]))
    require(not any(item.is_symlink() for item in (args.out, *args.out.parents)),
            "resume output contains a symlink")
    output = runner.validate_output_path(args.out, ROOT)
    for evidence in (parent_path.parent.parent, probe_path.parent.parent):
        require(not output.is_relative_to(evidence) and not evidence.is_relative_to(output),
                "resume output overlaps the seed or probe attempt")
    wall = None if new_configuration["phase"]["deadline_utc"] is None else (utc(new_configuration["phase"]["deadline_utc"]) - utc_now()).total_seconds()
    restart = {"schema": SCHEMA + "-raw-projection-admission", "packet": cfg["bindings"]["projection"],
               "packet_kind": "projection_receipt", "parent_manifest": cfg["bindings"]["run_manifest"],
               "state": {**snapshot["state"], "format": "memory_snapshot"},
               "meshes": meshes, "mesh_reference_base": snapshot["mesh_reference_base"],
               "mesh_exchange": {item["stored_path"]: item["path"] for item in meshes},
               "stage": {"kind": "retained_after_quality_projection", "bond_index": 0,
                         "bond": 0.0, "ca_in": 0.02, "bc_mode": "init"}}
    return {"case": case, "case_path": str(case_path), "component": component, "environment": environment,
            "implementation": implementation, "inputs": records, "output": str(output), "restart": restart,
            "original_failure": original_failure, "configuration": configuration,
            "phase": {"kind": "resume_existing_phase", "phase_id": configuration["phase_id"],
                      "original_phase": cfg["bindings"]["original_phase"], "deadline_utc": old.DEADLINE,
                      "caps": seed.BUDGET, "retained_debits": DEBITS, "fresh_numerical_budget_allocated": False},
            "remaining_wall_seconds": wall, "retained_stages": budget["stages"], "probe": probe,
            "snapshot_contract": snapshot, "compiler": compiler, "schedule": model.stationary_bond_schedule(case),
            "source_seal_pending": component["clean"] is not True,
            "native_execution_authorized": not args.check and component["clean"] is True}


PRIOR_ROOT_HASHES = {'prior_terminal': '2f42d7752b2674d5981112fc6d87b89bad742ee6df8cc75b753e88ff8f4e36ad', 'prior_whole_run_terminal': '39b4b6134b0ac9d9de870d88d1e11a1e49f609d6a8d4e413c2c1ef93a4916ec1', 'prior_archive': '38014939a9ae903e1f813539ea9251fe35ba13277b3037dd5640f28e23d3c87d', 'prior_release': '6d81e76b717ba52a326a85c9411e62f386f8eafdb3598f9d911034e9b495cc58', 'numerical_control_evidence': 'd184f7103fec2c8a55667760eb7ad7d06b2d175f49141a276885f79a24e08255'}


def polish_trial_stage(problem, case, output, phase, policy):
    before = seed.state_record(problem)
    result = seed.polish_stage(problem, case, output, phase, 'bond_000', .25)
    try:
        policy.finish_stage()
        phase.finish_bo_zero()
    except BaseException:
        # A pending refusal can occur on a returned solve's last update. Its
        # frozen helper has already restored controls; restore the full entry
        # state before rejecting it and before any checkpoint is made.
        phase.active = False
        phase.pending_remesh = False
        policy.post_solve_rollback = seed.restore(problem, before)
        write_json(output / 'bond_000-policy-full-rollback.json', policy.post_solve_rollback)
        raise
    return result


def execute(prepared, output):
    require(prepared["native_execution_authorized"] is True, "metadata inspection cannot authorise native resume")
    stable_inputs(prepared)
    phase = TrialPhaseBudget(prepared["phase"],
                                lambda event, **values: runner.append_progress(output, {"event": event, **values}),
                                prepared["retained_stages"])
    import numpy as np
    from pyoomph.generic.problem import GenericProblemHooks
    from pyoomph.meshes.mesh import ODEStorageMesh
    from pyoomph.meshes.remesher import RemesherViaRecreation

    cls = owned.owned_problem_type(model._native_problem_types()[0], GenericProblemHooks,
                                   prepared["compiler"], RemesherViaRecreation, output)
    case, stages, receipts = prepared["case"], [], []
    policy = damping_policy_type(GenericProblemHooks)(phase)
    phase.policy = policy
    try:
        with cls(case, initial_bond=0.0) as problem:
            seed.configure(problem, output)
            problem += policy  # Register while _quality_phase is None; Guard and Budget follow.
            seed.attach_budget(problem, phase, GenericProblemHooks)
            directory = output / "retained-load"
            directory.mkdir()
            loaded = load_retained(problem, prepared, directory, phase, ODEStorageMesh)
            _, owner, ownership = owned.canonicalise(problem, problem._domains_remesh_on_inversion,
                                                     prepared["compiler"], RemesherViaRecreation)
            require(len(ownership["raw_entries"]) == 2 and owner._cnt == 1,
                    "fresh c7 ownership does not match the accepted alias mechanism")
            write_json(directory / "ownership.json", ownership)
            for index, bond in enumerate(prepared["schedule"]):
                phase.check_wall()
                problem.get_global_parameter("Bo").value = bond
                polish = (polish_trial_stage(problem, case, output, phase, policy) if index == 0
                          else seed.polish_stage(problem, case, output, phase, f"bond_{index:03d}", 1.))
                if index == 0:
                    require(polish["status"] == "converged", "trial Bo=0 polish did not converge")
                    phase.emit("bo_zero_first_release_passed", new_completed_updates=phase.bo_zero_updates,
                               full_step_predicate_exercised=policy.full_step_predicate_exercised,
                               parent_is_unqualified=True)
                checkpoint = model._checkpoint_stationary_state(problem, case, output, index, True)
                stages.append({"bond_index": index, "bond": bond, "polish": polish,
                               "checkpoint": checkpoint, "phase_budget": phase.record()})
                stable_inputs(prepared)
            require(stages[-1]["bond"] == case["physics"]["bond"] == 1.56, "no complete Bond endpoint")
            warm = remesh.sample(problem, model, case, output, "warm-endpoint", np)
            checks = remesh.canonical_checks(case, warm["metrics"])
            receipts = getattr(problem, "_owned_remesh_receipts", [])
        cold = seed.qualify_cold(case, checkpoint, warm, output, prepared, phase, np)
        accepted = all(checks.values()) and all(cold["checks"].values())
        result = {"schema": SCHEMA, "status": "completed" if accepted else "failed",
                  "completion_reached": accepted, "numerical_acceptance": accepted,
                  "stages": stages, "checks": checks, "cold_load": cold, "cold_load_checks": cold["checks"],
                  "cold_load_verified": True, "retained_load": loaded, "ownership": ownership,
                  "physical_remesh_receipts": receipts, "phase_budget": phase.record(),
                  "original_failure": prepared["original_failure"], "lineage": prepared["lineage"], "damping_policy": policy.record(), **FLAGS,
                  "parent_is_unqualified": True, "first_release_new_update_cap": TRIAL_CAP, "checkpoint": checkpoint["resume_packet"],
                  "ca_in": 0.02, "bond": 1.56, **NO_CLAIMS, **warm["metrics"]}
        stable_inputs(prepared)
        phase.check_wall()
        write_json(output / "result.json", result)
        write_json(output / "summary.json", result)
        require(accepted, "unchanged warm/cold five-gate qualification failed")
        return result
    except BaseException:
        write_json(output / "partial-result.json", {"schema": SCHEMA, "completion_reached": False,
                                                    "numerical_acceptance": False, "checkpointed_stages": stages,
                                                    "phase_budget": phase.record(), "damping_policy": policy.record(), **FLAGS})
        raise
    finally:
        prepared["trial_policy_receipt"] = policy.record()
        write_json(output / "phase-budget.json", phase.record())
        write_json(output / "damping-policy.json", policy.record())


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "case", "configuration", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(arguments)
    os.umask(0o077)
    prepared = preflight(args)
    if args.check:
        print(json.dumps({"verified": True, "native_imported": False, "native_execution_authorized": False,
                          "source_seal_pending": prepared["source_seal_pending"], "inputs": prepared["inputs"],
                          "retained_debits": DEBITS2, "remaining": {"total_updates": 120, "physical_remeshes": 1,
                                                                    "new_trial_stage_updates": 40, "new_bo_zero_updates": 30},
                          "absolute_deadline_utc": prepared["phase"]["deadline_utc"], "remaining_wall_seconds": prepared["remaining_wall_seconds"],
                          "resource_policy": RESOURCES, "seed_is_stationary": False,
                          "original_failure": prepared["original_failure"], "lineage": prepared["lineage"],
                          "new_wall_authority_allocated": prepared['phase_declaration_materialised'], "parent_is_unqualified": True, **FLAGS, "phase_declaration_materialised": prepared["phase_declaration_materialised"], "trial_schedule": POLICY}, allow_nan=False))
        return 0
    output = Path(prepared["output"])
    stable_inputs(prepared)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    actual = seed.manifest(prepared, arguments)
    own = prepared["implementation"][ENTRYPOINT]
    actual.update(runner=own, runner_path=own["path"], runner_sha256=own["sha256"],
                  argv=[sys.executable, own["path"], *arguments], qualification_schema=SCHEMA,
                  resource_policy=RESOURCES, phase_resume=prepared["phase"], lineage=prepared["lineage"],
                  **FLAGS)
    runner.create_run_manifest(output, actual)
    runner.append_progress(output, {"event": "prepared", "retained_debits": DEBITS2, "absolute_deadline_utc": prepared["phase"]["deadline_utc"]}, first=True)
    try:
        write_json(output / "preparation.json", {key: value for key, value in prepared.items()
                                                if key not in {"compiler", "case"}})
        execute(prepared, output)
        return 0
    except BaseException as error:
        write_json(output / "failure.json", {"schema": SCHEMA, "status": "failed", "error": str(error),
                                             "error_type": type(error).__name__, "completion_reached": False,
                                             "numerical_acceptance": False,
                                             "inputs_unchanged": diagnostic.parents_unchanged(prepared["inputs"]),
                                             "original_failure": prepared["original_failure"], "lineage": prepared["lineage"],
                                             "new_wall_authority_allocated": True, "parent_is_unqualified": True,
                                             "failure_class": "phase-budget-refusal" if isinstance(error, TrialRefusal) or prepared.get("trial_policy_receipt", {}).get("trial_refusal") or any(word in str(error).lower()
                                                 for word in ("budget", "deadline")) else "native-or-contract-refusal",
                                             "trial_refusal": prepared.get("trial_policy_receipt", {}).get("trial_refusal"),
                                             "damping_policy": prepared.get("trial_policy_receipt"), **FLAGS})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

