"""Owned-selection resume contracts using fakes and metadata only."""

from __future__ import annotations

import copy
import importlib.abc
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_paper_volume_seed_remesh import FakeProblem, Hooks

from runners import paper_owned_remesher as owned
from runners import run_paper_volume_owned_resume as driver

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = datetime.fromisoformat(driver.DEADLINE)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyoomph" or fullname.startswith("pyoomph."):
            raise AssertionError("software tests forbid native solver imports")


@pytest.fixture(autouse=True)
def forbid_native(monkeypatch):
    assert not any(name.startswith("pyoomph") for name in sys.modules)
    guard = NoNative()
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])

    def refuse(*args, **kwargs):
        raise AssertionError("software tests forbid native construction")

    monkeypatch.setattr(driver.model, "_native_problem_types", refuse)
    monkeypatch.setattr(driver.model, "build_problem", refuse)
    for name in ("pyoomph", "pyoomph.generic", "pyoomph.generic.problem"):
        module = ModuleType(name)
        if name.endswith(".problem"):
            module.GenericProblemHooks = Hooks
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(scope="module")
def compiler():
    # Distribution metadata locates the immutable source without importing it.
    path = Path(metadata.distribution("pyoomph").locate_file("pyoomph/generic/problem.py"))
    return owned.PinnedSelection(path)


class Template:
    def __init__(self, problem):
        self.problem = problem
        self.remesher = None
        self._template_override = None
        self.domains = {"liquid", "gas"}

    def get_problem(self):
        return self.problem

    def _get_template(self):
        return self if self._template_override is None else self._template_override

    def has_domain(self, name):
        return name in self.domains

    def _remeshing_can_change_the_mesh(self):
        return True


class Remesher:
    def __init__(self, template, output):
        self.template, self.output = template, output
        self._cnt, self.base_trunk = 1, "PaperTwoPhaseMesh"

    def remesh(self):
        directory = self.output / "solver/_gmsh"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"PaperTwoPhaseMesh_REMESH_{self._cnt:06d}.msh"
        path.write_bytes(b"synthetic generation, never simulation evidence")
        self._cnt += 1
        self.template.problem.remesher_counter = self._cnt


@pytest.fixture
def bundle(tmp_path, monkeypatch, compiler):
    output = tmp_path / "runtime"
    output.mkdir()
    mesh = tmp_path / "original.msh"
    mesh.write_bytes(b"synthetic starting mesh")

    class Base(FakeProblem):
        def __init__(self):
            super().__init__(mesh, output)
            self.original, self.active = Template(self), Template(self)
            self.original._template_override = self.active
            self.owner = Remesher(self.active, output)
            self.original.remesher = self.active.remesher = self.owner
            self._meshtemplate_list = [self.original]
            self._meshdict = {name: SimpleNamespace(_templatemesh=self.active)
                              for name in ("liquid", "gas")}
            self._domains_remesh_on_inversion = {self.original, self.active}
            self._domains_to_remesh = {self.active}
            self.delegations, self.fault = [], None

        def force_remesh(self, only_domains=None, num_adapt=None, interpolator=None):
            self.delegations.append((only_domains, num_adapt, interpolator))
            selected = compiler.select(self, only_domains)
            for hook in self._hooks:
                hook.actions_before_remeshing(selected * (2 if self.fault == "duplicate" else 1))
            for remesher in selected:
                for _ in range(2 if self.fault == "double" else 1):
                    remesher.remesh()
            for hook in self._hooks:
                hook.actions_after_remeshing()
            return True

    cls = owned.owned_problem_type(Base, Hooks, compiler, Remesher, output)
    problem = cls()
    events = []
    phase = driver.RetainedPhaseBudget(
        1000, lambda event, **values: events.append({"event": event, **values}),
        [{"label": "bond_000", "carried_updates": 0, "carried_physical_remeshes": 1,
          "stage_updates_at_entry": 0}], clock=lambda: 0,
        wall_clock=lambda: DEADLINE - timedelta(hours=1),
    )
    driver.seed.attach_budget(problem, phase, Hooks)
    monkeypatch.setattr(driver.model, "_stationary_residual", lambda p: p.residual)
    monkeypatch.setattr(driver.model, "_stationary_state_signature", lambda p, az: p.signature())
    monkeypatch.setattr(driver.diagnostic, "actual_pins", lambda p: {"pins": p.pins.copy()})
    case = driver.model.read_case(ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56-volume3p8.json")
    return SimpleNamespace(problem=problem, phase=phase, events=events, output=output,
                           case=case, compiler=compiler, cls=cls)


def test_pinned_alias_selection_and_canonical_default_are_exact(bundle):
    p, compiler = bundle.problem, bundle.compiler
    before = owned.registration(p)
    assert compiler.select(p, {p.original, p.active}) == [p.owner, p.owner]
    assert compiler.select(p, {p.active}) == [p.owner]
    assert compiler.select(p, None) == [p.owner]
    assert compiler.path.is_file() and driver.sha(compiler.path) == driver.seed.API["python_problem_sha256"]
    assert owned.SELECTION_SHA == "4580b8d7a420ab9f8215fa540fd45110a742839ea34c850f8880781384b4729b"
    assert owned.RANK_SHA == "f1d48798174240a037044d699da8270f6ced2334e068b944e6f9b9ea058b4dfd"
    selected, owner, receipt = owned.canonicalise(p, p._domains_remesh_on_inversion, compiler, Remesher)
    assert selected == {p.active} and selected is not p._domains_remesh_on_inversion
    assert owner is p.owner and len(receipt["raw_entries"]) == 2
    assert owned.canonicalise(p, None, compiler, Remesher)[0] is None
    assert owned.registration(p) == before and not p.delegations


@pytest.mark.parametrize("fault", ["distinct", "absent", "foreign", "single_phase", "outside_chain",
                                   "different_phases", "inactive", "canonical", "unknown_type"])
def test_invalid_selection_refuses_before_delegation_and_keeps_registration(bundle, fault):
    p = bundle.problem
    requested = p._domains_remesh_on_inversion
    if fault == "distinct":
        p.original.remesher = Remesher(p.original, bundle.output)
    elif fault == "absent":
        p.original.remesher = None
    elif fault == "foreign":
        p.original.problem = object()
    elif fault == "single_phase":
        p.active.domains.remove("gas")
    elif fault == "outside_chain":
        extra = Template(p)
        extra.remesher = p.owner
        requested = {extra}
    elif fault == "different_phases":
        p._meshdict["gas"]._templatemesh = p.original
    elif fault == "inactive":
        p.owner.template = p.original
    elif fault == "canonical":
        compiler = bundle.compiler

        class EqualRemesher(Remesher):
            def __eq__(self, other):
                return True

        impostor = EqualRemesher(p.active, bundle.output)

        class BadCanonical:
            def select(self, problem, requested):
                return [impostor] if requested == {p.active} else compiler.select(problem, requested)

        p.__class__ = owned.owned_problem_type(bundle.cls.__mro__[2], Hooks, BadCanonical(), Remesher, bundle.output)
    else:
        p.original.remesher = SimpleNamespace(template=p.active, _cnt=1)
    before = owned.registration(p)
    with pytest.raises(ValueError):
        p.force_remesh(requested)
    assert owned.registration(p) == before and not p.delegations
    assert p.owner._cnt == 1 and not list(bundle.output.rglob("*.msh"))


@pytest.mark.parametrize("default", [False, True])
def test_one_request_has_one_generation_and_unchanged_arguments(bundle, default):
    p, phase = bundle.problem, bundle.phase
    interpolator = object()
    requested = None if default else p._domains_remesh_on_inversion
    assert p.force_remesh(requested, 7, interpolator) is True
    actual, adapt, passed = p.delegations[0]
    assert (actual is None) if default else actual == {p.active}
    assert adapt == 7 and passed is interpolator and len(p.delegations) == 1
    assert (p.owner._cnt, phase.remesh_starts, phase.remeshes) == (2, 2, 2)
    assert not phase.pending_remesh
    assert [event["event"] for event in bundle.events].count("physical_remesh_started") == 1
    assert [event["event"] for event in bundle.events].count("physical_remesh_completed") == 1
    assert [p.name for p in (bundle.output / "solver/_gmsh").glob("*.msh")] == ["PaperTwoPhaseMesh_REMESH_000001.msh"]


def test_implicit_inversion_and_default_routes_keep_pinned_adaption_default(bundle):
    p = bundle.problem
    assert p.force_remesh(p._domains_remesh_on_inversion) is True
    assert p.delegations == [({p.active}, None, None)]
    assert p.force_remesh() is True
    assert p.delegations[1] == (None, None, None)
    assert p.owner._cnt == bundle.phase.remesh_starts == bundle.phase.remeshes == 3


@pytest.mark.parametrize("fault, message", [("double", "one physical generation"),
                                            ("duplicate", "one shared two-phase remesher")])
def test_double_generation_or_duplicate_hook_list_is_refused(bundle, fault, message):
    bundle.problem.fault = fault
    with pytest.raises(ValueError, match=message):
        bundle.problem.force_remesh(bundle.problem._domains_remesh_on_inversion)
    if fault == "duplicate":
        assert bundle.problem.owner._cnt == 1 and bundle.phase.remesh_starts == 1


@pytest.mark.parametrize("quality", [False, True])
def test_carry_and_unchanged_model_hook_keep_inversion_or_reset_quality_stage(bundle, quality):
    p, phase = bundle.problem, bundle.phase
    observed = []

    def solve(**kwargs):
        hook = phase.current
        assert (hook.updates, hook.remeshes, hook.stage_updates, hook.remaining_updates()) == (3, 1, 3, 37)
        for item in p._hooks:
            item.actions_before_newton_solve()
        observed.append((p.newton_relaxation_factor, p.max_newton_iterations))
        if quality:
            for item in p._hooks:
                item.actions_after_newton_solve()
        p.force_remesh(p._domains_remesh_on_inversion)
        assert (hook.updates, hook.remeshes, phase.updates, phase.remeshes) == (3, 2, 3, 2)
        assert hook.stage_updates == (0 if quality else 3)
        assert hook.relaxation == (0.5 if quality else 0.25)
        for item in p._hooks:
            item.actions_before_newton_solve()
        observed.append((p.newton_relaxation_factor, p.max_newton_iterations))
        p.residual = 0.0
        for item in p._hooks:
            item.actions_after_newton_solve()

    p.solve = solve
    result = driver.model.polish_confined_state(p, bundle.case, bundle.output, "bond_000")
    assert result["status"] == "converged" and phase.resume_pending is False
    assert observed == [(0.5, 37), (0.5, 40) if quality else (0.25, 37)]
    assert phase.stages[0]["label"] == "bond_000" and phase.stages[1]["stage_updates_at_entry"] == 3
    assert phase.record()["fresh_numerical_budget_allocated"] is False


def test_absolute_deadline_refuses_independent_of_monotonic_start():
    now = [DEADLINE - timedelta(seconds=1)]
    phase = driver.RetainedPhaseBudget(100, lambda *a, **k: None, [], clock=lambda: 9000,
                                       wall_clock=lambda: now[0])
    phase.check_wall()
    now[0] = DEADLINE
    with pytest.raises(ValueError, match="absolute phase deadline"):
        phase.check_wall()


@pytest.mark.parametrize("wall", [0, -1])
def test_no_nonpositive_wall_authority(wall):
    with pytest.raises(ValueError, match="wall authority"):
        driver.RetainedPhaseBudget(wall, lambda *a, **k: None, [], wall_clock=lambda: DEADLINE)
    with pytest.raises(ValueError, match="absolute phase deadline"):
        driver.remaining_wall(DEADLINE)


@pytest.mark.parametrize("fault", [None, "physical_drift", "unexpected_flat", "strict_restore"])
def test_first_and_later_flat_views_keep_actual_values_and_strict_physical_gates(bundle, monkeypatch, fault):
    import numpy as np

    p = bundle.problem
    state = bundle.output.parent / "synthetic-before.dump"
    saved = p._snapshot_state()
    state.write_bytes(saved)
    p._run_statement_endtime = None
    record = {"path": str(state), "sha256": driver.sha(state)}
    mesh = {"path": p.reference, "sha256": driver.sha(p.reference)}
    restart = {"packet": record, "parent_manifest": record, "state": record, "meshes": [mesh],
               "mesh_reference_base": str(Path.cwd()), "mesh_exchange": {p.reference: p.reference}}
    physical = {"data_history_sha256": "unchanged synthetic histories", "history_values_with_repeats": 3}
    first = {**physical, "signature": {"flat_view": "first"}, "pins": [False, True]}
    later = {**physical, "signature": {"flat_view": "subsequent"}, "pins": [False, True]}
    actual_later = {**later, "signature": {"flat_view": "unexpected"}} if fault == "unexpected_flat" else later
    views = iter([(first, np.array([1.0])), (actual_later, np.array([2.0]))])
    reads = [physical, {**physical, "data_history_sha256": "drift"}
             if fault == "physical_drift" else physical]
    sequence = []
    verify = driver.history.verify_snapshots

    def prove_stream(*args):
        proof = verify(*args)
        sequence.append("full_stream_load_proof")
        return proof

    def pure_read(*args):
        sequence.append("flat_read")
        return next(views)

    def physical_read(*args):
        sequence.append("physical_history")
        return reads.pop(0)

    monkeypatch.setattr(driver.history, "verify_snapshots", prove_stream)
    monkeypatch.setattr(owned, "physical_history", physical_read)
    monkeypatch.setattr(owned, "pure_read", pure_read)
    if fault == "strict_restore":
        def refuse(*args):
            raise ValueError("unchanged strict snapshot gate")
        monkeypatch.setattr(driver.seed, "restore", refuse)
    prepared = {"restart": restart, "case": bundle.case,
                "probe": {"load": {"pure_loaded": first}, "inspection_before": later, "inspection_after": later},
                "snapshot_contract": {"payload_sha256": driver.runner.snapshot_bc_cache_signature(
                    saved, mesh_reference_base=Path.cwd())["payload_sha256"]}}
    directory = bundle.output / "retained-load"
    directory.mkdir()
    if fault:
        with pytest.raises(ValueError, match="physical pinned|accepted probe|strict snapshot"):
            driver.load_retained(p, prepared, directory, bundle.phase, object)
    else:
        loaded = driver.load_retained(p, prepared, directory, bundle.phase, object)
        assert loaded["original_snapshot_gate_unchanged"] is True and loaded["snapshot_restore"]["verified"] is True
    observation = driver.runner._read_json(directory / "pinned-read-observation.json")
    assert sequence == ["full_stream_load_proof", "flat_read", "physical_history",
                        "flat_read", "physical_history"]
    assert observation["first_read_follows_full_stream_load_proof"] is True
    assert "physical_before_first_read" not in observation
    assert observation["flat_hashes_equal"] is False and observation["representation_difference_waived"] is False
    for label, value in (("first", 1.0), ("subsequent", 2.0)):
        with np.load(directory / f"flat-pinned-{label}.npz", allow_pickle=False) as retained:
            assert retained["pinned_values_with_positions"].tolist() == [value]


@pytest.fixture
def admitted(tmp_path):
    configuration = os.environ.get("STONE_VOLUME_OWNED_CONFIGURATION")
    if configuration is None:
        pytest.skip("metadata integration needs a retained configuration outside source")
    configuration = Path(configuration)
    cfg = driver.runner._read_json(Path(driver.runner._read_json(configuration)["probe_configuration"]["path"]))
    args = SimpleNamespace(source=ROOT, case=Path(cfg["bindings"]["case"]["path"]),
                           configuration=configuration, out=tmp_path / "never-native-runtime", check=True)
    prepared = driver.preflight(args)
    return SimpleNamespace(args=args, prepared=prepared, cfg=cfg, directory=tmp_path)


def test_admission_binds_all_required_receipts_source_state_mesh_and_old_failure(admitted):
    prepared = admitted.prepared
    required_hashes = {
        driver.STATE_SHA, driver.LOAD_SHA, driver.PROBE_SHA,
        "90ad40f191aa76f4dffd8197f1f80330ebd8fca16062304b05706230d6b5e91f",
        "2e7ce87799e34b89eaf6a75a7c26b2105f8e4382437bb398c2de03c32b5d54ae",
        "8cdfc1acc2249c33279284af439737bcebdc2bc70288e0bd44e751d79f838af1",
        "fd759dab6995e44d3c00a8a54b703c7dcd6042819a652005e26d50931b836a19",
        "69fd0828edd99f25b3f0fb9e91e9d8efb0d2e6bfc95de71f9624389fb70b5fd2",
        "5a7371b13abf29f798531ed41a550835534553657890ee0a62c106023981ef77",
        "a74f63394b6e834cf9abc8a407385d76f2c6bb117a9b6f349d8f5f47e7d59b2f",
        "fb2086fa4db6f2426eab6a6ede8c69cf55cb5b025fc1c2dc6ba79f8cee54436d",
        "8e16f470b7e17b03e25e02f62b99cd4accc0451144314f39ac4a863e8c913a80",
    }
    assert required_hashes <= {record["sha256"] for record in prepared["inputs"]}
    restart = prepared["restart"]
    assert driver.runner.state_mesh_references(Path(restart["state"]["path"])) == [restart["meshes"][0]["stored_path"]]
    assert prepared["native_execution_authorized"] is False and prepared["source_seal_pending"] is True
    assert prepared["phase"]["retained_debits"] == driver.DEBITS
    assert prepared["original_failure"]["updates"] == 40
    assert prepared["original_failure"]["budget_transferred_to_new_phase"] is False
    assert not admitted.args.out.exists()


def test_any_bound_receipt_changed_hash_refuses_before_native(admitted):
    for record in admitted.prepared["inputs"]:
        if Path(record["path"]) == admitted.args.configuration:
            continue
        with pytest.raises(ValueError):
            driver.bind({**record, "sha256": "0" * 64}, [])


@pytest.mark.parametrize("fault", ["primary", "mesh_closure"])
def test_witness_primary_or_another_mesh_closure_refuses(admitted, monkeypatch, fault):
    cfg = copy.deepcopy(admitted.cfg)
    if fault == "primary":
        witness = Path(cfg["bindings"]["run_manifest"]["path"]).parent / "polish/bond_000/before.dump"
        cfg["snapshot"]["state"] = cfg["bindings"]["snapshot"] = {"path": str(witness), "sha256": driver.sha(witness)}
        assert driver.sha(witness).startswith("34950e91")
    else:
        cfg["snapshot"]["meshes"].append(copy.deepcopy(cfg["snapshot"]["meshes"][0]))
    probe_config = admitted.directory / "modified-probe-configuration.json"
    probe_config.write_text(json.dumps(cfg))
    configuration = driver.runner._read_json(admitted.args.configuration)
    original_directory = Path(configuration["probe_configuration"]["path"]).parent
    for name in cfg["files"]:
        copied = admitted.directory / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes((original_directory / name).read_bytes())
    configuration["probe_configuration"] = {"path": str(probe_config), "sha256": driver.sha(probe_config)}
    path = admitted.directory / "modified-configuration.json"
    path.write_text(json.dumps(configuration))
    monkeypatch.setattr(driver, "PROBE_CONFIG_SHA", driver.sha(probe_config))
    args = copy.copy(admitted.args)
    args.configuration = path
    with pytest.raises(ValueError, match="exact c7|mesh closure"):
        driver.preflight(args)


def test_frozen_c671_helper_bytes_cannot_change(admitted, monkeypatch):
    original = driver.runner._git

    def altered(root, *args):
        if args == ("show", f"{driver.BASE}:{driver.seed.ENTRYPOINT}"):
            return b"another implementation"
        return original(root, *args)

    monkeypatch.setattr(driver.runner, "_git", altered)
    with pytest.raises(ValueError, match="frozen c671 helper"):
        driver.source_records(driver.runner._read_json(Path(admitted.cfg["bindings"]["run_manifest"]["path"])), True)


@pytest.mark.parametrize("evidence", ["run_manifest", "probe"])
def test_output_must_be_outside_seed_and_probe_attempts(admitted, evidence):
    args = copy.copy(admitted.args)
    path = Path(admitted.cfg["bindings"]["run_manifest"]["path"]) if evidence == "run_manifest" else Path(
        driver.runner._read_json(admitted.args.configuration)["probe_summary"]["path"])
    args.out = path.parent.parent / "new-empty-resume-runtime"
    with pytest.raises(ValueError, match="overlaps the seed or probe"):
        driver.preflight(args)


def test_preflight_refuses_at_the_existing_deadline(admitted, monkeypatch):
    monkeypatch.setattr(driver, "utc_now", lambda: DEADLINE)
    with pytest.raises(ValueError, match="absolute phase deadline"):
        driver.preflight(admitted.args)
    assert not admitted.args.out.exists()


def test_check_denies_native_imports_and_creates_no_output(admitted):
    script = """
import importlib.abc, json, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyoomph' or fullname.startswith('pyoomph.'):
            raise AssertionError('metadata check imported native solver')
sys.meta_path.insert(0, Deny())
from runners import run_paper_volume_owned_resume as driver
assert driver.main(sys.argv[1:]) == 0
assert not any(name.startswith('pyoomph') for name in sys.modules)
"""
    args = admitted.args
    result = subprocess.run([sys.executable, "-B", "-c", script, "--source", str(ROOT),
                             "--case", str(args.case), "--configuration", str(args.configuration),
                             "--out", str(args.out), "--check"], cwd=ROOT,
                            capture_output=True, text=True, check=True, timeout=60)
    receipt = json.loads(result.stdout)
    assert receipt["verified"] is True and receipt["native_imported"] is False
    assert receipt["native_execution_authorized"] is False
    assert receipt["remaining"] == {"total_updates": 157, "physical_remeshes": 2, "mesh_stage_updates": 37}
    assert receipt["absolute_deadline_utc"] == driver.DEADLINE and not args.out.exists()
    with pytest.raises(ValueError, match="cannot authorise native"):
        driver.execute(admitted.prepared, args.out)
