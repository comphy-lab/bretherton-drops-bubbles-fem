"""Software-only ascending continuation, provenance and qualification controls."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from runners import run_paper_anchor_i as lane


@pytest.fixture
def cases():
    parent = lane.model.read_case(lane.ROOT / lane.seed.BASE_CASE)
    parent["case_id"] = "software_fresh_negative_seed"
    parent["physics"]["ca_in"] = lane.PARENT_CA
    target = copy.deepcopy(parent)
    target["case_id"] = "software_stationary_I_candidate"
    target["physics"]["ca_in"] = lane.TARGET_CA
    return parent, target


@pytest.mark.parametrize(
    "values",
    [
        [],
        [-0.02, -0.0016],
        [-0.01, -0.01, -0.0016],
        [-0.01, -0.015, -0.0016],
        [-0.021, -0.0016],
        [-0.01, -0.0015],
        [-0.01, True, -0.0016],
        [float("nan"), -0.0016],
        [-0.01, 0.0, -0.0016],
        [-0.019, -0.018, -0.017, -0.016, -0.015, -0.014, -0.013, -0.012, -0.0016],
    ],
)
def test_bad_ascending_ladder_is_refused(cases, tmp_path, values):
    parent, target = cases
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"schema": "herrada-paper-ca-ladder-v1", "values": values}))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        lane.read_ladder(path, parent, target)
    assert path.read_bytes() == before


def test_declared_ladder_and_each_truthful_stage(cases, tmp_path):
    parent, target = cases
    values = [-0.01, -0.005, -0.003, -0.002, -0.0016]
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"schema": "herrada-paper-ca-ladder-v1", "values": values}))
    assert lane.read_ladder(path, parent, target) == values
    previous = parent
    for index, value in enumerate(values, 1):
        current = lane.anchor.stage_case(target, value, index, index == len(values))
        assert current["physics"]["ca_in"] == value
        assert set(lane.remesh.differing_keys(previous, current)) == {"case_id", "physics.ca_in"}
        previous = current
    assert previous == target
    assert parent["physics"]["ca_in"] == -0.02


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("physics", "bond", 1.57),
        ("mesh", "film_elements", 8.0),
        ("mesh", "bulk_resolution", 0.04),
        ("tolerances", "pressure_boundary_absolute", 1e-5),
        ("solve", "max_newton_iterations", 41),
    ],
)
def test_other_case_fields_cannot_change(cases, section, key, value):
    parent, target = cases
    target[section][key] = value
    with pytest.raises(ValueError, match="only Ca and case_id"):
        lane.transition_contract(parent, target)


def test_positive_seed_and_other_endpoint_are_refused(cases):
    parent, target = cases
    parent["physics"]["ca_in"] = 0.02
    with pytest.raises(ValueError, match="declared negative-flow"):
        lane.transition_contract(parent, target)
    parent["physics"]["ca_in"] = -0.02
    target["physics"]["ca_in"] = -0.00315
    with pytest.raises(ValueError, match="declared negative-flow"):
        lane.transition_contract(parent, target)


def signature(ca=-0.02):
    payload = hashlib.sha256(repr(ca).encode()).hexdigest()
    return {
        "bc_cache_marker": "init",
        "parameters": {"Bo": 1.56, "Ca_in": ca},
        "layout": {"ndof": 6},
        "arrays": {"free_dofs": {"sha256": "software-fields"}},
        "volume": 1.0,
        "frame_speed": -0.02,
        "time": 8.0,
        "serialized_state": {"sha256": payload, "payload_sha256": payload},
    }


def bind_completion(runtime, files):
    (runtime / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "numerical_acceptance": True,
                "files": {name: lane.sha(runtime / name) for name in files},
            }
        )
    )


@pytest.fixture
def parent_bundle(cases, tmp_path):
    parent, target = cases
    runtime = tmp_path / "parent"
    (runtime / "terminal/normalization").mkdir(parents=True)
    checks = dict.fromkeys(lane.GATES, True)
    cold = {
        "verified": True,
        "loaded": signature(),
        "actual_pins": {"sha256": "software-pins"},
        "canonical_checks": checks,
    }
    files = {
        "run-manifest.json": {"software_fixture": True},
        "summary.json": {
            "completion_reached": True,
            "cold_load_verified": True,
            "checks": checks,
            "cold_load_checks": checks,
        },
        "cold-load.json": cold,
        "result.json": {
            "schema": "private-fresh-negative-paper-seed-v1",
            "numerical_acceptance": True,
            "terminal": {"explicit_force_remesh_calls": 1},
        },
        "terminal/normalization/receipt.json": {
            "verified": True,
            "checks": {"all_dummy_histories_equal_vertex_pressure_averages": True},
        },
    }
    for name, value in files.items():
        (runtime / name).write_text(json.dumps(value))
    bind_completion(runtime, files)
    restart = {
        "stage": {"bc_mode": "init"},
        "roundtrip": {
            "after": signature(),
            "layout": signature()["layout"],
        },
    }
    return SimpleNamespace(
        runtime=runtime, parent=parent, target=target, cold=cold, files=files, restart=restart
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_completion",
        "unbound_cold",
        "failed_gate",
        "cold_false",
        "changed_bytes",
        "history",
        "wrong_forcing",
        "no_remesh",
        "dummy_false",
        "not_accepted",
    ],
)
def test_failed_or_unbound_parent_is_refused(parent_bundle, fault):
    bundle = parent_bundle
    if fault == "missing_completion":
        (bundle.runtime / "completion.json").unlink()
    elif fault == "unbound_cold":
        bind_completion(bundle.runtime, set(bundle.files) - {"cold-load.json"})
    elif fault == "changed_bytes":
        (bundle.runtime / "summary.json").write_text("{}")
    elif fault == "not_accepted":
        path = bundle.runtime / "completion.json"
        value = json.loads(path.read_text())
        value["numerical_acceptance"] = False
        path.write_text(json.dumps(value))
    else:
        if fault == "failed_gate":
            bundle.files["summary.json"]["checks"] = {
                **dict.fromkeys(lane.GATES, True),
                "lower_static_pressure": False,
            }
        elif fault == "cold_false":
            bundle.files["cold-load.json"]["verified"] = False
        elif fault == "history":
            bundle.files["cold-load.json"]["loaded"]["serialized_state"]["payload_sha256"] = (
                "changed"
            )
        elif fault == "wrong_forcing":
            bundle.files["cold-load.json"]["loaded"]["parameters"]["Ca_in"] = -0.01
        elif fault == "no_remesh":
            bundle.files["result.json"]["terminal"]["explicit_force_remesh_calls"] = 0
        elif fault == "dummy_false":
            bundle.files["terminal/normalization/receipt.json"]["verified"] = False
        for name, value in bundle.files.items():
            (bundle.runtime / name).write_text(json.dumps(value))
        bind_completion(bundle.runtime, bundle.files)
    before = {str(p): p.read_bytes() for p in bundle.runtime.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        lane.qualified_parent(bundle.runtime / "run-manifest.json", bundle.parent, bundle.restart)
    assert {str(p): p.read_bytes() for p in bundle.runtime.rglob("*") if p.is_file()} == before


def test_qualified_parent_records_every_bound_byte(parent_bundle):
    bundle = parent_bundle
    cold, records = lane.qualified_parent(
        bundle.runtime / "run-manifest.json", bundle.parent, bundle.restart
    )
    assert cold == bundle.cold
    assert len(records) == len(bundle.files) + 1
    assert lane.diagnostic.parents_unchanged(records) is True


@pytest.mark.parametrize("fault", [None, "core", "interpreter", "dirty", "producer"])
def test_preflight_binds_parent_case_clean_source_and_exact_environment(
    parent_bundle,
    tmp_path,
    monkeypatch,
    fault,
):
    bundle = parent_bundle
    parent_case, target_case, ladder = [
        tmp_path / name
        for name in (
            "parent-case.json",
            "target-case.json",
            "ladder.json",
        )
    ]
    parent_case.write_text(json.dumps(bundle.parent))
    target_case.write_text(json.dumps(bundle.target))
    ladder.write_text(
        json.dumps({"schema": "herrada-paper-ca-ladder-v1", "values": [-0.01, -0.0016]})
    )
    component = lane.runner.component_revision(lane.ROOT)
    component.update(
        commit=lane.PARENT_SOURCE,
        head=lane.PARENT_SOURCE,
        clean=True,
        status=[],
        declared_commit=None,
    )
    current = {**component, "commit": "f" * 40, "head": "f" * 40}
    if fault == "dirty":
        current["clean"] = False
    environment = {
        "available": True,
        "commit": lane.diagnostic.PYOOMPH,
        "url": "https://software.invalid",
        "direct_url": {"vcs_info": {"vcs": "git", "commit_id": lane.diagnostic.PYOOMPH}},
        "native_core": {"path": "/software-only/core.so", "sha256": lane.diagnostic.CORE},
        "interpreter": {"resolved_executable": "/software-only/python"},
    }
    producer = {
        "path": str(lane.ROOT / lane.PRODUCER),
        "sha256": lane.sha(lane.ROOT / lane.PRODUCER),
    }
    parent_manifest = {
        "component_commit": lane.PARENT_SOURCE,
        "component": component,
        "runner": producer,
        "runner_path": producer["path"],
        "runner_sha256": producer["sha256"],
        "implementation": {lane.PRODUCER: producer},
        "fresh_seed_stage": {"kind": "fresh_negative_seed"},
        "parent_state_used": False,
        "azimuthal_stability": True,
        "pyoomph": copy.deepcopy(environment),
        "case_path": str(parent_case),
        "case_sha256": lane.sha(parent_case),
    }
    if fault == "producer":
        parent_manifest["runner_sha256"] = "wrong"
    if fault == "core":
        environment["native_core"]["sha256"] = "wrong"
    if fault == "interpreter":
        environment["interpreter"]["resolved_executable"] = "/different/python"
    path = bundle.runtime / "run-manifest.json"
    path.write_text(json.dumps(parent_manifest))
    bind_completion(bundle.runtime, bundle.files)
    state, packet = bundle.runtime / "state.dump", bundle.runtime / "resume.json"
    state.write_bytes(b"software-only; not a native state")
    packet.write_text("{}")
    restart = {
        **bundle.restart,
        "stage": {"kind": "completed_polish", "bond": 1.56, "bc_mode": "init"},
        "parent_manifest": {"path": str(path), "sha256": lane.sha(path)},
        "packet": {"path": str(packet), "sha256": lane.sha(packet)},
        "state": {"path": str(state), "sha256": lane.sha(state)},
        "meshes": [],
    }
    git = lane.diagnostic.git

    def read_git(root, *args):
        if args[:2] == ("merge-base", "--is-ancestor"):
            assert args[2:] == (lane.PARENT_SOURCE, current["commit"])
            return b""
        if args[0] == "show" and args[1].startswith(current["commit"] + ":"):
            return (lane.ROOT / args[1].partition(":")[2]).read_bytes()
        return git(root, *args)

    def validate(path, case, digest, **kwargs):
        assert path == packet and case == bundle.parent and digest == lane.sha(parent_case)
        assert kwargs == {
            "component": current,
            "environment": environment,
            "azimuthal_stability": True,
        }
        return restart

    monkeypatch.setattr(lane, "PARENT_CASE_SHA", lane.sha(parent_case))
    monkeypatch.setattr(lane.runner, "component_revision", lambda root: current)
    monkeypatch.setattr(lane.runner, "pyoomph_revision", lambda: environment)
    monkeypatch.setattr(lane.runner, "validate_restart", validate)
    monkeypatch.setattr(lane.diagnostic, "git", read_git)
    args = SimpleNamespace(
        source=lane.ROOT,
        case=target_case,
        manifest=path,
        packet=packet,
        state=state,
        ca_ladder=ladder,
        out=tmp_path / "unused-output",
    )
    if fault:
        with pytest.raises(ValueError):
            lane.preflight(args)
    else:
        prepared = lane.preflight(args)
        assert prepared["parent"] == bundle.parent and prepared["target"] == bundle.target
        assert prepared["contract"]["ca_ladder"]["direction"] == "ascending"
        assert prepared["contract"]["total_newton_update_bound"] == 480
        assert prepared["contract"]["per_polish_remesh_bound"] == 3
        assert prepared["implementation"][lane.ENTRYPOINT]["sha256"] == lane.sha(
            lane.ROOT / lane.ENTRYPOINT
        )
        assert lane.diagnostic.parents_unchanged(prepared["records"]) is True
    assert not args.out.exists()


@pytest.mark.parametrize("fault", [None, "source", "producer", "hash", "stage"])
def test_only_the_hash_bound_fresh_seed_producer_is_allowed(fault):
    digest = hashlib.sha256(
        lane.diagnostic.git(
            lane.ROOT,
            "show",
            f"{lane.PARENT_SOURCE}:{lane.PRODUCER}",
        )
    ).hexdigest()
    path = str(lane.ROOT / lane.PRODUCER)
    value = {
        "component_commit": lane.PARENT_SOURCE,
        "component": {"commit": lane.PARENT_SOURCE},
        "runner": {"path": path, "sha256": digest},
        "runner_path": path,
        "runner_sha256": digest,
        "implementation": {lane.PRODUCER: {"path": path, "sha256": digest}},
        "fresh_seed_stage": {"kind": "fresh_negative_seed"},
        "parent_state_used": False,
        "azimuthal_stability": True,
    }
    if fault == "source":
        value["component_commit"] = "0" * 40
    elif fault == "producer":
        value["runner"]["path"] = str(lane.ROOT / "runners/run_paper_anchor.py")
    elif fault == "hash":
        value["runner"]["sha256"] = "0" * 64
    elif fault == "stage":
        value["fresh_seed_stage"]["kind"] = "stationary_bo_stage"
    if fault:
        with pytest.raises(ValueError):
            lane.check_parent_producer(value)
    else:
        lane.check_parent_producer(value)


def prepared_fixture(bundle):
    return {
        "parent": bundle.parent,
        "target": bundle.target,
        "values": [-0.01, -0.0016],
        "parent_cold": bundle.cold,
        "restart": {
            **bundle.restart,
            "parent_manifest": {
                "path": str(bundle.runtime / "run-manifest.json"),
            },
        },
        "environment": {},
        "component": {"commit": "software-only"},
        "implementation": {},
        "records": [
            {
                "path": str(bundle.runtime / "summary.json"),
                "sha256": lane.sha(bundle.runtime / "summary.json"),
            }
        ],
        "contract": {"parent_case": {"path": "/software-only/parent-case.json"}},
    }


def test_every_manifest_binds_the_new_entrypoint(parent_bundle, tmp_path):
    prepared = prepared_fixture(parent_bundle)
    case_path = tmp_path / "target.json"
    case_path.write_text(json.dumps(prepared["target"]))
    for stage in (
        {"kind": "stationary_I_candidate"},
        {"kind": "ascending_ca_continuation"},
        {"kind": "terminal_remesh"},
    ):
        manifest = lane.manifest(case_path, prepared["target"], prepared, stage)
        assert manifest["runner_path"] == str(lane.ROOT / lane.ENTRYPOINT)
        assert manifest["runner"] == {
            "path": manifest["runner_path"],
            "sha256": lane.sha(lane.ROOT / lane.ENTRYPOINT),
        }
        assert manifest["runner_sha256"] == manifest["runner"]["sha256"]
        assert manifest["case_sha256"] == lane.sha(case_path)
        assert manifest["anchor_stage"] == stage
        assert "Ca=-0.0016" in manifest["qualification_scope"]
        assert lane.anchor.TARGET_CA == -0.00315


def arguments(bundle, case, output, check=False):
    values = [
        "--source",
        str(lane.ROOT),
        "--case",
        str(case),
        "--manifest",
        str(bundle.runtime / "run-manifest.json"),
        "--packet",
        "/software-only/resume.json",
        "--state",
        "/software-only/state.dump",
        "--ca-ladder",
        "/software-only/ladder.json",
        "--out",
        str(output),
    ]
    return values + (["--check"] if check else [])


def test_check_creates_no_output_and_never_executes(parent_bundle, tmp_path, monkeypatch):
    prepared = prepared_fixture(parent_bundle)
    monkeypatch.setattr(lane, "preflight", lambda args: prepared)
    monkeypatch.setattr(lane, "execute", lambda *args: pytest.fail("native execution forbidden"))
    output = tmp_path / "check-output"
    assert lane.main(arguments(parent_bundle, tmp_path / "case.json", output, True)) == 0
    assert not output.exists()


def test_import_uses_no_native_solver():
    probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; from runners import run_paper_anchor_i; "
                "assert not any(n == 'pyoomph' or n.startswith('pyoomph.') "
                "or n.startswith('_pyoomph_core') for n in sys.modules)"
            ),
        ],
        cwd=lane.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_failure_retains_manifest_and_parent_bytes(parent_bundle, tmp_path, monkeypatch):
    prepared = prepared_fixture(parent_bundle)
    case_path = tmp_path / "target.json"
    case_path.write_text(json.dumps(prepared["target"]))
    monkeypatch.setattr(lane, "preflight", lambda args: prepared)

    def fail(*args):
        raise RuntimeError("software stationary budget exhausted")

    monkeypatch.setattr(lane, "execute", fail)
    output = tmp_path / "failed-output"
    before = (parent_bundle.runtime / "summary.json").read_bytes()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        lane.main(arguments(parent_bundle, case_path, output))
    failure = json.loads((output / "failure.json").read_text())
    assert failure["parents_unchanged"] is True and failure["numerical_acceptance"] is False
    assert (parent_bundle.runtime / "summary.json").read_bytes() == before
    assert (output / "run-manifest.json").exists() and not (output / "completion.json").exists()


@pytest.mark.parametrize("cold_pressure_failure", [False, True])
def test_ascending_orchestration_uses_one_terminal_remesh_and_original_gates(
    parent_bundle,
    tmp_path,
    monkeypatch,
    cold_pressure_failure,
):
    prepared = prepared_fixture(parent_bundle)
    prepared["values"] = [-0.01, -0.005, -0.003, -0.002, -0.0016]
    case_path = tmp_path / "target.json"
    case_path.write_text(json.dumps(prepared["target"]))
    output = tmp_path / "output"
    output.mkdir()
    for name, payload in (
        (
            "run-manifest.json",
            lane.manifest(
                case_path, prepared["target"], prepared, {"kind": "stationary_I_candidate"}
            ),
        ),
    ):
        lane.write_json(output / name, payload)
    lane.runner.append_progress(output, {"event": "software-prepared"}, first=True)
    calls, packets, builds = [], {}, []
    pins = {"sha256": "software-pins"}

    class Problem:
        def __init__(self, case):
            self.case = copy.deepcopy(case)
            self.parameters = {
                "Bo": SimpleNamespace(value=1.56),
                "Ca_in": SimpleNamespace(value=case["physics"]["ca_in"]),
            }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_global_parameter(self, name):
            return self.parameters[name]

        def force_remesh(self):
            calls.append("force-remesh")
            return True

    def build(case, **kwargs):
        builds.append(case["physics"]["ca_in"])
        return Problem(case)

    def sample(problem, model, case, directory, phase, np):
        provisional = directory.name.startswith("ca_")
        failure = provisional or (cold_pressure_failure and directory.name == "target-cold")
        metrics = {
            "stationary_residual_linf": 1e-9,
            "lower_pressure_rms": 5e-6 if failure else 1e-9,
            "frame_speed": -0.001,
            "gas_pressure_mean": 1.0,
            "lower_max_divergence": 1e-10,
            "lower_flux": case["physics"]["ca_in"],
            "upper_flux": case["physics"]["ca_in"],
            "gas_volume": 1.0,
            "surface_volume": 1.0,
            "target_volume": 1.0,
        }
        return {
            "metrics": metrics,
            "actual_pins": pins,
            "port_adjacent_geometry": {"max_midside_departure_l2": 0.0},
        }

    def checkpoint(problem, case, directory, index, azimuthal):
        path = directory / "software-resume.json"
        lane.write_json(path, {"software_only": True})
        packets[path] = {
            "packet": {"sha256": lane.sha(path)},
            "roundtrip": {"after": signature(case["physics"]["ca_in"])},
        }
        return {"resume_packet": path.name}

    def validate(path, case, digest, **kwargs):
        manifest = json.loads((path.parent / "run-manifest.json").read_text())
        assert manifest["runner_path"] == str(lane.ROOT / lane.ENTRYPOINT)
        assert manifest["case_sha256"] == digest
        assert manifest["case_id"] == case["case_id"]
        return packets[path]

    def polish(*args, **kwargs):
        assert kwargs == {"initial_relaxation": 1.0}
        calls.append("polish")
        return {"status": "converged"}

    monkeypatch.setattr(lane.model, "build_problem", build)
    monkeypatch.setattr(lane.anchor, "configure", lambda *args: None)
    monkeypatch.setattr(lane.anchor, "stable_inputs", lambda *args: None)
    monkeypatch.setattr(lane.anchor, "restore_checked", lambda *args: {"verified": True})
    monkeypatch.setattr(
        lane.model,
        "_stationary_state_signature",
        lambda problem, *args: signature(problem.parameters["Ca_in"].value),
    )
    monkeypatch.setattr(lane.diagnostic, "actual_pins", lambda problem: pins)
    monkeypatch.setattr(lane.model, "polish_confined_state", polish)
    monkeypatch.setattr(lane.model, "_checkpoint_stationary_state", checkpoint)
    monkeypatch.setattr(lane.runner, "validate_restart", validate)
    monkeypatch.setattr(lane.remesh, "sample", sample)
    monkeypatch.setattr(
        lane.remesh, "retain_portable_sample", lambda *args: {"software_only": True}
    )
    monkeypatch.setattr(
        lane.remesh, "normalize_pressure_dummies", lambda *args: calls.append("dummy-history-audit")
    )
    args = SimpleNamespace(case=case_path)
    if cold_pressure_failure:
        with pytest.raises(ValueError, match="original scientific or cold-load"):
            lane.execute(args, prepared, output)
    else:
        result = lane.execute(args, prepared, output)
        assert result["numerical_acceptance"] is True
        assert result["branch_identity_verified"] is result["eigenanalysis_performed"] is False
        assert len(result["stages"]) == 5
        assert all(stage["numerical_acceptance"] is False for stage in result["stages"])
        assert all(
            not stage["canonical_checks_observed"]["lower_static_pressure"]
            for stage in result["stages"]
        )
    assert calls.count("force-remesh") == calls.count("dummy-history-audit") == 1
    assert calls.count("polish") == 6
    assert builds == [-0.02, -0.0016]
    assert json.loads((output / "summary.json").read_text())["completion_reached"] is (
        not cold_pressure_failure
    )
