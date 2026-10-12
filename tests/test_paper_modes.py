"""Software-only refusal controls for the one-anchor returned-spectrum runner."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
import subprocess
import sys
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from runners import run_paper_modes as modes


@pytest.fixture
def case():
    value = modes.model.read_case(
        modes.ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json"
    )
    value["case_id"] = "software_III_anchor"
    value["physics"]["ca_in"] = -0.00315
    return value


def test_module_import_cannot_import_native_pyoomph():
    program = """
import importlib.abc
import sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyoomph' or fullname.startswith('pyoomph.'):
            raise AssertionError('native import during metadata validation')
sys.meta_path.insert(0, NoNative())
from runners import run_paper_modes
assert 'pyoomph' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=modes.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_scope_and_growth_conversion_are_fixed(case):
    request = modes.request_contract(case, [-0.001, 0.001], 8)
    assert request["eigenproblem_calls"] == 4
    assert request["branch_candidate"] == "III"
    assert request["paper_time_factor"] == 3.3 ** (-1 / 3)
    assert not request["finite_spectrum_complete"]
    wrong = copy.deepcopy(case)
    wrong["physics"]["ca_in"] = +0.02
    with pytest.raises(ValueError, match="declared stationary anchors"):
        modes.request_contract(wrong, [-0.001, 0.001], 8)
    assert modes.request_contract(wrong, [-0.001, 0.001], 8, metadata_only=True)[
        "reference_metadata_only"
    ]
    for shifts, count in [([-0.001, 0.002], 8), ([0, 0.001], 8), ([-0.001, 0.001], 16)]:
        with pytest.raises(ValueError, match="bounded"):
            modes.request_contract(case, shifts, count)


@pytest.mark.parametrize("fault", [None, "core", "source_pin", "interpreter", "source_url"])
def test_environment_refuses_wrong_core_or_parent_provenance(fault):
    environment = {
        "available": True,
        "commit": modes.diagnostic.PYOOMPH,
        "url": "https://github.com/comphy-lab/pyoomph",
        "direct_url": {"vcs_info": {"vcs": "git", "commit_id": modes.diagnostic.PYOOMPH}},
        "native_core": {"path": "/software/core.so", "sha256": modes.diagnostic.CORE},
        "interpreter": {"resolved_executable": "/software/python"},
    }
    parent = copy.deepcopy(environment)
    if fault == "core":
        parent["native_core"]["sha256"] = "wrong-core"
    if fault == "source_pin":
        parent["commit"] = "wrong-pin"
    if fault == "interpreter":
        parent["interpreter"]["resolved_executable"] = "/wrong/python"
    if fault == "source_url":
        parent["url"] = "https://different.invalid/pyoomph"
    if fault is None:
        modes.environment_contract(environment, parent)
    else:
        with pytest.raises(ValueError):
            modes.environment_contract(environment, parent)


def qualified_files(tmp_path, case):
    gates = {key: True for key in modes.GATES}
    metrics = {
        "stationary_residual_linf": 1e-10,
        "lower_pressure_rms": 1e-10,
        "lower_flux": 0.02,
        "upper_flux": 0.02,
        "gas_volume": 4.0,
        "surface_volume": 4.0,
        "target_volume": 4.0,
    }
    summary = {
        "completion_reached": True,
        "cold_load_verified": True,
        "checks": gates,
        "cold_load_checks": gates,
        "ca_in": case["physics"]["ca_in"],
        "bond": case["physics"]["bond"],
        **metrics,
    }
    cold = {
        "verified": True,
        "canonical_checks": gates,
        "actual_pins": {"sha256": "pins"},
        "loaded": {
            "serialized_state": {"payload_sha256": "history"},
            "parameters": {"Ca_in": case["physics"]["ca_in"], "Bo": case["physics"]["bond"]},
        },
    }
    return summary, cold


def publish_fixture(tmp_path, summary, cold, *, accepted=True):
    for name, value in {
        "run-manifest.json": {"schema": "software"},
        "summary.json": summary,
        "cold-load.json": cold,
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    files = {
        name: modes.sha(tmp_path / name)
        for name in ("run-manifest.json", "summary.json", "cold-load.json")
    }
    (tmp_path / "completion.json").write_text(
        json.dumps({"status": "complete", "numerical_acceptance": accepted, "files": files})
    )


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "failed_pressure",
        "lying_pressure",
        "cold_pins",
        "cold_forcing",
        "acceptance",
        "unbound_summary",
    ],
)
def test_parent_qualification_refuses_failed_or_unbound_evidence(tmp_path, case, fault):
    summary, cold = qualified_files(tmp_path, case)
    if fault == "failed_pressure":
        summary["checks"]["lower_static_pressure"] = False
    if fault == "lying_pressure":
        summary["lower_pressure_rms"] = 1.3e-5
    if fault == "cold_pins":
        cold.pop("actual_pins")
    if fault == "cold_forcing":
        cold["loaded"]["parameters"]["Ca_in"] = 0.02
    publish_fixture(tmp_path, summary, cold, accepted=fault != "acceptance")
    if fault == "unbound_summary":
        (tmp_path / "summary.json").write_text("{}")
    if fault is None:
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        assert len(modes.qualified_parent(tmp_path, case)[2]) == 4
        assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    else:
        with pytest.raises(ValueError):
            modes.qualified_parent(tmp_path, case)


@pytest.mark.parametrize("fault", [None, "declared_bond", "cold_bond"])
def test_actual_anchor_summary_can_omit_only_redundant_bond(tmp_path, case, fault):
    summary, cold = qualified_files(tmp_path, case)
    summary.pop("bond")  # run_paper_anchor writes Ca plus metrics; Bo is in cold parameters/packet.
    if fault == "declared_bond":
        summary["bond"] = 1.57
    if fault == "cold_bond":
        cold["loaded"]["parameters"]["Bo"] = 1.57
    publish_fixture(tmp_path, summary, cold)
    if fault is None:
        accepted, _, _ = modes.qualified_parent(tmp_path, case)
        assert "bond" not in accepted
    else:
        with pytest.raises(ValueError, match="forcing"):
            modes.qualified_parent(tmp_path, case)


@pytest.mark.parametrize("typed_i", [False, True])
@pytest.mark.parametrize(
    "fault",
    [None, "producer_hash", "producer_name", "ancestor", "helper", "helper_declaration", "method"],
)
def test_seed_i_producer_requires_committed_hash_and_ancestor(monkeypatch, fault, typed_i):
    old_commit, current_commit = "a" * 40, "b" * 40
    sources = {}
    for name in modes.runner.METHOD_FILES:
        digest = modes.sha(modes.ROOT / name)
        sources[name] = {"sha256": digest, "committed_sha256": digest, "committed_matches": True}
    current = {
        "commit": current_commit,
        "head": current_commit,
        "clean": True,
        "status": [],
        "sources": sources,
    }
    old = {**current, "commit": old_commit, "head": old_commit}
    producer_bytes = b"software-only committed seed producer"
    producer_hash = hashlib.sha256(producer_bytes).hexdigest()
    parent = {
        "component_commit": old_commit,
        "component": old,
        "runner": {"path": "/software/parent/runners/run_paper_seed_i.py", "sha256": producer_hash},
        "runner_path": "/software/parent/runners/run_paper_seed_i.py",
        "runner_sha256": producer_hash,
    }
    if typed_i:
        path = "/software/parent/runners/run_paper_anchor_i.py"
        producer_hash = modes.sha(modes.ROOT / modes.anchor_i.ENTRYPOINT)
        parent.update(
            runner={"path": path, "sha256": producer_hash},
            runner_path=path,
            runner_sha256=producer_hash,
        )
        parent["implementation"] = {
            name: {"path": "/software/parent/" + name, "sha256": modes.sha(modes.ROOT / name)}
            for name in (*modes.HELPERS, *modes.I_HELPERS)
        }
    if fault == "producer_hash":
        parent["runner"]["sha256"] = "wrong-producer"
        parent["runner_sha256"] = "wrong-producer"
    elif fault == "producer_name":
        parent["runner"]["path"] = parent["runner_path"] = "/software/parent/runners/unknown.py"
    elif fault == "helper_declaration":
        parent.setdefault("implementation", {})[modes.HELPERS[0]] = {
            "path": "/software/parent/" + modes.HELPERS[0],
            "sha256": "mutated-helper",
        }
    ancestor_calls = []

    def git(root, *args):
        if args[0] == "merge-base":
            ancestor_calls.append(args)
            if fault == "ancestor":
                raise ValueError("software ancestor refusal")
            return b""
        assert args[0] == "show"
        name = args[1].split(":", 1)[1]
        if args[1].startswith(old_commit + ":") and (
            fault == "helper"
            and name == modes.HELPERS[0]
            or fault == "method"
            and name == modes.runner.METHOD_FILES[0]
        ):
            return b"mutated frozen source"
        if name == "runners/run_paper_seed_i.py" and not typed_i:
            return producer_bytes
        return (modes.ROOT / name).read_bytes()

    monkeypatch.setattr(modes.diagnostic, "git", git)
    monkeypatch.setattr(modes.runner, "_git", git)
    if fault is None:
        implementation = modes.source_contract(parent, current, typed_i=typed_i)
        assert "runners/run_paper_modes.py" in implementation
        assert ancestor_calls == [("merge-base", "--is-ancestor", old_commit, current_commit)]
    else:
        with pytest.raises(ValueError):
            modes.source_contract(parent, current, typed_i=typed_i)


def layout_fixture(mode):
    names = ["liquid/mesh_y"] * 10
    incidence = {
        "globals/U": [0],
        "gas_volume/pressure": [1],
        "liquid/pressure": [2],
        "gas/pressure": [3],
        "liquid/velocity_phi": [4],
        "gas/velocity_phi": [5],
        "liquid/interface/_kin_bc": [6],
        "liquid/interface/_lagr_velo_connvelocity_phi": [7],
    }
    forced = [0, 1] if mode else []
    selectors = set(modes.GLOBALS) if mode else set()
    return np.arange(10), names, incidence, forced, selectors


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "m0_removed_global",
        "m1_free_global",
        "gas_pressure_removed",
        "missing_phi_connection",
        "bad_id",
    ],
)
def test_native_incidence_and_actual_global_constraints_are_required(fault):
    mode = 0 if fault == "m0_removed_global" else 1
    types, names, incidence, forced, selectors = layout_fixture(mode)
    if fault == "m0_removed_global":
        forced, selectors = [0, 1], set(modes.GLOBALS)
    if fault == "m1_free_global":
        forced = []
    if fault == "gas_pressure_removed":
        forced.append(3)
    if fault == "missing_phi_connection":
        incidence["liquid/interface/_lagr_velo_connvelocity_phi"] = []
    if fault == "bad_id":
        incidence["globals/U"] = [-1]
    if fault is None:
        result = modes.validate_layout(types, names, incidence, forced, selectors, 10, mode)
        assert result["globals"]["gas_volume/pressure"] == [1]
    else:
        with pytest.raises(ValueError):
            modes.validate_layout(types, names, incidence, forced, selectors, 10, mode)


def small_pencil():
    j = sparse.diags([1.0, 1.0, -2.0], format="csr")
    m = sparse.diags([0.0, 0.0, 1.0], format="csr")
    layout = {"forced": [0, 1], "globals": {"globals/U": [0], "gas_volume/pressure": [1]}}
    return j, m, layout


def test_actual_forced_rows_and_signed_original_pencil_residual():
    j, m, layout = small_pencil()
    assert modes.verify_pencil(j, m, layout)["mass_rows"] == [2]
    vector = np.array([0.0, 0.0, 1.0], dtype=complex)
    good = modes.diagnose_pair(-2 + 0j, vector, j, m, layout, 0.6716793862139855)
    assert good["pencil_qualified"]
    assert good["paper_growth"] < 0
    bad = modes.diagnose_pair(2 + 0j, vector, j, m, layout, 0.6716793862139855)
    assert not bad["pencil_qualified"]
    assert "original-pencil backward residual" in bad["rejection_reasons"]
    forced = modes.diagnose_pair(-2 + 0j, np.array([1.0, 0.0, 1.0]), j, m, layout, 1)
    assert "forced coefficients" in forced["rejection_reasons"]
    near_null = modes.diagnose_pair(1 + 0j, np.array([1.0, 0.0, 0.0]), j, m, layout, 1)
    assert "unresolved zero/near-null mass action" in near_null["rejection_reasons"]
    j[0, 0] = 0.0
    with pytest.raises(ValueError, match="forced row"):
        modes.verify_pencil(j, m, layout)


def test_returned_wrong_mode_label_refused():
    with pytest.raises(ValueError, match="labels"):
        modes.validate_returned(np.ones(8), np.ones((8, 3)), np.zeros(8), 3, 1)


def test_full_restoration_mismatch_is_fatal(monkeypatch):
    expected = {
        "mode": 1,
        "flags": {},
        "steady": [],
        "eigen_data_in_states": True,
        "actual_pins": {"sha256": "before"},
    }
    saved = {
        "signature": expected,
        "payload": b"software",
        "eigendata": {name: None for name in modes.EIGENDATA_FIELDS},
    }
    problem = SimpleNamespace(
        _azimuthal_mode_param_m=SimpleNamespace(value=1),
        actions_before_eigen_solve=lambda **kwargs: None,
        _restore_state=lambda payload: None,
        invalidate_cached_mesh_data=lambda: None,
    )
    monkeypatch.setattr(
        modes,
        "take_state",
        lambda problem: {"signature": {**expected, "actual_pins": {"sha256": "changed"}}},
    )
    with pytest.raises(modes.RestorationFailure, match="did not restore"):
        modes.restore_state(problem, saved)


def normal_fixture(eta):
    weights = [0.0, 1.0, 0.0]
    norm = np.sqrt(sum(w * abs(x) ** 2 for w, x in zip(weights, eta, strict=True)))
    return {
        "samples": [
            {
                "z": float(i),
                "r": float(w),
                "weight_surface": float(w),
                "eta_real": float((x / norm).real),
                "eta_imag": float((x / norm).imag),
            }
            for i, (w, x) in enumerate(zip(weights, eta, strict=True))
        ]
    }


def pair(index, value, normal, qualified=True):
    return {
        "index": index,
        "lambda_real": value.real,
        "lambda_imag": value.imag,
        "paper_growth": value.real,
        "physical_qualified": qualified,
        "normal": normal,
        "rejection_reasons": [] if qualified else ["software rejection"],
    }


def test_phase_invariance_preserves_complex_families_and_unqualified_competitors():
    first = normal_fixture([0j, 1 + 2j, 0j])
    second = normal_fixture([0j, (1 + 2j) * np.exp(0.9j), 0j])
    assert modes.compare_normals(first, second, 0)["passed"]
    left = {"shift": -0.001, "pairs": [pair(0, -2 + 3j, first), pair(1, -2 - 3j, first)]}
    right = {"shift": 0.001, "pairs": [pair(0, -2 + 3j, second), pair(1, -2 - 3j, second)]}
    good = modes.compare_searches(left, right, 0)
    assert good["matched_returned_envelope"]
    assert not good["global_dominance_certified"]
    left["pairs"].append(pair(2, 1 + 0j, first, qualified=False))
    bad = modes.compare_searches(left, right, 0)
    assert not bad["matched_returned_envelope"]
    assert bad["unqualified_higher_or_unresolved_returns"][0]["index"] == 2
    with pytest.raises(ValueError, match="geometry"):
        wrong = copy.deepcopy(second)
        wrong["samples"][1]["r"] += 0.1
        modes.compare_normals(first, wrong, 0)


@pytest.mark.parametrize("mode,accepted", [(0, False), (1, True)])
def test_oscillatory_conjugate_requirement_is_sector_specific(mode, accepted):
    normal = normal_fixture([0j, 1 + 2j, 0j])
    left = {"shift": -0.001, "pairs": [pair(0, -2 + 3j, normal)]}
    right = {"shift": 0.001, "pairs": [pair(0, -2 + 3j, normal)]}
    result = modes.compare_searches(left, right, mode)
    assert result["matched_returned_envelope"] is accepted
    assert result["conjugate_requirement_satisfied"] is accepted
    assert result["returned_envelope_conjugate_complete"] is False
    family = modes.returned_families(left["pairs"], mode)[0]
    assert family["possible_conjugate_partners"] == []
    assert family["conjugate_requirement_satisfied"] is accepted
    assert family["negative_sector"] == {"mode": -1, "searched": False, "conjugacy_verified": False}
    assert not result["global_dominance_certified"]
    # An actual returned opposite-frequency candidate is retained even in the complex +1 sector.
    both = [*left["pairs"], pair(1, -2 - 3j, normal)]
    assert modes.returned_families(both, mode)[0]["possible_conjugate_partners"] == [1]


def test_plot_failure_preserves_four_completed_searches_and_scientific_results(
    tmp_path, case, monkeypatch, capsys
):
    normal = normal_fixture([0j, 1 + 2j, 0j])
    searches = []

    def search(prepared, output, mode, shift):
        searches.append((mode, shift))
        output.mkdir()
        stage = {
            "mode": mode,
            "shift": shift,
            "pairs": [pair(0, -2 + (3j if mode else 0j), normal)],
            "software_fixture": True,
        }
        modes.write_json(output / "status.json", stage)
        return stage

    recorded = {}

    def broken_plot(searches, case, output):
        assert all(
            (output / f"m{mode}-shift-{index}" / "status.json").is_file()
            for mode in modes.MODES
            for index in range(2)
        )
        for name in ("result.json", "summary.json"):
            recorded[name] = (output / name).read_bytes()
        raise RuntimeError("software rendering failure")

    monkeypatch.setattr(modes, "spectral_stage", search)
    monkeypatch.setattr(modes, "plot_returned", broken_plot)
    prepared = {"case": case, "request": modes.request_contract(case, modes.SHIFTS, 8)}
    modes.runner.append_progress(tmp_path, {"event": "software-prepared"}, first=True)
    result = modes.execute(prepared, tmp_path)
    assert searches == [(mode, shift) for mode in modes.MODES for shift in modes.SHIFTS]
    assert result["numerical_acceptance"] is True
    assert result["scientific_status"] == "qualified_returned_envelopes_at_one_anchor"
    for name, before in recorded.items():
        assert (tmp_path / name).read_bytes() == before
        assert json.loads(before)["numerical_acceptance"] is True
    receipt = json.loads((tmp_path / "plot-receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["error"] == "software rendering failure"
    assert receipt["scientific_results_preserved"] is True
    assert receipt["native_replay_required"] is False
    assert "Plot failed; scientific results retained" in capsys.readouterr().err
    assert "plot-failed" in (tmp_path / "progress.jsonl").read_text()


def test_complex_m0_comparison_uses_real_two_dimensional_span():
    def normal(values):
        weights = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        values = np.asarray(values, dtype=complex)
        values /= np.sqrt(np.sum(weights * abs(values) ** 2))
        return {
            "samples": [
                {
                    "z": float(i),
                    "r": float(w),
                    "weight_surface": float(w),
                    "eta_real": float(v.real),
                    "eta_imag": float(v.imag),
                }
                for i, (w, v) in enumerate(zip(weights, values, strict=True))
            ]
        }

    values = np.array([0.0, 1.0 + 0j, 0.0 + 1j, 1.0 + 1j, 0.0])
    first, second = normal(values), normal(values * np.exp(0.7j))
    result = modes.compare_normals(first, second, 0)
    assert result["real_span_dimensions"] == [2, 2]
    assert result["passed"]
    assert min(result["principal_cosines"]) > 0.999999999
    wrong = normal([0.0, 1.0, 0.0, 1.0, 0.0])
    assert not modes.compare_normals(first, wrong, 0)["passed"]


def test_manifest_binds_own_entrypoint_not_an_imported_helper(case):
    own = {"path": str(modes.ROOT / "runners/run_paper_modes.py"), "sha256": "own"}
    prepared = {
        "case": case,
        "implementation": {
            "runners/run_paper_modes.py": own,
            "runners/run_paper_anchor.py": {"sha256": "plain-helper"},
        },
        "component": {"commit": "source"},
        "environment": {},
        "request": {},
        "restart": {"packet": {"sha256": "packet"}},
        "cold": {"loaded": {"serialized_state": {"payload_sha256": "history"}}},
    }
    path = modes.ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json"
    actual = modes.manifest(SimpleNamespace(case=path), prepared, ["--check"])
    assert actual["runner"] == own
    assert actual["argv"][1] == own["path"]
    assert actual["qualification_scope"] == modes.SCOPE


@pytest.mark.parametrize("branch", ["I", "III"])
def test_cold_dispatch_uses_exact_helper_and_full_history(monkeypatch, tmp_path, case, branch):
    if branch == "I":
        case["physics"]["ca_in"] = modes.anchor_i.TARGET_CA
    problem = SimpleNamespace()
    cold = {
        "actual_pins": {"pin": True},
        "loaded": {"serialized_state": {"payload_sha256": "all-history"}},
    }
    prepared = {"case": case, "restart": {"packet": "exact-packet"}, "cold": cold}
    calls = []

    def typed(*args):
        calls.append(("I", args))
        return {"verified": True, "bc_preparation": {"verified": True}}

    def plain(*args):
        calls.append(("III", args))
        return {"verified": True}

    monkeypatch.setattr(modes.anchor, "restore_checked", plain)
    monkeypatch.setattr(modes.requalification, "restore_checked", typed)
    assert modes.restore_parent(problem, prepared, tmp_path)["verified"]
    assert calls == [
        (branch, (problem, case, prepared["restart"], cold["actual_pins"], "all-history"))
    ]
    if branch == "I":
        assert problem._seed_requalification_receipts_directory == tmp_path / "typed-loads"
    else:
        assert not vars(problem)


@pytest.fixture
def typed_i_bundle(tmp_path, case, monkeypatch):
    case["physics"]["ca_in"] = modes.anchor_i.TARGET_CA
    case["solve"]["relax_time"] = 8.0
    summary, cold = qualified_files(tmp_path, case)
    summary.update(
        status="completed",
        checkpoint="packet.json",
        branch_identity_verified=False,
        eigenanalysis_performed=False,
    )
    mesh = tmp_path / "mesh.msh"
    mesh.write_bytes(b"software mesh")

    def integer(value):
        return struct.pack("<q", value)

    def string(value):
        return integer(len(value)) + value.encode("ascii")

    state = tmp_path / "state.dump"
    state.write_bytes(
        string("pyoomph_dump")
        + string("0.1.6")
        + string("global")
        + struct.pack("<dqqqq", 8.0, 9, 0, 6, 1)
        + string("mesh.msh")
        + integer(0)
        + integer(2)
        + struct.pack("<ddqdd", 0.1, 0.2, 20, 0.125, 8.0)
        + b"all opaque fields and histories"
        + modes.runner._bc_cache_trailer("stationary")
    )
    packet = tmp_path / "packet.json"
    packet.write_text("{}")
    restart = {
        "packet": modes.regular_record(packet),
        "state": modes.regular_record(state),
        "meshes": [modes.regular_record(mesh)],
        "mesh_reference_base": str(tmp_path),
        "stage": {"bc_mode": "stationary"},
    }
    history = modes.runner.snapshot_bc_cache_signature(
        state.read_bytes(), mesh_reference_base=tmp_path
    )
    cold["loaded"].update(
        serialized_state=history,
        bc_cache_marker="stationary",
        layout={"n": 2},
        arrays={"all": "fields"},
        volume=4.0,
        frame_speed=0.01,
        time=8.0,
    )
    restart["roundtrip"] = {"after": copy.deepcopy(cold["loaded"]), "layout": {"n": 2}}
    metadata = modes.requalification.cold.serialized_metadata(state.read_bytes())
    cold.update(
        bc_preparation={"verified": True},
        intervention={
            "verified": True,
            "kind": "typed-native-endtime-restoration",
            "saved_endtime": 8.0,
            "api": modes.requalification.cold.API,
        },
        time_history={
            key: metadata[key] for key in ("time", "dts", "unsteady_steps", "suggested_next_dt")
        },
    )
    seed = {
        key: {"path": str(tmp_path / "qualified-seed" / name), "sha256": "seed-hash"}
        for key, name in (
            ("parent_manifest", "run-manifest.json"),
            ("packet", "packet.json"),
            ("state", "state.dump"),
        )
    }
    contract = {
        "parent_case": {"path": "seed-case", "sha256": "case"},
        "ca_ladder": {"path": str(tmp_path / "ladder.json")},
    }
    admitted = {
        "restart": copy.deepcopy(seed),
        "target": copy.deepcopy(case),
        "contract": copy.deepcopy(contract),
        "records": [{"path": "original-failed-producer", "sha256": "original-hash"}],
    }
    seed["compatibility"] = {
        "parent_commit": "seed-source",
        "current_commit": "I-source",
        "physics_ast_sha256": "same-physics",
    }
    admitted["restart"]["compatibility"] = {
        **seed["compatibility"],
        "current_commit": "spectrum-source",
    }
    parent = {
        "schema": "pyoomph-runner-manifest-v1",
        "runner": {"path": "/source/runners/run_paper_anchor_i.py"},
        "anchor_stage": {"kind": "stationary_I_candidate"},
        "parent_acceptance_changed": False,
        "parent_restart": seed,
        "case_transition_contract": contract,
        "construction_case": contract["parent_case"],
        "case_path": str(tmp_path / "case.json"),
        "implementation": {"typed-helper": "bound"},
        "component_commit": "I-source",
    }
    result = {
        "schema": "private-paper-stationary-anchor-v1",
        "completion_reached": True,
        "numerical_acceptance": True,
        "cold_load_verified": True,
        "canonical_checks": summary["checks"],
        "cold_load_checks": summary["cold_load_checks"],
        "ca_in": case["physics"]["ca_in"],
        "parent_acceptance_changed": False,
        "parents_unchanged": True,
        "branch_identity_verified": False,
        "eigenanalysis_performed": False,
        "checkpoint": {"resume_packet": "packet.json"},
        "terminal": {"after_polish": {"actual_pins": copy.deepcopy(cold["actual_pins"])}},
    }
    preparation = {
        "contract": copy.deepcopy(contract),
        "inputs": copy.deepcopy(admitted["records"]),
        "numerical_acceptance": False,
        "implementation": parent["implementation"],
    }
    calls = []

    def seed_admission(args):
        calls.append(args)
        assert args.manifest == tmp_path / "qualified-seed" / "run-manifest.json"
        assert args.packet == tmp_path / "qualified-seed" / "packet.json"
        assert args.state == tmp_path / "qualified-seed" / "state.dump"
        return admitted

    monkeypatch.setattr(modes.anchor_i, "preflight", seed_admission)
    values = {
        "run-manifest.json": parent,
        "summary.json": summary,
        "cold-load.json": cold,
        "result.json": result,
        "preparation.json": preparation,
    }

    def publish():
        for name, value in values.items():
            (tmp_path / name).write_text(json.dumps(value))
        restart["parent_manifest"] = modes.regular_record(tmp_path / "run-manifest.json")
        files = {
            name: modes.sha(tmp_path / name)
            for name in (*values, "packet.json", "state.dump", "mesh.msh")
        }
        (tmp_path / "completion.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "numerical_acceptance": True,
                    "parent_acceptance_changed": False,
                    "files": files,
                }
            )
        )

    return SimpleNamespace(
        runtime=tmp_path,
        case=case,
        parent=parent,
        restart=restart,
        summary=summary,
        cold=cold,
        result=result,
        preparation=preparation,
        admitted=admitted,
        publish=publish,
        calls=calls,
    )


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "producer",
        "schema",
        "packet",
        "pins",
        "fields",
        "history",
        "time",
        "forcing",
        "bc",
        "api",
        "endtime",
        "seed_lineage",
        "consumer_revision",
        "physics_contract",
        "admission",
        "inputs",
        "helper",
        "failure",
        "missing_result",
        "mutated_file",
    ],
)
def test_typed_i_admission_refuses_mutated_packets_and_lineage(typed_i_bundle, fault):
    b = typed_i_bundle
    if fault == "producer":
        b.parent["runner"]["path"] = "/source/runners/run_paper_anchor_unknown.py"
    elif fault == "schema":
        b.result["schema"] = modes.requalification.SCHEMA
    elif fault == "packet":
        b.result["checkpoint"]["resume_packet"] = "mesh.msh"
    elif fault == "pins":
        b.cold["actual_pins"] = {"sha256": "different"}
    elif fault == "fields":
        b.cold["loaded"]["arrays"] = {"all": "mutated"}
    elif fault == "history":
        b.cold["loaded"]["serialized_state"]["payload_sha256"] = "partial-history"
    elif fault == "time":
        b.cold["time_history"]["dts"] = [0.9]
    elif fault == "forcing":
        b.cold["loaded"]["parameters"]["Ca_in"] = 0.02
    elif fault == "bc":
        b.cold["bc_preparation"]["verified"] = False
    elif fault == "api":
        b.cold["intervention"]["api"] = {"unreviewed": True}
    elif fault == "endtime":
        b.cold["intervention"]["saved_endtime"] = 7.0
    elif fault == "seed_lineage":
        b.parent["parent_restart"]["state"]["sha256"] = "different"
    elif fault == "consumer_revision":
        b.parent["parent_restart"]["compatibility"]["current_commit"] = "unrelated-source"
    elif fault == "physics_contract":
        b.parent["parent_restart"]["compatibility"]["physics_ast_sha256"] = "mutated-physics"
    elif fault == "admission":
        b.preparation["contract"] = {}
    elif fault == "inputs":
        b.preparation["inputs"] = []
    elif fault == "helper":
        b.preparation["implementation"] = {}
    b.publish()
    if fault == "failure":
        (b.runtime / "failure.json").write_text("{}")
    elif fault == "missing_result":
        completion = json.loads((b.runtime / "completion.json").read_text())
        completion["files"].pop("result.json")
        (b.runtime / "completion.json").write_text(json.dumps(completion))
    elif fault == "mutated_file":
        (b.runtime / "result.json").write_text("{}")
    if fault is None:
        _, cold, records = modes.qualified_parent(
            b.runtime, b.case, parent=b.parent, restart=b.restart
        )
        assert cold == b.cold and b.admitted["records"][0] in records
        assert len(b.calls) == 1
    else:
        with pytest.raises(ValueError):
            modes.qualified_parent(b.runtime, b.case, parent=b.parent, restart=b.restart)


def mode_stream(
    reference,
    marker="init",
    pairs=0,
    *,
    history=b"synthetic-history-A",
    eigen_values=None,
    vectors=None,
    labels=None,
):
    """Real pinned header/eigen-block framing around synthetic software-only fields."""

    def integer(value):
        return struct.pack("<q", value)

    def string(value):
        return integer(len(value)) + value.encode("ascii")

    def array(value):
        data = io.BytesIO()
        np.save(data, value, allow_pickle=False)
        compressed = zlib.compress(data.getvalue(), 6)
        return integer(len(compressed)) + compressed

    head = (
        string("pyoomph_dump")
        + string("0.1.6")
        + string("global")
        + struct.pack("<d", 8.0)
        + integer(9)
        + integer(0)
        + integer(6)
        + integer(1)
        + string(reference)
        + integer(1)
        + integer(2)
    )
    body = history + integer(1) + string(marker) + integer(pairs)
    if pairs:
        body += array(np.arange(pairs, dtype=complex) if eigen_values is None else eigen_values)
        body += array(np.ones((pairs, 2), dtype=complex) if vectors is None else vectors)
        body += integer(1) + array(np.zeros(pairs, dtype=np.int32) if labels is None else labels)
    return head + body + integer(0) + string("EOF_pyoomph")


@pytest.mark.parametrize("marker", ["init", "stationary", "transient", "eigen"])
@pytest.mark.parametrize("pairs", [0, 8])
def test_full_mode_stream_covers_zero_and_eight_pair_payloads(tmp_path, marker, pairs):
    mesh = tmp_path / "mesh.msh"
    mesh.write_bytes(b"synthetic-mesh-only")
    stream = mode_stream("mesh.msh", marker, pairs)
    signature = modes.full_mode_snapshot_signature(
        stream, mesh_reference_base=tmp_path, mode=0, bc_marker=marker
    )
    assert signature["schema"] == "pyoomph-full-mode-snapshot-v1"
    assert signature["bc_cache_marker"] == marker
    assert signature["mesh_content_sha256"] == [modes.sha(mesh)]
    history_changed = stream.replace(b"synthetic-history-A", b"synthetic-history-B")
    assert (
        modes.full_mode_snapshot_signature(
            history_changed, mesh_reference_base=tmp_path, mode=0, bc_marker=marker
        )["full_stream_sha256"]
        != signature["full_stream_sha256"]
    )
    if pairs:
        changed = mode_stream(
            "mesh.msh", marker, pairs, eigen_values=np.arange(pairs, dtype=complex) + 0.5j
        )
        assert (
            modes.full_mode_snapshot_signature(
                changed, mesh_reference_base=tmp_path, mode=0, bc_marker=marker
            )["full_stream_sha256"]
            != signature["full_stream_sha256"]
        )
        with pytest.raises(ValueError, match="zero eigen/continuation"):
            modes.runner.snapshot_bc_cache_signature(stream, mesh_reference_base=tmp_path)
    elif marker != "eigen":
        assert (
            modes.runner.snapshot_bc_cache_signature(stream, mesh_reference_base=tmp_path)[
                "bc_cache_marker"
            ]
            == marker
        )


def test_reference_normalization_preserves_full_stream_and_requires_escape_authority(tmp_path):
    base = tmp_path / "cwd"
    base.mkdir()
    mesh = base / "mesh.msh"
    mesh.write_bytes(b"synthetic-mesh")

    def signature(reference, authority=None):
        return modes.full_mode_snapshot_signature(
            mode_stream(reference, "eigen", 8),
            mesh_reference_base=base,
            mode=1,
            bc_marker="eigen",
            mesh_authority=authority,
        )

    assert signature("mesh.msh") == signature(str(mesh))
    external = tmp_path / "outside.msh"
    external.write_bytes(b"bound-parent-mesh")
    with pytest.raises(ValueError, match="escaped"):
        signature("../outside.msh")
    authority = [{"path": str(external), "sha256": modes.sha(external)}]
    assert signature("../outside.msh", authority) == signature(str(external), authority)
    with pytest.raises(ValueError, match="authority"):
        signature("mesh.msh", authority)
    with pytest.raises(ValueError, match="footer"):
        modes.full_mode_snapshot_signature(
            mode_stream("mesh.msh")[:-1], mesh_reference_base=base, mode=0, bc_marker="init"
        )
    with pytest.raises(ValueError, match="regular"):
        signature("missing.msh")
    alias = base / "alias.msh"
    alias.symlink_to(mesh)
    with pytest.raises(ValueError, match="regular"):
        signature("alias.msh")


@pytest.mark.parametrize("pairs", [0, 8])
def test_known_zero_or_eight_pair_cache_and_full_state_restore_exactly(
    tmp_path, monkeypatch, pairs
):
    mesh = tmp_path / "mesh.msh"
    mesh.write_bytes(b"synthetic-mesh")
    monkeypatch.chdir(tmp_path)
    physical = {"values": np.array([1.0, 2.0])}
    saved_payloads = {}
    problem = SimpleNamespace(
        eigen_data_in_states=False,
        _first_step=False,
        _taken_already_an_unsteady_step=True,
        _last_step_was_stationary=True,
        _last_bc_setting="eigen" if pairs else "init",
        _azimuthal_mode_param_m=SimpleNamespace(value=1 if pairs else 0),
        _last_eigenvalues=np.arange(pairs, dtype=np.complex128),
        _last_eigenvectors=np.ones((pairs, 2), dtype=np.complex128)
        if pairs
        else np.array([], dtype=np.complex128),
        _last_eigenvalues_m=np.ones(pairs, dtype=np.int32) if pairs else None,
        _last_eigenvalues_k=None,
        ntime_stepper=lambda: 0,
        get_dof_description=lambda: (np.array([0, 1]), ["field/a", "field/b"]),
        get_current_dofs=lambda: (physical["values"].copy(), np.array([False, True])),
        get_current_pinned_values=lambda positions: np.array([0.0]),
        interface_polyline=lambda: (np.array([0.0, 1.0]), np.array([0.0, 0.5])),
        get_global_parameter=lambda name: SimpleNamespace(value=1.56 if name == "Bo" else -0.00315),
        observable=lambda *args: 4.0,
        frame_speed=lambda: 0.02,
        get_current_time=lambda **kwargs: 8.0,
        invalidate_cached_mesh_data=lambda: None,
        actions_before_stationary_solve=lambda **kwargs: None,
        actions_before_eigen_solve=lambda **kwargs: None,
    )

    def snapshot():
        count = len(problem._last_eigenvalues)
        stream = mode_stream(
            "mesh.msh",
            problem._last_bc_setting,
            count,
            history=physical["values"].tobytes(),
            eigen_values=problem._last_eigenvalues,
            vectors=problem._last_eigenvectors,
            labels=problem._last_eigenvalues_m,
        )
        saved_payloads[stream] = (physical["values"].copy(), problem._last_bc_setting)
        return stream

    def load(stream):
        # Mirrors the relevant pinned zero-count loader behaviour: cache remains stale.
        physical["values"], problem._last_bc_setting = saved_payloads[stream]

    problem._snapshot_state = snapshot
    problem._restore_state = load
    monkeypatch.setattr(
        modes.diagnostic, "actual_pins", lambda problem: {"sha256": "unchanged-pins"}
    )
    saved = modes.take_state(problem)
    assert saved["signature"]["last_eigendata"]["_last_eigenvalues"]["shape"] == [pairs]
    original_cache_signature = modes.eigendata_signature(saved["eigendata"])
    problem._last_eigenvalues = np.full(8, 99 + 2j, dtype=np.complex128)
    problem._last_eigenvectors = np.full((8, 2), -3 + 7j, dtype=np.complex128)
    problem._last_eigenvalues_m = np.full(8, 7, dtype=np.int32)
    physical["values"] += 5
    assert modes.eigendata_signature(modes.copy_eigendata(problem)) != original_cache_signature
    receipt = modes.restore_state(problem, saved)
    assert receipt["verified"] is True
    assert modes.eigendata_signature(modes.copy_eigendata(problem)) == original_cache_signature
    assert np.array_equal(physical["values"], [1.0, 2.0])
    problem._last_eigenvalues_k = np.array([], dtype=np.float64)
    assert modes.take_state(problem)["signature"] != saved["signature"]
