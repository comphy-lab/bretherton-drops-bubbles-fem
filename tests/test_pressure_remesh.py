"""Software contracts for the controlled remesh runner; no native solver runs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from problems import paper_bubble
from runners import run_pressure_remesh as remesh


@pytest.fixture
def case():
    return paper_bubble.read_case(
        remesh.ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json"
    )


def metrics():
    return {
        "stationary_residual_linf": 2e-9,
        "lower_pressure_rms": 2e-9,
        "lower_flux": 0.03,
        "upper_flux": 0.03,
        "gas_volume": 1.0,
        "surface_volume": 1.0,
        "target_volume": 1.0,
    }


@pytest.mark.parametrize(
    "gate,field,value",
    [
        ("stationary", "stationary_residual_linf", 1.01e-8),
        ("lower_static_pressure", "lower_pressure_rms", 1.37e-5),
        ("flux", "upper_flux", 0.03000002),
        ("volume", "target_volume", 1.000002),
        ("volume_integrals", "surface_volume", 1.000002),
    ],
)
def test_each_original_gate_refuses_the_measured_fault(case, gate, field, value):
    values = metrics()
    assert all(remesh.canonical_checks(case, values).values())
    values[field] = value
    result = remesh.canonical_checks(case, values)
    assert result[gate] is False
    assert all(passed for name, passed in result.items() if name != gate)


def test_nonfinite_gate_metric_is_refused(case):
    values = metrics()
    values["gas_volume"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite canonical metric"):
        remesh.canonical_checks(case, values)


def fields():
    geometry = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    bulk = {
        "coordinate_x": geometry[:, 0].copy(),
        "coordinate_y": geometry[:, 1].copy(),
        "pressure": geometry[:, 0].copy(),
        "velocity_x": np.zeros(6),
        "velocity_y": np.zeros(6),
        "connectivity": np.array([[0, 1, 2, 3, 4, 5]]),
        "element_types": np.array([9]),
    }
    port_indices = np.array([0, 3, 1])
    lower = {
        name: value[port_indices].copy()
        for name, value in bulk.items()
        if name not in {"connectivity", "element_types"}
    }
    lower.update(connectivity=np.array([[0, 1, 2]]), element_types=np.array([2]))
    return lower, bulk


def test_geometry_probe_detects_an_interior_midside_departure():
    lower, bulk = fields()
    assert remesh.port_adjacent_geometry(lower, bulk, np)["max_midside_departure_l2"] == 0
    bulk["coordinate_y"][5] += 4e-5
    result = remesh.port_adjacent_geometry(lower, bulk, np)
    assert result["max_midside_departure_l2"] == pytest.approx(4e-5, abs=1e-16)
    assert result["max_abs_axial_departure"] == pytest.approx(4e-5, abs=1e-16)
    assert result["max_abs_radial_departure"] == 0
    assert result["edges"][0]["bulk_triangle"] == 0


def test_ambiguous_port_adjacency_is_refused():
    lower, bulk = fields()
    bulk["connectivity"] = np.vstack((bulk["connectivity"], bulk["connectivity"]))
    bulk["element_types"] = np.array([9, 9])
    with pytest.raises(ValueError, match="exactly one adjacent triangle"):
        remesh.port_adjacent_geometry(lower, bulk, np)


def test_port_and_bulk_pressure_must_represent_the_same_native_values():
    lower, bulk = fields()
    lower["pressure"][0] = 1e-5
    with pytest.raises(ValueError, match="port/bulk field mismatch: pressure"):
        remesh.port_adjacent_geometry(lower, bulk, np)


def test_one_explicit_remesh_precedes_the_unchanged_polish(case, tmp_path):
    calls = []

    def force_remesh():
        calls.append("force")
        return True

    def polish(problem, actual_case, output, label, **options):
        assert actual_case is case
        assert output == tmp_path
        assert label == "terminal_remesh"
        assert options == {"initial_relaxation": 1.0}
        calls.append("polish")
        return {"status": "converged", "updates_total": 2, "remeshes_total": 0}

    def observe(phase):
        calls.append(phase)
        return {"phase": phase}

    original = copy.deepcopy(case)
    result = remesh.remesh_and_polish(
        SimpleNamespace(force_remesh=force_remesh),
        SimpleNamespace(polish_confined_state=polish),
        case,
        tmp_path,
        observe,
    )
    assert calls == ["before", "force", "after-remesh", "polish", "after-polish"]
    assert result["explicit_force_remesh_calls"] == 1
    assert result["state_preserved_across_intervention"] is False
    assert case == original


def test_a_missing_remesh_or_failed_polish_never_gets_an_extra_retry(case, tmp_path):
    calls = []

    def force():
        calls.append("force")
        return False

    def polish(*args, **options):
        calls.append("polish")
        raise RuntimeError("software budget failure")

    problem, model = (
        SimpleNamespace(force_remesh=force),
        SimpleNamespace(polish_confined_state=polish),
    )
    with pytest.raises(ValueError, match="did not replace a mesh"):
        remesh.remesh_and_polish(problem, model, case, tmp_path, lambda phase: phase)
    assert calls == ["force"]
    calls.clear()

    def successful_force():
        calls.append("force")
        return True

    problem.force_remesh = successful_force
    with pytest.raises(RuntimeError, match="software budget failure"):
        remesh.remesh_and_polish(problem, model, case, tmp_path, lambda phase: phase)
    assert calls == ["force", "polish"]


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("fields", "fields/positions/masks"),
        ("layout", "layout"),
        ("pins", "actual pins"),
        ("forcing", "forcing"),
        ("marker", "BC cache"),
    ],
)
def test_cold_load_refuses_changed_retained_native_identity(case, fault, expected):
    signature = {
        "bc_cache_marker": "stationary",
        "parameters": {"Bo": 1.56, "Ca_in": 0.02},
        "layout": {"ndof": 6},
        "arrays": {"free_dofs": {"sha256": "software-fields"}},
        "volume": 1.0,
        "frame_speed": 0.02,
        "time": 8.0,
    }
    restart = {
        "stage": {"bc_mode": "stationary"},
        "roundtrip": {
            "layout": copy.deepcopy(signature["layout"]),
            "after": copy.deepcopy(signature),
        },
    }
    pins = {"sha256": "software-pins"}
    remesh.check_loaded_state(signature, restart, case, pins, pins)
    expected_pins = copy.deepcopy(pins)
    if fault == "fields":
        signature["arrays"]["free_dofs"]["sha256"] = "changed"
    elif fault == "layout":
        signature["layout"]["ndof"] = 7
    elif fault == "pins":
        pins["sha256"] = "changed"
    elif fault == "forcing":
        signature["parameters"]["Bo"] = 1.55
    else:
        signature["bc_cache_marker"] = "init"
    with pytest.raises(ValueError, match=expected):
        remesh.check_loaded_state(signature, restart, case, pins, expected_pins)


def test_causal_predictions_are_separate_from_scientific_acceptance(case):
    after = {**metrics(), "frame_speed": 0.02, "gas_pressure_mean": 1.0}
    result = {
        "before": {"metrics": copy.deepcopy(after)},
        "after_remesh": {"port_adjacent_geometry": {"max_midside_departure_l2": 0.0}},
        "after_polish": {
            "metrics": after,
            "port_adjacent_geometry": {"max_midside_departure_l2": 2e-12},
        },
    }
    after["lower_pressure_rms"] = 5e-7
    assert all(remesh.canonical_checks(case, after).values())
    predictions = remesh.causal_predictions(result)
    assert predictions["checks"]["pressure_below_1e_minus_8"] is False
    assert predictions["checks"]["polished_midsides_at_1e_minus_12"] is False
    assert predictions["checks"]["fresh_midsides_at_1e_minus_12"] is True


def prepared(tmp_path, case):
    parent = tmp_path / "parent"
    parent.mkdir()
    case_file = parent / "case.json"
    case_file.write_text(json.dumps(case))
    args = SimpleNamespace(
        source=remesh.ROOT,
        case=case_file,
        manifest=parent / "run-manifest.json",
        packet=parent / "resume.json",
        state=parent / "state.dump",
        out=tmp_path / "new-output",
        check=False,
    )
    restart = {
        "parent_manifest": {"path": str(args.manifest)},
        "compatibility": {},
        "stage": {"bond_index": 16},
        "mesh_exchange": {},
    }
    component = {"commit": "software-source"}
    return args, (case, restart, {}, component, [], {})


def test_actual_manifest_identifies_this_entry_point(case, tmp_path):
    args, bundle = prepared(tmp_path, case)
    actual = remesh.run_manifest(args, bundle[0], bundle[1], bundle[2], bundle[3], bundle[5])
    script = Path(remesh.__file__).resolve()
    assert actual["runner"] == {"path": str(script), "sha256": remesh.sha(script)}
    assert actual["runner_path"] == str(script)
    assert "run_pressure_diagnostic.py" not in actual["runner_path"]
    assert actual["intervention"]["stage_update_limit"] == 40
    assert actual["intervention"]["total_update_limit"] == 160
    assert actual["intervention"]["polish_remesh_limit"] == 3


def arguments(args):
    return [
        item
        for name in ("source", "case", "manifest", "packet", "state", "out")
        for item in ("--" + name, str(getattr(args, name)))
    ]


def test_metadata_check_and_failure_protocol_import_no_native_solver(case, tmp_path, monkeypatch):
    args, bundle = prepared(tmp_path, case)
    monkeypatch.setattr(remesh, "preflight", lambda options: bundle)
    before = set(sys.modules)
    assert remesh.main([*arguments(args), "--check"]) == 0
    assert not args.out.exists()
    observed = []

    def fail(options, actual, output):
        manifest = json.loads((output / "run-manifest.json").read_text())
        assert manifest["runner_path"] == str(Path(remesh.__file__).resolve())
        assert (output / "preparation.json").is_file()
        observed.append("provenance precedes execution")
        raise RuntimeError("software-only bounded polish failure")

    monkeypatch.setattr(remesh, "execute", fail)
    with pytest.raises(RuntimeError, match="software-only bounded polish failure"):
        remesh.main(arguments(args))
    assert observed == ["provenance precedes execution"]
    failure = json.loads((args.out / "failure.json").read_text())
    assert failure["numerical_acceptance"] is False
    assert failure["parents_unchanged"] is True
    assert not (args.out / "completion.json").exists()
    events = [json.loads(line) for line in (args.out / "progress.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["prepared", "failure"]
    assert not any(
        name == "pyoomph" or name.startswith("pyoomph.") for name in set(sys.modules) - before
    )
    assert args.out.stat().st_mode & 0o777 == 0o700
    assert (args.out / "failure.json").stat().st_mode & 0o777 == 0o600


def test_occupied_output_preserves_existing_content(case, tmp_path, monkeypatch):
    args, bundle = prepared(tmp_path, case)
    args.out.mkdir()
    sentinel = args.out / "existing-receipt.json"
    sentinel.write_bytes(b"unchanged software fixture")
    monkeypatch.setattr(remesh, "preflight", lambda options: bundle)
    with pytest.raises(ValueError, match="new or empty"):
        remesh.main(arguments(args))
    assert sentinel.read_bytes() == b"unchanged software fixture"
    assert list(args.out.iterdir()) == [sentinel]


@pytest.mark.parametrize("changed", ["signature", "pins"])
def test_sampling_refusal_retains_full_phase_bound_delta(case, tmp_path, monkeypatch, changed):
    """A first-read mutation remains a refusal and leaves its direct signature evidence."""
    lower, bulk = fields()
    before = {
        "arrays": {"pinned_values_with_positions": {"sha256": "software-before"}},
        "serialized_state": {"payload_sha256": "software-payload"},
    }
    after = copy.deepcopy(before)
    pins_before = {"sha256": "software-pins", "flags_count_with_shared_node_repeats": 6}
    pins_after = copy.deepcopy(pins_before)
    if changed == "signature":
        after["arrays"]["pinned_values_with_positions"]["sha256"] = "software-after"
        message = "sampling changed native state"
    else:
        pins_after["sha256"] = "software-changed-pins"
        message = "sampling changed actual pins"
    signatures, pin_records = iter((before, after)), iter((pins_before, pins_after))
    model = SimpleNamespace(
        _stationary_state_signature=lambda *args: next(signatures),
        _stationary_residual=lambda problem: 2e-9,
    )

    def observable(name, domain="liquid"):
        if domain == "liquid/lower":
            return {
                "area": math.pi,
                "pressure_squared": math.pi / 2,
                "pressure": 2 * math.pi / 3,
                "flux": 0.03,
            }[name]
        return {"volume": 1.0, "surface_volume": 1.0, "pressure": 1.0, "flux": 0.03}[name]

    problem = SimpleNamespace(
        observable=observable,
        frame_speed=lambda: 0.02,
        geometry=SimpleNamespace(volume=1.0),
        get_ode=lambda name: SimpleNamespace(get_value=lambda *args, **kwargs: 0.0),
        interface_polyline=lambda: (np.array([0.0, 1.0]), np.array([0.0, 0.0])),
    )
    monkeypatch.setattr(remesh, "configuration", lambda *args: {})
    monkeypatch.setattr(remesh.diagnostic, "actual_pins", lambda problem: next(pin_records))
    monkeypatch.setattr(
        remesh.diagnostic,
        "extract",
        lambda problem, domain, np: (lower if domain == "liquid/lower" else bulk, {}),
    )
    with pytest.raises(ValueError, match=message):
        remesh.sample(problem, model, case, tmp_path, "after-remesh", np)
    receipt_path = tmp_path / "samples/after-remesh-preservation-failure.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "failed" and receipt["numerical_acceptance"] is False
    assert receipt["phase"] == "after-remesh"
    assert receipt["before"] == before and receipt["after"] == after
    assert receipt["actual_pins"] == {"before": pins_before, "after": pins_after}
    assert receipt["differing_signature_keys"] == (
        ["arrays.pinned_values_with_positions.sha256"] if changed == "signature" else []
    )
    assert receipt["differing_pin_keys"] == (["sha256"] if changed == "pins" else [])
    assert not (tmp_path / "samples/after-remesh.json").exists()
    assert not (tmp_path / "samples/after-remesh.npz").exists()
    assert receipt_path.stat().st_mode & 0o777 == 0o600


class DummyPressureNode:
    """Synthetic two-field, two-history storage; never native evidence."""

    def __init__(self, index):
        self.data = np.zeros((2, 2))
        if index < 3:
            self.data[:, 1] = [index + 1.0, 2 * (index + 1.0)]
        self.pins = [True, index >= 3]
        self.equations = [-1, index if index < 3 else -1]

    def nvalue(self):
        return 2

    def ntstorage(self):
        return 2

    def value_at_t(self, t, value):
        return self.data[t, value]

    def set_value_at_t(self, t, value, number):
        self.data[t, value] = number

    def is_pinned(self, value):
        return self.pins[value]

    def eqn_number(self, value):
        return self.equations[value]

    def is_hanging(self, value):
        return False


class DummyPressureMesh:
    def __init__(self, physical_fault=False):
        self.nodes = [DummyPressureNode(i) for i in range(6)]
        self.physical_fault = physical_fault

    def get_code_gen(self):
        return SimpleNamespace(get_space_of_field=lambda name: "C1" if name == "pressure" else "C2")

    def nnode(self):
        return 6

    def nelement(self):
        return 1

    def node_pt(self, index):
        return self.nodes[index]

    def element_pt(self, index):
        return SimpleNamespace(
            nnode=self.nnode,
            node_pt=self.node_pt,
            get_field_data_list=lambda name, full: [
                (node, 0 if name == "velocity_x" else 1)
                if name == "velocity_x" or i < 3
                else (None, -1)
                for i, node in enumerate(self.nodes)
            ],
        )

    def _interpolate_hanging_values(self):
        for a, b, middle in remesh.SIDES:
            self.nodes[middle].data[:, 1] = (
                self.nodes[a].data[:, 1] + self.nodes[b].data[:, 1]
            ) / 2
        if self.physical_fault:
            # A physical pinned velocity HISTORY changes; current fields and pins stay equal.
            self.nodes[0].data[1, 0] += 1


@pytest.mark.parametrize("physical_fault", [False, True])
def test_dummy_refresh_audits_every_history_and_refuses_physical_pinned_history(
    tmp_path, monkeypatch, physical_fault
):
    meshes = {"liquid": DummyPressureMesh(physical_fault), "gas": DummyPressureMesh()}
    problem = SimpleNamespace(get_mesh=lambda name: meshes[name], eigen_data_in_states=False)
    problem._snapshot_state = lambda: b"".join(
        node.data.tobytes() for mesh in meshes.values() for node in mesh.nodes
    )

    def signature(*args):
        digest = hashlib.sha256(problem._snapshot_state()).hexdigest()
        return {
            "arrays": {
                "free_dofs": {"sha256": "unchanged-current-fields"},
                "pinned_values_with_positions": {"sha256": digest},
            },
            "serialized_state": {"sha256": digest, "payload_sha256": digest},
        }

    model = SimpleNamespace(_stationary_state_signature=signature)
    monkeypatch.setattr(
        remesh.diagnostic,
        "actual_pins",
        lambda problem: {
            "sha256": hashlib.sha256(
                repr([node.pins for mesh in meshes.values() for node in mesh.nodes]).encode()
            ).hexdigest()
        },
    )
    if physical_fault:
        with pytest.raises(ValueError, match="changed physical state"):
            remesh.normalize_pressure_dummies(problem, model, tmp_path, np)
    else:
        remesh.normalize_pressure_dummies(problem, model, tmp_path, np)
    receipt = json.loads((tmp_path / "normalization/receipt.json").read_text())
    assert receipt["verified"] is not physical_fault
    assert receipt["checks"]["physical_serialized_state_preserved"] is not physical_fault
    assert receipt["checks"]["normalized_snapshot_restored"] is True
    assert receipt["checks"]["actual_pins_preserved"] is True
    assert receipt["checks"]["all_dummy_histories_equal_vertex_pressure_averages"] is True
    assert len(receipt["slots"]) == 6 and len(receipt["differences"]) == 12
    assert all(
        slot["history_count"] == 2 and slot["equation_number"] < 0 for slot in receipt["slots"]
    )
    with np.load(tmp_path / "normalization/dummy-histories.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["before"], np.zeros(12))
        assert np.array_equal(saved["after"], np.array([1.5, 3.0, 2.5, 5.0, 2.0, 4.0] * 2))
    assert problem.eigen_data_in_states is False
