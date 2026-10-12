"""Software-only Ca transition and cold-load refusal controls."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from runners import run_paper_anchor as anchor


@pytest.fixture
def cases():
    parent = anchor.model.read_case(
        anchor.ROOT / "validation/herradaEtAl_paper_model/cases/h15-bo1p56.json"
    )
    target = copy.deepcopy(parent)
    target["case_id"] = "software_stationary_III"
    target["physics"]["ca_in"] = anchor.TARGET_CA
    return parent, target


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("physics", "bond", 1.57),
        ("mesh", "film_elements", 7.0),
        ("tolerances", "pressure_boundary_absolute", 1e-5),
        ("solve", "max_newton_iterations", 41),
        ("mesh", "bulk_resolution", 0.04),
    ],
)
def test_transition_refuses_other_valid_case_changes(cases, section, key, value):
    parent, target = cases
    original = copy.deepcopy(parent)
    target[section][key] = value
    with pytest.raises(ValueError, match="only Ca and case_id"):
        anchor.transition_contract(parent, target)
    assert parent == original


def test_each_stage_has_its_own_truthful_case_and_exact_target(cases):
    parent, target = cases
    contract = anchor.transition_contract(parent, target)
    assert set(contract["changed_fields"]) == {"case_id", "physics.ca_in"}
    intermediate = anchor.stage_case(target, 0.01, 1, False)
    assert intermediate["physics"]["ca_in"] == 0.01
    assert intermediate["case_id"] != target["case_id"]
    assert anchor.stage_case(target, anchor.TARGET_CA, 5, True) == target
    assert parent["physics"]["ca_in"] == 0.02


@pytest.mark.parametrize(
    "values",
    [
        [0.01, 0.01, -0.00315],
        [0.01, 0.015, -0.00315],
        [0.03, -0.00315],
        [0.01, -0.003],
        [0.01, True, -0.00315],
        [float("nan"), -0.00315],
        [0.019, 0.018, 0.017, 0.016, 0.015, 0.014, 0.013, 0.012, -0.00315],
    ],
)
def test_wrong_order_endpoint_numeric_type_or_unbounded_ladder_is_refused(cases, tmp_path, values):
    parent, target = cases
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"schema": "herrada-paper-ca-ladder-v1", "values": values}))
    with pytest.raises(ValueError):
        anchor.read_ladder(path, parent, target)


def test_authorized_ladder_is_explicit_and_read_unchanged(cases, tmp_path):
    parent, target = cases
    values = [0.01, 0.0, -0.002, -0.003, -0.00315]
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"schema": "herrada-paper-ca-ladder-v1", "values": values}))
    before = path.read_bytes()
    assert anchor.read_ladder(path, parent, target) == values
    assert path.read_bytes() == before


def signature(ca=0.02, bo=1.56):
    payload = hashlib.sha256(repr((ca, bo)).encode()).hexdigest()
    return {
        "arrays": {"free_dofs": {"sha256": "software-fields"}},
        "layout": {"ndof": 6},
        "parameters": {"Ca_in": ca, "Bo": bo},
        "bc_cache_marker": "init",
        "volume": 1.0,
        "frame_speed": 0.02,
        "time": 8.0,
        "serialized_state": {"sha256": payload, "payload_sha256": payload},
    }


@pytest.mark.parametrize("fault", [None, "prior_forcing", "unexpected_bond"])
def test_adoption_records_actual_flow_and_refuses_undeclared_forcing(
    cases, tmp_path, monkeypatch, fault
):
    parent, target = cases
    parameters = {"Ca_in": SimpleNamespace(value=0.02), "Bo": SimpleNamespace(value=1.56)}
    problem = SimpleNamespace(
        case=copy.deepcopy(parent), get_global_parameter=lambda name: parameters[name]
    )
    pins = {"sha256": "software-pins"}
    calls = 0

    def current(*args):
        nonlocal calls
        calls += 1
        if fault == "unexpected_bond" and calls == 2:
            parameters["Bo"].value = 1.57
        return signature(parameters["Ca_in"].value, parameters["Bo"].value)

    monkeypatch.setattr(anchor.model, "_stationary_state_signature", current)
    monkeypatch.setattr(anchor.diagnostic, "actual_pins", lambda problem: pins)
    if fault == "prior_forcing":
        parameters["Ca_in"].value = 0.01
    if fault:
        with pytest.raises(ValueError, match="forcing"):
            anchor.adopt_case(problem, parent, target, tmp_path)
    else:
        receipt = anchor.adopt_case(problem, parent, target, tmp_path)
        assert receipt["verified"] is True
        assert parameters["Ca_in"].value == -0.00315
        assert problem.case == target
        assert parent["physics"]["ca_in"] == 0.02
        assert (
            json.loads((tmp_path / "transition-prepared.json").read_text())["before"]["parameters"][
                "Ca_in"
            ]
            == 0.02
        )
        assert (
            json.loads((tmp_path / "transition.json").read_text())["after"]["parameters"]["Ca_in"]
            == -0.00315
        )


@pytest.mark.parametrize("fault", [None, "fields", "pins", "history", "forcing"])
def test_parent_or_target_cold_load_refuses_changed_native_identity(cases, monkeypatch, fault):
    parent, _ = cases
    expected = signature()
    loaded = copy.deepcopy(expected)
    original_pins, pins = {"sha256": "software-pins"}, {"sha256": "software-pins"}
    if fault == "fields":
        loaded["arrays"]["free_dofs"]["sha256"] = "changed"
    elif fault == "pins":
        pins["sha256"] = "changed"
    elif fault == "history":
        loaded["serialized_state"]["payload_sha256"] = "changed"
    elif fault == "forcing":
        loaded["parameters"]["Ca_in"] = 0.01
    restart = {
        "state": {"path": "/software-only/state.dump"},
        "mesh_exchange": {},
        "stage": {"bc_mode": "init"},
        "roundtrip": {"after": expected, "layout": expected["layout"]},
    }
    operations = []
    problem = SimpleNamespace(load_state=lambda *args, **kwargs: operations.append("load"))
    monkeypatch.setattr(
        anchor.runner, "verify_restart_artifacts", lambda restart: operations.append("verify bytes")
    )
    monkeypatch.setattr(anchor.model, "_stationary_state_signature", lambda *args: loaded)
    monkeypatch.setattr(anchor.diagnostic, "actual_pins", lambda problem: pins)
    monkeypatch.setattr(
        anchor.model, "_prepare_stationary_bc_state", lambda *args: {"verified": True}
    )
    if fault:
        with pytest.raises(ValueError):
            anchor.restore_checked(
                problem,
                parent,
                restart,
                original_pins,
                expected["serialized_state"]["payload_sha256"],
            )
    else:
        assert (
            anchor.restore_checked(
                problem,
                parent,
                restart,
                original_pins,
                expected["serialized_state"]["payload_sha256"],
            )["verified"]
            is True
        )
    assert operations == ["verify bytes", "load"]
