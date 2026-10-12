"""Software-only fresh-seed case, schedule, preparation and refusal controls."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from runners import run_paper_seed_i as seed


@pytest.fixture
def cases():
    baseline = seed.model.read_case(seed.ROOT / seed.BASE_CASE)
    case = copy.deepcopy(baseline)
    case["case_id"] = "software_fresh_negative_seed"
    case["physics"]["ca_in"] = -0.02
    return baseline, case


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("physics", "ca_in", -0.01),
        ("physics", "bond", 1.57),
        ("mesh", "film_elements", 8.0),
        ("solve", "relax_time", 0.0),
        ("tolerances", "pressure_boundary_absolute", 1e-5),
    ],
)
def test_unapproved_case_or_control_changes_are_refused(cases, section, key, value):
    baseline, case = cases
    case[section][key] = value
    with pytest.raises(ValueError):
        seed.validate_case(case, baseline)


def test_real_bo_schedule_has_truthful_individual_cases(cases):
    baseline, case = cases
    assert seed.validate_case(case, baseline) == case
    schedule = seed.model.stationary_bond_schedule(case)
    assert len(schedule) == 17 and schedule[-1] == 1.56
    assert seed.validate_schedule(case, schedule) == schedule
    for index, bond in enumerate(schedule):
        stage = seed.stage_case(case, index, schedule)
        assert stage["physics"]["bond"] == bond
        assert stage["physics"]["ca_in"] == -0.02
        assert stage["solve"] == case["solve"]
        assert seed.model.stationary_bond_schedule(stage) == schedule[: index + 1]
    assert seed.stage_case(case, len(schedule) - 1, schedule) == case


def test_wrong_bo_order_is_refused(cases):
    _, case = cases
    schedule = seed.model.stationary_bond_schedule(case)
    schedule[1], schedule[2] = schedule[2], schedule[1]
    with pytest.raises(ValueError, match="Bo order"):
        seed.validate_schedule(case, schedule)


def test_wrong_core_is_refused_before_native_execution(monkeypatch):
    monkeypatch.setattr(seed.runner, "require_pinned_environment", lambda value: None)
    with pytest.raises(ValueError, match="source/core"):
        seed.validate_environment(
            {"commit": seed.diagnostic.PYOOMPH, "native_core": {"sha256": "software-wrong-core"}}
        )


@pytest.mark.parametrize("fault", [None, "forcing", "snapshot"])
def test_declared_preparation_is_bounded_and_preserves_failed_inputs(
    cases, tmp_path, monkeypatch, fault
):
    _, target = cases
    case = seed.stage_case(target, 0, seed.model.stationary_bond_schedule(target))
    parameters = {"Ca_in": SimpleNamespace(value=-0.02), "Bo": SimpleNamespace(value=0.0)}
    if fault == "forcing":
        parameters["Ca_in"].value = -0.01
    calls, retained = [], []
    problem = SimpleNamespace(
        case=case,
        get_global_parameter=lambda name: parameters[name],
        max_newton_iterations=10,
        run=lambda endtime, **kwargs: calls.append((endtime, kwargs)),
    )

    def retain(*args):
        if fault == "snapshot":
            raise ValueError("software full-snapshot refusal")
        phase = args[-1]
        retained.append(phase)
        return {"signature": {"time": 0.0 if phase == "before-relaxation" else 8.0}}

    monkeypatch.setattr(seed.remesh, "retain_portable_sample", retain)
    if fault:
        with pytest.raises(ValueError):
            seed.preparatory_relaxation(problem, case, tmp_path)
        assert not calls and not (tmp_path / "relaxation.json").exists()
    else:
        receipt = seed.preparatory_relaxation(problem, case, tmp_path)
        assert calls == [
            (8.0, {"startstep": 0.0001, "outstep": 1.0, "maxstep": 1.0, "temporal_error": 1e-4})
        ]
        assert retained == ["before-relaxation", "after-relaxation"]
        assert receipt["numerical_acceptance"] is False
        assert (
            json.loads((tmp_path / "relaxation-prepared.json").read_text())["transient_newton_cap"]
            == 10
        )


def test_failed_stationary_checkpoint_never_creates_a_completed_stage(cases, tmp_path, monkeypatch):
    _, target = cases
    case = seed.stage_case(target, 0, seed.model.stationary_bond_schedule(target))
    parameters = {"Bo": SimpleNamespace(value=0.0), "Ca_in": SimpleNamespace(value=-0.02)}
    problem = SimpleNamespace(case=case, get_global_parameter=lambda name: parameters[name])
    monkeypatch.setattr(
        seed.model, "polish_confined_state", lambda *args, **kwargs: {"status": "converged"}
    )
    monkeypatch.setattr(seed.remesh, "sample", lambda *args: {"metrics": {}})

    def failed_snapshot(*args):
        raise ValueError("software failed complete-state roundtrip")

    monkeypatch.setattr(seed.model, "_checkpoint_stationary_state", failed_snapshot)
    with pytest.raises(ValueError, match="complete-state roundtrip"):
        seed.stationary_stage(problem, case, tmp_path, 0, {}, np)
    assert not (tmp_path / "stage.json").exists()


def test_original_pressure_gate_remains_a_failure(cases):
    _, case = cases
    checks = seed.remesh.canonical_checks(
        case,
        {
            "stationary_residual_linf": 1e-9,
            "lower_pressure_rms": 1.1e-6,
            "lower_flux": -0.02,
            "upper_flux": -0.02,
            "gas_volume": 1.0,
            "target_volume": 1.0,
            "surface_volume": 1.0,
        },
    )
    assert checks["lower_static_pressure"] is False
    assert all(passed for name, passed in checks.items() if name != "lower_static_pressure")
