"""Focused geometry, model-limit and software-custody controls; no FEM import."""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest

from modelComparisonCases.herradaEtAl_nose import run_lubrication as nose


def test_zero_gravity_flux_closure_and_exact_flat_film():
    alpha, imposed = 0.2, 0.02
    speed = nose.bubble_speed(alpha, 0.0, imposed)
    # Far-field relative flux equals the wall-relative uniform annular-film flux.
    assert imposed - speed == pytest.approx(-speed * (1 - (1 - alpha) ** 2), abs=1e-16)
    flat = np.array([alpha, 3.0, 1 / (1 - alpha), 0.0])
    assert np.array_equal(nose.geometry_rhs(flat, alpha, speed, 0.0), [0.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="nonzero asymptotic"):
        nose.start_state(alpha, speed, 0.0, 0.0)


def test_growing_mode_matches_independent_directional_linearization():
    alpha, cab, bo = 0.15, 0.03, 0.4
    k = nose.growing_exponent(alpha, cab, bo)
    c = 1 / (1 - alpha) ** 2
    mode = np.array([1.0, 0.0, k * k + c, k])
    base = np.array([alpha, 0.0, 1 / (1 - alpha), 0.0])
    delta = 1e-7
    derivative = (
        nose.geometry_rhs(base + delta * mode, alpha, cab, bo)
        - nose.geometry_rhs(base - delta * mode, alpha, cab, bo)
    ) / (2 * delta)
    assert np.allclose(derivative, k * mode, rtol=1e-8, atol=2e-8)


def test_regular_spherical_pole_has_two_equal_principal_curvatures():
    # Independently parameterised sphere: r=a*cos(s/a), z=a*sin(s/a).
    radius = 0.8
    for angle in [0.2, 0.8, 1.4, math.pi / 2 - 0.001]:
        h, z = 1 - radius * math.cos(angle), radius * math.sin(angle)
        curvature = 2 / radius
        derivative = nose.geometry_rhs([h, z, curvature, angle], 0.1, 0.0, 0.0)
        assert np.allclose(
            derivative, [math.sin(angle), math.cos(angle), 0.0, 1 / radius], atol=1e-12
        )
        assert curvature == pytest.approx(2 * derivative[3], abs=1e-12)
        assert nose.pole_residual([h, z, curvature, angle], 0.1, 0.0, 0.0) == pytest.approx(
            0.0, abs=1e-15
        )


@pytest.mark.parametrize(
    "bo,ca",
    [(-1, 0.02), (1, 0), (1, -0.02), (float("nan"), 0.02), (1, float("inf")), (False, 0.02)],
)
def test_invalid_input_is_refused(bo, ca):
    with pytest.raises(ValueError, match="requires finite"):
        nose.validate_input(bo, ca)


def test_trial_budget_records_a_failure_without_qualifying_the_model():
    records = []
    controls = nose.replace(nose.Controls(), max_steps=1)
    with pytest.raises(ValueError, match="no physical event"):
        nose.shoot(0.15, 1.56, 0.02, controls, nose.Budget(), records)
    assert len(records) == 1 and records[0]["status"] == "failed"
    assert records[0]["rhs_calls"] > 0


def software_source():
    return {"component_commit": "software-only-source-standin", "paper": nose.PAPER}


def test_small_independent_case_has_exclusive_hash_bound_output(tmp_path, monkeypatch):
    output = tmp_path / "model-output"
    monkeypatch.setattr(nose, "source_provenance", software_source)
    before = set(sys.modules)
    assert nose.main(["--bo", "1.56", "--ca", ".02", "--out", str(output)]) == 0
    result = json.loads((output / "result.json").read_text())
    complete = json.loads((output / "completion.json").read_text())
    assert result["status"] == "complete" and result["numerical_acceptance"] is True
    assert 0 < result["alpha"] < 1 and result["ca_b"] > 0.02
    assert result["volume_input"] is None
    assert set(result["solutions"]) == {
        "baseline",
        "start-amplitude",
        "pole-and-integrator-refinement",
    }
    assert all(all(s["checks"].values()) for s in result["solutions"].values())
    assert result["budget"]["trials"] <= result["budget"]["max_trials"]
    assert complete["claim_kind"] == "model_comparison"
    for relative, digest in complete["files"].items():
        assert nose.hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest
    sentinel = (output / "result.json").read_bytes()
    with pytest.raises(ValueError, match="occupied"):
        nose.main(["--bo", "1.56", "--ca", ".02", "--out", str(output)])
    assert (output / "result.json").read_bytes() == sentinel
    assert not any(
        name == "pyoomph" or name.startswith("pyoomph.") for name in set(sys.modules) - before
    )
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "result.json").stat().st_mode & 0o777 == 0o600


def test_failure_retains_trial_controls_and_has_no_success_completion(tmp_path, monkeypatch):
    output = tmp_path / "failed-model"
    monkeypatch.setattr(nose, "source_provenance", software_source)

    def fail(*args, **kwargs):
        kwargs["log"].append({"alpha": 0.01, "status": "failed", "controls": {"max_steps": 1}})
        raise RuntimeError("software-only bounded shooting failure")

    monkeypatch.setattr(nose, "calculate", fail)
    with pytest.raises(RuntimeError, match="bounded shooting failure"):
        nose.main(["--bo", "1.56", "--ca", ".02", "--out", str(output)])
    assert json.loads((output / "failure.json").read_text())["numerical_acceptance"] is False
    assert json.loads((output / "trials.json").read_text())[0]["controls"] == {"max_steps": 1}
    assert not (output / "completion.json").exists()
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["prepared", "failure"]
