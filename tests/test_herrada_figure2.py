"""Pure contract tests for the Herrada--Yu--Stone figure 2 reproduction."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from problems.geometry import geometry_from_case
from validation.herradaEtAl_figure2.driver import (
    RT,
    acceptance_errors,
    paper_coordinates,
    read_case,
    run_case,
    validate_case,
)

CASES = Path(__file__).parents[1] / "validation" / "herradaEtAl_figure2" / "cases"


@pytest.mark.parametrize("path", sorted(CASES.glob("*.json")))
def test_explicit_cases_preserve_paper_scaling_and_reflection(path):
    case = read_case(path)
    physics = case["physics"]

    assert physics["r_tube"] == pytest.approx(3.3 ** (-1 / 3))
    assert physics["l_domain"] / physics["r_tube"] == pytest.approx(15.0)
    assert physics["z_rear"] / physics["r_tube"] == pytest.approx(5.0)
    assert physics["ca_in"] == abs(physics["cal_paper"])
    geometry = geometry_from_case(physics)
    assert geometry.z_front < geometry.l_domain
    assert geometry.capsule_volume == pytest.approx(geometry.volume)


def test_paper_coordinates_apply_axial_reflection_and_radial_gap_scaling():
    zhat, hhat = paper_coordinates([3.0, 4.0, 5.0], [0.0, RT / 2, RT], RT)

    assert zhat == pytest.approx([0.0, -1 / RT, -2 / RT])
    assert hhat == pytest.approx([1.0, 0.5, 0.0])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("bond", 0.1, "bond"), ("reynolds", 1.0, "reynolds"),
     ("laplace", 1.0, "laplace"), ("ca_in", -0.001, "ca_in")],
)
def test_incorrect_physics_is_refused(field, value, message):
    case = json.loads((CASES / "cal_m0p001_baseline.json").read_text())
    case["physics"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_case(case)


def test_unpolished_or_failed_conservation_result_is_not_accepted():
    case = read_case(CASES / "cal_m0p02_baseline.json")
    result = {
        "route": "transient",
        "ca_b": 0.021,
        "h_flat_over_r_tube": 0.08,
        "check_void_volume_error": 2e-5,
        "check_flux_error": 0.0,
        "check_film_flatness": 0.001,
        "check_central_nodes": 20,
    }

    errors = acceptance_errors(result, case["acceptance"])

    assert any("stationary" in error for error in errors)
    assert any("volume" in error for error in errors)


def test_case_validator_rejects_wrong_window_even_if_schema_shape_is_valid():
    case = copy.deepcopy(read_case(CASES / "cal_m0p001_baseline.json"))
    case["physics"]["l_domain"] = 14 * RT

    with pytest.raises(ValueError, match="H/R=15"):
        validate_case(case)


def test_precreated_runtime_is_accepted_once_and_failed_attempt_is_preserved(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    invalid_case = tmp_path / "invalid.json"
    invalid_case.write_text("{}")
    assert run_case(invalid_case, runtime) == 1
    assert json.loads((runtime / "status.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="not empty"):
        run_case(invalid_case, runtime)
