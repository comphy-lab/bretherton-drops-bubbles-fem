"""Pure contracts for the Herrada--Yu--Stone figure 3 continuation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validation.herradaEtAl_figure3.driver import (
    branch_digest,
    case_digest,
    prepare_output,
    read_case,
    trace_complete,
    trace_outcome,
    turning_candidates,
    validate_case,
    validate_resume,
)

HERE = Path(__file__).parents[1] / "validation" / "herradaEtAl_figure3"


@pytest.mark.parametrize(("name", "mode"), [("case.json", "run"), ("smoke.json", "smoke")])
def test_cases_encode_signed_figure3_scaling(name, mode):
    case = read_case(HERE / name, mode)
    physics = case["physics"]

    assert case["schema"] == "pyoomph-case-v1"
    assert case["benchmark_schema"] == "herrada-figure3-v1"
    assert physics["r_tube"] == pytest.approx(3.3 ** (-1 / 3))
    assert physics["domain_over_r"] == 15
    assert physics["upper_tip_over_r"] == 10
    assert physics["bond"] == 1.56
    assert physics["allow_signed_flow"] is True
    assert case["closure"]["frame_speed"] == "front_tip"
    if mode == "smoke":
        assert case["smoke"]["ca"] < 0
    else:
        assert case["continuation"]["initial_ds"] < 0
        assert case["continuation"]["stop_ca"] < 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["physics"].update(bond=0), "bond"),
        (lambda c: c["physics"].update(r_tube=0.7), "r_tube"),
        (lambda c: c["physics"].update(allow_signed_flow=False), "signed-flow"),
        (lambda c: c["closure"].update(frame_speed="centroid"), "front-tip"),
        (lambda c: c["continuation"].update(initial_ds=0.001), "initial_ds"),
        (lambda c: c["continuation"].update(max_ds=float("nan")), "max_ds"),
        (lambda c: c["continuation"].update(stop_ca=-0.001), "stop_ca"),
        (lambda c: c["acceptance"].update(max_volume_error=float("inf")), "max_volume_error"),
    ],
)
def test_malformed_production_cases_are_refused(mutation, message):
    case = json.loads((HERE / "case.json").read_text())
    mutation(case)
    with pytest.raises(ValueError, match=message):
        validate_case(case, "run")


def test_turns_are_chronological_and_extra_wiggles_are_retained():
    values = [0.02, 0.01, 0.015, 0.005, 0.008, -0.021]

    assert turning_candidates(values) == [
        {"station": 1, "ca": 0.01},
        {"station": 2, "ca": 0.015},
        {"station": 3, "ca": 0.005},
        {"station": 4, "ca": 0.008},
    ]
    rows = [{"station": index, "ca": value} for index, value in enumerate(values)]
    assert trace_complete(rows, -0.02, 2)


def test_reaching_turn_count_without_negative_endpoint_is_not_complete():
    values = [0.02, 0.01, 0.015, 0.005]
    rows = [{"station": index, "ca": value} for index, value in enumerate(values)]

    assert not trace_complete(rows, -0.02, 2)


def test_exhausted_budget_is_incomplete_not_complete():
    rows = [{"station": 0, "ca": 0.02}, {"station": 1, "ca": 0.01}]

    assert trace_outcome(rows, stop_ca=-0.02, min_turns=2,
                         steps_taken=1, max_steps=1) == "incomplete"


def test_resume_refuses_a_different_case_hash():
    case = read_case(HERE / "case.json", "run")
    restart = {
        "schema": "herrada-figure3-restart-v1",
        "case_sha256": case_digest(case),
        "phase": "trace",
        "station": 0,
        "next_ds": -0.005,
        "state_path": "checkpoints/station_00000.dump",
        "branches_path": "branches.csv",
        "run_directory": "/tmp/example",
        "state_sha256": "0" * 64,
        "branches_prefix_sha256": "1" * 64,
    }
    validate_resume(restart, case)
    changed = copy.deepcopy(case)
    changed["continuation"]["max_steps"] += 1

    with pytest.raises(ValueError, match="case hash"):
        validate_resume(restart, changed)


def test_branch_digest_survives_csv_conversion_but_detects_changed_data():
    row = {"station": 0, "bond": 1.56, "ca": 0.02, "bubble_length_over_r": 7.8,
           "signed_ca_b": 0.1, "volume_error": 1e-12, "flux_error": 0,
           "min_gap": 0.03}
    strings = {key: str(value) for key, value in row.items()}
    assert branch_digest([row]) == branch_digest([strings])
    strings["ca"] = "0.01"
    assert branch_digest([row]) != branch_digest([strings])


def test_attempt_output_is_claimed_once(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    prepare_output(runtime)
    with pytest.raises(ValueError, match="not empty"):
        prepare_output(runtime)
