"""Pure contracts for the Herrada--Yu--Stone figure 3 continuation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validation.herradaEtAl_figure3.driver import (
    branch_digest,
    case_digest,
    extract_row,
    meridional_length,
    prepare_output,
    read_branches,
    read_case,
    seed_geometry,
    trace_complete,
    trace_outcome,
    turning_candidates,
    validate_case,
    validate_resume,
    write_profile,
)

HERE = Path(__file__).parents[1] / "validation" / "herradaEtAl_figure3"
CONVERGENCE_CASES = {
    "mesh-medium.json": ("mesh-medium", 15.0, 10.0, 6.0, 0.08 * 4 / 6),
    "mesh-fine.json": ("mesh-fine", 15.0, 10.0, 8.0, 0.04),
    "domain20-fine.json": ("domain20-fine", 20.0, 12.5, 8.0, 0.04),
    "domain25-fine.json": ("domain25-fine", 25.0, 15.0, 8.0, 0.04),
}


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


@pytest.mark.parametrize(("name", "expected"), CONVERGENCE_CASES.items())
def test_declared_convergence_cases_are_exact_named_variants(name, expected):
    variant, domain, tip, film_elements, bulk_resolution = expected
    case = read_case(HERE / "cases" / name, "run")

    assert case["convergence"] == {
        "study": "figure3_numerical_convergence",
        "baseline_case_id": "herradaEtAl_figure3_bo1p56",
        "variant": variant,
    }
    assert case["physics"]["domain_over_r"] == domain
    assert case["physics"]["upper_tip_over_r"] == tip
    assert case["mesh"]["film_elements"] == film_elements
    assert case["mesh"]["bulk_resolution"] == pytest.approx(bulk_resolution)


def test_production_geometry_or_mesh_cannot_drift_without_a_declared_variant():
    baseline = json.loads((HERE / "case.json").read_text())
    for section, name, value in (
        ("physics", "domain_over_r", 20.0),
        ("physics", "upper_tip_over_r", 12.5),
        ("mesh", "film_elements", 8.0),
        ("mesh", "bulk_resolution", 0.04),
    ):
        changed = copy.deepcopy(baseline)
        changed[section][name] = value
        with pytest.raises(ValueError, match=name):
            validate_case(changed, "run")


def test_declared_variant_cannot_bless_an_unlisted_geometry():
    case = json.loads((HERE / "cases" / "domain20-fine.json").read_text())
    case["physics"]["upper_tip_over_r"] = 12.0

    with pytest.raises(ValueError, match="upper_tip_over_r"):
        validate_case(case, "run")


def test_mesh_levels_refine_film_and_bulk_by_the_same_factor():
    baseline = read_case(HERE / "case.json", "run")
    medium = read_case(HERE / "cases" / "mesh-medium.json", "run")
    fine = read_case(HERE / "cases" / "mesh-fine.json", "run")
    gap = seed_geometry(baseline).seed_gap

    baseline_sizes = (gap / baseline["mesh"]["film_elements"],
                      baseline["mesh"]["bulk_resolution"])
    medium_sizes = (gap / medium["mesh"]["film_elements"],
                    medium["mesh"]["bulk_resolution"])
    fine_sizes = (gap / fine["mesh"]["film_elements"],
                  fine["mesh"]["bulk_resolution"])
    assert medium_sizes[0] / baseline_sizes[0] == pytest.approx(2 / 3)
    assert medium_sizes[1] / baseline_sizes[1] == pytest.approx(2 / 3)
    assert fine_sizes[0] / baseline_sizes[0] == pytest.approx(1 / 2)
    assert fine_sizes[1] / baseline_sizes[1] == pytest.approx(1 / 2)


def test_domain_variants_add_equal_padding_at_both_ends():
    cases = [read_case(HERE / "case.json", "run"),
             read_case(HERE / "cases" / "domain20-fine.json", "run"),
             read_case(HERE / "cases" / "domain25-fine.json", "run")]
    geometries = [seed_geometry(case) for case in cases]
    base = geometries[0]

    for case, geometry, added_each_end in zip(cases, geometries, (0.0, 2.5, 5.0), strict=True):
        rt = geometry.r_tube
        assert geometry.l_domain / rt == pytest.approx(case["physics"]["domain_over_r"])
        assert geometry.z_front / rt == pytest.approx(case["physics"]["upper_tip_over_r"])
        assert geometry.pad_rear - base.pad_rear == pytest.approx(added_each_end)
        assert geometry.pad_front - base.pad_front == pytest.approx(added_each_end)


def test_profile_coordinate_follows_translated_pinned_tip(tmp_path):
    from types import SimpleNamespace

    case = read_case(HERE / "cases" / "domain20-fine.json", "run")
    problem = SimpleNamespace(geometry=SimpleNamespace(r_tube=2.0))
    write_profile(tmp_path, 0, problem, ([24.0, 25.0], [1.0, 0.0]), case)

    lines = (tmp_path / "profiles" / "station_00000.csv").read_text().splitlines()
    assert lines[0] == "z_solver,r_solver,z_paper,gap_over_r"
    assert lines[1].split(",")[2] == "-0.5"
    assert lines[2].split(",")[2] == "0.0"


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
        "schema": "herrada-figure3-restart-v2",
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
           "min_gap": 0.03, "axial_span_over_r": 6.8}
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


def test_meridional_length_matches_analytic_parabola_in_both_directions():
    import math

    # z=t, r=t² on [-1,1]: curved length differs from the axial span of 2.
    exact = math.sqrt(5) + math.asinh(2) / 2
    for z, r in [([-1, 0, 1], [1, 0, 1]), ([1, 0, -1], [1, 0, 1])]:
        assert meridional_length(z, r, quadrature_order=32) == pytest.approx(exact, abs=1e-12)
        assert meridional_length(z, r) > 2


def test_meridional_length_converges_to_semicircle_not_diameter():
    import math

    errors = []
    for elements in (8, 16, 32):
        theta = [math.pi * i / (2 * elements) for i in range(2 * elements + 1)]
        value = meridional_length([math.cos(t) for t in theta],
                                  [math.sin(t) for t in theta])
        errors.append(abs(value - math.pi))
    assert errors[2] < errors[1] / 10 < errors[0] / 100
    assert errors[2] < 4e-7


def test_legacy_axial_table_and_restart_are_rejected(tmp_path):
    table = tmp_path / "branches.csv"
    table.write_text("station,bubble_length_over_r\n0,7.0\n")
    with pytest.raises(ValueError, match="legacy axial-length"):
        read_branches(table)
    with pytest.raises(ValueError, match="meridional length"):
        validate_resume({"schema": "herrada-figure3-restart-v1"}, {})


def test_extract_row_uses_arc_length_for_paper_observable():
    from types import SimpleNamespace

    problem = SimpleNamespace(
        interface_polyline=lambda: ([1, 2, 3], [0, 1, 0]),
        geometry=SimpleNamespace(r_tube=2, volume=1, l_domain=5),
        observable=lambda *args: 1,
        current_bond=lambda: 1.56,
        current_ca_in=lambda: 0.02,
        frame_speed=lambda: 0.03,
    )
    row, _ = extract_row(problem, 0, {"max_volume_error": 1e-5,
                                    "max_flux_error": 1e-5,
                                    "containment_epsilon": 1e-8})
    assert row["axial_span_over_r"] == 1
    assert row["bubble_length_over_r"] == pytest.approx(1.4789428575446)


def test_remeasure_preserves_source_and_rejects_mismatched_profile(tmp_path):
    from validation.herradaEtAl_figure3.remeasure import remeasure

    branches = tmp_path / "old.csv"
    original = ("station,bond,ca,bubble_length_over_r,signed_ca_b,volume_error,flux_error,min_gap\n"
                "0,1.56,0.02,2,0.03,0,0,0.1\n")
    branches.write_text(original)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    profile = profiles / "station_00000.csv"
    profile.write_text("z_paper,gap_over_r\n-1,1\n0,0\n1,1\n")
    output = tmp_path / "corrected"
    remeasure(branches, profiles, output)
    assert branches.read_text() == original
    row = read_branches(output / "branches.csv")[0]
    assert float(row["bubble_length_over_r"]) == pytest.approx(2.957885715089)
    assert float(row["axial_span_over_r"]) == 2
    assert json.loads((output / "remeasure.json").read_text())["restart_modified"] is False
    profile.write_text("z_paper,gap_over_r\n-1,1\n0,0\n2,1\n")
    with pytest.raises(ValueError, match="axial-span mismatch"):
        remeasure(branches, profiles, tmp_path / "bad")
