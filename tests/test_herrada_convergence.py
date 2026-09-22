"""Contracts for order-preserving Herrada figure 3 convergence comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from validation.herradaEtAl_figure3.convergence import ConvergenceTolerances, compare_runs

TOLERANCES = ConvergenceTolerances(
    max_length_error=0.03,
    max_ca_error_branch_ii=0.002,
    max_fold_ca_error=0.002,
    max_fold_length_error=0.03,
    max_fold_ca_sampling_width=0.02,
    max_fold_length_sampling_width=1.0,
    max_profile_error=0.03,
    max_profile_ca_mismatch=0.006,
)


def _write_run(path: Path, *, shift: float = 0, extra_wiggles: bool = True,
               coarse_folds: bool = False) -> None:
    path.mkdir()
    json.dump({"status": "complete"}, (path / "status.json").open("w"))
    # Four physical pieces: descending III, II with harmless inner wiggles,
    # and descending I.  The first and last reversals delimit branch II.
    ca = [0.020, 0.010, 0.000]
    length = [8.00, 7.50, 7.00]
    if extra_wiggles:
        ca += [0.006, 0.004, 0.007]
        length += [6.80, 6.70, 6.60]
    ca += [0.005, -0.005, -0.020]
    length += [6.50, 6.45, 6.40]
    if coarse_folds:
        length[1] = 7.8
    fields = ["station", "bond", "ca", "bubble_length_over_r", "signed_ca_b",
              "volume_error", "flux_error", "min_gap", "axial_span_over_r"]
    with (path / "branches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for station, (ca_value, length_value) in enumerate(zip(ca, length, strict=True)):
            writer.writerow({
                "station": station, "bond": 1.56, "ca": ca_value,
                "bubble_length_over_r": length_value + shift,
                "signed_ca_b": ca_value, "volume_error": 0, "flux_error": 0,
                "min_gap": 0.1, "axial_span_over_r": length_value - 1,
            })
    profiles = path / "profiles"
    profiles.mkdir()
    for station, length_value in enumerate(length):
        with (profiles / f"station_{station:05d}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["z_solver", "r_solver", "z_paper", "gap_over_r"])
            writer.writerows([[0, 0, -length_value + shift, 1],
                              [0, 0, -length_value / 2 + shift, 0.1],
                              [0, 0, shift, 1]])


def test_comparison_preserves_order_and_ignores_internal_turns(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _write_run(reference)
    _write_run(candidate, shift=0.001)

    report = compare_runs(reference, candidate, TOLERANCES)

    assert report["status"] == "inconclusive"
    assert report["branch_partition"]["reference_turns"] == [2, 3, 4, 5]
    assert report["branch_partition"]["reference_internal_turns"] == [3, 4]
    assert report["curves"]["I"]["method"] == "branch-local interpolation at matched Ca"
    assert report["curves"]["II"]["method"].startswith("matched length")
    assert report["curves"]["III"]["max_abs_length_error"] == pytest.approx(0.001)


def test_fold_gate_is_inconclusive_when_station_bracket_is_too_wide(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _write_run(reference)
    _write_run(candidate, shift=0.001, coarse_folds=True)
    strict_resolution = ConvergenceTolerances(
        **{**TOLERANCES.__dict__, "max_fold_length_sampling_width": 0.05,
           "max_length_error": 1.0, "max_profile_error": 1.0}
    )

    report = compare_runs(reference, candidate, strict_resolution)

    assert report["folds"]["III_II"]["status"] == "inconclusive"
    assert report["folds"]["III_II"]["sampling_widths_resolved"] is False
    assert "not a bound" in report["folds"]["III_II"]["reference"]["sampling_note"]
    assert report["status"] == "inconclusive"


def test_shape_comparison_reports_nearest_state_ca_mismatch(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _write_run(reference)
    _write_run(candidate)

    report = compare_runs(reference, candidate, TOLERANCES)

    endpoint = report["profiles"][0]
    assert endpoint["method"].startswith("nearest accepted states")
    assert endpoint["ca_mismatch"] == 0
    assert endpoint["max_coordinate_error"] == pytest.approx(0)


def test_shape_ca_sampling_mismatch_is_inconclusive_not_failure(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _write_run(reference)
    _write_run(candidate)
    table = candidate / "branches.csv"
    rows = list(csv.DictReader(table.open(newline="")))
    rows[-1]["ca"] = "-0.018"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    strict_ca_match = ConvergenceTolerances(
        **{**TOLERANCES.__dict__, "max_profile_ca_mismatch": 0.001}
    )
    report = compare_runs(reference, candidate, strict_ca_match)

    endpoint = next(item for item in report["profiles"]
                    if item["landmark"] == "branch_I_endpoint")
    assert endpoint["ca_mismatch"] == pytest.approx(0.002)
    assert endpoint["status"] == "inconclusive"


def test_report_hashes_status_and_branch_inputs(tmp_path):
    import hashlib

    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _write_run(reference)
    _write_run(candidate)

    report = compare_runs(reference, candidate, TOLERANCES)

    for label, directory in (("reference", reference), ("candidate", candidate)):
        assert report["inputs"][label]["status_sha256"] == hashlib.sha256(
            (directory / "status.json").read_bytes()).hexdigest()
        assert report["inputs"][label]["branches_sha256"] == hashlib.sha256(
            (directory / "branches.csv").read_bytes()).hexdigest()
    assert "not certified" in report["claim_limit"]


def test_incomplete_runs_and_legacy_tables_are_refused(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _write_run(reference)
    _write_run(candidate)
    (candidate / "status.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="complete run"):
        compare_runs(reference, candidate, TOLERANCES)

    (candidate / "status.json").write_text('{"status":"complete"}')
    (candidate / "branches.csv").write_text("station,ca,bubble_length_over_r\n0,0,1\n")
    with pytest.raises(ValueError, match="corrected figure 3 fields"):
        compare_runs(reference, candidate, TOLERANCES)


def test_unequal_three_grid_order_uses_signed_same_observable():
    import math

    from validation.herradaEtAl_figure3.convergence import three_grid_order

    h = [0.08, 0.08 * 2 / 3, 0.04]
    q = [1 + 3 * x**2 for x in h]
    result = three_grid_order(h, q)
    assert result["order"] == pytest.approx(2)
    misleading_slope = math.log((q[0] - q[2]) / (q[1] - q[2])) / math.log(h[0] / h[1])
    assert misleading_slope == pytest.approx(3.33, abs=0.01)
    assert three_grid_order(h, [1, 2, 1.5])["order"] is None
    assert three_grid_order(h, [1, 1, 1])["order"] is None
    assert three_grid_order(h, [3, 2, 0])["order"] is None


def test_matched_length_ignores_endpoint_chord_phase():
    from validation.herradaEtAl_figure3.convergence import matched_length_diagnostic

    # Same exact straight curve, with different ranges and station spacing.
    reference = [{"length": x, "ca": 0.01 * x} for x in [1, 2, 3, 4]]
    candidate = [{"length": x, "ca": 0.01 * x} for x in [1.5, 2.2, 3.5]]
    result = matched_length_diagnostic(reference, candidate)
    assert result["max_abs_ca_error"] == pytest.approx(0, abs=1e-15)
    assert result["pchip_max_abs_ca_error"] == pytest.approx(0, abs=1e-15)
    assert result["length_overlap"] == [1.5, 3.5]
    assert result["status"] == "inconclusive"


def test_length_reversal_is_explicit_or_refused():
    from validation.herradaEtAl_figure3.convergence import matched_length_diagnostic

    rows = [{"length": x, "ca": 0.01 * x} for x in [3.9, 4, 3, 2, 1]]
    result = matched_length_diagnostic(rows, rows)
    assert result["reference_segment"]["excluded_endpoint_rows"] == [1, 0]
    assert result["length_overlap"] == [1, 4]
    rows = [{"length": x, "ca": 0.01 * x} for x in [4, 2, 3, 1]]
    with pytest.raises(ValueError, match="not single-valued"):
        matched_length_diagnostic(rows, rows)


def test_interpolation_sensitivity_is_reported_without_claiming_spatial_error():
    from validation.herradaEtAl_figure3.convergence import matched_length_diagnostic

    ref = [{"length": x, "ca": x**2} for x in [1, 1.5, 2, 2.5, 3]]
    cand = [{"length": x, "ca": x**2} for x in [1, 2, 3]]
    result = matched_length_diagnostic(ref, cand)
    assert result["interpolation_method_sensitivity"] > 0.1
    assert result["status"] == "inconclusive"
