"""Compare complete figure 3 continuations without destroying branch order.

This module only post-processes accepted stationary states.  It deliberately
does not infer stability from the continuation order or treat every small
parameter reversal on branch II as a physical fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path


@dataclass(frozen=True)
class ConvergenceTolerances:
    """Predeclared numerical-convergence gates, in paper coordinates."""

    max_length_error: float
    max_ca_error_branch_ii: float
    max_fold_ca_error: float
    max_fold_length_error: float
    max_fold_ca_sampling_width: float
    max_fold_length_sampling_width: float
    max_profile_error: float
    max_profile_ca_mismatch: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def _read_complete_run(run: Path, profiles: Path | None = None) -> dict:
    status_path = run / "status.json"
    table_path = run / "branches.csv"
    if not status_path.is_file() or not table_path.is_file():
        raise ValueError(f"run is missing status.json or branches.csv: {run}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"convergence requires a complete run: {run}")
    with table_path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    required = {"station", "ca", "bubble_length_over_r", "axial_span_over_r"}
    if not source or not required.issubset(source[0]):
        raise ValueError(f"branch table lacks corrected figure 3 fields: {table_path}")
    rows = []
    for source_row in source:
        row = {
            "station": int(source_row["station"]),
            "ca": float(source_row["ca"]),
            "length": float(source_row["bubble_length_over_r"]),
            "axial_span": float(source_row["axial_span_over_r"]),
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"non-finite branch value at station {row['station']}")
        rows.append(row)
    if [row["station"] for row in rows] != list(range(len(rows))):
        raise ValueError("branches.csv must remain in chronological station order")
    turns = _turning_indices([row["ca"] for row in rows])
    if len(turns) < 2:
        raise ValueError("a complete figure 3 trace must contain at least two reversals")
    first, last = turns[0], turns[-1]
    return {
        "path": str(run.resolve()),
        "status_sha256": _file_digest(status_path),
        "branches_sha256": _file_digest(table_path),
        "rows": rows,
        "profiles": (profiles or run / "profiles").resolve(),
        "turns": turns,
        "physical_folds": [first, last],
        "internal_turns": turns[1:-1],
        "branches": {
            "III": rows[: first + 1],
            "II": rows[first : last + 1],
            "I": rows[last:],
        },
    }


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _turning_indices(values: list[float], atol: float = 1e-12) -> list[int]:
    turns = []
    for index in range(1, len(values) - 1):
        before = values[index] - values[index - 1]
        after = values[index + 1] - values[index]
        if abs(before) > atol and abs(after) > atol and before * after < 0:
            turns.append(index)
    return turns


def _collapse_monotone(rows: list[dict], atol: float = 1e-14) -> list[tuple[float, float]]:
    """Sort one physical branch by Ca and combine effectively duplicate Ca."""
    points = sorted((row["ca"], row["length"]) for row in rows)
    collapsed: list[tuple[float, float]] = []
    for ca, length in points:
        if collapsed and abs(ca - collapsed[-1][0]) <= atol:
            old_ca, old_length = collapsed[-1]
            collapsed[-1] = ((old_ca + ca) / 2, (old_length + length) / 2)
        else:
            collapsed.append((ca, length))
    if len(collapsed) < 2:
        raise ValueError("physical branch needs at least two distinct Ca values")
    return collapsed


def _linear(points: list[tuple[float, float]], x: float) -> float:
    if not points[0][0] <= x <= points[-1][0]:
        raise ValueError("interpolation requested outside branch overlap")
    for left, right in pairwise(points):
        if x <= right[0]:
            fraction = (x - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
    return points[-1][1]


def _rms(values: Iterable[float]) -> float:
    values = list(values)
    return math.sqrt(sum(value * value for value in values) / len(values))


def _outer_branch_error(reference: list[dict], candidate: list[dict]) -> dict:
    ref = _collapse_monotone(reference)
    cand = _collapse_monotone(candidate)
    low, high = max(ref[0][0], cand[0][0]), min(ref[-1][0], cand[-1][0])
    if low >= high:
        raise ValueError("outer branches have no finite Ca overlap")
    abscissae = sorted({ca for ca, _ in ref + cand if low <= ca <= high} | {low, high})
    errors = [abs(_linear(ref, ca) - _linear(cand, ca)) for ca in abscissae]
    return {
        "method": "branch-local interpolation at matched Ca",
        "ca_overlap": [low, high],
        "samples": len(abscissae),
        "max_abs_length_error": max(errors),
        "rms_length_error": _rms(errors),
    }


def _ordered_coordinates(rows: list[dict]) -> list[tuple[float, float, float]]:
    distances = [0.0]
    for left, right in pairwise(rows):
        distances.append(distances[-1] + math.hypot(
            right["ca"] - left["ca"], right["length"] - left["length"]
        ))
    if distances[-1] == 0:
        raise ValueError("ordered branch II has zero extent")
    return [(distance / distances[-1], row["ca"], row["length"])
            for distance, row in zip(distances, rows, strict=True)]


def _ordered_interpolate(points: list[tuple[float, float, float]], s: float) -> tuple[float, float]:
    for left, right in pairwise(points):
        if s <= right[0]:
            fraction = (s - left[0]) / (right[0] - left[0]) if right[0] != left[0] else 0
            return (left[1] + fraction * (right[1] - left[1]),
                    left[2] + fraction * (right[2] - left[2]))
    return points[-1][1:]


def _middle_branch_error(reference: list[dict], candidate: list[dict], samples: int = 101) -> dict:
    ref, cand = _ordered_coordinates(reference), _ordered_coordinates(candidate)
    ca_errors, length_errors = [], []
    for index in range(samples):
        s = index / (samples - 1)
        ref_ca, ref_length = _ordered_interpolate(ref, s)
        cand_ca, cand_length = _ordered_interpolate(cand, s)
        ca_errors.append(abs(ref_ca - cand_ca))
        length_errors.append(abs(ref_length - cand_length))
    return {
        "method": "ordered-curve interpolation by normalised chord length",
        "samples": samples,
        "max_abs_ca_error": max(ca_errors),
        "rms_ca_error": _rms(ca_errors),
        "max_abs_length_error": max(length_errors),
        "rms_length_error": _rms(length_errors),
    }


def _fold(rows: list[dict], index: int) -> dict:
    neighbourhood = rows[index - 1:index + 2]
    ca_values = [row["ca"] for row in neighbourhood]
    lengths = [row["length"] for row in neighbourhood]
    return {
        "station": rows[index]["station"],
        "ca": rows[index]["ca"],
        "length": rows[index]["length"],
        "ca_sampling_interval": [min(ca_values), max(ca_values)],
        "length_sampling_interval": [min(lengths), max(lengths)],
        "sampling_note": (
            "The interval spans the accepted station and its two neighbours; "
            "it is a sampling-resolution diagnostic, not a bound on the unsampled extremum."
        ),
    }


def _interval_distance(left: list[float], right: list[float]) -> float:
    return max(0.0, left[0] - right[1], right[0] - left[1])


def _compare_fold(reference: dict, candidate: dict, tolerances: ConvergenceTolerances) -> dict:
    ca_width = max(reference["ca_sampling_interval"][1] - reference["ca_sampling_interval"][0],
                   candidate["ca_sampling_interval"][1] - candidate["ca_sampling_interval"][0])
    length_width = max(
        reference["length_sampling_interval"][1] - reference["length_sampling_interval"][0],
        candidate["length_sampling_interval"][1] - candidate["length_sampling_interval"][0],
    )
    resolved = (ca_width <= tolerances.max_fold_ca_sampling_width
                and length_width <= tolerances.max_fold_length_sampling_width)
    ca_interval_distance = _interval_distance(reference["ca_sampling_interval"],
                                               candidate["ca_sampling_interval"])
    length_interval_distance = _interval_distance(reference["length_sampling_interval"],
                                                   candidate["length_sampling_interval"])
    within = (ca_interval_distance <= tolerances.max_fold_ca_error
              and length_interval_distance <= tolerances.max_fold_length_error)
    return {
        "reference": reference,
        "candidate": candidate,
        "point_ca_difference": abs(reference["ca"] - candidate["ca"]),
        "point_length_difference": abs(reference["length"] - candidate["length"]),
        "ca_sampling_interval_distance": ca_interval_distance,
        "length_sampling_interval_distance": length_interval_distance,
        "sampling_widths_resolved": resolved,
        "comparison_note": (
            "Interval overlap measures discrete sampling compatibility only; it does not "
            "certify the location of an extremum between stations."
        ),
        "status": "pass" if resolved and within else "fail" if resolved else "inconclusive",
    }


def _read_profile(directory: Path, station: int) -> list[tuple[float, float]]:
    path = directory / f"station_{station:05d}.csv"
    if not path.is_file():
        raise ValueError(f"missing stationary profile: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"z_paper", "gap_over_r"}.issubset(rows[0]):
        raise ValueError(f"profile lacks paper coordinates: {path}")
    points = [(float(row["z_paper"]), float(row["gap_over_r"])) for row in rows]
    if len(points) < 2 or not all(math.isfinite(value) for point in points for value in point):
        raise ValueError(f"profile is too short or non-finite: {path}")
    return points


def _profile_coordinates(points: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    distances = [0.0]
    for left, right in pairwise(points):
        distances.append(distances[-1] + math.dist(left, right))
    if distances[-1] == 0:
        raise ValueError("profile has zero meridional extent")
    return [(distance / distances[-1], *point)
            for distance, point in zip(distances, points, strict=True)]


def _profile_error(reference: list[tuple[float, float]], candidate: list[tuple[float, float]],
                   samples: int = 201) -> tuple[float, float]:
    ref, cand = _profile_coordinates(reference), _profile_coordinates(candidate)
    errors = []
    for index in range(samples):
        s = index / (samples - 1)
        errors.append(math.dist(_ordered_interpolate(ref, s), _ordered_interpolate(cand, s)))
    return max(errors), _rms(errors)


def _nearest(rows: list[dict], ca: float) -> dict:
    return min(rows, key=lambda row: abs(row["ca"] - ca))


def _shape_comparisons(reference: dict, candidate: dict,
                       tolerances: ConvergenceTolerances) -> list[dict]:
    ref_rows, cand_rows = reference["rows"], candidate["rows"]
    ref_first, ref_last = reference["physical_folds"]
    cand_first, cand_last = candidate["physical_folds"]
    landmarks = [
        ("branch_III_endpoint", ref_rows[0], _nearest(candidate["branches"]["III"], ref_rows[0]["ca"])),
        ("branch_III_II_fold", ref_rows[ref_first], cand_rows[cand_first]),
        ("branch_II_I_fold", ref_rows[ref_last], cand_rows[cand_last]),
        ("branch_I_endpoint", ref_rows[-1], _nearest(candidate["branches"]["I"], ref_rows[-1]["ca"])),
    ]
    comparisons = []
    for name, ref_row, cand_row in landmarks:
        max_error, rms_error = _profile_error(
            _read_profile(reference["profiles"], ref_row["station"]),
            _read_profile(candidate["profiles"], cand_row["station"]),
        )
        ca_mismatch = abs(ref_row["ca"] - cand_row["ca"])
        status = ("inconclusive" if ca_mismatch > tolerances.max_profile_ca_mismatch
                  else "pass" if max_error <= tolerances.max_profile_error else "fail")
        comparisons.append({
            "landmark": name,
            "method": "nearest accepted states; profile interpolation only in normalised meridional coordinate",
            "reference_station": ref_row["station"],
            "candidate_station": cand_row["station"],
            "reference_ca": ref_row["ca"],
            "candidate_ca": cand_row["ca"],
            "ca_mismatch": ca_mismatch,
            "max_coordinate_error": max_error,
            "rms_coordinate_error": rms_error,
            "status": status,
        })
    return comparisons


def compare_runs(reference_dir: Path, candidate_dir: Path,
                 tolerances: ConvergenceTolerances, *,
                 reference_profiles: Path | None = None,
                 candidate_profiles: Path | None = None) -> dict:
    """Compare one refined/domain run with a declared complete reference run."""
    reference = _read_complete_run(reference_dir, reference_profiles)
    candidate = _read_complete_run(candidate_dir, candidate_profiles)
    curves = {
        "III": _outer_branch_error(reference["branches"]["III"], candidate["branches"]["III"]),
        "II": _middle_branch_error(reference["branches"]["II"], candidate["branches"]["II"]),
        "I": _outer_branch_error(reference["branches"]["I"], candidate["branches"]["I"]),
    }
    curves["III"]["status"] = ("pass" if curves["III"]["max_abs_length_error"]
                                  <= tolerances.max_length_error else "fail")
    curves["I"]["status"] = ("pass" if curves["I"]["max_abs_length_error"]
                                <= tolerances.max_length_error else "fail")
    curves["II"]["status"] = ("pass" if (
        curves["II"]["max_abs_length_error"] <= tolerances.max_length_error
        and curves["II"]["max_abs_ca_error"] <= tolerances.max_ca_error_branch_ii
    ) else "fail")
    folds = {
        "III_II": _compare_fold(
            _fold(reference["rows"], reference["physical_folds"][0]),
            _fold(candidate["rows"], candidate["physical_folds"][0]), tolerances),
        "II_I": _compare_fold(
            _fold(reference["rows"], reference["physical_folds"][1]),
            _fold(candidate["rows"], candidate["physical_folds"][1]), tolerances),
    }
    shapes = _shape_comparisons(reference, candidate, tolerances)
    statuses = ([item["status"] for item in curves.values()]
                + [item["status"] for item in folds.values()]
                + [item["status"] for item in shapes])
    overall = "pass" if all(value == "pass" for value in statuses) else (
        "inconclusive" if "inconclusive" in statuses and "fail" not in statuses else "fail"
    )
    return {
        "schema": "herrada-figure3-convergence-v1",
        "inputs": {
            "reference": {
                "path": reference["path"],
                "status_sha256": reference["status_sha256"],
                "branches_sha256": reference["branches_sha256"],
            },
            "candidate": {
                "path": candidate["path"],
                "status_sha256": candidate["status_sha256"],
                "branches_sha256": candidate["branches_sha256"],
            },
        },
        "tolerances": asdict(tolerances),
        "branch_partition": {
            "rule": "first and last chronological Ca reversals are physical folds",
            "reference_turns": reference["turns"],
            "candidate_turns": candidate["turns"],
            "reference_internal_turns": reference["internal_turns"],
            "candidate_internal_turns": candidate["internal_turns"],
        },
        "curves": curves,
        "folds": folds,
        "profiles": shapes,
        "status": overall,
        "claim_limit": (
            "This compares discrete axisymmetric stationary branches and selected saved shapes. "
            "Three-station fold intervals diagnose continuation sampling and are not certified "
            "bounds on unsampled extrema. The comparison does not establish stability or "
            "interpolate a solution between saved profiles."
        ),
    }


def _positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-profiles", type=Path)
    parser.add_argument("--candidate-profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for name in asdict(ConvergenceTolerances(*([1.0] * 8))):
        parser.add_argument("--" + name.replace("_", "-"), type=_positive, required=True)
    args = parser.parse_args(argv)
    tolerances = ConvergenceTolerances(**{
        name: getattr(args, name) for name in asdict(ConvergenceTolerances(*([1.0] * 8)))
    })
    report = compare_runs(args.reference, args.candidate, tolerances,
                          reference_profiles=args.reference_profiles,
                          candidate_profiles=args.candidate_profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
