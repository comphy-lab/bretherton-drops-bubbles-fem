"""Recover equation (2.3) lengths from saved quadratic interface profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from .driver import file_digest, meridional_length, write_branches


def read_csv_snapshot(path: Path) -> tuple[list[dict], str]:
    """Parse and hash the same byte snapshot, even if a live file changes."""
    data = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
    return rows, hashlib.sha256(data).hexdigest()


def remeasure(branches: Path, profiles: Path, output: Path) -> None:
    """Write a separate corrected table and provenance; leave restart data intact."""
    rows, branches_hash = read_csv_snapshot(branches)
    if not rows or [int(row["station"]) for row in rows] != list(range(len(rows))):
        raise ValueError("source must contain consecutive stations in continuation order")
    hashes = {}
    max_quadrature_change = 0.0
    max_polygon_deficit = 0.0
    for row in rows:
        path = profiles / f"station_{int(row['station']):05d}.csv"
        points, profile_hash = read_csv_snapshot(path)
        z = [float(p["z_paper"]) for p in points]
        r = [1.0 - float(p["gap_over_r"]) for p in points]
        span = max(z) - min(z)
        original_span = float(row.get("axial_span_over_r", row["bubble_length_over_r"]))
        if abs(span - original_span) > 1e-9:
            raise ValueError(f"profile/branch axial-span mismatch at station {row['station']}")
        length = meridional_length(z, r)
        check = meridional_length(z, r, quadrature_order=32)
        polygon = sum(math.hypot(z1 - z0, r1 - r0)
                      for z0, z1, r0, r1 in zip(z, z[1:], r, r[1:]))
        max_quadrature_change = max(max_quadrature_change, abs(length - check))
        max_polygon_deficit = max(max_polygon_deficit, abs(length - polygon))
        row["axial_span_over_r"] = span
        row["bubble_length_over_r"] = length
        hashes[path.name] = profile_hash
    output.mkdir(parents=True, exist_ok=False)
    write_branches(output / "branches.csv", rows)
    receipt = {
        "definition": "Herrada et al. (2023), equation (2.3): meridional arc length / R",
        "method": "16-point Gauss quadrature on each ordered quadratic interface element",
        "source_branches": str(branches.resolve()),
        "source_branches_sha256": branches_hash,
        "source_profiles": str(profiles.resolve()),
        "profile_sha256": hashes,
        "stations": len(rows),
        "max_16_vs_32_point_difference": max_quadrature_change,
        "max_quadratic_vs_polygon_difference": max_polygon_deficit,
        "corrected_branches_sha256": file_digest(output / "branches.csv"),
        "restart_modified": False,
    }
    (output / "remeasure.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    remeasure(args.branches, args.profiles, args.output_dir)


if __name__ == "__main__":
    main()
