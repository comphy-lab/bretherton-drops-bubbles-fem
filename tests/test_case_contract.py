"""Portable case-schema tests that do not import pyoomph.

These run without a built pyoomph environment, so a case file can be checked on
any machine. They test the contract between a case document and the solver, not
the solver.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from problems.geometry import geometry_from_case

ROOT = Path(__file__).resolve().parents[1]
CASES = sorted((ROOT / "cases").glob("*.json"))
REQUIRED = (
    "schema",
    "case_id",
    "geometry",
    "nondimensionalization",
    "physics",
    "boundaries",
    "closure",
    "tolerances",
    "mesh",
    "solve",
    "output",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_at_least_one_case_exists():
    assert CASES


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_is_pyoomph_case_v2(path: Path):
    payload = load(path)
    assert payload["schema"] == "pyoomph-case-v2"
    missing = [key for key in REQUIRED if key not in payload]
    assert not missing, f"{path.name} is missing {missing}"
    assert payload["geometry"]["kind"] == "axisymmetric"
    assert payload["geometry"]["coordinates"] == ["r", "z"]
    assert payload["geometry"]["frame"] == "co-moving"
    assert payload["physics"]["interior"] == "void"


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_physics_builds_a_valid_window(path: Path):
    payload = load(path)
    geom = geometry_from_case(payload["physics"])
    # The window must actually contain the bubble with the requested pads.
    assert geom.z_rear > 0
    assert geom.z_front < geom.l_domain
    assert geom.pad_rear == pytest.approx(payload["physics"]["pad_rear"])
    assert geom.pad_front == pytest.approx(payload["physics"]["pad_front"])
    # The seeded capsule must carry the target volume: R0 = 1 is defined by it.
    assert geom.capsule_volume == pytest.approx(geom.volume)


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_choices_are_ones_the_solver_accepts(path: Path):
    payload = load(path)
    assert payload["boundaries"]["outlet"] in ("poiseuille", "traction_free")
    assert payload["closure"]["frame_speed"] in ("centroid", "tip")
    assert payload["solve"]["mode"] in ("steady", "transient")
    assert payload["mesh"]["film_elements"] >= 1
    assert payload["mesh"]["bulk_resolution"] > 0
    assert payload["physics"]["ca_in"] > 0
    assert payload["physics"]["laplace"] >= 0
    for name, value in payload["tolerances"].items():
        assert value > 0, name
