"""Portable case-schema tests that do not import pyoomph."""

from __future__ import annotations

import json
from pathlib import Path

from problems.geometry import geometry_from_case


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "cases" / "bubble_ca0p05.json"
REQUIRED = (
    "schema",
    "case_id",
    "geometry",
    "nondimensionalization",
    "physics",
    "tolerances",
    "mesh",
    "output",
    "completion",
)


def test_default_case_is_pyoomph_case_v1():
    payload = json.loads(CASE.read_text(encoding="utf-8"))
    assert payload["schema"] == "pyoomph-case-v1"
    assert all(key in payload for key in REQUIRED)
    assert payload["geometry"]["kind"] == "axisymmetric"
    assert payload["geometry"]["coordinates"] == ["r", "z"]
    assert payload["geometry"]["frame"] == "co-moving"
    assert payload["physics"]["interior"] == "void"
    assert payload["output"]["write_states"] is True
    assert payload["output"]["cadence"] > 0
    assert payload["completion"]["kind"] == "time"
    assert payload["completion"]["value"] > 0
    assert payload["mesh"]["characteristic_size"] > 0
    for name, value in payload["tolerances"].items():
        assert value > 0, name
    geometry_from_case(payload["physics"])
