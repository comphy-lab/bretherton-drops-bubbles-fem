"""Geometry contract tests that do not import pyoomph."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from problems.geometry import CapsuleGeometry, geometry_from_case


ROOT = Path(__file__).resolve().parents[1]


def test_capsule_volume_matches_r0():
    geom = CapsuleGeometry(r0=1.0, r_tube=0.7, r_body=0.56, z_rear=1.0, l_domain=16.0)
    assert geom.cylinder_length > 0
    assert geom.capsule_volume == pytest.approx(geom.volume)
    assert geom.capsule_volume == pytest.approx((4.0 / 3.0) * math.pi)


def test_interface_radius_on_body_and_caps():
    geom = CapsuleGeometry(r0=1.0, r_tube=0.7, r_body=0.56, z_rear=1.0, l_domain=16.0)
    assert geom.interface_radius(geom.rear_centre_z) == pytest.approx(geom.r_body)
    assert geom.interface_radius(geom.front_centre_z) == pytest.approx(geom.r_body)
    assert geom.interface_radius(geom.z_rear) == pytest.approx(0.0)
    assert geom.interface_radius(geom.z_front) == pytest.approx(0.0)


def test_rejects_a_body_that_fills_the_tube():
    with pytest.raises(ValueError, match="inside the tube"):
        CapsuleGeometry(r0=1.0, r_tube=0.7, r_body=0.7, z_rear=1.0, l_domain=16.0)


def test_case_physics_builds_the_default_capsule():
    case = json.loads((ROOT / "cases" / "bubble_ca0p05.json").read_text(encoding="utf-8"))
    geom = geometry_from_case(case["physics"])
    assert geom.r_tube == 0.7
    assert geom.r_body == pytest.approx(0.56)
    assert geom.z_front < geom.l_domain
