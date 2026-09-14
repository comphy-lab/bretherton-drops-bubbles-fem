"""Geometry contract tests that do not import pyoomph."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from postprocess.film import aussillous_quere
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
    assert geom.z_front < geom.l_domain
    # The default case seeds the gap from the Aussillous-Quere film at its own Ca
    # rather than from a fixed fraction of the tube radius, so that the mesh
    # starts near the answer instead of at an arbitrary 0.8 R_tube.
    assert geom.r_body == pytest.approx(0.7 * (1.0 - aussillous_quere(case["physics"]["ca_in"])))
    assert geom.capsule_volume == pytest.approx(geom.volume)


def test_window_is_set_by_pads_not_by_a_total_length():
    """The window is a convergence parameter, so pads are the primitive."""
    geom = CapsuleGeometry.for_capillary_number(0.05, pad_rear=3.0, pad_front=3.0)
    assert geom.pad_rear == pytest.approx(3.0)
    assert geom.pad_front == pytest.approx(3.0)
    wider = geom.with_pads(pad_rear=5.0, pad_front=5.0)
    assert wider.l_domain > geom.l_domain
    # Changing the window must not change the bubble.
    assert wider.r_body == pytest.approx(geom.r_body)
    assert wider.capsule_volume == pytest.approx(geom.capsule_volume)
    assert wider.z_front - wider.z_rear == pytest.approx(geom.z_front - geom.z_rear)


def test_interface_polyline_spans_the_capsule_and_is_closed_at_the_tips():
    geom = CapsuleGeometry.for_capillary_number(0.05, pad_rear=3.0, pad_front=3.0)
    pts = geom.interface_polyline(n=101)
    assert len(pts) == 101
    assert pts[0][0] == pytest.approx(0.0, abs=1e-9)
    assert pts[-1][0] == pytest.approx(0.0, abs=1e-9)
    assert pts[0][1] == pytest.approx(geom.z_rear)
    assert pts[-1][1] == pytest.approx(geom.z_front)
    assert max(r for r, _ in pts) == pytest.approx(geom.r_body)
