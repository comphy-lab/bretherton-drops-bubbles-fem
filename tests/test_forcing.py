"""Portable force-scaling tests, plus constructor checks when pyoomph exists."""

from __future__ import annotations

import math

import pytest

from problems.forcing import (
    bond_force_density,
    validate_bond_number,
    validate_capillary_number,
)


def test_tube_radius_bond_force_has_physical_sign_and_scaling():
    assert bond_force_density(1.5, 0.5) == pytest.approx((0.0, -6.0))
    assert bond_force_density(1.5, 1.0) == pytest.approx((0.0, -1.5))
    assert bond_force_density(0.0, 0.5) == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize("bond", [-1.0, math.inf, -math.inf, math.nan])
def test_invalid_bond_numbers_are_rejected(bond):
    with pytest.raises(ValueError, match="Bo"):
        validate_bond_number(bond)


@pytest.mark.parametrize("radius", [0.0, -1.0, math.inf, math.nan])
def test_invalid_tube_radii_are_rejected(radius):
    with pytest.raises(ValueError, match="tube radius"):
        bond_force_density(1.0, radius)


def test_signed_flow_is_an_explicit_opt_in():
    for ca_in in (0.0, -0.2):
        with pytest.raises(ValueError, match="positive"):
            validate_capillary_number(ca_in)
        validate_capillary_number(ca_in, allow_signed=True)

    for ca_in in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="finite"):
            validate_capillary_number(ca_in, allow_signed=True)


def test_problem_constructor_exposes_bond_and_front_tip_closure():
    pytest.importorskip("pyoomph")
    from problems.bretherton_bubble import VoidBubbleProblem
    from problems.geometry import CapsuleGeometry

    geometry = CapsuleGeometry.for_capillary_number(0.05)
    problem = VoidBubbleProblem(
        geometry=geometry,
        ca_in=-0.05,
        bond=1.56,
        allow_signed_flow=True,
        closure="front_tip",
    )

    assert problem.ca_in == pytest.approx(-0.05)
    assert problem.current_bond() == pytest.approx(1.56)
    assert problem.closure == "front_tip"


def test_problem_defaults_preserve_figure2_configuration():
    pytest.importorskip("pyoomph")
    from problems.bretherton_bubble import VoidBubbleProblem
    from problems.geometry import CapsuleGeometry

    problem = VoidBubbleProblem(
        geometry=CapsuleGeometry.for_capillary_number(0.05), ca_in=0.05
    )

    assert problem.bond == 0.0
    assert problem.allow_signed_flow is False
    assert problem.closure == "centroid"
