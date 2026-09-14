"""Film-observable tests.

These use analytic interfaces, so the expected film thickness is known exactly
and the test measures the extractor rather than a simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from postprocess.film import (
    aussillous_quere,
    bretherton,
    film_from_interface_radius,
    measure_film,
    trim_sensitivity,
)
from problems.geometry import CapsuleGeometry

R_TUBE = 0.7
R_BODY = 0.56


def capsule_interface(n: int = 601, *, r_body: float = R_BODY) -> tuple[np.ndarray, np.ndarray]:
    """Analytic capsule interface: caps plus a cylinder of radius ``r_body``."""
    geom = CapsuleGeometry(r0=1.0, r_tube=R_TUBE, r_body=r_body, z_rear=1.0, l_domain=16.0)
    z = np.linspace(geom.z_rear, geom.z_front, n)
    r = np.array([geom.interface_radius(float(zi)) for zi in z])
    return z, r


def test_central_film_is_tube_minus_interface():
    assert film_from_interface_radius(r_interface=0.56, r_tube=0.7) == pytest.approx(0.14)


def test_rejects_an_interface_on_the_wall():
    with pytest.raises(ValueError, match="inside the tube"):
        film_from_interface_radius(r_interface=0.7, r_tube=0.7)


def test_flat_film_recovers_the_cylinder_radius():
    z, r = capsule_interface()
    m = measure_film(z, r, r_tube=R_TUBE)
    assert m.h_flat == pytest.approx(R_TUBE - R_BODY)
    assert m.h_scatter == pytest.approx(0.0, abs=1e-12)
    assert m.flatness == pytest.approx(0.0, abs=1e-12)


def test_minimum_gap_equals_the_flat_film_for_a_bare_capsule():
    """A capsule has no capillary ripple, so the two observables coincide.

    This is the control: on a real solution they must differ, and that
    difference is the reason the flat film is the reported quantity.
    """
    z, r = capsule_interface()
    m = measure_film(z, r, r_tube=R_TUBE)
    assert m.h_min == pytest.approx(m.h_flat)


def test_a_rear_ripple_depresses_the_minimum_but_not_the_flat_film():
    z, r = capsule_interface()
    geom = CapsuleGeometry(r0=1.0, r_tube=R_TUBE, r_body=R_BODY, z_rear=1.0, l_domain=16.0)
    # A localised bulge just behind the rear cap, well outside the central window.
    bulge = 0.02 * np.exp(-(((z - (geom.z_rear + 0.1 * geom.cylinder_length)) / 0.05) ** 2))
    m = measure_film(z, r + bulge, r_tube=R_TUBE)
    assert m.h_min < m.h_flat
    assert m.h_flat == pytest.approx(R_TUBE - R_BODY, rel=1e-9)


def test_trim_sensitivity_is_flat_for_a_uniform_film():
    z, r = capsule_interface()
    table = trim_sensitivity(z, r, r_tube=R_TUBE)
    values = np.array(list(table.values()))
    assert np.ptp(values) == pytest.approx(0.0, abs=1e-12)


def test_geometry_carries_the_bubble_extent():
    z, r = capsule_interface()
    geom = CapsuleGeometry(r0=1.0, r_tube=R_TUBE, r_body=R_BODY, z_rear=1.0, l_domain=16.0)
    m = measure_film(z, r, r_tube=R_TUBE)
    assert m.z_rear == pytest.approx(geom.z_rear)
    assert m.z_front == pytest.approx(geom.z_front)
    assert m.bubble_length == pytest.approx(geom.z_front - geom.z_rear)
    assert m.n_central > 100


def test_rejects_a_drained_film():
    z, r = capsule_interface()
    with pytest.raises(ValueError, match="drained"):
        measure_film(z, r * (R_TUBE / R_BODY), r_tube=R_TUBE)


def test_rejects_an_unresolved_central_window():
    z, r = capsule_interface(n=5)
    with pytest.raises(ValueError, match="coarsely resolved"):
        measure_film(z, r, r_tube=R_TUBE, trim=0.45)


def test_reference_laws_match_the_published_forms():
    # Bretherton: h/Rt = 0.643 (3Ca)^{2/3}, i.e. the prefactor 0.643*3^{2/3} = 1.3375
    assert 0.643 * 3.0 ** (2 / 3) == pytest.approx(1.3375, abs=5e-5)
    assert bretherton(0.01) == pytest.approx(1.3375 * 0.01 ** (2 / 3), rel=1e-4)
    # Aussillous-Quere is the Bretherton value softened by the 3.35 Ca^{2/3} term
    assert aussillous_quere(0.01) < bretherton(0.01)
    assert aussillous_quere(1e-6) == pytest.approx(bretherton(1e-6), rel=1e-3)


def test_flat_film_matches_the_basilisk_baseline_scale():
    """Sanity of units: the committed Basilisk table reports h/R_tube.

    At Ca_b = 0.0112 the parent records b_flat/R_tube = 0.0568, within 2% of the
    Aussillous-Quere fit. This test only checks that our reference-law helpers
    are on that scale, not that the FEM solver reproduces it.
    """
    assert aussillous_quere(0.0112414) == pytest.approx(0.0568, rel=0.02)
