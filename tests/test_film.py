"""Film-thickness helper tests."""

from __future__ import annotations

import pytest

from postprocess.film import film_from_interface_radius


def test_central_film_is_tube_minus_interface():
    assert film_from_interface_radius(r_interface=0.56, r_tube=0.7) == pytest.approx(0.14)


def test_rejects_an_interface_on_the_wall():
    with pytest.raises(ValueError, match="inside the tube"):
        film_from_interface_radius(r_interface=0.7, r_tube=0.7)
