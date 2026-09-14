"""Deposited central-film thickness for a void bubble in a tube.

The comparison thickness is the deposited central film, not the instantaneous
minimum gap and not the seeded capsule gap. Encode mesh spacing as Δ/R0 with
R0 the volume-equivalent radius.
"""

from __future__ import annotations


def film_from_interface_radius(*, r_interface: float, r_tube: float) -> float:
    """Return h = Rtube − r_interface for a cylindrical film station."""
    if r_tube <= 0:
        raise ValueError("tube radius must be positive")
    if r_interface < 0 or r_interface >= r_tube:
        raise ValueError("interface radius must lie strictly inside the tube")
    return r_tube - r_interface
