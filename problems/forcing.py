"""Pure parameter checks and force scaling for the bubble problem."""

from __future__ import annotations

import math


def validate_capillary_number(ca_in: float, *, allow_signed: bool = False) -> None:
    """Validate an imposed flow, optionally allowing zero and reverse flow."""
    if not math.isfinite(ca_in):
        raise ValueError("Ca_in must be finite")
    if not allow_signed and ca_in <= 0:
        raise ValueError("Ca_in must be positive unless allow_signed_flow=True")


def validate_bond_number(bond: float) -> None:
    """Validate the non-negative tube-radius Bond number."""
    if not math.isfinite(bond) or bond < 0:
        raise ValueError("Bo must be finite and non-negative")


def bond_force_density(bond: float, r_tube: float) -> tuple[float, float]:
    """Return the dimensionless ``(r, z)`` gravity force per unit volume."""
    validate_bond_number(bond)
    if not math.isfinite(r_tube) or r_tube <= 0:
        raise ValueError("tube radius must be finite and positive")
    return 0.0, -bond / r_tube**2
