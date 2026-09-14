"""Capsule geometry for a confined void bubble in a circular tube.

Lengths are scaled on the volume-equivalent radius R0, so a spherical bubble of
the same volume has radius 1. The bubble interior is not part of the liquid
domain: it is a void bounded by the free surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CapsuleGeometry:
    """Axisymmetric capsule (cylinder plus hemispherical caps) inside a tube."""

    r0: float
    r_tube: float
    r_body: float
    z_rear: float
    l_domain: float

    def __post_init__(self) -> None:
        if min(self.r0, self.r_tube, self.r_body, self.l_domain) <= 0:
            raise ValueError("R0, Rtube, Rb and Ldomain must be positive")
        if self.r_body >= self.r_tube:
            raise ValueError("capsule radius must lie strictly inside the tube")
        if self.r_body > self.r0:
            raise ValueError("capsule radius cannot exceed the volume-equivalent radius")
        if self.z_rear <= 0:
            raise ValueError("rear tip collides with the inlet")
        if self.z_front >= self.l_domain:
            raise ValueError("front tip collides with the outlet")

    @property
    def cylinder_length(self) -> float:
        """Axial length of the cylindrical body that restores volume 4/3 π R0³."""
        return (4.0 / 3.0) * (self.r0**3 / self.r_body**2 - self.r_body)

    @property
    def z_front(self) -> float:
        return self.z_rear + 2.0 * self.r_body + self.cylinder_length

    @property
    def rear_centre_z(self) -> float:
        return self.z_rear + self.r_body

    @property
    def front_centre_z(self) -> float:
        return self.z_front - self.r_body

    @property
    def volume(self) -> float:
        return (4.0 / 3.0) * math.pi * self.r0**3

    @property
    def capsule_volume(self) -> float:
        return (4.0 / 3.0) * math.pi * self.r_body**3 + math.pi * self.r_body**2 * self.cylinder_length

    def interface_radius(self, z: float) -> float:
        """Radial location of the void/liquid interface at axial station z."""
        if z < self.z_rear or z > self.z_front:
            raise ValueError("station is not on the capsule")
        if z < self.rear_centre_z:
            delta = z - self.z_rear
            return math.sqrt(max(self.r_body**2 - (self.r_body - delta) ** 2, 0.0))
        if z > self.front_centre_z:
            delta = self.z_front - z
            return math.sqrt(max(self.r_body**2 - (self.r_body - delta) ** 2, 0.0))
        return self.r_body


def geometry_from_case(physics: dict) -> CapsuleGeometry:
    """Build the capsule from the portable case physics block."""
    r0 = float(physics["r0"])
    r_tube = float(physics["r_tube"])
    return CapsuleGeometry(
        r0=r0,
        r_tube=r_tube,
        r_body=float(physics["rb0_frac"]) * r_tube,
        z_rear=float(physics["z_rear"]),
        l_domain=float(physics["l_domain"]),
    )
