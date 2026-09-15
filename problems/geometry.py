"""Capsule geometry for a confined void bubble in a circular tube.

Lengths are scaled on the volume-equivalent radius ``R0``, so a spherical bubble
of the same volume has radius 1. The bubble interior is not part of the liquid
domain: it is a void bounded by the free surface.

The capsule is a seeded shape, not an answer. Its cylindrical radius sets the
initial film gap, and the closer that gap is to the converged Bretherton film
the less the mesh has to compress on the way to the steady state. Hence
:meth:`CapsuleGeometry.for_capillary_number`, which seeds the gap from the
Aussillous-Quere fit rather than from a fixed fraction of the tube radius.

The window is described by *pads* - the liquid length ahead of and behind the
bubble - rather than by a total domain length. In a co-moving frame the bubble
does not travel, so the only thing the window has to do is give the cap
disturbance room to decay to the fully developed profile at each end. Pads are
therefore the quantity a window-independence study varies, and the total length
is derived.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from postprocess.film import aussillous_quere


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

    # ------------------------------------------------------------- constructors

    @staticmethod
    def bubble_length(r0: float, r_body: float) -> float:
        """Axial extent of the volume-matched capsule, without building one.

        Needed to size the window before the geometry exists, since the validator
        refuses a domain shorter than the bubble.
        """
        if r_body <= 0 or r0 <= 0:
            raise ValueError("R0 and Rb must be positive")
        return 2.0 * r_body + (4.0 / 3.0) * (r0**3 / r_body**2 - r_body)

    @staticmethod
    def expected_bubble_speed(ca_in: float, *, iterations: int = 60) -> float:
        """Self-consistent estimate of the bubble speed for an imposed inlet speed.

        ``Ca_b`` is an output of this formulation, not an input, and it exceeds
        ``Ca_in`` substantially. The ratio is the stagnant-film identity
        ``(R_t/(R_t - h))^2``: 1.03 at ``Ca_b ~ 1e-4`` but 2.29 at ``Ca_b ~ 2.3``.
        Combining it with the Aussillous-Quere film gives a fixed point in the
        ratio, solved here by direct iteration.

        This matters only for seeding, and the converged steady state does not
        depend on it. But at ``Ca_in = 2`` a capsule seeded from ``AQ(Ca_in)``
        carries a film of 0.337 where the bubble wants 0.363, and the transient
        must deform the mesh far enough that an element shrinks by a factor of
        three, triggering a remesh into a solve whose backward error exceeds the
        solver's own limit. Seeding from the bubble's own speed removes that.
        The parent Basilisk campaign inverts the same relation when it chooses
        ``Ca_in = 0.4788`` to target ``Ca_b = 1``.
        """
        if ca_in <= 0:
            raise ValueError("Ca_in must be positive")
        ratio = 1.0
        for _ in range(iterations):
            film = aussillous_quere(ca_in * ratio)
            if not 0.0 < film < 1.0:
                break
            ratio = 1.0 / (1.0 - film) ** 2
        return ca_in * ratio

    @classmethod
    def for_capillary_number(
        cls,
        ca: float,
        *,
        r0: float = 1.0,
        r_tube: float = 0.7,
        pad_rear: float = 3.0,
        pad_front: float = 3.0,
        seed_from_bubble_speed: bool = False,
    ) -> CapsuleGeometry:
        """Seed the capsule with the Aussillous-Quere film gap for this ``Ca``.

        ``pad_rear`` and ``pad_front`` are the liquid lengths behind and ahead of
        the bubble, in units of the tube radius. They are separate because the
        two ends are not physically alike: the rear meniscus carries the
        capillary ripple and the film relaxation, the front sees only the
        Poiseuille approach, so they need not converge at the same pad.
        """
        seed_ca = cls.expected_bubble_speed(ca) if seed_from_bubble_speed else ca
        if not 0.0 < aussillous_quere(seed_ca) < 1.0:
            raise ValueError(
                f"the Aussillous-Quere film at Ca={seed_ca} is not a fraction of R_tube"
            )
        r_body = r_tube * (1.0 - aussillous_quere(seed_ca))
        z_rear = pad_rear * r_tube
        z_front = z_rear + cls.bubble_length(r0, r_body)
        return cls(
            r0=r0,
            r_tube=r_tube,
            r_body=r_body,
            z_rear=z_rear,
            l_domain=z_front + pad_front * r_tube,
        )

    def with_pads(self, *, pad_rear: float, pad_front: float) -> CapsuleGeometry:
        """Same bubble, different window. The quantity a window study varies."""
        z_rear = pad_rear * self.r_tube
        z_front = z_rear + self.bubble_length(self.r0, self.r_body)
        return CapsuleGeometry(
            r0=self.r0,
            r_tube=self.r_tube,
            r_body=self.r_body,
            z_rear=z_rear,
            l_domain=z_front + pad_front * self.r_tube,
        )

    # ---------------------------------------------------------------- geometry

    @property
    def cylinder_length(self) -> float:
        """Axial length of the cylindrical body that restores volume 4/3 pi R0^3."""
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
    def z_centroid(self) -> float:
        """Axial centroid of the void. Fore-aft symmetric, so it is the midpoint."""
        return 0.5 * (self.z_rear + self.z_front)

    @property
    def volume(self) -> float:
        """The conserved void volume, 4/3 pi R0^3. This is a constraint, not an output."""
        return (4.0 / 3.0) * math.pi * self.r0**3

    @property
    def capsule_volume(self) -> float:
        return (4.0 / 3.0) * math.pi * self.r_body**3 + math.pi * self.r_body**2 * self.cylinder_length

    @property
    def axial_moment(self) -> float:
        """``\\int_V z dV`` over the void, the target of the centroid closure."""
        return self.volume * self.z_centroid

    @property
    def seed_gap(self) -> float:
        """Initial film gap ``R_tube - R_body``: the length the mesh must resolve."""
        return self.r_tube - self.r_body

    @property
    def pad_rear(self) -> float:
        """Liquid length behind the bubble, in tube radii."""
        return self.z_rear / self.r_tube

    @property
    def pad_front(self) -> float:
        """Liquid length ahead of the bubble, in tube radii."""
        return (self.l_domain - self.z_front) / self.r_tube

    def interface_radius(self, z: float, *, atol: float = 1e-12) -> float:
        """Radial location of the void/liquid interface at axial station z.

        The tolerance exists because callers legitimately evaluate at the tips:
        ``linspace(z_rear, z_front, n)`` can land a hair outside the interval by
        rounding, and rejecting that would be a spurious failure rather than a
        caught error.
        """
        if z < self.z_rear - atol or z > self.z_front + atol:
            raise ValueError("station is not on the capsule")
        z = min(max(z, self.z_rear), self.z_front)
        if z < self.rear_centre_z:
            delta = z - self.z_rear
            return math.sqrt(max(self.r_body**2 - (self.r_body - delta) ** 2, 0.0))
        if z > self.front_centre_z:
            delta = self.z_front - z
            return math.sqrt(max(self.r_body**2 - (self.r_body - delta) ** 2, 0.0))
        return self.r_body

    def interface_polyline(self, n: int = 241) -> list[tuple[float, float]]:
        """``(r, z)`` samples of the seeded interface, rear tip first.

        Used for mesh-size grading, which needs the interface as data rather than
        as a formula so that the same callback works after a remesh, when the
        interface is no longer a capsule.
        """
        if n < 3:
            raise ValueError("need at least three samples")
        out: list[tuple[float, float]] = []
        for i in range(n):
            z = self.z_rear + (self.z_front - self.z_rear) * i / (n - 1)
            out.append((self.interface_radius(z), z))
        return out


def geometry_from_case(physics: dict, *,
                       seed_from_bubble_speed: bool = False) -> CapsuleGeometry:
    """Build the capsule from the portable case physics block.

    ``rb0_frac`` seeds the gap directly; omitting it seeds from the
    Aussillous-Quere film at ``ca_in`` instead, which is the preferred route.
    """
    r0 = float(physics["r0"])
    r_tube = float(physics["r_tube"])
    if "rb0_frac" in physics:
        return CapsuleGeometry(
            r0=r0,
            r_tube=r_tube,
            r_body=float(physics["rb0_frac"]) * r_tube,
            z_rear=float(physics["z_rear"]),
            l_domain=float(physics["l_domain"]),
        )
    return CapsuleGeometry.for_capillary_number(
        float(physics["ca_in"]),
        r0=r0,
        r_tube=r_tube,
        pad_rear=float(physics["pad_rear"]),
        pad_front=float(physics["pad_front"]),
        seed_from_bubble_speed=seed_from_bubble_speed,
    )
