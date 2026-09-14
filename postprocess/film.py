"""Film-thickness observables for a void bubble in a tube.

The comparison quantity is the **deposited central film**, not the minimum
gap. The two differ substantially: the minimum sits in the rear-meniscus
capillary ripple and understates the flat film by tens of percent, increasingly
so as ``Ca`` falls. The Basilisk companion measures the flat film as the median
interface radius over the central portion of the bubble, discarding a fraction
``trim`` of the bubble length at each meniscus, and this module reproduces that
definition so the two solvers report the same number.

All radii are in the shared nondimensionalisation ``sigma = mu = R0 = 1``, with
``R0`` the volume-equivalent bubble radius. Thicknesses are returned in the same
units; divide by ``r_tube`` to obtain the ``h/R_tube`` convention used in the
Basilisk validation table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_TRIM = 0.30
"""Fraction of the bubble length discarded at each meniscus, matching Basilisk."""


def film_from_interface_radius(*, r_interface: float, r_tube: float) -> float:
    """Return ``h = r_tube - r_interface`` for a single cylindrical station."""
    if r_tube <= 0:
        raise ValueError("tube radius must be positive")
    if r_interface < 0 or r_interface >= r_tube:
        raise ValueError("interface radius must lie strictly inside the tube")
    return r_tube - r_interface


@dataclass(frozen=True)
class FilmMeasurement:
    """Film observables extracted from one interface polyline.

    ``h_flat`` is the comparison quantity. ``h_min`` is the global minimum gap,
    retained only because the Basilisk solver logs it in-flight; the two are not
    interchangeable. ``h_scatter`` is the standard deviation of the gap over the
    central window and is the evidence that the film is actually flat there: a
    bubble too short to develop a flat film shows ``h_scatter`` comparable to
    ``h_flat`` itself, and no amount of averaging repairs that.
    """

    h_flat: float
    h_min: float
    h_scatter: float
    z_rear: float
    z_front: float
    bubble_length: float
    n_central: int

    @property
    def flatness(self) -> float:
        """``h_scatter / h_flat``: dimensionless non-uniformity of the central film."""
        return self.h_scatter / self.h_flat


def measure_film(
    z: np.ndarray,
    r: np.ndarray,
    *,
    r_tube: float,
    trim: float = DEFAULT_TRIM,
) -> FilmMeasurement:
    """Extract the deposited central film from an interface polyline.

    ``z`` and ``r`` are the axial and radial coordinates of the liquid/void
    interface, in any order; the caps are located from the axial extent. The
    central window is ``z_rear + trim*L < z < z_front - trim*L`` with
    ``L = z_front - z_rear``, and ``h_flat`` is ``r_tube`` minus the median
    interface radius over that window.

    The median rather than the mean is deliberate and follows the Basilisk
    pipeline: it is insensitive to the handful of nodes a graded mesh places on
    the shoulders of the window. Node density is not uniform in a finite-element
    interface, so this is a median over nodes and not over arclength; on a flat
    film the distinction is immaterial, and ``h_scatter`` is reported so that
    assumption is checkable rather than assumed.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if z.shape != r.shape:
        raise ValueError("z and r must have the same shape")
    if z.size < 3:
        raise ValueError("need at least three interface points")
    if r_tube <= 0:
        raise ValueError("tube radius must be positive")
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must lie in [0, 0.5)")
    if float(r.max()) >= r_tube:
        raise ValueError("interface touches or crosses the wall; the film has drained")

    z_rear = float(z.min())
    z_front = float(z.max())
    length = z_front - z_rear
    if length <= 0:
        raise ValueError("interface has no axial extent")

    central = (z > z_rear + trim * length) & (z < z_front - trim * length)
    if int(central.sum()) < 3:
        raise ValueError(
            f"central window holds {int(central.sum())} points; "
            "the interface is too coarsely resolved or trim is too aggressive"
        )

    gap = r_tube - r[central]
    return FilmMeasurement(
        h_flat=float(np.median(gap)),
        h_min=float(r_tube - r.max()),
        h_scatter=float(np.std(gap)),
        z_rear=z_rear,
        z_front=z_front,
        bubble_length=length,
        n_central=int(central.sum()),
    )


def trim_sensitivity(
    z: np.ndarray,
    r: np.ndarray,
    *,
    r_tube: float,
    trims: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40),
) -> dict[float, float]:
    """Return ``h_flat`` for several trim fractions.

    The Basilisk post-processor asserts in prose that ``h_flat`` moves by less
    than 0.5% for ``trim`` between 0.20 and 0.40, with no committed table behind
    the claim. This function produces that table, so the insensitivity is
    measured here rather than inherited.
    """
    return {
        trim: measure_film(z, r, r_tube=r_tube, trim=trim).h_flat for trim in trims
    }


def bretherton(ca: float) -> float:
    """Bretherton's lubrication asymptote ``h/R_tube = 1.337 Ca^{2/3}``.

    Valid as ``Ca -> 0``; it is an asymptotic limit of the same equations, not
    independent data, so agreement is a limiting-case check and not validation.
    """
    if ca <= 0:
        raise ValueError("Ca must be positive")
    return 0.643 * (3.0 * ca) ** (2.0 / 3.0)


def aussillous_quere(ca: float) -> float:
    """Aussillous-Quere fit to Taylor's data, ``h/R_tube = 1.337 Ca^{2/3}/(1+3.35 Ca^{2/3})``.

    Empirical, and the appropriate comparison away from the asymptotic limit.
    """
    b = bretherton(ca)
    return b / (1.0 + 2.5 * b)
