"""Tangential placement of interface nodes that follow a moving, sharpening rim.

Why this exists
---------------
The free-surface kinematic condition fixes only the *normal* motion of an
interface node. Its tangential placement is left to the pseudo-solid mesh,
which keeps it roughly where it was. That is harmless while the interface moves
slowly, but not at a rear rim that sharpens into a cusp. The rim advances along
its own axis at a speed ``V`` that is large compared with ``R/t``, the flank
nodes do not follow it, and the tip runs into them: they converge on the tip at
a rate of about ``V/R``. Measured at ``Ca_in = 12``, that rate is ~150 per unit
time at a tip radius of 2.2e-5 and ~470 at 1.1e-5, against a collapse rate
``-d ln R/dt`` of 8 to 14. The packing, not the physics, then limits the step:
past about 45 elements per tip radius node-scale curvature kinks appear, and a
step that compresses the tip nodes by more than ~1.25 inverts elements.

What it does
------------
With ``sliding_nodes=True`` the interface nodes keep their normal motion from
the kinematic condition, while their tangential placement is prescribed. Each
node ``X`` of the reference (Lagrangian) configuration is mapped to

    X* = X + w_T(X) D(t) + w_C(X) (S(t) - 1) (X - c),
    D(t) = D0 + V (t - t0),     S(t) = exp(-(sigma0 + lambda (t - t0))),
    w_T = exp(-|X - c|^2 / L_T^2),   w_C = exp(-|X - c|^2 / L_C^2),

and ``(x - X*) . t = 0`` holds along the current tangent ``t``. ``c`` is the
rim position in the reference configuration. ``D`` translates the nodes near
the rim with it and ``S`` contracts them about it, so the refined patch stays on
the advancing tip and the number of elements per tip radius stays roughly
constant while the radius shrinks. ``L_T`` must be large compared with the
rim's displacement before the next remesh, so the bulk mesh is translated rather
than sheared; ``L_C`` spans the refined patch, a few tens of tip radii. The
windows are measured in the plane rather than along the chain, which is
adequate as long as no other part of the interface comes within ``L_T`` of the
rim.

The rim velocity has to be known far better than it can be predicted. Any
error ``dV`` along the rim's axis packs the nodes again at ``dV/R``, which at
``R = 5e-6`` makes a 1% error in ``V`` as bad as no sliding at a tenth of the
speed. The frame is therefore meant to be re-set after every time step by
:func:`relock`, from the rim position and radius just measured: ``D0`` and
``sigma0`` carry the map continuously across the re-set, and the new ``V`` and
``lambda`` steer the frame's image of the rim back onto the measured rim over a
few steps. Only the reference configuration (the mesh as built at the last
remesh) is fixed between remeshes, so the pseudo-solid keeps its memory.

The map is exact in the sense that matters: only the tangential *position* of
the nodes is prescribed, the normal one is still the kinematic condition, so the
computed interface and flow are those of the same equations. With ``D = 0`` and
``S = 1`` the constraint reduces to ``(x - X) . t = 0``: every interface node
moves only along the current normal through its reference position, which is
well posed in stationary solves as well as in a march.

This module is plain NumPy so that the map and the feedback can be tested
without pyoomph; ``problems.bretherton_bubble`` evaluates the same map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = ["SlidingFrame", "penalty_weight", "relock", "rim_motion", "tangential_target",
           "target_point", "window"]


@dataclass(frozen=True)
class SlidingFrame:
    """Parameters of the tangential map at time ``t0``.

    ``centre``, ``offset`` and ``velocity`` are ``(r, z)`` pairs, matching
    pyoomph's ``(x, y)`` in the axisymmetric meridional plane. ``offset`` is the
    translation ``D0`` already accumulated at ``t0`` and ``log_contraction`` the
    accumulated ``sigma0``. A window length of zero switches that window off,
    and the default frame is the pure normal motion ``(x - X) . t = 0``.

    The last four fields set how hard the map is held; see
    :func:`penalty_weight`. ``strength = 0`` releases the interface nodes to
    the pseudo-solid, which is what the first step after a remesh needs: the
    remesh changes the void's axial moment slightly, the centroid closure then
    shifts the whole bubble by up to a few tip radii, and a shift that large
    cannot be made of normal moves around a tip of radius ``R``.
    """

    t0: float = 0.0
    centre: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    contraction_rate: float = 0.0
    translation_length: float = 0.0
    contraction_length: float = 0.0
    offset: tuple[float, float] = (0.0, 0.0)
    log_contraction: float = 0.0
    strength: float = 1.0
    spacing_tip: float = 0.0
    spacing_growth: float = 0.0
    spacing_far: float = 0.0

    def __post_init__(self):
        if self.translation_length < 0 or self.contraction_length < 0:
            raise ValueError("window lengths must be non-negative")
        if self.strength < 0 or self.spacing_tip < 0 or self.spacing_growth < 0 or self.spacing_far < 0:
            raise ValueError("strength and spacings must be non-negative")

    def translation(self, t: float) -> np.ndarray:
        """``D(t)``: where the frame has carried the reference rim point."""
        return np.asarray(self.offset, dtype=float) + np.asarray(self.velocity, dtype=float) * (t - self.t0)

    def log_scale(self, t: float) -> float:
        """``sigma(t) = -ln S(t)``."""
        return self.log_contraction + self.contraction_rate * (t - self.t0)


def window(points, centre, length: float) -> np.ndarray:
    """``exp(-|X - c|^2 / L^2)`` at each point, identically zero when ``L = 0``."""
    pts = np.asarray(points, dtype=float)
    if length <= 0:
        return np.zeros(len(pts))
    d2 = np.sum((pts - np.asarray(centre, dtype=float)) ** 2, axis=1)
    return np.exp(-d2 / length**2)


def tangential_target(points, t: float, frame: SlidingFrame) -> np.ndarray:
    """Target positions ``X*`` of reference points ``X`` at time ``t``.

    The solver imposes only the tangential component of ``x - X*``; this is the
    full vector, for testing and for diagnosing a march.
    """
    X = np.asarray(points, dtype=float)
    c = np.asarray(frame.centre, dtype=float)
    w_t = window(X, c, frame.translation_length)[:, None]
    w_c = window(X, c, frame.contraction_length)[:, None]
    shrink = math.exp(-frame.log_scale(t)) - 1.0
    return X + w_t * frame.translation(t) + w_c * shrink * (X - c)


def target_point(xr: float, xz: float, t: float, frame: SlidingFrame) -> tuple[float, float]:
    """:func:`tangential_target` for one point, in plain floats.

    The solver calls this for every interface integration point at every
    assembly, so it avoids NumPy's per-call overhead.
    """
    cr, cz = frame.centre
    dr, dz = xr - cr, xz - cz
    d2 = dr * dr + dz * dz
    tau = t - frame.t0
    tr = tz = 0.0
    if frame.translation_length > 0:
        w = math.exp(-d2 / frame.translation_length**2)
        tr = w * (frame.offset[0] + frame.velocity[0] * tau)
        tz = w * (frame.offset[1] + frame.velocity[1] * tau)
    if frame.contraction_length > 0:
        w = math.exp(-d2 / frame.contraction_length**2) * (
            math.exp(-(frame.log_contraction + frame.contraction_rate * tau)) - 1.0)
        tr, tz = tr + w * dr, tz + w * dz
    return xr + tr, xz + tz


def penalty_weight(xr: float, xz: float, frame: SlidingFrame, far_spacing: float) -> float:
    """``strength / h(X)``, the factor of the tangential penalty at ``X``.

    ``h`` stands for the interface element length in the reference mesh:
    ``spacing_tip + spacing_growth |X - c|`` near the rim, capped at
    ``spacing_far`` (or at ``far_spacing`` when the frame leaves it at zero).
    Dividing by it makes the penalty a fixed multiple of the pseudo-solid's
    stiffness, which is scale-free in the plane, on elements of every size. It
    is a function of the reference position only, so it adds nothing to the
    Jacobian; the element's own current length would, and leaving that
    derivative out (as pyoomph does for element sizes) spoils Newton
    convergence wherever ``x - X*`` is comparable with ``h``.
    """
    far = frame.spacing_far if frame.spacing_far > 0 else far_spacing
    h = far
    if frame.spacing_tip > 0:
        d = math.hypot(xr - frame.centre[0], xz - frame.centre[1])
        h = min(far, frame.spacing_tip + frame.spacing_growth * d)
    return frame.strength / h


def relock(frame: SlidingFrame, t: float, *, rim, radius: float, reference_radius: float,
           velocity, rate: float, horizon: float) -> SlidingFrame:
    """Re-set the frame at time ``t`` so that it keeps following the measured rim.

    ``rim`` and ``radius`` are the rim position and tip radius measured at
    ``t``; ``reference_radius`` is the tip radius in the reference
    configuration; ``velocity`` and ``rate`` are the latest measured rim
    velocity and collapse rate. The map is continuous at ``t``: the accumulated
    translation and contraction are carried over. The new velocity and rate are
    the measured ones plus the correction that brings the frame's image of the
    rim onto the measured rim, and its contraction onto ``radius /
    reference_radius``, over ``horizon``.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    offset = frame.translation(t)
    sigma = frame.log_scale(t)
    image = np.asarray(frame.centre, dtype=float) + offset
    v = np.asarray(velocity, dtype=float) + (np.asarray(rim, dtype=float) - image) / horizon
    lam = float(rate) + (-math.log(radius / reference_radius) - sigma) / horizon
    return replace(frame, t0=float(t), offset=(float(offset[0]), float(offset[1])),
                   log_contraction=float(sigma), velocity=(float(v[0]), float(v[1])),
                   contraction_rate=lam)


def rim_motion(t_a: float, pos_a, radius_a: float, t_b: float, pos_b, radius_b: float):
    """Rim velocity and collapse rate between two measurements.

    Returns ``(velocity, rate)``: the mean velocity of the rim point and
    ``-d ln R/dt``, both as finite differences. Measure the two ends on the
    same mesh, i.e. not across a remesh, so that the remesh projection does not
    enter the estimate.
    """
    dt = float(t_b) - float(t_a)
    if dt <= 0:
        raise ValueError("t_b must be later than t_a")
    if radius_a <= 0 or radius_b <= 0:
        raise ValueError("radii must be positive")
    velocity = (np.asarray(pos_b, dtype=float) - np.asarray(pos_a, dtype=float)) / dt
    return velocity, float(np.log(radius_a / radius_b) / dt)
