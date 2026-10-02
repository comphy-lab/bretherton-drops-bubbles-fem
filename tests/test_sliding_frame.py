"""The tangential map behind ``sliding_nodes``, on synthetic rims.

:mod:`problems.sliding_frame` imports no pyoomph, so the map is checked here
against shapes whose answer is known: a parabolic tip that translates and
shrinks must be carried onto the translated, shrunk parabola with the same
number of nodes per tip radius.
"""

from __future__ import annotations

import numpy as np
import pytest

from problems.sliding_frame import (
    SlidingFrame,
    relock,
    rim_motion,
    tangential_target,
    target_point,
    window,
)

R = 1e-6
"""Tip radius of the synthetic rim."""
CENTRE = (0.11, 0.024)


def parabolic_tip(radius: float, centre=CENTRE, axis=(0.6, -0.8), n: int = 81,
                  half_width: float = 4.0) -> np.ndarray:
    """Nodes on a parabolic tip ``eta = xi^2 / (2 R)`` opening against ``axis``.

    Spaced uniformly in ``xi`` over ``|xi| <= half_width * radius``, i.e. a
    fixed number of nodes per tip radius. The flanks reach ``half_width^2 / 2``
    radii behind the tip, which is the distance the windows see.
    """
    a = np.asarray(axis, dtype=float) / np.hypot(*axis)
    tangent = np.array([-a[1], a[0]])
    xi = np.linspace(-half_width, half_width, n) * radius
    eta = xi**2 / (2.0 * radius)
    return np.asarray(centre) + xi[:, None] * tangent - eta[:, None] * a


def test_the_default_frame_leaves_every_point_where_it_is():
    pts = parabolic_tip(R)
    assert np.array_equal(tangential_target(pts, 3.7, SlidingFrame()), pts)


def test_a_zero_length_window_is_off_whatever_it_multiplies():
    frame = SlidingFrame(t0=1.0, centre=CENTRE, velocity=(0.1, 0.05), contraction_rate=30.0)
    pts = parabolic_tip(R)
    assert np.array_equal(tangential_target(pts, 1.01, frame), pts)
    assert np.array_equal(window(pts, CENTRE, 0.0), np.zeros(len(pts)))


def test_negative_window_lengths_are_rejected():
    with pytest.raises(ValueError):
        SlidingFrame(translation_length=-1.0)
    with pytest.raises(ValueError):
        SlidingFrame(contraction_length=-1.0)


def test_translation_moves_the_rim_and_leaves_the_far_interface_alone():
    velocity = np.array([0.068, 0.037])
    frame = SlidingFrame(t0=14.3, centre=CENTRE, velocity=tuple(velocity),
                         translation_length=3e-3)
    tau = 0.01
    near = parabolic_tip(R)
    moved = tangential_target(near, 14.3 + tau, frame) - near
    # The tip reaches 8 R from the rim, where the window is 1 to (8 R / L_T)^2.
    assert np.allclose(moved, velocity * tau, rtol=1e-5, atol=0)
    far = np.array([[CENTRE[0] + 0.05, CENTRE[1] - 0.03]])
    assert np.all(np.abs(tangential_target(far, 14.3 + tau, frame) - far) < 1e-30)


def test_a_translating_shrinking_tip_keeps_its_nodes_per_radius():
    """Nodes of the tip at t0 land on the tip at t, at the same xi/R."""
    rate, tau = 25.0, 0.012
    velocity = (0.068, 0.037)
    frame = SlidingFrame(t0=0.0, centre=CENTRE, velocity=velocity, contraction_rate=rate,
                         translation_length=5e-3, contraction_length=1e3 * R)
    shrink = np.exp(-rate * tau)
    before = parabolic_tip(R)
    after = tangential_target(before, tau, frame)
    expected_centre = np.asarray(CENTRE) + np.asarray(velocity) * tau
    expected = parabolic_tip(shrink * R, centre=tuple(expected_centre))
    # The windows are 1 only to (8 R / L)^2 at the flank ends: the translation
    # of 9.3e-4 then misplaces them by 2.4e-9, the contraction by 1e-10.
    assert np.max(np.hypot(*(after - expected).T)) < 5e-3 * shrink * R
    spacing_before = np.median(np.hypot(*np.diff(before, axis=0).T)) / R
    spacing_after = np.median(np.hypot(*np.diff(after, axis=0).T)) / (shrink * R)
    assert spacing_after == pytest.approx(spacing_before, rel=1e-3)


def test_contraction_without_translation_is_a_dilation_about_the_centre():
    rate, tau = 10.0, 0.03
    frame = SlidingFrame(t0=0.0, centre=CENTRE, contraction_rate=rate, contraction_length=1.0)
    pts = np.asarray(CENTRE) + np.array([[1e-7, 2e-7], [-3e-7, 1e-7]])
    out = tangential_target(pts, tau, frame)
    assert np.allclose(out - CENTRE, np.exp(-rate * tau) * (pts - CENTRE), rtol=1e-9, atol=0)


def test_the_scalar_path_matches_the_vector_path():
    frame = SlidingFrame(t0=14.3, centre=CENTRE, velocity=(0.07, -0.02), contraction_rate=20.0,
                         translation_length=2e-3, contraction_length=4e-5)
    rng = np.random.default_rng(3)
    pts = np.asarray(CENTRE) + rng.normal(scale=3e-5, size=(50, 2))
    vec = tangential_target(pts, 14.31, frame)
    sca = np.array([target_point(x, y, 14.31, frame) for x, y in pts])
    assert np.allclose(vec, sca, rtol=0, atol=1e-16)


def test_relock_is_continuous_in_time():
    """Re-setting the frame changes the rates, never the map at that instant."""
    frame = SlidingFrame(t0=14.3, centre=CENTRE, velocity=(0.07, 0.04), contraction_rate=20.0,
                         translation_length=3e-3, contraction_length=4e-5)
    pts = parabolic_tip(R)
    t = 14.3004
    new = relock(frame, t, rim=(CENTRE[0] + 3e-5, CENTRE[1] + 1e-5), radius=0.99 * R,
                 reference_radius=R, velocity=(0.08, 0.03), rate=30.0, horizon=1e-4)
    assert np.allclose(tangential_target(pts, t, new), tangential_target(pts, t, frame),
                       rtol=0, atol=1e-18)
    assert new.velocity != frame.velocity and new.contraction_rate != frame.contraction_rate


def test_relock_steers_a_mispredicted_frame_onto_the_rim():
    """A rim whose speed is 10% off the first guess is caught within a few steps.

    The rim moves at V_true and shrinks at lambda_true. Without feedback the
    frame's rim image drifts by 0.1 V_true t, i.e. ~100 R over this march at
    R = 1e-6; with a re-lock every step it stays within a fraction of R.
    """
    v_true, lam_true = np.array([0.074, 0.045]), 25.0
    dt, horizon, n = 5e-5, 1.5e-4, 40
    frame = SlidingFrame(t0=0.0, centre=CENTRE, velocity=tuple(0.9 * v_true),
                         contraction_rate=0.8 * lam_true, translation_length=3e-3,
                         contraction_length=40 * R)
    measured_v, measured_lam = 0.9 * v_true, 0.8 * lam_true
    for j in range(1, n + 1):
        t = j * dt
        rim = np.asarray(CENTRE) + v_true * t
        radius = R * np.exp(-lam_true * t)
        frame = relock(frame, t, rim=rim, radius=radius, reference_radius=R,
                       velocity=measured_v, rate=measured_lam, horizon=horizon)
        measured_v, measured_lam = v_true, lam_true
    image = np.asarray(CENTRE) + frame.translation(n * dt)
    assert np.hypot(*(image - (np.asarray(CENTRE) + v_true * n * dt))) < 1e-3 * R
    assert frame.log_scale(n * dt) == pytest.approx(lam_true * n * dt, rel=1e-3)


def test_rim_motion_is_the_finite_difference_of_position_and_log_radius():
    velocity, rate = rim_motion(14.0, (0.1, 0.02), 4e-6, 14.01, (0.1007, 0.0204), 3e-6)
    assert np.allclose(velocity, [0.07, 0.04])
    assert rate == pytest.approx(np.log(4.0 / 3.0) / 0.01)
    with pytest.raises(ValueError):
        rim_motion(1.0, (0, 0), 1e-6, 1.0, (0, 0), 1e-6)
    with pytest.raises(ValueError):
        rim_motion(1.0, (0, 0), 0.0, 2.0, (0, 0), 1e-6)
