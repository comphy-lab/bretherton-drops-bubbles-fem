"""Interface element sizing: curvature grading and the proximity floor.

The sizing is pure geometry on the interface polyline, so it is tested on
synthetic polylines with a known answer rather than on a simulation, and
:mod:`problems.interface_sizing` imports no pyoomph so these run in
milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from problems.interface_sizing import (
    _grade_along_chain,
    chain_arclength,
    interface_gap,
    interface_point_sizes,
)

FILM_SIZE = 0.05
NO_FLOOR = 400_000.0
"""lip_floor_divisor large enough that the floor never binds in these tests."""


def sizes(points, **knobs):
    """``interface_point_sizes`` with this module's defaults filled in."""
    knobs.setdefault("film_size", FILM_SIZE)
    knobs.setdefault("lip_floor_divisor", NO_FLOOR)
    return interface_point_sizes(points, **knobs)


SHEET_LENGTH = 1.0
TAIL_LENGTH = 1.0
FAR = 0.7
"""Tail points beyond this are further than gap_max_range from either sheet."""


def skirt_polyline(gap: float = 0.01, n: int = 201):
    """Two nearly parallel sheets ``gap`` apart, joined by a cap, plus a tail.

    Sheet A runs ``+z`` at ``r = 0.5`` from ``z = 0`` to ``z = 1``; a
    semicircular cap of radius ``gap/2`` turns it round; sheet B runs back to
    ``z = 0`` at ``r = 0.5 + gap``; the tail then continues to ``z = -1``.

    Returns the points and two masks: the facing parts of the two sheets, and
    the far end of the tail. The tail is the control - it is a single sheet
    with nothing within ``gap_max_range`` of it, so the proximity floor must
    leave it at the film size.
    """
    zA = np.linspace(0.0, SHEET_LENGTH, n)
    rA = np.full(n, 0.5)
    ang = np.linspace(-np.pi / 2, np.pi / 2, 21)[1:-1]
    zC = SHEET_LENGTH + (gap / 2) * np.cos(ang)
    rC = 0.5 + gap / 2 + (gap / 2) * np.sin(ang)
    zB = zA[::-1]
    rB = np.full(n, 0.5 + gap)
    zT = np.linspace(0.0, -TAIL_LENGTH, n)[1:]
    rT = np.full(len(zT), 0.5 + gap)

    z = np.concatenate([zA, zC, zB, zT])
    r = np.concatenate([rA, rC, rB, rT])
    # Away from the cap, so that the chain separation across the gap clears
    # gap_min_chain; near the cap the two sheets are close along the chain too
    # and the gradation, not the floor itself, is what refines them.
    on_sheet = (np.abs(r - 0.5) < 1.5 * gap) & (z > 0.05) & (z < 0.9 * SHEET_LENGTH)
    in_tail = z < -FAR
    return np.stack([z, r], axis=1), on_sheet, in_tail


def test_interface_gap_measures_the_sheet_separation():
    points, _, _ = skirt_polyline(gap=0.01)
    gap, i, j = interface_gap(points)
    assert gap == pytest.approx(0.01, rel=1e-6)
    assert abs(points[i, 1] - points[j, 1]) == pytest.approx(0.01, rel=1e-6)
    # The pair is a genuine fold: the chain has to travel out and back.
    s = chain_arclength(points)
    assert abs(s[i] - s[j]) > 20 * gap


def test_a_straight_sheet_has_no_gap():
    """Without the chain-to-gap ratio this would return ``gap_min_chain``."""
    z = np.linspace(0.0, 1.0, 201)
    points = np.stack([z, np.full_like(z, 0.5)], axis=1)
    assert interface_gap(points) == (float("inf"), -1, -1)


def test_a_smooth_u_turn_is_left_to_the_curvature_grading():
    """A half-circle of radius 0.05 is a lip, not a fold.

    Its two ends are 0.1 apart with a chain separation of only 0.157, a ratio
    of 1.57. ``lip_elements`` owns that length scale; the proximity floor must
    not also claim it, or the two would compound.
    """
    ang = np.linspace(-np.pi / 2, np.pi / 2, 201)
    points = np.stack([0.05 * np.cos(ang), 0.5 + 0.05 * np.sin(ang)], axis=1)
    assert interface_gap(points, min_chain=0.01) == (float("inf"), -1, -1)


def test_proximity_sizing_resolves_a_thin_skirt_and_leaves_the_rest_alone():
    points, on_sheet, in_tail = skirt_polyline(gap=0.01)
    plain = sizes(points)
    refined = sizes(points, gap_elements=6.0)

    assert np.allclose(plain, FILM_SIZE)
    assert on_sheet.sum() > 100 and in_tail.sum() > 10
    assert refined[on_sheet].max() <= 0.01 / 6.0 + 1e-12
    assert np.allclose(refined[in_tail], FILM_SIZE)


def test_proximity_sizing_scales_with_the_gap():
    for gap in (0.02, 0.004):
        points, on_sheet, in_tail = skirt_polyline(gap=gap)
        refined = sizes(points, gap_elements=6.0)
        assert refined[on_sheet].max() <= gap / 6.0 + 1e-12
        assert refined[on_sheet].min() >= 0.5 * gap / 6.0
        assert np.allclose(refined[in_tail], FILM_SIZE)


def test_gap_elements_sets_the_number_of_elements_across_the_gap():
    points, on_sheet, _ = skirt_polyline(gap=0.01)
    for n in (2.0, 6.0, 12.0):
        got = sizes(points, gap_elements=n)
        assert got[on_sheet].max() == pytest.approx(0.01 / n, rel=1e-9)


def test_a_single_sheet_near_the_wall_is_untouched():
    """The tube wall is not on the interface chain, so the film is not refined.

    A Bretherton film is a single interface sheet ~0.01 from the wall. If the
    proximity floor saw the wall it would demand 0.01/6 across the deposited
    film everywhere, which is a different and far more expensive mesh.
    """
    z = np.linspace(0.0, 8.0, 801)
    points = np.stack([z, np.full_like(z, 0.69)], axis=1)
    assert np.allclose(sizes(points, gap_elements=6.0), FILM_SIZE)


def test_the_floor_still_bounds_the_proximity_request():
    """gap_elements may not ask for a size below film_size/lip_floor_divisor."""
    points, on_sheet, _ = skirt_polyline(gap=0.01)
    got = sizes(points, gap_elements=6.0, lip_floor_divisor=16.0)
    assert got.min() >= FILM_SIZE / 16.0 - 1e-15
    assert got[on_sheet].min() == pytest.approx(FILM_SIZE / 16.0)


# --------------------------------------------------------------- regression lock

def legacy_point_sizes(points, *, film_size, lip_elements, lip_floor_divisor,
                       lip_patch_radii, lip_gradation):
    """The sizing exactly as it stood before the proximity floor was added.

    Kept verbatim, including the dense ``n x n`` gradation that the module now
    does in two linear sweeps, so that ``gap_elements=None`` can be shown to
    reproduce every earlier mesh to rounding rather than merely closely.
    """
    pts = np.asarray(points, dtype=float)
    size = np.full(len(pts), film_size, dtype=float)
    if not lip_elements or len(pts) < 5:
        return size
    z, r = pts[:, 0], pts[:, 1]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(z), np.diff(r)))])
    if not np.all(np.diff(s) > 0):
        return size
    dz, dr = np.gradient(z, s), np.gradient(r, s)
    d2z, d2r = np.gradient(dz, s), np.gradient(dr, s)
    speed = (dz**2 + dr**2) ** 1.5
    curvature = np.abs(dz * d2r - dr * d2z) / np.maximum(speed, 1e-300)
    with np.errstate(divide="ignore"):
        radius = np.where(curvature > 0, 1.0 / curvature, np.inf)
    size = np.minimum(size, radius / float(lip_elements))
    sharpest = int(np.argmin(size))
    if np.isfinite(radius[sharpest]) and lip_patch_radii > 0:
        half_width = lip_patch_radii * float(radius[sharpest])
        inside = np.abs(s - s[sharpest]) <= half_width
        size[inside] = np.minimum(size[inside], size[sharpest])
    if lip_gradation > 0:
        size = np.min(size[None, :] + lip_gradation * np.abs(s[:, None] - s[None, :]), axis=1)
    return np.maximum(size, film_size / lip_floor_divisor)


@pytest.mark.parametrize("lip_elements", [None, 8.0, 16.0])
@pytest.mark.parametrize("floor_divisor", [16.0, NO_FLOOR])
def test_gap_elements_none_reproduces_the_previous_sizes_exactly(lip_elements, floor_divisor):
    points, _, _ = skirt_polyline(gap=0.01)
    got = sizes(points, lip_elements=lip_elements,
                lip_floor_divisor=floor_divisor, gap_elements=None)
    want = legacy_point_sizes(points, film_size=FILM_SIZE, lip_elements=lip_elements,
                              lip_floor_divisor=floor_divisor, lip_patch_radii=10.0,
                              lip_gradation=0.25)
    assert np.allclose(got, want, rtol=1e-14, atol=0.0)


def dense_gradation(size, s, rate):
    """The O(n^2) reference: ``min_j(size_j + rate*|s_i - s_j|)`` by brute force."""
    return np.min(size[None, :] + rate * np.abs(s[:, None] - s[None, :]), axis=1)


def test_linear_gradation_matches_the_dense_distance_transform():
    """The two-sweep gradation is the dense minimum, on chains unlike the skirt.

    Uneven spacing from 1e-8 to 1 and sizes from 1e-7 to 1e-1 at arclengths up
    to ~20 are the conditions of a resolved lip far from the chain origin, where
    a cancelling rearrangement would lose digits.
    """
    rng = np.random.default_rng(20260924)
    for _ in range(50):
        n = int(rng.integers(2, 400))
        s = rng.uniform(0.0, 15.0) + np.cumsum(10.0 ** rng.uniform(-8.0, 0.0, n))
        size = 10.0 ** rng.uniform(-7.0, -1.0, n)
        rate = float(rng.uniform(0.01, 2.0))
        got = _grade_along_chain(size, s, rate)
        assert np.allclose(got, dense_gradation(size, s, rate), rtol=1e-14, atol=0.0)
