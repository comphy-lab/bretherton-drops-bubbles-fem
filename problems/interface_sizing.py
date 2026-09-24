"""Target element size along the interface polyline.

This is pure geometry on an ``(n, 2)`` array of ``(z, r)`` points: no pyoomph,
no mesh object, no problem. It lives apart from :mod:`problems.mesh` for two
reasons. It is the part of the meshing that is worth unit-testing on synthetic
interfaces with a known answer, and importing pyoomph is neither cheap nor
side-effect-free (it pulls in hwloc, whose ``gl`` component opens an X
connection and blocks on a headless host).

Three lengths compete for each node, and the smallest wins:

``film_size``
    the distance grading's own length, uniform along the interface.
``lip_elements``
    elements per local radius of *curvature*, for the re-entrant rear lip.
``gap_elements``
    elements across a *gap* between two sheets of the same interface, for a
    folded rear cap whose skirt is thinning towards self-contact. A skirt is
    invisible to the curvature rule: both of its faces can be perfectly smooth
    while the liquid between them is a hundred times thinner than either radius
    of curvature.

The tube wall is not part of the interface chain, so the deposited Bretherton
film is untouched by the gap rule and keeps its ``film_elements`` grading.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "chain_arclength",
    "facing_pairs",
    "interface_gap",
    "interface_point_sizes",
]


def chain_arclength(points) -> np.ndarray:
    """Cumulative chord length along a polyline."""
    pts = np.asarray(points, dtype=float)
    return np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])


def facing_pairs(points, s, *, min_chain: float = 0.05, max_range: float = 0.5,
                 chain_ratio: float = 4.0):
    """Index pairs ``(i, j)`` facing each other across a gap, and the gaps ``d``.

    A pair qualifies when it is close in space and *far apart along the chain*.
    Both halves are needed, and the second is where the naive version goes
    wrong.

    Requiring only ``|s_i - s_j| >= min_chain`` is not enough. On a straight
    stretch of interface the Euclidean distance equals the arclength, so the
    closest qualifying pair is exactly ``min_chain`` apart and the rule would
    demand ``min_chain / gap_elements`` over the *whole* interface: a uniform
    refinement of film and caps alike, for no reason. What distinguishes a fold
    from a straight sheet is the ratio of the two. For a fold the chain has to
    travel out and back, so ``|s_i - s_j|`` is many times the gap; for a
    straight sheet the ratio is 1. A circular arc of half-angle ``x`` has ratio
    ``x / sin x``, which only reaches 4 once the arc has wrapped 281 degrees, so
    ``chain_ratio = 4`` also leaves the rear lip's own U-turn to
    ``lip_elements``, the length scale that belongs to it. Measured on the
    ``Ca_in = 14`` state at ``t = 25``: the lip U-turn has a chain-to-gap ratio
    of 1.02, while the skirt, whose two sheets are 8.9e-3 apart across a chain
    separation of 0.54, has one of 60.

    ``max_range`` is the search radius. Nothing further apart than that can set
    an element size worth refining for, and bounding it keeps the neighbour
    query near-linear on a long interface.
    """
    pts = np.asarray(points, dtype=float)
    empty = (np.empty(0, int), np.empty(0, int), np.empty(0, float))
    if len(pts) < 3:
        return empty
    pairs = cKDTree(pts).query_pairs(r=float(max_range), output_type="ndarray")
    if len(pairs) == 0:
        return empty
    i, j = pairs[:, 0], pairs[:, 1]
    chain = np.abs(s[i] - s[j])
    d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
    keep = (chain >= float(min_chain)) & (chain >= float(chain_ratio) * d)
    return i[keep], j[keep], d[keep]


def interface_gap(points, *, min_chain: float = 0.05, max_range: float = 0.5,
                  chain_ratio: float = 4.0):
    """Closest approach between two parts of one interface chain.

    Returns ``(gap, i, j)``: the smallest gap over the pairs :func:`facing_pairs`
    admits, and the two indices realising it. ``gap`` is ``inf`` when the
    interface has no fold, which is the normal state of a Bretherton bubble.

    One definition, used both by the mesh sizing and by run drivers reporting
    the gap; two implementations would drift apart.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return float("inf"), -1, -1
    s = chain_arclength(pts)
    i, j, d = facing_pairs(pts, s, min_chain=min_chain, max_range=max_range,
                           chain_ratio=chain_ratio)
    if len(i) == 0:
        return float("inf"), -1, -1
    k = int(np.argmin(d))
    return float(d[k]), int(i[k]), int(j[k])


def _curvature_sizes(size, z, r, s, *, lip_elements, lip_patch_radii):
    """Elements per local radius of curvature, held over a patch around the lip.

    Distance grading alone sets one length on the interface, the film size, and
    that is the wrong length wherever the interface is more sharply curved than
    the film is thin. At high ``Ca`` the rear cap of a confined bubble becomes
    re-entrant and the liquid tongue turns through a lip whose radius of
    curvature is *smaller* than the deposited film thickness: measured on the
    frozen ``N_film = 6`` mesh, ``R_lip`` is 0.023 to 0.030 while
    ``film_element_size`` is 0.036, so the feature that carries the whole
    rear-cap force balance was spanned by about one element, and by
    ``Ca_in = 14`` the nodes there had stretched to 0.098, i.e. ``R_lip/ds`` of
    0.31. That under-resolution is invisible in ``h/R0`` and in ``Ca_b``,
    because both are set by the flat film, which is why it survived a window
    and bulk-resolution convergence study.

    Curvature is taken from the live polyline by finite differences on its own
    arclength, so it follows the solution rather than the seeded capsule,
    exactly as the distance grading does.

    The refined patch must be wider than the lip. Sizes are attached to nodes
    and relaxed by distance to the *nearest* node, so refining only the few
    nodes at the corner gives a patch a few radii wide, and the corner slides
    along the interface by tens of radii per continuation station at high
    ``Ca``: measured on lip8_adapt near ``Ca_in = 14.5``, the pre-remesh
    ``R_lip/ds`` fell from ~15 to ~6 within 0.03 in ``Ca_in``, and the plain
    resolve failed once it was below ~6. Hold the lip size over
    ``lip_patch_radii`` radii of arclength either side of the sharpest node.
    """
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
    return size


def _proximity_sizes(size, pts, s, *, gap_elements, gap_min_chain, gap_max_range,
                     gap_chain_ratio):
    """Lower the target size on both sheets of a narrow interface gap.

    Every pair :func:`facing_pairs` admits is treated as facing across a liquid
    gap of their separation ``d``, and *both* points are asked for
    ``d / gap_elements``. Doing both is the point: a one-sided version refines
    the concave sheet and leaves the convex one coarse, and the elements in the
    gap are then as coarse as the coarser side allows.

    Returns the updated sizes and the smallest gap seen, which is what a run
    driver wants to log.
    """
    target = np.full(len(pts), np.inf)
    i, j, d = facing_pairs(pts, s, min_chain=gap_min_chain, max_range=gap_max_range,
                           chain_ratio=gap_chain_ratio)
    min_gap = float("inf")
    if len(i):
        min_gap = float(d.min())
        want = d / float(gap_elements)
        np.minimum.at(target, i, want)
        np.minimum.at(target, j, want)
    return np.minimum(size, target), min_gap


def _grade_along_chain(size, s, rate):
    """``min_j(size_j + rate*|s_i - s_j|)`` for strictly increasing ``s``, in O(n).

    This is the 1-D distance transform, done in one forward and one backward
    sweep instead of an ``n x n`` table. The forward sweep carries the best
    candidate from ``j <= i`` and the backward sweep the best from ``j >= i``.
    Each step adds ``rate`` times one arclength increment, so rounding stays
    relative to the sizes themselves; the ``size_j - rate*s_j`` rearrangement
    would instead cancel a sub-micron lip size against ``rate*s ~ O(1)``.
    """
    out = [float(v) for v in size]
    step = (rate * np.diff(s)).tolist()
    for k in range(1, len(out)):
        out[k] = min(out[k], out[k - 1] + step[k - 1])
    for k in range(len(out) - 2, -1, -1):
        out[k] = min(out[k], out[k + 1] + step[k])
    return np.array(out, dtype=float)


def interface_point_sizes(points, *, film_size: float,
                          lip_elements: float | None = None,
                          lip_floor_divisor: float = 16.0,
                          lip_patch_radii: float = 10.0,
                          lip_gradation: float = 0.25,
                          gap_elements: float | None = None,
                          gap_min_chain: float = 0.05,
                          gap_max_range: float = 0.5,
                          gap_chain_ratio: float = 4.0,
                          with_min_gap: bool = False):
    """Target size at each interface node: the film size, floored by the features.

    With neither ``lip_elements`` nor ``gap_elements`` this is the uniform film
    size, bit for bit what every result before the curvature grading used.

    Both floors are applied before the gradation, so whichever length is smaller
    at a node is the one that mesh keeps; then the size is allowed to grow at
    most ``lip_gradation`` per unit arclength away from any refined node, and is
    finally clamped from below at ``film_size / lip_floor_divisor``. That floor
    is what stops one noisy node dictating a mesh the solver cannot pay for; it
    is a knob because a feature that keeps sharpening with ``Ca`` will otherwise
    hit it, and whether a measured radius is physical or floor-limited is
    decided by lowering it.
    """
    pts = np.asarray(points, dtype=float)
    size = np.full(len(pts), film_size, dtype=float)
    min_gap = float("inf")
    if (not lip_elements and not gap_elements) or len(pts) < 5:
        return (size, min_gap) if with_min_gap else size

    z, r = pts[:, 0], pts[:, 1]
    s = chain_arclength(pts)
    # A duplicated node would make the arclength non-monotone and the gradients
    # infinite; drop that case rather than emit NaN sizes.
    if not np.all(np.diff(s) > 0):
        return (size, min_gap) if with_min_gap else size

    if lip_elements:
        size = _curvature_sizes(size, z, r, s, lip_elements=lip_elements,
                                lip_patch_radii=lip_patch_radii)
    if gap_elements:
        size, min_gap = _proximity_sizes(
            size, pts, s, gap_elements=gap_elements, gap_min_chain=gap_min_chain,
            gap_max_range=gap_max_range, gap_chain_ratio=gap_chain_ratio)

    if lip_gradation > 0:
        size = _grade_along_chain(size, s, float(lip_gradation))

    size = np.maximum(size, film_size / lip_floor_divisor)
    return (size, min_gap) if with_min_gap else size
