"""Meridional mesh for the liquid around a void bubble in a tube.

The meshed region is the tube rectangle minus the bubble. The bubble interior is
not meshed at all, so the liquid/void interface is a *boundary* of the mesh and
its shape is carried by mesh position degrees of freedom rather than by a
captured field.

Two things here are load-bearing and easy to get wrong.

**Boundary names must be identical in both branches of** ``define_geometry``.
The remesh branch rebuilds the geometry from the deformed interface; pyoomph
then interpolates the old solution onto the new mesh boundary by boundary, and
raises if a name present on the old mesh is absent from the new one. The names
are listed once in :data:`BOUNDARY_NAMES` and used in both branches.

**The size callback must grade on the live interface, not on the seeded
capsule.** After a remesh the interface is a Bretherton bubble with a thin film
and a rear capillary ripple, not the capsule it started as. Grading on a stored
analytic capsule would coarsen exactly where the solution has developed
structure. The callback therefore grades on distance to whatever polyline the
current ``define_geometry`` call used.
"""

from __future__ import annotations

import numpy as np
from pyoomph.meshes.gmsh import GmshTemplate

from .geometry import CapsuleGeometry

BOUNDARY_NAMES = ("inlet", "wall", "outlet", "axis_front", "interface", "axis_rear")
"""Every boundary, in both branches. Interpolation across a remesh depends on this."""


class LiquidAnnulusMesh(GmshTemplate):
    """Liquid region: tube rectangle minus the void, graded towards the film."""

    def define_geometry(self):
        problem = self.get_problem()
        geom: CapsuleGeometry = problem.geometry
        self.mesh_mode = "tris"
        self.default_resolution = problem.bulk_resolution

        if self.is_remeshing():
            segments = self.get_boundary_coordinates("liquid/interface", sort_along_axis="y+")
            polyline = [(float(x), float(y)) for x, y in segments[0]]
        else:
            polyline = geom.interface_polyline()

        self._grading_points = np.asarray(polyline, dtype=float)
        self._film_size = problem.film_element_size()
        self._grading_length = problem.mesh_grading_length()
        self._mesh_size_callback = self._graded_size

        # Curvature-adaptive floor on the interface. ``lip_elements`` is the
        # number of elements demanded across one local radius of curvature; None
        # restores the pure distance grading that every earlier result used.
        self._lip_elements = getattr(problem, "lip_elements", None)
        self._point_size = self._interface_point_sizes()

        floor = float(np.min(self._point_size))
        self.gmsh_options["Mesh.MeshSizeMin"] = 0.5 * floor
        self.gmsh_options["Mesh.MeshSizeMax"] = 2.0 * problem.bulk_resolution

        # The window corners. The outer boundary is rigid in the co-moving frame,
        # so these are fixed points, unlike the interface.
        p_in_axis = self.point(0.0, 0.0)
        p_in_wall = self.point(geom.r_tube, 0.0)
        p_out_wall = self.point(geom.r_tube, geom.l_domain)
        p_out_axis = self.point(0.0, geom.l_domain)

        rear = polyline[0]
        front = polyline[-1]
        p_rear = self.point(rear[0], rear[1])
        p_front = self.point(front[0], front[1])

        self.line(p_in_axis, p_in_wall, name="inlet")
        self.line(p_in_wall, p_out_wall, name="wall")
        self.line(p_out_wall, p_out_axis, name="outlet")
        self.line(p_out_axis, p_front, name="axis_front")
        self._build_interface(polyline, p_rear, p_front, geom)
        self.line(p_rear, p_in_axis, name="axis_rear")
        self.plane_surface(*BOUNDARY_NAMES, name="liquid")

    def _build_interface(self, polyline, p_rear, p_front, geom: CapsuleGeometry) -> None:
        """Interface from the front tip round to the rear tip.

        The seeded capsule is drawn with exact arcs and a straight line, so the
        initial mesh carries no polygonal error on the caps. A deformed interface
        has no analytic description and is drawn as a spline through its own
        nodes.
        """
        if self.is_remeshing():
            interior = [self.point(x, y) for x, y in polyline[1:-1]]
            self.spline([p_front, *interior[::-1], p_rear], name="interface")
            return

        p_rear_shoulder = self.point(geom.r_body, geom.rear_centre_z)
        p_front_shoulder = self.point(geom.r_body, geom.front_centre_z)
        self.circle_arc(
            p_front, p_front_shoulder, center=(0.0, geom.front_centre_z), name="interface"
        )
        self.line(p_front_shoulder, p_rear_shoulder, name="interface")
        self.circle_arc(
            p_rear_shoulder, p_rear, center=(0.0, geom.rear_centre_z), name="interface"
        )

    def _interface_point_sizes(self) -> np.ndarray:
        """Target size at each interface node: the film size, floored by curvature.

        Distance grading alone sets one length on the interface, the film size,
        and that is the wrong length wherever the interface is more sharply
        curved than the film is thin. At high ``Ca`` the rear cap of a confined
        bubble becomes re-entrant and the liquid tongue turns through a lip whose
        radius of curvature is *smaller* than the deposited film thickness:
        measured on the frozen ``N_film = 6`` mesh, ``R_lip`` is 0.023 to 0.030
        while ``film_element_size`` is 0.036, so the feature that carries the
        whole rear-cap force balance was spanned by about one element, and by
        ``Ca_in = 14`` the nodes there had stretched to 0.098, i.e. ``R_lip/ds``
        of 0.31. That under-resolution is invisible in ``h/R0`` and in ``Ca_b``,
        because both are set by the flat film, which is why it survived a window
        and bulk-resolution convergence study.

        ``lip_elements`` demands that many elements per local radius of
        curvature. Curvature is taken from the live polyline by finite
        differences on its own arclength, so it follows the solution rather than
        the seeded capsule, exactly as the distance grading does.
        """
        pts = self._grading_points
        size = np.full(len(pts), self._film_size, dtype=float)
        if not self._lip_elements or len(pts) < 5:
            return size

        z, r = pts[:, 0], pts[:, 1]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(z), np.diff(r)))])
        # A duplicated node would make the arclength non-monotone and the
        # gradients infinite; drop that case rather than emit NaN sizes.
        if not np.all(np.diff(s) > 0):
            return size

        dz, dr = np.gradient(z, s), np.gradient(r, s)
        d2z, d2r = np.gradient(dz, s), np.gradient(dr, s)
        speed = (dz**2 + dr**2) ** 1.5
        curvature = np.abs(dz * d2r - dr * d2z) / np.maximum(speed, 1e-300)

        with np.errstate(divide="ignore"):
            radius = np.where(curvature > 0, 1.0 / curvature, np.inf)
        size = np.minimum(size, radius / float(self._lip_elements))

        # Never demand a size the solver cannot pay for, and never let one noisy
        # node dictate the whole mesh: the floor is a fixed fraction of the film.
        return np.maximum(size, self._film_size / 16.0)

    def _graded_size(self, dim, tag, x, y, z, *_ignored_lc):
        """Element size at ``(x, y)``, graded by distance to the interface.

        gmsh calls this with either five or six positional arguments depending on
        version; the trailing catch-all absorbs the optional ``lc``.

        The film is the thinnest length in the problem, so it sets the floor away
        from the caps. The grading length is deliberately several film
        thicknesses, so that the elements spanning the gap between the interface
        and the wall are all at the fine size rather than only those hugging the
        interface. Where ``lip_elements`` has lowered the target at a sharply
        curved node, that lower value is what this relaxes from, so a re-entrant
        lip keeps its own length scale over its own neighbourhood.
        """
        pts = self._grading_points
        distance = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        nearest = int(np.argmin(distance))
        d = float(distance[nearest])
        near_size = float(self._point_size[nearest])
        frac = min(1.0, d / self._grading_length)
        return near_size + (self.default_resolution - near_size) * frac


class PlainTubeMesh(GmshTemplate):
    """The same window with no bubble: a plain tube, for the Poiseuille check.

    Boundary names deliberately match :data:`BOUNDARY_NAMES` minus the interface,
    with the axis in one piece, so the verification problem exercises the same
    boundary-condition code paths as the real one.
    """

    def define_geometry(self):
        problem = self.get_problem()
        self.mesh_mode = "tris"
        self.default_resolution = problem.bulk_resolution
        r_tube, length = problem.r_tube, problem.l_domain

        p_in_axis = self.point(0.0, 0.0)
        p_in_wall = self.point(r_tube, 0.0)
        p_out_wall = self.point(r_tube, length)
        p_out_axis = self.point(0.0, length)

        self.line(p_in_axis, p_in_wall, name="inlet")
        self.line(p_in_wall, p_out_wall, name="wall")
        self.line(p_out_wall, p_out_axis, name="outlet")
        self.line(p_out_axis, p_in_axis, name="axis")
        self.plane_surface("inlet", "wall", "outlet", "axis", name="liquid")
