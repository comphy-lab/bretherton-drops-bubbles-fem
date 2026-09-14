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

        self.gmsh_options["Mesh.MeshSizeMin"] = 0.5 * self._film_size
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

    def _graded_size(self, dim, tag, x, y, z, *_ignored_lc):
        """Element size at ``(x, y)``, graded by distance to the interface.

        gmsh calls this with either five or six positional arguments depending on
        version; the trailing catch-all absorbs the optional ``lc``.

        The film is the thinnest length in the problem, so it sets the floor. The
        grading length is deliberately several film thicknesses, so that the
        elements spanning the gap between the interface and the wall are all at
        the fine size rather than only those hugging the interface.
        """
        pts = self._grading_points
        d = float(np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))
        frac = min(1.0, d / self._grading_length)
        return self._film_size + (self.default_resolution - self._film_size) * frac


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
