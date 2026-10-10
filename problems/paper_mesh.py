"""Conforming liquid and gas meshes for a bubble in a circular tube.

Both domains use the same named interface curve. The outer port names denote
their geometric locations, independently of the sign of the imposed flow.
"""

from __future__ import annotations

import math

import numpy as np
from pyoomph.meshes.gmsh import GmshTemplate

from .paper_bubble import MIN_SEED_CYLINDER_OVER_R0

LIQUID_BOUNDARIES = ("lower", "wall", "upper", "axis_front", "interface", "axis_rear")
GAS_BOUNDARIES = ("interface", "gas_axis")
TUBE_BOUNDARIES = ("lower", "wall", "upper", "axis")


def positive_wall_gap(polyline, tube_radius):
    """Smallest current wall gap, refusing contact or invalid geometry."""
    gaps = [tube_radius - float(point[0]) for point in polyline]
    if not gaps or any(not math.isfinite(gap) or gap <= 0 for gap in gaps):
        raise ValueError("every live interface wall gap must be finite and positive")
    return min(gaps)


class PaperTwoPhaseMesh(GmshTemplate):
    """A tube minus a capsule, with the capsule interior meshed as gas."""

    def define_geometry(self):
        problem = self.get_problem()
        geom = problem.geometry
        self.mesh_mode = "tris"
        self.default_resolution = problem.bulk_resolution
        if self.is_remeshing():
            segments = self.get_boundary_coordinates("liquid/interface", sort_along_axis="y+")
            if len(segments) != 1:
                raise ValueError("the paper model requires one connected closed bubble interface")
            polyline = [(float(r), float(z)) for r, z in segments[0]]
        else:
            polyline = geom.interface_polyline()
        if len(polyline) < 3 or any(not math.isfinite(x) for point in polyline for x in point):
            raise ValueError("the interface coordinates must be finite and connected")
        if abs(polyline[0][0]) > 1e-10 or abs(polyline[-1][0]) > 1e-10:
            raise ValueError("both bubble tips must lie on the axis")
        self._grading_points = np.asarray(polyline, dtype=float)
        gap = positive_wall_gap(polyline, geom.r_tube) if self.is_remeshing() else geom.seed_gap
        self._interface_size = min(gap / problem.film_elements, self.default_resolution)
        self._grading_length = max(4 * gap, 3 * self.default_resolution)
        self._mesh_size_callback = self._graded_size
        self.gmsh_options["Mesh.MeshSizeMin"] = 0.5 * self._interface_size
        self.gmsh_options["Mesh.MeshSizeMax"] = 2 * self.default_resolution

        lower_axis = self.point(0, 0)
        lower_wall = self.point(geom.r_tube, 0)
        upper_wall = self.point(geom.r_tube, geom.l_domain)
        upper_axis = self.point(0, geom.l_domain)
        rear = self.point(*polyline[0])
        front = self.point(*polyline[-1])
        self.line(lower_axis, lower_wall, name="lower")
        self.line(lower_wall, upper_wall, name="wall")
        self.line(upper_wall, upper_axis, name="upper")
        self.line(upper_axis, front, name="axis_front")
        self._interface(polyline, rear, front, geom)
        self.line(rear, lower_axis, name="axis_rear")
        self.plane_surface(*LIQUID_BOUNDARIES, name="liquid")
        self.line(rear, front, name="gas_axis")
        self.plane_surface(*GAS_BOUNDARIES, name="gas")

    def _interface(self, polyline, rear, front, geom):
        if self.is_remeshing():
            interior = [self.point(r, z) for r, z in polyline[1:-1]]
            self.spline([front, *interior[::-1], rear], name="interface")
            return
        front_shoulder = self.point(geom.r_body, geom.front_centre_z)
        rear_shoulder = self.point(geom.r_body, geom.rear_centre_z)
        if 0 < geom.cylinder_length <= MIN_SEED_CYLINDER_OVER_R0 * geom.r0:
            raise ValueError("seed cylinder must be zero or longer than 1e-12 R0")
        self.circle_arc(front, front_shoulder, center=(0, geom.front_centre_z), name="interface")
        if geom.cylinder_length > 0:
            self.line(front_shoulder, rear_shoulder, name="interface")
        self.circle_arc(rear_shoulder, rear, center=(0, geom.rear_centre_z), name="interface")

    def _graded_size(self, dim, tag, x, y, z, *_ignored):
        distance = float(np.min(np.hypot(self._grading_points[:, 0] - x,
                                         self._grading_points[:, 1] - y)))
        fraction = min(1, distance / self._grading_length)
        return self._interface_size + (self.default_resolution - self._interface_size) * fraction


class PaperTubeMesh(GmshTemplate):
    """The straight liquid-only tube used to verify the mixed physical ports."""

    def define_geometry(self):
        problem = self.get_problem()
        geom = problem.geometry
        self.mesh_mode = "tris"
        self.default_resolution = problem.bulk_resolution
        lower_axis = self.point(0, 0)
        lower_wall = self.point(geom.r_tube, 0)
        upper_wall = self.point(geom.r_tube, geom.l_domain)
        upper_axis = self.point(0, geom.l_domain)
        self.line(lower_axis, lower_wall, name="lower")
        self.line(lower_wall, upper_wall, name="wall")
        self.line(upper_wall, upper_axis, name="upper")
        self.line(upper_axis, lower_axis, name="axis")
        self.plane_surface(*TUBE_BOUNDARIES, name="liquid")
