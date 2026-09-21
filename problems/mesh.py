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

**Curvature is not the only small length on the interface.** When the rear cap
folds into a skirt, two sheets of the *same* interface chain approach each
other. The liquid between them is a film whose thickness is nowhere written on
the local curvature, so the curvature grading happily leaves one element
spanning it. ``gap_elements`` demands a number of elements across that gap.

The size rules themselves live in :mod:`problems.interface_sizing`, which is
pure geometry and imports no pyoomph, so they can be unit-tested on synthetic
interfaces.
"""

from __future__ import annotations

import numpy as np
from pyoomph.meshes.gmsh import GmshTemplate

from .geometry import CapsuleGeometry
from .interface_sizing import interface_point_sizes

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
            polyline = self._regularize_spacing(
                [(float(x), float(y)) for x, y in segments[0]])
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
        self._lip_floor_divisor = float(getattr(problem, "lip_floor_divisor", 16.0))
        self._lip_patch_radii = float(getattr(problem, "lip_patch_radii", 10.0))
        self._lip_gradation = float(getattr(problem, "lip_gradation", 0.25))

        # Proximity floor: elements demanded across a gap between two sheets of
        # the interface. None (the default) leaves the sizes bit-for-bit as the
        # curvature grading alone produced them.
        self._gap_elements = getattr(problem, "gap_elements", None)
        self._gap_min_chain = float(getattr(problem, "gap_min_chain", 0.05))
        self._gap_max_range = float(getattr(problem, "gap_max_range", 0.5))
        self._gap_chain_ratio = float(getattr(problem, "gap_chain_ratio", 4.0))
        self._point_size, self._min_gap = self._interface_point_sizes()

        floor = float(np.min(self._point_size))
        self.gmsh_options["Mesh.MeshSizeMin"] = 0.5 * floor
        # gmsh's geometric tolerance is absolute (1e-8 by default); keep it well
        # below the smallest element the lip grading asks for.
        self.gmsh_options["Geometry.Tolerance"] = min(1e-8, 1e-3 * floor)
        self.gmsh_options["Mesh.MeshSizeMax"] = 2.0 * problem.bulk_resolution

        # The window corners. The outer boundary is rigid in the co-moving frame,
        # so these are fixed points, unlike the interface.
        p_in_axis = self.point(0.0, geom.z_inlet)
        p_in_wall = self.point(geom.r_tube, geom.z_inlet)
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

    @staticmethod
    def _regularize_spacing(polyline, max_ratio: float = 2.0):
        """Split long segments next to short ones until neighbours differ by at most ``max_ratio``.

        The remeshed interface is a Catmull-Rom spline through the old nodes
        with a uniform parameter per node. Where a refined lip patch (nodes
        ~1e-6 apart) meets the film (nodes ~1e-2 apart) the parametrization
        speed jumps by four orders of magnitude across one node, the spline
        wiggles there, and pyoomph's projection of gmsh's nodes back onto it
        fails to converge: ``Cannot invert spline`` at Ca_in = 15.625 with a
        minimum spacing of 1.06e-6. Inserting chord midpoints into the longer
        segment until the ratio is bounded keeps the parametrization smooth;
        the chords sit on the nearly straight film, where they are exact to
        the resolution that matters.
        """
        pts = [tuple(p) for p in polyline]
        changed = True
        guard = 0
        while changed and guard < 60:
            changed = False
            guard += 1
            out = [pts[0]]
            for i in range(1, len(pts)):
                a, b = pts[i - 1], pts[i]
                seg = np.hypot(b[0] - a[0], b[1] - a[1])
                prev = np.hypot(a[0] - out[-2][0], a[1] - out[-2][1]) if len(out) > 1 else None
                nxt = np.hypot(pts[i + 1][0] - b[0], pts[i + 1][1] - b[1]) if i + 1 < len(pts) else None
                short = min(v for v in (prev, nxt) if v is not None) if (prev or nxt) else seg
                if seg > max_ratio * short and seg > 0:
                    out.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
                    changed = True
                out.append(b)
            pts = out
        return pts

    def _interface_point_sizes(self):
        """Target size at each interface node, and the smallest gap measured.

        A thin adapter: everything it decides lives in
        :func:`problems.interface_sizing.interface_point_sizes`.
        """
        return interface_point_sizes(
            self._grading_points, film_size=self._film_size,
            lip_elements=self._lip_elements,
            lip_floor_divisor=self._lip_floor_divisor,
            lip_patch_radii=self._lip_patch_radii,
            lip_gradation=self._lip_gradation,
            gap_elements=self._gap_elements,
            gap_min_chain=self._gap_min_chain,
            gap_max_range=self._gap_max_range,
            gap_chain_ratio=self._gap_chain_ratio,
            with_min_gap=True)

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
        lip keeps its own length scale over its own neighbourhood, and the same
        holds for a node inside a thin skirt refined by ``gap_elements``.
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
