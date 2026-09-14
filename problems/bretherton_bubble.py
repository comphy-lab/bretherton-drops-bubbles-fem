"""Axisymmetric void bubble translating in a liquid-filled circular tube.

The liquid is meshed; the bubble interior is a void. Surface tension acts on
the liquid/void interface through ``NavierStokesFreeSurface``. There is no
interior gas stress and no second-fluid volume fraction.

Nondimensionalisation matches the Basilisk companion: ``sigma = mu = R0 = 1``,
so the imposed inlet capillary number is the mean inlet speed and
``La = rho``. The calculation sits in a frame co-moving with the bubble. The
unknown bubble speed ``U`` is a global degree of freedom; the rear tip is
pinned. The wall and inlet velocities therefore depend on ``U`` and must use
``EnforcedBC``, not ``DirichletBC``.

pyoomph pin: ``comphy-2026.09.07.1``
(``a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618``).
"""

from __future__ import annotations

import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

from pyoomph import *  # noqa: E402
from pyoomph.equations.ALE import HyperelasticSmoothedMesh  # noqa: E402
from pyoomph.equations.generic import AxisymmetryBC  # noqa: E402
from pyoomph.equations.navier_stokes import NavierStokesEquations, NavierStokesFreeSurface  # noqa: E402
from pyoomph.expressions import *  # noqa: E402
from pyoomph.meshes.gmsh import GmshTemplate  # noqa: E402
from pyoomph.meshes.zeta import (  # noqa: E402
    AssignZetaCoordinatesByArclength,
    AssignZetaCoordinatesByEulerianCoordinate,
)

from .geometry import CapsuleGeometry, geometry_from_case


class LiquidAnnulusMesh(GmshTemplate):
    """Meridional liquid region: tube rectangle minus the capsule void."""

    def define_geometry(self):
        self.mesh_mode = "tris"
        problem = self.get_problem()
        assert isinstance(problem, VoidBubbleProblem)
        geom = problem.geometry
        size = problem.mesh_size
        self.gmsh_options["Mesh.MeshSizeMin"] = 0.25 * size
        self.gmsh_options["Mesh.MeshSizeMax"] = 4.0 * size
        self.gmsh_options["Mesh.MeshSizeFromCurvature"] = 12

        p_in_axis = self.point(0, 0)
        p_in_wall = self.point(geom.r_tube, 0)
        p_out_wall = self.point(geom.r_tube, geom.l_domain)
        p_out_axis = self.point(0, geom.l_domain)
        p_rear = self.point(0, geom.z_rear)
        p_front = self.point(0, geom.z_front)
        p_rear_shoulder = self.point(geom.r_body, geom.rear_centre_z)
        p_front_shoulder = self.point(geom.r_body, geom.front_centre_z)
        p_rear_centre = self.point(0, geom.rear_centre_z)
        p_front_centre = self.point(0, geom.front_centre_z)

        self.line(p_in_axis, p_in_wall, name="inlet")
        self.line(p_in_wall, p_out_wall, name="wall")
        self.line(p_out_wall, p_out_axis, name="outlet")
        self.line(p_out_axis, p_front, name="axis_front")
        self.circle_arc(
            p_front,
            p_front_shoulder,
            center=p_front_centre,
            name="interface",
        )
        self.line(p_front_shoulder, p_rear_shoulder, name="interface")
        self.circle_arc(
            p_rear_shoulder,
            p_rear,
            center=p_rear_centre,
            name="interface",
        )
        self.line(p_rear, p_in_axis, name="axis_rear")
        self.plane_surface(
            "inlet",
            "wall",
            "outlet",
            "axis_front",
            "interface",
            "axis_rear",
            name="liquid",
        )


class VoidBubbleProblem(Problem):
    """Co-moving void bubble in an axisymmetric capillary tube."""

    def __init__(self, *, geometry: CapsuleGeometry, ca_in: float, laplace: float, mesh_size: float):
        super().__init__()
        if ca_in <= 0 or laplace <= 0 or mesh_size <= 0:
            raise ValueError("Ca, La and mesh size must be positive")
        self.geometry = geometry
        self.ca_in = ca_in
        self.laplace = laplace
        self.mesh_size = mesh_size

    @classmethod
    def from_case(cls, case: dict) -> "VoidBubbleProblem":
        physics = case["physics"]
        return cls(
            geometry=geometry_from_case(physics),
            ca_in=float(physics["ca_in"]),
            laplace=float(physics["laplace"]),
            mesh_size=float(case["mesh"]["characteristic_size"]),
        )

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(LiquidAnnulusMesh())

        density = self.laplace
        viscosity = 1.0
        eqs = NavierStokesEquations(mass_density=density, dynamic_viscosity=viscosity)
        eqs += HyperelasticSmoothedMesh()
        eqs += MeshFileOutput()
        eqs += RemeshWhen(RemeshingOptions())

        # Bubble speed in the laboratory frame. The rear tip is pinned, so the
        # wall and inlet velocities depend on this unknown and cannot be Dirichlet.
        speed, speed_test = self.add_global_dof("U", initial_condition=self.ca_in)

        r = var("mesh_x")
        poiseuille = 2 * self.ca_in * (1 - (r / self.geometry.r_tube) ** 2)
        wall_axial = -speed
        inlet_axial = poiseuille - speed

        eqs += AxisymmetryBC() @ "axis_rear"
        eqs += AxisymmetryBC() @ "axis_front"
        eqs += DirichletBC(mesh_x=True, velocity_x=0) @ "wall"
        eqs += EnforcedBC(velocity_y=var("velocity_y") - wall_axial) @ "wall"
        eqs += DirichletBC(mesh_y=True, velocity_x=0) @ "inlet"
        eqs += EnforcedBC(velocity_y=var("velocity_y") - inlet_axial) @ "inlet"
        eqs += DirichletBC(mesh_y=True) @ "outlet"
        eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"

        # Pin the rear tip in the co-moving frame. The axisymmetric measure
        # vanishes at r = 0, so the constraint uses a Cartesian test.
        eqs += WeakContribution(
            var("mesh_y") - self.geometry.z_rear,
            speed_test,
            coordinate_system=cartesian,
        ) @ "interface/axis_rear"

        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="y+") @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "inlet"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "outlet"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "wall"
        eqs += IntegralObservables(volume=1)

        self += eqs @ "liquid"
        self += ODEFileOutput() @ "globals"
