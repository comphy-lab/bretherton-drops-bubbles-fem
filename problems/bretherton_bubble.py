"""Axisymmetric void bubble held stationary in a co-moving tube window.

Formulation
-----------
The liquid is meshed; the bubble interior is a void. Surface tension acts on the
liquid/void interface through ``NavierStokesFreeSurface``. There is no interior
gas domain and no volume-of-fluid field.

The calculation is done in a frame translating with the bubble at the unknown
speed ``U``. Writing the lab velocity as ``v = u + U e_z``, the ALE momentum
equation becomes the ordinary Navier-Stokes equation for ``u`` plus a single
extra term ``-rho dU/dt e_z``: the frame is non-inertial while ``U`` is still
changing. That term is fed in through the ``gravity`` argument, which pyoomph
multiplies by ``mass_density`` internally. On a stationary solve pyoomph zeroes
time derivatives, so it vanishes without special casing.

Physical gravity is a separate bulk force, ``-Bo/R_tube**2 e_z``, with
``Bo = rho_l g R_tube**2 / sigma`` and the axial coordinate positive upwards.
It remains present in the Stokes limit and is not multiplied by ``La``.
Signed imposed flow is enabled explicitly with ``allow_signed_flow=True``;
the default retains the positive-flow, zero-gravity configuration.

The consequence Vatsal is after: the bubble does not translate through the mesh,
so the window has to be long enough only for the cap disturbance to decay to the
fully developed profile - not long enough to contain a trajectory. The parent
Basilisk calculation uses ``L = 16 R0`` purely so the bubble has ~9 radii to
travel. Here the window is a convergence parameter, and
:meth:`VoidBubbleProblem.consistency` is how its adequacy is measured rather
than asserted.

Why the far field is exact
--------------------------
In a circular tube the fully developed solution *is* Poiseuille, so imposing
Poiseuille at both ends is not an approximation to an infinite tube: it is the
infinite-tube solution, provided the ends are far enough from the caps for the
cap disturbance to have decayed. That is what makes a short window legitimate.
The traction-free outlet of the parent code is available as a switch, and the
difference between the two at fixed pad is the direct, falsifiable test of the
claim.

The two global unknowns
-----------------------
``U``   the bubble speed, i.e. ``Ca_b`` directly, solved rather than fitted from
        a tip trajectory as the parent must do.
``P``   the void pressure, the Lagrange multiplier of the fixed-volume
        constraint.

``P`` cannot be set to zero. The cap curvature of a confined bubble is fixed
geometrically at roughly ``2/(R_t - h)``, so the normal-stress balance
determines ``P`` from the liquid pressure at the cap; changing the bubble volume
changes its length, not its cap curvature. Imposing ``P = 0`` and letting the
volume float therefore ties the answer to the axial pressure drop, hence to the
window length - the precise dependence this formulation exists to remove. The
bubble volume is held at ``4/3 pi R0^3``, matching the parent's incompressible
gas phase, and ``P`` is reported as a gauge relative to the mean outlet
pressure.

pyoomph pin: ``comphy-2026.09.07.1``
(``a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618``).
"""

from __future__ import annotations

import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import dataclass

from pyoomph import *
from pyoomph.equations.ALE import HyperelasticSmoothedMesh
from pyoomph.equations.generic import AxisymmetryBC, EnforcedDirichlet
from pyoomph.equations.navier_stokes import (
    NavierStokesEquations,
    NavierStokesFreeSurface,
    StokesEquations,
)
from pyoomph.expressions import *
from pyoomph.meshes.zeta import (
    AssignZetaCoordinatesByArclength,
    AssignZetaCoordinatesByEulerianCoordinate,
)
from pyoomph.output.generic import IntegralObservableOutput

from .forcing import validate_bond_number, validate_capillary_number
from .geometry import CapsuleGeometry
from .mesh import LiquidAnnulusMesh, PlainTubeMesh

NORMAL_INTO_VOID = -1
"""Sign converting the liquid outward normal to the void outward normal.

``var("normal")`` on ``liquid/interface`` points out of the liquid, i.e. into the
void, so the void's own outward normal is its negative. Both global constraints
are divergence-theorem integrals over the void's closed surface and therefore
carry this factor. It is asserted, not assumed: see
:meth:`VoidBubbleProblem.audit_constraints`, which evaluates both integrals on
the seeded capsule and compares them with the analytic values before any Newton
step is taken. A wrong sign here does not crash; it converges to nonsense.
"""


@dataclass(frozen=True)
class Consistency:
    """Checks that can fail, evaluated on a converged solution.

    ``volume_error`` is the relative departure of the void volume from
    ``4/3 pi R0^3``. It tests the constraint, the interface kinematics and the
    ALE mass balance together.

    ``flux_error`` is the relative mismatch between the inlet and outlet axial
    fluxes. With Poiseuille at both ends it is an identity and should sit at
    solver tolerance; with a traction-free outlet it is a genuine measurement.

    ``bubble_pressure`` is ``P`` referenced to the mean outlet pressure, and is
    **not** a window-independent quantity: it contains the Poiseuille drop
    between the front cap and the outlet, so it shifts with the front pad.
    Measured at ``Ca = 0.05``, it moves by 0.5714 per tube radius of front pad,
    against the analytic ``8 Ca R_t/R_t^2 = 0.5714``. ``bubble_pressure_excess``
    removes exactly that, referencing ``P`` to the fully developed liquid
    pressure extrapolated back to the front tip, and is window-independent to
    1e-4 across pads from 2 to 5 tube radii. Report the excess; the raw gauge is
    kept only because the difference between them is itself a check.

    ``stagnant_film_error`` is the departure from ``U/Ca_in = (R_t/(R_t-h))^2``,
    the exact statement that the deposited film is at rest in the laboratory
    frame. The parent's own four committed cases satisfy it to 0.13%. It is the
    sharpest available check because it couples the frame speed, the film
    thickness and the flux through one identity, and it fails if any of them is
    wrong. A residual larger than the mesh-convergence estimate is real physics -
    a film that is not stagnant - not an error, which is why it is reported
    rather than asserted.

    With nonzero Bond number the stagnant-film identity is not an acceptance
    condition. Its reported residual is undefined (NaN) at zero imposed flow.
    """

    volume_error: float
    flux_error: float
    stagnant_film_error: float
    h_flat: float
    frame_speed: float
    bubble_pressure: float
    bubble_pressure_excess: float


class VoidBubbleProblem(Problem):
    """Co-moving void bubble in an axisymmetric capillary tube."""

    def __init__(
        self,
        *,
        geometry: CapsuleGeometry,
        ca_in: float,
        laplace: float = 0.0,
        bond: float = 0.0,
        allow_signed_flow: bool = False,
        film_elements: float = 3.0,
        bulk_resolution: float = 0.08,
        closure: str = "centroid",
        outlet: str = "poiseuille",
        pressure_datum: str = "fixation",
        remesh: bool = True,
        newton_tolerance: float = 1e-8,
    ):
        super().__init__()
        validate_capillary_number(ca_in, allow_signed=allow_signed_flow)
        validate_bond_number(bond)
        if laplace < 0:
            raise ValueError("La must be non-negative")
        if film_elements < 1 or bulk_resolution <= 0:
            raise ValueError("need at least one element across the film, and a positive bulk size")
        if closure not in ("centroid", "tip", "front_tip"):
            raise ValueError("closure must be 'centroid', 'tip' or 'front_tip'")
        if outlet not in ("poiseuille", "traction_free"):
            raise ValueError("outlet must be 'poiseuille' or 'traction_free'")
        if pressure_datum not in ("fixation", "integral", "none"):
            raise ValueError("pressure_datum must be 'fixation', 'integral' or 'none'")
        self.geometry = geometry
        self.ca_in = ca_in
        self.laplace = laplace
        self.bond = bond
        self.allow_signed_flow = allow_signed_flow
        self.film_elements = film_elements
        self.bulk_resolution = bulk_resolution
        self.closure = closure
        self.outlet = outlet
        self.pressure_datum = pressure_datum
        self.remesh = remesh
        self.newton_solver_tolerance = newton_tolerance
        self.ignore_command_line = True

    # ------------------------------------------------------------------ meshing

    def film_element_size(self) -> float:
        """Target element size in the film, from the seeded gap.

        A boundary-fitted P2 velocity field resolves the near-parabolic film
        profile with far fewer elements across the gap than an interface-capturing
        scheme needs; three is the starting point and the convergence study is
        what settles it.
        """
        return self.geometry.seed_gap / self.film_elements

    def mesh_grading_length(self) -> float:
        """Distance over which the size relaxes from film to bulk.

        Several gaps wide, so that the elements spanning the film are uniformly
        fine rather than fanning out across it.
        """
        return max(4.0 * self.geometry.seed_gap, 3.0 * self.bulk_resolution)

    # ------------------------------------------------------------------ problem

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(LiquidAnnulusMesh())

        geom = self.geometry
        ca = self.define_global_parameter(Ca_in=self.ca_in)
        la = self.define_global_parameter(La=self.laplace)
        bo = self.define_global_parameter(Bo=self.bond)
        bulkforce = vector(0, -bo / geom.r_tube**2)

        # U: the bubble speed, i.e. Ca_b. Seeded from the stagnant-film identity,
        # which is a far better guess than Ca_in and matters for Newton. Its own
        # residual is the closure chosen below; the centroid form carries a
        # constant target, the tip pin does not.
        speed, speed_test = self.add_global_dof(
            "U",
            equation_contribution=-geom.axial_moment if self.closure == "centroid" else 0,
            initial_condition=self.ca_in * (geom.r_tube / geom.r_body) ** 2,
        )
        # P: the void pressure. Seeded at the cap Laplace pressure.
        pressure, pressure_test = self.add_global_dof(
            "P", equation_contribution=-geom.volume, initial_condition=2.0 / geom.r_body
        )

        # La = 0 is the Stokes limit and must use StokesEquations, not a zero
        # density: the continuity residual carries partial_t(rho)/rho, so
        # mass_density=0 is a division by zero at code-generation time. With no
        # inertia there is also no frame-acceleration force, so pyoomph's
        # density-scaled gravity term is omitted rather than multiplied by zero.
        # The physical Bond force remains as a density-independent bulk force.
        if self.laplace > 0:
            flow = NavierStokesEquations(
                mass_density=la, dynamic_viscosity=1, mode="TH",
                gravity=vector(0, -1) * partial_t(speed),
                bulkforce=bulkforce,
            )
        else:
            flow = StokesEquations(
                mass_density=1, dynamic_viscosity=1, mode="TH", bulkforce=bulkforce
            )
        # With Poiseuille at both ends every boundary carries a prescribed normal
        # velocity, and (p, P, inlet/outlet multipliers) -> (p+c, P+c, ...) is an
        # exact null vector: the free surface does NOT supply a datum once P is
        # itself an unknown, so a datum has to be supplied.
        #
        # Measured, `verification/s7_pressure_datum.py` at Ca = 0.05: the datum is
        # genuinely necessary - `pressure_datum="none"` fails to solve - and the
        # two mechanisms agree on every reported observable to 1e-14, so the
        # choice between them is free. An adversarial diagnostic pass reported a
        # near-null Jacobian mode supported on pressure dofs while the fixation
        # was active, and inferred that the fixation was not working; that
        # inference does not survive this test, since removing the datum breaks
        # the solve and the two mechanisms give identical answers. The residual
        # mode is either a different one or harmless. The switch is kept so the
        # question stays measurable rather than settled by argument.
        needs_datum = self.outlet == "poiseuille" and self.pressure_datum != "none"
        if not needs_datum:
            eqs = flow
        elif self.pressure_datum == "fixation":
            eqs = flow.with_pressure_fixation()
        else:
            eqs = flow.with_pressure_integral_constraint(self, 0)

        eqs += HyperelasticSmoothedMesh()
        eqs += MeshFileOutput()
        if self.remesh:
            eqs += RemeshWhen(RemeshingOptions())

        eqs += NavierStokesFreeSurface(
            surface_tension=1, additional_normal_traction=pressure
        ) @ "interface"

        # The two divergence-theorem constraints over the closed void surface.
        # The axis segments contribute nothing: the axisymmetric measure carries
        # 2*pi*r, which vanishes at r = 0, so the interface integral alone is the
        # whole closed-surface integral.
        void_volume = NORMAL_INTO_VOID * dot(var("coordinate"), var("normal")) / 3
        void_moment = NORMAL_INTO_VOID * var("coordinate_y") ** 2 * var("normal_y") / 2

        eqs += WeakContribution(void_volume, pressure_test) @ "interface"

        if self.closure == "centroid":
            # Fixing the void's axial first moment removes the translational null
            # mode with a smooth, dense row, splits the bubble's length change
            # between both ends as Ca varies, and survives remeshing to
            # interpolation error. Preferred over the point pin.
            eqs += WeakContribution(void_moment, speed_test) @ "interface"
        elif self.closure == "tip":
            # Point pin on the rear tip. The axisymmetric measure carries 2*pi*r
            # and the tip sits at r = 0, so this row would be identically zero
            # without the cartesian measure. Kept as a debugging alternative.
            eqs += WeakContribution(
                var("mesh_y") - geom.z_rear, speed_test, coordsys=cartesian
            ) @ "interface/axis_rear"
        else:
            # Pin the upper/front tip at its seeded axial coordinate. As for the
            # rear-tip pin, this needs a cartesian measure because the
            # axisymmetric measure vanishes on the axis.
            eqs += WeakContribution(
                var("mesh_y") - geom.z_front,
                speed_test,
                coordsys=cartesian,
            ) @ "interface/axis_front"

        # Boundary conditions, in the bubble frame. The wall moves at -U and the
        # inflow is Poiseuille shifted by -U, so both depend on the unknown U and
        # must go through EnforcedDirichlet. A strong DirichletBC may not depend
        # on an unknown, and combining the two on the SAME component silently
        # deactivates the enforced one rather than raising - hence velocity_x is
        # pinned strongly and velocity_y enforced weakly, never both on one
        # component.
        radius = var("coordinate_x")
        poiseuille = 2 * ca * (1 - (radius / geom.r_tube) ** 2)

        eqs += AxisymmetryBC() @ "axis_rear"
        eqs += AxisymmetryBC() @ "axis_front"

        # The window is rigid in the co-moving frame: outer mesh nodes do not
        # move, only the interface and the interior do.
        eqs += DirichletBC(velocity_x=0, mesh_x=True, mesh_y=True) @ "wall"
        eqs += EnforcedDirichlet(velocity_y=-speed) @ "wall"
        eqs += DirichletBC(velocity_x=0, mesh_x=True, mesh_y=True) @ "inlet"
        eqs += EnforcedDirichlet(velocity_y=poiseuille - speed) @ "inlet"

        eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "outlet"
        if self.outlet == "poiseuille":
            eqs += DirichletBC(velocity_x=0) @ "outlet"
            eqs += EnforcedDirichlet(velocity_y=poiseuille - speed) @ "outlet"
        # else: no velocity condition at all, which is the natural traction-free
        # outflow of the weak form - the parent Basilisk code's outlet.

        # The azimuthal stability code adds a third velocity component. No-slip
        # on the rigid wall and prescribed Poiseuille flow at both ends require
        # its perturbation to vanish there; the axis is handled by AxisymmetryBC.
        if self.is_normal_mode_stability_set_up() == "azimuthal":
            for boundary in ("wall", "inlet", "outlet"):
                eqs += DirichletBC(velocity_phi=0) @ boundary

        # Interface charts for old-to-new interpolation across a remesh. Arclength
        # rather than an Eulerian coordinate, because the interface is not
        # single-valued in r and need not stay single-valued in z at the caps.
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="y+") @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "inlet"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "outlet"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "wall"

        eqs += IntegralObservables(liquid_volume=1)
        eqs += IntegralObservables(
            void_volume=void_volume, void_moment=void_moment, interface_area=1
        ) @ "interface"
        eqs += IntegralObservables(inlet_flux=var("velocity_y"), inlet_area=1) @ "inlet"
        eqs += IntegralObservables(
            outlet_flux=var("velocity_y"), outlet_area=1, outlet_pressure=var("pressure")
        ) @ "outlet"

        self += eqs @ "liquid"
        self += ODEFileOutput() @ "globals"
        self += IntegralObservableOutput("interface") @ "liquid/interface"
        self += IntegralObservableOutput("bulk") @ "liquid"

    # ----------------------------------------------------------------- solving

    def solve_to_steady(
        self,
        *,
        relax_time: float = 8.0,
        start_step: float = 1e-4,
        temporal_error: float = 1e-4,
        max_newton_iterations: int = 40,
        try_stationary_first: bool = False,
    ) -> str:
        """Reach the steady co-moving state, and report which route got there.

        A plain stationary Newton solve from the seeded capsule does **not**
        converge for a confined bubble at any capillary number - measured across
        ``Ca`` from 0.5 down to 0.05, all diverging on the first step from a
        healthy initial residual and a cleanly factorised Jacobian.

        The cause is measured, and it is a step length rather than a missing
        solution. The first Newton step displaces interface mesh nodes by about
        16 film-element widths, elements invert, and the residual becomes NaN -
        which is what the ``Maximum residuals inf`` message actually reports. The
        direction itself is fine: damping the same step to a fifth of its length
        *reduces* the residual norm by 22%. For comparison, the unconfined
        control in ``verification/s1_static_sphere.py`` displaces nodes by half
        an element width and converges at full step. So the formulation is sound
        and the regime is merely stiff; what the confined geometry costs is the
        right to take an undamped first step.

        The route is therefore a short transient followed by a stationary polish.
        The transient is a **homotopy into the basin, not a relaxation to the
        answer**: measured, a run out to ``t = 20`` still has ``h_flat/R_tube``
        at 0.1201 with the film scatter still growing, and the stationary polish
        then moves it by +18% to 0.1422. Reading the transient as "the film being
        deposited" and stopping on a plateau would give the wrong number. Its only
        job is to supply an admissible shape. Consistent with that, the converged
        answer is insensitive to the schedule: ``endtime=8, temporal_error=1e-4,
        maxstep=1.0`` and ``endtime=20, temporal_error=1e-3, maxstep=0.05`` agree
        to every printed digit.

        The stationary polish is what makes the result a steady state rather than
        a transient sampled at a stopping criterion - the parent lab-frame code
        has no choice but the latter, and must march ``travelR/Ca`` = 160 time
        units at ``Ca = 0.05`` to get there.

        ``try_stationary_first`` defaults to False because a stationary attempt
        from a seeded capsule has never been observed to succeed, and a diverged
        Newton solve leaves the diverged values in the solution vector. Restoring
        a checkpoint afterwards is not reliably enough to recover - measured -
        so the attempt is skipped rather than paid for. Set it True only when
        continuing from an already converged nearby state, where it is the fast
        path.

        Returns the route taken, ``"stationary"`` or ``"transient+stationary"``,
        so a driver can record how the number was obtained rather than assume.
        """
        if try_stationary_first:
            # A diverged Newton solve leaves the diverged values in the solution
            # vector, so the fallback would start from garbage rather than from
            # the seeded capsule. Checkpoint first and restore on failure; this
            # is not defensive tidiness, it is the difference between the
            # fallback working and silently solving a different problem.
            import io

            checkpoint = io.BytesIO()
            self.save_state(checkpoint, quiet=True)
            try:
                self.solve(max_newton_iterations=max_newton_iterations)
                return "stationary"
            except Exception:  # noqa: BLE001 - any solver failure is the signal to fall back
                checkpoint.seek(0)
                self.load_state(checkpoint, quiet=True)
        self.run(
            relax_time,
            startstep=start_step,
            outstep=relax_time / 8,
            temporal_error=temporal_error,
            maxstep=relax_time / 8,
        )
        self.solve(max_newton_iterations=max_newton_iterations)
        return "transient+stationary"

    # ---------------------------------------------------------------- readback

    def _global(self, name: str) -> float:
        return float(self.get_ode("globals").get_value(name, as_float=True))

    def current_ca_in(self) -> float:
        """The imposed inlet speed **now**, read from the global parameter.

        Not ``self.ca_in``. That attribute records the value the problem was
        constructed with, and continuation moves the global parameter without
        touching it. Reading the attribute after ``go_to_param`` silently
        compares the new bubble speed against the old inlet speed: measured, a
        continuation from Ca_in = 2 to 2.5 reported a stagnant-film residual of
        +0.25 where the true value is -4.1e-4, and by Ca_in = 6 it read +2.0.
        The physics was right and only the check was wrong, which is the worst
        way for a check to fail.
        """
        try:
            return float(self.get_global_parameter("Ca_in").value)
        except Exception:  # noqa: BLE001 - before the parameter exists, fall back
            return float(self.ca_in)

    def current_bond(self) -> float:
        """The tube-radius Bond number now, including continuation updates."""
        if "Bo" not in self.get_global_parameter_names():
            return float(self.bond)
        try:
            return float(self.get_global_parameter("Bo").value)
        except Exception:  # noqa: BLE001 - before the parameter exists, fall back
            return float(self.bond)

    def frame_speed(self) -> float:
        """``U``, which in this nondimensionalisation *is* ``Ca_b``.

        Solved as an unknown, not fitted to a tip trajectory. The parent code has
        to regress the front-tip position over a trailing time window because it
        has no unknown holding the speed; here the value is exact to Newton
        tolerance and carries no fit-window sensitivity.
        """
        return self._global("U")

    def bubble_pressure(self) -> float:
        """Void pressure ``P`` in the solver's datum."""
        return self._global("P")

    def observable(self, name: str, domain: str = "liquid") -> float:
        return float(self.get_mesh(domain).evaluate_observable(name))

    def outlet_mean_pressure(self) -> float:
        return self.observable("outlet_pressure", "liquid/outlet") / self.observable(
            "outlet_area", "liquid/outlet"
        )

    def interface_polyline(self):
        """Interface node coordinates as ``(z, r)`` arrays, in curve order.

        The mesh nodes are the interface: there is no reconstruction step and no
        isoline to chase, which is the structural advantage of the sharp
        formulation over the parent's VOF facets.

        The order comes from walking the element chain, **not** from sorting on
        ``z``. At high capillary number the rear meniscus is re-entrant: it
        folds back, so ``r(z)`` is multivalued there and a z-sort interleaves the
        two branches. Measured at ``Ca_in = 6``, sorting produced jumps of 0.30
        in ``r`` between consecutive points at the same ``z``, drawing the rear
        cap as a zigzag that looked like a failed solve. The reported film was
        never affected - ``measure_film`` takes a median of gaps over a window in
        ``z``, which does not depend on order - but the profile was wrong, and a
        wrong-looking profile is indistinguishable from a wrong solution until
        someone checks.
        """
        import numpy as np

        mesh = self.get_mesh("liquid/interface")
        elements = list(mesh.elements())

        def key(node):
            return (round(node.x(0), 12), round(node.x(1), 12))

        # Quadratic line elements are stored [end, mid, end].
        segments = []
        for el in elements:
            if el.nnode() < 3:
                continue
            a, mid, b = el.node_pt(0), el.node_pt(1), el.node_pt(2)
            segments.append((key(a), key(mid), key(b)))
        if not segments:
            raise RuntimeError("interface mesh has no line elements to walk")

        adjacency: dict = {}
        for index, (a, _mid, b) in enumerate(segments):
            adjacency.setdefault(a, []).append(index)
            adjacency.setdefault(b, []).append(index)

        ends = [node for node, hits in adjacency.items() if len(hits) == 1]
        if len(ends) != 2:
            raise RuntimeError(
                f"interface chain has {len(ends)} endpoints, expected 2 (the two "
                "tips on the axis); the interface may be split or closed"
            )
        start = min(ends, key=lambda node: node[1])  # begin at the rear tip

        chain = [start]
        used = set()
        node = start
        while True:
            nxt = [i for i in adjacency[node] if i not in used]
            if not nxt:
                break
            index = nxt[0]
            used.add(index)
            a, mid, b = segments[index]
            far = b if a == node else a
            chain.extend([mid, far])
            node = far
        if len(used) != len(segments):
            raise RuntimeError(
                f"walked {len(used)} of {len(segments)} interface elements; "
                "the chain is not connected"
            )

        pts = np.array([[z, r] for r, z in chain], dtype=float)
        return pts[:, 0], pts[:, 1]

    def measure_film(self, trim: float = 0.30):
        """Deposited central film, by the parent's definition."""
        from postprocess.film import measure_film

        z, r = self.interface_polyline()
        return measure_film(z, r, r_tube=self.geometry.r_tube, trim=trim)

    # ---------------------------------------------------------------- checking

    def audit_constraints(self, rtol: float = 0.02) -> dict[str, float]:
        """Verify the two constraint integrals on the mesh as built.

        Both global constraints are divergence-theorem surface integrals whose
        sign depends on the orientation pyoomph gives ``var("normal")``. A flipped
        sign does not raise and does not diverge loudly - it converges to a
        different problem. This evaluates both integrals against the analytic
        capsule before any Newton step and raises if they disagree, so the
        orientation is established by measurement rather than by assumption.

        Call it immediately after ``initialise()`` and before the first solve.
        """
        geom = self.geometry
        measured_volume = self.observable("void_volume", "liquid/interface")
        measured_moment = self.observable("void_moment", "liquid/interface")
        errors = {
            "void_volume": measured_volume / geom.volume - 1.0,
            "void_moment": measured_moment / geom.axial_moment - 1.0,
        }
        bad = {k: v for k, v in errors.items() if abs(v) > rtol}
        if bad:
            raise RuntimeError(
                "constraint integrals disagree with the seeded capsule: "
                + ", ".join(
                    f"{k} measured/analytic - 1 = {v:+.4g}" for k, v in sorted(bad.items())
                )
                + f" (void_volume={measured_volume:.6g} vs {geom.volume:.6g}, "
                f"void_moment={measured_moment:.6g} vs {geom.axial_moment:.6g}). "
                "A sign near -2 means NORMAL_INTO_VOID has the wrong sign."
            )
        return errors

    def consistency(self, trim: float = 0.30) -> Consistency:
        """Checks that can fail, for a converged solution. See :class:`Consistency`."""
        geom = self.geometry
        film = self.measure_film(trim=trim)
        speed = self.frame_speed()
        ca_in = self.current_ca_in()

        void_volume = self.observable("void_volume", "liquid/interface")
        inlet_flux = self.observable("inlet_flux", "liquid/inlet")
        outlet_flux = self.observable("outlet_flux", "liquid/outlet")
        scale = max(abs(inlet_flux), abs(outlet_flux), 1e-300)

        stagnant = (geom.r_tube / (geom.r_tube - film.h_flat)) ** 2
        stagnant_film_error = (
            (speed / ca_in) / stagnant - 1.0 if ca_in != 0 else float("nan")
        )
        gauge = self.bubble_pressure() - self.outlet_mean_pressure()
        # Viscous and hydrostatic pressure drops from the measured front tip to
        # the outlet, which the raw gauge contains and which have nothing to do
        # with the bubble.
        front_pad_drop = (
            (8.0 * ca_in + self.current_bond())
            * (geom.l_domain - film.z_front)
            / geom.r_tube**2
        )
        return Consistency(
            volume_error=void_volume / geom.volume - 1.0,
            flux_error=(outlet_flux - inlet_flux) / scale,
            stagnant_film_error=stagnant_film_error,
            h_flat=film.h_flat,
            frame_speed=speed,
            bubble_pressure=gauge,
            bubble_pressure_excess=gauge - front_pad_drop,
        )


class PoiseuilleTubeProblem(Problem):
    """The same window with no bubble: the analytic verification case.

    Steady flow in a straight tube is exactly Poiseuille, and a P2 velocity space
    represents that parabola exactly on straight-sided elements, so the pressure
    drop must come out at ``8 Ca L / R_t^2`` to solver precision rather than to
    discretisation error. That makes this a test that can genuinely fail: it
    catches a wrong coordinate system, a mis-scaled axisymmetric measure, an
    ``EnforcedDirichlet`` that is silently inactive, and a missing pressure datum,
    none of which announce themselves in the full problem.
    """

    def __init__(
        self,
        *,
        r_tube: float = 0.7,
        l_domain: float = 8.0,
        ca_in: float = 0.05,
        bulk_resolution: float = 0.08,
        outlet: str = "poiseuille",
        newton_tolerance: float = 1e-10,
    ):
        super().__init__()
        if outlet not in ("poiseuille", "traction_free"):
            raise ValueError("outlet must be 'poiseuille' or 'traction_free'")
        self.r_tube = r_tube
        self.l_domain = l_domain
        self.ca_in = ca_in
        self.bulk_resolution = bulk_resolution
        self.outlet = outlet
        self.newton_solver_tolerance = newton_tolerance
        self.ignore_command_line = True

    @property
    def expected_pressure_drop(self) -> float:
        """``8 mu Ca L / R_t^2`` with ``mu = 1``."""
        return 8.0 * self.ca_in * self.l_domain / self.r_tube**2

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(PlainTubeMesh())

        ca = self.define_global_parameter(Ca_in=self.ca_in)
        radius = var("coordinate_x")
        poiseuille = 2 * ca * (1 - (radius / self.r_tube) ** 2)

        flow = StokesEquations(mass_density=1, dynamic_viscosity=1, mode="TH")
        eqs = flow.with_pressure_fixation() if self.outlet == "poiseuille" else flow
        eqs += MeshFileOutput()

        eqs += AxisymmetryBC() @ "axis"
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "wall"
        eqs += DirichletBC(velocity_x=0) @ "inlet"
        # EnforcedDirichlet rather than DirichletBC even though nothing here
        # depends on an unknown: this problem exists to exercise the same code
        # path the real one uses.
        eqs += EnforcedDirichlet(velocity_y=poiseuille) @ "inlet"
        if self.outlet == "poiseuille":
            eqs += DirichletBC(velocity_x=0) @ "outlet"
            eqs += EnforcedDirichlet(velocity_y=poiseuille) @ "outlet"

        eqs += IntegralObservables(
            inlet_pressure=var("pressure"), inlet_flux=var("velocity_y"), inlet_area=1
        ) @ "inlet"
        eqs += IntegralObservables(
            outlet_pressure=var("pressure"), outlet_flux=var("velocity_y"), outlet_area=1
        ) @ "outlet"
        self += eqs @ "liquid"

    def observable(self, name: str, domain: str) -> float:
        return float(self.get_mesh(domain).evaluate_observable(name))

    def measured_pressure_drop(self) -> float:
        p_in = self.observable("inlet_pressure", "liquid/inlet") / self.observable(
            "inlet_area", "liquid/inlet"
        )
        p_out = self.observable("outlet_pressure", "liquid/outlet") / self.observable(
            "outlet_area", "liquid/outlet"
        )
        return p_in - p_out

    def measured_flux(self) -> float:
        return self.observable("inlet_flux", "liquid/inlet")

    @property
    def expected_flux(self) -> float:
        """``Ca * pi R_t^2``: the axisymmetric measure already carries ``2 pi r``."""
        import math

        return self.ca_in * math.pi * self.r_tube**2
