"""Two-phase Stokes model with upper flow and lower static-pressure ports.

The material ratios and pressure conversion follow Herrada, Yu & Stone
(2023), doi:10.1017/jfm.2023.123. The lower radial velocity is set to zero;
that completion is explicit because the article specifies only its pressure.
Case validation imports no native solver. Native definitions are loaded only
when a caller constructs a problem.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

PYOOMPH_COMMIT = "a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618"
CASE_SECTIONS = ["geometry", "nondimensionalization", "physics", "boundaries", "pressure",
                 "closure", "mesh", "tolerances", "solve", "output", "completion"]
CASE_FIELDS_USED = ["schema", "benchmark_schema", "case_id", *CASE_SECTIONS]
PROBLEMS = {"finite_gas_bubble", "bare_tube", "static_laplace"}
MATERIAL_RATIOS = {"gas_density_ratio": 0.000949, "gas_viscosity_ratio": 0.000018}
MIN_SEED_CYLINDER_OVER_R0 = 1e-12


def _number(mapping, name, errors, *, positive=False, nonnegative=False):
    value = mapping.get(name)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)
            or (nonnegative and value < 0)):
        errors.append(name + " must be finite" + (" and positive" if positive else
                      " and nonnegative" if nonnegative else ""))
        return math.nan
    return float(value)


def validate_case(case: dict) -> dict:
    """Validate every physical and numerical field used by this model."""
    if not isinstance(case, dict):
        raise ValueError("case must be a JSON object")  # noqa: TRY004 - case validation contract
    errors = []
    if case.get("schema") != "pyoomph-case-v1":
        errors.append("schema must be pyoomph-case-v1")
    if case.get("benchmark_schema") != "herrada-paper-model-v2":
        errors.append("benchmark_schema must be herrada-paper-model-v2")
    if not isinstance(case.get("case_id"), str) or not case["case_id"].strip():
        errors.append("case_id must be nonempty")
    sections = {}
    for section in CASE_SECTIONS:
        value = case.get(section)
        if not isinstance(value, dict):
            errors.append(section + " must be an object")
            value = {}
        sections[section] = value
    if sections["geometry"] != {"kind": "axisymmetric", "coordinates": ["r", "z"],
                                "frame": "co-moving"}:
        errors.append("geometry must declare axisymmetric (r,z) co-moving coordinates")
    if sections["nondimensionalization"] != {"surface_tension": 1.0, "liquid_viscosity": 1.0,
                                             "volume_equivalent_radius": 1.0}:
        errors.append("nondimensionalization must set sigma=mu_liquid=R0=1")
    for name in ("surface_tension", "liquid_viscosity", "volume_equivalent_radius"):
        _number(sections["nondimensionalization"], name, errors, positive=True)
    if sections["pressure"] != {"field": "static", "reference": "lower_static_zero",
                                "body_force": "explicit_both_phases"}:
        errors.append("pressure must use static pressure and explicit phase body forces")
    if sections["boundaries"] != {"upper": "poiseuille", "lower": "static_pressure",
                                  "wall": "no_slip", "lower_tangential": "zero_velocity"}:
        errors.append("boundaries must declare upper flow, lower static pressure and zero tangential velocity")
    p, mesh, solve = sections["physics"], sections["mesh"], sections["solve"]
    kind = p.get("problem")
    if not isinstance(kind, str) or kind not in PROBLEMS:
        errors.append("physics.problem is unsupported")
    physical_names = {"problem", "r0", "r_tube", "domain_over_r", "vhat", "bond", "reynolds",
                      "ca_in", "upper_tip_over_r", *MATERIAL_RATIOS,
                      "frame_speed" if kind == "bare_tube" else "seed_radius_over_tube"}
    numerical_names = {
        "physics": physical_names,
        "mesh": {"template", "bulk_resolution", "film_elements", "remeshing"},
        "solve": {"mode", "relax_time", "start_step", "initial_bond", "bond_step",
                  "max_newton_iterations", "max_bond_steps"},
        "tolerances": {"newton", "temporal_error", "volume_relative", "flux_absolute",
                       "pressure_boundary_absolute", "velocity_reference_relative",
                       "pressure_reference_relative", "laplace_relative", "sphere_radius_absolute",
                       "static_velocity_absolute"},
    }
    for section, allowed in numerical_names.items():
        unknown = set(sections[section]) - allowed
        if unknown:
            errors.append(section + " has unused fields: " + ", ".join(sorted(unknown)))
    numbers = {}
    for name in ("r0", "r_tube", "domain_over_r", "vhat"):
        numbers[name] = _number(p, name, errors, positive=True)
    for name in ("bond", "reynolds"):
        numbers[name] = _number(p, name, errors, nonnegative=True)
    numbers["ca_in"] = _number(p, "ca_in", errors)
    numbers["upper_tip_over_r"] = _number(p, "upper_tip_over_r", errors, positive=True)
    if numbers["r0"] != 1:
        errors.append("physics.r0 must be 1 under the declared units")
    if numbers["reynolds"] != 0:
        errors.append("physics.reynolds must be zero: both phases use Stokes equations")
    for name, expected in MATERIAL_RATIOS.items():
        value = _number(p, name, errors, positive=True)
        if value != expected:
            errors.append("physics." + name + " differs from the prescribed paper material ratio")
    if (math.isfinite(numbers["r_tube"]) and math.isfinite(numbers["vhat"])
            and not math.isclose(numbers["vhat"], (numbers["r0"] / numbers["r_tube"])**3,
                                 rel_tol=1e-12, abs_tol=1e-12)):
        errors.append("physics.vhat must equal (r0/r_tube)^3")
    if not math.isclose(numbers["upper_tip_over_r"], 2 * numbers["domain_over_r"] / 3,
                        rel_tol=1e-12, abs_tol=1e-12):
        errors.append("physics.upper_tip_over_r must be 2*domain_over_r/3")
    expected_closure = ({"frame_speed": "prescribed", "gas_volume": "not_applicable"}
                        if kind == "bare_tube" else
                        {"frame_speed": "front_tip", "gas_volume": "pressure_constraint"})
    if sections["closure"] != expected_closure:
        errors.append("closure differs from the declared frame and gas-volume constraints")
    if kind == "bare_tube":
        _number(p, "frame_speed", errors)
    else:
        _number(p, "seed_radius_over_tube", errors, positive=True)
    if kind == "static_laplace" and (numbers["bond"] != 0 or numbers["ca_in"] != 0):
        errors.append("static_laplace requires bond=ca_in=0")
    expected_mesh = "PaperTubeMesh" if kind == "bare_tube" else "PaperTwoPhaseMesh"
    if mesh.get("template") != expected_mesh:
        errors.append("mesh.template must be " + expected_mesh)
    _number(mesh, "bulk_resolution", errors, positive=True)
    film = _number(mesh, "film_elements", errors, positive=True)
    if film < 1:
        errors.append("mesh.film_elements must be at least one")
    if not isinstance(mesh.get("remeshing"), bool):
        errors.append("mesh.remeshing must be boolean")
    if kind == "bare_tube" and mesh.get("remeshing") is not False:
        errors.append("the bare-tube verification has a fixed mesh")
    for name in ("newton", "temporal_error", "volume_relative", "flux_absolute",
                 "pressure_boundary_absolute", "velocity_reference_relative",
                 "pressure_reference_relative", "laplace_relative", "sphere_radius_absolute",
                 "static_velocity_absolute"):
        _number(sections["tolerances"], name, errors, positive=True)
    if solve.get("mode") != "stationary":
        errors.append("solve.mode must be stationary")
    relax = _number(solve, "relax_time", errors, nonnegative=True)
    step = _number(solve, "start_step", errors, positive=True)
    _number(solve, "bond_step", errors, positive=True)
    initial = _number(solve, "initial_bond", errors, nonnegative=True)
    if initial > numbers["bond"]:
        errors.append("solve.initial_bond cannot exceed physics.bond")
    if relax and step > relax:
        errors.append("solve.start_step cannot exceed relax_time")
    for name in ("max_newton_iterations", "max_bond_steps"):
        if isinstance(solve.get(name), bool) or not isinstance(solve.get(name), int) or solve[name] < 1:
            errors.append("solve." + name + " must be a positive integer")
    bond_step = solve.get("bond_step")
    max_bond_steps = solve.get("max_bond_steps")
    if (isinstance(max_bond_steps, int) and not isinstance(max_bond_steps, bool)
            and max_bond_steps > 0 and isinstance(bond_step, (int, float))
            and not isinstance(bond_step, bool) and math.isfinite(bond_step) and bond_step > 0
            and math.isfinite(initial) and math.isfinite(numbers["bond"])
            and initial + max_bond_steps * bond_step < numbers["bond"] - 1e-12):
        errors.append("solve.max_bond_steps cannot reach the selected Bond number")
    if (set(sections["output"]) != {"write_states", "cadence"}
            or sections["output"].get("write_states") is not True):
        errors.append("output must declare cadence and request a portable final state")
    _number(sections["output"], "cadence", errors, positive=True)
    if sections["completion"] != {"kind": "event", "value": "stationary_and_checks"}:
        errors.append("completion must be stationary_and_checks")
    if not errors and kind != "bare_tube":
        try:
            geom = case_geometry(case)
            if 0 < geom.cylinder_length <= MIN_SEED_CYLINDER_OVER_R0 * geom.r0:
                errors.append("seed cylinder must be zero or longer than 1e-12 R0")
            if kind == "static_laplace" and geom.r_tube <= geom.r0:
                errors.append("static_laplace requires an unconfined sphere")
        except ValueError as exc:
            errors.append("physics geometry: " + str(exc))
    if errors:
        raise ValueError("; ".join(errors))
    return copy.deepcopy(case)


def read_case(path: Path) -> dict:
    """Read a portable case and validate it before creating any output."""
    return validate_case(json.loads(Path(path).read_text(encoding="utf-8")))


def case_geometry(case):
    """Return the seeded geometry in the declared R0 units."""
    from .geometry import CapsuleGeometry

    p = case["physics"]
    radius, height = float(p["r_tube"]), float(p["domain_over_r"] * p["r_tube"])
    if p["problem"] == "bare_tube":
        return SimpleNamespace(r_tube=radius, l_domain=height, r0=float(p["r0"]))
    body = radius * float(p["seed_radius_over_tube"])
    front = radius * float(p["upper_tip_over_r"])
    return CapsuleGeometry(r0=float(p["r0"]), r_tube=radius, r_body=body,
                           z_rear=front - CapsuleGeometry.bubble_length(float(p["r0"]), body),
                           l_domain=height)


def phase_force_density(bond, radius, density_ratio=1.0):
    """Gravity per unit volume; density is included exactly once."""
    if not all(math.isfinite(x) for x in (bond, radius, density_ratio)):
        raise ValueError("force parameters must be finite")
    if bond < 0 or radius <= 0 or density_ratio <= 0:
        raise ValueError("force needs nonnegative Bond number and positive radius/density")
    return 0.0, -density_ratio * bond / radius**2


def analytic_tube_fields(r, z, *, radius, ca_in, bond, frame_speed=0):
    """Independent fully developed Stokes solution with lower static p=0."""
    return {"velocity_r": 0.0,
            "velocity_z": -frame_speed + 2 * ca_in * (1 - (r / radius)**2),
            "pressure_static": -(bond + 8 * ca_in) * z / radius**2}


def bulk_velocity_rms(velocity_squared_integral, volume):
    """Volume-weighted speed RMS from the bulk's coordinate-aware integrals."""
    if (isinstance(velocity_squared_integral, bool) or isinstance(volume, bool)
            or not all(isinstance(value, (int, float)) and math.isfinite(value)
                       for value in (velocity_squared_integral, volume))
            or velocity_squared_integral < 0 or volume <= 0):
        raise ValueError("velocity RMS requires a finite nonnegative squared integral and positive volume")
    return math.sqrt(velocity_squared_integral / volume)


def _bulk_viscous_stress(velocity, viscosity, bulk):
    """Trace the full bulk stress, retaining cylindrical and moving-mesh terms.

    The pinned public grad defaults to the current interface dimensions.
    Its underlying binding accepts the explicit dimensions of the parent bulk.
    """
    from pyoomph import _pyoomph_core
    from pyoomph.expressions import Expression, evaluate_in_domain, sym

    gradient = _pyoomph_core.GiNaC_grad(
        velocity, Expression(bulk.get_nodal_dimension()), Expression(bulk.get_element_dimension()),
        _pyoomph_core.GiNaC_wrap_coordinate_system(bulk.get_coordinate_system()), Expression(1))
    return evaluate_in_domain(2 * viscosity * sym(gradient), bulk)


@lru_cache(maxsize=1)
def _native_problem_types():
    """Load the pinned native APIs without compiling or initialising a problem."""
    from pyoomph import (
        DirichletBC,
        InitialCondition,
        IntegralObservables,
        InterfaceEquations,
        MeshFileOutput,
        ODEFileOutput,
        Problem,
        RemeshingOptions,
        RemeshWhen,
        WeakContribution,
    )
    from pyoomph.equations.ALE import (
        ConnectMeshAtInterface,
        EnforceVolumeByPressure,
        HyperelasticSmoothedMesh,
    )
    from pyoomph.equations.generic import AxisymmetryBC, EnforcedDirichlet
    from pyoomph.equations.navier_stokes import (
        ConnectVelocityAtInterface,
        NavierStokesFreeSurface,
        StokesEquations,
    )
    from pyoomph.expressions import (
        cartesian,
        dot,
        matproduct,
        testfunction,
        var,
        vector,
        weak,
    )
    from pyoomph.meshes.zeta import (
        AssignZetaCoordinatesByArclength,
        AssignZetaCoordinatesByEulerianCoordinate,
    )

    from .paper_mesh import PaperTubeMesh, PaperTwoPhaseMesh

    class StaticPressurePort(InterfaceEquations):
        """Prescribe static pressure by cancelling the normal viscous flux."""

        required_parent_type = StokesEquations

        def __init__(self, target=0):
            super().__init__()
            self.target = target

        def define_residuals(self):
            flow = self.get_parent_equations(StokesEquations)
            bulk = self.get_parent_domain()
            velocity = var(flow.velocity_name, domain=bulk)
            normal = self.get_normal()
            viscous = _bulk_viscous_stress(velocity, flow.dynamic_viscosity, bulk)
            normal_viscous = dot(normal, matproduct(viscous, normal))
            self.add_residual(weak(self.target - normal_viscous,
                                   dot(normal, testfunction(flow.velocity_name))))

    class PaperBubbleProblem(Problem):
        """Finite gas, static-pressure fields and a frame pinned to the front tip."""

        def __init__(self, case, *, initial_bond=None):
            super().__init__()
            self.case = validate_case(case)
            self.geometry = case_geometry(case)
            self.bulk_resolution = float(case["mesh"]["bulk_resolution"])
            self.film_elements = float(case["mesh"]["film_elements"])
            self.bond = float(case["physics"]["bond"] if initial_bond is None else initial_bond)
            self.newton_solver_tolerance = float(case["tolerances"]["newton"])
            self.ignore_command_line = True

        def define_problem(self):
            self.set_coordinate_system("axisymmetric")
            p = self.case["physics"]
            geom = self.geometry
            bare = p["problem"] == "bare_tube"
            self.add_mesh(PaperTubeMesh() if bare else PaperTwoPhaseMesh())
            ca = self.define_global_parameter(Ca_in=float(p["ca_in"]))
            bo = self.define_global_parameter(Bo=self.bond)
            if bare:
                speed = float(p["frame_speed"])
            else:
                speed, speed_test = self.add_global_dof("U", initial_condition=
                    float(p["ca_in"]) * (geom.r_tube / geom.r_body)**2)
            liquid = StokesEquations(dynamic_viscosity=1, mass_density=1, mode="TH",
                                     bulkforce=vector(0, -bo / geom.r_tube**2))
            liquid += MeshFileOutput()
            if not bare:
                liquid += HyperelasticSmoothedMesh()
                if self.case["mesh"]["remeshing"]:
                    liquid += RemeshWhen(RemeshingOptions(
                        on_inverted_element=p["problem"] == "finite_gas_bubble"))
            for boundary in ("wall", "lower", "upper"):
                if not bare:
                    liquid += DirichletBC(mesh_x=True, mesh_y=True) @ boundary
            liquid += DirichletBC(velocity_x=0) @ "wall"
            liquid += EnforcedDirichlet(velocity_y=-speed) @ "wall"
            liquid += DirichletBC(velocity_x=0) @ "upper"
            liquid += EnforcedDirichlet(velocity_y=
                2 * ca * (1 - (var("coordinate_x") / geom.r_tube)**2) - speed) @ "upper"
            liquid += DirichletBC(velocity_x=0) @ "lower"
            liquid += StaticPressurePort(0) @ "lower"
            if self.is_normal_mode_stability_set_up() == "azimuthal":
                for boundary in ("wall", "lower", "upper"):
                    liquid += DirichletBC(velocity_phi=0) @ boundary
            for boundary in (("axis",) if bare else ("axis_rear", "axis_front")):
                liquid += AxisymmetryBC() @ boundary
            for boundary in ("lower", "upper"):
                liquid += IntegralObservables(flux=var("velocity_y"), area=1,
                    pressure=var("pressure"), pressure_squared=var("pressure")**2) @ boundary
                if not bare:
                    liquid += AssignZetaCoordinatesByEulerianCoordinate("x") @ boundary
            liquid += IntegralObservables(volume=1, pressure=var("pressure"),
                                            velocity_squared=dot(var("velocity"), var("velocity")))
            if bare:
                expected_u = 2 * ca * (1 - (var("coordinate_x") / geom.r_tube)**2) - speed
                expected_p = -(bo + 8 * ca) * var("coordinate_y") / geom.r_tube**2
                liquid += IntegralObservables(velocity_error_squared=
                    var("velocity_x")**2 + (var("velocity_y") - expected_u)**2,
                    velocity_reference_squared=expected_u**2,
                    pressure_error_squared=(var("pressure") - expected_p)**2,
                    pressure_reference_squared=expected_p**2)
            else:
                liquid += (NavierStokesFreeSurface(surface_tension=1, static_interface=False)
                           + ConnectVelocityAtInterface() + ConnectMeshAtInterface()) @ "interface"
                liquid += WeakContribution(var("mesh_y") - geom.z_front,
                                            speed_test, coordsys=cartesian) @ "interface/axis_front"
                liquid += IntegralObservables(surface_volume=
                    -dot(var("coordinate"), var("normal")) / 3, area=1) @ "interface"
                liquid += AssignZetaCoordinatesByArclength(sort_along_axis="y+") @ "interface"
                liquid += AssignZetaCoordinatesByEulerianCoordinate("y") @ "wall"
            self += liquid @ "liquid"
            if not bare:
                gas = StokesEquations(dynamic_viscosity=float(p["gas_viscosity_ratio"]),
                    mass_density=float(p["gas_density_ratio"]), mode="TH",
                    bulkforce=vector(0, -float(p["gas_density_ratio"]) * bo / geom.r_tube**2))
                gas += HyperelasticSmoothedMesh() + MeshFileOutput()
                gas += AxisymmetryBC() @ "gas_axis"
                gas += EnforceVolumeByPressure(geom.volume, ode_storage_domain="gas_volume")
                gas += InitialCondition(pressure=2 / geom.r_body)
                gas += AssignZetaCoordinatesByArclength(sort_along_axis="y+") @ "interface"
                gas += AssignZetaCoordinatesByEulerianCoordinate("y") @ "gas_axis"
                gas += IntegralObservables(volume=1, pressure=var("pressure"),
                    velocity_squared=dot(var("velocity"), var("velocity")))
                self += gas @ "gas"
                self += ODEFileOutput() @ "globals"
                self += ODEFileOutput() @ "gas_volume"

        def observable(self, name, domain="liquid"):
            return float(self.get_mesh(domain).evaluate_observable(name))

        def frame_speed(self):
            if self.case["physics"]["problem"] == "bare_tube":
                return float(self.case["physics"]["frame_speed"])
            return float(self.get_ode("globals").get_value("U", as_float=True))

        def interface_polyline(self):
            # The existing walker uses this same liquid/interface element chain.
            from .bretherton_bubble import VoidBubbleProblem
            return VoidBubbleProblem.interface_polyline(self)

    return PaperBubbleProblem, StaticPressurePort


def build_problem(case, *, initial_bond=None):
    """Construct the model; native compilation begins only at initialise()."""
    case = validate_case(case)
    if initial_bond is not None and (
        isinstance(initial_bond, bool) or not isinstance(initial_bond, (int, float))
        or not math.isfinite(initial_bond) or not 0 <= initial_bond <= case["physics"]["bond"]
    ):
        raise ValueError("initial_bond must be finite and lie between zero and the case Bond number")
    return _native_problem_types()[0](case, initial_bond=initial_bond)


def _stationary_residual(problem):
    import numpy as np
    steppers = [problem.time_stepper_pt(i) for i in range(problem.ntime_stepper())]
    steady = [stepper.is_steady() for stepper in steppers]
    try:
        for stepper in steppers:
            stepper.make_steady()
        value = float(np.max(np.abs(np.asarray(problem.get_residuals()))))
        if not math.isfinite(value):
            raise ValueError("nonfinite stationary residual")
        return value
    finally:
        for stepper, was_steady in zip(steppers, steady, strict=True):
            if not was_steady:
                stepper.undo_make_steady()


def polish_confined_state(problem, case, output, label, *, initial_relaxation=0.5):
    """Bounded damped solve with the pinned core's evidenced-inversion recovery.

    The core restores its own full two-phase snapshot before remeshing, and
    propagates failures without inversion reports. Failed-inversion retries
    share a mesh stage's budget. A quality remesh after convergence starts a
    new stage at unchanged damping; interpolation requires another steady check.
    Each stage permits at most 40 updates, with at most 160 updates in total.
    Gmsh files referenced by the snapshots must remain in their run directory.
    """
    import numpy as np
    from pyoomph.generic.problem import GenericProblemHooks

    stage_limit = min(40, case["solve"]["max_newton_iterations"])

    class Budget(GenericProblemHooks):
        def __init__(self):
            super().__init__()
            self.updates = self.remeshes = 0
            self.stage_updates = 0
            self.active = True
            self.remeshing = False
            self.converged = False
            self.remesh_after_convergence = False
            self.relaxation = initial_relaxation
            self.trials = []

        def remaining_updates(self):
            return min(stage_limit - self.stage_updates, 160 - self.updates)

        def actions_before_newton_solve(self):
            if not self.active:
                return
            self.converged = False
            remaining = self.remaining_updates()
            if remaining <= 0:
                raise RuntimeError("confined polish exhausted its stage or total Newton budget")
            problem.max_newton_iterations = remaining
            problem.newton_relaxation_factor = self.relaxation
            self.trials.append({"relaxation": self.relaxation, "remaining_updates": remaining,
                                "stage_updates_before": self.stage_updates,
                                "updates_total_before": self.updates,
                                "inversion_reports_before": problem._get_inversion_reports()})

        def actions_after_newton_step(self):
            if self.active:
                self.updates += 1
                self.stage_updates += 1

        def actions_after_newton_solve(self):
            if self.active:
                self.converged = True

        def actions_before_remeshing(self, active_remeshers):
            if self.active and self.remeshes >= 3:
                raise RuntimeError("confined polish exhausted its three-remesh budget")
            self.remeshing = self.active
            self.remesh_after_convergence = self.active and self.converged
            self.converged = False

        def actions_after_remeshing(self):
            # load_state also calls this hook, without actions_before_remeshing.
            if self.active and self.remeshing:
                self.remeshing = False
                self.remeshes += 1
                if self.remesh_after_convergence:
                    self.stage_updates = 0
                else:
                    self.relaxation *= 0.5
                self.remesh_after_convergence = False

    settings = {name: getattr(problem, name) for name in (
        "max_newton_iterations", "newton_relaxation_factor", "newton_solver_tolerance",
        "inversion_remesh_max_retries", "eigen_data_in_states")}
    flags = {name: getattr(problem, name) for name in (
        "_last_bc_setting", "_first_step", "_taken_already_an_unsteady_step",
        "_last_step_was_stationary")}
    steady = [problem.time_stepper_pt(i).is_steady() for i in range(problem.ntime_stepper())]
    parameters = {name: float(problem.get_global_parameter(name).value) for name in ("Ca_in", "Bo")}
    fields = [np.asarray(value).copy() for value in problem.get_current_dofs()]
    profile = [np.asarray(value).copy() for value in problem.interface_polyline()]
    volume = problem.observable("volume", "gas")
    frame = problem.frame_speed()
    initial_residual = _stationary_residual(problem)
    problem.eigen_data_in_states = True  # Also stores the BC mode in this pin.
    try:
        snapshot = problem._snapshot_state()
        directory = Path(output) / "polish" / label
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = directory / "before.dump"
        with checkpoint.open("xb") as stream:
            stream.write(snapshot)
    except Exception:  # No trial may run without a complete snapshot.
        problem.eigen_data_in_states = settings["eigen_data_in_states"]
        raise
    counter = Budget()
    problem += counter
    problem.inversion_remesh_max_retries = 3
    record = {"checkpoint": str(checkpoint.relative_to(output)),
              "checkpoint_sha256": hashlib.sha256(snapshot).hexdigest(),
              "initial_residual_linf": initial_residual, "parameters": parameters,
              "status": "running", "trials": counter.trials}
    try:
        while counter.remaining_updates() > 0:
            before_remeshes = counter.remeshes
            problem.solve(max_newton_iterations=counter.remaining_updates(),
                          newton_relaxation_factor=counter.relaxation, do_not_set_IC=True)
            residual = _stationary_residual(problem)
            record["trials"][-1].update(residual_linf=residual, status="returned",
                                        updates_total=counter.updates, remeshes_total=counter.remeshes)
            if parameters != {name: float(problem.get_global_parameter(name).value) for name in parameters}:
                raise RuntimeError("confined polish changed its physical forcing")
            if residual <= case["tolerances"]["newton"]:
                record.update(status="converged", residual_linf=residual)
                return record
            if counter.remeshes == before_remeshes:
                raise RuntimeError("returned solve is not stationary and has no evidenced remesh")
        raise RuntimeError("confined polish exhausted its stage or total Newton budget")
    except Exception as error:  # Rollback, then propagate every failure.
        record.update(status="failed", error=f"{type(error).__name__}: {error}", rollback_verified=False)
        counter.active = False
        for name, value in flags.items():
            setattr(problem, name, value)
        problem.load_state(io.BytesIO(snapshot), quiet=True)
        for name, value in flags.items():
            setattr(problem, name, value)
        for index, was_steady in enumerate(steady):
            stepper = problem.time_stepper_pt(index)
            stepper.make_steady() if was_steady else stepper.undo_make_steady()
        problem._domains_to_remesh.clear()
        problem._remesh_requested_inside_solve = False
        problem.invalidate_cached_mesh_data()
        restored_fields = problem.get_current_dofs()
        restored_profile = problem.interface_polyline()
        restored = (all(np.array_equal(a, b) for a, b in zip(fields, restored_fields, strict=True))
                    and all(np.array_equal(a, b) for a, b in zip(profile, restored_profile, strict=True))
                    and parameters == {name: float(problem.get_global_parameter(name).value) for name in parameters}
                    and volume == problem.observable("volume", "gas") and frame == problem.frame_speed()
                    and math.isclose(_stationary_residual(problem), initial_residual, rel_tol=1e-12, abs_tol=1e-12))
        record["rollback_verified"] = restored
        if not restored:
            raise RuntimeError("confined polish rollback failed its same-state checks") from error
        raise
    finally:
        counter.active = False
        for name, value in settings.items():
            setattr(problem, name, value)
        record.update(updates_total=counter.updates, stage_updates=counter.stage_updates,
                      remeshes_total=counter.remeshes)
        with (directory / "polish.json").open("w") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")


def stationary_bond_schedule(case):
    """Retain the original repeated-addition Bond ladder and its step budget."""
    solve, physics = case["solve"], case["physics"]
    bond = float(solve["initial_bond"])
    schedule = [bond]
    for _ in range(solve["max_bond_steps"]):
        if bond >= physics["bond"]:
            break
        bond = min(bond + solve["bond_step"], float(physics["bond"]))
        schedule.append(bond)
    if bond != physics["bond"]:
        raise ValueError("target Bond number not reached within max_bond_steps")
    return schedule


def stationary_layout(problem, azimuthal_stability):
    """Record the full native DOF labels and enforce the requested AZ setup."""
    import numpy as np

    mode = problem.is_normal_mode_stability_set_up()
    parameter = problem._azimuthal_mode_param_m
    expected = "azimuthal" if azimuthal_stability else False
    if mode != expected or (azimuthal_stability and (parameter is None or parameter.value != 0)):
        raise ValueError("loaded state does not retain the selected base-state azimuthal setup")
    types, names = problem.get_dof_description()
    types = np.asarray(types, dtype="<i8")
    names = [str(name) for name in names]
    if types.shape != (problem.ndof(),) or np.any(types < 0) or np.any(types >= len(names)):
        raise ValueError("loaded state has an incomplete native DOF description")
    if azimuthal_stability and any(not any(name.startswith(domain + "/")
            and name.endswith("/velocity_phi") for name in names) for domain in ("liquid", "gas")):
        raise ValueError("loaded state lacks a phase's azimuthal velocity layout")
    digest = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode() + types.tobytes())
    return {"ndof": problem.ndof(), "mode_setup": mode, "dof_names": names,
            "dof_layout_sha256": digest.hexdigest(), "base_mode": 0 if parameter is not None else None}


def _stationary_state_signature(problem, azimuthal_stability):
    """Capture actual values plus all native serialized fields, pins and histories."""
    import numpy as np

    from runners.run_paper_case import snapshot_bc_cache_signature

    dofs, positional_dofs = problem.get_current_dofs()
    z, r = problem.interface_polyline()
    # Preserve the legacy interface hash representation; native positions remain raw.
    z, r = (np.asarray([round(float(value), 12) for value in axis]) for axis in (z, r))
    arrays = {"free_dofs": dofs, "positional_dof_mask": positional_dofs,
              "pinned_values_with_positions": problem.get_current_pinned_values(True),
              "interface_z": z, "interface_r": r}
    eigen_setting = problem.eigen_data_in_states
    try:
        problem.eigen_data_in_states = True
        snapshot = problem._snapshot_state()
        serialized = snapshot_bc_cache_signature(snapshot, mesh_reference_base=Path.cwd())
    finally:
        problem.eigen_data_in_states = eigen_setting
    if serialized["bc_cache_marker"] != problem._last_bc_setting:
        raise ValueError("native snapshot changed its literal BC cache marker")
    return {"arrays": {name: {"shape": list(np.asarray(value).shape),
                             "dtype": str(np.asarray(value).dtype),
                             "sha256": hashlib.sha256(np.asarray(value).tobytes()).hexdigest()}
                        for name, value in arrays.items()},
            "serialized_state": serialized,
            "parameters": {name: float(problem.get_global_parameter(name).value) for name in ("Ca_in", "Bo")},
            "volume": problem.observable("volume", "gas"), "frame_speed": problem.frame_speed(),
            "time": float(problem.get_current_time(dimensional=False, as_float=True)),
            "bc_cache_marker": problem._last_bc_setting,
            "layout": stationary_layout(problem, azimuthal_stability)}


def _prepare_stationary_bc_state(problem, azimuthal_stability):
    """Use the native stationary hook and prove it preserves the loaded state.

    _last_bc_setting is a cache label that may remain ``init`` after a steady
    solve. Never assign it to pass a gate. Ordinary stationary preparation may
    update only that serialized label; all other fields, pins, layout, forcing
    and native history bytes must remain identical.
    """
    before = _stationary_state_signature(problem, azimuthal_stability)
    problem.actions_before_stationary_solve(force_reassign_eqs=False)
    after = _stationary_state_signature(problem, azimuthal_stability)

    def physical_state(signature):
        value = {key: item for key, item in signature.items() if key != "bc_cache_marker"}
        value["serialized_state"] = {"payload_sha256": signature["serialized_state"]["payload_sha256"]}
        return value

    if physical_state(before) != physical_state(after):
        raise ValueError("stationary BC preparation changed the loaded fields, pins, layout or history")
    return {"verified": True, "method": "actions_before_stationary_solve(force_reassign_eqs=False)",
            "before": before, "after": after}


def _checkpoint_stationary_state(problem, case, output, bond_index, azimuthal_stability):
    """Save a self-contained post-polish state and prove its native round trip.

    A receipt is written exclusively only after fields, positional masks, pinned values, profile,
    parameters and full layout round-trip exactly. Steady/history flags are
    restored after the load notification; inactive polish hooks stay inactive.
    These are stationary restart checks, not a transient equivalence claim.
    """
    import numpy as np
    from pyoomph.output.states import copy_state_file

    from runners.run_paper_case import (
        RESUME_SCHEMA,
        _exclusive_json,
        _read_json,
        append_progress,
        sha256_file,
        state_bc_cache_marker,
        state_mesh_references,
    )

    output = Path(output).resolve()
    manifest_path = output / "run-manifest.json"
    manifest = _read_json(manifest_path)
    bond = stationary_bond_schedule(case)[bond_index]
    parameters = {name: float(problem.get_global_parameter(name).value) for name in ("Ca_in", "Bo")}
    if parameters != {"Ca_in": float(case["physics"]["ca_in"]), "Bo": bond}:
        raise ValueError("stationary checkpoint forcing differs from its case stage")
    bc_preparation = _prepare_stationary_bc_state(problem, azimuthal_stability)
    residual = _stationary_residual(problem)
    if residual > case["tolerances"]["newton"]:
        raise ValueError("stationary checkpoint requires a converged bound polish")
    before_signature = _stationary_state_signature(problem, azimuthal_stability)
    bc_marker = problem._last_bc_setting
    layout = stationary_layout(problem, azimuthal_stability)
    fields = [np.asarray(value).copy() for value in problem.get_current_dofs()]
    pinned_values = np.asarray(problem.get_current_pinned_values(True)).copy()
    profile = [np.asarray(value).copy() for value in problem.interface_polyline()]
    volume, speed = problem.observable("volume", "gas"), problem.frame_speed()
    flags = {name: getattr(problem, name) for name in (
        "_first_step", "_taken_already_an_unsteady_step", "_last_step_was_stationary")}
    steady = [problem.time_stepper_pt(i).is_steady() for i in range(problem.ntime_stepper())]
    eigen_setting = problem.eigen_data_in_states
    directory = output / "checkpoints" / f"bond_{bond_index:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    transfer = output / "solver" / f"stationary_{bond_index:03d}.dump"
    if transfer.exists():
        raise ValueError("stationary checkpoint transfer is occupied")
    state = directory / "state.dump"
    try:
        problem.eigen_data_in_states = True  # Saves the literal BC cache marker at this pin.
        problem.save_state(str(transfer), quiet=True)
        if state_bc_cache_marker(transfer) != bc_marker:
            raise ValueError("stationary checkpoint serialized a different BC cache marker")
        original_meshes = [(transfer.parent / value).resolve() for value in state_mesh_references(transfer)]
        original_hashes = [sha256_file(path) for path in original_meshes]
        copy_state_file(str(transfer), str(state), copy_mesh_files=True)
        references = state_mesh_references(state)
        if state_bc_cache_marker(state) != bc_marker:
            raise ValueError("stationary checkpoint relocation changed its BC cache marker")
        meshes = []
        for reference in references:
            path = (state.parent / reference).resolve()
            if not path.is_relative_to(directory) or path.is_symlink() or not path.is_file():
                raise ValueError("stationary checkpoint failed to retain a referenced mesh")
            meshes.append({"stored_path": reference, "path": str(path), "sha256": sha256_file(path)})
        if [mesh["sha256"] for mesh in meshes] != original_hashes:
            raise ValueError("stationary checkpoint mesh copy differs from its source")
        for path in (state, *(Path(mesh["path"]) for mesh in meshes)):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        problem.load_state(str(state), quiet=True, ignore_outstep=True)
        if (not all(np.array_equal(a, b) for a, b in zip(fields, problem.get_current_dofs(), strict=True))
                or not np.array_equal(pinned_values, problem.get_current_pinned_values(True))
                or not all(np.array_equal(a, b) for a, b in zip(profile, problem.interface_polyline(), strict=True))
                or parameters != {name: float(problem.get_global_parameter(name).value) for name in parameters}
                or layout != stationary_layout(problem, azimuthal_stability)
                or problem._last_bc_setting != bc_marker
                or volume != problem.observable("volume", "gas") or speed != problem.frame_speed()
                or not math.isclose(_stationary_residual(problem), residual, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("stationary checkpoint failed its native same-state round trip")
        after_signature = _stationary_state_signature(problem, azimuthal_stability)
        before_payload = {**before_signature, "serialized_state": {
            "payload_sha256": before_signature["serialized_state"]["payload_sha256"]}}
        after_payload = {**after_signature, "serialized_state": {
            "payload_sha256": after_signature["serialized_state"]["payload_sha256"]}}
        if before_payload != after_payload:
            raise ValueError("stationary checkpoint changed its fields, pins, layout or history on load")
        # The full fingerprint replaces only mesh-reference strings with their
        # content identity and separates the literal BC cache trailer. Every
        # other serialized field and history byte must round-trip exactly.
        problem.eigen_data_in_states = True
        snapshot = problem._snapshot_state()
        problem._restore_state(snapshot)
        restored_signature = _stationary_state_signature(problem, azimuthal_stability)
        if after_signature != restored_signature:
            raise ValueError("stationary checkpoint failed its full serialized-state restoration")
        packet = {"schema": RESUME_SCHEMA,
            "parent_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "case_sha256": manifest["case_sha256"], "azimuthal_stability": azimuthal_stability,
            "state": {"path": str(state), "sha256": sha256_file(state), "format": "portable_file"},
            "mesh_reference_base": str(directory), "meshes": meshes,
            "stage": {"kind": "completed_polish", "bond_index": bond_index, "bond": bond,
                      "ca_in": parameters["Ca_in"], "bc_mode": bc_marker},
            "roundtrip": {"verified": True, "stationary_residual_linf": residual, "layout": layout,
                          "stationary_residual_operator": "native steppers temporarily steady",
                          "bc_preparation": bc_preparation,
                          "before": before_signature, "after": after_signature,
                          "snapshot_restore": {"verified": True,
                                               "before": after_signature, "after": restored_signature}}}
        receipt = directory / "resume.json"
        _exclusive_json(receipt, packet)
        append_progress(output, {"event": "stationary_checkpoint", "bond_index": bond_index,
            "bond": bond, "stationary_residual_linf": residual,
            "resume_packet": str(receipt.relative_to(output)), "resume_packet_sha256": sha256_file(receipt)})
        return {"bond_index": bond_index, "bond": bond, "resume_packet": str(receipt.relative_to(output))}
    finally:
        problem.eigen_data_in_states = eigen_setting
        for name, value in flags.items():
            setattr(problem, name, value)
        for index, was_steady in enumerate(steady):
            stepper = problem.time_stepper_pt(index)
            stepper.make_steady() if was_steady else stepper.undo_make_steady()


def _load_stationary_restart(problem, case, output, restart, azimuthal_stability):
    from runners.run_paper_case import _exclusive_json, append_progress, verify_restart_artifacts

    verify_restart_artifacts(restart)
    problem.load_state(restart["state"]["path"], quiet=True,
                       additional_info={"exchange_msh_file": restart["mesh_exchange"]})
    parameters = {name: float(problem.get_global_parameter(name).value) for name in ("Ca_in", "Bo")}
    stage = restart["stage"]
    if parameters != {"Ca_in": stage["ca_in"], "Bo": stage["bond"]}:
        raise ValueError("loaded stationary restart differs from the bound forcing")
    if problem._last_bc_setting != stage["bc_mode"]:
        raise ValueError("loaded stationary restart differs from the bound BC cache marker")
    loaded_marker = problem._last_bc_setting
    layout = stationary_layout(problem, azimuthal_stability)
    bc_preparation = _prepare_stationary_bc_state(problem, azimuthal_stability)
    residual = _stationary_residual(problem)
    if stage["kind"] == "completed_polish":
        roundtrip = restart.get("roundtrip", {})
        if (roundtrip.get("verified") is not True or roundtrip.get("layout") != layout
                or residual > case["tolerances"]["newton"]):
            raise ValueError("loaded post-polish checkpoint is not a qualified stationary state")
    before = _stationary_state_signature(problem, azimuthal_stability)
    eigen_setting = problem.eigen_data_in_states
    flags = {name: getattr(problem, name) for name in (
        "_first_step", "_taken_already_an_unsteady_step", "_last_step_was_stationary")}
    steady = [problem.time_stepper_pt(i).is_steady() for i in range(problem.ntime_stepper())]
    try:
        problem.eigen_data_in_states = True
        snapshot = problem._snapshot_state()
        problem._restore_state(snapshot)
        after = _stationary_state_signature(problem, azimuthal_stability)
        if before != after:
            raise ValueError("loaded restart failed its full in-process snapshot restoration")
    finally:
        problem.eigen_data_in_states = eigen_setting
        for name, value in flags.items():
            setattr(problem, name, value)
        for index, was_steady in enumerate(steady):
            stepper = problem.time_stepper_pt(index)
            stepper.make_steady() if was_steady else stepper.undo_make_steady()
    receipt = {"schema": "herrada-stationary-resume-load-v1", "verified": True,
               "parameters": parameters, "bc_mode": problem._last_bc_setting, "layout": layout,
               "loaded_bc_cache_marker": loaded_marker, "bc_preparation": bc_preparation,
               "stationary_residual_linf": residual,
               "stationary_residual_operator": "native steppers temporarily steady",
               "parent_manifest": restart["parent_manifest"], "state": restart["state"], "stage": stage,
               "mesh_exchange": restart["mesh_exchange"],
               "snapshot_restore": {"verified": True, "before": before, "after": after,
                                    "snapshot_sha256": hashlib.sha256(snapshot).hexdigest()}}
    _exclusive_json(Path(output) / "restart-load.json", receipt)
    append_progress(Path(output), {"event": "restart_loaded", "bond_index": stage["bond_index"],
                                  "bond": stage["bond"], "receipt": "restart-load.json"})


def run_model(case, output, *, azimuthal_stability=False, restart=None):
    """Execute one bounded stationary case after its runner writes provenance."""
    if not isinstance(azimuthal_stability, bool):
        raise TypeError("azimuthal_stability must be a boolean")
    import numpy as np
    from pyoomph.output.states import copy_state_file

    case = validate_case(case)
    output = Path(output)
    p, solve, tol = case["physics"], case["solve"], case["tolerances"]
    if not (output / "run-manifest.json").is_file():
        raise ValueError("runner manifest must precede native execution")
    route = []
    with build_problem(case, initial_bond=solve["initial_bond"]) as problem:
        problem.set_output_directory(str(output / "solver"))
        problem.set_linear_solver("superlu")
        problem.set_c_compiler("system")
        if azimuthal_stability:
            problem.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        problem.initialise()
        schedule = stationary_bond_schedule(case)
        start_index = 0
        if restart is not None:
            _load_stationary_restart(problem, case, output, restart, azimuthal_stability)
            start_index = restart["stage"]["bond_index"]
            route.append({"restart": restart["stage"]})
            if restart["stage"]["kind"] == "pending_polish":
                route.append(polish_confined_state(problem, case, output, f"resume_{start_index:03d}",
                    initial_relaxation=0.5 if start_index == 0 else 1.0))
                route.append(_checkpoint_stationary_state(problem, case, output,
                                                          start_index, azimuthal_stability))
        elif solve["relax_time"]:
            problem.run(solve["relax_time"], startstep=solve["start_step"],
                outstep=case["output"]["cadence"], maxstep=solve["relax_time"] / 8,
                temporal_error=tol["temporal_error"])
            route.append("transient seed")
        if restart is None:
            if p["problem"] == "finite_gas_bubble":
                route.append(polish_confined_state(problem, case, output, "seed"))
                _checkpoint_stationary_state(problem, case, output, 0, azimuthal_stability)
            else:
                problem.solve(max_newton_iterations=solve["max_newton_iterations"])
        bond = schedule[start_index]
        for bond_index in range(start_index + 1, len(schedule)):
            bond = schedule[bond_index]
            problem.get_global_parameter("Bo").value = bond
            if p["problem"] == "finite_gas_bubble":
                route.append(polish_confined_state(problem, case, output, f"bond_{len(route):03d}",
                                                    initial_relaxation=1.0))
                _checkpoint_stationary_state(problem, case, output, bond_index, azimuthal_stability)
            else:
                problem.solve(max_newton_iterations=solve["max_newton_iterations"])
            route.append({"bond": bond})
        if bond != p["bond"]:
            raise ValueError("target Bond number not reached within max_bond_steps")
        residual = _stationary_residual(problem)
        lower_flux = problem.observable("flux", "liquid/lower")
        upper_flux = problem.observable("flux", "liquid/upper")
        pressure_rms = math.sqrt(max(problem.observable("pressure_squared", "liquid/lower")
                                    / problem.observable("area", "liquid/lower"), 0))
        liquid_velocity_rms = bulk_velocity_rms(problem.observable("velocity_squared"),
                                                problem.observable("volume"))
        summary = {"status": "completed", "completion_reached": False,
            "azimuthal_stability": azimuthal_stability,
            "problem": p["problem"], "route": route, "ca_in": float(p["ca_in"]),
            "bond": bond, "frame_speed": problem.frame_speed(),
            "liquid_velocity_rms": liquid_velocity_rms, "gas_velocity_rms": None,
            "stationary_residual_linf": residual, "lower_pressure_rms": pressure_rms,
            "lower_flux": lower_flux, "upper_flux": upper_flux,
            "flux_absolute_error": abs(lower_flux - upper_flux),
            "lower_velocity_completion": "zero radial and azimuthal velocity; free axial velocity"}
        checks = {"stationary": residual <= tol["newton"],
                  "lower_static_pressure": pressure_rms <= tol["pressure_boundary_absolute"],
                  "flux": abs(lower_flux - upper_flux) <= tol["flux_absolute"]}
        if p["problem"] == "bare_tube":
            velocity_error = math.sqrt(max(problem.observable("velocity_error_squared"), 0)
                / max(problem.observable("velocity_reference_squared"), 1e-300))
            pressure_error = math.sqrt(max(problem.observable("pressure_error_squared"), 0)
                / max(problem.observable("pressure_reference_squared"), 1e-300))
            summary.update(velocity_relative_l2=velocity_error, pressure_relative_l2=pressure_error)
            checks.update(velocity_reference=velocity_error <= tol["velocity_reference_relative"],
                          pressure_reference=pressure_error <= tol["pressure_reference_relative"])
        else:
            gas_volume = problem.observable("volume", "gas")
            surface_volume = problem.observable("surface_volume", "liquid/interface")
            volume_error = abs(gas_volume / problem.geometry.volume - 1)
            summary.update(gas_volume=gas_volume, surface_volume=surface_volume,
                gas_velocity_rms=bulk_velocity_rms(problem.observable("velocity_squared", "gas"),
                                                 gas_volume),
                volume_relative_error=volume_error,
                gas_pressure_mean=problem.observable("pressure", "gas") / gas_volume,
                gas_volume_multiplier=float(problem.get_ode("gas_volume").get_value("pressure", as_float=True)))
            checks.update(volume=volume_error <= tol["volume_relative"],
                volume_integrals=abs(surface_volume / gas_volume - 1) <= tol["volume_relative"])
            z, r = problem.interface_polyline()
            with (output / "interface.csv").open("x") as stream:
                stream.write("z_solver,r_solver\n")
                for zi, ri in zip(z, r, strict=True):
                    stream.write(f"{float(zi):.17g},{float(ri):.17g}\n")
            if p["problem"] == "static_laplace":
                centre = problem.geometry.z_front - problem.geometry.r0
                radius_error = float(np.max(np.abs(np.hypot(z - centre, r) - problem.geometry.r0)))
                jump = summary["gas_pressure_mean"] - problem.observable("pressure") / problem.observable("volume")
                laplace_error = abs(jump / (2 / problem.geometry.r0) - 1)
                summary.update(sphere_radius_max_error=radius_error, pressure_jump=jump,
                               laplace_relative_error=laplace_error)
                checks.update(sphere=radius_error <= tol["sphere_radius_absolute"],
                              laplace=laplace_error <= tol["laplace_relative"],
                              liquid_static_velocity=liquid_velocity_rms <= tol["static_velocity_absolute"],
                              gas_static_velocity=summary["gas_velocity_rms"] <= tol["static_velocity_absolute"],
                              static_frame_speed=math.isfinite(summary["frame_speed"]) and
                              abs(summary["frame_speed"]) <= tol["static_velocity_absolute"])
        transfer = output / "solver" / "final.dump"
        destination = output / "checkpoints" / "final.dump"
        destination.parent.mkdir(exist_ok=True)
        problem.save_state(str(transfer), quiet=True)
        copy_state_file(str(transfer), str(destination), copy_mesh_files=True)
        summary.update(checks=checks, completion_reached=all(checks.values()),
            scientific_status="checks_passed" if all(checks.values()) else "checks_failed",
            checkpoint=str(destination.relative_to(output)))
    return summary
