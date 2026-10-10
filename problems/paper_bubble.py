"""Two-phase Stokes model with upper flow and lower static-pressure ports.

The material ratios and pressure conversion follow Herrada, Yu & Stone
(2023), doi:10.1017/jfm.2023.123. The lower radial velocity is set to zero;
that completion is explicit because the article specifies only its pressure.
Case validation imports no native solver. Native definitions are loaded only
when a caller constructs a problem.
"""

from __future__ import annotations

import copy
import json
import math
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
    if case.get("benchmark_schema") != "herrada-paper-model-v1":
        errors.append("benchmark_schema must be herrada-paper-model-v1")
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
                       "pressure_reference_relative", "laplace_relative", "sphere_radius_absolute"},
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
                 "pressure_reference_relative", "laplace_relative", "sphere_radius_absolute"):
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
        grad,
        matproduct,
        sym,
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
            velocity = var(flow.velocity_name, domain=self.get_parent_domain())
            normal = self.get_normal()
            viscous = 2 * flow.dynamic_viscosity * sym(grad(velocity))
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
                    liquid += RemeshWhen(RemeshingOptions())
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


def run_model(case, output):
    """Execute one bounded stationary case after its runner writes provenance."""
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
        problem.initialise()
        if solve["relax_time"]:
            problem.run(solve["relax_time"], startstep=solve["start_step"],
                outstep=case["output"]["cadence"], maxstep=solve["relax_time"] / 8,
                temporal_error=tol["temporal_error"])
            route.append("transient seed")
        problem.solve(max_newton_iterations=solve["max_newton_iterations"])
        bond = float(solve["initial_bond"])
        for _ in range(solve["max_bond_steps"]):
            if bond >= p["bond"]:
                break
            bond = min(bond + solve["bond_step"], float(p["bond"]))
            problem.get_global_parameter("Bo").value = bond
            problem.solve(max_newton_iterations=solve["max_newton_iterations"])
            route.append({"bond": bond})
        if bond != p["bond"]:
            raise ValueError("target Bond number not reached within max_bond_steps")
        residual = _stationary_residual(problem)
        lower_flux = problem.observable("flux", "liquid/lower")
        upper_flux = problem.observable("flux", "liquid/upper")
        pressure_rms = math.sqrt(max(problem.observable("pressure_squared", "liquid/lower")
                                    / problem.observable("area", "liquid/lower"), 0))
        summary = {"status": "completed", "completion_reached": False,
            "problem": p["problem"], "route": route, "ca_in": float(p["ca_in"]),
            "bond": bond, "frame_speed": problem.frame_speed(),
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
                              laplace=laplace_error <= tol["laplace_relative"])
        transfer = output / "solver" / "final.dump"
        destination = output / "checkpoints" / "final.dump"
        destination.parent.mkdir()
        problem.save_state(str(transfer), quiet=True)
        copy_state_file(str(transfer), str(destination), copy_mesh_files=True)
        summary.update(checks=checks, completion_reached=all(checks.values()),
            scientific_status="checks_passed" if all(checks.values()) else "checks_failed",
            checkpoint=str(destination.relative_to(output)))
    return summary
