"""Independent nose-up lubrication model of Yu et al. (2021), JFM 911 A34.

Equations (2.5), (2.9), (2.12) and section 2.3, pages 5, 6, 8:
https://doi.org/10.1017/jfm.2020.1027
The model assumes a long bubble, thin liquid film, negligible gas stress and
inertia, with full axisymmetric curvature. Volume is not a local model input.
All lengths below use the tube radius; velocities use surface tension/viscosity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.integrate import DOP853
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
PAPER = {
    "citation": "Yu, Magnini, Zhu, Shim & Stone (2021), JFM 911 A34",
    "doi": "10.1017/jfm.2020.1027",
    "pdf_url": "https://www.cambridge.org/core/services/aop-cambridge-core/content/view/FC5B85459313B31ADEB4AA1C2C1A8526/S0022112020010277a.pdf/non-unique-bubble-dynamics-in-a-vertical-capillary-with-an-external-flow.pdf",
    "equations": ["2.5", "2.9", "2.12"],
    "section": "2.3",
    "pages": [5, 6, 8],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Controls:
    amplitude: float = 1e-6
    pole_radius: float = 1e-3
    rtol: float = 1e-10
    atol: float = 1e-12
    alpha_lower: float = 0.001
    alpha_upper: float = 0.3
    root_xtol: float = 2e-13
    root_rtol: float = 1e-13
    root_iterations: int = 40
    max_arclength: float = 40.0
    max_step: float = 0.05
    radial_step_fraction: float = 0.2
    max_steps: int = 2000
    max_rhs_per_trial: int = 20000
    dense_subdivisions: int = 16
    pole_relative_gate: float = 1e-4

    def validate(self):
        require(
            all(math.isfinite(v) and v > 0 for v in asdict(self).values()),
            "controls must be finite and positive",
        )
        require(
            0 < self.alpha_lower < self.alpha_upper < 1 - self.pole_radius,
            "invalid physical shooting bracket",
        )
        require(
            self.amplitude < 0.01 and self.pole_radius < 0.01 and self.radial_step_fraction < 0.25,
            "invalid asymptotic or event controls",
        )
        return self


class Budget:
    def __init__(self, *, seconds=120.0, max_trials=60, max_rhs=250000):
        self.started = time.monotonic()
        self.seconds, self.max_trials, self.max_rhs = seconds, max_trials, max_rhs
        self.trials = self.rhs_calls = 0

    def check(self):
        if time.monotonic() - self.started > self.seconds:
            raise RuntimeError("lubrication wall-time budget exhausted")

    def trial(self):
        self.check()
        self.trials += 1
        if self.trials > self.max_trials:
            raise RuntimeError("lubrication shooting-trial budget exhausted")

    def rhs(self):
        self.check()
        self.rhs_calls += 1
        if self.rhs_calls > self.max_rhs:
            raise RuntimeError("lubrication RHS budget exhausted")


def validate_input(bo, ca):
    require(
        not isinstance(bo, bool)
        and not isinstance(ca, bool)
        and isinstance(bo, (int, float))
        and isinstance(ca, (int, float))
        and math.isfinite(bo)
        and math.isfinite(ca)
        and bo >= 0
        and ca > 0,
        "nose-up model requires finite Bo>=0 and Ca_l>0",
    )


def bubble_speed(alpha, bo, ca):
    """Yu equation (2.9), retaining the cylindrical flux closure."""
    require(0 < alpha < 1, "film fraction must lie in (0,1)")
    radius = 1 - alpha
    drainage = -0.5 + 3 * radius**2 / 8 + 1 / (8 * radius**2) - 0.5 * radius**2 * math.log1p(-alpha)
    return ca / radius**2 + bo * drainage


def growing_exponent(alpha, cab, bo):
    """Linearise equation (2.5) about its flat-film equilibrium."""
    coefficient = 3 * cab / alpha**3 + 3 * bo / alpha
    require(coefficient > 0, "flat film has no selected positive growing exponent")
    return brentq(
        lambda k: k**3 + k / (1 - alpha) ** 2 - coefficient,
        0.0,
        coefficient ** (1 / 3) + 1,
        xtol=1e-14,
        rtol=1e-14,
    )


def start_state(alpha, cab, bo, amplitude):
    require(0 < amplitude < 0.01, "a nonzero asymptotic growing-mode start is required")
    k = growing_exponent(alpha, cab, bo)
    dh = alpha * amplitude
    return np.array(
        [alpha + dh, 0.0, 1 / (1 - alpha) + (k * k + 1 / (1 - alpha) ** 2) * dh, k * dh]
    ), k


def forcing(h, alpha, cab, bo):
    return 3 * cab * (h - alpha) / h**3 + bo * (1 - (alpha / h) ** 3)


def geometry_rhs(state, alpha, cab, bo):
    """Arclength equations equivalent to Yu (2.5), state=(h,z,K,theta).

    theta=atan(h_z). Thus K=cos(theta)/(1-h)+theta_s. At a regular axis pole,
    cos(theta)/(1-h) tends to theta_s, giving K=2*theta_s.
    """
    h, _, curvature, angle = state
    require(0 < h < 1, "ODE stage left the physical radial domain")
    sine, cosine = math.sin(angle), math.cos(angle)
    return np.array(
        [sine, cosine, forcing(h, alpha, cab, bo) * cosine, curvature - cosine / (1 - h)]
    )


def pole_residual(state, alpha, cab, bo, *, correct=False):
    """Regular pole has r*cos(theta)-K*r²/2=0 as r→0.

    Expanding dK/dr=-F*cot(theta) at the pole gives the finite-radius term
    F*K*r⁴/16. Subtract it only at a radius cutoff, never at an early turn.
    """
    h, _, curvature, angle = state
    radius = 1 - h
    result = radius * math.cos(angle) - curvature * radius**2 / 2
    if correct:
        result -= forcing(h, alpha, cab, bo) * curvature * radius**4 / 16
    return float(result)


def shoot(alpha, bo, ca, controls, budget, log, emit=None):
    controls.validate()
    budget.trial()
    record = {"alpha": float(alpha), "controls": asdict(controls), "status": "starting"}
    log.append(record)
    calls = 0
    try:
        cab = bubble_speed(alpha, bo, ca)
        initial, k = start_state(alpha, cab, bo, controls.amplitude)

        def rhs(_, value):
            nonlocal calls
            calls += 1
            budget.rhs()
            if calls > controls.max_rhs_per_trial:
                raise RuntimeError("one shooting trial exhausted its RHS budget")
            return geometry_rhs(value, alpha, cab, bo)

        solver = DOP853(
            rhs,
            0.0,
            initial,
            controls.max_arclength,
            rtol=controls.rtol,
            atol=controls.atol,
            max_step=controls.max_step,
        )
        segments = []
        event, stop = None, None
        events = (
            ("radius-cutoff", lambda y: 1 - y[0] - controls.pole_radius),
            ("early-vertical", lambda y: math.pi / 2 - y[3]),
            ("turn", lambda y: y[3]),
        )
        for step in range(controls.max_steps):
            before_t, before_y = solver.t, solver.y.copy()
            # Limit attempted steps by the remaining radius; keep the physical RHS guard active.
            solver.max_step = min(
                controls.max_step, controls.radial_step_fraction * (1 - solver.y[0])
            )
            solver.step()
            require(solver.status != "failed", "DOP853 shooting trial failed")
            dense = solver.dense_output()
            located = []
            for name, function in events:
                if function(before_y) > 0 and function(solver.y) <= 0:
                    location = brentq(
                        lambda s, fn=function, interpolate=dense: fn(interpolate(s)),
                        before_t,
                        solver.t,
                        xtol=1e-14,
                        rtol=1e-14,
                    )
                    located.append((location, name))
            if located:
                stop, event = min(located)
                segments.append((before_t, stop, dense))
                endpoint = dense(stop)
                break
            segments.append((before_t, solver.t, dense))
            if solver.status != "running":
                break
        require(
            event is not None, "shooting trial reached no physical event within its arc/step bounds"
        )
        residual = pole_residual(endpoint, alpha, cab, bo, correct=event == "radius-cutoff")
        record.update(
            status="returned",
            event=event,
            residual=residual,
            cab=cab,
            growing_exponent=k,
            rhs_calls=calls,
            accepted_steps=step + 1,
            endpoint=endpoint.tolist(),
            arclength=float(stop),
        )
        if emit:
            emit("shooting-trial", trial=record)
        return {
            "alpha": float(alpha),
            "bo": bo,
            "cab": cab,
            "endpoint": endpoint,
            "event": event,
            "residual": residual,
            "segments": segments,
            "record": record,
        }
    except BaseException as error:
        record.update(
            status="failed", error_type=type(error).__name__, error=str(error), rhs_calls=calls
        )
        if emit:
            emit("shooting-trial", trial=record)
        raise


def curve_from_shot(shot, controls):
    parts = []
    for index, (a, b, dense) in enumerate(shot["segments"]):
        s = np.linspace(a, b, controls.dense_subdivisions + 1)
        if index:
            s = s[1:]
        parts.append(np.column_stack((s, dense(s).T)))
    curve = np.vstack(parts)  # s,h,z,K,theta
    h, z, curvature, _ = shot["endpoint"]
    radius = 1 - h
    pole_z = (
        z + curvature * radius**2 / 4
    )  # local regular-cap expansion, tested by cutoff refinement
    curve[:, 2] -= pole_z
    pole_curvature = curvature * (
        1 + forcing(h, shot["alpha"], shot["cab"], shot["bo"]) * radius**2 / 4
    )
    pole_arc = radius + curvature**2 * radius**3 / 24
    pole = np.array([[curve[-1, 0] + pole_arc, 1.0, 0.0, pole_curvature, math.pi / 2]])
    curve = np.vstack((curve, pole))
    require(
        np.isfinite(curve).all()
        and np.all(np.diff(curve[:, 1]) > 0)
        and np.all(np.diff(curve[:, 2]) > 0),
        "selected nose curve is not monotone/finite",
    )
    return curve


def solve_nose(bo, ca, controls, budget, log, emit=None):
    cache = {}

    def evaluate(alpha):
        if alpha not in cache:
            cache[alpha] = shoot(alpha, bo, ca, controls, budget, log, emit)
        return cache[alpha]["residual"]

    left, right = evaluate(controls.alpha_lower), evaluate(controls.alpha_upper)
    require(left < 0 < right, "model shooting events do not bracket a regular nose")
    alpha, root = brentq(
        evaluate,
        controls.alpha_lower,
        controls.alpha_upper,
        xtol=controls.root_xtol,
        rtol=controls.root_rtol,
        maxiter=controls.root_iterations,
        full_output=True,
        disp=False,
    )
    require(root.converged, "film-fraction root did not converge")
    evaluate(alpha)
    selected = cache[alpha]
    h, _, curvature, angle = selected["endpoint"]
    meridional = curvature - math.cos(angle) / (1 - h)
    defect = float(abs(curvature - 2 * meridional) / abs(curvature))
    checks = {
        "root_converged": bool(root.converged),
        "radius_cutoff_reached": selected["event"] == "radius-cutoff",
        "regular_pole_relative_defect": defect <= controls.pole_relative_gate,
        "positive_finite_curvature": math.isfinite(curvature) and curvature > 0,
        "nose_up_angle": 0 < angle < math.pi / 2,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    require(all(checks.values()), "selected film fraction failed its regular-pole checks")
    return {
        "alpha": alpha,
        "cab": selected["cab"],
        "controls": asdict(controls),
        "checks": checks,
        "pole_relative_defect": defect,
        "shooting_residual": selected["residual"],
        "root_iterations": root.iterations,
        "curve_normalisation": "lengths in tube radii; regular pole aligned to z=0",
        "last_curve_row": "regular-pole asymptotic extension from the recorded finite radius cutoff",
        "curve": curve_from_shot(selected, controls),
    }


def calculate(bo, ca, *, budget=None, emit=None, log=None):
    validate_input(bo, ca)
    budget = budget or Budget()
    log = [] if log is None else log
    base = Controls()
    variants = [
        ("baseline", base),
        ("start-amplitude", replace(base, amplitude=1e-7)),
        ("pole-and-integrator-refinement", replace(base, pole_radius=5e-4, rtol=1e-11, atol=1e-13)),
    ]
    solutions = {}
    for name, controls in variants:
        solutions[name] = solve_nose(bo, ca, controls, budget, log, emit)
    baseline = solutions["baseline"]
    alpha_delta = max(abs(s["alpha"] - baseline["alpha"]) for s in solutions.values())
    cab_delta = max(abs(s["cab"] - baseline["cab"]) for s in solutions.values())
    # Compare the reference figure's displayed nose range h/R<=0.5; interpolation sensitivity
    # is included here and is separate from ODE local error and model approximation error.
    left = max(s["curve"][0, 2] for s in solutions.values())
    right = min(
        float(np.interp(0.5, s["curve"][:, 1], s["curve"][:, 2])) for s in solutions.values()
    )
    require(left < right, "nose curves have no common displayed comparison interval")
    grid = np.linspace(left, right, 1000)
    reference = np.interp(grid, baseline["curve"][:, 2], baseline["curve"][:, 1])
    curve_delta = max(
        float(np.max(np.abs(np.interp(grid, s["curve"][:, 2], s["curve"][:, 1]) - reference)))
        for s in solutions.values()
    )
    checks = {
        "alpha_absolute_sensitivity": alpha_delta <= 1e-6,
        "cab_absolute_sensitivity": cab_delta <= 1e-6,
        "displayed_nose_gap_sensitivity": curve_delta <= 5e-6,
    }
    return {
        "schema": "herrada-yu-lubrication-nose-v1",
        "status": "complete" if all(checks.values()) else "failed",
        "bo": bo,
        "ca_l": ca,
        "alpha": baseline["alpha"],
        "ca_b": baseline["cab"],
        "numerical_acceptance": all(checks.values()),
        "checks": checks,
        "solutions": solutions,
        "sensitivity": {
            "alpha_absolute": alpha_delta,
            "cab_absolute": cab_delta,
            "displayed_gap_max_absolute": curve_delta,
            "profile_z_interval": [left, right],
            "interpretation": "same-model numerical sensitivity; no bound on lubrication model error",
        },
        "budget": {
            "seconds_limit": budget.seconds,
            "max_trials": budget.max_trials,
            "max_rhs_calls": budget.max_rhs,
            "trials": budget.trials,
            "rhs_calls": budget.rhs_calls,
            "elapsed_seconds": time.monotonic() - budget.started,
        },
        "claim": "independent model comparison; no FEM, finite-volume or physical validation",
        "assumptions": [
            "long bubble",
            "thin film",
            "shear-free negligible gas stress",
            "Re=0",
            "full curvature",
        ],
        "volume_input": None,
    }


def atomic(path, data):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    atomic(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def source_provenance():
    script = Path(__file__).resolve()

    def git(*arguments):
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(ROOT), *arguments],
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout

    head = git("rev-parse", "HEAD").decode().strip()
    require(
        not git("status", "--porcelain", "--untracked-files=all"),
        "production model source must be clean and committed",
    )
    actual = hashlib.sha256(script.read_bytes()).hexdigest()
    require(
        actual == hashlib.sha256(git("show", f"{head}:{script.relative_to(ROOT)}")).hexdigest(),
        "model runner differs from commit",
    )
    return {
        "component_commit": head,
        "runner": {"path": str(script), "sha256": actual},
        "python": {"executable": sys.executable, "version": sys.version},
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "paper": PAPER,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bo", type=float, required=True)
    parser.add_argument("--ca", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="input/source custody only; no ODE calculation/output"
    )
    args = parser.parse_args(argv)
    validate_input(args.bo, args.ca)
    require(not args.out.is_symlink(), "output symlink refused")
    output = args.out.resolve()
    require(not output.is_relative_to(ROOT), "model outputs must stay outside the component")
    require(
        not output.exists() or (output.is_dir() and not any(output.iterdir())), "output is occupied"
    )
    provenance = source_provenance()
    if args.check:
        print(json.dumps({"verified": True, "source": provenance, "ode_performed": False}))
        return 0
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(directory, 0o700)
    finally:
        os.close(directory)
    budget = Budget()
    write_json(
        output / "run-manifest.json",
        {
            "schema": "independent-lubrication-manifest-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "bo": args.bo,
            "ca_l": args.ca,
            "source": provenance,
            "controls": asdict(Controls()),
            "claim_kind": "model_comparison",
            "budget": {
                "seconds": budget.seconds,
                "trials": budget.max_trials,
                "rhs_calls": budget.max_rhs,
            },
            "sensitivity_controls": {
                "start_amplitude": 1e-7,
                "refined_pole_radius": 5e-4,
                "refined_rtol": 1e-11,
                "refined_atol": 1e-13,
            },
            "volume_is_local_input": False,
            "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        },
    )
    descriptor = os.open(
        output / "progress.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    events = os.fdopen(descriptor, "wb")
    trials = []

    def emit(event, **payload):
        events.write(
            (
                json.dumps(
                    {"timestamp": datetime.now(UTC).isoformat(), "event": event, **payload},
                    allow_nan=False,
                )
                + "\n"
            ).encode()
        )
        events.flush()

    try:
        emit("prepared", numerical_acceptance=False)
        result = calculate(args.bo, args.ca, budget=budget, emit=emit, log=trials)
        for name, solution in result["solutions"].items():
            curve = solution.pop("curve")
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(
                ["s_over_R", "h_over_R", "z_tip_aligned_over_R", "curvature_times_R", "theta"]
            )
            writer.writerows(curve)
            atomic(output / f"nose-{name}.csv", stream.getvalue().encode())
        write_json(output / "result.json", result)
        write_json(output / "trials.json", trials)
        require(result["numerical_acceptance"], "lubrication numerical sensitivity checks failed")
        require(provenance == source_provenance(), "model source changed during calculation")
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file() and p.name != "progress.jsonl"
        }
        write_json(
            output / "completion.json",
            {
                "status": "complete",
                "files": files,
                "numerical_acceptance": True,
                "claim_kind": "model_comparison",
            },
        )
        emit("final", status="complete", numerical_acceptance=True)
        return 0
    except BaseException as error:
        if not (output / "trials.json").exists():
            write_json(output / "trials.json", trials)
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "numerical_acceptance": False,
                "trial_count": len(trials),
                "claim_kind": "model_comparison",
            },
        )
        emit("failure", status="failed", error=str(error), numerical_acceptance=False)
        raise
    finally:
        events.close()


if __name__ == "__main__":
    raise SystemExit(main())
