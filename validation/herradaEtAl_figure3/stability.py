"""Analyse the axisymmetric spectrum of one saved Figure 3 steady station.

Example::

    uv run --no-sync python -m validation.herradaEtAl_figure3.stability \
      --run-dir /path/to/accepted/run --station 80 --m 0 \
      --shift -0.01 --shift 0.01 --neigen 8 \
      --output-dir /path/to/new/stability-result

The fresh output directory contains ``status.json`` with source hashes, the
fixation-gauge steady observables and finite-eigenpair diagnostics, plus one
``eigenvectors_shift_XX.npz`` per shift for later mode-shape analysis. This is
a finite-spectrum diagnostic, not a stability classification or a sweep. The
void model and imposed end conditions remain those of the steady driver.
Nonzero azimuthal modes require a steady checkpoint written from an azimuthal
formulation and are rejected by this saved-state driver.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np

from validation.herradaEtAl_figure3 import driver

SCHEMA = "herrada-figure3-stability-v1"
EXPECTED_PYOOMPH_COMMIT = "a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618"
OBSERVABLES = (
    "bond", "ca", "bubble_length_over_r", "signed_ca_b", "volume_error",
    "flux_error", "min_gap", "axial_span_over_r",
)


def validate_request(station: int, mode: int, shifts: list[float], neigen: int,
                     residual_tolerance: float) -> None:
    if station < 0 or mode not in (0, 1, 2, 3):
        raise ValueError("station must be non-negative and m must be 0, 1, 2 or 3")
    if not 1 <= neigen <= 24:
        raise ValueError("neigen must be between 1 and 24")
    if not shifts or len(set(shifts)) != len(shifts) or any(
        not math.isfinite(shift) or shift == 0 for shift in shifts
    ):
        raise ValueError("provide distinct, finite, non-zero shifts")
    if not 0 < residual_tolerance <= 1e-6:
        raise ValueError("residual tolerance must be positive and at most 1e-6")


def compare_steady_observables(loaded: dict, source: dict) -> dict[str, float]:
    """Fail if a changed gauge/linearisation has moved the accepted steady state."""
    differences = {}
    changed = []
    for key in OBSERVABLES:
        actual, reference = float(loaded[key]), float(source[key])
        if not math.isfinite(actual) or not math.isfinite(reference):
            raise ValueError(f"non-finite steady observable: {key}")
        differences[key] = actual - reference
        # The gap is a nodal geometric minimum, not a solved global constraint.
        # Allow one Newton tolerance in R0 units after stationary re-polishing.
        absolute_tolerance = 1e-8 if key == "min_gap" else 1e-9
        if not math.isclose(actual, reference, rel_tol=1e-8,
                            abs_tol=absolute_tolerance):
            changed.append(f"{key}: {actual} vs {reference}")
    if changed:
        raise ValueError("stability-gauge steady state changed " + "; ".join(changed))
    return differences


def diagnose_eigenpairs(values, vectors, jacobian, mass, *, mode: int,
                        residual_tolerance: float) -> list[dict]:
    """Check finite generalised eigenpairs of ``J v = lambda M v``.

    The pinned pyoomph eigensolver returns ``-dR/dU`` as J, so this sign agrees
    with its ``lambda M v + dR/dU v = 0`` convention. Matrix manipulators have
    already imposed mode-dependent zero-DOF constraints on both matrices.
    """
    values = np.asarray(values, dtype=np.complex128)
    vectors = np.asarray(vectors, dtype=np.complex128)
    if values.ndim != 1 or not len(values) or vectors.shape != (len(values), mass.shape[1]):
        raise ValueError("eigenvalue/vector dimensions disagree with the pencil")
    if jacobian.shape != mass.shape or jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError("stability diagnostic requires serial square J and M")
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError("non-finite eigenvalue or eigenvector")
    if not np.isfinite(jacobian.data).all() or not np.isfinite(mass.data).all():
        raise ValueError("non-finite Jacobian or mass matrix")
    if mass.nnz == 0 or not np.any(mass.data):
        raise ValueError("empty mass matrix")
    mass_scale = float(np.max(np.abs(mass.data)))
    jacobian_norm = float(np.linalg.norm(jacobian.data))
    mass_norm = float(np.linalg.norm(mass.data))
    if jacobian_norm == 0:
        raise ValueError("empty Jacobian")
    records = []
    for index, (value, vector) in enumerate(zip(values, vectors, strict=True)):
        vnorm = float(np.linalg.norm(vector))
        jv, mv = jacobian @ vector, mass @ vector
        jnorm, mnorm = float(np.linalg.norm(jv)), float(np.linalg.norm(mv))
        absolute = float(np.linalg.norm(jv - value * mv))
        relative = absolute / max(jnorm + abs(value) * mnorm, 1e-300)
        backward = absolute / max(
            (jacobian_norm + abs(value) * mass_norm) * vnorm, 1e-300
        )
        stationary_null = jnorm / max(jacobian_norm * vnorm, 1e-300)
        mass_action = mnorm / max(mass_scale * vnorm, 1e-300)
        if not all(map(math.isfinite, (
            vnorm, jnorm, mnorm, absolute, relative, backward,
            stationary_null, mass_action,
        ))):
            raise ValueError(f"non-finite eigenpair diagnostic at index {index}")
        if vnorm == 0 or mass_action <= 1e-12:
            raise ValueError(f"zero/near-null mass action at eigenpair {index}")
        if backward > residual_tolerance:
            raise ValueError(f"eigenpair {index} backward residual {backward:g} exceeds {residual_tolerance:g}")
        records.append({
            "index": index, "m": mode, "lambda_real": float(value.real),
            "lambda_imag": float(value.imag), "absolute_residual_l2": absolute,
            "relative_residual_l2": relative, "backward_residual_l2": backward,
            "stationary_null_residual_l2": stationary_null,
            "mass_action_ratio": mass_action,
            "vector_norm_l2": vnorm,
        })
    return records


def mode_components(problem, vectors) -> list[list[dict]]:
    """Record where each eigenvector's coefficient norm sits, without a mode claim."""
    dof_types, names = problem.get_dof_description()
    if len(dof_types) != vectors.shape[1]:
        raise ValueError("DOF labels do not match eigenvector length")
    components = []
    for vector in vectors:
        weights = np.bincount(dof_types, weights=np.abs(vector) ** 2, minlength=len(names))
        total = float(np.sum(weights))
        if total <= 0 or not math.isfinite(total):
            raise ValueError("eigenvector has no finite component weight")
        top = np.argsort(weights)[-6:][::-1]
        components.append([{"field": str(names[i]), "coefficient_fraction": float(weights[i] / total)}
                           for i in top if weights[i] > 0])
    return components


def assemble_current_mode_pencil(problem, mode: int):
    """Reassemble the same mode and constrained rows that pyoomph just solved."""
    parameter = problem._azimuthal_mode_param_m
    if parameter is None and mode != 0:
        raise RuntimeError("azimuthal stability was not set up before initialisation")
    old_value = parameter.value if parameter is not None else None
    steppers = [problem.time_stepper_pt(index) for index in range(problem.ntime_stepper())]
    was_steady = [stepper.is_steady() for stepper in steppers]
    try:
        for stepper in steppers:
            stepper.make_steady()
        if parameter is not None:
            parameter.value = mode
        problem.actions_before_eigen_solve()
        problem.setup_forced_zero_dof_list_for_eigenproblems()
        return problem.get_eigen_solver().get_J_M_n_and_type()
    finally:
        if parameter is not None:
            parameter.value = old_value
        for stepper, steady in zip(steppers, was_steady, strict=True):
            if not steady:
                stepper.undo_make_steady()


def run(run_dir: Path, station: int, mode: int, shifts: list[float], neigen: int,
        residual_tolerance: float, output: Path) -> int:
    validate_request(station, mode, shifts, neigen, residual_tolerance)
    if mode != 0:
        raise ValueError(
            "nonzero azimuthal modes require a stationary checkpoint written "
            "by the integral-gauge azimuthal formulation; this saved-state "
            "driver currently accepts only the original axisymmetric m=0 state"
        )
    case_path = run_dir / "case.json"
    case = driver.read_case(case_path, "run")
    growth_rate_factor = float(case["physics"]["r_tube"]) / float(case["physics"]["r0"])
    rows = driver.read_branches(run_dir / "branches.csv")
    if station >= len(rows):
        raise ValueError("station is absent from the accepted branch table")
    state = run_dir / "checkpoints" / f"station_{station:05d}.dump"
    if not state.is_file():
        raise FileNotFoundError(f"saved steady station is absent: {state}")
    source_commit = subprocess.check_output(
        ["git", "-C", str(driver.ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    source_dirty = bool(subprocess.check_output(
        ["git", "-C", str(driver.ROOT), "status", "--porcelain"], text=True
    ).strip())
    driver.prepare_output(output)
    receipt = {
        "schema": SCHEMA, "status": "running", "source_commit": source_commit,
        "source_dirty": source_dirty, "expected_pyoomph_commit": EXPECTED_PYOOMPH_COMMIT,
        "source_run": str(run_dir.resolve()), "station": station, "m": mode,
        "case_sha256": driver.file_digest(case_path),
        "branches_sha256": driver.file_digest(run_dir / "branches.csv"),
        "state_sha256": driver.file_digest(state), "shifts": shifts,
        "neigen_requested": neigen, "residual_tolerance": residual_tolerance,
        "paper_growth_rate_factor": growth_rate_factor,
        "pressure_datum": "fixation",
        "eigenproblem": "axisymmetric_base",
        "eigensolver": "spectra", "spectra": [],
    }
    driver.atomic_json(output / "status.json", receipt)
    try:
        driver._require_one_rank()
        # The saved m=0 state was written by the fixation-gauge base problem.
        # A stability-enabled problem adds an azimuthal parameter, and the
        # integral datum changes the constrained state. Pyoomph correctly
        # rejects a dump from the different stationary formulation.
        # The ordinary axisymmetric eigenproblem is the m=0 meridional pencil.
        problem = driver.build_problem(
            case, ca=float(case["physics"]["seed_ca"]), bond=0.0,
            pressure_datum="fixation",
        )
        problem.set_output_directory(str(output / "solver"))
        problem.set_eigensolver("spectra")
        with problem:
            problem.initialise()
            problem.load_state(str(state), quiet=True, ignore_eigendata=True,
                               ignore_continuation_data=True)
            # Re-polish the stationary equations before comparing the saved state.
            problem.solve(max_newton_iterations=20)
            loaded, _ = driver.extract_row(problem, station, case["acceptance"])
            receipt["steady_observable_differences"] = compare_steady_observables(
                loaded, rows[station]
            )
            receipt["steady_state"] = loaded
            receipt["degrees_of_freedom"] = problem.ndof()
            for index, shift in enumerate(shifts):
                values, vectors = problem.solve_eigenproblem(
                    neigen, shift=shift, quiet=True,
                )
                if len(values) == 0:
                    raise ValueError(f"no finite eigenpairs returned near shift {shift}")
                jacobian, mass, n, is_complex = assemble_current_mode_pencil(problem, mode)
                if n != problem.ndof():
                    raise ValueError("assembled eigenproblem size differs from current DOF count")
                diagnostics = diagnose_eigenpairs(values, vectors, jacobian, mass,
                                                  mode=mode,
                                                  residual_tolerance=residual_tolerance)
                for item in diagnostics:
                    item["paper_omega_i"] = growth_rate_factor * item["lambda_real"]
                for item, fields in zip(diagnostics, mode_components(problem, vectors), strict=True):
                    item["dominant_dof_groups"] = fields
                filename = f"eigenvectors_shift_{index:02d}.npz"
                np.savez_compressed(output / filename, eigenvalues=values, eigenvectors=vectors)
                receipt["spectra"].append({
                    "shift": shift, "matrix_complex": bool(is_complex),
                    "matrix_nnz": {"J": int(jacobian.nnz), "M": int(mass.nnz)},
                    "eigenvectors_file": filename, "eigenpairs": diagnostics,
                })
                driver.atomic_json(output / "status.json", receipt)
        receipt["status"] = "complete"
        driver.atomic_json(output / "status.json", receipt)
        return 0
    except Exception as exc:  # noqa: BLE001 - retain an explicit failed-run receipt
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        driver.atomic_json(output / "status.json", receipt)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="accepted Figure 3 run containing case.json, branches.csv and checkpoints")
    parser.add_argument("--station", required=True, type=int)
    parser.add_argument("--m", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--shift", required=True, action="append", type=float)
    parser.add_argument("--neigen", type=int, default=8)
    parser.add_argument("--residual-tolerance", type=float, default=1e-7)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return run(args.run_dir.resolve(), args.station, args.m, args.shift, args.neigen,
               args.residual_tolerance, args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
