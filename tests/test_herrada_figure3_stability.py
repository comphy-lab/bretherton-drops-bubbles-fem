"""Software contracts for the saved-state Figure 3 stability driver."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from validation.herradaEtAl_figure3 import stability


def test_request_is_bounded_and_cannot_target_the_exact_fold_shift():
    stability.validate_request(12, 1, [-0.01, 0.01], 8, 1e-7)
    for station, mode, shifts, neigen, tolerance in (
        (-1, 0, [0.01], 8, 1e-7),
        (1, 4, [0.01], 8, 1e-7),
        (1, 0, [0.0], 8, 1e-7),
        (1, 0, [0.01, 0.01], 8, 1e-7),
        (1, 0, [0.01], 25, 1e-7),
        (1, 0, [0.01], 8, 1e-5),
    ):
        with pytest.raises(ValueError):
            stability.validate_request(station, mode, shifts, neigen, tolerance)


def test_changed_gauge_must_preserve_every_steady_observable():
    source = {name: float(index + 1) for index, name in enumerate(stability.OBSERVABLES)}
    same = {name: value + 1e-11 for name, value in source.items()}
    assert all(abs(delta) < 1e-10 for delta in
               stability.compare_steady_observables(same, source).values())
    same["signed_ca_b"] += 1e-4
    with pytest.raises(ValueError, match="signed_ca_b"):
        stability.compare_steady_observables(same, source)


def test_generalised_residual_and_mass_action_reject_spurious_pairs():
    jacobian = csr_matrix(np.diag([2.0, 3.0]))
    mass = csr_matrix(np.diag([1.0, 1.0]))
    vectors = np.eye(2, dtype=complex)
    good = stability.diagnose_eigenpairs([2.0, 3.0], vectors, jacobian, mass,
                                         mode=0, residual_tolerance=1e-7)
    assert [record["relative_residual_l2"] for record in good] == [0.0, 0.0]
    assert [record["backward_residual_l2"] for record in good] == [0.0, 0.0]
    with pytest.raises(ValueError, match="residual"):
        stability.diagnose_eigenpairs([2.1], vectors[:1], jacobian, mass,
                                      mode=0, residual_tolerance=1e-7)
    with pytest.raises(ValueError, match="mass action"):
        stability.diagnose_eigenpairs([3.0], vectors[1:], jacobian,
                                      csr_matrix(np.diag([1.0, 0.0])),
                                      mode=0, residual_tolerance=1e-7)
    with pytest.raises(ValueError, match="non-finite"):
        stability.diagnose_eigenpairs([complex(float("inf"), 0)], vectors[:1],
                                      jacobian, mass, mode=0, residual_tolerance=1e-7)


def test_near_fold_zero_mode_uses_matrix_scaled_residual():
    jacobian = csr_matrix(np.diag([1e-12, 1.0]))
    mass = csr_matrix(np.eye(2))
    pair = stability.diagnose_eigenpairs(
        [0.0], np.array([[1.0, 0.0]]), jacobian, mass,
        mode=0, residual_tolerance=1e-7,
    )[0]
    assert pair["relative_residual_l2"] == 1.0
    assert pair["backward_residual_l2"] == pytest.approx(1e-12)
    assert pair["stationary_null_residual_l2"] == pytest.approx(1e-12)


def test_reassembled_pencil_uses_requested_mode_and_restores_parameter():
    class FakeProblem:
        _azimuthal_mode_param_m = SimpleNamespace(value=0)

        def actions_before_eigen_solve(self):
            self.assembled_mode = self._azimuthal_mode_param_m.value

        def get_eigen_solver(self):
            return SimpleNamespace(get_J_M_n_and_type=lambda: self.assembled_mode)

    problem = FakeProblem()
    assert stability.assemble_current_mode_pencil(problem, 1) == 1
    assert problem._azimuthal_mode_param_m.value == 0


def test_run_checks_loaded_steady_state_before_solving_eigenproblem(tmp_path, monkeypatch):
    from validation.herradaEtAl_figure3 import driver

    run_dir = tmp_path / "steady"
    (run_dir / "checkpoints").mkdir(parents=True)
    case = driver.read_case(driver.HERE / "case.json", "run")
    driver.atomic_json(run_dir / "case.json", case)
    source = {"station": 0, **{name: 1.0 for name in stability.OBSERVABLES}}
    driver.write_branches(run_dir / "branches.csv", [source])
    (run_dir / "checkpoints" / "station_00000.dump").write_bytes(b"mock-state")

    class FakeProblem:
        _azimuthal_mode_param_m = SimpleNamespace(value=0)

        def __init__(self):
            self.events = []
            self.solver = SimpleNamespace(get_J_M_n_and_type=lambda: (
                csr_matrix([[2.0]]), csr_matrix([[1.0]]), 1, False))

        def set_output_directory(self, value):
            self.events.append("output")

        def set_eigensolver(self, value):
            assert value == "spectra"
            self.events.append("backend")

        def setup_for_stability_analysis(self, *, azimuthal_stability):
            assert azimuthal_stability
            self.events.append("setup")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def initialise(self):
            self.events.append("initialise")

        def load_state(self, *_args, **_kwargs):
            self.events.append("load")

        def solve(self, **_kwargs):
            self.events.append("polish")

        def ndof(self):
            return 1

        def solve_eigenproblem(self, *_args, **_kwargs):
            self.events.append("eigensolve")
            return np.array([2.0]), np.array([[1.0]])

        def get_last_eigenmodes_m(self):
            return np.array([0])

        def actions_before_eigen_solve(self):
            self.events.append("reassemble")

        def get_eigen_solver(self):
            return self.solver

        def get_dof_description(self):
            return np.array([0]), ["liquid/velocity_y"]

    problem = FakeProblem()

    def build(_case, *, pressure_datum, **_kwargs):
        assert pressure_datum == "integral"
        return problem

    monkeypatch.setattr(driver, "build_problem", build)
    monkeypatch.setattr(driver, "_require_one_rank", lambda: None)
    monkeypatch.setattr(driver, "extract_row", lambda *_: (dict(source), None))
    output = tmp_path / "spectrum"
    assert stability.run(run_dir, 0, 0, [0.01], 1, 1e-7, output) == 0
    assert problem.events == ["output", "backend", "setup", "initialise", "load",
                              "polish", "eigensolve", "reassemble"]
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete"
    assert len(status["source_commit"]) == 40
    assert status["expected_pyoomph_commit"] == stability.EXPECTED_PYOOMPH_COMMIT
    assert status["spectra"][0]["eigenpairs"][0]["lambda_real"] == 2.0
    assert status["spectra"][0]["eigenpairs"][0]["paper_omega_i"] == pytest.approx(
        2.0 * case["physics"]["r_tube"] / case["physics"]["r0"]
    )

    monkeypatch.setattr(driver, "extract_row", lambda *_: (dict(source, ca=2.0), None))
    failed = tmp_path / "changed-steady"
    assert stability.run(run_dir, 0, 0, [0.01], 1, 1e-7, failed) == 2
    assert "changed ca" in json.loads((failed / "status.json").read_text())["error"]
