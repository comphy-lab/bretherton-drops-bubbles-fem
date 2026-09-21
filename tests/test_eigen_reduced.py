"""Contract tests for the reduced-pencil eigen tooling.

None of these needs a bubble mesh. The pencil is built by hand so that its
finite spectrum is known before the solver is asked for it, which is the only
way a stability tool can fail a test rather than restate its own arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from eigen_helpers import (
    BorderedLU,
    eigenpair_residual,
    eigenvector_support,
    is_physical,
    reduced_finite_spectrum,
)


def stokes_like_pencil(nx: int = 5, seed: int = 7):
    """A singular-M pencil whose finite spectrum is known before solving.

    Unknowns, in three groups of ``nx``:

    * ``x`` -- interface positions, the only columns M touches;
    * ``q`` -- a second field, closed by the constraint ``q = W x``;
    * ``z`` -- a pressure-like field, closed by ``z = V x``.

    The kinematic rows are the only rows M touches, and they read
    ``(A x + B q + C z) = lambda S x``. Substituting the two constraints gives
    ``(A + B W + C V) x = lambda S x`` exactly, so the finite spectrum is the
    ordinary spectrum of ``S^{-1} (A + B W + C V)``: ``nx`` eigenvalues, and
    ``2 nx`` infinite ones carried by the M-nullspace.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((nx, nx))
    B = rng.standard_normal((nx, nx))
    C = rng.standard_normal((nx, nx))
    W = rng.standard_normal((nx, nx))
    V = rng.standard_normal((nx, nx))
    S = np.diag(0.5 + rng.random(nx))
    n = 3 * nx
    X = slice(0, nx)            # interface positions
    Q = slice(nx, 2 * nx)       # second field
    Z = slice(2 * nx, 3 * nx)   # pressure-like field
    J = np.zeros((n, n))
    # kinematic rows live on the X block of rows, which is where M is nonzero
    J[X, X], J[X, Q], J[X, Z] = A, B, C
    J[Q, X], J[Q, Q] = -W, np.eye(nx)
    J[Z, X], J[Z, Z] = -V, np.eye(nx)
    M = np.zeros((n, n))
    M[X, X] = S
    truth = np.linalg.eigvals(np.linalg.solve(S, A + B @ W + C @ V))
    kind = np.array(["other"] * nx + ["velocity"] * nx + ["pressure"] * nx)
    return sp.csr_matrix(J), sp.csr_matrix(M), kind, truth


def test_finite_spectrum_matches_the_analytic_reduction():
    J, M, kind, truth = stokes_like_pencil()
    nx = truth.size
    out = reduced_finite_spectrum(J, M, kind, n_vectors=None, refine=3)

    assert out["n_mass_rows"] == nx
    assert np.linalg.matrix_rank(M.toarray()) == nx
    assert out["n_finite"] == nx

    got = np.sort_complex(out["eigenvalues"])
    want = np.sort_complex(truth)
    assert np.allclose(got, want, rtol=1e-8, atol=1e-8)


def test_finite_spectrum_eigenpairs_pass_the_residual_screen():
    J, M, kind, truth = stokes_like_pencil()
    out = reduced_finite_spectrum(J, M, kind, n_vectors=None, refine=3)
    vectors = out["eigenvectors"]
    assert vectors.shape[1] == truth.size
    for i, lam in enumerate(out["eigenvalues"]):
        v = vectors[:, i]
        assert np.linalg.norm(v) > 0.0
        assert is_physical(J, M, complex(lam), v, tol=1e-8)
    assert all(out["physical"])


def test_residual_screen_rejects_a_wrong_eigenvalue_and_the_mass_nullspace():
    J, M, kind, _ = stokes_like_pencil()
    out = reduced_finite_spectrum(J, M, kind, n_vectors=None, refine=3)
    lam = complex(out["eigenvalues"][0])
    v = out["eigenvectors"][:, 0]
    assert not is_physical(J, M, lam + 1.0, v)
    # A vector in the nullspace of M has no mass action at all.
    nullish = np.zeros(J.shape[0])
    nullish[-1] = 1.0
    assert np.linalg.norm(M @ nullish) == 0.0
    assert not is_physical(J, M, lam, nullish)


def test_bordered_solver_reproduces_a_dense_solve_with_dense_rows():
    rng = np.random.default_rng(11)
    n = 60
    core = sp.diags([np.full(n - 1, -1.0), np.full(n, 4.0), np.full(n - 1, -1.0)],
                    offsets=[-1, 0, 1], format="lil")
    # Two rows and two columns that touch everything, as the pressure-datum
    # multiplier and the global dofs do in the real pencil.
    dense_rows = [0, n - 1]
    for r in dense_rows:
        core[r, :] = rng.standard_normal(n)
        core[:, r] = rng.standard_normal((n, 1))
    A = sp.csr_matrix(core)
    dense = A.toarray()
    assert np.linalg.cond(dense) < 1e12

    solver = BorderedLU(A, dense_threshold=10)
    assert solver.nb == len(dense_rows)

    b = rng.standard_normal((n, 3))
    got = solver.solve(b, refine=3)
    want = np.linalg.solve(dense, b)
    assert np.allclose(got, want, rtol=1e-8, atol=1e-10)

    single = rng.standard_normal(n)
    assert np.allclose(solver.solve(single, refine=3),
                       np.linalg.solve(dense, single), rtol=1e-8, atol=1e-10)


def test_bordered_solver_handles_a_complex_matrix():
    rng = np.random.default_rng(3)
    n = 24
    A = sp.csr_matrix(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    solver = BorderedLU(A, dense_threshold=1000)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert np.allclose(solver.solve(b, refine=3), np.linalg.solve(A.toarray(), b),
                       rtol=1e-8, atol=1e-10)


class _StubProblem:
    """Just enough of a problem to answer get_dof_description()."""

    def __init__(self, types, names):
        self._types = types
        self._names = names

    def get_dof_description(self):
        return self._types, self._names


def test_eigenvector_support_survives_an_over_long_vector():
    # Azimuthal k >= 1: pyoomph releases the strong axis conditions, so the
    # pencil is larger than ndof and the tail is the released axis dofs.
    types = np.array([0, 0, 1, 1, 2])
    names = ["liquid/pressure", "liquid/velocity_x", "liquid/interface/mesh_y"]
    problem = _StubProblem(types, names)
    vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])

    out = eigenvector_support(problem, vector)

    assert "error" not in out
    assert out["vector_length"] == 7
    assert out["ndof_described"] == 5
    assert out["truncated_description"] is True
    assert out["released_axis"] == pytest.approx(8.0 / 13.0)
    assert out["pressure"] == pytest.approx(2.0 / 13.0)
    total = sum(out[k] for k in ("pressure", "globals", "mesh", "velocity",
                                 "other", "released_axis"))
    assert total == pytest.approx(1.0)


def test_eigenvector_support_still_reports_an_exact_vector():
    types = np.array([0, 1, 2])
    names = ["liquid/pressure", "liquid/velocity_x", "liquid/interface/mesh_y"]
    out = eigenvector_support(_StubProblem(types, names), np.array([1.0, 1.0, 1.0]))
    assert out["truncated_description"] is False
    assert out["released_axis"] == 0.0
    assert out["pressure"] == pytest.approx(1.0 / 3.0)


def test_eigenpair_residual_is_scale_free():
    J, M, kind, _ = stokes_like_pencil()
    out = reduced_finite_spectrum(J, M, kind, n_vectors=1, refine=3)
    lam = complex(out["eigenvalues"][0])
    v = out["eigenvectors"][:, 0]
    a = eigenpair_residual(J, M, lam, v)
    b = eigenpair_residual(J, M, lam, 1e6 * v)
    assert b["residual_scale_free"] == pytest.approx(a["residual_scale_free"],
                                                     rel=1e-6, abs=1e-12)
