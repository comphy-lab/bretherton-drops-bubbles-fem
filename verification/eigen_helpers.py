"""Shared pieces for the azimuthal-stability stages.

Kept out of the problem class because these are campaign checks, not physics.
The wavenumber is ``azimuthal_k`` everywhere in receipts: the report already
uses ``m`` for the deposited-liquid fraction.

Two things in here are worth reading before use.

``finite_spectrum`` computes the *whole* finite spectrum of the void-bubble
pencil by an exact reduction rather than by a Krylov shift-invert. The mass
matrix carries entries only on the interface kinematic-condition rows against
the interface mesh-position columns, so its rank is a few hundred out of a few
hundred thousand and the pencil has that many finite eigenvalues; everything
else is the M-nullspace, which is what floods an ARPACK subspace near zero.

``is_physical`` is the verdict. A mode is accepted when M acts on it at all and
the scale-free residual ``||J v - lambda M v|| / ||lambda M v||`` is small.
The older support-by-dof screen is kept only as a diagnostic: see the note on
``eigenvector_support``.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from problems.bretherton_bubble import VoidBubbleProblem
from problems.geometry import CapsuleGeometry


SWEEP_CA = (
    0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
    0.1, 0.15, 0.2, 0.3, 0.4, 0.4788, 0.5, 0.7,
    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
)
FRESH_CA_LIMIT = 2.0
CONTINUATION_SEED = 0.7


def remesh_ceiling_for_film(film_elements: float, pad_rear: float = 1.5) -> float:
    """Highest Ca_in at which a forced remesh has been seen to resolve.

    Measured on S14: film=3 remeshes through Ca_in=5 and dies at 6;
    film=6 remeshes through 3 and dies at 4; doubled pads (rear=3)
    remesh through 2.5 and die at 3. A diverged remesh leaves garbage
    in the vector, so this is a ceiling, not a try-and-catch.
    """
    if pad_rear > 1.5 + 1e-12:
        return 2.5
    return 5.0 if film_elements <= 3.0 + 1e-12 else 3.0


def continuation_ladder(current: float, target: float) -> list[float]:
    """Committed sweep stations from ``current`` to ``target``, exclusive of start.

    A single ``0.7 → 6`` continuation stretches the seed mesh the same way the
    committed high-Ca rows did. Forcing a remesh from that wrecked geometry
    sends the first Newton residual to ``inf``. Remesh at stations at or
    below ``remesh_ceiling_for_film``, then continue without remesh.
    """
    if abs(target - current) <= 1e-12:
        return []
    if target > current:
        ladder = [c for c in SWEEP_CA if current + 1e-12 < c <= target + 1e-12]
    else:
        ladder = [c for c in reversed(SWEEP_CA)
                  if target - 1e-12 <= c < current - 1e-12]
    if not ladder or abs(ladder[-1] - target) > 1e-12:
        ladder.append(target)
    return ladder


def climb_with_forced_remesh(problem: VoidBubbleProblem, target: float,
                             *, max_step: float = 0.25,
                             remesh_ceiling: float | None = None) -> list[float]:
    """Continue to ``target``, remeshing only at stations at or below the ceiling."""
    if remesh_ceiling is None:
        remesh_ceiling = remesh_ceiling_for_film(
            problem.film_elements, problem.geometry.pad_rear)
    visited: list[float] = []
    for step in continuation_ladder(problem.current_ca_in(), target):
        problem.go_to_param(Ca_in=step, max_step=max_step)
        if step <= remesh_ceiling + 1e-12:
            problem.remesh_handler_during_continuation(force=True)
        visited.append(step)
    return visited


def load_sweep(path: Path) -> dict[float, dict]:
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    return {float(row["ca_in"]): row for row in payload["results"]}


def build_stability_problem(
    ca_in: float,
    *,
    film_elements: float,
    bulk_resolution: float,
    output_dir: Path,
    pressure_datum: str = "integral",
    remesh: bool = True,
    pad_rear: float = 1.5,
    pad_front: float = 3.0,
    laplace: float = 0.0,
    azimuthal_stability: bool = True,
    lip_elements: float | None = None,
    lip_floor_divisor: float = 16.0,
    lip_patch_radii: float = 10.0,
    lip_gradation: float = 0.25,
    z_inlet: float = 0.0,
) -> VoidBubbleProblem:
    if pressure_datum == "fixation" and azimuthal_stability:
        raise ValueError(
            "azimuthal eigensolving is refused with pressure_datum='fixation'; "
            "pass 'integral' as an argument, do not change the class default"
        )
    geom = CapsuleGeometry.for_capillary_number(
        ca_in, pad_rear=pad_rear, pad_front=pad_front, seed_from_bubble_speed=True,
        z_inlet=z_inlet)
    problem = VoidBubbleProblem(
        geometry=geom, ca_in=ca_in, laplace=laplace,
        film_elements=film_elements, lip_elements=lip_elements,
        lip_floor_divisor=lip_floor_divisor, lip_patch_radii=lip_patch_radii,
        lip_gradation=lip_gradation, bulk_resolution=bulk_resolution,
        closure="centroid", outlet="poiseuille",
        pressure_datum=pressure_datum, remesh=remesh,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    problem.set_output_directory(str(output_dir))
    problem.setup_for_stability_analysis(azimuthal_stability=azimuthal_stability)
    problem.initialise()
    problem.audit_constraints()
    return problem


def advective_rate(ca_b: float, r_tube: float) -> float:
    return abs(ca_b) / r_tube


def shift_sets(rate: float) -> tuple[list[float], list[float]]:
    """Two distinct shift-invert centres, scaled on this row's own rate.

    A shift-invert solve returns eigenvalues nearest the shift, not the largest
    real part. The S12 set {0, +/-0.05} was calibrated at one slow state; at
    Ca_b = 14.8 the advective rate is ~21 and eight eigenvalues nearest zero
    can all be a damped cluster.
    """
    rate = max(float(rate), 1e-8)
    set_a = [0.0, -0.25 * rate, 0.25 * rate]
    set_b = [-0.8 * rate, 0.8 * rate, 0.05 * rate]
    return set_a, set_b


def rightmost_sample(problem: VoidBubbleProblem, azimuthal_k: int | None, n: int,
                     shift: float, *, residual_tol: float = 1e-6) -> dict:
    """One shift-invert sample, with the physical screen applied on the spot.

    The eigensolver already assembles (J, M) and ``capture_pencil`` keeps them,
    so the residual screen of ``is_physical`` costs one matrix-vector product
    and no second assembly. The matrices are dropped again before returning:
    only the screen's numbers travel with the sample.
    """
    with capture_pencil(problem) as box:
        values, vectors = problem.solve_eigenproblem(
            n, azimuthal_m=azimuthal_k, shift=shift)
    index = int(max(range(len(values)), key=lambda i: values[i].real))
    lam = complex(values[index])
    screen = None
    physical = None
    pencil = box.get("pencil")
    if pencil is not None and pencil.J.shape[1] == np.asarray(vectors[index]).size:
        screen = eigenpair_residual(pencil.J, pencil.M, lam, vectors[index])
        physical = bool(screen["mass_action"] > 0.0
                        and screen["residual_scale_free"] < residual_tol)
    return {
        "residual_screen": screen,
        "physical": physical,
        "azimuthal_k": azimuthal_k,
        "n": n,
        "shift": float(shift),
        "real": lam.real,
        "imag": lam.imag,
        "index": index,
        "n_returned": int(len(values)),
        "positive_count": int(sum(1 for z in values if z.real > 0.0)),
        "near_zero_count": int(sum(1 for z in values if abs(z.real) < 1e-8)),
        "spectrum": [{"real": complex(z).real, "imag": complex(z).imag} for z in values],
        "vector": vectors[index],
        "values": values,
        "vectors": vectors,
    }


def _eig_close(a: dict, b: dict, *, rtol: float, atol: float) -> bool:
    """Same eigenvalue, treating conjugate partners as one pair."""
    scale = max(abs(a["real"]), abs(b["real"]), abs(a["imag"]), abs(b["imag"]), 1.0)
    tol = max(atol, rtol * scale)
    return (
        abs(a["real"] - b["real"]) <= tol
        and abs(abs(a["imag"]) - abs(b["imag"])) <= tol
    )


def consensus(samples: list[dict], *, rtol: float, atol: float) -> dict | None:
    if not samples:
        return None
    best = max(samples, key=lambda s: s["real"])
    if not all(_eig_close(s, best, rtol=rtol, atol=atol) for s in samples):
        return None
    spread = max(
        max(abs(s["real"] - best["real"]), abs(abs(s["imag"]) - abs(best["imag"])))
        for s in samples
    )
    return {
        "real": best["real"],
        "imag": best["imag"],
        "spread": spread,
        "n_samples": len(samples),
    }


def two_set_rightmost(
    problem: VoidBubbleProblem,
    azimuthal_k: int | None,
    *,
    n: int,
    rate: float,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> dict:
    set_a, set_b = shift_sets(rate)
    rows_a, rows_b = [], []
    failures = []
    for shift in set_a:
        try:
            rows_a.append(rightmost_sample(problem, azimuthal_k, n, shift))
        except Exception as exc:  # noqa: BLE001 - a refusal is the result
            failures.append(f"set_a shift={shift}: {type(exc).__name__}: {exc}")
    for shift in set_b:
        try:
            rows_b.append(rightmost_sample(problem, azimuthal_k, n, shift))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"set_b shift={shift}: {type(exc).__name__}: {exc}")
    cons_a = consensus(rows_a, rtol=rtol, atol=atol)
    cons_b = consensus(rows_b, rtol=rtol, atol=atol)
    agreed = None
    if cons_a and cons_b and _eig_close(cons_a, cons_b, rtol=rtol, atol=atol):
        agreed = cons_a if cons_a["real"] >= cons_b["real"] else cons_b
    return {
        "azimuthal_k": azimuthal_k,
        "n": n,
        "shift_set_a": set_a,
        "shift_set_b": set_b,
        "samples_a": [_public_sample(s) for s in rows_a],
        "samples_b": [_public_sample(s) for s in rows_b],
        "consensus_a": _strip_none(cons_a),
        "consensus_b": _strip_none(cons_b),
        "agreed": agreed,
        "failures": failures,
        "_private_a": rows_a,
        "_private_b": rows_b,
    }


def _public_sample(sample: dict) -> dict:
    out = {k: sample[k] for k in (
        "azimuthal_k", "n", "shift", "real", "imag", "index",
        "n_returned", "positive_count", "near_zero_count", "physical",
    )}
    screen = sample.get("residual_screen")
    if screen is not None:
        out["residual_scale_free"] = screen["residual_scale_free"]
        out["mass_action_over_v"] = screen["mass_action_over_v"]
    return out


def _strip_none(value):
    return value


# ---------------------------------------------------------------------------
# The pencil: assembly, exact reduction, and the residual verdict
# ---------------------------------------------------------------------------

DOF_KINDS = ("pressure", "globals", "mesh", "velocity", "other")


def classify_dof_name(name: str) -> str:
    """Bucket a pyoomph dof name. Diagnostic only; never a verdict."""
    low = str(name).lower()
    if low.endswith("/u") or low.endswith("/p") or "globals" in low:
        return "globals"
    if "pressure" in low:
        return "pressure"
    if "mesh" in low:
        return "mesh"
    if "velocity" in low:
        return "velocity"
    return "other"


@dataclass
class Pencil:
    """The matrices pyoomph's own eigensolver would have been handed.

    ``J`` and ``M`` are exactly ``GenericEigenSolver.get_J_M_n_and_type()``:
    ``matJ = -dR/dU`` and ``matM = dR/d(dU/dt)``, and the convention is
    ``matJ v = lambda matM v`` with growth ``exp(lambda t)``.

    ``types``/``names`` are the dof description **of this mode's numbering**.
    For ``azimuthal_k >= 1`` pyoomph releases the strong axis conditions and
    imposes them by matrix manipulation, so the pencil is complex and slightly
    larger than ``problem.ndof()``; the description has to be read while that
    numbering is live, which is why it is captured here and not afterwards.
    """

    J: sp.spmatrix
    M: sp.spmatrix
    n: int
    is_complex: bool
    types: np.ndarray
    names: list
    azimuthal_k: int | None = None
    kind: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.kind is None:
            self.kind = np.array(
                [classify_dof_name(self.names[int(t)]) for t in self.types])

    @property
    def dof_description(self) -> tuple:
        return self.types, self.names


class _PencilCaptured(Exception):
    """Raised out of the eigensolver once the matrices are in hand."""


@contextlib.contextmanager
def _steady_restored(problem):
    """Undo whatever ``solve_eigenproblem`` made steady on the way out."""
    nts = problem.ntime_stepper()
    was_steady = [problem.time_stepper_pt(i).is_steady() for i in range(nts)]
    try:
        yield
    finally:
        for i in range(nts):
            ts = problem.time_stepper_pt(i)
            if (not was_steady[i]) and ts.is_steady():
                ts.undo_make_steady()


def assemble_pencil(problem: VoidBubbleProblem, azimuthal_k: int | None = None) -> Pencil:
    """Assemble (J, M) for one azimuthal mode without solving anything.

    Takes the same route as ``problem.solve_eigenproblem``: the eigensolver is
    entered so that the renumbering and the normal-mode setup happen exactly as
    they would, and the matrices are taken from ``get_J_M_n_and_type()`` before
    the Krylov solver is ever reached.
    """
    es = problem.get_eigen_solver()
    cap: dict = {}
    original = es.solve

    def _capture(*args, **kwargs):
        J, M, n, is_complex = es.get_J_M_n_and_type()
        types, names = problem.get_dof_description()
        cap.update(J=J, M=M, n=n, is_complex=is_complex, types=types, names=names)
        raise _PencilCaptured()

    es.solve = _capture
    try:
        with _steady_restored(problem):
            try:
                problem.solve_eigenproblem(2, azimuthal_m=azimuthal_k, shift=1.0, quiet=True)
            except _PencilCaptured:
                pass
    finally:
        es.solve = original
    if "J" not in cap:
        raise RuntimeError("the eigensolver was never entered; no pencil to capture")
    return Pencil(
        J=cap["J"].tocsr(), M=cap["M"].tocsr(), n=int(cap["n"]),
        is_complex=bool(cap["is_complex"]),
        types=np.asarray(cap["types"]), names=list(cap["names"]),
        azimuthal_k=azimuthal_k,
    )


@contextlib.contextmanager
def capture_pencil(problem: VoidBubbleProblem):
    """Record the pencil a real ``solve_eigenproblem`` call uses, for free.

    ``eigen_solver.solve()`` already returns ``(values, vectors, J, M)`` and
    ``Problem._solve_eigenproblem_helper`` throws the matrices away. Wrapping
    the solver keeps them, together with the dof description of the numbering
    that call ran under, so a residual screen costs no second assembly.

    Yields a dict that holds a ``Pencil`` under ``"pencil"`` afterwards.
    """
    es = problem.get_eigen_solver()
    box: dict = {"pencil": None}
    original = es.solve

    def _wrapped(*args, **kwargs):
        types, names = problem.get_dof_description()
        out = original(*args, **kwargs)
        try:
            J, M = out[2], out[3]
            box["pencil"] = Pencil(
                J=sp.csr_matrix(J), M=sp.csr_matrix(M), n=int(J.shape[1]),
                is_complex=bool(np.iscomplexobj(J) or np.iscomplexobj(M)),
                types=np.asarray(types), names=list(names),
            )
        except Exception:  # noqa: BLE001 - a missing pencil is not a failed solve
            box["pencil"] = None
        return out

    es.solve = _wrapped
    try:
        yield box
    finally:
        es.solve = original


class BorderedLU:
    """Apply ``J^{-1}`` when a handful of dense rows/columns block a sparse LU.

    The pressure-datum Lagrange multiplier has O(n) entries and the two global
    dofs ``globals/U`` and ``globals/P`` couple to everything, so SuperLU on the
    whole matrix dies in memory expansion while the core block factorises in
    seconds. Those rows and columns go into a small dense border.

    Deleting them leaves a core with a one-dimensional pressure nullspace (the
    integral datum was what fixed the pressure constant), so the core is
    regularised by one diagonal bump on a pressure dof and that rank-1 change is
    carried as one extra border unknown (Woodbury). Solving is then one core
    back-solve per right-hand side plus a tiny dense border solve, and iterative
    refinement recovers the decades the ill-conditioned border Schur complement
    costs.

    Complex matrices go through unchanged: every dense workspace takes the
    dtype of ``J``, which is what the ``azimuthal_k >= 1`` pencils need.
    """

    def __init__(self, J, kind=None, *, dense_threshold: int = 1000,
                 permc_spec: str = "COLAMD", bump: float | None = None,
                 equilibrate: bool = True, verbose: bool = False):
        J = sp.csr_matrix(J)
        n = J.shape[0]
        if J.shape[0] != J.shape[1]:
            raise ValueError(f"BorderedLU needs a square matrix, got {J.shape}")
        self.dtype = np.complex128 if np.iscomplexobj(J.data) else np.float64
        J = J.astype(self.dtype)
        # Equilibrate first: the datum-multiplier row and the two global dofs sit
        # about ten decades away from the FEM rows, which is what gave the border
        # Schur complement condition 1e23 and cost four refinement steps per solve.
        self.Dr = np.ones(n)
        self.Dc = np.ones(n)
        if equilibrate:
            X = J
            for _ in range(4):
                rm = np.asarray(abs(X).max(axis=1).todense()).ravel()
                rm[rm == 0] = 1.0
                X = sp.diags(1.0 / rm) @ X
                self.Dr /= rm
                cm = np.asarray(abs(X).max(axis=0).todense()).ravel()
                cm[cm == 0] = 1.0
                X = X @ sp.diags(1.0 / cm)
                self.Dc /= cm
            J = X.tocsr()
        self.Jeq = J
        row_counts = np.diff(J.indptr)
        col_counts = np.diff(J.tocsc().indptr)
        self.border = np.flatnonzero((row_counts > dense_threshold)
                                     | (col_counts > dense_threshold))
        self.core = np.setdiff1d(np.arange(n), self.border)
        nb = int(self.border.size)
        m = int(self.core.size)
        self.n = n
        self.m = m
        self.nb = nb
        A = J[self.core, :][:, self.core].tocsr()
        F = np.asarray(J[self.core, :][:, self.border].todense(), dtype=self.dtype)
        self.G = np.asarray(J[self.border, :][:, self.core].todense(), dtype=self.dtype)
        H = np.asarray(J[self.border, :][:, self.border].todense(), dtype=self.dtype)
        # rank-1 regularisation of the core's pressure nullspace, when there is one
        self.p = None
        self.bump = 0.0
        if kind is not None:
            pressure = np.flatnonzero(np.asarray(kind)[self.core] == "pressure")
            if pressure.size:
                self.p = int(pressure[0])
                self.bump = float(np.abs(A.data).max()) if bump is None else float(bump)
        nr = 1 if self.p is not None else 0
        self.nr = nr
        if nr:
            A = (A + sp.coo_matrix(([self.bump], ([self.p], [self.p])),
                                   shape=(m, m))).tocsr()
        started = time.time()
        self.lu = spla.splu(A.tocsc(), permc_spec=permc_spec)
        self.lu_seconds = time.time() - started
        self.lu_nnz = int(self.lu.L.nnz + self.lu.U.nnz)
        if verbose:
            print(f"[bordered] core {m} border {nb} dtype {self.dtype}: "
                  f"LU {self.lu_seconds:.1f}s nnz={self.lu_nnz:.4g}", flush=True)
        U = np.zeros((m, nb + nr), dtype=self.dtype)
        U[:, :nb] = F
        if nr:
            U[self.p, nb] = -self.bump
        self.D = np.zeros((nb + nr, nb + nr), dtype=self.dtype)
        self.D[:nb, :nb] = H
        if nr:
            self.D[nb, nb] = -1.0
        self.Y = self._core_solve(U)
        blocks = [self.G @ self.Y]
        if nr:
            blocks.append(self.Y[self.p, :][None, :])
        self.S = self.D - (np.vstack(blocks) if blocks and (nb + nr)
                           else np.zeros((0, 0), dtype=self.dtype))
        self.S_cond = float(np.linalg.cond(self.S)) if self.S.size else 1.0

    def _core_solve(self, B):
        return self.lu.solve(np.asarray(B, dtype=self.dtype))

    def solve(self, b, refine: int = 3):
        """Solve ``J x = b`` for one or several right-hand sides.

        One pass through the ill-conditioned border gives only a few digits;
        each refinement step costs one more core back-solve.
        """
        b = np.asarray(b)
        single = b.ndim == 1
        if single:
            b = b[:, None]
        be = self.Dr[:, None] * b
        y = self._solve_once(be)
        for _ in range(int(refine)):
            y = y + self._solve_once(be - self.Jeq @ y)
        x = self.Dc[:, None] * y
        return x[:, 0] if single else x

    def _solve_once(self, b):
        b = np.asarray(b, dtype=self.dtype)
        bc = b[self.core, :]
        bb = b[self.border, :]
        xb = self._core_solve(bc)
        if self.nb + self.nr:
            blocks = [bb - self.G @ xb]
            if self.nr:
                blocks.append(-xb[self.p, :][None, :])
            Z = np.linalg.solve(self.S, np.vstack(blocks))
            xc = xb - self.Y @ Z
        else:
            Z = np.zeros((0, b.shape[1]), dtype=self.dtype)
            xc = xb
        x = np.zeros((self.n, b.shape[1]), dtype=self.dtype)
        x[self.core, :] = xc
        x[self.border, :] = Z[:self.nb, :]
        return x


def eigenpair_residual(J, M, lam, v) -> dict:
    """Scale-free residual of one candidate eigenpair, plus the mass action."""
    v = np.asarray(v).ravel()
    Mv = np.asarray(M @ v).ravel()
    lam_Mv = lam * Mv
    residual = float(np.linalg.norm(np.asarray(J @ v).ravel() - lam_Mv))
    norm_v = float(np.linalg.norm(v))
    norm_Mv = float(np.linalg.norm(Mv))
    norm_lam_Mv = float(np.linalg.norm(lam_Mv))
    return {
        "residual": residual,
        "residual_scale_free": (residual / norm_lam_Mv if norm_lam_Mv > 0.0
                                else float("inf")),
        "residual_over_v": residual / norm_v if norm_v > 0.0 else float("inf"),
        "mass_action": norm_Mv,
        "mass_action_over_v": norm_Mv / norm_v if norm_v > 0.0 else 0.0,
        "vector_norm": norm_v,
    }


def is_physical(J, M, lam, v, tol: float = 1e-6) -> bool:
    """Verdict on a candidate eigenpair: does it solve the pencil?

    This replaces the support-by-dof screen. For this pencil M is nonzero only
    on the kinematic-condition rows against the interface mesh-position
    columns, so a genuine eigenvector carries almost all of its 2-norm on
    pressure dofs and interface components about 1e-6 of that amplitude.
    Screening on where the amplitude sits therefore rejects every physical
    mode; screening on whether ``J v = lambda M v`` holds does not.
    """
    Mv = np.asarray(M @ np.asarray(v).ravel()).ravel()
    if not np.any(Mv):
        return False
    stats = eigenpair_residual(J, M, lam, v)
    return stats["residual_scale_free"] < tol


def reduced_finite_spectrum(J, M, kind=None, *, refine: int = 2,
                            permc_spec: str = "COLAMD",
                            dense_threshold: int = 1000, chunk: int = 64,
                            n_vectors: int | None = 6,
                            residual_tol: float = 1e-6,
                            restrict_blocks: bool = True,
                            verbose: bool = False) -> dict:
    """All finite eigenvalues of ``J v = lambda M v`` by an exact reduction.

    M carries mass only on the interface kinematic-condition rows, so
    ``rank(M) = nR << n`` and the pencil has at most ``nR`` finite eigenvalues.
    With ``E`` the ``n x nR`` injection of those rows and ``M_R = M[rows, :]``,
    ``w = M_R v`` obeys ``w = lambda G w`` with ``G = M_R J^{-1} E``, so the
    whole finite spectrum is ``1 / eig(G)``. No shift and no Krylov space: the
    M-nullspace, which is what floods a shift-invert subspace near zero, is
    projected out by construction rather than filtered afterwards.
    """
    J = sp.csr_matrix(J)
    M = sp.csr_matrix(M)
    n = J.shape[0]
    is_complex = bool(np.iscomplexobj(J.data) or np.iscomplexobj(M.data))
    dtype = np.complex128 if is_complex else np.float64
    J = J.astype(dtype)
    M = M.astype(dtype)
    rows = np.flatnonzero(np.asarray(abs(M).sum(axis=1)).ravel() > 0)
    mass_cols = np.flatnonzero(np.asarray(abs(M).sum(axis=0)).ravel() > 0)
    nR = int(rows.size)
    out: dict = {
        "n": int(n), "complex": is_complex, "n_mass_rows": nR,
        "n_mass_cols": int(mass_cols.size), "max_finite_eigenvalues": nR,
        "J_nnz": int(J.nnz), "M_nnz": int(M.nnz),
        "method": ("G = M[rows] J^{-1} E on the J block carrying the mass rows; "
                   "lambda = 1/eig(G); J^{-1} by a bordered LU with the dense "
                   "global rows/columns in the border and a rank-1 Woodbury fix "
                   "for the core pressure nullspace"),
        "refine_steps": int(refine), "permc_spec": permc_spec,
    }
    if nR == 0:
        out.update(eigenvalues=np.zeros(0, dtype=complex), n_finite=0,
                   eigenvectors=np.zeros((n, 0), dtype=complex),
                   residuals=[], physical=[], n_positive_real_part=0)
        return out
    # At k = 0 the azimuthal velocity block decouples exactly, so J is block
    # diagonal and a block with no mass rows can only carry infinite eigenvalues.
    keep = np.arange(n)
    n_blocks = 1
    if restrict_blocks:
        from scipy.sparse.csgraph import connected_components
        n_blocks, label = connected_components((abs(J) + abs(J).T).tocsr(),
                                               directed=False)
        same = bool(np.all(label[rows] == label[rows[0]])
                    and np.all(label[mass_cols] == label[rows[0]]))
        if same:
            keep = np.flatnonzero(label == label[rows[0]])
        out["mass_rows_in_one_J_block"] = same
    out["n_blocks_of_J"] = int(n_blocks)
    out["n_kept"] = int(keep.size)
    loc = -np.ones(n, dtype=np.int64)
    loc[keep] = np.arange(keep.size)
    rows_loc = loc[rows]
    Jk = J[keep, :][:, keep].tocsr()
    Mk = M[rows, :][:, keep]
    nk = Jk.shape[0]
    kind_k = np.asarray(kind)[keep] if kind is not None else None

    solver = BorderedLU(Jk, kind_k, dense_threshold=dense_threshold,
                        permc_spec=permc_spec, verbose=verbose)
    out.update(lu_seconds=solver.lu_seconds, lu_nnz=solver.lu_nnz,
               border_size=int(solver.nb), core_size=int(solver.m),
               schur_condition=solver.S_cond)
    rng = np.random.default_rng(20260921)
    probe = rng.standard_normal((nk, 2))
    if is_complex:
        probe = probe + 1j * rng.standard_normal((nk, 2))
    probe_x = solver.solve(probe, refine=refine)
    out["solver_relative_residual"] = [
        float(np.linalg.norm(Jk @ probe_x[:, i] - probe[:, i])
              / np.linalg.norm(probe[:, i])) for i in range(2)]

    started = time.time()
    X = np.empty((nk, nR), dtype=dtype)
    for a in range(0, nR, chunk):
        b = min(a + chunk, nR)
        rhs = np.zeros((nk, b - a), dtype=dtype)
        rhs[rows_loc[a:b], np.arange(b - a)] = 1.0
        X[:, a:b] = solver.solve(rhs, refine=refine)
        if verbose:
            print(f"  {b}/{nR} back-solves in {time.time() - started:.0f}s", flush=True)
    out["solve_seconds"] = time.time() - started
    G = np.asarray(Mk @ X)

    mu, W = np.linalg.eig(G)
    abs_mu = np.abs(mu)
    finite = abs_mu > 1e-12 * abs_mu.max() if abs_mu.max() > 0 else np.zeros(nR, bool)
    lam = np.where(finite, 1.0 / np.where(finite, mu, 1.0), np.inf + 0j)
    order = np.argsort(-np.real(np.where(finite, lam, -np.inf)))
    order = np.array([i for i in order if finite[i]], dtype=int)
    out["n_finite"] = int(finite.sum())
    out["eigenvalues"] = lam[order]
    out["mu"] = mu[order]
    out["n_positive_real_part"] = int((np.real(lam[order]) > 0.0).sum())

    want = order.size if n_vectors is None else min(int(n_vectors), order.size)
    vectors = np.zeros((n, want), dtype=complex)
    if want:
        vectors[keep, :] = X @ W[:, order[:want]]
    del X
    out["eigenvectors"] = vectors
    residuals = [eigenpair_residual(J, M, complex(lam[order[i]]), vectors[:, i])
                 for i in range(want)]
    out["residuals"] = residuals
    out["physical"] = [bool(r["mass_action"] > 0.0
                            and r["residual_scale_free"] < residual_tol)
                       for r in residuals]
    out["residual_tol"] = residual_tol
    return out


def finite_spectrum(problem: VoidBubbleProblem, azimuthal_k: int | None = None,
                    **kwargs) -> dict:
    """Whole finite spectrum of one azimuthal mode at the current state.

    Assembles the pencil exactly as ``problem.solve_eigenproblem`` would, then
    reduces it; see ``reduced_finite_spectrum``. Growth convention is
    ``exp(lambda t)``, i.e. ``Re(lambda) > 0`` is unstable.
    """
    pencil = assemble_pencil(problem, azimuthal_k)
    result = reduced_finite_spectrum(pencil.J, pencil.M, pencil.kind, **kwargs)
    result["azimuthal_k"] = azimuthal_k
    result["pencil"] = pencil
    return result


def shift_invert_start_vector(op_solve, M, mass_cols, rng, dtype=np.float64):
    """A start vector ARPACK will accept for ``OP = (J - sigma M)^{-1} M``.

    ``v0 = M r`` is what made ARPACK answer "error -9: Starting vector is zero":
    M's nonzero rows (the kinematic conditions) and its nonzero columns (the
    interface mesh positions) are disjoint dof sets, so ``M (M r) = 0``
    identically and the first application of OP annihilates it. Start instead
    from one application of OP to a random vector on M's *column* support.
    """
    n = M.shape[0]
    z = np.zeros(n, dtype=dtype)
    draw = rng.standard_normal(len(mass_cols))
    if np.iscomplexobj(z):
        draw = draw + 1j * rng.standard_normal(len(mass_cols))
    z[mass_cols] = draw
    v0 = np.asarray(op_solve(M @ z)).ravel()
    norm = float(np.linalg.norm(v0))
    if norm == 0.0:
        raise RuntimeError("OP applied to a random vector is exactly zero; "
                           "no starting vector in range(OP)")
    return v0 / norm


def shift_invert_control(J, M, kind=None, *, shift: float = 0.0, neig: int = 8,
                         refine: int = 2, permc_spec: str = "COLAMD",
                         dense_threshold: int = 1000, tol: float = 1e-9,
                         seed: int = 1234) -> list[dict]:
    """Independent Krylov control: ARPACK on ``(J - sigma M)^{-1} M``.

    The same path pyoomph's scipy eigensolver takes, with the start vector of
    ``shift_invert_start_vector``. Returns the pairs sorted by decreasing real
    part, each with its scale-free residual.
    """
    J = sp.csr_matrix(J)
    M = sp.csr_matrix(M)
    is_complex = bool(np.iscomplexobj(J.data) or np.iscomplexobj(M.data))
    dtype = np.complex128 if is_complex else np.float64
    J = J.astype(dtype)
    M = M.astype(dtype)
    n = J.shape[0]
    mass_cols = np.flatnonzero(np.asarray(abs(M).sum(axis=0)).ravel() > 0)
    solver = BorderedLU((J - shift * M).tocsr(), kind,
                        dense_threshold=dense_threshold, permc_spec=permc_spec)
    op = spla.LinearOperator(
        (n, n), matvec=lambda x: solver.solve(M @ x, refine=refine), dtype=dtype)
    rng = np.random.default_rng(seed)
    v0 = shift_invert_start_vector(
        lambda b: solver.solve(b, refine=refine), M, mass_cols, rng, dtype=dtype)
    theta, V = spla.eigs(op, k=neig, which="LM", v0=v0, tol=tol,
                         ncv=min(n - 1, max(4 * neig, 40)))
    pairs = []
    for i in range(len(theta)):
        th = complex(theta[i])
        lam = shift + 1.0 / th if abs(th) > 0 else complex(np.inf)
        stats = eigenpair_residual(J, M, lam, V[:, i])
        pairs.append({"shift": float(shift), "lambda_real": float(np.real(lam)),
                      "lambda_imag": float(np.imag(lam)), **stats})
    pairs.sort(key=lambda p: -p["lambda_real"])
    return pairs


def eigenvector_support(problem: VoidBubbleProblem, vector, *,
                        dof_description: tuple | None = None) -> dict:
    """Where an eigenvector's 2-norm sits, by dof kind. **Diagnostic only.**

    This is not a verdict and must not be used as one. For the void-bubble
    pencil M is nonzero only on the kinematic-condition rows against the
    interface mesh-position columns, so every genuine eigenvector carries
    essentially all of its 2-norm on pressure dofs; rejecting "pressure-only"
    modes rejects the physics. Use ``is_physical`` instead.

    Layout for ``azimuthal_k >= 1``. pyoomph releases the strong axis
    conditions and imposes them by matrix manipulation, so that pencil is
    complex and slightly larger than ``problem.ndof()`` (415193 against 412815
    at Ca_in 14 on the lip-8 station). ``problem.get_dof_description()`` called
    *outside* the eigensolve describes the ndof-sized system only; called while
    that numbering is live it describes the full pencil, which is what
    ``Pencil.types``/``names`` hold. Pass that through ``dof_description`` and
    the buckets are exact. Without it, the leading ``len(types)`` entries are
    classified as usual and the tail -- the released axis dofs, which at
    Ca_in 14 are the ``liquid/axis_rear`` and ``liquid/axis_front`` mesh and
    velocity unknowns -- is reported separately under ``released_axis`` rather
    than raising.
    """
    if dof_description is not None:
        types, names = dof_description
    else:
        types, names = problem.get_dof_description()
    types = np.asarray(types)
    amp = np.abs(np.asarray(vector).ravel())
    buckets = {k: 0.0 for k in DOF_KINDS}
    buckets["released_axis"] = 0.0
    total = float(np.dot(amp, amp)) or 1.0
    described = min(amp.size, types.size)
    weight = amp[:described] * amp[:described]
    kinds = np.array([classify_dof_name(names[int(k)]) for k in types[:described]])
    for k in DOF_KINDS:
        buckets[k] = float(weight[kinds == k].sum()) / total
    tail = amp[described:]
    buckets["released_axis"] = float(np.dot(tail, tail)) / total
    buckets["described_dofs"] = int(described)
    buckets["vector_length"] = int(amp.size)
    buckets["ndof_described"] = int(types.size)
    buckets["truncated_description"] = bool(amp.size > types.size)
    return buckets


def label_from_growth(real: float, rate: float, *, band: float = 1e-3) -> tuple[str, str]:
    """Map Re(lambda) to {stable, unstable, untested}.

    The band is in units of the row's own advective rate, so a raw threshold
    that is tiny at Ca_b = 14.8 is not enormous at Ca_b = 1e-4. Zero itself is
    convention-invariant; anything inside the band is 'neutral within
    resolution', which the campaign records as untested rather than as a crossing.
    """
    growth = real / max(rate, 1e-12)
    if abs(growth) <= band:
        return "untested", "neutral_within_resolution"
    if real > 0.0:
        return "unstable", "positive_growth"
    return "stable", "negative_growth"


def resample_film(z, r, z_grid):
    order = np.argsort(z)
    return np.interp(z_grid, np.asarray(z)[order], np.asarray(r)[order])


def film_grid(z, r, frac=0.25):
    z = np.asarray(z)
    length = float(z.max() - z.min())
    lo = float(z.min()) + frac * length
    hi = lo + (1.0 - 2.0 * frac) * length
    return np.linspace(lo, hi, 400)


def fit_exponential_envelope(times, values) -> dict:
    """Late-time log|signal| slope, plus a period from peak spacing.

    The eigensolver convention is exp(lambda t). A decaying oscillation has
    Re(lambda) < 0 and period 2 pi / |Im(lambda)|.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 8:
        return {"ok": False, "reason": "too_few_samples"}
    # Use the second half, so transients from other modes have decayed.
    cut = times.size // 2
    t = times[cut:]
    y = values[cut:]
    amp = np.abs(y)
    floor = max(1e-16, 1e-6 * float(np.max(amp)))
    mask = amp > floor
    if mask.sum() < 6:
        return {"ok": False, "reason": "signal_at_floor"}
    t_fit, log_amp = t[mask], np.log(amp[mask])
    slope, intercept = np.polyfit(t_fit, log_amp, 1)
    # Peaks of the full (not late-only) signal for a period.
    dy = np.diff(values)
    peaks = []
    for i in range(1, len(dy)):
        if dy[i - 1] > 0.0 >= dy[i]:
            peaks.append(times[i])
    period = None
    if len(peaks) >= 2:
        gaps = np.diff(peaks)
        gaps = gaps[gaps > 0.1 * np.median(gaps)]
        if len(gaps):
            period = float(np.median(gaps))
    return {
        "ok": True,
        "sigma": float(slope),
        "log_amp0": float(intercept),
        "period": period,
        "omega": (2.0 * np.pi / period) if period else None,
        "n_peaks": len(peaks),
        "n_fit": int(mask.sum()),
    }
