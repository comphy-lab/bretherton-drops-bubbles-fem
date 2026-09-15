"""Shared pieces for the azimuthal-stability stages.

Kept out of the problem class because these are campaign checks, not physics.
The wavenumber is ``azimuthal_k`` everywhere in receipts: the report already
uses ``m`` for the deposited-liquid fraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
) -> VoidBubbleProblem:
    if pressure_datum == "fixation":
        raise ValueError(
            "azimuthal eigensolving is refused with pressure_datum='fixation'; "
            "pass 'integral' as an argument, do not change the class default"
        )
    geom = CapsuleGeometry.for_capillary_number(
        ca_in, pad_rear=pad_rear, pad_front=pad_front, seed_from_bubble_speed=True)
    problem = VoidBubbleProblem(
        geometry=geom, ca_in=ca_in, laplace=laplace,
        film_elements=film_elements, bulk_resolution=bulk_resolution,
        closure="centroid", outlet="poiseuille",
        pressure_datum=pressure_datum, remesh=remesh,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    problem.set_output_directory(str(output_dir))
    problem.setup_for_stability_analysis(azimuthal_stability=True)
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


def rightmost_sample(problem: VoidBubbleProblem, azimuthal_k: int, n: int,
                     shift: float) -> dict:
    values, vectors = problem.solve_eigenproblem(n, azimuthal_m=azimuthal_k, shift=shift)
    index = int(max(range(len(values)), key=lambda i: values[i].real))
    lam = complex(values[index])
    return {
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
    azimuthal_k: int,
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
    return {k: sample[k] for k in (
        "azimuthal_k", "n", "shift", "real", "imag", "index",
        "n_returned", "positive_count", "near_zero_count",
    )}


def _strip_none(value):
    return value


def eigenvector_support(problem: VoidBubbleProblem, vector) -> dict:
    """Where the eigenvector lives, for the numerically-neutral check.

    AGENTS.md records a near-null Jacobian mode on pressure dofs under
    fixation. A candidate crossing whose amplitude sits on pressure or the
    global multipliers is not a physical instability.
    """
    types, names = problem.get_dof_description()
    amp = np.abs(np.asarray(vector).ravel())
    if amp.size != len(types):
        return {"error": f"eigenvector length {amp.size} != ndof {len(types)}"}
    total = float(np.dot(amp, amp)) or 1.0
    buckets = {
        "pressure": 0.0,
        "globals": 0.0,
        "mesh": 0.0,
        "velocity": 0.0,
        "other": 0.0,
    }
    for index, kind in enumerate(types):
        name = names[int(kind)]
        lower = name.lower()
        weight = float(amp[index] * amp[index])
        if lower.endswith("/u") or lower.endswith("/p") or "globals" in lower:
            buckets["globals"] += weight
        elif "pressure" in lower:
            buckets["pressure"] += weight
        elif "mesh" in lower:
            buckets["mesh"] += weight
        elif "velocity" in lower:
            buckets["velocity"] += weight
        else:
            buckets["other"] += weight
    return {k: v / total for k, v in buckets.items()}


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
