"""S19: Campaign A-reduced — named Laplace-number grid, not a surface.

S15 is a Reynolds bound (Stokes holds to 0.78% for Re<1.47). This stage
fills the named cells the campaign asked for:

    La in {0, 0.1, 1} at Ca_in in {0.05, 0.5} and the highest reachable
    Ca_in for that La; La=10 at the thin-film station and at Ca_in=0.5
    if it converges.

S15 numbers are reused where the (Ca_in, La, pads) cell already exists.
Pad-doubled windows are measured at every La>0. A failed high-Ca
inertial pad-double is recorded, not fatal, matching S15. Do not rebuild
h(Ca_b, La).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eigen_helpers import (
    CONTINUATION_SEED,
    FRESH_CA_LIMIT,
    build_stability_problem,
    climb_with_forced_remesh,
    remesh_ceiling_for_film,
)
from runners.run_case import record_case


R_TUBE = 0.7
S15_PATH = ROOT / "receipts" / "s15_inertia_bound.json"

# Production window first; doubled second. Highest-Ca La=1 is 2.9 (S17).
CELLS = (
    (0.05, 0.0, 3.0, 3.0),
    (0.05, 0.1, 3.0, 3.0),
    (0.05, 0.1, 6.0, 6.0),
    (0.05, 1.0, 3.0, 3.0),
    (0.05, 1.0, 6.0, 6.0),
    (0.05, 10.0, 3.0, 3.0),
    (0.05, 10.0, 6.0, 6.0),
    (0.5, 0.0, 1.5, 3.0),
    (0.5, 0.1, 1.5, 3.0),
    (0.5, 1.0, 1.5, 3.0),
    (0.5, 10.0, 1.5, 3.0),
    (2.9, 1.0, 1.5, 3.0),
    (2.0, 1.0, 3.0, 6.0),
    (6.0, 0.0, 1.5, 3.0),
    (6.0, 0.0, 3.0, 6.0),
    (6.0, 0.1, 1.5, 3.0),
    (6.0, 0.1, 3.0, 6.0),
)

REQUIRED = {
    (0.05, 0.0), (0.05, 0.1), (0.05, 1.0), (0.05, 10.0),
    (0.5, 0.0), (0.5, 0.1), (0.5, 1.0),
    (6.0, 0.0), (6.0, 0.1),
    (2.9, 1.0),
}

OPTIONAL_FAILURES = {
    (6.0, 0.1, 3.0, 6.0),
    (0.5, 10.0, 1.5, 3.0),
    (2.0, 10.0, 1.5, 3.0),
    (2.0, 1.0, 3.0, 6.0),
}


def we_groups(laplace: float, ca_in: float, ca_b: float) -> dict[str, float]:
    """Magnini & Herrada groups; see scratch/notes/we-365-convention-2026-09-15.md."""
    ratio = ca_b / ca_in
    return {
        "reynolds": laplace * ca_b,
        "weber": laplace * ca_b ** 2,
        "reynolds_them": 1.4 * laplace * ca_in,
        "weber_them": 1.4 * laplace * ca_in ** 2,
        "weber_p": 0.7 * laplace * ca_in ** 2 * (2.0 - ratio) ** 2,
    }


def _receipt_row(row: dict) -> dict:
    cleaned = dict(row)
    profile = cleaned.get("interface_profile")
    if isinstance(profile, str):
        try:
            cleaned["interface_profile"] = str(Path(profile).resolve().relative_to(ROOT))
        except ValueError:
            cleaned.pop("interface_profile", None)
    return cleaned


def _public(row: dict, *, source: str) -> dict:
    groups = we_groups(row["laplace"], row["ca_in"], row["ca_b"])
    return {
        "source": source,
        "ca_in": row["ca_in"],
        "laplace": row["laplace"],
        "pad_rear": row["pad_rear"],
        "pad_front": row["pad_front"],
        "ca_b": row["ca_b"],
        "h_flat_over_r_tube": row["h_flat_over_r_tube"],
        "ndof": row.get("ndof"),
        "route": row.get("route"),
        "film_elements": row.get("film_elements", 3.0),
        "bulk_resolution_over_r0": row.get("bulk_resolution_over_r0", 0.10),
        **groups,
    }


def _key(ca: float, la: float, pad_rear: float, pad_front: float) -> tuple:
    return (round(ca, 12), round(la, 12), round(pad_rear, 12), round(pad_front, 12))


def load_s15() -> dict[tuple, dict]:
    if not S15_PATH.is_file():
        return {}
    payload = json.loads(S15_PATH.read_text(encoding="utf-8"))
    out = {}
    for row in payload.get("results") or []:
        out[_key(row["ca_in"], row["laplace"], row["pad_rear"], row["pad_front"])] = row
    return out


def solve(ca_in: float, laplace: float, *, film_elements: float,
          bulk_resolution: float, pad_rear: float, pad_front: float,
          output_root: Path, remesh_on_climb: bool) -> dict:
    outdir = (output_root /
              f"s19_ca{ca_in:g}_la{laplace:g}_pad{pad_rear:g}x{pad_front:g}")
    if ca_in <= FRESH_CA_LIMIT + 1e-12:
        seed = ca_in
    elif laplace > 0.0:
        seed = FRESH_CA_LIMIT
    else:
        seed = CONTINUATION_SEED
    problem = build_stability_problem(
        seed, film_elements=film_elements, bulk_resolution=bulk_resolution,
        output_dir=outdir, remesh=True, pad_rear=pad_rear, pad_front=pad_front,
        laplace=laplace, azimuthal_stability=False,
    )
    route = problem.solve_to_steady()
    if ca_in > seed + 1e-12:
        if remesh_on_climb:
            ceiling = 2.0 if laplace > 0.0 else remesh_ceiling_for_film(
                film_elements, pad_rear)
            climb_with_forced_remesh(problem, ca_in, remesh_ceiling=ceiling)
            route = "continuation+forced_remesh"
        else:
            problem.do_call_remeshing_when_necessary = False
            problem.go_to_param(Ca_in=ca_in, max_step=0.25)
            route = "continuation_no_remesh"
    row = record_case(problem, output_dir=outdir, route=route, wall=0.0, audit={})
    row["pad_rear"] = pad_rear
    row["pad_front"] = pad_front
    return row


def window_pairs(results: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in results:
        grouped.setdefault((row["ca_in"], row["laplace"]), []).append(row)
    out = []
    for (ca, la), rows in sorted(grouped.items()):
        if la <= 0.0 or len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: r["pad_rear"])
        a, b = rows[0], rows[-1]
        rel = abs(b["h_flat_over_r_tube"] / a["h_flat_over_r_tube"] - 1.0)
        out.append({
            "ca_in": ca,
            "laplace": la,
            "h_small": a["h_flat_over_r_tube"],
            "h_large": b["h_flat_over_r_tube"],
            "rel_diff_h": rel,
            "agree_to_2pct": rel <= 0.02,
            "pads": [
                [a["pad_rear"], a["pad_front"]],
                [b["pad_rear"], b["pad_front"]],
            ],
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--film-elements", type=float, default=3.0)
    parser.add_argument("--bulk-resolution", type=float, default=0.10)
    parser.add_argument("--output-root", type=Path, default=ROOT / "receipts")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "receipts" / "s19_la_grid.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failures: list[dict] = []
    if args.resume and args.out.is_file():
        prev = json.loads(args.out.read_text(encoding="utf-8"))
        results = list(prev.get("results") or [])
        failures = list(prev.get("failures") or [])

    done = {_key(r["ca_in"], r["laplace"], r["pad_rear"], r["pad_front"])
            for r in results}
    failed_keys = {_key(f["ca_in"], f["laplace"], f["pad_rear"], f["pad_front"])
                   for f in failures}
    reused = load_s15()

    cells = list(CELLS)
    if any(abs(r["ca_in"] - 0.5) < 1e-12 and abs(r["laplace"] - 10.0) < 1e-12
           for r in results):
        cells.append((2.0, 10.0, 1.5, 3.0))

    def persist() -> None:
        payload = assemble(results, failures)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    for ca, la, pad_rear, pad_front in cells:
        key = _key(ca, la, pad_rear, pad_front)
        if key in done or key in failed_keys:
            print(
                f"skip Ca_in={ca:g} La={la:g} pads={pad_rear:g}/{pad_front:g}",
                flush=True,
            )
            continue
        prior = reused.get(key)
        if prior is not None:
            row = _public(prior, source="s15")
            results.append(row)
            done.add(key)
            print(
                f"reuse S15 Ca_in={ca:g} La={la:g} pads={pad_rear:g}/{pad_front:g} "
                f"Ca_b={row['ca_b']:.6g} H={row['h_flat_over_r_tube']:.6f}",
                flush=True,
            )
            persist()
            continue
        remesh_on_climb = not (
            abs(ca - 6.0) < 1e-12 and la > 0.0 and pad_rear > 1.5 + 1e-12)
        print(
            f"Ca_in={ca:g} La={la:g} pads={pad_rear:g}/{pad_front:g} "
            f"remesh_on_climb={remesh_on_climb}",
            flush=True,
        )
        try:
            raw = solve(
                ca, la, film_elements=args.film_elements,
                bulk_resolution=args.bulk_resolution,
                pad_rear=pad_rear, pad_front=pad_front,
                output_root=args.output_root,
                remesh_on_climb=remesh_on_climb,
            )
            row = _public(_receipt_row(raw), source="s19")
            results.append(row)
            done.add(key)
            print(
                f"  Ca_b={row['ca_b']:.6g} H={row['h_flat_over_r_tube']:.6f} "
                f"Re={row['reynolds']:.4g} We_p={row['weber_p']:.4g}",
                flush=True,
            )
            if abs(ca - 0.5) < 1e-12 and abs(la - 10.0) < 1e-12:
                extra = (2.0, 10.0, 1.5, 3.0)
                if extra not in cells:
                    cells.append(extra)
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "ca_in": ca, "laplace": la,
                "pad_rear": pad_rear, "pad_front": pad_front,
                "remesh_on_climb": remesh_on_climb,
                "failed": f"{type(exc).__name__}: {exc}",
            })
            failed_keys.add(key)
            print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)
        persist()

    payload = assemble(results, failures)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out} passed={payload['passed']}", flush=True)
    return 0 if payload["passed"] else 1


def assemble(results: list[dict], failures: list[dict]) -> dict:
    present = {(r["ca_in"], r["laplace"]) for r in results}
    missing = sorted(REQUIRED - present)
    windows = window_pairs(results)
    thin_windows = [w for w in windows if abs(w["ca_in"] - 0.05) < 1e-12]
    window_ok = bool(thin_windows) and all(w["agree_to_2pct"] for w in thin_windows)
    fatal = [
        f for f in failures
        if _key(f["ca_in"], f["laplace"], f["pad_rear"], f["pad_front"])
        not in {_key(*cell) for cell in OPTIONAL_FAILURES}
    ]
    grid = []
    for ca, la in sorted(present):
        prod = [r for r in results if abs(r["ca_in"] - ca) < 1e-12
                and abs(r["laplace"] - la) < 1e-12]
        prod.sort(key=lambda r: r["pad_rear"])
        row = prod[0]
        stokes = next((r for r in results
                       if abs(r["ca_in"] - ca) < 1e-12 and r["laplace"] == 0.0
                       and abs(r["pad_rear"] - row["pad_rear"]) < 1e-12), None)
        dh = None
        if stokes is not None and la > 0.0:
            dh = row["h_flat_over_r_tube"] / stokes["h_flat_over_r_tube"] - 1.0
        grid.append({
            "ca_in": ca,
            "laplace": la,
            "ca_b": row["ca_b"],
            "h_flat_over_r_tube": row["h_flat_over_r_tube"],
            "reynolds": row["reynolds"],
            "weber": row["weber"],
            "weber_p": row["weber_p"],
            "h_rel_change_vs_stokes": dh,
            "source": row["source"],
        })
    passed = not missing and not fatal and window_ok and bool(results)
    return {
        "check": "s19_la_grid",
        "passed": passed,
        "note": (
            "Named La grid at three Ca_in stations, with pad-doubled windows "
            "at La>0. S15 cells are reused. La=10 at Ca_in=0.5 and the "
            "high-Ca inertial pad-double may fail without failing the stage. "
            "This is not a full h(Ca_b, La) surface."
        ),
        "results": results,
        "failures": failures,
        "grid": grid,
        "windows": windows,
        "missing_required": [{"ca_in": c, "laplace": la} for c, la in missing],
        "window_independent_at_thin_film": window_ok,
        "la1_highest_ca_in": 2.9,
        "weber_p_note": (
            "weber_p is Magnini & Herrada's rear-impingement We, "
            "0.7 La Ca_in^2 (2 - Ca_b/Ca_in)^2. Do not compare our "
            "We = La Ca_b^2 to 3.65."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
