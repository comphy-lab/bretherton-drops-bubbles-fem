"""S20: sliding interface nodes move the mesh, not the interface.

``sliding_nodes=True`` prescribes where interface nodes sit *along* the
interface (``problems.sliding_frame``) and leaves where they sit *across* it to
the kinematic condition. If that is so, nothing physical may change, and the
unconfined void of S1 is the sharpest place to check it, because its answer is
known in closed form: a unit sphere with a Laplace pressure of 2.

Three solves, at one resolution:

* ``off``: S1 as it is, the stationary solve from the seeded capsule.
* ``on``: the same with ``sliding_nodes=True`` and the default frame, under
  which every interface node may move only along the normal through its seeded
  position. It must pass S1 and agree with ``off`` to the S1 tolerance.
* ``sliding``: from the converged ``on`` sphere, a march in which the frame
  drags the nodes near one point of the sphere along it, and contracts them
  about it, as it would at an advancing rim. The nodes must move along the
  sphere by many times the tolerance while the sphere stays a sphere, the
  Laplace pressure stays 2 and the volume is conserved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from problems.bretherton_bubble import VoidBubbleProblem
from problems.geometry import CapsuleGeometry
from problems.sliding_frame import SlidingFrame


def sphere_measures(problem: VoidBubbleProblem, geom: CapsuleGeometry) -> dict:
    z, r = problem.interface_polyline()
    centre = 0.5 * (float(z.min()) + float(z.max()))
    radii = np.hypot(r, z - centre)
    return {
        "sphere_radius_max_dev": float(np.max(np.abs(radii - 1.0))),
        "laplace_jump": problem.bubble_pressure() - problem.outlet_mean_pressure(),
        "volume_error": problem.observable("void_volume", "liquid/interface") / geom.volume - 1.0,
        "z": z, "r": r, "centre": centre,
    }


def build(sliding: bool, *, resolution: float, out: Path):
    geom = CapsuleGeometry(
        r0=1.0, r_tube=1.5, r_body=0.9, z_rear=2.0,
        l_domain=2.0 + CapsuleGeometry.bubble_length(1.0, 0.9) + 2.0,
    )
    problem = VoidBubbleProblem(
        geometry=geom, ca_in=1e-8, laplace=0.0, film_elements=2.0, bulk_resolution=resolution,
        closure="centroid", outlet="poiseuille", remesh=False, sliding_nodes=sliding,
    )
    problem.set_output_directory(str(out))
    problem.initialise()
    problem.audit_constraints()
    return problem, geom


def run(*, resolution: float, rtol: float, steps: int, dt: float) -> dict:
    result: dict = {"resolution": resolution, "rtol": rtol}
    base = ROOT / "receipts" / f"s20_{resolution:g}"

    off, geom = build(False, resolution=resolution, out=base / "off")
    off.solve(max_newton_iterations=40)
    m_off = sphere_measures(off, geom)

    on, geom = build(True, resolution=resolution, out=base / "on")
    on.solve(max_newton_iterations=40)
    m_on = sphere_measures(on, geom)
    for key in ("sphere_radius_max_dev", "laplace_jump", "volume_error"):
        result[f"off_{key}"] = m_off[key]
        result[f"on_{key}"] = m_on[key]
    result["on_minus_off_laplace"] = m_on["laplace_jump"] - m_off["laplace_jump"]

    # Drag and contract the nodes about the point of the sphere at 45 degrees
    # from the rear axis, along the sphere.
    angle = math.radians(45.0)
    c = (math.sin(angle), m_on["centre"] - math.cos(angle))
    tangent = (math.cos(angle), math.sin(angle))
    on.reset_lagrangian_coordinates()
    z0, r0 = on.interface_polyline()
    t0 = on.get_current_time(as_float=True)
    on.set_sliding_frame(SlidingFrame(
        t0=t0, centre=c, velocity=(0.5 * tangent[0], 0.5 * tangent[1]), contraction_rate=3.0,
        translation_length=0.4, contraction_length=0.15))
    for k in range(steps):
        on.run(t0 + (k + 1) * dt, timestep=dt, outstep=False)
    m_sl = sphere_measures(on, geom)
    # The node chain keeps its order and length, so nodes correspond by index.
    moved = np.hypot(m_sl["r"] - r0, m_sl["z"] - z0)
    result.update({
        "sliding_steps": steps, "sliding_dt": dt,
        "sliding_node_displacement_max": float(np.max(moved)),
        "sliding_sphere_radius_max_dev": m_sl["sphere_radius_max_dev"],
        "sliding_laplace_jump": m_sl["laplace_jump"],
        "sliding_volume_error": m_sl["volume_error"],
    })

    def sphere_ok(prefix: str) -> bool:
        return (abs(result[f"{prefix}laplace_jump"] / 2.0 - 1.0) < rtol
                and result[f"{prefix}sphere_radius_max_dev"] < rtol
                and abs(result[f"{prefix}volume_error"]) < 1e-6)

    result["passed_off"] = sphere_ok("off_")
    result["passed_on"] = sphere_ok("on_") and abs(result["on_minus_off_laplace"]) < 2.0 * rtol
    result["passed_sliding"] = (sphere_ok("sliding_")
                                and result["sliding_node_displacement_max"] > 10.0 * rtol)
    result["passed"] = result["passed_off"] and result["passed_on"] and result["passed_sliding"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=float, default=0.12)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--out", type=Path, default=ROOT / "receipts" / "s20_sliding_nodes.json")
    args = parser.parse_args(argv)
    try:
        result = run(resolution=args.resolution, rtol=args.rtol, steps=args.steps, dt=args.dt)
    except Exception as exc:  # noqa: BLE001 - a failure here is the result
        result = {"resolution": args.resolution, "passed": False,
                  "error": f"{type(exc).__name__}: {exc}"}
    payload = {"stage": "S20", "results": [result], "passed": bool(result["passed"])}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
