"""Small native FEM software test of signed Stokes forcing and the upper-tip pin.

Three tiny time steps check execution only; this does not compute a steady
Figure 3 branch. Run explicitly with a fresh --output-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(output: Path) -> dict:
    import numpy as np

    from problems.bretherton_bubble import VoidBubbleProblem
    from problems.geometry import CapsuleGeometry

    output.mkdir(parents=True, exist_ok=False)
    radius = 3.3 ** (-1 / 3)
    body = 0.9 * radius
    geometry = CapsuleGeometry(
        r0=1.0, r_tube=radius, r_body=body,
        z_rear=10 * radius - CapsuleGeometry.bubble_length(1.0, body),
        l_domain=15 * radius,
    )
    with VoidBubbleProblem(geometry=geometry, ca_in=-0.002, bond=0.0,
                           allow_signed_flow=True, closure="front_tip", laplace=0.0,
                           film_elements=1.0, bulk_resolution=0.3,
                           remesh=False, newton_tolerance=1e-9) as problem:
        problem.set_output_directory(str(output / "solver"))
        problem.set_linear_solver("superlu")
        problem.set_c_compiler("system")
        problem.initialise()
        problem.audit_constraints()
        residuals = []
        for bond in (0.0, 0.78, 1.56):
            problem.get_global_parameter("Bo").value = bond
            residuals.append(np.asarray(problem.get_residuals()).copy())
        force_response = residuals[2] - residuals[0]
        assert np.all(np.isfinite(force_response))
        assert np.linalg.norm(force_response, np.inf) > 1e-6
        linearity_error = float(np.linalg.norm(
            residuals[1] - residuals[0] - 0.5 * force_response, np.inf
        ))
        assert linearity_error < 1e-10
        problem.run(3e-5, startstep=1e-5, maxstep=1e-5, outstep=3e-5,
                    temporal_error=1e-3)
        checks = problem.consistency()
        z, _ = problem.interface_polyline()
        tip_error = abs(float(max(z)) - geometry.z_front)
        assert abs(checks.volume_error) < 1e-5
        assert abs(checks.flux_error) < 1e-5
        assert tip_error < 1e-8
        assert problem.current_ca_in() == -0.002
        assert problem.current_bond() == 1.56
        receipt = {
            "kind": "software_smoke", "passed": True, "laplace": 0.0, "bond": 1.56,
            "ca_l": -0.002, "physical_time": float(problem.get_current_time()),
            "degrees_of_freedom": problem.ndof(),
            "force_residual_linearity_error": linearity_error,
            "force_residual_change_inf": float(np.linalg.norm(force_response, np.inf)),
            "volume_error": checks.volume_error, "flux_error": checks.flux_error,
            "upper_tip_error": tip_error,
            "claim": "signed-flow Stokes FEM executes with live nonzero Bo and fixed upper tip; no steady-branch validation",
        }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()), indent=2))
