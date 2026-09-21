"""Bounded native arclength/restart software smoke test on x**2 = Ca_in.

The algebraic fold is an API test, not the bubble benchmark. Run explicitly
with a pyoomph interpreter and a fresh --output-dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(output: Path) -> dict:
    from pyoomph import InitialCondition, ODEEquations, Problem
    from pyoomph.expressions import var_and_test

    class Fold(ODEEquations):
        def __init__(self, ca):
            super().__init__()
            self.ca = ca

        def define_fields(self):
            self.define_ode_variable("x")

        def define_residuals(self):
            x, test = var_and_test("x")
            self.add_residual((x * x - self.ca) * test)

    class FoldProblem(Problem):
        def define_problem(self):
            ca = self.define_global_parameter(Ca_in=0.04)
            self += (Fold(ca) + InitialCondition(x=0.2)) @ "fold"

    def setup(problem, name):
        problem.ignore_command_line = True
        problem.newton_solver_tolerance = 1e-12
        problem.continuation_data_in_states = True
        problem.set_output_directory(str(output / name))
        problem.set_linear_solver("superlu")
        problem.set_c_compiler("system")
        problem.set_arclength_inner_product("ndof")

    def row(problem):
        x = float(problem.get_ode("fold").get_value("x", as_float=True))
        ca = float(problem.get_global_parameter("Ca_in").value)
        assert abs(x * x - ca) < 1e-9
        return [ca, x]

    output.mkdir(parents=True, exist_ok=False)
    history = []
    with FoldProblem() as problem:
        setup(problem, "original")
        problem.solve()
        history.append(row(problem))
        ds = -0.01
        for _ in range(50):
            ds = problem.arclength_continuation("Ca_in", ds, max_ds=0.01, min_ds=1e-10)
            history.append(row(problem))
        checkpoint = output / "fold.dump"
        problem.save_state(str(checkpoint), quiet=True)
        before = row(problem)
        saved_ds = ds
        problem.arclength_continuation("Ca_in", ds, max_ds=0.01, min_ds=1e-10)
        uninterrupted = row(problem)
    with FoldProblem() as resumed:
        setup(resumed, "resumed")
        resumed.initialise()
        resumed.load_state(str(checkpoint), quiet=True)
        restored = row(resumed)
        assert max(abs(a - b) for a, b in zip(before, restored, strict=True)) < 1e-12
        resumed.arclength_continuation("Ca_in", saved_ds, max_ds=0.01, min_ds=1e-10)
        restarted = row(resumed)
    error = max(abs(a - b) for a, b in zip(uninterrupted, restarted, strict=True))
    assert error < 1e-9, error
    xs = [r[1] for r in history]
    assert min(xs) < 0 < max(xs), "arclength did not cross the algebraic fold"
    receipt = {
        "kind": "software_smoke", "passed": True,
        "problem": "x**2 = Ca_in", "crossed_fold": True,
        "restart_next_step_max_error": error, "history": history,
        "claim": "native arclength traverses a fold and restart preserves its direction; no bubble validation",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir.resolve()), indent=2))
