"""Load a portable case and construct the void-bubble problem.

This runner does not launch a production campaign. It exists so a case file,
the FEM commit and the pyoomph pin can be bound in a run manifest. Execution
requires a project-local pyoomph environment from ``pyoomph-env``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.bretherton_bubble import VoidBubbleProblem  # noqa: E402


def load_case(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pyoomph-case-v1":
        raise ValueError(f"{path} is not a pyoomph-case-v1 document")
    if payload.get("physics", {}).get("interior") != "void":
        raise ValueError("this runner only accepts a void bubble interior")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="construct the problem and print the capsule geometry without solving",
    )
    args = parser.parse_args(argv)
    case = load_case(args.case)
    problem = VoidBubbleProblem.from_case(case)
    geom = problem.geometry
    print(
        json.dumps(
            {
                "case_id": case["case_id"],
                "ca_in": problem.ca_in,
                "laplace": problem.laplace,
                "r_tube": geom.r_tube,
                "r_body": geom.r_body,
                "z_rear": geom.z_rear,
                "z_front": geom.z_front,
                "cylinder_length": geom.cylinder_length,
                "dry_run": args.dry_run,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.dry_run:
        raise SystemExit(
            "live solves require a pyoomph-env receipt; pass --dry-run until that environment exists"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
