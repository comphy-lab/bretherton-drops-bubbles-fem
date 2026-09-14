# Bretherton void-bubble FEM

Axisymmetric sharp-interface finite-element solver for a gas bubble
translating through a liquid-filled circular tube. The bubble interior is a
void: only the liquid is meshed, and surface tension acts on the liquid/void
interface. There is no interior gas and no volume-of-fluid representation.

The companion Basilisk two-phase solver lives in a separate repository. This
codebase is the free-surface counterpart.

## Layout

```
problems/       problem and geometry definitions
cases/          portable pyoomph-case-v1 documents
tests/          software-contract tests (no pyoomph import)
verification/   intended equation/numerical checks
validation/     intended comparison to Bretherton and the Basilisk campaign
runners/        case loader
postprocess/    deposited-film helpers
docs/           authored reports, when they exist
```

## Requirements

- Python 3.13
- `uv` for the locked environment
- pyoomph from the CoMPhy fork, pinned at tag `comphy-2026.09.07.1`
  (`a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618`)

The environment is created with `pyoomph-env`; do not `pip install` into an
ambient interpreter. A source build compiles vendored oomph-lib and is slow
on a cold start.

## Usage

Software-contract tests do not need pyoomph:

```bash
uv run --no-sync pytest tests/test_geometry.py tests/test_film.py tests/test_case_contract.py
```

Construct the default case without solving:

```bash
uv run --no-sync python runners/run_case.py --case cases/bubble_ca0p05.json --dry-run
```

`--dry-run` still imports pyoomph. Until a project-local environment exists,
run only the contract tests.

The default case `cases/bubble_ca0p05.json` uses the same nondimensional
groups as the Basilisk companion: `Ca_in = 0.05`, `La = 1`, `Rtube/R0 = 0.7`.
`Ca_in` is the imposed mean inlet capillary number; the bubble speed is a
solved unknown.

## Film thickness

The comparison thickness is the deposited central film `h = Rtube - r_interface`,
not the instantaneous minimum gap. Mesh spacing is `Δ/R0`.
