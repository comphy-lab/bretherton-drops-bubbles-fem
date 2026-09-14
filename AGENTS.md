# Bretherton void-bubble FEM

Sharp-interface pyoomph solver for a confined bubble whose interior is a void.
Read `README.md` first for the layout and the case list.

## Environment

`uv` owns the lockfile. Create the interpreter with `pyoomph-env`; never
`pip install` into `.venv` by hand and never create a conda environment for
this repository.

pyoomph is pinned to CoMPhy tag `comphy-2026.09.07.1`
(`a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618`). Change that pin only with
`pyoomph-env pin` and a refreshed `uv.lock`.

## Physics invariants

Do not silently change any of these.

- Nondimensionalisation is `sigma = mu = R0 = 1`. `Ca_in` is the imposed mean
  inlet speed; `La = rho`; `Oh = 1/sqrt(La)`.
- The bubble interior is a void. Do not mesh a gas domain or reintroduce a
  second fluid to match the Basilisk VoF companion.
- Void gauge pressure is zero. A finite bubble pressure would be an explicit
  normal traction on the free surface, not an interior fluid.
- The comparison film is the deposited central thickness `h = Rtube - r`, not
  `b_min` and not the seeded capsule gap. Encode mesh as `Δ/R0`.
- `Ca_in` is imposed; `Ca_b = U` is measured from the solved bubble speed.

## pyoomph traps that apply here

- `NavierStokesFreeSurface` is moving only when a `BaseMovingMeshEquations`
  (here `HyperelasticSmoothedMesh`) is present on the bulk domain.
- `DirichletBC` cannot depend on an unknown. The co-moving speed `U` is a
  global degree of freedom, so wall and inlet axial velocities use
  `EnforcedBC`.
- Point constraints on the axis vanish under the axisymmetric measure
  `2*pi*r`. Pin the rear tip with a Cartesian `WeakContribution`.
- `AxisymmetryBC` must sit on `axis_rear` and `axis_front`; it recurses onto
  `interface/axis_rear` where the free surface adds fields.
- Do not multiply pyoomph integrals by `2*pi*r` again.

## Repository conventions

- `problems/` holds equations and the capsule geometry.
- `cases/` holds portable `pyoomph-case-v1` documents.
- `tests/` must keep at least one path that imports no pyoomph.
- `runs/` is attempt/restart custody, never source.
- Component `AGENTS.md` is public-candidate. Private operational history stays
  in the project-context manual.
