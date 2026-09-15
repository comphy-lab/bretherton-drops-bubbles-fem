# Bretherton void-bubble FEM

Sharp-interface pyoomph solver for a confined bubble whose interior is a void,
computed in a frame co-moving with the bubble. Read `README.md` first for the
layout and the results.

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
- **The bubble volume is held fixed at `4/3 pi R0^3`, and the void pressure is
  the Lagrange multiplier of that constraint.** It is not zero. Setting it to
  zero and letting the volume float makes the answer depend on the window
  length, because the cap curvature of a confined bubble is fixed
  geometrically at roughly `2/(R_t - h)`, so the normal-stress balance
  determines the void pressure from the local liquid pressure; changing the
  volume changes the bubble's length, not its cap curvature. Fixed volume is
  also the faithful analogue of the parent's incompressible gas phase, and it
  is what makes `R0 = 1` mean anything.
- The comparison film is the deposited central thickness
  `h = R_tube - r_interface`, taken as the median over the central 40% of the
  bubble, matching the parent's definition. It is not `b_min`, which sits in
  the rear capillary ripple and understates it by tens of percent, and it is
  not the seeded capsule gap. Encode mesh spacing as `Δ/R0`.
- `Ca_in` is imposed; `Ca_b = U` is the solved global unknown, not a fit to a
  tip trajectory.
- Report `bubble_pressure_excess`, not the raw gauge. The raw `P - p_out`
  contains the Poiseuille drop between the front cap and the outlet and is
  therefore window-dependent by construction: measured, it shifts by 0.5714 per
  tube radius of front pad at `Ca = 0.05`, against the analytic
  `8 Ca R_t/R_t^2 = 0.5714`.

## Measured facts about this solver

These were established by the checks in `verification/`; do not rediscover them.

- **A stationary solve from the seeded capsule never converges** for a confined
  bubble, at any capillary number from 0.5 down to 0.002. The cause is step
  length, not a missing solution: the first Newton step displaces interface
  nodes by about 16 film-element widths and elements invert, which is what the
  `Maximum residuals inf` message reports. Damping the same step to a fifth of
  its length reduces the residual norm by 22%, so the direction is sound. The
  unconfined control displaces nodes by half an element width and converges at
  full step.
  Damped Newton alone was tried and did not get there: `globally_convergent_newton=True`
  and `newton_relaxation_factor=0.2` were each given a 3000 s budget at
  `Ca = 0.05` and neither returned a converged result. That is a measured
  budget, not a proof of impossibility, but it is why the transient route is
  treated as the working one rather than as a convenience.
- **The route is therefore a short transient followed by a stationary polish**
  (`VoidBubbleProblem.solve_to_steady`). The transient is a homotopy into the
  Newton basin, **not** a relaxation to the answer: at `t = 20` the transient
  state still has `h_flat/R_t = 0.1201` with the film scatter still growing,
  and the stationary polish then moves it by +18% to 0.1422. Do not stop on a
  transient plateau. The converged answer is insensitive to the schedule.
- **The front pad controls window convergence; the rear does not.** Measured
  asymmetrically: `rear=1, front=5` gives the converged value while
  `rear=5, front=1` gives the pad-1 value. This is the Landau-Levich geometry -
  in the bubble frame the wall moves rearward, so the advancing front meniscus
  is the coating meniscus that sets the film.
- **Poiseuille at both ends is exact.** In the bare tube it reproduces the
  analytic pressure drop to 5.0e-14 and the flux to 4.4e-16, worst of the four
  resolution and length combinations S0 runs. The traction-free
  outlet carries a fixed *absolute* pressure deficit of 0.066 at `Ca = 0.05`,
  independent of window length, so its relative error falls as `1/L`. Keep it
  as a switch for comparison with the parent, not as production.
- **Inertia is a small, linear correction over the relevant range.** Between
  `La = 0` and `La = 1` at `Ca = 0.05` the film moves by −0.026%; `dh/Re` is
  constant to 2% between `Re = 0.068` and `Re = 0.679`. The parent's unswept
  `La = 1` is therefore not a confounder in any comparison.
- **The reported state is steady in the strong sense: displaced, it comes back.**
  The answer is a stationary solve, so persistence under time marching proves
  little on its own - a stationary solution is an exact fixed point of the
  implicit discretisation too, and marching from it stays on it to round-off
  whether it is an attractor or a saddle. It does stay: drift is under 6e-12 in
  `r/R_t` over at least 1.5 film renewals at every decade from `Ca_b = 0.001`
  to 1.06. The load-bearing check is the displacement test in
  `verification/s9_steadiness.py`: move the state to the converged solution at
  `Ca_in` 5% away, put the parameter back, and march in time alone with no
  stationary polish. The imposed interface displacement of 4.6e-4 to 3.6e-3 in
  `r/R_t` decays to between 0.03% and 0.57% of itself within 1.5 renewals, and
  `h` returns to within 3.8e-7 to 1.4e-5 relative.
- **Quote a march in film renewals, not in time units.** The film is renewed on
  `L_b/U_b`, which is 3172 time units at `Ca_b = 1.0e-3` and 0.51 at
  `Ca_b = 14.8` - a factor of 6000 across the sweep. A fixed 40-unit march is
  79 renewals at the top and 1.3% of one renewal at the bottom, so reported as
  a single number it claims far more at low `Ca` than it shows. S9 sets its
  horizon in renewals for this reason.
- **The two solve routes agree where both exist, and the ladder is not
  path-dependent.** A fresh transient start seeded from the self-consistent
  bubble speed does reach `Ca_in = 2`, i.e. `Ca_b = 4.79`; the committed sweep
  nevertheless switches to continuation from `Ca_in = 1` upward, for cost
  rather than necessity, and the `route` field records which produced each row.
  Where both work the answers agree to 9.1e-5 in `Ca_b` and 8.2e-5 in `h/R_t`
  (`Ca_in` = 1, 1.5, 2). Walking the ladder up to `Ca_in = 6` and back down
  returns the same states to 8e-13. `verification/s10_route.py`.
- **The film depends weakly but measurably on bubble length.** At fixed `R0`,
  changing `R_tube` from 0.7 to 0.5 lengthens the bubble by 1.8x and moves
  `h/R_t` by 1.31e-3 at `Ca_b = 0.011`, falling to 1.86e-4 at `Ca_b = 0.585`.
  This is resolved, not numerical: each confinement is self-converged to ~1e-5
  under doubled film elements and the difference is unchanged at 1.33e-3. The
  shorter bubble's film has not fully reached its plateau. It bounds how much
  of any cross-code difference can be attributed to a mismatch in `R0/R_tube`.
- **A pressure datum is required and the choice is free.** With Poiseuille at
  both ends, `pressure_datum="none"` fails to solve; `"fixation"` and
  `"integral"` agree on every reported observable to 2.6e-14.

## pyoomph traps that apply here

- `NavierStokesFreeSurface` resolves `static_interface="auto"` to *moving* only
  when a `BaseMovingMeshEquations` (here `HyperelasticSmoothedMesh`) is present
  on the bulk domain. Without it the interface silently becomes static: no mesh
  motion, no error, a plausible wrong answer.
- `DirichletBC` cannot depend on an unknown. The co-moving speed `U` is a
  global degree of freedom, so wall, inlet and outlet axial velocities use
  `EnforcedDirichlet`. Putting a `DirichletBC` and an `EnforcedBC` on the *same*
  component silently deactivates the enforced one - pin `velocity_x` strongly
  and enforce `velocity_y` weakly, never both on one component.
- `mass_density=0` is not the Stokes limit: the continuity residual carries
  `partial_t(rho)/rho` and raises a division by zero at code generation. Use
  `StokesEquations` instead, which is what `laplace=0` selects.
- Point constraints on the axis vanish under the axisymmetric measure `2*pi*r`.
  The optional tip closure passes `coordsys=cartesian` for this reason. The
  production closure is the centroid integral, which does not need the trick.
- `AxisymmetryBC` must sit on `axis_rear` and `axis_front`; it recurses onto
  `interface/axis_*` where the free surface adds fields.
- Do not multiply pyoomph integrals by `2*pi*r` again.
- A diverged Newton solve leaves the diverged values in the solution vector, so
  any fallback must checkpoint first or start from a fresh problem.
- Boundary names must be identical in both branches of `define_geometry`, or
  interpolation across a remesh raises.
- Angular (azimuthal) eigensolving is **refused** when the pressure datum is a
  fixation: `solve_eigenproblem(..., azimuthal_m=m)` raises
  `Do not use pressure_fixation with angular eigensolving`. Since
  `pressure_datum="fixation"` is this solver's default, azimuthal stability
  looks unavailable until the message is read. It is not: use
  `pressure_datum="integral"`, which is the `with_pressure_integral_constraint`
  the message names, and which `verification/s7_pressure_datum.py` shows agrees
  with the fixation on every reported observable to 2.6e-14.
- `setup_for_stability_analysis()` changes code generation and must be called
  **before** `initialise()`.
- A shift-invert eigensolve asked for too few eigenvalues can converge to a
  cluster that does not contain the rightmost one, and report a more negative
  growth rate with no warning. Measured in `verification/s12_azimuthal.py`:
  at four eigenvalues the `m = 0` rightmost comes out 41% too negative for one
  shift in three. Request at least eight.

## Repository conventions

- `problems/` holds equations, the mesh template and the capsule geometry.
- `cases/` holds portable `pyoomph-case-v2` documents.
- `verification/` holds staged checks against analytic answers; each is a
  script that exits non-zero on failure and writes a JSON receipt.
- `runners/` holds case drivers and sweeps.
- `tests/` must keep at least one path that imports no pyoomph.
- `runs/` is attempt/restart custody, never source.
- Component `AGENTS.md` is public-candidate. Private operational history stays
  in the project-context manual.
