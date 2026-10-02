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
- **Sliding interface nodes (`sliding_nodes=True`, off by default) are for
  following a rim that sharpens while it advances.** With the default
  pseudo-solid placement, interface nodes do not follow a rim that moves along
  its own axis, and they pack into the tip at about `V/R` (150 to 470 per unit
  time at `Ca_in = 12`, tip radius 2.2e-5 to 1.1e-5). That packing, not the
  physics, then caps the step. The option holds interface nodes on a prescribed
  tangential map (`problems/sliding_frame.py`) through a stiff tangential
  penalty on the mesh equations, while the kinematic condition still sets their
  normal motion. Measured:
  - Off, the solver is bit-for-bit unchanged: every dof of the `Ca_in = 12`
    station (stationary solve and five steps) and of a rim-collapse chunk (four
    steps) is identical to the base commit.
  - On, the same station moves by 4e-8 in tip radius and 8e-11 in `Ca_b`. The
    analytic Jacobian agrees with finite differences to 2e-7. S20 drags nodes
    0.115 along a static sphere and its radius stays within 3.1e-4, its Laplace
    pressure within 1.3e-5 and its volume within 2e-15.
  - In the rim collapse at `Ca_in = 12`:
    - the tip resolution no longer grows within a chunk (16 elements per tip
      radius held where the default packed it to 30 within a chunk);
    - steps are set by the collapse rate (`lambda dt = 0.004`) instead of by
      node compression, and a march from tip radius 2.3e-5 agrees with its
      halved-step repeat to a median 0.2%;
    - remesh jumps of the tip radius have a median of 4e-4.

  How to use it:
  - Re-lock the frame on the measured rim every step
    (`sliding_frame.relock`). A rim velocity known only to `dV` packs or
    spreads the nodes again at `dV/R`. Start each stretch from the velocity
    measured over the previous one, however short it was.
  - Call `rebase_constraints` after every remesh and every load. A remesh
    changes the void's volume and axial moment by about 1e-5, and fixed
    targets would make the next step inflate or shift the whole bubble by
    several tip radii at once, which nodes held tangentially around a sharp
    tip cannot follow. The moment target, a gauge, is re-based; the volume is
    relaxed back to `4/3 pi R0^3` over a set time.
  - Move the Newton initial guess with the frame before each step (a hook in
    `actions_before_newton_solve`, after the history shift). Steps then
    converge in about three iterations even when the tip moves ten tip radii
    per step.
  - Use translation only, with a window much wider than the rim's
    displacement between remeshes (3e-2 here). Contracting the nodes about the
    tip, or a narrow window, left the tip radius 0.1-0.3% away from a fresh
    remesh of the same state after 30 steps; translation in a 7e-3 window left
    it within 5e-5.
  - Remesh when the curvature-maximum radius falls below 0.993 of the fitted
    tip radius, or when the tip resolution leaves its bounds.
  - States load either way, because no field and no global parameter is
    added. Leave the option off for eigenproblems.

## Measured facts at high capillary number

These come from the stability campaign reported in `docs/validation/`; do not
rediscover them.

- **Audit the rim, not the film.** Above `Ca_b ~ 10` the re-entrant rear
  closes into a rim whose radius decays roughly as `exp(-0.5 Ca_in)`. On
  distance-graded meshes `R_lip/ds` falls to 0.31 by `Ca_in = 14` while every
  film diagnostic looks healthy, and the continuation produces a turning point
  near `Ca_in = 14.86` that does not exist. Use `lip_elements` and admit a
  station only with `R_lip/ds >= 8`.
- **The element floor is a driver setting, not a scale.** The curvature
  grading is clamped at `film_element_size / lip_floor_divisor`. The default
  divisor of 16 gives a floor near `2e-3`, which loses the rim near
  `Ca_in = 8`; a resolved branch needs the driver to reset the floor from the
  last measured `R_lip` before every remesh. A march that ends because
  `R_lip` fell below the floor reports the setting, not physics: with the floor
  lowered tenfold the same marches continue until the remesher or Newton fails.
- **Raise `newton_tolerance` to `1e-7` when the rim nears the floor.** There the
  nonlinear residual stalls at about `1.1e-8`, just above the default `1e-8`, so
  every step is rejected and the march stops advancing while its clock runs. A
  residual that has fallen seven decades is converged.
- **Interface nodes pack into an advancing rim.** Nodes move only normally, so a
  rim that advances along its own axis crowds them into its tip. That, not the
  element floor, caps the time step in a resolved rim collapse, at a cost of
  hours per e-fold of `R_lip` near `3e-6`. Going deeper needs tangentially
  sliding interface nodes.
- **Keep the rim near the coordinate origin.** The projection of mesh nodes onto
  the interface spline has a tolerance proportional to the local spacing; with
  rim elements near `1e-6` and coordinates near 13 it falls below coordinate
  rounding and the remesh raises `Cannot invert spline`. Placing the window so
  the rim sits near `z = 0` recovers about seventyfold headroom. Subdividing the
  polyline does not help.
- **Evaluate every station on a mesh adapted to its own `Ca_in`.** A chain
  continued downward on a frozen mesh biases the stability thresholds by
  `+0.027` and `+0.131`. Re-stationing a state onto an enlarged or translated
  window failed in four attempts of four; a window study needs a fresh climb.
- **Solve the stability pencil by its reduced finite spectrum**
  (`verification/eigen_helpers.py`). `M` has nonzero rows only on the kinematic
  condition, so `G = M_R J^-1 E` gives every finite eigenvalue densely. A
  shift-invert search near zero returns the pressure-datum null structure
  instead, and a screen on which unknowns carry an eigenvector's amplitude
  rejects the physics, because every genuine eigenvector is pressure-dominated.
  Judge modes by the scale-free residual.
- **The finite count belongs to the rule; the unstable count does not.** The
  fixed cutoff `|mu| < 1e-12 max|mu|` and the error-bar rule
  `|mu| < 10 kappa delta ||G||` retain 901 against 362 finite eigenvalues at
  `Ca_in = 12`, `k = 0`, but agree on every eigenvalue with positive real part.
  When `J_k` is nearly singular (`k = 1` at `Ca_in = 13.6`, `lambda ~ -0.01`)
  the error-bar rule classes everything infinite; reduce the shifted pencil
  `(J + 1.5 M, M)` and add the shift back. Some shifted `k = 1` stations still
  retain only 5 to 27 finite eigenvalues; their leading value then rests on its
  residual and an independent shift-invert control.
- **A shift-invert control needs a start vector in the operator's range.** The
  row and column supports of `M` are disjoint, so `M r` is annihilated and
  ARPACK reports a zero starting vector; use
  `shift_invert_start_vector`.
- **Remeshing rules that preserve limit-cycle amplitudes.** Remeshing whenever
  the interface deviates from the base state by `3e-3`, or only when it has
  moved `4e-3` since the last remesh or `R_lip/ds < 8`, gives the same
  saturated amplitude to 0.5% at `Ca_in = 10.45`, as do a halved time step and
  `lip_elements = 16`. A mesh rebuilt once at the displaced seed shape and then
  frozen is wrong: it decays nearly ten times too fast and settles to a
  different steady state with the rim still resolved. Remeshing every chunk
  adds an extra damping of order `0.01` at small amplitude, so read the cubic
  coefficient from runs whose remeshing does not change during the fit.
- **Quote marched growth rates corrected for BDF2.** The integrator amplifies
  slightly faster than `lambda` when `lambda dt` is not small: rates of 12.26
  and 12.20 at `dt = 0.01` and `0.0025` invert to 12.1992 and 12.1993 against
  the eigenvalue 12.1990. The chunk length must also track the rate: `0.1` suits
  `lambda ~ 13`, while `lambda ~ 48` needs `0.02`.

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
  growth rate with no warning. Request at least eight, or avoid the search
  altogether with the reduced finite spectrum above.

## Repository conventions

- `problems/` holds equations, the mesh template and the capsule geometry.
- `cases/` holds portable `pyoomph-case-v2` documents.
- `verification/` holds staged checks against analytic answers; each is a
  script that exits non-zero on failure and writes a JSON receipt.
- `runners/` holds case drivers and sweeps.
- `tests/` must keep at least one path that imports no pyoomph.
- `runs/` is attempt/restart custody, never source.
- Component `AGENTS.md` and the docs carry measured properties of the solver,
  never operational history; that stays in the project-context manual.
