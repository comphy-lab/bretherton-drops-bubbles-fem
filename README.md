# Bretherton void-bubble FEM

Axisymmetric sharp-interface finite-element solver for a gas bubble translating
through a liquid-filled circular tube, computed in a frame co-moving with the
bubble. Only the liquid is meshed: the bubble interior is a true void, and
surface tension acts on the liquid/void interface. There is no gas domain and
no volume-of-fluid representation.

The companion Basilisk two-phase solver lives in a separate repository. This
codebase is the free-surface counterpart, and the two are set up to be compared
point by point rather than only in shape.

## Why a co-moving frame

A lab-frame calculation has to give the bubble somewhere to travel, so its
domain length is set by the trajectory rather than by the physics, and it has
to march long enough for the film to settle — `travelR/Ca = 160` time units at
`Ca = 0.05` for the parent code, in a domain of `16 R0`.

In a co-moving frame the bubble does not move through the mesh at all. The
window then has to be long enough only for the cap disturbance to decay to the
fully developed profile, and the steady state is obtained by solving the steady
equations rather than by marching until a convergence criterion fires. The
bubble speed `Ca_b` stops being a regression against a tip trajectory and
becomes a solved unknown.

The far field is not an approximation. In a circular tube the fully developed
solution *is* Poiseuille, so imposing it at both ends is the infinite-tube
condition, and `verification/s0_poiseuille.py` confirms it reproduces the
analytic pressure drop to 5.0e-14 and the flux to 4.4e-16 in the bare tube. What
remains to be measured is how much pad the caps need, which is what
`verification/s5_window.py` does.

## Results

**The scientific results live in `docs/validation/`, not here.** That report
covers the deposited film over five decades of `Ca_b`, the recovery of the
Bretherton coefficient in the thin-film limit, the comparison with published
axisymmetric computations and with the companion volume-of-fluid campaign, and
the re-entrant rear cap at high capillary number. Build it with

```bash
make -C docs/validation
```

Every number it plots or tabulates is vendored as a compact CSV under
`docs/validation/data/`, and its figures regenerate from those alone, without a
solver or a simulation dump.

This file documents the code: what it computes, how it is arranged, how to run
it and what each verification stage asks. Measured properties of the solver that
are not campaign results live in `AGENTS.md`.

### The stagnant-film identity

With Poiseuille imposed at both ends, the bubble-frame flux through every
cross-section is the same, so a film at rest in the laboratory frame implies

```
Ca_b / Ca_in = (R_tube / (R_tube - h_flat))^2
```

exactly. This is the sharpest self-check available, because it couples the frame
speed, the film thickness and the flux in one identity and fails if any of them
is wrong. It is reported per case, never enforced: a film that is genuinely not
stagnant is physics, not error. The residual across the campaign is in the
validation report.

### Window independence

The claim that a short window is legitimate is tested by the quantity that
would expose it if it were false: the difference between the exact Poiseuille
outlet and the traction-free one at the same pad. If the window is effectively
infinite, the outlet condition cannot matter, so this gap must collapse faster
than either condition converges on its own. It does, by eight orders against
four; `verification/s5_window.py` writes the table.

Varying the pads separately shows that the **front** pad controls and the rear
does not. This is the Landau-Levich geometry: in the bubble frame the wall moves
rearward, so the advancing front meniscus is the coating meniscus that sets the
film and the rear meniscus only receives it. A symmetric-only study would have
concealed it. In practice `pad_rear = 1.5`, `pad_front = 3` suffices, a window
of about `7 R0` against the parent's 16, with no trajectory to contain.

## Verification

Each stage is a script that exits non-zero on failure and writes a JSON receipt
to `receipts/`. They are ordered so that a failure localises.

| stage | question | status |
|---|---|---|
| `s0_poiseuille.py` | Does the bare window reproduce Poiseuille exactly? | flux `4.4e-16`, `Δp` `5.0e-14`, worst of four |
| `s1_static_sphere.py` | Does an unconfined void relax to a sphere with a Laplace pressure of 2? | radius to `1.1e-5`, `P` to `2.2e-6` |
| `s5_window.py` | Is the answer independent of the window length? | see below |
| `s6_mesh.py` | Is it mesh-converged, and to what? | see below |
| `s7_pressure_datum.py` | Does the pressure datum matter, and is it working? | datum required; two mechanisms agree to `2.6e-14` |
| `s8_inertia.py` | Does inertia matter, and is the non-inertial frame term exercised? | linear in `Re`; `-0.0255%` from `La = 0` to `1` |
| `s9_steadiness.py` | Is the stationary state an attractor, and over how many film renewals is that shown? | displaced, it returns to within 0.6% of the displacement in 1.5 renewals |
| `s10_route.py` | Does the answer depend on the solve route, the ladder direction, or the bubble length? | routes agree to `9e-5`; retrace to `8e-13`; bubble length `1.3e-3` |
| `s11_rear_cap.py` | Where does the rear cap stop being convex, and is that onset resolved? | onset in `1.024 < Ca_b < 1.048` |

`s8` is also the only stage that executes the frame-acceleration body force
`-rho dU/dt`, which makes the co-moving frame consistent while `U` is still
changing. It exists only for `La > 0` — `La = 0` selects `StokesEquations` —
and only during the transient, since a stationary solve zeroes time
derivatives. Without this stage that term would be written, plausible and never
run.

`s9` and `s10` were added because the earlier stages all measure how well the
discrete equations are solved, and none of them can fail if the solver converges
neatly to the wrong state. `s9` asks whether the converged state is an attractor
rather than merely a fixed point — a stationary solution is an exact fixed point
of the implicit time discretisation as well, so marching from it proves almost
nothing and the stage displaces it instead. `s10` asks whether the curve carries
its own history, since points reached by continuation inherit their
predecessor's state.

`s1` is the one that matters most for trusting the formulation: it checks the
surface tension and its sign, the kinematic condition, the mesh motion, the
fixed-volume constraint and the sign of the void pressure, all at once, against
a number known in closed form. A confined bubble has no analytic static
equilibrium, so the tube is widened rather than the flow switched off.

## Layout

```
problems/       equations, mesh template, capsule geometry
cases/          portable pyoomph-case-v2 documents
verification/   staged checks against analytic answers
runners/        case driver and Ca sweep
postprocess/    deposited-film extraction and reference laws
receipts/       JSON receipts written by the drivers and checks
Documentation drafts: VatsalSy/bretherton-drops-bubbles-fem-docs (explicit approval required before promotion)
```

## Requirements

- Python 3.13
- `uv` for the locked environment
- pyoomph from the CoMPhy fork, pinned at tag `comphy-2026.09.07.1`
  (`a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618`)

The environment is created with `pyoomph-env`; do not `pip install` into an
ambient interpreter. A source build compiles vendored oomph-lib and is slow on
a cold start.

## Usage

Software-contract tests need no pyoomph, so they run anywhere:

```bash
uv run --no-sync pytest tests/
```

Solve the default case and print the result with its checks:

```bash
<env>/bin/python runners/run_case.py --case cases/bubble_ca0p05.json
```

Sweep the parent campaign's capillary numbers:

```bash
<env>/bin/python runners/sweep.py
```

Run a verification stage:

```bash
<env>/bin/python verification/s1_static_sphere.py
<env>/bin/python verification/s5_window.py --asymmetric
<env>/bin/python verification/s9_steadiness.py
<env>/bin/python verification/s10_route.py
```

Build the validation report. This needs LaTeX, NumPy and Matplotlib, but not
pyoomph and no simulation output:

```bash
make -C docs/validation figures PYTHON=<env>/bin/python
make -C docs/validation
```

## Film thickness

The comparison thickness is the deposited central film
`h = R_tube − r_interface`, taken as the median interface radius over the
central 40% of the bubble — the parent's definition, reproduced so the two
codes report the same quantity. It is not the minimum gap, which sits in the
rear capillary ripple and understates the flat film by tens of percent,
increasingly so as `Ca` falls. Mesh spacing is quoted as `Δ/R0`.

`postprocess.film.measure_film` also returns the scatter of the gap across the
central window. That is the evidence that the film is flat there at all: a
bubble too short to develop a flat film shows a scatter comparable to the film
itself, and no amount of averaging repairs that.
