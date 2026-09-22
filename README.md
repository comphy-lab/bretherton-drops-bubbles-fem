# Bretherton void-bubble FEM

Axisymmetric sharp-interface finite-element solver for a gas bubble translating
through a liquid-filled circular tube, computed in a frame co-moving with the
bubble. Only the liquid is meshed: the bubble interior is a true void, and
surface tension acts on the liquid/void interface. There is no gas domain and
no volume-of-fluid representation.

The companion Basilisk two-phase solver lives in a separate repository. This
codebase is the free-surface counterpart, and the two are set up to be compared
point by point rather than only in shape.

## Literature validation

The [Herrada–Yu–Stone benchmark](validation/herradaEtAl_figure2/) reproduces the
gravity-free film profiles in figure 2 of Herrada, Yu & Stone (2023) at
`Ca_l = -0.001` and `-0.02`. The
[figure 3 benchmark](validation/herradaEtAl_figure3/) reproduces the steady
three-branch response at `Bo = 1.56`, including the folds of the multivalued
relation between imposed flow and meridional interface length. The
[Newtonian validation report](docs/validation/newtonian-validation.tex) compares the
finite-element profiles with digitized reference curves and records the
quantitative agreement in film thickness and interface length, together with
the model assumptions. The report also separates mesh differences from
continuation sampling; full-branch mesh and domain convergence remains
unestablished. Stability and transient branch switching are separate
validation targets. Its
[build instructions](docs/validation/README.md) generate a standalone PDF.

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
analytic pressure drop to 3e-14 and the flux to 2e-16 in the bare tube. What
remains to be measured is how much pad the caps need, which is what
`verification/s5_window.py` does.

## Results

Five capillary numbers, matching the parent campaign's sweep, at `La = 0`
(Stokes), `R_tube/R0 = 0.7`, bubble volume `4/3 pi R0^3`. `h_flat` is the
deposited central film; AQ is the Aussillous–Quéré fit evaluated at the
*measured* `Ca_b`, as the parent does.

| `Ca_in` | `Ca_b` | `h_flat/R_tube` | AQ at `Ca_b` | deviation | parent `h/R_t` |
|---|---|---|---|---|---|
| 0.05  | 0.0679131 | 0.142165 | 0.143028 | −0.60% | 0.143663 |
| 0.02  | 0.0239223 | 0.085809 | 0.086915 | −1.27% | 0.086149 |
| 0.01  | 0.0112357 | 0.056669 | 0.057458 | −1.37% | 0.056791 |
| 0.005 | 0.0053892 | 0.036818 | 0.037281 | −1.24% | 0.036635 |
| 0.002 | 0.0020846 | 0.020502 | 0.020697 | −0.94% | *not run* |

Each point costs a few seconds on ~16k degrees of freedom. Every point carries
its checks: void volume conserved to `1e-15`–`1e-10`, inlet and outlet fluxes
matching to `1e-15`, and the stagnant-film identity below.

### The stagnant-film identity

With Poiseuille imposed at both ends, the bubble-frame flux through every
cross-section is the same, so a film at rest in the laboratory frame implies

```
Ca_b / Ca_in = (R_tube / (R_tube − h_flat))²
```

exactly. This is the sharpest check available, because it couples the frame
speed, the film thickness and the flux in one identity and fails if any of them
is wrong. This solver satisfies it to between `1e-5` and `5e-4` across the
sweep; the parent's own four committed cases satisfy it to `1.3e-3`. The
residual is reported, not asserted: a film that is genuinely not stagnant is
physics, not error.

### Window independence

The claim that a short window is legitimate is tested by the quantity that would
expose it if it were false: the difference between the exact Poiseuille outlet
and the traction-free one at the same pad. If the window is effectively
infinite, the outlet condition cannot matter, so this gap must collapse faster
than either condition converges on its own.

Relative gap in `h_flat/R_tube` between the two outlets, symmetric pads in tube
radii:

| `Ca` | pad 1 | pad 2 | pad 5 |
|---|---|---|---|
| 0.05  | 1.29e-3 | 5.05e-6 | 3.02e-11 |
| 0.005 | 5.07e-5 | 6.67e-7 | 1.14e-12 |

Eight orders of magnitude, against four for the individual convergence — the
claim survives the test that could have killed it. The asymptotic decay is
`e^(-4.0)` per tube radius, consistent with the classical Papkovich–Fadle decay
of a Stokes disturbance in a pipe.

The void pressure behaves differently, and the difference is itself a result.
Its cross-outlet gap is −1.65% at `Ca = 0.05` and does **not** shrink with pad
(−1.37%, −1.66%, −1.65% at pads 1, 2, 5). That is not a window effect: the
pressure excess is referenced to the mean outlet pressure, so comparing it
across two different pressure boundary conditions compares two data, not two
windows. It is the traction-free outlet's own pressure deficit, and the
agreement is quantitative:

| `Ca` | pad | implied absolute offset | S0 bare-tube deficit |
|---|---|---|---|
| 0.05  | 2 | −0.0662 | 0.0663 |
| 0.05  | 5 | −0.0656 | 0.0663 |
| 0.005 | 5 | −0.0066 | 0.00663 (linear in `Ca`) |

A quantity measured in the bare tube with no bubble present predicts the offset
in the full bubble calculation to better than 1%, at two capillary numbers a
decade apart, with the correct linear scaling. Neither stage was constructed to
predict the other. The pressure's *window* independence is therefore tested
where it belongs — against pad within a single outlet condition, where it holds
to `1.7e-6`.

**The front pad controls; the rear does not.** Varying one at a time at
`Ca = 0.05`, with the other held at 5 tube radii, the film moves by

| varied | 1 | 2 | 3 |
|---|---|---|---|
| rear only  | 3.0e-7 | 2.3e-7 | −1.6e-7 |
| front only | 1.8e-4 | −2.7e-6 | 1.3e-7 |

a factor of 600 at pad 1. This is the Landau–Levich geometry: in the bubble
frame the wall moves rearward, so the advancing *front* meniscus is the coating
meniscus that sets the film, and the rear meniscus only receives it. A
symmetric-only study would have hidden this. In practice `pad_rear = 1`,
`pad_front = 3` suffices — a window of about `7 R0` against the parent's 16,
with no trajectory to contain.

### Mesh convergence

Refining the elements across the film (2, 3, 4, 6) and the bulk element size
(0.16, 0.10, 0.06) independently, the reported observables move by less than
`7e-5` relative at `Ca = 0.05` and less than `3e-5` at `Ca = 0.005`. The spread
is small enough that a convergence *order* cannot be extracted — a Richardson
ratio on such a sequence returns nonsense, and `verification/s6_mesh.py`
reports that rather than quoting it.

This matters for how the comparison with the parent should be read. The fair
comparison is each code's deviation from the same reference law evaluated at its
own measured `Ca_b`:

| `Ca_in` | this solver | parent | difference |
|---|---|---|---|
| 0.005 | −1.24% | −1.89% | +0.65% |
| 0.01  | −1.37% | −1.35% | −0.02% |
| 0.02  | −1.27% | −1.06% | −0.21% |
| 0.05  | −0.60% | +0.21% | −0.81% |

The two agree to 0.02% at `Ca = 0.01` and diverge at both ends of the range. The
shapes differ: this solver's deviation is roughly flat at −0.6% to −1.4%, while
the parent's runs monotonically from −1.9% to +0.2%.

That difference is resolved on this side — the discretisation uncertainty here
is `~1e-4`, one to two orders below it. Its cause is **not established**. Two
candidates, and they are distinguishable:

1. The parent's bubble is not a void (`mu_gas/mu_liq = 0.01`, `rho_gas/rho_liq
   = 0.001`). A finite gas viscosity is a real physical difference, small but
   not obviously negligible at the top of the range.
2. The parent has no committed refinement sequence at all. Its own report
   records that its four baseline films span between 1.6 and 6.4 of its
   smallest cells — below its own four-cell warning threshold at the two lowest
   capillary numbers. Under-resolution that worsens as the film thins would
   produce a monotone trend in `Ca` of the kind observed.

A third candidate has been **eliminated**. The parent fixes `La = 1` and never
sweeps it, so inertia was a possible confounder. It is not: `s8_inertia.py`
measures the film moving by −0.026% between `La = 0` and `La = 1` at
`Ca = 0.05`, thirty times smaller than the discrepancy. The correction is clean
and linear in `Re = La Ca_b` — `dh/Re` is `-3.75e-3` at `Re = 0.068` and
`-3.67e-3` at `Re = 0.679`, constant to 2% across a factor of ten — which is
both what a leading-order inertial correction should look like and the evidence
that the frame-acceleration term is doing something physical rather than
numerical.

The monotone trend is more suggestive of (2) than of (1), since a gas-viscosity
effect should grow with `Ca` rather than change sign. But that is a hypothesis,
not a finding: settling it needs a refinement sequence on the parent side, which
does not yet exist.

## Verification

Each stage is a script that exits non-zero on failure and writes a JSON receipt
to `receipts/`. They are ordered so that a failure localises.

| stage | question | status |
|---|---|---|
| `s0_poiseuille.py` | Does the bare window reproduce Poiseuille exactly? | flux `2e-16`, `Δp` `3e-14` |
| `s1_static_sphere.py` | Does an unconfined void relax to a sphere with a Laplace pressure of 2? | radius to `1.1e-5`, `P` to `2.2e-6` |
| `s5_window.py` | Is the answer independent of the window length? | see below |
| `s6_mesh.py` | Is it mesh-converged, and to what? | see below |
| `s7_pressure_datum.py` | Does the pressure datum matter, and is it working? | datum required; two mechanisms agree to `1e-14` |
| `s8_inertia.py` | Does inertia matter, and is the non-inertial frame term exercised? | linear in `Re`; see below |
| `s9_steadiness.py` | Is the stationary state an attractor, and over how many film renewals is that shown? | displaced, it returns to within 0.6% of the displacement in 1.5 renewals |
| `s10_route.py` | Does the answer depend on the solve route, the ladder direction, or the bubble length? | routes agree to `9e-5`; retrace to `8e-13`; bubble length `1.3e-3` |

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
validation/     executable benchmarks against published results
runners/        case driver and Ca sweep
postprocess/    deposited-film extraction and reference laws
docs/validation/ LaTeX validation reports, references and figures
receipts/       JSON receipts written by the drivers and checks
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
