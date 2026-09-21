# Herrada–Yu–Stone figure 2 benchmark

This folder provides the benchmark configuration for the gravity-free profiles
in figure 2 of Herrada, Yu & Stone, *J. Fluid Mech.* **958**, A45 (2023), DOI
[`10.1017/jfm.2023.123`](https://doi.org/10.1017/jfm.2023.123). Each case is a
complete `pyoomph-case-v1` document and each invocation runs exactly one case:

```bash
<environment>/bin/python validation/stone_figure2/driver.py run \
  --case validation/stone_figure2/cases/cal_m0p001_baseline.json \
  --output-dir /path/to/run/stone-figure2/cal_m0p001_baseline
```

The paper uses negative external capillary numbers because its imposed liquid
flow points down its `z` axis. At `Bo=0`, reflecting the axis gives the identical
problem with the solver's positive `ca_in=abs(Ca_l)`. No gravity or negative-flow
term is added. With `R0=1`, the paper volume `Vhat=3.3` fixes
`R_t=3.3^(-1/3)`. Its `H/R=15` window maps to a solver domain of `15 R_t`; the
seed rear tip is at `5 R_t`. The tip closure keeps that anchor fixed. The
exported paper coordinates are

```text
zhat = (min(z_solver) - z_solver) / R_t
hhat = 1 - r_solver / R_t
```

so the reflected profile runs from the anchored tip towards decreasing paper
`z`. `interface.csv` retains both solver and paper coordinates to make the
orientation auditable.

The liquid domain is an axisymmetric sharp-interface void approximation: the
gas is represented by a fixed-volume pressure constraint rather than a meshed
gas phase. The primary cases impose fully developed Poiseuille flow at both
ends. This differs from the paper's prescribed upper inlet and static-pressure
lower outlet. The model and boundary-condition differences are stated in the
[validation report](../../docs/validation/stone_figure2.tex), which includes
the comparison with digitized profiles from the paper.

The two cases reproduce the nearly uniform film and the capillary depression
near the tail. At the higher imposed speed, the deposited film is thinner than
Bretherton's small-capillary-number prediction and close to the
Aussillous–Quéré expression. See the report for the quantitative comparison
and the scope of this result. The `refined` cases provide a separate mesh
comparison; the reported overlay uses the two `baseline` cases.

Successful runs write `progress.jsonl`, `status.json`, `summary.json`,
`interface.csv`, `state.dump`, and profile plots as PDF and PNG. `summary.json`
is written only after the stationary solution passes the volume, flux,
flatness, and finite-value gates. A failed run exits non-zero and writes only a
failed `status.json` plus any partial output. If plotting fails after the
stationary solve has passed its checks, the accepted solution remains available
for the collection command.

Combine completed cases without rerunning them:

```bash
<environment>/bin/python validation/stone_figure2/driver.py collect \
  --results /path/to/run/stone-figure2/* \
  --output-dir /path/to/run/stone-figure2-combined
```

The combined receipt reports two numerical observations, when the requisite
accepted baselines are present: whether the small-`Ca` film is near
Bretherton's law and whether the larger-`Ca` film lies below Bretherton and is
closer to Aussillous–Quéré. These are checks of the produced results, not
assumptions used to accept a solve.
