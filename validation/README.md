# Validation

This directory contains executable comparisons with published results.

- [`herradaEtAl_figure2/`](herradaEtAl_figure2/): the gravity-free,
  axisymmetric bubble profiles of Herrada, Yu & Stone (2023), at imposed capillary numbers
  `Ca_l = -0.001` and `-0.02`. The comparison reproduces the thin deposited
  film and its departure from Bretherton's small-capillary-number law at the
  higher imposed speed, within the stated void-interior approximation.
  The [Newtonian validation report](https://github.com/VatsalSy/bretherton-drops-bubbles-fem-docs/blob/main/validation/newtonian-validation.tex)
  contains the digitized profile overlay and quantitative film comparison.

The benchmark uses the paper's tube-radius normalization, `b/R`, and compares
film laws at the solved bubble speed. Each benchmark documents its own scales
and model assumptions. Analytic equation checks belong in
[`verification/`](../verification/).

[`herradaEtAl_figure3/`](herradaEtAl_figure3/) supplies the gravity-driven
steady-branch continuation case at `Bo=1.56`, with a separate bounded smoke
case. Its target is the bubble length `L_b/R` versus signed imposed `Ca_l`.
The driver records turning-point candidates; a completed branch calculation
and comparison with the paper are required before claiming reproduction.
