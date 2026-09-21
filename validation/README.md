# Validation

This directory contains executable comparisons with published results.

- [`stone_figure2/`](stone_figure2/): the gravity-free, axisymmetric bubble
  profiles of Herrada, Yu & Stone (2023), at imposed capillary numbers
  `Ca_l = -0.001` and `-0.02`. The comparison reproduces the thin deposited
  film and its departure from Bretherton's small-capillary-number law at the
  higher imposed speed, within the stated void-interior approximation.
  The [LaTeX report](../docs/validation/stone_figure2.tex) contains the
  digitized profile overlay and quantitative film comparison.

The benchmark uses the paper's tube-radius normalization, `b/R`, and compares
film laws at the solved bubble speed. Each benchmark documents its own scales
and model assumptions. Analytic equation checks belong in
[`verification/`](../verification/).
