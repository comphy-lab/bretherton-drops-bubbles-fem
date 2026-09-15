# Vendored comparison data

Two external series are kept here so the comparison figures render from the
checkout alone. Neither is produced by this solver.

## `parent-basilisk-h-ca.csv`

The parent Basilisk campaign's deposited-film table: 26 cases, `Ca_b` from
0.0054 to 2.33, at refinement levels 9 to 13.

Source: `bretherton-drops-bubbles` project scratch,
`newtonian-central-film-2026-09-12/comparison-h-ca.csv`, copied 2026-09-15.

**These are unpublished results of the parent project.** They are reproduced
here at Vatsal's explicit request to put the Basilisk points on the FEM
comparison figure. They are not this project's to publish onward; anything
built from them that leaves this repository needs the parent project's
agreement.

The `outlet_limited` column matters and must be honoured in any plot. A case
flagged `True` reached the end of its lab-frame domain, so its film carries a
domain-truncation error; two of the 26 are flagged. `min_cells` is the number
of cells across the deposited film and is the parent's own resolution measure —
its solver warns below four.

## `giavedoni-saita-1997-axisym.csv`

Giavedoni & Saita, *Phys. Fluids* **9** (1997), axisymmetric finite-element
computations of the same problem: 24 points, `Ca` from 5.3e-5 to 10.

Published literature, digitised from the paper's figures. The `source` column
records which figure, and `note` records the pixel coordinates read. Low-`Ca`
points therefore carry digitisation error of a few per cent, which is the right
scale to compare against and not a disagreement with the original work.

This is the canonical axisymmetric FEM study of a bubble in a tube, so it is
the closest independent check available for this solver: same geometry, same
formulation class, an independent implementation, published.
