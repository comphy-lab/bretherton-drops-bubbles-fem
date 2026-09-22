# Herrada–Yu–Stone figure 3 continuation

This package defines the gravity-driven continuation corresponding to figure 3
of Herrada, Yu & Stone, *J. Fluid Mech.* **958**, A45 (2023), DOI
[`10.1017/jfm.2023.123`](https://doi.org/10.1017/jfm.2023.123). It sets
$Re=0$, $Bo=1.56$, $\widehat V=3.3$ and $H/R=15$. The upper bubble tip is fixed
at solver coordinate $z=10R$, giving the paper coordinate
$\widehat z=z/R-10$.

The plotted length is the meridional interface arc length defined in equation
(2.3), $L_b=\int_0^1\sqrt{(\partial_s G_b)^2+(\partial_s F_b)^2}\,ds$.
The driver integrates the quadratic interface elements in curve order and
reports `bubble_length_over_r`; `axial_span_over_r` separately records the
axial extent. The arc-length integral has no axisymmetric surface-area factor.

The calculation uses the repository's sharp-interface void approximation and
fully developed Poiseuille conditions at both ends. It therefore tests the
branch structure under a stated model approximation; agreement with the paper
is not assumed. Signed paper capillary numbers are used directly because the
gravity term removes the reflection symmetry used by the figure 2 benchmark.

The steady continuation reproduces the three-branch diagram in the reference.
At `Ca_l=0.02`, the meridional length is `L_b/R=7.9897`, compared with
approximately `8.00` in the published figure; at `Ca_l=-0.02`, it is `6.6209`,
compared with approximately `6.61`. The
[Newtonian validation report](../../docs/validation/newtonian-validation.tex)
contains the comparison and the model assumptions. The agreement concerns
steady branch geometry. The report's mesh and sampling diagnostic does not
yet establish full-branch mesh or domain-length convergence. Stability and
transient branch switching remain separate tests.

The production case first obtains a stationary seed at $Bo=0$ and
$Ca_l=0.02$, continues in Bond number to 1.56 at fixed capillary number, and
then follows `Ca_in` with pyoomph's native pseudo-arclength continuation. The
continuation uses the fixed `ndof` metric, and the returned signed step is reused
unchanged, allowing the trace to pass turning points. Native continuation data
are included in every state checkpoint so a resumed trace retains its tangent.
Tracing ends only after $Ca_l\leq-0.02$ and at least two chronological parameter
reversals have been observed, or when the explicit step budget is exhausted.
Reversals are recorded only as candidate turns; the driver assigns no stability
label.

```sh
uv run --no-sync python validation/herradaEtAl_figure3/driver.py run \
  --case validation/herradaEtAl_figure3/case.json \
  --output-dir /path/to/fresh/run
```

Each accepted station adds one row to `branches.csv`, an interface profile and
a native state checkpoint. `restart.json` binds that checkpoint to the exact
case hash and stores the next signed arclength step. Resume always writes to a
fresh directory and retains the parent-run path:

```sh
uv run --no-sync python validation/herradaEtAl_figure3/driver.py run \
  --case validation/herradaEtAl_figure3/case.json \
  --output-dir /path/to/fresh/resume \
  --resume /path/to/parent/run/restart.json
```

The restart descriptor also binds the state file and accepted branch prefix
by SHA-256. If interruption leaves newer rows in the CSV than in the last
checkpoint, only the checkpointed prefix is resumed; the source run is kept.
Archive the parent runs and referenced mesh files as well as the latest
segment. Reaching the step budget returns `incomplete`, not a reproduction
claim. The full step budget is retained across resumes.

Restart descriptors use version 2, which separates meridional length from axial
extent. Version 1 runs retain their original tables and checkpoints. Recover
their comparison table from the saved profiles into a fresh directory:

```sh
uv run --no-sync python -m validation.herradaEtAl_figure3.remeasure \
  --branches /path/to/run/branches.csv --profiles /path/to/run/profiles \
  --output-dir /path/to/fresh/comparison
```

This records input hashes and quadrature checks; it does not convert restart
descriptors or alter the original run. Use the original source version if a
version 1 continuation must be resumed.

`case.json` is copied into each output directory, and `progress.jsonl` and
`status.json` report progress and termination. A nonempty output directory is
never reused. Keep outputs outside the source checkout. Plotting is a separate
command and preserves continuation order rather than sorting by capillary
number.

The bounded smoke case initializes the actual $Bo=1.56$, $Ca_l=-0.002$
forcing on a coarse mesh and takes three $10^{-5}$ transient steps. It checks
finite outputs, volume and flux residuals, and geometric containment. It does
not trace a branch or validate hysteresis.

```sh
uv run --no-sync python validation/herradaEtAl_figure3/driver.py smoke \
  --case validation/herradaEtAl_figure3/smoke.json \
  --output-dir /path/to/fresh/smoke
```

Plot an existing ordered branch table without rerunning the solver:

```sh
uv run --no-sync python validation/herradaEtAl_figure3/driver.py plot \
  --case validation/herradaEtAl_figure3/case.json \
  --output-dir /path/to/run
```

The continuation API is pinned to pyoomph `comphy-2026.09.07.1`
(`a186e07e1a7ea6b74ecc4f3a3fc39cc763d41618`).

The explicit software probes in `tests/smoke_herrada_forcing.py` and
`tests/smoke_herrada_continuation.py` exercise the Stokes forcing/constraints
and a scalar fold with a native-state restart, respectively. Each requires
a fresh `--output-dir`; neither establishes agreement with figure 3.
