# Verification

Software-contract tests live in `tests/`. This folder holds numerical checks of
the discrete equations: each stage is a script that exits non-zero on failure
and writes a JSON receipt beside `receipts/`.

A stage earns its place by being able to fail. Restating how well Newton
converged is not a check; every stage below compares against something the
solver did not choose - an analytic answer, an independent implementation, a
different route to the same state, or a limit in which the answer is known.

| Stage | Question | Receipt |
|---|---|---|
| `s0_poiseuille.py` | Does the bare window reproduce Poiseuille flow? | `s0.json` |
| `s1_static_sphere.py` | Does a static free surface give the Laplace jump? | `s1_static_sphere.json` |
| `s5_window.py` | Is the answer independent of the window length? | `s5_window.json` |
| `s6_mesh.py` | Is it independent of the mesh? | `s6_mesh.json` |
| `s7_pressure_datum.py` | Does the pressure datum affect any observable? | `s7_pressure_datum.json` |
| `s8_inertia.py` | How large is the inertial correction, and is the frame-acceleration term exercised? | `s8_inertia.json` |
| `s9_steadiness.py` | Is the stationary state an attractor, and over how many film renewals is that shown? | `s9_steadiness.json` |
| `s10_route.py` | Does the answer depend on the solve route, the ladder direction, or the bubble length? | `s10_route.json` |
| `s11_rear_cap.py` | Where does the rear cap stop being convex, and is that onset resolved? | `s11_rear_cap.json` |
| `s12_azimuthal.py` | Does azimuthal eigensolving run here, and how many eigenvalues does it need? | `s12_azimuthal.json` |
| `s13_eigen_dynamic.py` | Does the rightmost eigenvalue match a dynamic march? | `s13_eigen_dynamic.json` |
| `s14_azimuthal_sweep.py` | Which sweep states are stable at k = 0, 1 and 2? | `s14_azimuthal_sweep.json` |
| `s15_inertia_bound.py` | To what Reynolds number may the Stokes baseline be quoted? | `s15_inertia_bound.json` |
| `s16_pressure_datum_spectrum.py` | Does the pressure datum move the axisymmetric spectrum? | `s16_pressure_datum_spectrum.json` |
| `s17_la1_barrier.py` | Does La=1 continuation past Ca_in=2.91 survive with remesh off? | `s17_la1_barrier.json` |
| `s18_azimuthal_k3.py` | Which sweep states are stable at k = 3? | `s18_azimuthal_k3.json`, `s18_azimuthal_k3_highca.json` |
| `s19_la_grid.py` | What is H at the named (Ca_in, La) grid, and do the windows hold? | `s19_la_grid.json` |

Stage numbering has gaps because S2 to S4 were folded into the production path
during development rather than kept as separate scripts.

The S14 and S18 labels come from a spurious-mode screen based on where each
eigenvector's support lies. That screen rejects genuine modes of this pencil
(issue #2), so those labels are provisional until the residual-based screen
replaces it.

Five stages deserve a note on what they do *not* establish.

**S9** marches the converged state forward. Persistence alone is weak evidence:
a stationary solution is an exact fixed point of the implicit time
discretisation as well as of the steady equations, so marching from it stays on
it to round-off whether it is an attractor or a saddle. The load-bearing half of
the stage therefore displaces the state and marches it back. Its horizon is set
in film renewals `L_b/U_b`, not in time units, because that quantity spans a
factor of 6000 across the sweep.

**S10** compares routes, not accuracy. Agreement between a fresh solve and a
continued one says the curve is not carrying its own history; it says nothing
about whether either is right. The bubble-length comparison in the same stage
is the one that carries physics: the residual 1.3e-3 at `Ca_b = 0.011` survives
mesh refinement and is a real finite-bubble effect, which is why it has its own
looser tolerance rather than being tuned away.

**S11** reports a bracket rather than a value, deliberately. Near onset the
indentation is a few parts in `1e5` of the tube radius, small enough to be a
property of the mesh; two film resolutions are what separate the two, and the
stage's only pass condition is that they agree where the indentation is large
enough that they should.

**S15** is a Reynolds bound, not a Laplace-number grid. It fails if the
top-of-sweep inertial row is missing or if the Stokes pad-doubled windows
there disagree in film thickness by more than two percent. An inertial
solve that dies on the doubled window is recorded, not fatal.

**S19** is the named La grid that S15 deliberately was not. It reuses
S15 cells, fills La=0.1 and La=10 at the thin-film and mid stations, and
records pad-doubled windows at La>0. It does not build h(Ca_b, La).

## Linear stability: the pencil, the reduction, and the verdict

The stability stages (`s12`, `s13`, `s14`, `s18`) share `eigen_helpers.py`.
The convention throughout is pyoomph's: with `matJ = -dR/dU` and
`matM = dR/d(dU/dt)` as `GenericEigenSolver.get_J_M_n_and_type()` returns them,
the pencil is `matJ v = lambda matM v` and a perturbation evolves as
`exp(lambda t)`, so `Re(lambda) > 0` is growth.

**The mass matrix is nearly all zero.** For the void-bubble problem `M` carries
entries only on the interface kinematic-condition rows, against the interface
mesh-position columns: 923 rows of 412815 unknowns at `Ca_in = 14` on the lip-8
station. Its row and column supports are *disjoint*. Two consequences run
through everything below.

**`finite_spectrum` computes the whole finite spectrum exactly.** Because
`rank(M) = nR << n`, the pencil has at most `nR` finite eigenvalues and
everything else is the M-nullspace at infinity. With `E` the `n x nR` injection
of the mass rows and `M_R = M[rows, :]`, the substitution `w = M_R v` turns the
pencil into `w = lambda G w` with `G = M_R J^{-1} E`, a dense `nR x nR` matrix,
so the finite spectrum is `1/eig(G)` and the eigenvectors are `v = J^{-1} E w`.
There is no shift and no Krylov space, and the M-nullspace -- which is what
floods a shift-invert subspace near zero -- is projected out by construction
rather than filtered afterwards.

`J^{-1}` is applied by `BorderedLU`. SuperLU on the whole `J` dies in memory
expansion because a few rows and columns are dense (the pressure-datum
Lagrange multiplier has `O(n)` entries, and `globals/U` and `globals/P` couple
to everything); those go into a small dense border, and the sparse core
factorises in seconds after row/column equilibration. Deleting them leaves the
core with a one-dimensional pressure nullspace, which is regularised by one
diagonal bump carried as an extra border unknown (Woodbury), and iterative
refinement recovers the decades the ill-conditioned border Schur complement
costs. For `azimuthal_k >= 1` the three global rows decouple and the border is
empty, the pencil is complex, and it is slightly larger than `problem.ndof()`
(415193 against 412815 at `Ca_in = 14`) because pyoomph releases the strong
axis conditions and imposes them by matrix manipulation instead. The extra
unknowns are those released axis dofs.

**The verdict is a residual, not a dof-support pattern.** `is_physical(J, M,
lam, v)` accepts an eigenpair when `M` acts on it at all and the scale-free
residual `||J v - lambda M v|| / ||lambda M v||` is below tolerance. That is
what `rightmost_sample` records and what `s14`/`s18` label on.

**Earlier "spurious" verdicts based on dof support are void.** The screen that
`eigen_helpers` used before rejected any mode whose amplitude sat on pressure
or global dofs. Since `M` touches only the kinematic rows and the interface
positions, every genuine eigenvector of this pencil is pressure-dominated --
its interface components are about `1e-6` of its pressure amplitude -- so that
screen rejected the physics. Any receipt carrying
`label_reason = "eigenvector_on_pressure_or_globals"`, and the Ca_in 20.11
leading eigenvalue `+48.8` recorded as spurious, must be re-read: those are not
evidence of a spurious mode. `eigenvector_support` remains available as a
diagnostic, now reporting the released-axis tail instead of erroring on
`m != 0` vectors, but it is no longer a verdict anywhere.

The measured check: at `Ca_in = 14` (lip-8 oa station) the reduced pencil gives
`+12.1990` and `+0.4771` as the only two `k = 0` eigenvalues with positive real
part, at scale-free residual `~5e-9`, and `+1.0525` as the rightmost `k = 1`
eigenvalue; seeding the steady state with `+-eps v` and marching reproduces
`12.199` after BDF2 inversion.

**Shift-invert controls need a start vector in the range of the operator.**
`v0 = M r` is annihilated by `OP = (J - sigma M)^{-1} M` -- the row and column
supports of `M` are disjoint, so `M (M r) = 0` identically -- and ARPACK
answers "error -9: starting vector is zero". `shift_invert_start_vector` draws
`r` on `M`'s column support and returns `OP r` instead.
