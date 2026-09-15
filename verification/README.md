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
| `s12_azimuthal.py` | Is azimuthal linear stability available for this formulation, and how many eigenvalues does it need? | `s12_azimuthal.json` |

Stage numbering has gaps because S2 to S4 were folded into the production path
during development rather than kept as separate scripts.

Three stages deserve a note on what they do *not* establish.

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

**S12** is a capability and calibration check, not a physics result. It does not
locate a critical capillary number; it establishes that the azimuthal
eigenproblem runs on a moving mesh with a free surface, records the pressure-datum
requirement that makes it possible, and measures how many eigenvalues must be
requested before the rightmost one is a property of the operator rather than of
the call.
