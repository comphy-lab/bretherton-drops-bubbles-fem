# Verification of the mixed ports and two-phase interface

`cases/bare-tube.json` selects a straight liquid-filled tube with an upper
Poiseuille velocity and lower static pressure zero. The lower radial velocity
is zero, and its axial velocity is an output. The independent reference is

```text
u_r = 0
u_z = -U + 2 Ca_l (1 - r^2/R_t^2)
p_static = -(Bo + 8 Ca_l) z/R_t^2
```

The case measures volume-weighted velocity and pressure errors, lower pressure
and the flux balance. It tests the stated boundary completion, which is
different from prescribing zero total traction.

`cases/static-laplace.json` selects an unconfined spherical gas inclusion at
zero gravity and imposed flow, with the finite density and viscosity ratios.
Its references are zero velocity, radius `R0=1`, volume `4*pi/3` and pressure
jump `2`. It tests the interface stress, surface tension and gas-volume
constraint. It does not test a translating bubble or gravitational buoyancy.

Validate either case without native assembly:

```sh
python runners/run_paper_case.py verification/paper_model/cases/bare-tube.json --check
python runners/run_paper_case.py verification/paper_model/cases/static-laplace.json --check
```

Run one case into fresh output custody with the pinned project interpreter:

```sh
python runners/run_paper_case.py verification/paper_model/cases/bare-tube.json --out /path/to/new/run
```

The runner writes its provenance manifest before progress and native execution.
The summary records each check, and a failed check returns a nonzero exit.
Use the declared tolerances and compare refinement only to resolve a failed or
under-resolved observable. A successful import or case check is a software
check, not verification of these equations.
