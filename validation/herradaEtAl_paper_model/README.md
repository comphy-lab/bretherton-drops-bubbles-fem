# Herrada, Yu & Stone finite-gas model

`cases/h15-bo1p56.json` selects the two-fluid Stokes equations at `Re=0`,
`Bo=1.56`, `Vhat=3.3` and `H/R=15`, with gas/liquid density and viscosity
ratios `0.000949` and `0.000018`. It uses the geometry, material properties
and pressure conversion of Herrada, Yu & Stone, *J. Fluid Mech.* **958**, A45
(2023), doi:10.1017/jfm.2023.123, equations (2.1) and Appendix B.

Static pressure is the solved pressure field. Gravity acts in both phases;
velocity and mesh positions are connected across the interface, where surface
tension balances the two stresses. The gas volume is constrained, and the
moving frame is pinned to the upper bubble tip. The top boundary has prescribed
Poiseuille flow in that frame. The bottom has static pressure zero and free
axial velocity, with radial velocity set to zero. The article states the bottom
pressure but does not specify that velocity completion, so it is an explicit
model assumption rather than an attributed author boundary condition.

The seeded case approaches the selected Bond number through bounded stationary
steps after a transient at the declared initial Bond number. Its completion is
one stationary state with the declared consistency checks. This case alone
does not establish a branch curve, threshold or code-to-code agreement with
the paper. Those comparisons require the reference data, matching observable
and numerical uncertainty appropriate to each figure.

Validate without native assembly:

```sh
python runners/run_paper_case.py validation/herradaEtAl_paper_model/cases/h15-bo1p56.json --check
```

Run into a fresh directory using the pinned project interpreter:

```sh
python runners/run_paper_case.py validation/herradaEtAl_paper_model/cases/h15-bo1p56.json --out /path/to/new/run
```

The mixed-port and two-phase verification cases are in
`verification/paper_model/`. The case validator requires explicit units,
pressure convention, material ratios, mesh, frame/volume constraints, solve
limits and completion checks. The runner records their provenance before
execution, and generated fields, state and check results belong in the run.
