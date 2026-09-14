# Verification

Software-contract tests live in `tests/`. This folder is for independently
checkable numerical tests of the discrete equations.

Intended first checks, once a pyoomph environment exists:

1. Capsule volume of the meshed liquid-plus-void reconstruction equals
   `4/3 π R0³` at the seeded geometry.
2. A static spherical-cap limit recovers the Laplace pressure jump
   `Δp = σ (1/R1 + 1/R2)` with void gauge pressure zero.
3. An open-boundary ALE volume balance comparing `dV/dt` with boundary flux
   and mesh motion. That is a balance, not a conservation claim.

None of these checks has been executed in this repository yet.
