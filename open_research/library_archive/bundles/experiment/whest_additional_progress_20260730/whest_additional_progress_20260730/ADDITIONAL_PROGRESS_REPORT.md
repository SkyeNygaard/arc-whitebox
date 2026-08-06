# WHestBench additional closure and theorem strengthening

**Date:** 2026-07-30  
**Disposition:** Several previously deferred gates are now technically closed. The remaining blockers are genuinely external or mathematically open.

## Newly closed

### 1. Complete T22 clean regeneration

The full canonical v5.1 proof was rebuilt after deleting generated outputs. All 1,421 certified leaves and downstream theorem artifacts passed the 59-file manifest. This upgrades the local status from shipped-artifact verification to complete clean regeneration.

### 2. Independent T16 primal interval stack

A new `mpmath.iv` implementation independently reproduces the canonical Decimal/libmpdec numerical proof ingredients:

- Krawczyk contraction norm below `1.60e-72`;
- all five nonconstant Hermite/Gegenbauer coefficients positive;
- `F''/F' <= 2.398586389549084... < 3`;
- transformed sixth-derivative margin `8.149286225739272... > 0`;
- byte-stable output across repeated runs.

This closes the internal second-stack numerical objection. It does not replace named human review of the analytic Bell-polynomial and Hermite-remainder argument.

### 3. Strong global arbitrary-signed-node floor

The prior single-rank theorem retained only `6.99%` of Kerdock MSE. A new exact-feasible multi-rank combination raises the global floor to

`7.90161513053615965080819e-8`,

or at least `32.4680274552%` of complete-Kerdock MSE. Every static mass-one rule using at most 66,048 arbitrary nodes and arbitrary real weights therefore has improvement factor below `3.079954x` for the limiting kernel.

The LP is discovery-only. The published combination is rounded downward and verified with exact rational coefficient constraints, so solver optimality is not part of the theorem.

### 4. Cross-layer oracle coherence

The new OGAP campaign supplies the missing row-level checkpoint diagnostic on frozen and independent-confirmation cohorts. Checkpoint repair is substantial by layer 15 and extreme by layers 29–30. Successive repair increments have energy fractions approximately

`[0.395, 0.177, 0.235, 0.111, 0.0535, 0.0288]`,

with most off-diagonal cosines below `0.10` and maximum `0.146`. The error channel is distributed and approximately incoherent; it is neither one purely late injection nor equal layer contributions.

### 5. Quantitative symmetry-defect theorem

T41 is now written as a complete theorem. Under an explicit measure-preserving phase action, normalized correction alignment is bounded by

`((delta_error + delta_policy)/2)^2`.

An explicit randomized-orientation model gives exact zero value for orientation-blind policies and identifies orientation-aware odd features as the escape class.

### 6. Canonical v5.2 record

`FORMAL_CANONICAL_THEOREM_RECORD_V5_2.json` combines:

- tightened one-sided static bound `0.023324172950039%`;
- full T16 auxiliary optimum;
- two-stack T16 primal numerics;
- complete T22 regeneration;
- finite-width fixed-support theorem;
- arbitrary-signed global floor;
- exact scope exclusions.

`validate_v5_2.py` passes.

## Remaining external release requirements

1. Named human mathematical review of T16 and finite-width T27.
2. Named reproducibility reviewer operating from a public clean checkout.
3. External publication of the final archive digest.
4. Public multi-platform CI execution for the final frozen archive.
5. Final related-work and authorship review by the named human authors.

## Remaining mathematical frontiers

- arbitrary-node finite-width analogue of T22;
- arbitrary-signed near-optimality, rather than a factor-3.08 floor;
- a formal symmetry action for an actual deterministic legal WHestBench transcript;
- a constructive, stable legal phase observable;
- unrestricted nonlinear/network-adaptive estimation;
- residual-kernel recertification beyond tractable equivariant-linear surrogates.

## Current paper thesis

> Static arbitrary-node nonnegative cubature is certified near-optimal for the limiting kernel; its all-degree auxiliary certificate is uniquely optimal; fixed Kerdock-line allocation is exactly solved even at finite width under explicit assumptions; arbitrary signed static rules obey a global factor-3.08 floor; and the finite-width experiments localize the remaining practical difficulty to stable, legal, downstream-signed coefficient information rather than absence of corrective directions.
