# WHestBench agent-review closure report

**Date:** 2026-07-30  
**Coordinator role:** Agent 4 continuation / loose-end closer  
**Overall disposition:** **The mathematical core is now substantially closed and paper-ready under explicit scope. The broad adaptive-impossibility narrative remains invalid.**

## Executive result

I examined the available Agent 1–8 review Markdown, the Agent 10 paper-architecture report, the Agent 11 hostile-referee report, the original Agent 4 proof package, and the cited proof-release materials.

This pass closes two concrete issues that the reviewers identified as fatal:

1. **T16 is upgraded from reduced-cost negativity to full auxiliary-LP optimality.**
   The exact degree-five Hermite interpolant at the three algebraic dual contacts is certified feasible, has strictly positive nonconstant Gegenbauer coefficients, attains the exact dual value, and is the unique optimizer. The proof combines an exact dual moment construction, a directed certificate for `K_32^(6)>0`, Hermite remainder, Krawczyk coefficient enclosure, and the prior all-degree reduced-cost theorem.

2. **The T22 release is canonicalized as v5.1.**
   Its manifest now tracks 59 files, including the 32 canonical outputs and all 23 deterministic curvature chunks. The release language distinguishes fixed verification hashes from external authentication, pins `mpmath==1.3.0`, adds environment/provenance notes and a multi-platform CI workflow, and passes both the fast theorem verifier and the expanded manifest verifier.

The pass also resolves evidence governance:

- the stale v4 theorem JSON that asserted a positive lower bound on Kerdock suboptimality is formally quarantined by exact SHA-256;
- M146 is classified as provisional arithmetic consistency only, not an evidence-bearing `41.2x`/`~5e-4` claim;
- M152 is removed from the evidentiary claim register because no corpus, target, feature definition, grouped split, predictions, script, or hashes survive;
- the reproducible Path-2 scalar audit is retained only as a narrow negative result;
- global signed cubature, finite-width transfer, adaptive/network-dependent methods, nonlinear estimators, and public accounting remain explicitly open or unaudited.

## What is now closed

### T22 — scoped static nonnegative cubature theorem

**Status:** COMPUTER-ASSISTED CERTIFIED.

The theorem is one-sided: complete Kerdock is within the certified percentage of the infimum in the stated class. It does not establish strict suboptimality. Coverage is limited to the dimension-256, depth-32 infinite-width normalized ReLU kernel; at most 66,048 spherical nodes; nonnegative mass-one weights; and deterministic or admissible randomized rules independent of the realized field/network.

The v5.1 release verifier reports:

- 1,421 certified pointwise subintervals;
- global minorant margin below zero;
- directed kernel-mean enclosure;
- one-sided ratio logic verified;
- 59 manifest entries verified.

### T16 — all-degree auxiliary LP

**Status:** COMPUTER-ASSISTED CERTIFIED; full primal-dual optimum now established.

The exact contact cubic is

`22102 t^3 + 21930 t^2 - 87 t - 85`.

The new proof establishes:

- exact positive three-node dual quadrature through degree five;
- strict negativity of every unused reduced cost for all degrees at least six;
- strict positivity of `K_32^(6)` on `(-1,1)`;
- pointwise feasibility of the degree-five Hermite interpolant by the Hermite remainder formula;
- strict positivity of all five nonconstant Gegenbauer coefficients;
- exact primal-dual equality;
- uniqueness of the optimizer among finite admissible expansions, and among absolutely convergent admissible nonnegative expansions under the stated convergence conditions.

The tightened certified auxiliary optimum yields a Kerdock relative-excess upper bound of

`0.023324172950039%`,

improving the older safe T22 bound `0.02336550102948%`.

An independent C++17/Boost exact-integer implementation reproduces the finite reduced-cost sweep and tail cutoff. It is a genuinely different implementation stack, although it audits the same exact recurrence.

### T27 — fixed Kerdock-line universe

**Status:** PROVED UNDER AN EXPLICIT MODEL.

The Agent 3 theorem remains accepted without mathematical modification: among static linear rules on at most `P` symmetrized lines from the fixed 33,024-line Kerdock universe, arbitrary real weights are optimized by complete orthonormal bases and at most one partial basis, with positive equal within-basis weights and positive analytic basis masses.

It is not a theorem for arbitrary nodes, unpaired points, finite width, nonlinear estimators, or network-dependent support/weights.

### Signed negative-mass stability

**Status:** PROVED, with a COMPUTER-ASSISTED CERTIFIED constant.

The strengthened support-count lower bound and exact residual supremum remain valid. They are deliberately classified as a stability lemma rather than global signed-cubature closure because the numerical exclusion permits substantial relative improvement at extremely small negative mass.

### Layer-31 correction theory

**Status:** abstract results retained; universal scalar threshold rejected.

Retained:

- correction-risk identity;
- selector value theorem;
- general replacement formula;
- correlated-noise shrinkage;
- common-bias non-identifiability under its observation model;
- ReLU crossing/nonlinear-margin bounds.

The invariant criterion is downstream-weighted, for example `E||J xi||^2 < E||J d||^2`; no universal threshold in unweighted layer-31 relative mean error exists.

### Harmonic controls

**Status:** class-specific exact theorems and frozen negative experiment retained.

Retained:

- radialized polynomial annihilation through degree five;
- polynomial Stein fields of component degree at most four;
- bias-free one-hidden-layer ReLU Stein blockwise annihilation under its exact assumptions;
- frozen failure of the selected small degree-6+8 dictionary;
- the symmetrized Poisson-kernel counterexample showing that analytically integrable does not imply low harmonic degree.

## Evidence that is closed by removal or downgrade

### M146

No original cohort, perturbation directions, seeds, reference streams, replay outputs, or manifest were located. Its four headline points are arithmetically consistent with a quadratic curve, but this is not an empirical reproduction. The `41.2x` and `~5e-4` values must not appear as verified central evidence.

### M152

No primary experiment package was located. The 1,100-network claim, eleven features, correlations, grouped `R^2`, and candidate ratios are removed from the evidence register. The Path-2 substitute is a different experiment and cannot rehabilitate M152.

### Broad impossibility thesis

The sentence “no statistical path exists,” and close variants, are invalidated as theorem claims. The defensible operational conclusion is:

> No active branch in the tested information classes clears a credible continuation gate under the current evidence, deadline, and resource constraints.

## Genuinely remaining open questions

1. Arbitrary signed-node cubature outside the fixed Kerdock-line universe.
2. Exact finite-width width-256 optimality or a rigorous finite-width transfer theorem.
3. Network-adaptive support or weights.
4. Nonlinear and analytic-plus-residual estimators.
5. New absolute-phase observables outside the common-bias observation model.
6. Rich high-degree, biased, deep, or nonhomogeneous analytic controls.
7. Public score/high-row/FLOP/wall-time accounting until a separate evidence auditor traces it end to end.
8. Independent human mathematical and reproducibility sign-off.

## Publication recommendation

The theorem paper is now viable around T22, completed T16, T27, and the signed stability lemma. The finite-width correction work should be framed as a separate scoped falsification map, or as a sharply separated empirical half of an integrated paper. The title, abstract, theorem table, figures, and conclusion must state exclusions in the same place as the headline theorem.
