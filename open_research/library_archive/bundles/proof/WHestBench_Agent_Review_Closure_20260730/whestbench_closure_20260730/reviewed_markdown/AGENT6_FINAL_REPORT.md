# Agent 6 final report — layer-31 anchor theory and adversarial precision audit

**Date:** 2026-07-30  
**Disposition:** 6A verified after corrections; 6B reproduction blocked by missing M146 artifacts; universal scalar threshold rejected.

## Executive findings

1. The correction-risk, common-bias, and ReLU-crossing results are correct under explicit assumptions and belong in the paper.
2. The archived theorem package needs constrained-selector, general-replacement, correlated-shrinkage, and nonlinear-margin refinements.
3. The invariant full-replacement criterion is downstream weighted:

   `E||J xi||² < E||J d||²`.

   An unweighted “relative layer-31 mean error” threshold is not universal.
4. M146 is not independently reproducible from the shared archive. Canonical v15 itself marks the artifact, exact IDs, and perturbation manifest pending.
5. The four reported M146 gains are exceptionally consistent with a single quadratic perturbation model (`R²=0.9999965`), yielding a fitted break-even of `5.80e-4`. This is an arithmetic consistency result only.
6. In a reproducible synthetic adversarial audit, the median equal-norm break-even moved from `1.87e-4` in a leading downstream singular direction to `1.41e-2` in a trailing direction. This demonstrates direction dependence without claiming those numbers describe ARC networks.
7. Exact synthetic ReLU replay showed that a kink-focused `5e-4` shift can have a nonlinear remainder around 25% of the linear shift, while generic-margin shifts were nearly perfectly linear.
8. Existing legal T4 evidence is consistent with the absolute-phase diagnosis: the frozen policy was harmful, signed transfer was near zero/negative, and per-rotation oracle signal did not survive grouping.

## Evidence status

| Item | Status |
|---|---|
| Correction-risk identity | PROVED |
| Constrained selector theorem | PROVED in this review |
| Full replacement threshold | PROVED UNDER EXPLICIT SUBSPACE MODEL |
| General replacement cross-term formula | PROVED in this review |
| Correlated-noise shrinkage | PROVED in this review |
| Common-bias non-identifiability | PROVED UNDER EXPLICIT OBSERVATION MODEL |
| ReLU crossing and nonlinear margin | PROVED / bounded |
| M146 60-network values | EXPLORATORY EMPIRICAL; artifact missing |
| Quadratic consistency of M146 headlines | REPRODUCIBLE ARITHMETIC CHECK |
| Direction independence of `~5e-4` | FALSE IN GENERAL |
| Universal impossibility of adaptive anchors | NOT PROVED |

## Internal consistency of the M146 headline curve

Let reported gain be baseline MSE divided by candidate MSE. The exact-anchor point gives

`q = 1/41.2 = 0.02427184466`.

Fit the remaining candidate/base ratios to

`q + a epsilon²`.

The fitted coefficient is `a=2.9019646e6`; pointwise estimates differ by about 1%; `R²=0.99999647`; and the risk crosses one at `epsilon=5.798536e-4`. The consistency is too strong to be accidental arithmetic noise, but it cannot establish how perturbations were generated or whether the underlying cohort was valid.

## Direction and gate geometry

The theorem itself explains the problem. For a unit direction `v`, Euclidean break-even is

`epsilon*(v) = ||J d|| / ||J v||`.

It varies with direction unless `JᵀJ` is proportional to the identity on the tested span. The actual-defect direction has break-even exactly equal to `||d||`; a leading singular direction can require much less error, while a trailing direction can tolerate much more.

The final ReLU adds a second source of direction dependence: only particles with `|h|<=|W delta|` contribute nonlinear remainder. A direction concentrated on high kink mass can invalidate an otherwise favorable linear threshold.

## Relationship to legal empirical evidence

The T4 closure report found:

- frozen policy raw ratio `1.127854`, `17/48` wins, worst `2.480711`;
- mean error-correction inner product `-2.535e-08`;
- mean correction cosine `0.0969`;
- per-rotation positive oracle `0.915133`, but one coefficient vector per base network across rotations `1.019612`.

This is exactly the empirical pattern predicted when useful correction directions exist but available observables fail to recover stable absolute signed phase. It does not prove that every future observable must fail.

## Required M146 reproduction

Restore the original package and freeze:

- IDs, rotations, exact means, reference streams, and metric;
- perturbation normalization and seeds;
- the actual residual direction;
- isotropic directions;
- every legal analytic/companion residual direction;
- leading downstream singular directions;
- sparse target-free supports;
- kink-concentrated directions.

For each, report unweighted error, `eta_J`, exact nonlinear final MSE, linearized MSE, correction cosine, crossing fraction, remainder, wins, p90, worst, and grouped uncertainty. No protected cohort should be opened until the package and matrix are frozen.

## Final recommendation

Use the corrected appendix. Downgrade M146 to provisional. Replace the scalar reopening gate by a downstream-weighted exact-replay gate. The current tested anchor families remain closed, but a genuinely independent absolute-phase observable remains mathematically open.
