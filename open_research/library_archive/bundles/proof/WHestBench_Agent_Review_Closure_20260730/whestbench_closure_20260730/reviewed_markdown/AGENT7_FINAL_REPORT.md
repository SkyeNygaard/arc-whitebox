# Agent 7 — scalar-learning reproduction and skeptical audit

**Date:** 2026-07-30 ET  
**Role:** Agents 7A (positive reproduction) and 7B (skeptical audit), kept separate until the numerical reproduction was frozen.  
**Final status:** **M152 NOT REPRODUCIBLE; scoped Path-2 substitute independently reproduced; no feature-based scalar-learning value established.**

## Executive decision

The claimed M152 experiment cannot be reproduced or cited. The Library contains no `b6_scalar_predictability.py`, 1,100-network row table, scalar-target equation, eleven-feature definition, base-network/rotation manifest, preprocessing, regularization chronology, row predictions, or hashes. The only surviving M152 object is a ledger/transcript summary.

I therefore ran the strongest legitimate substitute available: the archived Path-2 exact K32-scale experiment. It has 32 development rows from 24 independent base networks, 1,528 shared legal numerical features, and a previously frozen 36-row holdout from 12 new base networks with rotations grouped. The compact archive’s SHA-256 manifest verifies, and its frozen ridge predictions and final ratios reproduce to maximum absolute error below `3.2e-15`.

That reproduction does **not** establish scalar predictability:

1. The best archived bounded model scored holdout pooled candidate/base `0.482151`, but fixed `alpha=0.75` scored `0.476160` with the same worst case `1.189742`. All other bounded learned models were also worse than fixed `0.75`.
2. More directly, every learned model was worse than a constant equal to its own mean prediction. For the bounded ensemble, feature-dependent variation added `+0.009186` pooled ratio; grouped-bootstrap 95% interval `[+0.004108, +0.013573]`.
3. The strongest weight-only model’s feature variation added `+0.004503`; interval `[-0.002502, +0.016557]`. There is no evidence that its network-specific deviations help.
4. The apparent improvement over the safe fixed `0.50` baseline comes from shifting the average scale upward toward `0.72–0.74`, not from correctly ranking networks.
5. The development panel itself selected `alpha=0.50` as the best constant satisfying hard-panel worst ratio `<=1.05`. The fresh holdout happened to favor a much more aggressive constant (`~0.81` oracle diagnostic), explaining why models clipped at `0.75` looked good in mean while retaining a tail.

The skeptical tests agree:

- With 24 independent development groups and 1,471 nonconstant features, the largest grouped absolute correlation was `0.610665`, but the familywise group-permutation p-value was `0.5456`; null median maximum correlation was `0.616751`.
- A fixed compact-feature ridge had grouped six-fold target `R²=-0.9297`; permutation p-value `0.6692` over 2,000 group-label permutations.
- All tested group-level models had negative leave-one-group-out `R²` versus a cross-fitted constant: ridge `-1.10`, extra trees `-0.30`, RBF kernel `-2.32`, and k-nearest neighbors `-0.20`.
- Row-wise leave-one-out is materially optimistic when rotations are not grouped. The weight model moves from `R²=+0.1226` row-wise to `R²=-0.1189` leave-one-base-network-out.

## What this says about M152

M152’s stated width-256 oracle scalar gain is `1.483×`, so only `32.57%` of baseline MSE is scalar-correctable. Reaching candidate/base `0.98` requires recovering at least `6.14%` of that scalar ceiling. A single feature with `|r|=0.13` has `R²=1.69%`, corresponding under ideal quadratic geometry to only about `0.99450` candidate/base. Multiple features could combine, so this is not an impossibility proof.

If M152 truly contains 1,100 independent base networks and clean labels, an 11-feature global test has essentially full power for `R²≈0.0614`; a genuinely negative grouped cross-validation result would be strong evidence against that particular linear feature set. But the number of independent base networks is unknown, rotations may reduce effective sample size, and no rows exist to audit the claim.

## Target-noise caveat

The Path-2 labels are exact quadratics relative to archived high-budget reference estimates, not exact expectations. On the 36-row holdout, reference truth-noise MSE was a median `33.36%` of baseline MSE and reached `140.29%`; the median absolute raw-versus-unbiased K32-ratio difference was `0.07594`. The original four-stream design is appropriate, but the compact bundle omits the underlying vectors, so I could reconstruct and verify every quadratic and model result but could not independently recompute the reference streams from first principles.

## Claim classifications

| Claim | Status | Verdict |
|---|---|---|
| M152 used a 1,100-network exact-label corpus | OPEN / UNVERIFIED | Corpus and manifest absent. |
| Eleven observables had `|corr|<=0.13` | EXPLORATORY CLAIM ONLY | No feature list or rows. |
| Ridge/LOOCV had negative `R²` | EXPLORATORY CLAIM ONLY | Grouping and chronology unknown. |
| M152 candidate/base was `~0.94–0.98` | EXPLORATORY CLAIM ONLY | No row predictions or complete metric reconstruction. |
| Path-2 frozen ridge holdout results | FROZEN EMPIRICAL, INDEPENDENTLY REPRODUCED | Reproduced to machine precision from immutable compact artifacts. |
| Path-2 learned features predict useful network-specific scale | INVALIDATED FOR TESTED MODELS | Same-mean constants match or beat all models; learned deviations do not help. |
| No scalar predictor can exist | OPEN / NOT PROVED | Small effective sample and missing M152 data prevent universal closure. |

## Reopening conditions

Do not reopen scalar learning merely with another regression table. Require all of:

1. immutable raw rows with base-network and rotation IDs;
2. the exact scalar target equation and sign convention;
3. independent target/reference streams or a label-reliability estimate;
4. an explicit legal-runtime feature list and extraction script;
5. grouped train/calibration/test splits frozen before labels are inspected;
6. comparisons against the development-optimal constant, safe constant frontier, and constant equal to the model’s mean prediction;
7. row-level predictions, complete final-MSE ratios, wins, p90, worst, grouped intervals, and hashes;
8. no Mini-100 use for feature or hyperparameter selection.

## Final recommendation

Keep M152 provisional and out of the paper’s evidence table. Add the present audit as a new frozen empirical result. It supports the narrow statement:

> In the archived Path-2 information class, grouped linear and small nonlinear models do not demonstrate network-specific scalar predictability; apparent frozen-holdout gains are explained by a more aggressive constant scale, while unrestricted prediction is tail-unsafe.

It does **not** support “scalar learning is impossible” or “no statistical path exists.”
