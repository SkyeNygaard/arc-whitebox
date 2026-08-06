# K32 Phase Identifiability and Non-Marginal Basis-Flux Experiment

**Date:** 2026-07-29  
**Terminal state:** **FAIL — scoped closure of tested absolute-phase sources**  
**Production decision:** Preserve the unchanged partial-tree complete-Kerdock estimator.

## Executive conclusion

The experiment found a genuine but insufficient new information source.

Across 112 previously generated base networks and 272 grouped rotations, changes in K32 phase between rotations were highly predictable from same-cloud final-output observables, while the absolute base-network offset remained weakly identifiable. A complete-basis, adjoint-weighted covariance statistic then achieved positive phase correlation on a new 50-network production-rotation terminal cohort, but it was raw-neutral, score-negative after K32 overhead, and tail-unsafe.

The project is therefore not missing another ordinary model architecture. It is missing the rotation-averaged absolute signed state.

## Phase-identifiability audit

- Examples: **272** rotations from **112** base networks.
- Bases whose oracle template scale changed sign across tested rotations: **39.3%**.
- Median within-base scale range: **0.000485718**.
- Weight-global absolute-scale prediction on the primary terminal: Pearson **0.320**, sign accuracy **64.6%**.
- Same-cloud final-output rotation-difference prediction: Pearson **0.839**, pairwise sign accuracy **95.8%**.
- All-observable rotation-difference prediction: Pearson **0.859**, sign accuracy **93.8%**.

This separates the target into a learnable rotation-dependent deviation and an unresolved rotation-averaged offset.

## Hierarchical learned decomposition

The frozen hierarchical estimator used:

1. pairwise-difference ridge over the same-cloud sample output, baseline output and their difference;
2. a weight-global ridge for the base-network absolute offset;
3. a training-only grouped calibration.

Validation candidate/base was **1.009785**, with worst **2.447**. It failed both exposed terminal diagnostics (**1.145** and **1.061**).

A mechanistic offset rescue using 3,680 deterministic weight-spectrum and forward/backward path-contraction features failed more strongly: validation **1.302**, exposed terminals **4.116** and **1.405**. This closes tested weight-only absolute-offset estimation.

## Complete-basis non-marginal features

For every Kerdock basis, the implementation retained complete-block means of:

- the 32 radial K32 features;
- target-layer activations;
- squared target activations;
- final outputs.

It then formed output-projection moments, target/output cross-moments, feature-output singular spectra and complete-block covariance spectra. No partial-basis allocation or extra trajectory was used.

Broad 355-feature regressions were harmful. The best coherent feature family, the feature-output cross spectrum, had validation ratio **1.129**, so the branch was reduced to one prederived scalar.

## Frozen adjoint basis covariance

The final statistic was

```text
z_b = (X_b - q) · template
w_b = (Y_b - Y_bar) · normalize(template · beta)
S   = mean_b[(z_b - z_bar)(w_b - w_bar)]
```

The affine mapping and replay shrink were frozen on the original 64-base training block. Development validation reached **0.963676**, but its interval **[0.838, 1.082]** and worst **1.313** did not clear promotion.

### Independent 50-network terminal

| Metric | Result |
|---|---:|
| Raw candidate/base | **0.991200** |
| Raw 95% bootstrap | **[0.922885, 1.078294]** |
| Adjusted candidate/base | **1.003895** |
| Adjusted 95% bootstrap | **[0.934705, 1.092104]** |
| Wins | **24/50** |
| Median ratio | **1.022365** |
| Worst ratio | **2.410772** |
| Phase Pearson | **0.436233** |
| Phase sign accuracy | **54.0%** |
| Exact K32 oracle ratio | **0.200966** |

The signal is real—phase Pearson is positive—but far below the precision required for cancellation-safe deployment. The raw point estimate is nearly neutral, K32 computation makes it score-negative, and the tail is unacceptable.

## Direct covariance-vector formulation

A separate algebraic candidate applied the complete-basis covariance directly as an output correction vector rather than predicting a scalar. Its validation ratio was **0.979022**, interval **[0.760, 1.180]**, worst **2.148**, and mean direction cosine **0.118**. It did not justify another terminal cohort.

## Scoped closure

Closed now:

- same-input hierarchical rotation-deviation plus weight-global offset estimation;
- richer deterministic weight-invariant absolute-offset regression;
- broad complete-basis feature regression;
- normalized and amplitude-preserving adjoint basis-flux scalar estimators;
- direct complete-basis covariance-vector correction;
- further thresholds, shrinkage, feature subsets, ridge strengths or neighboring basis-moment variants in these families.

Preserved:

- the exact K32 lower-order direct-output oracle mechanism;
- K128 as a robustness ceiling;
- the finding that rotation-dependent phase differences are observable;
- complete-basis reduction infrastructure for testing a future exact identity.

A future branch should start only from a new identity that supplies the **rotation-averaged absolute signed offset**. Another model over weights, same-cloud summaries or basis-moment features is not justified by this evidence.
