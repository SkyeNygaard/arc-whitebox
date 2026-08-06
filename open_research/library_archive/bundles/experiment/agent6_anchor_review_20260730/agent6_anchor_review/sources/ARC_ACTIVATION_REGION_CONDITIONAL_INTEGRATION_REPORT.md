# Activation-region conditional integration: targeted layer-31 investigation

**Date:** 2026-07-29  
**Decision:** **STOP as a deployable submission branch; retain the sparse layer-31 oracle as a mechanistic clue.**

## Executive result

This investigation separated three questions that had previously been conflated:

1. Is the valuable late-layer correction concentrated in only a few neurons?
2. Can those neurons be selected from observable local geometry?
3. Can their mean corrections be estimated by low-dimensional activation-region integration?

The answer is **yes / weakly yes / no**.

- The full layer-31 mean intervention reproduced the prior oracle: **5.7771×** raw-MSE improvement on the frozen eight-network screen, bootstrap 95% interval **4.2854–7.3965**, with every network improved.
- A cross-fitted target-aware sparse oracle using only **12 layer-31 neurons** achieved **1.7844×** raw-MSE improvement and captured **55.23%** of the full layer-31 benefit. Its worst network still improved to **0.661×** baseline MSE.
- An observable selector based on Gaussian mean discrepancy times final-layer sensitivity had real selection signal when granted exact true corrections: at K=12 it achieved **1.1795×**, and at K=24 **1.3645×**.
- But the actual low-dimensional conditional integrator failed. The best direct model was **0.8967×** (worse), 95% interval **0.8582–0.9156**, with **0/8 wins**.
- Turning the same model into a residual control variate produced a small screen signal: **1.0174×**, 7/8 wins, interval **0.9985–1.0289**. The exact frozen setting was then run on the untouched 24-network validation split and reversed to **0.9889×**, interval **0.9448–1.0331**, only **8/24 wins**, and a worst-network MSE ratio of **1.195**.

The validation result is raw-error negative before charging any added arithmetic. The branch is therefore closed for submission use.

## Why this was genuinely different from the prior negative experiments

The prior one-direction breakpoint experiment integrated a single input-space line through the whole 32-layer network. The prior penultimate low-rank experiment conditioned on a generic rank-r representation of the layer-31 state. This investigation instead used the newly established layer-31 error channel and asked whether a tiny selected set of late neurons and their immediate upstream gates carried most of the useful correction.

That distinction mattered scientifically: the sparse oracle passed strongly even though the deployable integrator failed.

## Experiment 1: strict sparse layer-31 oracle screen

### Construction

For each frozen screen network:

1. Run the protected complete Kerdock rule and retain the layer-31 cloud and final preactivations.
2. Build two independent reference groups from eight Haar-rotated complete Kerdock rules each.
3. For each reference group, compute the exact coordinatewise layer-31 mean-matching translation.
4. Linearize only the true final ReLU around the baseline cloud to obtain a 256-dimensional final-output contribution vector for each corrected layer-31 coordinate.
5. Select coordinates using one reference group and score the true nonlinear replay against the other, then reverse the groups.

This cross-fitting prevents the sparse support from being an artifact of reference noise.

### Target-aware concentration

| Corrected layer-31 neurons | Raw-MSE ratio | 95% interval | Full-oracle benefit captured | Worst network candidate/baseline |
|---:|---:|---:|---:|---:|
| 1 | 1.0887× | 1.0573–1.1142 | 9.79% | 0.976 |
| 2 | 1.1622× | 1.1211–1.2047 | 18.07% | 0.938 |
| 4 | 1.2991× | 1.2219–1.3774 | 29.83% | 0.869 |
| 8 | 1.5612× | 1.4540–1.6759 | 46.15% | 0.737 |
| 12 | 1.7844× | 1.6240–1.9318 | 55.23% | 0.661 |
| 16 | 2.0597× | 1.8438–2.2487 | 63.99% | 0.587 |
| 24 | 2.4737× | 2.1275–2.7898 | 73.79% | 0.524 |
| 32 | 2.9565× | 2.5073–3.4414 | 81.39% | 0.435 |

The preregistered K≤12 concentration gate passed: 12 coordinates captured more than half of the full oracle effect and improved every network.

### Observable selection, still granted exact corrections

| Selector | K | Raw-MSE ratio | Full-oracle benefit captured | Worst network candidate/baseline |
|---|---:|---:|---:|---:|
| Gaussian-gap × sensitivity | 12 | 1.1795× | 16.39% | 0.937 |
| Gaussian-gap × sensitivity | 24 | 1.3645× | 31.45% | 0.881 |
| Gate entropy × sensitivity | 12 | 1.0390× | 3.99% | 0.995 |
| Gate entropy × sensitivity | 24 | 1.0632× | 7.16% | 0.961 |
| Final sensitivity only | 12 | 1.0668× | 6.55% | 1.004 |
| Final sensitivity only | 24 | 1.0891× | 10.12% | 0.958 |
| True-delta contribution norm (semi-oracle) | 12 | 1.4088× | 36.74% | 0.832 |

The Gaussian-gap selector is useful but not sufficient: it identifies coordinates worth correcting, but it does not estimate the signed correction itself.

## Experiment 2: direct low-dimensional activation-region integrator

For each candidate layer-31 neuron j:

1. Select m layer-30 preactivation gates by `|W30→31[k,j]| × std(h30_k)`.
2. Write the layer-31 preactivation as

   `z_j = Σ_{k∈S} w_k ReLU(h_k) + r`.

3. Fit a joint Gaussian model for the selected preactivations and residual r on the Kerdock cloud.
4. Condition r on the selected preactivations, integrate its scalar Gaussian ReLU analytically, and integrate the remaining m-dimensional selected-gate distribution with a fixed 1,024-point Sobol-normal rule.
5. Rank layer-31 corrections by predicted magnitude times true final-layer sensitivity, apply the top K with shrinkage α, and replay the true final layer.

Grid: `m ∈ {0,2,4,6,8}`, `K ∈ {8,12,16,24}`, `α ∈ {0.25,0.5,0.75,1}`.

### Result

The best point in the entire screen grid was the **zero-gate Gaussian model**, K=8, α=0.25:

- raw-MSE ratio: **0.896670×**;
- 95% interval: **0.858168–0.915616**;
- wins: **0/8**;
- worst candidate/baseline: **1.351**.

Increasing the explicit region dimension did not help. Across the screen, the predicted correction had essentially zero correlation with the true layer-31 Kerdock defect. The model approximated the marginal distribution, but not the signed cubature error that matters.

## Experiment 3: conditional integrator as a residual control variate

The direct estimator can fail even when its pointwise surrogate is useful. Therefore the same conditional model was converted to

`E[f] ≈ E[g] + Q_K(f - g)`.

Here `g(x)` is the pointwise conditional-Gaussian ReLU expectation given the selected gates. Its Gaussian expectation is computed by the low-dimensional Sobol integral; only the residual is averaged over the complete Kerdock rule.

### Development screen

The best setting was frozen without further tuning:

- explicit gates `m=2`;
- corrected layer-31 neurons `K=8`;
- shrinkage `α=0.25`;
- 1,024 fixed Sobol-normal points.

Screen result: **1.017417×**, interval **0.998465–1.028852**, 7/8 wins.

### Untouched 24-network validation

| Metric | Frozen validation result |
|---|---:|
| Aggregate raw-MSE ratio | **0.988896×** |
| 95% network-bootstrap interval | **0.944832–1.033138** |
| Wins | **8/24** |
| Worst candidate/baseline | **1.195191** |
| Mean correction/true-defect correlation | **-0.009195** |
| Weighted cosine with scored downstream defect | **-0.001333** |

The screen gain was selection noise. Validation raw MSE was about **1.12% worse**, and the branch would also require an extra true final-layer replay plus integration overhead. Even the favorable screen result would not have paid for the known 3.125% final-replay compute increment.

## Interpretation

### 1. The late error channel is sparse, but the support is target-aware

A tiny set of layer-31 coordinates can explain a large fraction of the final error. This is a strong positive mechanistic result. It means future layer-31 residual surrogates do not necessarily need to predict all 256 means accurately.

However, the useful set depends on signed cancellation in final-output space. Magnitude-only local scores recover only a minority of the oracle effect.

### 2. The target is Kerdock phase error, not distributional approximation error

The conditional models produced plausible Gaussianized activation means, but those corrections were nearly orthogonal to the true Kerdock mean defect. Better approximation of the activation distribution is not enough; the surrogate must correlate with the deterministic cubature residual of the protected rule.

### 3. Explicitly modeling more gates made the wrong object more accurate

Moving from 0 to 8 selected upstream gates did not improve the signed correction. It generally enlarged the correction and worsened layer-31 MSE. This is evidence against simply spending more compute on higher-dimensional orthant integration.

### 4. Residualization was the right final check

The residual-control form was the strongest plausible rescue because it removes model bias under the Kerdock rule. Its failure on a frozen 24-network split is therefore much more informative than the direct model's failure.

## Decision and reopen conditions

**STOP activation-region conditional integration as a standalone or residual-control submission branch. Do not run the untouched 64-network holdout.**

Retain only the sparse layer-31 oracle finding. Reopen this family only if at least one of the following becomes available:

1. an observable, signed predictor of the layer-31 Kerdock defect—not merely activation uncertainty or Gaussian nonlinearity;
2. a surviving layer-31 surrogate from another branch whose pointwise residual already validates on fresh networks;
3. a phase-aware statistic from the existing Kerdock basis structure or compiler pilot that predicts which sparse coordinates need positive versus negative correction;
4. a construction that avoids the extra full final-layer replay and demonstrates raw gain large enough to repay exact FlopScope cost.

The most promising use of the new result is therefore **targeted evaluation of a different validated layer-31 residual surrogate**, restricting expensive correction machinery to roughly 8–16 coordinates. It is not further gate enumeration or higher-dimensional Gaussian orthant work.

## Reproduction

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Sparse layer-31 oracle screen
python sparse_layer31_gate_oracle.py \
  --n-networks 8 --workers 4 --refs-per-half 8 \
  --ks 1 2 4 8 12 16 24 32

# Direct conditional-region screen and residual-control grid
python conditional_region_layer31.py \
  --split screen --workers 4 \
  --gate-dims 0 2 4 6 8 --ks 8 12 16 24 \
  --alphas 0.25 0.5 0.75 1.0 --n-qmc 1024

# Frozen validation; no retuning
python conditional_region_layer31.py \
  --split validation --workers 5 \
  --gate-dims 2 --ks 8 --alphas 0.25 --n-qmc 1024
```
