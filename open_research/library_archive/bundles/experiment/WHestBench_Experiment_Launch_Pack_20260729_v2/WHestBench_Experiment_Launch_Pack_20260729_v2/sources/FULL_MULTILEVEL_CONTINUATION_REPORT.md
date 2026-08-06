# Full multilevel prefix/suffix estimator — continuation report

**Date:** July 29, 2026  
**Challenge:** ARC White-Box Estimation Challenge 2026  
**Evidence level:** architecture-matched synthetic networks with independent high-precision structured references. **Not** an official Mini-100 or exact FlopScope result.

## Executive conclusion

The continuation found one scientifically real improvement to the multilevel idea:

> Fit the control on **complete Kerdock basis-block means**, rather than on individual particles.

That change produced repeated raw-error gains at only about **1.50% added compute**. However, the gain is not robust enough to submit:

- a frozen 12-network gated evaluation had central adjusted gain **1.171×**, but its network-bootstrap 95% interval was **0.958–1.450×**;
- it contained a material **1.249×** network failure;
- adding a 1% safety margin made the estimator completely inert on twelve fresh networks;
- requiring two independent pilot bases to agree still selected a fresh **1.738×** failure and produced aggregate adjusted gain **0.950×**.

**Final verdict:** close the current multilevel estimator as a deployable submission branch. Preserve blockwise sparse cubic fitting as a useful mechanistic result and benchmark for any future phase-aware predictor.

## 1. Starting point

The previous continuation had already closed:

1. stable-gate suffix MLMC as a statistical estimator;
2. a generic homogeneous Gaussian layer-31 residual surrogate;
3. a scalar coefficient estimated from a small independent exact pilot.

The live target remained the dominant layer-31 mean-defect channel, but a better low-level control was needed.

## 2. Sparse independent layer-31 mean estimation

### Construction

An observable selector chose 8–24 layer-31 coordinates. Their mean defects were estimated using independent orthogonal-basis pilots and replayed through the true final ReLU.

### Result

The pilot signal existed but was far too weak for its cost:

- best discovery raw gain was about **1.029×** using 32 pilot bases;
- low-cost configurations gained only about **0.7% raw**;
- no nonzero frozen configuration was adjusted-score positive.

**Decision:** direct sparse mean estimation is closed.

## 3. Gaussian control inside the pilot

The next estimator used

\[
\widehat\mu_{31}
=E[g_{31}]
+\widehat E_{\rm pilot}[a_{31}-g_{31}],
\]

so that the independent pilot corrected the known bias of the Gaussian closure.

The result was decisively negative: for selected coordinates, the residual block variance was typically **7–13 times larger** than the raw layer-31 block variance. The closure is therefore not merely biased; it is a poor control variate in the target subspace.

**Decision:** closed.

## 4. Sparse Hermite-cubic control

### Rowwise version

The control used standardized coordinatewise cubic features

\[
H_3(z_i)=z_i^3-3z_i
\]

on the full Kerdock cloud, with their expectations estimated from an independent pilot.

A rowwise regression initially looked strong:

- discovery raw gain: **1.139×**;
- discovery adjusted gain: **1.104×**.

It then catastrophically reversed on frozen validation:

- aggregate raw gain: **0.592×**;
- validation wins: **0/4**;
- worst network: **4.18×** baseline MSE.

This confirmed that particle-level predictability does not imply correct Kerdock quadrature phase.

### Blockwise version

The correct statistical unit is one complete antipodal orthonormal-basis block. The revised method regressed the 129 final-output block means on 129 cubic-feature block means.

Frozen screen configuration:

```text
selector: Gaussian mean-gap × final sensitivity
coordinates: 8
feature: standardized H3
independent pilot: 2 complete bases
block ridge: 1.0
correction shrink: 0.25
added compute proxy: 0.015021
```

Initial frozen evidence:

| Split | Raw gain | Adjusted gain | Wins | Worst candidate/baseline |
|---|---:|---:|---:|---:|
| Discovery, 4 networks | 1.088× | 1.072× | 3/4 | 1.032× |
| Validation, 4 networks | 1.060× | 1.044× | 3/4 | 1.085× |

This was the first reproducible positive signal in the continuation.

## 5. Fresh-network instability

Ungated fresh networks showed that the mechanism is real but phase-unstable: several networks improved substantially, while others worsened by roughly 1.17×, 1.57×, and 2.92×.

The important distinction is:

- the control often reduces dispersion among Kerdock basis blocks;
- reducing block dispersion does **not** guarantee that the complete-design mean moves toward the Gaussian expectation;
- the remaining error is a signed, network-specific high-degree phase.

## 6. Frozen post-pilot gate

The first frozen safety rule applied the correction only when:

```text
cross-fitted corrected block-variance ratio < 1
and
fold-correction relative SD < 1
```

### Twelve-network result

| Metric | Result |
|---|---:|
| Networks | 12 |
| Corrections applied | 4 |
| Raw gain, ratio of aggregate MSE | **1.188450×** |
| Adjusted gain after pilot cost | **1.170862×** |
| Noise-corrected gain | **1.208437×** |
| Reference-half gains | 1.194861× / 1.150627× |
| Applied-network wins | 3/4 |
| Worst candidate/baseline | **1.249249×** |
| Adjusted bootstrap 95% | **0.958340–1.450233×** |
| Bootstrap probability adjusted gain > 1 | 86.8% |

Applied cases:

| Seed | Predicted block variance | Fold relative SD | Pilot correction cosine | Candidate/baseline |
|---:|---:|---:|---:|---:|
| 1245061180 | 0.996562 | 0.464 | 0.796 | 0.801× |
| 1123961422 | 0.993974 | 0.503 | -0.765 | 1.249× |
| 176687031 | 0.982200 | 0.183 | 0.972 | 0.533× |
| 1853497945 | 0.986850 | 0.850 | -0.329 | 0.994× |

The central result is positive, but the interval includes no gain and one selected network materially fails.

## 7. Two attempted safety repairs

### 7.1 One-percent variance margin

A principled margin required predicted block-variance ratio below 0.99.

On twelve new random networks:

- corrections applied: **0/12**;
- raw estimator: exact fallback;
- adjusted gain after paying the pilot: **0.985201×**.

The safe estimator is inert and score-negative.

### 7.2 Independent-pilot agreement

The material failure in the first 12-network block had opposite corrections from its two pilot bases: correction cosine **−0.765**. The large wins had positive cosines of approximately **0.796** and **0.972**.

A new frozen rule therefore required:

```text
block-variance ratio < 1
fold relative SD < 1
two-pilot correction cosine > 0.5
```

It selected two of twelve fresh networks:

| Seed | Pilot correction cosine | Candidate/baseline | Reference half 1 | Reference half 2 |
|---:|---:|---:|---:|---:|
| 992148826 | 0.733 | 1.738× | 1.672× | 1.652× |
| 1529301148 | 0.520 | 0.896× | 0.898× | 0.916× |

Aggregate result:

| Metric | Result |
|---|---:|
| Raw gain | **0.963922×** |
| Adjusted gain | **0.949657×** |
| Worst candidate/baseline | **1.738147×** |
| Adjusted bootstrap 95% | 0.852163–1.001622× |
| Bootstrap probability adjusted gain > 1 | 3.5% |

The 1.738× failure was confirmed by both independent reference halves. Pilot agreement therefore detects sampling instability but not systematic quadrature-phase error.

## 8. Interpretation

This continuation establishes four useful facts.

### 8.1 Multilevel allocation was not the main missing ingredient

Several increasingly careful allocations failed because the coarse estimator did not predict the signed quadrature defect. More levels cannot repair that.

### 8.2 Kerdock basis blocks are the right regression unit

Blockwise fitting materially outperformed rowwise fitting and should be retained in future residual-control work.

### 8.3 Variance reduction and bias correction separate

A control can reduce cross-block variance while moving the complete-design mean in the wrong direction. This explains why internal cross-validation and pilot agreement can both look excellent on a failing network.

### 8.4 The missing object is phase

The remaining Kerdock error is not ordinary sampling noise. It is a small signed high-degree component whose direction changes by network. A deployable method must predict that phase from genuinely external information, not from same-design dispersion alone.

## 9. Final decision

| Branch | Decision |
|---|---|
| Direct sparse layer-31 pilot mean | Closed; gain too small for cost |
| Gaussian-controlled sparse pilot | Closed; residual variance 7–13× larger |
| Rowwise sparse cubic control | Closed; frozen catastrophic reversal |
| Blockwise sparse cubic control | Scientifically positive, not robust |
| Block-variance/fold-stability gate | Positive central estimate, CI crosses no gain and has tail failure |
| 1% safety margin | Closed; never fires |
| Two-pilot agreement gate | Closed; fresh 1.738× failure |
| Full multilevel prefix/suffix estimator | **Closed as a submission branch** |

## 10. Reopen conditions

Reopen only if a new method supplies at least one of:

1. an external phase predictor validated on at least 50 untouched networks;
2. a correction algebraically zero-mean on complete Kerdock while targeting degree ≥6;
3. an independent anchor accurate enough that correction sign no longer comes from same-design block dispersion;
4. a white-box equivariant model trained specifically to predict the blockwise cubic correction direction.

Any reopened candidate should require:

- adjusted-score confidence interval excluding no gain;
- no network worse than 1.10× in a frozen holdout;
- exact FlopScope and residual-wall accounting;
- complete Kerdock preserved as the fallback.

## Reproduction artifacts

The accompanying bundle contains the fixed scripts, compact summaries, per-network rows, bootstrap-ready JSON, and SHA-256 checksums.
