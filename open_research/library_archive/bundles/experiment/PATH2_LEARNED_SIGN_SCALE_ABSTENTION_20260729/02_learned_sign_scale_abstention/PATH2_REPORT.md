# Path 2 — learned sign, scale, and abstention

## Decision

**CONTINUE one frozen abstention lead; do not promote a learned model.**

The generic learned models tested here do not generalize under the required base-network grouping. An exploratory, interpretable layer-8 basis-block dispersion gate does remove the catastrophic hard-rotation tail while retaining essentially all of the K32 oracle benefit, but that feature was discovered on the existing panel. It therefore needs a new immutable validation cohort before it can count as evidence.

This experiment is an **oracle-K32 overlay study**. It evaluates whether legal trajectory features can decide when to apply an already-correct K32 direct-output correction. It does not solve legal estimation of the K32 signed anchor itself.

## Frozen data and split

- 24 canonical-rotation width-256 networks from the independent K32/K128 oracle study.
- Eight additional rotations from the four-network hard panel, after deduplicating the canonical rotation.
- 32 examples, 24 base-network groups.
- Every rotation of a base network stays in the same leave-one-base-network-out fold.
- 1,529 legal features were extracted from weights, complete-Kerdock trajectories, basis-block geometry, internal six-fold stability, and sample-anchor geometry.
- No oracle mean, target residual, candidate ratio, or hard-panel label enters feature extraction.

The labels are sparse: four harmful K32 examples come from only two base networks; both no-headroom examples come from one base network. This makes ordinary harm classification underidentified.

## Model results

### Generic learned benefit prediction

A grouped leave-one-base-network-out ridge suite predicted log candidate/base from three legal feature families, with feature selection performed inside each training fold.

Best configuration:

- feature family: rotation/block/anchor geometry;
- selected features: 16;
- ridge alpha: 1000;
- OOF log-ratio RMSE: 0.5846;
- Pearson correlation: -0.1591;
- Spearman correlation: -0.0022;
- predicted ratio for the actual 3.0829 catastrophic case: 0.4136.

The model is not merely weak; it is confidently wrong on the tail. This closes ordinary small-sample ridge benefit prediction on the archived panel. The prior width-256 edge-DWS V80 result remains a separate negative control because it targeted a different scalar.

### Exploratory layer-8 dispersion gate

The promising statistic is:

1. At post-ReLU layer 8, average each complete 512-row Kerdock basis block.
2. Partition the 129 complete bases into six fixed groups.
3. Compute the six group mean vectors.
4. Let the risk score be the mean norm of each group mean's deviation from their center, divided by the center norm.
5. Apply K32 only when the score is at or below the training cohort's 95th percentile.

This statistic is permutation-compatible at the neuron level, uses complete basis blocks, and is available from the existing Kerdock trajectory. It is not correction-agreement confidence.

Grouped OOF screen:

| Metric | K32 always | Exploratory gate |
|---|---:|---:|
| Canonical candidate/base | 0.403016 | 0.404626 |
| Canonical bootstrap 95% interval | — | [0.309243, 0.558666] |
| Canonical coverage | 100% | 95.83% |
| Canonical wins | 23/24 | 22/24 |
| Canonical worst | 1.106870 | 1.106870 |
| Hard-panel pooled candidate/base | 0.920637 | 0.753284 |
| Hard-panel coverage | 100% | 75% |
| Hard-panel worst | 3.082862 | 1.106870 |
| Hard-panel p90 | 1.150512 | 1.014074 |

The gate abstains on all three rotations of base network 493891104, including the two no-headroom rotations, and applies K32 everywhere else in the hard panel. Its incremental arithmetic is approximately 16.9 million additions plus small vector reductions if block means are accumulated during the existing layer-8 pass—negligible relative to the production package's roughly 175.62B effective compute.

### Fully nested selection check

A stricter nested procedure selected among 55 layer/fold-dispersion statistics and five quantiles using training folds only. It selected `l08_fold_rel_mean` in 22 of 24 outer folds, confirming that the signal is not a one-off feature-ranking accident. However, when the catastrophic base network was held out, the selector switched to a layer-0 statistic because the remaining training set contained no no-headroom group. The resulting hard-panel worst remained 3.082862.

Therefore the fixed layer-8 rule is a strong **post-hoc lead**, while the end-to-end learned selection procedure fails the tail gate.

## Gate assessment

| Requirement | Result |
|---|---|
| Direct-control candidate/base <= 0.595 | Exploratory oracle overlay passes: 0.404626 |
| Favorable confidence interval | Exploratory upper bootstrap endpoint 0.558666 |
| Abstention materially reduces hard tail | Exploratory worst 3.082862 -> 1.106870; pooled 0.920637 -> 0.753284 |
| All rotations grouped | Pass |
| Hard panel in model selection | Pass for the nested procedure; fixed rule is post-hoc |
| Legal K32 anchor available | **Fail / outside this experiment** |
| Immutable validation of selected gate | **Fail** |
| Promotion | **No** |

## Required next experiment

Freeze before producing any new labels:

- feature: `l08_fold_rel_mean` exactly as implemented in `l08_abstention_gate.py`;
- direction: higher means more risk;
- threshold: 95th percentile of the frozen training base-network distribution;
- action: K32 direct-output correction below threshold, abstain above it;
- no learned feature selection or threshold retuning;
- all rotations of a base network remain grouped.

Run on a new immutable cohort with enough independent harmful groups to identify a tail model. A reasonable minimum is 32 new base networks with three fixed rotations each, with the full oracle-headroom and K32 direct-control labels produced for every pair—not selective deepening of suspicious cases.

For continuous scale/sign learning, future label artifacts must save enough information to evaluate the correction quadratic:

- baseline error norm;
- error-correction inner product;
- correction norm squared;
- or equivalently MSE at alpha = -1, 0, and +1;
- preferably the K32 and K128 final-output correction vectors themselves.

Without those fields, archived full-correction MSE tables identify only alpha in {0,1}; they cannot support honest continuous-alpha training.

## Reproduction

```bash
python extract_legal_features.py --threads 16
python evaluate_models.py
python test_gate.py
```

Primary outputs:

- `legal_features_and_labels.csv`
- `results/PATH2_RESULTS.json`
- `results/oof_decisions.csv`
- `l08_abstention_gate.py`
