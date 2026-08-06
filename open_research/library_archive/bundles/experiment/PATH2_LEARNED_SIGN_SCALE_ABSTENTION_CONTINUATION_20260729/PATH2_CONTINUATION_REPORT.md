# Path 2 continuation — learned sign, scale, and abstention

**Evidence cutoff:** 2026-07-29 21:40 ET  
**Workspace:** `paths/02_learned_sign_scale_abstention/`

## Decision

**Freeze conservative K32 scaling as a validated oracle-anchor mechanism. Do not promote a learned runtime model.**

The continuation changes the diagnosis of Path 2:

1. The damaging K32 cases are predominantly **positive-direction overscaling**, not sign reversals.
2. A bounded scale is sufficient to remove the observed rotation tail while retaining promotion-level mean improvement.
3. The original layer-8 dispersion abstention statistic does **not** generalize as a universal harm detector.
4. Generic and permutation-invariant learned scale models remain too unstable at the current group count.
5. The remaining blocker is now almost entirely **legal estimation of the K32 anchor/correction**, not sign or coarse scale selection.

No result below is a legal submission candidate by itself: the K32 correction used for these scale experiments is still the independently estimated oracle anchor.

## Audited continuous-scale infrastructure

Future scale and sign evaluations no longer require rerunning every alpha. For each network/rotation the runner saves

- baseline error norm,
- error–correction inner product,
- correction norm squared,
- full correction vectors and two independent target halves.

This reconstructs final-output MSE exactly as a quadratic in alpha.

The optimized reference implementation uses float32 matrix multiplication with float64 reductions and avoids unnecessary deep propagation for anchor streams. Against the original audited 262,144-point-per-half runner:

- target vectors were identical;
- maximum relative anchor-moment discrepancy was about `7e-8`;
- maximum K32/K128 ratio shift was about `2.3e-5`.

All production conclusions below use four independent 262,144-point streams per network: two target halves and two anchor halves.

## Development panels

### Exact K32 scale geometry

Across the 12-case hard-rotation panel, every optimal K32 scale was positive:

- minimum `0.224`;
- mean `0.694`;
- maximum `1.284`.

Across the 24 canonical networks, optimal K32 scale was also always positive:

- minimum `0.413`;
- mean `1.039`;
- maximum `1.782`.

Thus the main failure mode is excessive magnitude on a correct-direction correction.

### Simple fixed scales

| K32 policy | Canonical pooled | Canonical wins | Canonical worst | Hard pooled | Hard wins | Hard worst |
|---|---:|---:|---:|---:|---:|---:|
| full alpha 1.0 | 0.4044 | 23/24 | 1.1119 | 0.9225 | 8/12 | 3.0911 |
| fixed alpha 0.45 | 0.5841 | 24/24 | 0.9239 | 0.7735 | 11/12 | 1.0002 |
| fixed alpha 0.50 | **0.5528** | **24/24** | **0.9222** | **0.7677** | 10/12 | **1.0495** |

Fixed `0.50` is the simplest stable scale baseline for the exact K32 direction. It is algebraically distinct from the closed M111 universal-shrink surrogate.

### Frozen bounded two-level scale

A higher-gain candidate was frozen before opening the new holdout:

- ordinary scale: `0.55`;
- high-risk scale: `0.45`;
- high-risk threshold: layer-8 six-fold Kerdock block relative dispersion above `0.0027637032840478275`.

Development results:

- canonical 24: `0.52469`, 24/24 wins, worst `0.92386`;
- hard 12: `0.74888`, 11/12 wins, worst `1.00020`;
- unique development 32: `0.55660`, 31/32 wins, worst `1.00020`.

## Immutable fresh holdout

The candidate definitions, twelve new deterministic base-network seeds, rotations `{3,11,97}`, QMC size, and metrics were frozen before reference generation.

The holdout contains **12 new base networks × 3 rotations = 36 cases**.

### Primary K32 results

| Frozen policy | Pooled ratio | Grouped bootstrap 95% CI | Wins | Worst | P90 |
|---|---:|---:|---:|---:|---:|
| full alpha 1.0 | 0.50331 | [0.41266, 0.62734] | 32/36 | 2.47916 | 1.01009 |
| fixed alpha 0.45 | 0.57674 | [0.53391, 0.62413] | 36/36 | 0.89923 | 0.75748 |
| fixed alpha 0.50 | 0.54989 | [0.50373, 0.60148] | 36/36 | 0.89195 | 0.74014 |
| bounded 0.55/0.45 | **0.52707** | **[0.47764, 0.58312]** | **36/36** | **0.88546** | **0.74352** |
| oracle per-case scale | 0.42422 | — | 36/36 | 0.87579 | 0.70322 |

The bounded candidate clears the raw `<=0.595` screen and its grouped bootstrap upper endpoint is also below `0.595` on this immutable holdout.

However, no fresh case exceeded the frozen risk threshold, so the holdout validated the ordinary `0.55` scale but did not independently validate the high-risk branch.

### The old abstention signal does not generalize

Several fresh full-scale failures had low layer-8 dispersion. The worst fresh full K32 case was `2.479`, with risk approximately `0.002241`, well below both frozen thresholds.

Therefore the earlier layer-8 full-correction/abstain rule was panel-specific. It should not be promoted as a general harmful-tail classifier. Its remaining use is only as a bounded scale modifier, where failure to trigger is safe.

## K128 teacher/ceiling

K128 retains substantially more headroom under conservative scale:

| K128 policy | Fresh pooled | Grouped 95% CI | Wins | Worst |
|---|---:|---:|---:|---:|
| full alpha 1.0 | 0.40844 | [0.30897, 0.54058] | 32/36 | 2.45037 |
| fixed alpha 0.50 | **0.46899** | **[0.43372, 0.51089]** | **36/36** | **0.90498** |
| oracle scale | 0.32066 | — | 36/36 | 0.90206 |

K128 remains a useful teacher and robustness ceiling, but K32 remains the deployment-sized target.

## Learned-model results

### Grouped development selection

A fully nested leave-one-base-network-out model selector over weight, trajectory, late-layer, and compact feature sets produced:

- canonical pooled `0.5734`;
- hard pooled `0.9788`;
- hard worst `3.0144`.

Feature/model selection was unstable across folds.

### Frozen fresh-holdout models

Models were frozen before holdout labels completed and trained only on the 32 development examples.

| Model | Fresh pooled | Wins | Worst |
|---|---:|---:|---:|
| fixed alpha 0.50 | 0.54989 | 36/36 | 0.89195 |
| compact bounded ridge | 0.50404 | 35/36 | 1.18974 |
| bounded ensemble | 0.48829 | 35/36 | 1.18974 |
| weight-only bounded ridge | 0.48215 | 35/36 | 1.18974 |
| compact unbounded ridge | 0.64102 | 32/36 | 6.89726 |

Bounded learned scaling improves the mean but still misses one overscaling case. Unbounded scale learning is decisively unsafe.

A compact-feature Mahalanobis OOD score ranked the missed fresh case first and an exploratory fallback improved fresh pooled ratio to about `0.5001` with worst `0.9533`. But the same OOD construction did not rescue the development hard tail under grouped OOF evaluation. It remains a research lead, not evidence for promotion.

## Gate assessment

| Gate | Result |
|---|---|
| Raw direct-control candidate/base <= 0.595 | **Pass at oracle-anchor level**: bounded holdout 0.5271; fixed 0.50 holdout 0.5499 |
| Favorable confidence interval | **Pass raw mechanism screen**: bounded grouped upper 0.5831; all intervals exclude 1 |
| Safe hard rotations | **Pass for bounded/fixed scaling**: development hard worst 1.0002 / 1.0495; fresh worst 0.8855 / 0.8920 |
| Abstention materially reduces hard tail | **Not established generally**; old l08 abstention fails fresh discrimination |
| Legal runtime K32 anchor | **Fail / unresolved** |
| Final subprocess adjusted score and cost | **Not measurable until a legal anchor source exists** |

## Final interpretation

**Scale is no longer the central scientific uncertainty.** A conservative positive scale in roughly the `0.45–0.55` range preserves enough of the K32 oracle gain and removes the observed rotation tail. There is no evidence that sign reversal is a common K32 failure mode in the audited canonical, hard, or fresh panels.

The recommended freeze is:

1. use fixed `alpha=0.50` as the simplest robust baseline;
2. retain the frozen bounded `0.55` ordinary / `0.45` high-risk policy as the higher-gain mechanism candidate;
3. stop unrestricted scale regression and stop treating l08 dispersion as a general abstention classifier;
4. spend subsequent Path-2 compute only on prediction of the **legal K32 anchor/residual**, preferably around a Path-1 analytic estimate;
5. preserve K128 as an offline teacher for correction direction, magnitude, and uncertainty.

A future learned-anchor experiment should use the new quadratic labels, keep every rotation of a base network in one fold, and use the fresh 12-network panel only as a locked test set.
