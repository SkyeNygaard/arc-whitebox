# Non-K3 continuation report — ARC WhestBench

**Date:** 2026-07-28  
**Scope:** All transported-third-cumulant / K3 work was excluded. This continuation searched only distinct implementation, adaptive-sampling, control-variate, and direct-learning routes.

## Executive conclusion

The only new branch that survived repeated untouched holdouts was a generalized **adaptive exact-residual suffix compiler**.

It extends the earlier frozen two-layer compiler to candidate suffix depths 2–6. A 2,064-row, basis-balanced pilot classifies stable-on, stable-off, and kink coordinates in the final six layers. For each candidate depth, the method predicts arithmetic cost using only pilot kink counts, chooses the cheapest depth, algebraically fuses stable-on paths, drops stable-off paths, propagates full Kerdock rows only through kink coordinates, and adds a paired pilot residual correction.

The selector never sees reference targets, full-design errors, or leaderboard labels.

Across three independent 20-network width-256/depth-32 suites with complete 66,048-point Kerdock evaluation and shared independent rotated-Kerdock references:

| Metric | Result |
|---|---:|
| Networks | 60 |
| Chosen depths | 2: 10, 3: 14, 4: 14, 5: 10, 6: 12 |
| Mean raw-MSE ratio | 1.00170 |
| Raw-MSE mean 95% interval | 0.99976–1.00372 |
| Mean ideal arithmetic-cost ratio | 0.95358 |
| Mean ideal score ratio | 0.95517 |
| Ideal score-ratio interval | 0.94976–0.96047 |
| Mean calibrated effective-score ratio | **0.97865** |
| Calibrated interval, network bootstrap | **0.97566–0.98163** |
| Calibrated wins | 59/60 |
| Worst calibrated network | 1.00222 |

The raw estimator is statistically almost unchanged. The gain comes from eliminating certified linear suffix work.

## Why the 0.955 ideal ratio is not the score projection

The earlier two-layer compiler had:

- ideal analytical cost ratio approximately 0.96838;
- measured effective-score proxy 0.984341.

Only about 49.5% of ideal arithmetic savings survived the earlier effective accounting and overhead. Applying that same retention fraction to the new candidate gives a central calibrated ratio of 0.97865.

Using the earlier compiler's bootstrap interval to vary this calibration gives a projected score range of approximately:

- **1.431e-7 to 1.453e-7**
- central estimate: **1.442e-7**

from the assembly-free baseline of 1.473e-7.

This is not an official score. It may only be slightly better than the previous two-layer projection of 1.4499e-7. Exact blocked-layout integration, FlopScope accounting, residual wall time, and Mini-100 remain decisive.

## Frozen candidate

Use this exact configuration before any further tuning:

```text
candidate suffix depths: 2, 3, 4, 5, 6
pilot: 8 columns from each of 129 Kerdock bases, including antipodes
pilot rows: 2,064
stable rule: at most one minority-sign pilot event
selector: minimum predicted arithmetic cost
pilot residual correction: enabled
padding: exact shapes first; test padding-to-16 separately
```

Do not adopt the following without another untouched official-like holdout:

- minority threshold 2;
- suffix depths 7–8;
- reduced pilot size;
- learned score/error selector.

## Mechanism

For candidate suffix depth `k`, let `z_l` be the number of pilot-classified kink coordinates at suffix layer `l`, `P=2064`, `N=66048`, and width `D=256`.

The dense-equivalent suffix work proxy is:

```text
k * P/N
+ sum_l z_l/D
+ sum_{j<l} z_j*z_l/D^2
```

The total-network ratio is:

```text
(depth - k + suffix_work) / depth
```

Stable-on activations remain linear in the anchor activations and prior kink activations. Their paths are composed symbolically through the remaining weight matrices. Stable-off coordinates vanish. Full-row nonlinear propagation is reserved for kink coordinates.

The final estimate is:

```text
compiled full-cloud mean
+ mean_over_pilot(exact_suffix_output - compiled_suffix_output)
```

The pilot correction protects against rare switches missed by the stable classification.

## Evidence by suite

### Holdout A

- 20 untouched networks.
- Two independent complete-Kerdock rotations used as shared noisy references.
- Mean ideal ratio: 0.95847.
- Mean calibrated ratio: 0.97992.
- 20/20 ideal wins and 20/20 calibrated wins.

### Holdout B

- Separate 20-network seed family.
- Frozen minority threshold 1.
- Mean ideal ratio: 0.95046.
- Mean calibrated ratio: 0.97749.
- 20/20 ideal and calibrated wins.

A threshold-2 rule was marginally better in 16/20 comparisons, but its paired advantage interval barely included zero. Because threshold 2 had already shown selection risk in the earlier two-layer work, it remains research-only.

### Holdout C

- Third independent 20-network family.
- Direct comparison of maximum suffix depth 6 versus 8.
- Maximum-depth-6 selector mean calibrated ratio: 0.97855.
- Maximum-depth-8 selector mean calibrated ratio: 0.97829.
- Depth 7 was selected only once; depth 8 was never selected.
- The incremental gain was negligible and symbolic runtime increased sharply.

Freeze maximum depth 6.

## Other non-K3 branches tested

### Adaptive stopping of the Kerdock basis sequence — closed

There is a large oracle opportunity: choosing the best nested number of bases per network produced a mean oracle proxy near 0.6245. Cheap internal diagnostics did not identify it.

- Learned frozen selector mean ratio: approximately 1.390.
- Worst network: approximately 6.31.
- Fixed 64/96-basis rules were unstable.
- Cheap stopping diagnostics are not trustworthy.

### Direct residual prediction from weights/cheap trajectory statistics — closed in tested form

A first apparent success was traced to a leaked feature duplicating the full-design mean. After removing leakage, a genuine 16-basis feature model achieved:

- only 1.006x gain over the partial estimator;
- residual MSE still approximately 16.07x the full Kerdock MSE;
- four of five frozen test networks worsened.

Simple ridge/tree distillation from cheap summary statistics is not viable. A much richer equivariant model remains conceptually possible, but this suite provides no positive evidence for it.

### Exact layer-1 mean control variate on complete Kerdock — real but unsafe

A frozen rank-1 exact-mean control transferred modestly:

- pooled raw-MSE ratio: 0.97333;
- wins: 6/10;
- bootstrap interval: 0.8506–1.0560.

When composed with the suffix compiler, aggregate MSE improved but individual-network volatility remained unacceptable. Target-free basis-dispersion shrinkage did not fix it:

- pooled ratio across 25 networks: 0.96606;
- interval: 0.9206–1.0045;
- wins: 15/25;
- worst network: 1.263.

Do not include this control in the submission candidate.

### Pilot-size and geometry refinements — no frozen improvement

- Halving the pilot reduced arithmetic but increased classification error enough to erase the gain.
- Alternative fixed-cost column spreads showed no clear robust winner.
- Keep the existing eight columns per basis.

## Recommended next action

1. Integrate the adaptive 2–6 compiler into the protected persistent blocked layout.
2. Preserve the assembly-free estimator as a fallback.
3. Run exact-shape kernels first; padding-to-16 is a separate benchmark.
4. Run FlopScope and grader-like wall time on the complete Mini-100.
5. Compare paired per-network raw errors and adjusted scores.
6. Replace the protected submission only if:
   - aggregate adjusted score improves;
   - no network has a material failure;
   - conversion/packing overhead does not erase the gain.
7. Keep the original frozen two-layer compiler as the fallback if generalized symbolic bookkeeping is operationally expensive.

## Competition-level interpretation

This continuation found a credible additional implementation gain, not a winning statistical estimator. The public leader's score regime is still far below the projected Kerdock package. The generalized compiler is worth integrating because it is mechanistic, target-free, and nearly prediction-preserving, but it does not explain the leading regime.

