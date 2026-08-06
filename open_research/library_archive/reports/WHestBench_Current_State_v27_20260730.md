# WHestBench current state v27

**Date:** 2026-07-30  
**Purpose:** Reconcile the latest covariance-closure update with the complete experiment ledger and the newest post-v26 Library reports.  
**Protected or official cohort opened:** No.  
**Canonical workbook:** `whestbench_canonical_research_ledger_20260730_reconciled_v27_full_experiment_synthesis.xlsx`

## Executive conclusion

The project now has a much cleaner structure than v26 suggested.

1. **Static and ordinary sampling approaches are closed as winning routes.** The static signed theorem caps same-cost improvement near 1.067×, and the score-law audit explains why buying lower raw MSE with more ordinary rows is score-neutral or negative. The 90,624-row multifidelity estimator is therefore not a candidate despite improving raw MSE.
2. **The adaptive direct-output PCA source remains a genuine discovery, but its complete tested linear estimator class is closed.** The source captures most of the error energy with no replay and no extra evaluations. However, selected, dense, late-only and all-layer checkpoint-gauge programs fail by large margins, with explicit empirical dual certificates. This eliminates the former v26 lead path.
3. **Analytic moment propagation is the only remaining constructive branch with a measured winning-scale ceiling.** Oracle moments give raw MSE around `1.30e-7` in the score-law audit, which would score around `1.30e-8` below the 10% compute floor. An 80% reduction from `1.481e-7` only requires raw MSE at most `2.962e-7` below that floor.
4. **The dense bivariate Edgeworth correction is not the newly open question.** The ledger already records a third-plus-fourth-order pair correction reducing next-layer variance error from 2.564% under Gaussian closure to 0.288% with oracle mixed moments. The newest work is valuable because it isolates why this has not become an estimator: the mixed moment state has not been propagated legally, cheaply and compatibly through all 32 layers.
5. **The latest Tucker-spectrum result makes the cost side plausible, not solved.** The empirical third-cumulant tensor becomes strongly low multilinear rank with depth, suggesting a depth-adaptive Tucker representation around 3B FLOPs. But those factors were diagnosed from empirical/oracle activations. A competition estimator must generate them analytically from the weights and its previous legal state.

The canonical winning path is therefore:

> **Propagate a compressed, symmetric, PSD and mutually realizable mixed-moment state analytically through the full network, using a bivariate third/fourth-order ReLU covariance map, and pass the complete adjusted-score gate without oracle or sampled state information.**

No current implementation passes that gate.

---

## 1. The four evidence layers

Many apparent contradictions disappear when experiments are sorted by what they actually establish.

### 1.1 Oracle mechanism experiments

These answer: *What information would be sufficient if it were available exactly?*

Examples:

- Oracle marginal Edgeworth moments produce a winning-scale final mean.
- Oracle pair moments produce a 0.288% next-variance error.
- The adaptive direct-output PCA source leaves only about 7.5% pooled residual energy.
- Oracle late-coordinate and conic sources show large target capacity.

These experiments identify the right mathematical objects. They do **not** show that those objects can be obtained from legal runtime information.

### 1.2 Deployability and observability experiments

These answer: *Can the sufficient information be obtained legally, cheaply and out of sample?*

Examples:

- Direct-source scalar features predict magnitude but not signed direction.
- Finite-pilot pair moments do not beat empirical covariance.
- Sampled cumulants require too many rows.
- Generic pairwise or downstream regressors do not improve the required quadratic contractions.
- The complete linear checkpoint-gauge class misses its source-specific variance-cost allowance by large factors.

These experiments close implementations and sometimes precise classes, but they do not erase the corresponding oracle mechanism.

### 1.3 Free-rollout and compatibility experiments

These answer: *Do locally accurate approximations remain mutually consistent through 32 recursive layers?*

Examples:

- Learned covariance improves sigma but can worsen the final mean.
- Edgeworth mean and covariance corrections help separately while their full-strength combination fails.
- Damping does not robustly fix the incompatibility.
- Oracle recursive diagnostics reduce error substantially but remain above the candidate target.

This is why one-step pair accuracy cannot be treated as a candidate score.

### 1.4 Score and compute experiments

These answer: *Does the statistical gain move the actual competition metric?*

Above the 10% compute floor, an unbiased Monte Carlo-rate estimator with per-row variance constant `V` and per-row cost `f` has score proportional to `V f / B`. Increasing the number of rows lowers raw MSE and raises compute by matching factors. Consequently:

- the 90,624-row multifidelity rule is adjusted-score negative;
- many raw-MSE improvements from additional trajectories were never viable candidates;
- a bias-limited analytic estimator below the floor is qualitatively different, because its score is simply 0.1 times its bias-limited MSE.

---

## 2. Static cubature, stronger designs and sampling descendants

### What was learned

Kerdock is not merely a historical baseline. The theorem program shows that broad static rule classes cannot deliver the required improvement:

- infinite-width arbitrary-signed static rules have a same-cost gain cap near 1.067×;
- mass and diffuse-negativity loopholes are closed in the declared theorem scopes;
- the new finite-width 29-state tensor-degree certificate gives the strongest recorded actual-width arbitrary-node signed/static floor.

The finite-width theorem is a major publication result, but its certified floor remains 2.913× below what would be needed to exclude a 4.34× same-cost gain. Its next justified step is the degree-62 excursion-return moment dual, not another local design search.

### What is closed experimentally

The following do not constitute winning routes:

- partial Kerdock stopping and adaptive basis selection;
- fixed reweighting, signed static tuning and ordinary rotation mixtures;
- Sobol, Haar and stronger ordinary design pilots as standalone candidates;
- rotated multifidelity additions;
- low-rank activation MLMC;
- terminal smoothing and same-cloud Edgeworth blending;
- low-dimensional activation quadrature inferred from participation ratio;
- additional ordinary rows under the MC-rate score law.

### Current disposition

**Closed as a winning route.** Retain the production 129-basis implementation, the exact proof corpus and the degree-62 publication lane.

The latest fitted harmonic-spectrum diagnostic supports a qualitative high-frequency explanation for these failures, but it is not a theorem and should not override exact Hermite/Gegenbauer calculations or the certified static floors.

---

## 3. Adaptive direct-output source and phase

### What remains true

The adaptive direct-output PCA source is legal, target-free, gauge-invariant and constructed from the 129 baseline group means. Across confirmation cases:

- rank is about 34–38;
- pooled source-only residual is about 0.07486;
- worst confirmation residual is about 0.183;
- no new network evaluations or nonlinear replay are required.

This proves that the baseline error has a large structured component in an observable, legal output-space subspace.

### Why it did not become a candidate

The source coefficients are signed. The experiments repeatedly find that:

- error magnitude is strongly observable;
- coefficient direction or absolute phase is not;
- legal scalar phase-feature regressions have negative leave-one-network-out R² and chance sign accuracy;
- companion pilots, Walsh features, curvature features, invariant contractions and learned absolute offsets fail transfer or score economics.

The v26 plan correctly called for one complete source-specific checkpoint-gauge audit. That audit is now finished.

### New class closure

The source-specific SOCP and the v27 all-layer continuation establish:

- the selected `[1,4,32]` rule misses the permitted contraction difficulty by more than sixfold on untouched confirmation covariance;
- the empirical dual lower bound misses by more than fivefold;
- complete orthonormal-basis block sampling remains far outside the gate;
- late-only checkpoints `[1,29/30/31,32]` fail;
- the complete `[1,2,...,32]` empirical linear chain has dual adjusted-score lower bounds far above the required threshold, even on a favorable case.

### Current disposition

**The declared linear, unbiased, independent-block checkpoint-gauge class is closed.**

The source itself is retained as a structural result. The only remaining direct-source branch is a genuinely different class escape:

- an exact nonlinear or biased identity;
- a shared-arithmetic estimator of a transported late absolute innovation;
- or a broader impossibility theorem.

There should be no more checkpoint partition tuning or generic phase-feature learning.

---

## 4. Analytic moment propagation

This is the only branch with both a measured winning-scale oracle and a concrete local mathematical target.

### 4.1 Marginal closure is largely solved

The oracle marginal experiments show:

- Gaussian marginals are inadequate;
- third-order Edgeworth is a large improvement;
- third-plus-fourth order is the best tested truncation;
- fifth/sixth order and formal higher-order terms can worsen deep-layer behavior.

The latest layerwise diagnostic finds the Edgeworth marginal mean error around `1.9e-4` relative per layer, stable across depth and roughly 7–11× below the Gaussian error. This is consistent with the known oracle final-MSE scale.

**Interpretation:** the marginal ReLU expectation formula is not the primary blocker when its moments are correct.

### 4.2 Bivariate covariance closure is sufficient under oracle moments

The existing M40 result is decisive:

| Closure | Next-variance relative RMS |
|---|---:|
| Gaussian pair map | 2.564% |
| Third-order pair correction | 1.009% |
| Third + fourth order | 0.288% |

This already answers the question “can a bivariate higher-moment map reach approximately 0.3%?” in the affirmative under oracle pair moments.

The target-width learned transported-cumulant experiment also reaches 0.856% one-step variance error on held-out cases, showing that the local correction law is not purely a tiny-width artifact.

### 4.3 Why prior deployment attempts failed

The ledger contains several distinct failure modes:

1. **Sampled moments are too expensive or noisy.** Estimating first, second and higher pair moments from a finite pilot becomes comparable to direct covariance sampling and does not create a score advantage.
2. **Pairwise reconstruction error is the wrong loss.** Models can have high pair-level R² and still worsen the contracted next-layer variance.
3. **Downstream correction remains high-dimensional.** Oracle covariance block replacements need large ranks to preserve the required quadratic forms.
4. **Teacher-forced success does not imply free rollout.** Oracle current moments remove the state-acquisition problem.
5. **Mean and covariance states can be mutually inconsistent.** Full-strength combinations of individually helpful corrections worsen final output; damping does not robustly solve this.
6. **Same-cloud moment estimates do not break the MC-rate economics.** Using propagated rows to obtain every moment largely recreates direct estimation with additional complexity.

### 4.4 What the latest update adds

The latest diagnostics sharpen the failure into one transport problem:

- marginal Edgeworth closure is good;
- Gaussian covariance closure is around 1.34% RMS on average and grows with depth;
- the error is neuron-specific scatter, so a universal layer calibration gains only about 1.01×;
- delaying Gaussian closure makes error worse, suggesting distributional perturbations are amplified rather than damped;
- early activation states remain high-dimensional despite a low participation ratio at depth, so exact low-dimensional state quadrature is not viable;
- the empirical third-cumulant mode unfolding becomes strongly Tucker-compressible with depth.

Reported captured energy for rank 32 is approximately:

- layer 1: 0.639;
- layer 8: 0.967;
- layer 31: 0.9999.

A depth-adaptive dense/128 early, 64 middle and 32 deep schedule was estimated around 3B FLOPs, comfortably under the 27.2B 10%-floor threshold.

### 4.5 What remains unproved

The Tucker diagnostic was built from empirical/oracle activations. It therefore does not yet solve the competition problem.

A real estimator must answer all of the following:

1. How are the Tucker factors and core generated analytically from the previous legal state and the realized weights?
2. Does compression preserve the exact mixed contractions used by the bivariate correction, such as `κ_iij` and `κ_ijj`?
3. Does it preserve the downstream quadratic forms `wᵀ ΔCov w`, rather than merely tensor Frobenius energy?
4. Is the resulting covariance PSD and the combined mean/covariance/cumulant state realizable?
5. Does the state remain stable without oracle reinitialization for all 32 layers?
6. Does the final raw MSE fall below `2.962e-7` while complete effective compute remains at or below 10% of budget?

These are now the primary research questions.

---

## 5. The exact constructive program

### Gate 1 — Common-harness dense oracle consistency

Reproduce M40 in the same width-256 harness used by the newest covariance diagnostic.

Required checks:

- exact zero closure error at the truly Gaussian first layer, up to reference noise;
- independent reference halves and convergence with reference sample count;
- Gaussian, third-order and third-plus-fourth next-variance errors matching the archived result;
- no ambiguity in layer indexing or standardization conventions.

This is a consistency gate, not novel research.

### Gate 2 — Oracle Tucker compression on the right metric

Replace the dense oracle mixed-cumulant tensor with a depth-adaptive Tucker representation.

Measure:

- `κ_iij` and `κ_ijj` error;
- next-layer `wᵀ Cov(a) w` error;
- downstream-adjoint-weighted error;
- dense-versus-compressed final sensitivity;
- PSD and symmetry violations.

Global tensor energy capture is not a promotion metric.

### Gate 3 — Analytic factor/core recurrence

Derive a legal recurrence for:

- marginal mean, variance, skewness and kurtosis;
- Tucker factor matrices;
- the small Tucker core;
- required mixed pair contractions;
- any diagonal or residual correction needed for PSD.

No empirical activation rows, target moments or oracle decomposition may enter the runtime estimator.

### Gate 4 — Joint realizability and free rollout

Run all 32 layers without oracle refresh.

Promotion requires:

- both marginal and covariance corrections helping jointly;
- stable ranks and bounded condition numbers;
- PSD covariance without destructive clipping;
- safe dead-neuron and near-singular-correlation handling;
- grouped whole-network validation and tail reporting;
- raw MSE at most `2.962e-7` for the 80% target.

### Gate 5 — Complete score and packaging

Charge:

- all factor/core construction;
- all matrix and tensor contractions;
- any learned fixed coefficients or model inference;
- setup, gating and fallback paths;
- residual wall time under the competition model.

Promotion target:

- effective compute at most 10% of budget;
- adjusted score at most `2.962e-8`, with material confidence and tail slack;
- immutable protocol before protected evaluation.

---

## 6. Secondary and proof lanes

### A90 conic master

The Library still contains only the exact inclusion theorem and the provisional 12-case capacity result. No later full-48 reconstruction, tail or physical-covariance completion was found.

Disposition: **at most one bounded full-48 audit**, subordinate to the analytic moment program.

### Nonlinear late innovation

The exact late-innovation decomposition and direct-output source make this mathematically meaningful. The linear checkpoint class closure means any reopening must explicitly explain the class escape.

Disposition: **math-only dormant hedge**.

### Degree-62 finite-width proof

The 29-state result establishes an exact tensor-degree Markov framework and a strong actual-width theorem. The degree-28 component is locally saturated for competition purposes.

Disposition: **active publication lane**, with degree-62 excursion-return bounds and an independent implementation as the next work.

### Shipping baseline

The 129-basis production estimator remains the best validated executable. The current update's 112-basis timeout-insurance proposal is not canonical without repeated packaged timing on representative official-like hardware.

Disposition: **retain production; validate operational risk only**.

---

## 7. Important evidence and reproducibility caveats

1. The current conversation attachments contain the score-law and covariance-closure updates, but the named `15_score_law_and_v26_lead_path_audit.md` and the reported scripts (`closure.py`, `cov_closure.py`, `lag_closure.py`, `calib.py`, `rank.py`, `cp_rank.py`) were not located by the Library searches used for this reconciliation. The ledger marks these as reported current-update evidence rather than independently archived artifacts.
2. The harmonic-spectrum fit is a diagnostic, not a quantitative cubature theorem.
3. The Tucker result establishes low multilinear/unfolding rank, not low symmetric CP rank.
4. Oracle empirical Tucker factors are not a legal runtime state.
5. The all-layer checkpoint duals are exact for their empirical covariance programs; they are not a universal nonlinear impossibility theorem.
6. The finite-width 29-state certificate has a generated-row proof trust base and still needs independent implementation review before publication-strength wording.
7. The `1.30e-7` oracle result should be reconciled on a common cohort with the older approximately `2.21e-7` oracle-sigma figure before declaring a single canonical oracle ceiling.

---

## 8. Canonical disposition

### Primary constructive program

**Analytic propagation of a compressed, PSD, mutually realizable mixed-moment state.**

### Narrow hedge

**Exact nonlinear/shared-arithmetic late absolute innovation.**

### Optional bounded audit

**A90 full-48 capacity and physical covariance.**

### Proof lane

**Finite-width degree-62 excursion-return moment dual.**

### Closed winning routes

- static nodes, weights and stronger ordinary designs;
- ordinary additional-row sampling and the 90,624-row multifidelity candidate;
- low-rank activation particles and MLMC;
- terminal smoothing and same-cloud moment blending;
- generic direct-source phase learners;
- linear independent-block checkpoint gauges;
- sampled moment propagation and tested finite-pilot pair learners;
- universal covariance calibration, delayed closure and low-dimensional activation quadrature;
- conic A30, natural rank-4/rank-5 and rank-12 descendants.

## Bottom line

The project is no longer blocked by a lack of ideas or by uncertainty about where the error lives. It is blocked by one precise constructive gap:

> **Build a legal analytic recurrence for the compressed mixed cumulants that make the known 0.288% bivariate covariance closure work, while maintaining a compatible mean/covariance state through all 32 layers.**

That path is difficult but coherent. It has a measured winning oracle, a plausible compute envelope, explicit prior failures that define the necessary architecture and a finite sequence of decisive gates. No other completed experiment family currently offers the same combination.
