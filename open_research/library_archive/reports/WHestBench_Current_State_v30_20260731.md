# WHestBench Current State v30 — Local Hybrid and Mixture Results

**Audit cutoff:** 2026-07-31 08:49 ET  
**Canonical ledger:** `whestbench_canonical_research_ledger_20260731_reconciled_v30_local_hybrid_results.xlsx`  
**New protected data opened:** No

## Executive verdict

The local results materially update the v29 priorities.

There is still no deployable estimator. The strongest remaining program is still a coherent joint-distribution state, but the oracle ladder is now bracketed:

- a location-latent state at rank 64 appears accurate enough at the local closure gate but is currently implemented with uneconomic particles;
- a compact full-covariance mixture at K=32 captures a real copula effect but remains about 1.6× above the required closure error;
- the untested middle—tied covariance, shared low-rank covariance modulation, or deterministic compression of the passing latent state—is now the only primary representation program.

Two plausible hybrid estimators were honestly closed:

1. sampled layer moments followed by analytic Edgeworth mean propagation;
2. a smoothed-network analytic anchor plus complete-block residual sampling.

The work nevertheless produced one important positive structural result:

> Complete-block residual variance, not pointwise residual variance or ordinary R², is the correct gate for any design-aware hybrid.

A smoothed anchor reduced the blockwise residual variance by as much as roughly 25×. The hybrid failed because the anchor expectation could not be obtained cheaply enough, not because the residual remained large.

## What changed in the priority list

### M192 — still primary, but narrowed

The local results measured two endpoints of the mixture ladder.

#### Full covariance mixture

Reported sigma-closure errors:

| Layer | K=1 | K=8 | K=32 |
|---|---:|---:|---:|
| 16 | 1.641e-2 | 8.44e-3 | 5.33e-3 |
| 29 | 1.368e-2 | 6.08e-3 | 3.58e-3 |

The average over the tested layers was reported near `4.8e-3`, against a `3e-3` gate.

This is a real mechanism result: covariance modulation improves the Gaussian closure by roughly 3–4×. It is not enough at K=32, and naive K=64–128 full covariance propagation pressures or exceeds the compute floor.

#### Location-latent state

The local comparison reports:

- rank 32 at layer 29: approximately `2.37e-3`;
- rank 64 at layer 29: approximately `8.6e-4`;
- rank 64 average: approximately `1.96e-3`.

That passes the representation gate. But the present particle realization purchases the latent state through forward trajectories, so it does not provide a compact legal analytic recurrence.

The useful information is overwhelmingly joint: marginals reportedly recover only 5–11% of the improvement.

#### Canonical decision

Continue only:

- tied-covariance mixtures;
- shared low-rank covariance modulation;
- deterministic or analytic compression of the passing rank-64 latent state.

Do not continue simply increasing full-covariance K or particle count.

### M187 — fold into M192

The covariance-modulation half has now been run. The remaining sign–magnitude and weighted-kernel attribution remains useful, but only to choose the next M192 family.

M187 is no longer a separate candidate program.

### M188 — unchanged, but the legal gate is now more important

M188 remains conditional on finding a compact M192 state.

Passing an oracle closure gate does not show that the representation can be initialized, compressed and propagated through 32 layers without reference fitting or drift.

The rollout must have:

- analytic or otherwise legal initialization;
- fixed state dimension and compression;
- no per-layer oracle clustering;
- no particles whose cost restores the sampling law;
- complete PSD and realizability;
- raw MSE at most `2.962e-7`;
- complete effective compute within the relevant score regime.

### M190 — demoted as a standalone route

The corrected spectrum gives:

- residual energy above degree 5: approximately `0.399`;
- residual energy above degree 11: approximately `0.2532`.

Therefore, if M190 is interpreted only as exact integration of degrees 6–10, its maximum zero-cost gain is:

\[
0.399/0.2532 pprox 1.58.
\]

That cannot close the competition gap.

This does not close M190's broader role as an environment-weighted contraction or compression engine for M192/M188. It closes the standalone low-band interpretation.

### M189 — still relevant

The local work does not constrain Kerdock-index QTT rank or common-pivot structure.

M189 remains a cheap, low-prior existing-array falsifier.

### M193 — still relevant, but low prior

The local rank evidence is a warning, not an answer.

A participation ratio near 2.2 at late layers does not imply that the relevant geometry is genuinely low rank. The cited early-layer capture was only about:

- rank 32: 46%;
- rank 128: 93%.

More importantly, activation covariance rank is not output-weighted boundary-normal rank.

M193 should be run only as its declared boundary-normal and gate-current audit.

### M194 — not closed by the local static result

The local LORO/static rules converging to the uniform estimator do not test the proposed weight-coupled cubic boundary/Walsh bispectrum.

M194 remains one algebraically fixed test. It should not become a search campaign.

Promotion requires positive grouped correction covariance before any scalar fitting.

### M195 — tested subclass closed, broad class deferred

The local test used 29 weight/state-aware features over approximately 25,600 network-neuron observations with leave-one-network-out validation.

Reported results:

- \(R^2\) approximately `-0.018` to `-0.002`;
- sign performance at chance;
- MSE worse than the baseline target.

This decisively closes that handcrafted feature dictionary.

It does not close the full exact-quotient neural-operator class, because the complete weights still determine the answer in principle. But there is no justification for more feature expansion or a broad learning campaign.

## Hybrid results

### T104 — blockwise residual variance is the correct gate

For an anchor \(g\) with known expectation and a residual estimated over \(R\) complete Kerdock bases,

\[
\widehat\mu
=
E[g]
+
rac{1}{R}
\sum_{b\in\mathcal S} Q_b(f-g),
\]

the stochastic error is governed by

\[
rac{\operatorname{Var}_b(Q_b(f-g))}{R}.
\]

Pointwise residual variance is not the relevant quantity.

The local measurement exposed a large discrepancy:

- rank-128 anchor pointwise residual ratio: approximately `0.597`;
- complete-block residual ratio: approximately `0.949`.

This retroactively weakens the prior rank-MLMC interpretation.

### M197 — sampled-moment analytic-mean hybrid closed

Measured MSE:

| Rows | Direct | Gaussian hybrid | +κ₃ | +κ₃+κ₄ |
|---:|---:|---:|---:|---:|
| 4,096 | 3.21e-6 | 9.70e-5 | 5.34e-6 | 3.01e-6 |
| 16,384 | 6.43e-7 | 8.72e-5 | 3.76e-6 | 7.97e-7 |
| 66,048 | 1.68e-7 | 8.28e-5 | 3.71e-6 | 2.73e-7 |

The fourth-order marginal model is extremely accurate relative to Gaussian closure, but the estimator remains sampling-noise limited.

The reported chain variance constant was approximately `1.20e-2`, compared with approximately `1.11e-2` for direct averaging.

Sampling the moments from the same rows does not produce extra information.

### M198 — smoothed residual mechanism passed

The smoothed anchor produced genuine complete-block residual reductions:

- \(lpha=0.2\): \(S_r/S_fpprox0.0392\);
- \(lpha=0.25\): approximately `0.064`;
- \(lpha=0.5\): approximately `0.242–0.249`;
- \(lpha=1.0\): approximately `0.553`.

This is a real structural result. It shows that an anchor can preserve the Kerdock cancellation while substantially shrinking the difficult residual.

### M199 — honestly accounted smoothed hybrid closed

At matched cost:

| Residual blocks \(R\) | Direct using \(2R\) blocks | Hybrid α=0 | α=0.1 | α=0.2 | α=0.35 |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.278e-6 | 2.973e-6 | 2.940e-6 | 3.038e-6 | 4.744e-6 |
| 16 | 5.851e-7 | 1.630e-6 | 1.608e-6 | 1.713e-6 | 3.382e-6 |
| 64 | 1.839e-7 | 4.739e-7 | 4.655e-7 | 6.549e-7 | 2.464e-6 |

The hybrid was uniformly about 2–2.7× worse.

The earlier projected `10.02×` and `5687×` gains were accounting or objective-function artifacts and are explicitly quarantined.

### M200 — purely analytic smoothed anchor closed

Closure errors:

| Layer | α=0 | α=0.25 | α=0.5 | α=1 | α=2 |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.659e-2 | 1.668e-2 | 1.592e-2 | 1.204e-2 | 5.644e-3 |
| 29 | 1.360e-2 | 1.364e-2 | 1.309e-2 | 9.982e-3 | 5.268e-3 |

The useful residual regime requires mild smoothing, but mild smoothing does essentially nothing to the analytic covariance closure.

Strong smoothing improves closure by only about 2.6×, remains approximately eight times above the required error, and leaves a residual ratio around `0.805`.

The residual and tractability regimes do not overlap.

## Revised current order

1. **M192:** tied/shared-low-rank joint modulation or deterministic compression of the passing latent state.
2. **M188:** legal free rollout only if a compact M192 rung passes.
3. **M189:** one existing-array Kerdock-QTT falsifier.
4. **M194:** one algebraically fixed cubic boundary/Walsh kernel.
5. **M193:** one output-weighted boundary-normal and gate-current audit.
6. **M187:** remaining attribution folded into M192.
7. **M190:** internal contraction engine or proof lane only.
8. **M195:** no more feature dictionaries; full quotient learner deferred.
9. Keep protected evaluation sealed.

## Final state

The local priority analysis was directionally right about M192 and M190, but it overreached in three places:

- M193 was not answered by activation covariance ranks.
- M194 was not answered by static LORO.
- M195's 29-feature test does not close the complete full-weight quotient class.

The most important positive result is not a candidate estimator. It is the identification of the correct hybrid metric and a genuine residual-reduction mechanism.

The most important negative result is that two concrete hybrids fail because the anchor expectation costs as much as the answer.

The central live problem is now:

> Find a deterministic, compact, legally propagatable representation of the joint copula state that reaches the rank-64 latent oracle accuracy without particles, oracle clustering, or full K-dependent covariance transforms.
