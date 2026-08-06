# WHestBench current state — v26 post-agent audit

**As of:** 2026-07-30 18:50 ET  
**Competition objective:** maximize the probability of materially improving the scored submission, with an explicit focus on an **80% score reduction** where possible.  
**Protected or official data opened in the reviewed continuation reports:** **No.**

## Executive verdict

An **80% reduction in the current score**—an adjusted candidate/current ratio of at most **0.20**—is **not mathematically ruled out** by the present research. The reason is a genuine change in state: there is now a legal, target-free, gauge-invariant, zero-replay correction source with enough oracle capacity to support such a result.

There is still **no deployable candidate**. The source problem has been substantially solved, but the unknown **absolute signed contraction vector** has not. Every tested generic phase observable, feature learner, same-design residual method, independent pilot, and existing exact-control implementation fails the complete score economics.

The competition program should therefore be narrowed to **two bounded diagnostics**:

1. **Leading path:** freeze the adaptive direct-output PCA source and solve the complete source-specific checkpoint-gauge convex program, including a primal/dual certificate, untouched covariance validation, grouped tails, bias, and all compute.
2. **Secondary path:** reconstruct and audit the 90-column conic master dictionary on all 48 exposed cases, then measure its direct/no-fit physical covariance and score-optimal retained subspace.

Everything else is proof work, a stop-certificate, or closed.

## What changed relative to canonical v25

Canonical v25 was stale in three material ways.

First, it treated a literal rank-4/rank-5 late-interface source as the leading constructive target. Broad independent source tournaments now show that this inference was wrong. The previously observed four-to-five-dimensional object is a **target-informed repair-energy rank**, not a legal target-free source-coordinate rank. Natural legal rank-4/rank-5 sources leave roughly 40% of the error on confirmation and cannot win even with exact free coefficients.

Second, v25 described the source itself as missing. Agent 8 found a stronger direct-output construction from the 129 group-resolved output means already produced by the baseline. A development-frozen cumulative-energy rule selects about 34–38 modes and reaches:

- development: pooled `0.0633`, worst `0.2129`;
- validation: pooled `0.0782`, worst `0.2260`;
- confirmation: pooled `0.0749`, worst `0.1830`.

It is target-free, invariant to hidden-neuron permutations and positive ReLU rescalings, adds no network evaluations, and requires no nonlinear replay.

Third, the exact 30-column conic source is no longer a leading candidate. Its independent confirmation extension has pooled oracle ratio `0.156212`, median case ratio `0.239277`, and worst ratio `0.311349`; only 5 of 12 cases pass the zero-cost competition gate. Its independent exact-control pilots are still at least `27.4×` over the allowed variance-cost budget. The conic source remains useful as a mechanism and theorem benchmark, but the exact A30 candidate and its tested sampled-control descendants are closed.

## Is an 80% score reduction possible?

Interpret an 80% improvement as

\[
\frac{\text{adjusted candidate score}}{\text{current score}} \leq 0.20.
\]

For a frozen source with oracle residual ratio \(r_*\), fixed cost multiplier \(c_0\), and optimistic shared root contraction difficulty \(S\), the relevant continuous frontier is

\[
J_*=\left(\sqrt{c_0r_*}+S\right)^2.
\]

For the adaptive direct-output source, using \(r_*=0.0749\) and \(c_0\approx1\),

\[
S < \sqrt{0.20}-\sqrt{0.0749}
  \approx 0.173535,
\]

or equivalently the optimistic joint noise-cost budget is

\[
S^2 < 0.030114.
\]

This is demanding but not absurdly small. At the recorded 4.34× competition threshold, the corresponding squared allowance is about `0.042575`.

The source therefore leaves genuine room for an estimator. The problem is that no measured estimator is currently inside that room. Generic checkpoint-gauge screens usually collapse to direct estimation; the tested direct-output phase observables fail out of network; and ordinary full-pass coefficient samples are far too expensive and noisy.

### Other source frontiers

| Source | Evidence | Pooled \(r_*\) | Worst \(r_*\) | 80% optimistic noise-cost ceiling | Status |
|---|---:|---:|---:|---:|---|
| Adaptive direct-output PCA | 36 frozen cases | 0.0749 | 0.1830 | 0.030114 | Lead |
| Fixed direct-output rank 40 | 36 frozen cases | 0.067136 | 0.1613 | 0.035384 | Strong alternate |
| Fixed direct-output rank 32 | 36 frozen cases | 0.081362 | 0.213839 | 0.026235 | Lower-dimensional alternate |
| Conic A90 master | 12-case provisional | 0.02292 | unknown | 0.085490 after dense-ray cost | Secondary, unverified |
| Conic A30 | 12 confirmation cases | 0.156212 | 0.311349 | 0.001850 after dense-ray cost | Stop |
| Hidden nonlinear secant rank 32 | 36 frozen cases | 0.081667 | about 0.221 | 0.025355 after replay cost | Quarantine |
| Static arbitrary-node signed rule | theorem | at least 0.937061 | same | 0 | Closed |

## Ranked paths that could move the competition score

### 1. Adaptive direct-output PCA plus source-specific checkpoint-gauge SOCP

This is the lead.

The source has passed the capacity and tail gate across frozen development, validation, and confirmation splits. It also eliminates two former complications: source construction adds no network trajectories, and replay is exactly linear in output space.

Agent 2 proves that arbitrary checkpoint-control telescopes form an exact family and that the globally optimal linear estimator for a fixed source and checkpoint family is a convex second-order-cone program. Direct estimation is included as a feasible point. This makes the next test bounded and conclusive rather than another open-ended feature search.

The required result is not a regression metric. It is:

- a frozen direct-output source;
- full joint bias and covariance in physical coordinates;
- complete compute, including fitting or gating;
- primal and dual SOCP values;
- untouched covariance validation;
- grouped whole-network tails;
- adjusted ratio at or below `0.20` with material slack.

A pass would be the first mathematically justified candidate with 80%-scale potential. A certified fail would close the broad declared linear weight-aware checkpoint branch.

### 2. Full-48 conic A90 master-dictionary audit

The 90-column multiresolution construction is not simply “three times the 30-source cost.” It uses the same 1,920 source rays, contains the A30 span exactly, and permits a frozen lower-dimensional physical subspace to be sampled directly.

The provisional 12-case pooled oracle ratio `0.02292` is the largest source-capacity headroom currently visible. It is not promotion evidence because:

- full-48 capacity and tails have not been reconstructed;
- the numerical physical rank and cutoff stability are unarchived;
- direct and exact-control physical covariance is unknown;
- all per-network fitting, variance gating, signal gating, and estimation trajectories must be charged;
- the negative A30 results are a hostile prior.

The decisive experiment is to rebuild A90 on all 48 exposed cases, verify A30 inclusion, archive spectra, and select retained directions by the top eigenspace of \(S-N/n\), not by an arbitrary Gram condition threshold.

### 3. Fixed Walsh/direct-output contraction algebra

A fixed Walsh-character source has genuine capacity: rank 24 reaches confirmation `0.18546` and rank 32 reaches `0.13552`. It is weaker than adaptive direct-output PCA but has a simpler shared algebraic definition.

This path only makes sense as raw mathematics: exact character cancellations, harmonic alias identities, or a shared vector estimator unavailable to adaptive SVD coordinates. The tested odd references, Walsh summaries, and complete companion rotations do not work. Do not train another generic predictor.

### 4. Weight-aware information or phase-obstruction theorem

The finite-design blind-spot theorem proves that the 129 group-output transcript alone cannot universally determine the Gaussian target over homogeneous ReLU networks. It does **not** close weight-aware identities or random-network probabilistic estimators.

If the direct-source SOCP fails, the highest-value continuation is a quantitative lower bound for the larger legal information set: construct matched networks or rotations with nearly identical legal weights/checkpoint summaries and opposite contractions, or certify a conditional-variance floor above the source allowance. This would turn repeated empirical failure into a defensible stop theorem.

### 5. Radical compute reduction

With unchanged MSE, an 80% score reduction requires effective compute to fall to at most 20% of the current level—a true 5× reduction under official accounting. No current compiler or replay evidence is close. This remains a low-probability engineering branch and should be revived only by exact-output official subprocess measurements.

## Paths to stop

Stop competition effort on:

- static node, weight, mass, sign-count, or higher-degree tuning inside the proved static class;
- natural rank-4/rank-5 late-interface source searches;
- rank-12 boundary sources;
- the exact conic A30 source as a robust candidate;
- independent Gaussian, spherical, antithetic, orthogonal, QMC, or full-Kerdock coefficient pilots without a new order-of-magnitude variance identity;
- conic fan self-controls as a rescue for A30;
- same-design residual regression;
- full first-layer and low-degree pilot controls;
- three- and five-piece maxout/DC descendants;
- generic coefficient or phase predictors over the tested feature dictionaries;
- universal exact formulas using only the finite group-output transcript;
- protected-data promotion.

## Up-to-date status of the markdown corpus

The individual agent reports should remain immutable evidence artifacts. They are not all mutually current as recommendations because later reports supersede earlier optimism.

The canonical synthesis now treats them as follows:

- **Current positive source evidence:** Agent 8 direct-output source.
- **Current replay/information synthesis:** Agent 4 v3.
- **Current bounded contraction theorem:** Agent 2 checkpoint gauges.
- **Current source geometry and phase diagnosis:** Agent 5 continuation and Agent 1 tournament.
- **Current decisive negative for conic A30:** Agent 7 final continuation.
- **Current provisional conic A90 accounting:** Agent 6 v3.
- **Preserved theorem but superseded deployment optimism:** Agent 3 conic self-controls.

The v26 ledger and this document are the canonical recommendation layer. Historical files remain useful for proof provenance, exact experiment definitions, and reproduction.

## Immediate execution order

1. Run the adaptive direct-output source-specific checkpoint-gauge SOCP.
2. In parallel, reconstruct full-48 A90 and run only the direct/no-fit covariance gate.
3. Pursue fixed-Walsh or direct-output exact identities as math work, not learning.
4. If the lead SOCP fails, prioritize a quantitative weight-aware phase/conditional-variance obstruction.
5. Keep all protected data sealed until a complete, frozen estimator passes capacity, observability, economics, implementation, and grouped-validation gates.

## Bottom line

There is a path to move up substantially, and an 80% score reduction is still possible in principle. It is **not** a matter of further local optimization. The path requires a breakthrough in absolute signed observability for a source whose capacity is already adequate.

The best allocation is one conclusive direct-source SOCP audit, one provisional A90 covariance audit, and otherwise mathematics aimed either at an exact shared contraction identity or a stop theorem.
