# Oracle / Circularity Correction and Constructive Continuation

**Date:** 2026-07-30  
**Purpose:** Replace the invalid universal observability-gap story with a claim hierarchy and research program that matches the archived evidence.

## Executive correction

The oracle story remains central, but it supports a **mechanism and research-target claim**, not a universal impossibility theorem.

The challenge output is post-ReLU. Even when the limiting preactivation field is Gaussian, applying ReLU produces a non-Gaussian observed field. Therefore:

- Gaussian Bayes-linearity does not establish that the linear Kerdock estimator is Bayes-optimal among nonlinear algorithms.
- Gaussian no-adaptation theorems do not rule out adaptive or nonlinear point-evaluation algorithms for the actual output process.
- Failure of a finite collection of learners or correction families cannot upper-bound the supremum over all runtime-observable nonlinear corrections.

An explicit nonlinear ReLU example already demonstrates the logical gap: antipodal observations on one orthonormal basis reveal the coordinate absolute values, and nonlinear aggregation recovers the exact spherical integral of a ReLU ridge family while equal-weight linear aggregation does not.

## Strongest defensible integrated thesis

> The complete Kerdock rule nearly exhausts the fixed, network-independent, nonnegative linear cubature class for the specified infinite-width kernel. In realized width-256 networks, however, a large residual channel remains: oracle replacement of the layer-31 post-ReLU mean removes most final-output MSE. Exact correction-risk, downstream replacement, common-bias, and ReLU-crossing results characterize what a deployable repair must achieve. Several legal finite-width corrections exhibit real average signed alignment, but the tested implementations fail deployment because of tails, unstable phase, incomplete oracle capture, or compute. These failures close their named information classes, not the full nonlinear or adaptive problem.

## What the oracle evidence establishes

### 1. Late-layer repairability is large and replicated

The archived screen ladder reports:

| Layer | MSE removed |
|---:|---:|
| 1 | 13.83% |
| 4 | 13.22% |
| 8 | 22.54% |
| 12 | 40.20% |
| 16 | 48.96% |
| 20 | 53.55% |
| 24 | 62.69% |
| 28 | 75.45% |
| 29 | 77.83% |
| 30 | 79.97% |
| 31 | **82.69%** |

The broader archived evaluation reports:

- screen: 5.777x improvement;
- 24-network validation: 4.233x;
- 64-network holdout: 4.572x;
- holdout noise-corrected MSE removed: 78.13%;
- holdout wins: 64/64;
- layer-31/final-error linear CKA: 0.9828.

This localizes a highly repairable representation. It does **not** prove that the error originates at layer 31; upstream defects may accumulate and become compressed into a late mean channel.

### 2. The valuable object is downstream signed error

For a correction \(d\) applied with scalar \(\alpha\),

\[
R(\alpha)=R_0-2\alpha\,\mathbb E\langle e,d\rangle+\alpha^2\mathbb E\|d\|^2.
\]

The decisive quantity is signed alignment with the scored error, not correction magnitude or internal consistency.

For layer replacement with true defect \(d\), anchor error \(\xi\), and downstream linearization \(J\), the invariant first-order gate is

\[
\mathbb E\|J\xi\|^2 < \mathbb E\|Jd\|^2,
\]

with the general formula also containing relevant cross-terms. There is no universal gate stated only in unweighted layer-31 relative mean error.

The final ReLU adds a nonlinear remainder concentrated on particles whose preactivation margin is crossed. Consequently, two equal-norm anchor errors can have very different scored effects.

### 3. Same-design disagreement has a scoped identifiability limitation

Under the explicit observation model

\[
Z_i=\mu+b+\varepsilon_i,
\]

centered folds identify replicate noise but not the shared offset \(b\). This rigorously explains why folds, jackknives, rotations, and split halves can agree while retaining common absolute bias.

This is not universal. Unequal known bias loadings, bias-dependent noise, structural constraints, or an external reference can restore identifiability. The theorem closes the equal-loading/common-bias information model, not every possible runtime observable.

### 4. Legal correction signal is empirically nonzero

The archived stratified audit reports:

| Family | Approx. correction cosine | Outcome |
|---|---:|---|
| Compact companion | 0.400 | Real signal; tail and cost problems |
| Full companion | 0.490 | Stronger signal; heavy cost and bad worst case |
| Frozen radial-Hermite | 0.612 | Raw ratio 0.729; worst 1.583; incomplete oracle capture |
| T4 frozen policy | 0.0969 | Raw ratio 1.127854; harmful overall |

Therefore the correct empirical statement is:

> Some legal finite-width trajectory corrections carry real average signed signal, but current methods fail complete deployment because of variance, unstable phase, tails, incomplete oracle capture, or compute.

## Claims to retain, weaken, or remove

### Retain

1. The scoped static Kerdock near-optimality theorem.
2. Exact all-width uniform mass-one optimality on the complete Kerdock support among fixed linear weights.
3. The layer-31 oracle ladder as a replicated mechanism result.
4. The correction-risk identity and downstream-weighted replacement criterion.
5. Common-bias non-identifiability under its explicit observation model.
6. ReLU gate-crossing and nonlinear-margin bounds under their stated replay model.
7. Class-specific exact annihilation results for named low-degree and one-layer controls.
8. Frozen negative results for explicitly tested feature dictionaries, selectors, companions, and small harmonic controls.

### Weaken

- “Observability gap” should mean a **measured gap for tested information classes**, not a theorem over all legal algorithms.
- “Circularity” may be used as intuition for same-cloud self-anchoring, but the formal result must name the observation model.
- Scalar anchor-precision thresholds should be reported as experiment-specific and direction-dependent.
- Failed learning experiments should be written as closures of their feature, model, label, and grouping choices.

### Remove

- “The baseline is the exact Bayes rule for the post-ReLU challenge output.”
- “Adaptive or nonlinear evaluation cannot help at infinite width.”
- “All remaining headroom is an \(O(L/n)\) finite-width sector.”
- “A failed feature model upper-bounds all runtime-observable exploitability.”
- “The oracle headroom minus a fitted gamma is a proved observability gap.”
- “All legal corrections have zero alignment.”
- “No statistical or nonlinear path can exist.”

## Correct stopping statement

> **No active branch in the tested information classes clears a credible continuation gate under the current evidence, deadline, and resource constraints.**

This is a portfolio decision, not an impossibility theorem.

## Newly isolated evidence gap: cross-layer coherence

The blueprint proposed a cross-layer coherence matrix based on per-network signed increments between consecutive oracle swaps. The retained `LAYER_CHANNELS_SCREEN.csv` contains only aggregate statistics:

- aggregate candidate/base MSE ratio;
- aggregate MSE removed;
- confidence interval;
- wins;
- worst candidate/baseline ratio.

It does not preserve the per-network output-error vectors or signed layer-to-layer increments. Therefore the coherence matrix cannot be reconstructed from the published aggregate ladder.

### Required recovery artifact

For each base network, rotation, and layer \(k\), recover or regenerate:

- baseline final-output error vector;
- oracle-swap final-output error vector;
- exact reference mean and reference-noise estimate;
- signed correction vector;
- candidate/base squared-error contribution;
- layer-\(k\) mean defect;
- replay crossing fraction and nonlinear remainder where available.

Then compute:

1. covariance and correlation of consecutive signed correction increments;
2. cumulative versus incremental explained MSE;
3. cancellation and reinforcement terms;
4. principal modes shared across layers;
5. stability across base networks and rotations;
6. whether layer 31 is mainly a compressed sum of earlier defects or a genuinely late injection channel.

This is the highest-value missing diagnostic because it determines which upstream representations a constructive correction should target.

## Constructive research program

### Priority 0 — Recover the oracle coherence data

**Decision:** do before a new broad model search.

**Pass condition:** row-level oracle-swap data or a reproducible rerun package sufficient to compute the full covariance/cross-term matrix.

**Failure condition:** if the original rows are unrecoverable, document this and regenerate only on an explicitly development-only cohort. Do not treat aggregate ladder differences as signed source shares.

### Priority 1 — Downstream-weighted layer-31 residual synthesis

The next estimator should combine the real-signal families while targeting the scored channel directly.

#### Inputs

Use only legal runtime data:

- layer-31 Kerdock activation cloud;
- final-layer weights and preactivation margins;
- transported analytic/radial-Hermite mean correction;
- compact companion correction;
- fixed downstream-sensitivity basis derived from the realized final layer;
- optionally, a small frozen orientation codebook, but no target-based orientation choice.

#### Representation

Do not predict the full unweighted 256-vector defect. Predict coefficients in a low-dimensional downstream basis such as:

- leading right-singular directions of the local final-layer replay map;
- empirical final-error/PCA directions frozen from development networks;
- gate-aware directions separating high-margin and kink-sensitive particles.

#### Objective

Train or select using exact final-layer replay:

\[
\mathcal L =
\text{mean final MSE}
+\lambda_{\rm tail}\,\text{tail penalty}
+\lambda_{\rm norm}\,\|\widehat d\|^2
+\lambda_{\rm cross}\,\text{crossing penalty}.
\]

The loss should be grouped by base network, not by rotation row.

#### Controls

Every experiment must include:

- zero correction;
- fixed global shrinkage;
- analytic/radial-Hermite alone;
- compact companion alone;
- best frozen convex blend;
- downstream-basis constant coefficients;
- feature-dependent coefficients;
- oracle projection into the same basis.

This distinguishes value from the basis, the constant policy, and the learned network-specific component.

#### Preregistered continuation gate

Continue only if grouped validation shows all of:

- raw final-replay gain at least 1.10x;
- 95% interval entirely above 1.05x;
- worst-network candidate/baseline MSE at most 1.25;
- positive adjusted-score gain after complete compute accounting;
- no single base-network or rotation family drives the result;
- improvement over the best matched constant policy, not merely over zero correction.

This branch differs from failed simple selectors because it uses downstream-sensitive targets and exact nonlinear replay rather than generic geometry or correction agreement.

### Priority 2 — Bounded width-256 edge-state model

A single bounded learning experiment remains scientifically open:

- use public width-256 high-precision data;
- group all rotations by base network;
- use a full edge-state equivariant network with forward and backward states;
- expose Kerdock block summaries at layers 24, 28, 30, and 31;
- train through differentiable mean translation and final replay;
- predict low-dimensional downstream coefficients;
- use the same 1.10x / 1.05x-CI / 1.25-tail / compute-positive gate.

Do not run an open-ended architecture sweep. This is one preregistered representation test.

### Priority 3 — Rich exact-mean nonlinear controls

The harmonic audit leaves open nonpolynomial analytically integrable controls. A symmetrized Poisson kernel provides an explicit family with known mean and arbitrarily high even harmonic content.

A useful screen should:

- select a small fixed family before target inspection;
- evaluate the control residual on the existing Kerdock cloud;
- use exact known expectations;
- target the layer-31 downstream basis rather than an undifferentiated final scalar;
- compare against the frozen failed degree-6+8 dictionary;
- stop immediately if the exact-replay gain does not exceed 1.10x before compute.

This is higher risk than Priority 1 because high-degree control evaluation can be expensive and unstable.

## Closed branches that should not be repeated

- the nine-feature T4 ridge/tree dictionary;
- simple nested-convergence or fold-agreement safety rules;
- ranking orientation codebooks by two-basis norm or consensus;
- codebook expansion without a new predictor;
- the frozen four-direction degree-6+8 dictionary;
- unweighted full-vector layer-31 prediction;
- scalar universal precision gates;
- broad pooled “zero alignment” meta-tests;
- first-layer two-moment transport as a major path.

## Canonical conclusion for the paper

> Static nonnegative cubature for the specified limiting deep-ReLU kernel is essentially solved at the competition node budget, but finite-width nonlinear white-box estimation is not. Oracle replay identifies a large, stable, late-layer repair channel. Exact correction theory shows that successful deployment requires downstream signed alignment and control of nonlinear gate crossings. The tested corrections reveal real signal but do not jointly satisfy average gain, tail safety, oracle capture, and compute. We therefore close those named information classes under the present resources while leaving new nonlinear, adaptive, and externally anchored phase observables open.

## Immediate decision

1. Attempt recovery of row-level oracle-swap data and compute the cross-layer coherence matrix.
2. In parallel, specify—but do not yet validate—a single downstream-weighted layer-31 residual-synthesis experiment.
3. Do not open a new holdout until the representation, controls, grouped split, exact replay metric, tail gate, and compute accounting are frozen.
