# Path 5 Addendum — Early-Layer Downstream-Gramian Selector

**Date:** 2026-07-29  
**Status:** **Fail development gate; close this tested representation.**  
**New untouched holdout:** not opened.  
**Protected official/Mini holdouts:** not opened.

## Why this experiment was justified

The earlier Path 5 round left one material direction open: use weight-product or downstream-sensitivity subspaces to identify a precomputed support **before layer 29**. This experiment selects after layer 2, so a successful rule could retain almost all of the coreset compute benefit.

It is algebraically distinct from the failed final-coordinate sketches:

1. Propagate all 66,048 rows through only the first two layers.
2. Construct the downstream Gramian induced by weights 3–32 under the frozen mean-gate linearization.
3. Project complete layer-2 set states onto Gramian eigenmodes.
4. Form separate whole-support diagnostics from layer-2 activations and the ReLU residual `ReLU(z)-0.5z`.
5. Rank the frozen eight-support portfolio using global discrepancy, bounded-weight feasibility proxies, consensus, or a candidate-level ranker.
6. If selected, propagate only 8,192 rows through the remaining 30 layers.

No runtime NNLS, exchange, herding, or support optimizer is used.

## Data protocol

- Grouped training: 32 exact-geometry width-256 networks, seeds 64300–64331.
- Frozen development: eight exact-geometry networks, seeds 64000–64007.
- Portfolio IDs: `[81, 111, 88, 35, 91, 78, 51, 38]`.
- Exact same-support positive bounded-weight labels were reused from the frozen Path 5 corpus.
- The development block was not used to choose direct-score hyperparameters, consensus membership, learned-model regularization, or fallback thresholds.

The top-eight oracle ceiling remained strong: 30/32 training networks and 8/8 development networks contained a primary-gate support.

## Results

### Frozen direct rule

The training-selected direct rule was `gram_a_q16_global`.

- Training: **13/32** primary passes; worst `4.123e-07`.
- Development: **6/8** primary passes, **7/8** secondary passes; mean `1.113e-08`; worst `6.701e-08`.

This is modestly better than random support choice, but far from safe.

### Consensus

The training-selected ten-rule vote improved training fit but did not improve the development gate:

- Training: **16/32** primary passes; worst `2.917e-07`.
- Development: **6/8** primary passes; worst `1.430e-07`.

### Learned ranker

The grouped-CV-selected linear pass ranker also failed:

- Grouped training CV: **13/32** primary passes; worst `2.387e-07`.
- Development: **3/8** primary passes; worst `1.892e-07`.

### Fallback / abstention

A training-frozen consensus-margin fallback improved training to **17/32**, but development regressed to **5/8** with worst `1.934e-07`. The confidence statistic is therefore not a legal safety certificate.

## Compute interpretation

Ignoring the comparatively smaller Gramian and support-scan overhead, the candidate would execute one full-row dense layer and 30 selected-row dense layers instead of 31 full-row dense layers. With 8,192/66,048 rows retained, this is about **15.23%** of the baseline dense propagation work, or **84.77% gross dense-propagation savings**.

The compute upside is excellent. The statistical selector gate fails decisively, so no subprocess implementation or fresh holdout is justified.

## Verdict

**Close this exact representation:**

- layer-2 downstream-Gramian eigenmode discrepancy;
- Gramian-diagonal coordinate controls;
- layer-2 ReLU-residual set diagnostics;
- neighboring fixed ridge scales and mode counts tested here;
- consensus voting across those diagnostics;
- the tested linear candidate ranker;
- confidence-margin fallback to a universal support.

This strengthens the central Path 5 conclusion: even very early, downstream-weighted whole-set summaries do not reliably identify which precomputed support preserves the high-dimensional nonlinear cubature phase.

The only remaining Path 5 experiments I would consider are qualitatively larger commitments:

1. **Network-invariant-indexed support codebooks:** learn or quantize support families directly from weight-derived Grassmannian invariants, with at least 128 new grouped training networks and a frozen 32-network holdout.
2. **Jointly designed support library and selector:** offline differentiable codebook learning where support construction and weight-only support-error prediction are trained together, while runtime remains a closed lookup.

Neither is a small neighboring experiment; both require a new corpus and preregistration.
