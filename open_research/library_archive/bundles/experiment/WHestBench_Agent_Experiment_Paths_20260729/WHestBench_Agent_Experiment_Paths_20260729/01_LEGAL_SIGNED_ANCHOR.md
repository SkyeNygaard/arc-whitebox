# WHestBench Agent Experiment Path

**Canonical state:** reconciled v8, July 29, 2026  
**Production baseline:** complete 129-basis partial-tree/Winograd Kerdock package, approximately 175.62B effective compute.  
**Common winning endpoint:** full or reduced Kerdock + direct K32 lower-order radial-Hermite correction + network-specific sign/scale + safe abstention.  

All experiments must preserve the evidence hierarchy and split rules in the canonical ledger. Global IDs 0–199 and every named cohort already used in prior reports are exposed.

# Path 1 — Legal signed-anchor estimation

**Priority:** highest  
**Workspace:** `paths/01_legal_signed_anchor/`

## Goal

Estimate the frozen K32 lower-order radial-Hermite anchor accurately enough to retain most of the oracle direct-output gain under the real compute budget.

The frozen target contains only selected:

- target means;
- marginal second moments;
- row-direction pair moments.

Connected c21 is excluded unless its incremental cost is negligible and a frozen ablation proves value.

## Why this can win

Uniform independent oracle anchors reached approximately 0.403 candidate/base for K32 and 0.322 for K128. A legal K32 estimator retaining about 70% of the oracle reduction reaches the approximately 0.595 continuation threshold and has enough ceiling to beat the current production package.

## Broad method families

Agents may try:

- analytic Gaussian and non-Gaussian moment closures;
- jointly centered lower-order recurrences;
- exact or approximate Stein identities for selected contractions;
- shared-arithmetic contractions from existing FWHT/Winograd intermediates;
- downstream-weighted scalar Gramians;
- checkpoint-derived lower-order states;
- deterministic low-cost cubature specialized to the frozen directions;
- combinations of several weak analytic estimators;
- structured pilots only for the residual around an analytic estimate;
- q32/q64 scalar-state propagation;
- control variates for the anchor error;
- bias correction from weights, baseline outputs, basis folds, or package observables;
- compensated or higher-precision scalar accumulation;
- robust shrinkage derived from predicted estimator uncertainty.

## Forbidden repeats

- independent estimation of the absolute anchor with ordinary trajectories;
- one-basis or four-basis anchor pilots as the primary estimator;
- same-cloud fold anchors;
- full covariance or K3 construction;
- connected-only recurrences;
- layer-31 replay as the scored endpoint;
- judging by anchor cosine alone.

## Required comparisons

For each candidate report:

1. exact lower-order anchor;
2. sample-anchor null;
3. candidate anchor;
4. candidate plus frozen direct-output control;
5. candidate error split into means and pair moments;
6. oracle coefficients with candidate anchor versus candidate coefficients with exact anchor when coefficient estimation is changed;
7. total package cost at 129 bases.

## Gates

Development: complete candidate/base below 0.75.

Promotion:

- candidate/base at most 0.595, preferably 0.537;
- positive adjusted score;
- at least 75% wins;
- worst approximately at most 1.10–1.15;
- complete incremental compute below 14B, preferred below 10B;
- no hard-rotation catastrophe.

## Successful handoff

A passing artifact must expose a callable function that maps network weights plus allowed baseline observables to the frozen K32 anchor or correction, with exact cost accounting and no reference data.
