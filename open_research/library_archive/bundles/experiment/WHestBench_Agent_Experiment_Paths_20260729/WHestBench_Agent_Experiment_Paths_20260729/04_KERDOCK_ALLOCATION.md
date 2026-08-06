# WHestBench Agent Experiment Path

**Canonical state:** reconciled v8, July 29, 2026  
**Production baseline:** complete 129-basis partial-tree/Winograd Kerdock package, approximately 175.62B effective compute.  
**Common winning endpoint:** full or reduced Kerdock + direct K32 lower-order radial-Hermite correction + network-specific sign/scale + safe abstention.  

All experiments must preserve the evidence hierarchy and split rules in the canonical ledger. Global IDs 0–199 and every named cohort already used in prior reports are exposed.

# Path 4 — Kerdock allocation and adaptive basis count

**Priority:** conditional  
**Workspace:** `paths/04_kerdock_allocation/`

## Dependency

A fresh scored production test begins only after a legal anchor passes at all 129 bases. Offline oracle allocation and implementation work may proceed earlier.

## Goal

Remove enough Kerdock propagation to pay for the legal anchor without losing the correction’s statistical gain or tail safety.

The first production candidate is 112 bases, not 64.

## Known oracle frontier

The frozen K32 oracle correction remains compatible with aggressive thinning. Aggregate adjusted oracle score favored fewer bases, but 64–96 bases had unsafe tails. The 112-basis point preserved nearly the complete corrected MSE, saved about 13% of propagation, and had the best conservative tail profile.

## Broad method families

Agents may try:

- fixed 112-basis package;
- alternative nested basis orderings;
- 96-, 80-, and 64-basis frontiers;
- complementary-geometry basis ordering;
- basis-group selection and orbit-balanced packages;
- basis-specific correction recalibration frozen from development;
- incremental 64→80→96→112→129 fallback;
- stopping based on legal-anchor uncertainty;
- predicted correction benefit or no-headroom probability;
- K32/K128 agreement when the extra teacher cost is justified;
- basis-fold stability and basis-group disagreement;
- worst-tail-aware fixed allocation;
- target-free network-specific basis ordering;
- shared computation between incremental basis packages.

## Prohibited shortcut

Do not exchange removed Kerdock rows one-for-one for ordinary independent pilot trajectories. A point propagated to layer 29 costs nearly as much as a full Kerdock point and has poor anchor signal-to-noise.

## Required comparisons

At 112 bases compare:

- uncorrected 112 versus production 129 baseline;
- legal anchor at 112 versus the same legal anchor at 129;
- complete adjusted score and tails;
- exact measured compute and residual wall;
- basis-order sensitivity.

## Gate

The 112-basis complete package must beat both the production 129-basis baseline and the legal-anchor 129-basis package, with worst approximately at most 1.10–1.15.

Adaptive fallback begins only after fixed 112 is safe. Confidence features are valid only when they predict final benefit on a frozen cohort.
