# WHestBench Agent Experiment Path

**Canonical state:** reconciled v8, July 29, 2026  
**Production baseline:** complete 129-basis partial-tree/Winograd Kerdock package, approximately 175.62B effective compute.  
**Common winning endpoint:** full or reduced Kerdock + direct K32 lower-order radial-Hermite correction + network-specific sign/scale + safe abstention.  

All experiments must preserve the evidence hierarchy and split rules in the canonical ledger. Global IDs 0–199 and every named cohort already used in prior reports are exposed.

# Path 3 — Internally centered defect transport

**Priority:** medium-high  
**Workspace:** `paths/03_centered_defect_transport/`

## Goal

Construct a recurrence whose intermediate states are themselves small Kerdock-versus-Gaussian lower-order defects. The method must perform cancellation internally, rather than approximate large absolute moments and subtract them at the end.

## Required novelty relative to M107

Every proposal must state algebraically how it differs from M107. Reducing 128 scalars to 32, changing terminology from anchor to defect, or using a different checkpoint depth is not sufficient.

A valid new method should make truncation or approximation errors proportional to the small defect state, or cancel shared large terms exactly before approximation.

## Broad method families

Agents may explore:

- layerwise propagation of `Delta_l` rather than `A_l`;
- coupled Gaussian/Kerdock recurrences with shared arithmetic;
- common-basis and common-randomness cancellation;
- exact zero-mean complete-design identities;
- Stein identities applied directly to the quadrature difference;
- source terms expressed as complete-basis defects;
- adjoint-weighted scalar defects;
- centered checkpoint states only;
- analytic source plus learned residual source;
- signed source sparsification with cancellation preservation;
- compensated summation and higher-precision states;
- basis-group telescoping;
- control-variate recurrences where the analytic and Kerdock terms share all expensive contractions.

## Mandatory diagnostic

Compare on the same exposed cohort:

1. exact frozen K32 defect;
2. M107 projected onto the K32 lower-order target;
3. new internally centered recurrence;
4. direct-output candidate/base;
5. per-layer signed source, cumulative defect, approximation error, and marginal cost.

## Gates

- development complete candidate/base below 0.75;
- promotion at most 0.595, preferably 0.537;
- total added compute below 14B;
- positive signed correction alignment and safe tails;
- materially better final MSE than projected M107, not merely better state cosine.

A failure closes the tested centered algebra, not all defect recurrences.
