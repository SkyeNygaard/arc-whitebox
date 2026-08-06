# WHestBench Agent Experiment Path

**Canonical state:** reconciled v8, July 29, 2026  
**Production baseline:** complete 129-basis partial-tree/Winograd Kerdock package, approximately 175.62B effective compute.  
**Common winning endpoint:** full or reduced Kerdock + direct K32 lower-order radial-Hermite correction + network-specific sign/scale + safe abstention.  

All experiments must preserve the evidence hierarchy and split rules in the canonical ledger. Global IDs 0–199 and every named cohort already used in prior reports are exposed.

# Path 2 — Learned sign, scale, and abstention

**Priority:** high  
**Workspace:** `paths/02_learned_sign_scale_abstention/`

## Goal

Learn only the part that legal analytic estimators fail to provide: network-specific correction sign, scale, anchor residual, and no-headroom risk.

A preferred output is

`(alpha, sign, p_harm, delta32)`

where `alpha` is shrinkage/scale, `sign` is the correction orientation, `p_harm` is harmful-tail probability, and `delta32` is the K32 anchor residual or signed correction.

## Broad model families

Agents may try:

- permutation-equivariant weight models;
- edge-state or message-passing networks;
- DeepSets over neurons, edges, layers, or basis-fold diagnostics;
- bidirectional layer encoders;
- analytic-anchor residual models;
- direct K32 correction-vector prediction;
- scalar sign and scale classifiers/regressors;
- no-headroom and predicted-score-benefit classifiers;
- uncertainty-calibrated ensembles;
- conformal or quantile harmful-tail estimates;
- K128-teacher to K32-student distillation;
- rotation-equivariant objectives and augmentation;
- multi-task learning for correction, scale, and abstention;
- mixture-of-experts conditioned on architecture or weight-derived invariants;
- learned combination of several frozen analytic anchors.

## Target restrictions

Do not predict the entire layer-31 mean, complete Kerdock residual, full 256-vector anchor, or arbitrary fixed coordinate support.

All rotations from one base network remain in the same fold. Training losses must include complete direct-output replay or a validated differentiable proxy tied to final MSE.

## Required baselines

- zero correction;
- best frozen analytic anchor;
- universal shrinkage alpha=0.40;
- oracle K32 correction;
- K128 teacher ceiling;
- analytic anchor plus learned residual;
- learned abstention applied to the analytic anchor.

## Safety tests

Explicitly include:

- hard rotations where the full oracle has no headroom;
- large correction norms;
- analytic/learned disagreement;
- base-network grouped bootstrap;
- calibration of `p_harm`;
- selective risk versus coverage curves.

Correction split agreement is not a safety certificate.

## Gates

Promotion requires the complete direct-control estimator to reach:

- candidate/base at most 0.595, preferably 0.537;
- lower confidence bound showing positive adjusted gain;
- worst at most 1.25 during screening and 1.10–1.15 for promotion;
- abstention materially improving the hard-rotation panel;
- inference and feature cost included in the final package.

One bounded width-256 model campaign is justified. Broad reduced-width architecture sweeps are not.
