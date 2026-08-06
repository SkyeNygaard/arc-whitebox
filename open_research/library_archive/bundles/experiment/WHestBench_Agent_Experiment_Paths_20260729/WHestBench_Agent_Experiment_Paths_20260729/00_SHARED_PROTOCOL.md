# WHestBench Agent Experiment Path

**Canonical state:** reconciled v8, July 29, 2026  
**Production baseline:** complete 129-basis partial-tree/Winograd Kerdock package, approximately 175.62B effective compute.  
**Common winning endpoint:** full or reduced Kerdock + direct K32 lower-order radial-Hermite correction + network-specific sign/scale + safe abstention.  

All experiments must preserve the evidence hierarchy and split rules in the canonical ledger. Global IDs 0–199 and every named cohort already used in prior reports are exposed.

## Shared protocol

### Operating rule

An experiment is justified only when a passing result can be converted into a complete, costed executable in the next one or two experiments. Correlation, cosine, rank, support overlap, anchor error, or a standalone surrogate score is never a continuation gate unless the path brief explicitly says so.

### Stage 0 — Algebra, implementation, and cost

Before a large run, record:

- exact runtime algorithm and final-output identity;
- complete tracked-FLOP estimate and expected residual-wall impact;
- every asset and precomputed table shipped with the executable;
- how the proposal differs from closed experiments;
- target candidate/base ratio if the mechanism succeeds;
- immutable development, validation, rotation, and holdout manifests.

Reject before implementation when the complete package cannot plausibly fit the budget or reach winning-scale adjusted score.

### Stage 1 — Exposed development

Use development-only networks. Continue only when the complete candidate:

- reaches candidate/base below 0.75;
- has positive signed correction alignment;
- shows no obvious catastrophic tail;
- has credible final-package compute accounting.

This stage may compare many variants, but the survivor must be frozen before new data are opened.

### Stage 2 — Frozen validation

Generate or open at least 24 fresh width-256/depth-32 networks only after freezing code, probes, coefficients, shrinkage, fallback, cost accounting, and reference protocol.

Primary promotion gate:

- candidate/base at most 0.595;
- preferably at most 0.537;
- at least 75% wins;
- worst approximately at most 1.10–1.15;
- adjusted-score confidence interval favorable;
- complete added compute below 14B unless a path-specific stricter gate applies.

### Stage 3 — Rotation and adversarial panel

Test multiple allowed Kerdock rotations, low-headroom cases, large correction norms, basis-order perturbations, and independent expectation splits. Any confidence, shrinkage, or abstention rule must already be frozen.

Agreement is not sufficient: prior harmful corrections sometimes had strong split agreement.

### Stage 4 — Final holdout and executable

Only a complete subprocess package reaches a 64+ network holdout. Report exact FlopScope, residual wall time, peak memory, raw MSE, adjusted score, wins, tails, all fallbacks, and prediction equivalence. No tuning after opening.

## Global closures

Do not spend experiments on:

- suffix compiler tuning or proxy arithmetic; all integrated compiler variants lost to production;
- universal radial-Hermite shrinkage constants, mode truncation, or layer-31 replay;
- ordinary independent anchor trajectories or one-for-one replacement of removed Kerdock rows;
- connected-c21 as the default radial-Hermite anchor target;
- fixed target-free layer-31 coordinate rankings;
- marginal Stein/H3 phase statistics;
- same-cloud anchors, stable-gate MLMC, ordinary partial-MUB pilots, or harmonic controls;
- K2, independent row scores, or independent pairwise coreset scores;
- runtime NNLS, herding, exchange, or other iterative coreset solvers.

## Required handoff artifacts

Each experiment directory must contain:

- `EXPERIMENT_SPEC.md` — frozen hypothesis, algorithm, gates, and split manifest;
- `COST_MODEL.json` — operations, wall-time assumptions, and package total;
- `RESULTS.json` — aggregate and per-network outputs;
- `ROWS.csv` — one row per network/rotation;
- `MANIFEST.json` — code, asset, probe, coefficient, and data hashes;
- `DECISION.md` — continue, redirect, or close with exact scope;
- complete runnable code or a narrow compatibility handoff.

## Agent allocation

- Path 1 — 30%
- Path 2 — 25%
- Path 3 — 15%
- Path 4 — 10%
- Path 5 — 10%
- Path 6 — 10%
