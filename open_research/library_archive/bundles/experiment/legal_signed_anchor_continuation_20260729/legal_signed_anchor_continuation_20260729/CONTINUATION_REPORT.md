# Legal signed-anchor continuation report

**Date:** 2026-07-29  
**Terminal state:** **FAIL promotion gate; preserve only a weak analytic fallback and close the tested sign/abstention families.**

## Executive result

This continuation tested thirteen materially different legal constructions after the structured-pilot recurrence was closed. None met the required candidate/base development gate of 0.75, much less the promotion target of 0.595, on a frozen transferable block with safe tails.

The strongest repeatable candidate was an **internally centered covariance-defect transport with no later source injection**. A constrained ensemble collapsed to this single state. On eight fresh networks it scored **0.9386845**, won **5/8**, and had worst ratio **1.1019004**. This is a real legal signal but far too small to promote.

The most important negative result is stronger than “pilots are noisy”: several confidence mechanisms selected rare, huge source corrections on development or a first holdout, then failed on larger untouched cohorts. The shared failure is **joint bias**—analytic source and pilot/control-variate estimates can agree while both point in the wrong final-output direction.

## Frozen result table

| Branch | Algebraic distinction | Frozen result | Verdict |
|---|---|---:|---|
| 02 Centered analytic closures | Propagate Gaussian–Kerdock covariance defects internally; no final absolute subtraction | centered recurrence validation **0.9927**, 4/6, worst 1.149 | Close tested closure |
| 03 Source-blend transport | Separate inherited defect transport from later marginal source injection | safety package validation **0.9865**, 4/6, worst 1.071 | Weak signal only |
| 04 Weak analytic ensemble | Final-output constrained ensemble of source strengths | validation **0.9387**, 5/8, worst 1.102 | Best repeatable candidate; below gate |
| 05 Checkpoint reanchoring | Reset absolute analytic state at layers 4–28, then transport centered defect | every late reset selected no correction on 3 smoke networks | Close absolute checkpoint resets |
| 06 Analytic + residual pilot | Pilot estimates only residual around analytic anchor | gated validation **0.9937**, 4/8, worst 1.135 | Close residual-pilot form |
| 07 q128 adjoint sources | Contract first-layer exact defect and later marginal sources directly into 128 frozen probes | first-layer-only tuning **0.9589**, 6/6; later source unstable | Retain infrastructure; source closed |
| 08 Dual analytic ensemble | Combine full-covariance transport and q128 first-layer contraction | safe LOOCV shrank coefficients nearly to zero, **0.9993** | Redundant weak anchors |
| 09 q128 component sources | Separate mean and variance/pair source terms | best safe tuning **0.9394**, 3/6, worst 1.122 | Mean source weak; variance neutral |
| 10 Affine-CV anchor pilot | Frozen-gate affine target surrogate with exact Gaussian moments; pilot estimates anchor residual | safe optimizer chose gamma 0; **0.9690** | CV insufficient for anchor |
| 11 Pilot-phased q128 | Two affine-CV anchor pilots phase high-upside adjoint source | tuning **0.7055** → validation **1.1956**, worst 3.198 | Decisive reversal |
| 12 Direct-output affine CV | Estimate final Gaussian output directly as K mean plus nonlinear residual pilot | safe tuning **0.9748** | Cheap but weak standalone |
| 13 Output-benefit phased q128 | Direct-output pilots estimate quadratic benefit of adding source | tuning **0.4564** → validation **1.1413**, worst 2.791 | Rare-event selector failed |
| 13 cross-scale rescue | Require 1,024-row agreement and positive 2,048-row confirmation | first holdout **0.8119**, worst 1.076; expanded N=24 **1.1362**, worst 2.503 | Close external-pilot phasing |
| 14 Whole-basis fold stability | Estimate source independently on six complete-design basis folds | fold cosine >0.997 while full source was 51×–182× worse | Same-design agreement is invalid confidence |

## Strongest live artifact

The only candidate worth retaining as a baseline feature is the transport-only analytic state from `04_weak_anchor_ensemble`:

- validation candidate/base: **0.9386844971**;
- wins: **5/8**;
- worst: **1.1019004184**;
- no pilot trajectories;
- projected arithmetic is comfortably below the 14B cap, but no official FlopScope certificate was produced.

It should **not** be shipped or opened on a protected holdout. Its value is as a weak analytic residual baseline for a future fundamentally different sign source.

## Why the promising selectors failed

### Pilot-phased q128

The first pilot confidence rule scored 0.706 on six development networks by applying one large correction. On twelve untouched networks it again applied once, but that application produced a 3.20× tail and aggregate 1.196.

### Direct-output quadratic benefit

Replacing anchor cosine with an estimated final-output quadratic benefit improved the logic but not transfer. The frozen selector scored 0.456 on development, then 1.141 on twelve untouched networks. A cross-scale confirmation rescue initially scored 0.812 with worst 1.076 on a new twelve-network block, but failed on a larger 24-network block at 1.136 with worst 2.503. All three applications in that expansion were harmful.

### Fold stability

Fold stability is not merely weak; it is positively misleading. On three networks, each of six whole-basis source estimates had cosine above 0.997 with the complete source, while the complete source scored between 51× and 182× baseline. Complete Kerdock folds share the same Gaussianization bias.

## Compute accounting

These are projected arithmetic counts, not official FlopScope measurements:

- full covariance centered transport: approximately **1.95B** dense covariance-propagation FLOPs plus small marginal work;
- affine map composition: approximately **1.07B** FLOPs;
- 3,072 full-depth pilot rows: approximately **8.17B** baseline-proportional effective compute;
- q128 adjoint contractions and reductions: projected below roughly 1–2B depending on reuse.

Thus the cross-scale pilot package was plausibly below 14B, but it failed statistically. The transport-only candidate is cheaper still.

## Scientific conclusions

1. **Internal centering helps but is insufficient.** It converts catastrophic absolute closure into a small repeatable signal, not a winning anchor.
2. **Later Gaussian marginal sources are high-amplitude and phase-unstable.** They create rare 0.17–0.41 ratios and equally severe tails.
3. **Pilot agreement is not trustworthy when pilots share the same surrogate bias.** Larger pilots and cross-scale confirmation reduce but do not eliminate false positives.
4. **Same-design fold stability cannot estimate external Gaussian truth.** Complete blocks agree on the wrong answer.
5. **Final-output scoring remains essential.** Anchor cosine and local projection diagnostics repeatedly disagreed with complete-control benefit.

## Decision

Close the tested forms of:

- absolute or checkpointed bivariate-Gaussian closure;
- marginal-source q128 recurrence as a standalone correction;
- independent residual pilots around analytic anchors;
- affine-control-variate pilots as sign or benefit gates;
- cross-scale pilot confirmation;
- whole-basis fold stability and same-design abstention.

Retain:

- exact first-layer and q128 adjoint contraction code as infrastructure;
- transport-only internally centered analytic anchor as a weak legal baseline;
- all frozen per-network records as labels for a genuinely different external sign source.

No protected or official holdout was opened. Network IDs 4000–4262 used here are now exposed.
