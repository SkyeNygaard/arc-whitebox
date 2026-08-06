# K32 Signed-Anchor Sandbox Execution — Terminal Report

**Date:** 2026-07-29  
**Terminal state:** **FAIL, scoped to the two frozen learned estimators**  
**Production decision:** Preserve the unchanged partial-tree Kerdock baseline.

## Executive conclusion

This execution round completed every non-minor sandbox task that had a coherent frozen target:

1. The production compiler measurement finished. `adaptive_2_6` lost decisively at adjusted candidate/base **1.11956**, after `two_layer` and `fixed_three` had already lost. The production baseline remains final.
2. The direct-output sparse oracle program survived fresh validation. The active deployable-sized target is the **K32 lower-order signed anchor**, with K128 retained only as an oracle ceiling.
3. The four fixed-coordinate layer-31 selector families remain closed because their captured benefit was rotation dependent.
4. A new grouped width-256 probe-set learner was trained on 64 base networks and validated on 16, with complete rotations grouped by base network. It failed on a newly generated terminal block of 16 bases × 3 rotations.
5. One mechanistically directed rescue—a scalar sign/scale learner along a frozen K32 rank template—was then run on another independent 16-base × 3-rotation terminal block. It also failed.

The project remains **oracle-rich but estimator-poor**. The K32 correction is real and robust when its signed anchor is known; the tested weight/same-cloud learners do not recover its phase safely.

## Production compiler closure

| Candidate | Adjusted candidate/base | Decision |
|---|---:|---|
| two-layer | 1.027957 | Reject |
| fixed three-layer | 1.041620 | Reject |
| adaptive depth 2–6 | 1.119560 | Reject |

The adaptive candidate had zero wins on the official Mini-100 measurement and lost because its arithmetic and composition overhead exceeded the current partial-tree baseline's already-realized savings.

## Primary learned K32 anchor

### Frozen design

- 64 training base networks, rotations 3 and 11.
- 16 validation base networks, rotations 3, 11 and 97.
- Three-member probe-set equivariant DeepSets ensemble.
- Inputs were only weights and same-cloud Kerdock observables.
- Output was 32 lower-order signed anchor defects, never the 256-dimensional final answer.
- Candidate safety rule: ensemble-consistency shrinkage and abstention unless every member correction had cosine above 0.5 with the ensemble mean.
- The originally allocated test block 3080–3095 was quarantined after an oracle-label audit exposed it before model freeze. Terminal evidence used new IDs 3200–3215.

### Terminal result

| Metric | Result | Gate |
|---|---:|---:|
| Raw candidate/base | **1.094160** | ≤0.595 |
| Raw bootstrap 95% | **[1.004396, 1.219429]** | upper <1 after cost |
| Adjusted candidate/base | **1.108167** | <1 |
| Adjusted bootstrap 95% | **[1.017254, 1.235040]** | upper <1 |
| Wins | **18/48** | — |
| Worst | **2.619328** | ≤1.15 |
| Added FLOPs | **2.248B** | <14B |
| Exact oracle K32 ratio | **0.202132** | mechanism only |
| Final correction cosine | **-0.101621** | positive required |

The terminal anchor Pearson correlation was -0.117; anchor sign accuracy was 24.7%. The model's error is signed phase, not compute.

## Diagnostic ablation

The one permitted diagnostic separated direction, sign and scale:

| Diagnostic | Candidate/base |
|---|---:|
| Predicted direction with oracle scalar | 0.657522 |
| Predicted magnitudes with oracle token signs | 0.770685 |
| Exact direction with predicted norm | 0.679426 |
| Frozen rank template with oracle scalar | 0.590930 |
| Frozen rank template with train-median scalar | 1.007885 |

Useful span remained behind the failed phase prediction. This justified exactly one rescue: learn only a scalar sign/scale along a frozen rank template.

## Scalar sign/scale rescue

### Frozen design

- Mean K32 lower-order defect by selected rank, learned only from the original training split.
- Five-member invariant scalar MLP ensemble.
- Candidate applies the mean predicted scalar, shrunk by ensemble dispersion, only when all five signs agree.
- Terminal evidence used new IDs 3300–3315, three rotations each.

### Terminal result

| Metric | Result | Gate |
|---|---:|---:|
| Raw candidate/base | **0.982895** | ≤0.595 |
| Raw bootstrap 95% | **[0.931990, 1.037742]** | upper <1 after cost |
| Adjusted candidate/base | **0.995436** | <1 |
| Adjusted bootstrap 95% | **[0.943882, 1.050983]** | upper <1 |
| Wins | **26/48** | — |
| Worst | **1.333055** | ≤1.15 |
| Added FLOPs | **2.241B** | <14B |
| Oracle scalar on same template | **0.713922** | diagnostic |
| Scale Pearson | **0.476152** | positive but insufficient |
| Scale sign accuracy | **56.2%** | insufficient |

The rescue is nearly raw-neutral but not score-positive and remains tail-unsafe. It predicts a positive scalar on 93.8% of examples although the oracle scalar is positive on only 50.0%.

## Closure certificate

Closed now:

- this three-member probe-set anchor-vector learner;
- this five-member invariant scalar sign/scale rescue;
- further tuning of their depth, widths, thresholds, ensemble count, shrinkage or confidence rules;
- all compiler variants measured in the current production package.

Still active as mechanism evidence:

- the K32 lower-order direct-output oracle anchor;
- K128 as a robustness ceiling;
- the exact lower-order anchor algebra and crossfit infrastructure.

A future learned branch requires **qualitatively new information**, not a larger version of these models. The most defensible reopening would expose a rotation-conditioned, non-marginal statistic or exact/analytic state whose phase is independently identifiable before training. A new model should not begin until that input demonstrates positive signed correlation on a frozen cohort.

## Integrity

- Primary terminal model freeze: `30a8149355143c1eb4e84126962575da89478dab99d343c9bd6729d88207e19f`
- Rescue terminal freeze: `6ca1fb78fce4aa371e45cadff1769be1647e3a7d6108b0c468a00fbd3f71fe52`
- Primary terminal IDs: 3200–3215
- Rescue terminal IDs: 3300–3315
- All rotations of each base network remained in one split.
- No terminal-test tuning occurred.
- Exact oracle labels were used only for training targets, diagnostics and scoring; they are not runtime inputs.
