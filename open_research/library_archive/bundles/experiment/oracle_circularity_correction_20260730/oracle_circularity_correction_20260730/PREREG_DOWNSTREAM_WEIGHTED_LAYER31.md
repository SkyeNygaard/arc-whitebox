# Preregistration: Downstream-Weighted Layer-31 Residual Synthesis

**Status:** Draft; freeze before opening any new validation or holdout data.  
**Date:** 2026-07-30

## Research question

Can a fixed, legal, low-dimensional synthesis of archived real-signal corrections produce a tail-safe and compute-positive estimate of the layer-31 mean defect when trained and evaluated through the exact final-layer replay?

## Hypothesis

The compact companion and analytic/radial-Hermite corrections contain complementary but incomplete signed information about the layer-31 downstream defect. A low-dimensional, downstream-sensitive combination may outperform each arm and the best constant blend, provided it is trained by base network and penalized for ReLU crossing and worst-network risk.

## Runtime information

Allowed:

- realized network weights;
- complete Kerdock trajectories already produced by the baseline;
- layer-31 activation cloud;
- final-layer preactivations and margins;
- frozen analytic/radial-Hermite correction;
- frozen compact companion correction;
- fixed downstream basis derivable from the realized final layer without target labels.

Disallowed:

- exact layer-31 mean;
- exact final mean;
- oracle coefficients or orientation labels;
- validation/holdout labels during representation selection;
- per-rotation leakage across grouped splits.

## Candidate basis

Freeze one of the following before labels are inspected:

1. top \(r\) right-singular directions of a smoothed local final-layer replay;
2. a fixed hybrid of singular directions and gate-margin groups;
3. a precomputed development-only error PCA basis transported equivariantly.

Use \(r\in\{4,8,16\}\), with one primary \(r\) selected from development-only evidence.

## Candidate policies

1. Zero correction.
2. Global scalar shrinkage of analytic correction.
3. Global scalar shrinkage of companion correction.
4. Frozen global convex blend.
5. Constant coefficients in the downstream basis.
6. Feature-dependent bounded coefficients.
7. Feature-dependent coefficients plus abstention.
8. Oracle projection into the same basis, for ceiling only.

## Legal features

Use a small frozen set:

- correction norms and mutual cosine;
- downstream-projected coefficients from each arm;
- final-layer singular values;
- preactivation margin histogram;
- predicted gate-crossing mass under each correction;
- basis-block dispersion summaries;
- base-network invariant weight summaries.

Do not reuse the failed nine-feature T4 dictionary as the sole representation.

## Training and grouping

- Group all rotations from the same base network.
- Use nested grouped cross-validation.
- Choose architecture and regularization entirely inside development folds.
- Compare every feature-dependent policy to the best matched constant policy.
- Freeze one policy before any new validation cohort.

## Primary metric

Exact candidate/base final-output MSE after applying the layer-31 correction and replaying the true final layer.

## Secondary diagnostics

- correction-risk inner product;
- correction cosine;
- downstream-weighted error \(\|J\xi\|^2\);
- exact nonlinear versus linearized replay error;
- crossing fraction and crossing remainder;
- wins;
- median, p90, and worst ratio;
- grouped bootstrap interval;
- complete FLOPs, wall time, and adjusted score.

## Continuation gate

All conditions must hold on grouped validation:

1. raw gain at least 1.10x;
2. 95% interval lower bound above 1.05x;
3. worst-network candidate/baseline ratio at most 1.25;
4. adjusted-score improvement remains positive after full generation and replay cost;
5. feature-dependent policy beats the best matched constant policy;
6. no one base network or rotation family explains the result;
7. crossing diagnostics show no hidden harmful subgroup.

## Stop conditions

Stop the branch if any of the following occurs:

- the best policy is statistically indistinguishable from a constant blend;
- tail safety requires abstaining on most networks;
- the oracle projection into the chosen basis captures less than 30% of layer-31 oracle gain;
- exact nonlinear replay erases the linearized gain;
- projected added compute makes the adjusted score nonpositive;
- grouped performance depends on rotation leakage or one network family.

## Interpretation

A pass demonstrates a deployable correction in this specific information class.

A failure closes only:

- the frozen two-arm source family;
- the frozen downstream basis;
- the frozen feature set and policy class;
- the tested compute regime.

It does not upper-bound all nonlinear or adaptive estimators.
