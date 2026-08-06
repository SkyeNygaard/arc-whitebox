# Preregistration — orientation-odd phase observable

**Status:** specification only. Freeze before opening any new validation labels.  
**Purpose:** decide whether a canonical downstream orientation exposes transferable signed phase beyond the M153 quotient features.

## Research question

Does one explicitly canonical, orientation-odd runtime contraction add grouped predictive value for a frozen signed correction coefficient beyond the best matched constant and the nine M153 even features?

## Scientific target

Use one frozen correction direction or one low-dimensional source basis with a comfortably sufficient Oracle ceiling. Do not reuse the projected-ReLU scalar source whose untouched-test Oracle gain was below its continuation gate.

Primary target:

\[
a^*=\arg\min_a\|e-Aa\|^2
\]

for a fixed source matrix `A`, evaluated by exact final-output replay. A one-dimensional version may be used only if its independent Oracle ceiling exceeds 1.20x raw gain, leaving margin above a 1.10x promotion gate.

## Canonical orientation construction

Freeze exactly one construction before target inspection:

1. compute a downstream-sensitive basis from legal realized final-layer quantities;
2. order basis vectors by decreasing singular value, with deterministic tie breaking;
3. orient each vector by the sign of its largest-magnitude coordinate;
4. break equal-coordinate ties by the smallest coordinate index;
5. reject or abstain when the orientation margin is below a preregistered numerical threshold.

No label, Oracle coefficient, or reference mean may influence ordering or sign.

## Primary odd feature

For each oriented basis vector `u_j`, compute one signed contraction

\[
z_j=\langle u_j,s\rangle,
\]

where `s` is a frozen legal source summary. The primary experiment uses one prespecified `j` or a prespecified small vector `(z_1,...,z_r)` with `r<=4`.

## Baselines

1. zero correction;
2. best global constant coefficient;
3. best bounded constant vector in the same source basis;
4. M153 even features only;
5. odd feature only;
6. even plus odd features;
7. Oracle coefficient in the same basis, ceiling only.

## Grouping and chronology

- Group all rotations of one base network.
- Select hyperparameters only inside grouped development folds.
- Freeze one bounded model before a fresh validation cohort.
- Do not use protected official data.
- Report exposed IDs and overlap with every historical campaign.

## Model class

Use a bounded linear or monotone calibration model. No architecture sweep. Coefficients and predictions must be clipped to preregistered ranges. Runtime optimization is closed-form.

## Primary success conditions

All must hold on grouped validation:

1. source-basis Oracle raw gain at least 1.20x;
2. even-plus-odd policy raw gain at least 1.10x;
3. grouped 95% interval lower bound above 1.05x;
4. worst-network candidate/base at most 1.25;
5. policy beats the best matched constant policy;
6. policy beats the even-only feature model;
7. positive adjusted-score gain after full cost;
8. orientation abstention does not exceed 25% of networks;
9. no single base network contributes more than 25% of total gain.

## Theorem-facing diagnostics

Measure:

- empirical orientation defect under legal rotations;
- target sign change under deliberate representation reorientation;
- T45 `delta_c` for the learned policy;
- coefficient prediction value attributable uniquely to the odd feature;
- sign stability versus orientation margin.

## Stop conditions

Stop permanently for this construction if:

- the odd feature does not beat the even-only representation;
- orientation is numerically unstable on more than 10% of cases;
- the matched constant policy captures at least 90% of the feature policy gain;
- exact nonlinear replay erases the linearized gain;
- tails or complete cost fail.

A failure closes only the frozen orientation, source, feature and policy class.
