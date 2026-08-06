# Experiment 3 — Network-Specific Low-Rank Lower-Defect Factorization

Date: 2026-07-29 (America/New_York)

## Frozen question

Can the richer selected lower-order radial-Hermite defect be represented by a very small network-specific output subspace, and can that subspace and its signed coefficients be recovered from legal network/Kerdock information?

## Representation

For selected lower-defect slot `p` and final output coordinate `j`, define

`C[p,j] = delta_anchor[p] * beta_bar[p,j]`,

where `delta_anchor` is the exact selected lower-order anchor defect relative to the same-cloud anchor, and `beta_bar` is the fold-weighted cross-fitted direct-output coefficient map. Thus `C` has shape `128 x 256`, and the complete direct output correction is `sum_p C[p,:]`.

This representation was frozen before the grouped validation. It is not an arbitrary reshape of a 128-vector: every matrix row is the actual output contribution of one frozen selected lower-order slot.

## Cohorts and splits

- Training/development: 16 new width-256 base networks, IDs 4300–4315.
- Frozen validation: 24 new width-256 base networks, IDs 4400–4423.
- Rotations: 3, 11, and 97 for every base network.
- All rotations of a base network remained in one split.
- Gaussian reference moments in the primary run: two scrambled Sobol halves of 4,096 nodes each.
- Independent cross-reference audit: two disjoint 16,384-node streams for anchor moments and two disjoint 16,384-node streams for output targets.

The shared launch pack did not contain the official high-precision raw lower-defect arrays or matched K32/K128 vector teachers. Therefore, this is a self-contained width-256 sandbox experiment, not a protected challenge-holdout certification.

## Frozen candidate mechanisms

1. Exact per-network SVD modes of `C` (oracle ceiling).
2. Pooled output modes.
3. Legal right singular vectors of `beta_bar`.
4. Fold-output covariance modes.
5. Soft-gate suffix-adjoint modes.
6. Raw suffix weight-product modes.
7. Union of deterministic legal modes.
8. Eight-template exact-mode codebook selected by legal-subspace distance.
9. Grouped ridge prediction of a small signed coefficient vector.
10. Direct 256-vector learner and frozen-template scalar baselines.

## Promotion gate

A deployable candidate required approximately:

- aggregate candidate/base ratio at most 0.595;
- at least 75% wins;
- worst ratio at most 1.10–1.15;
- positive correction alignment;
- stable legal mode selection across rotations;
- credible complete-package cost.

Oracle-selected modes are mechanism evidence only.
