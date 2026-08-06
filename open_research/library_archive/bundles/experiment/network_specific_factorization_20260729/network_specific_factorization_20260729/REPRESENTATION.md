# Representation

## Chosen object

The experiment factorizes the output-contribution matrix

`C = diag(delta_anchor) @ beta_bar`,

with shape `128 x 256`.

- Rows correspond to the frozen selected lower-order mean/diagonal-second/pair-moment contractions.
- Columns correspond to the 256 final outputs.
- `delta_anchor[p]` is the lower-order anchor defect in slot `p`.
- `beta_bar[p,:]` is the fold-weighted direct-output sensitivity of that slot.
- The exact lower-order direct correction is `1^T C`.

## Why this is meaningful

A low-rank approximation of `C` says that the slotwise contributions occupy a small downstream output subspace. Rank is therefore tied to the final-control geometry, not to an arbitrary tensor reshape or component scaling.

The experiment always evaluates truncated matrices through the complete final-output correction. Frobenius energy is diagnostic only.

## Normalization and orientation

- SVD columns are ordered by singular value.
- Each mode sign is canonicalized by making its largest-magnitude coordinate positive.
- Legal subspaces use only weights, Kerdock activations, fold summaries, suffix gates, and fitted direct-control coefficients.
- Exact mode orientation and coefficients never enter a deployable candidate.

## Important naming qualification

The `probe32` diagnostic keeps the first 32 of the 128 frozen selected slots. It is not asserted to be the historical K32 teacher, because matched K32/K128 raw vector assets were absent from the shared bundle.
