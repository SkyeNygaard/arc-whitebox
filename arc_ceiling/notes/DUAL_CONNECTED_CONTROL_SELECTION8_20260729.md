# Dual connected-K3 anchor in the actual control

## Question

Does the cheap adjoint prediction of two connected-K3 contractions remain
useful when inserted as the anchor of the radially homogenized connected
cubic control?

The test uses only selection IDs 160--167.  It uses:

- rotation seed 3 and the standard Kerdock design;
- layer 29;
- sample activation mean as the pointwise center;
- the top two radial-corrected sample connected-cubic SVD directions;
- six-fold held-basis coefficient fitting with ridge 0.1;
- fixed cheap-dual scale grid `1.7, 1.8, 1.9, 1.95, 2.0, 2.1, 2.2`;
- both the full diagonal adjoint probe and its rank-32 coordinate truncation.

The deployable anchors consume only two transported connected-K3 scalar
contractions.  They do not reconstruct raw M21 and do not consume oracle
target mean/covariance.  Oracle moments are used only to measure ceilings and
anchor error.  No new holdout IDs were touched.

## Result

| Anchor | Mean MSE | Ratio to raw Kerdock | Wins / 8 | Pooled anchor error / `||Q-E||` |
|---|---:|---:|---:|---:|
| Raw Kerdock | 2.964e-7 | 1.000 | -- | -- |
| Oracle arbitrary sample-center anchor | 1.684e-7 | **0.568** | 8 | 0 |
| Oracle connected anchor approximation | 2.934e-7 | 0.990 | 4 | 1.033 |
| Cheap full adjoint, scale 1.7 | 4.920e-5 | 165.99 | 0 | 2.623 |
| Cheap rank-32 probe, scale 1.7 | 2.586e-6 | 8.724 | 0 | 3.344 |
| Cheap rank-32 probe, scale 1.95 | 2.086e-5 | 70.38 | 0 | 2.862 |

The remaining fixed scales are worse in MSE.  The best pooled anchor-norm
calibration is approximately 1.668 for the full probe and 1.967 for the
rank-32 probe, but their minimum pooled anchor errors are still respectively
2.61 and 2.86 times the true Kerdock quadrature discrepancy.

Per-network optimal full-probe scales are highly heterogeneous:

`2.628, 1.789, 2.682, 2.177, 1.966, 1.400, 1.447, 1.540`.

Thus a universal scalar calibration cannot repair the approximation.

## Interpretation

This answers the missing oracle-construction question positively: with oracle
state, the radially homogenized pointwise function and its exact
arbitrary-center anchor cancel enough Kerdock error to reduce MSE by 43.2%.

It also identifies a sharper obstruction than connected-K3 prediction
accuracy.  With the pointwise function centered on the sample mean, even the
*true* connected-K3 anchor is not the correct expectation.  The omitted
arbitrary-center lower-moment correction is already 1.03 times the quadrature
discrepancy, and the oracle connected-only control is neutral overall.
Consequently, better connected-K3 transport alone cannot make this control
win under sample centering.

The cheap adjoint anchor has high large-signal alignment (pooled cosine about
0.980), but the control needs accuracy on the much smaller `Q-E` scale.  Its
20% large-anchor error is 2.6--2.9 quadrature-error units, which turns the
cross-fitted correction into severe MSE inflation.

## Next implication

The next version must predict the exact arbitrary-center scalar anchor, not
only connected K3.  It can still avoid reconstructing raw M21 by adding
adjoint scalar contractions for the lower-order sample-center correction.
Those contractions involve only mean/second-moment transport and should be
much cheaper than a full third-order state.  A decisive next ceiling is:

1. exact connected-K3 scalar plus cheap-k2 scalar center correction;
2. oracle connected-K3 scalar plus cheap-k2 scalar center correction;
3. cheap adjoint K3 scalar plus cheap-k2 scalar center correction.

Only if (1) retains most of the 0.568 oracle ratio is it worth improving the
cheap K3 source model further.

## Artifacts

- `arc_ceiling/eval_dual_connected_control.py`
- `arc_ceiling/results/dual_connected_control_rank2_selection8.json`
