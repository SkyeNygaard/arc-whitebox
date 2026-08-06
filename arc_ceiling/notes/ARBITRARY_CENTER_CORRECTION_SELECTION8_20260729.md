# Arbitrary-center correction decomposition

Selection IDs 160--167 only; no holdout IDs were touched.

## Identity

For sample pointwise center `m`, target mean `mu`, raw second moment `M11`,
and marginal second moment `M2`,

```text
A(m) = C21 / (d + 1) + R(m; mu, M11) / (d + 1)

R_ij = -2 (m_i - mu_i) M11_ij
       - M2_i (m_j - mu_j)
       + 2 (m_i^2 - mu_i^2) mu_j.
```

This was validated numerically against the raw-M21 oracle on all eight
networks.  A deployable correction needs only K1/K2.

## Main results

| Configuration | Raw MSE ratio | Wins / 8 | Anchor error / `Q-E` |
|---|---:|---:|---:|
| Full arbitrary-center oracle | **0.5681** | 8 | 0 |
| True-mean-centered K3-only oracle | **0.5681** | 8 | 0 |
| Oracle correction-only control | 0.8651 | 4 | 0 |
| Oracle K3 and correction as separate features | **0.5367** | 7 | 0 |
| Exact K3 + factorized post-K1/K2 correction | 9.6521 | 2 | 4.351 |
| Exact K3 + global sample K1/K2 | 0.9900 | 4 | 1.033 |
| Exact K3 + held-fold radial sample K1/K2 | 1.0033 | 4 | 1.225 |
| Cheap K3 scale 1.7 + oracle correction | 163.265 | 0 | 2.641 |
| Rank-32 K3 scale 1.7 + oracle correction | 9.087 | 0 | 3.863 |

The global sample correction is exactly zero because its K1 estimate equals
the pointwise center.  A held-fold estimator was implemented explicitly; it
does become nonzero but does not estimate the population correction.

## Frozen correction shrinkage

The factorized correction is biased but has some useful alignment.  With
exact K3:

| Factorized correction multiplier | Raw MSE ratio | Wins / 8 | Anchor error / `Q-E` |
|---:|---:|---:|---:|
| 0.00 | 0.9900 | 4 | 1.033 |
| 0.05 | 0.9027 | 6 | 0.944 |
| 0.10 | **0.8654** | 6 | 0.909 |
| 0.15 | 0.8770 | 5 | 0.936 |
| 0.20 | 0.9370 | 4 | 1.018 |
| 1.00 | 9.6521 | 2 | 4.351 |

This small signal cannot rescue the inaccurate cheap K3 contraction.

## Interpretation

The 0.568 oracle gain does not fundamentally require estimating the
arbitrary-center correction.  Re-centering the pointwise polynomial at the
true mean and using the connected-K3 anchor reproduces it almost exactly.
The decisive state variable is therefore an integration-accurate target mean,
not a raw M21 reconstruction.

The sample mean is close in ordinary relative norm but not on the much smaller
quadrature-discrepancy scale.  Ordinary resampling cannot estimate its
deterministic Kerdock bias: global sample K1 is a no-op, and held-fold K1 is a
noisier estimate of the same integral.  A useful next path must predict the
sample-mean integration error from analytic transport or structured
Kerdock/Walsh error modes, rather than jackknifing the same point cloud.

## Artifacts

- `arc_ceiling/eval_arbitrary_center_correction.py`
- `arc_ceiling/results/arbitrary_center_correction_rank2_selection8.json`
- `arc_ceiling/results/arbitrary_center_correction_shrink_rank2_selection8.json`
