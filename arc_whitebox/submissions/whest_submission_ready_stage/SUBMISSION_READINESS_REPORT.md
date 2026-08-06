# Submission-readiness work completed

## Main algorithmic changes

- Added validation-only calibration of recursively predicted factorized-K3
  `x1/x1a` features against exact public moments.
- Expanded the hybrid from full replacement to a validation-tunable family that
  can retain ARC's native K3 mean and covariance corrections.
- Added pairwise correlation caps and a next-layer variance guard rather than
  destructive repeated PSD projection.
- Split the 15 validation MLPs into 7 calibration and 8 tuning MLPs. The existing
  15 test MLPs are evaluated once after selection.
- Added an optional truly fresh 100-network audit on global IDs 100-199.

## Numerical correction

The original fixed-order Plackett integral directly integrated correlation `rho`
and became inaccurate near `|rho|=1`. Reparameterizing with `rho=sin(theta)`
removes the endpoint singularity.

Tests:

- exact zero-mean ReLU kernel over 1,001 correlations in [-0.999,0.999]: machine
  precision;
- 2,000 random nonzero-mean bivariate cases: 20-node versus 128-node worst
  absolute moment difference `1.57e-7`, RMS `3.59e-9`.

## Submission engineering

- Added a float32, chunked `flopscope.numpy` CoefNet runtime.
- Added pickle-free `flops.Module` packing support.
- Added a package-size and forbidden-import auditor.
- Added a CoefNet FLOP proxy: about 11.84B FLOPs, or 4.35% of the 272B Phase-1
  budget, before factorized-K3 cost.
- Tested float32 versus float64 CoefNet inference on 500,000 standardized rows:
  `2.58e-7` relative RMS difference.

## Remaining blocker

A complete submission still requires a `flopscope.numpy` port of the required ARC
factorized-K3 update. That engineering is justified only after the deployable
factorized-K3 hybrid passes the fresh holdout audit.
