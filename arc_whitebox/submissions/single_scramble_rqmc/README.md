# Single-scramble seed-101 RQMC submission

This is the lower-scoring of the two validated honest sampling packages.

## Design

- One SciPy-scrambled Sobol block, fixed offline scramble seed 101.
- 16,384 stored float32 directions plus their antipodes, for 32,768 total
  forward rows.
- Every direction is normalized to `E[chi_256]`, integrating Gaussian radius
  exactly by positive homogeneity of the bias-free ReLU MLP.
- No covariance whitening, calibration, per-MLP lookup, or target-dependent
  data.
- MLP weights are cast once through tracked `flopscope.numpy` to float32,
  matching the benchmark's weight precision and the research harness.
- Only the ranked final row is sampled. The first row is analytic and exact;
  intermediate unranked rows are zero.

The fixed seed was selected using official full rows 0–49. Rows 50–99 remained
a strict whole-MLP holdout until the design was fixed.

## Verified results

Runtime: whestbench 0.13.0, flopscope 0.9.1, official Phase-1 mini data.

| Evaluation | Raw final MSE | Adjusted score | Mean multiplier | Failures |
|---|---:|---:|---:|---:|
| Selection rows 0–49 | 6.8750151e-7 | 3.4787962e-7 | ~0.5060 | 0/50 |
| Strict holdout rows 50–99 | 7.4482677e-7 | 3.7700910e-7 | ~0.5061 | 0/50 |
| All 100 | 7.1616414e-7 | 3.6244436e-7 | 0.5060780 | 0/100 |

All-100 mean accounting:

- tracked FLOPs: 137,489,433,344 per MLP (local runner);
- effective compute: 137,653,223,028;
- residual wall time: 0.001638 s per MLP;
- utilization: 50.6078%.

The isolated subprocess smoke test on mini row 0 passed:

- raw MSE: 3.0913628e-7;
- adjusted score: 1.5641154e-7;
- tracked FLOPs: 137,487,336,192;
- effective compute: 137,621,949,115;
- no budget or runtime failure.

## Files

- `estimator.py`: grader entrypoint, using only the standard library,
  `flopscope`, and `whestbench`.
- `sobol_sphere_seed101.npz`: fixed float32 directions loaded through
  `flopscope.numpy.load`.
- `submission.tar.gz`: packaged and validated artifact.

The offline asset builder and `research_*.json` reports are excluded by
`.whestignore`.
