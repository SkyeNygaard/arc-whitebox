# Near-full two-stream RQMC submission

This is the best locally validated honest sampling package.

## Frozen design

- Stream A: scrambled Sobol seed 101, 16,384 stored sphere directions plus
  antipodes, totaling 32,768 forward rows.
- Stream D: independent scrambled Sobol seed 404, 15,000 stored sphere
  directions plus antipodes, totaling 30,000 forward rows.
- Frozen blend:
  `0.4922222558500433 * A + 0.5077777441499567 * D`.
- Directions are normalized to `E[chi_256]`; Gaussian radius is integrated
  exactly using positive homogeneity of a bias-free ReLU network.
- No covariance whitening, target/MLP lookup, or grader-dependent adaptation.
- MLP weights are cast once through tracked `flopscope.numpy` to float32,
  matching the benchmark precision and the research harness.
- Only the ranked final row is sampled. The first row is analytic and exact;
  intermediate unranked rows are zero.

The streams and blend were selected on whole-MLP rows 0–49. Rows 50–99 were
kept as a strict holdout until all choices were frozen.

## Official mini results

Runtime: whestbench 0.13.0, flopscope 0.9.1.

| Evaluation | Raw final MSE | Adjusted score | Mean multiplier | Failures |
|---|---:|---:|---:|---:|
| Selection rows 0–49 | 3.3931958e-7 | 3.2904731e-7 | ~0.96975 | 0/50 |
| Strict holdout rows 50–99 | 3.7429937e-7 | 3.6309259e-7 | ~0.96999 | 0/50 |
| All 100 | 3.5680948e-7 | 3.4606995e-7 | 0.9698685 | 0/100 |

All-100 accounting:

- tracked FLOPs: 263,376,864,000 per MLP;
- mean effective compute: 263,804,231,159;
- mean residual wall time: 0.004274 s;
- maximum effective compute: 264,253,954,800;
- minimum observed budget margin: 7,746,045,200 FLOP-equivalents;
- failures: 0/100.

The isolated subprocess smoke test on mini row 0 passed:

- raw MSE: 1.4050846e-7;
- adjusted score: 1.3634303e-7;
- tracked FLOPs: 263,374,766,848;
- effective compute: 263,936,437,691;
- no budget, combined-budget, or runtime failure.

## Files

- `estimator.py`: grader entrypoint using only the standard library,
  `flopscope`, and `whestbench`.
- `sobol_sphere_a101_d404.npz`: fixed float32 base directions loaded via
  `flopscope.numpy.load`.
- `submission.tar.gz`: validated submission artifact.

The offline builder and `research_*.json` files are excluded by
`.whestignore`.
