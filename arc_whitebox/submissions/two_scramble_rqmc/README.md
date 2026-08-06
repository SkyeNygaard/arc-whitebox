# Two-scramble Sobol sphere-frame submission

This folder is a self-contained, honest WhestBench 0.13.0 submission.

## Design

- Two independently scrambled Sobol blocks (offline seeds 0 and 1).
- Six alternating exact-radius and covariance-whitening passes per block.
- Block A: 16,384 stored directions plus antipodes, 32,768 forward rows.
- Block B: 8,192 stored directions plus antipodes, 16,384 forward rows.
- The final estimate is the row-count-weighted combination:
  `2/3 * mean(A) + 1/3 * mean(B)`.
- All MLP weights are explicitly cast through tracked `flopscope.numpy` to
  float32 once before sampling, matching the benchmark's specified weight
  precision and the public research harness.
- Only the ranked final row is sampled. The first row is analytic and exact;
  intermediate, unranked rows are zero.

The exact-radius step is valid because a bias-free ReLU network is positively
homogeneous. The finite covariance-whitening transformation is a deterministic
variance-reduction device; it can introduce a small finite-frame bias and is
therefore less theoretically clean than a purely randomized shifted lattice.
It is included because strict held-out tests improved.

## Verified results

Runtime: whestbench 0.13.0, flopscope 0.9.1, official Phase-1 mini data.

| Evaluation | Raw final MSE | Adjusted score | Mean multiplier | Failures |
|---|---:|---:|---:|---:|
| Rows 0–49 | 4.8787564e-7 | 3.7023931e-7 | ~0.7589 | 0/50 |
| Strict holdout rows 50–99 | 5.2385926e-7 | 3.9756465e-7 | ~0.7589 | 0/50 |
| All 100 | 5.0586745e-7 | 3.8390198e-7 | 0.7588979 | 0/100 |

All-100 mean accounting:

- tracked FLOPs: 206,231,980,800 per MLP (local runner);
- effective compute: 206,420,219,492;
- residual wall time: 0.001882 s per MLP;
- utilization: 75.8898%.

The isolated subprocess smoke test on mini row 0 also passed:

- raw MSE: 7.4256981e-8;
- adjusted score: 5.6358760e-8;
- tracked FLOPs: 206,229,883,648;
- effective compute: 206,439,616,770;
- no budget or runtime failure.

## Files

- `estimator.py`: grader entrypoint; imports only the standard library,
  `flopscope`, and `whestbench`.
- `sobol_u32.npz`: float32 sphere-frame directions loaded with
  `flopscope.numpy.load`.
- `submission.tar.gz`: packaged and validated submission artifact.

`make_sobol_asset.py` and `research_*.json` are research/build artifacts and
are excluded from the tarball by `.whestignore`.
