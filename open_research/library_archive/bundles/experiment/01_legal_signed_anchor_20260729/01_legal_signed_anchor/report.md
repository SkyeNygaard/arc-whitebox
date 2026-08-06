# Legal signed-anchor experiment: reanchored structured-pilot defect

## Terminal verdict

**FAIL — close this exact estimator and both tested confidence/sign-scale rescues.**

The compute-compliant estimator did not pass the development gate and transferred with the wrong sign on the untouched block. The failure is statistical, not lack of oracle headroom.

## Why this was a materially new test

This candidate does not estimate a large absolute moment and subtract Kerdock afterward. At every ReLU layer it estimates a local, internally cancelled source around the already-computed Kerdock law:

1. The full 129-basis cloud supplies observed means and marginal variances.
2. Two disjoint two-basis pilots supply structured covariance sources without new trajectories.
3. A Gaussian closure is applied to the observed preactivation law plus the carried defect.
4. The observed pilot post-ReLU covariance is subtracted inside the layer update.
5. One shared covariance defect state is transported to layer 29.
6. Only the frozen selected means, marginal second moments, and row-direction pair moments are contracted into the 128 radial-Hermite anchors.
7. Final scoring is through the complete direct-output control, not anchor RMSE alone.

## Frozen protocol

- Full 129-basis Kerdock design, 66,048 rows.
- Frozen 128 sample-row radial-Hermite probes.
- Tuning IDs 3000–3005.
- Untouched validation IDs 3006–3011.
- Two independent 16,384-point scrambled Sobol reference halves per network, evaluation only.
- Candidate construction receives no reference moments or reference outputs.
- Frozen scale grid and disagreement gate are selected on tuning only.

## Primary results

| Method | Tuning candidate/base | Validation candidate/base | Validation wins | Validation worst |
|---|---:|---:|---:|---:|
| Exact lower-order oracle | 0.0727 | 0.1155 | 6/6 | not used for deployment |
| Raw recurrence, frozen alpha 0.1 | 0.9844 | **2.1610** | 0/6 | **3.0477** |
| Pilot-disagreement abstention | 0.8246 | **1.1776** | 0/6 | **1.9054** |
| Learned signed-scale rescue | 0.7812 | **1.6174** | 1/6 | **3.3867** |

The raw anchor direction had mean cosine +0.374 on tuning but **−0.058** on validation. Mean relative anchor error was 6.98 on tuning and 8.32 on validation. This is a phase/source failure, not merely a global scale miss.

## High-compute diagnostic

Before compression, two separate four-basis recurrences were tested on 8 tuning and 8 validation networks. The aggregate tuning grid selected alpha 0. A heavily shrunk three-feature signed-scale model was tail-safe but score-negative on validation: candidate/base 1.0039, 6/8 nominal wins, worst 1.0551. This diagnostic is retained because it shows that more pilot rows do not expose a useful transferable sign.

## Compute accounting

The shipping-shaped version uses four total pilot bases and one shared covariance state.

Nominal arithmetic estimate:

- Pilot post-covariance contractions: 8.590B FLOPs.
- Shared covariance transport (`W.T @ delta_C @ W`): 2.147B FLOPs.
- Fused full-cloud mean/second-moment reductions: approximately 2.2–3.25B primitive operations, depending on accounting/fusion.
- Scalar closure and selected-anchor contractions: small relative to the above and partly shared with the radial-Hermite control.

Projected added arithmetic is approximately **12.9–14.0B**, so it is only nominally under the 14B cap with fused reductions. No official FlopScope certificate was produced. Since the statistical gate fails by a large margin, further billing work is not warranted.

## Interpretation

The layerwise Gaussianization source is real on some networks and can produce large single-network gains, but its phase is not stable across networks. Two-pilot agreement is not a valid no-headroom signal: the pilots can agree on a confidently wrong source. A low-dimensional learned scale overfits the tuning source geometry and reverses on validation.

This closes:

- marginal/full-covariance Gaussianization defects estimated from four embedded Kerdock pilot bases;
- disagreement-only abstention for that estimator;
- a three-feature source-norm/growth/disagreement signed-scale model.

It does **not** close:

- a genuinely non-marginal adjoint-weighted source already available from shared arithmetic;
- a legal independent analytic anchor whose residual is then learned;
- deterministic cubature that estimates the selected lower-order defect directly rather than through Gaussian closure;
- weight-derived bias correction with a fresh immutable training/validation population.

## Reproduction

```bash
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
pytest -q tests
python src/run_single.py --network 3000 --truth-n 16384 --threads 8 \
  --out results/records_shared/network_3000.json
python src/signed_scale_rescue_shared.py \
  --records results/records_shared \
  --out results/compute_compliant_signed_scale_rescue.json
```

The aggregate primary result is `results/compute_compliant_development12.json`.
