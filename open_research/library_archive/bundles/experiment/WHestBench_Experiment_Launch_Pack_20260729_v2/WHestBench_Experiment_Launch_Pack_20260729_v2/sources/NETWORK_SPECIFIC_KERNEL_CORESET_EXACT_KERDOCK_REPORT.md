# Network-Specific Kernel Coreset — Exact Kerdock Continuation

**Date:** 2026-07-29  
**Status:** Full 20-network synthetic preregistration completed on the exact 66,048-point Kerdock geometry  
**Protected official/Mini-100 holdouts:** not opened

## Final decision

**Close this network-specific kernel coreset branch in its tested form.**

The exact Kerdock experiment resolves the scaled-surrogate ambiguity:

- A good 8,192-row positive weighted support **does exist** for each network.
- The deployable suffix kernel cannot identify that support consistently.
- Even when it identifies a usable support, its own feature calibration does not assign accurate weights.
- Increasing feature capacity, optimization rounds, or allowing regularized signed weights does not approach score break-even.

The failure is not marginal. The deployable arm's mean added MSE is **1.282e-07**, versus the preregistered **1.1e-8** gate: **11.7x too large**.

## Exact experimental setup

For each of 20 fresh width-256, depth-32, bias-free He-initialized ReLU MLPs:

- Exact Kerdock/MUB support: 66,048 rows, 129 orthonormal bases.
- Full propagation through layer 28.
- Retained support: exactly 4,096 antipodal pairs = 8,192 rows.
- Basis quotas: 31 or 32 pairs per basis, totaling exactly 4,096.
- Pilot: 8 antipodal pairs per basis = 2,064 rows.
- Tail: final four layers.
- Four deterministic balanced starts.
- 32 within-basis exchange rounds.
- Positive, basis-mass-preserving calibration.
- Relative weight bounds `[0.05, 4.0]`.
- ESS floor `0.8 M`.

The objective and calibration preserve equal total mass for every Kerdock basis. This prevents an apparent compression gain from silently sacrificing the design's basis weighting.

## Arms

- **R0:** balanced random support, basis-uniform weights.
- **R1:** balanced random support, anchor calibration.
- **K1:** anchor-kernel exchange selection and anchor calibration.
- **K2:** selective nonlinear suffix-residual kernel selection and calibration.
- **O1:** K2-selected support with oracle final-output calibration.
- **O2:** oracle final-output selection and calibration.

O1 and O2 are diagnostics and are not deployable.

## Aggregate results

| Arm | Mean added MSE | Median | Worst | Networks ≤1.1e-8 | Networks ≤2.2e-8 |
|---|---:|---:|---:|---:|---:|
| R0 | 3.655e-06 | 2.285e-06 | 1.909e-05 | 0/20 | 0/20 |
| R1 | 2.069e-07 | 2.029e-07 | 4.673e-07 | 0/20 | 0/20 |
| K1 | 1.341e-07 | 1.253e-07 | 2.781e-07 | 0/20 | 0/20 |
| K2 | **1.282e-07** | 1.207e-07 | 2.797e-07 | **0/20** | **0/20** |
| O1 | 2.941e-08 | 1.207e-08 | 1.646e-07 | 9/20 | 16/20 |
| O2 | **2.391e-09** | 1.645e-12 | 1.496e-08 | 19/20 | **20/20** |

Bootstrap 95% intervals for mean added MSE:

- K1: `1.117e-07` to `1.591e-07`
- K2: `1.089e-07` to `1.507e-07`
- O1: `1.158e-08` to `5.055e-08`
- O2: `9.399e-10` to `4.220e-09`

## Preregistered gate evaluation

### Gate 1 — support oracle: PASS

Required:

- O2 aggregate added MSE ≤ `5.5e-9`.
- O2 worst network ≤ `2.2e-8`.

Observed:

- Mean: **2.391e-09**
- Worst: **1.496e-08**
- 20/20 below `2.2e-8`.

The support size is sufficient in principle.

### Gate 2 — deployable selector: FAIL

Required:

- O1 aggregate ≤ `1.1e-8`.
- At least 18/20 below `2.2e-8`.
- Worst below `4.4e-8`.

Observed:

- Mean: **2.941e-08**
- Only **16/20** below `2.2e-8`.
- Worst: **1.646e-07**.

The deployable kernel sometimes finds an excellent support, but its support failures are large and network-specific.

### Gate 3 — deployable kernel: FAIL decisively

Required K2 aggregate added MSE ≤ `1.1e-8`.

Observed:

- Mean: **1.282e-07**
- Bootstrap 95% interval: **[1.089e-07, 1.507e-07]**
- 0/20 below `2.2e-8`.
- Worst: **2.797e-07**.

K2 misses the primary gate by **11.7x** before honest selection wall-time accounting.

## Per-network results

| Seed | R1 | K1 | K2 | O1 same support | O2 oracle | K2 ESS | Research runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 44000 | 2.004e-07 | 1.758e-07 | 1.445e-07 | 1.056e-08 | 5.928e-09 | 0.976 | 16.5s |
| 44001 | 7.594e-08 | 6.053e-08 | 6.059e-08 | 1.312e-08 | 4.735e-14 | 0.971 | 14.5s |
| 44002 | 2.309e-07 | 1.518e-07 | 1.604e-07 | 1.129e-07 | 6.139e-14 | 0.960 | 13.7s |
| 44003 | 1.056e-07 | 9.556e-08 | 7.856e-08 | 4.424e-10 | 2.658e-12 | 0.966 | 18.3s |
| 44004 | 2.069e-07 | 1.365e-07 | 1.505e-07 | 8.508e-09 | 1.833e-09 | 0.968 | 13.9s |
| 44005 | 2.055e-07 | 4.771e-08 | 6.145e-08 | 6.561e-14 | 5.735e-09 | 0.969 | 13.5s |
| 44006 | 4.588e-07 | 1.161e-07 | 1.204e-07 | 1.312e-08 | 8.824e-14 | 0.970 | 12.7s |
| 44007 | 2.363e-07 | 1.159e-07 | 1.001e-07 | 1.889e-08 | 8.649e-16 | 0.984 | 14.1s |
| 44008 | 4.673e-07 | 2.482e-07 | 2.797e-07 | 1.351e-08 | 1.496e-08 | 0.972 | 13.5s |
| 44009 | 1.044e-07 | 1.007e-07 | 1.007e-07 | 3.928e-10 | 3.540e-09 | 0.967 | 14.0s |
| 44010 | 3.128e-07 | 2.781e-07 | 1.634e-07 | 4.150e-13 | 7.660e-09 | 0.971 | 13.2s |
| 44011 | 1.644e-07 | 1.385e-07 | 9.662e-08 | 8.100e-09 | 4.139e-09 | 0.962 | 15.5s |
| 44012 | 2.254e-07 | 1.619e-07 | 9.544e-08 | 1.103e-08 | 3.992e-09 | 0.967 | 15.6s |
| 44013 | 2.289e-07 | 9.657e-08 | 1.166e-07 | 8.168e-14 | 2.648e-11 | 0.959 | 13.8s |
| 44014 | 1.671e-07 | 1.340e-07 | 1.951e-07 | 1.070e-07 | 6.325e-13 | 0.957 | 15.1s |
| 44015 | 1.401e-07 | 1.340e-07 | 1.210e-07 | 4.910e-13 | 2.057e-14 | 0.959 | 15.3s |
| 44016 | 1.171e-07 | 8.759e-08 | 1.153e-07 | 7.331e-08 | 5.610e-16 | 0.985 | 13.4s |
| 44017 | 2.582e-07 | 1.166e-07 | 1.391e-07 | 1.646e-07 | 1.432e-14 | 0.969 | 13.9s |
| 44018 | 1.435e-07 | 1.721e-07 | 1.288e-07 | 1.916e-08 | 3.137e-14 | 0.977 | 13.0s |
| 44019 | 8.864e-08 | 1.135e-07 | 1.364e-07 | 1.345e-08 | 3.967e-13 | 0.971 | 13.3s |

## Additional falsification checks

### Feature capacity

On scaled width-256 geometry, increasing selected nonlinear neurons from 8 to 64 reduced error only gradually. Because sketch cost rises linearly, the break-even gap worsened:

- q=16: approximately 11.3x above break-even.
- q=32: approximately 12.2x above break-even.
- q=64: approximately 20.9x above break-even.

On exact Kerdock seed 44000, increasing to q=32 improved K2 from `1.445e-7` to `1.041e-7`, but the q=32 break-even added MSE is only about `1.9e-8`. It remained more than 5x too noisy, and same-support oracle error worsened.

### More exchange optimization

On the worst early selector failure, increasing from 32 to 128 exchange rounds reduced O1 from `1.129e-7` to `6.200e-8`, still above every selector gate. It increased research runtime by about five seconds per network. Runtime-priced deployment would be worse even if the statistical gain generalized.

### Pilot-trained output-aware metric

A pilot-trained reduced-rank output metric was tested with 16–128 modes and up to 512 exact pilot pairs on the scaled geometry. It was consistently worse than SNSR-K. The remaining output discrepancy is not captured by a stable low-rank pilot regression.

### Signed calibration

Regularized basis-preserving signed calibration did not exploit negative weights. The selected optimum remained effectively positive and changed deployable error only marginally (`1.412e-7` on the tested exact-Kerdock network).

### Oracle rank

Oracle PCA diagnostics required roughly 128 final-output modes to approach the added-error target. The residual is not a rank-1 or rank-8 output phenomenon despite large leading variance modes.

## Interpretation

The full experiment separates three questions:

1. **Can 8,192 rows represent the full Kerdock tail mean?**  
   Yes. O2 passes cleanly.

2. **Can the realized suffix kernel find such rows without seeing final outputs?**  
   Not reliably. O1 has several catastrophic support misses.

3. **Can the kernel's own features calibrate a selected support?**  
   No. K2 fails all 20 networks by a wide margin.

The deep tail's important gate disagreements are high-dimensional and orientation-sensitive. Average activation geometry and a small selected residual dictionary are insufficient. More feature capacity spends the arithmetic savings faster than it reduces compression error.

## Final recommendation

- Do not open the protected 24-network or 64-network holdouts.
- Do not integrate this coreset into the submission estimator.
- Do not continue q sweeps, exchange-depth sweeps, signed-weight sweeps, or pilot low-rank regressions.
- Keep the exact-Kerdock diagnostic code as a reusable oracle harness for genuinely new feature maps.
- Return competition engineering effort to the protected suffix-compiler package.

A future revival should require a qualitatively new representation that predicts the full high-dimensional tail discrepancy—not a larger version of the same finite suffix sketch.
