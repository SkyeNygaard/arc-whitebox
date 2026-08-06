# Agent 1 — Residual-spectrum benchmark and dominant-channel report

## Decision

**CONTINUE depth-local layer-31 residual-surrogate research.** The frozen mechanism passes the major-theory gate on screen, validation, and the untouched 64-network holdout.

**STOP first-layer two-moment transport as a major path. STOP brute-force finite-width harmonic projection.** Degree 6 remains a possible cheap additive control, but its limiting-kernel oracle ceiling is below the 1.3× major-theory gate.

## Mechanism tested

For the protected seed-3 complete Kerdock rule, retain the full particle cloud through layer 31. For each neuron, apply the minimum scalar translation followed by ReLU projection that makes the empirical layer-31 post-ReLU mean equal an independent reference mean, then propagate the true final layer.

The reference is the average of independently Haar-rotated complete Kerdock rules. It is split into two groups of eight rotations. The intervention built from group A is scored against B, then reversed, so the oracle is cross-fitted and cannot improve by fitting the evaluation target noise.

This is an oracle attribution, not a deployable estimator: the true layer-31 mean is unavailable at inference. It measures how much final error is concentrated in that channel. Channels overlap and must not be summed.

## Staged results

| Stage | Networks | Raw cross-fitted MSE ratio | MSE removed | 95% CI | Wins | Worst candidate / baseline |
|---|---:|---:|---:|---:|---:|---:|
| Screen | 8 | **5.777×** | **82.69%** | 76.53%–86.46% | 8/8 | 0.378 |
| Validation | 24 | **4.233×** | **76.38%** | 64.93%–84.14% | 24/24 | 0.741 |
| Holdout | 64 | **4.572×** | **78.13%** | 73.41%–82.11% | 64/64 | 0.865 |

### Reference quality and baseline

The screen baseline raw MSE against the 16-rotation reference is `2.8535515e-07`. The independently estimated target-noise floor is `1.1287121e-08` (3.96%), leaving `2.7406803e-07` noise-corrected. This is close to the project’s previously observed strict-holdout scale.

Validation baseline: `3.3246702e-07` observed and `3.1328576e-07` noise-corrected. Holdout baseline: `2.8879353e-07` observed and `2.7370952e-07` noise-corrected.

### Raw MSE, effective compute, and adjusted score

Baseline effective compute is fixed at `175.500 B`. On validation its observed adjusted score is `2.1451457e-07`; on holdout it is `1.8633553e-07`.

A magical deployable layer-31 correction would require approximately one extra final-layer replay, idealized here as `180.984 B` effective compute (+3.125%). Using the cross-fitted oracle raw MSE gives adjusted scores `5.5275639e-08` on validation and `4.4225537e-08` on holdout. These are **mechanism ceilings, not submission projections**, because the reference mean is unavailable.

## Layer attribution

| Layer | Raw-MSE ratio | MSE removed | 95% CI | Wins | Worst candidate / baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.161× | 13.83% | 3.06%–20.93% | 6/8 | 1.352 |
| 4 | 1.152× | 13.22% | -3.41%–23.06% | 5/8 | 1.330 |
| 8 | 1.291× | 22.54% | 3.11%–32.63% | 6/8 | 1.445 |
| 12 | 1.672× | 40.20% | 20.54%–49.82% | 6/8 | 1.069 |
| 16 | 1.959× | 48.96% | 30.95%–59.40% | 7/8 | 1.009 |
| 20 | 2.153× | 53.55% | 39.22%–61.89% | 8/8 | 0.898 |
| 24 | 2.680× | 62.69% | 47.04%–72.89% | 8/8 | 0.768 |
| 28 | 4.073× | 75.45% | 65.68%–81.56% | 8/8 | 0.561 |
| 29 | 4.510× | 77.83% | 69.88%–82.41% | 8/8 | 0.493 |
| 30 | 4.993× | 79.97% | 72.58%–84.32% | 8/8 | 0.438 |
| 31 | 5.777× | 82.69% | 76.53%–86.46% | 8/8 | 0.378 |

**Dominant empirical channel:** layer 31 removes `82.69%` of final MSE on the frozen screen (95% CI `76.53%`–`86.46%`), and transfers almost exactly to validation and holdout.

The interpretation is locality, not causal creation at layer 31: upstream cubature defects accumulate, then compress into a layer-31 mean error that strongly controls the final output.

## Output PCA modes

Across 16 independent Kerdock rotations per screen network, the final-error first PCA mode has mean share `39.45%`, the first two have `54.25%`, and 90% requires `8.50` modes on average. Layer-31/final linear CKA is `0.9828` (minimum `0.9730`).

## Spherical-harmonic attribution

The direct independent-probe estimator was mathematically unbiased but unusable: degree-6 reproducing-kernel weights had RMS about `3.6e3`, and split halves disagreed in sign. An antipodal latitude-smoothing alternative was also variance-limited. These negative diagnostics are retained; no finite-width degree vector is claimed.

The exact depth-32 infinite-width ReLU-kernel decomposition is stable and reproduces the known total Kerdock kernel MSE:

| Degree | Fraction of limiting-kernel Kerdock MSE |
|---:|---:|
| 6 | 13.93% |
| 8 | 10.25% |
| 10 | 8.14% |
| 12 | 6.65% |
| 14 | 5.59% |

Degrees 6–14 account for `44.57%`; the remaining `55.43%` is above degree 14. Degree 6 is the largest individual degree, but exact removal would improve raw MSE by only `1.162×`, below the 1.3× major-theory gate. To deliver at least a 5% adjusted-score gain, an exact degree-6 control must add less than `10.38%` compute.

## First-layer moment defect

After fixing the inherited conditional/unconditional normalization bug, the exact first-layer mean-and-second-moment intervention gives only `1.052×` aggregate improvement (95% gain CI `-4.81%`–`16.32%`), wins 5/8, and has worst candidate/baseline `1.266`. **STOP as a major path.**

## Recommendation to the next estimator agent

Apply Gaussian-closure residual control at **layer 31**, not as a standalone final predictor:

1. Construct a homogeneous sample-level surrogate `g31(x)` for the layer-31 activation vector with analytic Gaussian expectation.
2. Estimate `mu31 = E[g31] + Q_K(a31 - g31)` on the existing Kerdock cloud.
3. Translate the real layer-31 particles to match `mu31`, then replay only the true final layer.
4. Insert the validated early covariance-eigenmode correction inside `g31` and judge only the composed residual estimator.
5. Charge surrogate evaluation plus the extra final replay. The oracle has enough margin that a useful fraction of its effect can tolerate modest added compute.

This directly operationalizes the finding: the valuable object is not global closure accuracy, but correlation with the layer-31 Kerdock mean defect that drives the scored output.

## Exact reproduction

```bash
cd residual_cv_agents_1_2
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.
pytest -q
python agent1_residual_spectrum_screen.py references
python agent1_residual_spectrum_screen.py layers
python agent1_residual_spectrum_screen.py first-layer
python agent1_kernel_spectrum_mp.py
python agent1_frozen_stage.py --split validation
python agent1_frozen_stage.py --split holdout
python agent1_rotation_pca.py
python build_agent1_report.py
```

Frozen declarations are in `config/agent1_screen_protocol_v2.json` and `config/agent1_selected_mechanism_v2.json`.

## Final verdict

**CONTINUE.** The depth-local layer-31 mean channel exceeds the 1.3× major-theory gate by a wide margin, transfers from 8-network screen to 24-network validation and 64-network untouched holdout, and has no adverse tail. The next task is to capture this channel with a deployable analytic residual surrogate; do not spend the next round on first-layer transport or brute-force harmonic estimation.
