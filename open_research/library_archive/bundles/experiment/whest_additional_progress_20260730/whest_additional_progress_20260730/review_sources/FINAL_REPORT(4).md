# Oracle-gap experiment campaign — final report

**Experiment:** OGAP-20260730  
**Architecture:** width 256, depth 32, bias-free He-ReLU MLPs  
**Scale:** 12 new base networks × 3 grouped rotations = 36 cases  
**Reference:** 16 independent complete-Kerdock rotations per base, split 8+8  
**Dense-equivalent compute lower estimate:** 1.235×10^14 floating-point operations  
**Official protected data opened:** no

## Executive verdict

The campaign substantially strengthens the proof story, but it does **not** produce a competition-ready estimator.

1. The right exact theorem is the **observability-envelope theorem**: the best correction from a declared information set is the conditional expectation of the baseline error given that information. This gives a rigorous class-relative ceiling without assuming independent layer injections or extending T22 beyond its scope.
2. The cascade oracle is real at many depths. Correcting only the activation mean at layer 15 already reduces pooled final MSE to about 0.42 of baseline; correcting layer 29 reduces it to about 0.053; layer 30 reduces it to about 0.024. Therefore a “late injections make earlier repair futile” theorem is false for this ensemble.
3. Anchor accuracy is radically direction-dependent. Errors in the leading downstream-Jacobian direction are damaging at roughly one-quarter to one-half of the true-defect norm, while errors in the trailing singular direction are almost invisible even at twice the defect norm. A universal scalar center-accuracy threshold should be retired.
4. Coherent companions contain genuine absolute information, and averaging four orientations removes a large orientation-specific component. But the cost and tail behavior erase the score gain.
5. The proposed Poisson spectral controls fail catastrophically in dimension 256. Exact spherical expectation is insufficient when the finite design misses the kernel's concentrated mass.
6. Exactly anchored first-layer ReLU controls also fail frozen validation.

The strongest defensible scientific conclusion is:

> Large oracle headroom is distributed across the cascade and remains accessible to target-dependent low-dimensional source combinations. The tested legal observables do not stably reveal the required instance-specific coefficients. Independent companion evaluations reduce raw error, but the number required to average away orientation-specific bias is too expensive under the current score.

## Protocol

### First frozen campaign

- Development bases: 910001, 910003, 910009, 910019.
- Untouched validation bases: 910033, 910043, 910051, 910067.
- Three predetermined rotations per base; all rotations grouped.
- Hyperparameters selected only on the four development bases.
- Development selection hash: `23f1931a8c272ef7bf91c5bfcf5c4f9f1008236ee41d6d25341ae5a725fce74b`.

### Independent confirmation

After the first 24 cases became exposed, four candidates were frozen and tested on new bases 910079, 910081, 910089, and 910103.

Confirmation configuration hash: `8b39f633dd760e45c23d5a934bbf1bbc9849db4256c9b094bcafb56a96a3ae2c`.

## 1. Checkpoint oracle ladder

Each intervention shifts every Kerdock particle at a checkpoint by the exact reference mean defect at that layer, then performs exact nonlinear suffix replay.

### First frozen validation

| Corrected post-ReLU layer | Pooled MSE ratio | Wins | Worst |
|---:|---:|---:|---:|
| 7 | 0.584 | 10/12 | 1.161 |
| 15 | 0.424 | 12/12 | 0.952 |
| 23 | 0.218 | 12/12 | 0.674 |
| 27 | 0.103 | 12/12 | 0.335 |
| 29 | 0.0528 | 12/12 | 0.133 |
| 30 | 0.0239 | 12/12 | 0.0757 |

### Independent confirmation

| Corrected post-ReLU layer | Pooled MSE ratio | Wins | Worst |
|---:|---:|---:|---:|
| 7 | 0.605 | 8/12 | 1.313 |
| 15 | 0.415 | 11/12 | 1.047 |
| 23 | 0.200 | 12/12 | 0.506 |
| 27 | 0.117 | 12/12 | 0.346 |
| 29 | 0.0539 | 12/12 | 0.140 |
| 30 | 0.0252 | 12/12 | 0.0657 |

The near-oracle noise-corrected estimates become reference-noise limited and can turn negative, so pooled ratios are reported for the deepest checkpoints.

### Cross-checkpoint coherence

On the first frozen validation, the energy fractions of successive checkpoint-repair increments were approximately

`[0.395, 0.177, 0.235, 0.111, 0.0535, 0.0288]`.

Most off-diagonal increment cosines had magnitude below 0.10; the largest was 0.146. This supports an approximately incoherent attribution picture, but emphatically not equal per-layer contributions.

### Implication

The current estimator is already a cascade from a known input, but that does not make alternative checkpoint estimates futile. A perfect mean repair at an early or middle checkpoint removes substantial final error. The unresolved question is informational and computational: how to estimate the appropriate checkpoint defect cheaply and independently.

## 2. Structured anchor perturbations

At layer 30, the exact defect was perturbed in five directions. Ratios below are relative to the original baseline MSE.

| Added error relative to true defect norm | Actual-defect direction | Leading Jacobian direction | Trailing Jacobian direction | Random | Companion-residual direction |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0239 | 0.0239 | 0.0239 | 0.0239 | 0.0239 |
| 0.10 | 0.0340 | 0.0863 | 0.0239 | 0.0360 | 0.0348 |
| 0.25 | 0.0858 | 0.412 | 0.0239 | 0.0962 | 0.0910 |
| 0.50 | 0.270 | 1.575 | 0.0239 | 0.309 | 0.291 |
| 1.00 | 1.006 | 6.227 | 0.0239 | 1.157 | 1.089 |
| 2.00 | 3.945 | 24.826 | 0.0239 | 4.538 | 4.278 |

The trailing singular direction is practically in the nullspace of the downstream map. The leading direction is more than an order of magnitude more consequential than a generic direction at the same Euclidean norm.

### Proof consequence

Replace claims of the form

`||anchor error|| / ||state|| < universal threshold`

with a downstream-weighted quantity such as

\[
\|J_k e\|^2
\]

plus a gate-crossing remainder. Any scalar threshold is, at most, a calibration for a particular perturbation distribution.

## 3. Poisson spectral controls

The proposed symmetrized Poisson kernels have exact spherical expectation, but in dimension 256 their mass is sharply concentrated. On the finite Kerdock design:

- maximum absolute feature sample mean on validation: 14.58;
- median per-case maximum absolute sample mean: approximately 1.00;
- maximum feature RMS: 2813;
- median maximum feature RMS: 199.

The frozen best ridge produced:

- noise-corrected pooled ratio: **4.73 million**;
- 0/12 wins;
- worst pooled case ratio: **8.21 million**.

**Verdict:** close this construction at the tested radii and directions. Exact analytic expectation does not make a control useful when its empirical quadrature value has extreme leverage.

## 4. Exactly anchored first-layer controls

The tested control used exact Gaussian first-layer ReLU moments with linear and radialized quadratic features and whole-basis cross-fitting.

Frozen validation:

- noise-corrected pooled ratio: **1.221**;
- pooled wins: 4/12;
- worst ratio: 1.593;
- grouped noise-corrected interval: [1.086, 1.301].

**Verdict:** close the tested shallow-control forms. Exact shallow expectations exist, but their high-degree residual correlation with final Kerdock error is not stable enough.

## 5. Companion results

### Why averaging helps

For four companion anchor errors `E_j`, the exact decomposition

\[
\frac14\sum_j\|E_j\|^2
=
\|\bar E\|^2+rac14\sum_j\|E_j-\bar E\|^2
\]

showed, on frozen validation:

- common-bias fraction: 28.1%;
- orientation-specific spread fraction: 71.9%.

Development gave a similar common fraction of 24.2%. Thus coherent-orientation averaging removes real error rather than merely smoothing scores.

This also corrects the earlier “gauge completion” intuition. A companion is not an external measurement of its own unknown error; it is another biased estimate of the same center. Relative companion differences reveal orientation-specific bias, but one companion does not fix the absolute gauge. Averaging works by variance reduction, not by exact gauge cancellation.

### Frozen first validation

The development-selected four-companion average at scale 0.3 achieved:

- noise-corrected raw ratio: 0.638;
- 6/12 pooled wins;
- worst pooled ratio: 1.338;
- lower-bound cost ratio: 1.494;
- optimistic adjusted ratio: 0.953;
- adjusted ratio including full two-layer replay: 0.993;
- grouped interval still crossed one.

This was promising enough to justify independent confirmation, but not promotion.

### Independent confirmation candidates

| Package | Noise-corrected raw | Wins | Worst | Optimistic adjusted |
|---|---:|---:|---:|---:|
| One companion j0, scale 0.10 | 0.931 | 6/12 | 1.448 | 1.046 |
| One companion j2, scale 0.05 | 0.973 | 8/12 | 1.031 | 1.094 |
| Four companions averaged, scale 0.20 | 0.845 | 11/12 | 1.390 | 1.262 |
| Four development-weighted companions | 0.849 | 10/12 | 1.846 | 1.269 |

Every package is score-negative even under optimistic cost accounting.

Across all 36 cases, the closest optimistic package is one companion at scale 0.10:

- noise-corrected raw ratio: 0.878;
- optimistic adjusted ratio: 0.987;
- adjusted with explicit suffix replay: 1.042;
- worst pooled ratio: 1.618.

Because the package was selected after part of this corpus became exposed, has a severe tail, and loses under the actual research implementation, it is not a submission candidate.

### Poisson-assisted orientation selection

The selector used the new Poisson correction direction to rank four two-basis companion probes, then applied one selected 17-basis companion at scale 0.10.

- Development noise-corrected ratio: 0.852.
- First validation: 0.839.
- Independent confirmation: 0.905.
- All 36 cases: 0.869.
- All-36 optimistic cost ratio for 8 probe bases plus 17 selected bases: about 1.182.
- Corresponding optimistic adjusted ratio: about 1.026, before feature-regression cost.
- Worst pooled ratio: 1.553.

**Verdict:** the new selector contains some signal but is not score-positive or tail-safe. The Poisson control is unusable as an estimator and insufficient as a selector observable.

## 6. Empirical observability envelopes

Using one exact-shallow source and four companion correction sources:

### Frozen validation

- Post-hoc global linear coefficients: noise-corrected ratio 0.596.
- Per-case oracle coefficients in the same five-source span: noise-corrected ratio 0.255.
- Frozen deployable combinations were much weaker and unstable on confirmation.

This is the sharpest explanation of the gap:

> The tested source span contains enough directions to remove roughly three-quarters of the remaining error when target-dependent coefficients are revealed. The legally observable data do not produce stable instance-specific coefficients.

The obstacle is therefore not absence of a corrective subspace. It is conditional coefficient observability and the cost of obtaining an independent absolute estimate.

## 7. Proof upgrade

The exact theorem in `OBSERVABILITY_ENVELOPE_THEOREM.md` should replace the attempted universal cascade lower bound.

It permits a strong scoped statement:

> For a declared legal information set `F`, no correction measurable with respect to `F` can outperform `E[R|F]`. Nested deterministic processing of the same information cannot enlarge the envelope. New cascade stages help only to the extent that they add information about the final error.

To turn this into a numerical impossibility certificate, future work must upper-bound `H(F)` for a precisely specified feature class using independent data and complexity control. The current experiments provide empirical estimates for several finite source classes, not a universal white-box lower bound.

## 8. Competition decision

**Do not change the current production estimator based on this campaign.**

Close:

- symmetrized Poisson spectral controls at the tested settings;
- exact first-layer residual controls;
- Poisson-assisted companion selection;
- four-companion average or weighted packages under current trajectory costs;
- universal scalar anchor thresholds;
- universal “cascades cannot help” claims.

Retain:

- the checkpoint oracle ladder as strong structural evidence;
- the downstream-weighted perturbation theorem/program;
- the observability-envelope theorem;
- companion variance decomposition;
- one-companion low-strength correction only as a direct-map engineering target, not as a validated submission candidate;
- the official-data runner for a final no-retuning check if the missing assets become available.

## 9. Limitations

- This is challenge-matched width-256/depth-32 synthetic evidence, not official Mini-100 evidence.
- Reference means use 16 independent complete-Kerdock rotations, not the official 1e9-sample targets.
- The exact full-cloud suffix replay costs more than the retained contraction-based direct center map; both optimistic and replay-inclusive costs are reported.
- Near-oracle checkpoint results are reference-noise limited.
- No unconditional computational impossibility theorem is claimed.
