# Shared-Arithmetic External Phase Estimator

## Decision

**CLOSE TESTED COMPANION.** The frozen final-gate Stein-flux (FGSF) source does not pass the preregistered 50-network gate. It is cheap and tail-safe, and its signed correlation is technically positive, but the correction is too weak and inconsistent to overcome the frozen V80 cost.

## Frozen candidate

The inherited correction vector is the unchanged V80 blockwise control:

- selector: `gaussian_gap_sens`;
- 8 standardized H3 features;
- two independent pilot bases;
- block ridge `1.0`;
- correction amplitude `0.25`.

No block-variance, fold-stability, safety-margin, ridge, shrinkage, amplitude, or pilot-agreement setting was retuned.

The sole new information source is

\[
S_j = Q_K[(h_j-m_j)\mathbf 1_{h_j>0}] - s_j\phi(m_j/s_j).
\]

For a Gaussian marginal, `E[(H-m)1(H>0)] = s phi(m/s)`, so this is exactly zero in the Gaussian null. Under a leading H3/Edgeworth perturbation with threshold `a=-m/s`, the ReLU mean defect is proportional to `a phi(a)` while this flux is proportional to `a^3 phi(a)`; therefore the signs agree. This fixes orientation analytically rather than from the test networks.

Let `c` be the raw frozen V80 output correction. The only applied phase rule is

\[
\rho = \cos(c,S), \qquad \widehat y = y_K - 0.25\,\rho\,c.
\]

Thus `sign(rho)` selects direction and `|rho|` continuously suppresses low-confidence corrections. There is no threshold and no fitted scale.

## Immutable test

- `50` newly derived width-256, depth-32 Gaussian-He networks.
- Seeds were committed by SHA-256 before opening references and checked against the supplied staged split and V80 result rows. A post-run scan of all materialized source JSON/CSV/Markdown/text/Python artifacts found zero seed overlaps.
- Truth proxy: two independent groups of eight complete Kerdock rotations; half disagreement supplies the reference-noise correction.
- Baseline: retained seed-3 complete Kerdock design.

## Results

| Metric | Result | Gate |
|---|---:|---:|
| Signed correlation, `rho` vs true optimal V80 scale | `0.058620` | positive: **pass** |
| Sign accuracy | `30/50 = 60.0%` | diagnostic |
| Mean / median `|rho|` | `0.093426` / `0.080905` | strong suppression |
| Mean predicted correction cosine | `0.065544` | weak |
| Noise-corrected raw gain | `1.002085x` | insufficient |
| Compute-adjusted gain | `0.986599x` | fail |
| Adjusted 95% network-bootstrap interval | `[0.977387, 0.997463]` | fail; entirely below 1 |
| Wins | `29/50` | diagnostic |
| Median candidate/base | `0.999189` | near neutral |
| Worst candidate/base, observed | `1.083634` | <=1.10: **pass** |
| Unphased frozen V80 raw gain | `0.911566x` | harmful |

The companion removes most of the damage from blindly applying V80—the unphased vector loses about 8.8%, whereas FGSF is nearly neutral—but it does so mainly by shrinking every correction. All 50 networks had `|rho|<0.25`; confidence was not monotone, and the highest-`|rho|` quartile had only 38.5% sign accuracy.

## Compute

- New trajectories for FGSF: **0**.
- FGSF arithmetic estimate: `118,360,064` operations, or `0.0674%` of the retained baseline effective compute.
- Frozen V80 plus FGSF arithmetic increment: `1.5696%`.
- Optimized exact-equivalent implementation: one H32 sum-of-squares reduction, with existing V80 `base`, `p32`, and `mean(A31)` reused. On ten networks it took `0.01676` seconds on average, `0.676%` of baseline wall.

The original direct NumPy expression formed large float64 temporaries and is retained only as a transparent test implementation. The production helper in the source file implements the algebraically equivalent fused form. Reference rotations are evaluation-only and are not included in candidate compute.

## Gate

```json
{
  "incremental_compute_le_2pct": true,
  "adjusted_interval_excludes_no_gain": false,
  "positive_signed_correlation": true,
  "worst_candidate_over_baseline_le_1_10": true,
  "overall_pass": false
}
```

The overall gate fails because the adjusted interval does not exclude no gain in the favorable direction. In fact, the entire interval is below break-even.

## Interpretation

The final marginal contains a real, mathematically oriented H3/Stein discrepancy, but that discrepancy does **not** identify the network-specific phase of the frozen V80 vector. The statistic is dominated by near-zero cosines and has negligible correlation with the required scale. This closes the specific family “final-preactivation marginal Stein flux used to phase V80,” not all possible shared-arithmetic phase sources.

A future reopening should require structurally different information—such as a non-marginal cross-output contraction or a prefix/suffix adjoint source—not another scalar final-marginal H3 transform or a retuned confidence threshold.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python shared_arithmetic_external_phase.py --init

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python shared_arithmetic_external_phase.py --all --workers 2

python verify_shared_phase_results.py
```

## Scope

This is an architecture-matched synthetic width-256 test, not an official Mini-100/FlopScope submission. The statistical decision uses high-precision complete-design references and the frozen arithmetic cost model.
