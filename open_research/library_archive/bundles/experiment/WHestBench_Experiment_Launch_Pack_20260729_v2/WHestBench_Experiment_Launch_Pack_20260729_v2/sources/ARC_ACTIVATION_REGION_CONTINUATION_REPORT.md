# ARC activation-region conditional integration — continuation report

**Date:** 2026-07-29  
**Decision:** **CLOSE as a deployable estimator branch**  
**Holdout governance:** the untouched 64-network high-precision holdout was not opened.

## Executive conclusion

The continuation resolves the central ambiguity left by the first activation-region study.

The layer-31 error channel really is sparse: a target-aware correction of only 12 of 256 layer-31 neurons captures **55.23%** of the full layer-31 oracle benefit and improves raw final-layer MSE by **1.7844x** on the eight-network cross-fitted screen. An observable Gaussian-gap × downstream-sensitivity ranking also identifies useful coordinates when it is granted their exact signed corrections.

But every tested deployable source of the **signed Kerdock defect** failed frozen validation or became cost-negative:

- low-dimensional conditional Gaussian region integration;
- residual-control use of that integrator;
- raw Walsh/basis-phase regression;
- affine-invariant cubic and higher phase contractions;
- cross-neuron PCA-skew and vector contractions;
- a small independent rotated-Kerdock phase pilot;
- fixed subdesigns already contained in the Kerdock cloud;
- larger learned phase models trained on 64 independent networks with a separate 18-network noisy-label development set.

The bottleneck is therefore not identifying uncertain gates or evaluating low-dimensional Gaussian orthants. It is predicting the **signed, high-degree cubature phase error** of the complete Kerdock rule. Better conditional integration does not solve that problem.

## 1. Sparse layer-31 oracle

| Coordinates | Raw-MSE gain | 95% interval | Full-oracle benefit captured | Worst candidate / baseline |
|---:|---:|---:|---:|---:|
| 4 | 1.2991x | 1.2219–1.3774 | 29.83% | 0.8693 |
| 8 | 1.5612x | 1.4540–1.6759 | 46.15% | 0.7371 |
| **12** | **1.7844x** | **1.6240–1.9318** | **55.23%** | **0.6607** |
| 16 | 2.0597x | 1.8438–2.2487 | 63.99% | 0.5870 |
| 32 | 2.9565x | 2.5073–3.4414 | 81.39% | 0.4351 |

This is a strong scientific result. It says the final error is controlled by a small coordinate set at layer 31 even though the originating cubature defects are globally distributed through depth.

The observable Gaussian-gap × final-sensitivity selector, while still using exact target corrections after selection, reached **1.1795x** at K=12 and **1.3645x** at K=24. Coordinate selection is thus not the main obstruction. Estimating the correction sign and magnitude is.

## 2. Conditional integration itself

The deployable-shaped integrator selected 0–8 influential layer-30 ReLU gates, enumerated their low-dimensional Gaussian states, treated the remaining preactivation as conditionally Gaussian, and analytically integrated the final scalar ReLU.

Direct correction failed decisively: best screen ratio **0.8967x** with a 95% interval **0.8582–0.9156**, 0/8 wins.

Using the integrator only as a residual control variate produced a small screen winner, **1.0174x**, but the exact frozen rule reversed on 24 untouched validation networks to **0.9889x** with 8/24 wins and a worst-network candidate/baseline ratio of **1.195**. Its predicted layer-31 correction had approximately zero ordinary and downstream-weighted correlation with the true defect.

## 3. Basis-phase decomposition

The complete Kerdock cloud naturally decomposes into 129 basis means. Extracting all basis means and 127 non-DC Walsh coefficients per layer-31 neuron is effectively free once the cloud has been evaluated.

A raw-Walsh Ridge model showed a tempting **1.0410x** frozen validation point estimate after an inconclusive eight-network screen. This was not accepted because:

1. screen- and validation-trained coefficient vectors had almost zero cosine similarity;
2. arbitrary Walsh labels break the affine symmetry of the basis family;
3. label permutation destroyed the effect;
4. invariant cubic, quintic, and septic contractions did not reproduce it.

A larger corpus was then generated: 64 independent training networks and 18 separate development networks, each labeled with one independent rotated complete-Kerdock layer-31 mean. Ninety-six linear and nonlinear configurations were evaluated against the untouched 24-network high-precision validation references.

The best validation point estimate among all 96 models was only **1.00056x**, and that number is optimistic because output shrinkage was selected on validation inside this diagnostic grid. Raw-Walsh Ridge reached only **0.99941x**. The earlier 4% result was therefore a small-sample phase accident, not a scalable predictor.

## 4. Symmetry-respecting global phase statistics

To ensure coordinatewise regression had not missed a joint low-rank error direction, the continuation tested:

- coordinatewise skewness and fifth moments;
- basis-label-invariant global cubic and quintic vector contractions;
- output-weighted cubic and quintic contractions;
- PCA-score skew reconstructions at ranks 1, 2, 4, 8, and 16.

All validation ratios were approximately 1.000. The largest point estimate was **1.00092x** for rank-16 PCA skew, with a worst-network ratio of **1.0266**. There is no useful symmetry-respecting phase statistic in this family.

## 5. Independent and internal phase pilots

### Independent rotated pilot

The strongest screen result used 1,032 extra points propagated to layer 31 and achieved **1.0969x**. Frozen validation reversed to **0.9296x**, with one network worsening by 81.1%.

Larger pilots were more stable:

| Extra points | Frozen validation raw gain | Minimum row-layer cost multiplier | Optimistic adjusted proxy |
|---:|---:|---:|---:|
| 1,032 | 0.9296x | 1.0151x | 1.0920x worse |
| 2,064 | 1.0394x | 1.0303x | 0.9912x |
| 8,256 | 1.0705x | 1.1211x | 1.0473x worse |

The cost estimate counts only extra point-layer work through 31 layers and omits rotation construction, packing, dense-kernel mismatch, memory traffic, and wall time. Thus even the borderline 2,064-point result has less than 1% optimistic adjusted headroom and no interval excluding no raw gain.

### Internal pilot

A fixed 516-row subset already present in the full Kerdock design appeared to gain **1.0445x** on screen. The frozen rule reversed to **0.9651x** on validation, with only 7/24 wins. Existing-cloud subdesign disagreement is not a reliable phase estimate.

## 6. Final scientific interpretation

The experiments separate three questions that had previously been conflated:

1. **Is the downstream-relevant error sparse?** Yes. Twelve layer-31 coordinates capture more than half the oracle benefit.
2. **Can uncertain-gate conditional integration approximate activation means?** Yes, locally, but that is not the scored object.
3. **Can those approximations predict complete-Kerdock signed cubature error?** No, across every tested local, basis-phase, invariant, pilot, and learned construction.

Activation-region integration attacks distributional approximation error. The remaining estimator error is dominated by a deterministic, high-degree phase of a highly symmetric cubature rule. Those errors are only weakly related.

## 7. Decision and reopening gate

**Close activation-region conditional integration and ordinary basis-phase prediction as submission research.** Do not spend more compute on larger orthant solvers, more explicit gates, deeper suffix enumeration, or further shrinkage grids.

Reopen only if a new mechanism supplies at least one of:

- a signed target-aware statistic without reference leakage;
- a companion cubature rule whose phase disagreement is available with less than roughly 1–2% true incremental compute through arithmetic sharing;
- another surviving estimator that already computes the required conditional/JVP state essentially for free.

The most defensible adjacent research direction is therefore not a better conditional integrator. It is a **shared-arithmetic companion design**: construct a second phase-sensitive estimate that reuses the protected Kerdock/Winograd prefix almost completely. The current pilot results show that independent phase information can help, but ordinary extra-point propagation costs too much.

## Reproduction inventory

The bundle contains the scripts, manifests, compact summaries, and model logs needed to reproduce or extend the continuation. Large activation/reference caches are deliberately excluded; paths and seed manifests are retained.
