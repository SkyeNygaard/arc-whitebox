# Weighted harmonic certificate search audit

## Purpose

Document the discovery chronology behind T47 without confusing exploratory optimization with the final exact certificate.

## Starting point

The unweighted degree-`<=3` T43 certificate gave:

- signed MSE floor: `1.7017556669835916e-8`;
- fraction of Kerdock MSE: `0.0699257668273`;
- permitted improvement: `14.30088x`.

The abstract proof suggested replacing the identity harmonic covariance by a diagonal weighted covariance.

## Search protocol

For a finite degree cutoff `L`, exploratory floating-point search varied nonnegative harmonic weights and maximized

\[
\min_r\frac{k_r}{b_r(a)}F_N(A(a)).
\]

Every numerical candidate was treated only as a discovery hint. Promotion required:

1. terminating-decimal rational weights;
2. exact rational expansion of `L_a^2`;
3. checking every active nonconstant degree;
4. directed lower intervals for every needed `k_r`;
5. exact rank-defect arithmetic;
6. a fresh verifier rerun from the frozen constants.

## Discovery milestones

| Candidate | Exploratory / certified cap |
|---|---:|
| T43 unweighted degrees 0–3 | 14.30088x certified |
| Weighted through degree 9 | 2.84458x exploratory; rationally certifiable |
| Weighted through degree 14 | 2.06132x certified during search |
| Weighted through degree 15 | **1.979503722x certified and frozen** |

## Stopping rule

After the degree-14 result approached two, the continuation froze this rule:

> Stop at the first exact-rational, all-active-degree certificate below 2x, or after degree 16 if the threshold is not reached.

Degree 15 crossed the threshold, so no degree-16 or higher search was promoted or continued.

## Final certificate

The canonical T47 weights are in `T47_WEIGHTED_HARMONIC_RANK_FLOOR.md`. The frozen and independently recomputed results are:

- `results/weighted_rank_floor_degree15_frozen.json`;
- `results/weighted_rank_floor_degree15_recomputed.json`.

The historical degree-9 and degree-14 files are retained under `exploratory_weight_search/` for provenance only.

## Claim boundary

The search does not prove that the frozen weights optimize the weighted comparison program. T47 proves only the bound furnished by this explicit candidate. Further optimization is closed for this package; any future continuation should state a formal optimization problem and a new stopping rule before search.
