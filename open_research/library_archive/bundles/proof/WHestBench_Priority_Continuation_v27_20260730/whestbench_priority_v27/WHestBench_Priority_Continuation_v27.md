# WHestBench priority continuation v27

**Date:** 2026-07-30  
**Protected data opened:** No  
**Primary source:** Agent 8 adaptive direct-output basis-PCA source  
**Disposition:** Stop the linear checkpoint-gauge continuation; transfer effort to analytic late absolute innovations or a scoped lower bound.

## Executive result

A new Agent 4 package appeared during this run and independently completed the frozen four-partition source-specific SOCP, structured orthonormal-basis audit, and representative dense-partition check. Those results are treated as prior work, not claimed again here.

The non-overlapping continuation closes the remaining checkpoint-depth loophole in two stages:

1. an exact terminal-innovation lower bound identifies where theoretical capacity first appears;
2. a dual-feasible empirical certificate tests the complete all-layer linear checkpoint chain rather than a selected partition menu.

The result is decisive. The full 31-block all-layer empirical SOCP fails on both a difficult and an unusually favorable confirmation case. On `seed910081_rot31013`, two independent 2,048-pair covariance draws give dual lower adjusted-score bounds **7.0893** and **6.9379**, versus the required **0.230415**. On the easier `seed910081_rot31033`, the dual lower bound is **1.2078**. These are lower bounds on the empirical optimum, not fitted-policy scores.

## 1. Cross-agent nonduplication audit

### Already completed elsewhere

- Agent 4: exact reconstruction of the adaptive direct-output source; four-partition fit/validation tournament; selected-partition primal/dual certificate; orthonormal-basis escape audit; representative dense partition.
- Agent 5: exact transported absolute-innovation decomposition; late-layer concentration; curvature, Walsh, and oracle-magnitude phase falsification.
- Agent 1: infinite Hermite-factor positivity program.
- Agent 2: width-256 moment-dual program and general checkpoint-gauge SOCP theorem.

### Added here

- exact terminal-innovation lower bound specialized across checkpoints 1, 4, 8, 16, 24, 27, 29, 30, and 31 on all 12 confirmation cases;
- complete confirmation audit of late single-checkpoint partitions `[1,29,32]`, `[1,30,32]`, and `[1,31,32]` under favorable oracle covariance;
- full all-layer `[1,2,...,32]` empirical SOCP dual certificates on confirmation cases, including an independent 2,048-pair replication.

## 2. Frozen source

The source is built before target loading from the 129 natural Kerdock/axis group output means. The adaptive rank is the smallest rank among the first 40 output PCA modes reaching cumulative energy `0.9939595959595959`.

On confirmation:

- rank range: 34–38;
- pooled source-only oracle residual ratio: **0.0748607100593**;
- source construction and final replay are target-free and output-linear.

The source has ample capacity. Estimating its signed contractions is the bottleneck.

## 3. Terminal-innovation frontier

For a telescope whose latest preterminal checkpoint is `t`, every earlier block is granted free. The remaining terminal innovation alone yields the following pooled lower-bound proxy:

| Latest checkpoint | RMS root difficulty lower bound | Score lower-bound proxy | Cases ruled out / 12 |
|---:|---:|---:|---:|
| 1 | 1.227161 | 2.252304 | 12 |
| 4 | 0.960385 | 1.522737 | 12 |
| 8 | 0.778557 | 1.107050 | 12 |
| 16 | 0.532125 | 0.649204 | 12 |
| 24 | 0.304161 | 0.333816 | 11 |
| 27 | 0.208411 | 0.232341 | 8 |
| 29 | 0.141181 | 0.172049 | 7 |
| 30 | 0.101284 | 0.140543 | 3 |
| 31 | 0.049166 | 0.104182 | 1 |

The competition target is `0.2304147465`.

**Interpretation:** a linear gauge ending at or before checkpoint 27 is already pooled-noncompetitive before charging any earlier term. The transition to possible terminal capacity occurs only around checkpoints 29–31.

## 4. Late single-checkpoint SOCP

The remaining single-intermediate programs were solved on all 12 confirmation cases with favorable oracle covariance and a spectral pseudoinverse that favors the candidate:

| Partition | Pooled oracle S | Adjusted-score proxy | Passing cases |
|---|---:|---:|---:|
| `[1,29,32]` | 1.140742 | 2.000383 | 0/12 |
| `[1,30,32]` | 1.136536 | 1.988503 | 0/12 |
| `[1,31,32]` | 1.125700 | 1.958061 | 0/12 |

Thus moving the checkpoint arbitrarily close to the output reduces the terminal innovation but makes the earlier checkpoint mean expensive enough that the complete two-block estimator still misses by roughly a factor of eight in adjusted score.

## 5. Complete all-layer dual certificate

The final loophole is that many small checkpoint bridges might outperform every single bridge. I therefore solved the empirical SOCP over every checkpoint `1,2,...,32` and constructed a dual witness by exact checkpoint-balance projection.

### Confirmation case `seed910081_rot31013`

| Covariance draw | Pairs | Allowed S | Dual lower S | Dual score lower bound | Max stationarity residual | Max ball ratio |
|---|---:|---:|---:|---:|---:|---:|
| A | 2,048 | 0.154512 | 2.337066 | 7.089275 | 1.17e-12 | 1.0 |
| B, independent | 2,048 | 0.154512 | 2.308478 | 6.937857 | 1.09e-11 | 1.0 |

### Favorable case `seed910081_rot31033`

| Pairs | Source ratio | Allowed S | Dual lower S | Dual score lower bound | Max stationarity residual |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 0.031390 | 0.302843 | 0.921829 | 1.207802 | 1.21e-12 |

The dual norm constraints are satisfied exactly after global scaling, and every dual objective is below its corresponding primal objective. Primal convergence is therefore not needed for the stop conclusion.

This is stronger than a representative dense-partition failure: it covers all 31 possible linear checkpoint bridges simultaneously for the empirical covariance program.

## 6. Synthesis with parallel work

Agent 4 independently found that the selected `[1,4,32]` rule validates at `S=1.274291533`, that its empirical dual lower bound is `1.155929800`, and that the structured orthonormal-basis escape still validates near `S=1.032107`. The all-layer certificates here show that adding every checkpoint does not expose a hidden linear-gauge rescue on the tested confirmation cases.

Agent 5 independently identifies the true target-informed mechanism:

\[
\delta_{31}=\sum_{\ell=0}^{31}\xi_\ell R_\ell,
\qquad
\xi_\ell=(P-Q)|z_\ell|,
\]

with layers 28–31 carrying most useful energy. Combined with the current result, the frontier is no longer “find a better checkpoint control.” It is:

> derive a cheap absolute estimator for one or more signed late innovations, or prove that such estimation cannot fit the legal compute budget.

## 7. Remaining priority paths

### Constructive

1. **Analytic late absolute innovation.** Seek an identity for `(P-Q)|z_l|`, especially `l=28..31`, using baseline arithmetic, row exchangeability, a conditional-Gaussian reduction, or a shared radial/angular transform.
2. **Joint innovation before coordinates.** Estimate a transported combination `sum xi_l R_l` directly; do not estimate 34–38 PCA coefficients separately.
3. **Exact structured integration with no new full passes.** A candidate must exploit already-computed design values or a sublinear shared transform. Independent pilots and ordinary checkpoint sampling are economically closed.

### Impossibility

1. Build a weight-aware computational lower bound for late absolute integration under the actual architecture and query budget.
2. Construct two architecture-valid instances agreeing on the permitted low-cost transcript while differing materially in a late transported innovation.
3. Quantify population transfer of the all-layer dual using covariance concentration if a publication-level class closure is desired.

## 8. Scope and honesty

- The terminal-innovation inequality is exact.
- The late single-checkpoint and all-layer results are empirical covariance audits on exposed synthetic confirmation cases.
- The all-layer dual certificates are exact for their empirical SOCPs and replicated across independent covariance draws.
- They do not close nonlinear, biased, same-sample, or analytic identities outside the independent-block linear checkpoint-gauge class.
- No protected or official cohort was opened.
