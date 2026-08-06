# Proposed ledger patch — Prompt 1 weighted signed floor

Use the next free canonical theorem/evidence IDs at merge; the labels below are provisional.

## T55 — Exact all-rank trace-preserving obstruction

- **Family:** Static signed arbitrary-node theory
- **Statement:** For a positive semidefinite covariance operator with eigenvalues \(\lambda_1\ge\lambda_2\ge\cdots\), the minimum Frobenius error over symmetric matrices of rank at most \(r\) and the same trace is
  \[
  F_r(A)=\sum_{j>r}\lambda_j^2+\frac1r\left(\sum_{j>r}\lambda_j\right)^2.
  \]
  The minimizer is positive semidefinite, uses the top-\(r\) eigenspace, and rank \(r\) is always weakly better than every smaller rank. Thus indefinite signed moment matrices and accidental rank loss cannot evade T47; using \(r=N\) is conservative.
- **Evidence:** Exact analytic proof in `PROMPT1_WEIGHTED_SIGNED_FLOOR_REPORT.md`.
- **Verdict:** Canonical strengthening of the weighted-rank lemma.

## T56 — Degree-47 CP-mixture dual ceiling

- **Family:** Weighted signed-floor optimization
- **Population:** Dimension 256, depth-32 infinite-width normalized ReLU kernel; node budget 66,048.
- **Declared comparison class:** Any finite or countable sum of squared rotation-invariant harmonic kernels supported on degrees 0–47, with nonnegative harmonic weights; equivalently a completely-positive mixture of degree-47 weight vectors.
- **Result:** The best weighted-rank floor in this class is bounded by
  \[
  0.79101228910008758459\,R_K
  \le \beta_{47}^{\rm CP}
  \le 0.87671080553396886557\,R_K.
  \]
  The lower endpoint is D47. The upper endpoint is a directed entrywise dual certificate with objective factor 1.10834030977113904366 over D47.
- **Implication:** Target A (0.769231) is achieved. Target B (0.909091) is impossible in this declared degree-47 CP comparison class. This does **not** rule out a stronger degree cutoff, an infinite comparison kernel, non-zonal/non-CP constructions, finite-width methods, or adaptive/nonlinear algorithms.
- **Evidence:** `PROMPT1_DEGREE47_ENTRYWISE_DUAL_CERTIFICATE.json`, `PROMPT1_DEGREE47_DUAL_VERIFICATION.json`, direct-C MPFR order-511 jet, exact Fraction harmonic algebra.
- **Verdict:** New theorem; publication-ready only after human review and an additional independent interval engine.

## C/QA — Quarantined false shortcut

- **Claim:** An order-95 Taylor jet supplies upper bounds on all degree-1–94 kernel coefficients.
- **Verdict:** False. Truncation is automatically a lower bound because the Maclaurin coefficients are nonnegative. It becomes an upper bound only after bounding the omitted tail. The first optimistic 0.851256 ceiling was invalid and is quarantined.
- **Replacement:** Direct-C MPFR order-511 jet plus a rigorous Maclaurin-tail mass and exact monomial-projection supremum bound. The corrected ceiling is 0.8767108055339689.

## C/QA — Simple PSD dual does not certify D47

- **Claim:** The active D47 KKT multipliers should directly produce a positive-semidefinite quadratic dual slack.
- **Verdict:** False in the balanced low-degree slice. The unique positive multiplier vector gives an indefinite slack (minimum eigenvalue about -0.00604 in normalized double-precision diagnostics). A chamber-aware or stronger cone is required.
- **Replacement:** The successful global upper certificate uses entrywise nonnegative domination, which is valid on the completely-positive cone.

## Next action

Move the same primal/dual machinery to cutoff 63 or 79. The degree-47 ceiling proves that reaching the 1.10 target requires leaving the degree-47 CP class; it does not identify whether higher finite cutoffs or an infinite factorization can cross it.
