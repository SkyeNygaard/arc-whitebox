# WHestBench Oracle proof completion report — v18

**Date:** 2026-07-30  
**Canonical predecessor:** v17 project closeout  
**Protected cohort:** not opened

## Executive result

The continuation converts the Oracle/circularity program from a mostly empirical phase narrative into a sharper theorem hierarchy.

The strongest new result is **T47**, a weighted harmonic rank obstruction for arbitrary signed static cubature. For the dimension-256, depth-32 limiting ReLU kernel, every network-independent mass-one linear rule with at most 66,048 arbitrary nodes and arbitrary real weights satisfies

\[
R(Q)\ge1.2294295437956858\times10^{-7}
=0.505177125470747\,R_{\mathrm{Kerdock}}.
\]

Thus no rule in this class can improve over complete Kerdock by more than

\[
1.979503722\times.
\]

This is a major strengthening over the initial unweighted signed floor, which permitted 14.30088x. The bounded degree-15 extension crossed the prespecified 2x stopping threshold. It remains a computer-assisted limiting-kernel result, not signed near-optimality and not a finite-width/adaptive theorem.

The second major advance is an exact phase framework:

- T44 quantifies how transcript information bounds phase correction value;
- T45 turns approximate symmetry into a measurable alignment ceiling;
- T46 proves that orientation-blind quotient representations cannot define nonzero signed coefficients.

These results do not resurrect the invalid universal observability theorem. They state precise obligations for any scoped application.

## 1. Theorem sequence

| ID | Result | Status | Strongest implication |
|---|---|---|---|
| T42 | `K_32-h_*` is positive definite | Computer-assisted candidate | Enables coefficientwise signed comparisons |
| T43 | Unweighted arbitrary-signed rank floor | Abstract proof + candidate constant | Improvement at most 14.30088x |
| T44 | Binary/sequential/action phase information bounds | Proved abstractly | Correction fraction at most `2I` in pure phase model |
| T45 | Symmetry-defect alignment bound | Proved | Gain bounded by squared symmetry defects |
| T46 | Gauge-invariant coefficient obstruction | Proved for representation | Orientation-blind coefficient policy must vanish under full sign gauge |
| T47 | Weighted harmonic arbitrary-signed floor | Abstract proof + candidate constant | Improvement at most 1.979503722x |

### Numerical trust boundary

T42, T43 and T47 use an order-47 directed `mpmath.iv` Taylor jet plus exact SymPy rational harmonic projection. The package reruns cleanly and checks every active coefficient. Publication requires a genuinely independent interval implementation, preferably Arb/FLINT or direct MPFR, and named human review.

## 2. Why T47 is stronger

The original rank proof used equal feature weights on degrees 0 through 3. T47 allows a diagonal harmonic covariance

\[
A=\operatorname{diag}(a_\ell I_{d_\ell}).
\]

For every rank-`N`, trace-preserving signed moment matrix, the exact best possible Frobenius approximation error is

\[
F_N(A)=\sum_{j>N}\lambda_j^2+
\frac1N\left(\sum_{j>N}\lambda_j\right)^2.
\]

A frozen degree-15 weighting balances the ReLU-kernel coefficient constraints against the enormous multiplicities of higher harmonic spaces. All nonconstant coefficients of `K-gamma L_a^2` are certified nonnegative through degree 30, with degree 7 binding.

No open-ended numerical search remains active. The degree-15 rational vector is frozen; further strengthening should formulate a principled optimization problem and recertify a prespecified solution.

## 3. Oracle coherence correction

The authenticated continuation arrays show pooled checkpoint-increment effective rank around 3.8–4.3 and modest pooled cross terms. This rejects a single common pooled direction. It does **not** prove that a typical network has several independent components: cross-network sign cancellation can make pooled increments orthogonal even when each network has scalar output.

Canonical status:

> Pooled checkpoint heterogeneity is replicated; within-network increment rank remains open.

The next diagnostic must report per-network Gram matrices or a common transported-mode decomposition.

## 4. Phase and representation result

The exact M153 feature map contains only norms, cosines, norm ratios and angle magnitudes. It is invariant under simultaneous reversal of all represented candidate directions. Signed coefficient targets are orientation-dependent.

T46 therefore proves a representation-level obstruction: an exactly orientation-blind, representation-consistent coefficient architecture cannot output nonzero signed coefficients under the full sign gauge. This explains why “more learning” over the same quotient features is not a principled continuation.

The constructive escape is one canonical orientation-odd feature. A full preregistration is included. It may run only around a source basis with independent Oracle gain at least 1.20x.

## 5. Empirical disposition

- Five-source ridge synthesis failed grouped development and is closed in its exact class.
- The Edge-DWS source ceiling was 1.144709x, below its 1.15x gate; the source is stopped, not the model class.
- Existing Poisson, projected-ReLU, T4, companion and global signed-probe families remain closed in their tested forms.
- No protected or official cohort was opened.

## 6. Updated priority order

1. Independent T42/T47 interval stack.
2. Named human proof review of weighted rank approximation and normalization.
3. Manuscript integration with front-page estimator-class scope matrix.
4. Per-network checkpoint increment-rank recovery.
5. One frozen orientation-odd phase experiment, only if its source ceiling passes.
6. Finite-width low-degree coefficient intervals to instantiate T47's finite-width schema.
7. Reproducibility repair for external empirical dependencies.

## 7. Canonical conclusion

> Static positive cubature is certified near-optimal in its limiting-kernel class, and static arbitrary-signed cubature now has a global lower bound within a factor 1.979504 of complete Kerdock. Oracle replay still exposes large finite-width repair capacity, but phase exploitability is class-specific: orientation-blind coefficient representations are exactly obstructed, while unrestricted adaptive and nonlinear white-box methods remain open. No tested deployable branch clears its continuation gate.
