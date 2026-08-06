---
title: "Complete Kerdock Cubature Is Near-Optimal for Nonnegative Static Rules"
subtitle: "Two-page overview for external mathematical review"
author: "Skye Nygaard"
date: "August 2, 2026"
---

# The problem

For a bias-free ReLU network and Gaussian input, positive homogeneity gives the exact reduction

\[
\mathbb E[f(X)] = \mathbb E[\|X\|_2]\,\mathbb E[f(U)],
\qquad U\sim\mathrm{Unif}(S^{255}).
\]

The numerical problem is therefore spherical integration. The baseline uses a complete maximal collection of 129 real mutually unbiased bases in dimension 256: 128 Kerdock/Walsh–Hadamard chirp bases plus the coordinate basis. Evaluating every vector and its antipode gives 66,048 spherical nodes.

The central question is deliberately scoped:

> At this node budget, how much can another **static, network-independent linear cubature rule** improve the expected integration risk for the dimension-256, depth-32 limiting normalized-ReLU kernel?

The claim does not cover finite-width, adaptive, nonlinear, or network-dependent estimators.

# Main results

## 1. Nonnegative rules are almost completely closed

Among arbitrary-node, nonnegative, mass-one static rules with at most 66,048 nodes, a computer-assisted Delsarte/RKHS certificate proves that complete Kerdock is at most

\[
1.0002332417295004
\]

times the infimum risk: a relative excess below **0.0233242%**.

The auxiliary optimization is completed across all harmonic degrees. Its unique optimum is a degree-five Hermite interpolant of the deep-ReLU zonal kernel at three algebraic contact points. Exact Gaussian quadrature and strict reduced-cost inequalities eliminate all higher Gegenbauer degrees.

## 2. Signed weights do not create a large hidden opportunity

For arbitrary real signed weights with total mass one, every static rule in the same node budget satisfies

\[
R(Q)\ge0.9370601683665084\,R(Q_{\mathrm K}),
\]

so the fixed-node-budget Kerdock-to-rule risk factor is at most

\[
1.0671673322143325\times.
\]

This is equivalent to at most a **6.2940% reduction in Kerdock risk**; it is not an equal-wall-time theorem.

The new ingredient is atomic inertia. If a symmetric moment matrix has trace \(T>0\) and at most \(p\) positive eigenvalues, then

\[
\|M\|_F^2\ge T^2/p.
\]

For an atomic signed rule, the number of positive eigenvalues of the moment matrix is at most the number of positive weights. This strengthens the earlier rank/block-trace relaxation.

The audited release does not use the earlier arbitrary-total-mass corollary because its standalone witness was not recovered. A frozen-witness sign-count hierarchy shows that, after consolidating duplicate locations and removing zero weights, at least 1,072 negative-weight support entries rule out a 1.05-fold Kerdock-to-rule factor, while at least 4,160 make the rule worse than Kerdock. This is a count statement, not a lower bound on total negative mass.

# Equality, sharpness, and the remaining theorem gap

The rank plus individual harmonic block-trace relaxation is exactly sharp over abstract symmetric matrices, even when all profiles share one common abstract Gram matrix. Stronger bounds must therefore use the fact that the matrices arise from spherical point evaluations.

For an actual atomic rule, equality forces:

- exactly \(N\) active nodes;
- equal positive weights;
- a pairwise zero-code condition for the relevant feature kernel.

Two positive profiles in the released certificate would require common zeros of consecutive Gegenbauer polynomials, which is impossible. Thus the older abstract floor is strictly unattainable by an actual atomic rule. However, signed rules can use coalescing positive/negative atoms with unbounded total variation, so strict nonattainment alone does not yield a uniform numerical gap without a conditioning assumption.

# Evidence and review request

The release contains the proof drafts, corrected v5.2 theorem record, all-degree verifier, second T16 interval-stack result, endpoint certificate, original signed witness, conservative inertia/sign-count verifier, equality proofs, canonical ledger, CSV exports, and claim manifest. The core downstream witnesses replay from a clean checkout. The main remaining publication dependency is an independent Arb/FLINT/MPFR reconstruction of the full inherited T22/kernel coefficient interval stack, plus named human review of the Bell/Hermite bridge. The T16 primal numerics already have a second implementation, but this does not close the full T22 trust boundary.

I am asking for bounded feedback on four questions:

1. Are the theorem statements and scope correct and natural?
2. Is “essentially optimal for the nonnegative static class, with a tight signed extension bound” justified by the two-tier result?
3. Are the nonnegative auxiliary theorem and signed inertia argument presented at publication quality?
4. What venue and minimum additional verification would make the result publishable?

The companion paper and open ledger document the experimental search and are not required to assess the core theorem.
