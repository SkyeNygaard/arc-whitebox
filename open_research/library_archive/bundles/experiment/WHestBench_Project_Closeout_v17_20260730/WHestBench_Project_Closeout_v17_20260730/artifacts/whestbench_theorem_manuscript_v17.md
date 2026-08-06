# A Certified Boundary for Static Deep-ReLU Cubature

**Closeout manuscript — 30 July 2026**

## Abstract

We study spherical integration for a dimension-256, depth-32 normalized ReLU random-network ensemble. For the corresponding infinite-width zonal kernel, we give a computer-assisted certificate showing that the uniform 66,048-point real-MUB/Kerdock rule is at most **0.023324172950039%** above the infimum over all static, network-independent linear cubature rules using at most 66,048 spherical nodes with nonnegative weights summing to one. The associated degree-5 Hermite minorant is computer-assisted certified to be the unique optimizer of the admissible all-degree Delsarte auxiliary program. Within the fixed 33,024-line Kerdock universe, arbitrary real line weights and deletions are optimized exactly by complete orthonormal bases plus at most one partial basis. We further show that this fixed-support result extends to finite-width Gaussian-first-layer ensembles under an explicit even-Hermite nondegeneracy condition, and that uniform mass-one weights minimize ensemble risk on the complete support at every width by symmetry. These theorems do not cover arbitrary-node finite-width rules, off-support signed weights, network-adaptive supports or weights, nonlinear estimators, or network-dependent analytic-plus-residual methods. We state those exclusions alongside the results and provide counterexamples showing why they cannot be inferred from the static certificate.

## 1. Introduction

White-box neural expectation estimation permits an algorithm to inspect a network and choose how to approximate its Gaussian or spherical expectation. A highly structured real-MUB/Kerdock rule is empirically strong at the target dimension and budget, but its success raises two distinct questions:

1. How much improvement remains within static linear cubature?
2. Which additional geometry or runtime information is required to escape that static class?

This paper answers the first question for an infinite-width deep-ReLU kernel and for several exact fixed-support models. It does not claim that adaptive or nonlinear white-box estimation is impossible.

### Contributions

1. **Arbitrary-node static certificate.** A one-sided computer-assisted lower bound for every network-independent nonnegative mass-one rule with at most 66,048 nodes.
2. **All-degree auxiliary optimum.** The degree-5 Hermite minorant at three algebraic contacts is the unique optimizer of the full admissible auxiliary LP.
3. **Fixed-support real-weight theorem.** Arbitrary real line weights in the fixed Kerdock/MUB universe are optimized by complete bases plus at most one partial basis.
4. **Finite-width fixed-support extension.** Gaussian noise-stability structure proves the same allocation theorem for finite-width ensembles under explicit nondegeneracy.
5. **Scope boundary.** Constructive counterexamples show why none of these results implies arbitrary signed-node, adaptive, nonlinear or transformed-residual optimality.

## 2. Scope matrix

- **Arbitrary-node near-optimality:** infinite-width kernel; any spherical nodes up to 66,048; nonnegative mass-one weights; network-independent linear rules; computer-assisted certified.
- **All-degree auxiliary LP:** named Gegenbauer certificate; nonnegative nonconstant coefficients; unique optimum; computer-assisted certified.
- **Kerdock-line allocation:** fixed 33,024 symmetrized lines; arbitrary real mass-one weights; static linear rules; proved at infinite width and, under explicit assumptions, finite width.
- **Complete-support uniform weights:** complete fixed Kerdock support; mass-one fixed linear weights; proved at every width under the rotationally invariant ensemble model.
- **Signed stability:** limiting kernel; arbitrary consolidated signed measures with bounded negative mass; proved but quantitatively weak.
- **Arbitrary-node finite-width T22:** open.
- **Adaptive and nonlinear methods:** outside the certified static class and open.

A symmetrized projective line ordinarily corresponds to evaluating both antipodal points. Line budgets and point-evaluation budgets are therefore kept distinct.

## 3. Kernel-risk reduction

Let `P` be normalized spherical measure and let

\[
Qf=\sum_{i=1}^{m}w_if(x_i),\qquad \sum_iw_i=1.
\]

For a rotationally invariant random field with second-moment kernel `K(<x,y>)`, any rule independent of the realized field has ensemble mean-squared error

\[
R_K(Q)=\iint K(x,y)\,d(P-Q)(x)d(P-Q)(y).
\]

For a zonal kernel this becomes a weighted kernel-energy discrepancy. Network independence is essential: if the support or weights are chosen from the realized network, the fixed-kernel identity does not compare the resulting joint law without additional conditioning.

## 4. Arbitrary-node nonnegative certificate

Write the normalized Gegenbauer expansion

\[
K_{32}(t)=\sum_{\ell\ge0}a_\ell G_\ell^{(256)}(t).
\]

For any auxiliary function

\[
h(t)=\sum_{\ell\ge0}c_\ell G_\ell^{(256)}(t),\qquad c_\ell\ge0\ (\ell\ge1),\qquad h\le K_{32},
\]

positive definiteness and nonnegative weights imply

\[
E_K(Q)\ge c_0+[K_{32}(1)-h(1)]\sum_iw_i^2
\ge c_0+\frac{K_{32}(1)-h(1)}{66,048}.
\]

### Theorem 1 — one-sided static near-optimality

For the dimension-256, depth-32 infinite-width normalized ReLU kernel, the complete 66,048-point Kerdock/MUB rule has ensemble MSE at most

\[
1.0002332417295003899
\]

times the infimum among network-independent nonnegative mass-one linear rules using at most 66,048 nodes. Equivalently, its relative excess lies in

\[
[0,\ 0.023324172950039\%].
\]

The lower endpoint is zero: the theorem does not prove strict suboptimality.

## 5. The all-degree auxiliary optimum

Let `t_1,t_2,t_3` be the roots of

\[
22102t^3+21930t^2-87t-85=0.
\]

Let `h_*` be the degree-5 Hermite interpolant satisfying

\[
h_*(t_j)=K_{32}(t_j),\qquad h_*'(t_j)=K_{32}'(t_j).
\]

### Theorem 2 — unique auxiliary optimizer

The polynomial `h_*` is feasible, all five nonconstant normalized-Gegenbauer coefficients are strictly positive, and it uniquely maximizes the all-degree auxiliary objective over finite admissible expansions and over absolutely convergent admissible nonnegative expansions under the stated convergence conditions.

### Proof outline

1. An exact positive three-node quadrature matches the auxiliary objective moments through degree five.
2. Exact integer recurrence plus an analytic Gegenbauer tail proves every reduced cost of degree at least six is strictly negative.
3. Directed interval arithmetic proves `K_32^(6)(t)>0` for `-1<t<1`.
4. The generalized Hermite remainder yields

\[
K_{32}(t)-h_*(t)=\frac{K_{32}^{(6)}(\xi_t)}{6!}\prod_{j=1}^{3}(t-t_j)^2\ge0.
\]

5. An interval linear-system certificate proves positivity of the nonconstant Gegenbauer coefficients.
6. Contact and exact moment matching give primal-dual equality. Strict reduced costs and the six Hermite conditions give uniqueness.

This theorem optimizes the lower-certificate program; it does not construct a cubature rule attaining the lower bound.

## 6. Fixed Kerdock-line universe

Let `A`, `O` and `C` denote the line-kernel values for identical lines, orthogonal lines in one basis, and lines from distinct mutually unbiased bases. For line weights `w_bi` and basis totals `S_b`, risk has the exact form

\[
R(w)=\text{constant}+(O-C)\sum_bS_b^2+(A-O)\sum_{b,i}w_{bi}^2.
\]

When

\[
A-O>0,\qquad O-C<0,\qquad (A-O)+d(O-C)>0,
\]

Cauchy–Schwarz and convexity show that a budget of `P` active lines is optimally allocated as complete bases plus at most one partial basis, with equal positive weights within each active basis and positive analytic basis masses.

### Theorem 3 — fixed-support arbitrary-real-weight optimum

The sign conditions hold for the depth-32 limiting ReLU kernel. Hence the complete-basis/one-partial-basis allocation is globally optimal over arbitrary real mass-one weights on the fixed Kerdock-line universe.

## 7. Finite-width extension

Consider an ensemble of the form

\[
Y(x)=F_Z(Wx),
\]

where the first-layer rows of `W` are independent standard Gaussian vectors, `Z` is independent later randomness, and `F_Z` is square integrable. Conditioning on `Z` and expanding in multivariate Hermite polynomials gives

\[
K_m(t)=\sum_{n\ge0}a_nt^n,\qquad a_n\ge0.
\]

The antipodally symmetrized line kernel contains only even powers. If its even component is nonconstant and some even Hermite coefficient of degree at least four is positive, then the three MUB association sign conditions hold strictly.

### Theorem 4 — finite-width fixed-support extension

Under the preceding Gaussian-first-layer and nondegeneracy conditions, Theorem 3 holds for the exact finite-width ensemble MSE at every finite width and depth.

This does not provide an arbitrary-node finite-width lower certificate or the absolute finite-width Kerdock MSE.

## 8. Complete-support fixed weights at all widths

If the complete Kerdock Gram matrix has constant row sum and rotational invariance makes the integral cross-covariance constant, write `w=w_0+v`, where `w_0` is uniform and `1^Tv=0`. Then

\[
R(w)-R(w_0)=v^TGv\ge0.
\]

Thus uniform mass-one weights minimize ensemble risk on the complete support at any width. Uniqueness requires positive definiteness on `1^perp`. If mass is unconstrained, the optimum is generally a scalar multiple of the uniform vector.

## 9. Scope guards and counterexamples

1. **Nonlinear processing.** For `f_a(u)=ReLU(a^Tu)`, antipodal evaluations on one orthonormal basis reveal `|a_i|`; nonlinear aggregation recovers `||a||` and hence the exact spherical integral. A static linear theorem cannot imply all-algorithm optimality.
2. **Network-dependent support.** A rule allowed to use the realized integrand can place nodes or construct a control tailored to that realization; T22 excludes this dependence.
3. **Residual transformation.** For an exact-integral surrogate `g_theta`, the transformed problem has residual kernel `K_res`. The original Kerdock certificate need not persist unless the residual kernel is recertified.
4. **MUB sign condition.** Positive definiteness alone does not force complete-basis concentration; a positive-definite degree-4 zonal kernel can reverse the association sign and favor balanced partial bases.
5. **Off-support signed weights.** The general negative-mass stability lemma permits nontrivial improvement at extremely small negative mass and therefore does not close arbitrary signed nodes.

## 10. Related work

The proof uses the spherical-design and linear-programming framework of Delsarte, Goethals and Seidel; the energy-optimality perspective of Cohn and Kumar; Kerdock/MUB line geometry from Calderbank, Cameron, Kantor and Seidel; Schoenberg's harmonic characterization of spherical positive-definite kernels; arc-cosine and infinite-width neural-kernel results of Cho–Saul and Lee et al.; and the kernel/probabilistic-integration viewpoint surveyed by Briol et al. and Kanagawa et al. The project contribution is a model- and budget-specific certified boundary, not a new general theory of spherical designs, Gaussian processes or kernel quadrature.

## 11. Reproducibility and trust base

T22 uses exact rational coefficients, directed interval arithmetic, deterministic interval chunks, exact multiplicities, a directed kernel-mean enclosure and a complete hash manifest. Its full theorem-critical numerical certificate has now been regenerated independently by two arithmetic stacks: CPython Decimal/libmpdec and direct-C GMP/MPFR. The second engine regenerated all 1,421 certified subintervals from 1,079 source intervals, recomputed the global minorant, spherical mean, Kerdock energy, Delsarte bound and final one-sided ratio, and produced byte-identical primary outputs under GCC and Clang. Sanitizer builds passed. T16 adds exact integer recurrences, interval sixth-derivative bounds and an interval linear-system certificate. The results are computer-assisted, not proof-assistant formalized.

The frozen local proof release includes the canonical archive, proof-critical intermediates, dependency pins, regeneration commands, complete dual-engine verification and a locally generated SHA-256 sidecar. Public release still requires:

- one canonical archive;
- all proof-critical and deterministic intermediate files;
- pinned dependencies and regeneration commands;
- a clean multi-platform CI run;
- an externally published archive digest;
- AI-assistance disclosure;
- named human mathematical and reproducibility sign-off.

## 12. Limitations

- Arbitrary-node finite-width near-optimality remains open.
- The finite-width theorem is fixed-support and ensemble-level.
- Off-support signed, adaptive and nonlinear methods remain open.
- The T16 primal–dual proof has internal interval and exact-arithmetic audits, but still requires named external human mathematical review before public release.
- No claim is made that the auxiliary lower bound is attained by a cubature rule.
- Empirical competition relevance belongs in a separate, explicitly scoped companion article.

## 13. Conclusion

At the target dimension and point budget, complete Kerdock essentially exhausts the static nonnegative limiting-kernel class, and its all-degree auxiliary certificate is itself optimized. Within the fixed Kerdock-line universe, complete-basis allocation remains exact even for arbitrary real weights and, under explicit assumptions, at finite width. Material improvement must therefore leave at least one certified regime—through new nodes, off-support signed structure, runtime information, nonlinear processing, a transformed residual or finite-width-specific geometry. Which such departures are useful is an empirical and model-specific question, not settled by the static theorem.


## 14. Release disposition

The assembled manuscript and numerical proof passed an internal hostile review after the scope, one-sided logic, stale-artifact, finite-width, signed-weight, M146 and M152 issues were corrected. This is not a substitute for independent human review. A named human mathematician and a named reproducibility reviewer must still approve the public release, and the frozen archive digest must be published through an external authenticated channel.
