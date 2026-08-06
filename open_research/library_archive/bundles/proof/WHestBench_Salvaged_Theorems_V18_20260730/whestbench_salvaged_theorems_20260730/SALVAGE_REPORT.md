# WHestBench salvaged-theorem report

**Date:** 2026-07-30  
**Disposition:** every explicit counterexample from the hostile audit now has a maximal correct replacement; two application-specific conclusions are stronger than the original repaired wording.

## Executive result

The hostile audit did not merely delete claims. It exposed the exact mathematical boundary of each claim. The replacement package proves:

1. a complete affine characterization of symmetric-Gram minimizers;
2. strict positive definiteness of the actual depth-32 limiting ReLU kernel, restoring unique Kerdock weights in that application;
3. an exact constant/quadratic/high-even trichotomy for the finite-width MUB-line theorem;
4. exact and approximate conditional-Haar no-value results under the right relative-orientation condition;
5. exact bias-covariance-compute formulas for replication;
6. a sharp ReLU gate-crossing bound with constant `1/3`, rather than the previous loose `2`;
7. a fully uniform optimizer-transfer theorem for kernel perturbations;
8. an always-defined observability reporting convention;
9. a directed endpoint certificate completing T16 equality localization.

## Most important recovery: T29

The general free-mass uniqueness statement was false because a symmetric Gram matrix can have zero-sum null directions. The correct general result is a full minimizer set.

However, the actual limiting K32 kernel is much less degenerate than the general model. Its normalized ReLU map has strictly positive constant, linear, and every even power coefficient. After one additional composition, every odd coefficient becomes positive too. Every later composition preserves positivity in every degree. Since `K32` is the 32-fold iterate starting from `K0(t)=t`, it lies well inside this strict regime.

This yields a direct tensor-feature/Vandermonde proof that K32 is strictly positive definite on any finite set of distinct sphere points. Therefore the specific complete-Kerdock limiting-kernel claim recovers full uniqueness:

- uniform is the unique mass-one fixed-linear optimum;
- the rigorously enclosed `alpha_*`-scaled uniform vector is the unique free-mass optimum.

This recovery is stronger and cleaner than adding uniqueness as an unverified assumption. A directed-rounding association-spectrum certificate additionally proves a full-line zero-sum stability modulus above `0.00956473382419646475783854720307667122`.

## Most useful interpretation of the T38 counterexample

The pure degree-two example is not a pathological dead end. It is the exact boundary between degeneracy and strict complete-basis concentration.

For every nonnegative even noise-stability expansion:

- nonconstant even mass makes `A-O>0` and `O-C<0`;
- degree-four-or-higher mass is exactly equivalent to the third strict sign;
- pure quadratic mass makes the between-basis eigenvalue exactly zero.

In the pure-quadratic case, any mixture of complete orthonormal bases with equal within-basis weights is optimal. With fewer than `d` lines, the optimum concentrates all lines in one basis. With at least `d`, one complete basis already removes the quadratic discrepancy. This is a positive theorem and a useful diagnostic for why low-degree controls saturate.

For a finite piecewise-affine ReLU network, a nonconstant even realization cannot terminate at degree two, so the original practical finite-ReLU conclusion survives after separating it from the false general square-integrable formulation.

## Information symmetry replacement

The false Haar statement confused invariance of reported features with randomness of the *relative orientation*. The exact valid condition is conditional Haar randomness after fixing the integrand and selected rule.

This can be enforced operationally: choose any legal integrand-dependent design shape and weights, then draw an independent Haar orientation. Every correction that does not observe orientation-sensitive post-rotation information has zero value.

The result also admits an approximate form. Conditional chi-square divergence from Haar multiplies the orientation-averaged risk to upper-bound all orientation-blind correction value. This converts symmetry into a quantitative empirical target instead of an unverifiable slogan.

## Replication replacement

Independent replication reduces centered variance but preserves common bias. The exact cost-adjusted formula identifies all useful regimes:

- unbiased independent replicas plus linear cost: neutral;
- biased independent replicas plus linear cost: strictly worse;
- sublinear shared compute: can win, with an exact threshold;
- negative centered covariance: can win, with an exact threshold;
- bias reduction: changes the conclusion and should be measured separately.

This is more actionable than either the false neutrality claim or a blanket rejection of replication.

## ReLU replacement

The ReLU remainder is a triangular gate-crossing profile, not merely an indicator-bounded error. Integrating the triangle gives

`E r^2 <= L |t|^3 / 3`

when the density is bounded by `L` on the actual crossing interval. The constant is asymptotically sharp. Conditional and vector versions directly support downstream replay certificates.

For a Gaussian preactivation with conditional standard deviation at least `sigma`, the bound becomes

`E r^2 <= E|T|^3 / (3 sigma sqrt(2 pi))`.

This is six times tighter than the already-corrected `2L|t|^3` statement.

## Verification

`verify_salvaged_theorems.py` passed. It checks:

- positivity and numerical accuracy of the ReLU-kernel power series;
- positive coefficients after composition;
- a K32 Gram matrix including antipodal points;
- the exact MUB block spectrum, including a directed-rounding K32 stability certificate;
- the pure-quadratic null multiplicity and finite-budget partition claim;
- the biased-replication formula;
- the sharp normal-density ReLU bound and its asymptotic constant;
- a finite-group analogue of the chi-square near-Haar inequality;
- strict T16 endpoint separation using certified coefficient intervals.

These computations are sanity checks; the theorem files contain the analytic proofs.

## Recommended canonical claim set

The recommended external manuscript should use the replacement matrix in `VALID_CLAIMS_MATRIX.md`, add theorem IDs T41–T47, and retain the hostile counterexamples as boundary examples rather than deleting them. The valid story is stronger when it says exactly when a theorem is strict, degenerate, approximate, or unique.
