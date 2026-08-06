# T41 — Symmetry-defect alignment theorem

**Status:** Proved under an explicit measure-preserving action and policy class.

## Theorem

Let \((\Omega,\mathcal F,\mathbb P)\) be a probability space, let \(\tau:\Omega\to\Omega\) be a measure-preserving involution, and let \(U:H\to H\) be a unitary involution on a real Hilbert space. For \(e,c\in L^2(\Omega;H)\), define

\[
\widetilde e(\omega)=Ue(\tau\omega),\qquad
\widetilde c(\omega)=Uc(\tau\omega).
\]

Then

\[
2\,\mathbb E\langle e,c\rangle
=\mathbb E\langle e+\widetilde e,c\rangle
+\mathbb E\langle \widetilde e,\widetilde c-c\rangle.
\]

Consequently,

\[
2|\mathbb E\langle e,c\rangle|
\le \|e+\widetilde e\|_{L^2}\|c\|_{L^2}
+\|e\|_{L^2}\|\widetilde c-c\|_{L^2}.
\]

For nonzero \(e,c\), put

\[
\delta_e={\|e+\widetilde e\|_{L^2}\over\|e\|_{L^2}},
\qquad
\delta_c={\|\widetilde c-c\|_{L^2}\over\|c\|_{L^2}}.
\]

The normalized signed alignment obeys

\[
{ |\mathbb E\langle e,c\rangle|^2
 \over \mathbb E\|e\|^2\,\mathbb E\|c\|^2 }
\le \left({\delta_e+\delta_c\over2}\right)^2.
\]

## Proof

Measure preservation and unitarity give

\[
\mathbb E\langle e,c\rangle
=\mathbb E\langle \widetilde e,\widetilde c\rangle.
\]

Add the two equal expressions and insert and subtract \(c\) in the second inner product. The displayed identity follows. Cauchy–Schwarz and \(\|\widetilde e\|_{L^2}=\|e\|_{L^2}\) give the inequality. Dividing by the two norms gives the normalized statement. \(\square\)

## Exact zero-defect corollary

If the error is anti-equivariant,

\[
e(\tau\omega)=-Ue(\omega),
\]

and the correction is equivariant,

\[
c(\tau\omega)=Uc(\omega),
\]

then \(\delta_e=\delta_c=0\), so \(\mathbb E\langle e,c\rangle=0\). Such a correction has no average signed value against that error component.

## Dictionary corollary

Let a random linear dictionary be \(A:\mathbb R^m\to H\), let \(z=A^*e\), and let a runtime policy choose coefficients \(a=h(X)\). Then

\[
\mathbb E\langle e,Aa\rangle=\mathbb E\langle z,a\rangle.
\]

Applying the theorem in coefficient space yields

\[
2|\mathbb E\langle z,a\rangle|
\le \|z+\widetilde z\|_{L^2}\|a\|_{L^2}
+\|z\|_{L^2}\|\widetilde a-a\|_{L^2}.
\]

This separates two measurable failure modes: imperfect sign reversal of the oracle coefficient signal and imperfect invariance/equivariance of the policy.

## Exact randomized-orientation model

Augment every signed probe with an independent uniform orientation bit \(S\in\{-1,+1\}\). Let \(\tau\) flip \(S\). Suppose:

1. the target cross-moment changes sign under the flip;
2. the declared runtime features omit the orientation bit and are invariant under the flip;
3. coefficients are measurable functions only of those orientation-blind features.

Then the oracle coefficient signal is anti-equivariant while the policy is invariant. Its expected signed correction value is exactly zero. Orientation-aware odd features are outside this closure and are the explicit escape class.

## Scope

The theorem does **not** prove that the existing deterministic WHestBench near-collision construction has the required symmetry. It does not cover orientation-aware features, full white-box weights without a computational restriction, added evaluations whose signed values enter the transcript, or policies outside the declared equivariance class. Its value is to replace vague “absolute phase” rhetoric with a quantitative, falsifiable, class-indexed bound.
