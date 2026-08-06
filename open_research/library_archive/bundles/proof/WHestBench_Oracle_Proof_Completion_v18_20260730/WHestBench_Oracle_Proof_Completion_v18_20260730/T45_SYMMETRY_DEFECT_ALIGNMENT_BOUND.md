# T45 — Symmetry-defect alignment and correction-value bound

**Status:** exact Hilbert-space theorem. Application to WHestBench requires a specified transformation and measured or proved defects.

## Setup

Let `tau` be a measure-preserving involution on the instance space, and let `U` be a unitary action on the error Hilbert space. For an error `e` and proposed correction `c`, define the pulled-back transforms

\[
\widetilde e(\omega)=U^*e(\tau\omega),
\qquad
\widetilde c(\omega)=U^*c(\tau\omega).
\]

## Exact identity

\[
\boxed{
2\mathbb E\langle e,c\rangle
=
\mathbb E\langle e+\widetilde e,c\rangle
+
\mathbb E\langle\widetilde e,\widetilde c-c\rangle
}
\]

### Proof

Measure preservation and unitarity imply

\[
\mathbb E\langle e,c\rangle
=
\mathbb E\langle\widetilde e,\widetilde c\rangle.
\]

Add the two equal expressions and insert/subtract `E<tilde e,c>`.

## Quantitative alignment bound

For nonzero `e` and `c`, define

\[
\delta_e=
\frac{\|e+\widetilde e\|_{L^2}}{\|e\|_{L^2}},
\qquad
\delta_c=
\frac{\|\widetilde c-c\|_{L^2}}{\|c\|_{L^2}}.
\]

Then Cauchy-Schwarz gives

\[
\boxed{
\frac{|\mathbb E\langle e,c\rangle|}
{\|e\|_{L^2}\|c\|_{L^2}}
\le
\frac{\delta_e+\delta_c}{2}
}.
\]

Thus an approximately anti-invariant error and approximately invariant correction can have only limited signed alignment.

## Correction-value corollary

For a fixed direction `c`, the best scalar correction `alpha c` removes

\[
\frac{\mathbb E\langle e,c\rangle^2}
{\mathbb E\|c\|^2}
\]

of MSE. Therefore its fractional correction value satisfies

\[
\boxed{
\frac{\mathrm{gain}(c)}{\mathbb E\|e\|^2}
\le
\min\left\{1,
\left(\frac{\delta_e+\delta_c}{2}\right)^2
\right\}.
}
\]

This is the quantitative version of a zero-alignment symmetry theorem.

## Exact zero case

If

\[
\widetilde e=-e,
\qquad
\widetilde c=c,
\]

then `E<e,c>=0`. Every nonzero correction in that class increases risk by its squared norm; zero correction is optimal.

## Feature-map corollary

Suppose `c=h(X)` and a transformation acts on the runtime feature map as `X -> X_tau`. If `h` is `L`-Lipschitz and has equivariance defect

\[
\epsilon_h=
\|U^*h(X_\tau)-h(X)\|_{L^2},
\]

then `delta_c` can be bounded by the feature symmetry defect plus the policy equivariance defect. This turns a proposed empirical symmetry into a falsifiable quantitative program rather than an assumed exact theorem.

## Required WHestBench application protocol

1. State the instance distribution and legal transcript exactly.
2. Define `tau` and `U` explicitly.
3. Prove measure preservation, or label it empirical.
4. Measure `delta_e` and `delta_c` on grouped development data.
5. Compare the observed correction value to the squared defect ceiling.
6. Do not claim a universal result if the bound is vacuous or if the action omits legal orientation-sensitive information.

## Counterexamples and scope guards

- A non-measure-preserving transformation invalidates the identity.
- Orientation-sensitive features can make `delta_c` large and escape the obstruction.
- A nonzero invariant error component permits useful invariant corrections.
- Full weights may break the proposed symmetry even when norms and Gram matrices do not.
- A theorem for one policy class does not close nonlinear policies outside that class.
