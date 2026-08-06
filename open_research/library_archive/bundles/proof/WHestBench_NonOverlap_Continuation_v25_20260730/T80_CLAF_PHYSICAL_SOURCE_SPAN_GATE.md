# T80 — Exact physical source-span gate for CLAF and late-interface fans

**Status:** exact deterministic theorem. Numerical application requires retained source-state arrays; those arrays were not present in the shared CLAF report package.

## 1. Late-source factorization

Freeze every network weight before the final linear map. Let the late-source state be

\[
c_s(x)\in\mathbb R^w,
\qquad
\Psi_s(x)=W_{31}c_s(x).
\]

The exact late-center defect has the form

\[
t_s=(P-Q)\Psi_s=W_{31}d_s,
\qquad
d_s=(P-Q)c_s.
\]

A fan compiled from sampled rays `x_1,...,x_m` sees

\[
y_j=W_{31}c_s(x_j).
\]

Every conic-fan mean and every coefficient-one cross-fitted fan output is a common linear combination of these output vectors. Therefore it has the form

\[
\widehat t_s=W_{31}C\alpha,
\qquad
C=[c_s(x_1)\ \cdots\ c_s(x_m)].
\]

## 2. Exact oracle capacity

Let

\[
G=W_{31}^\top W_{31},
\qquad
H=C^\top G C,
\qquad
b=C^\top Gd_s.
\]

The best possible fan coefficients—even target-informed oracle coefficients—satisfy

\[
\boxed{
\min_\alpha\|W_{31}(d_s-C\alpha)\|^2
=d_s^\top Gd_s-b^\top H^\dagger b.
}
\]

The exact source-span capture is

\[
\boxed{
\eta_{\mathrm{span}}
=\frac{b^\top H^\dagger b}{d_s^\top Gd_s}.
}
\]

No interpolation rule, cross-fitting identity, coefficient learning, or final replay can exceed this capacity while using the same sampled source states.

## 3. Cross-fitted version

If fold A selects a fan evaluated on B and fold B selects a fan evaluated on A, form the union source matrix

\[
C_{AB}=[C_A\ C_B].
\]

The unconstrained oracle projection onto `span(C_AB)` is an upper bound on every weighted cross-fitted construction. Therefore:

> If the union-span oracle fails, every corresponding cross-fitted fan fails.

This is stronger and cheaper than implementing the fan first.

## 4. Required gate for the frozen CLAF proposal

The CLAF report's own complete-score arithmetic leaves approximately `0.45%–0.55%` relative source-error slack, depending on start depth and ray count. For the frozen `s=9`, `M=64` design, the declared allowance is approximately `0.502%`.

Thus the first and mandatory screen is

\[
\boxed{1-\eta_{\mathrm{span}}\le0.00502.}
\]

A result such as 80%, 95%, or even 99% span capture is scientifically interesting but does not pass the competition gate.

## 5. Required exposed-only protocol

For every already-exposed network and rotation:

1. compute the exact or high-reference source defect `d_s`;
2. store the 128 source-state vectors from the two frozen 64-ray folds;
3. compute `G=W31^T W31`;
4. whiten with a stable eigendecomposition of `G`;
5. report union-span oracle residual, each-fold residual, effective rank, and conditioning;
6. compare same-basis, cross-basis, random-plane, score-selected-plane, and rank-one controls;
7. stop before fan construction unless the union-span residual is below `0.00502` on every required group and has sufficient numerical margin.

## 6. Interpretation

- Input-plane dimension two is not the relevant rank; the relevant object is the physical span of the emitted source states in the final-output Gram metric.
- Exact fan integration does not increase this span.
- Cross-fitting prevents direct fold reuse but does not add source capacity beyond the union span.
- This theorem does not require a Gaussian approximation.
- The test should precede all CLAF implementation work and all coefficient tuning.
