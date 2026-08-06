# Conditional Haar salvage — exact randomization and quantitative near-Haar bounds

**Status:** analytically proved.

Let a compact group `G` act transitively on the integration domain, let `h` be normalized Haar measure, and let

\[
Q=\sum_iw_i\delta_{x_i},
\qquad \sum_iw_i=1.
\]

For `U in G`, write

\[
Q_Uf=\sum_iw_if(Ux_i),
\qquad e(U,f,Q)=Q_Uf-I(f).
\]

The deterministic group-average identity is

\[
\int_G e(U,f,Q)\,dh(U)=0
\]

for every fixed integrable Hilbert-valued `f` and every fixed mass-one signed rule `Q`.

## Exact conditional theorem

Let `H` be a sigma-field containing the realized integrand `f`, the unrotated rule `Q`, and any runtime information used by a correction. If

\[
\operatorname{Law}(U\mid H)=h
\quad\text{almost surely},
\]

then

\[
\mathbb E[e\mid H]=0.
\]

Therefore, for every smaller runtime sigma-field `G_runtime subset H`,

\[
\mathbb E[e\mid G_{\rm runtime}]=0,
\]

and no additive correction measurable from that runtime information can reduce mean-squared error.

This formulation allows the shape, support, and weights of `Q` to depend on the realized integrand or legal features, provided an independent Haar orientation is drawn **after** those choices and before evaluations. It fails if the integrand co-rotates with `U` or if post-orientation observations enter the correction.

## Approximate Haar theorem using chi-square divergence

Let `mu_H=Law(U|H)` and assume `mu_H` is absolutely continuous with respect to Haar measure, with conditional chi-square divergence

\[
\chi_H^2=
\int_G\left(\frac{d\mu_H}{dh}-1\right)^2dh.
\]

Define the conditional Haar-orientation risk

\[
R_{\rm orient}(f,Q)
=
\int_G\|e(U,f,Q)\|^2dh(U).
\]

Then

\[
\|\mathbb E[e\mid H]\|^2
\le
\chi_H^2 R_{\rm orient}(f,Q).
\]

Consequently,

\[
\mathbb E\|\mathbb E[e\mid G_{\rm runtime}]\|^2
\le
\mathbb E[\chi_H^2R_{\rm orient}(f,Q)].
\]

This turns an approximate relative-orientation test into a quantitative upper bound on the value of every orientation-blind correction.

## Approximate Haar theorem using total variation

If `f` is essentially bounded and `B=sum_i|w_i|`, then

\[
\|e(U,f,Q)\|\le(1+B)\|f\|_\infty.
\]

With total variation defined by `TV(mu,h)=sup_A|mu(A)-h(A)|`,

\[
\|\mathbb E[e\mid H]\|
\le
2(1+B)\|f\|_\infty\,TV(\mu_H,h).
\]

Thus exact Haar randomness is not the only useful regime: a certified small conditional orientation defect yields a certified small correction capacity.

## Operationally valid claim

A valid paper statement is:

> Independently Haar-randomizing the rule orientation after fixing the integrand-dependent design makes every orientation-blind additive correction exactly useless. If the conditional orientation law is only approximately Haar, the maximum correction value is bounded by a divergence from Haar times the orientation-averaged baseline risk.
