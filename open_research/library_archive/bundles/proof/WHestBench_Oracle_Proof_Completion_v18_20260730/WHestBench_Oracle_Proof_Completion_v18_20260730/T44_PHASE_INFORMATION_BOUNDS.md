# T44 — Quantitative phase-information bounds

**Status:** exact abstract information-theoretic results. No numerical WHestBench ceiling follows until an actual legal transcript is given a valid information or likelihood bound.

## 1. Pure binary phase

Let `S` be uniform on `{-1,+1}`, let `v` be a fixed Hilbert-space vector, and let

\[
e=S v.
\]

For any runtime transcript `X`, define

\[
m(X)=\mathbb E[S\mid X].
\]

The optimal `X`-measurable correction is `m(X)v`, and therefore

\[
\frac{V(X;e)}{\mathbb E\|e\|^2}
=
\mathbb E[m(X)^2].
\]

Furthermore,

\[
\boxed{
\mathbb E[m(X)^2]
\le
\min\{1,2I(S;X)\}
}
\]

when mutual information is measured in nats.

### Proof

For a balanced binary prior,

\[
I(S;X)=\mathbb E\,\phi(m(X)),
\]

where

\[
\phi(m)=
\frac{1+m}{2}\log(1+m)
+
\frac{1-m}{2}\log(1-m).
\]

Now `phi(0)=phi'(0)=0` and

\[
\phi''(m)=\frac1{1-m^2}\ge1.
\]

Hence `phi(m)>=m^2/2` for `|m|<1`, with the endpoint result by continuity.

## 2. Sequential probe budget

For sequential observations `X_1,...,X_k`, if

\[
I(S;X_j\mid X_{<j})\le\eta_j,
\]

then the chain rule gives

\[
\frac{V(X_{1:k};e)}{\mathbb E\|e\|^2}
\le
2\sum_{j=1}^{k}\eta_j.
\]

This is useful only after a valid per-probe KL or mutual-information bound has been established.

## 3. Conditional phase with network-dependent magnitude

A closer abstraction for Oracle phase is

\[
e=S v(Z),
\]

where `Z` is phase-invariant runtime state and `S` is conditionally balanced given `Z`. If the transcript includes `Z`, then

\[
V(X,Z;e)
=
\mathbb E\left[
\|v(Z)\|^2
\mathbb E[S\mid X,Z]^2
\right].
\]

Pointwise application of the binary inequality yields the weighted bound

\[
V(X,Z;e)
\le
2\,\mathbb E\left[
\|v(Z)\|^2
D_{\mathrm{KL}}(
P_{S\mid X,Z}\|P_{S\mid Z})
\right].
\]

If `||v(Z)||^2 <= B` almost surely and `A=E||v(Z)||^2`, then

\[
\frac{V(X,Z;e)}{A}
\le
2\frac BA I(S;X\mid Z).
\]

This explicitly exposes the cost of heterogeneous Oracle magnitude.

## 4. Finite-action Oracle selection

Let `J` be uniform over `M` candidate actions. Fano's inequality gives

\[
\Pr(\widehat J(X)\ne J)
\ge
1-\frac{I(J;X)+\log2}{\log M}.
\]

If the correct action has normalized reward 1 and every incorrect action has reward at most `rho`, then

\[
\mathbb E[\mathrm{reward}]
\le
\min\left\{
1,
\rho+(1-\rho)\frac{I(J;X)+\log2}{\log M}
\right\}.
\]

## What this changes

These inequalities make “phase information” mathematically quantitative. They separate:

- Oracle capacity;
- the amount of information in the legal transcript about the phase/action;
- achievable correction value.

## What remains open

Failed ridge, tree, or neural predictors do **not** upper-bound mutual information. Full weights determine the target in principle, so no unconditional information-theoretic impossibility theorem is possible. A WHestBench application requires one of:

1. an exact symmetry giving zero information;
2. a generative model with certified KL bounds;
3. a restricted noisy probe protocol with a per-probe information bound;
4. a deterministic quotient/minimax argument such as T46.
