# T73 — Exact ReLU replay regions and target-free contraction identities

**Date:** 2026-07-30  
**Status:** Exact deterministic identities. They define the strongest mathematically clean constructive route, but do not by themselves provide the missing target cross-moments.

## 1. Risk-difference identity for a frozen source basis

Let \(z_0\in\mathbb R^p\) be the baseline output, \(\mu\) the unknown target, and let target-free candidate differences be

\[
c_j=z_j-z_0,
\qquad A=[c_1,\ldots,c_r].
\]

Put \(e=z_0-\mu\),

\[
G=A^\top A,
\qquad b=A^\top e.
\]

Then

\[
\|e+A\alpha\|^2
=\|e\|^2+2\alpha^\top b+\alpha^\top G\alpha.
\]

The oracle coefficient is

\[
\alpha_*=-G^\dagger b
\]

on the range of \(G\).

For each individual source,

\[
\boxed{
b_j={\|z_j-\mu\|^2-\|z_0-\mu\|^2-\|c_j\|^2\over2}.}
\]

Thus any setting in which baseline and source risks are analytically computable yields \(b\) without direct target labels.

For static linear quadratures under a known ensemble kernel, these risks and cross-risks are kernel bilinear forms. The resulting optimal constant combination remains a static signed cubature rule and is subject to the v21 signed floor. For nonlinear checkpoint replays, the identity remains exact, but the source risks are not automatically analytic.

## 2. Scalar adjoint reduction of each missing cross-moment

Suppose \(c_j\) is fixed given the realized network and runtime transcript. If the baseline is a cubature mean

\[
z_0=Qf,
\qquad \mu=Pf,
\]

then

\[
\boxed{
b_j=(Q-P)g_j,
\qquad g_j(x)=\langle c_j,f(x)\rangle.}
\]

A rank-4 or rank-5 Oracle coefficient problem is therefore exactly reducible from a 128-vector target to four or five scalar integration errors. This is a genuine dimensional reduction, not a solution: the scalar integrands remain physically oriented and must be integrated or transformed without circular access to \(P f\).

The correct constructive question is whether these \(g_j\) admit:

- exact Gaussian/Stein identities;
- low-complexity conditional expectations;
- gate-boundary formulas with controlled remainder;
- or reuse of already evaluated trajectories.

## 3. Exact scalar ReLU crossing identity

For \(\sigma(u)=\max(u,0)\), baseline preactivation \(a\), and perturbation \(d\),

\[
\sigma(a+d)-\sigma(a)
=\mathbf1_{a>0}d+\rho(a,d),
\]

where \(\rho(a,d)=0\) unless the perturbation crosses the ReLU gate, and

\[
\boxed{|\rho(a,d)|\le(|d|-|a|)_+\le|d|\mathbf1_{|a|\le|d|}.}
\]

The first bound is exact in magnitude on a genuine crossing.

## 4. Exact affine replay region for a suffix

Consider a deterministic ReLU suffix beginning at checkpoint state \(h_L\). Let \(\delta\) be a proposed checkpoint correction. Propagate the linear perturbation

\[
u_L=\delta,
\qquad
v_{k+1}=W_{k+1}u_k,
\qquad
u_{k+1}=D_{k+1}v_{k+1},
\]

where \(D_{k+1}=\operatorname{diag}(\mathbf1_{a_{k+1}>0})\) is the baseline gate matrix.

If, at every downstream neuron,

\[
\operatorname{sign}(a_{k+1,i}+v_{k+1,i})
=
\operatorname{sign}(a_{k+1,i}),
\]

then no gate changes and

\[
\boxed{
F(h_L+\delta)-F(h_L)=J_L\delta
}
\]

exactly. There is no Taylor remainder because a ReLU network is affine on each activation cell.

## 5. Uniform box certificate for several source coefficients

Let \(\delta=A\alpha\) and constrain \(|\alpha_j|\le r_j\). Propagate a matrix of linear source responses \(U_k\). A sufficient gate-stability certificate is

\[
\left|W_{k+1}U_k\right|r
<|a_{k+1}|
\]

coordinatewise at every layer, where the absolute value is entrywise.

If it passes, the entire coefficient box is one exact affine region. Consequently:

- every candidate output is exactly \(z_0+JA\alpha\);
- the source Gram \(G\) is exact throughout the box;
- nonlinear replay cannot reverse a linearized gain;
- coefficient optimization is an exact quadratic program.

This should be checked before any expensive exact-replay sweep.

## 6. Crossing remainder recursion

When gates are not uniformly stable, let \(e_k\) be the difference between the actual and baseline-gate linearized perturbations. Then

\[
e_{k+1}
=D_{k+1}W_{k+1}e_k
+\rho\!\left(a_{k+1},W_{k+1}(u_k+e_k)\right).
\]

Therefore

\[
\|e_{k+1}\|
\le
\|D_{k+1}W_{k+1}\|\,\|e_k\|
+
\left\|
\bigl(|W_{k+1}(u_k+e_k)|-|a_{k+1}|\bigr)_+
\right\|.
\]

This provides a deterministic gate-crossing certificate. Unlike a generic Hessian bound, it vanishes exactly when the suffix remains in one ReLU cell and localizes all nonlinearity to near-margin gates.

## 7. Best constructive use

For the authenticated four-to-five-component Oracle geometry, the mathematically disciplined sequence is:

1. freeze a target-free rank-4 or rank-5 source basis;
2. compute the source-span Oracle ceiling once;
3. certify the largest exact affine coefficient box;
4. reduce the missing coefficients to the scalar errors \((Q-P)g_j\);
5. search for identities for those exact scalar integrands, not generic features;
6. use the crossing recursion only for the residual gates;
7. stop if the source span cannot cross the competition ceiling before coefficient estimation.

This avoids another broad coefficient learner and preserves exact final-output geometry.
