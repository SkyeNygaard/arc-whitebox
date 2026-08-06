# T29 — All-width optimum on the complete Kerdock support

## Status

**PROVED UNDER AN EXPLICIT MODEL.**

This theorem is exact at every finite or infinite width. It concerns fixed linear weights on the complete 66,048-point Kerdock support. It does not concern network-dependent weights, changed support, nonlinear estimators, or arbitrary signed nodes.

## Theorem

Let `X={x_1,...,x_N}` be the complete Kerdock point set. Let `Y(x)` be a square-integrable random field with values in a real Hilbert space and rotationally invariant second moment

\[
\mathbb E\langle Y(x),Y(y)\rangle=K(\langle x,y\rangle).
\]

Assume the Gram matrix `G_ij=K(<x_i,x_j>)` has constant row sum, as it does on the complete Kerdock association scheme. Let

\[
Q_wY=\sum_iw_iY(x_i).
\]

### Mass-one rules

Among all deterministic fixed real weights with `sum_i w_i=1`, uniform weights

\[
u_i=1/N
\]

minimize ensemble mean-square integration error.

If `G` is positive definite on the zero-sum subspace, the minimizer is unique.

### Free total mass

If the mass constraint is removed, every minimizer is an alpha-scaled uniform vector

\[
w_i=\alpha_*/N,
\qquad
\alpha_*={A_0\over E_X},
\]

where

\[
A_0=\int K(\langle x,y\rangle)d\sigma(y)
\]

is the spherical kernel mean and

\[
E_X=u^TGu
\]

is the complete-support row average/energy.

## Proof

The ensemble risk is

\[
R(w)=c-2m^Tw+w^TGw,
\]

where isotropy makes `m=A_0 1`. Write `w=alpha u+v` with `1^Tv=0`. Constant Gram row sums give `Gu=lambda 1`, hence

\[
v^TGu=0.
\]

Positive semidefiniteness gives `v^TGv>=0`. Therefore

\[
R(\alpha u+v)=R(\alpha u)+v^TGv,
\]

so `v=0` is optimal. Under `sum w_i=1`, alpha is fixed at one. Without the constraint, minimizing the scalar quadratic

\[
R(\alpha u)=c-2\alpha A_0+\alpha^2E_X
\]

gives `alpha_*=A_0/E_X`.

## Exact scale audit for the depth-32 limiting kernel

Using the archived values

- `A0 = 0.9747299895417149`,
- `E_X = 0.9747302329077503`,

we obtain

\[
\alpha_*=0.9999997503247286441432571426\ldots
\]

The absolute MSE reduction relative to mass-one uniform weighting is

\[
{(E_X-A_0)^2\over E_X}
=6.0762480927294\times10^{-14},
\]

and the relative reduction is

\[
{E_X-A_0\over E_X}
=2.4967527135586\times10^{-7}.
\]

Thus the unconstrained global scale is mathematically distinct but operationally negligible.

## Scope guard

This theorem does not imply that uniform averaging is the Bayes-optimal algorithm among nonlinear estimators. The explicit ReLU counterexample in `INVALIDATED_CASCADE_OBSERVABILITY_ROUTE.md` demonstrates that a nonlinear estimator can use the same antipodal-basis observations to recover an integral exactly when equal-weight linear averaging does not.
