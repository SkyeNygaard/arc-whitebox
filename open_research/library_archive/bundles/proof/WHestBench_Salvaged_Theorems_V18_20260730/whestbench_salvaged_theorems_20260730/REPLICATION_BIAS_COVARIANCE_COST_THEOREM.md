# Replication salvage — exact bias, covariance, and compute economics

**Status:** analytically proved.

Let `e_1,...,e_m` be Hilbert-valued estimator errors with common mean

\[
b=\mathbb E e_i,
\]

common centered variance

\[
V=\mathbb E\|e_i-b\|^2,
\]

and common pairwise centered covariance

\[
\mathbb E\langle e_i-b,e_j-b\rangle=\rho V,
\qquad i\ne j.
\]

Necessarily

\[
-\frac1{m-1}\le\rho\le1.
\]

The averaged error satisfies the exact identity

\[
\mathbb E\left\|\frac1m\sum_{i=1}^me_i\right\|^2
=
\|b\|^2+
\frac{V}{m}\bigl(1+(m-1)\rho\bigr).
\]

Let

\[
R_0=\|b\|^2+V,
\qquad
\beta=\frac{\|b\|^2}{R_0}.
\]

If total compute is multiplied by `c_m`, the exact MSE-times-compute ratio is

\[
\operatorname{Ratio}_m
=
c_m\left[
\beta+(1-\beta)
\frac{1+(m-1)\rho}{m}
\right].
\]

## Independent replicas

Independence gives `rho=0` for the centered errors, even when the estimators are biased. Hence

\[
R_m=\|b\|^2+V/m.
\]

Under linear cost `c_m=m`,

\[
\operatorname{Ratio}_m
=1+(m-1)\beta.
\]

Therefore:

- unbiased independent replication (`beta=0`) is exactly score-neutral;
- any nonzero common bias makes linear-cost replication strictly worse;
- the deterministic-bias counterexample is the extreme case `beta=1`, giving ratio `m`.

With shared computation and independent errors, replication wins exactly when

\[
c_m<\frac{m}{1+(m-1)\beta}.
\]

For unbiased replicas this reduces to `c_m<m`: any genuine sublinear shared-cost implementation improves the adjusted score.

## Negative covariance

With linear cost, replication wins exactly when

\[
\rho< -\frac{\beta}{1-\beta}
\]

for `beta<1`. Thus antithetic construction can overcome linear cost only if its centered negative covariance is strong enough to offset common bias. Since exchangeability requires `rho>=-1/(m-1)`, such a win is feasible only when

\[
\beta<\frac1m.
\]

If the common-bias share is at least `1/m`, even maximally antithetic exchangeable centered errors cannot beat linear cost.

## Unequal or nonexchangeable replicas

For general mean vector and covariance operator, the optimal deterministic linear combination under a sum-one constraint is the generalized least-squares solution. The exchangeable formula above is the closed-form special case and should replace the false statement that independence alone makes raw errors uncorrelated.

## Operationally valid claim

> Replication removes variance, not common bias. Under linear MSE-times-compute accounting, independent replication is neutral only for unbiased estimators and strictly harmful otherwise. It becomes useful through shared sublinear compute, sufficiently negative centered covariance, or explicit bias reduction.
