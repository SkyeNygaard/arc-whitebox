# T47 — Weighted harmonic rank floor for arbitrary signed nodes

**Status:** exact abstract theorem plus a computer-assisted dimension-256 specialization. The abstract argument is conventional. The numerical certificate uses directed `mpmath.iv` intervals and exact rational harmonic algebra; it requires independent Arb/MPFR reproduction and named human review before publication.

## 1. Abstract theorem

Let

\[
K(t)=\sum_{r\ge0} k_r G_r(t),\qquad k_r\ge0\quad(r\ge1),
\]

be a zonal positive-definite kernel on a sphere. Select finitely many harmonic degrees and nonnegative weights `a_l`. Let `Y_{l,m}` be an orthonormal spherical-harmonic basis, and define

\[
v_a(x)=\big(\sqrt{a_\ell}\,Y_{\ell,m}(x)\big)_{\ell,m}.
\]

Its population covariance is

\[
A=\mathbb E_P[v_a(x)v_a(x)^T]
  =\operatorname{diag}(a_\ell I_{d_\ell}),
\]

and its reproducing kernel is

\[
L_a(x,y)=v_a(x)^T v_a(y)
=\sum_\ell a_\ell d_\ell G_\ell(\langle x,y\rangle).
\]

Write

\[
L_a(t)^2=\sum_{r\ge0} b_r G_r(t),
\qquad
\gamma_a=\min_{r\ge1:b_r>0}\frac{k_r}{b_r}.
\]

Let the eigenvalues of `A` be `lambda_1>=...>=lambda_D>=0`, including multiplicity, and define

\[
F_N(A)=
\sum_{j>N}\lambda_j^2
+
\frac1N\left(\sum_{j>N}\lambda_j\right)^2.
\]

Then every static linear rule

\[
Q=\sum_{i=1}^{m}w_i\delta_{x_i},
\qquad m\le N,
\qquad \sum_iw_i=1,
\]

with arbitrary real weights satisfies

\[
\boxed{R_K(Q)\ge \gamma_a F_N(A).}
\]

No nonnegativity, total-variation, or negative-mass bound on the cubature weights is required.

## 2. Proof

Define

\[
M_Q=\sum_i w_i v_a(x_i)v_a(x_i)^T.
\]

The addition theorem makes `||v_a(x)||^2=tr(A)` independent of `x`, hence mass one gives `tr(M_Q)=tr(A)`. Also `rank(M_Q)<=m<=N`, even when `M_Q` is indefinite.

The discrepancy for the squared reproducing kernel is exactly

\[
R_{L_a^2}(Q)=\|A-M_Q\|_F^2.
\]

To minimize this over real symmetric `M` with rank at most `N` and trace `tr(A)`, align `M` with the eigenbasis of `A` by Hoffman–Wielandt/von Neumann. For any selected `r<=N` directions, the trace-constrained least-squares eigenvalues are

\[
\mu_j=\lambda_j+c,
\qquad
c=\frac{\sum_{j\notin S}\lambda_j}{r}.
\]

Selecting the largest eigenvalues minimizes both omitted sums, and allowing `r=N` cannot worsen the optimum. Therefore

\[
\inf_{\substack{M=M^T,\ \operatorname{rank}M\le N\\
\operatorname{tr}M=\operatorname{tr}A}}
\|A-M\|_F^2=F_N(A).
\]

By definition of `gamma_a`, every nonconstant Gegenbauer coefficient of `K-gamma_a L_a^2` is nonnegative. Constant coefficients vanish from mass-one discrepancy, so the residual kernel has nonnegative energy for every signed mass-one measure. Hence

\[
R_K(Q)\ge\gamma_a R_{L_a^2}(Q)
\ge\gamma_a F_N(A).
\]

## 3. Frozen dimension-256 degree-15 certificate

Use degrees `0,...,15` with exact rational weights represented by these terminating decimals:

| degree | `a_l` |
|---:|---:|
| 0 | 1 |
| 1 | 1 |
| 2 | 1 |
| 3 | 1 |
| 4 | 0.007971493217727095 |
| 5 | 0.00009638797005852535 |
| 6 | 0.0000016379172467841997 |
| 7 | 0.00000003209923511000207 |
| 8 | 0.0000000006658135190281046 |
| 9 | 0.000000000028585879019099307 |
| 10 | 0.0000000000007795660364439458 |
| 11 | 0.000000000000025373546946236714 |
| 12 | 0.000000000000000928902436342902 |
| 13 | 0.00000000000000005496195450219314 |
| 14 | 0.0000000000000000026771772862001115 |
| 15 | 0.0000000000000000001382308855215168 |

The verifier:

1. propagates a 90-digit directed Taylor jet through 32 ReLU-kernel compositions to order 47;
2. obtains lower intervals for `k_r`, `1<=r<=30`, from exact monomial-to-Gegenbauer projections;
3. expands `L_a^2` exactly over the rationals;
4. checks every active nonconstant degree, including odd degrees;
5. computes `F_N(A)` exactly from the repeated weighted harmonic eigenvalues.

The binding coefficient is degree 7. The certified quantities are

\[
\gamma_a\ge
1.1102624925822271742\times10^{-16},
\]

\[
F_N(A)=
1.1073323218694907137\times10^9,
\]

and therefore

\[
\boxed{
R_{K_{32}}(Q)
\ge
1.2294295437956858\times10^{-7}
}.
\]

Relative to the complete-Kerdock MSE used by the proof package,

\[
\boxed{
R_{K_{32}}(Q)
\ge
0.505177125470747\,R_{\mathrm{Kerdock}}
}.
\]

Equivalently, every rule in the theorem's class can improve over complete Kerdock by at most

\[
\boxed{1.979503722\times}.
\]

This supersedes the unweighted T43 numerical floor and the exploratory degree-14 T47 certificate. The weighting search is frozen here because it crossed the prespecified 2x threshold.

## 4. Interpretation

The weighted feature covariance puts unit weight on degrees 0 through 3 and rapidly decreasing weights on degrees 4 through 15. Tiny high-degree eigenvalues still produce a large aggregate rank obstruction because their harmonic multiplicities are enormous, while coefficientwise domination remains feasible.

The result is a strong partial closure of arbitrary signed static cubature: the signed optimum in the theorem's class cannot halve complete-Kerdock MSE. It is not signed near-optimality; a factor just under two remains open.

Conditional competition implication: a same-cost static signed linear rule requiring at least 2x raw-MSE improvement cannot exist in the limiting-kernel class. This does not apply to lower-cost rules, network-adaptive rules, finite-width-specific methods, or nonlinear aggregation.

## 5. Scope

Covered:

- dimension 256;
- depth-32 infinite-width normalized ReLU kernel;
- at most 66,048 arbitrary spherical nodes;
- arbitrary real weights summing to one;
- static, network-independent linear cubature;
- randomized rules independent of the realized field, by conditioning.

Not covered:

- network-dependent support or weights;
- nonlinear processing of evaluations;
- finite-width kernels without new coefficient intervals;
- free-total-mass rules;
- compute reductions from using fewer or cheaper evaluations;
- proof that Kerdock is close to the signed optimum.

## 6. Verification

Run:

```bash
python code/verify_weighted_rank_floor.py
```

The frozen exact result is stored in:

`results/weighted_rank_floor_degree15_frozen.json`.
