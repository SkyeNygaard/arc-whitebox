# T72 — Exact finite-width noise-stability representation and moment-dual proof program

**Date:** 2026-07-30  
**Status:** Exact representation and exact primal/dual formulation. Numerical width-256 coefficient bounds remain open because the required rigorous finite-width kernel inputs are not yet available.

## 1. Exact representation at every finite width

Let \(G,H\in\mathbb R^m\) be jointly standard Gaussian with

\[
\operatorname{Cov}(G_i,H_j)=t\,\delta_{ij},
\qquad -1\le t\le1.
\]

Let \(U\) collect every network parameter downstream of the first Gaussian layer. Conditional on \(U\), write the scalar or Hilbert-valued network output as a square-integrable Gaussian function

\[
f_U\in L^2(\gamma_m;\mathcal H).
\]

Define the ensemble zonal kernel

\[
K_m(t)=\mathbb E_U\mathbb E\langle f_U(G),f_U(H)\rangle.
\]

Expand in the normalized multivariate Hermite basis:

\[
f_U(g)=\sum_{\alpha\in\mathbb N^m}\widehat f_{U,\alpha}H_\alpha(g).
\]

Mehler's identity gives

\[
\boxed{
K_m(t)=\sum_{n=0}^\infty a_n^{(m)}t^n,
\qquad
a_n^{(m)}=
\mathbb E_U\sum_{|\alpha|=n}
\|\widehat f_{U,\alpha}\|^2\ge0.
}
\]

This is exact for every width and depth. It does not require neurons to remain independent after the first layer. Dependence is absorbed into the random function \(f_U\); coefficient positivity follows from squared Hermite norms.

If the integration-error kernel is centered, only the degree-zero term changes. Every nonconstant chaos coefficient remains nonnegative.

## 2. Gegenbauer coefficients are positive moment functionals

In dimension \(d=256\), write

\[
t^n=\sum_{\ell=0}^n M_{n\ell}G_\ell(t),
\qquad M_{n\ell}\ge0.
\]

Therefore

\[
\boxed{
k_\ell^{(m)}=
\sum_{n\ge\ell}M_{n\ell}a_n^{(m)}.}
\]

Every harmonic coefficient needed by the signed certificate is a positive linear functional of the unknown chaos-energy measure on \(\mathbb N\).

This is the correct finite-width state variable. Reconstructing the whole kernel pointwise is unnecessary.

## 3. Exact coefficient lower-bound LP

Suppose directed analysis supplies interval observations

\[
L_j\le K_m(t_j)\le U_j,
\quad j=1,\ldots,J,
\]

plus normalization or moment constraints. Let \(a=(a_n)_{n\ge0}\) range over all nonnegative sequences satisfying

\[
L_j\le\sum_{n\ge0}a_nt_j^n\le U_j,
\qquad a_n\ge0.
\]

For a required harmonic degree \(\ell\), the sharp certified lower bound is the semi-infinite linear program

\[
\underline k_\ell
=
\inf_{a\ge0}
\sum_{n\ge0}M_{n\ell}a_n
\quad\text{subject to the interval constraints.}
\]

A cutoff witness \(h_{M,\ell}\) transfers with

\[
\alpha_M=
\min_{1\le\ell\le M:\,h_{M,\ell}>0}
{\underline k_\ell\over h_{M,\ell}}.
\]

If \(R_m(Q_K)\le\beta R_\infty^{\rm upper}(Q_K)\), then

\[
{R_m(Q)\over R_m(Q_K)}
\ge {\alpha_Mf_M\over\beta}.
\]

The exact v21 subcertificate frontier is:

| retained limiting signed floor | required harmonic cutoff |
|---:|---:|
| 80% | 62 |
| 85% | 84 |
| 90% | 128 |
| 92% | 164 |
| 93% | 194 |
| 93.7046% | 280 |

## 4. Dual certificate

For a fixed \(\ell\), introduce dual multipliers for the lower and upper kernel-value constraints and any normalization constraints. A feasible dual produces a sequence inequality of the form

\[
M_{n\ell}\ge
\lambda_0+
\sum_{j=1}^J\lambda_jt_j^n
\qquad(n=0,1,2,\ldots).
\]

Summing against \(a_n\ge0\) gives a rigorous coefficient lower bound from the interval data.

A complete computer-assisted proof therefore consists of:

1. directed intervals for the finite-width inputs;
2. rationalized primal and dual moment solutions;
3. a finite proof of the infinite tail inequality;
4. exact monomial-to-Gegenbauer coefficients;
5. an independent finite-width Kerdock-risk upper bound.

## 5. Nonidentifiability theorem for finite point samples

Finite pointwise kernel samples do not determine the chaos spectrum.

Let \(t_1,\ldots,t_J\in(-1,1)\) be fixed. Choose \(J+2\) distinct degrees \(n_0,\ldots,n_{J+1}\). The matrix

\[
\begin{bmatrix}
1&\cdots&1\\
t_1^{n_0}&\cdots&t_1^{n_{J+1}}\\
\vdots&&\vdots\\
t_J^{n_0}&\cdots&t_J^{n_{J+1}}
\end{bmatrix}
\]

has a nontrivial null vector \(v\). Because \(\sum v_i=0\), it has both signs. Starting from any strictly positive mass vector on those degrees, sufficiently small perturbations \(a\pm\varepsilon v\) remain nonnegative and produce exactly the same normalization and sampled kernel values.

Generically, they have different harmonic coefficients.

Thus:

- point samples are valid constraints;
- they are not a substitute for a moment LP;
- no coefficient claim may be read directly from pointwise closeness;
- tail control or a valid dual inequality is essential.

## 6. Why a naive layerwise concentration proof is unlikely to reach degree 128

At finite width, the next-layer two-input Gram matrix is a sample average of nonlinear transforms of a random previous-layer Gram matrix. The expectation therefore does not obey the infinite-width deterministic recursion exactly. A uniform sup-norm error bound must survive:

- 31 nonlinear transitions;
- ReLU-dual curvature singular near correlations \(\pm1\);
- normalization fluctuations in both marginal norms;
- conversion from pointwise error to 128 separate coefficient lower bounds.

Even a useful pointwise \(O(L/\sqrt m)\) estimate need not imply any positive lower bound for a small high-degree Gegenbauer coefficient. The compressed moment-dual route is not optional bookkeeping; it is what prevents an invalid pointwise-to-spectral inference.

## 7. Best rigorous next step

The first target should be the 80% degree-62 subcertificate, not degree 128 immediately.

A minimal successful result needs:

1. an exact Markov representation for the finite-width two-input Gram process;
2. directed intervals for a small family of Ornstein-Uhlenbeck functionals or kernel values;
3. a dual moment certificate for every active coefficient through 62;
4. a finite-width Kerdock denominator bound;
5. only then escalation to 84 and 128.

If degree 62 cannot obtain a useful \(\alpha_{62}/\beta\), the finite-width program should stop and report the exact missing information rather than add more sample points heuristically.
