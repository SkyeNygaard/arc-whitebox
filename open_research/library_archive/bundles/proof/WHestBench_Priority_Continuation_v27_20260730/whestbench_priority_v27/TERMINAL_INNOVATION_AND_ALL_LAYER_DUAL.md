# Terminal-innovation lower bound and all-layer checkpoint-gauge dual

## 1. Population terminal-innovation inequality

Let \(H_t\in L^2(\Omega;\mathbb R^{d_t})\) be centered checkpoint states. Fix checkpoints

\[
1=t_0<t_1<\cdots<t_m=L,
\]

controls \(C_j\), and terminal source \(C_m=U\). For independent unbiased block estimates, write

\[
v_j(C)=\mathbb E\left\|C_j^T H_{t_j}-C_{j-1}^T H_{t_{j-1}}\right\|_2^2,
\qquad
S(C)=\sum_{j=1}^m \sqrt{\gamma_j v_j(C)}.
\]

Every term in \(S\) is nonnegative. Therefore, if \(t=t_{m-1}\),

\[
S(C)\ge \sqrt{\gamma_m}\inf_C
\left(\mathbb E\|U^T H_L-C^T H_t\|_2^2\right)^{1/2}.
\]

With covariance blocks \(\Sigma_{LL},\Sigma_{Lt},\Sigma_{tt}\), the optimal linear residual is

\[
V_{L\mid t}(U)=
\operatorname{tr}\left[
U^T\left(\Sigma_{LL}-\Sigma_{Lt}\Sigma_{tt}^{\dagger}\Sigma_{tL}\right)U
\right].
\]

Hence

\[
\boxed{S(C)\ge \sqrt{\gamma_m V_{L\mid t}(U)}}.
\]

This bound grants every earlier block zero cost and zero variance. Failure of this optimistic bound closes every linear checkpoint telescope whose last preterminal checkpoint is \(t\) or earlier under the declared cost model.

For fixed-radius antithetic pairs requiring a full trajectory for the terminal block, \(\gamma_m=2/66048\).

## 2. Empirical all-layer SOCP and dual

For centered empirical state matrices \(X_t\), the complete all-layer program is

\[
\min_{C_1,\ldots,C_{31}}
\sum_{j=1}^{31}a_j
\|X_{j+1}C_{j+1}-X_jC_j\|_F,
\qquad C_{32}=U,
\]

where \(a_j=\sqrt{2(j+1)/(66048\cdot32)}\).

A dual-feasible collection \(Y_j\) obeys

\[
\|Y_j\|_F\le a_j,
\]

\[
X_1^TY_1=0,
\qquad
X_i^T(Y_{i-1}-Y_i)=0\quad(2\le i\le31).
\]

Its objective is

\[
\langle Y_{31},X_{32}U\rangle_F.
\]

The packaged certificate starts from the primal residual subgradient, orthogonally projects it onto the checkpoint-balance nullspace by solving the block-tridiagonal normal system, and globally rescales it into all Frobenius norm balls. Therefore its objective is a valid lower bound even though the primal was stopped after only four source-aware iterations.

## 3. Scope

The terminal inequality is exact at the population level. The numerical all-layer certificates are exact for their empirical covariance matrices, not universal population lower bounds. They use no protected data and deliberately grant free covariance/control construction. They close neither nonlinear or biased estimators nor an analytic late-innovation identity outside the independent-block linear checkpoint-gauge class.
