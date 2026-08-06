# WHestBench Agent 1 — Joint Moment Realizability / Zero-Code Gap

**Date:** 2026-07-30  
**Role:** extremal harmonic analyst and truncated-moment theorist  
**Status:** exact proof-method closure plus strict nonattainment; no positive uniform epsilon for unrestricted signed rules

## Executive result

The current degree-280 comparison certificate proves

\[
R_{K_{32}}(Q)\ge L_0
=2.28045159853140213494322646565331\times10^{-7}
=0.9370459569114724\,R_{\mathrm{Kerdock}}^{\mathrm{upper}}
\]

for every static, network-independent, mass-one signed rule on at most
\(N=66{,}048\) nodes on \(S^{255}\).

This round establishes three new facts.

1. **Shared second moments are still exactly sharp.** Even if all 146 comparison profiles are forced to arise from one common rank-\(N\) harmonic block matrix, with every cross-degree block shared, the abstract rank floors can be attained simultaneously. Thus a “joint SDP” that only shares one second-moment/Gram matrix cannot improve the theorem.
2. **The released certificate floor is not attained by any actual atomic rule.** Two positive certificate components use the same adjacent degrees \((3,4)\) with distinct radii. Equality would force both \(G_3(t)\) and \(G_4(t)\) to vanish at every off-diagonal inner product, which is impossible because consecutive Gegenbauer polynomials have no common zero.
3. **Strict nonattainment does not yield a uniform signed epsilon.** Signed atomic rules have unbounded total variation and a noncompact coalescing-cancellation limit. A quantitative improvement now requires a genuine product/Hankel/sphere-ideal constraint or an explicit total-variation/negative-mass bound.

A small exact positive-weight subclass improvement is also certified:

\[
R_{K_{32}}(Q)\ge L_0+1.22117340347331035709\times10^{-23}
\]

for nonnegative mass-one rules, using only the two \(s=3\) certificate components. This is mathematically positive but competition-negligible.

---

## 1. Inputs, notation, and scope

The released certificate contains 146 positive components. Each component is an adjacent-degree kernel

\[
L_{s,r}(t)=d_sG_s(t)+r\,d_{s+1}G_{s+1}(t),
\]

normalized by its exact rank floor and assigned a positive objective contribution \(y_{s,r}\). The exact sum of all component objectives is \(L_0\).

For a set \(\Lambda\) of active harmonic degrees, write

\[
\mathcal H=\bigoplus_{\ell\in\Lambda}H_\ell,
\qquad d_\ell=\dim H_\ell.
\]

Let \(Y_\ell(x)\in\mathbb R^{d_\ell}\) be an orthonormal harmonic evaluation vector, normalized so

\[
Y_\ell(x)^TY_\ell(y)=d_\ell G_\ell(x\cdot y).
\]

For a mass-one signed atomic rule \(Q=\sum_iw_i\delta_{x_i}\), define the common unweighted harmonic moment matrix

\[
B_Q=\sum_iw_iY(x_i)Y(x_i)^T,
\qquad Y(x)=\bigoplus_{\ell\in\Lambda}Y_\ell(x).
\]

Then \(\operatorname{rank}(B_Q)\le N\), and every diagonal harmonic block has fixed trace

\[
\operatorname{tr}(B_{Q,\ell\ell})=d_\ell.
\]

For a profile \(a=(a_\ell)_{\ell\in\Lambda}\), put

\[
D_a=\bigoplus_\ell\sqrt{a_\ell}I_{d_\ell},
\qquad
M_a=D_aB_QD_a,
\qquad
A_a=\bigoplus_\ell a_\ell I_{d_\ell}.
\]

Define

\[
T_a=\sum_\ell a_\ell d_\ell,
\qquad
S_a=\sum_\ell a_\ell^2d_\ell.
\]

The profile discrepancy is \(\|A_a-M_a\|_F^2\), and the standard rank floor is

\[
F_a={T_a^2\over N}-S_a.
\]

For the released certificate, active degrees run from 3 through 140. The smallest active harmonic dimension is

\[
d_3=2{,}828{,}800>N=66{,}048.
\]

---

## 2. Maximal theorem: simultaneous shared-Gram sharpness

### Theorem 1 — Common second-moment coupling cannot strengthen the certificate

Let \(\Lambda\) be a finite set of harmonic degrees satisfying \(d_\ell\ge N\) for every \(\ell\in\Lambda\). Let \(\mathcal A\) be any finite collection of nonnegative profiles \(a\).

Consider the relaxation in which all profile matrices must be generated from one common real symmetric matrix \(B\) satisfying

\[
\operatorname{rank}(B)\le N,
\qquad
\operatorname{tr}(B_{\ell\ell})=d_\ell
\quad(\ell\in\Lambda),
\]

through \(M_a=D_aBD_a\). Then there exists one feasible \(B\) for which, simultaneously for every \(a\in\mathcal A\),

\[
\|A_a-M_a\|_F^2={T_a^2\over N}-S_a.
\]

Therefore every nonnegative weighted sum of these profile discrepancies has optimum exactly equal to the sum of its separate rank floors.

### Proof

For every \(\ell\), choose a matrix

\[
V_\ell\in\mathbb R^{d_\ell\times N},
\qquad V_\ell^TV_\ell=I_N,
\]

which is possible because \(d_\ell\ge N\). Set

\[
U_\ell=\sqrt{d_\ell/N}\,V_\ell,
\qquad
U=\begin{bmatrix}U_{\ell_1}\\ \vdots\\ U_{\ell_k}\end{bmatrix},
\qquad
B=UU^T.
\]

Then \(B\) is real symmetric, positive semidefinite, and has rank \(N\). Moreover,

\[
\operatorname{tr}(B_{\ell\ell})
=\|U_\ell\|_F^2=d_\ell.
\]

For a profile \(a\), write \(Z_a=D_aU\). Its column Gram matrix is

\[
Z_a^TZ_a
=\sum_\ell a_\ell U_\ell^TU_\ell
={1\over N}\sum_\ell a_\ell d_\ell I_N
={T_a\over N}I_N.
\]

Hence \(M_a=Z_aZ_a^T\) has exactly \(N\) nonzero eigenvalues, all equal to \(T_a/N\), so

\[
\|M_a\|_F^2={T_a^2\over N}.
\]

The block traces also give \(\langle A_a,M_a\rangle=S_a\). Therefore

\[
\|A_a-M_a\|_F^2
=\|M_a\|_F^2-2S_a+S_a
={T_a^2\over N}-S_a.
\]

The same \(B\) works for every profile. ∎

### Application to the degree-280 certificate

All active degrees have \(d_\ell\ge d_3>N\). The theorem therefore applies to all 146 released profiles at once. In particular, the following proposed strengthening is now closed:

> one common rank-constrained harmonic second-moment matrix, including all shared cross-degree blocks and all individual block traces.

This is stronger than optimizing every profile independently, but it still returns exactly \(L_0\).

### Hostile interpretation

The constructed \(B\) is generally **not** a spherical evaluation moment matrix. It is an explicit counterexample to a proof method, not a cubature construction. Any next joint relaxation must impose constraints not captured by one arbitrary shared second-moment matrix, such as:

- linearization identities for products of harmonics;
- Hankel/catalecticant consistency across polynomial degrees;
- multiplication operators satisfying commutation and \(\sum_jX_j^2=I\);
- localizing matrices for the sphere ideal;
- higher-order moment constraints that force rank-one point evaluations.

Merely sharing more second-moment blocks cannot work.

---

## 3. Strict nonattainment by actual point evaluations

### Theorem 2 — The 146-profile certificate floor is unattainable

No actual mass-one signed atomic rule with at most \(N=66{,}048\) nodes attains the released comparison floor \(L_0\).

Consequently every such actual rule obeys the strict inequality

\[
R_{K_{32}}(Q)>L_0.
\]

### Proof

Every certificate coefficient is positive, and each normalized component separately satisfies its rank floor. If their weighted sum equaled \(L_0\), every active component would have to attain equality.

The certificate contains the two components

\[
(s,r_1)=\left(3,0.005623413251903491\right),
\]

\[
(s,r_2)=\left(3,0.0068129206905796083\right),
\qquad r_1\ne r_2.
\]

Atomic equality for either component forces the rule to have exactly \(N\) equal positive weights and, for every distinct pair of nodes,

\[
L_{3,r_j}(t_{ik})=0,
\qquad t_{ik}=x_i\cdot x_k.
\]

If equality held for both components, then

\[
d_3G_3(t_{ik})+r_1d_4G_4(t_{ik})=0,
\]

\[
d_3G_3(t_{ik})+r_2d_4G_4(t_{ik})=0.
\]

Subtracting and using \(r_1\ne r_2\) gives \(G_4(t_{ik})=0\), and then \(G_3(t_{ik})=0\).

Consecutive Gegenbauer polynomials have no common zero. Indeed, the three-term recurrence expresses \(G_{n-1}\) as a nonzero scalar multiple of a combination of \(G_n\) and \(G_{n+1}\). A common zero of \(G_n,G_{n+1}\) would propagate backward to a zero of \(G_0\equiv1\), a contradiction.

Thus simultaneous equality is impossible. ∎

### Stronger observation

There are 45 repeated \(s\)-values in the certificate, so the contradiction is highly redundant. The \(s=3\) pair alone suffices.

---

## 4. Why strictness is not yet a signed epsilon

A tempting but invalid argument is:

1. equality is impossible;
2. the domain of \(N\)-point rules is compact;
3. therefore the objective is separated from \(L_0\) by a positive distance.

Step 2 fails for unrestricted signed weights.

### Exact coalescing-cancellation counterexample

Take the compact node space \([0,1]\), feature vector \(\phi(x)=(1,x)\), and

\[
Q_\varepsilon=\left(1+{1\over\varepsilon}\right)\delta_0
-{1\over\varepsilon}\delta_\varepsilon.
\]

This is mass one, but

\[
\|Q_\varepsilon\|_{\mathrm{TV}}=1+{2\over\varepsilon}\to\infty.
\]

Its moment matrix is

\[
M_\varepsilon
=\sum_iw_i\phi(x_i)\phi(x_i)^T
=\begin{pmatrix}1&-1\\-1&-\varepsilon\end{pmatrix}
\longrightarrow
\begin{pmatrix}1&-1\\-1&0\end{pmatrix}.
\]

Thus bounded low-order moments do not control signed weights: arbitrarily large positive and negative masses can coalesce while the moment matrix converges.

### Correct compactness corollary

For every fixed total-variation bound \(B<\infty\), the class

\[
\sum_i|w_i|\le B,
\qquad m\le N,
\]

is compact after padding with zero-weight atoms. By Theorem 2 and continuity, there exists an \(\varepsilon(B)>0\) such that

\[
R_{K_{32}}(Q)\ge L_0+\varepsilon(B).
\]

This is existential, not numerically useful. Obtaining an explicit \(\varepsilon(B)\), or eliminating \(B\), remains open.

---

## 5. Explicit positive-weight epsilon

Although unrestricted signed weights remain noncompact, positive weights give an exact quantitative separation from the two \(s=3\) components.

Let their exact normalized coefficients be

\[
\alpha={y_1\over F_1},
\qquad
\beta={y_2\over F_2},
\]

and define

\[
h(t)=\alpha L_{3,r_1}(t)^2+\beta L_{3,r_2}(t)^2.
\]

Because the two linear forms have no common zero,

\[
\mu=\min_{-1\le t\le1}h(t)>0.
\]

For nonnegative weights \(w_i\), put \(S=\sum_iw_i^2\ge1/N\). If \(A=h(1)\), then the excess of these two components above their separate rank floors is

\[
\sum_{i,j}w_iw_jh(t_{ij})-{A\over N}.
\]

The diagonal contributes \(AS\), while the off-diagonal contribution is at least \(\mu(1-S)\). Hence

\[
\sum_{i,j}w_iw_jh(t_{ij})-{A\over N}
\ge (A-\mu)(S-1/N)+\mu(1-1/N)
\ge\mu(1-1/N).
\]

### Exact rational lower bound for \(\mu\)

On \(S^{255}\),

\[
G_3(x)={x(86x^2-1)\over85},
\]

\[
G_4(x)={22360x^4-516x^2+1\over21845}.
\]

The verifier checks the exact Bézout identity

\[
172x(16640x^2-319)G_3(x)
-257(11008x^2-85)G_4(x)=1.
\]

On \([-1,1]\), coefficient \(\ell_1\) bounds give

\[
|A(x)|\le2{,}916{,}948,
\qquad
|B(x)|\le2{,}850{,}901,
\]

and therefore

\[
G_3(x)^2+G_4(x)^2
\ge {1\over16{,}636{,}222{,}146{,}505}.
\]

Writing \(h=u^TQu\), where

\[
u=(d_3G_3,d_4G_4)^T,
\]

and

\[
Q=\alpha\binom{1}{r_1}(1,r_1)
+\beta\binom{1}{r_2}(1,r_2),
\]

we use

\[
\lambda_{\min}(Q)\ge{\det Q\over\operatorname{tr}Q}
={\alpha\beta(r_1-r_2)^2
\over\alpha(1+r_1^2)+\beta(1+r_2^2)}.
\]

Since \(d_4>d_3\), exact arithmetic yields

\[
\mu\ge1.22119189293389862469358725957958698\times10^{-23}.
\]

Thus

\[
\boxed{
R_{K_{32}}(Q)
\ge L_0+1.22117340347331035709086358002442135\times10^{-23}
}
\]

for every nonnegative mass-one rule on at most 66,048 nodes.

This increment is only

\[
5.0178464701877801\times10^{-17}
\]

of the certified Kerdock upper endpoint. Its value is conceptual: repeated profiles can produce an explicit realizability gap once sign cancellation is removed.

---

## 6. Competition translation

| Class / result | Risk fraction of Kerdock upper | Same-cost gain cap | Cost fraction needed for a 4.34x adjusted gap |
|---|---:|---:|---:|
| Pure compute, no MSE gain | 1 | 1 | 0.2304147465 |
| Current arbitrary-signed theorem | \(\ge0.9370459569114724\) | \(\le1.0671835171201482\) | \(\le0.2458948196129374\) |
| Actual arbitrary-signed rules, this round | **strictly** \(>0.9370459569114724\) | **strictly** \(<1.0671835171201482\) | **strictly** \(<0.2458948196129374\) |
| Positive rules, exact two-profile epsilon | \(\ge0.93704595691147245827\) | \(\le1.06718351712014818\) | \(\le0.24589481961293737\) |

The strict arbitrary-signed result changes no displayed competition threshold because no uniform epsilon is certified.

---

## 7. Scope and survival matrix

| Claim | Estimator / relaxation class | Width | Status | Counterexample / limitation |
|---|---|---|---|---|
| Degree-280 floor | Static, network-independent, linear, mass-one signed, \(m\le66{,}048\) | Infinite width, depth 32, dimension 256 | Inherited certificate | External independent certificate review remains a release gate |
| Shared-Gram sharpness | One arbitrary common symmetric rank-\(N\) harmonic second-moment matrix with fixed block traces | Not a network-width claim | **Proved exact** | Optimizer need not be point-realizable |
| Strict nonattainment | Actual atomic rules in the signed theorem class | Infinite width | **Proved** | Strictness alone gives no numerical epsilon |
| Explicit epsilon | Nonnegative mass-one atomic rules, \(m\le N\) | Infinite width | **Exact rational certificate** | Does not cover negative weights; increment is negligible |
| Existential \(\varepsilon(B)\) | Signed rules with total variation bounded by \(B\) | Infinite width | **Proved by compactness** | Nonconstructive and depends on \(B\) |
| Exact Kerdock optimality | All algorithms / all signed rules | Any | **Not proved** | Adaptive, nonlinear, finite-width and unbounded-TV escape classes remain |

---

## 8. Next theorem object

The highest-value next relaxation should use a single truncated moment functional \(\mathcal L\) on polynomials, with matrices

\[
H_k(\mathcal L)=\big(\mathcal L(p_i p_j)\big)_{\deg p_i,\deg p_j\le k},
\]

and enforce at least:

1. the sphere ideal
   \[
   \mathcal L\!\left((\|x\|^2-1)p(x)\right)=0;
   \]
2. exact Clebsch–Gordan/product linearization linking harmonic blocks;
3. shared Hankel entries whenever two polynomial products are identical;
4. rank/flat-extension conditions consistent with at most \(N\) atoms;
5. signed-measure handling that does not incorrectly impose PSD on \(H_k\).

The key warning is item 5: PSD moment matrices are valid for positive measures, not arbitrary signed rules. A valid signed relaxation may require a Jordan decomposition with bounded negative mass, a difference of two PSD moment sequences, or algebraic rank constraints without PSD.

Theorem 1 says that any relaxation stopping before these product identities is already exhausted.

---

## 9. Stop condition and coordinator handoff

The Agent-1 stop condition is met by a rigorous obstruction:

> **The declared shared-second-moment realizability relaxation is exactly sharp, even when every certificate profile shares one full harmonic block matrix.**

At the same time, the actual certificate floor is strictly unattainable. The gap between these statements pinpoints the missing ingredient: not “more shared matrices,” but the algebraic variety of point evaluations.

No protected or official cohort was opened. No broad experiment was run. All new constants were checked with exact rational arithmetic in the accompanying standard-library verifier and independently replayed from normalized Gegenbauer polynomials in SymPy.
