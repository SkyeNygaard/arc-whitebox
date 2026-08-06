# Stronger Claims: Proof and Disproof Attempts

**Date:** 2026-07-30  
**Scope:** Mathematical strengthening of the WHestBench/Kerdock paper claims, based on the audited theorem and experiment packages.

## Executive status

| Priority | Candidate claim | Status |
|---|---|---|
| 1 | Adaptive-information value and observability ratio | **PROVED in a general Hilbert-space form**; mathematically standard, but useful as the organizing principle |
| 2 | MUB support extremality for general kernels | **PROVED under sharp association-value sign conditions**; unconditional version **DISPROVED** by a PSD zonal-kernel counterexample |
| 3 | Same-design absolute-phase impossibility | **PROVED under the common-bias/equal-loading model**; universal version **DISPROVED** outside that model |
| 4 | Static-to-adaptive escape theorem | **PROVED as an exact corollary** of T22/T27; useful, but not a deep independent theorem |
| 5 | Finite-width transfer | A deterministic perturbation theorem is **PROVED**; qualitative transfer is **DISPROVED**; width-256 quantitative transfer remains open |
| 6 | Residual-kernel recertification | The residual-kernel identity is **PROVED**; persistence of Kerdock optimality is **DISPROVED** without recertification |
| 7 | Control-nullspace theorem | **PROVED**, including adaptive global parameter selection inside a uniformly annihilated family |

---

## 1. Adaptive-information value theorem

### Theorem 1.1 — unrestricted value of runtime information

Let \((\Omega,\mathcal F,\mathbb P)\) be a probability space, let \(H\) be a real Hilbert space, and let the baseline error be \(e\in L^2(\Omega;H)\). Let \(\mathcal G\subseteq\mathcal F\) be the runtime information available to an estimator. Among all \(\mathcal G\)-measurable corrections \(c\in L^2(\mathcal G;H)\),

\[
\inf_c \mathbb E\|e-c\|^2
=
\mathbb E\|e\|^2-
\mathbb E\|\mathbb E[e\mid\mathcal G]\|^2.
\]

The unique optimal correction is

\[
c^*_{\mathcal G}=\mathbb E[e\mid\mathcal G].
\]

Define the **correction value** of the information by

\[
V(\mathcal G;e)
:=
\mathbb E\|\mathbb E[e\mid\mathcal G]\|^2.
\]

#### Proof

Conditional expectation is the orthogonal projection of \(e\) onto the closed subspace \(L^2(\mathcal G;H)\). Put \(m=\mathbb E[e\mid\mathcal G]\). For every admissible \(c\),

\[
e-c=(e-m)+(m-c),
\]

and the two terms are orthogonal in \(L^2(\Omega;H)\). Therefore

\[
\mathbb E\|e-c\|^2
=
\mathbb E\|e-m\|^2+
\mathbb E\|m-c\|^2.
\]

The minimum is attained uniquely at \(c=m\), and

\[
\mathbb E\|e\|^2
=
\mathbb E\|e-m\|^2+
\mathbb E\|m\|^2.
\]

### Corollary 1.2 — exact incremental value and data processing

If \(\mathcal G_1\subseteq\mathcal G_2\), then

\[
V(\mathcal G_2;e)-V(\mathcal G_1;e)
=
\mathbb E\left\|
\mathbb E[e\mid\mathcal G_2]
-
\mathbb E[e\mid\mathcal G_1]
\right\|^2
\ge 0.
\]

Consequently, replacing observations by any statistic cannot increase correction value.

#### Proof

Apply the Pythagorean identity to the nested orthogonal projections. For a statistic \(T\) measurable with respect to \(\mathcal G\), \(\sigma(T)\subseteq\mathcal G\), so monotonicity gives \(V(\sigma(T);e)\le V(\mathcal G;e)\).

### Corollary 1.3 — restricted correction classes

Let \(\mathcal C\) be any closed linear subspace of admissible corrections in \(L^2(\Omega;H)\). Then the optimal correction is \(P_{\mathcal C}e\), and its attainable gain is

\[
V(\mathcal C;e)=\|P_{\mathcal C}e\|_{L^2}^2.
\]

For nested classes \(\mathcal C_1\subseteq\mathcal C_2\), value is monotone. This allows a class-specific **observability ratio**

\[
\operatorname{OR}(\mathcal C_{\rm runtime},\mathcal C_{\rm oracle})
=
\frac{\|P_{\mathcal C_{\rm runtime}}e\|_{L^2}^2}
{\|P_{\mathcal C_{\rm oracle}}e\|_{L^2}^2}
\in[0,1],
\]

when the denominator is nonzero and the runtime class is contained in the oracle class.

### Corollary 1.4 — scalar candidate direction

Suppose \(u\) is available at runtime and the admissible corrections are \(c=\alpha u\), with \(\alpha\) measurable with respect to runtime information \(\mathcal G\). Then, where \(\mathbb E[\|u\|^2\mid\mathcal G]>0\),

\[
\alpha^*(\mathcal G)
=
\frac{\mathbb E[\langle e,u\rangle\mid\mathcal G]}
{\mathbb E[\|u\|^2\mid\mathcal G]},
\]

and the gain is

\[
\mathbb E\left[
\frac{\mathbb E[\langle e,u\rangle\mid\mathcal G]^2}
{\mathbb E[\|u\|^2\mid\mathcal G]}
\right].
\]

This recovers the existing conditional-selector theorem.

### Counterexample to magnitude-based observability

Let \(e=SA v\), where \(v\) is fixed, \(S\in\{-1,+1\}\) is symmetric, and \(A>0\) is observed at runtime but independent of \(S\). Runtime information perfectly reveals the error magnitude, yet

\[
\mathbb E[e\mid A]=0,
\qquad
V(\sigma(A);e)=0.
\]

Thus perfect magnitude prediction can have zero correction value when signed phase is absent.

### Assessment

**Proved.** The mathematics is standard conditional-expectation/projection theory; it should not be advertised as a new probability theorem. The research contribution is the identification of this value functional as the correct normalization for adaptive numerical-integration observables and empirical oracle gaps.

---

## 2. General MUB support-extremality theorem

### Setup

Let \(B_1,\dots,B_M\) be mutually unbiased orthonormal bases of \(\mathbb R^d\), viewed as antipodal projective lines. Let \(k\) be an antipodally symmetrized zonal kernel. Assume its value depends on a pair of lines only through the three classes

\[
A=k(1),\qquad
O=k(0),\qquad
C=k(1/\sqrt d).
\]

Give line \((b,i)\) real weight \(w_{bi}\), with \(\sum_{b,i}w_{bi}=1\), and define \(S_b=\sum_iw_{bi}\).

### Theorem 2.1 — association-scheme support extremality

Put

\[
a=A-O,
\qquad
b=O-C.
\]

Assume

\[
a>0,
\qquad
b<0,
\qquad
a+bd>0.
\]

For a budget of at most \(P\) active lines, every minimum-energy real-weight rule is obtained by

1. using all \(P\) available lines;
2. filling \(q=\lfloor P/d\rfloor\) complete bases;
3. using at most one additional partial basis of size \(s=P-qd\);
4. assigning equal positive line weights within each active basis;
5. assigning positive basis masses proportional to
   \[
   \frac{1}{(O-C)+(A-O)/r_b}.
   \]

The result is unique up to permutations of bases and lines, except at degenerate budgets or kernel equalities.

#### Proof

The kernel energy, up to a rule-independent constant, is

\[
R(w)
=(O-C)\sum_bS_b^2+(A-O)\sum_{b,i}w_{bi}^2.
\]

If basis \(b\) has \(r_b\) active lines, Cauchy--Schwarz gives

\[
\sum_iw_{bi}^2\ge \frac{S_b^2}{r_b},
\]

with equality exactly at equal within-basis weights. Hence

\[
R(w)\ge \text{const}+\sum_b c(r_b)S_b^2,
\qquad
c(r)=b+\frac ar.
\]

Because \(b<0\) and \(a+bd>0\), \(c(r)>0\) for every \(1\le r\le d\). Minimizing over \(S_b\) subject to \(\sum_bS_b=1\) gives

\[
S_b=
\frac{1/c(r_b)}{\sum_j1/c(r_j)},
\qquad
R_{\min}(r_1,\dots,r_M)
=
\text{const}+rac1{\sum_bh(r_b)},
\]

where

\[
h(r)=\frac1{c(r)}=\frac{r}{a+br},
\qquad h(0)=0.
\]

On \([0,d]\),

\[
h'(r)=\frac{a}{(a+br)^2}>0,
\qquad
h''(r)=\frac{-2ab}{(a+br)^3}>0.
\]

Thus all available support is used, and \(h\) is strictly convex. Maximizing \(\sum_bh(r_b)\) over integer counts with fixed sum pushes mass to extreme points. Equivalently, whenever \(0<x\le y<d\), transferring one line from the smaller partial basis to the larger increases the objective:

\[
h(x-1)+h(y+1)>h(x)+h(y).
\]

Iteration leaves only complete bases, at most one partial basis, and empty bases. The formulas above then give positive basis masses and positive equal line weights.

### Sharpness of the sign condition: PSD counterexample

The unconditional statement “complete bases are optimal for every positive-definite zonal kernel” is false.

Take \(d=4\), the standard basis and a real Hadamard basis, and the positive-definite zonal kernel

\[
k(t)=1+\lambda P_4(t),
\qquad \lambda>0,
\]

where the normalized degree-4 Gegenbauer polynomial on \(S^3\) is

\[
P_4(t)=\frac{16t^4-12t^2+1}{5}.
\]

This kernel is positive definite because it is a nonnegative combination of the degree-0 and degree-4 spherical-harmonic reproducing kernels. Its association values are

\[
A=1+\lambda,
\qquad
O=1+\lambda/5,
\qquad
C=1-\lambda/5.
\]

Therefore

\[
a=4\lambda/5>0,
\qquad
b=2\lambda/5>0.
\]

For a four-line budget, one complete basis gives

\[
H_{\rm complete}=h(4)=\frac{5}{3\lambda},
\]

whereas two lines in each of two bases give

\[
H_{\rm balanced}=2h(2)=\frac{5}{2\lambda}>H_{\rm complete}.
\]

Since risk is constant plus \(1/H\), the balanced support is strictly better. Thus \(O<C\), not merely positive definiteness, drives complete-basis concentration. When \(O>C\), the allocation function is concave and balance is favored instead.

### Assessment

**Proved with sharp sufficient conditions; unconditional version disproved.** This is likely the strongest near-ready mathematical generalization. It can be presented as an association-scheme theorem and instantiated by the depth-32 ReLU kernel.

---

## 3. Same-design absolute-phase impossibility

### Theorem 3.1 — gauge non-identifiability and minimax lower bound

Let

\[
Z_i=\mu+b+\varepsilon_i,
\qquad i=1,\dots,k,
\]

where the joint law of \((\varepsilon_1,\dots,\varepsilon_k)\) does not depend on \((\mu,b)\). Then the transformation

\[
(\mu,b)\mapsto(\mu+t,b-t)
\]

leaves the law of the complete observation vector \(Z\) unchanged.

For every estimator \(T(Z)\) of \(\mu\), and every \(t\), at least one of the two gauge-equivalent parameter points has risk at least

\[
\frac{\|t\|^2}{4}.
\]

#### Proof

The observation law depends on \((\mu,b)\) only through \(\eta=\mu+b\), so the two parameter points have identical distributions. Pointwise, for any value \(x=T(Z)\),

\[
\|x-\mu\|^2+
\|x-(\mu+t)\|^2
=
2\left\|x-\mu-\frac t2\right\|^2+
\frac{\|t\|^2}{2}
\ge
\frac{\|t\|^2}{2}.
\]

Taking expectation under the common observation law shows that the sum of the two risks is at least \(\|t\|^2/2\), hence their maximum is at least \(\|t\|^2/4\).

### Corollary 3.2 — centered diagnostics contain no absolute phase

Let

\[
D_i=Z_i-\bar Z.
\]

Then \(D\) is a function only of the centered noise and its law is independent of both \(\mu\) and \(b\). No statistic measurable only from centered folds, rotations, jackknife replicates, or split halves can identify the sign or magnitude of the common defect in this model.

Under a symmetric two-point prior with \(b=\pm t/2\) and \(\mu\) adjusted to keep \(\mu+b\) fixed, every centered-diagnostic correction \(c(D)\) has

\[
\mathbb E\langle b,c(D)\rangle=0.
\]

It may reduce replicate noise, but it cannot reduce the shared-bias component on average.

### Counterexample outside the common-bias model

The universal claim is false. If

\[
Z_1=\mu+b+\varepsilon_1,
\qquad
Z_2=\mu-b+\varepsilon_2,
\]

then

\[
Z_1-Z_2=2b+(\varepsilon_1-\varepsilon_2),
\]

so a centered difference directly identifies the bias. More generally, unequal known loadings, bias-dependent noise, structural restrictions on \(b\), or an external reference can restore identifiability.

### Assessment

**Proved under the explicit equal-loading/common-bias model; universal version disproved.** The useful research statement is a class-restricted minimax theorem, not a universal claim about every cross-fit or paired estimator.

---

## 4. Static-to-adaptive escape corollary

Let \(R_K\) be Kerdock risk and let \(R_*\) be the infimum over the T22 class. The certified statement is

\[
R_K\le(1+\delta)R_*,
\qquad
\delta=0.0002336550102949.
\]

### Corollary 4.1 — exact escape threshold

Every estimator in the T22 class satisfies

\[
R\ge \frac{R_K}{1+\delta}.
\]

Therefore any method with fractional risk improvement over Kerdock greater than

\[
\frac{\delta}{1+\delta}
=0.0002336004283844
=0.02336004283844\%
\]

must violate at least one T22 assumption.

At the reported numerical bounds,

\[
R_K\le2.433660357543006\times10^{-7},
\qquad
R_*\ge2.433091853440941\times10^{-7},
\]

so the absolute certified room is at most

\[
5.685041020648603\times10^{-11}.
\]

T27 adds that, inside the fixed Kerdock-line universe, arbitrary real unequal and signed line weights still cannot improve on the complete-basis allocation.

### What the corollary does and does not imply

A material winner must leave at least one covered regime, for example through network dependence, pilot adaptation, nonlinear processing, new nodes, finite-width-specific structure, a transformed residual, or signed weights outside the fixed Kerdock support universe.

It does **not** prove that adaptation is the only escape. Static arbitrary-node signed rules, static nonlinear estimators, and finite-width-specific static rules are outside the present closure.

### Assessment

**Proved.** This is an exact and useful framing corollary, but not a deep independent theorem.

---

## 5. Finite-width transfer

### Theorem 5.1 — deterministic kernel perturbation transfer

Let \(P\) be the target distribution, let a rule be \(Q=\sum_iw_i\delta_{x_i}\), and write

\[
\nu_Q=P-Q.
\]

For two kernels \(K\) and \(\widetilde K\), define

\[
R_K(Q)=\iint K(x,y)\,d\nu_Q(x)d\nu_Q(y).
\]

If

\[
\|K-\widetilde K\|_\infty\le\varepsilon,
\qquad
\sum_i|w_i|\le B,
\]

then

\[
|R_K(Q)-R_{\widetilde K}(Q)|
\le
\varepsilon(1+B)^2.
\]

#### Proof

The total variation norm satisfies \(\|\nu_Q\|_{\rm TV}\le1+B\). Hence

\[
\left|\iint(K-\widetilde K)\,d\nu_Qd\nu_Q\right|
\le
\|K-\widetilde K\|_\infty\|\nu_Q\|_{\rm TV}^2.
\]

### Corollary 5.2 — transfer of additive near-optimality

If every rule in a class has \(\ell_1\) weight norm at most \(B\), and \(Q_K\) has infinite-width additive suboptimality at most \(g\), then

\[
R_{\widetilde K}(Q_K)-
\inf_QR_{\widetilde K}(Q)
\le
 g+2\varepsilon(1+B)^2.
\]

For nonnegative mass-one rules, \(B=1\), giving

\[
\text{finite-kernel excess}\le g+8\varepsilon.
\]

### Why the present certificate does not transfer automatically

The certified additive room is only

\[
g\le5.685041020648603\times10^{-11}.
\]

To keep the perturbation term below this scale using the uniform bound would require roughly

\[
\varepsilon<g/8
\approx7.1063\times10^{-12}.
\]

No such width-256 ensemble-kernel bound is presently supplied.

### Counterexample to qualitative transfer

Qualitative convergence \(K_m\to K_\infty\) does not preserve the optimizer. Let two distinct rules have linearly independent error functionals \(L_0,L_1\). Choose bounded functions \(\phi,\psi\) such that

\[
L_0\phi=0,
\quad L_1\phi=1,
\qquad
L_0\psi=1,
\quad L_1\psi=0.
\]

Set

\[
K_\infty=g\,\phi\otimes\phi,
\qquad
K_m=K_\infty+\varepsilon\,\psi\otimes\psi.
\]

Both are positive semidefinite. Under \(K_\infty\), rule 0 has risk 0 and rule 1 has risk \(g\). Under \(K_m\), their risks are \(\varepsilon\) and \(g\). For \(\varepsilon>g\), the optimizer reverses. Taking \(g\) arbitrarily small makes the required PSD perturbation arbitrarily small.

Thus an infinite-width optimum with a tiny gap is not structurally stable without a quantitative perturbation bound.

### Assessment

**Conditional theorem proved; naive transfer disproved; width-256 result open.** A publishable finite-width result needs an explicit bound on the finite-width second-moment kernel or on the relevant Kerdock association values and kernel means.

---

## 6. Residual-kernel recertification

### Theorem 6.1 — residual-kernel identity

Let \(f_\theta\) be a random integrand and let \(g_\theta\) be any network-dependent function whose exact integral \(I(g_\theta)\) is available. Define

\[
h_\theta=f_\theta-g_\theta
\]

and the estimator

\[
\widehat I_Q(f_\theta)
=I(g_\theta)+Q(h_\theta).
\]

Then the error is exactly

\[
\widehat I_Q(f_\theta)-I(f_\theta)
=(Q-I)h_\theta.
\]

If \(\mathbb E|h_\theta(x)h_\theta(y)|\) is integrable, define the residual second-moment kernel

\[
K_{\rm res}(x,y)
=
\mathbb E[h_\theta(x)h_\theta(y)].
\]

Then

\[
\mathbb E\left[
\bigl(\widehat I_Q(f_\theta)-I(f_\theta)\bigr)^2
\right]
=
\iint K_{\rm res}(x,y)\,d\nu_Q(x)d\nu_Q(y).
\]

#### Proof

Use the algebraic error identity, square, take expectation, and exchange expectation with the two linear functionals. The kernel is positive semidefinite because it is a second-moment kernel.

### Consequence

After a network-dependent analytic transformation, quadrature design is again a kernel-energy problem—but for \(K_{\rm res}\), not for the original deep-ReLU kernel. If the residual kernel is zonal and its MUB association values satisfy Theorem 2.1, complete-basis extremality can be recertified.

### Counterexample to automatic persistence of Kerdock optimality

Choose a residual family with rank-one kernel

\[
K_{\rm res}=\psi\otimes\psi
\]

such that another rule integrates \(\psi\) exactly while Kerdock does not. The other rule has zero residual risk and Kerdock has positive residual risk. Therefore the original Kerdock certificate says nothing about the transformed problem.

If the construction of \(g\) itself depends on the candidate rule, there may not be one common residual kernel for comparing rules; each candidate induces its own residual law. The transformation or information protocol must be frozen before recertification.

### Assessment

**Identity proved; automatic transfer disproved.** “Certify, transform, derive the residual kernel, recertify” is a valid research methodology, not a theorem that Kerdock remains optimal.

---

## 7. Control-nullspace theorem

### Theorem 7.1 — uniform control annihilation

For a fixed quadrature rule \(Q\) and integral \(I\), define its exactness/null space

\[
\mathcal N_Q=\{g:I(g)=Q(g)\}.
\]

For every baseline function \(f\), every \(g\in\mathcal N_Q\), and every scalar \(\alpha\),

\[
I(\alpha g)+Q(f-\alpha g)=Q(f).
\]

Hence every control family contained in \(\mathcal N_Q\) is pathwise useless, regardless of coefficient tuning.

More strongly, suppose \(\{g_\eta:\eta\in\Xi\}\subseteq\mathcal N_Q\) uniformly. The parameter \(\widehat\eta\) may be selected adaptively from the network, pilot observations, or quadrature outputs. Provided the same resulting function \(g_{\widehat\eta}\) is evaluated by \(I\) and \(Q\),

\[
I(g_{\widehat\eta})-Q(g_{\widehat\eta})=0
\]

pathwise. Independence of the fitted parameter is unnecessary.

#### Proof

Immediate from linearity and the uniform identity defining \(\mathcal N_Q\).

### Application

Low-degree radialized polynomials on the Kerdock 5-design and the audited global-parameter, bias-free one-hidden-layer ReLU-Stein family lie in this null space. The useful warning is therefore stronger than “the tested fit failed”: increasing model expressivity within a uniformly annihilated control family cannot help.

Node-specific parameters, leave-one-node-out fields, incorrect radialization, or functions outside the uniformly annihilated family are not covered.

### Assessment

**Proved.** This is simple but useful, and it strengthens the interpretation of the exact annihilation results.

---

## Recommended paper hierarchy

### Main theorem contribution

1. Association-scheme/MUB support extremality under explicit kernel sign conditions.
2. Depth-32 ReLU instantiation and T22 arbitrary-node nonnegative near-optimality.

### General adaptive-correction theory

3. Hilbert-space information value, restricted observability ratio, and monotonicity.
4. Common-bias gauge non-identifiability and class-restricted minimax lower bound.
5. Downstream-weighted and nonlinear replay conditions from the existing appendix.

### Boundary and constructive program

6. Exact escape threshold from the static class.
7. Residual-kernel recertification methodology.
8. Finite-width transfer as the principal unresolved theorem, with the perturbation lemma making the required quantitative target explicit.

## Bottom line

The strongest defensible synthesis is not a universal impossibility theorem. It is a **two-boundary theory**:

- **Geometric boundary:** within specified static support/weight classes, kernel energy is certified near-optimal or exactly optimized.
- **Information boundary:** within specified runtime-information classes, attainable correction gain is the squared Hilbert projection of the error onto the observable correction space; same-design centered diagnostics cannot recover a common absolute phase.

A material improvement must cross at least one boundary by introducing new geometry, new information, a transformed residual, or finite-width-specific structure.
