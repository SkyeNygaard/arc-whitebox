# Hostile-audit patched edition

This copy applies the 2026-07-30 hostile disproof audit. It repairs the conditional-Haar theorem, the local-density cubic ReLU bound, the biased-replication corollary, the zero-capacity observability-ratio edge case, and an optimizer-instability wording issue. It does **not** replace the separate T29/T38 errata, which concern round-two theorem notes.

---

---
title: "WHestBench Complete Proof Package"
subtitle: "Geometric optimality, correction observability, exact control means, and frozen empirical certificates"
author: "Research synthesis, 2026-07-30"
date: "2026-07-30"
geometry: margin=0.85in
fontsize: 10pt
header-includes:
  - \usepackage{amsmath,amssymb,mathtools,booktabs,longtable}
  - \usepackage{microtype}
  - \usepackage[hidelinks]{hyperref}
---

# 1. Executive theorem map

This document gives the strongest rigorous synthesis supported by the WHestBench proof and experiment artifacts. It deliberately distinguishes four statuses:

1. **Analytically proved:** the proof is included below.
2. **Computer-assisted certified dependency:** a separate interval/exact-arithmetic package supplies the proof; this document states the theorem, its trust base, and the logical consequences.
3. **Finite frozen certificate:** a deterministic verifier recomputes a claim from the saved binary artifacts.
4. **Empirical only:** the result motivates a theorem or closes a tested implementation, but does not imply population-level impossibility.

The central conclusion is not that no competition winner exists. The valid conclusion is a **two-boundary theorem**:

- **Geometric boundary.** In the specified infinite-width, static, network-independent cubature classes, complete Kerdock is certified near-optimal, and is exactly optimal inside the fixed Kerdock-line universe even when arbitrary real line weights are allowed.
- **Information boundary.** For any explicitly specified runtime information class, the best possible adaptive correction is the Hilbert projection of the baseline error onto that information class. Same-design centered information, Haar-orientation-invariant information, or any feature class with a phase-flip symmetry can have exactly zero correction value even when a large per-network oracle exists.

A material improvement must cross at least one boundary by introducing new geometry, new absolute-phase information, a transformed residual kernel, finite-width-specific structure, or a computationally richer white-box procedure.

## 1.1 Claim-status table

| ID | Claim | Status | Scope |
|---|---|---|---|
| G1 | Kerdock is within `0.02336550102948%` of the static nonnegative optimum | Computer-assisted certified | Infinite-width depth-32 ReLU kernel, `d=256`, at most 66,048 nodes, nonnegative mass-one weights, network-independent rule |
| G2 | Tightened auxiliary-LP comparison gives `0.023324172950039%` relative excess | Computer-assisted certified | Same limiting kernel; all-degree admissible auxiliary LP |
| G3 | Complete bases plus at most one partial basis exactly optimize arbitrary real weights on the fixed Kerdock-line universe | Analytically proved | 33,024 symmetrized lines; static linear rules |
| G4 | Any improvement exceeding `0.02331873404818%` must leave the tightened static class | Analytically proved corollary | Same as G2 |
| I1 | Runtime information value equals squared conditional-projection norm | Analytically proved | Any square-integrable Hilbert-valued error |
| I2 | Optimal adaptive coefficients in a finite correction dictionary satisfy conditional normal equations | Analytically proved | Finite-dimensional dictionary, square-integrable features |
| I3 | Phase-flip symmetry forces zero correction value | Analytically proved | Explicit measure-preserving involution and invariant information |
| I4 | Conditionally Haar-random relative orientation makes orientation-blind corrections useless | Analytically proved after correction | The conditional law of the design rotation given the integrand and runtime information is Haar; mass-one cubature |
| I5 | Same-design common bias is minimax non-identifiable | Analytically proved | Equal-loading observation model |
| I6 | Full weights defeat a pure information-theoretic impossibility theorem | Analytically proved | Target/error deterministic from complete weights |
| R1 | Exact correction-risk, shrinkage, and downstream replacement formulas | Analytically proved | Hilbert-space scoring model |
| R2 | ReLU nonlinear remainder is supported on gate crossings and has a cubic second-moment bound | Analytically proved | Explicit density assumption near zero |
| C1 | Controls in the quadrature exactness space are pathwise no-ops, even with adaptive global parameters | Analytically proved | Uniformly annihilated control family |
| C2 | Symmetrized spherical Poisson controls have exact mean one and nonzero all-even harmonics | Analytically proved | Sphere, `|r|<1` |
| C3 | Biased projected-ReLU controls have a closed-form exact spherical mean | Analytically proved | Uniform fixed-radius sphere |
| E1 | Ordinary independent replication is score-neutral under linear MSE-times-compute accounting | Analytically proved | Equal costs, independent errors; correlated extension included |
| F1 | Kernel near-optimality transfers only under a quantitative uniform finite-width kernel bound | Analytically proved | Bounded total variation of rules |
| F2 | Analytic-plus-residual methods require recertification using the residual kernel | Analytically proved | Exact integral of the analytic component |
| X1 | Frozen Poisson, projected-ReLU, and signed-probe candidates fail their stated gates | Finite frozen certificate | Saved IEEE-754 arrays and frozen scripts only |
| X2 | A universal no-winning-estimator theorem follows from the experiments | **False / not proved** | Countered by the full-information theorem and open estimator classes |

# 2. Mathematical setup

Let 

- $(\Omega,\mathcal F,\mathbb P)$ be the randomness space;
- $H$ be the real Hilbert space of scored output coordinates;
- $I(f)$ be the true spherical integral of an integrand $f$;
- $Q(f)=\sum_{i=1}^m w_i f(x_i)$ be a cubature rule with $\sum_i w_i=1$;
- $\widehat y_0$ be the protected baseline estimate;
- $y$ be the true target;
- $e=\widehat y_0-y\in L^2(\Omega;H)$ be baseline error.

A correction $c$ is applied as $\widehat y=\widehat y_0-c$, so its risk is

$$
R(c)=\mathbb E\|e-c\|_H^2.
$$

The competition-relevant empirical ratios use

$$
\text{raw ratio}=\frac{\text{candidate MSE}}{\text{baseline MSE}},
\qquad
\text{adjusted ratio}=\text{raw ratio}\times\text{compute ratio}.
$$

The cited `4.34x` improvement target corresponds to an adjusted ratio at most

$$
\frac1{4.34}=0.2304147465\ldots.
$$

# 3. Geometric boundary

## 3.1 Imported certified result: static nonnegative near-optimality

For the normalized infinite-width depth-32 ReLU kernel in dimension $256$, the T22 release proves a one-sided bound of the form

$$
R_K\le (1+\delta_{22})R_*,
\qquad
\delta_{22}=0.0002336550102948\ldots,
$$

where $R_K$ is complete-Kerdock ensemble MSE and $R_*$ is the infimum over static, network-independent, nonnegative mass-one cubature rules using at most 66,048 spherical nodes.

The release is a computer-assisted proof, not a proof-assistant formalization. Its documented trust base includes exact-rational witnesses, directed rounding, 1,421 certified pointwise intervals, a directed kernel-mean enclosure, one-sided ratio logic, and a 59-entry manifest. The stale two-sided artifact is excluded.

The later all-degree auxiliary-LP certificate gives the stronger comparison

$$
R_K\le (1+\delta_{30})R_*,
\qquad
\delta_{30}=0.00023324172950039,
$$

or `0.023324172950039%` relative excess. Its separate proof uses the degree-five Hermite interpolant at the roots of

$$
22102t^3+21930t^2-87t-85=0,
$$

strict positivity of $K_{32}^{(6)}$, strict negativity of every unused reduced cost above degree five, positive nonconstant Gegenbauer coefficients, and exact primal-dual equality.

These are imported certified dependencies. The bundled verifier in this package does not rerun their interval engines.

## 3.2 Exact structural escape threshold

**Corollary 3.1.** Every estimator in the tightened static nonnegative class satisfies

$$
R\ge \frac{R_K}{1+\delta_{30}}.
$$

Hence its fractional improvement over Kerdock is at most

$$
1-\frac1{1+\delta_{30}}
=
\frac{\delta_{30}}{1+\delta_{30}}
=
0.0002331873404817984\ldots,
$$

which is

$$
0.02331873404817984\ldots\%.
$$

**Proof.** Since $R_*\ge R_K/(1+\delta_{30})$ and every rule in the class has risk at least $R_*$, the first inequality follows. Subtract its ratio to $R_K$ from one. $\square$

Therefore any competition-scale improvement necessarily violates at least one assumption: nonnegativity, static/network independence, linear cubature, limiting-kernel modeling, the node budget, or the original function/kernel.

## 3.3 MUB/Kerdock-line support extremality

Let $B_1,\ldots,B_M$ be mutually unbiased orthonormal bases of $\mathbb R^d$, viewed as antipodal projective lines. Suppose an antipodally symmetrized zonal kernel takes only three association values on this universe:

$$
A=k(1),\qquad O=k(0),\qquad C=k(1/\sqrt d).
$$

Give line $(b,i)$ real weight $w_{bi}$, with total mass one, and define basis masses

$$
S_b=\sum_i w_{bi}.
$$

Up to a rule-independent constant, the kernel energy is

$$
R(w)= (O-C)\sum_b S_b^2 +(A-O)\sum_{b,i}w_{bi}^2.
$$

Define

$$
a=A-O,
\qquad
b=O-C.
$$

**Theorem 3.2 (association-scheme support extremality).** Assume

$$
a>0,\qquad b<0,\qquad a+bd>0.
$$

Among all real mass-one rules supported on at most $P$ lines in the fixed MUB universe, every minimum-energy rule:

1. uses all $P$ lines;
2. fills $q=\lfloor P/d\rfloor$ complete bases;
3. uses at most one additional partial basis of size $s=P-qd$;
4. assigns equal positive line weights within each active basis;
5. assigns positive basis masses proportional to
   $$
   \frac{1}{(O-C)+(A-O)/r_b}.
   $$

**Proof.** If basis $b$ contains $r_b$ active lines, Cauchy-Schwarz gives

$$
\sum_iw_{bi}^2\ge \frac{S_b^2}{r_b},
$$

with equality exactly for equal within-basis weights. Thus

$$
R(w)\ge \text{constant}+\sum_b c(r_b)S_b^2,
\qquad
c(r)=b+\frac ar.
$$

The assumptions imply $c(r)>0$ for $1\le r\le d$. Minimizing over $(S_b)$ with $\sum_bS_b=1$ gives

$$
S_b=\frac{c(r_b)^{-1}}{\sum_j c(r_j)^{-1}},
\qquad
R_{\min}=\text{constant}+\frac1{\sum_b h(r_b)},
$$

where

$$
h(r)=\frac1{c(r)}=\frac{r}{a+br},\qquad h(0)=0.
$$

On $[0,d]$,

$$
h'(r)=\frac{a}{(a+br)^2}>0,
\qquad
h''(r)=\frac{-2ab}{(a+br)^3}>0.
$$

Thus every available line is used and $h$ is strictly convex. For integers $0<x\le y<d$, convexity implies

$$
h(x-1)+h(y+1)>h(x)+h(y).
$$

Repeatedly transferring one line from a smaller partial basis to a larger one strictly increases $\sum_bh(r_b)$, and therefore strictly decreases risk, until only complete bases, at most one partial basis, and empty bases remain. The formulas above force positive basis masses and positive equal within-basis weights. $\square$

For the depth-32, $d=256$ Kerdock universe, the certified signs are

$$
A-O>0,
\qquad
O-C<0,
\qquad
(A-O)+256(O-C)>0.
$$

Therefore the theorem applies to all static linear rules with arbitrary real mass-one weights on the fixed 33,024 symmetrized lines. It does not apply to nodes outside that universe, unpaired points, finite-width adaptation, or nonlinear estimators.

## 3.4 Why positive definiteness alone is insufficient

The complete-basis conclusion does not follow from positive definiteness alone. In $d=4$, consider two real MUBs and

$$
k(t)=1+\lambda P_4(t),\qquad \lambda>0,
$$

where

$$
P_4(t)=\frac{16t^4-12t^2+1}{5}.
$$

This is positive definite, yet its association values satisfy $O-C>0$. Then $h(r)=r/(a+br)$ is concave rather than convex, and a balanced two-basis support beats one complete basis at a four-line budget. The sign pattern, not positive definiteness alone, is the structural mechanism.

## 3.5 Arbitrary signed nodes: a stability lemma, not closure

Let an auxiliary minorant satisfy

$$
h(t)=\sum_{\ell=0}^Lc_\ell G_\ell(t),
\qquad c_\ell\ge0\ (\ell\ge1),
\qquad h(t)\le K(t),
$$

and put $q=K-h$, $q_1=q(1)$, $M=\sup q$. For real weights summing to one, define total negative mass

$$
\beta=\sum_{w_i<0}|w_i|.
$$

**Proposition 3.3.** For at most $N$ nodes,

$$
E_K(w)
\ge
c_0+\frac{q_1}{N}-2M\beta(1+\beta).
$$

**Proof.** Positive definiteness of each Gegenbauer kernel gives

$$
\sum_{i,j}w_iw_jh(\langle x_i,x_j\rangle)\ge c_0.
$$

For the residual matrix, diagonal terms contribute $q_1\sum_iw_i^2$; same-sign off-diagonal terms are nonnegative; and the ordered absolute mass of opposite-sign products is $2\beta(1+\beta)$. Hence

$$
\sum_{i,j}w_iw_jq_{ij}
\ge
q_1\sum_iw_i^2-2M\beta(1+\beta).
$$

Finally, $\sum_iw_i^2\ge1/N$. $\square$

The exact residual supremum in the audited certificate is

$$
M=\frac{156999263604490023}{9223372036854775808}
=0.017021894267861247\ldots.
$$

Numerically the lemma is too weak to close arbitrary signed-node cubature: even a 10% relative improvement requires only a tiny certified lower bound on $\beta$. This is a stability result, not a universal signed-weight impossibility theorem.

# 4. Information and observability boundary

## 4.1 Runtime-information value

**Theorem 4.1 (correction observability principle).** Let $e\in L^2(\Omega;H)$ and let $\mathcal G\subseteq\mathcal F$ be all information available to a runtime correction. Among all $\mathcal G$-measurable corrections $c\in L^2(\mathcal G;H)$,

$$
\inf_c\mathbb E\|e-c\|^2
=
\mathbb E\|e\|^2-
\mathbb E\|\mathbb E[e\mid\mathcal G]\|^2.
$$

The unique optimum is

$$
c^*=\mathbb E[e\mid\mathcal G].
$$

**Proof.** Conditional expectation is the orthogonal projection of $e$ onto the closed subspace $L^2(\mathcal G;H)$. Put $m=\mathbb E[e\mid\mathcal G]$. For every admissible $c$,

$$
e-c=(e-m)+(m-c),
$$

and the two terms are orthogonal. Therefore

$$
\mathbb E\|e-c\|^2
=
\mathbb E\|e-m\|^2+
\mathbb E\|m-c\|^2.
$$

The second term is minimized uniquely by $c=m$, and the Pythagorean identity gives the value. $\square$

Define the correction value

$$
V(\mathcal G;e)
=
\mathbb E\|\mathbb E[e\mid\mathcal G]\|^2.
$$

If $\mathcal G_1\subseteq\mathcal G_2$, then

$$
V(\mathcal G_2;e)-V(\mathcal G_1;e)
=
\mathbb E\left\|
\mathbb E[e\mid\mathcal G_2]
-
\mathbb E[e\mid\mathcal G_1]
\right\|^2
\ge0.
$$

Thus replacing observations by a statistic cannot increase their correction value.

## 4.2 Finite-dictionary adaptive coefficient theorem

Let $A(\omega):\mathbb R^m\to H$ be a random linear feature operator. A coefficient vector $a$, measurable with respect to runtime information $\mathcal G$, produces correction $Aa$.

Define conditional normal-equation quantities

$$
\Sigma_{\mathcal G}
=
\mathbb E[A^*A\mid\mathcal G],
\qquad
\mu_{\mathcal G}
=
\mathbb E[A^*e\mid\mathcal G].
$$

**Theorem 4.2 (adaptive dictionary value).** The optimal coefficient is

$$
a^*_{\mathcal G}
=
\Sigma_{\mathcal G}^{\dagger}\mu_{\mathcal G},
$$

with minimum-norm convention, and the attainable gain is

$$
V_A(\mathcal G;e)
=
\mathbb E\left[
\mu_{\mathcal G}^{\mathsf T}
\Sigma_{\mathcal G}^{\dagger}
\mu_{\mathcal G}
\right].
$$

**Proof.** Condition on $\mathcal G$. The excess risk relative to zero correction is

$$
a^{\mathsf T}\Sigma_{\mathcal G}a
-2a^{\mathsf T}\mu_{\mathcal G}.
$$

If $v\in\ker\Sigma_{\mathcal G}$, then

$$
0=v^{\mathsf T}\Sigma_{\mathcal G}v
=
\mathbb E[\|Av\|^2\mid\mathcal G],
$$

so $Av=0$ almost surely conditionally and therefore $v^{\mathsf T}\mu_{\mathcal G}=0$. Hence $\mu_{\mathcal G}\in\operatorname{range}\Sigma_{\mathcal G}$, and completing the square with the Moore-Penrose inverse gives the stated optimizer and gain. $\square$

If the full per-network oracle observes $(A,e)$, then

$$
a_{\rm oracle}= (A^*A)^\dagger A^*e,
$$

and its gain is

$$
V_A({\rm oracle};e)
=
\mathbb E\|P_{\operatorname{range}(A)}e\|^2.
$$

This separates two questions:

- **Capacity:** does the dictionary span useful corrections?
- **Observability:** can legal runtime information recover their signed coefficients?

For nested runtime and oracle classes, define

$$
\operatorname{ObservabilityRatio}
=
\frac{V_A(\mathcal G_{\mathrm{runtime}};e)}
{V_A(\mathcal G_{\mathrm{oracle}};e)}
\in[0,1],
\qquad V_A(\mathcal G_{\mathrm{oracle}};e)>0.
$$

If the oracle value is zero, both values are zero and the ratio is left undefined (or assigned a separately declared convention).

A large oracle gain with a small observability ratio is exactly the “capacity without transferable phase” pattern seen in the signed-probe experiments.

## 4.3 Phase-flip impossibility theorem

**Theorem 4.3.** Suppose $\tau:\Omega\to\Omega$ is a measure-preserving involution such that

1. every runtime observable in $\mathcal G$ is invariant under $\tau$;
2. the dictionary cross-moment flips sign:
   $$
   A(\tau\omega)^*e(\tau\omega)
   =-A(\omega)^*e(\omega).
   $$

Then

$$
\mu_{\mathcal G}=0
$$

and every $\mathcal G$-measurable dictionary coefficient rule has nonnegative excess risk. The unique minimum-norm optimum is zero correction.

**Proof.** Invariance of $\mathcal G$, measure preservation, and anti-invariance of $A^*e$ imply

$$
\mathbb E[A^*e\mid\mathcal G]
=
-\mathbb E[A^*e\mid\mathcal G],
$$

so it is zero. Theorem 4.2 then gives zero attainable gain. $\square$

The same conclusion holds if the conditional law of $A^*e$ given $\mathcal G$ is centrally symmetric. This is the clean theorem one would need to prove for a specific weight-summary representation in order to convert empirical phase reversal into a population impossibility result.

## 4.4 Haar relative-orientation theorem

Let $G=O(d)$ act transitively on $S^{d-1}$, with normalized Haar measure $dg$. Let $Q=\sum_iw_i\delta_{x_i}$ have total mass one, and define its rotation by

$$
Q_gf=\sum_iw_i f(gx_i).
$$

**Theorem 4.4 (orientation-blind no-value theorem).** For every integrable Hilbert-valued $f$,

$$
\int_G Q_gf\,dg=I(f).
$$

Consequently, for a random integrand $f$, if the **conditional law** of $g$ given $\sigma(f)\vee\mathcal G$ is Haar (in particular, if $g$ is independent of both $f$ and $\mathcal G$), then

$$
\mathbb E[Q_gf-I(f)\mid\mathcal G]=0.
$$

No $\mathcal G$-measurable correction can lower mean squared error.

**Proof.** For every fixed node $x_i$, the pushforward of Haar measure under $g\mapsto gx_i$ is uniform spherical measure. Therefore

$$
\int_G f(gx_i)\,dg=I(f).
$$

Summing with $\sum_iw_i=1$ proves the first identity. Conditional mean zero follows by integrating against the conditional Haar law after fixing $(f,\mathcal G)$, and Theorem 4.1 gives zero correction value. Independence from $\mathcal G$ alone is insufficient when $f$ may depend on $g$. $\square$

This theorem rigorously closes corrections based only on rotation-invariant network summaries when relative design orientation is independently Haar randomized. It does **not** close orientation-sensitive white-box methods, network-derived nodes, or corrections using design evaluations.

## 4.5 Same-design common-bias non-identifiability

Suppose same-design estimates obey

$$
Z_i=\mu+b+\varepsilon_i,
\qquad i=1,\ldots,k,
$$

where the joint noise law does not depend on $(\mu,b)$.

**Theorem 4.5.** The transformation

$$
(\mu,b)\mapsto(\mu+t,b-t)
$$

leaves the complete observation law unchanged. For every estimator $T(Z)$ of $\mu$, at least one of the two gauge-equivalent parameter points has squared-error risk at least

$$
\frac{\|t\|^2}{4}.
$$

**Proof.** The observation law depends on $(\mu,b)$ only through $\mu+b$. Pointwise, for any estimate $x$,

$$
\|x-\mu\|^2+
\|x-(\mu+t)\|^2
=
2\left\|x-\mu-\frac t2\right\|^2+
\frac{\|t\|^2}{2}
\ge
\frac{\|t\|^2}{2}.
$$

Taking expectations under the common observation law shows that at least one risk is at least $\|t\|^2/4$. $\square$

Centered diagnostics

$$
D_i=Z_i-\overline Z
$$

depend only on centered noise. They can estimate dispersion or instability, but not the shared absolute defect in this model.

The theorem is model-specific. It fails with unequal known loadings, for example

$$
Z_1=\mu+b+\varepsilon_1,
\qquad
Z_2=\mu-b+\varepsilon_2,
$$

because $Z_1-Z_2$ directly contains $2b$.

## 4.6 General indistinguishability lower bound

**Theorem 4.6.** Suppose two admissible worlds $\theta_0,\theta_1$ induce exactly the same runtime transcript distribution but have targets $y_0,y_1\in H$. Then every estimator $T$ based on that transcript satisfies

$$
\max_{j\in\{0,1\}}
\mathbb E_j\|T-y_j\|^2
\ge
\frac{\|y_1-y_0\|^2}{4}.
$$

**Proof.** The transcript and hence $T$ have the same distribution under both worlds. Apply the parallelogram calculation from Theorem 4.5 with $t=y_1-y_0$. $\square$

This theorem converts a concrete indistinguishable-pair construction into a query or statistic lower bound. No such pair has yet been constructed for the full legal width-256 white-box network class.

## 4.7 Why a universal information impossibility theorem is false

**Proposition 4.7 (full-information obstruction).** If the complete network weights $W$ determine the target and baseline error, then

$$
\mathbb E[e\mid W]=e,
\qquad
V(\sigma(W);e)=\mathbb E\|e\|^2.
$$

Thus full weights contain, information-theoretically, enough information for a perfect correction.

**Proof.** The error is $\sigma(W)$-measurable, so its conditional expectation given $W$ is itself. Substitute in Theorem 4.1. $\square$

Therefore any meaningful lower bound must restrict at least one of:

- the statistics extracted from the weights;
- the number or geometry of network evaluations;
- the estimator family;
- the permitted arithmetic or time complexity;
- the information protocol.

The experiments alone cannot prove that no full-white-box algorithm exists.

# 5. Correction and downstream-replay mathematics

## 5.1 Exact risk identity

For a proposed direction $u\in L^2(\Omega;H)$ and scalar $\alpha$, define

$$
R(\alpha)=\mathbb E\|e-\alpha u\|^2.
$$

Let

$$
R_0=\mathbb E\|e\|^2,
\quad
C=\mathbb E\langle e,u\rangle,
\quad
U=\mathbb E\|u\|^2.
$$

**Theorem 5.1.**

$$
R(\alpha)=R_0-2\alpha C+\alpha^2U.
$$

For $U>0$,

$$
\alpha_*=\frac CU,
\qquad
R(\alpha_*)=R_0-\frac{C^2}{U}.
$$

If only $\alpha\ge0$ is legal, replace $C$ by $(C)_+$.

**Proof.** Expand the squared norm and minimize the resulting quadratic. $\square$

Magnitude, disagreement, and oracle span are insufficient. The binding quantity is the signed error-correction inner product.

## 5.2 Replacement in a correctable subspace

Let $\mathcal S\subset L^2(\Omega;H)$ be a closed linear subspace. Decompose

$$
s=P_{\mathcal S}e,
\qquad
r=e-s,
\qquad
r\perp\mathcal S.
$$

Suppose an anchor estimates $s$ as $\widehat s=s+n$.

**Theorem 5.2.** If $n\in\mathcal S$, then

$$
\mathbb E\|e-\widehat s\|^2
=
\mathbb E\|r\|^2+
\mathbb E\|n\|^2.
$$

Full replacement improves exactly when

$$
\mathbb E\|n\|^2<\mathbb E\|s\|^2.
$$

**Proof.** Since $e-\widehat s=r-n$ and $r\perp n$, Pythagoras gives the identity. Compare with $\mathbb E\|e\|^2=\mathbb E\|r\|^2+\mathbb E\|s\|^2$. $\square$

If $n$ is not in $\mathcal S$, the exact formula is

$$
\mathbb E\|e-\widehat s\|^2
=
\mathbb E\|r\|^2+
\mathbb E\|n\|^2
-2\mathbb E\langle r,n\rangle.
$$

Thus a norm-only replacement gate requires the subspace or orthogonality assumption.

## 5.3 Downstream-weighted anchor criterion

Condition on a realized network and protected particle cloud. Let

$$
d=\mu_{31}^{K}-\mu_{31}^*
$$

be the protected layer-31 mean defect and

$$
\xi=\widehat\mu_{31}-\mu_{31}^*
$$

be anchor error. Let $J$ be the frozen linearized map from layer-31 mean perturbations to scored final outputs. The relevant errors are

$$
s=Jd,
\qquad
n=J\xi.
$$

Under Theorem 5.2's subspace assumptions, full replacement improves exactly when

$$
\eta_J^2
:=
\frac{\mathbb E\|J\xi\|^2}
{\mathbb E\|Jd\|^2}
<1.
$$

No universal threshold in unweighted layer-31 Euclidean error exists. For a unit direction $v$, the linearized break-even amplitude is

$$
\epsilon_*(v)
=
\frac{\|Jd\|}{\|Jv\|},
$$

which varies with direction unless $J^*J$ is scalar on the tested span.

This formally justifies rescoring anchors by exact downstream replay rather than by a scalar coefficient-space threshold.

## 5.4 ReLU crossing remainder

For $\phi(z)=\max(z,0)$, define

$$
r(z,t)=\phi(z+t)-\phi(z)-\mathbf 1_{\{z>0\}}t.
$$

**Lemma 5.3.** For all $z,t\in\mathbb R$,

$$
|r(z,t)|
\le
|t|\mathbf 1_{\{|z|\le|t|\}}.
$$

**Proof.** If $z$ and $z+t$ have the same sign, ReLU is affine on the segment and the remainder is zero. A sign change implies $|z|\le|t|$, and direct inspection bounds the discrepancy by $|t|$. $\square$

If, conditional on $t$, the density of $z$ is bounded by $L$ throughout the whole interval $[-|t|,|t|]$, then

$$
\mathbb E[r(z,t)^2\mid t]
\le
|t|^2\Pr(|z|\le|t|\mid t)
\le
2L|t|^3.
$$

A merely local density bound on $[-\delta,\delta]$ gives this conclusion only on the event $|t|\le\delta$.

For a vector replay with linearized error $r_0-n$ and nonlinear remainder $q$, Minkowski gives

$$
\mathbb E\|r_0-n+q\|^2
\le
\left(
\sqrt{\mathbb E\|r_0-n\|^2}
+
\sqrt{\mathbb E\|q\|^2}
\right)^2.
$$

A sufficient exact-nonlinear improvement condition is that the right-hand side be below baseline risk.

# 6. Exact control mathematics

## 6.1 Uniform control-annihilation theorem

For a fixed quadrature rule $Q$, define its exactness space

$$
\mathcal N_Q=\{g:I(g)=Q(g)\}.
$$

**Theorem 6.1.** For every $f$, every $g\in\mathcal N_Q$, and every scalar $\alpha$,

$$
I(\alpha g)+Q(f-\alpha g)=Q(f).
$$

More generally, suppose $\{g_\eta:\eta\in\Xi\}\subseteq\mathcal N_Q$ uniformly. The parameter $\widehat\eta$ may be selected adaptively from the network, pilots, or quadrature outputs. If the same selected function is used in both $I(g_{\widehat\eta})$ and $Q(g_{\widehat\eta})$, the estimator remains exactly $Q(f)$ pathwise.

**Proof.** Linearity gives

$$
I(\alpha g)+Q(f-\alpha g)
=Q(f)+\alpha(I(g)-Q(g))
=Q(f).
$$

The adaptive statement is pointwise in the selected $\widehat\eta$. $\square$

Thus increasing model expressivity within a uniformly annihilated family cannot help. This closes named low-degree and exact ReLU-Stein families, not all high-degree or node-dependent controls.

## 6.2 Symmetrized spherical Poisson controls

For $u,x\in S^{d-1}$ and $|r|<1$, define

$$
P_r(\langle u,x\rangle)
=
\frac{1-r^2}
{(1-2r\langle u,x\rangle+r^2)^{d/2}}.
$$

**Theorem 6.2.** With normalized spherical measure $\sigma$,

$$
\int_{S^{d-1}}P_r(\langle u,x\rangle)\,d\sigma(x)=1.
$$

Therefore the antipodally symmetrized control

$$
P_r^{\rm even}(t)
=
\frac12\bigl(P_r(t)+P_r(-t)\bigr)
$$

also has exact mean one.

**Proof.** $P_r$ is the Poisson kernel for the unit ball. Its integral against boundary measure is the harmonic extension of the constant boundary function one, which is identically one. Equivalently, expand in zonal spherical harmonics: the degree-zero coefficient is one and every positive-degree harmonic integrates to zero. $\square$

The zonal expansion has the form

$$
P_r(t)=\sum_{\ell=0}^{\infty}r^\ell Z_\ell(t),
$$

where $Z_\ell$ is the degree-$\ell$ zonal reproducing kernel. Symmetrization cancels odd degrees:

$$
P_r^{\rm even}(t)
=
\sum_{k=0}^{\infty}r^{2k}Z_{2k}(t).
$$

For $r\ne0$, every even degree has a nonzero coefficient. Hence analytically integrable does not imply low harmonic degree.

## 6.3 Closed-form spherical mean of a biased projected ReLU

Let $X$ be uniform on the radius-$R$ sphere in $\mathbb R^d$. For $a\in\mathbb R^d$, put

$$
\rho=R\|a\|,
\qquad
T=\frac{\langle a,X\rangle}{\rho}
$$

when $\rho>0$. Then $T\in[-1,1]$ has density

$$
f_d(t)=c_d(1-t^2)^{(d-3)/2},
\qquad
c_d=
\frac{\Gamma(d/2)}
{\sqrt\pi\,\Gamma((d-1)/2)}.
$$

Consider

$$
g_{a,b}(X)=(\langle a,X\rangle+b)_+.
$$

**Theorem 6.3.** If $\rho=0$, then $\mathbb E g_{a,b}=b_+$. If $b\ge\rho$, then $\mathbb E g_{a,b}=b$; if $b\le-\rho$, it is zero. For $|b|<\rho$, put

$$
s=-\frac b\rho,
\qquad
\beta=\frac{d-1}{2}.
$$

Then

$$
\boxed{
\mathbb E g_{a,b}
=
\frac{\rho c_d}{d-1}(1-s^2)^\beta
+
 b\left[
\frac12-
\frac{\operatorname{sgn}(s)}2
I_{s^2}\left(\frac12,\beta\right)
\right]
}
$$

where $I_x(p,q)$ is the regularized incomplete beta function, with $\operatorname{sgn}(0)=0$.

**Proof.** The active region is $T>s$. Therefore

$$
\mathbb E g_{a,b}
=
\rho c_d\int_s^1t(1-t^2)^{(d-3)/2}\,dt
+b\Pr(T>s).
$$

The first integral is

$$
\int_s^1t(1-t^2)^{(d-3)/2}\,dt
=
\frac{(1-s^2)^{(d-1)/2}}{d-1}.
$$

Symmetry of $T$ and the substitution $u=t^2$ give

$$
\Pr(T>s)
=
\frac12-
\frac{\operatorname{sgn}(s)}2
I_{s^2}\left(\frac12,\frac{d-1}{2}\right).
$$

Substitution yields the formula. $\square$

For the experiment's normalized direction rows $\|a\|=1$, use $\rho=R$. The verifier checks the formula against adaptive numerical integration to approximately `1e-14` absolute error. The experiment used 384-point Gauss-Jacobi quadrature; the closed form removes that numerical dependency for future implementations.

# 7. Replication and compute economics

Suppose $m$ estimators have errors $e_1,\ldots,e_m\in H$, each with risk $R_0$, and pairwise normalized covariance

$$
\mathbb E\langle e_i,e_j\rangle=\rho R_0
\qquad(i\ne j).
$$

Their average has risk

$$
\mathbb E\left\|\frac1m\sum_{i=1}^me_i\right\|^2
=
R_0\left(\rho+\frac{1-\rho}{m}\right).
$$

**Proof.** Expand the squared norm. There are $m$ diagonal terms and $m(m-1)$ ordered off-diagonal terms. Divide by $m^2$. $\square$

If compute grows by a factor $c_m$, the adjusted ratio is

$$
c_m\left(\rho+\frac{1-\rho}{m}\right).
$$

For independent **mean-zero** full-cost replica errors (or, more generally, pairwise uncorrelated errors), $\rho=0$ and $c_m=m$, so the adjusted ratio is exactly one. Independence alone does not imply $\rho=0$ for biased estimators. Ordinary unbiased independent replication cannot improve an MSE-times-linear-compute score. It can win only through sublinear shared arithmetic, negative covariance, unequal quality/cost allocation, or a different scoring rule.

# 8. Finite-width and transformed-residual boundaries

## 8.1 Kernel perturbation theorem

Let $P$ be the target distribution and $Q=\sum_iw_i\delta_{x_i}$. Put $\nu_Q=P-Q$. For kernels $K$ and $\widetilde K$, define

$$
R_K(Q)=\iint K(x,y)\,d\nu_Q(x)d\nu_Q(y).
$$

**Theorem 8.1.** If

$$
\|K-\widetilde K\|_\infty\le\varepsilon,
\qquad
\sum_i|w_i|\le B,
$$

then

$$
|R_K(Q)-R_{\widetilde K}(Q)|
\le
\varepsilon(1+B)^2.
$$

**Proof.** $\|\nu_Q\|_{\rm TV}\le1+B$, so

$$
\left|\iint(K-\widetilde K)\,d\nu_Qd\nu_Q\right|
\le
\|K-\widetilde K\|_\infty\|\nu_Q\|_{\rm TV}^2.
$$

$\square$

If $Q_K$ has additive suboptimality at most $g$ for $K$, then

$$
R_{\widetilde K}(Q_K)-\inf_QR_{\widetilde K}(Q)
\le
 g+2\varepsilon(1+B)^2.
$$

For nonnegative mass-one rules, $B=1$, so the perturbation allowance is $g+8\varepsilon$.

The certified limiting-kernel gap is extremely small. A qualitative statement that width-256 kernels “approach” the limiting kernel is insufficient; a quantitative error bound at the relevant scale is required.

## 8.2 Optimizer instability under tiny PSD perturbations

Let two rules have linearly independent error functionals $L_0,L_1$. Choose bounded $\phi,\psi$ satisfying

$$
L_0\phi=0,
\quad L_1\phi=1,
\qquad
L_0\psi=1,
\quad L_1\psi=0.
$$

Set

$$
K_\infty=g\,\phi\otimes\phi,
\qquad
K_m=K_\infty+\varepsilon\,\psi\otimes\psi.
$$

Both kernels are positive semidefinite. Under $K_\infty$, rule zero has risk zero and rule one has risk $g$. Under $K_m$, their risks are $\varepsilon$ and $g$. For $\varepsilon>g$, the ranking of these two rules reverses; if the admissible comparison class consists of these rules (or all other rules are controlled), the optimizer reverses. Since $g$ can be arbitrarily small, pairwise ranking—and hence optimizer identity in a suitable class—is not structurally stable without a quantitative gap-versus-perturbation comparison.

## 8.3 Residual-kernel recertification

Let $g_\theta$ be a network-dependent analytic control with exactly available integral. Define

$$
h_\theta=f_\theta-g_\theta
$$

and estimator

$$
\widehat I_Q(f_\theta)=I(g_\theta)+Q(h_\theta).
$$

Then

$$
\widehat I_Q(f_\theta)-I(f_\theta)
=(Q-I)h_\theta.
$$

If

$$
K_{\rm res}(x,y)
=
\mathbb E[h_\theta(x)h_\theta(y)],
$$

then

$$
\mathbb E|
\widehat I_Q(f_\theta)-I(f_\theta)
|^2
=
\iint K_{\rm res}(x,y)\,d\nu_Q(x)d\nu_Q(y).
$$

Thus analytic-plus-residual estimation returns to a kernel-energy problem, but for $K_{\rm res}$, not the original deep-ReLU kernel. Kerdock must be recertified after the transformation. Original optimality does not automatically persist.

# 9. What the frozen experiments prove

The deterministic verifier bundled with this document checks all 154 entries in the reopened-path SHA-256 manifest and recomputes the principal terminal metrics from the saved row-level arrays.

The finite statements are:

1. The frozen Poisson terminal candidate has raw ratio
   $$
   1.0379391539423775>1.
   $$
2. The frozen projected-ReLU candidate over 48 terminal networks has raw ratio
   $$
   1.012682140618133>1,
   $$
   and favorable adjusted ratio
   $$
   1.0668717656262163>1.
   $$
3. The frozen network-derived signed-probe rule has raw ratio
   $$
   1.5573136434384325>1.
   $$
4. The same signed dictionary has a per-network ridge oracle ratio
   $$
   0.13117636626944396,
   $$
   below the `4.34x` target ratio, proving that its span has enough same-network capacity while its tested transferable coefficient rule fails.
5. The exact Poisson mean-one identity and the closed-form projected-ReLU mean agree with independent numerical integration to near machine precision.

These are propositions about the frozen files, candidate definitions, and IEEE-754 computations. They do not establish that the synthetic network generator equals the protected competition distribution or that every member of a broader estimator class fails.

## 9.1 Empirical observability diagnostics

A post-hoc decomposition, explicitly not a generalization claim, found:

| Family | Post-hoc global ratio | Per-network oracle ratio | Empirical global/oracle gain ratio |
|---|---:|---:|---:|
| Network-derived signed probes | `0.93821` | `0.13118` | `0.07112` |
| Random signed probes | `0.95802` | `0.33681` | `0.06330` |
| Fixed projected-ReLU direction with scalar coefficient | `0.99897` | `0.84762` | `0.00676` |
| Fixed Poisson direction with scalar coefficient | `0.99210` | `0.78358` | `0.03652` |

For the network-derived signed probes, leave-one-network-out fitting was harmful (`1.09909x`). These diagnostics strongly motivate a phase-observability theorem, but they do not themselves prove one. The required population proof obligation is Theorem 4.3: identify an exact conditional symmetry or bound the conditional cross-moment.

# 10. Claims that remain open

The following are not proved by this package:

1. Exact width-256 optimality of Kerdock.
2. A universal lower bound against all algorithms using the complete network weights.
3. Optimality over arbitrary signed spherical nodes.
4. Impossibility of network-adaptive support or weights.
5. Impossibility of nonlinear postprocessing.
6. Impossibility of all biased, deep, nonhomogeneous, or nonpolynomial analytic controls.
7. Population-level zero observability for the exact signed-probe representation.
8. Equality between the synthetic seed distribution and the protected competition cohort.
9. A computational lower bound for evaluating the target from full weights.

Any manuscript sentence equivalent to “no statistical path exists” is therefore false or unsupported.

# 11. Strongest defensible integrated theorem

The results can be stated compactly as follows.

**Integrated two-boundary theorem.** For the dimension-256, depth-32 limiting ReLU kernel:

1. complete Kerdock is computer-assisted certified within `0.023324172950039%` of the static nonnegative optimum at the stated node budget;
2. inside the fixed 33,024-line Kerdock universe, complete bases plus at most one partial basis exactly optimize all static linear mass-one rules, even with arbitrary real weights;
3. for any explicitly specified runtime information class, attainable adaptive correction gain is exactly the squared Hilbert projection of the baseline error onto that information class;
4. equal-loading common-bias information, conditionally Haar-random orientation-blind information, and feature classes satisfying a phase-flip symmetry have zero absolute-phase correction value under their explicit models;
5. full network weights do not satisfy a generic information-impossibility premise, so broader white-box and finite-width methods remain mathematically open.

Therefore a material winner must introduce either:

- support/weights outside the certified geometric class;
- information that breaks the absolute-phase symmetry;
- a transformed residual whose kernel is materially easier;
- finite-width-specific analytic structure;
- or a computational mechanism not represented by the tested correction families.

# 12. Reproducibility and trust base

Bundled files:

- `WHESTBENCH_COMPLETE_PROOF_PACKAGE_20260730.md` - this proof document;
- `WHESTBENCH_COMPLETE_PROOF_PACKAGE_20260730.tex` - LaTeX source;
- `WHESTBENCH_COMPLETE_PROOF_PACKAGE_20260730.pdf` - rendered proof document;
- `verify_complete_proof_package.py` - deterministic frozen-artifact verifier;
- `WHESTBENCH_PROOF_CERTIFICATE_20260730.json` - verifier output;
- `OBSERVABILITY_EMPIRICAL_AUDIT_20260730.json` - explicitly post-hoc empirical diagnostic;
- `reopened_paths_repro_20260730/` - scripts, frozen arrays, split registry, preregistrations, hashes, and results;
- `sources/` - materialized theorem-source summaries and the T16 primal-dual certificate.

The proof dependencies T22/T30 remain computer-assisted, not proof-assistant formalized. Before public release, a qualified human should review every analytic argument, rerun the original interval engines and CI matrix, publish an externally authenticated archive digest, and disclose the extent of language-model assistance.
