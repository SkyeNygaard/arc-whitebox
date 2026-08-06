# WHestBench proof work memo v1

**Date:** 2026-07-29  
**Purpose:** Replace the informal “layer-31 circularity” claim with precise theorems, clarify the status of the Kerdock proof, and derive a rigorous signed-weight extension under a stability constraint.

## Executive conclusions

1. **The scoped Kerdock near-optimality theorem is already complete.** The V5/T22 proof does not depend on T16's all-degree reduced-cost tail. T16 would strengthen the claim that the chosen degree-5 auxiliary polynomial is optimal for the unrestricted auxiliary LP, but it is not a missing logical step in T22.
2. **The useful layer-31 theorem is not “relative error is depth-independent.”** That remains an empirical hypothesis. The rigorous core is an exact correction-risk identity plus an anchor-replacement threshold: in the linearized correction subspace, a full replacement improves MSE exactly when the downstream error of the replacement anchor is smaller than the downstream correctable defect.
3. **Same-design folds cannot identify absolute phase without an external reference.** A common-bias model gives an exact non-identifiability theorem. Fold differences, jackknife statistics, and nested convergence can estimate dispersion while containing zero information about the common absolute defect.
4. **The final ReLU nonlinearity can be bounded exactly by gate-crossing mass.** Away from preactivations within the proposed shift magnitude of zero, translation is exactly linear.
5. **The signed-weight loophole can be quantified.** The positive-weight Delsarte lower bound extends to signed weights with a penalty controlled by total negative mass. This does not close arbitrary signed rules, but it turns the binary loophole into a stability theorem.

---

## 1. Setup

Let \(H\) be a finite-dimensional real Hilbert space representing all scored output coordinates, possibly aggregated over a distribution of networks. Let

\[
e = \widehat y_0-y \in L^2(\Omega;H)
\]

be the baseline estimation error. Let \(u\in L^2(\Omega;H)\) be a proposed signed correction, and define

\[
\widehat y_\alpha=\widehat y_0-\alpha u,
\qquad
R(\alpha)=\mathbb E\|e-\alpha u\|^2.
\]

The sign convention is that \(u=e\) would be a perfect correction.

## 2. Exact correction-risk theorem

### Theorem 1 — correction-risk identity

Define

\[
R_0=\mathbb E\|e\|^2,
\qquad
C=\mathbb E\langle e,u\rangle,
\qquad
U=\mathbb E\|u\|^2.
\]

Then, for every real \(\alpha\),

\[
R(\alpha)=R_0-2\alpha C+\alpha^2U.
\]

If \(U>0\), the unconstrained optimum is

\[
\alpha_*=\frac{C}{U},
\qquad
R(\alpha_*)=R_0-\frac{C^2}{U}.
\]

A positive scale improves on the baseline if and only if \(C>0\), and a fixed \(\alpha>0\) improves if and only if

\[
2C>\alpha U.
\]

#### Proof

Expand the square:

\[
\|e-\alpha u\|^2=\|e\|^2-2\alpha\langle e,u\rangle+\alpha^2\|u\|^2.
\]

Take expectations. The remaining claims follow by minimizing the resulting quadratic. \(\square\)

### Corollary 1.1 — correlation ceiling

Let

\[
\rho=\frac{C}{\sqrt{R_0U}}.
\]

Then the best scalar rescaling of the proposed direction satisfies

\[
\frac{R(\alpha_*)}{R_0}=1-\rho^2.
\]

Thus a direction with near-zero signed cosine cannot become useful through scale tuning alone.

### Corollary 1.2 — exact value of a selector or gate

Let \(\mathcal G\) be the information available to a target-free selector, and permit \(\alpha\) to be any \(\mathcal G\)-measurable scalar. Define

\[
C_{\mathcal G}=\mathbb E[\langle e,u\rangle\mid\mathcal G],
\qquad
U_{\mathcal G}=\mathbb E[\|u\|^2\mid\mathcal G].
\]

Where \(U_{\mathcal G}>0\), the conditionally optimal scale is

\[
\alpha_*(\mathcal G)=\frac{C_{\mathcal G}}{U_{\mathcal G}},
\]

and its expected gain is

\[
\mathbb E\!\left[\frac{C_{\mathcal G}^2}{U_{\mathcal G}}\right].
\]

Therefore a gate helps only to the extent that its observables predict the **signed error-correction inner product**, not merely correction magnitude, fold disagreement, convergence, or oracle headroom.

---

## 3. Anchor replacement theorem

The informal statement “the anchor must be better than the estimator it replaces” can be made exact.

Let \(\mathcal S\subset L^2(\Omega;H)\) be a closed linear correction subspace. Let

\[
s=P_{\mathcal S}e,
\qquad
r=e-s,
\]

where \(P_{\mathcal S}\) is the orthogonal projection. Hence \(r\perp\mathcal S\). Suppose the deployable anchor produces

\[
\widehat s=s+n,
\qquad n\in\mathcal S.
\]

### Theorem 2 — full-replacement threshold

Applying the full correction \(\widehat s\) gives

\[
\mathbb E\|e-\widehat s\|^2
=
\mathbb E\|r\|^2+
\mathbb E\|n\|^2.
\]

The baseline risk is

\[
\mathbb E\|e\|^2
=
\mathbb E\|r\|^2+
\mathbb E\|s\|^2.
\]

Therefore full replacement improves MSE if and only if

\[
\boxed{\mathbb E\|n\|^2<\mathbb E\|s\|^2.}
\]

#### Proof

Because \(r\perp\mathcal S\), it is orthogonal to both \(s\) and \(n\). Also

\[
e-\widehat s=(s+r)-(s+n)=r-n.
\]

Pythagoras gives both displayed decompositions. \(\square\)

### Layer-31 interpretation

Under a frozen local linearization of the final replay, let \(J\) map a layer-31 mean defect to final-output error. Let

\[
d=\mu_{31}^{K}-\mu_{31}^{*}
\]

be the baseline Kerdock mean defect and

\[
\xi=\widehat\mu_{31}-\mu_{31}^{*}
\]

be the replacement anchor's error. If the correctable output component is \(s=Jd\), the replacement noise is \(n=J\xi\), and the uncorrectable remainder is orthogonal to the range of \(J\), then full mean replacement improves exactly when

\[
\boxed{\mathbb E\|J\xi\|^2<\mathbb E\|Jd\|^2.}
\]

This is the defensible “circularity” result. An anchor with the same downstream-weighted accuracy as the baseline defect is at break-even; an anchor formed by another recurrence with comparable relative error cannot create contraction merely by being called an anchor.

### Corollary 2.1 — optimal shrinkage for independent anchor noise

If additionally \(\mathbb E\langle s,n\rangle=0\), then the optimal scalar applied to \(\widehat s=s+n\) is

\[
\alpha_*=
\frac{\mathbb E\|s\|^2}
{\mathbb E\|s\|^2+\mathbb E\|n\|^2}.
\]

The maximum removable risk is

\[
\frac{\bigl(\mathbb E\|s\|^2\bigr)^2}
{\mathbb E\|s\|^2+\mathbb E\|n\|^2}.
\]

This explains why conservative shrinkage may remain mildly useful when full replacement fails, while also showing that no scalar tuning overcomes a low signed correlation.

---

## 4. Absolute-phase non-identifiability

The repeated failure of cross-fitting, paired blocks, jackknifing, and nested convergence can be expressed as an identifiability result.

Suppose same-design sub-estimates satisfy

\[
Z_i=\mu+b+\varepsilon_i,
\qquad i=1,\dots,k,
\]

where \(\mu\) is the desired absolute expectation, \(b\) is a common design defect, and the joint law of \((\varepsilon_1,\dots,\varepsilon_k)\) does not depend on \((\mu,b)\).

### Theorem 3 — common-bias non-identifiability

Without an external restriction or reference on \(\mu\) or \(b\), the defect \(b\) is not identifiable from \((Z_1,\dots,Z_k)\).

More strongly, for every vector \(t\), the parameter pairs

\[
(\mu,b)
\quad\text{and}\quad
(\mu+t,b-t)
\]

induce exactly the same observation law. For any estimator \(T(Z_1,\dots,Z_k)\), at least one of these two parameter points has squared-error risk at least

\[
\frac{\|t\|^2}{4}.
\]

#### Proof

Both parameter pairs have the same sum \(\mu+b\), so they produce the same law of every \(Z_i\). For any realized estimate \(x\),

\[
\|x-b\|^2+\|x-(b-t)\|^2
\ge \frac{\|t\|^2}{2}.
\]

Taking expectation under the common observation law shows that the larger of the two risks is at least \(\|t\|^2/4\). \(\square\)

### Corollary 3.1 — centered diagnostics contain no absolute phase

Every statistic measurable from

\[
Z_i-\overline Z
\]

has a law independent of both \(\mu\) and \(b\). Such statistics can estimate variability, exchangeability failure, or instability, but cannot determine the sign or magnitude of the shared absolute defect.

This exactly describes the limitation of same-design cross-fitting and nested convergence under the common-bias model: smooth convergence can occur around the wrong external phase.

---

## 5. ReLU translation remainder

The layer-31 argument uses a local linear final replay. The nonlinearity can be isolated exactly.

Let \(\phi(z)=\max(z,0)\).

### Lemma 4 — scalar ReLU crossing bound

For all real \(z,t\),

\[
\phi(z+t)=\phi(z)+\mathbf 1_{\{z>0\}}t+r(z,t),
\]

where

\[
|r(z,t)|
\le
|t|\,\mathbf 1_{\{|z|\le |t|\}}.
\]

#### Proof

If \(z\) and \(z+t\) have the same strict sign, ReLU is affine on the segment and \(r=0\). A nonzero remainder requires a zero crossing, which implies \(|z|\le|t|\); direct evaluation in the two crossing cases gives \(|r|\le|t|\). The case \(z=0\) also satisfies the bound. \(\square\)

### Corollary 4.1 — particle-mean replay bound

For final-layer preactivations \(h_i\), a layer-31 translation \(\delta\), and \(t=W\delta\), define

\[
m(\delta)=\frac1n\sum_{i=1}^n\phi(h_i+t).
\]

Then

\[
m(\delta)=m(0)+J\delta+R(\delta),
\]

with

\[
J=\frac1n\sum_{i=1}^n \operatorname{diag}(\mathbf1_{\{h_i>0\}})W
\]

and the coordinatewise remainder bound

\[
|R(\delta)|
\le
\frac1n\sum_{i=1}^n
|W\delta|\odot
\mathbf1_{\{|h_i|\le|W\delta|\}}.
\]

Thus translation is exactly linear except on particles whose final preactivation lies within the proposed shift of a ReLU kink. A rigorous layer-31 result can combine Theorem 2 with an empirical or certified bound on this kink-mass remainder.

---

## 6. Signed-weight stability extension of the Delsarte bound

The completed T22 proof uses an auxiliary minorant

\[
h(t)=\sum_{\ell=0}^L c_\ell G_\ell(t),
\qquad c_\ell\ge0\ (\ell\ge1),
\qquad h(t)\le K(t).
\]

Write

\[
q(t)=K(t)-h(t),
\qquad 0\le q(t)\le M
\]

on \([-1,1]\), and let \(q_1=q(1)\). Consider real weights satisfying

\[
\sum_iw_i=1,
\qquad m\le N.
\]

Define their total negative mass

\[
\beta=\sum_{i:w_i<0}|w_i|.
\]

Then the positive mass is \(1+\beta\).

### Proposition 5 — signed-weight lower bound with negative-mass penalty

For any nodes \(x_i\) and such weights,

\[
\boxed{
\sum_{i,j}w_iw_jK(\langle x_i,x_j\rangle)
\ge
c_0+\frac{q_1}{N}-2M\beta(1+\beta).
}
\]

#### Proof

The addition theorem implies

\[
\sum_{i,j}w_iw_jG_\ell(\langle x_i,x_j\rangle)\ge0
\]

for arbitrary real weights. Therefore the coefficient signs give

\[
\sum_{i,j}w_iw_jh(\langle x_i,x_j\rangle)\ge c_0.
\]

For the residual matrix \(q_{ij}=q(\langle x_i,x_j\rangle)\):

- diagonal terms contribute \(q_1\sum_iw_i^2\);
- off-diagonal same-sign products are nonnegative and may be discarded;
- ordered opposite-sign products have total absolute mass
  \(2\beta(1+\beta)\), and each has \(q_{ij}\le M\).

Hence

\[
\sum_{i,j}w_iw_jq_{ij}
\ge
q_1\sum_iw_i^2-2M\beta(1+\beta).
\]

Cauchy--Schwarz gives

\[
\sum_iw_i^2\ge\frac{(\sum_iw_i)^2}{m}\ge\frac1N.
\]

Adding the two bounds proves the result. \(\square\)

### Consequence

At \(\beta=0\), Proposition 5 recovers the positive-weight Delsarte bound. For a signed rule to beat the positive lower bound by more than \(\Delta>0\), it must satisfy

\[
2M\beta(1+\beta)\ge\Delta.
\]

Equivalently,

\[
\beta\ge
\frac{-1+\sqrt{1+2\Delta/M}}{2}.
\]

This does not close unrestricted signed weights, but it proves that any improvement exceeding the certified positive-class gap requires a quantitatively nontrivial amount of negative mass unless \(M\) is large. The next computational task is to interval-certify \(M=\sup(K-h)\) and report the resulting exclusion curve.

---

## 7. Status of T16's analytic tail

The completed T22 theorem needs only one feasible certified minorant; it does **not** require proof that the degree-5 minorant is the best possible auxiliary certificate. The proof audit states that no logical gap is known inside T22's explicit scope.

T16 asks a different question: whether adding any harmonic degree \(\ell\ge6\) can improve the auxiliary LP. Its reduced cost is recorded as

\[
r_\ell=q_\ell-\sum_j\lambda_jG_\ell(t_j).
\]

The finite audit found \(r_\ell\le0\) for \(6\le\ell\le10^6\), with the closest case \(r_7=-2.5050045\times10^{-7}\), and values approaching \(-1/N\).

### What is still needed for an all-degree proof

1. Recover the exact or interval-enclosed \(q_\ell\), \(\lambda_j\), and contact points \(t_j\) used by the dual audit.
2. Separate any endpoint contribution using \(G_\ell(1)=1\) and \(G_\ell(-1)=(-1)^\ell\).
3. Prove a uniform fixed-interior bound for normalized Gegenbauer polynomials at the remaining contact points. In dimension 256, the parameter is \(\alpha=127\), so the expected fixed-angle decay is extremely fast after normalization.
4. Choose a finite cutoff \(L\) where the analytic bound forces

   \[
   r_\ell\le-\frac{1}{2N}<0
   \qquad(\ell\ge L),
   \]

   and interval-certify the finitely many modes \(6\le\ell<L\).

The existing report gives the reduced-cost formula but not the exact dual witness values required to complete this derivation. Therefore T16 is a well-scoped next lemma, not something that should be claimed complete from the current archive.

---

## 8. Claims suitable for the algorithmic-contribution article

### Fully proved or certified

- Scoped T22 Kerdock near-optimality for fixed, network-independent, nonnegative-weight linear cubature with support at most 66,048 under the dimension-256, depth-32 infinite-width ReLU kernel.
- Exact correction-risk identity (Theorem 1).
- Full-replacement anchor threshold under the stated linear-subspace assumptions (Theorem 2).
- Common-bias non-identifiability (Theorem 3).
- ReLU gate-crossing remainder bound (Lemma 4).
- Signed-weight negative-mass stability bound (Proposition 5).

### Empirical and should remain labeled empirical

- The layer-31 oracle removes roughly 78% of noise-corrected final MSE.
- The measured useful/break-even mean-accuracy thresholds near 0.3%/0.45% and current propagation near 0.65%.
- Cross-layer relative error near 1.012 or any claim of depth-independent relative error.
- The MSE scaling exponent near \(-1.150\).
- Failure rates and tail behavior of specific selectors, companion designs, and learned models.

### Not proved

- Optimality for arbitrary signed weights.
- Exact finite-width optimality at width 256.
- A universal impossibility theorem for every network-adaptive layer-31 estimator.
- T16 all-degree auxiliary-LP optimality.

---

## 9. Immediate proof-work order

1. Put Theorems 1--3 and Lemma 4 into the article; these are already complete under explicit assumptions.
2. Compute an interval upper bound on \(M=\sup(K-h)\) and turn Proposition 5 into a numerical signed-weight stability curve.
3. Locate/export the exact T16 dual contact measure and prove the normalized-Gegenbauer tail.
4. Only after those are stable, decide whether a stronger depth-homogeneous theorem is true. Do not promote the observed 1.012 ratio into a theorem without a model deriving it.

## Evidence files consulted

- `PROOF_AUDIT.md`
- `main.pdf`
- `EXPERIMENT_REPORT_V3.md`
- `SIMPLE_EXPLAINER.md`
- layer-31 tolerance and micro-cubature reports summarized in the canonical ledger
- `T0_Unblocked_20260729/REPORT.md`

