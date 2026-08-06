# T61 — Shared-profile relaxation sharpness and atomic strictness

**Date:** 2026-07-30  
**Status:** Exact theorem. The numerical specialization uses the independently rerun v21 certificate and the standard-library exact verifier in this package.

## 1. Why this theorem is needed

The v21 static signed certificate is a positive combination of 146 adjacent-harmonic comparisons

\[
L_{s,r}(t)=d_sG_s(t)+r\,d_{s+1}G_{s+1}(t),
\qquad
H_{s,r}(t)=L_{s,r}(t)^2,
\]

normalized by their rank-and-block-trace floors. It proves, for the dimension-256 depth-32 limiting ReLU kernel and every static, network-independent, mass-one signed rule with at most
\(N=66{,}048\) nodes,

\[
R_K(Q)\ge F_{21}
=2.28045159853140213494322646565331\times10^{-7}
=0.9370459569114724\,R_K(Q_{\rm Kerdock}).
\]

A natural next attempt is to couple all comparison profiles through one shared harmonic moment matrix. The first half of this theorem shows that **this still cannot improve the bound** if the relaxation keeps only rank and harmonic block traces. The second half shows that the resulting abstract equality is nevertheless impossible for a real atomic spherical rule.

---

## 2. General shared-block setup

Let

\[
\mathcal V=\bigoplus_{\ell\in\Lambda}H_\ell,
\qquad d_\ell=\dim H_\ell,
\]

where every active block satisfies \(d_\ell\ge N\). Let \(Y_\ell(x)\in H_\ell\) be a normalized spherical-harmonic evaluation vector, so

\[
\|Y_\ell(x)\|^2=d_\ell.
\]

For a mass-one signed atomic rule

\[
Q=\sum_{i=1}^m w_i\delta_{x_i},\qquad m\le N,
\]

define the one shared moment matrix

\[
M(Q)=Q\!\left[YY^\top\right],
\qquad
Y(x)=\bigoplus_{\ell\in\Lambda}Y_\ell(x).
\]

Then

\[
\operatorname{rank}M(Q)\le N,
\qquad
\operatorname{tr}M_{\ell\ell}(Q)=d_\ell.
\]

A comparison profile \(a=(a_\ell)_{\ell\in\Lambda}\), \(a_\ell\ge0\), rescales the shared matrix by

\[
D_a=\bigoplus_\ell \sqrt{a_\ell}I_{d_\ell},
\qquad M_a=D_aM D_a,
\]

with target

\[
A_a=\bigoplus_\ell a_\ell I_{d_\ell}.
\]

Put

\[
T_a=\sum_\ell a_\ell d_\ell,
\qquad
S_{2,a}=\sum_\ell a_\ell^2d_\ell.
\]

The usual rank-and-trace argument gives

\[
\|A_a-M_a\|_F^2\ge {T_a^2\over N}-S_{2,a}.
\]

---

## 3. Theorem A — one shared abstract matrix attains every profile floor

### Statement

Let \(\mathcal A\) be any finite family of nonnegative profiles on \(\Lambda\). If \(d_\ell\ge N\) for every active block, then there exists **one** symmetric positive-semidefinite rank-\(N\) matrix \(M_*\), with every required block trace, such that for every \(a\in\mathcal A\),

\[
\boxed{
\|A_a-D_aM_*D_a\|_F^2
={T_a^2\over N}-S_{2,a}.
}
\]

Thus no relaxation based only on

- one shared rank-\(N\) harmonic moment matrix,
- all harmonic block traces,
- all shared cross-block variables,
- and the separate Frobenius profile objectives,

can improve the v21 certificate.

### Construction and proof

For each block choose an isometry

\[
E_\ell:\mathbb R^N\to H_\ell,
\qquad E_\ell^\top E_\ell=I_N.
\]

Define a global factor \(Z:\mathbb R^N\to\mathcal V\) blockwise by

\[
Z_\ell=\sqrt{d_\ell/N}\,E_\ell,
\qquad
M_*=ZZ^\top.
\]

Then \(M_*\succeq0\), \(\operatorname{rank}M_*=N\), and

\[
\operatorname{tr}(M_{*,\ell\ell})
={d_\ell\over N}\operatorname{tr}(E_\ell E_\ell^\top)=d_\ell.
\]

For a profile \(a\), the \(N\) columns of \(D_aZ\) are mutually orthogonal. Every column has squared norm

\[
\sum_\ell {a_\ell d_\ell\over N}={T_a\over N}.
\]

Therefore \(D_aM_*D_a\) has exactly \(N\) nonzero eigenvalues, all equal to \(T_a/N\). Also

\[
\operatorname{tr}(A_aD_aM_*D_a)=S_{2,a}.
\]

Consequently

\[
\begin{aligned}
\|A_a-D_aM_*D_a\|_F^2
&=\|A_a\|_F^2+
\|D_aM_*D_a\|_F^2
-2\operatorname{tr}(A_aD_aM_*D_a)\\
&=S_{2,a}+{T_a^2\over N}-2S_{2,a}\\
&={T_a^2\over N}-S_{2,a}.
\end{aligned}
\]

This is simultaneous for every profile. ∎

### v21 specialization

The 146 released profiles use 123 harmonic blocks. The smallest active block is \(H_3\), with

\[
d_3=2{,}828{,}800>N.
\]

The verifier checks the construction profile by profile. Hence a larger SDP that merely shares all harmonic block moment variables remains exactly as weak as the sum of the independent rank floors.

---

## 4. Theorem B — a real atomic rule cannot attain the aggregate floor

### Single-profile equality lemma

For one profile \(L_a\), equality in the rank floor forces:

1. exactly \(N\) nonzero nodes;
2. equal positive weights \(w_i=1/N\);
3. pairwise comparison-kernel orthogonality
   \[
   L_a(x_i\!\cdot x_j)=0\quad(i\ne j).
   \]

This is the existing atomic equality characterization.

### Aggregate strictness theorem

Suppose a positive comparison certificate contains two profiles

\[
L_{s,r_1}=d_sG_s+r_1d_{s+1}G_{s+1},
\qquad
L_{s,r_2}=d_sG_s+r_2d_{s+1}G_{s+1},
\]

with \(r_1\ne r_2\), each assigned positive certificate weight. Then no atomic mass-one signed rule with at most \(N\) nodes can attain the sum of the two abstract floors.

### Proof

Aggregate equality would force equality in both nonnegative profile gaps. Hence for every distinct pair \(i\ne j\), with \(t=x_i\cdot x_j\),

\[
L_{s,r_1}(t)=L_{s,r_2}(t)=0.
\]

Subtracting gives

\[
(r_1-r_2)d_{s+1}G_{s+1}(t)=0,
\]

so \(G_{s+1}(t)=0\), and substitution then gives \(G_s(t)=0\).

Consecutive Gegenbauer polynomials have no common zero. Indeed their three-term recurrence implies that a common zero of \(G_s\) and \(G_{s+1}\) would also be a zero of \(G_{s-1}\), and induction would force \(G_0=0\), impossible. Thus no distinct node pair exists, contradicting the required \(N\)-node full-rank equality rule. ∎

### v21 witness

The certificate already contains, at \(s=3\), the two distinct positive-radius profiles

\[
r_1=0.005623413251903491,
\qquad
r_2=0.0068129206905796083.
\]

The exact verifier independently computes

\[
\gcd(G_3,G_4)=1.
\]

Therefore the v21 lower bound is **strict** for every actual atomic rule:

\[
\boxed{R_K(Q)>F_{21}.}
\]

This does not yet provide a uniform numerical improvement for arbitrary signed weights.

---

## 5. Theorem C — bounded-total-variation gap

For \(V<\infty\), let \(\mathcal Q_{N,V}\) be the class of mass-one signed atomic rules with at most \(N\) nodes and

\[
\|w\|_1\le V.
\]

### Statement

For every finite \(V\), there exists \(\varepsilon(V)>0\) such that

\[
\boxed{
R_K(Q)\ge F_{21}+\varepsilon(V)
\quad\text{for every }Q\in\mathcal Q_{N,V}.
}
\]

Consequently, any sequence of atomic rules whose risk approaches the abstract v21 floor must satisfy

\[
\|w^{(n)}\|_1\to\infty.
\]

Equivalently, its negative mass

\[
\nu_n={\|w^{(n)}\|_1-1\over2}
\]

must diverge.

### Proof

Pad every rule to exactly \(N\) labeled nodes by zero weights. The parameter set

\[
(S^{255})^N\times
\left\{w\in[-V,V]^N:\sum_iw_i=1,\ \sum_i|w_i|\le V\right\}
\]

is compact. Kernel discrepancy is continuous in nodes and weights, so it attains a minimum. Theorem B excludes equality with \(F_{21}\), hence the attained minimum is strictly larger. ∎

### Interpretation

The rank-and-block-trace boundary is not merely unavailable to a well-behaved signed rule. If it is an infimum at all, it can be approached only through increasingly ill-conditioned cancellation—unbounded total variation and, necessarily along subsequences, degenerating node configurations.

This sharply separates the abstract signed matrix optimizer from a numerically stable cubature algorithm.

---

## 6. Exact positive-rule Sturm corollary

For nonnegative weights one can make a tiny quantitative shared-profile improvement directly. Use only the four certificate profiles with \(s\in\{3,4\}\), and write

\[
q(t)=\sum_{j=1}^4 {y_j\over B_j}L_j(t)^2.
\]

An exact rational degree-10 Sturm calculation proves

\[
q(t)>1.30035\times10^{-12}
\qquad(-1\le t\le1).
\]

For a nonnegative mass-one rule, with \(u=\sum_iw_i^2\), the shared excess is

\[
A\left(u-{1\over N}\right)
+
\sum_{i\ne j}w_iw_jq(x_i\cdot x_j)
\ge
1.30035\times10^{-12}\left(1-{1\over N}\right).
\]

This gives

\[
R_K(Q)\ge2.2804646018345226\times10^{-7},
\]

or a same-cost cap \(1.067177432\times\) relative to the certified Kerdock upper endpoint.

This corollary is **not competitive with T22**, which already gives a 99.9767%-level positive-weight floor. Its purpose is to independently verify that shared-profile realizability creates a genuine quantitative penalty once signs cannot cancel.

---

## 7. What is now closed and what remains open

### Closed

- Re-optimizing the 146 profiles inside one shared rank/block-trace matrix relaxation.
- Adding unconstrained shared cross-harmonic blocks without sphere-evaluation identities.
- Claiming that aggregate equality might be realized by a finite, bounded-total-variation signed rule.

### Still open

- A **uniform explicit** \(\varepsilon>0\) for arbitrary signed rules with uncontrolled total variation.
- A quantitative \(\varepsilon(V)\) strong enough to matter at practical \(V\).
- Sphere-ideal, multiplication-operator, Hankel/catalecticant, or sum-of-squares constraints that separate the atomic moment variety from the shared abstract optimizer.
- Whether unbounded near-collision finite-difference rules can approach the shared abstract floor in the closure of signed atomic measures.

## 8. Correct next mathematical object

The next relaxation must contain at least one identity that every evaluation vector satisfies and the explicit shared optimizer does not—for example:

- commuting compressed coordinate multiplication operators;
- \(\sum_{k=1}^{256}X_k^2=I\) on a truncated quotient;
- joint localizing matrices for \(1-\|x\|^2\);
- catalecticant consistency across harmonic products;
- or a controlled jet/derivative extension that includes the closure generated by colliding signed nodes.

Merely enlarging the block-trace SDP cannot move the theorem.
