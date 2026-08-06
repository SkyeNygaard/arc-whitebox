# T70 — Inertia-strengthened arbitrary-signed atomic floor

**Date:** 2026-07-30  
**Status:** Exact spectral theorem plus exact-rational degree-280/order-320 specialization. The kernel lower endpoints inherit the v21 directed interval stack; an external Arb/FLINT reproduction remains the public-release gate.

## 1. Main result

For the dimension-256, depth-32 limiting normalized-ReLU kernel, every static network-independent mass-one linear rule using at most 66,048 arbitrary spherical nodes and arbitrary real weights satisfies

\[
\boxed{
R_K(Q)\ge
2.2804870463653914348948735097257249\times10^{-7}.
}
\]

Relative to the certified complete-Kerdock MSE upper endpoint,

\[
\boxed{
R_K(Q)\ge
0.9370605225569535\,R_K(Q_{\rm Kerdock}),
}
\]

so

\[
\boxed{
\text{same-cost raw improvement}\le1.0671669288460727\times.
}
\]

This strictly strengthens the v21 degree-280 block-trace floor

\[
0.9370459569114724\,R_K(Q_{\rm Kerdock}).
\]

The numerical gain is small, but the new ingredient is fundamental: **the inertia of an actual signed atomic moment matrix**.

---

## 2. Positive-index rank lemma

Let \(M=M^\top\) have trace \(T>0\), and suppose it has at most \(p\) positive eigenvalues. Then

\[
\boxed{\|M\|_F^2\ge {T^2\over p}.}
\]

### Proof

Let \(\lambda_i^+\) be the positive eigenvalues. Since the nonpositive eigenvalues have nonpositive sum,

\[
\sum_i\lambda_i^+\ge T.
\]

By Cauchy,

\[
\|M\|_F^2
\ge\sum_i(\lambda_i^+)^2
\ge {\left(\sum_i\lambda_i^+\right)^2\over p}
\ge {T^2\over p}.
\]

The infimum is approached when the \(p\) positive eigenvalues are equal and every remaining eigenvalue approaches zero from below. ∎

---

## 3. Atomic inertia transfer

For one comparison profile, let \(E\) be the node-evaluation matrix and \(W=\operatorname{diag}(w_i)\). The profile moment matrix is

\[
M=E^\top W E.
\]

The number of positive eigenvalues of \(M\) is at most the number of positive entries of \(W\). This follows from Sylvester inertia monotonicity under congruence. If \(E\) has full row rank, \(M\) and \(W\) have the same nonzero inertia; if it does not, the inequality is only stronger.

Now split all mass-one rules into two cases.

### Case A — exactly \(N\) positive nonzero weights

This is a nonnegative \(N\)-node probability rule, so the stronger T22 positive-weight theorem applies.

### Case B — every other rule

The rule has at most \(N-1\) positive weights. Therefore every comparison-profile moment matrix has at most \(N-1\) positive eigenvalues. If

\[
T=\sum_\ell a_\ell d_\ell,
\qquad
S_2=\sum_\ell a_\ell^2d_\ell,
\]

then

\[
R_{L_a^2}(Q)
=\|A-M\|_F^2
=\|M\|_F^2-S_2
\ge {T^2\over N-1}-S_2.
\]

Compared with the old block-trace floor,

\[
{T^2\over N}-S_2,
\]

the exact atomic inertia increment is

\[
\boxed{
{T^2\over N(N-1)}.
}
\]

This applies to:

- any rule with a negative weight;
- any positive rule with fewer than \(N\) active nodes;
- any rule whose evaluation matrix loses rank.

---

## 4. Reoptimized comparison certificate

For each v21 component \(j\), write

\[
B_j={T_j^2\over N}-S_{2,j},
\qquad
B_j^-={T_j^2\over N-1}-S_{2,j}.
\]

If \(y_j\) denotes the contribution under the old normalization, then its inertia-strengthened contribution is

\[
y_j{B_j^-\over B_j}.
\]

The Gegenbauer coefficient consumption remains exactly the same. Hence the strongest certificate on the frozen 146-profile grid is a linear program with:

- the original degree-1 through degree-320 coefficient-capacity constraints;
- objective \(\sum_j y_jB_j^-/B_j\);
- \(y_j\ge0\).

An exploratory floating-point solve was shrunk by \(0.9999999\), rounded downward to a \(10^{-30}\) grid, and then checked entirely with exact rational arithmetic. The final witness has 134 active components.

Certified quantities:

| quantity | value |
|---|---:|
| base rank-floor portion | \(2.2804524604975442\times10^{-7}\) |
| inertia-strengthened floor | \(2.2804870463653914\times10^{-7}\) |
| fraction of Kerdock upper MSE | \(0.9370605225569535\) |
| same-cost cap | \(1.0671669288460727\times\) |
| minimum active coefficient slack | \(3.55015\times10^{-26}\) at degree 267 |
| minimum audited slack | \(1.87222\times10^{-47}\) at degree 320 |

The old v21 weights, without reoptimization, already give

\[
2.2804861843861462\times10^{-7}.
\]

The exact reoptimization adds another

\[
8.61979\times10^{-14}.
\]

---

## 5. Arbitrary-total-mass version

The same inertia argument can be combined with the unused constant harmonic coefficient. Optimizing over total mass \(s=\sum_iw_i\) gives

\[
\boxed{
R_K(Q)\ge
2.2804865128207167354137975131119212\times10^{-7}
}
\]

for every static arbitrary-real-weight rule with at most 66,048 nodes, without assuming mass one.

Equivalently,

\[
R_K(Q)\ge0.9370603033214825\,R_K(Q_{\rm Kerdock}),
\]

with cap

\[
1.0671671785214067\times.
\]

The minimizing relaxed total mass is

\[
s_*=0.9999997660391557\ldots
\]

so the constant mode again forces near-exact mass.

---

## 6. Hostile checks

### Indefinite moment matrices

The proof does not assume \(M\succeq0\). Negative eigenvalues increase the Frobenius cost while reducing the trace carried by the positive eigenvalues, which is exactly why the positive-index lemma works.

### Node collisions and rank loss

Collisions cannot evade the result. They reduce the rank or positive index and therefore strengthen the lower bound.

### Vanishing negative weight

A rule with one negative weight tending to zero approaches the \(N-1\)-positive boundary. The constant \(T^2/(N-1)\) is the correct infimum for that inertia class; strict negativity does not allow replacing the infimum by a larger closed-form constant without another stability constraint.

### Positive rules

The inertia certificate is not used for exactly \(N\) positive weights. T22 supplies a much stronger bound there. The universal signed result is the smaller of the T22 floor and the nonpositive-inertia floor, namely the latter.

### Adaptation and nonlinear output

Nothing here covers nodes or weights selected from the realized network, pilot-value adaptation, or nonlinear estimators. The result is static and linear.

---

## 7. What this closes

- The exact v21 numerical floor was not the strongest consequence of its own profiles.
- Actual signed atomic inertia gives a rigorous improvement absent from abstract rank/block-trace matrices.
- Static arbitrary-linear same-cost improvement is now capped below \(1.067167\times\).
- The result covers unrestricted total mass after adding the constant-harmonic residual.

## 8. Remaining static gap

The remaining 6.294% risk gap cannot be closed by:

- more sharing of unconstrained block moment matrices (T61);
- radius-grid refinement alone;
- or ignoring atomic inertia.

Further progress requires stronger evaluation-variety constraints, such as sphere-ideal localizing identities, commuting multiplication operators, or a quantitative treatment of the unbounded-total-variation collision closure.
