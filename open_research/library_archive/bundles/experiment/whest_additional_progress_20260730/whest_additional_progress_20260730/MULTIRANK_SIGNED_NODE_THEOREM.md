# A multi-rank obstruction for arbitrary signed-node cubature

**Status:** Computer-assisted certified for the infinite-width dimension-256, depth-32 normalized ReLU kernel.

## Theorem

Let

\[
Q=\sum_{i=1}^m w_i\delta_{x_i},\qquad m\le 66{,}048,
\qquad \sum_iw_i=1,
\]

where the nodes \(x_i\in S^{255}\) are arbitrary and the weights are arbitrary real numbers. Then

\[
R_{K_{32}}(Q)
\ge 7.90161513053615965080819\times10^{-8}.
\]

Relative to the certified complete-Kerdock MSE upper endpoint, every such rule retains at least

\[
0.3246802745520963
\]

of Kerdock MSE. Thus the theorem-permitted improvement factor is at most

\[
3.079953044204864.
\]

This is a global signed-node floor, not signed near-optimality.

## Rank/trace lemma

For a finite set \(S\) of spherical-harmonic degrees containing degree zero, let

\[
H_S=\bigoplus_{\ell\in S}H_\ell,
\qquad D_S=\dim H_S,
\]

and let \(v_S(x)\) be the evaluation feature map of an orthonormal basis. The reproducing kernel is

\[
L_S(x,y)=\langle v_S(x),v_S(y)\rangle
=\sum_{\ell\in S}d_\ell G_\ell(\langle x,y\rangle).
\]

For a signed mass-one rule define

\[
M_S(Q)=\sum_iw_i v_S(x_i)v_S(x_i)^T.
\]

Then \(\operatorname{rank}M_S\le m\le N\) and \(\operatorname{tr}M_S=D_S\). Moreover,

\[
R_{L_S^2}(Q)=\|I_{D_S}-M_S(Q)\|_F^2
\ge {D_S^2\over N}-D_S.
\]

No positivity or total-variation bound on the weights is used.

## Coefficientwise combination

For each of eleven fixed degree sets \(S_j\), let \(B_j=L_{S_j}^2\) and let

\[
B_j(t)=\sum_{\ell\ge0}b_{j\ell}G_\ell(t).
\]

The certificate supplies exact positive rational multipliers \(\lambda_j\) satisfying

\[
\sum_j\lambda_j b_{j\ell}\le k_\ell
\qquad(\ell\ge1),
\]

where \(k_\ell\) are the Gegenbauer coefficients of \(K_{32}\). For degrees through 26, the right-hand sides use directed interval lower bounds from a 27th-order Taylor jet; all omitted Maclaurin terms contribute nonnegatively. Above degree 26 the selected squared kernels have zero coefficient.

Therefore

\[
K_{32}-\sum_j\lambda_jB_j
\]

has nonnegative nonconstant Gegenbauer coefficients. Its discrepancy is nonnegative for arbitrary real mass-one weights, and hence

\[
R_{K_{32}}(Q)
\ge\sum_j\lambda_jR_{B_j}(Q)
\ge\sum_j\lambda_j\left({D_{S_j}^2\over N}-D_{S_j}\right).
\]

The last sum is the stated lower bound.

## Active harmonic feature sets

The fixed certificate uses:

- \(\{0,1,2,3\}\);
- \(\{0,1,2,3,4\}\);
- \(\{0,1,2,4,5\}\);
- \(\{0,3,5,6\}\);
- \(\{0,3,4,6,7\}\);
- \(\{0,1,3,5,7,8\}\);
- \(\{0,2,4,6,8,9\}\);
- \(\{0,4,5,7,9,10\}\);
- \(\{0,6,8,10,11\}\);
- \(\{0,5,7,9,11,12\}\);
- \(\{0,6,8,10,12,13\}\).

A numerical linear program was used only to discover this combination. The released certificate rounds every contribution downward by a rational safety factor and verifies every coefficient constraint with exact rational arithmetic. The theorem does not depend on LP optimality or floating-point feasibility.

## Reproducibility

Run:

```bash
python generate_multirank_signed_certificate.py
```

The generated `MULTIRANK_SIGNED_NODE_CERTIFICATE.json` is byte-stable across repeated executions. Its SHA-256 is included in the package manifest.

## Limitations

- The theorem is for the infinite-width kernel.
- It allows a remaining factor of approximately 3.08 and is therefore not a signed near-optimality theorem.
- It does not cover nonlinear estimators or network-adaptive rules, since those are not represented by one static signed measure.
- A stronger combination or a different rank feature construction may improve the floor.
