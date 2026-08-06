# T43 — Global rank floor for arbitrary signed nodes

**Status:** exact abstract theorem plus a computer-assisted numerical specialization. The abstract proof is conventional. The depth-32 numerical constant depends on the directed coefficient certificate in T42 and requires independent interval-stack reproduction before publication.

## Abstract theorem

Let `P` be normalized spherical measure. Let

\[
K(t)=\sum_{r\ge0}k_rG_r(t),\qquad k_r\ge0\quad(r\ge1),
\]

be a zonal positive-definite kernel. Choose a finite set `S` of harmonic degrees and let `H_S` be the direct sum of their spherical-harmonic spaces, of dimension

\[
D=\sum_{\ell\in S}d_\ell.
\]

For an orthonormal basis of `H_S`, let `v(x) in R^D` be the evaluation vector. Its reproducing kernel is

\[
L_S(x,y)=\langle v(x),v(y)\rangle
=\sum_{\ell\in S}d_\ell G_\ell(\langle x,y\rangle).
\]

Write

\[
L_S(t)^2=\sum_{r=0}^{2\max S}b_rG_r(t),
\qquad b_r\ge0,
\]

and define

\[
\gamma_S=
\min_{r\ge1:b_r>0}\frac{k_r}{b_r}.
\]

Then every static linear cubature rule

\[
Q=\sum_{i=1}^{m}w_i\delta_{x_i},
\qquad m\le N,
\qquad \sum_iw_i=1,
\]

with arbitrary real weights satisfies

\[
R_K(Q)
\ge
\gamma_S\left(\frac{D^2}{N}-D\right)
\]

whenever `D>N`.

The rule may use arbitrary spherical nodes and arbitrary positive or negative weights. No total-variation or negative-mass bound is assumed.

## Proof

Define the moment matrix

\[
M_Q=\sum_iw_i v(x_i)v(x_i)^T.
\]

The addition theorem gives `||v(x)||^2=D`, hence

\[
\operatorname{tr}M_Q=D.
\]

Moreover, `rank(M_Q)<=m<=N`, even for signed weights.

If the nonzero eigenvalues of `M_Q` are `lambda_1,...,lambda_s`, where `s<=N`, then

\[
\begin{aligned}
\|I_D-M_Q\|_F^2
&=D-2\operatorname{tr}M_Q+\operatorname{tr}(M_Q^2)\\
&=-D+\sum_{j=1}^s\lambda_j^2\\
&\ge-D+\frac{(\sum_j\lambda_j)^2}{s}\\
&\ge\frac{D^2}{N}-D.
\end{aligned}
\]

The cubature discrepancy for the squared reproducing kernel is exactly

\[
R_{L_S^2}(Q)=\|I_D-M_Q\|_F^2.
\]

By the definition of `gamma_S`, every nonconstant Gegenbauer coefficient of `K-gamma_S L_S^2` is nonnegative. Constant coefficients cancel for mass-one rules, so

\[
R_K(Q)\ge\gamma_SR_{L_S^2}(Q).
\]

Combining the inequalities proves the theorem.

## Dimension-256 depth-32 specialization

Take

\[
S=\{0,1,2,3\},
\qquad
D=2,861,952,
\qquad
N=66,048.
\]

The exact coefficients of `L_S^2` are:

| degree | `b_r` |
|---:|---:|
| 0 | 2,861,952 |
| 1 | 16,842,752 |
| 2 | 3,657,761,472,384 / 1,703 |
| 3 | 556,905,594,880 / 131 |
| 4 | 38,766,033,821,696 / 143 |
| 5 | 23,823,820,062,720 / 131 |
| 6 | 11,140,944,743,219,200 / 1,441 |

The coefficient comparison must include **all active nonconstant degrees 1 through 6**. A prior exploratory script displayed only degrees 2, 4 and 6; that was an incomplete displayed proof obligation, although degree 6 was still the minimum and the numerical conclusion was unchanged.

The strengthened order-47 certificate gives

\[
\gamma_S
\ge
1.4046634297844856592\times10^{-16},
\]

with degree 6 binding. Therefore every static, network-independent, arbitrary-signed, mass-one rule with at most 66,048 nodes satisfies

\[
\boxed{
R_{K_{32}}(Q)
\ge
1.7017556669835916\times10^{-8}
}.
\]

Relative to the certified complete-Kerdock MSE used by the package,

\[
R_{K_{32}}(Q)
\ge
0.0699257668273\,R_{\mathrm{Kerdock}}.
\]

Thus the theorem permits at most

\[
\boxed{14.30088\times}
\]

improvement over Kerdock within this static signed class.

## Exhaustive low-degree feature-space check

The verifier checks every nonempty subset of harmonic degrees `{0,1,2,3}`. The full degree-`<=3` space gives the strongest certified rank floor. Spaces not containing degree 3 have dimension below the node budget and yield no positive rank obstruction; degree-3-only and smaller mixed spaces are slightly weaker.

## Scope

Covered:

- depth-32 infinite-width normalized ReLU kernel;
- dimension 256;
- at most 66,048 arbitrary nodes;
- arbitrary real mass-one weights;
- deterministic rules, or randomized rules independent of the realized field;
- static linear cubature.

Not covered:

- support or weights chosen from the realized network;
- nonlinear aggregation;
- finite-width kernels without new coefficient bounds;
- free-total-mass rules;
- signed near-optimality.

The theorem is a nonzero global floor, not a proof that Kerdock is close to the signed optimum. A 14.3x permitted improvement is still much larger than the competition-scale gap.
