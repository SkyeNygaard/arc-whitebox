# Algebraic obstruction and strictness of the 93.7046% signed-static floor

**Date:** 2026-07-30  
**Status:** exact symbolic theorem for the two positive degree-3 comparison profiles; independent human/CAS review recommended.

## 1. Result

The current block-trace certificate proves that every static, network-independent, mass-one signed rule with at most

\[
N=66{,}048
\]

nodes has limiting-kernel risk at least

\[
2.28045159853140213494322646565331\times10^{-7}
=0.9370459569114724\,R_{\rm Kerdock}^{\rm upper}.
\]

The abstract matrix relaxation is exactly sharp, but equality by an atomic spherical rule would require an equal-weight `N`-point zero code for every positive comparison profile.

For each of the two positive `s=3` profiles in the released certificate, this report proves the much stronger bound

\[
\boxed{|X|\le 33{,}152}
\]

for an exact zero code `X subset S^255`. Therefore the released matrix floor is not attainable by a 66,048-node spherical rule. Moreover, the unrestricted signed infimum is separated from that floor by a strictly positive constant; the separation is presently nonexplicit.

## 2. The two comparison kernels

Let `G_l` be normalized Gegenbauer polynomials in dimension 256 and `d_l` the dimension of degree-`l` spherical harmonics. The two profiles are

\[
L_r(t)=d_3G_3(t)+r d_4G_4(t),
\]

with

\[
r_1=0.005623413251903491,
\qquad
r_2=0.0068129206905796083.
\]

Here

\[
G_3(t)=\frac{t(86t^2-1)}{85},
\qquad
G_4(t)=\frac{22360t^4-516t^2+1}{21845}.
\]

For each profile, exact root isolation proves that `L_r` has one root below `-1` and exactly three roots in `[-1,1]`. Thus an exact zero code is a spherical three-distance set.

The three spherical roots are approximately

| profile | roots in `[-1,1]` |
|---|---|
| `r_1` | `-0.1064049605`, `0.0014154470`, `0.1092044378` |
| `r_2` | `-0.1060960401`, `0.0017141632`, `0.1094871655` |

The certificate contains exact rational isolating intervals and the primitive integer quartics.

## 3. Polynomial-space rank bound

For a three-distance set with inner products `alpha_1,alpha_2,alpha_3`, define the Lagrange polynomials

\[
F_i(t)=\prod_{j\ne i}\frac{t-\alpha_j}{\alpha_i-\alpha_j}.
\]

On the point set,

\[
[F_i(\langle x_a,x_b\rangle)]_{a,b}
=k_i I+A_i,
\qquad
k_i=F_i(1),
\]

where `A_i` is the integer `0/1` adjacency matrix for distance `alpha_i`.

Since `F_i` has degree at most two, the evaluation matrix has rank at most

\[
D_2=h_0+h_1+h_2
=1+256+\left(\binom{257}{2}-1\right)
=\frac{256(259)}2
=33{,}152.
\]

Therefore, if `n>D_2`, then `k_iI+A_i` is singular and `-k_i` must be an eigenvalue of the integer matrix `A_i`.

## 4. Algebraic-integer obstruction

For each profile, elimination over the four quartic roots gives the minimal polynomial of every ordered-pair interpolation value `k_i`.

The exact certificate proves:

1. modular factorization patterns give a 4-cycle, a 3-cycle, and a transposition, so the quartic Galois group is `S_4`;
2. the degree-12 resultant factor is squarefree;
3. `S_4` is transitive on ordered distinct root pairs, hence the squarefree degree-12 factor is irreducible;
4. its primitive integer leading coefficient is not `1`—it has 246 digits for `r_1` and 258 digits for `r_2`.

Consequently every `k_i` has algebraic degree 12 but is **not an algebraic integer**.

Every eigenvalue of an integer matrix is an algebraic integer. Hence `-k_i` cannot be an eigenvalue of `A_i`. The rank drop required when `n>D_2` is impossible. Thus

\[
\boxed{n\le D_2=33{,}152}.
\]

This strengthens the earlier conjugate-multiplicity bound `n<=36,165`.

## 5. Why signed weights cannot approach equality through a border-rank limit

Let `Phi(x)` be the feature map for either profile, with

\[
\langle\Phi(x),\Phi(y)\rangle=L_r(\langle x,y\rangle),
\qquad
\|\Phi(x)\|^2=T=L_r(1).
\]

For a signed atomic rule with at most `N` nonzero weights,

\[
M=\sum_iw_i\Phi(x_i)\Phi(x_i)^T,
\qquad \operatorname{tr}M=T.
\]

Write the excess above the abstract rank floor as

\[
\delta(M)=\|M\|_F^2-\frac{T^2}{N}.
\]

If `rank(M)<=N-1`, then

\[
\delta(M)\ge \frac{T^2}{N(N-1)}.
\]

The same bound holds if `rank(M)=N` but `M` has a nonpositive eigenvalue: the smallest squared-eigenvalue sum with trace `T` and one eigenvalue at most zero occurs with one zero eigenvalue and `N-1` equal positive eigenvalues.

Therefore any sequence with `delta->0` must eventually have rank exactly `N` and be positive definite on its image. With `M=E^TWE`, full row rank of `E` and Sylvester inertia imply that every diagonal entry of `W` is positive. Hence the weights lie in the probability simplex.

For positive weights the gap decomposes exactly as

\[
\delta
=T^2\left(\sum_iw_i^2-\frac1N\right)
+\sum_{i\ne j}w_iw_jL_r(\langle x_i,x_j\rangle)^2.
\]

Thus `delta->0` forces

\[
w_i\to\frac1N
\]

and every off-diagonal kernel value to approach zero. Compactness of `(S^255)^N` gives a convergent subsequence whose limit is an `N`-point zero code. This contradicts the exact `33,152` bound.

Therefore there exists a profile-dependent constant `eta_r>0` such that

\[
\delta(M)\ge\eta_r
\]

for every signed rule with at most `66,048` nodes.

Because both `s=3` components enter the full comparison certificate with strictly positive multipliers, the true static signed infimum is strictly larger than the displayed `93.70459569114724%` floor.

## 6. Explicit—though microscopic—resultant separation

The nonattainment argument can be made numerically explicit without solving a three-point SDP. Round every pair to its nearest spherical root and let `A_i` be the resulting integer adjacency matrix. The degree-two interpolation matrix `B_i` has rank at most `D2`, while

\[
B_i=k_iI+A_i+R_i.
\]

Root separation gives an exact constant `C_i` with

\[
\|R_i\|_F^2\le C_i^2\sum_{a\ne b}L_r(\langle x_a,x_b\rangle)^2.
\]

Let `P_i` be the primitive irreducible degree-12 polynomial of `k_i`, with leading coefficient `a_i>1`. A Cauchy root bound supplies an integer `H_i` containing every conjugate of `k_i` and every adjacency eigenvalue. Since

\[
\operatorname{Res}(P_i,\chi_{A_i})\in\mathbb Z\setminus\{0\},
\]

every adjacency eigenvalue `lambda` satisfies the explicit Liouville-type separation

\[
|k_i+\lambda|\ge
\varepsilon_i:=a_i^{-N}H_i^{-(12N-1)}.
\]

The Eckart–Young theorem then gives

\[
\sum_{a\ne b}L_r(\langle x_a,x_b\rangle)^2
\ge
\frac{(N-D_2)\varepsilon_i^2}{C_i^2}.
\]

After the positive-weight transfer lemma, this produces an explicit strict increase in the full MSE certificate. The two certified base-10 logarithms are approximately

\[
-100{,}429{,}270.28
\quad\text{and}\quad
-104{,}055{,}118.76.
\]

Thus the published floor can be strengthened literally to

\[
R(Q)>2.28045159853140213494322646565331\times10^{-7}
\]

with a fully explicit rational margin. The margin is far too small to change any printed decimal or competition conclusion, but it closes the logical distinction between “strict by compactness” and “quantitatively strict.”

Regenerate with `verify_explicit_resultant_gap.py`.

## 7. What is now closed

- Exact attainment of the abstract block-trace floor by spherical nodes is impossible.
- A sequence with unbounded signed weights cannot approach the floor.
- The unrestricted signed infimum has a genuine positive separation from the released constant.
- The obstruction already follows from either positive `s=3` profile.

## 8. What remains

The positive separation `eta_r` is not numerically evaluated. The next useful theorem is a **robust** zero-code bound: lower-bound

\[
\frac1{N^2}\sum_{i\ne j}L_r(\langle x_i,x_j\rangle)^2
\]

for every `N`-point configuration. Ordinary two-point Delsarte LPs appear unable to do this; a three-point semidefinite or sphere-ideal certificate is the leading route.

## 9. Reproduction

```bash
python verify_s3_zerocode_obstruction.py
```

Expected output:

```json
{
  "verified": true,
  "profiles": 2,
  "D2": 33152,
  "max_zero_code_size": 33152,
  "node_budget": 66048
}
```
