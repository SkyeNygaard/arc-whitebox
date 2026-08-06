# T38 — Exact finite-width Kerdock-line support theorem

## Status

**PROVED UNDER AN EXPLICIT RANDOM-NETWORK MODEL.**

This closes the finite-width qualification for T27, the fixed mutually-unbiased-basis line universe. It does not close finite-width T22 over arbitrary nodes.

## Assumptions

1. Inputs lie on `S^(d-1)`.
2. The first-layer rows are independent standard Gaussian vectors.
3. All later weights/randomness `Z` are independent of the first layer.
4. For fixed `Z`, the output is a square-integrable Hilbert-valued function
   \[
   Y(x)=F_Z(W^{(1)}x).
   \]
5. The rule is static/network-independent and supported on symmetrized antipodal lines from a fixed union of real mutually unbiased orthonormal bases.
6. Line weights are arbitrary real numbers summing to one.
7. The antipodally even output is nonconstant with positive probability. For a finite ReLU network this is the required nondegeneracy condition.

## Lemma 1 — exact finite-width Gaussian-noise expansion

For unit inputs with inner product `t`, define

\[
K_m(t)=\mathbb E\langle Y(x),Y(y)\rangle.
\]

Then at every finite width and depth

\[
K_m(t)=\sum_{n\ge0}a_nt^n,
\qquad a_n\ge0.
\]

### Proof

Choose coordinates so that first-layer preactivations are Gaussian vectors

\[
G' = tG+\sqrt{1-t^2}\widetilde G.
\]

Condition on all later randomness `Z` and expand `F_Z` in the orthonormal multivariate Hermite basis,

\[
F_Z(g)=\sum_\alpha c_\alpha(Z)H_\alpha(g).
\]

Mehler/noise stability gives

\[
\mathbb E[\langle F_Z(G),F_Z(G')\rangle\mid Z]
=\sum_\alpha\|c_\alpha(Z)\|^2t^{|\alpha|}.
\]

Average over `Z` and group by total degree. No width limit is used.

## Lemma 2 — line-kernel sign conditions

For the symmetrized line output

\[
S(x)={Y(x)+Y(-x)\over2},
\]

the line kernel is

\[
\overline K_m(t)=\sum_{r\ge0}a_{2r}t^{2r}.
\]

On a real MUB line universe define

\[
A=\overline K_m(1),\quad O=\overline K_m(0),\quad
C=\overline K_m(1/\sqrt d).
\]

Then

\[
A-O=\sum_{r\ge1}a_{2r}>0,
\]

\[
O-C=-\sum_{r\ge1}a_{2r}d^{-r}<0,
\]

and

\[
(A-O)+d(O-C)
=\sum_{r\ge2}a_{2r}(1-d^{1-r})>0.
\]

The last inequality follows from nondegeneracy. To justify it for finite ReLU networks, suppose all even Hermite coefficients of degree at least four vanished. For almost every later-weight realization, the continuous even piecewise-affine function `(F_Z(g)+F_Z(-g))/2` would equal a quadratic polynomial almost everywhere under a full-support Gaussian law, hence everywhere. A quadratic polynomial that is piecewise affine on a finite polyhedral partition has zero Hessian, so it is affine; evenness then makes it constant, contradicting nondegeneracy.

## Lemma 3 — MUB support extremality

For any three-value MUB line kernel satisfying

\[
a:=A-O>0,\qquad b:=O-C<0,\qquad a+bd>0,
\]

the risk, up to an irrelevant constant, is

\[
R(w)=b\sum_bS_b^2+a\sum_{b,i}w_{bi}^2.
\]

If basis `b` uses `r_b` lines and has mass `S_b`, equal within-basis weights minimize the second term. The reduced coefficient is

\[
c(r_b)=b+{a\over r_b}>0.
\]

Optimizing masses with `sum_bS_b=1` yields positive masses

\[
S_b\propto {1\over c(r_b)}
={r_b\over a+br_b}.
\]

The optimized risk decreases as

\[
H(r_1,\ldots,r_k)=\sum_bh(r_b),
\qquad h(r)={r\over a+br}
\]

increases. Since `b<0`, `h` is strictly increasing and strictly convex on `[0,d]`. At a fixed total line count, transferring a line from a smaller partial basis to a larger one strictly increases `H`. Therefore the unique support-size pattern is as many complete `d`-line bases as possible and at most one partial basis.

## Theorem

At every finite width and finite depth satisfying the assumptions, and for every line budget `P`, minimum finite-width ensemble MSE over arbitrary real mass-one weights on the fixed MUB line universe is attained by

1. `floor(P/d)` complete orthonormal bases;
2. at most one additional partial basis;
3. equal positive weights within each active basis;
4. positive basis masses proportional to
   \[
   {1\over (O-C)+(A-O)/r_b}.
   \]

All available lines are used.

## What this closes

- deletion patterns inside the fixed line universe;
- unequal weights inside a basis;
- negative line or basis masses;
- mixtures of multiple partial bases;
- the finite-width objection to T27.

## What remains open

- arbitrary spherical nodes at finite width;
- arbitrary off-universe signed nodes;
- network-dependent support or weights;
- nonlinear estimators;
- exact finite-width absolute Kerdock MSE and arbitrary-node near-optimality.
