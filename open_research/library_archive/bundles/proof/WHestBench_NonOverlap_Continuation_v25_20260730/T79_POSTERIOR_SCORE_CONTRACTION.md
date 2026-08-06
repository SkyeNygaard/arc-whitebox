# T79 — Posterior-score contraction for Gaussian second chaos

**Status:** exact abstract theorem; complete-Kerdock specialization independently reproduced in double precision. Directed-rounding publication audit remains open.

## 1. Setting

Let `W` be an isonormal Gaussian process on a real Hilbert space `H`. Let `A` be a self-adjoint Hilbert–Schmidt operator and use the normalized quadratic chaos

\[
\mathcal Q(A)=\frac{\langle W,AW\rangle-\operatorname{tr}A}{\sqrt2},
\qquad
\mathbb E\mathcal Q(A)^2=\|A\|_{HS}^2.
\]

Suppose a baseline transcript reveals the Gaussian subspace `U`, with orthogonal projections `P` onto `U` and `Q=I-P`. Write

\[
B=QAP:U\to U^\perp,
\qquad
C=QAQ:U^\perp\to U^\perp.
\]

If `u=PW`, the exact residual after the baseline posterior mean is

\[
\mathcal Q(A)-\mathbb E[\mathcal Q(A)\mid W_U]
=\sqrt2\,\langle W_Q,Bu\rangle+\mathcal Q(C).
\]

Consequently

\[
R_0=2\|B\|_{HS}^2+\|C\|_{HS}^2.
\]

## 2. Matched posterior-score observation

For a realized baseline transcript `u`, define its posterior-score direction

\[
r(u)=Bu\in U^\perp.
\]

When `r(u)\ne0`, reveal one additional Gaussian linear observation

\[
Y_u=\left\langle W_Q,\frac{r(u)}{\|r(u)\|}\right\rangle.
\]

Then the complete observed–unobserved cross term becomes known exactly. The remaining conditional risk is at most

\[
\boxed{R_1\le \|C\|_{HS}^2.}
\]

The observation is adaptive, but its role is exact and transparent: it measures the single latent scalar multiplying the realized score direction.

### Approximate direction

Let `\widehat r(u)` be a unit direction and let `\theta(u)` be its angle with `r(u)`. After observing `\langle W_Q,\widehat r(u)\rangle`,

\[
R_1\le
2\,\mathbb E\![\|Bu\|^2\sin^2\theta(u)\u001b]
+\|C\|_{HS}^2.
\]

Define the energy-weighted angular error

\[
\varepsilon=
\frac{\mathbb E[\|Bu\|^2\sin^2\theta(u)]}{\|B\|_{HS}^2}.
\]

Then

\[
\boxed{
R_1\le 2\varepsilon\|B\|_{HS}^2+\|C\|_{HS}^2.
}
\]

Thus ordinary sign accuracy or unweighted cosine is not the correct score. The correct gate is posterior-score-energy-weighted squared sine.

## 3. Kernel-matrix formulas

Let `phi_x` be the Gaussian feature map, and define

\[
A=\int \phi_x\otimes\phi_x\,d\mu(x).
\]

For baseline points `X=(x_i)`, let

\[
K_{ij}=\langle\phi_{x_i},\phi_{x_j}\rangle,
\qquad
M_{ij}=\langle\phi_{x_i},A\phi_{x_j}\rangle,
\qquad
E_{ij}=\langle\phi_{x_i},A^2\phi_{x_j}\rangle.
\]

Then

\[
\|PAP\|_{HS}^2=\operatorname{tr}[(K^{-1}M)^2],
\]

\[
\operatorname{tr}(PA^2)=\operatorname{tr}(K^{-1}E),
\]

and therefore

\[
\boxed{
\|B\|_{HS}^2
=\operatorname{tr}(K^{-1}E)
-\operatorname{tr}[(K^{-1}M)^2],
}
\]

\[
\boxed{
\|C\|_{HS}^2
=\|A\|_{HS}^2
-2\operatorname{tr}(K^{-1}E)
+\operatorname{tr}[(K^{-1}M)^2].
}
\]

If the covariance kernel has normalized Gegenbauer expansion

\[
C(t)=\sum_{\ell\ge0}c_\ell G_\ell(t),
\]

then

\[
D(t)=\sum_{\ell\ge0}\frac{c_\ell^2}{h_\ell}G_\ell(t),
\qquad
E(t)=\sum_{\ell\ge0}\frac{c_\ell^3}{h_\ell^2}G_\ell(t).
\]

## 4. Off-support point-query score

For a realized baseline vector `z`, define

\[
k_x=(C(x_i,x))_i,
\qquad
d_x=(D(x_i,x))_i.
\]

The posterior score evaluated at a new point is

\[
\boxed{
s_z(x)
=d_x^\top K^{-1}z
-k_x^\top K^{-1}MK^{-1}z.
}
\]

The posterior variance of the new Gaussian observation is

\[
v(x)=C(x,x)-k_x^\top K^{-1}k_x.
\]

Hence the exact one-point fraction of the cross-score energy captured at `x` is

\[
\frac{s_z(x)^2}{v(x)\|r(z)\|^2}.
\]

For a point set `Y`, replace the scalar denominator by the posterior covariance matrix and use the corresponding Schur complement. This is the exact acquisition objective; generic output magnitude is not.

## 5. Complete-Kerdock specialization

For the dimension-256, depth-31 preactivation kernel on the complete 129-basis antipodal Kerdock support, the independent reproduction gives

\[
\|A-PAP\|_{HS}^2
=4.9365176280\times10^{-7},
\]

\[
\|B\|_{HS}^2
=2.4675034893\times10^{-7},
\qquad
\|C\|_{HS}^2
=1.5106494\times10^{-10}.
\]

Therefore

\[
\frac{2\|B\|_{HS}^2}{R_0}=0.999693985,
\qquad
\frac{\|C\|_{HS}^2}{R_0}=0.000306015.
\]

The reported fixed-support second-chaos floor is thus almost entirely a cross-chaos information deficit, not an irreducible hidden quadratic component.

## 6. Rank-one global-sector concentration

The complete-MUB association algebra has five observation sectors. Their contributions to `\|B\|_{HS}^2` show that the one-dimensional even-global sector alone contributes approximately

\[
99.66665\%
\]

of the cross-score energy.

Let `r_0` be its fixed residual direction. Revealing the single scalar `W(r_0/\|r_0\|)` leaves second-chaos risk at most

\[
1.79618\times10^{-9}
\]

before multiplication by the ReLU Hermite coefficient. With `b_2^2=1/(2\pi)`, this is at most approximately

\[
0.001175\,R_{\mathrm{Kerdock}}.
\]

This does **not** give a cheap estimator. It identifies the exact missing scalar.

## 7. Interpretation

1. The fixed-support posterior floor is fragile to one correctly matched off-support contraction.
2. The missing object is a signed scalar posterior innovation, not a generic coefficient vector.
3. The dominant term is a product of the already observed global Gaussian mode and one unobserved matched scalar. This gives a theoretical explanation for the empirical importance of lower-order mean/pair anchors.
4. Independent Monte Carlo estimation is uneconomic; an exact identity, structured integration, or already-computed shared-arithmetic contraction is required.
5. This theorem does not close finite-width shared-source methods, CLAF's richer late-source field, or full-weight algorithms.
