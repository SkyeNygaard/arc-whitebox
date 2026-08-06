# T16 completion — exact all-degree auxiliary-LP optimum

**Date:** 2026-07-30  
**Status:** **COMPUTER-ASSISTED CERTIFIED**  
**Supersedes:** the earlier statement that only reduced-cost negativity had been proved.

## 1. Optimization problem

Let

\[
K_0(t)=t,\qquad K_{m+1}(t)=\kappa(K_m(t)),
\]

where

\[
\kappa(t)=\frac{\sqrt{1-t^2}+(\pi-\arccos t)t}{\pi},
\]

and take `d=256`, `m=32`, and `N=66,048`. Let

\[
G_\ell(t)=\frac{C_\ell^{127}(t)}{C_\ell^{127}(1)}
\]

be the normalized Gegenbauer polynomial, so `G_l(1)=1`.

For an admissible finite Gegenbauer polynomial

\[
h(t)=\sum_{\ell=0}^L c_\ell G_\ell(t),
\qquad c_\ell\ge0\quad(\ell\ge1),
\qquad h(t)\le K_{32}(t),
\]

define

\[
\Phi(h)
= c_0+\frac{K_{32}(1)-h(1)}N
=\frac1N+\frac{N-1}{N}c_0-\frac1N\sum_{\ell\ge1}c_\ell.
\]

The same proof covers absolutely convergent nonnegative Gegenbauer expansions for which the displayed quantities are well-defined.

## 2. Theorem

The auxiliary LP has a unique optimizer of degree five. Its three contact points are the three roots of

\[
P(t)=22102t^3+21930t^2-87t-85.
\]

The optimizer is the unique degree-five Hermite interpolant `h_*` satisfying

\[
h_*(t_j)=K_{32}(t_j),\qquad
h_*'(t_j)=K_{32}'(t_j),\qquad j=1,2,3.
\]

All five nonconstant normalized-Gegenbauer coefficients of `h_*` are strictly positive. Every reduced cost of degree at least six is strictly negative. Consequently every optimizer has no coefficient above degree five, and `h_*` is the unique optimizer.

The certified optimum MSE lower-bound interval is

\[
[2.43309285875659379174672051773578246,\;
  2.43309285875659608044705321866494159]\times10^{-7}.
\]

Using the existing certified Kerdock MSE interval, the resulting Kerdock-over-auxiliary-optimum ratio lies in

\[
[1.00023324172949850838933551289744743,\;
  1.00023324172950038991821730217119802],
\]

so the certified relative-excess upper bound tightens from `0.0233655011%` to

\[
\boxed{0.023324172950039\%}.
\]

This remains a bound relative to the optimum cubature rule in T22's static nonnegative class; it is not a finite-width, adaptive, nonlinear, or arbitrary-signed theorem.

## 3. Exact dual quadrature

Set

\[
q_0=\frac{N-1}{N},\qquad q_\ell=-\frac1N\quad(\ell\ge1).
\]

Let `L` be the linear functional satisfying `L[G_l]=q_l`. Exact conversion to monomial moments through degree five gives

\[
\left(L[1],L[t],L[t^2],L[t^3],L[t^4],L[t^5]\right)
=
\left(
\frac{66047}{66048},
-\frac1{66048},
\frac{257}{66048},
-\frac1{66048},
\frac1{33024},
-\frac1{66048}
\right).
\]

The monic cubic orthogonal to `1,t,t^2` under `L` is

\[
t^3+\frac{255}{257}t^2-\frac{87}{22102}t-\frac{85}{22102}.
\]

Its three roots are simple and lie in `(-1,1)`. The corresponding three-node Gaussian quadrature weights are strictly positive and satisfy

\[
L[p]=\sum_{j=1}^3\lambda_jp(t_j)
\]

for every polynomial of degree at most five. Positivity was certified by exact rational root intervals and the Lagrange-weight formula.

The all-degree certificate proves

\[
r_\ell=q_\ell-\sum_j\lambda_jG_\ell(t_j)<0
\qquad(\ell\ge6).
\]

Degrees `6..14,658` were checked with exact integer arithmetic. For `l>=14,659`, the normalized Gegenbauer bound

\[
|G_\ell(t)|\le
\frac{254!}{127![\ell(1-t^2)]^{127}}
\]

forces strict negativity. A separate C++17/Boost `cpp_int` implementation reproduced the entire finite sweep and the tail cutoff; its worst mode reduces to

\[
r_7=-\frac{2327215}{9290262647272}.
\]

## 4. Sixth-derivative positivity

The missing primal step is supplied by the following lemma.

### Lemma

\[
K_{32}^{(6)}(t)>0\qquad(-1<t<1).
\]

### Proof structure

Write

\[
K_{32}=F\circ u,\qquad u=\kappa,\qquad F=\kappa^{\circ31}.
\]

For `t>=0`, every derivative of `u` through order six is nonnegative, with `u^{(6)}>0`. Every intermediate argument of `F` lies in `[0,1)`, where the same derivative signs hold. Faà di Bruno's formula therefore makes `K_{32}^{(6)}>0` directly.

For `t<0`, set `s=sqrt(1-t^2)` and

\[
p=u'(t)=\frac{\pi/2+\arcsin t}{\pi}.
\]

The required derivatives are

\[
\begin{aligned}
u'&=p,\\
u''&=\frac1{\pi s},\\
u'''&=\frac{t}{\pi s^3},\\
u^{(4)}&=\frac{1+2t^2}{\pi s^5},\\
u^{(5)}&=\frac{3t(3+2t^2)}{\pi s^7},\\
u^{(6)}&=\frac{3(3+24t^2+8t^4)}{\pi s^9}.
\end{aligned}
\]

Faà di Bruno gives

\[
K_{32}^{(6)}
=F'u^{(6)}+F''B_{6,2}+F'''B_{6,3}
 +F^{(4)}B_{6,4}+F^{(5)}B_{6,5}+F^{(6)}(u')^6,
\]

where

\[
\begin{aligned}
B_{6,2}&=6u'u^{(5)}+15u''u^{(4)}+10(u''')^2,\\
B_{6,3}&=15(u')^2u^{(4)}+60u'u''u'''+15(u'')^3,\\
B_{6,4}&=20(u')^3u'''+45(u')^2(u'')^2,\\
B_{6,5}&=15(u')^4u''.
\end{aligned}
\]

All derivatives of `F` through order six are positive. If `t=-cos(phi)`, `0<phi<pi/2`, then

\[
B_{6,3}=\frac{15}{\pi^3s^5}
\left(2(\phi\cos\phi-\sin\phi)^2+\phi^2-\sin^2\phi\right)>0,
\]

and

\[
B_{6,4}=\frac{5p^2}{\pi^2s^3}
\left(9\sin\phi-4\phi\cos\phi\right)>0,
\]

using `sin(phi)>=phi cos(phi)`. The final two Bell terms are positive as well.

Only `B_{6,2}` can be negative. A directed-rounding recurrence through the 31 outer compositions certifies

\[
0\le \frac{F''(x)}{F'(x)}
\le2.398586389549085<3
\qquad(0\le x\le0.319).
\]

For `t<=0`, `u(t)<=1/pi<0.319`. A second directed certificate, on 20 rational subintervals of `[-1,0]`, proves

\[
u^{(6)}+3B_{6,2}>0.
\]

The certified transformed lower margin is `8.14928622573927`. Thus, when `B_{6,2}<0`,

\[
u^{(6)}+\frac{F''}{F'}B_{6,2}
\ge u^{(6)}+3B_{6,2}>0.
\]

This completes the lemma.

## 5. Feasible primal optimizer

Let `h_*` be the degree-five Hermite interpolant at the three exact algebraic nodes. The generalized Hermite remainder formula gives, for each noncontact `t` in `(-1,1)`,

\[
K_{32}(t)-h_*(t)
=\frac{K_{32}^{(6)}(\xi_t)}{6!}
  \prod_{j=1}^3(t-t_j)^2>0.
\]

Continuity extends the inequality to the endpoints. Equality occurs exactly at the three contact nodes.

A Krawczyk-style interval linear-system certificate encloses the normalized Gegenbauer coefficients. The nonconstant lower endpoints are

- `0.00279647306154118416616586023526018`;
- `0.00243629527371522242447068060976310`;
- `0.00180373485519710060891233424000157`;
- `0.00103172848676742614815821374777678`;
- `0.000179898923463644585494486989098646`.

The interval contraction norm is below `1.60e-72`; therefore all five coefficients are rigorously positive.

## 6. Primal-dual equality and uniqueness

Let

\[
\mu=\sum_j\lambda_j\delta_{t_j},
\qquad
\mu_\ell=\int G_\ell\,d\mu.
\]

For every admissible `h`,

\[
\Phi(h)
\le\frac1N+\int K_{32}\,d\mu,
\]

because `mu_0=q_0`, `mu_l>=q_l` for `l>=1`, all nonconstant coefficients are nonnegative, and `h<=K_32`.

For `h_*`, Gaussian exactness through degree five and contact at every support point give

\[
\Phi(h_*)
=\frac1N+\int h_*\,d\mu
=\frac1N+\int K_{32}\,d\mu.
\]

Hence `h_*` attains the dual bound and is globally optimal.

For any optimizer `h`, the nonnegative dual gap decomposes as

\[
\left(\frac1N+\int K_{32}\,d\mu\right)-\Phi(h)
=
\int(K_{32}-h)\,d\mu
+
\sum_{\ell\ge1}(\mu_\ell-q_\ell)c_\ell.
\]

Strict reduced-cost negativity gives `mu_l-q_l>0` for every `l>=6`, so every optimal higher coefficient is zero. Positive dual masses force contact at all three interior nodes; a nonnegative differentiable residual that vanishes at an interior point also has zero derivative there. The six Hermite conditions determine a unique degree-five polynomial. Therefore `h=h_*`.

## 7. Reproducibility and trust base

Proof artifacts:

- `prove_t16_all_degree.py` and `T16_ALL_DEGREE_CERTIFICATE.json`;
- `prove_t16_primal_dual.py` and `T16_PRIMAL_DUAL_CERTIFICATE.json`;
- `t16_independent_cpp_audit.cpp` and `T16_CPP_INDEPENDENT_AUDIT.json`.

The proof-critical interval code uses exact Python integers/Fractions and explicitly directed `decimal`/libmpdec operations. `mpmath` only proposes an approximate inverse for the Krawczyk certificate; the resulting rational matrix is validated by the contraction inequality, so its numerical correctness is not assumed.

The C++ audit uses Boost multiprecision exact integers and does not import the Python proof implementation.

This is a computer-assisted proof, not a proof-assistant formalization. Human review of the Hermite-remainder and Faà di Bruno argument is still required before publication.
