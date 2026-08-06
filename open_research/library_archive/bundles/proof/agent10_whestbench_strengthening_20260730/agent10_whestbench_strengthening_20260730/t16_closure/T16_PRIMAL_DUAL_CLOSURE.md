# T16 primal–dual closure: all-degree optimality of the degree-5 auxiliary

**Project:** WHestBench / ARC White-Box Estimation Challenge 2026  
**Date:** 2026-07-30  
**Status:** New proof package; proved under the stated interval-arithmetic trust base, pending an independent hostile audit.

## Result

Let

\[
K_0(t)=t,\qquad K_{r+1}(t)=\kappa(K_r(t)),\qquad
\kappa(t)=\frac{\sqrt{1-t^2}+(\pi-\arccos t)t}{\pi},
\]

and take `d=256`, depth `32`, and `N=66,048`. Write
\(G_\ell= C_\ell^{127}/C_\ell^{127}(1)\), and define

\[
q_0=1-\frac1N,\qquad q_\ell=-\frac1N\quad(\ell\ge1).
\]

Consider the all-degree auxiliary linear program

\[
\sup \left\{\sum_{\ell\ge0}q_\ell c_\ell:
 h(t)=\sum_{\ell\ge0}c_\ell G_\ell(t)\le K_{32}(t),\;
 c_\ell\ge0\ (\ell\ge1)\right\},
\]

where `c0` is unrestricted. The previously certified three-node dual measure is supported on the roots \(t_1<t_2<t_3\) of

\[
22102t^3+21930t^2-87t-85=0.
\]

Define \(h_*\) to be the unique polynomial of degree at most five satisfying

\[
h_*(t_j)=K_{32}(t_j),\qquad h_*'(t_j)=K_{32}'(t_j),
\qquad j=1,2,3.
\]

**Theorem.** The polynomial \(h_*\) is feasible, all five nonconstant normalized-Gegenbauer coefficients are strictly positive, and \(h_*\) is the unique optimizer of the unrestricted all-degree auxiliary LP.

The certified coefficient intervals are:

| coefficient | certified interval |
|---|---|
| \(c_0\) | `[0.9747299751309444413666593085802870785923869068234348747228323827800535066124142510656, 0.9747299751309444413666593085802870785923869068234348747228323827800535066124336317342]` |
| \(c_1\) | `[0.002796473061541184166165860235260182130169385363343326868046738754497541653959875743602, 0.002796473061541184166165860235260182130169385363343326868046738754497541654100495549543]` |
| \(c_2\) | `[0.00243629527371522242447068060976310829567257873525442749320203262172687045156449927791, 0.002436295273715222424470680609763108295672578735254427493202032621726870453829685505556]` |
| \(c_3\) | `[0.001803734855197100608912334240001576722030711898741029650192665061694266139150328014038, 0.001803734855197100608912334240001576722030711898741029650192665061694266149066475486634]` |
| \(c_4\) | `[0.001031728486767426148158213747776785267142038328379984234160947579169374032537656007344, 0.001031728486767426148158213747776785267142038328379984234160947579169374098147223104555]` |
| \(c_5\) | `[0.0001798989234636445854944869890986466385304715868303939932215788517516604354397283856907, 0.0001798989234636445854944869890986466385304715868303939932215788517516604924264123575439]` |

## 1. Dual feasibility was already closed

The existing T16 package proves that the positive three-node measure \(\mu=\sum_j\lambda_j\delta_{t_j}\) satisfies

\[
\int G_\ell\,d\mu=q_\ell\quad(0\le\ell\le5),
\]

and

\[
r_\ell:=q_\ell-\int G_\ell\,d\mu<0\quad\text{for every }\ell\ge6.
\]

Degrees `6..14,658` are handled by exact integer arithmetic, and all larger degrees by the normalized-Gegenbauer tail bound. Thus only primal feasibility and complementary slackness were missing.

## 2. Sixth derivative of the kernel is positive

Write

\[
K_{32}=F\circ\kappa,\qquad F=\kappa^{\circ31}.
\]

For `u in (0,1)`, every derivative \(\kappa^{(k)}(u)\), `1<=k<=6`, is nonnegative, and the first, second, fourth, and sixth are strictly positive. Composition and Faà di Bruno therefore give \(F^{(k)}(u)>0\), `1<=k<=6`.

The sixth derivative decomposes as

\[
K_{32}^{(6)}(t)=\sum_{k=1}^{6}F^{(k)}(\kappa(t))B_{6,k}(t),
\]

where, with \(u_j=\kappa^{(j)}(t)\),

\[
\begin{aligned}
B_{6,1}&=u_6,\\
B_{6,2}&=6u_1u_5+15u_2u_4+10u_3^2,\\
B_{6,3}&=15u_1^2u_4+60u_1u_2u_3+15u_2^3,\\
B_{6,4}&=20u_1^3u_3+45u_1^2u_2^2,\\
B_{6,5}&=15u_1^4u_2,\qquad B_{6,6}=u_1^6.
\end{aligned}
\]

For `t>=0`, all terms are nonnegative and \(B_{6,1}>0\). It remains to treat `t<0`.

### 2.1 The nonnegative Bell terms for `t<0`

Set

\[
t=-\cos\phi,\quad s=\sin\phi,\quad c=\cos\phi,
\quad 0<\phi<\frac\pi2.
\]

Then \(u_1=\phi/\pi\), and direct substitution gives:

* \(B_{6,5}>0\), \(B_{6,6}>0\).
* \(B_{6,4}\ge0\) because its only nontrivial factor is
  \(9s-4\phi c\ge5s>0\), using \(s\ge\phi c\).
* For \(B_{6,3}\), after removing a positive factor, the numerator is

  \[
  Q=\phi^2(1+2c^2)-4\phi cs+s^2.
  \]

  With \(r=s/\phi\le1\),

  \[
  Q/\phi^2=(r-2c)^2+1-2c^2.
  \]

  If \(c^2\le1/2\), this is immediate. If \(c^2\ge1/2\), then
  \(2c\ge1\ge r\), hence

  \[
  (r-2c)^2+1-2c^2\ge(1-2c)^2+1-2c^2=2(c-1)^2\ge0.
  \]

### 2.2 The only potentially negative Bell term

The certificate proves

\[
B_{6,2}(t)\ge-\frac14\,\kappa^{(6)}(t),\qquad -1<t<0.
\]

The proof uses

\[
\phi\cot\phi\le\frac{1+2\cos\phi}{3}.
\]

An elementary proof is obtained by differentiating

\[
g(\phi)=\sin\phi(1+2\cos\phi)-3\phi\cos\phi.
\]

Writing `a=phi/2` gives

\[
g'(\phi)=4\sin a\,[3a\cos a-\sin(3a)].
\]

The bracket is nonnegative because its derivative equals

\[
3\sin a\,[2\sin(2a)-a]\ge0,
\]

using concavity of sine on `[0,pi/2]` to obtain
\(\sin(2a)\ge4a/\pi>a/2\).

The remaining algebra reduces to

\[
3D(c)>4(1-c^2)^{3/2}R(c),
\]

where

\[
D(c)=24c^4+72c^2+9,
\qquad R(c)=24c^3-28c^2+36c+3.
\]

`R` has positive Bernstein coefficients `[3, 15, 53/3, 35]` on `[0,1]`. Squaring the positive sides yields the exact polynomial inequality

\[
9D(c)^2-16(1-c^2)^3R(c)^2>0.
\]

Exact rational Bernstein coefficients are positive on each of
`[0,1/4]`, `[1/4,1/2]`, `[1/2,3/4]`, and `[3/4,1]`; their respective minimum coefficients are

\[
\frac{1267463}{45056},\quad
\frac{2864025}{16384},\quad
\frac{10719}{2},\quad
\frac{467619201}{16384}.
\]

### 2.3 Bound on the outer derivative ratio

For \(F=\kappa^{\circ31}\), interval recurrence certifies

\[
\frac{F''(u)}{F'(u)}<2.226033<\frac94,
\qquad 0\le u\le\frac1\pi.
\]

The recurrence is

\[
\begin{aligned}
x_{n+1}&=\kappa(x_n),\\
p_{n+1}&=\kappa'(x_n)p_n,\\
r_{n+1}&=r_n+\frac{\kappa''(x_n)}{\kappa'(x_n)}p_n,
\end{aligned}
\]

with `x0=u`, `p0=1`, `r0=0`, so that `r31=F''/F'`. Four interval boxes cover `[0,1/pi]`; the largest certified upper endpoint is

`2.226032569855077694541983999275615705932171269702189314407833815053812517253956751291`.

Combining the bounds,

\[
\begin{aligned}
F'B_{6,1}+F''B_{6,2}
&\ge F'\kappa^{(6)}\left(1-\frac14\frac{F''}{F'}\right)\\
&>F'\kappa^{(6)}\left(1-\frac9{16}\right)>0.
\end{aligned}
\]

All remaining terms are nonnegative, so

\[
\boxed{K_{32}^{(6)}(t)>0\quad\text{for every }-1<t<1.}
\]

## 3. Hermite remainder proves primal feasibility

For every `t` in `(-1,1)`, the generalized Hermite remainder formula gives some \(\xi\) between `t` and the interpolation nodes such that

\[
K_{32}(t)-h_*(t)
=\frac{K_{32}^{(6)}(\xi)}{6!}
 \prod_{j=1}^{3}(t-t_j)^2\ge0.
\]

Continuity extends the inequality to the endpoints. Therefore \(h_*\le K_{32}\) on `[-1,1]`.

## 4. Exact complementary slackness

Let \(h_*=\sum_{\ell=0}^{5}c_\ell G_\ell\). Moment matching and contact give

\[
\begin{aligned}
\sum_{\ell=0}^{5}q_\ell c_\ell
&=\sum_{j=1}^{3}\lambda_j
  \sum_{\ell=0}^{5}c_\ell G_\ell(t_j)\\
&=\sum_{j=1}^{3}\lambda_j h_*(t_j)\\
&=\sum_{j=1}^{3}\lambda_j K_{32}(t_j),
\end{aligned}
\]

which is exactly equality between primal and dual objectives. Weak duality then proves optimality.

Strict reduced costs for every degree at least six force all higher coefficients to vanish in any optimizer. Any optimal degree-5 polynomial must contact the kernel at each positive-mass dual node; because the contact points are interior minima of `K32-h`, first derivatives also match. The six Hermite conditions determine a unique polynomial. Hence \(h_*\) is the unique optimizer.

## 5. What this closes—and what it does not

This closes the T16 statement:

> The KKT-selected degree-5 auxiliary is exactly optimal for the unrestricted all-degree Delsarte auxiliary LP at `d=256`, depth `32`, and `N=66,048`.

It does **not** prove exact optimality of Kerdock cubature. The auxiliary lower bound remains strictly below Kerdock’s kernel energy, so the certified `0.0233655%` near-optimality gap remains one-sided and nonzero at the certificate level.

It also does not address finite width, arbitrary signed-node rules, network adaptation, or nonlinear estimators.

## 6. Artifacts and trust base

* `close_t16_primal_dual.py`
* `T16_PRIMAL_DUAL_CLOSURE_CERTIFICATE.json`
* prior `prove_t16_all_degree.py`
* prior `T16_ALL_DEGREE_CERTIFICATE.json`

The new interval portions use `mpmath.iv` 1.3.0 at 80 decimal digits. Root isolation and Bernstein certificates use exact `Fraction`/SymPy rational arithmetic. Before publication, run a hostile reproduction using a second interval stack or port the four scalar interval boxes to the existing directed-Decimal proof framework.
