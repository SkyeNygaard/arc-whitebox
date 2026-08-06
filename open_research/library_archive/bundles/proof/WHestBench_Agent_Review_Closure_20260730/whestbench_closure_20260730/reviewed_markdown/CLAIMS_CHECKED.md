# Claims checked — Agent 4 / T16

## Primary claim

For the target WHestBench auxiliary-LP dual at

- dimension `d = 256`;
- Gegenbauer parameter `alpha = 127`;
- point budget `N = 66,048`;

all unused harmonic reduced costs

\[
r_\ell=q_\ell-\sum_{j=1}^3\lambda_jG_\ell(t_j)
\]

are strictly negative for every integer `ell >= 6`, where `G_ell(1)=1`,
`q_0=1-1/N`, and `q_ell=-1/N` for `ell>=1`.

**Verdict: PROVED.**

The proof is exact through a finite cutoff and analytic afterward. It does not rely on the previous floating-point scan through degree one million.

## Exact dual reconstruction

The six degree-0-through-5 moment constraints imply exact monomial moments

\[
(m_0,\dots,m_5)=\frac1{66048}(66047,-1,257,-1,2,-1).
\]

The monic cubic orthogonal to `1,t,t^2` is

\[
p(t)=t^3+\frac{255}{257}t^2-\frac{87}{22102}t-\frac{85}{22102},
\]

or equivalently

\[
P(t)=22102t^3+21930t^2-87t-85.
\]

Its three roots are the dual contacts. Exact rational sign changes isolate them in

- `[-0.992278935, -0.992278934]`;
- `[-0.062224856, -0.062224855]`;
- `[ 0.062285891,  0.062285892]`.

The corresponding Lagrange quadrature weights are rigorously enclosed by

- `[1.57406236647e-5, 1.57407507053e-5]`;
- `[0.500225801833, 0.500225820051]`;
- `[0.499743299364, 0.499743316371]`.

They are positive and sum exactly to `66047/66048`.

## Finite-degree theorem

Reducing the normalized Gegenbauer recurrence modulo `P(t)` gives a three-state exact integer recurrence. It certifies

\[
r_\ell<0\qquad(6\le \ell\le 14658).
\]

The least-negative value is exactly

\[
r_7=-\frac{2327215}{9290262647272}
   =-2.5050045282448125\times 10^{-7}.
\]

## Analytic tail theorem

For `alpha=127`, the Laplace integral representation gives, for `|t|<1`,

\[
|G_\ell(t)|\le
\frac{254!}{127!\,[\ell(1-t^2)]^{127}}.
\]

All three exact contacts satisfy `|t_j|<0.993`, hence

\[
1-t_j^2>\frac{13951}{10^6}.
\]

At `ell=14659`, exact integer comparison proves

\[
\frac{254!}{127!\,[14659(13951/10^6)]^{127}}
<\frac1{66048}.
\]

The left side decreases with `ell`. Since the masses are positive and sum to less than one,

\[
r_\ell
\le -\frac1N+\sum_j\lambda_j|G_\ell(t_j)|
<0
\qquad(\ell\ge14659).
\]

Combining the finite and tail results proves strict negativity for every `ell>=6`.

## Stronger LP-optimality wording

The result closes the **all-degree reduced-cost** part of T16. To state that a particular degree-5 primal minorant is exactly all-degree LP-optimal, the paper should also cite an exact primal-dual equality/contact certificate for that minorant. The previous safe rational minorant is shifted below contact, so reduced-cost closure and exact primal attainment should remain logically separate.
