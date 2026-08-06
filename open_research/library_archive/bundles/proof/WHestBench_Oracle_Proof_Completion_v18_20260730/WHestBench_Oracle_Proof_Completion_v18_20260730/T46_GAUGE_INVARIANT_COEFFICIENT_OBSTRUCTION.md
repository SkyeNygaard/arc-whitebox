# T46 — Gauge-invariant coefficient obstruction

**Status:** exact deterministic representation theorem. It closes orientation-blind coefficient architectures, not the full WHestBench runtime problem.

## Motivation

A signed correction dictionary is often represented by oriented columns. Replacing a column by its negative and flipping the corresponding coefficient leaves the physical correction unchanged. Column signs are therefore a representation gauge unless the construction supplies a canonical physical orientation.

## General theorem

Let a compact group `G` act on a feature space `X` and through a unitary representation `R_g` on coefficient space `K`. Let

\[
a=h(x)
\]

be a coefficient policy. Assume:

1. the runtime representation is invariant: `g x = x` for every `g in G`;
2. the coefficient rule is representation-consistent/equivariant:

\[
h(gx)=R_g h(x).
\]

Then

\[
h(x)\in K^G,
\]

where `K^G` is the invariant coefficient subspace.

### Proof

Since `gx=x`, equivariance gives `h(x)=R_g h(x)` for every `g`. This is exactly membership in `K^G`.

## Independent sign-flip corollary

Let `G={diag(s_1,...,s_m):s_j in {-1,+1}}` act on an `m`-coefficient dictionary by independent column reorientation. Its invariant subspace is `{0}`. Therefore any exactly gauge-invariant feature representation combined with an exactly gauge-equivariant coefficient policy must output

\[
\boxed{a=0}.
\]

An orientation-blind architecture cannot produce nonzero well-defined signed coefficients.

## Global-sign two-point minimax corollary

Suppose a feature map satisfies

\[
\Phi(C)=\Phi(-C)
\]

for a candidate bundle `C`, while a scalar signed target satisfies `y(-C)=-y(C)`. Any deterministic predictor `p(\Phi(C))` makes the same prediction at the two orientations, and

\[
\max\{(p-y)^2,(p+y)^2\}\ge y^2.
\]

Thus no predictor based only on the quotient representation can be uniformly accurate over an orientation-closed class.

This is a deterministic minimax statement; it requires no probability distribution.

## Approximate theorem

Let `Pi_G` be projection onto the invariant coefficient subspace. If `h` is `L`-Lipschitz, and define

\[
\eta_X(x)=\int_G\|gx-x\|\,dg,
\]

and policy equivariance defect

\[
\eta_h(x)=
\int_G\|h(gx)-R_gh(x)\|\,dg,
\]

then

\[
\boxed{
\|h(x)-\Pi_Gh(x)\|
\le
L\eta_X(x)+\eta_h(x).
}
\]

For the full sign group, `Pi_G=0`, so the coefficient norm itself is bounded by the symmetry-breaking and equivariance defects.

## M153 representation audit

The exact nine T4 features are:

1. `cos(c17,p2)`;
2. `cos(c17,p4)`;
3. `cos(p2,p4)`;
4. `||p2||/||c17||`;
5. `||p4||/||c17||`;
6. minimum successive nested cosine;
7. maximum leave-one-basis angle sine;
8. `cos(p32,p128)`;
9. `||p32||/||p128||`.

They are unchanged by a simultaneous sign reversal of all represented correction trajectories. The signed targets `cos(c17,e)`, `cos(p2,e)`, and `cos(p4,e)` reverse under candidate reversal with the scored error fixed.

This proves a **representation-level** quotient obstruction: the feature vector contains no explicit global orientation anchor. It does not prove that the actual network distribution is closed under that reversal, because the candidates may have a canonical physical orientation determined by their construction.

## Constructive escape route

A legal coefficient system should expose orientation-odd quantities, for example:

- signed coefficients in a downstream Jacobian singular basis whose vector signs are fixed canonically;
- contractions with a canonical final-layer adjoint direction;
- signed preactivation-margin or gate-crossing contractions;
- a network-derived reference vector with an explicit deterministic sign convention.

Any such feature must be frozen, grouped by base network, and compared against matched constant coefficients. Merely adding more norms, Gram matrices, disagreement magnitudes, or condition numbers remains inside the quotient class.
