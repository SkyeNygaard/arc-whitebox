# T39–T40 — Symmetry-limited information and residual spectral recertification

## T39 — invariant-information projection

Let a finite or compact group act measure-preservingly on the problem instance. Let the error transform under a unitary representation `rho`, and let runtime observation `X` be invariant under the group. Then

\[
\mathbb E[e\mid X]
=\mathbb E[\Pi_Ge\mid X]\in H^G,
\]

where `Pi_G` projects onto the invariant subspace. Hence the total MSE value of every `X`-measurable correction is bounded by

\[
\mathbb E\|\Pi_Ge\|^2.
\]

If the invariant subspace is zero, no `X`-measurable correction helps.

This theorem is exact, but applying it to WHestBench requires proving that the actual observation map is invariant under a measure-preserving action that removes the relevant signed error component. Feature norms and Gram matrices are not enough to establish that empirical condition.

## T40 — residual spectral multiplier

Let an isotropic random field have harmonic variances `q_l`, and let a deterministic bounded rotation-equivariant linear surrogate act by scalar `tau_l` on harmonic degree `l`. The residual `h=(I-T)f` has harmonic variances

\[
q_l^{res}=|1-\tau_l|^2q_l.
\]

Consequences:

1. A filter acting only on degrees through five leaves complete-Kerdock risk unchanged.
2. A live-degree surrogate must be recertified using its residual kernel.
3. On the fixed MUB line universe, complete-basis extremality persists if the residual association values retain the three T37 signs.
4. No such conclusion follows for nonlinear, non-equivariant, node-dependent, or candidate-dependent surrogates without a new residual analysis.
