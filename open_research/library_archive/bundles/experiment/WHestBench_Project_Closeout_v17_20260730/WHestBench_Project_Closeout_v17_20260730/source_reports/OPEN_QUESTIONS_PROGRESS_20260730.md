# Remaining Open Questions — Proof Progress

**Date:** 2026-07-30  
**Project:** WHestBench / Kerdock neural cubature  
**Purpose:** Continue the proof/disproof program after `STRONGER_CLAIMS_PROOF_ATTEMPTS_20260730.md`.

## Executive result

Three substantial advances are obtained.

1. **Finite-width fixed-support optimality is resolved.** The exact T27 complete-basis theorem extends to every finite width and depth for the standard Gaussian first-layer random-network ensemble, under a mild nondegeneracy condition. It does not rely on convergence to the NNGP kernel.
2. **The common-bias theorem generalizes to group-invariant information classes.** Runtime observations can recover only the group-invariant component of the error. Orientation-blind candidate diagnostics have zero correction value under conditional global-sign symmetry.
3. **Residual recertification becomes spectral under equivariance.** An equivariant linear surrogate multiplies each harmonic variance by a known squared residual multiplier. Low-degree filters cannot change complete-Kerdock risk; live-degree filters can be recertified by recomputing the residual association values.

The largest unresolved theorem is now narrower: **arbitrary-node finite-width near-optimality**, i.e. a finite-width analogue of T22. The finite-width fixed-Kerdock-support problem is no longer open.

---

# 1. Exact finite-width kernel structure

## 1.1 Setup

Let the input be a unit vector `x in S^(d-1)`. Let the first-layer matrix have `m` independent standard-Gaussian rows. Let `Z` denote every later-layer weight and every other source of network randomness, independent of the first-layer matrix.

The architecture may have any finite width and depth after the first layer. Its output may lie in a real Hilbert space `H`. The only structural requirement is the ordinary feed-forward one:

\[
Y(x)=F_Z(W^{(1)}x),
\]

for a square-integrable measurable map `F_Z : R^m -> H` that is the same for every input.

For unit inputs with inner product `t`, define the finite-width ensemble kernel

\[
K_m(t)=\mathbb E\langle Y(x),Y(y)\rangle_H,
\qquad x^Ty=t.
\]

This covers a standard finite-width bias-free ReLU network before or after exact radial reduction. A radial second-moment factor only multiplies the kernel by a positive constant.

## Theorem 1 — finite-width Gaussian-noise expansion

For every finite width and depth,

\[
K_m(t)=\sum_{n=0}^{\infty} a_n t^n,
\qquad a_n\ge 0,
\qquad -1\le t\le1.
\]

The series converges in the usual `L2`/Mehler sense and pointwise wherever the covariance is finite.

### Proof

Choose coordinates

\[
x=e_1,
\qquad
y=t e_1+\sqrt{1-t^2}e_2.
\]

Across the `m` first-layer neurons,

\[
G=W^{(1)}x,
\qquad
G'=W^{(1)}y
\]

are standard Gaussian vectors with coordinatewise correlation `t`. Equivalently,

\[
G'=tG+\sqrt{1-t^2}\,\widetilde G,
\]

where `G` and `G_tilde` are independent standard Gaussian vectors.

Condition on `Z`. Expand the Hilbert-valued function `F_Z` in the orthonormal multivariate Hermite basis:

\[
F_Z(g)=\sum_{\alpha\in\mathbb N^m} c_\alpha(Z)H_\alpha(g),
\qquad c_\alpha(Z)\in H.
\]

The Gaussian noise-stability identity gives

\[
\mathbb E[\langle F_Z(G),F_Z(G')\rangle_H\mid Z]
=
\sum_\alpha \|c_\alpha(Z)\|_H^2t^{|\alpha|}.
\]

Average over `Z` and group by total degree:

\[
a_n=
\mathbb E_Z\sum_{|\alpha|=n}\|c_\alpha(Z)\|_H^2\ge0.
\]

No infinite-width approximation appears. `square`

## Corollary 1.1 — exact finite-width antipodal-line kernel

Define the line-symmetrized output

\[
S(x)=\frac{Y(x)+Y(-x)}2.
\]

Its kernel is

\[
\overline K_m(t)
=
\mathbb E\langle S(x),S(y)\rangle
=
\frac{K_m(t)+K_m(-t)}2
=
\sum_{r=0}^{\infty}a_{2r}t^{2r}.
\]

Hence `Kbar_m(t)` is nondecreasing in `|t|` on `[0,1]`. It is strictly increasing if the antipodal even component is not almost surely constant.

This is stronger than a finite-width convergence statement: it identifies the exact shape class of the ensemble line kernel at every width.

---

# 2. Finite-width Kerdock-support theorem

Let the line universe be a union of mutually unbiased orthonormal bases in `R^d`. On this universe, the line kernel has three values

\[
A=\overline K_m(1),
\qquad
O=\overline K_m(0),
\qquad
C=\overline K_m(1/\sqrt d).
\]

Using the expansion above,

\[
A-O=\sum_{r\ge1}a_{2r},
\]

\[
O-C=-\sum_{r\ge1}a_{2r}d^{-r},
\]

and

\[
(A-O)+d(O-C)
=
\sum_{r\ge2}a_{2r}(1-d^{1-r}).
\]

Therefore:

- `A-O > 0` whenever the even output is nonconstant;
- `O-C < 0` under the same condition;
- `(A-O)+d(O-C) >= 0` automatically;
- the last inequality is strict whenever some even Hermite degree at least four is present.

A nonconstant even piecewise-linear ReLU-network output cannot consist only of Hermite degrees zero and two: such a function would be a global quadratic polynomial, whereas a nonconstant finite ReLU network is piecewise linear. Thus a nondegenerate ReLU ensemble has a positive coefficient at some even degree at least four.

## Theorem 2 — exact finite-width T27 extension

For every finite width and finite depth of a nondegenerate standard Gaussian-weight ReLU network, consider static linear cubature rules supported on symmetrized antipodal lines from a fixed union of real mutually unbiased bases. Permit arbitrary real line weights summing to one.

At every line budget `P`, the minimum **finite-width ensemble MSE** is attained by

1. `floor(P/d)` complete orthonormal bases;
2. at most one additional partial basis;
3. equal positive weights within every active basis;
4. positive basis masses proportional to
   \[
   \frac1{(O-C)+(A-O)/r_b}.
   \]

### Proof

The finite-width line kernel is projectively rotation invariant and has exactly the three association values on the MUB universe. Its risk therefore has the exact decomposition

\[
R(w)=\text{constant}
+(O-C)\sum_bS_b^2
+(A-O)\sum_{b,i}w_{bi}^2.
\]

The coefficient identities above establish the strict sign conditions required by the general MUB support-extremality theorem. Apply that theorem. `square`

## Scope

This resolves the finite-width qualification for **T27**, the fixed Kerdock-line-universe theorem.

It does **not** prove a finite-width analogue of T22, because T22 compares Kerdock against arbitrary nodes. The Hermite power-series structure does not by itself provide the required arbitrary-node Delsarte lower certificate.

## Numerical sanity check

A coupled empirical-covariance Markov simulation at width 256, depth 32, dimension 256, with 6,000 simulations gave

\[
A-O\approx 2.3069\times10^{-2},
\]

\[
O-C\approx-3.7541\times10^{-5},
\]

\[
(A-O)+256(O-C)\approx1.3458\times10^{-2}.
\]

The coupled standard errors were approximately `2.30e-3`, `7.18e-6`, and `1.75e-3`. These numbers are only a sanity check; the signs follow analytically from Theorem 1.

---

# 3. Group-invariant observability

The additive common-bias model is one instance of a more general obstruction.

## Theorem 3 — invariant-information projection theorem

Let a finite or compact group `G` act measure-preservingly on the random problem instance `omega`. Let `rho(g)` be a unitary representation of `G` on the error space `H`. Assume

\[
e(g\omega)=\rho(g)e(\omega).
\]

Let the runtime observation `X` be group invariant:

\[
X(g\omega)=X(\omega).
\]

Let

\[
H^G=\{v:\rho(g)v=v\text{ for every }g\}
\]

and let `Pi_G` be orthogonal projection onto `H^G`. Then

\[
\mathbb E[e\mid X]
=
\mathbb E[\Pi_Ge\mid X]
\in H^G.
\]

Consequently, the total value of every correction measurable from `X` is bounded by

\[
V(\sigma(X);e)
=
\mathbb E\|\mathbb E[e\mid X]\|^2
\le
\mathbb E\|\Pi_Ge\|^2.
\]

If the representation has no invariant component, then

\[
\mathbb E[e\mid X]=0,
\]

and no `X`-measurable correction can reduce MSE.

### Proof

The joint law is group invariant. Because `X` is unchanged by the action, the conditional law of `e` given `X` is invariant under every `rho(g)`. Its conditional mean must therefore satisfy

\[
\rho(g)\mathbb E[e\mid X]=\mathbb E[e\mid X]
\]

for every `g`, so it lies in `H^G`. Averaging over the group gives the projection identity. Jensen and orthogonal projection give the value bound. `square`

## Corollary 3.1 — orientation-blind candidate diagnostics

Let `U=(u_1,...,u_k)` be candidate correction directions. Suppose runtime information `Gcal` contains only quantities invariant under the common sign reversal

\[
U\mapsto-U,
\]

such as norms, pairwise cosines, Gram matrices, nested disagreement magnitudes, or condition numbers. Assume the conditional law given `Gcal` is invariant under this sign reversal while the target error `e` is unchanged.

Then

\[
\mathbb E[U^*e\mid\mathcal G]=0.
\]

For every coefficient vector `alpha(Gcal)`,

\[
\mathbb E\|e-U\alpha\|^2
=
\mathbb E\|e\|^2+
\mathbb E\|U\alpha\|^2.
\]

Every nonzero orientation-blind correction strictly worsens expected risk.

### Interpretation

This formalizes the statement that magnitude, internal agreement, and candidate-candidate geometry do not determine orientation relative to the unknown target error. It is broader than the common additive-bias model.

### Remaining empirical condition

The theorem does not assert that the actual WHestBench conditional distribution is exactly sign symmetric. That is now the precise empirical hypothesis to test. A useful audit is:

1. freeze the permitted sign-invariant feature sigma-field;
2. estimate the conditional signed alignment `E[U^*e | features]` with grouped cross-fitting;
3. compare its attainable value with a matched constant and with the full target-labeled oracle;
4. report a confidence bound on the observability ratio.

The existing T4 result—large per-rotation oracle value but failure of stable across-rotation coefficients—is consistent with a small invariant component, but is not by itself a proof of conditional symmetry.

---

# 4. Equivariant residual-kernel recertification

## Setup

Let `f_theta` be an isotropic square-integrable random field on the sphere, expanded as

\[
f_\theta=\sum_{\ell=0}^\infty\sum_{m=1}^{N_\ell}
\xi_{\ell m}(\theta)Y_{\ell m}.
\]

Assume its ensemble covariance is diagonal by harmonic degree:

\[
\mathbb E[\xi_{\ell m}\xi_{\ell' m'}]
=q_\ell\,\mathbf1_{\ell=\ell'}\mathbf1_{m=m'}.
\]

Let `T` be a deterministic bounded linear operator commuting with every rotation. By irreducibility of each spherical-harmonic space,

\[
T|_{\mathcal H_\ell}=\tau_\ell I.
\]

Use the surrogate `g_theta=T f_theta` and residual

\[
h_\theta=(I-T)f_\theta.
\]

## Theorem 4 — residual spectral multiplier

The residual ensemble kernel is isotropic with harmonic coefficients

\[
q_\ell^{\rm res}=|1-\tau_\ell|^2q_\ell.
\]

### Proof

The residual harmonic coefficient is `(1-tau_l) xi_lm`. Square and take expectation. `square`

## Corollary 4.1 — low-degree equivariant controls cannot help complete Kerdock

Complete Kerdock is exact through degree five. Its ensemble MSE depends only on harmonic degrees at least six. Therefore every equivariant filter satisfying

\[
\tau_\ell=0\quad\text{for all }\ell\ge6
\]

leaves complete-Kerdock risk unchanged, regardless of what it does to degrees zero through five.

This recovers and generalizes the low-degree control-nullspace result.

## Corollary 4.2 — exact recertification recipe

For a live-degree residual filter:

1. compute `q_l^res`;
2. reconstruct the residual zonal kernel;
3. evaluate its MUB association values `A_res,O_res,C_res`;
4. if
   \[
   A_{res}-O_{res}>0,
   \quad
   O_{res}-C_{res}<0,
   \quad
   (A_{res}-O_{res})+d(O_{res}-C_{res})>0,
   \]
   then complete-basis support extremality persists exactly for the residual;
5. otherwise the optimal support allocation can change and must be reoptimized.

## Limitation

A general network-dependent white-box surrogate need not be a deterministic rotation-equivariant linear filter. It may mix harmonic degrees, break isotropy, or induce a candidate-dependent residual law. The theorem provides a rigorous recertification path for an important tractable class, not all residual methods.

---

# 5. What remains genuinely open

## Open A — arbitrary-node finite-width near-optimality

A finite-width T22 analogue remains open. Required ingredients include one of:

- a finite-width Delsarte minorant for the exact ensemble kernel;
- a structure-aware comparison between the finite-width and limiting harmonic coefficients strong enough to preserve the certified sandwich;
- or a direct finite-width linear/semidefinite programming certificate.

Uniform kernel perturbation is far too crude because the infinite-width certified additive gap is only about `5.685e-11`.

## Open B — verify the actual information symmetry

The group theorem converts the broad impossibility question into a concrete empirical/theoretical one:

> Which group-noninvariant signed component, if any, is contained in the legal low-cost runtime observables?

An exact impossibility theorem for a real WHestBench selector class requires defining its observation map and proving that map invariant under a measure-preserving action that removes the relevant error component.

## Open C — constructive live-degree surrogate

The residual spectral theorem shows what a useful equivariant surrogate must do: attenuate degrees at least six without spending more compute than the saved sampling error. Constructing such a legal, exactly integrable, network-dependent surrogate remains open.

## Open D — finite-width arbitrary-node signed rules

Neither T22 nor the new finite-width T27 extension covers arbitrary new nodes with signed weights. The existing negative-mass stability lemma remains too weak to close this class.

---

# Recommended claim upgrades

## New main theorem candidate

> For any finite-width Gaussian first-layer random network, the ensemble kernel is a Gaussian noise-stability kernel with nonnegative power-series coefficients. Consequently, for every finite width and depth of a nondegenerate ReLU network, arbitrary real weighting and deletion within a fixed mutually-unbiased-basis line universe are optimized by complete bases plus at most one partial basis.

## New information-theoretic claim

> A runtime information class can recover only the component of integration error invariant under every symmetry that its observations discard. Under global orientation blindness, candidate norms and mutual agreement have exactly zero correction value.

## Revised finite-width limitation

Do not say that all mathematical results are infinite-width. The correct split is:

- **T22 arbitrary-node near-optimality:** infinite width only;
- **T27 fixed Kerdock-line support optimality:** exact at finite width under the standard random-network ensemble assumptions;
- **finite-width absolute Kerdock MSE:** still requires finite-width kernel evaluation.
