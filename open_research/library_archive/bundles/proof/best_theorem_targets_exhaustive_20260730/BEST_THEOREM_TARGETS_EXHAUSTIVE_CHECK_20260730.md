# Best Theorem Targets — Exhaustive Check

**Date:** 2026-07-30  
**Purpose:** Push the remaining WHestBench theorem targets as far as the currently available proof artifacts permit, actively checking counterexamples and non-extensions.

## Executive result

The strongest new result is a **global lower bound for arbitrary signed, mass-one, at-most-66,048-node rules**. It does not prove signed near-optimality of Kerdock, but it proves that signed weights cannot collapse the limiting-kernel MSE to zero:

\[
R_{K_{32}}(Q)\ge 1.70117812784334\times10^{-8}
\]

for every such rule. Relative to the certified complete-Kerdock MSE, this is at least

\[
0.0699020\,R_K.
\]

Thus the theorem permits at most a `14.3058x` improvement over Kerdock, rather than an unbounded improvement. This is a genuine partial closure of the arbitrary-signed-node loophole.

A second new result certifies that the all-degree auxiliary residual

\[
q=K_{32}-h_*
\]

has nonnegative normalized-Gegenbauer coefficients in every degree and is therefore positive definite. This is structurally stronger than pointwise nonnegativity, although by itself it does not recover the positive-weight diagonal term used by T22/T16.

A third set of results turns the qualitative observability story into quantitative information inequalities: binary phase observability is at most twice the available mutual information, and finite-action oracle capture is controlled by Fano's inequality.

---

# 1. Coefficientwise positivity of the optimized residual

Let

\[
K_{32}(t)=\sum_{\ell\ge0}k_\ell G_\ell(t)
\]

be the normalized-Gegenbauer expansion in dimension 256, and let

\[
h_*(t)=\sum_{\ell=0}^5c_\ell G_\ell(t)
\]

be the certified degree-five T16 optimizer.

## Theorem 1 — positive-definite residual

Every coefficient of

\[
q(t)=K_{32}(t)-h_*(t)
\]

is nonnegative, and coefficients `0..5` are strictly positive. Hence `q` is a positive-definite zonal kernel on `S^255`.

## Proof

### Step 1: nonnegative power-series coefficients

The normalized ReLU dual activation is

\[
\kappa(t)=\frac{\sqrt{1-t^2}+(\pi-\arccos t)t}{\pi}
=\frac1\pi+\frac t2+\sum_{m\ge1}a_{2m}t^{2m},
\]

where every `a_{2m}>0`. Therefore every Maclaurin coefficient of

\[
K_{32}=\kappa^{\circ32}
\]

is nonnegative: composition and multiplication of convergent power series with nonnegative coefficients preserve coefficientwise nonnegativity.

Every monomial kernel `t^n` is positive definite, since

\[
\langle x,y\rangle^n
=
\langle x^{\otimes n},y^{\otimes n}\rangle.
\]

Consequently its normalized-Gegenbauer expansion has nonnegative coefficients. Any finite lower partial sum of Maclaurin terms therefore gives a lower bound on each `k_l`.

### Step 2: interval jet through degree 11

The verifier propagates an 80-decimal interval Taylor jet of `K_32` at zero through degree 11. It then projects the retained monomials exactly onto `G_0,...,G_6` using rational spherical moments.

The resulting lower bounds compared with the certified upper endpoints of `c_l` are:

| degree | lower bound for `k_l` from powers `<=11` | certified upper `c_l` | positive margin |
|---:|---:|---:|---:|
| 0 | 0.9747299895416970160 | 0.9747299751309444414 | `1.4410753e-8` |
| 1 | 0.0027966328995347150 | 0.0027964730615411842 | `1.5983799e-7` |
| 2 | 0.0024438109009453949 | 0.0024362952737152224 | `7.5156272e-6` |
| 3 | 0.0018364439806912280 | 0.0018037348551971006 | `3.2709125e-5` |
| 4 | 0.0015312318862667014 | 0.0010317284867674261 | `4.9950340e-4` |
| 5 | 0.0012573109211224033 | 0.0001798989234636446 | `1.0774120e-3` |

All omitted Maclaurin terms make nonnegative contributions. Thus `k_l-c_l>0` for `0<=l<=5`. For `l>=6`, `h_*` has coefficient zero and `k_l>=0`. Schoenberg's characterization then makes `q` positive definite. `square`

## What this does and does not prove

It proves a coefficientwise kernel decomposition

\[
K_{32}=h_*+q,
\qquad q\succeq0.
\]

It does **not** extend the T16 optimum value unchanged to arbitrary signed rules. The positive-rule proof obtains an additional diagonal residual contribution of order `q(1)/N`; a signed rule can use cancellations, and positive definiteness alone gives only nonnegative centered residual energy.

This explicitly checks and rejects the tempting but invalid inference

> “positive-definite residual implies the positive-weight T22/T16 lower bound applies unchanged to signed weights.”

---

# 2. Global rank obstruction for arbitrary signed nodes

Let `H_<=3` be the space of spherical harmonics in dimensions `0,1,2,3`, with

\[
D=\dim H_{\le3}=2,861,952.
\]

Choose an orthonormal basis and let `v(x) in R^D` be its evaluation feature map. The addition theorem gives

\[
\|v(x)\|^2=D
\]

for every `x`.

For a signed mass-one rule

\[
Q=\sum_{i=1}^Nw_i\delta_{x_i},
\qquad \sum_iw_i=1,
\]

define

\[
M_Q=\sum_iw_iv(x_i)v(x_i)^T.
\]

Then `rank(M_Q)<=N` and `tr(M_Q)=D`, even when the weights are signed.

## Lemma 2.1 — rank/trace moment obstruction

For every real symmetric matrix `M` with `rank(M)<=N` and `tr(M)=D`,

\[
\|I_D-M\|_F^2\ge \frac{D^2}{N}-D.
\]

### Proof

Let the `r<=N` nonzero eigenvalues be `lambda_1,...,lambda_r`. Their sum is `D`. Then

\[
\|I-M\|_F^2
=D-2D+\sum_{j=1}^r\lambda_j^2
\ge -D+\frac{D^2}{r}
\ge \frac{D^2}{N}-D,
\]

where Cauchy-Schwarz gives `sum lambda_j^2 >=D^2/r`. No positivity of `M` or of the quadrature weights is used. `square`

## Lemma 2.2 — kernel interpretation

Let

\[
L_3(x,y)=\langle v(x),v(y)\rangle
=\sum_{\ell=0}^3d_\ell G_\ell(\langle x,y\rangle),
\]

where `d_l` is the dimension of the degree-`l` harmonic space. Then the cubature MSE for the positive-definite kernel `L_3^2` is exactly

\[
R_{L_3^2}(Q)=\|I_D-M_Q\|_F^2.
\]

### Proof

`tr(M_Q^2)=sum_ij w_iw_j L_3(x_i,x_j)^2`. The spherical mean of `L_3(x,y)^2` in either argument is `D`. Therefore

\[
R_{L_3^2}(Q)
=
\sum_{i,j}w_iw_jL_3(x_i,x_j)^2-D
=
\operatorname{tr}(M_Q^2)-D
=
\|I-M_Q\|_F^2.
\]

`square`

The exact normalized-Gegenbauer coefficients of `L_3^2` are computed symbolically. They are positive in degrees `1,...,6`. Define

\[
\gamma=
\min_{1\le\ell\le6}
\frac{k_\ell}{b_\ell},
\]

where `b_l` are those coefficients. The degree-11 interval jet gives a rigorous lower bound for every `k_l` required here. The binding degree is six.

## Theorem 2.3 — arbitrary-signed-node lower bound

For every static linear rule with at most `N=66,048` arbitrary spherical nodes and arbitrary real weights summing to one,

\[
R_{K_{32}}(Q)
\ge
\gamma\left(\frac{D^2}{N}-D\right)
\ge
1.7011781278433434\times10^{-8}.
\]

### Proof

By coefficientwise domination,

\[
K_{32}-\gamma L_3^2
\]

has nonnegative nonconstant Gegenbauer coefficients. Therefore its cubature error is nonnegative for every signed mass-one rule. Hence

\[
R_{K_{32}}(Q)
\ge\gamma R_{L_3^2}(Q).
\]

Apply Lemmas 2.1 and 2.2. `square`

Using the certified Kerdock MSE, this implies

\[
R_{K_{32}}(Q)\ge0.0699020355\,R_K.
\]

Equivalently, arbitrary signed nodes can improve over Kerdock by **at most approximately `14.3058x`** under this theorem.

## Sharpness and limitations

- The rank inequality is sharp over abstract symmetric rank-`N`, trace-`D` matrices.
- It need not be attainable by point-evaluation matrices `M_Q`; therefore the cubature bound may be improvable.
- The comparison is limited by the degree-six ReLU-kernel coefficient.
- Scans with pure harmonic feature spaces of degrees `4+` gave weaker numerical bounds; degree `<=3` is the best tested rank construction.
- This is not signed near-optimality. It only prevents the signed optimum from becoming arbitrarily small.
- The theorem is for the infinite-width depth-32 kernel. A finite-width version requires lower bounds on its first six Gegenbauer coefficients.

This theorem is materially stronger than the existing negative-mass stability lemma because it requires no bound on total variation or negative mass.

---

# 3. Quantitative phase observability

The projection theorem says that runtime information `X` has correction value

\[
V(X;e)=\mathbb E\|\mathbb E[e\mid X]\|^2.
\]

The following results quantify that value when the missing quantity is a discrete phase.

## Theorem 3.1 — binary phase information bound

Let `S` be uniform on `{−1,+1}`, let `v` be a fixed Hilbert vector, and let

\[
e=S v.
\]

For any runtime transcript `X`, put

\[
m(X)=\mathbb E[S\mid X].
\]

Then

\[
\frac{V(X;e)}{\mathbb E\|e\|^2}
=\mathbb E[m(X)^2]
\le \min\{1,2I(S;X)\},
\]

where mutual information is measured in nats.

### Proof

The optimal correction is `m(X)v`, so the normalized value is `E m^2`. For a balanced binary prior,

\[
I(S;X)
=
\mathbb E\left[
\frac{1+m}{2}\log(1+m)
+
\frac{1-m}{2}\log(1-m)
\right].
\]

The bracketed even function is at least `m^2/2` for `|m|<=1`; this follows by differentiating twice or from the binary relative-entropy/Pinsker inequality. Thus `E m^2<=2I`. `square`

### Consequence

Capturing `r` of a pure sign-oracle's correction value requires at least `r/2` nats of information about the sign. A large oracle span alone says nothing about deployable value unless the runtime transcript contains that phase information.

## Theorem 3.2 — probe-information budget

Suppose a transcript consists of sequential probes `X_1,...,X_k` and each probe satisfies

\[
I(S;X_j\mid X_{<j})\le\eta_j.
\]

Then

\[
\frac{V(X_{1:k};e)}{\mathbb E\|e\|^2}
\le
2\sum_{j=1}^k\eta_j.
\]

### Proof

Use the chain rule for mutual information and Theorem 3.1. `square`

This converts any valid per-probe KL or information bound into a correction-value ceiling.

## Theorem 3.3 — finite-action oracle identification

Let `J` be uniform on `M` candidate actions and let `X` be the runtime transcript. Every selector `hat J(X)` satisfies

\[
\Pr(\widehat J\ne J)
\ge
1-\frac{I(J;X)+\log2}{\log M}.
\]

This is Fano's inequality.

If only the correct action receives normalized oracle reward one and every incorrect action receives at most `rho`, then expected oracle capture is at most

\[
\rho+(1-\rho)
\frac{I(J;X)+\log2}{\log M}.
\]

The statement extends directly to any known reward gap.

## What is and is not closed

These theorems do not upper-bound the actual WHestBench transcript without a valid upper bound on `I(S;X)` or `I(J;X)`. Failed regressors provide neither a model-free information upper bound nor a universal impossibility theorem. They do, however, define the right next diagnostic:

1. freeze the phase/action variable;
2. define the exact legal transcript;
3. obtain a generative or likelihood model strong enough to bound transcript KL;
4. apply Theorem 3.1 or 3.3.

For unrestricted full weights, the target is deterministic from the weights in principle, so a pure information-theoretic impossibility theorem remains false.

---

# 4. Finite-width arbitrary-node target

## What is proved

The fixed-MUB-support allocation theorem extends to finite width under the explicit Gaussian-first-layer and even-degree nondegeneracy conditions already recorded in the proof package.

The rank theorem above also has an exact finite-width **schema**: for any finite-width ensemble kernel with normalized-Gegenbauer coefficients `k_l^(m)`, define

\[
\gamma_m=
\min_{1\le\ell\le6}\frac{k_\ell^{(m)}}{b_\ell}.
\]

Then every signed `N`-node rule satisfies

\[
R_{K_m}(Q)
\ge
\gamma_m\left(\frac{D^2}{N}-D\right).
\]

## What remains unavailable

No rigorous lower intervals for the width-256 coefficients `k_1^(256),...,k_6^(256)` are currently present. Likewise, no finite-width Delsarte minorant with a certified arbitrary-node near-optimality ratio has been constructed.

The existing approximate finite-width work estimates an `O(10%)` change in absolute Kerdock MSE, far larger than the `0.023%` limiting-kernel optimality gap. Consequently, infinite-width near-optimality cannot be transferred by a generic perturbation argument.

## Checked failure modes

- Uniform kernel convergence without a numerical rate is insufficient; arbitrarily small PSD perturbations can reverse a tiny optimizer gap.
- The finite-width fixed-support theorem does not imply arbitrary-node optimality.
- Nonnegative monomial/Hermite coefficients do not by themselves produce a Delsarte minorant with the required objective value.
- A finite-width result must either certify the relevant kernel directly or exploit additional architecture-specific structure.

**Release status:** open theorem target, with an explicit input requirement rather than an undisclosed proof gap.

---

# 5. Residual-kernel and control corners

The residual identity

\[
I(g)+Q(f-g)-I(f)=(Q-I)(f-g)
\]

holds for arbitrary network-dependent `g` whenever `I(g)` is exactly available. Thus every hybrid method has an exact residual second-moment kernel.

For bounded deterministic rotation-equivariant linear surrogates, degree `l` is multiplied by `tau_l`, so residual coefficients are

\[
q_l^{res}=|1-\tau_l|^2q_l.
\]

This yields exact spectral recertification.

The following corners are explicitly checked:

- modifying only degrees `0..5` cannot alter complete-Kerdock error;
- uniform annihilation remains true even when control parameters are selected adaptively;
- network-dependent nonlinear surrogates need not act diagonally in harmonic space;
- candidate-dependent fitting can break isotropy;
- exact integrability does not imply low harmonic degree;
- residualization can make another node set optimal, so Kerdock must be recertified rather than inherited automatically.

---

# 6. Final theorem hierarchy

## Promote now, subject to independent implementation review

1. **Coefficientwise-positive T16 residual:** `K_32-h_*` is positive definite.
2. **Global arbitrary-signed-node rank floor:** every signed mass-one rule with at most 66,048 nodes has MSE at least `1.7011781e-8`.
3. **Binary phase information ceiling:** observable correction fraction is at most `2I`.
4. **Sequential probe information ceiling:** information contributions add by the chain rule.
5. **Finite-action oracle-capture bound:** Fano converts transcript information into selection regret.

## Already proved and retained

- tightened static nonnegative near-optimality;
- finite-width fixed-MUB-support allocation under explicit assumptions;
- correction projection and adaptive dictionary theorems;
- phase-flip, Haar-orientation, and common-bias scoped impossibility results;
- downstream replacement and ReLU-crossing mathematics;
- exact control null spaces and residual recertification.

## Remain open

1. finite-width arbitrary-node near-optimality;
2. signed arbitrary-node near-optimality rather than a nonzero floor;
3. a KL/mutual-information upper bound for an actual legal WHestBench transcript;
4. a constructive high-degree, exactly integrable, compute-positive residual surrogate;
5. recovery of row-level oracle-swap data for cross-layer coherence.

## Explicitly false or unsupported

- universal impossibility of nonlinear or adaptive estimation;
- transfer of the `0.023%` certificate to width 256 without a new bound;
- extension of T27 beyond the fixed MUB support;
- inference that failed learners imply low mutual information;
- inference that positive-definite residual alone preserves the positive-weight lower bound;
- inference that exact-mean controls are necessarily low degree.

---

# 7. Verification artifact

`verify_best_theorem_targets.py` performs:

- 80-decimal interval Taylor-jet propagation through 32 ReLU-kernel compositions;
- exact rational monomial-to-Gegenbauer projections;
- comparison with certified T16 coefficient intervals;
- exact symbolic expansion of the squared degree-`<=3` reproducing kernel;
- rank-obstruction arithmetic;
- output of all margins and constants to `best_theorem_targets_verification.json`.

Trust note: the new numerical enclosure uses `mpmath.iv` plus exact SymPy rational algebra. It should be independently reproduced with Arb/FLINT or another directed-rounding implementation before final publication, just as the manuscript already requires independent human and release-level sign-off.
