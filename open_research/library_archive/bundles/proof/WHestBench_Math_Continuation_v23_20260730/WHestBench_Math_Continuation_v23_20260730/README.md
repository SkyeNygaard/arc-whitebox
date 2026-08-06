# WHestBench mathematical continuation v23

**Date:** 2026-07-30  
**Starting state:** canonical v22 post-agent-dispatch frontier  
**Method:** proof-first continuation, alternating construction with hostile counterexample and scope audit  
**Protected data:** not opened  
**Deployable estimator:** none claimed

## Executive conclusion

The strongest new result is an **inertia-strengthened arbitrary-signed static floor**. The v21/v22 comparison certificate already used rank and every harmonic block trace. An actual signed atomic rule contains additional information absent from that abstract matrix relaxation: the number of positive eigenvalues of every feature moment matrix is bounded by the number of positive atomic weights.

For the dimension-256, depth-32 limiting normalized-ReLU kernel, every static, network-independent, mass-one rule using at most 66,048 arbitrary spherical nodes and arbitrary real weights now satisfies

\[
R_K(Q)\ge
2.2804870463653914348948735097257249\times10^{-7}
=0.9370605225569535\,R_K(Q_{\rm Kerdock}).
\]

Hence

\[
\boxed{
\text{same-cost static signed improvement}\le1.0671669288460727\times.
}
\]

The numerical improvement over the degree-280 v21 floor is small, but the proof ingredient is new and genuinely atomic. It does not merely optimize another profile radius.

Dropping the mass-one assumption barely changes the theorem. The arbitrary-total-mass floor is

\[
R_K(Q)\ge0.9370603033214825\,R_K(Q_{\rm Kerdock}),
\]

with cap

\[
1.0671671785214067\times,
\]

and the relaxed minimizing total mass is \(0.9999997660391557\).

At the competition's recorded 4.34× adjusted gap, even a rule saturating the new raw-MSE boundary would need total effective evaluation cost below

\[
{1.0671669288460727\over4.34}=0.2458909974
\]

of baseline. Static geometry alone is therefore decisively not the winning-scale route.

The mathematical frontier now separates cleanly:

1. **Static equal-cost signed cubature:** capped at 1.06717× in the limiting kernel; stronger progress requires sphere-evaluation-variety constraints.
2. **Actual-width proof:** universal coefficient monotonicity is false; the exact remaining route is an architecture-specific noise-stability moment dual, beginning at degree 62.
3. **Constructive Oracle route:** reduce a physical rank-4/5 source to scalar normal-equation contractions and exact nonlinear replay; generic coefficient learning is not the bottleneck.
4. **Replay bridge:** an independent Gaussian–ReLU suffix is nonexpansive in expected squared checkpoint-state distance, so state approximation cannot amplify under exact replay. The unresolved issue is target replayability and source legality.

---

# I. New static theorem: inertia beyond rank

## I.1 Positive-index rank lemma

Let \(M=M^T\) have trace \(T>0\) and at most \(p\) positive eigenvalues. Then

\[
\|M\|_F^2\ge {T^2\over p}.
\]

The proof is immediate but powerful: positive eigenvalues carry at least the full positive trace, and Cauchy minimizes their squared sum when they are equal.

For an atomic feature moment

\[
M=E^TWE,
\]

Sylvester inertia monotonicity implies

\[
n_+(M)\le n_+(W).
\]

A mass-one rule with at most \(N\) nodes has two cases:

- if all \(N\) nonzero weights are positive, the much stronger T22 positive theorem applies;
- otherwise it has at most \(N-1\) positive weights, so every comparison moment matrix obeys the \(p=N-1\) floor.

For one harmonic comparison profile with

\[
T=\sum_\ell a_\ell d_\ell,
\qquad
S_2=\sum_\ell a_\ell^2d_\ell,
\]

the old abstract floor was

\[
{T^2\over N}-S_2.
\]

The actual signed atomic floor is

\[
\boxed{
{T^2\over N-1}-S_2.
}
\]

The coefficient consumption of the profile is unchanged. The frozen 146-profile linear program can therefore be reoptimized with a stronger objective while retaining every degree-1 through degree-320 kernel-capacity constraint.

## I.2 Exact certification

The floating LP was used only to discover a candidate. Its weights were:

- shrunk by a fixed safety factor;
- rounded downward to a \(10^{-30}\) grid;
- replayed with exact rational arithmetic;
- checked against the v21 directed kernel-coefficient lower endpoints through degree 320.

The exact witness has 134 active components. Its minimum active slack is positive at degree 267; the minimum all-degree audit slack is the inherited degree-320 tail endpoint.

The certificate is independently replayed from its rational component weights by `verify_inertia_certificates_exact.py`. This replay does not independently regenerate the v21 interval endpoints; external Arb/FLINT reproduction of those endpoints remains the publication gate.

## I.3 Arbitrary total mass

For total mass \(s>0\), the signed/inertia branch gives profile discrepancy

\[
S_2(1-2s)+{s^2T^2\over N-1}.
\]

The comparison kernel leaves a large unused constant harmonic residual. Adding its exact mass-mismatch cost yields a scalar quadratic in \(s\), whose exact minimizer is near one.

The exactly-\(N\)-positive branch is treated separately. If \(Q=sQ_0\) with \(Q_0\) a positive probability rule, orthogonality of the constant and nonconstant harmonics gives

\[
R(sQ_0)=k_0(1-s)^2+s^2R(Q_0),
\]

so T22 remains much stronger than the signed branch. For \(s\le0\), the constant-mode error is already enormous relative to the claimed floor.

This completes the unrestricted-total-mass static theorem without silently assuming unbiasedness.

---

# II. Sign-count hierarchy

The inertia argument strengthens whenever the rule has more than one negative atom. If at least \(q\) nonzero weights are negative, then every profile moment has at most \(N-q\) positive eigenvalues and

\[
\|M\|_F^2\ge {T^2\over N-q}.
\]

Separate exact-rational witnesses give:

| minimum negative atoms | risk / Kerdock | same-cost gain cap |
|---:|---:|---:|
| 1 | 0.9370605226 | 1.0671669289× |
| 2 | 0.9370739626 | 1.0671516230× |
| 16 | 0.9372729741 | 1.0669250343× |
| 64 | 0.9379559403 | 1.0661481600× |
| 256 | 0.9406984935 | 1.0630398655× |
| 1,024 | 0.9518267726 | 1.0506113389× |
| 1,072 | 0.9525316552 | **1.0498338765×** |
| 2,048 | 0.9670812292 | 1.0340393029× |
| 4,096 | 0.9991036420 | 1.0008971622× |
| 4,160 | 1.0001384731 | **0.9998615461×** |
| 8,192 | 1.0699484210 | 0.9346244925× |

Consequences:

- a rule with at least 1,072 negative atoms cannot achieve a 1.05× equal-cost improvement;
- a rule with at least 4,160 negative atoms is certified worse than Kerdock;
- any hypothetical rule near the universal 1.06717× boundary must have extremely sparse negativity.

This combines with the negative-mass and realizability theorems:

- the negative-mass bridge says a material raw gain requires a minimum **amount** of negative mass;
- the sign-count theorem says near-boundary performance requires very few negative **locations**;
- the shared-profile compactness theorem says approaching the older abstract floor requires unbounded total variation or degenerating geometry.

A near-optimal signed rule is therefore forced into a narrow unstable corner: few negative atoms, enough negative mass to matter, and increasingly singular cancellation.

---

# III. What static relaxations are now exhausted

## III.1 Simultaneous sharpness

One shared abstract rank-\(N\) block moment matrix can attain the rank floor for every nonnegative harmonic profile after diagonal rescaling. Therefore the following joint information is insufficient:

- one global moment matrix;
- rank at most \(N\);
- all harmonic block traces;
- all 146 profile rescalings simultaneously.

A larger SDP containing only those constraints cannot improve the theorem.

## III.2 Atomic strictness

The released degree-280 witness contains two positive profiles using the same adjacent degrees 3 and 4 but distinct mixing ratios. Equality in both would force every off-diagonal node inner product to be a common zero of \(G_3\) and \(G_4\). Their exact polynomial gcd is one, so no actual atomic rule attains the old floor.

For every finite total-variation cap \(V\), compactness therefore gives a positive but currently non-explicit gap above the old floor. The same logic identifies the only ways a near-attaining sequence can escape: unbounded weights, vanishing active weights, node coalescence, or collapsing evaluation singular values.

## III.3 Remaining realizability target

The residual static gap is no longer a coefficient-search problem. A successful proof must add constraints obeyed by actual spherical evaluation vectors but absent from arbitrary block matrices, such as:

- the sphere ideal \(\sum x_i^2=1\);
- multiplication-operator commutation;
- joint localizing/catalecticant matrices;
- addition-product identities between harmonic blocks;
- quantitative node-separation or collision localizations;
- a closed treatment of unbounded signed total variation.

The best immediate target is a **small-degree shared moment relaxation** for the incompatible degree-3/4 profiles, with exact rational dual output. A full degree-280 moment SDP is unnecessary until this pilot produces a nontrivial numeric gap.

---

# IV. Actual-width theorem route

## IV.1 Exact finite-width representation

For any fixed finite width, condition on all parameters downstream of the first Gaussian layer. The network output is a square-integrable function of a Gaussian vector and has a multivariate Hermite expansion. Mehler's identity gives

\[
K_m(t)=\sum_{n\ge0}a_n^{(m)}t^n,
\qquad a_n^{(m)}\ge0.
\]

Every Gegenbauer coefficient is a positive linear functional of this chaos-energy measure:

\[
k_\ell^{(m)}=\sum_{n\ge\ell}M_{n\ell}a_n^{(m)},
\qquad M_{n\ell}\ge0.
\]

This is the correct exact finite-width state variable.

## IV.2 Universal shortcut disproved

The parallel v22 audit proves that finite-width coefficients do not universally dominate infinite-width coefficients. At width one, every later scalar ReLU layer remains a random positive multiple of the first-layer feature, so the deep kernel remains \(\kappa(t)\), while the infinite-width depth-two kernel is \(\kappa(\kappa(t))\) and has a strictly positive cubic coefficient absent at width one.

Thus qualitative positivity or convergence cannot transfer the limiting certificate one-sidedly.

## IV.3 Moment-dual proof program

If directed analysis supplies interval observations

\[
L_j\le K_m(t_j)\le U_j,
\]

then each required coefficient lower bound is a semi-infinite nonnegative-moment LP. Its dual seeks an inequality

\[
M_{n\ell}\ge\lambda_0+\sum_j\lambda_jt_j^n
\quad\text{for every }n\ge0.
\]

A complete actual-width proof requires:

1. rigorous width-256 kernel or moment intervals;
2. rational primal and dual witnesses;
3. a finite tail proof for all \(n\);
4. exact monomial-to-Gegenbauer conversion;
5. a finite-width Kerdock denominator upper bound.

Finite point samples alone are nonidentifying: a finite Vandermonde null vector can move positive chaos mass without changing normalization or sampled values. The moment LP and tail dual are essential.

## IV.4 Recommended order

The exact limiting subcertificate frontier says:

- 80% floor: degree 62;
- 85%: degree 84;
- 90%: degree 128;
- 92%: degree 164;
- 93%: degree 194.

The correct sequence is therefore degree 62, then 84, then 128. If the degree-62 moment dual cannot produce a nontrivial \(\alpha/\beta\), the actual-width branch should stop rather than accumulate point samples.

---

# V. Constructive Oracle path

## V.1 Exact coefficient geometry

For a frozen target-free output source basis \(A=[c_1,\ldots,c_r]\), baseline error \(e=z_0-\mu\), Gram \(G=A^TA\), and cross-moment \(b=A^Te\), exact risk is

\[
\|e+A\alpha\|^2
=
\|e\|^2+2\alpha^Tb+\alpha^TG\alpha.
\]

The missing information is only \(b\), not the entire target vector.

If \(z_0=Qf\) and \(\mu=Pf\), then each component has the scalar adjoint reduction

\[
\boxed{
b_j=(Q-P)g_j,
\qquad g_j(x)=\langle c_j,f(x)\rangle.
}
\]

A rank-4 or rank-5 coefficient problem is exactly four or five scalar integration errors. This is a real dimensional reduction, although the scalar integrands remain physically oriented and may still be hard.

## V.2 Exact affine replay region

A ReLU suffix is affine on every fixed activation cell. If a coefficient box produces no gate changes, exact replay equals the baseline Jacobian action throughout the entire box. A uniform gate certificate is obtained by propagating source-response matrices and checking

\[
|W_{k+1}U_k|r<|a_{k+1}|
\]

coordinatewise.

When gates cross, the exact scalar remainder obeys

\[
\sigma(a+d)-\sigma(a)
=
\mathbf1_{a>0}d+\rho(a,d),
\qquad
|\rho(a,d)|\le(|d|-|a|)_+.
\]

The nonlinearity is therefore localized exactly to near-margin gates rather than bounded by a generic Hessian.

## V.3 Gaussian suffix nonexpansivity

For an independent width-\(m\) Gaussian He matrix, entries \(N(0,2/m)\),

\[
\mathbb E\|\sigma(Wu)-\sigma(Wv)\|^2
=
\|u\|^2+\|v\|^2-2\|u\|\|v\|\kappa(\rho)
\le\|u-v\|^2.
\]

Iterating through an independent Gaussian–ReLU suffix gives

\[
\boxed{
\mathbb E\|F(U)-F(V)\|^2
\le
\mathbb E\|U-V\|^2.
}
\]

For a checkpoint basis \(B\), coefficient error therefore obeys

\[
\mathbb E\|F(u_0+Ba)-F(u_0+B\widehat a)\|^2
\le
(a-\widehat a)^T(B^TB)(a-\widehat a).
\]

This is an exact nonlinear transfer theorem for two replayed states. It does **not** prove that the integration target is replay of a single checkpoint state, nor does it apply when the source basis inspects the same suffix weights being averaged over. Those are the real remaining obstacles.

## V.4 Best constructive protocol

1. Freeze four or five physical, covariant checkpoint channels.
2. Measure their exact source-span ceiling once.
3. Stop unless final-output oracle ratio is comfortably below 0.20–0.22.
4. Derive the scalar contractions \((Q-P)g_j\) by adjoint, Stein, telescoping, or reuse identities.
5. Ensure source construction is prefix-measurable, leave-one-row, or otherwise legally separated from suffix randomness.
6. Certify the largest affine coefficient box.
7. Use exact Gaussian crossing/nonexpansivity only for the residual suffix.
8. Compare a bounded residual policy against the matched global constant action, not against zero.

No more generic feature dictionaries should be fitted before steps 2–4 succeed.

---

# VI. Exhaustive disposition of mathematical paths

| Path | Result | Disposition |
|---|---|---|
| More degree/radius profile search | Degree-280 coefficient witness already near its abstract profile frontier | Stop unless the certificate family changes |
| Joint rank/block-trace SDP | Simultaneously sharp for all profiles | Closed |
| Atomic inertia | New universal improvement to 93.7060523% | Promoted |
| Sign-count inertia | Rules with ≥1,072 negative atoms cannot gain 1.05×; ≥4,160 are worse than Kerdock | Promoted |
| Arbitrary total mass | Constant mode forces mass 0.999999766; cap unchanged | Closed quantitatively |
| Bounded negative mass | Exact RKHS bridge useful for small signed perturbations | Retained |
| Strict nonattainment alone | No universal numeric gap without compactness/conditioning | Do not overclaim |
| Sphere-ideal/localizing moments | Only remaining static route with plausible theorem value | Top static proof target |
| Universal finite-width coefficient domination | Exact width-one counterexample | Closed false |
| Finite-width chaos/moment dual | Exact representation and dual formulation | Top actual-width proof target |
| Pointwise finite-width kernel closeness | Does not identify high-degree coefficients | Insufficient alone |
| Generic all-128 A51 estimation | Interface may encode full 256-dimensional defect | Deprioritize |
| Rank-4/5 scalar contractions | Exact normal-equation reduction | Top constructive path |
| Linearized replay only | Gate reversal risk remains | Insufficient |
| Exact affine-cell replay | Complete when gate box passes | Use first |
| Gaussian local crossing formula | Exact for prefix-measurable source and independent row | Retained bridge |
| Gaussian suffix nonexpansivity | Exact nonlinear state-error contraction in expectation | Promoted bridge |
| Generic coefficient learning | Existing representations fail transfer/phase | Closed |
| Constant-first residual | Logically open around a new high-capacity source | Conditional |
| Exactly integrable residual spectrum | Named families fail; methodology open | Reopen only with new analytic object |
| Universal information impossibility | False because full weights determine target | Closed false |
| Declared computational lower bound | Possible only after fixing a realistic query/action class | Later theory |

---

# VII. Ranked continuation

## Priority 1 — physical contracted source identity

This is the only proof-adjacent route with plausible winning-scale upside.

- Select a covariant rank-4/5 checkpoint source without target labels.
- Require source-span ratio below 0.20–0.22.
- Derive four/five exact scalar cross-moment identities.
- Use affine replay, exact crossing moments, and suffix nonexpansivity.

Failure of the source-capacity gate closes the branch before coefficient work.

## Priority 2 — degree-62 actual-width pilot

- Write the exact two-input Gram Markov transition for width 256.
- Certify a small set of moment/kernel intervals.
- Solve a rational moment dual through degree 62.
- Certify a finite-width Kerdock denominator.
- Escalate only if the resulting floor is nontrivial.

## Priority 3 — quantitative sphere-evaluation separation

- Start only with the two incompatible degree-3/4 profiles.
- Add low-degree sphere-ideal localizing constraints.
- Seek an exact rational dual giving an explicit gap above T70.
- Handle collision strata separately rather than assuming node separation.

## Priority 4 — replayability theorem

The suffix theorem solves state-error propagation, not target representation. The key open question is whether a useful Oracle repair can be represented as replay of a low-dimensional legal checkpoint state plus a controlled replacement-bias term.

## Priority 5 — scoped computational lower bound

Define a realistic class after the constructive route identifies the sufficient contractions. A theorem over “all white-box algorithms” is neither plausible nor useful.

---

# VIII. Verification and release audit

The v23 package reruns:

- the original v21 degree-280 verifier;
- the shared-profile/Sturm verifier;
- the exact rational inertia headline and sign-count hierarchy replay;
- the Gaussian suffix nonexpansivity verifier;
- the parallel Gaussian crossing and finite-width monotonicity scripts.

A release defect was found in the parallel `best_paths_beyond_v21` ZIP: its aggregate verifier contains hard-coded absolute `/mnt/data/...` input paths, and the A51 raw inputs are not vendored in that ZIP. Its saved JSON outputs remain useful evidence, but the clean-unpack aggregate replay is not self-contained. The v23 package does not repeat the claim that this aggregate verifier passes from a clean directory.

The remaining external theorem gate is unchanged: reproduce the v21 kernel interval endpoints and the new rational certificates with an independent Arb/FLINT-quality implementation and obtain named human review of the harmonic/rank/inertia bridge.

---

# IX. Final state

The static proof program has moved from an abstract 93.7046% floor to an actual-inertia 93.7061% floor and now classifies the sign patterns capable of approaching it. The gain is not competition-sized, but it sharply identifies what is and is not left:

- coefficient/radius search is no longer the bottleneck;
- many-negative-node signed rules are provably worse;
- arbitrary mass does not help;
- joint abstract moment matrices are exhausted;
- finite-width transfer needs real coefficient information;
- the best constructive route is a very small physical contraction interface plus exact replay.

The next breakthrough, if one exists, is most likely either:

1. a legal four-channel source with a theorem-derived scalar contraction identity, or
2. an actual-width degree-62 moment certificate that reveals where finite-width harmonic mass really lies.
