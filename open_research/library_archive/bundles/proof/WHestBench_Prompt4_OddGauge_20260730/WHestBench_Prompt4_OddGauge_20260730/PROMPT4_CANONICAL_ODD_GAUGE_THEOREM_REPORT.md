# Prompt 4 — Canonical Orientation-Odd Gauge Fixing

**Date:** 2026-07-30  
**Status:** theorem-first progress; no protected or official cohort opened  
**Canonical inputs:** v19 T49 gauge obstruction; M160 pooled heterogeneity; M161 five-source capacity/failure; A50 orientation-odd preregistration  
**Primary conclusion:** a globally continuous deterministic orientation gauge is impossible in every nontrivial sign problem. The mathematically correct escape is either (i) use raw orientation-odd contractions directly, without canonicalizing them, or (ii) use a local reference-frame gauge with a certified margin and abstention. For a genuinely ambiguous rank-`k` basis, polar alignment to a legal rank-`k` reference frame is the canonical local construction.

---

## 1. Canonical question and scope

The existing obstruction is representation-level. The M153/T4 feature map contains only norms, norm ratios, cosines and angle magnitudes. Under simultaneous reversal of all represented candidate trajectories, those features are unchanged while a signed target such as a correction cosine reverses. Therefore an exactly representation-consistent policy over that quotient cannot output a nonzero signed coefficient.

This does **not** prove that the physical estimator class lacks phase. It proves that the observation map discarded phase. Prompt 4 asks for the minimum legal phase-bearing observable and for a theorem that says when it is well defined, stable, useful and cheap.

We distinguish three different gauge groups, because conflating them produces false claims.

1. **Global sign:** one common `Z_2` action, `C -> -C`.
2. **Independent signs:** `m` dictionary columns may be reoriented independently, group `(Z_2)^m`.
3. **Basis gauge:** a rank-`k` source subspace is represented by an arbitrary orthonormal frame `U`, with `U ~ UO`, `O in O(k)`.

A single odd scalar is algebraically sufficient only for the first case. It cannot generically fix the second or third.

---

## 2. Formal model

Let `X` be the legal runtime transcript, `G` a compact gauge group, and `rho:G -> O(K)` the representation on coefficient space `K`. A coefficient policy `h:X -> K` is representation-consistent when

\[
h(gx)=\rho(g)h(x).
\]

The physical correction is unchanged when the dictionary and coefficients transform together. For a source matrix `A` and coefficient vector `a`, a right-basis change gives

\[
A\mapsto AO,\qquad a\mapsto O^\top a,
\]

so `Aa` is invariant.

An **orientation-odd observable** is a runtime quantity `z` transforming under a nontrivial representation of `G`. In the global-sign case, this means

\[
z(-C)=-z(C).
\]

The simplest example is a signed contraction

\[
z=\langle u,s\rangle,
\]

where `u` reverses with the candidate and `s` is a legal physically oriented reference that does **not** co-reverse under that representation gauge. If both factors reverse, the contraction is even and does not escape T49.

---

## 3. Result P4.1 — minimal odd information

### Theorem P4.1A: global sign

Let `G=Z_2`. If every observed feature is even, every equivariant scalar signed output is zero. Adding one nonzero odd scalar channel is representation-theoretically sufficient to escape this obstruction.

### Proof

The even observation is fixed by the nontrivial group element. Equivariance requires the signed output to equal its negative, hence it is zero. If an odd scalar `z` is supplied, the observation is no longer fixed: `z -> -z`. Nonzero odd policies, for example `h(z)=alpha z`, now exist. This proves algebraic sufficiency, not statistical usefulness. ∎

### Skeptical attack

An odd scalar can be independent of the target phase. Algebraic escape is not predictive escape. Section 8 proves that an independent random odd sign has zero optimal correction value and any nonzero unshrunk use increases risk.

### Theorem P4.1B: independent sign characters

Suppose `G=(Z_2)^m`. Let the available scalar odd channels transform by characters `chi_1,...,chi_r`. If these characters span a subgroup of rank `<m` in the dual group, then there is a nontrivial subgroup `H` acting trivially on all observed channels. Every equivariant output must lie in the `H`-fixed coefficient subspace. In particular, generic recovery of all `m` independently signed coefficients requires odd characters spanning the full dual group, so at least `m` scalar sign characters are needed in a diagonal real representation.

### Proof

Let

\[
H=\bigcap_{j=1}^r \ker \chi_j.
\]

If the character rank is below `m`, then `H` is nontrivial. Every `h in H` fixes the observation. Equivariance therefore implies `a=rho(h)a` for every `h in H`; hence `a` belongs to the `H`-fixed subspace. Any coefficient coordinate whose character is nontrivial on `H` must vanish. To leave no independent sign direction invisible, the observed characters must separate every element of `G`, which requires rank `m`. ∎

### Consequence for A50

“At most four signed contractions” can fully expose at most four independent sign characters unless other physical conventions have already removed the remaining gauge. If the actual ambiguity is only the single simultaneous reversal identified in the M153 audit, one contraction is enough. The experiment must state which group is being fixed.

---

## 4. Result P4.2 — no continuous global sign gauge

### Theorem

For `d>=2`, there is no continuous global rule that chooses one unit vector from every unoriented line in `R^d`. Equivalently, the antipodal covering

\[
\pi:S^{d-1}\to \mathbb{RP}^{d-1}
\]

has no continuous section.

### Proof

For `d>=3`, `pi_1(S^{d-1})=0` while `pi_1(RP^{d-1})=Z_2`. If a section `s` existed, then

\[
\pi_*\circ s_*=\operatorname{id}_{Z_2},
\]

but `s_*` maps into the trivial group, impossible.

For `d=2`, identify both spaces with circles. The antipodal quotient has degree two. A section would have to satisfy

\[
2\deg(s)=1,
\]

which is impossible over the integers. ∎

### Scalar-anchor corollary

Every continuous odd scalar `f:S^{d-1}->R` has a zero. Indeed, connect `u` to `-u` by a path; `f(-u)=-f(u)`, so the intermediate-value theorem gives a zero. Thus a sign rule `sign(f(u))` must either be undefined somewhere or be discontinuous.

### Skeptical attack

A measurable lexicographic or positive-pivot rule does exist. The theorem does not prohibit deterministic gauges; it prohibits **global continuity and uniform stability**. Therefore a legal construction must expose its degeneracy set and abstain near it. Silent tie breaking is not a stability theorem.

---

## 5. Result P4.3 — canonical local polar gauge

### Setup

Let `U in R^{n x k}` have orthonormal columns and represent a physical source subspace. Its frame is ambiguous under

\[
U\mapsto UO,\qquad O\in O(k).
\]

Let `B(X) in R^{n x k}` be a legal reference frame, constructed without labels or oracle coefficients. Crucially, `B` is a function of gauge-invariant physical data: changing only the arbitrary right frame `U -> UO` leaves `B` unchanged. It is covariant under any legal ambient orthogonal action `H`:

\[
U\mapsto HU,\qquad B\mapsto HB.
\]

Define the cross-Gram matrix

\[
C=U^\top B.
\]

Assume `C` is invertible and write its polar decomposition

\[
C=QP,\qquad Q\in O(k),\quad P=(C^\top C)^{1/2}\succ0.
\]

Define the oriented frame

\[
\bar U=UQ.
\]

### Theorem P4.3A: invariance and covariance

1. `bar U` is independent of the arbitrary right frame `O`.
2. `bar U` transforms covariantly under the ambient action `H`.
3. `bar U` is the unique frame in the source subspace satisfying

\[
\bar U^\top B\succ0.
\]

### Proof

Under `U'=UO`,

\[
C'=O^\top C=(O^\top Q)P.
\]

The polar factor of `C'` is `O^TQ`, so

\[
\bar U'=UO(O^\top Q)=UQ=\bar U.
\]

Under `U'=HU` and `B'=HB`, the cross-Gram is unchanged and `bar U'=Hbar U`.

Finally, any alternative frame is `UR`. The condition

\[
(UR)^\top B=R^\top C\succ0
\]

forces `R=Q` by uniqueness of the polar decomposition. ∎

### Scalar reduction

For `k=1`, `C=u^Tb`, `Q=sign(C)`, and

\[
\bar u=u\,\operatorname{sign}(u^\top b).
\]

This is exactly the reference-contraction sign gauge.

### Theorem P4.3B: degeneracy and minimum reference rank

Within this linear reference-frame contraction class, the gauge is unique exactly when `sigma_min(C)>0`. If `rank(C)=r<k`, an `O(k-r)` stabilizer remains and no unique full frame can be recovered from this reference.

### Proof

If `C` is singular, its polar factor is nonunique on the null space. Equivalently, every orthogonal transformation acting trivially on the observed `r`-dimensional row space and arbitrarily on its orthogonal complement leaves the reference alignment unchanged. Therefore a full `O(k)` gauge requires a rank-`k` cross-Gram, hence at least `k` independent reference directions. ∎

### Skeptical attack

A full-rank reference can still be statistically irrelevant. For example, an independent random Gaussian reference is full rank almost surely and fixes the algebraic gauge, but its orientation is independent of target phase. The local gauge theorem establishes well-defined coordinates, not useful coordinates.

A reference can also be circular. If `B(U)=U`, then under a right reparameterization both `U` and `B` co-rotate, `U^TB=I`, and the purported oriented frame remains `U`; no gauge has been removed. Therefore every proposed anchor must pass a **non-co-rotation test**: deliberate reorientation of the represented basis changes `U` but leaves the physical reference `B` fixed.

---

## 6. Result P4.4 — perturbation and finite-precision stability

Let `C` and `C_hat=C+E` be nonsingular cross-Gram matrices, with polar factors `Q` and `Q_hat`. For every unitarily invariant norm,

\[
\|Q_\text{hat}-Q\|
\le
\frac{2\|E\|}{\sigma_{\min}(C)+\sigma_{\min}(C_\text{hat})}.
\]

Using Weyl's inequality, if `||E||_2<gamma=sigma_min(C)`, then

\[
\|Q_\text{hat}-Q\|
\le
\frac{2\|E\|}{2\gamma-\|E\|_2}.
\]

Thus the correct stability statistic is the smallest singular value of the **cross-Gram reference alignment**, not merely a singular-value gap inside the downstream basis.

### Certified abstention rule

Suppose a computation provides `C_hat` and a deterministic total error bound `epsilon_C` such that

\[
\|C_\text{hat}-C\|_2\le\epsilon_C.
\]

For a preregistered physical margin `tau>0`, accept only if

\[
\sigma_{\min}(C_\text{hat})>\tau+\epsilon_C.
\]

Then `sigma_min(C)>tau`, the gauge is unique, and the orientation perturbation is certified.

### Dot-product error for `k=1`

For a length-`n` floating-point dot product with unit roundoff `u`,

\[
|\operatorname{fl}(u^Tb)-u^Tb|
\le
\gamma_n\sum_i |u_i b_i|,
\qquad
\gamma_n=\frac{nu}{1-nu}.
\]

Input perturbation bounds are added to this arithmetic term. The sign is certified only when the computed magnitude exceeds the full error bound plus the preregistered physical margin.

### Skeptical attack

Thresholding by a numerical roundoff bound alone is insufficient. A contraction can be computed with an unambiguous sign yet have tiny **physical** margin, so a small network or rotation perturbation flips it. A50 must separate numerical certification from distributional/physical stability.

---

## 7. Result P4.5 — the current largest-coordinate rule is only local

The A50 preregistration proposes:

1. order singular vectors by singular value;
2. orient each by its largest-magnitude coordinate;
3. break ties by coordinate index;
4. abstain below a margin.

This is deterministic in a fixed physical coordinate basis, but it has two independent degeneracies.

### 7.1 Repeated singular values

If `sigma_j=sigma_{j+1}`, the singular subspace is determined but its individual vectors are not. An eigensolver may return any `O(2)` rotation. Per-vector positive pivots do not repair this missing basis choice. The whole block must be polar-aligned to a legal reference frame, or the block must be rejected.

### 7.2 Pivot ties

Consider

\[
u_\varepsilon=\frac{(1,-1+\varepsilon)}{\|(1,-1+\varepsilon)\|}.
\]

For `epsilon>0`, coordinate 1 is the largest in magnitude and the positive-pivot orientation tends to `(1,-1)/sqrt(2)`. For `epsilon<0`, coordinate 2 is largest and negative, so the rule flips the vector and tends to `(-1,1)/sqrt(2)`. The oriented output jumps by norm `2` at `epsilon=0`.

### Required local margins

For each singular vector, a valid acceptance certificate needs at least:

- a singular-block separation margin;
- a pivot gap
  \[
  p=|u_{i_*}|-\max_{i\ne i_*}|u_i|;
  \]
- a perturbation bound on the singular vector;
- a declared physical coordinate convention.

If the singular gap is small, use subspace polar alignment, not arbitrary tie breaking.

### Recommendation

Replace the positive-pivot construction as the primary theorem object with one of:

1. **Direct odd features:** retain the raw signed contractions `z_j=<u_j,s>` and use an equivariant bounded policy. No sign section is formed.
2. **Polar reference gauge:** align the entire low-dimensional block using `U^TB`; abstain when its smallest singular value is small.

The positive-pivot rule remains a valid bounded comparator when all local margins are explicitly certified.

---

## 8. Result P4.6 — exact phase-error-to-risk theorem

### Projection identity

Let `e` be the baseline final-output error in a Hilbert space, and let `S` be the span of a frozen source matrix. Let

\[
c^*=P_S e
\]

be the exact source-span oracle correction. Then

\[
R_0=\|e\|^2,
\qquad
R_*=\|e-c^*\|^2,
\qquad
\Delta=R_0-R_*=\|c^*\|^2.
\]

For any approximate correction `c_hat in S`,

\[
\boxed{
R(c_\text{hat})=R_*+\|c_\text{hat}-c^*\|^2.
}
\]

### Proof

The residual `e-c*` is orthogonal to `S`, while `c_hat-c*` lies in `S`. Expand the squared norm. ∎

This identity is the exact transfer theorem for final-output source corrections. Coefficient error must be measured in the physical source Gram norm, not in an arbitrary coordinate Euclidean norm.

### Scalar phase law

Suppose `c_hat=t c*`, where `t` may vary by case. Weight cases by their oracle improvement energy `Delta`. Define

\[
\mathbb E_\Delta[f]
=
\frac{\mathbb E[\Delta f]}{\mathbb E[\Delta]}.
\]

Then the retained oracle-improvement fraction is exactly

\[
\boxed{
\eta=2\mathbb E_\Delta[t]-\mathbb E_\Delta[t^2].
}
\]

The aggregate risk is

\[
\frac{R}{R_0}=1-h\eta,
\qquad
h=\frac{\mathbb E\Delta}{R_0}=1-\frac1{G_s},
\]

where `G_s=R_0/R_*` is the source oracle gain.

### Pure sign with shrinkage

Let a sign predictor be correct with improvement-weighted probability `q`, and use amplitude `alpha>=0`:

\[
t=\alpha s,\qquad s\in\{-1,+1\}.
\]

Then

\[
\eta(\alpha)=2\alpha(2q-1)-\alpha^2.
\]

The optimal amplitude is

\[
\alpha^*=\max(0,2q-1),
\]

and the maximum retained fraction is

\[
\boxed{
\eta_{\max}=\max(0,2q-1)^2.
}
\]

Thus an odd feature with random phase (`q=1/2`) has zero optimal value. Any nonzero amplitude increases risk.

### Full-amplitude sign mistakes

For `alpha=1`,

\[
\eta=4q-3=1-4p,
\]

where `p=1-q` is the weighted wrong-sign rate. A full sign correction helps at all only when

\[
q>3/4.
\]

A wrong full sign changes the risk from `R0-Delta` to `R0+3Delta`; each wrong-sign unit costs four units of oracle improvement.

### Abstention

Let `c,w,a` be the improvement-weighted correct, wrong and abstain fractions, with `c+w+a=1`. At full amplitude,

\[
\boxed{
\eta=c-3w=1-a-4w.
}
\]

With a common optimized amplitude on accepted cases,

\[
\eta_{\max}=\frac{(c-w)^2}{c+w}
\]

when `c>w`; otherwise abstaining everywhere is optimal.

### Skeptical attack

Ordinary unweighted sign accuracy is the wrong metric. A method can be correct on many low-value networks and wrong on a few high-`Delta` networks, producing catastrophic tails. All phase accuracy, abstention and margin curves must be weighted by source-oracle improvement and also reported per base network.

---

## 9. Result P4.7 — source-capacity, nonlinear-replay and compute threshold

Let

- `G_s` be source oracle gain;
- `h=1-1/G_s` be its reducible risk fraction;
- `eta` be the retained linear oracle fraction after phase, amplitude and abstention errors;
- `nu=R_rem/R0` be a certified squared nonlinear-replay remainder ratio;
- `lambda` be the adjusted-score multiplier ratio caused by candidate compute relative to baseline compute, using the actual challenge accounting function.

The linearized risk ratio is

\[
r_{\rm lin}=1-\eta h.
\]

By Minkowski,

\[
\sqrt{r_{\rm exact}}
\le
\sqrt{1-\eta h}+\sqrt{\nu}.
\]

Therefore a sufficient adjusted-score win condition is

\[
\boxed{
\lambda
\left(\sqrt{1-\eta h}+\sqrt\nu\right)^2<1.
}
\]

Equivalently, when `lambda^{-1/2}>sqrt(nu)`, it is sufficient that

\[
\boxed{
\eta>
\frac{1-\left(\lambda^{-1/2}-\sqrt\nu\right)^2}{h}.
}
\]

For a raw target gain `G_t` with no nonlinear remainder and equal compute,

\[
\eta_{\rm req}
=
\frac{1-1/G_t}{1-1/G_s}.
\]

### Numerical consequences

#### Source gate exactly `1.20x`

Here `h=1/6`.

- To reach `1.05x`, `eta_req=2/7=0.285714`.
- To reach `1.10x`, `eta_req=6/11=0.545455`.
- To reach `1.30x` is impossible even with perfect coefficients, because the source ceiling is only `1.20x`.

For the `1.10x` policy gate:

- optimally shrunk pure-sign accuracy must satisfy
  \[
  q\ge\frac{1+\sqrt{6/11}}2\approx0.869274;
  \]
- full-amplitude sign accuracy must satisfy
  \[
  q\ge\frac{3+6/11}{4}\approx0.886364;
  \]
- equivalently the full-amplitude weighted wrong-sign rate must be at most `0.113636`, before any abstention, nonlinear remainder or compute penalty.

This explains why `1.20x` is a minimum source gate rather than a comfortable source gate.

#### Edge-DWS source `1.144709x`

Its reducible fraction is only about `0.126416`. Reaching `1.10x` would require

\[
\eta\ge0.719129,
\]

which in the optimally shrunk pure-sign model requires `q>=0.924007`. This mathematically explains why the frozen source leaves almost no model-error margin.

#### M161 source span

- Validation oracle ratio `0.400` gives `G_s=2.5`, `h=0.6`.
- Confirmation oracle ratio `0.454` gives `G_s≈2.20264`, `h≈0.546`.

At confirmation capacity, a `1.10x` target requires only `eta≈0.1665`, corresponding to optimally shrunk phase accuracy around `0.704`. This is the only currently documented source span with comfortable phase headroom, although its existing feature/ridge policy is closed and its five-source orientation structure must be stated exactly before any new run.

---

## 10. What the canonical observable should be

### Preferred construction A — no gauge, direct odd contraction

When a correction direction `u_j` already has a physical deterministic orientation from its construction, do not reorient it. Compute

\[
z_j=\langle u_j,s\rangle
\]

against a legal downstream-sensitive summary `s`, and feed the signed `z_j` directly into a bounded policy. This is continuous, odd and avoids the topological discontinuity because no sign section is formed.

The exact group action must be tested. Under deliberate representation reorientation `u_j -> -u_j`, the feature must reverse and the physical policy/dictionary composition must remain invariant.

### Preferred construction B — block polar gauge

When `U` is only a basis for a physical subspace and can rotate inside a nearly degenerate singular block, construct a legal reference frame `B`, form `C=U^TB`, and use the polar-aligned frame `U polar(C)`. Accept only with certified `sigma_min(C)` margin.

### Inferior but admissible comparator — positive pivot

The A50 largest-coordinate sign rule may remain a comparator if singular gaps and pivot gaps are both preregistered and the entire degenerate block is rejected. It should not be presented as a globally stable canonical gauge.

---

## 11. Legal reference candidates, proved and attacked

### 11.1 Fixed output coordinate or fixed output test vector

Example:

\[
z_j=q^T J u_j
\]

for a fixed canonical output vector `q` and legal downstream Jacobian `J`.

**Proof side:** this is orientation-odd in `u_j`, deterministic, and uses the fixed physical output coordinate system.  
**Attack:** one arbitrary `q` may be nearly orthogonal to the useful target phase. Multiple prespecified `q` vectors improve rank but consume the four-contraction budget.

### 11.2 Final-layer adjoint direction

Use a canonically defined adjoint, for example a fixed output aggregation pulled backward through the realized suffix.

**Proof side:** it is physically coupled to downstream sensitivity rather than a random gauge.  
**Attack:** if the output aggregation is arbitrary or if the adjoint itself comes from an unoriented singular vector, the ambiguity has only moved. Its computation cost must include the VJP or suffix pass.

### 11.3 Signed margin/gate-crossing contraction

Use a signed preactivation margin weighted by downstream effect.

**Proof side:** it is directly related to the nonlinear replay mechanism.  
**Attack:** near-zero margins are exactly where sign instability and ReLU crossing tails occur. This candidate needs the strongest abstention and T51 remainder accounting.

### 11.4 Random reference

**Proof side:** a continuous random distribution is full rank almost surely and fixes the algebraic gauge.  
**Disproof side:** conditional independence from target phase gives `q=1/2`, so the optimal correction amplitude is zero. Random symmetry breaking is not information.

---

## 12. Complete frozen falsification protocol

No empirical policy should run until the source basis is frozen and independently shows grouped oracle gain at least `1.20x` with safe tails. The currently documented five-source M161 oracle span passes the capacity condition; its old feature/ridge policy does not.

Freeze exactly one of the following:

1. one physically oriented source direction and one direct odd contraction; or
2. one rank-`k<=4` source block and one rank-`k` polar reference frame.

### Required pre-label declarations

- exact gauge group: global `Z_2`, independent signs, or `O(k)`;
- source construction and physical sign conventions;
- reference construction;
- odd feature equations;
- singular/pivot/cross-Gram acceptance margins;
- floating-point error bound;
- bounded calibration model and clipping;
- source oracle gain and tail statistics on an independent grouped cohort;
- complete operation and wall-time accounting;
- nonlinear replay remainder method, when applicable.

### Mandatory ablations

1. zero correction;
2. best matched global constant;
3. best bounded constant vector;
4. even M153 features only;
5. odd feature only;
6. even plus odd;
7. deliberately reoriented representation test;
8. random odd reference with matched marginal scale;
9. positive-pivot versus polar gauge, when a block basis is used;
10. oracle coefficient ceiling.

### Primary gates

Retain the A50 gates:

- source oracle gain `>=1.20x`;
- policy raw gain `>=1.10x`;
- grouped 95% lower bound `>1.05x`;
- worst candidate/base `<=1.25`;
- positive complete adjusted-score gain;
- beats matched constant and even-only policy;
- abstention `<=25%`;
- no base network contributes `>25%` of total gain.

Add theorem-derived gates:

- declare the improvement-weighted wrong-sign rate;
- verify `eta=2E_Delta[t]-E_Delta[t^2]` directly;
- accept only if the certified gauge margin exceeds numerical and physical perturbation bounds;
- for a rank-`k` basis, require cross-Gram rank `k` on every nonabstained case;
- report source-energy-weighted abstention and wrong-sign fractions, not only counts.

---

## 13. Final disposition

### Proved

1. Even quotient features cannot recover signed coefficients; one global odd channel is the minimal algebraic escape for a global sign gauge.
2. No globally continuous sign section exists in dimension at least two.
3. A local polar reference gauge is canonical, equivariant and unique exactly away from cross-Gram singularity.
4. Full `O(k)` gauge fixing requires a rank-`k` reference alignment; fewer than `k` independent anchor directions leave a stabilizer.
5. Gauge perturbation is controlled by the smallest singular value of the cross-Gram.
6. Source-span coefficient error transfers exactly through the physical source Gram norm.
7. Wrong sign is four times as costly as missed correct improvement; a random odd gauge has zero optimal value.
8. The `1.20x` source gate requires very high phase accuracy to reach the `1.10x` policy gate.

### Disproved or narrowed

1. A globally stable deterministic sign convention is impossible.
2. Singular-value ordering plus largest-coordinate sign is not globally stable and does not resolve rotations inside repeated singular subspaces.
3. “Mathematically odd” does not imply correlated with target phase.
4. Ordinary sign accuracy is not sufficient; improvement-weighted sign accuracy controls risk.
5. Four contractions cannot generically fix more than four independent sign characters or a full `O(k)` basis with `k>4`.

### Recommended next step

Do **not** open protected data. First freeze the exact group and source basis. If the M161 five-source span is used, determine whether its ambiguity is one shared global sign or several independent/source-basis gauges. Then implement only one theorem-derived construction:

- direct signed downstream contractions if the source columns are physically oriented; or
- rank-`k` polar alignment if the source is only an unoriented subspace.

A clean failure under the frozen protocol closes that orientation/source/policy class. A pass would be the first principled escape from T49.
