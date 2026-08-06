# WHestBench current state — v28 out-of-box agent consolidation

**Date:** 2026-07-30 22:22 ET  
**Protected or official cohort opened:** No  
**Canonical ledger:** `whestbench_canonical_research_ledger_20260730_reconciled_v28_out_of_box_agents.xlsx`  
**New reports consolidated:** Agents 1, 2, 5 and 6  
**New completed reports not located at cutoff:** Agents 3, 4, 7, 8, 9 and 10. Their absence is pending evidence, not a negative result. Agent 4’s proposed conditional-Gaussian-mixture concept is substantively covered by Agent 1.

## Executive conclusion

There is still **no deployable competition estimator**.

The newest agents do not leave ten independent paths. Their results collapse into:

1. **One primary constructive program:** a coherent heteroscedastic conditional-Gaussian latent state, with a full Gaussian covariance tail and a small number of PSD covariance prototypes.
2. **One possible computation/expectation engine:** environment-weighted compositional chaos for the exact degree-10 adjoint-harmonic expectation.
3. **One cheap, low-prior falsifier:** direct final-output QTT over the 16-bit Kerdock chirp index.
4. **One dormant class escape:** an exact nonlinear late-absolute-innovation identity outside the closed linear checkpoint-gauge class.

The broad original forms of pairwise sign-state propagation, continuous-input tensor trains and coordinate-sparse polynomial chaos are now closed or sharply narrowed.

The immediate research question is:

> Can a small conditional-covariance mixture represent the deep non-Gaussian copula accurately enough on held-out networks, and can the same state be generated legally and propagated through all 32 layers without oracle refresh?

The first gate is **M187**. No large implementation should precede it.

---

## 1. What changed relative to v27

V27 identified the missing object as a compressed, PSD, mutually realizable joint state. It still framed Tucker-compressed mixed moments as the main representation hypothesis.

V28 sharpens that statement:

- A collection of pairwise moment, sign or orthant matrices cannot be a complete recursive state.
- Independently transported cumulants and covariances risk reproducing the previous mean/covariance incompatibility.
- A coherent generative state is therefore not merely a numerical convenience; it is the natural way to satisfy realizability.
- The most evidence-consistent generative family is a **conditional covariance mixture**, not a location-only mixture and not a single Gaussian with additive Edgeworth corrections.

The canonical primary representation is now

\[
H\sim\nu,\qquad z\mid H=h\sim N(m(h),R(h)),
\]

with a practical finite approximation

\[
p(z)\approx\sum_{q=1}^{K}\pi_q N(m_q,S_q),
\]

and structured covariance modulation

\[
S_q\approx\sum_{h=1}^{H}\theta_{qh}P_h,
\qquad P_h\succeq0.
\]

The plausible envelope proposed by Agent 1 is approximately

\[
K=16\text{–}32,\qquad H=4\text{–}6,
\]

with a full shared covariance tail retained rather than discarded.

---

## 2. Agent 2: pairwise sign geometry is not a recursive state

### Exact closure result

T89 supplies a universal counterexample. Two three-coordinate parity laws have identical:

- every one- and two-coordinate sign law;
- every pairwise positive-orthant probability;
- every requested pairwise truncated first and product moment;
- the complete ReLU mean and covariance;
- the mean and variance of a selected next preactivation.

After the same dense linear map and ReLU, their next expectations are nevertheless

\[
\frac34\quad\text{and}\quad\frac12.
\]

Therefore no universally exact recurrence can use only pairwise sign distributions and pairwise truncated moments as its state.

This closes as complete recursive states:

- orthant-probability matrices;
- sign-correlation matrices;
- pairwise truncated moment matrices;
- low-rank logit/probit factorizations of those matrices;
- PSD projection without a generative law;
- sign-only Ising or dichotomized-Gaussian corrections;
- generic pair regression.

This does **not** close coherent latent distributions, approximate ensemble-specific closure or late contraction-only use.

### Exact current-layer interface

T90 gives

\[
E[z_+]=\frac12(E[z]+E|z|)
\]

and

\[
E[z_+z_+^\top]
=\frac14\left(M+A+B+B^\top\right),
\]

where

\[
M=E[zz^\top],\qquad
A=E[|z||z|^\top],\qquad
B=E[z|z|^\top].
\]

Once ordinary covariance is matched, the remaining pair problem is carried by weighted absolute kernels and sign–magnitude coupling. Raw orthant probability may therefore be the wrong headline statistic.

### Canonical effect

Agent 2 is no longer an independent sign-kernel branch. It supplies:

- the nonclosure theorem that forces a coherent joint law;
- the exact weighted-kernel diagnostic interface;
- M187’s attribution protocol.

---

## 3. Agents 1 and 2 merge into the primary latent covariance-mixture program

### Why covariance mixing fits the evidence

A random-correlation Gaussian mixture can have:

- exactly Gaussian marginals;
- zero third cumulants;
- a non-Gaussian copula;
- a biased ReLU pair moment under single-Gaussian closure.

For zero-mean unit-variance pairs, the ReLU kernel is strictly convex in correlation. Consequently covariance variation produces a systematic pair-moment correction even when the matched covariance and all marginal third-order diagnostics look benign.

The mixed fourth cumulant obeys

\[
\kappa_{22}=2\operatorname{Var}(R).
\]

This reconciles the latest calibrated third-order failure with the earlier M40 result:

- additive third order is insufficient at depth;
- mixed fourth-order information can still be decisive;
- a conditional covariance state is more natural than a pure latent-location state.

### Exact legal recurrence

For each Gaussian component, calculate exact conditional ReLU mean and covariance

\[
\alpha_q=E[z_+\mid q],\qquad C_q=\operatorname{Cov}(z_+\mid q).
\]

For the next linear layer,

\[
c_q=W^\top\alpha_q,\qquad V_q=W^\top C_qW.
\]

Project the conditional transformed law to

\[
y\mid q\approx N(c_q,V_q).
\]

This recurrence:

- uses no target labels;
- preserves each component’s conditional first and second moments under the current mixture model;
- preserves unconditional first and second moments;
- maintains PSD;
- has fixed component count if labels are kept persistent.

The unresolved error is the repeated Gaussian projection **inside** each component. That projection can erase newly generated non-Gaussianity, so a theorem-level recurrence is not yet an accuracy result.

### Cost envelope

Exact component-specific dense covariance transport is likely over budget for moderate K. Covariance prototypes reduce the expensive transports to H shared matrices.

The report estimates that K approximately 32 with H=4 could plausibly occupy roughly 16.9–23.5B FLOPs before compression overhead; H=6 is near the 27.2B boundary. These are envelopes, not profiler measurements.

All CDF evaluation, prototype fitting, eigensolvers, merging, basis construction and memory-accounted operations must be charged.

---

## 4. M187 is now the decisive first experiment

M187 combines the useful parts of Agents 1 and 2.

### Stage A: exact downstream attribution

Decompose the Gaussian pair-closure residual into:

- orthant probability;
- positive conditional means;
- positive conditional covariance/product;
- marginal ReLU mean error;
- weighted absolute-kernel terms.

Use downstream next-variance and adjoint-weighted attribution. Do not rank mechanisms using unweighted pair Frobenius error.

If raw orthant probability explains less than half of the downstream residual, stop calling the branch orthant transport.

### Stage B: nested latent oracle

Use independent fit and evaluation halves. Compare:

- H0: latent location mixture with one common full covariance;
- H1: nonlinear center with common covariance;
- H2: latent center plus 2–8 PSD covariance prototypes;
- H3: unrestricted conditional covariance only as a representation ceiling.

The primary promotion gate is:

\[
\text{next-variance RMS}\le0.3\%
\]

with approximately

\[
K\le32,\qquad H\le6,
\]

plus downstream-contraction, PSD and tail requirements.

### Kill conditions

Close or sharply demote the branch if:

- even a restricted held-out H2/H3 oracle remains above approximately 0.5%;
- only sample-identifying or effectively unrestricted bins pass;
- the representation passes only after target-oriented factor selection;
- the covariance-prototype count or effective latent dimension becomes uneconomic.

M187 is currently **unrun**. The project has a theorem-backed hypothesis, not evidence that it works on WHestBench distributions.

---

## 5. M188 is the actual candidate gate

Run only if M187 passes with material slack.

### Required construction

- initialize from the exact Gaussian first preactivation;
- use only weights and previous legal state;
- retain full covariance tail;
- propagate fixed K and H through all layers;
- start with persistent labels;
- use score-aware covariance-prototype compression;
- permit refreshed split–merge only after diagnosing why persistent labels fail;
- never refresh from reference activation samples.

### Final promotion requirements

\[
\text{raw MSE}\le2.962\times10^{-7},
\]

\[
\text{effective compute fraction}\le0.1,
\]

with grouped network/rotation holdouts, safe tails, no hidden PSD repair and no oracle factor orientation.

The actual target is full-rollout final score. One-step pair accuracy is not sufficient.

---

## 6. Agent 5: tensor integration mostly closes, leaving one bounded QTT audit

### Continuous-input functional tensor trains

Agent 5 proves a generic Cartesian separation-rank obstruction. For a balanced coordinate split, a generic width-256 shallow ReLU ridge sum already has cross-Hermite rank 128 almost surely.

At the same time, the 27.2B budget permits only roughly 6,500–10,200 full network queries, limiting a black-box 256-dimensional functional TT to ranks around two or three before realistic cross overhead.

Therefore close as primary routes:

- Cartesian functional TT/HT;
- ordinary random rotations;
- first-layer ridge coordinates as a global solution;
- weight-product singular coordinates as an assumed compression basis;
- layerwise shared-output tensor propagation.

### The Kerdock-index loophole

The 65,536 chirp nodes form an exact 16-bit Boolean tensor. This makes direct final-output QTT the only tensor formulation with overlapping query and rank economics.

However, T94 establishes a severe nonlinear densification result:

- one first-layer scalar preactivation has 256 Walsh atoms;
- its square has exactly 32,641 nonzero sign-even frequencies out of 32,768.

This is not itself a QTT-rank theorem, but it removes sparse frequency support as the hoped-for mechanism.

### M189

Use stored full-Kerdock arrays to measure:

- scalar TT/QTT ranks under several bit orderings;
- first-layer z, |z| and ReLU matricization ranks;
- whether ranks around 12–16 suffice at mean-relevant tolerances;
- whether one common pivot/query set works for all 256 outputs;
- legal final-output cross reconstruction and complete query union.

This is a cheap falsifier and should be run once. Close Agent 5 if:

- median scalar rank exceeds approximately 12–16;
- common pivots do not serve the output vector;
- query union or residual-block variance misses the score budget.

Do not build a large tensor optimizer before this audit.

---

## 7. Agent 6: conventional sparse PCE closes; contracted compositional chaos remains

### Structural closure

For generic dense continuous weights, every coordinate of every allowed nonzero Hermite tensor degree is nonzero almost surely after one ridge-ReLU layer. Coordinate support is also unstable under function-preserving rotations.

Close:

- sparse global Hermite multi-index recovery in original coordinates;
- ANOVA or low interaction order as the main compression;
- support selection from baseline point values without an oracle ceiling;
- explicit enumeration of global chaos tensors through degree 20 or higher.

### Exact surviving object

Positive homogeneity yields an exact state equation. Write

\[
h_l=r_ls_l,
\]

and define the radius-weighted directional measure

\[
\nu_l(\varphi)=E[r_l\varphi(s_l)].
\]

Then

\[
\nu_{l+1}(\varphi)
=\nu_l(q_{l+1}\,\varphi\circ F_{l+1}).
\]

The exact degree-10 adjoint-harmonic expectation reduces to contractions of six normalized homogeneous tensor orders

\[
Q_{l,k}=E\left[\frac{h_l^{\otimes 2k}}{\|h_l\|^{2k-1}}\right],
\qquad k=0,\ldots,5,
\]

against legal downstream query tensors.

This is not a closed finite recurrence. The remaining question is whether only the **environment-weighted contractions** have low legal ranks.

### M190

Run:

- global degree-20-and-higher relevance check;
- environment-weighted CP/Tucker/TT/tree rank ceilings;
- function-preserving rotation tests;
- oracle versus weight/adjoint-derived orientation;
- free 32-layer rank-growth and error-budget rollout.

Continue only if low downstream ranks survive legal orientation and fit the tight degree-10 expectation budget.

### Relationship to the primary path

Agent 6 should not become a second independent global state program. It is best viewed as:

- a possible compressed expectation engine inside the conditional latent state; or
- a direct engine for the separate winning-scale degree-10 adjoint-harmonic oracle.

---

## 8. Unified architecture of the surviving research program

The strongest synthesis is:

### Representation layer

A coherent latent conditional-covariance distribution supplies:

- mutually realizable means, covariances and signs;
- PSD automatically;
- mixed fourth-order information through covariance modulation;
- a full Gaussian tail.

### Diagnostic interface

T90’s weighted absolute kernels identify which sign–magnitude mechanism matters and provide the correct one-step attribution.

### Compression layer

Use score-aware covariance prototypes or environment-weighted tensor contractions. Do not compress by global Frobenius energy alone.

### Dynamics layer

The projection recurrence must be tested freely through 32 layers. Oracle refits are forbidden.

### Final objective

Optimize and evaluate final raw MSE and complete adjusted score, not pair reconstruction, likelihood or tensor energy.

---

## 9. Canonical priority order

### Priority 1 — M187

Run the exact attribution and H0/H2 oracle comparison immediately.

### Priority 2 — M188

Only after M187 passes, implement the simplest persistent legal mixture recurrence.

### Priority 3 — M190 in parallel

Measure whether the exact degree-10 contracted expectation has low legal environment rank. Fold a positive result into the primary program or the adjoint-harmonic direct source.

### Priority 4 — M189 once

Run the existing-array Kerdock QTT rank/common-pivot audit. Close quickly on failure.

### Priority 5 — nonlinear late innovation hedge

Continue only if an exact/shared-arithmetic identity is actually derived. The linear checkpoint-gauge class remains closed.

### Separate proof lane

Continue degree-62 actual-width static theory for publication/class closure, not as competition implementation work.

---

## 10. Current closed-route list

### Closed or stopped as competition routes

- static network-independent node/weight tuning;
- ordinary additional rows and multifidelity sampling;
- low-rank activation MLMC and particle truncation;
- linear checkpoint-gauge estimation for the direct-output PCA source;
- generic phase and coefficient learning on tested features;
- additive third-order single-Gaussian Edgeworth at depth;
- pairwise sign/orthant/truncated matrices as recursive states;
- sign-only Ising and Gaussian-copula sign models;
- continuous Cartesian or ordinary rotated functional TT/HT;
- layerwise shared-output tensor trains;
- coordinate-sparse and ANOVA polynomial chaos;
- independent fitting of mean, covariance and cumulant states;
- protected evaluation before a legal full rollout.

### Open only under narrow gates

- conditional covariance mixture: M187 then M188;
- environment-weighted compositional chaos: M190;
- direct final-output Kerdock QTT: M189 once;
- exact nonlinear late absolute innovation;
- degree-62 finite-width proof program.

---

## 11. Where this leaves the competition

The project is neither at a candidate nor at universal impossibility.

The best current statement is:

> The original sampling/cubature box and several natural analytic approximations are closed. The remaining competition-relevant possibility is a coherent, weight-derived, nonperturbative joint state whose latent covariance modulation captures the mixed fourth-order/copula defect and whose compressed recurrence survives free depth.

This is more concrete than v27 because:

- the state family is explicitly defined;
- a legal PSD recurrence exists;
- pairwise-state insufficiency is proved;
- the likely missing mechanism is named;
- accuracy, rank, complexity and kill gates are numerically bounded.

But it remains speculative because the decisive WHestBench oracle and legal rollout have not been run.

The protected cohort should remain sealed.

---

## 12. Canonical IDs added in v28

### Theorems and exact identities

- **T89:** pairwise sign/truncated-moment state nonclosure.
- **T90:** weighted sign–magnitude and absolute-kernel ReLU identities.
- **T91:** fixed-component conditional-Gaussian moment recurrence.
- **T92:** random-correlation covariance-mixture obstruction to single-Gaussian closure.
- **T93:** continuous-input TT separation-rank/economic obstruction.
- **T94:** Kerdock-index nonlinear Walsh densification theorem.
- **T95:** coordinate-sparse PCE density and rotation obstruction.
- **T96:** radius-weighted directional-measure transfer identity.

### Open experiments

- **M187:** weighted attribution and H0-versus-H2 latent oracle.
- **M188:** persistent legal mixture free rollout.
- **M189:** Kerdock-index QTT rank and legal-cross audit.
- **M190:** environment-weighted degree-10 rank/legal-orientation audit.
- **M191:** v28 cross-agent canonical synthesis.

