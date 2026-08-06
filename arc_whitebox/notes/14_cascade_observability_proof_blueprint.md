# 14 — Cascade / Observability Proof Blueprint

**Date:** 2026-07-30
**Purpose:** Executable blueprint for proving — or constructively disproving — that legal
estimators can approach the layer-31 oracle headroom via cascading improvements from
earlier layers. Written for an executor model to follow without re-deriving the theory.
**Both outcomes are wins.** Impossibility ⇒ the paper's central theorem. Disproof ⇒ a
concrete improvement prototype (each test's FAIL branch says exactly what to build next).

**How to use this document:**
1. Read §1 (notation) and §2 (claim registry). Every claim has a status tag and a width tag.
2. Execute tests in the §5 order (cheapest-decisive first). Never skip a test's gate.
3. Every empirical claim must follow the governance rules in §7 (frozen cohorts, no
   Mini-100 validation, noise-corrected targets, grouped rotations).
4. When a test FAILS its impossibility gate, go to §4 (improvement exits) — do not
   treat it as an error.

---

## 0. Thesis

The production estimator (complete 66,048-row Kerdock/MUB cubature, uniform weights,
positive-homogeneity radial integration) is provably the optimal *fixed linear* rule on
its own support at **every width** (C2, symmetry proof — new result, no computation
needed). At infinite width it is furthermore the exact Bayes rule for its information,
and adaptive point selection provably cannot help (C3, C4). Therefore every legal
improvement — including all cascading/intermediate-layer schemes — must live in exactly
one place: **the exploitable fraction γ(width) of the non-Gaussian finite-width sector
of trajectory statistics**, whose total size is O(depth/width) ≈ 12% at width 256
(matching the measured T20 multiplier 1.12). The program: prove everything outside γ
closed (mostly done or short), then measure γ directly with a width-scaling probe
(TEST-2, the crux). If γ is small ⇒ observability-gap theorem: 78% oracle headroom,
≤ γ legally reachable. If γ is material ⇒ the probe's own fitted correction is the
improvement prototype.

---

## 1. Notation and objects

- Ensemble: challenge networks, width n = 256, depth L = 32, weights iid Gaussian
  (He scaling assumed — **verify in TEST-0**), believed bias-free (**verify**).
- γ_in: known input Gaussian. Estimand θ = E_{x~γ_in}[f(x)] (final output mean,
  per output coordinate). Score = ensemble-mean MSE ⇒ all optimality statements are
  **average-case over the ensemble** (Bayes risk against the generator prior).
- Design X = {x_j}, N = 66,048 Kerdock/MUB points (129 orthonormal bases × 512
  antipodal vectors). Baseline estimate m̂ = uniform cubature + exact radial part.
- Defect e = m̂ − θ. Current relative center error ≈ 0.65%; layer-31 oracle removes
  ≈ 78% of noise-corrected final MSE; anchor tolerance: ≤ 0.3% useful, ≈ 0.45%
  break-even (M83 curve).
- Correction-risk identity (already proved in this program): for correction d with
  scale α, R(α) = R₀ − 2α·E⟨e,d⟩ + α²·E‖d‖². Maximum improvement over α equals
  ρ² · R₀ where ρ = corr(e, d). **ρ² ("exploitability") is the universal currency of
  this blueprint**: every scheme's possible value is its out-of-sample ρ².
- Information scopes (every claim carries one):
  - **S1** — algorithms using final-output values f(x_j) at ≤ N points.
  - **S2** — S1 + full intermediate trajectories at those points (all cascade,
    fold, selector, learned, and anchor schemes live here).
  - **S3** — algorithms using raw weights analytically without point evaluation
    (moment propagation, Gaussian-line integration). Excluded from theorems;
    empirically catastrophic (M01 ~270×, C24 ~6081× worse); see C10.
- Width tags: [∞] = infinite-width limit (GP model), [fw] = finite width 256.

---

## 2. Claim registry

Status legend: **PROVEN-ROUTE** (proof is short and specified; write it up),
**DONE** (already proved in this program), **CONDITIONAL** (theorem modulo a
measured input), **ASSUMPTION+TEST** (stated assumption with decisive test),
**EXCLUDED** (outside scope, documented).

### C1 — Cascade normal form. [∞, fw] PROVEN-ROUTE
With e_ℓ the state defect at layer ℓ, A_ℓ the linearized transfer, η_ℓ the fresh
per-layer cubature injection: e_{ℓ+1} = A_ℓ e_ℓ + η_ℓ with **e₀ = 0** (input known
exactly). Telescoping: e_31 = Σ_ℓ Φ_{31,ℓ} η_ℓ + remainder, remainder controlled by
the existing ReLU crossing lemma.
**Consequence:** the baseline already *is* the perfect cascade from a flawless early
node. "Cascading improvements" can only (i) shrink injections, (ii) cancel
accumulated error via an anchor, (iii) reallocate along transfer weights. Channel
(ii) recurses into (i)+(iii). No new mathematics needed beyond bookkeeping + the
crossing lemma; write as Lemma 1.

### C2 — Uniform weights are optimal at every width (symmetry theorem). [∞, fw] PROVEN-ROUTE — **key new result**
Proof (three steps, all exact):
1. Weights iid Gaussian ⇒ the ensemble is exactly rotation-invariant at every
   width ⇒ prior mean of f is 0 and the covariance kernel K_fw(x,y) is **zonal**
   (depends only on ⟨x,y⟩) — no infinite-width limit required.
2. The Kerdock support is antipodal and every point has the identical
   |inner-product| multiset to the rest of the design: {1 self, 1 antipode,
   0 ×510 own-basis, 1/16 ×65,536 cross-basis} (T22 verified Gram multiplicities).
   Antipodality pairs t with −t, so for ANY zonal kernel the Gram row sums are
   constant: Σ_j K(t_ij) = Σ_pairs [K(t)+K(−t)] depends only on the |t| profile.
3. z_i = ∫K(⟨x_i,y⟩)dγ_in(y) is constant in i (rotation-invariant γ_in, zonal K).
   Constant row sums + constant z ⇒ under the total-mass constraint, uniform
   weights exactly minimize the average-case quadratic risk w'Kw − 2w'z + c.
**Consequences (state as corollaries):** every fixed reweighting, fold-reweighting,
jackknife-weight, basis-reweighting, and partial-design-reweighting scheme is closed
**at finite width**, in one stroke. This subsumes T05/T26 and explains the fold-family
failures without the common-bias heuristic. Data-dependent weights w(data) are NOT
covered — they are nonlinear overall and belong to the γ sector (C9).
**Loophole left open deliberately:** the global output scale α (total mass) — see
TEST-1b; it is the only linear degree of freedom on the support and is improvement
candidate #1.

### C3 — Bayes optimality and zero alignment. [∞ only] PROVEN-ROUTE
At infinite width the ensemble is a GP with the compositional kernel (T01/T02
object). For Gaussian priors and a linear estimand, the Bayes rule given point
evaluations is the linear kernel-quadrature rule; by C2 that is the uniform rule
(up to α). Hence baseline = posterior mean ⇒ E[e | S1-information] = 0 ⇒ for ANY
S1-measurable correction d, E⟨e,d⟩ = 0 exactly ⇒ (risk identity) zero expected
improvement. **This converts the ~25-family neutrality from an empirical pattern
into a theorem prediction at [∞]** — the measured correction cosines (0.059, 0.066,
Spearman ≈ 0.05) are its confirmation (TEST-6 formalizes).
**[fw] status:** FAILS deliberately — the finite-width prior is non-Gaussian, so
nonlinear processing may beat the linear rule. That gap IS γ (C9). Do not claim C3
at finite width.

### C4 — No free adaptation. [∞ only] PROVEN-ROUTE (citation + adaptation)
Classical IBC theorem (Traub–Wasilkowski–Woźniakowski; Wasilkowski 1986): for linear
problems under Gaussian measures, adaptive information of fixed cardinality n has the
same average-case radius as nonadaptive; and the optimal algorithm is linear in the
data. With T13/T22 (certified: Kerdock within 0.0234% of the optimal nonnegative
static rule at N = 66,048): at [∞], **no adaptive, cascaded, bootstrapped, or
nonlinear evaluation-based scheme with ≤ N points beats the baseline by more than
the certified slack + signed-weight slack (C11)**, on average. Caveats to state:
fixed cardinality (budget-fixed ⇒ fine); Gaussian measure ([∞] only); estimand is
the OUTPUT functional (see guardrail G3). At [fw], adaptivity/selection value is
part of γ — C42's inaccessible 1.79× oracle rotation gap is the empirical face of
this.

### C5 — Static design floor. [∞] DONE
T13/T22/T23: certified ratio ≤ 1.0002336550102949 among fixed network-independent
nonnegative-weight rules, clean-room reproduced. Cite; do not rework.

### C6 — Transfer structure: what persists and what decays. [fw exact + measured] PROVEN-ROUTE + TEST-3
(i) **Exact scale-mode marginality:** for a bias-free ReLU network, scaling the
layer-ℓ activations by c > 0 scales the output by exactly c (positive homogeneity;
also the variance map is exactly marginal at He scaling). Relative scale errors
transfer with factor exactly 1 — at every width, no linearization. One-paragraph
theorem after TEST-0 confirms bias-free.
(ii) **Center-shift modes are expected to CONTRACT:** linearized transfer of a pure
center shift is diag(Φ(μ/σ))W; for centered preactivations E[Φ²]·σ_w² = ¼·2 = ½
per layer. At late layers with collapsed covariance (effective rank ~3, M13) gates
are near-deterministic and the factor can approach or exceed 1 — genuinely
depth-dependent ⇒ MEASURE (TEST-3), don't assert.
**Consequences:** (a) no compounding mechanism for early-node improvements — returns
are additive shares at best; (b) if center modes contract, early center fixes decay
and the surviving layer-31 defect is dominated by scale/variance common modes —
exactly the components fold/jackknife diagnostics cannot see (this connects C6 to
the existing common-bias non-identifiability theorem, and matches M10 "sigma
dominant"). This answers ledger row 42's request: the depth-flat relative error is a
theorem for the scale sector, an assumption+measurement for the rest.

### C7 — Early-node ceiling (attribution + coherence). [fw] CONDITIONAL on TEST-4
For any scheme whose modifications are confined to layers ≤ k, the final-RMSE
improvement is capped by the prefix attribution share π(≤k) (coherent case) or
1 − √(1 − π(≤k)) (incoherent). Archived shares (re-measure before quoting: they are
branch-specific synthetic evidence): last 8/16/24 transitions carry
23.6%/33.8%/82.5% of the signed defect ⇒ layers 1–8 carry ≈ 17.5% ⇒ **a perfect
oracle prefix through layer 8 cannot reach break-even** (0.54–0.59% residual vs
0.45% break-even), under either coherence model. Useful-anchor depth window: k ≳ 16.
TEST-4 re-measures shares + coherence under the frozen protocol and turns this into
a conditional theorem with published measured inputs.

### C8 — Self-anchoring recursion (synthesis). [∞ theorem, fw modulo γ]
Combining C1–C7: legal state error at every depth k is ≥ (1 − γ_total) × the
accumulated floor at k; the anchor-replacement condition (anchor must beat the
defect it replaces) therefore never triggers at any depth; cascading cannot
bootstrap. Final statement: **every legal estimator improves on the baseline by at
most γ_total = certified static slack (0.0234%) + signed-weight slack (C11) +
measured γ(256) (TEST-2)**. The 78% oracle headroom minus γ_total is the
observability gap.

### C9 — The γ sector (the single live quantity). [fw] ASSUMPTION+TEST (crux)
Definition: γ(n) = max over S2-measurable corrections d of ρ²(e, d) at width n —
the exploitable fraction of MSE. Theory: the non-Gaussian/finite-width sector has
total size O(L/n) (finite-width NNGP corrections scale as depth/width; L/n = 1/8
matches the measured T20 multiplier 1.122). Exploitability within the sector is
NOT derivable cheaply — measure it (TEST-2). Heuristic side-information argument
to include in the paper: each intermediate coordinate couples to the output at
O(1/√n), so trajectory information beyond output values vanishes as n → ∞; γ(n) → 0.
Known empirical lower bounds on γ(256): tested-family cosines ≈ 0.06 ⇒ ρ² ≈ 0.4%
(nearly nothing exploited so far — but tested features may have been the wrong ones,
which is exactly why the disproof branch is live and interesting).

### C10 — S3 exclusion (weight-analytic algorithms). EXCLUDED, documented
No theorem claimed. Document: unconditional impossibility over S3 would be a
compute lower bound (out of reach). Evidence of closure-error regeneration: M01
(270×), C24 (6081×), M44 rollout instability, M16 handoff failure. Optional
rigorous exemplar if desired: M14's rank obstruction (covariance rank ~3 but κ₃
requires rank ~64 ⇒ any rank-r closure has an error floor) — a self-contained
lemma, medium effort, not required for the main theorem.

### C11 — Signed-weight slack. [∞] OPEN proof task (bounded)
The C5 certificate covers nonnegative weights; the Bayes weights (C3) on the
Kerdock support are uniform by C2, so **the signed loophole on the OWN support is
closed**. Off-support signed rules: quantify via the existing derived bound
E_K(w) ≥ c₀ + q(1)/N − 2Mβ(1+β) by interval-certifying M = sup(K − h). This is the
one remaining certificate computation (TEST-7). Until done, report the T19 numeric
evidence (best signed found: none better; d=4 search) as the bound's empirical face.

---

## 3. Test registry

Format — **Purpose / Procedure / Gate / Cost / If PASS / If FAIL**. "PASS" always
means "supports impossibility"; "FAIL" is the improvement branch (§4). Cost tiers:
**T0** pure math/derivation (no compute), **T1** trivial numerics (minutes,
existing artifacts), **T2** moderate (existing harnesses, small ensembles),
**T3** the one big experiment.

### TEST-0 — Prior and architecture verification. Cost T1. BLOCKING.
Purpose: pin the assumptions every [fw] claim uses.
Procedure: inspect the challenge generator (not the networks): confirm (a) weights
iid Gaussian, He/2-fan-in scaling; (b) NO biases; (c) width 256 / depth 32; (d) the
official ensemble and the synthetic generator agree in these respects.
Gate: all four confirmed.
If FAIL (biases exist): C6(i) becomes first-order instead of exact; C2 survives
(rotation invariance of weight prior is enough IF the input-layer distribution is
still rotation-invariant — re-derive with biases); note in every claim.

### TEST-1a — Row-sum constancy (C2 verification). Cost T1.
Purpose: independent check of the |t|-profile step of C2.
Procedure: from the stored design (or T22 Gram multiplicity artifacts), verify each
point's |inner-product| multiset equals {1, 1, 0×510, (1/16)×65,536}. Spot-check
~1,000 random points directly if the artifact is not trusted.
Gate: exact equality for all checked points.
If FAIL: C2's step 2 is wrong for this design — check antipode pairing conventions;
the theorem still holds for whatever the true constant-profile orbit structure is.

### TEST-1b — Global scale α (improvement candidate #1). Cost T1 [∞] / T2 [fw].
Purpose: the only on-support linear degree of freedom C2 leaves open.
Procedure [∞]: with the harmonic machinery compute c = Σ_t K(t)·mult(t) (Gram row
sum) and z̄ = ∫K(⟨x,y⟩)dγ_in; α_∞ = N·z̄/c. Procedure [fw]: estimate the two
scalars E[m̂·θ] and E[m̂²] over an existing synthetic ensemble with reference
targets (they are exactly the moments needed: α_fw* = E⟨m̂,θ⟩/E‖m̂‖²); noise-correct.
Gate: |α − 1| ≤ 1e-6 [∞] and CI containing 1 [fw].
If FAIL: **free improvement found.** Fixed global shrinkage α* on the final
estimate; validate on a fresh frozen cohort with the standard tail gates; expected
gain from the risk identity. (Check the ledger first for an explicit prior test of
plain global output scaling; none was found in this review — the tested alphas were
correction-scales, not output mass.)

### TEST-2 — Width-scaling exploitability probe (THE CRUX). Cost T3.
Purpose: measure γ(n) and its scaling — decides between the impossibility theorem
and the improvement program. This is the experiment that answers "is there an
algorithmic way to estimate the intermediate layers."
Procedure:
1. Widths n ∈ {64, 128, 256, 512}; fresh synthetic ensembles, networks per width
   scaled to compute (e.g. 256/192/128/96); high-quality reference targets;
   noise-corrected metrics; grouped rotations. New immutable splits (governance).
2. For each network run the width-scaled baseline and record a FIXED, predeclared
   dictionary of legal trajectory features (reuse existing extractors): per-basis
   partial means, fold/jackknife disagreements, per-layer gate fractions, radial
   moments, Stein-flux stats, low-q harmonic features, per-layer scale statistics
   (C6 says the scale sector is where persistent signal must live — include it
   deliberately), plus per-network selector-style summary stats.
3. Target: signed defect e per output coordinate (and its projection on the K32
   anchor directions as a secondary target).
4. Fit ridge (and one gradient-boosted variant as a nonlinearity check) with
   grouped CV across networks and rotations. Report out-of-sample ρ̂², noise-
   corrected, per width. Fit ρ²(n) = c·n^(−p).
Gates: **PASS (impossibility)** if p ≥ ~0.8 AND ρ̂²(256) ≤ 1%. **FAIL (improvement,
good news)** if ρ̂²(256) ≥ 3% with stable feature attribution, or if ρ²(n) plateaus
in n.
If PASS: γ(256) ≤ ~1%; C8's γ_total is pinned; write the theorem.
If FAIL: the fitted correction IS the prototype — see §4-E2. Its feature attribution
says which observable carries absolute phase; hand that to a targeted estimator with
the standard frozen-validation pipeline (candidate/base ≤ 0.595 raw, safe tails,
positive adjusted score).
Stage the experiment: run n ∈ {64, 128} first (cheap); only proceed to 256/512 if
ρ̂² at small width is materially above noise (if there is no signal at width 64,
where the sector is 4× larger, there is nothing at 256 — early stop saves most of
the cost).

### TEST-3 — Mode-resolved transfer spectrum. Cost T2 (scale-mode part T1).
Purpose: C6 inputs; kills or confirms "compounding".
Procedure: on a few networks: (i) multiply the propagated cloud at layer ℓ by
(1+ε), ε = 1e-3, for each ℓ; verify final relative response = ε to machine
precision (exact homogeneity unit check — near-free); (ii) inject center shifts
(random directions and along μ) and covariance-shape perturbations at each ℓ;
record final response norms ⇒ per-layer, per-mode transfer factors.
Gate: scale mode exactly 1 (machine precision); cumulative products of measured
per-layer factors bounded ≤ ~1.5 for all modes (no compounding window).
If FAIL (some mode family amplifies substantially): early injections in that mode
are levered — cascading via that mode is live; combine with TEST-4 attribution to
locate it, then treat as §4-E4.

### TEST-4 — Oracle-depth ladder + coherence matrix. Cost T2.
Purpose: measured inputs for C7; the paper's Figure 1.
Procedure: k ∈ {0,4,8,12,16,20,24,26,28,30,31}. Operational oracle swap (keep ONE
fixed definition across k, matching the M83-style perturbation): replace the
propagated mean (and covariance, as a second arm) at layer k with reference values;
propagate legally onward; record noise-corrected final MSE ratio r(k) and
per-network signed contribution increments between consecutive k. Coherence matrix
= cross-k correlation of increments (decides amplitude-vs-variance sharing in C7).
Gate/sanity: r(31) reproduces the known ≈ 0.22 within CI; r(k) monotone.
Outputs: measured shares π, coherence, the useful-anchor depth window k*.
If shares differ materially from the archived 23.6/33.8/82.5 profile: use the new
numbers everywhere; C7's conclusion only strengthens if early shares shrink.

### TEST-5 — Marginal value of evaluations vs budget. Cost T1 (tier a) / optional T3 (tier b).
Purpose: price every "more/other points" scheme, including shared-arithmetic
companions, against the information floor.
Tier (a): assemble EXISTING numbers — M07 sigma-sampling sweep (1.83%/0.89%/0.55%/
0.22% at 6k/25k/67k/400k: confirms MC scaling; 0.45% needs ≳ 10⁵ independent
samples), M41 moment curve, companion-pilot results (2,064 pts < 1% headroom),
A48 basis economics, A43 per-point FLOP cost — into one marginal-value vs
marginal-cost curve. Gate: marginal value < marginal cost for all reachable N.
Tier (b, optional, only if the paper needs it rigorous): recompute the Delsarte-LP
lower bound at a grid of N values with the T13 machinery.
If FAIL (some N regime has value > cost under shared-arithmetic discounting):
§4-E3 — a companion design in that regime is worth one frozen experiment.

### TEST-6 — Zero-alignment meta-audit of the archived corpus. Cost T1.
Purpose: C3's prediction, confirmed retrospectively; the paper's falsification-map
figure.
Procedure: collect every archived legal-correction family's measured alignment
(cosines 0.0586, 0.0655, sign accuracies 30/50, Spearmans ≈ 0.05, neutral ratios
1.0008–1.014, …) from the ledger/JSONs into one table with Ns; test consistency
with alignment = 0 (within noise) per family and pooled; funnel plot.
Gate: pooled result consistent with 0.
If FAIL (some family has replicated nonzero alignment): that family found γ-sector
signal — promote it to §4-E2 immediately (this outcome contradicts its recorded
neutral deployment, so first re-check compute adjustment and tails explain the gap).

### TEST-7 — Interval-certified M for the signed-weight bound (C11). Cost T2 (proof-side).
Purpose: close the last certificate loophole quantitatively.
Procedure: interval-certify M = sup(K − h) with the existing directed-rounding
toolchain; plug into E_K(w) ≥ c₀ + q(1)/N − 2Mβ(1+β); state the resulting
quantitative signed-weight statement.
Gate: certified M finite and small enough that material signed-weight gains require
implausibly large negative mass β.
If FAIL: report the loophole honestly as unquantified; the theorem scope already
excludes it explicitly.

---

## 4. Improvement exits (the "good news" branches)

- **E1 — Global scale α ≠ 1 (TEST-1b).** Free scalar improvement; validate on a
  frozen cohort; ship if tail-safe. Cheapest possible win.
- **E2 — γ(256) material (TEST-2 FAIL, or TEST-6 FAIL).** The probe's fitted
  correction is a working prototype and its feature attribution identifies the
  first legal observable carrying absolute phase. Next: freeze the top features,
  build the targeted estimator, standard pipeline (frozen validation, candidate/base
  ≤ 0.595 raw for 70% oracle retention, safe tails, positive adjusted score). This
  would reopen the layer-31 branch with, for the first time, a measured signal
  instead of a hoped-for one.
- **E3 — Cheap evaluations regime (TEST-5 FAIL).** A shared-arithmetic companion
  design in the identified N regime; must reuse ≥ ~98% of prefix arithmetic per the
  existing companion-economics closure.
- **E4 — Amplifying transfer mode (TEST-3 FAIL).** An early-layer correction
  targeted at the amplifying mode family has leverage; cross-reference TEST-4
  attribution for where to apply it.
- **E5 — Contracting-everything regime (TEST-3 shows strong contraction of ALL
  modes).** Then late injections dominate ⇒ a suffix-only companion estimator
  could substitute for full-depth propagation — revisit the suffix-economics
  numbers with the measured contraction profile.

---

## 5. Execution order

1. TEST-0 (blocking, minutes).
2. TEST-1a, TEST-1b[∞], TEST-6, TEST-5a — all T1, one session, no new ensembles.
3. TEST-3 (scale-mode exactness first — it is a unit test), then full spectrum.
4. TEST-4 ladder (reuses oracle-replay infrastructure).
5. TEST-1b[fw].
6. TEST-2 staged: widths 64/128 first; early-stop rule as specified.
7. TEST-7 and TEST-5b only if the paper's referees need them.

Write-up can begin after step 2: C1, C2, C3, C4, C5 and the C6 scale-mode theorem
are provable with zero new experiments; TEST-4/TEST-2 outputs drop in as figures.

---

## 6. Paper mapping

| Paper section | Content | Source |
|---|---|---|
| 1. Setup + certified baseline | Existing theorem, scope | C5 (T13/T22) |
| 2. Optimality of the baseline at all widths | Symmetry theorem + α | C2, TEST-1 |
| 3. Why corrections fail: exact identities | Risk identity, zero alignment [∞], no-adaptation | C3, C4, TEST-6 (funnel fig) |
| 4. The cascade question | Normal form; transfer structure; attribution ladder | C1, C6 (TEST-3 fig), C7 (TEST-4 fig = Fig 1) |
| 5. The finite-width sector | γ definition, L/n scaling, probe result | C9, TEST-2 (headline empirical fig) |
| 6. Economics | Marginal value of evaluations vs budget | TEST-5 fig |
| 7. Scope + falsification map | S3 exclusion, signed slack, 25-family table | C10, C11, ledger distillation |
| 8. Conclusion | Observability-gap statement (or the E2 discovery) | C8 |

One-paragraph conclusion template (impossibility branch):
> The baseline cubature is the exact optimal fixed linear rule on its support at
> every width (symmetry), the exact Bayes rule at infinite width, and adaptive
> evaluation cannot beat it on average (no-free-adaptation + certificate). All
> remaining legal headroom therefore lies in the exploitable fraction of the
> finite-width non-Gaussian trajectory sector, of total size ≈ depth/width ≈ 12%,
> whose measured exploitable part is γ(256) = [TEST-2]. The layer-31 oracle
> headroom of ≈ 78% minus γ is an observability gap, not a computational or
> geometric one: closing it requires qualitatively new absolute-phase information,
> not further transformation of the same cloud.

---

## 7. Guardrails for the executor model

- **G1.** Tag every claim [∞] or [fw]. Never state C3 or C4 at finite width.
- **G2.** Never call depth-flat relative error a theorem except for the exact
  scale-mode statement (C6-i), and only after TEST-0 confirms bias-free.
- **G3.** Route all impossibility through the OUTPUT functional. Do not apply
  GP/kernel-quadrature/no-adaptation arguments to intermediate-layer estimands
  (layer-31 activations are ReLU-of-GP, not GP; the argument is invalid there).
- **G4.** Archived attribution shares (23.6/33.8/82.5) and the 0.526 checkpoint
  near-miss are branch-specific synthetic evidence — re-measure (TEST-4) before
  putting numbers in the paper.
- **G5.** Governance: fresh immutable cohorts for every new empirical claim;
  Mini-100 and IDs 0–199 and 1000–1023 are exposed, development-only; protected
  holdouts stay unopened; noise-correct all ratios; group rotations; predeclare
  gates before running.
- **G6.** Report failures as findings. Every FAIL branch in §3 is an §4 entry,
  not an error state.
- **G7.** Do not start new estimator searches outside the §4 exits. The broad
  search is closed; this blueprint is the proof-and-measurement program.
- **G8.** The theorem paper cites: Neal / Lee et al. / Matthews et al. (NNGP
  limit), Traub–Wasilkowski–Woźniakowski and Wasilkowski 1986 (average-case
  no-adaptation), plus the program's own T13/T22 certificate. Verify the exact
  no-adaptation statement (fixed cardinality, Gaussian measure, linear problem)
  before relying on it.
