# WHestBench paper strengthening patch

**Date:** 2026-07-30  
**Role:** Agent 10 continuation / paper architecture and open-end closure  
**Disposition:** Strengthen and proceed, with T16 promoted after independent interval audit; keep M146 and M152 out of the evidence-bearing core unless their raw artifacts are restored.

## 1. Revised contribution hierarchy

The paper should make four contributions, in this order.

### Contribution 1 — certified static boundary

For the dimension-256, depth-32 infinite-width normalized ReLU kernel, the complete 66,048-point Kerdock/MUB rule is at most `0.02336550102949%` above the infimum over network-independent, nonnegative-weight linear cubature rules with at most 66,048 nodes.

This is a one-sided computer-assisted theorem. It does not establish that Kerdock is genuinely suboptimal.

### Contribution 2 — exact optimality of the auxiliary certificate

The KKT-selected degree-5 auxiliary is not merely a low-degree approximation. The previous dual certificate proved strict negative reduced costs for every omitted degree. The new primal–dual closure constructs the exact Hermite primal at the three algebraic dual nodes, proves its feasibility through positivity of `K32^(6)`, and obtains exact complementary slackness.

Therefore the degree-5 auxiliary is the unique optimizer of the unrestricted all-degree Delsarte auxiliary LP.

This closes the “higher certificate degree” objection. It does **not** close the difference between the best auxiliary lower bound and the true cubature optimum.

### Contribution 3 — a restricted signed/support theorem

Within the fixed 33,024 antipodal Kerdock-line universe, arbitrary real weights and arbitrary deletion patterns cannot improve over complete bases plus at most one partial basis. This is exact under the stated infinite-width, static, linear, symmetrized model.

Outside that universe, the signed-weight stability lemma is quantitative but too loose to close arbitrary signed cubature.

### Contribution 4 — theory-guided falsification of adaptive correction

The finite-width correction program is not one impossibility theorem. It is a structured falsification map:

- correction benefit depends on signed downstream alignment;
- common-bias observations do not identify absolute phase under an explicit model;
- unweighted layer-31 error is not a universal gate;
- named low-degree and homogeneous one-layer controls are exactly annihilated;
- a frozen small high-degree harmonic dictionary failed;
- grouped scalar-learning models failed to demonstrate network-specific value in the archived Path-2 information class;
- coreset, companion, and moment-closure families repeatedly failed their preregistered deployment gates.

The correct conclusion is that no **tested active branch** clears a credible continuation gate—not that every adaptive estimator is impossible.

---

## 2. Recommended title

### Primary

**A Certified Boundary for Neural Cubature: Kerdock Near-Optimality and the Limits of Cheap Adaptive Correction**

### More mathematical

**Near-Optimal Kerdock Cubature for a Deep ReLU Kernel and Exact Optimality of Its Delsarte Certificate**

### Broader contribution framing

**Static Neural Cubature Is Nearly Exhausted: Certified Bounds and a Falsification Map for Adaptive Correction**

---

## 3. Replacement abstract

Estimating neural-network expectations under Gaussian input is a high-dimensional integration problem whose practical objective couples statistical error to evaluation cost. We study a width-256, depth-32 bias-free ReLU benchmark and separate two questions: how much improvement remains for static cubature under the corresponding infinite-width kernel, and whether cheap network-specific observables can unlock a useful finite-width correction.

After radial reduction, we give a computer-assisted Delsarte certificate for the normalized depth-32 ReLU kernel on the sphere. It proves that the 66,048-point Kerdock/real-MUB rule is at most `0.02336550102949%` above the infimum among network-independent nonnegative-weight linear rules with the same point budget. We further prove that the KKT-selected degree-5 auxiliary is the unique optimizer of the unrestricted all-degree auxiliary linear program: a three-node algebraic dual measure has strict negative reduced cost at every omitted harmonic degree, while the corresponding Hermite primal is globally feasible because the kernel has positive sixth derivative. Inside the fixed Kerdock-line universe, an exact association-scheme reduction also rules out improvements from arbitrary deletion patterns or signed line weights.

These static results do not cover finite-width, adaptive, signed arbitrary-node, or nonlinear estimators. We therefore develop exact correction-risk, replacement, common-bias non-identifiability, and ReLU gate-crossing results, and use them to organize a preregistered falsification program. Named low-degree and homogeneous one-layer controls vanish exactly; a frozen small degree-6/8 harmonic dictionary fails; and grouped scalar-learning models show no network-specific value beyond constant shrinkage in the archived information class. The resulting contribution is a certified static boundary together with a scoped map of which adaptive correction mechanisms were tested, why they failed, and what evidence would justify reopening them.

---

## 4. Revised paper structure

1. **Problem, score, and radial reduction**  
   Define the finite-width competition objective separately from the limiting-kernel theorem.

2. **Kernel discrepancy formulation**  
   Derive the ensemble-MSE identity only for rules independent of the realized random field.

3. **Computer-assisted Kerdock near-optimality**  
   State T22 with explicit node, weight, independence, dimension, depth, and infinite-width scope.

4. **All-degree optimality of the auxiliary LP**  
   Present the new T16 primal–dual closure. Distinguish optimality of the certificate from optimality of Kerdock.

5. **Restricted Kerdock-line optimization and signed stability**  
   Present T27 first; relegate the globally loose negative-mass curve to a stability lemma.

6. **Correction theory**  
   Correction-risk identity, constrained selector, general replacement formula, downstream-weighted replacement gate, correlated-noise shrinkage, common-bias theorem, and gate-crossing remainder.

7. **Empirical anchor geometry**  
   Lead with reproducible T4 evidence. Treat M146 as provisional unless restored.

8. **Harmonic-control taxonomy**  
   Exact class-specific annihilation, limiting-kernel oracle shares, frozen dictionary failure, and the Poisson-kernel counterexample.

9. **Learning and observability**  
   Exclude M152 from evidence. Include independently reproduced Path-2 results and comparisons to same-mean constants.

10. **Normalized falsification map**  
    Put anchors, companions, coresets, moment closures, learning, harmonics, and implementation economics in one table with evidence levels.

11. **Scope, reopening conditions, and reproducibility**  
    Explicitly state what remains open, release hashes, AI assistance, and protected-cohort governance.

---

## 5. Paper-ready theorem table

| Claim | Recommended status | Exact scope | Paper action |
|---|---|---|---|
| T22 Kerdock near-optimality | **COMPUTER-ASSISTED CERTIFIED** | `d=256`, depth-32 infinite-width normalized ReLU kernel; at most 66,048 nodes; nonnegative mass-one weights; rule independent of field | Keep as headline theorem; use canonical one-sided JSON |
| T16 all-degree reduced costs | **PROVED** | All normalized Gegenbauer degrees `>=6` for the named dual measure | Merge into all-degree auxiliary theorem |
| T16 primal attainment/complementarity | **NEW: PROVED UNDER EXPLICIT INTERVAL TRUST BASE** | Unique degree-5 Hermite primal, unrestricted all-degree auxiliary LP | Add theorem and appendix; require hostile second-stack audit before final release |
| T27 line-universe theorem | **PROVED UNDER EXPLICIT MODEL** | Static linear rules on fixed 33,024 symmetrized Kerdock lines, arbitrary real mass-one line weights | Keep; prohibit arbitrary-node extrapolation |
| Signed negative-mass lemma | **PROVED / COMPUTER-ASSISTED CONSTANT** | At most 66,048 consolidated nodes, total negative mass `beta`, limiting kernel | Appendix stability result; explicitly call weak for practical closure |
| Correction-risk identity | **PROVED** | Hilbert-space quadratic correction model | Main correction-theory section |
| Full replacement gate | **PROVED UNDER EXPLICIT SUBSPACE MODEL** | Downstream operator `J`; gate uses `E||J xi||^2 < E||J d||^2` | Replace scalar relative-error threshold |
| Common-bias non-identifiability | **PROVED UNDER EXPLICIT OBSERVATION MODEL** | Same shared absolute bias across folds/blocks | State as model theorem, not universal impossibility |
| ReLU crossing remainder | **PROVED / BOUNDED** | Named replay map and gate-crossing event | Appendix and correction diagnostics |
| M146 precision curve | **PROVISIONAL EXPLORATORY** | Reported 60-network perturbation experiment; raw package absent | Remove headline numbers or visibly mark unverified |
| Path-2 grouped learning audit | **FROZEN EMPIRICAL, REPRODUCED** | Archived Path-2 features/models and grouped splits | Keep narrow negative result |
| M152 | **OPEN / UNVERIFIED** | Claimed 1,100-network corpus absent | Remove from paper evidence and quantitative conclusions |
| Low-degree/Hermite/Stein annihilation | **PROVED, CLASS-SPECIFIC** | Exact radialization and named function classes | Keep narrow lemmas |
| Frozen degree-6+8 dictionary | **FROZEN EMPIRICAL — FAILED** | Four-direction selected dictionary on frozen 16-network panel | Keep as failed tested rule, not class theorem |

---

## 6. Replacement T16 manuscript subsection

### Exact optimality of the all-degree auxiliary certificate

A natural objection to a degree-5 Delsarte certificate is that higher harmonic coefficients might tighten the lower bound. The dual reduced-cost calculation addresses only one side of this question. Let

\[
q_0=1-1/N,\qquad q_\ell=-1/N\quad(\ell\ge1),
\]

and let \(\mu=\sum_{j=1}^3\lambda_j\delta_{t_j}\) be the positive dual measure whose support consists of the roots of

\[
22102t^3+21930t^2-87t-85=0.
\]

Exact finite arithmetic and an analytic Gegenbauer tail bound prove

\[
q_\ell-\int G_\ell\,d\mu<0\qquad(\ell\ge6).
\]

To establish primal attainment, define \(h_*\) as the degree-5 Hermite interpolant satisfying

\[
h_*(t_j)=K_{32}(t_j),\qquad h_*'(t_j)=K_{32}'(t_j).
\]

Interval enclosure certifies that all nonconstant normalized-Gegenbauer coefficients of \(h_*\) are positive. Moreover, a Faà di Bruno decomposition proves \(K_{32}^{(6)}>0\) on `(-1,1)`. The only potentially negative Bell term is bounded by one quarter of the leading term, while an interval recurrence gives \(F''/F'<9/4\) for the outer 31-fold composition on the relevant range. The Hermite remainder formula therefore yields

\[
K_{32}(t)-h_*(t)
=\frac{K_{32}^{(6)}(\xi)}{6!}\prod_{j=1}^3(t-t_j)^2\ge0.
\]

Finally, moment matching and contact give exact primal–dual equality:

\[
\sum_{\ell=0}^5q_\ell c_\ell
=\sum_{j=1}^3\lambda_j h_*(t_j)
=\sum_{j=1}^3\lambda_jK_{32}(t_j).
\]

Thus \(h_*\) is the unique optimizer of the unrestricted all-degree auxiliary LP. This proves that the remaining `0.0233655%` certificate gap is not an artifact of truncating the auxiliary at degree five.

**Scope sentence:** This theorem establishes optimality of the auxiliary lower-bound certificate, not exact optimality of Kerdock cubature.

---

## 7. Replacement correction-theory framing

Avoid “circularity” and “one theorem with three faces.” Use:

> The adaptive program encounters three complementary obstructions with different logical status. First, correction risk depends on signed downstream alignment, not merely correction magnitude. Second, under a common-bias observation model, folds and nested blocks reveal dispersion but not absolute phase. Third, in the tested feature classes, grouped learning did not recover stable network-specific sign or scale. The first two are exact model results; the third is empirical and representation-dependent.

The invariant replacement gate is

\[
\mathbb E\|J\xi\|^2<\mathbb E\|Jd\|^2,
\]

not a universal threshold on unweighted layer-31 relative error. Any scalar tolerance must be labeled direction- and cohort-specific.

### M146 wording

Use only:

> A reported perturbation curve is arithmetically consistent with a quadratic risk model and has a fitted break-even near `5.8e-4`, but the original 60-network row-level package and perturbation manifest were not located. We therefore treat the numerical threshold as provisional and do not use it as a universal continuation gate.

### Legal empirical anchor evidence

Use the reproducible frozen T4 result:

- raw candidate/base ratio `1.127854`;
- `17/48` wins;
- worst ratio `2.480711`;
- per-rotation positive oracle `0.915133`;
- one shared coefficient vector per base network across rotations `1.019612`.

Interpretation:

> Useful directions exist conditionally, but tested observables do not recover a stable absolute signed phase across rotations. This closes the tested policy, not all possible independent phase observables.

---

## 8. Replacement harmonic section framing

Use a taxonomy, not a universal claim.

1. Exactly radialized angular polynomials through degree five are integrated exactly by complete Kerdock.
2. Polynomial Stein fields with component degree at most four are annihilated.
3. Bias-free one-hidden-layer ReLU Stein fields are annihilated blockwise under the named exact-radialization construction.
4. Degree-6/8/10 zonal controls are live. Their limiting-kernel shares are oracle diagnostics, not finite-width measurements.
5. The frozen four-direction degree-6+8 rule scored `1.004439`; later shrinkage found `0.999876`, effectively switching it off.
6. Analytic integrability does not imply low harmonic degree; include the symmetrized Poisson-kernel counterexample.

Recommended paragraph:

> Complete Kerdock exactly removes several explicitly named low-degree and homogeneous one-layer control families. Higher harmonics remain live: in the limiting kernel, degrees 6, 8, and 10 carry substantial orthogonal error shares. A frozen small degree-6+8 zonal dictionary nevertheless failed to validate. These results exclude the named constructions, not general nonpolynomial, biased, deep, adaptive, or otherwise rich high-degree controls.

---

## 9. Replacement learning section framing

M152 must not appear as citable evidence. Replace it with the independently reproduced Path-2 audit:

> In the archived Path-2 information class, grouped linear and small nonlinear models did not demonstrate network-specific scalar predictability. Every learned model was matched or beaten by a constant equal to its own mean prediction; apparent holdout gains came from moving the average shrinkage toward a more aggressive constant. Row-wise validation was materially optimistic when rotations were not grouped. These results invalidate the tested feature-dependent models but do not prove that no scalar predictor exists.

Always compare against:

- the development-optimal constant;
- the safe constant frontier;
- a constant equal to the model’s mean prediction;
- grouped-by-base-network cross-validation.

---

## 10. Main negative-results table

| Family | Strongest legitimate result | Status | What is closed | What remains open |
|---|---|---|---|---|
| Static nonnegative designs | Kerdock within `0.0233655%` | Certified theorem | Static positive class at limiting kernel scale | Finite width, adaptive, signed arbitrary-node, nonlinear |
| Higher-degree auxiliary | Degree-5 certificate is unique all-degree LP optimum | New computer-assisted theorem | Higher certificate degree | True cubature optimum may still exceed certificate |
| Kerdock-line signed weights | Complete bases plus partial basis globally optimal | Exact restricted theorem | Fixed line universe | Nodes outside universe |
| Arbitrary signed weights | Quantitative negative-mass lower bound | Proved but loose | Stability near zero negative mass | Competition-scale signed closure |
| Layer-31 anchors | Downstream-weighted gate; tested T4 policy harmful | Theory + frozen empirical | Named policies and scalar universal gate | New independent absolute-phase observable |
| Harmonic controls | Exact named annihilation; frozen small dictionary failed | Mixed theorem/empirical | Named low-degree and one-layer classes; selected dictionary | Rich high-degree controls |
| Scalar learning | Path-2 feature deviations add no demonstrated value | Frozen reproduced | Tested grouped model families | New corpus/representation with clean labels |
| M152 | Raw corpus absent | Unverified | Nothing evidentiary | Reproduce or remove permanently |
| Coresets | Deployable support scorers miss oracle-support gate | Frozen/exploratory campaigns | Tested representations | Qualitatively new set-level representation |
| Moment closures | Oracle moments help; legal estimates miss precision/economics | Mixed | Tested closure recurrences | New independently anchored state estimator |

---

## 11. Limitations section

The central theorem concerns an explicit infinite-width kernel, not the realized width-256 network distribution. Finite-width experiments support the relevance of the kernel geometry but do not transfer its global optimality statement to finite networks.

The near-optimality theorem requires nonnegative mass-one weights and a rule independent of the realized network. It does not cover adaptive support, adaptive weights, pilot-dependent selection, nonlinear postprocessing, analytic-plus-residual estimators, or arbitrary signed rules.

The exact Kerdock-line theorem permits signed weights but only on a fixed symmetrized line universe. It cannot be extrapolated to arbitrary spherical nodes or unpaired evaluations.

The adaptive correction results combine theorems under explicit observation or linearization models with finite empirical campaigns. Failures of tested anchors, harmonic dictionaries, learning models, coresets, and moment closures do not constitute a universal no-free-lunch theorem.

The M146 perturbation package and the claimed M152 corpus were not found in the shared archive. Their headline values must remain provisional or be removed. The paper should not use missing artifacts to support quantitative thresholds or learning-impossibility claims.

The new T16 primal–dual closure uses a second interval-arithmetic trust base (`mpmath.iv`) plus exact rational Bernstein checks. Before final publication, it should be independently reproduced using the existing directed-Decimal stack or another interval library.

---

## 12. Explicit reopening conditions

Reopen an adaptive branch only when it supplies a genuinely new information source or representation and freezes:

- exact target equation and sign convention;
- base-network and rotation IDs;
- independent reference streams or target-noise estimates;
- legal runtime features and extraction code;
- grouped splits fixed before label inspection;
- constant and safe-frontier baselines;
- row-level predictions, wins, p90, worst, grouped intervals, and hashes;
- complete deployment compute and wall-time accounting.

Reopen M146 only after restoring the original IDs, perturbation directions, seeds, exact means, replay code, and row-level metrics.

Reopen M152 only after locating the claimed corpus and manifest or constructing a new preregistered corpus. A transcript summary is not evidence.

---

## 13. Claims that must not appear

- “Kerdock is exactly globally optimal.”
- “The theorem covers width-256 networks.”
- “The theorem covers signed or network-adaptive estimators.”
- “Higher-degree controls cannot improve the true cubature optimum.”
- “Anything analytically integrable is low degree.”
- “The whole Stein family vanishes.”
- “Degree-6+ controls cannot help.”
- “Only degree 6 remains.”
- “A `5e-4` layer-31 error threshold is universal.”
- “M152 proves scalar learning is impossible.”
- “No statistical path exists.”
- “Three faces are one theorem.”
- “The manifest is immutable” without an externally anchored digest.
- Any positive lower bound on Kerdock’s actual suboptimality.

---

## 14. Residual open-end register

| Open end | Current disposition | Required action |
|---|---|---|
| T16 primal attainment/complementarity | **Closed in new package under explicit trust base** | Independent second-stack audit |
| T22 stale two-sided machine-readable artifact | **Closed by canonical JSON and validator** | Replace release artifact; externally anchor digest |
| T23 release provenance | Mostly closed | Pin environment; CI on second platform; clarify 32+23 coverage |
| M146 reproducibility | Blocked | Restore raw package or remove quantitative threshold |
| M152 reproducibility | Blocked | Restore corpus/script/manifest or exclude entirely |
| Finite-width transfer theorem | Open | Do not make it central; future research |
| Arbitrary signed-node optimality | Open; global lemma weak | State loophole explicitly |
| Independent absolute-phase observable | Open | Reopen only with a new legal observable and preregistered gate |

---

## 15. Final editorial recommendation

The strongest paper is not a universal impossibility result. It is a theorem-led paper with a disciplined empirical second half:

> We certify that static nonnegative cubature is essentially exhausted under the limiting deep-ReLU kernel, prove that the auxiliary certificate itself cannot be tightened by adding harmonic degrees, and then identify—through exact correction theory and frozen falsification—which tested information classes fail to unlock a cheap network-specific correction.

That claim is novel, defensible, and substantially stronger than either a narrow certificate paper or a catalog of failed experiments alone.
