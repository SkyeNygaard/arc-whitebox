# Agent 10 — Integrated Paper Architecture Review

**Project:** ARC White-Box Estimation Challenge 2026 / WHestBench  
**Date:** 2026-07-30  
**Role:** Paper architect reviewing Agents 1–8  
**Overall disposition:** **PROCEED WITH A SCOPED PAPER, AFTER MANDATORY EVIDENCE AND RELEASE CLEANUP**

## Executive decision

The eight reviews support a coherent and potentially strong paper, but not the broad “one theorem explains why no statistical path exists” narrative.

The defensible paper has three layers:

1. **A certified static-cubature boundary.** For the dimension-256, depth-32 infinite-width normalized ReLU kernel, the full Kerdock rule is within `0.0233655%` of the optimum among fixed or network-independent, nonnegative, mass-one rules using at most 66,048 nodes.
2. **Exact structural extensions and limits.** Inside the fixed 33,024-line Kerdock universe, arbitrary signed line weights do not improve over complete bases plus at most one partial basis. The all-degree dual reduced-cost tail is now proved negative. A general signed-weight stability lemma is valid but too weak to close arbitrary signed cubature.
3. **A theory-guided falsification program for cheap white-box correction.** Correction-risk, replacement, common-bias non-identifiability, and ReLU-crossing results explain what a successful correction must recover. Existing anchor, scalar-learning, and small harmonic-control campaigns fail in their tested information classes. These failures are complementary evidence, not a universal impossibility theorem.

The paper should make the static theorem the primary contribution. The finite-width correction program is a second contribution: an unusually careful map of what was tested, what failed, why the failures are structurally related, and exactly what remains open.

Two claimed empirical pillars must be removed from the main evidence chain unless their missing artifacts are restored:

- **M146:** the reported `41.2x` exact-anchor gain and approximately `5e-4` tolerance curve are not independently reproducible, and the scalar tolerance is false as a universal direction-independent statement.
- **M152:** the claimed 1,100-network scalar-predictability experiment has no located rows, target definition, feature list, grouping manifest, predictions, script, or hashes.

## Review of Agents 1–8

### Agent 1 — T22 mathematical review

**Verdict:** Accept after two corrections.

The mathematical argument is valid within its explicit class. The agent independently checked the ensemble-MSE/kernel-discrepancy identity, Delsarte lower bound, diagonal residual term, node-budget inequality, Kerdock multiplicities, spherical-mean logic, and final one-sided inequality.

Required changes:

- Remove or replace the stale machine-readable artifact that falsely suggests a positive lower bound on Kerdock suboptimality. The theorem gives a one-sided upper bound on the relative gap; it does not prove Kerdock is strictly suboptimal.
- State randomized-rule coverage only for admissible random rules independent of the sampled random field/network, almost surely satisfying the node, weight, and mass constraints.

**Paper role:** Main theorem and theorem-scope box.

### Agent 2 — T22/T23 certificate and reproducibility audit

**Verdict:** Accept as a rigorous computer-assisted proof after release-hygiene corrections.

The certificate was reproduced across 32 manifest-tracked outputs and 23 intermediate curvature chunks. The exact-rational witness, directed rounding, interval coverage, derivative/curvature signs, endpoint handling, kernel-mean enclosure, and outward-rounded ratio were all checked.

Required release changes:

- Replace “31 files” by “32 manifest-tracked files.”
- Distinguish canonical manifest outputs from the 23 regenerated but individually untracked intermediate chunks.
- Replace “immutable manifest” by “fixed during verification” unless the archive digest is externally authenticated.
- Pin the software environment and add a cross-version/OS CI matrix.
- Describe the result as **computer-assisted certified**, not proof-assistant formalized.

**Paper role:** Reproducibility section, trust-base appendix, and artifact-release checklist.

### Agent 3 — T27 restricted Kerdock-line theorem

**Verdict:** Accept as a major exact theorem under an explicit model.

Within the fixed 33,024 symmetrized antipodal Kerdock lines, arbitrary real weights summing to one are optimized by complete orthonormal bases plus at most one partial basis, with equal positive weights within active bases and positive analytically determined basis masses. This rules out deletion-pattern, unequal-weight, and signed-weight improvements **inside that universe**.

It does not cover:

- arbitrary spherical nodes;
- unpaired or unequally paired antipodal points;
- finite-width width-256 networks;
- nonlinear estimators;
- network-dependent support or weights.

**Paper role:** Second main theorem. It is the cleanest response to the signed-weight loophole that remains inside the actual Kerdock support family.

### Agent 4 — T16 all-degree reduced-cost theorem

**Verdict:** Accept the reduced-cost theorem; do not yet claim full all-degree LP optimality without the final primal link.

The review proves every unused normalized Gegenbauer reduced cost is strictly negative:

- exact integer arithmetic for degrees `6..14,658`;
- an analytic `ell^-127` tail bound for all degrees at least `14,659`.

The remaining distinction is important. “All unused reduced costs are negative” is proved. “The selected degree-5 auxiliary is exactly the all-degree LP optimum” still requires an explicit exact primal-feasibility, attainment, and complementarity link to the named dual measure.

**Paper role:** Proof machinery supporting T22, preferably in the theorem section or technical appendix. Promote it to a headline theorem only after the primal link is formalized.

### Agent 5 — arbitrary signed-weight stability

**Verdict:** Include as a scoped lemma, not as closure of signed cubature.

The signed-weight lower bound is correct and strengthened. The residual supremum is certified exactly at `t=1`, and the integer support-count envelope improves the diagonal term. However, the resulting numerical exclusion is practically weak: even a 10% Kerdock-relative improvement requires only about `7.13e-7` total negative mass.

This means the bound is mathematically useful but competition-level vacuous outside the Kerdock-line universe. The paper must not present it as proving arbitrary signed rules cannot improve.

**Paper role:** Stability subsection or appendix, directly followed by an explanation of why the bound is loose. T27, not this lemma, should carry the main signed-weight message.

### Agent 6 — layer-31 correction theory and precision audit

**Verdict:** Accept the theory; downgrade the empirical precision curve.

The following belong in the paper under explicit assumptions:

- correction-risk identity;
- constrained selector theorem;
- general replacement formula;
- correlated-noise shrinkage;
- common-bias non-identifiability;
- ReLU gate-crossing and nonlinear-margin bounds.

The invariant replacement gate is downstream weighted:

`E ||J xi||^2 < E ||J d||^2`.

A universal threshold in unweighted relative layer-31 mean error does not exist. The break-even level varies with downstream singular direction and with proximity to final-layer ReLU kinks.

M146 is not reproducible from the shared archive. Its reported points are internally consistent with a quadratic perturbation model, but that consistency cannot establish the cohort, perturbation directions, or validity of the original experiment.

**Paper role:** The conceptual bridge from the static theorem to the empirical correction program. The paper should emphasize downstream-weighted phase recovery, not a universal `5e-4` scalar threshold.

### Agent 7 — scalar-learning reproduction and skeptical audit

**Verdict:** Exclude M152; include the reproduced Path-2 negative result.

M152 is not citable. The underlying corpus and experiment definition are absent.

The archived Path-2 substitute is independently reproducible and gives a useful narrow result: for the tested grouped linear and small nonlinear models, feature-dependent network-specific scale variation did not add value over appropriate constant controls. Apparent gains were explained by shifting the mean prediction toward a more aggressive global constant. Row-wise validation was materially optimistic relative to base-network-grouped validation.

This invalidates scalar predictability for the tested models and information class. It does not prove no scalar predictor exists.

**Paper role:** Main empirical learning result. The key baseline must be a constant equal to the model’s own mean prediction, alongside the safe and development-optimal constant frontiers.

### Agent 8 — harmonic-control taxonomy

**Verdict:** Accept after scope corrections.

The harmonic story should be a taxonomy:

- exactly radialized angular polynomials through degree 5 are annihilated by complete Kerdock;
- polynomial Stein fields with component degree at most 4 are annihilated;
- bias-free one-hidden-layer ReLU Stein fields are exactly annihilated blockwise under the stated radialization and antipodal-basis construction;
- small degree-6/8/10 zonal dictionaries are live but failed in tested forms;
- general analytically integrable controls remain open.

The symmetrized spherical Poisson kernel is an explicit counterexample to “analytically integrable implies low harmonic degree.”

**Paper role:** Exact no-op lemmas followed by a carefully separated frozen high-degree negative experiment and an explicit non-implication counterexample.

## Recommended paper thesis

> We certify a near-optimality boundary for static nonnegative cubature under a deep ReLU kernel, prove a stronger exact support-allocation theorem inside the Kerdock line universe, and develop a theory-guided falsification map showing why several cheap finite-width white-box corrections fail to recover the required signed downstream defect—without claiming universal adaptive impossibility.

This is more defensible and more interesting than a generic “negative results” paper. It combines a rigorous positive theorem, a structural restricted theorem, and a disciplined empirical account of the remaining adaptive gap.

## Alternative titles

1. **A Certified Boundary for Static Neural Cubature and the Limits of Cheap White-Box Correction**
2. **Near-Optimal Kerdock Cubature for Deep ReLU Kernels: Certificates, Signed Boundaries, and Falsified Corrections**
3. **What Static Cubature Can—and Cannot—Do for White-Box Neural Expectation Estimation**

Recommended default: **Title 1**. It foregrounds the strongest theorem while leaving room for the finite-width falsification program.

## Draft abstract

Estimating Gaussian expectations of deep neural activations is expensive, even when all network weights are available. We study deterministic white-box estimators for bias-free ReLU multilayer perceptrons, with emphasis on a 66,048-point Kerdock construction used for width-256, depth-32 networks. After radial reduction, the infinite-width ensemble error is an RKHS cubature discrepancy for an explicit depth-32 spherical ReLU kernel. We give a computer-assisted certificate showing that the complete Kerdock rule is within `0.0233655%` of optimal among all fixed or network-independent nonnegative mass-one rules using at most 66,048 nodes. We further prove that, among arbitrary real-weight linear rules supported on the fixed 33,024 antipodal Kerdock lines, the optimum for every support budget consists of complete orthonormal bases plus at most one partial basis, with positive equal weights within each active basis. An all-degree reduced-cost theorem closes the dual harmonic tail, while a general signed-weight stability bound is shown to be too weak to exclude arbitrary signed-node improvements.

The static theorems do not cover finite-width, nonlinear, or network-adaptive corrections. We therefore derive correction-risk, replacement, common-bias non-identifiability, and ReLU-crossing results that identify the missing object as a signed, downstream-weighted network-specific defect. We then audit a broad correction program. Several low-degree and homogeneous one-layer control families are exact Kerdock no-ops; a frozen small high-degree zonal dictionary fails; and grouped scalar-learning models do not outperform matched constant controls in the reproducible archived information class. Missing artifacts prevent two previously reported anchor and scalar-learning experiments from serving as evidence. The result is a scoped boundary: static nonnegative neural cubature is essentially solved for the limiting kernel, while adaptive finite-width improvement remains open but constrained by precise theoretical and empirical failure modes.

## Recommended manuscript structure

### 1. Problem, estimator classes, and claim taxonomy

Define the finite-width challenge first, then separate four estimator classes:

1. static nonnegative linear cubature;
2. static signed linear cubature;
3. network-dependent linear rules;
4. nonlinear or analytic-plus-residual estimators.

Introduce the evidence labels used throughout: proved, computer-assisted certified, proved under explicit model, frozen empirical, exploratory empirical, oracle diagnostic, open, and invalidated.

### 2. Radial reduction and deep-ReLU kernel cubature

Derive the spherical problem and the ensemble-MSE/RKHS discrepancy identity. State clearly that the kernel theorem is an infinite-width ensemble statement, not a finite-width width-256 theorem.

### 3. T22: certified near-optimality of complete Kerdock

Present the theorem, proof outline, certified interval sandwich, and exact node/weight scope. Put the one-sided nature of the result in the theorem statement itself.

### 4. T16 and the certificate machinery

Present the Delsarte minorant, contact structure, exact finite recurrence, and analytic harmonic tail. Until primal attainment is formally linked, title the result “all-degree reduced-cost negativity,” not “complete all-degree LP optimality.”

### 5. T27: exact optimization inside the Kerdock-line universe

Give the three-value association-scheme risk reduction and optimize over arbitrary real line weights. Include one explicit counterexample showing why the theorem does not extend to nonlinear or network-adaptive estimators.

### 6. Arbitrary signed rules: a stability lemma, not closure

State the negative-mass lower bound and exclusion curve. Explain geometrically why the bound is loose and why T27 is stronger only on the fixed line universe.

### 7. Correction theory: downstream phase is the missing quantity

Develop the correction-risk identity, replacement theorem, common-bias non-identifiability, correlated shrinkage, and ReLU crossing. Replace all universal Euclidean precision language with downstream-weighted criteria involving the replay Jacobian or exact replay loss.

### 8. Exact no-op controls and live harmonic directions

Prove the low-degree and named Stein/ReLU-Stein annihilation results. Include the Poisson-kernel counterexample. Then distinguish limiting-kernel oracle shares from finite-width empirical evidence.

### 9. Falsification map of tested adaptive corrections

Normalize every campaign by:

- estimator information class;
- target quantity;
- selection chronology;
- development/frozen/oracle status;
- complete candidate/base ratio;
- tail metrics;
- artifact completeness;
- explicit reopening gate.

Lead with the reproduced Path-2 scalar result and the frozen high-degree dictionary failure. Place M146 and M152 in a “reported but not evidence-bearing” box unless restored.

### 10. Public-method and compute-accounting context

Treat score and compute comparisons as context, not theorem evidence. Include only numbers that pass a separate traceability audit from executable package to row-level result.

### 11. Limitations and reopening conditions

Make the open classes prominent: arbitrary signed nodes, network-adaptive support, nonlinear estimators, biased/deep analytic controls, and genuinely new absolute-phase observables.

### 12. Reproducibility and AI-assistance disclosure

Publish scripts, exact environments, manifests, row tables, split IDs, certificates, generated intermediates, and immutable external archive digests. Disclose which proofs, experiments, and prose were generated or checked with language models and what was independently rerun.

## Theorem and empirical claim table

| ID / claim | Recommended status | Paper wording | Required qualification |
|---|---|---|---|
| T22 Kerdock gap `<=0.0233655%` | **COMPUTER-ASSISTED CERTIFIED** | Complete Kerdock is within `0.0233655%` of optimal in the stated static nonnegative class. | Infinite-width depth-32 kernel, `d=256`, at most 66,048 nodes, nonnegative mass-one weights, network independence. One-sided only. |
| T23 clean reproduction | **VERIFIED AFTER CORRECTIONS** | The theorem-critical certificate regenerates and verifies under the documented trust base. | Correct 32-file count; distinguish 23 intermediates; external digest and CI still needed. |
| T16 all-degree reduced costs | **PROVED** | Every unused reduced cost is strictly negative for all degrees `>=6`. | Full LP-optimality wording remains conditional on exact primal attainment/complementarity. |
| T27 fixed Kerdock-line optimum | **PROVED UNDER EXPLICIT MODEL** | Complete bases plus at most one partial basis optimize arbitrary real-weight linear rules in the fixed Kerdock-line universe. | No arbitrary nodes, nonlinear postprocessing, finite-width, or network adaptivity. |
| Signed negative-mass bound | **PROVED / COMPUTER-ASSISTED CONSTANT** | A signed rule with negative mass `beta` obeys the stated lower bound. | Numerically too weak to exclude useful arbitrary signed-node rules. |
| Correction-risk identity | **PROVED** | Improvement is controlled by the signed downstream error-correction inner product. | Define the Hilbert space and replay map explicitly. |
| Replacement criterion | **PROVED UNDER EXPLICIT MODEL** | Replacement helps when its downstream-weighted error is below the correctable defect. | Do not replace by a universal unweighted relative-error threshold. |
| Common-bias non-identifiability | **PROVED UNDER EXPLICIT OBSERVATION MODEL** | Fold agreement cannot identify a shared absolute phase/bias. | Not a theorem against all new observables. |
| ReLU crossing bound | **PROVED / BOUNDED** | Nonlinear replay error is controlled by gate crossings near margins. | Exact constants depend on the stated replay model. |
| M146 `41.2x` and `~5e-4` curve | **OPEN / PROVISIONAL** | Omit from evidence-bearing claims. | Original rows, cohort, directions, seeds, and manifest missing; threshold direction-dependent. |
| T4 frozen anchor policy `1.127854` | **FROZEN EMPIRICAL, PENDING TRACEABILITY AUDIT** | The tested frozen policy was harmful despite oracle headroom. | Retain only after source rows, metric, and selection chronology are independently traced. |
| M152 1,100-network scalar result | **OPEN / UNVERIFIED** | Omit. | No reproducible experiment package. |
| Reproduced Path-2 scalar models | **FROZEN EMPIRICAL** | Tested grouped models added no feature-dependent value over matched constants. | Narrow information class and small number of independent base networks. |
| Low-degree polynomial controls | **PROVED UNDER 5-DESIGN ASSUMPTION** | Exactly radialized angular polynomials through degree 5 are no-ops. | Nonlinear suffixes may regenerate high degree. |
| Polynomial Stein fields | **PROVED FOR COMPONENT DEGREE `<=4`** | Their Stein image has degree at most 5 and is integrated exactly. | Do not claim the whole polynomial Stein family. |
| Bias-free one-layer ReLU Stein fields | **PROVED UNDER EXPLICIT MODEL** | Exact blockwise annihilation under exact radialization and antipodal basis blocks. | Excludes biases, depth `>=2`, products, and node-dependent fitting. |
| Frozen degree-6+8 dictionary `1.004439` | **FROZEN EMPIRICAL — FAILED** | The preregistered small dictionary did not improve frozen validation. | Does not close general high-degree controls. |
| Limiting-kernel degree shares | **ORACLE DIAGNOSTIC** | Degree-specific shares describe the infinite-width kernel decomposition. | Not measured width-256 shares. |

## Main negative-results table

| Branch | Strongest legitimate conclusion | What failed | What remains open | Reopening gate |
|---|---|---|---|---|
| Static nonnegative cubature | Essentially closed for the specified limiting kernel and node budget. | Any fixed/network-independent nonnegative rule can improve by at most `0.0233655%`. | Finite-width and adaptive estimators. | New theorem class or changed estimator assumptions. |
| Signed weights on Kerdock lines | Closed exactly inside the fixed line universe. | Negative line/basis weights and irregular support cannot beat complete bases plus one partial basis. | Signed nodes outside the universe. | New support geometry with exact or certified gain. |
| Arbitrary signed cubature | Current general certificate is weak. | The negative-mass lower bound permits large relative gains at tiny `beta`. | Whether arbitrary signed nodes can materially improve. | Stronger geometry-aware certificate or explicit winning construction. |
| Layer-31 anchor | Theory identifies downstream-weighted phase; archived precision experiment is incomplete. | Tested legal anchor families and frozen T4 policy fail; universal scalar tolerance is invalid. | New independent absolute-phase observable. | Restored artifacts plus downstream-weighted exact-replay success on frozen grouped data. |
| Scalar learning | No feature-dependent value in reproduced Path-2 models. | Learned variation loses to matched constants; row-wise validation leaks rotation dependence. | Richer models with large genuinely independent corpus. | Immutable grouped rows, legal features, clean labels, matched-constant baselines, safe tails. |
| M152 | Not evidence-bearing. | Corpus and experiment package are absent. | The claimed 11-feature result itself. | Exact rows, target, grouping, chronology, predictions, and hashes. |
| Low-degree controls | Exact no-ops. | Polynomial/Hermite controls through the covered degrees and named Stein classes vanish. | Biased, deep, nonhomogeneous, and high-degree controls. | A control outside the proved annihilated classes with exact legal expectation. |
| Small high-degree harmonics | Frozen selected dictionary failed. | Degree-6+8 four-direction rule reversed on frozen validation. | Larger/richer high-degree analytic families. | Preregistered independent test with clear oracle ceiling and compute accounting. |

## Figure plan

1. **Estimator-class scope diagram.** Four nested/non-nested boxes: static nonnegative, static signed, network-dependent linear, nonlinear/analytic. Place T22, T27, signed stability, and the empirical program in their exact boxes.
2. **Radial reduction and kernel discrepancy.** Gaussian input to radius-direction decomposition, spherical quadrature, and ensemble-MSE/RKHS discrepancy.
3. **Certified T22 sandwich.** Lower certificate and Kerdock upper certificate, visually showing the `0.0233655%` maximum relative gap and its one-sided nature.
4. **T27 support allocation.** Risk versus line budget with complete-basis breakpoints and at most one partial basis; annotate that arbitrary signed weights collapse to positive structured weights inside the universe.
5. **Harmonic proof architecture.** Exact finite degrees, analytic tail cutoff at 14,659, and the remaining primal-attainment qualification.
6. **Downstream correction geometry.** Defect `d`, anchor error `xi`, replay map `J`, correction inner product, and ReLU crossing region. This should replace any scalar “precision barrier” graphic.
7. **Harmonic taxonomy.** Annihilated degrees/classes, live higher-degree directions, limiting-kernel oracle shares, and tested/frozen dictionary status.
8. **Scalar-learning matched-constant comparison.** Candidate ratios for learned models, fixed constants, and constants matched to each model’s mean prediction, grouped by base network.
9. **Falsification matrix.** Rows are estimator families; columns are theorem/oracle/development/frozen/artifact status/reopening gate.
10. **Reproducibility dependency graph.** Claim to theorem/certificate/script/rows/split manifest/hash. Highlight M146 and M152 as broken chains.

Figures 1–6 belong in the main paper. Figures 7–10 can be split between main text and appendix depending on length.

## Related-work map

The related-work section should be organized by technical relationship, not by chronology.

### Kernel quadrature and RKHS discrepancy

Connect the ensemble-MSE identity to kernel quadrature, Bayesian quadrature, worst-case integration error, and energy minimization. Emphasize that the present contribution is an explicit near-optimality certificate at a large fixed node budget for a deep compositional ReLU kernel.

### Spherical designs and Delsarte linear programming

Position Kerdock/MUB constructions among spherical designs, association schemes, universal optimality methods, and Delsarte bounds. T22 is an approximate global statement for a particular kernel and positive class; T27 is an exact optimization theorem inside a specific association-scheme universe.

### Signed cubature and stability

Relate the negative-mass lemma to signed measures, total variation, and stability of quadrature. The paper’s result is a necessary-condition bound, not a general signed-cubature optimality theorem.

### Random neural-network kernels

Connect the depth-32 kernel to NNGP/random-feature limits and compositional arc-cosine/ReLU kernels. State explicitly that finite width introduces network-specific deviations outside the limiting-kernel theorem.

### Control variates and Stein control variates

Separate exact-annihilation results from variance-reduction methods. The novelty is not “Stein controls fail,” but that several precisely defined families vanish identically under the complete Kerdock design, while richer classes remain open.

### Learned quadrature, coresets, and adaptive integration

Map the empirical program to learned control variates, neural quadrature, coreset selection, active sampling, and meta-learned estimators. The methodological contribution is the use of grouped rotations, matched-constant baselines, oracle ceilings, frozen gates, and artifact-level traceability to avoid mistaking average aggressiveness for network-specific prediction.

### White-box expectation propagation

Discuss analytic moment propagation, local linearization, Gaussian closures, and layerwise approximation. The correction theory clarifies that small unweighted state error is insufficient; the relevant error is signed and downstream weighted.

All citations for this section should be collected from primary papers in a separate literature pass. The eight agent reports establish the paper’s internal claims, not the external novelty comparison.

## Limitations section — paper-ready draft

The certified results apply to a deliberately narrow estimator class. T22 concerns the dimension-256, depth-32 infinite-width normalized ReLU kernel and fixed or network-independent nonnegative mass-one linear rules with at most 66,048 spherical nodes. It does not establish optimality for a particular finite width-256 network, for signed measures, for supports or weights selected from the realized network, or for nonlinear estimators. T27 permits arbitrary real weights but only on the fixed symmetrized Kerdock-line universe; its conclusion cannot be extrapolated to arbitrary spherical nodes or unpaired evaluations. The general negative-mass bound covers arbitrary signed supports but is too loose to exclude competition-relevant improvements.

The correction results also have explicit model boundaries. The replacement and non-identifiability theorems identify necessary geometry under stated observation and replay models, but they do not prove that every future white-box observable lacks absolute phase information. The relevant accuracy is downstream weighted and direction dependent, especially near ReLU gate boundaries. A previously reported universal layer-31 relative-error threshold is therefore not a theorem. The original artifact package for that perturbation experiment was not located, so its numerical headline is treated as provisional.

Our empirical negative results cover tested information classes, not all adaptive estimators. The reproducible scalar-learning audit has a small number of independent base networks and shows that the tested grouped linear and small nonlinear models add no value beyond matched constant controls. A separate claimed 1,100-network experiment is excluded because its rows, target definition, features, grouping, predictions, and code were unavailable. Likewise, the frozen high-degree harmonic failure concerns a small selected zonal dictionary; general high-degree, nonpolynomial, biased, deep, or network-adaptive analytic controls remain open.

Finally, the strongest theorem is computer-assisted. Its trust base includes exact-rational witness data, directed-interval arithmetic, certificate-generation code, the numerical libraries and runtime, and the correctness of the released source archive. We mitigate this dependence with clean regeneration, independent arithmetic checks, manifests, environment pinning, and explicit release provenance, but the proof is not formalized in a proof assistant.

## Claims that must not appear

1. “Kerdock is globally optimal.”
2. “Kerdock is proved strictly suboptimal by `0.0233655%`.”
3. “The theorem applies directly to every width-256 network.”
4. “Signed weights cannot help.”
5. “T27 closes arbitrary signed-node cubature.”
6. “The all-degree LP optimum is completely proved” before exact primal attainment/complementarity is linked.
7. “The certificate is formally verified.”
8. “The manifest is immutable” without external authentication.
9. “A universal layer-31 accuracy threshold is approximately `5e-4`.”
10. “The `41.2x` anchor gain is independently verified.”
11. “M152 used a verified 1,100-network exact-label corpus.”
12. “Scalar learning is impossible.”
13. “No statistical path exists.”
14. “Three faces are one theorem.”
15. “Analytically integrable controls are low degree.”
16. “The whole Stein family is annihilated.”
17. “Degree-6+ controls cannot help.”
18. “Only degree 6 remains.”
19. “The 13.93% degree-6 share is a measured width-256 fact.”
20. “Public or official score/accounting claims are verified” before a separate evidence-traceability audit.

## Mandatory pre-submission actions

### Fatal if unresolved

1. Remove or repair the stale T22 artifact that implies a positive suboptimality lower bound.
2. Publish a corrected certificate release with the 32-file count, intermediate-chunk policy, pinned environment, and externally anchored archive digest.
3. Either prove the exact primal-attainment/complementarity link for the degree-5 auxiliary or narrow every T16/LP-optimality sentence.
4. Exclude M146 and M152 numerical headlines from the abstract, main claims, and evidence table unless their complete original packages are restored and independently reproduced.

### Major

5. Run a dedicated evidence auditor over every retained empirical number, including T4, Path-2, the high-degree dictionary, and public compute/score accounting.
6. Rebuild every empirical table from row-level outputs with base-network-grouped uncertainty.
7. Add constants matched to each learned model’s mean prediction as mandatory scalar-learning baselines.
8. Replace unweighted anchor precision gates with downstream-weighted exact-replay gates.
9. Add a claim-to-artifact index and explicit AI-assistance disclosure.

### Desirable

10. Add the Poisson-kernel counterexample to the main text or a prominent appendix.
11. Include one nonlinear/network-adaptive counterexample immediately after T27 to prevent scope leakage.
12. Put the signed negative-mass curve in an appendix unless space permits a short “why the general bound is weak” discussion.

## Publication architecture verdict

**The paper is viable.** Its strongest version is not an impossibility paper. It is a certified-boundary paper with a rigorous and unusually transparent account of the remaining adaptive gap.

The main theorem package is strong enough to anchor the paper after the identified release and wording corrections. The T27 theorem materially strengthens the story. The correction-theory section gives the empirical program a principled target. Agents 6–8 prevent the narrative from overclaiming by showing that the apparent anchor threshold is direction dependent, the key scalar experiment is missing, and harmonic/learning failures have distinct logical statuses.

The recommended final message is:

> Static nonnegative cubature for the limiting deep-ReLU kernel is essentially solved at the competition node budget. Structured signed reweighting inside the Kerdock universe is exactly solved. Finite-width white-box improvement remains open, but successful methods must recover a signed downstream network-specific defect that the tested anchors, scalar predictors, and small harmonic controls did not reliably reveal.

That statement is both strong and defensible.

## Source files reviewed

- Agent 1: `DECISION(10).md`
- Agent 2: `DECISION(12).md` and proposed release corrections
- Agent 3: `DECISION.md` for T27 and associated scope package
- Agent 4: `DECISION(14).md` and T16 certificate summary
- Agent 5: `AGENT5_SIGNED_WEIGHT_REPORT.md`
- Agent 6: `AGENT6_FINAL_REPORT.md`
- Agent 7: `AGENT7_FINAL_REPORT.md`
- Agent 8: `AGENT8_HARMONIC_CONTROL_AUDIT.md`
- Coordinating prompt and prior evidence framing: `Pasted markdown(8).md`
