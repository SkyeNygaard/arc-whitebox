# Prompt 9 — Cross-Agent Reconciliation

## Executive verdict

The premise “after all runs” is not satisfied. The accessible record contains the specifications for Prompts 1–8, the v31 pre-reconciliation state, and earlier agent reports, but not completed deliverables for the independent reproduction, tied/shared-covariance test, M194 empirical gate, M189 QTT falsifier, M193 facet audit, M210 mechanism audit, or M190 internal-engine experiment. The v31 ledger itself records that scripts and official-result artifacts were referenced but not attached and that independent regeneration remained required.

**No new deployable improvement exists.** The only runnable candidate supported by the record is the existing 129-basis estimator. Its reported exposed Mini-100 result is:

| Metric | Reported result |
|---|---:|
| Adjusted score | $1.4641716\times10^{-7}$ |
| Raw MSE | $2.2819432\times10^{-7}$ |
| Mean multiplier | $0.6427$ |
| Effective compute | $1.748\times10^{11}$ |
| Estimator FLOPs per MLP | $1.70873\times10^{11}$ |
| Failures | $0/100$ |
| Observed wall time | about 16.5 seconds per MLP |

The local propagation prediction was reportedly within 0.03% of the raw MSE. These are exposed-development results, not protected results, and the package hash, asset hash, result JSON, and complete reproduction record are unavailable in the accessible artifacts.

The current root `estimator.py` is not a candidate. Under FlopScope 0.9.1, the two-network smoke test reportedly failed with `TypeError: dot() got an unexpected keyword argument 'out'`, leading to the zeros-baseline result near 0.83. Its current package identity must remain quarantined.

**Competition decision:** retain the 129-basis package as the shipping baseline; do not open protected data; run the independent reproduction bundle before interpreting any remaining branch as live.

---

## Evidence hierarchy

### Theorems and exact identities

The following are mathematical statements rather than candidate results:

- A finite baseline transcript cannot universally identify the Gaussian mean over the full homogeneous ReLU class; weight-aware methods are not ruled out.
- Boundary-Stein and gate-current identities are exact, but inherited ancestor-boundary terms cannot be omitted.
- Generic exact merging of live activation histories by affine-map equality is ineffective.
- Conventional coordinate-sparse global polynomial chaos becomes dense under generic ReLU propagation.
- Layerwise shared-output Kerdock QTT encounters an exact rank-256 first-layer flattening.
- Positive homogeneity gives an exact compositional transfer identity for environment-weighted chaos, but not a closed finite recurrence.

These results close or narrow specified classes. They do not supply a legal estimator.

### Oracle representation results

The strongest mixture representation reportedly reaches closure error

$$
\delta\approx1.781\times10^{-3}
\qquad\text{at }K=1536.
$$

An earlier rank-64 latent construction reportedly reached approximately $1.96\times10^{-3}$, while a $K=32$ full-covariance construction was around $4.8\times10^{-3}$. These are evidence that useful joint state exists, but they do not establish legal initialization, legal recurrence, or affordable evaluation.

### Legal estimator results

Only the fixed 129-basis package has a reported complete exposed-harness result. None of M192, M194, M189, M193, M190, M195, or M210 has produced a new frozen legal package with complete score accounting.

### Cost projections

Several results are economic projections, not measured package scores:

- exact heteroscedastic componentwise propagation: $8.68\times10^7$ operations per component;
- shared-reference Taylor: approximately $2.6\times10^5$ per component, but inaccurate;
- low-rank direct/Hermite: $2n^2r$, becoming approximately $n^3$ at the required rank;
- standalone degree-10 M190: at most about a $1.58\times$ zero-cost gain, insufficient to win;
- QTT query budgets and M195 inference economics remain projections until implemented in a legal package.

### Protected evidence

There is none. The protected evaluation remained sealed at the v31 cutoff.

---

## Path-by-path decision table

| Path | Scope and evidence audit | Reconciled decision |
|---|---|---|
| **Prompt 1: independent reproduction** | Not completed in the accessible record. The reported official run is strong local evidence, but hashes, exact environment record, scripts, saved arrays, and archived JSON were not supplied. The fixed-mean isolation experiment requested for the Taylor mechanism is also absent. | **INCOMPLETE.** Treat M205–M208 as reported, not independently reproduced. |
| **M192: tied/shared covariance** | The broad heteroscedastic full-covariance program was explored extensively. The exact tied/shared-correlation recurrence requested by Prompt 2 was not isolated or run. Earlier latent and particle results do not substitute for it. | **NO PROMPT-2 VERDICT.** General M192 is mostly closed; the strict tied/shared exception remains unrun. |
| **M194: cubic boundary/Walsh phase** | Agent 8 supplied obstruction theorems and a schematic cubic bispectrum. It did not freeze a complete algebraic kernel, measure grouped unfitted covariance, or produce a score. | **UNRUN ONE-SHOT HEDGE.** Proposal only; no candidate. |
| **M189: Kerdock-index QTT** | Agent 5 proved strong obstructions for Cartesian and layerwise tensor formulations and identified direct final-output QTT as the only plausible loophole. Rank curves, shared pivots, legal query union, and held-out mean errors were not run. | **UNRUN BOUNDED FALSIFIER.** Layerwise QTT closed; direct final-output QTT remains low-prior. |
| **M193: output-weighted activation fan** | Exact map-sharing is generically ineffective, but output-weighted normal rank and current telescoping were explicitly left as diagnostics. Activation covariance rank is not evidence for this object. | **UNRUN BOUNDED AUDIT.** No compressed facet estimator. |
| **M210: tail mechanism** | A heavy exposed-Mini tail was detected, but no grouped diagnostic analysis or intervention was completed. | **NO-SHIP for changes.** Retain baseline unchanged. |
| **M190: internal chaos engine** | Conventional global sparse PCE is closed. An exact environment-weighted transfer identity exists, but no target contraction, rank table, legal orientation, or parent-path cost reduction was demonstrated. | **FAIL as standalone; UNRUN as internal engine.** |
| **M195: full quotient operator** | The 29-feature handcrafted predictor failed with negative leave-one-network-out $R^2$ and chance sign. That does not test the complete quotient operator. No materially better frozen anchor or quotient representation exists. | **DEFER.** Handcrafted subclass closed; Prompt 8 prerequisites absent. |

---

## Detailed branch reconciliation

### 1. Independent reproduction

The local results are internally coherent, but they are not yet independently certified.

The pooled-within recentering reportedly changed the $K=64$ measurements as follows:

| Layer | Covariance offset | Second-order error |
|---:|---:|---:|
| 16 | $0.586\rightarrow0.476$ | $5.39\times10^{-3}\rightarrow5.58\times10^{-3}$ |
| 29 | $0.574\rightarrow0.357$ | $4.00\times10^{-3}\rightarrow5.41\times10^{-3}$ |

This strongly supports the interpretation that covariance displacement is not the main Taylor error source. It does **not** yet constitute the requested experimental isolation in which component means are held fixed while only the covariance reference changes. “Mean-offset dominated” should therefore be recorded as strongly supported, not proved or independently reproduced.

The direct/Hermite rank sweep reportedly found:

| Layer | $r=4$ | $r=16$ | $r=64$ | $r=128$ |
|---:|---:|---:|---:|---:|
| 16 | $2.16\times10^{-1}$ | $5.44\times10^{-2}$ | $6.73\times10^{-3}$ | $7.86\times10^{-4}$ |
| 29 | $1.74\times10^{-1}$ | $4.00\times10^{-2}$ | $3.76\times10^{-3}$ | $2.23\times10^{-4}$ |

Only $r=128$ clears the stated $1.5\times10^{-3}$ gate, while the local budget permits roughly $r\le4.4$. At $r=128$,

$$
2n^2r\approx1.7\times10^7\approx n^3.
$$

This closes the tested low-rank factorization/Hermite route, not every possible structured algorithm for computing the required diagonals.

### 2. M192

Three evaluated mechanisms jointly squeeze the current heteroscedastic implementation:

1. exact componentwise propagation is too expensive;
2. shared-reference Taylor loses accuracy as component means spread;
3. the tested low-rank diagonal extraction needs effectively dense rank.

That is a strong scoped closure. It does not cover a strict tied covariance

$$
R_{\ell k}=R_\ell
$$

or a genuinely shared low-rank modulation with one fixed orientation. The exact post-ReLU covariance remains component-dependent when the means differ, so merely declaring a shared pre-ReLU covariance does not establish the required computational reuse. Prompt 2’s recurrence, cost, oracle ladder, legal initialization, and 32-layer rollout have not been supplied.

**Decision:** do not promote M192, but do not write “all analytic mixtures are closed.” Record “general heteroscedastic implementation mostly closed; exact tied/shared exception unrun.”

### 3. M194

Agent 8 proposed a phase-bearing construction of the form

$$
\widehat b=
\sum_{\chi,\psi}
\kappa_{\chi,\psi}(W)\,
\widehat s_\chi
\left\langle\widehat s_\psi,\widehat s_{\chi+\psi}\right\rangle,
\qquad
\widehat\delta=V_r\widehat b.
$$

This is meaningfully outside a purely linear transcript rule only when $\kappa_{\chi,\psi}(W)$ contains genuinely weight- or boundary-derived orientation information. The report requires the kernel to be gauge invariant, permutation invariant, fixed before target access, and based on inherited boundary information.

However:

- no unique kernel was fully derived;
- no proof established that its simplified form survives antipodal and Kerdock symmetries;
- no grouped raw covariance was measured;
- no phase-scrambling null was run;
- no complete FLOP count or package exists.

Therefore M194 is a proposed class, not an exact identity result and not an estimator result. Agent 8 explicitly reported no complete estimator meeting the raw-MSE or adjusted-ratio target.

### 4. M189

Agent 5’s strongest exact result is nonlinear Kerdock spectral densification: the square of a first-layer preactivation has 32,641 nonzero sign-even Walsh frequencies out of 32,768. A proposed $128\times255$ matricization would give a rank-128 lower bound if full row rank were verified. But dense Walsh support alone is not a TT-rank theorem, and the symbolic nonvanishing-minor proof was not completed.

The direct final-output QTT loophole remains different from layerwise QTT. Its decisive issue is not oracle TT-SVD rank after reading all 65,536 entries, but whether one legal common pivot rule reconstructs all 256 outputs with a small union of full-network queries. Separate output models cannot each claim the same nominal query budget without charging their union.

**Decision:** close Cartesian TT, ordinary rotated TT, first-layer ridge TT, and shared-output layerwise Kerdock QTT. Keep direct final-output M189 only as a cheap, unrun falsifier.

### 5. M193

Agent 7 established useful scoped negatives:

- ordinary activation-history sharing does not give useful exact BDD/ZDD compression;
- generic live downstream affine maps are distinct;
- activation covariance participation ratio does not establish low boundary-normal rank;
- frequency or probability mass alone does not bound signed output contribution.

The still-open object is the matrix of **output-weighted boundary normals** and the signed decomposition of exact gate currents. Its two decisive tests—weighted normal rank and a possible layer telescope—were explicitly left unrun.

M193 and M194 overlap in their dependence on ancestor-boundary geometry, but they are not identical:

- M193 asks whether the exact facet-current integral compresses geometrically.
- M194 asks whether a cubic Walsh contraction extracts signed phase from that geometry.

A negative M193 normal-rank result would lower M194’s prior but would not formally kill every cubic phase kernel.

### 6. M210

The worst exposed Mini network reportedly has error $8.52\times10^{-7}$, about 5.8 times the mean, and contributes roughly 6% of total loss.

No evidence currently distinguishes:

- a few catastrophic output coordinates;
- broad network-wide degradation;
- a numerical/runtime anomaly;
- ordinary sampling variance;
- a legally predictable network mechanism.

A diagnostic that predicts error magnitude is not automatically a signed correction signal. Until a frozen network-independent rule lowers both grouped tail loss and mean adjusted score, the correct decision is to ship the unchanged baseline.

### 7. M190

The exact identity

$$
\nu_{\ell+1}(\varphi)
=
\nu_\ell(q_{\ell+1}\,\varphi\circ F_{\ell+1})
$$

is a legitimate compositional transfer law, and the degree-10 target can be written in terms of six normalized homogeneous tensor orders. But this is not a finite closed recurrence, and no environment-weighted rank experiment has shown that a legally chosen basis remains compact through depth.

M190 duplicates the computational target of other branches whenever it is used to evaluate:

- M192’s next-layer covariance or variance contractions;
- M193’s boundary-adjoint scalar;
- M194’s cubic kernel.

It should therefore be judged only by the reduction it supplies to a surviving parent path. It is not an independent candidate.

### 8. M195

The tested 29-feature weight/state predictor achieved leave-one-network-out $R^2$ between approximately $-0.018$ and $-0.002$, with chance-level sign performance. This closes further expansion of that handcrafted feature family.

It does not prove that a complete symmetry-quotiented full-weight operator is impossible. Earlier synthesis estimated observed direct prediction near $R^2\approx0.036$ against a required value near $0.84$, making the prior extremely low.

Prompt 8’s prerequisites are absent: there is no materially better frozen analytic anchor, no substantially reduced residual target, and no completed exact quotient representation. The correct verdict is:

> **M195 remains deferred; the prerequisites are absent.**

---

## Duplicated work and shared observables

| Overlap cluster | Reconciliation |
|---|---|
| **M192–M190** | Both can target the same next-layer variance or covariance contraction. M190 counts only as an engine inside M192, not as an independent success. |
| **M193–M194** | Both rely on output-weighted boundaries, downstream adjoints, and inherited gate currents. M194 adds a cubic Walsh phase contraction; it cannot cite the existence of the boundary identity as evidence that the kernel is cheap. |
| **M189–M194** | Both use Walsh/Kerdock structure. M189 concerns tensor rank and legal query complexity; M194 concerns a weight-coupled signed observable. Dense Walsh support affects both priors but is not a joint closure theorem. |
| **M210–M192** | Gaussian-closure disagreement or mixture diagnostics may help predict tail risk. That would be a diagnostic reuse, not a new estimator unless it supplies stable signed correction or a cost-effective intervention. |
| **M195–earlier weight features** | A full quotient operator must not be presented as completed by rerunning or enlarging the failed 29-feature dictionary. |

---

## Claims rejected under full cost and legality accounting

1. **Reject any mixture win** that omits component initialization, covariance/reference construction, eigendecompositions, factor construction, higher Hermite orders, or all layerwise transforms.

2. **Reject any QTT win** obtained after reading the complete tensor, using target-dependent subtraction, or charging a pivot count for one output while ignoring the union across 256 outputs.

3. **Reject any M194 win** based on a target-selected character pair, oracle coefficient phase, per-network shrinkage, or a kernel selected from many candidates using true errors.

4. **Reject any M210 intervention** selected by the observed worst-network identity or true network error.

5. **Reject pointwise residual variance** whenever complete antipodal or basis-block variance is the relevant economic object.

6. **Reject every protected-set result.** No candidate qualified to open protected data.

7. **Reject the root estimator package.** It is non-runnable under the tested FlopScope version.

---

## Contradiction map

| Claim | Reconciliation |
|---|---|
| “The last untested space closed.” | **Overstated.** The last measured evaluator in the heteroscedastic implementation closed; tied/shared covariance, M194, M189, M193, and other structured diagonal methods remain untested. |
| “Mean-offset domination is proved.” | **Too strong.** Covariance recentering strongly supports it, but the prescribed fixed-mean isolation and independent reproduction are absent. |
| “All analytic mixtures are closed.” | **False outside scope.** Only exact full-component, shared-reference Taylor, and tested low-rank Hermite evaluators are squeezed. |
| “The official package is independently reproduced.” | **Not established.** An exposed official-style result was reported and agrees with local prediction, but hashes, JSON, and reproduction artifacts are absent. |
| “Dense Walsh support proves high QTT rank.” | **False.** It removes one sparsity mechanism but does not prove high TT rank. |
| “Low activation covariance rank implies low activation-fan rank.” | **Invalid substitution.** The relevant object is output-weighted boundary-normal rank. |
| “Failure of 29 weight features closes full-weight learning.” | **False.** It closes the tested dictionary; the complete quotient remains logically open but deferred. |
| “The root estimator is a statistical competitor.” | **Invalid.** Its present package crashes and must not enter score comparisons. |

---

## Updated priority order

1. **Independent reproduction of M205–M208.**
2. Exact tied/shared-covariance M192 exception, only after reproduction succeeds.
3. One frozen M194 algebraic kernel.
4. One stored-array M189 direct final-output QTT falsifier.
5. One M193 output-weighted normal-rank/current-telescope audit.
6. One grouped, no-fit M210 tail audit.
7. M190 only when a surviving parent path specifies the contraction it must accelerate.
8. M195 remains deferred.
9. Keep the root estimator quarantined and protected data sealed.

---

## Permanently closed branches

“Permanently” here means closed inside the stated mathematical and implementation scope, not universal impossibility.

- Shared-reference first- and second-order Taylor centers and denser reference hierarchies.
- The tested low-rank Hermite/factorization covariance truncation route.
- Larger-$K$ arbitrary heteroscedastic full-covariance searches using the tested evaluators.
- Generic exact BDD/ZDD merging by activation history or affine-map equality.
- Global dense activation-region enumeration.
- Conventional coordinate-sparse global PCE/ANOVA and explicit high-degree tensor enumeration.
- Cartesian functional TT/HT and ordinary random-rotation descendants.
- Shared-output layerwise Kerdock QTT.
- Universal exact identities based solely on the existing finite output transcript.
- The tested 29-feature M195 dictionary.
- The current root `estimator.py` package identity.

## Merely unpromising or untested branches

- Strict tied/shared covariance or fixed shared-low-rank covariance modulation.
- One weight-coupled cubic boundary/Walsh identity.
- Direct final-output Kerdock-index QTT with common legal pivots.
- Output-weighted facet-current compression.
- A tail-risk intervention derived from a legal network diagnostic.
- Environment-weighted chaos as an internal contraction engine.
- A complete symmetry-quotiented full-weight operator.
- A non-mixture compact copula state.
- A genuinely new structured algorithm for the required covariance diagonals.

Their priors are low because each requires a strong conjunction: a compact representation, legal orientation or initialization, stable grouped behavior, complete cost advantage, and winning-scale accuracy. Existing work has repeatedly found oracle capacity without a legal, affordable evaluator.

---

## Canonical ledger patches

Use the existing IDs; do not rewrite historical evidence.

| ID | Proposed patch |
|---|---|
| **T106 / M205** | Replace “mean-offset domination proved” with: “Pooled-within recentering substantially reduced covariance offsets without improving error, strongly supporting mean-offset domination. Fixed-mean isolation and independent reproduction remain pending.” |
| **T107 / M206** | Preserve closure of the tested Hermite/factorization route; add: “Not an information-theoretic lower bound for all structured diagonal algorithms.” |
| **M207 / M192** | Status: “Mostly closed / exact tied-shared exception unrun.” Do not say every analytic mixture or compact joint state is closed. |
| **M208** | Status: “Reported exposed official Mini-100 validation; independent hash/JSON/environment reproduction pending.” Preserve the reported score and cost numbers. |
| **M209** | Status: “Broken and quarantined under FlopScope 0.9.1.” Record the 2/2 smoke failure; do not claim a universal all-network failure until independently rerun. |
| **M210** | Status: “Tail observed; mechanism and interventions unrun.” Baseline remains unchanged. |
| **M189** | Status: “Direct final-output QTT falsifier unrun.” Separate it from closed layerwise/shared-output QTT. |
| **M193** | Status: “Exact-map compression closed; weighted-normal/current audit unrun.” |
| **M194** | Status: “Proposed one-shot class; no frozen kernel, covariance gate, or score.” |
| **M190** | Status: “Standalone closed by upside ceiling; internal-engine gate unrun.” |
| **M195** | Status: “Handcrafted feature subclass closed; full quotient deferred because prerequisites are absent.” |
| **M211** | Verdict: “No new deployable improvement. Existing 129-basis baseline is the only reported runnable package. Reproduction is the next gate; protected data remain sealed.” |

---

## Single next experiment

### Run the complete Prompt 1 independent reproduction bundle

This is more urgent than beginning tied/shared M192 because every subsequent decision depends on the correctness and costing of the v31 local results.

The experiment must regenerate, from archived code and arrays:

- the $K$-ladder through $K=1536$;
- pooled-within versus global Taylor curves;
- the fixed-mean/covariance-reference isolation;
- the rank $4,16,64,128$ direct/Hermite sweep;
- the exact 129-basis Mini-100 subprocess result;
- package and asset hashes;
- FlopScope version and BLAS environment;
- per-network errors and wall-time residuals;
- deterministic root-package failure behavior.

**Decision gate:** if the reported values and costs reproduce, patch them from “reported local” to “independently reproduced” and proceed to the exact tied/shared M192 test. If they do not, stop all branch promotion and repair the canonical ledger first.

Until that reproduction exists, there is no evidential basis for opening protected data or claiming that any new winning path has survived.
