# M210 Grouped Tail-Mechanism Audit

**Date:** 2026-07-31  
**Scope:** Exposed official Mini-100 development result  
**Status:** Bounded robustness audit  
**Final recommendation:** **NO SHIP — retain the validated 129-basis estimator unchanged**

## Executive verdict

The available evidence establishes that the official Mini-100 result has a substantial upper tail. It does **not**, however, support the canonical statement that the worst network is 5.8 times the **raw-MSE** mean or contributes 6% of total **raw** loss.

The reported values imply:

\[
\frac{8.52\times10^{-7}}{2.2819432\times10^{-7}}
=3.7337,
\]

whereas

\[
\frac{8.52\times10^{-7}}{1.4641716\times10^{-7}}
=5.8190.
\]

Thus the quoted 5.8× ratio and approximately 6% loss contribution use the **adjusted-score mean**, not the raw-MSE mean. The source summary reports raw MSE \(2.2819432\times10^{-7}\), adjusted score \(1.4641716\times10^{-7}\), multiplier 0.6427, zero failures, and a worst-network value of approximately \(8.52\times10^{-7}\).

The official per-network JSON referenced by the final write-up was not present in the accessible artifacts. The ledger itself records that the JSON was not attached. Consequently, the exact median, variance, quantiles, coordinate decomposition, layer decomposition, and grouped diagnostic regressions cannot be reproduced honestly.

Despite that limitation, the shipping decision is clear. All three relevant intervention families represented in the saved grouped experiments fail at least one mandatory gate:

1. conditional basis allocation cannot reliably identify high-risk networks;
2. robust aggregation fails on frozen holdout data;
3. a companion rotation either costs too much, creates new tails, or is supported only by post-hoc selection.

---

## 1. Tail verification

### 1.1 Verified aggregate quantities

| Quantity | Reported value | Audit status |
|---|---:|---|
| Mean raw final-layer MSE | \(2.2819432\times10^{-7}\) | Verified from official-run summary |
| Mean adjusted score | \(1.4641716\times10^{-7}\) | Verified from official-run summary |
| Mean multiplier | 0.6427 | Verified |
| Effective compute | \(1.748\times10^{11}\) | Verified |
| Estimator FLOPs per network | approximately \(1.70873\times10^{11}\) | Verified |
| Failures | 0/100 | Verified |
| Best reported network | approximately \(3.57\times10^{-8}\) | Reported, but field type is ambiguous |
| Worst reported network | approximately \(8.52\times10^{-7}\) | Reported, but field type is ambiguous |

The run used the official subprocess path on all 100 exposed Mini networks with BLAS pinned to four threads. The local propagation prediction matched the official raw MSE within approximately 0.03%.

### 1.2 Metric inconsistency

There are two possible interpretations of the worst-network number.

| Interpretation of \(8.52\times10^{-7}\) | Ratio to corresponding mean | Contribution to 100-network total |
|---|---:|---:|
| Raw per-network MSE | \(3.7337\times\) | 3.7337% |
| Adjusted per-network score | \(5.8190\times\) | 5.8190% |

Therefore:

- “Worst network approximately 5.8× the mean” is consistent with the **adjusted** mean.
- “Worst network contributes approximately 6% of total loss” is likewise consistent with **adjusted** loss.
- Neither statement is consistent with the reported raw-MSE mean.

The likely explanation is that the official JSON’s `worst_mlp_adjusted_final_layer_score` field was compared against the adjusted aggregate score while later prose labeled the result as raw error. The available summary does not expose enough JSON structure to prove this field identification, but the arithmetic makes the metric mixing unambiguous.

### 1.3 Maximum possible upside from the tail

Even an oracle intervention that completely eliminated the worst adjusted-network loss, while changing nothing else, would improve the mean adjusted score only from

\[
1.4641716\times10^{-7}
\]

to

\[
1.4641716\times10^{-7}
-\frac{8.52\times10^{-7}}{100}
=
1.3789716\times10^{-7}.
\]

That is a maximum improvement of **5.82%**.

Eliminating half of that network’s adjusted loss would improve the mean by only **2.91%**, before accounting for:

- intervention compute;
- false positives on ordinary networks;
- threshold-estimation uncertainty;
- new upper-tail failures.

This validates the instruction to treat the likely upside as modest.

### 1.4 Unverified required decomposition

The following required outputs cannot be reconstructed without the referenced official JSON and associated prediction arrays:

- median per-network error;
- variance;
- 75th, 90th, 95th and 99th percentiles;
- exact sorted error distribution;
- raw and adjusted per-network values separately;
- per-output-coordinate error concentration;
- final-layer versus earlier-layer stability;
- whether the worst network contains a few catastrophic output neurons;
- whether degradation is broad across the output vector.

No conclusion about “catastrophic neurons” versus “network-wide degradation” is justified by the aggregate extrema alone.

---

## 2. Mechanism diagnostics

Three diagnostics are sufficiently motivated and legally available in principle. They must be interpreted separately as magnitude predictors, signed-error predictors, or numerical-failure detectors.

| Diagnostic | Legal role | Can predict | Current evidence | Audit verdict |
|---|---|---|---|---|
| Kerdock block or between-basis disagreement | Uses estimates already generated by the baseline | Error magnitude or tail risk | No official Mini-100 grouped rows were preserved; prior risk guards failed to catch known harmful cases | Plausible magnitude diagnostic, unvalidated |
| Gaussian-closure or late-layer analytic disagreement | Weight/state-derived analytic diagnostic | Primarily error magnitude | Existing closure discrepancies do not establish signed correction direction; exact official-cohort LONO results unavailable | Do not treat as correction signal |
| Runtime/numerical anomaly | Detects instability, overflow, failure or timeout risk | Numerical failure only | Official run had zero failures and substantial timing headroom | Negative mechanism result |

### 2.1 Kerdock block disagreement

Between-basis dispersion is the most natural no-pilot diagnostic because the 129-basis estimator already evaluates the constituent blocks. In principle it could identify networks on which basis estimates disagree unusually strongly.

However, three distinctions matter:

1. Large dispersion might predict large **absolute** error.
2. It does not reveal the sign of the baseline error.
3. A useful risk ranking does not automatically imply that any available intervention improves those networks.

Prior grouped guard experiments are discouraging. A frozen apply/abstain audit caught none of three severe exact-panel tails, made no useful decisions on the legal panel, and caught none of six legal-panel tails. The fixed-112 guard later marked every opened case safe and caught none of eight harmful cases.

Accordingly, between-basis disagreement remains a possible descriptive diagnostic but has not demonstrated the recall required for conditional evaluation.

### 2.2 Gaussian-closure disagreement

A discrepancy between an analytic Gaussian estimate and the Kerdock estimator may correlate with network difficulty. But it is not automatically a correction signal:

\[
\operatorname{Corr}(D_i,\lvert e_i\rvert)>0
\]

does not imply

\[
\operatorname{Corr}(D_i,e_i)\neq0.
\]

The surviving analytic evidence repeatedly shows this distinction: magnitude-like state diagnostics can identify difficult regimes while failing to recover absolute phase or signed residuals. Under the M210 restrictions, no new generic weight-feature predictor should be trained around these diagnostics.

The official-cohort LONO quantities required for promotion are unavailable:

- signed correlation;
- absolute-error correlation;
- top-decile tail AUC;
- threshold stability;
- false-positive intervention cost.

### 2.3 Numerical and runtime anomalies

The official package had zero failures. Reported wall time was approximately 16.5 seconds per network, materially below the guard after the stated grader adjustment. The residual charged wall time was also small compared with estimator FLOPs.

There is therefore no evidence that the statistical upper tail is caused by:

- timeouts;
- fallback activation;
- thread contention;
- nonfinite outputs;
- exceptional residual wall time.

A numerical fallback remains good defensive engineering, but it is not an MSE-tail correction mechanism supported by these data.

---

## 3. Grouped predictive results

A new grouped leave-one-network-out analysis on the official 100 networks was not possible because the per-network JSON and diagnostic rows are absent.

The relevant existing grouped evidence is nevertheless consistent:

| Existing grouped result | What it tests | Result |
|---|---|---|
| Frozen risk/abstention guards | Ability to identify harmful networks prospectively | Failed to catch the observed severe cases |
| Robust basis aggregation holdout | Whether dispersion can be converted into a stable robust estimate | Development improvement reversed on frozen holdout |
| Orientation-selection studies | Whether legal observables select a better rotation | Oracle selection worked; legal selectors did not |
| Companion-rotation correction | Whether a fixed second estimate supplies useful correction information | Raw benefit exists, but complete score and tails fail |

This supports the following separation:

- **Predicting magnitude:** still possible but not demonstrated on the official cohort.
- **Predicting signed error:** no stable legal signal demonstrated.
- **Identifying high-risk tails:** existing frozen guards have inadequate recall.
- **Improving the estimator:** no validated rule clears mean, tail and cost simultaneously.

---

## 4. Fixed mechanism interventions

### Intervention A — Conditional basis allocation or fallback

A natural policy would run a cheaper basis allocation on apparently low-risk networks and retain the full 129-basis estimator on high-risk networks.

The closest preregistered experiment, fixed-112 plus a scalar rescue, failed decisively:

- pooled raw ratio: 1.013228;
- adjusted ratio: 0.896245;
- worst ratio: 1.578572;
- baseline-only fixed-112 worst: 1.715077;
- guard recall on harmful fixed-112 cases: 0/8.

The experiment stopped after the preregistered 1.25 hard-stop threshold was exceeded.

**Decision:** Reject.

Although modeled compute produced a favorable mean adjusted ratio, the policy created catastrophic tails and the frozen guard failed to identify them. It violates the promotion requirements of no new catastrophic failures and a reliable frozen threshold.

### Intervention B — Fixed robust aggregation or conservative blend

A frozen geometric-median aggregation initially showed development improvement, but on four new networks across three rotations it produced:

- mean ratio: 1.005381;
- wins: 7/12;
- median ratio: 0.944836;
- worst ratio: 1.340251.

**Decision:** Reject.

The favorable median alongside a worse mean and 1.34 worst case is exactly the pattern M210 is intended to prevent.

### Intervention C — Fixed companion rotation

The frozen full companion candidate used a complete second 129-basis rotation, 32 probes, propagation through layer 29, and 50% shrinkage.

Its grouped 24-network results were:

- raw ratio: 0.772831;
- raw 95% interval: 0.615530–0.912435;
- wins: 19/24;
- median ratio: 0.835643;
- worst ratio: 1.519876;
- heavy-product cost proxy: 1.935484×;
- adjusted-score proxy: 1.495803×.

Thus it generated genuine correction information but lost badly after compute accounting and created a new 1.52× tail.

A post-hoc compressed version using 16 companion bases and 10% shrinkage appeared more favorable:

- raw ratio: 0.848861;
- adjusted proxy: 0.947353;
- worst ratio: 1.201863;
- adjusted 95% interval: 0.828527–1.051059.

But that candidate was selected after examining the same block, so it is not validation evidence; its adjusted interval also crosses parity. A safer 5% shrinkage version reduced the worst ratio to 1.078642 but had an adjusted proxy of 1.005593 even before all control overhead.

**Decision:** Reject for shipping.

The full frozen candidate fails cost and tail gates. The compressed candidate is post-hoc and statistically inconclusive. The safer candidate is score-negative.

---

## 5. Mean, tail and cost trade-off

| Intervention family | Mean raw effect | Adjusted/cost effect | Worst case | Frozen evidence? | Promotion |
|---|---:|---:|---:|---|---|
| Fixed-112 conditional allocation | 1.013× | 0.896× modeled | 1.579× | Yes, stopped early | Fail: catastrophic tail and guard failure |
| Frozen geometric-median aggregation | 1.005× | Not favorable even before meaningful overhead | 1.340× | Yes | Fail: worsens mean and tail |
| Full companion rotation | 0.773× | 1.496× | 1.520× | Yes | Fail: compute and tail |
| Companion-16, 10% shrink | 0.849× | 0.947× proxy | 1.202× | No, post-hoc | Not promotable |
| Companion-16, 5% shrink | 0.901× | 1.006× before full overhead | 1.079× | Post-hoc | Fail: mean adjusted score |

No intervention satisfies all five promotion conditions:

1. material upper-tail reduction;
2. lower mean adjusted score;
3. positive grouped validation;
4. no new catastrophic failures;
5. frozen threshold and exact runtime rule.

---

## 6. Ship/no-ship recommendation

### Recommendation

**SHIP the existing 129-basis baseline.**  
**DO NOT SHIP an M210 adaptive, blended, reduced-basis or companion intervention.**

The baseline is the only package with end-to-end official exposed-Mini validation, zero failures, and a fully reported aggregate score. The current-state memo also requires preserving the exact shipping package until a grouped alternative passes both mean and upper-tail adjusted-score tests.

M210 should be treated as a bounded negative robustness audit:

- a genuine adjusted-score tail exists;
- its maximum recoverable mean benefit is modest;
- the canonical tail description mixes raw and adjusted metrics;
- existing legal risk guards do not reliably identify harmful cases;
- available correction mechanisms trade raw improvement for compute or new tails;
- no frozen network-independent rule is deployable.

This does **not** prove that every possible target-free tail diagnostic is useless. It does close promotion of the currently evidenced intervention families.

---

## 7. Required canonical correction

The M210 ledger row should be amended to something equivalent to:

> **Reported official per-network tail:** The official summary reports a best network near \(3.57\times10^{-8}\) and a worst near \(8.52\times10^{-7}\). The worst is \(5.82\times\) the mean adjusted score \(1.4641716\times10^{-7}\) and represents \(5.82\%\) of total adjusted loss. It is only \(3.73\times\) the mean raw MSE \(2.2819432\times10^{-7}\). The previous wording conflated raw and adjusted metrics. The archived per-network JSON is not attached, so quantiles and coordinate/layer decomposition remain unreproduced.

Recommended status:

> **PARTIALLY VERIFIED / NO-SHIP — retain baseline.**

Recommended verdict:

> No existing frozen network-independent mechanism reduces both grouped upper-tail error and mean adjusted score without creating new failures. Reopen only after recovering the official JSON and basis-level diagnostic rows; do not tune to the worst network or restart generic weight-feature prediction.

---

## Final M210 verdict

\[
\boxed{\textbf{NO SHIP — RETAIN THE 129-BASIS BASELINE}}
\]

---

## Source artifacts reviewed

- `Pasted markdown(13).md` — Prompt 6 specification.
- `Pasted text(68).txt` — official benchmark summary and per-network extrema.
- `whestbench_canonical_research_ledger_20260731_reconciled_v31_final_local_writeup.xlsx`.
- `WHestBench_Current_State_v31_20260731.md`.
- `DIRECT32_COMPANION_VALIDATION_RESULTS.json`.
- `DIRECT32_DEFECT_ESTIMATOR_REPORT.md`.
- `RESULTS_test.json`.
- `DECISION(3).md`.

## Evidence limitation

A complete numerical Part I and official-cohort leave-one-network-out table requires the referenced `official_129basis_mini100_20260731.json`. The current no-ship decision does not depend on inventing those missing results.
