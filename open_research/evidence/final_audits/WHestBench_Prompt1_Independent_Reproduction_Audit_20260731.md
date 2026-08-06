# Prompt 1 — Independent Reproduction Audit of WHestBench v31

**Audit date:** July 31, 2026  
**Protected data accessed:** No  
**Overall verdict:** **NOT INDEPENDENTLY REPRODUCED — REQUIRED RAW ARTIFACTS MISSING**

## 1. Executive verdict

Prompt 1 requires regeneration from the scripts, saved arrays, exact estimator package, and official result JSON—not merely checking numbers copied into the ledger.

Those required artifacts were not attached to v31 and were not located in the Library:

- no executable mixture-ladder scripts or saved arrays;
- no pooled-within Taylor experiment script or arrays;
- no direct/Hermite rank-sweep script or arrays;
- no exact archived 129-basis shipping package tied to the reported run;
- no `official_129basis_mini100_20260731.json`;
- no complete root-estimator failure log or package snapshot.

The v31 decision memo itself records that the raw scripts and JSON were referenced but not attached and that independent regeneration remained open. The original experimenter also stated that 44 scripts could regenerate the work and explicitly requested an independent rerun after multiple self-corrections, but those scripts are absent from the available record.

Therefore:

> The reported results are internally coherent in several important respects, but v31 does not currently contain an artifact-complete reproduction package.

The main scientific conclusion is not overturned. The tested heteroscedastic full-covariance program remains strongly squeezed. However, M205–M209 should not be described as independently reproduced.

---

## 2. Reproduction table

| Experiment | Reported result | Independent finding | Status | Effect on conclusion |
|---|---|---|---|---|
| **A. Mixture \(K\)-ladder** | \(\delta\approx1.781\times10^{-3}\) at \(K=1536\) | Endpoint appears only as a reported number. No curve, per-layer values, metric implementation, fit metadata, arrays, or code were available. | **Not reproduced** | The representation-capacity claim remains plausible but unverified. |
| **B. Pooled-within Taylor** | Layer 16: offset \(0.586\to0.476\), error \(5.39\mathrm e{-3}\to5.58\mathrm e{-3}\). Layer 29: offset \(0.574\to0.357\), error \(4.00\mathrm e{-3}\to5.41\mathrm e{-3}\). | Values are consistently transcribed in the memo and ledger, but no rerun was possible. The comparison supports covariance-reference insensitivity; it does not directly isolate mean offsets. | **Numbers traced; experiment not reproduced** | Closes the tested recentering attempt provisionally. “Mean-offset dominated” should be qualified as supported, not experimentally established. |
| **C. Direct/Hermite rank sweep** | Errors at ranks 4, 16, 64, 128; only \(r=128\) passes. | Error values could not be regenerated. The rank-cost arithmetic is independently correct: \(r_{\max}=4.3488\), and at \(r=128\), \(2n^2r=n^3=16{,}777{,}216\) for \(n=256\). | **Cost identity reproduced; empirical errors not reproduced** | The tested shortcut loses its cost advantage if the reported rank requirement is correct. |
| **D. Official Mini-100 benchmark** | Adjusted \(1.4641716\mathrm e{-7}\); raw MSE \(2.2819432\mathrm e{-7}\); multiplier \(0.6427\); zero failures. | Headline values are repeatedly reported, but the JSON, exact package archive, package hash, network identities and per-network rows are unavailable. Several arithmetic checks pass; one tail-label inconsistency was found. | **Reported official run; not independently rerun** | Baseline remains the only reported runnable candidate, but “independently validated” is too strong. |
| **E. Root estimator failure** | `dot()` rejects `out=` under FlopScope 0.9.1; every MLP fails. | Available evidence documents a 2/2 smoke-test failure, not a complete 100-network rerun. Package selection and fallback paths could not be audited. | **Plausible; not reproduced** | Quarantine remains appropriate, but the scope of the failure claim must be narrowed. |

---

## 3. Experiment A — Mixture representation ladder

### What could be verified

The following claim is consistently present:

\[
\delta\approx1.781\times10^{-3}
\qquad\text{at}\qquad K=1536.
\]

It is used to argue that representation accuracy exists even though affordable evaluation does not.

### What could not be reproduced

None of the additional Prompt 1 requirements can be recovered from the supplied material:

- mean error across layers;
- median error;
- worst-layer error;
- layer 16 error;
- layer 29 error;
- the precise mathematical definition of \(\delta\);
- sampling uncertainty or a measurement floor;
- the complete sequence of tested \(K\);
- fitting seeds or initialization sensitivity;
- whether PCA dimension, \(K\), covariance structure, or clustering was selected using the same targets on which closure was evaluated.

### Oracle-use finding

The materials do not document the exact fitting procedure for the \(K=1536\) ladder sufficiently to answer whether it used oracle activation samples.

Related project documentation explicitly recognizes that clustering true activation rows creates oracle component labels and that per-layer rank selection using target performance is oracle tuning. That establishes what would be illegal in a deployable rollout, but it does not establish exactly what the missing ladder script did.

**Conclusion A:** The \(K=1536\) result is an **oracle representation claim pending reproduction**, not a legal estimator result.

---

## 4. Experiment B — Pooled-within Taylor recentering

The reported values are:

| Layer | Global offset | Pooled-within offset | Global second-order error | Pooled-within error |
|---:|---:|---:|---:|---:|
| 16 | 0.586 | 0.476 | \(5.39\times10^{-3}\) | \(5.58\times10^{-3}\) |
| 29 | 0.574 | 0.357 | \(4.00\times10^{-3}\) | \(5.41\times10^{-3}\) |

### What the experiment supports

The covariance offsets decrease by approximately:

- layer 16: \(18.8\%\);
- layer 29: \(37.8\%\).

Nevertheless, the reported Taylor errors increase by approximately:

- layer 16: \(3.5\%\);
- layer 29: \(35.3\%\).

Thus reducing the measured covariance offset does not reduce the observed total approximation error.

### What it does not establish

Prompt 1 requested an intervention that independently holds component means fixed while changing only the covariance reference. The reported experiment changes the covariance reference while leaving the same mean offsets present, but it does not separately evaluate:

1. a mean-offset-only Taylor approximation;
2. a covariance-offset-only approximation;
3. their interaction;
4. a controlled zero-mean-offset counterfactual.

Therefore the strongest defensible statement is:

> In the tested \(K=64\) cases, the total Taylor error is insensitive to a substantial reduction in the measured covariance offset. This supports—but does not by itself prove—that the remaining error is dominated by mean offsets or mean–covariance interactions.

The current ledger already uses the softer word “supports,” which is appropriate. Claims that the experiment “proved” mean domination are overstated.

**Conclusion B:** Pooled-within recentering appears unsuccessful. The causal mechanism remains incompletely isolated.

---

## 5. Experiment C — Direct/Hermite low-rank evaluation

### Reported empirical errors

| Layer | \(r=4\) | \(r=16\) | \(r=64\) | \(r=128\) |
|---:|---:|---:|---:|---:|
| 16 | 0.216 | 0.0544 | 0.00673 | 0.000786 |
| 29 | 0.174 | 0.0400 | 0.00376 | 0.000223 |

The reported gate is approximately \(1.5\times10^{-3}\), so only \(r=128\) passes.

These errors could not be independently regenerated.

### Independently verified cost arithmetic

Using the reported width \(n=256\) and budget \(570{,}000\) FLOPs per component:

\[
r_{\max}
=
\frac{570{,}000}{2(256)^2}
=
4.3487548828125.
\]

Thus “affordable rank \(r\le4.4\)” is correct.

At \(r=128\):

\[
2n^2r
=
2(256)^2(128)
=
16{,}777{,}216,
\]

while

\[
n^3=(256)^3=16{,}777{,}216.
\]

The equality is exact, not merely approximate.

### Omitted cost audit

The \(2n^2r\) figure is not the complete algorithm cost. It is the factor-application cost for a low-rank quadratic contraction. The supplied record does not account exactly for:

- eigendecomposition or truncated factor construction;
- sorting/selecting eigenmodes;
- conversion from covariance to correlations;
- component-specific marginal standardization;
- Hermite coefficient evaluation;
- construction or application of \(R^{\circ d}\) for \(d\ge2\);
- summation over Hermite orders;
- PSD stabilization;
- storage and data movement;
- any per-layer or per-component factor reuse;
- special-function calls.

Therefore:

> At the reported required rank, the cheapest displayed contraction already costs \(n^3\); the complete Hermite evaluator costs at least that much and probably more.

That strengthens the closure of this particular implementation. It is not an information-theoretic lower bound on every possible structured diagonal algorithm.

**Conclusion C:** The cost squeeze is mathematically sound conditional on the unreplicated rank sweep.

---

## 6. Experiment D — Official 129-basis Mini-100 benchmark

### Reported values

| Metric | Reported |
|---|---:|
| Adjusted score | \(1.4641716\times10^{-7}\) |
| Raw MSE | \(2.2819432\times10^{-7}\) |
| Mean multiplier | 0.6427 |
| Effective compute | \(1.748\times10^{11}\) |
| Estimator FLOPs | \(1.70873\times10^{11}\) |
| Failures | 0 / 100 |
| Charged residual wall time | 39.4 ms/MLP |
| End-to-end observed wall time | approximately 16.5 s/MLP |

The narrative says the run used the subprocess runner, the official Phase-1 Mini split, all 100 networks, and four BLAS threads.

### Arithmetic checks

#### Local prediction agreement

\[
\frac{2.2826\times10^{-7}-2.2819432\times10^{-7}}
     {2.2819432\times10^{-7}}
=
0.0287825\%.
\]

The “within 0.03%” statement is correct.

#### Effective-compute consistency

The transcript gives exact estimator FLOPs of:

\[
170{,}875{,}096{,}064.
\]

At the reported charge of \(10^{11}\) FLOP-equivalents per second, 39.4 ms contributes:

\[
0.0394(10^{11})=3{,}940{,}000{,}000.
\]

Hence:

\[
170{,}875{,}096{,}064+3{,}940{,}000{,}000
=
174{,}815{,}096{,}064
\approx1.74815\times10^{11}.
\]

This agrees with the reported \(1.748\times10^{11}\).

#### Small FLOP transcription discrepancy

The exact value is:

\[
1.70875096064\times10^{11},
\]

whereas v31 records \(1.70873\times10^{11}\). The difference is 2,096,064 FLOPs, approximately \(0.00123\%\). It does not affect the score conclusion, but it is not ordinary rounding to the displayed precision.

#### Adjusted-score aggregation

Multiplying the reported mean raw MSE by the reported mean multiplier gives:

\[
(2.2819432\times10^{-7})(0.6427)
=
1.4666049\times10^{-7},
\]

which is \(0.166\%\) above the reported adjusted score.

This is not necessarily a discrepancy. If adjustment is calculated per network and then averaged, the mean of products need not equal the product of means. The missing JSON is required to verify the aggregation.

### Package and hash status

Prior shipping records identify a package containing:

- `estimator.py`;
- `fast_matmul.py`;
- `kerdock_mub5_seed3.npz`;
- `.whestignore`.

They also provide a full asset hash:

```text
58eac1b69707b204d00f6d50cf4e1996b1fcd566154ec93a7ecb5668c1acbfad
```

and a full production `fast_matmul.py` hash:

```text
fb1b93cb625b66ce5f26220ea3b6b685dbb9887d50f8756cafa9426577d45085
```

However, those hashes are associated with a prior `production_partial_tree_source` package. A separate audit marks that source package’s estimator hash as mismatched. Consequently, these cannot be asserted as the hashes of the exact 129-basis package used in the reported official run.

### Per-network tail discrepancy

The ledger says the worst network error was \(8.52\times10^{-7}\), approximately \(5.8\times\) the mean and approximately 6% of total loss.

But:

\[
\frac{8.52\times10^{-7}}
     {2.2819432\times10^{-7}}
=
3.73,
\]

whereas:

\[
\frac{8.52\times10^{-7}}
     {1.4641716\times10^{-7}}
=
5.82.
\]

Thus the \(5.8\times\) and 6% statements are consistent only if \(8.52\times10^{-7}\) is an **adjusted per-network score**, not a raw MSE:

\[
\frac{8.52\times10^{-7}}
     {100(1.4641716\times10^{-7})}
=
5.82\%.
\]

If it is raw MSE, the contribution is only 3.73%.

The current description “best raw/adjusted network error” mixes two distinct metrics and is invalid as written. The archived JSON is needed to label the quantity correctly.

### Command and environment reconstruction

The exact command line was not preserved in the available artifacts. The strongest recoverable schematic invocation is:

```bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

whest run <whestbench_final_estimator_20260730> \
  --runner subprocess \
  --split mini \
  <all-100-networks option> \
  <JSON-output option>
```

This is **not an exact reproduction command**. The CLI flag spelling, package form, network-count option, output option, Python version, NumPy version, operating system, processor, and environment lockfile are absent.

Narratively reported environment:

- `whestbench` 0.13.0;
- FlopScope 0.9.1;
- subprocess runner;
- BLAS thread count 4;
- official Phase-1 Mini;
- 100 networks;
- machine otherwise idle.

**Conclusion D:** Headline figures are arithmetically credible but not independently reproducible from the v31 artifact set.

---

## 7. Experiment E — Root estimator failure

The reported exception is:

```text
TypeError: dot() got an unexpected keyword argument 'out'
```

under FlopScope 0.9.1.

The underlying narrative records a **2-of-2 smoke test**, followed by the broader statement that every MLP fails.

Without the root package, harness script and failure JSON, the following remain unverified:

- whether failure is deterministic across repeated processes;
- whether all 100 Mini MLPs fail;
- whether execution reaches the same `dot(out=...)` call for every network;
- whether the harness substitutes zeros, throws, or activates another fallback;
- whether multiplier 1.0 is forced by the harness;
- whether any packaging or submission script can select the root estimator;
- whether a relative-path or default-package ambiguity exists.

The API-level explanation is plausible: if the executed `FlopScopeArray.dot` method does not accept `out=`, a call using it should fail independently of network values. But the exact package/API combination was not independently inspected.

**Conclusion E:** Quarantine is justified. Replace “every MLP was independently shown to fail” with “the reported 2/2 smoke test failed through an apparently input-independent API incompatibility.”

---

## 8. Discrepancies and whether they alter the canonical conclusion

| Discrepancy | Severity | Changes conclusion? |
|---|---|---|
| Raw scripts, arrays, exact package and official JSON absent | Major evidence defect | Changes evidence status, not current practical recommendation |
| \(K=1536\) ladder lacks curve, metric definition and leakage audit | Major | Representation claim remains provisional |
| “Mean-offset dominated” not experimentally isolated | Moderate | Recentring remains negative; causal wording must soften |
| \(2n^2r\) omits factor construction and higher Hermite orders | Moderate | Strengthens, rather than weakens, closure of tested implementation |
| Exact estimator FLOPs differ slightly from ledger shorthand | Minor | No |
| Mean raw × mean multiplier differs from adjusted mean | Unresolved | Probably aggregation, but JSON required |
| Worst-network metric mixes raw and adjusted errors | Material labeling error | Changes M210 tail statement and contribution calculation |
| Root failure evidence is 2/2, not an archived all-100 run | Moderate | Quarantine still stands |
| Exact official package hash not tied to reported run | Major provenance defect | Baseline result remains reported, not independently authenticated |

---

## 9. Classification of conclusions

### Reproduced

1. The claimed 0.03% agreement is arithmetically correct: \(0.02878\%\).
2. The affordable rank under the stated budget is \(r\le4.3488\).
3. At \(n=256,r=128\), \(2n^2r=n^3=16{,}777{,}216\) exactly.
4. The stated effective compute is consistent with the exact reported estimator FLOPs plus 39.4 ms of charged residual time.
5. The \(5.8\times\) tail statement is mathematically consistent with the adjusted-score mean, not with the raw-MSE mean.

### Plausible but not reproduced

1. The full mixture closure-error ladder.
2. The \(K=1536\) endpoint.
3. The pooled-within Taylor measurements.
4. The rank-sweep error measurements.
5. The official Mini-100 headline result and zero failures.
6. The local analytic propagation’s end-to-end agreement.
7. The root estimator’s input-independent incompatibility.

### Overstated

1. “Mean-offset dominated” when presented as directly proved rather than strongly suggested.
2. “Every MLP fails” when the archived evidence described in the transcript is a 2/2 smoke test.
3. “Production baseline is end-to-end validated” without the exact package, JSON and hashes being attached.
4. Any claim that the direct/Hermite result is an information-theoretic lower bound.
5. Any claim that all analytic mixtures or compact joint states are closed.

### Invalid as written

1. The M210 phrase “best raw/adjusted network error” and the comparison of \(8.52\times10^{-7}\) to an unspecified mean.
2. A 6% raw-loss contribution claim using the supplied raw mean.
3. Any exact package-hash claim that substitutes the prior partial-tree package hashes for the missing official 129-basis package.

---

## 10. Proposed canonical ledger patch

| ID | Proposed patch |
|---|---|
| **T106** | Replace “mean offsets dominate” with: “Pooled-within recentering substantially reduced covariance-offset norms without reducing total second-order error. This supports mean-offset or mean–covariance-interaction domination; no fixed-mean factorial ablation was attached.” |
| **T107** | Add: “The empirical rank sweep has not been independently regenerated. \(2n^2r\) covers the displayed factor contraction, not eigendecomposition, factor construction or higher Hermite orders. The cost identity at \(r=128\) is independently verified.” |
| **M205** | Change status from “Closed tested construction” to **“Reported local closure; independent rerun open.”** Preserve closure of pooled-within recentering, but soften the causal mechanism claim. |
| **M206** | Change evidence text to: “Reported rank errors require \(r=128\). Independent arithmetic confirms the displayed contraction alone equals \(n^3\); full cost is no smaller. Raw rank-sweep artifacts remain missing.” |
| **M207** | Preserve **“Mostly closed / replication gate.”** Add that the conclusion is conditional on independent regeneration of the mixture ladder and rank/Taylor measurements. Do not broaden beyond exact componentwise, shared-reference Taylor and the tested low-rank construction. |
| **M208** | Change status from **“Production validated”** to **“Reported official exposed result; artifact verification pending.”** Record exact reported FLOPs as 170,875,096,064. Add: official JSON and exact package hash not attached. |
| **M209** | Replace “every MLP fails” with: “Reported 2/2 smoke-test failure from an apparently network-independent FlopScope 0.9.1 API incompatibility. Determinism, all-network behavior, fallbacks and package-selection paths were not independently audited.” Keep quarantine verdict. |
| **M210** | Replace “best raw/adjusted network error” with a metric-specific field. Provisionally label \(8.52\times10^{-7}\) **adjusted per-network score**, since only that interpretation yields \(5.8\times\) the mean and a 5.8% contribution. Require JSON confirmation before the tail audit. |
| **M211** | Add: “Independent Prompt-1 audit completed at the documentary and arithmetic level. Experimental reproduction failed because required scripts, arrays, package and JSON were absent. No substantive closure was overturned; evidence tiers were corrected.” |

---

## 11. Final canonical conclusion

The correct current conclusion is:

> The reported local evidence strongly suggests that the current heteroscedastic full-covariance mixture implementation has no affordable evaluator among exact componentwise propagation, shared-reference first/second-order Taylor approximation, and the tested low-rank Hermite/direct-diagonal route. The cost arithmetic is sound, but the empirical inputs have not been independently reproduced.

The 129-basis estimator remains the only reported runnable candidate and should remain the practical shipping baseline. That recommendation follows from the absence of a better validated package, not from artifact-complete independent reproduction of M208.

The root estimator should remain quarantined.

The independent-reproduction gate remains **OPEN** until the following are attached together:

1. all 44 regeneration scripts;
2. frozen configuration and seeds;
3. saved activation/mixture arrays;
4. exact metric definitions;
5. exact package archive and SHA-256;
6. exact asset hash;
7. environment lockfile;
8. command transcript;
9. `official_129basis_mini100_20260731.json`;
10. root-estimator smoke-test output and package-selection audit.

No new estimator was proposed, and no protected data were used, as required by Prompt 1.
