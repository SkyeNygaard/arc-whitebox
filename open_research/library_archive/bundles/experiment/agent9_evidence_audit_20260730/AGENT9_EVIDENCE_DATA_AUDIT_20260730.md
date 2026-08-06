# Agent 9 — Evidence and Data Audit of Agents 1–8

**Date:** 2026-07-30  
**Canonical ledger reviewed:** `whestbench_canonical_research_ledger_20260729_reconciled_v15_paper_proof_audit.xlsx`  
**Decision:** **ACCEPT THE PAPER PROGRAM WITH MAJOR EVIDENCE-GOVERNANCE CORRECTIONS.**

## Executive verdict

The eight reviews materially strengthen the paper, but they reveal an uneven evidence base:

- The central **T22/T23 certificate is paper-ready** after correcting one stale statement from a positive “gap” to a one-sided certified upper bound.
- **T27, the T16 all-degree tail, the signed-negative-mass stability proposition, the layer-31 abstract risk theory, and the narrow ReLU-Stein theorem** are defensible when their scopes are stated exactly.
- The **T4 `1.127854` result, degree-6+ `1.004439` result, and public high-row accounting analysis** are traceable enough to use with correct stage and attribution labels.
- The reported **M146 (`1.012`, `41.2x`, `~5e-4`), M76 numerical range, M145 width-scaling numbers, and every M152 number are not paper-ready** because the original rows/scripts/splits/reference metadata are missing.

This is not a minor documentation issue. The manuscript’s strongest safe form is a certified theorem boundary plus a carefully normalized falsification program. It must not depend on empirical numbers that exist only in a transcript or ledger summary.

## Evidence standards used

I treated these as distinct gates:

1. **Claim validity:** does the proof or experiment support the stated conclusion?
2. **Numerical traceability:** can each number be followed to a source file, rows, metric convention, cohort, reference construction, and selection chronology?
3. **Independent reproducibility:** can the saved bundle rerun the computation from its declared inputs, with hashes?

A proof can pass gate 1 while a bundle fails gate 3. An empirical number can be plausible and internally consistent while failing gates 2 and 3. The paper claim status below reflects the weakest relevant gate.

## Agent-by-agent adjudication

| Agent | Adjudication | Reproducibility | Required correction |
|---|---|---|---|
| 1 — T22 hostile math review | **Verified after specified corrections** | Manifest and independent computation pass | State a one-sided upper bound, not a proved positive gap; tighten randomized-rule and finite-width scope. |
| 2 — T22/T23 certificate audit | **T22 verified; T23 verified after documentation/release corrections** | Clean-room reproduction matched 32/32 tracked files and 23/23 chunks | Say “rigorous computer-assisted proof,” not proof-assistant formalization; a hash manifest is not authenticated provenance. |
| 3 — T27 adversary | **Verified after specified corrections** | Manifest and script rerun pass | Use a symmetrized-line budget and fixed Kerdock universe; do not imply arbitrary-node or adaptive signed optimality. |
| 4 — T16 proof | **All-degree reduced-cost negativity proved** | Manifest and script rerun pass | Full LP-optimality language remains conditional on primal-attainment/complementarity. |
| 5 — signed-weight extension | **Mathematically verified, practically weak** | Manifest and both certificate scripts pass | The negative-mass thresholds are far too small to close useful unrestricted signed rules. |
| 6 — anchor theory/adversary | **Theory accepted; M146 empirical source missing** | Theory/synthetic bundle passes; original M146 unavailable | Remove universal `~5e-4` threshold; report a downstream-weighted, direction-dependent condition. |
| 7 — scalar learning | **M152 not reproducible; scoped Path2 substitute negative** | Internal outputs verify, but full rerun needs an omitted archive | Do not relabel the substitute as M152. Remove all M152 numerical claims. |
| 8 — harmonic taxonomy | **Accept with scope corrections** | Algebra script passes; review ZIP omitted five cited source inputs | Package source inputs; restrict exact claims to named classes and empirical failure to the frozen degree-6+8 rule. |

## Priority headline audit

### 1. T22: `0.0233655%`

**Disposition: keep, but rewrite.**

The proof package certifies that, inside the stated class, the Kerdock rule’s relative excess lies in

`[0, 0.02336550102949%]`.

It does **not** prove that Kerdock is positively suboptimal by that amount. The safe wording is:

> Kerdock is within at most 0.0233655% of the optimum among fixed or network-independent randomized, nonnegative linear rules using at most 66,048 nodes for the infinite-width d=256, depth-32 normalized ReLU kernel.

The certificate was independently reconstructed and reproduced byte-for-byte. This is the strongest numerical headline in the project.

### 2. M146: layer-31/layer-32 ratio `1.012`

**Disposition: remove from headline; provisional only.**

The ledger reports approximately `5.55e-4` at layer 31 and `5.48e-4` at layer 32, but I could not locate the source rows, script, network/rotation manifest, reference streams, or metric implementation. Agent 6 correctly classifies the ratio as empirical rather than theorem-level. Its audit only proves that the reported points are mutually consistent, not that the experiment occurred as recorded.

### 3. M146: exact-anchor gain `41.2x`

**Disposition: remove exact multiplier pending source recovery.**

This is an oracle diagnostic and would never by itself establish a legal estimator. More importantly, the archived evidence does not support independent citation of the multiplier. The paper may state qualitatively that exact replacement was reported to have large oracle value, clearly labeled as provisional, or omit it entirely.

### 4. M146: `~5e-4` tolerance

**Disposition: reject as a universal threshold.**

Fitting the four reported points gives a break-even around `5.7985e-4` and an excellent quadratic fit, but this only establishes arithmetic consistency. Agent 6’s adversarial constructions show direction-dependent break-even scales spanning roughly `1.87e-4` to `1.41e-2`, and kink-focused perturbations can have a large nonlinear replay remainder. The correct theorem-level object is downstream-weighted anchor error plus a ReLU gate-crossing remainder, not scalar Euclidean relative error.

### 5. T4: raw ratio `1.127854`

**Disposition: keep as a frozen development negative result.**

This is the best-traced empirical headline in the audit:

- width 256, depth 32;
- base networks `6000–6015`;
- rotations `3, 11, 97`;
- six independent scrambled Sobol streams, aggregated into two `196,608`-sample halves;
- 2,250-rule policy grid frozen before literal-rotation labels;
- raw ratio `1.127854`, noise-corrected `1.145174`, `17/48` wins, worst `2.480711`;
- calibration `6016–6023` and validation `6024–6031` remained sealed.

The manuscript must call this **development**, not validation. Its value is a clean preregistered branch closure, not general impossibility.

### 6. M76 and M145 learning evidence

**Disposition: downgrade numeric claims.**

M76’s `0.94–0.99x` range appears only in the canonical summary; the decisive comparison package was not found. M145’s width-scaling values and shared-rank geometry appear in `Pasted text(57).txt`, while the ledger explicitly records the scripts as unpersisted scratchpad dependencies. These results are scientifically plausible and consistent with the broader negative map, but exact numbers should not enter the paper until rows, splits, metric code, and hashes are restored.

The qualitative conclusion remains supportable:

> Tested pooled or weights-only representations did not recover the signed, network-specific Kerdock defect.

### 7. M152

**Disposition: remove completely.**

The claimed 1,100-network result has no recoverable target equation, feature list, data manifest, grouped split, preprocessing, regularization chronology, row predictions, script, or hashes. Agent 7 therefore could not reproduce it. The independent Path2 substitute is useful but is not M152: it uses 32 development rows/24 groups, 36 holdout rows/12 groups, and 1,528 legal features. In that substitute, the bounded ensemble was worse than a constant with the same mean by `0.0091857` ratio, with grouped 95% interval `[0.004108, 0.013573]`.

M152 should be marked **OPEN / UNVERIFIED** and excluded from every paper claim table.

### 8. Degree-6+ harmonic correction: `1.004439`

**Disposition: keep narrowly.**

The selected four-direction degree-6+8 rule was chosen on 16 development networks, frozen, and evaluated on new seeds `10016–10031`. Each network used two independent `262,144`-point Sobol-sphere truth streams and the `66,048`-point Kerdock design. The frozen raw-MSE ratio was `1.004439`, with `6/16` wins and bootstrap interval `[0.996366, 1.013471]`.

This supports:

> A frozen small degree-6+8 zonal dictionary failed to validate.

It does not support “degree-6+ controls cannot help,” “analytic controls are low degree,” or any universal high-degree impossibility statement.

### 9. Public high-row accounting curve

**Disposition: keep as explanatory forensics with no participant attribution.**

The released 90,624-row package reports raw MSE `2.045449e-7`, adjusted score `3.899377e-8`, effective compute `51.847B`, and zero tracked FLOPs, despite at least `242.726B` intended later-layer arithmetic. A fresh 12-network high-row experiment measured a noise-corrected Sobol exponent `1.046616`; using that exponent predicts the visible rank-8 point within `0.139%`.

This establishes **explanatory sufficiency**: high-row sampling plus accounting can explain the frontier. It does not identify any participant’s private method and should not be framed as attribution.

## Additional proof results worth promoting

### T16

Agent 4 certifies every finite degree `6..14,658` and an analytic negative tail for all `l >= 14,659`. This should be added to the proof register. The paper must distinguish this reduced-cost theorem from the stronger claim of full all-degree LP optimality, which requires an exact primal-attainment/complementarity certificate.

### Signed negative-mass stability

Agent 5 verifies the formal bound and directed-interval exclusion curve. The thresholds are so small that the result functions mainly as a robustness lemma and limitation. It does not close unrestricted signed quadrature.

### Blockwise ReLU-Stein annihilation

Agent 8 provides a clean exact theorem for fixed bias-free one-hidden-layer ReLU Stein fields over antipodal orthonormal-basis blocks after radialization, plus a numerical check at `1.15e-14`. This is a useful new narrow theorem. The Poisson-kernel construction also gives an explicit counterexample to “analytically integrable implies low harmonic degree.”

### Layer-31 abstract theory

Agent 6’s correction-risk identity, replacement condition, correlated-noise shrinkage formula, common-bias non-identifiability theorem, and ReLU crossing bound are paper-ready under their explicit assumptions. They should be presented separately from the missing M146 empirical package.

## Reproducibility and packaging defects

1. **Agent 7 is not self-contained.** Its internal report, predictions, null tests, and hashes are coherent, but `agent7_scalar_audit.py` requires an omitted Path2 source archive. Archive the exact inputs or mark the bundle as derived-output-only.
2. **Agent 8 is not self-contained.** Its internal 11 review files verify, but its manifest references five inputs outside the ZIP. The Library contains the three principal named sources with matching hashes; the release bundle should include them.
3. **M146 has no source package.** Do not confuse a successful quadratic consistency check with reproduction.
4. **M76/M145 are archival debt.** Recover the original decisive-comparison rows and local scratchpad scripts before using exact numbers.
5. **T4 is well packaged.** It has a frozen config, row-level outputs, scripts, and hashes. Preserve it as the empirical packaging standard.
6. **T22/T23 is the gold standard.** The clean-room recreation, interval certificates, tamper tests, and independent numerical path are sufficient for a rigorous computer-assisted claim, subject to transparent trust-base language.

## Paper-ready claim matrix

### Keep now

- Scoped T22 one-sided near-optimality certificate.
- Scoped T27 optimum on the fixed Kerdock line universe.
- T16 all-degree reduced-cost negativity, with conditional full-LP caveat.
- Signed negative-mass stability as a weak robustness lemma.
- Abstract correction/replacement/non-identifiability/ReLU-remainder theory.
- T4 frozen-development failure.
- Frozen degree-6+8 validation failure.
- Narrow blockwise one-hidden-layer ReLU-Stein annihilation.
- Public high-row/accounting explanatory module, without attribution.

### Remove or make explicitly provisional

- M146 `1.012`, `41.2x`, and universal `~5e-4` wording.
- M76 exact `0.94–0.99x` range.
- M145 exact width-scaling/shared-rank numbers.
- Every M152 number.
- Any universal “no statistical path,” “all analytic controls are low degree,” or unrestricted signed-weight optimality claim.

## Final decision

The evidence supports a strong paper with this thesis:

> A certified boundary for static nonnegative neural cubature, exact theory for when white-box corrections can help, and a reproducibly scoped falsification map showing that the tested anchor, harmonic, learning, and companion information classes do not recover the required signed network-specific defect.

It does not support a universal no-free-lunch theorem for every adaptive nonlinear white-box estimator. The missing empirical packages should be treated as unresolved archival dependencies, not filled in by confident prose.

## Deliverables

- `CLAIM_EVIDENCE_REGISTER.csv` — row-level claim provenance and disposition.
- `REPRODUCIBILITY_AUDIT.json` — bundle checks, hashes, and dependency failures.
- `PROPOSED_LEDGER_CHANGES.md` — ledger-ready status changes.
- `SHA256SUMS.txt` — hashes for this Agent 9 package.
