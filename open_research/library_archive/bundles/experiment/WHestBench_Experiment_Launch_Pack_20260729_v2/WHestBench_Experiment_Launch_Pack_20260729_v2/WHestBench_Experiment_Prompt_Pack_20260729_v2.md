# WHestBench Experiment Prompt Pack — sparse-anchor reconciliation 2026-07-29 13:17 ET

These prompts are execution contracts for sandbox agents. They must finish in one of three states: **pass**, **fail**, or **externally blocked with a complete runnable local handoff**. A response that merely proposes more work is incomplete.

## Global execution contract

Before any experiment, search the shared Library for and read:

1. `/WHestBench/Canonical/whestbench_canonical_research_ledger_20260729_merged_v2.xlsx`.
2. `WHestBench_Canonical_Research_State_2026-07-29_reconciled.docx` (`file_0000000011a481f5974400a414382a55`).
3. `arc_code.zip` (`file_00000000111081f5a79d98e19461e7f8`).
4. `/WHestBench/Canonical/arc_code_deep_audit_20260729.md` (`file_000000008d6881f5ba0e7a9b689ea234`).
5. `/WHestBench/Canonical/WHestBench_Experiment_Prompt_Pack_20260729_v2.md`.
6. The exact branch artifacts listed in the selected prompt and in the ledger’s **Library Artifact Map**.

The canonical ratio is **candidate MSE / baseline MSE**; lower is better. When reporting inverse gain, label it explicitly.

### Evidence and split rules

- Global network IDs `0–199` are exposed. Do not use them for a fresh statistical claim.
- The new sparse-anchor development networks `0–7` and lower-pilot holdout networks `8–23` are also exposed.
- Generate a new immutable base-network split. Keep all rotations of one base network in one fold.
- Freeze configurations, feature definitions, shrinkage, probes, seeds, continuation gates and hashes before opening validation.
- Report reference noise and use network-level bootstrap intervals.
- Sandbox absence is not scientific falsification. Missing assets must yield a complete local runner and explicit import schema.

### Required experiment structure

```text
experiment/
├── TASK.md
├── ARTIFACT_MANIFEST.json
├── frozen_config.json
├── split_manifest.json
├── src/
├── tests/
├── results/
│   ├── development.json
│   ├── frozen_validation.json
│   ├── per_network.csv
│   └── correction_vectors.npz
├── report.md
└── LOCAL_HANDOFF.md
```

### Search and recovery rule

Search the Library by both exact filename and content. The following helper modules were imported by a late script but were not found as standalone Library artifacts:

- `sparse_adjoint_control.py`
- `sparse_crossfit_lower_fast.py`
- `sparse_lower_moment_pilot.py`

Search `arc_code.zip`, local experiment directories and bundles for them. When absent, recreate only the explicitly used interfaces with tests, mark the result as a compatibility implementation, and do not claim exact reproduction until seeds and outputs match.

### Rescue and stopping rule

After a primary failure, permit exactly:

1. one diagnostic ablation identifying whether the failure is support, sign, scale, lower-order state, connected source, or final replay; and
2. one mechanistically justified rescue directed at that measured failure.

Do not launch a neighboring hyperparameter sweep.

Terminal states:

1. **Pass:** preregistered gate clears with frozen evidence, code, exact/proxy cost labels, tails and hashes.
2. **Fail:** gate fails on appropriate evidence; produce a scoped closure certificate and preserve reusable code.
3. **Externally blocked:** missing local data prevents the real run; produce a tested runner, exact missing-asset manifest, commands, expected inputs and output schema.

Do not ask whether to continue. Continue until one terminal state is reached.

---

# Prompt 1 — Integrated compiler measurement

## Required Library artifacts

- `NONK3_CONTINUATION_REPORT.md` (`file_00000000986081f5949f2a3676f5354c`)
- `NONK3_CONTINUATION_BUNDLE_20260728.zip` (`file_000000003f8c81f6b2cd7ea89ab32805`)
- `arc_code.zip`
- Canonical v2 merged ledger

## Task

Integrate the assembly-free/partial-tree Kerdock baseline, frozen two-layer compiler, fixed three-layer compiler and adaptive depth-2–6 compiler into the same current production package and run one paired official-style measurement.

Do not compare projections derived from different baselines. The old absolute `~1.442e-7` adaptive projection is stale.

Measure:

- raw final-layer MSE;
- exact FlopScope cost;
- subprocess residual wall time and effective compute;
- peak memory;
- packing and symbolic-composition overhead;
- fallback frequency;
- per-network approximation error;
- wins, median, worst ratio and network-bootstrap interval.

Primary candidate specifications:

- fixed three layers, 2,064-row balanced pilot, rare threshold 8, middle correction 0, final shrinkage 0.875, full-suffix fallback when predicted cost ratio exceeds 0.995;
- adaptive depths 2–6 with the same pilot and minimum-saving guard;
- frozen two-layer fallback;
- unchanged production baseline.

**Gate:** adopt only a measured adjusted-score improvement with no material runtime, memory, prediction or tail regression. Otherwise preserve the production baseline and issue a scoped compiler decision.

---

# Prompt 2 — Fresh sparse radial-Hermite exact-anchor validation

## Required Library artifacts

- `sparse_radial_cubic_control.py` (`file_000000000920822f8d33deead109d167`)
- `sparse_radial_highref8_merged.json` (`file_000000000774822f9c5a2f8488020739`)
- `lowerpilot_screen8_merged.json` (`file_000000000a28820c8777db95e9668613`)
- `sparse_cubic_center_and_channel_report.md` (`file_00000000d87c81f5a8778af85303e347`)
- `arc_code.zip`
- Canonical v2 merged ledger

## Known prior results to reproduce before validation

On exposed development networks `0–7`:

- 128 sample-row probes: candidate/base `0.109280`, 8/8 wins, worst `0.258139`;
- diagonal probes: `0.245454`, 8/8, worst `0.403817`;
- complete 32-probe exact anchor: `0.223995`;
- exact lower-order anchor only: `0.633203`;
- exact connected-c21 correction only: `1.580440`.

The sample-row probe construction is observable, but the anchor is high-sample oracle information. Do not call this deployable.

## Task

1. Recover the exact radialized feature, Kerdock point ordering, target layer, sample-row probe selector, basis-block folds, ridge normalization and exact anchor definition.
2. Write hashes and reproduce the exposed development summaries within numerical tolerance.
3. Freeze one primary construction: 128 sample-row probes, complete basis-block crossfit, exact complete anchor.
4. Generate at least 24 genuinely new width-256/depth-32 networks with two independent high-quality reference halves.
5. Run the frozen exact-anchor candidate and the mandatory component matrix:
   - sample anchor;
   - complete exact anchor;
   - exact lower-order mean/covariance terms only;
   - exact connected term only;
   - complete anchor with mean omitted;
   - complete anchor with pair moments omitted.
6. Save probe indices/directions, fitted coefficients, correction vectors, reference halves and per-network final outputs.

Measure the downstream-weighted singular spectrum of the complete anchor defect. “Shared” does not imply low rank; proceed to a rank-16 model only when rank 16 retains at least 90% of relevant correction energy.

**Primary gate:** complete exact-anchor candidate/base `<=0.50`, preferably `<=0.30`, with a confidence interval and tails compatible with a winning-scale mechanism. Lower-only should explain a material fraction, while connected-only is diagnostic and is not expected to pass.

When the result fails, close the exact frozen radial-Hermite construction—not every sparse cubic control.

**Terminal deliverable:** `SPARSE_RADIAL_PROBE_MANIFEST.json`, frozen validation results, component ablation, vectors and a pass/fail certificate.

---

# Prompt 3 — Inherited-checkpoint or full-depth joint-scalar recurrence

## Required Library artifacts

- `adjoint_c21_prototype.py` (`file_000000000fbc822faa65d4ff42a7d22d`)
- `adjoint_source_localization.py` (`file_00000000f90c81f5abbc5c376b0128dc`)
- `adjoint_source_full8_merged.json` (`file_000000001d38822fbf1bd9b1a60c8b17`)
- `AGENT2_EXACT_SPARSE_ADJOINT_REPORT.md` (`file_0000000013d481f5a3693fa6824a9883`)
- `sparse_joint_adjoint_pilot.py` (`file_000000001074822fa010b71a103bd34e`)
- `joint_adjoint_pilot_screen8.json` (`file_00000000fb7081f5b7412dc7309d093e`)
- Frozen probes and component definitions from Prompt 2
- `arc_code.zip`

## Current conclusion that must be respected

The backward identity is solved. Direct and reconstructed contractions agree to about `2e-15`.

On eight exposed width-256 networks:

- rank 4 captures median `96.65%` of terminal c21-defect Frobenius energy;
- last 4/8/16/24 transitions supply median signed fractions `12.91%/23.55%/33.79%/82.49%`;
- median effective source-layer count is `19.22`.

Therefore a cheap last-4/8 source is not the default hypothesis. Do not spend time deriving another pullback identity or casually truncating the source.

## Task

Build a target-contracted recurrence for the **joint scalar anchor**, not connected c21 alone. The frozen target set should include only functionals needed by Prompt 2:

- selected target means;
- selected marginal second moments;
- selected row-direction second moments;
- selected connected-cubic contractions.

Evaluate two implementation families:

1. **Inherited checkpoint:** estimate a reliable contracted state at a selected checkpoint, then propagate all remaining scalar source contributions.
2. **Full-depth recurrence:** generate/approximate each required scalar source from layer 0 to target without constructing full covariance or K3 tensors.

Mandatory comparisons:

- oracle joint scalar state;
- exact inherited plus exact source;
- factorized inherited plus exact source;
- exact inherited plus candidate source;
- candidate inherited plus candidate source;
- lower-order candidate without connected source;
- connected candidate without lower-order state;
- full complete candidate.

For every layer, save signed source contribution, cumulative reconstructed anchor, error versus oracle and marginal cost. Judge final composed control MSE, not anchor cosine alone.

A truncated suffix may be reported only as an ablation. It may become the candidate only when a frozen cohort shows depth `<=12` retains at least 90% of complete-anchor final-MSE benefit.

**Gate:** legal complete candidate retains at least 70% of exact-anchor MSE reduction, preferably 90%, with added compute below 10B preferred and 14B hard, positive signed correction cosine and safe tails.

When the imported helper modules are absent, recreate the narrow interfaces used by `sparse_joint_adjoint_pilot.py`, test against the stored JSON, and produce a local handoff. Do not call a compatibility harness an exact reproduction without matching hashes.

---

# Prompt 4 — Sparse radial-Hermite layer-31 residual control

## Required Library artifacts

- Prompt 2 radial-Hermite artifacts and frozen probe output
- `lowerpilot_frozen_holdout16.json` (`file_000000005544820ca96e64b7ce5c4591`)
- `crossfit_lower_fast_screen8.json` (`file_000000008b14822f996b94d7b121488e`)
- `joint_adjoint_pilot_screen8.json`
- Layer-31 oracle reports and bundles, including `AGENT1_FINAL_REPORT.md`
- `arc_code.zip`
- Optional Prompt 3 scalar source outputs

## Current closures to respect

Do not retune:

- one independent Kerdock-basis lower-moment pilots: frozen holdout candidate/base `1.04936`, 6/16, worst `1.25430`;
- same-design fold-crossfit lower anchors: `1.006825`;
- tested joint lower-pilot plus Gaussian/adjoint connected source: `1.008463`, worst `1.23148`;
- connected-only correction.

## Objective

Use the frozen sparse radial-Hermite basis to construct a layer-31 surrogate \(g_{31}\), then estimate

\[
\widehat{\mu}_{31}
=
\widehat{\mathbb E[g_{31}]}
+
Q_K(a_{31}-g_{31}),
\]

followed by one true final-layer replay.

This formulation must be evaluated as a residual control. Do not judge \(g_{31}\) as a standalone activation model and do not require a coherent full deep moment state.

## Stage A — Oracle expectation ceiling

Freeze the feature construction and regression without examining the new validation targets. Supply a high-quality oracle \(\mathbb E[g_{31}]\) only to answer:

- Does the sparse basis produce a low-variance Kerdock residual?
- How much of the full layer-31 oracle correction does it capture?
- Which 8–32 modes or coordinates dominate?
- Is the final replay linearization accurate for this correction?

Continue only when oracle-\(E[g]\) residual control captures at least 50% of the full layer-31 oracle MSE reduction and wins on at least 75% of fresh networks.

## Stage B — Legal expectation

Test at most three expectation sources:

1. Prompt 3 joint-scalar analytic/checkpoint recurrence;
2. one algebraically complete-design-zero-mean shared-arithmetic companion statistic;
3. one small permutation-equivariant predictor of the residual of the analytic anchor.

Do not use ordinary extra MUB/Haar/rotated trajectories.

## Gate

- raw final replay gain at least `1.30x`;
- consistently positive correction cosine;
- at least 80% wins;
- worst candidate/base `<=1.10–1.15`;
- positive adjusted score after expectation, inference and replay cost.

Mandatory ablation: compare direct anchor correction, \(g_{31}\) residual control and a matched lower-order-only \(g_{31}\). This establishes whether residualization solves the anchor problem or merely hides it.

---

# Prompt 5 — Shared-arithmetic external phase estimator

## Required Library artifacts

- `FULL_MULTILEVEL_CONTINUATION_REPORT.md` (`file_00000000a83481f5956084d3120e0df9`)
- `FULL_MULTILEVEL_CONTINUATION_RESULTS.json` (`file_00000000020481f58979a7741c7e3a0d`)
- `BLOCKWISE_CUBIC_FROZEN_ROWS.csv` (`file_000000006f24820c9998742d81c2c3dd`)
- `FULL_MULTILEVEL_CONTINUATION_BUNDLE.zip` (`file_00000000bd9081f5995c7bbd31f47441`)
- `FRESH_SCREEN_SUMMARY.json` (`file_00000000b4a081f5853bef823cbf3561`)
- `AGENT1_CONTINUATION_SUMMARY.json` (`file_000000004e8081f5936f9f028281a0fe`)
- `ARC_ACTIVATION_REGION_CONTINUATION_REPORT.md` (`file_00000000ecac820c8a556da9f809231b`)
- Prompt 2 radial-Hermite artifacts
- `arc_code.zip`

## Task

V80 blockwise H3 and M105 radial-Hermite features are useful representations, but same-design gates are closed. Do not retune block variance, fold stability, safety margins, amplitudes, ridge, shrinkage or two-pilot agreement.

Freeze either:

- the V80 blockwise correction vector; or
- the Prompt 2 joint-anchor / Prompt 4 \(g_{31}\) residual correction.

Design exactly one source of external phase information that reuses almost all existing Kerdock/Winograd prefix arithmetic. Ordinary extra trajectories are forbidden.

The new statistic must be genuinely different information, such as:

- an algebraically zero-mean complete-design companion transform;
- a second contraction available from already-computed FWHT/Winograd intermediates;
- an independently derived analytic source projection.

Evaluate sign, scale, confidence/suppression, correction cosine and final replay gain.

Use at least 50 untouched width-256 base networks for the frozen test.

**Gate:** true incremental compute `<=1–2%`, adjusted-score interval excludes no gain, positive signed correlation and worst candidate/base `<=1.10`.

---

# Prompt 6 — Target-free layer-31 support and sparse-Hermite coefficients

## Required Library artifacts

- `agent1_layer31_deployability_round.zip` (`file_0000000017c481f5906cf70c0286686a`)
- `AGENT1_CONTINUATION_SUMMARY.json`
- `FRESH_SCREEN_SUMMARY.json`
- `ARC_ACTIVATION_REGION_CONTINUATION_REPORT.md`
- `activation_region_continuation_bundle.zip` (`file_00000000ccfc822fb48019aa1a2910ce`)
- Prompt 2 radial-Hermite script/results
- Prompt 3/4 outputs when available
- `arc_code.zip`

## Task

Separate support discovery from signed amplitude estimation.

### Stage A — target-free support

Freeze at most four selector families for K=8, 12, 16 and 32, using only weights and baseline observables. Include a sparse-Hermite selector derived from sample-row probe loadings or downstream sensitivity.

Evaluate exact signed corrections on a new cohort.

Continue only when K<=32 captures at least 50% of the full layer-31 oracle gap, wins on at least 75% of networks and is stable under allowed rotations.

### Stage B — amplitude or residual expectation

On the single passing support, test at most three legal sources:

1. Prompt 3 target-contracted joint scalar source;
2. Prompt 5 shared-arithmetic statistic;
3. a small equivariant predictor of the residual of the best independent analytic anchor.

Direct \(g_{31}\) residualization from Prompt 4 should be the default scored formulation.

Do not add ordinary independent trajectories.

**Gate:** raw final replay gain `>=1.30`, positive coefficient/correction cosine, worst candidate/base `<=1.25` at screening and `<=1.10–1.15` for promotion, and positive adjusted score.

---

# Prompt 7 — One width-256 edge-state DWS predictor

## Required Library artifacts

- `EQUIVARIANT_WEIGHT_MODEL_RESEARCH_20260729.md` (`file_00000000b9a8820cafe11b4e1fe38f01`)
- `equivariant_weight_model_repro_20260729.zip` (`file_00000000af48822f9d5f3bb43fb1b6e7`)
- Frozen low-dimensional labels from Prompts 4–6
- Canonical v2 ledger and split registry

## Precondition

Do not run broad weight-to-answer learning. Begin only after one concrete label is frozen:

- joint-anchor residual functionals;
- \(g_{31}\) residual coefficients;
- V80 correction sign/scale/confidence.

## Task

Train one permutation-equivariant width-256 edge-state model. Group every rotation of one base network in one fold. Train through the scored final replay or an exactly verified linear surrogate.

Compare against:

- constant shrinkage;
- invariant linear/ridge baseline;
- the analytic or shared-arithmetic anchor without learned correction.

Output only the frozen low-dimensional correction, scale and confidence—not 256 final answers.

Report base-network split sizes, hashes, equivariance and leakage tests, raw/adjusted replay, cosine, calibration, wins, worst, interval and inference cost.

**Gate:** grouped width-256 raw replay gain `>=1.15`, adjusted interval excludes no gain, worst candidate/base `<=1.10`, and inference cost is repaid. Otherwise pause the tested model class.

---

# Prompt 8 — Qualitatively new network-specific coreset representation

## Required Library artifacts

- `NETWORK_SPECIFIC_KERNEL_CORESET_EXACT_KERDOCK_REPORT.md` (`file_000000005adc81f591d9eabb791ad169`)
- `arc_code.zip`
- Canonical v2 ledger and contradiction map
- Optional near-free features produced by Prompts 3–6

## Task

Do not run another K2 kernel, optimizer, q-grid, exchange-round, signed-weight or calibration sweep. The exact K2 representation is closed.

Before selecting a coreset, test one qualitatively new, nearly free representation against oracle row importance/support on fresh exact-geometry width-256 networks. The representation must come from another live branch’s already-computed features or a genuinely new nonlinear phase-sensitive feature.

Measure row-importance correlation, top-support recall, same-support oracle-weight added MSE, uniform/optimized-weight MSE, selection/evaluation cost and worst network.

Only run the optimizer when the representation predicts a support whose same-support oracle added MSE is `<=1.1e-8` with safe tails. Complete selection/evaluation cost must be smaller than saved propagation cost.

A failure closes only the tested representation, not the mathematical existence of network-specific coresets.
