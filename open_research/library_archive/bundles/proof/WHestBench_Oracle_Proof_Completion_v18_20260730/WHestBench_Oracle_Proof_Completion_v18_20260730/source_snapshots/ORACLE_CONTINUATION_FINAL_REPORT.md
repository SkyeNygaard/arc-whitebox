# Corrected Oracle / Nonlinear Continuation — Final Experimental Report

**Date:** 2026-07-30  
**Disposition:** All feasible branches executed; no active tested branch passes its continuation gate.  
**Important scope:** This is a portfolio stop over named information classes, not a universal impossibility theorem.

## Executive result

The corrected program produced a sharper result than the old observability-gap story:

1. **The oracle ladder is not one accumulating scalar bias.** Signed increments between checkpoint repairs are mostly weakly correlated. Maximum absolute off-diagonal cosine was `0.146` on validation and `0.187` on confirmation.
2. **Late-layer repairability remains large, but the last two checkpoint increments are not most of the incremental energy.** Their combined shares were `8.23%` on validation and `9.84%` on confirmation. The late layer is a compressed repair interface for multiple upstream/downstream components.
3. **The five-source span contains substantial signal.** The per-case oracle span reached validation candidate/base `0.400` and confirmation `0.454`.
4. **The legal coefficient model failed.** Its selected development group-CV candidate/base ratio was `1.132` with worst `2.212`. On confirmation it scored `0.872` with worst `3.055`, worse than the fixed global linear combination `0.778`.
5. **The bounded edge-state experiment is stopped by its source ceiling, not by an impossibility claim.** The exact scalar oracle on the untouched test split gives only `1.144709x` raw gain, below the frozen `1.15x` gate. Cheap weight-summary models also fail.
6. **Rich exact-mean nonlinear controls were tested and failed in their named classes.** The 48-network projected-ReLU extension scored `1.01268` raw, `18/48` wins, worst `1.312`, and projected adjusted ratio `1.06687`.

The operational conclusion remains:

> **No active branch in the tested information classes clears a credible continuation gate under the current evidence, deadline, hardware, and resource constraints.**

It does **not** imply that no statistical, adaptive, or nonlinear path can exist.

## 1. Reproducibility and trust

Two recovered experiment archives were checked:

- Oracle-gap campaign: 213 tracked files.
- Reopened nonlinear-path campaign: 154 tracked files.

All 367 tracked hashes passed. The oracle campaign's deterministic summary and confirmation scripts were rerun, and the archive manifest still passed afterward. Unit tests passed:

- oracle-gap core tests: 3 passed;
- Edge-DWS implementation tests: 7 passed.

The large network propagations were not recomputed from scratch in this continuation; their row-level arrays were authenticated by archive hashes, and all downstream summaries used those preserved arrays.

## 2. Priority 0 — cross-layer coherence

### Method

For checkpoint depths `7, 15, 23, 27, 29, 30`, form each signed repair vector and then the increment from the previous checkpoint repair. Concatenate output-coordinate increments over all cases and compute cosine Gram matrices and energy shares.

### Results

| Split | Max |off-diagonal cosine| | Increment energy fractions |
|---|---:|---|
| Development | `0.302` | `0.293, 0.296, 0.217, 0.122, 0.052, 0.020` |
| Validation | `0.146` | `0.395, 0.177, 0.235, 0.111, 0.054, 0.029` |
| Confirmation | `0.187` | `0.274, 0.263, 0.250, 0.115, 0.066, 0.032` |

The sign pattern of small off-diagonal correlations is not stable across splits, but the low magnitude is. This rejects a simple picture in which each checkpoint reveals the same absolute-bias direction at increasing amplitude.

### Interpretation

Layer 31 remains an excellent **repair interface**, but a constructive estimator likely needs to recover several downstream-sensitive components. A one-number absolute-center diagnostic is structurally too narrow for the observed decomposition.

## 3. Priority 1 — downstream-weighted coefficient synthesis

### Frozen construction

Five legal correction sources were used:

- one selected shallow exact-mean final-output correction;
- four fixed companion orientation corrections.

Three prespecified feature sets and seven ridge penalties were evaluated by leave-one-base-network-out development CV. Runtime features contained correction geometry, probe/companion summaries, checkpoint summaries, and invariant weight summaries; no truth or oracle coefficients were runtime features.

### Results

| Split / policy | Candidate/base | Wins | Worst |
|---|---:|---:|---:|
| Development CV, selected feature rule | `1.132` | — | `2.212` |
| Validation, feature rule | `0.833` | `6/12` | `1.861` |
| Validation, global nonnegative | `0.675` | `7/12` | `1.523` |
| Confirmation, feature rule | `0.872` | `10/12` | `3.055` |
| Confirmation, global linear | `0.778` | `11/12` | `1.697` |
| Confirmation, per-case oracle span | `0.454` | `11/12` | `1.034` |

The feature-dependent rule did not pass development CV. Later split results are therefore diagnostics, not promotion evidence. They show that flexible coefficients add no reliable network-specific value beyond fixed global shrinkage and can create severe tails.

### Decision

Close this exact five-source, feature-set, ridge-policy class. Preserve the oracle span as a target for genuinely new representations.

## 4. Priority 2 — bounded width-256 phase prediction

### Exact label

For each of 68 independent networks, the frozen source direction was the difference between the projected exact-mean ReLU candidate and baseline output. The target was the exact optimal scalar in the additive replay equation.

### Source ceiling

| Split | Oracle candidate/base | Oracle raw gain |
|---|---:|---:|
| Train | `0.769` | `1.301x` |
| Calibration | `0.750` | `1.333x` |
| Validation | `0.918` | `1.089x` |
| Test | `0.874` | `1.144709x` |

The frozen gate required at least `1.15x` raw gain. Even perfect test-set coefficient prediction cannot pass. After the favorable measured source cost, the exact oracle would still improve adjusted score modestly, but it fails the preregistered raw scientific gate and leaves no margin for model error.

### Cheap observable models

Untouched test candidate/base ratios:

| Model | Raw ratio | Wins | Worst | Adjusted ratio |
|---|---:|---:|---:|---:|
| Frozen source coefficient | `1.052` | `6/16` | `1.312` | `1.108` |
| Invariant ridge | `1.070` | `3/16` | `1.369` | `1.128` |
| Extra Trees | `1.030` | `6/16` | `1.256` | `1.086` |
| Random forest | `1.011` | `7/16` | `1.320` | `1.066` |

None recovers transferable signed scale.

### Full Edge-DWS disposition

The frozen architecture passes all seven implementation/equivariance/leakage tests. On available CPU it required about `8.0 s` per forward. A reduced-capacity backprop run did not complete a report within a 15-minute cap. This is **not** treated as model failure. However, the exact source-ceiling result makes a full run on this label scientifically unnecessary.

Reopen the model only after freezing a source/basis whose untouched oracle ceiling clearly exceeds the target gate.

## 5. Priority 3 — nonlinear exact-mean controls

### Poisson family

The terminal network-derived mid-radius Poisson control scored candidate/base `1.03794` with `7/16` wins. This closes the tested dictionary, not all high-degree analytic controls.

### Projected shallow-ReLU family

The exact candidate was extended without refitting to 48 terminal networks:

- raw candidate/base: `1.01268`;
- cross-MSE ratio: `1.01625`;
- wins: `18/48`;
- p90: `1.192`;
- worst: `1.312`;
- bootstrap 95% interval: `[0.959, 1.062]`;
- favorable incremental compute: `9.398B`;
- projected adjusted candidate/base: `1.06687`.

The apparent earlier gain was concentrated in the hardest quartile and did not transfer broadly.

## 6. Updated scientific conclusion

The experiments support the following hierarchy:

- **Proved/scoped:** static Kerdock results and exact correction identities.
- **Replicated mechanism:** late-layer oracle repairability.
- **New mechanism refinement:** the repair ladder decomposes into approximately incoherent signed increments.
- **Empirical closures:** the named five-source coefficient model, cheap weight-summary phase models, tested Poisson controls, and tested projected shallow-ReLU controls.
- **Open:** new downstream bases, stronger exact-mean source families, network-adaptive nonlinear estimators, and edge-state models attached to a source with sufficient oracle ceiling.

The result is not “no nonlinear path exists.” It is that the current tested paths do not jointly clear signal, stability, tails, and compute gates.
