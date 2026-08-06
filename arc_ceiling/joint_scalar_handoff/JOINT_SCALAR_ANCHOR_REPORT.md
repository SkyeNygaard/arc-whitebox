# Joint Scalar Anchor — Canonical Audit and Narrow Handoff

**Date:** 2026-07-29  
**Terminal state:** `EXTERNALLY_BLOCKED_WITH_TESTED_SUBFAMILIES_FAILED`

## Decision

The lane is **not a compressed-K3 project**. The correct deployable object is a frozen batch of target-level scalar functionals that jointly composes the radial-Hermite anchor.

For the canonical 32-probe construction, freeze exactly 128 scalar slots before deduplication:

1. target mean projections, `z_p = E[v_p^T h]`;
2. marginal second moments, `s_p = E[h_i^2]`;
3. row-direction second moments, `u_p = E[h_i(v_p^T h)]`;
4. cubic contractions, `r_p = E[h_i^2(v_p^T h)]`.

With observable Kerdock center `m`, width `D`, selected row `i_p`, and direction `v_p`, the exact scalar composition is

```text
a_p = (r_p - (m^T v_p)s_p - 2m_i u_p + 2m_i^2 z_p) / (D + 1).
```

`joint_scalar_contract.py` verifies that this 4K-scalar formula is numerically identical to contracting the complete exact anchor matrix. It also verifies the equivalent true-centered connected-cubic-plus-lower-order representation.

The **current tested implementations fail**. The **full-depth analytic/shared-arithmetic joint-scalar class remains untested**, because the canonical package lacks the real/fresh weights, high-precision moments, frozen probe arrays, and three helper modules needed for the composed-control runner.

## Hard gate translated into MSE ratios

The canonical 32-probe complete exact anchor has aggregate candidate/base ratio

```text
r_exact = 0.2239952669.
```

Retaining 70% of its MSE reduction requires

```text
r_candidate <= 1 - 0.70(1-r_exact) = 0.4567966868.
```

The preferred 90% bar is

```text
r_candidate <= 0.3015957402.
```

Both are far stricter than ordinary adjusted-score break-even. The lane should therefore be judged directly on retained exact-anchor improvement, not on a small positive raw gain.

## What the component ablation proves

| Anchor supplied | Aggregate ratio | Exact improvement retained |
|---|---:|---:|
| Complete exact | 0.223995 | 100.0% |
| Exact lower-order only | 0.633203 | 47.27% |
| Exact connected only | 1.580440 | -74.80% |
| Tested fixed joint pilot | 1.008463 | -1.09% |
| Frozen one-basis lower holdout | 1.049361 | -6.36% |
| Same-design lower crossfit | 1.006825 | -0.88% |

Lower-order recentering removes 36.68 percentage points of baseline MSE ratio by itself. The complete anchor removes a further 40.92 points. Connected cubic is therefore a **conditional interaction channel**: useful only when the lower-order coordinates are simultaneously correct. It must never be evaluated or deployed as an isolated correction again.

## Source localization × compute intersection

The exact adjoint identity is solved: the archived source decomposition closes to approximately `2e-15`, and rank 4 captures a median 96.65% of terminal defect Frobenius energy. Compression is not the blocker.

The signed source is deep:

| Cumulative suffix | Median signed connected defect |
|---:|---:|
| 8 transitions | 23.55% |
| 16 transitions | 33.79% |
| 24 transitions | 82.49% |
| Full depth | 100% |

The independent 4,096-row source model produces the following relevant cost boundary:

| Support K | d=16 | d=24 | d=30 |
|---:|---:|---:|---:|
| 8 | 9.57B | 14.20B | 17.67B |
| 16 | 10.45B | 15.36B | 19.05B |
| 32 | 12.19B | 17.68B | 21.80B |

No tested suffix depth is both:

- below the 14B hard ceiling; and
- deep enough to contain at least 70% of the median signed connected defect.

This is a necessary diagnostic rather than a formal MSE-retention theorem, but it is already sufficient to reject ordinary short-suffix regeneration and the 4,096-row independent full-depth source stream.

A reduced 1,536-row full-depth stream was analytically estimated near 8.9B for a narrow shared span, but no accuracy evidence exists. It is not a candidate until it demonstrates the real lower-order tolerances and the complete composed-control MSE gate without tuning.

## Why the implemented checkpoint does not pass

The archive includes a real width-256 checkpoint/adjoint implementation:

- `q32_h24`: measured 10.338B, 2.31% relative error versus the factorized target, 31.15% versus oracle;
- `q32_h27`: measured 7.315B, 7.74% versus factorized, 35.00% versus oracle;
- `q64_h27`: measured 10.909B, 1.58% versus factorized, 30.65% versus oracle.

These runs profile **two connected controls**, not the complete 32–128-scalar anchor. They omit the joint lower-order state and the unchanged composed-control replay. Their excellent agreement with the factorized target proves the adjoint implementation, not the source model. The factorized target remains about 30% from oracle, so increasing adjoint rank cannot close the central error.

## Narrow experiment that remains scientifically live

Only two implementation families remain admissible.

### A. Full-depth shared-arithmetic scalar recurrence — primary

Freeze the 32 observable sample-row probes and propagate the 128 scalar slots as one batched state. Reuse one common pulled-back basis and one local source calculation per layer. Do not materialize covariance or K3 tensors. Do not introduce independent full-width trajectories.

The recurrence must report at every layer:

- the four scalar families and their exact-reference error;
- signed correction cosine;
- cumulative source contribution;
- charged FLOPs by mean, second, cubic, composition, and replay;
- numerical telescoping residual.

A source model is acceptable only if the same frozen equations estimate both lower-order and cubic terms. A connected-only recurrence is out of scope.

### B. Inherited **joint-state** checkpoint — fallback

A checkpoint is admissible only when it supplies the absolute signed 4K-scalar state at the checkpoint. It may then use the exact adjoint/source recurrence for the suffix. A checkpoint that supplies only factorized connected c21 is already insufficient.

The checkpoint must be validated directly against the same scalar target manifest. Same-cloud folds cannot certify its absolute expectation.

## Preregistered stopping rule

Run at most one frozen candidate from A and one from B. No nearby shrinkage grid.

Reject the lane when either condition holds:

1. honest added effective compute is `>=14B`; or
2. aggregate candidate/base is `>0.4567966868`, equivalent to retaining less than 70% of complete exact-anchor improvement.

The preferred continuation bar is candidate/base `<=0.3015957402` under 10B.

Also require positive mean signed cosine, a majority of network wins, and worst candidate/base `<=1.10`. These safety checks are implemented by `evaluate_joint_scalar_candidate.py`.

## Frozen evaluation protocol

1. Use only a new immutable network block. Canonical global IDs 0–199 and sparse continuation IDs 0–23 are exposed.
2. Select probes from weights and baseline observables only.
3. Freeze all target definitions, basis ranks, checkpoint, local source equations, coefficients, and shrinkage before looking at final truth.
4. Save exact-reference and candidate values for all 128 scalar slots per network.
5. Compose anchors only with `joint_scalar_contract.compose_anchor` or an algebraically identical implementation.
6. Replay the unchanged frozen composed control and final layer.
7. Charge recurrence, checkpoint construction, eigendecompositions, scalar composition, sparse replay, packing, and residual wall time.
8. Evaluate with `evaluate_joint_scalar_candidate.py`.

## Missing assets preventing a real completion run

- official/fresh weight files used by the local ARC harness;
- high-precision target moment corpora;
- frozen sparse probe arrays and coefficients for a fresh composed-control block;
- `sparse_adjoint_control.py`;
- `sparse_crossfit_lower_fast.py`;
- `sparse_lower_moment_pilot.py`.

The canonical launch pack explicitly lists the three modules as missing standalone dependencies. The code archive contains the adjoint and source machinery but not the official data or those exact helper interfaces.

## Reproduction

```bash
cd joint_scalar_handoff

python test_joint_scalar_contract.py

python audit_joint_scalar_lane.py \
  --sources /path/to/launch_pack/sources \
  --arc-results /path/to/arc_code/arc_ceiling/results \
  --out JOINT_SCALAR_ANCHOR_AUDIT.json

python evaluate_joint_scalar_candidate.py candidate_results.json \
  --out candidate_gate.json
```

Expected local test output:

```text
PASS test_scalar_anchor_matches_full_matrix
PASS test_connected_plus_lower_reconstructs_raw
PASS test_gate_math
PASS test_canonical_adjoint_telescope
```

## Final verdict

**Stop** all ordinary short-suffix, one-basis lower, same-cloud crossfit, Gaussian connected-only, and existing fixed-joint variants.

**Continue only** with one full-depth shared-arithmetic 128-scalar recurrence and, if a genuinely absolute checkpoint source exists, one inherited joint-state checkpoint. The first candidate that misses `0.4568` or exceeds `14B` closes this enabling lane.
