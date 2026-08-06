# Width-256 edge-state DWS predictor — terminal report

**Date:** 2026-07-29  
**Terminal state:** **EXTERNALLY BLOCKED**  
**Gate:** not evaluated  
**Protected validation/test:** not opened

## Decision

The requested actual-width training run cannot be performed honestly from the current Library snapshot. The precondition requires one frozen Prompt 4–6 low-dimensional label and the canonical grouped split registry. Neither is available as executable data. The V80 package contains scripts and MSE summaries but not the frozen correction vectors or matching weight tensors; the canonical v2 state still marks the `g31` residual-control label as untested. Therefore no width-256 base-network split sizes, replay metrics, cosine, calibration, wins, interval, or tail result can be reported.

This is **not** a failure of the edge-state DWS class, and the class is not paused by this run. Treating absent labels as a negative result would violate the experiment's evidence policy.

## Frozen implementation supplied

A single bounded model and runner are complete:

- **Edge state:** 8 channels on every one of the 32 width-256 weight matrices.
- **Node state:** 8 channels with incoming/outgoing row-column edge aggregates.
- **Messages:** row, column, global, source-node, and destination-node equivariant updates.
- **Ordered depth encoder:** 48-channel layer tokens and a two-layer Transformer over the 32-layer order.
- **Output:** only a frozen `D≤16` residual coefficient vector, nonnegative scale, and confidence; never 256 final answers.
- **Frozen size:** one message-passing pass; 116,587 parameters at `D=1`.
- **Training loss:** exact linear replay MSE plus small coefficient and confidence auxiliaries.
- **Controls:** anchor-only, calibration-frozen constant shrinkage, and invariant ridge.
- **Validation:** all rotations grouped by `base_network_id`; calibration alone selects residual shrink/ridge; test remains untouched until freeze.

For the V80 scalar contract, the label is the optimal signed scale

`scale* = <correction_direction, baseline - target> / ||correction_direction||²`,

with the original frozen 0.25 correction as the analytic anchor. The final-output correction direction gives an exactly affine replay surrogate.

## Integrity tests

| Check | Result |
|---|---:|
| Unit tests | **7 passed** |
| Hidden-neuron permutation invariance | **Pass** |
| Max correction difference after hidden permutations | `2.98e-08` |
| Max scale/confidence difference | `0` / `0` |
| Target-shuffle input leakage | **Pass** |
| Duplicate base network across splits rejected | **Pass** |
| Exposed/disallowed base IDs rejected | **Pass** |
| Label and split SHA-256 enforcement | **Pass** |
| Python compilation | **Pass** |

The equivariance test is a reduced-shape algebra test; it verifies the implementation, not predictive generalization.

## Requested reporting fields

| Field | Status |
|---|---|
| Base-network split sizes | **Unavailable — canonical registry missing** |
| Seeds and split hashes | Source/package hashes supplied; real split hash unavailable |
| Equivariance tests | Passed synthetic unit test |
| Leakage tests | Passed synthetic contract tests |
| Raw replay | Not run |
| Adjusted replay | Not run |
| Correction cosine | Not run |
| Calibration | Not run |
| Wins / median / worst | Not run |
| Grouped interval | Not run |
| Exact inference cost | Not run; analytical estimate supplied |

## Cost feasibility

For `D=1`, dense-operation accounting estimates **13.329B inference FLOPs**. With the 175.5B baseline and the frozen V80 anchor's 2.636B added compute, projected candidate compute is **191.465B**, or **1.09097×** baseline. Thus the model needs at least **1.09097× raw gain** merely to repay projected cost; at the preregistered 1.15× raw gate, projected adjusted gain is **1.0541×**.

This is not exact FlopScope accounting. Exact inference FLOPs, residual wall time, memory, and any required true final replay must be measured before scoring. The runner requires that extra replay cost be entered before training.

## Exact blockers

1. Frozen width-256 label/corpus bundle with matching weights and one concrete Prompt 4–6 target.
2. Frozen label manifest with source hashes and verified replay-surrogate evidence.
3. Canonical base-network split registry, including every exposed/disallowed ID.
4. CUDA execution plus exact FlopScope/wall-time environment for the real run.

The archive and V80 summaries are adequate to freeze the interface and model, but not to synthesize these missing scientific inputs.

## Handoff

Run `LOCAL_HANDOFF.md` exactly. The preflight deliberately refuses unfrozen manifests, hash mismatches, unverified surrogates, exposed IDs, split overlap, wrong width/depth, or labels wider than 16 dimensions. When real inputs are installed, one command trains the model and emits every requested gate metric. No neighboring architecture sweep is included.
