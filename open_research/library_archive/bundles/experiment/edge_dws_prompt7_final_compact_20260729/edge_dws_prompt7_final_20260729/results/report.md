# Width-256 edge-state DWS predictor — terminal report

**Experiment:** Prompt 7, one bounded width-256 edge-state Deep Weight Space predictor  
**Date:** 2026-07-29  
**Terminal state:** **FAIL — PAUSE TESTED MODEL CLASS**  
**Protected test use:** opened once after the epoch-20 checkpoint and all calibration choices were frozen

## Executive verdict

The tested model failed decisively. On the untouched 16-base, 32-rotation test block, the calibration-frozen edge-DWS achieved:

- **noise-corrected raw gain:** `0.95375x` baseline/candidate, so the model increased MSE by about 4.85%;
- **noise-corrected 95% grouped interval:** `0.89499–1.02344x`;
- **compute-adjusted gain:** `0.87422x`;
- **adjusted 95% grouped interval:** `0.82036–0.93810x`;
- **wins:** `3/16` base networks;
- **worst observed candidate/baseline:** `1.29308x`;
- **worst noise-corrected candidate/baseline:** `1.30655x`.

All preregistered gates failed. The tested model class must be paused under this target, architecture and data protocol.

## Frozen target and replay

The target was the frozen scalar V80 optimal signed scale for the unchanged blockwise-H3 correction. The analytic anchor coefficient was `0.25`. The model predicted only one residual coefficient plus scale and confidence; it never predicted 256 final answers.

The scored replay was an exactly verified affine surrogate:

```text
MSE(scale) = ||baseline_error - scale * correction_direction||² / 256
```

- Development replay parabolas were reconstructed exactly from stored baseline MSE, optimal scale and regenerated frozen correction vectors.
- Fresh test rows stored direct baseline-error and correction vectors.
- Verified surrogate error: exactly `0` in the frozen manifest.

## Data and split

| Split | Examples | Base networks | Status |
|---|---:|---:|---|
| Train | 34 | 34 | exposed Prompt-5 development |
| Calibration | 8 | 8 | exposed Prompt-5 development |
| Validation | 8 | 8 | exposed Prompt-5 development |
| Test | 32 | 16 | freshly generated, untouched before freeze |

Each test base has two orthogonal rotations. Both rotations remain in the same test group. No base appears in multiple splits. Global IDs `0–199` are rejected by the registry.

Test references used two independent groups of eight complete Kerdock rotations per base. Mean reference-noise floor was `2.1885e-08`, approximately 7.70% of observed baseline MSE.

## Model

- Width: `256`
- Depth: `32`
- Edge-state channels: `8`
- Node-state channels: `8`
- Ordered layer-token channels: `48`
- Edge/node message-passing passes: `1`
- Transformer layers: `2`
- Output dimension: `1` frozen residual coefficient, plus scale and confidence
- Parameters: `117,235`
- Training: 20 epochs, AdamW, learning rate `3e-4`, weight decay `1e-4`, gradient accumulation `8`
- Best checkpoint: epoch `4`, selected on calibration only
- Calibration-selected residual multiplier: `0.585`

Gradient checkpointing was used only to control memory. It is algebraically identical to the uncheckpointed forward path; the trained checkpoint produced bit-identical correction, scale and confidence in an execution-equivalence test.

## Training behavior

Best calibration raw gain was only `1.01785x`. Training loss continued to fall while calibration degraded below baseline, showing phase overfit. Later epochs returned near neutral but never surpassed epoch 4.

Total model-training wall time was approximately `30.9` minutes on five CPU cores.

## Controls and test replay

Gain is baseline MSE divided by candidate MSE; greater than one is better.

| Candidate | Raw gain | Adjusted gain | Wins | Median cand/base | Worst cand/base |
|---|---:|---:|---:|---:|---:|
| Analytic V80 anchor only | `0.87593x` | `0.86297x` | 3/16 | `1.13514` | `1.61876` |
| Constant shrinkage | `0.95717x` | `0.94301x` | 3/16 | `1.03860` | `1.29220` |
| Invariant ridge | `0.92811x` | `0.91437x` | 3/16 | `1.01471` | `1.44054` |
| Edge-DWS, uncalibrated residual | `0.99096x` | `0.90833x` | 6/16 | `1.00694` | `1.11530` |
| **Edge-DWS, calibration frozen** | **`0.95696x`** | **`0.87717x`** | **3/16** | **`1.03883`** | **`1.29308`** |

The calibrated edge-DWS is almost identical to constant shrinkage because it learned a nearly constant residual:

- mean predicted residual: `-0.108398`;
- standard deviation across test bases: `6.49e-06`;
- true residual standard deviation: `0.33772`;
- base-level Pearson correlation with the true residual: `-0.19473`.

Its nominal 14/16 sign accuracy is caused by target sign imbalance, not phase prediction. The prediction has effectively no network-specific variation.

## Cosine and calibration

For the calibration-frozen model on test:

- flattened correction cosine: `0.46928`;
- confidence Brier score: `0.24775`;
- confidence ECE-10: `0.08086`;
- mean confidence: `0.48711`;
- confidence standard deviation: `8.60e-06`.

The positive flattened cosine is not evidence of useful phase forecasting: the output is almost constant and its base-level Pearson correlation with the true residual is negative.

## Equivariance and leakage integrity

- Full-width hidden-permutation correction drift: `1.49e-08`
- Scale drift: `2.98e-08`
- Confidence drift: `0`
- Full-width equivariance threshold `1e-5`: pass
- Unit/integrity tests: `7/7` pass
- Target-shuffle input-leakage test: pass
- Duplicate base across splits rejected: pass
- Disallowed/exposed base IDs rejected: pass
- Dataset and split hash enforcement: pass
- All model inputs explicitly exclude targets, replay errors and replay Jacobians.

## Inference cost

- Analytical model inference: `13.32912B` FLOPs
- PyTorch profiler supported-operator count: `13.36960B` FLOPs
- Profiler / analytical ratio: `1.00304`
- Measured CPU inference on test: mean `0.7110 s`, median `0.7236 s`, p95 `0.8915 s` per rotation example
- Baseline effective compute: `175.500B`
- Frozen V80 anchor increment: `2.63625B`
- Model inference increment: `13.32912B`
- Total candidate compute: `191.46536B`
- Candidate/baseline compute multiplier: `1.09097x`

The model required at least `1.09097x` raw gain merely to repay inference and anchor cost. It achieved `0.95375x` noise-corrected raw gain.

## Gate

| Gate | Required | Result | Pass? |
|---|---:|---:|---|
| Grouped raw replay gain | `>=1.15x` | `0.95375x` noise-corrected | No |
| Adjusted interval excludes no gain | lower bound `>1` | `0.82036` | No |
| Worst candidate/base | `<=1.10` | `1.29308` observed | No |
| Inference cost repaid | adjusted gain `>1` | `0.87422x` | No |

**Overall: fail. Pause the tested width-256 edge-DWS model class.**

## Closure scope

This closes the exact tested form:

- V80 scalar residual target;
- one-pass, eight-channel width-256 edge-state DWS;
- 34/8/8 exposed-development split with a fresh 16-base grouped-rotation test;
- calibration-frozen residual shrinkage and the reported invariant controls.

It is not a theorem that every possible equivariant model or every future independent low-dimensional anchor must fail. Under the project’s stopping rule, however, no neighboring architecture sweep is justified.

## Frozen hashes

- Dataset SHA-256: `0236378ede6f61a9a3cb66f0202180ba3e78084acbee9c6d8d691e85892d59e9`
- Split registry file SHA-256: `0f70e91bdc9c4f8f4a979c201d3bd352c502a00e932436cad0251f32d3bc9f09`
- Canonical split-assignment SHA-256: `204fce78b9ce588ac8a747d9d4746d6b939deca92f4ba90e897b83defb57a472`
- Result JSON SHA-256: `3d1f9a06995fc0601b0769783ca1926fdcaccea57d3274007ef4bc2074ff0064`
- Model SHA-256: `a9293555dc6ca89225f7e5dc5c74031d0373a3b3a318311a0d6367eb8a7c56c5`
- Per-network CSV SHA-256: `f2308f0cb0212272508740721c101f5e3391b7fac0e397e355df5e587d1a5ef1`
