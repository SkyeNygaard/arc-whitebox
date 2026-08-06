# Equivariant weight-to-answer research round

**ARC White-Box Estimation Challenge 2026 — July 29, 2026**

## Executive verdict

This round does **not** support launching an open-ended, enormous weight-space training program as the next competition priority.

The prior negative result was tested fairly aggressively rather than merely repeated:

- the target was changed from the final Kerdock error to the much stronger **layer-31 post-ReLU mean-defect channel**;
- a forward-only equivariant model was compared with a genuinely bidirectional weight-space encoder;
- high-capacity nonlinear probes received the complete layer-31 Kerdock basis pattern;
- the training corpus was expanded to 1,024 independent networks with separate 128-network calibration and validation sets;
- width 64 was tested with two independent high-precision labels on calibration and validation;
- an exact, label-preserving **orthogonal phase augmentation** multiplied 512 ground-truth networks into 2,048 residual examples while exposing the rotated first-layer weights to the model.

The best weight-model result was a **1.0114x raw-MSE gain**, with 95% network-bootstrap interval **0.9827x–1.0326x**, and a worst-network candidate/baseline MSE ratio of **6.12x**. The gain is uncertain and too small: an extra final-layer replay alone costs 3.125%, so the learned correction needs at least **1.03125x raw gain merely to break even**, before charging model inference.

The branch is therefore **not submission-ready and not currently competitive**.

This is not a universal impossibility result. One final, bounded experiment remains scientifically defensible: train at the actual width 256 on the public high-precision challenge corpus, with grouped orthogonal phase augmentation, a full edge-state Deep Weight Space model, and a score-aligned layer-31/final-replay loss. It should be preregistered and stopped unless cross-validated raw gain is at least 1.10x with controlled tails.

## Why the target was redesigned

The earlier model tried to predict the final Kerdock residual directly. That model learned broad diagonal mean-field bias but failed on Kerdock residual phase.

The newest full-width oracle result changes the most defensible target. Correcting only the layer-31 post-ReLU Kerdock mean and replaying the true final layer removed 78.13% of noise-corrected MSE on 64 untouched width-256 networks, a 4.572x gain. The channel is highly aligned with final error, and replay costs only 3.125% extra compute in the idealized accounting.

Accordingly, every new model in this round predicted

```text
delta_31(W, K) = true_mean_31(W) - Kerdock_mean_31(W)
```

The predicted mean defect was applied by exact nonnegative particle translation, followed by a true final-layer replay. This evaluates the composed estimator, not just target-vector R-squared.

## Architectures tested

### Forward target-redesign control

A 23,394-parameter permutation-equivariant message-passing model close to the earlier architecture. It propagates per-neuron states forward through all 32 weight matrices using contractions with standardized weights, centered squared weights, centered absolute weights, local row summaries, Kerdock trajectory statistics, and layer embeddings.

### Bidirectional DWS-style encoder

A 24k–30k parameter compact bidirectional model with:

- forward prefix states;
- backward suffix states;
- contractions through `W`, `W^2 - E[W^2]`, centered `|W|`, and signed square-root weight channels;
- row, column, and global equivariant summaries;
- a set encoder over complete antipodal Kerdock basis blocks;
- layer-31 and final residual heads.

Hidden-neuron permutations commute with all operations. This is qualitatively different from flattening the weight tensors.

### High-capacity observable-signal probes

Ridge and LightGBM models were trained per neuron on:

- basis-block means, standard deviations, zero fractions, maxima, and antipodal-pair imbalance;
- ordered and basis-invariant sorted versions;
- final-layer column diagnostics;
- layer-31 Kerdock moments;
- layerwise global invariant summaries.

These probes ask whether useful nonlinear signal exists even when the neural architecture is imperfect.

## Data and validation

### Width-32 initial screen

- 224 training networks
- 32 calibration networks
- 64 untouched validation networks
- exact 1,088-point real Kerdock design
- independent antithetic Sobol labels

### Width-64 screen

- 176 training networks
- 32 calibration networks
- 32 untouched validation networks
- exact 4,224-point real Kerdock design
- two independent 131,072-point antithetic Sobol references for every calibration and validation network

### Large width-32 corpus

- 1,024 training networks
- 128 calibration networks
- 128 untouched validation networks
- training labels from 32,768 antithetic Sobol points
- calibration and validation labels from two independent 131,072-point antithetic Sobol references

All splits are by independently generated base network. Model choice and shrinkage use calibration only.

### Orthogonal phase augmentation

For an orthogonal matrix `Q`, replacing the first layer by

```text
W_1' = W_1 Q^T
```

leaves every Gaussian activation expectation unchanged because `Qx` has the same distribution as `x`. It changes the fixed-orientation Kerdock evaluation, creating a new residual label from the same high-precision ground truth.

Four independent Haar rotations were generated for each of 512 base networks, producing 2,048 training examples. All rotations from a base network remain in the same split. This directly attacks the residual-phase data shortage without target leakage.

## Results

A gain above 1 is better. Confidence intervals bootstrap whole validation networks.

| Experiment | Train corpus | Validation | Replay gain | 95% interval | Worst cand/base |
|---|---:|---:|---:|---:|---:|
| Forward target-31 control | 224 x width 32 | 64 | **1.00235x** | 0.99989–1.00493 | 1.55x |
| Bidirectional compact | 224 x width 32 | 64 | **1.00287x** | 0.99048–1.00896 | 9.85x |
| Forward target-31 control | 176 x width 64 | 32 | **1.00300x** | 0.99166–1.01773 | 1.08x |
| Bidirectional compact | 176 x width 64 | 32 | **0.98933x** | 0.96033–1.03495 | 1.23x |
| Large nonlinear basis probe | 1,024 x width 32 | 128 | **1.00815x** | 0.94623–1.05560 | 4.62x |
| Phase-augmented basis probe, sorted | 512 bases x 4 rotations | 128 | **0.96380x** | 0.82991–1.07185 | 16.62x |
| Phase-augmented basis probe, ordered | 512 bases x 4 rotations | 128 | **0.97973x** | 0.84929–1.07632 | 8.08x |
| Identity-only weight encoder control | 512 x width 32 | 128 | **1.00013x** | 1.00002–1.00022 | 1.01x |
| Phase-augmented weight encoder | 512 bases x 4 rotations | 128 | **1.01142x** | 0.98266–1.03259 | 6.12x |

The tiny identity-control interval above one should not be treated as meaningful: its calibrated shrinkage was `-0.0032`, effectively switching the model off. The phase-augmented model had nonzero signal, but it remained weak and unstable.

## Cost implication

The full-width oracle accounting moves from 175.500B to 180.984B effective compute for one extra final-layer replay:

```text
180.984 / 175.500 = 1.0312479
```

Ignoring model inference, break-even therefore requires at least `1.03125x` raw-MSE gain. A 5% adjusted-score improvement requires approximately `1.0855x` raw gain. A 10% adjusted improvement requires approximately `1.1458x` raw gain.

The best observed learned gain, `1.0114x`, would make adjusted score about **1.96% worse** even if the model itself were free.

## Data-integrity correction

The first resumable dataset run exposed three rows whose weight tensors had been written before process interruption but whose QMC targets were still zero. The original resume detector checked only the weight array and incorrectly treated those rows as complete.

The generator was corrected to require nonzero completion across weights, both targets, Kerdock statistics, and layer-31 particles. The three affected rows were regenerated before any reported model comparison. Large-memmap checkpointing was also changed to avoid whole-file `msync` stalls.

## Interpretation

### What this round rules against

It is no longer persuasive that the earlier failure was simply caused by:

- predicting at the wrong layer;
- lacking suffix information;
- using too little nonlinear capacity;
- having only a few hundred training networks;
- omitting basis-level Kerdock diagnostics;
- or observing only one residual phase per ground truth.

Each of those explanations received a direct test. None produced enough robust score gain.

### What remains open

The following were not executed here:

- a true width-256 edge-state DWS model;
- training directly on all public width-256 high-precision labeled networks;
- many rotations per width-256 base network;
- end-to-end differentiation through a smooth approximation to exact layer-31 translation and final replay;
- downstream-sensitivity-weighted low-dimensional targets.

Width transfer has already failed elsewhere, so success at width 32 or 64 would not have been sufficient. Conversely, these negative reduced-width results cannot mathematically close a width-256 model trained on the actual distribution.

## The one bounded continuation worth running

### Corpus

Use the public width-256 high-precision corpus. Group all derived examples by base network. Use grouped five-fold cross-validation for architecture selection and keep the independently generated Mini-100 suite completely untouched until one model is frozen.

For each base network, generate 8 fixed orthogonal rotations. Reuse the exact ground-truth means; recompute only the protected Kerdock trajectory. This multiplies residual-phase examples without new reference simulation.

### Model

Use a full edge-state Deep Weight Space network rather than only node contractions:

- 8–16 edge channels for every weight matrix;
- equivariant row, column, global, previous-edge, and next-edge aggregations;
- forward and backward node states;
- residual and layer normalization;
- Kerdock basis-block tokens at layers 24, 28, 30, and 31;
- a final-layer sensitivity module.

### Target

Do not optimize unweighted `delta_31` MSE alone. Train through the scored channel.

A practical loss is:

1. predict a layer-31 correction;
2. apply a differentiable approximation to nonnegative mean translation;
3. replay the true final layer;
4. minimize final-layer MSE;
5. regularize the correction norm and worst-network loss.

A lower-dimensional alternative is to predict coefficients in an equivariant downstream-sensitivity basis generated from the empirical final-layer Jacobian, rather than predicting all 256 coordinates equally.

### Controls

Every run must include:

- the old forward equivariant architecture;
- the bidirectional node-only model;
- a zero-correction model;
- the diagonal mean-field positive control;
- an oracle layer-31 correction;
- identity-only versus grouped-rotation training.

### Preregistered continuation gate

Continue to Mini-100 only if grouped cross-validation shows all of:

- raw replay gain at least **1.10x**;
- 95% interval entirely above **1.05x**;
- worst-network candidate/baseline MSE at most **1.25x**;
- gain remains after exact model and replay FLOP accounting;
- no fold or rotation family is individually responsible for the result.

Anything weaker is unlikely to survive the official adjusted metric.

## Final decision

**Current priority: low.** Protect Kerdock plus the suffix compiler. Do not allocate an open-ended training program to the present node-message-passing family.

**Scientific status: narrowed, not closed.** The only justified reopening is the bounded width-256, public-corpus, grouped-rotation, edge-DWS experiment above. It has a clear mechanism, a legal offline-training path, and a hard stopping rule.
