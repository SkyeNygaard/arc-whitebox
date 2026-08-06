# Path 6 — Compute Liberation and Implementation

**Date:** 2026-07-29  
**Workspace:** `paths/06_compute_liberation/`  
**Status:** Stage 0 complete; local implementation evidence is positive; official final-subprocess FlopScope and protected-score validation remain pending.

## Executive result

Two exact implementation opportunities survived the screen.

1. **Final-layer chunked direct accumulation** is the production-shaped candidate. It leaves the Kerdock set and first 31 layers unchanged, evaluates only the last layer in 2,048-row chunks, and reduces each chunk immediately. In a seven-pair single-thread local benchmark, final-layer median wall time fell from **2.032s to 0.990s** (51.27% lower), while isolated peak RSS fell from **2.03 GiB to 0.36 GiB** (82.42% lower). The exact operation audit projects only **47.5M** extra tracked FLOPs in the full package, so it breaks even if official residual time improves by merely **0.475 ms**.
2. **Cached-preactivation translation reuse** is an exact anchor-enabling primitive. When a correction uniformly translates layer-31 activations, it replaces a second full final-layer replay (**5.549B FLOPs**) with a shift/ReLU/reduction (**0.085B FLOPs**), saving **5.464B FLOPs** (98.47%). It matters only for formulations that truly require replay; the current direct-output control may already avoid that cost.

Neither result changes the estimator statistically. This path therefore does **not** independently satisfy the shared candidate/base <0.75 statistical gate; its gate is raw-MSE parity plus lower measured effective compute in the final subprocess.

## Stage 0 — Algebra and cost

### Candidate A: final-layer chunked direct accumulation

Let `H` be the unchanged 66,048×256 layer-31 activation and `W` the final weight. The baseline computes and materializes

`Y = ReLU(HW)`, then `mu = mean(Y, axis=0)`.

The candidate partitions rows into fixed chunks `C_j` and computes

`mu = (1/N) * sum_j sum_(i in C_j) ReLU(H_i W)`.

This is the same scoring identity and the same partial-tree Winograd kernel. Only floating-point reduction order differs. It is not a suffix compiler, pruning rule, or approximate propagation.

| Quantity | Baseline | Chunk 2,048 | Delta |
|---|---:|---:|---:|
| Final-layer tracked FLOPs | 5.548857B | 5.596379B | +47.522M |
| Projected package tracked FLOPs | 170.906815B | 170.954338B | +47.522M |
| Residual-time saving needed at 100B FLOP/s | — | 0.475 ms | — |
| Predicted raw candidate/base | 1.0 | ≈1.0 | numerical reduction order only |

**Expected wall impact:** positive locally; official impact unresolved because FlopScope and `whestbench` are not installed in this container. The package deliberately includes a fail-closed audit launcher rather than accepting a misleading local zero-FLOP result.

### Candidate B: cached-preactivation translation reuse

For a uniform penultimate translation `delta`,

`(H + 1 delta^T) W = HW + 1 (delta^T W)`.

Cache `Z=HW` from the baseline pass, compute the 256-vector `delta^T W`, broadcast it into `Z`, apply ReLU, and reduce. This is algebraically exact in real arithmetic.

| Quantity | Full replay | Dense reuse | Sparse q=8 reuse |
|---|---:|---:|---:|
| Tracked FLOPs | 5.548857B | 0.084672B | 0.084545B |
| Saving | — | 5.464184B | 5.464311B |
| Local median wall | 1.7602s | 0.0186s | same reduction path |

### Candidate C: K32/K128 summaries during existing propagation

Kerdock rows are already basis ordered. Basis means and selected low-dimensional Gramians can be reduced from the in-memory activation without another network pass.

| Feature | FLOP upper bound | Local median |
|---|---:|---:|
| K32 basis means | 12.583M | 2.516 ms |
| All 129 basis means | 50.725M | 16.384 ms |
| K32 selected 8-coordinate mean + Gram | 2.097M | 0.478 ms |

## Stage 1 — Development evidence

### Final-layer chunk candidate

- Paired benchmark: **2.032s → 0.990s**, 2.05× speedup for the final layer.
- Isolated peak RSS: **2080 MiB → 366 MiB**.
- Worst observed output-mean drift: max absolute **4.169e-07**, RMS **4.239e-08**. The corresponding squared drift is at most about **1.797e-15** in the local panel.
- Chunk 512 is retained only as a memory fallback: it costs an extra 240.8M package FLOPs and needs 2.408 ms residual-time saving to break even.

### Translation reuse

A 16-case panel spanning four seeds and correction scales 1e-4–3e-3 found worst RMS output-mean difference **3.729e-08** versus complete replay. This is a numerical identity check, not a statistical anchor validation.

## What is closed or deliberately not promoted

- **Persistent 64-block Winograd:** archived tracked FLOPs fell to 168.403B, but residual time rose to 0.435s and effective compute worsened to 211.901B. Do not revive unchanged.
- **Chunking every layer:** not promoted. It would perturb reduction order at every depth and can accumulate gate drift. The frozen candidate chunks only the final layer.
- **New mixed precision:** not promoted. The existing float32 propagation and float64 final reduction are preserved.
- **Another compiler:** explicitly out of scope. No rows, gates, or suffixes are removed.

## Frozen next gate

Run the packaged baseline and `submission_path6_final_chunk2048.tar.gz` through the same final subprocess on the official paired suite. Record exact FlopScope, residual wall time, peak memory, per-network raw MSE, adjusted score, and tails. No chunk-size tuning after opening the cohort.

Promotion rule:

1. raw MSE parity within the observed numerical drift;
2. measured effective compute below the production baseline;
3. no network-level tail regression;
4. package/API smoke tests pass.

If 2,048 rows fails only due memory, test the already frozen 512-row fallback. Otherwise stop.

## Artifacts

- `package/submission_path6_final_chunk2048.tar.gz` — runnable candidate submission.
- `package/translation_reuse.py` — exact cached-preactivation correction primitive.
- `package/k32_features.py` — no-extra-propagation K32/K128 reductions.
- `PATH6_RESULTS.json` — consolidated machine-readable results.
- `chunked_flop_counts.json` — exact shape-level operation counts matched to the archived partial-tree FlopScope breakdown.
- Raw benchmark and accuracy JSONs remain in the workspace.

## Limitations

The archive does not provide the official data/dependency environment in this container. A full depth-32 local NumPy run was attempted but exceeded the execution cap; this does not affect the final-layer identity because the preceding 31 layers are byte-for-byte unchanged in the candidate. The official subprocess result is therefore required before any shipping claim.
