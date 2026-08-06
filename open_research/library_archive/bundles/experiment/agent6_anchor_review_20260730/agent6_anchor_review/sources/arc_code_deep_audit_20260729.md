# ARC White-Box `arc_code.zip` Deep Audit

**Audit date:** 2026-07-29  
**Archive:** `arc_code.zip` (8,884,512 bytes)  
**Extracted files:** 655

## Executive conclusion

The archive contains two intertwined repositories:

- `arc_whitebox`: production Kerdock/Winograd estimator, historical experiments, and factorized-K3 submission scaffolding.
- `arc_ceiling`: late-stage oracle/control-variate research, connected-cubic experiments, adjoint compression, centering audits, and result JSONs.

The production estimator is compact and intelligible. It uses a 66,048-point antipodal Kerdock/maximal-real-MUB angular cubature, integrates the Gaussian radius analytically by positive homogeneity, evaluates the first layer with a batched Walsh–Hadamard transform, and propagates the remaining 31 layers with a tracked depth-5 Strassen–Winograd multiplication.

The archive does **not** contain all data needed to reproduce the research results. It excludes official parquet data, large high-precision moment corpora, weight caches, and several frozen sparse-probe arrays. It also predates or omits several agent continuation reports uploaded at approximately the same time. It should therefore be treated as a code-and-results snapshot, not the canonical complete experimental state.

## 1. Archive layout

### `arc_whitebox`

- `submissions/kerdock_mub5_winograd_tree/`: current self-contained production submission.
- `scripts/`: official-data evaluation harnesses, sampling and arithmetic experiments.
- `src/`: Gaussian/ReLU math and reusable research components.
- `submissions/whest_bounded_ml/`: learned covariance/K3 closure work and result files.
- `submissions/whest_submission_ready_stage/`: factorized-K3 calibration and deployment scaffolding.

### `arc_ceiling`

- Exact and cross-fitted residual controls.
- Connected `c21`/K3 feature experiments.
- Full, checkpointed, and adjoint-compressed anchor calculations.
- Arbitrary-center/recentering identities.
- Hundreds of result JSONs for selection and holdout blocks.

## 2. Production estimator: exact execution path

File: `arc_whitebox/submissions/kerdock_mub5_winograd_tree/estimator.py`

1. Load `chirps[128,256]` and `rotation[256,256]` from `kerdock_mub5_seed3.npz`.
2. Compute the exact mean radius of a 256-dimensional standard Gaussian.
3. Rotate the first weight matrix.
4. Generate 128 Walsh–Hadamard chirp bases, all antipodes, plus the coordinate basis and its antipodes:
   - `128 * 256 * 2 = 65,536`
   - coordinate rows: `256 * 2 = 512`
   - total: `66,048`.
5. Apply first-layer ReLU.
6. For each of the remaining 31 layers, apply depth-5 Winograd multiplication and ReLU.
7. Accumulate the final mean in float64.
8. Return the exact analytic first-layer mean, zeros for unscored intermediate layers, and the estimated final mean.

The estimator ignores the dynamic budget and runs one fixed protected design.

## 3. Why Kerdock works here

For a bias-free ReLU MLP, the network is positively homogeneous:

`f(r u) = r f(u)` for `r >= 0`.

For `X = R U`, with `U` uniform on the sphere and `R` independent chi radius:

`E[f(X)] = E[R] E[f(U)]`.

The problem therefore becomes angular integration. The 129 real mutually unbiased bases form an antipodal spherical 5-design, so low-degree angular structure is cancelled exactly. The network-specific error that remains is a higher-degree deterministic cubature phase.

## 4. Arithmetic implementation

File: `fast_matmul.py`

- Three outer Winograd encodes retain seven-product indices as tensor axes.
- The deepest two levels are evaluated as an output-quadrant tree.
- Sixteen leaves are assembled once.
- Three outer decodes reconstruct the result.

This removes repeated large `block` assemblies while keeping all arithmetic inside tracked `flopscope.numpy` operations.

Independent checks performed during this audit:

- All 199 Python files compile successfully.
- The packaged chirps are exactly `+/-1` with shape `(128,256)`.
- Tested MUB cross-basis absolute inner products are exactly `1/16`.
- The rotation is orthogonal to about `8.65e-7` spectral error in stored float precision.
- A NumPy shim of the Winograd routine matched ordinary multiplication to roughly `6e-6` to `9e-6` relative error in float32 test cases.

## 5. Reproducibility limitations

The archive contains no top-level `pyproject.toml` or requirements lock. The research code expects combinations of:

- NumPy, SciPy, PyArrow, Torch, scikit-learn;
- `flopscope`, `whestbench`, `mlp_kprop`;
- official parquet data and downloaded weight files;
- high-precision moment `.npz` files absent from the archive.

`compileall` passes, but `arc_ceiling/test_conventions.py` cannot run in the audit environment because PyArrow and official data are absent. Some result metadata contains absolute local paths. One script contains a hardcoded `/Users/skyenygaard/...` path.

The self-contained production submission tarball is much cleaner: it contains only the estimator, fast multiplication, Kerdock asset, and manifest.

## 6. Important code-level findings beyond the ledger

### Completed factorized-K3 test

The archive contains `submission_readiness/test15.json`:

- upstream K3 closure MSE: `9.03065e-5`;
- selected hybrid MSE: `6.08095e-5`;
- gain versus upstream K3: `1.4851x`;
- improved networks: `10/15`;
- worst per-network gain: `0.72765x` (candidate is worse on the worst network).

This is a real deployable-shaped closure result, but its absolute MSE remains hundreds of times above Kerdock. It is useful as a surrogate/feature source, not a standalone challenge estimator.

### Exact adjoint compression is implemented

`adjoint_k3_full_holdout8.json` shows that rank-32 checkpoint handoff at layer 24 reproduces the factorized anchor with `2.31%` relative error and almost unit cosine at about `10.16B` analytic FLOPs, with a separately measured FlopScope profile around `10.34B`.

The backward contraction problem is therefore substantially solved. The unresolved problem is generating accurate independent forward source defects and lower-order centering terms.

### Current source closures fail

- Born/Gaussian source approximations remain about `18%` away from the factorized anchor and about `38%` away from the oracle anchor.
- Frozen checkpoint sample-only variants at layers 20 and 24 produced candidate/base MSE ratios around `1.71` and `1.69`, with severe tails.
- Direct factorized-anchor insertion in the cross-fitted control was catastrophic; a 10% delta was approximately neutral but did not validate as a useful correction.

### Oracle-state cubic controls are strong

On 16 networks, the cross-fitted rank-4 `c21` control with oracle state achieved candidate/base ratio `0.5993`, 16/16 wins, and worst ratio `0.9835`. A sample-direction variant with oracle anchor reached `0.5832`.

This isolates the blocker: the pointwise features and coefficient regression can work. The legal absolute anchor is missing.

### M85 factorized anchor signal is real but tail-risky

On the frozen 168–175 block:

- unshrunk factorized K3: ratio `0.9588`, 5/8 wins, worst `1.8312`;
- frozen 70% blend: ratio `0.8814`, 6/8 wins, worst `1.4277`.

It is real transferable signal but not safe and not score-positive with full K3 propagation.

## 7. Reconciliation with updates outside the archive

The archive is not the final state. Relevant later/sibling reports add:

1. A fresh layer-31 partial-MUB pilot screen: point estimate `1.353x` raw gain, but only 4/8 wins, CI crossing no gain, worst network `1.586x` worse, and compute erodes the gain. Sampled pilot variants should be closed.
2. Exact sparse 128-probe adjoint algebra and common-basis cost derivation: 128 probes do not require 128 dense adjoints; a suffix-local implementation can plausibly fit under 10–14B, but the real source-localization experiment has not been run because frozen arrays/corpus are missing.
3. Sparse-cubic `0.190x` claim is not a verified ARC holdout result; the shared artifacts lack the frozen implementation and protected output file.
4. Activation-region continuation tested many phase predictors and larger learned models; all were neutral or negative. The bottleneck is the signed high-degree Kerdock phase.
5. Reduced-width equivariant weight models achieved at best `1.0114x` raw gain with an unstable `6.12x` worst tail. Only one bounded width-256 edge-DWS experiment remains defensible.
6. Short-suffix micro-cubature remains a proposed bounded experiment with synthetic evidence only.
7. Compiler work has progressed beyond the archive. The leading shipping comparison is now fixed three-layer shrinkage versus adaptive depth 2–6, both with a minimum-saving fallback. Boundary certification is a debugging oracle, not a runtime feature.

## 8. Current state of affairs

### Submission/shipping

- Existing Kerdock/partial-tree package: only concrete ready package.
- Best next action: official paired Mini-100 run comparing assembly-free baseline, frozen two-layer compiler, fixed three-layer compiler, and adaptive 2–6 compiler.
- The fixed three-layer configuration is currently the simplest frozen candidate: 2,064-row pilot, rare threshold 8, middle correction 0, final shrink 0.875, and full-suffix fallback above predicted cost ratio 0.995.

### Primary research

Run one frozen shared-basis adjoint source-localization experiment using the actual 128 probes and exact recentering. Stop unless suffix depth <=12 retains >=90% of exact sparse-control gain below 10–14B and has safe tails.

### Bounded speculative research

Independent externally centered short-suffix micro-cubature for only the sparse anchor or 8 layer-31 coordinates. No broader distribution model.

### Low priority / closed

- more partial-MUB or Haar pilots for layer-31 coefficients;
- activation-region conditional integration;
- ordinary basis/Walsh phase regression;
- network-specific suffix-kernel coreset in its tested representation;
- reduced-width node-message-passing weight models;
- full-tensor K3 deployment;
- sample-only checkpoint source replacement;
- same-cloud Edgeworth anchoring.

## 9. Bottom line

The program has solved:

- the best fixed angular design;
- a strong arithmetic implementation;
- the dominant oracle repair channel;
- the low-dimensional terminal mapping;
- the tensor-free backward contraction algebra.

It has **not** solved:

- legal, accurate, cheap estimation of the absolute signed late-layer center/source defect.

That is the single central research problem. The immediate competition action is compiler integration and official accounting; the only high-upside research action is the frozen adjoint source-localization/centering experiment.
