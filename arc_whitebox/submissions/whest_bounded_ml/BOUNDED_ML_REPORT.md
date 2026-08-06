# Bounded ML covariance-closure project

## Verdict

The bounded ML project found a **real, highly localized signal**, but it has not yet produced a validated challenge submission.

The useful target is the residual in the non-Gaussian bivariate ReLU covariance map. Nearly all predictive signal in the experiments came from two objects derived from the transported third cumulant:

- `x1`: the symmetric normalized `(2,1)` slice;
- `x1a`: its antisymmetric counterpart, coupled to `a_i-a_j`.

Marginal skew, kurtosis, and the more expensive fiber features added little after `x1/x1a`. Marginal-only models failed.

## Minimal learned closure

For each pair `(i,j)`, the model predicts

```text
Delta_ij = c_s(layer, a_i+a_j, a_i*a_j, |a_i-a_j|, rho_ij) * x1_ij
         + c_a(layer, a_i+a_j, a_i*a_j, |a_i-a_j|, rho_ij)
           * (a_i-a_j) * x1a_ij
```

The coefficient functions are a small two-hidden-layer SiLU MLP. The construction is exchange invariant under `i <-> j`.

## Main completed experiments

### 1. Feature ablation

On held-out width-256 synthetic networks:

| Features | Next-variance gain vs Gaussian covariance map |
|---|---:|
| Marginals only | 0.81x (worse) |
| `x1` only | 1.71x |
| `x1a` only | 1.47x |
| `x1 + x1a` | **3.58x** |
| Full 16-feature joint model | about 3.7x |

Linear and quadratic ridge models failed. The relationship is nonlinear.

### 2. Direct full-width model

A 10,082-parameter model trained and tested at width 256 reduced one-step propagated variance error from **3.25% to 0.856%**, a **3.80x gain**, on every held-out case in that run.

A separate model-size sweep on a fresh width-256 split was more conservative:

| Hidden width | Parameters | Gaussian RMS | Model RMS | Gain | Approx inference FLOPs across 31 layers |
|---:|---:|---:|---:|---:|---:|
| 8 | 138 | 3.074% | 2.274% | 1.35x | 0.26B |
| 16 | 402 | 3.074% | 2.533% | 1.21x | 0.75B |
| 24 | 794 | 3.074% | 2.112% | 1.46x | 1.51B |
| 32 | 1,314 | 3.074% | 1.641% | 1.87x | 2.53B |
| 48 | 2,738 | 3.074% | 1.860% | 1.65x | 5.36B |
| **64** | **4,674** | **3.074%** | **1.396%** | **2.20x** | **9.20B** |
| 96 | 10,082 | 3.074% | 1.317% | 2.33x | 20.01B |

The 64-unit model is the current recommended accuracy/compute compromise.

### 3. Width transfer

A flexible tree model trained at width 64 transferred to width 256 when it received the joint `x1/x1a` features, giving a 3.29x gain. A compact neural coefficient model trained at width 64 did **not** transfer. The compact model must be trained at the actual challenge width.

### 4. Recursive stability diagnostic

With oracle current marginal moments and oracle `k21` features, but recursively propagated off-diagonal covariance, the 96-unit model reduced final propagated variance error from **31.6% to 3.63%**, an **8.68x gain across four networks**.

This establishes that the learned closure is not inherently unstable. It is not a deployable score: the absolute rollout error remains too high and the diagnostic supplies oracle features.

### 5. Deployability warning

When the ordinary pair model was restricted to features estimated from a finite pilot sample, it improved Gaussian propagation but did **not** beat raw sampled covariance:

- model: 3.75% next-variance error;
- raw sample covariance: 3.56%.

The project therefore depends on obtaining `x1/x1a` through factorized K3 propagation, not from an ordinary 6,000-sample pilot.

## Real-data pipeline implemented

The package contains:

- `download_higher_moments.py` — selective downloader for the 1,000 per-MLP files;
- `train_higher_moments_x1.py` — MLP-level train/validation/test splits on the real moment files;
- `eval_higher_moments_x1.py` — evaluates a portable model on real files;
- `train_x1_coefnet_10k.py` and `train_x1_closure_10k.py` — streaming/memmap trainers for the 10,000-network joint-feature corpus;
- `kprop_x1_adapter.py` and `kprop_coefnet_adapter.py` — extract `get_dslice((2,1))` from ARC's factorized K3 state and apply the learned correction;
- `coefnet_numpy_runtime.py` — pure NumPy inference;
- portable 32/64/96-unit `.npz` models.

The real-file trainer and evaluator were smoke-tested end to end on generated NPZ files with the same schema. The pretrained synthetic model reduced pair-residual MSE 9.28x in that schema smoke test. This is only a software validation, not real-corpus evidence.

## Recommended real run

```bash
# Roughly 67 GB for all 1,000 files.
python download_higher_moments.py 0-999 --output ./higher_moments

python train_higher_moments_x1.py \
  --data-dir ./higher_moments \
  --out-dir ./real_x1_model \
  --train-files 700 --valid-files 150 --test-files 150 \
  --pairs-per-layer 128 --hidden 64 --epochs 80

python eval_higher_moments_x1.py \
  ./real_x1_model/higher_moments_x1_coefnet.npz \
  ./higher_moments/mlp_00850.npz ./higher_moments/mlp_00851.npz
```

The split is randomized and recorded in the results JSON. MLPs, rather than individual pairs, are the split unit.

## Go/no-go gates

Continue only if all four pass:

1. **Real held-out pair closure:** at least about 2x reduction in normalized covariance-residual MSE on held-out MLPs.
2. **Real next-layer contraction:** at least about 1.5x reduction in propagated variance error.
3. **Free rollout:** stable improvement through all 32 layers without oracle moments.
4. **Actual score:** factorized K3 + closure inference beats the 2.39e-7 Kerdock baseline under pinned FlopScope accounting.

Failing gate 1 or 2 should end the project. Passing them justifies implementing the complete estimator and measuring gates 3 and 4.

## Important limitations

- No real 67 GB/245 GB corpus was available inside this execution environment, and a direct single-file download could not be completed through the available network path.
- All headline accuracy measurements here are on fresh synthetic networks matching the published architecture and initialization.
- The portable synthetic checkpoints are demonstrations, not submission assets.
- The closure inference cost excludes factorized K3 propagation. ARC's implementation avoids materializing the full K3 tensor, but its exact FlopScope cost must be measured in the submission environment.
- One-step pair accuracy is not enough; earlier experiments showed that excellent pairwise R2 can still worsen downstream quadratic forms. The full rollout and score gates are mandatory.
