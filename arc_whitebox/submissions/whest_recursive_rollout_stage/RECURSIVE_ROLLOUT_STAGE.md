# Recursive rollout stage

## Status entering this stage

The real held-out next-layer contraction benchmark passed:

- fitted alpha, oracle diagonal: 2.044x relative-variance MSE gain;
- fitted alpha, Gaussian diagonal: 1.862x;
- fitted-alpha sigma MSE gain: 2.028x oracle / 1.848x Gaussian;
- 98.3% of held-out cases improved;
- no negative next-layer variances;
- fitted alpha was approximately 0.65.

This is strong enough to justify a full recursive rollout. Pairwise residual MSE is no longer the relevant gate.

## What the rollout tests

`eval_recursive_rollout_x1.py` starts at the exact first-layer Gaussian state and recursively propagates:

1. pre-activation mean and covariance;
2. Gaussian ReLU mean/covariance;
3. the learned x1/x1a covariance residual;
4. the corrected moments through the next weight matrix.

It evaluates final pre-activation mean MSE—the challenge-relevant quantity—as well as variance and sigma drift.

The `oracle_x1` mode still uses the true normalized k21 slice from the higher-moment files. It therefore tests dynamic stability and usefulness of the closure, not the accuracy/cost of factorized K3 itself.

## Run

From the directory containing the rollout scripts:

```bash
ROOT="$(cd ../whest_bounded_ml && pwd)"

./run_recursive_rollout.sh \
  "$ROOT/runs/pilot100/higher_moments_x1_results.json" \
  "$ROOT/runs/pilot100/higher_moments_x1_coefnet.npz" \
  "$ROOT/data/higher" \
  "$ROOT/data/official_weights" \
  "$ROOT/runs/pilot100/recursive_rollout"
```

This performs two expensive evaluations:

1. `recursive_rollout_raw.json`: no PSD repair; eigenvalues are measured only.
2. `recursive_rollout_psdclip.json`: same validation-selected alpha with diagonal-preserving PSD repair.

The alpha search uses all 15 validation MLPs and the grid:

```text
0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0
```

It minimizes final-output mean MSE, not pairwise loss.

## Decision rule

Proceed to fully deployable factorized-K3 rollout only when the **raw** rollout has:

- final mean MSE gain >= 1.25x;
- at least 75% of held-out MLPs improving in final mean;
- final relative-variance MSE gain >= 1.25x;
- no severe late-layer divergence;
- only mild negative covariance eigenvalues.

Interpretation:

- Raw and PSD-clipped both pass: proceed directly to factorized K3.
- PSD-clipped passes but raw fails: closure is useful, but covariance consistency is the next blocker.
- Both fail: stop this ML branch. The one-step gains do not survive recursion.

## Factorized-K3 hook

`kprop_coefnet_torch_patch.py` is a Torch-native callback intended for ARC's public factorized K3 implementation. Insert it after the nonlinear K3 step and before the next linear contraction:

```python
K_pre = linear_kprop(...)
K_post = factored_nonlin_kprop_k3(K_pre, relu_wick_coef, ...)
patch.apply_(K_pre, K_post, layer=layer)
```

The hook:

- obtains `K_pre[3].get_dslice((2, 1))` without materializing the full K3 tensor;
- evaluates the compact coefficient network on-device;
- adds the learned residual to `K_post[2]`;
- optionally performs diagonal-preserving PSD repair;
- leaves K1 and factorized K3 unchanged.

Do not evaluate a challenge submission from oracle-x1 rollout numbers. A fully factorized-K3 run and pinned FlopScope measurement remain required.

## Numerical changes

The previous contraction run emitted overflow warnings from float32 NumPy matrix products. `coefnet_numpy_runtime_stable.py` now:

- performs inference in float64;
- clips only extreme normalized base features;
- replaces nonfinite out-of-distribution rows with zero correction;
- records how often clipping or replacement occurs.

Normal in-distribution predictions are unchanged up to floating-point precision.
