# WhestBench submission-readiness stage

This package advances the learned `x1/x1a` closure from an oracle-feature result
to a deployable ARC factorized-K3 test, while keeping the final 15 MLPs untouched
until one configuration has been selected.

## Improvements over the first factorized-K3 stage

1. **Feature calibration:** ARC's recursively predicted `k21` slice may differ in
   scale from the exact public-data slice used during training. Seven validation
   MLPs fit conservative global and layerwise scales.
2. **Generalized hybrid:** validation chooses how much of ARC's native K3 mean and
   covariance to retain:

   `mean = Gaussian mean + gamma * (K3 mean - Gaussian mean)`

   `cov = Gaussian cov + beta * (K3 cov - Gaussian cov) + alpha * ML residual`

3. **Honest selection:** 7 validation MLPs calibrate features, the remaining 8
   choose the hybrid, and all 15 test MLPs are evaluated once.
4. **Stable Gaussian integration:** a sine substitution removes the endpoint
   singularity in the bivariate normal CDF integral. The 16-node implementation
   matches the exact zero-mean ReLU kernel to machine precision through
   correlations of +/-0.999.
5. **Non-destructive guards:** pairwise Cauchy-Schwarz caps and a next-layer
   variance guard replace repeated PSD eigendecomposition, which previously
   destabilized rollout.
6. **Submission scaffolding:** a chunked float32 `flopscope.numpy` CoefNet,
   pickle-free `flops.Module` packer, cost proxy, and forbidden-import/package
   auditor are included.

## Run using an already-installed ARC repository

```bash
cd /Users/skyenygaard/Programming/AI-Safety/arc_whitebox/submissions

ROOT="$PWD/whest_bounded_ml"
STAGE="$PWD/whest_submission_ready_stage"
KPROP="/absolute/path/to/your/existing/mlp_cumulant_propagation"

"$STAGE/run_submission_readiness_stage.sh" \
  "$KPROP" \
  "$ROOT/runs/pilot100/higher_moments_x1_results.json" \
  "$ROOT/runs/pilot100/higher_moments_x1_coefnet.npz" \
  "$ROOT/data/higher" \
  "$ROOT/data/official_weights" \
  "$ROOT/runs/pilot100/submission_readiness" \
  cpu float64
```

The script first checks the exact installed ARC API. It then performs calibration,
a broad 3-MLP smoke search, an 8-MLP tuning run, and one final evaluation on the
15 untouched test MLPs.

### Main outputs

- `api_compat.json`
- `calibration_layerwise.json`
- `smoke_none.json`, `smoke_global.json`, `smoke_layerwise.json`
- `tuning8.json`
- `selected_plan.json`
- `test15.json`
- `test15_summary.txt`

## Decision rule

A submission port is justified when `test15.json` shows:

- gain versus upstream factorized K3 >= 1.25x;
- at least 75% of test MLPs improve;
- no nonfinite values;
- little or no activation of the next-variance safety guard.

A 1.5x gain with at least 90% improving would be a strong result.

## Float32 follow-up

Run float64 first to establish the algorithm. If it passes, rerun the selected
configuration in float32. The bundled 64-unit CoefNet's float32 output differed
from float64 by only `2.6e-7` relative RMS on 500,000 standardized feature rows,
while current dtype-aware accounting may make float32 materially cheaper.

## Submission scaffold

`submission_skeleton/` contains only the ML portion and integration support. The
remaining major engineering task after a passing real test is to port the required
ARC factorized-K3 operations from Torch to `flopscope.numpy`.

The 64-unit CoefNet proxy cost is about 11.8B FLOPs, 4.35% of the 272B budget,
before the K3 propagation cost. Pair inference is chunked at 4,096 pairs, far
below the remote-array size limit.

## Truly fresh 100-network audit

The original 15 test MLPs have already informed earlier research decisions. After
the selected factorized-K3 hybrid passes them, run a second audit on IDs 100-199,
which were not used to train the model or design the closure:

```bash
"$STAGE/run_fresh_100_audit.sh" \
  "$KPROP" \
  "$ROOT/runs/pilot100/higher_moments_x1_results.json" \
  "$ROOT/runs/pilot100/higher_moments_x1_coefnet.npz" \
  "$ROOT/runs/pilot100/submission_readiness" \
  "$ROOT/data/higher" \
  "$ROOT/data/official_weights" \
  "$ROOT/runs/pilot100/fresh100" \
  100-199
```

This downloads only the additional per-MLP files and necessary official Parquet
shards. Treat `fresh100_gate.json` as the strongest go/no-go evidence before doing
the full `flopscope.numpy` K3 port.
