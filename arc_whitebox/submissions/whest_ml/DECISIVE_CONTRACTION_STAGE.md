# Decisive contraction stage

## What has already passed

The 100-network pair closure is robust:

- sampled-pair test gain: 3.34x;
- all-pairs gain: 3.36x;
- every held-out network-layer improved;
- independent split gains: 3.19x and 3.23x.

The unresolved question is whether those pairwise improvements survive the
actual next-weight quadratic form.

## Inputs required

From the completed pilot run:

- `runs/pilot100/higher_moments_x1_results.json`
- `runs/pilot100/higher_moments_x1_coefnet.npz`
- `data/higher/mlp_XXXXX.npz` for validation and test IDs

Official weights are separate from the higher-moment files. Extract them into:

- `data/official_weights/mlp_XXXXX.npy`

## Download only the required weights

```bash
pip install pyarrow huggingface_hub

python download_official_weights.py \
  --results-json runs/pilot100/higher_moments_x1_results.json \
  --splits valid,test \
  --output data/official_weights
```

Because this pilot uses global indices 0--99, the downloader stops after the
first few public Parquet shards rather than downloading the entire full split.

## Run

```bash
./run_decisive_contraction.sh \
  runs/pilot100/higher_moments_x1_results.json \
  runs/pilot100/higher_moments_x1_coefnet.npz \
  data/higher \
  data/official_weights \
  runs/pilot100/next_variance_eval.json
```

## What the evaluator does

For each held-out network and source layer it computes:

```text
C_base  = Gaussian bivariate ReLU covariance
C_model = C_base + alpha * learned residual
v_next  = diag(W_next^T C W_next)
```

`alpha` is fit only on validation MLPs, then frozen for the test MLPs. The
report also evaluates alpha=1 to detect whether shrinkage is hiding model
instability.

Two diagonal modes are reported:

- `oracle`: both methods get the true post-ReLU diagonal, isolating the learned
  off-diagonal closure.
- `gaussian`: the Gaussian marginal diagonal is retained, measuring the broader
  baseline without an oracle diagonal.

The evaluator verifies the weight orientation by contracting the true post
covariance and comparing it against the stored next pre-activation variance.
A mismatch over roughly 0.5% means the weights or layer convention are wrong.

## Decision thresholds

Proceed to full factorized-K3 rollout only when the held-out `oracle` result has:

- relative-variance gain >= 1.5x;
- improvement in >= 80% of network-layer cases;
- positive results at layers 24, 28, and 30;
- no negative predicted variances;
- no severe covariance indefiniteness requiring aggressive repair.

Stop this project when:

- gain < 1.3x; or
- fewer than 70% of cases improve; or
- validation-fitted alpha is near zero; or
- alpha=1 catastrophically fails across late layers.

A positive oracle-diagonal result is necessary but not sufficient. The next
stage must integrate the closure into factorized K3 and run a self-consistent
32-layer rollout with deployable marginal moments.
