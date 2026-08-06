# Local handoff

## 1. Export one frozen label

The narrowest available contract is the scalar V80 scale/sign target. In the original V80 environment, save each network's full weights, baseline final vector, high-quality target final vector, and **ungated frozen correction direction**. Do not retune the H3 feature, selector, ridge, or 0.25 anchor.

```bash
python adapters/extract_v80_contract.py \
  --records-json inputs/v80_records.json \
  --output inputs/v80_unfrozen_export.npz
```

For a Prompt 4/6 `g31` label or Prompt 3 joint-anchor label, export the same generic schema directly. The target dimension must remain 1–16.

## 2. Verify the replay surrogate

For V80, the correction is already an additive final-output vector, so use an algebraic proof document based on `inputs/surrogate_verification_v80_template.json`. For `g31` or joint-anchor replay, run direct true-final-layer comparisons and provide a JSON document with:

```json
{
  "verified": true,
  "proof_kind": "direct_numerical_replay",
  "max_abs_error": 1e-7,
  "tolerance": 1e-6,
  "network_count": 8,
  "source_sha256": "..."
}
```

Freeze and hash the label bundle:

```bash
python adapters/build_label_bundle.py \
  --input inputs/v80_unfrozen_export.npz \
  --output inputs/frozen_labels.npz \
  --manifest inputs/frozen_label_manifest.json \
  --label-kind v80_scale \
  --surrogate-verification inputs/surrogate_verification_v80.json \
  --source-hash SHA256_OF_FROZEN_V80_EXPORTER \
  --source-hash SHA256_OF_CORRECTION_CONFIG
```

## 3. Install the canonical split registry

Copy the immutable registry to `inputs/canonical_split_registry.json`. It must have `status: "frozen"`, group all rotations from one base network, include all exposed/disallowed IDs, and provide `splits_sha256` using `src.contracts.canonical_hash`.

Do not generate a substitute split locally. Do not use global IDs 0–199 or the branch-specific exposed blocks.

## 4. Preflight

```bash
PYTHONPATH=. python -m src.preflight \
  --data inputs/frozen_labels.npz \
  --manifest inputs/frozen_label_manifest.json \
  --splits inputs/canonical_split_registry.json

PYTHONPATH=. pytest -q
```

Preflight must fail on any hash mismatch, unverified surrogate, exposed ID, split overlap, missing split, wrong width/depth, or broad label dimension.

## 5. Freeze cost before training

The supplied one-pass architecture has an analytical estimate of 13.329B FLOPs at `D=1`, 116,587 parameters, and projected candidate compute 191.465B when paired with the frozen V80 1.5021% anchor. Exact FlopScope and wall-time measurement supersede this estimate. If the chosen target requires a true final replay, update `replay_extra_compute_B`; stop before training when the 1.15× gate can no longer repay cost.

## 6. Train once

```bash
PYTHONPATH=. python -m src.train \
  --data inputs/frozen_labels.npz \
  --manifest inputs/frozen_label_manifest.json \
  --splits inputs/canonical_split_registry.json \
  --config frozen_config.json \
  --out results/run_001
```

Run on a CUDA machine. No architecture or ridge sweep is authorized beyond the frozen ridge control grid and calibration shrink already encoded.

## 7. Expected outputs

`results/run_001/results.json` reports:

- train/calibration/validation/test example and base-network counts;
- parameter and inference-cost estimates;
- anchor-only, constant-shrinkage, invariant-ridge, and edge-DWS replay metrics;
- grouped raw and adjusted gains with 95% base-network bootstrap intervals;
- correction cosine, confidence Brier/ECE, wins, median and worst ratio;
- exact gate booleans.

`results/run_001/model.pt` contains the frozen checkpoint and calibration residual shrink. A real run passes only when all four preregistered gate booleans are true. Otherwise mark this exact one-pass model class paused.
