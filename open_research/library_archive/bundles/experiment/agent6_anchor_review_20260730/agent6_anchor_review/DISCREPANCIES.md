# Discrepancies and missing evidence

## D1. M146 is not reproducible from the shared archive

Canonical v15 reports 60 networks, exact all-layer means, a `41.2x` exact-anchor gain, and the perturbation curve, but its primary source is a user-provided local-model transcript. The ledger itself marks the artifact and exact IDs/perturbation manifest pending.

Missing minimum package:

- script and exact commit/hash;
- network IDs, base-network IDs, rotations, seeds, and split role;
- exact `all_layer_means` arrays and their construction;
- protected design means and final outputs;
- perturbation vectors, distribution, normalization, and random seeds;
- whether perturbations were isotropic, defect-aligned, per-coordinate, or downstream-whitened;
- exact nonlinear replay outputs and linearized outputs;
- gate-crossing counts/remainders;
- reference streams, noise correction, ordinary versus noise-corrected metric;
- row-level wins, tails, confidence intervals, and manifest.

Status must remain exploratory/provisional until this package is restored.

## D2. `5e-4` is described as break-even although the reported point still gains

At `5e-4`, the reported gain is `1.32x`, not break-even. A quadratic fit to all headline points gives break-even `5.80e-4`. “Approximately `5e-4`” is acceptable as a rough decision threshold, but the paper should not present it as the measured zero crossing without raw interpolation or a preregistered grid.

## D3. Oracle magnitude differs across summaries

The proof memo says a layer-31 oracle removes roughly 78% of noise-corrected MSE, while M146's `41.2x` gain implies a remaining fraction of only `2.43%`, or `97.57%` removed. These may be different interventions, cohorts, reference corrections, or metrics. They must not be merged without an experiment-ID/metric reconciliation.

## D4. Perturbation-direction scope is absent

The M146 threshold cannot currently be assigned to isotropic, actual-residual, analytic-residual, companion-residual, downstream-singular, or kink-concentrated directions. The synthetic audit proves these distinctions can be decisive.

## D5. Linear versus nonlinear replay is not separated

The archived theorem controls a frozen local linearization plus a ReLU remainder. M146's summary does not expose the linear surrogate, exact replay, crossing mass, or remainder. A claimed general threshold needs all four.

## D6. “Anchor task is not easier” is too broad

The layer31/layer32 relative-error ratio `1.012` is empirical and normalization-dependent. It does not prove that every selected downstream projection, sparse coordinate set, or nonlinear anchor functional is equally difficult. The correct conclusion is that the tested full-vector quantity did not become more accurate at layer 31.
