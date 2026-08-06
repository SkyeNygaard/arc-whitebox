# Experiment specification

Date: 2026-07-29

## Frozen question

Does an independently estimated selected pair-moment defect improve the lower radial-Hermite control enough to justify a deployable implementation, relative to directly reusing selected pair contractions from the primary Kerdock cloud?

## Cohorts

1. Frozen M109/M110 high-reference mechanism evidence (24 networks), used only as prior oracle context.
2. Exposed high-reference pair substitution (8 networks): exact center fixed; exact versus primary-cloud pair moments.
3. Exposed prospective companion validation block (24 networks): primary rotation 3, companion rotation 97, two independent 524,288-node final references. The original frozen arm used 32 probes. This experiment adds a post-hoc 128-probe extension and pair-source swaps; it is development evidence, not immutable validation.

## Arms

At 32 and 128 probes, companion basis counts 16 and 129, compare:

- companion diagonal + companion row moments;
- primary diagonal + primary row moments;
- companion diagonal + primary row moments;
- primary diagonal + companion row moments.

The companion mean is held fixed within each comparison. Shrinkages are 0.05, 0.10, 0.20, and 0.50. Final-output MSE, wins, tails, pair-increment output norm, and correction cosine are authoritative.

## Stop rule

Close independent pair estimation if primary-pair substitution is output-equivalent to independent pairs and the pair increment is negligible compared with center-estimation error. Retain only the direct fused accumulator if its exact incremental arithmetic fits the budget.
