# Checks performed

## Archive and provenance

- Materialized and hashed the canonical v15 ledger, proof memo, T4 closure report, activation-region report, and deep archive audit.
- Searched the Library for `5.55e-4`, `41.2x`, `60 networks`, `all_layer_means`, `1.012`, `layer31`, `anchor`, `perturb`, and `M146`.
- Found no original M146 script, row table, network/rotation manifest, perturbation arrays, exact means, replay outputs, or reference streams.
- Confirmed that canonical v15 itself marks M146 “artifact pending” and its IDs/perturbation manifest pending.

## Formal checks

- Re-derived the correction-risk identity.
- Derived positive-only and bounded conditional selector formulas.
- Derived the general replacement formula when anchor noise leaves the correction subspace.
- Derived optimal shrinkage for correlated in-subspace noise.
- Derived a nonlinear sufficient margin using the exact ReLU remainder.
- Verified the common-bias two-point lower bound.

## Executed numerical/property checks

- Correction-risk identity on 20,000 random 37-dimensional samples: maximum error `8.53e-14`.
- Conditional selector formulas with groups having positive and negative signed transfer: direct and formula gains agree to floating-point precision.
- Correlated-noise shrinkage: formula `0.53921995`, grid optimum `0.539`.
- General replacement formula: exact to displayed precision.
- Common-bias observational equivalence: maximum difference `8.88e-16`.
- ReLU crossing lemma: 5,000,006 random/edge cases; maximum numerical violation `1.37e-14`.

## M146 headline-curve consistency check

From the reported gains `(41.2, 1.32, 0.34, 0.086)` at perturbations `(0, 5e-4, 1e-3, 2e-3)`, convert gain to candidate/base risk and fit

`risk_ratio(epsilon) = q + a epsilon^2`.

Results:

- `q = 1/41.2 = 0.02427184466`;
- `a = 2.9019646e6`;
- `R² = 0.99999647`;
- fitted break-even `epsilon = 5.798536e-4`.

The three nonzero points have nearly identical pointwise quadratic coefficients. This is strong internal consistency with zero-mean fixed-shape perturbation noise. It does not identify the perturbation distribution or reproduce a single network.

## Synthetic structured-direction audit

Sixty synthetic 256-dimensional linearized replays were generated with downstream condition numbers from `31.2` to `275.2`. At equal Euclidean anchor error `5e-4`:

- actual-defect direction break-even: exactly `5.55e-4` by construction;
- isotropic median break-even: `5.59e-4`;
- leading downstream singular direction: median `1.87e-4`;
- defect-sensitivity-gradient direction: median `2.25e-4`;
- trailing singular direction: median `1.41e-2`.

This is a counterexample to a universal Euclidean threshold, not an estimate of the ARC operator spectrum.

## Synthetic ReLU audit

Thirty synthetic networks, 4,096 particles each, 256 outputs, three directions, two gate-margin regimes, and five shift sizes were evaluated by exact ReLU replay.

At shift `5e-4`:

- generic actual-defect direction: median remainder/linear shift `2.94e-5`;
- kink-enriched actual-defect direction: `6.13e-3`;
- kink-enriched kink-focused direction: `2.51e-1`.

Thus a frozen linear threshold can transfer well in generic-margin geometry and fail materially in a direction concentrated near ReLU kinks.
