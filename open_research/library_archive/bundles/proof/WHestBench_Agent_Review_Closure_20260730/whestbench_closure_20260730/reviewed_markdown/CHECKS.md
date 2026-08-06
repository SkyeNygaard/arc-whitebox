# Checks — Agent 4 / T16

## Source recovery

Recovered and inspected:

- `arc_cubature_experiments_v3.zip`;
- `dual_reduced_costs.py`;
- `results/tight_bounds_all_dims.json`;
- `results/dual_reduced_costs.json`;
- `results/reduced_costs_d256_to_1m_tail.npy`;
- `arc_cubature_proof_v5.zip` and its exact rational degree-5 witness.

The V3 implementation explicitly sets `q_l=-1/N` for `l>=1` and computes `r=q-G@lambda`.

## Exact algebra checks

The proof script independently:

1. Generates normalized Gegenbauer polynomials through degree 5 over `Fraction`.
2. Recovers the monomial moments from the six exact `q_l` values.
3. Solves the exact Hankel system for the monic orthogonal cubic.
4. Verifies all three orthogonality equations exactly.
5. Verifies exact sign changes of the integer cubic in three disjoint rational intervals.
6. Encloses all three Lagrange weights with exact rational interval arithmetic and proves positivity.

## Exact finite sweep

The quotient-ring recurrence checked `14,653` degrees (`6..14,658`) using only integers.

- Every reduced-cost numerator was strictly negative.
- Rational cross multiplication identified degree 7 as the maximum.
- Exact maximum: `-2327215/9290262647272`.
- Runtime in the archived run: approximately 3 seconds.
- Final recurrence denominator size: about 395,000 bits.

## Tail check

The cutoff inequality is checked by one exact big-integer comparison. Under the conservative root bound `|t|<0.993`, `14659` is the smallest integer for which this particular tail inequality succeeds.

At the cutoff:

- analytic upper bound: `1.5119414031766884e-5`;
- `1/N`: `1.5140503875968992e-5`;
- strict margin: `2.108984420210826e-8`.

## Agreement with recovered V3 outputs

The recovered V3 floating result reports degree 7 as the least-negative mode near `-2.5050045258e-7`. The exact theorem gives `-2.5050045282e-7`; the difference is ordinary binary64/witness rounding and does not affect the sign.

The archived one-million-degree NumPy array has shape `(1,000,001)` and approaches `-1/N`, consistent with the theorem.
