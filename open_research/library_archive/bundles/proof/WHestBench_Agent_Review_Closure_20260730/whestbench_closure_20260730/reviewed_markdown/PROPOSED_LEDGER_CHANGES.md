# Proposed canonical-ledger changes

## T16 row

**Old status:** Open / pending; reduced costs nonpositive through degree 1,000,000; analytic tail pending.

**Proposed status:** Theorem complete for the reduced-cost claim.

**Proposed result text:**

> The exact three-node dual moment measure has contacts equal to the roots of `22102 t^3 + 21930 t^2 - 87 t - 85`. Exact integer quotient-ring recurrence proves every reduced cost negative for degrees 6–14,658. A normalized-Gegenbauer Laplace bound with alpha 127 proves negativity for all degrees at least 14,659. The least-negative mode is exactly degree 7 with `r_7=-2327215/9290262647272`.

**Interpretation:**

> The all-degree reduced-cost tail is closed analytically. Exact LP-optimality wording still requires the named primal minorant's exact feasibility and primal-dual attainment to be linked to this measure.

**Source:**

- `prove_t16_all_degree.py`
- `T16_ALL_DEGREE_CERTIFICATE.json`
- Agent 4 report package and SHA256 manifest

## Current State / Paper Claims Matrix

Replace “all-degree proof remains open” with:

> All unused reduced costs are now proved strictly negative. The only remaining qualification for full all-degree LP-optimality is exact primal-attainment/complementarity for the selected degree-5 minorant.
