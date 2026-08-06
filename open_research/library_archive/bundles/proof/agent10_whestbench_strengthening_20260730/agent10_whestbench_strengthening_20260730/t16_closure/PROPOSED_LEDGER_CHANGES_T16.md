# Proposed canonical-ledger changes — T16 primal–dual closure

## T16 status

**Old:** Reduced-cost tail proved; exact all-degree LP optimality conditional on primal attainment/complementarity.

**Proposed:** **PROVED UNDER AN EXPLICIT INTERVAL-ARITHMETIC TRUST BASE; independent hostile audit pending.**

## Result text

> Let `t1<t2<t3` be the roots of `22102 t^3 + 21930 t^2 - 87 t - 85`, and let `h_*` be the degree-5 Hermite interpolant matching `K32` and `K32'` at those nodes. Interval enclosure proves all nonconstant normalized-Gegenbauer coefficients of `h_*` are positive. A Bell-polynomial argument, exact rational Bernstein certificate, and four-box interval recurrence prove `K32^(6)>0` on `(-1,1)`. The Hermite remainder therefore gives `h_*<=K32`. Moment matching and exact contact give primal–dual equality. Together with the prior strict reduced-cost certificate for every degree `>=6`, this proves that `h_*` is the unique optimizer of the unrestricted all-degree auxiliary LP.

## Scope

This proves auxiliary-LP optimality only. It does not prove exact Kerdock cubature optimality, finite-width optimality, arbitrary signed-node optimality, or adaptive-estimator impossibility.

## Sources

- `T16_PRIMAL_DUAL_CLOSURE.md`
- `close_t16_primal_dual.py`
- `T16_PRIMAL_DUAL_CLOSURE_CERTIFICATE.json`
- prior `prove_t16_all_degree.py`
- prior `T16_ALL_DEGREE_CERTIFICATE.json`

## Publication gate

Require one independent interval-stack reproduction before upgrading the manuscript wording from “new certificate” to unqualified “computer-assisted theorem.”
