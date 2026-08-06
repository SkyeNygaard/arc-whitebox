# Official execution runbook

## Unique submissions: seven

Reuse `A43.tar.gz` for T0.1/129, T0.2/clean, and T0.3/A43. Run, in one unchanged grader window:

1. `production_baseline.tar.gz`
2. `A42.tar.gz`
3. `A43.tar.gz`
4. `A43_delta64.tar.gz`
5. `A43_basis096.tar.gz`
6. `A43_basis064.tar.gz`
7. `A43_basis032.tar.gz`

Record per network: raw MSE, adjusted score, tracked FLOPs, residual wall time, effective compute, total wall, failures, median, p90, worst, and package hash. Do not tune or replace an arm after seeing any result. The 16/20-basis packages are local-only floor diagnostics and are not authorized submissions.

## Immediate calculations

- T0.2 residual delta: `(C_delta - C_clean - 2_147_483_648) / 1e11`.
- T0.3 A43 passes compute if `residual_A43 - residual_prod < 0.00524123904 s`, subject to raw/tail parity.
- T0.1 required control gain at each partial count: `(adjusted_partial / adjusted_129)` before adding control cost; then include exact control cost using T0.2's canonical exchange model.
