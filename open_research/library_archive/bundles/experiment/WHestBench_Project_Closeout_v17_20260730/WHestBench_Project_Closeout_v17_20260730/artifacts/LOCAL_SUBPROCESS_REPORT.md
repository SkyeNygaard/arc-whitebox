# Frozen Seven-Arm Local Subprocess Report

**Status:** local synthetic contract complete; true official grade externally blocked.

The exact seven frozen packages were executed as separate processes on the bundle’s deterministic seed-51000 synthetic contract. The official `whestbench` package, FlopScope runtime, official cohort and parquet data were not available, so these measurements must not be represented as an official score or deployment-cost comparison.

| Arm | Algorithm time (s) | Process wall (s) | Peak RSS (MiB) | Output relation |
|---|---:|---:|---:|---|
| production_baseline | 64.248 | 64.99 | 2115.4 | production reference |
| A42 | 25.671 | 26.28 | 532.8 | A42/A43-identical full output |
| A43 | 27.153 | 27.79 | 532.6 | A42/A43-identical full output |
| A43_delta64 | 25.346 | 25.97 | 533.1 | A42/A43-identical full output |
| A43_basis096 | 18.832 | 19.47 | 442.9 | partial-basis output |
| A43_basis064 | 12.813 | 13.38 | 386.5 | partial-basis output |
| A43_basis032 | 5.965 | 6.55 | 331.1 | partial-basis output |

## Numerical parity

- A42 and A43 were bit-identical: **True**.
- A42/A43 versus production RMS drift: `1.54805472474e-07`.
- Relative RMS drift: `2.55482701043e-07`.

## Decision

Retain production as the externally verified package. A42 was the fastest full-width arm in this single synthetic local run and A42/A43 were bit-identical, but official promotion remains blocked on the missing grader, FlopScope, cohort and raw-MSE/effective-cost comparison.

The partial-basis runtimes confirm the expected cost ordering but do not alter the prior statistical conclusion that 129 bases is the standalone design.
