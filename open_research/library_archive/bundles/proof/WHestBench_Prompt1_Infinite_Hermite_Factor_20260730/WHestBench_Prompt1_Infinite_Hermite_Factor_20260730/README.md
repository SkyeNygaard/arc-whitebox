# WHestBench Prompt 1 — infinite Hermite factor continuation

This bundle develops the apparent infinite limit behind the certified degree-123 weighted signed-floor theorem.

## Status

The metric and algebraic reduction are complete. The resulting `1.04753x` same-cost cap is **conditional** on one all-degree positivity lemma. The finite prefix and endpoint asymptotics are very strong, but a directed remainder bridge is still required.

## Main files

- `PROMPT1_INFINITE_HERMITE_FACTOR_REPORT.md` — mathematical report.
- `INFINITE_HERMITE_FACTOR_CHECK_SUMMARY.json` — compact status and constants.
- `INFINITE_HERMITE_FACTOR_CANDIDATE.json` — 100+ digit roots, Hermite polynomial and factor coefficients.
- `INFINITE_HERMITE_FACTOR_SERIES505.json` — high-precision series and Gegenbauer coefficients.
- `PROOF_GAP_AND_NEXT_CERTIFICATION.md` — exact remaining proof task.
- `factor_long_double.c` — independent long-prefix exploratory verifier.
- `FACTOR_LONG_DOUBLE_ORDER8191.txt`, `FACTOR_LONG_DOUBLE_ORDER16383.txt` — saved runs.
- `MPFR_KERNEL_JET_511.json` — independent kernel jet inherited from the degree-123 release.

## Reproduction notes

The Python scripts expect the existing WHestBench harmonic utilities or the local paths used by the Prompt-1 continuation environment. `factor_long_double.c` is standalone once its frozen roots and kernel recurrence are embedded.

The long-double calculation is discovery/cross-check evidence, not the publication interval engine.
