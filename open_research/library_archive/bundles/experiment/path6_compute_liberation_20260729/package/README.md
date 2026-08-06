# WHestBench Path 6 package

This package contains the frozen Path 6 implementation experiment.

- `submission_path6_final_chunk2048.tar.gz`: production-shaped final-layer chunked direct-accumulation candidate.
- `PATH6_FINAL_REPORT.md`: algebra, cost, measurements, gates, and limitations.
- `results/PATH6_RESULTS.json`: consolidated machine-readable result.
- `translation_reuse.py`: exact cached-preactivation translation primitive.
- `k32_features.py`: K32/K128 reductions from already propagated rows.
- `audit/`: local benchmark, shape-level FLOP counter, and fail-closed official audit launcher.
- `whestbench_canonical_research_ledger_20260729_reconciled_v8_path6.xlsx`: updated canonical ledger copy.

The official FlopScope/WHestBench subprocess was not available in this container. Do not ship the candidate until the frozen paired official gate passes.
