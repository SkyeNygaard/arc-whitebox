# Local handoff

From an ARC White-Box repository environment containing the official Mini-100 data and challenge runtime:

```bash
cd integrated_compiler_measurement_20260729
python -m pytest -q tests
python src/run_paired.py \
  --data /absolute/path/to/official_phase1_mini/data \
  --asset assets/kerdock_mub5_seed3.npz \
  --indices $(seq 0 99) \
  --outdir results/official_paired
```

The driver launches each candidate in a separate subprocess with single-thread BLAS environment variables, stores baseline final means, computes candidate approximation errors against exactly that baseline, and aggregates raw MSE, tracked FLOPs, FlopScope residual wall time, effective compute, memory, internal packing/symbolic timings, fallbacks, wins, median/worst ratios, and a 10,000-resample network bootstrap interval.

Do not use `results/historical_evidence_reanalysis.json` as an official result. It exists only to preserve the previous frozen evidence in a clearly labeled form.

Before trusting the official output, verify that `flopscope.numpy` supports the functional operations used by `compiler_core.py`, especially data-dependent indexing and `argsort`. This sandbox could not execute that backend. Any incompatibility is an integration failure, not permission to switch to untracked NumPy.
