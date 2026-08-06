# WHestBench salvaged theorem package

This package replaces every theorem wording falsified by the 2026-07-30 hostile audit with a maximal correct theorem and useful application-specific corollaries.

Start with:

- `SALVAGE_REPORT.md`
- `VALID_CLAIMS_MATRIX.md`
- `PROPOSED_LEDGER_PATCH_V18.md`

The theorem files are:

- `T29_MAXIMAL_CORRECT_AND_K32_UNIQUENESS.md`
- `T38_MINIMAL_CONDITION_AND_DEGENERATE_BOUNDARY.md`
- `CONDITIONAL_HAAR_EXACT_AND_APPROXIMATE.md`
- `REPLICATION_BIAS_COVARIANCE_COST_THEOREM.md`
- `RELU_REMAINDER_SHARP_THEOREM.md`
- `KERNEL_PERTURBATION_OPTIMIZER_TRANSFER.md`
- `K32_MUB_LINE_SPECTRUM_CERTIFICATE.json`
- `T16_ENDPOINT_EQUALITY_PATCH.md`
- `OBSERVABILITY_METRIC_REPLACEMENT.md`

Run:

```bash
python verify_salvaged_theorems.py
python certify_k32_mub_line_spectrum.py
python certify_t16_endpoints.py
python verify_package.py
```

The verifier performs symbolic and representative numerical checks and writes `SALVAGED_THEOREMS_VERIFICATION.json`.
