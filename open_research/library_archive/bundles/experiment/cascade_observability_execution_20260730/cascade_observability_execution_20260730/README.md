# Cascade / Observability Execution Package

This package completes and adversarially audits the supplied 2026-07-30 blueprint.

## Headline

The proposed observability-gap impossibility theorem is invalid as formulated because the actual final output is post-ReLU and therefore is not a Gaussian random element even in the wide-network GP limit. The package preserves the valid all-width fixed-linear symmetry theorem and completes all supported numerical tests.

## Main files

- `FINAL_REPORT.md` — complete verdict and evidence.
- `CLAIM_STATUS_MATRIX.csv` — claim-by-claim canonical status.
- `REVISED_BLUEPRINT.md` — valid replacement research program.
- `PROPOSED_LEDGER_CHANGES.md` — canonical ledger edits.
- `EXECUTION_SUMMARY.json` — machine-readable headline numbers.
- `code/` — runnable audits.
- `results/` — JSON/CSV outputs.
- `figures/` — generated figures.
- `sources/` — retained source artifacts required for reproduction.
- `MANIFEST.sha256` — package hashes.

## Reproduce

```bash
chmod +x REPRODUCE.sh
./REPRODUCE.sh
```

The scripts require Python, NumPy, SciPy, pandas, and scikit-learn. The transfer probe also uses PyTorch CPU.
