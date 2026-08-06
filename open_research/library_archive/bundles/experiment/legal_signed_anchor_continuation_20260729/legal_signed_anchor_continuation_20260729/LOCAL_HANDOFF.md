# Local handoff

Run all unit tests:

```bash
cd continue_path1
pytest -q 01_legal_signed_anchor/tests 02_centered_analytic_closures/tests 12_direct_output_affine_cv/tests
```

Key frozen summaries:

- `04_weak_anchor_ensemble/results/final.json`
- `11_pilot_phased_q128_source/results/final.json`
- `13_direct_pilot_phased_q128/results/final.json`
- `13_direct_pilot_phased_q128/results/rescue_final.json`
- `13_direct_pilot_phased_q128/results/large_holdout_summary.json`
- `14_fold_stable_q128_source/results/train/`

The source directories are self-contained compatibility harnesses. They use architecture-matched synthetic weights and do not claim official package reproduction or FlopScope billing.
