# WHestBench Complete Proof Package — 2026-07-30

Run the frozen-artifact verifier from this directory:

```bash
python verify_complete_proof_package.py
```

The verifier checks the reopened-path SHA-256 manifest, recomputes terminal metrics from row-level arrays, recomputes the signed-probe oracle values, and cross-checks the exact spherical control means. It does not rerun the separate T22/T30 interval proof engines.

Primary reading order:

1. `WHESTBENCH_COMPLETE_PROOF_PACKAGE_20260730.pdf`
2. `WHESTBENCH_COMPLETE_PROOF_PACKAGE_20260730.md`
3. `WHESTBENCH_PROOF_CERTIFICATE_20260730.json`
4. `verify_complete_proof_package.py`

The package proves scoped geometric and information-class theorems. It does not claim universal impossibility for complete-white-box, finite-width, adaptive, nonlinear, or arbitrary signed-node estimators.
