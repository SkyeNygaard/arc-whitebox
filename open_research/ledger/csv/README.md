# Ledger CSV exports

These CSV files are loss-minimizing exports of selected sheets from the canonical v31 workbook. They preserve historical rows, including claims that were later corrected, superseded, quarantined, or narrowed.

They are provided for search, version comparison, and machine analysis. They are **not** a flat list of current endorsed claims. Use the status, evidence-tier, contradiction, quarantine, and supersession columns together, and consult:

- `../../RELEASE_STATUS.md`;
- `../../audit/AUDIT_REPORT.md`;
- `../WHestBench_Public_Claim_Manifest.csv`;
- Paper A's explicit claim boundary.

The canonical workbook remains the authoritative historical ledger. The audited public theorem constants are the ones replayed by `../../scripts/run_core_verification.py`.
