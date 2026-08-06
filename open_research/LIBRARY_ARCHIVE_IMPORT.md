# Library archive import

This repository includes a curated import of the WHestBench files stored in the author's persistent ChatGPT Library as of 2026-08-03.

The import is under [`library_archive/`](library_archive/README.md). It contains:

- the canonical v31 research ledger;
- current-state and final agent reports;
- extracted proof-development packages;
- extracted experiment and negative-result packages;
- per-archive and per-file SHA-256 manifests;
- an exact-duplicate report;
- a manifest for large archives supplied as release assets.

Historical files are preserved as historical evidence. Their presence does not promote superseded or exploratory claims to current status. Current public claims are controlled by the audited papers, [`audit/AUDIT_REPORT.md`](audit/AUDIT_REPORT.md), [`RELEASE_STATUS.md`](RELEASE_STATUS.md), and the canonical ledger.

Verify the import with:

```bash
python scripts/verify_library_import.py
```
