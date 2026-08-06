# Library archive import

This directory was assembled from the user's persistent ChatGPT Library on 2026-08-03.

It supplements the audited public repository with the source materials that were spread across Library folders.

## Contents

- `ledger/`: canonical v31 research ledger.
- `reports/`: current-state reports, final subagent reports, and reconciliation material.
- `bundles/proof/`: extracted proof and theorem-development archives.
- `bundles/experiment/`: extracted experiment, negative-result, and continuation archives.
- `manifests/source_archives.csv`: hashes and extraction status for every imported ZIP.
- `manifests/archive_members.csv`: every extracted member with CRC32 and SHA-256.
- `manifests/duplicate_files_by_sha256.csv`: exact duplicate content retained under multiple historical bundle names.
- `manifests/release_assets_manifest.csv`: large original archives supplied separately.
- `manifests/excluded_large_library_files.csv`: raw split archives omitted because they add about 355 MB and are unsuitable for ordinary Git history.

## Evidence warning

This is a historical research archive. It contains superseded, failed, exploratory, and quarantined work as well as canonical results. The current claim boundary is controlled by the audited papers, `audit/AUDIT_REPORT.md`, `RELEASE_STATUS.md`, and the canonical v31 ledger—not by the mere presence of a historical file.

## Recommended GitHub handling

Commit this extracted `library_archive/` directory normally. Do **not** commit the sibling `release-assets/` files to ordinary Git history. Upload those as GitHub Release assets or manage them with Git LFS.
