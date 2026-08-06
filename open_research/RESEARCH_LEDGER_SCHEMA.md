# Research ledger schema

A future machine-readable ledger should use one row per claim-changing event.

Recommended fields:

```yaml
id: stable string
date: ISO-8601
family: estimator/proof family
kind: theorem | experiment | audit | action
evidence_level: controlled vocabulary
claim: exact current statement
scope: assumptions and exclusions
result: symbolic/numeric result
cost_status: measured | profiled | projected | omitted
cohort: development/holdout/protected/theorem-only
oracle_use: none | capacity-only | illegal-for-deployment
artifact_paths: list
verdict: pass | fail | close | defer | supersede
reopens_if: explicit condition
supersedes: list of IDs
protected_data_opened: boolean
```

The workbook remains canonical for this release; this schema is a migration target, not a claim that every historical row has already been normalized.
