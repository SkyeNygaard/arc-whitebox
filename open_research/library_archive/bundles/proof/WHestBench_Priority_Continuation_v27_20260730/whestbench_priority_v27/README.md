# WHestBench Priority Continuation v27

This package contains the non-overlapping continuation after Agent 4's sole-path audit and Agent 5's exact late-innovation analysis.

## Main claims

- exact terminal-innovation lower bound for checkpoint telescopes;
- all confirmation single-checkpoint partitions through layer 31 fail under favorable oracle covariance;
- dual-feasible certificates for the complete all-layer empirical SOCP on confirmation cases, including an independent 2,048-pair replication;
- explicit scope: independent-block linear unbiased checkpoint gauges only.

## Verify

```bash
python scripts/verify_priority_v27.py
sha256sum -c MANIFEST.sha256
```

The large authenticated OGAP archive and regenerated source-state files are not vendored. Their provenance is described in `SOURCE_PROVENANCE.json`.
