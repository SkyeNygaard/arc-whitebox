# Agent 6 layer-31 anchor review package

## Primary deliverables

- `AGENT6_FINAL_REPORT.md` — integrated verdict.
- `PAPER_APPENDIX_LAYER31.md` — paper-ready theorem appendix.
- `PROVED_VS_EMPIRICAL_DIAGRAM.md` — proved/model/empirical dependency diagram.
- `CLAIMS_CHECKED.md`, `ASSUMPTIONS.md`, `CHECKS.md`, `COUNTEREXAMPLES.md`, `DISCREPANCIES.md`, `DECISION.md` — coordinator-required review files.
- `PROPOSED_LEDGER_CHANGES.md` — proposed rows and status changes; canonical ledger was not mutated.

## Reproducible computation

```bash
python agent6_theorem_checks.py --relu-cases 5000000
python agent6_adversarial_anchor_audit.py \
  --output . --networks 60 --relu-networks 30 \
  --dimension 256 --particles 4096
sha256sum -c MANIFEST.sha256
```

The adversarial audit is synthetic and deliberately anisotropic. It is a counterexample to universal threshold wording, not a reproduction or estimate of the ARC M146 operator.

## M146 reproduction status

Blocked. The Library contains the ledger summary but not the original row-level arrays, network IDs, perturbation manifest, scripts, reference streams, or replay outputs. No artifact in this package should be labeled an M146 replication.
