# Agent 3 T27 Verification Package

This package independently verifies and adversarially scopes `GLOBAL_SUPPORT_THEOREM.md`.

## Main conclusion

`VERIFIED_AFTER_SPECIFIED_CORRECTIONS`

## Files

- `CLAIMS_CHECKED.md` — complete independent derivation.
- `ASSUMPTIONS.md` — exact theorem scope.
- `CHECKS.md` — numerical and exhaustive checks.
- `COUNTEREXAMPLES.md` — explicit examples blocking over-broad extrapolation.
- `DISCREPANCIES.md` — required wording and reproducibility corrections.
- `DECISION.md` — final claim classification and paper-ready theorem sentence.
- `PROPOSED_LEDGER_CHANGES.md` — canonical tracker patch text.
- `verify_t27.py` — independent executable verifier.
- `verification_results.json` — structured results.
- `run_output.json` — complete stdout from the verifier.
- `MANIFEST.sha256` — hashes.

Run:

```bash
python verify_t27.py
```
