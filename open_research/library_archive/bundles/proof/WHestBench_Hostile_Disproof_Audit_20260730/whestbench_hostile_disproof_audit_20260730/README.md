# WHestBench hostile disproof audit package

Start with:

- `HOSTILE_DISPROOF_AUDIT.md` — full verdict;
- `CLAIM_SURVIVAL_MATRIX.md` — theorem-by-theorem status;
- `REQUIRED_PATCHES.md` — exact release fixes;
- `HOSTILE_AUDIT_CERTIFICATE.json` — machine-checked attack summary.

Counterexamples and corrected statements:

- `T29_CORRECTED_THEOREM.md` / `T29_ATTACK_RESULTS.json`;
- `T38_CORRECTED_THEOREM.md` / `T38_ATTACK_RESULTS.json`;
- `INFORMATION_REPLICATION_ATTACK_RESULTS.json`;
- `MISC_ATTACK_RESULTS.json`;
- `T16_T22_ATTACK_RESULTS.json`.

A patched copy of the broad proof compendium is included as
`WHESTBENCH_COMPLETE_PROOF_PACKAGE_HOSTILE_PATCHED.md`.

Run:

```bash
python verify_hostile_audit.py
```
