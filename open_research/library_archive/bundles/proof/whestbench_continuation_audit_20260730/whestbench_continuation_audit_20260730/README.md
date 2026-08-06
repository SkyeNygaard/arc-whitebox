# WHestBench continuation audit package

This package contains the continuation report, updated claim register, manuscript errata, next research program, independent audit scripts, and captured verification outputs.

## Main files

- `CONTINUATION_AUDIT_REPORT.md`
- `UPDATED_CLAIM_REGISTER.csv`
- `MANUSCRIPT_ERRATA.md`
- `NEXT_RESEARCH_PROGRAM.md`
- `VERIFICATION_SUMMARY.json`
- `verification/`
- `code/`

## Reproduction scope

The audit scripts intentionally consume the source archives and row packages audited in the active runtime. They are not copies of those large upstream artifacts. `REPRODUCE.sh` therefore requires the source packages to be materialized at the paths declared inside the scripts. The exact source archive SHA-256 digests are recorded in `VERIFICATION_SUMMARY.json`.

The T22 direct-C GMP/MPFR replay and theorem verifier were executed from the separately retained dual-engine release. Their stdout, timing, and result JSONs are included under `verification/`.

The empirical scripts independently recompute metrics from the retained row-level NPZ/JSON/CSV outputs; they do not regenerate the expensive full-width network trajectories.
