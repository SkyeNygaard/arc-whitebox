# Reproduction

This bundle contains the frozen degree-123 primal and dual certificates for Prompt 1.

Run the mathematical certificate checks against the frozen directed MPFR jet:

```bash
./run_checks.sh
```

Independently regenerate the direct-C MPFR order-511 jet and compare it byte-for-byte:

```bash
./regenerate_mpfr_jet.sh
```

Or run both:

```bash
./run_all_checks.sh
```

The checks rebuild the degree-123 primal, run the untruncated closed-projection cross-check, verify all 7,381 dual inequalities in six chunks, and aggregate the directed dual ceiling.

The optimizer is not part of the trust base. Frozen decimal strings are interpreted as exact rationals. The package remains computer-assisted and requires qualified human review before publication.
