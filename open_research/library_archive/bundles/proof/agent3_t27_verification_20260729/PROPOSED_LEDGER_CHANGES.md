# Agent 3 — Proposed Canonical-Ledger Changes

## T27 result text

Replace the current result with:

> For arbitrary real weights summing to one on symmetrized antipodal lines inside the fixed 33,024-line Kerdock universe, the infinite-width depth-32 kernel risk reduces exactly to a constant plus `(O-C) sum_b S_b^2 + (A-O) sum_bi w_bi^2`. For every line budget `1≤P≤33,024`, the global optimum uses `floor(P/256)` complete bases plus at most one partial basis. Fixed-support optimal weights have positive basis totals and equal positive line weights; signed cancellations are strictly dominated.

## T27 verdict text

> PROVED UNDER AN EXPLICIT MODEL / VERIFIED AFTER WORDING CORRECTIONS. Scope: dimension 256, depth-32 infinite-width ReLU kernel, network-independent linear rules, symmetrized antipodal lines from the fixed Kerdock universe. Budget is antipodal lines (`2P` individual point evaluations), not arbitrary points.

## T27 source text

Add:

- `GLOBAL_SUPPORT_THEOREM.md`
- `agent3_t27_verification_20260729/CLAIMS_CHECKED.md`
- `agent3_t27_verification_20260729/verify_t27.py`
- `agent3_t27_verification_20260729/verification_results.json`
- `agent3_t27_verification_20260729/MANIFEST.sha256`

## Reproducibility note

> The archived 26,000-trial stress-test claim lacks a located original script/manifest. An independent clean verifier reproduced the kernel constants, exact risk decomposition, signed fixed-support optimum, convex allocation result, all boundary budgets, and a new random/exhaustive regression suite.
