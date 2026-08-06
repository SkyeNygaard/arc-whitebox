# Agent 10 continuation — closing open ends and strengthening the WHestBench paper

**Date:** 2026-07-30  
**Final disposition:** Two closeable proof/release gaps were closed in new artifacts; two empirical gaps remain blocked by missing raw data.

## Executive result

### Closed: T16 primal attainment and complementarity

The previous T16 package proved strict negative dual reduced costs for every degree at least six but left exact primal attainment as a separate obligation.

I constructed the missing primal:

- take the exact three algebraic dual nodes;
- define the degree-5 Hermite interpolant `h_*` matching `K32` and `K32'` at all three nodes;
- interval-enclose its normalized-Gegenbauer coefficients and certify all five constrained coefficients positive;
- prove `K32^(6)>0` on `(-1,1)` using a Bell-polynomial decomposition, exact rational Bernstein inequalities, and four interval boxes for the outer derivative ratio;
- apply the Hermite remainder formula to prove `h_*<=K32`;
- use exact moment matching and contact to obtain primal–dual equality.

This proves that `h_*` is the unique optimizer of the unrestricted all-degree auxiliary LP. It closes higher-degree improvement of the **certificate**, not the remaining gap between the certificate and the true cubature optimum.

The new proof uses an explicit `mpmath.iv` trust base and should receive an independent hostile audit or be ported to the existing directed-Decimal stack before final publication.

### Closed: T22 stale one-sided artifact

I created a canonical machine-readable theorem artifact that correctly represents actual Kerdock suboptimality as one-sided:

- additive excess lower endpoint `0`;
- multiplicative ratio lower endpoint `1`;
- relative excess lower endpoint `0`.

A validator rejects the stale two-sided object and accepts the canonical object. The release patch also incorporates the Agent-2 audit corrections: 32 tracked canonical files, 23 regenerated intermediate chunks, fixed-during-verification manifest wording, external archive digest, pinned environment, and no “formally verified” language.

### Strengthened: paper architecture and wording

The integrated patch now presents:

1. T22 as the certified static boundary;
2. T16 as exact all-degree optimality of the auxiliary certificate;
3. T27 as a restricted signed/support theorem;
4. correction theory and experiments as a scoped falsification map.

The revised abstract, theorem table, section order, limitations, reopening conditions, and forbidden-claim list are in `PAPER_STRENGTHENING_PATCH_20260730.md`.

### Still blocked: M146

The reported perturbation curve is arithmetically consistent, but the original 60-network row package, IDs, directions, seeds, and replay manifest remain missing. The numerical threshold must remain provisional. No further honest closure is possible without restoring those artifacts.

### Still blocked: M152

The claimed 1,100-network corpus, target equation, features, grouping, script, predictions, and hashes remain absent. M152 must remain outside the evidence-bearing paper. The independently reproduced Path-2 audit is the legitimate replacement.

## Validation performed

- Re-ran the prior `prove_t16_all_degree.py` successfully.
- Ran the new T16 closure certificate twice; the generated certificate file was byte-stable.
- Ran an independent high-precision/non-interval T16 sanity check; status `PASS`.
- Compiled every new Python script.
- Confirmed the T22 validator rejects the stale artifact.
- Confirmed the T22 validator accepts the canonical one-sided artifact.

## New artifacts

### T16

- `T16_PRIMAL_DUAL_CLOSURE.md`
- `close_t16_primal_dual.py`
- `T16_PRIMAL_DUAL_CLOSURE_CERTIFICATE.json`
- `independent_t16_closure_check.py`
- `T16_PRIMAL_DUAL_INDEPENDENT_CHECK.json`
- `PROPOSED_LEDGER_CHANGES_T16.md`

### T22/T23

- `FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32_CANONICAL.json`
- `build_t22_canonical_release.py`
- `validate_t22_one_sided.py`
- `T22_STALE_VALIDATION.json`
- `T22_CANONICAL_VALIDATION.json`
- `T22_RELEASE_PATCH.md`

### Manuscript

- `PAPER_STRENGTHENING_PATCH_20260730.md`

## Recommended next review order

1. Assign a hostile mathematical/interval reviewer to T16 only.
2. Port the four outer-ratio interval boxes to the existing directed-Decimal framework.
3. Replace the stale T22 JSON in the release and externally anchor the archive digest.
4. Apply the manuscript patch.
5. Run a final hostile referee after all claim-status labels are frozen.
