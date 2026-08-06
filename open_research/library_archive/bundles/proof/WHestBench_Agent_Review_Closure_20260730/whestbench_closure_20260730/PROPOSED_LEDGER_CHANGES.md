# Proposed canonical-ledger changes

## Add T30 — full T16 auxiliary-LP optimum

- **Evidence level:** Computer-assisted certified analytic proof.
- **Family:** Delsarte auxiliary LP / harmonic tail.
- **Claim:** The degree-five Hermite interpolant at the three roots of `22102t^3+21930t^2-87t-85` is the unique optimizer of the all-degree admissible auxiliary LP.
- **Result:** `K_32^(6)>0`; Hermite minorant feasible; all five nonconstant coefficients positive; exact primal-dual equality; all reduced costs above degree five strictly negative.
- **Numerical result:** auxiliary optimum MSE in the certified interval stored in `T16_PRIMAL_DUAL_CERTIFICATE.json`; Kerdock relative-excess upper `0.023324172950039%`.
- **Scope:** finite or absolutely convergent admissible nonnegative Gegenbauer expansions; limiting `d=256`, depth-32 kernel.
- **Primary source:** `T16_FULL_LP_THEOREM.md`, `prove_t16_primal_dual.py`, both T16 certificates, C++ audit.

## Amend T16

Replace `all-degree reduced-cost negativity; primal link open` by `reduced-cost component of T30; independently audited by Python exact integers and C++ Boost cpp_int`.

## Amend T22/T23 release row

- canonical archive becomes v5.1;
- 32 canonical files, 23 tracked generated chunks, 59 total manifest entries;
- wording `fixed during verification`, not internally authenticated immutable;
- environment pin and CI added;
- v4 archive/JSON quarantined by SHA-256;
- classify current coordinator verification as fast theorem+manifest pass and partial regeneration; retain Agent 2 full-clean-room result separately.

## Amend theorem percentage

Retain `0.0233655%` when citing the original rational safe T22 minorant. Add the stronger T30 auxiliary-optimum comparison `0.023324172950039%`, clearly identifying the different certificate.

## Amend M146

- **Status:** PROVISIONAL / NON-EVIDENCE.
- Remove verified `41.2x` and universal `~5e-4` language.
- Retain only arithmetic consistency and the requirement for a downstream-weighted exact-replay reproduction.

## Amend M152

- **Status:** REMOVED / UNVERIFIED.
- Do not include its 1,100-network, feature-correlation, `R^2`, or ratio numbers in paper evidence.
- Keep Agent 7 Path-2 reproduction as a separate new frozen empirical row.

## Add governance rows

1. Stale v4 theorem artifact quarantine with exact hashes.
2. Master claim register frozen after agent reconciliation.
3. Public-accounting claims remain pending a dedicated traceability audit.
4. AI-assistance/human-signoff requirements before publication.
