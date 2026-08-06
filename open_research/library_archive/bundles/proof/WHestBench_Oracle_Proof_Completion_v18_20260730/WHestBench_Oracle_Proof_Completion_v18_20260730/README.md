# WHestBench Oracle proof completion v18

This package continues the canonical v17 Oracle/correction program without opening a protected cohort.

## Main results

- `T42_POSITIVE_DEFINITE_AUXILIARY_RESIDUAL.md`
- `T43_ARBITRARY_SIGNED_NODE_RANK_FLOOR.md`
- `T44_PHASE_INFORMATION_BOUNDS.md`
- `T45_SYMMETRY_DEFECT_ALIGNMENT_BOUND.md`
- `T46_GAUGE_INVARIANT_COEFFICIENT_OBSTRUCTION.md`
- `T47_WEIGHTED_HARMONIC_RANK_FLOOR.md`

## Audits and continuation

- `ORACLE_CONTINUATION_HOSTILE_AUDIT.md`
- `CROSS_LAYER_COHERENCE_CORRECTED_INTERPRETATION.md`
- `M153_OBSERVATION_MAP_AUDIT.md`
- `PREREG_ORIENTATION_ODD_PHASE_OBSERVABLE.md`
- `UPDATED_ORACLE_RESEARCH_SEQUENCE.md`
- `PROPOSED_LEDGER_CHANGES.md`

## Verification

Run:

```bash
python code/verify_oracle_proof_completion.py
```

Expected top-level status: `PASS`.

Run the stronger weighted signed-floor verifier separately:

```bash
python code/verify_weighted_rank_floor.py
```

The verifier checks:

1. order-47 directed ReLU-kernel Taylor propagation;
2. exact monomial-to-Gegenbauer projection;
3. T42 coefficient margins;
4. all active degree ratios for T43, including odd degrees;
5. exhaustive subsets of degrees 0 through 3;
6. binary phase-information inequality;
7. T45 identity and exact zero case;
8. T46 sign-group and two-point minimax checks;
9. pooled Oracle-coherence statistics and a counterexample to the within-network inference.

The T47 verifier checks all active degrees 1 through 30 for the frozen exact-rational weighting and certifies the 1.979504x improvement ceiling.

## Trust labels

- T44–T46 abstract proofs are conventional exact arguments.
- T42, the numerical T43 specialization, and the numerical T47 specialization are computer-assisted theorem candidates. They need an independent interval stack and named human review before publication.
- OracleContinuation conclusions are authenticated-array analyses, not fully self-contained network regeneration.
- No claim in this package is a universal impossibility theorem for adaptive or nonlinear white-box estimation.
