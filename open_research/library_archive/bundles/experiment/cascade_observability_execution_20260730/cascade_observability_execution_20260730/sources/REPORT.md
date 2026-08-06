# Tree T0 unblock report — recovered full-width basis-count curve

**Date:** 2026-07-29  
**Status:** **T0.1 locally unblocked on a frozen full-width grouped test block; T0.2/T0.3 frozen for immediate official execution.**

## Executive decision

The missing public Mini-100 files were not recoverable inside this sandbox, but a preserved Prompt-7 frozen-input archive contained 82 complete width-256/depth-32 networks. Its final 32 examples form a previously frozen test block of 16 base networks with two grouped rotations each. Unlike the earlier development rows, these test rows store a direct 256-dimensional baseline-error vector against independent high-reference targets, so the frozen A43 basis-prefix deltas can be scored exactly up to the archived float16 weight representation.

The four cumulative outputs were independently compared with the frozen `A43_basis032/064/096/129` packages and were **bit-identical at every basis count**. The replay sign was separately verified by reproducing the archived Prompt-7 anchor-only test MSE (`3.439892586e-7` reconstructed versus `3.439892566e-7` reported).

The result is:

| Bases | Raw MSE | Raw candidate/base | Wins (base networks) | Adjusted candidate/base, row-scaled residual | Required raw gain with +5B control |
|---:|---:|---:|---:|---:|---:|
| 129 | 3.013100e-07 | 1.000 | reference | 0.9970 | — |
| 96 | 4.172349e-07 | 1.3847 | 0/16 | 1.0275 | 1.0670× |
| 64 | 6.772092e-07 | 2.2475 | 0/16 | 1.1123 | 1.1763× |
| 32 | 1.279858e-06 | 4.2476 | 0/16 | 1.0522 | 1.1732× |

### Interpretation

1. **129 remains the standalone operating point.** Every partial arm lost raw MSE on all 16 grouped base networks. The 96-base raw ratio was 1.385; 64 and 32 were 2.248 and 4.248.
2. **The adjusted frontier is much flatter than the raw frontier.** Under the preferred row-scaled residual model, 96 bases is only 2.75% worse; 64 is 11.23% worse; 32 is 5.22% worse.
3. **96 bases is effectively at parity in the tracked-only lower bound.** Its adjusted candidate/base ratio is 1.000554. Official residual-wall measurement decides whether it is merely close or genuinely neutral.
4. **Partial designs remain live only as control hosts.** With 5B added control compute, 96 requires a 1.067× raw gain; 64 requires 1.176×; 32 requires 1.173×. This directly prices T2/T3 continuation.
5. **The 32-base T2 gate is well chosen.** A 1.15× raw control is insufficient at 5B cost but sufficient if the complete incremental cost stays below roughly 4B under row-scaled residual economics.

## Statistical details

The test baseline raw MSE was `3.013099535e-07`. Grouped bootstrap intervals are over the 16 base-network IDs, retaining both rotations together.

| Bases | Raw gain baseline/candidate, 95% grouped CI | Adjusted gain, constant-residual 95% CI | Median base ratio | p90 | Worst |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.7222 [0.6591, 0.7786] | 0.9645 [0.8803, 1.0399] | 1.411 | 1.815 | 2.195 |
| 64 | 0.4449 [0.3803, 0.5156] | 0.8757 [0.7485, 1.0148] | 2.218 | 3.734 | 4.256 |
| 32 | 0.2354 [0.1929, 0.2844] | 0.8805 [0.7213, 1.0636] | 4.276 | 7.355 | 8.699 |

Reference-noise correction makes every partial arm less favorable; the test block's mean reference-noise floor was `2.1885e-8`.

## T0.2 and T0.3 unblock

The exact seven-arm execution is now packaged as one command for the user's existing Mac environment, which archived logs show already has WhestBench 0.13.0, FlopScope 0.9.1, and `official_phase1_mini`:

```bash
bash run_official_on_skyes_mac.sh /path/to/arc_whitebox /path/to/T0_official_grader_instrumentation_20260729
```

It runs, without tuning:

1. production baseline;
2. A42;
3. A43;
4. A43 delta64;
5. 96 bases;
6. 64 bases;
7. 32 bases.

`aggregate_official_results.py` computes the T0.2 residual-wall delta, the A43 promotion gate, and official basis-count ratios.

## Limits

- This is **not** an official Mini-100 result. The frozen test networks are full-width synthetic networks and were already opened once for Prompt 7.
- Test weights were archived in float16 and converted to float32. The baseline-error vectors are direct high-reference errors, but basis deltas inherit weight-quantization error. The deltas are large relative to baseline error (mean RMS ratios: 0.62 at 96, 1.11 at 64, 1.82 at 32), so quantization cannot plausibly explain the qualitative raw ordering; it could matter near 96-base adjusted parity.
- Development rows in the archive reconstruct only V80 replay parabolas and therefore were deliberately **not** reused for basis-delta scoring.
- Official residual-wall behavior remains hardware- and subprocess-dependent. The scenario table brackets zero, row-scaled, and constant partial residual time rather than pretending one is observed.

## Decision

**PASS ECONOMIC SCREEN / BLOCKED OFFICIAL.** Preserve 129 as the standalone baseline. Notify T2 and T3 that 96 bases is the cheapest control host: it needs approximately 1.028× raw gain with no added control compute, 1.067× at +5B, 1.106× at +10B, and 1.138× at +14B under row-scaled residual economics. At 32 bases, the corresponding hurdles are 1.052×, 1.173×, 1.294×, and 1.391×.

No additional local basis-count sweep is warranted. The next information-bearing act is the frozen official seven-arm run.
