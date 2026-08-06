# Nested Partial-MUB Convergence as a Free Uncertainty Signal

**Date:** 2026-07-29  
**Status:** **CLOSE as a standalone safety certificate.**  
**Protected/official holdout:** not opened.

## Executive result

A fresh exact-geometry width-256 experiment used **18 new base networks × 3 predetermined rotations = 54 records**, with all rotations grouped by base network. The frozen 112+17 construction accumulated `c2,c4,c8,c12,c17` in one companion pass. Final targets were uniformly refined to two independent **196,608-sample** aggregates per base network.

At selection time, the development-selected nested rule failed calibration at **40.0% catch / 71.4% false suppression**, so no rule was selected. After uniform reference refinement, the same calibration block is even worse: **16.7% catch / 100.0% false suppression**. On validation, the same frozen rule had catch **63.6%** and false suppression **50.0%**. It does not meet the required 75% / 20% gate.

The central negative finding is mechanistic: severe cases often have internally smooth nested convergence. Validation median `cos(c12,c17)` was **0.873** overall and **0.888** on severe records. The trajectory can converge coherently to the wrong externally phased answer.

## Final-output candidates on validation

| Candidate | Pooled ratio | Unbiased pooled | Wins | Median | p90 | Worst | Mean correction cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 1.1535 | 1.2003 | 3/15 | 1.2175 | 1.5191 | 1.9550 | 0.000 |
| c8 | 1.3997 | 1.5216 | 3/15 | 1.5160 | 2.2318 | 3.1646 | 0.220 |
| c12 | 1.4249 | 1.5545 | 3/15 | 1.3104 | 2.0530 | 2.3122 | 0.132 |
| c17 | 1.3664 | 1.4781 | 4/15 | 1.3771 | 1.7508 | 2.3096 | 0.093 |
| count_weighted_8_12_17 | 1.3009 | 1.3927 | 4/15 | 1.3598 | 1.8317 | 2.2523 | 0.150 |
| bounded_aitken | 1.4652 | 1.6072 | 3/15 | 1.6130 | 1.9132 | 2.3075 | 0.010 |
| bounded_richardson | 1.6067 | 1.7918 | 2/15 | 1.4243 | 1.9918 | 2.4998 | 0.018 |
| trimmed_basis_mean | 1.3659 | 1.4776 | 3/15 | 1.3560 | 1.8058 | 2.3797 | 0.090 |
| median_of_groups | 1.4656 | 1.6077 | 3/15 | 1.4018 | 2.0017 | 2.4702 | 0.086 |
| paired2_substitute | 1.1874 | 1.2446 | 5/15 | 1.1170 | 1.5987 | 2.1282 | 0.004 |
| rule_A | 1.1461 | 1.1907 | 5/15 | 1.2302 | 1.5705 | 1.9550 | 0.087 |
| rule_D_paired | 1.4442 | 1.5798 | 2/15 | 1.3918 | 1.8821 | 2.3096 | 0.011 |
| oracle_benefit_gate | 1.0403 | 1.0526 | 7/15 | 1.1508 | 1.3629 | 1.5776 | 0.158 |
| oracle_nested_chooser | 0.9180 | 0.8930 | 7/15 | 1.0653 | 1.2283 | 1.4313 | 0.346 |

The fixed `c17` package itself scored **1.3664** by mean-target MSE and **1.4781** by the independent-half unbiased estimator on this synthetic validation block; the no-correction reduced 112-basis arm scored **1.1535**. This cohort is diagnostic, not a replacement for the canonical partial-MUB validation.

## Priority questions resolved

1. **Do catastrophic tails show nonconvergence?** Not reliably. Smooth late convergence is common in both good and bad records; severe records remain after `c12` and `c17` are highly aligned.
2. **Does direction stabilize earlier than amplitude?** Direction relative to the terminal estimate stabilizes earlier, but terminal direction itself is not an absolute-truth certificate. Median validation `||c12||/||c17||` is **1.216**.
3. **Are later bases signal or variance?** `c8`, `c12`, and `c17` show gradual average convergence, but later bases do not monotonically improve every rotation. Robust trimming, median groups, Aitken, and Richardson do not remove the tail.
4. **Does basis ordering matter?** It changes early-prefix smoothness but not `c17`. The target-free geometry-maximin order is reported, but no ordering produces a transferable severe-tail certificate.
5. **Are patterns stable across rotations?** No. **60.0%** of validation base networks have mixed severe/nonsevere outcomes across their three rotations.
6. **Does per-probe instability find harmful modes?** The strongest validation per-probe concentration AUC was 0.795, but its development behavior did not transfer into a calibrated rule.
7. **Can nested estimates replace a paired probe?** No. Nested-only rules fail. The already-free two-basis original/external difference is weak as a certificate too; its frozen disagreement rule catches only **0.0%** of severe validation records.

## Compute and implementation

All nested prefixes are reductions of the same 17 propagated blocks. They add **zero trajectories**. The candidate diagnostics require only prefix sums, 128-probe reductions, and leave-one-basis recombinations. The sandbox did not provide official FlopScope, so this report does not claim official tracked FLOPs. Mean per-record diagnostic harness time was 4.33s and peak process RSS was 1413.1 MiB; these include research materialization and are not subprocess-package timings.

## Decision

Close nested convergence as a standalone free uncertainty signal. Retain the prefix arrays and per-basis influences as low-cost features for Priority 1 paired safety or a later tiny residual gate, but do not suppress, shrink, extrapolate, or choose prefixes based on nested convergence alone. The failure mode is coherent external angular bias, not merely finite-prefix variance.

## Provenance

- Development: base networks 5000–5008.
- Calibration: 5009–5012.
- Untouched-until-final validation: 5013–5017.
- Three predetermined rotations per base; all grouped by base network.
- Rule thresholds frozen before calibration; calibration selected no rule; validation did not change the decision.
- Full row-level vectors, records, freeze hashes, reference seeds, candidate tables, ordering diagnostics, and code are included.
