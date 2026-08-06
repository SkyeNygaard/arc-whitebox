# Tree T4 — Legal Layer-31 Anchor Hedge: Exhaustive Closure Report

**Date:** 2026-07-29  
**Overall disposition:** **CLOSE the specified T4 hedge.**  
**Protected cohorts:** Calibration IDs 6016–6023 and validation IDs 6024–6031 were never generated or opened.

## Executive conclusion

Both allowed T4 roots fail their hard gates.

1. **T4.1 partial-MUB plus paired certificate fails before calibration.** On 16 fresh base networks × literal rotations 3/11/97, the best of 2,250 frozen absolute/paired/zero policies catches only **57.14%** of severe absolute failures while falsely suppressing **35.00%** of nonsevere cases. The complete estimator scores **1.127854 raw**, **1.145174 noise-corrected**, **17/48 wins**, and **2.480711 worst**. Its grouped raw interval is **[1.004602, 1.305179]**.
2. **T4.2 selected center contractions is information- and cost-limited.** The strongest legal analytic family is **0.9386845**. The only measured under-14B independent arm optimizes to **0.940310** (or **0.948789** with worst ≤1.15). The first statistically stronger arms require 21.4B, and full companions cost about 165B.
3. **T4.3 is not permitted to open.** Neither parent passed an immutable gate.

The mechanistic diagnosis is sharper than “the gate is weak.” The corrections contain target-labeled *per-rotation* signal, but not a stable target-free absolute phase. A positive c17/p2/p4 oracle fit separately to each rotation reaches **0.915133**, whereas forcing one coefficient vector across a base network’s three rotations regresses to **1.019612**. Across the fresh cohort, **11/16** networks cross the 1.10 severe threshold across rotations and **12/16** contain both beneficial and harmful rotations.

## 1. Governance and protocol

The first generated stress panel inherited a network-offset rotation convention from the predecessor harness. Before any promotion decision, that mismatch was detected and quarantined as supplementary stress evidence. A second config was frozen before labels using the literal prescribed rotations **3, 11, and 97**.

The decisive development protocol was:

- architecture: width 256, depth 32, bias-free ReLU;
- development: network IDs 6000–6015;
- sealed calibration: 6016–6023;
- sealed validation: 6024–6031;
- grouping: all three rotations of a base network;
- references: six independently scrambled Gaussian Sobol streams, aggregated into two 196,608-sample halves;
- main design: bases 0–110 plus coordinate basis 128;
- external design: frozen `chirp17_r3`, 17 blocks, amplitude 0.20;
- paired diagnostics: fixed p2 and p4 from the first two/four original–external block differences;
- radial map: 128 probes;
- policy actions: absolute, bounded p4 paired, or zero;
- finite policy grid: 2,250 rules, with no post-hoc feature invention.

Every row stores independent target halves, c2/c4/c8/c12/c17, all prefix increments, every individual companion-basis anchor/correction, p2/p4 vectors, leave-one-basis corrections, P128 and a fixed P32-subset robustness diagnostic, final error geometry, seeds, and hashes.

## 2. T4.1 final-output results

| Arm | Raw ratio | Wins | Median | p90 | Worst |
|---|---:|---:|---:|---:|---:|
| Full 129-basis baseline | 1.000000 | — | 1.000000 | 1.000000 | 1.000000 |
| 112-basis zero correction | 1.133824 | 13/48 | 1.198636 | 1.614262 | 2.389523 |
| Frozen c17 absolute | 1.224619 | 16/48 | 1.169548 | 2.274341 | 3.011172 |
| p2 | 1.152007 | 13/48 | 1.175181 | 1.759454 | 2.330811 |
| p4, beta=.50 | 1.126984 | 11/48 | 1.159762 | 1.676696 | 2.184313 |
| Frozen finite policy | 1.127854 | 17/48 | 1.194809 | 1.876001 | 2.480711 |

The frozen policy applied c17 on 25 rows, p4 on one, and zero on 22. Mean error–correction inner product is **-2.535e-08**, mean correction norm squared **1.047e-07**, and mean correction cosine **0.0969**. Reference noise is about **11.93%** of pooled baseline MSE and does not reverse the conclusion.

### Detection gate

- severe absolute records: 28;
- nonsevere records: 20;
- required: recall ≥75%, false suppression ≤20%;
- observed grouped LOBO: recall **57.14%**, false suppression **35.00%**.

No individual feature has a feasible threshold. At false-positive rate ≤20%, the maximum severe recall is 39.3% for c17–p2 cosine, 28.6% for c17–p4 cosine, 46.4% for p4/c17 norm ratio, 39.3% for nested stability, 39.3% for leave-one-basis influence, and 42.9% for the P32/P128 subset norm ratio.

### Oracle ceilings and what they mean

| Oracle diagnostic | Raw ratio | Interpretation |
|---|---:|---|
| Best discrete action per rotation: c17 / p4 / zero | 0.976571 | Even target labels cannot make the prescribed action set hit 0.95. |
| Continuous c17 or p4 scale per rotation | 0.955584 | Slightly above 0.95 and tail-unsafe. |
| Continuous positive c17+p2+p4 per rotation | 0.915133 | There is inaccessible rotation-specific signal. |
| One positive coefficient vector per base network across all rotations | 1.019612 | The phase does not survive rotation grouping. |
| One global positive coefficient vector | 1.096981 | No universal mixture exists. |

The per-rotation oracle is not a continuation candidate: it uses the true target, wins only 26/48, and still has worst 2.062567. It is a mechanism diagnostic showing that the obstacle is signed phase identification rather than total absence of useful directions.

## 3. T4.2 selected center-contraction frontier

The exact lower-recentering interface is valid and minimal. For each of 128 probes the primary cloud supplies a diagonal pair scalar and row-direction pair scalar; the external estimator needs only one selected center coordinate and one directional center contraction. Independent pair moments are not required.

Two materially different external families were exhausted:

### Family A: internally centered analytic defect

The strongest transferable construction scored **0.9386845**, won **5/8**, and had worst **1.1019004**. It misses the <0.75 screen by a wide margin. Thirteen adjacent legal analytic/source/pilot constructions also failed; several favorable tuning results reversed on fresh cohorts because jointly biased estimators agreed in the wrong direction.

### Family B: independent companion / paired-difference centers

Exact global rescaling of the machine-readable 24-network geometry gives:

| Candidate | Best raw | Wins | Worst | Added compute |
|---|---:|---:|---:|---:|
| 8-basis first-layer GREG | 0.940310 | 13/24 | 1.266680 | 11.18B |
| 8-basis, tail bounded | 0.948789 | 14/24 | 1.149984 | 11.18B |
| 16-basis cross-fitted GREG | 0.813844 | 20/24 | 1.290361 | 21.39B |
| 16-basis, tail bounded | 0.842624 | 21/24 | 1.149995 | 21.39B |
| Full 129-basis direct | 0.713695 | 21/24 | 1.085891 | about 165B |

The only arm below 14B is near 0.94. The full companion passes the 0.75 development screen only by spending roughly another baseline’s compute and still misses the 0.595 promotion target. Low-rank compression cannot plausibly bridge this: the fixed output map retains only about 11.5%, 19.5%, 32.8%, and 52.3% energy at ranks 1, 2, 4, and 8; median rank 28 is needed for 90%.

## 4. Cost and implementation

T4.1’s p2/p4 computations reuse blocks already present in the coherent 17-block companion and add only reductions. The projected propagation ratio is **0.992** and projected adjusted ratio for the failed candidate is **1.118831**. These are proxies, not official scores.

The 48-record exact-rotation research run consumed 738.2 seconds of per-record harness time (mean 15.38s; maximum 21.82s) and mean peak RSS 1387.1 MiB. The harness materializes research arrays, so these are not deployable package timings.

## 5. Closed descendants

- No calibration or validation run for T4.1.
- No amplitude retuning.
- No orientation codebook.
- No nested-convergence gate.
- No generic confidence-feature sweep.
- No broad or tiny residual learner: T4.3’s prerequisite failed.
- No independent pair-moment branch.
- No additional companion basis-count tuning inside T4.2.

## 6. Retained assets and reopening criterion

Retain:

1. the exact selected-center contraction API;
2. fused primary pair-scalar accumulation;
3. complete c2/c4/c8/c12/c17, basis-increment, p2/p4, and error-geometry instrumentation;
4. the first-layer scalar GREG only as a nearly free add-on to some future independently accepted anchor;
5. the fixed `beta_bar` SVD for diagnostics.

Reopen only if a materially new runtime observable independently measures **absolute angular phase** and passes a fresh grouped P128 development block before any protected cohort is touched. Another policy, threshold, codebook, or learner on the same c17/p2/p4/nested/jackknife information is not a new hypothesis.

## 7. Limitations

This work closes **Tree T4 as specified**, not every conceivable future layer-31 estimator. The fresh cohort is full-width, exact-design, architecture-matched synthetic development evidence; it is not an official challenge subprocess. Official FlopScope and residual-wall measurements were unavailable. The P32/P128 robustness field is a fixed subset/full-map diagnostic, not the separately fitted historical K32/K128 pair. None of these limitations changes the preregistered stop decision because the primary grouped detection gate failed decisively and no protected cohort was opened.
