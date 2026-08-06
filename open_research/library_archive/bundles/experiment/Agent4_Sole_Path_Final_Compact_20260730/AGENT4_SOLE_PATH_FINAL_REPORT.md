# Agent 4 — Sole possible winning path: final source-specific checkpoint-gauge audit

**Competition:** WHestBench  
**Date:** 2026-07-30  
**Protected competition data opened:** **No**  
**Final disposition:** **STOP — the sole evidence-backed winning path fails a tight source-specific convex certificate.**

## Executive conclusion

The adaptive direct-output PCA source was the only surviving construction with competition-level oracle capacity. I reconstructed it exactly from the authenticated OGAP archive and then executed Agent 2's requested source-specific checkpoint-gauge SOCP with independent covariance fitting, untouched covariance validation, structured-sampling falsification, and an explicit primal/dual certificate.

The result is not close:

- confirmation source-only oracle residual ratio: **0.074860710**;
- maximum admissible contraction difficulty: **0.206408508**;
- development-selected `[1,4,32]` partition, untouched confirmation covariance: **S = 1.274291533**;
- miss factor: **6.17×**;
- unfair per-case oracle choice among the four frozen partitions: **S = 1.265269371**;
- empirical SOCP dual lower bound for selected partition: **S ≥ 1.155929800**;
- dual miss factor: **5.60×**;
- maximum primal/dual relative gap: **7.213e-06**;
- maximum dual checkpoint-balance residual: **1.749e-18**.

Even granting zero deployment cost for learning the controls, unlimited continuous sample allocation, and the source's exact oracle projection residual, the validated continuous optimum has adjusted ratio

\[
(\sqrt{r_*}+S)^2=2.395989,
\]

against the required **0.230415**. The theoretical optimum asks for **4.657 baseline compute units**, approximately **796.0B tracked FLOPs** at the current production trace.

This closes the declared **linear, unbiased, independent-block checkpoint-gauge estimator class** for the promoted adaptive direct-output source. The obvious orthonormal-basis structured-design escape also fails independently. There is no remaining candidate justified for protected evaluation.

## 1. Exact target being audited

For each realized network and rotation, let the 129 natural Kerdock/axis group output means be `y_g`, let `bar y` be their mean, and let `U` contain the adaptive leading eigenvectors of their output covariance. The frozen rule retains the smallest rank among the first 40 modes reaching cumulative energy threshold `0.9939596`.

The unknown source contraction is

\[
b=U^T(P-Q)h_32.
\]

Because `U` is built entirely from already-computed baseline group outputs, it is target-free, hidden-permutation invariant, positive-ReLU-rescaling invariant, and requires no nonlinear replay.

### Exact reconstruction check

I regenerated all 12 networks, all three frozen rotations, all 66,048 design outputs per case, and the independent reference targets from the authenticated archive. Across all 36 cases:

- rank range: 34–38;
- maximum baseline/source-capacity reconstruction error: **1.110e-16**;
- confirmation pooled source ratio: **0.074860710059**.

The numerical audit therefore targets the exact promoted source, not a random surrogate.

## 2. Convex estimator class

For a partition

\[
1=t_0<t_1<\cdots<t_m=32,
\]

choose arbitrary checkpoint controls `C_0,...,C_m` with `C_m=U`. Agent 2's exact telescope gives

\[
U^Th_32-C_0^Th_1
=\sum_{j=1}^m(C_j^Th_{t_j}-C_{j-1}^Th_{t_{j-1}}).
\]

The layer-1 mean is analytic. With independent unbiased block estimates, the exact variance-cost constant is

\[
S(C)=\sum_j\sqrt{\gamma_j v_j(C)}.
\]

For fixed covariance and costs, minimizing this over all intermediate controls is an SOCP. Direct terminal estimation is a feasible point. Thus a tight primal/dual result is a class-level certificate, not a statement about one regression fit.

## 3. Frozen protocol

### Corpus

- development: seeds 910001, 910003, 910009, 910019;
- untouched validation: 910033, 910043, 910051, 910067;
- independent confirmation: 910079, 910081, 910089, 910103;
- rotations: 31001, 31013, 31033;
- whole base network is the grouping unit.

### Covariance design

Primary audit:

- 2,048 independent antithetic fixed-radius Gaussian-direction pairs per base for fitting;
- a disjoint 2,048 pairs per base for covariance validation;
- fixed radius equal to `E chi_256`, which is exact for expectations of positively homogeneous degree-one network outputs;
- pair cost charged through the latest checkpoint in each block;
- covariance/control construction charged **zero** deployment cost, deliberately favoring continuation.

### Frozen partition family

- `[1,32]`;
- `[1,4,32]`;
- `[1,8,32]`;
- `[1,16,32]`.

The partition was selected only by development validation covariance.

## 4. Full 36-case frontier

| Partition | Development train S | Development valid S | Untouched validation S | Confirmation S |
|---|---:|---:|---:|---:|
| `p01_l1_final` | 1.162563 | 1.331143 | 1.540223 | 1.315091 |
| `p02_l1_4_final` | 1.174587 | 1.299339 | 1.492773 | 1.274292 |
| `p03_l1_8_final` | 1.178334 | 1.392483 | 1.478456 | 1.268230 |
| `p04_l1_16_final` | 1.189313 | 1.329039 | 1.519759 | 1.296173 |

Development selected `[1,4,32]`. On independent confirmation it validates at **1.274291533**, while the absolute allowable maximum is **0.206408508**.

The failure is not caused by partition selection: allowing an oracle to choose the best of the four partitions separately for each confirmation case only reaches **1.265269371**.

## 5. Explicit primal/dual certificate

For the selected `[1,4,32]` partition, I re-solved each of the 12 confirmation empirical SOCPs and projected the norm subgradient to the exact checkpoint-balance nullspace. Scaling to the dual norm balls produced:

| Quantity | Value |
|---|---:|
| Aggregate primal S | 1.155934202397 |
| Aggregate dual lower S | 1.155929799513 |
| Maximum relative gap | 7.213e-06 |
| Maximum balance residual | 1.749e-18 |
| Allowed S | 0.206408507645 |

The dual lower bound alone exceeds the gate by **5.60×**. Therefore solver convergence or a better choice of linear checkpoint controls cannot rescue the selected empirical program.

The dual certificate is for the empirical fit covariance. Population transfer is supplied separately by the disjoint covariance validation, which is even worse: **1.274291533**.

## 6. Dense-partition loophole

A representative exact extension on `seed910001_rot31001` tested the broader Agent 2 checkpoint family:

| Partition | Checkpoints | Train S | Valid S |
|---|---|---:|---:|
| `p01_l1_final` | `[1, 32]` | 1.293572 | 1.476470 |
| `p02_l1_4_final` | `[1, 4, 32]` | 1.327731 | 1.463923 |
| `p03_l1_8_final` | `[1, 8, 32]` | 1.325418 | 1.468716 |
| `p04_l1_16_final` | `[1, 16, 32]` | 1.344858 | 1.477920 |
| `p05_l1_24_final` | `[1, 24, 32]` | 1.343556 | 1.492093 |
| `p06_l1_29_final` | `[1, 29, 32]` | 1.319039 | 1.488625 |
| `p07_l1_4_8_final` | `[1, 4, 8, 32]` | 1.361599 | 1.471409 |
| `p08_l1_4_8_16_final` | `[1, 4, 8, 16, 32]` | 1.394460 | 1.474745 |
| `p09_l1_4_8_16_24_final` | `[1, 4, 8, 16, 24, 32]` | 1.416781 | 1.487015 |
| `p10_agent2_dense` | `[1, 4, 8, 16, 24, 27, 29, 31, 32]` | 1.443107 | 1.503543 |

The dense `[1,4,8,16,24,27,29,31,32]` partition validates at **1.503542834**, worse than `[1,4,32]` at **1.463923466**.

This is deliberately scoped: the dense family was not rerun over all 36 cases because the representative result worsened, while the frozen four-partition confirmation result already missed by more than sixfold.

## 7. Structured orthonormal-basis escape

A possible objection is that independent antithetic pairs are the wrong sampling unit and that a complete orthonormal basis could cancel low-order variation. I therefore generated independent Haar orthonormal bases with antipodes:

- 512 rows per unit;
- 1,024 independent bases for fitting;
- 1,024 untouched bases for validation;
- seed 910079, all three confirmation rotations.

| Partition | Train S | Untouched valid S |
|---|---:|---:|
| `p01_l1_final` | 0.780633 | 1.056754 |
| `p02_l1_4_final` | 0.851746 | 1.058783 |
| `p03_l1_8_final` | 0.846019 | 1.050765 |
| `p04_l1_16_final` | 0.837902 | 1.032107 |

Best untouched value: **1.032107109**. Local allowable maximum for those three cases: **0.211527053**. Miss factor: **4.88×**.

A prior small same-sample pilot had appeared to reach approximately `0.315`; the independent 1,024/1,024 audit shows this was covariance overfit, not a real escape.

## 8. Economics

For the selected validated `S`, the continuous T72 optimum is

\[
x^*=\frac{S}{\sqrt{r_*}}=4.657382
\]

baseline compute units, and

\[
J^*=(\sqrt{r_*}+S)^2=2.395989.
\]

At the production tracked trace `170.906815B`, `x*` corresponds to about **796.0B added FLOPs**. Under historical incremental caps the result is worse:

| Added compute cap | Adjusted ratio using validated S | Dual-certified lower-bound ratio |
|---:|---:|---:|
| 10B | 29.4552 | 24.2515 |
| 14B | 21.5278 | 17.7287 |

These calculations already grant free covariance fitting and free control construction. Real deployment costs can only worsen them.

## 9. Claims I tried to disprove

### “The source was reconstructed incorrectly.”

Disproved as an explanation. The regenerated baseline and source ratios match the archived Agent 8 values to at most `1.11e-16`.

### “The optimizer found a local minimum.”

Disproved for the selected empirical SOCP. The explicit dual lower bound matches the primal within `7.21e-6` relative error.

### “A different shallow partition wins.”

Disproved on the frozen tournament and confirmation cohort. Even per-case oracle partition choice remains at `S=1.26527`.

### “More checkpoints create the missing control.”

Not universally proved, but directly falsified on the representative dense audit: every denser tested partition was worse.

### “Pair sampling is needlessly noisy.”

The strongest obvious escape, independent complete Haar bases, still validates at `S=1.03211`, about 4.88× its local gate.

### “The covariance estimate merely overfit.”

Overfit exists and hurts: development train objectives are below validation objectives, and the small ONB pilot was misleading. Untouched validation and independent confirmation are the reported decision statistics.

### “Compute accounting caused the failure.”

False. The program fails even after assigning covariance/control learning zero deployment cost and optimizing added compute without a cap.

## 10. Exact scope of closure

### Closed by this package

- the promoted adaptive direct-output PCA source coupled to the declared linear checkpoint-gauge telescope;
- independent unbiased block allocation under the tested antithetic fixed-radius design;
- the development-selected `[1,4,32]` empirical SOCP, by primal/dual certificate;
- the frozen shallow partition family across all 36 cases;
- the obvious complete-orthonormal-basis structured sampling escape on an independent confirmation-base audit.

### Not proved impossible in full generality

- arbitrary nonlinear checkpoint controls;
- biased or shrinkage estimators exploiting favorable source-error cross terms;
- exotic dependent estimators outside the independent-block T72 model;
- a new exact analytic identity unavailable to the present transcript;
- every conceivable structured cubature unit;
- a different legal source not presently known.

These logical possibilities are not current competition paths. Reopening requires a concrete theorem or mechanism explaining how it evades the certified `S` barrier—not another generic predictor or source search.

## 11. Conflicts and corrections

1. **“Sole possible winning path” status:** it was conditional on the source-specific convex audit. That audit now fails decisively, so no evidence-backed path remains.
2. **Small ONB pilot:** the apparent near-gate value was same-sample covariance overfit. The independent structured audit supersedes it.
3. **Dense-partition completeness:** only one representative case was tested for the full dense family. Do not describe this as an exhaustive all-partition theorem.
4. **Dual versus population:** the dual certificate proves the empirical fit-covariance optimum. Population failure is established numerically by untouched covariance validation, not by the dual alone.
5. **Universal impossibility:** this package closes a broad and precisely declared class, not all adaptive nonlinear white-box estimators.

## 12. Final recommendation

**Stop competition promotion and do not open protected data.**

The adaptive direct-output source genuinely solved representation, but its absolute contractions are too expensive to estimate through the complete tested linear checkpoint-gauge family. The gap is so wide that incremental partition tuning, larger covariance samples, or ordinary implementation optimization cannot plausibly reverse the conclusion.

The defensible next state is project closure/documentation: preserve the source theorem, replay results, convex checkpoint-gauge theorem, and this negative certificate as a paper-quality map of exactly where the obstacle lies. Reopen constructive competition work only upon a new identity or nonlinear mechanism with a derivation that directly attacks absolute signed phase.

## 13. Reproduction

Primary full audit:

```bash
python run_source_specific_socp_audit.py \
  --ogap /path/to/whest_experiments_oracle_gap_20260730 \
  --out /tmp/sole_path_audit \
  --n-train 2048 --n-valid 2048 --quick
```

Selected-partition dual certificate:

```bash
python certify_selected_partition.py
```

Structured ONB escape audit:

```bash
python run_onb_escape_audit.py
```

Final package verification:

```bash
python verify_sole_path_package.py
```

The full reproducibility bundle includes the authenticated 30.7 MB OGAP archive. No protected competition data is included or was opened.
