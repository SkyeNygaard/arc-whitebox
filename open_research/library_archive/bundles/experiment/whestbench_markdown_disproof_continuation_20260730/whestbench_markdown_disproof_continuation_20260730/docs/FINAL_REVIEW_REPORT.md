# WHestBench Markdown Disproof Review and Stronger Signed-Node Certificate

**Date:** 2026-07-30  
**Disposition:** consequential stale claims reconciled; one proposed escape class tested; the signed-node theorem materially strengthened.

## Executive result

The review changed four live conclusions.

1. **Canonical v19 is stale about M164.** The exact 24-basis/rank-24 partial-MUB configuration was later frozen and evaluated on its untouched 24-network validation set. It retained real raw signal (`1.111064x`, 19/24 wins) but failed adjusted economics (`0.939729x` baseline/candidate), confidence, and tail (`1.260358` worst). The 64-network holdout remained sealed.
2. **Pooled checkpoint incoherence is not a cross-network cancellation artifact.** On confirmation bases, the six signed checkpoint increments have mean within-base effective rank `4.111`; every base needs 4–5 components for 90% energy. Layer 31 is a multi-component repair interface, not one scalar bias hidden by pooling.
3. **The canonical orientation-odd escape tested here fails.** Twenty-seven deterministic sign-odd contractions genuinely break the M153 quotient, but grouped development-only ridge replay is worse than both the original even features and a constant policy. This closes that exact feature/source/policy class, not orientation-aware algorithms generally.
4. **T47 was valid but numerically nonfinal.** A maximal degree-23 weighted harmonic certificate raises the arbitrary-signed-node floor from `0.5051771254707468684835910828016483122671976958137404698301809540413111` to `0.6107992099573098167548734683367801325016165687480884574205433422658960056609730815417199665904831677` of Kerdock MSE and lowers the maximum same-cost improvement from `1.979503721725632018365924276231889075530212725569634647432660120680737x` to `1.637199236177617740208281690867658785095971701858447488463638213863606096302494557408655906728062700x`.

## 1. Markdown reconciliation

### M164 / PM24

The v18/v19 files called the exposed PM24 screen a required future falsifier. A later frozen-validation package already resolves it:

- candidate/baseline MSE: `0.900038099`;
- baseline/candidate raw gain: `1.111064077x`;
- raw 95% interval: `[1.024848128, 1.224765471]`;
- compute ratio: `1.182324214`;
- adjusted baseline/candidate ratio: `0.939728768x`;
- adjusted interval: `[0.866808035, 1.035896463]`;
- wins: `19/24`;
- worst candidate/baseline: `1.260357877`.

**Decision:** close the tested 24-basis/rank-24 sampled phase family. Do not rewrite this as zero alignment: the mean raw signal is positive, but it is too expensive and tail-unsafe.

### Salvaged ancillary theorems

The later hostile/salvage files correctly repair T29 uniqueness, T38 degeneracy, conditional Haar, replication bias, the ReLU crossing bound, optimizer transfer, observability reporting, and T16 endpoint equality. I found no new contradiction requiring those replacements to be withdrawn. Their application assumptions must remain adjacent to each headline.

## 2. Within-network checkpoint geometry

The earlier pooled analysis concatenated cases before computing the increment Gram matrix. That permits a hostile alternative: each network might be rank one, with different directions canceling across networks.

I recomputed the six-increment spectrum at two levels:

- per rotation in `R^256`;
- per base network after concatenating its three predetermined rotations in `R^768`.

| Split | Mean base effective rank | Base rank for 90% energy | Mean absolute off-diagonal cosine |
|---|---:|---:|---:|
| Development | 3.680 | 4–5 | 0.0896 |
| Validation | 3.852 | 4–5 | 0.0907 |
| Confirmation | **4.111** | **4–5** | **0.0862** |

Every base network in every split has effective rank at least 3 and needs at least four components for 90% energy. The pooled conclusion therefore survives its strongest obvious artifact attack.

**Implication:** future oracle-derived source bases should be multi-mode. A single signed center amplitude is not supported by the observed decomposition.

## 3. Orientation-odd M153 escape diagnostic

The abstract gauge theorem is exact only for an invariant representation. The original nine M153 features are even under simultaneous sign reversal, but physical runtime vectors have canonical coordinates. I constructed nine sign-odd legal summaries per source, including fixed-coordinate projections, normalized mean, signed max-magnitude pivot, contractions with reduced output and primary-delta references, deterministic cosine axes, and a half-block contrast.

The 48 exposed rows were evaluated by nested leave-one-base-network-out ridge, with all three rotations grouped.

| Policy | Candidate/baseline | Wins | Worst |
|---|---:|---:|---:|
| Constant OOF coefficients | 1.1182 | 14/48 | 2.2332 |
| Original even features | 1.1351 | 15/48 | 2.4448 |
| Orientation-odd features | 1.1422 | 15/48 | 2.4403 |
| Even + odd | 1.1447 | 15/48 | 2.4485 |
| Per-case box oracle | 0.9151 | 26/48 | 2.0626 |

All coefficient-prediction R-squared values remain negative. The odd map disproves any claim that phase-sensitive legal features are mathematically absent, but it supplies no useful transferable signal in this exact class. The source oracle itself is only `1.0927x`, below the A50 `1.20x` prerequisite, so no validation cohort should be opened.

## 4. Strengthened arbitrary-signed-node theorem

### Theorem T47-R2

Let `Q = sum_i w_i delta_(x_i)`, with `m <= 66,048` and `sum_i w_i = 1`, where nodes on `S^255` and real weights are arbitrary. For the normalized depth-32 limiting ReLU kernel,

`R_K32(Q) >= 1.486477823691692199205348920799113356042073114405760804883283330382893220172192897042154673402266531E-7`.

Relative to the certified complete-Kerdock MSE upper endpoint,

`R_K32(Q) >= 0.6107992099573098167548734683367801325016165687480884574205433422658960056609730815417199665904831677 * R_Kerdock`,

so every same-cost static signed linear rule improves by at most `1.637199236177617740208281690867658785095971701858447488463638213863606096302494557408655906728062700x`.

### Certificate construction

Choose exact rational harmonic weights through degree 23 and form `L_a(t) = sum_(ell=0)^23 a_ell d_ell G_ell(t)`. Its square has positive Gegenbauer coefficients `b_r`. A 384-bit directed-MPFR Taylor jet supplies rigorous lower bounds `k_r^-` through degree 46. Put `gamma = min_(1<=r<=46) k_r^- / b_r`. The binding degree is **8**. The weighted rank/trace lemma gives the exact rank defect for any rank at most 66,048 matrix with the prescribed trace. Multiplying gives the floor.

### Independent verification

Two separate stacks agree:

1. direct-C MPFR interval jet plus a handwritten exact `Fraction` Gegenbauer recurrence, with no SymPy or `mpmath.iv`;
2. SymPy exact rational algebra plus the original directed interval stack.

They agree to more than 68 significant decimal digits on every headline quantity. The prior MPFR audit also obtained byte-identical GCC/Clang output and clean ASan/UBSan execution.

### Comparison

| Certificate | Kerdock fraction retained | Same-cost improvement cap |
|---|---:|---:|
| Earlier multirank theorem | 0.3246803 | 3.0799530x |
| T47 degree 15 | 0.5051771 | 1.9795037x |
| **T47-R2 degree 23** | **0.6107992** | **1.6371992x** |

The new floor is about `1.209079x` the T47 floor. It still does not prove signed near-optimality, and it does not cover finite width, adaptation, nonlinear aggregation, or free total mass.

## 5. Updated path ranking

1. **Radical compute reduction remains the only direct competition-scale operational route.** The stronger signed floor makes equal-cost static signed cubature still less plausible.
2. **Finite-width specialization of the weighted-rank theorem is the highest-value remaining proof.** It would transfer the static signed floor from the limiting kernel to the actual width-256 ensemble.
3. **Residual-kernel transformation remains mathematically open**, but only a surrogate with a large exact mean and a demonstrably simpler residual spectrum is worth reopening.
4. **Adaptive/nonlinear white-box methods remain open.** The tested even and canonical odd feature maps fail; this is not a full-transcript theorem.
5. **Official A47 timing remains externally blocked.** Its millisecond-scale wall-time break-even cannot be resolved honestly outside the pinned grader/FlopScope environment.

## 6. Canonical corrections

- Supersede canonical T47's numerical floor by T47-R2 while retaining the same scope.
- Mark M164/PM24 `FROZEN VALIDATION FAILED`; remove it from open-items lists.
- Replace “within-network oracle rank unknown” by the authenticated multi-component result.
- Mark the tested A50-style canonical odd features `DEVELOPMENT DIAGNOSTIC FAILED`; do not claim all odd observables fail.
- Keep all universal nonlinear/statistical impossibility wording prohibited.
