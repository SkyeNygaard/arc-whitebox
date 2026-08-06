# Decision

## Verdict: retain one module; close the standalone family

**Close independent selected pair-moment estimation as a standalone Path 1 branch.**

Retain only the fused primary-cloud accumulator for

- `s_p = M_s[i_p,i_p]`;
- `t_p = M_s[i_p,:] @ v_p`.

These are algebraically necessary, but an independent estimate does not materially change the final correction. The center defect supplies the sign and amplitude.

## Decisive results

- High-reference exact center: exact pairs **0.889350** versus primary pairs **0.889914**.
- New 24-network, 128-probe, full-companion-center ablation at alpha 0.50:
  - independent pairs: **0.726008**, 20/24 wins, worst 1.231;
  - primary pairs: **0.726019,** 20/24 wins, worst 1.235;
  - paired ratio difference: **-0.000010**, bootstrap 95% interval [-0.000390, +0.000366].
- Median independent-pair increment: **0.174%** of full correction norm; median full-versus-primary-pair cosine **0.999999479**.
- The safer alpha 0.20 arm reaches 23/24 wins and worst 1.012, but raw ratio **0.790** misses the 0.75 development gate.

The post-hoc 128-probe alpha 0.50 arm passes the development ratio at **0.726**, but fails the promotion ratio, tail, and compute gates. It is center-estimator mechanism evidence only.

## Smallest defensible representation

At runtime keep 256 observable primary-cloud pair scalars (`128 s_p + 128 t_p`) and estimate only 256 external center contractions (`128 d_i + 128 v_p^T d`). `mu_i` and `v_p^T mu` follow algebraically. Do not construct a full covariance and do not propagate an independent cloud for pair moments.

## Next branch

Redirect all statistical work to the selected center contractions. Reuse this pair module unchanged in analytic centered-defect, independent micro-cubature, or shared-arithmetic center estimators.
