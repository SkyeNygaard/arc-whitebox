# Costed subprocess design

## Retained path

During the existing layer-29 radial-feature pass, process complete basis blocks (512 rows) and retain only:

1. `H_i = H[:, selected_indices]`;
2. `HV = H @ V.T`, already required by the 128 radial probes;
3. reductions `sum(H_i^2)` and `sum(H_i * HV)`.

After radial scaling, these are the 128 diagonal moments and 128 row-direction moments. No `256 x 256` matrix is formed, no replay is required, and no new static asset is shipped.

For 66,048 rows and 128 probes, the incremental fused reductions cost 0.033816576B FLOPs. A standalone implementation without reuse costs 4.353883904B FLOPs; constructing the full second matrix costs 8.656977920B FLOPs and is rejected.

Use 512-row streaming buffers. Two float64 `512 x 128` buffers require about 1.05 MB. The final 256 scalar contractions require 2 KB in float64.

## External interface

The center estimator supplies only:

- `d_i = mu_i - m_i` for 128 selected coordinates;
- `a_p = v_p^T(mu-m)` for 128 probes.

The pair module supplies `s_p,t_p`, reconstructs `mu_i=m_i+d_i` and `z_p=v_p^T m+a_p`, evaluates the exact lower formula, and multiplies by the frozen `128 x 256` beta map.

## Rejected path

Do not use a companion cloud merely to estimate pair moments. Companion propagation dominates cost and the measured pair-source increment is output-negligible. A companion or micro-cubature may still be investigated for the center contractions, but it should reuse primary-cloud pair moments.
