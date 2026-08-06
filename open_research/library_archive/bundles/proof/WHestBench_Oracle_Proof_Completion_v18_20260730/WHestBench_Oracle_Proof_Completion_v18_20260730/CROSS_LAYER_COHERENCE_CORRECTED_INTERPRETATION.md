# Cross-layer Oracle coherence — corrected interpretation

## Data object

For checkpoint repair vectors `r_1,...,r_k`, define increments

\[
\Delta_1=r_1,\qquad \Delta_j=r_j-r_{j-1}.
\]

The continuation concatenates each `Delta_j` over all output coordinates and all cases in a split, then computes a normalized Gram matrix.

## Reproduced pooled diagnostics

| Split | Max absolute off-diagonal cosine | Effective rank | Cross-term fraction | Last-two energy share |
|---|---:|---:|---:|---:|
| Development | 0.3017 | 3.8245 | -6.28% | 7.25% |
| Validation | 0.1461 | 3.8025 | -2.93% | 8.23% |
| Confirmation | 0.1871 | 4.2869 | +7.80% | 9.84% |

The effective rank is

\[
r_{\mathrm{eff}}=\frac{(\operatorname{tr}G)^2}{\operatorname{tr}(G^2)}.
\]

## Valid conclusion

The pooled campaign is inconsistent with a single common checkpoint direction whose amplitude merely increases with depth. Multiple checkpoint intervals contribute substantial pooled energy, and the final two intervals are a minority of summed increment energy.

## Invalid stronger conclusion

Pooled orthogonality does not imply within-network orthogonality or multiple within-network components. Cross-network sign heterogeneity can create the same statistic.

### Minimal counterexample

Two cases have scalar output. Let their increment values be

| Case | Increment 1 | Increment 2 |
|---|---:|---:|
| A | 1 | 1 |
| B | 1 | -1 |

After concatenation, the increments are `(1,1)` and `(1,-1)`, with cosine zero. Yet every case is one-dimensional.

## Stronger diagnostic still needed

For each base network, preserve or regenerate:

1. the checkpoint repair vectors in a common scored-output coordinate system;
2. the per-network increment Gram matrix;
3. eigenvalues and effective rank;
4. signs and magnitudes of cross terms;
5. stability over rotations;
6. a transported downstream singular basis, so modes are comparable across checkpoints.

Report the distribution over base networks, not only a pooled matrix. The decisive claim would be one of:

- median within-network effective rank exceeds a preregistered threshold;
- a fixed transported basis captures several independently varying coefficients;
- one scalar component fails a held-out reconstruction test against a rank-`r` model.

Until then, use **pooled checkpoint heterogeneity**, not **approximately incoherent causal components**.
