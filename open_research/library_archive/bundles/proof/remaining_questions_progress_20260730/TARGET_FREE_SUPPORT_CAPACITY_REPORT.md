# Target-free layer-31 support capacity: exact-amplitude ceiling analysis

## Principle

For a frozen coordinate support `S`, scoring with the true layer-31 defect amplitudes on `S` is an upper bound on every deployable coefficient estimator restricted to that support. If this oracle support score fails a continuation gate, no amplitude learner on the same support can rescue the branch.

## High-reference results

At `K=32`:

| Rotation | radial-H3 support capture | PCA-sensitivity support capture | oracle best of four families per record |
|---:|---:|---:|---:|
| 3 | 51.08% | 53.21% | 53.48% |
| 19 | 36.50% | 38.90% | 40.53% |
| pooled | 45.52% | 47.75% | 48.54% |

The last column is target-labeled selection among the four named support families for each record. It is therefore an upper ceiling for any deployable selector that merely chooses among those four supports.

## Stable support is not stable capacity

The radial and PCA supports have high cross-rotation overlap, often near one, and satisfy the archived stability rule. Nevertheless, their exact-amplitude repair capacity differs sharply by rotation. Thus:

> index stability measures whether the same neurons are selected; it does not establish that those neurons span the same downstream-signed defect.

## Closure

The four-family `K<=32` support menu fails the required 50% capture gate on rotation 19 and also fails it pooled, even after target-labeled family selection. This closes:

- coefficient learning on each exact tested support;
- deployable mixing that only chooses among the four named supports;
- the claim that support stability alone makes Stage B worthwhile.

It does not close larger supports, a different basis, joint multi-checkpoint channels, or a support defined from an independently signed absolute-phase anchor.
