# Kerdock 5-design, depth-4 batched Winograd, exact dead-column elimination

Statistically identical to the previously graded Kerdock/Winograd submission:
same 66,048-point Kerdock/MUB spherical 5-design, same fixed radius `E[chi_256]`,
same propagated values. Every change is arithmetic and exact.

## What changed and why

The grader charges `effective_compute = tracked_flops + 1e11 * residual_wall_seconds`.
One second of untracked wall time therefore costs 37% of the whole budget.

Measuring the previous submission: tracked FLOPs are **1.709e11**, but the graded
multiplier was **0.785**, which implies the grader spends **~0.43 s per network**
on work flopscope does not attribute (this machine spends ~0.048 s).

That residual scales with the **number** of tracked calls, not their size
(~4.8 us/call locally, ~56 us/call on the grader). Each Winograd level cuts
multiplies by 7/8 but multiplies branches by 7, so kernel depth trades tracked
FLOPs against call count. Optimising tracked FLOPs alone picks the wrong point:

| kernel | tracked | calls | projected multiplier |
|---|---|---|---|
| depth-5 output tree (previous) | 170.9B | 7,592 | 0.785 (graded) |
| depth-4 batched | 185.7B | 2,226 | 0.7285 |
| **depth-4 batched + dead columns** | **173.3B** | **2,414** | **0.6868** |

Two changes:

1. **Depth-4 batched Winograd.** All `7**4` branches stay in leading batch axes,
   so there is one `matmul` call per layer and the call count is linear in depth
   rather than exponential.
2. **Exact dead-column elimination.** By depth 24, 20-27% of hidden units never
   fire on *any* design row. A zero activation column contributes exactly zero,
   so the matching rows of the next weight matrix are dropped. No pilot, no
   approximation: deadness is read off activations already computed
   (`max(axis=0) == 0`, valid because ReLU output is non-negative). The alive
   count is padded up to the granularity 16 using dead columns, which are zero,
   so the result stays exact. Verified bit-identical to the dense product.

## Accuracy

Unchanged. Against an exact dense fp64 reference on official IDs 0-19:

| kernel | mean final-layer MSE | vs fp64 |
|---|---|---|
| dense fp64 (reference) | 1.7170e-07 | 1.00000 |
| previous depth-5 tree | 1.7173e-07 | 1.00017 |
| this submission | 1.7170e-07 | 1.00002 |

Fewer recursion levels accumulate slightly less fp32 error, so this is marginally
*more* accurate than the kernel it replaces.

## Projected effect

Graded baseline: raw MSE 2.42e-7, multiplier 0.785, adjusted **1.90e-7**.
Two independent projections of the multiplier change:

* per-call model (56.1 us/call, calibrated from the graded multiplier): 0.6886
  -> adjusted **1.67e-7**
* residual-scaling model (grader ~7.9x local residual): 0.704 -> adjusted **1.70e-7**

i.e. a **10-12% improvement**, entirely from arithmetic. Raw MSE is unchanged,
so this is orthogonal to any statistical work.

## Selection protocol

Development and measurement used official Mini IDs 0-49 only. The frozen holdout
(IDs 50-99) was not opened. No hyperparameter here is fitted to network data:
the Winograd depth is chosen by a cost model, and dead-column elimination is
exact rather than tuned.
