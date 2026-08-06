# Agent 3 — Independent Checks

## Numerical reconstruction

`verify_t27.py` independently reconstructs the depth-32 normalized ReLU kernel from the arc-cosine recurrence and computes the spherical kernel mean with 512-node Gauss–Jacobi quadrature.

Reproduced constants:

| Quantity | Independent value |
|---|---:|
| `A-O` | `0.011988581160655598` |
| `O-C` | `-9.468153657654632e-06` |
| `C-A0` | `-4.6263743280761105e-08` |
| `c(256)` | `3.73622415011563e-05` |
| `(A-O)+256(O-C)` | `0.009564733824296012` |
| full 33,024-line risk | `2.43366035798e-07` |

## Risk decomposition

Five hundred independently generated signed rules were checked by constructing their explicit three-class Gram matrices and comparing `w^T G w-A0` with the reduced formula. Maximum absolute discrepancy:

`3.31e-14`.

## Fixed-support signed-weight optimization

Three thousand random support patterns and arbitrary signed feasible weight vectors were compared with the closed-form optimum. No violation occurred. The closed-form weight vector reproduced its stated risk to maximum absolute error:

`4.34e-19`.

The minimum observed signed-rule gap was zero only in degenerate/equality cases such as one-line supports; no signed rule beat the optimum.

## Support allocation

- Exhaustive enumeration was performed for every budget in a five-bin, capacity-eight integer analogue using the actual T27 `h(r)` function. Every optimum had complete-bin concentration plus at most one partial bin.
- Random allocation challenges were run at actual boundary budgets `1, 2, 255, 256, 257, 511, 512, 513, 33,023, 33,024`. No allocation exceeded the concentrated-basis value.
- The analytic discrete exchange proof is authoritative; random checks are regression tests only.

## Missing original dependency

The Library contains the theorem memo and a report stating that 26,000 random signed-weight trials had zero violations, but the original stress-test script, row-level output, and manifest were not located. The independent verifier supplied here reproduces and strengthens those checks, but the original 26,000-trial claim remains non-reproducible from the currently located files alone.
