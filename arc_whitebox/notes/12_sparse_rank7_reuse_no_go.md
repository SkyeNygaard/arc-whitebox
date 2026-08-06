# Generalized sparse rank-7 reuse: exact, but not release-competitive

## Construction

The audit implemented the five-coordinate `r=2` factorization of the current
Winograd coefficient operators.

For the left leg:

```text
Phi_U(A) = [a11, a12, a22, s1, s2]
s1 = a21 + a22
s2 = s1 - a11

U_phi = [x1, x2, x2-x5, x3, x4, x5, x3-x5]
```

For the right leg:

```text
Phi_V(B) = [b11, b21, b22, t1, t2]
t1 = b12 - b11
t2 = b22 - t1

V_phi = [x1, x2, x3, x5-x2, x4, x5, x5-x1]
```

The seven products decode into the five-coordinate output dictionary with
four additions:

```text
y1 = p1 + p2
y2 = p3 - p7
y3 = p4
y4 = p5
y5 = p1 + p6 + p7
```

The final canonical transform needs three:

```text
s = y4 + y5
C = [[y1, y2+s], [y5-y3, s]]
```

All recursion depths 1--3 and every packed prefix were checked in float64.
The worst small exactness error was below `3.0e-14`.

## Implementations tested

The stacked implementation keeps explicit five- and seven-coordinate tensor
axes. It has low Python residual time, but Flopscope bills the materializing
`stack` copies.

The tuple-tree implementation keeps five-coordinate objects as nested Python
tuples, materializes a flattened rank prefix only once, decodes that prefix
through strided views, and reconstructs canonical blocks without intermediate
stacks. This recovers the theoretical arithmetic reduction, but the large
number of tracked wrapper calls creates prohibitive residual and wall time.

A third mixed implementation applies sparse reuse only on outer recursion
levels and ordinary depth-first Winograd below them.

## Decisive measurements

| implementation | schedule | tracked/layer | residual/layer | effective/layer |
|---|---|---:|---:|---:|
| tuple tree | d5, p2, chunk64 | `5,220,824,768` | `0.253717 s` | `30.5925 B` |
| stacked | d5, p4, chunk128 | `6,425,508,160` | `0.002731 s` | `6.69865 B` |
| sparse outer + Winograd inner | k2/d5, chunk64 | `5,616,163,584` | `0.008112 s` | `6.42736 B` |
| mixed full-batch arithmetic minimum | k3/d5 | `5,493,622,528` | `0.083414 s` | `13.8350 B` |

The best low-residual sparse schedule projects to at least `199.25 B`
effective compute across the 31 later layers before first-layer overhead. The
best stacked schedule projects to `207.66 B`. Both are worse than the
`175.871 B` partial-tree release candidate.

The tuple representation establishes that the mathematical saving is real;
the tracker/runtime model prevents it from becoming a scoring improvement.
No submission package was created.

Implementation and raw audits:

- `scripts/eval_sparse_rank7_reuse.py`
- `results/sparse_rank7_reuse_smoke.json`
- `results/sparse_rank7_reuse_sweep_a.json`
- `results/sparse_rank7_reuse_sweep_chunks.json`
- `results/sparse_rank7_reuse_tuple_smoke.json`
- `results/sparse_rank7_reuse_tuple_chunks.json`
