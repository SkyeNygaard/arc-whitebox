# Kerdock multifidelity orientation control

## Result

On the untouched-during-development selection set, official IDs 0--49, a
matched low-fidelity orientation control reduces the full seed-3 Kerdock raw
MSE from `1.75874674e-7` to `1.35547477e-7`, a reduction of 22.93%.

The frozen rule uses 90,624 rows:

```text
F = F3 + (P0,S + P1,S - 2 P3,S) / 16.
```

`F3` is the complete 66,048-row Kerdock design at rotation seed 3. `P_r,S`
is the mean over the following 24 antipodal Kerdock bases at rotation seed
`r`:

```text
1, 3, 4, 5, 6, 13, 15, 16, 29, 35, 57, 59,
66, 72, 84, 85, 87, 95, 96, 101, 108, 118, 120, 124
```

Each basis has 256 vectors and their 256 antipodes. The seed-3 pilot is
already present in the full design, so only seeds 0 and 1 add work:

```text
2 rotations * 24 bases * 512 rows = 24,576 extra rows.
```

The unconstrained fitted coefficient was `0.120776`; freezing the nearby
dyadic value `1/8` changes MSE by less than 0.04%. Five-fold coefficients
were `0.1172, 0.1197, 0.1306, 0.1177, 0.1200`, and the five-fold refit-alpha
MSE was `1.36074846e-7`. The fixed rule improved 44 of 50 networks. Its
paired mean gain was `4.033e-8` with standard error `7.343e-9`.

IDs 50--99 were not inspected in this workstream.

## Why the construction works

Different rotations of a spherical design have the same invariant
degree-six error norm but differently oriented degree-six fingerprints. A
full average of the seed-3, seed-0, and seed-1 designs is extremely accurate
on selection, but it costs three full forwards.

For any alternate rotation `r`, the matched difference

```text
P_r,S - P_3,S
```

has much lower variance than an unmatched small design: both sides use the
same Kerdock bases, so much of the subset-specific cubature error cancels.
The difference is a low-fidelity proxy for the unavailable full-design
orientation difference `F_r-F_3`. Shrinking the proxy by `1/8` handles its
remaining subset noise.

Every component pilot is an antipodal union of orthonormal bases. Therefore
the correction has exactly zero total mass, zero odd moments, and zero
degree-two moment. It introduces only controlled even leakage beginning at
degree four. The frozen subset was in the best 1.9% of 10,000 random subsets
for quartic leakage and the best 8.0% for sixth-moment mismatch.

At the frozen coefficient the nominally signed quadrature is actually
positive. A selected seed-3 row retains weight
`1/66048 - 1/(8*12288) = 4.965e-6`, while alternate pilot rows have weight
`1/(16*12288)`. This also made the layer-2 weighted-variance experiment
well-defined without clamps.

## Why an exact sparse rotated mixture is unavailable

Let `M_B` denote the fourth-moment tensor of an orthonormal basis. A complete
129-basis maximal-MUB set has the exact spherical average:

```text
(1/129) sum_B M_B = M_sphere.
```

We evaluated 512 random quartic contractions for the union of the seed-0,
seed-1, and seed-3 systems. After enforcing each known complete-set relation,
the `512 x 387` feature matrix had rank 384 and nullity exactly 3. Those
three null directions are the uniform sums of the three complete systems.
Thus a generic rotated union has no additional resolved "MUB trade" that
could replace a sparse subset while retaining exact degree four. Controlled
leakage plus shrinkage is necessary unless every basis of another complete
design is evaluated.

## Compute path

The separately audited rectangular Winograd depth-4 `hybrid_p2` schedule
costs `7,829,873,664` tracked operations per 90,624-by-256 later-layer
product. Across 31 layers this is `242,726,083,584`, leaving enough margin
for the structured Kerdock first layers and reductions according to the
fast-matmul microbenchmark. An end-to-end Flopscope estimator remains the
required deployment gate.

We also searched the compute frontier. A 98,816-row, 32-basis pilot reached
`1.3301e-7`, but the projected effective compute was about `270.72B`: only
`1.28B` (roughly 12.8 ms of residual allowance) below the hard limit, which
was judged too fragile. Conservative 30- and 31-basis candidates were worse
than the frozen 24-basis rule. Greedily augmenting the frozen subset produced
an apparent `1.2146e-7`, but nested selection scored `1.3781e-7`; this was a
clear subset-selection overfit and was rejected.

Row ordering for implementation:

1. full seed-3 Kerdock rows, including the coordinate basis;
2. the 24 selected non-coordinate bases at seed 0;
3. the same 24 bases at seed 1.

The current structured Kerdock first layer orders each non-coordinate basis
as `(row 0 +, row 0 -, row 1 +, row 1 -, ...)`. Selection must therefore
happen on the leading basis index before flattening. The full seed-3 selected
mean can be reduced from its existing first 65,536 Kerdock rows.

## Layer-2 transport ablation

We also applied analytic fixed-radius mean/variance transport immediately
before the second ReLU, always aggregating with the same multifidelity
weights. Strength `0.5` produced only a 0.19% in-sample improvement:
`1.35290063e-7`. Its paired t-statistic was `0.188`, and five-fold
train-selected strength worsened the no-transport rule by 0.33%. The
transport is rejected; deployment should use strength zero.

## Artifacts

- `results/kerdock_multifidelity_selection.json`: canonical frozen result.
- `results/kerdock_multifidelity_size_ladder.json`: compute-safe pilots with
  8, 12, 16, and 20 bases per alternate rotation.
- `results/kerdock_multifidelity_h2_selection.json`: combined transport grid.
- `scripts/select_kerdock_multifidelity_pilots.py`: harmonic-screened subset
  selection and size ladder.
- `scripts/eval_kerdock_multifidelity_h2.py`: exact dense combined evaluation.
- `scripts/explore_kerdock_mixtures.py`: basis-cache and rotated-union
  research harness.
