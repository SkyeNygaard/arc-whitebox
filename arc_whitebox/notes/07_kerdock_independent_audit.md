# Independent audit: Kerdock maximal-MUB estimator

## Verdict

The construction is mathematically valid, the batched FWHT computes the intended
first layer, and the complete estimator is safely inside the Phase-1 compute
budget. The strongest audit checks every nontrivial pair of Kerdock bases rather
than relying on sampled moment checks.

Audit artifacts:

- `scripts/audit_kerdock_fwht.py`
- `results/kerdock_fwht_audit.json`

## Exact design certificate

Let `d = 256` and `M = d/2 + 1 = 129`. The construction contains 128
signed-Walsh Kerdock bases and the coordinate basis.

The finite-field and code checks passed exhaustively:

- all 127 nonzero residues have inverses under the implemented
  `GF(128) = GF(2)[x]/(x^7+x+1)` multiplication;
- multiplication by every nonzero residue permutes all 127 nonzero elements;
- all 128 chirps are distinct;
- all `128 choose 2 = 8,128` nontrivial chirp-basis pairs were checked;
- every pair-product Walsh spectrum has absolute value exactly `16 = sqrt(d)`;
- mutual unbiasedness with the coordinate basis is automatic because every
  signed-Walsh entry has magnitude `1/sqrt(d)`.

For a union of `M` real mutually unbiased bases, the fourth frame potential is

```text
FP4 = M d + M(M - 1).
```

The first term is the contribution within bases and the second is the
contribution between ordered pairs of bases. Here this is exactly

```text
FP4 = 129*256 + 129*128 = 49,536.
```

There are `N = Md = 33,024` unoriented unit vectors. The real projective
Welch bound is

```text
3 N^2 / (d(d + 2)) = 49,536.
```

Equality certifies a real projective 2-design, hence exact spherical moments
through degree four after choosing representatives. Adding every antipode
kills all odd moments, including degree five, while preserving the even
moments. The resulting `2N = 66,048` points therefore form a spherical
5-design.

The input Gaussian separates into an independent uniform direction and
chi-distributed radius. Because a bias-free ReLU MLP is positively homogeneous,
evaluating every direction at radius `E[chi_256]` integrates the radial variable
exactly. The only approximation is the remaining angular content above degree
five.

Primary construction reference: A. R. Calderbank, P. J. Cameron, W. M.
Kantor, and J. J. Seidel, *Z4-Kerdock codes, orthogonal spreads, and extremal
Euclidean line-sets* (1997). The standard upper bound for real MUBs is
`d/2 + 1`, attained here.

## Structured first layer

Let `H` be the unnormalized `256 x 256` Walsh matrix, `c_u` a Kerdock chirp,
`R` the fixed rotation, `W_1` the first weight matrix, and
`r = E[chi_256]`. Folding the rotation into the weight gives

```text
W_eff = R W_1.
```

The positive half of Kerdock basis `u` has first-layer preactivation

```text
(r/sqrt(d)) H diag(c_u) W_eff.
```

Thus all 128 bases and 256 output columns can be transformed together with
eight FWHT butterfly stages. Their antipodes are exact negatives and require
no second transform. The coordinate basis contributes `+/- r W_eff`.
After concatenation and the first ReLU, the remaining 31 layers are ordinary
dense forwards over all 66,048 rows.

Against the materialized dense design on official mini ID 0:

| comparison | max absolute error | RMS error |
|---|---:|---:|
| FWHT first layer vs folded dense product | `6.20e-6` | `4.22e-7` |
| final 256-vector | `1.68e-8` | `4.81e-9` |

The final-layer MSE changed from `1.73261319e-7` to `1.73261430e-7`.
This is ordinary float32 summation-order drift, not a formula discrepancy.

## Authoritative Flopscope 0.9.1 ledger

Profiling the actual submission estimator on official mini ID 0 gave:

| quantity | measured value |
|---|---:|
| instrumented FLOPs `F` | `268,833,079,552` |
| hard FLOP margin to `272e9` | `3,166,920,448` |
| residual wall time in the audit run | `0.000678245 s` |
| effective compute `F + 1e11 R` | `268,900,904,073` |
| effective-compute margin | `3,099,095,927` |

The dominant charge is 32 matrix multiplications:

```text
267,877,679,104 FLOPs
```

The full ledger also charges operations that older starter-kit prose may imply
are free: `reshape`, `stack`, `concatenate`, and `astype`. These charges are
already included above. In particular, the 32 redundant float32 weight casts
cost only `2,097,152` FLOPs and are not a budget threat, though they can be
removed because official weights are already float32.

The approximately `3.10e9` effective margin permits about `31 ms` of residual
Python time at the `1e11 FLOP/s` rate after instrumented work. The measured
residual is below `0.7 ms`, so the margin is comfortable.

## Caveats checked before packaging

1. **Generic validation shape.** `whest validate` in whestbench 0.13.0 calls
   `predict` on a synthetic width-4, depth-2 MLP. The specialized estimator
   must return a finite `(2, 4)` fallback for this probe. The final package
   includes that branch and passes both `whest validate` and
   `validate-package`.

2. **Float32 rotation.** The unrotated Kerdock set has the exact design
   certificate above. The stored QR rotation is float32, so in literal real
   arithmetic it is only approximately orthogonal:
   `||R^T R-I||_2 = 7.19e-8`, with maximum singular-value deviation
   `3.59e-8`. This is negligible relative to quadrature error and is covered
   by the dense-versus-structured numerical comparison.

3. **Frozen orientation.** Rotation seed 3 was selected on mini IDs 0--49 and
   frozen before IDs 50--99 were evaluated. It improves both halves, but its
   benefit on the disjoint 1,000-network full split remains an empirical
   generalization question, not a consequence of the 5-design theorem.

4. **Scope of exactness.** “Spherical 5-design” means exact angular integration
   for polynomials of total degree at most five. A deep ReLU network is
   piecewise linear, not a degree-five polynomial, so the observed score is an
   empirical quadrature result rather than an exact network integral.

5. **Architecture dependence.** The FWHT asset and cost calculation are
   specialized to the fixed challenge architecture `width=256, depth=32`.
   This is appropriate for Phase 1 but should remain explicit in the code and
   documentation.

