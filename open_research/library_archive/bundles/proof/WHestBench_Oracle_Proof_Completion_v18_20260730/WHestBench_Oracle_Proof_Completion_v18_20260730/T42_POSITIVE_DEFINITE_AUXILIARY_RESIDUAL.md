# T42 — Positive-definite optimized auxiliary residual

**Status:** computer-assisted theorem candidate; independently rerun inside this package, but still requires a second interval stack and named human proof review before publication.

## Statement

Let

\[
K_{32}(t)=\sum_{\ell\ge 0} k_\ell G_\ell^{(256)}(t)
\]

be the depth-32 normalized ReLU kernel in dimension 256, and let

\[
h_*(t)=\sum_{\ell=0}^{5} c_\ell G_\ell^{(256)}(t)
\]

be the certified degree-five T16 Hermite auxiliary optimizer. Then every normalized-Gegenbauer coefficient of

\[
q(t)=K_{32}(t)-h_*(t)
\]

is nonnegative, and the coefficients in degrees 0 through 5 are strictly positive. Consequently, `q` is a positive-definite zonal kernel on the sphere `S^255`.

## Analytic reduction

The normalized ReLU dual activation has the Maclaurin expansion

\[
\kappa(t)=\frac1\pi+\frac t2+
\sum_{m\ge1}
\frac{\operatorname{Cat}_{m-1}}
{\pi\,2^{2m-1}(2m-1)}t^{2m},
\]

so every Maclaurin coefficient is nonnegative. Composition and multiplication of convergent nonnegative-coefficient series preserve coefficientwise nonnegativity. Therefore every Maclaurin coefficient of `K_32 = kappa^{\circ 32}` is nonnegative.

For every integer `n >= 0`, the monomial kernel

\[
\langle x,y\rangle^n
=
\langle x^{\otimes n},y^{\otimes n}\rangle
\]

is positive definite. Its normalized-Gegenbauer coefficients are therefore nonnegative. A finite lower truncation of the Maclaurin series of `K_32` supplies a rigorous lower bound for each `k_ell`.

## Computer-assisted coefficient comparison

The bundled verifier propagates a directed 90-digit interval Taylor jet through all 32 kernel compositions, retaining powers through degree 47. It then projects each retained monomial exactly into normalized Gegenbauer coefficients using rational spherical moments.

| degree | lower bound for `k_l` | certified upper bound for `c_l` | certified positive margin |
|---:|---:|---:|---:|
| 0 | 0.9747299895417147123 | 0.9747299751309444414 | 1.4410770271e-8 |
| 1 | 0.0027966328997355175 | 0.0027964730615411842 | 1.5983819433e-7 |
| 2 | 0.0024438109271863426 | 0.0024362952737152224 | 7.5156534711e-6 |
| 3 | 0.0018364440800827765 | 0.0018037348551971006 | 3.2709224886e-5 |
| 4 | 0.0015312373489132821 | 0.0010317284867674261 | 4.9950886215e-4 |
| 5 | 0.0012573233584175919 | 0.0001798989234636446 | 1.0774244350e-3 |

All omitted Maclaurin terms contribute nonnegatively. Thus `k_l-c_l>0` for `0<=l<=5`. For `l>=6`, the auxiliary coefficient is zero and `k_l>=0`.

## Consequences

1. The optimized auxiliary is not merely a pointwise minorant: it subtracts a positive-definite component from the ReLU kernel.
2. Complete-Kerdock error, which annihilates the auxiliary degrees in its stated design scope, can be viewed as discrepancy in the residual kernel `q`.
3. This coefficientwise decomposition supports the signed-rule rank comparison in T43.

## Non-consequences

This theorem does **not** transfer the positive-weight T22/T16 lower bound unchanged to arbitrary signed rules. The positive-rule argument uses a diagonal residual term controlled by nonnegative weights. For signed rules, positive definiteness only guarantees nonnegative centered kernel discrepancy; cancellations remain possible.

It also does not apply automatically to finite-width kernels or to a network-dependent residual transformation. Each such kernel requires its own certified coefficients.

## Verification

Run:

```bash
python code/verify_oracle_proof_completion.py
```

The detailed interval and exact-rational output is stored in `results/signed_floor_order47.json`.
