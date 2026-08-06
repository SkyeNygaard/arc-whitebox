# Finite-width transfer: exact proof-cost frontier

The released degree-280 signed certificate can be truncated without changing any proof logic. At cutoff `M`, retain only components whose squared feature kernel has maximum harmonic degree at most `M`. The resulting comparison is still coefficientwise dominated by the infinite-width kernel and its objective is an exact lower bound.

## Key thresholds

| Infinite-width signed floor retained | Maximum coefficient degree needed |
|---:|---:|
| 50% | 22 |
| 60% | 28 |
| 70% | 40 |
| 80% | 62 |
| 85% | 84 |
| 90% | 128 |
| 92% | 164 |
| 93% | 194 |
| 93.5% | 214 |
| 93.7% | 242 |
| Full 93.7046% | 280 |

These are valid subcertificates, not reoptimized numerical approximations.

## Conditional finite-width theorem

Let `H_M` be a cutoff-`M` subcertificate with infinite-width fraction `f_M`. Suppose the width-256 ensemble coefficients obey

\[
k_\ell^{(256)}\ge \alpha_M h_{M,\ell}
\quad(1\le\ell\le M),
\]

and finite-width Kerdock MSE is at most `beta` times the certified infinite-width Kerdock upper endpoint. Then every arbitrary signed static width-256 rule satisfies

\[
{R_{256}(Q)\over R_{256}(Q_K)}
\ge {\alpha_M f_M\over\beta}.
\]

This exposes a tunable rigor burden. A useful finite-width result need not certify all 280 coefficients:

- a 90% infinite-width structural floor requires only degrees through 128;
- a 93% floor requires degrees through 194;
- lower cutoffs can serve as stepping-stone theorems.

## Compressed finite-width program

The exact finite-width kernel is a first-layer Gaussian noise-stability kernel

\[
K_m(t)=\sum_{n\ge0}a_n^{(m)}t^n,
\qquad a_n^{(m)}\ge0.
\]

Its Gegenbauer coefficients are positive linear functionals of the Hermite-chaos energies `a_n^(m)`. A practical proof program is therefore:

1. interval-bound a finite set of finite-width noise-stability values `K_m(t_j)` or chaos moments;
2. solve a Hausdorff-moment LP over nonnegative `a_n^(m)`;
3. lower-bound only the coefficient combination required by one cutoff subcertificate;
4. separately upper-bound finite-width Kerdock risk.

This is potentially much easier than reconstructing the entire finite-width harmonic spectrum.
