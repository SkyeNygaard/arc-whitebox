# Advanced cubature and control-functionals: outcome and cost audit

## Bottom line

The useful breakthrough was an exact low-degree spherical cubature rule, not a
more elaborate randomized sampler. A 66,048-point Kerdock maximal-real-MUB
spherical 5-design, with a structured Walsh-Hadamard first layer, achieved the
following frozen result:

| evaluation | previous two-stream RQMC | Kerdock 5-design | reduction |
|---|---:|---:|---:|
| Strict holdout, IDs 50--99, raw MSE | 3.74299369e-7 | 2.80643696e-7 | 25.02% |
| Official all 100, adjusted score | 3.46069946e-7 | **2.2565645893879035e-7** | **34.79%** |

The official score is a 1.534x improvement. The package used no holdout tuning,
had zero failures, and consumed a mean 268.899B effective compute against the
272B limit.

## Why this path worked

Positive homogeneity separates the Gaussian input into radius and direction:

```text
f(r u) = r f(u)
E[f(X)] = E[chi_256] E[f(U)].
```

The radius is therefore integrated analytically. The remaining angular
integrand is continuous and piecewise linear, but not smooth enough to realize
the theoretical advantage of higher-order digital nets. Antipodal symmetry
removes all odd harmonics; a spherical 5-design additionally annihilates the
degree-2 and degree-4 angular error exactly. Its first possible harmonic error
is degree 6.

In dimension 256, the general lower bound for an antipodal spherical 5-design
is `d(d+1) = 65,792` points. The Kerdock construction uses 66,048, only 256
above that bound. It is thus almost the smallest point set that can provide
this moment guarantee.

The design is the union of 129 real mutually unbiased bases and their
antipodes. The 128 non-coordinate bases are chirp-modulated Walsh-Hadamard
bases. This makes their first-layer matrix products computable by tracked
batched FWHTs; the remaining 31 network layers stay ordinary dense matrix
products.

## Fair compute comparison

The prior estimator evaluates 62,768 directions. Its tracked cost is
263.377B FLOPs and its measured effective cost is about 263.804B, leaving only
8.623B tracked or 8.196B effective-budget margin. Any proposed correction
must include its online feature formation and reductions.

| candidate | five-fold raw-MSE ratio vs RQMC | dominant extra online work | budget conclusion |
|---|---:|---:|---|
| Standard two-stream RQMC | 1.0000 | none | 263.377B tracked |
| Interlaced order-2 Sobol, both streams | 1.3007 | essentially same forward cost | Reject: 30.1% worse |
| Degree-2 spherical-harmonic CF, theory coefficient | 54.0653 | small final-output reductions after offline kernels | Reject: unstable high-dimensional projection |
| Degree-2 spherical-harmonic CF, tempered | 0.9835 | small online reductions | Only 1.65% gain; too small and coefficient-dependent |
| Cross-fitted full layer-1 linear CV | 0.9281 | `2 N d^2 = 8.227B` | About 271.604B tracked before residual overhead; marginal/no robust adjusted win |
| Cross-fitted antipodal-pair layer-1 CV | 0.9365 | `N d^2 = 4.114B` | Fits, but only 6.35% raw gain |
| Layer-1 linear plus rank-16 quadratic CV | 0.9576 | linear feature work plus quadratic projections | Worse than the simpler linear CV |
| Kerdock spherical 5-design + FWHT | strict holdout ratio 0.7498 | replaces first dense layer with structured work | **268.835B tracked; 268.899B effective; winner** |

For Kerdock, the independent audit measured:

| component | audited value |
|---|---:|
| Official tracked FLOPs/network | 268,835,176,704 |
| Mean official effective compute | 268,898,960,582 |
| Effective multiplier | 0.98859912 |
| Budget margin | 3.101B effective |
| Dense structured-output agreement, RMS | 4.81e-9 |
| Dense structured-output agreement, maximum | 1.68e-8 |

The standalone profiler recorded 268.823B tracked FLOPs. Its operation-level
accounting was dominated by 267.878B of matrix multiplies; FWHT adds/subtracts
and ancillary array operations made up the rest. The small difference from the
official package is explained by package-level accounting and does not alter
the budget conclusion.

## What the rejected paths taught us

### Higher-order digital nets and CBC lattices

Order-2 digit-interlaced scrambled Sobol used the same 32,768 + 30,000
direction budget but was 30.1% worse in five-fold selection. Mixed variants
were also worse: standard-A/interlaced-D was 1.0690x and
interlaced-A/standard-D was 1.1454x. Higher-order net and interlaced polynomial
lattice guarantees require high mixed smoothness. Inverse-normal/sphere maps
followed by 32 ReLU layers do not satisfy that regime.

CBC lattice construction can be offline and free at submission time, but the
online forward cost is unchanged. Existing lattice and lattice-frame trials
already lost to Sobol. Periodization would introduce another nonsmooth seam or
extra transforms, so there was no credible route to a 25%+ gain.

### Kernel quadrature and control functionals

Dense kernel quadrature at `N ~= 63k` needs an `N x N` kernel:
`O(N^2 d)` kernel formation and `O(N^3)` factorization are far beyond the
budget and memory limit. A compressed degree-2 reproducing-kernel control was
implemented instead:

```text
K2(u,v) = d(d+2)/2 ((u dot v)^2 - 1/d).
```

The theoretically unit-strength correction was 54x worse. It attempts to
estimate a 32,895-dimensional degree-2 harmonic projection from only about
15k unique antipodal pairs per stream. A fitted tempering coefficient
`-0.02117` reduced MSE by just 1.65% in five-fold selection. That is evidence
that low-degree residual energy is already small under antipodal RQMC, exactly
the component the 5-design removes deterministically.

The layer-1 control was more promising because its spherical mean and
covariance are available in closed form. Cross-fitting coefficients across
independent scrambles preserves randomized unbiasedness. It gave a real 7.19%
raw gain, but forming the full controls duplicates one dense layer over both
streams and consumes essentially all remaining budget margin. Pair-compressed
controls fit comfortably but sacrifice part of the gain. Quadratic extensions
did not recover it.

### Multilevel, adaptive Walsh, and orthogonal-array routes

MLQMC needs a cheap, convergent fidelity hierarchy. Depth truncation is still
31/32 of the full forward cost and is biased; nested point prefixes with
Richardson extrapolation were empirically worse. Walsh-spectrum diagnostics
are cheap enough, but cannot lower error without reallocating evaluations, and
nested adaptive allocation showed no gain. OA-LHS, ordinary orthogonal frames,
and tight frames impose substantially weaker moment structure than the
near-minimal spherical 5-design and were not competitive.

## Research decision

Freeze the Kerdock package as the submission candidate. Do not spend the final
budget margin on learned controls unless a fresh, disjoint benchmark shows an
adjusted improvement: the official Kerdock result already beats both the raw
and compute-adjusted baselines by a wide margin, while the auxiliary controls
were selected only on IDs 0--49.

The remaining plausible follow-up is a degree-6-aware rotation criterion or a
second exact-design family, evaluated under the same frozen 0--49/50--99
protocol. Further high-order QMC, generic kernel, and multilevel variants are
lower priority because their structural assumptions failed directly in this
integrand.

## Primary references

- Delsarte, Goethals, and Seidel,
  [Spherical codes and designs](https://doi.org/10.1007/BF03187604).
- Calderbank, Cameron, Kantor, and Seidel,
  [Z4-Kerdock codes, orthogonal spreads, and extremal Euclidean line-sets](https://doi.org/10.1112/S0024611597000403).
- Boykin et al.,
  [Real mutually unbiased bases](https://arxiv.org/abs/quant-ph/0502024).
- Dick,
  [Higher order scrambled digital nets achieve the optimal rate of the root mean square error for smooth integrands](https://arxiv.org/abs/1007.0842).
- Goda and Dick,
  [Interlaced polynomial lattice rules](https://arxiv.org/abs/1301.6441).
- Oates and Girolami,
  [Control functionals for quasi-Monte Carlo integration](https://proceedings.mlr.press/v51/oates16.html).
- Giles and Waterhouse,
  [Multilevel quasi-Monte Carlo path simulation](https://people.maths.ox.ac.uk/gilesm/files/jcf07.pdf).
- Tang,
  [Orthogonal array-based Latin hypercubes](https://doi.org/10.1080/01621459.1993.10476423).

## Reproducible artifacts

- `scripts/eval_interlaced_rqmc.py`
- `scripts/eval_spherical_harmonic_cf.py`
- `scripts/eval_crossfit_layer1_cv.py`
- `scripts/eval_layer1_quadratic_cv.py`
- `scripts/eval_kerdock_design.py`
- `scripts/audit_kerdock_fwht.py`
- `results/interlaced_rqmc.json`
- `results/spherical_harmonic_cf.json`
- `results/crossfit_layer1_cv.json`
- `results/crossfit_layer1_pair_cv.json`
- `results/layer1_quadratic_cv.json`
- `results/kerdock_design_holdout_50_99.json`
- `results/kerdock_fwht_audit.json`
- `results/kerdock_mub5_official_full100.json`
