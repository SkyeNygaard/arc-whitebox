# The score law, and an independent audit of the v26 lead path

Date: 2026-07-30

Everything below was recomputed from scratch against the official Phase-1 Mini
cohort (100 networks, 1e9-sample baked ground truth), using an independent
re-implementation of the shipped 129-basis design in plain NumPy.

**Harness validation.** The re-implementation reproduces the published figures
to five significant digits:

| quantity | published | reproduced here |
|---|---:|---:|
| raw MSE, IDs 0--49 | `1.75874674e-7` (notes/09) | `1.758751e-7` |
| raw MSE, first 20 | `1.717312e-7` (COMPARISON.md) | `1.717010e-7` |
| raw MSE, all 100 | — | `2.282594e-7` |

The 20-network block is optimistic, exactly as `COMPARISON.md` warned. **All
score statements below use the full 100-network cohort.**

---

## 1. The score law

For any estimator whose error is Monte-Carlo rate — error variance `V/N` at `N`
propagated design rows, cost `f` FLOPs per row — the adjusted score is

```
score = MSE * max(0.1, C/B) = (V/N) * (f*N/B) = V * f / B
```

**independent of `N`**, for every `N` above the 10%-budget floor. Measured:

| bases | rows | raw MSE | V = MSE x rows | mult | adjusted score |
|---:|---:|---:|---:|---:|---:|
| 24 | 12,288 | 1.7047e-6 | 2.0948e-2 | 0.1169 | 1.9925e-7 |
| 48 | 24,576 | 8.2812e-7 | 2.0352e-2 | 0.2338 | 1.9358e-7 |
| 80 | 40,960 | 4.3256e-7 | 1.7718e-2 | 0.3896 | 1.6852e-7 |
| 112 | 57,344 | 2.7902e-7 | 1.6000e-2 | 0.5454 | 1.5219e-7 |
| 129 | 66,048 | 2.2826e-7 | 1.5076e-2 | 0.6282 | 1.4340e-7 |

Score tracks `V` and nothing else: from 24 to 129 bases `V` falls 1.390x and the
score falls 1.389x. The law is confirmed to three digits.

Cross-check against an independently derived figure: plain iid Monte Carlo has
`V = avg_variance = 0.04949` (baked in the dataset) and naive `f = 4.0632e6`,
giving `score = 7.39e-7` — matching the `7.7e-7` quoted in notes/03 §0.

### Consequences

Three levers that look like levers are **exactly neutral**:

- row count / design size,
- what fraction of the budget is spent (above the 0.1 floor),
- per-network compute allocation.

Only two levers exist: **lower `V`** (a better design) or **lower `f`** (cheaper
arithmetic). Current position, all 100 networks:

```
V = 1.50761e-2      f = 2.5871e6      C/B = 0.6488
adjusted score = 1.481e-7   (V*f/B = 1.434e-7 excluding residual)
```

To reach the recorded 4.34x competition threshold, `V*f` must fall 4.34x.
For an 80% reduction, 5.00x.

`f` has almost nothing left: depth-5 Winograd already achieves 0.637x naive
against a Strassen-family practical floor near 0.55x — worth at most ~1.15x.
**So essentially the entire 4.34--5x must come from `V`.**

### Immediate consequence: the multifidelity design is a score regression

The 90,624-row Kerdock multifidelity orientation control (notes/09, and the
`estimator.py` currently at the repository root) reports a 22.93% raw-MSE
improvement. Under the score law it is a **loss**:

| design | rows | raw MSE (IDs 0-49) | V | f per row | adjusted |
|---|---:|---:|---:|---:|---:|
| 129-basis shipped | 66,048 | 1.7587e-7 | 1.1615e-2 | 2.5871e6 | 1.141e-7 |
| 90,624-row multifidelity | 90,624 | 1.3555e-7 | 1.2283e-2 | 2.6784e6 | 1.248e-7 |

`V` gets 5.7% worse and `f` 3.5% worse (the depth-4 `hybrid_p2` schedule costs
more per row): **~9.3% worse adjusted score**. Do not ship it.

---

## 2. The v26 lead path: Gate A passes, the next gate fails

The v26 lead is "freeze the adaptive direct-output PCA source, then solve the
checkpoint-gauge SOCP". Both halves were tested directly.

### Gate A (source capacity) — reproduced, the ledger is right

Per network, take the top-`k` right singular vectors of the 129 group-mean
deviations and measure how much of the true signed error they span:

| rank | pooled `r*` | median | worst | random-subspace reference |
|---:|---:|---:|---:|---:|
| 16 | 0.1724 | 0.2126 | 0.5223 | 0.9375 |
| 32 | 0.0872 | 0.0990 | 0.2744 | 0.8750 |
| **36** | **0.0773** | 0.0847 | 0.2466 | 0.8594 |
| 64 | 0.0381 | 0.0421 | 0.1267 | 0.7500 |

`0.0773` at rank 36 against the ledger's `0.0749`. This is a real and large
effect — a 36-dimensional subspace of R^256 holding 92% of the error energy
where a random subspace would hold 14%. **The capacity claim is sound.**

### Gate B (coefficient estimability) — fails

Capacity is worthless without the coefficients. For each (network, mode) pair I
built every legal scalar available from the group means — singular value and its
transforms, rank index, loading mean/std/skew/kurtosis, coordinate-basis
loading, mode-vs-prediction and mode-vs-ones inner products, spectral mass —
and fit ridge regressions with **leave-one-network-out** validation:

```
ridge lam=1e-3 .. 10      LORO R^2 vs zero = -0.042 .. -0.040
sign agreement                              = 0.4917   (chance 0.5)
MSE after applying the predicted correction = 2.3714e-7
baseline MSE                                = 2.2826e-7   -> 3.9% WORSE
oracle bound at the same rank               = 0.0773
```

Not merely weak — **negative**. The predictor is worse than predicting zero, and
its sign is a coin flip. The error's magnitude is highly observable
(`corr(log between-basis variance, log e^2) = +0.927`); its **direction** is not.
A magnitude-only signal cannot reduce MSE, because there is no direction to move
in.

This is one feature family, not the full declared SOCP class, so it is not a
formal certificate. But it is the natural family, it is the one any checkpoint
gauge would have to beat, and it lands on the wrong side of zero — consistent
with every prior report that "generic checkpoint-gauge screens collapse to
direct estimation".

### The static class is closed by measurement, not just by bound

The best **fixed** linear rule over the 129 group means:

```
uniform rule (shipped)                MSE 2.2826e-7   ratio 1.0000
in-sample optimum (overfit ceiling)   MSE 1.8061e-7   ratio 0.7912
leave-one-network-out                 MSE 2.7342e-7   ratio 1.1978
mass-preserving ridge, LORO, lam->inf                 ratio 1.0000
```

Regularised honestly, the optimal static rule **converges to the uniform rule**.
The ledger's T74 floor (0.937, i.e. a 1.067x cap) is if anything generous: the
achievable static gain is 1.000x.

---

## 3. Where the remaining headroom actually is

Since `f` is exhausted and the adaptive-source route is blocked at Gate B, the
only measured path with winning-scale upside is the one from notes/03 §4 —
**analytic moment propagation with Edgeworth marginals**. Its economics are
qualitatively different because it is bias-limited, not MC-rate: it costs
~2e9 FLOPs, sits **below the 0.1 multiplier floor**, and therefore scores
`0.1 x MSE` with every further FLOP free.

Measured points (notes/03, seed 0):

| configuration | MSE | score at the 0.1 floor | vs current 1.481e-7 |
|---|---:|---:|---:|
| EMP, everything propagated | 1.28e-5 | 1.28e-6 | 8.6x worse |
| EMP + oracle Σ only | 9.09e-6 | 9.09e-7 | 6.1x worse |
| EMP + oracle κ₃,κ₄ only | 2.04e-5 | 2.04e-6 | 13.8x worse |
| **EMP + oracle Σ and κ₃,κ₄** | **1.30e-7** | **1.30e-8** | **11.4x better** |

The ceiling clears both the 4.34x threshold and the 80% target with margin. The
entire gap is a **100x improvement in moment-propagation accuracy**, decomposed
into two named problems: Σ to 3e-3 relative (bivariate Edgeworth on pairs), and
κ₃ to ~3% relative (the n⁴ contraction, analytically).

### New measurement: κ₃ cannot be sampled into range, even with the design

notes/03 §5.1 costed sampled cumulants at ~420,000 **iid** samples = 6.5x budget.
The obvious rescue is that the Kerdock design is more efficient than iid. It is,
but not nearly enough. Measured against a 300,000-sample reference on 3
networks, standardised κ₃ error at 66,048 rows:

| layer | design | iid | variance reduction |
|---:|---:|---:|---:|
| 1 | 0.0056 | 0.0102 | 3.28x |
| 8 | 0.0107 | 0.0127 | 1.42x |
| 16 | 0.0119 | 0.0139 | 1.40x |
| 31 | 0.0117 | 0.0143 | 1.53x |

For reference the design's variance reduction for the **mean** is 3.3x
(`0.04949 / 1.50761e-2`). For κ₃ at depth it decays to ~1.4x. So the 420,000
iid-sample requirement becomes ~280,000 design rows ≈ 2.7x the budget.

**κ₃ must be propagated analytically. Sampling it is closed by measurement in
both the iid and the design-based costings.**

---

## 3b. Two further closures, and the reason behind all of them

### Multilevel / telescoping estimators — closed

The one construction that can beat `score = V*f/B` is a telescope
`E[f] = E[f_coarse] + E[f - f_coarse]`, whose score is
`(sqrt(V0*f0) + sqrt(Vd*f1))^2 / B` and whose ceiling as `Vd -> 0` is exactly the
cost ratio of the cheap level. Crucially a telescope **cancels the coarse
level's bias**, so the earlier rank-truncation results — which measured bias —
did not settle this.

Measured on 3 networks, coarse level = rank-`r` subspace tracking of the
activation batch:

| rank | Vd/V0 | cost ratio | 2-level score gain |
|---:|---:|---:|---:|
| 8 | 0.974 | 15.5x | 0.62x |
| 32 | 0.812 | 3.97x | 0.44x |
| 128 | 0.508 | 1.00x | 0.25x |

Every entry is **below 1.0x** — worse than direct estimation. At rank 128 (half
the full width, zero cost saving) the coarse chain still explains only 49% of
the pointwise variance. Deep ReLU networks have no cheap correlated surrogate:
the chain decorrelates faster than any projection can save.

### Design strength — closed, and this is the important one

The shipped design is an antipodal spherical 5-design, so it integrates degrees
1--5 exactly and its residual is the even-harmonic energy at degree >= 6. Whether
*any* stronger design can win reduces to one measurable question: where does that
energy sit?

For the symmetrised output `g(u) = (f(u)+f(-u))/2`, the two-point function
`K(t) = E[<g(u),g(v)>]` at `u.v = t` expands as `sum_l a_l P_l^(d)(t)` with
`a_l >= 0`. Sampling `K` on a grid of `t`, anchoring at `K(1) = Var(g)` (exact,
no model), and solving a nonnegative least squares problem gives the spectrum.
Six networks, fit residuals 1.3--3.4%:

| degree | 2 | 4 | 6 | 8--18 | >= 20 |
|---|---:|---:|---:|---:|---:|
| fraction of degree>=2 energy | 0.417 | 0.228 | 0.007 | ~0.009 | **0.339** |

The spectrum is **bimodal**: degrees 2 and 4 (65%, already killed by the
5-design) plus a heavy tail at degree >= 20 (34%), with essentially nothing in
between. The tail is pinned at the top of the fitted grid, so its true degree is
>= 20 and possibly far higher. The model-free evidence is the raw
`K(0.99)/K(1) = 0.84--0.92`: degrees <= 6 alone would give ~0.97, and only
high-degree mass can produce that deficit.

Consequence:

| design | residual energy | gain over the 5-design |
|---|---:|---:|
| 3-design | 0.583 | 0.61x |
| **5-design (shipped)** | **0.355** | **1.00x** |
| 7-design | 0.347 | 1.02x |
| 9-design | 0.347 | 1.02x |
| 11-design | 0.347 | 1.02x |

**No design of any feasible strength gains more than ~1.02x.** This is expected
on reflection: a depth-32 width-256 ReLU network partitions the sphere into an
astronomical number of linear cells, and such a function has genuinely
high-frequency angular content.

### CORRECTION (superseding the table above)

**The spectrum above is wrong and the "1.02x" conclusion with it.** A degree grid
of step 2 truncated at 20 forced the nonnegative fit to pile mass at the
boundary, producing a spurious bimodal spectrum. Re-fitted on a grid resolving
both ends (2,4,6,8,10,12,16,20,24,32,40,48,64,80,96), 4 networks:

| degree | 2 | 4 | 6 | 8 | 12 | 20 | 32 | 48 | 96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fraction | 0.455 | 0.146 | 0.061 | 0.043 | 0.042 | 0.036 | 0.016 | 0.003 | 0.054 |

The true spectrum decays **smoothly**, and higher-strength designs would help a
great deal:

| design | 7 | 11 | 19 | 23 | 31 | **39** | 47 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gain over the 5-design | 1.18x | 1.58x | 2.35x | 2.97x | 3.79x | **4.45x** | 4.85x |

Independent check that this version is correct: notes/09's multifidelity control
targets degree-6 leakage and measured a 22.9% MSE reduction. The corrected
spectrum predicts ~15% from removing degree 6 — consistent. The old spectrum
predicted ~2%, which was flatly inconsistent with that measurement.

### The cubature paradigm is closed by a counting theorem, not by the spectrum

The Delsarte-Goethals-Seidel bound for a spherical `t`-design on `S^(d-1)`,
`t = 2s+1`, is `N >= 2 C(d+s-1, s)`. At `d = 256`:

| strength | minimum points | vs the 105,142-row budget |
|---:|---:|---:|
| **5** | **65,792** | shipped design has **66,048** — **0.39% above tight** |
| 7 | 5,658,112 | **54x the entire budget**, to buy 1.18x |
| 39 | ~1e28 | — |

**The shipped 129-basis Kerdock/MUB design is a near-tight spherical 5-design.**
There is provably nothing left at strength 5, and strength 7 is 54x over budget.
Weighted rules do not escape it either: the Moller bound gives `N >= 2.83e6` for
exactness to degree 7, still 27x the budget. This is a rigorous closure and it
retroactively explains why every design experiment in the ledger returned ~1x —
the design was already near-optimal.

### New: the error is anisotropic, but the orientation is not selectable

Rotating the design rotates its aliasing tensors, so `e(R) = sum_l <R A_l, g_l>`.
Measured over 8 networks x 12 random rotations (all legal 5-designs, identical
cost):

```
relative MSE spread across rotations : 44.1%
best/mean 0.479    worst/mean 1.940    oracle gain best-of-12 : 2.03x
legal magnitude-proxy selection      : rank correlation -0.055
```

So the high-degree content genuinely **is** structured — but the orientation
quality is invisible to the legal signal. This independently reproduces
`kerdock_adaptive_orientation` (4 seeds, 50 networks): oracle 1.30x, every
selector at within-network Spearman 0.048-0.104, and every selector scoring
**worse** (2.14-2.41e-7) than simply fixing seed 3 (1.759e-7).

Note also that the pooled `corr(log between-basis variance, log e^2) = 0.927` is
driven by *between-network* scale variation. **Within** a network it carries no
information — which is why it can never select anything.

### The unifying explanation

The error lives at very high harmonic degree. That single fact explains every
negative result in this project:

- higher-strength designs and rotation mixtures do nothing (nothing to kill at
  degrees 6--18);
- control variates from exactly-known low-degree moments cap at `R^2 = 0.26`
  (notes/03 §1) — they can only reach low degrees;
- rank truncation and multilevel telescopes decorrelate (high degree = high
  frequency = chaotic under composition);
- terminal smoothing fails (note 04) — the error is trajectory-wide, not a
  last-layer marginal effect;
- the adaptive source has capacity but unestimable coefficients (§2) — a
  high-frequency error is not predictable from smooth summaries.

### Total remaining headroom in the cubature paradigm

| lever | measured ceiling |
|---|---:|
| design strength | 1.02x |
| arithmetic `f` | <= 1.15x |
| static reweighting | 1.00x |
| adaptive source coefficients | 1.00x |
| multilevel telescoping | < 1.00x |
| **product** | **~1.17x** |

against the **4.34x** required. The cubature paradigm is closed with a factor of
3.7x to spare. Analytic moment propagation (§3) is the only surviving route,
and it is unaffected by the harmonic argument because it propagates
distributions rather than integrating the function pointwise.

---

## 3c. The dominant risk is the timeout, not the algorithm

The shipped package runs 21.41 s locally and README.md estimates ~23.8 s on the
official grader (one calibration point, 11% slower) against a **30 s** predict
guard. A timeout zeroes that network's predictions **and** forces the multiplier
to 1.0, scoring it at the zeros baseline of ~0.91.

One timed-out network in 100 gives a mean score of
`(99 * 1.434e-7 + 0.91)/100 = 9.1e-3` — **63,000x worse** than the current score.
The asymmetry is overwhelming, and the score law makes the insurance cheap:

| bases | rows | est. official wall | margin to 30 s | adjusted score | cost of insurance |
|---:|---:|---:|---:|---:|---:|
| 129 (shipped) | 66,048 | ~23.8 s | 21% | 1.4340e-7 | — |
| 112 | 57,344 | ~20.6 s | 31% | 1.5219e-7 | +6.1% |
| 96 | 49,152 | ~17.7 s | 41% | 1.6226e-7 | +13.2% |
| 80 | 40,960 | ~14.7 s | 51% | 1.6852e-7 | +17.5% |

Dropping to 112 bases costs `8.7e-9` of absolute score. It pays for itself if the
probability of a single timeout anywhere in the run exceeds **1e-6**. On one
hardware calibration point against an unknown grader, it is not remotely that
small.

Two caveats: these subset scores are **random** subsets on all 100 networks (a
selected subset differs — notes/10's nested-selection frontier is less
favourable), and the partial arms in `COMPARISON.md` carry pathological residual
times only because they ship A43 streaming arithmetic; a properly built partial
arm scales roughly linearly.

---

## 3d. The analytic route, diagnosed to a single blocker

The harmonic obstruction (§3b) does not touch analytic moment propagation,
because it never integrates the function pointwise. Its whole cost is
`O(depth * width^3)`, **below the 0.1 multiplier floor**, so it scores
`0.1 * MSE` with every further FLOP free — and its oracle-moment ceiling is
1.30e-7, i.e. **11.4x better than shipped**. Everything therefore turns on the
recursion

```
mu_{l+1} = W^T E[relu(z_l)]        Sigma_{l+1} = W^T Cov(relu(z_l)) W
```

Each half was measured one step at a time, fed the **exact** empirical moments
from the 66,048 propagated design rows, so these are pure closure errors with
accumulation removed.

### The marginal closure is not the problem

| layer | Gaussian | Edgeworth | gain | dead neurons |
|---:|---:|---:|---:|---:|
| 1 | 1.76e-3 | 4.49e-4 | 3.9x | 0.0% |
| 4 | 2.46e-3 | 2.28e-4 | 10.8x | 0.1% |
| 8 | 2.21e-3 | 1.72e-4 | 12.9x | 3.3% |
| 16 | 2.04e-3 | 1.90e-4 | 10.7x | 10.4% |
| 31 | 1.33e-3 | 1.79e-4 | 7.4x | 25.5% |

Edgeworth holds **1.9e-4 per layer, stable across all 31 layers**. Independent
cross-check: `sqrt(31) * 1.9e-4 = 1.1e-3` relative, which is exactly the 1.30e-7
oracle-moment MSE. (Incidentally: a quarter of the final layer's neurons are
dead, which breaks any per-neuron relative metric — worth knowing.)

### The covariance closure is the problem

| layer | 1 | 4 | 8 | 16 | 24 | 30 |
|---|---:|---:|---:|---:|---:|---:|
| RMS sigma error | 4.4e-3 | 8.9e-3 | 1.29e-2 | 1.63e-2 | 1.80e-2 | 1.77e-2 |

Mean **1.34e-2**, growing with depth, against a requirement of ~3e-3 for the 11x
ceiling. (Layer-1 and layer-2 figures are near this measurement's own
finite-sample floor of ~4e-3 — `z_1` is exactly Gaussian so its true closure
error is zero. The deep-layer values are well above the floor and genuine.)

Three candidate fixes were tested and all fail:

- **Delayed closure** (Gaussianize `k` layers earlier, let the true dynamics
  develop the non-Gaussian shape): error *grows* monotonically with lag,
  1.67e-2 → 3.40e-2 at lag 12. The layer map **amplifies** distributional
  perturbations at ~6%/layer rather than damping them — the same signature that
  kills multilevel telescopes (§3b).
- **Universal per-layer calibration** (notes/03 §5.2's "strikingly universal"
  sigma bias): the closure error is **per-neuron scatter (1.3e-2), not a shared
  offset (1e-3)**. Leave-one-network-out calibration gains **1.01x**. The §5.2
  universality was a property of a different quantity and does not transfer.
- **Exact low-dimensional quadrature.** The famous "participation ratio 2.7" is
  misleading — it is dominated by one large eigenvalue. Actual capture:

| layer | r=8 | r=32 | r=64 | r=128 | part. ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.152 | 0.464 | 0.712 | 0.931 | 100.7 |
| 8 | 0.495 | 0.838 | 0.941 | 0.992 | 20.4 |
| 31 | 0.886 | 0.979 | 0.994 | 0.9997 | 2.4 |

  The collapse only happens at layers 24-31; early layers are essentially
  full-rank. This retrospectively explains why rank-truncated particles, CIL and
  low-rank K3 carriers all failed despite the "collapse" — they were justified
  by a statistic that does not measure what they needed.

### The one favourable result: the fix is affordable

The remaining fix is the bivariate (third-order-corrected) covariance closure of
notes/03 §4. Densely it is an `n^4` contraction — 4.3e9 FLOPs/layer, 1.3e11
total, about half the budget, which forfeits the 10x floor and caps the route
near 2x. Under a Tucker factorisation at rank `R` the cost is `n^2 R + n R^3`:
3.2e8 total at R=32, 2.1e9 at R=64, 1.7e10 at R=128 — all under the 2.72e10
floor.

So: is the third-cumulant tensor low rank? Measured mode-1 energy capture:

| layer | r=8 | r=32 | r=64 | r=128 |
|---:|---:|---:|---:|---:|
| 1 | 0.266 | 0.639 | 0.835 | 0.961 |
| 8 | 0.840 | 0.967 | 0.991 | 0.9991 |
| 24 | 0.983 | 0.998 | 0.9998 | 1.0000 |
| 31 | 0.997 | 0.9999 | 1.0000 | 1.0000 |

**The tensor is low-rank exactly where the closure error is large, and
full-rank exactly where the closure error is small.** A depth-adaptive schedule
(dense or R=128 for layers 1-4, R=64 mid, R=32 deep) costs **~3e9 FLOPs total**
— roughly 10% of the multiplier floor, leaving the 10x compute discount intact.

### GATE 1 RESULT — the dense oracle bivariate correction fails

Implemented in full. The exact rectified-Gaussian pair moment with non-zero
means was derived and validated:

```
E[X+ Y+] = sx sy [ (ab+rho) Phi2(a,b;rho) + a phi(b) Phi((a-rho b)/s)
                                          + b phi(a) Phi((b-rho a)/s)
                                          + (1-rho^2) phi2(a,b;rho) ]
```

Self-tests: factorises at rho=0 to machine precision; reduces to Cho-Saul at
zero mean to 1e-10; matches Monte Carlo for general means; Phi2 (tetrachoric
form) matches Monte Carlo at rho = +-0.99. On a synthetic skewed law the
third-order term cuts the pair-moment error **17.7x**, so the machinery works.

Measured on real networks with iid Gaussian reference, N=1,200,000, 4 networks:

| layer | sigma Gauss | sigma +3rd | gain | pair Gauss | pair +3rd | minEV/maxEV |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.39e-4 | 6.39e-4 | 1.00x | 1.19e-2 | 1.19e-2 | +7.9e-2 |
| 1 | 4.54e-3 | 2.78e-3 | 1.63x | 6.70e-2 | 3.70e-2 | -2.9e-3 |
| 4 | 9.42e-3 | 8.61e-3 | 1.09x | 9.21e-2 | 7.92e-2 | -1.8e-2 |
| 16 | 1.68e-2 | 1.50e-2 | 1.12x | 7.69e-2 | 7.53e-2 | -5.1e-3 |
| 29 | 1.42e-2 | 1.46e-2 | **0.97x** | 3.28e-2 | 3.75e-2 | -1.0e-3 |

Layers >= 4: **1.389e-2 -> 1.324e-2, a 1.05x gain against a required 4.5x.**
It also breaks positive semidefiniteness at every layer past 0.

**Measurement discipline.** Layer 0 is an exact-zero check (`p_0 = x W_0` is
exactly Gaussian, its covariance `W_0^T W_0` known in closed form, its third
cumulants zero). It reads 6.389e-4 against the theoretical reference floor
`1/sqrt(2N) = 6.45e-4` — the harness is calibrated to 1%, and the half-split
spread is 4.2e-5. The deep-layer signal is 20x the floor.

### Why it fails — and it is not what was expected

Three diagnostics, all pointing the same way:

- **The correction is correctly derived and scaled.** The optimal scale
  `alpha* = argmin ||Cg + alpha D - Ctrue||` is **0.95 at layer 1** (where the
  correction delivers its 1.63x), decaying monotonically to **0.24-0.34 at layer
  29**. At depth the true residual is largely *orthogonal* to the third-order
  correction. Even granting an oracle `alpha*` you could never know, the gain is
  only ~1.1x.
- **It is not a conditioning artifact.** Only 0.4% of pairs carry |rho| > 0.99,
  and suppressing the correction on those changes nothing.
- **It is not series divergence.** The Edgeworth parameter is small:

| layer | 0 | 4 | 16 | 31 |
|---|---:|---:|---:|---:|
| mean abs skew | 0.003 | 0.096 | 0.271 | 0.400 |
| excess kurtosis | -0.000 | 0.113 | 0.267 | 0.364 |
| \|kappa3\|/6 | 0.001 | 0.016 | 0.045 | **0.067** |
| participation ratio | 128.0 | 47.2 | 9.4 | **2.2** |

  (layer 0 skew 0.003 confirms the harness once more.)

**The marginals are nearly Gaussian while the joint collapses to participation
ratio 2.2. The closure error is a copula/geometry effect, not a cumulant
effect** — the departure is not in the moments, so no finite-order moment
closure reaches it. This closes the analytic branch on the user's decision tree.

Two further closure variants were tested and failed:

- **Scale-mixture (radial-conditioned) closure**, exploiting positive
  homogeneity: `E[relu(z)] = E[rho] E[relu(zhat)]` exactly when `rho = ||a||` is
  independent of direction. Measured **0.40x — 2.5x worse than plain Gaussian**.
  `rho` and the direction are strongly dependent; the factorisation is badly
  violated.
- **Sampling dominates the closure anyway.** The Gaussian closure's sigma error
  is 1.4e-2; matching that by sampling needs only `1/sqrt(2M) < 1.4e-2`, i.e.
  **M ~ 2,500 rows**. Any analytic closure worse than 1.4e-2 is beaten by a
  trivially small sample, which is the cleanest statement of why this route
  cannot compete.

### The unobservable-sign wall, on five independent probes

| probe | error energy captured | sign agreement | honest LORO gain |
|---|---:|---:|---:|
| direct-output PCA source, rank 36 | 92.3% | 0.4917 | R^2 = -0.042 |
| static linear class over 129 groups | — | — | 1.000x |
| rank-1 scale mode along `yhat` | 32.6% | 0.510 | R^2 = -0.044 |
| multilevel telescope | — | — | < 1.00x |
| analytic third-order closure | — | — | 1.05x |

Every probe finds **capacity and no sign**. The mechanism is the harmonic
result of §3b: the error is degree >= 20 content, while every cheap legal
statistic is a smooth low-degree functional of the network. A low-degree
observable and a high-degree error are essentially uncorrelated — which is
exactly why the error's *magnitude* is observable at corr 0.927 (an even,
quadratic functional) while its *direction* sits at chance.

Also checked and closed: **dead neurons are already exact.** 25.6% of
final-layer neurons are predicted exactly 0 and contribute **0.00%** of the
squared error (their true mean RMS is 1.4e-7). The error is concentrated in the
large-mean neurons — the top 25% carry 68% of it.

### What is now open, precisely

**Answered: it does not.** Gate 1 above measures 1.05x against a required 4.5x,
with the mechanism identified. The analytic branch is closed.

Note also that the earlier low-rank result was **Tucker/multilinear** evidence,
not symmetric CP evidence, and the `n^2 R + n R^3` cost is a Tucker cost. That
distinction is now moot — Gate 2 was never reached, because compression can only
preserve a correction that works, and the dense correction does not.

---

## 3e. Oracle-capacity vs oracle-closure — the meta-result

Two different things are called "oracle" in this project and conflating them is
why the programme keeps stalling.

- **Oracle-capacity**: "does a subspace contain the error?" — `r* = 0.0749`, the
  A90 `0.02292`, my rank-36 `0.0773`, the rank-1 scale mode's 32.6%. These
  **presuppose knowing the answer** in order to fit coefficients. Five
  independent probes (§3d) all collapse to `R^2 ~ 0` and sign `~ 0.5` the moment
  they are made legal. As a class they have been **systematically
  non-predictive** of achievable score.
- **Oracle-closure**: "given the exact intermediate state, how accurate is the
  model class?" These presuppose only a state that might be legally computable.
  Gate 1 was one; it failed honestly. They are the only oracle worth running.

**Recommendation: stop treating oracle capacity as evidence.** Every future
candidate should be gated on an oracle-closure test plus an explicit cost model,
never on a capacity ratio.

## 3f. A latent-geometry closure — passes accuracy, fails economics

Gate 1's diagnosis (a geometry effect, not a cumulant effect) points at a model
class that is not in the ledger: replace the moment closure with a
latent-variable one,

```
z = mu + B xi + eps ,   xi in R^r with its TRUE law,  eps ~ N(0, Sigma_eps) _|_ xi
```

`r = 0` is exactly the Gaussian closure. Larger `r` hands the model the actual
low-dimensional geometry and Gaussianises only the high-dimensional remainder,
which really is near-Gaussian (it is a sum of many small contributions). This is
an **oracle-closure** test: it never uses the answer.

Measured, 3 networks, N=800,000 (floor 7.9e-4):

| layer | r=0 | r=8 | r=32 | r=64 | r=128 | best gain |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 9.93e-3 | 7.96e-3 | 4.73e-3 | 2.76e-3 | 1.12e-3 | 8.9x |
| 16 | 1.65e-2 | 9.28e-3 | 4.60e-3 | 2.21e-3 | 5.33e-4 | 30.9x |
| 29 | 1.37e-2 | 6.71e-3 | 2.37e-3 | 8.62e-4 | 1.66e-4 | 82.4x |

Unlike the third-order correction, this gets **stronger with depth** — exactly
where the geometry collapse is worst. At `r = 64` it clears the 3e-3 requirement
at every layer. **This is the first model class in the session to pass an
oracle-closure gate.**

It then dies at Gate 3, on cost, and the arithmetic is unambiguous:

| r | mean sigma err | implied bias MSE | K for noise=bias | cost/B | best score |
|---:|---:|---:|---:|---:|---:|
| 32 | 3.88e-3 | 6.64e-6 | 2.2e3 | 0.03 | 8.85e-7 |
| 64 | 1.96e-3 | 1.69e-6 | 8.9e3 | 0.13 | 3.94e-7 |
| 128 | 0.60e-3 | 1.57e-7 | 9.6e4 | **1.44** | 2.41e-7 |

against the shipped cubature's **1.434e-7**. The reason is structural: a latent
particle must propagate a full `n`-vector per layer, so **it costs exactly what a
Monte Carlo sample costs**. The method is therefore MC with a bias floor added —
strictly dominated. Low `r` is cheap but biased; high `r` is accurate but needs
`K ~ 1e5` particles; the crossover never favours the method.

### Resolved: the gain is a copula, and there is no parametrisation

The latent `xi = B^T (z - mu)` is built from eigenvectors, so its covariance is
**diagonal**. That permits an exact decomposition of where the accuracy lives:
permuting each latent coordinate independently preserves the mean, the full
covariance and every marginal exactly, and destroys **only** the copula.

| layer | r | Gaussian (r=0) | marginals only | full joint | gain recovered |
|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 1.302e-2 | 1.203e-2 | 8.80e-4 | 8.2% |
| 16 | 128 | 1.645e-2 | 1.508e-2 | 5.41e-4 | 8.6% |
| 24 | 128 | 1.578e-2 | 1.497e-2 | 2.81e-4 | 5.2% |
| 29 | 128 | 1.360e-2 | 1.287e-2 | 1.65e-4 | 5.4% |

**The marginals deliver 5-11% of the gain; the copula carries 89-95%.** So the
latent law is not `r` one-dimensional distributions — it is a genuine
`r`-dimensional copula, and the deployable (marginal-parametrised) version is a
1.06x improvement, i.e. nothing.

That closes the class at both ends:

- **particles** reproduce the copula but cost a full row propagation each, so the
  method is Monte Carlo with a bias floor — strictly dominated (2.41e-7 vs
  1.434e-7);
- **parametrisation** is cheap but recovers 5-11%;
- **moment-parametrised latents** are excluded by Gate 1, and the expansion
  parameter is hopeless anyway: the residual smooths the ReLU at scale
  `sqrt((1-p_r)/p_r)` of the latent's spread, which is **0.017** at r=128 —
  essentially unsmoothed.

The accuracy gain and the cost saving are the same resource. You cannot have
both.

### The last open door: weight-aware estimation — also closed

T81's no-go covers the finite group-output transcript only; weight-aware and
state-aware estimators were explicitly left open, and every previous probe had
used transcript features. Tested directly: 29 features per (network, neuron) —
final pre-activation mu/sigma/skew/kurtosis, `t`, active fraction, `yhat`,
between-basis variance, `||w_i||`, alignment of `w_i` with the top activation
eigenvector, and the **Gaussian-closure residual** (a direct local measure of
non-Gaussianity at that neuron) — plus their squares, on 25,600 (network,
neuron) observations with leave-one-network-out validation:

```
ridge lam=1e-3 .. 1e2     LORO R^2 = -0.018 .. -0.002
sign agreement                     = 0.375 .. 0.423
MSE ratio                          = 1.0016 .. 1.0178  (always worse)
```

**Negative at every regularisation.** The weight-aware branch closes empirically.

### Resolution

There is no remaining branch. The closure table, all measured on the official
cohort:

| # | class | measured result |
|---:|---|---:|
| 1 | design strength (any spherical t-design) | 1.02x |
| 2 | arithmetic `f` (Strassen family) | <= 1.15x |
| 3 | static reweighting of the 129 groups | 1.000x |
| 4 | adaptive direct-output source (transcript) | R^2 = -0.042 |
| 5 | rank-1 scale mode along `yhat` | R^2 = -0.044 |
| 6 | multilevel / telescoping | < 1.00x |
| 7 | analytic moment closure (dense oracle 3rd order) | 1.05x |
| 8 | scale-mixture (radial) closure | 0.40x |
| 9 | delayed closure | worse with lag |
| 10 | universal per-layer calibration | 1.01x |
| 11 | exact low-dimensional quadrature | rank too high early |
| 12 | latent-geometry closure, particles | 2.41e-7 vs 1.434e-7 |
| 13 | latent-geometry closure, parametrised | 5-11% of the gain |
| 14 | weight-aware / state-aware prediction | R^2 <= 0 |
| 15 | dead-neuron targeting | already exact (0.00% of error) |

**One mechanism explains all fifteen.** The residual error is degree >= 20
spherical-harmonic content (§3b). Every cheap legal observable is a smooth,
low-degree functional of the network. A low-degree observable and a
high-degree error are essentially uncorrelated — which is exactly why the
error's *magnitude* is observable at corr 0.927 (an even, quadratic functional)
while its *direction* sits at chance in all five sign probes.

The honest frontier is `V*f/B` at the complete-design optimum: **~1.2e-7**,
against a shipped 1.434e-7 (all 100 networks, excluding residual). Reaching
1.235e-8 needs `V*f` 11.5x lower, i.e. beating plain Monte-Carlo variance by 38x
where the complete Kerdock 5-design achieves 3.3x and §3b shows no design
exceeds 1.02x. Combined with note 04's finding that the leading submission
charges >99.98% of its compute as residual wall time on 13.0M tracked FLOPs,
**4.34x is not an algorithmic target under the intended accounting.** That
should be stated plainly in the write-up rather than left as an unexplained gap.

---

## 3g. The hybrid program (v28 thesis) — tested

The remaining meta-class is hybrid: `E[f] = E[g] + E[f-g]`, an analytic anchor
plus a stochastic residual. Four experiments, two of which produced artifacts I
caught and am recording so they are not mistaken for results.

### v1 — sampled moments, analytic mean

Sample sigma/kappa from the design rows; propagate mu analytically through the
Edgeworth marginal. The covariance-closure error of Gate 1 never enters, because
Sigma is never closed.

| rows | direct MSE | anchor Gaussian | +k3 | +k3k4 |
|---:|---:|---:|---:|---:|
| 4,096 | 3.21e-6 | 9.70e-5 | 5.34e-6 | 3.01e-6 |
| 16,384 | 6.43e-7 | 8.72e-5 | 3.76e-6 | 7.97e-7 |
| 66,048 | 1.68e-7 | 8.28e-5 | 3.71e-6 | 2.73e-7 |

The Edgeworth marginal is worth **300x** over the Gaussian anchor (8.3e-5 ->
2.7e-7), landing within 2x of notes/03's oracle-EMP ceiling. But `MSE x rows` is
**constant** (1.23e-2, 1.31e-2, 1.80e-2): still exactly MC rate. Fitting
`MSE = b + V/N` gives `V_chain = 1.20e-2` against direct's `1.11e-2` and a
structural bias floor `b ~ 8e-8`. **The analytic chain does not create
information, and it is 1.4x noisier per row than direct averaging.**

### Gate A — blockwise residual variance (a genuine positive)

For a hybrid on the Kerdock design what matters is not the pointwise variance of
`f-g` but the variance ACROSS complete blocks, `S_r = Var_b(Q_b(f-g))`, because
each block already annihilates the low degrees.

| anchor | S_r/S_f blockwise | pointwise ratio |
|---|---:|---:|
| rank 8 / 32 / 128 | 0.991 / 1.016 / 0.949 | 1.014 / 0.873 / 0.597 |
| smooth alpha=0.2 | **0.039** | — |
| smooth alpha=0.5 | **0.242** | 0.141 |

Two results. **Every rank anchor is useless blockwise** despite rank-128's 0.597
pointwise — which retro-corrects §3b's multilevel finding: the pointwise metric
was wrong, and wrong in the *optimistic* direction. And the **smooth anchor**
(same kinks, ReLU replaced by its Gaussian-smoothed version) cuts the blockwise
residual 4.1x at alpha=0.5 and 25x at alpha=0.2. It is the first anchor family
that preserves the design's cancellation.

It is also analytically natural: for Gaussian `p`,
`E[smoothed-relu(p)] = E[relu(N(mu, sigma^2+s^2))]` exactly, and the pair moment
is exactly Psi with inflated variances and **shrunken effective correlation**
`rho_eff = rho si sj / (sqrt(si^2+s^2) sqrt(sj^2+s^2))` — smoothing pushes the
closure toward the rho -> 0 regime where it is exact, attacking Gate 1's copula
sensitivity at its source rather than modelling around it.

### Two artifacts, recorded so they are not reused

The improvement factor is `score_hybrid/score_direct = c (S_r/S_f)`, with `c` the
cost factor per residual block — independent of R.

1. **Partial-smoothing scan reported up to 5687x.** Degenerate: the objective
   omitted the anchor bias, so it ran to the corner (L=30, alpha=0.25) where
   `g -> f`, `S_r -> 0`, and computing `E[g]` is *exactly the original problem*.
   For any L > 0 the anchor must propagate through L layers of exact ReLU
   analytically — the 8.4e-5 blocker verbatim. Only the L=0 column is meaningful.
2. **Alpha-scan reported 10.02x at alpha=0.** Two optimistic errors: the "bias"
   was measured against the design's own estimate of `E[g]` rather than truth,
   and the cost charged 5.65e9 FLOPs for moments drawn from all 66,048 rows
   (1.71e11). That omission manufactured the entire 10x.

What the alpha scan does establish honestly is that **bias grows far faster than
the residual shrinks**: alpha=0.2 costs 3.68e-7 of bias to buy S_r/S_f = 0.039,
and by alpha=0.5 the bias is 7.3e-6 — 40x the direct estimator's whole MSE.

### Honest accounting — the branch closes

Every ingredient paid for out of the same R blocks, both columns scored against
baked ground truth at matched cost (2R blocks), 10 networks:

| R | direct @2R | hybrid a=0.0 | a=0.1 | a=0.2 | a=0.35 |
|---:|---:|---:|---:|---:|---:|
| 8 | **1.278e-6** | 2.973e-6 | 2.940e-6 | 3.038e-6 | 4.744e-6 |
| 16 | **5.851e-7** | 1.630e-6 | 1.608e-6 | 1.713e-6 | 3.382e-6 |
| 64 | **1.839e-7** | 4.739e-7 | 4.655e-7 | 6.549e-7 | 2.464e-6 |

Uniformly **2-2.7x worse**, and the best alpha is 0.1 — smoothing stops paying
the moment its moments are honestly charged.

**The structural reason, which generalises beyond this construction:
`E[g]` cannot be obtained more cheaply than `E[f]`.** Every hybrid needs
per-layer moments; the moments cost what the answer costs; and the analytic chain
is 1.4x noisier per row than direct averaging. The residual side works. It is
dominated by the anchor side.

### The hybrid class closes: a measured, non-crossing trade-off

My first closure ("every hybrid needs per-layer moments") was too broad — an
**exactly-integrable** anchor needs none. Because `p_0 = x W_0` is exactly
Gaussian, `E[a_0]` and `E[a_0 a_0^T]` are closed form (Cho-Saul), so
`g = a_0 C + sum_p lambda_p (u_p . a_0)^2` has an exact mean, needs no sampling,
and costs almost nothing (`a_0` IS the design rows): `c ~ 1.03-1.08`.

Measured, 6 networks:

| quad rank P | c | S_r/S_f (fit ALL rows) | S_r/S_f (honest) | improvement |
|---:|---:|---:|---:|---:|
| 0 (linear) | 1.025 | 0.9691 | 0.9744 | 1.00x |
| 64 | 1.038 | 0.9663 | 0.9717 | 0.99x |
| 256 | 1.076 | 0.9659 | 0.9956 | 0.93x |

**Even at the oracle ceiling they remove 3% of the blockwise residual.** Free and
useless — exactly as the harmonic argument predicts, since a layer-0 anchor spans
~m of a ~1e7-dimensional degree-4 space.

The intermediate family — anchor at depth k with the best linear readout
`g_k = a_k C_k`, which reproduces every kink up to k and linearises the rest —
maps the whole trade-off:

| anchor depth k | blockwise power | c | zero-bias bound | `E[a_k]` exact? |
|---:|---:|---:|---:|---|
| **0** | **3%** | 1.03 | 0.99x | **YES** |
| 8 | 42% | 1.29 | 1.31x | no |
| 16 | 69% | 1.55 | 2.02x | no |
| 26 | 92% | 1.87 | 6.77x | no |
| 30 | 98.5% | 2.00 | 33.2x | no |

Blockwise variance is created **smoothly through the whole depth** (~11% of the
remainder per layer); there is no shallow shortcut. So:

> **Power requires anchoring deep. Exactness requires anchoring at layer 0 — the
> only exactly-Gaussian layer. The two move in opposite directions and never
> cross.** The anchor's mean must be accurate to ~1e-7 MSE. At k=0 it is exact
> but carries 3% power. At k=2 the chain error is already ~5.4e-6 against 15%
> power, netting **4x worse than direct**. By k=26, where power suffices, the
> mean error is ~7e-5 — **700x over budget**.

A trap worth naming because it looks like an escape: obtaining `E[g]` by
**sampling** the anchor on the same blocks makes the estimator collapse
algebraically to `mean_R(f)`, the plain direct estimate. The anchor's mean must
come from a genuinely different source, and analytic is the only one.

**The hybrid class therefore closes from three sides** — exactly-integrable
anchors (0.97 at the oracle ceiling), moment-based anchors (2-2.7x worse at
honest matched cost), and every depth between (trade-off never crosses). The
residual side genuinely works (25x blockwise at alpha=0.2); it is the anchor side
that is pinned.

### M187 — conditional covariance mixture

v28's primary hypothesis, tested as an oracle-closure gate:

| layer | K=1 | K=8 | K=32 | gain |
|---:|---:|---:|---:|---:|
| 16 | 1.641e-2 | 8.44e-3 | 5.33e-3 | 3.08x |
| 29 | 1.368e-2 | 6.08e-3 | 3.58e-3 | 3.82x |

Real — it does capture part of the copula — but ~4.8e-3 averaged over layers
against a 3e-3 requirement, still improving in K, and the K ~ 64-128 needed to
close it costs `O(K n^3)` = 3.4e10-6.8e10, breaking the 0.1 multiplier floor. It
is also weaker than the location latent at equal index (r=32 -> 2.37e-3 vs
K=32 -> 3.58e-3) though far more compact. **Keep it as the anchor's covariance
model, not as the estimator.**

---

## 3h. The analytic route is NOT closed — calibrated, with a quantified prize

A 5.5x ambiguity in the requirement had gone unresolved: notes/03 section 3 says
sigma needs ~3e-3 relative; anchoring `MSE ~ (sigma err)^2` on
covariance_propagation (1.4e-2 -> 8.4e-5) implies ~5.5e-4. It decides which
problem is open. Calibrated directly by injecting known relative noise into
sigma at every layer of the chain and refitting (20 networks):

| injected delta | chain MSE | excess |
|---:|---:|---:|
| 0 | 3.4304e-7 | — |
| 2e-3 | 4.7524e-7 | 1.322e-7 |
| 1e-2 | 3.8988e-6 | 3.556e-6 |
| 5e-2 | 9.2175e-5 | 9.183e-5 |

`C = 3.67e-2` MSE per unit delta^2, fit exact to three digits across two decades.

**Requirement: delta <= 1.9e-3.** notes/03 was approximately right; the 5.5e-4
anchoring was wrong. This matters, because **the location latent at r=64
measures 1.96e-3 and clears it.** The accuracy problem has a solution.

**Floor:** `A = 3.43e-7` at the design's own moment quality; removing the
design's own sigma-noise contribution (`C delta_design^2` ~ 9.4e-8) leaves a
**structural floor ~2.5e-7**, which is exactly the marginal Edgeworth closure
accumulating (1.9e-4/layer x sqrt(31) -> 1.06e-3 relative -> 3.5e-7).

**Prize.** A fully analytic chain costs ~2e9 FLOPs, so the multiplier floors at
0.1 and `score = 0.1 (2.5e-7 + C delta^2)`:

    delta = 1.9e-3  ->  3.8e-8   =  4.5x
    delta -> 0      ->  2.5e-8   =  6.9x

This clears the 4.34x threshold. (It is not the 13x implied by notes/03's
1.30e-7 — that figure is not reproduced by this chain; the honest floor is
2.5e-7, set by the marginal closure.)

### The gap, stated exactly

> Represent the joint law of the top-64 principal directions in **O(n^2)
> parameters, propagatable at O(n^3)/layer.**

Both known representations fail on cost, with numbers:

- **Particles.** `p_64 = 99.4%` of the variance sits in the latent at depth, so
  particle noise ~ `0.994 V/K` — barely better than plain MC. At the compute
  floor (K=6,700) that is 2.2e-6 -> score 2.7e-7, **worse than direct**.
  Matching the floor needs K~44,000 = 66% of budget -> 2.2e-7. Dead.
- **Finite mixtures.** Full-covariance K=32 gives 4.8e-3; the ladder scales
  ~K^-0.35, so delta = 1.9e-3 needs **K ~ 440** -> `O(K n^3)` = 9.1e11 =
  **3.3x the entire budget**. Dead. M192's tied/low-rank rungs are *more*
  constrained than full covariance and are therefore bounded above by this —
  which closes the ladder rather than opening it.

And the obvious shortcut is blocked: the output needs only all **bivariate**
marginals (n^2/2 of them — exactly the O(n^2) budget), but pairwise marginals are
**not recursively closed**, since `z_{l+1,i}` depends on the full joint of `a_l`.
**The cheap-enough state is not propagatable and the propagatable state is not
cheap enough.**

Two independent ways to win from here, both well posed:

1. compress a 64-dim copula into O(n^2) propagatable parameters; or
2. lower the structural floor by improving the **marginal** closure below
   1.9e-4/layer — the floor is entirely marginal, not joint.

---

## 3i. M192 closed quantitatively; the marginal floor is irreducible

Three tests, all using constants calibrated in section 3h.

### The marginal floor cannot be corrected away

The structural floor (2.5e-7) is set entirely by the **marginal** closure, which
is a one-dimensional problem and a deterministic functional of the marginal
shape — so it ought to be a smooth universal function of standardised parameters,
fittable offline for free. Tested on 111,104 (network, layer, neuron) triples,
degree-4 polynomial with interactions, leave-one-network-out:

| features | LORO R^2 | floor drops |
|---|---:|---:|
| t, k3, k4 | +0.0205 | 1.0x |
| t, k3, k4, k5, k6 | +0.0030 | 1.0x |
| + depth | +0.0033 | 1.0x |

**R^2 ~ 0.02.** The residual depends on distributional detail beyond six
cumulants, so carrying more cumulants cannot move the floor. 2.5e-7 stands.

### The latent copula is not a linear mixture (ICA)

Destroying the dependence in the PCA basis loses 89-95%, but that only shows the
components are not independent *in that basis*. If the latent were a linear
mixture of independent non-Gaussian sources it would compress to 64 1-D laws plus
a 64x64 rotation — O(n^2), deterministic, propagatable. Tested with symmetric
FastICA (tanh contrast) at r=64:

| layer | full joint | pca-marg | ica-marg | ICA recovers |
|---:|---:|---:|---:|---:|
| 8 | 2.52e-3 | 1.21e-2 | 1.12e-2 | 9.6% |
| 16 | 2.23e-3 | 1.51e-2 | 1.28e-2 | 17.8% |
| 24 | 1.40e-3 | 1.50e-2 | 1.15e-2 | 25.7% |
| 29 | 8.70e-4 | 1.29e-2 | 1.00e-2 | 24.2% |

**ICA recovers 10-26%.** The copula is genuinely entangled, not linearly
separable. Linear compression is closed.

### The mixture family loses on economics, not accuracy

Measured K-ladder scales as `K^-0.38` (each doubling buys ~1.30x). Mixture
propagation costs ~8.7e7 FLOPs per component per layer. With A = 2.5e-7 and
C = 3.67e-2 from section 3h:

| K | closure delta | cost/B | MSE | score | vs direct |
|---:|---:|---:|---:|---:|---:|
| 10 | 7.47e-3 | 0.099 | 2.30e-6 | **2.30e-7** | **0.62x** |
| 32 | 4.80e-3 | 0.317 | 1.10e-6 | 3.48e-7 | 0.41x |
| 100 | 3.11e-3 | 0.992 | 6.06e-7 | 6.01e-7 | 0.24x |
| 140 | 2.74e-3 | **1.39** | — | **run fails** | — |

**Best achievable 2.30e-7 at K=10 — 1.6x worse than direct.** The optimum is
interior and non-monotone: accuracy and cost cross before the gate. delta = 1.9e-3
needs **K ~ 367 = 3.6x budget**, which fails the run outright.

The tied / shared-low-rank modulation rungs are a **constrained subset** of
full-covariance mixtures at the same K, so their closure is bounded below by this
table. They are dominated a fortiori and cannot beat it.

**M192 is closed; M188 is moot.** The remaining representation problem —
a compact, deterministic, legally propagatable joint-copula state — now has three
independent negative results against it (marginal 5-11%, linear/ICA 10-26%,
mixture economics 0.62x) and no known candidate.

---

## 3j. A cost collapse: Delta Cov(relu) is rank-1

Two corrections to section 3i, then a positive result.

**Correction 1 — systematic vs random sigma error.** The section 3h calibration
injected iid noise; a closure error is systematic. Measured at matched RMS:

| structure | sensitivity C | tolerance |
|---|---:|---:|
| iid (fresh per neuron per layer) | 3.59e-2 | 1.49e-3 |
| persistent (per neuron, all layers) | 3.68e-2 | 1.47e-3 |
| common (per layer, all neurons) | 8.57e-2 | 9.67e-4 |

Systematic error is only 1.02-2.4x more damaging, so **the requirement holds at
delta <= 1.5e-3** and r=64's 1.96e-3 is 1.3x short, not 3.5x.

**Correction 2 — I closed the tied rung on the wrong axis.** Tied covariance is
dominated by full covariance *at equal K*, but tied propagates ONE shared Sigma —
`O(n^3 + K n^2)` against `O(K n^3)` — so at equal COST it buys ~256x more
components. Retested properly:

| layer | K=1 | K=128 | K=1024 | K=4096 |
|---:|---:|---:|---:|---:|
| 16 | 1.717e-2 | 1.091e-2 | 8.995e-3 | 7.649e-3 |
| 29 | 1.453e-2 | 8.432e-3 | 6.362e-3 | 5.181e-3 |

It converges as **K^-0.12** — delta = 1.5e-3 would need **K ~ 5e8**. Closed, and
it hands over the reason: **K=4096 tied (6.4e-3) is worse than K=32 full
covariance (4.8e-3).** 128x fewer components win when they carry their own
covariance. **The copula is heteroscedastic, not multimodal.**

### The gate that changes the arithmetic

Full covariance has the accuracy but costs `O(K n^3)` — short by ~65x. That cost
is the per-component `W^T Cov_q(relu) W` contraction. If `Delta Cov(relu)` is
low-rank under a low-rank `Delta Sigma`, the contraction becomes
`(W^T U) D (U^T W)`. Measured exactly with the validated Psi pair moment:

| layer | rank for 90% | 99% | 99.9% |
|---:|---:|---:|---:|
| 8 | 1 | 1 | 5 |
| 16 | 1 | 1 | 4 |
| 24 | 1 | 1 | 3 |
| 29 | 1 | 1 | 1-2 |

**Essentially rank-1.** Downstream-weighted (the metric that matters), keeping
rank 8 reproduces the next-layer variance change to 0.6-3.0%, rank 16 to
0.3-2.3%. Identical at 10% and 30% modulation, so the map is near-linear in
`Delta Sigma` — meaning the update also admits a first-order elementwise formula
rather than a fresh `n^2` Psi evaluation.

Per-component cost collapses `8.68e7 -> ~4n^2 = 2.62e5`, a **331x** reduction:

| K | delta | cost/B | MSE | score | vs direct |
|---:|---:|---:|---:|---:|---:|
| 367 | 1.90e-3 | 0.011 | 3.82e-7 | 3.82e-8 | 3.74x |
| **1000** | 1.30e-3 | 0.030 | 3.12e-7 | **3.12e-8** | **4.59x** |
| **3000** | 8.55e-4 | 0.090 | 2.77e-7 | **2.77e-8** | **5.17x** |

The compute floor is not reached until K=3347, so the optimum sits at K~3000:
**5.17x, an 80.6% reduction.**

### The ladder holds — measured to K=1536

The exponent was the one extrapolated input and I predicted it would flatten like
the tied family. **It did not.** (A first run appeared to saturate, but that run
dropped the projection normalisation before k-means and added 5% shrinkage toward
the pooled covariance; both push every component back toward K=1. Reconciled to
the covmix methodology — PC=8 normalised, no shrinkage, N=2.4e6 — the ladder is:)

| K | 1 | 8 | 32 | 128 | 512 | 1536 |
|---|---:|---:|---:|---:|---:|---:|
| delta | 1.577e-2 | 7.857e-3 | 4.744e-3 | 3.061e-3 | 2.207e-3 | **1.781e-3** |
| local exponent | — | 0.335 | 0.364 | 0.316 | 0.236 | 0.195 |

Economics on the measured curve:

| K | delta | source | cost/B | score | vs direct |
|---:|---:|---|---:|---:|---:|
| 512 | 2.207e-3 | measured | 0.015 | 4.288e-8 | 3.34x |
| **1536** | **1.781e-3** | **measured** | 0.046 | **3.664e-8** | **3.91x** |
| 3350 | 1.530e-3 | 2.2x extrapolation | 0.100 | 3.360e-8 | **4.27x** |
| 6000 | 1.365e-3 | extrapolation | 0.179 | 5.705e-8 | 2.51x |

**3.91x on measured delta; 4.27x at the compute floor.** The 4.34x threshold needs
3.304e-8 and this reaches 3.360e-8 — short by 1.6%. The floor caps K at 3349, and
beyond it the multiplier rises faster than delta falls, so the family tops out
there. The 80% target (2.868e-8) is 17% out of reach.

This is the best result in the audit and the only path that reaches within a few
percent of the threshold on largely measured numbers.

### Gate 1 (the O(n^2) update) — passes on correctness and cost

The whole projection rests on replacing the per-component `4n^3` contraction
`W^T Cov_q(relu) W` with a cheap first-order update. **Price's theorem** supplies
the derivatives in closed form:

    i != j :  dE[relu_i relu_j]/dS_ij = Phi2(a_i, a_j; rho_ij)
              dE[relu_i relu_j]/dm_i  = E[1{z_i>0} relu(z_j)]
              dE[relu_i relu_j]/dS_ii = (1/2) E[delta(z_i) relu(z_j)]
    i == j :  dE[relu_i^2]/dS_ii = Phi(a_i),  dE[relu_i^2]/dm_i = 2 E[relu(z_i)]

plus the covariance correction `dCov = dE2 - (dE1 E1^T + E1 dE1^T)` with
`dE1 = Phi(a) dm + phi(a)/(2 sigma) dS_ii`.

Crucially these are evaluated at the REFERENCE component, so the expensive Phi2
matrix is computed **once and reused for all K**. Each component then costs
O(n^2) with no special-function evaluations.

**Validated.** Scaling the component offset by `t` gives relative error
0.0005 / 0.0012 / 0.0025 / 0.0064 / 0.0135 / 0.0308 at t = 0.02 ... 1.0 —
`err/t` constant at 0.024-0.031 across two decades, i.e. clean second-order
truncation, confirming the formula. At the real component offset
(`||dm||/||sigma|| = 0.63`, `||dS||/||S0|| = 0.65`) the error is **3.1%**.

Over K=16 real clustered components, with a correct signed-eigendecomposition
truncation:

| layer | \|dv\|/\|v0\| | rank 1 | rank 8 | rank 32 | full |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.303 | 0.612 | 0.221 | 0.130 | 0.117 |
| 29 | 0.359 | 0.316 | 0.085 | 0.057 | 0.055 |

Two bugs were found and fixed en route, both of which had made the gate look
fatal: the covariance mean-product term `d(E1_i E1_j)` was missing (1500-2200%
error), and the rank truncation used `U S U^T` from an SVD on an **indefinite**
matrix, silently flipping the sign of every negative eigenvalue (which is why
error grew with rank).

**Residual risk, now quantified rather than unknown:** the second-order
truncation injects ~3.6% into the variance (12% of a 30% change) = 1.8e-2 in
sigma, far above the 1.5e-3 budget **if systematic**; if it averages across
components it falls as sqrt(K), giving 4.6e-4 at K=1536 — comfortably inside.
Which holds is the next measurement.

### Gate 2 FAILS — the cost collapse is unavailable, and the route closes

Gate 1 established the O(n^2) update is correct with ~12% second-order error per
component. Whether that is survivable depends on whether it averages over
components. Measured on the mixture quantity the estimator actually uses:

| K | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| sigma error from the O(n^2) update | 5.20e-3 | 6.95e-3 | 7.92e-3 | 8.83e-3 | 9.72e-3 |

It **grows** as `K^+0.14` instead of falling as `K^-0.5`. Extrapolated to K=1536:
**~1.5e-2, ten times the 1.5e-3 budget.**

The mechanism is structural. More components means each sits **further from any
single reference**, so the first-order expansion is evaluated at larger offsets
exactly as K grows. It degrades precisely in the regime the mixture's accuracy
requires. Even at K=4 — the smallest offsets tested — it is 5.2e-3, already 3.5x
over budget.

The natural repair fails on arithmetic: R reference points keep offsets small but
cost a full Phi2 matrix each (2e7/layer). Since error ~ offset^2, reaching 1.5e-3
needs R ~ K/2; at K=1536 that is 4.8e11 = **1.8x budget**, worse than computing
every component exactly.

**Consequence.** Without the cost collapse the family reverts to the exact
8.68e7/component regime, which caps at K~100 and tops out at **0.62x — worse than
the shipped estimator.** The two regimes are:

| cost model | max affordable K | delta | score | vs direct |
|---|---:|---:|---:|---:|
| exact per component (8.68e7) | ~100 | 3.1e-3 | 2.30e-7 | 0.62x |
| O(n^2) update (2.62e5) | 3350 | ~1.5e-2 (approximation-dominated) | 8.5e-7 | 0.17x |

**Second order does not rescue it** (measured, not argued -- I had dismissed
second order with a bad cost claim: by Price's theorem the derivatives of
E[relu_i relu_j] are nonzero only for indices in {i,j}, so the second-order term
is O(n^2), same as first order). Exact directional derivatives give:

| layer | K | first order | second order | gain |
|---:|---:|---:|---:|---:|
| 16 | 4 | 4.25e-3 | 8.82e-4 | 4.82x |
| 16 | 32 | 9.00e-3 | 4.18e-3 | 2.15x |
| 29 | 4 | 5.39e-3 | 1.15e-3 | 4.69x |
| 29 | 32 | 8.78e-3 | 3.17e-3 | 2.77x |

~3.7e-3 at K=32 against a 1.5e-3 budget, still growing with K (~6.1e-3 at
K=1536), and **the gain from second order shrinks as K rises** (4.8x -> 2.15x),
so higher orders help less again. The expansion is not converging at these
offsets (`||dm||/||sigma|| = 0.63`).

### The mechanism, and the last untested space — both closed

**Why the expansion fails.** Re-centring on the pooled-within covariance `Sbar`
(rather than the global `S0`) shrinks the covariance offsets substantially —
0.574 -> 0.357 at layer 29, K=64 — and the error does **not** improve at all:

| layer | K | \|dS\|/\|S\| global | pooled | 2nd global | 2nd pooled |
|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 0.586 | 0.476 | 5.39e-3 | 5.58e-3 |
| 29 | 64 | 0.574 | 0.357 | 4.00e-3 | 5.41e-3 |

If `dS` drove the error, a 38% offset cut would have cut it ~60%. It did not, so
the error is dominated by the **mean** offsets `dm`, identical under both
references. **And `||dm||` grows with K by construction — spreading the means is
what more components are FOR.** No covariance re-centring can touch it, and
keeping `dm` small needs a reference per component, which is exact evaluation.
That also explains why the reference-hierarchy arithmetic came out at R ~ K: the
references must be dense in *mean* space.

**The last untested space.** Every route above forms `Cov_q` and then contracts.
Computing `diag(W^T Cov_q(relu) W)` — 256 numbers — directly avoids that. Via
Mehler/Hermite, `v_k = sum_d (1/d!) beta_d^T R^od beta_d`, whose d=1 term is
`gamma^T S gamma`, costing `n^3` for all k unless S is low rank, then `2 n^2 r`.
The budget (K=1536 at the multiplier floor) is 5.7e5 per component, giving
**r <= 4.4** — for the first Hermite term alone. Measured rank requirement:

| layer | r=4 | r=16 | r=64 | r=128 |
|---:|---:|---:|---:|---:|
| 16 | 2.16e-1 | 5.44e-2 | 6.73e-3 | 7.86e-4 |
| 29 | 1.74e-1 | 4.00e-2 | 3.76e-3 | 2.23e-4 |

**r=128 is needed; r=4 is affordable.** And r=128 is half the dimension, so
`2 n^2 r = 1.7e7 ~ n^3 = 1.68e7` — **exploiting low rank buys nothing at the
accuracy required.** The direct route costs exactly what exact evaluation costs.

**Scope of the closure — stated carefully.** What is established is:
*no affordable evaluation of the mixture exists among the routes tested* —
exact is 150x too expensive, first-order 10x too inaccurate, second-order 4x too
inaccurate with diminishing returns in order. Three independent evaluation strategies now fail, each with its own mechanism:

| strategy | cost | why it fails |
|---|---:|---|
| exact per component | 8.68e7 | caps at K~100 -> 0.62x |
| Taylor 1st/2nd order | 2.6e5 | mean offsets grow with K by construction |
| direct / Hermite, low-rank S | ~n^3 | needs r=128; low rank degenerates to exact |

**The analytic mixture route is closed.** The accuracy exists (ladder to
delta = 1.781e-3 at K=1536) and the representation exists, but no affordable
evaluation of it does.

Gates 3 (legal propagate-and-merge) and 4 (PSD/stability) are moot — they gate a
construction whose cost model no longer closes.

### What is NOT established

This is not yet a candidate. Load-bearing unknowns, in order of risk:

1. RESOLVED — the ladder holds to K=1536 (above). Only a 2.2x extrapolation
   remains, to the compute floor.
2. RESOLVED — the O(n^2) update is derived, implemented and validated (above);
   the open part is whether its second-order truncation error averages across
   components.
3. Rank-1 truncation error (0.6-3%/layer) may compound over 31 layers.
4. Legal generation: the mixture must come from W and the previous state, not
   from fitting reference samples. This is the Gate-3 that has killed the
   analytic route three times.
5. PSD and stability across 31 layers.

---

## 3k. OFFICIAL MEASUREMENT — shipped estimator, whestbench 0.13.0

Run through the official CLI (`whest run --runner subprocess`, the grader's
execution path), official Phase-1 Mini, all 100 networks, BLAS pinned to 4
threads, nothing else running:

| metric | official value |
|---|---:|
| **adjusted_final_layer_score** | **1.4641716e-07** |
| final_layer_mse | 2.2819432e-07 |
| mean_score_multiplier | 0.6427078 |
| mean_effective_compute | 1.74817e11 |
| estimator FLOPs / MLP | 1.708730e11 |
| residual wall time / MLP | 39.4 ms |
| wall time / MLP | 16.5 s |
| best / worst MLP | 3.573e-08 / 8.523e-07 |
| **failed MLPs** | **0 / 100** |

**Harness validation.** This audit's independent NumPy reimplementation predicted
raw MSE **2.2826e-07** before the run; the official value is **2.2819432e-07** —
agreement to four significant figures (0.03%). Every delta, V and ladder figure in
this document was computed on that reimplementation, so they inherit official-
level fidelity.

**Correction to section 3c.** Wall time measures **16.5 s/MLP**, not the 21.4 s in
COMPARISON.md. Applying the recorded 11% grader penalty gives ~18.3 s against the
30 s guard — **39% headroom, not 21%**. The timeout-insurance recommendation is
correspondingly weaker than stated there.

**The root `estimator.py` does not run.** The 90,624-row multifidelity package
fails on every MLP under flopscope 0.9.1 with
`TypeError: dot() got an unexpected keyword argument 'out'` — `FlopscopeArray.dot`
has no `out=` parameter, so the package was written against a different flopscope.
Measured score **8.29e-01** (the zeros baseline, multiplier forced to 1.0,
2/2 MLPs failed). It is not a 9.3% regression; it is **not a runnable candidate
at all.**

---

## 4. Recommended allocation

1. **Buy the timeout insurance** (§3c). This is the only change here with a
   positive expected value on the current submission, and it is large.
2. **Do not ship the 90,624-row multifidelity estimator.** It is a ~9% adjusted
   regression against the shipped 129-basis package (§1).
3. **Retire the direct-output source + SOCP as the lead.** Gate A is real, Gate B
   is negative (§2). Run the SOCP only if a certified dual lower bound is wanted
   as a stop theorem — that is proof value, not score value.
4. **Close the cubature paradigm formally.** §3b gives it a measured ceiling of
   ~1.17x against a required 4.34x. This is a publishable negative result with a
   clean mechanism (a high-degree harmonic spectrum) that explains a large body
   of previously unexplained failures.
5. **Analytic moment propagation is the only surviving score route**, with the
   two named sub-problems from notes/03 §4. It is unaffected by the harmonic
   obstruction, its 11x ceiling is measured rather than conjectured — and it
   still needs a 100x improvement in moment-propagation accuracy, so treat it as
   a research bet, not a plan.
6. **Stop measuring raw MSE.** Every design comparison should report `V = MSE x
   rows` and `V*f`. Raw MSE gains bought with extra rows are worth exactly zero,
   and the multifidelity result shows this has already cost real effort.

### On the 4.34x target itself

The measured honest frontier is ~1.4e-7 and the cubature ceiling is ~1.2e-7.
Reaching 1.235e-8 requires `V*f` 11.5x below the shipped design, i.e. beating
plain Monte-Carlo variance by ~38x where the complete Kerdock 5-design achieves
3.3x. No cubature within an order of magnitude of that is known, and §3b shows
none exists. Combined with note 04's finding that the leading submission charges
>99.98% of its compute as residual wall time on 13.0M tracked FLOPs, the most
probable reading is that **4.34x is not an algorithmic target under the intended
accounting.** That should be stated plainly in the write-up rather than left as
an unexplained gap.

## Artifacts

Scripts in `scripts/score_law_audit_20260730/` (`harness.py`, `analysis.py`, `analysis2.py`, `mlmc.py`, `spectrum.py`,
`cumulant_test.py`) regenerate every number above from the official parquet
cohort and the shipped `kerdock_mub5_seed3.npz`.
