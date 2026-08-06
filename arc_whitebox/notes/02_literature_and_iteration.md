# Literature pass + how the measurements changed the plan

## 1. What the measurements said (before reading anything)

### Rank collapse is real and severe

`scripts/diagnose_structure.py`, 256×32, 300k samples:

| layer | eff. rank of Cov(a_l) | top eigval share | RMS skew of h | excess kurt of h | frac. Var linear in x |
|---|---|---|---|---|---|
| 1 | 165.3 | 1.2% | 0.004 | 0.000 | 0.735 |
| 8 | 28.3 | 11.7% | 0.185 | 0.182 | 0.201 |
| 16 | 7.6 | 33.1% | 0.337 | 0.325 | 0.127 |
| 32 | **2.7** | **59.6%** | 0.473 | 0.502 | 0.105 |

The pushforward of `N(0,I_256)` through a depth-32 random ReLU MLP is, by the
last layer, essentially a **3-dimensional strongly non-Gaussian object**
embedded in R^256. `|mu_h|/sd_h` rises to ~2.9, so most neurons are far from
the ReLU kink and `Y_i ≈ max(mu_i, 0)` with a small tail correction.

### Where the Gaussian-propagation error actually comes from

`scripts/decompose_error.py`, seed 0, final-layer MSE:

| variant | MSE (layer 32) |
|---|---|
| A. GaussProp, everything propagated | **6.19e-5** |
| B. GaussProp with *oracle* (mu, Sigma) at every layer | **1.25e-6** |
| — plain MC at half budget, for scale | 1.11e-6 |

**50× of the error is moment-propagation error, not the Gaussian marginal
assumption.** And the per-layer profile shows the mechanism: variant B sits flat
at ~2-5e-6 per layer, variant A grows from 1.9e-6 (layer 2) to 6.2e-5 (layer 32)
— i.e. `MSE_A(L) ≈ L × MSE_per-layer`, a **random-walk accumulation** of the
per-layer marginal error. This is consistent with the Jacobian of the map
`Y_{l-1} -> Y_l` being `diag(Phi(t_l)) W_l`, which is norm-preserving on average
under He init (that is what He init is *for*). Errors neither explode nor decay;
they add in quadrature over 32 layers.

**Target arithmetic.** To reach the leaderboard top (1.24e-8) we need
`MSE_final ≈ 2.5e-8` at half budget, i.e. per-layer marginal MSE ≈ `8e-10`,
i.e. per-layer RMS error `2.8e-5` on values of order 0.7. The Gaussian marginal
delivers ~`1.1e-3` RMS per layer. **We need ~40× better per-layer marginals.**

### A trap worth recording

The exact identity `ReLU(h) = h + ReLU(-h)` gives
`Y_l = W_l Y_{l-1} + N_l`, `N_l = E[ReLU(-h_l)] ≥ 0`, which looks like a great
recursion (estimate only the small `N_l` by MC). **It is a trap.** With `N_l`
estimated independently of `Y_{l-1}`, the recursion's Jacobian is `W_l` — *no*
`diag(Phi)` factor — which amplifies by `sqrt(2)` per layer, i.e. `2^16` over 32
layers. Plain MC works only because its per-layer errors are perfectly
correlated and cancel. Any layerwise scheme must keep the `diag(Phi(t))`
feedback to stay norm-preserving. I suspect this is a large part of what "white
box methods break down as depth grows" means in practice.

---

## 2. Literature pass

**ARC, "Estimating the expected output of wide random MLPs more efficiently than
sampling"** ([blog](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/),
[arXiv 2605.05179](https://arxiv.org/abs/2605.05179),
[code](https://github.com/alignment-research-center/mlp_cumulant_propagation)).
Their algorithm is **cumulant propagation ("kprop")**: propagate an approximate
distribution layer to layer using **cumulants and Hermite expansions** — a
Gaussian approximation plus "the lowest-order deviations from that". Reported
scaling `O(n/eps²)` versus Monte Carlo's `Θ(n²/eps²)` — an `n`-fold speedup
**for wide networks** — with MSE `O(eps²)` matching MC. Empirically, ~1/100 the
FLOPs of sampling at 4 hidden layers, width 256. They state explicitly that the
depth dependence is worse and that the methods "break down as the depth grows".

**This is the single most useful fact in the literature pass, because my
measurements explain *why*.** A cumulant/Hermite expansion is a perturbative
expansion whose small parameter is essentially `1/n_eff`, where `n_eff` is the
number of *effectively independent* terms being summed at each neuron. At layer
1, `n_eff ≈ 165` and the expansion is superb. By layer 32 the effective rank has
collapsed to **2.7** — the expansion parameter is `O(1)` and the series has
nothing left to converge to. Depth doesn't break cumulant propagation by
accumulating error; it breaks it by **destroying the CLT the expansion is built
on.**

Supporting background (consulted, none of it changes the plan):
- Cho & Saul (2009), arc-cosine kernels — gives the exact zero-mean bivariate
  ReLU moment I use; the nonzero-mean case needs the bivariate normal CDF, which
  I get from Drezner–Wesolowsky (validated to 1e-16 against Cho–Saul).
- Poole et al. / Schoenholz et al., "Deep Information Propagation" (arXiv
  1611.01232) — the correlation-map fixed point at `c*=1` for ReLU is exactly the
  rank collapse measured above, and He init sits at the norm-preserving critical
  point, which is why per-layer errors accumulate as a random walk rather than
  exploding.
- Lee et al., "Deep Neural Networks as Gaussian Processes" (arXiv 1711.00165) —
  the NNGP kernel recursion. Note this is a *different* object: NNGP averages
  over `W`; here `W` is given and we average only over `x`.

### What the literature does *not* cover, and where I think the win is

Every method above represents the layer distribution **perturbatively around a
Gaussian**. The measured failure is that the true distribution becomes
**low-rank and strongly non-Gaussian in a handful of directions**, while
remaining nearly Gaussian in the other ~250. That structure calls for a
*non*-perturbative representation of the few bad directions and an exact
Gaussian treatment of the rest.

---

## 3. Revised plan: Active-Subspace Gaussian Mixture propagation (ASGM)

Represent the law of `h_l` as a **Gaussian mixture with `K` components sharing a
covariance**:

```
h_l  ~  (1/K) Σ_k  N( c_l^(k),  Σ_l^res )
```

- The `K` component means `c^(k)` are *particles* — they carry all the
  non-Gaussian, low-rank structure, non-perturbatively.
- `Σ^res` is integrated **exactly** at every ReLU, via the closed forms in
  `gaussmath.py`. This is Rao–Blackwellisation: the ~250 near-Gaussian
  directions never contribute Monte-Carlo noise.

**Which directions get particles?** The ones that matter downstream. Run a cheap
GaussProp pass first (0.8% of budget) to get `p_l = Phi(mu_l/sd_l)`, form the
mean end-to-end Jacobian `J = diag(p_L) W_L ... W_2 diag(p_1)`, and take its
top-`r` right singular vectors as the sampled subspace. Everything orthogonal to
it stays Gaussian.

**Initialisation is exact.** `h_1 ~ N(0, W_1 W_1ᵀ)` exactly, so splitting it into
`z = Pᵀ h_1` (particles) and the exact Gaussian conditional
`h_1 | z ~ N(M G^{-1} z, Σ_1 − M G^{-1} Mᵀ)` introduces **zero** error, where
`M = Σ_1 P`, `G = Pᵀ Σ_1 P`.

**Why this can beat MC rather than merely match it.** Only `r ≈ 4–32` dimensions
are sampled, so the particle design can be a *quadrature* or scrambled-Sobol
design rather than i.i.d. points. In `r` dimensions those converge at
`K^{-1}(log K)^r` rather than `K^{-1/2}` — a `p > 1` method, which by §3 of
notes/01 is also the only kind that makes the top half of the FLOP budget worth
spending.

**Cost.** Per layer: `2n²K` (propagate particles) + `~60nK` (component ReLU
moments) + `3n³` (residual covariance). That is `≈1.12×` the cost of plain MC
with `K` samples, so at equal budget `K ≈ 27,000` particles.

**Degenerate limits are the two baselines**, which is a good sign: `r=0, K=1`
recovers GaussProp exactly; `Σ^res → 0` recovers plain Monte Carlo.
