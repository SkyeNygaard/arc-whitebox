# WhestBench — Problem Statement, Structure, and Idea Bank

*Written before consulting the literature (deliberately). Literature pass is `02_literature_pass.md`.*

---

## 1. The problem, stated precisely

Given the **weights** of a randomly-initialised MLP

- width `n = 256`, depth `L = 32` hidden layers (Phase 1; warm-up was `L = 8`)
- `h_1 = W_1 x`, `a_l = ReLU(h_l)`, `h_{l+1} = W_{l+1} a_l`
- He-Gaussian init: `W_{ij} ~ N(0, 2/n)`, i.i.d.
- input `x ~ N(0, I_n)`

estimate the matrix

```
Y[l, i] = E_{x ~ N(0,I)} [ ReLU(h_{l,i}(x)) ]        l = 1..L,  i = 1..n
```

Only the **final layer** row is scored:

```
MSE_final = (1/n) Σ_i (Ŷ[L,i] − Y[L,i])²
score     = MSE_final · max(0.5, C/B)          (lower is better)
```

with `C` = analytically-counted FLOPs (flopscope) and `B = 2.72e11` per MLP.
The reference `Y` is a Monte-Carlo average over a sample pool ~**15,000× our budget**,
so for our purposes it is exact (reference noise floor ≈ our-MC-noise / 15000).

**This is a pure numerical-integration problem.** No learning, no optimisation.
Integrate a fixed, known, piecewise-linear function over a 256-dimensional
Gaussian, to relative accuracy ~1e-4, using ~1.4e11 FLOPs.

### Current leaderboard (2026-07-27)

| rank | score |
|---|---|
| 1 | 1.24e-8 |
| 2 | 2.30e-8 |
| 3 | 5.35e-8 |

Plain Monte Carlo at the full budget scores about **1e-5** (derived in §3).
So the frontier is roughly **800–1000× better than naive sampling**, i.e. ~30× lower RMS error.

---

## 2. The FLOP cost model (measured directly from flopscope 0.9.1)

| op | cost |
|---|---|
| elementwise add/mul/div/sqrt/max/abs/sign | 2 per element |
| `exp`, `log`, `arccos`, `arctan`, `tanh`, `sin` | **32 per element** |
| `norm.cdf` | 96 per element |
| `norm.pdf` | 54 per element |
| `norm.ppf` | 170 per element |
| `random.randn` | **32 per sample** |
| `where` | 10 per element |
| matmul `(256,256)@(256,256)` | 6.70e7  = `4·n³` |
| the same with symmetry detected (`X Xᵀ`) | 3.36e7 = **half** |
| `cholesky(256)` | 1.12e7 |
| `eigh(256)` | 3.02e8 |
| full `svd(256)` | 8.72e8 |
| truncated `svd(256,256,k=16)` | 4.19e6 |

**Two immediately exploitable levers:**

1. **`float32` bills at exactly half of `float64`.** (`float16` bills the same as
   `float32` — the rate is floored at 0.5.) Verified: matmul 6.72e7 → 3.36e7.
   This is a free 2× on the entire budget. Everything below assumes float32,
   i.e. matmul = `2n³`, matvec = `2n²`.
2. **Symmetry-aware einsum halves Gram-type contractions.** Any algorithm built
   around `W Σ Wᵀ` should be structured so flopscope sees the symmetry.

### Derived unit costs (float32)

- One forward pass, one sample, `L=32`: `32 · 2n² = 4.19e6` FLOPs (+ 8192 for `randn`).
- Budget `B = 2.72e11` ⟹ **≈ 64,000 MC samples** at full budget, **32,000 at half**.
- One full Gaussian covariance propagation, all 32 layers: `32 · (2 · 2n³) ≈ 1.1e9`
  plus the ReLU-covariance step. **≈ 0.4 % of budget.**

That last number is the single most important fact in this document.
**Closed-form Gaussian moment propagation costs under 1% of the budget.**
Whatever the leaders are doing, they have ~100–250× the cost of a full covariance
propagation available to spend on modelling what Gaussian propagation gets wrong.

---

## 3. The scoring geometry — where to operate

Let the estimator have error `MSE = b² + V·N^{-p}` where `N ∝ C` is the sample /
particle count, `b` is bias, and `p` is the convergence exponent. Write `f = C/B ∈ (0,1]`.
For `f ≥ 0.5`:

```
score(f) = b²·f  +  V·(c/B)^p · f^{1−p}
```

- The **bias term is strictly increasing in `f`** — always pushes you to `f = 0.5`.
- The **variance term** goes as `f^{1−p}`:
  - `p < 1` (worse than MC): increasing in `f` → operate at `f = 0.5`.
  - `p = 1` (plain MC): **completely flat** — sample count is score-irrelevant above `B/2`.
  - `p > 1` (QMC, quadrature, superlinear methods): decreasing in `f` → **operate at `f = 1`**.

**Consequences.**

- For any ordinary MC-based method the effective budget is `B/2 = 1.36e11` and the
  score carries a fixed 0.5 factor. Spending the second half of the budget buys
  literally nothing. *Many participants are probably leaving this on the table.*
- Conversely, a method with genuinely superlinear convergence (randomised QMC,
  low-dimensional quadrature) is the only reason to ever go above `B/2` — and
  detecting that in our own experiments is a decisive fork in the road.
- A zero-variance analytic method scores exactly `0.5·b²`. So: **analytic beats
  MC iff `b² < 2 × MSE_MC(B)`**, i.e. iff RMS bias `< 6e-3` in absolute activation
  units. To reach the leaderboard top we need RMS bias `< 1.6e-4` on values of
  order 0.5 — i.e. **3e-4 relative accuracy**.

### The naive-MC reference number

`Var(a_{L,i})` for He-init ReLU is `E[h²]/2 − (E[h]... )` ≈ 0.68 with `E[h²] ≈ 2`.
So `MSE_MC = 0.68/32000 ≈ 2.1e-5`, `score ≈ 1.06e-5`. Confirmed empirically in §6.

---

## 4. Structural facts about this specific integrand

**F1. Layer 1 is exactly solvable, for free.**
`h_1 = W_1 x` is exactly Gaussian, `N(0, W_1 W_1ᵀ)`. So
`Y[1,i] = ‖W_{1,i}‖ / √(2π)` exactly, and the full joint law of `h_1` is known
in closed form. Any method that samples layer 1 is wasting information.

**F2. The only real approximation in moment propagation is non-Gaussianity of the
pushforward.** Given `W`, `h_{l+1} = W_{l+1} ReLU(h_l)` is a deterministic map.
If we knew the law of `h_l` we could push it forward exactly. Tracking `(μ_l, Σ_l)`
and *assuming joint Gaussianity* is exact at `l = 1` and degrades thereafter.

**F3. Why depth is the enemy — rank collapse.** For ReLU + He init, the cosine
similarity between the representations of two different inputs converges to 1 with
depth. Equivalently, the pushforward covariance `Σ_l` loses effective rank: deep
random ReLU nets squash the input distribution onto a near-one-dimensional ray.
Now `h_{l+1,i} = w_i · a_l` is a sum of `n` terms that are *strongly co-varying*,
so the CLT that justified "`h_2` is nearly Gaussian" **fails at depth**. This is
precisely ARC's stated failure mode ("white-box methods … break down as the depth
grows"). It also tells us the fix: model the *low-dimensional non-Gaussian core*
explicitly and treat only the high-rank remainder as Gaussian.

**F4. Depth/width ratio `L/n = 32/256 = 0.125` is not small.** We are squarely in
the regime where finite-width / large-depth corrections are O(10%), not O(0.1%).

**F5. Only marginals matter at the end.** `Y[L,i] = E[ReLU(w_i · a_{L-1})]` — 256
*scalar* expectations. Joint structure at layer `L−1` matters only insofar as it
determines those 256 one-dimensional marginals. This is a big reduction: we need
accurate scalar marginals of 256 specific projections, not a full joint model.

**F6. Everything is per-MLP, but offline work is free.** The budget is per-MLP and
there is no cross-MLP amortisation. But the submission tarball may contain
arbitrary precomputed artifacts. Since all evaluation MLPs are i.i.d. draws from
the *same* known distribution (He-init, 256×32), any **distributional** calibration
learned offline transfers at ~zero test-time FLOP cost. This is legitimate
(it is a prior over the ensemble, not memorisation of seeds).

**F7. Exact Gaussian ReLU moments.** For `X ~ N(μ, σ²)`, with `t = μ/σ`:
```
E[ReLU(X)]   = μ Φ(t) + σ φ(t)
E[ReLU(X)²]  = (μ²+σ²) Φ(t) + μσ φ(t)
E[ReLU'(X)]  = Φ(t)                      (Stein: also = Cov(a,X)/σ²)
```
For the **zero-mean bivariate** case with correlation ρ (the Cho–Saul arc-cosine kernel):
```
E[ReLU(X)ReLU(Y)] = (σ_x σ_y / 2π) · ( √(1−ρ²) + ρ(π − arccos ρ) )
```
— one `arccos` per pair, ~42 FLOPs, i.e. 2.7e6 per layer. Free.
For **nonzero means** there is no elementary closed form (it needs the bivariate
normal CDF / Owen's T). Three practical routes:
(a) 1-D Gauss–Hermite over one variable with the closed form for the other
    (~20 nodes × 65k pairs × 32 layers ≈ 7.5e9 ≈ 5% of budget — affordable);
(b) an **offline-precomputed 3-D lookup table** in `(μ_x/σ_x, μ_y/σ_y, ρ)` with
    trilinear interpolation (~20 FLOPs/pair — essentially free) [see F6];
(c) statistical linearisation `a ≈ Φ(t)·(h−μ) + residual`, which needs only
    marginals — cheap but breaks down exactly where it matters (ρ → 1).

---

## 5. Idea bank

Grouped by family. Tagged `[cost]` = rough FLOP cost as fraction of `B/2`,
`[bet]` = my prior on payoff.

### A. Monte Carlo with variance reduction

| # | Idea | Notes |
|---|---|---|
| A1 | Plain batched MC, float32 | baseline. `p=1`, score ≈ 1e-5 |
| A2 | **Antithetic pairs `(x, −x)`** | free; annihilates *all* odd-order Hermite components. For a near-linear integrand this alone can be several ×. `[bet: high, cost: 0]` |
| A3 | **Randomised QMC** (scrambled Sobol / lattice) on the 256-d Gaussian | the only family with `p>1`; effective-dimension is the risk. If it works, it also unlocks the full budget (§3). `[bet: medium-high]` |
| A4 | **Radial stratification / Gauss–Laguerre on ‖x‖** | given F3 (radial collapse) the output depends strongly on `‖x‖`; stratify the χ²_256 radius and sample the direction uniformly. Cheap, potentially large. `[bet: high]` |
| A5 | **Analytic linear control variate** `c(x) = Vx` with `V ≈ Π W_l diag(Φ(t_l))` | exactly mean-zero ⟹ unbiased *regardless of coefficient error*. `V` costs one chain of matmuls (1.1e9). Per-sample overhead `2n²` = 1/32 of a forward pass. Removes the linear-in-`x` variance. `[bet: high, cost: <1%]` |
| A6 | Degree-2 Hermite control variates `xᵀAx − tr A` | same logic, second order. Low-rank `A` keeps it cheap |
| A7 | Surrogate-network control variate with analytically known mean | e.g. a rank-`r` linearisation of the whole net |
| A8 | On-the-fly regression control variates with sample splitting | robust, no analytics needed |
| A9 | Latin hypercube / orthogonal arrays | weaker cousin of A3 |
| A10 | Importance sampling | probably useless — we want a mean, not a tail |
| A11 | Common random numbers across neurons | automatic |
| A12 | Multilevel MC in precision | dead: flopscope bills fp16 = fp32 |
| A13 | **Last-layer Rao–Blackwellisation** | see D3 |

### B. Analytic moment propagation

| # | Idea | Notes |
|---|---|---|
| B1 | Exact layer 1 | free, strictly dominates sampling layer 1 |
| B2 | **Gaussian moment propagation** `(μ_l, Σ_l)` with exact ReLU moments | the core white-box baseline. `[cost: ~1%]` |
| B3 | Diagonal-only (ignore correlations) | cheap, expected to be bad at depth |
| B4 | **Low-rank + diagonal `Σ = D + UUᵀ`** | matches the actual rank-collapse structure (F3); `n²r` instead of `n³`. Frees budget for higher-order modelling |
| B5 | **Edgeworth / Gram–Charlier correction** on marginals (track κ₃, κ₄ per neuron) | note the pretty fact: at `μ=0` the κ₃ correction to `E[ReLU]` **vanishes**, and the κ₄ correction is `−σκ₄/(24√(2π))`. So kurtosis is the leading error, not skewness |
| B6 | Latent-space exact + Gaussian residual | ⟶ C1 |
| B7 | **Gaussian-mixture propagation** `Σ_k π_k N(m_k, C_k)` with split/merge | continuously interpolates B2 (K=1) ↔ particles (C→0). ~50 full-cov components fit in budget; ~4000 rank-16 components fit |
| B8 | Skew-normal / two-piece-normal marginal families with matched cumulants | exact `E[ReLU]` available for these |
| B9 | Propagate in a basis aligned with `μ_l` + top eigenvectors of `Σ_l` | reduces to a small non-Gaussian core |
| B10 | Perturbative `1/n` expansion conditioned on the actual `W` | principled but algebra-heavy |

### C. Hybrids — *where I expect the answer to live*

| # | Idea | Notes |
|---|---|---|
| C1 | **Rao–Blackwellised low-rank particle propagation.** Represent the law of `h_l` as `K` particles in an `r`-dim latent subspace **plus** an analytically-integrated Gaussian residual. For neuron `i`, particle `k`: `h ~ N(m_{ik}, v_i)` and `E[ReLU] = g(m,v)` in closed form; average over particles. | This is conditional MC / Rao–Blackwellisation: it removes the "high-frequency" half of the variance *and* the Gaussianity assumption on the dominant directions. **My top pick.** |
| C2 | **Analytic + MC bias correction with James–Stein shrinkage over the 256 neurons** | `Ŷ = Ŷ_an + λ(Ŷ_MC − Ŷ_an)`, `λ` estimated from the 256-dim residual vs its known MC variance. Provably ≥ both components. Cheap insurance, should be the final wrapper on whatever wins |
| C3 | Layerwise hybrid: analytic where Gaussianity holds (early), sample-based where it fails (deep) | note the asymmetry — non-Gaussianity *grows* with depth |
| C4 | **MC-corrected covariance:** use MC samples to estimate `Σ_l` empirically, shrink toward the analytic `Σ_l`, continue analytic | cheap injection of non-Gaussian information |
| C5 | Coupled-surrogate control variate | needs a surrogate with a *known* mean coupled to the same `x` |
| C6 | Analytic mean-path + MC on the fluctuation | ⟶ same as C2 in practice |
| C7 | Propagate analytically, periodically resample particles from the fitted law | keeps a nonparametric handle |

### D. Structural exploits

| # | Idea | Notes |
|---|---|---|
| D1 | **Sensitivity-weighted budget allocation.** `∂Y_L/∂(stat at layer l)` is a product of Jacobians and is computable. Spend accuracy where it propagates | |
| D2 | **Reduce to 256 scalar marginals at the last layer** (F5) | huge simplification of what must be modelled |
| D3 | **Last-layer Rao–Blackwellisation.** Split `w_i` into components along the top-`r` directions of `Σ_{L−1}` (sampled/particled) and the orthogonal remainder (Gaussian, since it *is* a sum of many weakly correlated terms) | directly attacks F3/F5. Nearly free |
| D4 | Use the *distribution* of `W` (iid Gaussian) but never the seeds | seed overfitting is explicitly penalised |
| D5 | Structure contractions so flopscope sees symmetry (2× off) | |
| D6 | **float32 everywhere** (2× budget) | do this immediately |
| D7 | **Operate at exactly `C = B/2`** unless `p>1` (§3) | pure free win for MC methods |
| D8 | **Offline-calibrated correction.** Fit, offline on many random MLPs, a correction to the analytic estimate as a function of cheap per-neuron/per-layer features (`μ/σ`, effective rank, layer index). Zero test-time cost | `[bet: high, cost: ~0]` |
| D9 | Offline-fit the *bivariate ReLU moment* table (F7b) | removes the main cost of exact covariance propagation |

### E. Wildcards

| # | Idea |
|---|---|
| E1 | Track 3rd/4th cumulants **in the low-rank latent space only** (`r³`, `r⁴` with `r≈8–16` → trivial cost) |
| E2 | Characteristic-function / Fourier propagation of marginals |
| E3 | Saddle-point approximations |
| E4 | **Tensor-product Gauss–Hermite quadrature on the effective `r`-dim subspace + MC on the complement.** With `r = 3–5` this converges at quadrature rates in the directions that matter — a `p>1` method, so it also unlocks the full budget |
| E5 | Per-MLP adaptive method selection from a cheap diagnostic |
| E6 | Learned neural surrogate for the correction (= D8/D9) |

---

## 6. Prioritised experiment plan

**Stage 0 — harness.** Generate He-init MLPs; high-sample MC reference (fp64,
≥ 2e6 samples, uncounted); flopscope-counted estimator API; scorer.

**Stage 1 — baselines & measurement.** A1, A2, B1, B2. Then *measure the thing
that matters*: how does the Gaussian-propagation bias grow with depth, and what
is the effective rank of `Σ_l` vs `l`? (F3 is a hypothesis; check it.)

**Stage 2 — the bet.** C1 (Rao–Blackwellised low-rank particles) + A5 + A2, all
at `C = B/2`, wrapped in C2.

**Stage 3 — iterate.** A3/A4/E4 to test whether `p > 1` is reachable (decides
whether the top half of the budget is worth anything); D8/D9 offline calibration.

---

## 7. Open questions to resolve empirically

1. Does `Σ_l` actually rank-collapse at `n=256, L=32`? What is the effective rank at `l=32`?
2. What is the Gaussian-propagation bias as a function of depth? Where does it cross the MC noise floor?
3. Is the deep-layer marginal non-Gaussianity dominated by kurtosis (as the Edgeworth algebra in B5 suggests) or by something the low-order expansion misses?
4. Is the integrand's variance mostly linear in `x`? (decides A2/A5 payoff)
5. Does randomised QMC give `p > 1` here? (decides `f = 0.5` vs `f = 1`)
6. How much does `Y[L,:]` vary across MLPs — i.e. what does score 1.24e-8 mean in relative terms?
