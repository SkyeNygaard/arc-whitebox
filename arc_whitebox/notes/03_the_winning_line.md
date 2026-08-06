# The winning line

> **Superseded by `04_round3_forensics_and_new_paths.md`.** This document was
> useful as a hypothesis generator, but several headline statements were based
> on oracle substitutions or four-network projections rather than a packaged
> estimator. In particular, official cumulant propagation already includes the
> cross terms described here, and the 1.24e-8 leaderboard target is now known
> to have a compute-accounting signature unlike an instrumented white-box
> method. Do not use the projected 1.30e-8 row as an achieved result.

Everything here supersedes the strategy section of `notes/01`, which was built on
a misread scoring rule.

---

## 0. The scoring rule (corrected — this changes the whole problem)

I originally took `s = MSE · max(0.5, C/B)` from the challenge overview page.
The leaderboard shows both *Adjusted Score* and *Final Layer MSE*; their ratios
run up to **9.2×**, which is impossible under a 0.5 floor (that caps the ratio at 2).

Reading the authoritative source — `whestbench 0.13.0`, `scoring.py:579` and `:863`:

```
s_m = final_layer_mse * max(0.1, C_m / B_m)      for valid runs, uncapped above
```

**The floor is 0.1, not 0.5.** Consequences:

- A bias-limited (white-box) method costing under `0.1·B = 2.72e10` FLOPs is
  scored at **0.1 × its MSE**, and every FLOP below that threshold is *free*.
  Gaussian moment propagation uses 2.1e9 — so there are 12× that many free FLOPs
  sitting unused.
- Monte Carlo is unaffected and cannot benefit: `MSE = Vc/(fB)` and the factor is
  `f`, so its score is `Vc/B`, **flat in compute over the entire range**. Plain
  MC scores 7.7e-7 whether it uses 10% or 100% of the budget.
- So at the low-compute end MC is simply not competitive, and the contest is
  purely about white-box bias.

Leaderboard, decoded: #1 has MSE 3.12e-8 at C/B ≈ 0.40; #2 has MSE 2.10e-7 at
C/B ≈ 0.11. **The target is MSE ≈ 1.2e-7 at ≤10% of budget.**

---

## 1. Why Monte-Carlo variance reduction is capped at ~1.5×

Worth stating because it closes off a whole family. The Hermite decomposition of
`ReLU(σ(t+u))` gives per-layer variance-reduction ceilings:

| t | Var | after He₁ | after He₂ | after He₁..₄ |
|---|---|---|---|---|
| 0.0 | 0.339 | 3.8× | 35.7× | 117.8× |
| 1.0 | 0.750 | 17.8× | 58.2× | 239.5× |
| 2.9 | 0.997 | 3724× | 3989× | 8366× |

so the per-layer headroom is large. But the *achieved* gain was only 1.4×, and
the reason is structural: a control variate needs an **exactly known mean**, and
the only exactly-known quantities are Hermite moments of `x` and the layer-1
moments (`E[a_1]` closed-form, `Cov(a_1)` = Cho–Saul). Regressing `a_L` on all of
those gives R² ≈ 0.26 — because a depth-32 random ReLU net has its Hermite
spectrum spread across high degrees. **1/(1−0.26) ≈ 1.35× is the ceiling for the
entire family**, and the measured 1.56× (anchored + sphere) is already there.

Two further routes closed off by measurement:

- **Gaussian handoff** — sample `h_k ~ N(μ_k, Σ_k)` with *oracle* moments at any
  layer `k` and propagate exactly to layer 32: MSE ≈ 5e-6 for every `k` from 2 to
  28. Distributional distortions are amplified, not damped, unlike mean errors.
- **Rank-truncated particle propagation** — the rank collapse looked like it
  should buy 30× more particles at the same FLOPs. It does not: projecting the
  activation batch to rank 8 for only the last 4 layers (where effective rank is
  2.7!) already gives bias² = 2.0e-4, i.e. 600× the MC noise it was meant to
  beat. Participation ratio ≈ 2.7 does *not* mean rank-8 captures the batch.

---

## 2. The result that matters: Edgeworth marginals

`E[ReLU(h)]` expanded in the Hermite basis, `t = μ/σ`, `a_{2+k} = (−1)^k He_k(t) φ(t)`:

```
E[ReLU(h)] = σ [ a_0(t) + a_3(t)·κ_3/6 + a_4(t)·κ_4/24 + ... ]
a_0 = tΦ(t)+φ(t),   a_3 = −tφ(t),   a_4 = (t²−1)φ(t)
```

With **oracle marginal moments** at every layer (isolating the marginal model
from the propagation), final-layer MSE on seed 0:

| marginal model | MSE | score at ≤10% budget |
|---|---|---|
| Gaussian (`a_0` only) | 1.25e-6 | 1.25e-7 |
| + κ₃ | 1.86e-7 | 1.86e-8 |
| **+ κ₃, κ₄** | **3.84e-8** | **3.84e-9** |
| + κ₃..κ₆ | 4.37e-8 | 4.37e-9 |

κ₃ and κ₄ buy **32×** over Gaussian. Going to κ₅, κ₆ makes it worse — the
Edgeworth series stops converging, as expected once |t| ~ 2.9 makes the Hermite
coefficients grow. **Stop at fourth order.** (3.84e-8 is at my oracle-moment
noise floor of 2.2e-8, so the true figure is lower.)

## 3. The precision asymmetry that determines the architecture

Perturbing each propagated quantity by relative noise and re-scoring:

| quantity | relative accuracy needed to beat #1 |
|---|---|
| μ | **1e-4** |
| σ | 3e-3 |
| κ₃ | **10%** |
| κ₄ | **10%** |

This is a gift. The cumulants are the only genuinely expensive things to
propagate — `κ₃(h_i) = Σ_jkm W_ij W_ik W_im κ₃(a_j,a_k,a_m)` is an `n⁴`
contraction, 2.7e11 FLOPs over 32 layers — and they are exactly the ones that
barely need to be right. μ is the most demanding and the cheapest: it is just
`W_l · Y_{l-1}`.

## 4. Where it stands, and the one remaining gap

`src/whest/edgeworth.py` implements EMP: closed-form (μ, Σ) propagation with
Edgeworth marginals, cumulants bought from a small MC side-channel. Seed 0:

| configuration | MSE | score |
|---|---|---|
| Gaussian propagation (baseline) | 6.19e-5 | 6.19e-6 |
| EMP, everything propagated | 1.28e-5 | 1.28e-6 |
| EMP + oracle Σ only | 9.09e-6 | 9.09e-7 |
| EMP + oracle κ₃,κ₄ only | 2.04e-5 | 2.04e-6 |
| **EMP + oracle Σ *and* κ₃,κ₄** | **1.30e-7** | **1.30e-8** |

The last row is level with the #1 leaderboard score of 1.24e-8. Neither
ingredient works alone — a strong interaction, because the μ-recursion needs
both correct at *every* layer to stay inside 1e-4.

**The gap is entirely moment propagation, and it is now two specific problems:**

1. **Σ must be propagated better than the Gaussian bivariate formula allows.**
   `E[a_i a_j]` currently uses the exact *Gaussian* bivariate ReLU moment; at
   κ₃ ≈ 0.47 that carries a few-% error where σ needs 3e-3. Fix: bivariate
   Edgeworth on the pair `(h_i, h_j)`.
2. **κ₃, κ₄ must be propagated, not sampled.** The 10% tolerance is a
   *final-layer* tolerance; injected at all 32 layers it tightens by √7.7. With
   oracle Σ, going from 6k sampled cumulants to oracle cumulants moves MSE from
   9.09e-6 to 1.30e-7 — a factor of 70, needing ~8× less cumulant noise, i.e.
   ~420,000 samples = 6.5× the entire budget. Sampling cannot get there.

Route for (2) — the `n⁴` contraction becomes affordable under the measured rank
collapse. With `a_l − μ ≈ U ξ + ε` (`U` the top-r eigenvectors of the Σ we are
already propagating, `ε` treated as Gaussian), `κ₃(h_i) = Σ_abc T_abc v_a v_b v_c`
with `v_i = Uᵀw_i`, costing `2n²r` for the projections plus `n·r³` for the
contraction — about 1e6 FLOPs per layer at r = 8, against 2.72e10 free. The
latent cumulant tensors `T³ (r³)`, `T⁴ (r⁴)` are then carried by a few thousand
r-dimensional particles, which is cheap precisely because they live in the
collapsed subspace rather than in R²⁵⁶ (the mistake ASGM made).

This is ARC's cumulant propagation, but with the division of labour set by §3:
closed form where precision is demanded and cost is low, crude low-rank
particles where cost is high and 10% suffices.

---

## 5. Second round: four more routes, all closed by measurement

### 5.1 The exact cumulant requirement

With oracle Σ and noise injected into the cumulants at *every* layer (not just
the final one, which is what the earlier §3 table did):

| κ relative error | MSE | score | vs #1 |
|---|---|---|---|
| 0 | 1.30e-7 | 1.30e-8 | 1.05× |
| 1% | 1.34e-7 | 1.34e-8 | 1.08× |
| 3% | 1.64e-7 | 1.64e-8 | 1.32× |
| 10% | 3.31e-7 | 3.31e-8 | 2.67× |
| 30% | 2.42e-6 | 2.42e-7 | 19.5× |

So **κ₃ needs ~3% relative — about 0.014 absolute.** The §3 figure of "10% is
fine" was a final-layer-only tolerance; injected at all 32 layers it tightens by
roughly √7.7, exactly as the sensitivity analysis predicts. Sampling to 0.014
needs `6/0.014² ≈ 30,600` full-cost samples = 1.3e11 FLOPs = 47% of budget.

### 5.2 Offline calibration — the bias is universal, but fixing it backfires

Every evaluation MLP is an i.i.d. draw from one distribution, so anything
universal is free at test time. The propagated σ bias **is** strikingly
universal: `σ_prop/σ_true` at layer 4 is 0.99740 / 0.99805 / 0.99718 / 0.99737
across four independent MLPs — a spread of 3e-4 against a 3e-3 requirement.

Fitting one scalar per layer on six training MLPs and testing on four held-out
ones (strict separation — this is a distributional prior, not seed
memorisation) made things **10× worse**: 1.28e-5 → 1.28e-4. The ablation
explains it: with cumulants sampled at 6k, fixing Σ buys only 1.4×, so σ is not
the binding constraint, and forcing it breaks a compensating cancellation
between the σ error and the marginal-model error.

**The cumulants are not universal.** κ₃ averaged per (layer, |t|) bin swings
from −0.20 to +0.15 across seeds — it is neuron-specific, not a function of
depth and standardised mean. No offline table is possible.

### 5.3 Conditionally-independent latent propagation (CIL) — right idea, wrong regime

`src/whest/cond.py`. Factor `Σ_l = F Fᵀ + D` with `D` diagonal; then **conditional
on the q-dimensional latent z, the pre-activations are exactly independent**, so
the ReLU factorises and one q-dim quadrature yields the mean, the full
covariance (`Cov(a_i,a_j) = Cov_z(α_i, α_j)` exactly, for i≠j — no bivariate
normal CDF anywhere), and the cumulants, all from the same pass. The latent need
not stay Gaussian, so carrying it as particles captures exactly the low-rank
non-Gaussian structure the rank collapse produces.

Measured: 5.15e-5 — no better than plain Gaussian propagation. Two measurements
say why:

- **rank-q + diagonal is a bad fit early and a good fit late.** Off-diagonal
  residual energy after removing the top q eigendirections: at layer 2, 0.56
  (q=16) and still 0.24 (q=64); at layer 32, 0.019 (q=16). The factorisation is
  only valid exactly where the rank has collapsed.
- **conditional independence does not survive a layer.** Given z, `h_{l+1}` has
  conditional covariance `W diag(v) Wᵀ`, which is measured to be **72%
  off-diagonal**. The property has to be re-established every layer, and doing so
  by forcing the residual diagonal is what destroys the accuracy.

An EMP→CIL hybrid switching at layer 17/21/25/29 gives no gain over EMP alone on
any of four MLPs — by layer 25 the accumulated drift is already the whole error,
consistent with the hybrid-oracle sweep.

### 5.4 Cheap cumulants from truncated particles — dead

Rank-r particles cost `4nr` per layer instead of `2n²`, i.e. 16× less at r=8, so
30k particles would fit in 29% of the free budget. But truncation destroys the
cumulants far worse than it destroyed the answer:

| rank | RMS κ₃ error | relative | needed |
|---|---|---|---|
| 8 | 1.009 | 308% | |
| 32 | 0.295 | 90% | |
| 64 | 0.179 | 55% | **3%** |
| none (40k particles) | 0.0147 | 4.5% | |

Even *untruncated* 40k particles give 4.5% on κ₃ and 12.6% on κ₄.

---

## 6. Conclusion

Everything routes back to the same place, and each detour is now closed with a
measured margin rather than an argument:

- MC variance reduction: capped at 1.5× by the Hermite spectrum (R² = 0.26).
- Gaussian handoff at any layer: ~5e-6.
- Rank truncation: 600× the noise it was meant to beat, for the answer; 55%
  κ₃ error at rank 64, for the cumulants.
- Offline calibration: works for σ, but σ is not binding; cumulants are not
  universal.
- CIL: exact and elegant, but its factorisation is only valid at depth.

**The problem is analytic third- and fourth-cumulant propagation with the cross
terms — the full `κ₃(h_i) = Σ_jkm W_ij W_ik W_im κ₃(a_j,a_k,a_m)` — to ~3%
relative, at under 2.7e10 FLOPs.** Everything else needed to convert that into a
winning score is built and measured: the Edgeworth marginal (32× over Gaussian),
the exact bivariate covariance propagation, and the harness that scores it.
With those cumulants and a correspondingly accurate Σ, the estimator scores
1.30e-8 against the leader's 1.24e-8.

The diagonal ("presumption of independence") approximation to that contraction
is *not* available here: it gives `Σ_j W_ij³ κ₃(a_j) ~ κ₃/n`, which is O(1/256),
while the measured κ₃ at layer 32 is 0.47. The cross terms are the entire
signal — which is the same rank-collapse fact in yet another guise.
