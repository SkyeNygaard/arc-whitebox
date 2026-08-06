# Trying to beat sampling on WhestBench: a full map, including every dead end

*A record of what we tried on the ARC White-Box Estimation Challenge, what
worked, what didn't, and — more usefully — the closed-form results that say which
whole families of ideas cannot work. Written for someone who knows a bit of
linear algebra and a bit of ML, not for a specialist.*

**Where we ended up:** a graded score of 2.39e-7 (rank ~52), which we can now
prove is essentially the best any *sampling-style* method can do. The useful
output isn't the score — it's a theory that tells you when to stop, validated to
0.78% against measurements it was never fitted to, plus a rigorous test you can
apply to your own approach to see whether it can possibly reach the top.

---

## Contents

1. [The problem](#1-the-problem-in-plain-words)
2. [The one structural fact](#2-the-one-structural-fact-everything-rests-on)
3. [How approaches stack up](#3-how-the-approaches-actually-stack-up)
4. [The ceiling theorem](#4-the-main-result-you-can-compute-the-ceiling-in-closed-form)
5. [Stronger: the best possible weights](#5-stronger-the-best-possible-estimator-not-just-the-best-design)
6. [How difficulty scales with depth](#6-how-hard-is-this-problem-really)
7. [The arithmetic axis, and a trap](#7-the-other-lever-arithmetic--and-the-trap-that-cost-us-a-submission)
8. [The detector: reading the leaderboard](#8-the-bound-is-a-detector)
9. [The white-box route and its one blocker](#9-the-white-box-route-and-the-single-number-it-reduces-to)
10. [Everything that didn't work](#10-everything-that-didnt-work-with-numbers)
11. [The leaderboard accounting situation](#11-about-the-top-of-the-leaderboard)
12. [Honesty notes and reproduction](#12-honesty-notes)

---

## 1. The problem, in plain words

You're handed the weights of a randomly-initialised neural network: 32 layers
deep, 256 neurons wide, ReLU activations, no biases. The question: *if I fed this
network random Gaussian noise, what would each neuron's average activation be?*

You could just do that — feed in 30,000 random vectors and average. That's Monte
Carlo, the baseline everyone has to beat. The catch is a compute budget:

```
score = (mean squared error) × max(0.1, compute_used / budget)
```

Lower is better. Budget is 2.72×10¹¹ FLOPs per network. Compute is counted two
ways: arithmetic through the provided `flopscope` library is counted
analytically, and anything else is charged by wall-clock time at 10¹¹ FLOPs per
second.

### The first thing worth internalising

That formula has a consequence people miss. If your method is Monte-Carlo-like —
error falling as 1/N with N samples — **your score doesn't depend on how much
compute you use.** Doubling samples halves error and doubles cost; they cancel
exactly. Score is flat across the entire budget range.

So the useful decomposition is:

```
score  =  V_eff  ×  (cost per sample) / budget       where  V_eff = N × MSE
```

`V_eff` measures how good your *sampling scheme* is, independent of how many
samples you take. **Two levers, and only two:** make each sample worth more, or
make each sample cheaper. Everything below is about how much room each has.

### A correction worth flagging early

The challenge overview states the multiplier floor as `max(0.5, C/B)`. It is
actually **0.1** — verified in `whestbench 0.13.0`, `scoring.py:579`. We built a
whole strategy on 0.5 before checking. The difference matters enormously: at 0.1,
a cheap analytic method is scored at one-tenth its MSE, which is exactly how the
top of the leaderboard is built (§8). **Read the scoring code, not the overview.**

---

## 2. The one structural fact everything rests on

The network has no bias terms, which makes it **positively homogeneous**:
`f(c·x) = c·f(x)` for `c > 0`.

A Gaussian vector splits as (random length) × (random direction), and for a
Gaussian those two are *independent*. So:

```
E[f(x)]  =  E[‖x‖] × E[f(direction)]
```

`E[‖x‖]` is a closed-form number (≈15.984 in 256 dimensions). **The entire radial
part of the integral is exact and free.** Every good submission does this; it's
worth about 4%.

More importantly it reframes everything: you are doing **cubature on a sphere** —
choosing directions and averaging. There is a century of mathematics about
choosing points on spheres well, and it turns out to give an exact answer to how
much that can ever buy you.

---

## 3. How the approaches actually stack up

Official Mini-100 adjusted scores (lower is better):

| approach | score | what it is |
|---|---|---|
| plain Gaussian Monte Carlo | ~7.7e-7 | just sample |
| MC + variance reduction (best we found) | 4.91e-7 | anchored control variate + sphere |
| scrambled Sobol + antipodal pairs | 3.46e-7 | low-discrepancy directions |
| **Kerdock/MUB spherical 5-design** | **2.26e-7** | structured point set |
| + batched Strassen (local measurement) | 1.79e-7 | same points, cheaper arithmetic |
| *(our actual graded submission)* | *2.39e-7* | *Strassen gain eaten by wall-clock, §7* |

### The 5-design, briefly

A **spherical t-design** is a point set on which averaging any polynomial of
degree ≤ t gives the *exact* integral over the sphere. Not approximately —
exactly.

In 256 dimensions you can build one from **mutually unbiased bases** (MUBs): sets
of orthonormal bases where every vector in one makes the same angle with every
vector in another. The maximum number of real MUBs in dimension 256 is 129, and
that maximum is achieved using **Kerdock codes** from coding theory. 129 bases ×
256 vectors × 2 signs = **66,048 directions**, forming a 5-design.

Two bonuses: including each point and its negation kills all odd-degree error for
free, and because the Kerdock bases are a sign pattern times a Walsh–Hadamard
matrix, the network's *first* layer can be evaluated with a fast Walsh–Hadamard
transform instead of a matmul — making layer 1 essentially free.

*Credit: the Kerdock design came from a parallel effort on this project, not from
us. What follows is our attempt to work out how much room it left.*

---

## 4. The main result: you can compute the ceiling in closed form

This is the piece we think is most useful to share, because it tells you when to
stop.

### The idea

Any function on a sphere decomposes into **harmonics** of increasing degree —
like a Fourier series, but spherical. Degree 1 is the smooth linear part, degree
2 the quadratic wiggle, and so on.

A t-design annihilates all error from degrees 1 through t. So the remaining error
is determined entirely by **how much of the function lives above degree t**. Know
that, and you know exactly what any design can buy.

And you can know it, in closed form, without running a single network.

### How

For a random ReLU network there's a classical result — the **dual activation**
(Daniely, Frostig & Singer) — saying the correlation between the network's
outputs at two input directions depends only on the angle between them, through a
function `κ` you can write down. For ReLU normalised to unit output variance:

```
κ(t) = Σ aₖ tᵏ ,   a₀ = 1/π,  a₁ = 1/2,  aₖ = ((k−3)!!)² / (π·k!) for even k ≥ 2
```

Stack 32 layers and the correlation function is just `κ` composed with itself 32
times. **The Gegenbauer coefficients of that composed function are exactly the
fraction of the network's variance living at each harmonic degree.**

Two implementation notes that cost us time. The naive projection is hopeless in
float64 — the degree-ℓ coefficient is `dim(H_ℓ)·E[C·G_ℓ]`, and `dim(H_ℓ)` exceeds
10³⁰ by degree 20 while the integral underflows to match; it returns *negative
variances*. Expanding monomials into the Gegenbauer basis by the three-term
recurrence keeps every coefficient positive and O(1), and is unconditionally
stable. Also compose by repeated squaring (`C₂ₘ = Cₘ∘Cₘ`): 5 compositions instead
of 31.

### The answer

| degree | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | **above 12** |
|---|---|---|---|---|---|---|---|---|---|
| share of variance | .111 | .097 | .073 | .061 | .050 | .043 | .033 | .021 | **.398** |

The spectrum decays as a **power law, not exponentially** — 40% of the variance
sits above degree 12. That's the signature of the `(1−t)^{3/2}` branch point of
the arccos kernel at `t = 1`, sharpened by 32 compositions at criticality.

### What that means a design can buy

| rule | gain vs i.i.d. | minimum antipodal points |
|---|---|---|
| antipodal pairs only | 1.055× | — |
| **Kerdock 5-design** | **1.580×** | 65,792 — *we use 66,048* |
| 7-design (exact through degree 6) | 1.832× | **5,658,112 = 86× the budget** |
| 9-design (exact through degree 8) | 2.084× | 5,547× the budget |

**The next rung is worth 16% and costs 86× the budget.** The design axis is done.

> **A correction we had to make.** `N·P_ℓ = 2` at even degrees, not 1. An
> antipodal rule *doubles* even-degree error while annihilating odd degrees,
> because 66,048 antipodal points are only 33,024 independent directions. So
> antipodal pairing is worth 1.055×, not the ~2× that "it kills half the
> spectrum" suggests. Our first version had this wrong; computing the design's
> frame potentials caught it.

### Why you should believe it

Parameter-free — nothing fitted. It predicts the Kerdock design beats two-stream
Sobol by **1.4972×**; the measured ratio on the official 100-network set is
**1.4855×**. **0.78% off.**

Independently, the same machinery computes the design's error at each degree
directly from its geometry and returns **exactly 0.000e+00 for degrees 1–5** —
the definition of a 5-design emerging from a calculation that doesn't know it's
looking at one.

### The consequence people keep re-discovering the hard way

> **A 5-design integrates every polynomial of degree ≤ 5 exactly. So a control
> variate built from polynomials of degree ≤ 5 contributes exactly nothing** —
> not "a little", not "noisily", but algebraically zero.

And the leftover 63% of variance is spread thinly over degrees 6→∞ with a heavy
tail no low-order model captures. That predicts, correctly, that the entire
control-variate family fails (see §10).

Two more open questions close analytically:

- **Weighting the 129 bases unequally.** All 129 are mutually unbiased, hence
  interchangeable under the configuration's symmetry. Unequal weights can only
  push the low-degree errors off zero. Nothing to search.
- **Choosing a rotation per network.** The error profile is rotation-invariant,
  so *for the ensemble every rotation is exactly equivalent*. A rotation only
  helps a specific network through that network's own degree-≥6 harmonics, which
  no cheap weight statistic sees — matching the measured Spearman ≈ 0.05 between
  every proposed selector and true error. There's a 23% oracle gap and it is not
  reachable: estimating it from probe directions is ~29× noisier than the error
  being estimated.

---

## 5. Stronger: the best possible *estimator*, not just the best design

The above bounds equal-weight rules, which is weaker than needed — a competitor
could weight points however they like. You can close that gap.

Treat the network as a random field with exactly the covariance we now know
(`C₃₂`). Then for **any** point set, the best possible linear estimator of the
integral has a closed form — this is **Bayesian quadrature** — and so does its
error. That bounds *every* linear estimator from those evaluations, optimal
weights included.

*(Again a stability trap: the direct form `A₀ − A₀²N/S` subtracts two numbers
agreeing to 1e-4 and returns a negative variance. Expanding in the harmonic basis
gives `σ²_BQ = A₀·R/(1+R)`, a ratio of positive quantities.)*

Three results:

**1. Equal weights are already optimal.** For a point set with enough symmetry —
which Kerdock has — the Bayes-optimal weights come out exactly equal. Computed
difference: **0.000%**. So reweighting the 129 bases can't help. Not "probably
won't" — *cannot*.

**2. The theory predicts the absolute error, not just ratios.**

| | value |
|---|---|
| predicted Kerdock MSE | 2.40e-7 |
| **measured** Kerdock MSE | **2.28e-7** |
| agreement | **5.2%** |

From a calculation that never runs a network.

**3. A counting argument explains why degree 5 is the wall.** A rule exact
through degree 2s must reproduce every moment up to order s — `C(d+s−1, s)`
independent conditions:

| exact through degree | minimum antipodal points |
|---|---|
| 2 | 512 |
| **4 (and 5 free, by antipodal symmetry)** | **65,792** ← *we use 66,048* |
| 6 | 5,658,112 |

66,048 points can annihilate degrees 1–5 **and no more**, and the Kerdock design
does exactly that. It doesn't merely happen to be good — **it sits on the
information-theoretic limit for its point count.**

> **Among all estimators that evaluate the network at 66,048 points and combine
> the results linearly with any weights whatsoever, the equal-weight Kerdock
> 5-design is optimal.**

**Scope, stated honestly:** this bounds *linear estimators built from point
evaluations*. It says nothing about a method that reads the weights and computes
analytically. That turns out to be exactly where everyone above us lives (§8).

---

## 6. How hard is this problem, really?

The same calculation answers a question we haven't seen asked: **how does the
difficulty scale with depth?**

| depth | total variance | share in degrees ≤5 | what a 5-design buys |
|---|---|---|---|
| 1 | 0.681 | 0.987 | **37.7×** |
| 2 | 0.506 | 0.968 | 22.5× |
| 4 | 0.319 | 0.912 | 9.47× |
| 8 | 0.165 | 0.783 | 4.12× |
| 16 | 0.071 | 0.590 | 2.28× |
| **32** (this challenge) | **0.025** | **0.390** | **1.59×** |
| 64 | 0.008 | 0.233 | 1.30× |

We think this explains the shape of the whole competition. At depth 4 — the
regime ARC's original *mechanistic estimation for wide random MLPs* work lives in
— a spherical design is worth **9.5×** over sampling; at depth 1 it's **37×**. At
depth 32 the identical design is worth **1.59×**.

When Phase 1 went from depth 8 to depth 32, the structural advantage available to
everyone fell from ~4.1× to ~1.6×. The mechanism is clean: every extra layer
composes another copy of the arccos kernel's branch point, pushing variance into
degrees no feasible design reaches. The fitted tail exponent drifts 2.75 → 2.04 →
1.52 as depth goes 4 → 32 → 64.

**If a later phase goes deeper, structured designs get *less* useful, not more.**

### Two doors this closes for free

**Gradient information.** The natural escape from a bound on "estimators built
from function values" is to use derivatives too — and a directional derivative
costs the *same* as a forward pass (propagate a perturbation, `δ_l = D_l W_l
δ_{l−1}`, the same 2n² per layer). Derivative-enhanced cubature is a real
technique that buys higher polynomial exactness per point. But on the sphere,
degree ℓ contributes `ℓ(ℓ+d−2)` times as much to a derivative as to a value:

> **a directional derivative carries 26.5× the variance of a function value.**

Each derivative observation is dominated by exactly the high-degree content no
design removes. More information per FLOP by raw count; worse information in
practice. (In 1-D this is the known result that Gauss with 2N points beats Turán
with N points-plus-derivatives.)

**Estimating moments instead of the answer.** `h_L`, the pre-activation, is "one
ReLU less kinked" than `a_L = ReLU(h_L)` — so maybe estimate *its* moments
accurately and reconstruct `E[ReLU]` by an Edgeworth expansion. The two kernels
are `C₃₁` and `C₃₂` and they differ by almost nothing: 1.6067× vs 1.5865× —
**1.3%**. And it degrades further, since you'd also need `h²`, `h³`, `h⁴`, and
products have convolved (heavier) spectra.

---

## 7. The other lever: arithmetic — and the trap that cost us a submission

If the design is fixed, the only thing left is making each direction cheaper.
99.6% of the cost is 31 matrix products: a 66,048×256 activation matrix times a
256×256 weight matrix, once per layer.

`flopscope` charges a matmul analytically as `M·N·(2K−1)`, so a **bilinear
algorithm doing fewer multiplications is charged less**. That's Strassen: multiply
2×2 blocks with 7 multiplications instead of 8, recursively. Legitimate — a real
arithmetic reduction, not an accounting trick.

Two things make it fit: `66,048 = 258 × 256`, so the activations are 258 square
blocks all hitting the same weight matrix; and all `7^L` subproblems at a level
can go through a single batched call, keeping Python-level calls at O(L) instead
of O(7^L).

Measured over 31 layers:

| recursion depth | counted FLOPs | relative score |
|---|---|---|
| dense | 268.4e9 | 1.000 |
| **L=3** | **211.1e9** | **0.776** |
| L=4 | 216.0e9 | 0.794 |

The optimum is L=3, not the L=4–5 an idealised model predicts, because flopscope
also charges `reshape`/`stack`/`concatenate` by element count. Locally this scored
**1.79e-7**, raw MSE bit-identical to dense, 0 failures.

### Trap 1: the wall-clock guard

A first submission built on this **failed on the grader**:

```
SCORER_INFRA_FAILURE — 4× TIME_EXHAUSTED
every MLP exceeded the 60s per-MLP wall-time limit.
This is the estimator being too slow, not a grader fault.
```

There's a **60-second hard wall-clock cap per network**, completely separate from
the FLOP budget. You can be well under budget on counted FLOPs and still be killed.

### Trap 2: the one that's genuinely invisible

> **`fnp.einsum` does not dispatch to BLAS. `fnp.matmul` does. On the identical
> contraction, matmul was 69× faster.**

| operation | einsum | matmul |
|---|---|---|
| dense (66048,256)@(256,256) | 0.268 s | **0.0039 s** |
| batched, L=3 shapes | 0.461 s | **0.0347 s** |

Changing that single call took our estimator from **18.75 s to 5.28 s per
network** — identical counted FLOPs, identical predictions. If you write anything
doing many contractions, check this first.

### Trap 3: and then the grader ate the gain anyway

The fixed version graded at **2.39e-7 adjusted, 2.42e-7 raw → multiplier 0.988**.
But our tracked FLOPs were 211e9 = 0.776 of budget. So the grader charged
`0.988 − 0.776 = 0.212` of budget as **residual wall time = 0.577 s**, where
locally it was 17.8 ms. **The grader is ~32× slower on the memory-bound parts,
and that ate the entire 22% Strassen saving.**

The deeper lesson: **the FLOP model and the wall-clock guard pull in opposite
directions.** Strassen buys counted FLOPs by replacing one enormous,
perfectly-BLAS-optimised GEMM with many smaller ones — exactly what real hardware
is worst at.

---

## 8. The bound is a detector

Here's the most practically useful consequence of §4–5. Since
`adjusted = MSE × max(0.1, f)` while `MSE ≥ V_eff/N` and `f = N·cost/B`, the floor
lands on the **adjusted score itself, independent of N**:

```
adjusted  ≥  V_eff × cost / B  =  2.25e-7
```

Our graded submission: **2.39e-7**. Just above it — exactly where a pure cubature
method must sit. We are the experimental control for our own theorem.

Applied to the public leaderboard, **every entry below 2.25e-7 is provably not a
linear estimator built from network evaluations**:

| entry | adjusted | raw MSE | implied budget f | vs the floor |
|---|---|---|---|---|
| abhinav (#2) | 2.30e-8 | 2.10e-7 | 0.110 | **9.8× below** |
| daddy_yours (#3) | 3.00e-8 | 2.88e-8 | 1.042 | **7.5× below** |
| mliston (#4) | 4.63e-8 | 1.68e-7 | 0.276 | 4.9× below |
| sweaty_dog (#14) | 1.21e-7 | 1.46e-7 | 0.829 | 1.9× below |
| **us (#52)** | 2.39e-7 | 2.42e-7 | 0.988 | **at the floor** |

**The top ~50 have all left the point-evaluation paradigm.** abhinav sits at the
0.1 floor with MSE 2.10e-7 — you cannot buy 66,048 evaluations with 11% of the
budget, so that is an analytic method.

We drew the wrong conclusion from our own theorem for a while. We proved the
statistical axis was closed and concluded "so grind the arithmetic." The right
conclusion was **"so leave it"** — which the top 50 had already done.

**The loophole, precisely stated:** the bound assumes you integrate `f` *with its
own spectrum*. It does not bind `Ŷ = ∫g + cubature(f − g)` when `g` has an exactly
known integral and `f − g` has a lighter spectrum. Low-degree `g` is useless (§4).
What works is an analytic `g` — the white-box object.

---

## 9. The white-box route, and the single number it reduces to

We spent a long time on analytic moment propagation before the cubature work.
Here's the state of it, which is where we think the remaining opportunity is.

### The structure

Track the mean and covariance of the pre-activations layer by layer and compute
`E[ReLU(h)]` from them. The exact Gaussian formulas are standard; the bivariate
second moment with non-zero means needs the bivariate normal CDF, which we get
from a Drezner–Wesolowsky integral **after the substitution r = sin θ** — that
removes an endpoint singularity that otherwise wrecks Gauss–Legendre, and hits
1.1e-16 against the Cho–Saul arc-cosine kernel with 8 nodes.

### Result 1: Gaussian marginals cap out at Monte-Carlo parity

| variant | final MSE |
|---|---|
| Gaussian propagation, everything propagated | 6.19e-5 |
| ...with **oracle** (μ, Σ) at every layer | **1.25e-6** |
| plain MC at half budget, for scale | ~1.1e-6 |

So **50× of the error is moment propagation** — but even with *perfect* moments,
Gaussian marginals only match Monte Carlo.

### Result 2: Edgeworth marginals are a 32× improvement

Expanding `E[ReLU(h)]` in the Hermite basis with `t = μ/σ` and
`a_{2+k} = (−1)^k He_k(t) φ(t)`:

| marginal model | MSE (oracle moments) |
|---|---|
| Gaussian (`a₀` only) | 1.25e-6 |
| + κ₃ | 1.86e-7 |
| **+ κ₃, κ₄** | **3.84e-8** |
| + κ₅, κ₆ | 4.37e-8 — *worse* |

Third and fourth cumulants buy **32×**. Going further makes it worse — the
Edgeworth series stops converging once |t| ~ 2.9 makes the Hermite coefficients
grow. **Stop at fourth order.**

### Result 3: the precision requirements split

Injecting relative error at **every** layer (not just the last — measuring only
the final layer understates the requirement by ~√7.7, which misled us initially):

| σ relative error | MSE | samples needed | % of budget |
|---|---|---|---|
| 1e-3 | 2.47e-7 | 2,000,000 | **2985%** |
| 1e-2 | 2.57e-6 | 20,000 | 30% |
| 1.8e-2 | 9.62e-6 | 6,173 | 9.2% |

| κ relative error | MSE |
|---|---|
| 1% | 2.15e-7 |
| 6.6% (6,000 samples) | **2.77e-7** |
| 10% | 5.55e-7 |

**κ₃ and κ₄ are cheap** — 6,000 samples (9% of budget) gives 6.6% accuracy,
costing only 25% in MSE. **σ cannot be sampled at all** — 0.1% would need 2
million samples, 30× the entire budget. We built the hybrid (propagate μ, sample
σ and κ) and measured MSE 6.0e-6 against 9.6e-6 predicted. The failure is
understood, not mysterious.

### Result 4: Edgeworth marginals fix σ propagation 10× — but not 100×

| layer | 2 | 8 | 18 | 26 | 32 |
|---|---|---|---|---|---|
| our propagated-σ RMS error | **0.13%** | 1.37% | 3.65% | 4.65% | 5.32% |

With *Gaussian* marginals the layer-32 figure was **11%**. Adding κ₃/κ₄ to the
marginal pulls it to ~1% (median) / 5% (per-neuron RMS). The requirement is 0.1%.

### Result 5: σ sensitivity is highest at the *earliest* layers

Injecting 1% σ noise one layer at a time:

| layer | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| excess MSE | **5.23e-7** | 2.61e-7 | 1.38e-7 | 1.99e-7 | 8.2e-8 | 6.3e-8 |

Layer 2 alone carries more than layers 20–32 combined. This is the *opposite* of
the mean-error sensitivity operator, which damps early-layer errors 16×. The
reason: early σ is large in absolute terms, so 1% of it is a big absolute error
that then propagates.

### Result 6: the two profiles anti-align, and the budget sits in the middle

| layer | 2 | 8 | 18 | **26** | 32 |
|---|---|---|---|---|---|
| our σ error | 0.13% | 1.37% | 3.65% | 4.65% | 5.32% |
| σ sensitivity | 5.23e-7 | 1.38e-7 | 1.60e-7 | 1.44e-7 | 6.3e-8 |
| **our error budget** | 8.9e-9 | 2.6e-7 | 2.1e-6 | **3.1e-6** | 1.8e-6 |

Predicted total excess **1.91e-5** vs measured ~2.0e-5 — the budget model is
validated. Layers 26, 18, 22, 16, 30, 32 carry **67%**.

### Result 7: the mixed cumulants are *not* low-rank

The obvious cheap route to the off-diagonal covariance correction: Cov(a_l)
collapses to effective rank ~3 at depth, so hope the third-cumulant tensor lives
in the same subspace, making its `r³` core estimable from a few thousand samples.
It doesn't:

| layer | Cov effective rank | κ₃ rel. error at r=8 | r=32 | r=64 |
|---|---|---|---|---|
| 26 | **3.4** | 0.285 | 0.113 | 0.046 |
| 30 | 2.6 | 0.294 | 0.092 | 0.033 |

You need r ≈ 64 where the covariance needs 3; at r=64 the core has 262,144
parameters — hopeless from 6,000 samples. **The cumulant structure is genuinely
higher-rank than the covariance structure.** We haven't seen this stated anywhere,
and it's why the cheap bivariate correction doesn't exist.

### What it's all worth

```
with oracle σ:   MSE 2.21e-7 at ~9% budget  →  adjusted 2.21e-8  →  rank #2
with our σ:      MSE 2.03e-5 at ~9% budget  →  adjusted 2.03e-6
```

That 2.21e-7-at-9% is *exactly* abhinav's profile (2.10e-7 at 11%). **The payoff
for solving σ is 100×, and it's the only thing left.** The target is one sentence:

> **Propagate Var(h_l) to 0.1% relative accuracy at depth 32.**

Everything else is in hand: marginal model, cumulants, mean recursion, cost
envelope under the 0.1 floor. The residual error enters through `E[aᵢaⱼ]`, which
uses the exact *Gaussian* bivariate ReLU moment; at κ₃ ≈ 0.47 that's worth a few
percent, and closing it needs a bivariate Edgeworth correction with the mixed
cumulants.

---

## 9b. Why everything fails: the rank collapse sits in the worst possible place

We pursued four more ideas to a decision after the above — exact high-order work
at the few worst layers, drift control, the Roberts–Yaida 1/n expansion, and a
learned correction. All four failed, and the useful part is that they fail for
**one shared reason**.

The effective rank of `Cov(a_l)` falls from 165 at layer 1 to **2.7** at layer 32.
That single fact is:

- **Too severe for perturbation theory.** Every expansion anyone reaches for here
  — cumulant propagation, Edgeworth beyond fourth order, the Roberts–Yaida 1/n
  formalism — needs many effectively independent terms in `h_{l+1,i} = Σⱼ Wᵢⱼ aₗⱼ`.
  At depth there are about three. The expansion parameter is O(1) and the series
  has nothing to converge to.
- **Not severe enough for explicit representation.** A genuinely rank-3
  distribution could be carried exactly by a particle cloud or a grid. But
  truncation is catastrophic — rank-8 applied to only the last four layers gives
  **600×** the noise it was meant to beat — because the discarded ~250 directions
  still carry enough variance to matter at the 1e-7 level.

**The distribution is simultaneously too low-rank to expand around and too
high-rank to write down.** Nothing we tried crosses that gap.

Two specific results from that round worth keeping:

**Patching the worst layers doesn't work, and the reason is subtle.** Six layers
carry 67% of the σ error budget, and the free-compute allowance buys almost
exactly six layers of exact `O(n⁴)` work. It gives **3.05×**, not the 100× the
arithmetic suggests — because you inject the exact Σ and the *next* layer's
propagation re-corrupts it. The budget accounting measured where error is
*generated*, not where it *persists*.

**σ does essentially all the damage, not mean drift.** Handing the propagation
the exact mean at layer 24 and letting it use its own σ from there gives MSE
1.29e-5; adding exact σ gives 3.08e-7 — a 42× gap attributable to σ in the last
eight layers alone. We had this backwards for a while.

Also worth noting for anyone tempted by Roberts–Yaida: it expands the ensemble
over **W** with inputs fixed. This problem has **W fixed** and randomness over
**x**. Different expansion, and the relevant parameter is the effective rank, not
1/n.

---

## 10. Everything that didn't work, with numbers

So you can skip these doors.

### Control variates (all of them)

Explained in one line by §4 — a 5-design already integrates degree ≤5 exactly.

| idea | outcome |
|---|---|
| spherical Stein directional CV | **2.10× worse** than baseline |
| degree-2 spherical harmonic CF (theory coefficient) | **54× worse** |
| degree-2 CF (tempered) | 0.98× — indistinguishable |
| quadratic controls, rank 16 / 32 | 1.011× / 1.029× — *harmful* |
| cross-fitted layer-1 linear CV | 0.93× MSE but costs 3% budget — net wash |

### Monte-Carlo variance reduction

Capped at ~1.5×, and we hit the cap. The reason: a control variate needs an
*exactly known* mean, and the only exactly-known quantities are Hermite moments
of `x` plus layer-1 moments (`E[a₁]` closed-form, `Cov(a₁)` = Cho–Saul).
Regressing `a_L` on all of it gives R² ≈ 0.26, because a depth-32 ReLU net's
Hermite spectrum is spread across high degrees.

| technique | gain |
|---|---|
| antithetic sampling | 1.09× (halves per-*unit* variance but costs 2 samples/unit) |
| sphere sampling (exact homogeneity) | 1.04× |
| anchored layer-1 control variate | 1.24–1.71× |
| **best combination** | **1.56×** |

Also measured: `a_L` is only 26% predictable from the input (degree-3 polynomials
in the top-16 active directions) but **99.1% predictable from layer 31** — which
is what motivated the anchored estimator, and also why it can't do better.

### Moment-propagation family

| idea | outcome |
|---|---|
| Gaussian moment propagation | 6.19e-5 — 270× worse than sampling |
| ...with oracle moments | 1.25e-6 — only Monte-Carlo parity |
| rank truncation of the activation batch | truncating only the last 4 layers to rank 8 gives **600×** the noise it was meant to beat |
| Gaussian handoff (sample from a fitted Gaussian at layer k, propagate exactly) | ~5e-6 for *every* k from 2 to 28 — distributional distortions are amplified, not damped |
| learned closures (neural moment map) | 16–21× worse under free rollout than teacher-forced — state insufficiency, not lack of data |
| Edgeworth correction to the covariance *diagonal* | 3.70% → 3.64% σ error — the error is off-diagonal |
| per-layer σ calibration (offline, held-out MLPs) | **10× worse** — the residual is per-neuron scatter, not global bias |

### Cubature and geometry

| idea | outcome |
|---|---|
| random 2-D plane angular integration | 1.66× to 4.17× *worse* than Sobol |
| nested Richardson extrapolation | 9.3e-7 vs 9.2e-7 direct — nothing |
| interlaced order-2 Sobol | 1.30× worse |
| adaptive stream allocation, seed selection | looked good on selection data, reversed sign on held-out — classic selection leakage |
| terminal smoothing of the last layer's marginal | 0.45× to 0.97× — no-op at best |
| exact Gaussian-line propagation | 6,081× worse at 90–96% of budget |
| smoothed-delta forward-Laplacian proxy | **360,482×** worse |

### A trap worth publishing

The exact identity `ReLU(h) = h + ReLU(−h)` gives the tempting recursion
`Y_l = W_l·Y_{l−1} + E[ReLU(−h_l)]`, where only the small non-negative correction
needs estimating. **It's a trap.** With that correction estimated independently,
the recursion's Jacobian is `W_l` with *no* `diag(Φ)` factor — amplifying by √2
per layer, i.e. 2¹⁶ over 32 layers. Plain MC survives only because its per-layer
errors are perfectly correlated and cancel. Any layerwise scheme must keep the
`diag(Φ(t))` feedback to stay norm-preserving.

---

## 11. About the top of the leaderboard

Worth stating plainly because it changes what you benchmark against.

The mechanism is public and acknowledged
([flopscope accounting bypass](https://discourse.aicrowd.com/t/potential-flopscope-accounting-bypass-bug/18099)):
operations on raw NumPy arrays do real computation but report **zero instrumented
FLOPs**, so the work is billed only through the wall-clock term. The forum notes
this can shift scores by more than 10× depending on grader hardware.

We verified the #1 entry firsthand (its submission page is JS-rendered, so this
needed a real browser). Its per-MLP ledger shows, for every network:

| field | value |
|---|---|
| adjusted score | 1.235e-8 |
| final-layer MSE | 3.12e-8 |
| budget used | 40.01% |
| **instrumented FLOPs, every MLP** | **1.30e7** |
| wall time per MLP | 940–1450 ms |

`1.30e7 + 1e11 × 0.98 s ≈ 1.09e11` reproduces its stated effective compute
exactly. So **0.012% of charged compute is instrumented**; over 99.98% is the
clock term. 13M FLOPs is about three 256×256 matmuls — it cannot produce the
~half a million forward passes that error level implies.

You can bound the effect without speculating about the method. Reaching MSE
3.12e-8 honestly with the best design known needs ~483,000 directions ≈ 1.96e12
real FLOPs, against 1.09e11 charged: **18×**. Even assuming a perfect 9-design
(1.32× better than any that exists), still **13.6×**. Backing that out puts the
underlying algorithm at ~2.2e-7 honest-equivalent — about level with the Kerdock
design.

**So benchmark against ~2.3e-7, not 1.2e-8.** Note also that ranking comes from
fresh private re-runs on a pinned flopscope, so this likely won't survive anyway.
We did not build on it, and wouldn't.

---

## 12. Honesty notes

- **LLM usage.** This work was done in collaboration with an LLM (Claude), which
  wrote essentially all the code and drafts of this document. The mathematics in
  §4–6 was derived and then checked two independent ways — against a known
  closed-form kernel, and against measured scores it was never fitted to. Numbers
  reported as measured were measured. Where we're uncertain we say so.
- **Credit.** The Kerdock/MUB design and most of the negative results in §10 came
  from a parallel effort on the same project. The ceiling theory, the Bayesian-
  quadrature bound, the depth-scaling curve, the leaderboard detector, the
  accounting verification, and the einsum/matmul finding are ours.
- **Mistakes we made and corrected**, left visible because the trail is useful:
  the scoring floor (0.5 → 0.1); the antipodal factor (`N·P_ℓ = 2`, not 1); a
  negative variance from catastrophic cancellation in the BQ formula; a claimed
  divergence at depth 64 that was a truncation artifact; and an apparent 1.8×
  disagreement with a measured i.i.d. baseline that turned out to be an
  unconverged 10-run mean sitting 1.52× above its median.
- **Scope limit, repeated because it matters.** The ceiling result bounds *linear
  estimators built from point evaluations*. It does not bound analytic white-box
  methods. If someone beats it — and per §8 about fifty people have — that is
  where they did it.

### Reproducing

The spectrum calculation is short, self-contained, needs no network evaluations,
and runs in about a second:

```
spectrum.py            # harmonic spectrum and the design ceilings
design_potentials.py   # exact per-degree error of the Kerdock configuration
bayes_quadrature.py    # optimal-weight bound and the counting argument
regularity.py          # depth-scaling curve and the gradient-information test
validate_ceiling.py    # the 0.78% parameter-free check against measurement
```
