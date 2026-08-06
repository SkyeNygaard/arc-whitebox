# ARC White-Box Estimation Challenge — strategy

*Living document. Last updated 2026-07-28.*

Everything below is either measured or derived in closed form. Where a number is
inherited from elsewhere and not independently checked, it says so.

---

## 1. The aim

Minimise, averaged over the private MLP suite,

```
s_m = final_layer_mse * max(0.1, C_m / B),     C_m = F_m + 1e11 * residual_seconds
B   = 2.72e11 FLOPs        (Phase 1: width 256, depth 32, no biases)
```

The organisers have said the final ranking comes from **fresh private reruns on a
pinned flopscope version**. So the thing to optimise is the *honest* score, and
the public leaderboard's top entries are not the benchmark (§3).

### The decomposition that matters

Every competitive estimator is a cubature rule on the sphere — the network is
positively homogeneous (no biases), so `f(ru) = r f(u)` and the Gaussian radius
integrates exactly at `E[chi_256] = 15.9843826666`. That leaves an angular
integral, and for any such rule:

```
score  =  V_eff  *  c / B          V_eff = N * MSE  (design quality)
                                   c     = FLOPs per direction
```

**Adding points is provably score-neutral** — MSE falls as 1/N, cost rises as N.
So there are exactly two levers, and the whole strategy is deciding how much
headroom each has. §2 shows `V_eff` is nearly exhausted; §5 shows `c` is not.

---

## 2. What is attainable — the ceiling theorem

*This is the new result. `spectrum.py`, `design_potentials.py`, `validate_ceiling.py`.*

For ReLU normalised to `E[phi(Z)^2]=1`, the dual activation (Daniely–Frostig–Singer)
is `kappa(t) = sum_k a_k t^k` with `a_0 = 1/pi`, `a_1 = 1/2`,
`a_k = ((k-3)!!)^2/(pi k!)` for even `k>=2`. Composing 32 layers gives the exact
two-point function of the network on the sphere, and **its Gegenbauer
coefficients in d=256 are exactly the per-degree variance shares of the
integrand**:

| degree | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | **>12** |
|---|---|---|---|---|---|---|---|---|---|---|
| share of Var | .1107 | .0967 | .0727 | .0606 | .0498 | .0430 | .0326 | .0259 | .0212 | **.398** |

The spectrum is **power-law, not geometric** — 40% of the variance sits above
degree 12. That is the fingerprint of the `(1-t)^{3/2}` branch point of the
arccos kernel at `t=1`, sharpened by 32 compositions at criticality. Deeper
networks have heavier tails, so cubature helps *less* as depth grows.

### What a rule can remove

Mean-square error is `V * sum_l A_l P_l`, with `N*P_l` the per-degree efficiency
(0 = annihilated, 1 = i.i.d.-like). The Kerdock configuration's Gram matrix takes
only four values (`+-1`, `0`, `+-1/16`), so its potentials are closed-form:

```
N*P_l  =  1 + (-1)^l + 510 G_l(0) + 32768 (G_l(1/16) + G_l(-1/16))
```

which is **exactly 0 for degrees 1–5** (verified: max |N*P_l| = 0.000e+00 — a hard
correctness check on the whole framework, nothing fitted) and 2.062, 1.998,
2.000, 2.000 … for degrees 6, 8, 10, 12.

> **A correction worth stating.** `N*P_l = 2` at even degrees, not 1. An antipodal
> rule doubles even-degree error while annihilating odd degrees, because 66,048
> antipodal points are only 33,024 independent directions. Antipodal pairing is
> therefore worth only **1.055×**, not the ~2× that "it kills half the spectrum"
> suggests. I had this wrong initially; the frame-potential computation caught it.

| rule | gain vs i.i.d. | minimum antipodal points |
|---|---|---|
| i.i.d. sphere | 1.000× | — |
| antipodal only | 1.055× | — |
| **Kerdock/MUB 5-design (current)** | **1.580×** | 65,792 (uses 66,048) |
| 7-design (exact to degree 6) | 1.832× | **5,658,112 = 86× budget** |
| 9-design (exact to degree 8) | 2.084× | 366,362,752 = 5,547× budget |

### Validation — parameter-free, 0.78%

The only clean like-for-like measurement is official Mini-100, where both rules
were scored on all 100 MLPs. Normalising by `V_eff = N * MSE`:

| rule | MSE | N | V_eff |
|---|---|---|---|
| two-stream Sobol (antipodal) | 3.5681e-7 | 62,768 | 2.2396e-2 |
| Kerdock 5-design | 2.2826e-7 | 66,048 | 1.5076e-2 |

**measured ratio 1.4855× vs predicted 1.4972× — 0.78% discrepancy, nothing fitted.**

*(Do not compare rules across different MLP subsets. The per-MLP MSE distribution
is chi-squared with ~1 effective degree of freedom because the final-layer
fluctuation is rank-1 dominated, so subset means swing ~1.5× — visible as the gap
between the i.i.d. mean 1.284e-6 and median 8.43e-7 over 10 runs.)*

### Consequences

1. **The design axis is finished.** The next exact rung is worth 15.9% and costs
   86× the entire budget. There is nothing to find there.
2. **Every failed control variate is explained in one line.** A 5-design already
   integrates every polynomial of degree ≤5 *exactly*, so any control variate of
   degree ≤5 contributes exactly nothing. The residual 63% of variance lives in
   degrees 6..∞ with a heavy tail no low-order model can capture. This retro-explains
   the whole negative-results table: spherical Stein (2.10× worse), degree-2
   harmonic CF (54× worse), quadratic controls (actively harmful), layer-1 linear
   CV (7% for 3% budget).
3. **Basis weighting is settled analytically.** All 129 bases are pairwise
   mutually unbiased, hence exchangeable under the configuration's symmetry
   group; any weighting that breaks exchangeability can only lift the low-degree
   potentials off zero. Equal weights are optimal — the empirical mixture search
   has no win in it.
4. **Per-network rotation selection cannot work.** `P_l` is rotation-invariant, so
   for the ensemble every rotation is *exactly* equivalent. A rotation only helps
   a specific network through that network's own degree-≥6 harmonic coefficients,
   which no weight-only statistic sees — hence the measured Spearman ≈ 0.05. The
   23% oracle gap is unreachable: estimating it from probe directions would be
   ~29× noisier than the error being estimated.

---

## 3. What is likely happening on the leaderboard

The mechanism is real and publicly documented
([flopscope accounting bypass](https://discourse.aicrowd.com/t/potential-flopscope-accounting-bypass-bug/18099)):
flopscope exposes NumPy-backed raw arrays; operations on them do real
computation but report **zero instrumented FLOPs**, so the work is graded only
through the wall-clock term. The forum states this "can increase someone's score
by more than 10× depending on the hardware of the machine where the grader runs."

**Verified firsthand** (in-app browser, 2026-07-28 — `WebFetch` returns only a JS
shell, so this needed a real browser). Submission 319679, PHASE 1, joe_wanza:

| field | value |
|---|---|
| adjusted score | 1.235e-8 |
| final-layer MSE | 3.12e-8 |
| budget used | **40.01%** |
| mean effective compute | **1.09e11** |
| **instrumented FLOPs, every MLP** | **1.30e7** |
| wall time per MLP | 940–1450 ms |
| AIcrowd's own MC baseline | 6.47e-7 |

The ledger reports `1.30e7` FLOPs for all 50 public MLPs, and
`1.30e7 + 1e11 * 0.98 s ≈ 1.09e11` reproduces the stated effective compute exactly.
So **0.012% of the charged compute is instrumented**; over 99.98% is the
wall-clock term. 13M FLOPs is about three 256×256 matmuls — it is not possible to
run the ~half a million 32-layer forward passes that MSE implies. The work is
being done on raw NumPy arrays and billed by the clock.

Note also that grading is on **50 public MLPs with 50 private sealed**, and the
private half decides the final rank.

---

## 4. Best score with the accounting leak

This is answerable from the mechanism alone. If all work is done on raw arrays,
instrumented FLOPs → 0 and `C = 1e11 * t`. A machine sustaining `R` FLOP/s of real
NumPy work does `R*t` FLOPs while being charged `1e11*t`, so the **effective budget
multiplier is `R / 1e11`**:

| grader raw NumPy throughput | effective budget | leaked score |
|---|---|---|
| 1e11 FLOP/s (λ exactly) | 1× | 1.78e-7 |
| 5e11 | 5× | 3.6e-8 |
| 1e12 (typical multicore BLAS) | 10× | 1.8e-8 |
| 2e12 | 20× | 8.9e-9 |

**The multiplier can now be pinned from the verified numbers, independently of
any claim about what 319679 actually runs.** Its final-layer MSE is 3.12e-8. The
best design efficiency anyone has demonstrated is the Kerdock 5-design's
`V_eff = 1.5076e-2`, so reaching that MSE honestly needs

```
N = 1.5076e-2 / 3.12e-8  =  483,205 directions
  = 483,205 * 4.056e6 FLOPs/direction  =  1.96e12 FLOPs of real arithmetic
```

against **1.09e11 charged** — a factor of **18.0×**. The ceiling theorem bounds
the other direction too: even a *perfect* 9-design would only improve `V_eff` by
1.32×, so no admissible design lowers the implied ratio below **13.6×**.

So the leak is **13–18×, and it is grader hardware, not algorithm**: ~1.96e12
real FLOPs in ~1 s of wall time implies the machine sustains ~2e12 FLOP/s, i.e.
`R/λ ≈ 20×`, which is exactly the observed multiplier. Everything is consistent.

The corollary that matters for us: **1.235e-8 is not a target.** Backing the leak
out puts that entry's underlying algorithm at roughly `18 × 1.235e-8 ≈ 2.2e-7`
honest-equivalent — i.e. about level with the Kerdock design, and *behind* the
1.776e-7 in §6.

**We do not build on this.** It is a known defect the organisers have said they
will close with fresh private reruns on a pinned flopscope; exploiting it is both
against the spirit of the rules and strategically worthless once the rerun lands.
It is quantified here only so we benchmark against honest scores.

---

## 5. The ladder: base → best attainable

| step | adjusted score | V_eff | note |
|---|---|---|---|
| plain Gaussian Monte Carlo | 7.687e-7 | 4.98e-2 | different subset — see caveat |
| i.i.d. sphere + exact radius | — | 4.21e-2 (mean) / 2.76e-2 (median) | 10 runs only — **unreliable** |
| Sobol antipodal, two streams | 3.4607e-7 | 2.2396e-2 | official Mini-100 |
| **Kerdock/MUB 5-design** | **2.2566e-7** | **1.5076e-2** | official Mini-100, current best |
| **+ L=3 batched Strassen** | **1.776e-7** | 1.5076e-2 | **measured, §6 — same V_eff, lower c** |
| *(hypothetical 7-design)* | *1.53e-7* | *1.30e-2* | **impossible — 86× budget** |

**Caveat on the top two rows — and an unresolved tension.** Only the Sobol and
Kerdock rows are official Mini-100 numbers on all 100 MLPs. The i.i.d. figure is
10 runs on a different subset, with mean 1.284e-6 against median 8.43e-7 — a 1.52×
skew, exactly the chi-squared-with-one-degree-of-freedom behaviour §2 warns
about. Taken at face value it implies Kerdock is 2.79× i.i.d. (median: 1.83×),
whereas the theory predicts 1.580×.

I am *not* smoothing this over. The theory nails the like-for-like comparison to
0.78% and is wrong by ~1.8× against the noisy i.i.d. anchor. The most likely
explanation is that the 10-run i.i.d. estimate is simply not converged, but a
second possibility is that scrambled Sobol suppresses more low-degree content
than "antipodal only" — which would mean the theory's *absolute* scale is right
while my identification of the two-stream Sobol rule with plain antipodal
sampling is too crude. **Both readings leave every conclusion in §2 intact**,
because those depend only on ratios between designs, which are validated. But
the absolute ladder is not pinned until someone runs i.i.d. sphere on the full
Mini-100 at matched N. That is cheap and it is open thread #4.

**Remaining honest headroom from today: 1.27× measured and in hand on the
arithmetic axis. The statistical axis is closed.**

---

## 6. The arithmetic axis (where the remaining win is)

flopscope charges einsum analytically as `M*N*(2K-1)`, so a bilinear algorithm
performing fewer multiplications is charged less. This is a genuine arithmetic
reduction, not an accounting trick.

Two structural facts make it work here:

1. **Batching keeps the call count at O(L), not O(7^L).** flopscope charges a
   batched einsum as the sum of its parts, so all `7^L` subproblems at a level go
   through one call. Critical: residual wall time is billed at 1e11 FLOP/s, so
   `7^4 = 2401` separate calls per layer would cost more than they save.
2. **The weight operand is shared.** `66,048 = 258 * 256`, so the activation matrix
   is 258 square blocks all multiplied by the same `W`; the entire right-hand
   Strassen tree is built once per layer and carries no batch axis.

### Measured, full 31-layer forward (`strassen.py`)

| mode | tracked FLOPs | residual | effective | C/B | rel. score | max rel. err |
|---|---|---|---|---|---|---|
| dense | 268,368,347,136 | 0.9 ms | 2.6845e11 | 0.987 | 1.0000 | — |
| L=2 | 219,550,011,392 | 11.1 ms | 2.2066e11 | 0.811 | 0.8220 | 4.7e-6 |
| **L=3** | **209,546,873,856** | **17.8 ms** | **2.1132e11** | **0.777** | **0.7872** | 7.3e-6 |
| L=4 | 214,514,611,456 | 29.0 ms | 2.1741e11 | 0.799 | 0.8099 | 9.8e-6 |

**L=3 is optimal: 21.3% score reduction → 1.776e-7.**

### Winograd's variant is strictly better, and free

Strassen's textbook form uses 18 additions (5 A-side, 5 B-side, 8 C-side);
Winograd's uses 15 (4/4/7). Since additions are exactly what caps the recursion,
this is a direct win. Measured on the same full 31-layer forward:

| variant | L | tracked FLOPs | rel. score |
|---|---|---|---|
| dense | — | 268,368,347,136 | 1.0000 |
| Strassen | 3 | 209,546,873,856 | 0.7869 |
| **Winograd** | **3** | **208,020,590,592** | **0.7766** |
| Winograd | 4 | 211,581,029,376 | 0.7942 |

A further 1.3%, for nothing. Max relative deviation 1.1e-5, still two orders
inside tolerance. *(Residual-time figures in this table were measured while the
Mini-100 scoring job was competing for CPU, so they are inflated; the tracked
FLOP counts are exact and load-independent.)*

Projected: **2.2566e-7 → 1.753e-7.**

- The optimum is at L=3, not the L=4–5 an idealised FLOP model predicts, because
  flopscope charges `reshape`/`stack`/`concatenate` by element count and those
  materialisations grow as `(7/4)^L` while multiplications fall as `(7/8)^L`.
  Measuring rather than modelling was worth 11 percentage points of error.
- **Residual wall time is safe.** 17.8 ms costs 1.78e9 effective FLOPs against
  58.8e9 saved — a 33× margin, so it survives a grader 3–5× slower than this
  machine.
- **Numerical stability passes.** 7.3e-6 max relative error against dense, versus
  a target MSE of 2.28e-7 on values of order 0.7 (7e-4 relative). Two orders of
  margin.
- Amortising the *weight-side* additions across the 258 blocks looked promising
  and is worth almost nothing: `W` is 256×256 against `A` at 66,048×256, so the
  A-side and C-side terms dominate.

---

## 7. Open threads, ranked

1. **Package and validate L=3 Strassen into the Kerdock estimator.** Everything is
   measured in isolation; it needs an end-to-end `whest run` on Mini-100 and a
   `whest validate-package`. Expected 2.2566e-7 → ~1.78e-7.
2. **Push the arithmetic further.** Winograd's 15-addition variant instead of
   Strassen's 18 should move the optimum deeper; and since materialisation is
   what caps L, any formulation that avoids `stack`/`concatenate` (strided views,
   pre-allocated buffers) directly buys recursion depth.
3. **Verify the 319679 numbers** with the in-app browser rather than inheriting them.
4. **Pin the absolute ladder.** Run i.i.d. sphere sampling on the full Mini-100 at
   matched N to resolve the tension in §5. Cheap, and it decides whether the
   theory's absolute scale is right or whether scrambled Sobol is doing more
   low-degree suppression than "antipodal only". Either way §2's conclusions hold
   — they rest on design-to-design ratios, which are validated to 0.78% — but I
   would rather know.
5. Nothing else. The statistical axis is closed by §2, and that is a result, not
   a shortage of ideas.

---

## 8. Reproduce

```bash
cd arc_ceiling
../arc_whitebox/.venv/bin/python spectrum.py           # spectrum + ceilings
../arc_whitebox/.venv/bin/python design_potentials.py  # frame potentials, deg 1-5 = 0
../arc_whitebox/.venv/bin/python validate_ceiling.py   # 0.78% validation
```

Reads `arc_whitebox/` (weights, results, venv) read-only; writes only inside
`arc_ceiling/`.

---

## 9. Round 4: the problem reduces to one number

### 9.1 The ceiling bound is a detector

Because `adjusted = MSE * max(0.1, f)` and `MSE >= V_eff/N` while `f = N*cost/B`,
the floor lands on the **adjusted score itself, independent of N**:

    adjusted  >=  V_eff * cost / B  =  2.25e-7

Our graded submission #320380 scored **2.39e-7** — just above it. That is exactly
where a pure cubature method must sit; we are the experimental control.

Applied to the public leaderboard, every entry below 2.25e-7 is **provably not a
linear estimator built from network evaluations**:

| entry | adjusted | raw MSE | implied f | vs the floor |
|---|---|---|---|---|
| abhinav (#2) | 2.30e-8 | 2.10e-7 | 0.110 | 9.8x below |
| daddy_yours (#3) | 3.00e-8 | 2.88e-8 | 1.042 | 7.5x below |
| mliston (#4) | 4.63e-8 | 1.68e-7 | 0.276 | 4.9x below |
| sweaty_dog (#14) | 1.21e-7 | 1.46e-7 | 0.829 | 1.9x below |
| **us (#52)** | 2.39e-7 | 2.42e-7 | 0.988 | **at the floor** |

The top ~50 have all left the point-evaluation paradigm. The loophole in the
theorem is precise: it bounds integrating `f` *with its own spectrum*. It does
not bind `Yhat = int g + cubature(f - g)` when `g` has an exactly known integral
and `f - g` has a lighter spectrum.

### 9.2 The requirement splits, and only one half is hard

Injecting relative error at **every** layer (not just the last — that was the
error in the earlier §3 table, which understated by ~sqrt(7.7)):

| sigma rel. error | MSE | samples needed | % of budget |
|---|---|---|---|
| 1e-3 | 2.47e-7 | 2,000,000 | **2985%** |
| 3e-3 | 4.48e-7 | 222,222 | 332% |
| 1e-2 | 2.57e-6 | 20,000 | 30% |
| 1.8e-2 | 9.62e-6 | 6,173 | 9.2% |

| kappa rel. error | MSE |
|---|---|
| 1% | 2.15e-7 |
| 6.6% (6k samples) | 2.77e-7 |
| 10% | 5.55e-7 |

**kappa_3 and kappa_4 are cheap** — 6,000 samples (9% of budget) gives 6.6%,
which costs only 25% in MSE. **sigma cannot be sampled at all**: 0.1% accuracy
would need 2 million samples, 30x the entire budget.

Confirmed end-to-end: the hybrid (propagate mu, sample sigma and kappas at 6,000
points) measured MSE 6.0e-6, against 9.6e-6 predicted by the table. The failure
is understood, not mysterious.

### 9.3 Edgeworth marginals fix sigma propagation 10x — but not 100x

Measured ratio of propagated to true sigma, per layer, with Edgeworth marginals:

| layer | 2 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| median ratio | 0.9999 | ~0.999 | ~0.999 | ~1.006 | ~1.010 |
| spread across MLPs | 0.0002 | 0.0013 | 0.0034 | 0.0123 | 0.0094 |

With **Gaussian** marginals the same quantity was off by 11% at layer 32. Adding
the third and fourth cumulants to the marginal pulls it to **~1%** — a 10x
improvement I had not measured before. But the requirement is 0.1%.

Per-layer calibration does **not** close the remaining gap: the medians are
already ~1.00, so there is no universal bias left to remove; what remains is
per-network scatter of ~1%. (This also explains why calibration made things
worse in round 2 — there it was correcting a real 11% bias but breaking a
cancellation against the Gaussian marginal error.)

### 9.4 Where this leaves the whole project

    with oracle sigma:  MSE 2.21e-7 at ~9% budget  ->  adjusted 2.21e-8  ->  rank #2
    with our sigma:     MSE 2.03e-5 at ~9% budget  ->  adjusted 2.03e-6

**The payoff for solving sigma is 100x, and it is the only thing left.** The
target is one sentence:

> Propagate Var(h_l) to 0.1% relative accuracy at depth 32.

Everything else is in hand: the marginal model (Edgeworth to 4th order), the
cumulants (6,000 samples), the mean recursion, and the cost envelope (~9% of
budget, under the 0.1 floor). The remaining error enters through `E[a_i a_j]`,
which currently uses the exact *Gaussian* bivariate ReLU moment; at kappa_3 ~ 0.47
that carries about a percent, and closing it means a bivariate Edgeworth
correction involving the mixed cumulants.

### 9.5 Round 5: where the sigma error actually is

Four measurements, and the third inverts an assumption I had been carrying.

**(a) The Edgeworth correction to the covariance diagonal barely matters.**
`E[ReLU(h)^2]` has its own Hermite expansion (`a_3 = 2phi(t)`, `a_4 = -2t phi(t)`),
worth 12.5% relative at t~0, and I was not applying it. Adding it moved the
layer-32 sigma error from 3.70% to 3.64% and MSE from 2.81e-5 to 2.72e-5. So the
error is **off-diagonal**, not diagonal.

**(b) sigma sensitivity is highest at the EARLIEST layers.** Injecting 1% sigma
noise at one layer at a time:

| layer | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| excess MSE | **5.23e-7** | 2.61e-7 | 1.38e-7 | 1.99e-7 | 8.2e-8 | 6.3e-8 |

Layer 2 alone carries more than layers 20-32 combined. This is the opposite of
the mean-error sensitivity operator (which damps early layers 16x) — because
early sigma is *large* in absolute terms, so 1% of it is a big absolute error.

**(c) The two profiles anti-align, and the budget sits in the middle.** Our
propagated-sigma RMS error grows smoothly 0.13% (layer 2) -> 5.3% (layer 32),
which is smallest exactly where sensitivity is largest. Multiplying:

| layer | 2 | 8 | 18 | 26 | 32 |
|---|---|---|---|---|---|
| our sigma RMS err | 0.13% | 1.37% | 3.65% | 4.65% | 5.32% |
| our error budget | 8.9e-9 | 2.6e-7 | 2.1e-6 | **3.1e-6** | 1.8e-6 |

Predicted total excess **1.91e-5** against measured ~2.0e-5 — the budget model is
validated. Layers 26, 18, 22, 16, 30, 32 carry **67%**.

**(d) The mixed cumulants are NOT low-rank.** The obvious cheap route to the
off-diagonal correction is to note that Cov(a_l) collapses to effective rank ~3
at depth, and hope the third-cumulant tensor lives in the same subspace — then
its `r^3` core is estimable from a few thousand samples. It does not:

| layer | Cov effective rank | rel. error in kappa3 at r=8 | at r=32 | at r=64 |
|---|---|---|---|---|
| 22 | 2.9 | 0.297 | 0.108 | 0.053 |
| 26 | 3.4 | 0.285 | 0.113 | 0.046 |
| 30 | 2.6 | 0.294 | 0.092 | 0.033 |

You need r ~ 64 where the covariance needs 3. At r=64 the core has 262,144
parameters, hopeless from 6,000 samples. **The cumulant structure is genuinely
higher-rank than the covariance structure** — a fact I have not seen stated
anywhere, and the reason the cheap bivariate correction does not exist.

### 9.6 Honest state

The white-box route needs off-diagonal covariance accuracy at layers ~16-32 that
neither the Gaussian bivariate moment (3-5% error) nor any cheap cumulant
shortcut delivers. The remaining untested idea is a **per-neuron** sigma
calibration `c(layer, t_i)` fitted offline — per-*layer* calibration failed
because the medians are already ~1.00 (the residual is per-neuron scatter, not
global bias), but the scatter may be a smooth function of the neuron's own
`t = mu/sigma`, which is observable at run time.

---

## 10. Round 6: taking the radical paths

Four ideas pursued to a decision. All four fail, and — this is the payoff — they
fail for **the same underlying reason**.

### 10.1 Exact n^4 contraction at the six worst layers — falsified

The 0.1 floor leaves ~12x free compute unused, one exact third-cumulant
contraction costs ~4.3e9, the free budget buys ~6.3 layers, and six layers carry
67% of the sigma error budget. The arithmetic lined up perfectly.

| exact-Sigma layers | MSE |
|---|---|
| none | 2.82e-5 |
| six worst (16,18,22,26,30,32) | 9.23e-6 |
| all >= 14 (19 layers) | 5.28e-6 |
| all 32 | **2.82e-7** |

**3.05x, not 100x.** Patching sigma at isolated layers does not stick: you inject
the exact Sigma and the very next layer's propagation re-corrupts it. The
"six layers carry 67%" accounting measured where error is *generated*, not where
it *persists*. Sigma must be right at essentially every layer from ~5 onward,
which is 27 x 4.3e9 = 6x over the full budget.

### 10.2 Separating mean drift from sigma drift

Supplying oracle values from layer k onward:

| from layer | mean only | mean **and** sigma |
|---|---|---|
| 5 | 2.65e-5 | **4.21e-7** |
| 25 | 1.29e-5 | **3.08e-7** |
| 31 | 3.56e-6 | 2.55e-7 |

**Sigma does essentially all the damage.** With the exact mean handed over at
layer 24 and only our own sigma after it, MSE is 1.29e-5; add exact sigma and it
is 3.08e-7 — a 42x gap from sigma in the last eight layers alone. Mean drift is
not the problem; I had that backwards. Also: the first four layers contribute
almost nothing (oracle from layer 5 gives 4.21e-7 against 2.82e-7 for all).

Independent consistency check: final-layer sigma alone accounts for 1.76e-6, and
the sensitivity profile predicts 5.3%^2 x 6.3e-8 = 1.8e-6. Two separate
measurements, same number. The error model is closed.

### 10.3 Roberts-Yaida 1/n expansion — does not apply

RY expands the ensemble over **W** with inputs fixed. We have **W fixed** and
randomness over **x**. Different expansion. And the parameter controlling
non-Gaussianity in our setting is not 1/n but the number of effectively
independent terms in `h_{l+1,i} = sum_j W_ij a_lj`, i.e. the effective rank of
`Cov(a_l)` — which collapses to **2.7**. There is no small parameter at depth, so
there is nothing to expand in.

### 10.4 Learned correction on EMP's residual — falsified

Regressing the final-layer residual on cheap per-neuron features (t, |t|, sigma,
kappa3, kappa4, phi(t), Phi(t), Yhat, ||w||), trained on 5 MLPs, tested on 5:

    held-out R^2  = -0.118      (worse than predicting zero)
    in-sample R^2 =  0.177      (pure overfit)

The error is network-specific and high-dimensional, not a smooth function of
local observables. This kills the cheap version of the learned-map idea; it does
not touch the full version (an equivariant weight-space network trained
end-to-end on a large corpus), which sees vastly more input.

### 10.5 Why everything fails: the rank collapse is in the worst possible place

Every dead end in this document — cumulant propagation, low-rank mixed cumulants,
rank-truncated particles, conditionally-independent latents, RY perturbation
theory, learned corrections — fails for one shared reason.

The effective rank of `Cov(a_l)` falls from 165 to 2.7 across the network. That is:

* **too severe for perturbation theory.** Every 1/n_eff expansion (cumulants,
  Edgeworth beyond 4th order, RY) needs many effectively independent terms. At
  depth there are ~3, so the expansion parameter is O(1) and the series has
  nothing to converge to.
* **not severe enough for explicit representation.** A genuinely rank-3
  distribution could be carried exactly by particles or a grid. But rank
  truncation is catastrophic (rank-8 on the last four layers alone gives 600x the
  noise it was meant to beat), because the discarded 250 directions still carry
  enough variance to matter at the 1e-7 level.

**The distribution is simultaneously too low-rank to expand around and too
high-rank to write down.** That gap is the whole difficulty of this problem, and
nothing we tried crosses it.

### 10.6 What is still standing

* **Nonlinear random matrix theory / free probability** (Pennington-Worah) for
  the spectrum of `Cov(a_l)`. Attacks the per-layer covariance map itself, so a
  fix propagates to all 27 layers at once — the one thing §10.1 shows is
  required. Not tested.
* **Full equivariant weight-space learning.** Never propagates anything, so it is
  immune to §10.5 entirely. Needs a corpus and real training compute.
* **Characteristic-function propagation.** Avoids the divergent Edgeworth series
  by construction; trades a convergence problem for a discretisation one.

---

## 11. Round 7: the sigma problem, characterised completely

Running the high-upside experiments with compute unconstrained. Net effect: the
target moved from "impossible" to "one specific 6x improvement", and every route
to it except one is now closed.

### 11.1 The requirement is 0.5%, not 0.1%

Replacing propagated sigma with *sampled* sigma at increasing sample counts:

| N samples | sigma error | % budget | MSE | adjusted |
|---|---|---|---|---|
| 6,000 | 1.83% | 8.9% | 3.62e-6 | 3.62e-7 |
| 25,000 | 0.89% | 37.3% | 1.10e-6 | 4.12e-7 |
| **67,000** | **0.55%** | 99.9% | **2.81e-7** | 2.81e-7 |
| 400,000 | 0.22% | 596% | 1.48e-7 | (over budget) |

At sigma = 0.55% the MSE (2.81e-7) is already close to the oracle value
(2.21e-7). So the requirement is **~0.5% relative**, not the 0.1% quoted in §9 —
that figure came from injecting i.i.d. noise, which is harsher than the structured
error a real method makes. Our propagation gives ~3%. **The gap is 6x, not 30x.**

### 11.2 Sampling sigma never wins, at any budget

Best sampled-sigma result is 2.81e-7 at 100% of budget — worse than the 2.39e-7
our cubature submission already scores. The MSE improves as 1/N but the budget
multiplier grows just as fast. Closed.

### 11.3 In the recursion, sigma is needed almost everywhere

Supplying oracle sigma only to neurons below a |t| threshold:

| tau | % of neurons | MSE | gain |
|---|---|---|---|
| 1 | 32% | 9.76e-6 | 2.75x |
| 2 | 54% | 8.48e-7 | 31.6x |
| **3** | **73%** | **2.44e-7** | **109.7x** |
| all | 100% | 2.43e-7 | 110.2x |

Fixing |t| < 3 recovers **99.5%** of the achievable gain. Note this contradicts
the one-shot sensitivity test (§9.5b), which found |t| > 2 neurons contribute
~zero. The resolution: one-shot perturbation measures `dY/dsigma = phi(t)`, which
really is ~0 at large |t|; but inside the recursion those neurons' Y feeds the
mean of every downstream layer, and *that* channel does not vanish. **Local
sensitivity is the wrong diagnostic for a recursive estimator.**

### 11.4 The error is scatter, not bias — calibration is dead

Decomposing the sigma ratio by (layer, |t| band) over layers 24-32:

| \|t\| band | bias | scatter across MLPs |
|---|---|---|
| [0, 0.5) | 0.0061 | 0.0120 |
| [1, 1.5) | 0.0019 | 0.0130 |
| [3, 4) | 0.0180 | 0.0146 |

Per-(layer,|t|) calibration fitted on 5 MLPs, tested on 5 held out: **0.96x** —
no gain. Combined with the per-layer calibration failure in round 2, the
conclusion is firm: **there is no removable systematic component.** What remains
is genuine per-network, per-neuron scatter of ~3% coming out of the Gaussian
bivariate ReLU moment.

### 11.5 Where this leaves it

    oracle sigma:  MSE 2.21e-7 at <=10% budget  ->  adjusted 2.21e-8  ->  rank #2
    our sigma:     MSE 2.03e-5 at <=10% budget  ->  adjusted 2.03e-6

Everything else in the estimator is solved. The single remaining requirement:

> **Reduce the per-neuron scatter in propagated Var(h_l) from ~3% to ~0.5%.**

Ruled out for achieving it: sampling (§11.2), calibration (§11.4), targeting a
neuron subset (§11.3 — you need 73% of them), patching individual layers
(§10.1), Roberts-Yaida perturbation (§10.3), and learned corrections (§10.4).

What is left is a genuinely better bivariate moment `E[a_i a_j]` for
non-Gaussian `(h_i, h_j)` — i.e. free probability / nonlinear RMT for the
covariance map, which is the one untested item that fixes all 27 layers at once
rather than patching.
