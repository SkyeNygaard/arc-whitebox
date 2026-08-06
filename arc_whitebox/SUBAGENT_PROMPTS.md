# Subagent prompts — Phase-1 research round

Ten independent research tasks. Each prompt below is self-contained: paste
**§0 Shared context** followed by the agent's own section.

---

## §0 Shared context (prepend to every agent prompt)

### The problem

Estimate `E_{x~N(0,I_256)}[ a_l(x) ]` — the per-layer **post-ReLU** activation
means — for a bias-free, He-initialised, 256-wide, 32-deep ReLU MLP, given only
its weights. The grader computes ground truth by sampling:
`x = relu(x @ w)` at every layer including the last, then averaging.

Scored metric:

```
score_m = raw_final_layer_MSE_m * max(0.1, C_m / B)
B   = 2.72e11
C_m = tracked_flops + 1e11 * residual_wall_seconds
```

The multiplier is **uncapped above 1.0**. Only the *final* layer's MSE is
scored; `all_layers_mse` is reported but not scored.

### Current state of the art in this repo

| checkpoint | raw MSE | effective compute | adjusted |
|---|---:|---:|---:|
| dense Kerdock, official full 100 | 2.2826e-7 | 268.9 B | 2.2566e-7 |
| depth-5 Winograd, row 0 | 1.7292e-7 | 181.3 B | 1.1525e-7 |
| partial-tree Winograd, row 0 | 1.7292e-7 | 175.5 B | ~1.12e-7 |

Best estimator: 66,048-point antipodal Kerdock / maximal-real-MUB spherical
**5-design** (129 real MUBs in dim 256, rotation seed 3), radius integrated
analytically via positive homogeneity, first layer evaluated by signed FWHT,
propagated with depth-5 Strassen–Winograd. `submissions/kerdock_mub5_winograd_tree/`.
Strict holdout raw MSE is **2.8064e-7** vs 2.2826e-7 in-sample — there is a real
generalisation gap; treat in-sample numbers as optimistic.

### Standing directive from the project owner

**Full budget is being used. Compute reduction / wall-time conversion is NOT a
valid scoring path — do not propose it.** Optimise MSE.

This still constrains you: the multiplier is linear in compute, so above the
floor `d(log score) = d(log MSE) + d(log C)`. A method that adds 20% compute
must cut MSE by more than 20% to be net-positive. Report both numbers always.

### Structural facts already established (do not re-derive)

1. **Rank collapse.** Effective rank of `Cov(a_l)` falls from 165 at layer 1 to
   **2.70** at layer 32. Perturbative expansions in `1/n_eff` have an O(1)
   expansion parameter at depth. This is *why* white-box methods degrade with depth.
2. Final-layer fluctuation is **99.1% predictable from layer 31**, but only
   R²=0.26 from degree-3 polynomials in the top-16 active input directions.
   Locality in depth is the exploitable structure; input-space smoothness is not.
3. Positive homogeneity makes the radial Gaussian integral exact; the hard part
   is purely angular. Antipodal designs kill all odd spherical harmonics for free.
   Kerdock is exact through degree 5, so the leading error is **degree 6**.
4. Exact Gaussian ReLU moments (incl. bivariate, nonzero-mean, Drezner–Wesolowsky)
   are implemented in `src/whest/gaussmath.py` and verified to 1.7e-16.

### Environment

```bash
# Grader toolchain: flopscope 0.9.1, whestbench 0.13.0, numpy 2.4.6, scipy. NO torch.
/Users/skyenygaard/Programming/AI-Safety/arc_whitebox/.venv/bin/python
/Users/skyenygaard/Programming/AI-Safety/arc_whitebox/.venv/bin/whest

# ARC reference cumulant-propagation repo (has torch), run via:
uv run --project /Users/skyenygaard/Programming/AI-Safety/arc_whitebox/vendor/mlp_cumulant_propagation python ...
```

Local evaluation:

```bash
.venv/bin/whest run --estimator path/to/estimator.py --n-mlps 10 --format json
.venv/bin/whest run --estimator path/to/estimator.py --dataset data/official_phase1_mini --split mini --runner local
```

Runtime sandbox for a real submission: `flopscope.numpy` + stdlib only. No numpy,
scipy, or torch at runtime. Offline-trained weights ship as a pickle-free `.npz`
and load free during `setup()`.

### Guardrails — these are hard-won, violating them has already cost real time

1. **`whest validate` is nearly worthless as a correctness check.** It verifies
   only output shape and finiteness. In this project it passed an estimator that
   scored 1.73 against a 6.97e-05 baseline — a 25,000× error caused by returning
   pre-activation instead of post-ReLU means. **Always** run `whest run` against
   a known baseline before believing anything.
2. **The target is post-ReLU at every layer.** An earlier stage in this project
   tuned its entire hyperparameter search against *pre-activation* means and
   produced a config that was invalid for the contest metric. In the moment files,
   `mean` is post-ReLU (never negative) and `pre_mean` is pre-activation (~50%
   negative). Check which one you are scoring against.
3. **Component accuracy is not additive — measure the composition.** Replacing an
   approximate post-ReLU covariance with the mathematically exact bivariate one
   made `all_layers_mse` 1.09× *better* and the scored `final_layer_mse` 0.815×
   *worse*. The crude approximation's error was partially cancelling a different
   missing term. Separately: damping a mean correction alone scored 0.904×,
   replacing a covariance alone scored 0.979×, but doing **both** scored 2.27×.
   Never conclude from a component's standalone accuracy.
4. **Check whether your optimum is interior.** An earlier search selected a
   parameter at the edge of its grid and shipped it; extending the grid found a
   materially better optimum in a different regime with a different mechanism.
5. **A real mechanism transfers almost exactly.** When the mechanism above was
   genuine, validation predicted 2.261× and 100 untouched networks delivered
   2.268×. Treat a large validation→holdout gap as evidence of overfitting, not
   noise.
6. **Three immutable stages, declared before you look:** 8-network screen → 24-network
   frozen validation → 50+ untouched networks. Never re-use a holdout. State your
   go/no-go thresholds in writing before running the final stage.
7. **Correct for reference-target noise.** The MC ground truth has its own
   variance; a 2–5% "effect" can be target noise. Quantify it before claiming.
8. **Integrity.** The public 1.235e-8 leaderboard score appears to exploit a
   flopscope raw-array accounting weakness (≈13M instrumented FLOPs vs 108.8B
   effective). Do not implement or propose accounting bypasses.

### Gates

- **Major theory path:** do not continue past screen unless an oracle or
  full-width probe suggests ≥1.3× raw-MSE improvement (prefer ≥1.5×).
- **Cheap additive method:** ≥5% adjusted-score improvement, a confidence
  interval excluding zero, and no severe per-network tail.

### Deliverable (identical for all agents)

A written report containing: the mechanism you tested; the oracle/screen result;
staged numbers with CIs; **raw MSE and effective compute separately** plus the
resulting adjusted score; per-network worst case; an explicit
CONTINUE / STOP recommendation against the gates above; and the exact commands to
reproduce. Negative results are valuable — report them fully rather than
searching for a variant that passes.

---

## Agent 1 — Residual-spectrum benchmark infrastructure (#91, #92)

**This is the highest-priority task and other agents depend on it. Do it first
and publish the harness.**

Build the measurement infrastructure that evaluates any analytic or learned
approximation `g` as a **control variate on the sampled function**, not as a
standalone predictor. The central claim to operationalise: a `g` that is 100×
too inaccurate standalone can still be highly valuable if it removes the specific
residual modes that Kerdock integrates badly.

Deliver:
1. A harness that takes any `g` and reports Kerdock cubature error on `f - g`,
   alongside standalone error of `g`, for the same networks and point set.
2. An oracle decomposition of current Kerdock error attributed by: layer, output
   PCA mode, spherical-harmonic degree (especially degree 6, the leading term
   given 5-design exactness), and first-layer moment defect.
3. A frozen, versioned network split (8 / 24 / 50+) that all other agents import,
   so nobody accidentally tunes on another agent's holdout.

Report which error channel actually dominates the final-layer MSE. That single
number should redirect the rest of the round.

---

## Agent 2 — Gaussian-closure residual Kerdock (#1, #2)

Construct a cheap sample-level surrogate `g(x)` consistent with Gaussian closure
whose Gaussian expectation is known analytically, then run Kerdock only on `f-g`.

Then extend it (#2): put the validated early covariance-eigenmode correction
*inside* `g`, and measure **residual Kerdock MSE**, not standalone closure accuracy.

Note the rank-collapse fact: by layer 32 the covariance has effective rank 2.70,
so a low-rank `g` may capture most of the integrand's angular structure. Use
Agent 1's harness. Exact Gaussian ReLU moments are already available in
`src/whest/gaussmath.py`.

Guardrail specific to this task: closure surrogates in this project have
repeatedly looked good standalone and failed in composition. Judge only on
`f - g` residual error.

---

## Agent 3 — Finite-width theory: explicit 1/n and adjoint-weighted defect (#51, #54)

Two linked pieces.

**#51:** derive and numerically test the leading finite-width `1/n` correction to
the infinite-width Gaussian mean at width 256. Validate the derivation by
measuring at widths 32/64/128/256/512 and checking the fitted power matches
theory — a correct mechanism must show the predicted scaling, not just fit at 256.

**#54:** compute the *oracle* local covariance defect (true minus Gaussian-closure
covariance), contract it with the final-output adjoint, and determine its
**effective rank**. If that rank is small (the rank-collapse result suggests it
may be), a low-rank correction could capture most of the recoverable error, which
would make several other families cheap.

The effective-rank number is the key deliverable — report it even if #51 fails.

---

## Agent 4 — Adjoint-derived and output-sensitive first-layer transport (#31, #32)

**#31:** derive the first-layer mean/covariance correction that minimises
*predicted final-output error*, rather than tuning a global 1× or 2× multiplier.
The existing work tuned a scalar; this asks for the correction that the adjoint
actually implies.

**#32:** match exact first-layer moments only in the 4–32 directions most
amplified by the downstream soft-gated Jacobian, rather than all 256 equally.

Prior experience directly relevant here: a global scalar multiplier on a
correction term had a sharply non-monotone optimum, and the correct value
depended entirely on what *else* was changed simultaneously. Expect interaction;
test the adjoint-derived correction jointly with whatever covariance treatment
you pair it with, not in isolation.

---

## Agent 5 — Exact first-layer Hermite subtraction with debiasing (#34, #40)

**#34:** compute exact degree-1 and degree-2 first-layer Hermite coefficients and
propagate their known contribution as a control variate.

**#40:** apply aggressive moment transport to all rows, then use a small untouched
subset to estimate and remove the induced bias.

The debiasing arm matters: aggressive transport is expected to be biased, and the
question is whether the bias is estimable cheaply enough to be worth the variance
reduction. Report the bias estimate with its own CI — if the debiasing correction
is noisier than the gain, say so.

---

## Agent 6 — Weight-conditioned Hermite and ridge control variates (#41, #45)

Train a **permutation-equivariant** model over the weight tensors that outputs
coefficients of an analytically integrable control function `g_W(x)`:

- **#41:** multivariate Hermite coefficients (degrees 1–4); all nonconstant
  Hermite terms have exactly known Gaussian expectations.
- **#45:** 4–16 projections `u_j` plus one-dimensional ridge functions `h_j` with
  exact Gaussian integrals, `g(x) = Σ_j h_j(u_j^T x)`.

Permutation equivariance is not optional — neuron ordering is arbitrary and a
non-equivariant model will memorise the training networks.

Hard-won guardrail: a learned model in this project (4,674 parameters, trained on
a related closure task) was reported as the source of a large gain. Ablation
showed it contributed **0.4%**, and at the properly tuned scalar it was actively
harmful. **Run the alpha=0 ablation first**, before any tuning: if the mechanism
works with the learned component switched off, you have found something simpler
and better. Budget the ablation as step one, not as a final check.

Runtime cost counts. Report inference FLOPs for the deployed control.

---

## Agent 7 — Stein-potential and signed neural control variates (#42, #44)

**#42:** learn a vector field `φ_W(x)` and use the Stein control
`∇·φ − x^T φ`, whose Gaussian expectation is exactly zero by construction — so it
cannot bias the estimate regardless of how badly the network is trained. That
zero-expectation property is the reason to prefer this family; verify it
numerically to machine precision before anything else.

**#44:** learn positive and negative control components separately (signed neural
control-variate construction).

Same ablation discipline as Agent 6: establish what a zero-capacity / analytic
baseline achieves before crediting the learned part. Report variance reduction on
`f - g` via Agent 1's harness, and the runtime FLOP cost of evaluating `φ` and
its divergence at every cubature point — the divergence term is easy to
underestimate and may dominate.

---

## Agent 8 — Signed network-dependent cubature (#21, #22, #26)

Kerdock's uniform weights are network-independent. Fit **signed** weights.

- **#21:** signed weights on the 129 Kerdock basis-block means, enforcing total
  weight one and exact low-degree constraints (preserve exactness through degree 5).
- **#22:** cross-fit — fit on half the basis blocks, apply to the other half,
  swap, average. This is the guard against selection bias, not an optional extra.
- **#26:** optimise signed weights for the residual *after* subtracting an analytic
  surrogate, not for the original network.

This family is "outside the theorem" — you are giving up the design guarantee in
exchange for network adaptivity, so the holdout discipline matters more here than
anywhere else. The in-sample→holdout gap on the existing design is already
2.2826e-7 → 2.8064e-7 with *no* fitting; expect worse with fitted weights, and
budget for it.

---

## Agent 9 — Preintegration and conditional integration oracles (#11, #17)

**#11:** fix 255 orthogonal coordinates and integrate the full network **exactly**
along one Gaussian direction by tracking one-dimensional ReLU breakpoints. The
network restricted to a line is piecewise linear with finitely many kinks, so
this is exactly integrable in principle. Establish the oracle value first — how
much error would vanish if one direction were integrated exactly? — before
worrying about cost or direction selection.

**#17:** condition on a low-dimensional approximation of the penultimate
activations and analytically integrate the final ReLU expectation. The 99.1%
predictability of the final layer from layer 31 makes this the most
theoretically motivated item in the entire list.

Both are oracle-first tasks: measure the ceiling, then decide whether a
deployable version is worth building. Do not build the deployable version during
this round if the oracle ceiling is below the 1.3× gate.

---

## Agent 10 — Multilevel surrogate residual estimator (#81)

Evaluate a cheap analytic surrogate on all cubature points and the exact network
on a smaller **coupled** subset, allocating cost by measured residual variance
(standard MLMC allocation).

This is the one construction in the round that is unbiased by design and
composes with every other agent's surrogate, so build it to accept an arbitrary
`g` — coordinate with Agent 1 on the interface.

Key measurement: the variance of `f - g` on the coupled subset, and whether the
optimal allocation actually beats spending the same compute on more full-fidelity
Kerdock points. Given the standing directive to use full budget, the question is
purely "does this allocation of a fixed budget beat the current allocation" —
report it that way.

---

## Coordination notes

- Agent 1 gates the round. Agents 2, 6, 7, 10 should consume its harness and its
  frozen splits; if they start earlier, they must not touch the 50+ holdout.
- Agents 3, 4, 5 overlap on first-layer / finite-width structure. Agent 3's
  effective-rank number should be shared with 4 and 5 as soon as it exists.
- **#100 (composition interaction matrix) is deliberately not assigned.** Run it
  only after this round, on the surviving methods. Every strong result in this
  project so far has come from an *interaction* between two changes, and the
  single largest error made was assuming component gains would add.
