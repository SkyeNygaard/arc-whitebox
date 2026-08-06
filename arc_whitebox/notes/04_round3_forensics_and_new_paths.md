# Round 3: leaderboard forensics, frozen holdouts, and the honest frontier

Date: 2026-07-27

This round replaced the earlier four-network projections with official
Phase-1 Mini data, disjoint selection/holdout protocols, the contest-matched
`whestbench==0.13.0` / `flopscope==0.9.1` stack, and submission-shaped tests.

## Executive result

There are two different targets:

1. The prior honest local result in this repository was an anchored sphere
   sampler projected at about **4.91e-7 adjusted**. A two-design,
   radial-Rao--Blackwellized scrambled Sobol estimator now reaches
   **about 3.98e-7 adjusted** on frozen MLPs 50--99. This is a real improvement.
2. Submission 319679 reports **1.2350105e-8 adjusted** from
   **3.1233099e-8 raw MSE**, but its public ledger reports only about **13.0M
   instrumented FLOPs** against **108.8B effective compute**. More than 99.98%
   of its charged work is therefore residual wall time. That signature matches
   the publicly disclosed raw-array accounting path; it is not evidence for a
   40x-better white-box estimator.

The safe conclusion is not that the 40x algorithm has been reverse engineered.
It is that the public number is not a meaningful algorithmic target under the
intended FLOP accounting. No accounting bypass is implemented here.

## Protocol

- Dataset: official Phase-1 Mini targets generated with 1e9 reference samples.
- Model shape: width 256, depth 32, no biases.
- Selection sets are stated in each artifact. Hyperparameters were frozen
  before opening the corresponding holdout.
- Main sampling holdout: MLP IDs 50--99.
- K3 mechanism holdout: MLP IDs 50--59.
- Score: `raw MSE * max(0.1, effective_compute / 2.72e11)`.
- FLOPs use multiply-plus-add = two FLOPs, matching Flopscope 0.9.1.

## Best honest sampler

The network is positively homogeneous, so Gaussian radius can be integrated
exactly. Inputs are spherical directions scaled by `E[chi_256]`; antipodes are
paired. After an initial validated two-frame result, a strict whole-MLP
selection/holdout study found that two unwhitened, independently scrambled
Sobol sphere streams use the budget more effectively.

Frozen design:

- block A: 32,768 total directions, scramble seed 101;
- block D: 30,000 total directions, scramble seed 404;
- output: `0.4922223 * mean_A + 0.5077777 * mean_D`;
- same universal direction blocks for every MLP.

The stream weights were fitted on whole MLPs 0--49 and frozen before opening
IDs 50--99. Exact contest-matched results:

| quantity | result |
|---|---:|
| raw final-layer MSE, strict IDs 50--99 | 3.7429937e-7 |
| raw final-layer MSE, full Mini 100 | 3.5680948e-7 |
| mean effective compute, all 100 | 2.6380423e11 |
| mean budget multiplier, all 100 | 0.9698685 |
| adjusted score, strict IDs 50--99 | **3.6309259e-7** |
| adjusted score, full Mini 100 | **3.4606995e-7** |
| worst per-MLP effective compute | 2.64254e11 |
| minimum hard-budget margin | 7.746e9 |
| failures | 0 / 100 |
| improvement over prior 4.91e-7 | **1.42x overall / 1.35x strict** |

This is an exact local `whestbench==0.13.0` score, not a leaderboard result.
The serialized-design package and its report are under
`submissions/two_nearfull_rqmc/`.

Two intermediate packages remain for robustness comparisons:

- one 32,768-direction stream: 3.6244436e-7 all-100 and 3.7700910e-7 strict;
- two covariance-whitened frames: 3.8390198e-7 all-100 and 3.9756465e-7 strict.

The clean per-MLP randomized rank-1-lattice version also remains valuable: it
has an exact unbiasedness argument, no fixed-design bias, and the public
write-up reports about 4.10e-7. The shipped Sobol assets are fixed after offline
scrambling and therefore give up that exact per-run guarantee for the measured
held-out gain.

## New white-box mechanism: sketched K3 carrier shrinkage

Official factorized K3 propagation carries many CP columns. The successful
variant:

1. gauge-balances the three carrier legs;
2. samples 256 columns with probability proportional to squared balanced leg
   norm;
3. forms an unbiased importance-weighted covariance sketch;
4. extracts its top-64 shared subspace;
5. shrinks each carrier leg's orthogonal component by 0.75; and
6. fuses the shrink transform into the next K3 weight contraction, avoiding an
   explicit projection of every carrier column.

`m=256`, rank 64, residual scale 0.75, and sketch seed 2026 were selected on
IDs 0--9. IDs 50--59 were then opened once:

| method | mean final target MSE | ratio |
|---|---:|---:|
| official exact K3-simple baseline | 4.4663014e-5 | 1.000 |
| true full-K3 basis shrink | 2.6897683e-5 | 0.602 |
| K2-covariance proxy basis | 4.0382343e-5 | 0.904 |
| **energy-weighted K3 sketch** | **2.4724021e-5** | **0.554** |

The sketch improved 9/10 holdout MLPs, cut aggregate error **44.6%**, and had a
median per-network ratio of 0.518. Estimated added compute is 8.91B FLOPs
(3.27% of budget); measured runtime was 1.28x baseline.

This is a robust mechanistic advance, but not yet a competitive standalone
estimator: its absolute MSE remains orders of magnitude above RQMC. Its best
next use is as a compressed joint-state representation inside a stronger
trajectory-calibrated K3/K4 chain, or as a prior/control for sampling.

## What failed, with falsification margins

| path | frozen result | decision |
|---|---:|---|
| terminal Gaussian fit | 2.0285e-6 vs direct 9.2199e-7 | reject |
| terminal K3 Edgeworth | 1.0712e-6 | reject |
| terminal K3+K4 Edgeworth | 9.4840e-7 | reject |
| sensitivity-aligned Sobol rotation | 3% worse on IDs 50--69 | reject |
| learned implied-sigma closure, 800/200 | 2.178e-5 external calibrated rollout | reject |
| learned K2 residual rollout | 2.666e-5 vs K2 8.366e-5 | useful 3.14x, still far away |
| stochastic K3 column resampling | cap 1,536 adds about 5e-5 MSE; 6,144 slower than exact | reject |
| cross-output identity/ridge denoising | CV chooses direct; best incidental test gain 1.32% | reject |
| K3-sketch prior blended into RQMC | raw gain 2.15%, but cost-adjusted gain 0.47x | reject |
| spherical Stein directional CV | 2.10x equal-cost MSE; correction itself near no-op | reject |
| random 2D-plane angular integration, 4 angles | 1.66x Sobol variance on disjoint IDs | reject |
| random 2D-plane, 8 / 16 angles | 2.28x / 4.17x Sobol variance | reject |
| nested Richardson extrapolation | 9.31e-7 / 9.57e-7 vs 9.22e-7 | reject |
| exact covariance/tight-frame rotation | unbiased rotated construction worse | reject |
| adaptive stream allocation | catches one outlier, but 3.7939e-7 vs fixed A 3.7639e-7 | reject |
| train-selected Sobol seed grid | seed 2 wins train, regresses to 4.3757e-7 test | reject |
| depth-staged K3-augment sketch shrink | incomplete at runtime bound; recovered ID0 gain 7.9%, no holdout opened | do not promote |

The terminal smoothing result is especially diagnostic: sampling error is
shared directional quadrature error accumulated through the whole trajectory,
not independent noise in the last preactivation marginal.

The learned closures show the complementary deterministic failure: local
features predict a one-step correction, but their free rollout leaves the
training manifold and accumulates depth error. More data and a larger local
MLP did not fix that state insufficiency. A successful learned method must
carry a higher-order joint state and train on its own rollout distribution.

## Why 1.24e-8 is not currently an honest target

The public Flopscope issue describes NumPy-backed raw array operations doing
real work with zero instrumented FLOPs, leaving only
`1e11 * residual_seconds` in effective compute. It also explains why hardware
throughput can then change the score by more than 10x. Submission 319679 has
the corresponding public signature:

- mean instrumented FLOPs per MLP: roughly 12.97M;
- mean effective compute: 108.82B;
- instrumented share: roughly 0.000119;
- raw final-layer MSE: 3.1233e-8.

Adjacent submissions with the same 13M-FLOP skeleton have raw errors around
5.5e-8 to 7.0e-8, making 319679 look like a high-throughput stochastic outlier,
not a reproducible 13M-FLOP analytic method.

The official cumulant-propagation paper likewise reports polynomial-in-depth
error accumulation for fixed truncation order and leaves a depth-independent
sample-free solution open. That is consistent with every honest experiment
here: no tested local closure approaches 1e-7 raw, much less 1e-8.

## Highest-upside next research

1. Put the energy-weighted K3 sketch inside the trajectory-calibrated moment
   chain and train calibration on free rollouts, not teacher-forced local
   states.
2. Extend the same shared sketch to K4 without materializing an `n^4` tensor.
   The oracle study says K3+K4 is the relevant truncation; K5/K6 regress.
3. Compress the K3 chain much further before revisiting it as an RQMC prior.
   A frozen global blend did reduce raw holdout MSE 2.15%, but exact K3 plus
   the sampler exceeded the budget and more than doubled cost-adjusted score.
4. Keep the clean randomized lattice as the submission fallback. It is robust,
   auditable, and near the observed unbiased accuracy-per-FLOP frontier.

## Reproduction map

- sampling: `scripts/eval_sampling_official.py`,
  `scripts/eval_nested_qmc.py`, `scripts/eval_multistream_rqmc.py`,
  `scripts/eval_two_nearfull_rqmc.py`;
- K3 sketch: `scripts/factor_k3_fused_proxy_ablation.py`,
  `scripts/eval_k3_sketch_basis.py`,
  `results/k3_sketch_basis.json`;
- learned closures: `scripts/implied_sigma_closure.py`,
  `scripts/learn_k2_residual.py`;
- terminal smoothing: `scripts/terminal_marginal_smoothing.py`;
- output denoising: `scripts/cross_output_denoising.py`;
- deterministic-prior blend: `scripts/moment_prior_rqmc_blend.py`;
- spherical Stein CV: `scripts/eval_spherical_stein_cv.py`;
- random-plane integration: `scripts/eval_random_plane_disjoint.py`;
- raw artifacts: `results/`.
