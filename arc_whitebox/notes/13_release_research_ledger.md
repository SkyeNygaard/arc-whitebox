# Phase-1 research ledger and release checkpoint

## Release checkpoint

The current release candidate is the 66,048-point, rotation-seed-3
Kerdock/maximal-real-MUB spherical 5-design, propagated with depth-5
Strassen--Winograd multiplication.  The deepest two decoded levels remain a
quadtree and are assembled once, instead of paying for a block assembly at
every recursive node.

Package:

```text
submissions/kerdock_mub5_winograd_tree/submission.tar.gz
SHA-256 a7f5e1e58639192e33e0886e776b4c8392399a7879e372bed557811516ec93e7
```

The archive contains only `estimator.py`, `fast_matmul.py`,
`kerdock_mub5_seed3.npz`, and `manifest.json`; `whest validate-package`
passes.

The final isolated local subprocess audit on mini row 0 used one thread and a
60-second limit:

| measurement | result |
|---|---:|
| end-to-end duration | `22.940 s` |
| tracked FLOPs | `170,906,815,488` |
| residual wall time | `0.046099 s` |
| effective compute | about `175.52 B` |
| compute multiplier | `0.6452821` |
| raw final MSE | about `1.7292e-7` |
| adjusted score | about `1.12e-7` |
| failures / exhausted gates | `0` |

The earlier repeated full row-0 audit gave `175.871 B` effective compute and
`1.1180792e-7` adjusted score.  The difference is residual-time noise, not a
prediction change.

For comparison:

| checkpoint | raw MSE | effective compute | adjusted score |
|---|---:|---:|---:|
| dense Kerdock, official full 100 | `2.2825913e-7` | `268.899 B` | `2.2565646e-7` |
| ordinary depth-5 Winograd, row 0 | `1.7291806e-7` | `181.288 B` | `1.1525015e-7` |
| partial-tree Winograd, row 0 | `1.7292035e-7` | `175.52–175.87 B` | `1.115–1.118e-7` |

The raw row-0 differences among dense and fast multiplication are at the
float32 reassociation level.  Selection IDs 0--9 gave `1.7153620e-7` for the
ordinary Winograd implementation versus `1.7154893e-7` for dense propagation.

## What produced real gains

### Radial integration and deterministic spherical cubature

The bias-free ReLU MLP is positively homogeneous.  Integrating the Gaussian
radius analytically reduces the hard part to angular integration on the
sphere.  Antipodal designs then remove all odd spherical harmonics for free.

The maximal family of 129 real mutually unbiased bases in dimension 256,
including the coordinate basis, supplies 66,048 antipodal points and is an
exact spherical 5-design.  Its first layer can be evaluated with signed
Walsh--Hadamard transforms.  This was the main statistical breakthrough:
official full-100 adjusted score improved from the prior two-nearfull RQMC
checkpoint `3.4607e-7` to `2.2566e-7`.

### Rectangular fast matrix multiplication

The expensive products are `(66048,256) @ (256,256)`, not square matrix
products.  Recursive Winograd splitting was applied directly to the last two
axes without padding the tall dimension.  A hybrid schedule with three packed
levels and two depth-first levels was the best runtime-aware point.

The partial-output tree removes repeated decoded-block copies.  It lowered the
full estimator from `175.823 B` tracked FLOPs to `170.907 B`, and from about
`181.29 B` to `175.5–175.9 B` effective compute.

### Careful selection discipline

Rotation seed 3 was selected on IDs 0--49.  IDs 50--99 remained frozen during
subsequent design choices.  All basis/rotation subset selection was nested at
the whole-network level.  This exposed several apparent 2x wins as severe
network-level overfitting.

## Exact fast-multiplication paths audited and rejected

### Winograd tensor-leg permutations

All six tensor-leg permutations remain `4/4/7` addition circuits once input
encoders and their adjoint output decoders are counted correctly.  They have
identical tracked arithmetic; runtime variants did not beat the standard
orientation.

### Alternative-basis Strassen

The Schwartz--Vaknin alternative-basis formulas were implemented exactly,
including recursive basis changes.  They improve the asymptotic addition
constant but have larger finite-size copy/basis-change cost here.  Pure and
mixed schedules were worse than Winograd.

### Generalized five-coordinate rank-7 reuse

An exact sparse factorization reduced the best honest arithmetic to
`5.2208 B` tracked FLOPs per layer.  The saving is mathematically real, but
the tuple implementation has prohibitive wrapper/residual time.  Low-residual
stacked and mixed implementations project to at least `207.66 B` and
`199.25 B` over the deep layers, respectively, before first-layer overhead.
Neither beats the partial-tree release.

### Persistent recursive layout

Keeping a Morton/quadtree layout through every ReLU reduced tracked compute
further, to `168.403 B` on row 0.  Thousands of small decode operations raised
residual time to `0.435 s`, however, for `211.9 B` effective compute and about
`1.35e-7` adjusted score.  Packing the leaves did not remove the straight-line
decode overhead.

### Rank-48 multiplication and approximate border rank

The exact rational 4-by-4 rank-48 algorithm has lower asymptotic rank, but its
finite linear-transform count makes the favorable depth-2 estimate roughly
`5.73 B` per layer before runtime costs, already above the partial-tree
frontier.  Bini/APA border-rank constructions add coefficient growth and
float32 stability risk; they did not produce a release candidate.

## Statistical and geometric paths audited

### Sampling and quasi-Monte Carlo

Tested families included iid MC, antithetic sampling, scrambled Sobol,
Kronecker/rank-1 lattices, interlaced and nested RQMC, multi-stream blends,
radial sphere integration, tight frames, and random-plane angular rules.
Sphere/RQMC was a large improvement over iid MC, but the best prior packaged
two-nearfull blend still scored `3.4607e-7`, behind Kerdock.

Random 2D-plane rules, sensitivity-aligned rotations, Richardson/nested
extrapolation, and adaptive stream allocation all failed held-out gates.

### Sparse or weighted Kerdock rules

Unions of fewer antipodal MUB bases remain exact spherical 3-designs, but lose
the complete rule's degree-four cancellation.  Honest adjusted-score
selection improved monotonically through all 129 bases.  Representative
projected adjusted scores were about `1.50e-7` at 80 bases, `1.34e-7` at 112,
and `1.17e-7` at all 129 under the older Winograd bill.

Unrestricted selection across multiple rotations appeared to reach raw MSE
near `8e-8` in-sample, then regressed to roughly `3e-7` under network-level
cross-validation.  Coordinate exchange and learned positive harmonic weights
showed the same failure mode.

### Multifidelity orientation controls

A matched-pilot construction using a full seed-3 rule plus partial rotated
rules reached about `1.355e-7` raw MSE with 90,624 rows.  Its extra propagation
cost outweighed the raw gain.  Two full rotated Kerdock designs also improved
raw MSE, but not raw-MSE-times-compute.

### Moment propagation and cumulants

Gaussian closure, Edgeworth marginal corrections, official K2/K3 propagation,
low-rank conditional models, observable-projected skeletons, hierarchy
resummation, implied-sigma closure, and learned K2 residual rollouts were
tested.  The consistent lesson is that local one-step closures can look good
under teacher forcing and then leave their training manifold during a
32-layer rollout.

Oracle studies show that third- and fourth-order marginal information is
valuable, but the missing cross-neuron joint cumulants are the signal.
Diagonal cumulants, marginal-only Edgeworth corrections, or low-rank particle
truncation do not transport them accurately enough.  An energy-weighted K3
carrier sketch gave a real 44.6% reduction over the official K3 baseline, but
the absolute result remained far behind sampling/cubature.

### Terminal smoothing and output controls

Gaussian, K3, and K3+K4 fits to the last sampled preactivation; final-layer
Rao--Blackwellization; cross-output identities; ridge denoising; and spherical
Stein controls were tested.  The best terminal corrections did not beat direct
propagation after held-out selection.  The dominant error is shared angular
trajectory error accumulated through depth, not independent last-layer
marginal noise.

### Weight-adaptive geometry

Tested adaptations included first-layer moment/covariance transport,
layer-2 moment transport, weight-adaptive rotations, basis-variance selection,
sensitivity alignment, Laplacian/boundary corrections, and exact line-region
diagnostics.  None survived frozen evaluation.  The main failure is that 50
selection networks supply far fewer independent observations than their
12,800 neuron outputs suggest.

## Interpretation of the public leader

Submission 319679 reports raw MSE `3.12e-8`, only about `13 M` instrumented
FLOPs, and roughly `109 B` mean effective compute.  Its bill is therefore
almost entirely residual wall time rather than tracked analytical operations.
Adjacent variants with the same skeleton move materially in raw MSE.

This is not evidence for a reproducible 13-million-FLOP analytic closure.  It
is consistent with untracked high-throughput numerical work, stochastic
selection luck, or visible-suite adaptation.  The final contest ranking uses
a fresh sealed rerun, so the release policy here remains: optimize honest
unseen-network performance and do not key predictions to public identities.

## Best next research bets

1. **Rollout-trained equivariant moment carrier.**  Use the public 10,500-MLP
   marginal-cumulant corpus to train a shared, permutation-equivariant
   layer update on its own rolled-out states, not teacher-forced local moments.
   It must carry a joint low-rank/sketched state; a marginal-only recurrence is
   already falsified.
2. **Kerdock plus learned trajectory correction.**  Train only a low-amplitude
   correction to the stable 5-design estimate, using deployment-noise features
   and whole-network cross-validation.  This is safer than replacing the
   estimator with a free-running learned closure.
3. **Vectorized sparse-ReLU propagation.**  Deep layers contain nearly dead,
   nearly linear, and kink neurons.  A tracker-honest block-sparse kernel could
   compose with Kerdock if it lowers both tracked matmul work and wrapper time.
4. **Higher-degree cancellation with nested algebraic designs.**  Search for a
   structured degree-7 correction whose extra nodes are propagated only
   through a compressed control channel, rather than as another full design.
5. **Compiled/vectorized five-coordinate reuse.**  The exact sparse rank-7
   factorization has already proven a lower arithmetic count.  It becomes
   useful if its dictionary transforms can be expressed in a handful of
   tracker-visible bulk operations rather than tuple leaves or billed stacks.

