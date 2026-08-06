# Observable-projected K3/K4 skeleton experiment

This is the concrete follow-up to the field-theory proposal in note 05. It uses
the K2/Gaussian trajectory as a dressed two-point propagator, evaluates the
leading connected Hermite tree diagrams, and transports their local Edgeworth
corrections through the linearized response to the scored final means.

## Computable surrogate

For a Gaussian preactivation `h_i = mu_i + sigma_i z_i`, expand its ReLU as

```text
ReLU(h_i) - E ReLU(h_i)
  ~= a1_i He1(z_i) + a2_i He2(z_i) + a3_i He3(z_i),

a1_i = sigma_i Phi(t_i)
a2_i = sigma_i phi(t_i) / 2
a3_i = -sigma_i t_i phi(t_i) / 6,
t_i = mu_i / sigma_i.
```

Let `rho` be the K2 preactivation correlation matrix and let `W` be the next
weight matrix. Define

```text
S_oi = sum_j W_oj a1_j rho_ij.
```

Keeping the leading connected trees gives marginal preactivation cumulants

```text
kappa3_o ~= 6 sum_i W_oi a2_i S_oi^2

kappa4_o ~= 24 sum_i W_oi a3_i S_oi^3
          + 48 sum_ij U_oi rho_ij U_oj,

U_oi = W_oi a2_i S_oi.
```

These are the `(2,1,1)` K3 tree and the `(3,1,1,1)` star plus
`(2,2,1,1)` K4 path. The expensive operations are only two `n x n` matrix
products per layer. This is a genuine connected-joint correction: unlike the
discarded diagonal cumulant approximation, it retains cross-neuron paths
through `rho`.

The local mean insertions are the first Edgeworth terms

```text
delta3 = (kappa3 / 6)  d_mu^3 E[ReLU(N(mu,sigma^2))]
delta4 = (kappa4 / 24) d_mu^4 E[ReLU(N(mu,sigma^2))].
```

They are transported through the K2-linearized Jacobians
`J_l = diag(Phi(t_l)) W_l`. This is a first-order skeleton expansion: the
internal covariance lines are dressed self-consistently by K2, while the K3/K4
vertices are inserted once. This use of connected diagrams and finite-width
ordering follows the logic of
[Dyer and Gur-Ari](https://arxiv.org/abs/1909.11304) and
[Yaida](https://proceedings.mlr.press/v107/yaida20a.html), but conditions on
the realized weights and computes neuron-specific observables.

## Frozen experiment

Global K3/K4 coefficients were fit on official Mini IDs 0--9. Selection was
evaluated by leave-one-network-out prediction. Because every tested response
rank cleared the pre-registered 10% gate, the coefficients were frozen and
official IDs 80--89 were opened once.

| response | selection LOONO ratio | frozen holdout MSE | holdout ratio | improved |
|---|---:|---:|---:|---:|
| exact | 0.3545 | 3.60199e-5 | 0.3124 | 10/10 |
| rank 8 | 0.3627 | **3.56211e-5** | **0.3090** | 9/10 |
| rank 16 | 0.3655 | 3.64003e-5 | 0.3157 | 10/10 |
| rank 32 | 0.3589 | 3.62399e-5 | 0.3144 | 10/10 |

Holdout K2 baseline MSE was `1.1528430e-4`. The frozen rank-8 coefficients were
`2.424834 * K3 + 6.247454 * K4`; the coefficients greater than one indicate
that omitted loop and higher-Hermite diagrams have the same average direction
as the retained trees.

Within-network signal/residual correlation was 0.761 for rank 8 and 0.767 for
the exact response on holdout. This is much stronger evidence than an aggregate
gain driven by a few networks.

## Is the backward response compressible?

Yes early, no late:

| rank | response energy, layers 0--23 | layers 24--31 | all layers |
|---:|---:|---:|---:|
| 8 | 0.880 | 0.368 | 0.752 |
| 16 | 0.969 | 0.553 | 0.865 |
| 32 | 0.997 | 0.736 | 0.932 |

For the actual inserted correction rather than generic Frobenius energy,
rank-8 retained about 85% of K3 and 84% of K4 early-layer effects, but only
37%/36% late. Despite that, final accuracy is essentially unchanged. The low
rank projection is acting partly as regularization; it is not needed for
compute because exact response transport is already cheap.

## FLOPs

Using the challenge convention (multiply and add count separately):

| component | FLOPs |
|---|---:|
| existing K2 forward lower bound | 2,147,483,648 |
| K3 shared `S` products, 31 layers | 1,040,187,392 |
| K4 path products, 31 layers | 1,040,187,392 |
| exact backward response products, 31 layers | 1,040,187,392 |
| **matrix-product subtotal** | **5,268,045,824** |

Elementwise Hermite, Edgeworth, correlation-normalization, and reduction work
is below 0.1B by a conservative direct operation count. Thus the deployable
exact-response version is below 5.4B, or 1.99% of the 272B budget. It uses only
ordinary PyTorch tensor operations; no NumPy compute or accounting bypass is
needed. The SVDs in the research script are diagnostics and are not required
by the exact-response estimator.

## Decision

The mechanism is strongly validated but does **not** beat the current score.
At the 0.1 scoring floor, the best holdout result corresponds to an adjusted
score near `3.56e-6`, about ten times worse than the current RQMC result.

- Promote connected Hermite trees as a cheap, useful K2 correction.
- Do not spend effort optimizing backward-response rank: exact transport costs
  only 1.04B and performs equivalently.
- The remaining error is diagram/closure error, not response compression.
  The next mathematical increment would be low-rank K3 triangle and K4 loop
  diagrams, or iterating the skeleton state so its correction changes later
  covariances. Either needs a new untouched evaluation block.

Reproduction:

- `scripts/eval_observable_skeleton.py`
- `scripts/eval_observable_skeleton_holdout.py`
- `results/observable_skeleton_selection.json`
- `results/observable_skeleton_holdout.json`
