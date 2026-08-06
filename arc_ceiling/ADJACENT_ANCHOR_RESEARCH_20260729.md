# Adjacent deep-cubic anchor research — 2026-07-29

## Decisive experiment: positive Hermite transport

For a bivariate standardized Gaussian `G`, replace the signed third-order
Edgeworth density by the positive pushforward

```text
Z_a = G_a + K3[a,b,c] (G_b G_c - delta_bc) / 6.
```

The map has the supplied third cumulant to first order.  Evaluating the ReLU
cubic after the map resums all powers of `K3`.  The covariance-renormalized
variant analytically removes

```text
Cov(Z) - I = K3[a,i,j] K3[b,i,j] / 18.
```

The pair expectation is evaluated with an 8x8 Gauss-Hermite rule.  Code:
`eval_quadratic_transport_anchor.py`.

Held-for-this-branch IDs 160–167:

| anchor | aggregate MSE ratio | wins | worst | anchor error / same-cloud |
|---|---:|---:|---:|---:|
| exact oracle post M21 | 0.54318 | 8/8 | 0.9002 | 0 |
| same-cloud | 0.96061 | 6/8 | 1.0637 | 1 |
| oracle pre K3, quadratic transport + covariance normalization | **0.53889** | **8/8** | **0.8778** | **0.0608** |
| factorized pre K3, same transport | 39.93 | 0/8 | 275.4 | 1.898 |
| target-layer sample/factorized global LS calibration + linear Edgeworth | 1.3562 | 3/8 | 3.683 | 1.107 |
| target-layer sample/factorized norm calibration + linear Edgeworth | 1.4377 | 2/8 | 4.776 | 1.043 |

The oracle result is important: a positive K3-only transport recovers even the
effective K4 behavior for this observable.  The negative result is equally
sharp: the current factorized K3 is not jointly compatible with the covariance
in nearly singular bivariate modes.  Pair-correlation floors from 0.99 down to
0 did not rescue it.  Aggregate target-layer self-calibration also overcorrects
(observed scales 1.44–1.87 in the first two networks).

## Ranked next algorithms

### 1. Adjoint / dual contracted-cumulant propagation

Do not propagate a generic `256^3` cumulant or even its whole `c21` slice.
After the main Kerdock forward supplies the four sample directions, form the
third-derivative tensor of only the four cubic observables,

```text
A_29,k = E_G[ D^3 F_k(Z_29) ] / 6.
```

Run the perturbative cumulant recursion backwards:

```text
A_l = (J_l tensor J_l tensor J_l)^* A_(l+1)
anchor_k = Gaussian_k + sum_l <generated_K3_l, A_l,k>.
```

Here `J_l = diag(Phi(mu_l / sd_l)) W_(l+1)`.  Compress each `A_l` after the
backward step with a randomized symmetric CP/Tucker or TensorSRHT sketch
(target rank 16–32).  Evaluate each generated-source contraction directly with
128–256 deterministic Gaussian/HOUT sigma points.  This is a Duhamel/adjoint
form of the cumulant hierarchy: irrelevant cumulant modes are never created.

Estimated incremental cost is roughly `1–2B` multiplies, versus about `173B`
for the current estimator and the much larger full factorized-K3 rollout.
The decisive gate is oracle mean/covariance on IDs 160–167: require mean anchor
error below `0.5x` same-cloud before replacing oracle moments with deployable
ones.  The positive Hermite transport above should be used only after this
contracted state passes.

### 2. Multi-checkpoint defect assimilation

The direct target-layer sample/factorized calibration failed because the
sample has exactly the quadrature error being corrected.  Calibrate or inject
defects several layers earlier, where Kerdock error is weaker, and transport
only their contraction to layer 29:

```text
delta_l = K3_sample,l - K3_rollout,l
correction_29,k = <delta_l, A_l,k>.
```

Use checkpoints such as layers 8, 12, 16, and 20 and a robust median or
precision-weighted combination.  The scalar calibration

```text
s_l = <K3_rollout,l, K3_sample,l> / ||K3_rollout,l||^2
```

can be computed without materializing sample `c21`:

```text
<F, K3_sample> = mean_x ((z(x)^2)^T F z(x)).
```

Eight Kerdock bases per checkpoint cost about `0.27B` multiplies, and three
checkpoints remain under `1B`.  Select checkpoints and aggregation on 100–107,
then freeze them for 168–175.  Do not calibrate at layer 29.

### 3. Observable-specific spherical Stein control

For a degree-one homogeneous normalized cubic `g(x)` on the sphere and fixed
direction `w`,

```text
C_w(x) = d (w^T x / ||x||) g(x) - ||x|| w^T grad g(x)
E_sphere[C_w] = 0.
```

This supplies an exact anchor with no cumulant model and directly targets kink
error.  Use one shared JVP direction chosen from a randomized range finder of
the four cubic gradients, rather than a random direction and final outputs as
in the prior Stein test.  The drawback is severe: a layer-29 JVP nearly doubles
network FLOPs.  This is only competitive if a one-direction pilot demonstrates
more than a twofold MSE reduction, or if a shallower checkpoint retains the
signal.  It ranks below the adjoint method because the earlier generic Stein
test was neutral at half cost and 2.10x worse than an equal-cost baseline.

## Literature connections

- Boris Hanin, *Random Fully Connected Neural Networks as Perturbatively
  Solvable Hierarchies*, JMLR 25 (2024): cumulants obey a layer-wise
  perturbative hierarchy.  The proposed adjoint computes one observable of
  that hierarchy rather than its full state.
- Easley and Berry, *A Higher Order Unscented Transform* (2020): rank-one
  tensor decompositions give sigma points matching skewness and kurtosis.  This
  is a principled replacement for Monte Carlo when evaluating generated-source
  contractions.
- Oates, Girolami, and Chopin, *Control Functionals for Monte Carlo
  Integration* (2016): Stein identities create exactly integrable
  gradient-based controls; the spherical identity above is the homogeneous
  specialization.
