# Factorized-K3 depth tradeoff for the connected-cubic control

## Question

Does an earlier layer provide a useful crossing point between:

1. cheaper and more accurate factorized cumulant transport; and
2. enough oracle connected-cubic control headroom to improve the final score?

The audit uses 100M-sample oracle moment caches and factorized SIMPLE \(k=3\)
rollouts. State accuracy and control MSE are measured on IDs 160--167.
Calibration scales used in the final shrinkage test are learned independently
on IDs 100--107.

## Factorized post-state accuracy

| Layer | Mean rel. error | Cov. rel. error | C21 rel. error | C21 cosine | Full-matrix scale | Scaled C21 error | Dominant-dir. scale | Dominant-dir. scaled error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.002 | 0.031 | 0.324 | 0.955 | 1.155 | 0.298 | 1.211 | 0.043 |
| 16 | 0.003 | 0.035 | 0.337 | 0.955 | 1.203 | 0.296 | 1.252 | 0.104 |
| 20 | 0.004 | 0.036 | 0.353 | 0.959 | 1.280 | 0.283 | 1.335 | 0.052 |
| 24 | 0.005 | 0.040 | 0.352 | 0.967 | 1.334 | 0.256 | 1.375 | 0.095 |
| 27 | 0.006 | 0.043 | 0.377 | 0.958 | 1.342 | 0.288 | 1.381 | 0.150 |
| 29 | 0.007 | 0.045 | 0.368 | 0.964 | 1.362 | 0.264 | 1.402 | 0.111 |

“Dominant directions” are the rank-4 SVD directions of the factorized C21.
The apparently excellent 4--15% directional error is misleading for the
control. The useful Kerdock discrepancy is a small residual between much
larger moment terms, so a few-percent anchor error can dominate it.

There is a second sensitivity: the deployable pointwise control is centered
on the factorized mean, not the oracle mean. Although mean relative error is
only 0.2--0.7%, changing the center shifts a connected third moment
materially. For factorized-mean centering, independently calibrated scales
and evaluation-optimal scales were:

| Layer | Calibration IDs 100--107 | Evaluation IDs 160--167 |
|---:|---:|---:|
| 12 | 1.150 | 1.236 |
| 16 | 1.163 | 1.263 |
| 20 | 1.571 | 1.424 |

The 7--10% cross-set drift is far outside the required anchor tolerance.

## Runtime and analytical FLOPs

The FLOPs are from the vendor's exact cached polynomial for width 256,
SIMPLE/factorized \(k=3\). Timings are cumulative local float64 rollout times
on IDs 160--167.

| Layer | Local seconds | Estimated 4x-slower seconds | Propagation FLOPs | Added to 172.67B Kerdock |
|---:|---:|---:|---:|---:|
| 12 | 1.38 | 5.5 | 51.0B | 223.7B total |
| 16 | 2.18 | 8.7 | 87.4B | 260.0B total |
| 20 | 3.11 | 12.4 | 133.4B | 306.1B total |
| 24 | 4.25 | 17.0 | 189.2B | 361.9B total |
| 27 | 5.39 | 21.6 | 237.4B | 410.1B total |
| 29 | 6.37 | 25.5 | 272.6B | 445.3B total |

Thus earlier transport can fit the wall-clock limit, but it is not cheap in
the analytical-FLOP score.

## Oracle control ceiling

Target-free pointwise coefficients are fitted across held-out whole Kerdock
bases. Only the anchor/state is oracle in the ceiling rows.

| Layer | Best sample-direction oracle anchor | Factor-direction oracle anchor |
|---:|---:|---:|
| 12 | 0.719x (6/8 wins) | 0.673x (7/8) |
| 16 | **0.600x (8/8)** | 0.663x (6/8) |
| 20 | 0.574x (8/8) | **0.566x (8/8)** |

If score cost were proportional to total FLOPs, the approximate
MSE-times-cost ratios for the best oracle cases would be:

- layer 12: \(0.673(223.7/172.7)=0.871\);
- layer 16: \(0.600(260.0/172.7)=0.903\);
- layer 20: \(0.566(306.1/172.7)=1.003\).

So layer 12 or 16 would be economically viable **if** the anchor were nearly
exact. Layer 20 is already at break-even even at the oracle ceiling.

## Fixed independent calibration and correction shrinkage

The factor-derived-direction endpoint was harmful, so the completed control
prediction was shrunk toward the original 129-basis estimate:

\[
\widehat\mu_\lambda
=\widehat\mu_0+\lambda(\widehat\mu_{\rm control}-\widehat\mu_0),
\qquad
\lambda\in\{.05,.1,.2,.3,.5,.75,1\}.
\]

Anchor scales were frozen from IDs 100--107 before scoring IDs 160--167.

| Layer | Frozen anchor scale | Best correction multiplier | MSE ratio | Wins | Worst |
|---:|---:|---:|---:|---:|---:|
| 12 | 1.150 | 0.20 | 0.9569x | 6/8 | 1.113x |
| 16 | 1.163 | 0.05 | 1.0715x | 5/8 | 1.909x |
| 20 | 1.571 | 0.05 | 1.2661x | 3/8 | 2.424x |

The precommitted gate for opening IDs 168+ was below 0.95. No layer passed,
so no holdout was consumed.

Even layer 12's weak 4.3% MSE signal cannot pay its 29.5% propagation-FLOP
increase: the MSE-times-cost proxy is approximately 1.24x.

## Decision

No tested depth makes the current factorized connected-cubic state deployable.

- Layer 12 is the only factorized endpoint whose first derivative points
  weakly downhill, but the gain is too small for its cost and misses the
  holdout gate.
- Layer 16 is the best *oracle* economic depth, but the transported anchor's
  first derivative is already harmful.
- Layer 20 and later have more statistical headroom but cannot pay their
  propagation cost, even before accounting for anchor error.

The next useful improvement would need to reduce contracted anchor error by
roughly an order of magnitude and stabilize the mean-centered C21 amplitude
across networks. Moving the same factorized state earlier or shrinking its
final correction does not achieve that.
