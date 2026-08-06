# Disjoint-basis and jackknife anchors (2026-07-29)

## Question

Can the 129 Kerdock basis blocks be treated as internal replicates, so that
late-layer preactivation cumulants estimated on some bases provide an anchor
for a cubic control evaluated on disjoint bases?

The late rank-\(r\) statistic is

\[
z_k(x)=
\frac{(u_k^\top h(x)^{\odot 2})(v_k^\top h(x))}{r_{\rm sph}^2}.
\]

With the true anchor \(a=E[z]\), the desired estimator is

\[
\widehat\mu=Qf-\beta^\top(Qz-a).
\]

The directions and all regression coefficients in the experiments below use
only pointwise activations and final outputs. Targets are used only for the
reported MSE.

## Algebraic limits

Write \(F_b,Z_b\) for the final-output and cubic-feature mean in basis block
\(b\). Split the bases into pilot \(A\) and evaluation \(C\). A direct
cross-basis estimator is

\[
\widehat\mu_{C\leftarrow A}
=\bar F_C-\beta_A^\top(\bar Z_C-\bar Z_A).
\]

If the bases are exchangeable under the isotropic network ensemble and
\(\beta_A\) is fixed or independently estimated, both block means have the
same expectation, so this is ensemble-unbiased. It is nevertheless a poor
replacement for the full 129-basis average: even in the idealized case where
the control explains all block error, its remaining error is the pilot-anchor
error based on \(|A|<129\) blocks. It gives up the variance reduction and
degree-five cancellation of the full design.

A symmetric \(K\)-fold average does not retain a first-order control
correction. For equal fold sizes and a fixed coefficient,

\[
\frac1K\sum_j(\bar Z_j-\bar Z_{-j})=0.
\]

Only fold-to-fold variation in the fitted coefficient/directions survives.
That is a second-order, noisy correction. Unequal 129/K fold sizes introduce
only a tiny deterministic remainder.

One can retain the full baseline and add a zero-sum contrast,

\[
\widehat\mu_{\rm contrast}
=\bar F-\lambda\beta_A^\top(\bar Z_C-\bar Z_A).
\]

But under exchangeability,

\[
\operatorname{Cov}\!\left(
\bar F,\bar Z_C-\bar Z_A
\right)=0.
\]

The pilot and evaluation contributions to the covariance cancel. A gain
requires a stable non-exchangeable basis mode. The held-network experiment
below found none; the large selection-set gain was chance.

Finally, an exact cross-basis U-statistic can remove finite-sample plug-in bias
from products in a cumulant. It still estimates the cumulant of the empirical
Kerdock basis mixture, not that of the sphere. It therefore cannot reconstruct
the missing angular degrees (six and above). A delete-fold jackknife is the
first-order version of this correction for the nonlinear Edgeworth map.

## Implemented protocols

The executable harness is `arc_ceiling/eval_basis_jackknife_anchor.py`.

It tests:

1. `cross_raw`: directions, raw cubic anchor, and ridge coefficient fit on the
   complement of each held fold;
2. `cross_edgeworth_third`: preactivation moments estimated on the complement
   and mapped through the third-cumulant Edgeworth closure;
3. `jackknife_edgeworth_third`: \(K T(\widehat P) -
   (K-1)\operatorname{mean}_j T(\widehat P_{-j})\);
4. a direct asymmetric pilot/evaluation estimator;
5. a full-baseline plus pilot/evaluation contrast.

The fixed protocol used layer 29, rank 4, rotation seed 3, partition seed
20260729, and ridge 0.1. The apparent best contrast on IDs 160--167
(64 pilot bases, shrinkage 0.2) was frozen before IDs 168--183 were run.

## Results

| Estimator | Selection IDs 160--167 | Held IDs 168--183 | Held wins | Held worst |
|---|---:|---:|---:|---:|
| Symmetric raw cross-fit, 2 folds | 1.0328x | 1.0250x | 5/16 | 1.32x |
| Disjoint third-Edgeworth, 2 folds | 1.0366x | 1.0257x | 5/16 | 1.29x |
| Third-Edgeworth delete-2 jackknife | 0.9706x | 1.0300x | 5/16 | 1.13x |
| Full + pilot64 contrast, scale 0.2 | **0.7807x** | **1.0661x** | 5/16 | 1.73x |
| Direct asymmetric pilot64 | 1.5433x | 2.2867x | 1/16 | 5.39x |

The held bootstrap intervals were:

- raw cross-fit: [1.0034, 1.0560];
- disjoint third-Edgeworth: [1.0024, 1.0580];
- Edgeworth jackknife: [0.9951, 1.0668];
- frozen contrast: [0.9512, 1.2061].

The jackknife anchor changed the full plug-in anchor by only
\(8.7\times10^{-6}\) of its norm on average (range
\(5.8\times10^{-7}\) to \(1.9\times10^{-5}\)). This is consistent with it
removing only a tiny nonlinear plug-in term rather than the Kerdock-versus-
sphere quadrature discrepancy.

## Cost

No method requires extra network evaluations, but its state computation is
not free:

- a naive pilot \(C_{21}\) construction needs dense \(H^\top H\) and
  \((H^{\odot2})^\top H\) products;
- a deployable rank-4 version could use matrix-free randomized SVD and reduce
  this to a few \(H/H^{\odot2}\) block multiplies;
- the third-Edgeworth moment state needs at least second- and third-order
  256-by-256 contractions, and delete-fold jackknifing repeats them.

Because the held MSE signal is non-positive, optimizing those costs is not
warranted.

## Decision

Close the single-cloud disjoint-basis/jackknife/U-statistic branch.

The reason is structural, not merely empirical: symmetric reuse cancels the
first-order correction; asymmetric reuse discards the full design's variance
reduction; zero-sum contrasts are orthogonal to the full mean under basis
exchangeability; and U-statistics correct plug-in bias but not the missing
spherical angular content. A useful anchor must introduce information not
contained in a repartitioning of the same 129 basis blocks (for example,
transported weight-state information that is independent of this rotation).
