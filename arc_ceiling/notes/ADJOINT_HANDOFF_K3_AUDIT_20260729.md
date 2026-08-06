# Adjoint-handoff K3 audit

## Result in one sentence

Backward contraction removes the full-C21 bottleneck and a rank-32 handoff
reproduces the full factor anchor to 2.31% in 10.338B exact flopscope FLOPs,
but every runtime-feasible source closure remains too inaccurate for the
arbitrary-center connected control.

## Dual identity

For a symmetric third cumulant \(K\), define

\[
\mathcal L_{A,b}(K)=K:\operatorname{Sym}(A\otimes b),
\qquad A=A^\top.
\]

The rankwise C21 observable is obtained with

\[
A=\operatorname{diag}(u_k),\qquad b=v_k,
\]

because

\[
\mathcal L_{\operatorname{diag}(u_k),v_k}(K)
=\sum_{ij}u_{ik}v_{jk}K_{iij}.
\]

If inherited K3 columns cross a layer through

\[
M_\ell=\operatorname{diag}(E[\operatorname{ReLU}'(Z_\ell)])W_\ell,
\]

the exact adjoint pullback is

\[
A\leftarrow M_\ell^\top A M_\ell,\qquad
b\leftarrow M_\ell^\top b.
\]

For a CP source

\[
S=\operatorname{Sym}\sum_r a_r\otimes c_r\otimes d_r,
\]

the scalar source contribution is

\[
\frac13\sum_r\left[
(a_r^\top A c_r)(d_r^\top b)
+(a_r^\top A d_r)(c_r^\top b)
+(c_r^\top A d_r)(a_r^\top b)
\right].
\]

The implementation reconstructs direct full-CP contractions to machine
precision. The corrected C21 diagonal convention matters:
`get_dslice((2,1))` has a zero diagonal and `get_dslice((3,))` supplies
\(K_{iii}\).

## One-time low-rank handoff

After processing a late dense tail, diagonalize the current symmetric adjoint
and retain the \(q\) eigenpairs with largest absolute eigenvalue:

\[
A=P\Lambda P^\top.
\]

All earlier pullbacks preserve that rank exactly:

\[
P\leftarrow M^\top P.
\]

On IDs 160--167, layer 29, two sample-SVD controls:

| Configuration | Relative error vs full factor anchor | Exact flopscope 0.9.1 fp32 |
|---|---:|---:|
| Dense adjoint | 0% | 33.997440062B |
| q64, dense through layer 27 | 1.584% | 10.909452350B |
| q32, dense through layer 24 | 2.310% | 10.338420542B |
| q32, dense through layer 27 | 7.742% | 7.314835262B |

Thus contracted evaluation is no longer the limiting cost.

## Source closures tested

### Gaussian/Born sources

Generate each local K3 source from the K1/K2/K4 state and discard K3 before
the next layer. This removes quadratic-in-depth carrier transport.

- Vendor named-counter source rollout: 17.003294283B.
- q32/h24 contraction: 10.338420542B under flopscope.
- Pooled raw error versus the full factor contraction: 18.41%.
- Pooled raw error versus the 100M-sample oracle: 38.24%.
- A fixed scale of 1.5498 reduces oracle contraction error to 15.28%.

The missing terms are not optional implementation details. Appended nonlinear
factor columns depend on incoming repeated K3 slices; after a dense weight
map, a C21-only forward recurrence is not closed.

### Goal-oriented repeated-slice closure

After every layer:

1. retain a fixed number of CP columns;
2. repair the `(3,)` and `(2,1)` slices exactly with a rank-width tensor;
3. rank retained all-distinct columns by their predicted contribution to the
   final two controls, not by tensor norm.

This is a dual-weighted-residual model reduction. The selection adjoint uses
the q32/h24 handoff above. At cap 768 it reduces the raw oracle contraction
error to 12.4%, versus 29.6% for the unscaled full factor artifact.

Its actual cost is approximately 81B beyond Kerdock:

- 79.456651083B vendor-counted closure and low-rank dual scoring;
- approximately 1.55B for the K2 response-map prepass.

Dense dual scoring is not deployable: it raises the closure to 112.5B.

Empirical Kerdock ReLU gates and analytic K2 gates gave nearly identical
selection quality.

## Necessary arbitrary-center correction

Connected C21 alone is an exact anchor only if the pointwise center \(m\)
equals the true mean \(\mu\). For arbitrary \(m\), the numerator is

\[
\begin{aligned}
A_m={}&C21
+2(\mu_i-m_i)M11_{ij}
+M2_i(\mu_j-m_j)\\
&+2(m_i^2-\mu_i^2)\mu_j.
\end{aligned}
\]

The corrected evaluator uses each closure's own mean and K2 state for this
lower-order term. The exact-oracle arbitrary-center ceiling is 0.56536 times
baseline MSE (8/8 wins) for these two controls.

The compact closures do not realize that ceiling:

- cap768 direct or output-shrunk controls are unstable and worse than baseline;
- shrinking its predicted quadrature-error delta toward the same-cloud anchor
  is neutral at best (about 0.998x);
- a cap512 + Born ensemble gives 8.2% leave-one-network-out anchor error but
  remains neutral as a control.

The full factor small-delta control reaches 0.864x on this selection block,
but it is the already-known non-robust factorized-delta family and has failed
on holdout.

## Decision

Keep the adjoint recursion as infrastructure. It is an algebraic and runtime
success. Do not integrate any current source closure:

- the exact source is not closed in mean, covariance, and C21;
- 8--15% contracted-anchor error is still far outside the control's tolerance;
- improving global contraction cosine does not reliably improve the
  output-weighted Kerdock residual.

The remaining technically coherent extension is a richer reverse closure that
includes the adjoints of repeated K3, radial K4, and lower-moment feedback, or
a surrogate trained directly against final output-weighted quadrature error.
