# Representation geometry: conic sums, exact lines, and boundary Laplacians

Date: 2026-07-28

## Executive conclusion

The strongest new identity in this track is exact:

\[
X\sim N(0,I_d),\quad f(\lambda x)=\lambda f(x)
\quad\Longrightarrow\quad
\mathbb E f(X)=\mathbb E\Delta f(X).
\]

For a ReLU network, the distributional Laplacian lives only on the bent
hyperplane arrangement formed by all ReLU kinks. This gives exact
boundary-surface and conditional-line estimators. Unfortunately, moving the
integral from volume to boundary makes the finite-budget estimator much
noisier:

- exact propagation along a Gaussian affine line is feasible in these actual
  networks (about 7,661 final pieces on average), but costs about 31.9B dense
  FLOPs per line;
- eight exact lines consume 90--96% of the 272B budget and have mean MSE
  `4.214e-3` on official selection MLPs 0--2;
- the existing 32,768-row sphere Sobol estimator has mean MSE `6.931e-7` on
  the same three MLPs, so exact one-coordinate conditioning is **6,081x
  worse** despite using roughly twice the compute;
- the exact slope-jump/Laplacian form is worse still (`3.727` mean MSE);
- a cheaper smoothed-delta forward-Laplacian proxy is **360,482x worse** than
  its equal-matmul RQMC baseline on disjoint selection IDs 10--19.

Global cone enumeration, tropical max-affine integration, and explicit
spherical-harmonic truncation all have harder representation barriers. This
closes the straightforward versions of the geometry path. It does not rule
out a new importance sampler that can draw kink surfaces in proportion to
their signed downstream contribution, but constructing that proposal appears
to require solving essentially the original arrangement problem.

## Primary-source adjacency

- [Zhang, Naitzat, and Lim (2018)](https://proceedings.mlr.press/v80/zhang18i.html)
  prove the equivalence between ReLU networks and tropical rational maps and
  relate linear regions to polytope vertices.
- [Serra, Tjandraatmadja, and Ramalingam (2018)](https://proceedings.mlr.press/v80/serra18b.html)
  derive region-count bounds that are exact in one input dimension and give
  an exact mixed-integer enumeration method.
- [Goujon, Etemadi, and Unser (2022/2024)](https://arxiv.org/abs/2206.08615)
  distinguish exponential worst-case region counts from much milder expected
  knot density along one-dimensional paths. The measured line counts below
  follow their average-case picture.
- [Botev (2017)](https://arxiv.org/abs/1603.04166) treats high-dimensional
  Gaussian integrals under linear restrictions using minimax tilting. It is a
  useful state-of-the-art primitive for one polyhedron, but does not remove
  the number of network regions.
- [Li et al. (2023)](https://arxiv.org/abs/2307.08214) develop forward
  Laplacian propagation for neural networks.
- [Hutchinson (1990)](https://doi.org/10.1080/03610919008812866) gives the
  Rademacher stochastic trace estimator used in the cheap Laplacian proxy.
- [Vempala and Wilmes (2019)](https://proceedings.mlr.press/v99/vempala19a.html)
  use the Funk--Hecke/spherical-harmonic view of ridge activations, which
  supplies the spectral framing used below.

## Formula 1: exact conic / tropical Gaussian integration

Let \(s\) be a joint activation pattern and \(C_s\) its central polyhedral
cone. Bias-free ReLU homogeneity means that, on this cone,

\[
f(x)=xM_s.
\]

Therefore

\[
\mathbb E f(X)
  =\sum_s m_s M_s,\qquad
m_s=\int_{C_s}x\,\phi_d(x)\,dx.
\]

Each \(m_s\) is a first moment of a linearly truncated multivariate normal.
Equivalently, a scalar CPWL output can be written as a difference of
max-affine (tropical polynomial) functions. For one max-affine term,

\[
\mathbb E\max_p a_p^\top X
=\sum_p
\mathbb E\!\left[
 a_p^\top X\,
 \mathbf 1\{(a_p-a_q)^\top X\geq0\ \forall q\}
\right],
\]

which is the same conic Gaussian-moment problem.

### Width-256/depth-32 cost

The first weight matrix is square and nonsingular almost surely. Its 256
central hyperplanes therefore already realize every orthant: exactly
\(2^{256}=1.158\times10^{77}\) nonempty first-layer cones. Later layers only
refine this fan. Merely storing one float32 \(256\times256\) slope map per
first-layer cone would require about \(3.04\times10^{82}\) bytes. Even a
perfect constant-time truncated-Gaussian oracle would not make the sum
tractable.

**Decision:** exact conic or tropical expansion is representationally
impossible here.

## Formula 2: exact conditional Gaussian line integration

Choose \(v\) uniformly on \(S^{d-1}\), draw
\(Z\sim N(0,I-vv^\top)\), and write \(X=Z+Tv\), with \(T\sim N(0,1)\)
independent. Along one affine line, every output is an exactly univariate
CPWL function. If

\[
f_j(Z+tv)=p_{rj}t+q_{rj},\qquad t\in(a_r,b_r),
\]

then the conditional mean is

\[
Q_j(Z,v)=\sum_r\left[
p_{rj}\{\phi(a_r)-\phi(b_r)\}
+q_{rj}\{\Phi(b_r)-\Phi(a_r)\}
\right],
\]

and \(\mathbb E_{Z,v}Q_j=\mathbb E f_j(X)\).

There is also an exact boundary form. Since
\(\Delta f=d\,\mathbb E_v D_v^2f\) and the second derivative of a univariate
CPWL function is its slope-jump measure,

\[
B_j(Z,v)
=d\sum_r \phi(t_r)
  \left(p_{r+1,j}-p_{rj}\right),
\qquad
\mathbb E_{Z,v}B_j=\mathbb E f_j(X).
\]

The second equality uses the Stein--Euler identity proved in Formula 3.

### Exact propagation cost

For \(R_{\ell-1}\) intervals entering layer \(\ell\), propagating both slope
and intercept costs two vector--matrix products:

\[
F_{\rm line}
=4w^2\sum_{\ell=1}^{L}R_{\ell-1}
\quad\text{dense FLOPs under FMA=2}.
\]

Across 24 exact lines from official selection MLPs 0--2:

- mean final pieces: 7,660.7 (range 5,789--9,054);
- mean \(\sum_\ell R_{\ell-1}\): 121,642.5;
- mean dense cost per line: 31.888B FLOPs
  (range 25.567--35.674B).

Eight-line results:

| MLP | exact conditional MSE | slope-jump MSE | dense FLOPs |
|---:|---:|---:|---:|
| 0 | 6.092e-3 | 9.616e-1 | 258.583B |
| 1 | 4.862e-3 | 3.525 | 261.835B |
| 2 | 1.689e-3 | 6.694 | 244.890B |
| mean | **4.214e-3** | **3.727** | 255.103B |

The exact conditional estimator integrates only one of 256 Gaussian
coordinates and leaves almost all outer variance. The boundary estimator also
multiplies a signed slope-jump sum by \(d=256\), causing severe cancellation
variance.

**Decision:** exact line propagation is computationally possible but
statistically noncompetitive.

## Formula 3: Stein--Euler boundary surfaces and a forward-Laplacian proxy

For a degree-one homogeneous \(f\), Euler's identity gives
\(x^\top\nabla f(x)=f(x)\) almost everywhere. Gaussian Stein integration by
parts gives

\[
\mathbb E[\partial_{ii}f(X)]
=\mathbb E[X_i\partial_i f(X)].
\]

Summing over \(i\) proves

\[
\boxed{\mathbb E\Delta f(X)=\mathbb E f(X)}.
\]

The result extends to continuous piecewise-linear \(f\) in the
distributional sense by mollification. Unrolling the distributional ReLU
chain rule gives, away from codimension-two intersections,

\[
\Delta f_j(x)
=\sum_{\ell,k}
g_{\ell k j}(x)\,
\delta(h_{\ell k}(x))\,
\|\nabla_x h_{\ell k}(x)\|_2^2,
\]

where \(g_{\ell k j}=\partial f_j/\partial a_{\ell k}\) is downstream
sensitivity. By coarea,

\[
\mathbb E[\delta(h(X))q(X)]
=\int_{h=0}
\frac{q(x)\phi_d(x)}{\|\nabla h(x)\|}\,
d\mathcal H^{d-1}(x),
\]

so the expectation is a signed, downstream-weighted sum of Gaussian surface
areas of all bent ReLU boundaries.

For a cheap falsifier, replace \(\delta\) by
\(\delta_\epsilon(h)=\phi(h/\epsilon)/\epsilon\). With one Rademacher input
direction, propagate

\[
\begin{aligned}
h&=aW,& \dot h&=\dot aW,& L_h&=L_aW,\\
a&=\operatorname{ReLU}(h),&
\dot a&=\mathbf1_{h>0}\dot h,&
L_a&=\mathbf1_{h>0}L_h+\delta_\epsilon(h)\dot h^2.
\end{aligned}
\]

Hutchinson's identity makes
\(\mathbb E_{\dot x}\dot h_i^2=\|\nabla_x h_i\|^2\). This costs three dense
matmul chains per row:

\[
F_{\rm smooth}=6NLw^2.
\]

At \(N=8192,w=256,L=32\), this is 103.079B dense FLOPs, exactly the
matmul cost of a 24,576-row direct baseline.

Bandwidths \(\{0.4,0.2,0.1,0.05,0.025\}\) were selected only on IDs 0--9.
The winner, \(\epsilon=0.025\), was frozen before evaluating IDs 10--19:

| method | validation mean MSE |
|---|---:|
| equal-cost sphere Sobol, 24,576 rows | 6.916e-7 |
| smoothed boundary Laplacian, 8,192 rows | 2.493e-1 |
| ratio | **360,482x worse** |

The smallest tested bandwidth won, indicating remaining smoothing bias, while
the kernel variance necessarily rises as the bandwidth shrinks. Deep signed
transport of the rare, large delta events makes the trade-off catastrophic.

**Decision:** reject the direct smoothed-delta/Hutchinson estimator.

## Formula 4: spherical harmonics and exact low-degree modes

Write \(X=RU\), where \(U\) is uniform on \(S^{255}\). Homogeneity yields

\[
\mathbb E f(X)=\mathbb E[R]\int_{S^{255}}g(U)\,dU,
\qquad f(RU)=R\,g(U).
\]

In the spherical-harmonic expansion
\(g=\sum_{\ell,m}c_{\ell m}Y_{\ell m}\), the desired answer is only the
degree-zero coefficient. All nonconstant harmonics have known zero mean and
can in principle be used as control variates.

The dimension of degree-\(\ell\) harmonics on \(S^{255}\) is

\[
H_{256,\ell}
=\binom{255+\ell}{\ell}
 -\binom{253+\ell}{\ell-2}.
\]

Thus:

| degree | number of modes |
|---:|---:|
| 0 | 1 |
| 1 | 256 |
| 2 | 32,895 |
| 3 | 2,828,800 |

Antithetic sampling already eliminates every odd harmonic exactly, so the
first unused block is degree 2. Fitting all degree-2/output cross-moments at
\(N=32768\) costs at least

\[
2N H_{256,2}\times256
=551.887\text{B FLOPs},
\]

before the 137.439B network forward cost. Degree 3 would cost 47.459T FLOPs
even though antithetic pairing makes it unnecessary. Structured low-rank
degree-2 sketches collapse back to frame/moment matching; the existing
tight-frame experiments were already negative.

**Decision:** explicit harmonic truncation starts above budget at the first
new even degree; low-rank versions have no new exactness guarantee.

## Reproducible artifacts

- `scripts/diagnose_exact_line_regions.py`
- `results/exact_line_region_counts.json`
- `scripts/eval_laplacian_geometry.py`
- `results/laplacian_geometry_proxy.json`

Both studies use only official Mini IDs 0--19 (or 0--2 for exact lines).
Challenge holdout IDs 50--99 were never loaded.

## Narrow remaining opening

The exact identity suggests one untested algorithmic class:

1. choose a ReLU boundary \((\ell,k)\) with probability proportional to an
   inexpensive approximation of its absolute downstream-weighted Gaussian
   surface mass;
2. sample \(X\) directly from \(h_{\ell k}(X)=0\) under the correct conic
   restrictions;
3. importance-weight the signed contribution
   \(g_{\ell k j}\|\nabla h_{\ell k}\|\).

This would avoid delta kernels and the random-direction factor of 256.
However, the boundary is itself bent across all preceding activation regions,
and sampling the correct restriction requires the conic/tropical machinery
that is already exponential. A useful future result would need a genuinely
new proposal distribution or shared-boundary factorization, not another
kernel-bandwidth or random-line variant.
