# Prompt 7 — M190 as an Internal Compositional-Chaos Engine

## Executive verdict

# **FAIL as an internal low-rank engine**

There is a useful exact identity:

> For tied-covariance Gaussian mixtures, the pooled post-ReLU covariance can be written as a sum of component-Gram matrices Hadamard-multiplied by powers of one shared correlation matrix.

This replaces \(K\) sets of bivariate Gaussian-CDF evaluations with matrix multiplications and Hadamard powers. It is legal, PSD-preserving, phase-preserving, and does not construct a global polynomial-chaos expansion.

However, it does **not** produce the low environment-weighted rank required to promote M190:

1. The actual matrix-valued contraction is generically full rank at every active Hermite degree.
2. The existing closest empirical rank sweep required rank \(128\), where low-rank evaluation costs approximately the same as a dense transform.
3. Low polynomial degrees have no uniform error certificate for negative component thresholds or highly correlated deep-layer states.
4. Degrees high enough to obtain a conservative certificate consume most or all of the parent path’s compute advantage.
5. M190 changes only the evaluator. It cannot repair a tied-covariance representation that loses essential covariance-modulation information.

The correct disposition is therefore:

> **Close M190 as a separate operational branch. Retain the shared-Hermite identity solely as an implementation option inside the final M192 tied-covariance experiment.**

Prompt 7 specifically requires one parent contraction, an exact recurrence, rank and error accounting, legal basis construction, parent cost before and after, and a binary verdict. It also says to close the branch when it does not materially improve a surviving parent path.

---

## 1. Selected parent contraction

The parent is the final tied-covariance exception inside M192. This is the only remaining mixture structure for which correlations might be reused across components; the broader heteroscedastic-mixture evaluator has already encountered an accuracy–rank–cost squeeze.

At layer \(\ell\), suppose

\[
Z\mid Q=q\sim\mathcal N(\mu_q,R),
\qquad
\Pr(Q=q)=\pi_q,
\]

where all components share the same covariance \(R\), but have different means \(\mu_q\).

Let

\[
H=\operatorname{ReLU}(Z).
\]

The minimum covariance object needed by the tied recurrence is

\[
\bar C
=
\sum_{q=1}^K\pi_q
\operatorname{Cov}(H\mid Q=q).
\]

For the next weight matrix \(W\), the required shared covariance is

\[
R^+
=
W^\top \bar C W.
\]

For one next-layer output direction \(w_r\), this contains the requested variance contraction

\[
v_r=w_r^\top\bar Cw_r,
\]

but a diagonal-only calculation is insufficient: the next ReLU layer requires the entire correlation matrix, so the parent path needs all cross-contractions

\[
(R^+)_{rs}=w_r^\top\bar Cw_s.
\]

The component means are propagated separately:

\[
\alpha_q
=
\mathbb E[H\mid q],
\qquad
\mu_q^+
=
W^\top\alpha_q.
\]

The next approximate state is

\[
Z^+\mid q
\sim
\mathcal N(\mu_q^+,R^+).
\]

This is the shared-covariance projection already identified as exactly preserving the approximate law’s global mean and covariance, although it deliberately removes component covariance variation.

---

## 2. Exact local chaos expansion

Write

\[
D=\operatorname{diag}(\sigma_1,\ldots,\sigma_n),
\qquad
\sigma_i=\sqrt{R_{ii}},
\]

and

\[
\rho=D^{-1}RD^{-1}.
\]

Then, for component \(q\),

\[
Z_{q,i}=\sigma_i(X_i+t_{q,i}),
\qquad
t_{q,i}=\frac{\mu_{q,i}}{\sigma_i},
\]

where

\[
X\sim\mathcal N(0,\rho).
\]

Define

\[
g_t(x)=\sigma(x+t)_+.
\]

Expand it in probabilists’ Hermite polynomials:

\[
g_t(x)
=
\sum_{d=0}^{\infty}
\frac{a_d(t,\sigma)}{d!}\operatorname{He}_d(x).
\]

The coefficients are explicit:

\[
a_0(t,\sigma)
=
\sigma\left[\phi(t)+t\Phi(t)\right],
\]

\[
a_1(t,\sigma)
=
\sigma\Phi(t),
\]

and, for \(d\ge 2\),

\[
\boxed{
a_d(t,\sigma)
=
\sigma\phi(t)\operatorname{He}_{d-2}(-t).
}
\]

Hence

\[
\alpha_{q,i}=a_0(t_{q,i},\sigma_i).
\]

For correlated standard Gaussian coordinates,

\[
\mathbb E[
\operatorname{He}_d(X_i)
\operatorname{He}_e(X_j)
]
=
\mathbf 1_{d=e}\,d!\,\rho_{ij}^{d}.
\]

Therefore the exact component covariance is

\[
\boxed{
(C_q)_{ij}
=
\sum_{d=1}^{\infty}
\frac{
a_d(t_{q,i},\sigma_i)
a_d(t_{q,j},\sigma_j)
}{d!}
\rho_{ij}^{d}.
}
\]

This is an exact Mehler/Hermite representation of the same component-specific ReLU covariance that would otherwise be evaluated using bivariate Gaussian moments.

### Pooled component contraction

For every degree \(d\), form the coefficient matrix

\[
A_d\in\mathbb R^{K\times n},
\qquad
(A_d)_{q,i}
=
a_d(t_{q,i},\sigma_i),
\]

and let

\[
\Pi=\operatorname{diag}(\pi_1,\ldots,\pi_K).
\]

Define

\[
G_d
=
A_d^\top\Pi A_d.
\]

Then

\[
(G_d)_{ij}
=
\sum_q\pi_q
a_d(t_{q,i},\sigma_i)
a_d(t_{q,j},\sigma_j).
\]

Consequently,

\[
\boxed{
\bar C
=
\sum_{d=1}^{\infty}
\frac{
\rho^{\circ d}\circ G_d
}{d!},
}
\]

where \(\circ\) denotes the Hadamard product.

The parent contraction becomes

\[
\boxed{
R^+
=
\sum_{d=1}^{\infty}
\frac{
W^\top
\left(
\rho^{\circ d}\circ G_d
\right)
W
}{d!}.
}
\]

This contracts immediately against the downstream matrix and never constructs a global input-coordinate PCE.

---

## 3. Exact layerwise transfer

For truncation degree \(D\), one layer is:

1. Compute the legal normalized thresholds

   \[
   t_{q,i}=\mu_{q,i}/\sigma_i.
   \]

2. Compute component ReLU means

   \[
   \alpha_{q,i}
   =
   \sigma_i[\phi(t_{q,i})+t_{q,i}\Phi(t_{q,i})].
   \]

3. Generate \(A_d\), \(d=1,\ldots,D\), using the Hermite recurrence

   \[
   \operatorname{He}_{d+1}(x)
   =
   x\operatorname{He}_d(x)
   -
   d\operatorname{He}_{d-1}(x).
   \]

4. Form

   \[
   G_d=A_d^\top\Pi A_d.
   \]

5. Recursively form the shared powers

   \[
   P_1=\rho,
   \qquad
   P_{d+1}=P_d\circ\rho.
   \]

6. Accumulate

   \[
   \bar C_D
   =
   \sum_{d=1}^{D}
   \frac{P_d\circ G_d}{d!}.
   \]

7. Replace the diagonal by its exact analytic value:

   \[
   (\bar C)_{ii}
   =
   \sum_q\pi_q
   \operatorname{Var}\!\left[
   (Z_{q,i})_+
   \right].
   \]

8. Propagate

   \[
   \mu_q^+=W^\top\alpha_q,
   \qquad
   R_D^+=W^\top\bar C_DW.
   \]

9. Continue with

   \[
   Z^+\mid q
   \sim\mathcal N(\mu_q^+,R_D^+).
   \]

This is a compositional transfer: local Hermite degrees are generated and contracted at each layer, but no global coefficient tensor is reconstructed.

### Degree interactions and zeros

Only equal Hermite degrees interact:

\[
d=e.
\]

The covariance removes \(d=0\) exactly.

If \(t=0\), then:

- \(d=1\) survives;
- even \(d\ge2\) survive;
- odd \(d\ge3\) vanish.

For nonzero component means, both odd and even degrees generally survive. Thus a tied-covariance mixture with separated component means does not retain the large symmetry simplification available to a centered Gaussian.

### Signed phase

This is not an energy-only statistic. The output contraction contains

\[
w_{ir}w_{js}
\]

with its sign, while odd powers of negative correlations also retain their sign. The resulting \(R^+_{rs}\) therefore preserves signed downstream orientation.

---

## 4. PSD and realizability

Every truncated degree term is PSD.

First,

\[
\rho^{\circ d}\succeq0
\]

by the Schur product theorem.

Second,

\[
G_d=A_d^\top\Pi A_d\succeq0.
\]

Therefore,

\[
\rho^{\circ d}\circ G_d\succeq0.
\]

Hence

\[
\bar C_D\succeq0.
\]

Replacing the approximate diagonal by the exact diagonal adds a nonnegative diagonal residual, so it also preserves PSD. Finally,

\[
R_D^+=W^\top\bar C_DW\succeq0.
\]

Thus the engine needs no numerical PSD projection and always yields a realizable tied Gaussian-mixture state.

---

## 5. Environment-weighted rank result

The low-rank hypothesis fails at the natural matrix cut.

| Object | Rank bound | Generic exact rank |
|---|---:|---:|
| Component coefficient matrix \(A_d\) | \(\le \min(K,n)\) | \(\min(K,n)\) |
| Component Gram \(G_d=A_d^\top\Pi A_d\) | \(\le K\) | \(\min(K,n)\) |
| Shared correlation power \(\rho^{\circ d}\) | \(\le n\) | \(n\) |
| Actual degree contribution \(M_d=\rho^{\circ d}\circ G_d\) | \(\le n\) | **\(n\)** |
| Output contraction \(W^\top M_dW\) | \(\le n\) | **\(n\)** |

### Proof of generic full rank

Write

\[
M_d
=
\sum_q
\pi_q
D_{q,d}\rho^{\circ d}D_{q,d},
\]

where

\[
D_{q,d}
=
\operatorname{diag}
\left(
a_d(t_{q,1},\sigma_1),
\ldots,
a_d(t_{q,n},\sigma_n)
\right).
\]

If \(\rho\succ0\), then \(\rho^{\circ d}\succ0\). If one component has nonzero degree-\(d\) coefficients in every coordinate, then

\[
D_{q,d}\rho^{\circ d}D_{q,d}\succ0.
\]

Therefore

\[
M_d\succ0
\]

and has rank \(n\).

The exceptional coefficient zeros occur on algebraic threshold sets and do not generically produce a stable cross-network rank reduction. Multiplication by a generic full-rank \(W\) preserves full rank.

So the low rank of \(G_d\) is destroyed by its Hadamard interaction with the shared correlation matrix. The component dimension \(K\) is not the relevant rank of the object required by the parent recurrence.

### Nearest existing empirical proxy

No M190 oracle-rank artifacts were found. The canonical ledger explicitly recorded the environment-weighted rank audit as not run, rather than as an empirical candidate.

The closest existing experiment is the direct/Hermite covariance-rank sweep:

| Layer | \(r=4\) | \(r=16\) | \(r=64\) | \(r=128\) |
|---:|---:|---:|---:|---:|
| 16 | \(2.16\times10^{-1}\) | \(5.44\times10^{-2}\) | \(6.73\times10^{-3}\) | \(7.86\times10^{-4}\) |
| 29 | \(1.74\times10^{-1}\) | \(4.00\times10^{-2}\) | \(3.76\times10^{-3}\) | \(2.23\times10^{-4}\) |

Only rank \(128\) cleared the stated local error gate. At that rank,

\[
2n^2r\approx n^3,
\]

so the low-rank evaluator had become a dense evaluator in cost.

This table is not an empirical measurement of the newly derived pooled tied-covariance object. It is only the nearest available downstream-weighted warning. Claiming held-out M190 rank stability from it would be invalid.

---

## 6. Truncation error certificate

For coordinate \(i\) in component \(q\), define its residual Hermite variance

\[
T_{q,i,D}
=
\sum_{d>D}
\frac{a_d(t_{q,i},\sigma_i)^2}{d!}.
\]

This is available analytically:

\[
T_{q,i,D}
=
\operatorname{Var}[(Z_{q,i})_+]
-
\sum_{d=1}^{D}
\frac{a_d(t_{q,i},\sigma_i)^2}{d!}.
\]

For an off-diagonal pair,

\[
E_{q,ij}^{(D)}
=
\sum_{d>D}
\frac{
a_{q,i,d}a_{q,j,d}
}{d!}
\rho_{ij}^{d}.
\]

Cauchy–Schwarz gives

\[
\boxed{
|E_{q,ij}^{(D)}|
\le
|\rho_{ij}|^{D+1}
\sqrt{
T_{q,i,D}T_{q,j,D}
}.
}
\]

After pooling,

\[
|\bar E_{ij}^{(D)}|
\le
|\rho_{ij}|^{D+1}
\sum_q\pi_q
\sqrt{
T_{q,i,D}T_{q,j,D}
}.
\]

Thus a fully legal, target-free residual bound can be computed from the propagated means and covariance alone.

For the output covariance, an entrywise certificate is

\[
\left|
(W^\top\bar EW)_{rs}
\right|
\le
\left(
|W|^\top B_D|W|
\right)_{rs},
\]

where \(B_D\) is the pairwise bound matrix above.

### Analytic univariate tail diagnostic

The following table gives the exact fraction of one-coordinate ReLU variance above degree \(D\), for

\[
(X+t)_+,
\qquad
X\sim\mathcal N(0,1).
\]

It is not the final covariance error, because off-diagonal terms gain the factor \(|\rho|^d\), and almost-dead neurons may have small absolute variance.

| Threshold \(t\) | \(D=10\) | \(D=32\) | \(D=64\) | \(D=100\) |
|---:|---:|---:|---:|---:|
| \(-3\) | \(8.14\times10^{-2}\) | \(1.32\times10^{-2}\) | \(4.64\times10^{-3}\) | \(2.35\times10^{-3}\) |
| \(-2\) | \(3.18\times10^{-2}\) | \(5.80\times10^{-3}\) | \(1.96\times10^{-3}\) | \(1.02\times10^{-3}\) |
| \(-1\) | \(1.14\times10^{-2}\) | \(2.09\times10^{-3}\) | \(7.46\times10^{-4}\) | \(3.75\times10^{-4}\) |
| \(0\) | \(3.74\times10^{-3}\) | \(6.76\times10^{-4}\) | \(2.41\times10^{-4}\) | \(1.24\times10^{-4}\) |
| \(1\) | \(1.04\times10^{-3}\) | \(1.90\times10^{-4}\) | \(6.79\times10^{-5}\) | \(3.42\times10^{-5}\) |
| \(2\) | \(1.89\times10^{-4}\) | \(3.44\times10^{-5}\) | \(1.16\times10^{-5}\) | \(6.02\times10^{-6}\) |

This establishes:

- degree 10 is not uniformly sufficient;
- degree 32 is still weak for moderately negative thresholds;
- degree 64 may be adequate for many active coordinates but is not a universal certificate;
- degree 100 may still struggle in relative terms for nearly dead coordinates;
- highly correlated deep-layer pairs weaken the \(|\rho|^{D+1}\) suppression.

---

## 7. Legal basis construction

No oracle orientation is required.

The complete legal construction is:

- shared covariance \(R\) from the propagated tied state;
- marginal scales \(\sigma_i\);
- shared correlation matrix \(\rho\);
- component thresholds \(t_{q,i}\);
- fixed probabilists’ Hermite basis;
- analytic coefficient recurrence;
- component weights \(\pi_q\);
- realized downstream matrix \(W\).

No activation cloud, target residual, true covariance defect, target-selected eigenspace, or per-network rank choice is used.

A degree schedule could legally be:

1. fixed globally before validation; or
2. selected from the analytic residual bound using a frozen tolerance.

A low-rank eigenspace chosen after inspecting reference activations would be oracle-only and invalid.

---

## 8. Compute accounting

Let

\[
n=256,
\qquad
K=64,
\qquad
L=31.
\]

### Shared-Hermite evaluator

For degree \(D\):

- component means:

  \[
  2KLn^2;
  \]

- \(D\) component Gram products:

  \[
  2DKLn^2;
  \]

- Hadamard powers and accumulation:

  \[
  O(DLn^2);
  \]

- one shared dense covariance transform per layer:

  \[
  4Ln^3.
  \]

Ignoring lower-order coefficient generation, special functions, certification and memory movement,

\[
\boxed{
F_{\mathrm{EWCC}}(D)
\approx
2(D+1)KLn^2+4Ln^3.
}
\]

| Degree | Projected FLOPs |
|---:|---:|
| \(D=10\) | \(4.94\)B |
| \(D=16\) | \(6.50\)B |
| \(D=32\) | \(10.66\)B |
| \(D=64\) | \(18.98\)B |
| \(D=100\) | \(28.35\)B |

These numbers omit:

- \(\Phi\) and \(\phi\) evaluations;
- coefficient recurrence;
- exact-diagonal replacement;
- residual certification;
- matrix materialization and copies;
- fallbacks;
- official-profiler wall-time residual.

### Exact tied-covariance evaluator

For \(K=64\), the existing cost model gives

\[
65{,}265{,}664
\]

upper-triangular component-pair evaluations over all layers.

At approximately \(200\)–\(400\) effective FLOPs per pair, this is

\[
13.05\text{B} \text{ to } 26.11\text{B}.
\]

Adding component means and one shared dense transform gives approximately

\[
15.4\text{B} \text{ to } 28.5\text{B}
\]

before other overhead.

### Parent-path comparison

| Evaluator | Cost projection | Accuracy status |
|---|---:|---|
| Exact bivariate tied moments | \(15.4\)–\(28.5\)B | Exact under tied law |
| Shared Hermite, \(D=10\) | \(4.94\)B | No adequate uniform certificate |
| Shared Hermite, \(D=32\) | \(10.66\)B | Potentially useful, but unverified |
| Shared Hermite, \(D=64\) | \(18.98\)B | Modest savings at best after overhead |
| Shared Hermite, \(D=100\) | \(28.35\)B | Over broad budget before overhead |

The engine materially changes the parent cost only if a frozen degree near \(16\)–\(32\) passes the complete downstream certificate. Nothing in the available artifacts establishes that.

At \(D\approx64\), the evaluator has already entered the exact evaluator’s cost range. At \(D\approx100\), its arithmetic alone consumes the approximate analytic budget.

---

## 9. Why this cannot rescue M192’s representation

A shared covariance projection preserves the approximate law’s mean and covariance, but it removes:

- component covariance variation;
- covariance–mean coupling;
- covariance-mixture fourth cumulants.

For a zero-mean conditional covariance mixture, the fourth cumulant is directly determined by covariance variation:

\[
\kappa_{ijkl}
=
\operatorname{Cov}(V_{ij},V_{kl})
+
\operatorname{Cov}(V_{ik},V_{jl})
+
\operatorname{Cov}(V_{il},V_{jk}).
\]

That is precisely the type of mixed fourth-order dependence that motivated the heteroscedastic mixture in the first place.

M190 therefore cannot improve the tied model’s representational closure. It can only evaluate the tied model more cheaply.

The logical ordering remains:

1. M192 must first show that tied or shared-low-rank covariance reaches the oracle closure gate.
2. Only then is evaluator cost relevant.
3. The Hermite pooling identity can be benchmarked against exact bivariate moments.
4. Failure of tied representation closes M192 regardless of M190.
5. Failure of a low-degree certificate means exact tied evaluation should be used rather than maintaining M190 as a separate branch.

---

## 10. Required PASS/FAIL decision

# **FAIL**

### Reason

The selected parent contraction has:

- an exact local compositional-chaos recurrence;
- a legal basis;
- PSD preservation;
- no global PCE;
- a potentially useful component-pooling implementation.

But it does **not** have:

- low exact environment-weighted rank;
- measured low approximate rank on the actual target;
- held-out rank stability;
- a demonstrated \(D\le32\) downstream error certificate;
- a complete score-positive implementation;
- evidence that its savings survive official accounting.

The closest rank experiment instead points toward rank \(128\), where the computational advantage disappears.

The standalone low-band interpretation is separately bounded to approximately

\[
1.58\times
\]

zero-cost gain and cannot win.

---

## 11. Ledger-ready patch

| Field | Proposed value |
|---|---|
| ID | `M190` |
| Family | Compositional chaos / internal contraction engine |
| Experiment | Shared-Hermite evaluation of the tied-covariance M192 recurrence |
| Evidence level | Exact identity, generic rank theorem and compute projection; no new empirical cohort |
| Canonical result | For a tied Gaussian mixture, the pooled post-ReLU covariance has the exact expansion \(\bar C=\sum_{d\ge1}(\rho^{\circ d}\circ A_d^\top\Pi A_d)/d!\). The recurrence is legal and PSD-preserving. Nevertheless, every active degree contribution is generically full matrix rank. Low-degree truncations lack a demonstrated complete downstream certificate; degrees near 64–100 consume most or all projected compute savings. |
| Verdict | **FAIL as a separate internal low-rank engine** |
| Status | Operationally closed; identity retained as an M192 evaluator implementation option |
| Protected data opened? | No |
| Candidate score | None |
| Compute status | \(4.94\)B at \(D=10\), \(10.66\)B at \(D=32\), \(18.98\)B at \(D=64\), \(28.35\)B at \(D=100\), before full overhead |
| Closure confidence | High for failure of exact low matrix rank; moderate for operational closure because actual tied-state truncation curves have not been run |
| Next action | Do not run an independent M190 workstream. During the final M192 tied-covariance test, benchmark this identity at frozen \(D=16,32,64\) against exact bivariate moments and evaluate the analytic downstream residual certificate. Promote neither M190 nor the approximation unless \(D\le32\) passes every network with score slack. |
| Caveat | This does not prove that every possible tensor-network cut has high approximate environment rank. It proves that the natural matrix object required by the tied recurrence is generically full rank and that no empirical alternative rank result currently exists. |
