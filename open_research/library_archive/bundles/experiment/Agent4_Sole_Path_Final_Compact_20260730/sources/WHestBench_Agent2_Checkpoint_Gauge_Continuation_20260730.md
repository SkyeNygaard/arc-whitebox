# WHestBench Agent 2 continuation — exact checkpoint gauges and the convex contraction frontier

**Date:** 2026-07-30  
**Role:** Agent 2 — exact scalar contraction identity hunter  
**Starting point:** Agent 2 exact first-band package; canonical v25; T70–T73  
**Disposition:** **NEW EXACT THEOREM FAMILY; GENERIC LINEAR CHECKPOINT TELESCOPE DEMOTED AS A WINNING BRANCH; RETAIN AS A SOURCE-SPECIFIC CONVEX AUDIT AND CERTIFICATE**

## Executive conclusion

The fixed adjoint potential of T70 is not the unique exact scalar contraction telescope. It is one point in a much larger exact family.

For arbitrary checkpoint-control matrices \(C_0,\ldots,C_L\), with terminal matrix \(C_L=U\),

\[
U^\top h_L(x)-C_0^\top x
=
\sum_{\ell=1}^L
\left[C_\ell^\top h_\ell(x)-C_{\ell-1}^\top h_{\ell-1}(x)\right].
\]

Whenever \((P-Q)x=0\), this gives an exact contraction identity

\[
\boxed{
U^\top(P-Q)h_L
=
\sum_{\ell=1}^L(P-Q)
\left[C_\ell^\top h_\ell-C_{\ell-1}^\top h_{\ell-1}\right].
}
\]

This freedom is a **checkpoint gauge**. T70 corresponds to the special recursion

\[
C_{\ell-1}=\frac12W_\ell^\top C_\ell,
\]

which converts each increment into an absolute-preactivation term. A Gaussian local-regression gauge instead uses expected gates,

\[
C_{\ell-1}=W_\ell^\top D_{p_\ell}C_\ell,
\qquad
(D_{p_\ell})_{jj}=\Pr(Z_{\ell j}>0),
\]

and is exactly optimal among linear preactivation controls when the current preactivation vector is jointly Gaussian.

The apparent opportunity is large: choose the controls and checkpoints to minimize the T72 variance-cost constant, rather than accepting the fixed \(1/2\)-adjoint.

The hostile result is equally important:

> For every fixed checkpoint partition, the globally optimal linear checkpoint-control telescope is a convex second-order-cone program. Direct estimation is always a feasible point. In architecture-matched width-256/depth-30 screens, the optimum usually collapsed to direct estimation or gave only unstable, small improvements.

Thus the new theorem does **not** establish a winning estimator. It establishes a finite, exhaustive audit of a broad exact family and supplies a defensible stop certificate when the optimized source-specific T72 margin fails.

### Final recommendation

1. **Adopt** the arbitrary checkpoint-gauge theorem and its convex optimization formulation.
2. **Adopt** the exact first-layer mean as the free base level.
3. **Adopt** the Gaussian soft-gate identity as a local optimality theorem, not as a fixed-network truth claim.
4. **Stop** generic all-layer linear checkpoint telescopes as a standalone winning branch.
5. **Continue only source-specifically:** after Agent 1 freezes a legal late source \(U\), solve the exact covariance/SOCP problem on exposed data and compare its certified optimum directly with the source-specific T72 allowance.
6. **Require a dual certificate and untouched covariance validation** before any protected evaluation.

No deployable winner is claimed.

---

## 1. Setup

Let

\[
h_0(x)=x,
\qquad
h_\ell(x)=\sigma(W_\ell h_{\ell-1}(x)),
\qquad \ell=1,\ldots,L,
\]

where the main application has \(\sigma(t)=\max(t,0)\) coordinatewise.

Let \(P\) be the target expectation and \(Q\) the production cubature rule. Assume

\[
(P-Q)x=0,
\]

as for centered input and an antipodal mass-one rule.

Let \(U\in\mathbb R^{d_L\times k}\) be a frozen physical terminal source. The desired signed contraction is

\[
b=U^\top(P-Q)h_L\in\mathbb R^k.
\]

All statements below are conditional on \(U\) being legal, target-free, and physically oriented. The theorem does not solve source selection.

---

## 2. Exact arbitrary checkpoint-gauge theorem

### Theorem A2-G1 — arbitrary checkpoint-control telescope

Choose arbitrary matrices

\[
C_0\in\mathbb R^{d_0\times k},\ldots,
C_L\in\mathbb R^{d_L\times k},
\qquad C_L=U.
\]

Define the layer increment

\[
R_\ell^{C}(x)
=
C_\ell^\top h_\ell(x)
-
C_{\ell-1}^\top h_{\ell-1}(x).
\]

Then pathwise,

\[
\boxed{
U^\top h_L(x)-C_0^\top x
=
\sum_{\ell=1}^L R_\ell^C(x).
}
\]

Therefore, if \((P-Q)x=0\),

\[
\boxed{
U^\top(P-Q)h_L
=
\sum_{\ell=1}^L(P-Q)R_\ell^C.
}
\]

#### Proof

The sum telescopes:

\[
\sum_{\ell=1}^L
\left(C_\ell^\top h_\ell-C_{\ell-1}^\top h_{\ell-1}\right)
=
C_L^\top h_L-C_0^\top h_0.
\]

Substitute \(C_L=U\), \(h_0=x\), and apply \(P-Q\). ∎

### Scope

The pathwise theorem does not require:

- ReLU;
- Gaussian weights;
- independent layers;
- equal widths;
- invertibility;
- smoothness;
- probabilistic approximation.

It is pure algebra.

### Interpretation

The intermediate controls \(C_\ell\) are a gauge: they alter the variance and cost of the individual exact increments while leaving their total signed contraction unchanged.

This separates two questions that previous work often conflated:

1. **Identity:** Does the decomposition sum to the correct contraction? Always, for every gauge.
2. **Economics:** Can the individual expectations be estimated more cheaply than the terminal contraction? Only for favorable gauges, partitions, covariances, and sample designs.

---

## 3. T70 as one exact gauge

For ReLU,

\[
h_\ell
=
\frac12W_\ell h_{\ell-1}
+
\frac12|z_\ell|,
\qquad z_\ell=W_\ell h_{\ell-1}.
\]

Choose

\[
C_{\ell-1}
=
\frac12W_\ell^\top C_\ell.
\]

Then

\[
\begin{aligned}
R_\ell^C
&=C_\ell^\top h_\ell
-C_{\ell-1}^\top h_{\ell-1}\\
&=\frac12C_\ell^\top|z_\ell|.
\end{aligned}
\]

Thus A2-G1 specializes exactly to T70:

\[
U^\top(P-Q)h_L
=
\frac12\sum_{\ell=1}^L
(P-Q)C_\ell^\top|z_\ell|.
\]

The old adjoint potential is therefore not wrong or arbitrary. It is the unique checkpoint gauge obtained by removing the explicit linear term of ReLU with the fixed centered slope \(1/2\) at every layer.

But \(1/2\) need not be variance-optimal for a noncentered realized layer distribution.

---

## 4. General slope gauges

For any diagonal matrix \(D_\ell\), write

\[
h_\ell
=D_\ell z_\ell+ho_{D_\ell}(z_\ell),
\]

where

\[
\rho_{D_\ell}(z)
=
\operatorname{ReLU}(z)-D_\ell z.
\]

Choose

\[
C_{\ell-1}=W_\ell^\top D_\ell C_\ell.
\]

Then

\[
\boxed{
R_\ell^C
=C_\ell^\top\rho_{D_\ell}(z_\ell).
}
\]

Special cases:

- \(D_\ell=\tfrac12I\): T70 absolute-value gauge;
- \(D_\ell=0\): no local control;
- \(D_\ell=I\): negative-part residual;
- \(D_\ell=\operatorname{diag}(p_\ell)\): soft-gate residual;
- data-derived full matrix bridges: general linear regression gauge.

The total identity remains exact for every frozen choice. Only the variance and legality differ.

---

## 5. Same-sample no-free-lunch theorem

### Theorem A2-G2 — pathwise collapse under common evaluation

For any gauge \(C\), if all increments are evaluated on the same point \(x\), then

\[
C_0^\top x+
\sum_{\ell=1}^LR_\ell^C(x)
=U^\top h_L(x).
\]

Consequently, evaluating all increments on the same cubature nodes and summing with their exact telescope coefficients gives exactly the original terminal cubature value. No gauge can create variance reduction merely by rewriting the same pathwise data.

### What can create value

At least one of the following is necessary:

1. a base expectation is known exactly, as at layer 1;
2. different increments use different independent sample counts;
3. shallower increments use cheaper off-support trajectories;
4. an increment has an analytic expectation;
5. a structured randomized design changes its variance-cost constant;
6. a control mean is estimated externally rather than from the same cloud;
7. a block is replaced by an exactly integrable surrogate with an explicit residual.

This is the checkpoint-gauge version of T71’s total-potential no-op.

---

## 6. Exact free base level at layer 1

For Gaussian input \(X\sim N(0,I)\), each first-layer preactivation is Gaussian:

\[
w_j^\top X\sim N(0,\|w_j\|^2).
\]

Therefore

\[
P h_{1,j}
=
\frac{\|w_j\|}{\sqrt{2\pi}}.
\]

For any control matrix \(C_1\),

\[
\boxed{
P(C_1^\top h_1)
=
\frac1{\sqrt{2\pi}}C_1^\top r_1,
\qquad (r_1)_j=\|w_j\|.
}
\]

This extends the prior first-band result: the exact layer-1 mean is not tied to the T70 adjoint. It is available for **every** checkpoint gauge.

Thus the useful partition begins at checkpoint 1, with \(C_1\) freely optimized and its expectation known at effectively zero sampling cost.

---

## 7. Fixed-partition multilevel estimator

Choose checkpoints

\[
1=t_0<t_1<\cdots<t_m=L.
\]

Let \(C_j\) denote the control matrix at checkpoint \(t_j\), with

\[
C_m=U.
\]

Define block increments

\[
G_j(X)
=
C_j^\top h_{t_j}(X)
-C_{j-1}^\top h_{t_{j-1}}(X),
\qquad j=1,\ldots,m.
\]

Then

\[
P(U^\top h_L)
=
P(C_0^\top h_1)
+
\sum_{j=1}^mP G_j.
\]

The first term is analytic. Estimate each \(P G_j\) using an independent sample/design of size \(n_j\).

Let:

- \(M\succeq0\) be the contraction score metric;
- \(v_j=\operatorname{tr}(M\operatorname{Cov}(G_j))\);
- \(\gamma_j\) be the cost of one block-\(j\) trajectory, normalized to baseline compute.

With independent unbiased blocks,

\[
V=\sum_{j=1}^m\frac{v_j}{n_j},
\qquad
x=\sum_{j=1}^m\gamma_jn_j.
\]

For fixed controls, the exact allocation constant is

\[
\boxed{
S(C)=\sum_{j=1}^m\sqrt{\gamma_jv_j(C)}.
}
\]

The minimum variance at added cost \(x\) is

\[
V_{\min}(x)=\frac{S(C)^2}{x}.
\]

This plugs directly into T72 or the later checkpoint-specific \(\kappa\) frontier.

---

## 8. Convex source–control optimization

### Theorem A2-G3 — fixed-partition SOCP

Let

\[
H_j=
\begin{bmatrix}
h_{t_j}\\h_{t_{j-1}}
\end{bmatrix},
\qquad
K_j=\operatorname{Cov}(H_j),
\qquad
B_j=
\begin{bmatrix}
C_j\\-C_{j-1}
\end{bmatrix}.
\]

Then

\[
G_j=B_j^\top H_j,
\]

and

\[
\begin{aligned}
v_j
&=\operatorname{tr}(M B_j^\top K_jB_j)\\
&=\left\|K_j^{1/2}B_jM^{1/2}\right\|_F^2.
\end{aligned}
\]

Therefore the globally optimal checkpoint controls solve

\[
\boxed{
\min_{C_0,\ldots,C_{m-1}}
\sum_{j=1}^m
\sqrt{\gamma_j}
\left\|K_j^{1/2}
\begin{bmatrix}C_j\\-C_{j-1}\end{bmatrix}
M^{1/2}\right\|_F,
\quad C_m=U.
}
\]

This is convex and has a standard second-order-cone representation.

#### Proof

Each block term is the norm of an affine map of the intermediate controls. A nonnegative weighted sum of norms of affine maps is convex. Introducing epigraph variables \(s_j\) and constraints

\[
\left\|K_j^{1/2}B_jM^{1/2}\right\|_F\le s_j
\]

produces an SOCP. ∎

### Direct estimation is always feasible

Set every intermediate control to zero:

\[
C_0=\cdots=C_{m-1}=0,
\qquad C_m=U.
\]

Only the terminal block remains, so the objective equals the direct estimator’s variance-cost constant.

Hence

\[
S_{\mathrm{SOCP}}\le S_{\mathrm{direct}}.
\]

A solver returning a value near equality is not a numerical failure; it is evidence that no checkpoint linear control has value under the declared covariance and cost model.

### Dual certificate

The SOCP has the usual norm-dual form. A dual-feasible collection of block matrices with Frobenius norm at most \(\sqrt{\gamma_j}\), satisfying the adjoint checkpoint-balance equations, gives a lower bound on the primal objective. Matching primal and dual values certify the exact optimum within the declared linear checkpoint-control class.

This is important operationally:

> The source-specific continuation can end with a machine-checkable impossibility certificate rather than another inconclusive regression sweep.

---

## 9. Local regression theorem

### Theorem A2-G4 — optimal linear checkpoint bridge

Let \(X=h_{t_{j-1}}-Ph_{t_{j-1}}\) and \(Y=h_{t_j}-Ph_{t_j}\). For a fixed terminal control \(C_j\), choose \(C_{j-1}\) to minimize

\[
\operatorname{tr}M\operatorname{Cov}
\left(C_j^\top Y-C_{j-1}^\top X\right).
\]

Let

\[
\Sigma_{XX}=E[XX^\top],
\qquad
\Sigma_{XY}=E[XY^\top].
\]

Then a minimum-norm optimum is

\[
\boxed{
C_{j-1}^*
=
\Sigma_{XX}^{\dagger}\Sigma_{XY}C_j.
}
\]

The irreducible residual covariance is

\[
\boxed{
C_j^\top
\left(
\Sigma_{YY}
-
\Sigma_{YX}\Sigma_{XX}^{\dagger}\Sigma_{XY}
\right)
C_j.
}
\]

This is the Schur complement or conditional linear-variance remainder.

### Important qualification

Greedily applying this optimum layer by layer does not minimize the global sum

\[
\sum_j\sqrt{\gamma_jv_j}.
\]

Each intermediate control enters two neighboring blocks. The globally optimal gauge must solve the joint convex problem. The architecture screen confirmed that greedy regression can be much worse than direct estimation.

---

## 10. Gaussian soft-gate theorem

Let a current preactivation vector be jointly Gaussian,

\[
Z\sim N(\mu,\Sigma),
\qquad H=\operatorname{ReLU}(Z).
\]

Let

\[
p_j=\Pr(Z_j>0)=\Phi(\mu_j/\sigma_j),
\qquad D_p=\operatorname{diag}(p_j).
\]

### Theorem A2-G5 — exact Gaussian local optimum

Gaussian Stein’s identity gives

\[
\boxed{
\operatorname{Cov}(Z,H)=\Sigma D_p.
}
\]

Therefore the optimal linear predictor of \(C^\top H\) from \(Z\) is

\[
\boxed{
(D_pC)^\top Z.
}
\]

If \(Z=W h_{\ell-1}\), the corresponding checkpoint gauge is

\[
\boxed{
C_{\ell-1}=W^\top D_pC_\ell.
}
\]

The exact Gaussian residual covariance is

\[
\boxed{
C_\ell^\top
\left(
\Sigma_{HH}-D_p\Sigma D_p
\right)
C_\ell.
}
\]

#### Proof

For a jointly Gaussian vector and an almost-everywhere differentiable scalar function,

\[
\operatorname{Cov}(Z_i,f(Z_j))
=
\Sigma_{ij}E[f'(Z_j)].
\]

For \(f(z)=\operatorname{ReLU}(z)\), \(f'(z)=1_{z>0}\) almost everywhere, so

\[
\operatorname{Cov}(Z_i,H_j)=\Sigma_{ij}p_j.
\]

Thus \(\Sigma_{ZH}=\Sigma D_p\). Apply the linear-regression theorem. ∎

### Relation to T70

For centered Gaussian preactivations, \(p_j=1/2\), so

\[
D_p=\frac12I.
\]

T70 is exactly the locally variance-optimal linear gauge under the centered Gaussian model.

For noncentered deep layers, expected-gate backprop is the correct Gaussian local bridge, explaining why historical soft-Jacobian controls outperformed hard gates and constant \(1/2\) gates in some experiments.

### Legality

This theorem is a distributional statement. It does not claim that deep finite-width preactivations are exactly Gaussian, nor that propagated gate probabilities are exact for a fixed realized network. It supplies:

- an analytic null gauge;
- a local lower benchmark;
- a covariance control;
- a testable approximation.

It is not an absolute fixed-network identity beyond the algebraic telescope itself.

---

## 11. Exact two-level correlation–cost frontier

The central economic question is not whether a shallow checkpoint predicts the terminal output well. It is whether it predicts well enough relative to the cost of learning its absolute mean.

Let:

- \(Y\) be a scalar target with variance \(\sigma_Y^2\) and per-sample cost \(C\);
- \(X\) be a cheap control with cost \(c\);
- \(R^2\) be the maximum linear explained fraction of \(Y\) from \(X\);
- \(r=c/C\).

Take \(n\) paired \((X,Y)\) samples and \(m\) additional \(X\)-only samples. Use the generalized-regression estimator that includes all \(n+m\) control observations.

### Theorem A2-G6 — correlation–cost gate

The minimum variance-cost factor relative to direct estimation is

\[
\boxed{
F(R^2,r)=
\begin{cases}
1, & R^2\le r,\\[4pt]
\left[
\sqrt r\,R+
\sqrt{1-r}\sqrt{1-R^2}
\right]^2,
& R^2>r.
\end{cases}
}
\]

The maximum risk-cost gain is

\[
\boxed{G(R^2,r)=1/F(R^2,r).}
\]

A nontrivial control helps **if and only if**

\[
\boxed{R^2>r.}
\]

#### Proof

With optimal linear regression, the estimator variance is

\[
\sigma_Y^2
\left[
\frac{1-R^2}{n}
+
\frac{R^2}{n+m}
\right].
\]

Its cost is

\[
Cn+cm=(C-c)n+c(n+m).
\]

For fixed homogeneity, minimize the product over total control count \(n+m\ge n\). Unconstrained Cauchy allocation gives

\[
\left[
\sqrt{(C-c)(1-R^2)}+
\sqrt{cR^2}
\right]^2\sigma_Y^2.
\]

The unconstrained solution has \(n+m\ge n\) exactly when \(R^2\ge r\). Otherwise the boundary \(m=0\) gives direct estimation. Divide by \(C\sigma_Y^2\). ∎

### Consequences

1. **Exact known control mean:** \(r=0\), so
   \[
   F=1-R^2,
   \qquad G=\frac1{1-R^2}.
   \]
   This is why layer 1 is uniquely valuable.

2. **Perfect control:** \(R^2=1\), so
   \[
   F=r,
   \qquad G=1/r=C/c.
   \]
   Even perfect prediction at a checkpoint costing 97% of a full trajectory can improve risk-cost by at most about 3%.

3. **Late-checkpoint trap:** \(R^2\to1\) does not imply a large gain when \(r\to1\).

4. **Geometric form:** if \(R=\cos\theta\) and \(\sqrt r=\cos\phi\), the nontrivial factor is \(\cos^2(\theta-\phi)\). Improvement requires the predictive angle to beat the cost angle.

5. **Multichannel extension:** replace scalar \(R^2\) by canonical correlations/generalized signal-to-noise eigenvalues in the physical score metric. The SOCP is the joint multilevel counterpart.

---

## 12. Architecture-matched numerical falsification

### Purpose

The numerical screen was not intended to estimate the actual competition source. It asked a narrower question:

> In challenge-shaped random ReLU networks, does the new gauge freedom generically turn exact checkpoint telescopes into a large variance-cost improvement?

### Environment

- width: 256;
- depth: 30;
- He-normal bias-free ReLU weights;
- scalar and rank-4 random terminal contractions;
- antithetic Gaussian trajectories;
- independent train/test trajectory sets;
- checkpoint sample cost proportional to depth;
- layer-1 expectation treated as exact;
- exposed no competition data and no protected cohort.

These are architecture-matched diagnostics, not official or source-specific evidence.

### 12.1 All-layer independent telescope

Every layer increment was estimated independently. The score was the variance-cost constant \(S\), divided by the direct terminal estimator’s constant.

| Gauge | Terminal rank | Median \(S/S_{direct}\) | Range | Wins over direct |
|---|---:|---:|---:|---:|
| T70 fixed \(1/2\) adjoint | 1 | 2.610 | 1.363–3.829 | 0/5 |
| soft expected-gate recursion | 1 | 2.804 | 2.164–3.046 | 0/5 |
| greedy local regression | 1 | 3.692 | 2.612–4.050 | 0/5 |
| global convex optimization | 1 | 1.000 | 0.893–1.000 | 2/5 |
| T70 fixed \(1/2\) adjoint | 4 | 2.240 | 1.991–3.082 | 0/5 |
| soft expected-gate recursion | 4 | 2.549 | 2.117–3.227 | 0/5 |
| global convex optimization | 4 | 1.000 | 0.924–1.000 | 1/5 |

The fixed adjoint looked dramatically improvable only because it was compared with a poor all-layer allocation. Once the direct estimator was included as a feasible gauge, the globally optimized median returned to essentially exactly one.

### 12.2 Fixed checkpoint partitions

A tournament tested partitions such as

\[
[1,30], [1,4,30], [1,8,30], [1,16,30],
[1,4,8,30],\ldots
\]

on six independent networks.

| Frozen partition | Median risk-cost gain | Mean | Range | Wins |
|---|---:|---:|---:|---:|
| \([1,30]\) | 0.977 | 1.003 | 0.844–1.198 | 3/6 |
| \([1,4,30]\) | 0.945 | 1.003 | 0.816–1.304 | 2/6 |
| \([1,8,30]\) | 0.984 | 1.043 | 0.844–1.336 | 3/6 |
| \([1,4,8,30]\) | 0.982 | 1.039 | 0.859–1.320 | 2/6 |

Selecting the partition on training covariance chose \([1,30]\) in five cases and \([1,16,30]\) once. Test median gain was 0.977, mean 1.036, with range 0.844–1.394 and 3/6 wins. The selected large win came from an optimizer that hit the iteration limit, so it cannot support a positive claim.

### 12.3 Correlation–cost diagnosis

The terminal scalar contraction was linearly predicted from each checkpoint. The table reports median test \(R^2\), checkpoint cost ratio \(r=\ell/30\), and the exact two-level risk-cost gain from A2-G6.

| Checkpoint | Median \(R^2\) | Cost ratio \(r\) | Median maximum gain |
|---:|---:|---:|---:|
| 1 | 0.125 | 0.000 | 1.143× |
| 2 | 0.167 | 0.067 | 1.026× |
| 4 | 0.291 | 0.133 | 1.040× |
| 8 | 0.498 | 0.267 | 1.061× |
| 12 | 0.616 | 0.400 | 1.049× |
| 16 | 0.774 | 0.533 | 1.069× |
| 20 | 0.839 | 0.667 | 1.042× |
| 24 | 0.925 | 0.800 | 1.036× |
| 27 | 0.971 | 0.900 | 1.023× |
| 29 | 0.990 | 0.967 | 1.007× |

Every tested checkpoint passed the bare \(R^2>r\) gate, yet none had a large median economic ceiling. Late checkpoints were extraordinarily predictive but too expensive to anchor independently. Layer 1 remained competitive because its expectation is exact.

### Interpretation

The screen aligns with historical project evidence:

- exact layer-1 controls can deliver modest real efficiency gains;
- deeper multifidelity controls often predict well but fail cost or phase transfer;
- all-layer decompositions can be much worse than direct estimation;
- adaptive partition selection can overfit covariance estimates.

The new result is not that linear controls never help. It is:

> Their exact best-case value can now be computed and certified before implementation. In a generic challenge-shaped ensemble, that value is modest and unstable, not winning-scale.

---

## 13. Relation to T72

Let a frozen physical source have oracle residual ratio \(r_*\). For unbiased independent contraction estimation, T72 requires

\[
(\sqrt{r_*}+S)^2<p,
\]

where \(p=1/4.34\) and \(S\) is the dimensionless contraction difficulty after cost normalization.

The checkpoint-gauge SOCP supplies the minimum \(S\) over the complete declared class of linear checkpoint-control telescopes for a fixed source, partition family, score metric, covariance model, and sample design.

The correct source-specific protocol is:

1. freeze \(U\) and measure \(r_*\);
2. estimate block covariance matrices on exposed independent designs;
3. solve the SOCP for each predeclared partition;
4. validate the selected controls on untouched covariance samples;
5. obtain primal and dual values;
6. compute the resulting T72 score lower bound;
7. stop if the certified lower bound misses the target.

### Why this is stronger than another control experiment

A failed fitted control only closes that fit. A tight SOCP primal/dual result closes **every** linear checkpoint control in the declared class simultaneously.

### What the SOCP does not close

- nonlinear checkpoint controls;
- conic fans or other exactly integrable nonlinear surrogates;
- target-dependent sources;
- joint estimators using nonstandard dependence across blocks;
- finite-width identities exploiting hidden row coupling;
- biased/shrunk estimators with favorable cross terms;
- computational sharing that makes the declared \(\gamma_j\) overstate cost;
- a different physical source \(U\).

---

## 14. Attempts to disprove the result

### Attack 1 — The telescope is just a notation change

**Partly true.** On a common sample cloud it collapses exactly to the terminal output. The useful content is the freedom to use an exact layer-1 expectation and unequal independent allocations. The same-sample caveat is explicit.

### Attack 2 — The T70 recursion is already optimal

False in general. T70 is locally optimal only for centered Gaussian preactivations. The soft-gate theorem gives the noncentered Gaussian optimum, and full covariance regression gives the general linear optimum.

### Attack 3 — Greedy regression should solve the global problem

False. Intermediate controls affect adjacent blocks, and the objective is a sum of square roots. The architecture screen found greedy regression substantially worse than direct estimation.

### Attack 4 — The convex program must always improve because it has many variables

False. Direct estimation is feasible, and equality is common. The optimizer often selects nearly zero intermediate controls.

### Attack 5 — High late-checkpoint correlation should yield a breakthrough

False economically. A checkpoint costing almost a full trajectory has a hard maximum gain near the reciprocal cost ratio, even with perfect correlation.

### Attack 6 — Training covariance proves the partition

False. The partition tournament showed selection instability and both large apparent wins and failures. Untouched covariance validation is mandatory.

### Attack 7 — The Gaussian soft gate is a legal fixed-network expectation

False. The local covariance identity is exact under a Gaussian law, not for a fixed finite-width realized distribution. Only the outer algebraic telescope is pathwise exact.

### Attack 8 — This closes all multilevel estimators

False. It closes only the declared linear checkpoint-control class under the stated covariance/cost/independence model.

---

## 15. Reconciliation with existing evidence

### T70/T71

- T70 is recovered exactly as the centered-slope gauge.
- T71’s total-potential no-op generalizes to every gauge under same-sample evaluation.

### Exact first-layer control

The prior first-band theorem becomes the free base level for arbitrary optimized gauges, not merely a fixed T70 component.

### Historical soft-Jacobian results

The Gaussian soft-gate theorem supplies the missing exact local rationale for using expected gates. It does not upgrade historical empirical gains to a theorem about the competition estimator.

### Historical multifidelity failures

The correlation-cost theorem explains why deeper controls with high predictability can still have little economic value. The full multilevel continuation’s phase instability remains a separate problem: low conditional variance does not identify the signed Kerdock defect.

### Layer-30 source program

The checkpoint gauge does not replace the layer-30 source. It is a possible contraction estimator after a legal low-rank source is frozen. The source-capacity gate remains prior.

---

## 16. Proposed source-specific experiment

This is the only continuation justified by the theorem.

### Stage A — freeze the physical source

Use Agent 1’s legal target-free source \(U\). Record:

- source construction and orientation;
- rank;
- physical score Gram;
- source-only oracle residual \(r_*\);
- per-network tails.

Stop if source capacity already fails.

### Stage B — covariance acquisition

On exposed independent angular designs, record checkpoint states at a preregistered small set, for example

\[
\{1,4,8,16,24,27,29,30\}.
\]

Estimate complete joint covariance blocks for the source-contracted checkpoint features. Use grouped network/rotation splits.

### Stage C — convex audit

For each frozen partition:

1. solve the SOCP;
2. report direct, T70, soft-gate, greedy-regression, and global-optimum objectives;
3. generate a dual lower bound;
4. report condition numbers and regularization sensitivity;
5. compute source-specific T72 score lower bounds.

### Stage D — untouched covariance validation

Apply the frozen controls to a disjoint covariance set. Reject if:

- \(S\) inflation removes the T72 margin;
- the selected partition changes materially;
- one generalized direction dominates noise;
- tails exceed the source-specific tolerance;
- the primal/dual gap is not small.

### Stage E — estimator only after a mathematical pass

Only then implement independent block sampling and exact late replay.

---

## 17. Provisional theorem and evidence rows

These IDs are intentionally namespaced and must not be inserted into the canonical ledger without coordinator collision review.

| Local ID | Claim | Status |
|---|---|---|
| `P-A2-T86` | Arbitrary checkpoint-control telescope for every feed-forward network and every terminal contraction | Proved exact algebra |
| `P-A2-T87` | Same-sample evaluation of every gauge collapses pathwise to the original terminal output | Proved exact algebra |
| `P-A2-T88` | Fixed-partition minimum T72 contraction difficulty over linear checkpoint controls is an SOCP; direct estimation is feasible | Proved convex formulation |
| `P-A2-T89` | Optimal local linear bridge is the covariance regression/Schur-complement solution | Proved linear algebra |
| `P-A2-T90` | For Gaussian preactivations, expected-gate backprop is the exact local linear optimum | Proved by Gaussian Stein identity |
| `P-A2-T91` | Two-level control helps iff \(R^2>c/C\), with exact minimum factor \(F(R^2,c/C)\) | Proved exact allocation theorem |
| `P-A2-M174` | Width-256/depth-30 architecture screen: fixed all-layer gauges fail; global optimum usually collapses to direct; fixed partitions show modest unstable gains | Numerical diagnostic, not competition evidence |

---

## 18. Verification and artifacts

The fast verifier checks:

- arbitrary telescope identity;
- T70 specialization;
- local regression normal equations;
- Gaussian soft-gate covariance identity;
- two-level closed form against numerical optimization;
- convexity of the fixed-partition objective.

All assertions pass.

Architecture-screen raw JSON and the exploratory reproduction scripts are included separately. Those screens are deliberately labeled numerical diagnostics.

---

## Bottom line

Agent 2’s continuation found a genuine structural generalization:

> **Every scalar late-source contraction admits infinitely many exact checkpoint gauges, and the best linear multilevel gauge for any fixed source and partition is a convex, certifiable optimization problem.**

The skeptical conclusion matters more for winning strategy:

> **When direct estimation is included as a feasible gauge, the generic architecture-matched optimum usually returns to direct estimation. High checkpoint predictability is largely canceled by the cost of learning the checkpoint’s absolute mean.**

Therefore this branch should not become another open-ended control-variate campaign. Its proper role is a bounded source-specific falsifier:

1. freeze a legal source;
2. solve the exact convex contraction frontier;
3. compare with T72;
4. continue only on a certified mathematical pass.
