# Agent 7 — Polyhedral Cone and Activation-Fan Compression

**Competition:** WHestBench / ARC White-Box Estimation Challenge 2026  
**Date:** 2026-07-30  
**Agent:** 7  
**Protected competition data opened:** No  
**Overall disposition:** No deployable candidate. Retain only a narrowly scoped low-normal-rank, output-weighted facet-DAG branch pending decisive oracle tests.

## Executive verdict

**No deployable candidate currently exists.**

The global activation-fan branch is not fully closed, but most of its proposed compression mechanisms admit strong mathematical obstructions:

1. **Exact BDD/ZDD compression by merging activation histories is generically impossible.** Distinct live activation histories almost surely produce distinct affine maps under continuously distributed weights.
2. **Merging regions with exactly identical downstream maps will capture only structural degeneracies**, such as dead paths, zero suffix sensitivities, or redundant constraints.
3. **Dominant-region enumeration with a rigorous neglected-mass bound requires extraordinarily high coverage** unless the neglected regions have very small output-weighted operator norm.
4. **Low activation-covariance participation ratio is not the relevant dimension.** The relevant dimension is the span of the input-space normals of output-relevant gate boundaries.
5. **Late-layer localization of the absolute-defect telescope does not imply late localization of the gate-current decomposition.** This is a central unresolved logical gap.
6. **Two-dimensional conditional boundary integration is exact but does not remove integration.** It produces a difficult outer expectation and potentially thousands of boundary events per slice.

The one surviving version is narrower:

> **A low-normal-rank, output-weighted facet DAG that integrates a small-dimensional quotient of the global activation fan, with a certified residual for omitted normals, maps, and regions.**

This remains viable only if two oracle facts hold:

- output-relevant gate-boundary normals have a very small global or piecewise-global span;
- early-layer gate currents either cancel, telescope, or have negligible output-weighted contribution.

Neither fact has yet been measured directly. Existing evidence gives them a low prior, but does not prove they fail.

The Agent-7 mandate correctly distinguishes this global fan question from earlier selected-gate and Gaussian-handoff attempts and demands certification rather than finite-sample pattern counting.

---

# 1. Reconciliation with the current ledger

The canonical v27 state says that static sampling, ordinary additional trajectories, low-rank activation particles, generic phase learning, and linear checkpoint gauges are not winning routes. The only currently established winning-scale oracle is the analytic mixed-moment closure, although no legal recurrence exists for it.

That does **not** automatically close activation-fan compression.

Three superficially similar branches must be kept separate.

## 1.1 Selected conic-source corrections

The previously tested A30 construction used 30 output-space source columns created from selected planes and rays. Its twelve-case oracle ratio was strong in aggregate but unstable: pooled `0.156212`, median `0.239277`, worst `0.311349`, with only five of twelve cases passing the zero-cost competition gate. Full unoptimized source construction cost was measured at `18.089%`.

Its coefficient-estimation variants were decisively noncompetitive, and the final report closes that exact conic source, its sampled estimators, same-design regressions, and low-piece descendants.

That is a closure of a **selected correction dictionary**, not a closure of the complete activation fan.

## 1.2 Conditional-Gaussian late-gate integration

The prior activation-region continuation selected a small number of influential layer-30 gates, enumerated their states, Gaussianized the remaining uncertainty, and integrated the final scalar ReLU. It demonstrated useful oracle sparsity at layer 31, but its signed correction had essentially no stable relationship to the complete Kerdock error.

In particular, twelve oracle-selected layer-31 coordinates captured 55.23% of the full layer-31 benefit, while 32 captured 81.39%. Coordinate selection therefore was not the principal obstruction; signed correction estimation was.

This closes **conditional Gaussian approximation of selected late gates as a phase predictor**. It does not close exact integration of the realized global PWL fan.

## 1.3 Low-piece maxout or conic surrogates

The prior exact-surrogate program showed that a homogeneous PWL network can in principle be represented by difference-of-maxout functions, but synthetic depth-32 outputs had a median of about 6,752 detected local-piece changes on a single 8,192-angle great circle. Even exact one-pair DC representation would therefore require roughly 58 pieces per side under the proved circle-switch lower bound.

The subsequently run three- and five-piece tournament was catastrophically negative. This closes that low-piece construction, but not every possible compressed representation of the fan.

---

# 2. Exact formulation of the activation-fan integral

Let

\[
h_0(x)=x,\qquad
z_\ell(x)=W_\ell h_{\ell-1}(x),\qquad
h_\ell(x)=\operatorname{ReLU}(z_\ell(x)).
\]

For a fixed complete activation pattern \(R\), the network is linear:

\[
f(x)=A_Rx,\qquad x\in C_R,
\]

where \(C_R\) is a polyhedral cone because the network is bias-free.

Let \(X\sim N(0,I_d)\). Write

\[
X=\mathcal R U,
\]

where \(U\) is uniform on \(S^{d-1}\), \(\mathcal R=\|X\|\), and \(\mathcal R\) and \(U\) are independent. Set

\[
\kappa_d=E\mathcal R
=
\sqrt{2}\,
\frac{\Gamma((d+1)/2)}{\Gamma(d/2)}.
\]

For

\[
\Omega_R=C_R\cap S^{d-1},
\qquad
q_R=\int_{\Omega_R}u\,d\sigma(u),
\]

with \(\sigma\) normalized spherical measure, we have the exact identity

\[
\boxed{
E[f(X)]
=
\kappa_d\sum_R A_Rq_R.
}
\tag{1}
\]

Thus each region contributes not merely its probability, but its **vector spherical first moment** transformed by its affine map.

This immediately corrects one tempting but inadequate diagnostic:

> Sorting activation regions by Gaussian mass is not sufficient. Regions must be sorted by output-weighted contribution \(A_Rq_R\), or by a valid upper bound on that contribution.

A small region can matter if its downstream map is large. A large region can contribute little if its first moment is weak or its map points into a low-weight output direction.

---

# 3. Cone first moments are facet currents

For a polyhedral cone \(C\), define

\[
m_C=E[X\,1_{\{X\in C\}}].
\]

Because

\[
\nabla\phi_d(x)=-x\phi_d(x),
\]

the divergence theorem gives

\[
m_C
=
-\int_{\partial C}
n_{\mathrm{out}}(x)\phi_d(x)\,
d\mathcal H^{d-1}(x).
\]

If \(F\) ranges over the facets of \(C\), with outward unit normal \(n_F\), then

\[
\boxed{
m_C
=
-\sum_F n_F\,\gamma_{d-1}(F),
}
\tag{2}
\]

where

\[
\gamma_{d-1}(F)
=
\int_F\phi_d(x)\,d\mathcal H^{d-1}(x)
\]

is the Gaussian surface mass of the facet.

Therefore

\[
E[A_RX1_{\{X\in C_R\}}]
=
-A_R\sum_{F\subset\partial C_R}
n_F\gamma_{d-1}(F).
\]

Summing over activation regions causes every shared facet to appear from its two neighboring regions. The difference of their affine maps is a rank-one gate-opening update. This reorganizes the region sum into a sum over ReLU boundary currents.

The exact gate-current identity already derived in the archive is

\[
E[G(X)]
=
\sum_{\ell,j}
\int_{\{u_{\ell j}=0\}}
s_{\ell j}(x)
\|\nabla u_{\ell j}(x)\|
\phi_d(x)\,
d\mathcal H^{d-1}(x),
\]

where \(s_{\ell j}\) is the downstream sensitivity of the final output to gate \((\ell,j)\).

This is not merely an alternative proof. It identifies the more compressed object:

> The natural primitive is a gate facet with a downstream vector, not a complete activation region.

A network may possess astronomically many regions but only \(Lw\) named gates. Unfortunately, each named gate surface is subdivided by later gate patterns because \(s_{\ell j}(x)\) is only piecewise constant. The identity removes redundancy but does not by itself solve complexity.

---

# 4. First scoped impossibility theorem: exact map merging is generically absent

Consider a complete gate pattern

\[
D=(D_1,\ldots,D_L)
\]

with diagonal zero-one gate matrices. Its affine map has the form

\[
A_D
=
W_{L+1}D_LW_L\cdots D_1W_1.
\]

## Theorem 1 — Generic distinctness of live activation maps

Suppose the weight entries are drawn from an absolutely continuous distribution. Let \(D\neq D'\) be two feasible patterns. Assume that at least one gate on which they differ lies on a live path from the input to a scored output.

Then

\[
P(A_D=A_{D'})=0.
\]

### Proof

Every entry of

\[
A_D-A_{D'}
\]

is a polynomial in the weight entries.

Because a differing gate lies on a live input-output path, one can choose a weight assignment that isolates that path. Under that assignment, the path contributes to one pattern and not the other, so the polynomial matrix is not identically zero.

The zero set of a nonzero polynomial has Lebesgue measure zero. An absolutely continuous weight law therefore assigns probability zero to equality. ∎

## Consequences

Exact equality can still occur when:

- a differing neuron has zero downstream sensitivity;
- all paths through it are structurally dead;
- a constraint is redundant and does not change the realized affine function;
- the output metric annihilates the map difference.

But these are degeneracies, not a general compression mechanism.

The ledger already found that final-layer dead neurons contributed none of the measured squared error, so even a successful dead-path merger is unlikely to capture the important residual.

### BDD implication

A reduced ordered BDD can merge two histories only if their remaining subproblems are equivalent. Here the exact state includes at least:

- the current affine prefix map;
- the current feasible cone;
- the downstream function induced by that prefix.

Under the theorem, two distinct live histories almost surely have distinct continuous states. Therefore an exact BDD based on map equality has essentially the same width as the feasible activation-history tree.

### ZDD implication

A ZDD is advantageous when the represented sets are sparse. Activation patterns in a deep ReLU network need not be sparse, and even identical active sets at one layer do not imply identical prefix maps. The continuous affine state remains the obstacle.

**Disposition:** close exact BDD/ZDD compression by ordinary subproblem or affine-map equality, except as an implementation device for structural dead paths and constraint redundancy.

---

# 5. Approximate map merging and its correct error bound

Exact equality is unnecessary if nearby maps can be grouped with a certified loss.

Let \(\mathcal G\) be a group of cones. Assign a representative map \(B_{\mathcal G}\). Let \(H\succeq0\) encode the scored output metric, and suppose

\[
\left\|
H^{1/2}(A_R-B_{\mathcal G})
\right\|_{\mathrm{op}}
\le \delta_R
\]

for every \(R\in\mathcal G\).

Since a union of activation cones depends only on direction,

\[
E[\|X\|1_{\{X\in\cup_{R\in\mathcal G}C_R\}}]
=
\kappa_d p_{\mathcal G},
\]

where \(p_{\mathcal G}\) is its Gaussian mass. Therefore

\[
\begin{aligned}
\left\|
\sum_{R\in\mathcal G}
E[(A_R-B_{\mathcal G})X1_{C_R}]
\right\|_H
&\le
\sum_{R\in\mathcal G}
\delta_R E[\|X\|1_{C_R}]\\
&=
\kappa_d
\sum_{R\in\mathcal G}p_R\delta_R.
\end{aligned}
\]

Thus

\[
\boxed{
\|\text{grouping bias}\|_H
\le
\kappa_d\sum_R p_R\delta_R.
}
\tag{3}
\]

This is the correct objective for approximate fan compression.

It is not enough to report:

- Frobenius similarity of affine maps;
- percentage of patterns merged;
- covariance explained;
- clustering reconstruction error on observed rows.

The relevant quantity is **Gaussian mass times output-weighted operator discrepancy**.

A useful clustering algorithm should therefore minimize

\[
\sum_R p_R
\left\|
H^{1/2}(A_R-B_{g(R)})
\right\|_{\mathrm{op}},
\]

or a safely computable upper bound.

---

# 6. Dominant-region integration and the neglected-mass requirement

Suppose a set \(\mathcal S\) of regions is integrated exactly. Let

\[
E=\bigcup_{R\notin\mathcal S}C_R
\]

be the neglected directional set, with probability \(\varepsilon\). Define

\[
L_E
=
\sup_{R\notin\mathcal S}
\|H^{1/2}A_R\|_{\mathrm{op}}.
\]

Then

\[
\begin{aligned}
\left\|
E[f(X)1_E]
\right\|_H
&\le
L_E E[\|X\|1_E]\\
&=
\kappa_d L_E\varepsilon.
\end{aligned}
\]

Therefore

\[
\boxed{
\|\text{neglected mean}\|_H^2
\le
\kappa_d^2L_E^2\varepsilon^2.
}
\tag{4}
\]

This is considerably sharper than a generic Cauchy–Schwarz bound because activation regions are conic and radial magnitude is independent of the omitted event.

To certify a target squared error \(\tau\), a sufficient condition is

\[
\varepsilon
\le
\frac{\sqrt{\tau}}{\kappa_dL_E}.
\tag{5}
\]

At \(d=256\), \(\kappa_d\) is approximately \(16\). As an illustration, when the normalized output Lipschitz factor is of order one and \(\tau\) is of order \(3\times10^{-7}\), the neglected directional mass must be only a few times \(10^{-5}\).

The exact numerical requirement depends on the official output normalization and \(H\), but the structural conclusion is robust:

> A worst-case mass certificate will usually require well above 99.99% directional coverage unless neglected maps have exceptionally small downstream norm.

Consequently, “the top hundred observed patterns cover much of the sample” would not be enough. The regions must cover nearly all **output-weighted mean contribution**, and the unseen remainder must be certified.

---

# 7. Finite-sample pattern counts can be converted into a valid certificate

The Agent-7 prompt explicitly warns that a small empirical pattern count is not a certification. There is, however, a clean holdout protocol.

Let a region set \(\mathcal S\) be discovered using one reference sample. Freeze it. Draw \(m\) independent Gaussian holdout points. Suppose none lands outside \(\mathcal S\).

If the true missing mass is \(\varepsilon\), then

\[
P(\text{zero holdout misses})
=
(1-\varepsilon)^m.
\]

Therefore, with confidence \(1-\alpha\),

\[
\boxed{
\varepsilon
\le
1-\alpha^{1/m}.
}
\tag{6}
\]

For small \(\varepsilon\),

\[
1-\alpha^{1/m}
\approx
\frac{\log(1/\alpha)}{m}.
\]

This is a valid mass certificate because discovery and certification are separated.

It still may be expensive statistically: certifying missing mass around \(10^{-5}\) at 95% confidence with zero misses requires on the order of \(3\times10^5\) independent holdout examples. That is acceptable for an offline oracle-ceiling study but not a runtime source of state.

More efficiently, one can certify the **residual contribution directly**. On the holdout sample, evaluate

\[
Y=f(X)1_{\{X\notin\mathcal S\}},
\]

and apply a vector concentration bound using a deterministic envelope from \(L_E\). This can be much sharper when omitted regions have small output sensitivity.

---

# 8. The relevant low dimension is boundary-normal rank

The observed deep covariance participation-ratio collapse is not sufficient.

Inside a fixed earlier activation history, every later preactivation is a linear form in the input:

\[
u_{\ell j}(x)=n_{\ell j,R}^{\top}x.
\]

The vector

\[
n_{\ell j,R}=\nabla_xu_{\ell j}(x)
\]

is the input-space normal of the corresponding gate boundary.

The dimension controlling cone integration is the rank of the collection of such normals, not the rank of the activation covariance.

## Theorem 2 — Exact low-normal-rank reduction

Let a cone be

\[
C=\{x:Nx\ge0\},
\]

and suppose the row space of \(N\) has dimension \(r\). Let \(U\in\mathbb R^{d\times r}\) be an orthonormal basis for that row space, so

\[
N=GU^\top.
\]

Write

\[
X=UY+U_\perp Z
\]

with independent standard Gaussian \(Y\) and \(Z\). Membership in \(C\) depends only on \(Y\), because

\[
NX=GY.
\]

Hence

\[
E[X1_C]
=
U E[Y1_{\{GY\ge0\}}],
\]

since the orthogonal Gaussian component has zero conditional mean. Therefore

\[
\boxed{
E[A_RX1_C]
=
A_RU\,E[Y1_{\{GY\ge0\}}],
}
\tag{7}
\]

an exactly \(r\)-dimensional cone integral.

This is the strongest possible exact dimension reduction. It does not require the affine output map itself to be low rank.

The archived low-subspace theorem for absolute defects has the same underlying structure and likewise emphasizes that a population residual, rather than a small residual on the Kerdock cloud, is required for certification.

---

# 9. Approximate normal rank has an exact sign-error formula

Suppose a unit gate normal \(n\) is approximated by its projection onto \(U\). Let

\[
a=\|U^\top n\|.
\]

The original score

\[
n^\top X
\]

and normalized projected score

\[
\frac{(UU^\top n)^\top X}{\|U^\top n\|}
\]

are jointly standard Gaussian with correlation \(a\).

For two centered jointly Gaussian variables with correlation \(a\), the probability that their signs differ is

\[
\boxed{
P(\text{gate sign mismatch})
=
\frac{\arccos(a)}{\pi}.
}
\tag{8}
\]

For a collection of \(q\) gate constraints with alignments \(a_i\), a union bound gives

\[
P(\text{any projected gate mismatch})
\le
\sum_{i=1}^q
\frac{\arccos(a_i)}{\pi}.
\tag{9}
\]

Combining this with the conic neglected-set bound gives a complete, if potentially conservative, certificate for a projected-normal fan.

If \(L_{\mathrm{mis}}\) bounds the output-weighted map norm on all directions where a gate mismatch occurs, then

\[
\boxed{
\|\text{projection-induced mean error}\|_H
\le
\kappa_dL_{\mathrm{mis}}
\min\left(
1,
\sum_i\frac{\arccos(a_i)}{\pi}
\right).
}
\tag{10}
\]

This theorem makes the decisive oracle experiment precise:

> Measure the output-relevant boundary normals and determine whether a target-free \(r\)-dimensional subspace makes the right-hand side small enough at \(r=2,4,8,12,16,32\).

A covariance eigenvalue spectrum is not an adequate substitute.

---

# 10. Exact integration in the reduced cone

For

\[
Y\sim N(0,I_r),
\qquad
C=\{GY\ge0\},
\]

the required objects are

\[
p_C=P(GY\ge0)
\]

and

\[
m_C=E[Y1_{\{GY\ge0\}}].
\]

These are not independent tasks. Let

\[
p_C(\mu)
=
P_{Y\sim N(\mu,I_r)}(GY\ge0).
\]

Differentiating the Gaussian density with respect to its mean yields

\[
\boxed{
m_C
=
\nabla_\mu p_C(\mu)\big|_{\mu=0}.
}
\tag{11}
\]

Thus an orthant-probability routine that supplies reliable derivatives provides the cone first moment needed for the network mean.

The facet formula offers another route:

\[
m_C
=
-\sum_F n_F\gamma_{r-1}(F).
\]

For \(r=2\), every facet is a ray and the integrals reduce to one-dimensional Gaussian terms. The archived conditional-plane construction gives the exact segment formula

\[
\int_{\text{edge}}\phi_2(y)\,d\mathcal H^1(y)
=
\phi(\rho)[\Phi(b)-\Phi(a)].
\]

It also proves that two dimensions are the minimum stable randomized slicing dimension: the inverse-angle factor has infinite expectation for random lines but finite expectation for random planes.

The problem is not the local edge integral. It is the number of edges and the remaining outer expectation. The archive explicitly identifies that outer variance as the main obstacle.

---

# 11. Assessment of each proposed Agent-7 mechanism

## 11.1 Binary decision diagrams

### Exact version

Generically no map-state merging, by Theorem 1.

### Approximate version

Possible only with a certified output-weighted map discrepancy such as Equation (3). A conventional Boolean BDD package is insufficient because the node state must include continuous error information.

### Verdict

**Exact BDD compression: scoped closure.**

**Approximate error-bounded DAG: remains a component of the low-normal-rank route.**

---

## 11.2 Zero-suppressed decision diagrams

ZDD compression would require activation sets or gate changes to be sparse in the chosen ordering. Even then, histories with the same set cardinality or similar support may have different affine prefix maps and different feasible cones.

### Verdict

**No independent mathematical advantage has been established.** Use only if the oracle audit discovers a genuinely sparse late-boundary event structure.

---

## 11.3 Merging identical downstream maps

Exact equality is generically absent except for structural dead paths.

### Verdict

**Closed as a general mechanism.**

Retain exact hashing as a diagnostic: any substantial collision rate would reveal unexpected architecture-specific algebra that deserves investigation.

---

## 11.4 Merging nearly identical maps

Mathematically valid under Equation (3).

The metric must be:

- output weighted;
- operator based;
- Gaussian-mass weighted;
- evaluated on untouched regions;
- combined with a residual certificate.

### Verdict

**Conditionally open**, but only as approximate fan compression, not as ordinary clustering.

---

## 11.5 Conic fan and hyperplane-arrangement methods

For \(q\) central hyperplanes in \(r\) dimensions, the maximum number of conic cells is

\[
2\sum_{j=0}^{r-1}\binom{q-1}{j}.
\]

For fixed \(r\), this is polynomial in \(q\), which is the main attraction of a true low-normal-rank reduction. But even modest \(r\) and hundreds of boundaries can produce a very large arrangement.

The prior two-dimensional slice diagnostic found thousands of local-piece changes, showing that low slice dimension alone does not guarantee a small realized fan. That result was synthetic and does not establish an impossibility theorem for the competition networks, but it is strong negative prior evidence.

### Verdict

**Open only if the realized output-weighted arrangement is much smaller than its worst-case bound.**

---

## 11.6 Low-dimensional latent projection followed by exact integration

This is the strongest surviving mechanism, but the projection target must be corrected:

- not activation covariance;
- not hidden-state PCA alone;
- not a subspace fitted only on baseline rows;
- but the span of output-relevant gate normals or gradients.

The projected-normal mismatch theorem supplies a direct certificate.

### Verdict

**Primary surviving oracle test.**

---

## 11.7 Branch-and-bound with Gaussian-mass upper bounds

This is mathematically sound.

At a partial activation node, let \(C_v\) be the current cone. Every descendant lies inside \(C_v\). Therefore

\[
P(C_v)
\]

upper-bounds the total descendant mass.

Moreover, if

\[
L_v
\ge
\sup_{R\text{ below }v}
\|H^{1/2}A_R\|_{\mathrm{op}},
\]

then all unexplored descendants contribute at most

\[
\kappa_dL_vP(C_v)
\]

in output norm.

An exact full-dimensional cone probability is unnecessary for an upper bound. Select a subset of the node's constraints. Removing constraints enlarges the cone, so

\[
P(C_v)
\le
P(\text{selected constraints}).
\]

Choosing a small set of nearly independent or strongly restrictive normals yields a low-dimensional orthant upper bound.

### Verdict

**Valid and necessary for certification.** Its usefulness depends on the bounds shrinking rapidly enough.

---

## 11.8 Exact treatment of dominant regions plus a bounded residual

The mathematics is clean, but Equation (5) shows that mass coverage alone is likely to be extremely demanding.

A more promising ranking is

\[
\text{priority}(R)
\approx
p_R
\|H^{1/2}A_R\|_{\mathrm{op}}
\]

or, when an oracle first moment is available,

\[
\|H^{1/2}A_Rm_R\|.
\]

### Verdict

**Retain as an oracle ceiling and branch-and-bound framework. Do not assume a small number of regions will suffice.**

---

## 11.9 Shared region structure across outputs

This is an exact and important economy.

All output coordinates share the hidden activation fan. Once a cone first moment \(m_R\) is known,

\[
A_Rm_R
\]

produces the complete output vector. There is no reason to enumerate separately for 256 outputs.

Across a single gate facet, the neighboring-map difference is rank one:

\[
A^+-A^-=s\,n^\top,
\]

where \(s\) is a downstream output vector and \(n\) is an input-space boundary normal. The contribution update is therefore

\[
(A^+-A^-)m
=
s(n^\top m).
\]

Only one scalar boundary moment is needed to scale the full downstream vector.

### Verdict

**Adopt.** Any implementation that runs separate region integrations per output is structurally wrong.

---

## 11.10 Recursive integration of only late boundaries

This is not yet justified.

The archived analysis distinguishes:

- the absolute-defect telescope, whose source energy is strongly late localized;
- the gate-current identity, which decomposes the Gaussian mean by boundary layer.

Late localization of one does not imply late localization of the other.

A successful late-only method therefore needs either:

1. a direct measurement that early gate-current layers are negligible in the scored metric; or
2. an exact telescope

\[
\sum_j J_{\ell j}
=
B_\ell-B_{\ell-1}+R_\ell
\]

with cheaply computable boundary states \(B_\ell\) and late residual \(R_\ell\).

This telescope is a genuine potential breakthrough.

### Verdict

**Open theorem target; no current algorithm.**

---

# 12. A scoped BDD impossibility result

The previous arguments support a useful class-level conclusion.

## Theorem 3 — No generic exact history compression

Consider an exact activation-history DAG whose node key after layer \(\ell\) is sufficient to determine:

- the feasible input cone;
- the current affine hidden-state map;
- the remaining network function.

Assume generic continuously distributed live weights.

Then, almost surely, two distinct feasible histories cannot be merged unless their differences occur only on gates that are functionally dead for every scored output.

### Proof sketch

A merge implies equality of the remaining piecewise-linear functions on an open subset. Piecewise-linear equality on an open cone implies equality of the corresponding affine maps there. This gives polynomial equalities in the weights. For any live differing gate, one can assign weights that isolate its contribution, proving the polynomial is nonzero. Thus equality occurs only on a measure-zero set. ∎

This does not rule out:

- approximate state merging;
- symmetry-induced equivalence in a specially structured deterministic network;
- compression of facet integrals without compression of region histories;
- low-dimensional projected fans.

It does close the optimistic idea that ordinary reduced BDD canonicalization will discover massive exact sharing in a generic realized network.

---

# 13. Proposed surviving algorithm: output-weighted facet DAG

I would not build a complete region enumerator. The mathematically best candidate is a **facet DAG**.

## 13.1 State

A DAG node stores

\[
(\ell,\ C,\ U,\ \mathcal N,\ \mathcal A,\ \mathcal E),
\]

where:

- \(\ell\) is the current layer;
- \(C\) is the current projected cone;
- \(U\) is an \(r\)-dimensional normal subspace;
- \(\mathcal N\) is the projected boundary representation;
- \(\mathcal A\) is a factorized affine-map or Jacobian state;
- \(\mathcal E\) is a certified residual-error budget.

## 13.2 Transition

When a gate boundary is crossed:

\[
A^+-A^-=s\,n^\top.
\]

Store the update as the pair \((s,U^\top n)\) rather than materializing a dense \(256\times256\) matrix.

The scalar moment associated with the crossing is calculated in the reduced cone. All 256 output coordinates are updated by scaling \(s\).

## 13.3 Approximate merging

Two nodes may merge only if all of the following are bounded:

1. symmetric difference in cone mass;
2. projected-normal sign mismatch;
3. output-weighted affine-map discrepancy;
4. discrepancy in all future boundary normals;
5. accumulated residual budget.

The merge criterion should be based on Equations (3), (9), and (10), not a heuristic embedding distance.

## 13.4 Pruning

For node \(v\), compute an upper bound

\[
B_v
=
\kappa_dL_v\bar p_v,
\]

where \(\bar p_v\) is a certified cone-mass upper bound.

Prune when the sum of all live \(B_v\) values is below the remaining error allowance.

## 13.5 Exact integration

For \(r\leq 2\), use exact angular/edge formulas.

For modest \(r\), compute both probability and first moment through

\[
m_C=\nabla_\mu p_C(\mu)|_{\mu=0},
\]

with audited numerical tolerances.

For larger \(r\), the branch is not plausibly semiexact unless additional graphical or product structure is proved.

---

# 14. Compute model

Let:

- \(M\): number of retained facet-DAG states or region groups;
- \(r\): reduced normal rank;
- \(q_v\): number of active constraints at state \(v\);
- \(k=256\): number of outputs;
- \(T_{\mathrm{cone}}(r,q_v,\eta)\): cost of a cone probability and first moment to tolerance \(\eta\).

A factorized event update costs approximately

\[
O(kr)
\]

rather than \(O(kd)\) or dense map storage per complete region.

The core arithmetic is approximately

\[
\boxed{
C_{\mathrm{fan}}
\approx
C_{\mathrm{discovery}}
+
\sum_{v=1}^{M}
\left[
T_{\mathrm{cone}}(r,q_v,\eta_v)
+
O(kr+q_vr)
\right]
+
C_{\mathrm{certification}}.
}
\tag{12}
\]

Dense storage of a \(256\times256\) map per region is immediately unattractive: \(10^5\) regions already correspond to roughly \(6.6\) billion stored scalar entries or comparable map-processing operations.

Rank-one factorization changes the map-update economics substantially, but it does not make the cone integrations free. The canonical approximate 10% arithmetic envelope is around 27.2B effective FLOPs, and all discovery, probability evaluation, error certification, and fallback work must be included.

This implies:

- millions of very cheap rank-one events might fit arithmetically;
- millions of generic orthant integrations will not;
- the branch needs both low \(r\) and substantial DAG sharing or pruning.

---

# 15. Decisive oracle-ceiling program

The goal of the oracle program is not to build the full estimator. It is to determine whether the required compressibility exists.

## Gate A — Exact affine-map collision audit

At selected layers and on untouched full-width networks:

1. generate activation patterns from a high-precision Gaussian reference;
2. reconstruct the exact local affine maps;
3. hash maps at strict numerical tolerance;
4. identify every collision;
5. classify collisions as dead-path, redundant-constraint, or unexplained;
6. repeat after random input rotations.

### Pass condition

A nontrivial fraction of output-weighted contribution, not merely pattern count, lies in exact collision groups.

### Expected result

Likely failure, by Theorem 1.

---

## Gate B — Contribution concentration

Using one reference half for discovery and the other for evaluation:

1. estimate each observed region's Gaussian mass;
2. estimate

\[
c_R=E[A_RX1_{C_R}];
\]

3. sort by \(\|c_R\|_H\);
4. measure residual mean after the top \(M\) regions or map clusters;
5. calculate a holdout upper bound for unseen contribution.

### Pass condition

A practically enumerable set yields a certified final error below `2.962e-7`, preferably with substantial slack.

Mass coverage without output-weighted contribution coverage does not pass.

---

## Gate C — Boundary-normal rank

For each relevant gate facet, calculate

\[
n_{\ell j,R}=\nabla_xu_{\ell j}
\]

and the downstream vector

\[
s_{\ell j,R}.
\]

Weight each normal by a proxy for its possible contribution, such as

\[
\|H^{1/2}s_{\ell j,R}\|
\,
\gamma_{d-1}(F_{\ell j,R})
\,
\|n_{\ell j,R}\|.
\]

Measure:

- unweighted normal rank;
- output-weighted rank;
- rank needed for 99%, 99.9%, and 99.99% weighted energy;
- the rigorous sign-mismatch bound from Equation (9);
- stability under networks and rotations.

Test

\[
r=2,4,8,12,16,32,64.
\]

### Pass condition

A rank small enough for cone integration also makes the certified sign-mismatch contribution compatible with the final target.

This is the most important unperformed oracle test.

---

## Gate D — Gate-current layer localization

Compute the true contribution of each gate layer in the exact boundary-current identity.

Measure both

\[
J_\ell
=
\sum_j
\int_{\{u_{\ell j}=0\}}
s_{\ell j}(x)\|\nabla u_{\ell j}(x)\|
\phi_d(x)\,d\mathcal H^{d-1}
\]

and cumulative residuals from keeping only late layers.

### Pass condition

A short late suffix preserves essentially all scored mean contribution, or the early layers exhibit a telescope that can be evaluated analytically.

Without this gate, late-layer activation-region enumeration has no logical foundation.

---

## Gate E — Exact reduced-fan integration

Grant the algorithm the oracle normal subspace \(U\), but not target means.

For each \(r\):

1. project relevant normals;
2. enumerate or branch over the reduced fan;
3. integrate cone first moments;
4. apply exact affine maps;
5. measure final mean error;
6. compare point estimate with the rigorous projection and omitted-region bounds.

### Pass condition

The oracle compressed class reaches the raw target with slack.

If it fails even with oracle \(U\) and oracle group selection, close the whole branch.

---

## Gate F — Legal construction

Only after Gates A–E pass:

- replace oracle subspaces with a target-free weight or baseline-transcript rule;
- freeze rank and pruning policies;
- run untouched grouped networks;
- charge every operation;
- validate random-rotation equivariance;
- inspect worst-network and tail behavior.

---

# 16. Hostile review

## 16.1 Is participation-ratio collapse doing hidden oracle work?

Yes, unless boundary-normal rank is measured directly.

Activation covariance concerns the distribution of \(h_\ell(X)\). Region complexity concerns the collection of gradients

\[
\nabla_x z_{\ell j}(X).
\]

These can have radically different ranks. A state may live near a low-dimensional cone while its orientation changes across many input directions.

The latest canonical work already warns that early activation states remain high dimensional despite deep participation-ratio collapse.

## 16.2 Is observed pattern compression a finite-sample illusion?

Potentially.

A region absent from \(10^6\) samples may still matter if its map norm is large enough. Discovery and certification must use independent samples, and the final claim must use a residual bound.

## 16.3 Does random rotation preserve the claimed structure?

A true Gaussian-mean method should transform equivariantly under an input rotation. A compression rule depending on arbitrary input coordinates, Kerdock basis labels, or unstable sign hashes may appear strong in one orientation and disappear in another.

## 16.4 Does grouping maps ignore cone geometry?

It can.

Two close maps on regions with oppositely directed first moments may be safe to merge, while two equally close maps on coherently oriented regions may not be. Equation (3) is safe; ordinary map clustering is not.

## 16.5 Does exact cone integration solve the scored estimator?

Only if the integrated fan represents the actual network mean or a residualized surrogate whose Kerdock value and target mean are both available.

A locally exact region calculation can still solve the wrong object, which was the central failure of the earlier conditional-Gaussian branch.

## 16.6 Does late error localization justify discarding early facets?

No. The absolute-innovation and gate-current decompositions partition the mean differently. This is one of the most important places where an attractive argument would currently be invalid.

---

# 17. Scoped impossibility statement

The following statement is justified now:

> For generic continuously distributed live weights, exact activation-history compression based on identical affine prefix maps or identical downstream linear maps provides no asymptotic reduction beyond structural dead paths and redundant constraints. A winning activation-fan method must therefore use approximate output-weighted merging, low-dimensional normal structure, cancellation identities, or certified pruning—not ordinary exact BDD reduction.

A stronger impossibility theorem for all activation-fan methods is not justified because the following remain unmeasured:

- output-weighted boundary-normal rank;
- gate-current layer localization;
- approximate map-cluster distortion under Gaussian mass;
- contribution concentration of exact region groups;
- branch-and-bound decay using actual cone masses.

---

# 18. Ledger-ready patch

| Provisional ID | Classification | Claim | Disposition |
|---|---|---|---|
| `A7F-T1` | Theorem | Gaussian mean of a homogeneous PWL fan is \(\kappa_d\sum_R A_Rq_R\); cone first moments equal sums of Gaussian facet currents. | Adopt |
| `A7F-T2` | Scoped theorem | Distinct feasible live activation histories have distinct affine maps almost surely under absolutely continuous weights. | Closes generic exact-map BDD merging |
| `A7F-T3` | Theorem | If cone normals span rank \(r\), its probability and first moment reduce exactly to an \(r\)-dimensional Gaussian cone problem. | Adopt |
| `A7F-T4` | Theorem | Projection of a gate normal onto \(U\) causes sign mismatch probability \(\arccos(\|U^\top n\|)/\pi\). | Basis for certified low-normal-rank approximation |
| `A7F-T5` | Theorem | Neglecting a conic directional set of mass \(\varepsilon\) contributes at most \(\kappa_dL\varepsilon\) in output norm. | Basis for branch-and-bound |
| `A7F-C1` | Scoped closure | Exact BDD/ZDD compression by ordinary history or affine-map equivalence is generically ineffective. | Close |
| `A7F-C2` | Scoped closure | Mass-only dominant-region claims without output-weighted residual bounds are insufficient. | Close as evidence standard |
| `A7F-D1` | Oracle diagnostic | Measure output-weighted gate-normal rank and exact projected-fan ceiling at \(r=2,\ldots,64\). | Highest-priority Agent-7 test |
| `A7F-D2` | Oracle diagnostic | Decompose exact gate-current contribution by layer and test for a layer telescope. | Highest-priority proof/mechanism test |
| `A7F-S1` | Strategic synthesis | No deployable activation-fan candidate; retain only low-normal-rank facet-DAG branch conditional on D1 and D2. | Provisional |

---

# 19. Final decision

## Close now

- exact BDD or ZDD reduction based on activation-pattern sharing;
- exact merging of generic live downstream affine maps;
- global dense enumeration of complete activation regions;
- dominant-region claims based only on observed frequency;
- low-dimensional projection justified by activation covariance participation ratio;
- late-only enumeration justified solely by absolute-defect localization;
- ordinary conditional-Gaussian enlargement of the prior ARC integrator;
- low-piece maxout descendants.

## Preserve

- the exact gate-current identity;
- facet rather than region organization;
- shared all-output rank-one gate updates;
- low-rank normal-space cone reduction;
- approximate map merging with a certified output-weighted bound;
- branch-and-bound using cone mass and local suffix norm;
- independent holdout certification of missing contribution.

## Decisive remaining question

\[
\boxed{
\text{Do the output-relevant gate currents of the realized network live in a
small, certifiable normal-space fan?}
}
\]

If the answer is no, Agent 7 should be closed as a competition route.

If the answer is yes, the surviving algorithm is not a generic activation-region enumerator. It is a **low-normal-rank, output-shared, error-certified facet DAG**, potentially combined with a new gate-current telescope that removes early-layer boundaries.

At present, this is a mathematically coherent but low-probability branch. It is weaker empirically than the analytic mixed-moment program because it lacks a measured winning oracle ceiling. Its value is that the next two tests are exact, binary, and capable of producing either a genuine new candidate class or a meaningful scoped impossibility result.

---

# Source artifacts consulted

- `Pasted text(66).txt`
- `WHestBench_Current_State_v27_20260730.md`
- `whestbench_canonical_research_ledger_20260730_reconciled_v27_full_experiment_synthesis.xlsx`
- `path_2_gate_current_continuation.md`
- `ARC_ACTIVATION_REGION_CONTINUATION_REPORT.md`
- `AGENT7_EXACT_SURROGATE_REPORT.md`
- `AGENT7_CONTINUATION_FINAL_REPORT.md`
