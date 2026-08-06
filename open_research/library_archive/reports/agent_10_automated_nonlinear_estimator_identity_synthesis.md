# Agent 10 — Automated nonlinear estimator and identity synthesis

## Executive verdict

**No submission-ready nonlinear estimator has emerged.** The current evidence supports a narrower and more useful conclusion:

1. The high-capacity direct-output source is real: its oracle residual ratio is about `0.07486`, so there is enough information in principle to exceed the 80% improvement target.
2. The complete tested linear, unbiased, independent-block checkpoint-gauge family is decisively closed.
3. Most obvious “nonlinear” extensions—shrinkage, norms, ratios, leave-one-group-out formulas, coordinatewise sign prediction, invariant spectral summaries—either:
   - algebraically reduce to a closed linear estimator;
   - contain only error magnitude and not signed phase;
   - are second-order perturbations too small to produce an order-one improvement;
   - or obtain order-one behavior only through singular ratios and unacceptable tails.
4. There are only two mathematically genuine escapes:
   - **analytic propagation of the missing distributional state**, now the primary canonical path;
   - **an exact or nearly free activation-boundary identity** for the transported late absolute innovations.

The latest canonical synthesis accordingly ranks compressed analytic mixed-moment propagation first and nonlinear late innovation only as a narrow dormant hedge. The former has an oracle raw MSE near `1.30e-7`, a dense one-step next-variance error of `0.288%`, and a provisional compressed-compute estimate near `3B` FLOPs; the missing piece is a legal, realizable 32-layer recurrence.

My strongest new result is a **class-level trichotomy**:

> A nonlinear estimator can materially escape the closed linear class only by carrying a new phase-bearing representation, using an unstable singularity, or evaluating distributional activation-boundary mass.

That substantially reduces the search space.

---

## 1. Canonical problem statement

Write the target final mean as

\[
\mu(W)=P f_W,
\]

the Kerdock estimator as

\[
B(W)=Q f_W,
\]

and the correction to be recovered as

\[
E(W)=\mu(W)-B(W).
\]

An Agent 10 estimator is

\[
\widehat\mu(W)=B(W)+C(T(W)),
\]

where `T(W)` is the legal transcript: group outputs, layer summaries, weights, adjoints, same-sample products, and any explicitly charged additional calculations.

The current score-law reference uses:

\[
S=MSE\cdot\max\!\left(0.1,\frac{C}{B_{\rm budget}}\right).
\]

The current audit reference is `1.481e-7`, so the 80% target is `2.962e-8`. At or below the 10% compute floor, this permits raw MSE up to `2.962e-7`.

The direct-output source has strong oracle capacity—rank approximately 34–38 and pooled residual ratio `r_* = 0.0748607`—but the source’s signed contractions are not legally observable through the tested checkpoint family.

The exact late-innovation decomposition explains where the correction lives:

\[
\delta_{31}
=
\sum_{\ell=0}^{31}\xi_\ell R_\ell,
\qquad
\xi_\ell=(P-Q)|z_\ell|.
\]

Layers 28–31 dominate. The exact contribution matrix has effective rank near four, with its top five singular directions containing about `96.5%` of contribution energy. The exact last four innovations leave residual ratio `0.0607`, but they use unavailable target expectations.

So the target of synthesis is no longer vague:

\[
\boxed{
\text{Recover one joint, signed, transported absolute-value correction cheaply.}
}
\]

---

## 2. Risk geometry: the first mandatory theorem filter

Let `C` be any proposed correction vector. Permit an optimally chosen frozen scalar `\alpha`:

\[
\widehat\mu=B+\alpha C.
\]

Define the Hilbert-space inner product over the development population by

\[
\langle U,V\rangle=\mathbb E[U^\top V].
\]

Then

\[
R(\alpha)
=
\mathbb E\|E-\alpha C\|^2
=
\|E\|^2-2\alpha\langle E,C\rangle+\alpha^2\|C\|^2.
\]

Hence

\[
\alpha_*=
\frac{\langle E,C\rangle}{\|C\|^2},
\]

and

\[
\boxed{
\frac{R(\alpha_*)}{\|E\|^2}
=
1-\rho(E,C)^2
}
\]

where

\[
\rho(E,C)=
\frac{\langle E,C\rangle}{\|E\|\,\|C\|}.
\]

This applies whether `C` is biased, nonlinear, dependent on the baseline sample, or has a nonzero mean. Bias is not a special escape from risk geometry.

### Consequences

At unchanged score multiplier, reaching residual ratio `0.20` requires

\[
|\rho(E,C)|\ge\sqrt{0.8}\approx0.8944.
\]

A candidate costing twice as much needs raw ratio at most `0.10`, requiring

\[
|\rho|\ge\sqrt{0.9}\approx0.9487.
\]

Thus a weakly correlated nonlinear expression cannot be saved by clever coefficient fitting.

For an adaptive scalar `\alpha(U)` based on a legal confidence statistic `U`,

\[
\inf_{\alpha(U)}
\mathbb E\|E-\alpha(U)C\|^2
=
\|E\|^2
-
\mathbb E\left[
\frac{\mathbb E[E^\top C\mid U]^2}
{\mathbb E[\|C\|^2\mid U]}
\right].
\]

This is the correct oracle ceiling for thresholding, confidence weighting, abstention, and heteroscedastic shrinkage.

### Search implication

The first empirical quantity for every expression should be **cross-fitted correction cosine**, not ordinary target `R^2`, pairwise reconstruction error, sign accuracy, or source-span capacity.

---

## 3. Stable-nonlinearity collapse theorem

This is the main grammar-level obstruction.

Suppose the rotation-varying portion of the legal transcript is small:

\[
T_\varepsilon=T_0+\varepsilon U,
\]

and the desired correction has scale

\[
E_\varepsilon=\varepsilon V+O(\varepsilon^2).
\]

Let `\Phi` be a twice continuously differentiable estimator with bounded derivatives near `T_0`, and require `\Phi(T_0)=0`. Taylor expansion gives

\[
\Phi(T_\varepsilon)
=
\varepsilon D\Phi(T_0)U
+
\frac{\varepsilon^2}{2}
D^2\Phi(T_0)[U,U]
+
O(\varepsilon^3).
\]

Therefore

\[
\boxed{
\Phi(T_\varepsilon)
=
\text{a linear estimator at leading order}
+
O(\varepsilon^2).
}
\]

Because squared risk is `O(\varepsilon^2)`, the nonlinear term changes the normalized risk by only `O(\varepsilon)`, unless the derivatives of `\Phi` grow like `1/\varepsilon`.

### Interpretation

A smooth product, norm, soft threshold, bounded activation, or regular ratio applied to small group deviations cannot provide a winning-scale escape if its first derivative belongs to a closed linear class.

To obtain order-one extra improvement, the expression must do one of the following:

1. Introduce a genuinely new phase-bearing primitive.
2. Use a derivative or normalization whose condition number grows like `1/\varepsilon`.
3. Be nonsmooth exactly where the transcript frequently lies.
4. Access a distributional term absent from ordinary pointwise calculus.

Items 2 and 3 imply severe tail risk. Item 4 turns out to mean activation-boundary measure.

### Scope

This theorem closes stable nonlinear wrappers around the tested transcript classes. It does **not** close arbitrary full-weight algorithms whose leading-order coefficients use previously untested orientation-bearing contractions of the entire weight tensor.

---

## 4. Symmetry and phase-support theorem

Let a compact group `G` represent input rotations. For a fixed underlying network orbit, let

\[
F(q)=Q f_{W,q},
\qquad
\mu=\int_G F(q)\,dq,
\qquad
E(q)=\mu-F(q).
\]

Then

\[
\int_G E(q)\,dq=0.
\]

If a statistic `I(q)` is invariant on the orbit, then every estimator `\phi(I(q))` is constant on that orbit. Its optimal constant under squared loss is zero. Therefore

\[
\boxed{
\text{Orbit-invariant information cannot predict orbit-specific signed error.}
}
\]

This formally rejects estimators based solely on:

- singular values;
- covariance spectra;
- participation ratios;
- source energies;
- norms;
- condition numbers;
- error-magnitude predictions;
- invariant rank selectors.

### Representation-valued extension

Suppose a feature transforms under a representation `\rho` of `G`, and the target component lies in representation `\sigma`. A degree-`k` polynomial in that feature transforms within

\[
\rho^{\otimes k}.
\]

By Schur orthogonality, it can correlate with the target only if `\sigma` occurs in the decomposition of `\rho^{\otimes k}`.

This gives the grammar a formal **phase-support type**:

\[
\operatorname{support}(uv)
\subseteq
\operatorname{support}(u)\otimes
\operatorname{support}(v).
\]

Expressions whose representation support cannot contain the target should be rejected before fitting.

### Important distinction

A norm may be useful for amplitude or confidence after a phase-bearing direction already exists. It cannot create that direction.

---

## 5. Exact shrinkage theorem

Consider shrinkage toward any rotation-invariant analytic anchor `A(W)`:

\[
\widehat\mu_\lambda
=
(1-\lambda)F(q)+\lambda A.
\]

Let

\[
V=\int_G\|F(q)-\mu\|^2\,dq,
\qquad
D=\|\mu-A\|^2.
\]

Because `\int_G(\mu-F(q))\,dq=0`,

\[
R(\lambda)
=
(1-\lambda)^2V+\lambda^2D.
\]

The optimum is

\[
\lambda_*=\frac{V}{V+D},
\]

with

\[
\boxed{
R_*=\frac{VD}{V+D},
\qquad
\frac{R_*}{V}=\frac{D}{V+D}.
}
\]

To achieve ratio `0.20`, the anchor must satisfy

\[
D\le0.25V.
\]

In words:

> An invariant anchor must already have at least four times lower MSE than the baseline orbit error before shrinkage toward it can produce an 80% reduction.

This closes generic James–Stein-style shrinkage toward Gaussian propagation, a population mean, a scale model, or another weak analytic anchor. Adaptive invariant shrinkage can stratify the formula but cannot recover phase.

The primary analytic mixed-moment program is different: its oracle anchor may actually satisfy this condition. If made legal, it is already the solution rather than a minor shrinkage control.

---

## 6. Coordinatewise sign prediction cannot be moderately good

Suppose an oracle supplies the exact correction magnitude, and an estimator either predicts its sign or abstains.

For a covered fraction `c` of error energy and weighted sign accuracy `p`, the residual ratio is

\[
r
=
1-c+4c(1-p)
=
1+c(3-4p).
\]

To improve at all relative to abstention requires

\[
p>0.75.
\]

To reach `r\le0.20`,

\[
c(4p-3)\ge0.80.
\]

Thus:

- at full coverage, `p\ge0.95`;
- at perfect accuracy, coverage must be at least `80%`;
- at `90%` accuracy, coverage must be greater than `133%`, which is impossible.

The existing oracle-magnitude nonlinear phase test covered only `4.10%` of error energy and had weighted sign accuracy `43.64%`, producing a confirmation ratio worse than one.

So “slightly above-chance nonlinear sign learning” is not a plausible route. It must be almost perfect.

---

## 7. Randomization and U-statistics do not evade information limits

Let an estimator use internal randomness `U`:

\[
C=C(T,U).
\]

Conditioning on the legal transcript,

\[
\mathbb E\big[\|E-C\|^2\mid T\big]
=
\left\|E-\mathbb E[C\mid T]\right\|^2
+
\mathbb E\left[
\|C-\mathbb E[C\mid T]\|^2
\mid T
\right].
\]

Therefore

\[
\boxed{
\text{Rao–Blackwellizing over internal randomness never increases risk.}
}
\]

Random sign controls, randomized group partitions, stochastic expression selection, and randomized U-statistics need only be considered through their deterministic conditional mean.

A randomized zero-mean correction conditional on `T` strictly adds variance.

Dependence on the baseline sample can matter, but only because it creates a deterministic phase-bearing function of that sample—not because random dependence itself is beneficial.

---

## 8. Ratios are either perturbatively linear or tail-unstable

Ratios are a natural proposed loophole:

\[
C=\frac{A(T)}{B(T)}.
\]

There are two regimes.

### 8.1 Denominator bounded away from zero

If `|B|\ge b_0>0`, the ratio is smooth. The stable-nonlinearity theorem applies, and its leading term lies in the linearized class.

### 8.2 Denominator can approach zero

Suppose `B` has a continuous density positive near zero and

\[
\mathbb E[A^2\mid B=b]\to c>0.
\]

Then

\[
\mathbb E\left[\frac{A^2}{B^2}\right]=\infty.
\]

Regularizing,

\[
\frac{A}{B^2+\tau^2},
\]

gives Lipschitz scale `O(\tau^{-2})`. To turn an `O(\varepsilon^2)` nonlinear term into an `O(\varepsilon)` correction requires `\tau=O(\varepsilon)`, precisely the regime where variance and worst-case tails become large.

Therefore every ratio primitive should carry one of:

- a symbolic denominator lower bound;
- a finite second-moment proof;
- a clipping-bias calculation;
- or automatic rejection.

---

## 9. Rotation derivatives: the exact nonlocal identity

There is an exact identity on the input-rotation group.

Let `\Delta_G` be the Laplace–Beltrami operator and `\Pi_0` the Haar projection. Then

\[
E(q)=\Pi_0F-F(q)
=
\int_0^\infty e^{t\Delta_G}\Delta_GF(q)\,dt.
\]

Spectrally, if

\[
F=F_0+\sum_{\lambda>0}F_\lambda,
\qquad
\Delta_GF_\lambda=-\lambda F_\lambda,
\]

then

\[
\int_0^\infty e^{t\Delta_G}\Delta_GF_\lambda\,dt
=
-F_\lambda.
\]

This is an exact expression for the signed error from rotational derivatives.

### Why finite curvature is insufficient

A finite differential rule `p(\Delta_G)F` would need

\[
p(0)=0,
\qquad
p(-\lambda)=-1
\]

for every nonconstant eigenvalue. No finite polynomial can satisfy these infinitely many conditions.

A rational approximation or heat-semigroup approximation can be accurate, but it is nonlocal. It requires either:

- evaluations at additional rotations;
- a global spectral model;
- or an analytic representation of the rotational harmonic content.

This explains the curvature evidence: finite local curvature carries some error information, but its capacity and pass cost are far from winning. Four finite-difference directions produced a confirmation residual around `0.862` under frozen coefficients, and nine total passes made even oracle unions uneconomic.

### Agent 10 conclusion

A local derivative grammar is not enough. A successful rotation identity must implement an approximate inverse Laplacian, not merely estimate a few directional curvatures.

---

## 10. Distributional Stein identity: the genuine local escape

This is the most important constructive mathematical identity found.

Let `g:\mathbb R^d\to\mathbb R` be positively one-homogeneous and piecewise linear. Euler’s identity gives, almost everywhere,

\[
x^\top\nabla g(x)=g(x).
\]

For `X\sim N(0,I)`, Gaussian integration by parts yields, distributionally,

\[
\mathbb E[g(X)]
=
\mathbb E[\Delta g(X)].
\]

For the absolute value of a sufficiently regular scalar function,

\[
\Delta |g|
=
\operatorname{sign}(g)\Delta g
+
2\delta(g)\|\nabla g\|^2.
\]

Thus, under generic transversality or a smoothing limit,

\[
\boxed{
P|g|
=
P\!\left[
\operatorname{sign}(g)\Delta g
\right]
+
2P\!\left[
\delta(g)\|\nabla g\|^2
\right].
}
\]

For a ReLU composition,

\[
\Delta\operatorname{ReLU}(g)
=
1_{\{g>0\}}\Delta g
+
\delta(g)\|\nabla g\|^2.
\]

Consequently, for a late preactivation

\[
z_{\ell,j}
=
\sum_i W_{\ell,ji}\operatorname{ReLU}(z_{\ell-1,i}),
\]

the distributional Laplacian obeys

\[
\Delta z_{\ell,j}
=
\sum_i W_{\ell,ji}
\left[
1_{\{z_{\ell-1,i}>0\}}\Delta z_{\ell-1,i}
+
\delta(z_{\ell-1,i})
\|\nabla z_{\ell-1,i}\|^2
\right].
\]

Therefore `P|z_{\ell,j}|` is exactly a sum of:

- terminal zero-surface mass `z_{\ell,j}=0`;
- upstream ReLU-boundary masses;
- signs of the downstream preactivation evaluated on those boundaries;
- gradient-jump magnitudes.

### Why this matters

The missing signed phase is literally an **activation-boundary surface measure**.

Ordinary pointwise automatic differentiation does not see it. Away from activation boundaries, a ReLU network is affine in its input and its Euclidean Hessian is zero. A finite Kerdock design almost surely contains no point exactly on a boundary. Therefore a grammar using only:

- activations;
- ordinary first derivatives;
- ordinary pointwise second derivatives;
- norms of pathwise Jacobians;

cannot reconstruct the distributional delta terms.

It must instead use one of:

1. finite perturbations that cross boundaries;
2. explicit boundary enumeration;
3. a probabilistic model of boundary density;
4. an analytic characteristic-function or mixed-moment recurrence.

This explains why the surviving Agent 10 path merges with the canonical analytic-distribution path rather than creating an unrelated estimator.

---

## 11. Characteristic-function form of the same obstruction

For any integrable scalar `Z`,

\[
\mathbb E|Z|
=
\frac{2}{\pi}
\int_0^\infty
\frac{1-\operatorname{Re}\phi_Z(t)}{t^2}\,dt,
\qquad
\phi_Z(t)=\mathbb E[e^{itZ}].
\]

Hence

\[
(P-Q)|z_{\ell,j}|
=
\frac{2}{\pi}
\int_0^\infty
\frac{
\operatorname{Re}\phi^{Q}_{\ell,j}(t)
-
\operatorname{Re}\phi^{P}_{\ell,j}(t)
}{t^2}\,dt.
\]

The exact late innovation can therefore be obtained from either:

- boundary densities, via distributional Stein;
- or complete marginal characteristic functions.

Finite moments are Taylor information near `t=0`. Boundary and absolute-value behavior depends on a much broader frequency range.

This gives a useful unification:

> Late absolute-innovation synthesis, characteristic-function closure, and mixed-cumulant propagation are three representations of the same missing distributional state.

The primary analytic path’s third/fourth-order closure is a controlled finite approximation to this object. The challenge is not finding another algebraic wrapper around Kerdock outputs; it is transporting enough of the characteristic/boundary state through depth.

---

## 12. Transcript-only collision theorem

Let `T(W)` be any transcript and `\mu(W)` the target. If two valid networks satisfy

\[
T(W_0)=T(W_1),
\qquad
\mu(W_0)\ne\mu(W_1),
\]

then every transcript-only estimator has the same output `a` on both. Therefore

\[
\max\left(
\|a-\mu(W_0)\|^2,
\|a-\mu(W_1)\|^2
\right)
\ge
\frac14
\|\mu(W_0)-\mu(W_1)\|^2.
\]

Under an equal two-point prior, the Bayes risk is also at least the quarter-separation.

A local differential version is useful. If network parameters are `\theta`, and there exists

\[
v\in\ker DT(\theta)
\quad\text{with}\quad
D\mu(\theta)v\ne0,
\]

then the transcript is locally insufficient. Under constant-rank assumptions, the level set of `T` contains directions along which the target changes.

This theorem does not apply when the full weight tensor is included verbatim in `T`. In that case the remaining barrier is computational, not informational. It does apply to claims that the finite Kerdock group-output transcript or a small collection of summaries universally determines the correction.

A benchmark-relevant stop theorem would need either:

- a collision inside a typical He-initialized region;
- a conditional-variance lower bound under the actual network distribution;
- or a circuit/query lower bound for extracting the required weight-dependent boundary information.

---

## 13. Typed estimator grammar

Every expression node should carry the following types.

### 13.1 Geometric type

- scalar invariant;
- layer-`\ell` neuron vector;
- layer-`\ell` covector/adjoint;
- map `V_k\to V_\ell`;
- output vector;
- Kerdock-group-indexed field;
- input-rotation representation or harmonic support.

### 13.2 Gauge type

Under positive hidden gauge transformations,

\[
h_\ell\mapsto D_\ell h_\ell,
\qquad
W_{\ell+1}\mapsto W_{\ell+1}D_\ell^{-1},
\]

each expression records its exponent under every `D_\ell`. Only balanced contractions may become output corrections.

### 13.3 Permutation type

Under hidden-neuron permutations, vectors and tensors must transform naturally. Arbitrary coordinate selection is illegal unless it is defined equivariantly, such as by a nondegenerate spectral projector with a sign/gauge convention.

### 13.4 Information tag

- target-free;
- uses only weights;
- uses baseline rows;
- uses baseline group reductions;
- requires extra network evaluations;
- uses learned corpus coefficients;
- oracle-only.

### 13.5 Expectation tag

- exact known mean;
- exact zero mean;
- conditionally zero mean;
- unknown mean;
- biased with calculable bias.

### 13.6 Stability tag

- global Lipschitz bound;
- denominator lower bound;
- clipping threshold;
- condition number;
- finite second/fourth moment;
- PSD/feasibility guarantee.

### 13.7 Cost tag

Charge separately:

- reductions over the 66,048 rows;
- matrix-vector and matrix-matrix contractions;
- Walsh transforms;
- reverse adjoints;
- JVP/VJP/Hessian-vector products;
- extra network passes;
- eigendecompositions;
- expression-evaluation overhead.

---

## 14. Mandatory symbolic normalization

Before any expression is evaluated, the engine should attempt to prove that it belongs to an already-closed class.

### 14.1 Leave-one-group-out elimination

For `m` group means `Y_b`,

\[
\bar Y_{-b}
=
\frac{m\bar Y-Y_b}{m-1}.
\]

Every affine expression in leave-one-group-out estimates is merely an affine expression in the original group outputs.

### 14.2 Linear-expectation collapse

If

\[
C=\sum_r a_r(T)Z_r
\]

and the coefficients `a_r` are frozen invariants, this is a linear control. It must be compared directly with the relevant checkpoint-gauge or group-output linear class.

### 14.3 Antithetic identity

For pair-even and pair-odd preactivations `u,v`,

\[
\frac{
\operatorname{ReLU}(u+v)+
\operatorname{ReLU}(u-v)
}{2}
=
\frac{u+\max(|u|,|v|)}{2}.
\]

This is exact. It shows that nonlinear antipodal propagation introduces a max/boundary statistic; it does not create a new known expectation.

### 14.4 Homogeneity simplification

Expressions involving `z`, gradients, and radial scale should be simplified using Euler identities before being treated as new controls. Several apparent Stein identities reduce pointwise to the original function.

### 14.5 Randomization removal

Replace every randomized expression by its conditional mean given the deterministic transcript.

### 14.6 Leading-order extraction

Compute `D\Phi(T_0)`. If the leading term is in a closed linear family and higher terms are stable, reject the expression under the stable-nonlinearity theorem.

### 14.7 Harmonic-support rejection

Reject any expression whose input-rotation representation cannot contain the target phase.

---

## 15. Candidate families, ranked

### Rank 1 — Analytic compressed mixed-moment recurrence

This is no longer merely an Agent 10 side idea; it is the primary canonical program.

The known one-step bivariate formula is accurate enough under oracle moments. The remaining tasks are:

- propagate the required mixed cumulants analytically;
- preserve `\kappa_{iij}`, `\kappa_{ijj}`, and downstream quadratic forms;
- maintain PSD covariance;
- maintain mutual realizability of mean, covariance, skew, and kurtosis;
- generate Tucker factors from the prior legal state rather than oracle activations;
- survive 32 free layers.

**Verdict:** highest probability of a real win.

---

### Rank 2 — Boundary-state recurrence for only late transported innovations

Use the exact identity

\[
P|z|
=
P[\operatorname{sign}(z)\Delta z]
+
2P[\delta(z)\|\nabla z\|^2].
\]

Try to propagate only the boundary statistics needed for

\[
\sum_{\ell=28}^{31}\xi_\ell R_\ell,
\]

rather than a complete joint distribution.

Required state might include:

- density of selected preactivations at zero;
- gradient norm conditional on zero crossing;
- sign of downstream preactivations on upstream boundaries;
- low-rank adjoint-weighted sums of these quantities.

The challenge is that the recursive `\Delta z` term introduces sign-conditioned upstream boundary measures, which are joint copula objects. It may quickly become equivalent in complexity to mixed-moment transport.

**Verdict:** mathematically genuine but currently only a reformulation. Promote only if the adjoint contraction allows major cancellation or state reduction.

---

### Rank 3 — Nonlocal rotational Poisson approximation

Approximate

\[
E=\int_0^\infty e^{t\Delta_G}\Delta_GF\,dt
\]

with a rational spectral filter.

A viable implementation would need to exploit:

- a small set of structured rotations with shared arithmetic;
- Kerdock/Clifford association algebra;
- or an analytic model of the rotation spectrum.

Generic finite differences are already too costly and weak. A few local derivatives cannot approximate the inverse Laplacian over the heavy non-bandlimited spectrum.

**Verdict:** exact mathematics, poor current economics.

---

### Rank 4 — Full-weight phase-bearing contractions

The full first-layer weight orientation contains the rotation phase. In principle, an equivariant contraction of all weights could predict the error even when invariant summaries cannot.

Candidate primitives should be tied to the Kerdock discrepancy tensors:

\[
M_k(Q)=Q[x^{\otimes k}]-P[x^{\otimes k}],
\]

which vanish through the design degree and begin at higher harmonics. A network-specific error expression can be viewed schematically as

\[
(P-Q)f_W
=
\sum_{k\ge6}
\langle C_k(W),M_k(Q)\rangle.
\]

The hard problem is computing the high-degree network coefficients `C_k(W)` cheaply. A bounded-degree polynomial grammar will miss the substantial high-degree tail; a deep recursive grammar becomes polynomial-chaos or analytic moment transport.

**Verdict:** logically open, but likely converges to the primary analytic path or the full-weight operator path.

---

### Rank 5 — Joint equivariant central-moment vectors of group outputs

One narrowly scoped empirical family remains conceivable:

\[
C_r
=
\sum_{b=1}^{129}
\psi_r\!\left(
D_b^\top A D_b
\right)D_b,
\qquad
D_b=Y_b-\bar Y,
\]

where `A` is an equivariant downstream-adjoint metric and `\psi_r` is a fixed low-degree polynomial or bounded odd function.

Unlike norms alone, this produces an output-oriented vector. It is nonlinear, permutation-symmetric over bases, and cheap.

However:

- it is a smooth wrapper around the group transcript;
- its leading term is likely in a closed linear span;
- norm-derived weights primarily encode magnitude;
- same-design residual regression and full nonlinear phase predictors have already failed;
- expression-search multiplicity would be severe.

**Verdict:** at most one bounded falsification screen. Do not elevate to a lead path.

---

## 16. Families that should be automatically rejected

1. **Pure invariant shrinkage.** Closed by the anchor theorem unless the anchor already has less than `0.25` of baseline MSE.
2. **Norm-only or spectrum-only estimators.** No signed phase.
3. **Coordinatewise sign classifiers.** Need approximately `95%` weighted accuracy at full coverage.
4. **Unregularized ratios.** Infinite or uncontrolled second moments.
5. **Regularized ratios with small floors.** Singular sensitivity and tail amplification.
6. **Randomized zero-mean controls.** Add conditional variance.
7. **Affine leave-one-group-out constructions.** Algebraically linear.
8. **Finite local curvature polynomials.** Cannot implement Haar projection on a non-bandlimited function class.
9. **Ordinary pointwise Hessian identities.** ReLU boundary curvature is distributional and invisible away from kinks.
10. **Generic symbolic regression over thousands of expressions.** Validation multiplicity dominates the small number of independent base networks.
11. **Expressions chosen by pairwise reconstruction loss.** The relevant loss is final contracted correction.
12. **Separate prediction of 34–38 source coefficients.** The observed obstruction is joint phase, not coefficient-regressor capacity.

---

## 17. Multiplicity-corrected search protocol

A search system is still useful, but only after theorem pruning.

### Stage A — Proof-carrying enumeration

For each expression:

- verify gauge and permutation legality;
- identify rotation representation support;
- calculate exact compute;
- canonicalize LOO, affine, homogeneous, and randomized forms;
- derive its leading-order linearization;
- prove denominator and moment bounds;
- label whether it accesses activation-boundary information.

Expressions failing any proof obligation are rejected without fitting.

### Stage B — Zero-cost oracle alignment

On exposed development data only, measure the cross-fitted direction cosine with the target correction.

Continuation thresholds should depend on cost. At roughly unchanged cost:

\[
\rho^2>0.80
\]

is the absolute mathematical minimum; use at least

\[
\rho^2>0.85
\]

for slack.

For doubled cost, require approximately

\[
\rho^2>0.92.
\]

Do not train amplitudes before this direction gate.

### Stage C — Frozen amplitude or conditional shrinkage

Fit only:

- one global scalar;
- one low-dimensional invariant stratification;
- or a predeclared analytic coefficient.

Report the exact MSE decomposition:

\[
\|E\|^2,
\qquad
\|C\|^2,
\qquad
\langle E,C\rangle.
\]

### Stage D — Nested multiplicity accounting

If `M` expressions were evaluated, the outer validation set must be untouched by all expression generation and ranking. Expression length, operation count, and fitted coefficient count should enter an MDL or PAC-Bayes-style penalty.

With only tens of independent base networks, a broad evolutionary or MCTS search is scientifically unreliable. The search must be dominated by algebraic rejection, not empirical tournament size.

### Stage E — Sealed confirmation

Require:

- grouped by base network;
- rotations kept together;
- replicated on an independently generated corpus;
- tails and worst cases;
- full adjusted score;
- no protected cohort until the complete estimator is frozen.

---

## 18. Proposed ledger patch

Use temporary IDs until collision checking against the workbook.

### `AG10-T1 — Correction-alignment theorem`

**Evidence:** Proved.

For any candidate direction `C`, optimal scalar correction has residual ratio `1-\rho(E,C)^2`. Same-cost 80% improvement requires `|\rho|\ge0.8944`; extra cost raises this requirement.

**Implication:** Directional alignment is the mandatory first gate for nonlinear expression search.

---

### `AG10-T2 — Stable nonlinear collapse`

**Evidence:** Proved under bounded `C^2` regularity and small transcript deviation.

A stable nonlinear expression equals its weight-dependent linearization at leading order. Order-one escape requires a new phase primitive, singular conditioning, or nonsmooth boundary information.

**Implication:** Smooth wrappers around closed linear transcripts cannot plausibly create a winning-scale gain.

---

### `AG10-T3 — Invariant-anchor shrinkage bound`

**Evidence:** Proved.

Shrinkage from the orbit-varying baseline toward an invariant anchor has minimum risk `VD/(V+D)`. An 80% reduction requires anchor error `D\le V/4`.

**Implication:** Magnitude-aware James–Stein descendants are closed unless the analytic anchor is already winning-scale.

---

### `AG10-T4 — Distributional Stein boundary identity`

**Evidence:** Proved subject to standard distributional regularization/transversality.

For positively homogeneous CPWL `g`,

\[
P|g|
=
P[\operatorname{sign}(g)\Delta g]
+
2P[\delta(g)\|\nabla g\|^2].
\]

The late absolute innovation is therefore an activation-boundary surface-measure problem.

**Implication:** Ordinary pathwise derivatives at Kerdock nodes omit the critical delta terms. A genuine identity must model or integrate boundary mass.

---

### `AG10-T5 — Rotation Poisson nonlocality`

**Evidence:** Proved.

\[
\Pi_0F-F
=
\int_0^\infty e^{t\Delta_G}\Delta_GF\,dt.
\]

No finite polynomial in local rotational derivatives implements this for a non-bandlimited class.

**Implication:** Curvature and finite derivative grammars need either nonlocal rotation evaluations or an analytic spectral model.

---

### `AG10-D1 — Agent 10 disposition`

**Evidence:** Mathematical synthesis plus existing empirical closures.

No deployable nonlinear estimator. Automated search should be theorem-pruned and restricted to:

1. analytic compressed distribution transport;
2. boundary-aware late-innovation identities;
3. a very small number of full-weight phase-bearing contractions.

Generic symbolic regression, invariant shrinkage, coordinatewise sign learning, LOO expressions, and singular ratios are stopped.

---

## Final assessment

The nonlinear search space is much smaller than it first appears.

The source geometry has already been solved: the final error is a low-effective-rank image of late absolute-value innovations. The coefficient problem has not been solved because the relevant sign is activation-boundary phase, not ordinary magnitude or low-rank geometry. Existing legal Walsh features can have excellent casewise oracle span while a transferable signed map predicts essentially nothing, which is the clearest empirical separation between representation and observability.

The most important conclusion is:

\[
\boxed{
\text{A winning nonlinear estimator must compute new distributional information.}
}
\]

It cannot merely rearrange existing invariant summaries.

The distributional Stein identity identifies that information as Gaussian activation-boundary mass. The characteristic-function identity identifies it as nonlocal frequency information. The known mixed-cumulant closure approximates the same object and already has a measured winning-scale oracle. Therefore the current canonical ranking is correct:

1. **Primary:** analytic propagation of a compressed, realizable mixed-moment or characteristic state.
2. **Narrow hedge:** direct boundary-aware evaluation of the last few transported absolute innovations.
3. **Stop:** broad automated nonlinear transcript search without a theorem showing where new phase information enters.

This does not prove that every nonlinear full-weight estimator is impossible. It does establish that a successful expression must visibly cross one of the three boundaries—new phase representation, controlled singularity, or activation-boundary integration—and that only the first and third are scientifically credible.
