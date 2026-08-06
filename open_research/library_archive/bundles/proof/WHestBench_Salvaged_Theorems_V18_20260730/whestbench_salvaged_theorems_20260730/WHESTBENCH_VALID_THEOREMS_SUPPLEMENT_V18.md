# WHestBench valid theorem supplement v18

This supplement is canonical over the superseded theorem wordings identified by the hostile audit. Each section is also available as a standalone file.


---

# WHestBench salvaged-theorem report

**Date:** 2026-07-30  
**Disposition:** every explicit counterexample from the hostile audit now has a maximal correct replacement; two application-specific conclusions are stronger than the original repaired wording.

## Executive result

The hostile audit did not merely delete claims. It exposed the exact mathematical boundary of each claim. The replacement package proves:

1. a complete affine characterization of symmetric-Gram minimizers;
2. strict positive definiteness of the actual depth-32 limiting ReLU kernel, restoring unique Kerdock weights in that application;
3. an exact constant/quadratic/high-even trichotomy for the finite-width MUB-line theorem;
4. exact and approximate conditional-Haar no-value results under the right relative-orientation condition;
5. exact bias-covariance-compute formulas for replication;
6. a sharp ReLU gate-crossing bound with constant `1/3`, rather than the previous loose `2`;
7. a fully uniform optimizer-transfer theorem for kernel perturbations;
8. an always-defined observability reporting convention;
9. a directed endpoint certificate completing T16 equality localization.

## Most important recovery: T29

The general free-mass uniqueness statement was false because a symmetric Gram matrix can have zero-sum null directions. The correct general result is a full minimizer set.

However, the actual limiting K32 kernel is much less degenerate than the general model. Its normalized ReLU map has strictly positive constant, linear, and every even power coefficient. After one additional composition, every odd coefficient becomes positive too. Every later composition preserves positivity in every degree. Since `K32` is the 32-fold iterate starting from `K0(t)=t`, it lies well inside this strict regime.

This yields a direct tensor-feature/Vandermonde proof that K32 is strictly positive definite on any finite set of distinct sphere points. Therefore the specific complete-Kerdock limiting-kernel claim recovers full uniqueness:

- uniform is the unique mass-one fixed-linear optimum;
- the rigorously enclosed `alpha_*`-scaled uniform vector is the unique free-mass optimum.

This recovery is stronger and cleaner than adding uniqueness as an unverified assumption. A directed-rounding association-spectrum certificate additionally proves a full-line zero-sum stability modulus above `0.00956473382419646475783854720307667122`.

## Most useful interpretation of the T38 counterexample

The pure degree-two example is not a pathological dead end. It is the exact boundary between degeneracy and strict complete-basis concentration.

For every nonnegative even noise-stability expansion:

- nonconstant even mass makes `A-O>0` and `O-C<0`;
- degree-four-or-higher mass is exactly equivalent to the third strict sign;
- pure quadratic mass makes the between-basis eigenvalue exactly zero.

In the pure-quadratic case, any mixture of complete orthonormal bases with equal within-basis weights is optimal. With fewer than `d` lines, the optimum concentrates all lines in one basis. With at least `d`, one complete basis already removes the quadratic discrepancy. This is a positive theorem and a useful diagnostic for why low-degree controls saturate.

For a finite piecewise-affine ReLU network, a nonconstant even realization cannot terminate at degree two, so the original practical finite-ReLU conclusion survives after separating it from the false general square-integrable formulation.

## Information symmetry replacement

The false Haar statement confused invariance of reported features with randomness of the *relative orientation*. The exact valid condition is conditional Haar randomness after fixing the integrand and selected rule.

This can be enforced operationally: choose any legal integrand-dependent design shape and weights, then draw an independent Haar orientation. Every correction that does not observe orientation-sensitive post-rotation information has zero value.

The result also admits an approximate form. Conditional chi-square divergence from Haar multiplies the orientation-averaged risk to upper-bound all orientation-blind correction value. This converts symmetry into a quantitative empirical target instead of an unverifiable slogan.

## Replication replacement

Independent replication reduces centered variance but preserves common bias. The exact cost-adjusted formula identifies all useful regimes:

- unbiased independent replicas plus linear cost: neutral;
- biased independent replicas plus linear cost: strictly worse;
- sublinear shared compute: can win, with an exact threshold;
- negative centered covariance: can win, with an exact threshold;
- bias reduction: changes the conclusion and should be measured separately.

This is more actionable than either the false neutrality claim or a blanket rejection of replication.

## ReLU replacement

The ReLU remainder is a triangular gate-crossing profile, not merely an indicator-bounded error. Integrating the triangle gives

`E r^2 <= L |t|^3 / 3`

when the density is bounded by `L` on the actual crossing interval. The constant is asymptotically sharp. Conditional and vector versions directly support downstream replay certificates.

For a Gaussian preactivation with conditional standard deviation at least `sigma`, the bound becomes

`E r^2 <= E|T|^3 / (3 sigma sqrt(2 pi))`.

This is six times tighter than the already-corrected `2L|t|^3` statement.

## Verification

`verify_salvaged_theorems.py` passed. It checks:

- positivity and numerical accuracy of the ReLU-kernel power series;
- positive coefficients after composition;
- a K32 Gram matrix including antipodal points;
- the exact MUB block spectrum, including a directed-rounding K32 stability certificate;
- the pure-quadratic null multiplicity and finite-budget partition claim;
- the biased-replication formula;
- the sharp normal-density ReLU bound and its asymptotic constant;
- a finite-group analogue of the chi-square near-Haar inequality;
- strict T16 endpoint separation using certified coefficient intervals.

These computations are sanity checks; the theorem files contain the analytic proofs.

## Recommended canonical claim set

The recommended external manuscript should use the replacement matrix in `VALID_CLAIMS_MATRIX.md`, add theorem IDs T41–T47, and retain the hostile counterexamples as boundary examples rather than deleting them. The valid story is stronger when it says exactly when a theorem is strict, degenerate, approximate, or unique.

---

# T29 salvage — exact minimizer set, stability, and restored K32 uniqueness

**Status:** analytically proved. The general theorem corrects the false uniqueness wording. The K32 corollary recovers uniqueness for the actual limiting depth-32 kernel.

## 1. General symmetric Gram theorem

Let `Y_1,...,Y_N` and the target `T` be square-integrable random variables in a real Hilbert space. Define

\[
G_{ij}=\mathbb E\langle Y_i,Y_j\rangle,
\qquad
m_i=\mathbb E\langle Y_i,T\rangle,
\]

and

\[
R(w)=\mathbb E\left\|\sum_i w_iY_i-T\right\|^2
=c-2m^Tw+w^TGw.
\]

Assume

\[
G\mathbf 1=\lambda\mathbf 1,
\qquad
m=\tau\mathbf 1.
\]

Let `u=1/N * 1`. Every weight vector has the unique decomposition

\[
w=\alpha u+v,
\qquad \mathbf 1^Tv=0,
\qquad \alpha=\mathbf 1^Tw.
\]

Then

\[
R(\alpha u+v)=R(\alpha u)+v^TGv.
\]

### Fixed total mass

For any prescribed mass `alpha`, the complete minimizer set is

\[
\alpha u+(\ker G\cap\mathbf 1^\perp).
\]

In particular, among mass-one rules, uniform weights minimize risk, and they are unique if and only if

\[
\ker G\cap\mathbf 1^\perp=\{0\}.
\]

Equivalently, uniqueness holds exactly when `G` is positive definite on the zero-sum subspace.

### Free total mass

Put

\[
E_X=u^TGu=\mathbb E\left\|\sum_i u_iY_i\right\|^2.
\]

If `E_X>0`, define

\[
\alpha_*=\frac{\tau}{E_X}.
\]

The complete free-mass minimizer set is

\[
\alpha_*u+(\ker G\cap\mathbf 1^\perp).
\]

If `E_X=0`, then `\tau=0` and the free minimizer set is simply `ker G`.

### Quantitative stability

Let

\[
\lambda_\perp=
\inf_{v\perp\mathbf 1,\ \|v\|=1}v^TGv.
\]

For fixed mass `alpha`,

\[
R(\alpha u+v)-R(\alpha u)=v^TGv
\ge \lambda_\perp\|v\|^2.
\]

For free mass with `E_X>0`,

\[
R(\alpha u+v)-R(\alpha_*u)
=E_X(\alpha-\alpha_*)^2+v^TGv.
\]

Thus `lambda_perp` is the exact transverse robustness modulus. If it vanishes, the correct object is the affine minimizer set rather than a unique vector.

### Canonical symmetry selector

At fixed mass, adding any ridge term `eta ||w||^2` with `eta>0` makes the uniform vector the unique minimizer among all previously risk-equivalent weights, because

\[
\|\alpha u+v\|^2=\alpha^2\|u\|^2+\|v\|^2.
\]

This gives a principled implementation rule even when the unregularized Gram matrix is singular.

## 2. Restored uniqueness for the actual K32 limiting kernel

The normalized ReLU covariance map is

\[
\kappa(t)=
\frac{\sqrt{1-t^2}+(\pi-\arccos t)t}{\pi}.
\]

Its absolutely convergent power series on `[-1,1]` is

\[
\kappa(t)=\frac1\pi+\frac12t+
\sum_{m\ge1}
\frac{\binom{2m-2}{m-1}}
{2m(2m-1)4^{m-1}\pi}
\,t^{2m}.
\]

Every displayed coefficient is strictly positive.

Let `K_L=kappa composed L times`. Once `L>=2`, every Maclaurin coefficient of `K_L` is strictly positive:

- constant, linear, and every even coefficient are inherited through the positive linear coefficient of the outer `kappa`;
- for every odd `n>=3`, the positive quadratic coefficient of the outer `kappa` multiplies a positive `t^n` coefficient in the square of the inner series, containing the term `2 a_1 a_{n-1}`.

Therefore the depth-32 kernel `K32=kappa composed 32 times` has a representation

\[
K_{32}(t)=\sum_{n\ge0}k_nt^n,
\qquad k_n>0\ \text{for every }n.
\]

### Strict positive definiteness

For distinct sphere points `x_1,...,x_N` and any nonzero real vector `c`,

\[
\sum_{i,j}c_ic_jK_{32}(\langle x_i,x_j\rangle)
=
\sum_{n\ge0}k_n
\left\|\sum_i c_i x_i^{\otimes n}\right\|^2
>0.
\]

Indeed, equality would force every tensor moment to vanish. Choose a vector `z` for which the scalar projections `s_i=<z,x_i>` are pairwise distinct. Contracting the first `N` tensor identities against `z^{tensor n}` gives

\[
\sum_i c_i s_i^n=0,
\qquad n=0,\ldots,N-1.
\]

The Vandermonde matrix is invertible, so `c=0`, a contradiction.

Hence the K32 Gram matrix is strictly positive definite on every finite set of distinct sphere points, including the complete 66,048-point Kerdock support.

### Arbitrary fixed-support corollary

For any finite distinct support under K32, the signed-weight risk is strictly convex. Therefore every convex weight-feasible set has at most one optimum. Without inequality constraints, the unique mass-one optimum is

\[
w_*=G^{-1}(m-\nu\mathbf1),
\qquad
\nu=\frac{\mathbf1^TG^{-1}m-1}{\mathbf1^TG^{-1}\mathbf1}.
\]

On a transitive support with `m=tau 1` and constant Gram row sum, this formula reduces to uniform weights. Thus K32 strict positive definiteness is useful beyond the complete Kerdock set: it removes all finite-support weight nullspaces and makes fixed-support optimization well posed.

### Quantitative full-line stability

On the symmetrized 33,024-line Kerdock/MUB universe, the association-scheme spectrum is explicitly computable. The bundled directed-rounding certificate proves that the smallest zero-sum eigenvalue of the K32 line Gram matrix is at least

\[
0.00956473382419646475783854720307667122.
\]

Thus, for full-support mass-one line weights, uniformity is not only unique but quantitatively stable:

\[
R(u+v)-R(u)
\ge
0.00956473382419646475783854720307667122\,\|v\|_2^2
\qquad(\mathbf1^Tv=0).
\]


## 3. Application-specific conclusion

For the depth-32 limiting kernel on the archived complete Kerdock point set:

- the mass-one uniform rule is the **unique** fixed-linear minimizer;
- the free-mass minimizer is the **unique** scaled-uniform rule;
- the general constant-field counterexample does not apply, because its rank-one kernel is not K32.

Using the directed archived enclosures,

\[
\alpha_*\in
[0.9999997503247282806575775152106693,
 0.9999997503247282806578123186727384].
\]

Thus the old general uniqueness theorem was false, but the specific limiting-K32 uniqueness conclusion can be restored by a stronger kernel argument.

---

# T38 salvage — minimal condition, exact trichotomy, and the quadratic boundary

**Status:** analytically proved.

Let the antipodally symmetrized finite-width kernel have the nonnegative even expansion

\[
\overline K(t)=\sum_{r\ge0}a_{2r}t^{2r},
\qquad a_{2r}\ge0.
\]

On a real MUB line universe define

\[
A=\overline K(1),
\qquad O=\overline K(0),
\qquad C=\overline K(1/\sqrt d).
\]

Then the three association quantities are exactly

\[
A-O=\sum_{r\ge1}a_{2r},
\]

\[
O-C=-\sum_{r\ge1}a_{2r}d^{-r},
\]

and

\[
\Delta:=(A-O)+d(O-C)
=\sum_{r\ge2}a_{2r}(1-d^{1-r}).
\]

Consequently:

1. `A-O>0` and `O-C<0` if and only if the even output is nonconstant.
2. `Delta>0` if and only if there is positive even Hermite/noise-stability mass at some degree at least four.
3. `Delta=0` if and only if the even kernel contains only constant and quadratic terms.
4. Under a nonnegative noise-stability expansion, `Delta<0` is impossible.

Let

\[
H_{\ge4}=\sum_{r\ge2}a_{2r}.
\]

Because `1-d^{1-r}` lies between `1-1/d` and `1`,

\[
\left(1-\frac1d\right)H_{\ge4}
\le\Delta\le H_{\ge4}.
\]

At `d=256`, the between-basis spectral gap captures at least `255/256` of the total degree-four-and-higher even mass. Thus the strictness margin is a quantitatively faithful measure of the high-even component, not an opaque sign condition.

This is the exact minimal assumption behind the strict T27/T38 conclusion.

## Exact Gram spectrum on the full MUB line universe

For `M` mutually unbiased bases, `d` lines per basis, and `N=Md`, the line Gram matrix is

\[
G=(A-O)I+(O-C)(I_M\otimes J_d)+CJ_N.
\]

Its eigenvalues are

\[
\lambda_{\rm within}=A-O
\quad\text{with multiplicity }M(d-1),
\]

\[
\lambda_{\rm between}=\Delta
\quad\text{with multiplicity }M-1,
\]

and

\[
\lambda_{\rm global}
=A+(d-1)O+(N-d)C
\quad\text{on }\mathbf1.
\]

Therefore the corrected high-even condition gives positive definiteness on the entire zero-sum subspace. On the complete line universe, uniform mass-one weights are unique. If the global eigenvalue is positive—as it is for a nonzero nonnegative noise-stability kernel—the free-mass scaled-uniform solution is also unique.

## Certified K32 line-spectrum corollary

The bundled directed-rounding certificate evaluates the depth-32 kernel at `-1`, `0`, `+/-1/16`, and `1` and proves, for `d=256` and all `129*256=33,024` Kerdock lines,

\[
A-O\in[0.01198858116065568726827329516210501907,
       0.01198858116065568726827329516210501908],
\]

\[
\Delta\in[0.00956473382419646475783854720307667122,
               0.00956473382419646475783854720307667123].
\]

Hence the full K32 line Gram matrix is positive definite. For every mass-one line-weight perturbation `v` with zero total mass,

\[
R(u+v)-R(u)
\ge
0.00956473382419646475783854720307667122\,\|v\|_2^2.
\]

This is a rigorous quantitative uniqueness and stability statement, not merely a strict-sign argument. The certificate is `K32_MUB_LINE_SPECTRUM_CERTIFICATE.json` and can be regenerated with `certify_k32_mub_line_spectrum.py`.

## Trichotomy

### Case 1: constant even kernel

If `a_{2r}=0` for all `r>=1`, every mass-one line rule has the same risk. No support or weighting conclusion is possible or needed.

### Case 2: constant plus quadratic only

Suppose `a_2>0` and `a_{2r}=0` for every `r>=2`. Then

\[
A-O=a_2,
\qquad O-C=-a_2/d,
\qquad \Delta=0.
\]

For line weights `w_{bi}` with basis masses `S_b=sum_i w_{bi}`, the nonconstant part of the risk is exactly

\[
a_2\sum_b\left(
\sum_iw_{bi}^2-\frac{S_b^2}{d}
\right)
=
\frac{a_2}{d}\sum_b
\left(d\sum_iw_{bi}^2-S_b^2\right)
\ge0.
\]

Equality holds exactly when every basis carrying nonzero mass is complete and has equal within-basis weights:

\[
w_{bi}=S_b/d,
\qquad \sum_bS_b=1.
\]

The basis masses may be arbitrary, including signed values. This fully characterizes the nonuniqueness exposed by the pure-quadratic counterexample.

For a budget of **at most** `P` nonzero lines:

- if `P<d`, the optimum puts all `P` lines in one basis with weights `1/P`, and the nonconstant risk is `a_2(1/P-1/d)`;
- if `P>=d`, one complete basis already attains zero nonconstant risk, and additional lines cannot improve it.

Thus the degree-two boundary has a useful interpretation: a complete orthonormal basis exactly captures the quadratic component, while mixtures across complete bases are redundant.

### Case 3: positive degree-four-or-higher mass

If

\[
\sum_{r\ge2}a_{2r}>0,
\]

then all three strict association signs hold. The T27 convex allocation theorem applies:

- complete bases plus at most one partial basis;
- equal positive weights within every active basis;
- positive analytic basis masses;
- no negative-weight improvement inside the fixed MUB line universe;
- all budgeted lines are used.

For a finite piecewise-affine ReLU realization, nonconstant antipodal-even output implies this case: an even piecewise-affine function with Hermite expansion terminating at degree two would coincide with a quadratic polynomial, whose Hessian must vanish on every affine cell, making it affine and then constant by evenness.

---

# Conditional Haar salvage — exact randomization and quantitative near-Haar bounds

**Status:** analytically proved.

Let a compact group `G` act transitively on the integration domain, let `h` be normalized Haar measure, and let

\[
Q=\sum_iw_i\delta_{x_i},
\qquad \sum_iw_i=1.
\]

For `U in G`, write

\[
Q_Uf=\sum_iw_if(Ux_i),
\qquad e(U,f,Q)=Q_Uf-I(f).
\]

The deterministic group-average identity is

\[
\int_G e(U,f,Q)\,dh(U)=0
\]

for every fixed integrable Hilbert-valued `f` and every fixed mass-one signed rule `Q`.

## Exact conditional theorem

Let `H` be a sigma-field containing the realized integrand `f`, the unrotated rule `Q`, and any runtime information used by a correction. If

\[
\operatorname{Law}(U\mid H)=h
\quad\text{almost surely},
\]

then

\[
\mathbb E[e\mid H]=0.
\]

Therefore, for every smaller runtime sigma-field `G_runtime subset H`,

\[
\mathbb E[e\mid G_{\rm runtime}]=0,
\]

and no additive correction measurable from that runtime information can reduce mean-squared error.

This formulation allows the shape, support, and weights of `Q` to depend on the realized integrand or legal features, provided an independent Haar orientation is drawn **after** those choices and before evaluations. It fails if the integrand co-rotates with `U` or if post-orientation observations enter the correction.

## Approximate Haar theorem using chi-square divergence

Let `mu_H=Law(U|H)` and assume `mu_H` is absolutely continuous with respect to Haar measure, with conditional chi-square divergence

\[
\chi_H^2=
\int_G\left(\frac{d\mu_H}{dh}-1\right)^2dh.
\]

Define the conditional Haar-orientation risk

\[
R_{\rm orient}(f,Q)
=
\int_G\|e(U,f,Q)\|^2dh(U).
\]

Then

\[
\|\mathbb E[e\mid H]\|^2
\le
\chi_H^2 R_{\rm orient}(f,Q).
\]

Consequently,

\[
\mathbb E\|\mathbb E[e\mid G_{\rm runtime}]\|^2
\le
\mathbb E[\chi_H^2R_{\rm orient}(f,Q)].
\]

This turns an approximate relative-orientation test into a quantitative upper bound on the value of every orientation-blind correction.

## Approximate Haar theorem using total variation

If `f` is essentially bounded and `B=sum_i|w_i|`, then

\[
\|e(U,f,Q)\|\le(1+B)\|f\|_\infty.
\]

With total variation defined by `TV(mu,h)=sup_A|mu(A)-h(A)|`,

\[
\|\mathbb E[e\mid H]\|
\le
2(1+B)\|f\|_\infty\,TV(\mu_H,h).
\]

Thus exact Haar randomness is not the only useful regime: a certified small conditional orientation defect yields a certified small correction capacity.

## Operationally valid claim

A valid paper statement is:

> Independently Haar-randomizing the rule orientation after fixing the integrand-dependent design makes every orientation-blind additive correction exactly useless. If the conditional orientation law is only approximately Haar, the maximum correction value is bounded by a divergence from Haar times the orientation-averaged baseline risk.

---

# Replication salvage — exact bias, covariance, and compute economics

**Status:** analytically proved.

Let `e_1,...,e_m` be Hilbert-valued estimator errors with common mean

\[
b=\mathbb E e_i,
\]

common centered variance

\[
V=\mathbb E\|e_i-b\|^2,
\]

and common pairwise centered covariance

\[
\mathbb E\langle e_i-b,e_j-b\rangle=\rho V,
\qquad i\ne j.
\]

Necessarily

\[
-\frac1{m-1}\le\rho\le1.
\]

The averaged error satisfies the exact identity

\[
\mathbb E\left\|\frac1m\sum_{i=1}^me_i\right\|^2
=
\|b\|^2+
\frac{V}{m}\bigl(1+(m-1)\rho\bigr).
\]

Let

\[
R_0=\|b\|^2+V,
\qquad
\beta=\frac{\|b\|^2}{R_0}.
\]

If total compute is multiplied by `c_m`, the exact MSE-times-compute ratio is

\[
\operatorname{Ratio}_m
=
c_m\left[
\beta+(1-\beta)
\frac{1+(m-1)\rho}{m}
\right].
\]

## Independent replicas

Independence gives `rho=0` for the centered errors, even when the estimators are biased. Hence

\[
R_m=\|b\|^2+V/m.
\]

Under linear cost `c_m=m`,

\[
\operatorname{Ratio}_m
=1+(m-1)\beta.
\]

Therefore:

- unbiased independent replication (`beta=0`) is exactly score-neutral;
- any nonzero common bias makes linear-cost replication strictly worse;
- the deterministic-bias counterexample is the extreme case `beta=1`, giving ratio `m`.

With shared computation and independent errors, replication wins exactly when

\[
c_m<\frac{m}{1+(m-1)\beta}.
\]

For unbiased replicas this reduces to `c_m<m`: any genuine sublinear shared-cost implementation improves the adjusted score.

## Negative covariance

With linear cost, replication wins exactly when

\[
\rho< -\frac{\beta}{1-\beta}
\]

for `beta<1`. Thus antithetic construction can overcome linear cost only if its centered negative covariance is strong enough to offset common bias. Since exchangeability requires `rho>=-1/(m-1)`, such a win is feasible only when

\[
\beta<\frac1m.
\]

If the common-bias share is at least `1/m`, even maximally antithetic exchangeable centered errors cannot beat linear cost.

## Unequal or nonexchangeable replicas

For general mean vector and covariance operator, the optimal deterministic linear combination under a sum-one constraint is the generalized least-squares solution. The exchangeable formula above is the closed-form special case and should replace the false statement that independence alone makes raw errors uncorrelated.

## Operationally valid claim

> Replication removes variance, not common bias. Under linear MSE-times-compute accounting, independent replication is neutral only for unbiased estimators and strictly harmful otherwise. It becomes useful through shared sublinear compute, sufficiently negative centered covariance, or explicit bias reduction.

---

# ReLU remainder salvage — exact crossing formula and sharp cubic bound

**Status:** analytically proved. This strengthens the hostile-patched `2L|t|^3` bound by a factor of six.

For `phi(z)=max(z,0)`, define the first-order gate remainder

\[
r(z,t)=\phi(z+t)-\phi(z)-\mathbf1_{\{z>0\}}t.
\]

## Exact pointwise formula

The remainder is nonnegative and supported exactly on a gate crossing:

For `t>=0`,

\[
r(z,t)=(z+t)\mathbf1_{\{-t\le z\le0\}}.
\]

For `t<0`,

\[
r(z,t)=-(z+t)\mathbf1_{\{0<z\le -t\}}.
\]

In particular,

\[
0\le r(z,t)\le |t|\mathbf1_{\{|z|\le|t|\}}.
\]

Without any density assumption, this already gives the distribution-free conditional bound

\[
\mathbb E[r(Z,T)^2\mid T]
\le |T|^2\Pr(\text{the segment from }Z\text{ to }Z+T\text{ crosses }0\mid T).
\]

This form remains valid with atoms and is the correct fallback when a density certificate is unavailable.

## Exact second moment

If `Z` has density `p`, then for `t>0`,

\[
\mathbb E r(Z,t)^2
=
\int_{-t}^0(z+t)^2p(z)\,dz,
\]

and for `t<0`,

\[
\mathbb E r(Z,t)^2
=
\int_0^{|t|}(|t|-z)^2p(z)\,dz.
\]

If the density is bounded by `L_t` on the actual crossing interval, then

\[
\mathbb E r(Z,t)^2
\le
\frac{L_t}{3}|t|^3.
\]

The constant `1/3` follows by integrating the squared triangular crossing profile and is sharp to first order.

If `p` is continuous at zero, then

\[
\lim_{t\downarrow0}\frac{\mathbb E r(Z,t)^2}{t^3}
=\frac{p(0^-)}3,
\qquad
\lim_{t\uparrow0}\frac{\mathbb E r(Z,t)^2}{|t|^3}
=\frac{p(0^+)}3.
\]

For a continuous density the common limit is `p(0)/3`.

## Random and dependent perturbations

Let `T` be random. If the conditional density of `Z` given `T` is bounded by `L(T)` on the corresponding crossing interval, then

\[
\mathbb E r(Z,T)^2
\le
\frac13\mathbb E[L(T)|T|^3].
\]

No independence between `Z` and `T` is needed; the condition is explicitly conditional.

For Gaussian `Z` with conditional standard deviation at least `sigma`, the global density bound gives

\[
\mathbb E r(Z,T)^2
\le
\frac{1}{3\sigma\sqrt{2\pi}}\mathbb E|T|^3.
\]

## Vector form

For coordinatewise ReLU with perturbation vector `T`, if each conditional coordinate density has crossing-interval bound at most `L`, then

\[
\mathbb E\|r(Z,T)\|_2^2
\le
\frac L3\mathbb E\sum_j|T_j|^3
\le
\frac L3\mathbb E\|T\|_2^3.
\]

After a downstream linear map `J`,

\[
\mathbb E\|Jr(Z,T)\|^2
\le
\frac{\|J\|_{\rm op}^2L}{3}
\mathbb E\|T\|_2^3.
\]

## Exact nonlinear improvement gate

If the baseline error has risk `R0`, the linearly corrected error has risk `Rlin`, and the downstream gate remainder has second moment at most `delta^2`, then

\[
R_{\rm exact}
\le(\sqrt{R_{\rm lin}}+\delta)^2.
\]

A sufficient exact-ReLU improvement condition is therefore

\[
\delta<\sqrt{R_0}-\sqrt{R_{\rm lin}}.
\]

The sharp cubic bound above supplies a legal, direction-sensitive way to certify `delta`.

---

# Kernel perturbation salvage — uniform optimizer-transfer theorem

**Status:** analytically proved.

Let `C_B` be a comparison class of signed rules with total variation bounded uniformly by

\[
\sum_i|w_i|\le B.
\]

If two kernels satisfy

\[
\|K-\widetilde K\|_\infty\le\varepsilon,
\]

then every rule in `C_B` obeys

\[
|R_K(Q)-R_{\widetilde K}(Q)|
\le\delta,
\qquad
\delta=\varepsilon(1+B)^2.
\]

If `Qtilde` is `eta`-suboptimal for the surrogate kernel,

\[
R_{\widetilde K}(\widetilde Q)
\le
\inf_{Q\in C_B}R_{\widetilde K}(Q)+\eta,
\]

then

\[
R_K(\widetilde Q)
\le
\inf_{Q\in C_B}R_K(Q)+\eta+2\delta.
\]

The same conclusion holds for a minimizing sequence because the variation bound is uniform over the entire class.

## Ranking preservation

For two candidates `Q1,Q2`, if

\[
R_{\widetilde K}(Q_1)+2\delta
<
R_{\widetilde K}(Q_2),
\]

then

\[
R_K(Q_1)<R_K(Q_2).
\]

Thus a surrogate-kernel winner is certified only when its margin exceeds twice the uniform perturbation radius. This is the correct alternative to an optimizer-transfer statement that silently bounds only the selected candidate and not the comparison class.

---

# Observability metric salvage — total capacity, transferred value, and fraction

**Status:** analytically proved bookkeeping convention.

Let `H_runtime` be a closed correction-information subspace and let `H_oracle` be a larger oracle subspace. For baseline error `e`, define

\[
V_{\rm runtime}=\|P_{H_{\rm runtime}}e\|_{L^2}^2,
\qquad
V_{\rm oracle}=\|P_{H_{\rm oracle}}e\|_{L^2}^2.
\]

Nestedness gives

\[
0\le V_{\rm runtime}\le V_{\rm oracle}.
\]

The always-defined quantities are:

- **oracle capacity:** `V_oracle`;
- **transferred value:** `V_runtime`;
- **unobserved value:**
  \[
  V_{\rm oracle}-V_{\rm runtime}\ge0.
  \]

Only when `V_oracle>0` define the transferred fraction

\[
F_{\rm transfer}
=\frac{V_{\rm runtime}}{V_{\rm oracle}}
\in[0,1].
\]

If `V_oracle=0`, nestedness forces `V_runtime=0`; report **zero oracle capacity** rather than assigning an arbitrary ratio. This separates two scientifically different findings:

1. `V_oracle=0`: the proposed correction class has no useful capacity;
2. `V_oracle>0` but a small transfer fraction: useful capacity exists but legal information does not recover it.

This three-number reporting convention avoids the undefined `0/0` edge case and preserves the capacity-versus-observability distinction.

---

# T16 endpoint-equality patch

**Status:** directed-decimal certified.

The interior Hermite remainder proves

\[
K_{32}(t)-h_*(t)>0
\]

for every noncontact point in `(-1,1)`, because `K32^(6)>0` and the squared contact polynomial is positive away from the three roots. Continuity alone gives only endpoint nonnegativity, so the old equality-only-at-contacts wording needed explicit endpoint separation.

Using the certified Gegenbauer coefficient intervals from `T16_PRIMAL_DUAL_CERTIFICATE.json` and the independent directed evaluation of `K32(-1)`, the bundled endpoint script proves

\[
K_{32}(1)-h_*(1)
\in
[0.0170218942683709807001391155978126223540679072640,
 0.0170218942683709807001391155978126223540679072641],
\]

and

\[
K_{32}(-1)-h_*(-1)
\in
[2.2051871290807434455869043041150906917944744889\times10^{-7},
 2.2051871290807434455869043041150906917944744890\times10^{-7}].
\]

Both are strictly positive. Therefore equality in the global minorant occurs exactly at the three interior Hermite contact nodes.

Regenerate with:

```bash
python certify_k32_mub_line_spectrum.py
python certify_t16_endpoints.py
```

The resulting machine-readable certificate is `T16_ENDPOINT_CERTIFICATE.json`.
