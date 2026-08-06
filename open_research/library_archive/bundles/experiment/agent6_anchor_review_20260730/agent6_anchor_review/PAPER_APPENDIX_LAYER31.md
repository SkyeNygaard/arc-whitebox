# Appendix: Risk, Identifiability, and Nonlinear Remainders for Layer-31 Anchor Replacement

## A.1 Setup

Let \(H\) be the finite-dimensional Hilbert space of scored final-output coordinates, with its scoring inner product. Randomness may include the network, allowed rotations, and any independent reference or estimator randomness. Let

\[
e=\widehat y_0-y\in L^2(\Omega;H)
\]

be the protected estimator's error. A proposed signed correction \(u\in L^2(\Omega;H)\) is applied with a scalar \(\alpha\):

\[
\widehat y_\alpha=\widehat y_0-\alpha u,
\qquad
R(\alpha)=\mathbb E\|e-\alpha u\|^2.
\]

The sign convention is that \(u=e\) is perfect.

## A.2 Exact correction risk and selector value

### Theorem A.1 (correction-risk identity)

Define

\[
R_0=\mathbb E\|e\|^2,\qquad
C=\mathbb E\langle e,u\rangle,\qquad
U=\mathbb E\|u\|^2.
\]

Then

\[
R(\alpha)=R_0-2\alpha C+\alpha^2U.
\]

For \(U>0\), the unrestricted optimum is

\[
\alpha_*=\frac CU,
\qquad
R(\alpha_*)=R_0-\frac{C^2}{U}.
\]

If only nonnegative scales are legal, then

\[
\alpha_+^*=\frac{(C)_+}{U},
\qquad
R(\alpha_+^*)=R_0-\frac{(C)_+^2}{U}.
\]

Thus magnitude, disagreement, instability, or oracle headroom alone cannot make a correction useful. The required object is its signed error-correction inner product.

### Theorem A.2 (conditional selector value)

Let \(\mathcal G\) denote all runtime information available to a target-free selector and define

\[
C_{\mathcal G}=\mathbb E[\langle e,u\rangle\mid\mathcal G],
\qquad
U_{\mathcal G}=\mathbb E[\|u\|^2\mid\mathcal G].
\]

Use the convention that the selected scale is zero where \(U_{\mathcal G}=0\). The unrestricted conditional optimum is

\[
\alpha^*(\mathcal G)=\frac{C_{\mathcal G}}{U_{\mathcal G}},
\]

with total gain

\[
\mathbb E\left[\frac{C_{\mathcal G}^2}{U_{\mathcal G}}\right].
\]

For a nonnegative selector, replace \(C_{\mathcal G}\) by \((C_{\mathcal G})_+\). For \(0\le\alpha\le A\),

\[
\alpha_A^*(\mathcal G)
=
\operatorname{clip}\!\left(\frac{C_{\mathcal G}}{U_{\mathcal G}},0,A\right).
\]

The corresponding conditional gain is

\[
\begin{cases}
0, & C_{\mathcal G}\le0,\\
C_{\mathcal G}^2/U_{\mathcal G}, & 0<C_{\mathcal G}<A U_{\mathcal G},\\
2A C_{\mathcal G}-A^2U_{\mathcal G}, & C_{\mathcal G}\ge A U_{\mathcal G}.
\end{cases}
\]

## A.3 Anchor replacement

Let \(\mathcal S\subset L^2(\Omega;H)\) be a closed linear correction subspace. Write

\[
s=P_{\mathcal S}e,
\qquad
r=e-s,
\qquad
r\perp\mathcal S.
\]

Suppose an estimated correction is \(\widehat s=s+n\).

### Theorem A.3 (full replacement, subspace form)

If \(n\in\mathcal S\), then

\[
\mathbb E\|e-\widehat s\|^2
=
\mathbb E\|r\|^2+
\mathbb E\|n\|^2.
\]

Consequently, full replacement improves exactly when

\[
\mathbb E\|n\|^2<\mathbb E\|s\|^2.
\]

### Proposition A.4 (general replacement error)

If \(n\) is not required to lie in \(\mathcal S\), then

\[
\mathbb E\|e-\widehat s\|^2
=
\mathbb E\|r\|^2+
\mathbb E\|n\|^2
-2\mathbb E\langle r,n\rangle.
\]

Full replacement improves exactly when

\[
\mathbb E\|n\|^2-2\mathbb E\langle r,n\rangle
<
\mathbb E\|s\|^2.
\]

The simpler norm comparison therefore requires the subspace or orthogonality condition. It is not valid for an arbitrary anchor error with an out-of-subspace component.

### Proposition A.5 (optimal shrinkage with correlated in-subspace noise)

Assume \(n\in\mathcal S\). Define

\[
S=\mathbb E\|s\|^2,
\quad
N=\mathbb E\|n\|^2,
\quad
K=\mathbb E\langle s,n\rangle.
\]

For the correction \(\alpha(s+n)\),

\[
R(\alpha)
=
\mathbb E\|r\|^2
+(1-\alpha)^2S+
\alpha^2N-
2\alpha(1-\alpha)K.
\]

When \(S+N+2K>0\), the unrestricted optimum is

\[
\alpha_*
=
\frac{S+K}{S+N+2K}.
\]

The familiar formula \(S/(S+N)\) needs only \(K=0\), not full probabilistic independence. Correlation or systematic bias can move the optimum substantially and can reverse the sign of the useful correction.

## A.4 Layer-31 mapping

Condition on a realized network and protected particle cloud. Let

\[
d=\mu_{31}^{K}-\mu_{31}^{*}
\]

be the protected design's layer-31 mean defect, and let

\[
\xi=\widehat\mu_{31}-\mu_{31}^{*}
\]

be an external anchor's error. For a frozen linearized final replay \(J\), the relevant quantities are

\[
s=Jd,
\qquad
n=J\xi.
\]

Under the subspace assumptions of Theorem A.3, full replacement improves exactly when

\[
\eta_J^2
:=
\frac{\mathbb E\|J\xi\|^2}
{\mathbb E\|Jd\|^2}
<1.
\]

This downstream-weighted ratio is the invariant threshold. A scalar claim such as “\(5\times10^{-4}\) relative mean error is break-even” is meaningful only after specifying:

1. the denominator used for relative error;
2. the perturbation distribution or structured direction;
3. whether \(J\) is frozen, cross-fitted, or target-selected;
4. the scored norm and network aggregation;
5. whether the true nonlinear final replay or only its linearization is scored.

An unweighted Euclidean threshold is generally direction-dependent because \(J\) is anisotropic.

## A.5 ReLU gate-crossing remainder

For \(\phi(z)=\max(z,0)\), define

\[
r(z,t)=\phi(z+t)-\phi(z)-\mathbf1_{\{z>0\}}t.
\]

### Lemma A.6 (scalar crossing bound)

For every \(z,t\in\mathbb R\),

\[
|r(z,t)|
\le
|t|\mathbf1_{\{|z|\le|t|\}}.
\]

Thus the linearization is exact unless the proposed shift can cross a ReLU kink.

For particle preactivations \(h_i\) and common shift \(t=W\delta\),

\[
m(\delta)
=
\frac1n\sum_i\phi(h_i+t)
=
m(0)+J\delta+R(\delta),
\]

where

\[
J
=
\frac1n\sum_i
\operatorname{diag}(\mathbf1_{\{h_i>0\}})W
\]

and, coordinatewise,

\[
|R(\delta)|
\le
\frac1n\sum_i
|W\delta|\odot
\mathbf1_{\{|h_i|\le|W\delta|\}}.
\]

If a coordinate's conditional preactivation density is bounded by \(L\) near zero, then the scalar squared remainder obeys the useful cubic-scale estimate

\[
\mathbb E[r(z,t)^2\mid t]
\le
2L|t|^3,
\]

provided the stated density bound holds conditionally on \(t\).

### Proposition A.7 (nonlinear robustness margin)

Let the exact corrected error be

\[
r-n+q,
\]

where \(q\) is the nonlinear replay remainder relative to the linearized model. Define

\[
L_0=\mathbb E\|r-n\|^2,
\qquad
Q=\mathbb E\|q\|^2.
\]

Then

\[
\mathbb E\|r-n+q\|^2
\le
(\sqrt{L_0}+\sqrt Q)^2.
\]

Therefore a sufficient condition for exact nonlinear improvement is

\[
(\sqrt{L_0}+\sqrt Q)^2
<
\mathbb E\|e\|^2.
\]

A paper-level empirical threshold should report \(Q\), gate-crossing mass, and structured directions rather than assuming that the linear threshold transfers unchanged.

## A.6 Absolute-phase non-identifiability

Suppose same-design sub-estimates obey

\[
Z_i=\mu+b+\varepsilon_i,
\qquad i=1,\dots,k,
\]

where the joint noise law does not depend on \((\mu,b)\).

### Theorem A.8 (common-bias non-identifiability)

For every vector \(t\), the parameter pairs

\[
(\mu,b)
\quad\text{and}\quad
(\mu+t,b-t)
\]

induce the same observation law. Hence \(b\) is not identifiable without an external restriction or reference. For any estimator \(T\), at least one of these two parameter points has squared-error risk at least \(\|t\|^2/4\).

Every statistic measurable only from centered differences \(Z_i-\overline Z\) is independent of the shared absolute phase under this model. Such statistics may estimate dispersion, instability, or exchangeability failure, but cannot recover the sign or magnitude of the common defect.

## A.7 Scope

The results above do not prove that every adaptive or nonlinear white-box anchor must fail. They prove exact statements for specified correction information, subspace assumptions, and common-bias observation models. The empirical question is whether a legal runtime observable supplies independent absolute phase in the downstream-sensitive layer-31 span with acceptable tails and compute.
