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
