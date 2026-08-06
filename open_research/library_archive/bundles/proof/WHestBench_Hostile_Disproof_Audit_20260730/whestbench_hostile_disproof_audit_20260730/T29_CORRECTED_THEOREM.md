# Corrected T29 — complete-support fixed-linear weights

## Correct statement

Let `G` be the complete-support Gram matrix, let `u=1/N`, assume `G1=lambda 1`, and let rotational invariance make the target cross-covariance `m=A0 1`. For `w=alpha u+v` with `1^T v=0`,

\[
R(w)=R(\alpha u)+v^T Gv.
\]

### Mass one

Uniform weights minimize risk. They are unique exactly when `G` is positive definite on `1^perp`.

### Free total mass

When `E_X=u^TGu>0`, a scalar-uniform minimizer exists with

\[
\alpha_*=A_0/E_X.
\]

The complete minimizer set is

\[
\alpha_*u + (\ker G\cap 1^\perp).
\]

Thus **not every minimizer is scaled-uniform** unless `G` is positive definite on the zero-sum subspace. If `E_X=0`, the quotient formula is undefined and the degenerate case must be handled separately.

## Counterexample to the old uniqueness wording

Take the square-integrable constant random field `Y(x)=Z`, normalized by `E||Z||^2=1`. Then `G=11^T`, the spherical integral is exactly `Z`, and every weight vector with total mass one has zero risk. For example,

\[
(1/4,1/4,1/4,1/4)
\quad\text{and}\quad
(5/4,-3/4,1/4,1/4)
\]

are both minimizers, but the second is not uniform.

## Rigorous limiting-kernel scale replacement

Using the archived directed enclosures rather than rounded 16-digit display values:

- `alpha_*` is in
  `[0.9999997503247282806575775152106693,
    0.9999997503247282806578123186727384]`;
- relative mass-relaxation reduction is in
  `[2.4967527171934218768132726164e-7,
    2.4967527171934242248478933068e-7]`;
- absolute MSE reduction is in
  `[6.0762481104214071859423925412e-14,
    6.0762481104214186145799415663e-14]`.

The old 28-digit value derived from 16-digit inputs was not citable at that precision.
