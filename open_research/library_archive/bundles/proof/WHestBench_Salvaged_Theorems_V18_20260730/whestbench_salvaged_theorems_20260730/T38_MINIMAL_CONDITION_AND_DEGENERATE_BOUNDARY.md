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
