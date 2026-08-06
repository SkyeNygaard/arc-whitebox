# Counterexamples

## Counterexample to “analytically integrable implies low degree”

For `0<r<1`, unit `v`, and `u` on `S^(d-1)`, define

`P_r(t)=(1-r^2)/(1-2rt+r^2)^(d/2)`

and

`A_r(u)=0.5(P_r(v^T u)+P_r(-v^T u))`.

The spherical expectation of `A_r` is exactly 1. It is nonpolynomial and its spherical-harmonic expansion contains all even degrees, with nonzero multipliers proportional to `r^ell`. Thus it has explicit degree-6, degree-8, degree-10, and arbitrarily high content despite an exact analytic expectation.

For Gaussian `X`, define `g_r(X)=A_r(X/||X||)` away from the measure-zero origin. Since `X/||X||` is uniform on the sphere, `E[g_r(X)]=1` exactly.

## Counterexample to “all polynomial Stein fields vanish”

Take a polynomial vector field with component degree 5. Its Gaussian Stein image can contain degree 6 through `-x^T phi(x)`. A spherical 5-design has no obligation to integrate that degree-6 term exactly. Therefore the unqualified statement is false even before constructing a particular network.

## Boundary example for the ReLU-Stein lemma

Add an input bias: `phi(x)=a ReLU(v^T x+b)`. Positive homogeneity and the antipodal pair identity used in the proof fail. The exact blockwise cancellation theorem no longer follows. This does not prove a biased field works; it shows the theorem’s boundary is real.
