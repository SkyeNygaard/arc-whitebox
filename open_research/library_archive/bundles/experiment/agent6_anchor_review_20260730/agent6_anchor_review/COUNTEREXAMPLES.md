# Counterexamples and adversarial cases

## 1. Equal Euclidean error, different downstream risk

Let `J=diag(1,kappa^-1)` and let the protected defect be `d=epsilon e1`. An anchor error of Euclidean norm `epsilon` has:

- downstream error `epsilon` in direction `e1`;
- downstream error `epsilon/kappa` in direction `e2`.

The first is exactly break-even; the second is safer by `kappa²` in MSE. Reversing the singular values makes the same Euclidean threshold arbitrarily too loose. Therefore only the `J`-weighted norm has an invariant replacement threshold.

## 2. Out-of-subspace anchor error

The simple theorem says replacement risk is `||r||²+||n||²` only when `n` is orthogonal to `r`. If `n` has an out-of-subspace component positively aligned with `r`, replacement can help even with `||n||²>||s||²`; if anti-aligned, it can fail despite `||n||²<||s||²`. The exact criterion is

`E||n||² - 2E<r,n> < E||s||²`.

## 3. Correlated anchor noise

For in-subspace anchor error with `K=E<s,n>`, optimal shrinkage is

`alpha=(S+K)/(S+N+2K)`.

The archived `S/(S+N)` formula is wrong when systematic anchor bias correlates with the true correctable component. Sufficiently negative `K` can make the estimated correction point in the wrong signed direction.

## 4. Positive-only selector

An unrestricted selector can choose a negative scale when `C_G<0`; a legal positive-only policy cannot. Its gain is `(C_G)_+²/U_G`, not `C_G²/U_G`. Omitting this distinction overstates the value of a gate that detects harmful phase but is not allowed to reverse the correction.

## 5. Kink-concentrated direction

A direction can have small Euclidean norm and ordinary downstream linear norm while shifting a high-density set of preactivations across zero. The exact ReLU remainder scales with near-kink mass, not only with `||delta||`. The executed synthetic audit produced a median remainder equal to 25.1% of the linear shift for a kink-focused `5e-4` translation, versus `2.94e-5` in generic geometry.

## 6. Smooth same-design convergence around the wrong phase

For `Zi=mu+b+eps_i`, every centered diagnostic depends only on the noise. Letting `(mu,b)` and `(mu+t,b-t)` vary leaves every fold, jackknife, and nested-convergence statistic unchanged while changing the true defect. Therefore smooth convergence is not evidence of correct absolute phase under this model.
