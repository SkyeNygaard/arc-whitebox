# Assumptions — Agent 4 / T16

1. The normalized Gegenbauer convention is
   \[
   G_\ell(t)=C_\ell^{(127)}(t)/C_\ell^{(127)}(1),\qquad G_\ell(1)=1.
   \]
2. The auxiliary objective gradient is exactly
   \[
   q_0=1-1/66048,\qquad q_\ell=-1/66048\quad(\ell\ge1).
   \]
   This is the convention used by the recovered V3 source.
3. The standard Laplace integral representation for normalized Gegenbauer polynomials is used:
   \[
   G_\ell(x)=\frac{\Gamma(\alpha+1/2)}{\sqrt\pi\Gamma(\alpha)}
   \int_{-1}^{1}(x+i\sqrt{1-x^2}s)^\ell(1-s^2)^{\alpha-1}\,ds.
   \]
4. Python integer and `fractions.Fraction` arithmetic implement exact integer/rational arithmetic. No floating-point value is used in a sign decision.
5. The theorem concerns only the target `d=256`, `N=66048` reduced-cost sequence. It does not cover finite-width kernels, signed-node estimator classes, network-adaptive rules, or nonlinear estimators.

No empirical cohort, protected holdout, or challenge grader was opened.
