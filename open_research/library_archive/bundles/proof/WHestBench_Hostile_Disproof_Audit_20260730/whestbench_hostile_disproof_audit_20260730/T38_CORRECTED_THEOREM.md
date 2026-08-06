# Corrected T38 — finite-width fixed-MUB-line theorem

## Missing assumption in the original round-two statement

The originally enumerated assumptions required only that the antipodally even output be nonconstant. That is not enough for the strict sign

\[
(A-O)+d(O-C)>0.
\]

A square-integrable even nonconstant counterexample is

\[
F(g)=g_1^2-1.
\]

Its noise-stability kernel is pure degree two, proportional to `t^2`, so

\[
A=1,\qquad O=0,\qquad C=1/d,
\]

and therefore

\[
(A-O)+d(O-C)=0.
\]

The complete-basis coefficient is then flat rather than strictly positive, and the old claims that every optimum uses all budgeted lines and has positive basis masses do not follow.

## Correct theorem

Retain the Gaussian-first-layer assumptions and additionally require

\[
\sum_{r\ge2}a_{2r}>0,
\]

that is, positive antipodally even Hermite/noise-stability mass at some degree at least four. Equivalently, assume the three strict MUB association signs directly:

\[
A-O>0,\qquad O-C<0,\qquad (A-O)+d(O-C)>0.
\]

For a nonconstant finite piecewise-affine ReLU realization, the high-even condition follows from the argument that an even piecewise-affine function whose Hermite expansion stops at degree two would have to be a globally quadratic, hence affine, hence constant function.

For a fixed union of `M` real MUBs and a feasible budget

\[
1\le P\le Md,
\]

the corrected theorem gives complete bases plus at most one partial basis, equal positive weights within active bases, and positive analytic basis masses.

The latest v17 theorem manuscript already uses this corrected high-even condition.
