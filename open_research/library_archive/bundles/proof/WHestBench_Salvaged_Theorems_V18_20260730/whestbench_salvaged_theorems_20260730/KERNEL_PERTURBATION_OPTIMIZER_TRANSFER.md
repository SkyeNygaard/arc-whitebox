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
