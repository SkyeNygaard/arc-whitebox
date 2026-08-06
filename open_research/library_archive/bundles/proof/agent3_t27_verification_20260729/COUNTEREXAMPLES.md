# Agent 3 — Scope Adversary and Counterexamples

These examples do not contradict T27. They show why its conclusion must not be extended beyond its stated class.

## 1. Outside the fixed Kerdock line universe

For two arbitrary projective lines with `|<u,v>|=1/2`, the even kernel pair value is

`0.9754394856788864`,

which is different from all three T27 values `A`, `O`, and `C`. The three-class reduction therefore fails immediately outside the Kerdock universe.

For `P≤256`, any generic Haar-rotated orthonormal `P`-frame outside the named Kerdock universe has the same pair geometry and risk as a `P`-line subset of one Kerdock basis. Thus T27 cannot imply geometric uniqueness or that optimal lines must literally be Kerdock lines.

No strict static antipodal outside-universe improvement was found or proved here. Existing small-dimensional searches are negative evidence, not a dimension-256 theorem.

## 2. Non-antipodal or unequally paired nodes

T27 uses the even kernel `Kbar(t)=(K(t)+K(-t))/2`. With individual unpaired nodes, the Gram entries are `K(t)`, not `Kbar(t)`, and odd harmonics reappear.

For a single antipodal pair with unequal point weights `alpha` and `1-alpha`, the risk contains

`alpha^2 K(1) + (1-alpha)^2 K(1) + 2 alpha(1-alpha) K(-1)`,

which is minimized at `alpha=1/2`, but this conclusion is a separate one-pair calculation. An arbitrary odd point budget or arbitrary unpaired support is not represented by T27's line counts.

## 3. Nonlinear estimator

Consider the finite bias-free ReLU integrand

`f_a(u)=ReLU(a^T u)`.

For an orthonormal basis `u_1,…,u_256`, the antipodal line observations are

`g_i=(f_a(u_i)+f_a(-u_i))/2=|a_i|/2`.

The spherical mean is

`I(a)=kappa_256 ||a||_2`,

where

`kappa_256 = Gamma(128)/(2 sqrt(pi) Gamma(128.5))`.

The nonlinear estimator

`I_hat = 2 kappa_256 sqrt(sum_i g_i^2)`

is exactly equal to `I(a)` for every `a`. No fixed linear weighting of the `g_i` can equal an L2 norm for every `a`.

This one-neuron function can be embedded as a rank-one path in a depth-32 bias-free ReLU network. Therefore T27 cannot be cited as a universal impossibility theorem for nonlinear white-box estimators on finite networks.

## 4. Network-dependent support or weights

For the same `f_a`, the exact mean is analytically known from the network weight `a`. A network-dependent node can be chosen with

`a^T u = kappa_256 ||a||`,

so the one-point evaluation `f_a(u)` equals the exact spherical mean. Alternatively, given two line values `y_1 != y_2`, network-dependent real weights can solve

`w y_1 + (1-w)y_2 = I(a)`

exactly.

These constructions may be oracle-like or special-purpose, but that is enough to refute any inference from T27 to universal network-adaptive impossibility. The kernel expectation cannot be pulled outside the weight products once the weights are correlated with the realized network.
