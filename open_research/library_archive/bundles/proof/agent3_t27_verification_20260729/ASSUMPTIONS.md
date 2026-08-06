# Agent 3 — Assumptions and Scope

T27 is valid under all of the following assumptions.

1. **Kernel model.** Risk is the ensemble mean-squared integration error induced by the normalized infinite-width, depth-32 ReLU kernel in dimension 256.
2. **Static rule.** The support and weights are fixed independently of the realized network/function. Randomization independent of the network is covered realization-by-realization when the line budget and mass constraint hold almost surely.
3. **Fixed universe.** Every node belongs to one of the 33,024 projective lines in the chosen 129-basis maximal real-MUB/Kerdock universe.
4. **Antipodal symmetrization.** A line observation is `(f(u)+f(-u))/2`; `w_bi` is the total weight on that symmetrized line observation.
5. **Mass constraint.** Real line weights satisfy `sum w_bi=1`.
6. **Budget unit.** `P` counts antipodal lines. Runtime evaluation uses `2P` individual points unless antipodal evaluations have a special shared-cost implementation.
7. **Linear estimator.** The estimator is the weighted sum of the line observations. No nonlinear postprocessing of observations or direct white-box analytic term is included.
8. **Support convention.** `r_b` counts available/nonzero retained lines. Empty bases have `r_b=0` and necessarily `S_b=0`.

T27 does not establish:

- optimality over nodes outside the fixed Kerdock universe;
- optimality for unpaired or unequally weighted antipodal points under an arbitrary point budget;
- finite-width width-256 optimality;
- optimality of network-dependent supports or weights;
- optimality of nonlinear estimators or analytic-plus-residual estimators;
- unrestricted signed-weight optimality over arbitrary spherical nodes;
- uniqueness of the geometric node set outside the fixed universe.
