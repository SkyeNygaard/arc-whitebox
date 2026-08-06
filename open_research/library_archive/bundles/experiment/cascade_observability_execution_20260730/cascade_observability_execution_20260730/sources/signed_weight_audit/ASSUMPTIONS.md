# Assumptions

1. `K=K_32` is the explicit normalized depth-32 ReLU kernel.
2. `h` is the exact degree-5 dimension-256 Gegenbauer witness from the verified T22 package.
3. The T22 pointwise certificate `h(t)<=K(t)` on `[-1,1]` is accepted within its stated CPython/Fraction/decimal trust base.
4. The signed rule is a consolidated finite signed measure with total mass one and at most `N=66,048` nonzero support points.
5. `beta` is its Jordan negative mass; positive mass is `1+beta` and total variation is `1+2 beta`.
6. For randomized rules, the bound is applied conditionally to each realized rule; network independence is required only for the ensemble-MSE interpretation.
7. The exclusion curve concerns the infinite-width kernel objective, not the finite-width competition objective pointwise on each network.
8. Low-dimensional searches are exploratory numerical tests on subspheres embedded in `S^255`; they are not proof certificates.
