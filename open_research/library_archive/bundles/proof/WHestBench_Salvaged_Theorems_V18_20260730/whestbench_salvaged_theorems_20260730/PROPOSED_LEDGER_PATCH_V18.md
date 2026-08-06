# Proposed ledger patch — v18 salvaged theorems

## Supersede

- Original general T29 free-mass uniqueness wording.
- Original general T38 implication from even nonconstancy alone.
- Any conditional-Haar claim conditioning only on runtime features and not the integrand/selected rule.
- Any replication claim equating independence of biased errors with zero raw cross moment.
- Any arbitrary-perturbation ReLU cubic bound based only on an unspecified density bound “near zero.”

## Add

### T41 — symmetric-Gram minimizer-set theorem

**Status:** analytically proved.

For constant Gram row sums and constant target cross-covariance, fixed-mass minimizers are

`alpha u + (ker G intersect 1-perp)`.

Free-mass minimizers are

`alpha_* u + (ker G intersect 1-perp)`

when the uniform energy is positive. Include the exact stability modulus `lambda_perp`.

### T42 — K32 strict positive-definiteness theorem

**Status:** analytically proved.

The second and every later iterate of the normalized ReLU covariance map has strictly positive Maclaurin coefficients in every degree. Hence K32 is strictly positive definite on every finite set of distinct sphere points. This restores unique mass-one and free-mass fixed-linear optima on the complete Kerdock point support. Add the directed K32 line-spectrum certificate and its `0.009564733824...` stability modulus.

### T43 — MUB even-kernel trichotomy

**Status:** analytically proved.

- constant case: complete degeneracy;
- constant-plus-quadratic case: complete-basis block-uniform minimizer family;
- degree-four-or-higher case: strict T27/T38 complete-basis concentration.

Record the exact MUB Gram eigenvalues and the finite-budget quadratic boundary.

### T44 — conditional relative-Haar theorem

**Status:** analytically proved.

Exact no-value requires the relative orientation to remain Haar conditional on the integrand, selected rule, and runtime information. Add chi-square and total-variation approximate versions.

### T45 — bias-covariance-compute replication theorem

**Status:** analytically proved.

Record the exact ratio

`c_m [ beta + (1-beta)(1+(m-1)rho)/m ]`

and the independent/shared-compute/antithetic corollaries.

### T46 — sharp ReLU crossing theorem

**Status:** analytically proved.

Replace the old cubic constant by the exact crossing integral and the bound

`E r^2 <= (1/3) E[L(T)|T|^3]`.

### T47 — uniform kernel optimizer-transfer theorem

**Status:** analytically proved.

An `eta`-optimal surrogate rule is `eta+2 epsilon(1+B)^2` optimal for the target kernel when the TV bound `B` is uniform over the entire class.

## Paper wording changes

1. General T29 wording should discuss the minimizer affine space, then state K32 strict uniqueness as a separate corollary.
2. T38 should present the pure-quadratic case as an exact boundary theorem rather than only as a counterexample.
3. Haar symmetry should be described as a controllable randomized-design guarantee or a quantitatively testable near-symmetry, not a property inferred from feature invariance alone.
4. Replication should be described as variance reduction with persistent bias, not as automatically score-neutral.
5. ReLU nonlinear transfer should use the sharp crossing-density constant and a downstream norm bound.

### T48 — observability metric convention

**Status:** analytically proved.

Always report oracle capacity, transferred value, and their nonnegative gap. Define a transferred fraction only when oracle capacity is positive.

### T49 — T16 endpoint separation

**Status:** directed-decimal certified.

Both endpoint residuals are strictly positive; equality in the Hermite minorant occurs exactly at the three interior contact nodes.
