# Discrepancies with the Supplied Blueprint

1. **Final output law:** blueprint treats the infinite-width output as Gaussian; retained code applies final ReLU.
2. **C2 prior mean:** claimed zero but post-ReLU mean is positive; proof does not require zero mean.
3. **C2 uniqueness:** minimization follows; uniqueness needs an additional positive-definiteness condition.
4. **Global alpha:** uniform mass-one is constrained optimum; unconstrained linear optimum is alpha-scaled.
5. **C3:** kernel quadrature is a best linear rule, not established all-algorithm posterior mean for the non-Gaussian output.
6. **C4:** Gaussian no-adaptation theorem assumptions fail for the actual output process.
7. **C8:** lacks a proved per-layer legal floor and noncancellation theorem even apart from C3/C4.
8. **C9:** finite-width covariance correction size is not a decomposition of total nonlinear exploitability.
9. **TEST-2:** model failure cannot upper-bound a supremum over all S2-measurable corrections.
10. **Width scaling:** changing square network width also changes input/design dimension, confounding geometry and width unless a rectangular fixed-input architecture is defined.
11. **TEST-4:** archived intervention curve is not strictly monotone early; coherence matrix unavailable.
12. **TEST-6:** pooling S1/S2, infinite/finite, oracle/legal, and cost-adjusted/raw families is not statistically meaningful.
13. **TEST-7:** certificate completes, but its intended material-exclusion gate fails because the resulting curve is weak.
