# Hostile audit of T41

## Verdict

**VALID UNDER ITS EXPLICIT MODEL.** The identity and normalized bound follow directly from measure preservation, unitarity and Cauchy–Schwarz.

## Checks

1. The transformation is applied to both the error and correction; omitting either would make the identity false.
2. The involution need only be measure preserving; independence assumptions are unnecessary.
3. Zero correction or zero error should be handled before defining normalized defects.
4. The zero-value corollary requires exact anti-equivariance of the relevant error component and exact equivariance of the policy.
5. Approximate symmetry yields only the quantitative defect bound, not zero alignment.
6. The randomized-orientation corollary closes orientation-blind policies by construction. It does not retroactively prove symmetry of a deterministic orientation convention.
7. A finite failed learner does not upper-bound the defect terms for every measurable policy.
8. Full network weights determine the target in principle; any impossibility statement for that transcript requires an explicit computational or representation restriction.

## Admissible paper claim

> Under a specified measure-preserving phase action, the signed value of a correction is bounded by the error anti-symmetry defect plus the correction equivariance defect. Exact orientation-blind policies have zero value in an explicitly randomized signed-orientation model.

## Claims that remain inadmissible

- all legal features are orientation blind;
- the current M158 deterministic features satisfy exact phase symmetry;
- no orientation-aware statistic can carry phase;
- no nonlinear policy can improve;
- a negative cross-validation result is an information-theoretic upper bound.
