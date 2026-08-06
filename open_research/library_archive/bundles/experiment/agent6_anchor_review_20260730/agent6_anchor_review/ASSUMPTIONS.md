# Assumptions required for the layer-31 theorems

## Scoring and probability space

1. The score is represented by a fixed Hilbert norm on all output coordinates.
2. Expectations and conditional expectations exist and all errors/corrections are square-integrable.
3. Network, rotation, reference, and estimator randomness are included consistently in the probability space.

## Correction and selector

4. The correction direction `u` and selector information `G` are legal runtime quantities.
5. The selector formula must use the actual allowed scale set: unrestricted, nonnegative, or bounded.
6. Evaluation targets are not used to form `G`, choose support, or tune scale on the scored cohort.

## Layer-31 replacement

7. `d = mu31_K - mu31_true` is defined with a fixed sign convention.
8. `xi = mu31_hat - mu31_true` is the replacement anchor error, not the proposed correction itself.
9. `J` is the frozen derivative of the actual final replay at the protected particle cloud.
10. The correctable output component is `s=Jd` and the remaining baseline error is orthogonal to the correction subspace, or the general cross-term formula is used.
11. The empirical “relative anchor error” uses an explicitly named denominator. The theorem-relevant metric is downstream weighted: `E||Jxi||²/E||Jd||²`.
12. Any coordinate/support selection is frozen without target leakage; target-aware oracle support is diagnostic only.

## Nonlinear replay

13. A layer-31 mean intervention is implemented as the same translation of every retained particle before the true final affine/ReLU layer.
14. The final ReLU is the only nonlinear operation after the intervention. Longer suffixes require a composition of crossing bounds or direct exact replay.
15. Gate-crossing remainder is measured or bounded for the actual direction; a generic isotropic perturbation does not certify a kink-concentrated direction.

## Common-bias non-identifiability

16. Same-design sub-estimates have the additive form `Zi=mu+b+eps_i`.
17. The joint noise law is invariant to `(mu,b)->(mu+t,b-t)`.
18. There is no external reference, prior restriction, analytic identity, independent design, or weight-derived observable that breaks this symmetry.
19. The theorem applies only to statistics measurable from these observations; it does not close genuinely new absolute-phase observables.
