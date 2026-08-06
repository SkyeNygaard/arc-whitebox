# Human review checklist — Oracle proof completion v18

## T42 numerical certificate

- [ ] Confirm the normalized ReLU dual activation convention and 32-fold composition.
- [ ] Check the formal Taylor recurrence for square root, inverse, `acos`, integration and composition.
- [ ] Confirm directed rounding at every operation.
- [ ] Verify the order-47 truncation is a lower certificate because all omitted Maclaurin contributions are nonnegative.
- [ ] Check exact monomial-to-normalized-Gegenbauer projections.
- [ ] Compare all T16 coefficient upper intervals with the new lower intervals.
- [ ] Reproduce in an independent interval stack.

## T43/T47 abstract rank proof

- [ ] Confirm spherical-harmonic normalization and addition theorem.
- [ ] Verify `tr(M_Q)=tr(A)` for arbitrary signed mass-one weights.
- [ ] Verify `rank(M_Q)<=N` with repeated nodes and signed weights.
- [ ] Check `R_{L_a^2}(Q)=||A-M_Q||_F^2` under the chosen probability normalization.
- [ ] Prove the trace-constrained rank-`N` approximation formula `F_N(A)` with indefinite `M` allowed.
- [ ] Check that selecting the largest eigenvalues and rank exactly `N` is optimal.
- [ ] Verify constant Gegenbauer coefficients cancel for mass-one discrepancy.
- [ ] Confirm positive-definite residual energy for arbitrary signed finite measures.
- [ ] Confirm conditioning extends the result to randomized rules independent of the realized field.

## T47 specialization

- [ ] Treat every displayed terminating decimal as an exact rational.
- [ ] Recompute all `b_r`, degrees 1–30, independently.
- [ ] Recompute all directed lower `k_r`, degrees 1–30.
- [ ] Verify degree 7 is binding.
- [ ] Verify the repeated harmonic dimensions and top-66,048 eigenvalue selection.
- [ ] Recompute the exact rank defect.
- [ ] Recompute the MSE floor, Kerdock-relative fraction and improvement factor.
- [ ] Confirm the quoted Kerdock MSE belongs to the same kernel normalization.

## T44–T46 exact proofs

- [ ] Check the binary mutual-information identity and units in nats.
- [ ] Check the conditional weighted phase bound and its magnitude factor.
- [ ] Check the T45 pullback action, measure preservation and unitary conventions.
- [ ] Check the exact alignment identity and fixed-direction risk corollary.
- [ ] Check Haar projection and invariant-subspace arguments in T46.
- [ ] Confirm the M153 application is explicitly representation-level, not distribution-level.

## Empirical interpretation

- [ ] Verify the pooled coherence computations from authenticated arrays.
- [ ] Confirm the scalar counterexample invalidates a within-network rank inference.
- [ ] Confirm five-source development failure precedes confirmation inspection.
- [ ] Confirm the Edge-DWS stop is source-ceiling based, not a model failure.
- [ ] Confirm no protected cohort was opened.

## Release decision

- [ ] All theorem statements carry width, support, signs, adaptation and estimator scope.
- [ ] Computer-assisted results are not called formally verified.
- [ ] External digest, clean environment and human names are recorded.
- [ ] Exploratory weight-search artifacts are segregated from the frozen certificate.
