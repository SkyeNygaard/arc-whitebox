# Hostile Review of the Assembled Revision Drafts

**Decision:** ACCEPT FOR EXTERNAL REVIEW; DO NOT RELEASE AS FULLY VERIFIED UNTIL THE NAMED RELEASE GATES PASS.

## Findings

### 1. Universal-impossibility scope leakage

**Pass.** The theorem draft is explicitly static and class-scoped. The empirical draft frames failures as tested-family closures and states reopening conditions. The prohibited universal sentences appear only in a “claims removed” section.

### 2. T22 direction and attainment

**Pass.** The draft uses a one-sided interval beginning at zero and compares Kerdock with an infimum. It does not claim strict suboptimality or cubature attainment of the auxiliary bound.

### 3. T16 upgrade

**Pass internally, external check required.** The draft states the unique all-degree auxiliary optimum and distinguishes it from cubature optimality. The proof outline includes exact quadrature, all-degree reduced costs, sixth-derivative positivity, Hermite feasibility, coefficient positivity, dual equality and uniqueness. The new primal step still lacks a genuinely independent second arithmetic/interval stack.

### 4. Finite-width wording

**Pass after tightening.** The finite-width result is stated only for the fixed MUB/Kerdock line universe and under explicit Gaussian-first-layer and even-Hermite nondegeneracy assumptions. Arbitrary-node finite-width T22 remains open. The draft avoids the misleading phrase “finite-width Kerdock optimality.”

### 5. Signed weights

**Pass.** T27 is fixed-support; the general signed stability result is labeled weak; off-support signed rules remain open.

### 6. Nonlinear and adaptive algorithms

**Pass.** The draft includes a constructive nonlinear ReLU counterexample, a network-dependence warning and residual-kernel recertification. It does not infer all-algorithm Bayes optimality from the limiting kernel.

### 7. Evidence governance

**Pass at manuscript level.** M152 is removed, M146 is quarantined, oracle/deployable labels are separated, base-network grouping is required, and a contamination chronology is supplied.

### 8. Target noise

**Pass as limitation, not as dataset-specific quantification.** The exact reliability attenuation identity is included. A numerical reliability estimate cannot be supplied from the missing source rows and therefore is not fabricated.

### 9. Novelty

**Pass provisionally.** The related-work note identifies the classical ingredients and limits novelty to the kernel-, budget- and estimator-class-specific certificate and support theorems. Priority still requires a human literature review.

### 10. Reproducibility and AI assistance

**Pass as disclosure.** The package states that agent agreement is not independent verification and assigns final responsibility to named humans. It does not call the proof formally verified.

## Remaining blockers

1. full clean T22 v5.1 regeneration in the final published CI;
2. external publication of the archive digest;
3. independent audit of the T16 sixth-derivative/Hermite/Krawczyk step;
4. named human theorem and reproducibility sign-off;
5. final human related-work and assembled-PDF referee review.

These are external verification requirements, not manuscript wording defects. No further internal prose revision can honestly substitute for them.
