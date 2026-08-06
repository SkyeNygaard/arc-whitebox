# External Human Review Packet

**Internal status:** hostile AI-assisted review completed; external named human sign-off pending.

This packet is designed for a mathematician and a reproducibility reviewer. It does not claim or simulate their approval.

## Requested mathematical checks

1. Confirm the one-sided T22 theorem statement and randomized-rule scope.
2. Audit `K_32^(6)>0`, the Faà di Bruno/Bell inequalities, Hermite remainder, coefficient enclosures and uniqueness in T16/T30.
3. Check the fixed-line versus arbitrary-node distinction in T27/T37 and the finite-width extension.
4. Check every theorem-scope row against the manuscript.

## Requested reproducibility checks

1. Run both Decimal/libmpdec and GMP/MPFR engines from a clean checkout.
2. Confirm 1,421 intervals, 23 chunks, manifest coverage and final ratio.
3. Verify the release ZIP SHA-256 against the externally published digest.
4. Confirm that M146 and M152 are absent from evidentiary claims.

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


---

# Release Gates and Human Review Checklist

The following items are **blocking release gates**, not optional improvements.

## Proof release

- [ ] Run the complete T22 v5.1 regeneration from a clean checkout in the pinned CI matrix.
- [ ] Confirm the resulting 59 manifest entries and archive SHA-256 on every supported runner.
- [ ] Publish the archive digest outside the archive in a signed release, immutable commit, DOI/archival record or equivalent.
- [ ] Implement or commission an independent second-stack audit of T16's sixth-derivative positivity, Hermite feasibility and coefficient enclosure. The current C++ audit covers the reduced-cost recurrence only.
- [ ] Have a named human mathematician sign the T22, T16, T27 and finite-width-extension theorem statements and proof outlines.

## Empirical release

- [ ] Preserve base-network grouping and frozen selection chronology in every table.
- [ ] Bundle the five omitted Agent 8 source inputs and the omitted Path-2 source archive, or label those packages derived-output-only.
- [ ] Do not reinstate M146 or M152 unless complete primary packages are recovered and independently reproduced.
- [ ] Quantify target reliability/noise for any learning claim that remains in the paper; otherwise retain only comparative, non-universal wording.
- [ ] Audit official score, FLOP and wall-time claims end to end before using “verified” language.

## Manuscript review

- [ ] Verify every numerical constant against `UPDATED_NUMERICAL_CONSTANTS.json`; remove the older escape-threshold values.
- [ ] Confirm that T22 always says “infimum” and one-sided gap.
- [ ] Confirm that the finite-width result is T27/fixed-support only.
- [ ] Confirm that every oracle result is labeled `ORACLE DIAGNOSTIC` and visually separated from deployable results.
- [ ] Confirm that no prohibited universal claim appears.
- [ ] Complete a named human related-work/priority review.
- [ ] Run a final hostile review on the assembled PDF, not only on component reports.

## Evidence that cannot be self-certified

An AI system cannot satisfy the named-human sign-off, independent-authorship or external-authentication gates by generating another report. These remain open until performed by outside people or infrastructure.
