# Proposed canonical-ledger changes

## Amend T22

**Evidence:** COMPUTER-ASSISTED CERTIFIED; COMPLETE DUAL-ENGINE NUMERICAL REPRODUCTION.

**Replacement result text:**

> The scoped near-optimality theorem is independently reproduced end to end. A direct-C GMP/MPFR verifier checks all 1,421 curvature subintervals, regenerates the 1,079-to-1,421 tree with the same 342 splits, verifies the complete global sign chain, independently recomputes the spherical kernel mean, Kerdock energy, exact Delsarte bound, and final one-sided ratio, and obtains ratio upper bound `1.00023365501029481377...`, nested inside the published outward-rounded `1.0002336550102949`. The explicit 129-basis construction and production asset match remain separately certified.

**Qualification:** The mathematical derivation and rational witness are shared; this is not proof-assistant formalization. The theorem scope is unchanged.

## Amend T23

- Correct the original manifest count from 31 to 32.
- Record exact all-file release coverage, including all 23 curvature chunks.
- Record GCC 14.2 and Clang 17 byte-identical outputs for every primary MPFR artifact.
- Record successful AddressSanitizer, UndefinedBehaviorSanitizer, and LeakSanitizer execution.
- Retain “fixed release manifest plus separately distributed archive SHA-256,” not “immutable manifest.”

## Retain T29 — explicit Kerdock incidence certificate

- **Evidence:** Exact finite computer-assisted certificate.
- **Result:** 128 distinct chirps; full field inverse/permutation checks; all 8,128 pair-product Walsh spectra have absolute value 16; exact production asset match; fixed-node multiplicities certified.

## Replace T30 — arithmetic implementation diversity

- **Family:** Proof software verification.
- **Evidence:** Complete independent directed-rounding implementation.
- **Result:** GMP/MPFR reproduces every theorem-critical numerical component, including the full curvature mesh, global minorant, spherical mean, Delsarte bound, and final ratio. GCC and Clang outputs are byte-identical; sanitizer builds pass.
- **Verdict:** Strong positive audit; remove the prior “partial” qualification.

## Add T31 — exact domain and subdivision audit

- **Family:** Proof coverage verification.
- **Evidence:** Exact `Fraction` audit independent of proof modules.
- **Result:** Every certified interval is a valid dyadic descendant of its source interval; all source intervals are exactly covered; the critical, inflection, endpoint, tail, and connecting regions cover `[-1,1]` without gaps or overlaps.
- **Verdict:** Retain as proof-governance evidence.
