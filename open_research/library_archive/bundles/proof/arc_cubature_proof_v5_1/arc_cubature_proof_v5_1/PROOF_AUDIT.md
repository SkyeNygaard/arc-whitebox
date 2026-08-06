# Final proof audit

## Status

The near-optimality theorem is complete as a rigorous computer-assisted proof
within the explicit trust base stated in `THEOREM_PROOF.md`.

A separate clean directory was created, every generated formal artifact was
deleted, all 1,079 original curvature intervals and 1,421 certified
subintervals were regenerated, the scalar certificates were recomputed, and
the resulting 32 canonical manifest files matched `PROOF_MANIFEST.sha256` exactly.
The regeneration also produces 23 deterministic curvature-chunk intermediates.
V5.1 tracks those chunks explicitly in the expanded manifest rather than leaving
them individually untracked.

## Defects found and corrected from V4

1. **Broken clean-room dependency.** The independent interval audit imported a
   missing `certificate.py`. It is now self-contained and uses direct mpmath
   formulas only as a non-proof audit.
2. **Invalid two-sided interpretation.** V4 displayed a positive lower bound on
   Kerdock suboptimality, although only an upper bound was proved. V5 correctly
   certifies
   - multiplicative ratio in `[1, 1.0002336550102949]`;
   - relative excess in `[0, 0.02336550102949%]`;
   - additive suboptimality in `[0, 5.68504102061682e-11]`.
3. **Unnecessary floating-point witness trust.** The auxiliary polynomial is now
   loaded from six exact rational coefficients. NumPy is not used by the formal
   proof.
4. **Manifest mutation during verification.** The verifier no longer rewrites the manifest.
   `generate_manifest.py` and `verify_manifest.py` are separate.
5. **Nondeterministic timing fields.** Timing data was removed from formal
   interval artifacts, allowing byte-for-byte clean-room reproduction.
6. **Ambiguous discovery artifacts.** Approximate root data is now a minimal
   file explicitly labeled as untrusted interval-box hints.
7. **Sign-chain presentation.** The special-box order, curvature pattern,
   derivative signs, and set of possible maxima are machine-checked separately.
8. **Theorem scope.** The deterministic kernel theorem is separated from its
   random-network MSE corollary, and adaptive/signed/finite-width exclusions are
   explicit.

## What “complete” does and does not mean

Complete means that every nonstandard inequality and numerical constant needed
for the stated theorem is regenerated and certified by the package, and the
mathematical deductions from those certificates are supplied.

The trust base still includes CPython integer/Fraction arithmetic,
`decimal`/libmpdec directed-rounding behavior, the standard spherical-harmonic
addition theorem, and the classical existence of the real Kerdock MUB family.
The result has not been formalized in Lean, Coq, or Isabelle, nor does the
package re-prove the Kerdock construction from finite-field axioms.

No unresolved logical gap is currently known inside the stated theorem scope.
