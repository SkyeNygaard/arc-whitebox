# Kerdock/MUB near-optimality proof package — V5

This package proves a one-sided, computer-assisted near-optimality theorem for
the explicit dimension-256, depth-32 infinite-width normalized ReLU kernel.

The uniform 66,048-point antipodal real-MUB/Kerdock rule is certified to be at
most **0.02336550102949%** above the infimum among nonnegative-weight rules with
weights summing to one and support size at most 66,048.

The theorem is about deterministic kernel discrepancy. It becomes an ensemble
MSE result for rules selected independently of the random network/field. It
does not cover adaptive rules, signed weights, nonlinear estimators, or finite
width.

## Fast verification

```bash
python verify_theorem_package.py
python verify_manifest.py
```

The first command reassembles the saved directed-rounding chunks and recomputes
all low-cost theorem-critical quantities. The second checks the fixed SHA-256
manifest; the verifier never rewrites the manifest.

## Full clean-room regeneration

```bash
./FULL_PROOF_REPRODUCE.sh
```

This deletes and regenerates the base certificate, every curvature chunk, the
sign proof, kernel mean, Delsarte bound, one-sided theorem JSON, and verification
result, then checks that all deterministic artifacts match the shipped
manifest.

## Main files

- `THEOREM_PROOF.md` — complete theorem, derivation, scope, and trust base.
- `auxiliary_coefficients_d256_L32_deg5.json` — exact rational polynomial witness.
- `results/FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32.json` — one-sided machine-readable theorem.
- `results/FORMAL_CERTIFICATE_D256_L32.json` — formal pointwise certificate for `h<K_32`.
- `results/FORMAL_KERNEL_MEAN_D256_L32.json` — rigorous spherical mean.
- `results/FORMAL_DELSARTE_BOUND_D256_L32.json` — exact weighted energy bound.
- `PROOF_MANIFEST.sha256` — immutable hashes of deterministic proof sources and artifacts.

## External mathematical input

The core kernel theorem is conditional on an antipodal union of 129 pairwise
real mutually unbiased bases in `R^256`. The classical Kerdock construction
supplies such a family. The proof package verifies all multiplicities used once
that incidence property is assumed; it does not re-prove the Kerdock existence
theorem.

## Requirements

- Python 3.11+
- `mpmath` only for independent implementation audits; the formal directed
  certificate itself uses the Python standard library.
