# Checks

## Artifact checks

- Read the canonical v15 Paper Claims Matrix, Paper Evidence Index, Agent Review Plan, and experiment rows T25, M79, M80, V67, and V71.
- Read the full frozen harmonic experiment report.
- Inspected the archived Agent-7/8 numerical JSON and depth-2 Stein reference implementation.
- Inspected the exact limiting-kernel high-degree frontier and the Agent-1 harmonic-attribution report.

## Independent algebra

- Derived the radialized polynomial `t`-design lemma.
- Derived the degree bound for the Gaussian Stein image.
- Derived exact blockwise cancellation for a one-hidden-layer bias-free ReLU vector field.
- Constructed the symmetrized Poisson-kernel counterexample.

## Independent numerical checks

Run:

```bash
python verify_harmonic_claims.py
```

Observed:

- ReLU-Stein block absolute mean: `1.1546319456101628e-14`.
- Poisson-kernel spherical mean: `1.000000000000003`.
- Poisson-kernel mean error: `3.1086244689504383e-15`.
- Numerically nonzero normalized Gegenbauer coefficients at degrees 6, 8, 10, and 12.

The numerical coefficient magnitudes depend on Gegenbauer normalization. Their nonzeroness, not their scale, is the relevant check.
