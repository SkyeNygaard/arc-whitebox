# Checks

## Proof-package checks

* `python verify_theorem_package.py`: passed.
* Certified pointwise subintervals: 1,421.
* Certified global upper margin for `h-K`: `-1.004586240658...e-13`.
* `python verify_manifest.py`: all 32 files verified.
* Full serial clean-room regeneration was attempted but interrupted by the environment's per-command runtime cap; it is not claimed complete in this audit.

## New M certificate

* Exact `h'` Bernstein coefficients computed; all positive.
* Left-region directed bound at cut `37/50` is strictly below `q(1)`.
* Right-region directed lower bound on `q'` is strictly positive.
* Exact rational and decimal output agree.
* Re-running `signed_weight_certificate.py` deterministically reproduces `M_CERTIFICATE.json`.

## Curve checks

* Original Proposition 5 inversion checked symbolically.
* Groupwise Cauchy envelope minimized over integer positive/negative support counts.
* All requested thresholds use one negative support point at the minimizer.
* Kerdock-relative deltas use `L-(1-p)U` with outward-safe theorem bounds.
* CSV and JSON outputs cross-check exactly.

## Sharpness checks

* Four-node searches on `S^1` and `S^2`, deterministic seeds.
* Node geometry and within-positive weight allocation jointly optimized.
* Complete Kerdock-line stress: 1,000 total signed perturbation trials, zero improvements.
