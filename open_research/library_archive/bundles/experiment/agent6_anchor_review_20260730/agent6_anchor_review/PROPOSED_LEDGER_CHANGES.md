# Proposed canonical-ledger changes

## Add theorem row T29

**ID:** T29  
**Evidence level:** Exact Hilbert-space derivation plus property checks  
**Family:** Layer-31 correction theory  
**Experiment/claim:** Constrained selector, generalized anchor replacement, correlated shrinkage, and nonlinear replay margin  
**Result:**

- unrestricted selector gain `E[C_G²/U_G]`;
- positive-only gain `E[(C_G)_+²/U_G]`;
- bounded scale is clipped pointwise;
- general full-replacement criterion `N-2E<r,n><S`;
- in-subspace correlated shrinkage `alpha=(S+K)/(S+N+2K)`;
- exact nonlinear sufficient condition `(sqrt(L0)+sqrt(Q))²<R0`.

**Verdict:** PROVED / PROVED UNDER EXPLICIT MODEL  
**Primary source:** `PAPER_APPENDIX_LAYER31.md`, `THEOREM_CHECKS.json`  
**Caveat:** Does not establish observability or a universal adaptive impossibility result.

## Downgrade/split M146

### M146a — headline perturbation curve

**Status:** EXPLORATORY EMPIRICAL / PROVISIONAL TRANSCRIPT  
**Result:** Preserve the reported numbers, but explicitly say original rows, IDs, perturbation manifest, and script are absent. Add the internal quadratic consistency check: `R²=0.9999965`, fitted break-even `5.80e-4`.  
**Do not label:** reproduced, frozen, exact empirical mechanism, or paper-citable.

### M146b — structured-direction robustness

**Status:** OPEN / REQUIRED REPLICATION  
**Required directions:** isotropic; actual protected-design residual; each legal analytic-estimator residual; each legal companion residual; leading `J` singular directions; sparse selected coordinates; near-kink directions.  
**Required metrics:** `eta_J`, exact final MSE, linearized MSE, correction cosine, crossing mass, remainder, wins, p90, worst, and grouped interval.

## Add diagnostic row M153

**ID:** M153  
**Evidence:** Reproducible synthetic counterexample, 60 anisotropic linear operators and 30 exact-ReLU simulations  
**Claim:** An unweighted scalar anchor-error threshold is not invariant to direction or gate geometry.  
**Result:** Median synthetic break-even ranged from `1.87e-4` in the leading downstream direction to `1.41e-2` in the trailing direction; kink-focused ReLU remainder reached `25.1%` of linear shift at `5e-4`.  
**Verdict:** Refutes universal Euclidean-threshold wording; does not estimate the ARC threshold.  
**Primary source:** `agent6_adversarial_anchor_audit.py` and generated rows.

## Replace reopening criterion

Old wording: “standalone selected layer31 accuracy below approximately `5e-4`.”

Proposed wording:

> Reopen only when a standalone legal source, frozen before scored replay, demonstrates downstream-weighted error `eta_J<1` with margin sufficient for measured ReLU remainder and deployment cost. For promotion, require exact nonlinear final replay on a fresh grouped cohort, positive signed transfer, safe tails, and complete cost. Use `~5e-4` only as a historical isotropic/full-vector screening scale until M146 is reproduced.
