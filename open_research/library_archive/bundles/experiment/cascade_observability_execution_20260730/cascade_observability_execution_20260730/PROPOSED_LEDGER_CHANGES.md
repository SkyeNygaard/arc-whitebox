# Proposed canonical-ledger changes

## New theorem/result entries

### T29 — All-width complete-support linear symmetry theorem

- **Evidence:** exact proof plus exhaustive Kerdock row-profile audit.
- **Claim:** For the architecture prior at any finite width, uniform mass-one weights minimize ensemble MSE among fixed linear rules on the complete 66,048-point Kerdock support. If total mass is free, the optimum is alpha-scaled uniform.
- **Scope:** fixed linear weights; complete support; rotationally invariant ensemble/input measure; square-integrable zonal second moment.
- **Exclusions:** data-dependent weights, nonlinear aggregation, changed support, compute-adjusted partial rules.
- **Source:** `FINAL_REPORT.md`, `results/MATH_DESIGN_AUDIT.json`, `code/run_math_and_design_audit.py`.

### T30 — Infinite-width global scale audit

- **Result:** `alpha*=0.9999997503247287`; relative risk reduction `2.49675e-7`.
- **Verdict:** exact but operationally negligible.

### T31 — Gaussian Bayes/no-adaptation route invalid for post-ReLU output

- **Evidence:** exact distributional counterargument plus explicit ReLU-ridge nonlinear counterexample.
- **Verdict:** invalidate C3/C4/C8 as universal claims for the challenge output.
- **Source:** `FINAL_REPORT.md`, `results/MATH_DESIGN_AUDIT.json`.

### M153 — T4 grouped phase-predictability audit

- **Cohort:** development IDs 6000–6015; rotations 3/11/97 grouped; calibration/validation sealed.
- **Features:** nine retained target-free geometry summaries.
- **Models:** nested grouped ridge and ExtraTrees.
- **Result:** negative OOF R2 for every phase/ratio target; arm-selection accuracy 56.25%; selected mean ideal cosine below fixed best arm.
- **Verdict:** close this feature dictionary; do not interpret as an upper bound on all S2 exploitability.
- **Source:** `results/TEST2_T4_OBSERVABILITY_PROBE.json`.

### M154 — Mode-resolved transfer diagnostic

- **Cohort:** 12 fresh architecture-matched synthetic networks, 192 Gaussian rows per network.
- **Result:** exact scale response to `2.25e-7`; max tested mean/random/shape gain 1.011/1.196/0.143; no tested mode exceeded the 1.5 amplification gate.
- **Verdict:** diagnostic support for no large compounding in tested directions; not a theorem.

### T32 — Signed-weight M certificate reproduction

- **Result:** `M=0.017021894267861247...`, supremum at `t=1`.
- **Verdict:** rigorous, but exclusion curve is practically vacuous for off-support signed rules.

## Downgrades / corrections

1. Mark “baseline exact Bayes rule at infinite width” **INVALIDATED for post-ReLU output**.
2. Mark “adaptive point selection cannot help at infinite width” **NOT ESTABLISHED / theorem assumptions fail**.
3. Mark “all remaining legal headroom is finite-width gamma=O(L/n)” **INVALIDATED**.
4. Mark “TEST-2 can prove gamma(256)<=1%” **INVALID TEST LOGIC**.
5. Mark “observability gap = 78% minus gamma” **OPEN / NOT A THEOREM**.
6. Correct C2 wording: prior mean zero unnecessary; minimizer not necessarily unique; free total mass gives alpha-scaled uniform.
7. Keep T22/T27 scopes unchanged and explicit.
8. Replace “zero-alignment meta-pattern” with a stratified statement: several finite-width S2 corrections have nonzero mean alignment but fail tails, cost, or completeness.
