# Agent 3 — Discrepancies and Required Corrections

## Required paper corrections

1. Replace **“every point budget”** with **“every antipodal line budget `P`”** or **“every even paired-point budget `N=2P`.”** Odd or unpaired budgets are outside the theorem.
2. Define `h(0)=0`, and state that `r_b=0` forces `S_b=0`.
3. State the feasible range `1≤P≤33,024`. `P=0` is infeasible; larger budgets exceed the fixed universe.
4. Clarify that `w_bi` is a total weight on the symmetrized line evaluation, not an arbitrary pair of separate weights on `u` and `-u`.
5. When saying “support size,” do not count zero-weight retained slots as support. If `r_b` means available slots instead, say so explicitly.
6. Add the stronger signed-weight conclusion: the optimum for every fixed support has strictly positive active basis totals and equal positive line weights. Negative weights and zero-total signed cancellations are strictly dominated.
7. Do not phrase T27 as “optimality for arbitrary signed weights” without appending **“inside the fixed antipodal Kerdock-line universe.”** The current proof-work memo's broader “not proved” warning remains correct for arbitrary spherical nodes.

## Reproducibility discrepancy

The source memo reports 2,000 trials at each of 13 budgets, and a later report summarizes 26,000 trials. No original stress-test script or result manifest was located in the Library search. Preserve the number only as an archived report claim unless that artifact is recovered. The new independent verifier is reproducible and should be used for the paper package.

## Non-discrepancies

- The sign of `O-C` is correct.
- `c(r)` remains positive through `r=256`.
- Strict convexity points in the stated concentration direction; it was not a sign reversal.
- Negative line weights and negative basis totals are genuinely covered by the fixed-support inequality.
- Partial and empty bases do not create an exception once `h(0)=0` is stated.
- The full-design risk reproduces.
