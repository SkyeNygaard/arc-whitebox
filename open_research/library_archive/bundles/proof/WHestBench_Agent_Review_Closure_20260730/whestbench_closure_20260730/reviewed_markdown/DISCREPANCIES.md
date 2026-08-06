# Discrepancies — Agent 4 / T16

1. **The published narrative says the finite audit ran through degree 1,000,000, while `dual_reduced_costs.py` in the V3 bundle sets the d=256 loop limit to 100,000.** The same bundle does contain `reduced_costs_d256_to_1m_tail.npy` with 1,000,001 entries, so the million-degree artifact exists, but its generation is not performed by the displayed script.

2. **The V3 contacts and masses are binary64 approximations, not exact witness data.** The exact dual nodes can instead be defined as roots of `22102 t^3 + 21930 t^2 - 87 t - 85`, eliminating this dependency for T16.

3. **The exact rational V5 primal minorant is intentionally shifted downward for safety.** It therefore cannot by itself establish exact complementary contact with the T16 dual measure. This does not affect the reduced-cost theorem, but it matters for the stronger phrase “the degree-5 primal is exactly all-degree LP-optimal.”

4. **The previous proof memo identified a missing tail inequality.** That inequality is now supplied with an explicit cutoff, so the ledger should no longer describe the analytic tail itself as open.
