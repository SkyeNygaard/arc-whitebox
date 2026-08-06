# Proposed ledger patch — infinite Hermite factor continuation

Use the next free canonical IDs at merge.

## T-next — Canonical three-contact Hermite factor reduction

- **Family:** Static signed arbitrary-node theory / weighted harmonic factorization.
- **Statement:** The stabilized cutoff-63 through cutoff-123 weight vectors are finite truncations of a canonical factor `L` satisfying `L^2=K32-q`, where `q` is the degree-five Hermite interpolant at three interior double-contact points. The contact points are selected by the exact equalization equations `c0=c1=c2=c3`.
- **Evidence:** 100+ digit root solve; exact Hermite algebra; high-precision factor series; agreement with all stabilized finite certificates.
- **Verdict:** Strong structural result; root isolation still needs directed interval certification.

## T-next — Conditional 95.4625% signed floor

- **Population:** Dimension 256, depth-32 limiting normalized ReLU kernel; arbitrary static, network-independent, mass-one real-weight rule; at most 66,048 nodes.
- **Conditional statement:** If every Taylor coefficient `[t^n]L(t)` is nonnegative for `n>=2`, then
  `R(Q) >= 2.3232332157460956e-7 >= 0.9546250809178666 R_Kerdock`, so same-cost improvement is at most `1.0475316645132617x`.
- **Reason:** `L` is then positive definite; `L^2=K-q`; all Gegenbauer coefficients of `q` through degree five are nonnegative; the rank defect is exactly `b0-2sL(1)+L(1)^2/N`.
- **Verdict:** Conditional theorem candidate, not yet canonical theorem.

## E-next — Long-prefix and asymptotic positivity evidence

- **Result:** High-precision positivity through Taylor degree 505 and Gegenbauer degree 300. Independent long-double recurrences at orders 8,191 and 16,383 contain no negative `S`, `sqrt(S)`, `L_{n>=2}`, or `log S` coefficients and agree on recorded interior indices through 8,000. The leading negative-endpoint singular amplitude is only `0.000486673` of the positive-endpoint amplitude.
- **Verdict:** Strong evidence for the missing all-degree sign lemma. Does not replace directed finite intervals plus an explicit asymptotic remainder bound.

## Next action

Prove all-degree coefficient positivity using an Arb finite-prefix check and a two-endpoint Puiseux/Darboux tail certificate. Do not spend the next cycle on a higher finite cutoff unless this proof route fails.
