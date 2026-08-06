# Claims checked

| Claim | Verdict | Notes |
|---|---|---|
| Correction-risk identity | **VERIFIED** | Exact Hilbert-space quadratic identity. Independent property check matched to `8.53e-14` absolute error. |
| Best unrestricted scalar and correlation ceiling | **VERIFIED** | Requires arbitrary real scale. For nonnegative scales use the positive part. |
| Conditional selector value | **VERIFIED AFTER CORRECTION** | The archived formula is unrestricted. Positive-only and bounded policies require clipping; exact formulas are in the appendix. |
| Full-replacement threshold | **VERIFIED UNDER EXPLICIT ASSUMPTIONS** | Exact when anchor noise lies in the correction subspace, or is orthogonal to the uncorrectable remainder. General formula has a cross term. |
| Optimal shrinkage with anchor noise | **VERIFIED AFTER GENERALIZATION** | Independence is stronger than needed; zero correlation suffices. Correlated noise gives `alpha=(S+K)/(S+N+2K)`. |
| Common-bias non-identifiability | **VERIFIED UNDER THE STATED MODEL** | Exact observational equivalence under `(mu,b)->(mu+t,b-t)`. It is not a theorem about observables outside that model. |
| ReLU crossing lemma | **VERIFIED** | Five million randomized scalar cases plus edge cases; maximum numerical bound violation `1.37e-14`. |
| Particle replay linearization | **VERIFIED** | Coordinatewise bound follows by averaging the scalar lemma. |
| Universal unweighted `~5e-4` anchor threshold | **REFUTED AS A UNIVERSAL CLAIM** | Only the downstream-weighted ratio `E||Jxi||²/E||Jd||²` has an invariant break-even of one. Euclidean thresholds depend on direction and kink geometry. |
| M146 60-network numeric replication | **UNRESOLVED — SOURCE ARTIFACTS ABSENT** | Ledger source is a local transcript; network IDs, rows, perturbation vectors/distribution, scripts, reference construction, and manifest are absent. |
| Internal arithmetic consistency of M146 headline points | **PASSED** | Candidate/base ratio fits `q+a epsilon²` with `R²=0.9999965`; fitted break-even `5.7985e-4`. This is not replication. |
| T4 legal anchor family recovers stable absolute phase | **FAILED IN TESTED FAMILY** | Frozen policy is harmful and grouped rotation phase does not transfer. This supports, but does not prove universally, the phase-observability diagnosis. |
