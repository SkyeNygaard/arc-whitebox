# Required patches after hostile disproof audit

## Must change before external circulation

1. **T29 free-mass uniqueness**
   - Replace “every minimizer is alpha-scaled uniform” by the full affine minimizer set.
   - Add the zero-sum positive-definiteness condition for uniqueness.
   - Replace rounded-input scale digits with the rigorous interval in `T29_ATTACK_RESULTS.json`.

2. **Original T38 assumptions**
   - Add positive even Hermite mass at degree at least four, or restrict explicitly to the finite piecewise-affine ReLU case and prove that implication.
   - State the feasible budget range `1 <= P <= Md`.

3. **Haar orientation theorem**
   - The unconditional group-average identity is correct.
   - The conditional no-value corollary must require `Law(g | f,G)=Haar`; independence from `G` alone is insufficient.

4. **Replication corollary**
   - Replace “independent errors imply rho=0” by “mean-zero independent or pairwise uncorrelated errors imply rho=0.”

5. **ReLU cubic remainder bound**
   - Replace “density bounded near zero” by a density bound over the full crossing interval `[-|t|,|t|]`, or restrict `|t|` to the radius on which the local density bound holds.

6. **Observability ratio**
   - Require positive oracle value or define the zero-capacity convention; otherwise the ratio can be `0/0`.

## Should change

7. **T16 endpoint prose**
   - Continuity proves endpoint nonnegativity, not strict endpoint positivity. Add the explicit endpoint residual checks from `T16_T22_ATTACK_RESULTS.json`.

8. **T16 trust-base wording**
   - Say clearly that the independent C++ implementation covers reduced costs/tail, not the sixth-derivative, Hermite feasibility, or Krawczyk coefficient certificate.

9. **Kernel perturbation transfer**
   - State that the total-variation bound `B` applies uniformly to the whole comparison class, including a minimizing sequence.

10. **Optimizer-instability example**
    - Call the displayed result a pairwise ranking reversal unless the admissible rule class is explicitly restricted or all other rules are controlled.

11. **Theorem identifiers**
    - Stop reusing short IDs. T29 and T41 have already denoted different claims in different packages. Use immutable namespaced IDs or content hashes.
