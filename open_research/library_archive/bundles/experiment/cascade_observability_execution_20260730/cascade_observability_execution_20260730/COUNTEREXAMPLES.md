# Counterexamples and Adversarial Findings

## 1. Nonlinear estimator beats the fixed linear rule on a ReLU family

For `f_a(u)=ReLU(a^T u)` and one antipodal orthonormal basis,

```
f_a(e_i)+f_a(-e_i)=|a_i|.
```

Nonlinear L2 aggregation recovers `||a||` and therefore the exact spherical integral. Equal-weight linear aggregation uses an L1 statistic and is not exact. This refutes the inference that wide Gaussian preactivations force linear Bayes optimality after ReLU.

## 2. Failed learner is not an exploitability upper bound

Let `gamma=sup_(d measurable from S2) corr(e,d)^2`. A failed finite feature model says only that one chosen function class failed. There can be another measurable correction outside the class. Therefore TEST-2's proposed PASS implication is invalid by definition.

## 3. Nonzero legal alignment exists

Archived finite-width S2 methods include correction cosines around 0.40, 0.49, and 0.61. They fail deployment for tails, cost, or completeness, not because all alignment is zero. This refutes the pooled zero-alignment narrative.

## 4. Oracle ladder is not strictly monotone

The archived operational layer-swap curve has early-layer reversals. Thus monotonicity cannot be used as a universal sanity requirement without a more precise intervention definition.

## 5. Signed certificate is rigorous but weak

The certified `M` permits material improvements with extremely small total negative mass according to the lower bound, so it cannot close off-support signed rules at competition-relevant scales.
