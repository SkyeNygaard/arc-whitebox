# Revised Cascade / Observability Program

## Decision

The original program cannot produce its proposed impossibility theorem because its Gaussian Bayes/no-adaptation route does not apply to the post-ReLU output process. The corrected program separates a rigorous **linear-estimator theorem track** from an empirical **nonlinear/cascade discovery track**.

## Track A — rigorous static linear boundary

1. **Complete-support symmetry theorem at every width.**  
   Prove uniform mass-one weights minimize ensemble MSE for any zonal second-moment kernel on the complete Kerdock support. State nonuniqueness caveat and the unconstrained alpha-scaled corollary.

2. **T22.**  
   Retain the certified infinite-width bound over fixed, network-independent, nonnegative linear rules with at most 66,048 nodes.

3. **T27.**  
   Retain exact optimization over arbitrary real weights and supports inside the Kerdock line universe.

4. **Off-support signed stability.**  
   Publish the certified `M` curve but state that it is quantitatively weak.

5. **No nonlinear/adaptive extrapolation.**  
   Do not call the post-ReLU field Gaussian. Do not import Gaussian no-adaptation theorems unless the estimand/model is changed to a genuinely Gaussian linear-output problem.

## Track B — empirical nonlinear and cascade corrections

Define a preregistered function class `F` and measure

```
gamma_F = max_(d in F) corr(e,d)^2.
```

This is a class-specific achievable lower bound, not an upper bound on all S2.

### Required evaluation protocol

- split by base network;
- keep all rotations together;
- immutable feature dictionary and hyperparameter grid;
- independent target halves and noise correction;
- report downstream correction MSE, cosine, norm, wins, p90, worst, CVaR, compute, and adjusted score;
- a failed class closes only that class;
- an observed alignment is not enough: require safe tails and complete deployment economics.

### Current class decisions

- T4 target-free geometry dictionary: **closed** by grouped negative OOF predictability.
- Full/compact companion directions: **real average signal, not deployable** because cost/tails.
- Sparse radial-Hermite correction: **real average signal, not deployable** because worst tail and incomplete oracle capture.
- Legal full-depth scalar recurrence: **closed tested form** near neutral.
- Broad “all trajectory observability is absent”: **open and unproved**.

### Only justified reopening conditions

1. A genuinely new external absolute-phase observable.
2. A tail predictor that replicates by base network on a fresh immutable cohort.
3. A correction retaining material raw gain after all expectation-source and replay costs.
4. A theorem for a precisely restricted nonlinear family, not an extrapolation from failed regressors.

## Paper structure

1. Problem and radial reduction.
2. Static linear kernel risk.
3. T22 certified global nonnegative boundary.
4. All-width complete-support symmetry theorem.
5. T27 Kerdock-line real-weight theorem.
6. Exact correction-risk and anchor identities.
7. Oracle-depth repairability.
8. Stratified falsification map: signal vs observability vs safety vs economics.
9. Why Gaussian Bayes/no-adaptation does not apply to the post-ReLU output.
10. Open nonlinear/adaptive problem and explicit reopening criteria.
