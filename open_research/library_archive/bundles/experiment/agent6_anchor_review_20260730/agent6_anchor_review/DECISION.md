# Agent 6 decision

## 6A — theorem formalizer

**VERIFIED AFTER SPECIFIED CORRECTIONS.**

The core theory is valid and paper-worthy:

- exact correction-risk identity;
- selector value as prediction of signed error-correction inner product;
- full replacement threshold in a stated correction subspace;
- common-bias non-identifiability;
- exact ReLU gate-crossing remainder.

Before publication, incorporate four corrections:

1. distinguish unrestricted, positive-only, and bounded selector scales;
2. state the general replacement cross term when anchor error leaves the subspace;
3. replace “independent noise” by the correlated-noise shrinkage theorem;
4. include an exact nonlinear remainder margin and define relative error in the `J`-weighted scored norm.

## 6B — adversarial structured-error reviewer

**M146 REPLICATION BLOCKED BY MISSING ORIGINAL ARTIFACTS.**

The shared archive does not contain the 60-network rows, exact means, perturbation vectors/distribution, scripts, IDs, references, or manifest. The numeric result must remain provisional.

The reported curve is internally coherent: a quadratic-noise model fits with `R²=0.9999965` and predicts break-even `5.80e-4`. This increases confidence that the four headline numbers came from one consistent calculation, but it is not evidence about cohort integrity, direction robustness, leakage, tails, or exact replay.

The adversarial synthetic audit decisively rejects a universal unweighted `5e-4` threshold. Equal-norm errors can be harmful in leading downstream directions and nearly inert in trailing directions; ReLU kink concentration can enlarge nonlinear error by orders of magnitude. The universal theorem-level threshold is instead `eta_J<1`, augmented by a measured nonlinear remainder margin.

## Research decision

- **Accept** the corrected theorem package for the paper.
- **Do not cite M146 as frozen empirical evidence** until its original package is restored and independently rerun.
- **Do not reopen broad anchor search** on the basis of another estimator with the same centered/common-bias information.
- **Permit one exact reproduction only:** restore M146 and run the preregistered structured-direction matrix. Stop afterward unless a legal source passes downstream-weighted exact replay with safe tails and cost.
