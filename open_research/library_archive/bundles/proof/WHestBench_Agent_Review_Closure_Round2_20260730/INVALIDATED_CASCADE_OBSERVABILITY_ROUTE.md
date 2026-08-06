# Superseded route — Gaussian Bayes/no-adaptation and universal cascade impossibility

## Decision

**INVALIDATED FOR THE ACTUAL POST-RELU OUTPUT.**

Earlier cascade blueprints attempted to combine the infinite-width kernel certificate, Gaussian posterior-mean optimality, and no-benefit-from-adaptation results into a theorem against adaptive, nonlinear, or trajectory-using estimators. That implication is not valid for the actual challenge output.

## Failure 1 — the observed output is not Gaussian

Even when a preactivation field is Gaussian in the infinite-width limit, applying the final ReLU produces a non-Gaussian field, including an atom at zero. Kernel discrepancy determines the risk of fixed linear rules from second moments. It does not identify the conditional expectation among all nonlinear algorithms.

## Failure 2 — explicit nonlinear ReLU counterexample

Let

\[
f_a(u)=\operatorname{ReLU}(a^Tu),\qquad u\in S^{d-1}.
\]

Its exact spherical integral is

\[
I(f_a)=c_d\|a\|_2,
\qquad
c_d={\Gamma(d/2)\over2\sqrt\pi\,\Gamma((d+1)/2)}.
\]

Observe `f_a` on an antipodal orthonormal basis. For every coordinate,

\[
f_a(e_i)+f_a(-e_i)=|a_i|.
\]

A nonlinear estimator therefore reconstructs

\[
\|a\|_2=\sqrt{\sum_i|a_i|^2}
\]

and recovers the integral exactly. Equal-weight linear averaging gives instead

\[
{1\over2d}\sum_i|a_i|,
\]

which is generally different from `c_d||a||_2`.

Thus the same evaluations can support an exact nonlinear estimator while the optimal fixed mass-one linear average remains imperfect.

## Failure 3 — finite model screens do not bound all runtime information

Negative grouped cross-validation for a frozen feature dictionary closes that dictionary. It cannot upper-bound

\[
\sup_{d\in L^2(\mathcal G)}
{\mathbb E\langle e,d\rangle^2\over
 \mathbb E\|e\|^2\,\mathbb E\|d\|^2},
\]

over every measurable correction available from the runtime sigma-field.

## Failure 4 — width corrections are not exploitability decompositions

A statement that finite-width covariance differs from the limiting kernel by order `depth/width` does not show that the whole difference is observable or correctable. It is a second-moment approximation, not a decomposition of algorithmically accessible gain.

## Surviving statements

The following remain valid:

- T22: scoped arbitrary-node nonnegative near-optimality for the infinite-width kernel;
- T29: uniform fixed linear weights on complete support at every width;
- T38: fixed-MUB-line real-weight optimum at every finite width under the Gaussian first-layer model;
- T33: the best correction measurable from information `G` is the conditional expectation projection;
- T39: a precisely invariant observation class can recover only the invariant error component;
- empirical failure of named feature dictionaries and estimator families.

## Required paper correction

Do not state that Kerdock is the posterior mean, that adaptive information cannot help the actual output, or that all nonlinear/cascaded corrections have zero value. The defensible synthesis is a static-linear geometric boundary plus class-specific information theorems and a frozen empirical falsification map.
