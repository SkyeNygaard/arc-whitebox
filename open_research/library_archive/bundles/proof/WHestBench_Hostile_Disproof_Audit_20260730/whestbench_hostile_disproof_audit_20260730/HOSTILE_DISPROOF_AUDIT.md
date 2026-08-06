# WHestBench hostile disproof audit

**Date:** 2026-07-30  
**Disposition:** **CORE GEOMETRIC THEOREMS SURVIVE; FIVE ANCILLARY STATEMENTS ARE FALSE AS WRITTEN; RELEASE PACKAGE REQUIRES ERRATA**

## Executive verdict

I tried to falsify the proof portfolio by attacking quantifiers, degeneracies, conditioning, independence, endpoint logic, numerical precision, and certificate independence. The result is mixed but useful:

- I did **not** disprove the scoped T22 arbitrary-node nonnegative near-optimality theorem.
- I did **not** find a contradiction in the T16/T30 all-degree auxiliary optimizer, although its primal interval proof still lacks a genuinely independent second implementation.
- The T27/T37 strict-sign fixed-MUB allocation theorem survives.
- The **latest v17 theorem manuscript** already repairs the original T38 nondegeneracy defect and is substantially safer than the broader complete-proof compendium.
- Five statements elsewhere in the portfolio are false as written:
  1. T29 free-mass “every minimizer is scaled-uniform”;
  2. the original broad T38 implication from even nonconstancy alone;
  3. the conditional Haar no-value corollary under independence from runtime features alone;
  4. the assertion that independent replicas automatically have zero cross-error moment;
  5. the global cubic ReLU remainder bound from a density bound merely “near zero.”

These defects do not collapse the main static cubature result. They do require withdrawing or patching the broader `WHESTBENCH_COMPLETE_PROOF_PACKAGE_20260730.md` and correcting the round-two T29/T38 notes before external circulation.

## 1. False theorem wording: T29 free total mass

The old statement said every unconstrained minimizer is `alpha_*` times the uniform vector. The proof only establishes

\[
R(\alpha u+v)=R(\alpha u)+v^T Gv,
\qquad 1^Tv=0.
\]

This proves existence of a scalar-uniform minimizer, not uniqueness when `G` has a zero-sum nullspace.

### Exact counterexample

Let `Y(x)=Z` be a constant square-integrable random field with `E||Z||^2=1`. Then `G=11^T`, and every mass-one weight vector integrates `Y` exactly. Both

\[
(1/4,1/4,1/4,1/4)
\quad\text{and}\quad
(5/4,-3/4,1/4,1/4)
\]

have zero risk, but the second is nonuniform.

The correct free-mass minimizer set, when `E_X>0`, is

\[
\alpha_*u+(\ker G\cap1^\perp).
\]

Uniqueness requires positive definiteness on `1^perp`.

### Precision failure

The old note printed roughly 28 digits of `alpha_*` from two 16-digit decimal inputs. The rigorous archived enclosures instead imply

\[
\alpha_*\in
[0.9999997503247282806575775152106693,
 0.9999997503247282806578123186727384].
\]

The operational conclusion—negligible scale benefit—survives, but the old high-precision digits do not.

## 2. False theorem scope: original T38 assumptions

The original T38 assumptions allowed a general square-integrable `F_Z` and required only a nonconstant antipodally even output. The proof then inferred positive even Hermite mass at degree at least four.

That implication is false in the stated general class. Take

\[
F(g)=g_1^2-1.
\]

This function is square-integrable, even, and nonconstant, but its noise-stability kernel is pure degree two, proportional to `t^2`. Hence

\[
A=1,\quad O=0,\quad C=1/d,
\quad (A-O)+d(O-C)=0,
\]

not strictly positive. The all-lines-used and positive-basis-mass conclusions can degenerate.

The repair is to assume explicitly

\[
\sum_{r\ge2}a_{2r}>0,
\]

or to restrict to a nonconstant finite piecewise-affine ReLU realization and retain the proof that such a function cannot have only degree-zero and degree-two even Hermite content. The latest v17 theorem manuscript already uses this repaired assumption.

## 3. False conditional Haar corollary

The group-average identity

\[
\int_GQ_gf\,dg=I(f)
\]

is correct for every fixed integrand. The later conditional statement is not valid if it assumes only that `g` is independent of runtime information `G`, because the random integrand may depend on `g`.

### Exact counterexample

In dimension two, let `Q=delta_{e1}`, let `g` be Haar on `SO(2)`, set `h(x)=x_1^2`, and define

\[
f_g(x)=h(g^{-1}x).
\]

With the trivial runtime sigma-field, `g` is independent of all runtime information. Nevertheless,

\[
Q_gf_g=f_g(ge_1)=h(e_1)=1,
\qquad I(f_g)=1/2.
\]

The conditional mean error is `1/2`, not zero.

The correct assumption is

\[
\operatorname{Law}(g\mid f,\mathcal G)=\text{Haar},
\]

for example independence of `g` from both the integrand and the runtime information.

## 4. False independence corollary for replication

The formula

\[
E\left\|m^{-1}\sum_ie_i\right\|^2
=R_0\left(\rho+\frac{1-\rho}{m}\right)
\]

is correct under its stated uncentered cross-moment assumption. The claim “independent replicas imply `rho=0`” is false for biased estimators.

Take `e_i=b` deterministically. Degenerate random variables are mutually independent, but averaging preserves risk `||b||^2`. With linear compute, the adjusted ratio is `m`, not one.

The score-neutral conclusion requires independent **mean-zero** errors or, directly, pairwise uncorrelated errors.

## 5. False global reading of the ReLU cubic bound

The pointwise gate-crossing bound is correct:

\[
|r(z,t)|\le |t|1_{\{|z|\le|t|\}}.
\]

But a density bound merely “near zero” does not imply

\[
E[r^2\mid t]\le2L|t|^3
\]

for arbitrary `t`.

Take `z~Uniform[9,10]` and `t=-10`. The density is zero in a neighborhood of the origin, so `L=0` is a valid local bound, yet the ReLU remainder is `10-z` and

\[
E[r^2]=1/3>0.
\]

The density must be bounded over the whole interval `[-|t|,|t|]`, or `|t|` must be restricted to the radius of the local bound.

## 6. Definition and prose defects

### Observability ratio

`V_runtime/V_oracle` is undefined when both values are zero. Add `V_oracle>0` or a declared zero-capacity convention.

### T16 endpoint equality

The Hermite remainder proves strict positivity on noncontact points in `(-1,1)`. Continuity alone proves only endpoint nonnegativity. An independent 100-digit attack found robust positive residuals:

- at `t=1`: approximately `1.702189426837098e-2`;
- at `t=-1`: approximately `2.205187129080743e-7`.

Thus the equality-only-at-contacts claim survived, but its written endpoint justification was incomplete.

### T16 implementation independence

The Boost/C++ implementation independently checks the finite reduced-cost sweep and analytic tail cutoff. It does not independently implement the sixth-derivative interval argument, Hermite feasibility, or Krawczyk coefficient enclosure. T16 should remain computer-assisted with an explicit external primal-audit gate.

### Kernel perturbation

The `epsilon(1+B)^2` estimate is correct. Its optimizer-transfer corollary needs the same `B` bound uniformly over the complete comparison class, including a minimizing sequence.

### Theorem IDs

Short IDs have been reused for unrelated statements. This makes ledger claims non-immutable even when files are hashed. Adopt namespaced theorem identifiers or content hashes.

## 7. What survived hostile attack

### T22

The scoped one-sided theorem survived. The local theorem and manifest verifiers pass, and the separate direct-C GMP/MPFR engine reports reproduction of all 1,421 interval leaves, global minorant, spherical mean, Kerdock energy, Delsarte bound, and final ratio. No counterexample was found inside the stated static, nonnegative, mass-one, limiting-kernel class.

### T16/T30

The analytic structure, exact reduced costs, Hermite construction, positive coefficients, primal-dual equality, and endpoint checks are mutually consistent. I found no mathematical counterexample. The remaining concern is independent reproduction of the primal interval stack, not an observed contradiction.

### T27/T37

The association-scheme reduction and convex allocation proof survive under

\[
A-O>0,\quad O-C<0,\quad (A-O)+d(O-C)>0.
\]

A complete brute-force count-pattern check for a small strict-sign example agreed for every feasible budget.

### Corrected T38

With explicit high-even Hermite nondegeneracy, the finite-width fixed-support extension survives. The latest theorem manuscript states this corrected version.

### T39/T40 and standard Hilbert identities

The exact symmetry-projection theorem, residual spectral multiplier identity, conditional-projection theorem, common-bias lower bound, correction-risk identity, and subspace-replacement formulas survive inside their explicit models. Their application to the actual legal WHestBench observation map remains model-specific.

### Exact controls

The uniform-nullspace identity survives. The Poisson mean-one identity survived an independent numerical integration check, and the projected-ReLU spherical mean formula is consistent with direct integration.

## 8. Release decision

The narrow v17 theorem manuscript remains viable after small wording changes and external review. The broader complete proof package should not be circulated in its old form because it contains the false Haar, replication, and ReLU-density statements. The round-two T29 and original T38 files should be retained only as superseded artifacts beside their corrected versions.

**Recommended status:**

- `whestbench_theorem_manuscript_v17.md`: **RETAIN WITH MINOR ERRATA AND EXTERNAL REVIEW**.
- `WHESTBENCH_COMPLETE_PROOF_PACKAGE_20260730.md`: **SUPERSEDE BY HOSTILE-PATCHED EDITION**.
- original `T29_ALL_WIDTH_FIXED_LINEAR_THEOREM.md`: **SUPERSEDE**.
- original `T38_FINITE_WIDTH_KERDOCK_LINE_THEOREM.md`: **SUPERSEDE BY HIGH-EVEN VERSION**.
- project closeout status “internal hostile review complete”: **REOPENED BY THIS AUDIT UNTIL PATCHES ARE MERGED**.
