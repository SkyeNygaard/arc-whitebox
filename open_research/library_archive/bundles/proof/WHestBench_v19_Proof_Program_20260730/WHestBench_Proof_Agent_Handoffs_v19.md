# WHestBench Proof-Agent Handoffs — Canonical v19

These prompts are designed for seven independent agents. Give each agent the v19 ledger, the canonical proof-status memo, and only the source files named in its prompt.

## Shared rules for every agent

- Work is **proof and mathematics first**. Do not begin with a generic experiment sweep.
- Do not open protected or official data.
- Treat the current estimator, arithmetic path, sampled-pilot tree, and generic learning tree as closed.
- Search actively for counterexamples before strengthening a claim.
- Preserve exact estimator-class scope: finite/infinite width, static/adaptive, signed/nonnegative, fixed/arbitrary support, linear/nonlinear, mass-one/free-mass, and cost assumptions.
- Do not assign a new canonical theorem ID. Use a descriptive local theorem name and propose a ledger patch; the coordinator assigns IDs.
- Any numerical theorem must include an independent directed-rounding implementation or a precise external verification gate.
- Required output files:
  1. `THEOREM_OR_COUNTEREXAMPLE.md`
  2. `HOSTILE_SELF_AUDIT.md`
  3. `COMPETITION_IMPLICATION.md`
  4. `PROPOSED_LEDGER_CHANGES.md`
  5. `DECISION.md`
  6. reproducible verifier code and machine-readable result files when numerical constants are used.
- `DECISION.md` must say one of: `PROVED`, `PROVED_UNDER_EXPLICIT_MODEL`, `COMPUTER_ASSISTED_CANDIDATE`, `COUNTEREXAMPLE_FOUND`, `BLOCKED_ON_NAMED_INPUT`, or `CLOSE_THIS_DIRECTION`.

---

# Prompt 1 — Optimize the Weighted Harmonic Signed Floor

You are the **weighted-rank cubature theorist**.

## Read first

- Canonical v19 ledger: T44, T45, T47, T54.
- `T47_WEIGHTED_HARMONIC_RANK_FLOOR.md`
- `ORACLE_CONTINUATION_HOSTILE_AUDIT.md`
- `REPORT(18).md`
- the frozen weighted-rank result JSON and verifier.
- the canonical K32 coefficient and T16 proof assets.

## Objective

Strengthen the global lower bound for static, network-independent, mass-one linear cubature with at most 66,048 arbitrary nodes and arbitrary real weights under the dimension-256, depth-32 limiting ReLU kernel.

The current certified floor is

\[
R(Q)\ge 0.505177125470747\,R_{\mathrm{Kerdock}}.
\]

Formulate the choice of harmonic weights \(a_\ell\), comparison kernel \(L_a^2\), and domination factor \(\gamma_a\) as a principled mathematical optimization problem rather than open-ended coefficient hunting.

## Primary targets

- Target A: certify a floor at least
  \[
  0.769230769\,R_{\mathrm{Kerdock}},
  \]
  which rules out every `1.30x` same-cost static signed improvement.
- Target B: certify at least
  \[
  0.909090909\,R_{\mathrm{Kerdock}},
  \]
  which rules out every `1.10x` same-cost static signed improvement.
- A valuable negative result is a dual upper bound showing that the current construction is near-optimal within a clearly declared finite-degree or diagonal-weight class.

## Mathematical tasks

1. Derive the finite-dimensional or infinite-dimensional primal optimization problem.
2. Derive a dual certificate or KKT system.
3. Prove the trace-constrained rank approximation formula for every rank \(r\le N\), including indefinite approximants.
4. Prove coefficientwise kernel domination in every active degree and control the tail.
5. Determine whether non-diagonal harmonic covariance, mixed feature blocks, or multiple squared kernels can improve the floor without losing the rank argument.
6. Freeze any numerical candidate before high-precision certification.
7. Reproduce the final constant with an implementation independent of both `mpmath.iv` and the current direct-MPFR code when feasible.

## Hostile checks

- Search for a rank-\(r<N\) optimum.
- Check every odd and even active degree.
- Check normalization of spherical harmonics and repeated eigenvalue multiplicities.
- Test whether a better abstract matrix approximation is unattainable by point-evaluation moment matrices; state which relaxation is used.
- Do not claim signed near-optimality unless the remaining factor is actually controlled.

## Competition output

Translate every floor into:
- maximum same-cost raw gain;
- maximum compute ratio that could still close the recorded adjusted gap;
- exact escape classes left open.

## Stop condition

Stop when either:
- a stronger independently certified floor is obtained;
- global optimality is proved in a useful declared certificate class; or
- a rigorous dual upper bound shows further work in the chosen class cannot materially improve the competition conclusion.

---

# Prompt 2 — Finite-Width Signed and Arbitrary-Node Certificate

You are the **finite-width kernel certifier**.

## Read first

- Canonical v19 ledger: T38, T47, T52.
- `T38_MINIMAL_CONDITION_AND_DEGENERATE_BOUNDARY.md`
- `T47_WEIGHTED_HARMONIC_RANK_FLOOR.md`
- the finite-width sections of `BEST_THEOREM_TARGETS_EXHAUSTIVE_CHECK_20260730.md`.
- the exact width-256, depth-32 network distribution and normalization specification.

## Objective

Move a strong static cubature theorem from the limiting kernel to the actual width-256 ensemble.

The first concrete goal is to obtain rigorous lower intervals for the finite-width normalized-Gegenbauer coefficients needed by the T47 rank schema. The stretch goal is an arbitrary-node finite-width Delsarte certificate analogous to T22.

## Mathematical tasks

1. Derive an exact finite-width ensemble-kernel representation from Gaussian first-layer noise stability and the later finite-width random network.
2. Isolate the low-degree coefficients required by the weighted-rank comparison.
3. Produce rigorous lower intervals, not Monte Carlo estimates.
4. Instantiate a finite-width signed rank floor:
   \[
   R_{K_{256}}(Q)\ge \gamma_{256}F_N(A).
   \]
5. Explore an architecture-specific Delsarte minorant for arbitrary nodes.
6. Use the uniform optimizer-transfer theorem only if a class-uniform kernel bound is genuinely small enough.
7. Explain why generic convergence, qualitative \(O(1/m)\) statements, or average kernel error are insufficient.

## Hostile checks

- Include the pure-quadratic boundary and small PSD perturbations that reverse tiny optimizer gaps.
- Verify that finite-width fixed-MUB support optimality does not imply arbitrary-node optimality.
- Check all width/depth normalization factors and post-ReLU versus pre-ReLU outputs.
- Do not import limiting coefficients as finite-width bounds.
- Distinguish ensemble MSE from a realized-network statement.

## Competition output

State whether the finite-width result rules out:
- equal-cost `1.10x`, `1.30x`, or `2x` static signed improvements;
- reduced-node variants at a specified compute ratio;
- any actual benchmark escape class.

## Stop condition

Stop with `BLOCKED_ON_NAMED_INPUT` if the proof reduces to a finite list of missing coefficient intervals or moments. The report must state those inputs precisely enough for a numerical-certification agent to execute.

---

# Prompt 3 — Literal-Transcript Phase Observability Bound

You are the **conditional observability and symmetry theorist**.

## Read first

- Canonical v19 ledger: T42, T43, T48, T49, T50.
- `WHESTBENCH_CONDITIONAL_OBSERVABILITY_THEOREM_PROGRAM_20260730.md`
- `ORACLE_PROOF_COMPLETION_REPORT.md`
- the exact source code/schema for one proposed legal runtime transcript.
- M153/M161 feature definitions and grouped results.

## Objective

Prove a nontrivial upper bound on the downstream correction value available to one **literal, fully specified runtime transcript**.

Do not prove a theorem about an informal phrase such as “same-design information.” Define the random instance, the oracle correction \(A\), downstream score operator \(J\), transcript \(Z\), and legal correction/action class exactly.

## Accepted proof routes

1. A measure-preserving involution preserving \(Z\) and approximately reversing \(JA\).
2. A compact-group action with a small invariant oracle component.
3. A conditional Haar or near-Haar relative-orientation theorem.
4. A phase-conditioned KL, TV, chi-square, or mutual-information upper bound.
5. A uniform population upper certificate for a frozen linear, finite, Lipschitz, bounded-degree, or compressed algorithm class.

## Required theorem form

Produce an explicit upper bound on one of:

\[
V_J(Z;A),
\qquad
\frac{V_J(Z;A)}{V_{\rm oracle}},
\qquad
\sup_{c\in\mathcal C}G(c).
\]

Translate it into a raw-MSE and adjusted-score ceiling.

## Hostile checks

- Condition on the realized integrand and selected rule when orientation is randomized.
- Handle whether \(J\) is observable; use the weighted normal equation when it is not.
- Do not infer low information from negative held-out \(R^2\).
- Full network weights determine the target in principle; the transcript must be genuinely restricted.
- Exhibit a counterexample outside the theorem assumptions.
- Check that the transformation preserves the full literal transcript, not only a prose summary.

## Competition output

The result is valuable only if it rules out a correction class that could plausibly pay for its compute. State the maximum raw gain and maximum allowable incremental compute implied by the theorem.

## Stop condition

Stop if no nontrivial symmetry or uniform certificate survives the exact transcript. Record the smallest transcript augmentation that destroys the obstruction; that becomes an input to Prompt 4.

---

# Prompt 4 — Canonical Orientation-Odd Gauge Fixing

You are the **equivariant gauge-fixing and constructive phase theorist**.

## Read first

- Canonical v19 ledger: T49, M160, M161.
- the gauge-obstruction proof and orientation-odd preregistration.
- exact downstream basis/sign conventions.
- available source-basis oracle-capacity summaries.

## Objective

Construct a legal, deterministic or randomized **orientation-odd observable** that makes signed correction coefficients well defined and stable under the relevant symmetry group.

T49 proves that the current even quotient features cannot output nonzero signed coefficients consistently. Your job is to characterize the minimal mathematical structure needed to escape that obstruction.

## Mathematical tasks

1. Define the acting group and the coefficient representation.
2. Define a canonical section or gauge choice.
3. Prove equivariance:
   \[
   a(g\cdot x)=\rho(g)a(x).
   \]
4. Prove uniqueness away from an explicitly characterized degeneracy set.
5. Bound sign instability under perturbations and finite precision.
6. Analyze whether a continuous global gauge is topologically impossible; if so, derive the best measurable/local alternative.
7. Bound the cost and information required to compute the anchor.
8. Prove how anchor error transfers into downstream correction risk.

## Source gate

No empirical policy may be run unless the frozen source basis has independent grouped oracle gain at least `1.20x` with safe tails. The theorem should explain why this is the minimum meaningful source condition.

## Hostile checks

- Zero-anchor and near-zero-anchor instability.
- Dependence on arbitrary basis ordering or sign conventions.
- Rotation/reflection leakage.
- An observable that is mathematically odd but uncorrelated with the target phase.
- Discontinuity that creates catastrophic tails.
- Cost exceeding the possible corrected MSE benefit.

## Competition output

Derive a complete threshold involving:
- source oracle capacity;
- transferred fraction;
- anchor instability;
- nonlinear replay margin;
- incremental compute.

## Stop condition

A clean no-go theorem for stable global gauge fixing is a successful result. A constructive theorem proceeds to a single frozen falsification protocol, not an architecture sweep.

---

# Prompt 5 — Exact Nonlinear ReLU Replay Certification

You are the **finite-width nonlinear replay analyst**.

## Read first

- Canonical v19 ledger: T42, T51.
- `RELU_REMAINDER_SHARP_THEOREM.md`
- the exact layer-31 correction/replay equations.
- authenticated preactivation and correction arrays only when their provenance is complete.

## Objective

Turn a linearized layer correction theorem into an exact post-ReLU final-MSE theorem, or prove that a declared correction class cannot retain enough linear benefit after gate crossings.

## Mathematical tasks

1. Define the preactivation \(Z\), perturbation \(T\), downstream operator(s), and exact corrected output.
2. Prove a conditional anti-concentration or crossing-probability bound on the actual crossing intervals.
3. Handle dependence between \(Z\) and \(T\).
4. Propagate the exact triangular ReLU remainder through downstream layers.
5. Replace a loose global operator norm when possible with direction- or covariance-weighted sensitivity.
6. Derive:
   \[
   R_{\rm exact}\le(\sqrt{R_{\rm lin}}+\delta)^2
   \]
   with a certified \(\delta\), or a sharper problem-specific inequality.
7. State an exact sufficient improvement criterion and an exact no-go criterion.

## Hostile checks

- Atoms at zero and heavy crossing mass.
- Large perturbations leaving the region where a local density bound holds.
- Anisotropic downstream sensitivity.
- Multiple-layer crossing accumulation.
- Random perturbations selected from the same transcript.
- Worst-network tails hidden by pooled averages.

## Competition output

For a named correction class, certify one of:
- exact raw and adjusted gain with margin;
- the maximum possible exact gain after nonlinear remainder;
- a proof that nonlinear replay uncertainty alone consumes all available headroom.

## Stop condition

Stop if the best rigorous remainder bound is too loose, but identify the exact missing anti-concentration or sensitivity quantity that would tighten it.

---

# Prompt 6 — Analytic-Plus-Residual Spectral Estimator Design

You are the **residual-kernel and harmonic variational theorist**.

## Read first

- Canonical v19 ledger: T35, T40, T44, T52.
- closed Stein, Poisson, harmonic, and projected-ReLU reports.
- the challenge score and full evaluation-cost model.

## Objective

Design a mathematically new exactly integrable surrogate \(g\) such that the residual \(h=f-g\) has a substantially easier kernel and can be integrated with far fewer evaluations or a different optimal design.

The estimator is

\[
I(g)+Q(f-g).
\]

Kerdock optimality does not transfer automatically; derive and certify the residual kernel.

## Mathematical tasks

1. Choose a tractable surrogate/operator class with exact \(I(g)\).
2. Derive its residual second-moment kernel exactly.
3. For equivariant linear operators, optimize spectral multipliers \(\tau_\ell\).
4. Consider richer structured operators only with a rigorous residual law.
5. Formulate a variational objective combining residual MSE, node count, surrogate cost, and nonlinear remainder.
6. Prove which harmonic degrees remain live.
7. Recompute MUB association values or construct a new Delsarte/rank certificate for the residual.
8. Determine whether the residual admits fewer nodes, cheaper evaluations, or a different support.
9. Include exact total compute, not only statistical risk.

## Hostile checks

- The surrogate changes only degrees 0–5 and therefore cannot affect complete-Kerdock error.
- Parameter fitting makes the residual candidate-dependent and destroys isotropy.
- Exact expectation is more expensive than the saved evaluations.
- The residual remains as hard as the original.
- Nonlinear replay invalidates a linear spectral argument.
- The design merely retunes a closed Stein/Poisson/projected-ReLU dictionary.

## Competition output

A positive theorem must identify a route to complete adjusted ratio below one with meaningful margin. A negative theorem should characterize the best attainable residual spectrum in the declared tractable class.

## Stop condition

Stop when the tractable surrogate class is solved variationally, whether the solution is positive or negative.

---

# Prompt 7 — Computational Lower Bound for White-Box Algorithms

You are the **computational complexity theorist for neural expectation estimation**.

## Read first

- Canonical v19 ledger: T32, T42, T43, A50.
- the low-degree Gaussian-chaos observability theorem.
- the exact challenge computation, query, memory, precision, and timing rules.
- the theorem showing full weights determine the target in principle.

## Objective

Define a realistic restricted model of white-box algorithms and prove a nontrivial lower bound connecting computational budget to MSE.

A universal information-theoretic impossibility theorem is invalid. The restriction must be computational.

## Candidate models

- bounded Gaussian-chaos or polynomial degree in network weights;
- bounded number of matrix-vector or network-evaluation queries;
- arithmetic circuits with depth/size/precision limits;
- statistical-query algorithms;
- rotation-equivariant or invariant circuits;
- streaming/memory-limited algorithms;
- finite-precision algorithms with explicit bit complexity.

Choose one model that contains a meaningful set of plausible competition algorithms.

## Mathematical tasks

1. Define the input, oracle/query access, output, cost and precision model.
2. Prove an exact projection theorem or lower bound within the model.
3. Relate circuit/query/degree budget to accessible harmonic or chaos components.
4. Derive an MSE lower bound at the competition budget.
5. Provide a matching or near-matching upper construction when possible.
6. Prove robustness to finite precision and randomization.
7. Identify which existing algorithms the model includes and excludes.

## Hostile checks

- The model secretly hides information already present in full weights.
- The model excludes the production Kerdock algorithm or every plausible adaptive method.
- Real arithmetic encodes unlimited information.
- Query lower bounds ignore preprocessing or shared matrix structure.
- The result is worst-case while the challenge is an average random ensemble, or vice versa.
- The lower bound has no numerical force at 272B operations / 30 seconds.

## Competition output

State the minimum risk or maximum correction value attainable at the actual budget, and compare it with:
- production;
- the T47 signed floor;
- the approximately `4.34x` adjusted winning gap.

## Stop condition

A rigorous demonstration that a proposed model is irrelevant is useful and should terminate that model. Do not weaken the model repeatedly until a theorem becomes easy but meaningless.
