# Shared instructions for every agent

You are working on the WHestBench competition. Your objective is **competitive progress**, not merely interesting mathematics.

Before beginning:

1. Read the latest canonical ledger:
   `whestbench_canonical_research_ledger_20260730_reconciled_v25_full_state_conflict_audit.xlsx`
2. Read:

   * `WHestBench_Current_State_v25.md`
   * `WHestBench_Best_Path_Continuation_v23.md`
   * the latest relevant agent reports and source snapshots.
3. Treat the ledger as a fallible index, not ground truth. Verify important claims against their source artifacts.
4. Do not spend substantial effort improving static, network-independent quadrature. That class is already constrained enough to be strategically irrelevant.
5. Do not train another generic predictor before demonstrating that its source span has sufficient oracle capacity.
6. Keep these concepts separate:

   * full-checkpoint oracle capacity;
   * capacity of a legal target-free source;
   * observability of its coefficients;
   * coefficient estimation noise and bias;
   * computation cost;
   * exact nonlinear replay;
   * adjusted competition score.
7. Do not open protected competition data unless a protocol, source, estimator, compute accounting and promotion gate are frozen in advance.
8. Alternate between constructive work and hostile attempts to disprove your own conclusions.
9. Use local computation aggressively when it can settle a mathematical or structural question.
10. Clearly label every result as:

    * proved;
    * computer-assisted proof;
    * conditional theorem;
    * numerical discovery;
    * oracle diagnostic;
    * deployable experiment;
    * speculation.

The current leading program is to construct a network-covariant rank-4 or rank-5 correction at the last hidden checkpoint, estimate only a few scalar contractions, and replay the final layer exactly.

For a whitened source with oracle residual ratio (r_*), coefficient-estimation difficulty must be evaluated using the full source–noise–compute frontier. Under the simplified independent unbiased model,

[
\text{best adjusted ratio}
==========================

\left(
\sqrt{r_*}
+
\sum_j\sqrt{v_j\gamma_j}
\right)^2.
]

Do not use a universal `0.20–0.22` oracle gate. A source near `0.20` leaves very little room for estimation noise or added compute. Prefer sources comfortably below approximately `0.15`, unless their contractions are essentially exact and free.

For every completed task, produce:

* a detailed Markdown report;
* reproducible scripts;
* machine-readable results;
* hashes or a manifest;
* a proposed ledger patch using provisional IDs;
* a section titled **Claims I tried to disprove**;
* a section titled **Conflicts with existing ledger entries**;
* an explicit recommendation: continue, conditionally continue, stop, or quarantine.

---

# Agent 1 — Physical late-interface source discovery tournament

Your job is to find a **legal, network-covariant rank-4 or rank-5 source at the final hidden checkpoint** with enough oracle capacity to support a winning estimator.

The full checkpoint oracle is not the desired result. You need a small source whose columns can be constructed from the realized network and permitted baseline computation without using the unknown integration target.

Explore sources derived from physically meaningful depth channels, including:

* early-, middle-, late- and final-layer defect proxies;
* transported ReLU absolute-value innovations;
* checkpoint differences;
* accumulated gate imbalance;
* downstream-weighted but legally measurable prefix quantities;
* banded adjoint potentials;
* causal interventions propagated to the final hidden checkpoint;
* controllability or observability directions derived from the network;
* cross-layer commutators or residuals;
* combinations of independently constructed checkpoint channels.

Do not default to generic PCA, SVD of target defects, or a basis selected using the oracle target. Oracle information may be used only to evaluate a frozen source construction.

For each candidate source:

1. Define its construction precisely.
2. Verify covariance or invariance under neuron permutations, sign symmetries and other relevant reparameterizations.
3. Replay each column through the true suffix or exact final-layer map.
4. Whiten it in the physical final-output Gram metric.
5. Measure:

   * development, validation and confirmation oracle residual ratio;
   * marginal value of each added channel;
   * effective rank;
   * conditioning;
   * stability across networks and rotations;
   * worst-case behavior.
6. Compare a rank-4 and rank-5 version.
7. Test whether the source is genuinely multi-channel or whether one channel dominates.
8. Test whether the apparent capacity depends on target-informed orientation, scaling or source selection.

Use a tournament format. Begin with many cheap candidate definitions, eliminate weak spans quickly, then run expensive exact replay only on survivors.

Primary success gate:

* confirmation oracle ratio preferably below `0.15`;
* no catastrophic network;
* construction completely target-free;
* source identity stable under equivalent network parameterizations.

Secondary target:

* determine whether the large layer-29 and layer-30 full-checkpoint oracle capacity can be compressed into a few physical channels at all.

A negative result is valuable if it proves that a broad natural family of physical late-interface sources cannot retain enough capacity.

Do not perform coefficient learning. Your entire task is to determine whether a winning-capacity legal source exists.

---

# Agent 2 — Exact scalar contraction identity hunter

Assume a rank-4 or rank-5 source (A) has been frozen at the late interface. Your job is to derive **absolute, target-free identities for the scalar normal-equation quantities**

[
b=A^\top e.
]

The existing adjoint-potential identity reduces the dimensionality but does not solve observability: evaluating the complete potential on the same Kerdock cloud is algebraically equivalent to the original output projection.

Search for identities that create genuinely new information.

Explore:

* depth-banded adjoint potentials;
* telescoping identities between checkpoints;
* conditional expectations over selected Gaussian layers;
* Stein and Gaussian integration-by-parts formulas;
* leave-one-layer-out identities;
* leave-one-row and leave-one-neuron formulas;
* martingale difference decompositions across depth;
* Rao–Blackwellization over unused network randomness;
* exact expectations of low-dimensional contractions;
* symmetry cancellations;
* antithetic network or node transformations;
* identities based on preactivation sign, magnitude and crossing statistics;
* reuse of quantities already computed by the baseline;
* analytic integration over one layer or one row at a time;
* conditional noise-stability representations.

For each candidate identity, explicitly answer:

1. Is it absolute, or does it still contain the unknown target?
2. Does it use information already implicit in the original estimator?
3. Is it exactly computable, unbiasedly estimable, or only approximate?
4. Does it require evaluating the full network again?
5. Does it depend on suffix weights in a way that breaks the intended independence theorem?
6. What are its bias, variance, covariance and computational cost?
7. Does it estimate the signed contraction, or only its magnitude?
8. Is orientation well-defined and stable?

Try to prove no-free-lunch results for attractive but circular identities. A failed identity should be reduced algebraically until the hidden copy of the original integration problem becomes obvious.

Prioritize identities that estimate one physical depth-band contribution much more cheaply than a full output evaluation. It is acceptable if no single band suffices, provided several low-cost band estimators combine favorably.

For any proposed estimator, calculate or bound its contribution to the full T72 economics. Do not report correlation alone.

The ideal output is one of:

* an exact analytic formula for one or more source contractions;
* an unbiased estimator with unusually low variance and cost;
* a provable control variate that reduces contraction variance;
* a theorem ruling out a broad family of circular contraction identities.

---

# Agent 3 — Conditional expectation and Gaussian integration specialist

Attack the coefficient-observability problem by integrating out parts of the realized random network exactly or approximately.

Choose one or more layers, rows, neurons or blocks whose randomness is conditionally Gaussian given the rest of the network. Seek conditional expectations that convert a difficult target contraction into a deterministic function of cheaper prefix information.

Potential approaches include:

* condition on the prefix and integrate the final row;
* integrate a suffix row-by-row;
* leave-one-row resampling;
* Gaussian integration by parts in a selected weight matrix;
* Ornstein–Uhlenbeck interpolation;
* conditional ReLU moments;
* exact bivariate or multivariate truncated-Gaussian formulas;
* conditional score functions;
* control variates generated from analytically integrated suffixes;
* randomized unbiased estimators using Russian roulette or multilevel truncation;
* antithetic resampling of a small suffix;
* partial Rao–Blackwellization.

Your main question is:

> Can a source contraction be estimated from a small number of analytically integrated or cheaply resampled network components, without another full 66,048-node network evaluation?

Develop exact formulas first. Then test them numerically against local oracle contractions.

For each construction:

1. Specify which randomness is conditioned on and which is integrated.
2. Prove measurability and independence claims.
3. Calculate the exact conditional mean.
4. Derive variance or a rigorous upper bound.
5. Include covariance between multiple contraction estimators.
6. Calculate actual additional computation.
7. Compare direct Monte Carlo, antithetic, stratified and conditional variants.
8. Check whether the estimator is still useful after fixed-network conditioning rather than only in ensemble expectation.
9. Test whether the conditional estimator preserves the sign and scale of the contraction across networks.
10. Insert the result into the full adjusted-score calculation.

Explore both exact and approximate conditional integration, but approximate formulas must include a signed remainder or empirical bias audit.

A useful negative result would show that integrating any bounded number of suffix rows cannot provide enough information, thereby forcing attention to prefix observables or a different source.

Do not assume the full target is replayable from a single mean checkpoint state. Verify the representation explicitly.

---

# Agent 4 — Replayability, replacement bias and nonlinear transfer theorem

Your task is to determine whether the desired late-interface correction can actually be represented and replayed legally.

Several related claims must not be conflated:

* approximating a checkpoint state;
* approximating the mean of checkpoint states;
* shifting every baseline particle by a common vector;
* replacing an integration distribution by one checkpoint vector;
* reproducing the final integration correction;
* controlling error through an independent Gaussian suffix.

Begin from the exact benchmark computation and define the physical target precisely.

Investigate:

1. Whether the true output correction is exactly representable as replay of:

   * a common shift of the final hidden cloud;
   * a low-rank affine transformation;
   * a mixture of a few shifted clouds;
   * per-particle shifts in a structured subspace;
   * corrected checkpoint moments.
2. The replacement bias introduced by each representation.
3. Whether the rank-4/5 checkpoint repair modes can be compressed into one late-interface state correction.
4. Whether exact final-layer compiled replay matches production indexing, normalization and architecture.
5. Whether the source basis or coefficients depend on suffix weights.
6. Whether sample splitting, leave-one-row arguments or conditional variants restore legality.
7. Whether the expected nonexpansivity theorem can be strengthened to:

   * high probability;
   * fixed-network average;
   * row-wise conditional;
   * deterministic Lipschitz or operator bounds.
8. How replay error combines with coefficient error and source approximation error.

Construct explicit counterexamples whenever a proposed replay theorem is too strong.

Produce an error decomposition of the form:

[
\text{final error}
==================

\text{source-span error}
+
\text{coefficient error}
+
\text{replacement bias}
+
\text{nonlinear replay error}
+
\text{dependence penalty}.
]

The decomposition should be exact when possible and otherwise have rigorously controlled remainder terms.

Implement a verifier comparing:

* full suffix replay;
* compiled final-layer replay;
* linearized replay;
* checkpoint replacement;
* common shift;
* any proposed low-rank nonlinear replay.

Success means either:

* a complete legal replay theorem for the intended correction object; or
* a decisive proof that the current late-interface formulation has unavoidable replacement bias, together with the minimal richer state required.

---

# Agent 5 — Network-covariant geometry and controllability analysis

Approach source construction as a geometric systems problem rather than a feature-engineering problem.

Treat perturbations introduced at different depths as inputs to a nonlinear dynamical system. Determine which low-dimensional combinations are both:

* strongly controllable at the final output; and
* observable from legal network quantities.

Explore:

* empirical controllability and observability Gramians;
* nonlinear balanced truncation;
* tangent and secant spaces;
* checkpoint innovation operators;
* Krylov spaces generated by layerwise Jacobians;
* Lie brackets or commutators of depth-local perturbations;
* low-rank Hankel operators between prefix observables and final corrections;
* canonical correlation between legal transcripts and oracle correction channels;
* equivariant representation theory under neuron permutations;
* transport of physical modes through the network;
* network-specific bases derived from operator pencils rather than target PCA.

Avoid generic global PCA of oracle defects. The basis must be constructible from the realized network and baseline transcript.

Questions to answer:

1. Why is the measured repair dimension approximately four or five?
2. Are these dimensions associated with stable physical depth bands?
3. Do the same abstract channels exist across networks, even if their coordinates rotate?
4. Is there a canonical gauge or transport rule aligning them?
5. Is the late-interface image of the depth-band source well-conditioned?
6. Does the source collapse under exact nonlinear replay?
7. Can controllability be large while observability remains negligible?
8. Is there a rank lower bound showing that four or five channels are insufficient?
9. Can a slightly larger source, such as rank 8 or 12, materially improve economics despite more coefficients?
10. Is there a minimal sufficient source dimension under the actual score function?

Use oracle targets to evaluate capacity only after the construction is frozen.

Try to derive structural theorems explaining both success and failure. In particular, distinguish:

* low-dimensional repair energy;
* low-dimensional coordinate support;
* low-dimensional legal observability;
* low-dimensional downstream action.

The desired result is a physically interpretable basis construction with excellent oracle capacity, or a no-go theorem showing that the observed rank-4/5 repair cannot be canonically extracted from available network information.

---

# Agent 6 — Score economics, optimal experimental design and estimator allocation

Take candidate source constructions and determine whether they can possibly win after estimation noise and compute are included.

Do not design source geometry. Your role is to build the exact decision framework that tells other agents which sources and contraction estimators deserve further work.

Generalize the simplified T72 model to include:

* biased estimators;
* correlated coefficient errors;
* non-orthogonal source columns;
* random compute;
* shared samples between contractions;
* common control variates;
* unequal source value;
* nonlinear replay;
* clipping and shrinkage;
* integer sample allocations;
* fixed setup costs;
* reuse of baseline computations;
* tail risk and worst-network constraints.

Derive the optimal allocation problem in the physical Gram metric.

For candidate estimators, estimate or bound:

* mean bias vector;
* full covariance matrix;
* cost covariance if relevant;
* source Gram matrix;
* oracle coefficient distribution;
* expected adjusted score;
* network-bootstrap uncertainty;
* worst-network degradation.

Explore optimal design methods:

* generalized least squares;
* common-random-number coupling;
* multilevel Monte Carlo;
* control variate allocation;
* adaptive stopping;
* sequential allocation among contractions;
* sparse coefficient selection;
* source truncation;
* shrinkage toward a global action;
* robust optimization over uncertainty sets.

Produce hard promotion thresholds, not vague rankings.

For example:

* minimum source ratio required for a measured covariance-cost profile;
* maximum allowable bias norm;
* when adding a fifth channel helps or hurts;
* when shared sampling dominates independent sampling;
* how much source capacity is worth sacrificing for cheaper contractions;
* whether a rank-8 source can outperform rank 4 after optimal allocation;
* whether a tiny residual learner can ever pay for itself.

Build a reusable script that consumes source and estimator measurements and outputs:

* optimal allocation;
* expected score;
* uncertainty interval;
* sensitivity analysis;
* pass/fail decision;
* dominant bottleneck.

Your work should kill bad ideas early. It is a success if you prove that several apparently strong oracle sources cannot win under any plausible estimator economics.

---

# Agent 7 — Exactly integrable surrogate and residual-kernel redesign

Search for a network-dependent surrogate (g_W(x)) whose spherical expectation is exact or very cheap, while the residual

[
f_W(x)-g_W(x)
]

is substantially easier for Kerdock or another certified rule to integrate.

This is different from directly correcting the Kerdock output. You may redesign the decomposition of the integration problem.

Candidate surrogate families include:

* locally linear or piecewise-linear network approximations;
* conditional expectations over selected weights or gates;
* low-degree Hermite or spherical-harmonic projections with exact coefficients;
* shallow network surrogates;
* frozen-gate models;
* Gaussianized suffixes;
* tractable radial models;
* mixtures indexed by depth bands;
* exact one-layer or two-layer integrations;
* analytic control variates;
* pathwise Taylor models with certified remainders;
* low-rank covariance or cumulant surrogates only when they show final-output value;
* surrogate networks sharing most baseline computation.

For each surrogate:

1. Show how (\mathbb E[g_W]) is obtained legally.
2. Measure the residual oracle MSE under the exact same evaluation budget.
3. Analyze the residual harmonic or chaos spectrum.
4. Determine whether existing quadrature certificates transfer.
5. Measure the computation needed for (g_W), the residual and any corrections.
6. Test final-output performance, not merely intermediate-state accuracy.
7. Audit replacement bias and nonlinear accumulation.
8. Compare to the late-interface source approach.
9. Identify whether the construction is simply another form of the original estimator.
10. Attempt to prove a residual norm or spectral-tail bound.

The central success criterion is not that (g_W) approximates (f_W) pointwise. It is that the adjusted score of

[
\mathbb E[g_W]+Q(f_W-g_W)
]

can plausibly beat the competition threshold.

Past Edgeworth work showed that excellent one-step moment accuracy can be irrelevant at the final output. Require an early final-output oracle ceiling before expensive implementation.

A useful outcome may be a theorem that a broad surrogate family cannot alter the residual spectrum enough, or one unexpected decomposition that exposes a nearly exact analytic band.

---

# Agent 8 — White-box computational discovery of new legal observables

Use the actual local competition data and realized network weights to search for **new observables**, not merely new regression models over old features.

Your goal is to discover network-computable scalar or low-dimensional quantities that predict the signed source contractions across networks and rotations.

Start with a frozen high-capacity source from Agent 1, or construct several provisional physical sources solely for discovery.

Generate observables from:

* layerwise gate imbalance;
* signed preactivation margins;
* near-zero gate populations;
* transported absolute activations;
* products of prefix and suffix sensitivities;
* row-wise leave-one-out effects;
* path norm imbalances;
* depth-local Jacobian traces;
* cross-layer Gram differences;
* Stein scores;
* finite-difference responses to controlled perturbations;
* antithetic weight or input transforms;
* checkpoint martingale increments;
* low-cost random projections of large interfaces;
* compressed sketches of the final-layer margin distribution;
* quantities derived from exact compiled replay curves;
* derivatives of the Kerdock estimate under legal network perturbations.

The discovery process must be hypothesis-generating rather than an unrestricted black-box learner.

Protocol:

1. Generate candidate observables on development networks.
2. Cluster them by mathematical mechanism and redundancy.
3. Test univariate signed value before multivariate models.
4. Require grouped whole-network validation.
5. Compare against a fixed global action and zero-residual baseline.
6. Measure incremental conditional projection in the physical source metric.
7. Test sign stability, not just correlation.
8. Test rotations separately.
9. Use confirmation only after the observable definition and model class are frozen.
10. Feed promising observables back into Agent 2 for identity derivation.

Try symbolic regression or program synthesis only if expressions remain interpretable and auditable. Penalize complexity heavily.

The desired discovery is an observable with enough stable signed information that a mathematician can plausibly derive why it works.

Do not conclude that a feature is useful from pooled correlation, development gain or target-informed orientation. It must improve grouped out-of-network correction value over the global baseline.

Produce a ranked catalog of mechanisms, including null results and instability modes.

---

# Agent 9 — Information-theoretic obstruction and hostile falsification

Your job is to determine whether the proposed rank-4/5 correction program is actually possible.

Act adversarially toward Agents 1–8.

Investigate whether legal transcripts contain enough information to identify the required source coefficients. Possible approaches:

* construct pairs of networks with identical or nearly identical legal observables but opposite oracle contractions;
* exploit sign, permutation or gauge symmetries;
* prove conditional variance lower bounds;
* derive Le Cam, Fano or two-point lower bounds;
* characterize sufficient statistics;
* bound mutual information between legal transcript and correction coefficient;
* prove unavoidable orientation ambiguity;
* show that a source basis depends on forbidden suffix information;
* quantify replacement bias;
* expose hidden target leakage in source construction;
* show that coefficient estimation cost must exceed the T72 margin;
* demonstrate that observed oracle rank does not imply legally observable rank;
* find networks on which the proposed source collapses or reverses.

For every positive proposal from another agent, ask:

1. Was the source frozen before target access?
2. Does orientation use oracle information?
3. Are coefficient labels independent of source selection?
4. Is the estimator absolute or merely correlated?
5. Is the same randomness used twice?
6. Is the replay theorem applicable to the actual target?
7. Was compute fully counted?
8. Does the result survive whole-network splits?
9. Does it survive hostile equivalent reparameterizations?
10. Is the claimed improvement larger than reference noise?

Develop minimal counterexamples wherever possible.

However, do not merely criticize. If you identify the exact missing information, propose the smallest legal observable or protocol change that would remove the obstruction.

The ideal output is either:

* a rigorous impossibility theorem for a broad family of late-interface sources or transcripts; or
* a precise statement of the minimal additional information needed, which directly guides constructive agents.

Maintain a live conflict matrix comparing all agent claims.

---

# Agent 10 — Hostile coordinator, synthesis and breakthrough allocator

You are the coordinating agent. Do not independently pursue one narrow technical idea unless needed to resolve a conflict.

Your responsibilities:

1. Read outputs from Agents 1–9 as they arrive.
2. Verify that each agent used the current ledger and source artifacts.
3. Identify duplicated work, inconsistent definitions and theorem-ID collisions.
4. Separate:

   * source capacity;
   * coefficient observability;
   * replayability;
   * estimator economics;
   * competition value.
5. Require every positive claim to include its strongest hostile interpretation.
6. Assign rapid follow-up checks when two agents disagree.
7. Prevent protected-data opening without frozen gates.
8. Maintain a canonical candidate table.
9. Decide which paths receive additional compute.
10. Produce a final synthesis designed to maximize probability of winning.

For every candidate, maintain:

* exact source definition;
* legality status;
* oracle ratio;
* coefficient target;
* estimator bias and covariance;
* compute cost;
* replay error;
* adjusted score;
* split provenance;
* worst case;
* open theorem obligations;
* reproduction status;
* conflict status.

Use a stage-gate process:

### Gate A — Source capacity

Reject unless the frozen legal source has enough oracle capacity. Prefer (r_*<0.15), with source-specific economics.

### Gate B — Observability

Reject unless there is an absolute estimator or identity for the contractions. Correlation is insufficient.

### Gate C — Economics

Reject unless full bias, covariance and compute pass the generalized T72 score calculation.

### Gate D — Replay

Reject unless the target representation, production indexing and suffix-dependence obligations are resolved.

### Gate E — Validation

Require whole-network grouped validation, frozen hyperparameters and safe tails.

### Gate F — Protected promotion

Open protected data only after every previous gate is documented and immutable.

Continuously compare two resource allocations:

* more work on the leading constructive program;
* switching to a fundamentally different decomposition such as an integrable surrogate.

Do not allocate significant competition resources to sharpening static Kerdock optimality, except when a proof can be completed cheaply or serves publication.

At the end, produce:

1. a simple current-state explanation;
2. the strongest candidate and exact reason it might win;
3. the strongest reason it might fail;
4. the next three decisive tests;
5. paths to stop;
6. unresolved conflicts;
7. a proposed canonical ledger update;
8. a recommendation for the next parallel agent round.
