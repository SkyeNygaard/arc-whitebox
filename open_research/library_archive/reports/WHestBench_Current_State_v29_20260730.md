# WHestBench Current State v29 — Late-Agent Cutoff Audit

**Audit cutoff:** 2026-07-30 23:01 ET  
**Canonical ledger:** `whestbench_canonical_research_ledger_20260730_reconciled_v29_late_agent_audit.xlsx`  
**Protected or official data newly opened:** No

## Executive conclusion

The v28 synthesis was not fully current. Six substantive reports arrived minutes after its cutoff:

- Agent 3 — characteristic-function and Fourier closure
- Agent 4 — deterministic Gaussian-mixture and transport propagation
- Agent 7 — polyhedral-cone / activation-fan compression
- Agent 8 — nonlinear late-innovation identities
- Agent 9 — full-weight equivariant neural operator
- Agent 10 — automated nonlinear estimator synthesis

After integrating them, there is still **no deployable estimator** and no new measured winning oracle. The main program remains a coherent nonperturbative latent distribution, but the decisive experiment is now sharper and cheaper:

> **Test a tied-covariance latent Gaussian mixture through K=64 before assuming component-dependent covariance or covariance prototypes are necessary.**

The primary sequence is now:

1. **M192:** full oracle mixture ladder, tied covariance first.
2. **M187:** weighted sign–magnitude and covariance-modulation attribution in the same harness.
3. **M188:** legal 32-layer rollout of the best frozen mixture family, only after M192 passes.
4. **M190:** environment-weighted compositional-chaos rank and legal-orientation audit in parallel.

Everything else is a bounded falsifier, a theorem lane, or closed.

## What v28 was missing

### 1. Agent 4 materially changes the primary test

The tied-covariance model is

\[
q_\ell(z)=\sum_{k=1}^{K}\pi_k\mathcal N(z;\mu_{\ell k},R_\ell).
\]

Componentwise Gaussian–ReLU integration is followed by:

\[
\mu_{\ell+1,k}=W_{\ell+1}m_{\ell k},
\]

\[
R_{\ell+1}
=
W_{\ell+1}
\left(\sum_k\pi_k C_{\ell k}\right)
W_{\ell+1}^{\mathsf T}.
\]

Under the approximate input mixture, this preserves the complete global mean and covariance exactly and keeps the state PSD. It requires one full covariance transform per layer rather than one per component.

The resulting dense algebra is approximately 1.04B FLOPs plus componentwise bivariate moment evaluations. This makes K=32 and K=64 scientifically necessary oracle tests.

The correct M192 ladder is:

1. tied covariance, K = 1, 2, 4, 8, 16, 32, 64;
2. full component covariance where economical;
3. shared low-rank covariance deviations;
4. latent-space mixture with a full covariance tail.

Promotion requires held-out next-variance error at most 0.3%. Tied covariance closes if K=64 remains above approximately 0.5%. If only full component covariance with K greater than 16 passes and no shared compression preserves the gain, the result is likely compute-negative hidden oracle structure.

### 2. Standalone characteristic-function closure is closed

Complete characteristic functions contain enough information, but finite collections of ordinary one- and two-dimensional characteristic functions are not recursively closed under dense linear mixing and coordinatewise ReLU.

One Fourier character becomes a different local character on each orthant. Exact propagation requires an all-orders hierarchy of orthant-truncated characteristic functions. Equal coordinate-pair characteristic functions can coexist with different next ReLU expectations.

Characteristic functions remain useful only as:

- a positive-definite representation of the low-dimensional latent law;
- a numerical integration basis;
- a local downstream-contraction tool.

They are absorbed into the primary latent-state program rather than retained as a separate branch.

### 3. Baseline-transcript-only nonlinear identities are closed

For any finite antipodal design, there exists an architecture-valid bias-free ReLU network that is zero at every design node but has positive Gaussian expectation.

Therefore no universal exact method can be formed solely by recombining existing baseline output values, even using:

- nonlinear jackknives;
- ratios or norms;
- Walsh transforms;
- U-statistics;
- leave-one-basis expressions;
- arbitrary measurable transcript functions.

This does not close full white-box use of weights or explicit boundary integration. It closes the idea that the missing expectation can be recovered exactly by clever algebra on the existing output transcript alone.

### 4. Local gate flux is incomplete

For a positively homogeneous CPWL scalar function:

\[
E|g(X)|
=
E[\operatorname{sign}(g)\Delta g]
+
2E[\delta(g)\|\nabla g\|^2].
\]

For a deep preactivation, the Laplacian term is supported on inherited upstream ReLU boundaries. A final-gate near-zero or density statistic sees only part of the identity.

The only surviving transcript-assisted late-phase idea is **M194**, one frozen weight-coupled cubic Walsh/boundary bispectrum. It must be derived algebraically, pass all symmetries, and show positive grouped correction covariance before any amplitude fitting. Broad kernel or symbolic search is not justified.

### 5. Activation-fan compression is narrowed sharply

Exact BDD/ZDD compression by merging equal live affine maps is generically ineffective: distinct feasible live gate histories almost surely produce distinct affine maps under continuous weights.

The surviving branch is not region enumeration. It is:

> a low-normal-rank, output-weighted, certified facet DAG.

M193 should measure:

- output-weighted boundary-normal rank;
- exact gate-current contribution by layer;
- certified sign mismatch after normal projection;
- oracle reduced-fan error.

This is a low-prior bounded diagnostic, not a co-lead.

### 6. Full-weight learning is not information-theoretically impossible

The complete realized weights determine the network and its integral. Agent 9 gives an explicit generic quotient removing input rotations and positive ReLU gauges while leaving permutation covariance. Therefore historical learning failures cannot imply that full weights lack signed information.

The competition evidence remains extremely negative:

- reported direct signed-target R² is approximately 0.036;
- the adaptive direct-output source requires roughly 0.84 grouped source-space R² at equal compute.

M195 is mathematically open but strategically deferred. It should be run only as a bounded exact-quotient falsifier or as a residual model around a much stronger analytic anchor.

### 7. Broad nonlinear search is closed by a perturbative filter

A smooth, well-conditioned nonlinear wrapper around small transcript deviations equals its leading linearization plus higher-order terms. If the leading linear class is closed, the nonlinear remainder is too small for an order-one score gain unless the method introduces:

1. a genuinely new phase-bearing representation;
2. a controlled singularity;
3. nonsmooth activation-boundary information.

At unchanged cost, an 80% residual reduction requires correction cosine at least:

\[
\sqrt{0.8}\approx0.8944.
\]

This eliminates broad symbolic regression, ordinary shrinkage, smooth ratios, norms and generic nonlinear transcript wrappers as serious paths.

## Current canonical portfolio

### Primary

#### M192 — Mixture-family oracle ladder

Run tied K through 64 first. Escalate to covariance dependence only if the tied family is insufficient.

#### M188 — Legal mixture rollout

Only after M192 passes:

- analytic nontrivial initialization from the Gaussian layer;
- no empirical per-layer refitting;
- frozen K, ranks, split layers and merge rules;
- full PSD and realizability audit;
- raw MSE at most `2.962e-7`;
- complete effective compute at most `27.2B`.

### Parallel secondary engine

#### M190 — Environment-weighted compositional chaos

Test whether the exact degree-10 adjoint-harmonic expectation has low downstream ranks that can be generated from weights and legal adjoints. This may become a cheaper contraction engine for the latent state.

### Bounded hedges

- **M189:** one existing-array Kerdock-index QTT rank/common-pivot audit.
- **M193:** one boundary-normal and gate-current oracle audit.
- **M194:** one algebraically fixed cubic boundary-bispectrum test.
- **M195:** deferred quotient-equivariant neural-operator falsifier.

### Dormant theorem-derived hedge

A genuinely exact nonlinear late-innovation identity using all-ancestor boundary information remains logically open. Generic phase learning, marginal gate flux and transcript recombination are closed.

## What is now closed

- standalone selected-direction characteristic-function propagation;
- independent pair-CF recursive states;
- pairwise orthant/sign/truncated matrices as complete states;
- transcript-only universal nonlinear identities;
- radial or positive layer-scale differentiation;
- smooth nonlinear wrappers around closed transcripts;
- ordinary symbolic/evolutionary estimator search;
- exact BDD/ZDD or affine-map-equality fan compression;
- mass-only dominant-region claims;
- Cartesian and ordinary rotated continuous-input tensor trains;
- coordinate-sparse and ANOVA polynomial chaos;
- ordinary sampling, multifidelity, activation MLMC and linear checkpoint gauges.

## Is anything still missing?

At the 23:01 ET cutoff, all ten relevant agent programs have a located final report or a substantively overlapping report:

- Agents 1 and 2 define the coherent latent-law and sign–magnitude state.
- Agent 3 closes standalone Fourier closure.
- Agent 4 sharpens the mixture recurrence and oracle ladder.
- Agent 5 narrows tensor integration to the QTT falsifier.
- Agent 6 narrows polynomial chaos to the contracted expectation engine.
- Agent 7 narrows fan integration to low-normal-rank facets.
- Agent 8 closes transcript-only nonlinear identities and defines the bispectrum hedge.
- Agent 9 corrects the full-weight information claim and defines the deferred quotient learner.
- Agent 10 supplies the nonlinear alignment and perturbative search filters.

Duplicate generated copies of several reports were present, but they contained the same substantive report and were not counted as independent evidence.

No later substantive Library result was found in the cutoff scan.

## Final assessment

The project is up to date at this cutoff.

The main uncertainty is now experimentally crisp:

> **Can a legally initialized tied-covariance mixture with K no greater than 64 reduce the held-out next-variance error to 0.3%, and can that state survive 32 layers without oracle refitting?**

If yes, there is a coherent route to a real candidate.

If tied mixtures fail but compressed covariance-dependent mixtures pass, the problem becomes one of economical conditional covariance transport.

If the entire M192 ladder fails, the strongest coherent distributional path closes, leaving only the compositional contraction engine and low-probability boundary/tensor/full-weight hedges.
