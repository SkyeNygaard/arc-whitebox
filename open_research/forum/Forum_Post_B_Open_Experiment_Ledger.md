# I am open-sourcing the experiment ledger from a failed estimator search

I spent an intense research sprint trying to beat a strong Kerdock-based estimator for Gaussian expectations of deep ReLU networks.

I did not produce a better deployable estimator. I did produce something I think may be independently useful: a fairly complete map of why many plausible improvements fail, with proofs, experiments, cost models, contradictions, and reopening conditions.

So I am releasing the research ledger, not just a polished paper.

## Why a ledger is different from a postmortem

A postmortem usually tells one clean story. The actual research did not proceed cleanly. A result would look positive under one metric and fail under another. An oracle state would pass an accuracy gate but require target information. A cheap-looking analytic evaluator would omit the cost of factor construction or special functions. A broad claim would later need a narrower scope.

The ledger keeps those corrections visible.

Every important row has:

- a stable ID;
- the estimator class;
- the hypothesis or theorem;
- evidence level;
- cohort and leakage status;
- result;
- complete cost model;
- verdict;
- scope limit;
- primary artifact;
- reopening condition;
- supersession history.

The goal is to make it possible for another researcher to answer: “Has this exact idea already been tested, what killed it, and what genuinely different version remains open?”

## The biggest methodological lesson: oracle headroom is cheap

The project repeatedly found large target-aware improvements. That is useful, but it is only the first of five gates:

1. **Capacity:** Does the representation contain enough correction?
2. **Observability:** Can the signed quantities be obtained from legal runtime information?
3. **Legal state:** Can the method be initialized and rolled through all layers without target refresh?
4. **Correct variance:** Does it reduce the error of the actual structured estimator?
5. **Full score:** Does the gain survive every compute and tail cost?

Many projects fail because a Gate-1 result gets described as if it passed Gate 5.

## Example 1: pointwise residual variance was the wrong metric

The estimator samples complete Kerdock bases. If an anchor \(g\) has known expectation and the residual is sampled in complete blocks, then

\[
\operatorname{Var}(\widehat I_R)
=
\frac1R\operatorname{Var}_b[Q_b(f-g)].
\]

The relevant variance is the variance of complete-block residual means, not rowwise residual variance.

One archived rank-128 anchor retained about 59.7% of pointwise variance but 94.9% of complete-block variance. It looked useful under the ordinary diagnostic and nearly useless under the estimator's real sampling unit.

This single correction invalidated several apparently large projected gains.

## Example 2: an anchor made the residual easy but its expectation stayed hard

Smoothing the network produced a real mechanism result. At one smoothing strength, the complete-block residual ratio fell to roughly 0.039 - about a 25-fold reduction.

But evaluating both the anchor and residual used twice the trajectories. Under matched cost, the hybrid was consistently 2 to 2.7 times worse than direct evaluation.

Could the anchor expectation be computed analytically? Mild smoothing, where the residual was excellent, barely improved Gaussian analytic closure. Strong smoothing improved closure but destroyed most of the residual benefit. The useful and tractable regimes did not overlap.

The correct conclusion is not “analytic anchors are impossible.” It is “this smoothed-network/Gaussian-closure family has no tested overlap between useful residual shrinkage and affordable expectation.”

## Example 3: a mixture represented the missing state but was too expensive to evaluate

A high-capacity mixture reportedly achieved the required state-approximation accuracy at \(K=1536\). That was a real clue: the missing joint information exists.

Three evaluators then failed differently:

- exact componentwise propagation cost about \(8.68\times10^7\) operations per component;
- a shared-reference Taylor expansion became inaccurate as the representation separated component means;
- a low-rank direct/Hermite method required rank 128, where its cheapest displayed contraction exactly equaled \(n^3\) for \(n=256\).

A tied-covariance exception had an exact recurrence, but shared pre-ReLU covariance did not make post-ReLU pair moments component-independent. At \(K=64\), it still required roughly 64.8 million component-specific bivariate CDF calls over the rollout.

The mixture family passed representation capacity and failed affordable evaluation.

## Other bounded audits

The final round ran one-shot tests instead of broad searches:

- a cubic Walsh-phase kernel based on quadratic boundary-normal energies collapsed identically to zero by complete-basis Parseval;
- shared-output/layerwise Kerdock QTT had structural rank obstructions, while the direct-output loophole lacked the necessary nodewise arrays;
- a compositional Hermite identity was exact but generically full rank;
- a full quotient learner was deferred because the required analytic anchor did not exist;
- tail interventions either missed harmful cases, created new tails, or lost after compute.

Each result closes a named construction, not every imaginable descendant.

## The proof result is a separate paper

The release now contains two papers.

The first is theorem-first: in the limiting kernel model, complete Kerdock is within about 0.0233% of the optimum nonnegative static rule at the same 66,048-point budget. The audited frozen signed witness gives a Kerdock-to-optimum risk factor of at most 1.067168, equivalent to at most a 6.2940% reduction in Kerdock risk at the same node budget.

The second is this empirical/methodological account of attempts to leave the static class.

Separating them matters. The clean theorem should not be weakened by a long competition diary, and the experiment ledger should not pretend every local result has the proof paper's evidence status.

## What is actually in the open-source release

The GitHub-ready repository includes:

- both papers in Markdown, PDF, and Word;
- the canonical v31 workbook;
- a readable ledger guide and claim manifest;
- primary theorem reports and available certificate code;
- final audit reports;
- evidence-tier definitions;
- reproducibility and missing-artifact documents;
- issue and pull-request templates;
- scripts that check manifest coverage, hashes, public wording, internal links, and the core proof replays;
- CSV exports of the public ledger sheets;
- an AI-assistance disclosure and hostile audit report.

It does **not** currently include the exact final 129-basis package or official Mini-100 result JSON. Those files were referenced in the local narrative but were not recoverable from the accessible archive. The repository says this prominently rather than substituting a similar package.

## What I hope people do with it

I am not trying to preserve an advantage. I would be happy for other people to reproduce the work, tie the estimator, or beat it.

High-value contributions include:

- independently reconstructing the proof interval stack;
- recovering and authenticating the exact final baseline package;
- rerunning the mixture, Taylor, and rank experiments from raw scripts;
- finding a counterexample to a claimed closure;
- implementing a clearly different finite-width, adaptive, nonlinear, or analytic-state method;
- converting the workbook into a better public research database.

The most important norm is simple: every improvement should state what new information or structure it uses, and every negative result should state exactly what it does not rule out.

Research ledgers are usually private debris. I think this one is more useful as shared infrastructure.
