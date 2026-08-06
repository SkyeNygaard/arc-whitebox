# WHestBench two-paper and open-research release strategy

**Prepared:** August 1, 2026

## Recommendation

Release the project as three linked objects rather than one monolithic paper:

1. **Paper A — mathematical result:** *Near-Optimality of Complete Kerdock Cubature for Static Deep-ReLU Gaussian Integration*.
2. **Paper B — empirical and methodological record:** *Oracle Headroom Is Not an Estimator: An Open Experiment Ledger for Compute-Constrained Gaussian Integration of Deep ReLU Networks*.
3. **Open research repository:** the canonical ledger, proof sources, final branch audits, code-audit records, figures, claim manifest, and explicit reproduction gaps.

Paper A should be the first object sent to mathematical reviewers. Paper B and the repository explain how the theorem changed the search process and preserve the negative results so other researchers do not repeat them.

## Why two papers

### Paper A has a coherent theorem contribution

Paper A answers one clean question: how much can any static, network-independent linear cubature rule improve on the complete Kerdock/MUB design at the same 66,048-node budget for the dimension-256, depth-32 limiting ReLU kernel?

The answer has two tiers:

- among nonnegative mass-one rules, complete Kerdock is certified within approximately 0.0233% of the infimum;
- among arbitrary-real-weight mass-one static rules at the same node budget, the fully replayable frozen witness permits at most approximately a 6.2940% reduction in Kerdock risk (equivalently, a Kerdock-to-optimum ratio at most 1.067168).

The proof stack—positive Delsarte auxiliary optimization, signed inertia, negative-weight support-count hierarchy, block-trace sharpness, equality characterization, and atomic nonattainment for the older abstract floor—belongs together. Mixing in the experimental search would weaken the main theorem and make the scope harder to understand.

### Paper B has a distinct methodological contribution

Paper B is about how apparent oracle headroom disappears at successive gates: observability, legal initialization, complete-block variance, evaluation cost, and adjusted score. Its most valuable result is not one theorem but a transparent map of tested estimator classes and why they failed.

The experiment ledger is central evidence for this paper. Publishing it allows readers to inspect superseded claims, negative results, scope corrections, and evidence tiers rather than seeing only a polished endpoint.

## Recommended publication sequence

### Stage 1 — public research release

Publish the repository with:

- both preprint drafts;
- canonical v31 ledger and public claim manifest;
- the proof source files and available verification scripts;
- final branch audits;
- a conspicuous `RELEASE_STATUS.md` and `BASELINE_PACKAGE_MISSING.md`;
- issues enabled for independent reproductions, corrections, and recovered artifacts.

Do **not** describe the empirical baseline package as independently reproduced. The exact reported shipping archive and official Mini-100 JSON remain absent from the assembled release.

### Stage 2 — targeted mathematical review

Send the two-page overview and Paper A to one or two reviewers. Ask for bounded feedback on:

- correctness and novelty of the nonnegative auxiliary theorem;
- correctness of the signed positive-index/inertia argument;
- whether the scope supports the phrase “essentially optimal”;
- whether the computer-assisted certificate has enough provenance;
- the most suitable mathematical venue.

### Stage 3 — forum release

Post the shorter Kerdock explanation first, linking Paper A and the repository. Post the open-ledger essay separately after readers can inspect the repository. Keeping the posts separate makes the theorem legible and lets the ledger discussion focus on research practice.

### Stage 4 — formal preprint

A formal Paper A preprint should wait until the inherited directed kernel-coefficient interval stack has been independently reconstructed or the paper is explicitly labeled as a computer-assisted result pending that reconstruction. Paper B can be posted earlier as a technical report if every numerical result retains its evidence label.

## Open-source policy

The prepared repository uses:

- **MIT License** for code;
- **CC BY 4.0** for papers, figures, and prose documentation;
- full preservation of negative results and superseded claims;
- no claim that absent artifacts are present;
- contribution templates for reproductions, corrections, and new estimator-class results.

This is compatible with the goal of helping others tie or improve the baseline. The most useful release is not only the estimator code: it is the theorem certificates, failed paths, exact gates, and provenance needed to understand which directions are still open.

## Highest-priority missing artifacts

1. Exact archive of the reported final 129-basis estimator.
2. `official_129basis_mini100_20260731.json` with per-network rows.
3. Environment lockfile and exact official command transcript.
4. Independent Arb/FLINT/MPFR reconstruction of inherited kernel coefficient intervals.
5. The mixture-ladder, pooled-within Taylor, and rank-sweep scripts and arrays.
6. A clean public benchmark/data-access guide consistent with the benchmark license.

The repository should remain public even if these are never recovered. Missingness is part of the scientific record and is explicitly documented.

## Suggested repository name

`whestbench-open-research`

Suggested description:

> Open proofs, estimator-class audits, experiment ledger, and reproducibility gaps from a compute-constrained Gaussian integration study of deep ReLU networks.

## Release labels

- `v0.1-open-ledger`: papers, ledger, available proofs, audits, and known gaps.
- `v0.2-proof-reproduction`: independent kernel-interval reconstruction added.
- `v0.3-baseline-reproduction`: exact final package and Mini-100 JSON recovered and rerun.

