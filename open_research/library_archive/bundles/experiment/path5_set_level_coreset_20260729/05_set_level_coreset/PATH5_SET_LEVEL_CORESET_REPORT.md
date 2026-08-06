# Path 5 — Set-Level Coreset and Subspace Compression

**Date:** 2026-07-29  
**Status:** **Pause the tested selector families; preserve the fixed support portfolio and labels.**  
**Protected official/Mini holdouts:** not opened.  
**New untouched synthetic test block:** not opened, because the development selector gate failed.

## Executive verdict

This round found a real positive result and a decisive bottleneck.

A small, completely precomputed portfolio of basis-balanced supports contains excellent legal coresets much more often than prior rowwise and pairwise work suggested. The full 128-support portfolio had a primary-gate support on every exact-geometry network for which the full portfolio was resolved: the smoke network, all eight development networks, and all 32 new ranker-training networks. On the two training networks missed by the frozen top-eight sublibrary, the full portfolio still contained 31 and 8 primary-gate supports.

The deployable problem remains unsolved: none of the tested set-level selectors identified those supports safely. The best fixed direct-sketch rule passed only 6/8 development networks. A learned candidate ranker trained on 32 new exact-geometry networks also passed only 6/8 development networks and had a `3.33e-7` worst tail. Therefore no new untouched test block was opened.

The branch should not continue with neighboring sketch dimensions, ridge constants, tree models, or larger versions of the same final-coordinate representation. Preserve the support portfolio because it establishes a useful new fact: **structured precomputed support libraries can cover the oracle support set; network-specific support identification is the remaining problem.**

## What is algebraically new

Prior failed methods assigned independent scores to rows or edges. This round instead manipulated complete 4,096-pair supports:

1. Construct a fixed library of 128 basis-balanced supports:
   - 64 deterministic balanced pseudorandom supports;
   - 64 deterministic affine-stratified supports.
2. Keep exactly 31 or 32 antipodal pairs per Kerdock basis, totaling 4,096 pairs = 8,192 rows.
3. Evaluate whole-set discrepancy, basiswise span/conditioning surrogates, and direct output-coordinate sketch feasibility.
4. Restrict to a frozen eight-support portfolio and train a candidate-level ranker from whole-support diagnostics.
5. Use positive, basis-mass-preserving oracle calibration only as the pre-implementation support label, with relative weights in `[0.05, 4.0]` and ESS at least `0.8 M`.

There is no runtime NNLS, greedy herding, exchange solver, or iterative support optimizer. The support candidates are precomputed. The direct-sketch ranker uses fixed ridge constants and a closed finite scan.

## Cohorts and split hygiene

| Cohort | Purpose | Networks | Status |
|---|---:|---:|---|
| Seed 63998 | Engineering smoke and full-library sweep | 1 | Exposed |
| Seeds 64000–64007 | Selector development | 8 | Exposed |
| Seeds 64300–64331 | New grouped ranker training | 32 | Exposed training |
| Seeds 64100–64115 | Earlier frozen-support validation | 16 | Previously exposed; reused only as context |
| Seeds 64400+ | Intended untouched test | 0 opened | Kept closed after development failure |

No official or Mini protected holdout was opened.

## Result 1 — The fixed support library is genuinely strong

### Full 128-support portfolio

| Cohort | Primary-gate coverage | Best-support worst MSE | Notes |
|---|---:|---:|---|
| Smoke 63998 | 1/1 | `2.276e-14` | 47/128 supports passed `1.1e-8` |
| Development 64000–64007 | 8/8 | `3.031e-13` | 19–62 passing supports per network |
| New training 64300–64331 | 32/32 implied | `2.385e-8` for top-eight only | Top-eight passed 30/32; full 128 recovered both misses |

On the two new-training top-eight misses:

- seed 64320: full library best `1.587e-14`, with 31/128 primary-gate supports;
- seed 64331: full library best `2.007e-13`, with 8/128 primary-gate supports.

This is the strongest positive finding of the round. Good supports are not isolated products of an expensive exchange solver. A small deterministic support family repeatedly contains them.

### Frozen top-eight portfolio

The frozen candidate IDs were `[81, 111, 88, 35, 91, 78, 51, 38]`.

- Development: oracle-best-of-eight passed 8/8; worst `3.793e-9`.
- New 32-network training corpus: passed 30/32; worst `2.385e-8`.
- Previously exposed 16-network validation: passed 15/16 at `1.1e-8`, 16/16 at `2.2e-8`; worst `1.248e-8`.

A single fixed support does not explain this coverage. Candidate 81 passed only 4/16 on the previously exposed validation block and had a `1.405e-7` worst tail. The identity of the good support is network-specific.

## Result 2 — Handcrafted whole-set scores still do not identify the support

On smoke seed 63998:

| Selector | Selected support oracle MSE |
|---|---:|
| Fixed random control | `1.367e-7` |
| Global set-mean discrepancy | `2.748e-8` |
| Basiswise diagonal effort/span | `2.231e-7` |
| Best support in the same fixed library | `2.276e-14` |

The basiswise feasibility formula was falsified immediately. It selected a support worse than random. Global discrepancy was better but still missed the primary gate.

Across all 128 candidates on the smoke network, global discrepancy had only `0.225` Spearman correlation with log oracle error; basis effort had `0.051`. The fixed library contained 47 primary-gate supports, yet these scores did not rank one first.

## Result 3 — Direct same-support sketches are closer, but still unsafe

The strongest representation propagated the full cloud through layer 31, computed selected final-output coordinates for every row, and scored complete fixed supports by one-shot basis-preserving ridge feasibility.

### Ranking all 128 supports

Best frozen development rule: `q=128`, ridge scale `1e-4`.

- primary passes: 6/8;
- secondary passes: 6/8;
- mean selected-support oracle MSE: `2.563e-8`;
- worst: `1.360e-7`;
- mean candidate-ranking Spearman: `0.012`.

The direct sketch can occasionally choose a nearly exact support, but it has almost no stable global ranking correlation and catastrophic misses remain.

### Ranking only the frozen top eight

| Rule | Primary passes | Secondary passes | Mean MSE | Worst MSE |
|---|---:|---:|---:|---:|
| `q=128`, ridge `1e-4` | 6/8 | 6/8 | `1.134e-8` | `5.640e-8` |
| `q=32`, ridge `1` | 5/8 | 8/8 | `7.296e-9` | `1.955e-8` |

The `q=32` rule has a relatively safe secondary tail, but it misses the primary support gate on three development networks. Its sketch residual does not separate passing from failing cases well enough to certify an abstention rule.

## Result 4 — More training did not rescue the set scorer

A new corpus of 32 exact-geometry networks was generated. For each network and each of the frozen eight supports, the dataset contains:

- fixed-ridge residuals for `q ∈ {16, 32, 64, 128}`;
- global support discrepancies;
- within-network ranks and standardized score features;
- candidate identity;
- exact same-support bounded-positive oracle labels.

Pointwise ridge, logistic pass prediction, pairwise logistic ranking, and small tree ensembles were evaluated with grouped network folds. The frozen development-selected model was ridge regression with alpha 10:

- grouped training CV: 13/32 primary passes, 15/32 secondary passes, worst `4.848e-7`;
- development: 6/8 primary passes, 6/8 secondary passes, worst `3.335e-7`.

It did not improve over the direct fixed sketch rule. The larger corpus therefore supports pausing this representation rather than opening a fresh holdout.

## Compute and runtime gate

The direct sketch is formed only after full propagation through layer 31, so it can save only the final dense layer. Approximate arithmetic for the frozen eight-support scan:

| Sketch | Candidate / dense final layer | Net saved | Fraction of ~175.5B baseline saved |
|---|---:|---:|---:|
| `q=32` | 0.288 | `6.164B` ops | 3.51% |
| `q=128` | 0.779 | `1.911B` ops | 1.09% |

These estimates include a 2,064-row coordinate pilot, the all-row sketch, eight whole-support Gram scores, and full final-output propagation for the selected 8,192 rows.

Thus the frozen top-eight `q=32` arithmetic condition passes. The statistical selector gate does not. Scanning all 128 supports with 128-dimensional Gram diagnostics is not an attractive runtime route and, independently, failed statistically.

## Gate evaluation

| Gate | Result |
|---|---|
| Same-support oracle added MSE about `1.1e-8` | **Portfolio passes; selectors fail** |
| Safe tails across fresh exact geometry | **Not demonstrated by a selector**; no new untouched test opened |
| Positive bounded weights | **Pass for oracle support labels**; `[0.05, 4.0]`, ESS ≥ `0.8 M` enforced |
| Selector cost below propagation saved | **Pass for top-eight q32 arithmetic**, but irrelevant after statistical failure |
| No runtime iterative support optimizer | **Pass** |

## Interpretation

The obstruction is narrower than before:

1. **Support existence is not the problem.** A fixed 128-support family covered every fully resolved exact-geometry network in this round.
2. **A small portfolio is nearly sufficient.** Eight supports covered 30/32 new training networks and 15/16 previously exposed validation networks.
3. **Support identity is highly network-specific.** One fixed support failed badly, and direct output sketches had unstable rankings.
4. **Low-dimensional final-coordinate agreement is not a certificate.** Even a direct 128-coordinate sketch missed supports that calibrate the unobserved output coordinates well.
5. **The current representation pays too late.** Because it requires full propagation to layer 31, its maximum upside is only the final layer, not the four-layer compression originally sought.

## Closure scope

Closed in this round:

- basiswise diagonal effort/span scoring;
- fixed global-discrepancy scans over the 128-support library;
- direct final-coordinate sketch ranking at 16, 32, 64, and 128 coordinates with the tested fixed ridge family;
- the frozen eight-support portfolio with those sketch scores;
- nearby pointwise, pairwise, logistic, ridge, and small tree rankers trained from the same support diagnostics;
- a single universal support chosen from development.

Not closed:

- existence of useful precomputed support libraries — positively supported;
- a weight-derived, earlier-layer set classifier that predicts support identity before expensive tail propagation;
- a substantially larger support-error training corpus paired with a qualitatively new representation;
- support families indexed directly by network invariants rather than selected by final-coordinate sketches.

## Recommended branch action

**Pause neighboring Path 5 work. Preserve the support library, exact labels, and grouped datasets.**

Reopen only if the proposed selector differs materially from the tested family and can be computed before layer 29. A credible reopening proposal should preregister:

- direct prediction of support error or support ID, not row importance or edge overlap;
- grouped-network training with at least 128 new base networks;
- a frozen 16–32 network exact-geometry holdout;
- an explicit fallback/abstention rule;
- arithmetic and wall cost that still saves at least two dense tail layers.

## Reproducibility

The bundle contains:

- deterministic support-library construction;
- exact Kerdock geometry asset;
- full 128-support development labels;
- 32-network top-eight training corpus and eight-network development corpus;
- direct-sketch and learned-ranker results;
- scripts for regeneration, scoring, validation, and aggregation;
- SHA-256 manifest.
