# WHestBench: static cubature limits, and a graded competition submission

Open research for the [ARC White-Box Estimation Challenge 2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026):
given the weights of a deep ReLU network, predict its expected per-neuron
activations more accurately than running it many times.

This repository holds three things that are usually kept apart — a graded
competition submission with its exact archive, a proof that the design it uses
is close to optimal in its class, and the full unedited record of everything
that did not work. The third is the largest.

```
Phase 1 submission #320802 · GRADED · adjusted 1.55e-7 · 50/50 public MLPs · 0 failures · 4.2x Monte Carlo
```

---

## Start here

| if you want to… | go to |
|---|---|
| see what was submitted and how it works | [`arc_whitebox/submissions/production_baseline_320802/`](arc_whitebox/submissions/production_baseline_320802/) |
| read the 6-page technical write-up | [`whestbench/papers/Phase1_Algorithmic_Contribution_320802.pdf`](whestbench/papers/Phase1_Algorithmic_Contribution_320802.pdf) |
| check a claim's evidence status | [`whestbench/claims.csv`](whestbench/claims.csv) |
| read the theorem and replay the proof | [`theory/`](theory/) |
| browse every experiment, including the failures | [`open_research/library_archive/`](open_research/library_archive/) |
| read the raw research ledger | [`whestbench/ledger/`](whestbench/ledger/) |
| find out what is still broken or missing | [`open_research/RECONCILIATION_20260804.md`](open_research/RECONCILIATION_20260804.md) |

## The mechanism, briefly

The challenge input distribution is an isotropic Gaussian, so every post-ReLU
mean is an integral over the sphere in dimension 256. Instead of sampling it,
the estimator evaluates it on a **fixed 66,048-point spherical 5-design** built
from the Kerdock code and its maximal real mutually unbiased bases — 128
Kerdock bases plus the coordinate basis, every direction carried with its
antipode. The design is frozen once and reused for every network; it never
looks at the weights. Antipodal closure kills every odd-degree term for free.

The part worth arguing about is that **the same algebraic object supplies both
the accuracy and the speed**. Kerdock bases are chirps acted on by a
Walsh–Hadamard transform, so applying the design to the first weight matrix is
eight butterfly stages rather than an explicit 66,048 × 256 by 256 × 256
product. The remaining 30 propagations run through an exact depth-5
Strassen–Winograd kernel charging `7^5 = 16,807` products where conventional
blocking charges `32^3 = 32,768`.

Layer 0 is closed form — the half-normal mean `‖w_j‖ / √(2π)`. Layers 1–30 are
returned as zeros. That last point is a deliberate choice against the scored
objective and is why all-layers MSE sits near 0.74; it is stated here rather
than buried, and again in §3.7 of the write-up.

### The measurement I would most like challenged

Holding the design, node set, radius, and returned rows exactly constant on one
fixed 100-network development cohort, and changing only the arithmetic:

| arm | raw final MSE | effective compute | budget used | adjusted score |
|---|---:|---:|---:|---:|
| dense propagation | 2.2826e-7 | 2.689e11 | 98.86% | 2.2565646e-7 |
| tracked depth-5 Winograd | 2.2819e-7 | 1.748e11 | 64.27% | **1.4641716e-7** |

Raw error is unchanged to three significant figures. The adjusted score
improves **1.5412×**, entirely through the charged-cost term. Under an
effective-compute score, exactly restructuring the arithmetic bought as much as
a better statistical rule would have.

## The ceiling

The companion question is how much room was left at all. For the dimension-256,
depth-32 limiting ReLU kernel at a 66,048-node budget:

- among **all** nonnegative mass-one rules on arbitrary spherical nodes,
  complete Kerdock is at most **0.0233242%** above the infimum;
- allowing **arbitrary signed** mass-one weights, no static rule reduces
  Kerdock risk by more than **6.2940%**.

So moving nodes or reweighting them was close to exhausted before the
competition work started, which is why the remaining budget went into
arithmetic. The paper, the replayable certificate archive, and an explicit
trust boundary are in [`theory/`](theory/). The proof archive is the frozen
arXiv ancillary bundle (not yet submitted); five checks replay in about 40
seconds. What actually gets uploaded, and how to rebuild it, is
[`theory/ARXIV_SUBMISSION.md`](theory/ARXIV_SUBMISSION.md).

**These are lower bounds on a limiting kernel in a static linear class.** They
say nothing about finite-width, adaptive, nonlinear, or network-dependent
estimators — which is very likely where the leaderboard leaders are working.

## Verify it yourself

The `whest` CLI is the official challenge starter kit, not part of this
repository. Install it first:

```bash
pip install "git+https://github.com/AIcrowd/whest-starterkit.git"
```

```bash
python scripts/check_competition_release.py
whest validate-package arc_whitebox/submissions/production_baseline_320802/submission.tar.gz
```

```bash
python -m pip install -r theory/proof_archive/requirements.txt
cd theory/proof_archive && python scripts/check_package.py && python scripts/run_verification_portable.py
```

## Map

```text
arc_whitebox/        competition estimators, submission archives, scripts, notes, results
  submissions/production_baseline_320802/   ← the graded #320802 estimator
arc_ceiling/         numerical diagnostics for static cubature geometry
theory/              arXiv paper + byte-identical replayable proof archive
whestbench/          release policy, evidence labels, claims, ledger CSVs, write-up
  ledger/            34 sheets of the canonical research ledger, as CSV
open_research/       the full research release: papers A/B, forum posts, audits,
                     evidence, and 35 experiment bundles under library_archive/
T0_unblocked_.../    grader instrumentation behind the graded submission
scripts/             release check, ledger export, write-up build
```

## On completeness

The release is deliberately wide. Every script, note, figure, audit, ledger
row, and result record small enough to read in a browser is published —
including superseded work, retracted claims, and experiments that went nowhere.
Only three categories are held back: bulk numerical arrays, benchmark weights
and vendor checkouts, and a handful of large archives whose source is published
beside them. The rules are in [`.gitignore`](.gitignore) with reasons attached.
If an experiment you expected is not here, treat that as a bug and open an
issue.

Two consequences worth stating plainly. First, the ledger and the library
archive contain rows that later measurement contradicted; the
`Reconciliation Audit`, `Contradiction Map`, and `Evidence Quarantine` sheets
exist because of that. A row appearing here is not a claim that its number is
correct. Second, the imported research release was assembled before the
competition submission was identified, and several of its "missing artifact"
entries turned out to be recoverable. Those documents are preserved as written
and corrected in
[`open_research/RECONCILIATION_20260804.md`](open_research/RECONCILIATION_20260804.md)
rather than silently edited.

### Evidence labels

Claims carry one of: `official_grader`, `checked_local`, `reported`,
`external_review`, `exact_identity`, `code_fact`, `process_record`,
`disclosure`. The distinctions are load-bearing — see
[`whestbench/claims.csv`](whestbench/claims.csv) and
[`whestbench/RELEASE_STATUS.md`](whestbench/RELEASE_STATUS.md). In particular
the theorems are **computer-assisted and replayable**, not independently
verified: the depth-32 coefficient and curvature interval stack has not been
rebuilt in a second directed-arithmetic implementation, and human review of the
analytic reduction is outstanding.

## Use of language models

Heavy, throughout. LLM agents did ideation, most of the implementation, the
experiment scaffolding, proof attempts, code review, and drafting — including
of this file. The research direction, the promotion and rejection decisions,
and the evidence labels are the author's. Agent agreement is not independent
verification, and agent reports in `open_research/` are provenance, not
referees. See [`open_research/AI_ASSISTANCE.md`](open_research/AI_ASSISTANCE.md)
and §10 of the Phase 1 write-up.

## Contributing

The most useful contributions, roughly in order:

1. an independent reconstruction of the depth-32 kernel interval stack;
2. a complete rerun of any reported empirical experiment;
3. a correction to a theorem, a cost model, or an evidence label;
4. a finite-width, adaptive, or nonlinear estimator that leaves the static class.

See [`open_research/CONTRIBUTING.md`](open_research/CONTRIBUTING.md) and
[`open_research/OPEN_PROBLEMS.md`](open_research/OPEN_PROBLEMS.md).

## Licensing

Code is MIT ([`LICENSE`](LICENSE)). Research text and ledger documentation are
intended for CC BY 4.0, subject to third-party benchmark and dataset rights —
see [`open_research/LICENSE-DOCS.md`](open_research/LICENSE-DOCS.md). Cite via
[`CITATION.cff`](CITATION.cff).
