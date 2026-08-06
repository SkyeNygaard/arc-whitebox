# WHestBench Open Research

This repository is an open research release for compute-constrained Gaussian integration of deep ReLU networks.

It has three purposes:

1. publish a proof-focused account of why complete Kerdock cubature is essentially optimal within its nonnegative static class, together with a replayable signed-rule barrier;
2. publish the experiment ledger and scoped negative results from attempts to leave that class;
3. make it easier for other researchers to reproduce, challenge, tie, or beat the baseline.

## Headline results

### Paper A: static Kerdock near-optimality

For the dimension-256, depth-32 limiting normalized-ReLU kernel and a budget of 66,048 spherical evaluations:

- complete Kerdock is certified to be at most **0.0233242%** above the infimum over static, network-independent, nonnegative mass-one rules;
- the fully replayable frozen signed witness retains at least **93.7060168%** of Kerdock risk. The Kerdock-to-optimum factor is at most **1.067168x**, equivalent to at most a **6.2940% reduction** in Kerdock risk at the same node budget;
- a marginally stronger reoptimized signed constant was reported historically, but its rational witness was not recovered and is excluded from the audited headline;
- after consolidating duplicate locations and removing zero weights, at least 1,072 negative-weight support entries rule out a 1.05x Kerdock-to-rule factor, and at least 4,160 make the rule certified worse than Kerdock.

These are limiting-kernel, static-linear results. They do not cover finite-width, adaptive, nonlinear, or network-dependent estimators.

### Paper B: oracle headroom and estimator gates

The companion paper documents attempts to exploit structure outside static cubature. Its central methodological claim is that an estimator must pass five distinct gates:

1. representation capacity;
2. runtime observability;
3. legal initialization and recurrence;
4. the correct structured-estimator variance;
5. complete MSE, compute, and tail accounting.

## Relation to neighboring Kerdock optimality results

Exact semidefinite-programming work has established packing optimality of the same cardinality family of antipodal Kerdock/mutually-unbiased-basis configurations, including the 66,048-point arrangement in dimension 256. This release studies a different objective: the RKHS/kernel-energy risk induced by a particular depth-32 ReLU kernel at a fixed integration-node budget. The packing results are important neighboring literature, but they neither imply nor are implied by the cubature-energy certificate proved here. See Paper A for the detailed comparison.

## Important release status

The theorem reports and several exact-rational artifacts are included. The canonical research ledger is included.

The release is **not yet artifact-complete**. In particular, the exact final 129-basis estimator package tied to the reported official Mini-100 run, the official per-network JSON, and several local mixture/Taylor/rank scripts were not recoverable from the accessible archive. See [`RELEASE_STATUS.md`](RELEASE_STATUS.md) and [`BASELINE_PACKAGE_MISSING.md`](BASELINE_PACKAGE_MISSING.md).

Do not replace missing artifacts with similarly named packages or hashes.

## Repository map

```text
papers/                         Two papers in Markdown/PDF/DOCX
forum/                          Public-facing posts
ledger/                         Canonical workbook, state memo, claim manifest
evidence/primary_theory/        Proof reports, recovered v5.2 artifacts, and replay scripts
evidence/final_audits/                   Bounded branch audits and reproduction audit
evidence/code_audit/            Code/package provenance audits
scripts/                        Release-integrity and core-verification scripts
audit/                          Hostile audit report, change log, and readiness gates
environment/                    Tested local environment
.github/                        Reproduction and result templates
```

## Fast start

```bash
python scripts/check_release_strict.py
```

Read in this order:

1. `papers/Paper_A_Kerdock_Near_Optimality.pdf`
2. `papers/Paper_B_Oracle_Headroom_Open_Ledger.pdf`
3. `LEDGER_GUIDE.md`
4. `REPRODUCIBILITY.md`
5. `OPEN_PROBLEMS.md`

## How to contribute

The most valuable contributions are:

- an independent reconstruction of the full T22/kernel interval stack;
- recovery/authentication of the exact final baseline package;
- a complete rerun of reported empirical experiments;
- a correction to a theorem, cost model, or evidence label;
- a genuinely different finite-width, adaptive, nonlinear, or analytic-state estimator;
- a public baseline-parity implementation.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licensing

New repository code is released under the MIT License. Original research text and the public ledger documentation are intended for release under CC BY 4.0, subject to third-party benchmark and dataset rights. See `LICENSE` and `LICENSE-DOCS.md`.

## Citation

Use `CITATION.cff`. Until external verification is complete, cite the proof constants as computer-assisted results using inherited directed kernel intervals. The T16 primal numerics have a second implementation, but the full T22/kernel interval archive still needs external reconstruction.

## Library research archive

A comprehensive import of the proof, experiment, audit, and current-state files stored in the project Library is available under [`library_archive/`](library_archive/README.md). The import preserves failed and superseded work for provenance; use the audited claim documents for current conclusions.
