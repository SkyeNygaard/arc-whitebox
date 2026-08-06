# Ledger guide

The workbook in `ledger/` is the canonical research ledger. It is a lab notebook, claim registry, evidence map, and decision history. It is not itself peer review.

## Why retain the workbook

The project contains many results that are close in wording but different in scope. Examples include:

- nonnegative versus arbitrary-signed static cubature;
- limiting-kernel versus exact finite-width claims;
- oracle representation versus legal estimator;
- pointwise variance versus complete-block variance;
- exact theorem versus computer-assisted constant;
- reported local run versus independent reproduction.

The ledger preserves these distinctions and records corrections without erasing history.

## Core fields

A public row should identify:

- `ID`: stable theorem, experiment, audit, or action identifier;
- `Evidence level`: proved, computer-assisted, reproduced, reported, oracle, etc.;
- `Family`: estimator or proof class;
- `Experiment`: exact action taken;
- `Result`: numerical or symbolic output;
- `Verdict`: what the result changes;
- `Primary source`: file or artifact that supports the row;
- `Status`: active, closed, deferred, superseded, or open;
- `Evidence tier`: compact classification;
- `Overlap cluster`: related branches that must not be double-counted;
- `Update batch`: provenance and date.

## Reading rules

1. Prefer the latest non-superseded row for a current claim.
2. Read the primary source before treating a row as a theorem.
3. Preserve scope words such as `static`, `limiting`, `reported`, and `tested`.
4. Do not infer that `closed` means universal impossibility.
5. Treat an oracle result as a capacity result only.
6. Treat a reported result as unreproduced until raw artifacts are attached.
7. Check whether a cost is measured, profiled, projected, or omitted.

## Evidence labels

| Label | Public meaning |
|---|---|
| PROVED | Analytic theorem under listed assumptions |
| COMPUTER-ASSISTED | Proof depends on exact or directed computation |
| REPRODUCED | Independent rerun from raw artifacts |
| ARITHMETIC CHECKED | Derived arithmetic checked; experiment not rerun |
| REPORTED | Narrative or summary exists; raw bundle missing |
| ORACLE MECHANISM | Uses target information unavailable to an estimator |
| OPERATIONALLY CLOSED | Frozen implementation/class fails a declared gate |
| DEFERRED | Logically open but prerequisites absent |
| SUPERSEDED | Retained for provenance, not current wording |

## Adding a result

New work should be submitted through the issue and pull-request templates. Do not rewrite prior rows in place unless correcting a transcription error. Add a superseding row, explain the contradiction, and link both sources.
