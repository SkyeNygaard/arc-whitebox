# Network-Specific Lower-Defect Factorization Bundle

This bundle contains the completed bounded Experiment 3 sandbox run.

Start with:

- `DECISION.md` — conclusion and key metrics.
- `REPRESENTATION.md` — exact factorization definition.
- `RESULTS_SUMMARY.json` — compact machine-readable summary.
- `ORACLE_RANK_CEILING.json` — primary rank and rotation audit.
- `MODE_PREDICTION_RESULTS.json` — legal subspace and learner ablations.
- `INDEPENDENT_CROSSREF_VALIDATION.json` — disjoint anchor/target reference audit.

Runnable entry points:

- `run_experiment.py`
- `run_crossref_audit.py`

The official high-precision challenge arrays were not included in the shared launch pack. Results are therefore a width-256 sandbox mechanism audit, not a protected holdout certification.
