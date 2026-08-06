# Canonical research ledger

Every sheet of the canonical WHestBench research ledger, exported to CSV
so it can be read, diffed, and searched without Excel. This is the full
internal experiment record, including entries that were later retracted,
quarantined, or reclassified. Read the evidence-status columns before
quoting any row.

Source workbook: `whestbench_canonical_research_ledger_20260731_reconciled_v31_final_local_writeup.xlsx`

Regenerate with:

```bash
python scripts/export_ledger.py path/to/ledger.xlsx
```

| sheet | file | rows | cols |
|---|---|---:|---:|
| Experiment Ledger | [`experiment_ledger.csv`](experiment_ledger.csv) | 521 | 11 |
| Current State | [`current_state.csv`](current_state.csv) | 161 | 4 |
| Agent Overlap | [`agent_overlap.csv`](agent_overlap.csv) | 26 | 5 |
| Next Actions | [`next_actions.csv`](next_actions.csv) | 10 | 4 |
| Update Log | [`update_log.csv`](update_log.csv) | 142 | 7 |
| Canonical Portfolio | [`canonical_portfolio.csv`](canonical_portfolio.csv) | 33 | 10 |
| Evidence Registry | [`evidence_registry.csv`](evidence_registry.csv) | 206 | 20 |
| Evidence Rubric | [`evidence_rubric.csv`](evidence_rubric.csv) | 12 | 3 |
| Split Registry | [`split_registry.csv`](split_registry.csv) | 52 | 6 |
| Reconciliation Audit | [`reconciliation_audit.csv`](reconciliation_audit.csv) | 106 | 7 |
| Contradiction Map | [`contradiction_map.csv`](contradiction_map.csv) | 98 | 5 |
| Paper Claims Matrix | [`paper_claims_matrix.csv`](paper_claims_matrix.csv) | 57 | 7 |
| Paper Evidence Index | [`paper_evidence_index.csv`](paper_evidence_index.csv) | 50 | 6 |
| Agent Review Plan | [`agent_review_plan.csv`](agent_review_plan.csv) | 9 | 6 |
| Evidence Quarantine | [`evidence_quarantine.csv`](evidence_quarantine.csv) | 8 | 6 |
| Project Closeout | [`project_closeout.csv`](project_closeout.csv) | 10 | 4 |
| Challenge Frontier | [`challenge_frontier.csv`](challenge_frontier.csv) | 21 | 12 |
| Statistical Frontier | [`statistical_frontier.csv`](statistical_frontier.csv) | 23 | 10 |
| Subagent Handoffs | [`subagent_handoffs.csv`](subagent_handoffs.csv) | 10 | 4 |
| Latest Markdown Audit | [`latest_markdown_audit.csv`](latest_markdown_audit.csv) | 37 | 7 |
| Sign Count Frontier | [`sign_count_frontier.csv`](sign_count_frontier.csv) | 13 | 5 |
| Non-Overlap v25 | [`non_overlap_v25.csv`](non_overlap_v25.csv) | 18 | 8 |
| V26 Decision Memo | [`v26_decision_memo.csv`](v26_decision_memo.csv) | 43 | 6 |
| V26 Source Matrix | [`v26_source_matrix.csv`](v26_source_matrix.csv) | 17 | 12 |
| V27 Decision Memo | [`v27_decision_memo.csv`](v27_decision_memo.csv) | 47 | 6 |
| V27 Integration Map | [`v27_integration_map.csv`](v27_integration_map.csv) | 20 | 9 |
| V28 Decision Memo | [`v28_decision_memo.csv`](v28_decision_memo.csv) | 21 | 6 |
| V28 Integration Map | [`v28_integration_map.csv`](v28_integration_map.csv) | 12 | 9 |
| V29 Decision Memo | [`v29_decision_memo.csv`](v29_decision_memo.csv) | 23 | 6 |
| V29 Integration Map | [`v29_integration_map.csv`](v29_integration_map.csv) | 15 | 9 |
| V30 Decision Memo | [`v30_decision_memo.csv`](v30_decision_memo.csv) | 26 | 6 |
| V30 Integration Map | [`v30_integration_map.csv`](v30_integration_map.csv) | 17 | 9 |
| V31 Decision Memo | [`v31_decision_memo.csv`](v31_decision_memo.csv) | 26 | 6 |
| V31 Integration Map | [`v31_integration_map.csv`](v31_integration_map.csv) | 17 | 9 |

## How to read this

The ledger is a working record, not a results table. It was maintained
across many agent sessions and repeatedly reconciled; the `Reconciliation
Audit`, `Contradiction Map`, and `Evidence Quarantine` sheets exist
precisely because earlier rows disagreed with later measurement. A row
appearing here is not a claim that its number is correct.

For claims that survived review with their evidence boundary attached,
use [`../claims.csv`](../claims.csv). For the graded competition result,
use [`../phase1_320802.json`](../phase1_320802.json).
