#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
R=Path(__file__).resolve().parent
summary=json.loads((R/'SOURCE_SPECIFIC_SOCP_SUMMARY.json').read_text())
dual=json.loads((R/'SELECTED_PARTITION_DUAL_CERTIFICATE.json').read_text())
onb=json.loads((R/'ONB_ESCAPE_AUDIT.json').read_text())
prov=json.loads((R/'SOURCE_PROVENANCE.json').read_text())
econ=json.loads((R/'SOLE_PATH_DERIVED_ECONOMICS.json').read_text())
dense=json.loads((R/'DENSE_PARTITION_REPRESENTATIVE.json').read_text())
checks={}
checks['status_stop']=summary['status']=='STOP'
checks['case_count_36']=summary['case_count']==36
checks['source_exact']=summary['max_source_reconstruction_error']<2e-15
checks['selected_misses_gate']=summary['confirmation_selected_valid_S']>summary['required_S_max']
checks['oracle_partition_misses_gate']=summary['confirmation_oracle_per_case_partition_valid_S']>summary['required_S_max']
checks['dual_closes_gate']=dual['dual_closes_gate'] and dual['aggregate_dual_lower_S']>dual['required_S_max']
checks['dual_gap_tight']=dual['max_relative_gap']<1e-5
checks['dual_balance_tight']=dual['max_balance']<1e-15
checks['onb_misses_gate']=onb['best_valid']>onb['required_S_max']
checks['dense_worse_than_shallow']=dense['dense_partition']['valid_S']>dense['best_valid_partition']['valid_S']
checks['economics_misses_target']=econ['selected_valid_min_adjusted_ratio']>econ['target_adjusted_ratio']
checks['protected_closed']=not prov['protected_data_opened']
archive=R/'inputs'/'oracle_gap_experiment_campaign_20260730.zip'
h=hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None
checks['archive_hash']=h==prov['authenticated_ogap_archive']['expected_sha256']
with (R/'CASE_LEVEL_SOCP_RESULTS.csv').open() as f: checks['case_csv_36']=sum(1 for _ in csv.DictReader(f))==36
assert all(checks.values()), {k:v for k,v in checks.items() if not v}
out={'status':'PASS','checks':checks,'selected_valid_S_over_gate':econ['selected_valid_S_over_gate'],'dual_lower_S_over_gate':econ['selected_dual_lower_S_over_gate'],'onb_S_over_gate':econ['onb_best_valid_S_over_local_gate']}
(R/'VERIFICATION_RESULTS.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
