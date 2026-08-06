#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGET=1/4.34
checks={}
term=json.loads((ROOT/'results/terminal_innovation_late_complete.json').read_text())
checks['terminal_complete']=term['status']=='COMPLETE' and len(term['rows'])==108
checks['terminal_le27_closed']=all(term['summaries'][str(t)]['ruled_out_pooled'] for t in [1,4,8,16,24,27])
checks['terminal_transition_29']=not term['summaries']['29']['ruled_out_pooled']
late=json.loads((ROOT/'results/adaptive_direct_late_checkpoint_socp.json').read_text())
checks['late_complete']=late['status']=='COMPLETE_FOR_LATE_PARTITIONS'
checks['late_all_cases']=all(late['summaries'][str([1,t,32])]['n_cases']==12 for t in [29,30,31])
checks['late_zero_passes']=all(late['summaries'][str([1,t,32])]['case_passes']==0 and not late['summaries'][str([1,t,32])]['pass'] for t in [29,30,31])
allrows=[]
for name in ['all_layer_socp_dual_2048_a.json','all_layer_socp_dual_2048_b.json','all_layer_socp_dual_2048_easy.json']:
 o=json.loads((ROOT/'results'/name).read_text());r=o['rows'][0];allrows.append(r)
checks['all_layer_partitions_full']=all(r['partition']==list(range(1,33)) and r['n_pairs']==2048 for r in allrows)
checks['dual_feasible']=all(r['dual']['max_ball_ratio']<=1+1e-12 and r['dual']['max_stationarity_norm']<2e-10 for r in allrows)
checks['dual_below_primal']=all(r['dual']['dual_objective']<=r['primal']['objective']*(1+1e-12) for r in allrows)
checks['dual_closes']=all(r['dual_score_lower_bound']>TARGET and r['dual_closes'] for r in allrows)
checks['hard_replication']=allrows[0]['case_id']==allrows[1]['case_id'] and abs(allrows[0]['dual_score_lower_bound']-allrows[1]['dual_score_lower_bound'])<0.2
agent4=json.loads((ROOT/'sources/AGENT4_SOURCE_SPECIFIC_SOCP_SUMMARY.json').read_text())
checks['agent4_prior_reconciled']=agent4['selected_partition']=='p02_l1_4_final' and agent4['confirmation_selected_valid_S']>agent4['required_S_max']
out={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'headline':{'terminal_t27_score_lower':term['summaries']['27']['score_lower_bound_proxy'],'late_t31_score':late['summaries'][str([1,31,32])]['score_proxy_oracle'],'all_layer_hard_dual_scores':[allrows[0]['dual_score_lower_bound'],allrows[1]['dual_score_lower_bound']],'all_layer_easy_dual_score':allrows[2]['dual_score_lower_bound'],'target':TARGET}}
(ROOT/'results/verification_v27.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
if out['status']!='PASS':raise SystemExit(1)
