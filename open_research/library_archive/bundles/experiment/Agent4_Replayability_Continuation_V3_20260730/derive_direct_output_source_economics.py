#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
TARGET=.2304147465437788
# Latest Agent 8 confirmation values.
ROWS=[
 {'name':'direct_rank20','rank':20,'pooled':.12829791392508938,'worst':.2875787635674405},
 {'name':'direct_rank32','rank':32,'pooled':.08136239903093326,'worst':.21383922422216337},
 {'name':'direct_rank40','rank':40,'pooled':.06713577221441935,'worst':.1613},
 {'name':'adaptive_direct','rank':36.25,'pooled':.0749,'worst':.1830},
]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=[]
 for x in ROWS:
  root_allow=math.sqrt(TARGET)-math.sqrt(x['pooled'])
  out.append({**x,'replay_cost_B':0.0,'zero_noise_adjusted_ratio':x['pooled'],'normalized_additive_risk_headroom':TARGET-x['pooled'],'shared_estimator_gamma_times_trace_ceiling':root_allow**2,'root_allowance':root_allow,'aggregate_gate_pass':x['pooled']<TARGET,'tail_gate_pass':x['worst']<TARGET})
 result={'target_ratio':TARGET,'source':'Agent 8 direct-output basis PCA','rows':out,'interpretation':'No nonlinear replay or translated-cloud source construction is required; remaining gate is absolute contraction observability/economics.'}
 a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
