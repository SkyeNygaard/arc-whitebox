#!/usr/bin/env python3
"""Small architecture smoke test. This is not benchmark evidence."""
from __future__ import annotations
import json,resource,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compiler_core import CandidateSpec, run_candidate
from fast_matmul_generic import winograd_hybrid_p3_d5_partial_tree

def main():
    rng=np.random.default_rng(20260729);networks=8;n,d,depth=64,32,8;pilot=np.arange(16,dtype=np.int64)
    specs={
      'baseline':CandidateSpec('baseline',(),0,0,0,None),
      'two_layer':CandidateSpec('two_layer',(2,),8,1,1,None),
      'fixed_three':CandidateSpec('fixed_three',(3,),8,8,.875,.995),
      'adaptive_2_6':CandidateSpec('adaptive_2_6',(2,3,4,5,6),8,1,1,.995),
    }
    rows=[]
    for network in range(networks):
        first=np.maximum(rng.standard_normal((n,d)),0).astype(np.float32)
        weights=[(rng.standard_normal((d,d))*np.sqrt(2/d)).astype(np.float32) for _ in range(depth-1)]
        means={};diags={}
        for name,spec in specs.items():
            started=time.perf_counter()
            mean,diag=run_candidate(first,weights,spec,lambda a,b:winograd_hybrid_p3_d5_partial_tree(a,b,np),np,pilot_rows=pilot,total_depth=depth)
            diag['outer_wall_s']=time.perf_counter()-started;means[name]=mean;diags[name]=diag
        base=means['baseline']
        for name in specs:
            delta=means[name]-base
            rows.append({'network':network,'candidate':name,'approximation_mse_vs_baseline':float(np.mean(delta*delta)),'approximation_max_abs_vs_baseline':float(np.max(np.abs(delta))),**diags[name]})
    summary={}
    for name in specs:
        r=[x for x in rows if x['candidate']==name]
        summary[name]={'networks':len(r),'mean_approximation_mse_vs_baseline':float(np.mean([x['approximation_mse_vs_baseline'] for x in r])),'worst_approximation_mse_vs_baseline':float(np.max([x['approximation_mse_vs_baseline'] for x in r])),'mean_outer_wall_s':float(np.mean([x['outer_wall_s'] for x in r])),'fallback_count':int(sum(x['fallback'] for x in r)),'selected_depths':[x['selected_depth'] for x in r]}
    payload={'label':'reduced-width structural smoke only; not official-style evidence','width':d,'depth':depth,'rows':n,'networks':networks,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'summary':summary,'per_network':rows}
    (ROOT/'results'/'smoke_measurement.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
